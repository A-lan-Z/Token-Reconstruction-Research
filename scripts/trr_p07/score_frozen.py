#!/usr/bin/env python3
"""Truth-gated retrospective TRR-P07 scorer.

The replay runner freezes all 48 prediction descriptors before this module is
called.  ``validate_prediction_freeze`` only reads JSON metadata and prediction
artifact headers.  ``score_frozen`` then opens the explicitly bound historical
truth manifest, loads its declared arrays, and writes one create-only score
artifact.  The module has no record-selection or method-selection path.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from token_reconstruction.trr_p07_metrics import (  # noqa: E402
    CONTRASTS,
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    DOMAINS,
    PANELS,
    P07MetricsError,
    TARGETS,
    classify_gate,
    paired_cluster_bootstrap,
    score_method,
)


TASK_ID = "TRR-P07"
APPROVED_PLAN_SHA256 = "a0a2339f1a4b77e02d7d1772459dc14d442a4ce24b5111a01e58622ca1ae7c3e"
REPLAY_SCHEMA = "token-reconstruction.trr-p07-frozen-replay.v1"
TRUTH_SCHEMA = "token-reconstruction.trr-p07-retrospective-truth.v1"
SCORE_SCHEMA = "token-reconstruction.trr-p07-score.v1"
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
VOCABULARY_SIZE = 128256
RECORDS_PER_DOMAIN = 256
REPLICATE_SEEDS = (6106, 6107)
P06_METHODS = ("p06_past_only", "p06_positionwise_diagonal")
P06_SELECTION_SHA256 = "d53ed8c972ec9ec00c6490dca22a99af833ea839fa68d9c4164ce061ee893a1a"
TRR0006_SELECTION_SHA256 = "75909aaf0f9e40176c197d86c09651097010a11519855f1db3dc50fe5e754f43"
RETAINED_METHODS = {
    "enriched__affine_trained_diagonal_attention128": "trr0006_positionwise_reference",
    "enriched__affine_causal_h_attention128": "trr0006_causal_enriched",
}
REPLAY_METHODS = tuple(f"{method}__seed{seed}" for method in P06_METHODS for seed in REPLICATE_SEEDS) + tuple(RETAINED_METHODS)
SCORE_METHODS = (*P06_METHODS, *RETAINED_METHODS.values())
METHODS = REPLAY_METHODS
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class P07ScoreError(RuntimeError):
    """Raised when the P07 score boundary fails closed."""


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise P07ScoreError(f"file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }
    if root is not None:
        try:
            record["path"] = path.relative_to(root).as_posix()
        except ValueError:
            pass
    return record


def _newline_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P07ScoreError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P07ScoreError(f"{description} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise P07ScoreError(f"{description} must be a JSON object")
    return dict(value)


def _resolve_path(record: Mapping[str, Any], *, root: Path, description: str) -> Path:
    value = record.get("path")
    if not isinstance(value, str) or not value:
        raise P07ScoreError(f"{description} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise P07ScoreError(f"{description} is unavailable: {path}")
    return path


def _verify_record(record: Mapping[str, Any], *, root: Path, description: str, hash_file: bool = True) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise P07ScoreError(f"{description} file record is malformed")
    size = record.get("bytes")
    digest = record.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise P07ScoreError(f"{description} byte binding is malformed")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise P07ScoreError(f"{description} SHA-256 binding is malformed")
    path = _resolve_path(record, root=root, description=description)
    actual_size = int(path.stat().st_size)
    if actual_size != size:
        raise P07ScoreError(f"{description} byte binding changed")
    actual_digest = _sha256_file(path) if hash_file else digest
    if hash_file and actual_digest != digest:
        raise P07ScoreError(f"{description} hash binding changed")
    return {"path": str(path), "bytes": actual_size, "sha256": digest}


def _descriptor_key(panel: str, cell_id: str, method_key: str) -> str:
    return f"{panel}::{cell_id}::{method_key}"


def _prediction_descriptors(manifest: Mapping[str, Any], *, root: Path) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    raw = manifest.get("predictions")
    if not isinstance(raw, Mapping):
        raise P07ScoreError("replay prediction matrix is missing")
    expected: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for key, descriptor in raw.items():
        if not isinstance(key, str) or not isinstance(descriptor, Mapping):
            raise P07ScoreError("replay prediction descriptor is malformed")
        fields = key.split("::")
        if len(fields) != 3:
            raise P07ScoreError(f"replay prediction key is malformed: {key}")
        panel, cell_id, method_key = fields
        if panel not in PANELS or cell_id not in {f"{d}__{t}" for d in DOMAINS for t in TARGETS}:
            raise P07ScoreError(f"replay prediction cell is unregistered: {key}")
        if method_key not in METHODS:
            raise P07ScoreError(f"replay method is unregistered: {method_key}")
        identity = (panel, cell_id, method_key)
        if identity in expected:
            raise P07ScoreError(f"duplicate replay prediction descriptor: {key}")
        expected[identity] = descriptor
    expected_count = len(PANELS) * 4 * len(METHODS)
    if len(expected) != expected_count:
        raise P07ScoreError(f"replay prediction matrix is incomplete: expected {expected_count}, got {len(expected)}")
    for descriptor in expected.values():
        if descriptor.get("schema") != "token-reconstruction.trr-p07-predictions.v1" or descriptor.get("task_id") != TASK_ID:
            raise P07ScoreError("prediction descriptor schema or task ID changed")
        for flag in ("truth_opened", "source_text_loaded", "target_labels_loaded", "candidate_arrays_persisted"):
            if descriptor.get(flag) is not False:
                raise P07ScoreError(f"prediction descriptor has forbidden flag: {flag}")
        if descriptor.get("records") != RECORDS_PER_DOMAIN or descriptor.get("shape") != [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS] or descriptor.get("scored_post_bos_tokens") != SCORED_POST_BOS:
            raise P07ScoreError("prediction descriptor geometry changed")
        if not isinstance(descriptor.get("record_ids_sha256"), str) or _SHA256.fullmatch(descriptor["record_ids_sha256"]) is None:
            raise P07ScoreError("prediction descriptor record-order binding is missing")
        _verify_record(descriptor.get("prediction"), root=root, description="prediction artifact", hash_file=True)
    return expected


def _validate_observation_manifests(
    replay: Mapping[str, Any],
    *,
    root: Path,
    descriptors: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> None:
    """Bind every prediction cell to its recorded source-free observation row."""

    for panel in PANELS:
        panel_descriptor = replay["panels"][panel]
        manifest_record = panel_descriptor.get("observation_manifest")
        actual = _verify_record(manifest_record, root=root, description=f"{panel} observation manifest", hash_file=True)
        manifest = _load_json(Path(actual["path"]), description=f"{panel} observation manifest")
        if manifest.get("truth_opened") is not False or manifest.get("target_labels_loaded") is not False:
            raise P07ScoreError(f"{panel} observation manifest has forbidden truth flags")
        raw_cells = manifest.get("cells")
        if not isinstance(raw_cells, list):
            raise P07ScoreError(f"{panel} observation manifest cell matrix is missing")
        rows = {str(row.get("cell_id")): row for row in raw_cells if isinstance(row, Mapping)}
        if set(rows) != {f"{domain}__{target}" for domain in DOMAINS for target in TARGETS}:
            raise P07ScoreError(f"{panel} observation manifest cell set changed")
        for cell_id, row in rows.items():
            observation = row.get("observation")
            record_ids_sha = row.get("record_ids_sha256")
            if not isinstance(observation, Mapping) or not isinstance(record_ids_sha, str):
                raise P07ScoreError(f"{panel}/{cell_id} observation binding is missing")
            for method in METHODS:
                descriptor = descriptors[(panel, cell_id, method)]
                descriptor_observation = descriptor.get("observation")
                if not isinstance(descriptor_observation, Mapping):
                    raise P07ScoreError(f"{panel}/{cell_id} replay observation binding is missing")
                if any(descriptor_observation.get(field) != observation.get(field) for field in ("path", "bytes", "sha256")) or descriptor.get("record_ids_sha256") != record_ids_sha:
                    raise P07ScoreError(f"{panel}/{cell_id} observation binding differs across replay and source manifest")


def validate_prediction_freeze(
    *,
    repository_root: Path,
    replay_manifest_path: Path,
    expected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the source-free 48-cell replay before any truth read."""

    root = Path(repository_root).expanduser().resolve()
    replay_path = Path(replay_manifest_path).expanduser().resolve()
    replay = _load_json(replay_path, description="P07 replay manifest")
    if replay.get("schema") != REPLAY_SCHEMA or replay.get("task_id") != TASK_ID or replay.get("status") != "FROZEN_P07_PREDICTIONS_NO_TRUTH":
        raise P07ScoreError("replay manifest is not the frozen no-truth P07 matrix")
    for flag in ("truth_opened", "source_text_loaded", "target_labels_loaded", "candidate_arrays_persisted"):
        if replay.get(flag) is not False:
            raise P07ScoreError(f"replay manifest has forbidden flag: {flag}")
    commit = replay.get("code_commit")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise P07ScoreError("replay code commit is not a full hash")
    if replay.get("prediction_count") != len(PANELS) * 4 * len(METHODS):
        raise P07ScoreError("replay prediction count changed")
    if expected_plan_sha256 is not None and expected_plan_sha256 != APPROVED_PLAN_SHA256:
        raise P07ScoreError("only the approved P07 plan hash is accepted")
    expected_plan_sha256 = APPROVED_PLAN_SHA256
    plan = replay.get("source_bindings", {}).get("canonical_plan")
    if not isinstance(plan, Mapping) or plan.get("sha256") != expected_plan_sha256:
        raise P07ScoreError("replay is bound to a different canonical plan")
    geometry = replay.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("sequence_tokens") != SEQUENCE_TOKENS or geometry.get("scored_post_bos_tokens") != SCORED_POST_BOS:
        raise P07ScoreError("replay geometry changed")
    panels = replay.get("panels")
    if not isinstance(panels, Mapping) or set(panels) != set(PANELS):
        raise P07ScoreError("replay panel set changed")
    for panel in PANELS:
        descriptor = panels[panel]
        if not isinstance(descriptor, Mapping) or descriptor.get("records_per_domain") != RECORDS_PER_DOMAIN:
            raise P07ScoreError(f"replay panel geometry changed: {panel}")
        if panel == "trr0006_subset":
            subset = descriptor.get("subset")
            subset_rule = descriptor.get("subset_rule") if isinstance(descriptor.get("subset_rule"), str) else (subset.get("rule") if isinstance(subset, Mapping) else None)
            if "6*k" not in str(subset_rule):
                raise P07ScoreError("TRR-0006 subset rule is not the approved every-sixth rule")
            if not isinstance(subset, Mapping) or subset.get("row_indices") != list(range(0, 1536, 6)):
                raise P07ScoreError("TRR-0006 subset row order is not the approved every-sixth sequence")
    fixtures = replay.get("fixtures")
    if not isinstance(fixtures, Mapping) or fixtures.get("status") != "PASS" or fixtures.get("truth_opened") is not False:
        raise P07ScoreError("replay fixture/provenance gate did not pass before scoring")
    method_rows = replay.get("methods")
    if not isinstance(method_rows, list) or len(method_rows) != len(METHODS):
        raise P07ScoreError("replay method binding matrix is incomplete")
    state_by_key: dict[str, Mapping[str, Any]] = {}
    for row in method_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("key"), str) or row["key"] not in METHODS:
            raise P07ScoreError("replay state binding has an unregistered method")
        if row["key"] in state_by_key or not isinstance(row.get("state"), Mapping):
            raise P07ScoreError("replay state binding matrix is duplicated or incomplete")
        state_by_key[row["key"]] = row
    descriptors = _prediction_descriptors(replay, root=root)
    _validate_observation_manifests(replay, root=root, descriptors=descriptors)
    checked_states: set[tuple[str, str]] = set()
    checked_observations: set[tuple[str, str]] = set()
    for (panel, cell_id, method_key), descriptor in descriptors.items():
        row = state_by_key[method_key]
        state_record = row.get("state")
        state_sha = state_record.get("sha256") if isinstance(state_record, Mapping) else None
        descriptor_state = descriptor.get("state")
        if not isinstance(state_record, Mapping) or not isinstance(descriptor_state, Mapping) or dict(state_record) != dict(descriptor_state):
            raise P07ScoreError(f"state binding differs for {method_key}")
        if descriptor.get("method_key") != method_key or descriptor.get("method_id") != row.get("method_id") or descriptor.get("seed") != row.get("seed") or descriptor.get("state_tensor_sha256") != row.get("state_tensor_sha256"):
            raise P07ScoreError(f"method identity differs for {method_key}")
        state_path = str(state_record.get("path"))
        state_key = (state_path, str(state_sha))
        if state_key not in checked_states:
            _verify_record(state_record, root=root, description=f"state {method_key}", hash_file=True)
            checked_states.add(state_key)
        observation = descriptor.get("observation")
        if not isinstance(observation, Mapping):
            raise P07ScoreError(f"observation binding is missing: {panel}/{cell_id}")
        observation_key = (str(observation.get("path")), str(observation.get("sha256")))
        if observation_key not in checked_observations:
            _verify_record(observation, root=root, description=f"observation {panel}/{cell_id}", hash_file=True)
            checked_observations.add(observation_key)
    return {
        "schema": "token-reconstruction.trr-p07-joint-validation.v1",
        "task_id": TASK_ID,
        "status": "JOINT_FREEZE_VALIDATED_NO_TRUTH",
        "repository_root": str(root),
        "replay_manifest": _file_record(replay_path, root=root),
        "replay": replay,
        "descriptors": descriptors,
        "truth_opened": False,
    }


def _load_array(path: Path, *, key: str | None, description: str) -> np.ndarray:
    try:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                selected = key or (list(archive.keys())[0] if len(archive.keys()) == 1 else None)
                if selected is None or selected not in archive:
                    raise P07ScoreError(f"{description} NPZ key is ambiguous")
                value = archive[selected]
        elif suffix == ".safetensors":
            from safetensors.numpy import load_file

            arrays = load_file(str(path))
            selected = key or (next(iter(arrays)) if len(arrays) == 1 else None)
            if selected is None or selected not in arrays:
                raise P07ScoreError(f"{description} safetensors key is ambiguous")
            value = arrays[selected]
        else:
            raise P07ScoreError(f"{description} has unsupported array format")
    except P07ScoreError:
        raise
    except Exception as exc:
        raise P07ScoreError(f"{description} could not be loaded") from exc
    return np.ascontiguousarray(value)


def _load_prediction(descriptor: Mapping[str, Any], *, root: Path, description: str) -> np.ndarray:
    record = _verify_record(descriptor.get("prediction"), root=root, description=f"{description} prediction")
    array = _load_array(Path(record["path"]), key="predictions", description=description)
    if array.shape != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or not np.issubdtype(array.dtype, np.integer):
        raise P07ScoreError(f"{description} prediction geometry or dtype changed")
    if np.any(array < 0) or np.any(array >= VOCABULARY_SIZE):
        raise P07ScoreError(f"{description} prediction IDs are outside the frozen vocabulary")
    return array.astype(np.int64, copy=False)


def _observation_geometry(descriptor: Mapping[str, Any], *, root: Path, description: str) -> tuple[np.ndarray, np.ndarray]:
    observation = descriptor.get("observation")
    path = _resolve_path(observation, root=root, description=f"{description} observation")
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="np", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise P07ScoreError(f"{description} observation keys changed")
            mask_slice = handle.get_slice("attention_mask")
            positions_slice = handle.get_slice("position_ids")
            if tuple(mask_slice.get_shape())[-1] != SEQUENCE_TOKENS or tuple(positions_slice.get_shape())[-1] != SEQUENCE_TOKENS:
                raise P07ScoreError(f"{description} observation tensor geometry changed")
            selected_rows = descriptor.get("timing", {}).get("selected_row_indices")
            if not isinstance(selected_rows, list):
                raise P07ScoreError(f"{description} selected observation rows are missing")
            mask = np.concatenate([np.asarray(mask_slice[row : row + 1]) for row in selected_rows], axis=0)
            positions = np.concatenate([np.asarray(positions_slice[row : row + 1]) for row in selected_rows], axis=0)
    except P07ScoreError:
        raise
    except Exception as exc:
        raise P07ScoreError(f"{description} observation geometry could not be loaded") from exc
    selected = descriptor.get("timing", {}).get("selected_row_indices")
    if not isinstance(selected, list) or len(selected) != RECORDS_PER_DOMAIN or any(isinstance(i, bool) or not isinstance(i, int) for i in selected):
        raise P07ScoreError(f"{description} selected observation rows are missing")
    if mask.shape != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or positions.shape != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
        raise P07ScoreError(f"{description} observation geometry changed")
    mask = mask.astype(bool, copy=False)
    if not mask[:, 0].all() or (mask[:, 1:] > mask[:, :-1]).any():
        raise P07ScoreError(f"{description} observation mask is not BOS/right-padded")
    expected = np.broadcast_to(np.arange(SEQUENCE_TOKENS, dtype=np.int64), positions.shape)
    if not np.array_equal(positions.astype(np.int64), expected):
        raise P07ScoreError(f"{description} position geometry changed")
    return mask, positions.astype(np.int64, copy=False)


def _truth_cell(truth_manifest: Mapping[str, Any], panel: str, domain: str) -> Mapping[str, Any]:
    for field in ("cells", "panel_domains", "domains"):
        values = truth_manifest.get(field)
        if not isinstance(values, Mapping):
            continue
        for key in (f"{panel}/{domain}", f"{panel}::{domain}", f"{panel}__{domain}"):
            value = values.get(key)
            if isinstance(value, Mapping):
                return value
        panel_value = values.get(panel)
        if isinstance(panel_value, Mapping) and isinstance(panel_value.get(domain), Mapping):
            return panel_value[domain]
    raise P07ScoreError(f"truth manifest lacks {panel}/{domain} descriptor")


def _selection_rows(
    path: Path,
    *,
    root: Path,
    expected_sha256: str,
    expected_records: int,
    indices: Sequence[int] | None = None,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Read source-selection metadata only and return ordered IDs/fingerprints."""

    actual = _file_record(path, root=root)
    if actual["sha256"] != expected_sha256:
        raise P07ScoreError(f"source selection binding changed: {path}")
    selection = _load_json(path, description="source-selection metadata")
    records = selection.get("selection_rule", {}).get("records")
    if not isinstance(records, Mapping) or set(records) != set(DOMAINS):
        raise P07ScoreError("source-selection record ledger is incomplete")
    selected_indices = tuple(range(expected_records)) if indices is None else tuple(int(index) for index in indices)
    selected_count = len(selected_indices)
    result: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for domain in DOMAINS:
        rows = records[domain]
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise P07ScoreError(f"source-selection rows are malformed: {domain}")
        selected = [rows[index] for index in selected_indices]
        if any(not isinstance(row.get("record_id"), str) or not row["record_id"] for row in selected):
            raise P07ScoreError(f"source-selection IDs are malformed: {domain}")
        ids = tuple(str(row["record_id"]) for row in selected)
        sequences = tuple(str(row.get("final_sequence_sha256", "")) for row in selected)
        if len(ids) != selected_count or len(set(ids)) != selected_count or any(_SHA256.fullmatch(value) is None for value in sequences):
            raise P07ScoreError(f"source-selection identity geometry changed: {domain}")
        result[domain] = (ids, sequences)
    return result


def _validate_truth_array(array: np.ndarray, *, description: str) -> np.ndarray:
    if array.shape != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or not np.issubdtype(array.dtype, np.integer):
        raise P07ScoreError(f"{description} geometry or dtype changed")
    if np.any(array < 0) or np.any(array >= VOCABULARY_SIZE) or np.any(array[:, 0] != 128000):
        raise P07ScoreError(f"{description} token IDs are outside the frozen geometry")
    return array.astype(np.int64, copy=False)


def _frozen_record_digest(validated: Mapping[str, Any], *, panel: str, domain: str) -> str:
    descriptors = validated.get("descriptors", {})
    digests = {
        descriptors[(panel, f"{domain}__{target}", method)].get("record_ids_sha256")
        for target in TARGETS
        for method in METHODS
    }
    if len(digests) != 1 or not isinstance(next(iter(digests)), str):
        raise P07ScoreError(f"frozen source-order bindings differ: {panel}/{domain}")
    return str(next(iter(digests)))


def _load_historical_truth_sidecars(
    *,
    validated: Mapping[str, Any],
    repository_root: Path,
    p06_truth_manifest_path: Path,
    trr0006_truth_binding_path: Path,
    p06_selection_path: Path,
    trr0006_selection_path: Path,
) -> dict[tuple[str, str], tuple[np.ndarray, tuple[str, ...]]]:
    """Adapt the two already-published truth sidecars after freeze.

    This reads no source text and creates no evaluator labels.  The P06 truth
    arrays are already 256-row arrays; the TRR-0006 sidecar is sliced at the
    frozen metadata-only rows 6*k, k=0..255.
    """

    root = Path(repository_root).expanduser().resolve()
    p06_manifest = _load_json(Path(p06_truth_manifest_path), description="P06 truth sidecar")
    if p06_manifest.get("schema") != "token-reconstruction.trr-p06-truth-manifest.v1" or p06_manifest.get("status") != "TRUTH_READY_AFTER_JOINT_FREEZE":
        raise P07ScoreError("P06 truth sidecar is not the published opened receipt")
    if p06_manifest.get("source_selection_sha256") != P06_SELECTION_SHA256:
        raise P07ScoreError("P06 truth sidecar source selection changed")
    trr_manifest = _load_json(Path(trr0006_truth_binding_path), description="TRR-0006 truth sidecar")
    if trr_manifest.get("schema") != "token-reconstruction.trr0006-private-label-binding.v1" or trr_manifest.get("source_selection_sha256") != TRR0006_SELECTION_SHA256:
        raise P07ScoreError("TRR-0006 truth sidecar identity changed")
    p06_ids = _selection_rows(Path(p06_selection_path), root=root, expected_sha256=P06_SELECTION_SHA256, expected_records=RECORDS_PER_DOMAIN)
    old_ids = _selection_rows(Path(trr0006_selection_path), root=root, expected_sha256=TRR0006_SELECTION_SHA256, expected_records=1536, indices=range(0, 1536, 6))
    full_old_ids = _selection_rows(Path(trr0006_selection_path), root=root, expected_sha256=TRR0006_SELECTION_SHA256, expected_records=1536)
    for domain, (ids, _) in p06_ids.items():
        if p06_manifest.get("record_ids_sha256", {}).get(domain) != _newline_digest(ids):
            raise P07ScoreError(f"P06 truth sidecar record order changed: {domain}")
        if _frozen_record_digest(validated, panel="p06_panel", domain=domain) != _newline_digest(ids):
            raise P07ScoreError(f"P06 frozen source order differs from truth sidecar: {domain}")
    for domain, (ids, _) in full_old_ids.items():
        if trr_manifest.get("record_ids_sha256", {}).get(domain) != _newline_digest(ids):
            raise P07ScoreError(f"TRR-0006 truth sidecar full record order changed: {domain}")
        if _frozen_record_digest(validated, panel="trr0006_subset", domain=domain) != _newline_digest(old_ids[domain][0]):
            raise P07ScoreError(f"TRR-0006 frozen subset source order differs from truth sidecar: {domain}")

    loaded: dict[tuple[str, str], tuple[np.ndarray, tuple[str, ...]]] = {}
    p06_domains = p06_manifest.get("domains")
    if not isinstance(p06_domains, Mapping):
        raise P07ScoreError("P06 truth sidecar domains are missing")
    for domain in DOMAINS:
        p06_record = _verify_record(p06_domains.get(domain), root=root, description=f"P06 truth {domain}")
        p06_array = _validate_truth_array(_load_array(Path(p06_record["path"]), key=None, description=f"P06 truth {domain}"), description=f"P06 truth {domain}")
        loaded[("p06_panel", domain)] = (p06_array, p06_ids[domain][0])
    old_record = trr_manifest.get("truth_file")
    old_record = _verify_record(old_record, root=root, description="TRR-0006 truth file")
    for domain in DOMAINS:
        full_array = _load_array(Path(old_record["path"]), key=f"{domain}__token_ids", description=f"TRR-0006 truth {domain}")
        if full_array.shape != (1536, SEQUENCE_TOKENS) or not np.issubdtype(full_array.dtype, np.integer):
            raise P07ScoreError(f"TRR-0006 truth {domain} full geometry changed")
        selected = _validate_truth_array(np.asarray(full_array)[list(range(0, 1536, 6))], description=f"TRR-0006 truth subset {domain}")
        loaded[("trr0006_subset", domain)] = (selected, old_ids[domain][0])
    return loaded


def _load_truth_manifest(
    truth_path: Path,
    *,
    root: Path,
    validated: Mapping[str, Any],
) -> dict[tuple[str, str], tuple[np.ndarray, tuple[str, ...]]]:
    manifest = _load_json(truth_path, description="P07 retrospective truth manifest")
    if manifest.get("schema") != TRUTH_SCHEMA or manifest.get("task_id") != TASK_ID or manifest.get("status") != "TRUTH_READY_AFTER_PREDICTION_FREEZE":
        raise P07ScoreError("truth manifest is not the declared post-freeze retrospective status")
    if manifest.get("truth_opened") is not True or manifest.get("prediction_freeze_sha256") != validated["replay_manifest"]["sha256"]:
        raise P07ScoreError("truth manifest is not bound to this frozen replay")
    loaded: dict[tuple[str, str], tuple[np.ndarray, tuple[str, ...]]] = {}
    for panel in PANELS:
        for domain in DOMAINS:
            descriptor = _truth_cell(manifest, panel, domain)
            ids = descriptor.get("record_ids")
            if not isinstance(ids, list) or len(ids) != RECORDS_PER_DOMAIN or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != RECORDS_PER_DOMAIN:
                raise P07ScoreError(f"truth {panel}/{domain} record IDs are missing or duplicated")
            record = _verify_record(descriptor.get("truth"), root=root, description=f"truth {panel}/{domain}")
            array = _load_array(Path(record["path"]), key=descriptor.get("key"), description=f"truth {panel}/{domain}")
            if array.shape != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or not np.issubdtype(array.dtype, np.integer):
                raise P07ScoreError(f"truth {panel}/{domain} geometry or dtype changed")
            if np.any(array < 0) or np.any(array >= VOCABULARY_SIZE) or np.any(array[:, 0] != 128000):
                raise P07ScoreError(f"truth {panel}/{domain} token IDs are outside the frozen geometry")
            loaded[(panel, domain)] = (array.astype(np.int64, copy=False), tuple(ids))
    return loaded


def score_arrays(
    validated: Mapping[str, Any],
    *,
    repository_root: Path,
    truth_by_cell: Mapping[tuple[str, str], tuple[np.ndarray, Sequence[str]]],
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Score arrays after validation; caller must already have opened truth."""

    if validated.get("status") != "JOINT_FREEZE_VALIDATED_NO_TRUTH" or validated.get("truth_opened") is not False:
        raise P07ScoreError("score_arrays requires the validated no-truth replay receipt")
    root = Path(repository_root).expanduser().resolve()
    descriptors = validated["descriptors"]
    cells: dict[str, dict[str, Any]] = {}
    geometry_cache: dict[tuple[str, tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}
    for panel in PANELS:
        for domain in DOMAINS:
            labels, ids = truth_by_cell[(panel, domain)]
            labels = np.asarray(labels)
            record_ids = tuple(ids)
            if labels.shape != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or len(record_ids) != RECORDS_PER_DOMAIN:
                raise P07ScoreError(f"truth geometry changed: {panel}/{domain}")
            if _newline_digest(record_ids) != _frozen_record_digest(validated, panel=panel, domain=domain):
                raise P07ScoreError(f"truth row order differs from frozen predictions: {panel}/{domain}")
            for target in TARGETS:
                cell_id = f"{domain}__{target}"
                score_maps: dict[str, dict[str, Any]] = {
                    method: {} for method in (*P06_METHODS, *RETAINED_METHODS.values())
                }
                mask: np.ndarray | None = None
                positions: np.ndarray | None = None
                for method_key in METHODS:
                    descriptor = descriptors[(panel, cell_id, method_key)]
                    selected_rows = tuple(descriptor.get("timing", {}).get("selected_row_indices", ()))
                    observation_path = _resolve_path(descriptor.get("observation"), root=root, description="observation")
                    geometry_key = (str(observation_path), selected_rows)
                    if geometry_key not in geometry_cache:
                        geometry_cache[geometry_key] = _observation_geometry(
                            descriptor,
                            root=root,
                            description=f"{panel}/{cell_id}/{method_key}",
                        )
                    current_mask, current_positions = geometry_cache[geometry_key]
                    if mask is None:
                        mask, positions = current_mask, current_positions
                    elif not np.array_equal(mask, current_mask) or not np.array_equal(positions, current_positions):
                        raise P07ScoreError(f"observation geometry differs across methods: {panel}/{cell_id}")
                    predictions = _load_prediction(descriptor, root=root, description=f"{panel}/{cell_id}/{method_key}")
                    scored = score_method(predictions, labels, record_ids=record_ids, attention_mask=mask, position_ids=positions, method_id=method_key)
                    if "__seed" in method_key:
                        base_method, _, seed_text = method_key.rpartition("__seed")
                        if base_method not in P06_METHODS or seed_text not in {str(seed) for seed in REPLICATE_SEEDS}:
                            raise P07ScoreError(f"P06 replicate method key is malformed: {method_key}")
                        score_maps[base_method][seed_text] = scored
                    else:
                        score_maps[RETAINED_METHODS[method_key]]["retained"] = scored
                cells[f"{panel}/{domain}/{target}"] = {"panel": panel, "domain": domain, "target": target, "scores": score_maps}
    bootstrap = paired_cluster_bootstrap(cells, draws=bootstrap_draws, seed=bootstrap_seed)
    gate = classify_gate(bootstrap)
    return {
        "schema": SCORE_SCHEMA,
        "task_id": TASK_ID,
        "status": "TRR-P07_SCORED_AFTER_PREDICTION_FREEZE",
        "prediction_freeze": validated["replay_manifest"],
        "bootstrap": bootstrap,
        "gate": gate,
        "truth_opened": True,
        "truth_payload_persisted": False,
        "claim_scope": "exploratory retrospective P07 comparison under the frozen P06/TRR-0006 panels and method states",
        "methods": {key: value for key, value in RETAINED_METHODS.items()},
        "contrasts": {key: list(value) for key, value in CONTRASTS.items()},
    }


def score_historical_sidecars(
    *,
    repository_root: Path,
    replay_manifest_path: Path,
    p06_truth_manifest_path: Path,
    trr0006_truth_binding_path: Path,
    p06_selection_path: Path,
    trr0006_selection_path: Path,
    output_path: Path,
    expected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Score the two already-published truth sidecars after freeze validation."""

    root = Path(repository_root).expanduser().resolve()
    validated = validate_prediction_freeze(
        repository_root=root,
        replay_manifest_path=replay_manifest_path,
        expected_plan_sha256=expected_plan_sha256,
    )
    truth = _load_historical_truth_sidecars(
        validated=validated,
        repository_root=root,
        p06_truth_manifest_path=p06_truth_manifest_path,
        trr0006_truth_binding_path=trr0006_truth_binding_path,
        p06_selection_path=p06_selection_path,
        trr0006_selection_path=trr0006_selection_path,
    )
    result = score_arrays(validated, repository_root=root, truth_by_cell=truth)
    result["truth_inputs"] = {
        "p06_truth_manifest": _file_record(Path(p06_truth_manifest_path), root=root),
        "trr0006_truth_binding": _file_record(Path(trr0006_truth_binding_path), root=root),
        "p06_selection": _file_record(Path(p06_selection_path), root=root),
        "trr0006_selection": _file_record(Path(trr0006_selection_path), root=root),
        "payloads_persisted_in_result": False,
    }
    _write_create_only(Path(output_path), result)
    return result


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise P07ScoreError(f"score output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score_frozen(
    *,
    repository_root: Path,
    replay_manifest_path: Path,
    truth_manifest_path: Path,
    output_path: Path,
    expected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the freeze, then open the declared retrospective truth once."""

    root = Path(repository_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    validated = validate_prediction_freeze(
        repository_root=root,
        replay_manifest_path=replay_manifest_path,
        expected_plan_sha256=expected_plan_sha256,
    )
    truth_path = Path(truth_manifest_path).expanduser().resolve()
    truth = _load_truth_manifest(truth_path, root=root, validated=validated)
    result = score_arrays(validated, repository_root=root, truth_by_cell=truth)
    result["truth_manifest"] = _file_record(truth_path, root=root)
    _write_create_only(output, result)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    truth_group = parser.add_mutually_exclusive_group(required=True)
    truth_group.add_argument("--truth-manifest", type=Path)
    truth_group.add_argument("--p06-truth-manifest", type=Path)
    parser.add_argument("--trr0006-truth-binding", type=Path)
    parser.add_argument("--p06-selection", type=Path)
    parser.add_argument("--trr0006-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.truth_manifest is not None:
            score_frozen(
                repository_root=args.repository_root,
                replay_manifest_path=args.replay_manifest,
                truth_manifest_path=args.truth_manifest,
                output_path=args.output,
            )
        elif all(value is not None for value in (args.trr0006_truth_binding, args.p06_selection, args.trr0006_selection)):
            score_historical_sidecars(
                repository_root=args.repository_root,
                replay_manifest_path=args.replay_manifest,
                p06_truth_manifest_path=args.p06_truth_manifest,
                trr0006_truth_binding_path=args.trr0006_truth_binding,
                p06_selection_path=args.p06_selection,
                trr0006_selection_path=args.trr0006_selection,
                output_path=args.output,
            )
        else:
            parser.error("sidecar mode requires --p06-truth-manifest, --trr0006-truth-binding, and both selections")
    except (P07ScoreError, P07MetricsError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

