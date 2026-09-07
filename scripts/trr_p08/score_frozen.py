"""TRR-P08 in-memory scorer plus fail-closed post-freeze artifact runner.

``score_arrays`` remains a pure CPU function over an explicit in-memory matrix.
``run`` first revalidates the create-only no-truth joint-freeze receipt, then
opens the truth manifest and prediction arrays only after that gate succeeds.
Its JSON result stores metrics and provenance hashes, never evaluator arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.trr_p08 import freeze_matrix  # noqa: E402
from token_reconstruction.trr_p08_metrics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    DOMAINS,
    METHOD_ORDER,
    P08MetricsError,
    REPLICATE_SEEDS,
    TARGETS,
    classify_general_staging_gate,
    classify_interaction_gate,
    paired_cluster_bootstrap,
    paired_metrics_from_scores,
    score_method,
)

TASK_ID = "TRR-P08"
SCORE_SCHEMA = "token-reconstruction.trr-p08-score.v1"
TRUTH_SCHEMA = "token-reconstruction.trr-p08-truth-manifest.v1"
TRUTH_STATUS = "TRUTH_READY_AFTER_JOINT_FREEZE"
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
BOS_TOKEN_ID = 128000
VOCABULARY_SIZE = 128256
RECORDS_PER_DOMAIN = 256
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class P08ScoreError(RuntimeError):
    """Raised when the already-frozen in-memory score matrix is malformed."""


def _cell_key(domain: str, target: str) -> str:
    return f"{domain}/{target}"


def _seed_scores(method_scores: Mapping[str, Mapping[str | int, Mapping[str, Any]]], seed: int) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for method in METHOD_ORDER:
        values = method_scores.get(method)
        if not isinstance(values, Mapping) or str(seed) not in {str(key) for key in values}:
            raise P08ScoreError(f"method matrix lacks {method}/{seed}")
        value = next(score for key, score in values.items() if str(key) == str(seed))
        result[method] = value
    return result


def _seed_interaction_points(method_scores: Mapping[str, Mapping[str | int, Mapping[str, Any]]]) -> list[dict[str, float | None]]:
    values: list[dict[str, float | None]] = []
    for seed in REPLICATE_SEEDS:
        scores = _seed_scores(method_scores, seed)
        staged = paired_metrics_from_scores(
            scores["p08_past_only_staged"],
            scores["p08_positionwise_staged"],
            contrast_id="past_staged_minus_positionwise_staged",
        )["metrics"]
        joint = paired_metrics_from_scores(
            scores["p08_past_only_joint"],
            scores["p08_positionwise_joint"],
            contrast_id="past_joint_minus_positionwise_joint",
        )["metrics"]
        token_staged = staged.get("token_delta_pp")
        token_joint = joint.get("token_delta_pp")
        exact_staged = staged.get("exact_delta_pp")
        exact_joint = joint.get("exact_delta_pp")
        values.append(
            {
                "token_delta_pp": None if token_staged is None or token_joint is None else float(token_staged - token_joint),
                "exact_delta_pp": None if exact_staged is None or exact_joint is None else float(exact_staged - exact_joint),
            }
        )
    return values


def _seed_general_points(method_scores: Mapping[str, Mapping[str | int, Mapping[str, Any]]]) -> dict[str, list[dict[str, float | None]]]:
    result: dict[str, list[dict[str, float | None]]] = {}
    for visibility, staged_method, joint_method in (
        ("past_staged_minus_past_joint", "p08_past_only_staged", "p08_past_only_joint"),
        ("positionwise_staged_minus_positionwise_joint", "p08_positionwise_staged", "p08_positionwise_joint"),
    ):
        points: list[dict[str, float | None]] = []
        for seed in REPLICATE_SEEDS:
            scores = _seed_scores(method_scores, seed)
            contrast = paired_metrics_from_scores(scores[staged_method], scores[joint_method], contrast_id=visibility)["metrics"]
            points.append({
                "token_delta_pp": contrast.get("token_delta_pp"),
                "exact_delta_pp": contrast.get("exact_delta_pp"),
            })
        result[visibility] = points
    return result


def _validate_matrix_cell(cell_id: str, cell: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...], np.ndarray, np.ndarray | None, np.ndarray | None, Mapping[str, Any]]:
    if not isinstance(cell, Mapping):
        raise P08ScoreError(f"{cell_id} is not an object")
    domain, target = cell.get("domain"), cell.get("target")
    if domain not in DOMAINS or target not in TARGETS:
        raise P08ScoreError(f"{cell_id} has an unregistered domain/target")
    ids = cell.get("record_ids")
    if not isinstance(ids, (list, tuple)) or not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise P08ScoreError(f"{cell_id} record IDs are malformed")
    truth = np.asarray(cell.get("truth"))
    if truth.ndim != 2 or truth.shape[0] != len(ids) or truth.shape[1] != 128:
        raise P08ScoreError(f"{cell_id} truth geometry changed")
    predictions = cell.get("predictions")
    if not isinstance(predictions, Mapping):
        raise P08ScoreError(f"{cell_id} prediction matrix is missing")
    if set(predictions) != set(METHOD_ORDER):
        raise P08ScoreError(f"{cell_id} method matrix is incomplete")
    return str(domain), str(target), tuple(ids), truth, cell.get("attention_mask"), cell.get("position_ids"), predictions


def score_arrays(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Score all eight arm/seed fits in an already-authorized matrix.

    ``cells`` must contain four domain/target entries.  Each entry supplies
    ``truth``, geometry, ordered ``record_ids``, and
    ``predictions[method][seed]``.  The function never reads paths and never
    chooses records or checkpoints.
    """

    if not isinstance(cells, Mapping):
        raise P08ScoreError("P08 cells are not a mapping")
    expected = {_cell_key(domain, target) for domain in DOMAINS for target in TARGETS}
    if set(cells) != expected:
        raise P08ScoreError(f"P08 requires exactly cells {sorted(expected)}")

    score_cells: dict[str, dict[str, Any]] = {}
    by_domain_ids: dict[str, tuple[str, ...]] = {}
    for cell_id, raw_cell in cells.items():
        domain, target, ids, truth, mask, positions, predictions = _validate_matrix_cell(cell_id, raw_cell)
        if domain in by_domain_ids and by_domain_ids[domain] != ids:
            raise P08ScoreError(f"source-record order differs across {domain} targets")
        by_domain_ids[domain] = ids
        score_maps: dict[str, dict[str, Any]] = {method: {} for method in METHOD_ORDER}
        reference_valid: np.ndarray | None = None
        reference_exact: np.ndarray | None = None
        for method in METHOD_ORDER:
            replicates = predictions[method]
            if not isinstance(replicates, Mapping) or {str(key) for key in replicates} != {str(seed) for seed in REPLICATE_SEEDS}:
                raise P08ScoreError(f"{cell_id}/{method} must contain seeds {REPLICATE_SEEDS}")
            for seed in REPLICATE_SEEDS:
                value = next(prediction for key, prediction in replicates.items() if str(key) == str(seed))
                try:
                    scored = score_method(
                        value,
                        truth,
                        record_ids=ids,
                        attention_mask=mask,
                        position_ids=positions,
                        method_id=method,
                    )
                except P08MetricsError as exc:
                    raise P08ScoreError(f"{cell_id}/{method}/{seed}: {exc}") from exc
                valid = np.asarray([row["valid_post_bos"] for row in scored["per_record"]], dtype=bool)
                exact = np.asarray([row["exact_eligible"] for row in scored["per_record"]], dtype=bool)
                if reference_valid is None:
                    reference_valid, reference_exact = valid, exact
                elif not np.array_equal(reference_valid, valid) or not np.array_equal(reference_exact, exact):
                    raise P08ScoreError(f"{cell_id} methods have different metric geometry")
                score_maps[method][str(seed)] = scored
        score_cells[cell_id] = {
            "domain": domain,
            "target": target,
            "records": len(ids),
            "record_ids": list(ids),
            "scores": score_maps,
            "seed_interaction": _seed_interaction_points(score_maps),
            "seed_general_staging": _seed_general_points(score_maps),
        }

    bootstrap_input = {
        cell_id: {
            "domain": cell["domain"],
            "target": cell["target"],
            "method_scores": cell["scores"],
        }
        for cell_id, cell in score_cells.items()
    }
    try:
        bootstrap = paired_cluster_bootstrap(bootstrap_input, draws=bootstrap_draws, seed=bootstrap_seed)
    except P08MetricsError as exc:
        raise P08ScoreError(str(exc)) from exc
    primary_summaries = {
        domain: bootstrap["cells"][_cell_key(domain, "public_base")]["bootstrap"]
        for domain in DOMAINS
    }
    seed_interactions = {
        domain: score_cells[_cell_key(domain, "public_base")]["seed_interaction"]
        for domain in DOMAINS
    }
    gate = classify_interaction_gate(primary_summaries, seed_interactions=seed_interactions)
    for cell_id, cell in score_cells.items():
        cell["general_staging"] = bootstrap["cells"][cell_id]["general_staging"]
    general_summaries = {
        domain: bootstrap["cells"][_cell_key(domain, "public_base")]["general_staging"]
        for domain in DOMAINS
    }
    general_seed_contrasts = {
        domain: score_cells[_cell_key(domain, "public_base")]["seed_general_staging"]
        for domain in DOMAINS
    }
    general_gate = classify_general_staging_gate(general_summaries, seed_contrasts=general_seed_contrasts)
    return {
        "schema": SCORE_SCHEMA,
        "task_id": TASK_ID,
        "status": "TRR-P08_SCORED_AFTER_JOINT_FREEZE",
        "cells": score_cells,
        "bootstrap": bootstrap,
        "gate": gate,
        "general_staging_gate": general_gate,
        "truth_opened": True,
        "truth_payload_persisted": False,
        "claim_scope": "exploratory task-local P08 staged-versus-joint interaction; not a canonical replacement or universal mechanism claim",
    }





def _artifact_sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P08ScoreError(f"artifact is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_actual_record(path: Path, *, root: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    actual = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _artifact_sha256_file(path)}
    try:
        actual["path"] = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    return actual


def _artifact_load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P08ScoreError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P08ScoreError(f"{description} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise P08ScoreError(f"{description} must be a JSON object")
    return dict(value)


def _artifact_resolve_record(record: Mapping[str, Any], *, root: Path, description: str) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise P08ScoreError(f"{description} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise P08ScoreError(f"{description} is unavailable: {path}")
    return path


def _artifact_verify_record(record: Any, *, root: Path, description: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise P08ScoreError(f"{description} record is malformed")
    size = record.get("bytes")
    digest = record.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise P08ScoreError(f"{description} byte count is malformed")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise P08ScoreError(f"{description} SHA-256 is malformed")
    path = _artifact_resolve_record(record, root=root, description=description)
    actual = _artifact_actual_record(path, root=root)
    if actual["bytes"] != size or actual["sha256"] != digest:
        raise P08ScoreError(f"{description} hash or byte binding changed")
    return actual


def _artifact_digest(value: Any, *, description: str) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise P08ScoreError(f"{description} is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _artifact_require_digest(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise P08ScoreError(f"{description} must be a lowercase SHA-256")
    return value


def _artifact_same_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("bytes") == right.get("bytes") and left.get("sha256") == right.get("sha256")


def _artifact_load_array(path: Path, *, description: str) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    try:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                keys = list(archive.keys())
                if len(keys) != 1:
                    raise P08ScoreError(f"{description} NPZ must contain exactly one array")
                value = archive[keys[0]]
        elif suffix == ".safetensors":
            from safetensors.numpy import load_file  # type: ignore

            tensors = load_file(str(path))
            if len(tensors) != 1:
                raise P08ScoreError(f"{description} safetensors must contain exactly one array")
            value = next(iter(tensors.values()))
        else:
            raise P08ScoreError(f"{description} format must be .npy, .npz, or .safetensors")
    except P08ScoreError:
        raise
    except Exception as exc:
        raise P08ScoreError(f"{description} could not be loaded") from exc
    return np.ascontiguousarray(value)


def _artifact_validate_joint_freeze(
    *,
    repository_root: Path,
    freeze_receipt_path: Path,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Revalidate the create-only no-truth receipt before any truth read."""

    root = Path(repository_root).expanduser().resolve()
    freeze_path = Path(freeze_receipt_path).expanduser().resolve()
    freeze = _artifact_load_json(freeze_path, description="P08 joint freeze receipt")
    freeze_record = _artifact_actual_record(freeze_path, root=root)
    if freeze.get("schema") != freeze_matrix.FREEZE_SCHEMA or freeze.get("task_id") != TASK_ID or freeze.get("status") != "JOINT_FREEZE_VALIDATED_NO_TRUTH":
        raise P08ScoreError("P08 joint freeze is not the registered no-truth status")
    if freeze.get("truth_opened") is not False or freeze.get("create_only") is not True:
        raise P08ScoreError("P08 joint freeze has a post-truth or mutable status")
    if freeze.get("plan", {}).get("sha256") != expected_plan_sha256:
        raise P08ScoreError("P08 joint freeze plan hash differs from the approved plan")
    names = ("plan", "approval", "fit_receipt", "observation_manifest", "prediction_manifest")
    bindings: dict[str, dict[str, Any]] = {}
    binding_paths: dict[str, Path] = {}
    for name in names:
        raw_binding = freeze.get(name)
        bindings[name] = _artifact_verify_record(raw_binding, root=root, description=f"P08 freeze {name}")
        binding_paths[name] = _artifact_resolve_record(raw_binding, root=root, description=f"P08 freeze {name}")
    try:
        validated = freeze_matrix.validate_prediction_freeze(
            repository_root=root,
            plan_path=binding_paths["plan"],
            approval_path=binding_paths["approval"],
            fit_receipt_path=binding_paths["fit_receipt"],
            prediction_manifest_path=binding_paths["prediction_manifest"],
            observation_manifest_path=binding_paths["observation_manifest"],
            expected_plan_sha256=expected_plan_sha256,
        )
    except (freeze_matrix.P08FreezeError, OSError, ValueError, TypeError) as exc:
        raise P08ScoreError(f"P08 joint freeze revalidation failed: {exc}") from exc
    for name in names:
        fresh = validated.get({"fit_receipt": "fit_receipt", "observation_manifest": "observation_manifest", "prediction_manifest": "prediction_manifest"}.get(name, name))
        if isinstance(fresh, Mapping) and not _artifact_same_record(bindings[name], fresh):
            raise P08ScoreError(f"P08 joint freeze {name} binding changed since receipt")
    if int(freeze.get("prediction_count", -1)) != freeze_matrix.PREDICTION_COUNT or int(validated.get("prediction_count", -1)) != freeze_matrix.PREDICTION_COUNT:
        raise P08ScoreError("P08 joint freeze prediction count is incomplete")
    return {"freeze": freeze, "freeze_record": freeze_record, "bindings": bindings, "validated": validated, "root": root}


def _artifact_selection_identity(
    *,
    observation_manifest_path: Path,
    observation_manifest: Mapping[str, Any],
    root: Path,
    expected_record_ids: Mapping[str, str],
) -> dict[str, Any]:
    selection_binding = observation_manifest.get("selection_plan")
    selection_path = _artifact_resolve_record(selection_binding, root=root, description="P08 source selection")
    selection_record = _artifact_verify_record(selection_binding, root=root, description="P08 source selection")
    selection = _artifact_load_json(selection_path, description="P08 source selection")
    if selection.get("task_id") != TASK_ID or selection.get("truth_opened") is not False:
        raise P08ScoreError("P08 source selection is not a frozen no-truth artifact")
    records = selection.get("selection_rule", {}).get("records")
    if not isinstance(records, Mapping) or set(records) != set(DOMAINS):
        raise P08ScoreError("P08 source selection lacks both ordered domain record lists")
    record_ids: dict[str, list[str]] = {}
    sequence_hashes: dict[str, list[str]] = {}
    for domain in DOMAINS:
        rows = records[domain]
        if not isinstance(rows, list) or len(rows) != RECORDS_PER_DOMAIN:
            raise P08ScoreError(f"P08 source selection record count changed: {domain}")
        ids: list[str] = []
        sequences: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str) or not row.get("record_id"):
                raise P08ScoreError(f"P08 source selection record ID is malformed: {domain}/{index}")
            sequence = row.get("final_sequence_sha256")
            if not isinstance(sequence, str) or _SHA256.fullmatch(sequence) is None:
                raise P08ScoreError(f"P08 source selection sequence digest is malformed: {domain}/{index}")
            ids.append(row["record_id"])
            sequences.append(sequence)
        if len(set(ids)) != RECORDS_PER_DOMAIN or len(set(sequences)) != RECORDS_PER_DOMAIN:
            raise P08ScoreError(f"P08 source selection rows are not unique: {domain}")
        digest = _artifact_digest(ids, description=f"P08 {domain} record IDs")
        if digest != expected_record_ids.get(domain):
            raise P08ScoreError(f"P08 source selection order differs from frozen observations: {domain}")
        record_ids[domain] = ids
        sequence_hashes[domain] = sequences
    cells = observation_manifest.get("cells")
    if not isinstance(cells, list):
        raise P08ScoreError("P08 observation manifest has no cells")
    for row in cells:
        if isinstance(row, Mapping):
            domain = row.get("style", row.get("domain"))
            if domain in DOMAINS and row.get("record_ids_sha256") != expected_record_ids.get(domain):
                raise P08ScoreError(f"P08 observation/source order differs: {domain}")
    return {"selection": selection, "selection_record": selection_record, "record_ids": record_ids, "record_ids_sha256": {domain: _artifact_digest(record_ids[domain], description=f"P08 {domain} record IDs") for domain in DOMAINS}, "final_sequence_sha256": sequence_hashes}


def _artifact_load_truth_manifest(
    path: Path,
    *,
    root: Path,
    validated: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _artifact_load_json(manifest_path, description="P08 truth manifest")
    if manifest.get("schema") != TRUTH_SCHEMA or manifest.get("task_id") != TASK_ID or manifest.get("status") != TRUTH_STATUS:
        raise P08ScoreError("P08 truth manifest is not the registered post-freeze status")
    freeze_sha = _artifact_require_digest(manifest.get("joint_freeze_sha256"), description="P08 truth joint-freeze binding")
    if freeze_sha != validated["freeze_record"]["sha256"]:
        raise P08ScoreError("P08 truth manifest is bound to a different joint freeze")
    for field, expected_key in (("observation_manifest_sha256", "observation_manifest"), ("prediction_manifest_sha256", "prediction_manifest")):
        declared = _artifact_require_digest(manifest.get(field), description=f"P08 truth {field}")
        if declared != validated["bindings"][expected_key]["sha256"]:
            raise P08ScoreError(f"P08 truth manifest {field} binding differs")
    declared_selection = _artifact_require_digest(manifest.get("source_selection_sha256"), description="P08 truth source-selection binding")
    if declared_selection != selection["selection_record"]["sha256"]:
        raise P08ScoreError("P08 truth manifest is bound to a different source selection")
    if manifest.get("records_per_domain") != RECORDS_PER_DOMAIN or manifest.get("sequence_tokens_including_bos") != SEQUENCE_TOKENS or manifest.get("scored_post_bos_tokens") != SCORED_POST_BOS or manifest.get("vocabulary_size") != VOCABULARY_SIZE:
        raise P08ScoreError("P08 truth geometry changed")
    if manifest.get("target_conditions") != list(TARGETS):
        raise P08ScoreError("P08 truth target condition order changed")
    for flag in ("source_text_persisted", "target_labels_loaded", "model_loaded"):
        if manifest.get(flag) is True:
            raise P08ScoreError(f"P08 truth manifest contains forbidden persisted access: {flag}")
    if manifest.get("truth_opened") is not True or manifest.get("truth_materialized") is not True:
        raise P08ScoreError("P08 truth manifest does not certify evaluator access")
    if manifest.get("record_ids_sha256") != selection["record_ids_sha256"] or manifest.get("final_sequence_sha256") != selection["final_sequence_sha256"]:
        raise P08ScoreError("P08 truth source rows differ from the frozen selection")
    domains = manifest.get("domains")
    if not isinstance(domains, Mapping) or set(domains) != set(DOMAINS):
        raise P08ScoreError("P08 truth manifest domain set changed")
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for domain in DOMAINS:
        descriptor = domains[domain]
        truth_path = _artifact_resolve_record(descriptor, root=root, description=f"P08 truth {domain}")
        actual = _artifact_verify_record(descriptor, root=root, description=f"P08 truth {domain}")
        array = _artifact_load_array(truth_path, description=f"P08 truth {domain}")
        if array.shape != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or not np.issubdtype(array.dtype, np.integer):
            raise P08ScoreError(f"P08 truth {domain} geometry or dtype changed")
        if np.any(array < 0) or np.any(array >= VOCABULARY_SIZE) or not np.all(array[:, 0] == BOS_TOKEN_ID):
            raise P08ScoreError(f"P08 truth {domain} token IDs are outside the registered vocabulary")
        observed = [hashlib.sha256(np.ascontiguousarray(row, dtype=np.int32).tobytes(order="C")).hexdigest() for row in array]
        if observed != selection["final_sequence_sha256"][domain]:
            raise P08ScoreError(f"P08 truth {domain} row order or sequence fingerprints differ")
        arrays[domain] = array.astype(np.int64, copy=False)
        records[domain] = actual
    return arrays, records, {"manifest": manifest, "manifest_record": _artifact_actual_record(manifest_path, root=root)}


def _artifact_prediction_cells(
    *,
    validated: Mapping[str, Any],
    root: Path,
    record_ids: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    descriptors = validated.get("descriptors")
    if not isinstance(descriptors, Mapping):
        raise P08ScoreError("P08 joint validation has no prediction descriptors")
    for domain in DOMAINS:
        for target in TARGETS:
            cell_id = f"{domain}__{target}"
            predictions: dict[str, dict[str, np.ndarray]] = {method: {} for method in METHOD_ORDER}
            expected_digest = validated["descriptors"][f"{cell_id}::{REPLICATE_SEEDS[0]}::{METHOD_ORDER[0]}"]["record_ids_sha256"]
            if expected_digest != _artifact_digest(list(record_ids[domain]), description=f"P08 {domain} record IDs"):
                raise P08ScoreError(f"P08 prediction source order differs: {cell_id}")
            for seed in REPLICATE_SEEDS:
                for method in METHOD_ORDER:
                    key = f"{cell_id}::{seed}::{method}"
                    descriptor = descriptors.get(key)
                    if not isinstance(descriptor, Mapping):
                        raise P08ScoreError(f"P08 prediction descriptor is missing: {key}")
                    if descriptor.get("record_ids_sha256") != expected_digest:
                        raise P08ScoreError(f"P08 prediction source order differs: {key}")
                    _artifact_verify_record(descriptor.get("tie_counts"), root=root, description=f"P08 tie counts {key}")
                    prediction_path = _artifact_resolve_record(descriptor.get("prediction"), root=root, description=f"P08 prediction {key}")
                    prediction_record = _artifact_verify_record(descriptor.get("prediction"), root=root, description=f"P08 prediction {key}")
                    array = _artifact_load_array(prediction_path, description=f"P08 prediction {key}")
                    if array.shape != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or not np.issubdtype(array.dtype, np.integer):
                        raise P08ScoreError(f"P08 prediction {key} geometry or dtype changed")
                    predictions[method][str(seed)] = array.astype(np.int64, copy=False)
            cells[f"{domain}/{target}"] = {"domain": domain, "target": target, "record_ids": list(record_ids[domain]), "truth": None, "predictions": predictions}
    return cells


def run(
    *,
    repository_root: Path,
    freeze_receipt_path: Path,
    truth_manifest_path: Path,
    output_path: Path,
    expected_plan_sha256: str = freeze_matrix.APPROVED_PLAN_SHA256,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Revalidate the no-truth freeze, then load truth/predictions and score once."""

    root = Path(repository_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise P08ScoreError(f"score output is create-only: {output}")
    frozen = _artifact_validate_joint_freeze(repository_root=root, freeze_receipt_path=freeze_receipt_path, expected_plan_sha256=expected_plan_sha256)
    observation_path = _artifact_resolve_record(frozen["freeze"]["observation_manifest"], root=root, description="P08 observation manifest")
    observation_manifest = _artifact_load_json(observation_path, description="P08 observation manifest")
    expected_ids = {domain: observation_manifest["source_pairing"]["record_ids_sha256"][domain] for domain in DOMAINS} if isinstance(observation_manifest.get("source_pairing"), Mapping) else {}
    selection = _artifact_selection_identity(observation_manifest_path=observation_path, observation_manifest=observation_manifest, root=root, expected_record_ids=expected_ids)
    truth_arrays, truth_records, truth_meta = _artifact_load_truth_manifest(truth_manifest_path, root=root, validated=frozen, selection=selection)
    cells = _artifact_prediction_cells(validated=frozen["validated"], root=root, record_ids=selection["record_ids"])
    for cell_id, cell in cells.items():
        cell["truth"] = truth_arrays[cell["domain"]]
    try:
        result = score_arrays(cells, bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed)
    except (P08ScoreError, P08MetricsError):
        raise
    finally:
        # The score artifact contains metrics and fixed record IDs only; release
        # evaluator arrays before writing it so truth is never serialized.
        truth_arrays.clear()
        for cell in cells.values():
            cell["truth"] = None
            for methods in cell["predictions"].values():
                methods.clear()
    result["provenance"] = {
        "joint_validation": {"status": frozen["validated"]["status"], "freeze_receipt": frozen["freeze_record"], "bindings": frozen["bindings"], "prediction_count": frozen["validated"]["prediction_count"]},
        "source_selection": selection["selection_record"],
        "truth_manifest": truth_meta["manifest_record"],
        "truth_files": truth_records,
        "command": list(sys.argv),
        "scored_after_joint_freeze": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise P08ScoreError(f"score output is create-only: {output}") from exc
    return {"task_id": TASK_ID, "status": result["status"], "disposition": result["gate"]["disposition"], "output": _artifact_actual_record(output, root=root), "truth_opened_once": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", default=freeze_matrix.APPROVED_PLAN_SHA256)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    try:
        result = run(repository_root=args.repository_root, freeze_receipt_path=args.freeze_receipt, truth_manifest_path=args.truth_manifest, output_path=args.output, expected_plan_sha256=args.expected_plan_sha256, bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed)
    except (P08ScoreError, P08MetricsError, freeze_matrix.P08FreezeError, OSError, ValueError, TypeError) as exc:
        print(f"TRR-P08 score failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["P08ScoreError", "SCORE_SCHEMA", "TRUTH_SCHEMA", "score_arrays", "run", "validate_joint_freeze_artifacts"]


def validate_joint_freeze_artifacts(*, repository_root: Path, freeze_receipt_path: Path, expected_plan_sha256: str = freeze_matrix.APPROVED_PLAN_SHA256) -> dict[str, Any]:
    """Public metadata-only pre-truth wrapper used by setup/root review."""
    return _artifact_validate_joint_freeze(repository_root=repository_root, freeze_receipt_path=freeze_receipt_path, expected_plan_sha256=expected_plan_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
