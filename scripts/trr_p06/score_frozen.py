#!/usr/bin/env python3
"""TRR-P06 joint-freeze scorer and registered decision gate.

This task-local boundary verifies the complete prediction matrix before any
truth artifact is opened.  The prediction manifest contract is deliberately
small and source-free:

* ``student_cells`` has one ``{domain}__{target}`` entry for each of pile and
  finance crossed with public_base and public_lora_2601;
* each student cell has ``replicates`` keyed by ``6106`` and ``6107`` and each
  replicate has the three registered method descriptors;
* each method descriptor carries a prediction file record, shape, record/mask/
  position/observation digests, and the frozen state digest;
* ``anchor_cells`` has one separate first-64 public_base descriptor per domain.

The freeze receipt binds the plan, source selection, observation manifest, and
prediction manifest by path and SHA-256, plus explicit finite-fit, capacity,
resource, and source-pairing preconditions.  Validation reads JSON metadata and
hashes prediction/state files only.  ``run`` opens evaluator truth only after
that validation succeeds; no truth payload is copied into the result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from token_reconstruction.trr_p06_metrics import (  # noqa: E402
    BOS_TOKEN_ID,
    METHOD_ORDER,
    P06MetricsError,
    paired_cluster_bootstrap,
    score_method,
)


TASK_ID = "TRR-P06"
FREEZE_SCHEMA = "token-reconstruction.trr-p06-joint-freeze.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr-p06-prediction-manifest.v1"
TRUTH_SCHEMA = "token-reconstruction.trr-p06-truth-manifest.v1"
SCORE_SCHEMA = "token-reconstruction.trr-p06-score.v1"
DOMAINS = ("pile", "finance")
TARGETS = ("public_base", "public_lora_2601")
REPLICATE_SEEDS = (6106, 6107)
ANCHOR_METHOD_ID = "frozen_a1_a2_k256"
RECORDS_PER_DOMAIN = 256
ANCHOR_RECORDS_PER_DOMAIN = 64
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
VOCABULARY_SIZE = 128256
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class P06ScoreError(RuntimeError):
    """Raised when the P06 score boundary fails closed."""



def _sha256_file(path: Path) -> str:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P06ScoreError(f"file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actual_record(path: Path, *, root: Path | None = None, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    record = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}
    if root is not None:
        try:
            record["path"] = path.relative_to(root).as_posix()
        except ValueError:
            pass
    return record


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P06ScoreError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P06ScoreError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise P06ScoreError(f"{description} must be a JSON object")
    return dict(value)


def _resolve_record_path(record: Mapping[str, Any], *, root: Path, description: str) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise P06ScoreError(f"{description} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise P06ScoreError(f"{description} is unavailable: {path}")
    return path


def _verify_file_record(record: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise P06ScoreError(f"{description} record is malformed")
    declared_bytes = record.get("bytes")
    declared_sha = record.get("sha256")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) or declared_bytes <= 0:
        raise P06ScoreError(f"{description} byte count is malformed")
    if not isinstance(declared_sha, str) or _SHA256.fullmatch(declared_sha) is None:
        raise P06ScoreError(f"{description} SHA-256 is malformed")
    path = _resolve_record_path(record, root=root, description=description)
    actual = _actual_record(path, root=root, description=description)
    if actual["bytes"] != declared_bytes or actual["sha256"] != declared_sha:
        raise P06ScoreError(f"{description} hash or byte binding changed")
    return actual


def _verify_binding(
    name: str,
    *,
    freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    freeze_record = freeze.get(name)
    manifest_record = manifest.get(name)
    if not isinstance(freeze_record, Mapping) or not isinstance(manifest_record, Mapping):
        raise P06ScoreError(f"joint freeze lacks {name} binding")
    actual = _verify_file_record(freeze_record, root=root, description=f"{name} binding")
    if dict(manifest_record) != dict(freeze_record):
        raise P06ScoreError(f"prediction manifest {name} binding differs from joint freeze")
    payload = _load_json(Path(actual["path"]) if Path(actual["path"]).is_absolute() else root / actual["path"], description=name)
    if payload.get("task_id") != TASK_ID:
        raise P06ScoreError(f"{name} task ID changed")
    if payload.get("truth_opened") is True:
        raise P06ScoreError(f"{name} was written after truth access")
    return actual


def _require_digest(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise P06ScoreError(f"{description} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise P06ScoreError(f"{description} must be a full commit hash")
    return value


def _matrix_key(domain: str, target: str) -> str:
    return f"{domain}__{target}"


def _validate_descriptor_common(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    description: str,
    domain: str,
    target: str,
    method_id: str,
    seed: int,
    records: int,
    state_sha: str,
) -> dict[str, Any]:
    if descriptor.get("task_id") != TASK_ID or descriptor.get("domain") != domain or descriptor.get("target") != target:
        raise P06ScoreError(f"{description} task/domain/target binding changed")
    if descriptor.get("method_id") != method_id:
        raise P06ScoreError(f"{description} method/seed binding changed")
    raw_seed = descriptor.get("seed")
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, (int, np.integer)) or int(raw_seed) != seed:
        raise P06ScoreError(f"{description} method/seed binding changed")
    if list(descriptor.get("shape", ())) != [records, SEQUENCE_TOKENS]:
        raise P06ScoreError(f"{description} prediction shape changed")
    if descriptor.get("records") != records or descriptor.get("sequence_tokens") != SEQUENCE_TOKENS or descriptor.get("scored_post_bos_tokens") != SCORED_POST_BOS:
        raise P06ScoreError(f"{description} prediction geometry changed")
    if descriptor.get("truth_opened") is True or descriptor.get("candidate_arrays_persisted") is True:
        raise P06ScoreError(f"{description} contains a post-truth or candidate payload")
    for field in ("record_ids_sha256", "attention_mask_sha256", "position_ids_sha256", "observation_sha256"):
        _require_digest(descriptor.get(field), description=f"{description}.{field}")
    if descriptor.get("state_sha256") != state_sha:
        raise P06ScoreError(f"{description} state hash differs from registered state")
    prediction_record = descriptor.get("prediction")
    actual_prediction = _verify_file_record(prediction_record, root=root, description=f"{description} prediction")
    descriptor_copy = dict(descriptor)
    descriptor_copy["prediction"] = actual_prediction
    return descriptor_copy


def _validate_student_matrix(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    if list(manifest.get("domains", ())) != list(DOMAINS) or list(manifest.get("target_conditions", ())) != list(TARGETS):
        raise P06ScoreError("prediction manifest domain/target order changed")
    if list(manifest.get("method_order", ())) != list(METHOD_ORDER):
        raise P06ScoreError("prediction manifest method order changed")
    raw_replicate_seeds = manifest.get("replicate_seeds")
    if not isinstance(raw_replicate_seeds, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in raw_replicate_seeds
    ) or [int(value) for value in raw_replicate_seeds] != list(REPLICATE_SEEDS):
        raise P06ScoreError("prediction manifest replicate seed order changed")
    state_bindings = manifest.get("state_bindings")
    if not isinstance(state_bindings, Mapping):
        raise P06ScoreError("prediction manifest has no state bindings")
    state_sha: dict[tuple[int, str], str] = {}
    checked_states: dict[str, Any] = {}
    for seed in REPLICATE_SEEDS:
        for method_id in METHOD_ORDER:
            key = f"{seed}::{method_id}"
            binding = state_bindings.get(key)
            if not isinstance(binding, Mapping):
                raise P06ScoreError(f"missing state binding: {key}")
            actual = _verify_file_record(binding, root=root, description=f"state {key}")
            digest = _require_digest(binding.get("sha256"), description=f"state {key}")
            if digest != actual["sha256"]:
                raise P06ScoreError(f"state {key} hash changed")
            state_sha[(seed, method_id)] = digest
            checked_states[key] = actual

    cells = manifest.get("student_cells")
    expected_cells = {_matrix_key(domain, target) for domain in DOMAINS for target in TARGETS}
    if not isinstance(cells, Mapping) or set(cells) != expected_cells:
        raise P06ScoreError("student prediction matrix is not exactly the four registered target cells")
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for domain in DOMAINS:
        for target in TARGETS:
            cell_id = _matrix_key(domain, target)
            cell = cells[cell_id]
            if not isinstance(cell, Mapping) or cell.get("domain") != domain or cell.get("target") != target:
                raise P06ScoreError(f"student cell binding changed: {cell_id}")
            replicates = cell.get("replicates")
            if not isinstance(replicates, Mapping) or {str(seed) for seed in replicates} != {str(seed) for seed in REPLICATE_SEEDS}:
                raise P06ScoreError(f"student cell replicate matrix changed: {cell_id}")
            normalized[cell_id] = {}
            for seed in REPLICATE_SEEDS:
                raw_methods = replicates.get(str(seed), replicates.get(seed))
                if not isinstance(raw_methods, Mapping) or set(raw_methods) != set(METHOD_ORDER):
                    raise P06ScoreError(f"student cell method matrix changed: {cell_id}/{seed}")
                normalized[cell_id][str(seed)] = {}
                for method_id in METHOD_ORDER:
                    normalized[cell_id][str(seed)][method_id] = _validate_descriptor_common(
                        raw_methods[method_id],
                        root=root,
                        description=f"student {cell_id}/{seed}/{method_id}",
                        domain=domain,
                        target=target,
                        method_id=method_id,
                        seed=seed,
                        records=RECORDS_PER_DOMAIN,
                        state_sha=state_sha[(seed, method_id)],
                    )

    # Every method/seed sees the same source order and geometry within a cell;
    # target pairing keeps those three identity digests equal across targets.
    geometry_by_domain: dict[str, tuple[str, str, str]] = {}
    observation_by_cell: dict[str, str] = {}
    for domain in DOMAINS:
        for target in TARGETS:
            cell_id = _matrix_key(domain, target)
            rows = [
                normalized[cell_id][str(seed)][method]
                for seed in REPLICATE_SEEDS
                for method in METHOD_ORDER
            ]
            geometry = (
                rows[0]["record_ids_sha256"],
                rows[0]["attention_mask_sha256"],
                rows[0]["position_ids_sha256"],
            )
            if any((row["record_ids_sha256"], row["attention_mask_sha256"], row["position_ids_sha256"]) != geometry for row in rows):
                raise P06ScoreError(f"student cell masks/positions/source order differ: {cell_id}")
            observation_digests = {row["observation_sha256"] for row in rows}
            if len(observation_digests) != 1:
                raise P06ScoreError(f"student cell observation binding differs across arms: {cell_id}")
            observation_by_cell[cell_id] = rows[0]["observation_sha256"]
            if domain in geometry_by_domain and geometry_by_domain[domain] != geometry:
                raise P06ScoreError(f"paired targets changed source order or mask geometry: {domain}")
            geometry_by_domain[domain] = geometry
    return normalized, {"states": checked_states, "observation_sha256": observation_by_cell, "geometry": geometry_by_domain}


def _validate_anchor_matrix(
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, dict[str, Any]]:
    anchors = manifest.get("anchor_cells")
    if not isinstance(anchors, Mapping) or set(anchors) != set(DOMAINS):
        raise P06ScoreError("anchor matrix must contain one separate cell per domain")
    anchor_state_sha = _require_digest(
        freeze.get("anchor_state_sha256"), description="joint freeze anchor state"
    )
    manifest_anchor_state = _require_digest(
        manifest.get("anchor_state_sha256"), description="prediction manifest anchor state"
    )
    if manifest_anchor_state != anchor_state_sha:
        raise P06ScoreError("anchor state binding differs between manifest and joint freeze")
    subset_hashes = freeze.get("anchor_subset_record_ids_sha256")
    manifest_subset_hashes = manifest.get("anchor_subset_record_ids_sha256")
    if not isinstance(subset_hashes, Mapping) or not isinstance(manifest_subset_hashes, Mapping) or dict(subset_hashes) != dict(manifest_subset_hashes):
        raise P06ScoreError("anchor source-subset bindings are missing or changed")
    normalized: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        descriptor = anchors[domain]
        if not isinstance(descriptor, Mapping):
            raise P06ScoreError(f"anchor descriptor is malformed: {domain}")
        if descriptor.get("task_id") != TASK_ID or descriptor.get("domain") != domain or descriptor.get("target") != "public_base":
            raise P06ScoreError(f"anchor target binding changed: {domain}")
        if descriptor.get("method_id") != ANCHOR_METHOD_ID or descriptor.get("subset") != "first64_public_base":
            raise P06ScoreError(f"anchor identity changed: {domain}")
        if list(descriptor.get("shape", ())) != [ANCHOR_RECORDS_PER_DOMAIN, SEQUENCE_TOKENS] or descriptor.get("records") != ANCHOR_RECORDS_PER_DOMAIN or descriptor.get("scored_post_bos_tokens") != SCORED_POST_BOS:
            raise P06ScoreError(f"anchor geometry changed: {domain}")
        if descriptor.get("anchor_subset_record_ids_sha256") != subset_hashes.get(domain):
            raise P06ScoreError(f"anchor source subset changed: {domain}")
        if descriptor.get("state_sha256") != anchor_state_sha:
            raise P06ScoreError(f"anchor state changed: {domain}")
        for field in ("record_ids_sha256", "attention_mask_sha256", "position_ids_sha256"):
            _require_digest(descriptor.get(field), description=f"anchor {domain}.{field}")
        actual = _verify_file_record(descriptor.get("prediction"), root=root, description=f"anchor {domain} prediction")
        item = dict(descriptor)
        item["prediction"] = actual
        normalized[domain] = item
    return normalized


def validate_joint_freeze(
    *,
    repository_root: Path,
    freeze_receipt_path: Path,
    prediction_manifest_path: Path,
) -> dict[str, Any]:
    """Verify the complete source/state/prediction matrix without truth access."""

    root = Path(repository_root).expanduser().resolve()
    freeze_path = Path(freeze_receipt_path).expanduser().resolve()
    manifest_path = Path(prediction_manifest_path).expanduser().resolve()
    freeze = _load_json(freeze_path, description="P06 joint freeze receipt")
    manifest = _load_json(manifest_path, description="P06 prediction manifest")
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("task_id") != TASK_ID or freeze.get("status") != "FROZEN_P06_MATRIX_NO_TRUTH":
        raise P06ScoreError("joint freeze receipt is not the registered no-truth status")
    if manifest.get("schema") != PREDICTION_SCHEMA or manifest.get("task_id") != TASK_ID or manifest.get("status") != "FROZEN_P06_PREDICTIONS_NO_TRUTH":
        raise P06ScoreError("prediction manifest is not the registered no-truth status")
    if freeze.get("truth_opened") is not False or manifest.get("truth_opened") is not False:
        raise P06ScoreError("joint freeze or prediction manifest already opened truth")
    if freeze.get("prediction_manifest_sha256") != _sha256_file(manifest_path):
        raise P06ScoreError("joint freeze prediction-manifest hash changed")
    if manifest.get("code_commit") != freeze.get("code_commit"):
        raise P06ScoreError("prediction and freeze code commits differ")
    _require_commit(freeze.get("code_commit"), description="joint freeze code commit")
    bindings = {
        name: _verify_binding(name, freeze=freeze, manifest=manifest, root=root)
        for name in ("plan", "source_selection", "observation_manifest")
    }
    preconditions = freeze.get("scientific_preconditions")
    required_preconditions = (
        "plan_frozen",
        "resource_qualified",
        "capacity_qualified",
        "all_fits_finite",
        "source_pairing_validated",
    )
    if not isinstance(preconditions, Mapping) or any(preconditions.get(key) is not True for key in required_preconditions):
        raise P06ScoreError("joint freeze scientific preconditions are incomplete")
    students, student_meta = _validate_student_matrix(manifest, root=root)
    anchors = _validate_anchor_matrix(manifest, freeze, root=root)
    return {
        "schema": "token-reconstruction.trr-p06-joint-validation.v1",
        "task_id": TASK_ID,
        "status": "JOINT_FREEZE_VALIDATED_NO_TRUTH",
        "code_commit": freeze["code_commit"],
        "freeze_receipt": _actual_record(freeze_path, root=root, description="joint freeze receipt"),
        "prediction_manifest": _actual_record(manifest_path, root=root, description="prediction manifest"),
        "bindings": bindings,
        "scientific_preconditions": dict(preconditions),
        "student_cells": students,
        "anchor_cells": anchors,
        "student_metadata": student_meta,
        "domains": list(DOMAINS),
        "target_conditions": list(TARGETS),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "methods": list(METHOD_ORDER),
        "truth_opened": False,
    }


def _load_array(path: Path, *, description: str) -> np.ndarray:
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                keys = list(archive.keys())
                if len(keys) != 1:
                    raise P06ScoreError(f"{description} NPZ must contain exactly one array")
                value = archive[keys[0]]
        elif suffix == ".safetensors":
            from safetensors.numpy import load_file  # type: ignore

            tensors = load_file(str(path))
            if len(tensors) != 1:
                raise P06ScoreError(f"{description} safetensors must contain exactly one array")
            value = next(iter(tensors.values()))
        else:
            raise P06ScoreError(f"{description} format must be .npy, .npz, or .safetensors")
    except P06ScoreError:
        raise
    except Exception as exc:
        raise P06ScoreError(f"{description} could not be loaded") from exc
    return np.ascontiguousarray(value)


def _load_truth_manifest(
    path: Path,
    *,
    root: Path,
    records_per_domain: int = RECORDS_PER_DOMAIN,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = _load_json(path, description="P06 truth manifest")
    if manifest.get("schema") != TRUTH_SCHEMA or manifest.get("task_id") != TASK_ID or manifest.get("status") != "TRUTH_READY_AFTER_JOINT_FREEZE":
        raise P06ScoreError("truth manifest is not the post-freeze status")
    domains = manifest.get("domains")
    if not isinstance(domains, Mapping) or set(domains) != set(DOMAINS):
        raise P06ScoreError("truth manifest domain set changed")
    truth: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for domain in DOMAINS:
        descriptor = domains[domain]
        truth_path = _resolve_record_path(descriptor, root=root, description=f"truth {domain}")
        actual = _verify_file_record(descriptor, root=root, description=f"truth {domain}")
        array = _load_array(truth_path, description=f"truth {domain}")
        if array.shape != (records_per_domain, SEQUENCE_TOKENS) or not np.issubdtype(array.dtype, np.integer):
            raise P06ScoreError(f"truth {domain} geometry or dtype changed")
        if not np.all(array[:, 0] == BOS_TOKEN_ID) or np.any(array < 0) or np.any(array >= VOCABULARY_SIZE):
            raise P06ScoreError(f"truth {domain} token IDs are outside the frozen public geometry")
        truth[domain] = array.astype(np.int64, copy=False)
        records[domain] = actual
    return truth, records


def _load_prediction_descriptor(descriptor: Mapping[str, Any], *, root: Path, description: str) -> np.ndarray:
    path = _resolve_record_path(descriptor["prediction"], root=root, description=f"{description} prediction")
    array = _load_array(path, description=f"{description} prediction")
    if array.shape != (int(descriptor["records"]), SEQUENCE_TOKENS) or not np.issubdtype(array.dtype, np.integer):
        raise P06ScoreError(f"{description} prediction geometry or dtype changed")
    return array.astype(np.int64, copy=False)


def _compact_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method_id": score.get("method_id"),
        "records": score["records"],
        "record_ids": score["record_ids"],
        "metrics": score["metrics"],
        "position_metrics": score["position_metrics"],
        "per_record": [
            {
                key: row[key]
                for key in (
                    "record_id",
                    "correct_tokens",
                    "scored_tokens",
                    "token_accuracy",
                    "exact_eligible",
                    "exact_record",
                )
            }
            for row in score["per_record"]
        ],
    }


def _compact_anchor_score(score: Mapping[str, Any], *, domain: str) -> dict[str, Any]:
    return {
        "domain": domain,
        "method_id": ANCHOR_METHOD_ID,
        "records": score["records"],
        "scored_tokens": score["metrics"]["scored_tokens"],
        "correct_tokens": score["metrics"]["correct_tokens"],
        "token_accuracy": score["metrics"]["token_accuracy"],
        "exact_records": score["metrics"]["exact_records"],
        "exact_denominator": score["metrics"]["exact_denominator"],
        "exact_record_rate": score["metrics"]["exact_record_rate"],
    }


def classify_registered_gate(
    bootstrap: Mapping[str, Any],
    *,
    scientific_preconditions: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared P06 public-base benefit/harm/negative gate."""

    required = ("resource_qualified", "capacity_qualified", "all_fits_finite", "source_pairing_validated")
    preconditions_pass = all(scientific_preconditions.get(key) is True for key in required)
    domains = bootstrap.get("domains")
    if not isinstance(domains, Mapping) or set(domains) != set(DOMAINS):
        raise P06ScoreError("bootstrap result lacks both P06 domains")
    outcomes: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    harm_failures: list[dict[str, Any]] = []
    qualification_rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        target = domains[domain].get("targets", {}).get("public_base")
        if not isinstance(target, Mapping):
            raise P06ScoreError(f"bootstrap result lacks public_base: {domain}")
        contrasts = target.get("contrasts")
        if not isinstance(contrasts, Mapping) or "full_minus_past" not in contrasts or "full_minus_positionwise" not in contrasts:
            raise P06ScoreError(f"bootstrap result lacks registered contrasts: {domain}")
        primary = contrasts["full_minus_past"]
        point = primary.get("point")
        if not isinstance(point, Mapping):
            raise P06ScoreError(f"bootstrap primary point is missing: {domain}")
        for metric, margin, harm in (("token_delta_pp", 1.0, -1.0), ("exact_delta_pp", 5.0, -5.0)):
            ci_key = "token_delta_ci95_percentile_pp" if metric == "token_delta_pp" else "exact_delta_ci95_percentile_pp"
            value = point.get(metric)
            ci = primary.get(ci_key)
            finite = (
                isinstance(value, (int, float))
                and np.isfinite(float(value))
                and isinstance(ci, list)
                and len(ci) == 2
                and all(isinstance(v, (int, float)) and np.isfinite(float(v)) for v in ci)
            )
            row = {
                "domain": domain,
                "target": "public_base",
                "contrast": "full_minus_past",
                "metric": metric,
                "point_pp": float(value) if finite else None,
                "ci95_pp": [float(ci[0]), float(ci[1])] if finite else [None, None],
                "benefit_margin_pp": margin,
                "harm_bound_pp": harm,
                "finite": finite,
                "harm_cleared": bool(finite and float(ci[0]) > harm),
                "benefit_supported": bool(finite and float(value) >= margin and float(ci[0]) > 0.0),
                "qualified_negative": bool(finite and float(ci[1]) < margin),
            }
            outcomes.append(row)
            if row["benefit_supported"]:
                support.append(row)
            if not row["harm_cleared"]:
                harm_failures.append(row)
            qualification_rows.append(row)
    all_finite = bool(qualification_rows) and all(row["finite"] for row in qualification_rows)
    harm_cleared = not harm_failures
    if not preconditions_pass:
        decision = "BLOCKED_PRECONDITIONS"
    elif support and harm_cleared:
        decision = "PROMOTE_VISIBILITY_FAMILY"
    elif all_finite and all(row["qualified_negative"] for row in qualification_rows):
        decision = "QUALIFIED_NEGATIVE_RETAIN_PAST_ONLY"
    else:
        decision = "INCONCLUSIVE"
    changed_target = {
        domain: {
            "target": "public_lora_2601",
            "primary_contrast_present": "full_minus_past" in domains[domain].get("targets", {}).get("public_lora_2601", {}).get("contrasts", {}),
            "used_for_registered_gate": False,
        }
        for domain in DOMAINS
    }
    return {
        "schema": "token-reconstruction.trr-p06-registered-gate.v1",
        "decision": decision,
        "preconditions_pass": preconditions_pass,
        "scientific_preconditions": dict(scientific_preconditions),
        "public_base_outcomes": outcomes,
        "supporting_outcomes": support,
        "harm_failures": harm_failures,
        "harm_cleared": harm_cleared,
        "all_public_base_outcomes_finite": all_finite,
        "changed_target": changed_target,
        "thresholds": {
            "token_benefit_pp": 1.0,
            "exact_benefit_pp": 5.0,
            "token_harm_bound_pp": -1.0,
            "exact_harm_bound_pp": -5.0,
        },
        "anchor_used_for_gate": False,
        "truth_opened": True,
    }


def score_arrays(
    verified: Mapping[str, Any],
    *,
    repository_root: Path,
    truth_by_domain: Mapping[str, np.ndarray],
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 6306,
) -> dict[str, Any]:
    """Score verified student/anchor arrays already materialized after the gate."""

    root = Path(repository_root).expanduser().resolve()
    if verified.get("status") != "JOINT_FREEZE_VALIDATED_NO_TRUTH" or verified.get("truth_opened") is not False:
        raise P06ScoreError("score_arrays requires the validated no-truth joint receipt")
    if set(truth_by_domain) != set(DOMAINS):
        raise P06ScoreError("truth arrays must contain both P06 domains")
    cells: dict[str, dict[str, Any]] = {}
    compact_scores: dict[str, Any] = {}
    for domain in DOMAINS:
        labels = np.asarray(truth_by_domain[domain])
        if labels.shape != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
            raise P06ScoreError(f"truth array geometry changed: {domain}")
        if (
            not np.issubdtype(labels.dtype, np.integer)
            or not np.all(labels[:, 0] == BOS_TOKEN_ID)
            or np.any(labels < 0)
            or np.any(labels >= VOCABULARY_SIZE)
        ):
            raise P06ScoreError(f"truth array IDs changed: {domain}")
        record_ids = tuple(f"{domain}-opaque-{index:04d}" for index in range(RECORDS_PER_DOMAIN))
        mask = np.ones((RECORDS_PER_DOMAIN, SEQUENCE_TOKENS), dtype=bool)
        for target in TARGETS:
            cell_id = _matrix_key(domain, target)
            cell_methods: dict[str, dict[str, Any]] = {}
            compact_scores[cell_id] = {}
            for seed in REPLICATE_SEEDS:
                cell_methods_for_seed: dict[str, Any] = {}
                compact_scores[cell_id][str(seed)] = {}
                for method_id in METHOD_ORDER:
                    descriptor = verified["student_cells"][cell_id][str(seed)][method_id]
                    predictions = _load_prediction_descriptor(
                        descriptor, root=root, description=f"student {cell_id}/{seed}/{method_id}"
                    )
                    score = score_method(
                        predictions,
                        labels,
                        record_ids=record_ids,
                        attention_mask=mask,
                        method_id=method_id,
                    )
                    cell_methods_for_seed[method_id] = score
                    compact_scores[cell_id][str(seed)][method_id] = _compact_score(score)
                cell_methods[str(seed)] = cell_methods_for_seed
            cells[cell_id] = {"domain": domain, "target": target, "replicates": cell_methods}

    bootstrap = paired_cluster_bootstrap(
        cells,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    anchor_scores: dict[str, Any] = {}
    for domain in DOMAINS:
        labels = np.asarray(truth_by_domain[domain])[:ANCHOR_RECORDS_PER_DOMAIN]
        descriptor = verified["anchor_cells"][domain]
        predictions = _load_prediction_descriptor(
            descriptor, root=root, description=f"anchor {domain}"
        )
        if predictions.shape[0] != ANCHOR_RECORDS_PER_DOMAIN:
            raise P06ScoreError(f"anchor prediction count changed: {domain}")
        mask = np.ones((ANCHOR_RECORDS_PER_DOMAIN, SEQUENCE_TOKENS), dtype=bool)
        anchor_ids = tuple(f"{domain}-anchor-{index:04d}" for index in range(ANCHOR_RECORDS_PER_DOMAIN))
        scored = score_method(
            predictions,
            labels,
            record_ids=anchor_ids,
            attention_mask=mask,
            method_id=None,
        )
        anchor_scores[domain] = _compact_anchor_score(scored, domain=domain)
    gate = classify_registered_gate(
        bootstrap,
        scientific_preconditions=verified["scientific_preconditions"],
    )
    return {
        "schema": SCORE_SCHEMA,
        "task_id": TASK_ID,
        "status": "TRR-P06_SCORED_AFTER_JOINT_FREEZE",
        "student": {
            "bootstrap": bootstrap,
            "method_scores": compact_scores,
        },
        "anchor": {
            "method_id": ANCHOR_METHOD_ID,
            "target": "public_base",
            "domains": anchor_scores,
            "separate_denominator": True,
        },
        "gate": gate,
        "truth_opened": True,
        "truth_payload_persisted": False,
        "claim_scope": "exploratory task-local P06 natural-panel result under the frozen H128 family and target conditions",
    }


def run(
    *,
    repository_root: Path,
    freeze_receipt_path: Path,
    prediction_manifest_path: Path,
    truth_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate the no-truth joint freeze, then open truth once and score."""

    root = Path(repository_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise P06ScoreError(f"score output is create-only: {output}")
    verified = validate_joint_freeze(
        repository_root=root,
        freeze_receipt_path=freeze_receipt_path,
        prediction_manifest_path=prediction_manifest_path,
    )
    truth_manifest = Path(truth_manifest_path).expanduser().resolve()
    truth_by_domain, truth_records = _load_truth_manifest(truth_manifest, root=root)
    result = score_arrays(verified, repository_root=root, truth_by_domain=truth_by_domain)
    result["provenance"] = {
        "joint_validation": {
            "status": verified["status"],
            "freeze_receipt": verified["freeze_receipt"],
            "prediction_manifest": verified["prediction_manifest"],
            "bindings": verified["bindings"],
            "code_commit": verified["code_commit"],
        },
        "truth_manifest": _actual_record(truth_manifest, root=root, description="truth manifest"),
        "truth_files": truth_records,
        "scored_after_joint_freeze": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {
        "task_id": TASK_ID,
        "status": result["status"],
        "decision": result["gate"]["decision"],
        "output": _actual_record(output, root=root, description="score output"),
        "truth_opened_once": True,
    }



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(
            repository_root=args.repository_root,
            freeze_receipt_path=args.freeze_receipt,
            prediction_manifest_path=args.prediction_manifest,
            truth_manifest_path=args.truth_manifest,
            output_path=args.output,
        )
    except (P06ScoreError, P06MetricsError, OSError, ValueError) as exc:
        print(f"TRR-P06 score failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
