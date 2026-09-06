#!/usr/bin/env python3
"""Freeze all P04 predictions before the evaluator truth gate.

This command is intentionally metadata-only.  It validates the fresh public
panel and every required affine/S/H/D prediction group for both paired target
conditions, plus the separate native A1+A2 anchor groups.  It never opens an
evaluation-truth file.  The resulting receipt is the only input that permits
``score_predictions.py`` to read private truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

from scripts.trr_p04 import score_predictions as scorer


FREEZE_SCHEMA = "token-reconstruction.trr-p04-freeze.v1"
PREDICTION_SCHEMA = scorer.PREDICTION_SCHEMA
METHODS = scorer.DEFAULT_METHODS
SEEDS = scorer.DEFAULT_SEEDS
CONDITIONS = scorer.DEFAULT_CONDITIONS


class FreezeError(ValueError):
    """Raised when a public prediction set is incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FreezeError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FreezeError(f"{description} must be an object")
    return value


def _prediction_descriptor(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FreezeError(f"prediction file is unavailable: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _expected_groups() -> list[tuple[str, int | None, str, bool]]:
    return [
        (method, seed, condition, False)
        for condition in CONDITIONS
        for seed in SEEDS
        for method in METHODS
    ] + [("native_a1_a2", None, condition, True) for condition in CONDITIONS]


def _read_prediction_groups(
    paths: Sequence[Path],
    *,
    panel: Mapping[str, Mapping[str, Any]],
    expected: Sequence[tuple[str, int | None, str, bool]],
) -> dict[tuple[str, int | None, str, bool], set[str]]:
    expected_set = set(expected)
    groups: dict[tuple[str, int | None, str, bool], set[str]] = {}
    for path in paths:
        rows = scorer._read_jsonl(path, description=f"prediction file {path}")
        for line_number, row in enumerate(rows, start=1):
            group, record_id, _ = scorer._prediction_row(
                row,
                panel=panel,
                description=f"prediction file {path} line {line_number}",
            )
            if group not in expected_set:
                raise FreezeError(f"prediction group is not part of the frozen P04 set: {group}")
            if group not in groups:
                groups[group] = set()
            if record_id in groups[group]:
                raise FreezeError(f"prediction record is duplicated in group {group}: {record_id}")
            groups[group].add(record_id)
    all_ids = set(panel)
    anchor_ids = {record_id for record_id, row in panel.items() if row["anchor"]}
    for group in expected:
        if group not in groups:
            raise FreezeError(f"prediction group is missing: {group}")
        required = anchor_ids if group[3] else all_ids
        if groups[group] != required:
            raise FreezeError(f"prediction group has incomplete record coverage: {group}")
    return groups


def _state_descriptors(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the eight implementation-owned state bindings without loading tensors."""

    if path.is_symlink() or not path.is_file():
        raise FreezeError(f"student-state manifest is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"student-state manifest is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise FreezeError("student-state manifest must be an object")
    if (
        value.get("schema") != "token-reconstruction.trr-p04-selected-state-manifest.v1"
        or value.get("status") != "PASS_SELECTED_ONLY_BEFORE_PREDICTION"
        or value.get("task_id") != scorer.TASK_ID
        or value.get("truth_accessed") is not False
        or value.get("evaluation_state_count") != len(METHODS) * len(SEEDS)
    ):
        raise FreezeError("student-state manifest provenance/status changed")
    rows = value.get("states")
    required = {(method, seed) for method in METHODS for seed in SEEDS}
    if not isinstance(rows, list) or len(rows) != len(required):
        raise FreezeError("student-state manifest must bind all eight method/seed states")
    descriptors: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FreezeError(f"student-state descriptor {index} is malformed")
        method = row.get("method_id")
        seed = row.get("seed")
        if method not in METHODS or seed not in SEEDS:
            raise FreezeError(f"student-state descriptor {index} has unexpected identity")
        identity = (str(method), int(seed))
        if identity in seen:
            raise FreezeError("student-state manifest has duplicate identities")
        seen.add(identity)
        state_path_value = row.get("path")
        if not isinstance(state_path_value, str) or not state_path_value:
            raise FreezeError(f"student-state descriptor {index} has no path")
        state_path = Path(state_path_value).expanduser()
        if not state_path.is_absolute():
            state_path = path.parent / state_path
        state_path = state_path.resolve()
        descriptor = {
            "method_id": str(method),
            "seed": int(seed),
            "path": str(state_path),
            "bytes": int(row.get("bytes", -1)),
            "sha256": row.get("sha256"),
            "selected_step": row.get("selected_step"),
        }
        actual = _prediction_descriptor(state_path)
        if descriptor["bytes"] != actual["bytes"] or descriptor["sha256"] != actual["sha256"]:
            raise FreezeError(f"student-state manifest hash or size changed: {state_path}")
        descriptors.append(descriptor)
    if seen != required:
        raise FreezeError("student-state manifest is incomplete")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}, {"states": descriptors}



STUDENT_FREEZE_SCHEMA = "token-reconstruction.trr-p04-student-prediction-freeze.v1"
OBSERVATION_INDEX_SCHEMA = "token-reconstruction.trr-p04-evaluator-observations.v1-index.v1"
NATIVE_ANCHOR_RECEIPT_SCHEMA = "token-reconstruction.trr-p04-native-anchor.v1-receipt.v1"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
ANCHOR_IMPLEMENTATION = "frozen_a1_a2_k256"


def _resolved_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FreezeError(f"{label} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _record_order_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise FreezeError(f"record {index} has no record_id")
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _panel_descriptor(path: Path, panel: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    rows = list(panel.values())
    anchors = [row for row in rows if bool(row["anchor"])]
    ordered = [{"record_id": row["record_id"]} for row in rows]
    anchor_ordered = [{"record_id": row["record_id"]} for row in anchors]
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "record_count": len(rows),
        "record_order_sha256": _record_order_sha256(ordered),
        "anchor_count": len(anchors),
        "anchor_order_sha256": _record_order_sha256(anchor_ordered),
    }


def _verify_descriptor_binding(
    declared: Any,
    actual: Mapping[str, Any],
    *,
    label: str,
    base: Path,
    fields: Sequence[str] = (),
) -> None:
    if not isinstance(declared, Mapping):
        raise FreezeError(f"{label} descriptor is absent or malformed")
    declared_path = _resolved_path(declared.get("path"), base=base, label=label)
    actual_path = _resolved_path(actual.get("path"), base=Path.cwd(), label=f"actual {label}")
    if declared_path != actual_path:
        raise FreezeError(f"{label} path binding changed")
    # Some existing receipts intentionally omit the redundant byte count, but
    # every binding must still carry the immutable content hash.
    if declared.get("sha256") != actual.get("sha256") or (
        "bytes" in declared and declared.get("bytes") != actual.get("bytes")
    ):
        raise FreezeError(f"{label} hash or size binding changed")
    for field in fields:
        if declared.get(field) != actual.get(field):
            raise FreezeError(f"{label} {field} binding changed")


def _verify_panel_binding(declared: Any, panel_descriptor: Mapping[str, Any], *, label: str, base: Path) -> None:
    _verify_descriptor_binding(declared, panel_descriptor, label=label, base=base)
    for field in ("record_count", "record_order_sha256", "anchor_count", "anchor_order_sha256"):
        if declared.get(field) != panel_descriptor.get(field):
            raise FreezeError(f"{label} {field} binding changed")


def _require_false_access(value: Any, *, label: str, fields: Sequence[str]) -> None:
    if not isinstance(value, Mapping):
        raise FreezeError(f"{label} access metadata is absent")
    for field in fields:
        if value.get(field) is not False:
            raise FreezeError(f"{label} access flag changed: {field}")


def _require_git_commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise FreezeError(f"{label} source commit is absent or malformed")
    return value


def _validate_observation_index_binding(
    path: Path,
    *,
    panel: Mapping[str, Mapping[str, Any]],
    panel_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    actual = _prediction_descriptor(path)
    index = _load_json(path, "evaluator observation index")
    if index.get("schema") != OBSERVATION_INDEX_SCHEMA or index.get("task_id") != scorer.TASK_ID:
        raise FreezeError("evaluator observation index identity changed")
    if index.get("status") != "EVALUATOR_OBSERVATION_INDEX_READY_NO_TRUTH":
        raise FreezeError("evaluator observation index is not truth-free and ready")
    if index.get("serialized_source_or_truth") is not False or index.get("serialized_token_ids") is not False:
        raise FreezeError("evaluator observation index exposes source or truth")
    rows = index.get("records")
    if not isinstance(rows, list) or len(rows) != len(panel):
        raise FreezeError("evaluator observation index record count changed")
    expected_rows = list(panel.values())
    for position, (expected, actual_row) in enumerate(zip(expected_rows, rows)):
        if not isinstance(actual_row, Mapping):
            raise FreezeError(f"evaluator observation index row {position} is malformed")
        if (
            actual_row.get("record_id") != expected["record_id"]
            or actual_row.get("style") != expected["style"]
            or actual_row.get("length_stratum") != expected["length_stratum"]
            or bool(actual_row.get("anchor")) != bool(expected["anchor"])
            or actual_row.get("active_token_count") != int(expected["length_stratum"]) + 1
            or actual_row.get("padded_tokens") != 192
        ):
            raise FreezeError(f"evaluator observation index geometry/order changed at row {position}")
    if index.get("record_order_sha256") != panel_descriptor["record_order_sha256"]:
        raise FreezeError("evaluator observation index record order changed")
    if index.get("conditions") != list(CONDITIONS):
        raise FreezeError("evaluator observation index condition order changed")
    _verify_panel_binding(index.get("selection"), panel_descriptor, label="observation index selection", base=path.parent)
    return {
        **actual,
        "record_count": len(rows),
        "record_order_sha256": index["record_order_sha256"],
        "conditions": list(CONDITIONS),
    }


def _tensor_sha256(value: Any) -> str:
    """Match the repository tensor digest without serializing observation data."""

    import torch

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _observation_tensor_digests(path: Path) -> dict[str, str]:
    """Read only the permitted observation tensors needed for geometry binding."""

    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"activations", "attention_mask", "position_ids"}
            if not required.issubset(keys) or {"token_ids", "input_ids", "labels", "truth"} & keys:
                raise FreezeError(f"observation tensor keys are not truth-free: {path}")
            return {
                "activations_sha256": _tensor_sha256(handle.get_tensor("activations")),
                "attention_mask_sha256": _tensor_sha256(handle.get_tensor("attention_mask")),
                "position_ids_sha256": _tensor_sha256(handle.get_tensor("position_ids")),
            }
    except FreezeError:
        raise
    except Exception as exc:
        raise FreezeError(f"cannot inspect permitted observation tensors: {path}") from exc


def _validate_observation_descriptors(
    student: Mapping[str, Any],
    *,
    panel_descriptor: Mapping[str, Any],
    observation_index_descriptor: Mapping[str, Any],
    base: Path,
) -> dict[str, Mapping[str, Any]]:
    observations = student.get("observations")
    if not isinstance(observations, Mapping) or set(observations) != set(CONDITIONS):
        raise FreezeError("student freeze must bind both condition observations")
    student_index = student.get("observation_index")
    _verify_descriptor_binding(
        student_index,
        observation_index_descriptor,
        label="student observation index",
        base=base,
        fields=("record_count", "record_order_sha256"),
    )
    validated: dict[str, Mapping[str, Any]] = {}
    shared_mask: str | None = None
    shared_positions: str | None = None
    for condition in CONDITIONS:
        descriptor = observations[condition]
        observation_path = _resolved_path(descriptor.get("path"), base=base, label=f"student observation {condition}")
        actual = _prediction_descriptor(observation_path)
        _verify_descriptor_binding(descriptor, actual, label=f"student observation {condition}", base=base)
        if descriptor.get("condition") != condition or descriptor.get("shape") != [72, 192, 2048]:
            raise FreezeError(f"student observation {condition} geometry/condition changed")
        if descriptor.get("dtype") != "torch.bfloat16":
            raise FreezeError(f"student observation {condition} dtype changed")
        tensor_digests = _observation_tensor_digests(observation_path)
        for field, expected_digest in tensor_digests.items():
            if descriptor.get(field) != expected_digest:
                raise FreezeError(f"student observation {condition} {field} changed")
        if shared_mask is None:
            shared_mask = str(descriptor.get("attention_mask_sha256"))
            shared_positions = str(descriptor.get("position_ids_sha256"))
        elif descriptor.get("attention_mask_sha256") != shared_mask or descriptor.get("position_ids_sha256") != shared_positions:
            raise FreezeError("paired observations do not share mask/position geometry")
        validated[condition] = descriptor
    return validated


def _validate_student_freeze_provenance(
    path: Path,
    *,
    panel: Mapping[str, Mapping[str, Any]],
    panel_descriptor: Mapping[str, Any],
    observation_index_descriptor: Mapping[str, Any],
    state_manifest_descriptor: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    prediction_descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    receipt_descriptor = _prediction_descriptor(path)
    student = _load_json(path, "student prediction freeze")
    if student.get("schema") != STUDENT_FREEZE_SCHEMA or student.get("task_id") != scorer.TASK_ID:
        raise FreezeError("student prediction freeze identity changed")
    if student.get("status") != "STUDENT_PREDICTIONS_FROZEN_BEFORE_JOINT_FREEZE":
        raise FreezeError("student prediction freeze is not the expected pre-joint receipt")
    if student.get("truth_accessed") is not False or student.get("evaluation_truth_opened") is not False:
        raise FreezeError("student prediction freeze truth flags changed")
    if student.get("prediction_files_rewritten") is not False:
        raise FreezeError("student prediction freeze permits prediction rewriting")
    if student.get("all_predictions_repeat_exact") is not True or student.get("all_anchor_slices_exact") is not True:
        raise FreezeError("student prediction freeze repeat/anchor checks did not pass")
    _require_false_access(
        student.get("access"),
        label="student prediction freeze",
        fields=("uses_source_tokens", "uses_public_prefix", "uses_target_update_weights", "uses_teacher_or_candidates", "uses_evaluation_truth"),
    )
    _verify_panel_binding(student.get("selection"), panel_descriptor, label="student selection", base=path.parent)
    _verify_descriptor_binding(
        student.get("state_manifest"),
        state_manifest_descriptor,
        label="student state manifest",
        base=path.parent,
    )
    observations = _validate_observation_descriptors(
        student,
        panel_descriptor=panel_descriptor,
        observation_index_descriptor=observation_index_descriptor,
        base=path.parent,
    )
    required = [
        {"method_id": method, "seed": seed, "condition": condition, "anchor": False}
        for condition in CONDITIONS
        for seed in SEEDS
        for method in METHODS
    ]
    raw_required = student.get("required_student_groups")
    if not isinstance(raw_required, list) or {
        (row.get("method_id"), row.get("seed"), row.get("condition"), bool(row.get("anchor", False)))
        for row in raw_required
        if isinstance(row, Mapping)
    } != {(row["method_id"], row["seed"], row["condition"], row["anchor"]) for row in required} or len(raw_required) != 16:
        raise FreezeError("student prediction freeze group matrix changed")
    student_descriptors = student.get("prediction_files")
    if not isinstance(student_descriptors, list) or len(student_descriptors) != 16:
        raise FreezeError("student prediction freeze prediction-file bindings are incomplete")
    input_by_path = {Path(str(descriptor["path"])).expanduser().resolve(): descriptor for descriptor in prediction_descriptors}
    freeze_by_path: dict[Path, Mapping[str, Any]] = {}
    for index, descriptor in enumerate(student_descriptors):
        descriptor_path = _resolved_path(descriptor.get("path"), base=path.parent, label=f"student prediction {index}")
        actual = _prediction_descriptor(descriptor_path)
        _verify_descriptor_binding(descriptor, actual, label=f"student prediction {index}", base=path.parent)
        if descriptor_path in freeze_by_path:
            raise FreezeError("student prediction freeze prediction files are duplicated")
        freeze_by_path[descriptor_path] = descriptor
    if set(freeze_by_path) != set(input_by_path):
        raise FreezeError("student prediction files do not match the student freeze")
    state_by_identity = {(row["method_id"], row["seed"]): row for row in state_rows}
    expected_keys = {(method, seed, condition) for condition in CONDITIONS for seed in SEEDS for method in METHODS}
    cells = student.get("cells")
    if not isinstance(cells, list) or len(cells) != 16:
        raise FreezeError("student freeze cell receipt matrix is incomplete")
    seen: set[tuple[str, int, str]] = set()
    cell_descriptors: list[dict[str, Any]] = []
    for outer in cells:
        if not isinstance(outer, Mapping):
            raise FreezeError("student freeze cell descriptor is malformed")
        key = (outer.get("method_id"), outer.get("seed"), outer.get("condition"))
        if key not in expected_keys or key in seen:
            raise FreezeError(f"student freeze cell identity is unexpected or duplicated: {key}")
        seen.add(key)
        if outer.get("repeated_prediction_exact") is not True or outer.get("anchor_full_panel_slice_exact") is not True:
            raise FreezeError(f"student freeze cell repeat/anchor check failed: {key}")
        cell_path = _resolved_path(outer.get("cell_receipt", {}).get("path") if isinstance(outer.get("cell_receipt"), Mapping) else None, base=path.parent, label=f"student cell {key}")
        cell_actual = _prediction_descriptor(cell_path)
        _verify_descriptor_binding(outer.get("cell_receipt"), cell_actual, label=f"student cell {key}", base=path.parent)
        cell = _load_json(cell_path, f"student cell receipt {key}")
        if cell.get("schema") != "token-reconstruction.trr-p04-student-prediction-cell.v1" or cell.get("status") != "PASS":
            raise FreezeError(f"student cell receipt status/schema changed: {key}")
        if cell.get("task_id") != scorer.TASK_ID or cell.get("method_id") != key[0] or cell.get("seed") != key[1] or cell.get("condition") != key[2]:
            raise FreezeError(f"student cell receipt identity changed: {key}")
        if cell.get("full_vocabulary") is not True:
            raise FreezeError(f"student cell is not full-vocabulary: {key}")
        _require_false_access(
            cell,
            label=f"student cell {key}",
            fields=("uses_source_tokens", "uses_public_prefix", "uses_target_update_weights", "uses_teacher_or_candidates", "uses_evaluation_truth"),
        )
        _verify_panel_binding(cell.get("selection"), panel_descriptor, label=f"student cell selection {key}", base=cell_path.parent)
        expected_observation = observations[key[2]]
        observed = cell.get("observation")
        if not isinstance(observed, Mapping):
            raise FreezeError(f"student cell observation binding is absent: {key}")
        for field in (
            "path", "bytes", "sha256", "condition", "shape", "dtype",
            "activations_sha256", "attention_mask_sha256", "position_ids_sha256",
        ):
            if observed.get(field) != expected_observation.get(field):
                raise FreezeError(f"student cell observation binding changed ({field}): {key}")
        if observed.get("post_bos_positions") != 4320 or observed.get("record_order_sha256") != panel_descriptor["record_order_sha256"]:
            raise FreezeError(f"student cell observation geometry/order changed: {key}")
        state = cell.get("state")
        state_row = state_by_identity.get((key[0], key[1]))
        if state_row is None or not isinstance(state, Mapping):
            raise FreezeError(f"student cell state binding is absent: {key}")
        for field in ("method_id", "seed", "path", "bytes", "sha256", "selected_step"):
            if state.get(field) != state_row.get(field):
                raise FreezeError(f"student cell state binding changed ({field}): {key}")
        prediction = cell.get("prediction")
        outer_prediction = outer.get("prediction")
        if not isinstance(prediction, Mapping) or not isinstance(outer_prediction, Mapping):
            raise FreezeError(f"student cell prediction binding is absent: {key}")
        prediction_path = _resolved_path(prediction.get("path"), base=cell_path.parent, label=f"student cell prediction {key}")
        prediction_actual = _prediction_descriptor(prediction_path)
        _verify_descriptor_binding(prediction, prediction_actual, label=f"student cell prediction {key}", base=cell_path.parent)
        _verify_descriptor_binding(outer_prediction, prediction_actual, label=f"student outer prediction {key}", base=path.parent)
        if prediction_path not in input_by_path or prediction_path not in freeze_by_path:
            raise FreezeError(f"student cell prediction is not bound by the student freeze: {key}")
        if prediction.get("rows") != 72 or prediction.get("post_bos_positions") != 4320:
            raise FreezeError(f"student cell prediction geometry changed: {key}")
        for artifact_name in ("tie_diagnostics", "timing"):
            outer_artifact = outer.get(artifact_name)
            cell_artifact = cell.get(artifact_name)
            if not isinstance(outer_artifact, Mapping) or not isinstance(cell_artifact, Mapping):
                raise FreezeError(f"student cell {artifact_name} binding is absent: {key}")
            artifact_path = _resolved_path(cell_artifact.get("path"), base=cell_path.parent, label=f"student {artifact_name} {key}")
            artifact_actual = _prediction_descriptor(artifact_path)
            _verify_descriptor_binding(cell_artifact, artifact_actual, label=f"student {artifact_name} {key}", base=cell_path.parent)
            _verify_descriptor_binding(outer_artifact, artifact_actual, label=f"student outer {artifact_name} {key}", base=path.parent)
        timing_payload = cell.get("timing_payload")
        if not isinstance(timing_payload, Mapping) or timing_payload.get("status") != "PASS" or timing_payload.get("truth_accessed") is not False:
            raise FreezeError(f"student timing payload status/truth changed: {key}")
        geometry = timing_payload.get("geometry")
        if geometry != {
            "full_records": 72,
            "full_tensor_shape": [72, 192],
            "full_scored_positions": 4320,
            "anchor_records": 12,
            "anchor_tensor_shape": [12, 192],
            "anchor_scored_positions": 384,
            "record_batch_size": 8,
            "projection_chunk": 512,
            "full_vocabulary": True,
        }:
            raise FreezeError(f"student timing geometry changed: {key}")
        timing_selection = timing_payload.get("selection")
        if (
            not isinstance(timing_selection, Mapping)
            or timing_selection.get("sha256") != panel_descriptor["sha256"]
            or timing_selection.get("record_order_sha256") != panel_descriptor["record_order_sha256"]
        ):
            raise FreezeError(f"student timing selection binding changed: {key}")
        timing_observation = timing_payload.get("observation")
        if not isinstance(timing_observation, Mapping) or any(timing_observation.get(field) != expected_observation.get(field) for field in ("condition", "sha256", "attention_mask_sha256", "position_ids_sha256")):
            raise FreezeError(f"student timing observation binding changed: {key}")
        if timing_observation.get("record_order_sha256") != panel_descriptor["record_order_sha256"] or timing_observation.get("post_bos_positions") != 4320:
            raise FreezeError(f"student timing observation geometry/order changed: {key}")
        _require_false_access(
            timing_payload.get("access"),
            label=f"student timing access {key}",
            fields=("uses_source_tokens", "uses_public_prefix", "uses_target_update_weights", "uses_teacher_or_candidates", "uses_evaluation_truth"),
        )
        binding = timing_payload.get("anchor_subset", {}).get("binding") if isinstance(timing_payload.get("anchor_subset"), Mapping) else None
        if not isinstance(binding, Mapping) or binding.get("record_count") != 12 or binding.get("post_bos_positions") != 384 or binding.get("record_ids") != [row["record_id"] for row in panel.values() if row["anchor"]] or binding.get("record_order_sha256") != panel_descriptor["anchor_order_sha256"] or binding.get("full_panel_slice_exact") is not True or binding.get("anchor_output_exact") is not True:
            raise FreezeError(f"student timing anchor binding changed: {key}")
        cell_descriptors.append(dict(outer["cell_receipt"]))
    if seen != expected_keys:
        raise FreezeError("student freeze cell matrix does not cover all method/seed/condition identities")
    execution = student.get("execution")
    implementation_commit = execution.get("implementation_commit") if isinstance(execution, Mapping) else None
    implementation_commit = _require_git_commit(implementation_commit, label="student freeze implementation")
    return {
        "receipt": receipt_descriptor,
        "cell_receipts": cell_descriptors,
        "observations": dict(observations),
        "implementation_commit": implementation_commit,
    }


def _validate_native_anchor_receipts(
    paths: Sequence[Path],
    *,
    panel_descriptor: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    observation_index_descriptor: Mapping[str, Any],
    anchor_prediction_descriptors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(paths) != len(CONDITIONS):
        raise FreezeError("joint freeze requires one native anchor receipt per target condition")
    prediction_by_path = {Path(str(row["path"])).expanduser().resolve(): row for row in anchor_prediction_descriptors}
    seen: set[str] = set()
    compact: list[dict[str, Any]] = []
    for path in paths:
        path = path.expanduser().resolve()
        receipt_descriptor = _prediction_descriptor(path)
        receipt = _load_json(path, "native anchor receipt")
        if receipt.get("schema") != NATIVE_ANCHOR_RECEIPT_SCHEMA or receipt.get("task_id") != scorer.TASK_ID or receipt.get("status") != "PASS_NATIVE_ANCHOR_NO_TRUTH":
            raise FreezeError(f"native anchor receipt status/schema changed: {path}")
        observation = receipt.get("observation")
        metadata = observation.get("metadata") if isinstance(observation, Mapping) else None
        diagnostics = receipt.get("diagnostics")
        target = receipt.get("target")
        candidate_conditions = {
            metadata.get("condition") if isinstance(metadata, Mapping) else None,
            diagnostics.get("condition") if isinstance(diagnostics, Mapping) else None,
            target.get("condition") if isinstance(target, Mapping) else None,
        }
        candidate_conditions.discard(None)
        if len(candidate_conditions) != 1:
            raise FreezeError(f"native anchor receipt condition binding is ambiguous: {path}")
        condition = str(next(iter(candidate_conditions)))
        if condition not in CONDITIONS or condition in seen:
            raise FreezeError(f"native anchor receipt condition is unexpected or duplicated: {condition}")
        seen.add(condition)
        _verify_panel_binding(receipt.get("selection"), panel_descriptor, label=f"native anchor selection {condition}", base=path.parent)
        if not isinstance(observation, Mapping) or not isinstance(metadata, Mapping):
            raise FreezeError(f"native anchor observation binding is absent: {condition}")
        expected_observation = observations[condition]
        observation_path = _resolved_path(observation.get("path"), base=path.parent, label=f"native anchor observation {condition}")
        expected_observation_path = Path(str(expected_observation["path"])).expanduser().resolve()
        if observation_path != expected_observation_path or observation.get("sha256") != expected_observation.get("sha256"):
            raise FreezeError(f"native anchor observation file binding changed: {condition}")
        index_path = _resolved_path(observation.get("index_path"), base=path.parent, label=f"native anchor observation index {condition}")
        if index_path != Path(str(observation_index_descriptor["path"])).expanduser().resolve() or observation.get("index_sha256") != observation_index_descriptor.get("sha256"):
            raise FreezeError(f"native anchor observation index binding changed: {condition}")
        for field in ("condition", "selection_sha256", "record_order_sha256", "activations_sha256", "attention_mask_sha256", "position_ids_sha256"):
            expected_field = {
                "condition": condition,
                "selection_sha256": panel_descriptor["sha256"],
                "record_order_sha256": panel_descriptor["record_order_sha256"],
                "activations_sha256": expected_observation.get("activations_sha256"),
                "attention_mask_sha256": expected_observation.get("attention_mask_sha256"),
                "position_ids_sha256": expected_observation.get("position_ids_sha256"),
            }[field]
            if metadata.get(field) != expected_field:
                raise FreezeError(f"native anchor observation metadata changed ({field}): {condition}")
        if metadata.get("schema") != "token-reconstruction.trr-p04-evaluator-observations.v1" or metadata.get("task_id") != scorer.TASK_ID or metadata.get("source_tokens_serialized") != "false" or metadata.get("evaluation_truth_opened") != "false" or metadata.get("target_update_weights_serialized") != "false":
            raise FreezeError(f"native anchor observation access/schema changed: {condition}")
        target_plan = receipt.get("target_plan")
        if not isinstance(target_plan, Mapping):
            raise FreezeError(f"native anchor target plan binding is absent: {condition}")
        target_plan_path = _resolved_path(target_plan.get("path"), base=path.parent, label=f"native anchor target plan {condition}")
        target_plan_actual = _prediction_descriptor(target_plan_path)
        _verify_descriptor_binding(target_plan, target_plan_actual, label=f"native anchor target plan {condition}", base=path.parent)
        if target_plan.get("condition_id") != "p04_evaluator_target_update_v1" or target_plan.get("lineage_id") != "p04-target-lora-no-robots-v1-seed20260910":
            raise FreezeError(f"native anchor target plan identity changed: {condition}")
        execution = receipt.get("execution")
        execution_commit = _require_git_commit(execution.get("git_commit") if isinstance(execution, Mapping) else None, label=f"native anchor {condition}")
        model = receipt.get("model")
        if not isinstance(model, Mapping) or model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
            raise FreezeError(f"native anchor public model identity changed: {condition}")
        algorithm = receipt.get("algorithm")
        algorithm_key = (
            algorithm.get("method_id"),
            algorithm.get("implementation_identity"),
            algorithm.get("proposal_k"),
            algorithm.get("proposal_chunk"),
            algorithm.get("candidate_k"),
            algorithm.get("record_batch_size"),
            algorithm.get("a2_fallback"),
            algorithm.get("tie_rule"),
        ) if isinstance(algorithm, Mapping) else None
        if algorithm_key != (
            "native_a1_a2", ANCHOR_IMPLEMENTATION, 512, 256, 256, 1, False,
            "published proposal order and first argmax",
        ):
            raise FreezeError(f"native anchor algorithm binding changed: {condition}")
        _require_false_access(
            receipt.get("access"),
            label=f"native anchor {condition}",
            fields=("evaluation_truth_opened", "evaluator_target_update_loaded", "target_update_loaded", "source_rows_read", "source_tokens_read", "student_states_loaded"),
        )
        if receipt.get("access", {}).get("public_reference_loaded") is not True:
            raise FreezeError(f"native anchor public reference was not loaded: {condition}")
        prediction = receipt.get("prediction")
        if not isinstance(prediction, Mapping):
            raise FreezeError(f"native anchor prediction binding is absent: {condition}")
        prediction_path = _resolved_path(prediction.get("path"), base=path.parent, label=f"native anchor prediction {condition}")
        prediction_actual = _prediction_descriptor(prediction_path)
        _verify_descriptor_binding(prediction, prediction_actual, label=f"native anchor prediction {condition}", base=path.parent)
        if prediction_path not in prediction_by_path or prediction.get("rows") != 12 or prediction.get("post_bos_positions") != 384:
            raise FreezeError(f"native anchor prediction is not bound by the joint input: {condition}")
        if receipt.get("algorithm", {}).get("prediction_exact_repeat") is not True:
            raise FreezeError(f"native anchor repeat evidence changed: {condition}")
        resources = receipt.get("resources")
        if not isinstance(resources, Mapping):
            raise FreezeError(f"native anchor public source resources are absent: {condition}")
        for role in ("reference", "lens", "legacy_proposal_decode", "policy"):
            resource = resources.get(role)
            if not isinstance(resource, Mapping):
                raise FreezeError(f"native anchor resource binding is absent ({role}): {condition}")
            resource_path = _resolved_path(resource.get("path"), base=path.parent, label=f"native anchor resource {condition}/{role}")
            resource_actual = _prediction_descriptor(resource_path)
            _verify_descriptor_binding(resource, resource_actual, label=f"native anchor resource {condition}/{role}", base=path.parent)
        compact.append(
            {
                "condition": condition,
                "receipt": receipt_descriptor,
                "prediction": dict(prediction),
                "target_plan": {
                    "path": str(target_plan_path),
                    "sha256": target_plan_actual["sha256"],
                    "condition_id": target_plan["condition_id"],
                    "lineage_id": target_plan["lineage_id"],
                },
                "source_commit": execution_commit,
                "observation": {
                    "path": observation.get("path"),
                    "sha256": observation.get("sha256"),
                    "index_path": observation.get("index_path"),
                    "index_sha256": observation.get("index_sha256"),
                    "metadata": {
                        "condition": condition,
                        "selection_sha256": metadata.get("selection_sha256"),
                        "record_order_sha256": metadata.get("record_order_sha256"),
                        "activations_sha256": metadata.get("activations_sha256"),
                        "attention_mask_sha256": metadata.get("attention_mask_sha256"),
                        "position_ids_sha256": metadata.get("position_ids_sha256"),
                    },
                },
                "model": {"id": model["id"], "revision": model["revision"], "snapshot": model.get("snapshot")},
                "algorithm": {
                    "method_id": algorithm["method_id"],
                    "implementation_identity": algorithm["implementation_identity"],
                    "proposal_k": algorithm["proposal_k"],
                    "proposal_chunk": algorithm["proposal_chunk"],
                    "candidate_k": algorithm["candidate_k"],
                    "record_batch_size": algorithm["record_batch_size"],
                    "a2_fallback": algorithm["a2_fallback"],
                    "tie_rule": algorithm["tie_rule"],
                },
                "access": {
                    "evaluation_truth_opened": False,
                    "evaluator_target_update_loaded": False,
                    "target_update_loaded": False,
                    "source_rows_read": False,
                    "source_tokens_read": False,
                    "student_states_loaded": False,
                },
            }
        )
    if seen != set(CONDITIONS):
        raise FreezeError("native anchor receipts do not cover both conditions")
    return sorted(compact, key=lambda row: row["condition"])

def build_freeze(
    *,
    panel_path: Path,
    prediction_paths: Sequence[Path],
    anchor_prediction_paths: Sequence[Path],
    state_manifest_path: Path,
    truth_dir: Path,
    output_path: Path,
    argv: Sequence[str],
    student_prediction_freeze_path: Path | None = None,
    observation_index_path: Path | None = None,
    native_anchor_receipt_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    panel = scorer._load_json(panel_path, description="P04 public panel")
    panel_records = scorer._validate_panel(panel)
    expected = _expected_groups()
    if student_prediction_freeze_path is None or observation_index_path is None or native_anchor_receipt_paths is None:
        raise FreezeError(
            "student prediction freeze, observation index, and native anchor receipts are required"
        )
    if not prediction_paths or not anchor_prediction_paths:
        raise FreezeError("both student/reference and native A1+A2 anchor prediction files are required")
    if len(native_anchor_receipt_paths) != len(CONDITIONS):
        raise FreezeError("exactly one native anchor receipt per target condition is required")
    student_prediction_paths = [path.expanduser().resolve() for path in prediction_paths]
    anchor_prediction_paths_resolved = [path.expanduser().resolve() for path in anchor_prediction_paths]
    all_prediction_paths = [*student_prediction_paths, *anchor_prediction_paths_resolved]
    descriptors = [_prediction_descriptor(path) for path in all_prediction_paths]
    _read_prediction_groups(all_prediction_paths, panel=panel_records, expected=expected)
    state_manifest_descriptor, state_payload = _state_descriptors(state_manifest_path.expanduser().resolve())
    panel_descriptor = _panel_descriptor(panel_path, panel_records)
    observation_index_descriptor = _validate_observation_index_binding(
        observation_index_path,
        panel=panel_records,
        panel_descriptor=panel_descriptor,
    )
    student_prediction_descriptor = [_prediction_descriptor(path) for path in student_prediction_paths]
    anchor_prediction_descriptor = [_prediction_descriptor(path) for path in anchor_prediction_paths_resolved]
    student_provenance = _validate_student_freeze_provenance(
        student_prediction_freeze_path,
        panel=panel_records,
        panel_descriptor=panel_descriptor,
        observation_index_descriptor=observation_index_descriptor,
        state_manifest_descriptor=state_manifest_descriptor,
        state_rows=state_payload["states"],
        prediction_descriptors=student_prediction_descriptor,
    )
    native_anchor_provenance = _validate_native_anchor_receipts(
        [path.expanduser().resolve() for path in native_anchor_receipt_paths],
        panel_descriptor=panel_descriptor,
        observations=student_provenance["observations"],
        observation_index_descriptor=observation_index_descriptor,
        anchor_prediction_descriptors=anchor_prediction_descriptor,
    )
    panel_sha = panel_descriptor["sha256"]
    freeze = {
        "schema": FREEZE_SCHEMA,
        "task_id": scorer.TASK_ID,
        "status": "FROZEN_BEFORE_TRUTH",
        "created_utc": _utc_now(),
        "panel_frozen": True,
        "predictions_frozen": True,
        "all_states_frozen": True,
        "truth_open_allowed": True,
        "truth_accessed": False,
        "panel": {
            "path": str(panel_path),
            "bytes": panel_path.stat().st_size,
            "sha256": panel_sha,
        },
        "prediction_files": descriptors,
        "state_manifest": state_manifest_descriptor,
        "state_files": state_payload["states"],
        "student_prediction_freeze": student_provenance["receipt"],
        "student_cell_receipts": student_provenance["cell_receipts"],
        "observation_index": observation_index_descriptor,
        "observations": student_provenance["observations"],
        "native_anchor_receipts": native_anchor_provenance,
        "provenance": {
            "status": "JOINT_PUBLIC_BINDINGS_VALIDATED_BEFORE_TRUTH",
            "student_prediction_freeze": student_provenance["receipt"],
            "student_cell_receipt_count": len(student_provenance["cell_receipts"]),
            "student_implementation_commit": student_provenance["implementation_commit"],
            "observation_index": observation_index_descriptor,
            "observation_conditions": list(CONDITIONS),
            "native_anchor_receipt_count": len(native_anchor_provenance),
        },
        "prediction_groups": [
            {
                "method_id": method,
                "seed": seed,
                "condition": condition,
                "anchor": anchor,
            }
            for method, seed, condition, anchor in expected
        ],
        "truth_files": [
            {
                "condition": condition,
                "path": str((truth_dir / f"{condition}.jsonl").resolve()),
                "content_hash_recorded_after_gate": True,
            }
            for condition in CONDITIONS
        ],
        "truth_boundary": {
            "prediction_and_panel_validation_completed_before_truth": True,
            "truth_rows_not_loaded_by_freezer": True,
            "student_inference_is_activation_only": True,
        },
        "execution": {
            "argv": list(argv),
            "python": sys.executable,
            "started_utc": None,
            "ended_utc": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "max_rss_bytes": _max_rss_bytes(),
            "truth_accessed": False,
            "model_loaded": False,
        },
    }
    freeze["execution"]["started_utc"] = freeze["created_utc"]
    if output_path.exists() or output_path.is_symlink():
        raise FreezeError(f"refusing to overwrite freeze receipt: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return freeze


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path, action="append", required=True)
    parser.add_argument("--anchor-prediction-file", type=Path, action="append", required=True)
    parser.add_argument("--state-manifest", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--student-prediction-freeze", type=Path, required=True)
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--native-anchor-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        build_freeze(
            panel_path=args.panel.expanduser().resolve(),
            prediction_paths=[path.expanduser().resolve() for path in args.prediction_file],
            anchor_prediction_paths=[path.expanduser().resolve() for path in args.anchor_prediction_file],
            state_manifest_path=args.state_manifest.expanduser().resolve(),
            truth_dir=args.truth_dir,
            output_path=args.output.expanduser().resolve(),
            argv=list(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]),
            student_prediction_freeze_path=args.student_prediction_freeze.expanduser().resolve(),
            observation_index_path=args.observation_index.expanduser().resolve(),
            native_anchor_receipt_paths=[path.expanduser().resolve() for path in args.native_anchor_receipt],
        )
    except FreezeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"output": str(args.output.resolve()), "status": "FROZEN_BEFORE_TRUTH"}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

