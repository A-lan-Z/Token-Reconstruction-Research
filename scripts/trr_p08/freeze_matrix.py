#!/usr/bin/env python3
"""Metadata-only TRR-P08 final-model/prediction freeze gate.

The helper validates the completed public fits, selected state files, frozen
source-free observation manifest, and all prediction/timing descriptors before
truth is opened.  It never loads a safetensor, prediction array, observation
payload, or truth file.  ``assemble_joint_freeze`` emits one create-only
receipt that a later scorer can require verbatim.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

TASK_ID = "TRR-P08"
APPROVED_PLAN_SHA256 = "9d1eb8dca89c76f4064c0c380636cb9d585672f7fae796d5295d6507b788caa3"
FIT_SCHEMA = "token-reconstruction.trr-p08-staged-fit.v1"
QUALIFICATION_SCHEMA = "token-reconstruction.trr-p08-staged-fit-qualification.v1"
OBSERVATION_SCHEMAS = {
    "token-reconstruction.trr-p08-public-observation-manifest.v1",
    "token-reconstruction.trr-p08-observation-manifest.v1",
}
PREDICTION_SCHEMAS = {
    "token-reconstruction.trr-p08-prediction-manifest.v1",
}
FREEZE_SCHEMA = "token-reconstruction.trr-p08-joint-freeze.v1"
DOMAINS = ("pile", "finance")
TARGETS = ("public_base", "public_lora_2601")
CELL_ORDER = tuple(f"{domain}__{target}" for domain in DOMAINS for target in TARGETS)
METHOD_ORDER = (
    "p08_positionwise_joint",
    "p08_positionwise_staged",
    "p08_past_only_joint",
    "p08_past_only_staged",
)
SEEDS = (6106, 6107)
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
HIDDEN_SIZE = 2048
RECORDS_PER_DOMAIN = 256
PREDICTION_COUNT = len(CELL_ORDER) * len(SEEDS) * len(METHOD_ORDER)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class P08FreezeError(RuntimeError):
    """Raised when the P08 final freeze contract fails closed."""


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise P08FreezeError(f"file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: Any, *, root: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise P08FreezeError(f"{description} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise P08FreezeError(f"{description} is unavailable: {path}")
    return path


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    path = path.resolve()
    digest = _sha256_file(path)
    try:
        display = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        display = str(path)
    return {"path": display, "bytes": int(path.stat().st_size), "sha256": digest}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P08FreezeError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise P08FreezeError(f"{description} must be an object")
    return dict(value)


def _load_record(value: Any, *, root: Path, description: str, hash_file: bool = True) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P08FreezeError(f"{description} file record is malformed")
    size = value.get("bytes")
    digest = value.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise P08FreezeError(f"{description} byte binding is malformed")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise P08FreezeError(f"{description} SHA-256 binding is malformed")
    path = _resolve(value.get("path"), root=root, description=description)
    if int(path.stat().st_size) != int(size):
        raise P08FreezeError(f"{description} byte binding changed")
    if hash_file and _sha256_file(path) != digest:
        raise P08FreezeError(f"{description} hash binding changed")
    return {"path": str(path), "bytes": int(size), "sha256": digest}


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise P08FreezeError(f"create-only freeze output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _same_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("bytes") == right.get("bytes") and left.get("sha256") == right.get("sha256")


def _validate_plan(path: Path, *, root: Path, expected_plan_sha256: str) -> dict[str, Any]:
    actual = _record(path, root=root)
    if actual["sha256"] != expected_plan_sha256:
        raise P08FreezeError(f"P08 plan hash differs: expected {expected_plan_sha256}, got {actual['sha256']}")
    plan = _load_json(path, description="P08 plan")
    if plan.get("task_id") != TASK_ID or plan.get("status") != "FROZEN_DESIGN_PRE_FIT":
        raise P08FreezeError("P08 plan is not the frozen pre-fit design")
    if plan.get("truth_opened") is True or plan.get("fresh_selection_started") is True:
        raise P08FreezeError("P08 plan contains a forbidden post-freeze state")
    return actual


def _validate_approval(path: Path, *, root: Path, expected_plan_sha256: str) -> dict[str, Any]:
    actual = _record(path, root=root)
    approval = _load_json(path, description="P08 root approval")
    if approval.get("task_id") != TASK_ID or approval.get("status") != "ROOT_DESIGN_APPROVED_PRE_FIT":
        raise P08FreezeError("P08 root approval is not pre-fit approval")
    if approval.get("plan_sha256") != expected_plan_sha256:
        raise P08FreezeError("P08 root approval is bound to a different plan")
    return actual


def _validate_fit_receipt(path: Path, *, root: Path) -> dict[str, Any]:
    receipt_record = _record(path, root=root)
    receipt = _load_json(path, description="P08 main-fit receipt")
    if receipt.get("schema") != FIT_SCHEMA or receipt.get("task_id") != TASK_ID or receipt.get("status") != "PASS":
        raise P08FreezeError("P08 main-fit receipt is not a PASS receipt")
    geometry = receipt.get("geometry")
    expected_geometry = {
        "fit": [1200, SEQUENCE_TOKENS, HIDDEN_SIZE],
        "validation": [48, SEQUENCE_TOKENS, HIDDEN_SIZE],
        "record_batch_size": 8,
        "position_budget": 512,
        "steps": 3000,
        "validation_every": 100,
    }
    if not isinstance(geometry, Mapping) or any(geometry.get(key) != value for key, value in expected_geometry.items()):
        raise P08FreezeError("P08 main-fit geometry or schedule changed")
    if receipt.get("seeds") != list(SEEDS):
        raise P08FreezeError("P08 fit seed order changed")
    initialization = receipt.get("initialization")
    if not isinstance(initialization, Mapping) or initialization.get("W") != "identity" or initialization.get("b") != "zeros" or initialization.get("s") != 3.0:
        raise P08FreezeError("P08 standard initialization binding changed")
    runtime = receipt.get("runtime_components")
    if not isinstance(runtime, Mapping) or runtime.get("source_token_access") is not False or runtime.get("target_truth_access") is not False or runtime.get("candidate_simulations") not in (0, False) or runtime.get("a2_student") is not False:
        raise P08FreezeError("P08 fit receipt has forbidden runtime access")
    source_commit = receipt.get("source_commit")
    if not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None:
        raise P08FreezeError("P08 fit source commit is not a full hash")
    if not isinstance(receipt.get("command"), list) or not receipt["command"] or any(not isinstance(value, str) for value in receipt["command"]):
        raise P08FreezeError("P08 fit command binding is missing")
    qualification = receipt.get("qualification_receipt")
    if not isinstance(qualification, Mapping) or qualification.get("status") != "PASS":
        raise P08FreezeError("P08 qualification binding is missing")
    qualification_path = _resolve(qualification.get("path"), root=root, description="P08 qualification receipt")
    # The fit writer records qualification path+SHA but does not duplicate its
    # byte count; derive the immutable file record here before reading JSON.
    qualification_actual = _record(qualification_path, root=root)
    if qualification_actual["sha256"] != qualification.get("sha256"):
        raise P08FreezeError("P08 qualification receipt hash changed")
    qualification_json = _load_json(qualification_path, description="P08 qualification receipt")
    if not isinstance(qualification_json.get("command"), list) or not qualification_json["command"] or any(not isinstance(value, str) for value in qualification_json["command"]):
        raise P08FreezeError("P08 qualification command binding is missing")
    if qualification_json.get("schema") != QUALIFICATION_SCHEMA or qualification_json.get("task_id") != TASK_ID or qualification_json.get("status") != "PASS" or qualification_json.get("states_retained") is not False:
        raise P08FreezeError("P08 qualification is not an accepted disposable PASS")
    methods = receipt.get("methods")
    if not isinstance(methods, list) or len(methods) != len(SEEDS) * len(METHOD_ORDER):
        raise P08FreezeError("P08 selected-state matrix is incomplete")
    states: dict[tuple[int, str], dict[str, Any]] = {}
    for row in methods:
        if not isinstance(row, Mapping):
            raise P08FreezeError("P08 method descriptor is malformed")
        seed, method = row.get("seed"), row.get("arm_id")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS or method not in METHOD_ORDER:
            raise P08FreezeError("P08 method seed/arm identity changed")
        key = (seed, str(method))
        if key in states or row.get("status") != "PASS" or row.get("steps") != 3000 or row.get("final_step") != 3000:
            raise P08FreezeError(f"P08 method is incomplete: {key}")
        state = row.get("state")
        final_state = row.get("final_state")
        if not isinstance(state, Mapping) or not isinstance(final_state, Mapping):
            raise P08FreezeError(f"P08 selected/final state is missing: {key}")
        selected_actual = _load_record(state, root=root, description=f"P08 selected state {key}")
        final_actual = _load_record(final_state, root=root, description=f"P08 final state {key}")
        for descriptor, actual, label in ((state, selected_actual, "selected"), (final_state, final_actual, "final")):
            tensor_sha = descriptor.get("state_sha256")
            if not isinstance(tensor_sha, str) or _SHA256.fullmatch(tensor_sha) is None:
                raise P08FreezeError(f"P08 {label} state tensor digest is missing: {key}")
        states[key] = {
            "arm_id": str(method),
            "seed": seed,
            "selected_step": int(row.get("selected_step", state.get("selected_step", -1))),
            "selected": selected_actual,
            "selected_tensor_sha256": str(state["state_sha256"]),
            "final": final_actual,
            "final_tensor_sha256": str(final_state["state_sha256"]),
            "schedule_sha256": row.get("schedule_sha256"),
            "row": dict(row),
        }
    if set(states) != {(seed, method) for seed in SEEDS for method in METHOD_ORDER}:
        raise P08FreezeError("P08 selected-state matrix has missing entries")
    return {
        "receipt": receipt,
        "receipt_record": receipt_record,
        "source_commit": source_commit,
        "qualification": qualification_actual,
        "states": states,
    }


def _validate_observation_manifest(path: Path, *, root: Path) -> dict[str, Any]:
    manifest_record = _record(path, root=root)
    manifest = _load_json(path, description="P08 observation manifest")
    if manifest.get("schema") not in OBSERVATION_SCHEMAS or manifest.get("task_id") != TASK_ID:
        raise P08FreezeError("P08 observation manifest schema changed")
    if manifest.get("status") not in {"FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH", "FROZEN_P08_OBSERVATIONS_NO_TRUTH"}:
        raise P08FreezeError("P08 observation manifest is not frozen no-truth metadata")
    for flag in ("truth_opened", "source_text_written", "token_ids_written", "target_labels_loaded"):
        if manifest.get(flag) is not False:
            raise P08FreezeError(f"P08 observation manifest has forbidden flag: {flag}")
    if manifest.get("records_per_domain") != RECORDS_PER_DOMAIN or manifest.get("sequence_tokens_including_bos") != SEQUENCE_TOKENS or manifest.get("scored_post_bos_tokens") != SCORED_POST_BOS or manifest.get("hidden_size") != HIDDEN_SIZE:
        raise P08FreezeError("P08 observation geometry changed")
    if manifest.get("cell_order") != list(CELL_ORDER):
        raise P08FreezeError("P08 observation cell order changed")
    pairing = manifest.get("source_pairing")
    if not isinstance(pairing, Mapping) or pairing.get("same_record_ids_across_targets") is not True:
        raise P08FreezeError("P08 observation target pairing is not frozen")
    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or [row.get("cell_id") for row in raw_cells if isinstance(row, Mapping)] != list(CELL_ORDER):
        raise P08FreezeError("P08 observation cells are incomplete or reordered")
    cells: dict[str, Any] = {}
    domain_ids: dict[str, str] = {}
    for row in raw_cells:
        if not isinstance(row, Mapping):
            raise P08FreezeError("P08 observation cell descriptor is malformed")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in CELL_ORDER:
            raise P08FreezeError(f"P08 observation cell is unregistered: {cell_id}")
        domain, target = cell_id.split("__", 1)
        if row.get("domain", row.get("style")) != domain or row.get("target", row.get("condition")) != target:
            raise P08FreezeError(f"P08 observation identity changed: {cell_id}")
        record_ids_sha = row.get("record_ids_sha256")
        if not isinstance(record_ids_sha, str) or _SHA256.fullmatch(record_ids_sha) is None:
            raise P08FreezeError(f"P08 record-order binding missing: {cell_id}")
        if domain in domain_ids and domain_ids[domain] != record_ids_sha:
            raise P08FreezeError(f"P08 target pairing changed: {domain}")
        domain_ids[domain] = record_ids_sha
        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise P08FreezeError(f"P08 observation asset missing: {cell_id}")
        if observation.get("shape") != [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, HIDDEN_SIZE] or observation.get("stored_sequence_tokens") != SEQUENCE_TOKENS or observation.get("scored_post_bos_tokens") != SCORED_POST_BOS:
            raise P08FreezeError(f"P08 observation asset geometry changed: {cell_id}")
        asset = _load_record(observation, root=root, description=f"P08 observation {cell_id}")
        cells[cell_id] = {"domain": domain, "target": target, "record_ids_sha256": record_ids_sha, "asset": asset}
    return {"manifest": manifest, "manifest_record": manifest_record, "cells": cells, "record_ids_sha256": domain_ids}


def _validate_prediction_manifest(path: Path, *, root: Path, fit: Mapping[str, Any], observations: Mapping[str, Any]) -> dict[str, Any]:
    manifest_record = _record(path, root=root)
    manifest = _load_json(path, description="P08 prediction manifest")
    if manifest.get("schema") not in PREDICTION_SCHEMAS or manifest.get("task_id") != TASK_ID:
        raise P08FreezeError("P08 prediction manifest schema changed")
    if manifest.get("status") not in {"FROZEN_P08_PREDICTIONS_NO_TRUTH", "STUDENT_PREDICTIONS_COMPLETE_NO_TRUTH", "P08_STUDENT_PREDICTIONS_COMPLETE_NO_TRUTH"}:
        raise P08FreezeError("P08 predictions are not frozen no-truth metadata")
    for flag in ("truth_opened", "source_text_loaded", "target_labels_loaded", "candidate_arrays_persisted"):
        if manifest.get(flag) is not False:
            raise P08FreezeError(f"P08 prediction manifest has forbidden flag: {flag}")
    code_commit = manifest.get("code_commit")
    if not isinstance(code_commit, str) or _COMMIT.fullmatch(code_commit) is None:
        raise P08FreezeError("P08 prediction source commit is not a full hash")
    command = manifest.get("command", manifest.get("argv"))
    if not isinstance(command, list) or not command or any(not isinstance(value, str) for value in command):
        raise P08FreezeError("P08 prediction command is missing")
    if manifest.get("domains") != list(DOMAINS) or manifest.get("target_conditions") != list(TARGETS) or manifest.get("cell_order") != list(CELL_ORDER) or manifest.get("method_order") != list(METHOD_ORDER) or manifest.get("replicate_seeds") != list(SEEDS):
        raise P08FreezeError("P08 prediction matrix order changed")
    if manifest.get("fit_source_commit") != fit["source_commit"]:
        raise P08FreezeError("P08 prediction fit source binding changed")
    geometry = manifest.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("records_per_domain") != RECORDS_PER_DOMAIN or geometry.get("sequence_tokens") != SEQUENCE_TOKENS or geometry.get("scored_post_bos_tokens") != SCORED_POST_BOS or geometry.get("hidden_size") != HIDDEN_SIZE or geometry.get("batch_records") != 8 or geometry.get("projection_chunk") != 512:
        raise P08FreezeError("P08 prediction geometry changed")
    fit_record = _load_record(manifest.get("fit_receipt"), root=root, description="P08 prediction fit receipt")
    if not _same_record(fit_record, fit["receipt_record"]):
        raise P08FreezeError("P08 prediction fit receipt binding changed")
    observation_record = _load_record(manifest.get("observation_manifest"), root=root, description="P08 prediction observation manifest")
    if not _same_record(observation_record, observations["manifest_record"]):
        raise P08FreezeError("P08 prediction observation binding changed")
    state_bindings = manifest.get("state_bindings")
    if not isinstance(state_bindings, Mapping):
        raise P08FreezeError("P08 prediction state bindings are missing")
    expected_state_keys = {f"{seed}::{method}" for seed in SEEDS for method in METHOD_ORDER}
    if set(state_bindings) != expected_state_keys:
        raise P08FreezeError("P08 prediction state matrix is incomplete")
    for seed in SEEDS:
        for method in METHOD_ORDER:
            state = fit["states"][(seed, method)]
            binding = _load_record(state_bindings[f"{seed}::{method}"], root=root, description=f"P08 prediction state {seed}/{method}")
            if not _same_record(binding, state["selected"]):
                raise P08FreezeError(f"P08 prediction state binding changed: {seed}/{method}")
    raw_cells = manifest.get("student_cells")
    if not isinstance(raw_cells, Mapping) or set(raw_cells) != set(CELL_ORDER):
        raise P08FreezeError("P08 prediction cell matrix is incomplete")
    descriptors: dict[tuple[str, int, str], dict[str, Any]] = {}
    timing_geometry: dict[str, tuple[str, str]] = {}
    for cell_id in CELL_ORDER:
        cell = raw_cells[cell_id]
        if not isinstance(cell, Mapping):
            raise P08FreezeError(f"P08 prediction cell is malformed: {cell_id}")
        observation = observations["cells"][cell_id]
        for raw_seed in SEEDS:
            seed_key = str(raw_seed)
            seed_cell = cell.get(seed_key)
            if not isinstance(seed_cell, Mapping) or set(seed_cell) != set(METHOD_ORDER):
                raise P08FreezeError(f"P08 prediction method matrix is incomplete: {cell_id}/{raw_seed}")
            for method in METHOD_ORDER:
                descriptor = seed_cell[method]
                if not isinstance(descriptor, Mapping):
                    raise P08FreezeError(f"P08 prediction descriptor is malformed: {cell_id}/{raw_seed}/{method}")
                if descriptor.get("task_id") != TASK_ID or descriptor.get("domain") != cell_id.split("__", 1)[0] or descriptor.get("target") != cell_id.split("__", 1)[1] or descriptor.get("method_id") != method or descriptor.get("seed") not in (raw_seed, seed_key):
                    raise P08FreezeError(f"P08 prediction identity changed: {cell_id}/{raw_seed}/{method}")
                if descriptor.get("records") != RECORDS_PER_DOMAIN or descriptor.get("shape") != [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS] or descriptor.get("sequence_tokens") != SEQUENCE_TOKENS or descriptor.get("scored_post_bos_tokens") != SCORED_POST_BOS:
                    raise P08FreezeError(f"P08 prediction geometry changed: {cell_id}/{raw_seed}/{method}")
                if descriptor.get("record_ids_sha256") != observation["record_ids_sha256"]:
                    raise P08FreezeError(f"P08 prediction observation binding changed: {cell_id}/{raw_seed}/{method}")
                state = fit["states"][(raw_seed, method)]
                observation_ref = descriptor.get("observation")
                if not isinstance(observation_ref, Mapping):
                    observation_ref = {"path": descriptor.get("observation_path"), "bytes": descriptor.get("observation_bytes"), "sha256": descriptor.get("observation_sha256")}
                observation_bound = _load_record(observation_ref, root=root, description=f"P08 prediction observation {cell_id}/{raw_seed}/{method}")
                if not _same_record(observation_bound, observation["asset"]):
                    raise P08FreezeError(f"P08 prediction observation binding changed: {cell_id}/{raw_seed}/{method}")
                state_ref = descriptor.get("state")
                if not isinstance(state_ref, Mapping):
                    state_ref = {"path": descriptor.get("state_path"), "bytes": descriptor.get("state_bytes"), "sha256": descriptor.get("state_sha256")}
                state_bound = _load_record(state_ref, root=root, description=f"P08 prediction selected state {cell_id}/{raw_seed}/{method}")
                if not _same_record(state_bound, state["selected"]):
                    raise P08FreezeError(f"P08 prediction selected-state binding changed: {cell_id}/{raw_seed}/{method}")
                if not isinstance(descriptor.get("state_tensor_sha256"), str) or descriptor.get("state_tensor_sha256") != state["selected_tensor_sha256"]:
                    raise P08FreezeError(f"P08 prediction state tensor binding changed: {cell_id}/{raw_seed}/{method}")
                timing = descriptor.get("timing")
                if not isinstance(timing, Mapping) or timing.get("repeat_prediction_exact") is not True or not isinstance(timing.get("measured_passes"), int) or timing.get("measured_passes", 0) <= 0 or not isinstance(timing.get("attention_mask_sha256"), str) or _SHA256.fullmatch(timing.get("attention_mask_sha256")) is None or not isinstance(timing.get("position_ids_sha256"), str) or _SHA256.fullmatch(timing.get("position_ids_sha256")) is None:
                    raise P08FreezeError(f"P08 prediction timing/geometry receipt is incomplete: {cell_id}/{raw_seed}/{method}")
                geometry_digest = (str(timing["attention_mask_sha256"]), str(timing["position_ids_sha256"]))
                if cell_id in timing_geometry and timing_geometry[cell_id] != geometry_digest:
                    raise P08FreezeError(f"P08 prediction mask/position digest changed within cell: {cell_id}")
                timing_geometry[cell_id] = geometry_digest
                prediction = _load_record(descriptor.get("prediction"), root=root, description=f"P08 prediction {cell_id}/{raw_seed}/{method}")
                ties = _load_record(descriptor.get("tie_counts"), root=root, description=f"P08 tie counts {cell_id}/{raw_seed}/{method}")
                if descriptor.get("truth_opened") is not False or descriptor.get("candidate_arrays_persisted") is not False:
                    raise P08FreezeError(f"P08 prediction descriptor has forbidden access: {cell_id}/{raw_seed}/{method}")
                descriptors[(cell_id, raw_seed, method)] = {"prediction": prediction, "tie_counts": ties, "timing": dict(timing), "state_sha256": state["selected"]["sha256"], "observation_sha256": observation["asset"]["sha256"], "record_ids_sha256": observation["record_ids_sha256"]}
    if manifest.get("predictions_count") != PREDICTION_COUNT or manifest.get("predictions_complete") is not True:
        raise P08FreezeError("P08 prediction count or completion flag changed")
    return {"manifest": manifest, "manifest_record": manifest_record, "descriptors": descriptors}


def validate_prediction_freeze(
    *,
    repository_root: Path,
    plan_path: Path,
    approval_path: Path,
    fit_receipt_path: Path,
    prediction_manifest_path: Path,
    observation_manifest_path: Path,
    expected_plan_sha256: str = APPROVED_PLAN_SHA256,
) -> dict[str, Any]:
    """Validate fit/state/observation/prediction metadata without truth."""

    root = Path(repository_root).expanduser().resolve()
    plan_record = _validate_plan(Path(plan_path).expanduser().resolve(), root=root, expected_plan_sha256=expected_plan_sha256)
    approval_record = _validate_approval(Path(approval_path).expanduser().resolve(), root=root, expected_plan_sha256=expected_plan_sha256)
    fit = _validate_fit_receipt(Path(fit_receipt_path).expanduser().resolve(), root=root)
    observations = _validate_observation_manifest(Path(observation_manifest_path).expanduser().resolve(), root=root)
    predictions = _validate_prediction_manifest(Path(prediction_manifest_path).expanduser().resolve(), root=root, fit=fit, observations=observations)
    return {
        "schema": "token-reconstruction.trr-p08-joint-validation.v1",
        "task_id": TASK_ID,
        "status": "PREDICTION_MATRIX_VALIDATED_NO_TRUTH",
        "truth_opened": False,
        "repository_root": str(root),
        "plan": plan_record,
        "approval": approval_record,
        "fit_receipt": fit["receipt_record"],
        "observation_manifest": observations["manifest_record"],
        "prediction_manifest": predictions["manifest_record"],
        "fit_source_commit": fit["source_commit"],
        "prediction_source_commit": predictions["manifest"].get("code_commit"),
        "states": {f"{seed}::{method}": {"selected": value["selected"], "final": value["final"], "selected_step": value["selected_step"], "schedule_sha256": value["schedule_sha256"]} for (seed, method), value in fit["states"].items()},
        "prediction_count": len(predictions["descriptors"]),
        "descriptors": {f"{cell}::{seed}::{method}": descriptor for (cell, seed, method), descriptor in predictions["descriptors"].items()},
    }


def assemble_joint_freeze(
    *,
    repository_root: Path,
    plan_path: Path,
    approval_path: Path,
    fit_receipt_path: Path,
    prediction_manifest_path: Path,
    observation_manifest_path: Path,
    output_path: Path,
    expected_plan_sha256: str = APPROVED_PLAN_SHA256,
) -> dict[str, Any]:
    """Validate and write one immutable no-truth joint-freeze receipt."""

    validated = validate_prediction_freeze(
        repository_root=repository_root,
        plan_path=plan_path,
        approval_path=approval_path,
        fit_receipt_path=fit_receipt_path,
        prediction_manifest_path=prediction_manifest_path,
        observation_manifest_path=observation_manifest_path,
        expected_plan_sha256=expected_plan_sha256,
    )
    receipt = dict(validated)
    receipt["schema"] = FREEZE_SCHEMA
    receipt["status"] = "JOINT_FREEZE_VALIDATED_NO_TRUTH"
    receipt["create_only"] = True
    receipt["scientific_prerequisite"] = "all eight selected/final P08 fit states and all 32 source-free prediction descriptors are bound before truth"
    _write_create_only(Path(output_path), receipt)
    receipt["output"] = _record(Path(output_path).expanduser().resolve(), root=Path(repository_root).expanduser().resolve())
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", default=APPROVED_PLAN_SHA256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        assemble_joint_freeze(
            repository_root=args.repository_root,
            plan_path=args.plan,
            approval_path=args.approval,
            fit_receipt_path=args.fit_receipt,
            prediction_manifest_path=args.prediction_manifest,
            observation_manifest_path=args.observation_manifest,
            output_path=args.output,
            expected_plan_sha256=args.expected_plan_sha256,
        )
    except (P08FreezeError, OSError, ValueError, TypeError) as exc:
        print(f"TRR-P08 freeze error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
