#!/usr/bin/env python3
"""Strict, truth-free Stage-1 joint pre-score validation.

This command is the narrow gate between frozen predictions and the evaluator
truth sidecar.  It validates both target arms, the complete Stage-1 method
matrix, the paired public observation geometry, and evaluator resource
receipts.  It never accepts a truth path and never opens a truth file.  A
successful invocation writes one create-only ``joint_validation_receipt.json``
which the Stage-1 scoring orchestration must require before invoking the
generic scorer.

The generic scorer remains useful for small synthetic fixtures.  This module
deliberately owns the stricter natural-panel contract instead of changing that
generic API.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
for _path in (_SOURCE_ROOT, _SOURCE_ROOT / "src", _SOURCE_ROOT / "scripts" / "trr_p01"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from token_reconstruction.trr_p03.io import (  # noqa: E402
    BOS_TOKEN_ID,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PREDICTION_SCHEMA,
    P03IOError,
    create_only_directory,
    file_record,
    load_index_and_observations,
    read_json,
    read_jsonl,
    sha256_file,
    verify_freeze_receipt,
    write_json_exclusive,
)


TASK_ID = "TRR-P03"
PLAN_SCHEMA = "token-reconstruction.trr-p03-stage1-stage2-plan.v2"
JOINT_VALIDATION_SCHEMA = "token-reconstruction.trr-p03-stage1-joint-validation.v1"
RESOURCE_MAP_SCHEMA = "token-reconstruction.trr-p03-evaluator-resource-map.v1"
GENERATION_EVIDENCE_SCHEMA = "token-reconstruction.trr-p03-generation-evidence.v1"
WATCHDOG_FINISH_SCHEMA = "token-reconstruction.trr-p03-resource-watchdog-finish.v1"
WATCHDOG_GUARD_SCHEMA = "token-reconstruction.trr-p03-resource-watchdog-guard.v1"

BASE_METHODS = (
    "raw_boundary.cosine",
    "projected_boundary.cosine",
    "historical_a1.cosine",
)
A2_METHOD = "historical_a1_a2_anchor.cosine"
STAGE1_METHODS = BASE_METHODS + (A2_METHOD,)
STAGE1_IDS = tuple(f"p03-s1-r{index:04d}" for index in range(1, 25))
STAGE1_SCORED_LENGTHS = tuple(
    length for length in (16, 39, 64, 128) for _ in range(6)
)
STAGE1_SEQUENCE_LENGTHS = tuple(length + 1 for length in STAGE1_SCORED_LENGTHS)
ANCHOR_IDS = (
    "p03-s1-r0007",
    "p03-s1-r0009",
    "p03-s1-r0011",
    "p03-s1-r0012",
)
EXPECTED_CONDITIONS = {
    "bundle-a": {
        "condition_id": "matched_public",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
    },
    "bundle-b": {
        "condition_id": "shifted_full_sft",
        "model_id": "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct",
        "revision": "7fa9d06a59246629244cdd3b6b92e4fc756baa0f",
    },
}
EXPECTED_NUMERICS = {
    "device": "cpu",
    "float32_scores": True,
    "query_chunk_size": 256,
    "prototype_chunk_size": 8192,
    "deterministic_algorithms": True,
    "top1_tie_rule": "descending score, ascending token ID",
    "torch_threads": 8,
    "torch_interop_threads": 1,
}
EXPECTED_MAX_RSS_BYTES = 8 * 1024**3
EXPECTED_MIN_AVAILABLE_BYTES = 10 * 1024**3
_TRUTH_PATH_MARKERS = ("private_truth", "truth_index", "truth.safetensors")


class Stage1ValidationError(RuntimeError):
    """Raised when the strict Stage-1 pre-score contract is not met."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    _require(path.is_file(), f"{label} must be a regular file: {path}")
    return path.resolve()


def _regular_directory(path: Path, label: str) -> Path:
    path = Path(path)
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    _require(path.is_dir(), f"{label} must be a directory: {path}")
    return path.resolve()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage1ValidationError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise Stage1ValidationError(f"{label} root must be an object")
    return dict(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage1ValidationError(message)


def _digest_tensor(value: torch.Tensor) -> str:
    """Match the task-local digest convention without exposing tensor values."""

    if not isinstance(value, torch.Tensor):
        raise Stage1ValidationError("digest input is not a tensor")
    value = value.detach().cpu().contiguous()
    descriptor = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(descriptor + b"\0" + raw).hexdigest()


def _canonical_path(raw: Any, *, base: Path, label: str) -> Path:
    _require(isinstance(raw, str) and raw, f"{label} path is missing")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    _require(not candidate.is_symlink(), f"{label} path must not be a symlink")
    _require(candidate.exists(), f"{label} path is missing: {candidate}")
    return candidate.resolve()


def _verify_file_record(
    record: Any,
    *,
    base: Path,
    label: str,
    rehash: bool = True,
) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(record, Mapping), f"{label} file record is malformed")
    raw_path = record.get("path")
    path = _canonical_path(raw_path, base=base, label=label)
    _require(path.is_file(), f"{label} is not a regular file: {path}")
    bytes_value = record.get("bytes")
    digest = record.get("sha256")
    _require(
        isinstance(bytes_value, int) and not isinstance(bytes_value, bool) and bytes_value > 0,
        f"{label} byte count is invalid",
    )
    _require(isinstance(digest, str) and len(digest) == 64, f"{label} hash is invalid")
    _require(path.stat().st_size == int(bytes_value), f"{label} byte count changed")
    if rehash:
        _require(sha256_file(path) == digest, f"{label} hash changed: {path}")
    normalized = {
        "path": str(path),
        "bytes": int(bytes_value),
        "sha256": str(digest),
    }
    return path, normalized


def _record_matches_path(
    record: Any,
    expected: Path,
    *,
    base: Path,
    label: str,
    rehash: bool = True,
) -> dict[str, Any]:
    path, normalized = _verify_file_record(
        record, base=base, label=label, rehash=rehash
    )
    _require(path == expected.resolve(), f"{label} points to the wrong artifact")
    return normalized


def _file_record_for_receipt(path: Path) -> dict[str, Any]:
    return file_record(_regular_file(path, "receipt artifact"))


def _plan_expectations(plan_path: Path) -> dict[str, Any]:
    plan_path = _regular_file(plan_path, "plan")
    plan = _read_object(plan_path, "plan")
    _require(plan.get("schema") == PLAN_SCHEMA, "plan schema is not the frozen P03 schema")
    _require(plan.get("task_id") == TASK_ID, "plan task ID changed")
    _require(plan.get("truth_opened") is False, "plan is truth-opened")
    _require(plan.get("source_truth_included") is False, "plan includes source truth")

    model = plan.get("model")
    _require(isinstance(model, Mapping), "plan model declaration is missing")
    _require(model.get("id") == MODEL_ID, "plan public model ID changed")
    _require(model.get("revision") == MODEL_REVISION, "plan public model revision changed")
    _require(int(model.get("cut_depth", -1)) == CUT_DEPTH, "plan cut depth changed")
    _require(int(model.get("hidden_size", -1)) == HIDDEN_SIZE, "plan hidden size changed")
    _require(int(model.get("vocab_size", -1)) == 128256, "plan vocabulary size changed")
    _require(int(model.get("bos_token_id", -1)) == BOS_TOKEN_ID, "plan BOS changed")

    conditions = plan.get("conditions")
    _require(isinstance(conditions, Mapping), "plan target conditions are missing")
    for bundle_id, expected in EXPECTED_CONDITIONS.items():
        condition = conditions.get(expected["condition_id"])
        _require(isinstance(condition, Mapping), f"plan condition missing: {expected['condition_id']}")
        _require(condition.get("required_for_stage1") is True, f"{bundle_id} is not required for Stage 1")
        _require(condition.get("target_model_id") == expected["model_id"], f"{bundle_id} model ID changed in plan")
        _require(condition.get("target_model_revision") == expected["revision"], f"{bundle_id} model revision changed in plan")

    panel = plan.get("panel")
    _require(isinstance(panel, Mapping), "plan panel declaration is missing")
    stage1 = panel.get("stage1")
    _require(isinstance(stage1, Mapping), "plan Stage 1 panel declaration is missing")
    _require(int(stage1.get("records", -1)) == 24, "plan Stage 1 record count changed")
    _require(int(stage1.get("records_per_length", -1)) == 6, "plan Stage 1 stratum count changed")
    _require(list(stage1.get("post_bos_lengths", [])) == [16, 39, 64, 128], "plan Stage 1 lengths changed")
    _require(int(stage1.get("scored_tokens_total_per_target", -1)) == 1482, "plan Stage 1 token count changed")
    anchor = panel.get("a1_a2_anchor")
    _require(isinstance(anchor, Mapping), "plan anchor declaration is missing")
    _require(tuple(anchor.get("record_ids", ())) == ANCHOR_IDS, "plan anchor IDs changed")
    _require(list(anchor.get("record_order_indices_zero_based", [])) == [0, 2, 4, 5], "plan anchor indices changed")
    _require(int(anchor.get("post_bos_length", -1)) == 39, "plan anchor length changed")

    return {
        "path": str(plan_path),
        "sha256": sha256_file(plan_path),
        "stage1_ids": list(STAGE1_IDS),
        "stage1_sequence_lengths": list(STAGE1_SEQUENCE_LENGTHS),
        "stage1_scored_lengths": list(STAGE1_SCORED_LENGTHS),
        "anchor_ids": list(ANCHOR_IDS),
        "conditions": {key: dict(value) for key, value in EXPECTED_CONDITIONS.items()},
    }


def _target_entries(resource_map: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize the frozen resource-discovery and evaluator-map variants."""

    targets = resource_map.get("targets")
    if isinstance(targets, Mapping):
        return {
            str(bundle): dict(value)
            for bundle, value in targets.items()
            if isinstance(value, Mapping)
        }

    evaluator_targets = resource_map.get("target_conditions_evaluator_only")
    if isinstance(evaluator_targets, Mapping):
        assets = resource_map.get("assets")
        assets = assets if isinstance(assets, Mapping) else {}
        result: dict[str, dict[str, Any]] = {}
        for bundle, raw in evaluator_targets.items():
            if not isinstance(raw, Mapping):
                continue
            entry = dict(raw)
            asset_key = "base_snapshot" if bundle == "bundle-a" else "shifted_snapshot"
            asset = assets.get(asset_key)
            if isinstance(asset, Mapping) and "snapshot" in asset:
                entry.setdefault("snapshot", asset.get("snapshot"))
            result[str(bundle)] = entry
        return result

    result = {}
    for bundle, key in (("bundle-a", "base_target"), ("bundle-b", "shifted_target")):
        raw = resource_map.get(key)
        if isinstance(raw, Mapping):
            result[bundle] = dict(raw)
    return result


def _mapping_value(entry: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = entry.get(key)
        if value is not None:
            return value
    nested = entry.get("model")
    if isinstance(nested, Mapping):
        for key in keys:
            value = nested.get(key)
            if value is not None:
                return value
    return None


def _validate_resource_mapping(
    mapping_path: Path, plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    mapping_path = _regular_file(mapping_path, "resource mapping")
    resource_map = _read_object(mapping_path, "resource mapping")
    if resource_map.get("schema") is not None:
        _require(
            resource_map.get("schema") in {RESOURCE_MAP_SCHEMA, "token-reconstruction.trr-p03-setup-plan-input.v1"},
            "resource mapping schema is unsupported",
        )
    _require(resource_map.get("truth_opened", False) is not True, "resource mapping is truth-opened")
    entries = _target_entries(resource_map)
    _require(set(entries) == set(EXPECTED_CONDITIONS), "resource mapping must cover exactly bundle-a and bundle-b")
    normalized: dict[str, dict[str, Any]] = {}
    for bundle_id, expected in EXPECTED_CONDITIONS.items():
        entry = entries[bundle_id]
        condition_id = _mapping_value(entry, "condition_id")
        model_id = _mapping_value(entry, "model_id", "id")
        revision = _mapping_value(entry, "revision", "model_revision")
        snapshot = _mapping_value(entry, "snapshot", "model_path", "path")
        _require(condition_id == expected["condition_id"], f"{bundle_id} condition mapping changed")
        _require(model_id == expected["model_id"], f"{bundle_id} target model mapping changed")
        _require(revision == expected["revision"], f"{bundle_id} target revision mapping changed")
        _require(isinstance(snapshot, str) and snapshot, f"{bundle_id} target snapshot is missing")
        snapshot_path = Path(snapshot)
        _require(not snapshot_path.is_symlink(), f"{bundle_id} target snapshot must not be a symlink")
        _require(snapshot_path.is_dir(), f"{bundle_id} target snapshot is not available: {snapshot_path}")
        snapshot_path = snapshot_path.resolve()
        normalized[bundle_id] = {
            "condition_id": str(condition_id),
            "model_id": str(model_id),
            "revision": str(revision),
            "snapshot": str(snapshot_path),
        }
    return _file_record_for_receipt(mapping_path), normalized


def _validate_resource_receipt(
    receipt_path: Path,
    *,
    bundle_id: str,
    resource: Mapping[str, Any],
    observation_index: Path,
) -> dict[str, Any]:
    receipt_path = _regular_file(receipt_path, f"{bundle_id} evaluator receipt")
    receipt = _read_object(receipt_path, f"{bundle_id} evaluator receipt")
    _require(receipt.get("schema") == GENERATION_EVIDENCE_SCHEMA, f"{bundle_id} evaluator receipt schema changed")
    _require(receipt.get("truth_opened") is False, f"{bundle_id} evaluator receipt is truth-opened")
    _require(receipt.get("source_truth_included") is False, f"{bundle_id} evaluator receipt includes source truth")

    bundle = receipt.get("bundle")
    _require(isinstance(bundle, Mapping), f"{bundle_id} evaluator bundle receipt is missing")
    bundle_value = bundle.get("bundle_id", receipt.get("bundle_id"))
    _require(list(bundle.get("stages", [])) == ["stage1"], f"{bundle_id} evaluator receipt includes a non-Stage-1 bundle")
    _require(bundle_value == bundle_id, f"{bundle_id} evaluator receipt bundle ID changed")

    model = receipt.get("model")
    model = model if isinstance(model, Mapping) else receipt
    model_id = _mapping_value(model, "model_id", "id")
    revision = _mapping_value(model, "revision", "model_revision")
    snapshot = _mapping_value(model, "snapshot", "model_path", "path")
    if model_id is not None:
        _require(model_id == resource["model_id"], f"{bundle_id} evaluator model ID changed")
    if revision is not None:
        _require(revision == resource["revision"], f"{bundle_id} evaluator model revision changed")
    _require(isinstance(snapshot, str) and snapshot, f"{bundle_id} evaluator receipt has no model snapshot")
    snapshot_path = Path(snapshot)
    _require(not snapshot_path.is_symlink(), f"{bundle_id} evaluator model snapshot must not be a symlink")
    _require(snapshot_path.resolve() == Path(resource["snapshot"]).resolve(), f"{bundle_id} evaluator model snapshot changed")

    index_record = receipt.get("observation_index")
    _require(isinstance(index_record, Mapping), f"{bundle_id} evaluator observation index receipt is missing")
    _record_matches_path(
        index_record,
        observation_index,
        base=receipt_path.parent,
        label=f"{bundle_id} evaluator observation index",
        rehash=True,
    )
    geometry = receipt.get("geometry")
    _require(isinstance(geometry, Mapping), f"{bundle_id} evaluator geometry receipt is missing")
    _require(int(geometry.get("records", -1)) == 24, f"{bundle_id} evaluator record count changed")
    _require(int(geometry.get("scored_tokens", -1)) == 1482, f"{bundle_id} evaluator token count changed")
    _require(list(geometry.get("lengths", [])) == [16, 39, 64, 128], f"{bundle_id} evaluator lengths changed")
    runtime = receipt.get("environment")
    _require(isinstance(runtime, Mapping), f"{bundle_id} evaluator runtime receipt is missing")
    _require(runtime.get("device") in {"cpu", "cpu:0"}, f"{bundle_id} evaluator used a non-CPU device")
    _require(runtime.get("cuda_visible_devices") in {"", None}, f"{bundle_id} evaluator exposes a CUDA device")
    _require(runtime.get("torch_threads") == 8, f"{bundle_id} evaluator Torch thread count changed")
    _require(runtime.get("torch_interop_threads") == 1, f"{bundle_id} evaluator Torch interop count changed")
    _require(runtime.get("deterministic_algorithms") is True, f"{bundle_id} evaluator determinism changed")
    guard = receipt.get("resource_guard")
    _require(isinstance(guard, Mapping) and guard.get("status") == "PASS", f"{bundle_id} evaluator resource guard did not pass")
    panel = receipt.get("panel")
    _require(isinstance(panel, Mapping), f"{bundle_id} evaluator panel receipt is missing")
    _require(int(panel.get("records", -1)) == 24, f"{bundle_id} evaluator panel record count changed")
    panel_sha = panel.get("sha256")
    _require(isinstance(panel_sha, str) and len(panel_sha) == 64, f"{bundle_id} evaluator panel hash is missing")
    panel_path = _canonical_path(panel.get("path"), base=receipt_path.parent, label=f"{bundle_id} evaluator panel")
    panel_record = _file_record_for_receipt(panel_path)
    _require(panel_record["sha256"] == panel_sha, f"{bundle_id} evaluator panel hash changed")
    if panel.get("bytes") is not None:
        _require(int(panel.get("bytes")) == panel_record["bytes"], f"{bundle_id} evaluator panel byte count changed")
    return {
        "receipt": _file_record_for_receipt(receipt_path),
        "bundle_id": bundle_id,
        "condition_id": resource["condition_id"],
        "model_id": resource["model_id"],
        "revision": resource["revision"],
        "snapshot": resource["snapshot"],
        "panel": panel_record,
    }


def _validate_watchdog_receipt(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, f"{label} watchdog receipt")
    finish = _read_object(path, f"{label} watchdog receipt")
    _require(finish.get("schema") == WATCHDOG_FINISH_SCHEMA, f"{label} watchdog finish schema changed")
    _require(finish.get("status") == "PASS", f"{label} watchdog did not pass")
    _require(finish.get("child_return_code") == 0, f"{label} watchdog child did not exit cleanly")
    _require(finish.get("termination_reason") is None, f"{label} watchdog recorded termination")
    guard_record = finish.get("guard")
    _require(isinstance(guard_record, Mapping), f"{label} watchdog guard record is missing")
    guard_path, _ = _verify_file_record(
        guard_record, base=path.parent, label=f"{label} watchdog guard", rehash=True
    )
    guard = _read_object(guard_path, f"{label} watchdog guard")
    _require(guard.get("schema") == WATCHDOG_GUARD_SCHEMA, f"{label} watchdog guard schema changed")
    _require(guard.get("status") == "PASS", f"{label} watchdog guard did not pass")
    thresholds = guard.get("thresholds")
    _require(isinstance(thresholds, Mapping), f"{label} watchdog thresholds are missing")
    _require(int(thresholds.get("max_rss_bytes", -1)) == EXPECTED_MAX_RSS_BYTES, f"{label} RSS bound changed")
    _require(int(thresholds.get("min_available_bytes", -1)) == EXPECTED_MIN_AVAILABLE_BYTES, f"{label} host-memory bound changed")
    _require(guard.get("termination_reason") is None, f"{label} watchdog guard recorded termination")
    _require(int(guard.get("peak_group_rss_bytes", EXPECTED_MAX_RSS_BYTES + 1)) < EXPECTED_MAX_RSS_BYTES, f"{label} peak RSS reached the bound")
    _require(int(guard.get("minimum_sampled_host_mem_available_bytes", 0)) >= EXPECTED_MIN_AVAILABLE_BYTES, f"{label} host memory fell below the bound")
    return {
        "finish": _file_record_for_receipt(path),
        "guard": _file_record_for_receipt(guard_path),
        "status": "PASS",
    }


def _declared_observation_digests(index: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    records = index.get("records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, Mapping) and isinstance(record.get("record_id"), str):
                result[str(record["record_id"])] = {
                    key: str(record[key])
                    for key in ("mask_digest", "position_digest")
                    if isinstance(record.get(key), str) and record.get(key)
                }
    groups = index.get("bundles")
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            ids = group.get("record_ids")
            if not isinstance(ids, list):
                continue
            for key, plural in (("mask_digest", "mask_digests"), ("position_digest", "position_digests")):
                values = group.get(plural)
                if isinstance(values, list) and len(values) == len(ids):
                    for record_id, value in zip(ids, values, strict=True):
                        if isinstance(record_id, str) and isinstance(value, str) and value:
                            result.setdefault(record_id, {})[key] = value
                # A grouped descriptor's scalar digest covers the complete
                # [records, sequence] tensor.  It must not be compared with a
                # per-record [1, sequence] digest below.
    return result


def _validate_observation_index(
    index_path: Path, *, bundle_id: str
) -> dict[str, Any]:
    index_path = _regular_file(index_path, f"{bundle_id} observation index")
    try:
        index, records, observations = load_index_and_observations(index_path)
    except (P03IOError, OSError, ValueError, TypeError) as exc:
        raise Stage1ValidationError(f"{bundle_id} observation index failed validation") from exc
    _require(index.get("truth_opened") is False, f"{bundle_id} observation index is truth-opened")
    _require(index.get("source_truth_included") is False, f"{bundle_id} observation index includes source truth")
    _require(index.get("bundle_id") == bundle_id, f"{bundle_id} observation bundle identity changed")
    _require(index.get("model") == {"id": MODEL_ID, "revision": MODEL_REVISION}, f"{bundle_id} public model identity changed")
    _require(int(index.get("cut_depth", -1)) == CUT_DEPTH, f"{bundle_id} observation cut depth changed")
    _require(int(index.get("bos_token_id", -1)) == BOS_TOKEN_ID, f"{bundle_id} observation BOS changed")
    _require(list(index.get("record_order", [])) == list(STAGE1_IDS), f"{bundle_id} observation order is not exact Stage 1 order")
    _require(len(records) == len(STAGE1_IDS) and len(observations) == len(STAGE1_IDS), f"{bundle_id} observation count is not 24")

    groups = index.get("bundles")
    if isinstance(groups, list):
        _require(len(groups) == 4, f"{bundle_id} observation index must have four Stage 1 length bundles")
        expected_start = 0
        for group, length in zip(groups, (16, 39, 64, 128), strict=True):
            _require(isinstance(group, Mapping), f"{bundle_id} observation group is malformed")
            _require(group.get("stage") == "stage1", f"{bundle_id} observation group is not Stage 1")
            _require(int(group.get("scored_tokens", -1)) == length, f"{bundle_id} observation group length changed")
            _require(int(group.get("sequence_length", -1)) == length + 1, f"{bundle_id} observation sequence length changed")
            ids = list(group.get("record_ids", []))
            _require(ids == list(STAGE1_IDS[expected_start : expected_start + 6]), f"{bundle_id} observation group order changed")
            expected_start += 6

    declared_digests = _declared_observation_digests(index)
    digest_rows: list[dict[str, Any]] = []
    for record, observation, expected_id, expected_sequence in zip(
        records, observations, STAGE1_IDS, STAGE1_SEQUENCE_LENGTHS, strict=True
    ):
        _require(record.get("record_id") == expected_id, f"{bundle_id} observation record ID changed")
        _require(int(record.get("sequence_length", -1)) == expected_sequence, f"{bundle_id} observation geometry changed for {expected_id}")
        _require(tuple(observation.activation.shape) == (1, expected_sequence, HIDDEN_SIZE), f"{bundle_id} activation geometry changed for {expected_id}")
        mask_digest = _digest_tensor(observation.attention_mask)
        position_digest = _digest_tensor(observation.position_ids)
        declared = declared_digests.get(expected_id, {})
        if declared.get("mask_digest"):
            _require(declared["mask_digest"] == mask_digest, f"{bundle_id} mask digest changed for {expected_id}")
        if declared.get("position_digest"):
            _require(declared["position_digest"] == position_digest, f"{bundle_id} position digest changed for {expected_id}")
        digest_rows.append(
            {
                "record_id": expected_id,
                "sequence_length": expected_sequence,
                "observation_sha256": str(record.get("sha256")),
                "mask_digest": mask_digest,
                "position_digest": position_digest,
            }
        )
    panel_sha = index.get("panel_sha256")
    if panel_sha is not None:
        _require(isinstance(panel_sha, str) and len(panel_sha) == 64, f"{bundle_id} panel identity is malformed")
    return {
        "index": _file_record_for_receipt(index_path),
        "bundle_id": bundle_id,
        "record_order": list(STAGE1_IDS),
        "sequence_lengths": list(STAGE1_SEQUENCE_LENGTHS),
        "panel_sha256": panel_sha,
        "records": digest_rows,
    }


def _prediction_safetensors_metadata(
    path: Path, *, expected_masks: Mapping[str, list[int]]
) -> None:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = set(handle.keys())
            expected_keys = {method.replace(".", "_") for method in STAGE1_METHODS}
            _require(keys == expected_keys, "prediction tensor method fields are incomplete")
            _require(metadata.get("schema") == PREDICTION_SCHEMA, "prediction tensor schema changed")
            _require(metadata.get("task_id") == TASK_ID, "prediction tensor task ID changed")
            _require(metadata.get("truth_opened") == "false", "prediction tensor is truth-aware")
            for key, expected in (
                ("methods_json", list(STAGE1_METHODS)),
                ("record_order_json", list(STAGE1_IDS)),
                ("sequence_lengths_json", list(STAGE1_SEQUENCE_LENGTHS)),
            ):
                raw = metadata.get(key)
                _require(isinstance(raw, str), f"prediction tensor metadata missing: {key}")
                _require(json.loads(raw) == expected, f"prediction tensor metadata changed: {key}")
            raw_field_map = metadata.get("field_map_json")
            _require(isinstance(raw_field_map, str), "prediction tensor field map is missing")
            _require(
                json.loads(raw_field_map)
                == {method.replace(".", "_"): method for method in STAGE1_METHODS},
                "prediction tensor field map changed",
            )
            raw_masks = metadata.get("method_masks_json")
            _require(isinstance(raw_masks, str), "prediction tensor method masks are missing")
            _require(json.loads(raw_masks) == dict(expected_masks), "prediction tensor method masks changed")
            for method in STAGE1_METHODS:
                tensor = handle.get_tensor(method.replace(".", "_"))
                _require(int(tensor.shape[0]) == 24 and int(tensor.shape[1]) == 129, "prediction tensor geometry changed")
    except Stage1ValidationError:
        raise
    except Exception as exc:
        raise Stage1ValidationError(f"prediction tensor artifact is invalid: {path}") from exc


def _validate_prediction_root(
    root: Path,
    *,
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    expected_public_model_path: Path,
    expected_implementation_commit: str | None,
) -> dict[str, Any]:
    root = _regular_directory(root, "prediction root")
    try:
        freeze = verify_freeze_receipt(root)
    except (P03IOError, OSError, ValueError, TypeError) as exc:
        raise Stage1ValidationError(f"prediction root failed freeze verification: {root}") from exc
    _require(freeze.get("truth_opened") is False, f"prediction root is truth-opened: {root}")
    _require(freeze.get("status") == "PREDICTIONS_FROZEN_BEFORE_TRUTH", f"prediction root freeze status changed: {root}")
    _require(freeze.get("plan_sha256") == plan["sha256"], f"prediction root plan hash changed: {root}")
    implementation_commit = freeze.get("implementation_commit")
    _require(isinstance(implementation_commit, str) and implementation_commit, f"prediction root source commit is missing: {root}")
    if expected_implementation_commit is not None:
        _require(implementation_commit == expected_implementation_commit, f"prediction root source commit differs from expected: {root}")

    metadata = freeze.get("metadata")
    _require(isinstance(metadata, Mapping), f"prediction root freeze metadata is missing: {root}")
    _require(list(metadata.get("methods", [])) == list(STAGE1_METHODS), f"prediction root method declaration is incomplete: {root}")
    _require(int(metadata.get("records", -1)) == 24, f"prediction root record count changed: {root}")
    _require(list(metadata.get("record_ids", [])) == list(STAGE1_IDS), f"prediction root record order changed: {root}")
    _require(list(metadata.get("anchor_record_ids", [])) == list(ANCHOR_IDS), f"prediction root anchor declaration changed: {root}")

    frozen_paths = {
        str(entry.get("path"))
        for entry in freeze.get("entries", [])
        if isinstance(entry, Mapping)
    }
    required_paths = {
        "predictions.jsonl",
        "predictions.safetensors",
        "lookup_diagnostics.safetensors",
        "candidate_sets.safetensors",
        "preflight.json",
        "reconstructor_evidence.json",
        "phase_progress.jsonl",
    }
    _require(required_paths.issubset(frozen_paths), f"prediction root freeze omits required artifacts: {root}")
    _require(
        not any(any(marker in path.lower() for marker in _TRUTH_PATH_MARKERS) for path in frozen_paths),
        f"prediction root freeze names a truth artifact: {root}",
    )

    rows_path = root / "predictions.jsonl"
    try:
        rows = read_jsonl(rows_path)
    except (P03IOError, OSError, ValueError, TypeError) as exc:
        raise Stage1ValidationError(f"prediction rows are invalid: {root}") from exc
    grouped: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in STAGE1_METHODS}
    method_order: list[str] = []
    expected_sha = {
        str(row["record_id"]): str(row["observation_sha256"])
        for row in observation["records"]
    }
    expected_lengths = dict(zip(STAGE1_IDS, STAGE1_SEQUENCE_LENGTHS, strict=True))
    for row in rows:
        _require(isinstance(row, Mapping), f"prediction row is malformed: {root}")
        _require(row.get("truth_opened") is False, f"prediction row is truth-aware: {root}")
        method = row.get("method")
        record_id = row.get("record_id")
        _require(method in STAGE1_METHODS, f"prediction method is not a Stage 1 method: {method}")
        _require(isinstance(record_id, str) and record_id in expected_lengths, f"prediction record ID is invalid: {record_id}")
        if method not in method_order:
            method_order.append(str(method))
        _require(record_id not in grouped[str(method)], f"duplicate prediction row: {method}/{record_id}")
        sequence_length = row.get("sequence_length")
        _require(isinstance(sequence_length, int) and not isinstance(sequence_length, bool), f"prediction sequence length is invalid: {method}/{record_id}")
        _require(int(sequence_length) == expected_lengths[record_id], f"prediction sequence length changed: {method}/{record_id}")
        tokens = row.get("prediction_tokens")
        _require(isinstance(tokens, list) and len(tokens) == int(sequence_length), f"prediction token geometry changed: {method}/{record_id}")
        try:
            token_values = [int(value) for value in tokens]
        except (TypeError, ValueError) as exc:
            raise Stage1ValidationError(f"prediction token values are invalid: {method}/{record_id}") from exc
        _require(token_values[0] == BOS_TOKEN_ID, f"prediction BOS changed: {method}/{record_id}")
        _require(all(0 <= value < 128256 for value in token_values), f"prediction vocabulary changed: {method}/{record_id}")
        _require(row.get("observation_sha256") == expected_sha[record_id], f"prediction observation identity changed: {method}/{record_id}")
        for key, expected_size in (("top1_tie_count", int(sequence_length) - 1), ("top1_scores", int(sequence_length) - 1), ("top1_runner_margins", int(sequence_length) - 1)):
            if key in row:
                _require(isinstance(row[key], list) and len(row[key]) == expected_size, f"prediction diagnostic geometry changed: {method}/{record_id}/{key}")
                if key == "top1_tie_count":
                    _require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 1 for value in row[key]), f"prediction tie diagnostics changed: {method}/{record_id}")
                else:
                    _require(all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in row[key]), f"prediction score diagnostics changed: {method}/{record_id}")
        grouped[str(method)][record_id] = dict(row)

    _require(method_order == list(STAGE1_METHODS), f"prediction methods are not serialized in canonical order: {root}")
    for method in BASE_METHODS:
        _require(list(grouped[method]) == list(STAGE1_IDS), f"{method} does not cover ordered Stage 1 records: {root}")
    _require(list(grouped[A2_METHOD]) == list(ANCHOR_IDS), f"A1+A2 coverage is not the exact anchor: {root}")

    masks = {
        method: [1 if record_id in grouped[method] else 0 for record_id in STAGE1_IDS]
        for method in STAGE1_METHODS
    }
    prediction_tensor_path = root / "predictions.safetensors"
    _prediction_safetensors_metadata(prediction_tensor_path, expected_masks=masks)

    preflight = _read_object(root / "preflight.json", f"{root} preflight")
    _require(preflight.get("truth_opened") is False and preflight.get("source_truth_included") is False, f"prediction preflight is truth-aware: {root}")
    _require(list(preflight.get("methods", [])) == list(STAGE1_METHODS), f"prediction preflight methods changed: {root}")
    _record_matches_path(preflight.get("input"), Path(observation["index"]["path"]), base=root, label=f"{root} preflight input", rehash=True)
    numerics = preflight.get("numerics")
    _require(isinstance(numerics, Mapping), f"prediction preflight numerics are missing: {root}")
    for key, expected in EXPECTED_NUMERICS.items():
        _require(numerics.get(key) == expected, f"prediction preflight numeric setting changed: {root}/{key}")
    _require(numerics.get("cuda_visible_devices") in {"", None}, f"prediction preflight exposes a CUDA device: {root}")
    local_guard = preflight.get("resource_guard")
    _require(isinstance(local_guard, Mapping) and local_guard.get("status") == "PASS", f"prediction preflight resource guard failed: {root}")

    evidence = _read_object(root / "reconstructor_evidence.json", f"{root} evidence")
    _require(evidence.get("truth_opened") is False and evidence.get("source_truth_included") is False, f"prediction evidence is truth-aware: {root}")
    _require(list(evidence.get("methods", [])) == list(STAGE1_METHODS), f"prediction evidence methods changed: {root}")
    _require(int(evidence.get("records", -1)) == 24, f"prediction evidence record count changed: {root}")
    _require(int(evidence.get("scored_tokens", -1)) == 1482, f"prediction evidence token count changed: {root}")
    _require(evidence.get("implementation_commit") == implementation_commit, f"prediction evidence source commit changed: {root}")
    command = evidence.get("command")
    _require(isinstance(command, Mapping), f"prediction command evidence is missing: {root}")
    argv = command.get("argv")
    _require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), f"prediction command argv is missing: {root}")
    model_values: list[str] = []
    commit_values: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--model-path":
            _require(index + 1 < len(argv), f"prediction model path argument is incomplete: {root}")
            model_values.append(argv[index + 1])
            index += 2
            continue
        if value.startswith("--model-path="):
            model_values.append(value.split("=", 1)[1])
        if value == "--implementation-commit":
            _require(index + 1 < len(argv), f"prediction implementation commit argument is incomplete: {root}")
            commit_values.append(argv[index + 1])
            index += 2
            continue
        if value.startswith("--implementation-commit="):
            commit_values.append(value.split("=", 1)[1])
        index += 1
    _require(len(model_values) == 1, f"prediction command must declare exactly one --model-path: {root}")
    command_cwd = command.get("cwd")
    command_base = Path(command_cwd) if isinstance(command_cwd, str) and command_cwd else root
    model_path = Path(model_values[0])
    if not model_path.is_absolute():
        model_path = command_base / model_path
    _require(not model_path.is_symlink(), f"prediction command model path must not be a symlink: {root}")
    _require(model_path.resolve() == expected_public_model_path.resolve(), f"prediction command does not use the pinned public base model: {root}")
    _require(len(commit_values) == 1 and commit_values[0] == implementation_commit, f"prediction command source commit changed: {root}")
    _record_matches_path(evidence.get("observation_index"), Path(observation["index"]["path"]), base=root, label=f"{root} evidence observation index", rehash=True)
    source_files = evidence.get("code_files")
    _require(isinstance(source_files, list) and source_files, f"prediction evidence source files are missing: {root}")
    normalized_sources: list[dict[str, Any]] = []
    for index, source in enumerate(source_files):
        _, normalized = _verify_file_record(source, base=root, label=f"{root} source file {index}", rehash=True)
        normalized_sources.append(normalized)
    normalized_sources.sort(key=lambda value: (value["path"], value["sha256"]))

    asset_records: dict[str, dict[str, Any]] = {}
    for key in ("prototype", "historical_lens", "projected_prototype"):
        # These are provenance-bearing inputs.  Rehash them here so a mutable
        # path with the same byte count cannot pass the pre-score gate.
        _, normalized = _verify_file_record(evidence.get(key), base=root, label=f"{root} {key}", rehash=True)
        asset_records[key] = normalized
    method_metadata = evidence.get("method_metadata")
    _require(isinstance(method_metadata, Mapping), f"prediction method metadata are missing: {root}")
    a2_metadata = method_metadata.get(A2_METHOD)
    _require(isinstance(a2_metadata, Mapping), f"prediction A1+A2 metadata are missing: {root}")
    _require(list(a2_metadata.get("anchor_record_ids", [])) == list(ANCHOR_IDS), f"prediction A1+A2 metadata anchor changed: {root}")
    return {
        "root": str(root),
        "freeze": _file_record_for_receipt(root / "freeze_receipt.json"),
        "plan_sha256": str(freeze["plan_sha256"]),
        "implementation_commit": str(implementation_commit),
        "methods": list(STAGE1_METHODS),
        "record_order": list(STAGE1_IDS),
        "anchor_record_ids": list(ANCHOR_IDS),
        "source_files": normalized_sources,
        "assets": asset_records,
        "numerics": {key: numerics.get(key) for key in EXPECTED_NUMERICS},
        "evidence": _file_record_for_receipt(root / "reconstructor_evidence.json"),
    }


def validate_stage1(
    *,
    plan_path: Path,
    observation_index_a: Path,
    observation_index_b: Path,
    prediction_root_a: Path,
    prediction_root_b: Path,
    resource_mapping: Path,
    resource_receipt_a: Path,
    resource_receipt_b: Path,
    output_root: Path,
    watchdog_receipt_a: Path | None = None,
    watchdog_receipt_b: Path | None = None,
    implementation_commit: str | None = None,
) -> dict[str, Any]:
    """Validate both frozen arms and write a create-only Stage-1 receipt."""

    plan = _plan_expectations(plan_path)
    mapping_record, resources = _validate_resource_mapping(resource_mapping, plan)
    observations = {
        "bundle-a": _validate_observation_index(observation_index_a, bundle_id="bundle-a"),
        "bundle-b": _validate_observation_index(observation_index_b, bundle_id="bundle-b"),
    }
    _require(observations["bundle-a"]["record_order"] == observations["bundle-b"]["record_order"], "paired observation record order differs")
    _require(observations["bundle-a"]["sequence_lengths"] == observations["bundle-b"]["sequence_lengths"], "paired observation geometry differs")
    _require(observations["bundle-a"]["panel_sha256"] == observations["bundle-b"]["panel_sha256"], "paired observation panel identity differs")
    for left, right in zip(observations["bundle-a"]["records"], observations["bundle-b"]["records"], strict=True):
        _require(left["record_id"] == right["record_id"], "paired observation record IDs differ")
        _require(left["sequence_length"] == right["sequence_length"], f"paired observation sequence differs: {left['record_id']}")
        _require(left["mask_digest"] == right["mask_digest"], f"paired observation mask differs: {left['record_id']}")
        _require(left["position_digest"] == right["position_digest"], f"paired observation positions differ: {left['record_id']}")

    resources_receipts = {
        "bundle-a": _validate_resource_receipt(resource_receipt_a, bundle_id="bundle-a", resource=resources["bundle-a"], observation_index=Path(observations["bundle-a"]["index"]["path"])),
        "bundle-b": _validate_resource_receipt(resource_receipt_b, bundle_id="bundle-b", resource=resources["bundle-b"], observation_index=Path(observations["bundle-b"]["index"]["path"])),
    }
    watchdog_receipts: dict[str, Any] = {}
    if watchdog_receipt_a is not None:
        watchdog_receipts["bundle-a"] = _validate_watchdog_receipt(watchdog_receipt_a, label="bundle-a")
    if watchdog_receipt_b is not None:
        watchdog_receipts["bundle-b"] = _validate_watchdog_receipt(watchdog_receipt_b, label="bundle-b")

    predictions = {
        "bundle-a": _validate_prediction_root(
            prediction_root_a,
            plan=plan,
            observation=observations["bundle-a"],
            expected_public_model_path=Path(resources["bundle-a"]["snapshot"]),
            expected_implementation_commit=implementation_commit,
        ),
        "bundle-b": _validate_prediction_root(
            prediction_root_b,
            plan=plan,
            observation=observations["bundle-b"],
            expected_public_model_path=Path(resources["bundle-a"]["snapshot"]),
            expected_implementation_commit=implementation_commit,
        ),
    }
    _require(predictions["bundle-a"]["implementation_commit"] == predictions["bundle-b"]["implementation_commit"], "paired prediction source commits differ")
    _require(predictions["bundle-a"]["source_files"] == predictions["bundle-b"]["source_files"], "paired prediction source hashes differ")
    _require(predictions["bundle-a"]["assets"] == predictions["bundle-b"]["assets"], "paired prediction public asset hashes differ")
    _require(predictions["bundle-a"]["numerics"] == predictions["bundle-b"]["numerics"], "paired prediction numeric settings differ")
    _require(resources_receipts["bundle-a"]["panel"]["sha256"] == resources_receipts["bundle-b"]["panel"]["sha256"], "paired evaluator panel sources differ")

    output_root = Path(output_root)
    _require(not output_root.exists() and not output_root.is_symlink(), f"joint receipt output must be create-only: {output_root}")
    receipt_root = create_only_directory(output_root.resolve())
    receipt_path = receipt_root / "joint_validation_receipt.json"
    payload = {
        "schema": JOINT_VALIDATION_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALIDATED",
        "validation": "STAGE1_JOINT_VALIDATION_PASS",
        "truth_opened": False,
        "created_utc": _utc_now(),
        "plan": {"path": plan["path"], "sha256": plan["sha256"]},
        "implementation_commit": predictions["bundle-a"]["implementation_commit"],
        "required_methods": list(STAGE1_METHODS),
        "stage1_record_order": list(STAGE1_IDS),
        "stage1_sequence_lengths": list(STAGE1_SEQUENCE_LENGTHS),
        "anchor_record_ids": list(ANCHOR_IDS),
        "observations": observations,
        "predictions": predictions,
        "evaluator_resource_mapping": mapping_record,
        "evaluator_resource_receipts": resources_receipts,
        "watchdog_receipts": watchdog_receipts,
        "score_prerequisite": {
            "paired_prediction_root_required": True,
            "allow_unequal_strata": False,
            "truth_read_after_this_receipt": True,
        },
    }
    write_json_exclusive(receipt_path, payload)
    receipt_path.chmod(0o444)
    return {
        "status": payload["status"],
        "truth_opened": False,
        "receipt": _file_record_for_receipt(receipt_path),
        "output_root": str(receipt_root),
        "bundles": ["bundle-a", "bundle-b"],
        "records": 24,
        "methods": list(STAGE1_METHODS),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--observation-index-a", type=Path, required=True)
    parser.add_argument("--observation-index-b", type=Path, required=True)
    parser.add_argument("--prediction-root-a", type=Path, required=True)
    parser.add_argument("--prediction-root-b", type=Path, required=True)
    parser.add_argument("--resource-mapping", "--resource-map", dest="resource_mapping", type=Path, required=True)
    parser.add_argument("--resource-receipt-a", type=Path, required=True)
    parser.add_argument("--resource-receipt-b", type=Path, required=True)
    parser.add_argument("--watchdog-receipt-a", type=Path, default=None)
    parser.add_argument("--watchdog-receipt-b", type=Path, default=None)
    parser.add_argument("--implementation-commit", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_stage1(
        plan_path=args.plan,
        observation_index_a=args.observation_index_a,
        observation_index_b=args.observation_index_b,
        prediction_root_a=args.prediction_root_a,
        prediction_root_b=args.prediction_root_b,
        resource_mapping=args.resource_mapping,
        resource_receipt_a=args.resource_receipt_a,
        resource_receipt_b=args.resource_receipt_b,
        output_root=args.output_root,
        watchdog_receipt_a=args.watchdog_receipt_a,
        watchdog_receipt_b=args.watchdog_receipt_b,
        implementation_commit=args.implementation_commit,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Stage1ValidationError as exc:
        print(f"TRR-P03 Stage 1 joint validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
