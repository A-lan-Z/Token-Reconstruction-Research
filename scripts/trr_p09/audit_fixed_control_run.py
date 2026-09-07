#!/usr/bin/env python3
"""Immutable CPU audit for a completed TRR-P09 fixed-control run.

This is deliberately a receipt audit, not a runner.  It reads only public
attention-mask/position sidecars from the fitting bank, hashes their bound
payload files, audits the serialized schedule and checkpoints, and loads the
selected decoder state on CPU for strict deployment compatibility.  It never
opens an H tensor, runs a model forward, or reads evaluation truth.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from scripts.trr_p09.fixed_control_runner import method_state_digest, tensor_digest
from token_reconstruction.trr0007_positionwise import build_residual_mlp512
from token_reconstruction.trr_p09_fixed_control_adapter import FixedPublicReadoutHook


TASK_ID = "TRR-P09"
SCHEMA = "token-reconstruction.trr-p09-fixed-control-audit-handoff.v3"
EXPECTED_GRID = [0, 1000, 2000, 4000, 8000, 12000, 13000]
EXPECTED_KEYS = [
    "base.W",
    "base.b",
    "base.key.bias",
    "base.key.weight",
    "base.output.bias",
    "base.output.weight",
    "base.query.bias",
    "base.query.weight",
    "base.s",
    "base.value.bias",
    "base.value.weight",
    "down.bias",
    "down.weight",
    "up.bias",
    "up.weight",
]
EXPECTED_SHAPES = {
    "base.W": [2048, 2048],
    "base.b": [2048],
    "base.key.bias": [128],
    "base.key.weight": [128, 2048],
    "base.output.bias": [2048],
    "base.output.weight": [2048, 128],
    "base.query.bias": [128],
    "base.query.weight": [128, 2048],
    "base.s": [],
    "base.value.bias": [128],
    "base.value.weight": [128, 2048],
    "down.bias": [512],
    "down.weight": [512, 2048],
    "up.bias": [2048],
    "up.weight": [2048, 512],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, label: str | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular file: {path}")
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if label is not None:
        record["label"] = label
    return record


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def safe_header(path: Path, keys: tuple[str, ...]) -> tuple[dict[str, str], list[str], dict[str, list[int]], dict[str, torch.Tensor]]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        names = list(handle.keys())
        shapes = {name: list(handle.get_slice(name).get_shape()) for name in names}
        for key in keys:
            if key not in names:
                raise RuntimeError(f"{path} lacks required tensor {key}")
            tensors[key] = handle.get_tensor(key)
    return metadata, names, shapes, tensors


def safe_header_only(path: Path) -> tuple[dict[str, str], list[str], dict[str, list[int]]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        names = list(handle.keys())
        shapes = {name: list(handle.get_slice(name).get_shape()) for name in names}
    return metadata, names, shapes


def load_public_sidecars(path: Path) -> dict[str, Any]:
    metadata, names, shapes, tensors = safe_header(path, ("attention_mask", "position_ids"))
    mask = tensors["attention_mask"].detach().cpu().contiguous()
    positions = tensors["position_ids"].detach().cpu().contiguous()
    if mask.ndim != 2 or positions.ndim != 2 or tuple(mask.shape) != tuple(positions.shape):
        raise RuntimeError(f"sidecar geometry mismatch: {path}")
    mask_np = mask.numpy().astype(bool, copy=False)
    positions_np = positions.numpy().astype(np.int64, copy=False)
    return {
        "metadata": metadata,
        "keys": names,
        "shapes": shapes,
        "mask": mask_np,
        "positions": positions_np,
        "mask_dtype": str(mask.dtype),
        "position_dtype": str(positions.dtype),
    }


def assert_public_positions(mask: np.ndarray, positions: np.ndarray, *, label: str) -> dict[str, Any]:
    bad: list[dict[str, int]] = []
    for row in range(mask.shape[0]):
        active = int(mask[row].sum())
        if active <= 0 or not bool(mask[row, 0]):
            bad.append({"row": row, "active": active, "reason": "BOS inactive/empty"})
            continue
        if not np.array_equal(positions[row, :active], np.arange(active, dtype=np.int64)):
            bad.append({"row": row, "active": active, "reason": "active positions not arange"})
        if np.any(positions[row, active:] != 0):
            bad.append({"row": row, "active": active, "reason": "inactive positions not zero"})
    if bad:
        raise RuntimeError(f"{label}: position contract failed: {bad[:3]}")
    return {
        "rows": int(mask.shape[0]),
        "sequence_tokens": int(mask.shape[1]),
        "active_positions": int(mask.sum()),
        "min_active_per_row": int(mask.sum(axis=1).min()),
        "max_active_per_row": int(mask.sum(axis=1).max()),
        "first_position_active": bool(mask[:, 0].all()),
        "position_rule": "active prefix has position_ids arange(active); inactive padding is zero",
    }


def curve_row(row: Mapping[str, Any]) -> dict[str, Any]:
    validation = row["validation"]
    domains = validation["domains"]
    scopes = row["state_binding"]["fitting_diagnostics"]["scopes"]
    out: dict[str, Any] = {
        "step": int(row["step"]),
        "domain_balanced_token_accuracy": float(validation["domain_balanced_token_accuracy"]),
        "Finance_token_accuracy": float(domains["Finance"]["token_accuracy"]),
        "Pile_token_accuracy": float(domains["Pile"]["token_accuracy"]),
        "Finance_correct_tokens": int(domains["Finance"]["correct_tokens"]),
        "Finance_token_rows": int(domains["Finance"]["token_rows"]),
        "Pile_correct_tokens": int(domains["Pile"]["correct_tokens"]),
        "Pile_token_rows": int(domains["Pile"]["token_rows"]),
        "validation_correct_tokens": int(validation["correct_tokens"]),
        "validation_token_rows": int(validation["token_rows"]),
        "validation_seconds": float(row["validation_seconds"]),
        "checkpoint_file": row["state_binding"]["checkpoint"]["path"],
        "checkpoint_file_sha256": row["state_binding"]["checkpoint"]["sha256"],
        "logical_state_sha256": row["state_sha256"],
    }
    by_scope = {scope["scope"]: scope for scope in scopes}
    for scope_name in ("frozen64", "full_bank"):
        scope = by_scope.get(scope_name)
        prefix = scope_name
        if scope is None:
            for key in ("row_count", "batch_count", "position_chunk_count", "full_vocab_logits_rows", "elapsed_seconds"):
                out[f"{prefix}_{key}"] = None
        else:
            out[f"{prefix}_row_count"] = int(scope["row_count"])
            out[f"{prefix}_batch_count"] = int(scope["batch_count"])
            out[f"{prefix}_position_chunk_count"] = int(scope["position_chunk_count"])
            out[f"{prefix}_full_vocab_logits_rows"] = int(scope["full_vocab_logits_rows"])
            out[f"{prefix}_elapsed_seconds"] = float(scope["elapsed_seconds"])
    return out


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    started = utc_now()
    receipt_path = Path(args.receipt).expanduser().resolve()
    schedule_path = Path(args.schedule).expanduser().resolve()
    watchdog_dir = Path(args.watchdog_dir).expanduser().resolve()
    root = Path(args.repository_root).expanduser().resolve()
    bank = str(args.bank)
    if bank != "B1":
        raise RuntimeError("this committed invocation is bounded to B1; use the same procedure with a separately reviewed B0 binding")
    receipt = load_json(receipt_path)
    manifest_path = Path(args.b1_manifest).expanduser().resolve()
    b0_binding_path = Path(args.b0_binding).expanduser().resolve()
    support_binding_path = Path(args.support_binding).expanduser().resolve()
    manifest = load_json(manifest_path)
    b0_binding = load_json(b0_binding_path)
    support_binding = load_json(support_binding_path)
    source_script = Path(__file__).expanduser().resolve()
    invocation = shlex.join([sys.executable, *sys.argv])
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, detail: Any) -> None:
        checks[name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            raise RuntimeError(f"{name} failed: {detail}")

    check("receipt_completed", receipt.get("status") == "COMPLETED", receipt.get("status"))
    check("bank_and_task", receipt.get("task_id") == TASK_ID and receipt.get("assets", {}).get("fixed_diagnostic", {}).get("bank") == bank, {"task_id": receipt.get("task_id"), "bank": receipt.get("assets", {}).get("fixed_diagnostic", {}).get("bank")})
    training = receipt["training_contract"]
    check("training_contract", training == {
        "checkpoint_grid": EXPECTED_GRID,
        "gradient_clip_norm": 1.0,
        "learning_rate": 0.0002,
        "optimizer": "AdamW foreach=False",
        "position_budget": 512,
        "record_batch_size": 8,
        "schedule_bank": "B1",
        "seed": 4010,
        "selection_metric": "domain_balanced_token_accuracy",
        "steps": 13000,
        "validation_sequence_tokens": 128,
        "weight_decay": 0.0,
    }, training)

    # Guarded-run evidence is retained by exact byte hash, while only its
    # compact summary is copied into the committed audit.
    finish_path = watchdog_dir / "finish.json"
    finish = load_json(finish_path)
    check("watchdog_pass", finish.get("status") == "PASS" and finish.get("child_return_code") == 0 and finish.get("wrapper_exit_code") == 0 and finish.get("termination_reason") is None, {k: finish.get(k) for k in ("status", "child_return_code", "wrapper_exit_code", "termination_reason")})
    guard_path = watchdog_dir / "resource_guard.json"
    guard = load_json(guard_path)
    check("resource_guard_pass", guard.get("status") == "PASS" and not guard.get("errors") and guard.get("termination_reason") is None and guard.get("process_group_fail_closed") and guard.get("disk_and_output_enforced_during_live_process"), {k: guard.get(k) for k in ("status", "errors", "termination_reason", "process_group_fail_closed", "disk_and_output_enforced_during_live_process")})
    guard_summary = {
        "status": guard.get("status"),
        "thresholds": guard.get("thresholds"),
        "sample_count": int(guard.get("sample_count", 0)),
        "peak_group_rss_bytes": int(guard["peak_group_rss_bytes"]),
        "minimum_sampled_host_mem_available_bytes": int(guard["minimum_sampled_host_mem_available_bytes"]),
        "minimum_sampled_disk_free_bytes": int(guard["minimum_sampled_disk_free_bytes"]),
        "peak_sampled_output_bytes": int(guard["peak_sampled_output_bytes"]),
        "process_group_fail_closed": bool(guard["process_group_fail_closed"]),
        "disk_and_output_enforced_during_live_process": bool(guard["disk_and_output_enforced_during_live_process"]),
        "errors": list(guard.get("errors", [])),
        "termination_actions": list(guard.get("termination_actions", [])),
    }

    # Open only public mask/position sidecars.  Payload hashes are full-file
    # hashes, but activations/token IDs are never deserialized.
    b0_payload_path = Path(b0_binding["artifacts"]["payload"]["path"]).expanduser().resolve()
    b0_payload_descriptor = b0_binding["artifacts"]["payload"]
    b0_side = load_public_sidecars(b0_payload_path)
    b0_contract = assert_public_positions(b0_side["mask"], b0_side["positions"], label="B0 public sidecars")
    b0_actual = file_record(b0_payload_path, label="B0 public H payload")
    check("b0_payload_hash", b0_actual["sha256"] == b0_payload_descriptor["sha256"], {"declared": b0_payload_descriptor["sha256"], "actual": b0_actual["sha256"]})

    rows = 12000
    sequence_tokens = 192
    mask = np.zeros((rows, sequence_tokens), dtype=np.bool_)
    positions = np.zeros((rows, sequence_tokens), dtype=np.int64)
    mask[:1200] = b0_side["mask"]
    positions[:1200] = b0_side["positions"]
    payload_records: list[dict[str, Any]] = [{
        "bank": "B0",
        "global_row_range": [0, 1200],
        "path": str(b0_payload_path),
        "declared_bytes": int(b0_payload_descriptor["bytes"]),
        "declared_sha256": b0_payload_descriptor["sha256"],
        "actual_bytes": b0_actual["bytes"],
        "actual_sha256": b0_actual["sha256"],
        "safe_open_keys": b0_side["keys"],
        "attention_mask_shape": b0_side["shapes"]["attention_mask"],
        "position_ids_shape": b0_side["shapes"]["position_ids"],
        "attention_mask_dtype": b0_side["mask_dtype"],
        "position_ids_dtype": b0_side["position_dtype"],
        "metadata_hashes": {k: b0_side["metadata"].get(k) for k in ("activations_sha256", "attention_mask_sha256", "position_ids_sha256", "token_ids_sha256")},
        "activations_values_loaded": False,
        "token_ids_values_loaded": False,
    }]
    shards = manifest["sharding"]["shards"]
    check("b1_shard_count", len(shards) == 169, len(shards))
    for shard in shards:
        start = int(shard["row_range"]["start"])
        stop = int(shard["row_range"]["stop"])
        path = (manifest_path.parent / shard["payload"]["path"]).resolve()
        side = load_public_sidecars(path)
        if tuple(side["mask"].shape) != (stop - start, sequence_tokens):
            raise RuntimeError(f"B1 shard geometry mismatch: {path}")
        mask[start:stop] = side["mask"]
        positions[start:stop] = side["positions"]
        actual = file_record(path, label="B1 public H payload")
        if actual["bytes"] != int(shard["payload"]["bytes"]) or actual["sha256"] != shard["payload"]["sha256"]:
            raise RuntimeError(f"B1 payload hash mismatch: {path}")
        payload_records.append({
            "bank": "B1",
            "shard_id": int(shard["shard_id"]),
            "global_row_range": [start, stop],
            "path": str(path),
            "declared_bytes": int(shard["payload"]["bytes"]),
            "declared_sha256": shard["payload"]["sha256"],
            "actual_bytes": actual["bytes"],
            "actual_sha256": actual["sha256"],
            "safe_open_keys": side["keys"],
            "attention_mask_shape": side["shapes"]["attention_mask"],
            "position_ids_shape": side["shapes"]["position_ids"],
            "attention_mask_dtype": side["mask_dtype"],
            "position_ids_dtype": side["position_dtype"],
            "metadata_hashes": {k: side["metadata"].get(k) for k in ("activations_sha256", "attention_mask_sha256", "position_ids_sha256", "token_ids_sha256")},
            "activations_values_loaded": False,
            "token_ids_values_loaded": False,
        })
    check("composed_rows", bool(np.all(mask.sum(axis=1) > 0)) and bool(np.all(mask[:, 0])), {"rows": rows, "active_positions": int(mask.sum())})
    position_summary = assert_public_positions(mask, positions, label="B1 composed public sidecars")
    available_positions = int(mask[:, 1:].sum())
    check("available_positions", available_positions == 1243710, available_positions)
    mask_digest = tensor_digest(torch.from_numpy(mask))

    # The support binding is a public fitting-label summary; it is not a
    # source-text or evaluation-truth asset.
    support_entry = next(item for item in support_binding["banks"] if item.get("bank") == bank)
    support_file = Path(support_entry["path"]).expanduser().resolve()
    support_actual = file_record(support_file, label="B1 public support labels")
    check("support_payload_hash", support_actual["sha256"] == support_entry["sha256"], {"declared": support_entry["sha256"], "actual": support_actual["sha256"]})
    check("support_geometry", support_entry["support_count"] == 45631 and support_entry["post_bos_valid_positions"] == available_positions, support_entry)

    # Load and audit only the serialized schedule tensors.
    schedule_actual = file_record(schedule_path, label="B1 common serialized schedule")
    schedule_declared = receipt["assets"]["common_schedule"]
    check("schedule_file_hash", schedule_actual["sha256"] == schedule_declared["sha256"], {"declared": schedule_declared["sha256"], "actual": schedule_actual["sha256"]})
    with safe_open(str(schedule_path), framework="pt", device="cpu") as handle:
        schedule_metadata = dict(handle.metadata() or {})
        schedule_names = list(handle.keys())
        schedule = {name: handle.get_tensor(name).detach().cpu().contiguous() for name in ("batch_record_indices", "draw_record_slots", "draw_position_slots", "used_replacement")}
    check("schedule_semantic", schedule_metadata.get("schedule_semantic_sha256") == receipt["exposure"]["schedule_semantic_sha256"] == "8acdb2c4f8e5afba546ad01cbe0adae340eb841c8e3322c919e976434d29ede5", schedule_metadata)
    check("schedule_mask_semantic", schedule_metadata.get("valid_mask_semantic_sha256") == mask_digest == "c9f72b4d3733f89171af1712505248a8c050d3220f1c4b43e4b343e94b7b5890", {"schedule": schedule_metadata.get("valid_mask_semantic_sha256"), "observed": mask_digest})
    batch = schedule["batch_record_indices"].numpy().astype(np.int64, copy=False)
    draw_record_slots = schedule["draw_record_slots"].numpy().astype(np.int64, copy=False)
    draw_position_slots = schedule["draw_position_slots"].numpy().astype(np.int64, copy=False)
    check("schedule_geometry", batch.shape == (13000, 8) and draw_record_slots.shape == (13000, 512) and draw_position_slots.shape == (13000, 512), {k: list(v.shape) for k, v in schedule.items()})
    global_records = np.take_along_axis(batch, draw_record_slots, axis=1)
    valid = mask[global_records, draw_position_slots]
    check("all_draws_valid", bool(valid.all()), {"invalid_draws": int((~valid).sum())})
    pair_codes = np.asarray((global_records * sequence_tokens + draw_position_slots).reshape(-1), dtype=np.int64)
    unique_pair_codes = np.unique(pair_codes)
    record_batch_codes = np.asarray((np.arange(batch.shape[0], dtype=np.int64)[:, None] * rows + batch).reshape(-1), dtype=np.int64)
    unique_record_batch_codes = np.unique(record_batch_codes)
    check("record_batch_schedule", len(unique_record_batch_codes) == 104000 and all(len(np.unique(row)) == 8 for row in batch), {"unique_record_batch_exposures": int(len(unique_record_batch_codes)), "steps": int(batch.shape[0])})
    exposures = {
        "coordinate_definition": "global_record=batch_record_indices[step,draw_record_slot]; postBOS_position=draw_position_slot (raw sequence column 1..191); BOS column 0 is excluded",
        "total_draws": int(pair_codes.size),
        "valid_draws": int(valid.sum()),
        "invalid_draws": int((~valid).sum()),
        "actually_sampled_unique_global_record_postBOS_position_pairs": int(len(unique_pair_codes)),
        "repeated_record_position_draws": int(pair_codes.size - len(unique_pair_codes)),
        "available_postBOS_positions": available_positions,
        "records_in_composed_bank": rows,
        "actually_sampled_unique_records": int(np.unique(global_records).size),
        "actually_sampled_unique_valid_records": int(np.unique(global_records[valid]).size),
        "actually_batched_unique_records": int(np.unique(batch).size),
        "record_batch_exposures": int(len(record_batch_codes)),
        "record_batch_exposure_unique_pairs": int(len(unique_record_batch_codes)),
        "record_batch_steps": int(batch.shape[0]),
        "record_batch_slots_per_step": int(batch.shape[1]),
        "used_replacement_steps": int(schedule["used_replacement"].sum().item()),
        "schedule_semantic_sha256": receipt["exposure"]["schedule_semantic_sha256"],
        "actually_sampled_pair_support_sha256_sorted_packed_int64": hashlib.sha256(np.ascontiguousarray(unique_pair_codes).tobytes()).hexdigest(),
        "record_batch_exposure_digest_sorted_packed_step_record": hashlib.sha256(np.ascontiguousarray(unique_record_batch_codes).tobytes()).hexdigest(),
        "sampled_pair_code_examples_sorted": [
            {"global_record": int(code // sequence_tokens), "postBOS_position": int(code % sequence_tokens), "zero_based_postBOS_offset": int(code % sequence_tokens - 1)}
            for code in unique_pair_codes[:10]
        ],
        "record_batch_exposure_counts": {
            "min": int(np.bincount(global_records.reshape(-1), minlength=rows).min()),
            "max": int(np.bincount(global_records.reshape(-1), minlength=rows).max()),
            "mean": float(np.bincount(global_records.reshape(-1), minlength=rows).mean()),
            "records_with_nonzero_exposure": int((np.bincount(global_records.reshape(-1), minlength=rows) > 0).sum()),
        },
        "mask_position_contract": {
            "shape": [rows, sequence_tokens],
            "dtype": "torch.bool digest namespace; source masks normalized from public uint8/bool",
            "semantic_sha256": mask_digest,
            "schedule_metadata_valid_mask_semantic_sha256": schedule_metadata["valid_mask_semantic_sha256"],
            "semantic_matches_schedule": True,
            "position_summary": position_summary,
        },
    }
    check("exposure_counts", exposures["total_draws"] == 6656000 and exposures["actually_sampled_unique_global_record_postBOS_position_pairs"] == 1237403 and exposures["repeated_record_position_draws"] == 5418597 and exposures["record_batch_exposures"] == 104000, exposures)

    # Checkpoint headers and full-file hashes are checked for every point.
    state_root = Path(receipt["state"]["checkpoint_state_bindings"][0]["checkpoint"]["path"]).parent
    expected_state = {int(item["checkpoint"]["path"].rsplit("_", 1)[-1].split(".")[0]): item for item in receipt["state"]["checkpoint_state_bindings"]}
    # The split above sees checkpoint_step_000000; use the explicit receipt step.
    expected_state = {int(item["fitting_diagnostics"]["step"]): item for item in receipt["state"]["checkpoint_state_bindings"]}
    base_state_sha = receipt["state"]["starting_state"]["file"]["sha256"]
    state_audit: list[dict[str, Any]] = []
    selected_step = int(receipt["selection"]["selected_step"])
    selected_path: Path | None = None
    selected_load: dict[str, Any] | None = None
    for step in EXPECTED_GRID:
        binding = expected_state[step]
        cp = binding["checkpoint"]
        path = Path(cp["path"]).expanduser().resolve()
        actual = file_record(path, label="fixed decoder checkpoint")
        check(f"checkpoint_file_hash_{step}", actual["sha256"] == cp["sha256"] and actual["bytes"] == int(cp["bytes"]), {"declared": cp, "actual": actual})
        metadata, names, shapes = safe_header_only(path)
        metadata_expected = {
            "schema": "token-reconstruction.trr-p09-fixed-state.v1",
            "method_id": receipt["state"]["method_id"],
            "selected_step": str(step),
            "bank_manifest_sha256": receipt["state"]["bank_manifest_sha256"],
            "fit_manifest_sha256": receipt["state"]["bank_manifest_sha256"],
            "base_state_sha256": base_state_sha,
            "schedule_semantic_sha256": receipt["exposure"]["schedule_semantic_sha256"],
            "embedding_sha256": receipt["state"]["embedding_sha256"],
            "runner_state_sha256": binding["state_sha256"],
            "serialization_only": "true",
            "optimizer_state_external": "true",
        }
        metadata_match = all(metadata.get(k) == v for k, v in metadata_expected.items())
        format_match = names == EXPECTED_KEYS and shapes == EXPECTED_SHAPES
        check(f"checkpoint_header_{step}", metadata_match and format_match, {"metadata": metadata, "names": names, "shapes": shapes})
        state_audit.append({
            "step": step,
            "checkpoint": {**actual, "declared_sha256": cp["sha256"], "declared_bytes": int(cp["bytes"]), "matches_binding": True},
            "logical_state_sha256": binding["state_sha256"],
            "metadata": metadata,
            "keys": names,
            "shapes": shapes,
            "safe_open_header_only": True,
            "tensor_values_loaded_by_audit": False,
        })
        if step == selected_step:
            selected_path = path
    if selected_path is None:
        raise RuntimeError("selected checkpoint is absent")
    # Selected deployment load is explicitly CPU-only and strict; no forward.
    selected_state = load_file(str(selected_path), device="cpu")
    decoder = build_residual_mlp512(hidden_size=2048, vocabulary_size=128256, context_width=128, bottleneck_size=512)
    missing, unexpected = decoder.load_state_dict(selected_state, strict=True)
    hook = FixedPublicReadoutHook(method_id=receipt["state"]["method_id"], embedding_sha256=receipt["state"]["embedding_sha256"])
    logical = method_state_digest(decoder, hook)
    selected_binding = expected_state[selected_step]
    check("selected_strict_cpu_load", not missing and not unexpected and logical == selected_binding["state_sha256"], {"missing_keys": list(missing), "unexpected_keys": list(unexpected), "logical_state_sha256": logical, "declared": selected_binding["state_sha256"]})
    selected_load = {
        "loader": "token_reconstruction.trr0007_positionwise.build_residual_mlp512 + Module.load_state_dict(strict=True)",
        "device": "cpu",
        "architecture": {"hidden_size": 2048, "vocabulary_size": 128256, "context_width": 128, "bottleneck_size": 512},
        "selected_step": selected_step,
        "path": str(selected_path),
        "file_sha256": selected_binding["checkpoint"]["sha256"],
        "logical_state_sha256": logical,
        "tensor_key_count": len(selected_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "forward_or_scoring_performed": False,
    }
    del selected_state, decoder, hook
    gc.collect()

    # Curve, selection, validation denominators, and diagnostics are receipt
    # assertions; no fresh scoring is performed.
    curve = receipt["learning_curve"]
    check("curve_grid", [int(row["step"]) for row in curve] == EXPECTED_GRID, [row["step"] for row in curve])
    metrics = [float(row["validation"]["domain_balanced_token_accuracy"]) for row in curve]
    max_metric = max(metrics)
    winners = [int(row["step"]) for row, value in zip(curve, metrics) if value == max_metric]
    check("earliest_max_selection", selected_step == min(winners) == 13000 and receipt["selection"]["rule"] == "earliest strict maximum; step zero eligible", {"metrics": dict(zip(EXPECTED_GRID, metrics)), "max": max_metric, "winners": winners, "selected": selected_step})
    curve_rows = []
    for row in curve:
        validation = row["validation"]
        domains = validation["domains"]
        check(f"validation_denominators_{row['step']}", validation["domain_order"] == ["Finance", "Pile"] and domains["Finance"]["token_rows"] == 32512 and domains["Pile"]["token_rows"] == 16256 and domains["Finance"]["batch_count"] == 32 and domains["Pile"]["batch_count"] == 16 and validation["token_rows"] == 48768 and domains["Finance"]["compute_base_logits"] and domains["Pile"]["compute_base_logits"], {"step": row["step"], "validation": validation})
        scopes = row["state_binding"]["fitting_diagnostics"]["scopes"]
        expected_scopes = {"frozen64": {"row_count": 64, "batch_count": 8, "position_chunk_count": 19, "full_vocab_logits_rows": 6972}, "full_bank": {"row_count": 12000, "batch_count": 1500, "position_chunk_count": 3104, "full_vocab_logits_rows": 1243710}}
        for scope in scopes:
            expected = expected_scopes[scope["scope"]]
            for key, value in expected.items():
                check(f"diagnostic_{row['step']}_{scope['scope']}_{key}", int(scope[key]) == value and int(scope["metrics"][key if key != "row_count" else "token_rows"] if key in ("full_vocab_logits_rows",) else scope["metrics"].get("token_rows", scope["row_count"])) >= 0, {"scope": scope})
            if scope["scope"] == "full_bank":
                check(f"full_bank_endpoint_{row['step']}", int(row["step"]) in (0, 13000), row["step"])
        if row["step"] not in (0, 13000):
            check(f"no_intermediate_full_bank_{row['step']}", not any(scope["scope"] == "full_bank" for scope in scopes), scopes)
        check(f"selection_metric_untouched_{row['step']}", bool(row["state_binding"]["fitting_diagnostics"]["selection_metric_untouched"]), row["state_binding"]["fitting_diagnostics"])
        curve_rows.append(curve_row(row))
    cost = receipt["cost"]
    check("cost_summary", cost["optimizer_steps"] == 13000 and cost["updates"] == 13000 and cost["total_position_draws"] == 6656000 and cost["validation_checkpoint_count"] == 7 and cost["validation_token_rows_across_checkpoints"] == 341376 and cost["fitting_diagnostics"]["scope_run_count"] == 9, cost)

    prior_path = root / "experiments/TRR-P09/results/current-fixed-r3/audit.json"
    prior_record = file_record(prior_path, label="prior B0 audit cost-lineage reference")
    prior = load_json(prior_path)
    prior_lineage = prior.get("cost_lineage", {}).get("prior_lineage_reference", {})
    cost_lineage = {
        "current_run": {"scope": cost["scope"], "timing": cost["timing"], "optimizer_steps": cost["optimizer_steps"], "total_position_draws": cost["total_position_draws"], "pretraining_or_ancestor_cost": cost["pretraining_or_ancestor_cost"]},
        "prior_b0_audit_reference": prior_record,
        "prior_lineage_reference": prior_lineage,
        "accounting_note": "B1 incremental timing is reported separately; shared starting-state/pretraining components are referenced and are not silently treated as zero or added to this run wall.",
    }
    raw_receipt = file_record(receipt_path, label="B1 raw training receipt")
    source_bindings = {
        "receipt": raw_receipt,
        "source_commit": receipt.get("source_commit"),
        "command": receipt.get("command"),
        "fit_prefix_manifest": file_record(b0_binding_path, label="B0 immutable prefix binding"),
        "fit_addition_manifest": file_record(manifest_path, label="B1 fit bank manifest"),
        "support_binding": file_record(support_binding_path, label="public support binding"),
        "schedule": schedule_actual,
        "schedule_receipt": file_record(schedule_path.parent.parent / "fixed-control-b1-r3/schedule_receipt.json", label="B1 schedule receipt") if (schedule_path.parent.parent / "fixed-control-b1-r3/schedule_receipt.json").is_file() else None,
        "watchdog_finish": file_record(finish_path, label="B1 watchdog finish"),
        "watchdog_guard": file_record(guard_path, label="B1 watchdog guard"),
        "watchdog_command": file_record(watchdog_dir / "command.json", label="B1 watchdog command"),
    }
    audit_payload = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "bank": bank,
        "audit_status": "PASS",
        "audit_scope": "CPU-only immutable B1 receipt/schedule/public-mask/state audit; no new fitting, scoring, model forward, H activation deserialization, private truth, or final panel access.",
        "created_utc": started,
        "completed_utc": utc_now(),
        "check_summary": {"passed": len(checks), "total": len(checks), "failed": [], "checks": checks},
        "source_provenance": {"audit_script": str(source_script), "audit_script_sha256": sha256_file(source_script), "invocation": invocation, "repository_root": str(root), "repository_head_at_audit": git_commit(root), "runtime_receipt": raw_receipt, "source_bindings": source_bindings},
        "audit_access_boundary": {
            "public_activation_payloads_opened_by_audit": True,
            "public_attention_mask_and_position_ids_loaded": True,
            "public_payload_full_file_hashes_verified": True,
            "activation_tensor_values_loaded_by_audit": False,
            "token_id_tensor_values_loaded_by_audit": False,
            "selected_checkpoint_tensor_values_loaded_cpu": True,
            "nonselected_checkpoint_tensor_values_loaded": False,
            "embedding_payload_opened_by_audit": False,
            "model_forward_performed_by_audit": False,
            "private_or_final_truth_opened": False,
            "source_text_persisted_by_audit": False,
            "public_run_boundary": "The audited B1 run itself is the recorded public fixed-control fit/validation run; this audit did not repeat that run.",
        },
        "payload_audit": {"composed_geometry": [rows, sequence_tokens], "b0_position_contract": b0_contract, "b1_position_contract": position_summary, "mask_digest_bool": mask_digest, "payloads": payload_records, "fit_manifest": file_record(manifest_path, label="B1 fit bank manifest"), "b0_binding": file_record(b0_binding_path, label="B0 immutable prefix binding")},
        "support_audit": {"binding": file_record(support_binding_path, label="public support binding"), "bank": bank, "support_count": int(support_entry["support_count"]), "post_bos_positions": int(support_entry["post_bos_valid_positions"]), "support_digest_trr0010": support_entry["support_digest_trr0010"], "payload": support_actual},
        "schedule_audit": {"file": schedule_actual, "metadata": schedule_metadata, "keys": schedule_names, "tensor_shapes": {key: list(value.shape) for key, value in schedule.items()}, "mask_digest": mask_digest, "exposure": exposures},
        "checkpoint_audit": {"all_states": state_audit, "selected_state_deployment": selected_load, "selected_step": selected_step, "selected_checkpoint_file_sha256": selected_binding["checkpoint"]["sha256"], "selected_logical_state_sha256": selected_binding["state_sha256"]},
        "validation_audit": {"checkpoint_grid": EXPECTED_GRID, "selection_metric": receipt["selection"]["metric"], "selection_rule": receipt["selection"]["rule"], "selected_step": selected_step, "raw_curve_sha256_to_be_written": None, "curve_rows": curve_rows, "validation_denominators": {"Finance": 32512, "Pile": 16256, "total": 48768}},
        "cost_lineage": cost_lineage,
        "resource_guard_audit": {"watchdog_directory": str(watchdog_dir), "finish": finish, "guard_summary": guard_summary},
        "reproduction": {"raw_receipt_runtime_bound": True, "raw_receipt_path": str(receipt_path), "no_raw_receipt_copied_to_repository": True, "exact_audit_command": invocation},
    }
    runtime_output = Path(args.output_root).expanduser().resolve()
    repo_output = Path(args.repo_output).expanduser().resolve()
    if runtime_output.exists() or repo_output.exists():
        raise RuntimeError(f"create-only output already exists: runtime={runtime_output} repo={repo_output}")
    runtime_output.mkdir(parents=True)
    repo_output.mkdir(parents=True)

    # Full raw curve is intentionally exported separately from the compact
    # audit so no diagnostic or validation rows are silently dropped.
    raw_curve = receipt["learning_curve"]
    curve_json = {"bank": bank, "checkpoint_grid": EXPECTED_GRID, "rows": curve_rows}
    curve_csv_path = runtime_output / "learning_curve.csv"
    fieldnames = list(curve_rows[0].keys())
    with curve_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)
    write_json(runtime_output / "learning_curve.json", curve_json)
    write_json(runtime_output / "learning_curve_raw.json", raw_curve)
    write_json(runtime_output / "schedule_exposure.json", exposures)
    write_json(runtime_output / "audit_access_boundary_clarification.json", {
        "schema": "token-reconstruction.trr-p09-fixed-control-audit-access-boundary-clarification.v1",
        "task_id": TASK_ID,
        "bank": bank,
        "status": "PASS",
        "all_checkpoint_files": "All seven checkpoint files received full-file SHA-256 verification plus safe_open metadata/header and shape inspection.",
        "selected_checkpoint": "The selected step-13000 checkpoint additionally had tensor values loaded on CPU into the identical residual_mlp512 architecture and strict load_state_dict validation.",
        "selected_checkpoint_deployment": selected_load,
        "public_payload_access": "The audit opened only attention_mask and position_ids sidecars from public fitting payloads and computed full-file hashes; activations and token IDs were not deserialized.",
        "forward_or_scoring_performed": False,
        "private_or_final_truth_accessed": False,
        "source_text_persisted": False,
    })
    # Bind the curve hash into the compact audit after the curve is written.
    audit_payload["validation_audit"]["raw_curve_sha256_to_be_written"] = sha256_file(runtime_output / "learning_curve_raw.json")
    write_json(runtime_output / "audit.json", audit_payload)
    summary = f"""# TRR-P09 frozen B1 audit\n\nStatus: **PASS** ({len(checks)}/{len(checks)} checks).\n\nSelected step `{selected_step}`; checkpoint file SHA `{selected_binding['checkpoint']['sha256']}`; logical state SHA `{selected_binding['state_sha256']}`. The selected state strict-loaded on CPU into residual_mlp512 (hidden 2048, vocabulary 128256, context 128, bottleneck 512); no forward or scoring was performed.\n\nExact schedule: `{exposures['total_draws']}` valid draws, `{exposures['actually_sampled_unique_global_record_postBOS_position_pairs']}` unique valid record-position pairs, `{exposures['repeated_record_position_draws']}` repeats, `{exposures['actually_sampled_unique_records']}` records actually sampled, `{exposures['actually_batched_unique_records']}` records present in batches, `{exposures['record_batch_exposures']}` record-batch exposures. Available post-BOS positions: `{available_positions}`; separate public support: `{support_entry['support_count']}` token IDs.\n\nRaw receipt remains runtime-bound at `{receipt_path}` SHA `{raw_receipt['sha256']}`. Full raw learning curve is exported separately. Watchdog evidence remains at `{watchdog_dir}`.\n\nAudit command: `{invocation}`\nAudit script SHA: `{sha256_file(source_script)}`\n\nPublic payload access was limited to attention-mask/position sidecars and full-file hashing; no activation or token-ID tensor values, private/final truth, embedding, model forward, or source text persistence occurred during this audit.\n"""
    (runtime_output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    # Runtime handoff first; repository handoff binds compact files and leaves
    # the 7 MB raw receipt in the runtime directory.
    runtime_files = {}
    for name in ("SUMMARY.md", "audit.json", "audit_access_boundary_clarification.json", "learning_curve.csv", "learning_curve.json", "learning_curve_raw.json", "schedule_exposure.json"):
        runtime_files[name] = file_record(runtime_output / name)
    runtime_handoff = {
        "schema": "token-reconstruction.trr-p09-fixed-control-b1-audit-runtime-handoff.v1",
        "task_id": TASK_ID,
        "bank": bank,
        "status": "PASS",
        "files": runtime_files,
        "raw_receipt_runtime_binding": raw_receipt,
        "source_script": {"path": str(source_script), "sha256": sha256_file(source_script), "invocation": invocation},
        "selected_checkpoint_file_sha256": selected_binding["checkpoint"]["sha256"],
        "selected_logical_state_sha256": selected_binding["state_sha256"],
    }
    write_json(runtime_output / "handoff_manifest.json", runtime_handoff)
    # Copy only compact handoff files into the repository output.
    for name in ("SUMMARY.md", "audit.json", "audit_access_boundary_clarification.json", "learning_curve.csv", "learning_curve.json", "learning_curve_raw.json", "schedule_exposure.json"):
        data = (runtime_output / name).read_bytes()
        (repo_output / name).write_bytes(data)
    repo_files = {name: file_record(repo_output / name) for name in ("SUMMARY.md", "audit.json", "audit_access_boundary_clarification.json", "learning_curve.csv", "learning_curve.json", "learning_curve_raw.json", "schedule_exposure.json")}
    repo_handoff = {
        "schema": "token-reconstruction.trr-p09-fixed-control-b1-audit-repository-handoff.v1",
        "task_id": TASK_ID,
        "bank": bank,
        "status": "PASS",
        "committed_files": repo_files,
        "raw_receipt_runtime_binding": raw_receipt,
        "runtime_source_handoff": {"path": str(runtime_output / "handoff_manifest.json"), "sha256": sha256_file(runtime_output / "handoff_manifest.json")},
        "audit_script": {"path": str(source_script), "sha256": sha256_file(source_script), "invocation": invocation},
        "selected_checkpoint_file_sha256": selected_binding["checkpoint"]["sha256"],
        "selected_logical_state_sha256": selected_binding["state_sha256"],
        "exposure_summary": {key: exposures[key] for key in ("total_draws", "actually_sampled_unique_global_record_postBOS_position_pairs", "repeated_record_position_draws", "actually_sampled_unique_records", "actually_batched_unique_records", "record_batch_exposures", "available_postBOS_positions")},
        "source_bindings": source_bindings,
        "cost_lineage": cost_lineage,
        "reproduction_note": "Run the exact audit command recorded in audit.json against the immutable completed B1 receipt; the command is create-only and performs no model forward.",
    }
    write_json(repo_output / "repository_handoff_manifest.json", repo_handoff)
    # Audit itself is written after the runtime files are available; print the
    # selected descriptor before any caller performs optional report work.
    print(json.dumps({
        "status": "PASS",
        "bank": bank,
        "selected_step": selected_step,
        "selected_checkpoint_path": str(selected_path),
        "selected_checkpoint_file_sha256": selected_binding["checkpoint"]["sha256"],
        "selected_logical_state_sha256": selected_binding["state_sha256"],
        "unique_pairs": exposures["actually_sampled_unique_global_record_postBOS_position_pairs"],
        "repeats": exposures["repeated_record_position_draws"],
        "sampled_records": exposures["actually_sampled_unique_records"],
        "record_batch_exposures": exposures["record_batch_exposures"],
        "runtime_audit": str(runtime_output / "audit.json"),
        "repository_audit": str(repo_output / "audit.json"),
    }, sort_keys=True))
    return audit_payload, exposures, repo_handoff, curve_rows, invocation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="B1")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--watchdog-dir", required=True)
    parser.add_argument("--b0-binding", required=True)
    parser.add_argument("--b1-manifest", required=True)
    parser.add_argument("--support-binding", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-output", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args()
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
