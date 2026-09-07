#!/usr/bin/env python3
"""Truth-free STAGE1 public-base activation capture.

This is the small production adapter for the signed TRR-P09 STAGE1 plan.  It
consumes a hash-bound, CPU-resident input-shard manifest and reuses the
published cut-4 ``ContiguousPublicPrefix.forward_full`` path.  Input shards
contain only public fitting token IDs, masks, positions, and identity hashes;
the producer never opens evaluator truth or source text.

The producer has two explicit modes.  ``qualify`` runs at most three complete
B8-by-192 forwards, repeats them, and checks active outputs and the future
padding metamorphic check.  ``capture`` streams one 64-record input shard at a
time, invokes only B8 forwards, and publishes each immutable output shard
through :func:`prepare_streamed_bank.write_shard_create_only`.  Existing
complete output shards are verified and skipped before a model forward, so a
restart cannot overwrite or silently recompute a completed shard.

No public model is loaded by import, preflight, input validation, or tests.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Protocol

import torch
from safetensors import safe_open

from scripts.trr_p09.prepare_streamed_bank import (
    BankContractError,
    BankGeometry,
    HIDDEN_DTYPE,
    INPUT_SCHEMA,
    TASK_ID,
    build_bank_manifest,
    canonical_bytes,
    file_record,
    relative_file_record,
    sha256_file,
    tensor_digest,
    verify_file_record,
    write_json_create_only,
    write_shard_create_only,
)
from token_reconstruction.public_activation import (
    PaddedTokenBatch,
    PublicActivationError,
    capture_public_prefix,
)


STAGE1_PLAN_SCHEMA = "token-reconstruction.trr-p09-stage1-public-bank-plan.v1"
INPUT_MANIFEST_SCHEMA = "token-reconstruction.trr-p09-stage1-public-input.v1"
INPUT_SHARD_SCHEMA = "token-reconstruction.trr-p09-stage1-public-input-shard.v1"
CAPTURE_RECEIPT_SCHEMA = "token-reconstruction.trr-p09-stage1-public-capture.v1"
QUALIFICATION_RECEIPT_SCHEMA = "token-reconstruction.trr-p09-stage1-public-capture-qualification.v1"
FORWARD_BATCH_RECORDS = 8
SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
NEW_RECORDS = 10800
SHARD_RECORDS = 64
FULL_SHARDS = 168
FINAL_SHARD_RECORDS = 48
TOTAL_SHARDS = 169
B0_ROWS = 1200
ADDITION_ROWS = NEW_RECORDS
TARGET_RECORDS = B0_ROWS + ADDITION_ROWS
STAGE1_CONDITION = "public_base"
AUTHORIZATION_SCHEMA = "token-reconstruction.trr-p09-stage1-capture-authorization.v1"
WATCHDOG_SCHEMA = "token-reconstruction.trr-p09-stage1-capture-watchdog.v1"
GIB = 2**30
SIGNED_STAGE1_PLAN_SHA256 = "bca93d6099760791c5ecbf3be8177d250cc672c62b2177a3d72d79b934c3826c"
SIGNED_STAGE1_PLAN_BYTES = 26127
COUNTERSIGNATURE_SCHEMA = "token-reconstruction.trr-p09-stage1-plan-countersignature.v1"
PREPARED_INPUT_SCHEMA = "token-reconstruction.trr-p09-stage1-public-inputs.v1"

_INPUT_KEYS = ("attention_mask", "position_ids", "token_ids")
_SENSITIVE_KEYS = {
    "source_text",
    "text",
    "truth",
    "labels",
    "target_labels",
    "oracle",
    "private_truth",
}


class CaptureError(RuntimeError):
    """Raised when a capture contract or resource guard fails closed."""


@dataclass(frozen=True)
class CaptureCaps:
    """The signed STAGE1 cap addendum, in bytes and seconds."""

    max_reserved_gpu_bytes: int = 8 * GIB
    min_gpu_free_before_bytes: int = 8 * GIB
    min_gpu_free_during_bytes: int = 2 * GIB
    max_rss_bytes: int = 12 * GIB
    min_host_available_before_bytes: int = 10 * GIB
    min_host_available_during_bytes: int = 6 * GIB
    max_retained_bytes: int = 20 * GIB
    min_disk_free_bytes: int = 20 * GIB
    qualification_wall_seconds: float = 600.0
    production_wall_seconds: float = 7200.0

    def as_json(self) -> dict[str, int | float]:
        return {
            "max_reserved_gpu_bytes": self.max_reserved_gpu_bytes,
            "min_gpu_free_before_bytes": self.min_gpu_free_before_bytes,
            "min_gpu_free_during_bytes": self.min_gpu_free_during_bytes,
            "max_rss_bytes": self.max_rss_bytes,
            "min_host_available_before_bytes": self.min_host_available_before_bytes,
            "min_host_available_during_bytes": self.min_host_available_during_bytes,
            "max_retained_bytes": self.max_retained_bytes,
            "min_disk_free_bytes": self.min_disk_free_bytes,
            "qualification_wall_seconds": self.qualification_wall_seconds,
            "production_wall_seconds": self.production_wall_seconds,
        }


@dataclass(frozen=True)
class InputShardDescriptor:
    shard_id: int
    start: int
    stop: int
    payload_path: Path
    sidecar_path: Path
    payload_record: dict[str, Any]
    sidecar_record: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    source_start: int = 0
    source_stop: int = 0

    @property
    def count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class InputManifest:
    path: Path
    root: Path
    manifest: dict[str, Any]
    shards: tuple[InputShardDescriptor, ...]
    static_files: tuple[dict[str, Any], ...]
    expanded_row_origin: int = 0
    verified_file_record: dict[str, Any] | None = None

    @property
    def record_count(self) -> int:
        return int(self.manifest["bank"]["record_count"])

    @property
    def geometry(self) -> BankGeometry:
        value = self.manifest["geometry"]
        return BankGeometry(
            sequence_tokens=int(value["sequence_tokens"]),
            hidden_size=int(value["hidden_size"]),
            batch_records=int(value["loader_batch_records"]),
            shard_records=int(value["shard_records"]),
            hidden_dtype=str(value.get("hidden_dtype", "torch.bfloat16")),
        )


class PrefixLike(Protocol):
    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return [B, 192, 2048] hidden states."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"{label} must be a JSON object: {path}")
    return value


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _source_record(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return file_record(path, label=label)
    except (OSError, BankContractError) as exc:
        raise CaptureError(str(exc)) from exc


def _stat_bound_record(record: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Attach an internal stat snapshot for post-hash mutation checks."""
    try:
        stat = Path(path).stat()
    except OSError as exc:
        raise CaptureError(f"cannot stat hash-bound input: {path}") from exc
    value = dict(record)
    value["_stat"] = {
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    return value


def _assert_static_inputs_unchanged(input_manifest: InputManifest) -> None:
    """Check stat invariance after one hash verification and before reads."""
    for record in input_manifest.static_files:
        raw_stat = record.get("_stat")
        raw_path = record.get("path")
        if not isinstance(raw_stat, Mapping) or not isinstance(raw_path, str):
            raise CaptureError("input static-file binding lacks its stat snapshot")
        path = Path(raw_path)
        try:
            current = path.stat()
        except OSError as exc:
            raise CaptureError(f"hash-bound input disappeared: {path}") from exc
        observed = {
            "inode": int(current.st_ino),
            "size": int(current.st_size),
            "mtime_ns": int(current.st_mtime_ns),
        }
        expected = {key: int(raw_stat.get(key, -1)) for key in ("inode", "size", "mtime_ns")}
        if observed != expected:
            raise CaptureError(f"hash-bound input changed after verification: {path}")


def _relative_descriptor(root: Path, value: Mapping[str, Any], *, label: str) -> tuple[Path, dict[str, Any]]:
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise CaptureError(f"{label} path is absent")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CaptureError(f"{label} escapes input root: {candidate}") from exc
    bound = dict(value)
    bound["path"] = str(candidate)
    try:
        return candidate, verify_file_record(bound, label=label)
    except BankContractError as exc:
        raise CaptureError(str(exc)) from exc


def _verify_plan(path: Path, *, expected_sha256: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _source_record(path, label="STAGE1 plan")
    if expected_sha256 is not None and expected_sha256 != SIGNED_STAGE1_PLAN_SHA256:
        raise CaptureError("caller-supplied STAGE1 plan SHA is not the signed plan SHA")
    expected = SIGNED_STAGE1_PLAN_SHA256
    if record["sha256"] != expected:
        raise CaptureError("STAGE1 plan SHA-256 differs from the signed plan")
    if int(record["bytes"]) != SIGNED_STAGE1_PLAN_BYTES:
        raise CaptureError("STAGE1 plan byte count differs from the signed plan")
    plan = _load_json(path, label="STAGE1 plan")
    if plan.get("schema") != STAGE1_PLAN_SCHEMA or plan.get("task_id") != TASK_ID:
        raise CaptureError("STAGE1 plan schema/task identity changed")
    recipe = plan.get("stage1_deterministic_recipe")
    caps = plan.get("capture_caps_and_lease")
    if not isinstance(recipe, Mapping) or not isinstance(caps, Mapping):
        raise CaptureError("STAGE1 plan lacks deterministic recipe or cap addendum")
    expanded = recipe.get("expanded_bank")
    if not isinstance(expanded, Mapping) or int(expanded.get("new_records", -1)) != NEW_RECORDS:
        raise CaptureError("STAGE1 plan new-record count changed")
    if caps.get("condition") != STAGE1_CONDITION or int(caps.get("new_records", -1)) != NEW_RECORDS:
        raise CaptureError("STAGE1 plan capture condition/count changed")
    storage = caps.get("storage")
    if not isinstance(storage, Mapping):
        raise CaptureError("STAGE1 plan storage contract is absent")
    expected_storage = {
        "activation_dtype": "torch.bfloat16",
        "activation_geometry": ["records", SEQUENCE_TOKENS, HIDDEN_SIZE],
        "duplicate_H128_artifact": False,
        "shard_records": SHARD_RECORDS,
        "native_forward_batch": FORWARD_BATCH_RECORDS,
        "full_64_shards": FULL_SHARDS,
        "final_shard_records": FINAL_SHARD_RECORDS,
        "total_shards": TOTAL_SHARDS,
    }
    for key, expected in expected_storage.items():
        if storage.get(key) != expected:
            raise CaptureError(f"STAGE1 plan storage field changed: {key}")
    production = caps.get("production_caps")
    if not isinstance(production, Mapping):
        raise CaptureError("STAGE1 production cap contract is absent")
    expected_caps = {
        "gpu_reserved_child_bytes_max": 8 * GIB,
        "gpu_free_before_launch_bytes_min": 8 * GIB,
        "gpu_free_during_child_bytes_min": 2 * GIB,
        "child_rss_bytes_max": 12 * GIB,
        "host_available_before_launch_bytes_min": 10 * GIB,
        "host_available_during_child_bytes_min": 6 * GIB,
        "retained_capture_data_bytes_max": 20 * GIB,
        "filesystem_free_bytes_min": 20 * GIB,
        "production_wall_seconds_cap": 7200,
    }
    for key, expected in expected_caps.items():
        if int(production.get(key, -1)) != expected:
            raise CaptureError(f"STAGE1 plan cap changed: {key}")
    return plan, record


def _validate_record(row: Mapping[str, Any], *, expected_global_row: int) -> dict[str, Any]:
    if any(key in row for key in _SENSITIVE_KEYS):
        raise CaptureError("input sidecar contains source text, labels, or truth payload")
    required = (
        "record_id",
        "parent_record_id",
        "sequence_id",
        "source_record_sha256",
        "sequence_sha256",
        "active_token_count",
        "global_row",
    )
    if any(not isinstance(row.get(key), str) or not row.get(key) for key in required[:4]):
        raise CaptureError("input sidecar record lacks identity/hash fields")
    for key in ("source_record_sha256", "sequence_sha256"):
        if len(str(row[key])) != 64 or any(ch not in "0123456789abcdef" for ch in str(row[key])):
            raise CaptureError(f"input sidecar {key} is not a lowercase SHA-256")
    if int(row.get("global_row", -1)) != expected_global_row:
        raise CaptureError("input sidecar global_row is not contiguous")
    active = int(row.get("active_token_count", 0))
    if active < 2 or active > SEQUENCE_TOKENS:
        raise CaptureError("input sidecar active_token_count is outside 2..192")
    return dict(row)


def _validate_input_payload_header(path: Path, *, rows: int, geometry: BankGeometry, label: str) -> None:
    expected_shapes = {
        "token_ids": [rows, geometry.sequence_tokens],
        "attention_mask": [rows, geometry.sequence_tokens],
        "position_ids": [rows, geometry.sequence_tokens],
    }
    allowed_dtypes = {
        "token_ids": {"I32", "I64"},
        "attention_mask": {"BOOL", "U8"},
        "position_ids": {"I32", "I64"},
    }
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(_INPUT_KEYS):
                raise CaptureError(f"{label} keys differ from {list(_INPUT_KEYS)}")
            for key in _INPUT_KEYS:
                sliced = handle.get_slice(key)
                shape = [int(v) for v in sliced.get_shape()]
                dtype = str(sliced.get_dtype())
                if shape != expected_shapes[key] or dtype not in allowed_dtypes[key]:
                    raise CaptureError(f"{label} {key} header changed")
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"cannot inspect {label} header") from exc


def _full_sequence_digest(token_ids: torch.Tensor, active_count: int) -> str:
    """Hash active int32 IDs for short rows lacking the published H128 hash."""

    values = torch.as_tensor(token_ids[:active_count], dtype=torch.int32).contiguous()
    return hashlib.sha256(values.numpy().tobytes(order="C")).hexdigest()


def _normalize_prepared_row(
    row: Mapping[str, Any],
    *,
    local_index: int,
    token_ids: torch.Tensor,
) -> dict[str, Any]:
    """Adapt setup's full B0+B1 metadata to the new-row capture sidecar."""

    record_id = str(row.get("record_id", ""))
    parent = str(row.get("source_record_id") or record_id)
    rendered = str(row.get("rendered_sha256", ""))
    if len(rendered) != 64 or any(ch not in "0123456789abcdef" for ch in rendered):
        raise CaptureError("prepared input rendered identity hash is malformed")
    active = int(row.get("target_post_bos_token_count", -1)) + 1
    h128 = row.get("sequence_h128_sha256")
    sequence = str(h128) if isinstance(h128, str) and len(h128) == 64 else _full_sequence_digest(token_ids, active)
    value = dict(row)
    value.update(
        {
            "record_id": record_id,
            "parent_record_id": parent,
            "sequence_id": f"{record_id}::sequence",
            "source_record_sha256": rendered,
            "sequence_sha256": sequence,
            "active_token_count": active,
            "global_row": B0_ROWS + local_index,
            "sequence_hash_convention": "published H128 digest when available; otherwise active int32 token bytes",
        }
    )
    return _validate_record(value, expected_global_row=B0_ROWS + local_index)


def _resolve_repo_bound_path(raw_path: str, *, input_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    roots = [
        input_path.parent.resolve(),
        Path(__file__).resolve().parents[2],
        Path.cwd().resolve(),
    ]
    for root in roots:
        candidate_path = (root / candidate).resolve()
        if candidate_path.is_file():
            return candidate_path
    return (Path(__file__).resolve().parents[2] / candidate).resolve()


def _validate_plan_countersignature(
    manifest_path: Path,
    value: Mapping[str, Any],
    *,
    require_stage1: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Bind an input manifest to the exact signed plan and countersignature."""
    raw_plan = value.get("plan")
    raw_counter = value.get("countersignature")
    if not require_stage1 and raw_plan is None and raw_counter is None:
        return None, None
    if not isinstance(raw_plan, Mapping):
        raise CaptureError("input manifest lacks the signed STAGE1 plan binding")
    expected_plan = {
        "sha256": SIGNED_STAGE1_PLAN_SHA256,
        "bytes": SIGNED_STAGE1_PLAN_BYTES,
        "commit": "5bfed9ec6a7bb29a988ec0a4b1343b7745d8b81b",
    }
    for key, expected in expected_plan.items():
        if raw_plan.get(key) != expected:
            raise CaptureError(f"input manifest plan binding differs: {key}")
    if not isinstance(raw_counter, Mapping) or not isinstance(raw_counter.get("path"), str):
        raise CaptureError("input manifest lacks the signed plan countersignature path")
    counter_path = _resolve_repo_bound_path(str(raw_counter["path"]), input_path=manifest_path)
    counter_record = _source_record(counter_path, label="STAGE1 plan countersignature")
    counter = _load_json(counter_path, label="STAGE1 plan countersignature")
    if counter.get("schema") != COUNTERSIGNATURE_SCHEMA or counter.get("task_id") != TASK_ID:
        raise CaptureError("STAGE1 countersignature schema/task identity changed")
    counter_plan = counter.get("plan")
    if not isinstance(counter_plan, Mapping):
        raise CaptureError("STAGE1 countersignature lacks plan binding")
    for key, expected in expected_plan.items():
        if counter_plan.get(key) != expected:
            raise CaptureError(f"STAGE1 countersignature binding differs: {key}")
    attestation = counter.get("attestation")
    if not isinstance(attestation, Mapping):
        raise CaptureError("STAGE1 countersignature attestation is absent")
    required_attestation = {
        "root": "COUNTERSIGNED",
        "agent1": "COUNTERSIGNED",
        "status": "CPU_INPUT_COMPILATION_AUTHORIZED",
        "scope": "public-source-token-input-preparation-only",
        "model_loaded": False,
        "gpu_used": False,
        "fitting_started": False,
        "final_evaluation_truth_opened": False,
    }
    for key, expected in required_attestation.items():
        if attestation.get(key) != expected:
            raise CaptureError(f"STAGE1 countersignature attestation differs: {key}")
    return dict(raw_plan), {"file": counter_record, "record": counter}


def _load_prepared_input_manifest(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_record_count: int,
    require_stage1: bool,
    manifest_record: Mapping[str, Any],
) -> InputManifest:
    """Adapt the setup compiler's one-file B0+B1 input payload.

    The compiler intentionally writes one CPU input payload before capture.
    This adapter exposes only rows 1200:12000 as 64-row logical shards and
    uses ``source_start`` to slice the shared payload without materializing it
    repeatedly.
    """

    if value.get("status") != "CPU_INPUTS_COMPILED_NO_ACTIVATIONS":
        raise CaptureError("prepared input manifest is not a complete CPU-only input compile")
    _, counter_binding = _validate_plan_countersignature(path, value, require_stage1=True)
    boundary = value.get("truth_boundary")
    if not isinstance(boundary, Mapping) or any(bool(boundary.get(key)) for key in ("evaluation_truth_opened", "source_text_persisted", "activations_created")):
        raise CaptureError("prepared input manifest violates the truth-free boundary")
    plan = value.get("plan")
    if not isinstance(plan, Mapping) or plan.get("sha256") != SIGNED_STAGE1_PLAN_SHA256 or int(plan.get("bytes", -1)) != SIGNED_STAGE1_PLAN_BYTES:
        raise CaptureError("prepared input manifest is bound to a different STAGE1 plan")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CaptureError("prepared input manifest artifacts are absent")
    root = path.parent.resolve()
    payload_path, payload_record = _relative_descriptor(root, artifacts.get("inputs"), label="prepared stage1 input payload")
    records_path, records_record = _relative_descriptor(root, artifacts.get("records"), label="prepared stage1 input records")
    records_raw = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records_raw, list) or len(records_raw) != TARGET_RECORDS:
        raise CaptureError("prepared input records do not contain B0 plus 10,800 additions")
    try:
        with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(_INPUT_KEYS):
                raise CaptureError("prepared input payload keys changed")
            header = [int(v) for v in handle.get_slice("token_ids").get_shape()]
            if header != [TARGET_RECORDS, SEQUENCE_TOKENS]:
                raise CaptureError("prepared input payload geometry changed")
            token_ids_all = handle.get_tensor("token_ids").contiguous()
            attention_all = handle.get_tensor("attention_mask").contiguous()
            positions_all = handle.get_tensor("position_ids").contiguous()
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError("cannot inspect prepared stage1 input payload") from exc
    if tuple(attention_all.shape) != (TARGET_RECORDS, SEQUENCE_TOKENS) or tuple(positions_all.shape) != (TARGET_RECORDS, SEQUENCE_TOKENS):
        raise CaptureError("prepared input mask/position geometry changed")
    if token_ids_all.dtype not in (torch.int32, torch.int64) or attention_all.dtype not in (torch.uint8, torch.bool) or positions_all.dtype not in (torch.int32, torch.int64):
        raise CaptureError("prepared input dtypes changed")
    rows = tuple(
        _normalize_prepared_row(records_raw[B0_ROWS + local], local_index=local, token_ids=token_ids_all[B0_ROWS + local])
        for local in range(ADDITION_ROWS)
    )
    shards: list[InputShardDescriptor] = []
    static_files: list[dict[str, Any]] = [
        _stat_bound_record(_source_record(path, label="prepared input manifest"), path),
        _stat_bound_record(payload_record, payload_path),
        _stat_bound_record(records_record, records_path),
    ]
    if isinstance(counter_binding, Mapping):
        counter_file = counter_binding.get("file")
        if isinstance(counter_file, Mapping) and isinstance(counter_file.get("path"), str):
            static_files.append(_stat_bound_record(counter_file, Path(str(counter_file["path"]))))
    for shard_id, local_start in enumerate(range(0, ADDITION_ROWS, SHARD_RECORDS)):
        count = min(SHARD_RECORDS, ADDITION_ROWS - local_start)
        local_rows = rows[local_start : local_start + count]
        shards.append(
            InputShardDescriptor(
                shard_id=shard_id,
                start=local_start,
                stop=local_start + count,
                payload_path=payload_path,
                sidecar_path=records_path,
                payload_record=payload_record,
                sidecar_record=records_record,
                records=tuple(local_rows),
                source_start=B0_ROWS + local_start,
                source_stop=B0_ROWS + local_start + count,
            )
        )
    longest = max(range(ADDITION_ROWS), key=lambda index: int(rows[index]["active_token_count"]))
    qual_pairs = [(0, 0), (longest // SHARD_RECORDS, (longest % SHARD_RECORDS) // FORWARD_BATCH_RECORDS)]
    qualification = [{"shard_id": int(shard), "batch_index": int(batch)} for shard, batch in dict.fromkeys(qual_pairs)]
    _validate_qualification_selection(shards, qualification, require_stage1=True)
    synthetic = json.loads(json.dumps(value))
    synthetic.update(
        {
            "schema": INPUT_MANIFEST_SCHEMA,
            "input_source_schema": PREPARED_INPUT_SCHEMA,
            "bank": {"record_count": ADDITION_ROWS, "expanded_row_origin": B0_ROWS},
            "geometry": {"sequence_tokens": SEQUENCE_TOKENS, "hidden_size": HIDDEN_SIZE, "loader_batch_records": FORWARD_BATCH_RECORDS, "shard_records": SHARD_RECORDS, "hidden_dtype": "torch.bfloat16"},
            "shards": [],
            "qualification_batches": qualification,
            "source_payload": {"path": str(payload_path), "rows": TARGET_RECORDS, "new_row_slice": [B0_ROWS, TARGET_RECORDS]},
        }
    )
    if counter_binding is not None:
        synthetic["countersignature_binding"] = counter_binding
    return InputManifest(
        path=path,
        root=root,
        manifest=synthetic,
        shards=tuple(shards),
        static_files=tuple(static_files),
        expanded_row_origin=B0_ROWS,
        verified_file_record=dict(manifest_record),
    )


def _validate_qualification_selection(
    shards: Sequence[InputShardDescriptor],
    qualification: Any,
    *,
    require_stage1: bool,
) -> None:
    if not require_stage1:
        return
    if not isinstance(qualification, list) or not qualification or len(qualification) > 3:
        raise CaptureError("STAGE1 input manifest lacks at most-three qualification batches")
    seen: set[tuple[int, int]] = set()
    selected_active: list[int] = []
    all_active = [int(row["active_token_count"]) for shard in shards for row in shard.records]
    max_active = max(all_active, default=0)
    for entry in qualification:
        if not isinstance(entry, Mapping):
            raise CaptureError("qualification batch descriptor is malformed")
        shard_id = int(entry.get("shard_id", -1))
        batch_index = int(entry.get("batch_index", -1))
        key = (shard_id, batch_index)
        if shard_id < 0 or batch_index < 0 or key in seen:
            raise CaptureError("qualification batch descriptor is malformed or duplicated")
        if shard_id >= len(shards) or batch_index >= shards[shard_id].count // FORWARD_BATCH_RECORDS:
            raise CaptureError("qualification batch is outside the declared shard")
        seen.add(key)
        rows_for_batch = shards[shard_id].records[
            batch_index * FORWARD_BATCH_RECORDS : (batch_index + 1) * FORWARD_BATCH_RECORDS
        ]
        selected_active.extend(int(row["active_token_count"]) for row in rows_for_batch)
    if not selected_active or not any(value < SEQUENCE_TOKENS for value in selected_active):
        raise CaptureError("qualification lacks a future-padding representative")
    if max(selected_active) != max_active:
        raise CaptureError("qualification lacks the longest declared representative")


def load_input_manifest(
    path: Path,
    *,
    expected_record_count: int = NEW_RECORDS,
    require_stage1: bool = True,
) -> InputManifest:
    """Hash and validate the CPU input shards before model loading."""

    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"input manifest is unavailable: {path}")
    manifest_record = _source_record(path, label="input manifest")
    value = _load_json(path, label="input manifest")
    if value.get("schema") == PREPARED_INPUT_SCHEMA:
        if not require_stage1 or expected_record_count != NEW_RECORDS:
            raise CaptureError("prepared full-bank input is only valid for the signed STAGE1 production count")
        return _load_prepared_input_manifest(
            path,
            value,
            expected_record_count=expected_record_count,
            require_stage1=require_stage1,
            manifest_record=manifest_record,
        )
    if value.get("schema") != INPUT_MANIFEST_SCHEMA or value.get("task_id") != TASK_ID:
        raise CaptureError("input manifest schema/task identity changed")
    if value.get("condition") != STAGE1_CONDITION:
        raise CaptureError("input manifest condition is not public_base")
    _, counter_binding = _validate_plan_countersignature(path, value, require_stage1=require_stage1)
    if counter_binding is not None:
        value = dict(value)
        value["countersignature_binding"] = counter_binding
    boundary = value.get("truth_boundary")
    if not isinstance(boundary, Mapping) or any(bool(boundary.get(key)) for key in ("evaluation_truth_opened", "private_truth_opened", "source_text_persisted", "target_labels_persisted")):
        raise CaptureError("input manifest violates the truth-free capture boundary")
    geometry_value = value.get("geometry")
    if not isinstance(geometry_value, Mapping):
        raise CaptureError("input manifest geometry is absent")
    try:
        geometry = BankGeometry(
            sequence_tokens=int(geometry_value["sequence_tokens"]),
            hidden_size=int(geometry_value.get("hidden_size", HIDDEN_SIZE)),
            batch_records=int(geometry_value["loader_batch_records"]),
            shard_records=int(geometry_value["shard_records"]),
            hidden_dtype=str(geometry_value.get("hidden_dtype", "torch.bfloat16")),
        )
        geometry.validate()
    except (KeyError, TypeError, ValueError, BankContractError) as exc:
        raise CaptureError("input manifest geometry is invalid") from exc
    if geometry.sequence_tokens != SEQUENCE_TOKENS or geometry.hidden_size != HIDDEN_SIZE or geometry.batch_records != FORWARD_BATCH_RECORDS:
        raise CaptureError("input manifest uses non-STAGE1 geometry")
    bank = value.get("bank")
    if not isinstance(bank, Mapping) or int(bank.get("record_count", -1)) != expected_record_count:
        raise CaptureError("input manifest record count differs from the STAGE1 contract")
    if require_stage1 and int(bank.get("record_count", -1)) != NEW_RECORDS:
        raise CaptureError("production input manifest is not the 10,800-record STAGE1 bank")
    root = path.parent.resolve()
    expanded_row_origin = int(bank.get("expanded_row_origin", 0))
    raw_shards = value.get("shards")
    if not isinstance(raw_shards, list):
        raise CaptureError("input manifest shards are absent")
    expected_shards = (expected_record_count + geometry.shard_records - 1) // geometry.shard_records
    if require_stage1 and len(raw_shards) != TOTAL_SHARDS:
        raise CaptureError("input manifest shard count differs from 168x64 plus 48")
    shards: list[InputShardDescriptor] = []
    static_files: list[dict[str, Any]] = [_stat_bound_record(manifest_record, path)]
    if isinstance(counter_binding, Mapping):
        counter_file = counter_binding.get("file")
        if isinstance(counter_file, Mapping) and isinstance(counter_file.get("path"), str):
            static_files.append(_stat_bound_record(counter_file, Path(str(counter_file["path"]))))
    cursor = 0
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping):
            raise CaptureError(f"input shard {index} descriptor is malformed")
        try:
            shard_id = int(raw["shard_id"])
            row_range = raw["row_range"]
            start = int(row_range["start"])
            stop = int(row_range["stop"])
            count = int(row_range["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureError(f"input shard {index} row range is malformed") from exc
        if shard_id != index or start != cursor or stop != start + count or count <= 0 or count % FORWARD_BATCH_RECORDS:
            raise CaptureError(f"input shard {index} is not contiguous and batch aligned")
        if require_stage1 and count != (FINAL_SHARD_RECORDS if index == TOTAL_SHARDS - 1 else SHARD_RECORDS):
            raise CaptureError(f"input shard {index} has the wrong STAGE1 row count")
        payload_path, payload_record = _relative_descriptor(root, raw.get("payload"), label=f"input shard {index} payload")
        sidecar_path, sidecar_record = _relative_descriptor(root, raw.get("sidecar"), label=f"input shard {index} sidecar")
        _validate_input_payload_header(payload_path, rows=count, geometry=geometry, label=f"input shard {index} payload")
        sidecar = _load_json(sidecar_path, label=f"input shard {index} sidecar")
        if sidecar.get("schema") != INPUT_SHARD_SCHEMA or int(sidecar.get("shard_id", -1)) != shard_id:
            raise CaptureError(f"input shard {index} sidecar identity changed")
        if sidecar.get("row_range") != {"start": start, "stop": stop, "count": count}:
            raise CaptureError(f"input shard {index} sidecar range changed")
        records_raw = sidecar.get("records")
        if not isinstance(records_raw, list) or len(records_raw) != count:
            raise CaptureError(f"input shard {index} records are absent or incomplete")
        records: list[dict[str, Any]] = []
        for local, row in enumerate(records_raw):
            if not isinstance(row, Mapping):
                raise CaptureError(f"input shard {index} row {local} is malformed")
            normalized = _validate_record(row, expected_global_row=expanded_row_origin + start + local)
            if normalized["record_id"] in seen_ids:
                raise CaptureError(f"input record ID is duplicated: {normalized['record_id']}")
            seen_ids.add(normalized["record_id"])
            records.append(normalized)
        shards.append(
            InputShardDescriptor(
                shard_id=shard_id,
                start=start,
                stop=stop,
                payload_path=payload_path,
                sidecar_path=sidecar_path,
                payload_record=payload_record,
                sidecar_record=sidecar_record,
                records=tuple(records),
            )
        )
        static_files.extend([
            _stat_bound_record(payload_record, payload_path),
            _stat_bound_record(sidecar_record, sidecar_path),
        ])
        cursor = stop
    if len(raw_shards) != expected_shards or cursor != expected_record_count:
        raise CaptureError("input shard ranges do not cover the declared record count")
    qualification = value.get("qualification_batches")
    _validate_qualification_selection(shards, qualification, require_stage1=require_stage1)
    return InputManifest(
        path=path,
        root=root,
        manifest=value,
        shards=tuple(shards),
        static_files=tuple(static_files),
        expanded_row_origin=expanded_row_origin,
        verified_file_record=dict(manifest_record),
    )


def _validate_input_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    rows: int,
    geometry: BankGeometry,
) -> dict[str, torch.Tensor]:
    expected_shape = (rows, geometry.sequence_tokens)
    normalized: dict[str, torch.Tensor] = {}
    for key in _INPUT_KEYS:
        value = torch.as_tensor(tensors[key]).detach().cpu().contiguous()
        if tuple(value.shape) != expected_shape:
            raise CaptureError(f"input {key} shape differs from {expected_shape}")
        normalized[key] = value
    if normalized["token_ids"].dtype not in (torch.int32, torch.int64):
        raise CaptureError("input token_ids must be integer")
    if normalized["attention_mask"].dtype not in (torch.bool, torch.uint8):
        raise CaptureError("input attention_mask must be bool or uint8")
    if normalized["position_ids"].dtype not in (torch.int32, torch.int64):
        raise CaptureError("input position_ids must be integer")
    token_ids = normalized["token_ids"].to(torch.long)
    mask = normalized["attention_mask"].to(torch.bool)
    positions = normalized["position_ids"].to(torch.long)
    if token_ids.lt(0).any().item() or token_ids.ge(VOCAB_SIZE).any().item():
        raise CaptureError("input token ID is outside the public vocabulary")
    full_positions = torch.arange(geometry.sequence_tokens, dtype=torch.long).expand(rows, -1)
    padded_positions = torch.where(mask, full_positions, torch.zeros_like(full_positions))
    if not torch.equal(positions, padded_positions):
        raise CaptureError("input position IDs do not use active-prefix positions with zero padding")
    for row in range(rows):
        active_count = int(mask[row].sum().item())
        if active_count < 2 or not bool(mask[row, 0].item()):
            raise CaptureError("input sequence lacks BOS/active prefix")
        if not bool(torch.equal(mask[row, :active_count], torch.ones(active_count, dtype=torch.bool))):
            raise CaptureError("input attention mask is not a contiguous prefix")
        if int(token_ids[row, 0].item()) != BOS_TOKEN_ID:
            raise CaptureError("input sequence does not begin with the pinned BOS token")
        if active_count < geometry.sequence_tokens and not bool(token_ids[row, active_count:].eq(PAD_TOKEN_ID).all().item()):
            raise CaptureError("input padded token IDs differ from the pinned PAD token")
    normalized["attention_mask"] = mask
    normalized["position_ids"] = positions
    normalized["token_ids"] = token_ids
    return normalized


def _read_input_shard(shard: InputShardDescriptor, *, geometry: BankGeometry) -> tuple[dict[str, torch.Tensor], tuple[dict[str, Any], ...]]:
    try:
        with safe_open(str(shard.payload_path), framework="pt", device="cpu") as handle:
            source_start = shard.source_start
            source_stop = shard.source_stop if shard.source_stop else shard.count
            tensors = {
                key: handle.get_slice(key)[source_start:source_stop].contiguous()
                for key in _INPUT_KEYS
            }
    except Exception as exc:
        raise CaptureError(f"cannot load input shard payload: {shard.payload_path}") from exc
    normalized = _validate_input_tensors(tensors, rows=shard.count, geometry=geometry)
    observed_active = normalized["attention_mask"].to(torch.bool).sum(dim=1).tolist()
    expected_active = [int(row["active_token_count"]) for row in shard.records]
    if [int(value) for value in observed_active] != expected_active:
        raise CaptureError(f"input sidecar active_token_count differs from mask: shard {shard.shard_id}")
    return normalized, shard.records


def _to_padded_batch(tensors: Mapping[str, torch.Tensor]) -> PaddedTokenBatch:
    token_ids = tensors["token_ids"].to(dtype=torch.int32, device="cpu").contiguous()
    mask = tensors["attention_mask"].to(dtype=torch.bool, device="cpu").contiguous()
    positions = tensors["position_ids"].to(dtype=torch.long, device="cpu").contiguous()
    rows, width = token_ids.shape
    selectors = torch.zeros((rows, width), dtype=torch.uint8)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for row in range(rows):
        active_count = int(mask[row].sum().item())
        selectors[row, 1:active_count] = 1
        ranges.append((cursor, cursor + active_count - 1))
        cursor += active_count - 1
    padded_positions = torch.where(mask, torch.arange(width, dtype=torch.long).expand(rows, -1), torch.zeros((rows, width), dtype=torch.long))
    return PaddedTokenBatch(
        token_ids=token_ids,
        attention_mask=mask.to(dtype=torch.uint8),
        position_ids=padded_positions,
        post_bos_selector_small=selectors.clone(),
        post_bos_selector_large=selectors,
        post_bos_ranges=tuple(ranges),
    )


def _capture_batch(prefix: PrefixLike, tensors: Mapping[str, torch.Tensor], *, device: torch.device, resource_check: Any | None = None) -> torch.Tensor:
    token_ids = torch.as_tensor(tensors["token_ids"])
    if token_ids.ndim != 2 or token_ids.shape[0] != FORWARD_BATCH_RECORDS:
        raise CaptureError("public capture must receive exactly one B8 batch")
    try:
        result = capture_public_prefix(
            prefix, _to_padded_batch(tensors), device=device, batch_size=FORWARD_BATCH_RECORDS, resource_check=resource_check
        )
        if tuple(result.shape) != (FORWARD_BATCH_RECORDS, SEQUENCE_TOKENS, HIDDEN_SIZE):
            raise CaptureError("public prefix returned the wrong activation geometry")
        if result.dtype != HIDDEN_DTYPE:
            raise CaptureError("public prefix returned a non-BF16 activation tensor")
        if not bool(torch.isfinite(result.float()).all().item()):
            raise CaptureError("public prefix returned non-finite activations")
        return result.contiguous()
    except (PublicActivationError, RuntimeError) as exc:
        raise CaptureError("public prefix B8 forward failed") from exc


def _future_padding_variant(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {key: torch.as_tensor(value).clone() for key, value in tensors.items()}
    token_ids = result["token_ids"].to(torch.long)
    mask = result["attention_mask"].to(torch.bool)
    changed = False
    positions = result["position_ids"].to(torch.long)
    for row in range(token_ids.shape[0]):
        active = int(mask[row].sum().item())
        if active < token_ids.shape[1]:
            token_ids[row, active:] = PAD_TOKEN_ID - 1
            mask[row, active:] = True
            positions[row] = torch.arange(token_ids.shape[1], dtype=torch.long)
            changed = True
    if not changed:
        raise CaptureError("qualification batch has no future padding pattern")
    result["token_ids"] = token_ids
    result["attention_mask"] = mask
    result["position_ids"] = positions
    return result


def _read_batches(manifest: InputManifest, *, selectors: Sequence[Mapping[str, Any]] | None = None) -> Iterator[tuple[dict[str, torch.Tensor], tuple[dict[str, Any], ...], dict[str, Any]]]:
    _assert_static_inputs_unchanged(manifest)
    selected: set[tuple[int, int]] | None = None
    if selectors is not None:
        selected = set()
        for entry in selectors:
            selected.add((int(entry["shard_id"]), int(entry["batch_index"])))
    for shard in manifest.shards:
        if selected is not None and not any(shard.shard_id == shard_id for shard_id, _ in selected):
            continue
        _assert_static_inputs_unchanged(manifest)
        tensors, records = _read_input_shard(shard, geometry=manifest.geometry)
        for batch_index, start in enumerate(range(0, shard.count, FORWARD_BATCH_RECORDS)):
            if selected is not None and (shard.shard_id, batch_index) not in selected:
                continue
            stop = start + FORWARD_BATCH_RECORDS
            yield (
                {key: value[start:stop].contiguous() for key, value in tensors.items()},
                records[start:stop],
                {
                    "shard_id": shard.shard_id,
                    "batch_index": batch_index,
                    "local_start": shard.start + start,
                    "global_start": manifest.expanded_row_origin + shard.start + start,
                },
            )


def _read_mem_available() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, sep, raw = line.partition(":")
            if key == "MemAvailable" and sep:
                parts = raw.strip().split()
                return int(parts[0]) * (1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1)
    except (OSError, ValueError):
        return None
    return None


def _read_rss() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def retained_output_bytes(output_root: Path) -> int:
    """Return the output-root footprint without reading tensor values."""

    root = Path(output_root).expanduser().resolve()
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += int(path.stat().st_size)
        except OSError as exc:
            raise CaptureError(f"cannot stat retained capture artifact: {path}") from exc
    return total


def live_resource_snapshot(*, device: torch.device, output_root: Path, retained_bytes: int) -> dict[str, Any]:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise CaptureError("STAGE1 capture requires a readable CUDA device")
    try:
        free_gpu, total_gpu = torch.cuda.mem_get_info(device)
        reserved_gpu = torch.cuda.memory_reserved(device)
        peak_reserved_gpu = torch.cuda.max_memory_reserved(device)
        peak_allocated_gpu = torch.cuda.max_memory_allocated(device)
    except Exception as exc:
        raise CaptureError("unable to read CUDA memory guard") from exc
    host_available = _read_mem_available()
    rss = _read_rss()
    if host_available is None or rss is None:
        raise CaptureError("unable to read host memory guard")
    disk_path = Path(output_root).expanduser().resolve()
    disk_path.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(disk_path)
    return {
        "created_utc": utc_now(),
        "device": str(device),
        "gpu_free_bytes": int(free_gpu),
        "gpu_total_bytes": int(total_gpu),
        "gpu_reserved_bytes": int(reserved_gpu),
        "gpu_peak_reserved_bytes": int(peak_reserved_gpu),
        "gpu_peak_allocated_bytes": int(peak_allocated_gpu),
        "host_available_bytes": int(host_available),
        "process_rss_bytes": int(rss),
        "disk_free_bytes": int(disk.free),
        "retained_bytes": int(retained_bytes),
    }


class ResourceGuard:
    """Fail-closed inner guard; an external watchdog remains mandatory."""

    def __init__(
        self,
        *,
        device: torch.device,
        output_root: Path,
        caps: CaptureCaps | None = None,
        snapshot_fn: Any | None = None,
    ) -> None:
        self.device = device
        self.output_root = Path(output_root).expanduser().resolve()
        self.caps = caps or CaptureCaps()
        self.snapshot_fn = snapshot_fn or live_resource_snapshot
        self.max_snapshot: dict[str, Any] | None = None
        self.highwater: dict[str, int] = {}
        self.lowwater: dict[str, int] = {}

    def check(self, *, phase: str, prelaunch: bool, retained_bytes: int = 0) -> dict[str, Any]:
        snapshot = self.snapshot_fn(device=self.device, output_root=self.output_root, retained_bytes=retained_bytes)
        if int(snapshot.get("gpu_reserved_bytes", -1)) > self.caps.max_reserved_gpu_bytes:
            raise CaptureError(f"resource guard GPU reserved cap exceeded during {phase}")
        min_gpu = self.caps.min_gpu_free_before_bytes if prelaunch else self.caps.min_gpu_free_during_bytes
        min_host = self.caps.min_host_available_before_bytes if prelaunch else self.caps.min_host_available_during_bytes
        checks = {
            "gpu_free": int(snapshot.get("gpu_free_bytes", -1)) >= min_gpu,
            "host_available": int(snapshot.get("host_available_bytes", -1)) >= min_host,
            "process_rss": int(snapshot.get("process_rss_bytes", -1)) <= self.caps.max_rss_bytes,
            "disk_free": int(snapshot.get("disk_free_bytes", -1)) >= self.caps.min_disk_free_bytes,
            "retained": int(retained_bytes) <= self.caps.max_retained_bytes,
        }
        if not all(checks.values()):
            raise CaptureError(f"resource guard failed during {phase}: {checks}")
        for key in ("gpu_reserved_bytes", "gpu_peak_reserved_bytes", "gpu_peak_allocated_bytes", "process_rss_bytes", "retained_bytes"):
            if key in snapshot:
                self.highwater[key] = max(self.highwater.get(key, 0), int(snapshot[key]))
        for key in ("gpu_free_bytes", "host_available_bytes", "disk_free_bytes"):
            if key in snapshot:
                value = int(snapshot[key])
                self.lowwater[key] = min(self.lowwater.get(key, value), value)
        if self.max_snapshot is None or int(snapshot.get("gpu_reserved_bytes", 0)) > int(self.max_snapshot.get("gpu_reserved_bytes", 0)):
            self.max_snapshot = dict(snapshot)
        snapshot["phase"] = phase
        snapshot["prelaunch"] = bool(prelaunch)
        snapshot["checks"] = checks
        return snapshot


def _verify_existing_output_shard(
    *,
    output_root: Path,
    shard: InputShardDescriptor,
    geometry: BankGeometry,
    row_origin: int = 0,
) -> dict[str, Any] | None:
    final_dir = Path(output_root).expanduser().resolve() / "shards" / f"shard-{shard.shard_id:06d}"
    if not final_dir.exists():
        return None
    if final_dir.is_symlink() or not final_dir.is_dir():
        raise CaptureError(f"existing output shard is not a regular directory: {final_dir}")
    payload_path = final_dir / "payload.safetensors"
    sidecar_path = final_dir / "shard.json"
    complete_path = final_dir / "COMPLETE"
    if not payload_path.is_file() or not sidecar_path.is_file() or not complete_path.is_file():
        raise CaptureError(f"existing output shard is incomplete: {final_dir}")
    if complete_path.read_bytes() != b"\n":
        raise CaptureError(f"existing output shard completion marker changed: {final_dir}")
    sidecar = _load_json(sidecar_path, label="existing output shard sidecar")
    if sidecar.get("schema") != "token-reconstruction.trr-p09-bank-shard.v1":
        raise CaptureError(f"existing output shard schema changed: {final_dir}")
    expected_range = {"start": row_origin + shard.start, "stop": row_origin + shard.stop, "count": shard.count}
    if sidecar.get("row_range") != expected_range:
        raise CaptureError(f"existing output shard row range differs: {final_dir}")
    if sidecar.get("records") != [dict(row) for row in shard.records]:
        raise CaptureError(f"existing output shard input identities differ: {final_dir}")
    payload = sidecar.get("payload")
    if not isinstance(payload, Mapping) or payload.get("path") != "payload.safetensors":
        raise CaptureError(f"existing output shard payload binding is malformed: {final_dir}")
    actual_payload = _source_record(payload_path, label="existing output shard payload")
    if int(payload.get("bytes", -1)) != actual_payload["bytes"] or str(payload.get("sha256")) != actual_payload["sha256"]:
        raise CaptureError(f"existing output shard payload hash differs: {final_dir}")
    _validate_output_payload_header(payload_path, rows=shard.count, geometry=geometry)
    current_inputs, _ = _read_input_shard(shard, geometry=geometry)
    try:
        with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
            for key in ("token_ids", "attention_mask", "position_ids"):
                observed = handle.get_tensor(key).contiguous()
                expected = torch.as_tensor(current_inputs[key]).contiguous()
                if observed.dtype != expected.dtype or not torch.equal(observed, expected):
                    raise CaptureError(f"existing output shard {key} differs from current input: {final_dir}")
            activations = handle.get_tensor("activations").contiguous()
            if not bool(torch.isfinite(activations.float()).all().item()):
                raise CaptureError(f"existing output shard activations contain non-finite values: {final_dir}")
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"cannot compare existing output shard inputs: {final_dir}") from exc
    return {
        "shard_id": shard.shard_id,
        "path": str(final_dir.relative_to(Path(output_root).expanduser().resolve())),
        "row_range": dict(sidecar["row_range"]),
        "payload": {
            "path": str((final_dir / "payload.safetensors").relative_to(Path(output_root).expanduser().resolve())),
            "bytes": int(payload.get("bytes", 0)),
            "sha256": str(payload.get("sha256")),
        },
        "sidecar": relative_file_record(sidecar_path, root=Path(output_root), label="existing output shard sidecar"),
        "complete": relative_file_record(complete_path, root=Path(output_root), label="existing output shard COMPLETE"),
        "status": "SKIPPED_EXISTING_VERIFIED",
    }


def _validate_output_payload_header(path: Path, *, rows: int, geometry: BankGeometry) -> None:
    expected_shapes = {
        "activations": [rows, geometry.sequence_tokens, geometry.hidden_size],
        "token_ids": [rows, geometry.sequence_tokens],
        "attention_mask": [rows, geometry.sequence_tokens],
        "position_ids": [rows, geometry.sequence_tokens],
    }
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(("activations", "attention_mask", "position_ids", "token_ids")):
                raise CaptureError("existing output shard keys changed")
            expected_dtypes = {"activations": "BF16", "attention_mask": "BOOL", "position_ids": "I64", "token_ids": "I64"}
            for key, expected in expected_shapes.items():
                sliced = handle.get_slice(key)
                if [int(v) for v in sliced.get_shape()] != expected:
                    raise CaptureError(f"existing output shard {key} geometry changed")
                if str(sliced.get_dtype()) != expected_dtypes[key]:
                    raise CaptureError(f"existing output shard {key} dtype changed")
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"cannot inspect existing output shard: {path}") from exc


def _input_snapshot_bindings(input_manifest: InputManifest, plan_record: Mapping[str, Any]) -> dict[str, Any]:
    input_schema = str(input_manifest.manifest.get("input_source_schema", input_manifest.manifest.get("schema", INPUT_MANIFEST_SCHEMA)))
    bindings: dict[str, Any] = {
        "stage1_input_manifest": {
            "schema": input_schema,
            "file": dict(input_manifest.verified_file_record or _source_record(input_manifest.path, label="STAGE1 input manifest")),
        },
        "stage1_plan": {
            "schema": STAGE1_PLAN_SCHEMA,
            "file": dict(plan_record),
        },
    }
    counter = input_manifest.manifest.get("countersignature_binding")
    if isinstance(counter, Mapping):
        bindings["stage1_plan_countersignature"] = {
            "schema": COUNTERSIGNATURE_SCHEMA,
            "file": dict(counter.get("file", {})),
        }
    return bindings


def _capture_manifest_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return only immutable binding fields for restart comparison."""
    bank = manifest.get("bank") if isinstance(manifest.get("bank"), Mapping) else {}
    raw_shards = manifest.get("sharding", {}).get("shards") if isinstance(manifest.get("sharding"), Mapping) else None
    shards = []
    if isinstance(raw_shards, list):
        for item in raw_shards:
            if not isinstance(item, Mapping):
                shards.append(item)
                continue
            value = dict(item)
            value.pop("status", None)
            shards.append(value)
    return {
        "schema": manifest.get("schema"),
        "task_id": manifest.get("task_id"),
        "status": manifest.get("status"),
        "geometry": manifest.get("geometry"),
        "input_snapshots": manifest.get("input_snapshots"),
        "bank": bank,
        "shards": shards,
        "expanded_row_origin": bank.get("expanded_row_origin") if isinstance(bank, Mapping) else None,
        "global_row_range": bank.get("global_row_range") if isinstance(bank, Mapping) else None,
    }


def _build_capture_manifest(
    *,
    output_root: Path,
    geometry: BankGeometry,
    record_count: int,
    row_origin: int,
    input_snapshots: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the standard bank contract with explicit B0→B1 row translation."""
    value = build_bank_manifest(
        output_root=output_root,
        geometry=geometry,
        record_count=record_count,
        current_record_count=0,
        input_snapshots=input_snapshots,
        shards=shards,
        status="STAGE1_PUBLIC_BASE_CAPTURE_COMPLETE_NO_TRUTH",
        fixture=False,
    )
    global_start = int(row_origin)
    value["bank"].update({
        "expanded_row_origin": global_start,
        "local_row_range": {"start": 0, "stop": record_count, "count": record_count},
        "global_row_range": {"start": global_start, "stop": global_start + record_count, "count": record_count},
        "prefix_artifact": {
            "required": global_start > 0,
            "local_rows_are_virtual": global_start > 0,
            "translation": f"local capture row r maps to immutable expanded bank global_row {global_start}+r",
            "preserve_existing_prefix_rows": global_start > 0,
        },
    })
    value["loader"].update({
        "global_row_translation": {
            "local_start": 0,
            "local_stop": record_count,
            "expanded_start": global_start,
            "expanded_stop": global_start + record_count,
            "rule": "read shard row_range as expanded global_row; translate local selection by expanded_row_origin",
        },
    })
    value["artifacts"]["expanded_row_contract"] = {
        "schema": "token-reconstruction.trr-p09-expanded-row-translation.v1",
        "b0_prefix_rows": B0_ROWS,
        "new_rows": record_count,
        "global_rows": [global_start, global_start + record_count],
    }
    return value


def _verify_capture_receipt_bindings(receipt: Mapping[str, Any], bindings: Mapping[str, Any]) -> None:
    if receipt.get("status") != "CAPTURE_COMPLETE_NO_TRUTH":
        raise CaptureError("existing capture receipt is not complete")
    expected_plan = bindings.get("plan")
    expected_input = bindings.get("input_manifest")
    if not isinstance(expected_plan, Mapping) or not isinstance(expected_input, Mapping):
        return
    if receipt.get("plan", {}).get("sha256") != expected_plan.get("sha256"):
        raise CaptureError("existing capture receipt plan binding differs")
    if receipt.get("input_manifest", {}).get("sha256") != expected_input.get("sha256"):
        raise CaptureError("existing capture receipt input binding differs")
    for key in ("authorization", "watchdog"):
        expected = bindings.get(key)
        actual = receipt.get(key)
        if isinstance(expected, Mapping) and (not isinstance(actual, Mapping) or actual.get("sha256") != expected.get("sha256")):
            raise CaptureError(f"existing capture receipt {key} binding differs")
    expected_model = bindings.get("model_binding")
    actual_model = receipt.get("model_binding")
    if isinstance(expected_model, Mapping):
        if not isinstance(actual_model, Mapping) or actual_model.get("weights", {}).get("sha256") != expected_model.get("weights", {}).get("sha256"):
            raise CaptureError("existing capture receipt model binding differs")


def qualify_capture(
    *,
    prefix: PrefixLike,
    input_manifest: InputManifest,
    device: torch.device,
    guard: ResourceGuard,
    max_batches: int = 3,
) -> dict[str, Any]:
    """Run the bounded three-B8 qualification without writing activations."""

    selectors = input_manifest.manifest.get("qualification_batches")
    if not isinstance(selectors, list) or not selectors or len(selectors) > max_batches:
        raise CaptureError("qualification batches are not predeclared and bounded")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for tensors, records, location in _read_batches(input_manifest, selectors=selectors):
        if len(rows) >= max_batches:
            raise CaptureError("qualification selected more than three B8 batches")
        guard.check(phase=f"qualification_before_{len(rows)}", prelaunch=False)
        first = _capture_batch(prefix, tensors, device=device, resource_check=lambda: guard.check(phase="qualification_forward", prelaunch=False))
        repeat = _capture_batch(prefix, tensors, device=device, resource_check=lambda: guard.check(phase="qualification_repeat", prelaunch=False))
        if not torch.equal(first, repeat):
            raise CaptureError(f"qualification repeated output differs at {location}")
        variant = _future_padding_variant(tensors)
        padded = _capture_batch(prefix, variant, device=device, resource_check=lambda: guard.check(phase="qualification_padding", prelaunch=False))
        active = tensors["attention_mask"].to(torch.bool).unsqueeze(-1).expand_as(first)
        if bool(active.any().item()) and not torch.equal(first.masked_select(active), padded.masked_select(active)):
            raise CaptureError(f"qualification future-padding output differs at {location}")
        rows.append({
            **location,
            "record_ids_sha256": _canonical_digest([row["record_id"] for row in records]),
            "active_token_count": [int(value) for value in tensors["attention_mask"].to(torch.bool).sum(dim=1)],
            "repeat_torch_equal": True,
            "future_padding_active_torch_equal": True,
        })
    if not rows:
        raise CaptureError("qualification selected no batches")
    elapsed = time.perf_counter() - started
    if elapsed > guard.caps.qualification_wall_seconds:
        raise CaptureError("qualification wall cap exceeded")
    return {
        "schema": QUALIFICATION_RECEIPT_SCHEMA,
        "task_id": TASK_ID,
        "status": "QUALIFICATION_PASS",
        "condition": STAGE1_CONDITION,
        "geometry": {"records": FORWARD_BATCH_RECORDS, "sequence_tokens": SEQUENCE_TOKENS, "hidden_size": HIDDEN_SIZE, "dtype": str(HIDDEN_DTYPE)},
        "max_representative_batches": max_batches,
        "batches": rows,
        "elapsed_seconds": elapsed,
        "max_resource_snapshot": guard.max_snapshot,
        "resource_highwater": dict(guard.highwater),
        "resource_lowwater": dict(guard.lowwater),
        "truth_opened": False,
        "source_rows_persisted": False,
    }


def run_capture(
    *,
    prefix: PrefixLike,
    input_manifest: InputManifest,
    plan: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    guard: ResourceGuard,
    execution_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture all new records, resuming only verified complete shards."""

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    geometry = input_manifest.geometry
    started_utc = utc_now()
    started = time.perf_counter()
    shards: list[dict[str, Any]] = []
    skipped = 0
    created = 0
    phase_seconds = {"input_validation": 0.0, "forward": 0.0, "write": 0.0, "resume_verify": 0.0}
    retained_bytes = retained_output_bytes(output_root)
    _assert_static_inputs_unchanged(input_manifest)
    for input_shard in input_manifest.shards:
        _assert_static_inputs_unchanged(input_manifest)
        guard.check(phase=f"before_shard_{input_shard.shard_id}", prelaunch=False, retained_bytes=retained_bytes)
        resume_started = time.perf_counter()
        existing = _verify_existing_output_shard(
            output_root=output_root,
            shard=input_shard,
            geometry=geometry,
            row_origin=input_manifest.expanded_row_origin,
        )
        phase_seconds["resume_verify"] += time.perf_counter() - resume_started
        if existing is not None:
            shards.append(existing)
            skipped += 1
            retained_bytes = retained_output_bytes(output_root)
            guard.check(phase=f"after_resume_shard_{input_shard.shard_id}", prelaunch=False, retained_bytes=retained_bytes)
            continue
        tensors, records = _read_input_shard(input_shard, geometry=geometry)
        forward_started = time.perf_counter()
        activations = _capture_batch(
            prefix,
            tensors,
            device=device,
            resource_check=lambda: guard.check(phase=f"forward_shard_{input_shard.shard_id}", prelaunch=False, retained_bytes=retained_bytes),
        ) if input_shard.count == FORWARD_BATCH_RECORDS else _capture_shard_batches(
            prefix, tensors, device=device, guard=guard, phase=f"forward_shard_{input_shard.shard_id}", retained_bytes=retained_bytes
        )
        phase_seconds["forward"] += time.perf_counter() - forward_started
        write_started = time.perf_counter()
        output_tensors = {
            "activations": activations,
            "token_ids": tensors["token_ids"],
            "attention_mask": tensors["attention_mask"],
            "position_ids": tensors["position_ids"],
        }
        descriptor = write_shard_create_only(
            output_root=output_root,
            shard_id=input_shard.shard_id,
            tensors=output_tensors,
            records=records,
            geometry=geometry,
            global_start=input_manifest.expanded_row_origin + input_shard.start,
        )
        phase_seconds["write"] += time.perf_counter() - write_started
        shards.append(descriptor)
        created += 1
        retained_bytes = retained_output_bytes(output_root)
        del activations, output_tensors, tensors
        guard.check(phase=f"after_shard_{input_shard.shard_id}", prelaunch=False, retained_bytes=retained_bytes)
        if time.perf_counter() - started > guard.caps.production_wall_seconds:
            raise CaptureError("STAGE1 production wall cap exceeded")
    expected_count = input_manifest.record_count
    expected_shards = len(input_manifest.shards)
    if len(shards) != expected_shards or sum(int(item["row_range"]["count"]) for item in shards) != expected_count:
        raise CaptureError("capture did not produce every declared input shard")
    input_bindings = _input_snapshot_bindings(input_manifest, plan_record)
    bank_manifest = _build_capture_manifest(
        output_root=output_root,
        geometry=geometry,
        record_count=expected_count,
        row_origin=input_manifest.expanded_row_origin,
        input_snapshots=input_bindings,
        shards=shards,
    )
    bank_manifest_path = output_root / "bank_manifest.json"
    if bank_manifest_path.exists():
        existing_manifest = _load_json(bank_manifest_path, label="existing bank manifest")
        if _capture_manifest_bindings(existing_manifest) != _capture_manifest_bindings(bank_manifest):
            raise CaptureError("existing bank manifest immutable bindings differ")
        bank_manifest = existing_manifest
    else:
        write_json_create_only(bank_manifest_path, bank_manifest)
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": CAPTURE_RECEIPT_SCHEMA,
        "task_id": TASK_ID,
        "status": "CAPTURE_COMPLETE_NO_TRUTH",
        "condition": STAGE1_CONDITION,
        "geometry": {"records": expected_count, "sequence_tokens": SEQUENCE_TOKENS, "hidden_size": HIDDEN_SIZE, "dtype": str(HIDDEN_DTYPE)},
        "storage": {
            "new_records_only": True,
            "shards": expected_shards,
            "full_64_shards": expected_shards - 1 if expected_shards and input_manifest.shards[-1].count != SHARD_RECORDS else expected_shards,
            "final_shard_records": input_manifest.shards[-1].count if input_manifest.shards else 0,
            "duplicate_H128_artifact": False,
        },
        "execution": {
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "command": list(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "device": str(device),
            "truth_opened": False,
            "source_text_written": False,
            "model_forward_semantics": "ContiguousPublicPrefix.forward_full, complete B8x192, BF16 SDPA",
        },
        "plan": dict(plan_record),
        "input_manifest": _source_record(input_manifest.path, label="STAGE1 input manifest"),
        "phase_seconds": phase_seconds,
        "shard_counts": {"created": created, "skipped_existing_verified": skipped},
        "max_resource_snapshot": guard.max_snapshot,
        "resource_highwater": dict(guard.highwater),
        "resource_lowwater": dict(guard.lowwater),
        "bank_manifest": _source_record(bank_manifest_path, label="STAGE1 bank manifest"),
        "expanded_row_translation": {
            "local_start": 0,
            "local_stop": expected_count,
            "expanded_start": input_manifest.expanded_row_origin,
            "expanded_stop": input_manifest.expanded_row_origin + expected_count,
            "b0_prefix_rows_preserved": input_manifest.expanded_row_origin == B0_ROWS,
        },
        "countersignature": input_manifest.manifest.get("countersignature_binding"),
        "execution_bindings": dict(execution_bindings or {}),
        "authorization": dict((execution_bindings or {}).get("authorization", {})),
        "watchdog": dict((execution_bindings or {}).get("watchdog", {})),
        "model_binding": dict((execution_bindings or {}).get("model_binding", {})),
        "truth_boundary": {"truth_opened": False, "target_labels_loaded": False, "source_text_loaded": False},
    }
    receipt_path = output_root / "capture-receipt.json"
    if not receipt_path.exists():
        write_json_create_only(receipt_path, receipt)
    else:
        receipt = _load_json(receipt_path, label="existing capture receipt")
        _verify_capture_receipt_bindings(receipt, execution_bindings or {})
    return receipt


def _capture_shard_batches(
    prefix: PrefixLike,
    tensors: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    guard: ResourceGuard,
    phase: str,
    retained_bytes: int,
) -> torch.Tensor:
    rows = int(tensors["token_ids"].shape[0])
    if rows <= 0 or rows % FORWARD_BATCH_RECORDS:
        raise CaptureError("capture shard rows must be a positive multiple of B8")
    pieces: list[torch.Tensor] = []
    for start in range(0, rows, FORWARD_BATCH_RECORDS):
        stop = start + FORWARD_BATCH_RECORDS
        guard.check(phase=f"{phase}_before_batch_{start // FORWARD_BATCH_RECORDS}", prelaunch=False, retained_bytes=retained_bytes)
        pieces.append(
            _capture_batch(
                prefix,
                {key: torch.as_tensor(value[start:stop]).contiguous() for key, value in tensors.items()},
                device=device,
                resource_check=lambda: guard.check(phase=f"{phase}_batch_{start // FORWARD_BATCH_RECORDS}", prelaunch=False, retained_bytes=retained_bytes),
            )
        )
    return torch.cat(pieces, dim=0).contiguous()


def _verify_model_identity(plan: Mapping[str, Any], model_snapshot: Path) -> dict[str, Any]:
    """Bind the exact public snapshot before importing/loading transformers."""

    bound_inputs = plan.get("already_bound_inputs")
    public_model = bound_inputs.get("public_model") if isinstance(bound_inputs, Mapping) else None
    if not isinstance(public_model, Mapping):
        raise CaptureError("STAGE1 plan lacks the bound public model")
    supplied = Path(model_snapshot).expanduser().resolve()
    expected_raw = public_model.get("snapshot_path")
    if not isinstance(expected_raw, str) or supplied != Path(expected_raw).expanduser().resolve():
        raise CaptureError("model snapshot path differs from the signed public snapshot")
    revision = str(public_model.get("snapshot_revision", ""))
    if not revision or revision not in str(supplied):
        raise CaptureError("model snapshot revision is not the signed public revision")
    weight = supplied / "model.safetensors"
    expected_bytes = int(public_model.get("weight_bytes", -1))
    expected_sha = str(public_model.get("weight_sha256", ""))
    actual_weight = _source_record(weight, label="pinned public model weights")
    if actual_weight["bytes"] != expected_bytes or actual_weight["sha256"] != expected_sha:
        raise CaptureError("pinned public model weights changed")
    config = _source_record(supplied / "config.json", label="pinned public model config")
    generation = _source_record(supplied / "generation_config.json", label="pinned public generation config")
    tokenizer_revision = str(plan.get("already_bound_inputs", {}).get("tokenizer", {}).get("snapshot_revision", ""))
    tokenizer_candidates = [supplied / "tokenizer.json", supplied / "tokenizer_config.json"]
    tokenizer = next((path for path in tokenizer_candidates if path.is_file()), None)
    if tokenizer is None:
        raise CaptureError("pinned tokenizer artifact is absent from the signed snapshot")
    tokenizer_record = _source_record(tokenizer, label="pinned public tokenizer artifact")
    return {
        "model_id": str(public_model.get("model_id", "")),
        "snapshot_revision": revision,
        "tokenizer_snapshot_revision": tokenizer_revision,
        "snapshot_path": str(supplied),
        "weights": actual_weight,
        "config": config,
        "generation_config": generation,
        "tokenizer": tokenizer_record,
        "loader_settings": dict(public_model.get("loader_settings", {})),
    }


def _verify_watchdog_receipt(
    path: Path,
    *,
    mode: str,
    plan_record: Mapping[str, Any],
    input_record: Mapping[str, Any],
    caps: CaptureCaps,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _source_record(path, label="external capture watchdog receipt")
    value = _load_json(path, label="external capture watchdog receipt")
    if value.get("schema") != WATCHDOG_SCHEMA or value.get("task_id") != TASK_ID:
        raise CaptureError("external watchdog receipt schema/task identity is not bound")
    if value.get("status") not in {"ARMED", "PASS"}:
        raise CaptureError("external watchdog is not armed or passing")
    if value.get("condition") != STAGE1_CONDITION or value.get("mode") != mode:
        raise CaptureError("external watchdog condition/mode binding differs")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping):
        raise CaptureError("external watchdog lacks input bindings")
    if bindings.get("plan_sha256") != plan_record.get("sha256") or bindings.get("input_manifest_sha256") != input_record.get("sha256"):
        raise CaptureError("external watchdog is bound to a different plan or input manifest")
    command = value.get("command")
    if not command or not isinstance(command, (str, list, tuple)):
        raise CaptureError("external watchdog command is absent")
    limits = value.get("limits")
    if not isinstance(limits, Mapping):
        raise CaptureError("external watchdog limits are absent")
    wall_cap = caps.qualification_wall_seconds if mode == "qualify" else caps.production_wall_seconds
    required_limits = {
        "wall_seconds_cap": wall_cap,
        "gpu_reserved_bytes_max": caps.max_reserved_gpu_bytes,
        "gpu_free_bytes_min": caps.min_gpu_free_before_bytes,
        "host_rss_bytes_max": caps.max_rss_bytes,
    }
    for key, expected in required_limits.items():
        if int(limits.get(key, -1)) != int(expected):
            raise CaptureError(f"external watchdog limit differs: {key}")
    if value.get("post_child_race_safe") is not True:
        raise CaptureError("external watchdog does not declare post-child race-safe failure handling")
    return value, record


def _verify_qualification_receipt(
    value: Mapping[str, Any],
    *,
    manifest_path: Path,
    plan_record: Mapping[str, Any],
    input_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = value.get("qualification")
    if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
        raise CaptureError("capture authorization lacks a qualification receipt binding")
    path = _resolve_repo_bound_path(str(binding["path"]), input_path=manifest_path)
    record = _source_record(path, label="STAGE1 qualification receipt")
    expected_sha = binding.get("sha256")
    if expected_sha != record.get("sha256"):
        raise CaptureError("qualification receipt SHA differs from authorization binding")
    receipt = _load_json(path, label="STAGE1 qualification receipt")
    if receipt.get("schema") != QUALIFICATION_RECEIPT_SCHEMA or receipt.get("status") != "QUALIFICATION_PASS":
        raise CaptureError("capture authorization is not bound to a passing STAGE1 qualification")
    if receipt.get("condition") != STAGE1_CONDITION or receipt.get("truth_opened") is not False:
        raise CaptureError("qualification receipt condition/truth boundary is invalid")
    if receipt.get("plan", {}).get("sha256") != plan_record.get("sha256"):
        raise CaptureError("qualification receipt plan binding differs")
    if receipt.get("input_manifest", {}).get("sha256") != input_record.get("sha256"):
        raise CaptureError("qualification receipt input binding differs")
    return receipt, record


def _verify_authorization_receipt(
    path: Path,
    *,
    mode: str,
    plan_record: Mapping[str, Any],
    input_record: Mapping[str, Any],
    watchdog_record: Mapping[str, Any],
    caps: CaptureCaps,
    input_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _source_record(path, label="root capture authorization")
    value = _load_json(path, label="root capture authorization")
    if value.get("schema") != AUTHORIZATION_SCHEMA or value.get("task_id") != TASK_ID:
        raise CaptureError("root authorization schema/task identity is not bound")
    if value.get("status") != "AUTHORIZED" or value.get("condition") != STAGE1_CONDITION or value.get("mode") != mode:
        raise CaptureError("root authorization is not active for this capture mode")
    if value.get("plan_sha256") != plan_record.get("sha256") or value.get("input_manifest_sha256") != input_record.get("sha256"):
        raise CaptureError("root authorization is bound to a different plan or input manifest")
    if value.get("watchdog_sha256") != watchdog_record.get("sha256"):
        raise CaptureError("root authorization is bound to a different watchdog receipt")
    lease = value.get("lease")
    if not isinstance(lease, Mapping) or not isinstance(lease.get("expires_utc"), str):
        raise CaptureError("root authorization lease expiry is absent")
    try:
        expiry = datetime.fromisoformat(str(lease["expires_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureError("root authorization lease expiry is malformed") from exc
    if expiry <= datetime.now(timezone.utc):
        raise CaptureError("root authorization lease has expired")
    wall_cap = caps.qualification_wall_seconds if mode == "qualify" else caps.production_wall_seconds
    if int(lease.get("wall_seconds_cap", -1)) != int(wall_cap):
        raise CaptureError("root authorization wall cap differs")
    if mode == "capture":
        qualification, qualification_record = _verify_qualification_receipt(
            value,
            manifest_path=input_manifest_path,
            plan_record=plan_record,
            input_record=input_record,
        )
        extrapolation = value.get("production_extrapolation")
        if not isinstance(extrapolation, Mapping):
            raise CaptureError("capture authorization lacks production wall extrapolation")
        if extrapolation.get("qualification_receipt_sha256") != qualification_record.get("sha256"):
            raise CaptureError("production extrapolation is bound to a different qualification")
        if int(extrapolation.get("production_wall_seconds_cap", -1)) != int(caps.production_wall_seconds):
            raise CaptureError("production extrapolation wall cap differs")
        estimate = float(extrapolation.get("estimated_wall_seconds", -1.0))
        if estimate < 0.0 or estimate > caps.production_wall_seconds:
            raise CaptureError("production extrapolated wall time exceeds the signed cap")
        if qualification.get("status") != "QUALIFICATION_PASS":
            raise CaptureError("production extrapolation is not based on a passing qualification")
    return value, record


def _load_public_prefix(model_snapshot: Path, *, device: torch.device) -> tuple[PrefixLike, dict[str, Any]]:
    try:
        scripts_root = str(Path(__file__).resolve().parents[1])
        if scripts_root not in sys.path:
            sys.path.insert(0, scripts_root)
        import trr0004_prepare_public_activations as trusted

        prefix, snapshot, config = trusted._load_public_prefix(model_snapshot, device=device, cut_depth=4)
    except Exception as exc:
        raise CaptureError("pinned public prefix loading failed") from exc
    return prefix, {"snapshot": snapshot, "config": config, "condition": STAGE1_CONDITION}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qualify", "capture"), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--stage1-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--watchdog-receipt", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def _write_failure(path: Path, exc: BaseException, *, plan_record: Mapping[str, Any] | None, command: Sequence[str]) -> None:
    payload = {
        "schema": "token-reconstruction.trr-p09-stage1-capture-failure.v1",
        "task_id": TASK_ID,
        "status": "CAPTURE_FAILED_PRESERVED",
        "created_utc": utc_now(),
        "command": list(command),
        "plan": dict(plan_record or {}),
        "exception_chain": [],
        "truth_opened": False,
    }
    current: BaseException | None = exc
    while current is not None:
        payload["exception_chain"].append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    try:
        write_json_create_only(path, payload)
    except Exception:
        # The original exception is the actionable failure; do not mask it if
        # a prior attempt already owns the create-only failure path.
        pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan_record: dict[str, Any] | None = None
    device: torch.device | None = None
    model_loaded = False
    try:
        plan, plan_record = _verify_plan(args.stage1_plan, expected_sha256=args.plan_sha256)
        input_manifest = load_input_manifest(args.input_manifest)
        input_record = dict(input_manifest.verified_file_record or _source_record(input_manifest.path, label="STAGE1 input manifest"))
        if args.mode == "capture" and input_manifest.record_count != NEW_RECORDS:
            raise CaptureError("capture input is not the signed 10,800-record STAGE1 input")
        caps = CaptureCaps()
        watchdog, watchdog_record = _verify_watchdog_receipt(
            args.watchdog_receipt,
            mode=args.mode,
            plan_record=plan_record,
            input_record=input_record,
            caps=caps,
        )
        authorization, authorization_record = _verify_authorization_receipt(
            args.authorization_receipt,
            mode=args.mode,
            plan_record=plan_record,
            input_record=input_record,
            watchdog_record=watchdog_record,
            caps=caps,
            input_manifest_path=input_manifest.path,
        )
        model_binding = _verify_model_identity(plan, Path(args.model_snapshot).expanduser().resolve())
        device = torch.device(args.device)
        guard = ResourceGuard(device=device, output_root=output_root, caps=caps)
        guard.check(phase="pre_model_load", prelaunch=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prefix, model_evidence = _load_public_prefix(Path(args.model_snapshot).expanduser().resolve(), device=device)
        model_loaded = True
        if args.mode == "qualify":
            result = qualify_capture(prefix=prefix, input_manifest=input_manifest, device=device, guard=guard)
            result.update({
                "plan": plan_record,
                "input_manifest": input_record,
                "model_binding": model_binding,
                "model": model_evidence,
                "authorization": authorization_record,
                "watchdog": watchdog_record,
            })
            result_path = output_root / "qualification-receipt.json"
            if not result_path.exists():
                write_json_create_only(result_path, result)
        else:
            result = run_capture(
                prefix=prefix,
                input_manifest=input_manifest,
                plan=plan,
                plan_record=plan_record,
                output_root=output_root,
                device=device,
                guard=guard,
                execution_bindings={
                    "plan": plan_record,
                    "input_manifest": input_record,
                    "authorization": authorization_record,
                    "watchdog": watchdog_record,
                    "model_binding": model_binding,
                },
            )
            result["model_binding"] = model_binding
            result["model"] = model_evidence
            result["authorization"] = authorization_record
            result["watchdog"] = watchdog_record
        if args.json_output is not None:
            write_json_create_only(Path(args.json_output), result)
        return 0
    except Exception as exc:
        _write_failure(output_root / "failure.json", exc, plan_record=plan_record, command=sys.argv)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if model_loaded and device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
