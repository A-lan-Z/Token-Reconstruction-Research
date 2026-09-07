#!/usr/bin/env python3
"""Prepare and stream a public fitting activation bank without loading a model.

This module owns the TRR-P09 bank artifact contract and a small synthetic
fixture writer. Production capture is intentionally absent: preflight only
inspects geometry/resources and fixture writes deterministic synthetic tensors.
A future authorized public-forward runner must bind an input snapshot, provide
complete 8 x 192 batches, and call the create-only shard writer.

The bank payload uses the established public-activation names:
activations is H, while token_ids, attention_mask and position_ids describe
the same complete sequence. The loader streams one 8-record batch from one
shard at a time and never materializes the bank on the GPU.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid
from typing import Any, Protocol

import torch
from safetensors.torch import save_file
from safetensors import safe_open


TASK_ID = "TRR-P09"
SCHEMA = "token-reconstruction.trr-p09-streamed-public-bank.v1"
SHARD_SCHEMA = "token-reconstruction.trr-p09-bank-shard.v1"
INPUT_SCHEMA = "token-reconstruction.trr-p09-public-input-snapshot.v1"
FIXTURE_SCHEMA = "token-reconstruction.trr-p09-bank-fixture.v1"

EXPECTED_SEQUENCE_TOKENS = 192
EXPECTED_HIDDEN_SIZE = 2048
FORWARD_BATCH_RECORDS = 8
HIDDEN_DTYPE = torch.bfloat16
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
DEFAULT_SHARD_RECORDS = 64
DEFAULT_MIN_DISK_FREE_GIB = 20.0
DEFAULT_MIN_HOST_AVAILABLE_GIB = 10.0
DEFAULT_MIN_GPU_FREE_GIB = 8.0
DEFAULT_MAX_GPU_RESERVED_GIB = 6.0
DEFAULT_MAX_HOST_RSS_GIB = 16.0

_PAYLOAD_KEYS = ("activations", "attention_mask", "position_ids", "token_ids")


class BankContractError(ValueError):
    """Raised when a streamed-bank contract is incomplete or changed."""


@dataclass(frozen=True)
class BankGeometry:
    sequence_tokens: int = EXPECTED_SEQUENCE_TOKENS
    hidden_size: int = EXPECTED_HIDDEN_SIZE
    batch_records: int = FORWARD_BATCH_RECORDS
    shard_records: int = DEFAULT_SHARD_RECORDS
    hidden_dtype: str = "torch.bfloat16"

    def validate(self) -> None:
        if self.sequence_tokens != EXPECTED_SEQUENCE_TOKENS:
            raise BankContractError("P09 requires complete 192-token sequences")
        if self.hidden_size != EXPECTED_HIDDEN_SIZE:
            raise BankContractError("P09 public H width must remain 2048")
        if self.batch_records <= 0:
            raise BankContractError("loader batch_records must be positive")
        if self.shard_records <= 0 or self.shard_records % self.batch_records:
            raise BankContractError("shard_records must be a positive multiple of loader batch_records")
        if self.hidden_dtype != "torch.bfloat16":
            raise BankContractError("P09 stored H dtype must remain torch.bfloat16")

    @property
    def bytes_per_hidden_value(self) -> int:
        return 2

    @property
    def batch_hidden_bytes(self) -> int:
        return self.batch_records * self.sequence_tokens * self.hidden_size * self.bytes_per_hidden_value


@dataclass(frozen=True)
class StreamBatch:
    """One loader batch; the bank is never loaded as a whole."""

    activations: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    global_rows: tuple[int, ...]
    record_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]


class PublicForwardProvider(Protocol):
    """Adapter boundary for a separately authorized public-model capture."""

    def forward_public(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return BF16 H with shape [B, 192, 2048] for a qualified public batch."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BankContractError("value cannot be canonically encoded") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise BankContractError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise BankContractError(f"{label} is unavailable or a symlink: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def verify_file_record(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Verify one hash-bound public input descriptor without loading payloads."""

    raw_path = value.get("path")
    digest = value.get("sha256")
    try:
        path = Path(str(raw_path)).expanduser().resolve()
        expected_bytes = int(value.get("bytes"))
    except (TypeError, ValueError) as exc:
        raise BankContractError(f"{label} descriptor is malformed") from exc
    if not isinstance(raw_path, str) or not raw_path:
        raise BankContractError(f"{label} path is absent")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise BankContractError(f"{label} SHA-256 is absent or malformed")
    actual = file_record(path, label=label)
    if actual["bytes"] != expected_bytes or actual["sha256"] != digest:
        raise BankContractError(f"{label} changed after binding")
    return actual


def verify_input_snapshot_bindings(
    bindings: Mapping[str, Any],
    *,
    required_roles: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Verify required public snapshot-manifest records before any forward."""

    if not isinstance(bindings, Mapping):
        raise BankContractError("input snapshot bindings must be an object")
    verified: dict[str, dict[str, Any]] = {}
    for role in required_roles:
        entry = bindings.get(role)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("file"), Mapping):
            raise BankContractError(f"required input snapshot binding is absent: {role}")
        verified[role] = verify_file_record(entry["file"], label=role)
    return verified


def tensor_digest(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    header = canonical_bytes({"shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    return sha256_bytes(header + tensor.view(torch.uint8).numpy().tobytes(order="C"))


def record_id_digest(record_ids: Sequence[str]) -> str:
    return canonical_digest(list(record_ids))


def _read_meminfo() -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"MemAvailable": None, "MemTotal": None, "SwapFree": None}
    for line in lines:
        name, separator, raw = line.partition(":")
        if not separator:
            continue
        parts = raw.strip().split()
        try:
            value = int(parts[0])
        except (IndexError, ValueError):
            continue
        values[name] = value * 1024 if len(parts) > 1 and parts[1].lower() == "kb" else value
    return {
        "MemAvailable": values.get("MemAvailable"),
        "MemTotal": values.get("MemTotal"),
        "SwapFree": values.get("SwapFree"),
    }


def _query_gpu() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": type(exc).__name__}
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_bytes": int(fields[2]) * 1024 * 1024,
                    "memory_used_bytes": int(fields[3]) * 1024 * 1024,
                    "memory_free_bytes": int(fields[4]) * 1024 * 1024,
                }
            )
        except ValueError:
            continue
    return {"available": bool(rows), "gpus": rows, "command": command}


def estimate_storage(
    *,
    record_count: int,
    geometry: BankGeometry,
    include_hidden: bool = True,
    sidecar_bytes_per_record: int = 320,
) -> dict[str, int | float | bool]:
    geometry.validate()
    if record_count <= 0 or record_count % geometry.batch_records:
        raise BankContractError("record_count must be positive and divisible by 8")
    token_bytes = record_count * geometry.sequence_tokens * 8
    mask_bytes = record_count * geometry.sequence_tokens
    position_bytes = record_count * geometry.sequence_tokens * 8
    hidden_bytes = (
        record_count * geometry.sequence_tokens * geometry.hidden_size * geometry.bytes_per_hidden_value
        if include_hidden
        else 0
    )
    sidecar_bytes = record_count * sidecar_bytes_per_record
    total = token_bytes + mask_bytes + position_bytes + hidden_bytes + sidecar_bytes
    shard_count = (record_count + geometry.shard_records - 1) // geometry.shard_records
    return {
        "record_count": record_count,
        "hidden_bytes": hidden_bytes,
        "token_id_bytes": token_bytes,
        "attention_mask_bytes": mask_bytes,
        "position_id_bytes": position_bytes,
        "sidecar_estimate_bytes": sidecar_bytes,
        "payload_estimate_bytes": total - sidecar_bytes,
        "total_estimate_bytes": total,
        "total_estimate_gib": total / 2**30,
        "shard_count": shard_count,
        "one_batch_hidden_bytes": geometry.batch_hidden_bytes,
        "include_hidden": include_hidden,
    }


def run_preflight(
    *,
    output_root: Path,
    record_count: int,
    geometry: BankGeometry,
    include_hidden: bool = True,
    minimum_disk_free_gib: float = DEFAULT_MIN_DISK_FREE_GIB,
    minimum_host_available_gib: float = DEFAULT_MIN_HOST_AVAILABLE_GIB,
    minimum_gpu_free_gib: float = DEFAULT_MIN_GPU_FREE_GIB,
    maximum_gpu_reserved_gib: float = DEFAULT_MAX_GPU_RESERVED_GIB,
    maximum_host_rss_gib: float = DEFAULT_MAX_HOST_RSS_GIB,
) -> dict[str, Any]:
    """Perform metadata/resource checks only; this never imports a model."""

    geometry.validate()
    estimate = estimate_storage(record_count=record_count, geometry=geometry, include_hidden=include_hidden)
    target = Path(output_root).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(target.parent)
    mem = _read_meminfo()
    gpu = _query_gpu()
    gib = 2**30
    min_disk = int(minimum_disk_free_gib * gib)
    min_host = int(minimum_host_available_gib * gib)
    min_gpu = int(minimum_gpu_free_gib * gib)
    checks = {
        "disk_free": int(disk.free) >= min_disk,
        "host_available": mem["MemAvailable"] is not None and int(mem["MemAvailable"]) >= min_host,
        "gpu_available": bool(gpu.get("available")),
        "gpu_free": bool(gpu.get("available"))
        and all(int(row["memory_free_bytes"]) >= min_gpu for row in gpu["gpus"]),
    }
    return {
        "schema": "token-reconstruction.trr-p09-bank-preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "compute_started": False,
        "model_loaded": False,
        "public_forward_started": False,
        "source_rows_loaded": False,
        "truth_opened": False,
        "created_utc": utc_now(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "disk": {
                "path": str(target.parent),
                "free_bytes": int(disk.free),
                "total_bytes": int(disk.total),
                "minimum_free_bytes": min_disk,
            },
            "memory": mem,
            "gpu": gpu,
        },
        "geometry": {
            "sequence_tokens": geometry.sequence_tokens,
            "hidden_size": geometry.hidden_size,
            "loader_batch_records": geometry.batch_records,
            "shard_records": geometry.shard_records,
            "hidden_dtype": geometry.hidden_dtype,
            "full_sequence_forward": "complete [B=8, T=192] public sequence",
        },
        "storage_estimate": estimate,
        "guard_policy": {
            "minimum_disk_free_bytes": min_disk,
            "minimum_host_available_bytes": min_host,
            "minimum_gpu_free_bytes": min_gpu,
            "maximum_gpu_reserved_bytes": int(maximum_gpu_reserved_gib * gib),
            "maximum_host_rss_bytes": int(maximum_host_rss_gib * gib),
            "fail_closed": True,
            "gpu_bank_residency": "forbidden; stream one batch from one shard",
        },
        "checks": checks,
        "next_step": (
            "Bind and verify exact public input snapshots, then obtain root authorization "
            "before any public-model forward."
            if all(checks.values())
            else "Do not launch capture until every guard check passes."
        ),
    }


def _write_create_only(path: Path, data: bytes) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise BankContractError(f"create-only artifact already exists: {path}") from exc


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    _write_create_only(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n",
    )


def _write_safetensors_create_only(path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    try:
        save_file(
            {key: torch.as_tensor(value).detach().cpu().contiguous() for key, value in tensors.items()},
            str(temp),
            metadata=dict(metadata),
        )
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise BankContractError(f"create-only tensor artifact already exists: {path}") from exc
    finally:
        temp.unlink(missing_ok=True)


def _validate_payload_tensors(tensors: Mapping[str, torch.Tensor], *, rows: int, geometry: BankGeometry) -> None:
    geometry.validate()
    if set(tensors) != set(_PAYLOAD_KEYS):
        raise BankContractError(f"payload keys must be {list(_PAYLOAD_KEYS)}")
    expected = (rows, geometry.sequence_tokens)
    for key in ("token_ids", "attention_mask", "position_ids"):
        value = torch.as_tensor(tensors[key])
        if tuple(value.shape) != expected:
            raise BankContractError(f"{key} shape differs: {tuple(value.shape)} vs {expected}")
    activation = torch.as_tensor(tensors["activations"])
    if tuple(activation.shape) != (rows, geometry.sequence_tokens, geometry.hidden_size):
        raise BankContractError(f"activations shape differs: {tuple(activation.shape)}")
    if activation.dtype != HIDDEN_DTYPE:
        raise BankContractError("activations must be BF16")
    if torch.as_tensor(tensors["token_ids"]).dtype not in (torch.int32, torch.int64):
        raise BankContractError("token_ids must be int32 or int64")
    if torch.as_tensor(tensors["attention_mask"]).dtype not in (torch.bool, torch.uint8):
        raise BankContractError("attention_mask must be bool or uint8")
    if torch.as_tensor(tensors["position_ids"]).dtype not in (torch.int32, torch.int64):
        raise BankContractError("position_ids must be int32 or int64")


def _relative(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def relative_file_record(path: Path, *, root: Path, label: str) -> dict[str, Any]:
    record = file_record(path, label=label)
    record["path"] = _relative(path, root)
    return record


def write_shard_create_only(
    *,
    output_root: Path,
    shard_id: int,
    tensors: Mapping[str, torch.Tensor],
    records: Sequence[Mapping[str, Any]],
    geometry: BankGeometry,
    global_start: int,
) -> dict[str, Any]:
    """Publish one immutable shard; existing complete shards are never replaced."""

    output_root = Path(output_root).expanduser().resolve()
    geometry.validate()
    rows = len(records)
    if rows > geometry.shard_records or rows % geometry.batch_records:
        raise BankContractError("shard rows must be a positive loader-batch multiple no larger than shard_records")
    if rows <= 0:
        raise BankContractError("shard cannot be empty")
    _validate_payload_tensors(tensors, rows=rows, geometry=geometry)
    shard_name = f"shard-{shard_id:06d}"
    final_dir = output_root / "shards" / shard_name
    if final_dir.exists():
        complete = final_dir / "COMPLETE"
        sidecar = final_dir / "shard.json"
        if not complete.is_file() or not sidecar.is_file():
            raise BankContractError(f"existing shard is incomplete; refusing overwrite: {final_dir}")
        existing = _load_json(sidecar)
        expected_digests = {key: tensor_digest(tensors[key]) for key in _PAYLOAD_KEYS}
        existing_payload = existing.get("payload", {})
        existing_range = existing.get("row_range", {})
        if (final_dir / "COMPLETE").read_bytes() != b"\n":
            raise BankContractError(f"existing shard completion marker is invalid: {final_dir}")
        actual_payload = relative_file_record(
            final_dir / "payload.safetensors", root=output_root, label="existing shard payload"
        )
        payload_matches = (
            int(existing_payload.get("bytes", -1)) == actual_payload["bytes"]
            and str(existing_payload.get("sha256")) == actual_payload["sha256"]
        )
        if (
            existing.get("schema") == SHARD_SCHEMA
            and int(existing_range.get("start", -1)) == global_start
            and int(existing_range.get("count", -1)) == rows
            and existing_payload.get("tensor_digests") == expected_digests
            and existing.get("records") == [dict(row) for row in records]
            and payload_matches
        ):
            return {
                "shard_id": shard_id,
                "path": _relative(final_dir, output_root),
                "row_range": existing_range,
                "payload": {
                    "path": str(existing_payload.get("path")),
                    "bytes": int(existing_payload.get("bytes", 0)),
                    "sha256": str(existing_payload.get("sha256")),
                },
                "sidecar": relative_file_record(final_dir / "shard.json", root=output_root, label="existing shard sidecar"),
                "complete": relative_file_record(final_dir / "COMPLETE", root=output_root, label="existing shard completion marker"),
                "status": "SKIPPED_EXISTING_VERIFIED",
            }
        raise BankContractError(f"existing shard does not match requested immutable content: {final_dir}")
    staging_root = output_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / f"{shard_name}.{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        payload_path = staging_dir / "payload.safetensors"
        _write_safetensors_create_only(
            payload_path,
            tensors,
            metadata={
                "schema": SHARD_SCHEMA,
                "task_id": TASK_ID,
                "tensor_semantics": "complete_public_sequence",
            },
        )
        row_records = [dict(row) for row in records]
        if any("record_id" not in row or "sequence_id" not in row for row in row_records):
            raise BankContractError("each shard row requires record_id and sequence_id")
        shard_manifest = {
            "schema": SHARD_SCHEMA,
            "task_id": TASK_ID,
            "shard_id": shard_id,
            "row_range": {"start": global_start, "stop": global_start + rows, "count": rows},
            "geometry": {
                "sequence_tokens": geometry.sequence_tokens,
                "hidden_size": geometry.hidden_size,
                "loader_batch_records": geometry.batch_records,
                "hidden_dtype": geometry.hidden_dtype,
            },
            "payload": {
                "path": "payload.safetensors",
                "bytes": int(payload_path.stat().st_size),
                "sha256": sha256_file(payload_path),
                "keys": list(_PAYLOAD_KEYS),
                "tensor_digests": {key: tensor_digest(tensors[key]) for key in _PAYLOAD_KEYS},
            },
            "records": row_records,
            "exposure_manifest": {
                "loader_batch_start": global_start // geometry.batch_records,
                "loader_batch_count": rows // geometry.batch_records,
                "public_forward_group": "PENDING_SHARED_GEOMETRY_QUALIFICATION",
                "complete_sequence_tokens": geometry.sequence_tokens,
                "positions_are_exposed": True,
                "record_ids_only": True,
            },
            "created_utc": utc_now(),
        }
        write_json_create_only(staging_dir / "shard.json", shard_manifest)
        _write_create_only(staging_dir / "COMPLETE", b"\n")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            raise BankContractError(f"shard appeared during create-only publish: {final_dir}")
        staging_dir.rename(final_dir)
    except Exception:
        raise
    return {
        "shard_id": shard_id,
        "path": _relative(final_dir, output_root),
        "row_range": shard_manifest["row_range"],
        "payload": {
            "path": _relative(final_dir / "payload.safetensors", output_root),
            "bytes": shard_manifest["payload"]["bytes"],
            "sha256": shard_manifest["payload"]["sha256"],
        },
        "sidecar": relative_file_record(final_dir / "shard.json", root=output_root, label="shard sidecar"),
        "complete": relative_file_record(final_dir / "COMPLETE", root=output_root, label="shard completion marker"),
        "status": "CREATED",
    }


def _prefix_contract(
    *,
    records: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, torch.Tensor],
    current_count: int,
) -> dict[str, Any]:
    if current_count < 0 or current_count > len(records):
        raise BankContractError("current prefix count is outside expanded bank")
    ids = [str(row["record_id"]) for row in records]
    return {
        "record_count": current_count,
        "record_ids_sha256": record_id_digest(ids[:current_count]),
        "token_ids_sha256": tensor_digest(torch.as_tensor(tensors["token_ids"])[:current_count]),
        "attention_mask_sha256": tensor_digest(torch.as_tensor(tensors["attention_mask"])[:current_count]),
        "position_ids_sha256": tensor_digest(torch.as_tensor(tensors["position_ids"])[:current_count]),
        "activations_sha256": tensor_digest(torch.as_tensor(tensors["activations"])[:current_count]),
        "equivalence_rule": "expanded bank rows [0:current_count] must match these exact digests",
        "verification": "FIXTURE_VERIFIED",
    }


def build_bank_manifest(
    *,
    output_root: Path,
    geometry: BankGeometry,
    record_count: int,
    current_record_count: int,
    input_snapshots: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    prefix_contract: Mapping[str, Any] | None = None,
    status: str = "CAPTURE_PENDING_ROOT_AUTHORIZATION",
    fixture: bool = False,
) -> dict[str, Any]:
    geometry.validate()
    if record_count <= 0 or record_count % geometry.batch_records:
        raise BankContractError("record_count must be positive and divisible by 8")
    if current_record_count < 0 or current_record_count > record_count:
        raise BankContractError("current_record_count is outside bank")
    if current_record_count % geometry.batch_records:
        raise BankContractError("current prefix must end on a complete forward batch")
    if fixture and not prefix_contract:
        raise BankContractError("fixture manifest requires a verified prefix contract")
    shard_rows = sum(int(item["row_range"]["count"]) for item in shards)
    if shard_rows != record_count:
        raise BankContractError(f"shard rows {shard_rows} do not equal bank records {record_count}")
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": status,
        "artifact_role": "public_fitting_activation_bank",
        "create_only": True,
        "fixture": fixture,
        "truth_boundary": {
            "public_fitting_h_and_token_labels_allowed": True,
            "source_plaintext_persisted": False,
            "evaluation_source_rows_or_labels_opened": False,
            "fresh_evaluation_truth_opened": False,
        },
        "geometry": {
            "sequence_tokens": geometry.sequence_tokens,
            "hidden_size": geometry.hidden_size,
            "loader_batch_records": geometry.batch_records,
            "shard_records": geometry.shard_records,
            "hidden_dtype": geometry.hidden_dtype,
            "forward_batch_records_proposed": FORWARD_BATCH_RECORDS,
            "forward_batch_qualification": "PENDING_SHARED_GEOMETRY_QUALIFICATION",
            "forward_semantics": "complete public sequence [T=192], no truncation before forward",
        },
        "bank": {
            "record_count": record_count,
            "current_prefix_record_count": current_record_count,
            "expanded_is_strictly_larger": record_count > current_record_count,
            "current_prefix_equivalence": dict(prefix_contract or {
                "record_count": current_record_count,
                "verification": "PENDING_CAPTURE",
            }),
        },
        "input_snapshots": dict(input_snapshots),
        "input_binding": {
            "required_before_forward": not fixture,
            "roles": sorted(str(key) for key in input_snapshots),
            "descriptor_field": "file",
            "verification": "verify-once before loader iteration; stat-only thereafter",
        },
        "tensor_layout": {
            "payload_format": "safetensors",
            "keys": list(_PAYLOAD_KEYS),
            "activation_key_semantics": "activations is complete public-model H",
            "token_ids_semantics": "public fitting token labels; no final-evaluation truth",
            "mask_semantics": "contiguous prefix mask for each complete sequence",
            "position_semantics": "absolute positions 0..191 for each complete sequence",
        },
        "record_identity": {
            "sidecar_fields": [
                "record_id",
                "parent_record_id",
                "sequence_id",
                "source_record_sha256",
                "sequence_sha256",
                "active_token_count",
                "global_row",
            ],
            "plaintext_or_source_text": "not part of shard payload; separate public cache requires exact binding",
            "record_order_digest": "record_id_digest in current_prefix_equivalence",
        },
        "exposure_manifest": {
            "unit": "one complete sequence per record",
            "loader_batch_order": "global row order, proposed loader batches of 8",
            "loader_batch_count": record_count // geometry.batch_records,
            "public_forward_count": None,
            "repeated_exposure": "recorded by downstream schedule; not inferred from shard order",
            "positions_exposed": True,
        },
        "fixed_public_training_diagnostic": {
            "selection": "predeclare row indices and digest in common fit contract",
            "row_indices": None,
            "row_indices_sha256": None,
            "status": "PENDING_COMMON_CONTRACT",
        },
        "sharding": {
            "root": _relative(output_root / "shards", output_root),
            "resume_rule": "verify and skip only complete immutable shard directories; never overwrite",
            "partial_rule": "retain unique .staging directory as failure evidence; fail closed",
            "shards": [dict(item) for item in shards],
        },
        "loader": {
            "interface": "trr-p09.streamed-bank-loader.v1",
            "method": "get_records(global_indices)",
            "returns": ["activations", "token_ids", "attention_mask", "position_ids", "global_rows", "record_ids", "sequence_ids"],
            "max_resident_bank_scope": "one shard batch [8,192,2048] plus one decoder batch",
            "device_transfer": "transfer batch after CPU validation; never transfer whole bank",
        },
        "preflight_policy": {
            "capture_requires_root_authorization": True,
            "capture_requires_verified_input_snapshots": True,
            "minimum_gpu_free_gib": DEFAULT_MIN_GPU_FREE_GIB,
            "maximum_gpu_reserved_gib": DEFAULT_MAX_GPU_RESERVED_GIB,
            "minimum_host_available_gib": DEFAULT_MIN_HOST_AVAILABLE_GIB,
            "minimum_disk_free_gib": DEFAULT_MIN_DISK_FREE_GIB,
            "guard_mode": "fail_closed_external_watchdog_plus_inner_checks",
        },
        "artifacts": {
            "manifest_path": _relative(output_root / "bank_manifest.json", output_root),
            "payloads_excluded_from_git_by_default": False,
            "raw_source_or_evaluation_truth_excluded": True,
        },
        "created_utc": utc_now(),
    }


def _fixture_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "record_id": f"fixture-record-{index:04d}",
            "parent_record_id": f"fixture-parent-{index:04d}",
            "sequence_id": f"fixture-sequence-{index:04d}",
            "source_record_sha256": sha256_bytes(f"fixture-source-{index}".encode()),
            "sequence_sha256": sha256_bytes(f"fixture-sequence-{index}".encode()),
            "active_token_count": EXPECTED_SEQUENCE_TOKENS,
            "global_row": index,
        }
        for index in range(count)
    ]


def write_fixture_bank(
    output_root: Path,
    *,
    record_count: int = 16,
    current_record_count: int = 8,
    geometry: BankGeometry | None = None,
) -> dict[str, Any]:
    """Write a tiny deterministic bank used only by contract tests."""

    geometry = geometry or BankGeometry(shard_records=8)
    geometry.validate()
    if record_count <= 0 or record_count % geometry.shard_records:
        raise BankContractError("fixture count must divide shard_records")
    if current_record_count % geometry.batch_records:
        raise BankContractError("fixture prefix must end at a batch boundary")
    rows = _fixture_rows(record_count)
    token_ids = torch.arange(record_count * geometry.sequence_tokens, dtype=torch.int64).reshape(
        record_count, geometry.sequence_tokens
    )
    token_ids[:, 0] = BOS_TOKEN_ID
    attention_mask = torch.ones((record_count, geometry.sequence_tokens), dtype=torch.bool)
    position_ids = torch.arange(geometry.sequence_tokens, dtype=torch.int64).expand(record_count, -1).clone()
    activations = torch.arange(
        record_count * geometry.sequence_tokens * geometry.hidden_size,
        dtype=torch.float32,
    ).reshape(record_count, geometry.sequence_tokens, geometry.hidden_size).to(dtype=HIDDEN_DTYPE)
    tensors = {
        "activations": activations,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise BankContractError(f"fixture output must be empty/create-only: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, Any]] = []
    for shard_id, start in enumerate(range(0, record_count, geometry.shard_records)):
        stop = start + geometry.shard_records
        shard_records.append(
            write_shard_create_only(
                output_root=output_root,
                shard_id=shard_id,
                tensors={key: value[start:stop] for key, value in tensors.items()},
                records=rows[start:stop],
                geometry=geometry,
                global_start=start,
            )
        )
    input_snapshots = {
        "fixture": {
            "schema": INPUT_SCHEMA,
            "kind": "synthetic_contract_fixture",
            "model_snapshot": "not_loaded",
            "tokenizer_snapshot": "not_loaded",
            "source_rows": False,
        }
    }
    manifest = build_bank_manifest(
        output_root=output_root,
        geometry=geometry,
        record_count=record_count,
        current_record_count=current_record_count,
        input_snapshots=input_snapshots,
        shards=shard_records,
        prefix_contract=_prefix_contract(
            records=rows,
            tensors=tensors,
            current_count=current_record_count,
        ),
        status="FIXTURE_COMPLETE",
        fixture=True,
    )
    write_json_create_only(output_root / "bank_manifest.json", manifest)
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BankContractError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BankContractError(f"JSON object required: {path}")
    return value


def _resolve_bank_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BankContractError(f"{label} path is absent")
    raw = Path(value).expanduser()
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise BankContractError(f"{label} escapes bank root: {candidate}") from exc
    # Reject symlinks in the bank artifact path, including a symlinked parent.
    probe = candidate
    while probe != root.resolve():
        if probe.is_symlink():
            raise BankContractError(f"{label} is a symlink: {probe}")
        probe = probe.parent
    return candidate


def _stat_record(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise BankContractError(f"{label} disappeared or became a symlink: {path}")
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "inode": int(stat_result.st_ino),
    }


def _verify_bank_file_descriptor(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(descriptor, Mapping):
        raise BankContractError(f"{label} descriptor is absent")
    path = _resolve_bank_path(root, descriptor.get("path"), label=label)
    actual = verify_file_record(
        {**dict(descriptor), "path": str(path)},
        label=label,
    )
    return path, _stat_record(path, label=label)


def _verify_payload_metadata(
    payload_path: Path,
    *,
    row_count: int,
    geometry: BankGeometry,
    label: str,
) -> dict[str, Any]:
    expected_shapes = {
        "activations": [row_count, geometry.sequence_tokens, geometry.hidden_size],
        "token_ids": [row_count, geometry.sequence_tokens],
        "attention_mask": [row_count, geometry.sequence_tokens],
        "position_ids": [row_count, geometry.sequence_tokens],
    }
    allowed_dtypes = {
        "activations": {"BF16"},
        "token_ids": {"I32", "I64"},
        "attention_mask": {"BOOL", "U8"},
        "position_ids": {"I32", "I64"},
    }
    try:
        with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(_PAYLOAD_KEYS):
                raise BankContractError(f"{label} tensor keys changed")
            tensors: dict[str, Any] = {}
            for key in _PAYLOAD_KEYS:
                sliced = handle.get_slice(key)
                shape = [int(value) for value in sliced.get_shape()]
                dtype = str(sliced.get_dtype())
                if shape != expected_shapes[key]:
                    raise BankContractError(
                        f"{label} {key} shape changed: {shape} vs {expected_shapes[key]}"
                    )
                if dtype not in allowed_dtypes[key]:
                    raise BankContractError(f"{label} {key} dtype changed: {dtype}")
                tensors[key] = {"shape": shape, "dtype": dtype}
    except BankContractError:
        raise
    except Exception as exc:
        raise BankContractError(f"cannot inspect {label} payload header") from exc
    return tensors


def _manifest_input_bindings(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if bool(manifest.get("fixture")):
        return {"fixture": "synthetic_contract_fixture"}, []
    bindings = manifest.get("input_snapshots")
    binding_config = manifest.get("input_binding")
    if not isinstance(bindings, Mapping) or not isinstance(binding_config, Mapping):
        raise BankContractError("non-fixture bank lacks exact input snapshot bindings")
    roles = binding_config.get("roles")
    if not isinstance(roles, list) or not roles:
        raise BankContractError("non-fixture bank lacks required input snapshot roles")
    verified: dict[str, Any] = {}
    static_files: list[dict[str, Any]] = []
    for role in roles:
        entry = bindings.get(role)
        if not isinstance(entry, Mapping):
            raise BankContractError(f"input snapshot binding is absent: {role}")
        descriptor = entry.get("file") if isinstance(entry.get("file"), Mapping) else entry
        if not isinstance(descriptor, Mapping):
            raise BankContractError(f"input snapshot file descriptor is absent: {role}")
        # Public model/tokenizer manifests may be external to the bank root.
        actual = verify_file_record(descriptor, label=f"input snapshot {role}")
        path = Path(actual["path"]).resolve()
        static_files.append(_stat_record(path, label=f"input snapshot {role}"))
        verified[str(role)] = actual
    return verified, static_files


def _check_static_files(files: Sequence[Mapping[str, Any]]) -> None:
    for entry in files:
        path = Path(str(entry["path"]))
        current = _stat_record(path, label="immutable bank artifact")
        for key in ("bytes", "mtime_ns", "inode"):
            if int(current[key]) != int(entry[key]):
                raise BankContractError(f"immutable bank artifact changed after integrity gate: {path}")



def validate_bank_manifest(path: Path) -> dict[str, Any]:
    """Validate contract-level metadata without opening tensor payloads."""

    path = Path(path).expanduser().resolve()
    manifest = _load_json(path)
    if manifest.get("schema") != SCHEMA or manifest.get("task_id") != TASK_ID:
        raise BankContractError("bank manifest schema/task identity changed")
    geometry = manifest.get("geometry")
    if not isinstance(geometry, Mapping):
        raise BankContractError("bank manifest geometry is missing")
    configured = BankGeometry(
        sequence_tokens=int(geometry.get("sequence_tokens", -1)),
        hidden_size=int(geometry.get("hidden_size", -1)),
        batch_records=int(geometry.get("loader_batch_records", -1)),
        shard_records=int(geometry.get("shard_records", -1)),
        hidden_dtype=str(geometry.get("hidden_dtype", "")),
    )
    configured.validate()
    bank = manifest.get("bank")
    if not isinstance(bank, Mapping):
        raise BankContractError("bank counts are missing")
    count = int(bank.get("record_count", 0))
    current = int(bank.get("current_prefix_record_count", -1))
    if count <= 0 or count % configured.batch_records or current < 0 or current > count:
        raise BankContractError("bank counts are invalid")
    if current % configured.batch_records:
        raise BankContractError("current prefix is not batch aligned")
    if manifest.get("create_only") is not True:
        raise BankContractError("bank must be create-only")
    tensor_layout = manifest.get("tensor_layout")
    if not isinstance(tensor_layout, Mapping) or tuple(tensor_layout.get("keys", ())) != _PAYLOAD_KEYS:
        raise BankContractError("bank tensor layout changed")
    shards = manifest.get("sharding", {}).get("shards")
    if not isinstance(shards, list) or sum(int(item["row_range"]["count"]) for item in shards) != count:
        raise BankContractError("shard row counts do not cover bank")
    for item in shards:
        row_range = item.get("row_range", {})
        if int(row_range.get("count", 0)) <= 0 or int(row_range.get("count", 0)) % configured.batch_records:
            raise BankContractError("shard is not batch aligned")
    return manifest


def verify_bank_integrity(manifest_path: Path) -> dict[str, Any]:
    """Hash and validate a bank once before any schedule read.

    This is intentionally the expensive boundary: payload, sidecar, completion
    marker, manifest, and declared input snapshot files are hashed here. Later
    reads perform only inode/size/mtime checks and never rehash the bank.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = validate_bank_manifest(manifest_path)
    root = manifest_path.parent.resolve()
    geometry_data = manifest["geometry"]
    geometry = BankGeometry(
        sequence_tokens=int(geometry_data["sequence_tokens"]),
        hidden_size=int(geometry_data["hidden_size"]),
        batch_records=int(geometry_data["loader_batch_records"]),
        shard_records=int(geometry_data["shard_records"]),
        hidden_dtype=str(geometry_data["hidden_dtype"]),
    )
    manifest_digest = file_record(manifest_path, label="bank manifest")
    input_bindings, input_static = _manifest_input_bindings(manifest, root=root)
    static_files: list[dict[str, Any]] = [
        _stat_record(manifest_path, label="bank manifest"),
        *input_static,
    ]
    shards = manifest["sharding"]["shards"]
    expected_total = int(manifest["bank"]["record_count"])
    cursor = 0
    seen_record_ids: set[str] = set()
    verified_shards: list[dict[str, Any]] = []
    for shard_index, item in enumerate(shards):
        if not isinstance(item, Mapping):
            raise BankContractError(f"shard descriptor {shard_index} is malformed")
        row_range = item.get("row_range")
        if not isinstance(row_range, Mapping):
            raise BankContractError(f"shard descriptor {shard_index} lacks row range")
        try:
            start = int(row_range["start"])
            stop = int(row_range["stop"])
            count = int(row_range["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BankContractError(f"shard descriptor {shard_index} row range is malformed") from exc
        if start != cursor or stop != start + count or count <= 0:
            raise BankContractError(
                f"shard row ranges are not contiguous at {shard_index}: "
                f"expected start {cursor}, got {start}:{stop}"
            )
        if count > geometry.shard_records or count % geometry.batch_records:
            raise BankContractError(f"shard {shard_index} is not loader-batch aligned")
        payload_path, payload_static = _verify_bank_file_descriptor(
            item.get("payload"), root=root, label=f"shard {shard_index} payload"
        )
        sidecar_path, sidecar_static = _verify_bank_file_descriptor(
            item.get("sidecar"), root=root, label=f"shard {shard_index} sidecar"
        )
        complete_path, complete_static = _verify_bank_file_descriptor(
            item.get("complete"), root=root, label=f"shard {shard_index} COMPLETE marker"
        )
        shard_dir = payload_path.parent
        if _resolve_bank_path(root, item.get("path"), label=f"shard {shard_index}") != shard_dir:
            raise BankContractError(f"shard {shard_index} directory binding differs")
        if sidecar_path != shard_dir / "shard.json" or complete_path != shard_dir / "COMPLETE":
            raise BankContractError(f"shard {shard_index} sidecar/completion paths differ")
        if complete_path.read_bytes() != b"\n":
            raise BankContractError(f"shard {shard_index} COMPLETE marker is invalid")
        sidecar = _load_json(sidecar_path)
        if sidecar.get("schema") != SHARD_SCHEMA or sidecar.get("task_id") != TASK_ID:
            raise BankContractError(f"shard {shard_index} sidecar schema changed")
        if int(sidecar.get("shard_id", -1)) != int(item.get("shard_id", -2)):
            raise BankContractError(f"shard {shard_index} sidecar ID differs")
        if sidecar.get("row_range") != dict(row_range):
            raise BankContractError(f"shard {shard_index} sidecar row range differs")
        sidecar_payload = sidecar.get("payload")
        if not isinstance(sidecar_payload, Mapping):
            raise BankContractError(f"shard {shard_index} sidecar payload record is absent")
        sidecar_payload_path = (sidecar_path.parent / str(sidecar_payload.get("path", ""))).resolve()
        if sidecar_payload_path != payload_path:
            raise BankContractError(f"shard {shard_index} sidecar payload path differs")
        payload_descriptor = item["payload"]
        if (
            int(sidecar_payload.get("bytes", -1)) != int(payload_descriptor["bytes"])
            or str(sidecar_payload.get("sha256")) != str(payload_descriptor["sha256"])
            or tuple(sidecar_payload.get("keys", ())) != _PAYLOAD_KEYS
        ):
            raise BankContractError(f"shard {shard_index} sidecar payload binding differs")
        tensor_digests = sidecar_payload.get("tensor_digests")
        if not isinstance(tensor_digests, Mapping) or set(tensor_digests) != set(_PAYLOAD_KEYS):
            raise BankContractError(f"shard {shard_index} tensor digest metadata is incomplete")
        records = sidecar.get("records")
        if not isinstance(records, list) or len(records) != count:
            raise BankContractError(f"shard {shard_index} record metadata count differs")
        local_ids: set[str] = set()
        for offset, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise BankContractError(f"shard {shard_index} record {offset} is malformed")
            global_row = record.get("global_row")
            if isinstance(global_row, bool) or not isinstance(global_row, int) or global_row != start + offset:
                raise BankContractError(f"shard {shard_index} record {offset} global_row differs")
            record_id = record.get("record_id")
            sequence_id = record.get("sequence_id")
            parent_record_id = record.get("parent_record_id")
            if not all(isinstance(value, str) and value for value in (record_id, sequence_id, parent_record_id)):
                raise BankContractError(f"shard {shard_index} record {offset} identity is incomplete")
            if record_id in local_ids or record_id in seen_record_ids:
                raise BankContractError(f"duplicate record_id in bank: {record_id}")
            local_ids.add(record_id)
            seen_record_ids.add(record_id)
        tensor_headers = _verify_payload_metadata(
            payload_path,
            row_count=count,
            geometry=geometry,
            label=f"shard {shard_index}",
        )
        static_files.extend([payload_static, sidecar_static, complete_static])
        verified_shards.append(
            {
                "shard_index": shard_index,
                "row_range": {"start": start, "stop": stop, "count": count},
                "record_count": count,
                "payload": dict(payload_descriptor),
                "sidecar": dict(item["sidecar"]),
                "complete": dict(item["complete"]),
                "tensor_headers": tensor_headers,
            }
        )
        cursor = stop
    if cursor != expected_total:
        raise BankContractError(f"shards cover {cursor} rows, expected {expected_total}")
    return {
        "schema": "token-reconstruction.trr-p09-bank-integrity-gate.v1",
        "task_id": TASK_ID,
        "manifest": manifest_digest,
        "input_snapshots": input_bindings,
        "shard_count": len(verified_shards),
        "record_count": cursor,
        "verified_shards": verified_shards,
        "static_files": static_files,
        "hashes_performed_once": True,
        "stat_checks_after_gate": True,
    }



class StreamedBankLoader:
    """Read one complete public sequence batch at a time from immutable shards."""

    interface = "trr-p09.streamed-bank-loader.v1"

    def __init__(self, manifest_path: Path, *, device: str = "cpu") -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = validate_bank_manifest(self.manifest_path)
        self._integrity = verify_bank_integrity(self.manifest_path)
        self.root = self.manifest_path.parent
        self.geometry = BankGeometry(
            sequence_tokens=int(self.manifest["geometry"]["sequence_tokens"]),
            hidden_size=int(self.manifest["geometry"]["hidden_size"]),
            batch_records=int(self.manifest["geometry"]["loader_batch_records"]),
            shard_records=int(self.manifest["geometry"]["shard_records"]),
            hidden_dtype=str(self.manifest["geometry"]["hidden_dtype"]),
        )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BankContractError("requested loader CUDA device is unavailable")
        self.device = torch.device(device)

    def _assert_integrity_current(self) -> None:
        _check_static_files(self._integrity["static_files"])

    def _read_shard(self, item: Mapping[str, Any]) -> Iterator[StreamBatch]:
        self._assert_integrity_current()
        relative = item.get("payload", {}).get("path")
        if not isinstance(relative, str):
            raise BankContractError("shard payload path is missing")
        payload_path = _resolve_bank_path(self.root, relative, label="loader shard payload")
        shard_json = payload_path.parent / "shard.json"
        complete = payload_path.parent / "COMPLETE"
        if not shard_json.is_file() or not complete.is_file():
            raise BankContractError(f"shard is not complete: {payload_path.parent}")
        shard = _load_json(shard_json)
        if shard.get("schema") != SHARD_SCHEMA:
            raise BankContractError("shard schema changed")
        rows = shard.get("records")
        if not isinstance(rows, list) or not rows:
            raise BankContractError("shard records are missing")
        row_count = len(rows)
        if row_count % self.geometry.batch_records:
            raise BankContractError("shard row count is not batch aligned")
        try:
            with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != set(_PAYLOAD_KEYS):
                    raise BankContractError("shard tensor keys changed")
                for start in range(0, row_count, self.geometry.batch_records):
                    stop = start + self.geometry.batch_records
                    tensors = {
                        key: handle.get_slice(key)[start:stop].contiguous()
                        for key in _PAYLOAD_KEYS
                    }
                    _validate_payload_tensors(tensors, rows=self.geometry.batch_records, geometry=self.geometry)
                    mask = tensors["attention_mask"].to(dtype=torch.bool)
                    if not bool(mask[:, 0].all().item()):
                        raise BankContractError("a public sequence has no BOS/first position")
                    positions = tensors["position_ids"].to(dtype=torch.long)
                    expected = torch.arange(self.geometry.sequence_tokens, dtype=torch.long).expand(
                        self.geometry.batch_records, -1
                    )
                    if not torch.equal(positions, expected):
                        raise BankContractError("position_ids are not the complete 0..191 sequence")
                    local_rows = rows[start:stop]
                    batch = StreamBatch(
                        activations=tensors["activations"].to(self.device),
                        token_ids=tensors["token_ids"].to(self.device),
                        attention_mask=mask.to(self.device),
                        position_ids=positions.to(self.device),
                        global_rows=tuple(int(row["global_row"]) for row in local_rows),
                        record_ids=tuple(str(row["record_id"]) for row in local_rows),
                        sequence_ids=tuple(str(row["sequence_id"]) for row in local_rows),
                    )
                    yield batch
        except BankContractError:
            raise
        except Exception as exc:
            raise BankContractError(f"cannot read shard payload: {payload_path}") from exc

    def get_records(self, global_indices: Sequence[int]) -> StreamBatch:
        """Load arbitrary schedule rows across shards, preserving order/repeats.

        This is the training-loop interface.  A schedule may request rows in a
        different order, cross shard boundaries, or repeat a row.  Only the
        smallest contiguous slice needed from one shard is opened at a time.
        """

        self._assert_integrity_current()
        requested = [int(index) for index in global_indices]
        if not requested:
            raise BankContractError("get_records requires at least one row")
        total = int(self.manifest["bank"]["record_count"])
        if any(index < 0 or index >= total for index in requested):
            raise BankContractError("schedule row is outside the bank")
        shards = list(self.manifest["sharding"]["shards"])
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, global_index in enumerate(requested):
            matches = []
            for shard_index, item in enumerate(shards):
                row_range = item["row_range"]
                if int(row_range["start"]) <= global_index < int(row_range["stop"]):
                    matches.append((shard_index, global_index - int(row_range["start"])))
                    break
            if not matches:
                raise BankContractError(f"no shard covers schedule row {global_index}")
            shard_index, local_index = matches[0]
            grouped.setdefault(shard_index, []).append((output_index, local_index))

        output_tensors: dict[str, list[torch.Tensor | None]] = {key: [None] * len(requested) for key in _PAYLOAD_KEYS}
        output_rows: list[Mapping[str, Any] | None] = [None] * len(requested)
        for shard_index, positions in grouped.items():
            self._assert_integrity_current()
            item = shards[shard_index]
            payload_path = _resolve_bank_path(self.root, item["payload"]["path"], label="scheduled shard payload")
            shard = _load_json(payload_path.parent / "shard.json")
            rows = shard.get("records")
            if not isinstance(rows, list):
                raise BankContractError("shard records are missing")
            local_values = [local for _, local in positions]
            lo, hi = min(local_values), max(local_values) + 1
            try:
                with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != set(_PAYLOAD_KEYS):
                        raise BankContractError("shard tensor keys changed")
                    window = {
                        key: handle.get_slice(key)[lo:hi].contiguous()
                        for key in _PAYLOAD_KEYS
                    }
            except BankContractError:
                raise
            except Exception as exc:
                raise BankContractError(f"cannot read scheduled shard payload: {payload_path}") from exc
            _validate_payload_tensors(window, rows=hi - lo, geometry=self.geometry)
            for output_index, local_index in positions:
                for key in _PAYLOAD_KEYS:
                    output_tensors[key][output_index] = window[key][local_index - lo]
                output_rows[output_index] = rows[local_index]

        tensors = {
            key: torch.stack([value for value in values if value is not None], dim=0)
            for key, values in output_tensors.items()
        }
        if any(value is None for value in output_rows):
            raise BankContractError("scheduled row metadata is incomplete")
        mask = tensors["attention_mask"].to(dtype=torch.bool)
        positions = tensors["position_ids"].to(dtype=torch.long)
        expected = torch.arange(self.geometry.sequence_tokens, dtype=torch.long).expand(len(requested), -1)
        if not bool(mask[:, 0].all().item()) or not torch.equal(positions, expected):
            raise BankContractError("scheduled rows do not contain complete public sequences")
        rows = [value for value in output_rows if value is not None]
        return StreamBatch(
            activations=tensors["activations"].to(self.device),
            token_ids=tensors["token_ids"].to(self.device),
            attention_mask=mask.to(self.device),
            position_ids=positions.to(self.device),
            global_rows=tuple(requested),
            record_ids=tuple(str(row["record_id"]) for row in rows),
            sequence_ids=tuple(str(row["sequence_id"]) for row in rows),
        )

    def iter_batches(self) -> Iterator[StreamBatch]:
        """Yield sequential convenience batches; schedules should call get_records."""

        self._assert_integrity_current()
        shards = self.manifest["sharding"]["shards"]
        for item in shards:
            self._assert_integrity_current()
            yield from self._read_shard(item)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "fixture"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=12000)
    parser.add_argument("--current-records", type=int, default=1200)
    parser.add_argument("--shard-records", type=int, default=DEFAULT_SHARD_RECORDS)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--no-hidden", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    geometry = BankGeometry(shard_records=args.shard_records)
    if args.mode == "preflight":
        result = run_preflight(
            output_root=args.output_root,
            record_count=args.records,
            geometry=geometry,
            include_hidden=not args.no_hidden,
        )
    else:
        result = write_fixture_bank(
            args.output_root,
            record_count=args.records,
            current_record_count=args.current_records,
            geometry=geometry,
        )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        write_json_create_only(args.json_output, result)
    else:
        print(serialized, end="")
    return 0 if result.get("status") not in {"BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
