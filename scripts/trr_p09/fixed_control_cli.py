#!/usr/bin/env python3
"""Run the TRR-P09 fixed public-readout continuation through the shared runner.

The production path is deliberately narrow: it verifies the corrected B0/B1
streamed-bank manifests, derives the signed 512-position schedule and common
checkpoint grid from their metadata, binds the sole published P09 starting
state and public embedding, then calls :func:`run_training` once.  Public
validation labels are metadata-only and are joined by domain before the first
checkpoint.  Stage-3 agreement, largest-cell qualification, and an external
watchdog receipt are required before any model or embedding payload is loaded.

``--synthetic`` is a model/data-free smoke path for serialization and runner
integration tests.  It uses the same caller/runner/checkpoint interfaces with
small in-memory tensors and never reads a public asset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

# Direct script execution from the repository root remains reproducible.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.trr_p09.fixed_control_caller import (  # noqa: E402
    DEFAULT_VALIDATION_DOMAINS,
    build_fixed_control_receipt,
    inherited_schedule_steps,
    join_public_validation_labels,
    load_serialized_schedule,
    make_domain_validation_callback,
    make_fixed_checkpoint_callback,
    compose_checkpoint_callbacks,
    make_fitting_metric_callback,
    signed_p09_checkpoint_grid,
    write_fixed_control_receipt,
)
from scripts.trr_p09.fixed_control_runner import (  # noqa: E402
    DOMAIN_BALANCED_SELECTION_METRIC,
    RandomAccessLoaderSource,
    RunnerConfig,
    run_training,
    write_create_only_json,
)
from scripts.trr_p09.b0_immutable_loader import (  # noqa: E402
    B0ImmutableLoader,
    CombinedB0StreamedBankLoader,
)
from scripts.trr_p09.public_validation_loader import (  # noqa: E402
    PublicValidationLoader,
    PublicValidationLoaderError,
)
from scripts.trr_p09.prepare_streamed_bank import (  # noqa: E402
    StreamBatch,
    StreamedBankLoader,
    file_record,
    tensor_digest,
    sha256_file,
    validate_bank_manifest,
)
from token_reconstruction.trr0007_positionwise import (  # noqa: E402
    DEFAULT_BOTTLENECK_SIZE,
    RESIDUAL_MLP_METHOD_ID,
    load_positionwise_model_state,
)
from token_reconstruction.trr_p09_fixed_control_adapter import (  # noqa: E402
    AssetBinding,
    BankContract,
    FixedPublicReadoutHook,
    contract_digest,
)


TASK_ID = "TRR-P09"
METHOD_ID = "continued_fixed_readout"
EXPECTED_STATE_SHA256 = "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14"
EXPECTED_EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
EXPECTED_SELECTION_SUPPLEMENT_SHA256 = "760adf507438111b03c7308740e6666b0961088e6fc8868f999240d58d2814f7"
EXPECTED_RECORD_BATCH_SIZE = 8
EXPECTED_POSITION_BUDGET = 512
EXPECTED_SEQUENCE_TOKENS = 192
EXPECTED_VALIDATION_SEQUENCE_TOKENS = 128
EXPECTED_HIDDEN_SIZE = 2048
EXPECTED_VOCABULARY_SIZE = 128256
DEFAULT_SEED = 4005
DEFAULT_VALIDATION_EVERY = 1000
GIB = 2**30


class FixedControlCLIError(RuntimeError):
    """Raised when the fixed-control launch contract is incomplete."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedControlCLIError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FixedControlCLIError(f"{label} is not a JSON object: {path}")
    return value


def _require_file(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FixedControlCLIError(f"{label} must be a regular file: {path}")
    return path


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise FixedControlCLIError(f"{label} must be a lowercase SHA-256")
    return value


def _record(path: Path, *, label: str) -> dict[str, Any]:
    path = _require_file(path, label=label)
    return {**file_record(path, label=label), "label": label}


def _git_commit(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FixedControlCLIError("cannot determine source commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise FixedControlCLIError("source commit is malformed")
    return value


def _mem_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                return int(fields[1]) * 1024
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_bytes = rss_raw * 1024 if sys.platform != "darwin" else rss_raw
    snapshot: dict[str, Any] = {
        "host_rss_bytes": rss_bytes,
        "host_available_bytes": _mem_available_bytes(),
        "device": str(device),
        "gpu_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot.update(
            {
                "gpu_free_bytes": int(free_bytes),
                "gpu_total_bytes": int(total_bytes),
                "gpu_max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "gpu_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
    return snapshot


@dataclass
class ResourceTracker:
    """Small in-process guard; an external watchdog remains mandatory."""

    device: torch.device
    max_rss_bytes: int
    min_host_available_bytes: int
    min_gpu_free_bytes: int
    max_gpu_reserved_bytes: int

    def __post_init__(self) -> None:
        self.peak: dict[str, Any] = {}
        self.low_water: dict[str, Any] = {}
        self.checks: list[dict[str, Any]] = []

    def check(self, stage: str) -> None:
        current = _resource_snapshot(self.device)
        self.checks.append({"stage": stage, **current})
        for key, value in current.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if key in {"host_available_bytes", "gpu_free_bytes"}:
                prior = self.low_water.get(key)
                if prior is None or value < prior:
                    self.low_water[key] = value
            else:
                prior = self.peak.get(key)
                if prior is None or value > prior:
                    self.peak[key] = value
        rss = int(current["host_rss_bytes"])
        if rss > self.max_rss_bytes:
            raise FixedControlCLIError(f"resource guard exceeded host RSS at {stage}: {rss}")
        available = current.get("host_available_bytes")
        if available is None:
            raise FixedControlCLIError(f"resource guard cannot measure host availability at {stage}")
        if int(available) < self.min_host_available_bytes:
            raise FixedControlCLIError(f"resource guard exceeded host availability at {stage}: {available}")
        if self.device.type == "cuda" and torch.cuda.is_available():
            free = int(current.get("gpu_free_bytes", -1))
            reserved = int(current.get("gpu_max_reserved_bytes", -1))
            if free < self.min_gpu_free_bytes:
                raise FixedControlCLIError(f"resource guard exceeded GPU free floor at {stage}: {free}")
            if reserved > self.max_gpu_reserved_bytes:
                raise FixedControlCLIError(f"resource guard exceeded GPU reserved cap at {stage}: {reserved}")


def _verify_gate(path: Path, *, label: str, kind: str) -> dict[str, Any]:
    value = _load_json(_require_file(path, label=label), label=label)
    status = str(value.get("status", "")).upper()
    if kind == "agreement":
        accepted = ("STAGE3" in status and "AGREE" in status) or value.get("fit_authorized") is True
    else:
        accepted = status in {"QUALIFICATION_PASS", "STAGE3_QUALIFICATION_PASS", "PASS"} or value.get("qualification_pass") is True
    if not accepted:
        raise FixedControlCLIError(f"{label} is not an accepted {kind} receipt: {status!r}")
    if value.get("truth_opened") is True or value.get("final_evaluation_truth_opened") is True:
        raise FixedControlCLIError(f"{label} claims final truth access")
    return {"file": _record(path, label=label), "status": status}


def _verify_watchdog(path: Path) -> dict[str, Any]:
    value = _load_json(_require_file(path, label="external watchdog receipt"), label="external watchdog receipt")
    status = str(value.get("status", "")).upper()
    if status not in {"ARMED", "PASS", "WATCHDOG_PASS", "QUALIFICATION_PASS"}:
        raise FixedControlCLIError(f"external watchdog is not armed/pass: {status!r}")
    command = value.get("command", value.get("watchdog_command"))
    if not isinstance(command, (list, str)) or not command:
        raise FixedControlCLIError("external watchdog receipt lacks its executable command")
    enforcement = value.get("enforcement", value.get("wrapper_enforcement", {}))
    if not isinstance(enforcement, Mapping) or enforcement.get("process_group_fail_closed") is not True:
        raise FixedControlCLIError("external watchdog is not marked process-group fail-closed")
    return {"file": _record(path, label="external watchdog receipt"), "status": status, "command": command}


def _verify_supplement(path: Path) -> dict[str, Any]:
    path = _require_file(path, label="selection supplement")
    actual = sha256_file(path)
    if actual != EXPECTED_SELECTION_SUPPLEMENT_SHA256:
        raise FixedControlCLIError("selection supplement hash differs from the signed P09 supplement")
    value = _load_json(path, label="selection supplement")
    if value.get("task_id") != TASK_ID:
        raise FixedControlCLIError("selection supplement task identity changed")
    return {"file": _record(path, label="selection supplement"), "schema": value.get("schema"), "status": value.get("status")}


def _read_bank_rows(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read metadata for either an immutable B0 binding or a streamed B1 bank."""

    manifest_path = _require_file(manifest_path, label="bank manifest/binding")
    raw = _load_json(manifest_path, label="bank manifest/binding")
    if raw.get("schema") == "token-reconstruction.trr-p09-b0-immutable-loader-binding.v1":
        try:
            loader = B0ImmutableLoader(manifest_path, device="cpu")
        except Exception as exc:
            raise FixedControlCLIError("cannot validate immutable B0 binding") from exc
        rows = []
        for index, value in enumerate(loader.records):
            row = dict(value)
            # The immutable B0 records are slot-indexed; the combined loader
            # exposes that slot as the expanded global row.
            row.setdefault("global_row", index)
            rows.append(row)
        manifest = {
            "schema": raw.get("schema"),
            "geometry": {
                "sequence_tokens": loader.geometry.sequence_tokens,
                "hidden_size": loader.geometry.hidden_size,
                "loader_batch_records": loader.geometry.batch_records,
                "shard_records": loader.geometry.shard_records,
                "hidden_dtype": loader.geometry.hidden_dtype,
            },
            "bank": {
                "record_count": loader.record_count,
                "expanded_row_origin": loader.expanded_row_origin,
            },
            "_bytes": manifest_path.stat().st_size,
            "_sha256": sha256_file(manifest_path),
        }
    else:
        try:
            manifest = validate_bank_manifest(manifest_path)
        except Exception as exc:
            raise FixedControlCLIError("cannot validate streamed bank manifest") from exc
        root = manifest_path.parent
        rows = []
        for shard in manifest["sharding"]["shards"]:
            sidecar_path = (root / str(shard["sidecar"]["path"])).resolve()
            sidecar = _load_json(sidecar_path, label="bank sidecar")
            values = sidecar.get("records")
            if not isinstance(values, list):
                raise FixedControlCLIError(f"bank sidecar records are missing: {sidecar_path}")
            rows.extend(dict(value) for value in values if isinstance(value, Mapping))
        manifest["_bytes"] = manifest_path.stat().st_size
        manifest["_sha256"] = sha256_file(manifest_path)
    expected_start = int(manifest["bank"].get("expanded_row_origin", 0))
    expected_rows = list(range(expected_start, expected_start + len(rows)))
    actual_rows = [int(row.get("global_row", -1)) for row in rows]
    if actual_rows != expected_rows:
        raise FixedControlCLIError("bank metadata global-row order is not contiguous")
    for row in rows:
        active = row.get("active_token_count")
        if isinstance(active, bool) or not isinstance(active, int) or active < 2 or active > EXPECTED_SEQUENCE_TOKENS:
            raise FixedControlCLIError("bank active_token_count is invalid")
    return manifest, rows


def _valid_mask(rows: Sequence[Mapping[str, Any]], *, sequence_tokens: int) -> torch.Tensor:
    if sequence_tokens <= 1:
        raise FixedControlCLIError("bank sequence width is too short")
    mask = torch.zeros((len(rows), sequence_tokens), dtype=torch.bool)
    for index, row in enumerate(rows):
        active = int(row["active_token_count"])
        if active > sequence_tokens:
            raise FixedControlCLIError("bank active-token count exceeds manifest width")
        mask[index, :active] = True
    return mask


def _combined_file_binding(prefix: Path, addition: Path) -> tuple[dict[str, Any], str]:
    payload = {"prefix": _record(prefix, label="B0 fit bank manifest"), "addition": _record(addition, label="B1 fit bank manifest")}
    return payload, _canonical_digest(payload)


def _schedule_pass(
    valid_mask: torch.Tensor,
    global_rows: Sequence[int],
    *,
    steps: int,
    seed: int,
    record_batch_size: int,
    position_budget: int,
) -> tuple[str, dict[str, Any]]:
    """Digest/exposure pass without retaining the 6.6M-draw schedule."""

    hasher = hashlib.sha256()
    hasher.update(("{\"seed\":" + str(int(seed)) + ",\"steps\":[").encode("ascii"))
    pair_counts: Counter[tuple[int, int]] = Counter()
    used_replacement = 0
    total = 0
    first = True
    for step in inherited_schedule_steps(
        valid_mask,
        global_rows,
        steps=steps,
        seed=seed,
        record_batch_size=record_batch_size,
        position_budget=position_budget,
    ):
        encoded = _canonical_bytes(step.as_dict())
        if not first:
            hasher.update(b",")
        first = False
        hasher.update(encoded)
        used_replacement += int(step.used_replacement)
        total += len(step.draw_position_slots)
        for record_slot, position in zip(step.draw_record_slots, step.draw_position_slots, strict=True):
            pair_counts[(int(step.batch_global_rows[record_slot]), int(position))] += 1
    hasher.update(b"]}")
    if first or total != steps * position_budget:
        raise FixedControlCLIError("derived schedule has the wrong number of steps or draws")
    semantic = hasher.hexdigest()
    exposure = {
        "seed": int(seed),
        "steps": int(steps),
        "draws_per_step": int(position_budget),
        "total_draws": int(total),
        "used_replacement_steps": int(used_replacement),
        "unique_record_position_pairs": len(pair_counts),
        "repeated_record_position_pairs": sum(value > 1 for value in pair_counts.values()),
        "max_exposures_per_record_position": max(pair_counts.values(), default=0),
        "schedule_semantic_sha256": semantic,
    }
    return semantic, exposure


def _write_schedule_receipt(output_root: Path, *, seed: int, steps: int, grid: Sequence[int], valid_positions: int, semantic: str, exposure: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema": "token-reconstruction.trr-p09-fixed-control-schedule.v1",
        "task_id": TASK_ID,
        "status": "DERIVED_BEFORE_MODEL_LOAD",
        "seed": int(seed),
        "steps": int(steps),
        "position_budget": EXPECTED_POSITION_BUDGET,
        "record_batch_size": EXPECTED_RECORD_BATCH_SIZE,
        "actual_valid_positions": int(valid_positions),
        "checkpoint_grid": [int(step) for step in grid],
        "semantic_sha256": semantic,
        "exposure": dict(exposure),
        "truth_opened": False,
    }
    path = output_root / "schedule_receipt.json"
    write_create_only_json(path, value)
    return _record(path, label="derived fixed-control schedule receipt")


def _load_embedding(path: Path, *, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    path = _require_file(path, label="public normalized embedding")
    record = _record(path, label="public normalized embedding")
    if record["sha256"] != EXPECTED_EMBEDDING_SHA256:
        raise FixedControlCLIError("public embedding hash differs from the signed P09 E")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"embeddings"}:
                raise FixedControlCLIError("public embedding tensor key changed")
            embedding = handle.get_tensor("embeddings").contiguous()
    except FixedControlCLIError:
        raise
    except Exception as exc:
        raise FixedControlCLIError("cannot load public embedding") from exc
    if tuple(embedding.shape) != (EXPECTED_VOCABULARY_SIZE, EXPECTED_HIDDEN_SIZE) or not embedding.dtype.is_floating_point:
        raise FixedControlCLIError("public embedding geometry or dtype changed")
    if not torch.isfinite(embedding).all().item():
        raise FixedControlCLIError("public embedding contains non-finite values")
    return embedding.to(device=device), {"file": record, "shape": list(embedding.shape), "dtype": str(embedding.dtype), "finite_checked_once": True}


def _load_decoder(path: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    path = _require_file(path, label="P09 fixed starting state")
    record = _record(path, label="P09 fixed starting state")
    if record["sha256"] != EXPECTED_STATE_SHA256:
        raise FixedControlCLIError("starting state hash differs from the sole approved P09 state")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise FixedControlCLIError("cannot inspect fixed starting-state metadata") from exc
    if metadata.get("method_id") != RESIDUAL_MLP_METHOD_ID:
        raise FixedControlCLIError("starting state is not the registered residual positionwise model")
    if str(metadata.get("selected_step")) != "400":
        raise FixedControlCLIError("starting state selected step is not 400")
    try:
        decoder = load_positionwise_model_state(
            path,
            method_id=RESIDUAL_MLP_METHOD_ID,
            hidden_size=EXPECTED_HIDDEN_SIZE,
            vocabulary_size=EXPECTED_VOCABULARY_SIZE,
            bottleneck_size=DEFAULT_BOTTLENECK_SIZE,
        ).to(device=device)
    except Exception as exc:
        raise FixedControlCLIError("cannot load the approved fixed starting state") from exc
    return decoder, {"file": record, "metadata": metadata, "method_id": METHOD_ID, "selected_step": 400}


def _label_rows(path: Path) -> list[dict[str, Any]]:
    value = _load_json(_require_file(path, label="public validation labels"), label="public validation labels")
    rows = value.get("rows", value.get("records"))
    if not isinstance(rows, list):
        raise FixedControlCLIError("public validation labels must contain rows or records")
    return [dict(row) for row in rows if isinstance(row, Mapping)]




def _load_fixed_diagnostic_rows(path: Path, *, bank: str, row_limit: int, expected_sha256: str | None = None) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Load the setup-owned frozen 64-row diagnostic binding."""

    path = _require_file(path, label="fixed diagnostic binding")
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != _require_sha(expected_sha256, label="expected fixed diagnostic SHA"):
        raise FixedControlCLIError("fixed diagnostic binding hash differs")
    value = _load_json(path, label="fixed diagnostic binding")
    if value.get("schema") != "token-reconstruction.trr-p09-fixed-diagnostic-binding.v1" or value.get("task_id") != TASK_ID:
        raise FixedControlCLIError("fixed diagnostic binding schema/task differs")
    if value.get("status") != "PASS_FIXED_DIAGNOSTIC64_BOUND_PUBLIC_INPUTS_ONLY":
        raise FixedControlCLIError("fixed diagnostic binding is not a public-input PASS")
    verification = value.get("verification")
    if not isinstance(verification, Mapping) or verification.get("evaluation_truth_opened") is not False or verification.get("source_plaintext_read") is not False:
        raise FixedControlCLIError("fixed diagnostic binding crosses the truth/source boundary")
    banks = value.get("banks")
    if not isinstance(banks, Mapping) or not isinstance(banks.get(bank), Mapping):
        raise FixedControlCLIError(f"fixed diagnostic binding lacks bank {bank}")
    binding = dict(banks[bank])
    indices = binding.get("global_indices")
    rows = binding.get("rows")
    if binding.get("bank") != bank or int(binding.get("record_count", -1)) != 64 or not isinstance(indices, list) or not isinstance(rows, list) or len(indices) != 64 or len(rows) != 64:
        raise FixedControlCLIError("fixed diagnostic binding row count/schema differs")
    parsed_indices = tuple(int(row) for row in indices)
    if len(set(parsed_indices)) != 64 or any(row < 0 or row >= int(row_limit) for row in parsed_indices):
        raise FixedControlCLIError("fixed diagnostic binding contains an out-of-range/duplicate row")
    if str(binding.get("global_indices_sha256")) != hashlib.sha256(_canonical_bytes(list(parsed_indices))).hexdigest():
        raise FixedControlCLIError("fixed diagnostic global-index digest differs")
    by_row: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise FixedControlCLIError("fixed diagnostic row descriptor is malformed")
        global_row = int(row.get("global_row", -1))
        if global_row in by_row:
            raise FixedControlCLIError("fixed diagnostic row descriptor is duplicated")
        by_row[global_row] = row
    if set(by_row) != set(parsed_indices):
        raise FixedControlCLIError("fixed diagnostic descriptor/index sets differ")
    descriptor_digest = hashlib.sha256(_canonical_bytes([dict(row) for row in rows])).hexdigest()
    if str(binding.get("rows_sha256")) != descriptor_digest:
        raise FixedControlCLIError("fixed diagnostic descriptor digest differs")
    binding["binding_file"] = _record(path, label="fixed diagnostic binding")
    binding["binding_sha256"] = actual_sha
    binding["row_descriptors_by_global_row"] = by_row
    return binding, parsed_indices


def _validate_fixed_diagnostic_rows(
    binding: Mapping[str, Any],
    row_indices: Sequence[int],
    loader: Any,
) -> None:
    """Compare the frozen diagnostic identities/masks to the active bank loader."""

    batch = loader.get_records(tuple(int(row) for row in row_indices))
    descriptors = binding.get("row_descriptors_by_global_row")
    if not isinstance(descriptors, Mapping):
        raise FixedControlCLIError("fixed diagnostic descriptors were not loaded")
    for index, global_row in enumerate(row_indices):
        descriptor = descriptors.get(int(global_row))
        if not isinstance(descriptor, Mapping):
            raise FixedControlCLIError("fixed diagnostic descriptor is missing")
        if batch.record_ids[index] != str(descriptor.get("record_id", "")):
            raise FixedControlCLIError(f"fixed diagnostic record identity differs at row {global_row}")
        mask = batch.attention_mask[index].detach().cpu().contiguous()
        positions = batch.position_ids[index].detach().cpu().contiguous()
        tokens = batch.token_ids[index].detach().cpu().contiguous()
        if tensor_digest(mask.to(dtype=torch.uint8)) != str(descriptor.get("attention_mask_sha256", "")):
            raise FixedControlCLIError(f"fixed diagnostic mask digest differs at row {global_row}")
        if tensor_digest(positions) != str(descriptor.get("position_ids_sha256", "")):
            raise FixedControlCLIError(f"fixed diagnostic position digest differs at row {global_row}")
        active = int(mask.to(dtype=torch.bool).sum().item())
        if active <= 0 or active > EXPECTED_SEQUENCE_TOKENS:
            raise FixedControlCLIError(f"fixed diagnostic active length is invalid at row {global_row}")
        sequence = tokens[:active].to(dtype=torch.int32).contiguous()
        if tensor_digest(sequence) != str(descriptor.get("sequence_sha256", "")):
            raise FixedControlCLIError(f"fixed diagnostic sequence digest differs at row {global_row}")


class _ValidationCropLoader:
    """Expose the signed H128 validation view from a full-width bank."""

    def __init__(self, loader: StreamedBankLoader, *, sequence_tokens: int) -> None:
        if int(sequence_tokens) != EXPECTED_VALIDATION_SEQUENCE_TOKENS:
            raise FixedControlCLIError("P09 validation view must contain exactly 128 tokens including BOS")
        if int(loader.geometry.sequence_tokens) != EXPECTED_SEQUENCE_TOKENS:
            raise FixedControlCLIError("validation bank must preserve full 192-token capture geometry")
        self._loader = loader
        self.sequence_tokens = int(sequence_tokens)

    def get_records(self, global_indices: Sequence[int]) -> StreamBatch:
        batch = self._loader.get_records(global_indices)
        stop = self.sequence_tokens
        return StreamBatch(
            activations=batch.activations[:, :stop].contiguous(),
            token_ids=batch.token_ids[:, :stop].contiguous(),
            attention_mask=batch.attention_mask[:, :stop].contiguous(),
            position_ids=batch.position_ids[:, :stop].contiguous(),
            global_rows=batch.global_rows,
            record_ids=batch.record_ids,
            sequence_ids=batch.sequence_ids,
        )


def _validate_label_identities(
    labels: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    expanded_row_origin: int,
) -> None:
    expected = {
        int(row.get("global_row", -1)): str(row.get("record_id", ""))
        for row in rows
    }
    for label in labels.rows:
        actual = expected.get(int(label.global_row))
        if actual is None or actual != label.record_id:
            raise FixedControlCLIError(
                f"validation label identity differs at global row {label.global_row}"
            )
    if any(
        int(label.global_row) < int(expanded_row_origin)
        or int(label.global_row) >= int(expanded_row_origin) + len(rows)
        for label in labels.rows
    ):
        raise FixedControlCLIError("validation label join points outside the validation bank")


@dataclass
class _SyntheticBatch:
    activations: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    global_rows: tuple[int, ...]


class _SyntheticLoader:
    def __init__(self, *, count: int, sequence_tokens: int = 4, hidden_size: int = 3) -> None:
        self.sequence_tokens = sequence_tokens
        self.hidden_size = hidden_size
        self.rows: dict[int, _SyntheticBatch] = {}
        for row in range(count):
            activation = (torch.arange(sequence_tokens * hidden_size, dtype=torch.float32).reshape(sequence_tokens, hidden_size) + row).to(torch.bfloat16)
            token_ids = (torch.arange(sequence_tokens, dtype=torch.long) + row) % 7
            mask = torch.ones(sequence_tokens, dtype=torch.bool)
            positions = torch.arange(sequence_tokens, dtype=torch.long)
            self.rows[row] = _SyntheticBatch(activation.unsqueeze(0), token_ids.unsqueeze(0), mask.unsqueeze(0), positions.unsqueeze(0), (row,))

    def get_records(self, global_indices: Sequence[int]) -> _SyntheticBatch:
        requested = tuple(int(row) for row in global_indices)
        batches = [self.rows[row] for row in requested]
        return _SyntheticBatch(
            activations=torch.cat([batch.activations for batch in batches]),
            token_ids=torch.cat([batch.token_ids for batch in batches]),
            attention_mask=torch.cat([batch.attention_mask for batch in batches]),
            position_ids=torch.cat([batch.position_ids for batch in batches]),
            global_rows=requested,
        )


class _SyntheticDecoder(nn.Module):
    def __init__(self, hidden_size: int = 3) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def projected_hidden(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        projected = F.normalize(self.projection(activation), dim=-1)
        return torch.where(valid_mask.unsqueeze(-1), projected, torch.zeros_like(projected))

    def logits_from_rows(self, projected_hidden: torch.Tensor, record_slots: torch.Tensor, position_slots: torch.Tensor, embedding_table: torch.Tensor) -> torch.Tensor:
        rows = projected_hidden[record_slots.to(projected_hidden.device), position_slots.to(projected_hidden.device)]
        return rows @ embedding_table.transpose(0, 1) * self.logit_scale


def _run_synthetic(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FixedControlCLIError(f"synthetic output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    steps = int(args.steps or 2)
    seed = int(args.seed if args.seed is not None else 13)
    torch.manual_seed(seed)
    fit_loader = _SyntheticLoader(count=8)
    val_loader = _SyntheticLoader(count=4)
    fit_mask = torch.ones(8, 4, dtype=torch.bool)
    global_rows = tuple(range(8))
    semantic, exposure = _schedule_pass(fit_mask, global_rows, steps=steps, seed=seed, record_batch_size=2, position_budget=3)
    grid = tuple(range(0, steps + 1))
    schedule_record = _write_schedule_receipt(output_root, seed=seed, steps=steps, grid=grid, valid_positions=int(fit_mask[:, 1:].sum()), semantic=semantic, exposure=exposure)
    contract = BankContract(
        fit_manifest=AssetBinding("synthetic-fit", "synthetic://fit", 1, "a" * 64),
        validation_manifest=AssetBinding("synthetic-validation", "synthetic://validation", 1, "b" * 64),
        embedding=AssetBinding("synthetic-E", "synthetic://E", 1, "c" * 64),
        schedule=AssetBinding("synthetic-schedule", schedule_record["path"], schedule_record["bytes"], schedule_record["sha256"]),
        fit_shape=(8, 4, 3), validation_shape=(4, 4, 3), fit_mask_shape=(8, 4), validation_mask_shape=(4, 4),
        fit_post_bos_rows=24, fit_supported_token_count=7, schedule_semantic_sha256=semantic,
    )
    decoder = _SyntheticDecoder()
    hook = FixedPublicReadoutHook(method_id=METHOD_ID, embedding_sha256="c" * 64)
    embedding = torch.randn(7, 3)
    source = RandomAccessLoaderSource(fit_loader)
    labels = join_public_validation_labels([
        {"global_row": 0, "record_id": "f0", "domain": "Finance"},
        {"global_row": 1, "record_id": "f1", "domain": "Finance"},
        {"global_row": 2, "record_id": "p0", "domain": "Pile"},
        {"global_row": 3, "record_id": "p1", "domain": "Pile"},
    ])
    val_source = RandomAccessLoaderSource(val_loader)
    def factory(_step: int, _domain: str, rows: tuple[int, ...]) -> Iterable[_SyntheticBatch]:
        return [val_source.batch_for_global_rows(rows[index:index + 2]) for index in range(0, len(rows), 2)]
    callback = make_domain_validation_callback(labels, factory)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01, weight_decay=0.0, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    config = RunnerConfig(steps=steps, record_batch_size=2, position_budget=3, validation_every=1, selection_metric=DOMAIN_BALANCED_SELECTION_METRIC, seed=seed, learning_rate=0.01, weight_decay=0.0, gradient_clip_norm=1.0, train_sequence_tokens=4, hidden_size=3, expected_activation_dtype=str(torch.bfloat16))
    checkpoint = make_fixed_checkpoint_callback(output_root=output_root / "states", bank_contract=contract, bank_manifest_sha256="d" * 64, base_state_sha256="e" * 64, fit_manifest_sha256="f" * 64)
    result = run_training(decoder, hook, source, inherited_schedule_steps(fit_mask, global_rows, steps=steps, seed=seed, record_batch_size=2, position_budget=3), schedule_steps_count=steps, schedule_seed=seed, schedule_semantic_sha256=semantic, schedule_exposure=exposure, optimizer=optimizer, embedding=embedding, config=config, validation_callback=callback, validation_sequence_tokens=4, validation_batch_records=2, validation_activation_dtype=torch.bfloat16, training_activation_dtype=torch.bfloat16, checkpoint_steps=grid, scheduler=scheduler, checkpoint_callback=checkpoint)
    receipt = build_fixed_control_receipt(training_result=result, source_commit="synthetic", command=["fixed_control_cli.py", "--synthetic"], started_utc="2026-09-07T00:00:00Z", finished_utc="2026-09-07T00:00:01Z", environment={"device": "cpu", "synthetic": True}, assets={"embedding": {"sha256": "c" * 64}, "stage3": "bypassed_for_synthetic"}, training_contract={"steps": steps, "record_batch_size": 2, "position_budget": 3, "seed": seed}, resource_peak={"synthetic": True}, bank_manifest_sha256="d" * 64, state={"base_state_sha256": "e" * 64, "method_id": METHOD_ID})
    write_fixed_control_receipt(output_root / "receipt.json", receipt)
    print(json.dumps({"status": "SYNTHETIC_PASS", "selected_step": result["selected_step"], "schedule_semantic_sha256": semantic}, sort_keys=True))
    return 0



def _run_configuration_dry_run(args: argparse.Namespace) -> int:
    """Validate actual bank/schedule wiring without loading model or E."""

    required = ("fit_prefix_manifest", "fit_addition_manifest", "common_schedule")
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        raise FixedControlCLIError("configuration dry-run lacks required arguments: " + ", ".join(missing))
    if args.device != "cpu":
        raise FixedControlCLIError("configuration dry-run is CPU-only; pass --device cpu")
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FixedControlCLIError(f"configuration dry-run output is create-only: {output_root}")
    prefix_path = Path(args.fit_prefix_manifest).expanduser().resolve()
    addition_path = Path(args.fit_addition_manifest).expanduser().resolve()
    common_schedule_path = Path(args.common_schedule).expanduser().resolve()
    fit_asset, computed_fit_sha = _combined_file_binding(prefix_path, addition_path)
    if args.expected_fit_binding_sha256 is not None and _require_sha(args.expected_fit_binding_sha256, label="expected fit binding SHA") != computed_fit_sha:
        raise FixedControlCLIError("fit manifest binding differs from the supplied expected SHA")
    fit_manifest, prefix_rows = _read_bank_rows(prefix_path)
    addition_manifest, addition_rows = _read_bank_rows(addition_path)
    if int(fit_manifest["bank"].get("expanded_row_origin", 0)) != 0:
        raise FixedControlCLIError("B0 binding does not begin at global row zero")
    if int(addition_manifest["bank"].get("expanded_row_origin", -1)) != len(prefix_rows):
        raise FixedControlCLIError("B1 binding does not begin after the B0 prefix")
    fit_rows = [*prefix_rows, *addition_rows]
    if len(fit_rows) != 12000:
        raise FixedControlCLIError(f"configuration dry-run requires 12000 composed rows, got {len(fit_rows)}")
    valid_mask = _valid_mask(fit_rows, sequence_tokens=EXPECTED_SEQUENCE_TOKENS)
    seed = int(args.seed) if args.seed is not None else 4010
    expected_steps = int(args.steps) if args.steps is not None else 13000
    schedule = load_serialized_schedule(
        common_schedule_path,
        expected_seed=seed,
        expected_steps=expected_steps,
        expected_record_batch_size=EXPECTED_RECORD_BATCH_SIZE,
        expected_position_budget=EXPECTED_POSITION_BUDGET,
        expected_sequence_tokens=EXPECTED_SEQUENCE_TOKENS,
        expected_global_row_exclusive=len(fit_rows),
        expected_bank=args.schedule_bank,
        expected_valid_mask_semantic_sha256=None,
        expected_file_sha256=(
            _require_sha(args.expected_schedule_sha256, label="expected schedule SHA")
            if args.expected_schedule_sha256 is not None
            else None
        ),
    )
    schedule_mask = valid_mask if schedule.bank == "B1" else valid_mask[: len(prefix_rows)]
    if schedule.valid_mask_semantic_sha256 != tensor_digest(schedule_mask):
        raise FixedControlCLIError("configuration schedule mask digest does not match composed bank metadata")
    schedule_row_limit = len(fit_rows) if schedule.bank == "B1" else len(prefix_rows)
    if int(schedule.batch_record_indices.max().item()) >= schedule_row_limit:
        raise FixedControlCLIError("configuration schedule selects rows outside its bank namespace")
    loader = CombinedB0StreamedBankLoader(prefix_path, addition_path, device="cpu")
    requested_rows = (0, 7, 8, 1199, 1200, 1207, 1264, 1265, 11999)
    batch = loader.get_records(requested_rows)
    if tuple(batch.global_rows) != requested_rows:
        raise FixedControlCLIError("combined loader changed dry-run row order")
    if tuple(batch.activations.shape) != (len(requested_rows), EXPECTED_SEQUENCE_TOKENS, EXPECTED_HIDDEN_SIZE):
        raise FixedControlCLIError("combined loader activation geometry changed")
    if batch.activations.dtype != torch.bfloat16:
        raise FixedControlCLIError("combined loader activation dtype changed")
    mask = batch.attention_mask.to(dtype=torch.bool)
    expected_positions = torch.where(
        mask,
        torch.arange(EXPECTED_SEQUENCE_TOKENS, dtype=torch.long).expand_as(mask),
        torch.zeros_like(batch.position_ids, dtype=torch.long),
    )
    if not torch.equal(batch.position_ids.to(dtype=torch.long), expected_positions):
        raise FixedControlCLIError("combined loader position/mask convention changed")
    diagnostic_meta = None
    diagnostic_rows = None
    if args.fixed_diagnostic_binding is not None:
        diagnostic_meta, diagnostic_rows = _load_fixed_diagnostic_rows(
            Path(args.fixed_diagnostic_binding),
            bank=str(args.schedule_bank or schedule.bank),
            row_limit=len(fit_rows) if schedule.bank == "B1" else len(prefix_rows),
            expected_sha256=args.expected_fixed_diagnostic_sha256,
        )
        _validate_fixed_diagnostic_rows(diagnostic_meta, diagnostic_rows, loader)
    validation_meta = None
    validation_batch = None
    validation_rows_path = getattr(args, "validation_rows", None) or getattr(args, "validation_labels", None)
    if args.validation_manifest is not None or validation_rows_path is not None:
        if args.validation_manifest is None or validation_rows_path is None:
            raise FixedControlCLIError("configuration dry-run validation binding needs both manifest and rows")
        try:
            validation_loader = PublicValidationLoader(Path(args.validation_manifest), Path(validation_rows_path))
            validation_rows = (0, 7, 255, 256, 263, 383)
            validation_batch = validation_loader.get_records(validation_rows)
        except PublicValidationLoaderError as exc:
            raise FixedControlCLIError(str(exc)) from exc
        if tuple(validation_batch.activations.shape) != (len(validation_rows), EXPECTED_VALIDATION_SEQUENCE_TOKENS, EXPECTED_HIDDEN_SIZE):
            raise FixedControlCLIError("public validation dry-run geometry changed")
        validation_meta = validation_loader.metadata()
    output_root.mkdir(parents=True)
    receipt = {
        "schema": "token-reconstruction.trr-p09-fixed-control-configuration-dry-run.v1",
        "task_id": TASK_ID,
        "status": "PASS_FIXED_CONTROL_CPU_CONFIGURATION",
        "command": list(sys.argv),
        "source_commit": _git_commit(_REPOSITORY_ROOT),
        "scope": "CPU-only configuration validation; no model, embedding, validation labels, optimizer, fit, or evaluation truth",
        "fit_binding": fit_asset,
        "fixed_diagnostic": diagnostic_meta,
        "schedule": {
            "file": _record(common_schedule_path, label="serialized common schedule"),
            "bank": schedule.bank,
            "seed": schedule.seed,
            "steps": schedule.steps,
            "record_batch_size": schedule.record_batch_size,
            "position_budget": schedule.position_budget,
            "sequence_tokens": schedule.sequence_tokens,
            "semantic_sha256": schedule.semantic_sha256,
            "valid_mask_semantic_sha256": schedule.valid_mask_semantic_sha256,
            "exposure": schedule.exposure_summary(),
        },
        "loader": {
            "interface": loader.interface,
            "b0_binding": _record(prefix_path, label="B0 immutable binding"),
            "b1_manifest": _record(addition_path, label="B1 bank manifest"),
            "global_row_range": [loader.expanded_row_origin, loader.global_row_stop],
            "requested_rows": list(requested_rows),
            "returned_shape": list(batch.activations.shape),
            "activation_dtype": str(batch.activations.dtype),
            "record_ids_sha256": hashlib.sha256(("\\n".join(batch.record_ids) + "\\n").encode()).hexdigest(),
            "mask_position_contract": "arange(T) on active positions, zero on inactive padding",
        },
        "validation": validation_meta,
        "truth_boundary": {
            "model_loaded": False,
            "embedding_loaded": False,
            "validation_labels_loaded": bool(validation_meta is not None),
            "optimizer_created": False,
            "fit_started": False,
            "evaluation_truth_opened": False,
        },
    }
    write_create_only_json(output_root / "configuration_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "output_root": str(output_root), "schedule_sha256": schedule.semantic_sha256, "schedule_bank": schedule.bank}, sort_keys=True))
    return 0

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--configuration-dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--fit-prefix-manifest", "--fit-prefix-binding", dest="fit_prefix_manifest", type=Path)
    parser.add_argument("--fit-addition-manifest", type=Path)
    parser.add_argument("--common-schedule", type=Path)
    parser.add_argument("--expected-schedule-sha256")
    parser.add_argument("--schedule-bank", choices=("B0", "B1"))
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--validation-rows", type=Path)
    parser.add_argument("--validation-labels", type=Path, help="legacy alias for --validation-rows")
    parser.add_argument("--fixed-diagnostic-binding", type=Path)
    parser.add_argument("--expected-fixed-diagnostic-sha256")
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--selection-supplement", type=Path)
    parser.add_argument("--stage3-agreement", type=Path)
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--watchdog-receipt", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--validation-every", type=int, default=DEFAULT_VALIDATION_EVERY)
    parser.add_argument("--validation-sequence-tokens", type=int, default=EXPECTED_VALIDATION_SEQUENCE_TOKENS)
    parser.add_argument("--deadline-seconds", type=float, default=7200.0)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=16.0)
    parser.add_argument("--minimum-host-available-gib", type=float, default=10.0)
    parser.add_argument("--minimum-gpu-free-gib", type=float, default=2.0)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=8.0)
    parser.add_argument("--expected-fit-binding-sha256")
    return parser


def _require_production_args(args: argparse.Namespace) -> None:
    names = ("fit_prefix_manifest", "fit_addition_manifest", "common_schedule", "schedule_bank", "expected_schedule_sha256", "validation_manifest", "fixed_diagnostic_binding", "embedding", "state", "selection_supplement", "stage3_agreement", "qualification_receipt", "watchdog_receipt", "seed", "learning_rate", "weight_decay", "gradient_clip_norm")
    missing = [name for name in names if getattr(args, name) is None]
    if args.validation_rows is None and args.validation_labels is None:
        missing.append("validation_rows")
    if missing:
        raise FixedControlCLIError("production mode lacks required arguments: " + ", ".join(missing))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise FixedControlCLIError("production CUDA device is unavailable")
    if args.validation_every <= 0 or args.deadline_seconds <= 0:
        raise FixedControlCLIError("validation interval/deadline must be positive")
    if int(args.validation_sequence_tokens) != EXPECTED_VALIDATION_SEQUENCE_TOKENS:
        raise FixedControlCLIError("validation sequence width is signed to 128 tokens")
    if args.weight_decay < 0 or args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        raise FixedControlCLIError("optimizer settings are invalid")
    if args.steps is not None and args.steps <= 0:
        raise FixedControlCLIError("--steps must be positive")


def _run_production(args: argparse.Namespace) -> int:
    _require_production_args(args)
    root = Path(args.repository_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FixedControlCLIError(f"production output root is create-only: {output_root}")
    source_commit = _git_commit(root)
    stage3 = _verify_gate(Path(args.stage3_agreement), label="stage3 agreement", kind="agreement")
    qualification = _verify_gate(Path(args.qualification_receipt), label="largest-cell qualification", kind="qualification")
    watchdog = _verify_watchdog(Path(args.watchdog_receipt))
    supplement = _verify_supplement(Path(args.selection_supplement))
    device = torch.device(args.device)
    tracker = ResourceTracker(device=device, max_rss_bytes=int(args.maximum_host_rss_gib * GIB), min_host_available_bytes=int(args.minimum_host_available_gib * GIB), min_gpu_free_bytes=int(args.minimum_gpu_free_gib * GIB), max_gpu_reserved_bytes=int(args.maximum_gpu_reserved_gib * GIB))
    tracker.check("pre_model_load")

    prefix_path = Path(args.fit_prefix_manifest).expanduser().resolve()
    addition_path = Path(args.fit_addition_manifest).expanduser().resolve()
    validation_path = Path(args.validation_manifest).expanduser().resolve()
    fit_asset, computed_fit_sha = _combined_file_binding(prefix_path, addition_path)
    if args.expected_fit_binding_sha256 is not None and _require_sha(args.expected_fit_binding_sha256, label="expected fit binding SHA") != computed_fit_sha:
        raise FixedControlCLIError("fit manifest binding differs from the supplied expected SHA")
    fit_manifest, prefix_rows = _read_bank_rows(prefix_path)
    addition_manifest, addition_rows = _read_bank_rows(addition_path)
    if int(fit_manifest["bank"].get("expanded_row_origin", 0)) != 0 or int(addition_manifest["bank"].get("expanded_row_origin", 0)) != len(prefix_rows):
        raise FixedControlCLIError("B0/B1 expanded-row origins are not adjacent")
    fit_rows = [*prefix_rows, *addition_rows]
    if len(fit_rows) != int(fit_manifest["bank"]["record_count"]) + int(addition_manifest["bank"]["record_count"]):
        raise FixedControlCLIError("B0+B1 fit row count changed")
    fit_geometry = fit_manifest["geometry"]
    if int(fit_geometry["sequence_tokens"]) != EXPECTED_SEQUENCE_TOKENS or int(fit_geometry["hidden_size"]) != EXPECTED_HIDDEN_SIZE:
        raise FixedControlCLIError("fit bank geometry differs from P09 full-width contract")
    valid_mask = _valid_mask(fit_rows, sequence_tokens=EXPECTED_SEQUENCE_TOKENS)
    seed = int(args.seed)
    expected_steps = int(args.steps) if args.steps is not None else 13000
    common_schedule_path = Path(args.common_schedule).expanduser().resolve()
    schedule = load_serialized_schedule(
        common_schedule_path,
        expected_seed=seed,
        expected_steps=expected_steps,
        expected_record_batch_size=EXPECTED_RECORD_BATCH_SIZE,
        expected_position_budget=EXPECTED_POSITION_BUDGET,
        expected_sequence_tokens=EXPECTED_SEQUENCE_TOKENS,
        expected_global_row_exclusive=len(fit_rows),
        expected_bank=args.schedule_bank,
        expected_file_sha256=(
            _require_sha(args.expected_schedule_sha256, label="expected schedule SHA")
            if args.expected_schedule_sha256 is not None
            else None
        ),
    )
    schedule_mask = valid_mask if schedule.bank == "B1" else valid_mask[: len(prefix_rows)]
    expected_schedule_mask_sha256 = tensor_digest(schedule_mask)
    if schedule.valid_mask_semantic_sha256 != expected_schedule_mask_sha256:
        raise FixedControlCLIError("serialized schedule valid-mask digest does not match the selected bank rows")
    schedule_row_limit = len(fit_rows) if schedule.bank == "B1" else len(prefix_rows)
    if int(schedule.batch_record_indices.max().item()) >= schedule_row_limit:
        raise FixedControlCLIError("serialized schedule selects rows outside its bank namespace")
    valid_positions = int(schedule_mask[:, 1:].sum().item())
    steps = schedule.steps
    checkpoint_grid = tuple(sorted(set(signed_p09_checkpoint_grid(valid_positions) + (steps,))))
    if args.steps is not None and int(args.steps) != steps:
        raise FixedControlCLIError(f"--steps {args.steps} differs from serialized schedule steps {steps}")
    exposure = schedule.exposure_summary()
    output_root.mkdir(parents=True)
    common_schedule_record = _record(common_schedule_path, label="serialized common schedule")
    schedule_receipt = _write_schedule_receipt(
        output_root,
        seed=seed,
        steps=steps,
        grid=checkpoint_grid,
        valid_positions=valid_positions,
        semantic=schedule.semantic_sha256,
        exposure=exposure,
    )

    validation_rows_path = Path(args.validation_rows or args.validation_labels).expanduser().resolve()
    try:
        val_loader = PublicValidationLoader(validation_path, validation_rows_path)
    except PublicValidationLoaderError as exc:
        raise FixedControlCLIError(str(exc)) from exc
    labels = val_loader.label_join
    val_rows = list(val_loader.rows)
    fit_loader = CombinedB0StreamedBankLoader(prefix_path, addition_path, device="cpu")
    diagnostic_binding, diagnostic_rows = _load_fixed_diagnostic_rows(
        Path(args.fixed_diagnostic_binding),
        bank=str(schedule.bank),
        row_limit=schedule_row_limit,
        expected_sha256=args.expected_fixed_diagnostic_sha256,
    )
    _validate_fixed_diagnostic_rows(diagnostic_binding, diagnostic_rows, fit_loader)
    fit_source = RandomAccessLoaderSource(fit_loader)
    val_source = RandomAccessLoaderSource(val_loader)
    def validation_factory(_step: int, _domain: str, rows: tuple[int, ...]) -> Iterable[Any]:
        if len(rows) % EXPECTED_RECORD_BATCH_SIZE:
            raise FixedControlCLIError("validation domain row count is not batch aligned")
        return [val_source.batch_for_global_rows(rows[index:index + EXPECTED_RECORD_BATCH_SIZE]) for index in range(0, len(rows), EXPECTED_RECORD_BATCH_SIZE)]
    validation_callback = make_domain_validation_callback(labels, validation_factory)
    embedding, embedding_meta = _load_embedding(Path(args.embedding), device=device)
    decoder, state_meta = _load_decoder(Path(args.state), device=device)
    hook = FixedPublicReadoutHook(method_id=METHOD_ID, embedding_sha256=EXPECTED_EMBEDDING_SHA256)
    tracker.check("after_model_load")
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay), foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    config = RunnerConfig(steps=steps, record_batch_size=EXPECTED_RECORD_BATCH_SIZE, position_budget=EXPECTED_POSITION_BUDGET, validation_every=int(args.validation_every), selection_metric=DOMAIN_BALANCED_SELECTION_METRIC, seed=seed, learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay), gradient_clip_norm=float(args.gradient_clip_norm), train_sequence_tokens=EXPECTED_SEQUENCE_TOKENS, hidden_size=EXPECTED_HIDDEN_SIZE, expected_activation_dtype=str(torch.bfloat16))
    bank_contract = BankContract(
        fit_manifest=AssetBinding("combined-fit-bank-manifests", "combined://b0+b1", 1, computed_fit_sha),
        validation_manifest=AssetBinding("public-validation-preparation-manifest", str(validation_path), int(validation_path.stat().st_size), sha256_file(validation_path)),
        embedding=AssetBinding("public-normalized-E", str(args.embedding), int(embedding_meta["file"]["bytes"]), EXPECTED_EMBEDDING_SHA256),
        schedule=AssetBinding("serialized-common-schedule", str(common_schedule_record["path"]), int(common_schedule_record["bytes"]), str(common_schedule_record["sha256"])),
        fit_shape=(schedule_row_limit, EXPECTED_SEQUENCE_TOKENS, EXPECTED_HIDDEN_SIZE),
        validation_shape=(len(val_rows), EXPECTED_VALIDATION_SEQUENCE_TOKENS, EXPECTED_HIDDEN_SIZE),
        fit_mask_shape=(schedule_row_limit, EXPECTED_SEQUENCE_TOKENS),
        validation_mask_shape=(len(val_rows), EXPECTED_VALIDATION_SEQUENCE_TOKENS),
        fit_post_bos_rows=valid_positions,
        fit_supported_token_count=EXPECTED_VOCABULARY_SIZE,
        schedule_semantic_sha256=schedule.semantic_sha256,
    )
    checkpoint = make_fixed_checkpoint_callback(output_root=output_root / "states", bank_contract=bank_contract, bank_manifest_sha256=computed_fit_sha, base_state_sha256=EXPECTED_STATE_SHA256, fit_manifest_sha256=computed_fit_sha)
    diagnostic_callback = make_fitting_metric_callback(
        source=fit_source,
        embedding=embedding,
        fixed_rows=diagnostic_rows,
        full_bank_rows=tuple(range(schedule_row_limit)),
        final_step=steps,
        expected_sequence_tokens=EXPECTED_SEQUENCE_TOKENS,
        expected_hidden_size=EXPECTED_HIDDEN_SIZE,
        record_batch_size=EXPECTED_RECORD_BATCH_SIZE,
        position_budget=EXPECTED_POSITION_BUDGET,
        activation_dtype=torch.bfloat16,
        fixed_row_count=64,
    )
    checkpoint = compose_checkpoint_callbacks(checkpoint, diagnostic_callback)
    started_utc = _utc_now()
    result = run_training(decoder, hook, fit_source, schedule.iter_steps(), schedule_steps_count=steps, schedule_seed=seed, schedule_semantic_sha256=schedule.semantic_sha256, schedule_exposure=exposure, optimizer=optimizer, embedding=embedding, config=config, validation_callback=validation_callback, validation_sequence_tokens=EXPECTED_VALIDATION_SEQUENCE_TOKENS, validation_batch_records=EXPECTED_RECORD_BATCH_SIZE, validation_activation_dtype=torch.bfloat16, training_activation_dtype=torch.bfloat16, checkpoint_steps=checkpoint_grid, scheduler=scheduler, checkpoint_callback=checkpoint, deadline_seconds=float(args.deadline_seconds), resource_guard_callback=tracker.check)
    finished_utc = _utc_now()
    assets = {"fit_manifests": fit_asset, "validation_manifest": val_loader.metadata(), "validation_rows": _record(validation_rows_path, label="public validation rows"), "embedding": embedding_meta, "starting_state": state_meta, "selection_supplement": supplement, "stage3_agreement": stage3, "qualification": qualification, "watchdog": watchdog, "fixed_diagnostic": diagnostic_binding, "common_schedule": common_schedule_record, "schedule_receipt": schedule_receipt, "contract_digest": contract_digest(bank_contract)}
    receipt = build_fixed_control_receipt(training_result=result, source_commit=source_commit, command=sys.argv, started_utc=started_utc, finished_utc=finished_utc, environment={"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "pid": os.getpid()}, assets=assets, training_contract={"steps": steps, "record_batch_size": EXPECTED_RECORD_BATCH_SIZE, "position_budget": EXPECTED_POSITION_BUDGET, "seed": seed, "schedule_bank": schedule.bank, "checkpoint_grid": list(checkpoint_grid), "validation_sequence_tokens": EXPECTED_VALIDATION_SEQUENCE_TOKENS, "optimizer": "AdamW foreach=False", "learning_rate": float(args.learning_rate), "weight_decay": float(args.weight_decay), "gradient_clip_norm": float(args.gradient_clip_norm)}, resource_peak={"in_process_peak": tracker.peak, "in_process_low_water": tracker.low_water, "checks": tracker.checks, "external_watchdog": watchdog}, bank_manifest_sha256=computed_fit_sha, state={"method_id": METHOD_ID, "starting_state": state_meta, "embedding_sha256": EXPECTED_EMBEDDING_SHA256})
    write_fixed_control_receipt(output_root / "receipt.json", receipt)
    print(json.dumps({"status": "PASS", "selected_step": result["selected_step"], "steps": steps, "total_position_draws": exposure["total_draws"], "output_root": str(output_root)}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.synthetic and args.configuration_dry_run:
        raise FixedControlCLIError("--synthetic and --configuration-dry-run are mutually exclusive")
    if args.configuration_dry_run:
        return _run_configuration_dry_run(args)
    if args.synthetic:
        return _run_synthetic(args)
    return _run_production(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FixedControlCLIError, OSError, ValueError) as exc:
        print(f"TRR-P09 fixed-control refused: {exc}", file=sys.stderr)
        raise SystemExit(125) from exc
