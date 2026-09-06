#!/usr/bin/env python3
"""Bounded TRR-0007 crossed positionwise capacity fitting.

The current-family control is the TRR-0005 strict diagonal decoder.  The one
registered extension adds a zero-output 2048 -> 512 -> 2048 GELU residual over
the current activation.  Both arms are trained from the same neutral
initialization on the retained enriched bank and the TRR-0007 improved public
bank.  No target panel is loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from token_reconstruction.trr0005_joint_decoder import (
    ATTENTION_SCORE_MODE_DOT_PRODUCT,
    DEFAULT_CONTEXT_WIDTH,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_POSITION_BUDGET,
    DEFAULT_RECORD_BATCH_SIZE,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SEED as TRR5_DEFAULT_SEED,
    DEFAULT_VALIDATION_EVERY,
    PublicJointData,
    PositionSchedule,
    build_position_schedule,
    checkpoint_steps,
    evaluate_dataset,
    file_sha256,
    load_public_joint_data,
    save_schedule,
    schedule_digest,
    schedule_metadata,
    tensor_sha256,
)
from token_reconstruction.trr0007_positionwise import (
    BASE_METHOD_ID,
    BASE_STATE_SHA256,
    CURRENT_METHOD_ID,
    DEFAULT_BOTTLENECK_SIZE,
    METHODS,
    RESIDUAL_MLP_METHOD_ID,
    build_current_positionwise,
    build_residual_mlp512,
    load_retained_diagonal_state,
    save_positionwise_state,
    step_zero_equivalence,
)


TASK_ID = "TRR-0007"
SCHEMA = "token-reconstruction.trr0007-positionwise-fit.v1"
DEFAULT_STEPS = 3000
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_MAX_SECONDS = 7200.0
DEFAULT_MINIMUM_FREE_GIB = 8.0
DEFAULT_MAXIMUM_GPU_RESERVED_GIB = 6.0
DEFAULT_MAXIMUM_HOST_RSS_GIB = 16.0
DEFAULT_MINIMUM_HOST_AVAILABLE_GIB = 10.0
DEFAULT_CHALLENGE_ROWS = 2048
DEFAULT_QUALIFICATION_STEPS = 2
MAX_QUALIFICATION_STEPS = 8
MEASURED_TRR0005_PEAK_BYTES = 2_942_304_256
MEASURED_TRR0005_FLOOR_BYTES = math.ceil(MEASURED_TRR0005_PEAK_BYTES * 1.5)
RETAINED_REFERENCE_DEFAULT = Path(
    "experiments/TRR-0005/joint_fit_v1/enriched/"
    "affine_trained_diagonal_attention128/selected.safetensors"
)


class TRR0007FitError(RuntimeError):
    """Raised when a guarded TRR-0007 fit cannot proceed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_status(root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in result.stdout.splitlines() if line]


def _file_record(path: Path, *, hash_bytes: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TRR0007FitError(f"resource must be a regular file: {path}")
    result: dict[str, Any] = {"path": str(path), "bytes": int(path.stat().st_size)}
    if hash_bytes:
        result["sha256"] = file_sha256(path)
    return result


def _safe_environment() -> dict[str, str]:
    names = (
        "CUDA_VISIBLE_DEVICES", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "PYTHONPATH",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise TRR0007FitError(f"invalid device: {value}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TRR0007FitError("CUDA requested but unavailable")
    if device.type not in ("cpu", "cuda"):
        raise TRR0007FitError("device must be cpu, cuda, or auto")
    return device


def _host_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                return int(fields[1]) * 1024
    except (OSError, UnicodeError):
        return None
    return None


def _host_memory_snapshot() -> dict[str, Any]:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    available = _host_available_bytes()
    return {
        "process_max_rss_bytes": rss,
        "host_available_bytes": available,
        "host_available_gib": None if available is None else available / (1024**3),
    }


def _resource_policy(args: argparse.Namespace) -> dict[str, float]:
    return {
        "minimum_free_gpu_gib": float(args.minimum_free_gib),
        "maximum_gpu_reserved_gib": float(args.maximum_gpu_reserved_gib),
        "maximum_host_rss_gib": float(args.maximum_host_rss_gib),
        "minimum_host_available_gib": float(args.minimum_host_available_gib),
    }


def _resource_guard(
    args: argparse.Namespace,
    device: torch.device,
    *,
    stage: str,
    deadline: float | None = None,
) -> dict[str, Any]:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TRR0007FitError(f"wall-time guard expired at {stage}")
    memory = _host_memory_snapshot()
    if memory["process_max_rss_bytes"] > int(args.maximum_host_rss_gib * 1024**3):
        raise TRR0007FitError(f"host RSS guard exceeded at {stage}")
    available = memory["host_available_bytes"]
    if available is not None and available < int(args.minimum_host_available_gib * 1024**3):
        raise TRR0007FitError(f"host available-memory guard exceeded at {stage}")
    result: dict[str, Any] = {
        "stage": stage,
        **memory,
        "minimum_host_available_gib": float(args.minimum_host_available_gib),
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
        "cuda_reserved_bytes": None,
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        reserved = int(torch.cuda.memory_reserved(device))
        result.update({
            "cuda_free_bytes": int(free),
            "cuda_total_bytes": int(total),
            "cuda_reserved_bytes": reserved,
        })
        if free < int(args.minimum_free_gib * 1024**3):
            raise TRR0007FitError(f"GPU free-memory guard exceeded at {stage}")
        if reserved > int(args.maximum_gpu_reserved_gib * 1024**3):
            raise TRR0007FitError(f"GPU reserved-memory guard exceeded at {stage}")
    return result


def _manifest_json(path: Path) -> Mapping[str, Any]:
    path = path.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TRR0007FitError(f"cannot read manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise TRR0007FitError(f"manifest is not an object: {path}")
    if payload.get("schema") not in {
        "token-reconstruction.trr0005-public-fit-data.v1",
        "token-reconstruction.trr0005-joint-fit-data.v1",
        "token-reconstruction.trr0004-public-fit-data.v1",
    }:
        raise TRR0007FitError(f"unsupported manifest schema: {payload.get('schema')}")
    return payload


def _shape_from_manifest(
    manifest: Mapping[str, Any], name: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    resources = manifest.get("resources")
    if isinstance(resources, Mapping) and isinstance(resources.get(name), Mapping):
        shape = resources[name].get("shape")
        if isinstance(shape, list) and all(isinstance(v, int) for v in shape):
            return tuple(int(v) for v in shape)
    return default


def _manifest_resource_shapes(manifest: Mapping[str, Any]) -> dict[str, Any]:
    resources = manifest.get("resources")
    if not isinstance(resources, Mapping):
        return {}
    result: dict[str, Any] = {}
    for name, resource_desc in resources.items():
        if not isinstance(resource_desc, Mapping):
            continue
        shape = resource_desc.get("shape")
        result[str(name)] = {
            "path": resource_desc.get("path"),
            "shape": list(shape) if isinstance(shape, list) else None,
            "bytes": resource_desc.get("bytes"),
            "dtype": resource_desc.get("dtype"),
            "tensor_key": resource_desc.get("tensor_key"),
        }
    return result


def _resource_preflight(
    args: argparse.Namespace,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compute a conservative peak estimate from the actual manifest geometry.

    This is metadata-only: no activation, label, embedding, or retained-state
    tensor is opened.  The runtime receipt later records every materialized
    batch shape, so a successful preflight cannot hide a larger gathered batch.
    """
    fit_manifest = Path(args.fit_manifest).expanduser().resolve()
    validation_manifest = Path(args.validation_manifest).expanduser().resolve()
    fit_payload = _manifest_json(fit_manifest)
    validation_payload = _manifest_json(validation_manifest)
    fit_shapes = _manifest_resource_shapes(fit_payload)
    validation_shapes = _manifest_resource_shapes(validation_payload)
    fit_shape = tuple(
        _shape_from_manifest(
            fit_payload, "fit_observations",
            (args.fit_records_hint, DEFAULT_SEQUENCE_LENGTH, DEFAULT_HIDDEN_SIZE),
        )
    )
    validation_shape = tuple(
        _shape_from_manifest(
            validation_payload, "validation_observations",
            (args.validation_records_hint, DEFAULT_SEQUENCE_LENGTH, DEFAULT_HIDDEN_SIZE),
        )
    )
    if len(fit_shape) != 3 or len(validation_shape) != 3:
        raise TRR0007FitError("manifest observations must have rank three")
    if tuple(fit_shape[1:]) != tuple(validation_shape[1:]):
        raise TRR0007FitError(
            f"fit/validation geometry differs: fit={fit_shape}, validation={validation_shape}"
        )
    hidden_size = int(fit_shape[-1])
    sequence_length = int(fit_shape[1])
    embedding_shape = _shape_from_manifest(
        fit_payload, "embedding_table", (128256, hidden_size)
    )
    if len(embedding_shape) != 2 or int(embedding_shape[1]) != hidden_size:
        raise TRR0007FitError(f"embedding geometry differs: {embedding_shape}")
    vocabulary_size = int(embedding_shape[0])
    if hidden_size != int(args.hidden_size) or sequence_length != int(args.sequence_length):
        raise TRR0007FitError(
            "manifest geometry does not match the fixed TRR-0007 recipe: "
            f"observed={(hidden_size, sequence_length)}, "
            f"expected={(args.hidden_size, args.sequence_length)}"
        )
    if int(args.position_budget) != DEFAULT_POSITION_BUDGET:
        raise TRR0007FitError("TRR-0007 uses exactly 512 post-BOS draws per step")
    records_per_batch = int(args.record_batch_size)
    positions_per_batch = int(args.sequence_length)
    bottleneck = int(args.bottleneck_size)
    base_parameters = (
        hidden_size * hidden_size + hidden_size + 1
        + 3 * (hidden_size * int(args.context_width) + int(args.context_width))
        + int(args.context_width) * hidden_size + hidden_size
    )
    mlp_parameters = (
        hidden_size * bottleneck + bottleneck
        + bottleneck * hidden_size + hidden_size
    )
    parameter_bytes = {
        "current_family_fp32": int(base_parameters * 4),
        "residual_extension_fp32": int(mlp_parameters * 4),
        "residual_total_model_fp32": int((base_parameters + mlp_parameters) * 4),
        "residual_adam_states_and_gradients_fp32": int(
            (base_parameters + mlp_parameters) * 4 * 4
        ),
    }
    activation_bytes = records_per_batch * positions_per_batch * hidden_size * 4
    selected_logits_bytes = int(args.position_budget) * vocabulary_size * 4
    attention_scores_bytes = records_per_batch * positions_per_batch * positions_per_batch * 4
    mlp_workspace_bytes = (
        records_per_batch * positions_per_batch * bottleneck * 4 * 2
        + activation_bytes
    )
    # The selected logits are retained through CE backward; duplicate this
    # tensor for the conservative autograd workspace.  The embedding table is
    # resident on the device for every step.
    analytical_peak = (
        int(embedding_shape[0]) * hidden_size * 4
        + activation_bytes
        + attention_scores_bytes
        + mlp_workspace_bytes
        + 2 * selected_logits_bytes
        + parameter_bytes["residual_adam_states_and_gradients_fp32"]
    )
    conservative_peak = max(
        int(analytical_peak * 1.5),
        int(MEASURED_TRR0005_FLOOR_BYTES),
    )
    available_host = _host_available_bytes()
    result: dict[str, Any] = {
        "schema": "token-reconstruction.trr0007-resource-preflight.v1",
        "task_id": TASK_ID,
        "created_utc": _utc_now(),
        "mode": "metadata_only",
        "device_requested": str(args.device),
        "fixed_recipe": {
            "hidden_size": hidden_size,
            "vocabulary_size": vocabulary_size,
            "context_width": int(args.context_width),
            "sequence_length": sequence_length,
            "record_batch_size": records_per_batch,
            "position_budget": int(args.position_budget),
            "bottleneck_size": bottleneck,
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "seed": int(args.seed),
        },
        "manifest_geometry": {
            "fit_observations": list(fit_shape),
            "validation_observations": list(validation_shape),
            "embedding_table": list(embedding_shape),
            "fit_records": int(fit_shape[0]),
            "validation_records": int(validation_shape[0]),
            "fit_resources": fit_shapes,
            "validation_resources": validation_shapes,
        },
        "materialized_batch_geometry": {
            "activation": [records_per_batch, positions_per_batch, hidden_size],
            "valid_mask": [records_per_batch, positions_per_batch],
            "labels": [records_per_batch, positions_per_batch],
            "draws": [int(args.position_budget)],
            "selected_logits": [int(args.position_budget), vocabulary_size],
            "note": (
                "This is the trainer's actual gathered-record batch contract; "
                "runtime receipts must repeat the observed activation and draw shapes."
            ),
        },
        "parameter_counts": {
            "current_family": int(base_parameters),
            "residual_mlp_added": int(mlp_parameters),
            "residual_mlp_total": int(base_parameters + mlp_parameters),
        },
        "bytes": {
            **parameter_bytes,
            "resident_embedding_fp32": int(embedding_shape[0] * hidden_size * 4),
            "materialized_activation_fp32": int(activation_bytes),
            "diagonal_attention_scores_fp32": int(attention_scores_bytes),
            "mlp_workspace_fp32": int(mlp_workspace_bytes),
            "selected_logits_fp32": int(selected_logits_bytes),
            "analytical_peak_fp32": int(analytical_peak),
            "conservative_peak_fp32": int(conservative_peak),
            "measured_trr0005_peak_fp32": int(MEASURED_TRR0005_PEAK_BYTES),
            "measured_trr0005_1_5x_floor_fp32": int(MEASURED_TRR0005_FLOOR_BYTES),
        },
        "safety": {
            "minimum_host_available_bytes": int(args.minimum_host_available_gib * 1024**3),
            "observed_host_available_bytes": available_host,
            "host_margin_after_conservative_peak_bytes": (
                None if available_host is None else int(available_host - conservative_peak)
            ),
            "resource_guard": _resource_policy(args),
            "qualified_largest_representative_batch": False,
            "qualification_required_before_long_fit": True,
        },
        "inputs": {
            "fit_manifest": _file_record(fit_manifest),
            "validation_manifest": _file_record(validation_manifest),
            "retained_reference": _file_record(
                Path(args.retained_reference), hash_bytes=True
            ),
        },
        "model_contract": {
            "methods": list(METHODS),
            "base_method_id": BASE_METHOD_ID,
            "retained_reference_sha256": BASE_STATE_SHA256,
            "inference": "current_H_i_only; normalized public E; full vocabulary CE",
            "initialization": (
                "both crossed arms use neutral identity W, zero b, s=3, "
                "deterministic diagonal QKV, zero added output"
            ),
        },
    }
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        if output_path.exists() or output_path.is_symlink():
            raise TRR0007FitError(f"preflight is create-only: {output_path}")
        _json_write(output_path, result)
    return result


def _geometry_signature(preflight: Mapping[str, Any]) -> dict[str, Any]:
    geometry = preflight.get("manifest_geometry")
    if not isinstance(geometry, Mapping):
        raise TRR0007FitError("preflight omitted manifest geometry")
    return {
        key: geometry.get(key)
        for key in ("fit_observations", "validation_observations", "embedding_table")
    }


def _validate_args(args: argparse.Namespace) -> None:
    expected = {
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "vocabulary_size": 128256,
        "context_width": DEFAULT_CONTEXT_WIDTH,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "record_batch_size": DEFAULT_RECORD_BATCH_SIZE,
        "position_budget": DEFAULT_POSITION_BUDGET,
        "seed": TRR5_DEFAULT_SEED,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "gradient_clip_norm": DEFAULT_GRADIENT_CLIP_NORM,
        "validation_every": DEFAULT_VALIDATION_EVERY,
        "bottleneck_size": DEFAULT_BOTTLENECK_SIZE,
    }
    for name, fixed in expected.items():
        value = getattr(args, name)
        if isinstance(fixed, float):
            if not math.isclose(float(value), fixed, rel_tol=0.0, abs_tol=1e-12):
                raise TRR0007FitError(f"{name} is fixed at {fixed}")
        elif int(value) != int(fixed):
            raise TRR0007FitError(f"{name} is fixed at {fixed}")
    if not (1 <= int(args.steps) <= DEFAULT_STEPS):
        raise TRR0007FitError(f"steps must be within 1..{DEFAULT_STEPS}")
    if not (1 <= int(args.qualification_steps) <= MAX_QUALIFICATION_STEPS):
        raise TRR0007FitError(
            f"qualification_steps must be within 1..{MAX_QUALIFICATION_STEPS}"
        )
    if int(args.challenge_rows) <= 0:
        raise TRR0007FitError("challenge_rows must be positive")
    for name in (
        "maximum_host_rss_gib", "minimum_host_available_gib",
        "minimum_free_gib", "maximum_gpu_reserved_gib", "max_seconds",
    ):
        if float(getattr(args, name)) <= 0:
            raise TRR0007FitError(f"{name} must be positive")
    bank_mode = getattr(args, "banks", "both")
    if bank_mode not in ("both", "enriched", "improved"):
        raise TRR0007FitError(f"unknown bank mode: {bank_mode}")
    if bank_mode in ("both", "enriched") and (
        args.fit_manifest is None or args.validation_manifest is None
    ):
        raise TRR0007FitError("fit and validation manifests are required for the enriched bank")
    if bank_mode in ("both", "improved") and (
        args.improved_fit_manifest is None or args.improved_validation_manifest is None
    ):
        raise TRR0007FitError("improved bank manifests are required for the selected bank mode")
    if args.preflight_only and args.fit_manifest is None:
        raise TRR0007FitError("preflight needs manifests")


def _record_resource_paths(data: PublicJointData) -> dict[str, Any]:
    paths = data.metadata.get("fit_paths", {})
    validation = data.metadata.get("validation_paths", {})
    return {
        "fit": paths if isinstance(paths, Mapping) else {},
        "validation": validation if isinstance(validation, Mapping) else {},
    }


def _sampler_receipt(schedule: PositionSchedule, *, actual_geometry: Mapping[str, Any]) -> dict[str, Any]:
    values = schedule_metadata(schedule)
    return {
        "schema": "token-reconstruction.trr0007-sampler-receipt.v1",
        "task_id": TASK_ID,
        "created_utc": _utc_now(),
        "schedule": values,
        "actual_gathered_geometry": dict(actual_geometry),
        "same_ordered_schedule_for_methods": True,
        "replacement_allowed_only_when_batch_has_fewer_than_position_budget": True,
    }


def _actual_geometry(data: PublicJointData, args: argparse.Namespace) -> dict[str, Any]:
    fit = tuple(int(value) for value in data.fit_observations.shape)
    mask = tuple(int(value) for value in data.fit_valid_mask.shape)
    truth = tuple(int(value) for value in data.fit_truth.shape)
    if fit != (fit[0], int(args.sequence_length), int(args.hidden_size)):
        raise TRR0007FitError(f"fit activation geometry differs: {fit}")
    if mask != fit[:2] or truth != fit[:2]:
        raise TRR0007FitError(
            f"fit mask/label geometry differs: activation={fit}, mask={mask}, truth={truth}"
        )
    if int(args.record_batch_size) > int(fit[0]):
        raise TRR0007FitError(
            "the actual gathered-record batch would require replacement records; "
            f"fit_records={fit[0]}, batch={args.record_batch_size}"
        )
    return {
        "activation": [int(args.record_batch_size), fit[1], fit[2]],
        "valid_mask": [int(args.record_batch_size), fit[1]],
        "labels": [int(args.record_batch_size), fit[1]],
        "schedule_batch_record_indices": [
            int(args.record_batch_size)
        ],
        "draws": [int(args.position_budget)],
        "fit_storage": list(fit),
        "validation_storage": [
            int(value) for value in data.validation_observations.shape
        ],
        "batching": "gather exact schedule record indices, then materialize float32 H",
    }


def _check_data(data: PublicJointData, args: argparse.Namespace) -> dict[str, Any]:
    geometry = _actual_geometry(data, args)
    if int(data.embedding_table.shape[0]) != 128256 or int(data.embedding_table.shape[1]) != 2048:
        raise TRR0007FitError(
            f"public normalized E must be [128256,2048], got {tuple(data.embedding_table.shape)}"
        )
    if not data.fit_valid_mask[:, 0].all().item():
        raise TRR0007FitError("fit masks must include BOS")
    if not data.validation_valid_mask[:, 0].all().item():
        raise TRR0007FitError("validation masks must include BOS")
    validation_shape = tuple(int(value) for value in data.validation_observations.shape)
    if validation_shape[1:] != (int(args.sequence_length), int(args.hidden_size)):
        raise TRR0007FitError(
            f"validation activation geometry differs: {validation_shape}"
        )
    if len(data.validation_groups) != int(data.validation_observations.shape[0]):
        raise TRR0007FitError("validation groups do not match validation rows")
    # The bank manifest declares public E normalized.  Check a deterministic
    # sample of rows at runtime without allocating a second full-table norm.
    sample_stride = max(1, int(data.embedding_table.shape[0]) // 4096)
    sample = data.embedding_table[::sample_stride]
    sample_norms = sample.float().norm(dim=-1)
    if not torch.isfinite(sample_norms).all().item() or not torch.allclose(
        sample_norms, torch.ones_like(sample_norms), rtol=2e-3, atol=2e-3
    ):
        raise TRR0007FitError("public embedding table is not normalized")
    fit_payload = data.metadata.get("fit_payload", {})
    if isinstance(fit_payload, Mapping) and fit_payload.get("embedding_table_normalized") is False:
        raise TRR0007FitError("fit manifest does not declare normalized public E")
    return geometry


def _collect_wrong_mask(
    model: torch.nn.Module,
    data: PublicJointData,
    *,
    device: torch.device,
    position_budget: int,
    record_batch_size: int,
) -> torch.Tensor:
    """Return public fit positions misclassified by the neutral model."""
    wrong = torch.zeros_like(data.fit_valid_mask, dtype=torch.bool, device="cpu")
    runtime_embedding = data.embedding_table.to(device=device, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(data.fit_observations.shape[0]), record_batch_size):
            stop = min(start + record_batch_size, int(data.fit_observations.shape[0]))
            activation = data.fit_observations[start:stop].to(device=device, dtype=torch.float32)
            mask = data.fit_valid_mask[start:stop].to(device=device, dtype=torch.bool)
            labels = data.fit_truth[start:stop].to(device=device, dtype=torch.long)
            hidden = model.projected_hidden(activation, mask)
            indices = torch.nonzero(mask, as_tuple=False)
            indices = indices[indices[:, 1] > 0]
            for chunk in indices.split(position_budget):
                logits = model.logits_from_rows(
                    hidden, chunk[:, 0], chunk[:, 1], runtime_embedding
                )
                prediction = logits.argmax(dim=-1)
                for local_index, hit in enumerate(prediction.eq(labels[chunk[:, 0], chunk[:, 1]])):
                    if not bool(hit.item()):
                        wrong[start + int(chunk[local_index, 0]), int(chunk[local_index, 1])] = True
    return wrong


def _select_challenge(
    wrong_mask: torch.Tensor,
    *,
    cap: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    wrong_mask = wrong_mask.to(device="cpu", dtype=torch.bool)
    wrong_mask[:, 0] = False
    indices = torch.nonzero(wrong_mask, as_tuple=False)
    total = int(indices.shape[0])
    if total <= cap:
        selected = indices
        selection_rule = "all_initially_wrong_rows"
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        chosen = torch.randperm(total, generator=generator)[: int(cap)]
        selected = indices.index_select(0, chosen)
        position_base = int(wrong_mask.shape[1])
        order = torch.argsort(
            selected[:, 0].to(torch.int64) * position_base
            + selected[:, 1].to(torch.int64)
        )
        selected = selected.index_select(0, order)
        selection_rule = "seeded_uniform_initially_wrong_rows"
    mask = torch.zeros_like(wrong_mask)
    if int(selected.shape[0]):
        mask[selected[:, 0], selected[:, 1]] = True
    return mask, {
        "all_initially_wrong_rows": total,
        "selected_rows": int(selected.shape[0]),
        "cap": int(cap),
        "selection_rule": selection_rule,
        "selection_seed": int(seed),
        "mask_sha256": tensor_sha256(mask.to(dtype=torch.uint8)),
        "initial_selected_accuracy": 0.0 if int(selected.shape[0]) else None,
        "empty_challenge": not bool(total),
    }


def _selected_metrics(
    model: torch.nn.Module,
    observations: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    device: torch.device,
    record_batch_size: int,
    position_budget: int,
) -> dict[str, Any]:
    selected_mask = selected_mask.to(device="cpu", dtype=torch.bool)
    indices = torch.nonzero(selected_mask, as_tuple=False)
    if int(indices.shape[0]) == 0:
        return {
            "loss": None,
            "token_accuracy": None,
            "correct_tokens": 0,
            "token_rows": 0,
            "empty": True,
        }
    total_loss = 0.0
    total_correct = 0
    runtime_embedding = embedding_table.to(device=device, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(observations.shape[0]), record_batch_size):
            stop = min(start + record_batch_size, int(observations.shape[0]))
            local = indices[(indices[:, 0] >= start) & (indices[:, 0] < stop)].clone()
            if int(local.shape[0]) == 0:
                continue
            local[:, 0] -= start
            activation = observations[start:stop].to(device=device, dtype=torch.float32)
            mask = valid_mask[start:stop].to(device=device, dtype=torch.bool)
            labels = truth[start:stop].to(device=device, dtype=torch.long)
            hidden = model.projected_hidden(activation, mask)
            for chunk in local.split(position_budget):
                logits = model.logits_from_rows(
                    hidden, chunk[:, 0], chunk[:, 1], runtime_embedding
                )
                target = labels[chunk[:, 0], chunk[:, 1]]
                total_loss += float(F.cross_entropy(logits, target, reduction="sum").cpu())
                total_correct += int(logits.argmax(dim=-1).eq(target).sum().cpu())
    rows = int(indices.shape[0])
    return {
        "loss": total_loss / rows,
        "token_accuracy": total_correct / rows,
        "correct_tokens": total_correct,
        "token_rows": rows,
        "empty": False,
    }


def _train_step(
    model: torch.nn.Module,
    data: PublicJointData,
    schedule: PositionSchedule,
    step_index: int,
    *,
    device: torch.device,
    runtime_embedding: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float,
) -> dict[str, Any]:
    if int(step_index) < 0 or int(step_index) >= schedule.steps:
        raise TRR0007FitError(f"schedule step is out of range: {step_index}")
    model.train()
    batch_indices = schedule.batch_record_indices[step_index]
    activation = data.fit_observations.index_select(0, batch_indices).to(
        device=device, dtype=torch.float32
    )
    valid_mask = data.fit_valid_mask.index_select(0, batch_indices).to(
        device=device, dtype=torch.bool
    )
    labels = data.fit_truth.index_select(0, batch_indices).to(
        device=device, dtype=torch.long
    )
    record_slots = schedule.draw_record_slots[step_index].to(device=device)
    position_slots = schedule.draw_position_slots[step_index].to(device=device)
    if tuple(activation.shape) != (
        int(schedule.record_batch_size),
        int(data.fit_observations.shape[1]),
        int(data.fit_observations.shape[2]),
    ):
        raise TRR0007FitError(f"materialized activation geometry changed: {tuple(activation.shape)}")
    if int(record_slots.numel()) != int(schedule.position_budget):
        raise TRR0007FitError("schedule draw count changed")
    if position_slots.eq(0).any().item() or (~valid_mask[record_slots, position_slots]).any().item():
        raise TRR0007FitError("schedule contains an invalid or BOS draw")
    hidden = model.projected_hidden(activation, valid_mask)
    logits = model.logits_from_rows(
        hidden, record_slots, position_slots, runtime_embedding
    )
    target = labels[record_slots, position_slots]
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(logits, target)
    if not torch.isfinite(loss).item():
        raise TRR0007FitError("training loss is non-finite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(model.parameters()), gradient_clip_norm, error_if_nonfinite=True
    )
    optimizer.step()
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all().item():
            raise TRR0007FitError("model parameter became non-finite")
    predictions = logits.detach().argmax(dim=-1)
    gradients = {
        name: (
            None
            if parameter.grad is None
            else float(parameter.grad.detach().norm().cpu())
        )
        for name, parameter in model.named_parameters()
    }
    return {
        "step": int(step_index + 1),
        "loss": float(loss.detach().cpu()),
        "token_accuracy": float(predictions.eq(target).float().mean().cpu()),
        "correct_tokens": int(predictions.eq(target).sum().cpu()),
        "token_rows": int(target.numel()),
        "gradient_norm": float(grad_norm.detach().cpu()),
        "gradient_norms": gradients,
        "record_batch_indices": batch_indices.tolist(),
        "activation_shape": [int(value) for value in activation.shape],
        "valid_mask_shape": [int(value) for value in valid_mask.shape],
        "label_shape": [int(value) for value in labels.shape],
        "draw_shape": [int(record_slots.numel())],
        "unique_records_in_batch": int(torch.unique(batch_indices).numel()),
        "used_replacement": bool(schedule.used_replacement[step_index].item()),
        "eligible_positions": int(schedule.eligible_counts[step_index].item()),
    }


def _build_model(method_id: str, args: argparse.Namespace) -> torch.nn.Module:
    if method_id == CURRENT_METHOD_ID:
        return build_current_positionwise(
            hidden_size=int(args.hidden_size),
            vocabulary_size=int(args.vocabulary_size),
            context_width=int(args.context_width),
            seed=int(args.seed),
        )
    if method_id == RESIDUAL_MLP_METHOD_ID:
        return build_residual_mlp512(
            hidden_size=int(args.hidden_size),
            vocabulary_size=int(args.vocabulary_size),
            context_width=int(args.context_width),
            bottleneck_size=int(args.bottleneck_size),
            seed=int(args.seed),
        )
    raise TRR0007FitError(f"unregistered method: {method_id}")


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _fit_method(
    method_id: str,
    data: PublicJointData,
    challenge_mask: torch.Tensor,
    challenge_receipt: Mapping[str, Any],
    schedule: PositionSchedule,
    *,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
    run_steps: int,
    bank_name: str,
    deadline: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = _build_model(method_id, args).to(device=device)
    parameter_count = sum(int(value.numel()) for value in model.parameters())
    runtime_embedding = data.embedding_table.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(run_steps)), eta_min=0.0
    )
    stage_times = {
        "train_seconds": 0.0,
        "validation_seconds": 0.0,
        "challenge_seconds": 0.0,
    }
    curve: list[dict[str, Any]] = []
    checkpoints = set(checkpoint_steps(int(run_steps)))
    # Measure and retain the common neutral initialization before any update.
    zero_started = time.perf_counter()
    zero_validation = evaluate_dataset(
        model,
        data.validation_observations,
        data.validation_truth,
        data.validation_valid_mask,
        runtime_embedding,
        data.validation_groups,
        device=device,
        record_batch_size=int(args.record_batch_size),
        position_budget=int(args.position_budget),
    )
    stage_times["validation_seconds"] += time.perf_counter() - zero_started
    zero_challenge_started = time.perf_counter()
    zero_challenge = _selected_metrics(
        model,
        data.fit_observations,
        data.fit_truth,
        data.fit_valid_mask,
        challenge_mask,
        runtime_embedding,
        device=device,
        record_batch_size=int(args.record_batch_size),
        position_budget=int(args.position_budget),
    )
    stage_times["challenge_seconds"] += time.perf_counter() - zero_challenge_started
    curve.append({
        "step": 0,
        "train": None,
        "validation": zero_validation,
        "challenge_initially_wrong": zero_challenge,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    })
    best_score = float(zero_validation["style_balanced_token_accuracy"])
    best_step = 0
    best_state = _clone_state(model)
    last_train: dict[str, Any] | None = None
    for step_index in range(int(run_steps)):
        _resource_guard(
            args,
            device,
            stage=f"{bank_name}/{method_id}/step_{step_index}",
            deadline=deadline,
        )
        train_started = time.perf_counter()
        train_receipt = _train_step(
            model,
            data,
            schedule,
            step_index,
            device=device,
            runtime_embedding=runtime_embedding,
            optimizer=optimizer,
            gradient_clip_norm=float(args.gradient_clip_norm),
        )
        scheduler.step()
        stage_times["train_seconds"] += time.perf_counter() - train_started
        last_train = train_receipt
        completed_step = int(step_index + 1)
        if completed_step not in checkpoints:
            continue
        validation_started = time.perf_counter()
        validation = evaluate_dataset(
            model,
            data.validation_observations,
            data.validation_truth,
            data.validation_valid_mask,
            runtime_embedding,
            data.validation_groups,
            device=device,
            record_batch_size=int(args.record_batch_size),
            position_budget=int(args.position_budget),
        )
        stage_times["validation_seconds"] += time.perf_counter() - validation_started
        challenge_started = time.perf_counter()
        challenge = _selected_metrics(
            model,
            data.fit_observations,
            data.fit_truth,
            data.fit_valid_mask,
            challenge_mask,
            runtime_embedding,
            device=device,
            record_batch_size=int(args.record_batch_size),
            position_budget=int(args.position_budget),
        )
        stage_times["challenge_seconds"] += time.perf_counter() - challenge_started
        curve.append({
            "step": completed_step,
            "train": train_receipt,
            "validation": validation,
            "challenge_initially_wrong": challenge,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        score = float(validation["style_balanced_token_accuracy"])
        # Strict greater-than implements the registered earliest-maximum rule.
        if score > best_score:
            best_score = score
            best_step = completed_step
            best_state = _clone_state(model)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    final_fit = evaluate_dataset(
        model,
        data.fit_observations,
        data.fit_truth,
        data.fit_valid_mask,
        runtime_embedding,
        tuple("fit" for _ in range(int(data.fit_observations.shape[0]))),
        device=device,
        record_batch_size=int(args.record_batch_size),
        position_budget=int(args.position_budget),
    )
    final_challenge = _selected_metrics(
        model,
        data.fit_observations,
        data.fit_truth,
        data.fit_valid_mask,
        challenge_mask,
        runtime_embedding,
        device=device,
        record_batch_size=int(args.record_batch_size),
        position_budget=int(args.position_budget),
    )
    method_dir = output_root / bank_name / method_id
    selected_path = method_dir / "selected.safetensors"
    state_receipt = save_positionwise_state(
        selected_path,
        model,
        method_id=method_id,
        selected_step=int(best_step),
        initialization=(
            "TRR-0005 neutral identity-affine W/b/s=3; deterministic diagonal "
            "QKV seed 4005; residual up zero"
        ),
        distribution=bank_name,
        bottleneck_size=(
            int(args.bottleneck_size) if method_id == RESIDUAL_MLP_METHOD_ID else None
        ),
        metadata={
            "task_id": TASK_ID,
            "bank": bank_name,
            "base_method_id": BASE_METHOD_ID,
            "retained_reference_sha256": BASE_STATE_SHA256,
            "selection_metric": "earliest maximum validation style_balanced_token_accuracy",
            "schedule_sha256": schedule_digest(schedule),
            "challenge_mask_sha256": challenge_receipt["mask_sha256"],
            "training_steps": int(run_steps),
            "current_H_only": True,
            "public_embedding_normalized": True,
            "full_vocabulary_cross_entropy": True,
        },
    )
    curve_path = method_dir / "learning_curve.json"
    _json_write(curve_path, {
        "schema": "token-reconstruction.trr7-learning-curve.v1",
        "task_id": TASK_ID,
        "method_id": method_id,
        "bank": bank_name,
        "parameter_count": parameter_count,
        "selected_step": int(best_step),
        "selected_validation_style_balanced_token_accuracy": best_score,
        "selection": "earliest maximum of validation style-balanced token accuracy",
        "challenge_definition": dict(challenge_receipt),
        "points": curve,
    })
    elapsed = time.perf_counter() - started
    result = {
        "method_id": method_id,
        "bank": bank_name,
        "parameter_count": parameter_count,
        "selected_step": int(best_step),
        "selected_state": state_receipt,
        "learning_curve": {
            "path": str(curve_path),
            "bytes": int(curve_path.stat().st_size),
            "sha256": file_sha256(curve_path),
            "points": len(curve),
        },
        "selected_metrics": {
            "fit": final_fit,
            "challenge_initially_wrong": final_challenge,
            "selected_step": int(best_step),
            "selected_validation_style_balanced_token_accuracy": best_score,
        },
        "challenge": dict(challenge_receipt),
        "schedule_sha256": schedule_digest(schedule),
        "runtime": {
            **stage_times,
            "fit_wall_seconds": elapsed,
            "train_step_count": int(run_steps),
            "peak_memory": _host_memory_snapshot(),
            "last_train_geometry": last_train,
            "device": str(device),
        },
        "failure": None,
    }
    _resource_guard(
        args,
        device,
        stage=f"{bank_name}/{method_id}/complete",
        deadline=deadline,
    )
    del optimizer
    del scheduler
    del runtime_embedding
    del model
    gc.collect()
    return result

def _run_bank(
    bank_name: str,
    fit_manifest: Path,
    validation_manifest: Path,
    *,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
    run_steps: int,
    deadline: float,
) -> dict[str, Any]:
    bank_started = time.perf_counter()
    _resource_guard(args, device, stage=f"{bank_name}/before_load", deadline=deadline)
    data = load_public_joint_data(
        fit_manifest,
        validation_manifest,
        embedding_path=Path(args.embedding_path).expanduser().resolve()
        if args.embedding_path is not None
        else None,
    )
    geometry = _check_data(data, args)
    _resource_guard(args, device, stage=f"{bank_name}/after_load", deadline=deadline)
    schedule = build_position_schedule(
        data.fit_valid_mask,
        steps=int(run_steps),
        record_batch_size=int(args.record_batch_size),
        position_budget=int(args.position_budget),
        seed=int(args.seed),
    )
    bank_dir = output_root / bank_name
    bank_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = bank_dir / "schedule.safetensors"
    schedule_receipt = save_schedule(schedule_path, schedule)
    sampler_receipt = _sampler_receipt(schedule, actual_geometry=geometry)
    sampler_receipt["schedule_file"] = schedule_receipt
    _json_write(bank_dir / "sampler_receipt.json", sampler_receipt)
    runtime_embedding = data.embedding_table.to(device=device, dtype=torch.float32)
    reference_started = time.perf_counter()
    retained_reference = load_retained_diagonal_state(
        Path(args.retained_reference),
        hidden_size=int(args.hidden_size),
        vocabulary_size=int(data.vocabulary_size),
        context_width=int(args.context_width),
    ).to(device=device)
    retained_fit = evaluate_dataset(
        retained_reference,
        data.fit_observations,
        data.fit_truth,
        data.fit_valid_mask,
        runtime_embedding,
        tuple("fit" for _ in range(int(data.fit_observations.shape[0]))),
        device=device,
        record_batch_size=int(args.record_batch_size),
        position_budget=int(args.position_budget),
    )
    retained_validation = evaluate_dataset(
        retained_reference,
        data.validation_observations,
        data.validation_truth,
        data.validation_valid_mask,
        runtime_embedding,
        data.validation_groups,
        device=device,
        record_batch_size=int(args.record_batch_size),
        position_budget=int(args.position_budget),
    )
    reference_receipt = {
        "path": str(Path(args.retained_reference).expanduser().resolve()),
        "sha256": BASE_STATE_SHA256,
        "method_id": BASE_METHOD_ID,
        "fit_metrics": retained_fit,
        "validation_metrics": retained_validation,
        "measured_selected_state_fit_accuracy": retained_fit["token_accuracy"],
        "seconds": time.perf_counter() - reference_started,
        "separate_frozen_evaluation_reference": True,
    }
    del retained_reference
    neutral_current = _build_model(CURRENT_METHOD_ID, args).to(device=device)
    neutral_extension = _build_model(RESIDUAL_MLP_METHOD_ID, args).to(device=device)
    equivalence = step_zero_equivalence(
        neutral_current,
        neutral_extension,
        data.fit_observations[: min(8, int(data.fit_observations.shape[0]))].to(
            device=device, dtype=torch.float32
        ),
        data.fit_valid_mask[: min(8, int(data.fit_valid_mask.shape[0]))].to(
            device=device, dtype=torch.bool
        ),
        runtime_embedding,
        max_rows=min(int(args.position_budget), 512),
    )
    if not equivalence["projected_hidden_exact"] or not equivalence["logits_exact"]:
        raise TRR0007FitError(
            f"neutral current/extension mismatch in {bank_name}: {equivalence}"
        )
    wrong_mask = _collect_wrong_mask(
        neutral_current,
        data,
        device=device,
        position_budget=int(args.position_budget),
        record_batch_size=int(args.record_batch_size),
    )
    challenge_mask, challenge_receipt = _select_challenge(
        wrong_mask,
        cap=int(args.challenge_rows),
        seed=int(args.seed) + 7007,
    )
    if challenge_receipt["empty_challenge"]:
        raise TRR0007FitError(
            f"{bank_name} neutral current model solved every public fit row; "
            "capacity challenge would be uninformative"
        )
    _json_write(bank_dir / "challenge_receipt.json", {
        "schema": "token-reconstruction.trr0007-challenge-receipt.v1",
        "task_id": TASK_ID,
        "bank": bank_name,
        "definition": (
            "positions initially wrong under the common neutral diagonal current "
            "model, before either crossed method is optimized"
        ),
        "equivalence": equivalence,
        **challenge_receipt,
    })
    del neutral_current
    del neutral_extension
    method_results: dict[str, Any] = {}
    for method_id in METHODS:
        method_results[method_id] = _fit_method(
            method_id,
            data,
            challenge_mask,
            challenge_receipt,
            schedule,
            args=args,
            device=device,
            output_root=output_root,
            run_steps=int(run_steps),
            bank_name=bank_name,
            deadline=deadline,
        )
    bank_result = {
        "schema": "token-reconstruction.trr0007-bank-result.v1",
        "task_id": TASK_ID,
        "bank": bank_name,
        "fit_manifest": _file_record(Path(fit_manifest)),
        "validation_manifest": _file_record(Path(validation_manifest)),
        "embedding_table": _file_record(
            Path(data.metadata["fit_paths"]["embedding_table"]["path"])
        ),
        "geometry": geometry,
        "schedule": schedule_receipt,
        "sampler_receipt": sampler_receipt,
        "reference": reference_receipt,
        "neutral_initialization_equivalence": equivalence,
        "challenge": challenge_receipt,
        "methods": method_results,
        "same_schedule_for_methods": all(
            result["schedule_sha256"] == schedule_digest(schedule)
            for result in method_results.values()
        ),
        "run_steps": int(run_steps),
        "wall_seconds": time.perf_counter() - bank_started,
        "resource_after_bank": _resource_guard(
            args, device, stage=f"{bank_name}/after_fit", deadline=deadline
        ),
    }
    _json_write(bank_dir / "bank_result.json", bank_result)
    del runtime_embedding
    del data
    del schedule
    del challenge_mask
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bank_result


def _write_failure(output_root: Path, exc: BaseException, *, started: str) -> None:
    path = output_root / "failure.json"
    if path.exists() or path.is_symlink():
        return
    _json_write(path, {
        "schema": "token-reconstruction.trr0007-failure.v1",
        "task_id": TASK_ID,
        "started_utc": started,
        "failed_utc": _utc_now(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "output_root": str(output_root.resolve()),
    })


def _manifest_namespace(
    args: argparse.Namespace,
    fit_manifest: Path,
    validation_manifest: Path,
) -> argparse.Namespace:
    result = argparse.Namespace(**vars(args))
    result.fit_manifest = fit_manifest
    result.validation_manifest = validation_manifest
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_root = Path(args.output).expanduser().resolve()
    bank_mode = getattr(args, "banks", "both")
    improved_args = (
        _manifest_namespace(
            args,
            Path(args.improved_fit_manifest),
            Path(args.improved_validation_manifest),
        )
        if bank_mode in ("both", "improved")
        else None
    )
    selected_args = improved_args if bank_mode == "improved" else args
    assert selected_args is not None
    if args.preflight_only:
        output_path = output_root / "resource_preflight.json"
        if output_path.exists() or output_path.is_symlink():
            raise TRR0007FitError(f"preflight output is create-only: {output_path}")
        output_root.mkdir(parents=True, exist_ok=True)
        primary = _resource_preflight(selected_args)
        improved = None
        if bank_mode == "both":
            assert improved_args is not None
            improved = _resource_preflight(improved_args)
            if _geometry_signature(primary) != _geometry_signature(improved):
                raise TRR0007FitError("crossed banks do not share exact tensor geometry")
        payload = {
            "schema": (
                "token-reconstruction.trr0007-crossed-resource-preflight.v1"
                if bank_mode == "both"
                else "token-reconstruction.trr0007-resource-preflight.v1"
            ),
            "task_id": TASK_ID,
            "created_utc": _utc_now(),
            "bank_mode": bank_mode,
            "primary": primary,
            "improved": improved,
            "qualification_required_before_long_fit": True,
            "largest_representative_batch_qualified": False,
        }
        _json_write(output_path, payload)
        return {"preflight": payload}
    if output_root.exists() and any(output_root.iterdir()):
        raise TRR0007FitError(f"output root must be new and empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started_utc = _utc_now()
    started = time.perf_counter()
    deadline = started + float(args.max_seconds)
    device = _choose_device(args.device)
    _resource_guard(args, device, stage="before_preflight", deadline=deadline)
    primary_preflight = _resource_preflight(selected_args)
    improved_preflight = None
    if bank_mode == "both":
        assert improved_args is not None
        improved_preflight = _resource_preflight(improved_args)
        if _geometry_signature(primary_preflight) != _geometry_signature(improved_preflight):
            raise TRR0007FitError("crossed banks do not share exact tensor geometry")
    _json_write(output_root / "resource_preflight.json", {
        "schema": (
            "token-reconstruction.trr0007-crossed-resource-preflight.v1"
            if bank_mode == "both"
            else "token-reconstruction.trr0007-resource-preflight.v1"
        ),
        "task_id": TASK_ID,
        "created_utc": _utc_now(),
        "bank_mode": bank_mode,
        "primary": primary_preflight,
        "improved": improved_preflight,
        "qualification_required_before_long_fit": True,
        "largest_representative_batch_qualified": False,
    })
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        run_steps = int(args.qualification_steps if args.qualification_only else args.steps)
        banks: dict[str, Any] = {}
        if bank_mode in ("both", "enriched"):
            banks["current_enriched"] = _run_bank(
                "current_enriched",
                Path(args.fit_manifest),
                Path(args.validation_manifest),
                args=args,
                device=device,
                output_root=output_root,
                run_steps=run_steps,
                deadline=deadline,
            )
        if bank_mode in ("both", "improved"):
            assert improved_args is not None
            banks["improved_public_bank"] = _run_bank(
                "improved_public_bank",
                Path(args.improved_fit_manifest),
                Path(args.improved_validation_manifest),
                args=improved_args,
                device=device,
                output_root=output_root,
                run_steps=run_steps,
                deadline=deadline,
            )
    except BaseException as exc:
        _write_failure(output_root, exc, started=started_utc)
        raise
    finished_utc = _utc_now()
    evidence = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "bank_mode": bank_mode,
        "qualification_only": bool(args.qualification_only),
        "run_steps": run_steps,
        "args": {key: str(value) for key, value in vars(args).items()},
        "environment": _safe_environment(),
        "git_commit": _git_commit(Path.cwd()),
        "git_status": _git_status(Path.cwd()),
        "resource_preflight": str(output_root / "resource_preflight.json"),
        "banks": banks,
        "largest_representative_batch_qualified": bool(args.qualification_only),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        "resource_after": _resource_guard(
            args, device, stage="complete", deadline=deadline
        ),
    }
    _json_write(output_root / "run_evidence.json", evidence)
    return evidence

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        default=Path("experiments/TRR-0005/public_activation_v1/enriched_manifest.json"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("experiments/TRR-0005/public_activation_v1/enriched_manifest.json"),
    )
    parser.add_argument(
        "--improved-fit-manifest",
        type=Path,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--improved-validation-manifest",
        type=Path,
        required=False,
        default=None,
    )
    parser.add_argument("--embedding-path", type=Path, default=None)
    parser.add_argument("--retained-reference", type=Path, default=RETAINED_REFERENCE_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--banks",
        choices=("both", "enriched", "improved"),
        default="both",
        help="fit both crossed banks, or only one bank while the other is prepared",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--qualification-steps", type=int, default=DEFAULT_QUALIFICATION_STEPS)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--vocabulary-size", type=int, default=128256)
    parser.add_argument("--context-width", type=int, default=DEFAULT_CONTEXT_WIDTH)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--record-batch-size", type=int, default=DEFAULT_RECORD_BATCH_SIZE)
    parser.add_argument("--position-budget", type=int, default=DEFAULT_POSITION_BUDGET)
    parser.add_argument("--bottleneck-size", type=int, default=DEFAULT_BOTTLENECK_SIZE)
    parser.add_argument("--seed", type=int, default=TRR5_DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--validation-every", type=int, default=DEFAULT_VALIDATION_EVERY)
    parser.add_argument("--challenge-rows", type=int, default=DEFAULT_CHALLENGE_ROWS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_GIB)
    parser.add_argument(
        "--maximum-gpu-reserved-gib",
        type=float,
        default=DEFAULT_MAXIMUM_GPU_RESERVED_GIB,
    )
    parser.add_argument("--maximum-host-rss-gib", type=float, default=DEFAULT_MAXIMUM_HOST_RSS_GIB)
    parser.add_argument(
        "--minimum-host-available-gib",
        type=float,
        default=DEFAULT_MINIMUM_HOST_AVAILABLE_GIB,
    )
    parser.add_argument("--fit-records-hint", type=int, default=1200)
    parser.add_argument("--validation-records-hint", type=int, default=48)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"TRR-0007 fit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
