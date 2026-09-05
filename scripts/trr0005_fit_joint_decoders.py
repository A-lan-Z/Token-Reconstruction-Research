#!/usr/bin/env python3
"""Fit the six registered TRR-0005 joint decoder states.

The command runs three jointly trainable arms (affine, causal H attention,
and strict diagonal H attention) under each of two public fitting
distributions.  Every arm in a distribution receives one shared deterministic
8-record schedule and exactly 512 post-BOS cross-entropy draws per update.
The same public validation split is used for checkpoint selection.  No final
evaluation panel is loaded by this runner.

The command is intentionally separate from the TRR-0004 frozen-base runner.
It records the identity-affine and retained-TRR-0004 affine diagnostics on
both fitting distributions before joint training, and writes a complete
failure receipt if a guarded run stops.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
    AFFINE_METHOD,
    BOS_TOKEN_ID,
    CAUSAL_ATTENTION_METHOD,
    DATA_SCHEMA,
    DEFAULT_CONTEXT_WIDTH,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_POSITION_BUDGET,
    DEFAULT_RECORD_BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_STEPS,
    DEFAULT_VALIDATION_EVERY,
    VOCAB_SIZE,
    DEFAULT_WEIGHT_DECAY,
    DIAGONAL_ATTENTION_METHOD,
    JointDecoderError,
    METHODS,
    PublicJointData,
    build_decoder,
    build_position_schedule,
    checkpoint_steps,
    evaluate_dataset,
    file_sha256,
    load_decoder_state,
    tensor_sha256,
    load_public_joint_data,
    save_decoder_state,
    save_schedule,
    schedule_metadata,
    train_step,
)


TASK_ID = "TRR-0005"
SCHEMA = "token-reconstruction.trr0005-joint-fit.v1"
DEFAULT_MINIMUM_FREE_GIB = 8.0
DEFAULT_MAXIMUM_GPU_RESERVED_GIB = 6.0
DEFAULT_MAXIMUM_HOST_RSS_GIB = 16.0
DEFAULT_MINIMUM_HOST_AVAILABLE_GIB = 10.0
DEFAULT_MAX_SECONDS = 3600.0
DEFAULT_QUALIFICATION_STEPS = 2
MAX_QUALIFICATION_STEPS = 8
# V1 completed the registered causal largest cell but stopped because its
# analytic forecast was low. Keep this empirical calibration explicit and
# cite the preserved failure receipt; it does not relax the live guards.
QUALIFICATION_V1_MEASURED_PEAK_BYTES = 2_942_304_256
QUALIFICATION_MEASURED_FLOOR_MULTIPLIER = 1.5
QUALIFICATION_MEASURED_FLOOR_BYTES = math.ceil(
    QUALIFICATION_V1_MEASURED_PEAK_BYTES * QUALIFICATION_MEASURED_FLOOR_MULTIPLIER
)
QUALIFICATION_V1_FAILURE_RECEIPT = (
    "experiments/TRR-0005/joint_qualification_v1/failure.json"
)
DISTRIBUTION_CONTRACT_IDS = {
    "original": "original_like_alpaca_v1",
    "enriched": "coverage_mix_v1",
}
SCHEDULE_IDENTITY_FIELDS = (
    "fit_valid_mask_sha256",
    "batch_record_indices_sha256",
    "draw_record_slots_sha256",
    "draw_position_slots_sha256",
    "eligible_counts_sha256",
    "used_replacement_sha256",
    "schedule_sha256",
)


class JointFitRunnerError(RuntimeError):
    """Raised when a guarded TRR-0005 fit cannot proceed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_status(root: Path) -> list[str] | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in completed.stdout.splitlines() if line]


def _file_record(path: Path, *, hash_bytes: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise JointFitRunnerError(f"resource must be a regular file: {path}")
    result: dict[str, Any] = {"path": str(path), "bytes": int(path.stat().st_size)}
    if hash_bytes:
        result["sha256"] = file_sha256(path)
    return result


def _safe_environment() -> dict[str, str]:
    names = (
        "CUDA_VISIBLE_DEVICES",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise JointFitRunnerError(f"invalid device: {value}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise JointFitRunnerError("CUDA requested but unavailable")
    if device.type not in ("cpu", "cuda"):
        raise JointFitRunnerError("device must be cpu, cuda, or auto")
    return device


def _host_available_bytes() -> int | None:
    """Read host MemAvailable without importing another runtime dependency."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                return int(fields[1]) * 1024
    except (OSError, UnicodeError):
        return None
    return None


def _host_memory_snapshot() -> dict[str, Any]:
    rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    available_bytes = _host_available_bytes()
    return {
        "process_max_rss_bytes": rss_bytes,
        "host_available_bytes": available_bytes,
        "host_available_gib": (
            None if available_bytes is None else available_bytes / (1024**3)
        ),
    }


def _resource_policy(args: argparse.Namespace) -> dict[str, float]:
    return {
        "minimum_free_gpu_gib": float(args.minimum_free_gib),
        "maximum_gpu_reserved_gib": float(args.maximum_gpu_reserved_gib),
        "maximum_host_rss_gib": float(args.maximum_host_rss_gib),
        "minimum_host_available_gib": float(
            getattr(args, "minimum_host_available_gib", DEFAULT_MINIMUM_HOST_AVAILABLE_GIB)
        ),
    }


def _resource_guard(
    args: argparse.Namespace,
    device: torch.device,
    *,
    stage: str,
    deadline: float | None = None,
) -> dict[str, Any]:
    if deadline is not None and time.perf_counter() >= deadline:
        raise JointFitRunnerError(f"wall-time guard expired at {stage}")
    memory = _host_memory_snapshot()
    rss_bytes = int(memory["process_max_rss_bytes"])
    max_rss = int(args.maximum_host_rss_gib * 1024**3)
    if rss_bytes > max_rss:
        raise JointFitRunnerError(f"host RSS guard exceeded at {stage}: {rss_bytes} > {max_rss}")
    available_bytes = memory["host_available_bytes"]
    minimum_available_gib = float(
        getattr(args, "minimum_host_available_gib", DEFAULT_MINIMUM_HOST_AVAILABLE_GIB)
    )
    if available_bytes is not None and available_bytes < int(minimum_available_gib * 1024**3):
        raise JointFitRunnerError(
            f"host available-memory guard exceeded at {stage}: "
            f"{available_bytes} < {int(minimum_available_gib * 1024**3)}"
        )
    result: dict[str, Any] = {
        "stage": stage,
        **memory,
        "minimum_host_available_gib": minimum_available_gib,
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
        "cuda_reserved_bytes": None,
    }
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        reserved = int(torch.cuda.memory_reserved(device))
        result.update(
            {
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                "cuda_reserved_bytes": reserved,
            }
        )
        if free_bytes < int(args.minimum_free_gib * 1024**3):
            raise JointFitRunnerError(
                f"GPU free-memory guard exceeded at {stage}: {free_bytes} < {int(args.minimum_free_gib * 1024**3)}"
            )
        if reserved > int(args.maximum_gpu_reserved_gib * 1024**3):
            raise JointFitRunnerError(
                f"GPU reserved-memory guard exceeded at {stage}: {reserved} > {int(args.maximum_gpu_reserved_gib * 1024**3)}"
            )
    return result


def _manifest_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except Exception as exc:
        raise JointFitRunnerError(f"cannot read manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise JointFitRunnerError(f"manifest must be an object: {path}")
    if payload.get("schema") not in (DATA_SCHEMA, "token-reconstruction.trr0005-joint-fit-data.v1", "token-reconstruction.trr0004-public-fit-data.v1"):
        raise JointFitRunnerError(f"unsupported manifest schema in {path}: {payload.get('schema')}")
    return payload


def _shape_from_manifest(
    manifest: Mapping[str, Any],
    name: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    resources = manifest.get("resources")
    if isinstance(resources, Mapping):
        resource = resources.get(name)
        if isinstance(resource, Mapping):
            shape = resource.get("shape")
            if isinstance(shape, list) and all(isinstance(item, int) for item in shape):
                return tuple(int(item) for item in shape)
    return default


def resource_preflight(
    *,
    hidden_size: int,
    vocabulary_size: int,
    sequence_length: int,
    record_batch_size: int,
    position_budget: int,
    context_width: int,
) -> dict[str, Any]:
    """Calculate a conservative one-arm largest-cell memory envelope."""

    # All projection and optimizer quantities are explicit so the receipt can
    # be audited without loading a checkpoint.  The model arms execute one at
    # a time; the embedding table is resident once on the selected device.
    embedding_bytes = vocabulary_size * hidden_size * 4
    activation_bytes = record_batch_size * sequence_length * hidden_size * 4
    affine_parameters = hidden_size * hidden_size + hidden_size + 1
    attention_parameters = (
        3 * (hidden_size * context_width + context_width)
        + context_width * hidden_size
        + hidden_size
    )
    max_parameters = affine_parameters + attention_parameters
    parameter_bytes = max_parameters * 4
    optimizer_bytes = parameter_bytes * 2  # AdamW m and v; gradients are separate.
    gradient_bytes = parameter_bytes
    selected_logits_bytes = position_budget * vocabulary_size * 4
    attention_scores_bytes = record_batch_size * sequence_length * sequence_length * 4
    hidden_workspace_bytes = record_batch_size * sequence_length * hidden_size * 4 * 5
    # Backward retains at least the selected logits and their gradient/softmax
    # workspace.  Keep validation as a separately reported stage instead of
    # treating one forward logits allocation as the whole memory requirement.
    training_workspace_bytes = (
        activation_bytes
        + attention_scores_bytes
        + hidden_workspace_bytes
        + selected_logits_bytes * 2
    )
    validation_workspace_bytes = (
        activation_bytes
        + attention_scores_bytes
        + hidden_workspace_bytes
        + selected_logits_bytes
    )
    resident_training_bytes = (
        embedding_bytes
        + parameter_bytes
        + optimizer_bytes
        + gradient_bytes
    )
    training_peak_bytes = resident_training_bytes + training_workspace_bytes
    validation_peak_bytes = embedding_bytes + parameter_bytes + validation_workspace_bytes
    raw_sum = max(training_peak_bytes, validation_peak_bytes)
    safety_margin_bytes = math.ceil(raw_sum * 0.50)
    analytic_envelope_bytes = raw_sum + safety_margin_bytes
    measured_floor_bytes = int(QUALIFICATION_MEASURED_FLOOR_BYTES)
    envelope_bytes = max(analytic_envelope_bytes, measured_floor_bytes)
    gib = 1024**3
    return {
        "geometry": {
            "hidden_size": hidden_size,
            "vocabulary_size": vocabulary_size,
            "sequence_length": sequence_length,
            "record_batch_size": record_batch_size,
            "position_budget": position_budget,
            "context_width": context_width,
        },
        "bytes": {
            "embedding_table_fp32": embedding_bytes,
            "activation_batch_fp32": activation_bytes,
            "max_model_parameters_fp32": parameter_bytes,
            "adamw_m_v": optimizer_bytes,
            "gradient_buffer": gradient_bytes,
            "selected_logits_fp32": selected_logits_bytes,
            "selected_logits_backward_workspace_fp32": selected_logits_bytes * 2,
            "attention_scores_fp32": attention_scores_bytes,
            "hidden_workspace_envelope": hidden_workspace_bytes,
            "training_workspace_envelope": training_workspace_bytes,
            "validation_workspace_envelope": validation_workspace_bytes,
            "resident_training_envelope": resident_training_bytes,
            "training_peak_envelope": training_peak_bytes,
            "validation_peak_envelope": validation_peak_bytes,
            "raw_sum": raw_sum,
            "safety_margin_50_percent": safety_margin_bytes,
            "analytic_conservative_envelope": analytic_envelope_bytes,
            "measured_v1_qualification_peak": QUALIFICATION_V1_MEASURED_PEAK_BYTES,
            "measured_qualification_floor": measured_floor_bytes,
            "conservative_envelope": envelope_bytes,
        },
        "gib": {
            "embedding_table_fp32": embedding_bytes / gib,
            "selected_logits_fp32": selected_logits_bytes / gib,
            "training_peak_envelope": training_peak_bytes / gib,
            "validation_peak_envelope": validation_peak_bytes / gib,
            "raw_sum": raw_sum / gib,
            "safety_margin_50_percent": safety_margin_bytes / gib,
            "analytic_conservative_envelope": analytic_envelope_bytes / gib,
            "measured_v1_qualification_peak": QUALIFICATION_V1_MEASURED_PEAK_BYTES / gib,
            "measured_qualification_floor": measured_floor_bytes / gib,
            "conservative_envelope": envelope_bytes / gib,
        },
        "forecast_basis": {
            "resident_embedding": "full normalized public embedding table",
            "training_peak_bytes": training_peak_bytes,
            "validation_peak_bytes": validation_peak_bytes,
            "worst_case_stage": "training_backward_with_AdamW",
            "selected_logits_backward_multiplier": 2,
            "analytic_conservative_envelope_bytes": analytic_envelope_bytes,
            "measured_v1_peak_bytes": QUALIFICATION_V1_MEASURED_PEAK_BYTES,
            "measured_floor_multiplier": QUALIFICATION_MEASURED_FLOOR_MULTIPLIER,
            "measured_floor_bytes": measured_floor_bytes,
            "measured_floor_source": QUALIFICATION_V1_FAILURE_RECEIPT,
            "measured_floor_source_status": "FAILED_PRESERVED",
            "measured_floor_source_observation": (
                "V1 completed two optimizer steps and validation before stopping "
                "because measured device peak exceeded its analytic forecast"
            ),
            "training_backward": [
                "8x192 float32 activation batch",
                "affine and Q/K/V/output parameters",
                "AdamW first and second moments",
                "gradient buffers",
                "causal attention score workspace",
                "512xvocabulary selected-logit buffer",
                "saved hidden/autograd workspace envelope",
            ],
            "validation_forward": [
                "same 8-record chunk geometry",
                "512xvocabulary selected-logit buffer",
                "hidden/projection workspace",
            ],
            "combination": (
                "conservative_envelope=max(analytic 50-percent envelope, "
                "1.5x preserved V1 measured peak); live allocator and host "
                "guards remain independently enforced"
            ),
            "uncertainty": (
                "The analytic terms remain an auditable geometry estimate. The "
                "empirical floor is calibrated from one largest-cell V1 run and "
                "is not a capacity guarantee; every qualification and fit still "
                "records measured peaks and stops on guard anomalies."
            ),
        },
        "arm_execution": "sequential; one contextual or affine arm resident at a time",
        "qualification_requirement": "largest registered 8x192 cell passes live guard before 3000-step matrix",
    }


def write_preflight(args: argparse.Namespace, *, output_root: Path) -> dict[str, Any]:
    original_path = args.original_manifest.expanduser().resolve()
    enriched_path = args.enriched_manifest.expanduser().resolve()
    validation_path = (
        args.validation_manifest.expanduser().resolve()
        if args.validation_manifest is not None
        else original_path
    )
    original = _manifest_json(original_path)
    enriched = _manifest_json(enriched_path)
    validation = _manifest_json(validation_path)
    hidden = _shape_from_manifest(original, "fit_observations", (1200, DEFAULT_SEQUENCE_LENGTH, DEFAULT_HIDDEN_SIZE))[-1]
    sequence = _shape_from_manifest(original, "fit_observations", (1200, DEFAULT_SEQUENCE_LENGTH, hidden))[-2]
    vocab_shape = _shape_from_manifest(original, "embedding_table", (VOCAB_SIZE, hidden))
    vocabulary = vocab_shape[0]
    preflight = resource_preflight(
        hidden_size=int(hidden),
        vocabulary_size=int(vocabulary),
        sequence_length=int(sequence),
        record_batch_size=args.record_batch_size,
        position_budget=args.position_budget,
        context_width=args.context_width,
    )
    payload = {
        "schema": "token-reconstruction.trr0005-joint-fit-preflight.v1",
        "task_id": TASK_ID,
        "status": "SOURCE_ONLY_PREFLIGHT; NO_TENSORS_LOADED",
        "created_utc": _utc_now(),
        "manifests": {
            "original": _file_record(original_path, hash_bytes=False),
            "enriched": _file_record(enriched_path, hash_bytes=False),
            "validation": _file_record(validation_path, hash_bytes=False),
        },
        "manifest_declared_geometry": {
            "original_fit": list(_shape_from_manifest(original, "fit_observations", (1200, sequence, hidden))),
            "enriched_fit": list(_shape_from_manifest(enriched, "fit_observations", (1200, sequence, hidden))),
            "validation": list(_shape_from_manifest(validation, "validation_observations", (48, sequence, hidden))),
        },
        "memory_preflight": preflight,
        "resource_policy": _resource_policy(args),
        "required_live_checks": [
            "resource guard before loading public tensors",
            "largest 8-record x 192-position contextual cell first",
            "peak reserved and host RSS receipt retained",
            "stop and preserve failure on allocator, driver, thermal, or memory anomaly",
        ],
    }
    if output_root.exists() or output_root.is_symlink():
        raise JointFitRunnerError(f"preflight output is create-only: {output_root}")
    output_root.mkdir(parents=True)
    _json_write(output_root / "preflight.json", payload)
    return payload


def _sampler_receipt(name: str, data: PublicJointData, schedule: Any) -> dict[str, Any]:
    """Record semantic mask and every deterministic draw component for comparison."""

    normalized_mask = data.fit_valid_mask.to(device="cpu", dtype=torch.bool).contiguous()
    return {
        "distribution": name,
        "contract_distribution_id": DISTRIBUTION_CONTRACT_IDS[name],
        "fit_record_count": len(data.fit_record_ids),
        "fit_geometry": list(data.fit_observations.shape),
        "fit_valid_mask_sha256": tensor_sha256(normalized_mask),
        "batch_record_indices_sha256": tensor_sha256(schedule.batch_record_indices),
        "draw_record_slots_sha256": tensor_sha256(schedule.draw_record_slots),
        "draw_position_slots_sha256": tensor_sha256(schedule.draw_position_slots),
        "eligible_counts_sha256": tensor_sha256(schedule.eligible_counts),
        "used_replacement_sha256": tensor_sha256(schedule.used_replacement),
        "schedule_sha256": schedule_metadata(schedule)["schedule_sha256"],
        "schedule": schedule_metadata(schedule),
    }


def _compare_sampler_receipts(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail closed unless original/enriched masks and draws are bit-identical."""

    if tuple(row.get("distribution") for row in receipts) != ("original", "enriched"):
        raise JointFitRunnerError("sampler receipt order must be original then enriched")
    baseline = receipts[0]
    mismatches: dict[str, dict[str, str]] = {}
    for row in receipts[1:]:
        for field in SCHEDULE_IDENTITY_FIELDS:
            if row.get(field) != baseline.get(field):
                mismatches[field] = {
                    "original": str(baseline.get(field)),
                    "enriched": str(row.get(field)),
                }
    if mismatches:
        raise JointFitRunnerError(
            "cross-distribution sampler mask/vector/draw mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "status": "IDENTICAL_MASK_AND_SCHEDULE",
        "identity_fields": list(SCHEDULE_IDENTITY_FIELDS),
        "receipts": [dict(row) for row in receipts],
    }


def _validate_public_geometry(data: PublicJointData, *, name: str, args: argparse.Namespace) -> None:
    if data.hidden_size != DEFAULT_HIDDEN_SIZE:
        raise JointFitRunnerError(
            f"{name} hidden size {data.hidden_size} != registered {DEFAULT_HIDDEN_SIZE}"
        )
    if data.vocabulary_size != VOCAB_SIZE:
        raise JointFitRunnerError(
            f"{name} vocabulary size {data.vocabulary_size} != registered {VOCAB_SIZE}"
        )
    if data.fit_observations.shape[1] != args.sequence_length:
        raise JointFitRunnerError(
            f"{name} fit sequence length {data.fit_observations.shape[1]} != {args.sequence_length}"
        )
    if data.fit_observations.shape[0] < args.record_batch_size:
        raise JointFitRunnerError(
            f"{name} fit has {data.fit_observations.shape[0]} records; registered cell needs 8-record batches"
        )
    if data.validation_observations.shape[2] != data.fit_observations.shape[2]:
        raise JointFitRunnerError(f"{name} validation hidden size differs from fit")


def _validate_args(args: argparse.Namespace) -> None:
    if args.record_batch_size != DEFAULT_RECORD_BATCH_SIZE:
        raise JointFitRunnerError("TRR-0005 requires record-batch-size 8")
    if args.position_budget != DEFAULT_POSITION_BUDGET:
        raise JointFitRunnerError("TRR-0005 requires position-budget 512")
    if args.sequence_length != DEFAULT_SEQUENCE_LENGTH:
        raise JointFitRunnerError("TRR-0005 requires sequence-length 192")
    if args.steps != DEFAULT_STEPS:
        raise JointFitRunnerError("TRR-0005 requires 3000 training steps")
    if args.validation_every != DEFAULT_VALIDATION_EVERY:
        raise JointFitRunnerError("TRR-0005 requires validation checkpoints every 100 steps")
    if args.seed != DEFAULT_SEED:
        raise JointFitRunnerError("TRR-0005 requires seed 4005")
    if not math.isclose(args.learning_rate, DEFAULT_LEARNING_RATE):
        raise JointFitRunnerError("TRR-0005 requires learning rate 1e-3")
    if not math.isclose(args.weight_decay, DEFAULT_WEIGHT_DECAY):
        raise JointFitRunnerError("TRR-0005 requires weight decay 0")
    if not math.isclose(args.gradient_clip_norm, DEFAULT_GRADIENT_CLIP_NORM):
        raise JointFitRunnerError("TRR-0005 requires gradient clipping 1")
    if args.context_width != DEFAULT_CONTEXT_WIDTH:
        raise JointFitRunnerError("TRR-0005 requires attention width 128")
    preflight_only = bool(getattr(args, "preflight_only", False))
    qualification_only = bool(getattr(args, "qualification_only", False))
    if preflight_only and qualification_only:
        raise JointFitRunnerError("preflight and qualification modes are mutually exclusive")
    qualification_steps = int(
        getattr(args, "qualification_steps", DEFAULT_QUALIFICATION_STEPS)
    )
    if qualification_steps <= 0 or qualification_steps > MAX_QUALIFICATION_STEPS:
        raise JointFitRunnerError(
            f"qualification-steps must be in [1, {MAX_QUALIFICATION_STEPS}]"
        )
    if not preflight_only and not qualification_only and args.retained_affine_state is None:
        raise JointFitRunnerError("actual TRR-0005 fits require --retained-affine-state for the pretraining diagnostic")
    for name in (
        "minimum_free_gib",
        "maximum_gpu_reserved_gib",
        "maximum_host_rss_gib",
        "minimum_host_available_gib",
        "max_seconds",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise JointFitRunnerError(f"{name} must be finite and positive")


def _gradient_summary(model: torch.nn.Module) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, parameter in model.named_parameters():
        result[name] = None if parameter.grad is None else float(parameter.grad.detach().norm().cpu())
    return result


def _initial_diagnostic(
    data: PublicJointData,
    *,
    device: torch.device,
    retained_state: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    identity = build_decoder(
        AFFINE_METHOD,
        hidden_size=data.hidden_size,
        vocabulary_size=data.vocabulary_size,
        context_width=args.context_width,
        seed=args.seed,
    )
    embedding = data.embedding_table.to(device=device)
    frequency_reference = data.fit_truth[:, 1:][data.fit_valid_mask[:, 1:]]
    identity_metrics = evaluate_dataset(
        identity.to(device),
        data.fit_observations,
        data.fit_truth,
        data.fit_valid_mask,
        embedding,
        tuple("fit_public" for _ in data.fit_record_ids),
        device=device,
        record_batch_size=args.record_batch_size,
        position_budget=args.position_budget,
        frequency_reference=frequency_reference,
    )
    result: dict[str, Any] = {
        "identity_initialization": {
            "method_id": AFFINE_METHOD,
            "W_initial": "identity",
            "b_initial": "zero",
            "s_initial": 3.0,
            "fit_metrics": identity_metrics,
        },
        "retained_trr0004_affine": None,
        "training_split_record_count": len(data.fit_record_ids),
        "training_split_token_rows_post_bos": int(data.fit_valid_mask[:, 1:].sum().item()),
    }
    if retained_state is not None:
        retained = load_decoder_state(
            retained_state,
            method_id=AFFINE_METHOD,
            hidden_size=data.hidden_size,
            vocabulary_size=data.vocabulary_size,
            context_width=args.context_width,
        ).to(device)
        retained_metrics = evaluate_dataset(
            retained,
            data.fit_observations,
            data.fit_truth,
            data.fit_valid_mask,
            embedding,
            tuple("fit_public" for _ in data.fit_record_ids),
            device=device,
            record_batch_size=args.record_batch_size,
            position_budget=args.position_budget,
            frequency_reference=frequency_reference,
        )
        result["retained_trr0004_affine"] = {
            "state": _file_record(retained_state),
            "fit_metrics": retained_metrics,
            "interpretation": "public diagnostic on this task's fitting rows; no target/evaluation truth",
        }
        del retained
    del identity
    return result


def _validation_tuple(data: PublicJointData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    return (
        data.validation_observations,
        data.validation_truth,
        data.validation_valid_mask,
        data.validation_groups,
    )


def _train_arm(
    method_id: str,
    data: PublicJointData,
    schedule: Any,
    *,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    deadline: float,
    guard: list[dict[str, Any]],
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_decoder(
        method_id,
        hidden_size=data.hidden_size,
        vocabulary_size=data.vocabulary_size,
        context_width=args.context_width,
        seed=args.seed,
    ).to(device)
    trainable = list(model.parameters())
    if not trainable or any(not parameter.requires_grad for parameter in trainable):
        raise JointFitRunnerError(f"{method_id} has no fully trainable parameter set")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    checkpoints = set(checkpoint_steps(args.steps))
    validation = _validation_tuple(data)
    embedding = data.embedding_table.to(device=device)
    curve: list[dict[str, Any]] = []
    best_metric = -float("inf")
    best_step = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    started = time.perf_counter()
    optimization_seconds = 0.0
    selection_validation_seconds = 0.0
    final_fit_diagnostic_seconds = 0.0
    state_io_seconds = 0.0
    cleanup_seconds = 0.0
    last_train_point: dict[str, Any] | None = None
    for step_index in range(args.steps + 1):
        guard.append(_resource_guard(args, device, stage=f"{method_id}:before_step_{step_index}", deadline=deadline))
        if step_index in checkpoints:
            val_started = time.perf_counter()
            val_metrics = evaluate_dataset(
                model,
                validation[0],
                validation[1],
                validation[2],
                embedding,
                validation[3],
                device=device,
                record_batch_size=args.record_batch_size,
                position_budget=args.position_budget,
                frequency_reference=data.fit_truth[:, 1:][data.fit_valid_mask[:, 1:]],
            )
            validation_elapsed = time.perf_counter() - val_started
            selection_validation_seconds += validation_elapsed
            point = {
                "step": int(step_index),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_batch": None if last_train_point is None else dict(last_train_point),
                "validation": val_metrics,
                "validation_wall_seconds": validation_elapsed,
            }
            curve.append(point)
            metric = float(val_metrics["style_balanced_token_accuracy"])
            # Strict > retains the earliest maximum, including step zero.
            if metric > best_metric:
                best_metric = metric
                best_step = int(step_index)
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
        if step_index == args.steps:
            break
        update_started = time.perf_counter()
        last_train_point = train_step(
            model,
            data.fit_observations,
            data.fit_truth,
            data.fit_valid_mask,
            embedding,
            schedule,
            step_index,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        scheduler.step()
        optimization_seconds += time.perf_counter() - update_started
        last_train_point["step"] = step_index + 1
    guard.append(_resource_guard(args, device, stage=f"{method_id}:before_final_fit_evaluation", deadline=deadline))
    final_fit_started = time.perf_counter()
    final_fit = evaluate_dataset(
        model,
        data.fit_observations,
        data.fit_truth,
        data.fit_valid_mask,
        embedding,
        tuple("fit_public" for _ in data.fit_record_ids),
        device=device,
        record_batch_size=args.record_batch_size,
        position_budget=args.position_budget,
        frequency_reference=data.fit_truth[:, 1:][data.fit_valid_mask[:, 1:]],
    )
    final_fit_diagnostic_seconds = time.perf_counter() - final_fit_started
    final_fit["selection_independent"] = True
    final_fit["evaluation_wall_seconds"] = final_fit_diagnostic_seconds
    guard.append(_resource_guard(args, device, stage=f"{method_id}:after_final_fit_evaluation", deadline=deadline))
    state_io_started = time.perf_counter()
    curve_path = output_dir / "learning_curve.json"
    _json_write(
        curve_path,
        {
            "schema": "token-reconstruction.trr0005-learning-curve.v1",
            "task_id": TASK_ID,
            "method_id": method_id,
            "distribution": output_dir.parent.name,
            "canonical_method_id": f"{output_dir.parent.name}__{method_id}",
            "selection_metric": "validation_style_balanced_token_accuracy",
            "selection_rule": "earliest maximum, including step 0",
            "curve": curve,
        },
    )
    selected_model = build_decoder(
        method_id,
        hidden_size=data.hidden_size,
        vocabulary_size=data.vocabulary_size,
        context_width=args.context_width,
        seed=args.seed,
    )
    selected_model.load_state_dict(best_state, strict=True)
    state_record = save_decoder_state(
        output_dir / "selected.safetensors",
        selected_model,
        selected_step=best_step,
        metadata={"distribution": output_dir.parent.name, "canonical_method_id": f"{output_dir.parent.name}__{method_id}"},
    )
    state_io_seconds = time.perf_counter() - state_io_started
    gradient_disclosure = {
        "canonical_method_id": f"{output_dir.parent.name}__{method_id}",
        "method_id": method_id,
        "diagonal_context_off": method_id == DIAGONAL_ATTENTION_METHOD,
        "qk_parameters_present": method_id != AFFINE_METHOD,
        "qk_gradients_theoretically_zero_for_diagonal": method_id == DIAGONAL_ATTENTION_METHOD,
        "effective_capacity_claim": "not equal; diagonal Q/K are inactive by construction"
        if method_id == DIAGONAL_ATTENTION_METHOD
        else "full trainable parameter set",
        "last_train_batch_gradient_norms": None if last_train_point is None else last_train_point.get("gradient_norms"),
    }
    peak = {
        **_host_memory_snapshot(),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
    }
    cleanup_started = time.perf_counter()
    del selected_model
    del model, optimizer, scheduler, embedding
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cleanup_seconds = time.perf_counter() - cleanup_started
    arm_wall_seconds = time.perf_counter() - started
    return {
        "method_id": method_id,
        "distribution": output_dir.parent.name,
        "canonical_method_id": f"{output_dir.parent.name}__{method_id}",
        "seed": args.seed,
        "steps": args.steps,
        "checkpoint_steps": sorted(checkpoints),
        "selected_step": best_step,
        "best_validation_style_balanced_token_accuracy": best_metric,
        "curve": {
            "path": str(curve_path),
            "bytes": int(curve_path.stat().st_size),
            "sha256": file_sha256(curve_path),
            "points": len(curve),
        },
        "state": state_record,
        "parameter_count": sum(int(value.numel()) for value in best_state.values()),
        "arm_wall_seconds": arm_wall_seconds,
        "optimization_update_seconds": optimization_seconds,
        "selection_validation_seconds": selection_validation_seconds,
        "final_fit_diagnostic_seconds": final_fit_diagnostic_seconds,
        "state_io_seconds": state_io_seconds,
        "cleanup_seconds": cleanup_seconds,
        "timing_accounting": {
            "optimization_update_seconds": optimization_seconds,
            "selection_validation_seconds": selection_validation_seconds,
            "final_fit_diagnostic_seconds": final_fit_diagnostic_seconds,
            "state_io_seconds": state_io_seconds,
            "cleanup_seconds": cleanup_seconds,
            "arm_wall_seconds": arm_wall_seconds,
            "components_are_subinterval_or_accumulated": True,
        },
        "final_fit_evaluation": final_fit,
        "gradient_disclosure": gradient_disclosure,
        "peak_memory": peak,
        "runtime_components": {
            "public_prefix_calls": 0,
            "candidate_simulations": 0,
            "a2_fallback": False,
            "teacher_prefix": False,
            "supervision": "same-position current-token CE only",
        },
    }


def _run_distribution(
    name: str,
    manifest: Path,
    *,
    args: argparse.Namespace,
    validation_manifest: Path | None,
    embedding_path: Path | None,
    retained_state: Path | None,
    output_root: Path,
    device: torch.device,
    deadline: float,
    guard: list[dict[str, Any]],
    sampler_reference: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    guard.append(_resource_guard(args, device, stage=f"{name}:before_public_tensor_load", deadline=deadline))
    load_started = time.perf_counter()
    data = load_public_joint_data(
        manifest,
        validation_manifest,
        embedding_path=embedding_path,
    )
    load_seconds = time.perf_counter() - load_started
    _validate_public_geometry(data, name=name, args=args)
    batch_note = "record batches use without-replacement draws when at least eight records exist"
    guard.append(_resource_guard(args, device, stage=f"{name}:after_public_tensor_load", deadline=deadline))
    schedule_started = time.perf_counter()
    schedule = build_position_schedule(
        data.fit_valid_mask,
        steps=args.steps,
        record_batch_size=args.record_batch_size,
        position_budget=args.position_budget,
        seed=args.seed,
    )
    schedule_construction_seconds = time.perf_counter() - schedule_started
    sampler_hash_started = time.perf_counter()
    sampler_receipt = _sampler_receipt(name, data, schedule)
    sampler_receipt_hash_seconds = time.perf_counter() - sampler_hash_started
    if sampler_reference is not None:
        _compare_sampler_receipts([sampler_reference, sampler_receipt])
    distribution_root = output_root / name
    distribution_root.mkdir(parents=True, exist_ok=False)
    schedule_io_started = time.perf_counter()
    schedule_record = save_schedule(distribution_root / "position_schedule.safetensors", schedule)
    schedule_io_seconds = time.perf_counter() - schedule_io_started
    sampler_io_started = time.perf_counter()
    _json_write(distribution_root / "sampler_receipt.json", sampler_receipt)
    sampler_receipt_io_seconds = time.perf_counter() - sampler_io_started
    diagnostic_started = time.perf_counter()
    diagnostic = _initial_diagnostic(
        data,
        device=device,
        retained_state=retained_state,
        args=args,
    )
    _json_write(distribution_root / "pretraining_diagnostic.json", diagnostic)
    diagnostic_seconds = time.perf_counter() - diagnostic_started
    method_results: dict[str, Any] = {}
    for method_id in METHODS:
        method_root = distribution_root / method_id
        method_root.mkdir(parents=True, exist_ok=False)
        method_results[method_id] = _train_arm(
            method_id,
            data,
            schedule,
            args=args,
            device=device,
            output_dir=method_root,
            deadline=deadline,
            guard=guard,
        )
    result = {
        "distribution": name,
        "manifest": str(manifest.expanduser().resolve()),
        "validation_manifest": str((validation_manifest or manifest).expanduser().resolve()),
        "contract_distribution_id": DISTRIBUTION_CONTRACT_IDS[name],
        "load_seconds": load_seconds,
        "preparation_timing": {
            "public_tensor_load_seconds": load_seconds,
            "schedule_construction_seconds": schedule_construction_seconds,
            "sampler_receipt_hash_seconds": sampler_receipt_hash_seconds,
            "schedule_io_seconds": schedule_io_seconds,
            "sampler_receipt_io_seconds": sampler_receipt_io_seconds,
            "pretraining_diagnostic_seconds": diagnostic_seconds,
        },
        "fit_record_count": len(data.fit_record_ids),
        "validation_record_count": len(data.validation_record_ids),
        "fit_geometry": list(data.fit_observations.shape),
        "validation_geometry": list(data.validation_observations.shape),
        "fit_post_bos_positions": int(data.fit_valid_mask[:, 1:].sum().item()),
        "validation_post_bos_positions": int(data.validation_valid_mask[:, 1:].sum().item()),
        "record_batch_note": batch_note,
        "sampler_receipt": sampler_receipt,
        "schedule": schedule_record,
        "methods": method_results,
        "pretraining_diagnostic": {
            "path": str(distribution_root / "pretraining_diagnostic.json"),
            "sha256": file_sha256(distribution_root / "pretraining_diagnostic.json"),
        },
        "data_metadata": data.metadata,
    }
    del data, schedule
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, diagnostic



def _run_qualification_distribution(
    name: str,
    data: PublicJointData,
    schedule: Any,
    *,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
    preflight: Mapping[str, Any],
    sampler_receipt: Mapping[str, Any],
    deadline: float,
    guard: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the bounded real causal largest-cell qualification for one distribution."""

    _validate_public_geometry(data, name=name, args=args)
    distribution_root = output_root / name
    distribution_root.mkdir(parents=True, exist_ok=False)
    schedule_io_started = time.perf_counter()
    schedule_record = save_schedule(distribution_root / "position_schedule.safetensors", schedule)
    schedule_io_seconds = time.perf_counter() - schedule_io_started
    sampler_io_started = time.perf_counter()
    _json_write(distribution_root / "sampler_receipt.json", dict(sampler_receipt))
    sampler_io_seconds = time.perf_counter() - sampler_io_started
    method_id = CAUSAL_ATTENTION_METHOD
    qualification_steps = int(args.qualification_steps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    setup_started = time.perf_counter()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = build_decoder(
        method_id,
        hidden_size=data.hidden_size,
        vocabulary_size=data.vocabulary_size,
        context_width=args.context_width,
        seed=args.seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    model_optimizer_setup_seconds = time.perf_counter() - setup_started
    embedding_transfer_started = time.perf_counter()
    embedding = data.embedding_table.to(device=device)
    embedding_transfer_seconds = time.perf_counter() - embedding_transfer_started
    setup_seconds = time.perf_counter() - setup_started
    frequency_reference = data.fit_truth[:, 1:][data.fit_valid_mask[:, 1:]]
    validation_points: list[dict[str, Any]] = []
    train_points: list[dict[str, Any]] = []
    qualification_validation_seconds = 0.0
    optimization_update_seconds = 0.0

    def evaluate_point(step: int) -> None:
        nonlocal qualification_validation_seconds
        guard.append(
            _resource_guard(
                args,
                device,
                stage=f"{name}:qualification_before_validation_step_{step}",
                deadline=deadline,
            )
        )
        validation_started = time.perf_counter()
        metrics = evaluate_dataset(
            model,
            data.validation_observations,
            data.validation_truth,
            data.validation_valid_mask,
            embedding,
            data.validation_groups,
            device=device,
            record_batch_size=args.record_batch_size,
            position_budget=args.position_budget,
            frequency_reference=frequency_reference,
        )
        elapsed = time.perf_counter() - validation_started
        qualification_validation_seconds += elapsed
        validation_points.append(
            {
                "step": int(step),
                "metrics": metrics,
                "validation_wall_seconds": elapsed,
            }
        )
        guard.append(
            _resource_guard(
                args,
                device,
                stage=f"{name}:qualification_after_validation_step_{step}",
                deadline=deadline,
            )
        )

    try:
        evaluate_point(0)
        for step_index in range(qualification_steps):
            guard.append(
                _resource_guard(
                    args,
                    device,
                    stage=f"{name}:qualification_before_train_step_{step_index}",
                    deadline=deadline,
                )
            )
            update_started = time.perf_counter()
            train_point = train_step(
                model,
                data.fit_observations,
                data.fit_truth,
                data.fit_valid_mask,
                embedding,
                schedule,
                step_index,
                device=device,
                optimizer=optimizer,
                gradient_clip_norm=args.gradient_clip_norm,
            )
            scheduler.step()
            update_elapsed = time.perf_counter() - update_started
            optimization_update_seconds += update_elapsed
            train_point.update(
                {
                    "step": int(step_index + 1),
                    "wall_seconds": update_elapsed,
                    "learning_rate_after_step": float(optimizer.param_groups[0]["lr"]),
                }
            )
            train_points.append(train_point)
            guard.append(
                _resource_guard(
                    args,
                    device,
                    stage=f"{name}:qualification_after_train_step_{step_index}",
                    deadline=deadline,
                )
            )
        evaluate_point(qualification_steps)
        guard.append(
            _resource_guard(
                args,
                device,
                stage=f"{name}:qualification_before_peak_snapshot",
                deadline=deadline,
            )
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak = {
            **_host_memory_snapshot(),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
        }
        forecast_bytes = int(preflight["bytes"]["conservative_envelope"])
        measured_device_peak = (
            None
            if peak["cuda_peak_reserved_bytes"] is None
            else max(
                int(peak["cuda_peak_reserved_bytes"]),
                int(peak["cuda_peak_allocated_bytes"]),
            )
        )
        forecast_comparison = {
            "forecast_bytes": forecast_bytes,
            "forecast_gib": forecast_bytes / (1024**3),
            "measured_device_peak_bytes": measured_device_peak,
            "measured_device_peak_gib": (
                None
                if measured_device_peak is None
                else measured_device_peak / (1024**3)
            ),
            "forecast_covers_measured_device_peak": (
                None if measured_device_peak is None else measured_device_peak <= forecast_bytes
            ),
            "forecast_scope": "device tensors; host public-resource RSS is reported separately",
        }
        forecast_passed = (
            measured_device_peak is None or measured_device_peak <= forecast_bytes
        )
        wall_before_forecast_seconds = time.perf_counter() - started
        result = {
            "distribution": name,
            "contract_distribution_id": DISTRIBUTION_CONTRACT_IDS[name],
            "method_id": method_id,
            "canonical_method_id": f"{name}__{method_id}",
            "status": (
                "QUALIFIED_CAUSAL_LARGEST_CELL"
                if forecast_passed
                else "FAILED_FORECAST_COMPARISON"
            ),
            "schedule_steps": int(schedule.steps),
            "qualification_steps": qualification_steps,
            "draws_per_step": args.position_budget,
            "qualification_train_draws": qualification_steps * args.position_budget,
            "geometry": {
                "record_batch_size": args.record_batch_size,
                "sequence_length": args.sequence_length,
                "hidden_size": data.hidden_size,
                "position_budget": args.position_budget,
            },
            "setup_seconds": setup_seconds,
            "embedding_transfer_seconds": embedding_transfer_seconds,
            "optimization_update_seconds": optimization_update_seconds,
            "qualification_validation_seconds": qualification_validation_seconds,
            "selection_validation_seconds": None,
            "final_fit_diagnostic_seconds": None,
            "cleanup_seconds": None,
            "total_qualification_wall_seconds": wall_before_forecast_seconds,
            "qualification_progress_path": str(
                distribution_root / "qualification_progress.json"
            ),
            "timing_accounting": {
                "public_tensor_load_seconds": None,
                "schedule_construction_seconds": None,
                "schedule_io_seconds": schedule_io_seconds,
                "sampler_receipt_io_seconds": sampler_io_seconds,
                "model_optimizer_setup_seconds": model_optimizer_setup_seconds,
                "embedding_transfer_seconds": embedding_transfer_seconds,
                "optimization_update_seconds": optimization_update_seconds,
                "qualification_validation_seconds": qualification_validation_seconds,
                "selection_validation_seconds": None,
                "final_fit_diagnostic_seconds": None,
                "cleanup_seconds": None,
                "total_qualification_wall_seconds": wall_before_forecast_seconds,
                "state_io_seconds": None,
                "components_are_subinterval_or_accumulated": True,
            },
            "schedule": schedule_record,
            "sampler_receipt": dict(sampler_receipt),
            "validation_points": validation_points,
            "train_points": train_points,
            "peak_memory": peak,
            "conservative_forecast": preflight,
            "forecast_comparison": forecast_comparison,
            "runtime_components": {
                "public_prefix_calls": 0,
                "candidate_simulations": 0,
                "a2_fallback": False,
                "future_activation_reads": False,
                "source_token_inputs": False,
                "supervision": "same-position current-token CE only",
            },
        }
        # Persist metrics, per-step timings, peak memory, and the forecast
        # comparison before enforcing the forecast check.  If the check fails,
        # the caller's failure.json and this partial receipt together preserve
        # the completed qualification work.
        _json_write(distribution_root / "qualification_progress.json", result)
        if not forecast_passed:
            raise JointFitRunnerError(
                f"{name} qualification measured device peak exceeds conservative forecast: "
                f"{measured_device_peak} > {forecast_bytes}"
            )
        guard.append(
            _resource_guard(
                args,
                device,
                stage=f"{name}:qualification_after_peak_snapshot",
                deadline=deadline,
            )
        )
        cleanup_started = time.perf_counter()
        del model, optimizer, scheduler, embedding
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        cleanup_seconds = time.perf_counter() - cleanup_started
        wall_seconds = time.perf_counter() - started
        result["cleanup_seconds"] = cleanup_seconds
        result["total_qualification_wall_seconds"] = wall_seconds
        result["timing_accounting"]["cleanup_seconds"] = cleanup_seconds
        result["timing_accounting"]["total_qualification_wall_seconds"] = wall_seconds
        return result
    except Exception:
        # Preserve the caller's failure receipt and resource guard history.
        try:
            del model, optimizer, scheduler, embedding
        except UnboundLocalError:
            pass
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        raise


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    """Qualify both public fit distributions with a bounded real causal cell."""

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise JointFitRunnerError(f"qualification output is create-only: {output_root}")
    output_root.mkdir(parents=True)
    root = args.repository_root.expanduser().resolve()
    device = _choose_device(args.device)
    started = _utc_now()
    started_clock = time.perf_counter()
    deadline = started_clock + args.max_seconds
    guard: list[dict[str, Any]] = []
    try:
        original_manifest = args.original_manifest.expanduser().resolve()
        enriched_manifest = args.enriched_manifest.expanduser().resolve()
        validation_manifest = (
            args.validation_manifest.expanduser().resolve()
            if args.validation_manifest is not None
            else None
        )
        embedding_path = (
            args.embedding_table.expanduser().resolve()
            if args.embedding_table is not None
            else None
        )
        preflight = resource_preflight(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            vocabulary_size=VOCAB_SIZE,
            sequence_length=args.sequence_length,
            record_batch_size=args.record_batch_size,
            position_budget=args.position_budget,
            context_width=args.context_width,
        )
        preflight["mode"] = "qualification_only"
        preflight["qualification_steps"] = int(args.qualification_steps)
        preflight["resource_policy"] = _resource_policy(args)
        _json_write(output_root / "memory_preflight.json", preflight)
        guard.append(_resource_guard(args, device, stage="qualification_before_any_public_tensor_load", deadline=deadline))
        sampler_receipts: list[Mapping[str, Any]] = []
        distributions: dict[str, Any] = {}
        preparation_timing: dict[str, Any] = {}
        for name, manifest in (("original", original_manifest), ("enriched", enriched_manifest)):
            guard.append(
                _resource_guard(
                    args,
                    device,
                    stage=f"{name}:qualification_before_public_tensor_load",
                    deadline=deadline,
                )
            )
            load_started = time.perf_counter()
            data = load_public_joint_data(
                manifest,
                validation_manifest,
                embedding_path=embedding_path,
            )
            load_seconds = time.perf_counter() - load_started
            _validate_public_geometry(data, name=name, args=args)
            schedule_started = time.perf_counter()
            schedule = build_position_schedule(
                data.fit_valid_mask,
                steps=args.steps,
                record_batch_size=args.record_batch_size,
                position_budget=args.position_budget,
                seed=args.seed,
            )
            schedule_construction_seconds = time.perf_counter() - schedule_started
            sampler_hash_started = time.perf_counter()
            sampler_receipt = _sampler_receipt(name, data, schedule)
            sampler_hash_seconds = time.perf_counter() - sampler_hash_started
            if sampler_receipts:
                _compare_sampler_receipts([sampler_receipts[0], sampler_receipt])
            sampler_receipts.append(sampler_receipt)
            guard.append(
                _resource_guard(
                    args,
                    device,
                    stage=f"{name}:qualification_after_public_tensor_load",
                    deadline=deadline,
                )
            )
            result = _run_qualification_distribution(
                name,
                data,
                schedule,
                args=args,
                device=device,
                output_root=output_root,
                preflight=preflight,
                sampler_receipt=sampler_receipt,
                deadline=deadline,
                guard=guard,
            )
            result["sampler_receipt"] = sampler_receipt
            result["preparation_timing"] = {
                "public_tensor_load_seconds": load_seconds,
                "schedule_construction_seconds": schedule_construction_seconds,
                "sampler_receipt_hash_seconds": sampler_hash_seconds,
                "schedule_io_seconds": result["timing_accounting"]["schedule_io_seconds"],
                "sampler_receipt_io_seconds": result["timing_accounting"]["sampler_receipt_io_seconds"],
            }
            evidence_io_started = time.perf_counter()
            _json_write(output_root / name / "qualification.json", result)
            result["timing_accounting"]["state_io_seconds"] = time.perf_counter() - evidence_io_started
            distributions[name] = result
            preparation_timing[name] = result["preparation_timing"]
            del data, schedule
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            guard.append(
                _resource_guard(
                    args,
                    device,
                    stage=f"{name}:qualification_after_public_tensor_release",
                    deadline=deadline,
                )
            )
        sampler_cross_distribution = _compare_sampler_receipts(sampler_receipts)
        evidence = {
            "schema": "token-reconstruction.trr0005-qualification.v1",
            "task_id": TASK_ID,
            "status": "QUALIFICATION_COMPLETE_NO_FINAL_EVALUATION",
            "started_utc": started,
            "ended_utc": _utc_now(),
            "total_qualification_wall_seconds": time.perf_counter() - started_clock,
            "repository_root": str(root),
            "git_commit": _git_commit(root),
            "git_status_start": _git_status(root),
            "environment": {
                "python": sys.executable,
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": str(device),
                "env": _safe_environment(),
            },
            "mode": {
                "method_id": CAUSAL_ATTENTION_METHOD,
                "canonical_method_ids": [
                    f"original__{CAUSAL_ATTENTION_METHOD}",
                    f"enriched__{CAUSAL_ATTENTION_METHOD}",
                ],
                "qualification_steps": int(args.qualification_steps),
                "schedule_steps": args.steps,
                "same_registered_schedule_prefix": True,
                "no_final_holdout_loaded": True,
            },
            "fixed_settings": {
                "seed": args.seed,
                "qkv_init_seed": args.seed,
                "record_batch_size": args.record_batch_size,
                "sequence_length": args.sequence_length,
                "position_budget": args.position_budget,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR(T_max=3000)",
                "gradient_clip_norm": args.gradient_clip_norm,
                "minimum_host_available_gib": args.minimum_host_available_gib,
                "supervision": "H_i -> x_i same-position CE; BOS excluded from loss",
                "context_contract": "causal arm sees H_0...H_i; no future activation",
                "distribution_contract_ids": dict(DISTRIBUTION_CONTRACT_IDS),
            },
            "memory_preflight": preflight,
            "sampler_cross_distribution": sampler_cross_distribution,
            "preparation_timing": preparation_timing,
            "distributions": distributions,
            "resource_guard": {"checks": len(guard), "events": guard},
            "current_evaluator_truth_accessed": False,
            "final_holdout_loaded": False,
            "runtime_components": {
                "public_prefix_calls": 0,
                "candidate_simulations": 0,
                "a2_fallback": False,
                "future_activation_reads": False,
                "source_token_inputs": False,
            },
        }
        _json_write(output_root / "qualification_evidence.json", evidence)
        return evidence
    except Exception as exc:
        failure = {
            "schema": "token-reconstruction.trr0005-qualification-failure.v1",
            "task_id": TASK_ID,
            "status": "FAILED_PRESERVED",
            "started_utc": started,
            "failed_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "git_commit": _git_commit(root),
            "resource_guard": {"checks": len(guard), "events": guard},
            "output_root": str(output_root),
        }
        try:
            _json_write(output_root / "failure.json", failure)
        except Exception:
            pass
        if isinstance(exc, JointFitRunnerError):
            raise
        raise JointFitRunnerError(str(exc)) from exc

def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_root = args.output_root.expanduser().resolve()
    if args.preflight_only:
        return write_preflight(args, output_root=output_root)
    if args.qualification_only:
        return run_qualification(args)
    if output_root.exists() or output_root.is_symlink():
        raise JointFitRunnerError(f"fit output is create-only: {output_root}")
    output_root.mkdir(parents=True)
    root = args.repository_root.expanduser().resolve()
    device = _choose_device(args.device)
    started = _utc_now()
    started_clock = time.perf_counter()
    deadline = started_clock + args.max_seconds
    guard: list[dict[str, Any]] = []
    try:
        original_manifest = args.original_manifest.expanduser().resolve()
        enriched_manifest = args.enriched_manifest.expanduser().resolve()
        validation_manifest = (
            args.validation_manifest.expanduser().resolve() if args.validation_manifest is not None else None
        )
        embedding_path = args.embedding_table.expanduser().resolve() if args.embedding_table is not None else None
        preflight = resource_preflight(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            vocabulary_size=VOCAB_SIZE,
            sequence_length=args.sequence_length,
            record_batch_size=args.record_batch_size,
            position_budget=args.position_budget,
            context_width=args.context_width,
        )
        preflight["resource_policy"] = _resource_policy(args)
        _json_write(output_root / "memory_preflight.json", preflight)
        guard.append(_resource_guard(args, device, stage="before_any_public_tensor_load"))
        retained_state = args.retained_affine_state.expanduser().resolve() if args.retained_affine_state else None
        distributions: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        sampler_receipts: list[Mapping[str, Any]] = []
        for name, manifest in (("original", original_manifest), ("enriched", enriched_manifest)):
            result, diagnostic = _run_distribution(
                name,
                manifest,
                args=args,
                validation_manifest=validation_manifest,
                embedding_path=embedding_path,
                retained_state=retained_state,
                output_root=output_root,
                device=device,
                deadline=deadline,
                guard=guard,
                sampler_reference=sampler_receipts[0] if sampler_receipts else None,
            )
            distributions[name] = result
            diagnostics[name] = diagnostic
            sampler_receipts.append(result["sampler_receipt"])
        sampler_cross_distribution = _compare_sampler_receipts(sampler_receipts)
        evidence = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "status": "JOINT_FIT_COMPLETE_NO_FINAL_EVALUATION",
            "started_utc": started,
            "ended_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "repository_root": str(root),
            "git_commit": _git_commit(root),
            "git_status_start": _git_status(root),
            "environment": {
                "python": sys.executable,
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": str(device),
                "env": _safe_environment(),
            },
            "fixed_settings": {
                "methods": list(METHODS),
                "distributions": ["original", "enriched"],
                "distribution_contract_ids": dict(DISTRIBUTION_CONTRACT_IDS),
                "seed": args.seed,
                "qkv_init_seed": args.seed,
                "identity_affine_initialization": {"W": "identity", "b": "zero", "s": 3.0},
                "output_correction_initialization": "zero weight and zero bias",
                "steps": args.steps,
                "record_batch_size": args.record_batch_size,
                "sequence_length": args.sequence_length,
                "position_budget": args.position_budget,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "gradient_clip_norm": args.gradient_clip_norm,
                "minimum_host_available_gib": args.minimum_host_available_gib,
                "selection_rule": "earliest maximum public validation style-balanced accuracy, including step 0",
                "supervision": "H_i -> x_i same-position CE; BOS excluded from loss",
                "context_contract": "causal arm sees H_0...H_i; diagonal arm sees H_i only",
            },
            "memory_preflight": preflight,
            "sampler_cross_distribution": sampler_cross_distribution,
            "pretraining_diagnostics": diagnostics,
            "distributions": distributions,
            "resource_guard": {"checks": len(guard), "events": guard},
            "current_evaluator_truth_accessed": False,
            "final_holdout_loaded": False,
            "runtime_components": {
                "public_prefix_calls": 0,
                "candidate_simulations": 0,
                "a2_fallback": False,
                "future_activation_reads": False,
                "source_token_inputs": False,
            },
        }
        _json_write(output_root / "run_evidence.json", evidence)
        return evidence
    except Exception as exc:
        failure = {
            "schema": "token-reconstruction.trr0005-joint-fit-failure.v1",
            "task_id": TASK_ID,
            "status": "FAILED_PRESERVED",
            "started_utc": started,
            "failed_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "git_commit": _git_commit(root),
            "resource_guard": {"checks": len(guard), "events": guard},
            "output_root": str(output_root),
        }
        try:
            _json_write(output_root / "failure.json", failure)
        except Exception:
            pass
        if isinstance(exc, JointFitRunnerError):
            raise
        raise JointFitRunnerError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--enriched-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--embedding-table", type=Path)
    parser.add_argument("--retained-affine-state", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--record-batch-size", type=int, default=DEFAULT_RECORD_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--position-budget", type=int, default=DEFAULT_POSITION_BUDGET)
    parser.add_argument("--validation-every", type=int, default=DEFAULT_VALIDATION_EVERY)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--context-width", type=int, default=DEFAULT_CONTEXT_WIDTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_GIB)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=DEFAULT_MAXIMUM_GPU_RESERVED_GIB)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=DEFAULT_MAXIMUM_HOST_RSS_GIB)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--minimum-host-available-gib", type=float, default=DEFAULT_MINIMUM_HOST_AVAILABLE_GIB)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--qualification-steps", type=int, default=DEFAULT_QUALIFICATION_STEPS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run(args)
    except (JointFitRunnerError, JointDecoderError) as exc:
        print(f"TRR-0005 joint fit failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
