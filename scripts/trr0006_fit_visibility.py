#!/usr/bin/env python3
"""Fit the bounded TRR-P06 visibility-mask decoder family.

This runner has three explicit modes:

``preflight``
    Read only public manifest metadata and the pinned affine initializer.  It
    writes a source-only H128 geometry and guard receipt.
``probe``
    Load only the published public fit/validation bank, crop to H128, freeze
    the competent direct W/b/s path, and run the fixed 300-update capacity
    probe on a public-fit error ledger.  Probe states are diagnostic and are
    never used to initialize or select the main fits.
``main``
    Load the same public bank, crop to H128, and run six sequential fits (three
    masks x two registered seeds) with one shared 3,000-update schedule per
    seed.  Each arm selects the earliest maximum public-validation accuracy.

No fresh panel, target condition, private truth, guessed token, candidate
search, or A2 resource is loaded by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
import traceback
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from token_reconstruction.trr0005_joint_decoder import (
    PublicJointData,
    PositionSchedule,
    build_position_schedule,
    evaluate_dataset,
    load_public_joint_data,
    save_schedule,
    schedule_metadata,
    tensor_sha256,
    train_step,
)
from token_reconstruction.trr0006_visibility_decoder import (
    DEFAULT_CONTEXT_WIDTH,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_QKV_SEED,
    DEFAULT_VOCABULARY_SIZE,
    FULL_RECORD_METHOD,
    METHODS,
    PAST_ONLY_METHOD,
    POSITIONWISE_METHOD,
    VisibilityAffineAttentionDecoder,
    VisibilityDecoderError,
    build_visibility_decoder,
    deterministic_top1,
    file_sha256,
    load_direct_affine_initialization,
    save_visibility_state,
    state_sha256,
)


TASK_ID = "TRR-P06"
SCRIPT_SCHEMA = "token-reconstruction.trr-p06-visibility-fit.v1"
PREFLIGHT_SCHEMA = "token-reconstruction.trr-p06-visibility-preflight.v1"
PROBE_SCHEMA = "token-reconstruction.trr-p06-capacity-probe.v1"
MAIN_SCHEMA = "token-reconstruction.trr-p06-main-fit.v1"
FAILURE_SCHEMA = "token-reconstruction.trr-p06-visibility-fit-failure.v1"

SOURCE_SEQUENCE_LENGTH = 192
DECLARED_SEQUENCE_LENGTH = 128
HIDDEN_SIZE = DEFAULT_HIDDEN_SIZE
VOCABULARY_SIZE = DEFAULT_VOCABULARY_SIZE
CONTEXT_WIDTH = DEFAULT_CONTEXT_WIDTH
RECORD_BATCH_SIZE = 8
POSITION_BUDGET = 512
PROBE_STEPS = 300
MAIN_STEPS = 3000
VALIDATION_EVERY = 100
PROBE_SEED = 6106
MAIN_SEEDS = (6106, 6107)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0
DIRECT_AFFINE_SHA256 = "09c5b852373d8555b06508a79bb00c94041202702b61b121b35fa2b6f9f64e65"
DIRECT_AFFINE_RELATIVE_PATH = (
    "experiments/TRR-0004/evidence/affine/selected_states/"
    "fit_large_v1.historical_affine_ce_no_vocab_bias.safetensors"
)
MINIMUM_FREE_GPU_GIB = 8.0
MAXIMUM_GPU_RESERVED_GIB = 6.0
MAXIMUM_HOST_RSS_GIB = 16.0
MINIMUM_HOST_AVAILABLE_GIB = 10.0
MAX_SECONDS = 1800.0


class VisibilityFitError(RuntimeError):
    """Raised when a P06 fit/probe contract or guard fails."""


class CapacityProbeError(VisibilityFitError):
    """Raised when the fixed public-fit capacity probe does not pass."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write_create(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise VisibilityFitError(f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_read(path: Path, *, label: str) -> Mapping[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise VisibilityFitError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisibilityFitError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise VisibilityFitError(f"{label} must be a JSON object: {path}")
    return value


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _safe_environment() -> dict[str, str]:
    names = (
        "CUDA_VISIBLE_DEVICES",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONPATH",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def _rusage_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _host_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                return int(fields[1]) * 1024
    except (OSError, UnicodeError):
        return None
    return None


def _memory_snapshot(device: torch.device) -> dict[str, Any]:
    available = _host_available_bytes()
    result: dict[str, Any] = {
        "process_max_rss_bytes": _rusage_rss_bytes(),
        "host_available_bytes": available,
        "host_available_gib": None if available is None else available / (1024**3),
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
        "cuda_reserved_bytes": None,
    }
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result.update(
            {
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
        )
    return result


def _resource_guard(
    args: argparse.Namespace,
    device: torch.device,
    *,
    stage: str,
    deadline: float | None,
) -> dict[str, Any]:
    if deadline is not None and time.perf_counter() >= deadline:
        raise VisibilityFitError(f"wall-time guard expired at {stage}")
    snapshot = _memory_snapshot(device)
    max_rss = int(float(args.maximum_host_rss_gib) * 1024**3)
    if snapshot["process_max_rss_bytes"] > max_rss:
        raise VisibilityFitError(
            f"host RSS guard exceeded at {stage}: "
            f"{snapshot['process_max_rss_bytes']} > {max_rss}"
        )
    available = snapshot["host_available_bytes"]
    min_available = int(float(args.minimum_host_available_gib) * 1024**3)
    if available is not None and available < min_available:
        raise VisibilityFitError(
            f"host available-memory guard exceeded at {stage}: {available} < {min_available}"
        )
    if device.type == "cuda":
        free_bytes = int(snapshot["cuda_free_bytes"])
        reserved = int(snapshot["cuda_reserved_bytes"])
        min_free = int(float(args.minimum_free_gib) * 1024**3)
        max_reserved = int(float(args.maximum_gpu_reserved_gib) * 1024**3)
        if free_bytes < min_free:
            raise VisibilityFitError(
                f"GPU free-memory guard exceeded at {stage}: {free_bytes} < {min_free}"
            )
        if reserved > max_reserved:
            raise VisibilityFitError(
                f"GPU reserved-memory guard exceeded at {stage}: {reserved} > {max_reserved}"
            )
    return {"stage": stage, **snapshot, "timestamp_utc": _utc_now()}


def _device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise VisibilityFitError(f"invalid device: {value}") from exc
    if device.type not in ("cpu", "cuda"):
        raise VisibilityFitError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise VisibilityFitError("CUDA requested but unavailable")
    return device


def _set_runtime_threads(args: argparse.Namespace) -> None:
    if int(args.torch_threads) <= 0 or int(args.torch_interop_threads) <= 0:
        raise VisibilityFitError("torch thread counts must be positive")
    torch.set_num_threads(int(args.torch_threads))
    torch.set_num_interop_threads(int(args.torch_interop_threads))


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise VisibilityFitError(f"expected regular file: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _direct_state(args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_path = Path(args.direct_affine_state).expanduser()
    if not raw_path.is_absolute():
        raw_path = Path(args.repository_root).expanduser().resolve() / raw_path
    path = raw_path.resolve()
    state, descriptor = load_direct_affine_initialization(
        path,
        hidden_size=HIDDEN_SIZE,
        expected_sha256=str(args.direct_affine_sha256),
    )
    if descriptor["sha256"] != DIRECT_AFFINE_SHA256:
        raise VisibilityFitError("production direct affine hash is not the registered P06 hash")
    return state, descriptor


def _manifest_payload(path: Path) -> Mapping[str, Any]:
    value = _json_read(path, label="public fit manifest")
    resources = value.get("resources")
    if not isinstance(resources, Mapping):
        raise VisibilityFitError(f"manifest has no resources object: {path}")
    return value


def _resource_shape(payload: Mapping[str, Any], name: str) -> tuple[int, ...]:
    resources = payload.get("resources")
    assert isinstance(resources, Mapping)
    resource = resources.get(name)
    if not isinstance(resource, Mapping) or not isinstance(resource.get("shape"), list):
        raise VisibilityFitError(f"manifest resource shape is missing: {name}")
    shape = resource["shape"]
    if not all(isinstance(item, int) for item in shape):
        raise VisibilityFitError(f"manifest resource shape is malformed: {name}")
    return tuple(int(item) for item in shape)


def _preflight_geometry(
    *,
    hidden_size: int,
    vocabulary_size: int,
    sequence_length: int,
    record_batch_size: int,
    position_budget: int,
    context_width: int,
) -> dict[str, Any]:
    embedding_tensor = vocabulary_size * hidden_size * 4
    activation = record_batch_size * sequence_length * hidden_size * 4
    scores = record_batch_size * sequence_length * sequence_length * 4
    selected_logits = position_budget * vocabulary_size * 4
    affine_parameters = hidden_size * hidden_size + hidden_size + 1
    attention_parameters = (
        3 * (hidden_size * context_width + context_width)
        + context_width * hidden_size
        + hidden_size
    )
    parameters = (affine_parameters + attention_parameters) * 4
    adam = parameters * 2
    gradients = parameters
    hidden_workspace = activation * 5
    training_workspace = activation + scores + hidden_workspace + selected_logits * 2
    resident = embedding_tensor + parameters + adam + gradients
    training_peak = resident + training_workspace
    safety = math.ceil(training_peak * 0.50)
    conservative = training_peak + safety
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
            "embedding_tensor_fp32": embedding_tensor,
            "activation_batch_fp32": activation,
            "attention_scores_fp32": scores,
            "selected_logits_fp32": selected_logits,
            "model_parameters_fp32": parameters,
            "adamw_m_v_fp32": adam,
            "gradient_buffer_fp32": gradients,
            "hidden_workspace_envelope": hidden_workspace,
            "training_workspace_envelope": training_workspace,
            "resident_training_envelope": resident,
            "training_peak_envelope": training_peak,
            "safety_margin_50_percent": safety,
            "conservative_envelope": conservative,
        },
        "gib": {
            "embedding_tensor_fp32": embedding_tensor / gib,
            "training_peak_envelope": training_peak / gib,
            "conservative_envelope": conservative / gib,
        },
        "asset_note": {
            "declared_embedding_file_bytes": 1050673488,
            "embedding_tensor_bytes_exclude_safetensors_header": True,
        },
    }


def _crop_public_data(data: PublicJointData) -> PublicJointData:
    """Crop source [*,192,H] tensors before any P06 schedule or metric."""

    if data.fit_observations.shape[1] < DECLARED_SEQUENCE_LENGTH:
        raise VisibilityFitError("public fit observations are shorter than the H128 declaration")
    if data.validation_observations.shape[1] < DECLARED_SEQUENCE_LENGTH:
        raise VisibilityFitError("public validation observations are shorter than the H128 declaration")
    return replace(
        data,
        fit_observations=data.fit_observations[:, :DECLARED_SEQUENCE_LENGTH].contiguous(),
        fit_truth=data.fit_truth[:, :DECLARED_SEQUENCE_LENGTH].contiguous(),
        fit_valid_mask=data.fit_valid_mask[:, :DECLARED_SEQUENCE_LENGTH].contiguous(),
        validation_observations=data.validation_observations[:, :DECLARED_SEQUENCE_LENGTH].contiguous(),
        validation_truth=data.validation_truth[:, :DECLARED_SEQUENCE_LENGTH].contiguous(),
        validation_valid_mask=data.validation_valid_mask[:, :DECLARED_SEQUENCE_LENGTH].contiguous(),
        metadata={
            **dict(data.metadata),
            "p06_geometry_crop": {
                "source_sequence_tokens": SOURCE_SEQUENCE_LENGTH,
                "declared_sequence_tokens": DECLARED_SEQUENCE_LENGTH,
                "positions_used": [0, DECLARED_SEQUENCE_LENGTH - 1],
                "source_positions_ignored": [DECLARED_SEQUENCE_LENGTH, SOURCE_SEQUENCE_LENGTH - 1],
            },
        },
    )


def _validate_public_data(data: PublicJointData) -> None:
    if data.hidden_size != HIDDEN_SIZE or data.vocabulary_size != VOCABULARY_SIZE:
        raise VisibilityFitError(
            f"public geometry mismatch: hidden={data.hidden_size}, vocab={data.vocabulary_size}"
        )
    for label, observations, truth, mask in (
        ("fit", data.fit_observations, data.fit_truth, data.fit_valid_mask),
        ("validation", data.validation_observations, data.validation_truth, data.validation_valid_mask),
    ):
        if tuple(observations.shape[1:]) != (DECLARED_SEQUENCE_LENGTH, HIDDEN_SIZE):
            raise VisibilityFitError(f"{label} observation geometry is not H128x2048")
        if tuple(truth.shape) != tuple(mask.shape) or tuple(truth.shape) != tuple(observations.shape[:2]):
            raise VisibilityFitError(f"{label} truth/mask geometry differs from observations")
        if not mask[:, 0].all().item():
            raise VisibilityFitError(f"{label} mask does not contain BOS in every row")
    if len(data.validation_groups) != int(data.validation_observations.shape[0]):
        raise VisibilityFitError("validation group metadata does not match rows")


def _load_cropped_public_data(
    args: argparse.Namespace,
    device: torch.device,
    *,
    deadline: float | None = None,
    guards: list[dict[str, Any]] | None = None,
) -> tuple[PublicJointData, dict[str, Any]]:
    fit_manifest = Path(args.fit_manifest).expanduser().resolve()
    validation_manifest = (
        None if args.validation_manifest is None else Path(args.validation_manifest).expanduser().resolve()
    )
    before = _resource_guard(args, device, stage="before_public_tensor_load", deadline=deadline)
    if guards is not None:
        guards.append(before)
    started = time.perf_counter()
    data = load_public_joint_data(
        fit_manifest,
        validation_manifest,
        embedding_path=(None if args.embedding_path is None else Path(args.embedding_path).expanduser().resolve()),
    )
    load_seconds = time.perf_counter() - started
    data = _crop_public_data(data)
    _validate_public_data(data)
    after = _resource_guard(args, device, stage="after_public_tensor_load_and_crop", deadline=deadline)
    if guards is not None:
        guards.append(after)
    embedding_path = Path(data.metadata["fit_paths"]["embedding_table"]["path"])
    receipt = {
        "fit_manifest": _file_record(fit_manifest),
        "validation_manifest": _file_record(validation_manifest or fit_manifest),
        "embedding_table": _file_record(embedding_path),
        "fit_record_count": len(data.fit_record_ids),
        "validation_record_count": len(data.validation_record_ids),
        "fit_geometry": list(data.fit_observations.shape),
        "validation_geometry": list(data.validation_observations.shape),
        "fit_valid_post_bos": int(data.fit_valid_mask[:, 1:].sum().item()),
        "validation_valid_post_bos": int(data.validation_valid_mask[:, 1:].sum().item()),
        "fit_valid_mask_sha256": tensor_sha256(data.fit_valid_mask),
        "validation_valid_mask_sha256": tensor_sha256(data.validation_valid_mask),
        "fit_record_ids_sha256": _canonical_hash(list(data.fit_record_ids)),
        "validation_record_ids_sha256": _canonical_hash(list(data.validation_record_ids)),
        "load_seconds": load_seconds,
        "crop": data.metadata["p06_geometry_crop"],
    }
    return data, receipt


def _position_bin(position: int) -> str:
    if 1 <= position <= 15:
        return "1-15"
    if 16 <= position <= 39:
        return "16-39"
    if 40 <= position <= 79:
        return "40-79"
    if 80 <= position <= 127:
        return "80-127"
    raise VisibilityFitError(f"position is outside the declared probe bins: {position}")


def _direct_error_ledger(
    data: PublicJointData,
    direct_state: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    seed: int,
    record_batch_size: int = RECORD_BATCH_SIZE,
    position_budget: int = POSITION_BUDGET,
) -> list[dict[str, Any]]:
    """Collect and freeze 64 initial direct-path errors in each position bin."""

    model = build_visibility_decoder(
        POSITIONWISE_METHOD,
        hidden_size=data.hidden_size,
        vocabulary_size=data.vocabulary_size,
        context_width=CONTEXT_WIDTH,
        qkv_seed=seed,
        direct_state=direct_state,
        direct_init_label="competent_public_affine",
    ).to(device)
    embedding = data.embedding_table.to(device=device, dtype=torch.float32)
    model.validate_embedding_table(embedding)
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in ("1-15", "16-39", "40-79", "80-127")}
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(data.fit_observations.shape[0]), record_batch_size):
            stop = min(start + record_batch_size, int(data.fit_observations.shape[0]))
            activation = data.fit_observations[start:stop].to(device=device, dtype=torch.float32)
            mask = data.fit_valid_mask[start:stop].to(device=device, dtype=torch.bool)
            truth = data.fit_truth[start:stop].to(device=device, dtype=torch.long)
            base = model.direct_pre_normalized_hidden(activation, mask)
            hidden = F.normalize(base, dim=-1)
            indices = torch.nonzero(mask, as_tuple=False)
            indices = indices[indices[:, 1] > 0]
            for chunk in indices.split(position_budget):
                rows = model.logits_from_rows(hidden, chunk[:, 0], chunk[:, 1], embedding)
                predictions, ties = deterministic_top1(rows)
                targets = truth[chunk[:, 0], chunk[:, 1]]
                wrong = predictions.ne(targets)
                chunk_cpu = chunk.detach().cpu()
                predictions_cpu = predictions.detach().cpu()
                targets_cpu = targets.detach().cpu()
                ties_cpu = ties.detach().cpu()
                wrong_cpu = wrong.detach().cpu()
                for row_index in torch.nonzero(wrong_cpu, as_tuple=False).reshape(-1).tolist():
                    global_record = start + int(chunk_cpu[row_index, 0])
                    position = int(chunk_cpu[row_index, 1])
                    bin_name = _position_bin(position)
                    candidates[bin_name].append(
                        {
                            "record_index": global_record,
                            "record_id": data.fit_record_ids[global_record],
                            "position": position,
                            "bin": bin_name,
                            "initial_prediction": int(predictions_cpu[row_index]),
                            "target": int(targets_cpu[row_index]),
                            "initial_tie_count": int(ties_cpu[row_index]),
                        }
                    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected: list[dict[str, Any]] = []
    for bin_name in ("1-15", "16-39", "40-79", "80-127"):
        rows = sorted(candidates[bin_name], key=lambda row: (row["record_index"], row["position"]))
        if len(rows) < 64:
            raise CapacityProbeError(
                f"public fit has only {len(rows)} initial affine errors in bin {bin_name}; need 64"
            )
        order = torch.randperm(len(rows), generator=generator)[:64].tolist()
        chosen = [rows[index] for index in order]
        selected.extend(sorted(chosen, key=lambda row: (row["record_index"], row["position"])))
    del model, embedding
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(selected) != 256 or len({(row["record_index"], row["position"]) for row in selected}) != 256:
        raise CapacityProbeError("capacity error ledger is not exactly 256 unique positions")
    return selected


def build_probe_schedule(
    error_ledger: Sequence[Mapping[str, Any]],
    *,
    steps: int = PROBE_STEPS,
    record_batch_size: int = RECORD_BATCH_SIZE,
    position_budget: int = POSITION_BUDGET,
    seed: int = PROBE_SEED,
) -> PositionSchedule:
    """Build the fixed 8-record/512-draw schedule over the 256-error ledger.

    Record batches are sampled from records represented in the ledger.  Within
    each selected batch, the 512 query positions are sampled with replacement
    only from that batch's ledger errors, so no non-error context silently
    enters the capacity probe.
    """

    if len(error_ledger) != 256:
        raise VisibilityFitError("capacity probe requires exactly 256 ledger rows")
    rows = []
    seen: set[tuple[int, int]] = set()
    by_record: dict[int, list[int]] = {}
    record_order: list[int] = []
    for row in error_ledger:
        record = int(row["record_index"])
        position = int(row["position"])
        key = (record, position)
        if key in seen or position <= 0 or position >= DECLARED_SEQUENCE_LENGTH:
            raise VisibilityFitError("capacity ledger contains duplicate or invalid position")
        seen.add(key)
        if record not in by_record:
            by_record[record] = []
            record_order.append(record)
        by_record[record].append(position)
        rows.append((record, position))
    if not record_order:
        raise VisibilityFitError("capacity ledger has no records")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    batch_records: list[torch.Tensor] = []
    draw_records: list[torch.Tensor] = []
    draw_positions: list[torch.Tensor] = []
    eligible_counts: list[int] = []
    used_replacement: list[bool] = []
    record_tensor = torch.tensor(record_order, dtype=torch.long)
    for _ in range(int(steps)):
        if len(record_order) >= record_batch_size:
            picked = torch.randperm(len(record_order), generator=generator)[:record_batch_size]
            global_records = record_tensor[picked]
        else:
            global_records = record_tensor[torch.randint(len(record_order), (record_batch_size,), generator=generator)]
        candidates: list[tuple[int, int]] = []
        for local_slot, global_record in enumerate(global_records.tolist()):
            candidates.extend((local_slot, position) for position in by_record[int(global_record)])
        if not candidates:
            raise VisibilityFitError("probe schedule batch contains no error positions")
        choices = torch.randint(len(candidates), (position_budget,), generator=generator)
        selected = [candidates[int(index)] for index in choices.tolist()]
        batch_records.append(global_records.to(dtype=torch.long))
        draw_records.append(torch.tensor([row[0] for row in selected], dtype=torch.long))
        draw_positions.append(torch.tensor([row[1] for row in selected], dtype=torch.long))
        eligible_counts.append(len(candidates))
        used_replacement.append(True)
    return PositionSchedule(
        batch_record_indices=torch.stack(batch_records).contiguous(),
        draw_record_slots=torch.stack(draw_records).contiguous(),
        draw_position_slots=torch.stack(draw_positions).contiguous(),
        eligible_counts=torch.tensor(eligible_counts, dtype=torch.int32),
        used_replacement=torch.tensor(used_replacement, dtype=torch.bool),
        seed=int(seed),
        position_budget=int(position_budget),
        record_batch_size=int(record_batch_size),
    )


def _ledger_hash(ledger: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_hash([dict(row) for row in ledger])


def _evaluate_error_ledger(
    model: VisibilityAffineAttentionDecoder,
    data: PublicJointData,
    embedding: torch.Tensor,
    ledger: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    model.validate_embedding_table(embedding)
    by_record: dict[int, list[Mapping[str, Any]]] = {}
    for row in ledger:
        by_record.setdefault(int(row["record_index"]), []).append(row)
    correct = 0
    total = 0
    bin_counts: dict[str, dict[str, int]] = {}
    records = sorted(by_record)
    with torch.inference_mode():
        for start in range(0, len(records), RECORD_BATCH_SIZE):
            selected_records = records[start : start + RECORD_BATCH_SIZE]
            global_indices = torch.tensor(selected_records, dtype=torch.long)
            activation = data.fit_observations.index_select(0, global_indices).to(device=device, dtype=torch.float32)
            mask = data.fit_valid_mask.index_select(0, global_indices).to(device=device, dtype=torch.bool)
            truth = data.fit_truth.index_select(0, global_indices).to(device=device, dtype=torch.long)
            hidden = model.projected_hidden(activation, mask)
            local_by_global = {record: local for local, record in enumerate(selected_records)}
            rows = [row for record in selected_records for row in by_record[record]]
            record_slots = torch.tensor([local_by_global[int(row["record_index"])] for row in rows], device=device)
            position_slots = torch.tensor([int(row["position"]) for row in rows], device=device)
            for chunk_start in range(0, len(rows), POSITION_BUDGET):
                chunk_stop = min(chunk_start + POSITION_BUDGET, len(rows))
                logits = model.logits_from_rows(
                    hidden,
                    record_slots[chunk_start:chunk_stop],
                    position_slots[chunk_start:chunk_stop],
                    embedding,
                )
                predictions, _ = deterministic_top1(logits)
                targets = truth[record_slots[chunk_start:chunk_stop], position_slots[chunk_start:chunk_stop]]
                hits = predictions.eq(targets).detach().cpu().tolist()
                for row, hit in zip(rows[chunk_start:chunk_stop], hits):
                    bin_name = str(row["bin"])
                    entry = bin_counts.setdefault(bin_name, {"rows": 0, "correct": 0})
                    entry["rows"] += 1
                    entry["correct"] += int(hit)
                    correct += int(hit)
                    total += 1
    return {
        "correct": correct,
        "total": total,
        "token_accuracy": correct / total if total else None,
        "by_bin": {
            name: {
                **values,
                "token_accuracy": values["correct"] / values["rows"] if values["rows"] else None,
            }
            for name, values in sorted(bin_counts.items())
        },
    }


def _peak_snapshot(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    snapshot = _memory_snapshot(device)
    snapshot.update(
        {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        }
    )
    return snapshot


def _run_probe_arm(
    method_id: str,
    data: PublicJointData,
    direct_state: Mapping[str, torch.Tensor],
    embedding: torch.Tensor,
    schedule: PositionSchedule,
    ledger: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
    deadline: float,
    guards: list[dict[str, Any]],
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = build_visibility_decoder(
        method_id,
        hidden_size=HIDDEN_SIZE,
        vocabulary_size=VOCABULARY_SIZE,
        context_width=CONTEXT_WIDTH,
        qkv_seed=PROBE_SEED,
        direct_state=direct_state,
        direct_init_label="competent_public_affine",
    ).to(device)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_digest = state_sha256(initial_state)
    for name in ("W", "b", "s"):
        getattr(model, name).requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable) != 8:
        raise VisibilityFitError(f"probe {method_id} does not expose exactly 8 added-path tensors")
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROBE_STEPS)
    runtime_embedding = embedding.to(device=device, dtype=torch.float32)
    model.validate_embedding_table(runtime_embedding)
    curve: list[dict[str, Any]] = []
    initial_metrics = _evaluate_error_ledger(model, data, runtime_embedding, ledger, device=device)
    curve.append({"step": 0, "metrics": initial_metrics, "learning_rate": LEARNING_RATE})
    update_seconds = 0.0
    eval_seconds = 0.0
    for step_index in range(PROBE_STEPS):
        guards.append(_resource_guard(args, device, stage=f"probe:{method_id}:before_step_{step_index}", deadline=deadline))
        update_started = time.perf_counter()
        train_point = train_step(
            model,
            data.fit_observations,
            data.fit_truth,
            data.fit_valid_mask,
            runtime_embedding,
            schedule,
            step_index,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=GRADIENT_CLIP_NORM,
        )
        scheduler.step()
        elapsed = time.perf_counter() - update_started
        update_seconds += elapsed
        guards.append(_resource_guard(args, device, stage=f"probe:{method_id}:after_step_{step_index}", deadline=deadline))
        next_step = step_index + 1
        if next_step % 50 == 0 or next_step == PROBE_STEPS:
            eval_started = time.perf_counter()
            metrics = _evaluate_error_ledger(model, data, runtime_embedding, ledger, device=device)
            eval_seconds += time.perf_counter() - eval_started
            curve.append(
                {
                    "step": next_step,
                    "metrics": metrics,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train": train_point,
                }
            )
    direct_after = {name: getattr(model, name).detach().cpu().clone() for name in ("W", "b", "s")}
    direct_before = {name: initial_state[name] for name in ("W", "b", "s")}
    if any(not torch.equal(direct_before[name], direct_after[name]) for name in direct_before):
        raise VisibilityFitError(f"probe {method_id} mutated frozen direct affine parameters")
    final_metrics = curve[-1]["metrics"]
    passed = bool(final_metrics["correct"] >= 52 and all(math.isfinite(float(value)) for value in (final_metrics["token_accuracy"],)))
    curve_path = output_root / f"{method_id}.learning_curve.json"
    _json_write_create(
        curve_path,
        {
            "schema": PROBE_SCHEMA,
            "task_id": TASK_ID,
            "method_id": method_id,
            "selection": "diagnostic only; probe state discarded",
            "direct_affine_frozen": True,
            "curve": curve,
        },
    )
    peak = _peak_snapshot(model, device)
    result = {
        "method_id": method_id,
        "status": "PASS" if passed else "FAIL",
        "seed": PROBE_SEED,
        "steps": PROBE_STEPS,
        "query_draws_per_step": POSITION_BUDGET,
        "direct_affine_frozen": True,
        "initial_state_sha256": initial_digest,
        "final_state_sha256": state_sha256({name: value.detach().cpu() for name, value in model.state_dict().items()}),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "pass_threshold_correct": 52,
        "learning_curve": {"path": str(curve_path), "sha256": file_sha256(curve_path)},
        "update_seconds": update_seconds,
        "evaluation_seconds": eval_seconds,
        "arm_wall_seconds": time.perf_counter() - started,
        "peak_memory": peak,
        "last_train": curve[-1].get("train"),
    }
    del model, optimizer, scheduler, runtime_embedding
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_capacity_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise VisibilityFitError(f"probe output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    repository_root = Path(args.repository_root).expanduser().resolve()
    device = _device(args.device)
    deadline = time.perf_counter() + float(args.max_seconds)
    guards: list[dict[str, Any]] = []
    direct_state, direct_descriptor = _direct_state(args)
    preflight_receipt = _validate_preflight_receipt(
        Path(args.preflight_receipt).expanduser().resolve(),
        direct_hash=direct_descriptor["sha256"],
        fit_manifest=Path(args.fit_manifest).expanduser().resolve(),
        validation_manifest=(None if args.validation_manifest is None else Path(args.validation_manifest).expanduser().resolve()),
    )
    data, data_receipt = _load_cropped_public_data(
        args, device, deadline=deadline, guards=guards
    )
    ledger_started = time.perf_counter()
    ledger = _direct_error_ledger(data, direct_state, device=device, seed=PROBE_SEED)
    ledger_hash = _ledger_hash(ledger)
    ledger_path = output_root / "capacity_error_ledger.json"
    _json_write_create(
        ledger_path,
        {
            "schema": "token-reconstruction.trr-p06-capacity-error-ledger.v1",
            "task_id": TASK_ID,
            "source": "public_fit_only",
            "selection_seed": PROBE_SEED,
            "row_count": len(ledger),
            "rows": ledger,
            "sha256": ledger_hash,
        },
    )
    schedule = build_probe_schedule(ledger, seed=PROBE_SEED)
    schedule_record = save_schedule(output_root / "capacity_schedule.safetensors", schedule)
    embedding = data.embedding_table
    method_results: list[dict[str, Any]] = []
    for method_id in METHODS:
        method_results.append(
            _run_probe_arm(
                method_id,
                data,
                direct_state,
                embedding,
                schedule,
                ledger,
                args=args,
                device=device,
                output_root=output_root,
                deadline=deadline,
                guards=guards,
            )
        )
    passed = all(row["status"] == "PASS" for row in method_results)
    receipt = {
        "schema": PROBE_SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS" if passed else "FAIL",
        "created_utc": _utc_now(),
        "source_commit": _git_commit(repository_root),
        "command": list(sys.argv),
        "environment": _safe_environment(),
        "device": str(device),
        "direct_affine": direct_descriptor,
        "preflight_receipt": {
            "path": str(Path(args.preflight_receipt).expanduser().resolve()),
            "sha256": file_sha256(Path(args.preflight_receipt).expanduser().resolve()),
            "status": preflight_receipt["status"],
        },
        "data": data_receipt,
        "geometry": {
            "fit": list(data.fit_observations.shape),
            "validation": list(data.validation_observations.shape),
            "record_batch_size": RECORD_BATCH_SIZE,
            "query_draws_per_step": POSITION_BUDGET,
        },
        "capacity_ledger": {
            "path": str(ledger_path),
            "sha256": file_sha256(ledger_path),
            "row_count": len(ledger),
            "selection_bins": {name: 64 for name in ("1-15", "16-39", "40-79", "80-127")},
            "construction_seconds": time.perf_counter() - ledger_started,
        },
        "schedule": schedule_record,
        "schedule_metadata": schedule_metadata(schedule),
        "methods": method_results,
        "resource_policy": {
            "minimum_free_gpu_gib": float(args.minimum_free_gib),
            "maximum_gpu_reserved_gib": float(args.maximum_gpu_reserved_gib),
            "maximum_host_rss_gib": float(args.maximum_host_rss_gib),
            "minimum_host_available_gib": float(args.minimum_host_available_gib),
            "max_seconds": float(args.max_seconds),
        },
        "resource_guards": guards,
        "failure_policy": "a failed arm stops before main fits and fresh evaluation",
    }
    receipt_path = output_root / "capacity_probe_receipt.json"
    _json_write_create(receipt_path, receipt)
    if not passed:
        raise CapacityProbeError("capacity probe failed for at least one visibility arm")
    return receipt


def _validation_steps(steps: int) -> tuple[int, ...]:
    if steps <= 0:
        raise VisibilityFitError("main fit steps must be positive")
    return tuple(range(0, int(steps) + 1, VALIDATION_EVERY)) if steps % VALIDATION_EVERY == 0 else tuple([0, *range(VALIDATION_EVERY, steps, VALIDATION_EVERY), steps])


def _train_main_arm(
    method_id: str,
    seed: int,
    data: PublicJointData,
    direct_state: Mapping[str, torch.Tensor],
    embedding: torch.Tensor,
    schedule: PositionSchedule,
    schedule_record: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    deadline: float,
    guards: list[dict[str, Any]],
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_visibility_decoder(
        method_id,
        hidden_size=HIDDEN_SIZE,
        vocabulary_size=VOCABULARY_SIZE,
        context_width=CONTEXT_WIDTH,
        qkv_seed=seed,
        direct_state=direct_state,
        direct_init_label="competent_public_affine",
    ).to(device)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_digest = state_sha256(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAIN_STEPS)
    runtime_embedding = embedding.to(device=device, dtype=torch.float32)
    model.validate_embedding_table(runtime_embedding)
    checkpoints = _validation_steps(MAIN_STEPS)
    curve: list[dict[str, Any]] = []
    best_metric = -float("inf")
    best_step = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    last_train: dict[str, Any] | None = None
    validation_seconds = 0.0
    update_seconds = 0.0
    for step_index in range(MAIN_STEPS + 1):
        guards.append(_resource_guard(args, device, stage=f"main:{seed}:{method_id}:before_step_{step_index}", deadline=deadline))
        if step_index in checkpoints:
            validation_started = time.perf_counter()
            metrics = evaluate_dataset(
                model,
                data.validation_observations,
                data.validation_truth,
                data.validation_valid_mask,
                runtime_embedding,
                data.validation_groups,
                device=device,
                record_batch_size=RECORD_BATCH_SIZE,
                position_budget=POSITION_BUDGET,
            )
            elapsed = time.perf_counter() - validation_started
            validation_seconds += elapsed
            point = {
                "step": int(step_index),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": None if last_train is None else dict(last_train),
                "validation": metrics,
                "validation_wall_seconds": elapsed,
            }
            curve.append(point)
            metric = float(metrics["style_balanced_token_accuracy"])
            if metric > best_metric:
                best_metric = metric
                best_step = int(step_index)
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if step_index == MAIN_STEPS:
            break
        update_started = time.perf_counter()
        last_train = train_step(
            model,
            data.fit_observations,
            data.fit_truth,
            data.fit_valid_mask,
            runtime_embedding,
            schedule,
            step_index,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=GRADIENT_CLIP_NORM,
        )
        scheduler.step()
        update_seconds += time.perf_counter() - update_started
        last_train["step"] = int(step_index + 1)
        guards.append(_resource_guard(args, device, stage=f"main:{seed}:{method_id}:after_step_{step_index}", deadline=deadline))
    selected_model = build_visibility_decoder(
        method_id,
        hidden_size=HIDDEN_SIZE,
        vocabulary_size=VOCABULARY_SIZE,
        context_width=CONTEXT_WIDTH,
        qkv_seed=seed,
        direct_state=direct_state,
        direct_init_label="competent_public_affine",
    )
    selected_model.load_state_dict(best_state, strict=True)
    curve_path = output_dir / "learning_curve.json"
    _json_write_create(
        curve_path,
        {
            "schema": "token-reconstruction.trr-p06-learning-curve.v1",
            "task_id": TASK_ID,
            "method_id": method_id,
            "seed": seed,
            "selection_metric": "validation_style_balanced_token_accuracy",
            "selection_rule": "earliest maximum, validation every 100 steps including step 0",
            "curve": curve,
        },
    )
    state_record = save_visibility_state(
        output_dir / "selected.safetensors",
        selected_model,
        selected_step=best_step,
        metadata={
            "fit_seed": seed,
            "schedule_sha256": schedule_record["schedule_sha256"],
            "direct_affine_sha256": str(args.direct_affine_sha256),
            "selection_metric": "validation_style_balanced_token_accuracy",
            "selection_rule": "earliest maximum, validation every 100 steps including step 0",
        },
    )
    guards.append(_resource_guard(args, device, stage=f"main:{seed}:{method_id}:after_state_save", deadline=deadline))
    peak = _peak_snapshot(model, device)
    result = {
        "method_id": method_id,
        "seed": seed,
        "status": "PASS",
        "steps": MAIN_STEPS,
        "checkpoint_steps": list(checkpoints),
        "selected_step": best_step,
        "best_validation_style_balanced_token_accuracy": best_metric,
        "initial_state_sha256": initial_digest,
        "learning_curve": {"path": str(curve_path), "bytes": curve_path.stat().st_size, "sha256": file_sha256(curve_path)},
        "state": state_record,
        "schedule_sha256": schedule_record["schedule_sha256"],
        "direct_affine_sha256": str(args.direct_affine_sha256),
        "validation_seconds": validation_seconds,
        "update_seconds": update_seconds,
        "arm_wall_seconds": time.perf_counter() - started,
        "peak_memory": peak,
        "last_train": last_train,
        "parameter_count": int(model.parameter_count),
        "effective_parameter_count": int(model.effective_parameter_count),
    }
    del selected_model, model, optimizer, scheduler, runtime_embedding
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _validate_preflight_receipt(
    path: Path,
    *,
    direct_hash: str,
    fit_manifest: Path,
    validation_manifest: Path | None,
) -> Mapping[str, Any]:
    receipt = _json_read(path, label="P06 preflight receipt")
    if receipt.get("status") != "SOURCE_ONLY_PREFLIGHT_PASS":
        raise VisibilityFitError("probe/main mode requires a passing source-only P06 preflight")
    direct = receipt.get("direct_affine")
    if not isinstance(direct, Mapping) or direct.get("sha256") != direct_hash:
        raise VisibilityFitError("P06 preflight direct-affine hash does not match this run")
    expected_fit = _file_record(fit_manifest)
    expected_validation = _file_record(validation_manifest or fit_manifest)
    if receipt.get("fit_manifest") != expected_fit:
        raise VisibilityFitError("P06 preflight fit manifest binding differs from this run")
    if receipt.get("validation_manifest") != expected_validation:
        raise VisibilityFitError("P06 preflight validation manifest binding differs from this run")
    geometry = receipt.get("memory_preflight", {}).get("geometry")
    if geometry != {
        "hidden_size": HIDDEN_SIZE,
        "vocabulary_size": VOCABULARY_SIZE,
        "sequence_length": DECLARED_SEQUENCE_LENGTH,
        "record_batch_size": RECORD_BATCH_SIZE,
        "position_budget": POSITION_BUDGET,
        "context_width": CONTEXT_WIDTH,
    }:
        raise VisibilityFitError("P06 preflight geometry differs from the registered H128 recipe")
    return receipt


def _validate_probe_receipt(path: Path, direct_hash: str, data_receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt = _json_read(path, label="capacity probe receipt")
    if receipt.get("status") != "PASS":
        raise VisibilityFitError("main fit requires a PASS capacity probe receipt")
    direct = receipt.get("direct_affine")
    if not isinstance(direct, Mapping) or direct.get("sha256") != direct_hash:
        raise VisibilityFitError("capacity probe direct-affine hash does not match main fit")
    data = receipt.get("data")
    if not isinstance(data, Mapping):
        raise VisibilityFitError("capacity probe has no public data binding")
    if data.get("fit_manifest", {}).get("sha256") != data_receipt["fit_manifest"]["sha256"]:
        raise VisibilityFitError("capacity probe fit manifest differs from main fit")
    if data.get("fit_geometry") != [1200, 128, 2048]:
        raise VisibilityFitError("capacity probe was not run on the declared H128 fit geometry")
    return receipt


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise VisibilityFitError(f"main output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    repository_root = Path(args.repository_root).expanduser().resolve()
    device = _device(args.device)
    deadline = time.perf_counter() + float(args.max_seconds)
    guards: list[dict[str, Any]] = []
    direct_state, direct_descriptor = _direct_state(args)
    preflight_receipt = _validate_preflight_receipt(
        Path(args.preflight_receipt).expanduser().resolve(),
        direct_hash=direct_descriptor["sha256"],
        fit_manifest=Path(args.fit_manifest).expanduser().resolve(),
        validation_manifest=(None if args.validation_manifest is None else Path(args.validation_manifest).expanduser().resolve()),
    )
    data, data_receipt = _load_cropped_public_data(
        args, device, deadline=deadline, guards=guards
    )
    probe_receipt = _validate_probe_receipt(Path(args.probe_receipt), direct_descriptor["sha256"], data_receipt)
    schedules: dict[str, dict[str, Any]] = {}
    methods: list[dict[str, Any]] = []
    for seed in MAIN_SEEDS:
        schedule = build_position_schedule(
            data.fit_valid_mask,
            steps=MAIN_STEPS,
            record_batch_size=RECORD_BATCH_SIZE,
            position_budget=POSITION_BUDGET,
            seed=seed,
        )
        seed_root = output_root / f"seed-{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        schedule_record = save_schedule(seed_root / "position_schedule.safetensors", schedule)
        schedules[str(seed)] = schedule_record
        for method_id in METHODS:
            method_root = seed_root / method_id
            method_root.mkdir(parents=True, exist_ok=False)
            methods.append(
                _train_main_arm(
                    method_id,
                    seed,
                    data,
                    direct_state,
                    data.embedding_table,
                    schedule,
                    schedule_record,
                    args=args,
                    device=device,
                    output_dir=method_root,
                    deadline=deadline,
                    guards=guards,
                )
            )
    receipt = {
        "schema": MAIN_SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS",
        "created_utc": _utc_now(),
        "source_commit": _git_commit(repository_root),
        "command": list(sys.argv),
        "environment": _safe_environment(),
        "device": str(device),
        "direct_affine": direct_descriptor,
        "preflight_receipt": {
            "path": str(Path(args.preflight_receipt).expanduser().resolve()),
            "sha256": file_sha256(Path(args.preflight_receipt).expanduser().resolve()),
            "status": preflight_receipt["status"],
        },
        "capacity_probe_receipt": {
            "path": str(Path(args.probe_receipt).expanduser().resolve()),
            "sha256": file_sha256(Path(args.probe_receipt).expanduser().resolve()),
            "status": probe_receipt["status"],
        },
        "data": data_receipt,
        "geometry": {
            "fit": list(data.fit_observations.shape),
            "validation": list(data.validation_observations.shape),
            "record_batch_size": RECORD_BATCH_SIZE,
            "query_draws_per_step": POSITION_BUDGET,
            "steps": MAIN_STEPS,
            "validation_every": VALIDATION_EVERY,
        },
        "seeds": list(MAIN_SEEDS),
        "schedules": schedules,
        "methods": methods,
        "resource_policy": {
            "minimum_free_gpu_gib": float(args.minimum_free_gib),
            "maximum_gpu_reserved_gib": float(args.maximum_gpu_reserved_gib),
            "maximum_host_rss_gib": float(args.maximum_host_rss_gib),
            "minimum_host_available_gib": float(args.minimum_host_available_gib),
            "max_seconds": float(args.max_seconds),
        },
        "resource_guards": guards,
        "selection": "earliest maximum public-validation style-balanced token accuracy, checked every 100 steps including step 0",
        "runtime_components": {
            "source_token_access": False,
            "target_truth_access": False,
            "guessed_token_feedback": False,
            "candidate_simulations": 0,
            "a2_student": False,
            "supervision": "same-position full-vocabulary public CE only",
        },
    }
    receipt_path = output_root / "main_fit_receipt.json"
    _json_write_create(receipt_path, receipt)
    return receipt


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise VisibilityFitError(f"preflight output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    repository_root = Path(args.repository_root).expanduser().resolve()
    direct_state, direct_descriptor = _direct_state(args)
    del direct_state
    fit_manifest = Path(args.fit_manifest).expanduser().resolve()
    validation_manifest = fit_manifest if args.validation_manifest is None else Path(args.validation_manifest).expanduser().resolve()
    fit_payload = _manifest_payload(fit_manifest)
    validation_payload = _manifest_payload(validation_manifest)
    fit_shape = _resource_shape(fit_payload, "fit_observations")
    validation_shape = _resource_shape(validation_payload, "validation_observations")
    if fit_shape[-1] != HIDDEN_SIZE or validation_shape[-1] != HIDDEN_SIZE:
        raise VisibilityFitError("preflight manifest hidden geometry differs from 2048")
    if fit_shape[1] < DECLARED_SEQUENCE_LENGTH or validation_shape[1] < DECLARED_SEQUENCE_LENGTH:
        raise VisibilityFitError("preflight source sequence is shorter than H128")
    geometry = _preflight_geometry(
        hidden_size=HIDDEN_SIZE,
        vocabulary_size=VOCABULARY_SIZE,
        sequence_length=DECLARED_SEQUENCE_LENGTH,
        record_batch_size=RECORD_BATCH_SIZE,
        position_budget=POSITION_BUDGET,
        context_width=CONTEXT_WIDTH,
    )
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID,
        "status": "SOURCE_ONLY_PREFLIGHT_PASS",
        "created_utc": _utc_now(),
        "source_commit": _git_commit(repository_root),
        "command": list(sys.argv),
        "environment": _safe_environment(),
        "direct_affine": direct_descriptor,
        "fit_manifest": _file_record(fit_manifest),
        "validation_manifest": _file_record(validation_manifest),
        "source_manifest_geometry": {"fit": list(fit_shape), "validation": list(validation_shape)},
        "declared_crop_geometry": [None, DECLARED_SEQUENCE_LENGTH, HIDDEN_SIZE],
        "memory_preflight": geometry,
        "resource_policy": {
            "minimum_free_gpu_gib": float(args.minimum_free_gib),
            "maximum_gpu_reserved_gib": float(args.maximum_gpu_reserved_gib),
            "maximum_host_rss_gib": float(args.maximum_host_rss_gib),
            "minimum_host_available_gib": float(args.minimum_host_available_gib),
            "max_seconds": float(args.max_seconds),
        },
        "execution_hold": "no public tensors, fresh panel, truth, or GPU computation opened by this mode",
    }
    _json_write_create(output_root / "preflight.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "probe", "main"), required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--embedding-path", type=Path)
    parser.add_argument("--direct-affine-state", type=Path, required=True)
    parser.add_argument("--direct-affine-sha256", default=DIRECT_AFFINE_SHA256)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--probe-receipt", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--minimum-free-gib", type=float, default=MINIMUM_FREE_GPU_GIB)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=MAXIMUM_GPU_RESERVED_GIB)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=MAXIMUM_HOST_RSS_GIB)
    parser.add_argument("--minimum-host-available-gib", type=float, default=MINIMUM_HOST_AVAILABLE_GIB)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode in ("probe", "main") and args.preflight_receipt is None:
        raise VisibilityFitError(f"{args.mode} mode requires --preflight-receipt from a source-only PASS")
    if args.mode == "preflight" and args.preflight_receipt is not None:
        raise VisibilityFitError("--preflight-receipt is valid only after preflight mode")
    if args.mode == "main" and args.probe_receipt is None:
        raise VisibilityFitError("main mode requires --probe-receipt from a PASS capacity probe")
    if args.mode != "main" and args.probe_receipt is not None:
        raise VisibilityFitError("--probe-receipt is valid only in main mode")
    for name in (
        "minimum_free_gib",
        "maximum_gpu_reserved_gib",
        "maximum_host_rss_gib",
        "minimum_host_available_gib",
        "max_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise VisibilityFitError(f"{name} must be finite and positive")
    if args.direct_affine_sha256 != DIRECT_AFFINE_SHA256:
        raise VisibilityFitError("P06 production runner requires the registered direct affine hash")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    try:
        _validate_args(args)
        _set_runtime_threads(args)
        if args.mode == "preflight":
            run_preflight(args)
        elif args.mode == "probe":
            run_capacity_probe(args)
        else:
            run_main(args)
        return 0
    except Exception as exc:
        try:
            if output_root.exists() and output_root.is_dir() and not (output_root / "failure.json").exists():
                _json_write_create(
                    output_root / "failure.json",
                    {
                        "schema": FAILURE_SCHEMA,
                        "task_id": TASK_ID,
                        "status": "FAIL",
                        "created_utc": _utc_now(),
                        "command": list(sys.argv),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "source_commit": _git_commit(Path(args.repository_root).expanduser().resolve()),
                        "environment": _safe_environment(),
                    },
                )
        except Exception:
            pass
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
