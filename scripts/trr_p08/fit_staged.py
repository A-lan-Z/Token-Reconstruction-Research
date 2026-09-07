#!/usr/bin/env python3
"""Run the bounded TRR-P08 joint-versus-staged decoder comparison.

The runner reuses the published P06 visibility decoder and public fit/validation
loader.  It has three create-only modes: ``preflight`` (metadata only),
``qualify`` (one disposable largest-geometry backward cell), and ``main`` (the
predeclared 2 visibility x 2 schedule x 2 seed matrix).  No source tokens,
private truth, guessed prefixes, candidate search, or A2 computation are used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from token_reconstruction.trr0005_joint_decoder import (
    PublicJointData,
    PositionSchedule,
    build_position_schedule,
    evaluate_dataset,
    load_public_joint_data,
    save_schedule,
    schedule_digest,
    schedule_metadata,
    tensor_sha256,
    train_step,
)
from token_reconstruction.trr0006_visibility_decoder import (
    DEFAULT_CONTEXT_WIDTH,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_VOCABULARY_SIZE,
    PAST_ONLY_METHOD,
    POSITIONWISE_METHOD,
    VisibilityAffineAttentionDecoder,
    build_visibility_decoder,
    file_sha256,
    save_visibility_state,
    state_sha256,
)

TASK_ID = "TRR-P08"
SCHEMA = "token-reconstruction.trr-p08-staged-fit.v1"
PREFLIGHT_SCHEMA = "token-reconstruction.trr-p08-staged-fit-preflight.v1"
QUALIFICATION_SCHEMA = "token-reconstruction.trr-p08-staged-fit-qualification.v1"
FAILURE_SCHEMA = "token-reconstruction.trr-p08-staged-fit-failure.v1"

HIDDEN_SIZE = DEFAULT_HIDDEN_SIZE
VOCABULARY_SIZE = DEFAULT_VOCABULARY_SIZE
CONTEXT_WIDTH = DEFAULT_CONTEXT_WIDTH
SOURCE_SEQUENCE_LENGTH = 192
SEQUENCE_LENGTH = 128
RECORD_BATCH_SIZE = 8
POSITION_BUDGET = 512
TOTAL_STEPS = 3000
STAGED_AFFINE_STEPS = 1000
VALIDATION_EVERY = 100
SEEDS = (6106, 6107)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0
QUALIFICATION_STEPS = 2
COHORT_PER_BIN = 64
POSITION_BINS = ("1-15", "16-39", "40-79", "80-127")
MINIMUM_FREE_GPU_GIB = 8.0
MAXIMUM_GPU_RESERVED_GIB = 6.0
MAXIMUM_HOST_RSS_GIB = 16.0
MINIMUM_HOST_AVAILABLE_GIB = 10.0
MAX_SECONDS = 1800.0

P08_POSITIONWISE_JOINT = "p08_positionwise_joint"
P08_POSITIONWISE_STAGED = "p08_positionwise_staged"
P08_PAST_ONLY_JOINT = "p08_past_only_joint"
P08_PAST_ONLY_STAGED = "p08_past_only_staged"

@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    visibility_method: str
    schedule_kind: str

    @property
    def staged(self) -> bool:
        return self.schedule_kind == "staged_affine_then_full"


ARM_SPECS = (
    ArmSpec(P08_POSITIONWISE_STAGED, POSITIONWISE_METHOD, "staged_affine_then_full"),
    ArmSpec(P08_PAST_ONLY_STAGED, PAST_ONLY_METHOD, "staged_affine_then_full"),
    ArmSpec(P08_POSITIONWISE_JOINT, POSITIONWISE_METHOD, "joint_full"),
    ArmSpec(P08_PAST_ONLY_JOINT, PAST_ONLY_METHOD, "joint_full"),
)
ARM_BY_ID = {spec.arm_id: spec for spec in ARM_SPECS}


class P08FitError(RuntimeError):
    """Raised when a P08 contract or resource guard fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_read(path: Path, *, label: str) -> Mapping[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P08FitError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise P08FitError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise P08FitError(f"{label} must be a JSON object: {path}")
    return value


def _json_write_create(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise P08FitError(f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P08FitError(f"expected regular file: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _environment() -> dict[str, str]:
    names = ("CUDA_VISIBLE_DEVICES", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONPATH")
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
        result.update({
            "cuda_free_bytes": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        })
    return result


def _guard(args: argparse.Namespace, device: torch.device, *, stage: str, deadline: float | None) -> dict[str, Any]:
    if deadline is not None and time.perf_counter() >= deadline:
        raise P08FitError(f"wall-time guard expired at {stage}")
    snapshot = _memory_snapshot(device)
    max_rss = int(float(args.maximum_host_rss_gib) * 1024**3)
    if snapshot["process_max_rss_bytes"] > max_rss:
        raise P08FitError(f"host RSS guard exceeded at {stage}")
    available = snapshot["host_available_bytes"]
    if available is not None and available < int(float(args.minimum_host_available_gib) * 1024**3):
        raise P08FitError(f"host available-memory guard exceeded at {stage}")
    if device.type == "cuda":
        if int(snapshot["cuda_free_bytes"]) < int(float(args.minimum_free_gib) * 1024**3):
            raise P08FitError(f"GPU free-memory guard exceeded at {stage}")
        if int(snapshot["cuda_reserved_bytes"]) > int(float(args.maximum_gpu_reserved_gib) * 1024**3):
            raise P08FitError(f"GPU reserved-memory guard exceeded at {stage}")
    return {"stage": stage, **snapshot, "timestamp_utc": _utc_now()}


def _peak_snapshot(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = _memory_snapshot(device)
    result.update({
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
    })
    return result


def _set_threads(args: argparse.Namespace) -> None:
    if int(args.torch_threads) <= 0 or int(args.torch_interop_threads) <= 0:
        raise P08FitError("Torch thread counts must be positive")
    torch.set_num_threads(int(args.torch_threads))
    torch.set_num_interop_threads(int(args.torch_interop_threads))


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise P08FitError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise P08FitError("CUDA requested but unavailable")
    return device


def standard_direct_state(hidden_size: int = HIDDEN_SIZE) -> dict[str, torch.Tensor]:
    """Return the registered P08 W=I, b=0, s=3 standard initialization."""

    return {
        "W": torch.eye(hidden_size, dtype=torch.float32),
        "b": torch.zeros(hidden_size, dtype=torch.float32),
        "s": torch.tensor(3.0, dtype=torch.float32),
    }


def standard_initialization_sha256(hidden_size: int = HIDDEN_SIZE) -> str:
    return state_sha256(standard_direct_state(hidden_size))


def build_arm_model(spec: ArmSpec, seed: int, *, hidden_size: int = HIDDEN_SIZE, vocabulary_size: int = VOCABULARY_SIZE, context_width: int = CONTEXT_WIDTH) -> VisibilityAffineAttentionDecoder:
    if spec.arm_id not in ARM_BY_ID:
        raise P08FitError(f"unknown P08 arm: {spec.arm_id}")
    model = build_visibility_decoder(
        spec.visibility_method,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
        qkv_seed=int(seed),
        direct_state=standard_direct_state(hidden_size),
        direct_init_label="p08_standard_identity_affine",
    )
    return model


def set_training_phase(model: VisibilityAffineAttentionDecoder, phase: str) -> None:
    """Freeze the added correction path during affine-only updates."""

    if phase not in ("affine_only", "full"):
        raise P08FitError(f"unknown training phase: {phase}")
    affine_names = {"W", "b", "s"}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(phase == "full" or name in affine_names)
    if phase == "affine_only":
        if not all(getattr(model, name).requires_grad for name in affine_names):
            raise P08FitError("affine parameters were not enabled")
        correction_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.split(".", 1)[0] in {"query", "key", "value", "output"}
        ]
        if any(parameter.requires_grad for parameter in correction_parameters):
            raise P08FitError("correction parameters remained trainable during affine phase")
        with torch.no_grad():
            if not torch.equal(model.output.weight, torch.zeros_like(model.output.weight)) or not torch.equal(model.output.bias, torch.zeros_like(model.output.bias)):
                raise P08FitError("affine phase correction output is not zero initialized")
    else:
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise P08FitError("full phase did not unfreeze every parameter")


def _crop_public_data(data: PublicJointData) -> PublicJointData:
    def crop(value: torch.Tensor) -> torch.Tensor:
        return value[:, :SEQUENCE_LENGTH].contiguous()
    return replace(
        data,
        fit_observations=crop(data.fit_observations),
        fit_truth=crop(data.fit_truth),
        fit_valid_mask=crop(data.fit_valid_mask),
        validation_observations=crop(data.validation_observations),
        validation_truth=crop(data.validation_truth),
        validation_valid_mask=crop(data.validation_valid_mask),
        metadata={
            **dict(data.metadata),
            "p08_geometry_crop": {
                "source_sequence_tokens": SOURCE_SEQUENCE_LENGTH,
                "declared_sequence_tokens": SEQUENCE_LENGTH,
                "positions_used": [0, SEQUENCE_LENGTH - 1],
                "source_positions_ignored": [SEQUENCE_LENGTH, SOURCE_SEQUENCE_LENGTH - 1],
            },
        },
    )


def _validate_public_data(data: PublicJointData) -> None:
    if data.hidden_size != HIDDEN_SIZE or data.vocabulary_size != VOCABULARY_SIZE:
        raise P08FitError("public geometry differs from H128/V128256")
    for label, observations, truth, mask in (("fit", data.fit_observations, data.fit_truth, data.fit_valid_mask), ("validation", data.validation_observations, data.validation_truth, data.validation_valid_mask)):
        if tuple(observations.shape[1:]) != (SEQUENCE_LENGTH, HIDDEN_SIZE):
            raise P08FitError(f"{label} is not H128x2048 after crop")
        if tuple(truth.shape) != tuple(observations.shape[:2]) or tuple(mask.shape) != tuple(observations.shape[:2]):
            raise P08FitError(f"{label} truth/mask geometry differs")
        if not mask[:, 0].all().item():
            raise P08FitError(f"{label} does not contain BOS in every record")


def _load_data(args: argparse.Namespace, device: torch.device, *, deadline: float | None, guards: list[dict[str, Any]]) -> tuple[PublicJointData, dict[str, Any]]:
    guards.append(_guard(args, device, stage="before_public_tensor_load", deadline=deadline))
    started = time.perf_counter()
    data = load_public_joint_data(
        Path(args.fit_manifest).expanduser().resolve(),
        None if args.validation_manifest is None else Path(args.validation_manifest).expanduser().resolve(),
        embedding_path=None if args.embedding_path is None else Path(args.embedding_path).expanduser().resolve(),
    )
    load_seconds = time.perf_counter() - started
    data = _crop_public_data(data)
    _validate_public_data(data)
    guards.append(_guard(args, device, stage="after_public_tensor_load_and_crop", deadline=deadline))
    fit_manifest = Path(args.fit_manifest).expanduser().resolve()
    validation_manifest = fit_manifest if args.validation_manifest is None else Path(args.validation_manifest).expanduser().resolve()
    return data, {
        "fit_manifest": _file_record(fit_manifest),
        "validation_manifest": _file_record(validation_manifest),
        "embedding_table": {"path": str(data.metadata["fit_paths"]["embedding_table"]["path"]), "sha256": file_sha256(Path(data.metadata["fit_paths"]["embedding_table"]["path"]))},
        "fit_record_count": len(data.fit_record_ids),
        "validation_record_count": len(data.validation_record_ids),
        "fit_geometry": list(data.fit_observations.shape),
        "validation_geometry": list(data.validation_observations.shape),
        "fit_valid_mask_sha256": tensor_sha256(data.fit_valid_mask),
        "validation_valid_mask_sha256": tensor_sha256(data.validation_valid_mask),
        "fit_record_ids_sha256": hashlib.sha256(json.dumps(list(data.fit_record_ids), separators=(",", ":")).encode()).hexdigest(),
        "validation_record_ids_sha256": hashlib.sha256(json.dumps(list(data.validation_record_ids), separators=(",", ":")).encode()).hexdigest(),
        "load_seconds": load_seconds,
        "crop": data.metadata["p08_geometry_crop"],
    }


def _bin(position: int) -> str:
    if 1 <= position <= 15:
        return "1-15"
    if 16 <= position <= 39:
        return "16-39"
    if 40 <= position <= 79:
        return "40-79"
    if 80 <= position <= 127:
        return "80-127"
    raise P08FitError(f"invalid post-BOS position: {position}")


def _cohort_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [{"record_index": int(row["record_index"]), "record_id": str(row["record_id"]), "position": int(row["position"]), "bin": str(row["bin"])} for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cohort_artifact_payload(
    *,
    seed: int,
    arm_id: str,
    transition_step: int,
    transition_state_sha256: str,
    cohorts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the small, truth-free transition cohort binding.

    The rows contain only public record identities and positions. Saving this
    before the first full-path update makes the staged comparison restartable
    and prevents a later arm from silently selecting a different error set.
    """

    payload: dict[str, Any] = {
        "schema": "token-reconstruction.trr-p08-transition-cohorts.v1",
        "task_id": TASK_ID,
        "source_seed": int(seed),
        "source_arm": str(arm_id),
        "transition_step": int(transition_step),
        "transition_state_sha256": str(transition_state_sha256),
        "selection": {
            "rule": "direct-affine top-1 errors at the staged affine transition, sorted by record index then position",
            "truth_used_only_for_error_selection": True,
            "rows_expose_no_truth_or_predictions": True,
        },
        "cohorts": {},
    }
    for split in ("validation", "fit"):
        cohort = cohorts[split]
        payload["cohorts"][split] = {
            key: value for key, value in cohort.items() if key != "rows"
        }
        payload["cohorts"][split]["rows"] = [
            {
                "record_index": int(row["record_index"]),
                "record_id": str(row["record_id"]),
                "position": int(row["position"]),
                "bin": str(row["bin"]),
            }
            for row in cohort["rows"]
        ]
    return payload


def _row_predictions(
    model: VisibilityAffineAttentionDecoder,
    data: PublicJointData,
    embedding: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    device: torch.device,
    direct_only: bool,
    guard_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    if split not in ("fit", "validation"):
        raise P08FitError("cohort split must be fit or validation")
    observations = data.fit_observations if split == "fit" else data.validation_observations
    valid_mask = data.fit_valid_mask if split == "fit" else data.validation_valid_mask
    truth = data.fit_truth if split == "fit" else data.validation_truth
    model = model.to(device)
    model.eval()
    if guard_callback is not None:
        guard_callback(f"row_predictions:{split}:before_embedding_validation")
    model.validate_embedding_table(embedding)
    if guard_callback is not None:
        guard_callback(f"row_predictions:{split}:after_embedding_validation")
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["record_index"]), []).append(row)
    result: list[dict[str, Any]] = []
    path_kind = "direct" if direct_only else "full"
    with torch.inference_mode():
        records = sorted(grouped)
        for start in range(0, len(records), RECORD_BATCH_SIZE):
            stop = min(start + RECORD_BATCH_SIZE, len(records))
            if guard_callback is not None:
                guard_callback(f"row_predictions:{split}:{path_kind}:before_batch:{start}:{stop}")
            selected = records[start:stop]
            local = {record: index for index, record in enumerate(selected)}
            record_indices = torch.tensor(selected, dtype=torch.long)
            activation = observations.index_select(0, record_indices).to(device=device, dtype=torch.float32)
            mask = valid_mask.index_select(0, record_indices).to(device=device, dtype=torch.bool)
            if direct_only:
                hidden = F.normalize(model.direct_pre_normalized_hidden(activation, mask), dim=-1)
            else:
                hidden = model.projected_hidden(activation, mask)
            row_entries = [
                (row, local[int(row["record_index"])], int(row["record_index"]), int(row["position"]))
                for record in selected
                for row in grouped[record]
            ]
            for row_start in range(0, len(row_entries), POSITION_BUDGET):
                if guard_callback is not None:
                    guard_callback(f"row_predictions:{split}:{path_kind}:before_chunk:{start}:{row_start}")
                row_chunk = row_entries[row_start : row_start + POSITION_BUDGET]
                local_slots = torch.tensor([entry[1] for entry in row_chunk], device=device, dtype=torch.long)
                original_slots = torch.tensor([entry[2] for entry in row_chunk], device=device, dtype=torch.long)
                position_slots = torch.tensor([entry[3] for entry in row_chunk], device=device, dtype=torch.long)
                logits = model.logits_from_rows(hidden, local_slots, position_slots, embedding.to(device=device, dtype=torch.float32))
                predictions = logits.argmax(dim=-1).detach().cpu().tolist()
                targets = truth[original_slots.cpu(), position_slots.cpu()].tolist()
                for entry, prediction, target in zip(row_chunk, predictions, targets):
                    row = entry[0]
                    result.append({**dict(row), "prediction": int(prediction), "correct": int(prediction) == int(target)})
                if guard_callback is not None:
                    guard_callback(f"row_predictions:{split}:{path_kind}:after_chunk:{start}:{row_start}")
            if guard_callback is not None:
                guard_callback(f"row_predictions:{split}:{path_kind}:after_batch:{start}:{stop}")
    return result


def select_transition_error_cohort(
    model: VisibilityAffineAttentionDecoder,
    data: PublicJointData,
    embedding: torch.Tensor,
    *,
    split: str,
    device: torch.device,
    per_bin: int = COHORT_PER_BIN,
    guard_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Freeze transition errors and report the full direct-path denominator.

    Validation is the public development cohort and therefore retains every
    transition error.  The fit cohort is a bounded, deterministic per-bin
    sample used only for a compact correction diagnostic.  Both reports carry
    the full valid-position and error counts so a capped fit sample cannot be
    mistaken for a complete accuracy estimate.
    """

    if split not in ("fit", "validation"):
        raise P08FitError("transition cohort split must be fit or validation")
    observations = data.fit_observations if split == "fit" else data.validation_observations
    valid_mask = data.fit_valid_mask if split == "fit" else data.validation_valid_mask
    truth = data.fit_truth if split == "fit" else data.validation_truth
    ids = data.fit_record_ids if split == "fit" else data.validation_record_ids
    candidates: list[dict[str, Any]] = []
    valid_post_bos_count = 0
    with torch.inference_mode():
        model.eval()
        model = model.to(device)
        if guard_callback is not None:
            guard_callback(f"transition_cohort:{split}:before_embedding_validation")
        model.validate_embedding_table(embedding)
        if guard_callback is not None:
            guard_callback(f"transition_cohort:{split}:after_embedding_validation")
        runtime_embedding = embedding.to(device=device, dtype=torch.float32)
        for start in range(0, int(observations.shape[0]), RECORD_BATCH_SIZE):
            stop = min(start + RECORD_BATCH_SIZE, int(observations.shape[0]))
            if guard_callback is not None:
                guard_callback(f"transition_cohort:{split}:before_batch:{start}:{stop}")
            activation = observations[start:stop].to(device=device, dtype=torch.float32)
            mask = valid_mask[start:stop].to(device=device, dtype=torch.bool)
            hidden = F.normalize(model.direct_pre_normalized_hidden(activation, mask), dim=-1)
            indices = torch.nonzero(mask, as_tuple=False)
            indices = indices[indices[:, 1] > 0]
            valid_post_bos_count += int(indices.shape[0])
            for chunk_index, chunk in enumerate(indices.split(POSITION_BUDGET)):
                if guard_callback is not None:
                    guard_callback(f"transition_cohort:{split}:before_chunk:{start}:{chunk_index}")
                logits = model.logits_from_rows(hidden, chunk[:, 0], chunk[:, 1], runtime_embedding)
                prediction = logits.argmax(dim=-1).detach().cpu()
                target = truth[start + chunk[:, 0].cpu(), chunk[:, 1].cpu()]
                for index, wrong in enumerate(prediction.ne(target)):
                    if bool(wrong):
                        record_index = start + int(chunk[index, 0])
                        position = int(chunk[index, 1])
                        candidates.append({"record_index": record_index, "record_id": ids[record_index], "position": position, "bin": _bin(position)})
                if guard_callback is not None:
                    guard_callback(f"transition_cohort:{split}:after_chunk:{start}:{chunk_index}")
            if guard_callback is not None:
                guard_callback(f"transition_cohort:{split}:after_batch:{start}:{stop}")
    candidates.sort(key=lambda row: (int(row["record_index"]), int(row["position"])))
    error_count = len(candidates)
    if split == "validation":
        selected = list(candidates)
        shortfall = {name: 0 for name in POSITION_BINS}
        selection_mode = "all_transition_errors"
    else:
        selected = []
        shortfall = {}
        for name in POSITION_BINS:
            rows = [row for row in candidates if row["bin"] == name]
            selected.extend(rows[: int(per_bin)])
            shortfall[name] = max(0, int(per_bin) - len(rows))
        selected.sort(key=lambda row: (int(row["record_index"]), int(row["position"])))
        selection_mode = "fit_error_sample_per_position_bin"
    return {
        "split": split,
        "selection_mode": selection_mode,
        "requested_per_bin": None if split == "validation" else int(per_bin),
        "valid_post_bos_count": int(valid_post_bos_count),
        "transition_error_count": int(error_count),
        "transition_token_accuracy": (1.0 - (error_count / valid_post_bos_count)) if valid_post_bos_count else None,
        "count": len(selected),
        "shortfall_by_bin": shortfall,
        "complete": split == "validation" or all(value == 0 for value in shortfall.values()),
        "rows": selected,
        "cohort_sha256": _cohort_digest(selected),
    }


def correction_summary(affine_rows: Sequence[Mapping[str, Any]], full_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare final full and final direct-affine predictions on one frozen cohort."""

    affine = {(int(row["record_index"]), int(row["position"])): bool(row["correct"]) for row in affine_rows}
    full = {(int(row["record_index"]), int(row["position"])): bool(row["correct"]) for row in full_rows}
    if set(affine) != set(full):
        raise P08FitError("affine/full correction cohorts differ")
    both_correct = sum(affine[key] and full[key] for key in affine)
    both_wrong = sum(not affine[key] and not full[key] for key in affine)
    correction_gain = sum((not affine[key]) and full[key] for key in affine)
    correction_regression = sum(affine[key] and (not full[key]) for key in affine)
    return {"rows": len(affine), "affine_correct": sum(affine.values()), "full_correct": sum(full.values()), "both_correct": both_correct, "both_wrong": both_wrong, "correction_gain": correction_gain, "correction_regression": correction_regression}


def _validation_steps(steps: int) -> tuple[int, ...]:
    return tuple(range(0, int(steps) + 1, VALIDATION_EVERY))


def _phase_for_update(spec: ArmSpec, step_index: int) -> str:
    if spec.staged and int(step_index) < STAGED_AFFINE_STEPS:
        return "affine_only"
    return "full"


def _manifest_geometry(path: Path) -> dict[str, Any]:
    payload = _json_read(path, label="P08 fit manifest")
    resources = payload.get("resources")
    if not isinstance(resources, Mapping):
        raise P08FitError("fit manifest has no resources")
    required = ("embedding_table", "fit_observations", "fit_truth", "fit_valid_mask", "validation_observations", "validation_truth", "validation_valid_mask")
    for name in required:
        resource = resources.get(name)
        if not isinstance(resource, Mapping) or not isinstance(resource.get("shape"), list):
            raise P08FitError(f"fit manifest resource shape missing: {name}")
    return {name: resources[name]["shape"] for name in required}


def _preflight_geometry() -> dict[str, Any]:
    embedding = VOCABULARY_SIZE * HIDDEN_SIZE * 4
    activation = RECORD_BATCH_SIZE * SEQUENCE_LENGTH * HIDDEN_SIZE * 4
    scores = RECORD_BATCH_SIZE * SEQUENCE_LENGTH * SEQUENCE_LENGTH * 4
    selected_logits = POSITION_BUDGET * VOCABULARY_SIZE * 4
    parameters = (HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + 1 + 3 * (HIDDEN_SIZE * CONTEXT_WIDTH + CONTEXT_WIDTH) + CONTEXT_WIDTH * HIDDEN_SIZE + HIDDEN_SIZE) * 4
    adam = 2 * parameters
    gradients = parameters
    workspace = activation * 5 + scores + 2 * selected_logits
    training_peak = embedding + parameters + adam + gradients + workspace
    return {"geometry": {"hidden_size": HIDDEN_SIZE, "vocabulary_size": VOCABULARY_SIZE, "sequence_length": SEQUENCE_LENGTH, "record_batch_size": RECORD_BATCH_SIZE, "position_budget": POSITION_BUDGET, "context_width": CONTEXT_WIDTH}, "bytes": {"embedding_tensor_fp32": embedding, "training_peak_envelope": training_peak, "conservative_envelope": training_peak + math.ceil(training_peak * 0.5)}, "gib": {"training_peak_envelope": training_peak / 1024**3, "conservative_envelope": (training_peak + math.ceil(training_peak * 0.5)) / 1024**3}}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise P08FitError(f"preflight output root is create-only: {output_root}")
    fit_manifest = Path(args.fit_manifest).expanduser().resolve()
    shapes = _manifest_geometry(fit_manifest)
    if shapes["embedding_table"] != [VOCABULARY_SIZE, HIDDEN_SIZE] or shapes["fit_observations"][1:] != [SOURCE_SEQUENCE_LENGTH, HIDDEN_SIZE] or shapes["validation_observations"][1:] != [SOURCE_SEQUENCE_LENGTH, HIDDEN_SIZE]:
        raise P08FitError("public manifest does not bind the expected H192 source geometry")
    receipt = {"schema": PREFLIGHT_SCHEMA, "task_id": TASK_ID, "status": "SOURCE_ONLY_PREFLIGHT_PASS", "created_utc": _utc_now(), "source_commit": _git_commit(Path(args.repository_root).expanduser().resolve()), "command": list(sys.argv), "environment": _environment(), "fit_manifest": _file_record(fit_manifest), "manifest_shapes": shapes, "geometry": _preflight_geometry(), "standard_initialization": {"W": "identity", "b": "zeros", "s": 3.0, "sha256": standard_initialization_sha256()}, "schedule": {"steps": TOTAL_STEPS, "record_batch_size": RECORD_BATCH_SIZE, "position_budget": POSITION_BUDGET, "seeds": list(SEEDS), "staged_affine_steps": STAGED_AFFINE_STEPS}, "resource_policy": {"minimum_free_gpu_gib": float(args.minimum_free_gib), "maximum_gpu_reserved_gib": float(args.maximum_gpu_reserved_gib), "maximum_host_rss_gib": float(args.maximum_host_rss_gib), "minimum_host_available_gib": float(args.minimum_host_available_gib), "max_seconds": float(args.max_seconds)}, "execution_hold": "no public tensors, GPU, truth, or fit updates opened by preflight"}
    output_root.mkdir(parents=True)
    _json_write_create(output_root / "preflight.json", receipt)
    return receipt


def _validate_preflight(path: Path) -> Mapping[str, Any]:
    receipt = _json_read(path, label="P08 preflight receipt")
    if receipt.get("schema") != PREFLIGHT_SCHEMA or receipt.get("status") != "SOURCE_ONLY_PREFLIGHT_PASS":
        raise P08FitError("P08 preflight receipt is not PASS")
    if receipt.get("standard_initialization", {}).get("sha256") != standard_initialization_sha256():
        raise P08FitError("P08 standard initialization binding changed")
    if receipt.get("schedule", {}).get("steps") != TOTAL_STEPS or receipt.get("schedule", {}).get("staged_affine_steps") != STAGED_AFFINE_STEPS:
        raise P08FitError("P08 schedule binding changed")
    return receipt


def _train_arm(spec: ArmSpec, seed: int, data: PublicJointData, embedding: torch.Tensor, schedule: PositionSchedule, schedule_record: Mapping[str, Any], *, args: argparse.Namespace, device: torch.device, output_dir: Path, deadline: float, guards: list[dict[str, Any]], shared_cohorts: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise P08FitError(f"arm output directory is unavailable: {output_dir}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_utc = _utc_now()
    started = time.perf_counter()
    _json_write_create(output_dir / "arm_start.json", {
        "schema": "token-reconstruction.trr-p08-arm-start.v1",
        "task_id": TASK_ID,
        "status": "RUNNING",
        "arm_id": spec.arm_id,
        "visibility_method": spec.visibility_method,
        "schedule_kind": spec.schedule_kind,
        "seed": int(seed),
        "started_utc": started_utc,
        "source_commit": _git_commit(Path(args.repository_root).expanduser().resolve()),
        "schedule_sha256": str(schedule_record["schedule_sha256"]),
        "standard_initialization_sha256": standard_initialization_sha256(),
    })
    model = build_arm_model(spec, seed).to(device)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_digest = state_sha256(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
    runtime_embedding = embedding.to(device=device, dtype=torch.float32)
    model.validate_embedding_table(runtime_embedding)
    checkpoints = _validation_steps(TOTAL_STEPS)
    curve: list[dict[str, Any]] = []
    best_metric = -float("inf")
    best_step = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    last_train: dict[str, Any] | None = None
    update_seconds = {"affine_only": 0.0, "full": 0.0}
    validation_seconds = 0.0
    transition_state_sha: str | None = None
    transition_cohort_artifact: dict[str, Any] | None = None
    transition_state_record: dict[str, Any] | None = None

    def diagnostic_guard(stage: str) -> None:
        """Apply the same fail-closed limits inside long diagnostic loops."""

        guards.append(_guard(args, device, stage=f"{spec.arm_id}:{seed}:diagnostic:{stage}", deadline=deadline))

    for step_index in range(TOTAL_STEPS + 1):
        guards.append(_guard(args, device, stage=f"{spec.arm_id}:{seed}:before_{step_index}", deadline=deadline))
        phase = _phase_for_update(spec, step_index)
        set_training_phase(model, phase)
        if step_index == STAGED_AFFINE_STEPS and spec.staged:
            transition_state_sha = state_sha256({name: value.detach().cpu() for name, value in model.state_dict().items()})
            if not shared_cohorts:
                shared_cohorts["validation"] = select_transition_error_cohort(model, data, runtime_embedding, split="validation", device=device, guard_callback=diagnostic_guard)
                shared_cohorts["fit"] = select_transition_error_cohort(model, data, runtime_embedding, split="fit", device=device, guard_callback=diagnostic_guard)
                shared_cohorts["source_seed"] = int(seed)
                shared_cohorts["source_arm"] = spec.arm_id
                shared_cohorts["transition_direct_state_sha"] = state_sha256({name: getattr(model, name).detach().cpu() for name in ("W", "b", "s")})
                for cohort_name in ("validation", "fit"):
                    shared_cohorts[cohort_name]["source_seed"] = int(seed)
                    shared_cohorts[cohort_name]["source_arm"] = spec.arm_id
                    shared_cohorts[cohort_name]["transition_direct_state_sha256"] = shared_cohorts["transition_direct_state_sha"]
            elif spec.staged:
                direct_names = ("W", "b", "s")
                existing = shared_cohorts.get("transition_direct_state_sha")
                current = state_sha256({name: getattr(model, name).detach().cpu() for name in direct_names})
                if existing is not None and current != existing:
                    raise P08FitError("staged visibility arms diverged before attention was enabled")
                shared_cohorts.setdefault("transition_direct_state_sha", current)
            if spec.staged:
                transition_payload = _cohort_artifact_payload(
                    seed=seed,
                    arm_id=str(shared_cohorts["source_arm"]),
                    transition_step=STAGED_AFFINE_STEPS,
                    transition_state_sha256=str(shared_cohorts["transition_direct_state_sha"]),
                    cohorts={"validation": shared_cohorts["validation"], "fit": shared_cohorts["fit"]},
                )
                cohort_path = output_dir / "transition_cohorts.json"
                _json_write_create(cohort_path, transition_payload)
                transition_cohort_artifact = {"path": str(cohort_path), "bytes": cohort_path.stat().st_size, "sha256": file_sha256(cohort_path)}
                transition_state_record = save_visibility_state(
                    output_dir / "transition.safetensors",
                    model,
                    selected_step=STAGED_AFFINE_STEPS,
                    metadata={
                        "task_id": TASK_ID,
                        "p08_method_id": spec.arm_id,
                        "checkpoint_role": "staged_affine_transition_before_full_update",
                        "schedule_kind": spec.schedule_kind,
                        "fit_seed": int(seed),
                        "standard_initialization_sha256": standard_initialization_sha256(),
                        "schedule_sha256": schedule_record["schedule_sha256"],
                        "transition_direct_state_sha256": shared_cohorts["transition_direct_state_sha"],
                        "transition_cohort_sha256": _cohort_digest(shared_cohorts["validation"]["rows"]),
                    },
                )
        if step_index in checkpoints:
            started_validation = time.perf_counter()
            metrics = evaluate_dataset(model, data.validation_observations, data.validation_truth, data.validation_valid_mask, runtime_embedding, data.validation_groups, device=device, record_batch_size=RECORD_BATCH_SIZE, position_budget=POSITION_BUDGET)
            elapsed = time.perf_counter() - started_validation
            validation_seconds += elapsed
            point = {"step": int(step_index), "phase": phase, "learning_rate": float(optimizer.param_groups[0]["lr"]), "train": None if last_train is None else dict(last_train), "validation": metrics, "validation_wall_seconds": elapsed}
            curve.append(point)
            metric = float(metrics["token_accuracy"])
            if metric > best_metric:
                best_metric = metric
                best_step = int(step_index)
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if step_index == TOTAL_STEPS:
            break
        started_update = time.perf_counter()
        last_train = train_step(model, data.fit_observations, data.fit_truth, data.fit_valid_mask, runtime_embedding, schedule, step_index, device=device, optimizer=optimizer, gradient_clip_norm=GRADIENT_CLIP_NORM)
        scheduler.step()
        update_seconds[phase] += time.perf_counter() - started_update
        last_train["step"] = int(step_index + 1)
        last_train["phase"] = phase
        guards.append(_guard(args, device, stage=f"{spec.arm_id}:{seed}:after_{step_index}", deadline=deadline))
    final_step = TOTAL_STEPS
    if not curve or int(curve[-1]["step"]) != final_step:
        raise P08FitError("final validation checkpoint was not recorded")
    final_state_record = save_visibility_state(
        output_dir / "final.safetensors",
        model,
        selected_step=final_step,
        metadata={
            "task_id": TASK_ID,
            "p08_method_id": spec.arm_id,
            "checkpoint_role": "final_last_update",
            "schedule_kind": spec.schedule_kind,
            "fit_seed": int(seed),
            "standard_initialization_sha256": standard_initialization_sha256(),
            "schedule_sha256": schedule_record["schedule_sha256"],
            "selection_metric": "validation_token_accuracy",
            "final_step": final_step,
        },
    )
    selected_model = build_arm_model(spec, seed).to(device)
    selected_model.load_state_dict(best_state, strict=True)
    cohort_metrics: dict[str, Any] = {}
    for split in ("validation", "fit"):
        cohort = shared_cohorts.get(split)
        if cohort is None:
            continue
        rows = cohort["rows"]
        selected_affine_rows = _row_predictions(selected_model, data, runtime_embedding, rows, split=split, device=device, direct_only=True, guard_callback=diagnostic_guard)
        selected_full_rows = _row_predictions(selected_model, data, runtime_embedding, rows, split=split, device=device, direct_only=False, guard_callback=diagnostic_guard)
        final_affine_rows = _row_predictions(model, data, runtime_embedding, rows, split=split, device=device, direct_only=True, guard_callback=diagnostic_guard)
        final_full_rows = _row_predictions(model, data, runtime_embedding, rows, split=split, device=device, direct_only=False, guard_callback=diagnostic_guard)
        cohort_metrics[split] = {
            "cohort": {key: value for key, value in cohort.items() if key != "rows"},
            "selected_checkpoint": {"step": int(best_step), "comparison": correction_summary(selected_affine_rows, selected_full_rows)},
            "final_checkpoint": {"step": TOTAL_STEPS, "comparison": correction_summary(final_affine_rows, final_full_rows)},
        }
    curve_path = output_dir / "learning_curve.json"
    _json_write_create(curve_path, {"schema": "token-reconstruction.trr-p08-learning-curve.v1", "task_id": TASK_ID, "arm_id": spec.arm_id, "visibility_method": spec.visibility_method, "schedule_kind": spec.schedule_kind, "seed": int(seed), "selection_metric": "validation_token_accuracy", "selection_rule": "earliest maximum, validation every 100 updates including step 0", "curve": curve})
    state_record = save_visibility_state(output_dir / "selected.safetensors", selected_model, selected_step=best_step, metadata={"task_id": TASK_ID, "p08_method_id": spec.arm_id, "schedule_kind": spec.schedule_kind, "fit_seed": int(seed), "standard_initialization_sha256": standard_initialization_sha256(), "schedule_sha256": schedule_record["schedule_sha256"], "selection_metric": "validation_token_accuracy", "selection_rule": "earliest maximum, validation every 100 updates including step 0"})
    peak = _peak_snapshot(device)
    finished_utc = _utc_now()
    result = {"arm_id": spec.arm_id, "visibility_method": spec.visibility_method, "schedule_kind": spec.schedule_kind, "seed": int(seed), "status": "PASS", "steps": TOTAL_STEPS, "staged_affine_steps": STAGED_AFFINE_STEPS if spec.staged else 0, "selected_step": best_step, "final_step": final_step, "best_validation_token_accuracy": best_metric, "initial_state_sha256": initial_digest, "transition_state_sha256": transition_state_sha, "schedule_sha256": schedule_record["schedule_sha256"], "started_utc": started_utc, "finished_utc": finished_utc, "learning_curve": {"path": str(curve_path), "bytes": curve_path.stat().st_size, "sha256": file_sha256(curve_path)}, "state": state_record, "final_state": final_state_record, "transition_state": transition_state_record, "transition_cohort_artifact": transition_cohort_artifact, "final_validation_metrics": curve[-1]["validation"], "phase_update_seconds": update_seconds, "validation_seconds": validation_seconds, "arm_wall_seconds": time.perf_counter() - started, "peak_memory": peak, "correction_diagnostics": cohort_metrics, "parameter_count": int(model.parameter_count), "effective_parameter_count": int(model.effective_parameter_count)}
    _json_write_create(output_dir / "arm_receipt.json", {
        "schema": "token-reconstruction.trr-p08-arm-receipt.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "source_commit": _git_commit(Path(args.repository_root).expanduser().resolve()),
        "result": result,
    })
    del selected_model, model, optimizer, scheduler, runtime_embedding
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise P08FitError(f"qualification output root is create-only: {output_root}")
    preflight = _validate_preflight(Path(args.preflight_receipt).expanduser().resolve())
    current_fit_record = _file_record(Path(args.fit_manifest).expanduser().resolve())
    if preflight.get("fit_manifest", {}).get("sha256") != current_fit_record["sha256"]:
        raise P08FitError("qualification fit manifest differs from the preflight binding")
    output_root.mkdir(parents=True)
    _json_write_create(output_root / "run_start.json", {
        "schema": "token-reconstruction.trr-p08-run-start.v1",
        "task_id": TASK_ID,
        "mode": "qualify",
        "status": "RUNNING",
        "started_utc": _utc_now(),
        "source_commit": _git_commit(Path(args.repository_root).expanduser().resolve()),
    })
    device = _device(args.device)
    guards: list[dict[str, Any]] = []
    deadline = time.perf_counter() + float(args.max_seconds)
    data, data_receipt = _load_data(args, device, deadline=deadline, guards=guards)
    schedule = build_position_schedule(data.fit_valid_mask, steps=QUALIFICATION_STEPS, record_batch_size=RECORD_BATCH_SIZE, position_budget=POSITION_BUDGET, seed=SEEDS[0])
    model = build_arm_model(ARM_BY_ID[P08_PAST_ONLY_JOINT], SEEDS[0]).to(device)
    set_training_phase(model, "full")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    embedding = data.embedding_table.to(device=device, dtype=torch.float32)
    model.validate_embedding_table(embedding)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(QUALIFICATION_STEPS):
        guards.append(_guard(args, device, stage=f"qualification:before_{step}", deadline=deadline))
        train_step(model, data.fit_observations, data.fit_truth, data.fit_valid_mask, embedding, schedule, step, device=device, optimizer=optimizer, gradient_clip_norm=GRADIENT_CLIP_NORM)
        guards.append(_guard(args, device, stage=f"qualification:after_{step}", deadline=deadline))
    receipt = {"schema": QUALIFICATION_SCHEMA, "task_id": TASK_ID, "status": "PASS", "created_utc": _utc_now(), "source_commit": _git_commit(Path(args.repository_root).expanduser().resolve()), "command": list(sys.argv), "device": str(device), "data": data_receipt, "geometry": {"fit": list(data.fit_observations.shape), "record_batch_size": RECORD_BATCH_SIZE, "position_budget": POSITION_BUDGET, "updates": QUALIFICATION_STEPS}, "standard_initialization_sha256": standard_initialization_sha256(), "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()), "peak_memory": _peak_snapshot(device), "resource_guards": guards, "states_retained": False}
    receipt["finished_utc"] = _utc_now()
    _json_write_create(output_root / "qualification.json", receipt)
    return receipt


def _validate_qualification(path: Path) -> Mapping[str, Any]:
    receipt = _json_read(path, label="P08 qualification receipt")
    if receipt.get("schema") != QUALIFICATION_SCHEMA or receipt.get("status") != "PASS" or receipt.get("states_retained") is not False:
        raise P08FitError("P08 qualification receipt is not an accepted disposable PASS")
    if receipt.get("standard_initialization_sha256") != standard_initialization_sha256():
        raise P08FitError("qualification standard initialization differs")
    return receipt


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise P08FitError(f"main output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    run_started_utc = _utc_now()
    _json_write_create(output_root / "run_start.json", {
        "schema": "token-reconstruction.trr-p08-run-start.v1",
        "task_id": TASK_ID,
        "mode": "main",
        "status": "RUNNING",
        "started_utc": run_started_utc,
        "source_commit": _git_commit(Path(args.repository_root).expanduser().resolve()),
    })
    preflight = _validate_preflight(Path(args.preflight_receipt).expanduser().resolve())
    current_fit_record = _file_record(Path(args.fit_manifest).expanduser().resolve())
    if preflight.get("fit_manifest", {}).get("sha256") != current_fit_record["sha256"]:
        raise P08FitError("main fit manifest differs from the preflight binding")
    qualification = _validate_qualification(Path(args.qualification_receipt).expanduser().resolve())
    device = _device(args.device)
    guards: list[dict[str, Any]] = []
    deadline = time.perf_counter() + float(args.max_seconds)
    data, data_receipt = _load_data(args, device, deadline=deadline, guards=guards)
    embedding = data.embedding_table
    schedules: dict[str, Any] = {}
    methods: list[dict[str, Any]] = []
    shared_cohorts: dict[str, Any] = {}
    for seed in SEEDS:
        schedule = build_position_schedule(data.fit_valid_mask, steps=TOTAL_STEPS, record_batch_size=RECORD_BATCH_SIZE, position_budget=POSITION_BUDGET, seed=seed)
        seed_root = output_root / f"seed-{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        schedule_record = save_schedule(seed_root / "position_schedule.safetensors", schedule)
        schedules[str(seed)] = {**schedule_record, "metadata": schedule_metadata(schedule)}
        for spec in ARM_SPECS:
            arm_root = seed_root / spec.arm_id
            arm_root.mkdir(parents=True, exist_ok=False)
            methods.append(_train_arm(spec, seed, data, embedding, schedule, schedules[str(seed)], args=args, device=device, output_dir=arm_root, deadline=deadline, guards=guards, shared_cohorts=shared_cohorts))
        # A cohort is tied to the seed and must not leak into the next seed.
        shared_cohorts = {}
    receipt = {"schema": SCHEMA, "task_id": TASK_ID, "status": "PASS", "created_utc": _utc_now(), "started_utc": run_started_utc, "finished_utc": _utc_now(), "source_commit": _git_commit(Path(args.repository_root).expanduser().resolve()), "command": list(sys.argv), "environment": _environment(), "device": str(device), "data": data_receipt, "geometry": {"fit": list(data.fit_observations.shape), "validation": list(data.validation_observations.shape), "record_batch_size": RECORD_BATCH_SIZE, "position_budget": POSITION_BUDGET, "steps": TOTAL_STEPS, "validation_every": VALIDATION_EVERY}, "initialization": {"W": "identity", "b": "zeros", "s": 3.0, "sha256": standard_initialization_sha256()}, "seeds": list(SEEDS), "arms": [spec.__dict__ for spec in ARM_SPECS], "schedules": schedules, "methods": methods, "qualification_receipt": {"path": str(Path(args.qualification_receipt).expanduser().resolve()), "sha256": file_sha256(Path(args.qualification_receipt).expanduser().resolve()), "status": qualification["status"]}, "resource_policy": {"minimum_free_gpu_gib": float(args.minimum_free_gib), "maximum_gpu_reserved_gib": float(args.maximum_gpu_reserved_gib), "maximum_host_rss_gib": float(args.maximum_host_rss_gib), "minimum_host_available_gib": float(args.minimum_host_available_gib), "max_seconds": float(args.max_seconds)}, "resource_guards": guards, "runtime_components": {"source_token_access": False, "target_truth_access": False, "guessed_token_feedback": False, "candidate_simulations": 0, "a2_student": False, "supervision": "public same-position full-vocabulary CE"}}
    _json_write_create(output_root / "main_fit_receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "qualify", "main"), required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--embedding-path", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--qualification-receipt", type=Path)
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
    if args.mode in ("qualify", "main") and args.preflight_receipt is None:
        raise P08FitError(f"{args.mode} requires --preflight-receipt")
    if args.mode == "main" and args.qualification_receipt is None:
        raise P08FitError("main requires --qualification-receipt")
    if args.mode in ("preflight", "qualify") and args.qualification_receipt is not None:
        raise P08FitError("qualification receipt is valid only for main")
    for name in ("minimum_free_gib", "maximum_gpu_reserved_gib", "maximum_host_rss_gib", "minimum_host_available_gib", "max_seconds"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise P08FitError(f"{name} must be finite and positive")
    if STAGED_AFFINE_STEPS <= 0 or STAGED_AFFINE_STEPS >= TOTAL_STEPS:
        raise P08FitError("staged phase must be a strict interior split")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    try:
        _validate_args(args)
        _set_threads(args)
        if args.mode == "preflight":
            run_preflight(args)
        elif args.mode == "qualify":
            run_qualification(args)
        else:
            run_main(args)
        return 0
    except Exception as exc:
        try:
            if output_root.exists() and output_root.is_dir() and not (output_root / "failure.json").exists():
                _json_write_create(output_root / "failure.json", {"schema": FAILURE_SCHEMA, "task_id": TASK_ID, "status": "FAIL", "created_utc": _utc_now(), "command": list(sys.argv), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "source_commit": _git_commit(Path(args.repository_root).expanduser().resolve()), "environment": _environment()})
        except Exception:
            pass
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
