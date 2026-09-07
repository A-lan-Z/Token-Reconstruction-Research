#!/usr/bin/env python3
"""TRR-0009 supported-token readout continuation trainer.

The runner keeps the published TRR-0007 residual MLP512 state and schedule
fixed while comparing an unchanged anchor, a continued decoder, and the same
continued decoder with a bounded supported-row gain/bias readout.  It only
consumes public H/labels and the public normalized embedding table.  All
outputs are create-only and every expensive mode has a bounded preflight or
qualification path.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, asdict
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

# Keep direct ``python3 scripts/trr0009_train.py`` invocation reproducible.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from token_reconstruction.trr0005_joint_decoder import (
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_POSITION_BUDGET,
    DEFAULT_RECORD_BATCH_SIZE,
    DEFAULT_STEPS,
    DEFAULT_VALIDATION_EVERY,
    PositionSchedule,
    PublicJointData,
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
    DEFAULT_BOTTLENECK_SIZE,
    RESIDUAL_MLP_METHOD_ID,
    ResidualMLPPositionwiseDecoder,
    load_positionwise_model_state,
    save_positionwise_state,
)

from trr0009_model import (
    DEFAULT_ANCHOR_STRENGTH,
    METHOD_ID as ADAPTABLE_METHOD_ID,
    SupportedTokenReadout,
    build_adaptable_from_base,
    build_support_from_counts,
    load_published_residual_state,
    save_supported_readout_state,
    support_digest,
)


TASK_ID = "TRR-0009"
RUN_SCHEMA = "token-reconstruction.trr0009-training-run.v1"
PREFLIGHT_SCHEMA = "token-reconstruction.trr0009-training-preflight.v1"
CHALLENGE_SCHEMA = "token-reconstruction.trr0009-selected-start-challenge.v1"
QUALIFICATION_SCHEMA = "token-reconstruction.trr0009-largest-cell-qualification.v1"
EXPECTED_SCHEDULE_DIGEST = "5a2daa0087b1877bb5f9be4bd59ef201a4fa6478fcd5b16a1b88808963eab472"
EXPECTED_SCHEDULE_FILE_SHA = "dcf439e2221bf34cd526a2b3fa0e5ebba0e6393e6ee2d4525d3b7edf6e9eea94"
EXPECTED_STEPS = 3000
EXPECTED_SEED = 4005
EXPECTED_RECORD_BATCH_SIZE = 8
EXPECTED_POSITION_BUDGET = 512
EXPECTED_SEQUENCE_LENGTH = 192
EXPECTED_CONTEXT_WIDTH = 128
EXPECTED_HIDDEN_SIZE = 2048
EXPECTED_VOCABULARY_SIZE = 128256
CHALLENGE_SEED = 11012
CHALLENGE_CAP = 2048
STARTING_STATE_SHA256 = "2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8"
STARTING_FIT_ROWS = 124371
DEFAULT_LEARNING_RATE = 2.0e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_VALIDATION_EVERY = 100
DEFAULT_MAX_SECONDS = 7200.0
DEFAULT_MAX_RESERVED_GIB = 6.0
DEFAULT_MIN_FREE_GIB = 8.0
DEFAULT_MAX_RSS_GIB = 16.0
DEFAULT_MIN_HOST_GIB = 10.0
NUMERICAL_SETTINGS = {
    "activation_dtype": "torch.float32",
    "logit_accumulation_dtype": "torch.float32",
    "cuda_matmul_allow_tf32": False,
    "cuda_cudnn_allow_tf32": True,
    "cpu_intraop_threads": 8,
    "cpu_interop_threads": 32,
}


class TRR0009TrainError(RuntimeError):
    """Raised when a TRR-0009 training contract cannot be satisfied."""


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = EXPECTED_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    readout_learning_rate: float = 1.0e-3
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM
    record_batch_size: int = EXPECTED_RECORD_BATCH_SIZE
    position_budget: int = EXPECTED_POSITION_BUDGET
    validation_every: int = DEFAULT_VALIDATION_EVERY
    seed: int = EXPECTED_SEED
    challenge_seed: int = CHALLENGE_SEED
    challenge_cap: int = CHALLENGE_CAP
    max_seconds: float = DEFAULT_MAX_SECONDS
    maximum_gpu_reserved_gib: float = DEFAULT_MAX_RESERVED_GIB
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB
    maximum_host_rss_gib: float = DEFAULT_MAX_RSS_GIB
    minimum_host_available_gib: float = DEFAULT_MIN_HOST_GIB


@dataclass(frozen=True)
class ResourceGuard:
    maximum_gpu_reserved_bytes: int
    minimum_gpu_free_bytes: int
    maximum_host_rss_bytes: int
    minimum_host_available_bytes: int

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "ResourceGuard":
        gib = float(1024**3)
        return cls(
            maximum_gpu_reserved_bytes=int(config.maximum_gpu_reserved_gib * gib),
            minimum_gpu_free_bytes=int(config.minimum_free_gib * gib),
            maximum_host_rss_bytes=int(config.maximum_host_rss_gib * gib),
            minimum_host_available_bytes=int(config.minimum_host_available_gib * gib),
        )


@dataclass(frozen=True)
class Paths:
    repository_root: Path
    fit_manifest: Path
    validation_manifest: Path
    embedding_path: Path
    starting_state: Path
    schedule_path: Path
    output_root: Path


def _configure_numerics() -> dict[str, Any]:
    """Apply and return the inherited TRR-0007 numerical settings."""

    torch.set_default_dtype(torch.float32)
    try:
        torch.set_num_threads(int(NUMERICAL_SETTINGS["cpu_intraop_threads"]))
        torch.set_num_interop_threads(int(NUMERICAL_SETTINGS["cpu_interop_threads"]))
    except RuntimeError as exc:
        raise TRR0009TrainError("CPU numerical settings must be configured before parallel work") from exc
    torch.backends.cuda.matmul.allow_tf32 = bool(NUMERICAL_SETTINGS["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(NUMERICAL_SETTINGS["cuda_cudnn_allow_tf32"])
    actual = dict(NUMERICAL_SETTINGS)
    actual.update({
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    })
    return actual


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_binding(root: Path, numerical_settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    files = {}
    for path in (Path(__file__).resolve(), Path(__file__).with_name("trr0009_model.py").resolve()):
        files[str(path)] = _file_record(path, description="TRR-0009 source")
    return {
        "task_id": TASK_ID,
        "code_commit": _git_commit(root),
        "source_files": files,
        "command": {"argv": list(sys.argv), "cwd": str(Path.cwd()), "python": sys.executable, "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "numerical_settings": dict(numerical_settings or NUMERICAL_SETTINGS),
    }


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += int(value.numel()) * int(value.element_size())
    return total


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TRR0009TrainError(f"{description} must be a regular file: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def _write_create_only(path: Path, value: Mapping[str, Any], *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TRR0009TrainError(f"{description} is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise TRR0009TrainError(f"{description} is create-only and already exists: {path}") from exc
    return _file_record(path, description=description)


def _default_paths(repository_root: Path) -> Paths:
    root = repository_root.expanduser().resolve()
    project_root = root.parents[1] if root.name == "TRR-0009" and root.parent.name == ".worktrees" else root
    trr7_root = root.parent / "TRR-0007"
    return Paths(
        repository_root=root,
        fit_manifest=trr7_root / "experiments/TRR-0007/support/broader_capture_v2/enriched_manifest.json",
        validation_manifest=trr7_root / "experiments/TRR-0007/support/broader_capture_v2/original_manifest.json",
        embedding_path=project_root / "outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors",
        starting_state=trr7_root / "experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_residual_mlp512/selected.safetensors",
        schedule_path=trr7_root / "experiments/TRR-0007/improved_fit_v1/improved_public_bank/schedule.safetensors",
        output_root=root / "experiments/TRR-0009/training/run_v1",
    )


def _resource_paths(args: argparse.Namespace) -> Paths:
    defaults = _default_paths(Path(args.repository_root))
    values: dict[str, Path] = {}
    for name in ("fit_manifest", "validation_manifest", "embedding_path", "starting_state", "schedule_path", "output_root"):
        supplied = getattr(args, name)
        values[name] = Path(supplied).expanduser() if supplied else getattr(defaults, name)
    values["repository_root"] = Path(args.repository_root).expanduser()
    paths = Paths(**values)
    paths = Paths(
        repository_root=paths.repository_root.resolve(),
        fit_manifest=paths.fit_manifest.resolve(),
        validation_manifest=paths.validation_manifest.resolve(),
        embedding_path=paths.embedding_path.resolve(),
        starting_state=paths.starting_state.resolve(),
        schedule_path=paths.schedule_path.resolve(),
        output_root=paths.output_root.resolve(),
    )
    task_root = (paths.repository_root / "experiments/TRR-0009/training").resolve()
    try:
        paths.output_root.relative_to(task_root)
    except ValueError as exc:
        raise TRR0009TrainError(f"output root must be below {task_root}: {paths.output_root}") from exc
    if paths.output_root == task_root:
        raise TRR0009TrainError("output root must be a run-specific child of the task training root")
    return paths


def _validate_config(config: TrainingConfig, *, qualification: bool = False) -> None:
    if config.steps <= 0 or config.record_batch_size <= 0 or config.position_budget <= 0:
        raise TRR0009TrainError("training dimensions must be positive")
    if config.learning_rate <= 0 or config.readout_learning_rate <= 0 or config.weight_decay < 0 or config.gradient_clip_norm <= 0:
        raise TRR0009TrainError("optimizer configuration is invalid")
    if config.validation_every != DEFAULT_VALIDATION_EVERY or config.seed != EXPECTED_SEED:
        raise TRR0009TrainError("frozen schedule seed or validation interval changed")
    if abs(config.learning_rate - DEFAULT_LEARNING_RATE) > 1e-15 or abs(config.readout_learning_rate - 1.0e-3) > 1e-15:
        raise TRR0009TrainError("frozen decoder/readout learning rates changed")
    if config.weight_decay != DEFAULT_WEIGHT_DECAY or config.gradient_clip_norm != DEFAULT_GRADIENT_CLIP_NORM:
        raise TRR0009TrainError("frozen optimizer configuration changed")
    if config.challenge_seed != CHALLENGE_SEED or config.challenge_cap != CHALLENGE_CAP:
        raise TRR0009TrainError("frozen challenge configuration changed")
    if not qualification and config.steps != EXPECTED_STEPS:
        raise TRR0009TrainError("full TRR-0009 runs require exactly 3000 steps")
    if config.record_batch_size != EXPECTED_RECORD_BATCH_SIZE or config.position_budget != EXPECTED_POSITION_BUDGET:
        raise TRR0009TrainError("record batch or position budget differs from frozen geometry")
    if config.max_seconds <= 0:
        raise TRR0009TrainError("max_seconds must be positive")


def _host_available_bytes() -> int | None:
    """Return Linux MemAvailable, including reclaimable cache.

    SC_AVPHYS_PAGES reports immediately free pages and can understate the
    memory that the OS can reclaim safely for this bounded guard.
    """
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                return int(fields[1]) * 1024
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def _host_free_bytes() -> int | None:
    """Return immediately free pages for diagnostic context only."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        free_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    return page_size * free_pages


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    host_available = _host_available_bytes()
    host_free = _host_free_bytes()
    snapshot: dict[str, Any] = {
        "host_peak_rss_bytes": peak_rss,
        "host_peak_rss_gib": peak_rss / float(1024**3),
        "host_available_bytes": host_available,
        "host_available_gib": None if host_available is None else host_available / float(1024**3),
        "host_free_bytes": host_free,
        "host_free_gib": None if host_free is None else host_free / float(1024**3),
        "host_available_source": "/proc/meminfo:MemAvailable",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        snapshot.update({
            "cuda_free_bytes": int(free),
            "cuda_total_bytes": int(total),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        })
    else:
        snapshot.update({"cuda_free_bytes": None, "cuda_total_bytes": None, "cuda_reserved_bytes": 0, "cuda_allocated_bytes": 0})
    return snapshot


def _enforce_resource_guard(snapshot: Mapping[str, Any], guard: ResourceGuard, *, stage: str, device: torch.device) -> None:
    reserved = int(snapshot.get("cuda_reserved_bytes") or 0)
    if device.type == "cuda" and reserved > guard.maximum_gpu_reserved_bytes:
        raise TRR0009TrainError(f"resource guard exceeded at {stage}: CUDA reserved bytes {reserved}")
    free = snapshot.get("cuda_free_bytes")
    if device.type == "cuda" and free is not None and int(free) < guard.minimum_gpu_free_bytes:
        raise TRR0009TrainError(f"resource guard exceeded at {stage}: CUDA free bytes {free}")
    peak_rss = int(snapshot.get("host_peak_rss_bytes") or 0)
    if peak_rss > guard.maximum_host_rss_bytes:
        raise TRR0009TrainError(f"resource guard exceeded at {stage}: peak RSS bytes {peak_rss}")
    host_available = snapshot.get("host_available_bytes")
    if host_available is None:
        raise TRR0009TrainError(f"resource guard telemetry unavailable at {stage}: host available bytes missing")
    if int(host_available) < guard.minimum_host_available_bytes:
        raise TRR0009TrainError(f"resource guard exceeded at {stage}: host available bytes {host_available}")


def resource_estimate(*, records: int = EXPECTED_RECORD_BATCH_SIZE, sequence: int = EXPECTED_SEQUENCE_LENGTH, hidden: int = EXPECTED_HIDDEN_SIZE, vocabulary: int = EXPECTED_VOCABULARY_SIZE, position_budget: int = EXPECTED_POSITION_BUDGET) -> dict[str, Any]:
    hidden_bytes = records * sequence * hidden * 4
    full_logits_bytes = position_budget * vocabulary * 4
    full_b1_logits_bytes = sequence * vocabulary * 4
    embedding_bytes = vocabulary * hidden * 4
    adaptable_parameters = 2 * 17126
    return {
        "largest_activation_shape": [records, sequence, hidden],
        "largest_draw_shape": [position_budget],
        "hidden_materialized_bytes": hidden_bytes,
        "full_logits_bytes_per_draw": full_logits_bytes,
        "full_logits_mib_per_draw": full_logits_bytes / float(1024**2),
        "b1_full_logits_bytes_for_zero_equivalence": full_b1_logits_bytes,
        "b1_two_full_logits_peak_bytes_if_compared_simultaneously": 2 * full_b1_logits_bytes,
        "embedding_bytes": embedding_bytes,
        "embedding_gib": embedding_bytes / float(1024**3),
        "adaptable_parameter_count": adaptable_parameters,
        "adaptable_parameter_fp32_bytes": adaptable_parameters * 4,
        "adaptable_parameter_with_grad_and_adam_bytes": adaptable_parameters * 4 * 4,
        "historical_trr0007_peak_reserved_bytes": 3599152128,
        "historical_trr0007_peak_reserved_gib": 3599152128 / float(1024**3),
        "qualification_full_output_fixture": "B=1 only; largest B=8 qualification uses selected-row path",
    }


def _manifest_resource_bindings(manifest_path: Path) -> dict[str, Any]:
    """Hash every declared public resource, including record sidecars.

    The inherited loader checks tensor shape and finiteness but does not bind
    resource bytes.  TRR-0009 verifies the manifest declarations itself before
    fitting and records the exact paths actually consumed.
    """

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TRR0009TrainError(f"cannot read public manifest: {manifest_path}") from exc
    resources = payload.get("resources")
    if not isinstance(resources, Mapping):
        raise TRR0009TrainError(f"public manifest has no resources mapping: {manifest_path}")
    bindings: dict[str, Any] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for name, spec in resources.items():
        if not isinstance(spec, Mapping) or not isinstance(spec.get("path"), str):
            raise TRR0009TrainError(f"manifest resource is malformed: {manifest_path}:{name}")
        raw = Path(str(spec["path"]))
        path = (raw if raw.is_absolute() else manifest_path.parent / raw).resolve()
        record = _file_record(path, description=f"manifest resource {name}")
        declared_sha = spec.get("sha256")
        if not isinstance(declared_sha, str) or record["sha256"] != declared_sha:
            raise TRR0009TrainError(f"manifest resource hash mismatch: {manifest_path}:{name}")
        if "bytes" in spec and int(spec["bytes"]) != record["bytes"]:
            raise TRR0009TrainError(f"manifest resource byte count mismatch: {manifest_path}:{name}")
        entry = {"name": str(name), "declared": dict(spec), "actual": record}
        if str(path) in by_path:
            prior = by_path[str(path)]
            if prior["actual"]["sha256"] != record["sha256"]:
                raise TRR0009TrainError(f"deduplicated resource has conflicting hashes: {path}")
            prior.setdefault("aliases", []).append(str(name))
        else:
            by_path[str(path)] = entry
            bindings[str(name)] = entry
    return {"manifest": _file_record(manifest_path, description="public manifest"), "resources": list(bindings.values()), "unique_resource_files": len(by_path)}


def _verify_resource_bindings_unchanged(bindings: Mapping[str, Any]) -> dict[str, Any]:
    after: list[dict[str, Any]] = []
    for entry in bindings.get("resources", ()):
        actual = entry.get("actual", {})
        path = Path(str(actual.get("path", "")))
        record = _file_record(path, description="bound public resource recheck")
        if record["sha256"] != actual.get("sha256") or record["bytes"] != actual.get("bytes"):
            raise TRR0009TrainError(f"bound public resource changed after load: {path}")
        after.append(record)
    return {"resources": after, "checked_after_loader": True}


def _validate_embedding_asset(data: PublicJointData, embedding_path: Path) -> dict[str, Any]:
    embedding = data.embedding_table
    if tuple(embedding.shape) != (EXPECTED_VOCABULARY_SIZE, EXPECTED_HIDDEN_SIZE):
        raise TRR0009TrainError(f"embedding geometry differs: {tuple(embedding.shape)}")
    if not embedding.dtype.is_floating_point or not torch.isfinite(embedding).all().item():
        raise TRR0009TrainError("public embedding table is not finite floating point")
    return {
        "file": _file_record(embedding_path, description="public embedding table"),
        "tensor_shape": list(embedding.shape),
        "tensor_dtype": str(embedding.dtype),
        "tensor_sha256": tensor_sha256(embedding),
        "finite_checked_once": True,
    }


def _validate_data_geometry(data: PublicJointData) -> dict[str, Any]:
    fit_shape = tuple(int(x) for x in data.fit_observations.shape)
    val_shape = tuple(int(x) for x in data.validation_observations.shape)
    if len(fit_shape) != 3 or fit_shape[1:] != (EXPECTED_SEQUENCE_LENGTH, EXPECTED_HIDDEN_SIZE):
        raise TRR0009TrainError(f"fit activation geometry differs: {fit_shape}")
    if len(val_shape) != 3 or val_shape[1:] != (EXPECTED_SEQUENCE_LENGTH, EXPECTED_HIDDEN_SIZE):
        raise TRR0009TrainError(f"validation activation geometry differs: {val_shape}")
    if tuple(data.fit_truth.shape) != fit_shape[:2] or tuple(data.fit_valid_mask.shape) != fit_shape[:2]:
        raise TRR0009TrainError("fit truth/mask geometry differs")
    if tuple(data.validation_truth.shape) != val_shape[:2] or tuple(data.validation_valid_mask.shape) != val_shape[:2]:
        raise TRR0009TrainError("validation truth/mask geometry differs")
    if int(data.fit_observations.shape[0]) != 1200 or int(data.validation_observations.shape[0]) != 48:
        raise TRR0009TrainError("published improved-bank record counts changed")
    if not data.fit_valid_mask[:, 0].all().item() or not data.validation_valid_mask[:, 0].all().item():
        raise TRR0009TrainError("public masks must include BOS")
    return {
        "fit_observations": list(fit_shape),
        "validation_observations": list(val_shape),
        "fit_post_bos_rows": int((data.fit_valid_mask[:, 1:]).sum().item()),
        "validation_post_bos_rows": int((data.validation_valid_mask[:, 1:]).sum().item()),
        "fit_record_ids": len(data.fit_record_ids),
        "validation_record_ids": len(data.validation_record_ids),
    }


def _frequency_counts(data: PublicJointData) -> torch.Tensor:
    labels = data.fit_truth[:, 1:][data.fit_valid_mask[:, 1:].to(dtype=torch.bool)]
    if labels.numel() != STARTING_FIT_ROWS:
        raise TRR0009TrainError(f"fit row count changed: {int(labels.numel())}")
    labels = labels.to(dtype=torch.long, device="cpu")
    if labels.lt(0).any().item() or labels.ge(EXPECTED_VOCABULARY_SIZE).any().item():
        raise TRR0009TrainError("fit labels contain out-of-range vocabulary IDs")
    return torch.bincount(labels, minlength=EXPECTED_VOCABULARY_SIZE).to(dtype=torch.long)


def _support_receipt(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    ids, values = build_support_from_counts(counts, vocabulary_size=EXPECTED_VOCABULARY_SIZE)
    bins: dict[str, dict[str, int]] = {
        "unseen_0": {"token_ids": int((counts == 0).sum()), "rows": 0},
        "seen_1_4": {"token_ids": int(((counts >= 1) & (counts <= 4)).sum()), "rows": int(counts[(counts >= 1) & (counts <= 4)].sum())},
        "seen_5_9": {"token_ids": int(((counts >= 5) & (counts <= 9)).sum()), "rows": int(counts[(counts >= 5) & (counts <= 9)].sum())},
        "seen_10_49": {"token_ids": int(((counts >= 10) & (counts <= 49)).sum()), "rows": int(counts[(counts >= 10) & (counts <= 49)].sum())},
        "seen_50_plus": {"token_ids": int((counts >= 50).sum()), "rows": int(counts[counts >= 50].sum())},
    }
    receipt = {
        "vocabulary_size": EXPECTED_VOCABULARY_SIZE,
        "supported_token_count": int(ids.numel()),
        "support_digest": support_digest(ids, values),
        "support_ids_tensor_sha256": tensor_sha256(ids),
        "support_counts_tensor_sha256": tensor_sha256(values),
        "frequency_vector_tensor_sha256": tensor_sha256(counts),
        "frequency_bins": bins,
        "derived_from": "fit_truth valid post-BOS positions only",
    }
    return ids, values, receipt


def _load_schedule(path: Path, *, expected_records: int, config: TrainingConfig) -> tuple[PositionSchedule, dict[str, Any]]:
    record = _file_record(path, description="published schedule")
    if record["sha256"] != EXPECTED_SCHEDULE_FILE_SHA:
        raise TRR0009TrainError("published schedule file hash differs")
    tensors = load_file(str(path), device="cpu")
    required = {"batch_record_indices", "draw_record_slots", "draw_position_slots", "eligible_counts", "used_replacement"}
    if set(tensors) != required:
        raise TRR0009TrainError("schedule tensor keys differ")
    schedule = PositionSchedule(
        batch_record_indices=tensors["batch_record_indices"].to(dtype=torch.long),
        draw_record_slots=tensors["draw_record_slots"].to(dtype=torch.long),
        draw_position_slots=tensors["draw_position_slots"].to(dtype=torch.long),
        eligible_counts=tensors["eligible_counts"].to(dtype=torch.int32),
        used_replacement=tensors["used_replacement"].to(dtype=torch.bool),
        seed=EXPECTED_SEED,
        position_budget=EXPECTED_POSITION_BUDGET,
        record_batch_size=EXPECTED_RECORD_BATCH_SIZE,
    )
    if schedule.steps != config.steps or schedule.total_draws != config.steps * config.position_budget:
        raise TRR0009TrainError("schedule step or draw count differs")
    if int(schedule.batch_record_indices.shape[1]) != config.record_batch_size:
        raise TRR0009TrainError("schedule record batch differs")
    if int(schedule.batch_record_indices.max().item()) >= expected_records:
        raise TRR0009TrainError("schedule references records outside the fit bank")
    digest = schedule_digest(schedule)
    if digest != EXPECTED_SCHEDULE_DIGEST:
        raise TRR0009TrainError(f"schedule semantic digest differs: {digest}")
    return schedule, {"file": record, **schedule_metadata(schedule), "semantic_digest": digest}


def _challenge_from_wrong_mask(wrong_mask: torch.Tensor, *, cap: int, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    wrong = wrong_mask.detach().to(device="cpu", dtype=torch.bool).clone()
    if wrong.ndim != 2:
        raise TRR0009TrainError("wrong mask must be [records,positions]")
    wrong[:, 0] = False
    indices = torch.nonzero(wrong, as_tuple=False)
    total = int(indices.shape[0])
    if total <= cap:
        selected = indices
        rule = "all_initially_wrong_rows"
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        chosen = torch.randperm(total, generator=generator)[:cap]
        selected = indices.index_select(0, chosen)
        position_base = int(wrong.shape[1])
        order = torch.argsort(selected[:, 0].to(torch.int64) * position_base + selected[:, 1].to(torch.int64))
        selected = selected.index_select(0, order)
        rule = "seeded_uniform_initially_wrong_rows"
    selected_mask = torch.zeros_like(wrong)
    if selected.numel():
        selected_mask[selected[:, 0], selected[:, 1]] = True
    return selected_mask, {
        "all_initially_wrong_rows": total,
        "selected_rows": int(selected.shape[0]),
        "cap": int(cap),
        "selection_rule": rule,
        "selection_seed": int(seed),
        "mask_tensor_sha256": tensor_sha256(selected_mask.to(dtype=torch.uint8)),
        "initial_selected_accuracy": 0.0 if total else None,
        "empty_challenge": total == 0,
    }


def _collect_wrong_mask(model: torch.nn.Module, data: PublicJointData, *, device: torch.device, record_batch_size: int, position_budget: int) -> torch.Tensor:
    wrong = torch.zeros_like(data.fit_valid_mask, dtype=torch.bool, device="cpu")
    embedding = data.embedding_table.to(device=device, dtype=torch.float32)
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
                logits = model.logits_from_rows(hidden, chunk[:, 0], chunk[:, 1], embedding)
                predictions = logits.argmax(dim=-1)
                errors = predictions.ne(labels[chunk[:, 0], chunk[:, 1]]).detach().cpu()
                if errors.any().item():
                    error_indices = chunk.detach().cpu()[errors]
                    wrong[start + error_indices[:, 0], error_indices[:, 1]] = True
    return wrong


def _challenge_metrics(model: torch.nn.Module, data: PublicJointData, challenge_mask: torch.Tensor, *, device: torch.device, record_batch_size: int, position_budget: int) -> dict[str, Any]:
    indices_cpu = torch.nonzero(challenge_mask.to(dtype=torch.bool), as_tuple=False)
    rows = int(indices_cpu.shape[0])
    if rows == 0:
        return {"empty": True, "correct_tokens": 0, "token_rows": 0, "token_accuracy": None, "loss": None}
    embedding = data.embedding_table.to(device=device, dtype=torch.float32)
    total_loss = 0.0
    correct = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(data.fit_observations.shape[0]), record_batch_size):
            local = indices_cpu[(indices_cpu[:, 0] >= start) & (indices_cpu[:, 0] < min(start + record_batch_size, int(data.fit_observations.shape[0])))]
            if not local.numel():
                continue
            stop = min(start + record_batch_size, int(data.fit_observations.shape[0]))
            activation = data.fit_observations[start:stop].to(device=device, dtype=torch.float32)
            mask = data.fit_valid_mask[start:stop].to(device=device, dtype=torch.bool)
            labels = data.fit_truth[start:stop].to(device=device, dtype=torch.long)
            hidden = model.projected_hidden(activation, mask)
            local = local.clone()
            local[:, 0] -= start
            for chunk in local.split(position_budget):
                logits = model.logits_from_rows(hidden, chunk[:, 0].to(device=device), chunk[:, 1].to(device=device), embedding)
                target = labels[chunk[:, 0].to(device=device), chunk[:, 1].to(device=device)]
                total_loss += float(F.cross_entropy(logits, target, reduction="sum").detach().cpu())
                correct += int(logits.argmax(dim=-1).eq(target).sum().detach().cpu())
    return {"empty": False, "correct_tokens": correct, "token_rows": rows, "token_accuracy": correct / rows, "loss": total_loss / rows}


def _train_step(model: torch.nn.Module, data: PublicJointData, schedule: PositionSchedule, step_index: int, *, device: torch.device, embedding: torch.Tensor, optimizer: torch.optim.Optimizer, gradient_clip_norm: float, adaptable: bool) -> dict[str, Any]:
    model.train()
    batch_indices = schedule.batch_record_indices[step_index]
    activation = data.fit_observations.index_select(0, batch_indices).to(device=device, dtype=torch.float32)
    mask = data.fit_valid_mask.index_select(0, batch_indices).to(device=device, dtype=torch.bool)
    labels = data.fit_truth.index_select(0, batch_indices).to(device=device, dtype=torch.long)
    record_slots = schedule.draw_record_slots[step_index].to(device=device)
    position_slots = schedule.draw_position_slots[step_index].to(device=device)
    if position_slots.eq(0).any().item() or (~mask[record_slots, position_slots]).any().item():
        raise TRR0009TrainError("schedule contains invalid or BOS draw")
    hidden = model.projected_hidden(activation, mask)
    logits = model.logits_from_rows(hidden, record_slots, position_slots, embedding)
    target = labels[record_slots, position_slots]
    ce_loss = F.cross_entropy(logits, target)
    penalty = model.readout_penalty() if adaptable else ce_loss.new_zeros(())
    loss = ce_loss + penalty
    if not torch.isfinite(loss).item():
        raise TRR0009TrainError("training loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), gradient_clip_norm, error_if_nonfinite=True)
    gradient_norms = {name: (None if param.grad is None else float(param.grad.detach().norm().cpu())) for name, param in model.named_parameters()}
    optimizer.step()
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all().item():
            raise TRR0009TrainError("model parameter became non-finite")
    predictions = logits.detach().argmax(dim=-1)
    return {
        "step": step_index + 1,
        "loss": float(loss.detach().cpu()),
        "cross_entropy_loss": float(ce_loss.detach().cpu()),
        "readout_penalty": float(penalty.detach().cpu()),
        "correct_tokens": int(predictions.eq(target).sum().cpu()),
        "token_rows": int(target.numel()),
        "token_accuracy": float(predictions.eq(target).float().mean().cpu()),
        "gradient_norm": float(grad_norm.detach().cpu()),
        "gradient_norms": gradient_norms,
        "activation_shape": list(activation.shape),
        "label_shape": list(labels.shape),
        "draw_shape": list(target.shape),
        "record_batch_indices": [int(x) for x in batch_indices.tolist()],
        "unique_records_in_batch": int(torch.unique(batch_indices).numel()),
        "eligible_positions": int(schedule.eligible_counts[step_index].item()),
        "used_replacement": bool(schedule.used_replacement[step_index].item()),
    }


def _new_model(starting_state: Path, *, device: torch.device) -> ResidualMLPPositionwiseDecoder:
    return load_published_residual_state(starting_state, hidden_size=EXPECTED_HIDDEN_SIZE, vocabulary_size=EXPECTED_VOCABULARY_SIZE, context_width=EXPECTED_CONTEXT_WIDTH, bottleneck_size=DEFAULT_BOTTLENECK_SIZE).to(device=device)


def _fresh_frequency_metrics(model: torch.nn.Module, observations: torch.Tensor, truth: torch.Tensor, valid_mask: torch.Tensor, embedding: torch.Tensor, counts: torch.Tensor, *, device: torch.device, record_batch_size: int, position_budget: int) -> dict[str, Any]:
    """Compute the registered 0/1-4/5-9/10-49/50+ target-frequency bins."""

    names = ("unseen_0", "seen_1_4", "seen_5_9", "seen_10_49", "seen_50_plus")
    rows = {name: 0 for name in names}
    correct = {name: 0 for name in names}
    runtime_embedding = embedding.to(device=device, dtype=torch.float32)
    reference = counts.to(device="cpu", dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(observations.shape[0]), record_batch_size):
            stop = min(start + record_batch_size, int(observations.shape[0]))
            activation = observations[start:stop].to(device=device, dtype=torch.float32)
            mask = valid_mask[start:stop].to(device=device, dtype=torch.bool)
            labels = truth[start:stop].to(device=device, dtype=torch.long)
            hidden = model.projected_hidden(activation, mask)
            indices = torch.nonzero(mask, as_tuple=False)
            indices = indices[indices[:, 1] > 0]
            for chunk in indices.split(position_budget):
                logits = model.logits_from_rows(hidden, chunk[:, 0], chunk[:, 1], runtime_embedding)
                target = labels[chunk[:, 0], chunk[:, 1]]
                prediction = logits.argmax(dim=-1)
                target_cpu = target.detach().cpu()
                hit_cpu = prediction.eq(target).detach().cpu()
                freq = reference.index_select(0, target_cpu)
                selectors = (
                    freq.eq(0),
                    (freq >= 1) & (freq <= 4),
                    (freq >= 5) & (freq <= 9),
                    (freq >= 10) & (freq <= 49),
                    freq >= 50,
                )
                for name, selector in zip(names, selectors):
                    rows[name] += int(selector.sum().item())
                    correct[name] += int(hit_cpu[selector].sum().item())
    return {
        "scheme": "fit-current-enriched-frequency-0/1-4/5-9/10-49/50+",
        "metrics": {name: {"rows": rows[name], "correct": correct[name], "token_accuracy": (correct[name] / rows[name] if rows[name] else None)} for name in names},
    }


def _evaluate_point(model: torch.nn.Module, data: PublicJointData, challenge_mask: torch.Tensor, counts: torch.Tensor, *, device: torch.device, config: TrainingConfig) -> dict[str, Any]:
    frequency_reference = data.fit_truth[:, 1:][data.fit_valid_mask[:, 1:].to(dtype=torch.bool)]
    fit = evaluate_dataset(model, data.fit_observations, data.fit_truth, data.fit_valid_mask, data.embedding_table, tuple("fit" for _ in data.fit_record_ids), device=device, record_batch_size=config.record_batch_size, position_budget=config.position_budget, frequency_reference=frequency_reference)
    validation = evaluate_dataset(model, data.validation_observations, data.validation_truth, data.validation_valid_mask, data.embedding_table, data.validation_groups, device=device, record_batch_size=config.record_batch_size, position_budget=config.position_budget, frequency_reference=frequency_reference)
    fit["legacy_frequency_bucket_scheme"] = "TRR-0005 evaluate_dataset bins: unseen_0/seen_1_4/seen_5_19/seen_20_plus"
    validation["legacy_frequency_bucket_scheme"] = "TRR-0005 evaluate_dataset bins: unseen_0/seen_1_4/seen_5_19/seen_20_plus"
    fit["fresh_frequency_bucket_metrics"] = _fresh_frequency_metrics(model, data.fit_observations, data.fit_truth, data.fit_valid_mask, data.embedding_table, counts, device=device, record_batch_size=config.record_batch_size, position_budget=config.position_budget)
    validation["fresh_frequency_bucket_metrics"] = _fresh_frequency_metrics(model, data.validation_observations, data.validation_truth, data.validation_valid_mask, data.embedding_table, counts, device=device, record_batch_size=config.record_batch_size, position_budget=config.position_budget)
    challenge = _challenge_metrics(model, data, challenge_mask, device=device, record_batch_size=config.record_batch_size, position_budget=config.position_budget)
    return {"fit": fit, "validation": validation, "challenge_initially_wrong": challenge}


def _select_step(points: Sequence[Mapping[str, Any]], *, unchanged: bool) -> int:
    if not points:
        raise TRR0009TrainError("no learning-curve points")
    if unchanged:
        return 0
    best_step = int(points[0]["step"])
    best_score = float(points[0]["validation"]["style_balanced_token_accuracy"])
    for point in points[1:]:
        score = float(point["validation"]["style_balanced_token_accuracy"])
        step = int(point["step"])
        if score > best_score:
            best_score, best_step = score, step
    return best_step


def _arm_output_paths(root: Path, arm: str) -> dict[str, Path]:
    arm_root = root / arm
    return {"root": arm_root, "curve": arm_root / "learning_curve.json", "state": arm_root / "selected.safetensors", "receipt": arm_root / "receipt.json"}


def _check_create_only_output(root: Path, arms: Sequence[str]) -> None:
    if root.exists() or root.is_symlink():
        raise TRR0009TrainError(f"training output root is create-only and already exists: {root}")
    for arm in arms:
        if arm not in {"unchanged_anchor", "continued_fixed_readout", "continued_adaptable_readout"}:
            raise TRR0009TrainError(f"unknown arm: {arm}")


def _prepare_common(paths: Paths, config: TrainingConfig, *, device: torch.device) -> tuple[PublicJointData, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], PositionSchedule]:
    for path, description in ((paths.fit_manifest, "fit manifest"), (paths.validation_manifest, "validation manifest"), (paths.embedding_path, "embedding table"), (paths.starting_state, "starting state"), (paths.schedule_path, "schedule")):
        _file_record(path, description=description)
    if file_sha256(paths.starting_state) != STARTING_STATE_SHA256:
        raise TRR0009TrainError("starting state hash differs from the published selected residual state")
    fit_resources = _manifest_resource_bindings(paths.fit_manifest)
    validation_resources = _manifest_resource_bindings(paths.validation_manifest)
    declared_embedding_hashes = {
        entry["actual"]["sha256"]
        for entry in fit_resources["resources"]
        if entry.get("name") == "embedding_table"
    }
    if not declared_embedding_hashes or file_sha256(paths.embedding_path) not in declared_embedding_hashes:
        raise TRR0009TrainError("explicit embedding path does not match the manifest-declared embedding bytes")
    data = load_public_joint_data(paths.fit_manifest, paths.validation_manifest, embedding_path=paths.embedding_path)
    geometry = _validate_data_geometry(data)
    embedding = _validate_embedding_asset(data, paths.embedding_path)
    input_resource_bindings = {"fit_manifest_resources": fit_resources, "validation_manifest_resources": validation_resources}
    input_resource_recheck = {"fit_manifest_resources": _verify_resource_bindings_unchanged(fit_resources), "validation_manifest_resources": _verify_resource_bindings_unchanged(validation_resources)}
    counts = _frequency_counts(data)
    support_ids, support_counts, support = _support_receipt(counts)
    schedule, schedule_receipt = _load_schedule(paths.schedule_path, expected_records=int(data.fit_observations.shape[0]), config=config)
    metadata = {
        "fit_manifest": _file_record(paths.fit_manifest, description="fit manifest"),
        "validation_manifest": _file_record(paths.validation_manifest, description="validation manifest"),
        "starting_state": _file_record(paths.starting_state, description="starting state"),
        "embedding": embedding,
        "input_resource_bindings": input_resource_bindings,
        "input_resource_recheck": input_resource_recheck,
        "geometry": geometry,
        "support": support,
        "schedule": schedule_receipt,
        "device": str(device),
        "asset_finiteness_validation": "embedding table scanned once before model calls",
    }
    return data, counts, support_ids, support_counts, metadata, embedding, schedule


def _build_challenge(data: PublicJointData, paths: Paths, support_ids: torch.Tensor, support_counts: torch.Tensor, *, device: torch.device, config: TrainingConfig) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    model = _new_model(paths.starting_state, device=device)
    wrong = _collect_wrong_mask(model, data, device=device, record_batch_size=config.record_batch_size, position_budget=config.position_budget)
    challenge_mask, receipt = _challenge_from_wrong_mask(wrong, cap=config.challenge_cap, seed=config.challenge_seed)
    receipt.update({"schema": CHALLENGE_SCHEMA, "task_id": TASK_ID, "starting_state_sha256": STARTING_STATE_SHA256, "definition": "positions initially wrong under the actual selected step-2100 residual MLP state on the public fitting bank", "fit_rows": int((data.fit_valid_mask[:, 1:]).sum().item())})
    selected = challenge_mask.to(dtype=torch.bool)
    if (selected & ~wrong.to(dtype=torch.bool)).any().item():
        raise TRR0009TrainError("selected challenge contains a row that was correct under the actual starting model")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return challenge_mask, receipt, {"wrong_mask_tensor_sha256": tensor_sha256(wrong.to(dtype=torch.uint8))}


def run_qualification(paths: Paths, config: TrainingConfig, *, device: torch.device, numerical_settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    _validate_config(config, qualification=True)
    guard = ResourceGuard.from_config(config)
    started = time.time()
    before = _resource_snapshot(device)
    _enforce_resource_guard(before, guard, stage="qualification_start", device=device)
    preparation_started = time.perf_counter()
    data, counts, support_ids, support_counts, common, embedding_record, schedule = _prepare_common(paths, config, device=device)
    preparation_seconds = time.perf_counter() - preparation_started
    embedding = data.embedding_table.to(device=device, dtype=torch.float32)
    base = _new_model(paths.starting_state, device=device)
    adaptable = build_adaptable_from_base(_new_model(paths.starting_state, device=device), support_ids, support_counts).to(device=device)
    # Full-output equality uses a separate B=1 fixture; the largest B=8 cell
    # below uses the selected-row path and therefore preserves the fit geometry.
    fixture_activation = data.fit_observations[:1].to(device=device, dtype=torch.float32)
    fixture_mask = data.fit_valid_mask[:1].to(device=device, dtype=torch.bool)
    from trr0009_model import zero_correction_equivalence
    equivalence_started = time.perf_counter()
    equivalence = zero_correction_equivalence(base, adaptable, fixture_activation, fixture_mask, embedding, max_rows=config.position_budget)
    equivalence_seconds = time.perf_counter() - equivalence_started
    if not equivalence["projected_hidden_exact"] or not equivalence["selected_logits_exact"] or not equivalence["full_logits_exact"]:
        raise TRR0009TrainError(f"zero correction equivalence failed: {equivalence}")
    optimizer = torch.optim.AdamW([
        {"params": adaptable.base.parameters(), "lr": config.learning_rate},
        {"params": [adaptable.raw_gain, adaptable.raw_bias], "lr": config.readout_learning_rate},
    ], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.steps))
    steps: list[dict[str, Any]] = []
    for step in range(min(2, config.steps)):
        step_started = time.perf_counter()
        result = _train_step(adaptable, data, schedule, step, device=device, embedding=embedding, optimizer=optimizer, gradient_clip_norm=config.gradient_clip_norm, adaptable=True)
        scheduler.step()
        result["elapsed_seconds"] = time.perf_counter() - step_started
        steps.append(result)
        current = _resource_snapshot(device)
        _enforce_resource_guard(current, guard, stage=f"qualification_step_{step + 1}", device=device)
    after = _resource_snapshot(device)
    _enforce_resource_guard(after, guard, stage="qualification_end", device=device)
    if not steps or any(not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["gradient_norm"])) for row in steps):
        raise TRR0009TrainError("qualification produced non-finite loss or gradient")
    return {
        "schema": QUALIFICATION_SCHEMA,
        "task_id": TASK_ID,
        "created_utc": _utc_now(),
        "paths": common,
        "runtime": _runtime_binding(paths.repository_root, numerical_settings),
        "config": asdict(config),
        "resource_estimate": resource_estimate(),
        "resource_guard": asdict(guard),
        "resource_before": before,
        "resource_after": after,
        "preparation_seconds": preparation_seconds,
        "equivalence_seconds": equivalence_seconds,
        "gradient_update_seconds": sum(float(row["elapsed_seconds"]) for row in steps),
        "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
        "actual_geometry": steps[-1]["activation_shape"],
        "actual_draw_shape": steps[-1]["draw_shape"],
        "qualification_steps": len(steps),
        "steps": steps,
        "zero_correction_equivalence_b1": equivalence,
        "elapsed_seconds": time.time() - started,
        "qualification_status": "PASS",
    }


def _validation_steps(config: TrainingConfig) -> tuple[int, ...]:
    values = list(range(0, config.steps + 1, config.validation_every))
    if values[-1] != config.steps:
        values.append(config.steps)
    return tuple(values)


def _clone_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _consider_checkpoint(model: torch.nn.Module, point: Mapping[str, Any], best_score: float, best_step: int, best_state: dict[str, torch.Tensor] | None) -> tuple[float, int, dict[str, torch.Tensor] | None]:
    score = float(point["validation"]["style_balanced_token_accuracy"])
    if score > best_score:
        return score, int(point["step"]), _clone_state_cpu(model)
    return best_score, best_step, best_state


def run_training(paths: Paths, config: TrainingConfig, *, device: torch.device, arms: Sequence[str], numerical_settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    _validate_config(config)
    _check_create_only_output(paths.output_root, arms)
    guard = ResourceGuard.from_config(config)
    started = time.time()
    before = _resource_snapshot(device)
    _enforce_resource_guard(before, guard, stage="run_start", device=device)
    preparation_started = time.perf_counter()
    data, counts, support_ids, support_counts, common, embedding_record, schedule = _prepare_common(paths, config, device=device)
    preparation_seconds = time.perf_counter() - preparation_started
    challenge_started = time.perf_counter()
    challenge_mask, challenge_receipt, challenge_hashes = _build_challenge(data, paths, support_ids, support_counts, device=device, config=config)
    challenge_seconds = time.perf_counter() - challenge_started
    common["challenge"] = {**challenge_receipt, **challenge_hashes}
    _write_create_only(paths.output_root / "preflight.json", {"schema": PREFLIGHT_SCHEMA, "task_id": TASK_ID, "created_utc": _utc_now(), "config": asdict(config), "runtime": _runtime_binding(paths.repository_root, numerical_settings), "paths": common, "resource_estimate": resource_estimate(), "resource_guard": asdict(guard), "resource_before": before, "preparation_seconds": preparation_seconds, "challenge_seconds": challenge_seconds}, description="training preflight")
    _write_create_only(paths.output_root / "challenge_receipt.json", common["challenge"], description="selected-start challenge receipt")
    output_arms: dict[str, Any] = {}
    embedding = data.embedding_table.to(device=device, dtype=torch.float32)
    checkpoints = set(_validation_steps(config))
    for arm in arms:
        arm_started = time.perf_counter()
        paths_for_arm = _arm_output_paths(paths.output_root, arm)
        model: torch.nn.Module
        adaptable = arm == "continued_adaptable_readout"
        unchanged = arm == "unchanged_anchor"
        if adaptable:
            model = build_adaptable_from_base(_new_model(paths.starting_state, device=device), support_ids, support_counts).to(device=device)
        else:
            model = _new_model(paths.starting_state, device=device)
        paths_for_arm["root"].mkdir(parents=True, exist_ok=False)
        progress_path = paths_for_arm["root"] / "progress.jsonl"
        points: list[dict[str, Any]] = []
        last_train: dict[str, Any] | None = None
        evaluation_seconds = 0.0
        gradient_update_seconds = 0.0
        optimizer_state_bytes = 0
        best_step = 0
        best_score = float("-inf")
        best_state: dict[str, torch.Tensor] | None = None
        if unchanged:
            evaluation_started = time.perf_counter()
            metrics = _evaluate_point(model, data, challenge_mask, counts, device=device, config=config)
            evaluation_elapsed = time.perf_counter() - evaluation_started
            point = {"step": 0, "learning_rate": 0.0, "readout_learning_rate": None, "train": None, "evaluation_wall_seconds": evaluation_elapsed, **metrics}
            points.append(point)
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(json.dumps(point, sort_keys=True, allow_nan=False) + "\n")
            evaluation_seconds += evaluation_elapsed
            best_step = 0
        else:
            if adaptable:
                optimizer = torch.optim.AdamW([
                    {"params": model.base.parameters(), "lr": config.learning_rate},
                    {"params": [model.raw_gain, model.raw_bias], "lr": config.readout_learning_rate},
                ], weight_decay=config.weight_decay)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
            for step in range(config.steps + 1):
                if step in checkpoints:
                    evaluation_started = time.perf_counter()
                    metrics = _evaluate_point(model, data, challenge_mask, counts, device=device, config=config)
                    evaluation_elapsed = time.perf_counter() - evaluation_started
                    point = {"step": step, "learning_rate": float(optimizer.param_groups[0]["lr"]), "readout_learning_rate": (float(optimizer.param_groups[1]["lr"]) if adaptable else None), "train": last_train, "evaluation_wall_seconds": evaluation_elapsed, **metrics}
                    points.append(point)
                    with progress_path.open("a", encoding="utf-8") as progress:
                        progress.write(json.dumps(point, sort_keys=True, allow_nan=False) + "\n")
                    evaluation_seconds += evaluation_elapsed
                    best_score, best_step, best_state = _consider_checkpoint(model, point, best_score, best_step, best_state)
                if step == config.steps:
                    break
                _enforce_resource_guard(_resource_snapshot(device), guard, stage=f"{arm}_before_step_{step + 1}", device=device)
                update_started = time.perf_counter()
                last_train = _train_step(model, data, schedule, step, device=device, embedding=embedding, optimizer=optimizer, gradient_clip_norm=config.gradient_clip_norm, adaptable=adaptable)
                gradient_update_seconds += time.perf_counter() - update_started
                optimizer_state_bytes = max(optimizer_state_bytes, _optimizer_state_bytes(optimizer))
                scheduler.step()
                if time.time() - started > config.max_seconds:
                    raise TRR0009TrainError(f"run exceeded max_seconds during {arm}")
            if best_state is None:
                raise TRR0009TrainError(f"{arm} produced no selected checkpoint state")
            model.load_state_dict(best_state, strict=True)
        selected_step = best_step
        selected_point = next(point for point in points if int(point["step"]) == selected_step)
        if unchanged:
            state_record = {"path": str(paths.starting_state), "sha256": STARTING_STATE_SHA256, "selected_step": 0, "separate_published_anchor": True}
        elif adaptable:
            state_record = save_supported_readout_state(paths_for_arm["state"], model, selected_step=selected_step, starting_state={"sha256": STARTING_STATE_SHA256}, fit_manifest=common["fit_manifest"], metadata={"selection_metric": "earliest maximum validation style_balanced_token_accuracy", "learning_rate": config.learning_rate, "readout_learning_rate": config.readout_learning_rate, "schedule_sha256": EXPECTED_SCHEDULE_DIGEST})
        else:
            state_record = save_positionwise_state(paths_for_arm["state"], model, method_id=RESIDUAL_MLP_METHOD_ID, selected_step=selected_step, initialization="published TRR-0007 current-bank residual selected checkpoint", distribution="current_enriched_start_on_improved_public_bank", bottleneck_size=DEFAULT_BOTTLENECK_SIZE, metadata={"task_id": TASK_ID, "starting_state_sha256": STARTING_STATE_SHA256, "fit_manifest_sha256": common["fit_manifest"]["sha256"], "selection_metric": "earliest maximum validation style_balanced_token_accuracy", "learning_rate": config.learning_rate, "schedule_sha256": EXPECTED_SCHEDULE_DIGEST})
        curve_record = {"schema": "token-reconstruction.trr0009-learning-curve.v1", "task_id": TASK_ID, "arm": arm, "starting_state_sha256": STARTING_STATE_SHA256, "fit_manifest_sha256": common["fit_manifest"]["sha256"], "points": points, "selected_step": selected_step, "selected_point": selected_point, "challenge": challenge_receipt}
        _write_create_only(paths_for_arm["curve"], curve_record, description=f"{arm} learning curve")
        materialization = {"performed": False, "reason": "direct supported-row correction is the training and inference path; full dictionary materialization is a separately accounted deployment preparation step"} if adaptable else None
        receipt = {"schema": "token-reconstruction.trr0009-arm-receipt.v1", "task_id": TASK_ID, "arm": arm, "selected_step": selected_step, "state": state_record, "materialization": materialization, "wall_seconds": time.perf_counter() - arm_started, "preparation_seconds": preparation_seconds, "challenge_seconds": challenge_seconds, "gradient_update_seconds": gradient_update_seconds, "evaluation_seconds": evaluation_seconds, "optimizer_state_bytes": optimizer_state_bytes, "resource_after": _resource_snapshot(device), "runtime": _runtime_binding(paths.repository_root, numerical_settings)}
        _write_create_only(paths_for_arm["receipt"], receipt, description=f"{arm} receipt")
        output_arms[arm] = receipt
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    after = _resource_snapshot(device)
    _enforce_resource_guard(after, guard, stage="run_end", device=device)
    result = {"schema": RUN_SCHEMA, "task_id": TASK_ID, "created_utc": _utc_now(), "config": asdict(config), "runtime": _runtime_binding(paths.repository_root, numerical_settings), "paths": common, "arms": output_arms, "resource_before": before, "resource_after": after, "resource_guard": asdict(guard), "resource_estimate": resource_estimate(), "preparation_seconds": preparation_seconds, "challenge_seconds": challenge_seconds, "wall_seconds": time.time() - started}
    _write_create_only(paths.output_root / "run_receipt.json", result, description="training run receipt")
    return result


def _parse_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TRR0009TrainError("CUDA requested but unavailable")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--fit-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--embedding-path", type=Path)
    parser.add_argument("--starting-state", type=Path)
    parser.add_argument("--schedule-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--readout-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--record-batch-size", type=int, default=EXPECTED_RECORD_BATCH_SIZE)
    parser.add_argument("--position-budget", type=int, default=EXPECTED_POSITION_BUDGET)
    parser.add_argument("--validation-every", type=int, default=DEFAULT_VALIDATION_EVERY)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--challenge-seed", type=int, default=CHALLENGE_SEED)
    parser.add_argument("--challenge-cap", type=int, default=CHALLENGE_CAP)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=DEFAULT_MAX_RESERVED_GIB)
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=DEFAULT_MAX_RSS_GIB)
    parser.add_argument("--minimum-host-available-gib", type=float, default=DEFAULT_MIN_HOST_GIB)
    parser.add_argument("--arms", nargs="+", default=["unchanged_anchor", "continued_fixed_readout", "continued_adaptable_readout"])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--qualification-only", action="store_true")
    return parser


def _failure_output_root(args: argparse.Namespace) -> Path:
    supplied = getattr(args, "output_root", None)
    if supplied:
        path = Path(supplied).expanduser()
        if not path.is_absolute():
            path = Path(args.repository_root).expanduser() / path
        return path.resolve()
    return (Path(args.repository_root).expanduser().resolve() / "experiments/TRR-0009/training/run_v1").resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = _resource_paths(args)
        config = TrainingConfig(steps=args.steps, learning_rate=args.learning_rate, readout_learning_rate=args.readout_learning_rate, weight_decay=args.weight_decay, gradient_clip_norm=args.gradient_clip_norm, record_batch_size=args.record_batch_size, position_budget=args.position_budget, validation_every=args.validation_every, seed=args.seed, challenge_seed=args.challenge_seed, challenge_cap=args.challenge_cap, max_seconds=args.max_seconds, maximum_gpu_reserved_gib=args.maximum_gpu_reserved_gib, minimum_free_gib=args.minimum_free_gib, maximum_host_rss_gib=args.maximum_host_rss_gib, minimum_host_available_gib=args.minimum_host_available_gib)
        numerical_settings = _configure_numerics()
        device = _parse_device(args.device)
        if args.preflight_only:
            _validate_config(config)
            guard = ResourceGuard.from_config(config)
            resource_before = _resource_snapshot(device)
            _enforce_resource_guard(resource_before, guard, stage="preflight_start", device=device)
            data, counts, support_ids, support_counts, common, embedding_record, schedule = _prepare_common(paths, config, device=device)
            resource_after = _resource_snapshot(device)
            _enforce_resource_guard(resource_after, guard, stage="preflight_after_asset_load", device=device)
            payload = {"schema": PREFLIGHT_SCHEMA, "task_id": TASK_ID, "created_utc": _utc_now(), "config": asdict(config), "runtime": _runtime_binding(paths.repository_root, numerical_settings), "paths": common, "resource_estimate": resource_estimate(), "resource_guard": asdict(guard), "resource_before": resource_before, "resource_after": resource_after, "status": "METADATA_PREFLIGHT_PASS"}
            _write_create_only(paths.output_root / "preflight.json", payload, description="training preflight")
        elif args.qualification_only:
            _check_create_only_output(paths.output_root, args.arms)
            payload = run_qualification(paths, config, device=device, numerical_settings=numerical_settings)
            _write_create_only(paths.output_root / "qualification.json", payload, description="largest-cell qualification")
        else:
            run_training(paths, config, device=device, arms=args.arms, numerical_settings=numerical_settings)
    except Exception as exc:
        try:
            failure_root = _failure_output_root(args)
            if not failure_root.is_symlink():
                failure_root.mkdir(parents=True, exist_ok=True)
                failure_path = failure_root / "failure.json"
                if not failure_path.exists():
                    runtime_root = paths.repository_root if "paths" in locals() else Path(args.repository_root).expanduser().resolve()
                    _write_create_only(failure_path, {"schema": "token-reconstruction.trr0009-failure.v1", "task_id": TASK_ID, "created_utc": _utc_now(), "error_type": type(exc).__name__, "error": str(exc), "runtime": _runtime_binding(runtime_root, locals().get("numerical_settings"))}, description="training failure")
        except Exception:
            pass
        print(f"TRR-0009 training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
