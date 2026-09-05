"""Shared validation and runtime helpers for the TRR-P01 pilot scripts.

The public reconstruction process imports this module only from the task-local
source tree.  The helpers deliberately make the selected device explicit so a
CPU integrity test cannot initialize a reserved CUDA device by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Iterable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from token_reconstruction.trr_p01 import BOS_TOKEN_ID


TASK_ID = "TRR-P01"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
TARGET_MODEL_ID = "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"
TARGET_MODEL_REVISION = "7fa9d06a59246629244cdd3b6b92e4fc756baa0f"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
CUT_DEPTH = 4
VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
SEQUENCE_TOKENS = 40
SCORED_TOKENS = 39
CONDITIONS = ("matched_public", "shifted_target_lora")
METRICS = ("cosine", "l2")
METHODS = ("boundary", "raw_embedding")
CORRECTION_METHOD = "reference_corrected"
REFERENCE_TOKEN = 220
OBSERVATION_SCHEMA = "token-reconstruction.trr-p01-observation.v1"
OBSERVATION_INDEX_SCHEMA = "token-reconstruction.trr-p01-observation-index.v1"
CONFIG_SCHEMA = "token-reconstruction.trr-p01-public-config.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr-p01-predictions.v1"
EVIDENCE_SCHEMA = "token-reconstruction.trr-p01-reconstructor-evidence.v1"
FREEZE_SCHEMA = "token-reconstruction.trr-p01-freeze-receipt.v1"


class PilotError(RuntimeError):
    """Raised when a task-local pilot contract is violated."""


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


def utc_now() -> str:
    from datetime import datetime, timezone

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise PilotError(f"artifact must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    label = (
        path.resolve().relative_to(root.resolve()).as_posix()
        if root is not None
        else str(path)
    )
    return {
        "path": label,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise PilotError(f"JSON input must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"invalid JSON: {path}") from exc


def write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise PilotError(f"JSON output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise PilotError(f"JSONL output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PilotError(f"JSONL input must be a regular file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PilotError(f"invalid JSONL at {path}:{number}") from exc
            if not isinstance(value, dict):
                raise PilotError(f"JSONL row is not an object at {path}:{number}")
            rows.append(value)
    return rows


def require_create_only_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise PilotError(f"output directory must be create-only: {path}")
    path.mkdir(parents=True)
    return path


def require_create_only_file(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise PilotError(f"output file must be create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def public_path(root: Path, relative: str) -> Path:
    """Resolve a relative artifact entry and verify its bytes and regular-file state."""

    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise PilotError(f"public path is not relative: {relative!r}")
    base = root.resolve()
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise PilotError(f"public path escaped input root: {relative!r}") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise PilotError(f"public artifact is not a regular file: {relative!r}")
    return candidate


def artifact_entry(path: Path, *, relative_to: Path, **extra: Any) -> dict[str, Any]:
    return {
        **extra,
        **file_record(path, root=relative_to),
    }


def digest_tensor(tensor: torch.Tensor) -> str:
    """Digest a CPU contiguous tensor including dtype, shape, and bytes."""

    value = tensor.detach().cpu().contiguous()
    descriptor = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(descriptor + b"\0" + raw).hexdigest()


def expected_mask_and_positions(records: int) -> tuple[torch.Tensor, torch.Tensor]:
    if records <= 0:
        raise PilotError("record count must be positive")
    mask = torch.ones((records, SEQUENCE_TOKENS), dtype=torch.int64)
    positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.int64).view(1, -1)
    return mask, positions.expand(records, -1).clone()


def mask_digest() -> str:
    return digest_tensor(torch.ones(SEQUENCE_TOKENS, dtype=torch.int64))


def position_digest() -> str:
    return digest_tensor(torch.arange(SEQUENCE_TOKENS, dtype=torch.int64))


def observation_row_digest(observation: torch.Tensor) -> str:
    value = observation.detach().cpu().contiguous()
    if tuple(value.shape) != (SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise PilotError("observation row geometry changed")
    return digest_tensor(value)


def _requested_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    return torch.device(device)


def cuda_requested(device: torch.device | str | None) -> bool:
    selected = _requested_device(device)
    return selected is not None and selected.type == "cuda"


def environment_record(device: torch.device | str | None = None) -> dict[str, Any]:
    """Return environment metadata without touching CUDA unless explicitly selected."""

    result: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "kernel": platform.uname()._asdict(),
        "pid": os.getpid(),
        "selected_device": str(_requested_device(device) or "unspecified"),
        "tf32": {
            "matmul_allow": bool(torch.backends.cuda.matmul.allow_tf32) if cuda_requested(device) else False,
            "cudnn_allow": bool(torch.backends.cudnn.allow_tf32) if cuda_requested(device) else False,
            "queried": cuda_requested(device),
        },
    }
    if cuda_requested(device):
        if not torch.cuda.is_available():
            raise PilotError("CUDA was selected but is unavailable")
        selected = _requested_device(device)
        assert selected is not None
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        result.update(
            {
                "cuda_device": torch.cuda.get_device_name(index),
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
                "cuda_index": int(index),
            }
        )
    return result


def peak_memory(device: torch.device | str | None = None) -> dict[str, int]:
    """Measure process RSS and optional selected-device CUDA peaks."""

    values = {
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    }
    if cuda_requested(device):
        if not torch.cuda.is_available():
            raise PilotError("CUDA was selected but is unavailable")
        selected = _requested_device(device)
        assert selected is not None
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        values.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
    return values


def synchronize(device: torch.device | str | None = None) -> None:
    if cuda_requested(device):
        if not torch.cuda.is_available():
            raise PilotError("CUDA was selected but is unavailable")
        torch.cuda.synchronize(_requested_device(device))


class PhaseTimer:
    """Record wall-clock phases with synchronization on an explicitly selected device."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = device
        self.records: list[dict[str, Any]] = []

    def measure(self, name: str):
        return _MeasuredPhase(self, name)


class _MeasuredPhase:
    def __init__(self, owner: PhaseTimer, name: str) -> None:
        self.owner = owner
        self.name = name

    def __enter__(self):
        synchronize(self.owner.device)
        self.started_utc = utc_now()
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        synchronize(self.owner.device)
        self.owner.records.append(
            {
                "phase": self.name,
                "started_utc": self.started_utc,
                "ended_utc": utc_now(),
                "elapsed_seconds": time.perf_counter() - self.started,
                "exit_status": 0 if exc_type is None else 1,
            }
        )
        return False


def seed_everything(seed: int, device: torch.device | str | None = None) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # These settings are part of the numerical contract.  They are set even
    # for CPU tests, but CUDA RNG and backend calls require explicit selection.
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if cuda_requested(device):
        if not torch.cuda.is_available():
            raise PilotError("CUDA was selected but is unavailable")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def command_record(device: torch.device | str | None = None) -> dict[str, Any]:
    return {
        "argv": [str(value) for value in sys.argv],
        "cwd": os.getcwd(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "TOKENIZERS_PARALLELISM",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
            )
        },
        "selected_device": str(_requested_device(device) or "unspecified"),
    }


def validate_model(model: Any, *, device: torch.device | str) -> None:
    if getattr(model.config, "hidden_size", None) != HIDDEN_SIZE:
        raise PilotError("public model hidden size changed")
    if getattr(model.config, "vocab_size", None) != VOCAB_SIZE:
        raise PilotError("public model vocabulary changed")
    if not model.training:
        pass
    else:
        raise PilotError("model must be in evaluation mode")
    actual = next(model.parameters()).device
    if actual != torch.device(device):
        raise PilotError(f"model loaded on {actual}, expected {device}")


def load_model(
    *,
    device: torch.device,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    model_path: Path | None = None,
) -> Any:
    """Load one pinned local model; callers decide whether it is public or evaluator-only."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise PilotError("CUDA was requested but unavailable")
    from transformers import AutoModelForCausalLM

    source: str | Path = model_path if model_path is not None else model_id
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if model_path is None:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model = model.to(device).eval()
    model.requires_grad_(False)
    validate_model(model, device=device)
    return model


def load_public_model(*, device: torch.device, model_path: Path | None = None) -> Any:
    return load_model(
        device=device,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        model_path=model_path,
    )


def load_target_model(*, device: torch.device, model_path: Path | None = None) -> Any:
    return load_model(
        device=device,
        model_id=TARGET_MODEL_ID,
        revision=TARGET_MODEL_REVISION,
        model_path=model_path,
    )


def validate_public_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "task_id": TASK_ID,
        "status": "COMMITTED_BEFORE_OBSERVATIONS",
        "truth_opened": False,
        "source_truth_included": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise PilotError(f"frozen plan field {key} changed")
    model = plan.get("model")
    if not isinstance(model, Mapping):
        raise PilotError("frozen model section missing")
    if (
        model.get("id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or model.get("hidden_size") != HIDDEN_SIZE
        or model.get("vocab_size") != VOCAB_SIZE
        or model.get("cut_depth") != CUT_DEPTH
        or model.get("bos_token_id") != BOS_TOKEN_ID
    ):
        raise PilotError("frozen public model identity changed")
    panel = plan.get("panel")
    if not isinstance(panel, Mapping) or panel.get("records") != 16:
        raise PilotError("frozen panel size changed")
    if panel.get("sequence_tokens") != SEQUENCE_TOKENS or panel.get("scored_tokens") != SCORED_TOKENS:
        raise PilotError("frozen panel geometry changed")
    if panel.get("style_counts") != {
        "prose": 4,
        "code": 4,
        "numeric_plus_punctuation": 4,
        "unicode_plus_instruction": 4,
    }:
        raise PilotError("frozen panel strata changed")
    if plan.get("condition_order") != list(CONDITIONS):
        raise PilotError("frozen condition order changed")
    if plan.get("metric_order") != list(METRICS):
        raise PilotError("frozen metric order changed")


def _validate_artifact_entry(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise PilotError(f"{label} artifact entry changed")
    if not isinstance(value["path"], str) or not isinstance(value["sha256"], str):
        raise PilotError(f"{label} artifact identity changed")
    if int(value["bytes"]) < 0 or len(value["sha256"]) != 64:
        raise PilotError(f"{label} artifact digest changed")
    return dict(value)


def validate_observation_index(index: Mapping[str, Any], *, records: int) -> None:
    if (
        index.get("schema") != OBSERVATION_INDEX_SCHEMA
        or index.get("task_id") != TASK_ID
        or index.get("truth_opened") is not False
        or index.get("source_material_included") is not False
        or index.get("geometry")
        != {
            "records": records,
            "sequence_tokens": SEQUENCE_TOKENS,
            "scored_tokens": SCORED_TOKENS,
            "hidden_size": HIDDEN_SIZE,
        }
    ):
        raise PilotError("observation index schema or geometry changed")
    rows = index.get("records")
    if not isinstance(rows, list) or len(rows) != records:
        raise PilotError("observation index record count changed")
    expected_ids = [f"p01-r{position:04d}" for position in range(1, records + 1)]
    observed_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "record_id",
            "sequence_length",
            "mask_digest",
            "position_digest",
            "observation_digest",
        }:
            raise PilotError("observation index exposes non-stage metadata")
        record_id = row.get("record_id")
        if not isinstance(record_id, str):
            raise PilotError("opaque record ID is invalid")
        observed_ids.append(record_id)
        if row.get("sequence_length") != SEQUENCE_TOKENS:
            raise PilotError("observation sequence length changed")
        if row.get("mask_digest") != mask_digest() or row.get("position_digest") != position_digest():
            raise PilotError("mask or position digest changed")
        if not isinstance(row.get("observation_digest"), str) or len(row["observation_digest"]) != 64:
            raise PilotError("observation digest is invalid")
    if observed_ids != expected_ids:
        raise PilotError("opaque record order changed")
    _validate_artifact_entry(index.get("observation"), label="observation")


def load_public_interface(
    input_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, Path, Path, Path]:
    """Load one condition-free public arm and verify all stage-local identities."""

    root = input_root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise PilotError("public input root must be a regular directory")
    config_path = root / "sanitized_config.json"
    index_path = root / "observation_index.json"
    config = load_json(config_path)
    index = load_json(index_path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("task_id") != TASK_ID
        or config.get("truth_opened") is not False
        or config.get("source_truth_included") is not False
        or "condition" in config
    ):
        raise PilotError("sanitized public config is not condition-free")
    if config.get("record_order") != [f"p01-r{position:04d}" for position in range(1, 17)]:
        raise PilotError("sanitized record order changed")
    if config.get("geometry") != {
        "records": 16,
        "sequence_tokens": SEQUENCE_TOKENS,
        "scored_tokens": SCORED_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "cut_depth": CUT_DEPTH,
    }:
        raise PilotError("sanitized geometry changed")
    validate_observation_index(index, records=16)
    obs_entry = _validate_artifact_entry(index["observation"], label="observation")
    obs_path = public_path(root, str(obs_entry["path"]))
    if obs_path.stat().st_size != int(obs_entry["bytes"]) or sha256_file(obs_path) != obs_entry["sha256"]:
        raise PilotError("observation artifact hash changed")
    with safe_open(obs_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"activations"}:
            raise PilotError("observation tensor fields changed")
        metadata = handle.metadata() or {}
        if metadata != {
            "schema": OBSERVATION_SCHEMA,
            "opaque_records": "true",
            "source_truth_included": "false",
        }:
            raise PilotError("observation metadata changed")
        observations = handle.get_tensor("activations")
    if tuple(observations.shape) != (16, SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise PilotError("observation tensor geometry changed")
    if not torch.isfinite(observations).all().item():
        raise PilotError("observation tensor contains non-finite values")
    for row, observation in zip(index["records"], observations):
        if row["observation_digest"] != observation_row_digest(observation):
            raise PilotError(f"observation digest changed for {row['record_id']}")
    return config, index, observations, config_path, index_path, obs_path


def validate_contiguous_observations(observations: torch.Tensor) -> None:
    if observations.ndim != 3 or tuple(observations.shape[1:]) != (SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise PilotError("observations must be [records, 40, 2048]")
    if not torch.isfinite(observations).all().item():
        raise PilotError("observations contain non-finite values")


def load_prediction_tensor(path: Path, *, keys: set[str]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise PilotError(f"prediction artifact must be a regular file: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != keys:
            raise PilotError("prediction tensor fields changed")
        metadata = handle.metadata() or {}
        expected = {
            "schema": PREDICTION_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": "false",
        }
        if metadata != expected:
            raise PilotError("prediction metadata or truth state changed")
        tensors = {key: handle.get_tensor(key) for key in keys}
    return tensors, metadata


def estimate_resource_need(
    *,
    table_bytes: int,
    model_bytes: int,
    query_rows: int = 256,
    prototype_chunk: int = 8192,
    hidden_size: int = HIDDEN_SIZE,
    safety_fraction: float = 0.80,
) -> dict[str, int | float]:
    """Conservative memory estimate used by build and reconstruction guards."""

    if table_bytes < 0 or model_bytes < 0 or query_rows <= 0 or prototype_chunk <= 0:
        raise PilotError("resource estimate inputs are invalid")
    # Two float32 matrices plus scores, a model working margin, and a 20%
    # allocator margin.  This is an estimate; the guard also probes a small
    # allocation immediately before the representative cell.
    lookup_bytes = query_rows * hidden_size * 4 + prototype_chunk * hidden_size * 4
    score_bytes = query_rows * prototype_chunk * 4
    working = lookup_bytes + score_bytes
    raw = int(table_bytes + model_bytes + working)
    return {
        "table_bytes": int(table_bytes),
        "model_bytes_estimate": int(model_bytes),
        "lookup_workspace_bytes": int(working),
        "raw_required_bytes": raw,
        "guard_required_bytes": int(raw / safety_fraction),
        "safety_fraction": float(safety_fraction),
    }


def _host_memory_bytes() -> tuple[int, int]:
    """Return Linux MemAvailable and MemTotal without allocating a probe."""

    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            fields = raw.strip().split()
            if fields and fields[0].isdigit():
                # /proc/meminfo reports kB for these fields.
                values[key] = int(fields[0]) * 1024
    except (OSError, UnicodeError, ValueError):
        return 0, 0
    return int(values.get("MemAvailable", 0)), int(values.get("MemTotal", 0))


def resource_guard(
    *,
    device: torch.device,
    required_bytes: int,
    allocation_bytes: int = 0,
) -> dict[str, Any]:
    """Fail closed using live host RAM and, for CUDA, selected-device memory.

    The guard intentionally performs no synthetic stress allocation.  The
    representative operation itself is the qualification cell; ``allocation_bytes``
    is retained as a backwards-compatible field and must remain zero.
    """

    if required_bytes <= 0 or allocation_bytes != 0:
        raise PilotError("resource guard requires positive bytes and no synthetic allocation")
    host_free, host_total = _host_memory_bytes()
    if host_free <= 0 or host_total <= 0:
        raise PilotError("resource guard failed closed: live host RAM is unavailable")
    if host_free < int(required_bytes):
        raise PilotError(
            f"resource guard failed closed on host RAM: free={host_free} required={required_bytes} total={host_total}"
        )
    result: dict[str, Any] = {
        "selected_device": str(device),
        "status": "PASS",
        "host_free_bytes_before": int(host_free),
        "host_total_bytes": int(host_total),
        "host_required_bytes": int(required_bytes),
        "required_bytes": int(required_bytes),
        "allocation_bytes": 0,
    }
    if device.type != "cuda":
        result["device_guard"] = "CPU_NO_CUDA_GUARD"
        return result
    if not torch.cuda.is_available():
        raise PilotError("CUDA resource guard requested but CUDA is unavailable")
    free, total = torch.cuda.mem_get_info(device)
    if int(free) < int(required_bytes):
        raise PilotError(
            f"resource guard failed closed: free={free} required={required_bytes} total={total}"
        )
    result.update(
        {
            "device_free_bytes_before": int(free),
            "device_total_bytes": int(total),
            "device_required_bytes": int(required_bytes),
            "device_guard": "PASS",
        }
    )
    return result
