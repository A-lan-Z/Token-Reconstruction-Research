"""Run the preregistered TRR-P12 evaluator-only target trajectory.

The trainer deliberately consumes only a source-worker tensor bundle.  It
never selects public rows, reads source text, or writes target weights.  The
full trajectory starts from the frozen Llama-3.2-1B-Instruct base and retains
the baseline binding plus LoRA adapters after 64, 128, and 256 updates.

``qualify`` is a throwaway 2--4 update check on an already-opened fixture;
``full`` is the retained trajectory.  ``--dry-run`` performs no model load,
training, or artifact creation and is useful for CPU-side command checks.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

# The source tree is intentionally used directly: this script is run from a
# checkout before a package installation is guaranteed.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from token_reconstruction.target_update import (
    TargetLoRAConfig,
    TargetUpdateError,
    install_target_lora,
    save_target_lora,
    target_lora_parameters,
)


TASK_ID = "TRR-P12"
SCHEMA = "token-reconstruction.trr-p12-target-trajectory.v1"
SOURCE_SCHEMA = "token-reconstruction.trr-p12-target-source-bundle.v1"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
BASE_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
BASE_WEIGHT_SHA256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"
BASE_TOKENIZER_JSON_SHA256 = "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
TRAIN_ROWS = 256
VALIDATION_ROWS = 64
SEQUENCE_LENGTH = 128
STAGES = (64, 128, 256)
MIN_BACKUP_FREE_BYTES = 10 * 1024**3
MODULES = ("q_proj", "v_proj")
LAYERS = (0, 1, 2, 3)
LORA_RANK = 8
LORA_ALPHA = 16.0
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0
BATCH_SIZE = 2
DEFAULT_QUALIFICATION_STEPS = 4


class TargetTrainError(RuntimeError):
    """Raised when a target trajectory cannot be run fail-closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path, *, label: str, allow_symlink: bool = False) -> dict[str, Any]:
    if path.is_symlink() and not allow_symlink:
        raise TargetTrainError(f"{label} must not be symlinked: {path}")
    if not path.is_file():
        raise TargetTrainError(f"{label} is unavailable: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "readonly": True,
    }


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise TargetTrainError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _relative_or_absolute(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TargetTrainError(f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise TargetTrainError(f"{label} escapes source bundle directory") from exc
    return path


def _checked_bundle_file(manifest_path: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TargetTrainError(f"{label} record must be an object")
    path = _relative_or_absolute(manifest_path.parent, record.get("path"), label=label)
    if path.is_symlink() or not path.is_file():
        raise TargetTrainError(f"{label} must be a regular file: {path}")
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise TargetTrainError(f"{label} SHA-256 is missing")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise TargetTrainError(f"{label} SHA-256 changed")
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
        raise TargetTrainError(f"{label} byte count changed")
    return path


def _load_tensor_file(path: Path, *, label: str) -> dict[str, torch.Tensor]:
    try:
        tensors = load_file(str(path), device="cpu")
    except Exception as exc:  # safetensors exposes several exception classes
        raise TargetTrainError(f"cannot load {label}") from exc
    if not isinstance(tensors, dict):
        raise TargetTrainError(f"{label} is not a tensor mapping")
    return tensors


def _validate_ids(value: torch.Tensor, *, label: str, rows: int) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (rows, SEQUENCE_LENGTH):
        raise TargetTrainError(f"{label} must have shape [{rows},{SEQUENCE_LENGTH}]")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TargetTrainError(f"{label} must be an integer tensor")
    value = value.to(dtype=torch.long)
    if bool(value.lt(0).any().item()) or bool(value.ge(VOCAB_SIZE).any().item()):
        raise TargetTrainError(f"{label} contains an out-of-vocabulary id")
    if bool(value[:, 0].ne(BOS_TOKEN_ID).any().item()):
        raise TargetTrainError(f"{label} does not have the fixed BOS token")
    return value


def _validate_mask(value: torch.Tensor, *, label: str, rows: int) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (rows, SEQUENCE_LENGTH):
        raise TargetTrainError(f"{label} must have shape [{rows},{SEQUENCE_LENGTH}]")
    if value.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TargetTrainError(f"{label} must be boolean/integer")
    value = value.to(dtype=torch.bool)
    if bool((~value[:, 0]).any().item()):
        raise TargetTrainError(f"{label} masks BOS")
    # Right padding is part of the prepared-bundle contract.  Any 0 -> 1
    # transition would make the loss geometry depend on an undocumented row.
    if bool((~value[:, :-1] & value[:, 1:]).any().item()):
        raise TargetTrainError(f"{label} is not right padded")
    return value


def load_source_bundle(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TargetTrainError(f"source bundle must be a regular JSON file: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetTrainError("source bundle is not valid JSON") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SOURCE_SCHEMA:
        raise TargetTrainError("source bundle schema is not the registered P12 schema")
    train_file = _checked_bundle_file(path, manifest.get("train_tensor"), label="train tensor")
    validation_file = _checked_bundle_file(path, manifest.get("validation_tensor"), label="validation tensor")
    train = _load_tensor_file(train_file, label="train tensor")
    validation = _load_tensor_file(validation_file, label="validation tensor")
    expected_train = {"train_input_ids", "train_attention_mask"}
    expected_validation = {"validation_input_ids", "validation_attention_mask"}
    if set(train) != expected_train or set(validation) != expected_validation:
        raise TargetTrainError("source tensor fields differ from the registered contract")
    train_ids = _validate_ids(train["train_input_ids"], label="train_input_ids", rows=TRAIN_ROWS)
    train_mask = _validate_mask(train["train_attention_mask"], label="train_attention_mask", rows=TRAIN_ROWS)
    validation_ids = _validate_ids(
        validation["validation_input_ids"], label="validation_input_ids", rows=VALIDATION_ROWS
    )
    validation_mask = _validate_mask(
        validation["validation_attention_mask"], label="validation_attention_mask", rows=VALIDATION_ROWS
    )
    return {
        "manifest": dict(manifest),
        "manifest_sha256": sha256_file(path),
        "manifest_binding": file_binding(path, label="source bundle"),
        "train_tensor_binding": file_binding(train_file, label="train tensor"),
        "validation_tensor_binding": file_binding(validation_file, label="validation tensor"),
        "train_input_ids": train_ids,
        "train_attention_mask": train_mask,
        "validation_input_ids": validation_ids,
        "validation_attention_mask": validation_mask,
    }


def _model_binding(snapshot: Path) -> dict[str, Any]:
    if not snapshot.is_dir():
        raise TargetTrainError(f"model snapshot directory is unavailable: {snapshot}")
    config_path = snapshot / "config.json"
    weight_path = snapshot / "model.safetensors"
    tokenizer_path = snapshot / "tokenizer.json"
    config = file_binding(config_path, label="base config", allow_symlink=True)
    weights = file_binding(weight_path, label="base model weights", allow_symlink=True)
    tokenizer = file_binding(tokenizer_path, label="base tokenizer", allow_symlink=True)
    if weights["sha256"] != BASE_WEIGHT_SHA256:
        raise TargetTrainError("base model weight SHA-256 differs from the registered frozen base")
    if tokenizer["sha256"] != BASE_TOKENIZER_JSON_SHA256:
        raise TargetTrainError("base tokenizer SHA-256 differs from the registered tokenizer")
    try:
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetTrainError("base config is not valid JSON") from exc
    if not isinstance(config_value, Mapping):
        raise TargetTrainError("base config is not an object")
    if int(config_value.get("vocab_size", -1)) != VOCAB_SIZE:
        raise TargetTrainError("base vocabulary size differs from the registered model")
    return {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "snapshot": str(snapshot.resolve()),
        "config": config,
        "weights": weights,
        "tokenizer_json": tokenizer,
    }


def trajectory_config(seed: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "layers": list(LAYERS),
        "modules": list(MODULES),
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "train_rows": TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "stages": list(STAGES),
        "stage_policy": "retain baseline and every fixed update checkpoint; no checkpoint selection or early stopping",
        "validation_policy": "post-run meaningfulness assessment only; never used for checkpoint selection",
        "target_scope": "layers 0-3 q_proj/v_proj LoRA; evaluator-only target prefix",
    }


def _resource_estimate() -> dict[str, Any]:
    modules = len(LAYERS) * len(MODULES)
    per_layer = LORA_RANK * (2048 + 2048) + LORA_RANK * (2048 + 512)
    params = len(LAYERS) * per_layer
    return {
        "lora_modules": modules,
        "lora_trainable_parameters": params,
        "q_proj_shape": [2048, 2048],
        "v_proj_shape": [512, 2048],
        "adapter_tensor_bytes_fp32": params * 4,
        "adapter_tensor_bytes_bf16_equivalent": params * 2,
        "expected_stage_artifact_bytes": "approximately 0.86 MB (safetensors metadata plus fp32 A/B tensors)",
        "base_weight_bytes": 2471645608,
        "peak_cuda_estimate_bytes": 3_600_000_000,
        "peak_cuda_estimate_basis": "P11 public calibration 2.673 GB allocated / 2.691 GB reserved at seq40,batch8, scaled conservatively to seq128,batch2; preflight expects approximately 12 GiB free on the 16.3 GiB desktop and retains a 2 GiB runtime guard",
        "qualification": "2-4 updates, same adapter geometry, throwaway output; repeated fixture loss is smoke-only and not scientific held-out validation",
    }


def plan_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "DRY_RUN_NO_MODEL_NO_TRAINING" if args.dry_run else "REGISTERED",
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "model": {
            "model_id": BASE_MODEL_ID,
            "revision": BASE_REVISION,
            "expected_weight_sha256": BASE_WEIGHT_SHA256,
            "expected_tokenizer_json_sha256": BASE_TOKENIZER_JSON_SHA256,
        },
        "trajectory": trajectory_config(args.seed),
        "resource_estimate": _resource_estimate(),
        "source_bundle": None if args.source_bundle is None else str(Path(args.source_bundle).resolve()),
        "output_root": None if args.output_root is None else str(Path(args.output_root).resolve()),
        "backup_root": None if args.backup_root is None else str(Path(args.backup_root).resolve()),
        "backup_policy": {
            "full_run_only": False,
            "create_only": True,
            "copy_and_rehash_before_next_stage": True,
            "minimum_free_bytes_after_copy": MIN_BACKUP_FREE_BYTES,
            "target_weights_evaluator_only": True,
        },
    }


def _prepare_output_root(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise TargetTrainError(f"output root already exists; create-only run refuses it: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _free_bytes(path: Path) -> int:
    probe = path if path.exists() else path.parent
    return int(shutil.disk_usage(probe).free)


def backup_and_verify(source: Path, backup_root: Path, *, stage: int) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise TargetTrainError(f"cannot back up non-regular stage artifact: {source}")
    if backup_root.is_symlink() or (backup_root.exists() and not backup_root.is_dir()):
        raise TargetTrainError(f"backup root must be a directory, not a symlink/file: {backup_root}")
    backup_root.mkdir(parents=True, exist_ok=True)
    before_free = _free_bytes(backup_root)
    if before_free < MIN_BACKUP_FREE_BYTES:
        raise TargetTrainError("backup volume has less than the required 10 GiB free")
    destination = backup_root / source.name
    if destination.exists() or destination.is_symlink():
        raise TargetTrainError(f"backup destination already exists: {destination}")
    shutil.copyfile(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash or source.stat().st_size != destination.stat().st_size:
        raise TargetTrainError(f"independent backup verification failed for stage {stage}")
    after_free = _free_bytes(backup_root)
    if after_free < MIN_BACKUP_FREE_BYTES:
        raise TargetTrainError("backup volume fell below the required 10 GiB free after copy")
    return {
        "stage_updates": stage,
        "source": file_binding(source, label="stage adapter"),
        "backup": file_binding(destination, label="stage backup"),
        "source_sha256": source_hash,
        "backup_sha256": destination_hash,
        "free_bytes_before": before_free,
        "free_bytes_after": after_free,
        "verified_at_utc": utc_now(),
    }


def _write_baseline(
    output_root: Path,
    *,
    model: dict[str, Any],
    source: dict[str, Any],
    seed: int,
    validation_lm_loss: float,
) -> None:
    write_json_create_only(
        output_root / "baseline.json",
        {
            "schema": f"{SCHEMA}.baseline",
            "task_id": TASK_ID,
            "stage_updates": 0,
            "target_weights": "evaluator-only; no decoder or reconstruction input",
            "model": model,
            "source_bundle": {
                "manifest": source["manifest_binding"],
                "manifest_sha256": source["manifest_sha256"],
            },
            "trajectory": trajectory_config(seed),
            "validation_lm_loss": float(validation_lm_loss),
            "created_at_utc": utc_now(),
        },
    )


def _tensor_digest(value: torch.Tensor) -> str:
    """Hash tensor bytes without printing or retaining tensor payloads."""

    cpu_value = value.detach().to(device="cpu").contiguous()
    raw = cpu_value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _frozen_parameter_snapshot(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter, str]]:
    """Capture references and byte digests before installing evaluator LoRA."""

    return [(name, parameter, _tensor_digest(parameter)) for name, parameter in model.named_parameters()]


def _frozen_parameter_checks(
    snapshot: list[tuple[str, torch.nn.Parameter, str]],
    installed: Mapping[str, torch.nn.Module],
) -> dict[str, Any]:
    adapter_ids = {id(parameter) for parameter in target_lora_parameters(installed.values())}
    changed: list[str] = []
    gradients: list[str] = []
    trainable_non_adapter: list[str] = []
    for name, parameter, digest in snapshot:
        if _tensor_digest(parameter) != digest:
            changed.append(name)
        if parameter.grad is not None:
            gradients.append(name)
        if parameter.requires_grad and id(parameter) not in adapter_ids:
            trainable_non_adapter.append(name)
    return {
        "frozen_base_parameter_count": len(snapshot),
        "frozen_base_digest_mismatches": changed,
        "frozen_base_gradients_present": gradients,
        "frozen_base_all_unchanged": not changed,
        "frozen_base_all_gradients_none": not gradients,
        "non_adapter_trainable_count": len(trainable_non_adapter),
        "adapter_parameter_count": len(adapter_ids),
        "optimizer_scope_check": len(trainable_non_adapter) == 0,
    }


def _effective_delta_metrics(installed: Mapping[str, torch.nn.Module]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    total_norm_sq = 0.0
    for name, module in installed.items():
        # LoRALinear stores B[out,rank] and A[rank,in].  This is the actual
        # effective delta used by its forward path, including alpha/rank.
        delta = (module.B.detach().float() @ module.A.detach().float()).mul(float(module.scale))
        norm = float(torch.linalg.vector_norm(delta).cpu().item())
        total_norm_sq += norm * norm
        metrics[name] = {
            "delta_shape": list(delta.shape),
            "alpha_over_rank": float(module.scale),
            "delta_l2": norm,
            "delta_max_abs": float(delta.abs().max().cpu().item()),
            "changed_fraction_atol_1e-12": float(delta.abs().gt(1e-12).float().mean().cpu().item()),
        }
    metrics["aggregate"] = {
        "delta_l2": total_norm_sq**0.5,
        "nonzero_effective_delta": total_norm_sq > 0.0,
    }
    return metrics


def _evaluate_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    cuda_min_free_bytes: int | None = None,
) -> float:
    """Compute fixed held-out NLL in B2 chunks with token-weighted reduction."""

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise TargetTrainError("validation tensors have incompatible geometry")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            for start in range(0, int(input_ids.shape[0]), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, int(input_ids.shape[0]))
                batch_sum, batch_tokens = _masked_loss_components(
                    model,
                    input_ids[start:stop],
                    attention_mask[start:stop],
                )
                total_loss += float(batch_sum.cpu().item())
                total_tokens += int(batch_tokens)
                if cuda_min_free_bytes is not None:
                    _cuda_guard(input_ids.device, minimum_free_bytes=cuda_min_free_bytes)
    finally:
        model.train(was_training)
    if total_tokens <= 0:
        raise TargetTrainError("held-out validation has no valid next-token labels")
    return total_loss / total_tokens


def _step_stats(
    *,
    device: torch.device,
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    elapsed_seconds: float,
    step: int,
    cuda_min_free_bytes: int,
    max_rss_bytes: int | None = None,
) -> dict[str, Any]:
    if not bool(torch.isfinite(loss.detach()).item()):
        raise TargetTrainError(f"non-finite loss at update {step}")
    grad_values = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grad_values or not all(bool(torch.isfinite(value).all().item()) for value in grad_values):
        raise TargetTrainError(f"missing or non-finite LoRA gradient at update {step}")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP).cpu().item())
    if not torch.isfinite(torch.tensor(grad_norm)):
        raise TargetTrainError(f"non-finite LoRA gradient norm at update {step}")
    stats: dict[str, Any] = {
        "update": step,
        "loss": float(loss.detach().cpu().item()),
        "gradient_norm_before_clip": grad_norm,
        "elapsed_seconds": float(elapsed_seconds),
        "finite_loss": True,
        "finite_gradients": True,
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        if int(free) < cuda_min_free_bytes:
            raise TargetTrainError(f"CUDA free memory guard tripped at update {step}")
        stats.update({
            "cuda_free_bytes": int(free),
            "cuda_total_bytes": int(total),
            "cuda_allocated_bytes": int(allocated),
            "cuda_reserved_bytes": int(reserved),
            "cuda_peak_allocated_bytes": int(peak_allocated),
            "cuda_peak_reserved_bytes": int(peak_reserved),
        })
    try:
        import resource

        stats["max_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except (ImportError, AttributeError):
        stats["max_rss_bytes"] = None
    if max_rss_bytes is not None and stats["max_rss_bytes"] is not None and stats["max_rss_bytes"] > max_rss_bytes:
        raise TargetTrainError(f"host RSS guard tripped at update {step}")
    return stats


def _finish_step_stats(
    stats: dict[str, Any],
    *,
    device: torch.device,
    step: int,
    started: float,
    backward_done: float,
    cuda_min_free_bytes: int,
    max_rss_bytes: int | None,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        if int(free) < cuda_min_free_bytes:
            raise TargetTrainError(f"CUDA free memory guard tripped after update {step}")
        stats.update({
            "cuda_free_bytes": int(free),
            "cuda_total_bytes": int(total),
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "resource_sample_phase": "post_optimizer_synchronize",
        })
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except (ImportError, AttributeError):
        rss = None
    if max_rss_bytes is not None and rss is not None and rss > max_rss_bytes:
        raise TargetTrainError(f"host RSS guard tripped after update {step}")
    stats["max_rss_bytes"] = rss
    stats["forward_backward_clip_seconds"] = float(backward_done - started)
    stats["optimizer_elapsed_seconds"] = float(time.monotonic() - backward_done)
    stats["elapsed_seconds"] = float(time.monotonic() - started)
    return stats


def _masked_loss_components(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    labels = input_ids[:, 1:].clone()
    label_mask = attention_mask[:, 1:]
    labels[~label_mask] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    if not bool(label_mask.any().item()):
        raise TargetTrainError("source batch has no valid next-token labels")
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    valid = labels.reshape(-1).ne(-100)
    return token_loss[valid].sum(), int(valid.sum().item())


def _masked_loss(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    total, count = _masked_loss_components(model, input_ids, attention_mask)
    return total / count


def _cuda_guard(device: torch.device, *, minimum_free_bytes: int) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    if int(free) < minimum_free_bytes:
        raise TargetTrainError(
            f"CUDA free memory {int(free)} is below guard {minimum_free_bytes}; refusing target run"
        )
    return {"free_bytes": int(free), "total_bytes": int(total)}


def _load_model(snapshot: Path, device: torch.device) -> torch.nn.Module:
    try:
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        }
        if device.type == "cuda":
            kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), **kwargs)
    except Exception as exc:
        raise TargetTrainError("could not load the frozen local base model") from exc
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _stage_record(
    output_root: Path,
    *,
    stage: int,
    adapter: Path,
    backup: dict[str, Any],
    source: dict[str, Any],
    model: dict[str, Any],
    seed: int,
    losses: list[float],
    validation_lm_loss: float,
    baseline_validation_lm_loss: float,
    effective_delta: dict[str, Any],
    frozen_checks: dict[str, Any],
    step_stats: list[dict[str, Any]],
) -> None:
    write_json_create_only(
        output_root / f"stage-{stage:04d}.json",
        {
            "schema": f"{SCHEMA}.stage",
            "task_id": TASK_ID,
            "stage_updates": stage,
            "adapter": file_binding(adapter, label="stage adapter"),
            "backup": backup,
            "model": model,
            "source_bundle": {
                "manifest_sha256": source["manifest_sha256"],
                "manifest": source["manifest_binding"],
            },
            "trajectory": trajectory_config(seed),
            "losses_by_update": [float(value) for value in losses],
            "step_stats": step_stats,
            "validation_lm_loss": float(validation_lm_loss),
            "effective_delta": effective_delta,
            "frozen_parameter_checks": frozen_checks,
            "meaningfulness": {
                "criterion": "held-out target LM loss lower than baseline and nonzero effective prefix delta",
                "pass": bool(validation_lm_loss < baseline_validation_lm_loss and effective_delta["aggregate"]["nonzero_effective_delta"]),
                "failure_policy": "report and terminate any extension; do not select, early-stop, or add stages",
            },
            "validation": "not used for training or checkpoint selection; post-run meaningfulness only",
            "created_at_utc": utc_now(),
        },
    )


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.source_bundle is None
        or args.model_snapshot is None
        or args.output_root is None
        or args.backup_root is None
    ):
        raise TargetTrainError(
            "qualification requires source bundle, model snapshot, output root, and backup root"
        )
    source = load_source_bundle(Path(args.source_bundle).resolve())
    model_path = Path(args.model_snapshot).resolve()
    model_binding = _model_binding(model_path)
    output_root = Path(args.output_root).resolve()
    backup_root = Path(args.backup_root).resolve()
    _prepare_output_root(output_root)
    device = torch.device(args.device)
    _cuda_guard(device, minimum_free_bytes=args.cuda_min_free_bytes)
    set_seed(args.seed)
    model = _load_model(model_path, device)
    validation_ids = source["validation_input_ids"].to(device)
    validation_mask = source["validation_attention_mask"].to(device)
    baseline_validation_lm_loss = _evaluate_loss(
        model,
        validation_ids,
        validation_mask,
        cuda_min_free_bytes=args.cuda_min_free_bytes,
    )
    frozen_snapshot = _frozen_parameter_snapshot(model)
    config = TargetLoRAConfig(
        layers=LAYERS,
        modules=MODULES,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        seed=args.seed,
    )
    installed = install_target_lora(model, config)
    adapter_parameters = target_lora_parameters(installed.values())
    optimizer = torch.optim.AdamW(adapter_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.train()
    train_ids = source["train_input_ids"].to(device)
    train_mask = source["train_attention_mask"].to(device)
    steps = max(2, min(int(args.steps), 4))
    losses: list[float] = []
    step_stats: list[dict[str, Any]] = []
    for update in range(steps):
        started = time.monotonic()
        start_row = (update * BATCH_SIZE) % TRAIN_ROWS
        batch_ids = train_ids[start_row : start_row + BATCH_SIZE]
        batch_mask = train_mask[start_row : start_row + BATCH_SIZE]
        optimizer.zero_grad(set_to_none=True)
        loss = _masked_loss(model, batch_ids, batch_mask)
        loss.backward()
        stats = _step_stats(
            device=device,
            loss=loss,
            parameters=adapter_parameters,
            elapsed_seconds=0.0,
            step=update + 1,
            cuda_min_free_bytes=args.cuda_min_free_bytes,
            max_rss_bytes=args.max_rss_bytes,
        )
        backward_done = time.monotonic()
        optimizer.step()
        step_stats.append(
            _finish_step_stats(
                stats,
                device=device,
                step=update + 1,
                started=started,
                backward_done=backward_done,
                cuda_min_free_bytes=args.cuda_min_free_bytes,
                max_rss_bytes=args.max_rss_bytes,
            )
        )
        losses.append(float(loss.detach().cpu().item()))
    qualification_validation_lm_loss = _evaluate_loss(
        model,
        validation_ids,
        validation_mask,
        cuda_min_free_bytes=args.cuda_min_free_bytes,
    )
    effective_delta = _effective_delta_metrics(installed)
    frozen_checks = _frozen_parameter_checks(frozen_snapshot, installed)
    adapter = output_root / f"target-lora-qualification-{steps:04d}.safetensors"
    save_target_lora(installed, adapter)
    backup = backup_and_verify(adapter, backup_root, stage=steps)
    receipt = {
        "schema": f"{SCHEMA}.qualification",
        "task_id": TASK_ID,
        "status": "QUALIFICATION_COMPLETE_THROWAWAY_ADAPTER",
        "steps": steps,
        "no_checkpoint_selection_or_early_stopping": True,
        "model": model_binding,
        "source_bundle": {
            "manifest": source["manifest_binding"],
            "manifest_sha256": source["manifest_sha256"],
        },
        "adapter": file_binding(adapter, label="qualification adapter"),
        "backup": backup,
        "trajectory": trajectory_config(args.seed),
        "losses_by_update": losses,
        "step_stats": step_stats,
        "baseline_validation_lm_loss": baseline_validation_lm_loss,
        "qualification_validation_lm_loss": qualification_validation_lm_loss,
        "validation_batch_size": BATCH_SIZE,
        "validation": "repeated rows from the same eight-row public fitting qualification fixture; smoke diagnostic only, not scientific held-out validation",
        "scientific_heldout_validation": False,
        "effective_delta": effective_delta,
        "frozen_parameter_checks": frozen_checks,
        "target_weights": "evaluator-only; qualification adapter is retained only for the B1 smoke capture",
        "created_at_utc": utc_now(),
    }
    write_json_create_only(output_root / "qualification.json", receipt)
    return receipt


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.source_bundle is None
        or args.model_snapshot is None
        or args.output_root is None
        or args.backup_root is None
    ):
        raise TargetTrainError(
            "full run requires source bundle, model snapshot, output root, and backup root"
        )
    source = load_source_bundle(Path(args.source_bundle).resolve())
    model_path = Path(args.model_snapshot).resolve()
    model_binding = _model_binding(model_path)
    output_root = Path(args.output_root).resolve()
    backup_root = Path(args.backup_root).resolve()
    _prepare_output_root(output_root)
    device = torch.device(args.device)
    memory_before = _cuda_guard(device, minimum_free_bytes=args.cuda_min_free_bytes)
    set_seed(args.seed)
    model = _load_model(model_path, device)
    memory_after_load = _cuda_guard(device, minimum_free_bytes=args.cuda_min_free_bytes)
    validation_ids = source["validation_input_ids"].to(device)
    validation_mask = source["validation_attention_mask"].to(device)
    baseline_validation_lm_loss = _evaluate_loss(
        model,
        validation_ids,
        validation_mask,
        cuda_min_free_bytes=args.cuda_min_free_bytes,
    )
    frozen_snapshot = _frozen_parameter_snapshot(model)
    _write_baseline(
        output_root,
        model=model_binding,
        source=source,
        seed=args.seed,
        validation_lm_loss=baseline_validation_lm_loss,
    )
    config = TargetLoRAConfig(
        layers=LAYERS,
        modules=MODULES,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        seed=args.seed,
    )
    installed = install_target_lora(model, config)
    adapter_parameters = target_lora_parameters(installed.values())
    optimizer = torch.optim.AdamW(adapter_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.train()
    train_ids = source["train_input_ids"].to(device)
    train_mask = source["train_attention_mask"].to(device)
    losses: list[float] = []
    step_stats: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    for update in range(1, max(STAGES) + 1):
        started = time.monotonic()
        start_row = ((update - 1) * BATCH_SIZE) % TRAIN_ROWS
        batch_ids = train_ids[start_row : start_row + BATCH_SIZE]
        batch_mask = train_mask[start_row : start_row + BATCH_SIZE]
        optimizer.zero_grad(set_to_none=True)
        loss = _masked_loss(model, batch_ids, batch_mask)
        loss.backward()
        stats = _step_stats(
            device=device,
            loss=loss,
            parameters=adapter_parameters,
            elapsed_seconds=0.0,
            step=update,
            cuda_min_free_bytes=args.cuda_min_free_bytes,
            max_rss_bytes=args.max_rss_bytes,
        )
        backward_done = time.monotonic()
        optimizer.step()
        step_stats.append(
            _finish_step_stats(
                stats,
                device=device,
                step=update,
                started=started,
                backward_done=backward_done,
                cuda_min_free_bytes=args.cuda_min_free_bytes,
                max_rss_bytes=args.max_rss_bytes,
            )
        )
        losses.append(float(loss.detach().cpu().item()))
        if update in STAGES:
            adapter = output_root / f"target-lora-stage-{update:04d}.safetensors"
            save_target_lora(installed, adapter)
            backup = backup_and_verify(adapter, backup_root, stage=update)
            validation_lm_loss = _evaluate_loss(
                model,
                validation_ids,
                validation_mask,
                cuda_min_free_bytes=args.cuda_min_free_bytes,
            )
            effective_delta = _effective_delta_metrics(installed)
            frozen_checks = _frozen_parameter_checks(frozen_snapshot, installed)
            _stage_record(
                output_root,
                stage=update,
                adapter=adapter,
                backup=backup,
                source=source,
                model=model_binding,
                seed=args.seed,
                losses=losses,
                validation_lm_loss=validation_lm_loss,
                baseline_validation_lm_loss=baseline_validation_lm_loss,
                effective_delta=effective_delta,
                frozen_checks=frozen_checks,
                step_stats=step_stats,
            )
            stage_records.append(
                {
                    "stage_updates": update,
                    "adapter": file_binding(adapter, label="stage adapter"),
                    "backup": backup,
                    "validation_lm_loss": validation_lm_loss,
                    "effective_delta": effective_delta,
                    "frozen_parameter_checks": frozen_checks,
                }
            )
    receipt = {
        "schema": f"{SCHEMA}.run",
        "task_id": TASK_ID,
        "status": "FULL_TRAJECTORY_COMPLETE",
        "model": model_binding,
        "source_bundle": {
            "manifest": source["manifest_binding"],
            "manifest_sha256": source["manifest_sha256"],
        },
        "trajectory": trajectory_config(args.seed),
        "memory_preflight": {
            "before_model_load": memory_before,
            "after_model_load": memory_after_load,
            "runtime_guard_min_free_bytes": args.cuda_min_free_bytes,
            "runtime_guard_max_rss_bytes": args.max_rss_bytes,
            "validation_batch_size": BATCH_SIZE,
        },
        "baseline_validation_lm_loss": baseline_validation_lm_loss,
        "stage_records": stage_records,
        "step_stats": step_stats,
        "validation": "held-out 64 rows scored at baseline and every retained stage; never used for training or checkpoint selection",
        "meaningfulness": {
            "criterion": "each stage is described against baseline target LM loss and nonzero effective prefix delta",
            "failure_policy": "report a weak/failing stage and terminate any extension; do not select, early-stop, or add stages",
        },
        "target_weights": "evaluator-only; do not provide adapters or target logits to reconstruction",
        "created_at_utc": utc_now(),
    }
    write_json_create_only(output_root / "run.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("qualify", "full"):
        sub = subparsers.add_parser(mode, help=f"run the {mode} trajectory")
        sub.add_argument("--source-bundle", type=Path)
        sub.add_argument("--model-snapshot", type=Path)
        sub.add_argument("--output-root", type=Path)
        sub.add_argument("--backup-root", type=Path)
        sub.add_argument("--seed", type=int, default=6012)
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        sub.add_argument("--cuda-min-free-bytes", type=int, default=2 * 1024**3)
        sub.add_argument("--max-rss-bytes", type=int, default=12 * 1024**3)
        sub.add_argument("--steps", type=int, default=DEFAULT_QUALIFICATION_STEPS)
        sub.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(plan_record(args), indent=2, sort_keys=True))
        return 0
    try:
        receipt = run_qualification(args) if args.mode == "qualify" else run_full(args)
    except (TargetTrainError, TargetUpdateError, OSError, RuntimeError, ValueError) as exc:
        print(f"target trajectory refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": receipt.get("status"), "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
