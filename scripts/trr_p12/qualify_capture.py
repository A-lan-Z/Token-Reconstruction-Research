#!/usr/bin/env python3
"""Capture the bounded TRR-P12 B1 qualification observations.

This harness is evaluator-only.  It consumes the already-built qualification
fixture, keeps token IDs in process memory, and writes only sanitized BF16
cut-4 observations, masks, positions, and a hash-bound receipt.  Each
condition is run through the P11 public-prefix path as one complete batch of
8 records padded to 192 tokens; the first 128 positions are retained.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from token_reconstruction.public_activation import (
    PaddedTokenBatch,
    capture_public_prefix,
    pad_public_token_sequences,
    validate_activation_tensor,
)
from token_reconstruction.public_prefix import ContiguousPublicPrefix
from token_reconstruction.target_update import (
    TargetLoRAConfig,
    TargetUpdateError,
    install_target_lora,
    load_target_lora,
)


TASK_ID = "TRR-P12"
CAPTURE_SCHEMA = "token-reconstruction.trr-p12-qualification-capture.v1"
SOURCE_SCHEMA = "token-reconstruction.trr-p12-target-source-bundle.v1"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
BASE_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
BASE_WEIGHT_SHA256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"
BASE_WEIGHT_BYTES = 2471645608
BASE_CONFIG_SHA256 = "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"
BASE_TOKENIZER_SHA256 = "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"
BASE_GENERATION_SHA256 = "88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36"
SOURCE_ARTIFACT_SHA256 = "d1c78fcf1acc91b57d51355ee11f267bf4c12f1bc7d5160164b3b6ea11b45344"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
CUT_DEPTH = 4
FULL_SEQUENCE_TOKENS = 192
STORED_SEQUENCE_TOKENS = 128
CAPTURE_RECORDS = 8
CAPTURE_BATCH_RECORDS = 8
MAX_RESERVED_GPU_BYTES = 8 * 1024**3
MIN_FREE_GPU_BYTES = 8 * 1024**3
MAX_HOST_RSS_BYTES = 16 * 1024**3
DEFAULT_TIMEOUT_SECONDS = 600

DEFAULT_FIXTURE = REPOSITORY_ROOT / "outputs" / "TRR-P12" / "qualification-fixture-r1" / "source-bundle.json"
DEFAULT_MODEL = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
) / "snapshots" / BASE_REVISION
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "TRR-P12" / "qualification-capture-r1"


class QualificationCaptureError(RuntimeError):
    """Raised when the bounded capture contract cannot be established."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    path = path.expanduser()
    if not path.is_file():
        raise QualificationCaptureError(f"file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    allow_symlink: bool = True,
) -> dict[str, Any]:
    original = path.expanduser()
    if original.is_symlink() and not allow_symlink:
        raise QualificationCaptureError(f"{label} cannot be a symlink: {original}")
    resolved = original.resolve()
    if not resolved.is_file():
        raise QualificationCaptureError(f"{label} is unavailable: {original}")
    actual_bytes = int(resolved.stat().st_size)
    actual_sha256 = sha256_file(resolved)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise QualificationCaptureError(f"{label} SHA-256 changed")
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise QualificationCaptureError(f"{label} byte count changed")
    return {
        "path": str(original.absolute()),
        "resolved_path": str(resolved),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "symlink": original.is_symlink(),
        "readonly": True,
    }


def resolve_path(value: str | Path, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationCaptureError(f"{label} must be a regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationCaptureError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise QualificationCaptureError(f"{label} must contain an object")
    return value


def checked_bundle_file(bundle_path: Path, record: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise QualificationCaptureError(f"{label} binding is missing")
    value = record.get("path")
    if not isinstance(value, str) or not value:
        raise QualificationCaptureError(f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = bundle_path.parent / path
    path = path.resolve()
    try:
        path.relative_to(bundle_path.parent.resolve())
    except ValueError as exc:
        raise QualificationCaptureError(f"{label} escapes the fixture directory") from exc
    expected_sha256 = record.get("sha256")
    expected_bytes = record.get("bytes")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise QualificationCaptureError(f"{label} SHA-256 is missing")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        raise QualificationCaptureError(f"{label} byte count is missing")
    binding = file_binding(
        path,
        label=label,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        allow_symlink=False,
    )
    return path, binding


def _validate_fixture_fields(bundle_path: Path) -> tuple[PaddedTokenBatch, dict[str, Any]]:
    manifest = read_json(bundle_path, label="qualification source bundle")
    if manifest.get("schema") != SOURCE_SCHEMA:
        raise QualificationCaptureError("fixture source bundle schema changed")
    if manifest.get("task_id") != TASK_ID or manifest.get("status") != "QUALIFICATION_FIXTURE_ONLY":
        raise QualificationCaptureError("fixture is not the registered qualification-only bundle")
    selection = manifest.get("selection")
    provenance = manifest.get("provenance")
    if not isinstance(selection, Mapping) or not isinstance(provenance, Mapping):
        raise QualificationCaptureError("fixture selection/provenance binding is missing")
    if selection.get("public_fitting_fixture") is not True or selection.get("fresh_evaluation_data") is not False:
        raise QualificationCaptureError("fixture is not bound to the public fitting data")
    if provenance.get("public_fitting_data") is not True or provenance.get("fresh_evaluation_data") is not False:
        raise QualificationCaptureError("fixture provenance is not public fitting data")
    if provenance.get("p03_data") is not False:
        raise QualificationCaptureError("fixture provenance is not explicitly P03-free")
    source_artifact = provenance.get("source_artifact")
    if not isinstance(source_artifact, Mapping) or source_artifact.get("sha256") != SOURCE_ARTIFACT_SHA256:
        raise QualificationCaptureError("fixture source artifact binding changed")
    validation_path, validation_binding = checked_bundle_file(
        bundle_path,
        manifest.get("validation_tensor"),
        label="fixture validation tensor",
    )
    try:
        with safe_open(str(validation_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"validation_input_ids", "validation_attention_mask"}:
                raise QualificationCaptureError("fixture validation tensor fields changed")
            input_ids = handle.get_tensor("validation_input_ids")
            attention_mask = handle.get_tensor("validation_attention_mask")
    except QualificationCaptureError:
        raise
    except Exception as exc:
        raise QualificationCaptureError("cannot load fixture validation tensor") from exc
    if tuple(input_ids.shape) != (64, STORED_SEQUENCE_TOKENS) or tuple(attention_mask.shape) != (64, STORED_SEQUENCE_TOKENS):
        raise QualificationCaptureError("fixture validation geometry changed")
    if input_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise QualificationCaptureError("fixture validation IDs are not integer-valued")
    if attention_mask.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise QualificationCaptureError("fixture validation mask is not boolean/integer-valued")
    input_ids = input_ids[:CAPTURE_RECORDS].to(dtype=torch.long)
    attention_mask = attention_mask[:CAPTURE_RECORDS].to(dtype=torch.bool)
    if bool(input_ids.lt(0).any().item()) or bool(input_ids.ge(VOCAB_SIZE).any().item()):
        raise QualificationCaptureError("fixture validation IDs are outside the public vocabulary")
    if bool(input_ids[:, 0].ne(BOS_TOKEN_ID).any().item()):
        raise QualificationCaptureError("fixture validation rows do not retain BOS")
    if bool((~attention_mask[:, 0]).any().item()):
        raise QualificationCaptureError("fixture validation masks BOS")
    if bool((~attention_mask[:, :-1] & attention_mask[:, 1:]).any().item()):
        raise QualificationCaptureError("fixture validation masks are not right padded")
    sequences: list[list[int]] = []
    active_lengths: list[int] = []
    for row in range(CAPTURE_RECORDS):
        active_count = int(attention_mask[row].sum().item())
        if active_count < 2:
            raise QualificationCaptureError("fixture validation row is shorter than BOS plus one token")
        active_lengths.append(active_count)
        sequences.append(input_ids[row, :active_count].tolist())
    token_batch = pad_public_token_sequences(
        sequences,
        maximum_tokens=FULL_SEQUENCE_TOKENS,
        pad_token_id=PAD_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        small_post_bos_positions=5000,
    )
    if tuple(token_batch.token_ids.shape) != (CAPTURE_RECORDS, FULL_SEQUENCE_TOKENS):
        raise QualificationCaptureError("padded qualification geometry changed")
    return token_batch, {
        "source_bundle": file_binding(bundle_path, label="qualification source bundle", allow_symlink=False),
        "validation_tensor": validation_binding,
        "records_used": CAPTURE_RECORDS,
        "record_rule": "first 8 rows of the already-built qualification validation tensor",
        "active_lengths": active_lengths,
        "token_ids_written": False,
        "source_text_written": False,
        "source_payload_written": False,
    }


def bind_base_snapshot(snapshot: Path) -> dict[str, Any]:
    if not snapshot.is_dir():
        raise QualificationCaptureError(f"base snapshot directory is unavailable: {snapshot}")
    if snapshot.name != BASE_REVISION:
        raise QualificationCaptureError("base snapshot revision directory differs from the registered revision")
    files = {
        "config": file_binding(snapshot / "config.json", label="base config", expected_sha256=BASE_CONFIG_SHA256),
        "generation_config": file_binding(
            snapshot / "generation_config.json",
            label="base generation config",
            expected_sha256=BASE_GENERATION_SHA256,
        ),
        "tokenizer": file_binding(
            snapshot / "tokenizer.json",
            label="base tokenizer",
            expected_sha256=BASE_TOKENIZER_SHA256,
        ),
        "weights": file_binding(
            snapshot / "model.safetensors",
            label="base weights",
            expected_sha256=BASE_WEIGHT_SHA256,
            expected_bytes=BASE_WEIGHT_BYTES,
        ),
    }
    return {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "snapshot_path": str(snapshot.resolve()),
        "files": files,
        "weights_hash_verified_before_model_load": True,
        "hash_scope": "full model.safetensors sequential read",
    }


def adapter_binding(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"present": False, "target_weights": "none loaded"}
    binding = file_binding(path, label="qualification adapter", allow_symlink=False)
    return {
        "present": True,
        "binding": binding,
        "target_weights": "evaluator-only; adapter is read for capture and never written to output",
        "config": {
            "layers": [0, 1, 2, 3],
            "modules": ["q_proj", "v_proj"],
            "rank": 8,
            "alpha": 16.0,
            "seed": 6012,
        },
    }


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value * 1024 * 1024
    return value * 1024


def resource_snapshot(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {"host_rss_bytes": rss_bytes()}
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result.update(
            {
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
    return result


def guard_resources(
    device: torch.device,
    *,
    phase: str,
    started: float,
    timeout_seconds: float,
    min_free_gpu_bytes: int,
    max_reserved_gpu_bytes: int,
    max_rss_bytes: int,
) -> dict[str, Any]:
    if time.monotonic() - started > timeout_seconds:
        raise QualificationCaptureError(f"capture timeout exceeded during {phase}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise QualificationCaptureError("CUDA was requested but is unavailable")
        snapshot = resource_snapshot(device)
        if snapshot["cuda_free_bytes"] < min_free_gpu_bytes:
            raise QualificationCaptureError(f"CUDA free-memory guard failed during {phase}")
        if snapshot["cuda_reserved_bytes"] > max_reserved_gpu_bytes:
            raise QualificationCaptureError(f"CUDA reserved-memory guard failed during {phase}")
    else:
        snapshot = resource_snapshot(device)
    if snapshot["host_rss_bytes"] > max_rss_bytes:
        raise QualificationCaptureError(f"host RSS guard failed during {phase}")
    snapshot["phase"] = phase
    return snapshot


def load_prefix(
    snapshot: Path,
    *,
    device: torch.device,
    adapter: Path | None,
) -> ContiguousPublicPrefix:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
    except Exception as exc:
        raise QualificationCaptureError("base model load failed") from exc
    if int(model.config.hidden_size) != HIDDEN_SIZE or int(model.config.vocab_size) != VOCAB_SIZE:
        raise QualificationCaptureError("base model geometry changed")
    if int(model.config.num_hidden_layers) <= CUT_DEPTH:
        raise QualificationCaptureError("cut depth leaves no downstream layer")
    model.requires_grad_(False)
    if adapter is not None:
        try:
            installed = install_target_lora(
                model,
                TargetLoRAConfig(
                    layers=(0, 1, 2, 3),
                    modules=("q_proj", "v_proj"),
                    rank=8,
                    alpha=16.0,
                    seed=6012,
                ),
            )
            load_target_lora(installed, adapter)
        except (TargetUpdateError, RuntimeError, OSError) as exc:
            raise QualificationCaptureError("qualification adapter installation failed") from exc
        for module in installed.values():
            module.A.requires_grad_(False)
            module.B.requires_grad_(False)
    try:
        prefix = ContiguousPublicPrefix(model, cut_depth=CUT_DEPTH).to(device).eval()
    except Exception as exc:
        raise QualificationCaptureError("cut-4 public prefix construction failed") from exc
    del model
    gc.collect()
    return prefix


def save_sanitized_condition(
    *,
    output_path: Path,
    condition: str,
    activations: torch.Tensor,
    token_batch: PaddedTokenBatch,
) -> dict[str, Any]:
    if tuple(activations.shape) != (CAPTURE_RECORDS, FULL_SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise QualificationCaptureError("full capture geometry changed")
    if activations.dtype != torch.bfloat16:
        raise QualificationCaptureError("full capture dtype changed")
    validate_activation_tensor(activations, token_batch, hidden_size=HIDDEN_SIZE, require_bfloat16=True)
    compact = activations[:, :STORED_SEQUENCE_TOKENS].detach().cpu().contiguous()
    mask = token_batch.attention_mask[:, :STORED_SEQUENCE_TOKENS].detach().cpu().contiguous()
    positions = token_batch.position_ids[:, :STORED_SEQUENCE_TOKENS].detach().cpu().contiguous()
    if tuple(compact.shape) != (CAPTURE_RECORDS, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise QualificationCaptureError("stored activation geometry changed")
    if not torch.isfinite(compact.float()).all().item():
        raise QualificationCaptureError("stored activation contains non-finite values")
    for row in range(CAPTURE_RECORDS):
        active_count = int(mask[row].sum().item())
        if active_count < STORED_SEQUENCE_TOKENS and not compact[row, active_count:].eq(0).all().item():
            raise QualificationCaptureError("stored padded activations are not zero")
        if not torch.equal(positions[row, :active_count], torch.arange(active_count, dtype=torch.int64)):
            raise QualificationCaptureError("stored position IDs are not contiguous")
        if active_count < STORED_SEQUENCE_TOKENS and not positions[row, active_count:].eq(0).all().item():
            raise QualificationCaptureError("stored padded position IDs are not zero")
    if output_path.exists() or output_path.is_symlink():
        raise QualificationCaptureError(f"condition output is create-only: {output_path}")
    save_file(
        {
            "activations": compact,
            "attention_mask": mask,
            "position_ids": positions,
        },
        str(output_path),
        metadata={
            "schema": CAPTURE_SCHEMA,
            "task_id": TASK_ID,
            "condition": condition,
            "cut_depth": str(CUT_DEPTH),
            "capture_batch_records": str(CAPTURE_BATCH_RECORDS),
            "capture_sequence_tokens": str(FULL_SEQUENCE_TOKENS),
            "stored_sequence_tokens": str(STORED_SEQUENCE_TOKENS),
            "hidden_size": str(HIDDEN_SIZE),
            "activation_dtype": "bfloat16",
            "public_full_forward": "true",
            "token_ids_written": "false",
            "source_text_written": "false",
            "target_weights_written": "false",
        },
    )
    return {
        "condition": condition,
        "path": str(output_path.resolve()),
        "bytes": int(output_path.stat().st_size),
        "sha256": sha256_file(output_path),
        "keys": ["activations", "attention_mask", "position_ids"],
        "activation_shape": list(compact.shape),
        "activation_dtype": str(compact.dtype).replace("torch.", ""),
        "attention_mask_shape": list(mask.shape),
        "attention_mask_dtype": str(mask.dtype).replace("torch.", ""),
        "position_ids_shape": list(positions.shape),
        "position_ids_dtype": str(positions.dtype).replace("torch.", ""),
        "token_ids_written": False,
        "source_text_written": False,
        "target_weights_written": False,
    }


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise QualificationCaptureError(f"receipt is create-only: {path}")
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    fixture_path = resolve_path(args.fixture, root=REPOSITORY_ROOT)
    model_path = resolve_path(args.model_snapshot, root=REPOSITORY_ROOT)
    output_root = resolve_path(args.output_root, root=REPOSITORY_ROOT)
    adapter_path = None if args.adapter is None else resolve_path(args.adapter, root=REPOSITORY_ROOT)
    allowed_output_root = (REPOSITORY_ROOT / "outputs" / "TRR-P12").resolve()
    try:
        output_root.relative_to(allowed_output_root)
    except ValueError as exc:
        raise QualificationCaptureError("capture output must be below outputs/TRR-P12") from exc
    if output_root.exists() or output_root.is_symlink():
        raise QualificationCaptureError(f"capture output root is create-only: {output_root}")
    token_batch, fixture_binding = _validate_fixture_fields(fixture_path)
    model_binding = bind_base_snapshot(model_path)
    adapter_record = adapter_binding(adapter_path)
    if args.dry_run:
        return {
            "status": "READY_NO_MODEL_LOAD",
            "task_id": TASK_ID,
            "fixture": fixture_binding,
            "model": model_binding,
            "adapter": adapter_record,
            "capture_contract": {
                "full_forward": "ContiguousPublicPrefix.forward_full",
                "padded_shape": [CAPTURE_RECORDS, FULL_SEQUENCE_TOKENS],
                "stored_shape": [CAPTURE_RECORDS, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE],
                "cut_depth": CUT_DEPTH,
                "batch_records": CAPTURE_BATCH_RECORDS,
                "no_microbatch_substitution": True,
            },
            "output_root": str(output_root),
        }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise QualificationCaptureError("CUDA was requested but is unavailable")
    started_wall = time.monotonic()
    started_utc = utc_now()
    output_root.mkdir(parents=True, exist_ok=False)
    conditions: list[tuple[str, Path | None]] = [("baseline", None)]
    if adapter_path is not None:
        conditions.append(("qualification", adapter_path))
    output_bindings: list[dict[str, Any]] = []
    phase_records: list[dict[str, Any]] = []
    initial_resource = guard_resources(
        device,
        phase="pre_model_load",
        started=started_wall,
        timeout_seconds=float(args.timeout_seconds),
        min_free_gpu_bytes=int(args.min_free_gpu_bytes),
        max_reserved_gpu_bytes=int(args.max_reserved_gpu_bytes),
        max_rss_bytes=int(args.max_rss_bytes),
    )
    for condition, condition_adapter in conditions:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        phase_started = time.monotonic()
        guard_resources(
            device,
            phase=f"before_{condition}_model_load",
            started=started_wall,
            timeout_seconds=float(args.timeout_seconds),
            min_free_gpu_bytes=int(args.min_free_gpu_bytes),
            max_reserved_gpu_bytes=int(args.max_reserved_gpu_bytes),
            max_rss_bytes=int(args.max_rss_bytes),
        )
        prefix = load_prefix(model_path, device=device, adapter=condition_adapter)
        after_load = guard_resources(
            device,
            phase=f"after_{condition}_model_load",
            started=started_wall,
            timeout_seconds=float(args.timeout_seconds),
            min_free_gpu_bytes=int(args.min_free_gpu_bytes),
            max_reserved_gpu_bytes=int(args.max_reserved_gpu_bytes),
            max_rss_bytes=int(args.max_rss_bytes),
        )
        after_load_time = time.monotonic()
        capture_started = after_load_time
        activations = capture_public_prefix(
            prefix,
            token_batch,
            device=device,
            batch_size=CAPTURE_BATCH_RECORDS,
            hidden_size=HIDDEN_SIZE,
            resource_check=lambda: guard_resources(
                device,
                phase=f"during_{condition}_forward",
                started=started_wall,
                timeout_seconds=float(args.timeout_seconds),
                min_free_gpu_bytes=int(args.min_free_gpu_bytes),
                max_reserved_gpu_bytes=int(args.max_reserved_gpu_bytes),
                max_rss_bytes=int(args.max_rss_bytes),
            ),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        after_capture = guard_resources(
            device,
            phase=f"after_{condition}_capture",
            started=started_wall,
            timeout_seconds=float(args.timeout_seconds),
            min_free_gpu_bytes=int(args.min_free_gpu_bytes),
            max_reserved_gpu_bytes=int(args.max_reserved_gpu_bytes),
            max_rss_bytes=int(args.max_rss_bytes),
        )
        capture_finished_time = time.monotonic()
        output_path = output_root / f"{condition}.safetensors"
        output_bindings.append(
            save_sanitized_condition(
                output_path=output_path,
                condition=condition,
                activations=activations,
                token_batch=token_batch,
            )
        )
        phase_records.append(
            {
                "condition": condition,
                "adapter_present": condition_adapter is not None,
                "model_load_seconds": after_load_time - phase_started,
                "capture_seconds": capture_finished_time - capture_started,
                "after_model_load_resource": after_load,
                "after_capture_resource": after_capture,
            }
        )
        del activations, prefix
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        guard_resources(
            device,
            phase=f"after_{condition}_release",
            started=started_wall,
            timeout_seconds=float(args.timeout_seconds),
            min_free_gpu_bytes=int(args.min_free_gpu_bytes),
            max_reserved_gpu_bytes=int(args.max_reserved_gpu_bytes),
            max_rss_bytes=int(args.max_rss_bytes),
        )
    ended_utc = utc_now()
    receipt = {
        "schema": CAPTURE_SCHEMA,
        "task_id": TASK_ID,
        "status": "QUALIFICATION_CAPTURE_COMPLETE",
        "created_utc": ended_utc,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "wall_seconds": time.monotonic() - started_wall,
        "fixture": fixture_binding,
        "model": model_binding,
        "adapter": adapter_record,
        "capture_contract": {
            "forward_path": "token_reconstruction.public_activation.capture_public_prefix -> ContiguousPublicPrefix.forward_full",
            "padded_shape": [CAPTURE_RECORDS, FULL_SEQUENCE_TOKENS],
            "full_activation_shape": [CAPTURE_RECORDS, FULL_SEQUENCE_TOKENS, HIDDEN_SIZE],
            "stored_activation_shape": [CAPTURE_RECORDS, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE],
            "cut_depth": CUT_DEPTH,
            "capture_batch_records": CAPTURE_BATCH_RECORDS,
            "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
            "activation_dtype": "bfloat16",
            "pad_token_id": PAD_TOKEN_ID,
            "bos_token_id": BOS_TOKEN_ID,
            "no_microbatch_substitution": True,
            "same_fixture_records_for_all_conditions": True,
        },
        "resource_guards": {
            "min_free_gpu_bytes": int(args.min_free_gpu_bytes),
            "max_reserved_gpu_bytes": int(args.max_reserved_gpu_bytes),
            "max_rss_bytes": int(args.max_rss_bytes),
            "timeout_seconds": float(args.timeout_seconds),
            "initial_resource": initial_resource,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "conditions": phase_records,
        "outputs": output_bindings,
        "sanitization": {
            "token_ids_written": False,
            "source_text_written": False,
            "target_weights_written": False,
            "evaluator_only": True,
        },
    }
    receipt_binding = write_json_create_only(output_root / "capture-receipt.json", receipt)
    result = {
        "status": receipt["status"],
        "output_root": str(output_root),
        "outputs": output_bindings,
        "receipt": receipt_binding,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-free-gpu-bytes", type=int, default=MIN_FREE_GPU_BYTES)
    parser.add_argument("--max-reserved-gpu-bytes", type=int, default=MAX_RESERVED_GPU_BYTES)
    parser.add_argument("--max-rss-bytes", type=int, default=MAX_HOST_RSS_BYTES)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="verify fixture/base/adapter bindings without loading a model")
    args = parser.parse_args()
    try:
        result = run_capture(args)
    except (QualificationCaptureError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
