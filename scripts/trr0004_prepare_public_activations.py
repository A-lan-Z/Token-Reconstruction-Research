#!/usr/bin/env python3
"""Prepare public cut-4 activation tensors for the TRR-0004 controlled fit.

This is a public-data preparation command.  It materializes the registered
Alpaca fit and validation labels, runs only the pinned public prefix, and writes
BF16 activations with current-token aligned IDs and masks.  It does not load a
target checkpoint, use evaluator-private truth, or create confirmation data.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_reconstruction.alpaca_split import (
    ALPACA_CACHE_REVISION,
    DEFAULT_BOS_TOKEN_ID,
    HISTORICAL_MAX_TOKENS,
    HISTORICAL_MIN_FULL_TOKENS,
)
from token_reconstruction.public_activation import (
    CUT_DEPTH,
    HIDDEN_SIZE,
    PAD_TOKEN_ID,
    PUBLIC_ACTIVATION_SCHEMA,
    PublicActivationError,
    PaddedTokenBatch,
    capture_public_prefix,
    make_artifact_metadata,
    materialize_plan_split,
    pad_public_token_sequences,
    record_ids_sha256,
    save_public_artifact,
    tensor_sha256,
    validate_activation_tensor,
    validate_padded_token_batch,
)
from token_reconstruction.public_prefix import ContiguousPublicPrefix


TASK_ID = "TRR-0004"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
PLAN_SCHEMA = "token-reconstruction.trr0004-public-alpaca-split.v1"
VOCAB_SIZE = 128256
DEFAULT_CAPTURE_BATCH = 8
DEFAULT_MIN_FREE_GPU_BYTES = 8 * 1024**3
DEFAULT_MAX_RESERVED_GPU_BYTES = 8 * 1024**3
DEFAULT_MAX_HOST_RSS_BYTES = 16 * 1024**3
KNOWN_MODEL_WEIGHT_BYTES = 2_471_645_608
KNOWN_MODEL_WEIGHT_SHA256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"


class ActivationPreparationError(RuntimeError):
    """Raised when public activation preparation cannot proceed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, role: str, allow_symlink: bool = True) -> dict[str, Any]:
    original = path.expanduser()
    if original.is_symlink() and not allow_symlink:
        raise ActivationPreparationError(f"{role} cannot be a symbolic link: {original}")
    resolved = original.resolve()
    if not resolved.is_file():
        raise ActivationPreparationError(f"{role} must be an existing file: {original}")
    return {
        "role": role,
        "path": str(original.absolute()),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "symlink": original.is_symlink(),
    }


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ActivationPreparationError(f"{role} must be a regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivationPreparationError(f"{role} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ActivationPreparationError(f"{role} must contain an object")
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
    return result.stdout.strip() or None


def _device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise ActivationPreparationError("CUDA requested but unavailable")
    return torch.device(raw)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory(device: torch.device) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return result


@contextmanager
def _phase(phases: list[dict[str, Any]], name: str, device: torch.device) -> Iterator[None]:
    _sync(device)
    started = time.perf_counter()
    started_utc = _utc_now()
    status = 0
    try:
        yield
    except Exception:
        status = 1
        raise
    finally:
        _sync(device)
        phases.append(
            {
                "phase": name,
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "exit_status": status,
            }
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--dataset-arrow", type=Path, required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pile-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-records", type=int, default=DEFAULT_CAPTURE_BATCH)
    parser.add_argument("--cut-depth", type=int, default=CUT_DEPTH)
    parser.add_argument("--min-free-gpu-bytes", type=int, default=DEFAULT_MIN_FREE_GPU_BYTES)
    parser.add_argument("--max-reserved-gpu-bytes", type=int, default=DEFAULT_MAX_RESERVED_GPU_BYTES)
    parser.add_argument("--max-host-rss-bytes", type=int, default=DEFAULT_MAX_HOST_RSS_BYTES)
    return parser


def _check_plan_and_sources(
    plan_path: Path,
    arrow_path: Path,
    info_path: Path,
    tokenizer_path: Path,
    *,
    dataset: Dataset,
    tokenizer: Any,
) -> dict[str, Any]:
    plan = _load_json(plan_path, role="public Alpaca split plan")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("task_id") != TASK_ID:
        raise ActivationPreparationError("public split plan schema or task changed")
    if plan.get("status") != "REGISTERED_PUBLIC_FIT_SPLIT_NO_CONFIRMATION_GENERATED":
        raise ActivationPreparationError("public split plan is not a registered no-confirmation plan")
    if plan.get("execution", {}).get("truth_accessed") is not False:
        raise ActivationPreparationError("split plan truth-access marker changed")
    registration = plan.get("registration", {})
    if registration.get("contains_source_text") is not False or registration.get("contains_token_ids") is not False:
        raise ActivationPreparationError("split plan contains prohibited source material")
    dataset_declared = registration.get("dataset", {})
    if dataset_declared.get("id") != "tatsu-lab/alpaca" or dataset_declared.get("split") != "train":
        raise ActivationPreparationError("public Alpaca dataset identity changed")
    if dataset_declared.get("revision") != ALPACA_CACHE_REVISION:
        raise ActivationPreparationError("public Alpaca revision changed")
    if dataset_declared.get("row_count") != len(dataset):
        raise ActivationPreparationError("public Alpaca row count changed")
    source_dataset = plan.get("source", {}).get("dataset", {})
    expected_arrow = source_dataset.get("arrow", {})
    expected_info = source_dataset.get("dataset_info", {})
    actual_arrow = _file_record(arrow_path, role="public Alpaca Arrow cache")
    actual_info = _file_record(info_path, role="public Alpaca dataset metadata")
    if actual_arrow["sha256"] != expected_arrow.get("sha256"):
        raise ActivationPreparationError("public Arrow cache hash differs from registered plan")
    if actual_info["sha256"] != expected_info.get("sha256"):
        raise ActivationPreparationError("public dataset metadata hash differs from registered plan")
    if tokenizer_path.resolve() != Path(plan["source"]["tokenizer"]["path"]).resolve():
        raise ActivationPreparationError("tokenizer snapshot path differs from registered plan")
    if tokenizer.bos_token_id != DEFAULT_BOS_TOKEN_ID:
        raise ActivationPreparationError("tokenizer BOS ID changed")
    if tokenizer.pad_token_id not in (None, PAD_TOKEN_ID) and tokenizer.eos_token_id != PAD_TOKEN_ID:
        raise ActivationPreparationError("tokenizer padding/eos ID changed")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id != PAD_TOKEN_ID:
        raise ActivationPreparationError("tokenizer has no declared public padding fallback")
    if not getattr(tokenizer, "chat_template", None):
        raise ActivationPreparationError("tokenizer chat template is unavailable")
    split_info = _load_json(info_path, role="public Alpaca dataset metadata").get("splits", {}).get("train", {})
    if split_info.get("num_examples") != len(dataset):
        raise ActivationPreparationError("Arrow and dataset metadata row counts disagree")
    if tuple(dataset.column_names) != ("instruction", "input", "output", "text"):
        raise ActivationPreparationError("public Alpaca columns changed")
    return plan


def _load_public_prefix(
    model_path: Path,
    *,
    device: torch.device,
    cut_depth: int,
) -> tuple[ContiguousPublicPrefix, dict[str, Any], dict[str, Any]]:
    if not model_path.is_dir():
        raise ActivationPreparationError(f"public model snapshot directory is missing: {model_path}")
    config_path = model_path / "config.json"
    generation_path = model_path / "generation_config.json"
    weight_record = _file_record(model_path / "model.safetensors", role="public model weights")
    if weight_record["bytes"] != KNOWN_MODEL_WEIGHT_BYTES or weight_record["sha256"] != KNOWN_MODEL_WEIGHT_SHA256:
        raise ActivationPreparationError("public model weight hash or size differs from the pinned snapshot")
    snapshot = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_path": str(model_path.resolve()),
        "config": _file_record(config_path, role="public model config"),
        "generation_config": _file_record(generation_path, role="public generation config"),
        "weights": {
            **weight_record,
            "sha256_expected": KNOWN_MODEL_WEIGHT_SHA256,
            "verified_during_run": True,
            "hash_scope": "full model.safetensors read before model load",
        },
    }
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    if model.config.hidden_size != HIDDEN_SIZE or model.config.vocab_size != VOCAB_SIZE:
        raise ActivationPreparationError("public model geometry changed")
    if int(model.config.num_hidden_layers) <= cut_depth:
        raise ActivationPreparationError("cut depth leaves no downstream public layer")
    model.requires_grad_(False)
    prefix = ContiguousPublicPrefix(model, cut_depth=cut_depth).to(device).eval()
    model_config = {
        "hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "torch_dtype": str(next(model.parameters()).dtype),
        "cut_depth": cut_depth,
    }
    # The prefix retains only the embedding and the first cut layers.  Release
    # the full-model wrapper and its unused downstream references before capture.
    del model
    gc.collect()
    return prefix, snapshot, model_config


def _qualify_public_prefix_padding(
    prefix: ContiguousPublicPrefix,
    token_batch: PaddedTokenBatch,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Qualify causal padding and measure, without gating, batch-size drift.

    ``forward_full`` intentionally has no attention-mask argument.  The
    primary fixed batch-8 x 192 path is therefore checked by changing only
    token IDs in future padded positions, keeping the batch shape unchanged,
    and requiring bit-exact active outputs.  A separate batch-1 unpadded call
    is retained as a numerical diagnostic because kernel/batch changes can
    drift even when causal masking is correct.  That diagnostic never gates
    the primary path and never authorizes an alternate batching strategy.
    """

    validate_padded_token_batch(token_batch)
    if batch_size <= 0:
        raise ActivationPreparationError("qualification batch size must be positive")
    count = min(batch_size, int(token_batch.token_ids.shape[0]))
    inputs = token_batch.token_ids[:count].to(device=device, dtype=torch.long)
    active_mask = token_batch.attention_mask[:count].to(torch.bool)
    future_pad_inputs = inputs.clone()
    changed_future_pad_tokens = 0
    for row in range(count):
        active_count = int(token_batch.attention_mask[row].sum().item())
        if active_count < future_pad_inputs.shape[1]:
            # 0 is a valid public vocabulary ID and differs from the declared
            # pad ID.  Only future pad token IDs change; masks, positions, and
            # the [batch, time] geometry stay identical.
            future_pad_inputs[row, active_count:] = 0
            changed_future_pad_tokens += future_pad_inputs.shape[1] - active_count
    if changed_future_pad_tokens <= 0:
        raise ActivationPreparationError(
            "qualification fixture has no future padding token IDs to perturb"
        )

    with torch.inference_mode():
        padded_native = prefix.forward_full(inputs).detach().cpu()
        perturbed_native = prefix.forward_full(future_pad_inputs).detach().cpu()
        padded_active = padded_native[active_mask]
        perturbed_active = perturbed_native[active_mask]
        future_pad_bit_exact = torch.equal(padded_active, perturbed_active)
        future_pad_difference = (padded_active.float() - perturbed_active.float()).abs()
        if not future_pad_bit_exact:
            raise ActivationPreparationError(
                "future padding token IDs changed active outputs; causal padding qualification failed"
            )

        padded = padded_native.float()
        maximum_abs = 0.0
        relative_l2 = 0.0
        unpadded_bit_exact = True
        compared = 0
        for row in range(count):
            active_count = int(token_batch.attention_mask[row].sum().item())
            single_input = token_batch.token_ids[row : row + 1, :active_count].to(
                device=device, dtype=torch.long
            )
            reference_native = prefix.forward_full(single_input).detach().cpu()
            reference = reference_native.float()
            actual = padded[row : row + 1, :active_count]
            if tuple(actual.shape) != tuple(reference.shape):
                raise ActivationPreparationError("qualification reference geometry changed")
            difference = (actual - reference).abs()
            maximum_abs = max(maximum_abs, float(difference.max().item()))
            reference_l2 = float(torch.linalg.vector_norm(reference).item())
            difference_l2 = float(torch.linalg.vector_norm(difference).item())
            relative_l2 = max(relative_l2, difference_l2 / max(reference_l2, 1e-12))
            compared += int(reference.numel())
            unpadded_bit_exact = unpadded_bit_exact and torch.equal(
                padded_native[row : row + 1, :active_count], reference_native
            )
    return {
        "status": "passed",
        "primary_geometry": {
            "batch_records": count,
            "sequence_tokens": int(token_batch.token_ids.shape[1]),
            "path": "fixed batch-8 x 192 ContiguousPublicPrefix.forward_full",
            "future_pad_token_id": 0,
            "changed_future_pad_tokens": changed_future_pad_tokens,
            "same_batch_shape": True,
            "active_output_bit_exact": future_pad_bit_exact,
            "maximum_absolute_difference": float(future_pad_difference.max().item()),
            "decision_rule": "torch.equal on native active outputs; no tolerance",
        },
        "unpadded_batch1_diagnostic": {
            "reference_path": "ContiguousPublicPrefix.forward_full on each unpadded active record",
            "batch_records": count,
            "compared_active_values": compared,
            "maximum_absolute_difference": maximum_abs,
            "relative_l2": relative_l2,
            "active_output_bit_exact": bool(unpadded_bit_exact),
            "status": "equivalent_bit_exact" if unpadded_bit_exact else "non_equivalent_not_used",
            "used_for_capture": False,
            "decision_rule": "diagnostic only; no tolerance gate and no batching substitution",
        },
        "batching_substitution_allowed": False,
    }

def _source_code_records(root: Path, script_path: Path) -> dict[str, Any]:
    paths = {
        "runner": script_path,
        "activation_module": root / "src/token_reconstruction/public_activation.py",
        "public_prefix_module": root / "src/token_reconstruction/public_prefix.py",
        "split_module": root / "src/token_reconstruction/alpaca_split.py",
        "runtime_module": root / "src/token_reconstruction/experiment_runtime.py",
    }
    return {
        name: _file_record(path, role=f"executed {name} source", allow_symlink=False)
        for name, path in paths.items()
    }


def _record_manifest(
    rows: list[Mapping[str, Any]],
    token_batch: PaddedTokenBatch,
    *,
    split: str,
) -> dict[str, Any]:
    if len(rows) != token_batch.token_ids.shape[0]:
        raise ActivationPreparationError(f"{split} record count does not match tensors")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        active = int(token_batch.attention_mask[index].sum().item())
        start, end = token_batch.post_bos_ranges[index]
        item = dict(row)
        item.update(
            {
                "padded_length": int(token_batch.token_ids.shape[1]),
                "active_token_count": active,
                "post_bos_start": start,
                "post_bos_end_exclusive": end,
                "small_post_bos_count": int(token_batch.post_bos_selector_small[index].sum().item()),
                "large_post_bos_count": int(token_batch.post_bos_selector_large[index].sum().item()),
            }
        )
        records.append(item)
    ids = [str(item["record_id"]) for item in records]
    return {
        "schema": "token-reconstruction.trr0004-public-activation-records.v1",
        "task_id": TASK_ID,
        "split": split,
        "current_token_alignment": "activations[record,position] -> token_ids[record,position]",
        "source_text_included": False,
        "token_ids_included": False,
        "record_count": len(records),
        "record_ids_sha256": record_ids_sha256(ids),
        "records": records,
    }


def _save_record_manifest(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise ActivationPreparationError(f"record manifest is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _resource_preflight(
    device: torch.device,
    *,
    min_free_gpu_bytes: int,
    max_reserved_gpu_bytes: int,
    max_host_rss_bytes: int,
) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "status": "host_only",
            "guard": "CUDA guard not applicable; model capture is still geometry-checked",
        }
    free, total = torch.cuda.mem_get_info(device)
    if free < min_free_gpu_bytes:
        raise ActivationPreparationError(
            f"fail-closed GPU guard: only {free} bytes free; need {min_free_gpu_bytes}"
        )
    return {
        "device": str(device),
        "status": "passed",
        "free_bytes": int(free),
        "total_bytes": int(total),
        "minimum_free_bytes": min_free_gpu_bytes,
        "maximum_reserved_bytes": max_reserved_gpu_bytes,
        "maximum_host_rss_bytes": max_host_rss_bytes,
        "failure_policy": "stop and preserve failure on OOM, non-finite activation, allocator/driver anomaly, or source mismatch",
    }


def _enforce_resource_ceiling(
    device: torch.device,
    *,
    max_reserved_gpu_bytes: int,
    max_host_rss_bytes: int,
) -> dict[str, int | None]:
    """Fail closed at every major phase and after every capture batch."""

    peak = _peak_memory(device)
    reserved = peak.get("cuda_peak_reserved_bytes")
    if reserved is not None and reserved > max_reserved_gpu_bytes:
        raise ActivationPreparationError("measured GPU reservation exceeded the fail-closed ceiling")
    host = peak.get("host_max_rss_bytes")
    if host is not None and host > max_host_rss_bytes:
        raise ActivationPreparationError("measured host RSS exceeded the fail-closed ceiling")
    return peak


def _run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ActivationPreparationError(f"output root is create-only and already exists: {output_root}")
    if args.batch_records <= 0 or args.cut_depth != CUT_DEPTH:
        raise ActivationPreparationError("declared batch size or cut depth changed")
    device = _device(args.device)
    plan_path = args.split_plan.expanduser().resolve()
    arrow_path = args.dataset_arrow.expanduser().resolve()
    info_path = args.dataset_info.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    pile_receipt_path = args.pile_receipt.expanduser().resolve()
    if not pile_receipt_path.is_file():
        raise ActivationPreparationError(f"public Pile validation receipt is missing: {pile_receipt_path}")
    pile_receipt_sha256 = _sha256_file(pile_receipt_path)
    if pile_receipt_sha256 != "d7dbfccc70f6a60b92ca4849870fea077c068584fd1eeca119e560c69600d22c":
        raise ActivationPreparationError("public Pile validation receipt hash changed")
    output_root.mkdir(parents=True)
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    phases: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with _phase(phases, "public_resource_guard", device):
            guard = _resource_preflight(
                device,
                min_free_gpu_bytes=args.min_free_gpu_bytes,
                max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
                max_host_rss_bytes=args.max_host_rss_bytes,
            )
        with _phase(phases, "load_public_alpaca_cache_and_tokenizer", device):
            dataset = Dataset.from_file(str(arrow_path))
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), local_files_only=True, use_fast=True
            )
            plan = _check_plan_and_sources(
                plan_path,
                arrow_path,
                info_path,
                tokenizer_path,
                dataset=dataset,
                tokenizer=tokenizer,
            )
        dataset_revision = str(plan["registration"]["dataset"]["revision"])
        with _phase(phases, "materialize_registered_public_token_sequences", device):
            fit_sequences, fit_rows = materialize_plan_split(
                plan,
                dataset,
                tokenizer,
                "fit",
                dataset_revision=dataset_revision,
                maximum_tokens=HISTORICAL_MAX_TOKENS,
                minimum_full_tokens=HISTORICAL_MIN_FULL_TOKENS,
                expected_bos_token_id=DEFAULT_BOS_TOKEN_ID,
            )
            validation_sequences, validation_rows = materialize_plan_split(
                plan,
                dataset,
                tokenizer,
                "validation",
                dataset_revision=dataset_revision,
                maximum_tokens=HISTORICAL_MAX_TOKENS,
                minimum_full_tokens=HISTORICAL_MIN_FULL_TOKENS,
                expected_bos_token_id=DEFAULT_BOS_TOKEN_ID,
            )
            fit_batch = pad_public_token_sequences(fit_sequences)
            validation_batch = pad_public_token_sequences(validation_sequences)
            validate_padded_token_batch(fit_batch)
            validate_padded_token_batch(validation_batch)
            if fit_batch.post_bos_positions != int(plan["registration"]["fit"]["large_nested"]["post_bos_positions"]):
                raise ActivationPreparationError("registered fit post-BOS position count changed")
            if fit_batch.small_positions != int(plan["registration"]["fit"]["small_nested"]["post_bos_positions"]):
                raise ActivationPreparationError("registered nested fit position count changed")
            if set(row["record_id"] for row in fit_rows) & set(row["record_id"] for row in validation_rows):
                raise ActivationPreparationError("public fit and validation records overlap")
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            if pad_token_id != PAD_TOKEN_ID:
                raise ActivationPreparationError("public tokenizer padding ID changed")
            fit_manifest = _record_manifest(fit_rows, fit_batch, split="train_large")
            validation_manifest = _record_manifest(validation_rows, validation_batch, split="validation_alpaca")
        with _phase(phases, "load_public_model_and_prefix", device):
            prefix, model_snapshot, model_config = _load_public_prefix(
                model_path,
                device=device,
                cut_depth=args.cut_depth,
            )
            _enforce_resource_ceiling(
                device,
                max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
                max_host_rss_bytes=args.max_host_rss_bytes,
            )
        with _phase(phases, "qualify_public_prefix_batch8_x_192_padding", device):
            qualification = _qualify_public_prefix_padding(
                prefix,
                fit_batch,
                device=device,
                batch_size=args.batch_records,
            )
            qualification["measured_peak_after_phase"] = _enforce_resource_ceiling(
                device,
                max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
                max_host_rss_bytes=args.max_host_rss_bytes,
            )
        with _phase(phases, "capture_public_train_large_cut4_activations", device):
            train_activations = capture_public_prefix(
                prefix,
                fit_batch,
                device=device,
                batch_size=args.batch_records,
                resource_check=lambda: _enforce_resource_ceiling(
                    device,
                    max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
                    max_host_rss_bytes=args.max_host_rss_bytes,
                ),
            )
        with _phase(phases, "capture_public_alpaca_validation_cut4_activations", device):
            validation_activations = capture_public_prefix(
                prefix,
                validation_batch,
                device=device,
                batch_size=args.batch_records,
                resource_check=lambda: _enforce_resource_ceiling(
                    device,
                    max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
                    max_host_rss_bytes=args.max_host_rss_bytes,
                ),
            )
        _enforce_resource_ceiling(
            device,
            max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
            max_host_rss_bytes=args.max_host_rss_bytes,
        )
        source_plan = _file_record(plan_path, role="registered public Alpaca split plan", allow_symlink=False)
        source_arrow = _file_record(arrow_path, role="public Alpaca Arrow cache")
        source_info = _file_record(info_path, role="public Alpaca dataset metadata")
        common_metadata = {
            "source_plan_sha256": source_plan["sha256"],
            "source_dataset_arrow_sha256": source_arrow["sha256"],
            "source_dataset_info_sha256": source_info["sha256"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
        train_metadata = make_artifact_metadata(
            split="train_large",
            source_plan_sha256=source_plan["sha256"],
            source_arrow_sha256=source_arrow["sha256"],
            source_info_sha256=source_info["sha256"],
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            cut_depth=args.cut_depth,
            token_batch=fit_batch,
            activations=train_activations,
            records=fit_rows,
        )
        validation_metadata = make_artifact_metadata(
            split="validation_alpaca",
            source_plan_sha256=source_plan["sha256"],
            source_arrow_sha256=source_arrow["sha256"],
            source_info_sha256=source_info["sha256"],
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            cut_depth=args.cut_depth,
            token_batch=validation_batch,
            activations=validation_activations,
            records=validation_rows,
        )
        with _phase(phases, "serialize_public_activation_artifacts", device):
            train_path = output_root / "train_large_cut4.safetensors"
            validation_path = output_root / "validation_alpaca_cut4.safetensors"
            save_public_artifact(
                train_path,
                activations=train_activations,
                token_batch=fit_batch,
                metadata=train_metadata,
            )
            save_public_artifact(
                validation_path,
                activations=validation_activations,
                token_batch=validation_batch,
                metadata=validation_metadata,
            )
            train_records_path = output_root / "train_large_records.json"
            validation_records_path = output_root / "validation_alpaca_records.json"
            train_records_file = _save_record_manifest(train_records_path, fit_manifest)
            validation_records_file = _save_record_manifest(validation_records_path, validation_manifest)

        source_codes = _source_code_records(root, Path(__file__).resolve())
        public_pile_reference = {
            "role": "existing public Pile24 validation reference; not regenerated or read for this preparation",
            "path": str(pile_receipt_path),
            "expected_records": 24,
            "sha256": pile_receipt_sha256,
        }
        evidence = {
            "schema": "token-reconstruction.trr0004-public-activation-preparation.v1",
            "task_id": TASK_ID,
            "status": "PUBLIC_ACTIVATION_PREPARATION_COMPLETE_NO_CONFIRMATION",
            "access_contract": {
                "public_alpaca_labels_read": True,
                "public_prefix_only": True,
                "target_weights_accessed": False,
                "evaluator_private_truth_accessed": False,
                "confirmation_records_generated": False,
                "current_token_alignment": "activations[record,position] predicts token_ids[record,position]",
            },
            "execution": {
                "argv": [str(value) for value in sys.argv],
                "working_directory": str(Path.cwd().resolve()),
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started_clock,
                "git_commit_start_and_end": _git_commit(root),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "batch_records": args.batch_records,
                "cut_depth": args.cut_depth,
                "network_used": False,
            },
            "source": {
                "split_plan": source_plan,
                "dataset_arrow": source_arrow,
                "dataset_info": source_info,
                "model_snapshot": model_snapshot,
                "model_config": model_config,
                "public_pile_validation_reference": public_pile_reference,
            },
            "geometry": {
                "hidden_size": HIDDEN_SIZE,
                "vocab_size": VOCAB_SIZE,
                "padded_tokens": HISTORICAL_MAX_TOKENS,
                "train_large": {
                    "records": len(fit_rows),
                    "activations": list(train_activations.shape),
                    "token_ids": list(fit_batch.token_ids.shape),
                    "attention_mask": list(fit_batch.attention_mask.shape),
                    "position_ids": list(fit_batch.position_ids.shape),
                    "post_bos_positions": fit_batch.post_bos_positions,
                    "nested_small_positions": fit_batch.small_positions,
                },
                "validation_alpaca": {
                    "records": len(validation_rows),
                    "activations": list(validation_activations.shape),
                    "token_ids": list(validation_batch.token_ids.shape),
                    "attention_mask": list(validation_batch.attention_mask.shape),
                    "position_ids": list(validation_batch.position_ids.shape),
                    "post_bos_positions": validation_batch.post_bos_positions,
                },
            },
            "tensor_hashes": {
                "train_activations": tensor_sha256(train_activations),
                "train_token_ids": tensor_sha256(fit_batch.token_ids),
                "train_attention_mask": tensor_sha256(fit_batch.attention_mask),
                "train_position_ids": tensor_sha256(fit_batch.position_ids),
                "train_selector_small": tensor_sha256(fit_batch.post_bos_selector_small),
                "train_selector_large": tensor_sha256(fit_batch.post_bos_selector_large),
                "validation_activations": tensor_sha256(validation_activations),
                "validation_token_ids": tensor_sha256(validation_batch.token_ids),
                "validation_attention_mask": tensor_sha256(validation_batch.attention_mask),
                "validation_position_ids": tensor_sha256(validation_batch.position_ids),
            },
            "outputs": {
                "train_large": {
                    "path": str(train_path.resolve()),
                    "bytes": train_path.stat().st_size,
                    "sha256": _sha256_file(train_path),
                    "role": "public fit activations and labels; nested selectors included",
                },
                "validation_alpaca": {
                    "path": str(validation_path.resolve()),
                    "bytes": validation_path.stat().st_size,
                    "sha256": _sha256_file(validation_path),
                    "role": "public continuation validation activations and labels",
                },
                "train_records": train_records_file,
                "validation_records": validation_records_file,
            },
            "phases": phases,
            "qualification": qualification,
            "resource_guard": guard,
            "peak_memory": _peak_memory(device),
            "runtime_components": {
                "public_prefix": "ContiguousPublicPrefix.forward_full; public embedding + layers[0:4]",
                "candidate_simulations": 0,
                "private_target_prefix_calls": 0,
                "padding": "right padded to 192; padded activation rows zeroed; active rows retain causal current-token alignment",
            },
            "reproducibility": {
                "source_code": source_codes,
                "common_metadata": common_metadata,
                "fit_record_ids_sha256": record_ids_sha256([str(row["record_id"]) for row in fit_rows]),
                "validation_record_ids_sha256": record_ids_sha256([str(row["record_id"]) for row in validation_rows]),
                "fit_validation_disjoint": True,
                "nested_selector": "first 5,000 post-BOS positions in the registered ordered fit stream",
            },
        }
        final_peak = _peak_memory(device)
        if device.type == "cuda" and final_peak["cuda_peak_reserved_bytes"] is not None and final_peak["cuda_peak_reserved_bytes"] > args.max_reserved_gpu_bytes:
            raise ActivationPreparationError("measured GPU reservation exceeded the fail-closed ceiling")
        if final_peak["host_max_rss_bytes"] is not None and final_peak["host_max_rss_bytes"] > args.max_host_rss_bytes:
            raise ActivationPreparationError("measured host RSS exceeded the fail-closed ceiling")
        evidence["peak_memory"] = final_peak
        evidence_path = output_root / "preparation_evidence.json"
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": evidence["status"],
            "output_root": str(output_root),
            "train_shape": list(train_activations.shape),
            "validation_shape": list(validation_activations.shape),
            "train_post_bos_positions": fit_batch.post_bos_positions,
            "train_nested_small_positions": fit_batch.small_positions,
            "validation_post_bos_positions": validation_batch.post_bos_positions,
            "peak_memory": evidence["peak_memory"],
        }, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "token-reconstruction.trr0004-public-activation-preparation-failure.v1",
            "task_id": TASK_ID,
            "status": "FAILED_PUBLIC_ACTIVATION_PREPARATION",
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "command": [str(value) for value in sys.argv],
            "git_commit": _git_commit(root),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "phases": phases,
            "peak_memory": _peak_memory(device),
            "target_weights_accessed": False,
            "evaluator_private_truth_accessed": False,
            "output_root": str(output_root),
        }
        try:
            (output_root / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (ActivationPreparationError, PublicActivationError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
