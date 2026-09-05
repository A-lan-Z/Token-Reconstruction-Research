#!/usr/bin/env python3
"""Fit the controlled historical-style affine CE decoders for TRR-0004.

The runner consumes a pinned public fit-data manifest prepared by the footing
workstream.  It fits the same full hidden affine map and learned global scale
with either no vocabulary bias (the historical-style arm) or a zero-initialized
trainable vocabulary bias (the ablation).  Both arms emit direct vocabulary
predictions and never call a public prefix or an A2 candidate search.

This command is source and configuration infrastructure.  A model run must be
started only from a frozen task commit after the public manifest and resource
preflight have been reviewed.  The output directory is create-only so an
interrupted fit cannot be mistaken for a complete state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource as sys_resource
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from token_reconstruction.historical_affine_ce import (
    BOS_TOKEN_ID,
    HistoricalAffineCEConfig,
    HistoricalAffineCEDecoder,
    HistoricalAffineCEError,
    direct_prediction_tensor,
    evaluation_schedule,
    file_sha256,
    fixed_training_probe,
    load_public_fit_bundle,
    save_historical_affine_ce,
    tensor_sha256,
    train_historical_affine_ce,
)


TASK_ID = "TRR-0004"
SCRIPT_SCHEMA = "token-reconstruction.trr0004-controlled-affine-ce-run.v1"
DEFAULT_BIAS_MODES = ("none", "vocab")


class ControlledFitRunnerError(RuntimeError):
    """Raised when a controlled fit cannot be run safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ControlledFitRunnerError(f"{label} must be a regular file: {path}")
    return path


def _file_record(path: Path, *, label: str = "resource") -> dict[str, Any]:
    path = _regular_file(path.expanduser().resolve(), label=label)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


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


def _source_records(root: Path) -> dict[str, Any]:
    script = Path(__file__).resolve()
    module = root / "src/token_reconstruction/historical_affine_ce.py"
    return {
        "runner": _file_record(script, label="fit runner"),
        "module": _file_record(module, label="fit module"),
    }


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
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def _json_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def _device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise ControlledFitRunnerError(f"invalid device: {value}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ControlledFitRunnerError("CUDA was requested but is unavailable")
    if device.type not in ("cpu", "cuda"):
        raise ControlledFitRunnerError("controlled fit supports only CPU or CUDA")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory(device: torch.device) -> dict[str, int | None]:
    rss_kib = int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss)
    result: dict[str, int | None] = {
        "process_max_rss_kib": rss_kib,
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


def _resource_limits(args: argparse.Namespace) -> dict[str, int]:
    gib = 1024**3
    return {
        "minimum_free_gpu_bytes": int(args.minimum_free_gib * gib),
        "maximum_gpu_reserved_bytes": int(args.maximum_gpu_reserved_gib * gib),
        "maximum_host_rss_bytes": int(args.maximum_host_rss_gib * gib),
    }


def _guard(
    args: argparse.Namespace,
    device: torch.device,
    *,
    deadline: float | None = None,
) -> dict[str, int | None]:
    if deadline is not None and time.perf_counter() >= deadline:
        raise ControlledFitRunnerError("fit exceeded its wall-time guard")
    limits = _resource_limits(args)
    rss_bytes = int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss) * 1024
    if rss_bytes > limits["maximum_host_rss_bytes"]:
        raise ControlledFitRunnerError(
            f"host RSS guard exceeded: {rss_bytes} > {limits['maximum_host_rss_bytes']}"
        )
    result: dict[str, int | None] = {
        "process_max_rss_bytes": rss_bytes,
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
        "cuda_reserved_bytes": None,
    }
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        reserved_bytes = int(torch.cuda.memory_reserved(device))
        result.update(
            {
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                "cuda_reserved_bytes": reserved_bytes,
            }
        )
        if free_bytes < limits["minimum_free_gpu_bytes"]:
            raise ControlledFitRunnerError(
                f"GPU free-memory guard exceeded: {free_bytes} < {limits['minimum_free_gpu_bytes']}"
            )
        if reserved_bytes > limits["maximum_gpu_reserved_bytes"]:
            raise ControlledFitRunnerError(
                f"GPU reserved-memory guard exceeded: {reserved_bytes} > {limits['maximum_gpu_reserved_bytes']}"
            )
    return result


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"trr0004-historical-affine-ce-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parse_bias_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes or any(mode not in DEFAULT_BIAS_MODES for mode in modes):
        raise ControlledFitRunnerError("bias modes must be a comma-separated subset of none,vocab")
    if len(set(modes)) != len(modes):
        raise ControlledFitRunnerError("bias modes must not repeat an arm")
    return modes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bias-modes", default=",".join(DEFAULT_BIAS_MODES))
    parser.add_argument("--fit-record-limit", type=int)
    parser.add_argument("--fit-position-limit", type=int)
    parser.add_argument("--exclude-bos-from-fit", action="store_true")
    parser.add_argument("--include-bos-in-validation", action="store_true")
    parser.add_argument("--retained-a1-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=0.0,
        help="gradient clipping max norm; 0 disables clipping (historical default)",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--init-logit-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=6.0)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=16.0)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    return parser


def _group_prediction_metrics(
    predictions: torch.Tensor, labels: torch.Tensor, groups: tuple[str, ...]
) -> dict[str, Any]:
    """Return aggregate and equal-group validation metrics for one prediction vector."""

    predictions = predictions.reshape(-1).to(device="cpu", dtype=torch.long)
    labels = labels.reshape(-1).to(device="cpu", dtype=torch.long)
    if predictions.shape != labels.shape or len(groups) != int(labels.numel()):
        raise ControlledFitRunnerError("validation group metrics have incompatible geometry")
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for prediction, label, group in zip(predictions.tolist(), labels.tolist(), groups):
        if not group:
            raise ControlledFitRunnerError("validation group is empty")
        totals[group] = totals.get(group, 0) + 1
        correct[group] = correct.get(group, 0) + int(prediction == label)
    group_accuracy = {
        group: correct[group] / totals[group] for group in sorted(totals)
    }
    aggregate_correct = sum(correct.values())
    aggregate_rows = sum(totals.values())
    return {
        "token_accuracy": aggregate_correct / aggregate_rows,
        "correct_tokens": aggregate_correct,
        "examples": aggregate_rows,
        "group_token_accuracy": group_accuracy,
        "group_token_rows": {group: totals[group] for group in sorted(totals)},
        "style_balanced_token_accuracy": sum(group_accuracy.values()) / len(group_accuracy),
        "style_balanced_groups": sorted(group_accuracy),
    }


def _evaluate_retained_a1(
    checkpoint: Path,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    device: torch.device,
    validation_groups: tuple[str, ...],
) -> dict[str, Any]:
    # This optional public development comparator is loaded only after the
    # fit/validation split has been checked by load_public_fit_bundle.
    from token_reconstruction.historical_inputlens_bridge import load_historical_lens_checkpoint

    bridge = load_historical_lens_checkpoint(checkpoint, device=device)
    embeddings = embedding_table.to(device=device)
    predictions = bridge.predict(validation_x.to(device=device), embeddings, batch_size=512)
    labels = validation_y.to(device="cpu", dtype=torch.long)
    metrics = _group_prediction_metrics(predictions, labels, validation_groups)
    result = {
        "method_id": bridge.method_id,
        "checkpoint": _file_record(checkpoint, label="retained A1 checkpoint"),
        **metrics,
        "uses_a2": False,
        "public_prefix_calls": 0,
    }
    del bridge, predictions
    return result


def run(args: argparse.Namespace) -> int:
    root = args.repository_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ControlledFitRunnerError(f"output root is create-only and already exists: {output}")
    if args.fit_record_limit is not None and args.fit_record_limit <= 0:
        raise ControlledFitRunnerError("fit record limit must be positive")
    if args.fit_position_limit is not None and args.fit_position_limit <= 0:
        raise ControlledFitRunnerError("fit position limit must be positive")
    if args.fit_record_limit is not None and args.fit_position_limit is not None:
        raise ControlledFitRunnerError("choose a record limit or a position limit, not both")
    if args.max_seconds <= 0:
        raise ControlledFitRunnerError("wall-time guard must be positive")
    modes = _parse_bias_modes(args.bias_modes)
    config = HistoricalAffineCEConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        log_every=args.log_every,
        init_logit_scale=args.init_logit_scale,
        seed=args.seed,
    )
    config.validate()
    device = _device(args.device)
    output.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    started_clock = time.perf_counter()
    commit_start = _git_commit(root)
    status_start = _git_status(root)
    source_start = _source_records(root)
    preflight_before = _guard(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    bundle = load_public_fit_bundle(args.data_manifest.expanduser().resolve())
    bos_token_id = int(bundle.metadata.get("bos_token_id", BOS_TOKEN_ID))
    fit_x, fit_y = bundle.fit_tensors(
        record_limit=args.fit_record_limit,
        position_limit=args.fit_position_limit,
        bos_token_id=bos_token_id,
        include_bos=not args.exclude_bos_from_fit,
    )
    validation_x, validation_y = bundle.validation_tensors(
        bos_token_id=bos_token_id,
        include_bos=args.include_bos_in_validation,
    )
    validation_groups = bundle.validation_flat_groups(
        include_bos=args.include_bos_in_validation
    )
    if fit_x.shape[0] <= 0 or validation_x.shape[0] <= 0:
        raise ControlledFitRunnerError("public fit or validation flattening produced no rows")
    if len(validation_groups) != int(validation_x.shape[0]):
        raise ControlledFitRunnerError("public validation groups do not cover all rows")
    probe_x, probe_y, probe_indices = fixed_training_probe(
        fit_x, fit_y, size=2048, seed=17
    )
    load_seconds = time.perf_counter() - load_started
    embedding_table = bundle.embedding_table.contiguous()
    deadline = time.perf_counter() + args.max_seconds
    validation_baseline: dict[str, Any] | None = None
    if args.retained_a1_checkpoint is not None:
        baseline_started = time.perf_counter()
        validation_baseline = _evaluate_retained_a1(
            args.retained_a1_checkpoint.expanduser().resolve(),
            validation_x,
            validation_y,
            embedding_table,
            device=device,
            validation_groups=validation_groups,
        )
        validation_baseline["elapsed_seconds"] = time.perf_counter() - baseline_started
    method_results: dict[str, Any] = {}
    fit_phase_records: list[dict[str, Any]] = []
    for mode in modes:
        _guard(args, device, deadline=deadline)
        method = HistoricalAffineCEDecoder(
            bundle.hidden_size,
            bundle.vocabulary_size,
            bias_mode=mode,
            init_logit_scale=config.init_logit_scale,
        )
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.cuda.reset_peak_memory_stats(device)
        fit_started_at = _utc_now()
        fit_started = time.perf_counter()
        model, evidence = train_historical_affine_ce(
            method,
            fit_x,
            fit_y,
            embedding_table,
            config=config,
            device=device,
            validation=(validation_x, validation_y),
            validation_groups=validation_groups,
            training_probe=(probe_x, probe_y),
            deadline=deadline,
            resource_guard=lambda: _guard(args, device, deadline=deadline),
        )
        _synchronize(device)
        fit_elapsed = time.perf_counter() - fit_started
        selected_state = evidence.pop("selected_state_dict")
        final_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        selected_path = output / f"{model.resolved_method_id}.safetensors"
        final_path = output / f"{model.resolved_method_id}_final.safetensors"
        common_metadata = {
            "task_id": TASK_ID,
            "fit_data_manifest_sha256": bundle.metadata["manifest_sha256"],
            "embedding_table_sha256": tensor_sha256(embedding_table),
            "alignment": bundle.metadata["alignment"]["mode"],
            "fit_includes_bos": str(not args.exclude_bos_from_fit),
            "selected_step": str(evidence["selected_step"]),
        }
        save_historical_affine_ce(model, selected_path, metadata=common_metadata, state=selected_state)
        save_historical_affine_ce(model, final_path, metadata={**common_metadata, "state_role": "final"}, state=final_state)
        selected_evidence = dict(evidence)
        selected_evidence["selected_state_sha256"] = _state_sha256(selected_state)
        selected_evidence["final_state_sha256"] = _state_sha256(final_state)
        selected_evidence["selected_artifact"] = _file_record(selected_path, label="selected decoder state")
        selected_evidence["final_artifact"] = _file_record(final_path, label="final decoder state")
        selected_evidence["fit_elapsed_seconds"] = fit_elapsed
        selected_evidence["peak_memory"] = _peak_memory(device)
        selected_evidence["runtime_components"] = {
            "public_embedding_table_required": True,
            "public_prefix_calls": 0,
            "candidate_simulations": 0,
            "a2_fallback": False,
        }
        method_results[model.resolved_method_id] = selected_evidence
        fit_phase_records.append(
            {
                "phase": f"fit_{model.resolved_method_id}",
                "started_utc": fit_started_at,
                "elapsed_seconds": fit_elapsed,
                "peak_memory": _peak_memory(device),
            }
        )
        del model, method, selected_state, final_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import gc

        gc.collect()
    _guard(args, device, deadline=deadline)
    _synchronize(device)
    source_end = _source_records(root)
    commit_end = _git_commit(root)
    if source_start != source_end:
        raise ControlledFitRunnerError("source files changed during controlled fit")
    if commit_start != commit_end:
        raise ControlledFitRunnerError("git commit changed during controlled fit")
    result = {
        "schema": SCRIPT_SCHEMA,
        "task_id": TASK_ID,
        "status": "fit_complete",
        "claim_scope": "controlled public recipe recreation; no replacement claim",
        "started_utc": started_at,
        "ended_utc": _utc_now(),
        "command": {
            "argv": [str(value) for value in sys.argv],
            "cwd": str(Path.cwd()),
            "python": sys.executable,
            "args": _json_args(args),
            "environment": _safe_environment(),
        },
        "provenance": {
            "git_commit_start": commit_start,
            "git_commit_end": commit_end,
            "source_files_start": source_start,
            "source_files_end": source_end,
            "git_status_start": status_start,
            "git_status_end": _git_status(root),
        },
        "configuration": {
            **asdict(config),
            "bias_modes": list(modes),
            "fit_includes_bos": not args.exclude_bos_from_fit,
            "validation_includes_bos": args.include_bos_in_validation,
            "fit_record_limit": args.fit_record_limit,
            "fit_position_limit": args.fit_position_limit,
            "evaluation_schedule": list(evaluation_schedule(config.steps)),
            "training_probe": {
                "source": "public fit rows only",
                "requested_rows": 2048,
                "seed": 17,
                "rows_used": int(probe_x.shape[0]),
                "index_sha256": tensor_sha256(probe_indices),
                "activation_sha256": tensor_sha256(probe_x),
                "label_sha256": tensor_sha256(probe_y),
                "curve_evaluation": "fixed probe at every registered checkpoint",
            },
            "validation_selection": {
                "metric": "style_balanced_token_accuracy",
                "groups": sorted(set(validation_groups)),
                "all_rows_used": int(validation_x.shape[0]),
                "definition": "unweighted mean of per-group token accuracies; one public group equals aggregate accuracy",
            },
        },
        "resource_preflight": {
            "limits": _resource_limits(args),
            "before_loading": preflight_before,
            "max_seconds": args.max_seconds,
        },
        "fit_data": {
            "manifest": bundle.metadata,
            "layout": bundle.layout,
            "fit_records": bundle.fit_record_count,
            "validation_records": bundle.validation_record_count,
            "fit_rows_used": int(fit_x.shape[0]),
            "validation_rows_used": int(validation_x.shape[0]),
            "hidden_size": bundle.hidden_size,
            "vocabulary_size": bundle.vocabulary_size,
            "alignment_mode": bundle.metadata["alignment"]["mode"],
            "fit_supervision": "current-token, with optional BOS row controlled by fit_includes_bos",
            "validation_groups": sorted(set(validation_groups)),
            "validation_group_rows": {
                group: int(validation_groups.count(group)) for group in sorted(set(validation_groups))
            },
        },
        "public_validation_baseline": validation_baseline,
        "methods": method_results,
        "phases": [
            {"phase": "load_public_fit_bundle", "elapsed_seconds": load_seconds},
            *fit_phase_records,
        ],
        "peak_memory": _peak_memory(device),
        "notes": [
            "The no-vocabulary-bias arm retains hidden-space b and learned s and is the historical-style control.",
            "The vocabulary-bias arm differs only by a zero-initialized trainable vocabulary bias.",
            "Validation curves use all public validation rows; checkpoint selection is the unweighted mean of per-group token accuracies.",
            "Train curves use one fixed 2048-row public fit probe; one final full-fit pass is retained as a capacity diagnostic and is not used for selection.",
            "Selected artifacts use the earliest step attaining maximum style-balanced public validation token accuracy.",
            "Gradient clipping is disabled (norm 0) to match the historical source; total gradient norms are still checked for finiteness.",
            "The retained TRR-0002 A1 asset, when supplied, is a frozen public validation comparator and not a newly fitted state.",
        ],
    }
    output_file = output / "fit_evidence.json"
    output_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output_root": str(output), "methods": list(method_results)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (ControlledFitRunnerError, HistoricalAffineCEError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
