#!/usr/bin/env python3
"""Measure selected affine-decoder errors by exact fit-label coverage.

This is a post-fit public-development diagnostic.  It binds one selected
state to its own fit evidence and manifest, counts labels in the exact fit
stream used by that run, and predicts the public Alpaca/Pile validation rows
without A2 or a public-prefix call.  The output stores prediction and source
hashes plus aggregate error buckets; it never copies validation token IDs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource as sys_resource
import subprocess
import time
from typing import Any, Mapping

import torch

from token_reconstruction.historical_affine_ce import (
    HistoricalAffineCEError,
    direct_prediction_tensor,
    file_sha256,
    load_historical_affine_ce,
    load_public_fit_bundle,
    tensor_sha256,
)


TASK_ID = "TRR-0004"
SCRIPT_SCHEMA = "token-reconstruction.trr0004-affine-coverage-diagnostic.v1"
METHOD_BIAS = {
    "historical_affine_ce_no_vocab_bias": "none",
    "historical_affine_ce_vocab_bias": "vocab",
}


class CoverageDiagnosticError(RuntimeError):
    """Raised when the public coverage diagnostic cannot bind safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regular_file(path: Path, *, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise CoverageDiagnosticError(f"{label} must be a regular file: {path}")
    return path.resolve()


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    path = _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageDiagnosticError(f"cannot parse {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise CoverageDiagnosticError(f"{label} must contain an object")
    return value


def _recorded_artifact(path: Path, expected: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise CoverageDiagnosticError(f"{label} evidence entry is missing")
    actual = _file_record(path, label=label)
    for key in ("bytes", "sha256"):
        if expected.get(key) != actual[key]:
            raise CoverageDiagnosticError(f"{label} binding changed")
    return actual


def _coverage_json_path(path: Path) -> str:
    """Encode a filesystem path as a JSON scalar in coverage evidence."""

    return str(path)


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


def _device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise CoverageDiagnosticError(f"invalid device: {value}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CoverageDiagnosticError("CUDA was requested but is unavailable")
    if device.type not in ("cpu", "cuda"):
        raise CoverageDiagnosticError("coverage diagnostic supports only CPU or CUDA")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resource_limits(args: argparse.Namespace) -> dict[str, int]:
    gib = 1024**3
    return {
        "minimum_free_gpu_bytes": int(args.minimum_free_gib * gib),
        "maximum_gpu_reserved_bytes": int(args.maximum_gpu_reserved_gib * gib),
        "maximum_host_rss_bytes": int(args.maximum_host_rss_gib * gib),
    }


def _query_compute_apps() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CoverageDiagnosticError("cannot query GPU compute applications") from exc
    return result.stdout.strip()


def _guard(
    args: argparse.Namespace, device: torch.device, *, deadline: float | None = None
) -> dict[str, int | None]:
    if deadline is not None and time.perf_counter() >= deadline:
        raise CoverageDiagnosticError("coverage diagnostic exceeded its wall-time guard")
    limits = _resource_limits(args)
    rss_bytes = int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss) * 1024
    if rss_bytes > limits["maximum_host_rss_bytes"]:
        raise CoverageDiagnosticError(
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
            raise CoverageDiagnosticError(
                f"GPU free-memory guard exceeded: {free_bytes} < {limits['minimum_free_gpu_bytes']}"
            )
        if reserved_bytes > limits["maximum_gpu_reserved_bytes"]:
            raise CoverageDiagnosticError(
                f"GPU reserved-memory guard exceeded: {reserved_bytes} > {limits['maximum_gpu_reserved_bytes']}"
            )
    return result


def _frequency_bucket(frequency: int) -> str:
    if frequency == 0:
        return "unseen_0"
    if frequency == 1:
        return "seen_1"
    if frequency <= 4:
        return "seen_2_4"
    if frequency <= 16:
        return "seen_5_16"
    if frequency <= 64:
        return "seen_17_64"
    return "seen_65_plus"


def _stats(correct: int, frequencies: list[int]) -> dict[str, Any]:
    rows = len(frequencies)
    return {
        "rows": rows,
        "correct": int(correct),
        "token_accuracy": (float(correct) / rows) if rows else None,
        "fit_frequency_min": min(frequencies) if frequencies else None,
        "fit_frequency_max": max(frequencies) if frequencies else None,
        "fit_frequency_mean": (sum(frequencies) / rows) if rows else None,
    }


def _coverage_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    groups: tuple[str, ...],
    validation_frequencies: torch.Tensor,
    fit_frequencies: torch.Tensor,
) -> dict[str, Any]:
    predictions = predictions.reshape(-1).to(device="cpu", dtype=torch.long)
    labels = labels.reshape(-1).to(device="cpu", dtype=torch.long)
    validation_frequencies = validation_frequencies.reshape(-1).to(device="cpu", dtype=torch.long)
    fit_frequencies = fit_frequencies.reshape(-1).to(device="cpu", dtype=torch.long)
    if predictions.shape != labels.shape or labels.shape != validation_frequencies.shape:
        raise CoverageDiagnosticError("prediction, label, and frequency geometry disagree")
    if len(groups) != int(labels.numel()):
        raise CoverageDiagnosticError("validation groups do not cover predictions")
    overall_correct = 0
    overall_freq: list[int] = []
    grouped: dict[str, dict[str, list[int]]] = {}
    grouped_correct: dict[str, dict[str, int]] = {}
    bucket_freq: dict[str, list[int]] = {}
    bucket_correct: dict[str, int] = {}
    for prediction, label, group, frequency_tensor in zip(
        predictions.tolist(), labels.tolist(), groups, validation_frequencies.tolist()
    ):
        if not group:
            raise CoverageDiagnosticError("validation group is empty")
        frequency = int(frequency_tensor)
        bucket = _frequency_bucket(frequency)
        correct = int(prediction == label)
        overall_correct += correct
        overall_freq.append(frequency)
        bucket_freq.setdefault(bucket, []).append(frequency)
        bucket_correct[bucket] = bucket_correct.get(bucket, 0) + correct
        grouped.setdefault(group, {}).setdefault(bucket, []).append(frequency)
        grouped_correct.setdefault(group, {})[bucket] = (
            grouped_correct.setdefault(group, {}).get(bucket, 0) + correct
        )
    by_bucket = {
        bucket: _stats(bucket_correct[bucket], bucket_freq[bucket])
        for bucket in sorted(bucket_freq)
    }
    by_group: dict[str, Any] = {}
    for group in sorted(grouped):
        buckets = grouped[group]
        group_correct = sum(grouped_correct[group].values())
        group_freq = [frequency for values in buckets.values() for frequency in values]
        by_group[group] = {
            "overall": _stats(group_correct, group_freq),
            "by_frequency_bucket": {
                bucket: _stats(grouped_correct[group][bucket], buckets[bucket])
                for bucket in sorted(buckets)
            },
        }
    validation_frequency_counts: dict[str, int] = {}
    for frequency in overall_freq:
        key = str(frequency)
        validation_frequency_counts[key] = validation_frequency_counts.get(key, 0) + 1
    validation_unique = torch.unique(labels)
    validation_unseen_unique = fit_frequencies.index_select(0, validation_unique).eq(0)
    return {
        "overall": _stats(overall_correct, overall_freq),
        "by_frequency_bucket": by_bucket,
        "by_validation_group": by_group,
        "validation_rows_by_exact_fit_frequency": dict(
            sorted(validation_frequency_counts.items(), key=lambda item: int(item[0]))
        ),
        "fit_label_coverage": {
            "distinct_labels_seen": int((fit_frequencies > 0).sum().item()),
            "vocabulary_size": int(fit_frequencies.shape[0]),
            "distinct_validation_labels": int(torch.unique(labels).numel()),
            "distinct_validation_labels_unseen": int(validation_unseen_unique.sum().item()),
        },
    }


def run(args: argparse.Namespace) -> int:
    if args.method_id not in METHOD_BIAS:
        raise CoverageDiagnosticError(f"unknown affine method: {args.method_id}")
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise CoverageDiagnosticError(f"output is create-only and already exists: {output}")
    if args.batch_size <= 0 or args.max_seconds <= 0:
        raise CoverageDiagnosticError("batch size and wall-time guard must be positive")
    device = _device(args.device)
    root = args.repository_root.expanduser().resolve()
    started_utc = _utc_now()
    evidence_path = _regular_file(args.fit_evidence, label="fit evidence")
    manifest_path = _regular_file(args.data_manifest, label="fit data manifest")
    evidence = _load_json(evidence_path, label="fit evidence")
    if evidence.get("status") != "fit_complete":
        raise CoverageDiagnosticError("fit evidence is not complete")
    if evidence.get("schema") != "token-reconstruction.trr0004-controlled-affine-ce-run.v1":
        raise CoverageDiagnosticError("fit evidence schema changed")
    method_evidence = evidence.get("methods")
    if not isinstance(method_evidence, Mapping) or not isinstance(method_evidence.get(args.method_id), Mapping):
        raise CoverageDiagnosticError("requested method is absent from fit evidence")
    method = method_evidence[args.method_id]
    selected = method.get("selected_artifact")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("path"), str):
        raise CoverageDiagnosticError("selected state evidence is missing")
    selected_path = _regular_file(
        Path(str(selected["path"])), label="selected affine decoder state"
    )
    _recorded_artifact(selected_path, selected, label="selected affine decoder state")
    expected_manifest_hash = evidence.get("fit_data", {}).get("manifest", {}).get("manifest_sha256")
    actual_manifest_hash = file_sha256(manifest_path)
    if expected_manifest_hash != actual_manifest_hash:
        raise CoverageDiagnosticError("fit evidence and requested data manifest differ")
    config = evidence.get("configuration")
    if not isinstance(config, Mapping):
        raise CoverageDiagnosticError("fit configuration is missing")
    fit_position_limit = config.get("fit_position_limit")
    fit_record_limit = config.get("fit_record_limit")
    fit_includes_bos = config.get("fit_includes_bos")
    if fit_includes_bos is not True or isinstance(fit_position_limit, bool):
        raise CoverageDiagnosticError("fit evidence does not declare the registered BOS/position contract")
    if fit_position_limit is not None:
        fit_position_limit = int(fit_position_limit)
    if fit_record_limit is not None:
        fit_record_limit = int(fit_record_limit)
    bundle = load_public_fit_bundle(manifest_path)
    fit_x, fit_y = bundle.fit_tensors(
        record_limit=fit_record_limit,
        position_limit=fit_position_limit,
        include_bos=True,
    )
    validation_x, validation_y = bundle.validation_tensors(
        include_bos=bool(config.get("validation_includes_bos", False))
    )
    expected_fit_rows = evidence.get("fit_data", {}).get("fit_rows_used")
    expected_validation_rows = evidence.get("fit_data", {}).get("validation_rows_used")
    if expected_fit_rows != int(fit_x.shape[0]) or expected_validation_rows != int(validation_x.shape[0]):
        raise CoverageDiagnosticError("fit evidence row counts do not match reconstructed streams")
    groups = bundle.validation_flat_groups(
        include_bos=bool(config.get("validation_includes_bos", False))
    )
    frequencies = torch.bincount(fit_y.to(device="cpu", dtype=torch.long), minlength=bundle.vocabulary_size)
    validation_frequencies = frequencies.index_select(0, validation_y.to(device="cpu", dtype=torch.long))
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    preflight_apps = _query_compute_apps() if device.type == "cuda" else "not_queried_cpu"
    if preflight_apps:
        raise CoverageDiagnosticError(f"GPU compute applications are active: {preflight_apps}")
    preflight = _guard(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    deadline = started + args.max_seconds
    load_started = time.perf_counter()
    model = load_historical_affine_ce(
        selected_path,
        hidden_size=bundle.hidden_size,
        vocab_size=bundle.vocabulary_size,
        bias_mode=METHOD_BIAS[args.method_id],
        device=device,
    )
    _synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    _guard(args, device, deadline=deadline)
    prediction_started = time.perf_counter()
    predictions = direct_prediction_tensor(
        model,
        validation_x,
        bundle.embedding_table,
        device=device,
        batch_size=args.batch_size,
    )
    _synchronize(device)
    prediction_seconds = time.perf_counter() - prediction_started
    if predictions.shape != validation_y.shape:
        raise CoverageDiagnosticError("decoder predictions do not cover public validation")
    metrics = _coverage_metrics(
        predictions, validation_y, groups, validation_frequencies, frequencies
    )
    _guard(args, device, deadline=deadline)
    _synchronize(device)
    ended = time.perf_counter()
    peak = {
        "process_max_rss_bytes": int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        peak["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        peak["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    result = {
        "schema": SCRIPT_SCHEMA,
        "task_id": TASK_ID,
        "status": "coverage_diagnostic_complete",
        "claim_scope": "public validation coverage diagnostic; no replacement claim",
        "truth_policy": "public auxiliary validation labels only; no evaluator-private truth or target weights",
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "command": {
            "argv": [str(value) for value in __import__("sys").argv],
            "cwd": str(Path.cwd()),
            "python": __import__("sys").executable,
            "environment": _safe_environment(),
        },
        "provenance": {
            "git_commit": _git_commit(root),
            "script": _file_record(Path(__file__), label="coverage diagnostic script"),
            "fit_evidence": _file_record(evidence_path, label="fit evidence"),
            "data_manifest": _file_record(manifest_path, label="fit data manifest"),
            "selected_state": _coverage_json_path(selected_path),
        },
        "binding": {
            "method_id": args.method_id,
            "bias_mode": METHOD_BIAS[args.method_id],
            "selected_step": method.get("selected_step"),
            "selected_state_sha256": selected["sha256"],
            "fit_position_limit_post_bos": fit_position_limit,
            "fit_record_limit": fit_record_limit,
            "fit_includes_bos": True,
            "validation_includes_bos": bool(config.get("validation_includes_bos", False)),
            "fit_rows_used": int(fit_x.shape[0]),
            "validation_rows_used": int(validation_x.shape[0]),
            "fit_supervision_label_count_sha256": tensor_sha256(frequencies),
        },
        "preflight": {
            "guard": _resource_limits(args),
            "before_model_load": preflight,
            "gpu_compute_apps": preflight_apps,
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "validation_prediction_seconds": prediction_seconds,
            "wall_seconds": ended - started,
        },
        "peak_memory": peak,
        "runtime_components": {
            "public_embedding_table_required": True,
            "public_prefix_calls": 0,
            "candidate_simulations": 0,
            "a2_fallback": False,
        },
        "predictions": {
            "rows": int(predictions.shape[0]),
            "prediction_sha256": tensor_sha256(predictions),
            "serialization": "hash only; per-row labels/predictions are not copied",
        },
        "coverage": metrics,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "method_id": args.method_id}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-evidence", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=6.0)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=16.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (CoverageDiagnosticError, HistoricalAffineCEError, OSError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
