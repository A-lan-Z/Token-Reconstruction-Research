#!/usr/bin/env python3
"""Freeze and score the TRR-0004 five-method confirmation matrix.

This is a thin orchestration layer around ``trr0004_fresh_confirmation``.
``freeze`` validates all public prediction artifacts and writes an immutable
receipt outside the prediction directory.  ``score`` validates that receipt
and the complete public matrix again before it reads the private truth
sidecar.  Everything after that boundary is ordinary metric aggregation;
the sidecar is never touched by the pre-truth path.

The scorer reports per-cell and per-record results, paired condition
comparisons, public-fit frequency bins, position bins, and paired record
bootstrap intervals.  Prediction timing is read from the isolated prediction
receipts when available and is kept separate from cold loading and peak
memory.  This file does not generate observations or predictions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
import torch

import trr0004_fresh_confirmation as fc
from token_reconstruction.dual_benchmark import score_predictions
from token_reconstruction.footing import (
    FootingError,
    external_file_record,
    file_record,
    sha256_file,
)
from token_reconstruction.freeze import (
    FreezeError,
    create_freeze_receipt,
    verify_freeze_receipt,
)


TASK_ID = "TRR-0004"
SCHEMA = "token-reconstruction.trr0004-confirmation-score.v1"
FREEZE_SCHEMA = "token-reconstruction.trr0004-confirmation-freeze.v1"
DEFAULT_BOOTSTRAP_SEED = 4004
DEFAULT_BOOTSTRAP_DRAWS = 2000
FIT_FREQUENCY_BINS = (
    ("0", 0, 0),
    ("1-4", 1, 4),
    ("5-19", 5, 19),
    ("20+", 20, None),
)
POSITION_BINS = (
    ("1-15", 1, 15),
    ("16-39", 16, 39),
    ("40-79", 40, 79),
    ("80+", 80, None),
)


class ConfirmationScoreError(RuntimeError):
    """Raised when confirmation inputs or scoring evidence are incomplete."""


def _json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConfirmationScoreError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmationScoreError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ConfirmationScoreError(f"{description} must be a JSON object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ConfirmationScoreError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _outside(path: Path, directory: Path, *, description: str) -> None:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return
    raise ConfirmationScoreError(f"{description} must be outside prediction root: {path}")


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ConfirmationScoreError(f"{description} is unavailable: {path}")
    try:
        return file_record(path, repository_root=root.resolve())
    except (FootingError, OSError, ValueError):
        try:
            return external_file_record(path)
        except FootingError as exc:
            raise ConfirmationScoreError(f"unable to bind {description}: {path}") from exc


def _scorer_record(root: Path) -> dict[str, Any]:
    return _record(Path(__file__).resolve(), root=root, description="confirmation scorer")


def _load_context(
    *,
    root: Path,
    panel_path: Path,
    selection_plan_path: Path,
    registration_path: Path,
) -> tuple[dict[str, Any], tuple[fc.FreshCell, ...], dict[str, Any]]:
    panel = fc.load_fresh_panel(panel_path, repository_root=root)
    cells = fc.load_fresh_cells(panel, repository_root=root)
    registration = fc.load_confirmation_registration(
        registration_path,
        repository_root=root,
        panel_path=panel_path,
        selection_plan_path=selection_plan_path,
    )
    if tuple(registration["method_ids"]) != fc.METHOD_IDS:
        raise ConfirmationScoreError("confirmation registration is not the fixed five-method set")
    return panel, cells, registration


def _truth_binding(path: Path) -> dict[str, Any]:
    value = _json(path, description="truth binding")
    if value.get("schema") != fc.TRUTH_BINDING_SCHEMA or value.get("task_id") != TASK_ID:
        raise ConfirmationScoreError("truth binding identity changed")
    return value


def _gate_kwargs(
    *,
    receipt_path: Path,
    root: Path,
    truth_path: Path,
    output_root: Path,
    panel_path: Path,
    selection_plan_path: Path,
    registration_path: Path,
    registration: Mapping[str, Any],
    truth_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "receipt_path": receipt_path,
        "repository_root": root,
        "truth_path": truth_path,
        "output_root": output_root,
        "panel_path": panel_path,
        "selection_plan_path": selection_plan_path,
        "registration_path": registration_path,
        "method_ids": tuple(registration["method_ids"]),
        "expected_bindings": registration["bindings"],
        "candidate_policies": registration["candidate_policies"],
        "truth_binding": truth_binding,
    }


def freeze_confirmation(
    *,
    root: Path,
    panel_path: Path,
    selection_plan_path: Path,
    registration_path: Path,
    truth_binding_path: Path,
    output_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Validate public predictions and write the receipt before any truth read."""

    _outside(receipt_path, output_root, description="freeze receipt")
    panel, _cells, registration = _load_context(
        root=root,
        panel_path=panel_path,
        selection_plan_path=selection_plan_path,
        registration_path=registration_path,
    )
    truth_binding = _truth_binding(truth_binding_path)
    panel_record = file_record(panel_path, repository_root=root)
    plan_record = file_record(selection_plan_path, repository_root=root)
    if panel.get("selection_plan_sha256") != plan_record["sha256"]:
        raise ConfirmationScoreError("panel selection-plan binding changed")
    sidecar = truth_binding.get("sidecar")
    if not isinstance(sidecar, Mapping) or not isinstance(sidecar.get("path"), str):
        raise ConfirmationScoreError("truth binding has no sidecar path")
    _outside(Path(str(sidecar["path"])), output_root, description="truth sidecar")
    validated = fc.validate_complete_confirmation_predictions(
        output_root,
        panel_path=panel_path,
        repository_root=root,
        method_ids=registration["method_ids"],
        expected_bindings=registration["bindings"],
        candidate_policies=registration["candidate_policies"],
    )
    metadata = {
        "task_id": TASK_ID,
        "panel_sha256": panel_record["sha256"],
        "selection_plan_sha256": plan_record["sha256"],
        "method_ids": list(registration["method_ids"]),
        "registration_sha256": sha256_file(registration_path),
        "truth_binding": dict(truth_binding),
        "scorer": _scorer_record(root),
        "public_gate": {
            "validated_artifact_count": len(validated),
            "expected_artifact_count": 4 * len(registration["method_ids"]),
            "truth_opened": False,
        },
    }
    payload = create_freeze_receipt(
        repository_root=root,
        frozen_root=output_root,
        plan_path=selection_plan_path,
        receipt_path=receipt_path,
        preregistration_commit=str(
            registration["bindings"][registration["method_ids"][0]]["code_commit"]
        ),
        created_utc=datetime.now(timezone.utc).isoformat(),
        metadata=metadata,
    )
    verify_freeze_receipt(receipt_path, repository_root=root)
    result = {
        "schema": FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_MATRIX_FROZEN_NO_TRUTH_OPENED",
        "receipt": _record(receipt_path, root=root, description="freeze receipt"),
        "frozen_root": str(output_root.relative_to(root).as_posix()),
        "validated_artifact_count": len(validated),
        "truth_opened": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_fit_frequency(
    path: Path,
    *,
    vocab_size: int,
    token_key: str,
    mask_key: str,
    root: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ConfirmationScoreError(f"public fit artifact is unavailable: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if token_key not in keys or mask_key not in keys:
                raise ConfirmationScoreError(
                    f"public fit artifact needs {token_key!r} and {mask_key!r} tensors"
                )
            token_ids = handle.get_tensor(token_key).to(torch.long).contiguous()
            mask = handle.get_tensor(mask_key).to(torch.bool).contiguous()
    except ConfirmationScoreError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ConfirmationScoreError(f"public fit artifact is unreadable: {path}") from exc
    if token_ids.ndim != 2 or mask.shape != token_ids.shape or token_ids.shape[1] < 2:
        raise ConfirmationScoreError("public fit token/mask geometry changed")
    if not mask[:, 0].all().item() or (mask[:, 1:] > mask[:, :-1]).any().item():
        raise ConfirmationScoreError("public fit mask is not BOS/right-padded")
    active = mask.clone()
    active[:, 0] = False
    if token_ids[active].lt(0).any().item() or token_ids[active].ge(vocab_size).any().item():
        raise ConfirmationScoreError("public fit labels are outside the vocabulary")
    counts = torch.bincount(token_ids[active], minlength=vocab_size).to(torch.long)
    return counts, {
        "source": _record(path, root=root, description="public fit frequency artifact"),
        "tensor_keys": {"token_ids": token_key, "attention_mask": mask_key},
        "records": int(token_ids.shape[0]),
        "sequence_tokens": int(token_ids.shape[1]),
        "post_bos_fit_examples": int(active.sum().item()),
        "fit_unique_tokens": int(counts.gt(0).sum().item()),
        "vocab_size": vocab_size,
    }


def _bin_index(values: torch.Tensor, bins: Sequence[tuple[str, int, int | None]]) -> torch.Tensor:
    result = torch.full(values.shape, -1, dtype=torch.long)
    for index, (_name, lower, upper) in enumerate(bins):
        match = values.ge(lower)
        if upper is not None:
            match &= values.le(upper)
        result[match] = index
    if result.eq(-1).any().item():
        raise ConfirmationScoreError("group bins do not cover all scored rows")
    return result


def _group_metrics(correct: torch.Tensor, groups: torch.Tensor, bins: Sequence[tuple[str, int, int | None]]) -> dict[str, Any]:
    correct = correct.to(torch.bool).reshape(-1).cpu()
    groups = groups.to(torch.long).reshape(-1).cpu()
    if correct.shape != groups.shape:
        raise ConfirmationScoreError("group metric rows do not agree")
    result: dict[str, Any] = {}
    for index, (name, lower, upper) in enumerate(bins):
        selected = groups.eq(index)
        examples = int(selected.sum().item())
        correct_count = int(correct[selected].sum().item())
        result[name] = {
            "lower": lower,
            "upper": upper,
            "examples": examples,
            "correct_tokens": correct_count,
            "token_accuracy": correct_count / examples if examples else None,
        }
    return result


def _paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right) or not left or draws <= 0:
        raise ConfirmationScoreError("paired bootstrap inputs are invalid")
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ConfirmationScoreError("paired bootstrap inputs are non-finite")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, left_array.size, size=(draws, left_array.size))
    left_estimates = left_array[indices].mean(axis=1)
    right_estimates = right_array[indices].mean(axis=1)
    delta_estimates = left_estimates - right_estimates
    return {
        "unit": "paired record token accuracy",
        "records": int(left_array.size),
        "draws": int(draws),
        "seed": int(seed),
        "left_estimate": float(left_array.mean()),
        "right_estimate": float(right_array.mean()),
        "delta_estimate": float(left_array.mean() - right_array.mean()),
        "delta_ci95_percentile": [
            float(np.quantile(delta_estimates, 0.025)),
            float(np.quantile(delta_estimates, 0.975)),
        ],
    }


def _load_prediction(path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            predictions = handle.get_tensor("predictions").to(torch.long).contiguous()
            candidates = (
                handle.get_tensor("candidates").to(torch.long).contiguous()
                if "candidates" in keys
                else None
            )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ConfirmationScoreError(f"prediction artifact is unreadable: {path}") from exc
    return predictions, candidates


def _timing_sources(
    *,
    output_root: Path,
    evidence_paths: Sequence[Path],
    root: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Collect timing from isolated receipts and copied cell records.

    Top-level ``run_evidence.json`` files supply cold and run-level peak
    fields.  Per-cell ``*.run.json`` files supply steady intervals after a
    five-root byte-copy merge; they are deliberately not counted as five
    additional cold runs.
    """

    top_level: list[tuple[Path, dict[str, Any]]] = []
    candidates = list(evidence_paths)
    default = output_root / "run_evidence.json"
    if default.is_file() and default not in candidates:
        candidates.append(default)
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            top_level.append((path, _json(path, description="prediction run evidence")))

    timing: dict[tuple[str, str], dict[str, Any]] = {}
    cell_sources: list[dict[str, Any]] = []

    def add_timing(source_path: Path, row: Mapping[str, Any]) -> None:
        cell_id = row.get("cell_id")
        method_id = row.get("method_id")
        if not isinstance(cell_id, str) or not isinstance(method_id, str):
            return
        value = row.get("timed_interval_total_seconds")
        if not isinstance(value, (int, float)):
            return
        source = _record(source_path, root=root, description="prediction timing evidence")
        item = {"source": source, "record": dict(row)}
        key = (cell_id, method_id)
        previous = timing.get(key)
        if previous is not None and previous["record"] != item["record"]:
            raise ConfirmationScoreError(f"conflicting timing records for {cell_id}/{method_id}")
        timing[key] = item

    cold_runs: list[dict[str, Any]] = []
    for path, evidence in top_level:
        source = _record(path, root=root, description="prediction run evidence")
        method_timings = evidence.get("method_timings")
        if isinstance(method_timings, Mapping):
            for method_id, rows in method_timings.items():
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, Mapping):
                            add_timing(path, {"method_id": method_id, **dict(row)})
        execution = evidence.get("execution")
        execution = execution if isinstance(execution, Mapping) else {}
        model = evidence.get("model")
        model = model if isinstance(model, Mapping) else {}
        startup = evidence.get("startup")
        startup = startup if isinstance(startup, Mapping) else None
        cold_peak_memory = evidence.get("cold_peak_memory")
        cold_peak_memory = cold_peak_memory if isinstance(cold_peak_memory, Mapping) else None
        per_cell_peak_memory = evidence.get("per_cell_peak_memory")
        per_cell_peak_memory = (
            per_cell_peak_memory if isinstance(per_cell_peak_memory, Mapping) else None
        )
        cold_runs.append(
            {
                "source": source,
                "method_id": evidence.get("registration", {}).get("executed_method_id")
                if isinstance(evidence.get("registration"), Mapping)
                else None,
                "wall_seconds": evidence.get("wall_seconds"),
                "started_utc": execution.get("started_utc", evidence.get("started_utc")),
                "ended_utc": execution.get("ended_utc", evidence.get("ended_utc")),
                "startup": dict(startup) if startup is not None else None,
                "cold_components": {
                    key: model.get(key)
                    for key in (
                        "model_load_seconds",
                        "public_embedding_load_seconds",
                        "method_state_load_seconds",
                    )
                    if model.get(key) is not None
                },
                "peak_memory": execution.get("peak_memory", evidence.get("peak_memory")),
                "cold_peak_memory": dict(cold_peak_memory) if cold_peak_memory is not None else None,
                "per_cell_peak_memory": (
                    {str(key): dict(value) for key, value in per_cell_peak_memory.items()}
                    if per_cell_peak_memory is not None
                    and all(isinstance(value, Mapping) for value in per_cell_peak_memory.values())
                    else None
                ),
            }
        )

    # A footing merge normally copies these records beside each prediction.
    # Discover them without treating them as independent cold runs.
    for path in sorted(output_root.rglob("*.run.json")):
        if path.is_symlink() or not path.is_file():
            continue
        evidence = _json(path, description="per-cell timing evidence")
        row = evidence.get("method")
        if isinstance(row, Mapping):
            add_timing(path, row)
            cell_sources.append(_record(path, root=root, description="per-cell timing evidence"))

    return timing, {
        "source_count": len(top_level),
        "cell_timing_source_count": len(cell_sources),
        "sources": [
            _record(path, root=root, description="prediction run evidence")
            for path, _evidence in top_level
        ],
        "cell_timing_sources": cell_sources,
        "cold_runs": cold_runs,
    }

def _steady_costs(
    timing: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    method_ids: Sequence[str],
    cells: Sequence[fc.FreshCell],
) -> dict[str, Any]:
    per_method: dict[str, Any] = {}
    missing: list[str] = []
    for method_id in method_ids:
        rows: list[dict[str, Any]] = []
        total_interval = 0.0
        total_measured = 0.0
        total_records = 0
        per_record_measured: list[float] = []
        peak_max = {
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
            "process_max_rss_bytes": None,
        }
        for cell in cells:
            item = timing.get((cell.cell_id, method_id))
            if item is None:
                missing.append(f"{cell.cell_id}/{method_id}")
                continue
            record = item["record"]
            interval = record.get("timed_interval_total_seconds")
            measured = record.get("measured_seconds_sum")
            if not isinstance(interval, (int, float)) or not isinstance(measured, (int, float)):
                missing.append(f"{cell.cell_id}/{method_id}")
                continue
            records = int(record.get("records", cell.records))
            measured_runs = int(record.get("measured_runs_per_record", 3))
            if records <= 0 or measured_runs <= 0:
                raise ConfirmationScoreError(f"invalid measured timing counts for {cell.cell_id}/{method_id}")
            total_interval += float(interval)
            total_measured += float(measured)
            total_records += records
            raw_per_record = record.get("per_record_measured_seconds")
            if isinstance(raw_per_record, list) and len(raw_per_record) == records:
                values = [float(value) for value in raw_per_record]
                if not all(np.isfinite(values)):
                    raise ConfirmationScoreError(f"non-finite per-record timing for {cell.cell_id}/{method_id}")
                per_record_measured.extend(value / measured_runs for value in values)
            peak = record.get("peak_memory")
            if isinstance(peak, Mapping):
                for key in peak_max:
                    value = peak.get(key)
                    if isinstance(value, int):
                        previous = peak_max[key]
                        peak_max[key] = value if previous is None else max(previous, value)
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "style": cell.style,
                    "condition": cell.condition,
                    "records": records,
                    "warmup_runs_per_record": record.get("warmup_runs_per_record", 1),
                    "measured_runs_per_record": measured_runs,
                    "timed_interval_total_seconds": float(interval),
                    "warmup_seconds_sum": record.get("warmup_seconds_sum"),
                    "measured_seconds_sum": float(measured),
                    "deployed_measured_seconds_sum_per_one_run": float(measured) / measured_runs,
                    "deployed_latency_mean_seconds_per_record": float(measured) / (records * measured_runs),
                    "deployed_latency_median_seconds_per_record": (
                        float(np.median([float(value) / measured_runs for value in raw_per_record]))
                        if isinstance(raw_per_record, list) and len(raw_per_record) == records
                        else None
                    ),
                    "timing_samples_available": isinstance(raw_per_record, list),
                    "source": item["source"],
                    "peak_memory": peak,
                }
            )
        per_method[method_id] = {
            "cells": rows,
            "cell_count": len(rows),
            "total_timed_interval_seconds": total_interval if len(rows) == len(cells) else None,
            "total_measured_seconds_sum": total_measured if len(rows) == len(cells) else None,
            "deployed_measured_seconds_sum_per_one_run": (
                total_measured / 3.0 if len(rows) == len(cells) else None
            ),
            "deployed_latency_mean_seconds_per_record": (
                total_measured / (total_records * 3.0)
                if len(rows) == len(cells) and total_records
                else None
            ),
            "deployed_latency_median_seconds_per_record": (
                float(np.median(per_record_measured))
                if len(rows) == len(cells) and len(per_record_measured) == total_records
                else None
            ),
            "median_unavailable_reason": (
                None
                if len(rows) == len(cells) and len(per_record_measured) == total_records
                else "producer receipt retained sums but not every per-record measured sample"
            ),
            "peak_memory_max_across_cells": peak_max,
        }
    return {
        "definition": "CPU activation H -> device preprocessing -> method -> predicted IDs CPU",
        "timed_interval_includes": "one warmup plus three measured runs per record",
        "deployed_latency_uses": "measured_seconds_sum / 3, with per-record means/medians where producer samples exist",
        "cold_costs_excluded": True,
        "per_method": per_method,
        "missing_cell_methods": missing,
        "complete": not missing,
    }

def _score(
    *,
    root: Path,
    panel_path: Path,
    selection_plan_path: Path,
    registration_path: Path,
    truth_binding_path: Path,
    truth_path: Path,
    output_root: Path,
    receipt_path: Path,
    result_path: Path,
    fit_data_path: Path | None,
    fit_token_key: str,
    fit_mask_key: str,
    evidence_paths: Sequence[Path],
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    _outside(result_path, output_root, description="score result")
    panel, cells, registration = _load_context(
        root=root,
        panel_path=panel_path,
        selection_plan_path=selection_plan_path,
        registration_path=registration_path,
    )
    truth_binding = _truth_binding(truth_binding_path)
    gate = fc.validate_before_confirmation_truth(
        **_gate_kwargs(
            receipt_path=receipt_path,
            root=root,
            truth_path=truth_path,
            output_root=output_root,
            panel_path=panel_path,
            selection_plan_path=selection_plan_path,
            registration_path=registration_path,
            registration=registration,
            truth_binding=truth_binding,
        )
    )
    metadata = gate.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("scorer") != _scorer_record(root):
        raise ConfirmationScoreError("freeze receipt scorer binding changed")

    # This is the first actual truth read in this process.  The complete
    # public gate above has already rehashed the receipt, states, registration,
    # panel, and every prediction artifact.
    truth = fc.validate_confirmation_truth_sidecar(
        truth_path,
        cells=cells,
        truth_binding=truth_binding,
    )
    frequency_counts = None
    frequency_source: dict[str, Any] | None = None
    if fit_data_path is not None:
        frequency_counts, frequency_source = _load_fit_frequency(
            fit_data_path,
            vocab_size=fc.VOCAB_SIZE,
            token_key=fit_token_key,
            mask_key=fit_mask_key,
            root=root,
        )

    timing_map, isolated_costs = _timing_sources(
        output_root=output_root,
        evidence_paths=evidence_paths,
        root=root,
    )
    method_ids = tuple(registration["method_ids"])
    cell_results: dict[str, Any] = {}
    records_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        active = cell.attention_mask.to(torch.bool)
        scored = active.clone()
        scored[:, 0] = False
        for method_id in method_ids:
            prediction_path = fc.expected_prediction_path(output_root, cell=cell, method_id=method_id)
            predictions, candidates = _load_prediction(prediction_path)
            metrics, per_record = score_predictions(
                predictions=predictions,
                truth=truth[cell.cell_id],
                attention_mask=cell.attention_mask,
                candidates=candidates,
                record_ids=cell.record_ids,
            )
            correct = predictions[scored].eq(truth[cell.cell_id][scored]).cpu()
            row: dict[str, Any] = {
                "cell_id": cell.cell_id,
                "style": cell.style,
                "condition": cell.condition,
                "method_id": method_id,
                "metrics": metrics,
                "per_record": per_record,
                "prediction_artifact": _record(prediction_path, root=root, description="prediction artifact"),
                "timing": timing_map.get((cell.cell_id, method_id), {
                    "status": "timing_receipt_unavailable",
                    "source": None,
                }),
            }
            position_values = cell.position_ids[scored].to(torch.long).cpu()
            row["position_bins"] = _group_metrics(correct, _bin_index(position_values, POSITION_BINS), POSITION_BINS)
            if frequency_counts is not None:
                labels = truth[cell.cell_id][scored].to(torch.long).cpu()
                token_counts = frequency_counts.index_select(0, labels)
                row["frequency_bins"] = _group_metrics(
                    correct,
                    _bin_index(token_counts, FIT_FREQUENCY_BINS),
                    FIT_FREQUENCY_BINS,
                )
            records_by_key[(cell.style, cell.condition, method_id)] = per_record
            cell_results[f"{cell.cell_id}__{method_id}"] = row

    paired: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for style in fc.STYLE_ORDER:
        for method_id in method_ids:
            left = records_by_key[(style, "public_base", method_id)]
            right = records_by_key[(style, "public_lora_2601", method_id)]
            if len(left) != len(right):
                raise ConfirmationScoreError(f"paired record count changed for {style}/{method_id}")
            rows: list[dict[str, Any]] = []
            for left_row, right_row in zip(left, right):
                if left_row["record_id"] != right_row["record_id"]:
                    raise ConfirmationScoreError(f"paired record ordering changed for {style}/{method_id}")
                rows.append(
                    {
                        "record_id": left_row["record_id"],
                        "public_base": {
                            "token_accuracy": left_row["token_accuracy"],
                            "correct_tokens": left_row["correct_tokens"],
                            "scored_tokens": left_row["scored_tokens"],
                            "exact_record": left_row["exact_record"],
                        },
                        "public_lora_2601": {
                            "token_accuracy": right_row["token_accuracy"],
                            "correct_tokens": right_row["correct_tokens"],
                            "scored_tokens": right_row["scored_tokens"],
                            "exact_record": right_row["exact_record"],
                        },
                        "token_accuracy_delta": float(left_row["token_accuracy"] - right_row["token_accuracy"]),
                        "correct_tokens_delta": int(left_row["correct_tokens"] - right_row["correct_tokens"]),
                    }
                )
            key = f"{style}__{method_id}"
            paired[key] = rows
            bootstrap[key] = _paired_bootstrap(
                [float(row["public_base"]["token_accuracy"]) for row in rows],
                [float(row["public_lora_2601"]["token_accuracy"]) for row in rows],
                draws=bootstrap_draws,
                seed=bootstrap_seed,
            )

    steady = _steady_costs(timing_map, method_ids=method_ids, cells=cells)
    peak_values: list[int] = []
    for item in timing_map.values():
        peak = item["record"].get("peak_memory")
        if isinstance(peak, Mapping):
            for key in ("cuda_peak_reserved_bytes", "process_max_rss_bytes"):
                value = peak.get(key)
                if isinstance(value, int):
                    peak_values.append(value)
    result = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "FRESH_CONFIRMATION_SCORED_AFTER_COMPLETE_PUBLIC_GATE",
        "claim_scope": "fresh public confirmation matrix; exploratory TRR-0004 evidence, no replacement claim",
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "truth_gate": {
            "receipt": _record(receipt_path, root=root, description="freeze receipt"),
            "verified_before_truth": True,
            "truth_sidecar": _record(truth_path, root=root, description="truth sidecar"),
            "truth_binding": _record(truth_binding_path, root=root, description="truth binding"),
            "truth_opened_after_gate": True,
        },
        "inputs": {
            "panel": file_record(panel_path, repository_root=root),
            "selection_plan": file_record(selection_plan_path, repository_root=root),
            "registration": file_record(registration_path, repository_root=root),
            "method_ids": list(method_ids),
            "cells": [cell.cell_id for cell in cells],
            "frequency_reference": frequency_source,
        },
        "bootstrap": {
            "seed": bootstrap_seed,
            "draws": bootstrap_draws,
            "unit": "paired record",
            "comparisons": bootstrap,
        },
        "paired_record_comparisons": paired,
        "cells": cell_results,
        "costs": {
            "isolated_prediction_runs": isolated_costs,
            "steady_state": steady,
            "peak_observed_values": {
                "timing_peak_values_bytes": peak_values,
                "maximum_timing_peak_value_bytes": max(peak_values) if peak_values else None,
                "per_method_max_across_cells": {
                    method_id: steady["per_method"][method_id]["peak_memory_max_across_cells"]
                    for method_id in method_ids
                },
                "interpretation": "timing records retain their own CUDA/RSS fields; values are not a single method-independent peak",
            },
        },
        "runtime_components": {
            method_id: {
                "candidate_policy": registration["candidate_policies"][method_id],
                "public_prefix_calls": "from method timing/evidence; scorer does not infer zero",
                "candidate_simulations": "from method timing/evidence; scorer does not infer zero",
            }
            for method_id in method_ids
        },
        "limitations": [
            "The public_base cell is a matched public control; public_lora_2601 is one synthetic target-shift diagnostic.",
            "Public fit frequency bins describe coverage of the shared large fit and do not establish target-transfer coverage.",
            "Canonical dual-benchmark confirmation and any replacement claim remain outside this exploratory result.",
        ],
    }
    _write_create_only(result_path, result)
    print(json.dumps({"result": str(result_path), "cells": len(cell_results), "truth_opened_after_gate": True}, sort_keys=True))
    return result


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--truth-binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="validate public matrix and write freeze receipt")
    _common(freeze)
    score = sub.add_parser("score", help="revalidate, open truth after gate, and score")
    _common(score)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--result", type=Path, required=True)
    score.add_argument("--fit-data", type=Path)
    score.add_argument("--fit-token-key", default="token_ids")
    score.add_argument("--fit-mask-key", default="attention_mask")
    score.add_argument("--prediction-evidence", type=Path, action="append", default=[])
    score.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    score.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.expanduser().resolve()
    try:
        if args.command == "freeze":
            freeze_confirmation(
                root=root,
                panel_path=args.panel.expanduser().resolve(),
                selection_plan_path=args.selection_plan.expanduser().resolve(),
                registration_path=args.registration.expanduser().resolve(),
                truth_binding_path=args.truth_binding.expanduser().resolve(),
                output_root=args.output_root.expanduser().resolve(),
                receipt_path=args.receipt.expanduser().resolve(),
            )
            return 0
        if args.bootstrap_draws <= 0:
            raise ConfirmationScoreError("bootstrap draws must be positive")
        _score(
            root=root,
            panel_path=args.panel.expanduser().resolve(),
            selection_plan_path=args.selection_plan.expanduser().resolve(),
            registration_path=args.registration.expanduser().resolve(),
            truth_binding_path=args.truth_binding.expanduser().resolve(),
            truth_path=args.truth.expanduser().resolve(),
            output_root=args.output_root.expanduser().resolve(),
            receipt_path=args.receipt.expanduser().resolve(),
            result_path=args.result.expanduser().resolve(),
            fit_data_path=args.fit_data.expanduser().resolve() if args.fit_data else None,
            fit_token_key=args.fit_token_key,
            fit_mask_key=args.fit_mask_key,
            evidence_paths=[path.expanduser().resolve() for path in args.prediction_evidence],
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
        )
        return 0
    except (ConfirmationScoreError, fc.ConfirmationError, FreezeError) as exc:
        raise SystemExit(f"TRR-0004 confirmation scorer error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
