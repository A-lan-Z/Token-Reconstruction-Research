#!/usr/bin/env python3
"""Create compact TRR-0005 final result tables after truth-gated scoring.

This post-processing helper deliberately reads only JSON: the scorer result,
the predictor's compact run evidence, and the 32 per-cell ``*.run.json``
receipts.  It never opens prediction safetensors, activations, model weights,
source text, or evaluator truth.  It is intentionally create-only and must be
run only after the scorer has produced a successful final result.

Example (run only after root grants final assembly):

    PYTHONPATH=.:src:scripts .venv-trr0005/bin/python \
      experiments/TRR-0005/footing/summarize_confirmation.py \
      --result experiments/TRR-0005/fresh_confirmation_v1/result.json \
      --run-evidence experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/run_evidence.json \
      --predictions-root experiments/TRR-0005/fresh_confirmation_v1/predictions_v1 \
      --summary experiments/TRR-0005/fresh_confirmation_v1/summary.json \
      --report-tables experiments/TRR-0005/fresh_confirmation_v1/report_tables.md
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from token_reconstruction.trr0005_contract import (
    CONDITION_ORDER,
    EXPECTED_CELL_IDS,
    METHOD_IDS,
    RECORDS_PER_DOMAIN,
    STYLE_ORDER,
)


TASK_ID = "TRR-0005"
RESULT_STATUS = "FRESH_CONFIRMATION_SCORED_AFTER_COMPLETE_PUBLIC_GATE"
SUMMARY_SCHEMA = "token-reconstruction.trr0005-confirmation-summary.v1"
MARGIN = {
    "extra_context_token_accuracy_pp": 0.5,
    "extra_context_exact_record_rate_pp": 5.0,
    "enrichment_token_accuracy_pp": 2.0,
    "enrichment_exact_record_rate_pp": 5.0,
}
PRIMARY_LABELS = (
    "enriched__causal_vs_diagonal",
    "enriched__causal_vs_best_positionwise",
)
ENRICHMENT_STATES = (
    "joint_full_affine",
    "affine_causal_h_attention128",
    "affine_trained_diagonal_attention128",
)


class SummaryError(ValueError):
    """Raised when final JSON evidence is incomplete or inconsistent."""


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SummaryError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise SummaryError(f"{description} must be a JSON object: {path}")
    return dict(value)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise SummaryError(f"refusing to overwrite summary artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise SummaryError("summary is not JSON serializable") from exc
    path.write_text(encoded, encoding="utf-8")


def _finite_float(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{description} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SummaryError(f"{description} is not finite")
    return result


def _integer(value: Any, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"{description} is not an integer")
    return int(value)


def _optional_float(value: Any, *, description: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, description=description)


def _cell_method_key(cell_id: str, method_id: str) -> str:
    return f"{cell_id}__{method_id}"


def _required_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryError(f"{description} is not a mapping")
    return value


def _cell_rows(result: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    payload = result.get("cells_results")
    if not isinstance(payload, Mapping):
        raise SummaryError("scorer result has no cells_results mapping")
    expected = {
        (cell_id, method_id)
        for cell_id in EXPECTED_CELL_IDS
        for method_id in METHOD_IDS
    }
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, Mapping):
            raise SummaryError("cells_results contains a malformed row")
        cell_id = raw_value.get("cell_id")
        method_id = raw_value.get("method_id")
        if not isinstance(cell_id, str) or not isinstance(method_id, str):
            matches = [
                key
                for key in expected
                if _cell_method_key(*key) == raw_key
            ]
            if len(matches) != 1:
                raise SummaryError("cells_results row lacks cell/method identity")
            cell_id, method_id = matches[0]
        key = (cell_id, method_id)
        if key not in expected or key in rows:
            raise SummaryError(f"cells_results has an unknown or duplicate key: {raw_key}")
        rows[key] = dict(raw_value)
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise SummaryError(f"cells_results matrix changed: missing={missing!r} extra={extra!r}")
    return rows


def _compact_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _required_mapping(row.get("metrics"), description="cell metrics")
    fields = (
        "scored_tokens",
        "correct_tokens",
        "token_accuracy",
        "records",
        "exact_records",
        "exact_record_rate",
    )
    output: dict[str, Any] = {}
    for field in fields:
        value = metrics.get(field)
        if field in {"scored_tokens", "correct_tokens", "records", "exact_records"}:
            output[field] = _integer(value, description=f"metrics.{field}")
        elif value is None:
            output[field] = None
        else:
            output[field] = _finite_float(value, description=f"metrics.{field}")
    if output["records"] != RECORDS_PER_DOMAIN:
        raise SummaryError("cell metrics record count changed")
    return output


def _paired_exact_bootstrap(
    comparison: Mapping[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any] | None:
    """Compute a descriptive paired exact-rate interval from score rows.

    The scorer stores ordered source-level differences so this calculation does
    not touch truth or predictions.  It uses the same seed and source ordering
    contract as the token bootstrap and is not treated as an exact guarantee.
    """

    raw_rows = comparison.get("paired_record_differences")
    if not isinstance(raw_rows, list) or not raw_rows:
        return None
    baseline_values: list[float] = []
    method_values: list[float] = []
    record_ids: list[str] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise SummaryError(f"paired exact row {index} is malformed")
        record_id = row.get("record_id")
        baseline = row.get("baseline")
        method = row.get("method")
        if not isinstance(record_id, str) or not record_id:
            raise SummaryError("paired exact row has invalid record ID")
        if not isinstance(baseline, Mapping) or not isinstance(method, Mapping):
            raise SummaryError("paired exact row lacks baseline/method records")
        baseline_exact = baseline.get("exact_record")
        method_exact = method.get("exact_record")
        if not isinstance(baseline_exact, bool) or not isinstance(method_exact, bool):
            raise SummaryError("paired exact row has invalid exact flags")
        record_ids.append(record_id)
        baseline_values.append(float(baseline_exact))
        method_values.append(float(method_exact))
    if len(set(record_ids)) != len(record_ids):
        raise SummaryError("paired exact bootstrap record IDs are not unique")
    if len(record_ids) != RECORDS_PER_DOMAIN:
        raise SummaryError("paired exact bootstrap record count changed")
    if draws <= 0:
        raise SummaryError("paired exact bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(record_ids), size=(draws, len(record_ids)))
    baseline = np.asarray(baseline_values, dtype=np.float64)
    method = np.asarray(method_values, dtype=np.float64)
    delta = (method[indices] - baseline[indices]).mean(axis=1)
    point = float(method.mean() - baseline.mean())
    return {
        "unit": "paired source record; exact-record indicator difference",
        "records": len(record_ids),
        "draws": int(draws),
        "seed": int(seed),
        "delta_estimate": point,
        "delta_estimate_pp": 100.0 * point,
        "delta_ci95_percentile": [
            float(np.quantile(delta, 0.025)),
            float(np.quantile(delta, 0.975)),
        ],
        "delta_ci95_percentile_pp": [
            100.0 * float(np.quantile(delta, 0.025)),
            100.0 * float(np.quantile(delta, 0.975)),
        ],
    }


def _token_effect(comparison: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = _required_mapping(
        comparison.get("token_bootstrap"), description="token bootstrap"
    )
    point_rate = bootstrap.get("delta_estimate", comparison.get("micro_token_accuracy_delta"))
    point = _finite_float(point_rate, description="token delta")
    ci = bootstrap.get("delta_ci95_percentile")
    if not isinstance(ci, Sequence) or isinstance(ci, (str, bytes)) or len(ci) != 2:
        raise SummaryError("token bootstrap 95% interval is malformed")
    ci_values = [
        _finite_float(value, description="token bootstrap CI endpoint") for value in ci
    ]
    upper = _optional_float(
        bootstrap.get("delta_upper_bound"), description="token upper bound"
    )
    return {
        "delta_pp": 100.0 * point,
        "ci95_pp": [100.0 * value for value in ci_values],
        "upper_bound_pp": None if upper is None else 100.0 * upper,
        "upper_tail_alpha": _optional_float(
            bootstrap.get("upper_tail_alpha"), description="token upper-tail alpha"
        ),
    }


def _exact_effect(
    comparison: Mapping[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    records = _integer(comparison.get("records"), description="comparison records")
    exact_delta = _finite_float(
        comparison.get("exact_record_delta"), description="exact record delta"
    )
    gain_loss = _required_mapping(
        comparison.get("gains_and_regressions"), description="exact gains/regressions"
    )
    gain = _integer(
        gain_loss.get("beneficial_exact_records"),
        description="beneficial exact records",
    )
    loss = _integer(
        gain_loss.get("harmful_exact_records"),
        description="harmful exact records",
    )
    beneficial_bound = _required_mapping(
        comparison.get("exact_beneficial_bound"),
        description="exact beneficial bound",
    )
    net_bound = _required_mapping(
        comparison.get("exact_net_benefit_bound"),
        description="exact net benefit bound",
    )
    bootstrap = _paired_exact_bootstrap(comparison, draws=draws, seed=seed)
    if bootstrap is None:
        ci95_pp: list[float] | None = None
    else:
        ci95_pp = list(bootstrap["delta_ci95_percentile_pp"])
    net_lower = net_bound.get("net_lower_pp")
    if net_lower is None and net_bound.get("net_lower_rate") is not None:
        net_lower = 100.0 * _finite_float(
            net_bound.get("net_lower_rate"), description="net exact lower rate"
        )
    net_upper = net_bound.get("net_upper_pp")
    if net_upper is None and net_bound.get("net_upper_rate") is not None:
        net_upper = 100.0 * _finite_float(
            net_bound.get("net_upper_rate"), description="net exact upper rate"
        )
    return {
        "delta_pp": 100.0 * exact_delta / records,
        "ci95_pp": ci95_pp,
        "ci95_source": (
            "paired_record_differences_source_bootstrap"
            if bootstrap is not None
            else "unavailable"
        ),
        "beneficial_exact_records": gain,
        "harmful_exact_records": loss,
        "net_exact_discordance": gain - loss,
        "beneficial_upper_bound_pp": _finite_float(
            beneficial_bound.get("beneficial_upper_pp"),
            description="beneficial exact upper bound",
        ),
        "net_lower_bound_pp": _finite_float(
            net_lower, description="net exact lower bound"
        ),
        "net_upper_bound_pp": _finite_float(
            net_upper, description="net exact upper bound"
        ),
        "tail_alpha_each": _finite_float(
            net_bound.get("tail_alpha_each"), description="exact tail alpha"
        ),
        "zero_discordance_is_not_no_effect": bool(
            gain == 0 and loss == 0
        ),
    }


def _comparison_summary(
    label: str,
    comparison: Mapping[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "baseline_method_id": comparison.get("baseline_method_id"),
        "method_id": comparison.get("method_id"),
        "records": _integer(comparison.get("records"), description="comparison records"),
        "token": _token_effect(comparison),
        "exact": _exact_effect(comparison, draws=draws, seed=seed),
    }


def _compact_frequency(row: Mapping[str, Any]) -> dict[str, Any]:
    frequency_references = row.get("frequency_references")
    if not isinstance(frequency_references, Mapping):
        raise SummaryError("cell result has no frequency_references mapping")
    output: dict[str, Any] = {}
    for reference in ("original", "enriched"):
        value = frequency_references.get(reference)
        if not isinstance(value, Mapping):
            raise SummaryError(f"frequency reference is missing: {reference}")
        joint = value.get("joint_frequency_position")
        if not isinstance(joint, Mapping):
            raise SummaryError("joint frequency-position diagnostic is missing")
        joint_rows = joint.get("rows")
        if not isinstance(joint_rows, Mapping):
            raise SummaryError("joint frequency-position rows are missing")
        output[reference] = {
            "frequency_reference_id": value.get("frequency_reference_id"),
            "frequency_bins": value.get("frequency_bins"),
            "joint_frequency_position_rows": dict(joint_rows),
        }
    return output


def _compact_gate(result: Mapping[str, Any]) -> dict[str, Any]:
    gate = _required_mapping(result.get("truth_gate"), description="truth gate")
    # The executable gate is nested in the scored-result schema. Keep the
    # outer truth_gate fields for the post-truth transition, while taking the
    # pre-truth artifact/timing counts and bindings from its public descriptor.
    # Do not infer these counts from prediction files.
    public_gate = gate.get("executable_public_gate")
    if not isinstance(public_gate, Mapping):
        public_gate = gate
    required = (
        "status",
        "verified_before_truth",
        "truth_opened_after_gate",
        "prediction_artifact_count",
        "timing_receipt_count",
    )
    compact = {
        "status": public_gate.get("status", gate.get("status")),
        "verified_before_truth": public_gate.get(
            "verified_before_truth", gate.get("verified_before_truth")
        ),
        "truth_opened_after_gate": gate.get(
            "truth_opened_after_gate", public_gate.get("truth_opened_after_gate")
        ),
        "prediction_artifact_count": public_gate.get(
            "prediction_artifact_count",
            len(public_gate.get("prediction_artifacts", []))
            if isinstance(public_gate.get("prediction_artifacts"), list)
            else None,
        ),
        "timing_receipt_count": public_gate.get(
            "timing_receipt_count",
            len(public_gate.get("timing_receipts", []))
            if isinstance(public_gate.get("timing_receipts"), list)
            else None,
        ),
    }
    if compact["verified_before_truth"] is not True or compact["truth_opened_after_gate"] is not True:
        raise SummaryError("scorer truth gate is not complete")
    if compact["prediction_artifact_count"] != len(EXPECTED_CELL_IDS) * len(METHOD_IDS):
        raise SummaryError("scorer prediction artifact count is not 32")
    if compact["timing_receipt_count"] != len(EXPECTED_CELL_IDS) * len(METHOD_IDS):
        raise SummaryError("scorer timing receipt count is not 32")
    for key in ("panel", "registration", "selection_plan"):
        value = public_gate.get(key, gate.get(key))
        if isinstance(value, Mapping):
            compact[key] = dict(value)
    observations = public_gate.get("observations", gate.get("observations"))
    if isinstance(observations, Mapping):
        compact["observation_count"] = len(observations)
    artifacts = public_gate.get("prediction_artifacts", gate.get("prediction_artifacts"))
    if isinstance(artifacts, list):
        compact["prediction_artifacts"] = [dict(value) for value in artifacts if isinstance(value, Mapping)]
    return compact


def _method_evidence_map(run_evidence: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = run_evidence.get("methods")
    if isinstance(raw, Mapping):
        result = {
            str(method_id): value
            for method_id, value in raw.items()
            if isinstance(value, Mapping)
        }
    elif isinstance(raw, list):
        result = {
            str(value.get("method_id")): value
            for value in raw
            if isinstance(value, Mapping) and isinstance(value.get("method_id"), str)
        }
    else:
        raise SummaryError("run evidence has no methods mapping/list")
    if set(result) != set(METHOD_IDS):
        raise SummaryError("run evidence method set changed")
    return result


def _runtime_summary(
    run_evidence: Mapping[str, Any],
    receipt_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    methods = _method_evidence_map(run_evidence)
    per_method: dict[str, Any] = {}
    aggregate_load = 0.0
    all_loads = True
    for method_id in METHOD_IDS:
        evidence = methods[method_id]
        runtime = evidence.get("runtime_load")
        if not isinstance(runtime, Mapping):
            runtime = {}
        load_value = runtime.get("runtime_load_seconds", evidence.get("runtime_load_seconds"))
        load = _optional_float(load_value, description=f"{method_id} runtime load")
        if load is None:
            all_loads = False
        else:
            aggregate_load += load
        state = {
            key: evidence.get(key)
            for key in ("state_path", "state_sha256")
            if evidence.get(key) is not None
        }
        cells: dict[str, Any] = {}
        simulation_totals: dict[str, int] = {}
        for cell_id in EXPECTED_CELL_IDS:
            receipt = receipt_rows[(cell_id, method_id)]
            timing = receipt.get("timing")
            if not isinstance(timing, Mapping):
                timing = receipt
            records = _integer(
                timing.get("records", receipt.get("records")),
                description=f"{cell_id}/{method_id} timing records",
            )
            if records != RECORDS_PER_DOMAIN:
                raise SummaryError(f"{cell_id}/{method_id} timing record count changed")
            measured_sum = timing.get("measured_seconds_sum")
            if measured_sum is None:
                per_record = timing.get("per_record_measured_seconds")
                if isinstance(per_record, list):
                    measured_sum = sum(
                        _finite_float(value, description="per-record measured seconds")
                        for value in per_record
                    )
            measured_sum = _finite_float(
                measured_sum,
                description=f"{cell_id}/{method_id} measured seconds sum",
            )
            warmup_sum = _optional_float(
                timing.get("warmup_seconds_sum"),
                description=f"{cell_id}/{method_id} warmup seconds sum",
            )
            interval_total = _optional_float(
                timing.get("timed_interval_total_seconds"),
                description=f"{cell_id}/{method_id} timed interval total",
            )
            outer_elapsed = _optional_float(
                receipt.get("measured_elapsed_seconds"),
                description=f"{cell_id}/{method_id} outer measured elapsed",
            )
            peak = timing.get("peak_memory", receipt.get("peak_memory"))
            if not isinstance(peak, Mapping):
                peak = {}
            specific = timing.get("method_specific", receipt.get("method_specific"))
            if not isinstance(specific, Mapping):
                specific = {}
            simulations: dict[str, int] = {}
            for name in (
                "candidate_simulations",
                "public_prefix_calls",
                "calls",
                "candidate_budget",
            ):
                value = specific.get(name, timing.get(name))
                if value is None:
                    continue
                integer_value = _integer(value, description=f"{cell_id}/{method_id} {name}")
                simulations[name] = integer_value
                if name != "candidate_budget":
                    simulation_totals[name] = simulation_totals.get(name, 0) + integer_value
            cells[cell_id] = {
                "records": records,
                "warmup_seconds_sum": warmup_sum,
                "measured_seconds_sum": measured_sum,
                "measured_seconds_mean": measured_sum / records,
                "measured_seconds_mean_ms": 1000.0 * measured_sum / records,
                "timed_interval_total_seconds": interval_total,
                "outer_measured_elapsed_seconds": outer_elapsed,
                "peak_memory": dict(peak),
                "simulation_counts": simulations,
                "simulation_counts_scope": (
                    specific.get("calls_scope")
                    or timing.get("adapter_call_scope")
                    or "per cell; method-specific evidence"
                ),
                "steady_interval": timing.get("steady_interval"),
                "runtime_load_seconds_repeated_in_receipt": load,
            }
        per_method[method_id] = {
            "runtime_load_seconds_once": load,
            "runtime_load_scope": "one method load across four cells; do not sum per-cell copies",
            **state,
            "cells": cells,
            "simulation_totals_across_cells": simulation_totals,
        }
    return {
        "method_count": len(per_method),
        "cell_method_receipt_count": len(receipt_rows),
        "per_method": per_method,
        "cold_load_seconds_sum_once_per_method": aggregate_load if all_loads else None,
        "cold_load_scope": "runtime_load_seconds counted once per method; per-cell copies are repeated attestations",
        "timing_boundary": "CPU activation H -> device preprocessing -> method execution -> predicted IDs CPU",
        "speed_claim_limit": "Report measured mean from measured_seconds_sum/records; do not use outer measured_elapsed_seconds as the steady per-record mean or claim near-identical speed differences.",
    }


def _load_receipts(predictions_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if predictions_root.is_symlink() or not predictions_root.is_dir():
        raise SummaryError(f"prediction root is unavailable: {predictions_root}")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(predictions_root.rglob("*.run.json")):
        if path.is_symlink() or not path.is_file():
            continue
        value = _read_json(path, description="prediction run receipt")
        cell_id = value.get("cell_id")
        method_id = value.get("method_id")
        if not isinstance(cell_id, str) or not isinstance(method_id, str):
            raise SummaryError(f"run receipt lacks string cell/method identity: {path}")
        key = (cell_id, method_id)
        expected = {
            (cell, method)
            for cell in EXPECTED_CELL_IDS
            for method in METHOD_IDS
        }
        if key not in expected:
            raise SummaryError(f"run receipt has unknown cell/method: {path}")
        if key in rows:
            raise SummaryError(f"duplicate run receipt: {key}")
        value["receipt_path"] = str(path)
        rows[key] = value
    expected = {
        (cell_id, method_id)
        for cell_id in EXPECTED_CELL_IDS
        for method_id in METHOD_IDS
    }
    if set(rows) != expected:
        raise SummaryError(
            f"run receipt matrix changed: missing={sorted(expected - set(rows))!r} extra={sorted(set(rows) - expected)!r}"
        )
    return rows


def _margin_check(effect: Mapping[str, Any], *, margin_pp: float, upper_key: str) -> dict[str, Any]:
    point = _finite_float(effect.get("delta_pp"), description="margin point estimate")
    ci = effect.get("ci95_pp")
    upper = _optional_float(effect.get(upper_key), description="margin upper bound")
    result: dict[str, Any] = {
        "margin_pp": float(margin_pp),
        "point_delta_pp": point,
        "ci95_pp": ci,
        "point_reaches_margin": point >= margin_pp,
        "upper_bound_pp": upper,
        "upper_bound_below_margin": None if upper is None else upper <= margin_pp,
        "decision_status": "informational_margin_check; root decision rule applies",
    }
    if isinstance(ci, list) and len(ci) == 2:
        result["ci95_lower_above_margin"] = float(ci[0]) >= margin_pp
    else:
        result["ci95_lower_above_margin"] = None
    return result


def _decision_support(
    comparisons: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    for cell_id in EXPECTED_CELL_IDS:
        causal_label = f"{cell_id}__enriched__causal_vs_diagonal"
        best_label = f"{cell_id}__enriched__causal_vs_best_positionwise"
        causal = comparisons.get(causal_label)
        best = comparisons.get(best_label)
        entry: dict[str, Any] = {
            "extra_context_comparisons": {
                "causal_vs_diagonal": causal,
                "causal_vs_best_positionwise": best,
            },
            "extra_context_margin_checks": None,
            "enrichment_margin_checks": {},
        }
        if causal is not None:
            entry["extra_context_margin_checks"] = {
                "token_accuracy": _margin_check(
                    causal["token"],
                    margin_pp=MARGIN["extra_context_token_accuracy_pp"],
                    upper_key="upper_bound_pp",
                ),
                "exact_record_rate": _margin_check(
                    causal["exact"],
                    margin_pp=MARGIN["extra_context_exact_record_rate_pp"],
                    upper_key="net_upper_bound_pp",
                ),
            }
        for state in ENRICHMENT_STATES:
            label = f"{cell_id}__coverage__{state}__enriched_vs_original"
            comparison = comparisons.get(label)
            entry["enrichment_margin_checks"][state] = None if comparison is None else {
                "comparison": comparison,
                "token_accuracy": _margin_check(
                    comparison["token"],
                    margin_pp=MARGIN["enrichment_token_accuracy_pp"],
                    upper_key="upper_bound_pp",
                ),
                "exact_record_rate": _margin_check(
                    comparison["exact"],
                    margin_pp=MARGIN["enrichment_exact_record_rate_pp"],
                    upper_key="net_upper_bound_pp",
                ),
            }
        by_cell[cell_id] = entry
    selected_ids: Mapping[str, Any] = {}
    if isinstance(selection, Mapping):
        candidate = selection.get("selected_method_ids")
        if isinstance(candidate, Mapping):
            selected_ids = candidate
    duplicate = (
        selected_ids.get("original") == "original__affine_trained_diagonal_attention128"
        and selected_ids.get("enriched") == "enriched__affine_trained_diagonal_attention128"
    )
    return {
        "by_cell": by_cell,
        "frozen_margins_pp": dict(MARGIN),
        "positionwise_selection_is_affine_vs_diagonal_only": True,
        "duplicate_primary_contrasts_when_both_diagonal": duplicate,
        "decision_status": "informational checks only; apply the frozen decision plan after reviewing every domain/target",
    }


def _report_tables(summary: Mapping[str, Any]) -> str:
    lines = [
        "# TRR-0005 final result tables",
        "",
        "These tables are generated from the truth-gated scorer JSON and compact run receipts. The study is exploratory: all target conditions share paired sources within each domain, domains are not pooled, and exact zero discordance is not equivalence.",
        "",
        "## Fresh cell outcomes",
        "",
        "| Cell | Method | Correct/scored tokens | Token accuracy | Exact records/records | Exact rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    outcomes = summary["fresh_outcomes"]
    for cell_id in EXPECTED_CELL_IDS:
        methods = outcomes["cells"][cell_id]["methods"]
        for method_id in METHOD_IDS:
            metric = methods[method_id]
            lines.append(
                f"| {cell_id} | `{method_id}` | {metric['correct_tokens']}/{metric['scored_tokens']} | {_fmt_pct(metric['token_accuracy'])} | {metric['exact_records']}/{metric['records']} | {_fmt_pct(metric['exact_record_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired primary comparisons and margin checks",
            "",
            "Positive deltas follow the scorer's `method_id` minus `baseline_method_id` orientation. The two enriched causal labels duplicate when the frozen best-positionwise arm is diagonal.",
            "",
            "| Cell | Label | Token delta (95% CI) | Token upper pp | Exact delta (95% CI) | Exact net upper pp |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    comparisons = summary["comparisons"]
    for cell_id in EXPECTED_CELL_IDS:
        for label in PRIMARY_LABELS:
            key = f"{cell_id}__{label}"
            comparison = comparisons.get(key)
            if comparison is None:
                continue
            token = comparison["token"]
            exact = comparison["exact"]
            lines.append(
                f"| {cell_id} | `{label}` | {_fmt_num(token['delta_pp'])} ({_fmt_ci(token['ci95_pp'])}) | {_fmt_num(token['upper_bound_pp'])} | {_fmt_num(exact['delta_pp'])} ({_fmt_ci(exact['ci95_pp'])}) | {_fmt_num(exact['net_upper_bound_pp'])} |"
            )
    lines.extend(
        [
            "",
            "## Runtime scope",
            "",
            "Measured steady means are `measured_seconds_sum / records`; outer `measured_elapsed_seconds` includes cell-level overhead and is not used as the per-record mean. Process load/init `runtime_load_seconds` is counted once per method in the existing cache environment; it is not a flushed cold-disk measurement.",
            "",
            "| Method | Load/init seconds (once; cached) | Cell | Measured mean ms | Timed interval seconds | Peak memory | Simulation counts |",
            "|---|---:|---|---:|---:|---|---|",
        ]
    )
    runtime = summary["runtime"]
    for method_id in METHOD_IDS:
        method = runtime["per_method"][method_id]
        for cell_id in EXPECTED_CELL_IDS:
            cell = method["cells"][cell_id]
            peak = cell["peak_memory"]
            peak_text = ", ".join(
                f"{name}={value}"
                for name, value in peak.items()
                if value is not None
            ) or "unavailable"
            sim_text = ", ".join(
                f"{name}={value}"
                for name, value in cell["simulation_counts"].items()
            ) or "none reported"
            lines.append(
                f"| `{method_id}` | {_fmt_num(method['runtime_load_seconds_once'])} | {cell_id} | {_fmt_num(cell['measured_seconds_mean_ms'])} | {_fmt_num(cell['timed_interval_total_seconds'])} | {peak_text} | {sim_text} |"
            )
    lines.extend(
        [
            "",
            "## Joint frequency × position × domain diagnostics",
            "",
            "Each row remains attached to its domain, target condition, method, and fitting-frequency reference. Empty bins are retained in JSON even when omitted from visual summaries.",
            "",
            "| Domain | Cell | Method | Frequency reference | Frequency bin | Position bin | Examples | Correct tokens | Token accuracy |",
            "|---|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    frequency = summary["frequency_position_by_domain"]
    for domain in STYLE_ORDER:
        for cell_id in EXPECTED_CELL_IDS:
            if not cell_id.startswith(f"{domain}__"):
                continue
            condition = cell_id.split("__", 1)[1]
            for method_id in METHOD_IDS:
                per_method = frequency[domain][condition][method_id]
                for reference in ("original", "enriched"):
                    rows = per_method[reference]["joint_frequency_position_rows"]
                    for row_key in sorted(rows):
                        row = rows[row_key]
                        lines.append(
                            f"| {domain} | {cell_id} | `{method_id}` | {reference} | {row.get('frequency_bin')} | {row.get('position_bin')} | {row.get('examples')} | {row.get('correct_tokens')} | {_fmt_pct(row.get('token_accuracy'))} |"
                        )
    lines.append("")
    return "\n".join(lines)


def _fmt_num(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.6g}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.4f}%"


def _fmt_ci(value: Any) -> str:
    # Summary CI fields are already percentage-point values.
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return "—"
    return f"{float(value[0]):.4f}, {float(value[1]):.4f}"


def build_summary(
    result: Mapping[str, Any],
    run_evidence: Mapping[str, Any],
    receipt_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if result.get("task_id") != TASK_ID or result.get("status") != RESULT_STATUS:
        raise SummaryError("result is not a completed TRR-0005 truth-gated score")
    gate = _compact_gate(result)
    rows = _cell_rows(result)
    bootstrap = _required_mapping(result.get("bootstrap"), description="bootstrap")
    draws = _integer(bootstrap.get("draws"), description="bootstrap draws")
    seed = _integer(bootstrap.get("seed"), description="bootstrap seed")
    comparisons_raw = _required_mapping(
        result.get("method_comparisons"), description="method comparisons"
    )
    comparisons: dict[str, dict[str, Any]] = {}
    for raw_key, raw_comparison in comparisons_raw.items():
        if not isinstance(raw_key, str) or not isinstance(raw_comparison, Mapping):
            raise SummaryError("method comparison is malformed")
        comparisons[raw_key] = _comparison_summary(
            raw_key,
            raw_comparison,
            draws=draws,
            seed=seed,
        )
    selection = result.get("public_validation_selection")
    if not isinstance(selection, Mapping):
        raise SummaryError("completed result has no frozen public selection")
    normalized_selection = dict(selection)
    cells: dict[str, Any] = {}
    frequency_by_domain: dict[str, Any] = {domain: {} for domain in STYLE_ORDER}
    for cell_id in EXPECTED_CELL_IDS:
        condition = cell_id.split("__", 1)[1]
        method_values: dict[str, Any] = {}
        frequency_by_domain[cell_id.split("__", 1)[0]][condition] = {}
        for method_id in METHOD_IDS:
            row = rows[(cell_id, method_id)]
            metrics = _compact_metrics(row)
            method_values[method_id] = metrics
            frequency_by_domain[cell_id.split("__", 1)[0]][condition][method_id] = _compact_frequency(row)
        cells[cell_id] = {
            "domain": cell_id.split("__", 1)[0],
            "condition": condition,
            "records": RECORDS_PER_DOMAIN,
            "methods": method_values,
        }
    run_status = run_evidence.get("status")
    if run_status != "PUBLIC_PREDICTION_MATRIX_COMPLETE_NO_TRUTH":
        raise SummaryError("run evidence is not the completed no-truth prediction matrix")
    runtime = _runtime_summary(run_evidence, receipt_rows)
    return {
        "schema": SUMMARY_SCHEMA,
        "task_id": TASK_ID,
        "status": "FINAL_SUMMARY_DERIVED_FROM_TRUTH_GATED_SCORE",
        "source_result": {
            "status": result.get("status"),
            "bootstrap": dict(bootstrap),
        },
        "truth_gate": gate,
        "scope": {
            "claim_scope": result.get("claim_scope"),
            "exploratory": True,
            "domains": list(STYLE_ORDER),
            "conditions": list(CONDITION_ORDER),
            "records_per_cell": RECORDS_PER_DOMAIN,
            "cell_count": len(EXPECTED_CELL_IDS),
            "method_count": len(METHOD_IDS),
            "method_cell_count": len(EXPECTED_CELL_IDS) * len(METHOD_IDS),
            "targets_are_paired_within_domain": True,
            "pooled_domain_headline": False,
        },
        "public_validation_selection": normalized_selection,
        "fresh_outcomes": {"cells": cells},
        "comparisons": comparisons,
        "decision_support": _decision_support(comparisons, normalized_selection),
        "frequency_position_by_domain": frequency_by_domain,
        "runtime": runtime,
        "uncertainty": {
            "token_and_exact_descriptive_intervals": "paired source-record bootstrap 95% percentile intervals",
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "exact_primary_bound": "one-sided finite-sample U(p_gain)-L(p_loss) Clopper-Pearson bound",
            "exact_zero_discordance_warning": "zero discordance retains a positive finite-sample upper bound and is not equivalence",
            "target_pairing_warning": "the same source IDs are reused across target conditions within each domain",
        },
        "warnings": [
            "No pooled domain or target headline is valid for this exploratory matrix.",
            "Best-positionwise is affine-versus-diagonal only; when both frozen selections are diagonal, the two causal primary labels are duplicate method pairs rather than independent evidence.",
            "Trained diagonal retains contextual H_i and adds a positionwise nonlinear layer-normalized value correction; it is not context-free or merely a redundant affine map. Causal adds earlier H through the same path, while qknorm repairs routing.",
            "Frequency maps are alternate diagnostics on the same predictions, not additional samples.",
            "Steady runtime means use measured_seconds_sum/records; runtime_load_seconds is counted once per method.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-evidence", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report-tables", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _read_json(args.result, description="score result")
    run_evidence = _read_json(args.run_evidence, description="prediction run evidence")
    receipt_rows = _load_receipts(args.predictions_root.expanduser().resolve())
    summary = build_summary(result, run_evidence, receipt_rows)
    report = _report_tables(summary)
    summary_path = args.summary.expanduser().resolve()
    report_path = args.report_tables.expanduser().resolve()
    if summary_path.exists() or summary_path.is_symlink():
        raise SummaryError(f"refusing to overwrite summary artifact: {summary_path}")
    if report_path.exists() or report_path.is_symlink():
        raise SummaryError(f"refusing to overwrite report tables: {report_path}")
    _write_create_only(summary_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"summary": str(args.summary.resolve()), "report_tables": str(report_path), "status": summary["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SummaryError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0005 summary error: {exc}") from exc
