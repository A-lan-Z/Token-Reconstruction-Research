"""Resolve TRR-0010 deployment-cost gates from a public run manifest.

The prediction runner records one truth-free timing receipt for every method
and cell.  This adapter consumes those receipts only after the public output
gate has validated them.  It keeps post-warmup inference separate from method
preparation, uses the per-cell CUDA peak that the runner resets before each
cell, and never treats the runner's cumulative host RSS as an isolated
deployment measurement.  Missing deployment measurements remain ``UNKNOWN``.

The thresholds are read from the canonical v4 proposal and checked against
the reviewed values.  No quality score, source row, model state, or truth
payload is read here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts import trr0010_eval_gate as gate


COST_SCHEMA = "token-reconstruction.trr0010-cost-evidence.v1"
REPORT_TABLE_SCHEMA = "token-reconstruction.trr0010-report-comparison-table.v1"
COST_STATUS = "COST_EVIDENCE_COMPLETE_PUBLIC_TIMING"
PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

# These are the reviewed v4 values.  Reading them from the contract prevents
# post-hoc constants in this adapter, while checking them here prevents a
# stale v1/v2/v3 planning file from silently changing the gate.
EXPECTED_V4_THRESHOLDS = {
    "matched_fixed_runtime_ratio_max": 1.5,
    "matched_fixed_gpu_peak_ratio_max": 1.5,
    "a1_a2_runtime_ratio_max": 0.25,
}

PRIMARY_CANDIDATE = gate.EXPANDED_DIRECTIONAL_METHOD_ID
PRIMARY_FIXED = gate.EXPANDED_FIXED_METHOD_ID
ROBUSTNESS_FIXED = gate.CURRENT_FIXED_METHOD_ID
CANDIDATE_METHODS = (gate.CURRENT_DIRECTIONAL_METHOD_ID, gate.EXPANDED_DIRECTIONAL_METHOD_ID)


class CostEvidenceError(ValueError):
    """Raised when a public cost receipt is malformed or unbound."""


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CostEvidenceError(f"{description} must be an object")
    return value


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CostEvidenceError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CostEvidenceError(f"{description} must be a JSON object")
    return value


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        return gate.file_record(path, root=root)
    except (OSError, gate.GateError) as exc:
        raise CostEvidenceError(f"{description} cannot be recorded: {path}") from exc


def _number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise CostEvidenceError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CostEvidenceError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        raise CostEvidenceError(f"{label} must be finite and nonnegative")
    return result


def _optional_number(value: Any, *, label: str) -> float | None:
    """Return a measured nonnegative number, or ``None`` for absent data."""

    if value is None:
        return None
    return _number(value, label=label)


def _criteria(contract_path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = Path(contract_path).expanduser().resolve()
    payload = _load_json(contract_path, description="v4 cost contract")
    if payload.get("schema") != "token-reconstruction.trr0010-shared-contract-proposal.v4":
        raise CostEvidenceError("cost criteria are not the canonical TRR-0010 v4 proposal")
    if payload.get("task_id") != gate.TASK_ID:
        raise CostEvidenceError("cost contract task identity changed")
    safeguards = _mapping(payload.get("safeguards_and_cost"), "v4 safeguards_and_cost")
    values = {
        "matched_fixed_runtime_ratio_max": _number(
            safeguards.get("warmed_inference_runtime_ratio_max"),
            label="v4 warmed runtime threshold",
            positive=True,
        ),
        "matched_fixed_gpu_peak_ratio_max": _number(
            safeguards.get("isolated_method_peak_gpu_memory_ratio_max"),
            label="v4 isolated GPU threshold",
            positive=True,
        ),
        "a1_a2_runtime_ratio_max": _number(
            safeguards.get("candidate_to_a1_a2_warmed_runtime_ratio_max"),
            label="v4 A1+A2 runtime threshold",
            positive=True,
        ),
    }
    for name, expected in EXPECTED_V4_THRESHOLDS.items():
        if not math.isclose(values[name], expected, rel_tol=0.0, abs_tol=1e-12):
            raise CostEvidenceError(f"v4 cost threshold changed: {name}={values[name]!r}")
    gate_text = safeguards.get("candidate_to_a1_a2_cost_gate")
    if not isinstance(gate_text, str) or "candidate_warmed_runtime / A1+A2_warmed_runtime" not in gate_text:
        raise CostEvidenceError("v4 A1+A2 cost rule does not bind candidate/A1+A2 warmed runtime")
    values["criteria_status"] = "V4_THRESHOLDS_VERIFIED"
    return values, _record(contract_path, root=root, description="v4 cost contract")


def _validated_timing_matrix(
    *, run_manifest_path: Path, root: Path, validate_public_gate: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_manifest_path = Path(run_manifest_path).expanduser().resolve()
    run = _load_json(run_manifest_path, description="public run manifest")
    if run.get("schema") != gate.RUN_SCHEMA or run.get("task_id") != gate.TASK_ID or run.get("status") != gate.RUN_STATUS:
        raise CostEvidenceError("run manifest is not a completed TRR-0010 public prediction run")
    if validate_public_gate:
        registration = _mapping(run.get("registration"), "run registration binding")
        registration_path = Path(str(registration.get("path"))).expanduser().resolve()
        try:
            freeze = gate.validate_public_outputs(
                registration_path=registration_path,
                run_manifest_path=run_manifest_path,
                repository_root=root,
                require_current_head=False,
            )
        except gate.GateError as exc:
            raise CostEvidenceError(f"public output gate failed before cost extraction: {exc}") from exc
        timing_map = _mapping(freeze.get("timings"), "validated timing matrix")
        run_record = _record(run_manifest_path, root=root, description="public run manifest")
        return run, dict(timing_map), run_record

    raw_timings = _mapping(run.get("timings"), "run manifest timings")
    expected = {f"{method}::{cell}" for method in gate.METHOD_ORDER for cell in gate.CELL_ORDER}
    if set(raw_timings) != expected:
        raise CostEvidenceError("run manifest timing matrix is incomplete")
    return run, dict(raw_timings), _record(run_manifest_path, root=root, description="public run manifest")


def _measurement(
    *, method_id: str, cell_id: str, timing_binding: Mapping[str, Any]
) -> dict[str, Any]:
    binding = _mapping(timing_binding, f"timing binding {method_id}/{cell_id}")
    payload = binding.get("payload")
    if not isinstance(payload, Mapping):
        path = binding.get("path")
        if not isinstance(path, str):
            raise CostEvidenceError(f"timing path is absent: {method_id}/{cell_id}")
        payload = _load_json(Path(path).expanduser().resolve(), description=f"timing {method_id}/{cell_id}")
    if payload.get("method_id") != method_id or payload.get("cell_id") != cell_id:
        raise CostEvidenceError(f"timing method/cell binding changed: {method_id}/{cell_id}")
    for flag in (
        "truth_opened",
        "source_text_loaded",
        "source_text_written",
        "target_labels_loaded",
        "token_ids_written",
        "candidate_arrays_persisted",
    ):
        if payload.get(flag) is not False:
            raise CostEvidenceError(f"timing receipt is not truth-free: {method_id}/{cell_id}/{flag}")
    records_raw = payload.get("records", gate.RECORDS_PER_CELL)
    if isinstance(records_raw, bool):
        raise CostEvidenceError(f"timing record count malformed: {method_id}/{cell_id}")
    try:
        records = int(records_raw)
    except (TypeError, ValueError) as exc:
        raise CostEvidenceError(f"timing record count malformed: {method_id}/{cell_id}") from exc
    if records <= 0:
        raise CostEvidenceError(f"timing record count must be positive: {method_id}/{cell_id}")
    measured = _optional_number(payload.get("measured_seconds_sum"), label=f"measured runtime {method_id}/{cell_id}")
    warmup = _optional_number(payload.get("warmup_seconds_sum"), label=f"warmup runtime {method_id}/{cell_id}")
    preparation = _optional_number(payload.get("model_preparation_seconds"), label=f"preparation runtime {method_id}/{cell_id}")
    peak = payload.get("peak_memory")
    peak_map = dict(peak) if isinstance(peak, Mapping) else {}
    gpu_reserved = _optional_number(
        peak_map.get("cuda_peak_reserved_bytes"),
        label=f"isolated CUDA reserved peak {method_id}/{cell_id}",
    )
    gpu_allocated = _optional_number(
        peak_map.get("cuda_peak_allocated_bytes"),
        label=f"isolated CUDA allocated peak {method_id}/{cell_id}",
    )
    host_rss = _optional_number(
        peak_map.get("process_max_rss_bytes"),
        label=f"host RSS {method_id}/{cell_id}",
    )
    if measured is None:
        measurement_status = UNKNOWN
        measurement_reason = "measured_seconds_sum_unavailable"
    elif gpu_reserved is None:
        measurement_status = "PARTIAL"
        measurement_reason = "isolated_cuda_peak_unavailable"
    else:
        measurement_status = "MEASURED"
        measurement_reason = None
    row: dict[str, Any] = {
        "method_id": method_id,
        "cell_id": cell_id,
        "records": records,
        "warmed_runtime_seconds": measured,
        "warmed_runtime_seconds_per_record": None if measured is None else measured / records,
        "warmup_seconds_sum": warmup,
        "model_preparation_seconds": preparation,
        "preparation_charge_count": 1,
        "preparation_shared_across_cells": True,
        "isolated_gpu_peak_reserved_bytes": gpu_reserved,
        "isolated_gpu_peak_allocated_bytes": gpu_allocated,
        "host_process_max_rss_bytes": host_rss,
        "status": measurement_status,
        "reason": measurement_reason,
        "runtime_scope": "measured_seconds_sum after one warmup per record; preparation excluded",
        "memory_scope": "cuda_peak_* reset immediately before this cell while this method remains resident",
        "host_memory_scope": "process_max_rss_bytes is whole-runner ru_maxrss high-water, not an isolated deployment gate",
    }
    return row


def _ratio(
    *,
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
    field: str,
    threshold: float,
    metric: str,
    numerator_method_id: str,
    denominator_method_id: str,
) -> dict[str, Any]:
    left = numerator.get(field)
    right = denominator.get(field)
    result: dict[str, Any] = {
        "metric": metric,
        "status": UNKNOWN,
        "numerator_method_id": numerator_method_id,
        "denominator_method_id": denominator_method_id,
        "numerator": left,
        "denominator": right,
        "threshold": threshold,
        "ratio": None,
    }
    if left is None or right is None:
        result["reason"] = "required_measurement_unavailable"
        return result
    if float(right) <= 0.0 or float(left) < 0.0:
        result["reason"] = "nonpositive_ratio_denominator_or_invalid_numerator"
        return result
    ratio = float(left) / float(right)
    if not math.isfinite(ratio):
        result["reason"] = "nonfinite_ratio"
        return result
    result["ratio"] = ratio
    result["status"] = PASS if ratio <= threshold else FAIL
    return result


def _combined(*checks: Mapping[str, Any]) -> str:
    statuses = [str(check.get("status", UNKNOWN)) for check in checks]
    if any(status == FAIL for status in statuses):
        return FAIL
    if any(status != PASS for status in statuses):
        return UNKNOWN
    return PASS


def _comparison(
    *,
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
    numerator_method_id: str,
    denominator_method_id: str,
    runtime_threshold: float,
    memory_threshold: float | None,
) -> dict[str, Any]:
    runtime = _ratio(
        numerator=numerator,
        denominator=denominator,
        field="warmed_runtime_seconds",
        threshold=runtime_threshold,
        metric="warmed_runtime",
        numerator_method_id=numerator_method_id,
        denominator_method_id=denominator_method_id,
    )
    checks: list[Mapping[str, Any]] = [runtime]
    memory: dict[str, Any] | None = None
    if memory_threshold is not None:
        memory = _ratio(
            numerator=numerator,
            denominator=denominator,
            field="isolated_gpu_peak_reserved_bytes",
            threshold=memory_threshold,
            metric="isolated_gpu_peak_reserved",
            numerator_method_id=numerator_method_id,
            denominator_method_id=denominator_method_id,
        )
        checks.append(memory)
    result: dict[str, Any] = {
        "status": _combined(*checks),
        "numerator_method_id": numerator_method_id,
        "denominator_method_id": denominator_method_id,
        "runtime": runtime,
    }
    if memory is not None:
        result["memory"] = memory
    return result


def _table_rows(measurements: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in gate.METHOD_ORDER:
        for cell_id in gate.CELL_ORDER:
            row = measurements[method_id][cell_id]
            domain, target = cell_id.split("__", 1)
            rows.append(
                {
                    "method_id": method_id,
                    "method_role": gate.METHOD_ROLES[method_id],
                    "cell_id": cell_id,
                    "domain": domain,
                    "target": target,
                    "records": row["records"],
                    "warmed_runtime_seconds": row["warmed_runtime_seconds"],
                    "warmed_runtime_seconds_per_record": row["warmed_runtime_seconds_per_record"],
                    "isolated_gpu_peak_reserved_bytes": row["isolated_gpu_peak_reserved_bytes"],
                    "model_preparation_seconds": row["model_preparation_seconds"],
                    "quality_metrics": "supplied by the score artifact; absent from this cost table",
                }
            )
    return rows


def build_cost_evidence(
    *,
    run_manifest_path: Path,
    repository_root: Path,
    contract_path: Path | None = None,
    validate_public_gate: bool = True,
) -> dict[str, Any]:
    """Return cost gates and a six-method/four-cell report table.

    ``validate_public_gate`` is intended only for synthetic fixture debugging;
    production extraction leaves it enabled.  Even when disabled, timing
    files are read from the run manifest and all method/cell bindings remain
    explicit.
    """

    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise CostEvidenceError(f"repository root unavailable: {root}")
    run, timing_map, run_record = _validated_timing_matrix(
        run_manifest_path=run_manifest_path,
        root=root,
        validate_public_gate=validate_public_gate,
    )
    if contract_path is None:
        contract_path = root / "experiments" / gate.TASK_ID / "planning" / "shared_contract.proposal.json"
    threshold, contract_record = _criteria(Path(contract_path), root=root)
    measurements: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in gate.METHOD_ORDER}
    for method_id in gate.METHOD_ORDER:
        for cell_id in gate.CELL_ORDER:
            key = f"{method_id}::{cell_id}"
            if key not in timing_map:
                raise CostEvidenceError(f"timing matrix is missing {key}")
            measurements[method_id][cell_id] = _measurement(
                method_id=method_id,
                cell_id=cell_id,
                timing_binding=timing_map[key],
            )

    primary_by_cell: dict[str, Any] = {}
    robustness_by_cell: dict[str, Any] = {}
    current_directional_by_cell: dict[str, Any] = {}
    data_fixed_by_cell: dict[str, Any] = {}
    data_a1_a2_by_cell: dict[str, Any] = {}
    a1_a2_by_cell: dict[str, Any] = {}
    for cell_id in gate.CELL_ORDER:
        primary_by_cell[cell_id] = _comparison(
            numerator=measurements[PRIMARY_CANDIDATE][cell_id],
            denominator=measurements[PRIMARY_FIXED][cell_id],
            numerator_method_id=PRIMARY_CANDIDATE,
            denominator_method_id=PRIMARY_FIXED,
            runtime_threshold=threshold["matched_fixed_runtime_ratio_max"],
            memory_threshold=threshold["matched_fixed_gpu_peak_ratio_max"],
        )
        robustness_by_cell[cell_id] = _comparison(
            numerator=measurements[PRIMARY_CANDIDATE][cell_id],
            denominator=measurements[ROBUSTNESS_FIXED][cell_id],
            numerator_method_id=PRIMARY_CANDIDATE,
            denominator_method_id=ROBUSTNESS_FIXED,
            runtime_threshold=threshold["matched_fixed_runtime_ratio_max"],
            memory_threshold=threshold["matched_fixed_gpu_peak_ratio_max"],
        )
        current_directional_by_cell[cell_id] = _comparison(
            numerator=measurements[gate.CURRENT_DIRECTIONAL_METHOD_ID][cell_id],
            denominator=measurements[gate.CURRENT_FIXED_METHOD_ID][cell_id],
            numerator_method_id=gate.CURRENT_DIRECTIONAL_METHOD_ID,
            denominator_method_id=gate.CURRENT_FIXED_METHOD_ID,
            runtime_threshold=threshold["matched_fixed_runtime_ratio_max"],
            memory_threshold=threshold["matched_fixed_gpu_peak_ratio_max"],
        )
        data_fixed_by_cell[cell_id] = _comparison(
            numerator=measurements[gate.EXPANDED_FIXED_METHOD_ID][cell_id],
            denominator=measurements[gate.CURRENT_FIXED_METHOD_ID][cell_id],
            numerator_method_id=gate.EXPANDED_FIXED_METHOD_ID,
            denominator_method_id=gate.CURRENT_FIXED_METHOD_ID,
            runtime_threshold=threshold["matched_fixed_runtime_ratio_max"],
            memory_threshold=threshold["matched_fixed_gpu_peak_ratio_max"],
        )
        data_a1_a2_by_cell[cell_id] = _comparison(
            numerator=measurements[gate.EXPANDED_FIXED_METHOD_ID][cell_id],
            denominator=measurements[gate.A1_A2_METHOD_ID][cell_id],
            numerator_method_id=gate.EXPANDED_FIXED_METHOD_ID,
            denominator_method_id=gate.A1_A2_METHOD_ID,
            runtime_threshold=threshold["a1_a2_runtime_ratio_max"],
            memory_threshold=None,
        )
        a1_a2_by_cell[cell_id] = _comparison(
            numerator=measurements[PRIMARY_CANDIDATE][cell_id],
            denominator=measurements[gate.A1_A2_METHOD_ID][cell_id],
            numerator_method_id=PRIMARY_CANDIDATE,
            denominator_method_id=gate.A1_A2_METHOD_ID,
            runtime_threshold=threshold["a1_a2_runtime_ratio_max"],
            memory_threshold=None,
        )

    primary_status = _status_for_rows(primary_by_cell)
    a1_a2_status = _status_for_rows(a1_a2_by_cell)
    directional_route_by_cell = {
        cell_id: {
            "status": _combined(primary_by_cell[cell_id], robustness_by_cell[cell_id], a1_a2_by_cell[cell_id]),
            "candidate_vs_fixed": primary_by_cell[cell_id],
            "candidate_vs_current_fixed": robustness_by_cell[cell_id],
            "candidate_vs_a1_a2": a1_a2_by_cell[cell_id],
        }
        for cell_id in gate.CELL_ORDER
    }
    data_route_by_cell = {
        cell_id: {
            "status": _combined(data_fixed_by_cell[cell_id], data_a1_a2_by_cell[cell_id]),
            "expanded_fixed_vs_current_fixed": data_fixed_by_cell[cell_id],
            "expanded_fixed_vs_a1_a2": data_a1_a2_by_cell[cell_id],
        }
        for cell_id in gate.CELL_ORDER
    }
    cost_gate_by_cell = {
        cell_id: {
            "status": _combined(primary_by_cell[cell_id], a1_a2_by_cell[cell_id]),
            "candidate_vs_fixed": primary_by_cell[cell_id],
            "candidate_vs_a1_a2": a1_a2_by_cell[cell_id],
        }
        for cell_id in gate.CELL_ORDER
    }
    return {
        "schema": COST_SCHEMA,
        "task_id": gate.TASK_ID,
        "status": COST_STATUS,
        "run_manifest": run_record,
        "contract": contract_record,
        "criteria": {
            **threshold,
            "candidate_method_id": PRIMARY_CANDIDATE,
            "matched_fixed_method_id": PRIMARY_FIXED,
            "robustness_fixed_method_id": ROBUSTNESS_FIXED,
            "a1_a2_method_id": gate.A1_A2_METHOD_ID,
            "memory_metric": "cuda_peak_reserved_bytes",
            "host_rss_is_gate": False,
            "preparation_is_warm_runtime": False,
        },
        "method_order": list(gate.METHOD_ORDER),
        "cell_order": list(gate.CELL_ORDER),
        "measurements": measurements,
        "comparisons": {
            "candidate_vs_fixed": {
                "status": primary_status,
                "candidate_method_id": PRIMARY_CANDIDATE,
                "fixed_method_id": PRIMARY_FIXED,
                "by_cell": primary_by_cell,
            },
            "candidate_vs_current_fixed_robustness": {
                "status": _status_for_rows(robustness_by_cell),
                "candidate_method_id": PRIMARY_CANDIDATE,
                "fixed_method_id": ROBUSTNESS_FIXED,
                "by_cell": robustness_by_cell,
            },
            "current_directional_vs_current_fixed": {
                "status": _status_for_rows(current_directional_by_cell),
                "by_cell": current_directional_by_cell,
            },
            "candidate_vs_a1_a2": {
                "status": a1_a2_status,
                "candidate_method_id": PRIMARY_CANDIDATE,
                "a1_a2_method_id": gate.A1_A2_METHOD_ID,
                "by_cell": a1_a2_by_cell,
            },
            "data_expanded_fixed_vs_current_fixed": {
                "status": _status_for_rows(data_fixed_by_cell),
                "candidate_method_id": gate.EXPANDED_FIXED_METHOD_ID,
                "fixed_method_id": gate.CURRENT_FIXED_METHOD_ID,
                "by_cell": data_fixed_by_cell,
            },
            "data_expanded_fixed_vs_a1_a2": {
                "status": _status_for_rows(data_a1_a2_by_cell),
                "candidate_method_id": gate.EXPANDED_FIXED_METHOD_ID,
                "a1_a2_method_id": gate.A1_A2_METHOD_ID,
                "by_cell": data_a1_a2_by_cell,
            },
        },
        "route_cost_gates": {
            "directional": {
                "status": _status_for_rows(directional_route_by_cell),
                "by_cell": directional_route_by_cell,
            },
            "data": {
                "status": _status_for_rows(data_route_by_cell),
                "by_cell": data_route_by_cell,
            },
        },
        "cost_gate": {
            "status": _status_for_rows(cost_gate_by_cell),
            "required_cells": list(gate.CELL_ORDER),
            "by_cell": cost_gate_by_cell,
            "candidate_vs_fixed_status": primary_status,
            "candidate_vs_a1_a2_status": a1_a2_status,
        },
        "report_comparison_table": {
            "schema": REPORT_TABLE_SCHEMA,
            "method_order": list(gate.METHOD_ORDER),
            "cell_order": list(gate.CELL_ORDER),
            "rows": _table_rows(measurements),
            "quality_metrics_source": "score artifact; intentionally not fabricated by the cost adapter",
        },
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
    }


def _status_for_rows(rows: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = [str(row.get("status", UNKNOWN)) for row in rows.values()]
    if any(status == FAIL for status in statuses):
        return FAIL
    if any(status != PASS for status in statuses):
        return UNKNOWN
    return PASS


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise CostEvidenceError(f"refusing to overwrite cost evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise CostEvidenceError(f"could not write cost evidence: {path}") from exc
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": gate.sha256_file(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="canonical v4 contract; defaults to experiments/TRR-0010/planning/shared_contract.proposal.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-public-gate",
        action="store_true",
        help="synthetic debugging only; production extraction must keep the public gate enabled",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_cost_evidence(
            run_manifest_path=args.run_manifest,
            repository_root=args.repository_root,
            contract_path=args.contract,
            validate_public_gate=not args.skip_public_gate,
        )
        artifact = _write_create_only(args.output, result)
    except (CostEvidenceError, OSError, ValueError) as exc:
        print(f"TRR-0010 cost evidence failed closed: {exc}")
        return 2
    print(json.dumps({"status": result["status"], "cost_gate": result["cost_gate"], "artifact": artifact}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
