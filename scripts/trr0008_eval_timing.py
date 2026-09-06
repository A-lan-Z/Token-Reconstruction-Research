"""Compatibility-only timing math for TRR-0008.

The canonical executable is scripts/trr0008_timing.py. This module retains
small schedule and ratio helpers for historical unit tests but deliberately
refuses all plan/run CLI execution so two timing authorities cannot diverge.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import torch
from safetensors import safe_open

from scripts import trr0008_eval_contract as contract
from scripts import trr0008_eval_runner as runner


class TimingError(contract.ContractError):
    pass


def build_order_schedule(
    *,
    blocks: int,
    cells: Sequence[str] = contract.CELL_ORDER,
    methods: Sequence[str] = contract.TIMING_METHOD_ORDER,
    records_per_domain: int = contract.TIMING_RECORDS_PER_DOMAIN,
    records_per_block_cell: int = 32,
    seed: int = 8008,
) -> dict[str, Any]:
    """Create a balanced, predeclared block/cell order and row schedule."""

    if blocks <= 0 or records_per_block_cell <= 0 or records_per_domain <= 0:
        raise TimingError("timing dimensions must be positive")
    if tuple(cells) != contract.CELL_ORDER:
        raise TimingError("timing cells must be the four declared cells")
    if set(methods) != set(contract.TIMING_METHOD_ORDER) or len(methods) != len(contract.TIMING_METHOD_ORDER):
        raise TimingError("timing methods must include the four scientific paths and alias")
    if records_per_domain % records_per_block_cell:
        raise TimingError("timing block size must divide the fixture rows")
    base = list(methods)
    random.Random(int(seed)).shuffle(base)
    schedule: list[dict[str, Any]] = []
    for block_index in range(blocks):
        start = (block_index * records_per_block_cell) % records_per_domain
        rows = list(range(start, start + records_per_block_cell))
        for cell_index, cell_id in enumerate(cells):
            shift = (block_index * len(cells) + cell_index) % len(base)
            order = base[shift:] + base[:shift]
            schedule.append(
                {
                    "block": block_index,
                    "cell_id": cell_id,
                    "record_indices": rows,
                    "method_order": order,
                }
            )
    return {
        "schema": "token-reconstruction.trr0008-timing-plan.v1",
        "task_id": contract.TASK_ID,
        "status": "FROZEN_TIMING_ORDER_BEFORE_EXECUTION",
        "cells": list(cells),
        "methods": list(methods),
        "scientific_methods": list(contract.METHOD_ORDER),
        "timing_control": contract.TIMING_CONTROL_METHOD_ID,
        "blocks": blocks,
        "records_per_domain": records_per_domain,
        "records_per_block_cell": records_per_block_cell,
        "warmup_runs_per_record": 1,
        "measured_runs_per_record": 1,
        "order_seed": int(seed),
        "record_schedule": schedule,
        "timed_interval": "synchronized current-H row prediction including CPU BF16 staging, FP32 decoder, full-vocabulary argmax, and CPU ID transfer",
        "excluded_from_timed_interval": ["model preparation", "E load", "observation I/O", "prediction serialization", "equivalence validation"],
        "qualification": {
            "ratio": "sum candidate measured seconds over four cells / sum reference measured seconds over four cells per block",
            "limit": 1.25,
            "pass": "upper two-sided 95% Student-t interval <= 1.25",
            "fail": "lower two-sided 95% Student-t interval > 1.25",
            "otherwise": "inconclusive",
        },
    }


def validate_order_schedule(plan: Mapping[str, Any]) -> None:
    if plan.get("task_id") != contract.TASK_ID or plan.get("status") != "FROZEN_TIMING_ORDER_BEFORE_EXECUTION":
        raise TimingError("timing plan is not frozen")
    methods = tuple(plan.get("methods", ()))
    if methods != contract.TIMING_METHOD_ORDER:
        raise TimingError("timing method order set changed")
    cells = tuple(plan.get("cells", ()))
    if cells != contract.CELL_ORDER:
        raise TimingError("timing cell order changed")
    schedule = plan.get("record_schedule")
    if not isinstance(schedule, Sequence) or not schedule:
        raise TimingError("timing record schedule is absent")
    block_count = int(plan.get("blocks", 0))
    expected_count = block_count * len(contract.CELL_ORDER)
    if len(schedule) != expected_count:
        raise TimingError("timing schedule has the wrong number of cell blocks")
    expected_rows = int(plan.get("records_per_block_cell", 0))
    for entry in schedule:
        if not isinstance(entry, Mapping):
            raise TimingError("timing schedule entry is malformed")
        if tuple(entry.get("method_order", ())) != tuple(methods):
            # Balanced schedules may rotate the order, but every entry must be
            # a permutation of the declared paths.
            if set(entry.get("method_order", ())) != set(methods) or len(entry.get("method_order", ())) != len(methods):
                raise TimingError("timing method order entry is malformed")
        rows = entry.get("record_indices")
        if not isinstance(rows, Sequence) or len(rows) != expected_rows:
            raise TimingError("timing record block geometry changed")
        if any(int(row) < 0 or int(row) >= int(plan["records_per_domain"]) for row in rows):
            raise TimingError("timing record index is out of range")


def _t_critical(df: int) -> float:
    # Exact two-sided 95% values for the small planned block counts; the
    # normal limit is adequate only for larger future diagnostic plans.
    table = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139, 11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.13145, 16: 2.119905, 17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963, 21: 2.079614, 22: 2.073873, 23: 2.068658, 24: 2.063899, 25: 2.059539, 26: 2.055529, 27: 2.051831, 28: 2.048407, 29: 2.04523, 30: 2.042272}
    return table.get(int(df), 1.959964)


def summarize_block_ratios(
    block_seconds: Sequence[Mapping[str, Any]],
    *,
    candidate: str = contract.PRIMARY_METHOD_ID,
    reference: str = contract.REFERENCE_METHOD_ID,
    limit: float = 1.25,
) -> dict[str, Any]:
    """Summarize paired block ratios without selecting a favorable order."""

    ratios: list[float] = []
    for block in block_seconds:
        values = block.get("method_cell_seconds")
        if not isinstance(values, Mapping):
            raise TimingError("block timing values are absent")
        candidate_total = 0.0
        reference_total = 0.0
        for cell_id in contract.CELL_ORDER:
            row = values.get(cell_id)
            if not isinstance(row, Mapping):
                raise TimingError(f"block cell timing missing: {cell_id}")
            try:
                candidate_total += float(row[candidate])
                reference_total += float(row[reference])
            except (KeyError, TypeError, ValueError) as exc:
                raise TimingError("block timing seconds are malformed") from exc
        if candidate_total <= 0.0 or reference_total <= 0.0:
            raise TimingError("block timing seconds must be positive")
        ratios.append(candidate_total / reference_total)
    if not ratios:
        raise TimingError("no timing blocks were measured")
    mean = statistics.fmean(ratios)
    if len(ratios) > 1:
        standard_error = statistics.stdev(ratios) / math.sqrt(len(ratios))
        margin = _t_critical(len(ratios) - 1) * standard_error
    else:
        margin = float("inf")
    lower, upper = mean - margin, mean + margin
    if upper <= float(limit):
        decision = "passes"
    elif lower > float(limit):
        decision = "fails"
    else:
        decision = "inconclusive"
    return {
        "candidate": candidate,
        "reference": reference,
        "block_count": len(ratios),
        "block_ratios": ratios,
        "mean_ratio": mean,
        "two_sided_95_student_t": {"lower": lower, "upper": upper, "margin": margin, "df": max(0, len(ratios) - 1)},
        "limit": float(limit),
        "decision": decision,
    }


def _load_archived(path: Path, *, records: int) -> torch.Tensor:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"predictions"}:
                raise TimingError("archived prediction contains unexpected tensors")
            values = handle.get_tensor("predictions")
    except TimingError:
        raise
    except Exception as exc:
        raise TimingError(f"archived prediction is unreadable: {path}") from exc
    return contract.validate_prediction_tensor(values, records=records)


def _archived_path(root: Path, archived_root: Path, cell_id: str, method_id: str) -> Path:
    return runner._prediction_path(archived_root, cell_id, method_id)


def execute(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Retired: use scripts/trr0008_timing.py for the canonical receipt."""
    raise TimingError("retired timing execution path; use scripts/trr0008_timing.py")



def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--blocks", type=int, default=10)
    plan.add_argument("--records-per-domain", type=int, default=contract.TIMING_RECORDS_PER_DOMAIN)
    plan.add_argument("--records-per-block-cell", type=int, default=32)
    plan.add_argument("--seed", type=int, default=8008)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--repository-root", type=Path, default=Path("."))
    run.add_argument("--registration", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--archived-predictions-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        raise TimingError("retired timing CLI; use scripts/trr0008_timing.py")
        if args.command == "plan":
            value = build_order_schedule(
                blocks=args.blocks,
                records_per_domain=args.records_per_domain,
                records_per_block_cell=args.records_per_block_cell,
                seed=args.seed,
            )
            contract.write_create_only(args.output, value)
        else:
            value = execute(
                registration_path=args.registration,
                plan_path=args.plan,
                repository_root=args.repository_root,
                archived_predictions_root=args.archived_predictions_root,
                output=args.output,
                device_name=args.device,
            )
    except (TimingError, contract.ContractError, runner.RunnerError) as exc:
        print(f"TRR-0008 timing failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": value.get("status"), "blocks": value.get("blocks")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
