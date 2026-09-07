"""TRR-0009 balanced timing design, execution, and qualification.

The live executor consumes the owner-frozen registration and timing plan,
reuses the TRR9 current-H prediction boundary, and writes a create-only
source-free timing receipt.  It never opens source labels or truth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from scripts import trr0009_eval_contract as contract
from scripts import trr0009_eval_runner as runner

TASK_ID = contract.TASK_ID
SCHEMA = "token-reconstruction.trr0009-balanced-timing.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0009-timing-failure.v1"
TIMING_PATH_IDS = tuple(contract.METHOD_ORDER) + ("continued_fixed_readout__alias",)
ALIAS_METHOD_ID = "continued_fixed_readout__alias"
DEFAULT_RECORDS_PER_CELL = 32
DEFAULT_BLOCKS = 40
DEFAULT_WARMUP_RUNS = 1
DEFAULT_MEASURED_RUNS = 1
DEFAULT_SEED = 8008
DEFAULT_MAX_SECONDS = 600
QUALIFICATION_THRESHOLD = 1.25
ALIAS_RUNTIME_BAND = (0.95, 1.05)
CI_LEVEL = 0.95

# Keep the validated df=39 value explicit.  Other values are resolved through
# scipy's Student-t distribution; a normal approximation is never substituted.
_T_CRITICAL_95 = {39: 2.022690920036761}


class TimingError(ValueError):
    pass


def _t_critical_95(degrees_of_freedom: int) -> float:
    df = int(degrees_of_freedom)
    if df <= 0:
        raise TimingError("Student-t degrees of freedom must be positive")
    if df in _T_CRITICAL_95:
        return float(_T_CRITICAL_95[df])
    try:
        from scipy.stats import t
        value = float(t.ppf(0.975, df))
    except ImportError as exc:
        raise TimingError(f"scipy is required for Student-t critical value df={df}") from exc
    if not math.isfinite(value):
        raise TimingError(f"Student-t critical value is unavailable for df={df}")
    return value


def balanced_orders(*, method_ids: Sequence[str] = TIMING_PATH_IDS, cell_ids: Sequence[str] = contract.CELL_ORDER, blocks: int = DEFAULT_BLOCKS, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Return fixed rotations plus reversals, repeated in complete cycles."""

    methods = tuple(str(x) for x in method_ids)
    cells = tuple(str(x) for x in cell_ids)
    if methods != TIMING_PATH_IDS:
        raise TimingError("timing path order must contain the four methods plus the fixed-weight alias")
    if cells != contract.CELL_ORDER:
        raise TimingError("timing cell order changed")
    if int(blocks) <= 0 or int(blocks) % (2 * len(methods)):
        raise TimingError(f"blocks must be a positive multiple of {2 * len(methods)}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(len(methods), generator=generator).tolist()
    permuted = tuple(methods[i] for i in permutation)
    orders: list[dict[str, Any]] = []
    cycle_length = 2 * len(permuted)
    for block_index in range(int(blocks)):
        cycle_index = block_index % cycle_length
        for cell_index, cell_id in enumerate(cells):
            offset = (cycle_index + cell_index) % len(permuted)
            rotated = permuted[offset:] + permuted[:offset]
            order = rotated if cycle_index < len(permuted) else tuple(reversed(rotated))
            orders.append({
                "block_index": block_index,
                "cell_index": cell_index,
                "cell_id": cell_id,
                "order": list(order),
            })
    return orders


def schedule_rows(*, method_ids: Sequence[str] = TIMING_PATH_IDS, cell_ids: Sequence[str] = contract.CELL_ORDER, blocks: int = DEFAULT_BLOCKS, seed: int = DEFAULT_SEED) -> tuple[list[dict[str, Any]], str]:
    rows = balanced_orders(method_ids=method_ids, cell_ids=cell_ids, blocks=blocks, seed=seed)
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return rows, hashlib.sha256(encoded).hexdigest()


def ratio_summary(values: Sequence[float], *, threshold: float = QUALIFICATION_THRESHOLD) -> dict[str, Any]:
    ratios = [float(x) for x in values]
    if len(ratios) < 2 or any(not math.isfinite(x) or x <= 0.0 for x in ratios):
        raise TimingError("timing ratio observations are malformed")
    mean = statistics.fmean(ratios)
    stdev = statistics.stdev(ratios)
    df = len(ratios) - 1
    critical = _t_critical_95(df)
    margin = critical * stdev / math.sqrt(len(ratios))
    lower, upper = mean - margin, mean + margin
    if upper <= float(threshold):
        decision = "PASS"
    elif lower > float(threshold):
        decision = "FAIL"
    else:
        decision = "INCONCLUSIVE"
    return {
        "blocks": len(ratios),
        "degrees_of_freedom": df,
        "critical_value": critical,
        "block_ratios": ratios,
        "mean_ratio": mean,
        "stdev_ratio": stdev,
        "ci_level": CI_LEVEL,
        "ci_lower": lower,
        "ci_upper": upper,
        "threshold": float(threshold),
        "decision": decision,
    }


def method_variability(seconds: Sequence[float]) -> dict[str, float | int]:
    values = [float(x) for x in seconds]
    if not values or any(not math.isfinite(x) or x < 0.0 for x in values):
        raise TimingError("method timing observations are malformed")
    ordered = sorted(values)
    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - position) + ordered[high] * (position - low)
    return {
        "records": len(values),
        "mean_seconds": statistics.fmean(values),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "p05_seconds": quantile(0.05),
        "p50_seconds": quantile(0.50),
        "p95_seconds": quantile(0.95),
        "max_seconds": max(values),
    }


def _totals_by_block_cell(blocks: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, dict[str, float]]]:
    totals: dict[int, dict[str, dict[str, float]]] = {}
    for block in blocks:
        block_index = int(block["block_index"])
        cell_totals: dict[str, dict[str, float]] = {}
        entries = block.get("entries")
        if not isinstance(entries, Sequence):
            raise TimingError(f"timing block lacks entries: {block_index}")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise TimingError("timing block entry is malformed")
            cell_id = str(entry["cell_id"])
            method_id = str(entry["method_id"])
            duration = float(entry["measured_seconds_sum"])
            if cell_id not in contract.CELL_ORDER or method_id not in TIMING_PATH_IDS or not math.isfinite(duration) or duration <= 0.0:
                raise TimingError("timing block entry binding/value changed")
            if method_id in cell_totals.setdefault(cell_id, {}):
                raise TimingError(f"duplicate timing entry: {block_index}/{cell_id}/{method_id}")
            cell_totals[cell_id][method_id] = duration
        if set(cell_totals) != set(contract.CELL_ORDER) or any(set(row) != set(TIMING_PATH_IDS) for row in cell_totals.values()):
            raise TimingError(f"timing block matrix incomplete: {block_index}")
        totals[block_index] = cell_totals
    if len(totals) != DEFAULT_BLOCKS:
        raise TimingError(f"authoritative timing requires exactly {DEFAULT_BLOCKS} blocks")
    if sorted(totals) != list(range(DEFAULT_BLOCKS)):
        raise TimingError("timing block indices are not contiguous")
    return totals


def _comparison(totals: Mapping[int, Mapping[str, Mapping[str, float]]], *, numerator_id: str, denominator_id: str, threshold: float) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    for cell_id in contract.CELL_ORDER:
        ratios: list[float] = []
        for block_index in sorted(totals):
            row = totals[block_index][cell_id]
            ratios.append(float(row[numerator_id]) / float(row[denominator_id]))
        by_cell[cell_id] = ratio_summary(ratios, threshold=threshold)
    return {"numerator_method_id": numerator_id, "denominator_method_id": denominator_id, "by_cell": by_cell}


def summarize_blocks(blocks: Sequence[Mapping[str, Any]], *, candidate_method_id: str = contract.PRIMARY_METHOD_ID, denominator_method_id: str = contract.PRIMARY_CONTROL_METHOD_ID, alias_method_id: str = ALIAS_METHOD_ID) -> dict[str, Any]:
    """Classify candidate cost only after the alias-control criterion passes."""

    if candidate_method_id not in contract.METHOD_ORDER or denominator_method_id not in contract.METHOD_ORDER or alias_method_id != ALIAS_METHOD_ID:
        raise TimingError("timing comparison roles changed")
    totals = _totals_by_block_cell(blocks)
    comparisons = {
        method_id: _comparison(totals, numerator_id=method_id, denominator_id=denominator_method_id, threshold=QUALIFICATION_THRESHOLD)
        for method_id in contract.METHOD_ORDER if method_id != denominator_method_id
    }
    alias_by_cell = _comparison(totals, numerator_id=alias_method_id, denominator_id=denominator_method_id, threshold=1.0)["by_cell"]
    alias_outside = [cell for cell, row in alias_by_cell.items() if row["ci_lower"] > ALIAS_RUNTIME_BAND[1] or row["ci_upper"] < ALIAS_RUNTIME_BAND[0]]
    alias_contained = [cell for cell, row in alias_by_cell.items() if row["ci_lower"] >= ALIAS_RUNTIME_BAND[0] and row["ci_upper"] <= ALIAS_RUNTIME_BAND[1]]
    alias_decision = "FAIL" if alias_outside else ("PASS" if len(alias_contained) == len(contract.CELL_ORDER) else "INCONCLUSIVE")
    candidate_cells = comparisons[candidate_method_id]["by_cell"]
    candidate_failed = [cell for cell, row in candidate_cells.items() if row["ci_lower"] > QUALIFICATION_THRESHOLD]
    candidate_passed = [cell for cell, row in candidate_cells.items() if row["ci_upper"] <= QUALIFICATION_THRESHOLD]
    raw_decision = "FAIL" if candidate_failed else ("PASS" if len(candidate_passed) == len(contract.CELL_ORDER) else "INCONCLUSIVE")
    decision = raw_decision if alias_decision == "PASS" else ("INVALID_ALIAS_CONTROL" if alias_decision == "FAIL" else "INCONCLUSIVE")
    all_entries = [entry for block in blocks for entry in block["entries"]]
    variability: dict[str, list[float]] = {method_id: [] for method_id in TIMING_PATH_IDS}
    for entry in all_entries:
        variability[str(entry["method_id"])].extend(float(x) for x in entry.get("per_record_measured_seconds", []))
    return {
        "qualification": {
            "decision": decision,
            "raw_per_cell_decision": raw_decision,
            "measurement_valid": alias_decision == "PASS",
            "cost_failure_demonstrated": alias_decision == "PASS" and raw_decision == "FAIL",
            "candidate_method_id": candidate_method_id,
            "denominator_method_id": denominator_method_id,
            "threshold": QUALIFICATION_THRESHOLD,
            "per_cell": candidate_cells,
            "failed_cells": candidate_failed,
            "passed_cells": candidate_passed,
        },
        "comparisons_vs_fixed": comparisons,
        "alias_control": {
            "method_id": alias_method_id,
            "runtime_ratio_band": list(ALIAS_RUNTIME_BAND),
            "by_cell": alias_by_cell,
            "persistent_deviation_cells": alias_outside,
            "fully_contained_cells": alias_contained,
            "runtime_order_control_valid": alias_decision == "PASS",
            "decision": alias_decision,
            "exact_prediction_equivalence_required": True,
        },
        "record_variability": {method_id: method_variability(values) for method_id, values in variability.items()},
        "block_count": len(totals),
    }


def validate_alias_equivalence(*, state_bindings: Mapping[str, Mapping[str, Any]], predictions: Mapping[str, torch.Tensor], alias_method_id: str = ALIAS_METHOD_ID, denominator_method_id: str = contract.PRIMARY_CONTROL_METHOD_ID) -> dict[str, Any]:
    """Require the alias to use identical state/loader bindings and IDs."""

    if alias_method_id not in state_bindings or denominator_method_id not in state_bindings:
        raise TimingError("alias state bindings are absent")
    if dict(state_bindings[alias_method_id]) != dict(state_bindings[denominator_method_id]):
        raise TimingError("alias state/loader binding differs from fixed method")
    mismatches = []
    for cell_id in contract.CELL_ORDER:
        left = predictions.get(f"{denominator_method_id}::{cell_id}")
        right = predictions.get(f"{alias_method_id}::{cell_id}")
        if left is None or right is None or not torch.equal(torch.as_tensor(left), torch.as_tensor(right)):
            mismatches.append(cell_id)
    if mismatches:
        raise TimingError(f"alias predictions differ: {mismatches}")
    return {"status": "PASS", "alias_method_id": alias_method_id, "denominator_method_id": denominator_method_id, "cells": list(contract.CELL_ORDER), "exact_prediction_equivalence": True, "identical_state_loader_binding": True}


def build_timing_plan(*, generated_utc: str, records_per_cell: int = DEFAULT_RECORDS_PER_CELL, blocks: int = DEFAULT_BLOCKS, seed: int = DEFAULT_SEED, maximum_seconds: int = DEFAULT_MAX_SECONDS) -> dict[str, Any]:
    if blocks != DEFAULT_BLOCKS or records_per_cell != DEFAULT_RECORDS_PER_CELL:
        raise TimingError("authoritative TRR-0009 timing requires 40 blocks and 32 records per cell")
    rows, schedule_sha256 = schedule_rows(blocks=blocks, seed=seed)
    return {
        "schema": contract.TIMING_PLAN_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_TIMING_PLAN_BEFORE_MEASUREMENT",
        "generated_utc": str(generated_utc),
        "method_order": list(contract.METHOD_ORDER),
        "timing_paths": list(TIMING_PATH_IDS),
        "alias_control": {"method_id": ALIAS_METHOD_ID, "denominator_method_id": contract.PRIMARY_CONTROL_METHOD_ID, "band": list(ALIAS_RUNTIME_BAND), "criterion": "each cell 95% CI fully contained in [0.95,1.05]"},
        "cell_order": list(contract.CELL_ORDER),
        "records_per_cell": int(records_per_cell),
        "blocks": int(blocks),
        "block_cycle_length": 2 * len(TIMING_PATH_IDS),
        "block_cycles": int(blocks // (2 * len(TIMING_PATH_IDS))),
        "warmup_runs": DEFAULT_WARMUP_RUNS,
        "measured_runs": DEFAULT_MEASURED_RUNS,
        "seed": int(seed),
        "candidate_ratio_threshold": QUALIFICATION_THRESHOLD,
        "ci_level": CI_LEVEL,
        "maximum_seconds": int(maximum_seconds),
        "schedule_sha256": schedule_sha256,
        "schedule_rows": rows,
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }



@dataclass(frozen=True)
class TimingCell:
    cell_id: str
    activations: torch.Tensor
    valid_mask: torch.Tensor
    position_ids: torch.Tensor
    descriptor: dict[str, Any]


@dataclass(frozen=True)
class TimingConfig:
    repository_root: Path
    registration_path: Path
    output_path: Path
    device: str = "cuda"
    records_per_cell: int = DEFAULT_RECORDS_PER_CELL
    blocks: int = DEFAULT_BLOCKS
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    seed: int = DEFAULT_SEED
    maximum_seconds: int = DEFAULT_MAX_SECONDS


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise TimingError(f"repository root is unavailable: {root}")
    return root


def _rss_bytes() -> int:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError) as exc:
        raise TimingError("process RSS telemetry is unavailable") from exc
    return value if platform.system() == "Darwin" else value * 1024


def _host_available_bytes() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise TimingError("host available-memory telemetry is unavailable") from exc
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1]) * 1024
                except ValueError as exc:
                    raise TimingError("host available-memory telemetry is malformed") from exc
    raise TimingError("host available-memory telemetry is unavailable")


def _load_average() -> list[float] | None:
    try:
        fields = Path("/proc/loadavg").read_text(encoding="ascii").split()
        return [float(value) for value in fields[:3]]
    except (OSError, UnicodeError, ValueError):
        try:
            return [float(value) for value in os.getloadavg()]
        except (AttributeError, OSError, ValueError):
            return None


def _gpu_telemetry(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    if not torch.cuda.is_available():
        raise TimingError("CUDA is unavailable")
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,memory.used,memory.free,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(query, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TimingError("GPU telemetry is unavailable") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise TimingError(f"GPU telemetry failed: {result.stderr.strip()}")
    fields = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    free, total = torch.cuda.mem_get_info(device)
    return {
        "nvidia_smi": {str(index): value for index, value in enumerate(fields)},
        "cuda_free_bytes": int(free),
        "cuda_total_bytes": int(total),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
    }


def _telemetry(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "utc": _utc_now(),
        "monotonic_seconds": time.perf_counter(),
        "process_max_rss_bytes": _rss_bytes(),
        "host_available_bytes": _host_available_bytes(),
        "load_average": _load_average(),
    }
    if device.type == "cuda":
        result["gpu"] = _gpu_telemetry(device)
        result["foreign_compute_apps"] = runner._gpu_compute_apps()
    return result


def _full_guard(*, device: torch.device, started: float, maximum_seconds: int, stage: str) -> dict[str, Any]:
    guard_started = time.perf_counter()
    # The runner guard is fail-closed and performs the foreign-process check;
    # use it at block boundaries so nvidia-smi is outside every measured row.
    runner._guard(device=device, guard=contract.RESOURCE_GUARD, started=started, stage=stage)
    telemetry = _telemetry(device)
    return {
        "stage": stage,
        "status": "PASS",
        "elapsed_seconds": float(time.perf_counter() - started),
        "guard_overhead_seconds": float(time.perf_counter() - guard_started),
        "telemetry": telemetry,
    }


def _light_guard(*, device: torch.device, started: float, maximum_seconds: int, stage: str) -> None:
    if time.perf_counter() - started > float(maximum_seconds):
        raise TimingError(f"timing wall-time guard exceeded at {stage}")
    available = _host_available_bytes()
    rss = _rss_bytes()
    if available < int(contract.RESOURCE_GUARD["minimum_host_available_bytes"]):
        raise TimingError(f"host available-memory guard exceeded at {stage}: {available}")
    if rss > int(contract.RESOURCE_GUARD["maximum_rss_bytes"]):
        raise TimingError(f"host RSS guard exceeded at {stage}: {rss}")
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        if int(free) < int(contract.RESOURCE_GUARD["minimum_free_gpu_bytes"]):
            raise TimingError(f"free-GPU guard exceeded at {stage}: {free}")
        if int(torch.cuda.memory_reserved(device)) > int(contract.RESOURCE_GUARD["maximum_reserved_gpu_bytes"]):
            raise TimingError(f"reserved-GPU guard exceeded at {stage}")


def _file_record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TimingError(f"{description} is unavailable: {path}")
    try:
        return {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": contract.sha256_file(path),
        }
    except (OSError, contract.ContractError) as exc:
        raise TimingError(f"{description} could not be hashed") from exc


def _load_registered(config: TimingConfig, *, root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    registration_path = Path(config.registration_path).expanduser().resolve()
    registration = contract.load_json(registration_path, description="TRR-0009 registration")
    try:
        contract.validate_registration(registration, repository_root=root, verify_assets=True)
    except contract.ContractError as exc:
        raise TimingError(f"registration validation failed: {exc}") from exc
    initial_registration_sha256 = contract.sha256_file(registration_path)
    if registration.get("initialization_equivalence", {}).get("status") != "PASS":
        raise TimingError("initialization/forward equivalence is not PASS")
    code_commit = str(registration.get("code_commit", ""))
    if runner._git_head(root) != code_commit:
        raise TimingError("registration code commit is not current")
    timing_binding = registration.get("timing_plan")
    if not isinstance(timing_binding, Mapping):
        raise TimingError("registered timing plan is absent")
    try:
        timing_record = contract.validate_file_record(timing_binding, repository_root=root, description="timing plan", verify=True)
        plan = contract.load_json(Path(timing_record["path"]), description="timing plan")
    except contract.ContractError as exc:
        raise TimingError(str(exc)) from exc
    if plan.get("schema") != contract.TIMING_PLAN_SCHEMA or plan.get("task_id") != TASK_ID or plan.get("status") != "FROZEN_TIMING_PLAN_BEFORE_MEASUREMENT":
        raise TimingError("timing plan identity or status changed")
    if plan.get("truth_opened") is not False or plan.get("candidate_arrays_persisted") is not False:
        raise TimingError("timing plan records forbidden access")
    expected_plan = build_timing_plan(
        generated_utc=str(plan.get("generated_utc")),
        records_per_cell=config.records_per_cell,
        blocks=config.blocks,
        seed=config.seed,
        maximum_seconds=config.maximum_seconds,
    )
    for key in ("method_order", "timing_paths", "alias_control", "cell_order", "records_per_cell", "blocks", "block_cycle_length", "block_cycles", "warmup_runs", "measured_runs", "seed", "candidate_ratio_threshold", "ci_level", "maximum_seconds", "schedule_sha256", "schedule_rows"):
        if plan.get(key) != expected_plan.get(key):
            raise TimingError(f"timing plan binding changed: {key}")
    return registration, timing_record, plan, initial_registration_sha256


def _load_timing_cells(*, registration: Mapping[str, Any], root: Path, records_per_cell: int) -> tuple[dict[str, TimingCell], dict[str, Any]]:
    try:
        observation_binding = registration["observation_manifest"]
        observation_record = contract.validate_file_record(observation_binding, repository_root=root, description="observation manifest", verify=True)
        observations = contract.load_json(Path(observation_record["path"]), description="observation manifest")
        observations = contract.validate_observation_manifest(observations, repository_root=root, verify_assets=True)
    except (KeyError, contract.ContractError) as exc:
        raise TimingError(f"observation binding failed: {exc}") from exc
    cells: dict[str, TimingCell] = {}
    rows = contract._as_cells(observations)
    for cell_id in contract.CELL_ORDER:
        total = contract.records_for_cell(observations, cell_id)
        if total < records_per_cell:
            raise TimingError(f"timing cell has fewer than {records_per_cell} records: {cell_id}")
        activations: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        iterator = runner._iter_rows(rows[cell_id], records=total)
        try:
            for row_index, activation, mask, position in iterator:
                if row_index >= records_per_cell:
                    break
                activations.append(activation)
                masks.append(mask)
                positions.append(position)
        finally:
            del iterator
        if len(activations) != records_per_cell:
            raise TimingError(f"timing row extraction is incomplete: {cell_id}")
        cells[cell_id] = TimingCell(
            cell_id=cell_id,
            activations=torch.stack(activations, dim=0).contiguous(),
            valid_mask=torch.stack(masks, dim=0).to(dtype=torch.bool).contiguous(),
            position_ids=torch.stack(positions, dim=0).to(dtype=torch.long).contiguous(),
            descriptor=dict(rows[cell_id].get("observation", rows[cell_id])),
        )
    return cells, observation_record


def _same_public_record(actual: Mapping[str, Any], expected: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise TimingError(f"{description} {key} binding changed")


def _load_frozen_prediction_prefixes(*, registration: Mapping[str, Any], registration_path: Path, root: Path, records_per_cell: int) -> tuple[dict[str, str], dict[str, Any]]:
    """Load the public runner matrix and digest its first timed rows.

    Timing uses the first 32 rows of each frozen public prediction cell.  The
    run manifest and every prediction file are rehashed here so a timing run
    cannot silently compare against a changed output root or stale prediction.
    """
    output_root = runner._output_root(registration, root=root)
    run_path = output_root / "run_manifest.json"
    try:
        run = contract.load_json(run_path, description="frozen public prediction run manifest")
        run_record = contract.validate_file_record(
            {"path": str(run_path), "bytes": run_path.stat().st_size, "sha256": contract.sha256_file(run_path)},
            repository_root=root,
            description="frozen public prediction run manifest",
            verify=True,
        )
    except (OSError, contract.ContractError) as exc:
        raise TimingError(f"frozen public prediction run manifest is unavailable: {run_path}") from exc
    if run.get("schema") != contract.RUN_SCHEMA or run.get("task_id") != TASK_ID or run.get("status") != "PUBLIC_PREDICTIONS_COMPLETE_BEFORE_TRUTH":
        raise TimingError("frozen public prediction run manifest identity/status changed")
    if run.get("truth_opened") is not False or run.get("candidate_arrays_persisted") is not False:
        raise TimingError("frozen public prediction run manifest records forbidden access")
    if run.get("code_commit") != registration.get("code_commit"):
        raise TimingError("frozen public prediction code binding differs from registration")
    registration_record = _file_record(Path(registration_path), root=root, description="registration")
    nested_registration = run.get("registration")
    if not isinstance(nested_registration, Mapping):
        raise TimingError("frozen public prediction run manifest lacks registration binding")
    _same_public_record(nested_registration, registration_record, description="frozen public prediction registration")
    if run.get("observation_manifest") != registration.get("observation_manifest"):
        raise TimingError("frozen public prediction observation binding differs from registration")
    predictions = run.get("predictions")
    expected_keys = {f"{method_id}::{cell_id}" for method_id in contract.METHOD_ORDER for cell_id in contract.CELL_ORDER}
    if not isinstance(predictions, Mapping) or set(predictions) != expected_keys:
        raise TimingError("frozen public prediction matrix is incomplete")
    prefix_digests: dict[str, str] = {}
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            key = f"{method_id}::{cell_id}"
            binding = predictions.get(key)
            if not isinstance(binding, Mapping):
                raise TimingError(f"frozen public prediction binding is malformed: {key}")
            expected_path = contract.expected_prediction_path(output_root, cell_id=cell_id, method_id=method_id).resolve()
            actual_path = Path(str(binding.get("path", ""))).expanduser().resolve()
            if actual_path != expected_path:
                raise TimingError(f"frozen public prediction path changed: {key}")
            checked = contract.validate_file_record(binding, repository_root=root, description=f"frozen public prediction {key}", verify=True)
            domain = cell_id.split("__", 1)[0]
            records = int(registration["records_by_domain"][domain])
            expected_metadata = {
                "schema": contract.PREDICTION_SCHEMA,
                "task_id": TASK_ID,
                "registration_sha256": registration_record["sha256"],
                "cell_id": cell_id,
                "method_id": method_id,
                "records": str(records),
                "truth_opened": "false",
                "candidate_arrays_persisted": "false",
            }
            try:
                values, _metadata = contract.load_prediction_file(Path(checked["path"]), records=records, expected_metadata=expected_metadata)
            except contract.ContractError as exc:
                raise TimingError(f"frozen public prediction metadata changed: {key}") from exc
            prefix = values[:records_per_cell].contiguous()
            prefix_digests[key] = contract.tensor_digest(prefix)
    return prefix_digests, run_record


def _load_timing_models(*, registration: Mapping[str, Any], root: Path, device: torch.device) -> tuple[dict[str, torch.nn.Module], dict[str, Any], dict[str, Mapping[str, Any]]]:
    rows = {str(row["id"]): row for row in registration.get("methods", ()) if isinstance(row, Mapping)}
    models: dict[str, torch.nn.Module] = {}
    evidence: dict[str, Any] = {}
    for method_id in contract.METHOD_ORDER:
        row = rows.get(method_id)
        if row is None:
            raise TimingError(f"registered method is absent: {method_id}")
        started = time.perf_counter()
        try:
            model, loaded = runner._load_decoder(row, root=root, device=device)
            runner._synchronize(device)
        except Exception as exc:
            raise TimingError(f"registered model could not be loaded: {method_id}") from exc
        models[method_id] = model
        loaded["model_preparation_seconds"] = float(time.perf_counter() - started)
        evidence[method_id] = loaded
    # The alias is the exact same model object and binding as the fixed arm;
    # it never introduces a second state or loader.
    models[ALIAS_METHOD_ID] = models[contract.PRIMARY_CONTROL_METHOD_ID]
    bindings: dict[str, Mapping[str, Any]] = {
        method_id: {"state": rows[method_id]["state"], "loader": rows[method_id]["loader"]}
        for method_id in contract.METHOD_ORDER
    }
    bindings[ALIAS_METHOD_ID] = bindings[contract.PRIMARY_CONTROL_METHOD_ID]
    return models, evidence, bindings


def _check_alias_before_timing(*, models: Mapping[str, torch.nn.Module], bindings: Mapping[str, Mapping[str, Any]], embedding: torch.Tensor, cells: Mapping[str, TimingCell], device: torch.device) -> dict[str, Any]:
    checked: dict[str, torch.Tensor] = {}
    for cell_id in contract.CELL_ORDER:
        cell = cells[cell_id]
        fixed_rows: list[torch.Tensor] = []
        alias_rows: list[torch.Tensor] = []
        for row_index in range(int(cell.activations.shape[0])):
            fixed = runner.predict_current_h(models[contract.PRIMARY_CONTROL_METHOD_ID], embedding, cell.activations[row_index], cell.valid_mask[row_index], device=device)
            alias = runner.predict_current_h(models[ALIAS_METHOD_ID], embedding, cell.activations[row_index], cell.valid_mask[row_index], device=device)
            if not torch.equal(fixed, alias):
                raise TimingError(f"fixed/alias predictions differ before timing: {cell_id}/{row_index}")
            fixed_rows.append(fixed)
            alias_rows.append(alias)
        checked[f"{contract.PRIMARY_CONTROL_METHOD_ID}::{cell_id}"] = torch.stack(fixed_rows)
        checked[f"{ALIAS_METHOD_ID}::{cell_id}"] = torch.stack(alias_rows)
    receipt = validate_alias_equivalence(state_bindings=bindings, predictions=checked)
    receipt["records_per_cell"] = int(next(iter(cells.values())).activations.shape[0])
    receipt["scope"] = "exact current-H prediction IDs on the timed fixture before any measured block"
    return receipt


def _run_timed_cell(*, model: torch.nn.Module, embedding: torch.Tensor, cell: TimingCell, method_id: str, block_index: int, device: torch.device, started: float, config: TimingConfig, expected_prediction_sha256: str | None = None) -> dict[str, Any]:
    if config.warmup_runs != 1:
        raise TimingError("the frozen timing contract requires exactly one warmup run")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    warmup_seconds: list[float] = []
    measured_seconds: list[float] = []
    outputs: list[torch.Tensor] = []
    for row_index in range(config.records_per_cell):
        if row_index % 8 == 0:
            _light_guard(device=device, started=started, maximum_seconds=config.maximum_seconds, stage=f"row_{method_id}_{cell.cell_id}_{block_index}_{row_index}")
        runner._synchronize(device)
        t0 = time.perf_counter()
        warm = runner.predict_current_h(model, embedding, cell.activations[row_index], cell.valid_mask[row_index], device=device)
        runner._synchronize(device)
        warmup_seconds.append(float(time.perf_counter() - t0))
        runner._synchronize(device)
        t1 = time.perf_counter()
        measured = runner.predict_current_h(model, embedding, cell.activations[row_index], cell.valid_mask[row_index], device=device)
        runner._synchronize(device)
        measured_seconds.append(float(time.perf_counter() - t1))
        if not torch.equal(warm, measured):
            raise TimingError(f"warmup/measured prediction mismatch: {method_id}/{cell.cell_id}/{row_index}")
        outputs.append(measured)
    output_tensor = torch.stack(outputs, dim=0).contiguous()
    prediction_sha256 = contract.tensor_digest(output_tensor)
    if expected_prediction_sha256 is not None and prediction_sha256 != str(expected_prediction_sha256):
        raise TimingError(f"timed predictions differ from frozen public outputs: {method_id}/{cell.cell_id}/block_{block_index}")
    peak = {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "process_max_rss_bytes": _rss_bytes(),
        "host_available_bytes": _host_available_bytes(),
    }
    return {
        "method_id": method_id,
        "cell_id": cell.cell_id,
        "block_index": int(block_index),
        "records": int(config.records_per_cell),
        "warmup_runs_per_record": int(config.warmup_runs),
        "measured_runs_per_record": 1,
        "warmup_seconds_sum": float(sum(warmup_seconds)),
        "measured_seconds_sum": float(sum(measured_seconds)),
        "warmup_variability": method_variability(warmup_seconds),
        "measured_variability": method_variability(measured_seconds),
        "per_record_measured_seconds": measured_seconds,
        "warmup_output_exact_match_measured": True,
        "measured_output_selected": True,
        "prediction_sha256": prediction_sha256,
        "peak_memory": peak,
        "timed_interval": "synchronized BF16 current-H staging -> FP32 decoder -> full-vocabulary argmax -> CPU IDs",
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }


def _timed_blocks(*, models: Mapping[str, torch.nn.Module], embedding: torch.Tensor, cells: Mapping[str, TimingCell], orders: Sequence[Mapping[str, Any]], device: torch.device, config: TimingConfig, started: float, frozen_prediction_prefix_digests: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block_index in range(config.blocks):
        guard_before = _full_guard(device=device, started=started, maximum_seconds=config.maximum_seconds, stage=f"before_block_{block_index}")
        block_started = time.perf_counter()
        cell_orders = [row for row in orders if int(row["block_index"]) == block_index]
        if len(cell_orders) != len(contract.CELL_ORDER) or {str(row["cell_id"]) for row in cell_orders} != set(contract.CELL_ORDER):
            raise TimingError(f"timing order schedule is incomplete for block {block_index}")
        entries: list[dict[str, Any]] = []
        for order_row in cell_orders:
            cell_id = str(order_row["cell_id"])
            order = [str(value) for value in order_row["order"]]
            if tuple(order) != tuple(TIMING_PATH_IDS):
                if set(order) != set(TIMING_PATH_IDS) or len(order) != len(TIMING_PATH_IDS):
                    raise TimingError(f"timing method order is malformed for {block_index}/{cell_id}")
            for order_index, method_id in enumerate(order):
                guard_started = time.perf_counter()
                _light_guard(device=device, started=started, maximum_seconds=config.maximum_seconds, stage=f"before_{method_id}_{cell_id}_{block_index}")
                guard_before_seconds = float(time.perf_counter() - guard_started)
                frozen_method_id = contract.PRIMARY_CONTROL_METHOD_ID if method_id == ALIAS_METHOD_ID else method_id
                frozen_key = f"{frozen_method_id}::{cell_id}"
                expected_prediction_sha256 = None if frozen_prediction_prefix_digests is None else frozen_prediction_prefix_digests.get(frozen_key)
                if frozen_prediction_prefix_digests is not None and expected_prediction_sha256 is None:
                    raise TimingError(f"frozen public prediction digest is absent: {frozen_key}")
                entry = _run_timed_cell(model=models[method_id], embedding=embedding, cell=cells[cell_id], method_id=method_id, block_index=block_index, device=device, started=started, config=config, expected_prediction_sha256=expected_prediction_sha256)
                guard_started = time.perf_counter()
                _light_guard(device=device, started=started, maximum_seconds=config.maximum_seconds, stage=f"after_{method_id}_{cell_id}_{block_index}")
                guard_after_seconds = float(time.perf_counter() - guard_started)
                entry["order_index"] = int(order_index)
                entry["resource_guard_overhead_seconds"] = guard_before_seconds + guard_after_seconds
                entries.append(entry)
        runner._synchronize(device)
        guard_after = _full_guard(device=device, started=started, maximum_seconds=config.maximum_seconds, stage=f"after_block_{block_index}")
        blocks.append({
            "block_index": int(block_index),
            "order_by_cell": {str(row["cell_id"]): list(row["order"]) for row in cell_orders},
            "started_utc": guard_before["telemetry"]["utc"],
            "ended_utc": guard_after["telemetry"]["utc"],
            "wall_seconds": float(time.perf_counter() - block_started),
            "telemetry_before": guard_before["telemetry"],
            "telemetry_after": guard_after["telemetry"],
            "resource_guard_before": guard_before,
            "resource_guard_after": guard_after,
            "resource_guard_overhead_seconds": float(guard_before["guard_overhead_seconds"] + guard_after["guard_overhead_seconds"]),
            "entries": entries,
        })
    return blocks


def _write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TimingError(f"timing output is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise TimingError(f"timing output could not be written: {path}") from exc
    return _file_record(path, root=path.parents[0], description="timing output")


def _failure_receipt(config: TimingConfig, *, started_utc: str, exc: BaseException) -> dict[str, Any]:
    return {
        "schema": FAILURE_SCHEMA,
        "task_id": TASK_ID,
        "status": "TIMING_FAILED_CLOSED",
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "repository_root": str(config.repository_root),
        "registration_path": str(config.registration_path),
        "output_path": str(config.output_path),
        "device": config.device,
        "records_per_cell": config.records_per_cell,
        "blocks": config.blocks,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "candidate_arrays_persisted": False,
    }


def run_timing(config: TimingConfig) -> dict[str, Any]:
    started_utc = _utc_now()
    started = time.perf_counter()
    root = _root(config.repository_root)
    try:
        if config.records_per_cell != DEFAULT_RECORDS_PER_CELL or config.blocks != DEFAULT_BLOCKS or config.warmup_runs != DEFAULT_WARMUP_RUNS or config.seed != DEFAULT_SEED or config.maximum_seconds != DEFAULT_MAX_SECONDS:
            raise TimingError("authoritative TRR-0009 timing requires the frozen 40-block/32-record/seed/guard settings")
        registration, timing_plan_record, timing_plan, initial_registration_sha256 = _load_registered(config, root=root)
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise TimingError("CUDA is unavailable")
        numerics = runner._configure_numerics(registration["numerical_settings"])
        if numerics.get("settings") != dict(registration["numerical_settings"]):
            raise TimingError("timing numerical settings differ from the runner")
        numerics["deterministic_algorithms_enabled"] = bool(torch.are_deterministic_algorithms_enabled()) if hasattr(torch, "are_deterministic_algorithms_enabled") else None
        initial_guard = _full_guard(device=device, started=started, maximum_seconds=config.maximum_seconds, stage="initial")
        embedding, embedding_evidence = runner._load_embedding(registration, root=root, device=device)
        cells, observation_record = _load_timing_cells(registration=registration, root=root, records_per_cell=config.records_per_cell)
        frozen_prediction_prefix_digests, frozen_run_record = _load_frozen_prediction_prefixes(registration=registration, registration_path=config.registration_path, root=root, records_per_cell=config.records_per_cell)
        models, model_evidence, state_bindings = _load_timing_models(registration=registration, root=root, device=device)
        alias_equivalence = _check_alias_before_timing(models=models, bindings=state_bindings, embedding=embedding, cells=cells, device=device)
        orders = list(timing_plan["schedule_rows"])
        timing_blocks = _timed_blocks(models=models, embedding=embedding, cells=cells, orders=orders, device=device, config=config, started=started, frozen_prediction_prefix_digests=frozen_prediction_prefix_digests)
        summary = summarize_blocks(timing_blocks)
        if summary["block_count"] != DEFAULT_BLOCKS:
            raise TimingError("timing block count changed")
        # Recheck every public binding before the create-only success write.
        contract.validate_registration(registration, repository_root=root, verify_assets=True)
        if runner._git_head(root) != str(registration["code_commit"]):
            raise TimingError("runner code commit changed during timing")
        if contract.sha256_file(Path(config.registration_path).expanduser().resolve()) != initial_registration_sha256:
            raise TimingError("registration file changed during timing")
        _rechecked_prefix_digests, rechecked_run_record = _load_frozen_prediction_prefixes(registration=registration, registration_path=config.registration_path, root=root, records_per_cell=config.records_per_cell)
        if rechecked_run_record != frozen_run_record or _rechecked_prefix_digests != frozen_prediction_prefix_digests:
            raise TimingError("frozen public prediction outputs changed during timing")
        registration_record = _file_record(Path(config.registration_path), root=root, description="registration")
        result = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "status": "TIMING_COMPLETE",
            "truth_opened": False,
            "source_text_or_target_labels": False,
            "candidate_arrays_persisted": False,
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "code_commit": str(registration["code_commit"]),
            "repository_root": str(root),
            "registration": registration_record,
            "observation_manifest": observation_record,
            "timing_plan": timing_plan_record,
            "code_bindings": list(registration.get("code_bindings", [])),
            "method_bindings": {method_id: dict(state_bindings[method_id]) for method_id in TIMING_PATH_IDS},
            "numerical_settings": numerics,
            "frozen_prediction_run_manifest": frozen_run_record,
            "frozen_prediction_prefix_digests": frozen_prediction_prefix_digests,
            "configuration": {
                "device": str(device),
                "records_per_cell": config.records_per_cell,
                "blocks": config.blocks,
                "block_cycle_length": int(timing_plan["block_cycle_length"]),
                "block_cycles": int(timing_plan["block_cycles"]),
                "warmup_runs_per_record": config.warmup_runs,
                "measured_runs_per_record": DEFAULT_MEASURED_RUNS,
                "seed": config.seed,
                "maximum_seconds": config.maximum_seconds,
                "candidate_ratio_threshold": QUALIFICATION_THRESHOLD,
                "ci_level": CI_LEVEL,
            },
            "resource_guard": dict(contract.RESOURCE_GUARD),
            "initial_guard": initial_guard,
            "runtime_embedding": embedding_evidence,
            "model_startup": model_evidence,
            "alias_execution_identity": alias_equivalence,
            "order_schedule": {"rows": orders, "sha256": timing_plan["schedule_sha256"]},
            "blocks": timing_blocks,
            "summary": summary,
        }
        output_record = _write_create_only(config.output_path, result)
        result["timing_receipt"] = output_record
        return result
    except BaseException as exc:
        failure_path = Path(config.output_path).expanduser().resolve().with_name(Path(config.output_path).stem + ".failure.json")
        try:
            _write_create_only(failure_path, _failure_receipt(config, started_utc=started_utc, exc=exc))
        except Exception:
            pass
        if isinstance(exc, TimingError):
            raise
        raise TimingError("TRR-0009 timing failed closed") from exc
    finally:
        gc.collect()
        if "device" in locals() and isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.empty_cache()

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--execute", action="store_true", help="run the owner-frozen balanced timing matrix")
    parser.add_argument("--generated-utc", default="UNSET_BEFORE_OWNER_FREEZE")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--records-per-cell", type=int, default=DEFAULT_RECORDS_PER_CELL)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maximum-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_plan:
        print(json.dumps(build_timing_plan(generated_utc=args.generated_utc), indent=2, sort_keys=True))
        return 0
    if not args.execute:
        print("TRR-0009 timing requires --execute or --print-plan", file=sys.stderr)
        return 2
    if args.registration is None or args.output is None:
        print("TRR-0009 timing --execute requires --registration and --output", file=sys.stderr)
        return 2
    config = TimingConfig(
        repository_root=args.repository_root.expanduser().resolve(),
        registration_path=args.registration.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        device=args.device,
        records_per_cell=args.records_per_cell,
        blocks=args.blocks,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        maximum_seconds=args.maximum_seconds,
    )
    try:
        result = run_timing(config)
    except Exception as exc:
        print(f"TRR-0009 timing failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(config.output_path), "summary": result["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
