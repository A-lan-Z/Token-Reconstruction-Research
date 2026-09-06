#!/usr/bin/env python3
"""Derive a compact report-only cost table from frozen P04 student receipts.

This utility never loads model weights, observations, source material, tokens, or
truth.  It reads the student freeze, timing receipts, selected-state manifest,
and the outer watchdog receipts and writes create-only JSON/CSV/Markdown files.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

TASK_ID = "TRR-P04"
CONDITIONS = ("public_base", "p04_evaluator_target_update_v1")
METHODS = ("affine_same_data", "student_s", "student_h", "student_d")
SEEDS = (1737, 2711)
FREEZE_SCHEMA = "token-reconstruction.trr-p04-student-prediction-freeze.v1"


class CostTableError(RuntimeError):
    pass


def _load(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CostTableError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CostTableError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CostTableError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_descriptor(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CostTableError(f"{label} must be a regular file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _create_only(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise CostTableError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _git_head() -> str | None:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _implementation_commit(command: Mapping[str, Any]) -> str | None:
    argv = command.get("command")
    if not isinstance(argv, list):
        return None
    for index, value in enumerate(argv[:-1]):
        if value == "--implementation-commit":
            candidate = argv[index + 1]
            if isinstance(candidate, str):
                return candidate
    return None


def _state_rows(manifest: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    if manifest.get("truth_accessed") is not False or manifest.get("evaluation_state_count") != 8:
        raise CostTableError("selected-state manifest is not the expected truth-free eight-state binding")
    rows = manifest.get("states")
    if not isinstance(rows, list) or len(rows) != 8:
        raise CostTableError("selected-state manifest must contain eight selected states")
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CostTableError("selected-state row is malformed")
        key = (str(row.get("method_id")), int(row.get("seed")))
        if key in result or key[0] not in METHODS or key[1] not in SEEDS:
            raise CostTableError(f"unexpected/duplicate selected state: {key}")
        result[key] = row
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(result) != expected:
        raise CostTableError("selected-state manifest identities are incomplete")
    return result


def _finite_number(value: Any, label: str) -> float:
    number = float(value)
    if number < 0:
        raise CostTableError(f"{label} is negative")
    return number


def _row(
    cell: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    timing: Mapping[str, Any],
    watchdog_guard: Mapping[str, Any],
    watchdog_finish: Mapping[str, Any],
) -> dict[str, Any]:
    condition = str(cell.get("condition"))
    method = str(cell.get("method_id"))
    seed = int(cell.get("seed"))
    if condition not in CONDITIONS or method not in METHODS or seed not in SEEDS:
        raise CostTableError(f"unexpected cell identity: {condition}/{method}/{seed}")
    if timing.get("status") != "PASS" or timing.get("truth_accessed") is not False:
        raise CostTableError(f"timing receipt is not a truth-free PASS: {condition}/{method}/{seed}")
    full = timing.get("full_panel")
    anchor = timing.get("anchor_subset")
    startup = timing.get("startup")
    table = timing.get("embedding_table")
    resource = timing.get("resource")
    geometry = timing.get("geometry")
    if not all(isinstance(value, Mapping) for value in (full, anchor, startup, table, resource, geometry)):
        raise CostTableError(f"timing receipt lacks cost sections: {condition}/{method}/{seed}")
    if full.get("repeated_prediction_exact") is not True or anchor.get("repeated_prediction_exact") is not True:
        raise CostTableError(f"timing repeats are not exact: {condition}/{method}/{seed}")
    binding = anchor.get("binding")
    if not isinstance(binding, Mapping) or binding.get("full_panel_slice_exact") is not True:
        raise CostTableError(f"anchor slice is not exact: {condition}/{method}/{seed}")
    actual_gpu_peak = resource.get("peak_gpu_reserved_bytes")
    if actual_gpu_peak is None:
        actual_gpu_peak = resource.get("gpu_peak_reserved_bytes")
    return {
        "condition": condition,
        "method_id": method,
        "seed": seed,
        "selected_step": int(state.get("selected_step")),
        "selected_state_bytes": int(state.get("bytes")),
        "selected_state_sha256": str(state.get("sha256")),
        "shared_table_bytes": int(table.get("bytes")),
        "shared_table_sha256": str(table.get("sha256")),
        "shared_table_load_seconds": _finite_number(startup.get("table_load_seconds"), "table load"),
        "selected_state_load_seconds": _finite_number(startup.get("state_load_seconds"), "state load"),
        "table_plus_selected_state_bytes": int(table.get("bytes")) + int(state.get("bytes")),
        "full_panel_records": int(geometry.get("full_records")),
        "full_panel_scored_positions": int(geometry.get("full_scored_positions")),
        "full_panel_warmup_seconds": [float(value) for value in full.get("warmup_seconds", [])],
        "full_panel_measurement_seconds": [float(value) for value in full.get("measurement_seconds", [])],
        "full_panel_mean_seconds": _finite_number(full.get("mean_seconds"), "full-panel mean"),
        "full_panel_milliseconds_per_record": _finite_number(full.get("milliseconds_per_record"), "full-panel ms/record"),
        "full_panel_milliseconds_per_scored_position": _finite_number(full.get("milliseconds_per_scored_position"), "full-panel ms/position"),
        "full_panel_prediction_digest": str(full.get("prediction_digest")),
        "full_panel_repeated_prediction_exact": bool(full.get("repeated_prediction_exact")),
        "anchor_records": int(geometry.get("anchor_records")),
        "anchor_scored_positions": int(geometry.get("anchor_scored_positions")),
        "anchor_warmup_seconds": [float(value) for value in anchor.get("warmup_seconds", [])],
        "anchor_measurement_seconds": [float(value) for value in anchor.get("measurement_seconds", [])],
        "anchor_mean_seconds": _finite_number(anchor.get("mean_seconds"), "anchor mean"),
        "anchor_milliseconds_per_record": _finite_number(anchor.get("milliseconds_per_record"), "anchor ms/record"),
        "anchor_milliseconds_per_scored_position": _finite_number(anchor.get("milliseconds_per_scored_position"), "anchor ms/position"),
        "anchor_prediction_digest": str(anchor.get("prediction_digest")),
        "anchor_repeated_prediction_exact": bool(anchor.get("repeated_prediction_exact")),
        "anchor_full_panel_slice_exact": bool(binding.get("full_panel_slice_exact")),
        "runner_peak_rss_bytes": int(resource.get("peak_rss_bytes")),
        "runner_gpu_peak_reserved_bytes": int(actual_gpu_peak) if actual_gpu_peak is not None else None,
        "runner_gpu_peak_status": "RECORDED" if actual_gpu_peak is not None else "NOT_RECORDED_BY_RUNNER",
        "watchdog_status": str(watchdog_guard.get("status")),
        "watchdog_child_return_code": watchdog_finish.get("child_return_code"),
        "watchdog_wrapper_exit_code": watchdog_finish.get("wrapper_exit_code"),
        "watchdog_termination_reason": watchdog_guard.get("termination_reason"),
        "watchdog_peak_group_rss_bytes": int(watchdog_guard.get("peak_group_rss_bytes", 0)),
        "watchdog_min_host_mem_available_bytes": int(watchdog_guard.get("minimum_sampled_host_mem_available_bytes", 0)),
        "watchdog_errors": list(watchdog_guard.get("errors", [])),
        "truth_accessed": False,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    prediction_root = args.prediction_root.expanduser().resolve()
    freeze_path = prediction_root / "student_prediction_freeze.json"
    freeze = _load(freeze_path, "student freeze")
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("task_id") != TASK_ID:
        raise CostTableError("student freeze identity changed")
    if freeze.get("status") != "STUDENT_PREDICTIONS_FROZEN_BEFORE_JOINT_FREEZE" or freeze.get("truth_accessed") is not False:
        raise CostTableError("student freeze is not a truth-free PASS")
    cells = freeze.get("cells")
    if not isinstance(cells, list) or len(cells) != 16:
        raise CostTableError("student freeze must contain 16 cells")
    if freeze.get("all_predictions_repeat_exact") is not True or freeze.get("all_anchor_slices_exact") is not True:
        raise CostTableError("student freeze exactness flags are not PASS")
    manifest_path = args.state_manifest.expanduser().resolve()
    manifest = _load(manifest_path, "selected-state manifest")
    states = _state_rows(manifest)
    watchdog_root = args.watchdog_root.expanduser().resolve()
    command = _load(watchdog_root / "command.json", "watchdog command")
    guard = _load(watchdog_root / "resource_guard.json", "watchdog guard")
    finish = _load(watchdog_root / "finish.json", "watchdog finish")
    if finish.get("child_return_code") != 0:
        raise CostTableError("watchdog child did not return zero")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    table_binding: tuple[int, str, float] | None = None
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise CostTableError("freeze cell is malformed")
        method = str(cell.get("method_id")); seed = int(cell.get("seed")); condition = str(cell.get("condition"))
        key = (method, seed, condition)
        if key in seen:
            raise CostTableError(f"duplicate freeze cell: {key}")
        seen.add(key)
        timing_path = Path(str(cell["timing"]["path"])).expanduser().resolve()
        timing = _load(timing_path, f"timing receipt {key}")
        row = _row(cell, state=states[(method, seed)], timing=timing, watchdog_guard=guard, watchdog_finish=finish)
        binding = (row["shared_table_bytes"], row["shared_table_sha256"], row["shared_table_load_seconds"])
        if table_binding is None:
            table_binding = binding
        elif table_binding != binding:
            raise CostTableError("shared table binding differs between timing receipts")
        rows.append(row)
    expected = {(method, seed, condition) for method in METHODS for seed in SEEDS for condition in CONDITIONS}
    if seen != expected:
        raise CostTableError("student freeze cell matrix is incomplete")
    rows.sort(key=lambda row: (CONDITIONS.index(row["condition"]), SEEDS.index(row["seed"]), METHODS.index(row["method_id"])))
    assert table_binding is not None
    source_commit = _implementation_commit(command)
    payload = {
        "schema": "token-reconstruction.trr-p04-student-cost-table.v1",
        "task_id": TASK_ID,
        "status": "REPORT_ONLY_FROM_FROZEN_STUDENT_PREDICTIONS",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "truth_accessed": False,
        "source": {
            "cost_table_generator_checkout_commit": _git_head(),
            "prediction_runner_implementation_commit": source_commit,
            "freeze": _file_descriptor(freeze_path, "student freeze"),
            "selected_state_manifest": _file_descriptor(manifest_path, "selected-state manifest"),
            "watchdog_command": _file_descriptor(watchdog_root / "command.json", "watchdog command"),
            "watchdog_guard": _file_descriptor(watchdog_root / "resource_guard.json", "watchdog guard"),
            "watchdog_finish": _file_descriptor(watchdog_root / "finish.json", "watchdog finish"),
        },
        "shared_startup": {
            "table_bytes": table_binding[0],
            "table_sha256": table_binding[1],
            "table_load_seconds_once_per_runner_process": table_binding[2],
            "table_load_amortization_note": "Each timing receipt repeats this shared startup descriptor; the table was loaded once by the runner process.",
        },
        "watchdog": {
            "status": guard.get("status"),
            "child_return_code": finish.get("child_return_code"),
            "wrapper_exit_code": finish.get("wrapper_exit_code"),
            "termination_reason": guard.get("termination_reason"),
            "peak_group_rss_bytes": guard.get("peak_group_rss_bytes"),
            "minimum_sampled_host_mem_available_bytes": guard.get("minimum_sampled_host_mem_available_bytes"),
            "errors": guard.get("errors", []),
            "interpretation": "The child returned 0 and all frozen cells passed; the outer watchdog remained FAIL_CLOSED because the leader /proc status vanished during post-child cleanup. This is retained as an execution exception, not a watchdog PASS.",
        },
        "rows": rows,
        "summary": {
            "cell_count": len(rows),
            "conditions": list(CONDITIONS),
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "all_repeat_predictions_exact": True,
            "all_anchor_slices_exact": True,
            "runner_gpu_peak_recorded": any(row["runner_gpu_peak_reserved_bytes"] is not None for row in rows),
            "runner_gpu_peak_note": "The actual runner timing receipts record host RSS only; no runner GPU reserved/allocated peak was recorded.",
            "truth_accessed": False,
        },
    }
    return payload


def _csv_rows(payload: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    rows = payload["rows"]
    fields = [
        "condition", "method_id", "seed", "selected_step", "selected_state_bytes", "table_plus_selected_state_bytes",
        "shared_table_load_seconds", "selected_state_load_seconds", "full_panel_records", "full_panel_scored_positions",
        "full_panel_mean_seconds", "full_panel_milliseconds_per_record", "anchor_records", "anchor_scored_positions",
        "anchor_mean_seconds", "anchor_milliseconds_per_record", "runner_peak_rss_bytes", "runner_gpu_peak_reserved_bytes",
        "runner_gpu_peak_status", "watchdog_status", "watchdog_child_return_code", "watchdog_wrapper_exit_code",
        "watchdog_peak_group_rss_bytes", "watchdog_min_host_mem_available_bytes", "full_panel_repeated_prediction_exact",
        "anchor_repeated_prediction_exact", "anchor_full_panel_slice_exact", "truth_accessed",
    ]
    return fields, [{field: row.get(field) for field in fields} for row in rows]


def _markdown(payload: Mapping[str, Any]) -> str:
    fields, rows = _csv_rows(payload)
    table_fields = ["condition", "method_id", "seed", "selected_state_bytes", "full_panel_mean_seconds", "anchor_mean_seconds", "full_panel_milliseconds_per_record", "anchor_milliseconds_per_record", "runner_peak_rss_bytes", "runner_gpu_peak_status", "watchdog_status"]
    lines = [
        "# TRR-P04 student prediction cost table",
        "",
        "Report-only derivation from frozen student prediction/timing receipts; evaluator truth remains unopened.",
        "",
        f"Shared table: {payload['shared_startup']['table_bytes']} bytes; load once per runner process: {payload['shared_startup']['table_load_seconds_once_per_runner_process']:.9f} s.",
        "",
        "| " + " | ".join(table_fields) + " |",
        "|" + "|".join("---" for _ in table_fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in table_fields) + " |")
    lines += [
        "",
        "Runner GPU peak: not recorded by the runner timing receipts. Runner host RSS and outer watchdog RSS are retained in JSON/CSV; the watchdog status is FAIL_CLOSED after child return 0 because the leader `/proc` status disappeared during cleanup.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--state-manifest", type=Path, required=True)
    parser.add_argument("--watchdog-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build(args)
    fields, rows = _csv_rows(payload)
    _create_only(args.output_json, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _create_only(args.output_csv, stream.getvalue())
    _create_only(args.output_md, _markdown(payload))
    print(json.dumps({"status": payload["status"], "rows": len(rows), "output_json": str(args.output_json), "output_csv": str(args.output_csv), "output_md": str(args.output_md)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CostTableError as exc:
        raise SystemExit(f"TRR-P04 student cost table failed: {exc}")
