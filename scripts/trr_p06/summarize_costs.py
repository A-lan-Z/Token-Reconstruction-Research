#!/usr/bin/env python3
"""Summarize frozen TRR-P06 method and cell costs from compact receipts.

This is a metadata-only auditor.  It reads run manifests, timing JSON, fit
receipts, probe receipts, qualifier receipts, and watchdog time records.  It
does not open observation tensors, prediction tensors, source text, or truth.
The JSON and CSV outputs are report artifacts; the underlying scientific
outputs remain immutable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping


SCHEMA = "token-reconstruction.trr-p06-cost-summary.v1"
METHODS = (
    "p06_positionwise_diagonal",
    "p06_past_only",
    "p06_full_record",
)


class SummaryError(RuntimeError):
    """Raised when compact receipt metadata is incomplete or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"invalid JSON receipt: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path, *, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "relative_to_runtime": str(path.relative_to(root.resolve())) if path.is_relative_to(root.resolve()) else None,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _gib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / (1024**3)


def _peak(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cuda_peak_allocated_bytes": record.get("cuda_peak_allocated_bytes"),
        "cuda_peak_allocated_gib": _gib(record.get("cuda_peak_allocated_bytes")),
        "cuda_peak_reserved_bytes": record.get("cuda_peak_reserved_bytes"),
        "cuda_peak_reserved_gib": _gib(record.get("cuda_peak_reserved_bytes")),
        "process_max_rss_bytes": record.get("process_max_rss_bytes"),
        "process_max_rss_gib": _gib(record.get("process_max_rss_bytes")),
    }


def _max_peak(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    fields = (
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "process_max_rss_bytes",
    )
    result: dict[str, Any] = {}
    for field in fields:
        values = [row.get(field) for row in rows if row.get(field) is not None]
        value = max(values) if values else None
        result[field] = value
        result[field.replace("_bytes", "_gib")] = _gib(value)
    return result


def _watchdog(runtime: Path, name: str) -> dict[str, Any]:
    path = runtime / name / "time.json"
    if not path.exists():
        return {"available": False, "path": str(path)}
    record = _read(path)
    return {
        "available": True,
        "status": record.get("status"),
        "wrapper_exit_code": record.get("wrapper_exit_code"),
        "child_return_code": record.get("child_return_code"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "start_utc": record.get("start_utc"),
        "end_utc": record.get("end_utc"),
        "termination_reason": record.get("termination_reason"),
        "receipt": _asset(path, root=runtime),
    }


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"head": head, "working_tree_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "working_tree_dirty": None}


def _student_rows(runtime: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = runtime / "predictions-r1" / "student_predictions.json"
    manifest = _read(manifest_path)
    cells = manifest.get("student_cells")
    if not isinstance(cells, dict):
        raise SummaryError("student manifest has no student_cells mapping")
    rows: list[dict[str, Any]] = []
    for cell_id, seed_map in sorted(cells.items()):
        if "__" not in cell_id or not isinstance(seed_map, dict):
            raise SummaryError(f"invalid student cell id: {cell_id}")
        domain, target = cell_id.split("__", 1)
        for seed_text, method_map in sorted(seed_map.items(), key=lambda item: int(item[0])):
            if not isinstance(method_map, dict):
                raise SummaryError(f"invalid student seed map: {cell_id}/{seed_text}")
            for method_id in METHODS:
                record = method_map.get(method_id)
                if not isinstance(record, dict):
                    raise SummaryError(f"student method missing: {cell_id}/{seed_text}/{method_id}")
                timing = record.get("timing")
                peak = timing.get("peak_memory") if isinstance(timing, dict) else None
                if not isinstance(timing, dict) or not isinstance(peak, dict):
                    raise SummaryError(f"student timing missing: {cell_id}/{seed_text}/{method_id}")
                measured = [float(value) for value in timing.get("measured_seconds", [])]
                if not measured:
                    raise SummaryError(f"student measured_seconds empty: {cell_id}/{seed_text}/{method_id}")
                rows.append(
                    {
                        "phase": "student_prediction",
                        "scope": cell_id,
                        "cell_id": cell_id,
                        "domain": domain,
                        "target": target,
                        "seed": int(seed_text),
                        "method_id": method_id,
                        "records": int(timing["records"]),
                        "scored_post_bos_tokens": int(record.get("scored_post_bos_tokens", timing.get("records", 0) and 127)),
                        "measured_seconds": sum(measured),
                        "measured_mean_seconds": float(timing["measured_mean_seconds"]),
                        "warmup_seconds": float(timing["warmup_seconds"]),
                        "observation_load_seconds": float(timing["observation_load_seconds"]),
                        "measured_ms_per_record": float(timing["measured_ms_per_record"]),
                        "measured_passes": int(timing["measured_passes"]),
                        "warmup_passes": int(timing["warmup_passes"]),
                        "batch_records": int(timing["batch_records"]),
                        "projection_chunk": int(timing["projection_chunk"]),
                        "repeat_prediction_exact": bool(timing["repeat_prediction_exact"]),
                        "state_selected_step": int(record["state_selected_step"]),
                        "state_sha256": record["state_sha256"],
                        **_peak(peak),
                    }
                )
    expected = len(manifest.get("replicate_seeds", [])) * len(manifest.get("domains", [])) * 2 * len(METHODS)
    if len(rows) != 24 or (expected and len(rows) != expected):
        raise SummaryError(f"expected 24 student method/cell rows, found {len(rows)}")
    method_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cell_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method_groups[row["method_id"]].append(row)
        cell_groups[row["cell_id"]].append(row)

    def aggregate(group_rows: list[dict[str, Any]], *, scope: str) -> dict[str, Any]:
        records = sum(row["records"] for row in group_rows)
        measured = sum(row["measured_seconds"] for row in group_rows)
        return {
            "scope": scope,
            "runs": len(group_rows),
            "records_total": records,
            "scored_post_bos_tokens_per_record": group_rows[0]["scored_post_bos_tokens"],
            "measured_seconds_sum": measured,
            "warmup_seconds_sum": sum(row["warmup_seconds"] for row in group_rows),
            "observation_load_seconds_sum": sum(row["observation_load_seconds"] for row in group_rows),
            "measured_ms_per_record_weighted": 1000.0 * measured / records,
            "all_repeat_predictions_exact": all(row["repeat_prediction_exact"] for row in group_rows),
            "peak": _max_peak(group_rows),
        }

    summary = {
        "manifest": _asset(manifest_path, root=runtime),
        "run_manifest": _asset(runtime / "predictions-r1" / "run_manifest.json", root=runtime),
        "outer_elapsed_seconds": _read(runtime / "predictions-r1" / "run_manifest.json").get("elapsed_seconds"),
        "watchdog": _watchdog(runtime, "watchdog-predictions-r1"),
        "geometry": manifest.get("geometry"),
        "rows": rows,
        "by_method": [aggregate(method_groups[method_id], scope=method_id) for method_id in METHODS],
        "by_cell": [aggregate(cell_groups[cell_id], scope=cell_id) for cell_id in sorted(cell_groups)],
    }
    return summary, rows


def _anchor_rows(runtime: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for domain in ("pile", "finance"):
        path = runtime / "anchor-r2" / "timing" / f"{domain}.json"
        timing_record = _read(path)
        timing = timing_record["timing"]
        peak = timing["peak_memory"]
        rows.append(
            {
                "phase": "a1_a2_anchor",
                "scope": domain,
                "domain": domain,
                "target": "public_base",
                "method_id": timing_record["method_id"],
                "records": int(timing["records"]),
                "measured_seconds": float(timing["measured_seconds_sum"]),
                "warmup_seconds": float(timing["warmup_seconds_sum"]),
                "observation_load_seconds": float(timing["observation_load_seconds"]),
                "row_staging_seconds": float(timing["row_staging_seconds"]),
                "measured_ms_per_record": float(timing["measured_ms_per_record"]),
                "calls": int(timing["calls"]),
                "candidate_simulations": int(timing["candidate_simulations"]),
                "public_prefix_calls": int(timing["public_prefix_calls"]),
                "warmup_output_exact_match_measured": bool(timing["warmup_output_exact_match_measured"]),
                "timed_interval_includes_adapter_cpu_gpu_staging_and_cuda_synchronization": bool(
                    timing["timed_interval_includes_adapter_cpu_gpu_staging_and_cuda_synchronization"]
                ),
                **_peak(peak),
                "timing_asset": _asset(path, root=runtime),
            }
        )
    total_records = sum(row["records"] for row in rows)
    total_measured = sum(row["measured_seconds"] for row in rows)
    summary = {
        "run_manifest": _asset(runtime / "anchor-r2" / "run_manifest.json", root=runtime),
        "outer_elapsed_seconds": _read(runtime / "anchor-r2" / "run_manifest.json").get("elapsed_seconds"),
        "watchdog": _watchdog(runtime, "watchdog-anchor-r2"),
        "rows": rows,
        "total": {
            "domains": len(rows),
            "records_total": total_records,
            "measured_seconds_sum": total_measured,
            "warmup_seconds_sum": sum(row["warmup_seconds"] for row in rows),
            "observation_load_seconds_sum": sum(row["observation_load_seconds"] for row in rows),
            "row_staging_seconds_sum": sum(row["row_staging_seconds"] for row in rows),
            "calls_total": sum(row["calls"] for row in rows),
            "candidate_simulations_total": sum(row["candidate_simulations"] for row in rows),
            "public_prefix_calls_total": sum(row["public_prefix_calls"] for row in rows),
            "measured_ms_per_record_weighted": 1000.0 * total_measured / total_records,
            "warmup_output_exact_match_all": all(row["warmup_output_exact_match_measured"] for row in rows),
            "peak": _max_peak(rows),
        },
    }
    return summary, rows


def _fit_rows(runtime: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = runtime / "main-r1" / "main_fit_receipt.json"
    receipt = _read(path)
    rows: list[dict[str, Any]] = []
    for record in receipt["methods"]:
        peak = record["peak_memory"]
        rows.append(
            {
                "phase": "main_fit",
                "scope": f"seed-{record['seed']}/{record['method_id']}",
                "seed": int(record["seed"]),
                "method_id": record["method_id"],
                "updates": int(record["steps"]),
                "selected_step": int(record["selected_step"]),
                "arm_wall_seconds": float(record["arm_wall_seconds"]),
                "update_seconds": float(record["update_seconds"]),
                "validation_seconds": float(record["validation_seconds"]),
                "selection_metric": record["selection_metric"],
                "best_validation_metric": float(record["best_validation_metric"]),
                "state_sha256": record.get("selected_state_sha256"),
                **_peak(peak),
            }
        )
    summary = {
        "receipt": _asset(path, root=runtime),
        "source_commit": receipt.get("source_commit"),
        "status": receipt.get("status"),
        "selection_metric": receipt.get("selection_metric"),
        "rows": rows,
        "total": {
            "arms": len(rows),
            "updates": sum(row["updates"] for row in rows),
            "arm_wall_seconds_sum": sum(row["arm_wall_seconds"] for row in rows),
            "update_seconds_sum": sum(row["update_seconds"] for row in rows),
            "validation_seconds_sum": sum(row["validation_seconds"] for row in rows),
            "peak": _max_peak(rows),
        },
        "watchdog": _watchdog(runtime, "watchdog-main-r1"),
    }
    return summary, rows


def _probe_summary(runtime: Path, role: str, dirname: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = runtime / dirname
    path = root / "capacity_probe_receipt.json"
    receipt = _read(path)
    rows: list[dict[str, Any]] = []
    for record in receipt["methods"]:
        rows.append(
            {
                "phase": role,
                "scope": record["method_id"],
                "method_id": record["method_id"],
                "seed": int(record["seed"]),
                "updates": int(record["steps"]),
                "arm_wall_seconds": float(record["arm_wall_seconds"]),
                "update_seconds": float(record["update_seconds"]),
                "evaluation_seconds": float(record.get("evaluation_seconds", 0.0)),
                "final_token_accuracy": float(record["final_metrics"]["token_accuracy"]),
                **_peak(record["peak_memory"]),
            }
        )
    summary = {
        "receipt": _asset(path, root=runtime),
        "source_commit": receipt.get("source_commit"),
        "status": receipt.get("status"),
        "steps_per_arm": sorted({row["updates"] for row in rows}),
        "rows": rows,
        "total": {
            "arms": len(rows),
            "updates": sum(row["updates"] for row in rows),
            "arm_wall_seconds_sum": sum(row["arm_wall_seconds"] for row in rows),
            "update_seconds_sum": sum(row["update_seconds"] for row in rows),
            "evaluation_seconds_sum": sum(row["evaluation_seconds"] for row in rows),
            "peak": _max_peak(rows),
        },
        "watchdog": _watchdog(
            runtime,
            "watchdog-capacity-r1" if role == "original_probe" else "watchdog-capacity-retention-replay-r1",
        ),
    }
    return summary, rows


def _qualification_summary(runtime: Path) -> dict[str, Any]:
    path = runtime / "qualification-r2" / "qualification_receipt.json"
    receipt = _read(path)
    row = {
        "phase": "largest_cell_qualification",
        "scope": receipt["method_id"],
        "method_id": receipt["method_id"],
        "updates": int(receipt["updates"]),
        "optimizer_state_parameter_count": int(receipt["optimizer_state_parameter_count"]),
        **_peak(receipt["peak_memory"]),
    }
    return {
        "receipt": _asset(path, root=runtime),
        "source_commit": receipt.get("source_commit"),
        "status": receipt.get("status"),
        "row": row,
        "watchdog": _watchdog(runtime, "watchdog-qualification-r2"),
    }


def _csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    columns = (
        "phase", "scope", "domain", "target", "seed", "method_id", "records", "updates",
        "selected_step", "measured_seconds", "arm_wall_seconds", "update_seconds", "validation_seconds",
        "warmup_seconds", "observation_load_seconds", "row_staging_seconds", "measured_ms_per_record",
        "candidate_simulations", "public_prefix_calls", "calls", "peak_cuda_reserved_bytes", "process_max_rss_bytes",
        "notes",
    )
    output: list[dict[str, Any]] = []
    for row in summary["student"]["rows"]:
        output.append({key: row.get(key) for key in columns})
    for row in summary["anchor"]["rows"]:
        output.append({key: row.get(key) for key in columns})
    for row in summary["main_fit"]["rows"]:
        output.append({key: row.get(key) for key in columns})
    for role in ("original_probe", "retention_replay"):
        for row in summary[role]["rows"]:
            output.append({key: row.get(key) for key in columns})
    output.append({key: summary["qualification"]["row"].get(key) for key in columns})
    return output


def summarize(runtime: Path, output_json: Path, output_csv: Path) -> dict[str, Any]:
    runtime = runtime.expanduser().resolve()
    student, _ = _student_rows(runtime)
    anchor, _ = _anchor_rows(runtime)
    main_fit, _ = _fit_rows(runtime)
    original_probe, _ = _probe_summary(runtime, "original_probe", "capacity-r1")
    retention_replay, _ = _probe_summary(runtime, "retention_replay", "capacity-retention-replay-r1")
    qualification = _qualification_summary(runtime)
    updates = {
        "main_fit_updates": main_fit["total"]["updates"],
        "original_probe_updates": original_probe["total"]["updates"],
        "retention_replay_updates": retention_replay["total"]["updates"],
        "qualification_updates": qualification["row"]["updates"],
    }
    updates["total_updates"] = sum(updates.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": "TRR-P06",
        "status": "PASS_METADATA_ONLY",
        "created_utc": _utc_now(),
        "command": ["python3", "scripts/trr_p06/summarize_costs.py", "--runtime-root", str(runtime), "--output-json", str(output_json), "--output-csv", str(output_csv)],
        "repository": _git_metadata(runtime.parent.parent.parent),
        "scope": {
            "truth_opened": False,
            "source_text_loaded": False,
            "observation_payloads_opened": False,
            "prediction_tensors_opened": False,
            "description": "Compact receipt and timing metadata only; no new inference or scoring.",
        },
        "optimizer_updates": updates,
        "student": student,
        "anchor": anchor,
        "main_fit": main_fit,
        "original_probe": original_probe,
        "retention_replay": retention_replay,
        "qualification": qualification,
        "source_receipts": {
            "capacity_replay_equivalence": _asset(runtime / "capacity-retention-replay-r1" / "retention_equivalence.json", root=runtime),
            "anchor_retry_equivalence": _asset(runtime / "anchor-r2" / "retry_equivalence.json", root=runtime),
            "student_run_manifest": _asset(runtime / "predictions-r1" / "run_manifest.json", root=runtime),
        },
    }
    rows = _csv_rows(result)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "phase", "scope", "domain", "target", "seed", "method_id", "records", "updates",
            "selected_step", "measured_seconds", "arm_wall_seconds", "update_seconds", "validation_seconds",
            "warmup_seconds", "observation_load_seconds", "row_staging_seconds", "measured_ms_per_record",
            "candidate_simulations", "public_prefix_calls", "calls", "peak_cuda_reserved_bytes", "process_max_rss_bytes", "notes",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize(args.runtime_root, args.output_json, args.output_csv)
    except (SummaryError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"TRR-P06 cost summary failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "optimizer_updates": result["optimizer_updates"], "student_rows": len(result["student"]["rows"]), "anchor_rows": len(result["anchor"]["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
