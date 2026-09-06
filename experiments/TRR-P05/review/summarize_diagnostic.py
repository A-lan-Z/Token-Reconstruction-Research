#!/usr/bin/env python3
"""Make a compact, deterministic summary of the P05 truth-free diagnostic.

This reads only the diagnostic receipt, forward summaries, and gradient-cell
JSON files.  It never opens an evaluator/truth artifact or the per-row JSONL
prediction payloads.  The output is intended to make the report tables
reproducible without copying the large runtime artifacts into the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHOD_ORDER = {"student_s": 0, "student_h": 1, "student_d": 2}
PHASE_ORDER = {"selected": 0, "final": 1}
GROUPS = ("difficult_a1_error", "uniform_audit", "control")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sort_key(state_id: str) -> tuple[int, int, int, str]:
    if state_id == "affine_initial_function":
        return (0, 0, 0, state_id)
    method, seed, phase = state_id.split("-")
    return (
        1,
        int(seed) * 10 + METHOD_ORDER[method],
        PHASE_ORDER[phase],
        state_id,
    )


def compact_group(group: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "rows",
        "correct",
        "accuracy",
        "mean_gold_margin",
        "median_gold_margin",
        "rows_with_top1_tie",
        "teacher_order_agreement",
        "teacher_rank_loss_global",
        "teacher_rank_loss_row_mean",
        "teacher_pairs",
        "teacher_pair_weight_sum",
        "teacher_omitted_ties",
        "teacher_student_ties",
    )
    return {key: group[key] for key in fields if key in group}


def compact_forward(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return {
        "state_id": data["state_id"],
        "accuracy": data["accuracy"],
        "correct": data["correct"],
        "total_rows": data["total_rows"],
        "teacher_rows": data["teacher_rows"],
        "groups": {name: compact_group(data["groups"][name]) for name in GROUPS},
    }


def compact_gradient(path: Path) -> dict[str, Any]:
    data = read_json(path)
    norms = data["gradient_norms"]
    cosines = data["gradient_cosines"]
    losses = data["losses"]
    reductions = data["reductions"]
    ce_norm = norms.get("ce")
    rank_weighted = norms.get("rank_weighted")
    return {
        "state_id": data["state_id"],
        "method_id": data["method_id"],
        "seed": data["seed"],
        "schedule_step": data["schedule_step"],
        "selected_rows": data["selected_rows"],
        "teacher_rows": data["teacher_rows"],
        "teacher_kind_counts": data.get("teacher_kind_counts", {}),
        "rank_pairs": reductions["diagnostic_rank_pairs"],
        "rank_pair_weight_sum": reductions["diagnostic_rank_pair_weight_sum"],
        "rank_loss": losses.get("rank"),
        "ce_loss": losses.get("ce"),
        "negative_gold_margin_loss": losses.get("negative_gold_margin"),
        "gradient_norms": {
            "ce": norms.get("ce"),
            "hard_weighted": norms.get("hard_weighted"),
            "rank_raw": norms.get("rank_raw"),
            "rank_weighted": rank_weighted,
            "negative_gold_margin": norms.get("negative_gold_margin"),
            "actual_total_preclip": norms.get("actual_total_preclip"),
            "clip_factor": norms.get("clip_factor"),
            "post_clip_actual_total": norms.get("post_clip_actual_total"),
        },
        "rank_to_ce_norm_ratio": (
            rank_weighted / ce_norm if ce_norm not in (None, 0) else None
        ),
        "gradient_cosines": {
            "rank_negative_gold_margin": cosines.get("rank_negative_gold_margin"),
            "rank_ce_weighted": cosines.get("ce_rank_weighted"),
            "actual_total_negative_gold_margin": cosines.get(
                "actual_total_negative_gold_margin"
            ),
        },
        "state_tensor_unchanged": data["state_tensor_digest_before"]
        == data["state_tensor_digest_after"],
        "optimizer_step_called": data["optimizer_step_called"],
    }


def delta(a: dict[str, Any], b: dict[str, Any], field: str) -> float | None:
    av = a.get(field)
    bv = b.get(field)
    if av is None or bv is None:
        return None
    return av - bv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    receipt_path = run_dir / "diagnostic_receipt.json"
    receipt = read_json(receipt_path)
    forward = [
        compact_forward(path)
        for path in run_dir.glob("summary-*.json")
    ]
    forward.sort(key=lambda row: state_sort_key(row["state_id"]))
    gradients = [
        compact_gradient(path)
        for path in run_dir.glob("gradient-*.json")
    ]
    gradients.sort(
        key=lambda row: (
            row["seed"],
            METHOD_ORDER[row["method_id"]],
            PHASE_ORDER[row["state_id"].rsplit("-", 1)[-1]],
            row["schedule_step"],
        )
    )

    # One row per (seed, schedule step) is the unique schedule exposure.  The
    # same original batch is intentionally reused at three stored states.
    unique_batches: dict[tuple[int, int], dict[str, Any]] = {}
    for row in gradients:
        unique_batches.setdefault((row["seed"], row["schedule_step"]), row)
    unique_batch_rows = sorted(
        unique_batches.values(), key=lambda row: (row["seed"], row["schedule_step"])
    )
    teacher_rows_by_seed_step = {
        str(seed): [
            {
                "schedule_step": row["schedule_step"],
                "selected_rows": row["selected_rows"],
                "teacher_rows": row["teacher_rows"],
                "rank_pairs": row["rank_pairs"],
                "teacher_kind_counts": row["teacher_kind_counts"],
            }
            for row in unique_batch_rows
            if row["seed"] == seed
        ]
        for seed in sorted({row["seed"] for row in unique_batch_rows})
    }
    active_cells = [row for row in gradients if row["teacher_rows"] > 0]
    zero_cells = [row for row in gradients if row["teacher_rows"] == 0]

    state_by_id = {row["state_id"]: row for row in forward}
    comparisons: dict[str, Any] = {}
    for seed in (1737, 2711):
        d_final = state_by_id[f"student_d-{seed}-final"]["groups"]
        initial = state_by_id["affine_initial_function"]["groups"]
        comparisons[str(seed)] = {
            "final_d_minus_initial": {
                group: {
                    field: delta(d_final[group], initial[group], field)
                    for field in (
                        "accuracy",
                        "mean_gold_margin",
                        "teacher_order_agreement",
                        "teacher_rank_loss_global",
                    )
                    if field in d_final[group] and field in initial[group]
                }
                for group in ("difficult_a1_error", "uniform_audit")
            },
            "final_d_minus_final_h": {
                group: {
                    field: delta(
                        d_final[group],
                        state_by_id[f"student_h-{seed}-final"]["groups"][group],
                        field,
                    )
                    for field in (
                        "accuracy",
                        "mean_gold_margin",
                        "teacher_order_agreement",
                        "teacher_rank_loss_global",
                    )
                    if field in d_final[group]
                    and field in state_by_id[f"student_h-{seed}-final"]["groups"][group]
                }
                for group in ("difficult_a1_error", "uniform_audit", "control")
            },
            "final_d_minus_final_s": {
                group: {
                    field: delta(
                        d_final[group],
                        state_by_id[f"student_s-{seed}-final"]["groups"][group],
                        field,
                    )
                    for field in (
                        "accuracy",
                        "mean_gold_margin",
                        "teacher_order_agreement",
                        "teacher_rank_loss_global",
                    )
                    if field in d_final[group]
                    and field in state_by_id[f"student_s-{seed}-final"]["groups"][group]
                }
                for group in ("difficult_a1_error", "uniform_audit", "control")
            },
        }

    snapshots = receipt["resource_guard"]["snapshots"]
    cuda_snapshots = [row for row in snapshots if row.get("device") == "cuda"]
    resource = {
        "maximum_host_rss_bytes": receipt["peak_rss_bytes"],
        "maximum_recorded_gpu_reserved_bytes": max(
            row.get("max_reserved_bytes", 0) for row in cuda_snapshots
        ),
        "maximum_recorded_gpu_allocated_bytes": max(
            row.get("max_allocated_bytes", 0) for row in cuda_snapshots
        ),
        "minimum_recorded_gpu_free_bytes": min(
            row["free_bytes"] for row in cuda_snapshots if row.get("available")
        ),
        "guard_limits": {
            "maximum_host_rss_bytes": receipt["resource_guard"][
                "maximum_host_rss_bytes"
            ],
            "maximum_reserved_gpu_bytes": receipt["resource_guard"][
                "maximum_reserved_gpu_bytes"
            ],
            "minimum_free_gpu_bytes": receipt["resource_guard"][
                "minimum_free_gpu_bytes"
            ],
        },
    }

    output = {
        "schema": "token-reconstruction.trr-p05-derived.v1",
        "task_id": receipt["task_id"],
        "run": {
            "status": receipt["status"],
            "source_commit": receipt["source_commit"],
            "started_utc": receipt["started_utc"],
            "ended_utc": receipt["ended_utc"],
            "wall_seconds": receipt["wall_seconds"],
            "no_truth_access": receipt["no_truth_access"],
            "optimizer_step_called": receipt["optimizer_step_called"],
            "forward_state_ids": [
                row["state_id"] for row in receipt["states"]["forward"]
            ],
            "forward_count": receipt["sample"]["forward_count"],
            "unique_gradient_batch_count": receipt["sample"]["gradient_batch_count"],
            "gradient_cell_count": len(gradients),
            "receipt_sha256": sha256(receipt_path),
            "inputs": {
                name: (
                    value["sha256"]
                    if "sha256" in value
                    else {subname: subvalue["sha256"] for subname, subvalue in value.items()}
                )
                for name, value in receipt["inputs"].items()
            },
        },
        "resource": resource,
        "forward": forward,
        "comparisons": comparisons,
        "gradient": {
            "cells": gradients,
            "active_cells": len(active_cells),
            "zero_rank_cells": len(zero_cells),
            "active_cell_ids": [
                f"{row['state_id']}@{row['schedule_step']}" for row in active_cells
            ],
            "zero_rank_cell_ids": [
                f"{row['state_id']}@{row['schedule_step']}" for row in zero_cells
            ],
            "unique_schedule_batches": len(unique_batch_rows),
            "active_schedule_batches": sum(
                row["teacher_rows"] > 0 for row in unique_batch_rows
            ),
            "selected_rows_unique_schedule": sum(
                row["selected_rows"] for row in unique_batch_rows
            ),
            "teacher_rows_unique_schedule": sum(
                row["teacher_rows"] for row in unique_batch_rows
            ),
            "rank_pairs_unique_schedule": sum(
                row["rank_pairs"] for row in unique_batch_rows
            ),
            "teacher_rows_across_state_cells": sum(
                row["teacher_rows"] for row in gradients
            ),
            "rank_pairs_across_state_cells": sum(row["rank_pairs"] for row in gradients),
            "teacher_rows_by_seed_step": teacher_rows_by_seed_step,
            "unique_teacher_kind_counts": {
                kind: sum(
                    row["teacher_kind_counts"].get(kind, 0)
                    for row in unique_batch_rows
                )
                for kind in sorted(
                    {
                        kind
                        for row in unique_batch_rows
                        for kind in row["teacher_kind_counts"]
                    }
                )
            },
            "all_clip_factors_one": all(
                row["gradient_norms"]["clip_factor"] == 1.0 for row in gradients
            ),
            "all_state_tensors_unchanged": all(
                row["state_tensor_unchanged"] for row in gradients
            ),
            "rank_margin_cosine_sign_counts": {
                "positive": sum(
                    row["gradient_cosines"]["rank_negative_gold_margin"] > 0
                    for row in active_cells
                ),
                "negative": sum(
                    row["gradient_cosines"]["rank_negative_gold_margin"] < 0
                    for row in active_cells
                ),
                "zero_or_missing": sum(
                    row["gradient_cosines"]["rank_negative_gold_margin"] in (None, 0)
                    for row in active_cells
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
