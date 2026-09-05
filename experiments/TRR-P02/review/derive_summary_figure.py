#!/usr/bin/env python3
"""Derive a compact TRR-P02 summary figure from finalized diagnostics.json only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runtime" / "cpu-public-geometry-20260905-run4"
DIAG = RUN / "diagnostics.json"
OUT = Path(__file__).with_name("summary_geometry.png")


def main() -> None:
    data = json.loads(DIAG.read_text())
    variants = data["shared_offset"]["targeted_full_vocab"]["variants"]
    names = [
        ("raw_boundary", "Raw\nboundary"),
        ("reference_subtraction", "Reference\nsubtract"),
        ("projected_prototype", "Projected\nprototype"),
        ("historical_A1", "Frozen\nA1"),
        ("oracle_mean_centered", "Oracle\ncenter"),
    ]
    top1 = [
        variants[key]["summary"]["top1_correct_count"] / variants[key]["summary"]["count"]
        for key, _ in names
    ]
    mean_rank = [variants[key]["summary"]["true_rank"]["mean"] for key, _ in names]
    median_margin = [
        variants[key]["summary"]["true_other_margin"]["median"] for key, _ in names
    ]

    raw = data["lens_diagnostic"]["raw_separation_equal_position_C1_to_C4"]
    projected = data["lens_diagnostic"]["projected_separation_equal_position_C1_to_C4"]
    ratios = np.array(
        [
            [raw["same_to_different_l2_mean_ratio"], projected["same_to_different_l2_mean_ratio"]],
            [
                raw["same_to_different_cosine_distance_mean_ratio"],
                projected["same_to_different_cosine_distance_mean_ratio"],
            ],
        ]
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 5.05))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.20, wspace=0.36)

    colors = ["#3b6fb6", "#3b6fb6", "#d07a2d", "#d07a2d", "#777777"]
    x0 = np.arange(len(names))
    bars = ax0.bar(x0, top1, color=colors, width=0.66)
    ax0.set_xlim(-0.62, len(names) - 0.38)
    ax0.set_ylim(0, 1.12)
    ax0.set_ylabel("Top-1 fraction")
    ax0.set_title("Full-vocabulary rows (n=12)", pad=10)
    ax0.set_xticks(x0, [label for _, label in names], fontsize=8.5)
    ax0.tick_params(axis="x", pad=4)
    ax0.grid(axis="y", alpha=0.25)
    for bar, frac, rank, margin in zip(bars, top1, mean_rank, median_margin):
        y = max(bar.get_height() + 0.032, 0.035)
        count = round(frac * 12)
        rank_label = f"{rank:.0f}" if rank > 100 else f"{rank:.2f}"
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{count}/12\nrank {rank_label}\nmed m {margin:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            clip_on=False,
        )

    x1 = np.arange(2)
    width = 0.33
    bars_raw = ax1.bar(x1 - width / 2, ratios[:, 0], width, label="Raw", color="#3b6fb6")
    bars_proj = ax1.bar(x1 + width / 2, ratios[:, 1], width, label="Projected", color="#d07a2d")
    ax1.set_xlim(-0.58, 1.58)
    ax1.set_ylim(0, 0.95)
    ax1.set_ylabel("Same / different mean ratio")
    ax1.set_title("Equal-position lens geometry (C1–C4)", pad=10)
    ax1.set_xticks(x1, ["L2", "Cosine distance"], fontsize=9)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, loc="upper right")
    for bars_group in (bars_raw, bars_proj):
        for bar in bars_group:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.022,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
            )

    fig.suptitle(
        "TRR-P02 public geometry diagnostic (finalized run4 JSON)",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.045,
        "Ratios are same-token cross-context / different-token within-context; absolute values are reported in the audit.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.savefig(OUT, dpi=180)
    print(OUT)
    print("bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
