#!/usr/bin/env python3
"""Build the compact TRR-0004 affine public-validation learning-curve figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    ("historical_affine_ce_no_vocab_bias", "no vocabulary bias", "-"),
    ("historical_affine_ce_vocab_bias", "vocabulary bias", "--"),
)
STYLE_COLORS = {"alpaca": "#1769aa", "pile": "#d95f02"}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"fit evidence must be an object: {path}")
    return value


def _plot_arm(ax, path: Path, title: str) -> None:
    evidence = _load(path)
    methods = evidence.get("methods")
    if not isinstance(methods, dict):
        raise ValueError(f"methods missing from {path}")
    for method_id, bias_label, linestyle in METHODS:
        method = methods.get(method_id)
        if not isinstance(method, dict):
            raise ValueError(f"{method_id} missing from {path}")
        curve = method.get("learning_curve")
        if not isinstance(curve, list) or not curve:
            raise ValueError(f"learning curve missing from {path}: {method_id}")
        for style, color in STYLE_COLORS.items():
            x = [int(row["step"]) for row in curve]
            y = [float(row["validation_group_token_accuracy"][style]) for row in curve]
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                marker="o",
                markersize=2.2,
                label=f"{style.title()} / {bias_label}",
            )
        selected_step = method.get("selected_step")
        if isinstance(selected_step, int):
            ax.axvline(selected_step, color="#555555", linewidth=0.7, alpha=0.35)
    configuration = evidence.get("configuration", {})
    position_limit = configuration.get("fit_position_limit")
    if position_limit == 5000:
        scope = "5,000 post-BOS positions"
    else:
        scope = "125,571 positions"
    ax.set_title(f"{title}\n{scope}", fontsize=10)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("public validation token accuracy")
    ax.set_xlim(left=0)
    ax.set_ylim(0.2, 1.01)
    ax.grid(True, linewidth=0.4, alpha=0.35)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small",
        type=Path,
        default=Path("outputs/TRR-0004/fit_small_v1/fit_evidence.json"),
    )
    parser.add_argument(
        "--large",
        type=Path,
        default=Path("outputs/TRR-0004/fit_large_v1/fit_evidence.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/TRR-0004/evidence/affine/affine_learning_curves_v1.svg"),
    )
    args = parser.parse_args()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True, constrained_layout=True)
    _plot_arm(axes[0], args.small, "Small fit")
    _plot_arm(axes[1], args.large, "Large fit")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("TRR-0004 affine decoder public-validation curves", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format=args.output.suffix.lstrip("."), dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
