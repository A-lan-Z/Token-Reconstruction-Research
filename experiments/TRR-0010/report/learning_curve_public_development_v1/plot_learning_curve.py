#!/usr/bin/env python3
"""Plot the registered public-development checkpoint curves for TRR-0010.

This reads only combined_fit_diagnostics_cost_table_v3.json.  It does not
load model states, observations, labels, or truth data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EXPECTED_INPUT_SHA256 = "f0831f79c4a6e737ebbbe6ad9f837fd61967421b421683ddcb8b6db4b01db8c0"
ARM_ORDER = [
    "current_fixed",
    "current_directional",
    "expanded_fixed",
    "expanded_directional",
]
ARM_LABELS = {
    "current_fixed": "Current fixed",
    "current_directional": "Current directional",
    "expanded_fixed": "Expanded fixed",
    "expanded_directional": "Expanded directional",
}
ARM_COLORS = {
    "current_fixed": "#1f77b4",
    "current_directional": "#d95f02",
    "expanded_fixed": "#2ca02c",
    "expanded_directional": "#7b3294",
}
ARM_MARKERS = {
    "current_fixed": "o",
    "current_directional": "s",
    "expanded_fixed": "^",
    "expanded_directional": "D",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_curves(table: dict) -> tuple[list[int], dict[str, list[float]], dict[str, int], float]:
    checkpoint_grid = [int(x) for x in table["checkpoint_grid"]]
    curves: dict[str, list[float]] = {}
    selected: dict[str, int] = {}
    for arm in ARM_ORDER:
        record = table["arms"][arm]
        selected[arm] = int(record["selected_step"])
        points = record["checkpoint_curve"]
        by_step = {int(point["step"]): point for point in points}
        if sorted(by_step) != checkpoint_grid:
            raise ValueError(f"{arm}: checkpoint grid does not match table grid")
        values = []
        for step in checkpoint_grid:
            validation = by_step[step].get("validation") or {}
            value = validation.get("domain_balanced_token_accuracy")
            if not isinstance(value, (float, int)):
                raise ValueError(f"{arm} step {step}: missing domain-balanced accuracy")
            values.append(float(value))
        curves[arm] = values
    baseline = curves["current_directional"][checkpoint_grid.index(0)]
    for arm, values in curves.items():
        if abs(values[0] - baseline) > 1e-12:
            raise ValueError(f"{arm}: step-0 baseline differs from unchanged-start reference")
    return checkpoint_grid, curves, selected, baseline


def render(table_path: Path, output_svg: Path, output_png: Path) -> dict:
    raw = table_path.read_bytes()
    source_path = Path(__file__).resolve()
    source_raw = source_path.read_bytes()
    input_sha256 = hashlib.sha256(raw).hexdigest()
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(
            f"unexpected input hash {input_sha256}; expected frozen public table {EXPECTED_INPUT_SHA256}"
        )
    table = json.loads(raw)
    steps, curves, selected, baseline = extract_curves(table)

    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=180)
    for arm in ARM_ORDER:
        ax.plot(
            steps,
            [100.0 * value for value in curves[arm]],
            color=ARM_COLORS[arm],
            marker=ARM_MARKERS[arm],
            markersize=4.8,
            linewidth=2.0,
            label=ARM_LABELS[arm],
        )
        selected_step = selected[arm]
        selected_index = steps.index(selected_step)
        selected_value = 100.0 * curves[arm][selected_index]
        ax.scatter(
            [selected_step],
            [selected_value],
            s=115,
            marker="*",
            color=ARM_COLORS[arm],
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )

    ax.axhline(
        100.0 * baseline,
        color="#555555",
        linestyle=(0, (3, 2)),
        linewidth=1.1,
        alpha=0.9,
        label="Unchanged shared start (step 0)",
    )
    ax.annotate(
        "selected current directional\n(step 0; unchanged start)",
        xy=(0, 100.0 * baseline),
        xytext=(950, 96.52),
        textcoords="data",
        fontsize=8.5,
        color="#333333",
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#bbbbbb", "alpha": 0.9},
    )

    ax.set_title("Domain-balanced validation learning curves", fontsize=14, pad=18)
    ax.text(
        0.5,
        1.015,
        "Public development / checkpoint selection · unchanged shared start reference",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    ax.set_xlabel("Training checkpoint (optimizer updates)")
    ax.set_ylabel("Domain-balanced validation token accuracy (%)")
    ax.set_xticks(steps)
    ax.set_ylim(95.5, 99.7)
    ax.set_xlim(-300, max(steps) + 300)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=8.5)
    ax.text(
        0.0,
        -0.19,
        "Star markers = registered selected checkpoints; curves are diagnostics-only and use opened public development validation.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#444444",
    )

    metadata = {
        "Title": "TRR-0010 public development checkpoint learning curves",
        "Description": f"Input SHA256: {input_sha256}; truth_opened={table['truth_opened']}; final_quality_available={table['final_quality_available']}",
        "Creator": "TRR-0010 public diagnostics plotting script",
    }
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, metadata=metadata, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, metadata={"Software": "matplotlib", "Description": metadata["Description"]}, bbox_inches="tight")
    plt.close(fig)

    return {
        "plot_source": {
            "path": str(source_path),
            "bytes": len(source_raw),
            "sha256": hashlib.sha256(source_raw).hexdigest(),
        },
        "input": {
            "path": str(table_path),
            "bytes": len(raw),
            "sha256": input_sha256,
            "schema": table["schema"],
            "task_id": table["task_id"],
            "truth_opened": table["truth_opened"],
            "final_quality_available": table["final_quality_available"],
        },
        "checkpoint_grid": steps,
        "arm_order": ARM_ORDER,
        "selected_steps": selected,
        "baseline": {
            "value_fraction": baseline,
            "value_percent": 100.0 * baseline,
            "label": "unchanged shared start (current directional step 0)",
        },
        "outputs": {
            "svg": {"path": str(output_svg), "bytes": output_svg.stat().st_size, "sha256": sha256(output_svg)},
            "png": {"path": str(output_png), "bytes": output_png.stat().st_size, "sha256": sha256(output_png)},
        },
        "limitations": [
            "Uses only registered domain-balanced validation accuracy from the public development diagnostics table.",
            "The table records truth_opened=false and final_quality_available=false; this figure is not a final truth-quality result.",
            "The figure shows checkpoint-selection diagnostics and does not add a new selection rule or re-evaluate any checkpoint.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = render(args.input, args.output_svg, args.output_png)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
