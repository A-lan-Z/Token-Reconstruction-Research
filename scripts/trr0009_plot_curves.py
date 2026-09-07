#!/usr/bin/env python3
"""Plot truth-free TRR-0009 continuation training diagnostics.

The figure reads only the task's learning-curve receipts.  It is a diagnostic
of the public fitting bank and the fixed 48-record development selection set;
it is not a fresh natural evaluation or an independent test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK_ID = "TRR-0009"
CURVE_SCHEMA = "token-reconstruction.trr0009-learning-curve.v1"
ARMS = (
    ("unchanged_anchor", "Unchanged anchor", "#4c566a"),
    ("continued_fixed_readout", "Continued fixed readout", "#2e8b57"),
    ("continued_adaptable_readout", "Continued adaptable readout", "#c44e52"),
)


class CurveError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CurveError(f"invalid learning-curve JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("task_id") != TASK_ID or value.get("schema") != CURVE_SCHEMA:
        raise CurveError(f"learning-curve identity changed: {path}")
    if not isinstance(value.get("points"), list) or not value["points"]:
        raise CurveError(f"learning-curve points are absent: {path}")
    return value


def _number(mapping: dict[str, Any], key: str, description: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise CurveError(f"{description} is malformed") from exc
    return value


def _series(curve: dict[str, Any]) -> dict[str, list[float]]:
    steps: list[float] = []
    fit_errors: list[float] = []
    validation_accuracy: list[float] = []
    challenge_recovered: list[float] = []
    challenge_total: int | None = None
    for point in curve["points"]:
        if not isinstance(point, dict):
            raise CurveError("learning-curve point is malformed")
        fit = point.get("fit")
        validation = point.get("validation")
        challenge = point.get("challenge_initially_wrong")
        if not isinstance(fit, dict) or not isinstance(validation, dict) or not isinstance(challenge, dict):
            raise CurveError("learning-curve point lacks fit, validation, or challenge metrics")
        step = _number(point, "step", "learning-curve step")
        fit_rows = _number(fit, "token_rows", "fit token rows")
        fit_correct = _number(fit, "correct_tokens", "fit correct tokens")
        validation_accuracy.append(_number(validation, "style_balanced_token_accuracy", "validation style-balanced accuracy"))
        total = int(_number(challenge, "token_rows", "challenge rows"))
        if challenge_total is None:
            challenge_total = total
        elif challenge_total != total:
            raise CurveError("challenge denominator changed within a curve")
        steps.append(step)
        fit_errors.append(fit_rows - fit_correct)
        challenge_recovered.append(_number(challenge, "correct_tokens", "challenge recovered tokens"))
    if challenge_total is None or challenge_total <= 0:
        raise CurveError("challenge denominator is absent")
    return {
        "steps": steps,
        "fit_errors": fit_errors,
        "validation_accuracy": validation_accuracy,
        "challenge_recovered": challenge_recovered,
        "challenge_total": [float(challenge_total)],
        "selected_step": [float(curve.get("selected_step", -1))],
    }


def plot_curves(run_root: Path, output: Path, svg_output: Path | None) -> dict[str, Any]:
    curves: dict[str, dict[str, Any]] = {}
    for arm, _label, _color in ARMS:
        curves[arm] = _load(run_root / arm / "learning_curve.json")
    series = {arm: _series(curves[arm]) for arm, _label, _color in ARMS}
    baseline = series["unchanged_anchor"]["validation_accuracy"][0]

    # Reserve a dedicated footer: the legend and scope note must not compete
    # with x-axis labels when the figure is rendered tightly.
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.25, wspace=0.28)
    fig.suptitle("TRR-0009 continuation training diagnostics", fontsize=14, fontweight="bold")
    for arm, label, color in ARMS:
        values = series[arm]
        axes[0].plot(values["steps"], values["fit_errors"], marker="o", markersize=3.2, linewidth=1.6, label=label, color=color)
        axes[1].plot(values["steps"], values["validation_accuracy"], marker="o", markersize=3.2, linewidth=1.6, label=label, color=color)
        axes[2].plot(values["steps"], values["challenge_recovered"], marker="o", markersize=3.2, linewidth=1.6, label=label, color=color)
        selected_step = values["selected_step"][0]
        selected_indices = [i for i, step in enumerate(values["steps"]) if step == selected_step]
        if selected_indices:
            i = selected_indices[0]
            axes[0].scatter([values["steps"][i]], [values["fit_errors"][i]], s=75, facecolors="white", edgecolors=color, linewidths=1.7, zorder=5)
            axes[1].scatter([values["steps"][i]], [values["validation_accuracy"][i]], s=75, facecolors="white", edgecolors=color, linewidths=1.7, zorder=5)
            axes[2].scatter([values["steps"][i]], [values["challenge_recovered"][i]], s=75, facecolors="white", edgecolors=color, linewidths=1.7, zorder=5)

    axes[0].set_title("Fit-bank errors")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("incorrect post-BOS positions")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Selection validation")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("style-balanced token accuracy")
    axes[1].axhline(baseline, color="#4c566a", linestyle="--", linewidth=1.0, alpha=0.8, label="unchanged baseline")
    axes[1].set_ylim(max(0.0, baseline - 0.02), 1.002)
    axes[2].set_title("621-row challenge recovery")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("initially wrong rows recovered")
    axes[2].set_ylim(0, 621)
    for axis in axes:
        axis.grid(True, alpha=0.25, linewidth=0.7)
        axis.set_xlim(left=0)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.105))
    fig.text(0.5, 0.025, "Public fitting-bank and fixed 48-record development diagnostics; selected-step rings; not a fresh natural evaluation or independent test.", ha="center", va="bottom", fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    if svg_output is not None:
        svg_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg_output, bbox_inches="tight")
    plt.close(fig)
    return {
        "task_id": TASK_ID,
        "source_scope": "learning_curve receipts only; public fitting bank and fixed development diagnostics",
        "independent_test": False,
        "selected_steps": {arm: int(series[arm]["selected_step"][0]) for arm, _label, _color in ARMS},
        "challenge_denominator": int(series["continued_fixed_readout"]["challenge_total"][0]),
        "outputs": [str(output)] + ([str(svg_output)] if svg_output is not None else []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/TRR-0009/training/run_v1"))
    parser.add_argument("--output", type=Path, default=Path("experiments/TRR-0009/training/diagnostic_curves_v1.png"))
    parser.add_argument("--svg-output", type=Path, default=Path("experiments/TRR-0009/training/diagnostic_curves_v1.svg"))
    args = parser.parse_args()
    result = plot_curves(args.run_root, args.output, args.svg_output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
