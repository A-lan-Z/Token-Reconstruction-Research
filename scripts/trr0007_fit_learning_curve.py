"""Render the compact TRR-0007 fit learning-curve figure.

The four completed fit JSONs are read as immutable evidence.  The figure has
one held-out-development panel and one initially-wrong challenge panel; the
selected checkpoint from each frozen fit is marked explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUTS = (
    ("current / trained diagonal", "experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_current_positionwise/learning_curve.json"),
    ("current / residual MLP-512", "experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_residual_mlp512/learning_curve.json"),
    ("improved / trained diagonal", "experiments/TRR-0007/improved_fit_v1/improved_public_bank/trr0007_current_positionwise/learning_curve.json"),
    ("improved / residual MLP-512", "experiments/TRR-0007/improved_fit_v1/improved_public_bank/trr0007_residual_mlp512/learning_curve.json"),
)
COLORS = ("#2166ac", "#67a9cf", "#b2182b", "#ef8a62")


def _load(root: Path, label: str, relative: str) -> dict:
    path = (root / relative).resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"{path}: missing learning-curve points")
    selected = payload.get("selected_step")
    if not isinstance(selected, int):
        raise ValueError(f"{path}: selected_step is missing")
    by_step = {int(point["step"]): point for point in points}
    if selected not in by_step:
        raise ValueError(f"{path}: selected step {selected} is absent")
    for point in points:
        if "validation" not in point or "style_balanced_token_accuracy" not in point["validation"]:
            raise ValueError(f"{path}: validation metric is missing")
        if "challenge_initially_wrong" not in point or "token_accuracy" not in point["challenge_initially_wrong"]:
            raise ValueError(f"{path}: challenge metric is missing")
    return {"label": label, "path": str(path), "payload": payload, "by_step": by_step}


def render(root: Path, output: Path) -> dict:
    curves = [_load(root, label, relative) for label, relative in DEFAULT_INPUTS]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True, sharey=True)
    for curve, color in zip(curves, COLORS):
        payload = curve["payload"]
        points = payload["points"]
        steps = [int(point["step"]) for point in points]
        dev = [float(point["validation"]["style_balanced_token_accuracy"]) for point in points]
        challenge = [float(point["challenge_initially_wrong"]["token_accuracy"]) for point in points]
        selected = int(payload["selected_step"])
        selected_point = curve["by_step"][selected]
        selected_dev = float(selected_point["validation"]["style_balanced_token_accuracy"])
        selected_challenge = float(selected_point["challenge_initially_wrong"]["token_accuracy"])
        axes[0].plot(steps, dev, color=color, lw=2.0, label=curve["label"])
        axes[0].scatter([selected], [selected_dev], color=color, s=52, marker="o", edgecolor="white", linewidth=0.8, zorder=4)
        axes[0].annotate(f"{selected}", (selected, selected_dev), xytext=(3, 5), textcoords="offset points", fontsize=8, color=color)
        axes[1].plot(steps, challenge, color=color, lw=2.0, label=curve["label"])
        axes[1].scatter([selected], [selected_challenge], color=color, s=52, marker="o", edgecolor="white", linewidth=0.8, zorder=4)
        axes[1].annotate(f"{selected}", (selected, selected_challenge), xytext=(3, 5), textcoords="offset points", fontsize=8, color=color)
    axes[0].set_title("Held-out development")
    axes[1].set_title("Initially-wrong challenge")
    for axis in axes:
        axis.set_xlabel("Fit step")
        axis.grid(True, alpha=0.25, linewidth=0.7)
        axis.set_xlim(left=0)
        axis.set_ylim(-0.02, 1.04)
    axes[0].set_ylabel("Token accuracy")
    axes[1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle("TRR-0007 frozen fit learning curves", fontsize=13)
    fig.text(0.5, 0.01, "Markers and labels identify the selected validation checkpoint; figure is descriptive preselection evidence.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, metadata={"Title": "TRR-0007 frozen fit learning curves", "Software": "matplotlib"})
    plt.close(fig)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "inputs": [{"label": curve["label"], "path": curve["path"], "selected_step": curve["payload"]["selected_step"]} for curve in curves],
        "panels": ["held_out_development_style_balanced_token_accuracy", "initially_wrong_challenge_token_accuracy"],
        "selected_marker": "selected_step from each learning_curve.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("experiments/TRR-0007/fit_learning_curves.png"))
    args = parser.parse_args()
    print(json.dumps(render(args.repository_root.resolve(), args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
