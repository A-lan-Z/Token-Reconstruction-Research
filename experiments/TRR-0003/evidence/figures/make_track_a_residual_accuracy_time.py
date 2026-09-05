from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
ARCHIVE_ROOT = ROOT / "experiments/TRR-0003/evidence/comparison_sources_v1"
SCORE_PATH = ROOT / "experiments/TRR-0003/evidence/common_score_v2.json"
OUT_DIR = Path(__file__).resolve().parent
ITERATIONS = [0, 1, 2, 4, 8, 16, 32]
CELLS = [
    ("pile", "public_base", "Pile / base"),
    ("pile", "public_lora_2601", "Pile / shifted"),
    ("finance", "public_base", "Finance / base"),
    ("finance", "public_lora_2601", "Finance / shifted"),
]
METHOD_PREFIX = "checkpoint_reverse_fixed_point_euclidean_k16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_path(style: str, condition: str) -> Path:
    return (
        ARCHIVE_ROOT
        / "outputs/TRR-0003/track_a_diagnostics"
        / f"{style}_{condition}"
        / style
        / condition
        / METHOD_PREFIX
        / "evidence.json"
    )


def main() -> None:
    score = json.loads(SCORE_PATH.read_text())
    data = {
        "schema": "token-reconstruction.trr0003-track-a-residual-accuracy-time.v1",
        "task_id": "TRR-0003",
        "iterations": ITERATIONS,
        "score_source": {"path": str(SCORE_PATH.relative_to(ROOT)), "sha256": sha256(SCORE_PATH)},
        "timing_sources": {},
        "cells": {},
    }
    for style, condition, label in CELLS:
        raw_path = evidence_path(style, condition)
        raw = json.loads(raw_path.read_text())
        residual = []
        timings = []
        accuracy = []
        for iteration, item in zip(ITERATIONS, raw["iterations"]):
            aggregate = item["aggregate"]
            method_id = f"{METHOD_PREFIX}_i{iteration:03d}"
            score_cell = score["cells"][f"{style}__{condition}__{method_id}"]
            residual.append(float(aggregate["mean_continuous_cycle_relative_l2_scored"]))
            timings.append(float(aggregate["inference_seconds"]))
            accuracy.append(float(score_cell["metrics"]["token_accuracy"]))
        key = f"{style}__{condition}"
        data["timing_sources"][key] = {
            "path": str(raw_path.relative_to(ROOT)),
            "sha256": sha256(raw_path),
        }
        data["cells"][key] = {
            "label": label,
            "continuous_cycle_relative_l2": residual,
            "token_accuracy": accuracy,
            "inference_seconds": timings,
        }

    json_path = OUT_DIR / "track_a_residual_accuracy_time.json"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    x = list(range(len(ITERATIONS)))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), dpi=180)
    for (key, cell), color in zip(data["cells"].items(), colors):
        label = cell["label"]
        axes[0].plot(x, cell["continuous_cycle_relative_l2"], marker="o", linewidth=1.8, color=color, label=label)
        axes[1].plot(x, [100.0 * value for value in cell["token_accuracy"]], marker="o", linewidth=1.8, color=color, label=label)
        axes[2].plot(x, cell["inference_seconds"], marker="o", linewidth=1.8, color=color, label=label)
    axes[0].set_title("Cycle residual")
    axes[0].set_ylabel("mean relative L2")
    axes[1].set_title("Token recovery")
    axes[1].set_ylabel("post-BOS accuracy (%)")
    axes[2].set_title("Cell inference time")
    axes[2].set_ylabel("seconds")
    for axis in axes:
        axis.set_xlabel("fixed-point iterations")
        axis.set_xticks(x, [str(value) for value in ITERATIONS])
        axis.grid(True, alpha=0.25)
    axes[1].set_ylim(bottom=0)
    fig.suptitle("Track A: fixed-point residual, token correctness, and cost", y=1.04)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.72, 1.23), ncol=2, frameon=False)
    fig.text(0.5, -0.04, "Residual and timing are truth-free; accuracy comes from the frozen retrospective development-panel receipt.", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "track_a_residual_accuracy_time.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "track_a_residual_accuracy_time.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
