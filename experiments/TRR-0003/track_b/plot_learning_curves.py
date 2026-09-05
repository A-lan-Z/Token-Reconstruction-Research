#!/usr/bin/env python3
"""Make the compact Track B learning-curve figure and summary table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = (
    ("angular_inverse_control", "Angular inverse", "#1b6ca8"),
    ("tied_affine_token_ce", "Tied affine CE", "#c44e52"),
    ("residual_mlp256_token_ce", "Residual MLP-256 CE", "#2f855a"),
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # The source curve is logged at fixed 25-step points.  Earliest tie is
    # encoded in the negative step key so this remains deterministic.
    return dict(max(rows, key=lambda row: (float(row["public_validation_token_accuracy"]), -int(row["step"]))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/extended_fit_1800_v1/fit_evidence.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/figures_v1"),
    )
    args = parser.parse_args()
    evidence_path = args.evidence.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be empty/create-only: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    summary_methods: dict[str, Any] = {}
    for method, label, color in METHODS:
        rows = list(evidence["methods"][method]["training"]["learning_curve"])
        steps = [int(row["step"]) for row in rows]
        train = [float(row["train_token_accuracy"]) for row in rows]
        validation = [float(row["public_validation_token_accuracy"]) for row in rows]
        axis.plot(steps, train, color=color, linewidth=1.8, label=f"{label} train")
        axis.plot(
            steps,
            validation,
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{label} public validation",
        )
        overfit = evidence["methods"][method]["tiny_subset_overfit"]
        overfit_curve = overfit["training"]["learning_curve"]
        summary_methods[method] = {
            "best_public_validation": _best(rows),
            "final_main": rows[-1],
            "tiny_overfit_endpoint": overfit_curve[-1],
            "tiny_overfit_steps": overfit["steps"],
            "main_state": evidence["methods"][method]["artifact"],
            "tiny_state": overfit["artifact"],
        }
    axis.set_xlabel("optimizer steps")
    axis.set_ylabel("post-BOS token accuracy")
    axis.set_xlim(0, 1800)
    axis.set_ylim(0, 1.02)
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(ncol=2, fontsize=8, frameon=False)
    axis.set_title("TRR-0003 Track B public-fit learning curves")
    png_path = output_dir / "track_b_learning_curves.png"
    svg_path = output_dir / "track_b_learning_curves.svg"
    figure.savefig(png_path, dpi=220)
    figure.savefig(svg_path)
    plt.close(figure)

    prepare_path = Path("outputs/TRR-0003/track_b/public_fit_v2/prepare_evidence.json").resolve()
    qualification_path = Path("experiments/TRR-0003/track_b/qualification_affine_guard_v2.json").resolve()
    table = {
        "schema": "token-reconstruction.trr0003-track-b-curve-and-cost-summary.v1",
        "task_id": "TRR-0003",
        "track": "track_b",
        "curve_source": _record(evidence_path),
        "figure": {"png": _record(png_path), "svg": _record(svg_path)},
        "methods": summary_methods,
        "cost_sources": {
            "public_preparation": _record(prepare_path) if prepare_path.is_file() else None,
            "largest_cell_qualification": _record(qualification_path) if qualification_path.is_file() else None,
        },
        "notes": [
            "Solid lines are fit token accuracy; dashed lines are disjoint public auxiliary validation token accuracy.",
            "The tiny-overfit endpoint is a separate 8-record optimization/capacity diagnostic.",
            "No panel or evaluator-private truth is used by this plotting utility.",
        ],
    }
    summary_path = output_dir / "track_b_curve_and_cost_summary.json"
    summary_path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"png": str(png_path), "svg": str(svg_path), "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
