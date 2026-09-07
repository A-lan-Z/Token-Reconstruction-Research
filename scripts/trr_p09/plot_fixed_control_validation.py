#!/usr/bin/env python3
"""Plot validation-only curves from completed fixed-control receipts.

This reads receipt JSON metadata only. It does not load model, embedding,
activation-bank, target, or evaluation-truth payloads and does not recompute
scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _curve(receipt_path: Path, bank: str) -> dict:
    receipt = json.loads(receipt_path.read_text())
    rows = []
    for item in receipt["learning_curve"]:
        validation = item["validation"]
        domains = validation["domains"]
        rows.append(
            {
                "step": item["step"],
                "Finance_token_accuracy": domains["Finance"]["token_accuracy"],
                "Pile_token_accuracy": domains["Pile"]["token_accuracy"],
                "domain_balanced_token_accuracy": validation[
                    "domain_balanced_token_accuracy"
                ],
                "validation_token_rows": validation["token_rows"],
                "Finance_token_rows": domains["Finance"]["token_rows"],
                "Pile_token_rows": domains["Pile"]["token_rows"],
            }
        )
    return {
        "bank": bank,
        "receipt": {
            "path": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256(receipt_path),
        },
        "status": receipt["status"],
        "selection": receipt["selection"],
        "validation_only": True,
        "truth_opened": False,
        "target_observation_opened": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-receipt", type=Path, required=True)
    parser.add_argument("--b1-receipt", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--png-out", type=Path, required=True)
    args = parser.parse_args()

    b0 = _curve(args.b0_receipt, "B0")
    b1 = _curve(args.b1_receipt, "B1")
    payload = {
        "schema": "token-reconstruction.trr-p09-fixed-control-validation-curves.v1",
        "task_id": "TRR-P09",
        "status": "PASS_RECEIPT_DERIVED_VALIDATION_ONLY",
        "description": "Three-panel public-validation learning curves; no final-panel or evaluation-truth data.",
        "validation_only": True,
        "truth_opened": False,
        "target_observation_opened": False,
        "panels": [
            "Finance token accuracy",
            "Pile token accuracy",
            "equal-domain mean (domain_balanced_token_accuracy)",
        ],
        "curves": [b0, b1],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")

    # Import plotting only after receipt parsing, keeping the metadata path
    # independently useful on systems without a graphical backend.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"B0": {"color": "#1f77b4", "marker": "o"}, "B1": {"color": "#d62728", "marker": "s"}}
    panels = [
        ("Finance", "Finance token accuracy", "Finance_token_accuracy"),
        ("Pile", "Pile token accuracy", "Pile_token_accuracy"),
        ("Equal-domain mean", "Equal-domain mean accuracy", "domain_balanced_token_accuracy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    for ax, (_, title, key) in zip(axes, panels):
        for curve in (b0, b1):
            xs = [row["step"] for row in curve["rows"]]
            ys = [row[key] for row in curve["rows"]]
            style = styles[curve["bank"]]
            selected = curve["selection"]["selected_step"]
            ax.plot(xs, ys, label=f"{curve['bank']} (selected {selected})", linewidth=1.6, markersize=3.5, **style)
            idx = xs.index(selected)
            ax.scatter([xs[idx]], [ys[idx]], color=style["color"], s=28, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("optimizer step")
        ax.grid(True, alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("token accuracy")
    axes[0].set_ylim(0.94, 1.002)
    axes[0].legend(fontsize=7, loc="lower right", frameon=True)
    fig.suptitle("TRR-P09 frozen controls: public validation only", fontsize=11)
    fig.text(0.5, 0.01, "Validation curves from completed receipts; not final-panel performance.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    args.png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png_out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
