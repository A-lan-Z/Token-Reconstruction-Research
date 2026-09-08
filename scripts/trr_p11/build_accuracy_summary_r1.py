"""Private additive descriptive accuracy summary for the frozen TRR-P11 score."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from scripts.trr_p11 import evaluation as ev

SEED = ev.BOOTSTRAP_SEED
DRAWS = ev.BOOTSTRAP_DRAWS
ALPHA = ev.BOOTSTRAP_ALPHA


def file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(".").resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite {output}")

    score_path = args.score if args.score.is_absolute() else root / args.score
    score_payload = json.loads(score_path.read_text(encoding="utf-8"))
    if score_payload.get("status") != "SCORE_COMPLETE_AFTER_TRUTH":
        raise RuntimeError("accuracy summary requires the complete post-truth score")
    frozen = ev._load_frozen(args.freeze, root=root)
    truth, truth_binding = ev.load_truth_after_freeze(
        freeze_path=args.freeze,
        truth_descriptor_path=args.truth,
        repository_root=root,
    )
    scorer = ev._load_scorer(root)

    method_specs = (
        ("new_current_fixed_B0", "full256", None),
        ("new_expanded_fixed_B1", "full256", None),
        ("a1_a2_k256", "first128", tuple(range(ev.COMPARATOR_RECORDS_PER_DOMAIN))),
    )
    cells: dict[str, Any] = {}
    for cell in ev.CELL_ORDER:
        methods: dict[str, Any] = {}
        target = truth[cell]
        for method, coverage, expected_indices in method_specs:
            key = f"{method}::{cell}"
            prediction = frozen.predictions[key]
            indices = frozen.subsets[key]
            if expected_indices is None:
                if indices is not None:
                    raise RuntimeError(f"unexpected subset for {key}")
                target_method = target
            else:
                if indices != expected_indices:
                    raise RuntimeError(f"A1 subset changed for {key}")
                target_method = target.index_select(0, torch.tensor(indices, dtype=torch.long))
            correct = prediction[:, 1:].eq(target_method[:, 1:])
            record_correct = correct.sum(dim=1).tolist()
            accuracies = [float(value) / float(ev.SCORED_POST_BOS_TOKENS) for value in record_correct]
            interval = scorer.bootstrap_token_delta(
                accuracies,
                seed=SEED,
                draws=DRAWS,
                one_sided_alpha=ALPHA,
            )
            methods[method] = {
                "coverage": coverage,
                "records": int(correct.shape[0]),
                "scored_post_bos_tokens_per_record": ev.SCORED_POST_BOS_TOKENS,
                "scored_tokens": int(correct.numel()),
                "correct_tokens": int(correct.sum().item()),
                "token_errors": int((~correct).sum().item()),
                "exact_correct_records": int(correct.all(dim=1).sum().item()),
                "exact_error_records": int((~correct.all(dim=1)).sum().item()),
                "observed_token_accuracy": float(correct.sum().item()) / float(correct.numel()),
                "source_record_accuracy_interval": interval,
                "interval_interpretation": "bootstrap of per-record correct-token fraction against zero",
            }
        cells[cell] = {"methods": methods}
    payload = {
        "schema": "token-reconstruction.trr-p11-descriptive-token-accuracy.v1",
        "task_id": ev.TASK_ID,
        "status": "DESCRIPTIVE_ACCURACY_INTERVALS_COMPLETE_AFTER_TRUTH",
        "score": file_record(score_path),
        "freeze": file_record(args.freeze if args.freeze.is_absolute() else root / args.freeze),
        "truth": dict(truth_binding),
        "cells": cells,
        "method_order": [item[0] for item in method_specs],
        "cell_order": list(ev.CELL_ORDER),
        "pooling": False,
        "scorer": {
            "path": ev.SCORER_RELATIVE_PATH,
            "source_commit": ev.SCORER_SOURCE_COMMIT,
            "sha256": ev.SCORER_SHA256,
            "bootstrap_seed": SEED,
            "bootstrap_draws": DRAWS,
            "one_sided_alpha": ALPHA,
            "interval_level": 0.95,
            "primitive": "bootstrap_token_delta",
        },
        "truth_opened": True,
        "p03_holdout_accessed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": payload["status"], "path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
