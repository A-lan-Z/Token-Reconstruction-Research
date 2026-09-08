#!/usr/bin/env python3
"""Aggregate the immutable A1 proposal trace after the registered truth gate.

This reads only persisted candidate arrays, frozen predictions, and the
post-freeze truth sidecar.  It never invokes a model or emits token/source
identifiers.  The output is intentionally an aggregate diagnostic receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open


CELLS = (
    "finance__public_base",
    "finance__public_lora_2601",
    "pile__public_base",
    "pile__public_lora_2601",
)
BINS = (
    ("rank_1", 1, 1),
    ("rank_2_16", 2, 16),
    ("rank_17_64", 17, 64),
    ("rank_65_128", 65, 128),
    ("rank_129_256", 129, 256),
    ("rank_257_512", 257, 512),
    ("not_in_512", None, None),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def one_cell(root: Path, cell: str) -> dict[str, object]:
    trace_path = root / "outputs/TRR-P11/private-evaluation/a1_a2-execution-r3/traces" / f"{cell}.safetensors"
    prediction_path = root / "outputs/TRR-P11/private-evaluation/a1_a2-execution-r3/predictions" / f"{cell}.safetensors"
    truth_path = root / "outputs/TRR-P11/private-evaluation/evaluation/truth-r1/truth_tokens.safetensors"

    with (
        safe_open(str(trace_path), framework="pt", device="cpu") as trace,
        safe_open(str(truth_path), framework="pt", device="cpu") as truth_file,
        safe_open(str(prediction_path), framework="pt", device="cpu") as prediction_file,
    ):
        candidates = trace.get_tensor("candidates")
        candidate_scores = trace.get_tensor("candidate_scores")
        truth = truth_file.get_tensor(f"{cell}__token_ids")
        prediction = prediction_file.get_tensor("predictions")

        assert tuple(candidates.shape) == (128, 128, 512)
        assert tuple(candidate_scores.shape) == (128, 128, 512)
        assert tuple(truth.shape) == (256, 128)
        assert tuple(prediction.shape) == (128, 128)

        result: dict[str, object] = {
            "records": 128,
            "scored_positions_per_record": 127,
            "scored_tokens": 128 * 127,
            "proposal_budget": 512,
            "executed_budget": 256,
            "proposal_hits": 0,
            "executed_top256_hits": 0,
            "proposal_only_hits_257_512": 0,
            "proposal_misses": 0,
            "final_correct": 0,
            "final_wrong": 0,
            "final_wrong_with_true_in_executed_top256": 0,
            "final_wrong_with_true_only_in_proposal_512": 0,
            "final_wrong_with_true_outside_proposal_512": 0,
            "proposal_rank_sum": 0,
            "proposal_rank_count": 0,
            "rank_bin_counts": {name: 0 for name, _, _ in BINS},
        }
        unique = True
        for record in range(128):
            for position in range(1, 128):
                row = candidates[record, position]
                unique = unique and int(torch.unique(row).numel()) == 512
                target = int(truth[record, position].item())
                matches = (row == target).nonzero(as_tuple=False).flatten()
                rank = int(matches[0].item()) + 1 if matches.numel() else None
                if rank is None:
                    result["proposal_misses"] += 1
                    result["rank_bin_counts"]["not_in_512"] += 1
                else:
                    result["proposal_hits"] += 1
                    result["proposal_rank_sum"] += rank
                    result["proposal_rank_count"] += 1
                    if rank <= 256:
                        result["executed_top256_hits"] += 1
                    else:
                        result["proposal_only_hits_257_512"] += 1
                    for name, lower, upper in BINS[:-1]:
                        if lower <= rank <= upper:
                            result["rank_bin_counts"][name] += 1
                            break

                if int(prediction[record, position].item()) == target:
                    result["final_correct"] += 1
                else:
                    result["final_wrong"] += 1
                    if rank is None:
                        result["final_wrong_with_true_outside_proposal_512"] += 1
                    elif rank <= 256:
                        result["final_wrong_with_true_in_executed_top256"] += 1
                    else:
                        result["final_wrong_with_true_only_in_proposal_512"] += 1

        scored_tokens = result["scored_tokens"]
        result["proposal_hit_rate"] = result["proposal_hits"] / scored_tokens
        result["executed_top256_hit_rate"] = result["executed_top256_hits"] / scored_tokens
        result["proposal_only_hit_rate"] = result["proposal_only_hits_257_512"] / scored_tokens
        result["proposal_miss_rate"] = result["proposal_misses"] / scored_tokens
        result["mean_proposal_rank"] = result["proposal_rank_sum"] / result["proposal_rank_count"]
        result["trace_contract_checks"] = {
            "bos_candidates_all_minus_one": bool((candidates[:, 0, :] == -1).all().item()),
            "bos_scores_all_negative_infinity": bool((candidate_scores[:, 0, :] == float("-inf")).all().item()),
            "scored_scores_all_finite": bool(torch.isfinite(candidate_scores[:, 1:, :]).all().item()),
            "scored_candidate_rows_unique": unique,
            "scored_scores_nonincreasing": bool((candidate_scores[:, 1:, :-1] >= candidate_scores[:, 1:, 1:]).all().item()),
        }
        result["trace_artifact"] = artifact(root, trace_path)
        result["prediction_artifact"] = artifact(root, prediction_path)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    truth_path = root / "outputs/TRR-P11/private-evaluation/evaluation/truth-r1/truth_tokens.safetensors"
    freeze_path = root / "outputs/TRR-P11/private-evaluation/evaluation/freeze-r1.json"

    cells = {cell: one_cell(root, cell) for cell in CELLS}
    aggregate = {
        key: sum(int(c[key]) for c in cells.values())
        for key in (
            "proposal_hits", "executed_top256_hits", "proposal_only_hits_257_512",
            "proposal_misses", "final_correct", "final_wrong",
            "final_wrong_with_true_in_executed_top256",
            "final_wrong_with_true_only_in_proposal_512",
            "final_wrong_with_true_outside_proposal_512", "proposal_rank_sum", "proposal_rank_count",
        )
    }
    aggregate.update({
        "records": 512,
        "scored_positions_per_record": 127,
        "scored_tokens": 512 * 127,
        "rank_bin_counts": {
            name: sum(int(c["rank_bin_counts"][name]) for c in cells.values())
            for name, _, _ in BINS
        },
    })
    aggregate["proposal_hit_rate"] = aggregate["proposal_hits"] / aggregate["scored_tokens"]
    aggregate["executed_top256_hit_rate"] = aggregate["executed_top256_hits"] / aggregate["scored_tokens"]
    aggregate["proposal_only_hit_rate"] = aggregate["proposal_only_hits_257_512"] / aggregate["scored_tokens"]
    aggregate["proposal_miss_rate"] = aggregate["proposal_misses"] / aggregate["scored_tokens"]
    aggregate["mean_proposal_rank"] = aggregate["proposal_rank_sum"] / aggregate["proposal_rank_count"]

    output = {
        "task_id": "TRR-P11",
        "schema": "token-reconstruction.trr-p11-a1-proposal-diagnostics.v1",
        "status": "COMPUTED_FROM_FROZEN_A1_TRACE_AND_POSTFREEZE_TRUTH",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "truth_boundary": {
            "predictions_freeze": "outputs/TRR-P11/private-evaluation/evaluation/freeze-r1.json",
            "freeze_sha256": sha256(freeze_path),
            "truth_opened_after_freeze": True,
            "source_text_or_target_labels_used_for_selection": False,
            "p03_holdout_accessed": False,
        },
        "definition": {
            "proposal": "persisted native A1 candidate list of 512 IDs ordered by direct-cosine proposal scores",
            "executed": "persisted first 256 candidates passed to the A2 decoder policy",
            "post_bos_positions": "positions 1 through 127; BOS position 0 is excluded",
            "proposal_rank": "one-based position in the persisted ordered 512-candidate list",
            "conditional_error": "final output differs while the truth token is available in the executed 256-candidate input; no mechanism attribution is made",
            "proposal_only_error": "final output differs while truth is in proposal ranks 257-512 and therefore omitted from executed top256",
            "proposal_miss": "truth is absent from the persisted 512-candidate proposal",
        },
        "a1_proposal": {"status": "COMPUTED", "trace_shape": [128, 128, 512], "cells": cells, "aggregate": aggregate},
        "a2_decoder_ranking": {
            "status": "UNAVAILABLE",
            "reason": "Persisted scores are A1 direct-cosine proposal scores; no causal decoder per-candidate score was persisted.",
        },
        "artifacts": {
            "truth_sidecar": artifact(root, truth_path),
            "freeze": artifact(root, freeze_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "bytes": args.output.stat().st_size}, sort_keys=True))


if __name__ == "__main__":
    main()
