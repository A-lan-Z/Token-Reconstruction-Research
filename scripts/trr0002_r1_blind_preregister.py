#!/usr/bin/env python3
"""Preregister the exact frozen-winner blind run before observations exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from token_reconstruction.a1a2_configuration_search import resolved_policy_from_dict
from token_reconstruction.experiment_runtime import (
    command_record,
    file_record,
    load_json,
    utc_now,
    write_json_exclusive,
)


METHOD_ID = "a1_a2_exhaustive_configuration_winner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--public-commitment", type=Path, required=True)
    parser.add_argument("--target-lora", type=Path, required=True)
    parser.add_argument("--public-lens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("blind preregistration is create-only")
    root = args.repository_root.resolve(strict=True)
    winner = load_json(args.winner)
    if (
        winner.get("schema") != "token-reconstruction.trr0002-owner-r1-frozen-winner.v1"
        or winner.get("status")
        != "FROZEN_BEFORE_PUBLIC_HELDOUT_FRESH_BLIND_OR_CANONICAL_ACCESS"
    ):
        raise RuntimeError("frozen winner identity changed")
    policy = resolved_policy_from_dict(winner["policy"])
    if winner.get("policy_id") != policy.policy_id:
        raise RuntimeError("winner policy ID changed")
    commitment = load_json(args.public_commitment)
    if (
        commitment.get("schema")
        != "token-reconstruction.trr0002-owner-r1-selection-commitment.v1"
        or commitment.get("record_count") != 64
        or commitment.get("source_identity_disclosed") is not False
        or commitment.get("selection_key_disclosed") is not False
    ):
        raise RuntimeError("fresh blind public commitment changed")
    execution_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    payload = {
        "schema": "token-reconstruction.trr0002-owner-r1-blind-preregistration.v1",
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R1",
        "status": "COMMITTED_BEFORE_OWNER_R1_FRESH_BLIND_OBSERVATIONS",
        "created_utc": utc_now(),
        "truth_opened": False,
        "method_id": METHOD_ID,
        "policy_id": policy.policy_id,
        "policy": policy.serialized(),
        "execution_parent": execution_commit,
        "selection": file_record(args.public_commitment),
        "winner": file_record(args.winner),
        "retained_state": {
            "target_lora": file_record(args.target_lora),
            "public_lens": file_record(args.public_lens),
            "fresh_training_steps": 0,
            "fresh_adaptation_steps": 0,
        },
        "setup": {
            "id": "fresh-clean-pile-lora-r2-64x40",
            "records": 64,
            "positions": 40,
            "known_prefix_tokens": 1,
            "scored_tokens_per_record": 39,
            "cut_depth": 4,
            "condition": "same unavailable rank-4 LoRA state as the earlier fresh run; wholly new disjoint Pile records",
        },
        "execution": {
            "seed": 20260826,
            "record_batch_size": 8,
            "policy_source": "src/token_reconstruction/a1a2_configuration_search.py",
            "proposal_source": "src/token_reconstruction/component_crossover.py",
            "reference_source": "reference/strict_bos/round001_teacher.py",
            "abstention": policy.spec.terminal_action == "abstain_and_stop_suffix",
            "target_prefix_calls": 0,
        },
        "access_contract": {
            "reconstructor_inputs": "sanitized observations, attention/position geometry, public lens, exact frozen winner, public model and frozen code only",
            "prohibited": "workspace, dataset, private selection, truth, unavailable target update, historical source, canonical truth, and network",
            "prediction_freeze_before_truth": True,
            "no_policy_revision_after_truth": True,
        },
        "command": command_record(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(args.output, payload)
    print(json.dumps({"status": payload["status"], "policy_id": policy.policy_id, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
