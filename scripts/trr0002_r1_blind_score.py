#!/usr/bin/env python3
"""Verify the owner-R1 freeze, reveal private truth, and score once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
import torch

from token_reconstruction.blind_commitment import commitment_digest
from token_reconstruction.experiment_runtime import (
    command_record,
    file_record,
    load_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from token_reconstruction.metrics import bootstrap_mean


METHOD_ID = "a1_a2_exhaustive_configuration_winner"
PUBLIC_SCHEMA = "token-reconstruction.trr0002-owner-r1-selection-commitment.v1"
PRIVATE_SCHEMA = "token-reconstruction.trr0002-owner-r1-private-selection.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--public-commitment", type=Path, required=True)
    parser.add_argument("--private-selection", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def verify_tree(root: Path, records: list[dict[str, Any]]) -> None:
    resolved = root.resolve(strict=True)
    actual_paths = {
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file()
    }
    expected_paths = {str(row["path"]) for row in records}
    if actual_paths != expected_paths:
        raise RuntimeError("freeze tree file set changed")
    for row in records:
        path = resolved / str(row["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("freeze tree contains a missing file or symlink")
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen file changed: {path}")


def main() -> int:
    args = parse_args()
    if args.result.exists() or args.result.is_symlink():
        raise RuntimeError("blind score result is create-only")
    receipt = load_json(args.receipt)
    if (
        receipt.get("schema")
        != "token-reconstruction.trr0002-owner-r1-blind-freeze-receipt.v1"
        or receipt.get("status")
        != "OWNER_R1_FROZEN_AND_VERIFIED_BEFORE_TRUTH_OPEN"
        or receipt.get("method_id") != METHOD_ID
        or receipt.get("truth_opened") is not False
        or receipt.get("truth_or_private_selection_arguments") != 0
    ):
        raise RuntimeError("blind freeze receipt identity changed")
    verify_tree(args.input_root, receipt["input_files"])
    verify_tree(args.output_root, receipt["output_files"])
    verify_tree(args.code_root, receipt["code_files"])

    # Private state is first read only after all frozen trees pass verification.
    public = load_json(args.public_commitment)
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or public.get("task_id") != "TRR-0002"
        or public.get("record_count") != 64
        or public.get("source_identity_disclosed") is not False
        or public.get("selection_key_disclosed") is not False
    ):
        raise RuntimeError("owner-R1 public commitment changed")
    private = load_json(args.private_selection)
    if private.get("schema") != PRIVATE_SCHEMA or private.get("task_id") != "TRR-0002":
        raise RuntimeError("private selection schema changed")
    try:
        key = bytes.fromhex(str(private["selection_key_hex"]))
    except ValueError as exc:
        raise RuntimeError("private selection key is invalid") from exc
    private_records = private.get("records")
    expected_ids = [f"blind-r2-{position:06d}" for position in range(1, 65)]
    if len(key) != 32 or not isinstance(private_records, list) or len(private_records) != 64:
        raise RuntimeError("private selection geometry changed")
    if [row["record_id"] for row in private_records] != expected_ids:
        raise RuntimeError("owner-R1 opaque record order changed")
    if commitment_digest(key, private_records) != public["commitment"]:
        raise RuntimeError("private selection does not open the public commitment")
    truth_rows = read_jsonl(args.truth)
    if len(truth_rows) != 64:
        raise RuntimeError("fresh blind truth geometry changed")
    if [row["record_id"] for row in truth_rows] != expected_ids:
        raise RuntimeError("fresh truth order differs from private commitment")
    if [row["token_ids"] for row in truth_rows] != [row["token_ids"] for row in private_records]:
        raise RuntimeError("fresh truth tokens differ from private commitment")
    truth = torch.tensor([row["token_ids"] for row in truth_rows], dtype=torch.long)
    prediction_path = args.output_root / "predictions.safetensors"
    state = load_file(prediction_path, device="cpu")
    expected_keys = {
        "predictions",
        "candidates",
        "proposal_top1_confidence",
        "routes",
        "selected_k",
        "selected_signal",
    }
    if set(state) != expected_keys:
        raise RuntimeError("blind prediction artifact fields changed")
    predictions = state["predictions"].to(torch.long)
    candidates = state["candidates"].to(torch.long)
    routes = state["routes"].to(torch.long)
    selected_k = state["selected_k"].to(torch.long)
    if predictions.shape != truth.shape or candidates.shape[:2] != (64, 40):
        raise RuntimeError("blind prediction geometry changed")
    expected = truth[:, 1:]
    predicted = predictions[:, 1:]
    correct = predicted.eq(expected)
    included = candidates[:, 1:].eq(expected[:, :, None]).any(dim=2)
    covered = predicted.ge(0)
    correct_tokens = int(correct.sum().item())
    scored_tokens = int(correct.numel())
    covered_tokens = int(covered.sum().item())
    included_tokens = int(included.sum().item())
    per_record_accuracy = correct.float().mean(dim=1).tolist()
    first_error_positions: list[int | None] = []
    for row in correct:
        failures = torch.nonzero(~row, as_tuple=False)
        first_error_positions.append(None if failures.numel() == 0 else int(failures[0].item()) + 1)
    evidence = load_json(args.output_root / "reconstructor_evidence.json")
    route = load_json(args.output_root / "route.json")
    if evidence.get("policy_id") != receipt.get("policy_id") or route.get("policy_id") != receipt.get("policy_id"):
        raise RuntimeError("frozen policy identity changed at scoring")
    result = {
        "schema": "token-reconstruction.trr0002-owner-r1-fresh-blind-result.v1",
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R1",
        "status": "OWNER_R1_FRESH_BLIND_SCORED_AFTER_VERIFIED_FREEZE",
        "scored_utc": utc_now(),
        "command": command_record(),
        "method_id": METHOD_ID,
        "policy_id": receipt["policy_id"],
        "setup_id": "fresh-clean-pile-lora-r2-64x40",
        "truth_open_order": {
            "freeze_verified_before_private_selection_read": True,
            "freeze_verified_before_truth_read": True,
            "public_commitment_opened": True,
            "opaque_record_order_verified": True,
            "prediction_revision_after_truth": False,
        },
        "metrics": {
            "correct_tokens": correct_tokens,
            "scored_tokens": scored_tokens,
            "token_accuracy": correct_tokens / scored_tokens,
            "covered_tokens": covered_tokens,
            "coverage": covered_tokens / scored_tokens,
            "selective_accuracy": correct_tokens / covered_tokens if covered_tokens else None,
            "exact_records": int(correct.all(dim=1).sum().item()),
            "records": 64,
            "exact_record_rate": float(correct.all(dim=1).float().mean().item()),
            "candidate_budget": int(candidates.shape[2]),
            "candidate_recall": included_tokens / scored_tokens,
            "candidate_included_tokens": included_tokens,
            "conditional_selector_accuracy": int((correct & included).sum().item()) / included_tokens,
            "proposal_exclusions": int((~included).sum().item()),
            "selector_errors_given_inclusion": int((included & ~correct).sum().item()),
            "abstentions": int((~covered).sum().item()),
            "routes": route["routes"],
            "selected_k": route["selected_k"],
            "record_accuracy_bootstrap_95": bootstrap_mean(per_record_accuracy, draws=10000, seed=20260827),
            "first_error_positions": first_error_positions,
        },
        "success_rule": {
            "new_accuracy_strictly_greater_than": 0.8397,
            "passed": (correct_tokens / scored_tokens) > 0.8397,
        },
        "cost": {
            key: evidence[key]
            for key in (
                "proposal_seconds",
                "selection_seconds",
                "method_compute_seconds",
                "logical_candidate_simulations",
                "executed_candidate_simulations",
                "prefix_commit_tokens",
                "record_batch_size",
                "memory",
            )
        },
        "route": route,
        "artifacts": {
            "freeze_receipt": file_record(args.receipt),
            "public_commitment": file_record(args.public_commitment),
            "prediction": file_record(prediction_path),
            "access_manifest": file_record(args.output_root / "access_manifest.json"),
            "reconstructor_evidence": file_record(args.output_root / "reconstructor_evidence.json"),
            "truth": file_record(args.truth),
        },
        "private_values_committed": False,
        "exit_status": 0,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(args.result, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "policy_id": receipt["policy_id"],
                "correct_tokens": correct_tokens,
                "scored_tokens": scored_tokens,
                "token_accuracy": result["metrics"]["token_accuracy"],
                "exact_records": result["metrics"]["exact_records"],
                "success": result["success_rule"]["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
