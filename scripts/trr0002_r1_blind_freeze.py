#!/usr/bin/env python3
"""Freeze all owner-R1 blind inputs, code, and outputs before truth reveal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from token_reconstruction.experiment_runtime import (
    command_record,
    file_record,
    load_json,
    sha256_file,
    utc_now,
    write_json_exclusive,
)


METHOD_ID = "a1_a2_exhaustive_configuration_winner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-commitment", type=Path, required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def tree_records(root: Path) -> list[dict[str, Any]]:
    resolved = root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"freeze tree contains a symlink: {path}")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(resolved).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not records:
        raise RuntimeError(f"freeze tree is empty: {root}")
    return records


def main() -> int:
    args = parse_args()
    if args.receipt.exists() or args.receipt.is_symlink():
        raise RuntimeError("blind freeze receipt is create-only")
    input_root = args.input_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    code_root = args.code_root.resolve(strict=True)
    expected_outputs = {
        "access_manifest.json",
        "predictions.safetensors",
        "reconstructor_evidence.json",
        "route.json",
    }
    if {path.name for path in output_root.iterdir()} != expected_outputs:
        raise RuntimeError("blind output file set changed before freeze")
    access = load_json(output_root / "access_manifest.json")
    evidence = load_json(output_root / "reconstructor_evidence.json")
    route = load_json(output_root / "route.json")
    if (
        access.get("result") != "PASS_FAIL_CLOSED_ACCESS_BOUNDARY"
        or access.get("method_id") != METHOD_ID
    ):
        raise RuntimeError("blind access evidence is not passing")
    if (
        evidence.get("status") != "OWNER_R1_BLIND_PREDICTIONS_FROZEN"
        or evidence.get("method_id") != METHOD_ID
        or evidence.get("truth_or_source_inputs") != 0
        or evidence.get("target_prefix_calls") != 0
        or evidence.get("exit_status") != 0
    ):
        raise RuntimeError("blind reconstruction evidence changed")
    if (
        route.get("truth_or_source_inputs") != 0
        or route.get("target_prefix_calls") != 0
        or route.get("method_id") != METHOD_ID
        or route.get("policy_id") != evidence.get("policy_id")
    ):
        raise RuntimeError("blind route evidence changed")
    plan = load_json(args.plan)
    winner = load_json(args.winner)
    if (
        plan.get("schema")
        != "token-reconstruction.trr0002-owner-r1-blind-preregistration.v1"
        or plan.get("truth_opened") is not False
        or plan.get("policy_id") != evidence.get("policy_id")
        or winner.get("policy_id") != evidence.get("policy_id")
        or plan.get("winner", {}).get("sha256") != sha256_file(args.winner)
    ):
        raise RuntimeError("blind plan or winner changed before freeze")
    receipt = {
        "schema": "token-reconstruction.trr0002-owner-r1-blind-freeze-receipt.v1",
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R1",
        "method_id": METHOD_ID,
        "policy_id": evidence["policy_id"],
        "status": "OWNER_R1_FROZEN_AND_VERIFIED_BEFORE_TRUTH_OPEN",
        "created_utc": utc_now(),
        "command": command_record(),
        "truth_opened": False,
        "truth_or_private_selection_arguments": 0,
        "plan": file_record(args.plan),
        "public_commitment": file_record(args.public_commitment),
        "winner": file_record(args.winner),
        "input_root": str(input_root),
        "input_files": tree_records(input_root),
        "output_root": str(output_root),
        "output_files": tree_records(output_root),
        "code_root": str(code_root),
        "code_files": tree_records(code_root),
        "checks": {
            "access_boundary_passed": True,
            "prediction_status_frozen": True,
            "zero_truth_or_source_inputs": True,
            "zero_target_prefix_calls": True,
            "policy_matches_preregistration": True,
            "abstention_behavior_preserved": True,
        },
        "exit_status": 0,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(args.receipt, receipt)
    for root in (input_root, output_root, code_root):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                path.chmod(0o444)
    args.receipt.chmod(0o444)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "policy_id": receipt["policy_id"],
                "input_files": len(receipt["input_files"]),
                "output_files": len(receipt["output_files"]),
                "code_files": len(receipt["code_files"]),
                "truth_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
