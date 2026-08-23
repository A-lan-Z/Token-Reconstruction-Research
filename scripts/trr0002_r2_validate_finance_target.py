#!/usr/bin/env python3
"""Integrity validation for the TRR-0002 owner-R2 Finance target panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file

from token_reconstruction.experiment_runtime import command_record, utc_now

import trr0001_r2_dual_benchmark as r2
import trr0002_r2_finance_target_shortlist as panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prior-winner-predictions", type=Path, required=True)
    parser.add_argument("--prior-calibrated-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def same_file(record: Mapping[str, Any], path: Path) -> bool:
    observed = r2.file_record(path)
    return all(observed[field] == record.get(field) for field in ("bytes", "sha256"))


def aggregate_per_record(row: Mapping[str, Any]) -> dict[str, int]:
    records = row["per_record"]
    return {
        "records": len(records),
        "scored_tokens": sum(item["scored_tokens"] for item in records),
        "covered_tokens": sum(item["covered_tokens"] for item in records),
        "correct_tokens": sum(item["correct_tokens"] for item in records),
        "exact_records": sum(bool(item["exact_record"]) for item in records),
        "candidate_hits": sum(item["candidate_hits"] for item in records),
    }


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("validation output is create-only")
    plan = load_json(args.plan)
    entries = panel.validate_plan(plan)
    evidence = load_json(args.evidence)
    receipt = load_json(args.receipt)
    result = load_json(args.result)
    checks: dict[str, bool] = {}

    checks["plan_status"] = (
        plan["status"] == "FROZEN_BEFORE_FINANCE_TARGET_SHORTLIST_RECONSTRUCTION"
    )
    checks["receipt_status"] = (
        receipt["status"] == "VERIFIED_BEFORE_SEPARATE_SCORING_PROCESS"
    )
    checks["result_status"] = (
        result["status"] == "RETROSPECTIVE_FINANCE_TARGET_SHORTLIST_SCORED"
    )
    checks["revision_identity"] = all(
        payload["revision_id"] == panel.REVISION_ID
        for payload in (plan, evidence, receipt, result)
    )
    checks["receipt_plan_hash"] = same_file(receipt["plan"], args.plan)
    checks["receipt_prediction_hash"] = same_file(
        receipt["prediction_artifact"], args.predictions
    )
    checks["receipt_evidence_hash"] = same_file(
        receipt["reconstruction_evidence"], args.evidence
    )
    checks["result_plan_hash"] = same_file(result["artifacts"]["plan"], args.plan)
    checks["result_prediction_hash"] = same_file(
        result["artifacts"]["prediction_artifact"], args.predictions
    )
    checks["result_receipt_hash"] = same_file(
        result["artifacts"]["freeze_receipt"], args.receipt
    )
    checks["result_evidence_hash"] = same_file(
        result["artifacts"]["reconstruction_evidence"], args.evidence
    )
    checks["truth_free_prediction_process"] = (
        evidence["access"]["dataset_inputs"] == 0
        and evidence["access"]["truth_token_inputs"] == 0
        and evidence["access"]["target_prefix_calls"] == 0
        and receipt["prediction_process_truth_loaded"] is False
        and receipt["prediction_process_target_prefix_calls"] == 0
    )

    frozen = load_file(args.predictions, device="cpu")
    checks["tensor_registry"] = set(frozen) == panel.expected_tensor_keys(entries)
    checks["prediction_geometry"] = all(
        tuple(frozen[f"{entry['policy_id']}.predictions"].shape) == (128, 128)
        for entry in entries
    )
    checks["candidate_geometry"] = tuple(
        frozen["common.candidates_top512"].shape
    ) == (128, 128, 512)

    ordered = result["diagnostic_ranking"]
    checks["ranking_count"] = len(ordered) == len(entries) == 12
    checks["ranking_sequence"] = [row["target_diagnostic_rank"] for row in ordered] == list(
        range(1, 13)
    )
    checks["distinct_accuracies"] = (
        result["differentiation"]["distinct_token_accuracy_values"] == 11
    )
    checks["perfect_count"] = result["differentiation"]["perfect_configurations"] == 0

    aggregate_checks = []
    for row in ordered:
        aggregate = aggregate_per_record(row)
        metrics = row["metrics"]
        aggregate_checks.append(
            aggregate["records"] == metrics["records"]
            and aggregate["scored_tokens"] == metrics["scored_tokens"]
            and aggregate["covered_tokens"] == metrics["covered_tokens"]
            and aggregate["correct_tokens"] == metrics["correct_tokens"]
            and aggregate["exact_records"] == metrics["exact_records"]
            and aggregate["candidate_hits"] == metrics["candidate_hits"]
            and metrics["token_accuracy"]
            == metrics["correct_tokens"] / metrics["scored_tokens"]
        )
    checks["per_record_aggregates"] = all(aggregate_checks)

    by_label = {row["label"]: row for row in ordered}
    recall = {row["k"]: row for row in result["candidate_recall_curve"]}
    checks["centered_k512_reaches_recall"] = (
        by_label["fixed_k512_centered"]["metrics"]["correct_tokens"]
        == recall[512]["hits"]
        == 13980
    )
    checks["centered_k256_one_selector_error"] = (
        recall[256]["hits"]
        - by_label["fixed_k256_centered"]["metrics"]["correct_tokens"]
        == 1
    )
    checks["direct_k256_twelve_selector_errors"] = (
        recall[256]["hits"]
        - by_label["fixed_k256_direct"]["metrics"]["correct_tokens"]
        == 12
    )
    checks["multistage_cost_and_quality"] = (
        by_label["multistage_historical_gate"]["metrics"]["correct_tokens"] == 13958
        and by_label["multistage_historical_gate"]["cost"]["candidate_simulations"]
        == 60200
    )

    prior_winner = load_file(args.prior_winner_predictions, device="cpu")
    direct = frozen["a1a2_43ea0bb737bc075531ca.predictions"]
    winner_mismatches = int(direct.ne(prior_winner["historical.predictions"]).sum())
    checks["prior_direct_k256_reproduction"] = winner_mismatches == 0

    prior_calibrated = load_file(args.prior_calibrated_predictions, device="cpu")
    calibrated = frozen["a1a2_c316cdf581012bd81cfa.predictions"]
    calibrated_mismatches = int(
        calibrated.ne(prior_calibrated["historical.predictions"]).sum()
    )
    checks["calibrated_common_runner_difference_recorded"] = calibrated_mismatches == 1

    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Finance-target validation failed: {failures}")
    payload = {
        "schema": "token-reconstruction.trr0002-owner-r2-finance-target-validation.v1",
        "task_id": panel.TASK_ID,
        "revision_id": panel.REVISION_ID,
        "status": "PASS_ALL_FINANCE_TARGET_INTEGRITY_CHECKS",
        "created_utc": utc_now(),
        "command": command_record(),
        "exit_status": 0,
        "checks": checks,
        "policy_count": len(entries),
        "direct_k256_prior_prediction_mismatches": winner_mismatches,
        "calibrated_prior_prediction_mismatches": calibrated_mismatches,
        "artifacts": {
            "plan": r2.file_record(args.plan),
            "predictions": r2.file_record(args.predictions),
            "evidence": r2.file_record(args.evidence),
            "receipt": r2.file_record(args.receipt),
            "result": r2.file_record(args.result),
            "prior_winner_predictions": r2.file_record(args.prior_winner_predictions),
            "prior_calibrated_predictions": r2.file_record(
                args.prior_calibrated_predictions
            ),
            "validator_source": r2.file_record(Path(__file__)),
        },
    }
    r2.write_json_exclusive(args.output, payload)
    print(json.dumps({"status": payload["status"], "checks": len(checks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    result = main()
    raise SystemExit(result)
