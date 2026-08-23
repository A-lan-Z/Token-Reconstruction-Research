#!/usr/bin/env python3
"""Fit and freeze the TRR-0002 normalized-confidence expansion threshold."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any

from safetensors.torch import load_file
import torch
import transformers

import trr0001_r2_dual_benchmark as r2
from token_reconstruction.calibrated_selector import (
    BASE_BUDGET,
    MAX_BUDGET,
    ROUTE_EXPANDED,
    CalibratedSelectorResult,
    select_calibrated_adaptive,
)
from token_reconstruction.component_crossover import propose_public_a1, true_token_ranks
from token_reconstruction.dual_benchmark import scored_mask
from token_reconstruction.experiment_runtime import seed_everything


NO_EXPANSION = -1.0
ALL_EXPANSION = 1_000_000.0
CONDITIONS = (
    "public_base",
    "public_lora_2601",
    "public_lora_2602",
    "public_lora_2603",
)
FIT_CONDITIONS = CONDITIONS[:3]
HELD_OUT = CONDITIONS[3]
QUANTILES = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_recall(candidates: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    ranks = true_token_ranks(
        candidates=candidates,
        truth=truth,
        attention_mask=torch.ones_like(truth),
    )
    values = ranks[:, 1:]
    return {
        "k32": int(((values > 0) & (values <= BASE_BUDGET)).sum().item()),
        "k64": int(((values > 0) & (values <= MAX_BUDGET)).sum().item()),
        "scored_tokens": int(values.numel()),
    }


def score_result(
    result: CalibratedSelectorResult,
    truth: torch.Tensor,
) -> dict[str, Any]:
    predicted = result.predictions[:, 1:]
    expected = truth[:, 1:]
    correct = predicted.eq(expected)
    expanded = result.routes[:, 1:].eq(ROUTE_EXPANDED)
    return {
        "correct_tokens": int(correct.sum().item()),
        "scored_tokens": int(correct.numel()),
        "token_accuracy": float(correct.float().mean().item()),
        "exact_records": int(correct.all(dim=1).sum().item()),
        "expanded_positions": int(expanded.sum().item()),
        "expansion_rate": float(expanded.float().mean().item()),
        "base_candidate_simulations": result.base_candidate_simulations,
        "extra_candidate_simulations": result.extra_candidate_simulations,
        "logical_candidate_simulations": (
            result.base_candidate_simulations + result.extra_candidate_simulations
        ),
        "executed_candidate_simulations": result.executed_candidate_simulations,
        "prefix_commit_tokens": result.prefix_commit_tokens,
        "selection_seconds": result.elapsed_seconds,
    }


def threshold_key(value: float) -> str:
    return format(value, ".17g")


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    public_root = args.public_root.resolve(strict=True)
    if args.fit_output.exists() or args.calibration_output.exists():
        raise RuntimeError("calibration outputs are create-only")
    plan = load_json(args.plan)
    if plan.get("schema") != "token-reconstruction.trr0002-calibration-preregistration.v1":
        raise RuntimeError("calibration plan schema changed")
    generation = load_json(public_root / "generation.json")
    if (
        generation.get("canonical_evaluation_truth_inputs") != 0
        or generation.get("canonical_evaluation_observation_inputs") != 0
        or generation.get("target_lora_inputs") != 0
    ):
        raise RuntimeError("public-development generation accessed evaluation material")
    seed_everything(2700)
    execution_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    started_utc = r2.utc_now()

    truth_state = load_file(public_root / "truth.safetensors", device="cpu")
    if set(truth_state) != {"token_ids"}:
        raise RuntimeError("public-development truth fields changed")
    truth = truth_state["token_ids"].to(torch.long)
    if tuple(truth.shape) != (32, 40) or not truth[:, 0].eq(128000).all().item():
        raise RuntimeError("public-development truth geometry changed")
    attention_mask = torch.ones_like(truth)
    position_ids = torch.arange(40, dtype=torch.long).view(1, -1).expand(32, -1)

    reference_path = repository_root / "reference" / "strict_bos" / "round001_teacher.py"
    reference = r2.import_path("trr0002_calibration_reference", reference_path)
    identity_path = (
        historical_root
        / "research"
        / "adaptive_a1_a2_strict_bos_20260817_goal_01a00b08"
        / "audit"
        / "AUDIT-0004-public-teacher-identity-v2.json"
    )
    lens_path = historical_root / "inversion_20260730" / "out" / "lens_alpaca.pt"
    identity = load_json(identity_path)
    precut, lens, embeddings, device, observed_identity = reference.load_public_teacher(
        r2.MODEL_SPEC,
        identity,
        lens_path=lens_path,
    )
    torch.cuda.reset_peak_memory_stats(device)

    observations: dict[str, torch.Tensor] = {}
    proposals: dict[str, Any] = {}
    proposal_evidence: dict[str, Any] = {}
    for condition in CONDITIONS:
        path = public_root / "observations" / f"{condition}_cut4.safetensors"
        state = load_file(path, device="cpu")
        if set(state) != {"activations"} or tuple(state["activations"].shape) != (32, 40, 2048):
            raise RuntimeError(f"public observation changed: {condition}")
        observations[condition] = state["activations"]
        proposal = propose_public_a1(
            observations=observations[condition],
            attention_mask=attention_mask,
            lens=lens,
            normalized_embeddings=embeddings,
        )
        proposals[condition] = proposal
        proposal_evidence[condition] = {
            "seconds": proposal.elapsed_seconds,
            "candidate_recall": candidate_recall(
                proposal.candidates[:, :, :MAX_BUDGET], truth
            ),
            "observation": r2.file_record(path),
        }

    baseline_results: dict[str, CalibratedSelectorResult] = {}
    fit_gaps: list[torch.Tensor] = []
    for condition in FIT_CONDITIONS:
        result = select_calibrated_adaptive(
            observations=observations[condition],
            attention_mask=attention_mask,
            position_ids=position_ids,
            candidates=proposals[condition].candidates[:, :, :MAX_BUDGET].contiguous(),
            precut=precut,
            device=device,
            threshold=NO_EXPANSION,
        )
        baseline_results[condition] = result
        fit_gaps.append(result.normalized_gap[scored_mask(attention_mask)])
    pooled_gaps = torch.cat(fit_gaps).float()
    quantile_values = torch.quantile(
        pooled_gaps,
        torch.tensor(QUANTILES, dtype=torch.float32),
    ).tolist()
    thresholds = sorted(
        {
            NO_EXPANSION,
            ALL_EXPANSION,
            *[float(value) for value in quantile_values],
        }
    )
    if thresholds[0] != NO_EXPANSION or thresholds[-1] != ALL_EXPANSION:
        raise RuntimeError("calibration threshold sentinels changed")

    table: list[dict[str, Any]] = []
    for threshold in thresholds:
        condition_rows: dict[str, Any] = {}
        for condition in CONDITIONS:
            if threshold == NO_EXPANSION and condition in baseline_results:
                result = baseline_results[condition]
            else:
                result = select_calibrated_adaptive(
                    observations=observations[condition],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    candidates=proposals[condition].candidates[:, :, :MAX_BUDGET].contiguous(),
                    precut=precut,
                    device=device,
                    threshold=threshold,
                )
            condition_rows[condition] = score_result(result, truth)
        fit_correct = sum(condition_rows[name]["correct_tokens"] for name in FIT_CONDITIONS)
        fit_scored = sum(condition_rows[name]["scored_tokens"] for name in FIT_CONDITIONS)
        fit_simulations = sum(
            condition_rows[name]["logical_candidate_simulations"]
            for name in FIT_CONDITIONS
        )
        table.append(
            {
                "threshold": threshold,
                "threshold_key": threshold_key(threshold),
                "fit_correct_tokens": fit_correct,
                "fit_scored_tokens": fit_scored,
                "fit_accuracy": fit_correct / fit_scored,
                "fit_logical_candidate_simulations": fit_simulations,
                "conditions": condition_rows,
            }
        )

    by_threshold = {row["threshold"]: row for row in table}
    base = by_threshold[NO_EXPANSION]
    full = by_threshold[ALL_EXPANSION]
    full_gain = full["fit_correct_tokens"] - base["fit_correct_tokens"]
    if full_gain > 0:
        target_correct = base["fit_correct_tokens"] + math.ceil(0.9 * full_gain)
        eligible = [row for row in table if row["fit_correct_tokens"] >= target_correct]
        selected = min(
            eligible,
            key=lambda row: (
                row["fit_logical_candidate_simulations"],
                row["threshold"],
            ),
        )
        rule_branch = "minimum_cost_at_least_90_percent_of_full_k64_gain"
    else:
        target_correct = None
        selected = min(
            table,
            key=lambda row: (
                -row["fit_correct_tokens"],
                row["fit_logical_candidate_simulations"],
                row["threshold"],
            ),
        )
        rule_branch = "maximum_accuracy_when_full_k64_does_not_beat_k32"
    selected_threshold = float(selected["threshold"])
    held_out = selected["conditions"][HELD_OUT]

    fit_payload = {
        "schema": "token-reconstruction.trr0002-calibration-fit.v1",
        "task_id": "TRR-0002",
        "status": "PUBLIC_DEVELOPMENT_FIT_COMPLETE",
        "execution_commit": execution_commit,
        "started_utc": started_utc,
        "ended_utc": r2.utc_now(),
        "truth_scope": "public development only",
        "canonical_evaluation_truth_inputs": 0,
        "canonical_evaluation_observation_inputs": 0,
        "target_lora_inputs": 0,
        "fit_conditions": list(FIT_CONDITIONS),
        "held_out_condition": HELD_OUT,
        "quantile_probabilities": list(QUANTILES),
        "quantile_values": quantile_values,
        "candidate_thresholds": thresholds,
        "selection": {
            "rule_branch": rule_branch,
            "base_correct_tokens": base["fit_correct_tokens"],
            "full_k64_correct_tokens": full["fit_correct_tokens"],
            "full_k64_gain_tokens": full_gain,
            "required_correct_tokens": target_correct,
            "selected_threshold": selected_threshold,
            "selected_fit_correct_tokens": selected["fit_correct_tokens"],
            "selected_fit_logical_candidate_simulations": selected[
                "fit_logical_candidate_simulations"
            ],
            "held_out_result_frozen_without_revision": held_out,
        },
        "threshold_table": table,
        "proposal_evidence": proposal_evidence,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "public_teacher_identity": observed_identity,
        "artifacts": {
            "plan": r2.file_record(args.plan),
            "generation": r2.file_record(public_root / "generation.json"),
            "truth": r2.file_record(public_root / "truth.safetensors"),
            "lens": r2.file_record(lens_path),
            "selector_source": r2.file_record(
                repository_root / "src" / "token_reconstruction" / "calibrated_selector.py"
            ),
            "runner_source": r2.file_record(Path(__file__)),
        },
    }
    r2.write_json_exclusive(args.fit_output, fit_payload)

    calibration_payload = {
        "schema": "token-reconstruction.trr0002-frozen-calibration.v1",
        "task_id": "TRR-0002",
        "method_id": "a1_scale_calibrated_adaptive_causal_k32_to64",
        "status": "FROZEN_BEFORE_FRESH_BLIND_SELECTION",
        "created_utc": r2.utc_now(),
        "execution_commit": execution_commit,
        "base_budget": BASE_BUDGET,
        "maximum_budget": MAX_BUDGET,
        "normalized_confidence": "top1_minus_top2_divided_by_candidate_score_rms_deviation",
        "expand_when": "normalized_confidence_less_than_or_equal_to_threshold",
        "threshold": selected_threshold,
        "abstention": "none",
        "target_prefix_calls": 0,
        "fit_conditions": list(FIT_CONDITIONS),
        "held_out_condition": HELD_OUT,
        "fit_result": r2.file_record(args.fit_output),
        "plan": r2.file_record(args.plan),
        "selector_source": r2.file_record(
            repository_root / "src" / "token_reconstruction" / "calibrated_selector.py"
        ),
        "lens": r2.file_record(lens_path),
        "public_teacher_identity": observed_identity,
        "revision_policy": "immutable; any change requires a new preregistration",
    }
    r2.write_json_exclusive(args.calibration_output, calibration_payload)
    print(
        json.dumps(
            {
                "status": "CALIBRATION_FROZEN",
                "selected_threshold": selected_threshold,
                "fit_correct_tokens": selected["fit_correct_tokens"],
                "held_out_correct_tokens": held_out["correct_tokens"],
                "fit_output": str(args.fit_output),
                "calibration_output": str(args.calibration_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
