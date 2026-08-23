#!/usr/bin/env python3
"""Evaluate the frozen calibrated selector on both canonical setups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from safetensors.torch import save_file
import torch
import transformers

import trr0001_r2_dual_benchmark as r2
from token_reconstruction.calibrated_selector import (
    ROUTE_EXPANDED,
    select_calibrated_adaptive,
)
from token_reconstruction.component_crossover import (
    propose_public_a1,
    quantile_summary,
    selector_error_attribution,
)
from token_reconstruction.dual_benchmark import (
    SETUP_IDS,
    score_predictions,
    scored_mask,
)
from token_reconstruction.experiment_runtime import seed_everything


NEW_SETUP, OLD_SETUP = SETUP_IDS
METHOD_ID = "a1_scale_calibrated_adaptive_causal_k32_to64"
CALIBRATION_SHA256 = "ad1801ec348a61cbcd50bfbc4a991c8deaa503b79f454c7f1d779567042ebf47"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--new-input-root", type=Path, required=True)
    parser.add_argument("--new-truth-jsonl", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--crossover-result", type=Path, required=True)
    parser.add_argument("--fresh-blind-result", type=Path, required=True)
    parser.add_argument("--prediction-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_setup(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    lens: torch.nn.Module,
    embeddings: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    proposal = propose_public_a1(
        observations=observations,
        attention_mask=attention_mask,
        lens=lens,
        normalized_embeddings=embeddings,
    )
    selector = select_calibrated_adaptive(
        observations=observations,
        attention_mask=attention_mask,
        position_ids=position_ids,
        candidates=proposal.candidates[:, :, :64].contiguous(),
        precut=precut,
        device=device,
        threshold=threshold,
        record_batch_size=8,
    )
    return {
        "proposal": proposal,
        "selector": selector,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def freeze_predictions(
    *,
    setups: dict[str, dict[str, Any]],
    path: Path,
    execution_commit: str,
    threshold: float,
) -> dict[str, Any]:
    aliases = {NEW_SETUP: "clean", OLD_SETUP: "historical"}
    tensors: dict[str, torch.Tensor] = {}
    for setup_id, state in setups.items():
        prefix = aliases[setup_id]
        proposal = state["proposal"]
        selector = state["selector"]
        tensors[f"{prefix}.predictions"] = selector.predictions.to(torch.int32).contiguous()
        tensors[f"{prefix}.candidates_k64"] = (
            proposal.candidates[:, :, :64].to(torch.int32).contiguous()
        )
        tensors[f"{prefix}.proposal_top1_confidence"] = (
            proposal.top1_confidence.float().contiguous()
        )
        tensors[f"{prefix}.base_selection_scores"] = selector.base_scores.float().contiguous()
        tensors[f"{prefix}.extra_selection_scores"] = selector.extra_scores.float().contiguous()
        tensors[f"{prefix}.normalized_gap"] = selector.normalized_gap.float().contiguous()
        tensors[f"{prefix}.routes"] = selector.routes.to(torch.int8).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        path,
        metadata={
            "schema": "token-reconstruction.trr0002-calibrated-dual-freeze.v1",
            "task_id": "TRR-0002",
            "method_id": METHOD_ID,
            "execution_commit": execution_commit,
            "threshold": format(threshold, ".17g"),
            "truth_status": "already-open-retrospective-but-loaded-after-freeze",
        },
    )
    return r2.file_record(path)


def score_setup(
    *,
    state: dict[str, Any],
    truth: torch.Tensor,
    record_ids: list[str],
) -> dict[str, Any]:
    proposal = state["proposal"]
    selector = state["selector"]
    attention_mask = state["attention_mask"]
    candidates = proposal.candidates[:, :, :64]
    metrics, per_record = score_predictions(
        predictions=selector.predictions,
        truth=truth,
        attention_mask=attention_mask,
        candidates=candidates,
        record_ids=record_ids,
    )
    attribution = selector_error_attribution(
        predictions=selector.predictions,
        truth=truth,
        attention_mask=attention_mask,
        candidates=candidates,
    )
    mask = scored_mask(attention_mask)
    expected = truth[mask].to(torch.long)
    predicted = selector.predictions[mask].to(torch.long)
    correct = predicted.eq(expected)
    expanded = selector.routes[mask].eq(ROUTE_EXPANDED)
    gaps = selector.normalized_gap[mask]
    return {
        "metrics": metrics,
        "per_record": per_record,
        "error_attribution": attribution,
        "routing": {
            "expanded_positions": int(expanded.sum().item()),
            "expansion_rate": float(expanded.float().mean().item()),
            "correct_when_expanded": quantile_summary(correct[expanded].float()),
            "correct_without_expansion": quantile_summary(correct[~expanded].float()),
            "normalized_gap_all": quantile_summary(gaps),
            "normalized_gap_correct": quantile_summary(gaps[correct]),
            "normalized_gap_incorrect": quantile_summary(gaps[~correct]),
            "abstentions": int(predicted.lt(0).sum().item()),
        },
        "cost": {
            "proposal_seconds": proposal.elapsed_seconds,
            "selection_seconds": selector.elapsed_seconds,
            "compute_seconds": proposal.elapsed_seconds + selector.elapsed_seconds,
            "base_candidate_simulations": selector.base_candidate_simulations,
            "extra_candidate_simulations": selector.extra_candidate_simulations,
            "logical_candidate_simulations": (
                selector.base_candidate_simulations
                + selector.extra_candidate_simulations
            ),
            "executed_candidate_simulations": selector.executed_candidate_simulations,
            "prefix_commit_tokens": selector.prefix_commit_tokens,
        },
    }


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    if args.prediction_artifact.exists() or args.output.exists():
        raise RuntimeError("calibrated dual outputs are create-only")
    if r2.sha256_file(args.calibration) != CALIBRATION_SHA256:
        raise RuntimeError("frozen calibration hash changed")
    calibration = load_json(args.calibration)
    if (
        calibration.get("schema") != "token-reconstruction.trr0002-frozen-calibration.v1"
        or calibration.get("method_id") != METHOD_ID
        or calibration.get("threshold") != 1.2544946670532227
    ):
        raise RuntimeError("frozen calibration fields changed")
    threshold = float(calibration["threshold"])
    seed_everything(3001)
    execution_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    started_utc = r2.utc_now()

    reference_path = repository_root / "reference" / "strict_bos" / "round001_teacher.py"
    reference = r2.import_path("trr0002_calibrated_dual_reference", reference_path)
    source_path = historical_root / "scripts" / "score_a1_a2_source300_20260809.py"
    source300 = r2.import_path("trr0002_calibrated_dual_source300", source_path)
    new_observations, new_mask, new_positions = r2.new_inputs(
        args.new_input_root.resolve(strict=True)
    )
    old_config, old_captures, old_observations, old_mask, old_positions = r2.historical_inputs(
        historical_root, source300
    )
    identity_path = (
        historical_root
        / "research"
        / "adaptive_a1_a2_strict_bos_20260817_goal_01a00b08"
        / "audit"
        / "AUDIT-0004-public-teacher-identity-v2.json"
    )
    lens_path = historical_root / "inversion_20260730" / "out" / "lens_alpaca.pt"
    precut, lens, embeddings, device, observed_identity = reference.load_public_teacher(
        r2.MODEL_SPEC,
        load_json(identity_path),
        lens_path=lens_path,
    )
    torch.cuda.reset_peak_memory_stats(device)
    setups = {
        NEW_SETUP: run_setup(
            observations=new_observations,
            attention_mask=new_mask,
            position_ids=new_positions,
            lens=lens,
            embeddings=embeddings,
            precut=precut,
            device=device,
            threshold=threshold,
        ),
        OLD_SETUP: run_setup(
            observations=old_observations,
            attention_mask=old_mask,
            position_ids=old_positions,
            lens=lens,
            embeddings=embeddings,
            precut=precut,
            device=device,
            threshold=threshold,
        ),
    }
    peak_cuda = int(torch.cuda.max_memory_allocated(device))
    prediction_record = freeze_predictions(
        setups=setups,
        path=args.prediction_artifact,
        execution_commit=execution_commit,
        threshold=threshold,
    )

    # Both canonical truths are loaded only after the prediction artifact exists.
    new_truth, new_ids = r2.load_new_truth(args.new_truth_jsonl, 64)
    old_truth, old_ids = r2.load_old_truth(source300, old_captures, old_config)
    scored = {
        NEW_SETUP: score_setup(state=setups[NEW_SETUP], truth=new_truth, record_ids=new_ids),
        OLD_SETUP: score_setup(state=setups[OLD_SETUP], truth=old_truth, record_ids=old_ids),
    }
    crossover = load_json(args.crossover_result)
    fresh = load_json(args.fresh_blind_result)
    new_accuracy = scored[NEW_SETUP]["metrics"]["token_accuracy"]
    old_accuracy = scored[OLD_SETUP]["metrics"]["token_accuracy"]
    fresh_accuracy = fresh["metrics"]["token_accuracy"]
    success = old_accuracy >= 0.9822 and new_accuracy > 0.8397 and fresh_accuracy > 0.8397
    result = {
        "schema": "token-reconstruction.trr0002-calibrated-dual-result.v1",
        "task_id": "TRR-0002",
        "status": "FROZEN_CALIBRATED_METHOD_EVALUATED_ON_BOTH_CANONICAL_SETUPS",
        "method_id": METHOD_ID,
        "execution_commit": execution_commit,
        "started_utc": started_utc,
        "ended_utc": r2.utc_now(),
        "threshold": threshold,
        "truth_status": {
            "canonical_setups": "already-open retrospective; prediction artifact written before this runner loaded truth",
            "fresh_confirmation": "separately frozen and scored blind",
        },
        "setups": scored,
        "fresh_blind_confirmation": {
            "result": r2.file_record(args.fresh_blind_result),
            "correct_tokens": fresh["metrics"]["correct_tokens"],
            "scored_tokens": fresh["metrics"]["scored_tokens"],
            "token_accuracy": fresh_accuracy,
            "exact_records": fresh["metrics"]["exact_records"],
        },
        "comparison": {
            "canonical_new_a1_causal_k32": crossover["matrix"][NEW_SETUP]["a1_causal_k32"]["metrics"],
            "canonical_new_a1_causal_k64": crossover["matrix"][NEW_SETUP]["a1_causal_k64"]["metrics"],
            "historical_a1_causal_k32": crossover["matrix"][OLD_SETUP]["a1_causal_k32"]["metrics"],
            "historical_a1_causal_k64": crossover["matrix"][OLD_SETUP]["a1_causal_k64"]["metrics"],
            "historical_strict_a1_a2": crossover["matrix"][OLD_SETUP]["strict_bos_adaptive_a1_a2"]["metrics"],
        },
        "success_rule": {
            "historical_accuracy_minimum": 0.9822,
            "canonical_new_accuracy_strictly_greater_than": 0.8397,
            "fresh_new_accuracy_strictly_greater_than": 0.8397,
            "same_frozen_method_passed_all": success,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "peak_cuda_memory_allocated_bytes": peak_cuda,
            "process_max_rss_kib": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss,
            "pid": os.getpid(),
        },
        "public_teacher_identity": observed_identity,
        "artifacts": {
            "prediction_freeze": prediction_record,
            "calibration": r2.file_record(args.calibration),
            "crossover_result": r2.file_record(args.crossover_result),
            "new_observation": r2.file_record(
                args.new_input_root / "observations" / "unavailable_target_lora_cut4.safetensors"
            ),
            "new_truth": r2.file_record(args.new_truth_jsonl),
            "old_source": r2.file_record(source300.resolve_inside_ersoy(old_config["source"]["path"])),
            "lens": r2.file_record(lens_path),
            "selector_source": r2.file_record(
                repository_root / "src" / "token_reconstruction" / "calibrated_selector.py"
            ),
            "runner_source": r2.file_record(Path(__file__)),
        },
    }
    r2.write_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "canonical_new_accuracy": new_accuracy,
                "historical_accuracy": old_accuracy,
                "fresh_blind_accuracy": fresh_accuracy,
                "same_frozen_method_passed_all": success,
                "prediction_artifact": str(args.prediction_artifact),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
