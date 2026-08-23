#!/usr/bin/env python3
"""Run the frozen exhaustive A1+A2 winner on both canonical setups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
from typing import Any, Mapping

from safetensors.torch import save_file
import torch
import transformers

import trr0001_r2_dual_benchmark as r2
from token_reconstruction.a1a2_configuration_search import (
    decode_policy,
    resolved_policy_from_dict,
)
from token_reconstruction.component_crossover import propose_public_a1
from token_reconstruction.dual_benchmark import (
    SETUP_IDS,
    score_predictions,
    scored_mask,
)
from token_reconstruction.experiment_runtime import (
    command_record,
    peak_memory,
    seed_everything,
)
from token_reconstruction.metrics import bootstrap_mean


NEW_SETUP, OLD_SETUP = SETUP_IDS
METHOD_ID = "a1_a2_exhaustive_configuration_winner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--new-input-root", type=Path, required=True)
    parser.add_argument("--new-truth-jsonl", type=Path, required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--crossover-result", type=Path, required=True)
    parser.add_argument("--calibrated-result", type=Path, required=True)
    parser.add_argument("--prediction-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_winner(path: Path) -> tuple[dict[str, Any], Any]:
    payload = load_json(path)
    if (
        payload.get("schema")
        != "token-reconstruction.trr0002-owner-r1-frozen-winner.v1"
        or payload.get("status")
        != "FROZEN_BEFORE_PUBLIC_HELDOUT_FRESH_BLIND_OR_CANONICAL_ACCESS"
        or payload.get("canonical_evaluation_observation_inputs") != 0
        or payload.get("canonical_evaluation_truth_inputs") != 0
        or payload.get("heldout_public_inputs") != 0
        or payload.get("fresh_blind_inputs") != 0
    ):
        raise RuntimeError("winner is not the pre-confirmation frozen artifact")
    policy = resolved_policy_from_dict(payload["policy"])
    if payload.get("policy_id") != policy.policy_id:
        raise RuntimeError("winner policy identity changed")
    return payload, policy


def counts(values: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    unique, frequencies = torch.unique(values[mask].to(torch.long), return_counts=True)
    return {
        str(int(key.item())): int(value.item())
        for key, value in zip(unique, frequencies, strict=True)
    }


def run_setup(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    lens: torch.nn.Module,
    embeddings: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
    policy: Any,
) -> dict[str, Any]:
    proposal = propose_public_a1(
        observations=observations,
        attention_mask=attention_mask,
        lens=lens,
        normalized_embeddings=embeddings,
    )
    selector = decode_policy(
        observations=observations,
        attention_mask=attention_mask,
        position_ids=position_ids,
        candidates=proposal.candidates,
        a1_confidence=proposal.top1_confidence,
        precut=precut,
        device=device,
        policy=policy,
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
    setups: Mapping[str, Mapping[str, Any]],
    path: Path,
    execution_commit: str,
    policy: Any,
) -> dict[str, Any]:
    aliases = {NEW_SETUP: "clean", OLD_SETUP: "historical"}
    tensors: dict[str, torch.Tensor] = {}
    max_k = max(policy.spec.schedule)
    for setup_id, state in setups.items():
        prefix = aliases[setup_id]
        proposal = state["proposal"]
        selector = state["selector"]
        tensors[f"{prefix}.predictions"] = selector.predictions.to(torch.int32)
        tensors[f"{prefix}.candidates_k{max_k}"] = proposal.candidates[:, :, :max_k].to(torch.int32)
        tensors[f"{prefix}.proposal_top1_confidence"] = proposal.top1_confidence.float()
        tensors[f"{prefix}.routes"] = selector.routes.to(torch.int8)
        tensors[f"{prefix}.selected_k"] = selector.selected_k.to(torch.int16)
        tensors[f"{prefix}.selected_signal"] = selector.selected_signal.float()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        path,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r1-canonical-freeze.v1",
            "task_id": "TRR-0002",
            "method_id": METHOD_ID,
            "policy_id": policy.policy_id,
            "execution_commit": execution_commit,
            "truth_status": "already-open-retrospective-but-loaded-after-freeze",
        },
    )
    return r2.file_record(path)


def score_setup(
    *,
    state: Mapping[str, Any],
    truth: torch.Tensor,
    record_ids: list[str],
    policy: Any,
) -> dict[str, Any]:
    proposal = state["proposal"]
    selector = state["selector"]
    attention_mask = state["attention_mask"]
    max_k = max(policy.spec.schedule)
    candidates = proposal.candidates[:, :, :max_k]
    metrics, per_record = score_predictions(
        predictions=selector.predictions,
        truth=truth,
        attention_mask=attention_mask,
        candidates=candidates,
        record_ids=record_ids,
    )
    mask = scored_mask(attention_mask)
    return {
        "metrics": metrics,
        "per_record": per_record,
        "routes": counts(selector.routes, mask),
        "selected_k": counts(selector.selected_k, mask),
        "cost": {
            "proposal_seconds": proposal.elapsed_seconds,
            "selection_seconds": selector.elapsed_seconds,
            "compute_seconds": proposal.elapsed_seconds + selector.elapsed_seconds,
            "candidate_simulations": selector.candidate_simulations,
            "executed_candidate_simulations": selector.executed_candidate_simulations,
            "prefix_commit_tokens": selector.prefix_commit_tokens,
            "record_batch_size": selector.record_batch_size,
        },
    }


def paired_comparison(
    winner_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if len(winner_rows) != len(baseline_rows):
        raise RuntimeError("paired comparator record count changed")
    differences: list[float] = []
    for winner, baseline in zip(winner_rows, baseline_rows, strict=True):
        if winner["record_id"] != baseline["record_id"]:
            raise RuntimeError("paired comparator record order changed")
        differences.append(float(winner["token_accuracy"]) - float(baseline["token_accuracy"]))
    return {
        "mean_record_accuracy_difference": statistics.mean(differences),
        "record_difference_bootstrap_95": bootstrap_mean(differences, draws=10000, seed=seed),
        "winner_better_records": sum(value > 0 for value in differences),
        "tied_records": sum(value == 0 for value in differences),
        "winner_worse_records": sum(value < 0 for value in differences),
    }


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    if args.prediction_artifact.exists() or args.output.exists():
        raise RuntimeError("canonical outputs are create-only")
    winner_payload, policy = load_winner(args.winner)
    seed_everything(20260825)
    execution_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    started_utc = r2.utc_now()

    reference_path = repository_root / "reference" / "strict_bos" / "round001_teacher.py"
    reference = r2.import_path("trr0002_r1_dual_reference", reference_path)
    source_path = historical_root / "scripts" / "score_a1_a2_source300_20260809.py"
    source300 = r2.import_path("trr0002_r1_dual_source300", source_path)
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
        r2.MODEL_SPEC, load_json(identity_path), lens_path=lens_path
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
            policy=policy,
        ),
        OLD_SETUP: run_setup(
            observations=old_observations,
            attention_mask=old_mask,
            position_ids=old_positions,
            lens=lens,
            embeddings=embeddings,
            precut=precut,
            device=device,
            policy=policy,
        ),
    }
    peak_cuda = int(torch.cuda.max_memory_allocated(device))
    prediction_record = freeze_predictions(
        setups=setups,
        path=args.prediction_artifact,
        execution_commit=execution_commit,
        policy=policy,
    )
    prediction_frozen_utc = r2.utc_now()

    # The retrospective truths are deliberately loaded only after predictions exist.
    new_truth, new_ids = r2.load_new_truth(args.new_truth_jsonl, 64)
    old_truth, old_ids = r2.load_old_truth(source300, old_captures, old_config)
    truth_loaded_utc = r2.utc_now()
    scored = {
        NEW_SETUP: score_setup(state=setups[NEW_SETUP], truth=new_truth, record_ids=new_ids, policy=policy),
        OLD_SETUP: score_setup(state=setups[OLD_SETUP], truth=old_truth, record_ids=old_ids, policy=policy),
    }
    crossover = load_json(args.crossover_result)
    calibrated = load_json(args.calibrated_result)
    comparisons: dict[str, Any] = {}
    for setup_index, setup_id in enumerate(SETUP_IDS):
        comparisons[setup_id] = {
            "historical_strict_a1_a2": paired_comparison(
                scored[setup_id]["per_record"],
                crossover["matrix"][setup_id]["strict_bos_adaptive_a1_a2"]["per_record"],
                seed=20260830 + setup_index,
            ),
            "always_causal_k64": paired_comparison(
                scored[setup_id]["per_record"],
                crossover["matrix"][setup_id]["a1_causal_k64"]["per_record"],
                seed=20260840 + setup_index,
            ),
            "previous_calibrated_successor": paired_comparison(
                scored[setup_id]["per_record"],
                calibrated["setups"][setup_id]["per_record"],
                seed=20260850 + setup_index,
            ),
        }

    new_accuracy = scored[NEW_SETUP]["metrics"]["token_accuracy"]
    old_accuracy = scored[OLD_SETUP]["metrics"]["token_accuracy"]
    result = {
        "schema": "token-reconstruction.trr0002-owner-r1-canonical-result.v1",
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R1",
        "status": "FROZEN_EXHAUSTIVE_WINNER_EVALUATED_ON_BOTH_CANONICAL_SETUPS",
        "method_id": METHOD_ID,
        "policy_id": policy.policy_id,
        "policy": policy.serialized(),
        "execution_commit": execution_commit,
        "started_utc": started_utc,
        "ended_utc": r2.utc_now(),
        "command": command_record(),
        "exit_status": 0,
        "truth_status": "already-open retrospective canonical truth; prediction artifact written first",
        "prediction_frozen_utc": prediction_frozen_utc,
        "truth_loaded_utc": truth_loaded_utc,
        "winner_revision_after_selection": False,
        "setups": scored,
        "paired_comparisons": comparisons,
        "success_rule": {
            "historical_accuracy_at_least": 0.9822,
            "canonical_new_accuracy_strictly_greater_than": 0.8397,
            "canonical_thresholds_passed": old_accuracy >= 0.9822 and new_accuracy > 0.8397,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "peak_cuda_memory_allocated_bytes": peak_cuda,
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "pid": os.getpid(),
            "peak_memory": peak_memory(),
        },
        "public_teacher_identity": observed_identity,
        "artifacts": {
            "winner": r2.file_record(args.winner),
            "prediction_freeze": prediction_record,
            "crossover_result": r2.file_record(args.crossover_result),
            "calibrated_result": r2.file_record(args.calibrated_result),
            "new_observation": r2.file_record(
                args.new_input_root / "observations" / "unavailable_target_lora_cut4.safetensors"
            ),
            "new_truth": r2.file_record(args.new_truth_jsonl),
            "old_source": r2.file_record(source300.resolve_inside_ersoy(old_config["source"]["path"])),
            "lens": r2.file_record(lens_path),
            "method_source": r2.file_record(
                repository_root / "src" / "token_reconstruction" / "a1a2_configuration_search.py"
            ),
            "runner_source": r2.file_record(Path(__file__)),
        },
    }
    r2.write_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "policy_id": policy.policy_id,
                "canonical_new_accuracy": new_accuracy,
                "historical_accuracy": old_accuracy,
                "canonical_thresholds_passed": result["success_rule"]["canonical_thresholds_passed"],
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
