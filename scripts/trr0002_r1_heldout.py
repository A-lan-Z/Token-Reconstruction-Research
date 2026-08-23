#!/usr/bin/env python3
"""Evaluate the frozen exhaustive A1+A2 winner on public LoRA 2603."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from safetensors.torch import load_file, save_file
import torch

from token_reconstruction.a1a2_configuration_search import (
    decode_policy,
    resolved_policy_from_dict,
)
from token_reconstruction.component_crossover import propose_public_a1
from token_reconstruction.dual_benchmark import (
    BOS_TOKEN_ID,
    score_predictions,
    scored_mask,
    validate_observations,
)
from token_reconstruction.experiment_runtime import (
    command_record,
    file_record,
    peak_memory,
    seed_everything,
    utc_now,
    write_json_exclusive,
)


CONDITION = "public_lora_2603"
DOMAINS = ("pile", "finance")
METHOD_ID = "a1_a2_exhaustive_configuration_winner"
MODEL_SPEC = {
    "id": "meta-llama/Llama-3.2-1B-Instruct",
    "revision": "9213176726f574b556790deb65791e0c5aa438b6",
    "prefix_layers": [0, 1, 2, 3],
    "dtype": "bfloat16",
    "attention_implementation": "sdpa",
    "local_files_only": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--pile-root", type=Path, required=True)
    parser.add_argument("--finance-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path.resolve(strict=True))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_winner(path: Path) -> tuple[dict[str, Any], Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "token-reconstruction.trr0002-owner-r1-frozen-winner.v1"
        or payload.get("status")
        != "FROZEN_BEFORE_PUBLIC_HELDOUT_FRESH_BLIND_OR_CANONICAL_ACCESS"
        or payload.get("heldout_public_inputs") != 0
        or payload.get("fresh_blind_inputs") != 0
        or payload.get("canonical_evaluation_truth_inputs") != 0
    ):
        raise RuntimeError("winner is not the pre-heldout frozen artifact")
    policy = resolved_policy_from_dict(payload["policy"])
    if payload.get("policy_id") != policy.policy_id:
        raise RuntimeError("winner policy identity changed")
    return payload, policy


def load_domain(domain: str, pile_root: Path, finance_root: Path) -> dict[str, Any]:
    if domain == "pile":
        truth_path = pile_root / "truth.safetensors"
        truth = load_file(truth_path, device="cpu")["token_ids"].to(torch.long)
        mask = torch.ones(truth.shape, dtype=torch.long)
        positions = torch.arange(truth.shape[1], dtype=torch.long).view(1, -1).expand_as(truth)
        records = json.loads((pile_root / "records.json").read_text(encoding="utf-8"))["development"]
        observation_path = pile_root / "observations" / f"{CONDITION}_cut4.safetensors"
    elif domain == "finance":
        truth_path = finance_root / "truth.safetensors"
        truth_state = load_file(truth_path, device="cpu")
        truth = truth_state["token_ids"].to(torch.long)
        mask = truth_state["attention_mask"].to(torch.long)
        positions = truth_state["position_ids"].to(torch.long)
        records = json.loads((finance_root / "records.json").read_text(encoding="utf-8"))["records"]
        observation_path = finance_root / "observations" / f"{CONDITION}_cut4.safetensors"
    else:
        raise RuntimeError("unknown held-out domain")
    observations = load_file(observation_path, device="cpu")["activations"].contiguous()
    record_ids = [f"{CONDITION}:{row['record_id']}" for row in records]
    if observations.shape[:2] != truth.shape or len(record_ids) != truth.shape[0]:
        raise RuntimeError("public held-out geometry changed")
    if truth.shape[0] != 32 or not truth[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise RuntimeError("public held-out record count or BOS changed")
    validate_observations(observations, mask, positions)
    return {
        "observations": observations,
        "attention_mask": mask,
        "position_ids": positions,
        "truth": truth,
        "record_ids": record_ids,
        "observation_path": observation_path,
        "truth_path": truth_path,
    }


def counts(values: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    unique, frequencies = torch.unique(values[mask].to(torch.long), return_counts=True)
    return {
        str(int(key.item())): int(value.item())
        for key, value in zip(unique, frequencies, strict=True)
    }


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise RuntimeError("held-out output is create-only")
    args.output_root.mkdir(parents=True)
    started_utc = utc_now()
    seed_everything(20260824)
    winner_payload, policy = load_winner(args.winner)

    reference_path = repository_root / "reference" / "strict_bos" / "round001_teacher.py"
    reference = import_path("trr0002_r1_heldout_reference", reference_path)
    identity_path = (
        historical_root
        / "research"
        / "adaptive_a1_a2_strict_bos_20260817_goal_01a00b08"
        / "audit"
        / "AUDIT-0004-public-teacher-identity-v2.json"
    )
    lens_path = historical_root / "inversion_20260730" / "out" / "lens_alpaca.pt"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    precut, lens, embeddings, device, observed_identity = reference.load_public_teacher(
        MODEL_SPEC, identity, lens_path=lens_path
    )
    domains = {
        name: load_domain(name, args.pile_root, args.finance_root)
        for name in DOMAINS
    }
    tensors: dict[str, torch.Tensor] = {}
    scored: dict[str, Any] = {}
    max_k = max(policy.spec.schedule)
    torch.cuda.reset_peak_memory_stats(device)
    for domain, state in domains.items():
        proposal = propose_public_a1(
            observations=state["observations"],
            attention_mask=state["attention_mask"],
            lens=lens,
            normalized_embeddings=embeddings,
        )
        result = decode_policy(
            observations=state["observations"],
            attention_mask=state["attention_mask"],
            position_ids=state["position_ids"],
            candidates=proposal.candidates,
            a1_confidence=proposal.top1_confidence,
            precut=precut,
            device=device,
            policy=policy,
            record_batch_size=8,
        )
        candidate_view = proposal.candidates[:, :, :max_k].contiguous()
        metrics, per_record = score_predictions(
            predictions=result.predictions,
            truth=state["truth"],
            attention_mask=state["attention_mask"],
            candidates=candidate_view,
            record_ids=state["record_ids"],
        )
        mask = scored_mask(state["attention_mask"])
        scored[domain] = {
            "metrics": metrics,
            "per_record": per_record,
            "routes": counts(result.routes, mask),
            "selected_k": counts(result.selected_k, mask),
            "cost": {
                "proposal_seconds": proposal.elapsed_seconds,
                "selection_seconds": result.elapsed_seconds,
                "compute_seconds": proposal.elapsed_seconds + result.elapsed_seconds,
                "candidate_simulations": result.candidate_simulations,
                "executed_candidate_simulations": result.executed_candidate_simulations,
                "prefix_commit_tokens": result.prefix_commit_tokens,
                "record_batch_size": result.record_batch_size,
            },
        }
        tensors[f"{domain}.predictions"] = result.predictions.to(torch.int32)
        tensors[f"{domain}.candidates_k{max_k}"] = candidate_view.to(torch.int32)
        tensors[f"{domain}.proposal_top1_confidence"] = proposal.top1_confidence.float()
        tensors[f"{domain}.routes"] = result.routes.to(torch.int8)
        tensors[f"{domain}.selected_k"] = result.selected_k.to(torch.int16)
        tensors[f"{domain}.selected_signal"] = result.selected_signal.float()

    prediction_path = args.output_root / "predictions.safetensors"
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        prediction_path,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r1-public-heldout.v1",
            "method_id": METHOD_ID,
            "policy_id": policy.policy_id,
            "condition": CONDITION,
            "winner_revision_after_selection": "false",
        },
    )
    result_path = args.output_root / "result.json"
    result_payload = {
        "schema": "token-reconstruction.trr0002-owner-r1-public-heldout-result.v1",
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R1",
        "status": "FROZEN_WINNER_EVALUATED_ON_PUBLIC_HELDOUT_WITHOUT_REVISION",
        "method_id": METHOD_ID,
        "policy_id": policy.policy_id,
        "policy": policy.serialized(),
        "winner_revision_after_selection": False,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "command": command_record(),
        "exit_status": 0,
        "condition": CONDITION,
        "domains": scored,
        "winner": file_record(args.winner),
        "prediction": file_record(prediction_path),
        "inputs": {
            domain: {
                "observation": file_record(domains[domain]["observation_path"]),
                "truth": file_record(domains[domain]["truth_path"]),
                "records": [0, 32],
            }
            for domain in DOMAINS
        },
        "public_teacher_identity": observed_identity,
        "lens": file_record(lens_path),
        "peak_memory": peak_memory(),
    }
    write_json_exclusive(result_path, result_payload)
    print(
        json.dumps(
            {
                "status": result_payload["status"],
                "policy_id": policy.policy_id,
                "pile_accuracy": scored["pile"]["metrics"]["token_accuracy"],
                "finance_accuracy": scored["finance"]["metrics"]["token_accuracy"],
                "output": str(result_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
