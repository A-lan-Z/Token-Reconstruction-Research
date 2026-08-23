#!/usr/bin/env python3
"""Select the frozen A1+A2 winner on untouched public causal trajectories."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

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


CONDITION = "public_lora_2602"
DOMAINS = ("pile", "finance")
SELECTION_START = 16
SELECTION_STOP = 32
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pile-root", type=Path, required=True)
    parser.add_argument("--finance-root", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timing-passes", type=int, default=3)
    return parser.parse_args()


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path.resolve(strict=True))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_domain(
    domain: str,
    pile_root: Path,
    finance_root: Path,
) -> dict[str, Any]:
    selection = slice(SELECTION_START, SELECTION_STOP)
    if domain == "pile":
        truth_state = load_file(pile_root / "truth.safetensors", device="cpu")
        truth = truth_state["token_ids"].to(torch.long)[selection].contiguous()
        mask = torch.ones(truth.shape, dtype=torch.long)
        positions = torch.arange(truth.shape[1], dtype=torch.long).view(1, -1).expand_as(truth)
        records = json.loads((pile_root / "records.json").read_text(encoding="utf-8"))[
            "development"
        ][selection]
        observation_path = pile_root / "observations" / f"{CONDITION}_cut4.safetensors"
        truth_path = pile_root / "truth.safetensors"
    elif domain == "finance":
        truth_state = load_file(finance_root / "truth.safetensors", device="cpu")
        truth = truth_state["token_ids"].to(torch.long)[selection].contiguous()
        mask = truth_state["attention_mask"].to(torch.long)[selection].contiguous()
        positions = truth_state["position_ids"].to(torch.long)[selection].contiguous()
        records = json.loads((finance_root / "records.json").read_text(encoding="utf-8"))[
            "records"
        ][selection]
        observation_path = finance_root / "observations" / f"{CONDITION}_cut4.safetensors"
        truth_path = finance_root / "truth.safetensors"
    else:
        raise RuntimeError("unknown public domain")
    observations = load_file(observation_path, device="cpu")["activations"][selection].contiguous()
    record_ids = [f"{CONDITION}:{row['record_id']}" for row in records]
    if observations.shape[:2] != truth.shape or len(record_ids) != truth.shape[0]:
        raise RuntimeError("public causal selection geometry changed")
    if not truth[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise RuntimeError("public causal truth lost BOS")
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


def counts(values: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, int]:
    selected = values if mask is None else values[mask]
    unique, frequencies = torch.unique(selected.to(torch.long), return_counts=True)
    return {
        str(int(key.item())): int(value.item())
        for key, value in zip(unique, frequencies, strict=True)
    }


def winner_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    pile = row["domains"]["pile"]
    finance = row["domains"]["finance"]
    minimum_accuracy = min(
        pile["metrics"]["token_accuracy"], finance["metrics"]["token_accuracy"]
    )
    mean_accuracy = (
        pile["metrics"]["token_accuracy"] + finance["metrics"]["token_accuracy"]
    ) / 2.0
    pile_exact = pile["metrics"]["exact_records"] / pile["metrics"]["records"]
    finance_exact = finance["metrics"]["exact_records"] / finance["metrics"]["records"]
    minimum_exact = min(pile_exact, finance_exact)
    mean_exact = (pile_exact + finance_exact) / 2.0
    total_tokens = pile["metrics"]["scored_tokens"] + finance["metrics"]["scored_tokens"]
    total_runtime = pile["timing"]["median_selection_seconds"] + finance["timing"][
        "median_selection_seconds"
    ]
    seconds_per_1000 = 1000.0 * total_runtime / total_tokens
    peak = max(pile["timing"]["peak_cuda_allocated_bytes"], finance["timing"]["peak_cuda_allocated_bytes"])
    simulations = pile["timing"]["candidate_simulations"] + finance["timing"][
        "candidate_simulations"
    ]
    simulations_per_1000 = 1000.0 * simulations / total_tokens
    spec = row["policy"]["spec"]
    policy_number = int(row["policy_id"].split("_", 1)[1], 16)
    return (
        minimum_accuracy,
        mean_accuracy,
        minimum_exact,
        mean_exact,
        -seconds_per_1000,
        -peak,
        -simulations_per_1000,
        -len(spec["schedule"]),
        -max(spec["schedule"]),
        -policy_number,
    )


def main() -> int:
    args = parse_args()
    if args.timing_passes < 1 or args.timing_passes > 5:
        raise RuntimeError("timing passes must be between one and five")
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    finalists_payload = json.loads(
        (args.search_root / "finalists.json").read_text(encoding="utf-8")
    )
    if plan.get("schema") != "token-reconstruction.trr0002-owner-r1-configuration-search-preregistration.v1":
        raise RuntimeError("configuration-search preregistration changed")
    if finalists_payload.get("status") != "FROZEN_FOR_EXACT_CAUSAL_SELECTION":
        raise RuntimeError("public finalist set is not frozen")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise RuntimeError("causal-selection output is create-only")
    args.output_root.mkdir(parents=True)
    started_utc = utc_now()
    seed_everything(20260823)

    reference_path = repository_root / "reference" / "strict_bos" / "round001_teacher.py"
    reference = import_path("trr0002_r1_causal_reference", reference_path)
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
        MODEL_SPEC,
        identity,
        lens_path=lens_path,
    )

    domains = {domain: load_domain(domain, args.pile_root, args.finance_root) for domain in DOMAINS}
    proposal_evidence: dict[str, Any] = {}
    for domain, state in domains.items():
        proposal = propose_public_a1(
            observations=state["observations"],
            attention_mask=state["attention_mask"],
            lens=lens,
            normalized_embeddings=embeddings,
        )
        state["candidates"] = proposal.candidates
        state["a1_confidence"] = proposal.top1_confidence
        proposal_evidence[domain] = {
            "seconds": proposal.elapsed_seconds,
            "scored_tokens": int(scored_mask(state["attention_mask"]).sum().item()),
        }

    prediction_tensors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for policy_index, finalist in enumerate(finalists_payload["policies"]):
        policy = resolved_policy_from_dict(finalist["policy"])
        row: dict[str, Any] = {
            "policy_id": policy.policy_id,
            "policy": policy.serialized(),
            "surface_metrics": finalist["metrics"],
            "finalist_sources": finalist["finalist_sources"],
            "domains": {},
        }
        for domain in DOMAINS:
            state = domains[domain]
            pass_results = []
            elapsed = []
            peaks = []
            for _ in range(args.timing_passes):
                torch.cuda.reset_peak_memory_stats(device)
                result = decode_policy(
                    observations=state["observations"],
                    attention_mask=state["attention_mask"],
                    position_ids=state["position_ids"],
                    candidates=state["candidates"],
                    a1_confidence=state["a1_confidence"],
                    precut=precut,
                    device=device,
                    policy=policy,
                )
                pass_results.append(result)
                elapsed.append(result.elapsed_seconds)
                peaks.append(int(torch.cuda.max_memory_allocated(device)))
            baseline = pass_results[0]
            for repeated in pass_results[1:]:
                if not torch.equal(repeated.predictions, baseline.predictions):
                    raise RuntimeError("policy predictions changed across timing passes")
                if not torch.equal(repeated.routes, baseline.routes):
                    raise RuntimeError("policy routes changed across timing passes")
                if repeated.candidate_simulations != baseline.candidate_simulations:
                    raise RuntimeError("policy simulation count changed across timing passes")
            metrics, per_record = score_predictions(
                predictions=baseline.predictions,
                truth=state["truth"],
                attention_mask=state["attention_mask"],
                candidates=state["candidates"],
                record_ids=state["record_ids"],
            )
            mask = scored_mask(state["attention_mask"])
            row["domains"][domain] = {
                "metrics": metrics,
                "per_record": per_record,
                "timing": {
                    "passes": args.timing_passes,
                    "selection_seconds": elapsed,
                    "median_selection_seconds": statistics.median(elapsed),
                    "minimum_selection_seconds": min(elapsed),
                    "peak_cuda_allocated_bytes_by_pass": peaks,
                    "peak_cuda_allocated_bytes": max(peaks),
                    "candidate_simulations": baseline.candidate_simulations,
                    "executed_candidate_simulations": baseline.executed_candidate_simulations,
                    "prefix_commit_tokens": baseline.prefix_commit_tokens,
                    "record_batch_size": baseline.record_batch_size,
                },
                "routes": counts(baseline.routes, mask),
                "selected_k": counts(baseline.selected_k, mask),
            }
            prefix = f"{policy.policy_id}.{domain}"
            prediction_tensors[f"{prefix}.predictions"] = baseline.predictions.to(torch.int32)
            prediction_tensors[f"{prefix}.routes"] = baseline.routes.to(torch.int8)
            prediction_tensors[f"{prefix}.selected_k"] = baseline.selected_k.to(torch.int16)
        row["winner_key"] = list(winner_key(row))
        rows.append(row)
        print(
            json.dumps(
                {
                    "status": "CAUSAL_SELECTION_PROGRESS",
                    "completed": policy_index + 1,
                    "total": len(finalists_payload["policies"]),
                    "policy_id": policy.policy_id,
                    "pile_accuracy": row["domains"]["pile"]["metrics"]["token_accuracy"],
                    "finance_accuracy": row["domains"]["finance"]["metrics"]["token_accuracy"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len({row["policy_id"] for row in rows}) != len(rows):
        raise RuntimeError("causal finalist policy IDs are not unique")
    ordered = sorted(rows, key=winner_key, reverse=True)
    winner = ordered[0]
    prediction_path = args.output_root / "predictions.safetensors"
    save_file(
        {name: tensor.contiguous() for name, tensor in prediction_tensors.items()},
        prediction_path,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r1-public-causal-finalists.v1",
            "condition": CONDITION,
            "truth_status": "public-auxiliary",
        },
    )
    table_path = args.output_root / "table.json"
    write_json_exclusive(
        table_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-public-causal-table.v1",
            "winner_rule": plan["phase_2_exact_causal_selection"]["winner_rule"],
            "rows": ordered,
        },
    )
    winner_path = args.output_root / "winner.json"
    write_json_exclusive(
        winner_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-frozen-winner.v1",
            "task_id": "TRR-0002",
            "revision_id": "TRR-0002-OWNER-REVISION-R1",
            "status": "FROZEN_BEFORE_PUBLIC_HELDOUT_FRESH_BLIND_OR_CANONICAL_ACCESS",
            "created_utc": utc_now(),
            "policy_id": winner["policy_id"],
            "policy": winner["policy"],
            "winner_key": winner["winner_key"],
            "selection_metrics": winner["domains"],
            "selection_condition": CONDITION,
            "selection_records": [SELECTION_START, SELECTION_STOP],
            "plan": file_record(args.plan),
            "finalists": file_record(args.search_root / "finalists.json"),
            "fitted_thresholds": file_record(args.search_root / "fitted_thresholds.json"),
            "table": file_record(table_path),
            "predictions": file_record(prediction_path),
            "method_source": file_record(
                repository_root / "src" / "token_reconstruction" / "a1a2_configuration_search.py"
            ),
            "runner_source": file_record(repository_root / "scripts" / "trr0002_r1_causal_select.py"),
            "lens": file_record(lens_path),
            "public_teacher_identity": observed_identity,
            "canonical_evaluation_observation_inputs": 0,
            "canonical_evaluation_truth_inputs": 0,
            "heldout_public_inputs": 0,
            "fresh_blind_inputs": 0,
            "revision_policy": "immutable; held-out or blind failure weakens the claim and cannot replace this policy",
        },
    )
    result_path = args.output_root / "result.json"
    write_json_exclusive(
        result_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-public-causal-selection.v1",
            "status": "WINNER_FROZEN_NO_CANONICAL_INPUTS",
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "command": command_record(),
            "exit_status": 0,
            "timing_passes": args.timing_passes,
            "finalist_count": len(rows),
            "winner_policy_id": winner["policy_id"],
            "winner": file_record(winner_path),
            "table": file_record(table_path),
            "predictions": file_record(prediction_path),
            "proposal": proposal_evidence,
            "inputs": {
                domain: {
                    "observation": file_record(domains[domain]["observation_path"]),
                    "truth": file_record(domains[domain]["truth_path"]),
                    "records": [SELECTION_START, SELECTION_STOP],
                }
                for domain in DOMAINS
            },
            "canonical_evaluation_observation_inputs": 0,
            "canonical_evaluation_truth_inputs": 0,
            "peak_memory": peak_memory(),
        },
    )
    print(
        json.dumps(
            {
                "status": "WINNER_FROZEN_NO_CANONICAL_INPUTS",
                "winner": winner["policy_id"],
                "pile_accuracy": winner["domains"]["pile"]["metrics"]["token_accuracy"],
                "finance_accuracy": winner["domains"]["finance"]["metrics"]["token_accuracy"],
                "output": str(result_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
