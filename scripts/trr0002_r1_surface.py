#!/usr/bin/env python3
"""Build truth-prefix score surfaces for every frozen A1 candidate budget."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch

from token_reconstruction.a1a2_configuration_search import (
    BUDGET_GRID,
    ROUTING_SIGNALS,
    SCORE_RULES,
    _candidate_hidden,
    routing_signal,
    score_candidates,
)
from token_reconstruction.component_crossover import propose_public_a1
from token_reconstruction.dual_benchmark import BOS_TOKEN_ID, scored_mask, validate_observations
from token_reconstruction.experiment_runtime import (
    PhaseTimer,
    command_record,
    file_record,
    peak_memory,
    seed_everything,
    utc_now,
    write_json_exclusive,
)


CONDITIONS = ("public_base", "public_lora_2601")
DOMAINS = ("pile", "finance")
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


def load_domain(
    domain: str,
    condition: str,
    pile_root: Path,
    finance_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], Path, Path]:
    if domain == "pile":
        truth_path = pile_root / "truth.safetensors"
        state = load_file(truth_path, device="cpu")
        truth = state["token_ids"].to(torch.long).contiguous()
        mask = torch.ones(truth.shape, dtype=torch.long)
        positions = torch.arange(truth.shape[1], dtype=torch.long).view(1, -1).expand_as(truth)
        records = json.loads((pile_root / "records.json").read_text(encoding="utf-8"))[
            "development"
        ]
        record_ids = [str(row["record_id"]) for row in records]
        observation_path = pile_root / "observations" / f"{condition}_cut4.safetensors"
    elif domain == "finance":
        truth_path = finance_root / "truth.safetensors"
        state = load_file(truth_path, device="cpu")
        truth = state["token_ids"].to(torch.long).contiguous()
        mask = state["attention_mask"].to(torch.long).contiguous()
        positions = state["position_ids"].to(torch.long).contiguous()
        records = json.loads((finance_root / "records.json").read_text(encoding="utf-8"))[
            "records"
        ]
        record_ids = [str(row["record_id"]) for row in records]
        observation_path = finance_root / "observations" / f"{condition}_cut4.safetensors"
    else:
        raise RuntimeError("unknown public domain")
    observations = load_file(observation_path, device="cpu")["activations"].contiguous()
    if len(record_ids) != truth.shape[0] or observations.shape[:2] != truth.shape:
        raise RuntimeError("public domain records or observations changed")
    if not truth[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise RuntimeError("public domain truth lost BOS")
    validate_observations(observations, mask, positions)
    return observations, mask, positions, truth, record_ids, observation_path, truth_path


@torch.inference_mode()
def build_surface(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    truth: torch.Tensor,
    lens: torch.nn.Module,
    embeddings: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    proposal = propose_public_a1(
        observations=observations,
        attention_mask=attention_mask,
        lens=lens,
        normalized_embeddings=embeddings,
    )
    candidates = proposal.candidates.to(torch.long).contiguous()
    records, positions = attention_mask.shape
    budget_count = len(BUDGET_GRID)
    output: dict[str, torch.Tensor] = {
        "truth": truth.to(torch.int32).contiguous(),
        "attention_mask": attention_mask.to(torch.uint8).contiguous(),
        "position_ids": position_ids.to(torch.int32).contiguous(),
        "a1_candidates": candidates.to(torch.int32).contiguous(),
        "a1_confidence": proposal.top1_confidence.to(torch.float32).contiguous(),
        "true_rank": torch.zeros((records, positions), dtype=torch.int16),
    }
    for score_rule in SCORE_RULES:
        output[f"winner_rank.{score_rule}"] = torch.full(
            (records, positions, budget_count), -1, dtype=torch.int16
        )
        output[f"winner_token.{score_rule}"] = torch.full(
            (records, positions, budget_count), -1, dtype=torch.int32
        )
        for signal in ROUTING_SIGNALS:
            output[f"signal.{score_rule}.{signal}"] = torch.full(
                (records, positions, budget_count), float("nan"), dtype=torch.float32
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    simulations = 0
    for record in range(records):
        valid_length = int(attention_mask[record].sum().item())
        cache = precut.new_cache()
        precut.run_cached(
            torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long, device=device), cache, 0
        )
        for position in range(1, valid_length):
            ids = candidates[record, position].to(device=device).view(1, -1)
            hidden = _candidate_hidden(
                precut,
                cache=cache,
                parent_indices=torch.zeros(1, dtype=torch.long, device=device),
                candidate_ids=ids,
                position=position,
            )
            target = observations[record, position].to(device).float().view(1, -1)
            truth_token = int(truth[record, position].item())
            matches = ids[0].eq(truth_token)
            if matches.any().item():
                output["true_rank"][record, position] = int(
                    matches.to(torch.long).argmax().item()
                ) + 1
            for score_rule in SCORE_RULES:
                for budget_index, budget in enumerate(BUDGET_GRID):
                    if score_rule == "group_centered_cosine" and budget == 1:
                        continue
                    scores = score_candidates(hidden[:, :budget], target, score_rule)
                    winner = int(scores.argmax(dim=1).item())
                    output[f"winner_rank.{score_rule}"][
                        record, position, budget_index
                    ] = winner
                    output[f"winner_token.{score_rule}"][
                        record, position, budget_index
                    ] = int(ids[0, winner].item())
                    if budget >= 2:
                        for signal in ROUTING_SIGNALS:
                            output[f"signal.{score_rule}.{signal}"][
                                record, position, budget_index
                            ] = float(routing_signal(scores, signal).item())
            precut.run_cached(
                truth[record, position].to(device=device, dtype=torch.long).view(1, 1),
                cache,
                position,
            )
            simulations += 512
            del hidden, target
        del cache
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    true_rank = output["true_rank"][scored_mask(attention_mask)]
    recall = {
        str(k): float(true_rank.gt(0).logical_and(true_rank.le(k)).float().mean().item())
        for k in BUDGET_GRID
    }
    return output, {
        "proposal_seconds": proposal.elapsed_seconds,
        "surface_seconds": elapsed,
        "candidate_simulations": simulations,
        "scored_tokens": int(scored_mask(attention_mask).sum().item()),
        "candidate_recall": recall,
    }


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "token-reconstruction.trr0002-owner-r1-configuration-search-preregistration.v1":
        raise RuntimeError("configuration-search preregistration changed")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise RuntimeError("score-surface output is create-only")
    args.output_root.mkdir(parents=True)

    started_utc = utc_now()
    timer = PhaseTimer()
    seed_everything(20260823)
    torch.cuda.reset_peak_memory_stats()
    reference = import_path(
        "trr0002_r1_surface_reference",
        repository_root / "reference" / "strict_bos" / "round001_teacher.py",
    )
    identity_path = (
        historical_root
        / "research"
        / "adaptive_a1_a2_strict_bos_20260817_goal_01a00b08"
        / "audit"
        / "AUDIT-0004-public-teacher-identity-v2.json"
    )
    lens_path = historical_root / "inversion_20260730" / "out" / "lens_alpaca.pt"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    with timer.measure("load_pinned_public_teacher"):
        precut, lens, embeddings, device, observed_identity = reference.load_public_teacher(
            MODEL_SPEC,
            identity,
            lens_path=lens_path,
        )

    cells: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for condition in CONDITIONS:
            with timer.measure(f"surface_{domain}_{condition}"):
                (
                    observations,
                    mask,
                    positions,
                    truth,
                    record_ids,
                    observation_path,
                    truth_path,
                ) = load_domain(
                    domain,
                    condition,
                    args.pile_root,
                    args.finance_root,
                )
                surface, metrics = build_surface(
                    observations=observations,
                    attention_mask=mask,
                    position_ids=positions,
                    truth=truth,
                    lens=lens,
                    embeddings=embeddings,
                    precut=precut,
                    device=device,
                )
                surface_path = args.output_root / f"{domain}-{condition}.safetensors"
                save_file(
                    surface,
                    surface_path,
                    metadata={
                        "schema": "token-reconstruction.trr0002-owner-r1-truth-prefix-surface.v1",
                        "domain": domain,
                        "condition": condition,
                        "prefix": "public-truth-prefix-component-diagnostic",
                        "record_ids": json.dumps(record_ids, separators=(",", ":")),
                    },
                )
            cells.append(
                {
                    "domain": domain,
                    "condition": condition,
                    "surface": file_record(surface_path),
                    "observation": file_record(observation_path),
                    "truth": file_record(truth_path),
                    "metrics": metrics,
                }
            )

    evidence_path = args.output_root / "generation.json"
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-surface-generation.v1",
            "task_id": "TRR-0002",
            "revision_id": "TRR-0002-OWNER-REVISION-R1",
            "status": "PUBLIC_TRUTH_PREFIX_SURFACES_COMPLETE",
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "command": command_record(),
            "exit_status": 0,
            "plan": file_record(args.plan),
            "reference_source": file_record(
                repository_root / "reference" / "strict_bos" / "round001_teacher.py"
            ),
            "search_source": file_record(
                repository_root / "src" / "token_reconstruction" / "a1a2_configuration_search.py"
            ),
            "lens": file_record(lens_path),
            "identity_input": file_record(identity_path),
            "observed_public_teacher_identity": observed_identity,
            "cells": cells,
            "phases": timer.records,
            "peak_memory": peak_memory(),
            "canonical_evaluation_observation_inputs": 0,
            "canonical_evaluation_truth_inputs": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "PUBLIC_TRUTH_PREFIX_SURFACES_COMPLETE",
                "cells": len(cells),
                "output": str(evidence_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
