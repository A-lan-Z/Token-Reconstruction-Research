#!/usr/bin/env python3
"""Evaluate the complete preregistered A1+A2 public score surface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from safetensors.torch import load_file
import torch

from token_reconstruction.a1a2_configuration_search import (
    BUDGET_GRID,
    EMPIRICAL_GATE_MODES,
    PolicySpec,
    ROUTING_SIGNALS,
    SCORE_RULES,
    current_anchor_spec,
    declared_policy_count,
    historical_anchor_spec,
    iter_policy_specs,
    resolve_policy,
)
from token_reconstruction.experiment_runtime import (
    command_record,
    file_record,
    peak_memory,
    utc_now,
    write_json_exclusive,
)


DOMAINS = ("pile", "finance")
CONDITIONS = ("public_base", "public_lora_2601")
FIT_RECORDS = range(0, 16)
QUANTILES = {
    "fit_quantile_0.05": 0.05,
    "fit_quantile_0.10": 0.10,
    "fit_quantile_0.20": 0.20,
    "fit_quantile_0.35": 0.35,
    "fit_quantile_0.50": 0.50,
    "fit_quantile_0.65": 0.65,
    "fit_quantile_0.80": 0.80,
    "fit_quantile_0.90": 0.90,
    "fit_quantile_0.95": 0.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DomainSurface:
    truth: np.ndarray
    a1_top: np.ndarray
    a1_confidence: np.ndarray
    record_index: np.ndarray
    record_slices: tuple[tuple[int, int], ...]
    winners: Mapping[str, Mapping[int, np.ndarray]]
    signals: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]]

    @property
    def scored_tokens(self) -> int:
        return int(self.truth.size)

    @property
    def records(self) -> int:
        return len(self.record_slices)


def load_domain_surface(root: Path, domain: str) -> DomainSurface:
    truth_parts: list[np.ndarray] = []
    a1_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    record_parts: list[np.ndarray] = []
    winner_parts: dict[str, dict[int, list[np.ndarray]]] = {
        score: {k: [] for k in BUDGET_GRID} for score in SCORE_RULES
    }
    signal_parts: dict[str, dict[str, dict[int, list[np.ndarray]]]] = {
        score: {
            signal: {k: [] for k in BUDGET_GRID if k >= 2}
            for signal in ROUTING_SIGNALS
        }
        for score in SCORE_RULES
    }
    slices: list[tuple[int, int]] = []
    flat_offset = 0
    trajectory_index = 0
    for condition in CONDITIONS:
        state = load_file(root / f"{domain}-{condition}.safetensors", device="cpu")
        truth_tensor = state["truth"].to(torch.long)
        mask_tensor = state["attention_mask"].to(torch.bool)
        candidates = state["a1_candidates"].to(torch.long)
        confidence = state["a1_confidence"].to(torch.float32)
        for record in FIT_RECORDS:
            mask = mask_tensor[record].clone()
            mask[0] = False
            count = int(mask.sum().item())
            if count <= 0:
                raise RuntimeError("public fit trajectory has no scored tokens")
            truth_parts.append(truth_tensor[record][mask].numpy())
            a1_parts.append(candidates[record, :, 0][mask].numpy())
            confidence_parts.append(confidence[record][mask].numpy())
            record_parts.append(np.full(count, trajectory_index, dtype=np.int32))
            slices.append((flat_offset, flat_offset + count))
            flat_offset += count
            trajectory_index += 1
            for score in SCORE_RULES:
                winner_tensor = state[f"winner_token.{score}"]
                for budget_index, budget in enumerate(BUDGET_GRID):
                    if score == "group_centered_cosine" and budget == 1:
                        continue
                    winner_parts[score][budget].append(
                        winner_tensor[record, :, budget_index][mask].to(torch.long).numpy()
                    )
                for signal in ROUTING_SIGNALS:
                    signal_tensor = state[f"signal.{score}.{signal}"]
                    for budget_index, budget in enumerate(BUDGET_GRID):
                        if budget < 2:
                            continue
                        signal_parts[score][signal][budget].append(
                            signal_tensor[record, :, budget_index][mask]
                            .to(torch.float32)
                            .numpy()
                        )
    winners = {
        score: {
            budget: np.concatenate(parts)
            for budget, parts in by_budget.items()
            if parts
        }
        for score, by_budget in winner_parts.items()
    }
    signals = {
        score: {
            signal: {
                budget: np.concatenate(parts)
                for budget, parts in by_budget.items()
                if parts
            }
            for signal, by_budget in by_signal.items()
        }
        for score, by_signal in signal_parts.items()
    }
    value = DomainSurface(
        truth=np.concatenate(truth_parts),
        a1_top=np.concatenate(a1_parts),
        a1_confidence=np.concatenate(confidence_parts),
        record_index=np.concatenate(record_parts),
        record_slices=tuple(slices),
        winners=winners,
        signals=signals,
    )
    if not np.isfinite(value.a1_confidence).all():
        raise RuntimeError("public fit A1 confidence is non-finite")
    return value


def fit_thresholds(domains: Mapping[str, DomainSurface]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for score in SCORE_RULES:
        output[score] = {}
        for signal in ROUTING_SIGNALS:
            output[score][signal] = {}
            for budget in BUDGET_GRID:
                if budget < 2:
                    continue
                values = np.concatenate(
                    [domains[domain].signals[score][signal][budget] for domain in DOMAINS]
                ).astype(np.float64)
                if not np.isfinite(values).all():
                    raise RuntimeError("public fit routing values are non-finite")
                output[score][signal][str(budget)] = {
                    mode: float(np.quantile(values, quantile, method="linear"))
                    for mode, quantile in QUANTILES.items()
                }
    return output


def apply_suffix_stop(
    prediction: np.ndarray,
    selected_k: np.ndarray,
    abstained: np.ndarray,
    record_slices: tuple[tuple[int, int], ...],
) -> None:
    for start, stop in record_slices:
        offsets = np.flatnonzero(abstained[start:stop])
        if offsets.size:
            suffix_start = start + int(offsets[0]) + 1
            prediction[suffix_start:stop] = -1
            selected_k[suffix_start:stop] = 0


def evaluate_domain(
    spec: PolicySpec,
    policy: Any,
    data: DomainSurface,
) -> dict[str, Any]:
    count = data.scored_tokens
    prediction = np.full(count, -1, dtype=np.int64)
    selected_k = np.zeros(count, dtype=np.int32)
    eligible = np.ones(count, dtype=bool)
    if spec.fast_path_threshold is not None:
        fast = data.a1_confidence >= spec.fast_path_threshold
        prediction[fast] = data.a1_top[fast]
        eligible &= ~fast
    if spec.kind == "fixed":
        budget = spec.schedule[0]
        prediction[eligible] = data.winners[spec.score_rule][budget][eligible]
        selected_k[eligible] = budget
    else:
        abstained = np.zeros(count, dtype=bool)
        for stage_index, budget in enumerate(spec.schedule):
            if not eligible.any():
                break
            values = data.signals[spec.score_rule][spec.routing_signal][budget]
            threshold = policy.threshold_at(budget)
            passes = values >= threshold if spec.gate_comparator == "ge" else values > threshold
            last = stage_index == len(spec.schedule) - 1
            accept = passes.copy()
            if last and spec.terminal_action == "commit_last_winner":
                accept[:] = True
            take = eligible & accept
            prediction[take] = data.winners[spec.score_rule][budget][take]
            selected_k[take] = budget
            remaining = eligible & ~accept
            if last and remaining.any():
                selected_k[remaining] = budget
                if spec.terminal_action == "fallback_to_a1":
                    prediction[remaining] = data.a1_top[remaining]
                elif spec.terminal_action == "abstain_and_stop_suffix":
                    abstained[remaining] = True
                else:
                    raise RuntimeError("unhandled terminal behavior")
                remaining[:] = False
            eligible = remaining
        if abstained.any():
            apply_suffix_stop(prediction, selected_k, abstained, data.record_slices)
    correct = prediction == data.truth
    record_errors = np.bincount(
        data.record_index,
        weights=(~correct).astype(np.int64),
        minlength=data.records,
    )
    correct_tokens = int(correct.sum())
    exact_records = int((record_errors == 0).sum())
    simulations = int(selected_k.sum())
    return {
        "correct_tokens": correct_tokens,
        "scored_tokens": count,
        "token_accuracy": correct_tokens / count,
        "exact_records": exact_records,
        "records": data.records,
        "exact_record_fraction": exact_records / data.records,
        "covered_tokens": int((prediction >= 0).sum()),
        "candidate_simulations": simulations,
        "simulations_per_1000": 1000.0 * simulations / count,
    }


def ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    pile = row["metrics"]["pile"]
    finance = row["metrics"]["finance"]
    minimum_accuracy = min(pile["token_accuracy"], finance["token_accuracy"])
    mean_accuracy = (pile["token_accuracy"] + finance["token_accuracy"]) / 2.0
    minimum_exact = min(pile["exact_record_fraction"], finance["exact_record_fraction"])
    simulations = pile["simulations_per_1000"] + finance["simulations_per_1000"]
    spec = row["policy"]["spec"]
    policy_number = int(row["policy_id"].split("_", 1)[1], 16)
    return (
        minimum_accuracy,
        mean_accuracy,
        minimum_exact,
        -simulations,
        -len(spec["schedule"]),
        -max(spec["schedule"]),
        -policy_number,
    )


def compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    spec = row["policy"]["spec"]
    return {
        "id": row["policy_id"],
        "kind": spec["kind"],
        "score": spec["score_rule"],
        "schedule": spec["schedule"],
        "fast": spec["fast_path_id"],
        "signal": spec["routing_signal"],
        "gate": spec["gate_mode"],
        "terminal": spec["terminal_action"],
        "pile": row["metrics"]["pile"],
        "finance": row["metrics"]["finance"],
    }


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "token-reconstruction.trr0002-owner-r1-configuration-search-preregistration.v1":
        raise RuntimeError("configuration-search preregistration changed")
    if int(plan["search_space"]["declared_distinct_policy_count_before_runtime_deduplication"]) != declared_policy_count():
        raise RuntimeError("declared configuration count differs from code")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise RuntimeError("configuration-search output is create-only")
    args.output_root.mkdir(parents=True)
    started_utc = utc_now()
    started = time.perf_counter()

    domains = {domain: load_domain_surface(args.surface_root, domain) for domain in DOMAINS}
    fitted = fit_thresholds(domains)
    thresholds_path = args.output_root / "fitted_thresholds.json"
    write_json_exclusive(
        thresholds_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-fitted-thresholds.v1",
            "source_roles": {
                "domains": list(DOMAINS),
                "conditions": list(CONDITIONS),
                "records": [0, 16],
                "canonical_inputs": 0,
            },
            "quantiles": QUANTILES,
            "thresholds": fitted,
        },
    )

    raw_path = args.output_root / "all_policies.jsonl.gz"
    global_heap: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
    best_structural: dict[tuple[str, str, str], dict[str, Any]] = {}
    best_fixed: dict[tuple[str, str], dict[str, Any]] = {}
    controls: dict[str, dict[str, Any]] = {}
    identifiers: set[str] = set()
    enumerated = 0
    current_id = current_anchor_spec().policy_id
    historical_id = historical_anchor_spec().policy_id
    with gzip.open(raw_path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as raw:
        for spec in iter_policy_specs():
            spec.validate()
            policy = resolve_policy(spec, fitted)
            if policy.policy_id in identifiers:
                raise RuntimeError("duplicate policy ID in full search")
            identifiers.add(policy.policy_id)
            metrics = {
                domain: evaluate_domain(spec, policy, domains[domain])
                for domain in DOMAINS
            }
            row = {
                "policy_id": policy.policy_id,
                "policy": policy.serialized(),
                "metrics": metrics,
            }
            raw.write(json.dumps(compact_row(row), sort_keys=True, separators=(",", ":")))
            raw.write("\n")
            key = ranking_key(row)
            enumerated += 1
            if len(global_heap) < 12:
                heapq.heappush(global_heap, (key, enumerated, row))
            elif key > global_heap[0][0]:
                heapq.heapreplace(global_heap, (key, enumerated, row))
            if spec.kind == "fixed":
                group = (spec.score_rule, spec.fast_path_id)
                old = best_fixed.get(group)
                if old is None or key > ranking_key(old):
                    best_fixed[group] = row
                if spec.fast_path_id == "off":
                    controls[policy.policy_id] = row
            else:
                group = (spec.score_rule, str(spec.routing_signal), spec.terminal_action)
                old = best_structural.get(group)
                if old is None or key > ranking_key(old):
                    best_structural[group] = row
            if policy.policy_id in {current_id, historical_id}:
                controls[policy.policy_id] = row
            if enumerated % 25000 == 0:
                print(
                    json.dumps(
                        {
                            "status": "SEARCH_PROGRESS",
                            "evaluated": enumerated,
                            "declared": declared_policy_count(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if enumerated != declared_policy_count() or len(identifiers) != enumerated:
        raise RuntimeError("full configuration enumeration changed")

    finalists_by_id: dict[str, dict[str, Any]] = {}
    finalist_sources: dict[str, set[str]] = {}
    groups = (
        ("global_top12", [row for _, _, row in global_heap]),
        ("structural_best", list(best_structural.values())),
        ("fixed_fast_group_best", list(best_fixed.values())),
        ("anchors_and_fixed_curve", list(controls.values())),
    )
    for source, rows in groups:
        for row in rows:
            identifier = row["policy_id"]
            finalists_by_id[identifier] = row
            finalist_sources.setdefault(identifier, set()).add(source)
    if len(finalists_by_id) > int(plan["finalist_rule"]["maximum"]):
        raise RuntimeError("deterministic finalist union exceeded preregistered maximum")
    finalists = sorted(finalists_by_id.values(), key=ranking_key, reverse=True)
    finalists_path = args.output_root / "finalists.json"
    write_json_exclusive(
        finalists_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-public-finalists.v1",
            "status": "FROZEN_FOR_EXACT_CAUSAL_SELECTION",
            "count": len(finalists),
            "maximum": int(plan["finalist_rule"]["maximum"]),
            "winner_not_selected_here": True,
            "policies": [
                {
                    **row,
                    "surface_rank_key": list(ranking_key(row)),
                    "finalist_sources": sorted(finalist_sources[row["policy_id"]]),
                }
                for row in finalists
            ],
        },
    )

    ended_utc = utc_now()
    result_path = args.output_root / "result.json"
    write_json_exclusive(
        result_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-public-surface-search.v1",
            "task_id": "TRR-0002",
            "revision_id": "TRR-0002-OWNER-REVISION-R1",
            "status": "COMPLETE_PUBLIC_SURFACE_NO_CANONICAL_INPUTS",
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "elapsed_seconds": time.perf_counter() - started,
            "command": command_record(),
            "exit_status": 0,
            "plan": file_record(args.plan),
            "surface_generation": file_record(args.surface_root / "generation.json"),
            "fitted_thresholds": file_record(thresholds_path),
            "raw_full_policy_table": {
                "path": str(raw_path),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
                "rows": enumerated,
                "compression": "gzip",
            },
            "finalists": file_record(finalists_path),
            "finalist_count": len(finalists),
            "surface_roles": {
                "conditions": list(CONDITIONS),
                "records_per_condition_per_domain": 16,
                "truth_prefix_component_diagnostic": True,
            },
            "domain_geometry": {
                domain: {
                    "trajectories": domains[domain].records,
                    "scored_tokens": domains[domain].scored_tokens,
                }
                for domain in DOMAINS
            },
            "enumeration": {
                "declared": declared_policy_count(),
                "evaluated": enumerated,
                "unique_policy_ids": len(identifiers),
                "complete": True,
            },
            "canonical_evaluation_observation_inputs": 0,
            "canonical_evaluation_truth_inputs": 0,
            "peak_memory": peak_memory(),
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_PUBLIC_SURFACE_NO_CANONICAL_INPUTS",
                "evaluated": enumerated,
                "finalists": len(finalists),
                "raw_sha256": sha256_file(raw_path),
                "output": str(result_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
