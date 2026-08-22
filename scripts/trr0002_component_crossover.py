#!/usr/bin/env python3
"""Run the preregistered TRR-0002 component crossover on both setups."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch
import transformers

import trr0001_r2_dual_benchmark as r2
from token_reconstruction.component_crossover import (
    BUDGETS,
    METHOD_IDS,
    ROUTE_ABSTAIN,
    SelectorResult,
    prediction_from_rank_one,
    propose_public_a1,
    propose_residual_affine,
    quantile_summary,
    rank_summary,
    round_robin_union,
    select_fixed_budget,
    selector_error_attribution,
    true_token_ranks,
)
from token_reconstruction.dual_benchmark import (
    SETUP_IDS,
    paired_record_differences,
    score_predictions,
    scored_mask,
)
from token_reconstruction.experiment_runtime import seed_everything
from token_reconstruction.inverse import load_inverse
from token_reconstruction.metrics import bootstrap_mean


NEW_SETUP, OLD_SETUP = SETUP_IDS
DIRECT = "direct_inverse_k16"
CAUSAL_BASELINE = "causal_public_surrogate_k16"
STRICT = "strict_bos_adaptive_a1_a2"
PRIMARY = "a1_causal_k32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--new-input-root", type=Path, required=True)
    parser.add_argument("--new-truth-jsonl", type=Path, required=True)
    parser.add_argument("--old-native-json", type=Path, required=True)
    parser.add_argument("--previous-prediction-artifact", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--prediction-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def batch_size_for_k(k: int) -> int:
    return {8: 16, 16: 16, 32: 8, 64: 4}[k]


def method_id(proposal: str, selector: str, k: int) -> str:
    if proposal == "residual_affine" and selector == "causal" and k == 16:
        return CAUSAL_BASELINE
    prefix = {
        ("a1", "a2_fixed_budget"): "a1_a2",
        ("a1", "causal"): "a1_causal",
        ("residual_affine", "a2_fixed_budget"): "residual_affine_a2",
        ("residual_affine", "causal"): "residual_affine_causal",
        ("a1_residual_union", "causal"): "a1_residual_union_causal",
    }[(proposal, selector)]
    return f"{prefix}_k{k}"


def validate_registry(plan: dict[str, Any], registry: dict[str, Any]) -> None:
    methods = [row["id"] for row in registry["methods"]]
    setups = [row["id"] for row in registry["setups"]]
    cells = {(row["setup_id"], row["method_id"]) for row in registry["required_cells"]}
    expected = {(setup, method) for setup in setups for method in methods}
    if methods != list(METHOD_IDS):
        raise RuntimeError("registry method order differs from frozen implementation")
    if setups != list(SETUP_IDS) or cells != expected:
        raise RuntimeError("registry is not the full setup-method Cartesian product")
    if len(methods) != 22 or len(cells) != 44:
        raise RuntimeError("preregistered matrix size changed")
    if plan["matrix"]["required_cells"] != 44:
        raise RuntimeError("plan matrix size changed")


def run_selectors(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
    a1: Any,
    residual: Any,
) -> tuple[dict[str, dict[str, Any]], dict[int, torch.Tensor], dict[int, float]]:
    methods: dict[str, dict[str, Any]] = {}
    unions: dict[int, torch.Tensor] = {}
    union_seconds: dict[int, float] = {}
    proposal_map = {
        "a1": a1,
        "residual_affine": residual,
    }
    for k in BUDGETS:
        started = time.perf_counter()
        unions[k] = round_robin_union(
            a1_candidates=a1.candidates,
            residual_candidates=residual.candidates,
            attention_mask=attention_mask,
            k=k,
        )
        union_seconds[k] = time.perf_counter() - started

    for proposal, selector in (
        ("a1", "a2_fixed_budget"),
        ("a1", "causal"),
        ("residual_affine", "a2_fixed_budget"),
        ("residual_affine", "causal"),
        ("a1_residual_union", "causal"),
    ):
        for k in BUDGETS:
            candidates = (
                unions[k]
                if proposal == "a1_residual_union"
                else proposal_map[proposal].candidates[:, :, :k].contiguous()
            )
            result = select_fixed_budget(
                observations=observations,
                attention_mask=attention_mask,
                position_ids=position_ids,
                candidates=candidates,
                precut=precut,
                device=device,
                selector=selector,
                record_batch_size=batch_size_for_k(k),
            )
            identifier = method_id(proposal, selector, k)
            if identifier in methods:
                raise RuntimeError(f"duplicate selector execution: {identifier}")
            if proposal == "a1":
                proposal_seconds = a1.elapsed_seconds
            elif proposal == "residual_affine":
                proposal_seconds = residual.elapsed_seconds
            else:
                proposal_seconds = (
                    a1.elapsed_seconds + residual.elapsed_seconds + union_seconds[k]
                )
            methods[identifier] = {
                "proposal": proposal,
                "selector": selector,
                "candidate_budget": k,
                "candidates": candidates,
                "result": result,
                "timing": {
                    "proposal_seconds": proposal_seconds,
                    "selection_seconds": result.elapsed_seconds,
                    "compute_seconds": proposal_seconds + result.elapsed_seconds,
                    "candidate_simulations": result.candidate_simulations,
                    "executed_candidate_simulations": result.executed_candidate_simulations,
                    "prefix_commit_tokens": result.prefix_commit_tokens,
                },
            }
    return methods, unions, union_seconds


def run_setup(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    reference: Any,
    lens: torch.nn.Module,
    embeddings: torch.Tensor,
    inverse: torch.nn.Module,
    precut: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    a1 = propose_public_a1(
        observations=observations,
        attention_mask=attention_mask,
        lens=lens,
        normalized_embeddings=embeddings,
    )
    residual = propose_residual_affine(
        observations=observations,
        attention_mask=attention_mask,
        inverse=inverse,
        embedding_table=embeddings,
    )
    methods, unions, union_seconds = run_selectors(
        observations=observations,
        attention_mask=attention_mask,
        position_ids=position_ids,
        precut=precut,
        device=device,
        a1=a1,
        residual=residual,
    )
    strict_predictions, strict_candidates, strict_routes, strict_timing = r2.strict_decode(
        reference=reference,
        observations=observations,
        attention_mask=attention_mask,
        position_ids=position_ids,
        lens=lens,
        embeddings=embeddings,
        precut=precut,
        device=device,
    )
    methods[DIRECT] = {
        "proposal": "residual_affine",
        "selector": "rank_one",
        "candidate_budget": 16,
        "candidates": residual.candidates[:, :, :16].contiguous(),
        "result": None,
        "predictions": prediction_from_rank_one(residual.candidates, attention_mask),
        "timing": {
            "proposal_seconds": residual.elapsed_seconds,
            "selection_seconds": 0.0,
            "compute_seconds": residual.elapsed_seconds,
            "candidate_simulations": 0,
            "executed_candidate_simulations": 0,
            "prefix_commit_tokens": 0,
        },
    }
    methods[STRICT] = {
        "proposal": "a1",
        "selector": "strict_adaptive_a1_a2",
        "candidate_budget": 512,
        "candidates": strict_candidates,
        "result": None,
        "predictions": strict_predictions,
        "routes": strict_routes,
        "timing": {
            **strict_timing,
            "executed_candidate_simulations": strict_timing["candidate_simulations"],
            "prefix_commit_tokens": None,
        },
    }
    if set(methods) != set(METHOD_IDS):
        missing = sorted(set(METHOD_IDS) - set(methods))
        extra = sorted(set(methods) - set(METHOD_IDS))
        raise RuntimeError(f"method matrix differs: missing={missing} extra={extra}")
    return {
        "methods": methods,
        "a1": a1,
        "residual": residual,
        "unions": unions,
        "union_seconds": union_seconds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def predictions_for(state: dict[str, Any]) -> torch.Tensor:
    result = state.get("result")
    return result.predictions if isinstance(result, SelectorResult) else state["predictions"]


def freeze_predictions(
    *,
    setups: dict[str, dict[str, Any]],
    path: Path,
    execution_commit: str,
) -> dict[str, Any]:
    tensors: dict[str, torch.Tensor] = {}
    aliases = {NEW_SETUP: "clean", OLD_SETUP: "historical"}
    for setup_id, setup in setups.items():
        prefix = aliases[setup_id]
        tensors[f"{prefix}.proposal.a1.candidates_k512"] = (
            setup["a1"].candidates.to(torch.int32).contiguous()
        )
        tensors[f"{prefix}.proposal.a1.top1_confidence"] = (
            setup["a1"].top1_confidence.float().contiguous()
        )
        tensors[f"{prefix}.proposal.residual.candidates_k512"] = (
            setup["residual"].candidates.to(torch.int32).contiguous()
        )
        tensors[f"{prefix}.proposal.residual.top1_confidence"] = (
            setup["residual"].top1_confidence.float().contiguous()
        )
        for k, candidates in setup["unions"].items():
            tensors[f"{prefix}.proposal.union.candidates_k{k}"] = (
                candidates.to(torch.int32).contiguous()
            )
        for identifier, state in setup["methods"].items():
            tensors[f"{prefix}.method.{identifier}.predictions"] = (
                predictions_for(state).to(torch.int32).contiguous()
            )
            result = state.get("result")
            if isinstance(result, SelectorResult):
                tensors[f"{prefix}.method.{identifier}.winner_margin"] = (
                    result.winner_margin.float().contiguous()
                )
                tensors[f"{prefix}.method.{identifier}.normalized_winner"] = (
                    result.normalized_winner.float().contiguous()
                )
                tensors[f"{prefix}.method.{identifier}.routes"] = (
                    result.routes.to(torch.int8).contiguous()
                )
            elif identifier == STRICT:
                tensors[f"{prefix}.method.{identifier}.routes"] = (
                    state["routes"].to(torch.int8).contiguous()
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        path,
        metadata={
            "schema": "token-reconstruction.trr0002-component-crossover-freeze.v1",
            "task_id": "TRR-0002",
            "execution_commit": execution_commit,
            "truth_status": "already-open-retrospective-but-loaded-after-freeze",
            "method_count": str(len(METHOD_IDS)),
            "setup_count": str(len(SETUP_IDS)),
        },
    )
    return r2.file_record(path)


def verify_baseline_reproduction(
    setups: dict[str, dict[str, Any]],
    previous_path: Path,
) -> dict[str, Any]:
    previous = load_file(previous_path, device="cpu")
    mapping = {
        NEW_SETUP: {DIRECT: "new.direct.predictions", CAUSAL_BASELINE: "new.causal.predictions", STRICT: "new.strict.predictions"},
        OLD_SETUP: {DIRECT: "old.direct.predictions", CAUSAL_BASELINE: "old.causal.predictions", STRICT: "old.strict.predictions"},
    }
    checks: dict[str, Any] = {}
    for setup_id, by_method in mapping.items():
        checks[setup_id] = {}
        for identifier, key in by_method.items():
            current = predictions_for(setups[setup_id]["methods"][identifier]).to(torch.long)
            expected = previous[key].to(torch.long)
            mismatches = int(current.ne(expected).sum().item())
            checks[setup_id][identifier] = {
                "compared_values": int(current.numel()),
                "prediction_mismatches": mismatches,
            }
            if mismatches:
                raise RuntimeError(
                    f"baseline reproduction failed for {setup_id}/{identifier}: {mismatches}"
                )
    return checks


def first_abstention_summary(routes: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
    positions: list[int] = []
    rows_without = 0
    for row in range(routes.shape[0]):
        valid = attention_mask[row].to(torch.bool)
        found = torch.nonzero((routes[row] == ROUTE_ABSTAIN) & valid, as_tuple=False)
        if found.numel():
            positions.append(int(found[0].item()))
        else:
            rows_without += 1
    return {
        "records_with_abstention": len(positions),
        "records_without_abstention": rows_without,
        "first_abstention_position": quantile_summary(torch.tensor(positions, dtype=torch.float32)),
    }


def margin_diagnostics(
    *,
    result: SelectorResult,
    predictions: torch.Tensor,
    truth: torch.Tensor,
    attention_mask: torch.Tensor,
    candidates: torch.Tensor,
) -> dict[str, Any]:
    mask = scored_mask(attention_mask)
    expected = truth[mask].to(torch.long)
    predicted = predictions[mask].to(torch.long)
    included = candidates[mask].to(torch.long).eq(expected[:, None]).any(dim=1)
    correct = predicted.eq(expected) & predicted.ge(0)
    covered = predicted.ge(0)
    margins = result.winner_margin[mask]
    normalized = result.normalized_winner[mask]
    finite = torch.isfinite(margins) & torch.isfinite(normalized)
    return {
        "winner_margin": {
            "all_evaluated": quantile_summary(margins[finite]),
            "correct": quantile_summary(margins[finite & correct]),
            "incorrect_or_abstained": quantile_summary(margins[finite & ~correct]),
            "true_token_in_candidates": quantile_summary(margins[finite & included]),
            "true_token_excluded": quantile_summary(margins[finite & ~included]),
        },
        "normalized_winner": {
            "all_evaluated": quantile_summary(normalized[finite]),
            "accepted": quantile_summary(normalized[finite & covered]),
            "abstained": quantile_summary(normalized[finite & ~covered]),
            "correct": quantile_summary(normalized[finite & correct]),
            "incorrect_or_abstained": quantile_summary(normalized[finite & ~correct]),
        },
        "first_abstention": first_abstention_summary(result.routes, attention_mask),
    }


def proposal_diagnostics(
    *,
    setup: dict[str, Any],
    truth: torch.Tensor,
) -> dict[str, Any]:
    mask = setup["attention_mask"]
    output: dict[str, Any] = {}
    for name, proposal in (("a1", setup["a1"]), ("residual_affine", setup["residual"])):
        ranks = true_token_ranks(
            candidates=proposal.candidates,
            truth=truth,
            attention_mask=mask,
        )
        output[name] = {
            "true_token_rank": rank_summary(ranks, mask),
            "top1_confidence": quantile_summary(proposal.top1_confidence[scored_mask(mask)]),
            "proposal_seconds": proposal.elapsed_seconds,
        }
    output["a1_residual_union"] = {}
    for k, candidates in setup["unions"].items():
        ranks = true_token_ranks(candidates=candidates, truth=truth, attention_mask=mask)
        output["a1_residual_union"][str(k)] = {
            "true_token_rank": rank_summary(ranks, mask, budgets=tuple(value for value in BUDGETS if value <= k)),
            "construction_seconds": setup["union_seconds"][k],
        }
    return output


def score_setup(
    *,
    setup: dict[str, Any],
    truth: torch.Tensor,
    record_ids: list[str],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    matrix: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for identifier in METHOD_IDS:
        state = setup["methods"][identifier]
        predictions = predictions_for(state)
        metrics, per_record = score_predictions(
            predictions=predictions,
            truth=truth,
            attention_mask=setup["attention_mask"],
            candidates=state["candidates"],
            record_ids=record_ids,
        )
        attribution = selector_error_attribution(
            predictions=predictions,
            truth=truth,
            attention_mask=setup["attention_mask"],
            candidates=state["candidates"],
        )
        diagnostics: dict[str, Any] = {
            "error_attribution": attribution,
        }
        result = state.get("result")
        if isinstance(result, SelectorResult):
            diagnostics["confidence_and_margin"] = margin_diagnostics(
                result=result,
                predictions=predictions,
                truth=truth,
                attention_mask=setup["attention_mask"],
                candidates=state["candidates"],
            )
        matrix[identifier] = {
            "status": "retrospective_complete",
            "proposal_family": state["proposal"],
            "selector_family": state["selector"],
            "candidate_budget": state["candidate_budget"],
            "metrics": metrics,
            "per_record": per_record,
            "timing": state["timing"],
            "diagnostics": diagnostics,
        }
        rows[identifier] = per_record
    return matrix, rows


def comparisons_for_setup(
    rows: dict[str, list[dict[str, Any]]],
    reference_method: str,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for identifier in METHOD_IDS:
        if identifier == reference_method:
            continue
        output[f"{identifier}_minus_{reference_method}"] = bootstrap_mean(
            paired_record_differences(rows[identifier], rows[reference_method]),
            draws=10000,
            seed=seed,
        )
    return output


def main() -> int:
    args = parse_args()
    started_utc = r2.utc_now()
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    if args.prediction_artifact.exists() or args.output.exists():
        raise RuntimeError("TRR-0002 outputs are create-only")
    plan = load_json(args.plan)
    registry = load_json(args.registry)
    validate_registry(plan, registry)
    seed_everything(1729)
    execution_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()

    reference_path = repository_root / "reference" / "strict_bos" / "round001_teacher.py"
    reference = r2.import_path("trr0002_strict_reference", reference_path)
    source_path = historical_root / "scripts" / "score_a1_a2_source300_20260809.py"
    source300 = r2.import_path("trr0002_source300", source_path)
    new_observations, new_mask, new_positions = r2.new_inputs(args.new_input_root.resolve(strict=True))
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
    identity = load_json(identity_path)
    precut, lens, embeddings, device, observed_identity = reference.load_public_teacher(
        r2.MODEL_SPEC,
        identity,
        lens_path=lens_path,
    )
    inverse_path = args.new_input_root / "inverses" / "cut4.safetensors"
    inverse = load_inverse(inverse_path, hidden_size=2048, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    setups = {
        NEW_SETUP: run_setup(
            observations=new_observations,
            attention_mask=new_mask,
            position_ids=new_positions,
            reference=reference,
            lens=lens,
            embeddings=embeddings,
            inverse=inverse,
            precut=precut,
            device=device,
        ),
        OLD_SETUP: run_setup(
            observations=old_observations,
            attention_mask=old_mask,
            position_ids=old_positions,
            reference=reference,
            lens=lens,
            embeddings=embeddings,
            inverse=inverse,
            precut=precut,
            device=device,
        ),
    }
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    prediction_record = freeze_predictions(
        setups=setups,
        path=args.prediction_artifact,
        execution_commit=execution_commit,
    )
    reproduction = verify_baseline_reproduction(setups, args.previous_prediction_artifact)

    # Truth is loaded only after every prediction and diagnostic tensor is frozen.
    new_truth, new_ids = r2.load_new_truth(args.new_truth_jsonl, 64)
    old_truth, old_ids = r2.load_old_truth(source300, old_captures, old_config)
    truth_by_setup = {NEW_SETUP: new_truth, OLD_SETUP: old_truth}
    ids_by_setup = {NEW_SETUP: new_ids, OLD_SETUP: old_ids}

    matrix: dict[str, Any] = {}
    per_record: dict[str, dict[str, list[dict[str, Any]]]] = {}
    proposal_analysis: dict[str, Any] = {}
    for setup_id in SETUP_IDS:
        matrix[setup_id], per_record[setup_id] = score_setup(
            setup=setups[setup_id],
            truth=truth_by_setup[setup_id],
            record_ids=ids_by_setup[setup_id],
        )
        proposal_analysis[setup_id] = proposal_diagnostics(
            setup=setups[setup_id],
            truth=truth_by_setup[setup_id],
        )

    comparisons = {
        NEW_SETUP: comparisons_for_setup(per_record[NEW_SETUP], CAUSAL_BASELINE, 3201),
        OLD_SETUP: comparisons_for_setup(per_record[OLD_SETUP], STRICT, 3202),
    }
    primary_clean = matrix[NEW_SETUP][PRIMARY]["metrics"]["correct_tokens"]
    primary_old = matrix[OLD_SETUP][PRIMARY]["metrics"]["correct_tokens"]
    primary_pass = (
        (primary_clean > 2096 and primary_old >= 13741)
        or (primary_clean >= 2096 and primary_old > 13741)
    )
    completed_cells = sum(len(matrix[setup]) for setup in SETUP_IDS)
    if completed_cells != 44:
        raise RuntimeError(f"completed cell count changed: {completed_cells}")

    ended_utc = r2.utc_now()
    payload = {
        "schema": "token-reconstruction.trr0002-component-crossover-result.v1",
        "task_id": "TRR-0002",
        "status": "RETROSPECTIVE_COMPLETE_CROSSOVER",
        "execution_commit": execution_commit,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "truth_status": {
            "classification": "already-open retrospective",
            "prediction_artifact_written_and_hashed_before_runner_loaded_truth": True,
            "claim_limit": "component diagnosis, not fresh blind confirmation",
        },
        "matrix_complete": True,
        "required_cells": 44,
        "completed_cells": completed_cells,
        "setup_order": list(SETUP_IDS),
        "method_order": list(METHOD_IDS),
        "matrix": matrix,
        "proposal_rank_and_confidence": proposal_analysis,
        "paired_record_bootstrap": comparisons,
        "primary_hypothesis": {
            "method_id": PRIMARY,
            "clean_correct_tokens": primary_clean,
            "clean_required_strictly_greater_than": 2096,
            "historical_correct_tokens": primary_old,
            "historical_required_at_least": 13741,
            "dominance_pass": primary_pass,
        },
        "baseline_reproduction": reproduction,
        "cost_scope": {
            "model_load_excluded_from_method_compute_seconds": True,
            "file_io_excluded_from_method_compute_seconds": True,
            "cuda_synchronized_at_proposal_and_selection_boundaries": True,
            "shared_proposal_cost_reported_as_full_standalone_cost_per_method": True,
            "peak_cuda_memory_allocated_bytes": peak_memory_bytes,
            "target_model_calls": 0,
            "training_or_adaptation": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "pid": os.getpid(),
        },
        "public_teacher_identity": observed_identity,
        "artifacts": {
            "prediction_freeze": prediction_record,
            "plan": r2.file_record(args.plan),
            "registry": r2.file_record(args.registry),
            "previous_prediction_artifact": r2.file_record(args.previous_prediction_artifact),
            "new_observation": r2.file_record(
                args.new_input_root / "observations" / "unavailable_target_lora_cut4.safetensors"
            ),
            "new_inverse": r2.file_record(inverse_path),
            "new_truth": r2.file_record(args.new_truth_jsonl),
            "old_source": r2.file_record(source300.resolve_inside_ersoy(old_config["source"]["path"])),
            "old_native_rerun": r2.file_record(args.old_native_json),
            "strict_lens": r2.file_record(lens_path),
            "strict_reference_source": r2.file_record(reference_path),
            "crossover_primitive_source": r2.file_record(
                repository_root / "src" / "token_reconstruction" / "component_crossover.py"
            ),
            "runner_source": r2.file_record(Path(__file__)),
        },
    }
    r2.write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "completed_cells": completed_cells,
                "primary": payload["primary_hypothesis"],
                "clean_accuracy": {
                    method: matrix[NEW_SETUP][method]["metrics"]["token_accuracy"]
                    for method in METHOD_IDS
                },
                "historical_accuracy": {
                    method: matrix[OLD_SETUP][method]["metrics"]["token_accuracy"]
                    for method in METHOD_IDS
                },
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
