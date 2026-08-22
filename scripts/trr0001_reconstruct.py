#!/usr/bin/env python3
"""Truth-blind reconstruction process for both frozen TRR-0001 baselines."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from token_reconstruction.experiment_runtime import (
    BOS_TOKEN_ID,
    CONDITIONS,
    CUT_DEPTHS,
    MODEL_ID,
    MODEL_REVISION,
    PhaseTimer,
    command_record,
    file_record,
    load_json,
    peak_memory,
    require_create_only_directory,
    require_plan,
    seed_everything,
    sha256_file,
    synchronize,
    utc_now,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from token_reconstruction.inverse import (
    load_inverse,
    normalized_embeddings,
    topk_candidates,
)
from token_reconstruction.public_prefix import (
    ContiguousPublicPrefix,
    PublicPrefixCache,
)
from token_reconstruction.separation import ReconstructionInputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--inverse-directory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    return parser.parse_args()


def load_public_model(model_id: str, revision: str) -> torch.nn.Module:
    if model_id != MODEL_ID or revision != MODEL_REVISION:
        raise RuntimeError("public model identity is not the preregistered pin")
    if not torch.cuda.is_available():
        raise RuntimeError("TRR-0001 reconstruction requires CUDA")
    return (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(torch.device("cuda"))
        .eval()
    )


def copy_exclusive(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"method-state source is invalid: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def stable_candidate_order(
    candidate_ids: torch.Tensor, candidate_scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    ordered_ids = torch.empty_like(candidate_ids)
    ordered_scores = torch.empty_like(candidate_scores)
    for row in range(candidate_ids.shape[0]):
        order = sorted(
            range(candidate_ids.shape[1]),
            key=lambda column: (
                -float(candidate_scores[row, column]),
                int(candidate_ids[row, column]),
            ),
        )
        index = torch.tensor(order, dtype=torch.long)
        ordered_ids[row] = candidate_ids[row].index_select(0, index)
        ordered_scores[row] = candidate_scores[row].index_select(0, index)
    return ordered_ids, ordered_scores


@torch.inference_mode()
def causal_search(
    *,
    model: torch.nn.Module,
    cut_depth: int,
    candidates: torch.Tensor,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score frozen candidates under only the reconstructed public prefix."""

    if candidates.shape[:2] != (64, 39) or candidates.shape[2] != 16:
        raise RuntimeError("frozen candidate geometry changed")
    if observations.shape[:2] != (64, 40):
        raise RuntimeError("blind observation geometry changed")
    device = next(model.parameters()).device
    prefix = ContiguousPublicPrefix(model, cut_depth)
    selected = torch.empty((64, 39), dtype=torch.long)
    all_scores = torch.empty((64, 39, 16), dtype=torch.float32)
    if cut_depth == 0:
        table = model.get_input_embeddings().weight
        for position in range(39):
            ids = candidates[:, position, :].to(device)
            simulated = prefix.embed_tokens(ids).float()
            target = observations[:, position + 1, :].to(device).float()
            score = F.cosine_similarity(simulated, target[:, None, :], dim=-1)
            choice = score.argmax(dim=-1)
            selected[:, position] = ids.gather(1, choice[:, None]).squeeze(1).cpu()
            all_scores[:, position, :] = score.cpu()
        return selected, all_scores

    for record_start in range(0, 64, 16):
        record_end = min(record_start + 16, 64)
        records = record_end - record_start
        cache = prefix.new_cache()
        bos = torch.full(
            (records, 1), BOS_TOKEN_ID, dtype=torch.long, device=device
        )
        prefix.run_cached(bos, cache, 0)
        for position in range(39):
            ids = candidates[record_start:record_end, position, :].to(device)
            fork_backend = copy.deepcopy(cache.backend)
            fork_backend.batch_repeat_interleave(16)
            fork = PublicPrefixCache(backend=fork_backend, length=cache.length)
            simulated = prefix.run_cached(ids.reshape(-1, 1), fork, position + 1)
            simulated = simulated[:, -1, :].reshape(records, 16, -1).float()
            target = observations[
                record_start:record_end, position + 1, :
            ].to(device).float()
            score = F.cosine_similarity(simulated, target[:, None, :], dim=-1)
            choice = score.argmax(dim=-1)
            winner = ids.gather(1, choice[:, None])
            selected[record_start:record_end, position] = winner.squeeze(1).cpu()
            all_scores[record_start:record_end, position, :] = score.cpu()
            prefix.run_cached(winner, cache, position + 1)
            del fork, fork_backend, simulated, target, score
        del cache
        torch.cuda.empty_cache()
    return selected, all_scores


def main() -> int:
    args = parse_args()
    inputs = ReconstructionInputs(
        observation_index=args.observation_index,
        inverse_directory=args.inverse_directory,
        plan_path=args.plan,
        output_directory=args.output_directory,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    inputs.validate()
    plan = load_json(inputs.plan_path)
    require_plan(plan)
    index = load_json(inputs.observation_index)
    if index.get("schema") != "token-reconstruction.observation-index.v1":
        raise RuntimeError("observation index schema changed")
    if index.get("source_tokens_or_text_included") is not False:
        raise RuntimeError("observation index does not certify truth exclusion")
    records = index.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("blind record order changed")
    for record in records:
        if set(record) != {"record_id", "dataset_index", "text_sha256"}:
            raise RuntimeError("observation record exposes unexpected fields")

    entries = {
        (entry["condition"], int(entry["cut_depth"])): entry
        for entry in index.get("entries", [])
    }
    if set(entries) != {
        (condition, cut) for condition in CONDITIONS for cut in CUT_DEPTHS
    }:
        raise RuntimeError("observation arm coverage changed")

    seed_everything(1729)
    torch.cuda.reset_peak_memory_stats()
    timer = PhaseTimer()
    started_utc = utc_now()
    with timer.measure("load_pinned_public_surrogate"):
        model = load_public_model(inputs.model_id, inputs.model_revision)
    device = next(model.parameters()).device
    embedding_table = normalized_embeddings(
        model.get_input_embeddings().weight
    ).to(device)

    observations: dict[tuple[str, int], torch.Tensor] = {}
    observation_records: list[dict[str, Any]] = []
    with timer.measure("load_permitted_boundary_observations"):
        for key, entry in entries.items():
            path = inputs.observation_index.parent / entry["path"]
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"observation path is invalid: {path}")
            if sha256_file(path) != entry["artifact"]["sha256"]:
                raise RuntimeError(f"observation hash changed: {path}")
            state = load_file(path, device="cpu")
            if set(state) != {"activations"}:
                raise RuntimeError("observation tensor fields changed")
            value = state["activations"]
            if tuple(value.shape) != (64, 40, 2048):
                raise RuntimeError("observation tensor shape changed")
            observations[key] = value
            observation_records.append(file_record(path))

    inverses: dict[int, torch.nn.Module] = {}
    with timer.measure("load_frozen_public_inverse_states"):
        for cut in (4, 8):
            path = inputs.inverse_directory / f"cut{cut}.safetensors"
            inverses[cut] = load_inverse(
                path, hidden_size=2048, device=device
            )

    require_create_only_directory(inputs.output_directory)
    method_state = inputs.output_directory / "method_state"
    method_state.mkdir()
    copied_state: list[dict[str, Any]] = []
    for cut in (4, 8):
        source = inputs.inverse_directory / f"cut{cut}.safetensors"
        destination = method_state / source.name
        copy_exclusive(source, destination)
        copied_state.append(file_record(destination, root=inputs.output_directory))
    frequency_source = (
        inputs.inverse_directory.parent / "auxiliary_frequency_counts.json"
    )
    frequency_destination = method_state / frequency_source.name
    copy_exclusive(frequency_source, frequency_destination)
    copied_state.append(
        file_record(frequency_destination, root=inputs.output_directory)
    )

    all_rows: list[dict[str, Any]] = []
    frozen_queries: dict[str, torch.Tensor] = {}
    arm_timings: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for cut in CUT_DEPTHS:
            arm = f"{condition}.cut{cut}"
            observed = observations[(condition, cut)]
            synchronize()
            proposal_start = time.perf_counter()
            with torch.inference_mode():
                flat = observed[:, 1:, :].reshape(-1, 2048).to(device)
                if cut == 0:
                    queries = F.normalize(flat.float(), dim=-1)
                else:
                    queries = inverses[cut](flat)
                candidate_ids, candidate_scores = topk_candidates(
                    queries, embedding_table, k=16, score_batch_size=64
                )
            candidate_ids, candidate_scores = stable_candidate_order(
                candidate_ids, candidate_scores
            )
            synchronize()
            proposal_seconds = time.perf_counter() - proposal_start
            shaped_candidates = candidate_ids.reshape(64, 39, 16)
            shaped_direct_scores = candidate_scores.reshape(64, 39, 16)
            direct_tokens = shaped_candidates[:, :, 0].clone()
            frozen_queries[arm] = queries.detach().reshape(64, 39, 2048).to(
                device="cpu", dtype=torch.float32
            )
            del flat, queries

            synchronize()
            causal_start = time.perf_counter()
            causal_tokens, causal_scores = causal_search(
                model=model,
                cut_depth=cut,
                candidates=shaped_candidates,
                observations=observed,
            )
            synchronize()
            causal_seconds = time.perf_counter() - causal_start
            arm_timings.append(
                {
                    "condition": condition,
                    "cut_depth": cut,
                    "direct_proposal_seconds": proposal_seconds,
                    "causal_reconstruction_seconds": causal_seconds,
                    "direct_amortized_seconds_per_record": proposal_seconds / 64,
                    "causal_amortized_seconds_per_record": causal_seconds / 64,
                    "direct_seconds_per_scored_token": proposal_seconds / (64 * 39),
                    "causal_seconds_per_scored_token": causal_seconds / (64 * 39),
                }
            )
            for record_index, record in enumerate(records):
                all_rows.append(
                    {
                        "condition": condition,
                        "cut_depth": cut,
                        "record_index": record_index,
                        "record_id": record["record_id"],
                        "direct_tokens": direct_tokens[record_index].tolist(),
                        "causal_tokens": causal_tokens[record_index].tolist(),
                        "candidate_ids": shaped_candidates[record_index].tolist(),
                        "direct_candidate_scores": shaped_direct_scores[
                            record_index
                        ].tolist(),
                        "causal_candidate_scores": causal_scores[
                            record_index
                        ].tolist(),
                        "candidate_budget": 16,
                        "scored_tokens": 39,
                        "abstained_tokens": 0,
                        "direct_amortized_seconds": proposal_seconds / 64,
                        "causal_amortized_seconds": causal_seconds / 64,
                    }
                )
            del candidate_ids, candidate_scores, shaped_candidates
            del shaped_direct_scores, direct_tokens, causal_tokens, causal_scores
            torch.cuda.empty_cache()

    queries_path = inputs.output_directory / "queries.safetensors"
    save_file(
        {key: value.contiguous() for key, value in frozen_queries.items()},
        queries_path,
        metadata={
            "schema": "token-reconstruction.frozen-inverse-queries.v1",
            "truth_opened": "false",
        },
    )
    outputs_path = inputs.output_directory / "reconstructions.jsonl"
    write_jsonl_exclusive(outputs_path, all_rows)
    route_path = inputs.output_directory / "route.json"
    write_json_exclusive(
        route_path,
        {
            "schema": "token-reconstruction.trr0001-frozen-route.v1",
            "task_id": "TRR-0001",
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "plan": file_record(inputs.plan_path),
            "observation_index": file_record(inputs.observation_index),
            "observations": observation_records,
            "method_state": copied_state,
            "condition_order": list(CONDITIONS),
            "cut_order": list(CUT_DEPTHS),
            "record_order": [record["record_id"] for record in records],
            "methods": ["direct_inverse", "causal_public_surrogate_search"],
            "candidate_budget": 16,
            "stopping": "all 39 scored positions",
            "target_prefix_calls": 0,
            "truth_or_correctness_inputs": 0,
            "implementation": {
                "direct": "normalized public embedding full-vocabulary scoring",
                "causal": "batched DynamicCache fork, candidate simulation, winner-only commit",
                "ties": "score, then frozen proposal rank, then token ID ascending",
            },
        },
    )
    evidence_path = inputs.output_directory / "reconstructor_evidence.json"
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr0001-reconstructor-evidence.v1",
            "task_id": "TRR-0001",
            "command": command_record(),
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "exit_status": 0,
            "phases": timer.records,
            "arm_timings": arm_timings,
            "blind_records": 64,
            "scored_tokens_per_arm": 64 * 39,
            "arms": len(CONDITIONS) * len(CUT_DEPTHS),
            "direct_embedding_comparisons": 64 * 39 * 128256 * 6,
            "causal_candidate_simulations": 64 * 39 * 16 * 6,
            "public_surrogate_model_loads": 1,
            "target_prefix_calls": 0,
            "peak_memory": peak_memory(),
        },
    )
    print(
        {
            "status": "reconstructed_truth_blind",
            "rows": len(all_rows),
            "queries": str(queries_path),
            "outputs": str(outputs_path),
            "target_prefix_calls": 0,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
