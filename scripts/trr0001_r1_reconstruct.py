#!/usr/bin/env python3
"""One-method, config-only reconstruction process for TRR-0001-R1."""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
from pathlib import Path
import time
from typing import Any, Callable

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from token_reconstruction.blind_commitment import (
    validate_observation_index,
    validate_sanitized_config,
)
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
    read_jsonl,
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
from token_reconstruction.isolation import validate_isolation_manifest
from token_reconstruction.public_prefix import (
    ContiguousPublicPrefix,
    PublicPrefixCache,
)


METHODS = ("direct_inverse", "causal_public_surrogate_search")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--access-manifest", type=Path, required=True)
    return parser.parse_args()


def public_path(root: Path, relative: str) -> Path:
    if relative.startswith("/") or ".." in relative.split("/"):
        raise RuntimeError("public artifact path must remain relative")
    root_resolved = root.resolve()
    path = (root_resolved / relative).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError("public artifact escaped its input root") from exc
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"public artifact is not a regular file: {relative}")
    return path


def load_public_model(model_path: Path, config: dict) -> torch.nn.Module:
    if config["model"]["id"] != MODEL_ID or config["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("pinned public model identity changed")
    if not torch.cuda.is_available():
        raise RuntimeError("TRR-0001-R1 reconstruction requires CUDA")
    if model_path.resolve() != Path(
        f"/model-repo/snapshots/{MODEL_REVISION}"
    ):
        raise RuntimeError("isolated model mount path changed")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(torch.device("cuda"))
        .eval()
    )
    if model.config.hidden_size != 2048 or model.config.vocab_size != 128256:
        raise RuntimeError("pinned public model geometry changed")
    model.requires_grad_(False)
    return model


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
def propose_candidates(
    *,
    cut_depth: int,
    observations: torch.Tensor,
    inverses: dict[int, torch.nn.Module],
    embedding_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = observations[:, 1:, :].reshape(-1, 2048).to(embedding_table.device)
    if cut_depth == 0:
        queries = F.normalize(flat.float(), dim=-1)
    else:
        queries = inverses[cut_depth](flat)
    candidate_ids, candidate_scores = topk_candidates(
        queries, embedding_table, k=16, score_batch_size=64
    )
    candidate_ids, candidate_scores = stable_candidate_order(
        candidate_ids, candidate_scores
    )
    return (
        queries.detach().reshape(64, 39, 2048).to(device="cpu", dtype=torch.float32),
        candidate_ids.reshape(64, 39, 16),
        candidate_scores.reshape(64, 39, 16),
    )


@torch.inference_mode()
def causal_search(
    *,
    model: torch.nn.Module,
    cut_depth: int,
    candidates: torch.Tensor,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidates.shape != (64, 39, 16):
        raise RuntimeError("candidate geometry changed")
    if observations.shape != (64, 40, 2048):
        raise RuntimeError("observation geometry changed")
    device = next(model.parameters()).device
    prefix = ContiguousPublicPrefix(model, cut_depth)
    selected = torch.empty((64, 39), dtype=torch.long)
    all_scores = torch.empty((64, 39, 16), dtype=torch.float32)
    if cut_depth == 0:
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
        record_count = record_end - record_start
        cache = prefix.new_cache()
        prefix.run_cached(
            torch.full(
                (record_count, 1), BOS_TOKEN_ID, dtype=torch.long, device=device
            ),
            cache,
            0,
        )
        for position in range(39):
            ids = candidates[record_start:record_end, position, :].to(device)
            fork_backend = copy.deepcopy(cache.backend)
            fork_backend.batch_repeat_interleave(16)
            fork = PublicPrefixCache(backend=fork_backend, length=cache.length)
            simulated = prefix.run_cached(ids.reshape(-1, 1), fork, position + 1)
            simulated = simulated[:, -1, :].reshape(record_count, 16, -1).float()
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


def function_complexity(function: Callable[..., Any]) -> dict[str, int]:
    source = inspect.getsource(function)
    tree = ast.parse(inspect.cleandoc(source))
    return {
        "source_lines": len(source.splitlines()),
        "ast_statements": sum(isinstance(node, ast.stmt) for node in ast.walk(tree)),
    }


def implementation_complexity(method: str) -> dict[str, Any]:
    common = {
        "stable_candidate_order": function_complexity(stable_candidate_order),
        "propose_candidates": function_complexity(propose_candidates),
    }
    specific = (
        {"causal_search": function_complexity(causal_search)}
        if method == "causal_public_surrogate_search"
        else {"direct_top1_selection": {"source_lines": 1, "ast_statements": 1}}
    )
    dependencies = {}
    for module in (__import__("token_reconstruction.inverse", fromlist=["x"]), __import__("token_reconstruction.public_prefix", fromlist=["x"])):
        path = Path(inspect.getsourcefile(module) or "")
        source = path.read_text(encoding="utf-8")
        dependencies[path.name] = {
            "source_lines": len(source.splitlines()),
            "sha256": sha256_file(path),
        }
    return {"common": common, "method_specific": specific, "dependencies": dependencies}


def main() -> int:
    args = parse_args()
    started_utc = utc_now()
    output_root = args.output_directory.resolve()
    if output_root.is_symlink() or not output_root.is_dir():
        raise RuntimeError("isolated output must be a regular mounted directory")
    if {path.name for path in output_root.iterdir()} != {args.access_manifest.name}:
        raise RuntimeError("isolated method output was not empty before its access manifest")
    access = load_json(args.access_manifest)
    validate_isolation_manifest(access, method=args.method)

    input_root = args.input_root.resolve()
    if args.config.resolve() != input_root / "sanitized_config.json":
        raise RuntimeError("only the sanitized reconstruction config may be supplied")
    config = load_json(args.config)
    validate_sanitized_config(config)
    observation_index_path = public_path(
        input_root, config["observation_index"]["path"]
    )
    if (
        observation_index_path.stat().st_size != config["observation_index"]["bytes"]
        or sha256_file(observation_index_path) != config["observation_index"]["sha256"]
    ):
        raise RuntimeError("observation index bytes changed")
    index = load_json(observation_index_path)
    validate_observation_index(index)
    if [row["record_id"] for row in index["records"]] != config["record_order"]:
        raise RuntimeError("opaque record order differs between sanitized inputs")
    records = index["records"]

    seed_everything(int(config["execution"]["seed"]))
    timer = PhaseTimer()
    with timer.measure("load_pinned_public_surrogate"):
        model = load_public_model(args.model_path, config)
    device = next(model.parameters()).device
    embedding_table = normalized_embeddings(
        model.get_input_embeddings().weight
    ).to(device)

    entries = {
        (entry["condition"], int(entry["cut_depth"])): entry
        for entry in index["entries"]
    }
    expected_arms = {
        (condition, cut) for condition in CONDITIONS for cut in CUT_DEPTHS
    }
    if set(entries) != expected_arms:
        raise RuntimeError("sanitized observation arm coverage changed")
    observations: dict[tuple[str, int], torch.Tensor] = {}
    observation_artifacts = []
    with timer.measure("load_sanitized_boundary_observations"):
        for key in sorted(entries):
            entry = entries[key]
            path = public_path(input_root, entry["path"])
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise RuntimeError("boundary observation bytes changed")
            state = load_file(path, device="cpu")
            if set(state) != {"activations"} or tuple(state["activations"].shape) != (64, 40, 2048):
                raise RuntimeError("boundary observation tensor changed")
            observations[key] = state["activations"]
            observation_artifacts.append(file_record(path))

    inverses: dict[int, torch.nn.Module] = {}
    inverse_artifacts = []
    with timer.measure("load_exact_public_inverse_states"):
        for entry in config["inverse_states"]:
            cut = int(entry["cut_depth"])
            path = public_path(input_root, entry["path"])
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise RuntimeError("public inverse bytes changed")
            inverses[cut] = load_inverse(path, hidden_size=2048, device=device)
            inverse_artifacts.append(file_record(path))

    preparation_peak = peak_memory()
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    method_started_utc = utc_now()
    method_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    frozen_queries: dict[str, torch.Tensor] = {}
    arm_timings: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for cut in CUT_DEPTHS:
            arm = f"{condition}.cut{cut}"
            observed = observations[(condition, cut)]
            synchronize()
            arm_started = time.perf_counter()
            proposal_started = time.perf_counter()
            queries, candidates, proposal_scores = propose_candidates(
                cut_depth=cut,
                observations=observed,
                inverses=inverses,
                embedding_table=embedding_table,
            )
            synchronize()
            proposal_seconds = time.perf_counter() - proposal_started
            selection_seconds = 0.0
            if args.method == "direct_inverse":
                prediction = candidates[:, :, 0].clone()
                selection_scores = proposal_scores
            else:
                synchronize()
                selection_started = time.perf_counter()
                prediction, selection_scores = causal_search(
                    model=model,
                    cut_depth=cut,
                    candidates=candidates,
                    observations=observed,
                )
                synchronize()
                selection_seconds = time.perf_counter() - selection_started
            frozen_queries[arm] = queries
            for record_index, record in enumerate(records):
                rows.append(
                    {
                        "condition": condition,
                        "cut_depth": cut,
                        "record_index": record_index,
                        "record_id": record["record_id"],
                        "method": args.method,
                        "prediction_tokens": prediction[record_index].tolist(),
                        "candidate_ids": candidates[record_index].tolist(),
                        "proposal_scores": proposal_scores[record_index].tolist(),
                        "selection_scores": selection_scores[record_index].tolist(),
                        "candidate_budget": 16,
                        "scored_tokens": 39,
                        "abstained_tokens": 0,
                        "proposal_amortized_seconds": proposal_seconds / 64,
                        "selection_amortized_seconds": selection_seconds / 64,
                        "total_amortized_seconds": (
                            proposal_seconds + selection_seconds
                        )
                        / 64,
                    }
                )
            synchronize()
            arm_total_seconds = time.perf_counter() - arm_started
            arm_timings.append(
                {
                    "condition": condition,
                    "cut_depth": cut,
                    "proposal_seconds": proposal_seconds,
                    "selection_seconds": selection_seconds,
                    "compute_seconds": proposal_seconds + selection_seconds,
                    "end_to_end_arm_seconds": arm_total_seconds,
                    "proposal_seconds_per_record": proposal_seconds / 64,
                    "selection_seconds_per_record": selection_seconds / 64,
                    "total_seconds_per_record": (
                        proposal_seconds + selection_seconds
                    )
                    / 64,
                    "proposal_seconds_per_scored_token": proposal_seconds / 2496,
                    "selection_seconds_per_scored_token": selection_seconds / 2496,
                    "total_seconds_per_scored_token": (
                        proposal_seconds + selection_seconds
                    )
                    / 2496,
                    "embedding_comparisons": 2496 * 128256,
                    "candidate_simulations": (
                        2496 * 16
                        if args.method == "causal_public_surrogate_search"
                        else 0
                    ),
                }
            )
            del candidates, proposal_scores, selection_scores, prediction, queries
            torch.cuda.empty_cache()
    synchronize()
    method_seconds = time.perf_counter() - method_started
    method_ended_utc = utc_now()
    method_peak = peak_memory()

    with timer.measure("write_create_only_method_outputs"):
        queries_path = output_root / "queries.safetensors"
        save_file(
            {key: value.contiguous() for key, value in frozen_queries.items()},
            queries_path,
            metadata={
                "schema": "token-reconstruction.trr0001-r1-frozen-queries.v1",
                "method": args.method,
                "truth_opened": "false",
            },
        )
        reconstructions_path = output_root / "reconstructions.jsonl"
        write_jsonl_exclusive(reconstructions_path, rows)
        route_path = output_root / "route.json"
        write_json_exclusive(
            route_path,
            {
                "schema": "token-reconstruction.trr0001-r1-route.v1",
                "task_id": "TRR-0001",
                "revision_id": "TRR-0001-R1",
                "method": args.method,
                "config": file_record(args.config),
                "access_manifest": file_record(args.access_manifest),
                "observation_index": file_record(observation_index_path),
                "observations": observation_artifacts,
                "inverse_states": inverse_artifacts,
                "model": config["model"],
                "condition_order": config["condition_order"],
                "cut_order": config["cut_order"],
                "record_order": config["record_order"],
                "candidate_budget": 16,
                "stopping": "all 39 scored positions",
                "abstention": "none",
                "target_prefix_calls": 0,
                "truth_or_source_inputs": 0,
            },
        )

    evidence_path = output_root / "reconstructor_evidence.json"
    total_embedding_comparisons = 2496 * 128256 * 6
    total_candidate_simulations = (
        2496 * 16 * 6 if args.method == "causal_public_surrogate_search" else 0
    )
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr0001-r1-reconstructor-evidence.v1",
            "task_id": "TRR-0001",
            "revision_id": "TRR-0001-R1",
            "method": args.method,
            "command": command_record(),
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "method_started_utc": method_started_utc,
            "method_ended_utc": method_ended_utc,
            "exit_status": 0,
            "access_manifest_verified_before_inputs": True,
            "phases": timer.records,
            "arm_timings": arm_timings,
            "method_compute_seconds": method_seconds,
            "method_compute_seconds_per_record_arm": method_seconds / (64 * 6),
            "method_compute_seconds_per_scored_token": method_seconds / (2496 * 6),
            "records": 64,
            "arms": 6,
            "scored_tokens_per_arm": 2496,
            "candidate_budget": 16,
            "embedding_comparisons": total_embedding_comparisons,
            "candidate_simulations": total_candidate_simulations,
            "public_surrogate_model_loads": 1,
            "target_prefix_calls": 0,
            "fresh_training_steps": 0,
            "fresh_adaptation_steps": 0,
            "persisted_method_state": inverse_artifacts,
            "memory": {
                "preparation_peak": preparation_peak,
                "method_peak_after_cuda_reset": method_peak,
                "process_max_rss_kib_is_process_specific": True,
            },
            "implementation_complexity": implementation_complexity(args.method),
            "outputs": [
                file_record(path)
                for path in (queries_path, reconstructions_path, route_path)
            ],
        },
    )
    print(
        {
            "status": "isolated_reconstruction_complete",
            "method": args.method,
            "rows": len(rows),
            "target_prefix_calls": 0,
            "truth_or_source_inputs": 0,
            "method_compute_seconds": method_seconds,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
