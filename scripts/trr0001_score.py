#!/usr/bin/env python3
"""Open truth through the freeze gate and score frozen TRR-0001 outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import html
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from token_reconstruction.freeze import require_truth_open_allowed
from token_reconstruction.inverse import normalized_embeddings
from token_reconstruction.metrics import (
    bootstrap_mean,
    frequency_bin,
    percentile,
    record_metrics,
    summarize_numeric,
    token_group,
)
from token_reconstruction.public_prefix import (
    ContiguousPublicPrefix,
    PublicPrefixCache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--per-record", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--plot-svg", type=Path, required=True)
    parser.add_argument("--score-evidence", type=Path, required=True)
    return parser.parse_args()


def load_public_model_and_tokenizer() -> tuple[Any, torch.nn.Module]:
    if not torch.cuda.is_available():
        raise RuntimeError("TRR-0001 scoring requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(torch.device("cuda"))
        .eval()
    )
    return tokenizer, model


@torch.inference_mode()
def exact_true_ranks(
    queries: torch.Tensor,
    true_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute 1-based full-vocabulary ranks with token-ID ascending tie-break."""

    device = embedding_table.device
    vocabulary_ids = torch.arange(
        embedding_table.shape[0], device=device, dtype=torch.long
    )
    result: list[torch.Tensor] = []
    for start in range(0, queries.shape[0], batch_size):
        query = F.normalize(
            queries[start : start + batch_size].to(device).float(), dim=-1
        )
        truth = true_ids[start : start + batch_size].to(device)
        scores = query @ embedding_table.transpose(0, 1)
        true_score = scores.gather(1, truth[:, None])
        greater = (scores > true_score).sum(dim=1)
        tied_lower = (
            (scores == true_score) & (vocabulary_ids[None, :] < truth[:, None])
        ).sum(dim=1)
        result.append((greater + tied_lower + 1).cpu())
        del scores, true_score, greater, tied_lower
    return torch.cat(result)


@torch.inference_mode()
def teacher_prefix_counterfactual(
    *,
    model: torch.nn.Module,
    cut_depth: int,
    candidates: torch.Tensor,
    observations: torch.Tensor,
    truth_with_bos: torch.Tensor,
) -> torch.Tensor:
    """Choose frozen candidates while committing the true prefix post-freeze."""

    device = next(model.parameters()).device
    prefix = ContiguousPublicPrefix(model, cut_depth)
    selected = torch.empty((64, 39), dtype=torch.long)
    if cut_depth == 0:
        for position in range(39):
            ids = candidates[:, position, :].to(device)
            simulated = prefix.embed_tokens(ids).float()
            target = observations[:, position + 1, :].to(device).float()
            score = F.cosine_similarity(simulated, target[:, None, :], dim=-1)
            choice = score.argmax(dim=-1)
            selected[:, position] = ids.gather(1, choice[:, None]).squeeze(1).cpu()
        return selected

    for record_start in range(0, 64, 16):
        record_end = min(record_start + 16, 64)
        records = record_end - record_start
        cache = prefix.new_cache()
        prefix.run_cached(
            torch.full(
                (records, 1), BOS_TOKEN_ID, device=device, dtype=torch.long
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
            simulated = simulated[:, -1, :].reshape(records, 16, -1).float()
            target = observations[
                record_start:record_end, position + 1, :
            ].to(device).float()
            score = F.cosine_similarity(simulated, target[:, None, :], dim=-1)
            choice = score.argmax(dim=-1)
            selected[record_start:record_end, position] = (
                ids.gather(1, choice[:, None]).squeeze(1).cpu()
            )
            true_next = truth_with_bos[
                record_start:record_end, position + 1
            ].to(device).view(records, 1)
            prefix.run_cached(true_next, cache, position + 1)
            del fork, fork_backend, simulated, target, score
        del cache
        torch.cuda.empty_cache()
    return selected


def aggregate_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_total = sum(row["correct_tokens"] for row in rows)
    candidate_total = sum(row["top16_true_token_count"] for row in rows)
    conditional_correct = sum(row["conditional_correct_count"] for row in rows)
    record_accuracies = [row["token_accuracy"] for row in rows]
    exact_values = [float(row["exact_sequence_match"]) for row in rows]
    ranks = [rank for row in rows for rank in row["true_token_ranks"]]
    return {
        "records": len(rows),
        "scored_tokens": len(rows) * 39,
        "token_accuracy": token_total / (len(rows) * 39),
        "token_accuracy_bootstrap": bootstrap_mean(
            record_accuracies, draws=10000, seed=1732
        ),
        "exact_sequence_match_rate": sum(exact_values) / len(rows),
        "exact_sequence_match_bootstrap": bootstrap_mean(
            exact_values, draws=10000, seed=1732
        ),
        "correct_prefix_length": summarize_numeric(
            [row["correct_prefix_length"] for row in rows]
        ),
        "first_error_position": summarize_numeric(
            [
                row["first_error_position"]
                for row in rows
                if row["first_error_position"] is not None
            ]
        ),
        "coverage": 1.0,
        "selective_accuracy": token_total / (len(rows) * 39),
        "top16_true_token_recall": candidate_total / (len(rows) * 39),
        "conditional_selection_accuracy": (
            conditional_correct / candidate_total if candidate_total else None
        ),
        "exact_true_token_rank": {
            "count": len(ranks),
            "mean": statistics.fmean(ranks),
            "median": statistics.median(ranks),
            "p95": percentile(ranks, 0.95),
            "maximum": max(ranks),
        },
        "runtime": {
            "total_seconds": sum(row["amortized_seconds"] for row in rows),
            "mean_seconds_per_record": statistics.fmean(
                row["amortized_seconds"] for row in rows
            ),
            "mean_seconds_per_scored_token": sum(
                row["amortized_seconds"] for row in rows
            )
            / (len(rows) * 39),
        },
    }


def breakdown(
    token_rows: Iterable[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        groups[str(row[key])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for label, rows in sorted(groups.items()):
        result[label] = {
            "tokens": len(rows),
            "direct_accuracy": sum(row["direct_correct"] for row in rows) / len(rows),
            "causal_accuracy": sum(row["causal_correct"] for row in rows) / len(rows),
            "top16_recall": sum(row["true_in_top16"] for row in rows) / len(rows),
        }
    return result


def write_csv_exclusive(path: Path, aggregates: dict[tuple[str, int, str], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "condition",
                "cut_depth",
                "method",
                "records",
                "scored_tokens",
                "token_accuracy",
                "token_accuracy_ci_low",
                "token_accuracy_ci_high",
                "exact_sequence_match_rate",
                "top16_recall",
                "conditional_selection_accuracy",
                "mean_seconds_per_record",
            ]
        )
        for condition in CONDITIONS:
            for cut in CUT_DEPTHS:
                for method in ("direct_inverse", "causal_public_surrogate_search"):
                    value = aggregates[(condition, cut, method)]
                    writer.writerow(
                        [
                            condition,
                            cut,
                            method,
                            value["records"],
                            value["scored_tokens"],
                            value["token_accuracy"],
                            *value["token_accuracy_bootstrap"]["ci95_percentile"],
                            value["exact_sequence_match_rate"],
                            value["top16_true_token_recall"],
                            value["conditional_selection_accuracy"],
                            value["runtime"]["mean_seconds_per_record"],
                        ]
                    )


def write_svg_exclusive(
    path: Path, aggregates: dict[tuple[str, int, str], dict[str, Any]]
) -> None:
    width, height = 1100, 520
    margin_left, margin_bottom, margin_top = 70, 125, 45
    plot_height = height - margin_bottom - margin_top
    values: list[tuple[str, float, str]] = []
    colors = {
        "direct_inverse": "#3b82f6",
        "causal_public_surrogate_search": "#f97316",
    }
    for condition in CONDITIONS:
        for cut in CUT_DEPTHS:
            for method in ("direct_inverse", "causal_public_surrogate_search"):
                label = f"{condition.replace('_', ' ')} / cut {cut} / {method.split('_')[0]}"
                values.append((label, aggregates[(condition, cut, method)]["token_accuracy"], colors[method]))
    usable = width - margin_left - 25
    step = usable / len(values)
    bar_width = step * 0.72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">TRR-0001 blind token accuracy (2,496 scored tokens per arm)</text>',
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        y = margin_top + (1 - value) * plot_height
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-25}" y2="{y:.1f}" stroke="#d1d5db"/>')
        parts.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    for index, (label, value, color) in enumerate(values):
        x = margin_left + index * step + (step - bar_width) / 2
        y = margin_top + (1 - value) * plot_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{value*plot_height:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{y-4:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.3f}</text>')
        parts.append(f'<text transform="translate({x+bar_width/2:.1f},{height-margin_bottom+8}) rotate(55)" text-anchor="start" font-family="sans-serif" font-size="10">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(parts) + "\n")


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    started_utc = utc_now()
    timer = PhaseTimer()
    seed_everything(1729)
    torch.cuda.reset_peak_memory_stats()

    with timer.measure("verify_freeze_and_open_truth_gate"):
        receipt = require_truth_open_allowed(
            receipt_path=args.receipt,
            repository_root=root,
            truth_path=args.truth,
        )
    with timer.measure("read_truth_after_successful_gate"):
        truth_rows = read_jsonl(args.truth)
    if len(truth_rows) != 64:
        raise RuntimeError("truth record count changed")
    truth_by_id = {row["record_id"]: row for row in truth_rows}
    if len(truth_by_id) != 64:
        raise RuntimeError("truth record IDs are duplicated")
    truth_with_bos = torch.tensor(
        [row["token_ids"] for row in truth_rows], dtype=torch.long
    )
    if tuple(truth_with_bos.shape) != (64, 40):
        raise RuntimeError("truth token geometry changed")

    frozen_rows = read_jsonl(args.frozen_root / "reconstructions.jsonl")
    route = load_json(args.frozen_root / "route.json")
    if len(frozen_rows) != 384 or route.get("truth_or_correctness_inputs") != 0:
        raise RuntimeError("frozen reconstruction coverage or separation changed")
    if [row["record_id"] for row in truth_rows] != route["record_order"]:
        raise RuntimeError("truth and frozen record orders differ")
    row_by_arm_record = {
        (row["condition"], int(row["cut_depth"]), int(row["record_index"])): row
        for row in frozen_rows
    }
    if len(row_by_arm_record) != 384:
        raise RuntimeError("frozen arm records are absent or duplicated")

    with timer.measure("load_public_model_for_post_truth_scoring"):
        tokenizer, model = load_public_model_and_tokenizer()
    device = next(model.parameters()).device
    embedding_table = normalized_embeddings(
        model.get_input_embeddings().weight
    ).to(device)
    query_state = load_file(args.frozen_root / "queries.safetensors", device="cpu")
    expected_query_keys = {
        f"{condition}.cut{cut}" for condition in CONDITIONS for cut in CUT_DEPTHS
    }
    if set(query_state) != expected_query_keys:
        raise RuntimeError("frozen query keys changed")

    true_scored = truth_with_bos[:, 1:].reshape(-1)
    exact_rank_by_arm: dict[tuple[str, int], torch.Tensor] = {}
    with timer.measure("compute_exact_full_vocabulary_true_token_ranks"):
        for condition in CONDITIONS:
            for cut in CUT_DEPTHS:
                key = f"{condition}.cut{cut}"
                queries = query_state[key].reshape(-1, 2048)
                exact_rank_by_arm[(condition, cut)] = exact_true_ranks(
                    queries, true_scored, embedding_table
                ).reshape(64, 39)

    observation_index = load_json(args.observation_index)
    entries = {
        (entry["condition"], int(entry["cut_depth"])): entry
        for entry in observation_index["entries"]
    }
    teacher_predictions: dict[tuple[str, int], torch.Tensor] = {}
    teacher_timings: list[dict[str, Any]] = []
    with timer.measure("teacher_prefix_public_surrogate_counterfactual"):
        for condition in CONDITIONS:
            for cut in CUT_DEPTHS:
                entry = entries[(condition, cut)]
                observation_path = args.observation_index.parent / entry["path"]
                if sha256_file(observation_path) != entry["artifact"]["sha256"]:
                    raise RuntimeError("post-truth observation hash changed")
                observations = load_file(
                    observation_path, device="cpu"
                )["activations"]
                candidates = torch.tensor(
                    [
                        row_by_arm_record[(condition, cut, index)]["candidate_ids"]
                        for index in range(64)
                    ],
                    dtype=torch.long,
                )
                synchronize()
                start = time.perf_counter()
                teacher_predictions[(condition, cut)] = teacher_prefix_counterfactual(
                    model=model,
                    cut_depth=cut,
                    candidates=candidates,
                    observations=observations,
                    truth_with_bos=truth_with_bos,
                )
                synchronize()
                teacher_timings.append(
                    {
                        "condition": condition,
                        "cut_depth": cut,
                        "elapsed_seconds": time.perf_counter() - start,
                        "candidate_simulations": 64 * 39 * 16,
                    }
                )

    frequency_payload = load_json(
        args.frozen_root / "method_state" / "auxiliary_frequency_counts.json"
    )
    frequency_counts = {
        int(key): int(value) for key, value in frequency_payload["counts"].items()
    }
    per_record: list[dict[str, Any]] = []
    token_diagnostics: list[dict[str, Any]] = []
    records_by_arm_method: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for condition in CONDITIONS:
        for cut in CUT_DEPTHS:
            ranks = exact_rank_by_arm[(condition, cut)]
            teacher = teacher_predictions[(condition, cut)]
            for record_index in range(64):
                frozen = row_by_arm_record[(condition, cut, record_index)]
                truth = truth_by_id[frozen["record_id"]]["token_ids"][1:]
                candidates = frozen["candidate_ids"]
                method_predictions = {
                    "direct_inverse": frozen["direct_tokens"],
                    "causal_public_surrogate_search": frozen["causal_tokens"],
                }
                for method, prediction in method_predictions.items():
                    measured = record_metrics(prediction, truth, candidates)
                    row = {
                        "condition": condition,
                        "cut_depth": cut,
                        "method": method,
                        "record_index": record_index,
                        "record_id": frozen["record_id"],
                        **{key: value for key, value in measured.items() if key not in ("correctness", "true_in_top16")},
                        "correctness": measured["correctness"],
                        "true_in_top16": measured["true_in_top16"],
                        "true_token_ranks": ranks[record_index].tolist(),
                        "proposal_amortized_seconds": frozen["direct_amortized_seconds"],
                        "selection_amortized_seconds": (
                            0.0
                            if method == "direct_inverse"
                            else frozen["causal_amortized_seconds"]
                        ),
                        "amortized_seconds": (
                            frozen["direct_amortized_seconds"]
                            + (
                                0.0
                                if method == "direct_inverse"
                                else frozen["causal_amortized_seconds"]
                            )
                        ),
                    }
                    if method == "causal_public_surrogate_search":
                        teacher_metric = record_metrics(
                            teacher[record_index].tolist(), truth, candidates
                        )
                        row["teacher_prefix_counterfactual_accuracy"] = teacher_metric[
                            "token_accuracy"
                        ]
                    per_record.append(row)
                    records_by_arm_method[(condition, cut, method)].append(row)

                direct_prediction = frozen["direct_tokens"]
                causal_prediction = frozen["causal_tokens"]
                first_causal_error = next(
                    (
                        position
                        for position, (predicted, expected) in enumerate(
                            zip(causal_prediction, truth), 1
                        )
                        if predicted != expected
                    ),
                    None,
                )
                for position, expected in enumerate(truth, 1):
                    raw_token = tokenizer.convert_ids_to_tokens(int(expected))
                    decoded = tokenizer.decode(
                        [int(expected)], clean_up_tokenization_spaces=False
                    )
                    direct_score = float(
                        frozen["direct_candidate_scores"][position - 1][0]
                    )
                    causal_id = causal_prediction[position - 1]
                    chosen_index = candidates[position - 1].index(causal_id)
                    causal_score = float(
                        frozen["causal_candidate_scores"][position - 1][chosen_index]
                    )
                    token_diagnostics.append(
                        {
                            "condition": condition,
                            "cut_depth": cut,
                            "record_index": record_index,
                            "position": position,
                            "frequency_bin": frequency_bin(
                                frequency_counts.get(int(expected), 0)
                            ),
                            "token_group": token_group(raw_token, decoded),
                            "direct_correct": direct_prediction[position - 1] == expected,
                            "causal_correct": causal_prediction[position - 1] == expected,
                            "true_in_top16": expected in candidates[position - 1],
                            "direct_chosen_match_score": direct_score,
                            "causal_chosen_match_score": causal_score,
                            "causal_after_first_error": (
                                first_causal_error is not None
                                and position > first_causal_error
                            ),
                        }
                    )

    aggregates = {
        key: aggregate_records(rows) for key, rows in records_by_arm_method.items()
    }
    comparisons: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for cut in CUT_DEPTHS:
            direct = records_by_arm_method[(condition, cut, "direct_inverse")]
            causal = records_by_arm_method[
                (condition, cut, "causal_public_surrogate_search")
            ]
            differences = [
                right["token_accuracy"] - left["token_accuracy"]
                for left, right in zip(direct, causal)
            ]
            comparison = bootstrap_mean(differences, draws=10000, seed=1732)
            low, high = comparison["ci95_percentile"]
            comparison["disposition"] = (
                "causal_supported_improvement"
                if low > 0
                else "causal_supported_degradation"
                if high < 0
                else "inconclusive"
            )
            comparisons.append(
                {
                    "type": "causal_minus_direct",
                    "condition": condition,
                    "cut_depth": cut,
                    **comparison,
                }
            )
    mismatch: list[dict[str, Any]] = []
    for cut in CUT_DEPTHS:
        for method in ("direct_inverse", "causal_public_surrogate_search"):
            matched = records_by_arm_method[("matched_public", cut, method)]
            target = records_by_arm_method[("unavailable_target_lora", cut, method)]
            differences = [
                right["token_accuracy"] - left["token_accuracy"]
                for left, right in zip(matched, target)
            ]
            mismatch.append(
                {
                    "type": "unavailable_target_minus_matched_public",
                    "cut_depth": cut,
                    "method": method,
                    **bootstrap_mean(differences, draws=10000, seed=1732),
                }
            )

    aggregate_json = [
        {
            "condition": condition,
            "cut_depth": cut,
            "method": method,
            **aggregates[(condition, cut, method)],
        }
        for condition in CONDITIONS
        for cut in CUT_DEPTHS
        for method in ("direct_inverse", "causal_public_surrogate_search")
    ]
    primary_direct = aggregates[("unavailable_target_lora", 4, "direct_inverse")]
    primary_causal = aggregates[
        ("unavailable_target_lora", 4, "causal_public_surrogate_search")
    ]
    primary_comparison = next(
        item
        for item in comparisons
        if item["condition"] == "unavailable_target_lora"
        and item["cut_depth"] == 4
    )
    frozen_evidence = load_json(
        args.frozen_root / "reconstructor_evidence.json"
    )
    metrics = {
        "schema": "token-reconstruction.trr0001-metrics.v1",
        "task_id": "TRR-0001",
        "truth_opened_after_verified_freeze": True,
        "freeze_receipt_sha256": sha256_file(args.receipt),
        "records": 64,
        "scored_tokens_per_arm": 2496,
        "aggregates": aggregate_json,
        "paired_method_comparisons": comparisons,
        "paired_target_surrogate_mismatch": mismatch,
        "primary": {
            "condition": "unavailable_target_lora",
            "cut_depth": 4,
            "metric": "token_accuracy_excluding_bos",
            "direct_inverse": primary_direct["token_accuracy"],
            "causal_public_surrogate_search": primary_causal["token_accuracy"],
            "causal_minus_direct": primary_comparison,
        },
        "statistics": {
            "bootstrap_draws": 10000,
            "seed": 1732,
            "unit": "record",
            "confidence_interval": "95% percentile bootstrap",
        },
        "cost": {
            "direct_embedding_comparisons": frozen_evidence[
                "direct_embedding_comparisons"
            ],
            "causal_candidate_simulations": frozen_evidence[
                "causal_candidate_simulations"
            ],
            "target_prefix_calls": 0,
            "candidate_budget": 16,
            "persisted_method_state_bytes": sum(
                entry["bytes"] for entry in route["method_state"]
            ),
            "frozen_bundle_bytes": sum(
                entry["bytes"] for entry in receipt["entries"]
            ),
            "peak_memory": {
                "reconstruction": frozen_evidence["peak_memory"],
                "scoring": peak_memory(),
            },
        },
        "claim_scope": "Pinned Llama-3.2-1B-Instruct, Pile-10k, one rank-4 target LoRA, cuts 0/4/8, candidate budget 16; no broader generalization.",
    }

    arm_token_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in token_diagnostics:
        arm_token_rows[(row["condition"], row["cut_depth"])].append(row)
    diagnostics_arms: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for cut in CUT_DEPTHS:
            rows = arm_token_rows[(condition, cut)]
            correct_direct_scores = [
                row["direct_chosen_match_score"] for row in rows if row["direct_correct"]
            ]
            incorrect_direct_scores = [
                row["direct_chosen_match_score"] for row in rows if not row["direct_correct"]
            ]
            correct_causal_scores = [
                row["causal_chosen_match_score"] for row in rows if row["causal_correct"]
            ]
            incorrect_causal_scores = [
                row["causal_chosen_match_score"] for row in rows if not row["causal_correct"]
            ]
            suffix = [row for row in rows if row["causal_after_first_error"]]
            teacher_accuracy = statistics.fmean(
                row["teacher_prefix_counterfactual_accuracy"]
                for row in records_by_arm_method[
                    (condition, cut, "causal_public_surrogate_search")
                ]
            )
            diagnostics_arms.append(
                {
                    "condition": condition,
                    "cut_depth": cut,
                    "by_position": breakdown(rows, "position"),
                    "by_frequency_bin": breakdown(rows, "frequency_bin"),
                    "by_token_group": breakdown(rows, "token_group"),
                    "activation_match_scores": {
                        "direct_correct": summarize_numeric(correct_direct_scores),
                        "direct_incorrect": summarize_numeric(incorrect_direct_scores),
                        "causal_correct": summarize_numeric(correct_causal_scores),
                        "causal_incorrect": summarize_numeric(incorrect_causal_scores),
                    },
                    "error_propagation": {
                        "tokens_strictly_after_first_causal_error": len(suffix),
                        "causal_accuracy_strictly_after_first_error": (
                            sum(row["causal_correct"] for row in suffix) / len(suffix)
                            if suffix
                            else None
                        ),
                        "teacher_prefix_counterfactual_accuracy": teacher_accuracy,
                        "frozen_reconstructed_prefix_accuracy": aggregates[
                            (
                                condition,
                                cut,
                                "causal_public_surrogate_search",
                            )
                        ]["token_accuracy"],
                    },
                }
            )
    diagnostics = {
        "schema": "token-reconstruction.trr0001-diagnostics.v1",
        "task_id": "TRR-0001",
        "post_truth_only": True,
        "frozen_outputs_revised": False,
        "token_group_precedence": [
            "numeric",
            "punctuation",
            "whitespace_prefixed",
            "other",
        ],
        "arms": diagnostics_arms,
        "teacher_prefix_timings": teacher_timings,
        "target_surrogate_mismatch": mismatch,
    }

    write_jsonl_exclusive(args.per_record, per_record)
    write_json_exclusive(args.metrics, metrics)
    write_json_exclusive(args.diagnostics, diagnostics)
    write_csv_exclusive(args.summary_csv, aggregates)
    write_svg_exclusive(args.plot_svg, aggregates)
    evidence = {
        "schema": "token-reconstruction.trr0001-score-evidence.v1",
        "task_id": "TRR-0001",
        "command": command_record(),
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "exit_status": 0,
        "truth_gate": {
            "passed_before_truth_read": True,
            "receipt": file_record(args.receipt, root=root),
            "preregistration_commit": receipt["preregistration_commit"],
            "entry_count": len(receipt["entries"]),
        },
        "truth": file_record(args.truth, root=root),
        "phases": timer.records,
        "exact_rank_queries": 64 * 39 * 6,
        "teacher_prefix_candidate_simulations": 64 * 39 * 16 * 6,
        "bootstrap_draws_per_estimate": 10000,
        "outputs": [
            file_record(path, root=root)
            for path in (
                args.metrics,
                args.per_record,
                args.diagnostics,
                args.summary_csv,
                args.plot_svg,
            )
        ],
        "peak_memory": peak_memory(),
    }
    write_json_exclusive(args.score_evidence, evidence)
    print(
        {
            "status": "scored_after_verified_freeze",
            "primary_direct": primary_direct["token_accuracy"],
            "primary_causal": primary_causal["token_accuracy"],
            "disposition": primary_comparison["disposition"],
            "metrics": str(args.metrics),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
