"""Geometry-independent primitives for the dual-benchmark comparison matrix."""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .inverse import topk_candidates


BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
METHOD_IDS = (
    "direct_inverse_k16",
    "causal_public_surrogate_k16",
    "strict_bos_adaptive_a1_a2",
)
SETUP_IDS = (
    "clean-pile-lora-64x40",
    "historical-finance-strict-bos-128x128",
)


class DualBenchmarkError(RuntimeError):
    """Raised when a method port or benchmark cell violates the frozen contract."""


def validate_observations(
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> None:
    if observations.ndim != 3 or observations.shape[-1] != 2048:
        raise DualBenchmarkError("observations must be [records, positions, 2048]")
    if attention_mask.shape != observations.shape[:2]:
        raise DualBenchmarkError("attention mask geometry changed")
    if position_ids.shape != attention_mask.shape:
        raise DualBenchmarkError("position geometry changed")
    if observations.shape[0] == 0 or observations.shape[1] < 2:
        raise DualBenchmarkError("benchmark geometry is empty")
    mask = attention_mask.to(torch.bool)
    if not mask[:, 0].all().item():
        raise DualBenchmarkError("every benchmark row must begin with BOS")
    if (attention_mask[:, 1:] > attention_mask[:, :-1]).any().item():
        raise DualBenchmarkError("benchmark rows must be right padded")
    expected_positions = attention_mask.to(torch.long).cumsum(1).sub(1).clamp_min(0)
    if not torch.equal(position_ids.to(torch.long), expected_positions):
        raise DualBenchmarkError("position ids disagree with the attention mask")
    if not torch.isfinite(observations).all().item():
        raise DualBenchmarkError("observations contain non-finite values")


def scored_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(torch.bool).clone()
    mask[:, 0] = False
    return mask


def stable_candidate_order(
    candidate_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_ids.shape != candidate_scores.shape or candidate_ids.ndim != 2:
        raise DualBenchmarkError("candidate IDs and scores must be matching matrices")
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
def propose_k16(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    inverse: torch.nn.Module,
    embedding_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Run the frozen direct-inverse K16 proposal rule on variable geometry."""

    if observations.shape[:2] != attention_mask.shape:
        raise DualBenchmarkError("proposal geometry changed")
    mask = scored_mask(attention_mask)
    flat_observations = observations[mask].to(embedding_table.device)
    started = time.perf_counter()
    queries = inverse(flat_observations)
    candidate_ids, candidate_scores = topk_candidates(
        queries,
        embedding_table,
        k=16,
        score_batch_size=64,
    )
    candidate_ids, candidate_scores = stable_candidate_order(
        candidate_ids,
        candidate_scores,
    )
    if embedding_table.is_cuda:
        torch.cuda.synchronize(embedding_table.device)
    elapsed = time.perf_counter() - started

    shape = (*attention_mask.shape, 16)
    ids = torch.full(shape, INVALID_TOKEN_ID, dtype=torch.long)
    scores = torch.full(shape, float("-inf"), dtype=torch.float32)
    ids[mask] = candidate_ids.to(torch.long)
    scores[mask] = candidate_scores.to(torch.float32)
    predictions = torch.full(attention_mask.shape, INVALID_TOKEN_ID, dtype=torch.long)
    predictions[:, 0] = BOS_TOKEN_ID
    predictions[mask] = candidate_ids[:, 0].to(torch.long)
    return predictions, ids, scores, elapsed


def _repeat_cache(cache: Any, repeats: int) -> Any:
    candidate_cache = copy.deepcopy(cache)
    repeat = getattr(candidate_cache, "batch_repeat_interleave", None)
    if not callable(repeat):
        raise DualBenchmarkError("public-prefix cache cannot repeat candidates")
    repeat(repeats)
    return candidate_cache


@torch.inference_mode()
def causal_k16_record_serial(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    candidates: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Run the K16 causal public-surrogate rule record by record."""

    validate_observations(observations, attention_mask, position_ids)
    if candidates.shape != (*attention_mask.shape, 16):
        raise DualBenchmarkError("causal K16 candidate geometry changed")
    predictions = torch.full(attention_mask.shape, INVALID_TOKEN_ID, dtype=torch.long)
    predictions[:, 0] = BOS_TOKEN_ID
    selection_scores = torch.full(
        candidates.shape,
        float("-inf"),
        dtype=torch.float32,
    )
    simulations = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for row_index in range(observations.shape[0]):
        valid = torch.nonzero(
            attention_mask[row_index].to(torch.bool),
            as_tuple=False,
        ).flatten().tolist()
        cache = precut.new_cache()
        precut.run_cached(
            torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long, device=device),
            cache,
            0,
        )
        for logical_position, physical_position in enumerate(valid[1:], start=1):
            ids = candidates[row_index, physical_position].to(
                device=device,
                dtype=torch.long,
            )
            if ids.shape != (16,) or ids.lt(0).any().item():
                raise DualBenchmarkError("causal K16 received an invalid proposal row")
            candidate_cache = _repeat_cache(cache, 16)
            simulated = precut.run_cached(
                ids.view(-1, 1),
                candidate_cache,
                logical_position,
            )[:, -1].float()
            target = observations[row_index, physical_position].to(device).float()
            scores = F.cosine_similarity(
                simulated,
                target.view(1, -1),
                dim=-1,
            )
            if not torch.isfinite(scores).all().item():
                raise DualBenchmarkError("causal K16 scores are non-finite")
            winner = int(scores.argmax().item())
            chosen = int(ids[winner].item())
            predictions[row_index, physical_position] = chosen
            selection_scores[row_index, physical_position] = scores.cpu()
            precut.run_cached(
                torch.tensor([[chosen]], dtype=torch.long, device=device),
                cache,
                logical_position,
            )
            simulations += 16
            del candidate_cache, simulated, scores
        del cache
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return predictions, selection_scores, elapsed, simulations


@torch.inference_mode()
def causal_k16(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    candidates: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
    record_batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Run K16 causally while preserving the native 16-record batch geometry."""

    validate_observations(observations, attention_mask, position_ids)
    if candidates.shape != (*attention_mask.shape, 16):
        raise DualBenchmarkError("causal K16 candidate geometry changed")
    if record_batch_size != 16:
        raise DualBenchmarkError("causal K16 record batch size must remain 16")
    predictions = torch.full(attention_mask.shape, INVALID_TOKEN_ID, dtype=torch.long)
    predictions[:, 0] = BOS_TOKEN_ID
    selection_scores = torch.full(
        candidates.shape,
        float("-inf"),
        dtype=torch.float32,
    )
    simulations = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for record_start in range(0, observations.shape[0], record_batch_size):
        record_end = min(record_start + record_batch_size, observations.shape[0])
        record_count = record_end - record_start
        batch_mask_cpu = attention_mask[record_start:record_end].to(torch.bool)
        maximum_length = int(batch_mask_cpu.sum(dim=1).max().item())
        cache = precut.new_cache()
        precut.run_cached(
            torch.full(
                (record_count, 1),
                BOS_TOKEN_ID,
                dtype=torch.long,
                device=device,
            ),
            cache,
            0,
        )
        for physical_position in range(1, maximum_length):
            active_cpu = batch_mask_cpu[:, physical_position]
            if not active_cpu.any().item():
                raise DualBenchmarkError("right-padded batch has an interior gap")
            active = active_cpu.to(device)
            ids = candidates[
                record_start:record_end,
                physical_position,
            ].to(device=device, dtype=torch.long)
            if ids[active].shape[1:] != (16,) or ids[active].lt(0).any().item():
                raise DualBenchmarkError("causal K16 received an invalid proposal row")
            ids = ids.clone()
            ids[~active] = BOS_TOKEN_ID
            candidate_cache = _repeat_cache(cache, 16)
            simulated = precut.run_cached(
                ids.reshape(-1, 1),
                candidate_cache,
                physical_position,
            )[:, -1].reshape(record_count, 16, -1).float()
            target = observations[
                record_start:record_end,
                physical_position,
            ].to(device).float()
            scores = F.cosine_similarity(
                simulated,
                target[:, None, :],
                dim=-1,
            )
            if not torch.isfinite(scores).all().item():
                raise DualBenchmarkError("causal K16 scores are non-finite")
            choice = scores.argmax(dim=-1)
            chosen = ids.gather(1, choice[:, None]).squeeze(1)
            chosen[~active] = BOS_TOKEN_ID

            prediction_view = predictions[
                record_start:record_end,
                physical_position,
            ]
            prediction_view[active_cpu] = chosen[active].cpu()
            score_view = selection_scores[
                record_start:record_end,
                physical_position,
            ]
            score_view[active_cpu] = scores[active].cpu()
            precut.run_cached(
                chosen.view(-1, 1),
                cache,
                physical_position,
            )
            simulations += int(active_cpu.sum().item()) * 16
            del candidate_cache, simulated, target, scores
        del cache
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return predictions, selection_scores, elapsed, simulations


def score_predictions(
    *,
    predictions: torch.Tensor,
    truth: torch.Tensor,
    attention_mask: torch.Tensor,
    candidates: torch.Tensor | None = None,
    record_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if predictions.shape != truth.shape or predictions.shape != attention_mask.shape:
        raise DualBenchmarkError("prediction, truth, and mask geometry must match")
    if candidates is not None and candidates.shape[:2] != predictions.shape:
        raise DualBenchmarkError("candidate geometry differs from predictions")
    if record_ids is None:
        record_ids = [f"row-{index:06d}" for index in range(predictions.shape[0])]
    if len(record_ids) != predictions.shape[0]:
        raise DualBenchmarkError("record ID count changed")

    per_record: list[dict[str, Any]] = []
    total_scored = 0
    total_covered = 0
    total_correct = 0
    total_candidate_hits = 0
    exact_records = 0
    for row_index, record_id in enumerate(record_ids):
        mask = attention_mask[row_index].to(torch.bool).clone()
        mask[0] = False
        expected = truth[row_index][mask].to(torch.long)
        predicted = predictions[row_index][mask].to(torch.long)
        scored = int(expected.numel())
        covered_mask = predicted.ge(0)
        correct_mask = covered_mask & predicted.eq(expected)
        covered = int(covered_mask.sum().item())
        correct = int(correct_mask.sum().item())
        exact = bool(scored > 0 and correct == scored)
        candidate_hits = None
        if candidates is not None:
            proposed = candidates[row_index][mask].to(torch.long)
            candidate_hits = int(
                proposed.eq(expected.view(-1, 1)).any(dim=1).sum().item()
            )
            total_candidate_hits += candidate_hits
        total_scored += scored
        total_covered += covered
        total_correct += correct
        exact_records += int(exact)
        per_record.append(
            {
                "record_id": str(record_id),
                "scored_tokens": scored,
                "covered_tokens": covered,
                "correct_tokens": correct,
                "token_accuracy": correct / scored,
                "coverage": covered / scored,
                "selective_accuracy": correct / covered if covered else None,
                "exact_record": exact,
                "candidate_hits": candidate_hits,
                "candidate_recall": (
                    candidate_hits / scored if candidate_hits is not None else None
                ),
            }
        )

    if total_scored <= 0:
        raise DualBenchmarkError("no post-BOS tokens were scored")
    return (
        {
            "records": predictions.shape[0],
            "scored_tokens": total_scored,
            "covered_tokens": total_covered,
            "correct_tokens": total_correct,
            "token_accuracy": total_correct / total_scored,
            "coverage": total_covered / total_scored,
            "selective_accuracy": (
                total_correct / total_covered if total_covered else None
            ),
            "exact_records": exact_records,
            "exact_record_rate": exact_records / predictions.shape[0],
            "candidate_hits": total_candidate_hits if candidates is not None else None,
            "candidate_recall": (
                total_candidate_hits / total_scored if candidates is not None else None
            ),
        },
        per_record,
    )


def paired_record_differences(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
) -> list[float]:
    if len(left) != len(right) or not left:
        raise DualBenchmarkError("paired record metrics are empty or unmatched")
    differences: list[float] = []
    for left_row, right_row in zip(left, right):
        if left_row["record_id"] != right_row["record_id"]:
            raise DualBenchmarkError("paired record ordering changed")
        differences.append(
            float(left_row["token_accuracy"]) - float(right_row["token_accuracy"])
        )
    if not all(math.isfinite(value) for value in differences):
        raise DualBenchmarkError("paired record differences are non-finite")
    return differences

