"""Public-development-calibrated adaptive causal selection for TRR-0002."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from .dual_benchmark import BOS_TOKEN_ID, INVALID_TOKEN_ID, validate_observations


BASE_BUDGET = 32
MAX_BUDGET = 64
ROUTE_BASE = 2
ROUTE_EXPANDED = 3
ROUTE_BOS = 1
ROUTE_PADDING = 0


class CalibratedSelectorError(RuntimeError):
    """Raised when the frozen calibrated-selector contract changes."""


@dataclass(frozen=True)
class CalibratedSelectorResult:
    predictions: torch.Tensor
    base_scores: torch.Tensor
    extra_scores: torch.Tensor
    normalized_gap: torch.Tensor
    routes: torch.Tensor
    elapsed_seconds: float
    base_candidate_simulations: int
    extra_candidate_simulations: int
    executed_candidate_simulations: int
    prefix_commit_tokens: int


def scale_normalized_gap(scores: torch.Tensor) -> torch.Tensor:
    """Return the top-two gap divided by the candidate-score RMS scale.

    The statistic is invariant to adding a constant or multiplying every
    candidate score by the same positive scalar. This avoids A2's historical
    raw-score threshold while retaining a simple, auditable confidence signal.
    """

    if scores.ndim != 2 or scores.shape[1] != BASE_BUDGET:
        raise CalibratedSelectorError("normalized-gap score geometry changed")
    if not torch.isfinite(scores).all().item():
        raise CalibratedSelectorError("normalized-gap scores are non-finite")
    top_two = scores.topk(k=2, dim=1).values
    centered = scores - scores.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1).sqrt().clamp_min(1e-8)
    result = (top_two[:, 0] - top_two[:, 1]) / scale
    if not torch.isfinite(result).all().item() or result.lt(0).any().item():
        raise CalibratedSelectorError("normalized gap is invalid")
    return result


def _repeat_cache(cache: Any, repeats: int) -> Any:
    candidate_cache = copy.deepcopy(cache)
    repeat = getattr(candidate_cache, "batch_repeat_interleave", None)
    if not callable(repeat):
        raise CalibratedSelectorError("public-prefix cache cannot repeat candidates")
    repeat(repeats)
    return candidate_cache


@torch.inference_mode()
def select_calibrated_adaptive(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    candidates: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
    threshold: float,
    record_batch_size: int = 8,
) -> CalibratedSelectorResult:
    """Select causally at K=32 and expand uncertain positions to K=64.

    Expansion is decided only from the public-surrogate K=32 score vector.
    The method never abstains and never accesses the unavailable target prefix.
    """

    validate_observations(observations, attention_mask, position_ids)
    if candidates.shape[:2] != attention_mask.shape or candidates.shape[2] != MAX_BUDGET:
        raise CalibratedSelectorError("calibrated candidate geometry changed")
    if candidates.dtype not in (torch.int32, torch.int64):
        raise CalibratedSelectorError("calibrated candidates must be integer IDs")
    if not 0 < record_batch_size <= 8:
        raise CalibratedSelectorError("calibrated selector batch changed")
    if math.isnan(threshold):
        raise CalibratedSelectorError("calibration threshold is NaN")

    predictions = torch.full(attention_mask.shape, INVALID_TOKEN_ID, dtype=torch.long)
    predictions[:, 0] = BOS_TOKEN_ID
    base_scores_out = torch.full(
        (*attention_mask.shape, BASE_BUDGET), float("nan"), dtype=torch.float32
    )
    extra_scores_out = torch.full(
        (*attention_mask.shape, MAX_BUDGET - BASE_BUDGET),
        float("nan"),
        dtype=torch.float32,
    )
    gaps = torch.full(attention_mask.shape, float("nan"), dtype=torch.float32)
    routes = torch.full(attention_mask.shape, ROUTE_PADDING, dtype=torch.int8)
    routes[:, 0] = ROUTE_BOS
    base_simulations = 0
    extra_simulations = 0
    executed_simulations = 0
    prefix_commit_tokens = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for record_start in range(0, observations.shape[0], record_batch_size):
        record_end = min(record_start + record_batch_size, observations.shape[0])
        record_count = record_end - record_start
        batch_mask = attention_mask[record_start:record_end].to(torch.bool)
        maximum_length = int(batch_mask.sum(dim=1).max().item())
        cache = precut.new_cache()
        precut.run_cached(
            torch.full((record_count, 1), BOS_TOKEN_ID, dtype=torch.long, device=device),
            cache,
            0,
        )
        prefix_commit_tokens += record_count

        for position in range(1, maximum_length):
            active_cpu = batch_mask[:, position]
            active = active_cpu.to(device)
            ids = candidates[record_start:record_end, position].to(device=device, dtype=torch.long)
            if ids[active].lt(0).any().item():
                raise CalibratedSelectorError("active candidate row contains invalid IDs")
            ids = ids.clone()
            ids[~active] = BOS_TOKEN_ID
            target = observations[record_start:record_end, position].to(device).float()

            base_cache = _repeat_cache(cache, BASE_BUDGET)
            simulated_base = precut.run_cached(
                ids[:, :BASE_BUDGET].reshape(-1, 1), base_cache, position
            )[:, -1].reshape(record_count, BASE_BUDGET, -1).float()
            base_score = F.cosine_similarity(
                simulated_base, target[:, None, :], dim=-1
            )
            if not torch.isfinite(base_score[active]).all().item():
                raise CalibratedSelectorError("base selection scores are non-finite")
            gap = scale_normalized_gap(base_score)
            expand_cpu = active_cpu & gap.cpu().le(threshold)
            expand = expand_cpu.to(device)

            combined_score = torch.full(
                (record_count, MAX_BUDGET),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )
            combined_score[:, :BASE_BUDGET] = base_score
            extra_score = torch.full(
                (record_count, MAX_BUDGET - BASE_BUDGET),
                float("nan"),
                dtype=torch.float32,
                device=device,
            )
            if expand.any().item():
                extra_ids = ids[:, BASE_BUDGET:].clone()
                extra_ids[~expand] = BOS_TOKEN_ID
                extra_cache = _repeat_cache(cache, MAX_BUDGET - BASE_BUDGET)
                simulated_extra = precut.run_cached(
                    extra_ids.reshape(-1, 1), extra_cache, position
                )[:, -1].reshape(
                    record_count, MAX_BUDGET - BASE_BUDGET, -1
                ).float()
                evaluated_extra = F.cosine_similarity(
                    simulated_extra, target[:, None, :], dim=-1
                )
                if not torch.isfinite(evaluated_extra[expand]).all().item():
                    raise CalibratedSelectorError("expanded selection scores are non-finite")
                extra_score[expand] = evaluated_extra[expand]
                combined_score[expand, BASE_BUDGET:] = evaluated_extra[expand]
                extra_simulations += int(expand.sum().item()) * (
                    MAX_BUDGET - BASE_BUDGET
                )
                executed_simulations += record_count * (MAX_BUDGET - BASE_BUDGET)
                del extra_cache, simulated_extra, evaluated_extra

            choice = combined_score.argmax(dim=1)
            chosen = ids.gather(1, choice[:, None]).squeeze(1)
            prediction_view = predictions[record_start:record_end, position]
            prediction_view[active_cpu] = chosen[active].cpu()
            base_scores_out[record_start:record_end, position][active_cpu] = (
                base_score[active].cpu()
            )
            extra_scores_out[record_start:record_end, position][expand_cpu] = (
                extra_score[expand].cpu()
            )
            gaps[record_start:record_end, position][active_cpu] = gap[active].cpu()
            route_view = routes[record_start:record_end, position]
            route_view[active_cpu] = ROUTE_BASE
            route_view[expand_cpu] = ROUTE_EXPANDED

            commit = torch.full(
                (record_count,), BOS_TOKEN_ID, dtype=torch.long, device=device
            )
            commit[active] = chosen[active]
            precut.run_cached(commit[:, None], cache, position)
            base_simulations += int(active.sum().item()) * BASE_BUDGET
            executed_simulations += record_count * BASE_BUDGET
            prefix_commit_tokens += record_count
            del base_cache, simulated_base, base_score, extra_score, combined_score, target
        del cache

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return CalibratedSelectorResult(
        predictions=predictions,
        base_scores=base_scores_out,
        extra_scores=extra_scores_out,
        normalized_gap=gaps,
        routes=routes,
        elapsed_seconds=elapsed,
        base_candidate_simulations=base_simulations,
        extra_candidate_simulations=extra_simulations,
        executed_candidate_simulations=executed_simulations,
        prefix_commit_tokens=prefix_commit_tokens,
    )
