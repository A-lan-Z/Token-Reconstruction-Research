"""Preregistered component-crossover primitives for TRR-0002."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .dual_benchmark import (
    BOS_TOKEN_ID,
    INVALID_TOKEN_ID,
    DualBenchmarkError,
    scored_mask,
    validate_observations,
)
from .inverse import topk_candidates


BUDGETS = (8, 16, 32, 64)
A2_NORMALIZED_WINNER_MIN = 2.0
ROUTE_PADDING = 0
ROUTE_BOS = 1
ROUTE_ACCEPT = 2
ROUTE_ABSTAIN = 3
ROUTE_ABSTAINED_SUFFIX = 4

BASE_METHOD_IDS = (
    "direct_inverse_k16",
    "causal_public_surrogate_k16",
    "strict_bos_adaptive_a1_a2",
)
FACTORIAL_METHOD_IDS = (
    "a1_a2_k8",
    "a1_a2_k16",
    "a1_a2_k32",
    "a1_a2_k64",
    "a1_causal_k8",
    "a1_causal_k16",
    "a1_causal_k32",
    "a1_causal_k64",
    "residual_affine_a2_k8",
    "residual_affine_a2_k16",
    "residual_affine_a2_k32",
    "residual_affine_a2_k64",
    "residual_affine_causal_k8",
    "causal_public_surrogate_k16",
    "residual_affine_causal_k32",
    "residual_affine_causal_k64",
    "a1_residual_union_causal_k8",
    "a1_residual_union_causal_k16",
    "a1_residual_union_causal_k32",
    "a1_residual_union_causal_k64",
)
METHOD_IDS = tuple(dict.fromkeys((*BASE_METHOD_IDS, *FACTORIAL_METHOD_IDS)))


class ComponentCrossoverError(DualBenchmarkError):
    """Raised when a frozen crossover rule or geometry changes."""


@dataclass(frozen=True)
class ProposalResult:
    candidates: torch.Tensor
    scores: torch.Tensor
    top1_confidence: torch.Tensor
    elapsed_seconds: float


@dataclass(frozen=True)
class SelectorResult:
    predictions: torch.Tensor
    scores: torch.Tensor
    winner_margin: torch.Tensor
    normalized_winner: torch.Tensor
    routes: torch.Tensor
    elapsed_seconds: float
    candidate_simulations: int
    executed_candidate_simulations: int
    prefix_commit_tokens: int


def method_spec(method_id: str) -> tuple[str, str, int] | None:
    if method_id in BASE_METHOD_IDS:
        return None
    for prefix, proposal, selector in (
        ("a1_a2_k", "a1", "a2_fixed_budget"),
        ("a1_causal_k", "a1", "causal"),
        ("residual_affine_a2_k", "residual_affine", "a2_fixed_budget"),
        ("residual_affine_causal_k", "residual_affine", "causal"),
        ("a1_residual_union_causal_k", "a1_residual_union", "causal"),
    ):
        if method_id.startswith(prefix):
            return proposal, selector, int(method_id.removeprefix(prefix))
    raise ComponentCrossoverError(f"unregistered crossover method: {method_id}")


def _stable_score_order(
    candidate_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort score descending and token ID ascending for exact score ties."""

    if candidate_ids.shape != candidate_scores.shape or candidate_ids.ndim != 2:
        raise ComponentCrossoverError("candidate IDs and scores must be matrices")
    by_id = torch.argsort(candidate_ids, dim=1, stable=True)
    ids = candidate_ids.gather(1, by_id)
    scores = candidate_scores.gather(1, by_id)
    by_score = torch.argsort(scores, dim=1, descending=True, stable=True)
    return ids.gather(1, by_score), scores.gather(1, by_score)


def _pack_proposal(
    *,
    attention_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = scored_mask(attention_mask)
    k = int(candidate_ids.shape[1])
    if candidate_ids.shape != candidate_scores.shape:
        raise ComponentCrossoverError("proposal IDs and scores differ")
    if candidate_ids.shape[0] != int(mask.sum().item()):
        raise ComponentCrossoverError("proposal count differs from scored positions")
    if confidence.shape != (candidate_ids.shape[0],):
        raise ComponentCrossoverError("proposal confidence geometry differs")
    shape = (*attention_mask.shape, k)
    ids = torch.full(shape, INVALID_TOKEN_ID, dtype=torch.long)
    scores = torch.full(shape, float("-inf"), dtype=torch.float32)
    packed_confidence = torch.full(attention_mask.shape, float("nan"), dtype=torch.float32)
    ids[mask] = candidate_ids.to(torch.long)
    scores[mask] = candidate_scores.to(torch.float32)
    packed_confidence[mask] = confidence.to(torch.float32)
    return ids, scores, packed_confidence


@torch.inference_mode()
def propose_residual_affine(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    inverse: torch.nn.Module,
    embedding_table: torch.Tensor,
    max_k: int = 512,
) -> ProposalResult:
    """Rank the full vocabulary with the existing residual-affine inverse."""

    if observations.shape[:2] != attention_mask.shape or max_k < max(BUDGETS):
        raise ComponentCrossoverError("residual proposal geometry or maximum K changed")
    mask = scored_mask(attention_mask)
    flat = observations[mask].to(embedding_table.device)
    if embedding_table.is_cuda:
        torch.cuda.synchronize(embedding_table.device)
    started = time.perf_counter()
    queries = inverse(flat)
    ids, scores = topk_candidates(
        queries,
        embedding_table,
        k=max_k,
        score_batch_size=64,
    )
    ids, scores = _stable_score_order(ids, scores)
    if embedding_table.is_cuda:
        torch.cuda.synchronize(embedding_table.device)
    elapsed = time.perf_counter() - started
    # Cosine proposal scores have no fitted vocabulary temperature. This value is
    # diagnostic only and is never used by a selector.
    confidence = torch.softmax(scores, dim=1)[:, 0]
    packed = _pack_proposal(
        attention_mask=attention_mask,
        candidate_ids=ids,
        candidate_scores=scores,
        confidence=confidence,
    )
    return ProposalResult(*packed, elapsed)


@torch.inference_mode()
def propose_public_a1(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    lens: torch.nn.Module,
    normalized_embeddings: torch.Tensor,
    max_k: int = 512,
    chunk: int = 256,
) -> ProposalResult:
    """Rank the full vocabulary with the pinned public Alpaca A1 lens."""

    if observations.shape[:2] != attention_mask.shape or max_k != 512 or chunk != 256:
        raise ComponentCrossoverError("public A1 proposal contract changed")
    mask = scored_mask(attention_mask)
    flat = observations[mask]
    all_ids: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_confidence: list[torch.Tensor] = []
    if normalized_embeddings.is_cuda:
        torch.cuda.synchronize(normalized_embeddings.device)
    started = time.perf_counter()
    for start in range(0, flat.shape[0], chunk):
        logits = lens(
            flat[start : start + chunk].to(normalized_embeddings.device),
            normalized_embeddings,
        ).float()
        scores, ids = torch.topk(logits, k=max_k, dim=1, largest=True, sorted=True)
        confidence = torch.exp(scores[:, 0] - torch.logsumexp(logits, dim=1))
        ids, scores = _stable_score_order(ids.cpu(), scores.float().cpu())
        all_ids.append(ids)
        all_scores.append(scores)
        all_confidence.append(confidence.float().cpu())
    if normalized_embeddings.is_cuda:
        torch.cuda.synchronize(normalized_embeddings.device)
    elapsed = time.perf_counter() - started
    packed = _pack_proposal(
        attention_mask=attention_mask,
        candidate_ids=torch.cat(all_ids, dim=0),
        candidate_scores=torch.cat(all_scores, dim=0),
        confidence=torch.cat(all_confidence, dim=0),
    )
    return ProposalResult(*packed, elapsed)


def round_robin_union(
    *,
    a1_candidates: torch.Tensor,
    residual_candidates: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Build the preregistered A1-first alternating deduplicated union."""

    if k not in BUDGETS:
        raise ComponentCrossoverError("union budget is not preregistered")
    if a1_candidates.shape[:2] != attention_mask.shape:
        raise ComponentCrossoverError("A1 union geometry changed")
    if residual_candidates.shape[:2] != attention_mask.shape:
        raise ComponentCrossoverError("residual union geometry changed")
    if a1_candidates.shape[2] < k or residual_candidates.shape[2] < k:
        raise ComponentCrossoverError("union sources are shallower than K")
    output = torch.full((*attention_mask.shape, k), INVALID_TOKEN_ID, dtype=torch.long)
    mask = scored_mask(attention_mask)
    for row, position in torch.nonzero(mask, as_tuple=False).tolist():
        selected: list[int] = []
        seen: set[int] = set()
        maximum = max(a1_candidates.shape[2], residual_candidates.shape[2])
        for rank in range(maximum):
            for source in (a1_candidates, residual_candidates):
                if rank >= source.shape[2]:
                    continue
                token = int(source[row, position, rank].item())
                if token >= 0 and token not in seen:
                    seen.add(token)
                    selected.append(token)
                    if len(selected) == k:
                        break
            if len(selected) == k:
                break
        if len(selected) != k:
            raise ComponentCrossoverError("union could not fill its candidate budget")
        output[row, position] = torch.tensor(selected, dtype=torch.long)
    return output


def _repeat_cache(cache: Any, repeats: int) -> Any:
    candidate_cache = copy.deepcopy(cache)
    repeat = getattr(candidate_cache, "batch_repeat_interleave", None)
    if not callable(repeat):
        raise ComponentCrossoverError("public-prefix cache cannot repeat candidates")
    repeat(repeats)
    return candidate_cache


@torch.inference_mode()
def select_fixed_budget(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    candidates: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
    selector: str,
    record_batch_size: int,
) -> SelectorResult:
    """Apply the frozen causal or fixed-budget A2 selector."""

    validate_observations(observations, attention_mask, position_ids)
    if selector not in {"causal", "a2_fixed_budget"}:
        raise ComponentCrossoverError("selector is not preregistered")
    if candidates.shape[:2] != attention_mask.shape:
        raise ComponentCrossoverError("candidate geometry differs from observations")
    k = int(candidates.shape[2])
    if k not in BUDGETS or not 0 < record_batch_size <= 16:
        raise ComponentCrossoverError("selector budget or record batch changed")
    if record_batch_size * k > 256:
        raise ComponentCrossoverError("selector candidate batch cap exceeded")

    predictions = torch.full(attention_mask.shape, INVALID_TOKEN_ID, dtype=torch.long)
    predictions[:, 0] = BOS_TOKEN_ID
    scores_out = torch.full(candidates.shape, float("-inf"), dtype=torch.float32)
    margins = torch.full(attention_mask.shape, float("nan"), dtype=torch.float32)
    normalized = torch.full(attention_mask.shape, float("nan"), dtype=torch.float32)
    routes = torch.full(attention_mask.shape, ROUTE_PADDING, dtype=torch.int8)
    routes[:, 0] = ROUTE_BOS
    logical_simulations = 0
    executed_simulations = 0
    prefix_commit_tokens = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for record_start in range(0, observations.shape[0], record_batch_size):
        record_end = min(record_start + record_batch_size, observations.shape[0])
        record_count = record_end - record_start
        batch_mask = attention_mask[record_start:record_end].to(torch.bool)
        stopped = torch.zeros(record_count, dtype=torch.bool)
        maximum_length = int(batch_mask.sum(dim=1).max().item())
        cache = precut.new_cache()
        precut.run_cached(
            torch.full((record_count, 1), BOS_TOKEN_ID, dtype=torch.long, device=device),
            cache,
            0,
        )
        prefix_commit_tokens += record_count

        for position in range(1, maximum_length):
            active = batch_mask[:, position]
            eligible = active if selector == "causal" else active & ~stopped
            if selector == "a2_fixed_budget":
                suffix = active & stopped
                routes[record_start:record_end, position][suffix] = ROUTE_ABSTAINED_SUFFIX
            if not eligible.any().item():
                precut.run_cached(
                    torch.full(
                        (record_count, 1),
                        BOS_TOKEN_ID,
                        dtype=torch.long,
                        device=device,
                    ),
                    cache,
                    position,
                )
                prefix_commit_tokens += record_count
                continue

            ids = candidates[record_start:record_end, position].to(
                device=device,
                dtype=torch.long,
            )
            if ids[eligible].lt(0).any().item():
                raise ComponentCrossoverError("eligible candidate row contains invalid IDs")
            ids = ids.clone()
            ids[~eligible.to(device)] = BOS_TOKEN_ID
            candidate_cache = _repeat_cache(cache, k)
            simulated = precut.run_cached(
                ids.reshape(-1, 1),
                candidate_cache,
                position,
            )[:, -1].reshape(record_count, k, -1).float()
            target = observations[record_start:record_end, position].to(device).float()
            if selector == "causal":
                score = F.cosine_similarity(simulated, target[:, None, :], dim=-1)
            else:
                mean = simulated.mean(dim=1, keepdim=True)
                score = F.cosine_similarity(
                    simulated - mean,
                    target[:, None, :] - mean,
                    dim=-1,
                )
            if not torch.isfinite(score[eligible.to(device)]).all().item():
                raise ComponentCrossoverError("eligible selection scores are non-finite")
            choice = score.argmax(dim=1)
            chosen = ids.gather(1, choice[:, None]).squeeze(1)
            top_two = score.topk(k=2, dim=1).values
            margin = top_two[:, 0] - top_two[:, 1]
            confidence = k * torch.softmax(score.float(), dim=1).max(dim=1).values
            accepted = eligible.clone()
            if selector == "a2_fixed_budget":
                accepted &= confidence.cpu().ge(A2_NORMALIZED_WINNER_MIN)
                rejected = eligible & ~accepted
                routes[record_start:record_end, position][rejected] = ROUTE_ABSTAIN
                stopped |= rejected
            else:
                rejected = torch.zeros_like(eligible)

            prediction_view = predictions[record_start:record_end, position]
            prediction_view[accepted] = chosen[accepted.to(device)].cpu()
            route_view = routes[record_start:record_end, position]
            route_view[accepted] = ROUTE_ACCEPT
            score_view = scores_out[record_start:record_end, position]
            score_view[eligible] = score[eligible.to(device)].cpu()
            margin_view = margins[record_start:record_end, position]
            margin_view[eligible] = margin[eligible.to(device)].cpu()
            confidence_view = normalized[record_start:record_end, position]
            confidence_view[eligible] = confidence[eligible.to(device)].cpu()

            commit = torch.full(
                (record_count,),
                BOS_TOKEN_ID,
                dtype=torch.long,
                device=device,
            )
            commit[accepted.to(device)] = chosen[accepted.to(device)]
            precut.run_cached(commit[:, None], cache, position)
            logical_simulations += int(eligible.sum().item()) * k
            executed_simulations += record_count * k
            prefix_commit_tokens += record_count
            del candidate_cache, simulated, target, score, top_two, confidence
        del cache

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return SelectorResult(
        predictions=predictions,
        scores=scores_out,
        winner_margin=margins,
        normalized_winner=normalized,
        routes=routes,
        elapsed_seconds=elapsed,
        candidate_simulations=logical_simulations,
        executed_candidate_simulations=executed_simulations,
        prefix_commit_tokens=prefix_commit_tokens,
    )


def prediction_from_rank_one(
    candidates: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if candidates.shape[:2] != attention_mask.shape:
        raise ComponentCrossoverError("rank-one proposal geometry changed")
    predictions = torch.full(attention_mask.shape, INVALID_TOKEN_ID, dtype=torch.long)
    predictions[:, 0] = BOS_TOKEN_ID
    mask = scored_mask(attention_mask)
    predictions[mask] = candidates[:, :, 0][mask].to(torch.long)
    return predictions


def true_token_ranks(
    *,
    candidates: torch.Tensor,
    truth: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one-based true-token rank, or zero when absent."""

    if candidates.shape[:2] != truth.shape or truth.shape != attention_mask.shape:
        raise ComponentCrossoverError("true-rank geometry changed")
    ranks = torch.zeros(truth.shape, dtype=torch.long)
    mask = scored_mask(attention_mask)
    proposed = candidates[mask].to(torch.long)
    expected = truth[mask].to(torch.long)
    matches = proposed.eq(expected[:, None])
    hit = matches.any(dim=1)
    values = torch.zeros(expected.shape, dtype=torch.long)
    values[hit] = matches[hit].to(torch.long).argmax(dim=1) + 1
    ranks[mask] = values
    return ranks


def quantile_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    finite = values.detach().float().flatten()
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {"count": 0, "minimum": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None, "p95": None, "p99": None, "maximum": None, "mean": None}
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]),
    )
    names = ("minimum", "p10", "p25", "median", "p75", "p90", "p95", "p99", "maximum")
    result: dict[str, float | int | None] = {"count": int(finite.numel())}
    result.update({name: float(value.item()) for name, value in zip(names, quantiles)})
    result["mean"] = float(finite.mean().item())
    return result


def rank_summary(
    ranks: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    budgets: Sequence[int] = (8, 16, 32, 64, 128, 512),
) -> dict[str, Any]:
    mask = scored_mask(attention_mask)
    values = ranks[mask]
    total = int(values.numel())
    hits = values.gt(0)
    return {
        "scored_tokens": total,
        "found_within_maximum": int(hits.sum().item()),
        "recall_within_maximum": float(hits.float().mean().item()),
        "recall_at": {
            str(k): float(((values.gt(0)) & values.le(k)).float().mean().item())
            for k in budgets
            if k <= int(max(budgets))
        },
        "rank_when_found": quantile_summary(values[hits].float()),
    }


def selector_error_attribution(
    *,
    predictions: torch.Tensor,
    truth: torch.Tensor,
    attention_mask: torch.Tensor,
    candidates: torch.Tensor,
) -> dict[str, int | float | None]:
    mask = scored_mask(attention_mask)
    expected = truth[mask].to(torch.long)
    predicted = predictions[mask].to(torch.long)
    proposed = candidates[mask].to(torch.long)
    included = proposed.eq(expected[:, None]).any(dim=1)
    covered = predicted.ge(0)
    correct = covered & predicted.eq(expected)
    included_count = int(included.sum().item())
    conditional_correct = int((correct & included).sum().item())
    return {
        "scored_tokens": int(expected.numel()),
        "proposal_exclusions": int((~included).sum().item()),
        "included_tokens": included_count,
        "correct_with_true_token_in_candidates": conditional_correct,
        "selector_errors_or_abstentions_given_inclusion": int((included & ~correct).sum().item()),
        "selector_accuracy_given_inclusion": conditional_correct / included_count if included_count else None,
        "abstentions": int((~covered).sum().item()),
    }


def check_finite_scalar(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ComponentCrossoverError(f"{name} is invalid")
