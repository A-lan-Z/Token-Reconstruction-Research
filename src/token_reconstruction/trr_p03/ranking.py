"""Deterministic bounded full-vocabulary ranking for TRR-P03.

Scores are higher-is-better.  Candidate IDs are the ascending vocabulary
indices, and exact finite-precision score ties are resolved by the lowest ID.
The merge keeps only the best two candidates per query while scanning bounded
candidate blocks; it never sorts a full vocabulary row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


class RankingError(RuntimeError):
    """Raised when a ranking tensor violates the readout contract."""


@dataclass(frozen=True)
class RankResult:
    """Top-two ranking diagnostics for a flat query matrix.

    All fields are one-dimensional CPU tensors.  ``top1_tie_count`` counts
    candidates whose score is exactly equal to the finite-precision top score;
    it is independent of any truth-relative tie definition.
    """

    top1_ids: torch.Tensor
    top1_scores: torch.Tensor
    runner_up_ids: torch.Tensor
    runner_up_scores: torch.Tensor
    margins: torch.Tensor
    top1_tie_count: torch.Tensor

    def validate(
        self, *, query_count: int | None = None, vocab_size: int | None = None
    ) -> None:
        fields = (
            self.top1_ids,
            self.top1_scores,
            self.runner_up_ids,
            self.runner_up_scores,
            self.margins,
            self.top1_tie_count,
        )
        if any(not isinstance(value, torch.Tensor) or value.ndim != 1 for value in fields):
            raise RankingError("rank result fields must be one-dimensional tensors")
        if self.top1_ids.dtype != torch.long or self.runner_up_ids.dtype != torch.long:
            raise RankingError("rank result IDs must be int64 tensors")
        lengths = {int(value.numel()) for value in fields}
        if len(lengths) != 1 or (
            query_count is not None and lengths != {int(query_count)}
        ):
            raise RankingError("rank result field lengths differ")
        if not torch.isfinite(self.top1_scores).all().item() or not torch.isfinite(
            self.runner_up_scores
        ).all().item():
            raise RankingError("rank scores are non-finite")
        if not torch.isfinite(self.margins).all().item():
            raise RankingError("rank margins are non-finite")
        if self.top1_tie_count.dtype != torch.long or self.top1_tie_count.lt(1).any().item():
            raise RankingError("every query must have a positive top-score count")
        if vocab_size is not None:
            if int(vocab_size) < 2:
                raise RankingError("ranking requires at least two candidates")
            for ids, name in (
                (self.top1_ids, "top-1"),
                (self.runner_up_ids, "runner-up"),
            ):
                if ids.lt(0).any().item() or ids.ge(int(vocab_size)).any().item():
                    raise RankingError(f"{name} candidate ID is outside the vocabulary")


def _validate_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RankingError(f"{name} must be a rank-2 tensor")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise RankingError(f"{name} must be non-empty")
    if not value.dtype.is_floating_point:
        raise RankingError(f"{name} must be floating point")
    if not torch.isfinite(value).all().item():
        raise RankingError(f"{name} contains non-finite values")
    return value.detach().cpu().float().contiguous()


def score_block(
    queries: torch.Tensor, prototypes: torch.Tensor, metric: str = "cosine"
) -> torch.Tensor:
    """Return a finite higher-is-better score matrix for one block pair."""

    q = _validate_matrix(queries, "queries")
    p = _validate_matrix(prototypes, "prototypes")
    if q.shape[1] != p.shape[1]:
        raise RankingError("query and prototype hidden widths differ")
    if metric == "cosine":
        result = F.normalize(q, dim=1, eps=1e-12) @ F.normalize(p, dim=1, eps=1e-12).T
    elif metric == "l2":
        q_sq = q.square().sum(dim=1, keepdim=True)
        p_sq = p.square().sum(dim=1, keepdim=True).T
        result = -(q_sq + p_sq - 2.0 * (q @ p.T))
    else:
        raise RankingError(f"unsupported ranking metric: {metric!r}")
    result = result.float()
    if not torch.isfinite(result).all().item():
        raise RankingError("score block contains non-finite values")
    return result


def _lowest_id_for_score(
    scores: torch.Tensor,
    ids: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Select the lowest ID among entries exactly equal to each target."""

    sentinel = torch.full_like(ids, torch.iinfo(ids.dtype).max)
    matching = scores.eq(target[:, None])
    return torch.where(matching, ids, sentinel).min(dim=1).values


def _stable_top_two(
    current_scores: torch.Tensor,
    current_ids: torch.Tensor,
    new_scores: torch.Tensor,
    new_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge candidates using max passes and lowest-ID exact tie handling."""

    if current_scores.ndim != 2 or new_scores.ndim != 2:
        raise RankingError("ranking merge scores must be rank-2")
    if current_scores.shape[0] != new_scores.shape[0]:
        raise RankingError("ranking merge row counts differ")
    if current_ids.shape != current_scores.shape or new_ids.shape != new_scores.shape:
        raise RankingError("ranking merge IDs and scores differ")
    scores = torch.cat((current_scores, new_scores), dim=1).float()
    ids = torch.cat((current_ids, new_ids), dim=1).long()
    best_scores = scores.max(dim=1).values
    best_ids = _lowest_id_for_score(scores, ids, best_scores)
    remaining = scores.masked_fill(ids.eq(best_ids[:, None]), -float("inf"))
    runner_scores = remaining.max(dim=1).values
    runner_ids = _lowest_id_for_score(remaining, ids, runner_scores)
    return torch.stack((best_scores, runner_scores), dim=1), torch.stack(
        (best_ids, runner_ids), dim=1
    )


def _merge_tie_counts(
    current_best: torch.Tensor,
    current_count: torch.Tensor,
    block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update exact top-score multiplicity while scanning candidate blocks."""

    block_best = block.max(dim=1).values
    block_count = block.eq(block_best[:, None]).sum(dim=1).to(torch.long)
    improved = block_best > current_best
    tied = block_best == current_best
    best = torch.where(improved, block_best, current_best)
    count = torch.where(
        improved, block_count, torch.where(tied, current_count + block_count, current_count)
    )
    return best, count


def _positive_int(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RankingError(f"{name} must be a positive integer")
    return value


def rank_queries(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    metric: str = "cosine",
    query_chunk_size: int = 256,
    prototype_chunk_size: int = 8192,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> RankResult:
    """Rank every query against a full vocabulary in bounded blocks.

    A custom ``score_fn`` receives CPU float32 query/prototype blocks and must
    return a finite ``[query_rows, prototype_rows]`` higher-is-better matrix.
    """

    q = _validate_matrix(queries, "queries")
    p = _validate_matrix(prototypes, "prototypes")
    if q.shape[1] != p.shape[1]:
        raise RankingError("query and prototype hidden widths differ")
    query_chunk_size = _positive_int(query_chunk_size, name="query_chunk_size")
    prototype_chunk_size = _positive_int(
        prototype_chunk_size, name="prototype_chunk_size"
    )
    if int(p.shape[0]) < 2:
        raise RankingError("ranking requires at least two candidates")
    if score_fn is not None and not callable(score_fn):
        raise RankingError("score_fn must be callable")
    scorer = score_fn or (lambda left, right: score_block(left, right, metric))
    rows = int(q.shape[0])
    vocab = int(p.shape[0])
    predictions: list[torch.Tensor] = []
    top_scores: list[torch.Tensor] = []
    runner_ids: list[torch.Tensor] = []
    runner_scores: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    tie_counts: list[torch.Tensor] = []

    for q_start in range(0, rows, query_chunk_size):
        q_block = q[q_start : q_start + query_chunk_size]
        q_rows = int(q_block.shape[0])
        best_scores = torch.full((q_rows, 2), -float("inf"), dtype=torch.float32)
        best_ids = torch.full((q_rows, 2), vocab, dtype=torch.long)
        best_value = torch.full((q_rows,), -float("inf"), dtype=torch.float32)
        best_count = torch.zeros((q_rows,), dtype=torch.long)

        for p_start in range(0, vocab, prototype_chunk_size):
            p_block = p[p_start : p_start + prototype_chunk_size]
            block = scorer(q_block, p_block)
            if not isinstance(block, torch.Tensor) or tuple(block.shape) != (
                q_rows,
                int(p_block.shape[0]),
            ):
                raise RankingError("score function returned invalid block geometry")
            block = block.detach().cpu().float().contiguous()
            if not torch.isfinite(block).all().item():
                raise RankingError("score function returned non-finite values")
            ids = torch.arange(
                p_start, p_start + p_block.shape[0], dtype=torch.long
            ).view(1, -1).expand(q_rows, -1)
            block_scores, block_ids = _stable_top_two(
                torch.full((q_rows, 2), -float("inf")),
                torch.full((q_rows, 2), vocab, dtype=torch.long),
                block,
                ids,
            )
            best_scores, best_ids = _stable_top_two(
                best_scores, best_ids, block_scores, block_ids
            )
            best_value, best_count = _merge_tie_counts(best_value, best_count, block)

        if best_ids[:, 0].ge(vocab).any().item() or best_count.lt(1).any().item():
            raise RankingError("ranking produced no valid candidate")
        predictions.append(best_ids[:, 0].cpu())
        top_scores.append(best_scores[:, 0].cpu())
        runner_ids.append(best_ids[:, 1].cpu())
        runner_scores.append(best_scores[:, 1].cpu())
        margins.append((best_scores[:, 0] - best_scores[:, 1]).cpu())
        tie_counts.append(best_count.cpu())

    result = RankResult(
        top1_ids=torch.cat(predictions),
        top1_scores=torch.cat(top_scores),
        runner_up_ids=torch.cat(runner_ids),
        runner_up_scores=torch.cat(runner_scores),
        margins=torch.cat(margins),
        top1_tie_count=torch.cat(tie_counts),
    )
    result.validate(query_count=rows, vocab_size=vocab)
    return result


def rank_from_score_blocks(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    query_chunk_size: int = 256,
    prototype_chunk_size: int = 8192,
) -> RankResult:
    """Convenience wrapper for callers with a transformed score function."""

    return rank_queries(
        queries,
        prototypes,
        query_chunk_size=query_chunk_size,
        prototype_chunk_size=prototype_chunk_size,
        score_fn=score_fn,
    )


__all__ = [
    "RankResult",
    "RankingError",
    "rank_from_score_blocks",
    "rank_queries",
    "score_block",
]
