"""Bounded ranking tests, including ties split across candidate chunks."""

from __future__ import annotations

import pytest
import torch

from token_reconstruction.trr_p03.ranking import (
    RankingError,
    rank_from_score_blocks,
    rank_queries,
)


def test_lowest_id_ties_are_stable_across_candidate_chunk_boundaries() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    prototypes = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    result = rank_queries(
        queries,
        prototypes,
        query_chunk_size=1,
        prototype_chunk_size=2,
    )

    assert result.top1_ids.tolist() == [0, 1]
    assert result.runner_up_ids.tolist() == [2, 4]
    assert result.top1_tie_count.tolist() == [3, 3]
    assert result.margins.tolist() == pytest.approx([0.0, 0.0])
    result.validate(query_count=2, vocab_size=7)


def test_custom_score_blocks_use_the_same_deterministic_merge() -> None:
    queries = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    prototypes = torch.arange(6, dtype=torch.float32).reshape(6, 1)

    def score_fn(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # Deliberate plateaus exercise lowest-ID selection without relying on
        # cosine normalization or an implementation-specific sort.
        values = torch.zeros((left.shape[0], right.shape[0]), dtype=torch.float32)
        values[left[:, 0] > 0, :] = right[:, 0]
        return values

    result = rank_from_score_blocks(
        queries,
        prototypes,
        score_fn,
        query_chunk_size=2,
        prototype_chunk_size=3,
    )

    assert result.top1_ids.tolist() == [0, 5]
    assert result.runner_up_ids.tolist() == [1, 4]
    assert result.top1_tie_count.tolist() == [6, 1]


def test_ranking_rejects_invalid_geometry_and_nonfinite_scores() -> None:
    queries = torch.ones((1, 2), dtype=torch.float32)
    prototypes = torch.ones((2, 2), dtype=torch.float32)
    with pytest.raises(RankingError, match="at least two"):
        rank_queries(queries, prototypes[:1])
    with pytest.raises(RankingError, match="non-finite"):
        rank_queries(
            queries,
            prototypes,
            score_fn=lambda left, right: torch.full(
                (left.shape[0], right.shape[0]), float("nan")
            ),
        )
