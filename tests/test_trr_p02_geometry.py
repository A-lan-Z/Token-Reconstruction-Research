"""CPU-only tests for the model-free TRR-P02 geometry helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "trr_p02"))
from diagnose_geometry import (  # noqa: E402
    DiagnosticError,
    _restricted_rank,
    _top_k_neighbors,
)

from token_reconstruction.trr_p02 import (
    ContextSpec,
    GeometryDiagnosticError,
    pairwise_token_deformation,
    rank_metrics,
    reference_corrected_query,
    separation_summary,
    summarize_offsets,
)


def test_context_spec_requires_bos_and_valid_ids() -> None:
    ContextSpec("ok", (128000, 13)).validate(bos_token_id=128000, vocab_size=128256)
    with pytest.raises(GeometryDiagnosticError, match="BOS"):
        ContextSpec("bad", (13,)).validate(bos_token_id=128000, vocab_size=128256)
    with pytest.raises(GeometryDiagnosticError, match="outside"):
        ContextSpec("bad", (128000, 128256)).validate(
            bos_token_id=128000, vocab_size=128256
        )


def test_shared_offset_summary_and_pair_deformation_detect_token_dependence() -> None:
    baseline = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )
    # Context 1 has a common [0.5,0.5] shift plus a token-specific [0.3,0].
    panel = torch.cat(
        (
            baseline,
            baseline + torch.tensor([[[0.8, 0.5], [0.5, 0.5]]]),
        ),
        dim=0,
    )
    offsets = panel - panel[0:1]
    summary = summarize_offsets(offsets)
    assert summary["geometry"] == {"contexts": 2, "tokens": 2, "hidden": 2}
    assert summary["context_rows"][0]["mean_offset_norm"] == 0.0
    assert summary["context_rows"][1]["residual_norm"]["max"] > 0.0

    pairs = pairwise_token_deformation(panel, token_ids=(10, 11))
    context_one = [row for row in pairs["pairs"] if row["context_index"] == 1]
    assert len(context_one) == 1
    assert context_one[0]["deformation_norm"] > 0.0


def test_reference_sign_control_and_chunked_rank_metrics() -> None:
    observation = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    reference_output = torch.tensor([[1.5, 1.0]], dtype=torch.float32)
    reference_prototype = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    minus = reference_corrected_query(
        observation, reference_output, reference_prototype, sign=-1
    )
    plus = reference_corrected_query(
        observation, reference_output, reference_prototype, sign=1
    )
    assert torch.equal(minus, torch.tensor([[1.5, 1.0]]))
    assert torch.equal(plus, torch.tensor([[2.5, 1.0]]))

    prototypes = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32
    )
    result = rank_metrics(
        torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32),
        prototypes,
        [0, 1],
        query_chunk_size=1,
        prototype_chunk_size=1,
    )
    assert result["top1_ids"].tolist() == [0, 1]
    assert result["true_rank"].tolist() == [1, 1]
    assert torch.all(result["top1_runner_margin"] > 0)


def test_rank_metrics_counts_earlier_blocks_when_true_is_in_last_chunk() -> None:
    # IDs 0/1 are tied winners in the first block.  The true ID 3 is in the
    # final block and ties ID 2; strict rank must count both earlier winners.
    prototypes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    result = rank_metrics(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        prototypes,
        [3],
        query_chunk_size=1,
        prototype_chunk_size=2,
    )
    assert result["top1_ids"].tolist() == [0]
    assert result["runner_up_ids"].tolist() == [1]
    assert result["true_rank"].tolist() == [3]
    assert result["true_equal_count"].tolist() == [2]
    assert result["best_other_scores"].tolist() == [1.0]


def test_local_dictionary_uses_n8_other_plus_true_and_ranks_competitors() -> None:
    prototypes = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=torch.float32,
    )
    query_ids = [0, 1]
    neighbors, scores = _top_k_neighbors(
        prototypes[query_ids],
        prototypes,
        query_token_ids=query_ids,
        k=2,
        prototype_chunk_size=2,
    )
    assert neighbors.shape == (2, 2)
    assert scores.shape == (2, 2)
    assert not neighbors.eq(torch.tensor(query_ids)[:, None]).any().item()
    dictionary = torch.cat((neighbors, torch.tensor(query_ids)[:, None]), dim=1)
    assert dictionary.shape == (2, 3)
    assert all(int(query) in dictionary[row].tolist() for row, query in enumerate(query_ids))
    result = _restricted_rank(
        prototypes[query_ids], query_ids, dictionary, prototypes
    )
    assert result["top1_ids"].tolist() == query_ids
    assert result["true_rank"].tolist() == [1, 1]
    assert result["true_equal_count"].tolist() == [1, 1]


def test_restricted_rank_rejects_missing_true_label() -> None:
    prototypes = torch.eye(3, dtype=torch.float32)
    with pytest.raises(DiagnosticError, match="missing"):
        _restricted_rank(
            prototypes[[0]],
            [0],
            torch.tensor([[1, 2]], dtype=torch.long),
            prototypes,
        )


def test_rank_metrics_stable_ties_and_separation() -> None:
    prototypes = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    result = rank_metrics(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        prototypes,
        [0],
        query_chunk_size=1,
        prototype_chunk_size=1,
    )
    assert result["top1_ids"].tolist() == [0]
    assert result["runner_up_ids"].tolist() == [1]
    assert result["true_equal_count"].tolist() == [2]

    panel = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.1], [0.1, 1.0]],
        ],
        dtype=torch.float32,
    )
    spread = separation_summary(panel)
    assert spread["same_token_cross_context"]["l2"]["mean"] > 0.0
    assert spread["different_token_within_context"]["l2"]["mean"] > 0.0
