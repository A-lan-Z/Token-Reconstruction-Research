from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from token_reconstruction.checkpoint_inverse import (
    CheckpointInverseError,
    clamp_known_bos,
    forward_public_embeddings,
    invert_public_prefix,
    nearest_public_embeddings,
)


class FakeRotary:
    def __call__(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        return (position_ids, position_ids)


class FakeAttention(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(float(factor), dtype=torch.float32))

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: torch.Tensor,
        past_key_values=None,
    ) -> torch.Tensor:
        assert hidden_states.ndim == 3
        assert attention_mask.shape[-2:] == (hidden_states.shape[1], hidden_states.shape[1])
        assert position_embeddings is not None
        assert past_key_values is None
        return hidden_states * self.factor.to(hidden_states.dtype)


class FakeLayer(nn.Module):
    def __init__(self, attention_factor: float, mlp_factor: float) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.self_attn = FakeAttention(attention_factor)
        self.mlp = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(torch.eye(3) * float(mlp_factor))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
        position_embeddings,
    ) -> torch.Tensor:
        del position_ids, use_cache
        attention = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        after_attention = hidden_states + attention
        return after_attention + self.mlp(after_attention)


class FakePrefix(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeLayer(0.1, 0.2), FakeLayer(0.05, 0.1)]
        )
        self.rotary_emb = FakeRotary()


def test_reverse_residual_inverse_recovers_linear_prefix() -> None:
    torch.manual_seed(17)
    prefix = FakePrefix().eval()
    embeddings = torch.randn(2, 4, 3)
    observed = forward_public_embeddings(prefix, embeddings)

    result = invert_public_prefix(prefix, observed, iterations=32, damping=0.5)

    assert result.all_finite
    assert result.embedding_estimate.shape == embeddings.shape
    assert torch.allclose(result.embedding_estimate, embeddings, atol=2e-5, rtol=2e-5)
    reconstructed = forward_public_embeddings(prefix, result.embedding_estimate)
    assert torch.allclose(reconstructed, observed, atol=2e-5, rtol=2e-5)
    assert len(result.branch_stats_reverse_order) == 4
    assert all(len(item.steps) == 32 for item in result.branch_stats_reverse_order)
    assert all(
        item.steps[-1].relative_residual <= item.steps[0].relative_residual
        for item in result.branch_stats_reverse_order
    )


def test_fixed_point_stats_expose_nonmonotone_but_finite_progress() -> None:
    prefix = FakePrefix().eval()
    observed = forward_public_embeddings(prefix, torch.ones(1, 3, 3))
    result = invert_public_prefix(prefix, observed, iterations=4, damping=0.5)

    assert result.all_finite
    residuals = [
        step.relative_residual
        for branch in result.branch_stats_reverse_order
        for step in branch.steps
    ]
    assert all(math.isfinite(value) and value >= 0 for value in residuals)
    assert any(
        branch.steps[-1].relative_update > 0
        for branch in result.branch_stats_reverse_order
    )


def test_chunked_projection_matches_dense_euclidean_and_cosine() -> None:
    torch.manual_seed(23)
    weight = torch.randn(19, 5)
    query = torch.randn(3, 2, 5)
    for normalize in (False, True):
        chunked_ids, chunked_distances = nearest_public_embeddings(
            query,
            weight,
            top_k=5,
            vocab_chunk_size=4,
            normalize=normalize,
        )
        if normalize:
            dense_query = torch.nn.functional.normalize(query.reshape(-1, 5), dim=-1)
            dense_weight = torch.nn.functional.normalize(weight, dim=-1)
            dense_distances = 1 - dense_query @ dense_weight.T
        else:
            dense_query = query.reshape(-1, 5)
            dense_distances = torch.cdist(dense_query, weight).square()
        dense_distances, dense_ids = torch.sort(dense_distances, dim=1, stable=True)
        dense_ids = dense_ids[:, :5].reshape_as(chunked_ids)
        dense_distances = dense_distances[:, :5].reshape_as(chunked_distances)
        assert torch.equal(chunked_ids, dense_ids)
        assert torch.allclose(chunked_distances, dense_distances, atol=2e-5, rtol=2e-5)


def test_projection_tie_breaks_by_token_id() -> None:
    weight = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([[0.0, 0.0]])
    ids, distances = nearest_public_embeddings(
        query, weight, top_k=3, vocab_chunk_size=2
    )
    assert ids.tolist() == [[0, 1, 2]]
    assert distances.tolist() == [[1.0, 1.0, 1.0]]


def test_bos_clamp_only_changes_first_position() -> None:
    estimate = torch.zeros(1, 3, 2)
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    clamped = clamp_known_bos(estimate, weight, bos_token_id=1)
    assert clamped.tolist() == [[[3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]]
    assert estimate.sum().item() == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0, "damping": 0.5},
        {"iterations": 2, "damping": 0.0},
        {"iterations": 2, "damping": 1.1},
    ],
)
def test_inverse_rejects_invalid_controls(kwargs: dict[str, float | int]) -> None:
    prefix = FakePrefix().eval()
    observed = torch.ones(1, 2, 3)
    with pytest.raises(CheckpointInverseError):
        invert_public_prefix(prefix, observed, **kwargs)


def test_inverse_rejects_nonfinite_projection() -> None:
    with pytest.raises(CheckpointInverseError, match="non-finite"):
        nearest_public_embeddings(
            torch.tensor([[float("nan"), 0.0]]),
            torch.eye(2),
            top_k=1,
        )
