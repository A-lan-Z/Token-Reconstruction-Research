"""Small CPU checks for the frozen historical comparator contract."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from token_reconstruction.trr_p01.historical_comparators import (
    HistoricalComparatorError,
    run_fixed_k256_a1_a2,
)


class _SmallPrefix(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(2, 2)
        self.anchor = nn.Parameter(torch.zeros(1))
        self.cut_depth = 4

    def new_cache(self):  # pragma: no cover - geometry guard fires first
        raise AssertionError("cache should not be touched by the guard")

    def run_cached(self, *args, **kwargs):  # pragma: no cover - geometry guard fires first
        raise AssertionError("cache should not be touched by the guard")


def test_historical_comparator_rejects_non_pilot_public_geometry() -> None:
    observations = torch.zeros((1, 40, 2048), dtype=torch.bfloat16)
    with pytest.raises(HistoricalComparatorError, match="pinned public vocabulary geometry"):
        run_fixed_k256_a1_a2(
            observations=observations,
            public_prefix=_SmallPrefix().eval(),
            frozen_lens=nn.Identity().eval(),
            device=torch.device("cpu"),
        )


def test_historical_comparator_rejects_invalid_observation_geometry_before_model_calls() -> None:
    observations = torch.zeros((1, 39, 2048), dtype=torch.bfloat16)
    with pytest.raises(HistoricalComparatorError, match=r"\[records,40,2048\]"):
        run_fixed_k256_a1_a2(
            observations=observations,
            public_prefix=_SmallPrefix(),
            frozen_lens=nn.Identity(),
            device=torch.device("cpu"),
        )


def test_historical_comparator_rejects_trainable_lens_before_prefix_calls() -> None:
    observations = torch.zeros((1, 40, 2048), dtype=torch.bfloat16)
    with pytest.raises(HistoricalComparatorError, match="eval mode"):
        run_fixed_k256_a1_a2(
            observations=observations,
            public_prefix=_SmallPrefix().eval(),
            frozen_lens=nn.Identity(),
            device=torch.device("cpu"),
        )
