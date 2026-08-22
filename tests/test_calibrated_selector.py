from __future__ import annotations

import torch

from token_reconstruction.calibrated_selector import (
    ROUTE_BASE,
    ROUTE_EXPANDED,
    scale_normalized_gap,
    select_calibrated_adaptive,
)


class FakeCache:
    def __init__(self) -> None:
        self.repeats = 1

    def __deepcopy__(self, memo):
        del memo
        clone = FakeCache()
        clone.repeats = self.repeats
        return clone

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.repeats *= repeats


class FakePrecut:
    def __init__(self) -> None:
        self.embedding = torch.zeros((128, 2048), dtype=torch.float32)
        for token in range(128):
            self.embedding[token, token] = 1.0

    def new_cache(self) -> FakeCache:
        return FakeCache()

    def run_cached(
        self,
        input_ids: torch.Tensor,
        cache: FakeCache,
        start_pos: int,
    ) -> torch.Tensor:
        del cache, start_pos
        mapped = input_ids.cpu().clone()
        mapped[mapped == 128000] = 0
        return self.embedding[mapped].to(input_ids.device)


def test_normalized_gap_is_shift_and_positive_scale_invariant() -> None:
    scores = torch.linspace(-0.3, 0.8, 32).view(1, 32)
    expected = scale_normalized_gap(scores)
    actual = scale_normalized_gap(scores * 7.5 + 19.0)
    assert torch.allclose(expected, actual, atol=2e-6, rtol=0.0)


def test_adaptive_selector_expands_without_abstaining() -> None:
    mask = torch.tensor([[1, 1]], dtype=torch.long)
    positions = torch.tensor([[0, 1]], dtype=torch.long)
    observations = torch.zeros((1, 2, 2048), dtype=torch.float32)
    observations[0, 1, 50] = 1.0
    candidates = torch.full((1, 2, 64), -1, dtype=torch.long)
    candidates[0, 1] = torch.arange(64)

    base = select_calibrated_adaptive(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        candidates=candidates,
        precut=FakePrecut(),
        device=torch.device("cpu"),
        threshold=-1.0,
        record_batch_size=1,
    )
    expanded = select_calibrated_adaptive(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        candidates=candidates,
        precut=FakePrecut(),
        device=torch.device("cpu"),
        threshold=0.0,
        record_batch_size=1,
    )

    assert base.predictions.tolist() == [[128000, 0]]
    assert base.routes[0, 1].item() == ROUTE_BASE
    assert base.extra_candidate_simulations == 0
    assert expanded.predictions.tolist() == [[128000, 50]]
    assert expanded.routes[0, 1].item() == ROUTE_EXPANDED
    assert expanded.extra_candidate_simulations == 32
    assert expanded.predictions.ge(0).all().item()
