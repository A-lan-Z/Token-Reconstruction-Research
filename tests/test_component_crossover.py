from __future__ import annotations

import copy

import torch

from token_reconstruction.component_crossover import (
    BASE_METHOD_IDS,
    BUDGETS,
    METHOD_IDS,
    ROUTE_ABSTAIN,
    ROUTE_ABSTAINED_SUFFIX,
    method_spec,
    rank_summary,
    round_robin_union,
    select_fixed_budget,
    selector_error_attribution,
    true_token_ranks,
)


class FakeCache:
    def __init__(self) -> None:
        self.repeats = 1

    def __deepcopy__(self, memo):
        clone = FakeCache()
        clone.repeats = self.repeats
        return clone

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.repeats *= repeats


class FakePrecut:
    def __init__(self, *, flat: bool = False) -> None:
        self.flat = flat
        self.embedding = torch.zeros((32, 2048), dtype=torch.float32)
        if flat:
            self.embedding[:, 0] = 1.0
        else:
            for token in range(32):
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


def fixture(k: int = 8):
    mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    observations = torch.zeros((1, 3, 2048), dtype=torch.float32)
    observations[0, 1, 3] = 1.0
    observations[0, 2, 5] = 1.0
    candidates = torch.full((1, 3, k), -1, dtype=torch.long)
    candidates[0, 1] = torch.tensor([2, 3, 4, 6, 7, 8, 9, 10])[:k]
    candidates[0, 2] = torch.tensor([2, 3, 4, 5, 7, 8, 9, 10])[:k]
    return observations, mask, positions, candidates


def test_registry_contains_complete_factorial_without_duplicate_alias() -> None:
    assert len(METHOD_IDS) == 22
    assert len(set(METHOD_IDS)) == 22
    assert set(BASE_METHOD_IDS).issubset(METHOD_IDS)
    assert method_spec("a1_a2_k32") == ("a1", "a2_fixed_budget", 32)
    assert method_spec("a1_residual_union_causal_k64") == (
        "a1_residual_union",
        "causal",
        64,
    )
    assert method_spec("causal_public_surrogate_k16") is None


def test_round_robin_union_is_a1_first_and_deduplicated() -> None:
    mask = torch.tensor([[1, 1]], dtype=torch.long)
    a1 = torch.full((1, 2, 8), -1, dtype=torch.long)
    residual = torch.full((1, 2, 8), -1, dtype=torch.long)
    a1[0, 1] = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    residual[0, 1] = torch.tensor([1, 9, 2, 10, 11, 12, 13, 14])
    union = round_robin_union(
        a1_candidates=a1,
        residual_candidates=residual,
        attention_mask=mask,
        k=8,
    )
    assert union[0, 1].tolist() == [1, 2, 9, 3, 4, 10, 5, 11]
    assert union[0, 0].eq(-1).all().item()


def test_causal_selector_chooses_matching_public_activation() -> None:
    observations, mask, positions, candidates = fixture()
    result = select_fixed_budget(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        candidates=candidates,
        precut=FakePrecut(),
        device=torch.device("cpu"),
        selector="causal",
        record_batch_size=1,
    )
    assert result.predictions.tolist() == [[128000, 3, 5]]
    assert result.candidate_simulations == 2 * 8
    assert result.executed_candidate_simulations == 2 * 8
    assert result.winner_margin[0, 1].item() == 1.0


def test_fixed_budget_a2_accepts_clear_winner() -> None:
    observations, mask, positions, candidates = fixture()
    result = select_fixed_budget(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        candidates=candidates,
        precut=FakePrecut(),
        device=torch.device("cpu"),
        selector="a2_fixed_budget",
        record_batch_size=1,
    )
    assert result.predictions.tolist() == [[128000, 3, 5]]
    assert result.normalized_winner[0, 1].item() >= 2.0


def test_fixed_budget_a2_abstains_and_stops_suffix() -> None:
    observations, mask, positions, candidates = fixture()
    result = select_fixed_budget(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        candidates=candidates,
        precut=FakePrecut(flat=True),
        device=torch.device("cpu"),
        selector="a2_fixed_budget",
        record_batch_size=1,
    )
    assert result.predictions.tolist() == [[128000, -1, -1]]
    assert result.routes[0, 1].item() == ROUTE_ABSTAIN
    assert result.routes[0, 2].item() == ROUTE_ABSTAINED_SUFFIX
    assert result.candidate_simulations == 8


def test_rank_and_error_diagnostics_use_scored_positions_only() -> None:
    observations, mask, _positions, candidates = fixture()
    del observations
    truth = torch.tensor([[128000, 3, 5]], dtype=torch.long)
    ranks = true_token_ranks(
        candidates=candidates,
        truth=truth,
        attention_mask=mask,
    )
    assert ranks.tolist() == [[0, 2, 4]]
    summary = rank_summary(ranks, mask, budgets=BUDGETS)
    assert summary["recall_at"]["8"] == 1.0
    predictions = torch.tensor([[128000, 3, 2]], dtype=torch.long)
    attribution = selector_error_attribution(
        predictions=predictions,
        truth=truth,
        attention_mask=mask,
        candidates=candidates,
    )
    assert attribution["proposal_exclusions"] == 0
    assert attribution["correct_with_true_token_in_candidates"] == 1
    assert attribution["selector_errors_or_abstentions_given_inclusion"] == 1
