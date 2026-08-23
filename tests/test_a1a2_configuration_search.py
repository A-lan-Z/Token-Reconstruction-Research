from __future__ import annotations

import torch

from token_reconstruction.a1a2_configuration_search import (
    CURRENT_THRESHOLD,
    PolicySpec,
    ROUTE_FAST_A1,
    current_anchor_spec,
    declared_policy_count,
    decode_policy,
    historical_anchor_spec,
    iter_policy_specs,
    resolve_policy,
    resolved_policy_from_dict,
    routing_signal,
    score_candidates,
)
from token_reconstruction.dual_benchmark import BOS_TOKEN_ID, INVALID_TOKEN_ID


class _FakeCache:
    def __init__(self) -> None:
        self.state: torch.Tensor | None = None
        self.length = 0

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        assert self.state is not None
        self.state = self.state[indices].clone()

    def batch_repeat_interleave(self, repeats: int) -> None:
        assert self.state is not None
        self.state = self.state.repeat_interleave(repeats, dim=0)


class _FakePrecut:
    width = 2048

    def new_cache(self) -> _FakeCache:
        return _FakeCache()

    def run_cached(
        self, input_ids: torch.Tensor, cache: _FakeCache, start_pos: int
    ) -> torch.Tensor:
        assert input_ids.ndim == 2 and input_ids.shape[1] == 1
        assert cache.length == start_pos
        if cache.state is None:
            cache.state = torch.zeros(
                (input_ids.shape[0], self.width),
                dtype=torch.float32,
                device=input_ids.device,
            )
        assert cache.state.shape[0] == input_ids.shape[0]
        token = torch.nn.functional.one_hot(
            input_ids[:, 0].remainder(self.width), num_classes=self.width
        ).float()
        hidden = cache.state + token
        cache.state = hidden.clone()
        cache.length += 1
        return hidden[:, None, :]


def _observations(tokens: torch.Tensor) -> torch.Tensor:
    state = torch.zeros((tokens.shape[0], _FakePrecut.width), dtype=torch.float32)
    values = []
    for position in range(tokens.shape[1]):
        state = state + torch.nn.functional.one_hot(
            tokens[:, position].remainder(_FakePrecut.width),
            num_classes=_FakePrecut.width,
        ).float()
        values.append(state.clone())
    return torch.stack(values, dim=1)


def test_declared_policy_enumeration_and_anchors() -> None:
    current = current_anchor_spec()
    historical = historical_anchor_spec()
    count = current_count = historical_count = 0
    for spec in iter_policy_specs():
        count += 1
        current_count += int(spec == current)
        historical_count += int(spec == historical)
    assert count == declared_policy_count() == 512136
    assert current_count == historical_count == 1


def test_current_anchor_resolves_exact_threshold_and_comparator() -> None:
    policy = resolve_policy(current_anchor_spec(), {})
    assert policy.threshold_at(32) == CURRENT_THRESHOLD
    assert policy.threshold_at(64) == CURRENT_THRESHOLD
    assert policy.spec.gate_comparator == "gt"


def test_resolved_policy_strict_json_round_trip_handles_infinity() -> None:
    spec = PolicySpec(
        kind="adaptive",
        score_rule="direct_cosine",
        schedule=(2, 4),
        fast_path_id="off",
        fast_path_threshold=None,
        routing_signal="raw_margin",
        gate_mode="never_accept",
        terminal_action="commit_last_winner",
    )
    policy = resolve_policy(spec, {})
    serialized = policy.serialized()
    assert serialized["numeric_thresholds"] == {"2": "Infinity", "4": "Infinity"}
    assert resolved_policy_from_dict(serialized) == policy


def test_resolved_fixed_policy_round_trip_needs_no_threshold() -> None:
    spec = PolicySpec(
        kind="fixed",
        score_rule="direct_cosine",
        schedule=(512,),
        fast_path_id="off",
        fast_path_threshold=None,
        routing_signal=None,
        gate_mode=None,
        terminal_action="commit_last_winner",
    )
    policy = resolve_policy(spec, {})
    serialized = policy.serialized()
    assert serialized["numeric_thresholds"] == {}
    assert resolved_policy_from_dict(serialized) == policy


def test_score_rules_and_normalized_margin_invariances() -> None:
    hidden = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]]
    )
    target = torch.tensor([[1.0, 0.0, 0.0]])
    direct = score_candidates(hidden, target, "direct_cosine")
    assert direct.argmax(dim=1).item() == 0

    shift = torch.tensor([[[3.0, -2.0, 4.0]]])
    centered_a = score_candidates(hidden, target, "group_centered_cosine")
    centered_b = score_candidates(
        hidden + shift,
        target + shift[:, 0],
        "group_centered_cosine",
    )
    assert torch.allclose(centered_a, centered_b, atol=1e-6, rtol=0.0)

    scores = torch.tensor([[0.2, 0.1, -0.3, -0.5]])
    base = routing_signal(scores, "rms_normalized_margin")
    transformed = routing_signal(scores * 7.0 + 11.0, "rms_normalized_margin")
    assert torch.allclose(base, transformed, atol=1e-5, rtol=0.0)


def test_fixed_policy_uses_only_reconstructed_prefix() -> None:
    tokens = torch.tensor(
        [
            [BOS_TOKEN_ID, 9, 10, 11],
            [BOS_TOKEN_ID, 12, 13, 14],
        ],
        dtype=torch.long,
    )
    observations = _observations(tokens)
    mask = torch.ones(tokens.shape, dtype=torch.long)
    positions = torch.arange(tokens.shape[1]).view(1, -1).expand_as(tokens)
    candidates = torch.full((*tokens.shape, 2), INVALID_TOKEN_ID, dtype=torch.long)
    candidates[:, 1:, 0] = (tokens[:, 1:] + 3) % 100
    candidates[:, 1:, 1] = tokens[:, 1:]
    confidence = torch.full(tokens.shape, 0.5, dtype=torch.float32)
    confidence[:, 0] = float("nan")
    spec = PolicySpec(
        kind="fixed",
        score_rule="direct_cosine",
        schedule=(2,),
        fast_path_id="off",
        fast_path_threshold=None,
        routing_signal=None,
        gate_mode=None,
        terminal_action="commit_last_winner",
    )
    result = decode_policy(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        candidates=candidates,
        a1_confidence=confidence,
        precut=_FakePrecut(),
        device=torch.device("cpu"),
        policy=resolve_policy(spec, {}),
        record_batch_size=2,
    )
    assert torch.equal(result.predictions, tokens)
    assert result.candidate_simulations == 2 * 3 * 2
    assert result.executed_candidate_simulations == result.candidate_simulations


def test_immediate_a1_route_changes_decision_without_simulation() -> None:
    tokens = torch.tensor([[BOS_TOKEN_ID, 9, 10]], dtype=torch.long)
    observations = _observations(tokens)
    mask = torch.ones(tokens.shape, dtype=torch.long)
    positions = torch.arange(tokens.shape[1]).view(1, -1)
    candidates = torch.full((*tokens.shape, 2), INVALID_TOKEN_ID, dtype=torch.long)
    candidates[:, 1:, 0] = 15
    candidates[:, 1:, 1] = tokens[:, 1:]
    confidence = torch.ones(tokens.shape, dtype=torch.float32)
    confidence[:, 0] = float("nan")
    spec = PolicySpec(
        kind="fixed",
        score_rule="direct_cosine",
        schedule=(2,),
        fast_path_id="historical_a1_ge_0.999",
        fast_path_threshold=0.999,
        routing_signal=None,
        gate_mode=None,
        terminal_action="commit_last_winner",
    )
    result = decode_policy(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        candidates=candidates,
        a1_confidence=confidence,
        precut=_FakePrecut(),
        device=torch.device("cpu"),
        policy=resolve_policy(spec, {}),
    )
    assert result.predictions[0, 1:].eq(15).all().item()
    assert result.routes[0, 1:].eq(ROUTE_FAST_A1).all().item()
    assert result.candidate_simulations == 0
