from __future__ import annotations

import pytest
import torch

from token_reconstruction.public_activation import (
    DEFAULT_BOS_TOKEN_ID,
    PAD_TOKEN_ID,
    PublicActivationError,
    PaddedTokenBatch,
    pad_public_token_sequences,
    validate_activation_tensor,
    validate_padded_token_batch,
)


def test_padding_preserves_current_token_alignment_and_nested_selector() -> None:
    batch = pad_public_token_sequences(
        [[DEFAULT_BOS_TOKEN_ID, 11, 12], [DEFAULT_BOS_TOKEN_ID, 21, 22, 23]],
        maximum_tokens=6,
        small_post_bos_positions=3,
    )
    assert batch.token_ids.tolist() == [
        [DEFAULT_BOS_TOKEN_ID, 11, 12, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
        [DEFAULT_BOS_TOKEN_ID, 21, 22, 23, PAD_TOKEN_ID, PAD_TOKEN_ID],
    ]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]]
    assert batch.position_ids.tolist() == [[0, 1, 2, 0, 0, 0], [0, 1, 2, 3, 0, 0]]
    assert batch.post_bos_selector_large.tolist() == [[0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0]]
    assert batch.post_bos_selector_small.tolist() == [[0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
    assert batch.post_bos_positions == 5
    assert batch.small_positions == 3


def test_validation_rejects_selector_on_bos_or_padding() -> None:
    batch = pad_public_token_sequences(
        [[DEFAULT_BOS_TOKEN_ID, 11, 12]], maximum_tokens=4, small_post_bos_positions=10
    )
    bad = PaddedTokenBatch(
        token_ids=batch.token_ids,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
        post_bos_selector_small=batch.post_bos_selector_small.clone(),
        post_bos_selector_large=batch.post_bos_selector_large.clone(),
        post_bos_ranges=batch.post_bos_ranges,
    )
    bad.post_bos_selector_large[0, 0] = 1
    with pytest.raises(PublicActivationError, match="exclude BOS"):
        validate_padded_token_batch(bad, maximum_tokens=4, small_post_bos_positions=10)


def test_activation_validation_requires_bfloat16_and_zero_padding() -> None:
    batch = pad_public_token_sequences(
        [[DEFAULT_BOS_TOKEN_ID, 11, 12]], maximum_tokens=4, small_post_bos_positions=10
    )
    good = torch.zeros((1, 4, 2048), dtype=torch.bfloat16)
    good[:, :3] = 1
    validate_activation_tensor(good, batch)
    with pytest.raises(PublicActivationError, match="BF16"):
        validate_activation_tensor(good.float(), batch)
    bad = good.clone()
    bad[0, 3, 0] = 1
    with pytest.raises(PublicActivationError, match="padded activation"):
        validate_activation_tensor(bad, batch)


def test_padding_rejects_missing_bos_and_overlong_sequences() -> None:
    with pytest.raises(PublicActivationError, match="BOS"):
        pad_public_token_sequences([[1, 2]], maximum_tokens=4)
    with pytest.raises(PublicActivationError, match="exceeds"):
        pad_public_token_sequences(
            [[DEFAULT_BOS_TOKEN_ID, 1, 2, 3]], maximum_tokens=3
        )


def test_capture_calls_resource_guard_after_each_fixed_batch() -> None:
    class FakePrefix:
        cut_depth = 4

        def __init__(self) -> None:
            self.calls = 0

        def eval(self):
            return self

        def forward_full(self, input_ids):
            self.calls += 1
            return torch.ones((*input_ids.shape, 2048), dtype=torch.bfloat16)

    from token_reconstruction.public_activation import capture_public_prefix

    batch = pad_public_token_sequences(
        [[DEFAULT_BOS_TOKEN_ID, 11, 12]] * 3,
        maximum_tokens=192,
        small_post_bos_positions=5000,
    )
    checks: list[int] = []
    activations = capture_public_prefix(
        FakePrefix(), batch, device=torch.device("cpu"), batch_size=2, resource_check=lambda: checks.append(1)
    )
    assert tuple(activations.shape) == (3, 192, 2048)
    assert len(checks) == 2
    assert activations[:, 3:].eq(0).all().item()



def test_padding_qualification_requires_bit_exact_future_pad_causality() -> None:
    from scripts.trr0004_prepare_public_activations import _qualify_public_prefix_padding

    class CausalFakePrefix:
        def __init__(self) -> None:
            self.calls: list[tuple[int, ...]] = []

        def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
            self.calls.append(tuple(input_ids.shape))
            # A causal cumulative representation must ignore all future IDs.
            values = input_ids.to(torch.bfloat16).cumsum(dim=1)
            return values.unsqueeze(-1).expand(*input_ids.shape, 2048)

    batch = pad_public_token_sequences(
        [[DEFAULT_BOS_TOKEN_ID, 11, 12], [DEFAULT_BOS_TOKEN_ID, 21, 22, 23]],
        maximum_tokens=192,
        small_post_bos_positions=5000,
    )
    prefix = CausalFakePrefix()
    result = _qualify_public_prefix_padding(
        prefix,
        batch,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert result["primary_geometry"]["same_batch_shape"] is True
    assert result["primary_geometry"]["active_output_bit_exact"] is True
    assert result["primary_geometry"]["maximum_absolute_difference"] == 0.0
    assert result["unpadded_batch1_diagnostic"]["status"] == "equivalent_bit_exact"
    assert result["batching_substitution_allowed"] is False
    assert prefix.calls[:2] == [(2, 192), (2, 192)]
