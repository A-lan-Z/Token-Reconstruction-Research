from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from token_reconstruction.causal_decoder_extension import (
    CAUSAL_ATTENTION_METHOD,
    FIXED_INPUT_NORMALIZATION,
    POSITIONWISE_MLP_METHOD,
    CausalDecoderExtensionError,
    FrozenAffineBase,
    build_causal_extension,
    extension_parameter_counts,
    fixed_input_normalization,
    validate_runtime_embeddings,
)


BOS = 128000


def _base(hidden_size: int = 8) -> FrozenAffineBase:
    torch.manual_seed(101)
    state = {
        "W": torch.randn(hidden_size, hidden_size),
        "b": torch.randn(hidden_size),
        "s": torch.tensor(0.25),
    }
    return FrozenAffineBase.from_state_dict(state)


def _inputs(*, batch: int = 2, sequence: int = 5, hidden_size: int = 8):
    torch.manual_seed(103)
    activation = torch.randn(batch, sequence, hidden_size)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    if batch > 1:
        valid[1, -1] = False
    table = F.normalize(torch.randn(11, hidden_size), dim=-1)
    return activation, valid, table


@pytest.mark.parametrize("method_id", [CAUSAL_ATTENTION_METHOD, POSITIONWISE_MLP_METHOD])
def test_zero_initialized_extension_matches_frozen_base_on_valid_positions(method_id: str) -> None:
    base = _base()
    extension = build_causal_extension(base, method_id)
    activation, valid, table = _inputs()
    with torch.inference_mode():
        expected = extension.base_logits(activation, valid, table)
        actual = extension(activation, valid, table)
    assert torch.equal(actual[valid], expected[valid])
    assert torch.equal(actual[~valid], torch.zeros_like(actual[~valid]))
    assert extension.trainable_parameters() == extension_parameter_counts(8)[method_id]
    assert all(not parameter.requires_grad for parameter in extension.base.parameters())


@pytest.mark.parametrize("method_id", [CAUSAL_ATTENTION_METHOD, POSITIONWISE_MLP_METHOD])
def test_future_activation_perturbation_cannot_change_earlier_valid_logits(method_id: str) -> None:
    base = _base()
    extension = build_causal_extension(base, method_id)
    activation, valid, table = _inputs(batch=1, sequence=6)
    with torch.no_grad():
        # Make the added path nonzero while preserving its public interface.
        for parameter in extension.added_path.parameters():
            parameter.add_(0.01)
        if method_id == CAUSAL_ATTENTION_METHOD:
            extension.added_path.output.weight.fill_(0.01)
        else:
            extension.added_path.up.weight.fill_(0.01)
    changed = activation.clone()
    torch.manual_seed(104)
    changed[:, 3:, :] += torch.randn_like(changed[:, 3:, :]) * 20.0
    with torch.inference_mode():
        before = extension(activation, valid, table)
        after = extension(changed, valid, table)
    assert torch.equal(before[:, :3], after[:, :3])
    assert not torch.equal(before[:, 3:], after[:, 3:])


@pytest.mark.parametrize("method_id", [CAUSAL_ATTENTION_METHOD, POSITIONWISE_MLP_METHOD])
def test_masked_padding_does_not_change_valid_prefix_and_all_padding_is_finite(method_id: str) -> None:
    base = _base()
    extension = build_causal_extension(base, method_id)
    activation, valid, table = _inputs(batch=1, sequence=4)
    valid[:] = True
    with torch.no_grad():
        for parameter in extension.added_path.parameters():
            parameter.add_(0.005)
        if method_id == CAUSAL_ATTENTION_METHOD:
            extension.added_path.output.weight.fill_(0.01)
        else:
            extension.added_path.up.weight.fill_(0.01)
    padded_activation = torch.cat([activation, torch.randn(1, 3, activation.shape[-1])], dim=1)
    padded_valid = torch.cat([valid, torch.zeros(1, 3, dtype=torch.bool)], dim=1)
    with torch.inference_mode():
        prefix_logits = extension(activation, valid, table)
        padded_logits = extension(padded_activation, padded_valid, table)
        empty_logits = extension(
            torch.randn(1, 4, activation.shape[-1]),
            torch.zeros(1, 4, dtype=torch.bool),
            table,
        )
    assert torch.equal(prefix_logits, padded_logits[:, :4])
    assert torch.isfinite(empty_logits).all()
    assert torch.equal(empty_logits, torch.zeros_like(empty_logits))




@pytest.mark.parametrize("method_id", [CAUSAL_ATTENTION_METHOD, POSITIONWISE_MLP_METHOD])
def test_selected_projection_avoids_full_sequence_vocabulary_logits(method_id: str) -> None:
    base = _base()
    extension = build_causal_extension(base, method_id)
    activation, valid, table = _inputs(batch=2, sequence=5)
    selected = torch.zeros_like(valid)
    selected[0, 0] = True
    selected[0, 3] = True
    selected[1, 1] = True
    with torch.inference_mode():
        hidden = extension.projected_hidden(activation, valid)
        selected_logits = extension.selected_logits(activation, valid, selected, table)
        full_logits = extension(activation, valid, table)
    assert tuple(hidden.shape) == tuple(activation.shape)
    assert tuple(selected_logits.shape) == (3, table.shape[0])
    assert torch.equal(selected_logits, full_logits[selected])
    invalid_selection = selected.clone()
    invalid_selection[1, -1] = True
    with pytest.raises(CausalDecoderExtensionError, match="must be valid"):
        extension.selected_logits(activation, valid, invalid_selection, table)


def test_parameter_matching_and_fixed_normalization_have_no_fitted_norm_parameters() -> None:
    counts = extension_parameter_counts(2048)
    assert counts[CAUSAL_ATTENTION_METHOD] == 1_051_008
    assert counts[POSITIONWISE_MLP_METHOD] == 1_050_880
    assert abs(counts[CAUSAL_ATTENTION_METHOD] - counts[POSITIONWISE_MLP_METHOD]) == 128
    attention = build_causal_extension(_base(), CAUSAL_ATTENTION_METHOD)
    mlp = build_causal_extension(_base(), POSITIONWISE_MLP_METHOD)
    assert attention.added_path.input_normalization == FIXED_INPUT_NORMALIZATION
    assert mlp.added_path.input_normalization == FIXED_INPUT_NORMALIZATION
    assert all("norm" not in name.lower() for name, _ in attention.added_path.named_parameters())
    assert all("norm" not in name.lower() for name, _ in mlp.added_path.named_parameters())
    x = torch.randn(2, 4, 8)
    expected = F.layer_norm(x, (8,), weight=None, bias=None, eps=1e-5)
    assert torch.equal(fixed_input_normalization(x, hidden_size=8), expected)


def test_input_contract_rejects_nonboolean_masks_and_bad_base_state() -> None:
    with pytest.raises(CausalDecoderExtensionError, match="exactly W, b, and s"):
        FrozenAffineBase({"W": torch.eye(4), "b": torch.zeros(4)})
    extension = build_causal_extension(_base(), POSITIONWISE_MLP_METHOD)
    activation, valid, table = _inputs()
    with pytest.raises(CausalDecoderExtensionError, match="valid mask must be boolean"):
        extension(activation, valid.to(torch.int64), table)
    with pytest.raises(CausalDecoderExtensionError, match="embedding table.*non-finite"):
        bad_table = table.clone()
        bad_table[0, 0] = float("nan")
        validate_runtime_embeddings(bad_table, hidden_size=8)
