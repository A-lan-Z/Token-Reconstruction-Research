from __future__ import annotations

from pathlib import Path

import pytest
import torch

from token_reconstruction.trr0007_positionwise import build_residual_mlp512, save_positionwise_state
from trr0009_model import (
    TRR0009ModelError,
    build_adaptable_from_base,
    build_support_from_counts,
    load_supported_readout_state,
    save_supported_readout_state,
    support_digest,
    zero_correction_equivalence,
)
from token_reconstruction.trr0005_joint_decoder import file_sha256


def _fixture():
    torch.manual_seed(9009)
    base = build_residual_mlp512(hidden_size=16, vocabulary_size=32, context_width=4, bottleneck_size=8, seed=4005)
    adaptable = build_adaptable_from_base(base, [1, 3, 9], [1, 4, 20])
    activation = torch.randn(2, 5, 16)
    valid = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])
    embedding = torch.nn.functional.normalize(torch.randn(32, 16), dim=-1)
    return base, adaptable, activation, valid, embedding


def test_zero_correction_is_bit_exact_for_rows_and_full_logits() -> None:
    base, adaptable, activation, valid, embedding = _fixture()
    receipt = zero_correction_equivalence(base, adaptable, activation, valid, embedding, max_rows=32)
    assert receipt["projected_hidden_exact"]
    assert receipt["selected_logits_exact"]
    assert receipt["full_logits_exact"]
    assert receipt["max_full_logits_abs_delta"] == 0.0


def test_supported_correction_changes_only_supported_columns() -> None:
    base, adaptable, activation, valid, embedding = _fixture()
    with torch.no_grad():
        adaptable.raw_gain.fill_(0.4)
        adaptable.raw_bias.fill_(-0.3)
    base_logits = base(activation, valid, embedding)
    adapted_logits = adaptable(activation, valid, embedding)
    absent = [0, 2, 4, 5, 6, 7, 8, 10, 31]
    assert torch.equal(base_logits[..., absent], adapted_logits[..., absent])
    assert not torch.equal(base_logits[..., [1]], adapted_logits[..., [1]])
    assert not torch.equal(base_logits[..., [3]], adapted_logits[..., [3]])
    assert not torch.equal(base_logits[..., [9]], adapted_logits[..., [9]])


def test_current_position_output_is_invariant_to_other_positions() -> None:
    _base, adaptable, activation, valid, embedding = _fixture()
    changed = activation.clone()
    changed[0, 4] += 20.0
    changed[1, 0] += 20.0
    row_mask = torch.tensor([[False, True, False, False, False], [False, True, False, False, False]])
    first = adaptable.selected_logits(activation, valid, row_mask, embedding)
    second = adaptable.selected_logits(changed, valid, row_mask, embedding)
    assert torch.equal(first, second)


def test_frequency_anchor_penalty_matches_contract() -> None:
    _base, adaptable, _activation, _valid, _embedding = _fixture()
    with torch.no_grad():
        adaptable.raw_gain.copy_(torch.tensor([1.0, 2.0, 3.0]))
        adaptable.raw_bias.copy_(torch.tensor([-1.0, -2.0, -3.0]))
    expected_weights = 1.0 + 4.0 / torch.sqrt(torch.tensor([1.0, 4.0, 20.0]))
    expected = 1.0e-4 * (expected_weights * (adaptable.raw_gain.square() + adaptable.raw_bias.square())).mean()
    assert torch.allclose(adaptable.readout_penalty(), expected)


def test_support_validation_is_strict() -> None:
    with pytest.raises(TRR0009ModelError):
        build_support_from_counts(torch.tensor([1, 0, 2]), vocabulary_size=2)
    base, _adaptable, _activation, _valid, _embedding = _fixture()
    with pytest.raises(TRR0009ModelError):
        build_adaptable_from_base(base, [3, 1], [1, 1])
    with pytest.raises(TRR0009ModelError):
        build_adaptable_from_base(base, [1, 1], [1, 1])


def test_materializer_binds_full_dictionary_and_bias() -> None:
    _base, adaptable, _activation, _valid, embedding = _fixture()
    with torch.no_grad():
        adaptable.raw_gain.copy_(torch.tensor([0.1, 0.2, 0.3]))
        adaptable.raw_bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    effective, bias = adaptable.materialize_effective_readout(embedding)
    assert tuple(effective.shape) == (32, 16)
    assert tuple(bias.shape) == (32,)
    assert torch.equal(effective[[0, 2, 4]], embedding[[0, 2, 4]])
    ids = adaptable.support_ids
    assert torch.allclose(effective[ids], embedding[ids] * adaptable.effective_gain().unsqueeze(-1))
    assert torch.allclose(bias[ids], adaptable.effective_bias())


def test_save_load_roundtrip_preserves_support_binding(tmp_path: Path) -> None:
    base, adaptable, _activation, _valid, _embedding = _fixture()
    base_path = tmp_path / "base.safetensors"
    base_record = save_positionwise_state(
        base_path,
        base,
        method_id="trr0007_residual_mlp512",
        selected_step=2100,
        initialization="synthetic",
        distribution="synthetic",
        bottleneck_size=8,
    )
    state_path = tmp_path / "adaptable.safetensors"
    record = save_supported_readout_state(
        state_path,
        adaptable,
        selected_step=100,
        starting_state={"sha256": file_sha256(base_path)},
        fit_manifest={"sha256": "a" * 64},
    )
    loaded = load_supported_readout_state(
        state_path,
        base_state_path=base_path,
        support_ids=adaptable.support_ids,
        support_counts=adaptable.support_counts,
        hidden_size=16,
        vocabulary_size=32,
        context_width=4,
        bottleneck_size=8,
    )
    assert record["metadata"]["support_digest"] == support_digest(adaptable.support_ids, adaptable.support_counts)
    assert torch.equal(loaded.support_ids, adaptable.support_ids)
    assert torch.equal(loaded.support_counts, adaptable.support_counts)
    assert all(torch.equal(loaded.state_dict()[name], adaptable.state_dict()[name]) for name in adaptable.state_dict())
