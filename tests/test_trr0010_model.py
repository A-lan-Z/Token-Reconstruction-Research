from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from token_reconstruction.trr0005_joint_decoder import file_sha256
from token_reconstruction.trr0007_positionwise import build_residual_mlp512
from trr0010_model import (
    TRR0010ModelError,
    build_directional_from_base,
    directional_memory_estimate,
    export_effective_embedding,
    load_directional_state,
    save_directional_state,
    support_digest,
)


def _fixture():
    torch.manual_seed(10010)
    base = build_residual_mlp512(
        hidden_size=8,
        vocabulary_size=17,
        context_width=4,
        bottleneck_size=3,
        seed=4005,
    )
    ids = torch.tensor([1, 4, 8, 13], dtype=torch.long)
    counts = torch.tensor([1, 4, 25, 100], dtype=torch.long)
    directional = build_directional_from_base(base, ids, counts)
    activation = torch.randn(2, 4, 8)
    valid = torch.tensor(
        [[True, True, True, True], [True, True, True, False]],
        dtype=torch.bool,
    )
    embedding = torch.nn.functional.normalize(torch.randn(17, 8), dim=-1)
    directional.bind_embedding_statistics(embedding)
    return base, directional, activation, valid, embedding, ids, counts


def test_zero_direction_is_bit_exact_for_full_and_selected_logits() -> None:
    base, directional, activation, valid, embedding, _ids, _counts = _fixture()
    base_logits = base(activation, valid, embedding)
    directional_logits = directional(activation, valid, embedding)
    assert torch.equal(base_logits, directional_logits)
    assert torch.equal(base_logits.argmax(dim=-1), directional_logits.argmax(dim=-1))

    selected = torch.tensor(
        [[False, True, False, True], [False, True, True, False]],
        dtype=torch.bool,
    )
    base_selected = base.selected_logits(activation, valid, selected, embedding)
    directional_selected = directional.selected_logits(activation, valid, selected, embedding)
    assert torch.equal(base_selected, directional_selected)


def test_directional_update_changes_only_supported_columns_and_has_expected_score() -> None:
    base, directional, activation, valid, embedding, ids, _counts = _fixture()
    with torch.no_grad():
        directional.delta_rows.zero_()
        directional.delta_rows[1, 0] = 0.5
        directional.delta_rows[1, 3] = -0.25

    base_logits = base(activation, valid, embedding)
    adapted_logits = directional(activation, valid, embedding)
    hidden = directional.projected_hidden(activation, valid)
    record_slots = torch.tensor([0, 0, 1], dtype=torch.long)
    position_slots = torch.tensor([1, 2, 1], dtype=torch.long)
    base_rows = base.logits_from_rows(hidden, record_slots, position_slots, embedding)
    adapted_rows = directional.logits_from_rows(
        hidden,
        record_slots,
        position_slots,
        embedding,
        base_logits=base_rows,
    )
    direct_rows = directional.score_rows(
        hidden[record_slots, position_slots],
        directional.logit_scale,
        embedding,
        base_logits=base_rows,
    )
    assert torch.equal(adapted_rows, direct_rows)
    assert torch.equal(adapted_rows, directional.selected_logits(
        activation,
        valid,
        torch.tensor([[False, True, True, False], [False, True, False, False]]),
        embedding,
    ))
    absent = torch.tensor([0, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16])
    assert torch.equal(base_logits[..., absent], adapted_logits[..., absent])
    changed_column = int(ids[1])
    assert not torch.equal(base_logits[..., changed_column], adapted_logits[..., changed_column])
    assert torch.equal(base_logits[..., int(ids[0])], adapted_logits[..., int(ids[0])])

    expected = hidden[..., 0] * 0.5 + hidden[..., 3] * -0.25
    expected = expected * directional.logit_scale
    observed = adapted_logits[..., changed_column] - base_logits[..., changed_column]
    assert torch.allclose(observed, expected, rtol=0.0, atol=4e-6)


def test_current_position_output_is_invariant_to_other_positions() -> None:
    _base, directional, activation, valid, embedding, _ids, _counts = _fixture()
    changed = activation.clone()
    changed[0, 3] += 20.0
    changed[1, 0] += 20.0
    selected = torch.tensor(
        [[False, True, False, False], [False, True, False, False]],
        dtype=torch.bool,
    )
    first = directional.selected_logits(activation, valid, selected, embedding)
    second = directional.selected_logits(changed, valid, selected, embedding)
    assert torch.equal(first, second)


def test_primary_score_rows_matches_exported_effective_readout_and_gradients() -> None:
    base, directional, activation, valid, embedding, _ids, _counts = _fixture()
    with torch.no_grad():
        directional.delta_rows.normal_(mean=0.0, std=0.05)
    hidden = directional.projected_hidden(activation, valid)
    embedding_before = embedding.clone()
    records = torch.tensor([0, 0, 1], dtype=torch.long)
    positions = torch.tensor([1, 2, 1], dtype=torch.long)
    query_rows = hidden[records, positions]
    targets = torch.tensor([1, 4, 8], dtype=torch.long)
    primary = directional.score_rows(query_rows, directional.logit_scale, embedding)
    effective = directional.materialize_effective_embedding(embedding)
    deployed = directional.merged_score_rows(query_rows, directional.logit_scale, effective)
    assert torch.equal(embedding, embedding_before)
    assert torch.allclose(primary, deployed, rtol=0.0, atol=0.0)
    assert torch.equal(primary.argmax(dim=-1), deployed.argmax(dim=-1))
    directional.zero_grad(set_to_none=True)
    total = torch.nn.functional.cross_entropy(primary, targets) + directional.readout_penalty()
    total.backward()
    assert directional.delta_rows.grad is not None
    assert torch.isfinite(directional.delta_rows.grad).all().item()
    assert float(directional.delta_rows.grad.abs().sum()) > 0.0
    base_gradients = [parameter.grad for parameter in base.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all().item() for gradient in base_gradients)


def test_sparse_forward_is_diagnostic_only_and_matches_primary_with_tolerance() -> None:
    _base, directional, activation, valid, embedding, _ids, _counts = _fixture()
    with torch.no_grad():
        directional.delta_rows.normal_(mean=0.0, std=0.05)
    primary = directional(activation, valid, embedding)
    sparse = directional.sparse_forward(activation, valid, embedding)
    effective = directional.materialize_effective_embedding(embedding)
    deployed = directional.merged_forward(activation, valid, effective)
    assert torch.allclose(primary, deployed, rtol=0.0, atol=0.0)
    assert torch.allclose(sparse, deployed, rtol=2e-6, atol=2e-6)
    assert torch.equal(effective[[0, 2, 3]], embedding[[0, 2, 3]])


def test_penalty_and_gradients_are_finite_and_frequency_weighted() -> None:
    _base, directional, _activation, _valid, embedding, _ids, counts = _fixture()
    with torch.no_grad():
        directional.delta_rows.fill_(0.1)
    penalty = directional.readout_penalty()
    expected_weights = 1.0 + 4.0 / torch.sqrt(counts.float())
    expected = 1.0e-4 * (
        expected_weights
        * directional.delta_rows.square().sum(dim=1)
        / (directional.hidden_size * directional.embedding_row_rms.square() + directional.embedding_epsilon)
    ).mean()
    assert torch.allclose(penalty, expected)
    directional.zero_grad(set_to_none=True)
    (penalty + directional(  # exercise CE-like gradient through the readout
        torch.randn(2, 4, 8),
        torch.ones(2, 4, dtype=torch.bool),
        embedding,
    ).square().mean()).backward()
    assert directional.delta_rows.grad is not None
    assert torch.isfinite(directional.delta_rows.grad).all().item()


def test_save_load_and_full_export_roundtrip(tmp_path: Path) -> None:
    base, directional, activation, valid, embedding, ids, counts = _fixture()
    with torch.no_grad():
        directional.delta_rows.normal_(mean=0.0, std=0.05)
    base_state_path = tmp_path / "base.safetensors"
    # The directional state only binds the immutable hash in metadata; a
    # synthetic value is sufficient for this serialization test.
    base_state_path.write_bytes(b"synthetic-base-state")
    state_path = tmp_path / "directional.safetensors"
    saved = save_directional_state(
        state_path,
        directional,
        selected_step=400,
        base_state={"sha256": file_sha256(base_state_path)},
        fit_manifest={"sha256": "a" * 64},
    )
    loaded = load_directional_state(
        state_path,
        hidden_size=8,
        vocabulary_size=17,
        context_width=4,
        bottleneck_size=3,
    )
    loaded.bind_embedding_statistics(embedding)
    assert saved["metadata"]["support_digest"] == support_digest(ids, counts)
    assert all(torch.equal(loaded.state_dict()[name], directional.state_dict()[name]) for name in directional.state_dict())
    assert torch.equal(loaded(activation, valid, embedding), directional(activation, valid, embedding))

    export_path = tmp_path / "effective_embeddings.safetensors"
    exported = export_effective_embedding(export_path, directional, embedding)
    assert exported["tensor_shape"] == [17, 8]
    assert export_path.exists()
    reloaded_effective = load_file(str(export_path))["embeddings"]
    direct_effective = directional.materialize_effective_embedding(embedding).detach().cpu()
    assert torch.equal(reloaded_effective, direct_effective)
    primary = directional(activation, valid, embedding)
    deployed = directional.merged_forward(activation, valid, reloaded_effective)
    assert torch.equal(primary, deployed)
    assert torch.equal(primary.argmax(dim=-1), deployed.argmax(dim=-1))


def test_shared_optimizer_and_loss_hooks_are_explicit() -> None:
    _base, directional, _activation, _valid, embedding, _ids, _counts = _fixture()
    groups = directional.optimizer_groups(2.0e-4, {"directional_lr": 5.0e-4})
    assert [group["name"] for group in groups] == ["decoder_base", "directional_delta"]
    assert groups[0]["lr"] == 2.0e-4
    assert groups[1]["lr"] == 5.0e-4
    protocol_groups = directional.optimizer_param_groups(
        directional.base, base_learning_rate=2.0e-4
    )
    assert [group["name"] for group in protocol_groups] == ["decoder", "directional_delta"]
    assert protocol_groups[0]["lr"] == 2.0e-4
    assert protocol_groups[1]["lr"] == 1.0e-4
    logits = directional(
        torch.randn(2, 4, 8),
        torch.ones(2, 4, dtype=torch.bool),
        embedding,
    )[:, 1:]
    targets = torch.tensor([1, 4, 8, 13, 1, 4], dtype=torch.long)
    terms = directional.loss_terms(logits.reshape(-1, 17), targets)
    assert set(terms) == {"cross_entropy", "readout_penalty", "total"}
    assert torch.isfinite(terms["total"]).item()
    assert directional.state_metadata()["readout_mode"] == "supported_token_directional"


def test_support_and_memory_contracts_are_strict() -> None:
    base, _directional, _activation, _valid, _embedding, _ids, _counts = _fixture()
    with pytest.raises(TRR0010ModelError):
        build_directional_from_base(base, [[1, 4]], [[1, 1]])
    with pytest.raises(TRR0010ModelError):
        build_directional_from_base(base, [4, 1], [1, 1])
    with pytest.raises(TRR0010ModelError):
        build_directional_from_base(base, [1, 1], [1, 1])
    estimate = directional_memory_estimate(
        4,
        hidden_size=8,
        vocabulary_size=17,
        position_budget=5,
        base_peak_reserved_bytes=100,
    )
    assert estimate["parameter_count"] == 32
    assert estimate["parameter_gradient_adam_bytes"] == 32 * 4 * 4
    assert estimate["diagnostic_sparse_workspace_bytes_fp32"] == 5 * 4 * 4
    assert estimate["materialized_effective_embedding_bytes_fp32"] == 17 * 8 * 4
    assert estimate["materialized_effective_embedding_gradient_bytes_fp32"] == 17 * 8 * 4
    assert estimate["primary_support_row_gather_bytes_fp32"] == 4 * 8 * 4
    assert estimate["primary_support_row_add_bytes_fp32"] == 4 * 8 * 4
    assert estimate["estimated_peak_sparse_diagnostic_bytes"] == 100 + 32 * 4 * 4 + 5 * 4 * 4
    assert estimate["estimated_peak_merged_primary_bytes"] == 100 + 32 * 4 * 4 + 2 * 4 * 8 * 4 + 2 * 17 * 8 * 4
    assert estimate["estimated_peak_plus_directional_bytes"] == estimate["estimated_peak_merged_primary_bytes"]
