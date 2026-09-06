from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from token_reconstruction.trr0006_visibility_decoder import (
    ATTENTION_SCORE_MODE,
    FULL_RECORD_METHOD,
    METHODS,
    PAST_ONLY_METHOD,
    POSITIONWISE_METHOD,
    build_visibility_decoder,
    build_visibility_mask,
    deterministic_top1,
    future_valid_count,
    load_visibility_state,
    save_visibility_state,
)


def _fixture(*, records: int = 2, positions: int = 6, hidden: int = 8, vocab: int = 13):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6001)
    activation = torch.randn(records, positions, hidden, generator=generator)
    mask = torch.ones(records, positions, dtype=torch.bool)
    mask[0, -1] = False
    table = F.normalize(torch.randn(vocab, hidden, generator=generator), dim=-1)
    direct = {
        "W": torch.eye(hidden, dtype=torch.float32) + 0.02 * torch.randn(hidden, hidden, generator=generator),
        "b": 0.03 * torch.randn(hidden, generator=generator),
        "s": torch.tensor(2.25, dtype=torch.float32),
    }
    return activation, mask, table, direct


def _models(*, hidden: int = 8, vocab: int = 13):
    _, _, _, direct = _fixture(hidden=hidden, vocab=vocab)
    return [
        build_visibility_decoder(
            method,
            hidden_size=hidden,
            vocabulary_size=vocab,
            context_width=4,
            qkv_seed=6002,
            direct_state=direct,
            direct_init_label="fixture_competent_affine",
        )
        for method in METHODS
    ]


def test_visibility_masks_have_exact_diagonal_past_and_full_keys() -> None:
    valid = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    diagonal = build_visibility_mask(POSITIONWISE_METHOD, valid)
    past = build_visibility_mask(PAST_ONLY_METHOD, valid)
    full = build_visibility_mask(FULL_RECORD_METHOD, valid)

    assert diagonal.shape == (2, 5, 5)
    assert torch.equal(diagonal[0], torch.diag(torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)))
    assert bool(past[0, 1, 0]) and not bool(past[0, 1, 2])
    assert bool(full[0, 0, 3])
    assert not bool(full[0, 0, 4])
    assert not bool(full[0, 4].any())
    assert torch.equal(future_valid_count(valid), torch.tensor([[3, 2, 1, 0, 0], [4, 3, 2, 1, 0]]))


def test_all_arms_share_competent_affine_initial_function() -> None:
    activation, mask, table, direct = _fixture()
    base = activation.float() @ direct["W"].T + direct["b"]
    expected = F.normalize(torch.where(mask.unsqueeze(-1), base, torch.zeros_like(base)), dim=-1)
    expected = torch.where(mask.unsqueeze(-1), expected, torch.zeros_like(expected))

    models = _models()
    parameter_names = tuple(name for name, _ in models[0].named_parameters())
    parameter_counts = [model.parameter_count for model in models]
    expected_parameter_count = 8 * 8 + 8 + 1 + 3 * (4 * 8 + 4) + (8 * 4 + 8)
    assert parameter_counts == [expected_parameter_count] * 3
    assert all(tuple(name for name, _ in model.named_parameters()) == parameter_names for model in models)
    for model in models:
        assert model.attention_score_mode == ATTENTION_SCORE_MODE
        assert model.direct_init_label == "fixture_competent_affine"
        with torch.inference_mode():
            actual = model.projected_hidden(activation, mask)
        # The output projection is zero initialized, so every arm starts at
        # exactly the same supplied affine function despite different masks.
        assert torch.equal(actual, expected)


def test_future_activation_is_visible_only_to_full_record() -> None:
    activation, mask, table, direct = _fixture(records=1, positions=6)
    mask[:] = True
    models = {
        method: build_visibility_decoder(
            method,
            hidden_size=8,
            vocabulary_size=13,
            context_width=4,
            qkv_seed=6002,
            direct_state=direct,
        )
        for method in METHODS
    }
    for model in models.values():
        with torch.no_grad():
            model.output.weight.normal_(0.0, 0.15)
            model.output.bias.normal_(0.0, 0.05)
    changed = activation.clone()
    changed[:, 4:, :] += 13.0
    with torch.inference_mode():
        before = {method: model.projected_hidden(activation, mask) for method, model in models.items()}
        after = {method: model.projected_hidden(changed, mask) for method, model in models.items()}
    assert torch.equal(before[POSITIONWISE_METHOD][:, 1:4], after[POSITIONWISE_METHOD][:, 1:4])
    assert torch.equal(before[PAST_ONLY_METHOD][:, 1:4], after[PAST_ONLY_METHOD][:, 1:4])
    assert not torch.equal(before[FULL_RECORD_METHOD][:, 1:4], after[FULL_RECORD_METHOD][:, 1:4])


def test_full_and_past_agree_at_last_valid_position() -> None:
    activation, mask, _, direct = _fixture(records=1, positions=6)
    mask[:] = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    full = build_visibility_decoder(
        FULL_RECORD_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
        qkv_seed=6002,
        direct_state=direct,
    )
    past = build_visibility_decoder(
        PAST_ONLY_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
        qkv_seed=6002,
        direct_state=direct,
    )
    past.load_state_dict(full.state_dict(), strict=True)
    with torch.no_grad():
        full.output.weight.normal_(0.0, 0.1)
        full.output.bias.normal_(0.0, 0.05)
        past.load_state_dict(full.state_dict(), strict=True)
    with torch.inference_mode():
        full_hidden = full.pre_normalized_hidden(activation, mask)
        past_hidden = past.pre_normalized_hidden(activation, mask)
    assert torch.equal(full_hidden[:, 3], past_hidden[:, 3])


def test_padding_is_inert_and_invalid_queries_are_zero() -> None:
    activation, mask, table, direct = _fixture(records=1, positions=6)
    mask[:] = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    model = build_visibility_decoder(
        FULL_RECORD_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
        qkv_seed=6002,
        direct_state=direct,
    )
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.1)
    changed = activation.clone()
    changed[:, 3:, :] = 10000.0
    with torch.inference_mode():
        before = model(activation, mask, table)
        after = model(changed, mask, table)
    assert torch.isfinite(after).all()
    assert torch.equal(before[:, :3], after[:, :3])
    assert torch.equal(after[:, 3:], torch.zeros_like(after[:, 3:]))


def test_records_are_isolated_and_permutation_separable() -> None:
    activation, mask, table, direct = _fixture(records=2, positions=5)
    mask[:] = True
    model = build_visibility_decoder(
        FULL_RECORD_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
        qkv_seed=6002,
        direct_state=direct,
    )
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.1)
    changed = activation.clone()
    changed[1] += 17.0
    with torch.inference_mode():
        before = model(activation, mask, table)
        after = model(changed, mask, table)
        swapped = model(activation.flip(0), mask.flip(0), table).flip(0)
    assert torch.equal(before[0], after[0])
    assert not torch.equal(before[1], after[1])
    assert torch.equal(before, swapped)


def test_positionwise_qk_gradients_are_zero_with_common_cosine_attention() -> None:
    activation, _, table, direct = _fixture(records=1, positions=5)
    mask = torch.ones(1, 5, dtype=torch.bool)
    model = build_visibility_decoder(
        POSITIONWISE_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
        qkv_seed=6002,
        direct_state=direct,
    )
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.1)
    selected = mask.clone()
    selected[:, 0] = False
    truth = torch.tensor([[128000, 2, 3, 4, 5]], dtype=torch.long)
    loss = F.cross_entropy(model.selected_logits(activation, mask, selected, table), truth[selected])
    loss.backward()
    assert torch.equal(model.query.weight.grad, torch.zeros_like(model.query.weight.grad))
    assert torch.equal(model.key.weight.grad, torch.zeros_like(model.key.weight.grad))
    assert float(model.value.weight.grad.norm()) > 0.0


def test_state_roundtrip_and_deterministic_lowest_id_ties(tmp_path: Path) -> None:
    _, _, _, direct = _fixture(records=1)
    model = build_visibility_decoder(
        FULL_RECORD_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
        qkv_seed=6002,
        direct_state=direct,
        direct_init_label="competent_public_affine",
    )
    path = tmp_path / "state.safetensors"
    descriptor = save_visibility_state(path, model, selected_step=0, metadata={"fit_seed": 6106})
    loaded = load_visibility_state(
        path,
        method_id=FULL_RECORD_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
        expected_sha256=descriptor["sha256"],
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    assert loaded.attention_score_mode == ATTENTION_SCORE_MODE
    ids, ties = deterministic_top1(torch.tensor([[2.0, 5.0, 5.0, 1.0], [7.0, 7.0, 6.0, 7.0]]))
    assert torch.equal(ids, torch.tensor([1, 0]))
    assert torch.equal(ties, torch.tensor([2, 3]))
