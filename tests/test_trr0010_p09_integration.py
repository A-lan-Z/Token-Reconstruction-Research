from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from token_reconstruction.trr0007_positionwise import build_residual_mlp512
from trr0010_p09_integration import (
    P09IntegrationError,
    build_directional_hook,
    dispatch_shared_decoder_rows,
    optimizer_groups_for_p09,
    validate_p09_module,
)


def _fixture() -> tuple[object, object, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1010)
    base = build_residual_mlp512(
        hidden_size=8,
        vocabulary_size=17,
        context_width=4,
        bottleneck_size=3,
        seed=4005,
    )
    embedding = torch.nn.functional.normalize(torch.randn(17, 8), dim=-1)
    hook = build_directional_hook(
        base,
        torch.tensor([1, 4, 8, 13]),
        torch.tensor([1, 4, 25, 100]),
        embedding,
    )
    activation = torch.randn(2, 4, 8)
    valid_mask = torch.ones(2, 4, dtype=torch.bool)
    return base, hook, activation, valid_mask, embedding


def _synthetic_p09_protocol() -> SimpleNamespace:
    """Minimal executable stand-in for the committed A2 row dispatcher."""

    def shared_decoder_rows(
        decoder,
        hook,
        activation,
        valid_mask,
        record_slots,
        position_slots,
        embedding,
        target_ids,
        *,
        compute_base_logits=True,
    ):
        projected = decoder.projected_hidden(activation, valid_mask)
        base_logits = None
        if compute_base_logits:
            base_logits = decoder.logits_from_rows(
                projected, record_slots, position_slots, embedding
            )
        query_rows = projected[record_slots, position_slots]
        logits = hook.score_rows(
            query_rows,
            decoder.logit_scale,
            embedding,
            base_logits=base_logits,
        )
        losses = hook.loss_terms(logits, target_ids)
        return base_logits, logits, losses

    return SimpleNamespace(
        BankContract=object,
        TrainingContract=object,
        ReadoutHook=object,
        shared_decoder_rows=shared_decoder_rows,
        contract_digest=lambda contract: "a" * 64,
    )


def test_directional_hook_dispatch_matches_merged_primary_path() -> None:
    base, hook, activation, valid_mask, embedding = _fixture()
    protocol = _synthetic_p09_protocol()
    record_slots = torch.tensor([0, 0, 1], dtype=torch.long)
    position_slots = torch.tensor([1, 2, 1], dtype=torch.long)
    targets = torch.tensor([1, 4, 8], dtype=torch.long)

    _base_logits, logits, losses = dispatch_shared_decoder_rows(
        protocol,
        base,
        hook,
        activation,
        valid_mask,
        record_slots,
        position_slots,
        embedding,
        targets,
    )
    query_rows = hook.projected_hidden(activation, valid_mask)[record_slots, position_slots]
    expected = hook.score_rows(query_rows, base.logit_scale, embedding)
    assert torch.equal(logits, expected)
    assert torch.isfinite(losses["total"]).item()
    assert losses["total"].ndim == 0

    hook.zero_grad(set_to_none=True)
    losses["total"].backward()
    assert hook.delta_rows.grad is not None
    assert torch.isfinite(hook.delta_rows.grad).all().item()


def test_wrapper_binds_anchor_and_exposes_common_optimizer_groups() -> None:
    base, hook, _activation, _valid_mask, _embedding = _fixture()
    groups = optimizer_groups_for_p09(hook, base, base_learning_rate=2.0e-4)
    assert [group["name"] for group in groups] == ["decoder", "directional_delta"]
    assert groups[0]["lr"] == 2.0e-4
    assert groups[1]["lr"] == 1.0e-4
    assert hook.embedding_row_rms.shape == hook.support_ids.shape


def test_wrapper_rejects_incomplete_common_protocol() -> None:
    with pytest.raises(P09IntegrationError, match="missing required symbols"):
        validate_p09_module(SimpleNamespace(shared_decoder_rows=lambda *args, **kwargs: None))

