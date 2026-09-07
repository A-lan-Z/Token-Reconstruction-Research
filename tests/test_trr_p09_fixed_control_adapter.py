"""Synthetic contract tests for the TRR-P09 fixed-readout adapter.

These tests never load a public asset, model checkpoint, observation, or truth
file.  They exercise only the shared decoder/readout interface and metadata
validation.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from token_reconstruction.trr_p09_fixed_control_adapter import (
    AssetBinding,
    BankContract,
    FixedControlContractError,
    FixedPublicReadoutHook,
    TrainingContract,
    contract_digest,
    shared_decoder_rows,
)


_DIGEST = "a" * 64


def _asset(label: str) -> AssetBinding:
    return AssetBinding(label=label, path=f"/synthetic/{label}", bytes=11, sha256=_DIGEST)


def _bank() -> BankContract:
    return BankContract(
        fit_manifest=_asset("fit-manifest"),
        validation_manifest=_asset("validation-manifest"),
        embedding=_asset("embedding"),
        schedule=_asset("schedule"),
        fit_shape=(12, 192, 4),
        validation_shape=(6, 128, 4),
        fit_mask_shape=(12, 192),
        validation_mask_shape=(6, 128),
        fit_post_bos_rows=120,
        fit_supported_token_count=9,
        schedule_semantic_sha256=_DIGEST,
    )


def test_bank_contract_keeps_192_training_and_128_validation_independent() -> None:
    bank = _bank()
    bank.validate()
    assert bank.fit_shape[1] == 192
    assert bank.validation_shape[1] == 128
    assert len(contract_digest(bank)) == 64


def test_bank_contract_rejects_mask_or_digest_drift() -> None:
    bad = _bank().__class__(
        **{
            **_bank().__dict__,
            "validation_mask_shape": (6, 127),
        }
    )
    with pytest.raises(FixedControlContractError, match="validation mask"):
        bad.validate()
    with pytest.raises(FixedControlContractError, match="SHA-256"):
        AssetBinding("x", "/synthetic/x", 1, "bad").validate()


def test_training_contract_has_no_scientific_defaults() -> None:
    contract = TrainingContract(
        steps=12000,
        record_batch_size=8,
        position_budget=512,
        validation_every=500,
        selection_metric="earliest maximum public validation accuracy",
        seed=4005,
    )
    contract.validate()


def test_fixed_hook_is_identity_and_has_no_readout_parameters() -> None:
    decoder = nn.Linear(4, 7)
    hook = FixedPublicReadoutHook(method_id="synthetic_fixed", embedding_sha256=_DIGEST)
    base_logits = torch.randn(5, 7, requires_grad=True)
    query_rows = torch.randn(5, 4)
    embedding = torch.randn(7, 4)
    transformed = hook.score_rows(
        query_rows, torch.tensor(2.0), embedding, base_logits=base_logits
    )
    assert transformed is base_logits
    targets = torch.tensor([0, 1, 2, 3, 4])
    losses = hook.loss_terms(transformed, targets)
    assert set(losses) == {"cross_entropy", "total"}
    assert torch.equal(losses["total"], losses["cross_entropy"])
    groups = hook.optimizer_param_groups(decoder, base_learning_rate=2e-4)
    assert len(groups) == 1
    assert groups[0]["name"] == "decoder"
    assert groups[0]["lr"] == pytest.approx(2e-4)
    assert sum(parameter.numel() for parameter in groups[0]["params"]) == 35
    assert hook.metadata()["trainable_readout_parameters"] == 0


class _SyntheticDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = torch.tensor(1.0)

    def projected_hidden(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        assert activation.shape[:2] == valid_mask.shape
        return activation

    def logits_from_rows(
        self,
        projected: torch.Tensor,
        record_slots: torch.Tensor,
        position_slots: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        rows = projected[record_slots, position_slots]
        return rows @ embedding.transpose(0, 1)


def test_shared_decoder_rows_keeps_target_access_in_loss_only() -> None:
    decoder = _SyntheticDecoder()
    hook = FixedPublicReadoutHook(method_id="synthetic_fixed", embedding_sha256=_DIGEST)
    activation = torch.randn(2, 4, 3)
    valid_mask = torch.ones(2, 4, dtype=torch.bool)
    embedding = torch.randn(7, 3)
    record_slots = torch.tensor([0, 1, 0])
    position_slots = torch.tensor([1, 2, 3])
    targets = torch.tensor([0, 1, 2])
    base, logits, losses = shared_decoder_rows(
        decoder,
        hook,
        activation,
        valid_mask,
        record_slots,
        position_slots,
        embedding,
        targets,
    )
    assert torch.equal(base, logits)
    direct = hook.score_rows(
        base.new_empty((3, 3)).copy_(activation[record_slots, position_slots]),
        decoder.logit_scale,
        embedding,
    )
    assert torch.allclose(direct, base)
    assert losses["total"].ndim == 0
    assert base.shape == (3, 7)


def test_directional_fixture_depends_on_query_orientation_and_gets_gradients() -> None:
    class DirectionalFixture(nn.Module):
        method_id = "synthetic_directional"
        readout_mode = "token_direction"

        def __init__(self) -> None:
            super().__init__()
            self.delta = nn.Parameter(torch.zeros(2, 3))

        def score_rows(
            self,
            query_rows: torch.Tensor,
            logit_scale: torch.Tensor,
            embedding: torch.Tensor,
            *,
            base_logits: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del base_logits
            effective = embedding + self.delta
            return (query_rows @ effective.transpose(0, 1)) * logit_scale

        def loss_terms(
            self, logits: torch.Tensor, target_ids: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            total = torch.nn.functional.cross_entropy(logits, target_ids)
            return {"total": total}

        def optimizer_param_groups(self, decoder: nn.Module, *, base_learning_rate: float):
            del decoder
            return ({"params": [self.delta], "lr": base_learning_rate},)

        def metadata(self):
            return {"method_id": self.method_id}

    decoder = _SyntheticDecoder()
    hook = DirectionalFixture()
    with torch.no_grad():
        hook.delta[0, 0] = 0.5
    activation = torch.zeros(2, 2, 3)
    activation[0, 1, :2] = torch.tensor([1.0, 0.0])
    activation[1, 1, :2] = torch.tensor([0.0, 1.0])
    valid_mask = torch.ones(2, 2, dtype=torch.bool)
    embedding = torch.zeros(2, 3)
    base, logits, losses = shared_decoder_rows(
        decoder,
        hook,
        activation,
        valid_mask,
        torch.tensor([0, 1]),
        torch.tensor([1, 1]),
        embedding,
        torch.tensor([1, 1]),
        compute_base_logits=False,
    )
    assert base is None
    assert float(logits[0, 0].detach()) == pytest.approx(0.5)
    assert float(logits[1, 0].detach()) == pytest.approx(0.0)
    losses["total"].backward()
    assert hook.delta.grad is not None
    assert float(hook.delta.grad.abs().sum()) > 0.0
