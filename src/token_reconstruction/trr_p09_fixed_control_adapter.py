"""Shared fixed-readout hook and bank contract for TRR-P09.

This module contains contract plumbing only.  The common runner owns the
public decoder, sampler, validation, checkpoint grid, and resource guards.  A
readout hook receives the decoder's normalized query rows, public embedding,
and inherited logit scale. It may score a method-specific full-vocabulary
readout and add a method-specific loss or optimizer group. The fixed hook
reuses the decoder's base logits exactly and exposes only decoder parameters.

No dataset, model checkpoint, target observation, or truth asset is loaded by
this module.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FixedControlContractError(ValueError):
    """Raised when a shared P09 control contract is malformed."""


@dataclass(frozen=True)
class AssetBinding:
    """Opaque file identity used by a bank contract.

    The contract records identity; the outer runner performs the actual file
    existence/hash check before loading.  Keeping that operation outside this
    module makes synthetic tests independent of private/public assets.
    """

    label: str
    path: str
    bytes: int
    sha256: str

    def validate(self) -> None:
        if not self.label or not self.path:
            raise FixedControlContractError("asset label and path are required")
        if self.bytes <= 0:
            raise FixedControlContractError("asset byte count must be positive")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise FixedControlContractError(f"{self.label} SHA-256 is malformed")


@dataclass(frozen=True)
class BankContract:
    """Public data/schedule bindings shared by fixed and directional methods.

    Fit and validation sequence widths are intentionally independent.  P09
    fitting preserves the published 192-position training bank while a future
    natural validation contract may use an H128 observation; only the hidden
    width must agree between them.
    """

    fit_manifest: AssetBinding
    validation_manifest: AssetBinding
    embedding: AssetBinding
    schedule: AssetBinding
    fit_shape: tuple[int, int, int]
    validation_shape: tuple[int, int, int]
    fit_mask_shape: tuple[int, int]
    validation_mask_shape: tuple[int, int]
    fit_post_bos_rows: int
    fit_supported_token_count: int
    schedule_semantic_sha256: str

    def validate(self) -> None:
        for binding in (
            self.fit_manifest,
            self.validation_manifest,
            self.embedding,
            self.schedule,
        ):
            binding.validate()
        if len(self.fit_shape) != 3 or len(self.validation_shape) != 3:
            raise FixedControlContractError("activation shapes must be rank three")
        if len(self.fit_mask_shape) != 2 or len(self.validation_mask_shape) != 2:
            raise FixedControlContractError("mask shapes must be rank two")
        if self.fit_shape[:2] != self.fit_mask_shape:
            raise FixedControlContractError("fit mask does not match fit activation rows")
        if self.validation_shape[:2] != self.validation_mask_shape:
            raise FixedControlContractError("validation mask does not match validation rows")
        if self.fit_shape[2] <= 0 or self.validation_shape[2] <= 0:
            raise FixedControlContractError("hidden width must be positive")
        if self.fit_shape[2] != self.validation_shape[2]:
            raise FixedControlContractError("fit and validation hidden widths differ")
        if self.fit_shape[0] <= 0 or self.validation_shape[0] <= 0:
            raise FixedControlContractError("fit and validation record counts must be positive")
        if self.fit_shape[1] <= 1 or self.validation_shape[1] <= 1:
            raise FixedControlContractError("each sequence must contain BOS and a scored position")
        if self.fit_post_bos_rows <= 0:
            raise FixedControlContractError("fit post-BOS row count must be positive")
        if self.fit_post_bos_rows > self.fit_shape[0] * (self.fit_shape[1] - 1):
            raise FixedControlContractError("fit post-BOS row count exceeds mask capacity")
        if self.fit_supported_token_count < 0:
            raise FixedControlContractError("supported token count cannot be negative")
        if _SHA256_RE.fullmatch(self.schedule_semantic_sha256) is None:
            raise FixedControlContractError("schedule semantic SHA-256 is malformed")


@dataclass(frozen=True)
class TrainingContract:
    """Shared optimizer/checkpoint dimensions without scientific defaults."""

    steps: int
    record_batch_size: int
    position_budget: int
    validation_every: int
    selection_metric: str
    seed: int

    def validate(self) -> None:
        if self.steps <= 0 or self.record_batch_size <= 0 or self.position_budget <= 0:
            raise FixedControlContractError("steps, batch size, and position budget must be positive")
        if self.validation_every <= 0:
            raise FixedControlContractError("validation interval must be positive")
        if not self.selection_metric:
            raise FixedControlContractError("selection metric is required")


class ReadoutHook(Protocol):
    """Readout surface consumed by a shared decoder/sampler runner."""

    method_id: str
    readout_mode: str

    def score_rows(
        self,
        query_rows: torch.Tensor,
        logit_scale: torch.Tensor,
        embedding: torch.Tensor,
        *,
        base_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return full-vocabulary logits without access to target labels."""

    def loss_terms(
        self, logits: torch.Tensor, target_ids: torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        """Return named differentiable losses, including ``total``."""

    def optimizer_param_groups(
        self, decoder: nn.Module, *, base_learning_rate: float
    ) -> Sequence[Mapping[str, Any]]:
        """Return optimizer groups; the shared runner owns optimizer creation."""

    def metadata(self) -> Mapping[str, Any]:
        """Return serializable method provenance."""


@dataclass(frozen=True)
class FixedPublicReadoutHook:
    """Identity hook for the public fixed-readout continuation control."""

    method_id: str
    embedding_sha256: str
    readout_mode: str = "fixed_public_E"

    def __post_init__(self) -> None:
        if not self.method_id:
            raise FixedControlContractError("method ID is required")
        if _SHA256_RE.fullmatch(self.embedding_sha256) is None:
            raise FixedControlContractError("embedding SHA-256 is malformed")

    def score_rows(
        self,
        query_rows: torch.Tensor,
        logit_scale: torch.Tensor,
        embedding: torch.Tensor,
        *,
        base_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_query_geometry(query_rows, embedding, logit_scale)
        if base_logits is not None:
            if tuple(base_logits.shape) != (query_rows.shape[0], embedding.shape[0]):
                raise FixedControlContractError("base logits geometry differs")
            # Reusing the inherited decoder result preserves its exact
            # normalization, dtype conversion, and exp(s) scale path.
            return base_logits
        return _public_logits(query_rows, embedding, logit_scale)

    def loss_terms(
        self, logits: torch.Tensor, target_ids: torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        if logits.ndim != 2 or target_ids.ndim != 1 or logits.shape[0] != target_ids.shape[0]:
            raise FixedControlContractError("logit and target row geometry differs")
        if not logits.is_floating_point():
            raise FixedControlContractError("logits must be floating point")
        total = F.cross_entropy(logits, target_ids)
        return {"cross_entropy": total, "total": total}

    def optimizer_param_groups(
        self, decoder: nn.Module, *, base_learning_rate: float
    ) -> Sequence[Mapping[str, Any]]:
        if base_learning_rate <= 0:
            raise FixedControlContractError("base learning rate must be positive")
        parameters = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
        if not parameters:
            raise FixedControlContractError("fixed decoder has no trainable parameters")
        return ({"name": "decoder", "params": parameters, "lr": float(base_learning_rate)},)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "method_id": self.method_id,
            "readout_mode": self.readout_mode,
            "embedding_sha256": self.embedding_sha256,
            "target_access": "loss_only",
            "full_vocabulary": True,
            "a2": False,
            "trainable_readout_parameters": 0,
            "query_interface": "normalized_query_rows_plus_inherited_logit_scale",
        }


def _validate_query_geometry(
    query_rows: torch.Tensor,
    embedding: torch.Tensor,
    logit_scale: torch.Tensor,
) -> None:
    if query_rows.ndim != 2 or query_rows.shape[0] <= 0 or query_rows.shape[1] <= 0:
        raise FixedControlContractError("query rows must be a non-empty [rows,hidden] tensor")
    if embedding.ndim != 2 or embedding.shape[1] != query_rows.shape[1]:
        raise FixedControlContractError("public embedding geometry differs from query rows")
    if not query_rows.is_floating_point() or not embedding.is_floating_point():
        raise FixedControlContractError("query rows and embedding must be floating point")
    if logit_scale.ndim != 0 or not logit_scale.is_floating_point():
        raise FixedControlContractError("inherited logit scale must be a floating scalar")
    if not torch.isfinite(logit_scale).item():
        raise FixedControlContractError("inherited logit scale is non-finite")


def _public_logits(
    query_rows: torch.Tensor,
    embedding: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    _validate_query_geometry(query_rows, embedding, logit_scale)
    logits = query_rows.to(embedding.dtype) @ embedding.transpose(0, 1)
    result = logits.float() * logit_scale
    if not torch.isfinite(result).all().item():
        raise FixedControlContractError("readout logits are non-finite")
    return result


def shared_decoder_rows(
    decoder: Any,
    hook: ReadoutHook,
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
    record_slots: torch.Tensor,
    position_slots: torch.Tensor,
    embedding: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    compute_base_logits: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor, Mapping[str, torch.Tensor]]:
    """Run the common decoder path and then the method-specific hook.

    The decoder owns query normalization and the inherited logit scale. Target
    IDs are passed only to ``loss_terms``; the hook never receives targets
    while scoring rows, which prevents target-dependent routing or readout
    behavior. ``compute_base_logits=False`` lets a directional hook avoid a
    duplicate public-E projection when the baseline tensor is not needed.
    """

    projected = decoder.projected_hidden(activation, valid_mask)
    base_logits = None
    if compute_base_logits:
        base_logits = decoder.logits_from_rows(
            projected, record_slots, position_slots, embedding
        )
    query_rows = projected[
        record_slots.to(projected.device), position_slots.to(projected.device)
    ]
    logit_scale = getattr(decoder, "logit_scale", None)
    if not isinstance(logit_scale, torch.Tensor):
        raise FixedControlContractError("common decoder must expose a tensor logit_scale")
    logits = hook.score_rows(
        query_rows,
        logit_scale,
        embedding,
        base_logits=base_logits,
    )
    expected_shape = (query_rows.shape[0], embedding.shape[0])
    if tuple(logits.shape) != expected_shape:
        raise FixedControlContractError("readout hook changed full-vocabulary logit geometry")
    losses = hook.loss_terms(logits, target_ids)
    if "total" not in losses:
        raise FixedControlContractError("readout hook did not return total loss")
    return base_logits, logits, losses


def contract_digest(contract: BankContract) -> str:
    """Stable digest of opaque contract metadata, excluding private payloads."""

    contract.validate()
    payload = {
        "fit_manifest": contract.fit_manifest.__dict__,
        "validation_manifest": contract.validation_manifest.__dict__,
        "embedding": contract.embedding.__dict__,
        "schedule": contract.schedule.__dict__,
        "fit_shape": contract.fit_shape,
        "validation_shape": contract.validation_shape,
        "fit_mask_shape": contract.fit_mask_shape,
        "validation_mask_shape": contract.validation_mask_shape,
        "fit_post_bos_rows": contract.fit_post_bos_rows,
        "fit_supported_token_count": contract.fit_supported_token_count,
        "schedule_semantic_sha256": contract.schedule_semantic_sha256,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
