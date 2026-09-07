"""Thin TRR-0010 adapter surface for the shared TRR-P09 runner.

The P09 runner owns decoder execution, sampling, validation, checkpoints,
optimizer construction, resource guards, and timing.  This module only
builds the directional readout and dispatches one batch through the reviewed
P09 ``shared_decoder_rows`` hook.  It deliberately does not load data,
choose positions, or implement a second training loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from types import ModuleType
from typing import Any

import torch
from torch import nn

from trr0010_model import (
    DEFAULT_ANCHOR_STRENGTH,
    DEFAULT_EMBEDDING_EPSILON,
    DirectionalTokenReadout,
    build_directional_from_base,
)


TASK_ID = "TRR-0010"
P09_ADAPTER_MODULE = "token_reconstruction.trr_p09_fixed_control_adapter"
P09_API_COMMIT = "0aee09c4abb8d65b6ed3a42dc8ae93789509a518"
P09_LATEST_REVIEWED_SNAPSHOT = "eaa1f1e5aa0be1e6d0dcb896834f4d6ac4e43861"

_REQUIRED_P09_SYMBOLS = (
    "BankContract",
    "TrainingContract",
    "ReadoutHook",
    "shared_decoder_rows",
    "contract_digest",
)


class P09IntegrationError(ValueError):
    """Raised when the reviewed common-runner surface is unavailable."""


def validate_p09_module(protocol: ModuleType | Any) -> Any:
    """Validate the minimal reviewed P09 module surface.

    The wrapper does not copy or vendor the common adapter.  The caller hands
    in the module loaded from the shared runner's checkout, which lets the
    outer run bind its exact source commit and artifact hashes.
    """

    missing = [name for name in _REQUIRED_P09_SYMBOLS if not hasattr(protocol, name)]
    if missing:
        raise P09IntegrationError(
            "P09 protocol is missing required symbols: " + ", ".join(missing)
        )
    if not callable(getattr(protocol, "shared_decoder_rows")):
        raise P09IntegrationError("P09 shared_decoder_rows must be callable")
    if not callable(getattr(protocol, "contract_digest")):
        raise P09IntegrationError("P09 contract_digest must be callable")
    return protocol


def load_p09_module(module_name: str = P09_ADAPTER_MODULE) -> Any:
    """Load and validate the common adapter without duplicating its source."""

    try:
        protocol = import_module(module_name)
    except ImportError as exc:  # pragma: no cover - exercised by handoff only
        raise P09IntegrationError(
            f"P09 adapter is not installed at {module_name}; bind the A2 handoff first"
        ) from exc
    return validate_p09_module(protocol)


def build_directional_hook(
    base_decoder: nn.Module,
    support_ids: torch.Tensor | Sequence[int],
    support_counts: torch.Tensor | Sequence[int],
    public_embedding: torch.Tensor,
    *,
    anchor_strength: float = DEFAULT_ANCHOR_STRENGTH,
    embedding_epsilon: float = DEFAULT_EMBEDDING_EPSILON,
) -> DirectionalTokenReadout:
    """Build and bind the directional hook once per shared-runner arm."""

    hook = build_directional_from_base(
        base_decoder,
        support_ids,
        support_counts,
        anchor_strength=anchor_strength,
        embedding_epsilon=embedding_epsilon,
    )
    # The public embedding is bound once for the frequency-weighted anchor;
    # the runner continues to pass the immutable table to score_rows.
    hook.bind_embedding_statistics(public_embedding)
    return hook


def dispatch_shared_decoder_rows(
    protocol: Any,
    decoder: Any,
    hook: DirectionalTokenReadout,
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
    record_slots: torch.Tensor,
    position_slots: torch.Tensor,
    embedding: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    compute_base_logits: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, Mapping[str, torch.Tensor]]:
    """Dispatch one sampled batch through A2's common row hook.

    ``compute_base_logits=False`` is the directional default because the
    merged effective-E path computes the full-vocabulary result once.  The
    common runner remains responsible for selecting this setting and owns all
    target/position tensors; this function merely forwards them unchanged.
    """

    validate_p09_module(protocol)
    if not isinstance(hook, DirectionalTokenReadout):
        raise P09IntegrationError("TRR-0010 dispatch requires DirectionalTokenReadout")
    return protocol.shared_decoder_rows(
        decoder,
        hook,
        activation,
        valid_mask,
        record_slots,
        position_slots,
        embedding,
        target_ids,
        compute_base_logits=compute_base_logits,
    )


def optimizer_groups_for_p09(
    hook: DirectionalTokenReadout,
    decoder: nn.Module,
    *,
    base_learning_rate: float,
) -> Sequence[Mapping[str, Any]]:
    """Expose hook groups; the P09 runner still creates/steps the optimizer."""

    if decoder is not hook.base:
        raise P09IntegrationError("decoder must be the directional hook's shared base")
    return hook.optimizer_param_groups(decoder, base_learning_rate=base_learning_rate)


__all__ = [
    "P09_ADAPTER_MODULE",
    "P09_API_COMMIT",
    "P09_LATEST_REVIEWED_SNAPSHOT",
    "P09IntegrationError",
    "build_directional_hook",
    "dispatch_shared_decoder_rows",
    "load_p09_module",
    "optimizer_groups_for_p09",
    "validate_p09_module",
]
