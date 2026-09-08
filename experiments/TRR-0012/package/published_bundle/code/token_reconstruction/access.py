"""Data structures that make the permitted reconstruction boundary explicit."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

import torch


BOS_TOKEN_ID = 128000
OBSERVATION_SCHEMA = "token-reconstruction.boundary-observation.v1"


class AccessContractError(ValueError):
    """Raised when an observation violates the declared access contract."""


_FORBIDDEN_METADATA_FRAGMENTS = (
    "input_id",
    "token_id",
    "target",
    "truth",
    "oracle",
    "label",
    "decoded",
    "scalar_loss",
)


def _check_metadata(value: Any, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise AccessContractError(f"{path} keys must be strings")
            key = raw_key.casefold().replace("-", "_").replace(" ", "_")
            if any(fragment in key for fragment in _FORBIDDEN_METADATA_FRAGMENTS):
                raise AccessContractError(
                    f"{path}.{raw_key} could carry prohibited per-record truth"
                )
            _check_metadata(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_metadata(child, f"{path}[{index}]")


@dataclass(frozen=True)
class BoundaryObservation:
    """One permitted observation, intentionally containing no source-token field."""

    activation: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cut_depth: int
    source_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.activation.ndim != 3:
            raise AccessContractError("activation must have shape [batch, sequence, hidden]")
        batch, sequence, hidden = self.activation.shape
        if batch <= 0 or sequence <= 0 or hidden <= 0:
            raise AccessContractError("activation dimensions must all be positive")
        if not self.activation.dtype.is_floating_point:
            raise AccessContractError("activation must use a floating-point dtype")
        if not torch.isfinite(self.activation).all().item():
            raise AccessContractError("activation contains a non-finite value")

        expected_shape = (batch, sequence)
        if tuple(self.attention_mask.shape) != expected_shape:
            raise AccessContractError("attention_mask must match activation batch/sequence")
        if tuple(self.position_ids.shape) != expected_shape:
            raise AccessContractError("position_ids must match activation batch/sequence")
        if self.attention_mask.dtype.is_floating_point:
            raise AccessContractError("attention_mask must use a Boolean or integer dtype")
        if self.position_ids.dtype.is_floating_point:
            raise AccessContractError("position_ids must use an integer dtype")
        if not torch.logical_or(self.attention_mask.eq(0), self.attention_mask.eq(1)).all().item():
            raise AccessContractError("attention_mask must be binary")
        if not self.attention_mask.to(torch.bool).any(dim=1).all().item():
            raise AccessContractError("every observation row must contain an active position")
        active_positions = self.position_ids[self.attention_mask.to(torch.bool)]
        if active_positions.lt(0).any().item():
            raise AccessContractError("active position IDs must be non-negative")

        if not isinstance(self.cut_depth, int) or isinstance(self.cut_depth, bool):
            raise AccessContractError("cut_depth must be an integer")
        if self.cut_depth < 0:
            raise AccessContractError("cut_depth must be non-negative")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise AccessContractError("source_id must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise AccessContractError("metadata must be a mapping")
        _check_metadata(self.metadata)
        try:
            json.dumps(dict(self.metadata), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise AccessContractError("metadata must be finite JSON data") from exc

    @property
    def schema(self) -> str:
        return OBSERVATION_SCHEMA
