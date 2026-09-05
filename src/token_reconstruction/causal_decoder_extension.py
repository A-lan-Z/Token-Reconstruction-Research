"""Causal, activation-only residual extensions for a frozen affine decoder.

This module defines the small TRR-0004 contextual pilot.  A freshly fitted
historical-style affine state (``W``, ``b``, and scalar log-scale ``s``) is the
competent base and remains frozen.  The added path sees only the sequence of
cut-4 activations and a validity mask.  It includes the current/BOS activation,
uses a fixed non-affine per-position layer normalization before either added
path, and has a zero-initialized output projection so the initial extension is
exactly the base on valid positions.

There is no token input, teacher prefix, candidate generation, or A2 fallback
in this forward path.  The two added paths are deliberately parameter matched:
width-128 one-head causal attention and width-256 positionwise MLP.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn


CAUSAL_ATTENTION_METHOD = "causal_h_attention128"
POSITIONWISE_MLP_METHOD = "positionwise_mlp256"
EXTENSION_METHODS = (CAUSAL_ATTENTION_METHOD, POSITIONWISE_MLP_METHOD)
FIXED_INPUT_NORMALIZATION = "torch.nn.functional.layer_norm(x, (hidden_size,), weight=None, bias=None, eps=1e-5)"


class CausalDecoderExtensionError(RuntimeError):
    """Raised when a frozen base or contextual extension violates its contract."""


def _check_matrix(value: torch.Tensor, *, name: str, shape: tuple[int, ...] | None = None) -> None:
    if value.ndim != 2 or (shape is not None and tuple(value.shape) != shape):
        raise CausalDecoderExtensionError(f"{name} has invalid matrix geometry")
    if not value.dtype.is_floating_point:
        raise CausalDecoderExtensionError(f"{name} must be floating point")
    if not torch.isfinite(value).all().item():
        raise CausalDecoderExtensionError(f"{name} contains non-finite values")


def _checked_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if set(state) != {"W", "b", "s"}:
        raise CausalDecoderExtensionError("affine base state must contain exactly W, b, and s")
    W, b, s = state["W"], state["b"], state["s"]
    if not all(isinstance(value, torch.Tensor) for value in (W, b, s)):
        raise CausalDecoderExtensionError("affine base state values must be tensors")
    if W.ndim != 2 or W.shape[0] <= 0 or W.shape[1] != W.shape[0]:
        raise CausalDecoderExtensionError("affine base W must be square and non-empty")
    hidden_size = int(W.shape[0])
    _check_matrix(W, name="affine base W", shape=(hidden_size, hidden_size))
    if b.ndim != 1 or tuple(b.shape) != (hidden_size,):
        raise CausalDecoderExtensionError("affine base b geometry changed")
    if s.ndim != 0:
        raise CausalDecoderExtensionError("affine base s must be scalar")
    if any(value.dtype != torch.float32 for value in (W, b, s)):
        raise CausalDecoderExtensionError("affine base state must remain float32")
    if not all(torch.isfinite(value).all().item() for value in (W, b, s)):
        raise CausalDecoderExtensionError("affine base state contains non-finite values")
    return {
        "W": W.detach().cpu().contiguous().clone(),
        "b": b.detach().cpu().contiguous().clone(),
        "s": s.detach().cpu().contiguous().clone(),
    }


def _runtime_embedding_geometry(
    embedding_table: torch.Tensor,
    *,
    hidden_size: int,
    vocab_size: int | None = None,
    device: torch.device | None = None,
) -> None:
    """Perform cheap checks suitable for a hot projection loop."""

    if embedding_table.ndim != 2 or embedding_table.shape[1] != hidden_size:
        raise CausalDecoderExtensionError("embedding table hidden geometry changed")
    if vocab_size is not None and embedding_table.shape[0] != vocab_size:
        raise CausalDecoderExtensionError("embedding table vocabulary geometry changed")
    if not embedding_table.dtype.is_floating_point:
        raise CausalDecoderExtensionError("embedding table must be floating point")
    if device is not None and embedding_table.device != device:
        raise CausalDecoderExtensionError("embedding table must be transferred to the activation device once")


def validate_runtime_embeddings(
    embedding_table: torch.Tensor,
    *,
    hidden_size: int,
    vocab_size: int | None = None,
) -> None:
    """Validate the fixed normalized public table at its resource boundary.

    Call this once after loading the table. Forward paths use only the cheap
    geometry/device checks above so a 1 GiB table is never rescanned per step.
    """

    _runtime_embedding_geometry(embedding_table, hidden_size=hidden_size, vocab_size=vocab_size)
    if not torch.isfinite(embedding_table).all().item():
        raise CausalDecoderExtensionError("embedding table contains non-finite values")


def _runtime_inputs(
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    if activation.ndim != 3 or activation.shape[-1] != hidden_size:
        raise CausalDecoderExtensionError("activation input must be [batch, sequence, hidden]")
    if activation.shape[0] <= 0 or activation.shape[1] <= 0:
        raise CausalDecoderExtensionError("activation input cannot be empty")
    if not activation.dtype.is_floating_point:
        raise CausalDecoderExtensionError("activation input must be floating point")
    if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(activation.shape[:2]):
        raise CausalDecoderExtensionError("valid mask geometry does not match activations")
    if valid_mask.dtype not in (torch.bool, torch.uint8):
        raise CausalDecoderExtensionError("valid mask must be boolean")
    return valid_mask.to(device=activation.device, dtype=torch.bool)


def fixed_input_normalization(activation: torch.Tensor, *, hidden_size: int) -> torch.Tensor:
    """Apply the shared nonlearned per-position normalization for added paths."""

    if activation.shape[-1] != hidden_size:
        raise CausalDecoderExtensionError("input normalization hidden size changed")
    # No affine parameters are fitted or retained here.  This normalization is
    # shared by attention and MLP additions and is applied only to their input;
    # the historical-style affine base still receives its original activation.
    return F.layer_norm(
        activation.float(),
        (hidden_size,),
        weight=None,
        bias=None,
        eps=1e-5,
    )


class FrozenAffineBase(nn.Module):
    """A historical-style affine base with frozen W, b, and log-scale s."""

    def __init__(self, state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        checked = _checked_state(state)
        self.hidden_size = int(checked["W"].shape[0])
        self.register_buffer("W", checked["W"])
        self.register_buffer("b", checked["b"])
        self.register_buffer("s", checked["s"])
        self.requires_grad_(False)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> "FrozenAffineBase":
        """Construct a base from a Track-A-compatible ``W``, ``b``, ``s`` state."""

        return cls(state).eval()

    @property
    def logit_scale(self) -> float:
        return float(self.s.float().exp().item())

    def projected(self, activation: torch.Tensor) -> torch.Tensor:
        """Apply W.T orientation and bias in float32, before normalization."""

        _runtime_inputs(
            activation,
            torch.ones(activation.shape[:2], dtype=torch.bool, device=activation.device),
            hidden_size=self.hidden_size,
        )
        value = activation.float()
        return value @ self.W.float().T + self.b.float()

    def logits(self, activation: torch.Tensor, embedding_table: torch.Tensor) -> torch.Tensor:
        """Return the frozen base's historical normalized tied-projection logits."""

        _runtime_embedding_geometry(
            embedding_table, hidden_size=self.hidden_size, device=activation.device
        )
        projected = F.normalize(self.projected(activation), dim=-1)
        logits = projected.to(embedding_table.dtype) @ embedding_table.transpose(-1, -2)
        return logits.float() * self.s.float().exp()


class _AddedPath(nn.Module):
    """Common contract for causal and positionwise added paths."""

    method_id: str
    added_parameter_count: int

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise CausalDecoderExtensionError("hidden size must be positive")
        self.hidden_size = int(hidden_size)
        self.input_normalization = FIXED_INPUT_NORMALIZATION

    def _normalized_input(self, activation: torch.Tensor) -> torch.Tensor:
        return fixed_input_normalization(activation, hidden_size=self.hidden_size)

    def _mask(self, valid_mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        return valid_mask.to(device=device, dtype=torch.bool)


class CausalHAttention128(_AddedPath):
    """One-head causal attention over H-only activations, width 128."""

    method_id = CAUSAL_ATTENTION_METHOD

    def __init__(self, hidden_size: int, *, context_width: int = 128) -> None:
        super().__init__(hidden_size)
        if context_width <= 0:
            raise CausalDecoderExtensionError("attention context width must be positive")
        self.context_width = int(context_width)
        self.query = nn.Linear(hidden_size, context_width)
        self.key = nn.Linear(hidden_size, context_width)
        self.value = nn.Linear(hidden_size, context_width)
        self.output = nn.Linear(context_width, hidden_size)
        # Zero output makes the complete extension equal to the frozen base at
        # initialization while still leaving a trainable contextual path.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.added_parameter_count = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = _runtime_inputs(activation, valid_mask, hidden_size=self.hidden_size)
        value = self._normalized_input(activation)
        query = self.query(value)
        key = self.key(value)
        val = self.value(value)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.context_width)
        sequence = int(activation.shape[1])
        positions = torch.arange(sequence, device=activation.device)
        causal = positions[None, :] <= positions[:, None]
        allowed_keys = causal.unsqueeze(0) & mask[:, None, :]
        masked_scores = scores.masked_fill(~allowed_keys, float("-inf"))
        valid_query = mask.unsqueeze(-1)
        # Avoid all -inf rows for padded queries before softmax.
        safe_scores = torch.where(valid_query, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(safe_scores, dim=-1)
        weights = torch.where(valid_query, weights, torch.zeros_like(weights))
        attended = weights @ val
        output = self.output(attended)
        return output * mask.unsqueeze(-1).to(dtype=output.dtype)


class PositionwiseMLP256(_AddedPath):
    """Positionwise nonlinear residual path with width 256."""

    method_id = POSITIONWISE_MLP_METHOD

    def __init__(self, hidden_size: int, *, bottleneck_size: int = 256) -> None:
        super().__init__(hidden_size)
        if bottleneck_size <= 0:
            raise CausalDecoderExtensionError("MLP bottleneck size must be positive")
        self.bottleneck_size = int(bottleneck_size)
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        # The path is zero at initialization, while its hidden transform is
        # otherwise ordinary trainable capacity.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.added_parameter_count = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = _runtime_inputs(activation, valid_mask, hidden_size=self.hidden_size)
        value = self._normalized_input(activation)
        output = self.up(F.gelu(self.down(value)))
        return output * mask.unsqueeze(-1).to(dtype=output.dtype)


class CausalResidualDecoder(nn.Module):
    """Frozen affine base plus one zero-initialized activation-context path."""

    def __init__(self, base: FrozenAffineBase, added_path: _AddedPath) -> None:
        super().__init__()
        if base.hidden_size != added_path.hidden_size:
            raise CausalDecoderExtensionError("base and added path hidden sizes differ")
        self.base = base.eval()
        self.added_path = added_path
        self.method_id = added_path.method_id
        self.hidden_size = base.hidden_size
        self.added_parameter_count = sum(parameter.numel() for parameter in self.added_path.parameters())
        self.base.requires_grad_(False)

    @property
    def logit_scale(self) -> float:
        return self.base.logit_scale

    def pre_normalized_hidden(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the combined hidden state before output normalization.

        This is the training-facing interface: callers can compute the causal
        sequence pass once, select a common set of at most 512 valid loss
        positions, and apply the full-vocabulary projection only to those
        rows.
        """

        mask = _runtime_inputs(activation, valid_mask, hidden_size=self.hidden_size)
        base_raw = self.base.projected(activation)
        added = self.added_path(activation, mask)
        combined = base_raw + added
        return torch.where(mask.unsqueeze(-1), combined, torch.zeros_like(combined))

    def projected_hidden(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized hidden rows before the tied vocabulary projection."""

        mask = _runtime_inputs(activation, valid_mask, hidden_size=self.hidden_size)
        combined = self.pre_normalized_hidden(activation, mask)
        projected = F.normalize(combined, dim=-1)
        return torch.where(mask.unsqueeze(-1), projected, torch.zeros_like(projected))

    def logits_from_projected_hidden(
        self,
        projected_hidden: torch.Tensor,
        selected_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        """Project only selected hidden rows through the public vocabulary.

        ``projected_hidden`` is normally produced once by
        :meth:`projected_hidden`.  Keeping this projection separate lets a
        trainer cap each full-vocabulary logits tensor at the common selected
        position budget while still giving attention the complete sequence.
        """

        if projected_hidden.ndim != 3 or projected_hidden.shape[-1] != self.hidden_size:
            raise CausalDecoderExtensionError("projected hidden geometry changed")
        if selected_mask.ndim != 2 or tuple(selected_mask.shape) != tuple(projected_hidden.shape[:2]):
            raise CausalDecoderExtensionError("selected loss mask geometry does not match hidden rows")
        if selected_mask.dtype not in (torch.bool, torch.uint8):
            raise CausalDecoderExtensionError("selected loss mask must be boolean")
        selected = selected_mask.to(device=projected_hidden.device, dtype=torch.bool)
        if not selected.any().item():
            raise CausalDecoderExtensionError("selected loss positions are empty")
        _runtime_embedding_geometry(
            embedding_table, hidden_size=self.hidden_size, device=projected_hidden.device
        )
        hidden = projected_hidden[selected]
        logits = hidden.to(embedding_table.dtype) @ embedding_table.transpose(-1, -2)
        return logits.float() * self.base.s.float().exp()

    def selected_logits(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        selected_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        """Project only selected valid rows through the full public vocabulary."""

        mask = _runtime_inputs(activation, valid_mask, hidden_size=self.hidden_size)
        if selected_mask.ndim != 2 or tuple(selected_mask.shape) != tuple(mask.shape):
            raise CausalDecoderExtensionError("selected loss mask geometry does not match activations")
        if selected_mask.dtype not in (torch.bool, torch.uint8):
            raise CausalDecoderExtensionError("selected loss mask must be boolean")
        selected = selected_mask.to(device=activation.device, dtype=torch.bool)
        if (selected & ~mask).any().item():
            raise CausalDecoderExtensionError("selected loss positions must be valid")
        hidden = self.projected_hidden(activation, mask)
        return self.logits_from_projected_hidden(hidden, selected, embedding_table)

    def forward(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        """Emit full-vocabulary logits from H and a causal validity mask only."""

        mask = _runtime_inputs(activation, valid_mask, hidden_size=self.hidden_size)
        _runtime_embedding_geometry(
            embedding_table, hidden_size=self.hidden_size, device=activation.device
        )
        projected = self.projected_hidden(activation, mask)
        logits = projected.to(embedding_table.dtype) @ embedding_table.transpose(-1, -2)
        logits = logits.float() * self.base.s.float().exp()
        # Invalid positions are never loss positions and serialize as a finite
        # neutral row, avoiding accidental use of padding in a caller.
        return torch.where(mask.unsqueeze(-1), logits, torch.zeros_like(logits))

    def base_logits(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        """Return the frozen base logits with the same invalid-row convention."""

        mask = _runtime_inputs(activation, valid_mask, hidden_size=self.hidden_size)
        logits = self.base.logits(activation, embedding_table)
        return torch.where(mask.unsqueeze(-1), logits, torch.zeros_like(logits))

    def trainable_parameters(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.added_path.parameters())


def build_causal_extension(
    base: FrozenAffineBase,
    method_id: Literal["causal_h_attention128", "positionwise_mlp256"],
) -> CausalResidualDecoder:
    """Build one registered parameter-matched extension around a frozen base."""

    if method_id == CAUSAL_ATTENTION_METHOD:
        path: _AddedPath = CausalHAttention128(base.hidden_size, context_width=128)
    elif method_id == POSITIONWISE_MLP_METHOD:
        path = PositionwiseMLP256(base.hidden_size, bottleneck_size=256)
    else:
        raise CausalDecoderExtensionError(f"unknown contextual extension method: {method_id}")
    return CausalResidualDecoder(base, path)


def extension_parameter_counts(hidden_size: int = 2048) -> dict[str, int]:
    """Return added-path counts used by the preregistered comparison."""

    if hidden_size <= 0:
        raise CausalDecoderExtensionError("hidden size must be positive")
    attention = 3 * (hidden_size * 128 + 128) + (128 * hidden_size + hidden_size)
    mlp = (hidden_size * 256 + 256) + (256 * hidden_size + hidden_size)
    return {
        CAUSAL_ATTENTION_METHOD: attention,
        POSITIONWISE_MLP_METHOD: mlp,
    }
