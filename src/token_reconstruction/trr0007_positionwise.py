"""TRR-0007 current-H positionwise capacity extension.

The current-family control is the TRR-0005 trained-diagonal decoder.  The
single extension in this task adds a zero-output nonlinear residual path over
the current activation, while retaining the complete diagonal affine path and
the tied full-vocabulary public embedding projection.

Both methods consume only ``H_i`` at inference.  The strict diagonal base has
one allowed attention key for each query, so its output correction is
positionwise even though it retains the TRR-0005 state layout.  The extension
uses a fixed per-position layer normalization followed by a 2048 -> 512 ->
2048 GELU MLP.  The final MLP projection is zero-initialized, making a freshly
built extension exactly equal to a freshly built current-family control.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn

from .trr0005_joint_decoder import (
    ATTENTION_SCORE_MODE_DOT_PRODUCT,
    BOS_TOKEN_ID,
    DEFAULT_CONTEXT_WIDTH,
    DIAGONAL_ATTENTION_METHOD,
    JointAffineAttentionDecoder,
    JointDecoderError,
    build_decoder,
    file_sha256,
)


TASK_ID = "TRR-0007"
SCHEMA = "token-reconstruction.trr0007-positionwise.v1"
CURRENT_METHOD_ID = "trr0007_current_positionwise"
RESIDUAL_MLP_METHOD_ID = "trr0007_residual_mlp512"
METHODS = (CURRENT_METHOD_ID, RESIDUAL_MLP_METHOD_ID)
DEFAULT_BOTTLENECK_SIZE = 512
DEFAULT_SEED = 4005
BASE_METHOD_ID = DIAGONAL_ATTENTION_METHOD
BASE_STATE_SHA256 = "696eb9fc951e85356a06575faf18a2011616692a086bdac3b2fa368e69d599a2"


class PositionwiseDecoderError(JointDecoderError):
    """Raised when a TRR-0007 state or current-H model contract is invalid."""


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"trr0007-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }


def _check_model_id(method_id: str) -> str:
    if method_id not in METHODS:
        raise PositionwiseDecoderError(f"unknown TRR-0007 method: {method_id}")
    return method_id


def build_current_positionwise(
    *,
    hidden_size: int,
    vocabulary_size: int,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    seed: int = DEFAULT_SEED,
) -> JointAffineAttentionDecoder:
    """Build the neutral current-family diagonal control."""

    return build_decoder(
        BASE_METHOD_ID,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
        seed=seed,
        attention_score_mode=ATTENTION_SCORE_MODE_DOT_PRODUCT,
    )


def _deterministic_down_init(linear: nn.Linear, *, seed: int) -> None:
    """Initialize the nonlinear input projection without touching global RNG."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bound = (1.0 / float(linear.in_features)) ** 0.5
    with torch.no_grad():
        linear.weight.copy_(
            torch.rand(linear.weight.shape, generator=generator, dtype=linear.weight.dtype)
            * (2.0 * bound)
            - bound
        )
        linear.bias.zero_()


class ResidualMLPPositionwiseDecoder(nn.Module):
    """Trainable TRR-0005 diagonal base plus a zero-output residual MLP."""

    method_id = RESIDUAL_MLP_METHOD_ID

    def __init__(
        self,
        base: JointAffineAttentionDecoder,
        *,
        bottleneck_size: int = DEFAULT_BOTTLENECK_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> None:
        super().__init__()
        if base.method_id != BASE_METHOD_ID:
            raise PositionwiseDecoderError(
                "TRR-0007 residual model requires the strict diagonal base"
            )
        if bottleneck_size <= 0:
            raise PositionwiseDecoderError("MLP bottleneck must be positive")
        self.base = base
        self.hidden_size = int(base.hidden_size)
        self.vocabulary_size = int(base.vocabulary_size)
        self.context_width = int(base.context_width)
        self.bottleneck_size = int(bottleneck_size)
        self.attention_score_mode = str(base.attention_score_mode)
        self.base_method_id = BASE_METHOD_ID
        self.input_normalization = (
            "torch.nn.functional.layer_norm(H_i, (hidden_size,), "
            "weight=None, bias=None, eps=1e-5)"
        )
        self.down = nn.Linear(self.hidden_size, self.bottleneck_size)
        self.up = nn.Linear(self.bottleneck_size, self.hidden_size)
        _deterministic_down_init(self.down, seed=seed)
        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.base.logit_scale

    @property
    def attention_mode(self) -> str:
        return "diagonal"

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _check_inputs(
        self, activation: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        # Calling the base validator keeps the exact TRR-0005 BOS/mask rules.
        return self.base._check_inputs(activation, valid_mask)  # noqa: SLF001

    def _added_mlp(self, activation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = F.layer_norm(
            activation.float(),
            (self.hidden_size,),
            weight=None,
            bias=None,
            eps=1e-5,
        )
        added = self.up(F.gelu(self.down(value)))
        return added * mask.unsqueeze(-1).to(dtype=added.dtype)

    def pre_normalized_hidden(
        self, activation: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        # The base has a strict diagonal mask and therefore reads only H_i for
        # each output position.  The MLP is likewise evaluated independently
        # for each current position.
        combined = self.base.pre_normalized_hidden(activation, mask)
        combined = combined + self._added_mlp(activation, mask)
        return torch.where(mask.unsqueeze(-1), combined, torch.zeros_like(combined))

    def projected_hidden(
        self, activation: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        projected = F.normalize(self.pre_normalized_hidden(activation, mask), dim=-1)
        return torch.where(mask.unsqueeze(-1), projected, torch.zeros_like(projected))

    def logits_from_rows(
        self,
        projected_hidden: torch.Tensor,
        record_slots: torch.Tensor,
        position_slots: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        if projected_hidden.ndim != 3 or int(projected_hidden.shape[-1]) != self.hidden_size:
            raise PositionwiseDecoderError("projected hidden geometry changed")
        if record_slots.ndim != 1 or position_slots.ndim != 1:
            raise PositionwiseDecoderError("draw indices must be vectors")
        if tuple(record_slots.shape) != tuple(position_slots.shape):
            raise PositionwiseDecoderError("draw index vectors differ")
        if embedding_table.ndim != 2 or not embedding_table.dtype.is_floating_point:
            raise PositionwiseDecoderError("embedding table must be a floating matrix")
        if tuple(embedding_table.shape) != (self.vocabulary_size, self.hidden_size):
            raise PositionwiseDecoderError("embedding table geometry changed")
        rows = projected_hidden[
            record_slots.to(projected_hidden.device),
            position_slots.to(projected_hidden.device),
        ]
        logits = rows.to(embedding_table.dtype) @ embedding_table.transpose(0, 1)
        result = logits.float() * self.logit_scale
        if not torch.isfinite(result).all().item():
            raise PositionwiseDecoderError("decoder logits are non-finite")
        return result

    def selected_logits(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        selected_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        if selected_mask.ndim != 2 or tuple(selected_mask.shape) != tuple(mask.shape):
            raise PositionwiseDecoderError("selected mask geometry changed")
        selected = selected_mask.to(device=activation.device, dtype=torch.bool)
        if (selected & ~mask).any().item() or selected[:, 0].any().item():
            raise PositionwiseDecoderError("selected rows must be valid post-BOS positions")
        indices = torch.nonzero(selected, as_tuple=False)
        if int(indices.shape[0]) <= 0:
            raise PositionwiseDecoderError("selected rows are empty")
        hidden = self.projected_hidden(activation, mask)
        return self.logits_from_rows(hidden, indices[:, 0], indices[:, 1], embedding_table)

    def forward(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        if tuple(embedding_table.shape) != (self.vocabulary_size, self.hidden_size):
            raise PositionwiseDecoderError("embedding table geometry changed")
        hidden = self.projected_hidden(activation, mask)
        logits = hidden.to(embedding_table.dtype) @ embedding_table.transpose(0, 1)
        logits = logits.float() * self.logit_scale
        return torch.where(mask.unsqueeze(-1), logits, torch.zeros_like(logits))


def build_residual_mlp512(
    *,
    hidden_size: int,
    vocabulary_size: int,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    bottleneck_size: int = DEFAULT_BOTTLENECK_SIZE,
    seed: int = DEFAULT_SEED,
) -> ResidualMLPPositionwiseDecoder:
    """Build the one registered TRR-0007 capacity extension."""

    return ResidualMLPPositionwiseDecoder(
        build_current_positionwise(
            hidden_size=hidden_size,
            vocabulary_size=vocabulary_size,
            context_width=context_width,
            seed=seed,
        ),
        bottleneck_size=bottleneck_size,
        seed=seed,
    )


def load_retained_diagonal_state(
    path: Path,
    *,
    hidden_size: int,
    vocabulary_size: int,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
) -> JointAffineAttentionDecoder:
    """Load the published TRR-0006 diagonal state for a separate reference."""

    from .trr0005_joint_decoder import load_decoder_state

    loaded = load_decoder_state(
        path,
        method_id=BASE_METHOD_ID,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
    )
    if file_sha256(path) != BASE_STATE_SHA256:
        raise PositionwiseDecoderError(
            "retained reference state hash differs from the TRR-0006 selected state"
        )
    return loaded


def save_positionwise_state(
    path: Path,
    model: nn.Module,
    *,
    method_id: str,
    selected_step: int,
    initialization: str,
    distribution: str,
    bottleneck_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one selected TRR-0007 state with explicit model metadata."""

    _check_model_id(method_id)
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PositionwiseDecoderError(f"positionwise state is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _state_cpu(model)
    hidden_size = int(getattr(model, "hidden_size"))
    vocabulary_size = int(getattr(model, "vocabulary_size"))
    context_width = int(getattr(model, "context_width", DEFAULT_CONTEXT_WIDTH))
    if method_id == RESIDUAL_MLP_METHOD_ID and bottleneck_size is None:
        bottleneck_size = int(getattr(model, "bottleneck_size"))
    state_metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "method_id": method_id,
        "base_method_id": BASE_METHOD_ID,
        "attention_mode": "diagonal",
        "attention_score_mode": str(getattr(model, "attention_score_mode", "dot_product")),
        "hidden_size": hidden_size,
        "vocabulary_size": vocabulary_size,
        "context_width": context_width,
        "bottleneck_size": "none" if bottleneck_size is None else int(bottleneck_size),
        "selected_step": int(selected_step),
        "initialization": initialization,
        "distribution": distribution,
        "inference_contract": "current_activation_H_i_only; full_vocabulary_tied_E",
    }
    if metadata:
        state_metadata.update({str(key): value for key, value in metadata.items()})
    save_file(state, str(path), metadata={key: str(value) for key, value in state_metadata.items()})
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "state_sha256": _state_digest(state),
        "state_bytes": sum(int(value.numel()) * value.element_size() for value in state.values()),
        "tensor_keys": sorted(state),
        "selected_step": int(selected_step),
        "metadata": state_metadata,
    }


def load_positionwise_model_state(
    path: Path,
    *,
    method_id: str,
    hidden_size: int,
    vocabulary_size: int,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    bottleneck_size: int = DEFAULT_BOTTLENECK_SIZE,
) -> nn.Module:
    """Load a selected TRR-0007 current or residual state exactly."""

    _check_model_id(method_id)
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PositionwiseDecoderError(f"positionwise state must be a regular file: {path}")
    try:
        state = load_file(str(path), device="cpu")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise PositionwiseDecoderError(f"cannot load positionwise state: {path}") from exc
    if metadata.get("schema") != SCHEMA:
        raise PositionwiseDecoderError("positionwise state schema is not TRR-0007")
    if metadata.get("method_id") != method_id:
        raise PositionwiseDecoderError(
            f"positionwise state method is {metadata.get('method_id')!r}, expected {method_id!r}"
        )
    if method_id == CURRENT_METHOD_ID:
        model: nn.Module = build_current_positionwise(
            hidden_size=hidden_size,
            vocabulary_size=vocabulary_size,
            context_width=context_width,
        )
    else:
        stored_bottleneck = metadata.get("bottleneck_size")
        if stored_bottleneck not in (None, "none"):
            try:
                bottleneck_size = int(stored_bottleneck)
            except (TypeError, ValueError) as exc:
                raise PositionwiseDecoderError("state bottleneck metadata is malformed") from exc
        model = build_residual_mlp512(
            hidden_size=hidden_size,
            vocabulary_size=vocabulary_size,
            context_width=context_width,
            bottleneck_size=bottleneck_size,
        )
    expected = set(model.state_dict())
    if set(state) != expected:
        raise PositionwiseDecoderError("positionwise state tensor keys differ")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise PositionwiseDecoderError("positionwise state geometry differs") from exc
    return model


def step_zero_equivalence(
    current: nn.Module,
    extension: ResidualMLPPositionwiseDecoder,
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    max_rows: int = 512,
) -> dict[str, Any]:
    """Check the neutral current/extension pair before training."""

    if max_rows <= 0:
        raise PositionwiseDecoderError("max_rows must be positive")
    with torch.inference_mode():
        current_hidden = current.projected_hidden(activation, valid_mask)
        extension_hidden = extension.projected_hidden(activation, valid_mask)
        indices = torch.nonzero(valid_mask.to(dtype=torch.bool), as_tuple=False)
        indices = indices[indices[:, 1] > 0][:max_rows]
        if int(indices.shape[0]) <= 0:
            raise PositionwiseDecoderError("equivalence fixture has no post-BOS rows")
        current_logits = current.logits_from_rows(
            current_hidden, indices[:, 0], indices[:, 1], embedding_table
        )
        extension_logits = extension.logits_from_rows(
            extension_hidden, indices[:, 0], indices[:, 1], embedding_table
        )
    hidden_delta = (current_hidden - extension_hidden).abs()
    logits_delta = (current_logits - extension_logits).abs()
    return {
        "rows_checked": int(indices.shape[0]),
        "projected_hidden_exact": bool(torch.equal(current_hidden, extension_hidden)),
        "logits_exact": bool(torch.equal(current_logits, extension_logits)),
        "max_projected_hidden_abs_delta": float(hidden_delta.max().cpu()),
        "max_logits_abs_delta": float(logits_delta.max().cpu()),
        "initialization": "neutral_diagonal_base; residual_up_zero",
    }


