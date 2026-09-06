"""TRR-P06 visibility-mask decoder family.

The P06 comparison changes only which already-observed activations may be used
by the added path.  Every arm has the same trainable direct affine readout and
the same zero-initialized activation path::

    z_i = H_i W.T + b + O(attention_i(H) V(LN(H)))

The three registered visibility modes are ``positionwise`` (the diagonal
control), ``past_only`` (a causal prefix), and ``full_record`` (all valid
positions in the current record).  The latter is intentionally an offline
full-record decoder.  No source tokens, guessed prefixes, or target feedback
are represented by this module.

The production fit supplies ``direct_state`` from the frozen competent public
affine checkpoint.  A missing direct state is permitted only for small unit
fixtures and is explicitly labelled as the identity fixture initialization;
the fit runner should always pass the published W/b/s state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn


TASK_ID = "TRR-P06"
SCHEMA = "token-reconstruction.trr0006-visibility-decoder.v1"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_VOCABULARY_SIZE = 128256
DEFAULT_CONTEXT_WIDTH = 128
DEFAULT_QKV_SEED = 6206
COSINE_SCALE = 4.0
ATTENTION_SCORE_MODE = "cosine_qk_normalized_scale4"

POSITIONWISE_METHOD = "p06_positionwise_diagonal"
PAST_ONLY_METHOD = "p06_past_only"
FULL_RECORD_METHOD = "p06_full_record"
METHODS = (POSITIONWISE_METHOD, PAST_ONLY_METHOD, FULL_RECORD_METHOD)
VisibilityMode = Literal["positionwise", "past_only", "full_record"]


class VisibilityDecoderError(RuntimeError):
    """Raised when a P06 decoder or its visibility contract is invalid."""


def file_sha256(path: Path) -> str:
    """Hash one regular file in bounded blocks."""

    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise VisibilityDecoderError(f"resource must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor shape, dtype, and contiguous CPU bytes."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a decoder state deterministically by sorted tensor name."""

    digest = hashlib.sha256(b"trr-p06-visibility-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        encoded = str(name).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        shape = json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        digest.update(len(shape).to_bytes(8, "big"))
        digest.update(shape)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_method(method_id: str) -> str:
    if method_id not in METHODS:
        raise VisibilityDecoderError(f"unknown P06 method: {method_id}")
    return method_id


def method_visibility(method_id: str) -> VisibilityMode:
    """Return the registered visibility mode for a method ID."""

    _validate_method(method_id)
    if method_id == POSITIONWISE_METHOD:
        return "positionwise"
    if method_id == PAST_ONLY_METHOD:
        return "past_only"
    return "full_record"


def _validate_valid_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.ndim != 2 or valid_mask.shape[0] <= 0 or valid_mask.shape[1] <= 1:
        raise VisibilityDecoderError("valid mask must be [records, positions>1]")
    if valid_mask.dtype not in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise VisibilityDecoderError("valid mask must be boolean or integer")
    mask = valid_mask.to(dtype=torch.bool)
    if not mask[:, 0].all().item():
        raise VisibilityDecoderError("every row must contain a valid BOS position")
    # Right padding is part of the P06 observation contract.  Rejecting a
    # hole here prevents a future activation from entering through malformed
    # metadata while retaining the actual per-row lengths.
    for row in mask.detach().cpu().tolist():
        false_seen = False
        for item in row:
            if not item:
                false_seen = True
            elif false_seen:
                raise VisibilityDecoderError("valid mask must be right-padded")
    return mask


def build_visibility_mask(method_id: str, valid_mask: torch.Tensor) -> torch.Tensor:
    """Build ``[records, query, key]`` visibility for one batch.

    Invalid queries and invalid keys are both masked.  Thus a padded query has
    no allowed keys; the attention implementation handles that case without
    applying softmax to an all-``-inf`` row.
    """

    mode = method_visibility(method_id)
    mask = _validate_valid_mask(valid_mask)
    positions = torch.arange(int(mask.shape[1]), device=mask.device)
    if mode == "positionwise":
        base = torch.eye(int(mask.shape[1]), dtype=torch.bool, device=mask.device)
    elif mode == "past_only":
        base = positions[:, None] >= positions[None, :]
    else:
        base = torch.ones(
            (int(mask.shape[1]), int(mask.shape[1])),
            dtype=torch.bool,
            device=mask.device,
        )
    return (
        base.unsqueeze(0)
        & mask[:, :, None]
        & mask[:, None, :]
    )


def future_valid_count(valid_mask: torch.Tensor) -> torch.Tensor:
    """Return valid later-key counts for every query position."""

    mask = _validate_valid_mask(valid_mask)
    positions = torch.arange(int(mask.shape[1]), device=mask.device)
    later = positions[None, :] > positions[:, None]
    return (
        mask[:, None, :]
        & later.unsqueeze(0)
        & mask[:, :, None]
    ).sum(dim=-1)


def _deterministic_linear_init(linear: nn.Linear, generator: torch.Generator) -> None:
    fan_in, fan_out = linear.in_features, linear.out_features
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    with torch.no_grad():
        linear.weight.copy_(
            torch.rand(
                linear.weight.shape,
                dtype=torch.float32,
                generator=generator,
            )
            * (2.0 * bound)
            - bound
        )
        linear.bias.zero_()


def _validate_direct_state(
    direct_state: Mapping[str, torch.Tensor],
    *,
    hidden_size: int,
) -> dict[str, torch.Tensor]:
    required = {"W", "b", "s"}
    if not required.issubset(direct_state):
        raise VisibilityDecoderError(
            f"direct affine state must contain {sorted(required)}"
        )
    result: dict[str, torch.Tensor] = {}
    expected = {
        "W": (hidden_size, hidden_size),
        "b": (hidden_size,),
        "s": (),
    }
    for name, shape in expected.items():
        value = direct_state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise VisibilityDecoderError(
                f"direct affine tensor {name} has incorrect shape: "
                f"expected {shape}, observed {getattr(value, 'shape', None)}"
            )
        if not value.dtype.is_floating_point:
            raise VisibilityDecoderError(f"direct affine tensor {name} is not floating point")
        if not torch.isfinite(value).all().item():
            raise VisibilityDecoderError(f"direct affine tensor {name} is non-finite")
        result[name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return result


def load_direct_affine_initialization(
    path: Path,
    *,
    hidden_size: int,
    expected_sha256: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load only the public competent affine tensors from a safetensors file.

    The source file is expected to contain ``W``, ``b``, and scalar ``s``.  A
    path/hash binding is returned for a fit receipt; no observations or target
    resources are opened by this helper.
    """

    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise VisibilityDecoderError(f"direct affine initialization is unavailable: {path}")
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise VisibilityDecoderError(
            f"direct affine initialization hash changed: expected {expected_sha256}, observed {digest}"
        )
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            state = {name: handle.get_tensor(name).contiguous() for name in ("W", "b", "s") if name in keys}
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise VisibilityDecoderError(f"cannot load direct affine initialization: {path}") from exc
    validated = _validate_direct_state(state, hidden_size=hidden_size)
    return validated, {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": digest,
        "source_metadata": metadata,
        "state_sha256": state_sha256(validated),
        "initialization": "competent_public_affine",
    }


class VisibilityAffineAttentionDecoder(nn.Module):
    """One P06 decoder with an explicit activation visibility mask."""

    def __init__(
        self,
        hidden_size: int,
        vocabulary_size: int,
        method_id: str,
        *,
        context_width: int = DEFAULT_CONTEXT_WIDTH,
        qkv_seed: int = DEFAULT_QKV_SEED,
        direct_state: Mapping[str, torch.Tensor] | None = None,
        direct_init_label: str | None = None,
    ) -> None:
        super().__init__()
        _validate_method(method_id)
        if hidden_size <= 0 or vocabulary_size <= 0 or context_width <= 0:
            raise VisibilityDecoderError("decoder geometry must be positive")
        self.hidden_size = int(hidden_size)
        self.vocabulary_size = int(vocabulary_size)
        self.context_width = int(context_width)
        self.method_id = str(method_id)
        self.attention_score_mode = ATTENTION_SCORE_MODE
        self.qkv_seed = int(qkv_seed)

        if direct_state is None:
            # This branch is for model-free fixtures only.  Production callers
            # pass the frozen competent public affine state explicitly.
            W = torch.eye(self.hidden_size, dtype=torch.float32)
            b = torch.zeros(self.hidden_size, dtype=torch.float32)
            s = torch.tensor(3.0, dtype=torch.float32)
            self.direct_init_label = direct_init_label or "identity_fixture"
        else:
            validated = _validate_direct_state(direct_state, hidden_size=self.hidden_size)
            W, b, s = validated["W"], validated["b"], validated["s"]
            self.direct_init_label = direct_init_label or "competent_public_affine"
        self.W = nn.Parameter(W.clone())
        self.b = nn.Parameter(b.clone())
        self.s = nn.Parameter(s.clone())

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.qkv_seed)
        self.query = nn.Linear(self.hidden_size, self.context_width)
        self.key = nn.Linear(self.hidden_size, self.context_width)
        self.value = nn.Linear(self.hidden_size, self.context_width)
        self.output = nn.Linear(self.context_width, self.hidden_size)
        _deterministic_linear_init(self.query, generator)
        _deterministic_linear_init(self.key, generator)
        _deterministic_linear_init(self.value, generator)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()

    @property
    def visibility_mode(self) -> VisibilityMode:
        return method_visibility(self.method_id)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.s.float().exp()

    @property
    def parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

    @property
    def effective_parameter_count(self) -> int:
        count = self.parameter_count
        if self.visibility_mode == "positionwise":
            # With one key per valid query, softmax is exactly one and Q/K do
            # not affect the output or its gradient.
            count -= int(self.query.weight.numel() + self.query.bias.numel())
            count -= int(self.key.weight.numel() + self.key.bias.numel())
        return count

    def _check_inputs(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            activation.ndim != 3
            or int(activation.shape[0]) <= 0
            or int(activation.shape[1]) <= 1
            or int(activation.shape[2]) != self.hidden_size
        ):
            raise VisibilityDecoderError(
                "activation must be [records, positions>1, hidden_size]"
            )
        if not activation.dtype.is_floating_point:
            raise VisibilityDecoderError("activation must be floating point")
        if not torch.isfinite(activation).all().item():
            raise VisibilityDecoderError("activation contains non-finite values")
        if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(activation.shape[:2]):
            raise VisibilityDecoderError("valid mask geometry does not match activation")
        mask = _validate_valid_mask(valid_mask.to(device=activation.device))
        return mask

    def visibility_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """Return the actual mask used by this model's attention path."""

        return build_visibility_mask(self.method_id, valid_mask)

    def _added_path(
        self,
        activation: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        value = F.layer_norm(
            activation.float(),
            (self.hidden_size,),
            weight=None,
            bias=None,
            eps=1e-5,
        )
        query = self.query(value)
        key = self.key(value)
        projected_value = self.value(value)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        scores = (query @ key.transpose(-1, -2)) * COSINE_SCALE
        allowed = self.visibility_mask(mask)
        masked_scores = scores.masked_fill(~allowed, float("-inf"))
        has_key = allowed.any(dim=-1, keepdim=True)
        # A padded query has no allowed key.  Replacing its all--inf row with
        # zeros before softmax avoids NaNs; its output is zeroed below.
        safe_scores = torch.where(has_key, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(safe_scores, dim=-1)
        weights = torch.where(mask.unsqueeze(-1) & has_key, weights, torch.zeros_like(weights))
        attended = weights @ projected_value
        output = self.output(attended)
        return output * mask.unsqueeze(-1).to(dtype=output.dtype)

    def direct_pre_normalized_hidden(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the common direct affine path without activation context."""

        mask = self._check_inputs(activation, valid_mask)
        base = activation.float() @ self.W.float().T + self.b.float()
        return torch.where(mask.unsqueeze(-1), base, torch.zeros_like(base))

    def pre_normalized_hidden(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        base = activation.float() @ self.W.float().T + self.b.float()
        combined = base + self._added_path(activation, mask)
        return torch.where(mask.unsqueeze(-1), combined, torch.zeros_like(combined))

    def projected_hidden(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        projected = F.normalize(self.pre_normalized_hidden(activation, mask), dim=-1)
        return torch.where(mask.unsqueeze(-1), projected, torch.zeros_like(projected))

    def _validate_embedding_table(
        self,
        embedding_table: torch.Tensor,
        *,
        check_finite: bool,
    ) -> torch.Tensor:
        """Validate table geometry; scan immutable table bytes only when requested."""

        if (
            embedding_table.ndim != 2
            or tuple(embedding_table.shape) != (self.vocabulary_size, self.hidden_size)
            or not embedding_table.dtype.is_floating_point
        ):
            raise VisibilityDecoderError(
                "embedding table must have [vocabulary_size, hidden_size] floating geometry"
            )
        if check_finite and not torch.isfinite(embedding_table).all().item():
            raise VisibilityDecoderError("embedding table contains non-finite values")
        return embedding_table

    def validate_embedding_table(self, embedding_table: torch.Tensor) -> torch.Tensor:
        """Validate an immutable readout table once before chunked inference."""

        return self._validate_embedding_table(embedding_table, check_finite=True)

    def logits_from_rows(
        self,
        projected_hidden: torch.Tensor,
        record_slots: torch.Tensor,
        position_slots: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        if (
            projected_hidden.ndim != 3
            or int(projected_hidden.shape[-1]) != self.hidden_size
        ):
            raise VisibilityDecoderError("projected hidden geometry changed")
        if (
            record_slots.ndim != 1
            or position_slots.ndim != 1
            or tuple(record_slots.shape) != tuple(position_slots.shape)
        ):
            raise VisibilityDecoderError("draw indices must be equal-length vectors")
        # The caller validates the immutable embedding table once before a
        # chunked readout.  Keep this hot path to cheap geometry/dtype checks;
        # rescanning 1.05 GB of F32 values for every chunk would dominate the
        # decoder cost.
        table = self._validate_embedding_table(embedding_table, check_finite=False)
        rows = projected_hidden[
            record_slots.to(device=projected_hidden.device, dtype=torch.long),
            position_slots.to(device=projected_hidden.device, dtype=torch.long),
        ]
        logits = rows.float() @ table.to(device=rows.device, dtype=torch.float32).T
        result = logits * self.logit_scale
        if not torch.isfinite(result).all().item():
            raise VisibilityDecoderError("decoder logits are non-finite")
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
            raise VisibilityDecoderError("selected mask geometry changed")
        selected = selected_mask.to(device=activation.device, dtype=torch.bool)
        if (selected & ~mask).any().item() or selected[:, 0].any().item():
            raise VisibilityDecoderError("selected rows must be valid post-BOS positions")
        indices = torch.nonzero(selected, as_tuple=False)
        if int(indices.shape[0]) <= 0:
            raise VisibilityDecoderError("selected rows are empty")
        hidden = self.projected_hidden(activation, mask)
        return self.logits_from_rows(
            hidden, indices[:, 0], indices[:, 1], embedding_table
        )

    def forward(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        hidden = self.projected_hidden(activation, mask)
        table = self.validate_embedding_table(embedding_table)
        logits = hidden.float() @ table.to(device=hidden.device, dtype=torch.float32).T
        logits = logits * self.logit_scale
        return torch.where(mask.unsqueeze(-1), logits, torch.zeros_like(logits))


def build_visibility_decoder(
    method_id: str,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    qkv_seed: int = DEFAULT_QKV_SEED,
    direct_state: Mapping[str, torch.Tensor] | None = None,
    direct_init_label: str | None = None,
) -> VisibilityAffineAttentionDecoder:
    """Build one registered P06 arm.

    ``direct_state`` is required by the production fit runner and should come
    from :func:`load_direct_affine_initialization`.  The identity fallback is
    retained for compact unit fixtures only.
    """

    return VisibilityAffineAttentionDecoder(
        hidden_size,
        vocabulary_size,
        method_id,
        context_width=context_width,
        qkv_seed=qkv_seed,
        direct_state=direct_state,
        direct_init_label=direct_init_label,
    )


def deterministic_top1(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ascending-ID top-1 IDs and exact tie counts.

    ``torch.argmax`` returns the first maximal index along the vocabulary
    dimension, so equal-score ties resolve to the lowest token ID.  No
    vocabulary-sized cumsum or sort buffer is materialized.
    """

    if logits.ndim < 1 or not logits.dtype.is_floating_point:
        raise VisibilityDecoderError("logits must be a floating tensor with a vocabulary axis")
    if not torch.isfinite(logits).all().item():
        raise VisibilityDecoderError("logits contain non-finite values")
    values, ids = logits.max(dim=-1)
    ties = logits.eq(values.unsqueeze(-1)).sum(dim=-1).to(dtype=torch.int64)
    return ids.to(dtype=torch.long), ties


def _state_cpu(model: VisibilityAffineAttentionDecoder) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }


def save_visibility_state(
    path: Path,
    model: VisibilityAffineAttentionDecoder,
    *,
    selected_step: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save one create-only P06 state and return its immutable descriptor."""

    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise VisibilityDecoderError(f"decoder state is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _state_cpu(model)
    state_metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "method_id": model.method_id,
        "visibility_mode": model.visibility_mode,
        "attention_score_mode": model.attention_score_mode,
        "context_width": int(model.context_width),
        "hidden_size": int(model.hidden_size),
        "vocabulary_size": int(model.vocabulary_size),
        "qkv_init_seed": int(model.qkv_seed),
        "direct_init_label": str(model.direct_init_label),
        "selected_step": int(selected_step),
        "parameter_count": int(model.parameter_count),
        "effective_parameter_count": int(model.effective_parameter_count),
    }
    if metadata:
        state_metadata.update({str(key): value for key, value in metadata.items()})
    save_file(
        state,
        str(path),
        metadata={key: str(value) for key, value in state_metadata.items()},
    )
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "state_sha256": state_sha256(state),
        "tensor_sha256": {key: tensor_sha256(value) for key, value in state.items()},
        "state_bytes": sum(int(value.numel()) * value.element_size() for value in state.values()),
        "selected_step": int(selected_step),
        "metadata": state_metadata,
    }


def load_visibility_state(
    path: Path,
    *,
    method_id: str,
    hidden_size: int,
    vocabulary_size: int,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    expected_sha256: str | None = None,
) -> VisibilityAffineAttentionDecoder:
    """Load and metadata-bind one frozen P06 decoder state."""

    _validate_method(method_id)
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise VisibilityDecoderError(f"decoder state is unavailable: {path}")
    if expected_sha256 is not None:
        observed = file_sha256(path)
        if observed != expected_sha256:
            raise VisibilityDecoderError(
                f"decoder state hash changed: expected {expected_sha256}, observed {observed}"
            )
    try:
        state = load_file(str(path), device="cpu")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise VisibilityDecoderError(f"cannot load decoder state: {path}") from exc
    if metadata.get("schema") not in (SCHEMA, None):
        raise VisibilityDecoderError(f"unsupported P06 decoder schema: {metadata.get('schema')}")
    if metadata.get("method_id", method_id) != method_id:
        raise VisibilityDecoderError("decoder state method ID does not match requested method")
    if metadata.get("visibility_mode", method_visibility(method_id)) != method_visibility(method_id):
        raise VisibilityDecoderError("decoder state visibility mode does not match method")
    if metadata.get("attention_score_mode", ATTENTION_SCORE_MODE) != ATTENTION_SCORE_MODE:
        raise VisibilityDecoderError("decoder state does not use the common cosine/QK normalization")
    stored_hidden = metadata.get("hidden_size")
    stored_vocab = metadata.get("vocabulary_size")
    if stored_hidden is not None and int(stored_hidden) != int(hidden_size):
        raise VisibilityDecoderError("decoder state hidden size differs from requested geometry")
    if stored_vocab is not None and int(stored_vocab) != int(vocabulary_size):
        raise VisibilityDecoderError("decoder state vocabulary size differs from requested geometry")
    stored_width = metadata.get("context_width")
    if stored_width is not None and int(stored_width) != int(context_width):
        raise VisibilityDecoderError("decoder state context width differs from requested geometry")
    qkv_seed = int(metadata.get("qkv_init_seed", DEFAULT_QKV_SEED))
    model = build_visibility_decoder(
        method_id,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
        qkv_seed=qkv_seed,
        direct_init_label=str(metadata.get("direct_init_label", "loaded_state")),
    )
    if set(state) != set(model.state_dict()):
        raise VisibilityDecoderError(f"decoder state tensor keys differ for {method_id}")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise VisibilityDecoderError(f"decoder state tensor geometry differs for {method_id}") from exc
    return model


__all__ = [
    "ATTENTION_SCORE_MODE",
    "BOS_TOKEN_ID",
    "COSINE_SCALE",
    "DEFAULT_CONTEXT_WIDTH",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_QKV_SEED",
    "DEFAULT_VOCABULARY_SIZE",
    "FULL_RECORD_METHOD",
    "METHODS",
    "PAD_TOKEN_ID",
    "PAST_ONLY_METHOD",
    "POSITIONWISE_METHOD",
    "SCHEMA",
    "TASK_ID",
    "VisibilityAffineAttentionDecoder",
    "VisibilityDecoderError",
    "build_visibility_decoder",
    "build_visibility_mask",
    "deterministic_top1",
    "file_sha256",
    "future_valid_count",
    "load_direct_affine_initialization",
    "load_visibility_state",
    "method_visibility",
    "save_visibility_state",
    "state_sha256",
    "tensor_sha256",
]
