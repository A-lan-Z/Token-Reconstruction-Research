"""TRR-0009 anchored supported-token readout adaptation.

The adapter wraps the published TRR-0007 residual MLP512 decoder.  It adds a
small, explicitly token-specific calibration to rows that occur in the public
fit bank: a bounded row gain and bias, both initialized at zero correction.
Absent vocabulary IDs remain exactly tied to the public normalized embedding.
The full vocabulary is still scored for every row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from token_reconstruction.trr0005_joint_decoder import file_sha256
from token_reconstruction.trr0007_positionwise import (
    DEFAULT_BOTTLENECK_SIZE,
    RESIDUAL_MLP_METHOD_ID,
    ResidualMLPPositionwiseDecoder,
    load_positionwise_model_state,
)


TASK_ID = "TRR-0009"
SCHEMA = "token-reconstruction.trr0009-supported-readout.v1"
METHOD_ID = "trr0009_supported_row_readout"
DEFAULT_GAIN_LIMIT = 0.25
DEFAULT_BIAS_LIMIT = 0.25
DEFAULT_ANCHOR_STRENGTH = 1.0e-4
BOS_TOKEN_ID = 128000


class TRR0009ModelError(ValueError):
    """Raised when the TRR-0009 readout contract is violated."""


def _tensor_digest(value: torch.Tensor, *, prefix: bytes = b"trr0009-tensor-v1\0") -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(prefix)
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def support_digest(support_ids: torch.Tensor, support_counts: torch.Tensor) -> str:
    """Digest the exact sorted support/count binding."""

    ids = _validate_support(support_ids, support_counts, vocabulary_size=None)[0]
    counts = support_counts.detach().cpu().to(dtype=torch.int64).contiguous()
    digest = hashlib.sha256(b"trr0009-support-v1\0")
    digest.update(_tensor_digest(ids, prefix=b"trr0009-support-ids\0").encode("ascii"))
    digest.update(_tensor_digest(counts, prefix=b"trr0009-support-counts\0").encode("ascii"))
    return digest.hexdigest()


def _validate_support(
    support_ids: torch.Tensor | Sequence[int],
    support_counts: torch.Tensor | Sequence[int],
    *,
    vocabulary_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.as_tensor(support_ids, dtype=torch.long, device="cpu").flatten().contiguous()
    counts = torch.as_tensor(support_counts, dtype=torch.long, device="cpu").flatten().contiguous()
    if ids.ndim != 1 or counts.ndim != 1 or tuple(ids.shape) != tuple(counts.shape):
        raise TRR0009ModelError("support IDs and counts must be equal-length vectors")
    if int(ids.numel()) <= 0:
        raise TRR0009ModelError("supported-token set must be non-empty")
    if (ids < 0).any().item() or (counts <= 0).any().item():
        raise TRR0009ModelError("support IDs must be nonnegative and counts positive")
    if int(torch.unique(ids).numel()) != int(ids.numel()):
        raise TRR0009ModelError("support IDs must be unique")
    if not torch.equal(ids, torch.sort(ids).values):
        raise TRR0009ModelError("support IDs must be sorted ascending")
    if vocabulary_size is not None and (ids >= int(vocabulary_size)).any().item():
        raise TRR0009ModelError("support ID exceeds vocabulary size")
    return ids, counts


def build_support_from_counts(
    counts: torch.Tensor | Sequence[int], *, vocabulary_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted positive-count IDs and counts from a vocabulary vector."""

    count_tensor = torch.as_tensor(counts, dtype=torch.long, device="cpu").flatten().contiguous()
    if tuple(count_tensor.shape) != (int(vocabulary_size),):
        raise TRR0009ModelError("frequency vector geometry differs from vocabulary")
    ids = torch.nonzero(count_tensor > 0, as_tuple=False).flatten().to(dtype=torch.long)
    values = count_tensor.index_select(0, ids)
    return _validate_support(ids, values, vocabulary_size=int(vocabulary_size))


class SupportedTokenReadout(nn.Module):
    """TRR-0007 residual decoder plus bounded supported-row calibration.

    ``base`` is retained as a child module so continued fixed-readout and
    adaptable arms share exactly the same starting parameter state.  The
    adapter never changes the public embedding tensor in-place.
    """

    method_id = METHOD_ID

    def __init__(
        self,
        base: ResidualMLPPositionwiseDecoder,
        support_ids: torch.Tensor | Sequence[int],
        support_counts: torch.Tensor | Sequence[int],
        *,
        gain_limit: float = DEFAULT_GAIN_LIMIT,
        bias_limit: float = DEFAULT_BIAS_LIMIT,
        anchor_strength: float = DEFAULT_ANCHOR_STRENGTH,
    ) -> None:
        super().__init__()
        if not isinstance(base, ResidualMLPPositionwiseDecoder):
            raise TRR0009ModelError("TRR-0009 requires the published residual MLP512 base")
        if gain_limit <= 0 or bias_limit <= 0 or anchor_strength < 0:
            raise TRR0009ModelError("readout limits must be positive and anchor nonnegative")
        ids, counts = _validate_support(
            support_ids,
            support_counts,
            vocabulary_size=int(base.vocabulary_size),
        )
        self.base = base
        self.hidden_size = int(base.hidden_size)
        self.vocabulary_size = int(base.vocabulary_size)
        self.context_width = int(base.context_width)
        self.bottleneck_size = int(base.bottleneck_size)
        self.attention_score_mode = str(base.attention_score_mode)
        self.base_method_id = str(base.base_method_id)
        self.gain_limit = float(gain_limit)
        self.bias_limit = float(bias_limit)
        self.anchor_strength = float(anchor_strength)
        self.register_buffer("support_ids", ids, persistent=True)
        self.register_buffer("support_counts", counts, persistent=True)
        weights = 1.0 + 4.0 / torch.sqrt(counts.to(dtype=torch.float32))
        self.register_buffer("anchor_weights", weights, persistent=True)
        self.raw_gain = nn.Parameter(torch.zeros(int(ids.numel()), dtype=torch.float32))
        self.raw_bias = nn.Parameter(torch.zeros(int(ids.numel()), dtype=torch.float32))

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.base.logit_scale

    @property
    def attention_mode(self) -> str:
        return self.base.attention_mode

    @property
    def parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters() if value.requires_grad)

    @property
    def readout_parameter_count(self) -> int:
        return int(self.raw_gain.numel() + self.raw_bias.numel())

    @property
    def support_count(self) -> int:
        return int(self.support_ids.numel())

    def effective_gain(self) -> torch.Tensor:
        return 1.0 + self.gain_limit * torch.tanh(self.raw_gain)

    def effective_bias(self) -> torch.Tensor:
        return self.bias_limit * torch.tanh(self.raw_bias)

    def readout_penalty(self) -> torch.Tensor:
        """Frequency-weighted zero-anchor penalty for the adaptable rows."""

        if self.anchor_strength == 0.0:
            return (self.raw_gain.sum() + self.raw_bias.sum()) * 0.0
        return self.anchor_strength * (
            self.anchor_weights
            * (self.raw_gain.square() + self.raw_bias.square())
        ).mean()

    def support_binding(self) -> dict[str, Any]:
        return {
            "support_count": self.support_count,
            "support_ids_sha256": _tensor_digest(self.support_ids, prefix=b"trr0009-support-ids\0"),
            "support_counts_sha256": _tensor_digest(self.support_counts, prefix=b"trr0009-support-counts\0"),
            "support_digest": support_digest(self.support_ids, self.support_counts),
            "frequency_min": int(self.support_counts.min().item()),
            "frequency_max": int(self.support_counts.max().item()),
        }

    def _check_embedding(self, embedding_table: torch.Tensor) -> None:
        # Asset finiteness and the immutable file hash are checked once by the
        # trainer/evaluator loader.  Keep the hot path to geometry/dtype checks
        # so adapter calls have the same validation workload as the published
        # decoder rather than rescanning the 1.05 GB E table every draw.
        if embedding_table.ndim != 2 or not embedding_table.dtype.is_floating_point:
            raise TRR0009ModelError("embedding table must be a floating matrix")
        if tuple(embedding_table.shape) != (self.vocabulary_size, self.hidden_size):
            raise TRR0009ModelError("embedding table geometry changed")

    def _adapt_rows(self, base_logits: torch.Tensor, *, valid: torch.Tensor | None) -> torch.Tensor:
        if base_logits.ndim < 2 or int(base_logits.shape[-1]) != self.vocabulary_size:
            raise TRR0009ModelError("base logits geometry changed")
        ids = self.support_ids.to(device=base_logits.device)
        selected = base_logits.index_select(-1, ids)
        gain = self.effective_gain().to(device=base_logits.device, dtype=base_logits.dtype)
        bias = self.effective_bias().to(device=base_logits.device, dtype=base_logits.dtype)
        corrected = selected * gain + bias
        if valid is not None:
            valid = valid.to(device=base_logits.device, dtype=torch.bool)
            while valid.ndim < corrected.ndim:
                valid = valid.unsqueeze(-1)
            corrected = torch.where(valid, corrected, selected)
        return base_logits.scatter(-1, ids.expand(*base_logits.shape[:-1], -1), corrected)

    def _check_inputs(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.base._check_inputs(activation, valid_mask)  # noqa: SLF001

    def pre_normalized_hidden(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.base.pre_normalized_hidden(activation, valid_mask)

    def projected_hidden(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.base.projected_hidden(activation, valid_mask)

    def logits_from_rows(
        self,
        projected_hidden: torch.Tensor,
        record_slots: torch.Tensor,
        position_slots: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        self._check_embedding(embedding_table)
        base_logits = self.base.logits_from_rows(
            projected_hidden,
            record_slots,
            position_slots,
            embedding_table,
        )
        return self._adapt_rows(base_logits, valid=None)

    def selected_logits(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        selected_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        if selected_mask.ndim != 2 or tuple(selected_mask.shape) != tuple(mask.shape):
            raise TRR0009ModelError("selected mask geometry changed")
        selected = selected_mask.to(device=activation.device, dtype=torch.bool)
        if (selected & ~mask).any().item() or selected[:, 0].any().item():
            raise TRR0009ModelError("selected rows must be valid post-BOS positions")
        indices = torch.nonzero(selected, as_tuple=False)
        if int(indices.shape[0]) <= 0:
            raise TRR0009ModelError("selected rows are empty")
        hidden = self.projected_hidden(activation, mask)
        return self.logits_from_rows(hidden, indices[:, 0], indices[:, 1], embedding_table)

    def forward(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        self._check_embedding(embedding_table)
        base_logits = self.base(activation, mask, embedding_table)
        return self._adapt_rows(base_logits, valid=mask)

    def materialize_effective_readout(
        self, embedding_table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the deployed dictionary and full vocabulary bias.

        This intentionally allocates a full `[vocabulary, hidden]` copy so its
        deployment preparation cost is visible in receipts.
        """

        self._check_embedding(embedding_table)
        started = torch.cuda.Event(enable_timing=True) if embedding_table.is_cuda else None
        if started is not None:
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
        effective = embedding_table.detach().clone()
        ids = self.support_ids.to(device=effective.device)
        gain = self.effective_gain().to(device=effective.device, dtype=effective.dtype)
        effective_rows = effective.index_select(0, ids) * gain.unsqueeze(-1)
        effective = effective.index_copy(0, ids, effective_rows)
        bias = torch.zeros(self.vocabulary_size, dtype=effective.dtype, device=effective.device)
        row_bias = self.effective_bias().to(device=bias.device, dtype=bias.dtype)
        bias = bias.index_copy(0, ids, row_bias)
        if started is not None:
            ended.record()
            torch.cuda.synchronize(embedding_table.device)
        return effective, bias


def build_adaptable_from_base(
    base: ResidualMLPPositionwiseDecoder,
    support_ids: torch.Tensor | Sequence[int],
    support_counts: torch.Tensor | Sequence[int],
    *,
    gain_limit: float = DEFAULT_GAIN_LIMIT,
    bias_limit: float = DEFAULT_BIAS_LIMIT,
    anchor_strength: float = DEFAULT_ANCHOR_STRENGTH,
) -> SupportedTokenReadout:
    return SupportedTokenReadout(
        base,
        support_ids,
        support_counts,
        gain_limit=gain_limit,
        bias_limit=bias_limit,
        anchor_strength=anchor_strength,
    )


def load_published_residual_state(
    path: Path,
    *,
    hidden_size: int,
    vocabulary_size: int,
    context_width: int,
    bottleneck_size: int = DEFAULT_BOTTLENECK_SIZE,
) -> ResidualMLPPositionwiseDecoder:
    model = load_positionwise_model_state(
        path,
        method_id=RESIDUAL_MLP_METHOD_ID,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
        bottleneck_size=bottleneck_size,
    )
    if not isinstance(model, ResidualMLPPositionwiseDecoder):
        raise TRR0009ModelError("published state did not load as residual MLP512")
    return model


def save_supported_readout_state(
    path: Path,
    model: SupportedTokenReadout,
    *,
    selected_step: int,
    starting_state: Mapping[str, Any],
    fit_manifest: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TRR0009ModelError(f"TRR-0009 state is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    state_metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "method_id": METHOD_ID,
        "base_method_id": RESIDUAL_MLP_METHOD_ID,
        "hidden_size": model.hidden_size,
        "vocabulary_size": model.vocabulary_size,
        "context_width": model.context_width,
        "bottleneck_size": model.bottleneck_size,
        "selected_step": int(selected_step),
        "gain_limit": model.gain_limit,
        "bias_limit": model.bias_limit,
        "anchor_strength": model.anchor_strength,
        "support_digest": model.support_binding()["support_digest"],
        "starting_state_sha256": starting_state.get("sha256"),
        "fit_manifest_sha256": fit_manifest.get("sha256"),
        "current_H_only": True,
        "full_vocabulary_cross_entropy": True,
        "inference_contract": "current_activation_H_i_only; supported-row calibrated full-vocabulary E",
    }
    if metadata:
        state_metadata.update({str(key): value for key, value in metadata.items()})
    save_file(state, str(path), metadata={key: str(value) for key, value in state_metadata.items()})
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "state_tensor_digest": _tensor_digest(
            torch.cat([value.reshape(-1).to(dtype=torch.float32) for _, value in sorted(state.items())])
        ),
        "state_bytes": sum(int(value.numel()) * value.element_size() for value in state.values()),
        "tensor_keys": sorted(state),
        "selected_step": int(selected_step),
        "metadata": state_metadata,
    }


def load_supported_readout_state(
    path: Path,
    *,
    base_state_path: Path,
    support_ids: torch.Tensor | Sequence[int],
    support_counts: torch.Tensor | Sequence[int],
    hidden_size: int,
    vocabulary_size: int,
    context_width: int,
    bottleneck_size: int = DEFAULT_BOTTLENECK_SIZE,
) -> SupportedTokenReadout:
    """Load an adaptable state while rebinding the published base explicitly."""

    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TRR0009ModelError(f"TRR-0009 state must be a regular file: {path}")
    try:
        state = load_file(str(path), device="cpu")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise TRR0009ModelError(f"cannot load TRR-0009 state: {path}") from exc
    if metadata.get("schema") != SCHEMA or metadata.get("method_id") != METHOD_ID:
        raise TRR0009ModelError("TRR-0009 state schema or method ID differs")
    ids, counts = _validate_support(support_ids, support_counts, vocabulary_size=vocabulary_size)
    expected_digest = support_digest(ids, counts)
    if metadata.get("support_digest") != expected_digest:
        raise TRR0009ModelError("TRR-0009 state support binding differs")
    base = load_published_residual_state(
        base_state_path,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
        bottleneck_size=bottleneck_size,
    )
    expected_base_sha = metadata.get("starting_state_sha256")
    if expected_base_sha and file_sha256(base_state_path.expanduser().resolve()) != expected_base_sha:
        raise TRR0009ModelError("adaptable state starting-base binding differs")
    model = build_adaptable_from_base(
        base,
        ids,
        counts,
        gain_limit=float(metadata.get("gain_limit", DEFAULT_GAIN_LIMIT)),
        bias_limit=float(metadata.get("bias_limit", DEFAULT_BIAS_LIMIT)),
        anchor_strength=float(metadata.get("anchor_strength", DEFAULT_ANCHOR_STRENGTH)),
    )
    expected = set(model.state_dict())
    if set(state) != expected:
        raise TRR0009ModelError("TRR-0009 state tensor keys differ")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise TRR0009ModelError("TRR-0009 state geometry differs") from exc
    return model


def zero_correction_equivalence(
    base: ResidualMLPPositionwiseDecoder,
    adaptable: SupportedTokenReadout,
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    max_rows: int = 512,
) -> dict[str, Any]:
    """Check exact selected-row and full-logit equality at initialization."""

    if not torch.equal(adaptable.raw_gain.detach(), torch.zeros_like(adaptable.raw_gain)):
        raise TRR0009ModelError("adaptable gain is not zero initialized")
    if not torch.equal(adaptable.raw_bias.detach(), torch.zeros_like(adaptable.raw_bias)):
        raise TRR0009ModelError("adaptable bias is not zero initialized")
    with torch.inference_mode():
        mask = valid_mask.to(device=activation.device, dtype=torch.bool)
        base_hidden = base.projected_hidden(activation, mask)
        adapt_hidden = adaptable.projected_hidden(activation, mask)
        indices = torch.nonzero(mask, as_tuple=False)
        indices = indices[indices[:, 1] > 0][: int(max_rows)]
        if int(indices.shape[0]) <= 0:
            raise TRR0009ModelError("equivalence fixture has no post-BOS rows")
        base_rows = base.logits_from_rows(
            base_hidden, indices[:, 0], indices[:, 1], embedding_table
        )
        adapt_rows = adaptable.logits_from_rows(
            adapt_hidden, indices[:, 0], indices[:, 1], embedding_table
        )
        base_full = base(activation, mask, embedding_table)
        adapt_full = adaptable(activation, mask, embedding_table)
    return {
        "rows_checked": int(indices.shape[0]),
        "projected_hidden_exact": bool(torch.equal(base_hidden, adapt_hidden)),
        "selected_logits_exact": bool(torch.equal(base_rows, adapt_rows)),
        "full_logits_exact": bool(torch.equal(base_full, adapt_full)),
        "max_hidden_abs_delta": float((base_hidden - adapt_hidden).abs().max().cpu()),
        "max_selected_logits_abs_delta": float((base_rows - adapt_rows).abs().max().cpu()),
        "max_full_logits_abs_delta": float((base_full - adapt_full).abs().max().cpu()),
        "support_binding": adaptable.support_binding(),
    }
