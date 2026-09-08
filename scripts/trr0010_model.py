"""TRR-0010 token-specific directional readout.

This module wraps the published TRR-0007 current-H residual decoder and adds
one trainable additive direction per fitting-bank-supported vocabulary row.
The correction starts at zero, so the initialized model has the same logits as
the fixed decoder.  Full-vocabulary cross-entropy remains in force: the sparse
correction only changes the supported columns of the full public-E score.
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
    build_residual_mlp512,
    load_positionwise_model_state,
)


TASK_ID = "TRR-0010"
SCHEMA = "token-reconstruction.trr0010-directional-readout.v1"
METHOD_ID = "trr0010_supported_token_directional_readout"
DEFAULT_ANCHOR_STRENGTH = 1.0e-4
DEFAULT_EMBEDDING_EPSILON = 1.0e-8
DEFAULT_DIRECTIONAL_LEARNING_RATE = 1.0e-4


class TRR0010ModelError(ValueError):
    """Raised when the TRR-0010 directional-readout contract is violated."""


def _tensor_digest(value: torch.Tensor, *, prefix: bytes) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(prefix)
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_support(
    support_ids: torch.Tensor | Sequence[int],
    support_counts: torch.Tensor | Sequence[int],
    *,
    vocabulary_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids_raw = torch.as_tensor(support_ids, device="cpu")
    counts_raw = torch.as_tensor(support_counts, device="cpu")
    if ids_raw.ndim != 1 or counts_raw.ndim != 1:
        raise TRR0010ModelError("support IDs and counts must be rank-1 vectors")
    if ids_raw.dtype == torch.bool or counts_raw.dtype == torch.bool:
        raise TRR0010ModelError("support IDs and counts must be integer vectors")
    if ids_raw.is_floating_point() or counts_raw.is_floating_point():
        raise TRR0010ModelError("support IDs and counts must be integer vectors")
    ids = ids_raw.to(dtype=torch.long).contiguous()
    counts = counts_raw.to(dtype=torch.long).contiguous()
    if tuple(ids.shape) != tuple(counts.shape):
        raise TRR0010ModelError("support IDs and counts must be equal-length vectors")
    if int(ids.numel()) <= 0:
        raise TRR0010ModelError("supported-token set must be non-empty")
    if (ids < 0).any().item() or (counts <= 0).any().item():
        raise TRR0010ModelError("support IDs must be nonnegative and counts positive")
    if int(torch.unique(ids).numel()) != int(ids.numel()):
        raise TRR0010ModelError("support IDs must be unique")
    if not torch.equal(ids, torch.sort(ids).values):
        raise TRR0010ModelError("support IDs must be sorted ascending")
    if vocabulary_size is not None and (ids >= int(vocabulary_size)).any().item():
        raise TRR0010ModelError("support ID exceeds vocabulary size")
    return ids, counts


def support_digest(
    support_ids: torch.Tensor | Sequence[int],
    support_counts: torch.Tensor | Sequence[int],
) -> str:
    """Digest the exact sorted supported-ID/count binding."""

    ids, counts = _validate_support(support_ids, support_counts, vocabulary_size=None)
    digest = hashlib.sha256(b"trr0010-support-v1\0")
    digest.update(_tensor_digest(ids, prefix=b"trr0010-support-ids\0").encode("ascii"))
    digest.update(_tensor_digest(counts, prefix=b"trr0010-support-counts\0").encode("ascii"))
    return digest.hexdigest()


def directional_memory_estimate(
    support_count: int,
    hidden_size: int = 2048,
    *,
    vocabulary_size: int = 128256,
    position_budget: int = 512,
    base_peak_reserved_bytes: int | None = None,
) -> dict[str, int | float | None]:
    """Return transparent FP32 parameter/optimizer/workspace estimates.

    Adam's two FP32 moments and one FP32 gradient are counted separately from
    the parameter table.  The correction workspace is the dense score update
    for the sampled position rows and supported vocabulary columns.
    """

    if support_count <= 0 or hidden_size <= 0 or vocabulary_size <= 0 or position_budget <= 0:
        raise TRR0010ModelError("directional memory dimensions must be positive")
    parameter_bytes = int(support_count) * int(hidden_size) * 4
    adam_moment_bytes = parameter_bytes * 2
    gradient_bytes = parameter_bytes
    diagnostic_workspace_bytes = int(position_budget) * int(support_count) * 4
    materialized_embedding_bytes = int(vocabulary_size) * int(hidden_size) * 4
    support_row_gather_bytes = parameter_bytes
    support_row_add_bytes = parameter_bytes
    primary_materialization_support_bytes = support_row_gather_bytes + support_row_add_bytes
    total_parameter_gradient_adam = parameter_bytes + adam_moment_bytes + gradient_bytes
    primary_dense_bytes = materialized_embedding_bytes * 2
    result: dict[str, int | float | None] = {
        "support_count": int(support_count),
        "hidden_size": int(hidden_size),
        "vocabulary_size": int(vocabulary_size),
        "position_budget": int(position_budget),
        "parameter_count": int(support_count) * int(hidden_size),
        "parameter_bytes_fp32": parameter_bytes,
        "adam_moment_bytes_fp32": adam_moment_bytes,
        "gradient_bytes_fp32": gradient_bytes,
        "parameter_gradient_adam_bytes": total_parameter_gradient_adam,
        "diagnostic_sparse_workspace_bytes_fp32": diagnostic_workspace_bytes,
        "materialized_effective_embedding_bytes_fp32": materialized_embedding_bytes,
        "materialized_effective_embedding_gradient_bytes_fp32": materialized_embedding_bytes,
        "primary_support_row_gather_bytes_fp32": support_row_gather_bytes,
        "primary_support_row_add_bytes_fp32": support_row_add_bytes,
        "primary_materialization_support_temporary_bytes_fp32": primary_materialization_support_bytes,
        "primary_dense_effective_and_gradient_bytes_fp32": primary_dense_bytes,
        "optimizer_transient_policy": "foreach_false_or_equivalent_bounded_scratch",
        "base_peak_reserved_bytes": None if base_peak_reserved_bytes is None else int(base_peak_reserved_bytes),
        "estimated_peak_sparse_diagnostic_bytes": (
            None
            if base_peak_reserved_bytes is None
            else int(base_peak_reserved_bytes) + total_parameter_gradient_adam + diagnostic_workspace_bytes
        ),
        "estimated_peak_merged_primary_bytes": (
            None
            if base_peak_reserved_bytes is None
            else int(base_peak_reserved_bytes)
            + total_parameter_gradient_adam
            + primary_materialization_support_bytes
            + primary_dense_bytes
        ),
        # Backward-compatible key retained for setup consumers; it now means
        # the merged-primary bound rather than the old sparse estimate.
        "estimated_peak_plus_directional_bytes": (
            None
            if base_peak_reserved_bytes is None
            else int(base_peak_reserved_bytes)
            + total_parameter_gradient_adam
            + primary_materialization_support_bytes
            + primary_dense_bytes
        ),
    }
    if result["estimated_peak_plus_directional_bytes"] is not None:
        result["estimated_peak_merged_primary_gib"] = float(
            int(result["estimated_peak_merged_primary_bytes"]) / (1024**3)
        )
        result["estimated_peak_plus_directional_gib"] = float(
            int(result["estimated_peak_plus_directional_bytes"]) / (1024**3)
        )
        result["estimated_peak_sparse_diagnostic_gib"] = float(
            int(result["estimated_peak_sparse_diagnostic_bytes"]) / (1024**3)
        )
    return result


class DirectionalTokenReadout(nn.Module):
    """Residual decoder plus a sparse token-specific direction table.

    ``delta_rows[j]`` is added to public embedding row ``support_ids[j]``.
    The public embedding tensor remains immutable and absent rows receive no
    parameter.  The class exposes full-vocabulary logits and a row-level hook
    for the shared TRR-0010 trainer.
    """

    method_id = METHOD_ID
    readout_mode = "supported_token_directional"

    def __init__(
        self,
        base: ResidualMLPPositionwiseDecoder,
        support_ids: torch.Tensor | Sequence[int],
        support_counts: torch.Tensor | Sequence[int],
        *,
        anchor_strength: float = DEFAULT_ANCHOR_STRENGTH,
        embedding_epsilon: float = DEFAULT_EMBEDDING_EPSILON,
    ) -> None:
        super().__init__()
        if not isinstance(base, ResidualMLPPositionwiseDecoder):
            raise TRR0010ModelError("TRR-0010 requires the published residual MLP512 base")
        if anchor_strength < 0 or embedding_epsilon <= 0:
            raise TRR0010ModelError("anchor strength must be nonnegative and epsilon positive")
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
        self.anchor_strength = float(anchor_strength)
        self.embedding_epsilon = float(embedding_epsilon)
        self.register_buffer("support_ids", ids, persistent=True)
        self.register_buffer("support_counts", counts, persistent=True)
        anchor_weights = 1.0 + 4.0 / torch.sqrt(counts.to(dtype=torch.float32))
        self.register_buffer("anchor_weights", anchor_weights, persistent=True)
        # Zero initialization is the decision-equivalent starting point.
        self.delta_rows = nn.Parameter(torch.zeros(int(ids.numel()), self.hidden_size, dtype=torch.float32))
        # Bound once against the immutable public E before training.  It is a
        # non-persistent buffer so the large public asset is never serialized
        # into the adapter state.
        self.register_buffer("embedding_row_rms", torch.empty(0, dtype=torch.float32), persistent=False)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.base.logit_scale

    @property
    def support_count(self) -> int:
        return int(self.support_ids.numel())

    @property
    def directional_parameter_count(self) -> int:
        return int(self.delta_rows.numel())

    @property
    def parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters() if value.requires_grad)

    def support_binding(self) -> dict[str, Any]:
        return {
            "support_count": self.support_count,
            "support_ids_sha256": _tensor_digest(self.support_ids, prefix=b"trr0010-support-ids\0"),
            "support_counts_sha256": _tensor_digest(self.support_counts, prefix=b"trr0010-support-counts\0"),
            "support_digest": support_digest(self.support_ids, self.support_counts),
            "frequency_min": int(self.support_counts.min().item()),
            "frequency_max": int(self.support_counts.max().item()),
        }

    def _check_embedding(self, embedding_table: torch.Tensor) -> None:
        if embedding_table.ndim != 2 or not embedding_table.dtype.is_floating_point:
            raise TRR0010ModelError("embedding table must be a floating matrix")
        if tuple(embedding_table.shape) != (self.vocabulary_size, self.hidden_size):
            raise TRR0010ModelError("embedding table geometry changed")

    def bind_embedding_statistics(self, embedding_table: torch.Tensor) -> dict[str, Any]:
        """Bind row RMS values once from the immutable public embedding."""

        self._check_embedding(embedding_table)
        with torch.no_grad():
            rows = embedding_table.detach().index_select(0, self.support_ids.to(embedding_table.device))
            rms = rows.float().square().mean(dim=1).sqrt().clamp_min(self.embedding_epsilon)
        self.embedding_row_rms = rms.to(device=self.delta_rows.device, dtype=torch.float32).contiguous()
        return {
            "support_count": self.support_count,
            "embedding_row_rms_sha256": _tensor_digest(
                self.embedding_row_rms, prefix=b"trr0010-embedding-row-rms\0"
            ),
            "embedding_epsilon": self.embedding_epsilon,
        }

    def _require_embedding_statistics(self) -> torch.Tensor:
        if tuple(self.embedding_row_rms.shape) != (self.support_count,):
            raise TRR0010ModelError("bind_embedding_statistics must run before the anchor penalty")
        return self.embedding_row_rms

    def readout_penalty(self) -> torch.Tensor:
        """Frequency-weighted relative L2 anchor to the public E rows."""

        if self.anchor_strength == 0.0:
            return self.delta_rows.sum() * 0.0
        rms = self._require_embedding_statistics().to(device=self.delta_rows.device)
        denominator = self.hidden_size * rms.square() + self.embedding_epsilon
        relative_row_norm = self.delta_rows.float().square().sum(dim=1) / denominator
        return self.anchor_strength * (
            self.anchor_weights.to(relative_row_norm.device) * relative_row_norm
        ).mean()

    def _check_projected_hidden(self, projected_hidden: torch.Tensor) -> None:
        if projected_hidden.ndim != 3 or int(projected_hidden.shape[-1]) != self.hidden_size:
            raise TRR0010ModelError("projected hidden geometry changed")

    def pre_normalized_hidden(
        self, activation: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Expose the unchanged current-H decoder hidden hook."""

        return self.base.pre_normalized_hidden(activation, valid_mask)

    def projected_hidden(
        self, activation: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return the decoder's normalized current-position rows ``q``."""

        return self.base.projected_hidden(activation, valid_mask)

    def _delta_scores(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.ndim < 2 or int(rows.shape[-1]) != self.hidden_size:
            raise TRR0010ModelError("directional rows have incorrect geometry")
        delta = rows.float() @ self.delta_rows.to(device=rows.device, dtype=torch.float32).transpose(0, 1)
        return delta * self.logit_scale.to(device=rows.device, dtype=torch.float32)

    def _apply_delta(self, base_logits: torch.Tensor, delta_scores: torch.Tensor) -> torch.Tensor:
        if base_logits.ndim < 2 or int(base_logits.shape[-1]) != self.vocabulary_size:
            raise TRR0010ModelError("base logits geometry changed")
        if tuple(delta_scores.shape[:-1]) != tuple(base_logits.shape[:-1]):
            raise TRR0010ModelError("directional score prefix geometry changed")
        if int(delta_scores.shape[-1]) != self.support_count:
            raise TRR0010ModelError("directional score support geometry changed")
        ids = self.support_ids.to(device=base_logits.device)
        view = (1,) * (base_logits.ndim - 1) + (self.support_count,)
        index = ids.view(view).expand(*base_logits.shape[:-1], self.support_count)
        return base_logits.scatter_add(-1, index, delta_scores.to(dtype=base_logits.dtype))

    def score_rows(
        self,
        query_rows: torch.Tensor,
        logit_scale: torch.Tensor | float,
        embedding: torch.Tensor,
        *,
        base_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Primary full-vocabulary score through materialized effective E.

        The common runner uses this same hook for fit, validation, and
        registered inference.  ``base_logits`` is accepted for protocol
        compatibility and geometry validation, but the registered path uses
        one merged E matmul so its semantics match the exported deployment
        dictionary exactly.
        """

        if query_rows.ndim != 2 or int(query_rows.shape[-1]) != self.hidden_size:
            raise TRR0010ModelError("query rows must be [rows, hidden]")
        self._check_embedding(embedding)
        if base_logits is not None:
            if base_logits.ndim != 2 or tuple(base_logits.shape) != (query_rows.shape[0], self.vocabulary_size):
                raise TRR0010ModelError("precomputed base logits geometry changed")
            if not torch.isfinite(base_logits).all().item():
                raise TRR0010ModelError("precomputed base logits are non-finite")
        effective_embedding = self.materialize_effective_embedding(embedding)
        return self.merged_score_rows(query_rows, logit_scale, effective_embedding)

    def sparse_score_rows(
        self,
        query_rows: torch.Tensor,
        logit_scale: torch.Tensor | float,
        embedding: torch.Tensor,
        *,
        base_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Diagnostic sparse score; never the registered method path."""

        if query_rows.ndim != 2 or int(query_rows.shape[-1]) != self.hidden_size:
            raise TRR0010ModelError("query rows must be [rows, hidden]")
        self._check_embedding(embedding)
        scale = torch.as_tensor(logit_scale, dtype=torch.float32, device=query_rows.device)
        if scale.ndim != 0 or not torch.isfinite(scale).item() or float(scale.item()) <= 0.0:
            raise TRR0010ModelError("logit scale must be one finite positive scalar")
        if base_logits is None:
            base_logits = (query_rows.to(embedding.dtype) @ embedding.transpose(0, 1)).float() * scale
        else:
            if base_logits.ndim != 2 or tuple(base_logits.shape) != (query_rows.shape[0], self.vocabulary_size):
                raise TRR0010ModelError("precomputed base logits geometry changed")
            if not torch.isfinite(base_logits).all().item():
                raise TRR0010ModelError("precomputed base logits are non-finite")
        delta_scores = (
            query_rows.float()
            @ self.delta_rows.to(device=query_rows.device, dtype=torch.float32).transpose(0, 1)
        ) * scale
        return self._apply_delta(base_logits, delta_scores)

    def merged_score_rows(
        self,
        query_rows: torch.Tensor,
        logit_scale: torch.Tensor | float,
        effective_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Score with one materialized effective E for validation/deployment."""

        if query_rows.ndim != 2 or int(query_rows.shape[-1]) != self.hidden_size:
            raise TRR0010ModelError("query rows must be [rows, hidden]")
        self._check_embedding(effective_embedding)
        scale = torch.as_tensor(logit_scale, dtype=torch.float32, device=query_rows.device)
        if scale.ndim != 0 or not torch.isfinite(scale).item() or float(scale.item()) <= 0.0:
            raise TRR0010ModelError("logit scale must be one finite positive scalar")
        return (
            query_rows.to(effective_embedding.dtype) @ effective_embedding.transpose(0, 1)
        ).float() * scale

    def merged_forward(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        effective_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Run the registered validation/deployment path through materialized E."""

        mask = self.base._check_inputs(activation, valid_mask)  # noqa: SLF001
        self._check_embedding(effective_embedding)
        hidden = self.base.projected_hidden(activation, mask)
        logits = self.merged_score_rows(hidden.reshape(-1, self.hidden_size), self.logit_scale, effective_embedding)
        logits = logits.reshape(*hidden.shape[:-1], self.vocabulary_size)
        return torch.where(mask.unsqueeze(-1), logits, torch.zeros_like(logits))

    def logits_from_rows(
        self,
        projected_hidden: torch.Tensor,
        record_slots: torch.Tensor,
        position_slots: torch.Tensor,
        embedding_table: torch.Tensor,
        *,
        base_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_projected_hidden(projected_hidden)
        if base_logits is None:
            base_logits = self.base.logits_from_rows(
                projected_hidden, record_slots, position_slots, embedding_table
            )
        else:
            if base_logits.ndim != 2 or int(base_logits.shape[-1]) != self.vocabulary_size:
                raise TRR0010ModelError("precomputed base logits geometry changed")
            if int(base_logits.shape[0]) != int(record_slots.numel()):
                raise TRR0010ModelError("precomputed base logits row count changed")
            if not torch.isfinite(base_logits).all().item():
                raise TRR0010ModelError("precomputed base logits are non-finite")
        rows = projected_hidden[
            record_slots.to(projected_hidden.device), position_slots.to(projected_hidden.device)
        ]
        return self.score_rows(
            rows, self.logit_scale, embedding_table, base_logits=base_logits
        )

    def selected_logits(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        selected_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        mask = self.base._check_inputs(activation, valid_mask)  # noqa: SLF001
        if selected_mask.ndim != 2 or tuple(selected_mask.shape) != tuple(mask.shape):
            raise TRR0010ModelError("selected mask geometry changed")
        selected = selected_mask.to(device=activation.device, dtype=torch.bool)
        if (selected & ~mask).any().item() or selected[:, 0].any().item():
            raise TRR0010ModelError("selected rows must be valid post-BOS positions")
        indices = torch.nonzero(selected, as_tuple=False)
        if int(indices.shape[0]) <= 0:
            raise TRR0010ModelError("selected rows are empty")
        hidden = self.base.projected_hidden(activation, mask)
        return self.logits_from_rows(hidden, indices[:, 0], indices[:, 1], embedding_table)

    def forward(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        mask = self.base._check_inputs(activation, valid_mask)  # noqa: SLF001
        self._check_embedding(embedding_table)
        return self.merged_forward(activation, mask, self.materialize_effective_embedding(embedding_table))

    def sparse_forward(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        """Diagnostic sparse forward; excluded from registered scoring."""

        mask = self.base._check_inputs(activation, valid_mask)  # noqa: SLF001
        self._check_embedding(embedding_table)
        hidden = self.base.projected_hidden(activation, mask)
        base_logits = hidden.to(embedding_table.dtype) @ embedding_table.transpose(0, 1)
        base_logits = base_logits.float() * self.logit_scale
        logits = self._apply_delta(base_logits, self._delta_scores(hidden))
        return torch.where(mask.unsqueeze(-1), logits, torch.zeros_like(logits))

    def effective_embedding_rows(self, embedding_table: torch.Tensor) -> torch.Tensor:
        """Return only the supported effective rows for export/equivalence."""

        self._check_embedding(embedding_table)
        rows = embedding_table.index_select(0, self.support_ids.to(embedding_table.device)).detach().clone()
        rows = rows + self.delta_rows.to(device=rows.device, dtype=rows.dtype)
        return rows

    def materialize_effective_embedding(self, embedding_table: torch.Tensor) -> torch.Tensor:
        """Materialize one full tied-E readout for deployment qualification."""

        self._check_embedding(embedding_table)
        detached_embedding = embedding_table.detach()
        ids = self.support_ids.to(device=detached_embedding.device)
        effective_rows = detached_embedding.index_select(0, ids) + self.delta_rows.to(
            device=detached_embedding.device, dtype=detached_embedding.dtype
        )
        # Functional index_copy allocates one output E_eff while leaving the
        # immutable public E untouched and preserving autograd to Delta.
        return torch.index_copy(detached_embedding, 0, ids, effective_rows)

    def optimizer_param_groups(
        self, decoder: nn.Module, *, base_learning_rate: float
    ) -> list[dict[str, Any]]:
        """Return the protocol-exact base and directional optimizer groups."""

        if base_learning_rate <= 0:
            raise TRR0010ModelError("base learning rate must be positive")
        parameters = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
        if not parameters:
            raise TRR0010ModelError("decoder has no trainable parameters")
        if decoder is not self.base:
            raise TRR0010ModelError("optimizer decoder must be this adapter's base")
        return [
            {"name": "decoder", "params": parameters, "lr": float(base_learning_rate)},
            {
                "name": "directional_delta",
                "params": [self.delta_rows],
                "lr": DEFAULT_DIRECTIONAL_LEARNING_RATE,
            },
        ]

    def optimizer_groups(
        self, base_lr: float, method_config: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Expose base and directional parameters to the shared optimizer.

        The default uses the fixed TRR-0010 directional rate (1e-4) for the
        extra group. A frozen synthetic/configuration test may explicitly
        provide ``directional_lr``; this hook does not choose a data schedule
        or sampler.
        """

        if base_lr <= 0:
            raise TRR0010ModelError("base learning rate must be positive")
        config = {} if method_config is None else dict(method_config)
        directional_lr = float(config.get("directional_lr", DEFAULT_DIRECTIONAL_LEARNING_RATE))
        if directional_lr <= 0:
            raise TRR0010ModelError("directional learning rate must be positive")
        return [
            {
                "name": "decoder_base",
                "params": [parameter for parameter in self.base.parameters() if parameter.requires_grad],
                "lr": float(base_lr),
            },
            {"name": "directional_delta", "params": [self.delta_rows], "lr": directional_lr},
        ]

    def loss_terms(
        self, batch_logits: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return full-vocabulary CE and the anchored directional term."""

        if batch_logits.ndim != 2 or int(batch_logits.shape[-1]) != self.vocabulary_size:
            raise TRR0010ModelError("batch logits must be [rows, vocabulary]")
        if targets.ndim != 1 or int(targets.shape[0]) != int(batch_logits.shape[0]):
            raise TRR0010ModelError("targets must match the scored row count")
        cross_entropy = F.cross_entropy(batch_logits, targets.to(device=batch_logits.device))
        penalty = self.readout_penalty().to(device=batch_logits.device, dtype=cross_entropy.dtype)
        return {
            "cross_entropy": cross_entropy,
            "readout_penalty": penalty,
            "total": cross_entropy + penalty,
        }

    def state_metadata(self) -> dict[str, Any]:
        """Shared-runner alias for the immutable architecture/state contract."""

        return {
            **self.metadata(),
            "readout_mode": self.readout_mode,
            "trainable_parameter_count": self.trainable_parameter_count,
            "directional_parameter_count": self.directional_parameter_count,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "method_id": METHOD_ID,
            "base_method_id": RESIDUAL_MLP_METHOD_ID,
            "current_H_only": True,
            "full_vocabulary_cross_entropy": True,
            "directional_parameterization": "supported-row additive Delta in tied public E space",
            "support_count": self.support_count,
            "support_digest": support_digest(self.support_ids, self.support_counts),
            "anchor_strength": self.anchor_strength,
            "embedding_epsilon": self.embedding_epsilon,
            "directional_learning_rate": DEFAULT_DIRECTIONAL_LEARNING_RATE,
            "training_score_path": "materialized_effective_E_full_vocab_matmul",
            "validation_score_path": "materialized_effective_E_full_vocab_matmul",
            "diagnostic_score_path": "sparse_supported_delta_after_base_full_vocab_score",
            "absent_rows_fixed": True,
            "global_linear_reparameterization": False,
        }


def build_directional_from_base(
    base: ResidualMLPPositionwiseDecoder,
    support_ids: torch.Tensor | Sequence[int],
    support_counts: torch.Tensor | Sequence[int],
    *,
    anchor_strength: float = DEFAULT_ANCHOR_STRENGTH,
    embedding_epsilon: float = DEFAULT_EMBEDDING_EPSILON,
) -> DirectionalTokenReadout:
    return DirectionalTokenReadout(
        base,
        support_ids,
        support_counts,
        anchor_strength=anchor_strength,
        embedding_epsilon=embedding_epsilon,
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
        raise TRR0010ModelError("published state did not load as residual MLP512")
    return model


def save_directional_state(
    path: Path,
    model: DirectionalTokenReadout,
    *,
    selected_step: int,
    base_state: Mapping[str, Any],
    fit_manifest: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a complete base-plus-directional state create-only."""

    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TRR0010ModelError(f"directional state is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    state_metadata = model.metadata()
    state_metadata.update(
        {
            "selected_step": int(selected_step),
            "starting_state_sha256": base_state.get("sha256"),
            "fit_manifest_sha256": fit_manifest.get("sha256"),
            "hidden_size": model.hidden_size,
            "vocabulary_size": model.vocabulary_size,
            "context_width": model.context_width,
            "bottleneck_size": model.bottleneck_size,
        }
    )
    if metadata:
        state_metadata.update({str(key): value for key, value in metadata.items()})
    save_file(state, str(path), metadata={key: str(value) for key, value in state_metadata.items()})
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "state_tensor_digest": _state_digest(state),
        "state_bytes": sum(int(value.numel()) * value.element_size() for value in state.values()),
        "tensor_keys": sorted(state),
        "metadata": state_metadata,
    }


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"trr0010-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_directional_state(
    path: Path,
    *,
    hidden_size: int,
    vocabulary_size: int,
    context_width: int,
    bottleneck_size: int = DEFAULT_BOTTLENECK_SIZE,
) -> DirectionalTokenReadout:
    """Load a complete directional state and validate its schema/geometry."""

    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TRR0010ModelError(f"directional state must be a regular file: {path}")
    try:
        state = load_file(str(path), device="cpu")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise TRR0010ModelError(f"cannot load directional state: {path}") from exc
    if metadata.get("schema") != SCHEMA or metadata.get("method_id") != METHOD_ID:
        raise TRR0010ModelError("directional state schema or method ID differs")
    if "support_ids" not in state or "support_counts" not in state or "delta_rows" not in state:
        raise TRR0010ModelError("directional state is missing support or delta tensors")
    base = build_residual_mlp512(
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
        bottleneck_size=bottleneck_size,
    )
    model = DirectionalTokenReadout(
        base,
        state["support_ids"],
        state["support_counts"],
        anchor_strength=float(metadata.get("anchor_strength", DEFAULT_ANCHOR_STRENGTH)),
        embedding_epsilon=float(metadata.get("embedding_epsilon", DEFAULT_EMBEDDING_EPSILON)),
    )
    expected = set(model.state_dict())
    if set(state) != expected:
        raise TRR0010ModelError("directional state tensor keys differ")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise TRR0010ModelError("directional state geometry differs") from exc
    return model


def export_effective_embedding(
    path: Path,
    model: DirectionalTokenReadout,
    embedding_table: torch.Tensor,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a single full-vocabulary effective-E tensor create-only."""

    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TRR0010ModelError(f"effective embedding export is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        effective = model.materialize_effective_embedding(embedding_table).detach().cpu().contiguous()
    export_metadata = model.metadata()
    export_metadata.update({"tensor_key": "embeddings", "materialized": True})
    if metadata:
        export_metadata.update({str(key): value for key, value in metadata.items()})
    save_file({"embeddings": effective}, str(path), metadata={key: str(value) for key, value in export_metadata.items()})
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "tensor_shape": list(effective.shape),
        "tensor_dtype": str(effective.dtype),
        "tensor_sha256": _tensor_digest(effective, prefix=b"trr0010-effective-embedding\0"),
        "metadata": export_metadata,
    }
