"""Controlled historical-style affine CE decoder for TRR-0004.

This module is intentionally separate from the TRR-0003 standalone decoders.
It provides the smallest controlled fit needed to test whether the retained
historical InputLens gap was caused by its fitting recipe rather than by the
need for A2.  The decoder keeps a full hidden-space affine map, a learned
global logit scale, and an optional vocabulary bias ablation::

    q = normalize(x @ W.T + b)
    logits = (q.to(E.dtype) @ E.T).float() * exp(s) + vocab_bias

``W``, ``b``, and ``s`` are fitted public-data state.  ``E`` is the fixed,
normalized public input-embedding table and is never part of the retained
decoder state.  ``bias_mode="none"`` is the historical-style arm; the
``"vocab"`` arm adds one trainable vocabulary bias to isolate that parameter
class.  Neither arm calls a public prefix or performs candidate simulation at
inference.

The data loader has a deliberately strict current-token contract.  It checks
fit/validation record disjointness from record manifests before opening public
labels, then validates the public labels and flattens positions ``1:`` while
preserving ``H_i -> token_ids[i]`` alignment.  This is a development/fitting
interface, not an evaluation-truth loader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Literal, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn


HISTORICAL_AFFINE_CE_SCHEMA = "token-reconstruction.trr0004-historical-affine-ce.v1"
FIT_DATA_SCHEMA = "token-reconstruction.trr0004-public-fit-data.v1"
CURRENT_TOKEN_ALIGNMENT = "current_token"
BOS_TOKEN_ID = 128000
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256


class HistoricalAffineCEError(RuntimeError):
    """Raised when the controlled decoder or public fit contract is invalid."""


BiasMode = Literal["none", "vocab"]


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HistoricalAffineCEError(f"{label} must be a regular file: {path}")
    return path


def file_sha256(path: Path) -> str:
    """Hash one external public resource without changing its bytes."""

    _regular_file(path, label="resource")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash shape, dtype, and raw contiguous tensor bytes."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _integer_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise HistoricalAffineCEError(f"{name} must be an integer tensor")
    return value.to(device="cpu", dtype=torch.long).contiguous()


def _finite_float_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not value.dtype.is_floating_point:
        raise HistoricalAffineCEError(f"{name} must be floating point")
    if not torch.isfinite(value).all().item():
        raise HistoricalAffineCEError(f"{name} contains non-finite values")
    return value.contiguous()


def normalized_embedding_table(
    embedding_table: torch.Tensor,
    *,
    vocabulary_size: int | None = None,
    hidden_size: int | None = None,
) -> torch.Tensor:
    """Validate and normalize a public embedding table once at its boundary.

    The controlled runner normally receives the already-normalized table used
    by the historical reference.  Normalizing here is retained as an explicit
    preparation helper for raw public embedding resources and matches the
    historical ``F.normalize(...float32...)`` convention.
    """

    if embedding_table.ndim != 2:
        raise HistoricalAffineCEError("embedding table must be a matrix")
    if vocabulary_size is not None and embedding_table.shape[0] != vocabulary_size:
        raise HistoricalAffineCEError("embedding table vocabulary size changed")
    if hidden_size is not None and embedding_table.shape[1] != hidden_size:
        raise HistoricalAffineCEError("embedding table hidden size changed")
    _finite_float_tensor(embedding_table, name="embedding table")
    result = F.normalize(embedding_table.detach().float(), dim=-1)
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    if not torch.isfinite(result).all().item():
        raise HistoricalAffineCEError("normalized embedding table is non-finite")
    return result


def validate_normalized_embedding_table(
    embedding_table: torch.Tensor,
    *,
    vocabulary_size: int,
    hidden_size: int,
    require_unit_norm: bool = True,
) -> None:
    """Validate a pre-normalized runtime table without silently renormalizing it."""

    if tuple(embedding_table.shape) != (vocabulary_size, hidden_size):
        raise HistoricalAffineCEError("normalized embedding table geometry changed")
    _finite_float_tensor(embedding_table, name="normalized embedding table")
    if not require_unit_norm:
        return
    norms = torch.linalg.vector_norm(embedding_table.float(), dim=-1)
    unit = torch.ones_like(norms)
    zero = torch.zeros_like(norms)
    allowed = torch.isclose(norms, unit, atol=2e-4, rtol=2e-4) | torch.isclose(
        norms, zero, atol=2e-6, rtol=0.0
    )
    if not allowed.all().item():
        raise HistoricalAffineCEError("embedding table is not normalized")


@dataclass(frozen=True)
class HistoricalAffineCEConfig:
    """Fixed optimizer and initialization settings for one controlled fit.

    ``gradient_clip_norm=0`` matches the historical source's disabled clipping;
    the training loop still rejects non-finite total gradient norms.
    """

    steps: int = 3000
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 0.0
    log_every: int = 25
    init_logit_scale: float = 3.0
    seed: int = 0
    scheduler: str = "CosineAnnealingLR"
    selection_metric: str = "validation_style_balanced_token_accuracy"

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.log_every <= 0:
            raise HistoricalAffineCEError("fit schedule must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise HistoricalAffineCEError("learning rate must be finite and positive")
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise HistoricalAffineCEError("weight decay must be finite and non-negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm < 0:
            raise HistoricalAffineCEError(
                "gradient clipping must be finite and non-negative; zero disables clipping"
            )
        if not math.isfinite(self.init_logit_scale):
            raise HistoricalAffineCEError("initial logit scale must be finite")
        if self.scheduler != "CosineAnnealingLR":
            raise HistoricalAffineCEError("controlled fit requires CosineAnnealingLR")
        if self.selection_metric != "validation_style_balanced_token_accuracy":
            raise HistoricalAffineCEError(
                "controlled fit selects on style-balanced public validation token accuracy"
            )


class HistoricalAffineCEDecoder(nn.Module):
    """Full hidden-affine, learned-scale, tied-embedding CE decoder."""

    method_id = "historical_affine_ce_no_vocab_bias"
    vocab_bias_method_id = "historical_affine_ce_vocab_bias"

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        bias_mode: BiasMode = "none",
        init_logit_scale: float = 3.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0:
            raise HistoricalAffineCEError("decoder geometry must be positive")
        if bias_mode not in ("none", "vocab"):
            raise HistoricalAffineCEError(f"unknown vocabulary-bias mode: {bias_mode}")
        if not math.isfinite(init_logit_scale):
            raise HistoricalAffineCEError("initial logit scale must be finite")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.bias_mode: BiasMode = bias_mode
        self.W = nn.Parameter(torch.eye(self.hidden_size, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(float(init_logit_scale), dtype=torch.float32))
        if bias_mode == "vocab":
            self.vocab_bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.vocab_size, dtype=torch.float32)
            )
        else:
            self.register_parameter("vocab_bias", None)

    @property
    def resolved_method_id(self) -> str:
        return (
            self.vocab_bias_method_id
            if self.bias_mode == "vocab"
            else self.method_id
        )

    @property
    def logit_scale_value(self) -> float:
        return float(self.s.detach().float().exp().item())

    def _check_activation(self, activation: torch.Tensor) -> None:
        if activation.ndim < 1 or activation.shape[-1] != self.hidden_size:
            raise HistoricalAffineCEError("activation hidden geometry changed")
        _finite_float_tensor(activation, name="activation")

    def _check_embeddings(self, embedding_table: torch.Tensor) -> None:
        if tuple(embedding_table.shape) != (self.vocab_size, self.hidden_size):
            raise HistoricalAffineCEError("runtime embedding table geometry changed")
        if not embedding_table.dtype.is_floating_point:
            raise HistoricalAffineCEError("runtime embedding table must be floating point")

    def projected(self, activation: torch.Tensor) -> torch.Tensor:
        """Apply ``activation @ W.T + b`` in float32."""

        self._check_activation(activation)
        value = activation.float()
        projected = value @ self.W.float().T + self.b.float()
        if not torch.isfinite(projected).all().item():
            raise HistoricalAffineCEError("projected activation is non-finite")
        return projected

    def forward(self, activation: torch.Tensor, embedding_table: torch.Tensor) -> torch.Tensor:
        self._check_embeddings(embedding_table)
        projected = F.normalize(self.projected(activation), dim=-1, eps=1e-12)
        if not torch.isfinite(projected).all().item():
            raise HistoricalAffineCEError("normalized projection is non-finite")
        logits = projected.to(embedding_table.dtype) @ embedding_table.transpose(0, 1)
        scale = self.s.float().exp()
        if not torch.isfinite(scale).item():
            raise HistoricalAffineCEError("learned logit scale is non-finite")
        result = logits.float() * scale
        if self.vocab_bias is not None:
            result = result + self.vocab_bias.float()
        if not torch.isfinite(result).all().item():
            raise HistoricalAffineCEError("decoder logits are non-finite")
        return result

    def state_bytes(self) -> int:
        return sum(
            int(parameter.numel()) * parameter.element_size()
            for parameter in self.parameters()
        )

    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())


@dataclass(frozen=True)
class PublicFitBundle:
    """Validated public fitting and development-validation tensors."""

    fit_observations: torch.Tensor
    fit_truth: torch.Tensor
    validation_observations: torch.Tensor
    validation_truth: torch.Tensor
    embedding_table: torch.Tensor
    metadata: Mapping[str, Any]
    layout: Literal["padded_records", "packed_records", "packed_post_bos"] = "padded_records"
    fit_record_counts: tuple[int, ...] = ()
    validation_record_counts: tuple[int, ...] = ()
    fit_valid_mask: torch.Tensor | None = None
    validation_valid_mask: torch.Tensor | None = None

    @property
    def hidden_size(self) -> int:
        return int(self.fit_observations.shape[-1])

    @property
    def vocabulary_size(self) -> int:
        return int(self.embedding_table.shape[0])

    @property
    def fit_record_count(self) -> int:
        return len(self.fit_record_counts) or int(self.fit_observations.shape[0])

    @property
    def validation_record_count(self) -> int:
        return len(self.validation_record_counts) or int(self.validation_observations.shape[0])

    def validation_flat_groups(self, *, include_bos: bool = False) -> tuple[str, ...]:
        """Expand record-level public groups to the flattened validation rows.

        Group membership is read from the sanitized record metadata retained by
        :func:`load_public_fit_bundle`.  The expansion follows the same BOS and
        layout rules as :meth:`validation_tensors`, so every validation row is
        assigned exactly one group.  A manifest without an explicit style or
        group field gets the single declared ``public`` group; in that case the
        balanced metric equals the ordinary aggregate metric.
        """

        record_resource = self.metadata.get("validation_records", {})
        rows = record_resource.get("records", []) if isinstance(record_resource, Mapping) else []
        names: list[str] = []
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            for row in rows:
                names.append(_record_group(row) if isinstance(row, Mapping) else "public")
        if len(names) != len(self.validation_record_counts):
            names = ["public"] * len(self.validation_record_counts)

        expanded: list[str] = []
        for group, count in zip(names, self.validation_record_counts):
            if self.layout == "packed_records":
                row_count = int(count) if include_bos else int(count) - 1
            elif self.layout == "packed_post_bos":
                if include_bos:
                    raise HistoricalAffineCEError("packed post-BOS validation has no BOS row")
                row_count = int(count)
            else:
                row_count = int(count) + 1 if include_bos else int(count)
            if row_count <= 0:
                raise HistoricalAffineCEError("validation group expansion produced no rows")
            expanded.extend([group] * row_count)
        if self.layout == "padded_records":
            expected = sum(
                int(count) + 1 if include_bos else int(count)
                for count in self.validation_record_counts
            )
        else:
            expected = int(self.validation_observations.shape[0])
        if len(expanded) != expected:
            raise HistoricalAffineCEError(
                f"validation groups do not match flattened rows: {len(expanded)} != {expected}"
            )
        return tuple(expanded)

    def fit_tensors(
        self,
        *,
        record_limit: int | None = None,
        position_limit: int | None = None,
        bos_token_id: int = BOS_TOKEN_ID,
        include_bos: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return flatten_current_token_records(
            self.fit_observations,
            self.fit_truth,
            bos_token_id=bos_token_id,
            record_limit=record_limit,
            position_limit=position_limit,
            layout=self.layout,
            record_counts=self.fit_record_counts,
            include_bos=include_bos,
            valid_mask=self.fit_valid_mask,
        )

    def validation_tensors(
        self,
        *,
        bos_token_id: int = BOS_TOKEN_ID,
        include_bos: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return flatten_current_token_records(
            self.validation_observations,
            self.validation_truth,
            bos_token_id=bos_token_id,
            layout=self.layout,
            record_counts=self.validation_record_counts,
            include_bos=include_bos,
            valid_mask=self.validation_valid_mask,
        )


def _padded_valid_lengths(
    valid_mask: torch.Tensor | None, observations: torch.Tensor
) -> tuple[int, ...]:
    """Validate right-padding and return each row's active token count."""

    rows, positions = int(observations.shape[0]), int(observations.shape[1])
    if valid_mask is None:
        return tuple(positions for _ in range(rows))
    if valid_mask.ndim != 2 or tuple(valid_mask.shape) != (rows, positions):
        raise HistoricalAffineCEError("padded validity mask must match observation rows and positions")
    if valid_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise HistoricalAffineCEError("padded validity mask must be boolean or integer")
    mask = valid_mask.to(device="cpu", dtype=torch.bool)
    lengths: list[int] = []
    for row in range(rows):
        active = int(mask[row].sum().item())
        if active < 2:
            raise HistoricalAffineCEError("each padded public row needs BOS and one post-BOS token")
        expected = torch.zeros(positions, dtype=torch.bool)
        expected[:active] = True
        if not torch.equal(mask[row], expected):
            raise HistoricalAffineCEError("padded validity mask must be right-padded and contiguous")
        lengths.append(active)
    return tuple(lengths)


def flatten_current_token_records(
    observations: torch.Tensor,
    truth: torch.Tensor,
    *,
    bos_token_id: int = BOS_TOKEN_ID,
    record_limit: int | None = None,
    position_limit: int | None = None,
    layout: Literal["padded_records", "packed_records", "packed_post_bos"] = "padded_records",
    record_counts: Sequence[int] | None = None,
    include_bos: bool = False,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten post-BOS rows while preserving ``H_i -> token_ids[i]``.

    ``record_limit`` is applied before flattening and therefore supports nested
    public fit panels.  ``position_limit`` limits the total number of scored
    post-BOS positions across selected records and is useful only for bounded
    diagnostics; neither option changes the alignment itself.
    """

    if layout not in ("padded_records", "packed_records", "packed_post_bos"):
        raise HistoricalAffineCEError(f"unknown fit tensor layout: {layout}")
    if valid_mask is not None and layout != "padded_records":
        raise HistoricalAffineCEError("validity masks are supported only for padded records")
    if not observations.dtype.is_floating_point or not torch.isfinite(observations).all().item():
        raise HistoricalAffineCEError("observations must be finite floating point")
    labels = _integer_tensor(truth, name="truth")
    if layout in ("packed_records", "packed_post_bos"):
        if observations.ndim != 2 or observations.shape[0] <= 0:
            raise HistoricalAffineCEError("packed observations must be [rows,hidden]")
        expected_label_rank = 1
        if labels.ndim != expected_label_rank or labels.shape[0] != observations.shape[0]:
            raise HistoricalAffineCEError("packed labels must match packed observations")
        counts = tuple(int(value) for value in (record_counts or ()))
        if not counts or any(value <= 0 for value in counts):
            raise HistoricalAffineCEError("packed layout requires positive record counts")
        if sum(counts) != observations.shape[0]:
            raise HistoricalAffineCEError("packed record counts do not cover observations")
        if record_limit is not None and (record_limit <= 0 or record_limit > len(counts)):
            raise HistoricalAffineCEError("record limit is outside the packed public rows")
        selected_counts = counts[: record_limit if record_limit is not None else len(counts)]
        if layout == "packed_post_bos":
            if include_bos:
                raise HistoricalAffineCEError("packed post-BOS data has no BOS rows to include")
            if position_limit is not None and (position_limit <= 0 or position_limit > sum(selected_counts)):
                raise HistoricalAffineCEError("position limit is outside packed post-BOS geometry")
            end = sum(selected_counts)
            if position_limit is not None:
                end = min(end, int(position_limit))
            x = observations[:end].contiguous()
            y = labels[:end].contiguous()
        else:
            if position_limit is not None and position_limit <= 0:
                raise HistoricalAffineCEError("position limit must be positive")
            chunks_x: list[torch.Tensor] = []
            chunks_y: list[torch.Tensor] = []
            cursor = 0
            post_bos_seen = 0
            for count in selected_counts:
                next_cursor = cursor + count
                if labels[cursor].item() != int(bos_token_id):
                    raise HistoricalAffineCEError("packed public record does not begin with BOS")
                available_post = count - 1
                take_post = available_post
                if position_limit is not None:
                    take_post = max(0, min(available_post, int(position_limit) - post_bos_seen))
                if include_bos:
                    chunks_x.append(observations[cursor : cursor + 1])
                    chunks_y.append(labels[cursor : cursor + 1])
                if take_post:
                    chunks_x.append(observations[cursor + 1 : cursor + 1 + take_post])
                    chunks_y.append(labels[cursor + 1 : cursor + 1 + take_post])
                post_bos_seen += take_post
                cursor = next_cursor
                if position_limit is not None and post_bos_seen >= int(position_limit):
                    break
            if position_limit is not None and post_bos_seen < int(position_limit):
                raise HistoricalAffineCEError("position limit exceeds packed post-BOS geometry")
            if not chunks_x:
                raise HistoricalAffineCEError("packed current-token flattening produced no rows")
            x = torch.cat(chunks_x, dim=0).contiguous()
            y = torch.cat(chunks_y, dim=0).contiguous()
    else:
        if observations.ndim != 3 or observations.shape[0] <= 0 or observations.shape[1] <= 1:
            raise HistoricalAffineCEError("observations must be [records,positions>1,hidden]")
        if truth.ndim != 2 or tuple(truth.shape) != tuple(observations.shape[:2]):
            raise HistoricalAffineCEError("truth geometry must match observations")
        lengths = _padded_valid_lengths(valid_mask, observations)
        if labels[:, 0].ne(int(bos_token_id)).any().item():
            raise HistoricalAffineCEError("public rows must begin with the declared BOS token")
        if record_limit is not None and (record_limit <= 0 or record_limit > observations.shape[0]):
            raise HistoricalAffineCEError("record limit is outside the available public rows")
        if position_limit is not None and position_limit <= 0:
            raise HistoricalAffineCEError("position limit must be positive")
        rows = int(observations.shape[0] if record_limit is None else record_limit)
        chunks_x: list[torch.Tensor] = []
        chunks_y: list[torch.Tensor] = []
        post_bos_seen = 0
        for row in range(rows):
            active = lengths[row]
            available_post = active - 1
            take_post = available_post
            if position_limit is not None:
                take_post = max(0, min(available_post, int(position_limit) - post_bos_seen))
            if include_bos:
                chunks_x.append(observations[row, :1, :])
                chunks_y.append(labels[row, :1])
            if take_post:
                chunks_x.append(observations[row, 1 : 1 + take_post, :])
                chunks_y.append(labels[row, 1 : 1 + take_post])
            post_bos_seen += take_post
            if position_limit is not None and post_bos_seen >= int(position_limit):
                break
        if position_limit is not None and post_bos_seen < int(position_limit):
            raise HistoricalAffineCEError("position limit exceeds padded post-BOS geometry")
        if not chunks_x:
            raise HistoricalAffineCEError("padded current-token flattening produced no rows")
        x = torch.cat(chunks_x, dim=0).reshape(-1, observations.shape[-1]).contiguous()
        y = torch.cat(chunks_y, dim=0).reshape(-1).contiguous()
    if x.shape[0] == 0:
        raise HistoricalAffineCEError("current-token flattening produced no rows")
    return x, y


def fixed_training_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    *,
    size: int = 2048,
    seed: int = 17,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one deterministic fit-only probe for learning-curve diagnostics.

    The probe is selected from the public fitting rows before any optimizer
    update.  Its indices are returned so the runner can bind the diagnostic to
    the exact fit stream.  It is never used for validation selection.
    """

    if activations.ndim != 2 or labels.ndim != 1 or activations.shape[0] != labels.shape[0]:
        raise HistoricalAffineCEError("training probe tensors have incompatible geometry")
    if activations.shape[0] <= 0 or size <= 0 or seed < 0:
        raise HistoricalAffineCEError("training probe settings must be positive")
    if not activations.dtype.is_floating_point or not torch.isfinite(activations).all().item():
        raise HistoricalAffineCEError("training probe activations must be finite floating point")
    labels = _integer_tensor(labels, name="training probe labels")
    count = min(int(size), int(activations.shape[0]))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if count == int(activations.shape[0]):
        indices = torch.arange(int(activations.shape[0]), dtype=torch.long)
    else:
        indices = torch.randperm(int(activations.shape[0]), generator=generator)[:count]
    return (
        activations.index_select(0, indices).contiguous(),
        labels.index_select(0, indices).contiguous(),
        indices.contiguous(),
    )


def _resource_from_manifest(
    root: Path,
    entry: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(entry, Mapping):
        raise HistoricalAffineCEError(f"{label} resource entry is malformed")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise HistoricalAffineCEError(f"{label} resource path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = _regular_file(path, label=label).resolve()
    expected_hash = entry.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise HistoricalAffineCEError(f"{label} resource must pin sha256")
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise HistoricalAffineCEError(f"{label} resource sha256 does not match its manifest")
    expected_bytes = entry.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
        raise HistoricalAffineCEError(f"{label} resource byte count does not match its manifest")
    record = dict(entry)
    record.update({"path": str(path), "sha256": actual_hash, "bytes": path.stat().st_size})
    return path, record


def _load_tensor_resource(
    root: Path,
    entry: Mapping[str, Any],
    *,
    key: str,
    label: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    path, record = _resource_from_manifest(root, entry, label=label)
    try:
        # Use safe_open so a combined public activation artifact can expose
        # observations, masks, and labels without opening the label tensor when
        # the caller is loading observations first.
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                raise HistoricalAffineCEError(
                    f"{label} does not contain the {key!r} tensor"
                )
            value = handle.get_tensor(key).contiguous()
    except HistoricalAffineCEError:
        raise
    except Exception as exc:  # pragma: no cover - backend-specific errors
        raise HistoricalAffineCEError(f"cannot load {label}: {path}") from exc
    expected_shape = record.get("shape")
    if expected_shape is not None and list(value.shape) != list(expected_shape):
        raise HistoricalAffineCEError(f"{label} tensor shape does not match its manifest")
    expected_dtype = record.get("dtype")
    if expected_dtype is not None and str(value.dtype) != expected_dtype:
        raise HistoricalAffineCEError(f"{label} tensor dtype does not match its manifest")
    return value, record


def _record_group(item: Mapping[str, Any]) -> str:
    """Return the declared public validation group for one metadata row."""

    for key in ("style", "group", "source", "dataset", "domain"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return "public"


def _record_ids(root: Path, entry: Mapping[str, Any], *, label: str) -> tuple[list[str], dict[str, Any]]:
    path, record = _resource_from_manifest(root, entry, label=label)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HistoricalAffineCEError(f"cannot parse {label} record manifest: {path}") from exc
    values = data.get("records") if isinstance(data, Mapping) else data
    if not isinstance(values, list) or not values:
        raise HistoricalAffineCEError(f"{label} record manifest has no records")
    identifiers: list[str] = []
    post_bos_counts: list[int] = []
    full_token_counts: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    post_counts_present = True
    full_counts_present = True
    for index, item in enumerate(values):
        if not isinstance(item, Mapping) or not isinstance(item.get("record_id"), str):
            raise HistoricalAffineCEError(f"{label} record {index} has no record_id")
        record_id = str(item["record_id"])
        identifiers.append(record_id)
        count = item.get("post_bos_token_count")
        if count is None:
            post_counts_present = False
        elif isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise HistoricalAffineCEError(f"{label} record {index} has invalid post-BOS count")
        else:
            post_bos_counts.append(int(count))
        full_count = item.get("full_token_count")
        if full_count is None:
            full_counts_present = False
        elif isinstance(full_count, bool) or not isinstance(full_count, int) or full_count <= 1:
            raise HistoricalAffineCEError(f"{label} record {index} has invalid full-token count")
        else:
            full_token_counts.append(int(full_count))
        # Keep only non-sensitive group and geometry metadata for metric
        # construction.  Source text and token IDs are never copied into the
        # bundle metadata.
        metadata_row: dict[str, Any] = {"record_id": record_id}
        for key in ("style", "group", "source", "dataset", "domain"):
            value = item.get(key)
            if isinstance(value, str) and value:
                metadata_row[key] = value
        if isinstance(count, int) and not isinstance(count, bool):
            metadata_row["post_bos_token_count"] = int(count)
        if isinstance(full_count, int) and not isinstance(full_count, bool):
            metadata_row["full_token_count"] = int(full_count)
        metadata_rows.append(metadata_row)
    if len(set(identifiers)) != len(identifiers):
        raise HistoricalAffineCEError(f"{label} record IDs are duplicated")
    if post_counts_present and len(post_bos_counts) != len(identifiers):
        raise HistoricalAffineCEError(f"{label} record post-BOS counts are incomplete")
    if full_counts_present and len(full_token_counts) != len(identifiers):
        raise HistoricalAffineCEError(f"{label} record full-token counts are incomplete")
    record["post_bos_token_counts"] = post_bos_counts if post_counts_present else None
    record["full_token_counts"] = full_token_counts if full_counts_present else None
    record["records"] = metadata_rows
    record["record_count"] = len(identifiers)
    record["record_ids_sha256"] = hashlib.sha256(
        json.dumps(identifiers, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identifiers, record


def _require_alignment(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = manifest.get("alignment")
    if not isinstance(value, Mapping):
        raise HistoricalAffineCEError("fit manifest must declare an alignment contract")
    result = dict(value)
    if result.get("mode") != CURRENT_TOKEN_ALIGNMENT:
        raise HistoricalAffineCEError("controlled fit requires current-token alignment")
    if result.get("observation_index") not in ("i", "position_i"):
        raise HistoricalAffineCEError("alignment must map observation H_i to position i")
    if result.get("label_index") not in ("i", "position_i"):
        raise HistoricalAffineCEError("alignment must map labels to position i")
    if result.get("bos_position") != 0:
        raise HistoricalAffineCEError("alignment must reserve position 0 for BOS")
    if result.get("scored_positions") not in ("1..L-1", "post_bos"):
        raise HistoricalAffineCEError("alignment must declare post-BOS scoring")
    return result


def load_public_fit_bundle(path: Path) -> PublicFitBundle:
    """Load a pinned public fit/validation bundle fail-closed.

    Record manifests are read and checked for overlap before any truth tensor
    is opened.  The truth files are public auxiliary labels by contract; this
    loader has no path for private evaluator truth.
    """

    manifest_path = _regular_file(path.resolve(), label="fit data manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HistoricalAffineCEError(f"cannot parse fit data manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != FIT_DATA_SCHEMA:
        raise HistoricalAffineCEError("fit data manifest schema changed")
    alignment = _require_alignment(manifest)
    layout = manifest.get("layout", "padded_records")
    if layout not in ("padded_records", "packed_records", "packed_post_bos"):
        raise HistoricalAffineCEError(f"unknown fit data layout: {layout}")
    root = manifest_path.parent
    resources = manifest.get("resources")
    if not isinstance(resources, Mapping):
        raise HistoricalAffineCEError("fit data manifest resources are missing")
    fit_ids, fit_records = _record_ids(root, resources.get("fit_records", {}), label="fit")
    validation_ids, validation_records = _record_ids(
        root, resources.get("validation_records", {}), label="validation"
    )
    overlap = sorted(set(fit_ids).intersection(validation_ids))
    if overlap:
        raise HistoricalAffineCEError(
            "fit and validation records overlap before public truth access: " + ",".join(overlap[:3])
        )

    fit_x, fit_x_record = _load_tensor_resource(
        root, resources.get("fit_observations", {}), key="activations", label="fit observations"
    )
    validation_x, validation_x_record = _load_tensor_resource(
        root,
        resources.get("validation_observations", {}),
        key="activations",
        label="validation observations",
    )

    # The public activation preparation stores right-padded rows and exposes
    # its attention masks in the same combined safetensors files.  The adapter
    # may register those keys explicitly; masks are loaded before labels so
    # padding can never enter a fit or validation flattening.
    fit_mask = None
    fit_mask_record = None
    validation_mask = None
    validation_mask_record = None
    for name in ("fit_valid_mask", "fit_attention_mask"):
        if name in resources:
            fit_mask, fit_mask_record = _load_tensor_resource(
                root, resources[name], key="attention_mask", label="fit validity mask"
            )
            break
    for name in ("validation_valid_mask", "validation_attention_mask"):
        if name in resources:
            validation_mask, validation_mask_record = _load_tensor_resource(
                root, resources[name], key="attention_mask", label="validation validity mask"
            )
            break

    # Only after split identity and observation geometry checks do we open the
    # public labels.  This preserves the charter's ordering discipline.
    fit_y, fit_y_record = _load_tensor_resource(
        root, resources.get("fit_truth", {}), key="token_ids", label="fit public labels"
    )
    validation_y, validation_y_record = _load_tensor_resource(
        root,
        resources.get("validation_truth", {}),
        key="token_ids",
        label="validation public labels",
    )
    embedding_table, embedding_record = _load_tensor_resource(
        root,
        resources.get("embedding_table", {}),
        key="embeddings",
        label="public embedding table",
    )

    bos_token_id = int(manifest.get("bos_token_id", BOS_TOKEN_ID))
    if layout == "packed_records":
        fit_counts = tuple(fit_records.get("full_token_counts") or ())
        validation_counts = tuple(validation_records.get("full_token_counts") or ())
    else:
        fit_counts = tuple(fit_records.get("post_bos_token_counts") or ())
        validation_counts = tuple(validation_records.get("post_bos_token_counts") or ())
    if layout in ("packed_records", "packed_post_bos"):
        if fit_mask is not None or validation_mask is not None:
            raise HistoricalAffineCEError("validity masks are only accepted for padded records")
        if fit_x.ndim != 2 or validation_x.ndim != 2:
            raise HistoricalAffineCEError("packed observations must be rank-2")
        if fit_y.ndim != 1 or validation_y.ndim != 1:
            raise HistoricalAffineCEError("packed public labels must be rank-1")
        if not fit_counts or not validation_counts:
            raise HistoricalAffineCEError("packed data requires per-record post-BOS counts")
        if sum(fit_counts) != fit_x.shape[0] or sum(validation_counts) != validation_x.shape[0]:
            raise HistoricalAffineCEError("packed counts do not match observation rows")
        if fit_y.shape[0] != fit_x.shape[0] or validation_y.shape[0] != validation_x.shape[0]:
            raise HistoricalAffineCEError("packed labels do not match observation rows")
        if fit_x.shape[1] != validation_x.shape[1]:
            raise HistoricalAffineCEError("fit and validation hidden geometry differ")
        if layout == "packed_records":
            fit_cursor = 0
            for count in fit_counts:
                if int(fit_y[fit_cursor].item()) != bos_token_id:
                    raise HistoricalAffineCEError("packed fit record does not begin with BOS")
                fit_cursor += count
            validation_cursor = 0
            for count in validation_counts:
                if int(validation_y[validation_cursor].item()) != bos_token_id:
                    raise HistoricalAffineCEError("packed validation record does not begin with BOS")
                validation_cursor += count
        fit_label_rows = fit_y
        validation_label_rows = validation_y
    else:
        if fit_x.ndim != 3 or validation_x.ndim != 3:
            raise HistoricalAffineCEError("padded observations must be rank-3")
        if fit_x.shape[0] != len(fit_ids) or validation_x.shape[0] != len(validation_ids):
            raise HistoricalAffineCEError("record manifest count does not match observations")
        if tuple(fit_x.shape[1:]) != tuple(validation_x.shape[1:]):
            raise HistoricalAffineCEError("fit and validation observation geometry differ")
        if fit_y.ndim != 2 or validation_y.ndim != 2:
            raise HistoricalAffineCEError("padded public labels must be rank-2")
        if tuple(fit_y.shape) != tuple(fit_x.shape[:2]) or tuple(validation_y.shape) != tuple(validation_x.shape[:2]):
            raise HistoricalAffineCEError("padded public labels do not match observation geometry")
        if fit_y[:, 0].ne(bos_token_id).any().item() or validation_y[:, 0].ne(bos_token_id).any().item():
            raise HistoricalAffineCEError("padded public rows must begin with BOS")
        fit_lengths = _padded_valid_lengths(fit_mask, fit_x)
        validation_lengths = _padded_valid_lengths(validation_mask, validation_x)
        fit_label_rows = fit_y[:, 1:]
        validation_label_rows = validation_y[:, 1:]
    _finite_float_tensor(fit_x, name="fit observations")
    _finite_float_tensor(validation_x, name="validation observations")
    fit_y = _integer_tensor(fit_y, name="fit public labels")
    validation_y = _integer_tensor(validation_y, name="validation public labels")
    if layout == "padded_records":
        fit_scored_mask = fit_mask[:, 1:].to(torch.bool) if fit_mask is not None else None
        validation_scored_mask = (
            validation_mask[:, 1:].to(torch.bool) if validation_mask is not None else None
        )
        if fit_scored_mask is not None:
            fit_label_rows = fit_y[:, 1:].masked_select(fit_scored_mask)
        if validation_scored_mask is not None:
            validation_label_rows = validation_y[:, 1:].masked_select(validation_scored_mask)
    if fit_label_rows.lt(0).any().item() or validation_label_rows.lt(0).any().item():
        raise HistoricalAffineCEError("public labels contain negative token IDs")
    if fit_label_rows.ge(embedding_table.shape[0]).any().item() or validation_label_rows.ge(embedding_table.shape[0]).any().item():
        raise HistoricalAffineCEError("public labels exceed embedding vocabulary")
    validate_normalized_embedding_table(
        embedding_table,
        vocabulary_size=int(embedding_table.shape[0]),
        hidden_size=int(fit_x.shape[-1]),
        require_unit_norm=bool(manifest.get("embedding_table_normalized", True)),
    )
    if layout == "padded_records":
        fit_counts = tuple(int(length - 1) for length in fit_lengths)
        validation_counts = tuple(int(length - 1) for length in validation_lengths)
        declared_fit = fit_records.get("post_bos_token_counts")
        declared_validation = validation_records.get("post_bos_token_counts")
        if declared_fit is not None and tuple(declared_fit) != fit_counts:
            raise HistoricalAffineCEError("fit record lengths disagree with the validity mask")
        if declared_validation is not None and tuple(declared_validation) != validation_counts:
            raise HistoricalAffineCEError("validation record lengths disagree with the validity mask")
    if len(fit_ids) != len(fit_counts) or len(validation_ids) != len(validation_counts):
        raise HistoricalAffineCEError("record identities and count metadata disagree")
    metadata = {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "schema": FIT_DATA_SCHEMA,
        "alignment": alignment,
        "layout": layout,
        "bos_token_id": int(manifest.get("bos_token_id", BOS_TOKEN_ID)),
        "fit_records": fit_records,
        "validation_records": validation_records,
        "fit_observations": fit_x_record,
        "validation_observations": validation_x_record,
        "fit_truth": fit_y_record,
        "validation_truth": validation_y_record,
        "embedding_table": embedding_record,
        "fit_valid_mask": fit_mask_record,
        "validation_valid_mask": validation_mask_record,
        "fit_record_ids": fit_ids,
        "validation_record_ids": validation_ids,
        "public_labels_only": True,
        "fit_record_counts": list(fit_counts),
        "validation_record_counts": list(validation_counts),
    }
    return PublicFitBundle(
        fit_observations=fit_x,
        fit_truth=fit_y,
        validation_observations=validation_x,
        validation_truth=validation_y,
        embedding_table=embedding_table.float().contiguous(),
        metadata=metadata,
        layout=layout,
        fit_record_counts=fit_counts,
        validation_record_counts=validation_counts,
        fit_valid_mask=fit_mask,
        validation_valid_mask=validation_mask,
    )


def evaluation_schedule(steps: int) -> tuple[int, ...]:
    """Return the preregistered early-stop diagnostic checkpoints.

    Step zero records the untrained identity-affine state.  The denser early
    checkpoints avoid missing a short-lived public-validation peak; after step
    200, checkpoints are every 100 updates.  The final step is always included
    for non-grid schedules used by tests or bounded smoke runs.
    """

    if steps <= 0:
        raise HistoricalAffineCEError("evaluation schedule requires positive steps")
    values = [0, 25, 50, 75, 100, 150, 200]
    values.extend(range(300, steps + 1, 100))
    values.append(int(steps))
    return tuple(sorted({value for value in values if value <= steps}))



def _evaluate_metrics(
    model: HistoricalAffineCEDecoder,
    observations: torch.Tensor,
    labels: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate direct predictions and, optionally, record-group metrics."""

    if observations.ndim != 2 or labels.ndim != 1 or observations.shape[0] != labels.shape[0]:
        raise HistoricalAffineCEError("evaluation tensors have incompatible geometry")
    if observations.shape[0] == 0:
        raise HistoricalAffineCEError("evaluation set is empty")
    if batch_size <= 0:
        raise HistoricalAffineCEError("evaluation batch size must be positive")
    group_names: tuple[str, ...] | None = None
    if groups is not None:
        group_names = tuple(str(value) for value in groups)
        if len(group_names) != int(observations.shape[0]) or any(not value for value in group_names):
            raise HistoricalAffineCEError("evaluation groups must align with every validation row")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_rows = 0
    group_loss: dict[str, float] = {}
    group_correct: dict[str, int] = {}
    group_rows: dict[str, int] = {}
    with torch.inference_mode():
        for start in range(0, int(observations.shape[0]), batch_size):
            stop = min(start + batch_size, int(observations.shape[0]))
            logits = model(
                observations[start:stop].to(device=device, dtype=torch.float32),
                embedding_table,
            )
            target = labels[start:stop].to(device=device, dtype=torch.long)
            losses = F.cross_entropy(logits, target, reduction="none")
            prediction = logits.argmax(dim=-1)
            correct = prediction.eq(target)
            total_loss += float(losses.sum().cpu())
            total_correct += int(correct.sum().cpu())
            total_rows += stop - start
            if group_names is not None:
                for local_index, group in enumerate(group_names[start:stop]):
                    group_loss[group] = group_loss.get(group, 0.0) + float(
                        losses[local_index].cpu()
                    )
                    group_correct[group] = group_correct.get(group, 0) + int(
                        correct[local_index].cpu()
                    )
                    group_rows[group] = group_rows.get(group, 0) + 1
    result: dict[str, Any] = {
        "loss": total_loss / total_rows,
        "token_accuracy": total_correct / total_rows,
        "correct_tokens": total_correct,
        "token_rows": total_rows,
    }
    if group_names is not None:
        group_accuracy = {
            group: group_correct[group] / group_rows[group] for group in sorted(group_rows)
        }
        result.update(
            {
                "group_token_accuracy": group_accuracy,
                "group_token_rows": {group: group_rows[group] for group in sorted(group_rows)},
                "style_balanced_token_accuracy": sum(group_accuracy.values()) / len(group_accuracy),
                "style_balanced_groups": sorted(group_accuracy),
            }
        )
    return result


def _evaluate(
    model: HistoricalAffineCEDecoder,
    observations: torch.Tensor,
    labels: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """Compatibility wrapper returning aggregate loss and token accuracy."""

    metrics = _evaluate_metrics(
        model, observations, labels, embedding_table, device=device, batch_size=batch_size
    )
    return float(metrics["loss"]), float(metrics["token_accuracy"])


def _cpu_state(model: HistoricalAffineCEDecoder) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}


def train_historical_affine_ce(
    model: HistoricalAffineCEDecoder,
    activations: torch.Tensor,
    labels: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    config: HistoricalAffineCEConfig,
    device: torch.device,
    validation: tuple[torch.Tensor, torch.Tensor] | None = None,
    validation_groups: Sequence[str] | None = None,
    training_probe: tuple[torch.Tensor, torch.Tensor] | None = None,
    deadline: float | None = None,
    resource_guard: Callable[[], Any] | None = None,
) -> tuple[HistoricalAffineCEDecoder, dict[str, Any]]:
    """Run the fixed public-data CE schedule and retain earliest best state."""

    config.validate()
    if activations.ndim != 2 or labels.ndim != 1 or activations.shape[0] != labels.shape[0]:
        raise HistoricalAffineCEError("fit tensors have incompatible geometry")
    if activations.shape[0] <= 0:
        raise HistoricalAffineCEError("fit tensors are empty")
    if labels.lt(0).any().item() or labels.ge(embedding_table.shape[0]).any().item():
        raise HistoricalAffineCEError("fit labels are outside the embedding vocabulary")
    validate_normalized_embedding_table(
        embedding_table,
        vocabulary_size=int(model.vocab_size),
        hidden_size=int(model.hidden_size),
        require_unit_norm=False,
    )
    if validation is not None:
        if len(validation) != 2:
            raise HistoricalAffineCEError("validation must contain observations and labels")
        if validation[0].ndim != 2 or validation[1].ndim != 1 or validation[0].shape[0] != validation[1].shape[0]:
            raise HistoricalAffineCEError("validation tensors have incompatible geometry")
        if validation[1].lt(0).any().item() or validation[1].ge(embedding_table.shape[0]).any().item():
            raise HistoricalAffineCEError("validation labels are outside the embedding vocabulary")
        if validation_groups is None:
            validation_groups = ("public",) * int(validation[0].shape[0])
        else:
            validation_groups = tuple(str(value) for value in validation_groups)
            if len(validation_groups) != int(validation[0].shape[0]) or any(
                not value for value in validation_groups
            ):
                raise HistoricalAffineCEError(
                    "validation groups must align with every public validation row"
                )
    elif validation_groups is not None:
        raise HistoricalAffineCEError("validation groups require validation tensors")

    probe_x = activations
    probe_y = labels
    probe_source = "full_fit_fallback"
    if training_probe is not None:
        if len(training_probe) != 2:
            raise HistoricalAffineCEError("training probe must contain observations and labels")
        probe_x, probe_y = training_probe
        if probe_x.ndim != 2 or probe_y.ndim != 1 or probe_x.shape[0] != probe_y.shape[0]:
            raise HistoricalAffineCEError("training probe tensors have incompatible geometry")
        if probe_x.shape[1] != activations.shape[1] or probe_x.shape[0] <= 0:
            raise HistoricalAffineCEError("training probe geometry does not match fit data")
        if probe_y.lt(0).any().item() or probe_y.ge(embedding_table.shape[0]).any().item():
            raise HistoricalAffineCEError("training probe labels are outside the embedding vocabulary")
        probe_source = "fixed_public_fit_probe"

    model = model.to(device=device)
    x = activations.detach().to(device=device, dtype=torch.float32)
    y = labels.detach().to(device=device, dtype=torch.long)
    embeddings = embedding_table.detach().to(device=device)
    probe_x = probe_x.detach().to(device=device, dtype=torch.float32)
    probe_y = probe_y.detach().to(device=device, dtype=torch.long)
    val_x = validation[0].detach() if validation is not None else None
    val_y = validation[1].detach() if validation is not None else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    curve: list[dict[str, Any]] = []
    minibatch_losses: list[float] = []
    gradient_norms: list[float] = []
    best_metric = -float("inf")
    best_step: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    scheduled_steps = set(evaluation_schedule(config.steps))
    started = time.perf_counter()

    def record_curve(step: int) -> None:
        nonlocal best_metric, best_step, best_state
        # Curve train metrics use one fixed fit-only probe.  This prevents the
        # repeated checkpoints from turning into dozens of full 124k-row
        # training-set passes while preserving a comparable learning trace.
        probe_metrics = _evaluate_metrics(
            model, probe_x, probe_y, embeddings, device=device, batch_size=config.batch_size
        )
        point: dict[str, Any] = {
            "step": int(step),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_probe_loss": float(probe_metrics["loss"]),
            "train_probe_token_accuracy": float(probe_metrics["token_accuracy"]),
            # Keep these aliases for consumers of the earlier smoke-run schema;
            # they explicitly refer to the fixed probe in the evidence notes.
            "train_loss": float(probe_metrics["loss"]),
            "train_token_accuracy": float(probe_metrics["token_accuracy"]),
        }
        if val_x is not None and val_y is not None:
            validation_metrics = _evaluate_metrics(
                model,
                val_x,
                val_y,
                embeddings,
                device=device,
                batch_size=config.batch_size,
                groups=validation_groups,
            )
            point.update(
                {
                    "validation_loss": float(validation_metrics["loss"]),
                    "validation_token_accuracy": float(validation_metrics["token_accuracy"]),
                    "validation_group_token_accuracy": validation_metrics[
                        "group_token_accuracy"
                    ],
                    "validation_group_token_rows": validation_metrics["group_token_rows"],
                    "validation_style_balanced_token_accuracy": float(
                        validation_metrics["style_balanced_token_accuracy"]
                    ),
                    "validation_style_balanced_groups": validation_metrics[
                        "style_balanced_groups"
                    ],
                }
            )
            metric = float(validation_metrics["style_balanced_token_accuracy"])
            if metric > best_metric:
                best_metric = metric
                best_step = int(step)
                best_state = _cpu_state(model)
        curve.append(point)

    # Evaluate the fixed identity initialization before any update.  This is
    # public validation only and makes an earliest-best selection explicit.
    record_curve(0)
    model.train()
    for step_index in range(config.steps):
        if resource_guard is not None:
            try:
                resource_guard()
            except HistoricalAffineCEError:
                raise
            except Exception as exc:
                raise HistoricalAffineCEError("fit resource guard failed") from exc
        if deadline is not None and time.perf_counter() >= deadline:
            raise HistoricalAffineCEError("fit exceeded its wall-time guard")
        count = min(config.batch_size, int(x.shape[0]))
        indices = torch.randint(0, x.shape[0], (count,), generator=generator).to(device)
        logits = model(x.index_select(0, indices), embeddings)
        loss = F.cross_entropy(logits, y.index_select(0, indices))
        if not torch.isfinite(loss).item():
            raise HistoricalAffineCEError("fit loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # The historical source did not clip gradients.  Passing infinity to
        # PyTorch computes the total norm and leaves finite gradients unchanged;
        # error_if_nonfinite still fails closed on NaN/Inf gradients.
        max_norm = config.gradient_clip_norm if config.gradient_clip_norm > 0 else float("inf")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm, error_if_nonfinite=True
        )
        if not torch.isfinite(norm).item():
            raise HistoricalAffineCEError("gradient norm is non-finite")
        optimizer.step()
        scheduler.step()
        minibatch_losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(norm.detach().cpu()))
        step = step_index + 1
        if step in scheduled_steps:
            record_curve(step)

    # One full fit-set pass is retained for a final capacity diagnostic.  The
    # checkpoint curve itself remains probe-based, and validation selection
    # never uses this metric.
    full_train_metrics = _evaluate_metrics(
        model, x, y, embeddings, device=device, batch_size=config.batch_size
    )
    if best_state is None:
        # A fit without validation still gets an explicit final state.  The
        # controlled runner normally passes public validation tensors.
        best_step = config.steps
        best_state = _cpu_state(model)
        best_metric = float("nan")
    evidence = {
        "config": asdict(config),
        "bias_mode": model.bias_mode,
        "method_id": model.resolved_method_id,
        "examples": int(x.shape[0]),
        "initial_minibatch_loss": minibatch_losses[0],
        "final_minibatch_loss": minibatch_losses[-1],
        "minimum_minibatch_loss": min(minibatch_losses),
        "gradient_norm_max": max(gradient_norms),
        "learning_curve": curve,
        "evaluation_schedule": list(evaluation_schedule(config.steps)),
        "training_probe": {
            "source": probe_source,
            "rows": int(probe_x.shape[0]),
        },
        "final_full_fit": {
            "loss": float(full_train_metrics["loss"]),
            "token_accuracy": float(full_train_metrics["token_accuracy"]),
            "rows": int(x.shape[0]),
            "evaluation_role": "one final capacity diagnostic; never used for selection",
        },
        "best_validation_token_accuracy": (
            None
            if math.isnan(best_metric)
            else float(curve[[point["step"] for point in curve].index(best_step)]["validation_token_accuracy"])
        ),
        "best_validation_style_balanced_token_accuracy": (
            None if math.isnan(best_metric) else float(best_metric)
        ),
        "selected_step": best_step,
        "trainable_parameters": model.parameter_count(),
        "trainable_state_bytes": model.state_bytes(),
        "elapsed_seconds": time.perf_counter() - started,
        "selection_metric": config.selection_metric,
        "selection_rule": "earliest step attaining the maximum style-balanced public validation token accuracy",
        "validation_groups": sorted(set(validation_groups)) if validation_groups is not None else ["public"],
    }
    # Return the final model; the caller can serialize both it and best_state.
    evidence["selected_state_dict"] = best_state
    model.eval()
    return model, evidence


def save_historical_affine_ce(
    model: HistoricalAffineCEDecoder,
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
    state: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Serialize decoder-only state; the fixed public table stays external."""

    if path.exists() or path.is_symlink():
        raise HistoricalAffineCEError(f"decoder artifact is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = (
        {key: value.detach().cpu().contiguous() for key, value in state.items()}
        if state is not None
        else _cpu_state(model)
    )
    expected = set(model.state_dict())
    if set(tensors) != expected:
        raise HistoricalAffineCEError("decoder state fields do not match model bias mode")
    save_file(
        tensors,
        str(path),
        metadata={
            "schema": HISTORICAL_AFFINE_CE_SCHEMA,
            "method_id": model.resolved_method_id,
            "bias_mode": model.bias_mode,
            "hidden_size": str(model.hidden_size),
            "vocab_size": str(model.vocab_size),
            **{str(key): str(value) for key, value in (metadata or {}).items()},
        },
    )


def load_historical_affine_ce(
    path: Path,
    *,
    hidden_size: int,
    vocab_size: int,
    bias_mode: BiasMode,
    device: torch.device,
) -> HistoricalAffineCEDecoder:
    """Load a controlled-fit state with an explicit architecture contract."""

    _regular_file(path, label="historical affine CE state")
    state = load_file(str(path), device="cpu")
    model = HistoricalAffineCEDecoder(hidden_size, vocab_size, bias_mode=bias_mode)
    if set(state) != set(model.state_dict()):
        raise HistoricalAffineCEError("historical affine CE state fields changed")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device=device).eval()


def direct_prediction_tensor(
    model: HistoricalAffineCEDecoder,
    activations: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Emit one direct token per activation row, with no A2 fallback."""

    if batch_size <= 0:
        raise HistoricalAffineCEError("prediction batch size must be positive")
    result: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(activations.shape[0]), batch_size):
            stop = min(start + batch_size, int(activations.shape[0]))
            logits = model(
                activations[start:stop].to(device=device, dtype=torch.float32),
                embedding_table.to(device=device),
            )
            result.append(logits.argmax(dim=-1).to(device="cpu", dtype=torch.int32))
    if not result:
        raise HistoricalAffineCEError("prediction received no rows")
    return torch.cat(result, dim=0).contiguous()


def method_ids() -> tuple[str, str]:
    return (
        HistoricalAffineCEDecoder.method_id,
        HistoricalAffineCEDecoder.vocab_bias_method_id,
    )

