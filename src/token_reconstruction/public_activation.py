"""Public activation preparation for the TRR-0004 controlled fit.

The preparation path runs the pinned public Llama prefix over the registered
public Alpaca fit and validation records.  It stores the current-token aligned
labels and masks beside BF16 cut-4 activations, with a nested post-BOS selector
for the first 5,000 positions.  It never reads target weights or evaluator
private truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .alpaca_split import (
    DEFAULT_BOS_TOKEN_ID,
    HISTORICAL_MAX_TOKENS,
    HISTORICAL_MIN_FULL_TOKENS,
    _input_ids,
    historical_rendered_text,
    metadata_for_record,
)
from .public_prefix import ContiguousPublicPrefix, PublicPrefixError


PUBLIC_ACTIVATION_SCHEMA = "token-reconstruction.trr0004-public-activation.v1"
PUBLIC_RECORD_MANIFEST_SCHEMA = "token-reconstruction.trr0004-public-activation-records.v1"
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
CUT_DEPTH = 4
PAD_TOKEN_ID = 128001


class PublicActivationError(RuntimeError):
    """Raised when public activation materialization violates its contract."""


@dataclass(frozen=True)
class PaddedTokenBatch:
    """Padded public labels and masks retaining current-token alignment."""

    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    post_bos_selector_small: torch.Tensor
    post_bos_selector_large: torch.Tensor
    post_bos_ranges: tuple[tuple[int, int], ...]

    @property
    def post_bos_positions(self) -> int:
        return int(self.post_bos_selector_large.sum().item())

    @property
    def small_positions(self) -> int:
        return int(self.post_bos_selector_small.sum().item())


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def record_ids_sha256(record_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for record_id in record_ids:
        digest.update(str(record_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _as_token_list(sequence: Sequence[int], *, bos_token_id: int, vocab_size: int) -> list[int]:
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)):
        raise PublicActivationError("token sequence must be a one-dimensional integer sequence")
    values: list[int] = []
    for value in sequence:
        if not isinstance(value, int):
            raise PublicActivationError("token IDs must be integers")
        if value < 0 or value >= vocab_size:
            raise PublicActivationError("token ID is outside the public vocabulary")
        values.append(int(value))
    if len(values) < 2:
        raise PublicActivationError("every public sequence needs BOS and at least one current token")
    if values[0] != bos_token_id:
        raise PublicActivationError("public sequence does not begin with the declared BOS token")
    return values


def pad_public_token_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    maximum_tokens: int = HISTORICAL_MAX_TOKENS,
    pad_token_id: int = PAD_TOKEN_ID,
    bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
    vocab_size: int = VOCAB_SIZE,
    small_post_bos_positions: int = 5000,
) -> PaddedTokenBatch:
    """Pad public sequences and construct nested post-BOS selectors.

    The same index ``[record, position]`` addresses an activation and its
    current token label.  BOS is retained at position zero for prefix context;
    both selectors exclude it.  Right-padding is represented only through the
    binary attention mask, and padded activation rows are zeroed after capture.
    """

    if not sequences:
        raise PublicActivationError("cannot pad an empty public sequence collection")
    if maximum_tokens <= 1 or small_post_bos_positions <= 0:
        raise PublicActivationError("invalid public sequence geometry")
    if pad_token_id < 0 or pad_token_id >= vocab_size:
        raise PublicActivationError("padding token is outside the public vocabulary")

    normalized = [_as_token_list(seq, bos_token_id=bos_token_id, vocab_size=vocab_size) for seq in sequences]
    if any(len(seq) > maximum_tokens for seq in normalized):
        raise PublicActivationError("public sequence exceeds the declared padded geometry")

    count = len(normalized)
    token_ids = torch.full((count, maximum_tokens), pad_token_id, dtype=torch.int32)
    attention_mask = torch.zeros((count, maximum_tokens), dtype=torch.uint8)
    position_ids = torch.zeros((count, maximum_tokens), dtype=torch.int64)
    selector_small = torch.zeros((count, maximum_tokens), dtype=torch.uint8)
    selector_large = torch.zeros((count, maximum_tokens), dtype=torch.uint8)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for row, sequence in enumerate(normalized):
        length = len(sequence)
        token_ids[row, :length] = torch.tensor(sequence, dtype=torch.int32)
        attention_mask[row, :length] = 1
        position_ids[row, :length] = torch.arange(length, dtype=torch.int64)
        post_count = length - 1
        start = cursor
        end = cursor + post_count
        ranges.append((start, end))
        selector_large[row, 1:length] = 1
        small_start = max(0, 0 - start)
        small_end = min(post_count, small_post_bos_positions - start)
        if small_end > small_start:
            selector_small[row, 1 + small_start : 1 + small_end] = 1
        cursor = end

    result = PaddedTokenBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        post_bos_selector_small=selector_small,
        post_bos_selector_large=selector_large,
        post_bos_ranges=tuple(ranges),
    )
    validate_padded_token_batch(
        result,
        maximum_tokens=maximum_tokens,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        vocab_size=vocab_size,
        small_post_bos_positions=small_post_bos_positions,
    )
    return result


def validate_padded_token_batch(
    batch: PaddedTokenBatch,
    *,
    maximum_tokens: int = HISTORICAL_MAX_TOKENS,
    pad_token_id: int = PAD_TOKEN_ID,
    bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
    vocab_size: int = VOCAB_SIZE,
    small_post_bos_positions: int = 5000,
) -> None:
    """Validate geometry, ranges, padding, and selector/current-label alignment."""

    tensors = (
        batch.token_ids,
        batch.attention_mask,
        batch.position_ids,
        batch.post_bos_selector_small,
        batch.post_bos_selector_large,
    )
    if tuple(batch.token_ids.shape) != (len(batch.post_bos_ranges), maximum_tokens):
        raise PublicActivationError("token batch geometry changed")
    if any(tuple(value.shape) != tuple(batch.token_ids.shape) for value in tensors[1:]):
        raise PublicActivationError("token batch auxiliary geometry changed")
    if batch.token_ids.dtype not in (torch.int32, torch.int64):
        raise PublicActivationError("token IDs must use an integer dtype")
    if batch.attention_mask.dtype != torch.uint8:
        raise PublicActivationError("attention mask must use uint8")
    if batch.position_ids.dtype != torch.int64:
        raise PublicActivationError("position IDs must use int64")
    for name, selector in (
        ("small selector", batch.post_bos_selector_small),
        ("large selector", batch.post_bos_selector_large),
    ):
        if selector.dtype != torch.uint8:
            raise PublicActivationError(f"{name} must use uint8")
        if not torch.logical_or(selector.eq(0), selector.eq(1)).all().item():
            raise PublicActivationError(f"{name} must be binary")
    if not torch.logical_or(batch.attention_mask.eq(0), batch.attention_mask.eq(1)).all().item():
        raise PublicActivationError("attention mask must be binary")
    if batch.token_ids.lt(0).any().item() or batch.token_ids.ge(vocab_size).any().item():
        raise PublicActivationError("token IDs are outside the public vocabulary")

    for row in range(batch.token_ids.shape[0]):
        active = batch.attention_mask[row].to(torch.bool)
        active_count = int(active.sum().item())
        if active_count < 2 or int(batch.token_ids[row, 0].item()) != bos_token_id:
            raise PublicActivationError("each row must have BOS and a current token")
        if not torch.equal(batch.position_ids[row, :active_count], torch.arange(active_count)):
            raise PublicActivationError("active position IDs are not contiguous")
        if active_count < maximum_tokens:
            if not batch.token_ids[row, active_count:].eq(pad_token_id).all().item():
                raise PublicActivationError("padded token IDs changed")
            if not batch.position_ids[row, active_count:].eq(0).all().item():
                raise PublicActivationError("padded position IDs must be zero")
        if batch.post_bos_selector_small[row, 0].item() or batch.post_bos_selector_large[row, 0].item():
            raise PublicActivationError("selectors must exclude BOS")
        if not (batch.post_bos_selector_small[row].le(batch.post_bos_selector_large[row])).all().item():
            raise PublicActivationError("small selector is not nested in the large selector")
        if not (batch.post_bos_selector_large[row].le(active.to(torch.uint8))).all().item():
            raise PublicActivationError("large selector includes padding")

    if batch.small_positions != min(batch.post_bos_positions, small_post_bos_positions):
        raise PublicActivationError("nested small selector has the wrong position count")
    if len(batch.post_bos_ranges) and batch.post_bos_ranges[-1][1] != batch.post_bos_positions:
        raise PublicActivationError("post-BOS range metadata disagrees with selector")


def validate_activation_tensor(
    activations: torch.Tensor,
    token_batch: PaddedTokenBatch,
    *,
    hidden_size: int = HIDDEN_SIZE,
    require_bfloat16: bool = True,
) -> None:
    """Validate captured activations and zeroed right-padding."""

    if tuple(activations.shape) != tuple(token_batch.token_ids.shape) + (hidden_size,):
        raise PublicActivationError("activation geometry does not match public token batch")
    if require_bfloat16 and activations.dtype != torch.bfloat16:
        raise PublicActivationError("public activations must be BF16")
    if not activations.dtype.is_floating_point or not torch.isfinite(activations).all().item():
        raise PublicActivationError("public activations must be finite floating point")
    # Check one row at a time.  Converting the complete ~1 GiB BF16 train
    # tensor to FP32 or materializing a full masked_select result would create
    # an avoidable multi-GiB host temporary during preparation.
    for row in range(activations.shape[0]):
        active_count = int(token_batch.attention_mask[row].sum().item())
        if active_count < activations.shape[1] and not activations[row, active_count:].eq(0).all().item():
            raise PublicActivationError("padded activation rows must be zero")


def capture_public_prefix(
    prefix: ContiguousPublicPrefix,
    token_batch: PaddedTokenBatch,
    *,
    device: torch.device,
    batch_size: int = 8,
    hidden_size: int = HIDDEN_SIZE,
    resource_check: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Capture public cut activations in fixed right-padded batches."""

    validate_padded_token_batch(token_batch)
    if batch_size <= 0:
        raise PublicActivationError("capture batch size must be positive")
    if prefix.cut_depth != CUT_DEPTH:
        raise PublicActivationError(f"public prefix cut depth must be {CUT_DEPTH}")
    prefix.eval()
    pieces: list[torch.Tensor] = []
    for start in range(0, token_batch.token_ids.shape[0], batch_size):
        stop = min(start + batch_size, token_batch.token_ids.shape[0])
        inputs = token_batch.token_ids[start:stop].to(device=device, dtype=torch.long)
        try:
            hidden = prefix.forward_full(inputs)
        except (PublicPrefixError, RuntimeError) as exc:
            raise PublicActivationError(f"public prefix capture failed for rows {start}:{stop}") from exc
        if tuple(hidden.shape) != (stop - start, token_batch.token_ids.shape[1], hidden_size):
            raise PublicActivationError("public prefix returned unexpected activation geometry")
        hidden = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        active = token_batch.attention_mask[start:stop].to(torch.bool)
        hidden.masked_fill_(~active.unsqueeze(-1), 0)
        if not torch.isfinite(hidden.float()).all().item():
            raise PublicActivationError("public prefix produced non-finite activations")
        pieces.append(hidden)
        if resource_check is not None:
            resource_check()
    if not pieces:
        raise PublicActivationError("public prefix captured no rows")
    activations = torch.cat(pieces, dim=0).contiguous()
    validate_activation_tensor(activations, token_batch, hidden_size=hidden_size)
    return activations


def _declared_record_rows(plan: Mapping[str, Any], split: str) -> list[Mapping[str, Any]]:
    try:
        rows = plan["registration"][split]["records"]
    except (KeyError, TypeError) as exc:
        raise PublicActivationError(f"split {split!r} is missing from the public plan") from exc
    if not isinstance(rows, list) or not rows:
        raise PublicActivationError(f"split {split!r} has no declared records")
    if any(not isinstance(row, Mapping) for row in rows):
        raise PublicActivationError(f"split {split!r} contains malformed record metadata")
    ids = [str(row.get("record_id", "")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise PublicActivationError(f"split {split!r} contains duplicate or empty record IDs")
    return rows


def materialize_plan_split(
    plan: Mapping[str, Any],
    dataset: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    split: str,
    *,
    dataset_revision: str,
    maximum_tokens: int = HISTORICAL_MAX_TOKENS,
    minimum_full_tokens: int = HISTORICAL_MIN_FULL_TOKENS,
    expected_bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Recreate public token labels and verify every declared source fingerprint."""

    rows = _declared_record_rows(plan, split)
    sequences: list[list[int]] = []
    manifest_rows: list[dict[str, Any]] = []
    for declared in rows:
        try:
            index = int(declared["row_index"])
            if index < 0 or index >= len(dataset):
                raise PublicActivationError("declared dataset row is outside the public cache")
            actual = metadata_for_record(
                index,
                dataset[index],
                tokenizer,
                dataset_revision=dataset_revision,
                max_tokens=maximum_tokens,
                expected_bos_token_id=expected_bos_token_id,
            )
            rendered = historical_rendered_text(dataset[index], tokenizer)
            token_ids = _input_ids(tokenizer, rendered)[:maximum_tokens]
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise PublicActivationError(f"public {split} row metadata is malformed") from exc
        if actual.full_token_count < minimum_full_tokens:
            raise PublicActivationError(f"declared {split} row became shorter than the minimum")
        for key in ("row_index", "record_id", "rendered_sha256", "full_token_count", "post_bos_token_count"):
            if str(actual.as_dict()[key]) != str(declared.get(key)):
                raise PublicActivationError(f"public {split} source binding changed for {actual.record_id}: {key}")
        if not token_ids or token_ids[0] != expected_bos_token_id:
            raise PublicActivationError(f"public {split} row lost its BOS token")
        sequences.append([int(value) for value in token_ids])
        manifest_rows.append(dict(declared))
    return sequences, manifest_rows


def make_artifact_metadata(
    *,
    split: str,
    source_plan_sha256: str,
    source_arrow_sha256: str,
    source_info_sha256: str,
    model_id: str,
    model_revision: str,
    cut_depth: int,
    token_batch: PaddedTokenBatch,
    activations: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Build the small safetensors metadata contract for one public split."""

    validate_padded_token_batch(token_batch)
    validate_activation_tensor(activations, token_batch)
    ids = [str(row["record_id"]) for row in records]
    return {
        "schema": PUBLIC_ACTIVATION_SCHEMA,
        "task_id": "TRR-0004",
        "split": split,
        "public_truth_role": "public auxiliary fit/validation labels; not evaluator-private truth",
        "evaluator_private_truth_accessed": "false",
        "target_weights_accessed": "false",
        "current_token_alignment": "activations[record,position] predicts token_ids[record,position]",
        "bos_token_id": str(DEFAULT_BOS_TOKEN_ID),
        "pad_token_id": str(PAD_TOKEN_ID),
        "cut_depth": str(cut_depth),
        "hidden_size": str(activations.shape[-1]),
        "maximum_tokens": str(token_batch.token_ids.shape[1]),
        "record_count": str(token_batch.token_ids.shape[0]),
        "post_bos_positions": str(token_batch.post_bos_positions),
        "small_post_bos_positions": str(token_batch.small_positions),
        "record_ids_sha256": record_ids_sha256(ids),
        "token_ids_sha256": tensor_sha256(token_batch.token_ids),
        "attention_mask_sha256": tensor_sha256(token_batch.attention_mask),
        "position_ids_sha256": tensor_sha256(token_batch.position_ids),
        "post_bos_selector_small_sha256": tensor_sha256(token_batch.post_bos_selector_small),
        "post_bos_selector_large_sha256": tensor_sha256(token_batch.post_bos_selector_large),
        "activations_sha256": tensor_sha256(activations),
        "source_plan_sha256": source_plan_sha256,
        "source_dataset_arrow_sha256": source_arrow_sha256,
        "source_dataset_info_sha256": source_info_sha256,
        "model_id": model_id,
        "model_revision": model_revision,
        "runtime_components": "public tokenizer + public embedding/layers[0:4]; no target or candidate simulator",
    }


def save_public_artifact(
    path: Path,
    *,
    activations: torch.Tensor,
    token_batch: PaddedTokenBatch,
    metadata: Mapping[str, str],
) -> None:
    """Create one public activation artifact without overwriting an existing file."""

    if path.exists() or path.is_symlink():
        raise PublicActivationError(f"artifact is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_padded_token_batch(token_batch)
    validate_activation_tensor(activations, token_batch)
    save_file(
        {
            "activations": activations.detach().cpu().contiguous(),
            "token_ids": token_batch.token_ids.contiguous(),
            "attention_mask": token_batch.attention_mask.contiguous(),
            "position_ids": token_batch.position_ids.contiguous(),
            "post_bos_selector_small": token_batch.post_bos_selector_small.contiguous(),
            "post_bos_selector_large": token_batch.post_bos_selector_large.contiguous(),
        },
        str(path),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )
