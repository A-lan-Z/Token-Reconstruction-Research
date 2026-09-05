#!/usr/bin/env python3
"""Capture the TRR-0005 enriched public prefix and build decoder manifests.

The original-like arm reuses the registered TRR-0004 public activation file
read-only.  The coverage arm is different: its complete constructed token
sequences are sent through the pinned public embedding and first four decoder
layers with :func:`capture_public_prefix`.  The script also binds the shared
TRR-0004 Alpaca24+Pile24 public validation slice and the normalized public
embedding table into the compact manifest schema consumed by the TRR-0005
joint decoder.

``--mode manifest`` performs header, token/mask, record, and validation checks
without a model load.  ``--mode capture`` additionally runs the fixed TRR4
batch-8 x 192 public-prefix path for the enriched arm.  Capture is therefore
explicit and cannot happen during an ordinary metadata review.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open

from token_reconstruction.public_activation import (
    CUT_DEPTH,
    HIDDEN_SIZE,
    PAD_TOKEN_ID,
    PUBLIC_ACTIVATION_SCHEMA,
    PaddedTokenBatch,
    capture_public_prefix,
    make_artifact_metadata,
    pad_public_token_sequences,
    record_ids_sha256,
    save_public_artifact,
    tensor_sha256,
    validate_padded_token_batch,
)

# Reuse the TRR4 model loader and qualifier verbatim.  This keeps model
# snapshot binding and the fixed public-prefix qualification on the registered
# capture path instead of introducing a second implementation.
from trr0004_prepare_public_activations import (  # type: ignore
    MODEL_ID,
    MODEL_REVISION,
    _enforce_resource_ceiling,
    _load_public_prefix,
    _qualify_public_prefix_padding,
    _resource_preflight,
)


TASK_ID = "TRR-0005"
DATA_SCHEMA = "token-reconstruction.trr0005-public-fit-data.v1"
ADAPTER_SCHEMA = "token-reconstruction.trr0005-public-activation-manifest-adapter.v1"
CORPUS_SCHEMA = "token-reconstruction.trr0005-public-corpus-plan.v1"
BOS_TOKEN_ID = 128000
VOCAB_SIZE = 128256
MAXIMUM_TOKENS = 192
CAPTURE_BATCH_RECORDS = 8
COMMON_VALIDATION_RECORDS = 48
COMMON_VALIDATION_POST_BOS = 3133
COMMON_STYLE_COUNTS = {"alpaca": 24, "pile": 24}
DEFAULT_MAX_RESERVED_GPU_BYTES = 8 * 1024**3
DEFAULT_MAX_HOST_RSS_BYTES = 16 * 1024**3
DEFAULT_MIN_FREE_GPU_BYTES = 8 * 1024**3


class ActivationPreparationError(RuntimeError):
    """Raised when a public activation or manifest binding changes."""


_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    path = path.expanduser().resolve()
    stat = path.stat()
    cache_key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_HASH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = value
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ActivationPreparationError(f"{label} must be a regular file: {path}")
    return path


def _json(path: Path, *, label: str) -> Mapping[str, Any]:
    path = _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivationPreparationError(f"cannot parse {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ActivationPreparationError(f"{label} must contain an object")
    return value


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _header(path: Path, key: str, *, label: str) -> dict[str, Any]:
    """Read only a safetensors header and return shape/dtype plus file hash."""

    path = _regular_file(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in set(handle.keys()):
                raise ActivationPreparationError(f"{label} has no {key!r} tensor")
            view = handle.get_slice(key)
            dtype = str(view.get_dtype())
            shape = list(view.get_shape())
    except ActivationPreparationError:
        raise
    except Exception as exc:
        raise ActivationPreparationError(f"cannot inspect {label}: {path}") from exc
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "tensor_key": key,
        "shape": shape,
        "dtype": dtype,
    }


def _tensor(path: Path, key: str, *, label: str) -> torch.Tensor:
    path = _regular_file(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in set(handle.keys()):
                raise ActivationPreparationError(f"{label} has no {key!r} tensor")
            return handle.get_tensor(key).contiguous()
    except ActivationPreparationError:
        raise
    except Exception as exc:
        raise ActivationPreparationError(f"cannot load {label}: {path}") from exc


def _relative(path: Path, *, root: Path) -> str:
    return os.path.relpath(path, root)


def _git_source_records(root: Path, script_path: Path) -> dict[str, Any]:
    paths = {
        "runner": script_path,
        "activation_module": root / "src/token_reconstruction/public_activation.py",
        "public_prefix_module": root / "src/token_reconstruction/public_prefix.py",
        "corpus_module": root / "src/token_reconstruction/trr0005_public_corpus.py",
        "corpus_preparation_script": root / "scripts/trr0005_prepare_public_corpus.py",
    }
    return {
        name: {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for name, path in paths.items()
        if path.is_file() and not path.is_symlink()
    }


def _load_records(path: Path, *, label: str) -> list[dict[str, Any]]:
    payload = _json(path, label=label)
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ActivationPreparationError(f"{label} has no records list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str):
            raise ActivationPreparationError(f"{label} record {index} has no record_id")
        record_id = str(row["record_id"])
        if not record_id or record_id in seen:
            raise ActivationPreparationError(f"{label} has duplicate or empty record IDs")
        seen.add(record_id)
        result.append(dict(row))
    return result


def _plan(path: Path) -> Mapping[str, Any]:
    payload = _json(path, label="TRR-0005 corpus plan")
    if payload.get("schema") != CORPUS_SCHEMA or payload.get("task_id") != TASK_ID:
        raise ActivationPreparationError("TRR-0005 corpus plan schema or task ID changed")
    if payload.get("status") != "PREPARED_PUBLIC_DATA_NO_MODEL_FORWARD":
        raise ActivationPreparationError("corpus plan is not the accepted no-forward preparation")
    design = payload.get("design")
    if not isinstance(design, Mapping):
        raise ActivationPreparationError("corpus plan has no design")
    if (
        int(design.get("record_count", -1)) != 1200
        or int(design.get("stored_rows_including_bos", -1)) != 125571
        or int(design.get("post_bos_positions", -1)) != 124371
        or int(design.get("max_sequence_length", -1)) != MAXIMUM_TOKENS
    ):
        raise ActivationPreparationError("TRR-0005 corpus geometry changed")
    exposure = payload.get("joint_training_exposure")
    if not isinstance(exposure, Mapping) or {
        "batch_size": int(exposure.get("batch_size", -1)) if isinstance(exposure, Mapping) else -1,
        "steps": int(exposure.get("steps", -1)) if isinstance(exposure, Mapping) else -1,
        "seed": int(exposure.get("seed", -1)) if isinstance(exposure, Mapping) else -1,
        "post_bos_positions": int(exposure.get("post_bos_positions", -1)) if isinstance(exposure, Mapping) else -1,
    } != {"batch_size": 512, "steps": 3000, "seed": 4005, "post_bos_positions": 124371}:
        raise ActivationPreparationError("joint exposure metadata changed")
    return payload


def _arm_records(plan: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    arms = plan.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
        raise ActivationPreparationError(f"corpus plan has no {arm} arm")
    rows = arms[arm].get("records")
    if not isinstance(rows, list) or len(rows) != 1200:
        raise ActivationPreparationError(f"{arm} arm record count changed")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str):
            raise ActivationPreparationError(f"{arm} record {index} is malformed")
        record_id = str(row["record_id"])
        if record_id in seen:
            raise ActivationPreparationError(f"{arm} record IDs are duplicated")
        if int(row.get("slot", -1)) != index:
            raise ActivationPreparationError(f"{arm} record order changed at slot {index}")
        seen.add(record_id)
        result.append(dict(row))
    return result


def _batch_from_artifact(path: Path, *, label: str) -> tuple[PaddedTokenBatch, torch.Tensor, torch.Tensor]:
    token_ids = _tensor(path, "token_ids", label=f"{label} token IDs")
    attention_mask = _tensor(path, "attention_mask", label=f"{label} attention mask")
    if tuple(token_ids.shape) != (1200, MAXIMUM_TOKENS) or tuple(attention_mask.shape) != tuple(token_ids.shape):
        raise ActivationPreparationError(f"{label} token/mask geometry changed")
    if token_ids.dtype not in (torch.int32, torch.int64) or attention_mask.dtype != torch.uint8:
        raise ActivationPreparationError(f"{label} token/mask dtypes changed")
    if token_ids[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise ActivationPreparationError(f"{label} rows lost BOS")
    if attention_mask.lt(0).any().item() or attention_mask.gt(1).any().item():
        raise ActivationPreparationError(f"{label} mask is not binary")
    sequences: list[list[int]] = []
    for row in range(token_ids.shape[0]):
        active_count = int(attention_mask[row].sum().item())
        if active_count <= 1:
            raise ActivationPreparationError(f"{label} row {row} has no post-BOS position")
        if not attention_mask[row, :active_count].eq(1).all().item() or not attention_mask[row, active_count:].eq(0).all().item():
            raise ActivationPreparationError(f"{label} mask is not contiguous right-padding")
        if not token_ids[row, active_count:].eq(PAD_TOKEN_ID).all().item():
            raise ActivationPreparationError(f"{label} padding labels changed")
        sequences.append([int(value) for value in token_ids[row, :active_count].tolist()])
    batch = pad_public_token_sequences(sequences, maximum_tokens=MAXIMUM_TOKENS)
    validate_padded_token_batch(batch, maximum_tokens=MAXIMUM_TOKENS)
    if not torch.equal(batch.token_ids, token_ids.to(dtype=batch.token_ids.dtype)):
        raise ActivationPreparationError(f"{label} token ordering changed during batch reconstruction")
    if not torch.equal(batch.attention_mask, attention_mask):
        raise ActivationPreparationError(f"{label} mask ordering changed during batch reconstruction")
    return batch, token_ids, attention_mask


def _check_plan_lengths(rows: Sequence[Mapping[str, Any]], mask: torch.Tensor, *, label: str) -> None:
    if len(rows) != int(mask.shape[0]):
        raise ActivationPreparationError(f"{label} records do not match mask rows")
    for index, row in enumerate(rows):
        expected = int(mask[index].sum().item()) - 1
        declared = int(row.get("target_post_bos_token_count", row.get("post_bos_token_count", -1)))
        if declared != expected:
            raise ActivationPreparationError(f"{label} row {index} length disagrees with mask")


def _check_activation_token_binding(
    path: Path,
    expected_tokens: torch.Tensor,
    expected_mask: torch.Tensor,
    *,
    label: str,
) -> None:
    actual_tokens = _tensor(path, "token_ids", label=f"{label} token IDs")
    actual_mask = _tensor(path, "attention_mask", label=f"{label} attention mask")
    if not torch.equal(actual_tokens.to(dtype=expected_tokens.dtype), expected_tokens):
        raise ActivationPreparationError(f"{label} token IDs do not match the prepared sequence artifact")
    if not torch.equal(actual_mask.to(dtype=expected_mask.dtype), expected_mask):
        raise ActivationPreparationError(f"{label} attention mask does not match the prepared sequence artifact")


def _sanitized_fit_records(rows: Sequence[Mapping[str, Any]], mask: torch.Tensor) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        active = int(mask[index].sum().item())
        allowed = {
            "record_id", "source_record_id", "dataset_key", "domain", "style", "slot", "synthetic",
            "source_full_token_count", "target_full_token_count", "target_post_bos_token_count",
            "rendered_sha256", "replacement_count", "replacement_positions", "replacement_token_ids",
        }
        item = {key: row[key] for key in allowed if key in row}
        item.update(
            {
                "full_token_count": active,
                "post_bos_token_count": active - 1,
                "active_token_count": active,
                "padded_length": int(mask.shape[1]),
            }
        )
        result.append(item)
    return result


def _common_validation(
    artifact_path: Path, records_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    records = _load_records(records_path, label="common validation records")
    if len(records) != COMMON_VALIDATION_RECORDS:
        raise ActivationPreparationError("common validation record count changed")
    style_counts = Counter()
    style_positions: Counter[str] = Counter()
    for index, row in enumerate(records):
        style = row.get("style")
        if style not in COMMON_STYLE_COUNTS:
            raise ActivationPreparationError(f"common validation record {index} has no declared Alpaca/Pile style")
        style_counts[str(style)] += 1
        post_count = int(row.get("post_bos_token_count", -1))
        if post_count <= 0:
            raise ActivationPreparationError(f"common validation record {index} has no post-BOS count")
        style_positions[str(style)] += post_count
    if dict(style_counts) != COMMON_STYLE_COUNTS:
        raise ActivationPreparationError(f"common validation styles changed: {dict(style_counts)}")
    validation_x = _tensor(artifact_path, "activations", label="common validation activations")
    validation_y = _tensor(artifact_path, "token_ids", label="common validation labels")
    validation_mask = _tensor(artifact_path, "attention_mask", label="common validation mask")
    if tuple(validation_x.shape) != (48, MAXIMUM_TOKENS, HIDDEN_SIZE):
        raise ActivationPreparationError("common validation activation geometry changed")
    if tuple(validation_y.shape) != (48, MAXIMUM_TOKENS) or tuple(validation_mask.shape) != tuple(validation_y.shape):
        raise ActivationPreparationError("common validation label/mask geometry changed")
    if validation_x.dtype != torch.bfloat16 or validation_y.dtype not in (torch.int32, torch.int64) or validation_mask.dtype != torch.uint8:
        raise ActivationPreparationError("common validation dtypes changed")
    if validation_y[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise ActivationPreparationError("common validation rows lost BOS")
    if validation_mask.lt(0).any().item() or validation_mask.gt(1).any().item():
        raise ActivationPreparationError("common validation mask is not binary")
    for row in range(COMMON_VALIDATION_RECORDS):
        active_count = int(validation_mask[row].sum().item())
        if active_count <= 1:
            raise ActivationPreparationError("common validation row has no post-BOS position")
        if not validation_mask[row, :active_count].eq(1).all().item() or not validation_mask[row, active_count:].eq(0).all().item():
            raise ActivationPreparationError("common validation mask is not contiguous right-padding")
        if not validation_y[row, active_count:].eq(PAD_TOKEN_ID).all().item():
            raise ActivationPreparationError("common validation padding labels changed")
    actual_positions = int(validation_mask[:, 1:].sum().item())
    declared_positions = sum(int(row["post_bos_token_count"]) for row in records)
    if actual_positions != COMMON_VALIDATION_POST_BOS or declared_positions != COMMON_VALIDATION_POST_BOS:
        raise ActivationPreparationError("common validation post-BOS positions changed")
    if dict(style_positions) != {"alpaca": 2197, "pile": 936}:
        raise ActivationPreparationError(f"common validation style positions changed: {dict(style_positions)}")
    groups = [str(row["style"]) for row in records]
    grouping = {
        "record_count": len(records),
        "post_bos_positions": actual_positions,
        "style_counts": dict(sorted(style_counts.items())),
        "post_bos_positions_by_style": dict(sorted(style_positions.items())),
        "groups_in_record_order": groups,
        "selection_metric": "unweighted mean of per-style post-BOS token accuracies",
        "style_mapping": "validation record style field is authoritative; Alpaca24 then Pile24",
        "record_ids_sha256": record_ids_sha256([str(row["record_id"]) for row in records]),
    }
    resource_info = {
        "artifact": _header(artifact_path, "activations", label="common validation activations"),
        "truth": _header(artifact_path, "token_ids", label="common validation labels"),
        "mask": _header(artifact_path, "attention_mask", label="common validation mask"),
        "records": {
            "path": str(records_path),
            "bytes": int(records_path.stat().st_size),
            "sha256": _sha256_file(records_path),
        },
    }
    return resource_info, records, grouping


def _coverage_diagnostics(
    original_tokens: torch.Tensor,
    original_mask: torch.Tensor,
    enriched_tokens: torch.Tensor,
    enriched_mask: torch.Tensor,
    enriched_rows: Sequence[Mapping[str, Any]],
    selected_controlled_ids: Sequence[int],
) -> dict[str, Any]:
    def post_set(tokens: torch.Tensor, mask: torch.Tensor, rows: Sequence[int] | None = None) -> set[int]:
        if rows is None:
            values = tokens[:, 1:][mask[:, 1:].to(torch.bool)]
        else:
            values = tokens[list(rows), 1:][mask[list(rows), 1:].to(torch.bool)]
        return {int(value) for value in values.tolist()}

    original_set = post_set(original_tokens, original_mask)
    enriched_set = post_set(enriched_tokens, enriched_mask)
    natural_rows = [index for index, row in enumerate(enriched_rows) if not bool(row.get("synthetic", False))]
    controlled_rows = [index for index, row in enumerate(enriched_rows) if bool(row.get("synthetic", False))]
    natural_set = post_set(enriched_tokens, enriched_mask, natural_rows)
    controlled_context_set = post_set(enriched_tokens, enriched_mask, controlled_rows)
    selected = {int(value) for value in selected_controlled_ids}
    replacement_ids: set[int] = set()
    replacement_occurrences = 0
    for row in enriched_rows:
        if bool(row.get("synthetic", False)):
            values = row.get("replacement_token_ids", ())
            if not isinstance(values, Sequence):
                raise ActivationPreparationError("controlled replacement token metadata is malformed")
            replacement_ids.update(int(value) for value in values)
            replacement_occurrences += len(values)
    return {
        "primary_sets_exclude_bos": True,
        "original_like_distinct_post_bos": len(original_set),
        "enriched_distinct_post_bos": len(enriched_set),
        "original_enriched_overlap_distinct_post_bos": len(original_set & enriched_set),
        "newly_covered_by_enriched_distinct_post_bos": len(enriched_set - original_set),
        "lost_from_original_distinct_post_bos": len(original_set - enriched_set),
        "natural_only": {
            "records": len(natural_rows),
            "post_bos_positions": int(enriched_mask[natural_rows, 1:].sum().item()),
            "distinct_post_bos": len(natural_set),
        },
        "controlled_supplement": {
            "records": len(controlled_rows),
            "post_bos_positions": int(enriched_mask[controlled_rows, 1:].sum().item()),
            "constructed_context_distinct_post_bos": len(controlled_context_set),
            "selected_ids": len(selected),
            "replacement_occurrences": replacement_occurrences,
            "replacement_distinct_ids": len(replacement_ids),
            "replacement_ids_in_selected": len(replacement_ids & selected),
        },
        "descriptive_only": True,
        "selection_changed_after_preparation": False,
    }


def _artifact_metadata(
    *,
    split: str,
    plan_path: Path,
    token_batch: PaddedTokenBatch,
    activations: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, str]:
    metadata = make_artifact_metadata(
        split=split,
        source_plan_sha256=_sha256_file(plan_path),
        source_arrow_sha256="PUBLIC_MIX_SOURCES_BOUND_IN_CORPUS_PLAN",
        source_info_sha256="PUBLIC_MIX_SOURCES_BOUND_IN_CORPUS_PLAN",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        cut_depth=CUT_DEPTH,
        token_batch=token_batch,
        activations=activations,
        records=records,
    )
    metadata.update(
        {
            "task_id": TASK_ID,
            "arm": arm,
            "schema": PUBLIC_ACTIVATION_SCHEMA,
            "capture_contract": "full constructed public token sequence through public embedding/layers[0:4]",
            "coverage_scope": "post_bos_only",
        }
    )
    return metadata


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise ActivationPreparationError(f"output is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path.resolve()), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _manifest(
    *,
    root: Path,
    manifest_path: Path,
    arm: str,
    fit_artifact: Path,
    fit_records_path: Path,
    common_artifact: Path,
    common_records_path: Path,
    embedding_path: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    grouping: Mapping[str, Any],
    capture_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    fit_x = _header(fit_artifact, "activations", label=f"{arm} fit activations")
    fit_y = _header(fit_artifact, "token_ids", label=f"{arm} fit labels")
    fit_mask = _header(fit_artifact, "attention_mask", label=f"{arm} fit mask")
    fit_token_tensor = _tensor(fit_artifact, "token_ids", label=f"{arm} fit labels")
    fit_mask_tensor = _tensor(fit_artifact, "attention_mask", label=f"{arm} fit mask")
    if fit_x["shape"] != [1200, MAXIMUM_TOKENS, HIDDEN_SIZE] or fit_x["dtype"] != "BF16":
        raise ActivationPreparationError(f"{arm} fit activation geometry/dtype changed")
    if fit_y["shape"] != [1200, MAXIMUM_TOKENS] or fit_mask["shape"] != [1200, MAXIMUM_TOKENS]:
        raise ActivationPreparationError(f"{arm} fit label/mask geometry changed")
    common_x = _header(common_artifact, "activations", label="common validation activations")
    common_y = _header(common_artifact, "token_ids", label="common validation labels")
    common_mask = _header(common_artifact, "attention_mask", label="common validation mask")
    if common_x["shape"] != [COMMON_VALIDATION_RECORDS, MAXIMUM_TOKENS, HIDDEN_SIZE] or common_x["dtype"] != "BF16":
        raise ActivationPreparationError("common validation activation geometry/dtype changed")
    if common_y["shape"] != [COMMON_VALIDATION_RECORDS, MAXIMUM_TOKENS] or common_mask["shape"] != [COMMON_VALIDATION_RECORDS, MAXIMUM_TOKENS]:
        raise ActivationPreparationError("common validation label/mask geometry changed")
    embedding = _header(embedding_path, "embeddings", label="shared normalized public embedding table")
    if embedding["shape"] != [VOCAB_SIZE, HIDDEN_SIZE] or embedding["dtype"] != "F32":
        raise ActivationPreparationError("shared normalized public embedding geometry/dtype changed")
    records = _load_records(fit_records_path, label=f"{arm} fit records")
    manifest = {
        "schema": DATA_SCHEMA,
        "task_id": TASK_ID,
        "distribution": arm,
        "arm": arm,
        "layout": "padded_records",
        "bos_token_id": BOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "embedding_table_normalized": True,
        "alignment": {
            "mode": "current_token",
            "observation_index": "i",
            "label_index": "i",
            "bos_position": 0,
            "scored_positions": "post_bos",
        },
        "geometry": {
            "fit": fit_x["shape"],
            "validation": common_x["shape"],
            "post_bos_fit": 124371,
            "post_bos_validation": COMMON_VALIDATION_POST_BOS,
            "length_vector_digest": plan["design"]["length_vector_digest"],
            "fit_token_ids_sha256": tensor_sha256(fit_token_tensor),
            "fit_attention_mask_sha256": tensor_sha256(fit_mask_tensor),
            "joint_training_exposure": plan["joint_training_exposure"],
        },
        "resources": {
            "fit_observations": {**fit_x, "path": _relative(fit_artifact, root=root)},
            "fit_truth": {**fit_y, "path": _relative(fit_artifact, root=root)},
            "fit_valid_mask": {**fit_mask, "path": _relative(fit_artifact, root=root)},
            "fit_records": {
                "path": _relative(fit_records_path, root=root),
                "bytes": int(fit_records_path.stat().st_size),
                "sha256": _sha256_file(fit_records_path),
            },
            "validation_observations": {**common_x, "path": _relative(common_artifact, root=root)},
            "validation_truth": {**common_y, "path": _relative(common_artifact, root=root)},
            "validation_valid_mask": {**common_mask, "path": _relative(common_artifact, root=root)},
            "validation_records": {
                "path": _relative(common_records_path, root=root),
                "bytes": int(common_records_path.stat().st_size),
                "sha256": _sha256_file(common_records_path),
            },
            "embedding_table": {**embedding, "path": _relative(embedding_path, root=root)},
        },
        "validation_grouping": grouping,
        "coverage_diagnostics": diagnostics,
        "source": {
            "adapter_schema": ADAPTER_SCHEMA,
            "adapter_script": {
                "path": str(Path(__file__).resolve()),
                "bytes": int(Path(__file__).stat().st_size),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "corpus_plan": {
                "path": str(plan_path),
                "bytes": int(plan_path.stat().st_size),
                "sha256": _sha256_file(plan_path),
            },
            "original_fit_reuse": "TRR-0004 public activation H reused read-only for original_like_alpaca_v1",
            "enriched_capture": capture_receipt.get("enriched_capture", {}),
            "shared_cpu_embedding_table": {
                "path": str(embedding_path),
                "sha256": _sha256_file(embedding_path),
                "same_resource_for_both_arms": True,
            },
            "public_only": True,
            "target_weights_accessed": False,
            "evaluator_private_truth_accessed": False,
        },
        "record_ids_sha256": record_ids_sha256([str(row["record_id"]) for row in records]),
        "fit_record_count": len(records),
        "generated_at_utc": _utc_now(),
    }
    return manifest


def _device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise ActivationPreparationError("CUDA requested but unavailable")
    return torch.device(raw)


def _capture_enriched(
    *,
    args: argparse.Namespace,
    root: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    enriched_tokens_path: Path,
    enriched_records: Sequence[Mapping[str, Any]],
    output_artifact: Path,
) -> dict[str, Any]:
    device = _device(args.device)
    if args.batch_records != CAPTURE_BATCH_RECORDS or args.cut_depth != CUT_DEPTH:
        raise ActivationPreparationError("capture geometry must remain fixed at batch 8 and cut depth 4")
    if device.type != "cuda":
        raise ActivationPreparationError("TRR-0005 capture requires the explicit public GPU path")
    started = _utc_now()
    started_clock = time.perf_counter()
    if output_artifact.exists() or output_artifact.is_symlink():
        raise ActivationPreparationError(f"enriched activation output is create-only: {output_artifact}")
    batch, _, mask = _batch_from_artifact(enriched_tokens_path, label="enriched constructed tokens")
    _check_plan_lengths(enriched_records, mask, label="enriched corpus plan")
    guard = _resource_preflight(
        device,
        min_free_gpu_bytes=int(args.min_free_gpu_gib * 1024**3),
        max_reserved_gpu_bytes=int(args.max_reserved_gpu_gib * 1024**3),
        max_host_rss_bytes=int(args.max_host_rss_gib * 1024**3),
    )
    prefix, model_snapshot, model_config = _load_public_prefix(
        args.model.expanduser().resolve(), device=device, cut_depth=CUT_DEPTH
    )
    qualification = _qualify_public_prefix_padding(
        prefix,
        batch,
        device=device,
        batch_size=args.batch_records,
    )
    qualification["measured_peak_after_phase"] = _enforce_resource_ceiling(
        device,
        max_reserved_gpu_bytes=int(args.max_reserved_gpu_gib * 1024**3),
        max_host_rss_bytes=int(args.max_host_rss_gib * 1024**3),
    )
    activations = capture_public_prefix(
        prefix,
        batch,
        device=device,
        batch_size=args.batch_records,
        resource_check=lambda: _enforce_resource_ceiling(
            device,
            max_reserved_gpu_bytes=int(args.max_reserved_gpu_gib * 1024**3),
            max_host_rss_bytes=int(args.max_host_rss_gib * 1024**3),
        ),
    )
    final_peak = _enforce_resource_ceiling(
        device,
        max_reserved_gpu_bytes=int(args.max_reserved_gpu_gib * 1024**3),
        max_host_rss_bytes=int(args.max_host_rss_gib * 1024**3),
    )
    metadata = _artifact_metadata(
        split="fit_coverage_mix_v1",
        plan_path=plan_path,
        token_batch=batch,
        activations=activations,
        records=enriched_records,
        arm="coverage_mix_v1",
    )
    save_public_artifact(output_artifact, activations=activations, token_batch=batch, metadata=metadata)
    return {
        "status": "ENRICHED_PUBLIC_ACTIVATION_CAPTURE_COMPLETE",
        "started_utc": started,
        "ended_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started_clock,
        "argv": [str(value) for value in sys.argv],
        "device": str(device),
        "model_snapshot": model_snapshot,
        "model_config": model_config,
        "qualification": qualification,
        "resource_preflight": guard,
        "peak_memory": final_peak,
        "source_code": _git_source_records(root, Path(__file__).resolve()),
        "enriched_capture": {
            "path": str(output_artifact.resolve()),
            "bytes": int(output_artifact.stat().st_size),
            "sha256": _sha256_file(output_artifact),
            "actual_public_forward": "ContiguousPublicPrefix.forward_full over every complete constructed sequence",
            "candidate_simulations": 0,
            "private_truth_accessed": False,
            "target_weights_accessed": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("manifest", "capture"), default="manifest")
    parser.add_argument("--corpus-plan", type=Path, required=True)
    parser.add_argument("--original-artifact", type=Path, required=True)
    parser.add_argument("--original-records", type=Path, required=True)
    parser.add_argument("--common-validation-artifact", type=Path, required=True)
    parser.add_argument("--common-validation-records", type=Path, required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--constructed-token-artifact", type=Path)
    parser.add_argument("--enriched-activation-artifact", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-records", type=int, default=CAPTURE_BATCH_RECORDS)
    parser.add_argument("--cut-depth", type=int, default=CUT_DEPTH)
    parser.add_argument("--min-free-gpu-gib", type=float, default=8.0)
    parser.add_argument("--max-reserved-gpu-gib", type=float, default=8.0)
    parser.add_argument("--max-host-rss-gib", type=float, default=16.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    plan_path = args.corpus_plan.expanduser().resolve()
    original_artifact = args.original_artifact.expanduser().resolve()
    original_records_path = args.original_records.expanduser().resolve()
    common_artifact = args.common_validation_artifact.expanduser().resolve()
    common_records_path = args.common_validation_records.expanduser().resolve()
    embedding_path = args.embedding_table.expanduser().resolve()
    plan = _plan(plan_path)
    original_rows = _arm_records(plan, "original_like_alpaca_v1")
    enriched_rows = _arm_records(plan, "coverage_mix_v1")
    _original_batch, original_tokens, original_mask = _batch_from_artifact(original_artifact, label="original TRR4 fit artifact")
    _check_plan_lengths(original_rows, original_mask, label="original corpus plan")
    enriched_token_entry = plan["arms"]["coverage_mix_v1"]["token_artifact"]
    if not isinstance(enriched_token_entry, Mapping):
        raise ActivationPreparationError("coverage arm has no constructed token artifact")
    enriched_tokens_path = (
        args.constructed_token_artifact.expanduser().resolve()
        if args.constructed_token_artifact is not None
        else Path(str(enriched_token_entry["path"])).expanduser().resolve()
    )
    _enriched_batch, enriched_tokens, enriched_mask = _batch_from_artifact(
        enriched_tokens_path, label="enriched constructed tokens"
    )
    _check_plan_lengths(enriched_rows, enriched_mask, label="enriched corpus plan")
    if not torch.equal(original_mask.sum(dim=1), enriched_mask.sum(dim=1)):
        raise ActivationPreparationError("original and enriched ordered length vectors differ")
    if [str(row["record_id"]) for row in original_rows] == [str(row["record_id"]) for row in enriched_rows]:
        raise ActivationPreparationError("enriched arm did not change record identities")
    selected = plan.get("controlled_token_selection", {}).get("selected_token_ids", [])
    if not isinstance(selected, list) or len(selected) != 2000:
        raise ActivationPreparationError("controlled token selection count changed")
    diagnostics = _coverage_diagnostics(
        original_tokens,
        original_mask,
        enriched_tokens,
        enriched_mask,
        enriched_rows,
        [int(value) for value in selected],
    )
    common_resources, _common_records, grouping = _common_validation(common_artifact, common_records_path)
    if args.mode == "capture":
        if args.model is None:
            raise ActivationPreparationError("--model is required in capture mode")
        output_root = args.output_root.expanduser().resolve()
        if output_root.exists() or output_root.is_symlink():
            raise ActivationPreparationError(f"capture output root is create-only: {output_root}")
        output_root.mkdir(parents=True)
        enriched_artifact = (
            args.enriched_activation_artifact.expanduser().resolve()
            if args.enriched_activation_artifact is not None
            else output_root / "enriched_fit_cut4.safetensors"
        )
        capture_receipt = _capture_enriched(
            args=args,
            root=root,
            plan_path=plan_path,
            plan=plan,
            enriched_tokens_path=enriched_tokens_path,
            enriched_records=enriched_rows,
            output_artifact=enriched_artifact,
        )
    else:
        if args.enriched_activation_artifact is None:
            raise ActivationPreparationError("--enriched-activation-artifact is required in manifest mode")
        enriched_artifact = args.enriched_activation_artifact.expanduser().resolve()
        enriched_header = _header(enriched_artifact, "activations", label="enriched public activations")
        if enriched_header["shape"] != [1200, MAXIMUM_TOKENS, HIDDEN_SIZE] or enriched_header["dtype"] != "BF16":
            raise ActivationPreparationError("enriched public activation geometry/dtype changed")
        _check_activation_token_binding(
            enriched_artifact,
            enriched_tokens,
            enriched_mask,
            label="enriched public activations",
        )
        capture_receipt = {
            "status": "MANIFEST_ONLY_NO_MODEL_FORWARD",
            "enriched_capture": {
                "path": str(enriched_artifact),
                "actual_public_forward": "pre-existing artifact bound read-only; no forward in manifest mode",
            },
            "private_truth_accessed": False,
            "target_weights_accessed": False,
        }
        output_root = args.output_root.expanduser().resolve()
        if output_root.exists() or output_root.is_symlink():
            raise ActivationPreparationError(f"manifest output root is create-only: {output_root}")
        output_root.mkdir(parents=True)
    original_records_out = output_root / "original_fit_records.json"
    enriched_records_out = output_root / "enriched_fit_records.json"
    _write_json(original_records_out, {"records": _sanitized_fit_records(original_rows, original_mask)})
    _write_json(enriched_records_out, {"records": _sanitized_fit_records(enriched_rows, enriched_mask)})
    original_manifest_path = output_root / "original_manifest.json"
    enriched_manifest_path = output_root / "enriched_manifest.json"
    original_manifest = _manifest(
        root=output_root,
        manifest_path=original_manifest_path,
        arm="original_like_alpaca_v1",
        fit_artifact=original_artifact,
        fit_records_path=original_records_out,
        common_artifact=common_artifact,
        common_records_path=common_records_path,
        embedding_path=embedding_path,
        plan_path=plan_path,
        plan=plan,
        diagnostics=diagnostics,
        grouping=grouping,
        capture_receipt=capture_receipt,
    )
    enriched_manifest = _manifest(
        root=output_root,
        manifest_path=enriched_manifest_path,
        arm="coverage_mix_v1",
        fit_artifact=enriched_artifact,
        fit_records_path=enriched_records_out,
        common_artifact=common_artifact,
        common_records_path=common_records_path,
        embedding_path=embedding_path,
        plan_path=plan_path,
        plan=plan,
        diagnostics=diagnostics,
        grouping=grouping,
        capture_receipt=capture_receipt,
    )
    _write_json(original_manifest_path, original_manifest)
    _write_json(enriched_manifest_path, enriched_manifest)
    receipt = {
        "schema": ADAPTER_SCHEMA,
        "task_id": TASK_ID,
        "status": capture_receipt["status"],
        "execution": {
            "argv": [str(value) for value in sys.argv],
            "started_utc": _utc_now(),
            "ended_utc": _utc_now(),
            "git_commit": _git_commit(root),
            "python": sys.version,
            "platform": platform.platform(),
            "resource_usage": {
                "user_cpu_seconds": float(resource.getrusage(resource.RUSAGE_SELF).ru_utime),
                "system_cpu_seconds": float(resource.getrusage(resource.RUSAGE_SELF).ru_stime),
                "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            },
        },
        "sources": {
            "corpus_plan": {"path": str(plan_path), "bytes": int(plan_path.stat().st_size), "sha256": _sha256_file(plan_path)},
            "original_artifact": _header(original_artifact, "activations", label="original fit activations"),
            "constructed_tokens": _header(enriched_tokens_path, "token_ids", label="constructed public tokens"),
            "common_validation": common_resources,
            "shared_embedding": _header(embedding_path, "embeddings", label="shared normalized public embedding table"),
        },
        "geometry": {
            "ordered_length_vector_match": True,
            "original_fit": [1200, MAXIMUM_TOKENS, HIDDEN_SIZE],
            "enriched_fit": [1200, MAXIMUM_TOKENS, HIDDEN_SIZE],
            "common_validation": [48, MAXIMUM_TOKENS, HIDDEN_SIZE],
            "post_bos_fit": 124371,
            "post_bos_validation": COMMON_VALIDATION_POST_BOS,
        },
        "validation_grouping": grouping,
        "coverage_diagnostics": diagnostics,
        "outputs": {
            "original_manifest": {"path": str(original_manifest_path), "bytes": original_manifest_path.stat().st_size, "sha256": _sha256_file(original_manifest_path)},
            "enriched_manifest": {"path": str(enriched_manifest_path), "bytes": enriched_manifest_path.stat().st_size, "sha256": _sha256_file(enriched_manifest_path)},
            "original_records": {"path": str(original_records_out), "bytes": original_records_out.stat().st_size, "sha256": _sha256_file(original_records_out)},
            "enriched_records": {"path": str(enriched_records_out), "bytes": enriched_records_out.stat().st_size, "sha256": _sha256_file(enriched_records_out)},
            "enriched_artifact": {"path": str(enriched_artifact), "bytes": enriched_artifact.stat().st_size, "sha256": _sha256_file(enriched_artifact)},
        },
        "capture": capture_receipt,
        "source_code": _git_source_records(root, Path(__file__).resolve()),
        "access_contract": {
            "public_prefix_only": True,
            "target_weights_accessed": False,
            "evaluator_private_truth_accessed": False,
            "future_holdout_rows_accessed": False,
            "original_h_reused_read_only": True,
            "enriched_full_sequence_forward_required": True,
        },
    }
    receipt_path = output_root / "capture_manifest_receipt.json"
    _write_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run(args)
    except (ActivationPreparationError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": receipt["status"],
        "output_root": str(args.output_root.expanduser().resolve()),
        "original_manifest": str(args.output_root.expanduser().resolve() / "original_manifest.json"),
        "enriched_manifest": str(args.output_root.expanduser().resolve() / "enriched_manifest.json"),
        "validation_styles": receipt["validation_grouping"]["style_counts"],
        "validation_post_bos_positions": receipt["validation_grouping"]["post_bos_positions"],
        "ordered_length_vector_match": receipt["geometry"]["ordered_length_vector_match"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
