#!/usr/bin/env python3
"""Public fitting-bank support histograms for TRR-P11.

This producer reads only the public fitting token, mask, position, and row
metadata payloads named by the immutable contract.  It never opens hidden
states, source text, reconstruction truth, model weights, or predictions.
The B1 payload is a single nested file: its first 1,200 rows must be byte
identical in tensor value to B0 and are counted once as the B1 prefix.  The
producer separately reports B0, all B1 rows, and B1 additions; it never
concatenates B0 with the full B1 file.

The command requires ``--execute`` and creates its output exclusively.  The
checked-in contract deliberately points at the historical common frequency
payload, which may be restored later.  Missing or changed resources fail
closed rather than silently substituting a bank-local frequency map.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors import safe_open


TASK_ID = "TRR-P11"
SCHEMA = "token-reconstruction.trr-p11-public-support-histogram.v1"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
VOCAB_SIZE = 128256
EXPECTED_WIDTH = 192
EVALUATION_LAST_POSITION = 127

POSITION_BINS: tuple[tuple[str, int, int], ...] = (
    ("1-15", 1, 15),
    ("16-39", 16, 39),
    ("40-79", 40, 79),
    ("80-127", 80, 127),
    ("128-191", 128, 191),
)
FREQUENCY_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("unseen_0", 0, 0),
    ("seen_1", 1, 1),
    ("seen_2_4", 2, 4),
    ("seen_5_16", 5, 16),
    ("seen_17_64", 17, 64),
    ("seen_65_plus", 65, None),
)


class SupportHistogramError(RuntimeError):
    """Raised when a frozen public support contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise SupportHistogramError(f"{label} must be an existing regular file: {candidate}")
    return candidate.resolve()


def file_record(path: Path, *, label: str, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    actual_path = _regular_file(path, label=label)
    record = {
        "path": str(actual_path),
        "bytes": int(actual_path.stat().st_size),
        "sha256": sha256_file(actual_path),
    }
    if expected is not None:
        for key in ("bytes", "sha256"):
            expected_value = expected.get(key)
            if expected_value is not None and record[key] != expected_value:
                raise SupportHistogramError(
                    f"{label} {key} changed: expected {expected_value}, got {record[key]}"
                )
    return record


def _load_json(path: Path, *, label: str) -> Any:
    actual = _regular_file(path, label=label)
    try:
        return json.loads(actual.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupportHistogramError(f"cannot parse {label}: {actual}") from exc


def _resolve_path(config_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SupportHistogramError(f"{label} path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path


def _tensor_digest(value: torch.Tensor, *, prefix: bytes) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256(prefix)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _plain_tensor_sha256(value: torch.Tensor) -> str:
    """Established P10 tensor_digest: canonical shape/dtype header plus bytes."""
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(contiguous.shape), "dtype": str(contiguous.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _support_digest(ids: Sequence[int], counts: Sequence[int]) -> str:
    id_tensor = torch.as_tensor(list(ids), dtype=torch.long, device="cpu").contiguous()
    count_tensor = torch.as_tensor(list(counts), dtype=torch.long, device="cpu").contiguous()
    if id_tensor.ndim != 1 or count_tensor.ndim != 1 or tuple(id_tensor.shape) != tuple(count_tensor.shape):
        raise SupportHistogramError("support IDs/counts must be equal-length rank-1 vectors")
    if int(id_tensor.numel()) == 0:
        raise SupportHistogramError("support vector is empty")
    if (id_tensor < 0).any().item() or (id_tensor >= VOCAB_SIZE).any().item():
        raise SupportHistogramError("support ID is outside the declared vocabulary")
    if (count_tensor <= 0).any().item():
        raise SupportHistogramError("support counts must be positive")
    if not torch.equal(id_tensor, torch.sort(id_tensor).values):
        raise SupportHistogramError("support IDs must be sorted ascending")
    if int(torch.unique(id_tensor).numel()) != int(id_tensor.numel()):
        raise SupportHistogramError("support IDs must be unique")
    digest = hashlib.sha256(b"trr0010-support-v1\0")
    digest.update(_tensor_digest(id_tensor, prefix=b"trr0010-support-ids\0").encode("ascii"))
    digest.update(_tensor_digest(count_tensor, prefix=b"trr0010-support-counts\0").encode("ascii"))
    return digest.hexdigest()


def _frequency_bin(value: int) -> str:
    for name, lower, upper in FREQUENCY_BINS:
        if value >= lower and (upper is None or value <= upper):
            return name
    raise SupportHistogramError(f"frequency has no declared bin: {value}")


def _position_bin(value: int) -> str:
    for name, lower, upper in POSITION_BINS:
        if lower <= value <= upper:
            return name
    raise SupportHistogramError(f"post-BOS position has no declared bin: {value}")


def _style(row: Mapping[str, Any]) -> str:
    for key in ("style", "stratum", "dataset_key", "domain"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _source_type(row: Mapping[str, Any]) -> str:
    if bool(row.get("synthetic", False)):
        return "controlled"
    value = _style(row).lower()
    return "controlled" if "controlled" in value else "natural"


def _load_rows(spec: Mapping[str, Any], *, config_path: Path, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = _resolve_path(config_path, spec.get("path"), label=f"{label} records")
    record = file_record(source_path, label=f"{label} records", expected=spec)
    raw = _load_json(source_path, label=f"{label} records")
    rows = raw.get("records") if isinstance(raw, Mapping) else raw
    if not isinstance(rows, list):
        raise SupportHistogramError(f"{label} records must be a list or an object with records")
    start = int(spec.get("row_start", 0))
    end = int(spec.get("row_end", len(rows)))
    if start < 0 or end < start or end > len(rows):
        raise SupportHistogramError(f"{label} row slice [{start}, {end}) is outside records")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expected_global, value in enumerate(rows[start:end], start):
        if not isinstance(value, Mapping):
            raise SupportHistogramError(f"{label} row {expected_global} is not an object")
        row = dict(value)
        if int(row.get("global_row", expected_global)) != expected_global:
            raise SupportHistogramError(f"{label} row {expected_global} has a non-contiguous global_row")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise SupportHistogramError(f"{label} row {expected_global} has no record_id")
        if record_id in seen:
            raise SupportHistogramError(f"{label} has duplicate record_id {record_id}")
        seen.add(record_id)
        selected.append(row)
    expected_count = spec.get("records")
    if expected_count is not None and len(selected) != int(expected_count):
        raise SupportHistogramError(f"{label} records {len(selected)} != {expected_count}")
    return selected, {"file": record, "row_start": start, "row_end": end, "records": len(selected)}


def _safe_tensors(path: Path, *, label: str) -> tuple[dict[str, torch.Tensor], list[str]]:
    actual = _regular_file(path, label=label)
    try:
        with safe_open(str(actual), framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            required = {"token_ids", "attention_mask", "position_ids"}
            missing = sorted(required - set(keys))
            if missing:
                raise SupportHistogramError(f"{label} is missing tensors {missing}; keys={keys}")
            tensors = {
                name: handle.get_tensor(name).detach().cpu().contiguous()
                for name in sorted(required)
            }
    except SupportHistogramError:
        raise
    except Exception as exc:
        raise SupportHistogramError(f"cannot load public tensors from {label}: {actual}") from exc
    return tensors, keys


def _validate_bank(
    tensors: Mapping[str, torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    token_ids = tensors["token_ids"]
    attention_mask = tensors["attention_mask"]
    position_ids = tensors["position_ids"]
    expected_rows = int(spec["records"])
    expected_width = int(spec.get("width", EXPECTED_WIDTH))
    if tuple(token_ids.shape) != (expected_rows, expected_width):
        raise SupportHistogramError(f"{label} token_ids shape {list(token_ids.shape)} differs from contract")
    if tuple(attention_mask.shape) != tuple(token_ids.shape) or tuple(position_ids.shape) != tuple(token_ids.shape):
        raise SupportHistogramError(f"{label} tensor geometries differ")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise SupportHistogramError(f"{label} token_ids dtype must be int32 or int64")
    if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64):
        raise SupportHistogramError(f"{label} attention_mask dtype must be integer-like")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise SupportHistogramError(f"{label} position_ids dtype must be int32 or int64")
    if token_ids[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise SupportHistogramError(f"{label} has a row without BOS {BOS_TOKEN_ID}")
    if attention_mask.lt(0).any().item() or attention_mask.gt(1).any().item():
        raise SupportHistogramError(f"{label} attention_mask is not binary")
    mask = attention_mask.to(dtype=torch.bool)
    expected_positions = torch.arange(expected_width, dtype=position_ids.dtype)
    total_active = 0
    invalid_rows = 0
    for index, row in enumerate(rows):
        active = int(mask[index].sum().item())
        total_active += active
        if active <= 1 or not mask[index, :active].all().item() or mask[index, active:].any().item():
            invalid_rows += 1
            continue
        if not token_ids[index, active:].eq(PAD_TOKEN_ID).all().item():
            invalid_rows += 1
            continue
        if token_ids[index, 1:active].eq(PAD_TOKEN_ID).any().item():
            invalid_rows += 1
            continue
        if not torch.equal(position_ids[index], expected_positions):
            invalid_rows += 1
            continue
        declared = row.get("post_bos_token_count", row.get("target_post_bos_token_count"))
        if declared is not None and int(declared) != active - 1:
            raise SupportHistogramError(f"{label} row {index} length disagrees with attention_mask")
    if invalid_rows:
        raise SupportHistogramError(f"{label} has {invalid_rows} invalid BOS/padding/position rows")
    expected_post_bos = spec.get("post_bos_positions")
    post_bos = int(mask[:, 1:].sum().item())
    if expected_post_bos is not None and post_bos != int(expected_post_bos):
        raise SupportHistogramError(f"{label} post-BOS positions {post_bos} != {expected_post_bos}")
    return {
        "records": expected_rows,
        "stored_width": expected_width,
        "active_tokens_including_bos": total_active,
        "post_bos_positions": post_bos,
        "bos_positions": expected_rows,
        "padding_positions": int((~mask).sum().item()),
        "position_coordinate": "zero-based tensor columns; support bins are one-based after BOS",
    }


def _load_bank(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, config_path: Path, label: str) -> dict[str, Any]:
    path = _resolve_path(config_path, spec.get("path"), label=f"{label} payload")
    file_info = file_record(path, label=f"{label} payload", expected=spec)
    tensors, keys = _safe_tensors(path, label=f"{label} payload")
    geometry = _validate_bank(tensors, rows, spec, label=label)
    return {
        "label": label,
        "path": path,
        "file": file_info,
        "tensor_keys": keys,
        "tensors": tensors,
        "geometry": geometry,
    }


def _validate_nested_prefix(
    b0: Mapping[str, Any],
    b1: Mapping[str, Any],
    b0_rows: Sequence[Mapping[str, Any]],
    b1_rows: Sequence[Mapping[str, Any]],
    prefix_rows: int,
) -> dict[str, Any]:
    if len(b0["tensors"]["token_ids"]) != prefix_rows:
        raise SupportHistogramError("B0 row count does not equal the declared B1 prefix")
    for name in ("token_ids", "attention_mask", "position_ids"):
        if not torch.equal(b0["tensors"][name], b1["tensors"][name][:prefix_rows]):
            raise SupportHistogramError(f"B1 prefix differs from B0 in {name}")
    for index, row in enumerate(b1_rows[:prefix_rows]):
        if row.get("b0_prefix_row") != index:
            raise SupportHistogramError(f"B1 prefix row {index} lacks b0_prefix_row={index}")
    for index, row in enumerate(b1_rows[prefix_rows:], prefix_rows):
        if row.get("b0_prefix_row") is not None:
            raise SupportHistogramError(f"B1 addition row {index} unexpectedly claims a B0 prefix row")
        if row.get("capture_local_row") != index - prefix_rows:
            raise SupportHistogramError(f"B1 addition row {index} has a non-contiguous capture_local_row")
    if len(b0_rows) != prefix_rows:
        raise SupportHistogramError("B0 row metadata does not equal the declared B1 prefix")
    for index, (b0_row, b1_row) in enumerate(zip(b0_rows, b1_rows[:prefix_rows])):
        for key in ("record_id", "dataset_key", "stratum", "synthetic"):
            if key in b0_row and key in b1_row and b0_row.get(key) != b1_row.get(key):
                raise SupportHistogramError(
                    f"B1 prefix record metadata differs from B0 at row {index}: {key}"
                )
    return {
        "prefix_rows": prefix_rows,
        "addition_rows": len(b1_rows) - prefix_rows,
        "full_b1_rows_counted_once": True,
        "b0_concatenation_forbidden": True,
        "tensor_fields_verified": ["token_ids", "attention_mask", "position_ids"],
        "row_metadata_verified": ["b0_prefix_row", "capture_local_row"],
    }


def _frequency_summary(counts: Mapping[int, int]) -> dict[str, Any]:
    values = [int(value) for value in counts.values()]
    summary: dict[str, Any] = {
        "vocabulary_size": VOCAB_SIZE,
        "distinct_token_ids": len(values),
        "unseen_vocabulary_ids": VOCAB_SIZE - len(values),
        "token_occurrences": sum(values),
        "frequency_bins": {},
    }
    for name, lower, upper in FREQUENCY_BINS:
        selected = [value for value in values if value >= lower and (upper is None or value <= upper)]
        summary["frequency_bins"][name] = {
            "distinct_token_ids": len(selected),
            "token_occurrences": sum(selected),
        }
    if values:
        ordered = sorted(values)
        summary["min_count"] = ordered[0]
        summary["max_count"] = ordered[-1]
    else:
        summary["min_count"] = None
        summary["max_count"] = None
    return summary


def _sparse_count_identity(counts: Mapping[int, int]) -> dict[str, Any]:
    ordered = sorted((int(token), int(value)) for token, value in counts.items() if int(value) > 0)
    ids = [token for token, _value in ordered]
    values = [value for _token, value in ordered]
    return {
        "support_count": len(ordered),
        "support_ids_tensor_sha256": _tensor_digest(torch.as_tensor(ids, dtype=torch.int64), prefix=b"trr-p11-support-ids\0") if ordered else None,
        "support_counts_tensor_sha256": _tensor_digest(torch.as_tensor(values, dtype=torch.int64), prefix=b"trr-p11-support-counts\0") if ordered else None,
        "sparse_count_map_sha256": hashlib.sha256(
            "".join(f"{token}:{value}\n" for token, value in ordered).encode("ascii")
        ).hexdigest(),
        "trr0010_support_digest": _support_digest(ids, values) if ordered else None,
    }


def _empty_joint(styles: Sequence[str]) -> dict[str, Any]:
    return {
        style: {
            position: {frequency: 0 for frequency, _lower, _upper in FREQUENCY_BINS}
            for position, _lower, _upper in POSITION_BINS
        }
        for style in sorted(set(styles))
    }


def _count_view(
    tensors: Mapping[str, torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
    frequency_reference: Mapping[int, int],
    *,
    label: str,
    row_slice: tuple[int, int],
) -> dict[str, Any]:
    token_ids = tensors["token_ids"]
    mask = tensors["attention_mask"].to(dtype=torch.bool)
    own_counts: Counter[int] = Counter()
    reference_bins: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    style_positions: Counter[tuple[str, str]] = Counter()
    style_frequencies: Counter[tuple[str, str]] = Counter()
    joint = _empty_joint([_style(row) for row in rows])
    all_tokens: set[int] = set()
    eval_tokens: set[int] = set()
    post_bos = 0
    for index, row in enumerate(rows):
        style = _style(row)
        source = _source_type(row)
        styles[style] += 1
        source_types[source] += 1
        active = int(mask[index].sum().item())
        for position, token_value in enumerate(token_ids[index, 1:active].tolist(), 1):
            token = int(token_value)
            own_counts[token] += 1
            all_tokens.add(token)
            post_bos += 1
            position_name = _position_bin(position)
            frequency_name = _frequency_bin(int(frequency_reference.get(token, 0)))
            positions[position_name] += 1
            reference_bins[frequency_name] += 1
            style_positions[(style, position_name)] += 1
            style_frequencies[(style, frequency_name)] += 1
            joint[style][position_name][frequency_name] += 1
            if position <= EVALUATION_LAST_POSITION:
                eval_tokens.add(token)
    own_identity = _sparse_count_identity(own_counts)
    position_summary = {
        name: {
            "lower": lower,
            "upper": upper,
            "post_bos_positions": int(positions[name]),
            "in_evaluation_range": bool(upper <= EVALUATION_LAST_POSITION),
        }
        for name, lower, upper in POSITION_BINS
    }
    style_summary: dict[str, Any] = {}
    for style in sorted(styles):
        style_summary[style] = {
            "records": int(styles[style]),
            "source_type": "controlled" if any(
                _source_type(row) == "controlled" and _style(row) == style for row in rows
            ) else "natural",
            "post_bos_positions": sum(
                int(style_positions[(style, name)]) for name, _lower, _upper in POSITION_BINS
            ),
            "evaluation_range_positions": sum(
                int(style_positions[(style, name)])
                for name, _lower, upper in POSITION_BINS
                if upper <= EVALUATION_LAST_POSITION
            ),
            "post_evaluation_positions": sum(
                int(style_positions[(style, name)])
                for name, _lower, upper in POSITION_BINS
                if upper > EVALUATION_LAST_POSITION
            ),
            "position_rows": {
                name: int(style_positions[(style, name)])
                for name, _lower, _upper in POSITION_BINS
            },
            "frequency_rows": {
                name: int(style_frequencies[(style, name)])
                for name, _lower, _upper in FREQUENCY_BINS
            },
        }
    return {
        "label": label,
        "row_slice_in_source": list(row_slice),
        "records": len(rows),
        "post_bos_positions": post_bos,
        "own_bank_frequency": {
            "summary": _frequency_summary(own_counts),
            "identity": own_identity,
        },
        "common_reference_frequency": {
            "bin_occurrences": {
                name: int(reference_bins[name]) for name, _lower, _upper in FREQUENCY_BINS
            },
            "reference_definition": "the frozen common public B0 fit-bank frequency map; zero means unseen in that map",
        },
        "position_bins": position_summary,
        "style_summary": style_summary,
        "source_type_positions": {
            source: int(value) for source, value in sorted(source_types.items())
        },
        "coverage": {
            "distinct_post_bos_token_ids": len(all_tokens),
            "distinct_post_bos_token_ids_in_evaluation_range": len(eval_tokens),
            "evaluation_range_positions": sum(
                int(positions[name])
                for name, _lower, upper in POSITION_BINS
                if upper <= EVALUATION_LAST_POSITION
            ),
            "post_evaluation_positions": sum(
                int(positions[name])
                for name, _lower, upper in POSITION_BINS
                if upper > EVALUATION_LAST_POSITION
            ),
        },
        "joint_style_position_frequency": joint,
        "correctness_status": "not_computed",
    }


def _load_common_reference(spec: Mapping[str, Any], *, config_path: Path) -> tuple[dict[int, int], dict[str, Any]]:
    path = _resolve_path(config_path, spec.get("path"), label="common frequency reference")
    file_info = file_record(path, label="common frequency reference", expected=spec)
    # The compact reference has support_ids/support_counts rather than bank
    # tensors, so it is opened directly and never passed through the bank loader.
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            vector_keys = sorted(handle.keys())
            required = {"support_ids", "support_counts"}
            missing = sorted(required - set(vector_keys))
            if missing:
                raise SupportHistogramError(f"common frequency reference is missing {missing}")
            ids = handle.get_tensor("support_ids").detach().cpu().contiguous()
            counts = handle.get_tensor("support_counts").detach().cpu().contiguous()
            frequency_vector = (
                handle.get_tensor("frequency_counts").detach().cpu().contiguous()
                if "frequency_counts" in vector_keys
                else None
            )
    except SupportHistogramError:
        raise
    except Exception as exc:
        raise SupportHistogramError("cannot load common frequency reference") from exc
    if ids.dtype != torch.int64 or counts.dtype != torch.int64:
        raise SupportHistogramError("common frequency reference IDs/counts must be int64")
    if ids.ndim != 1 or counts.ndim != 1 or tuple(ids.shape) != tuple(counts.shape):
        raise SupportHistogramError("common frequency reference IDs/counts have incompatible geometry")
    if frequency_vector is not None:
        if frequency_vector.dtype != torch.int64 or tuple(frequency_vector.shape) != (VOCAB_SIZE,):
            raise SupportHistogramError("common frequency reference frequency_counts has incompatible geometry")
        if frequency_vector.lt(0).any().item():
            raise SupportHistogramError("common frequency reference frequency_counts contains a negative count")
    ids_list = [int(value) for value in ids.tolist()]
    counts_list = [int(value) for value in counts.tolist()]
    digest = _support_digest(ids_list, counts_list)
    expected_count = spec.get("support_count")
    if expected_count is not None and len(ids_list) != int(expected_count):
        raise SupportHistogramError(
            f"common frequency reference support count changed: expected {expected_count}, got {len(ids_list)}"
        )
    expected_occurrences = spec.get("positive_occurrences")
    if expected_occurrences is not None and sum(counts_list) != int(expected_occurrences):
        raise SupportHistogramError(
            "common frequency reference positive occurrence count changed: "
            f"expected {expected_occurrences}, got {sum(counts_list)}"
        )
    expected_digest = spec.get("support_digest_trr0010")
    if expected_digest is not None and digest != expected_digest:
        raise SupportHistogramError(
            f"common frequency reference support digest changed: expected {expected_digest}, got {digest}"
        )
    ids_sha = _plain_tensor_sha256(ids)
    counts_sha = _plain_tensor_sha256(counts)
    for field, actual in (("support_ids_tensor_sha256", ids_sha), ("support_counts_tensor_sha256", counts_sha)):
        expected = spec.get(field)
        if expected is not None and actual != expected:
            raise SupportHistogramError(f"common frequency reference {field} changed")
    frequency_sha = _plain_tensor_sha256(frequency_vector) if frequency_vector is not None else None
    expected_frequency_sha = spec.get("frequency_vector_tensor_sha256")
    if expected_frequency_sha is not None and frequency_sha is not None and frequency_sha != expected_frequency_sha:
        raise SupportHistogramError("common frequency reference frequency_vector_tensor_sha256 changed")
    if frequency_vector is not None:
        dense_ids = torch.nonzero(frequency_vector > 0, as_tuple=False).flatten().to(dtype=torch.int64)
        dense_counts = frequency_vector.index_select(0, dense_ids)
        if not torch.equal(dense_ids, ids) or not torch.equal(dense_counts, counts):
            raise SupportHistogramError("common frequency reference dense/sparse tensors differ")
    return (
        dict(zip(ids_list, counts_list)),
        {
            "file": file_info,
            "tensor_keys": vector_keys,
            "support_ids_shape": list(ids.shape),
            "support_counts_shape": list(counts.shape),
            "support_ids_dtype": str(ids.dtype),
            "support_counts_dtype": str(counts.dtype),
            "support_count": len(ids_list),
            "positive_occurrences": sum(counts_list),
            "support_digest_trr0010": digest,
            "support_ids_tensor_sha256": ids_sha,
            "support_counts_tensor_sha256": counts_sha,
            "frequency_vector_tensor_sha256": frequency_sha,
        },
    )


def build_report(config_path: Path) -> dict[str, Any]:
    config_path = _regular_file(config_path, label="support histogram config")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupportHistogramError(f"cannot parse support histogram config: {config_path}") from exc
    if not isinstance(config, Mapping) or config.get("schema") != SCHEMA:
        raise SupportHistogramError(f"unsupported support histogram schema in {config_path}")
    banks_spec = config.get("banks")
    if not isinstance(banks_spec, Mapping) or set(banks_spec) != {"B0", "B1"}:
        raise SupportHistogramError("contract must define exactly B0 and B1 banks")
    common_spec = config.get("common_frequency_reference")
    if not isinstance(common_spec, Mapping):
        raise SupportHistogramError("contract has no common frequency reference")
    records_spec = config.get("records")
    if not isinstance(records_spec, Mapping):
        raise SupportHistogramError("contract has no records binding")
    b1_rows, b1_records = _load_rows(records_spec, config_path=config_path, label="B1")
    b0_rows = b1_rows[: int(banks_spec["B0"]["records"])]
    if len(b0_rows) != int(banks_spec["B0"]["records"]):
        raise SupportHistogramError("B1 records do not contain the declared B0 prefix")
    b0 = _load_bank(banks_spec["B0"], b0_rows, config_path=config_path, label="B0")
    b1 = _load_bank(banks_spec["B1"], b1_rows, config_path=config_path, label="B1")
    prefix_rows = int(config.get("nested_bank", {}).get("prefix_rows", len(b0_rows)))
    nested = _validate_nested_prefix(b0, b1, b0_rows, b1_rows, prefix_rows)
    common_counts, common_identity = _load_common_reference(common_spec, config_path=config_path)
    b1_addition_start = prefix_rows
    b1_additions = {
        "token_ids": b1["tensors"]["token_ids"][b1_addition_start:],
        "attention_mask": b1["tensors"]["attention_mask"][b1_addition_start:],
        "position_ids": b1["tensors"]["position_ids"][b1_addition_start:],
    }
    reports = {
        "B0": _count_view(
            b0["tensors"], b0_rows, common_counts, label="B0 full bank", row_slice=(0, len(b0_rows))
        ),
        "B1": _count_view(
            b1["tensors"], b1_rows, common_counts, label="B1 full nested bank", row_slice=(0, len(b1_rows))
        ),
        "B1_additions": _count_view(
            b1_additions,
            b1_rows[b1_addition_start:],
            common_counts,
            label="B1 additions only",
            row_slice=(b1_addition_start, len(b1_rows)),
        ),
    }
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "COMPLETE_PUBLIC_SUPPORT_HISTOGRAM",
        "created_utc": utc_now(),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "scope": config.get("scope", {}),
        "records_binding": b1_records,
        "banks": {
            "B0": {key: value for key, value in b0.items() if key not in {"path", "tensors"}},
            "B1": {key: value for key, value in b1.items() if key not in {"path", "tensors"}},
        },
        "nested_bank_audit": nested,
        "common_frequency_reference": common_identity,
        "histograms": reports,
        "truth_boundary": {
            "public_token_ids_read": True,
            "public_attention_masks_read": True,
            "public_position_ids_read": True,
            "source_text_read": False,
            "hidden_states_read": False,
            "model_opened": False,
            "reconstruction_predictions_read": False,
            "evaluation_truth_opened": False,
        },
        "counting_audit": {
            "labels": "token_ids[:,1:][attention_mask[:,1:]]",
            "position_coordinate": "one-based offsets after BOS",
            "exclude_bos": True,
            "exclude_padding": True,
            "include_positions": [1, 191],
            "evaluation_positions": [1, 127],
            "post_evaluation_positions": [128, 191],
            "frequency_reference_is_common": True,
            "own_bank_counts_are_diagnostic_only": True,
            "b1_full_file_counted_once": True,
            "b0_prefix_not_concatenated_to_b1": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the public-only read and create the output exclusively",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing to run without explicit --execute")
    output = Path(args.output).expanduser()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"output must be create-only and absent: {output}")
    try:
        report = build_report(Path(args.config))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except SupportHistogramError as exc:
        raise SystemExit(f"support histogram contract failure: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
