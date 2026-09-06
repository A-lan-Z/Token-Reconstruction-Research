#!/usr/bin/env python3
"""TRR-0007 public fitting-support diagnostics and sampling recipe.

The default command reads only public token-ID and attention-mask tensors from
the existing TRR-0005 enriched fit bank, plus public record metadata.  It
reports exact position, token-frequency, input-style, natural/controlled, and
joint coverage counts.  An optional public-development projection can score
the frozen trained-diagonal decoder against the reused TRR-0004 validation
labels; that path is explicitly post-hoc and never selects a fit or evaluation
record.

The script never opens evaluator-private truth, final-panel records, target
weights, or a public model.  The improved-bank recipe is metadata-only.  Its
controlled rows must later be materialized by the registered real P0 public
forward before fitting or evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TASK_ID = "TRR-0007"
SUPPORT_SCHEMA = "token-reconstruction.trr0007-public-support.v1"
RECIPE_SCHEMA = "token-reconstruction.trr0007-improved-public-sampling-recipe.v1"
EXCLUSION_SCHEMA = "token-reconstruction.trr0007-public-exclusion-manifest.v1"
CANDIDATE_FREQUENCY_SCHEMA = "token-reconstruction.trr0007-public-fit-candidate-frequency.v1"
PROJECTION_SCHEMA = "token-reconstruction.trr0007-public-development-projection.v1"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
VOCAB_SIZE = 128256
EXPECTED_FIT_RECORDS = 1200
EXPECTED_SEQUENCE_LENGTH = 192
EXPECTED_POST_BOS = 124371
EXPECTED_EVAL_LAST_POSITION = 127
EXPECTED_DRAW_BATCH = 512
EXPECTED_DRAW_STEPS = 3000
EXPECTED_CONTROLLED_RECORDS = 120
EXPECTED_CONTROLLED_REPLACEMENTS = 3600
EXPECTED_CONTROLLED_TOKEN_IDS = 2000
PROPOSED_CONTROLLED_TOKEN_IDS = 3600
PROPOSED_ADDITIONAL_TOKEN_IDS = PROPOSED_CONTROLLED_TOKEN_IDS - EXPECTED_CONTROLLED_TOKEN_IDS

# Position values are offsets after BOS.  The first four bins cover the
# first-127-token evaluation range; the final bin makes the old long-prefix
# supplement visible rather than silently pooling it with evaluation support.
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
RECIPE_LENGTH_STRATA: tuple[tuple[str, int, int, int], ...] = (
    ("40-63", 40, 63, 24),
    ("64-95", 64, 95, 36),
    ("96-127", 96, 127, 36),
    ("128-191", 128, 191, 24),
)
# The first four position bins are the permitted one-based evaluation range.
# Three, six, nine, and twelve replacements per controlled row sum to the
# frozen 30-replacement budget while giving every row support in each bin.
RECIPE_POSITION_PLAN: tuple[tuple[str, int, int, int], ...] = (
    ("1-15", 1, 15, 3),
    ("16-39", 16, 39, 6),
    ("40-79", 40, 79, 9),
    ("80-127", 80, 127, 12),
)


class SupportDiagnosticError(RuntimeError):
    """Raised when a support artifact violates the public-data contract."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SupportDiagnosticError(f"resource must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> Any:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SupportDiagnosticError(f"{label} must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupportDiagnosticError(f"cannot parse {label}: {path}") from exc


def _records(path: Path, *, label: str) -> list[dict[str, Any]]:
    value = _json(path, label=label)
    rows = value.get("records") if isinstance(value, Mapping) else value
    if not isinstance(rows, list) or not rows:
        raise SupportDiagnosticError(f"{label} has no records list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str):
            raise SupportDiagnosticError(f"{label} record {index} has no record_id")
        record_id = str(row["record_id"])
        if not record_id or record_id in seen:
            raise SupportDiagnosticError(f"{label} record IDs are empty or duplicated")
        seen.add(record_id)
        result.append(dict(row))
    return result


def _plan_records(path: Path) -> list[dict[str, Any]]:
    value = _json(path, label="public corpus plan")
    try:
        rows = value["arms"]["coverage_mix_v1"]["records"]
    except (KeyError, TypeError) as exc:
        raise SupportDiagnosticError("public corpus plan has no coverage_mix_v1 records") from exc
    if not isinstance(rows, list):
        raise SupportDiagnosticError("public corpus plan records are malformed")
    return _records_from_value(rows, label="public corpus plan records")


def _records_from_value(rows: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise SupportDiagnosticError(f"{label} has no records")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str):
            raise SupportDiagnosticError(f"{label} record {index} is malformed")
        record_id = str(row["record_id"])
        if record_id in seen:
            raise SupportDiagnosticError(f"{label} has duplicate record_id {record_id}")
        seen.add(record_id)
        result.append(dict(row))
    return result


def _load_label_tensors(path: Path, *, label: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SupportDiagnosticError(f"{label} tensor artifact must be a regular file: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if "token_ids" not in keys or "attention_mask" not in keys:
                raise SupportDiagnosticError(
                    f"{label} must contain token_ids and attention_mask; keys={sorted(keys)}"
                )
            token_ids = handle.get_tensor("token_ids").detach().cpu().contiguous()
            attention_mask = handle.get_tensor("attention_mask").detach().cpu().contiguous()
    except SupportDiagnosticError:
        raise
    except Exception as exc:
        raise SupportDiagnosticError(f"cannot read {label} labels/mask: {path}") from exc
    if token_ids.ndim != 2 or attention_mask.ndim != 2:
        raise SupportDiagnosticError(f"{label} labels and mask must be rank two")
    if tuple(token_ids.shape) != tuple(attention_mask.shape):
        raise SupportDiagnosticError(f"{label} labels and mask geometry differ")
    return token_ids, attention_mask, {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "tensor_keys": sorted(keys),
        "token_ids_dtype": str(token_ids.dtype),
        "attention_mask_dtype": str(attention_mask.dtype),
        "shape": list(token_ids.shape),
    }


def _position_bin(position: int) -> str:
    value = int(position)
    for name, lower, upper in POSITION_BINS:
        if lower <= value <= upper:
            return name
    raise SupportDiagnosticError(f"post-BOS position is outside padded geometry: {value}")


def _frequency_bin(frequency: int) -> str:
    value = int(frequency)
    for name, lower, upper in FREQUENCY_BINS:
        if value >= lower and (upper is None or value <= upper):
            return name
    raise SupportDiagnosticError(f"frequency has no declared bin: {value}")


def _style(row: Mapping[str, Any]) -> str:
    domain = row.get("domain")
    style = row.get("style")
    if isinstance(domain, str) and domain:
        mapping = {
            "alpaca_instruction": "natural_alpaca",
            "pile_natural": "natural_pile",
            "finance_instruction": "natural_finance",
            "controlled_pile_context": "controlled_pile",
            "controlled_finance_context": "controlled_finance",
        }
        return mapping.get(domain, domain)
    if isinstance(style, str) and style:
        return style
    dataset = row.get("dataset_key")
    if isinstance(dataset, str) and dataset:
        return dataset
    return "unknown"


def _is_synthetic(row: Mapping[str, Any]) -> bool:
    return bool(row.get("synthetic", False)) or _style(row).startswith("controlled_")


def _validate_rows(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    expected_records: int | None = None,
    expected_width: int | None = None,
) -> dict[str, Any]:
    if expected_records is not None and len(rows) != expected_records:
        raise SupportDiagnosticError(f"{label} record count {len(rows)} != {expected_records}")
    if token_ids.shape[0] != len(rows):
        raise SupportDiagnosticError(f"{label} records do not match tensor rows")
    if expected_width is not None and token_ids.shape[1] != expected_width:
        raise SupportDiagnosticError(f"{label} width {token_ids.shape[1]} != {expected_width}")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise SupportDiagnosticError(f"{label} token IDs must be int32 or int64")
    if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64):
        raise SupportDiagnosticError(f"{label} attention mask must be integer-like")
    if token_ids[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise SupportDiagnosticError(f"{label} rows do not begin with BOS {BOS_TOKEN_ID}")
    if attention_mask.lt(0).any().item() or attention_mask.gt(1).any().item():
        raise SupportDiagnosticError(f"{label} attention mask is not binary")
    mask = attention_mask.to(dtype=torch.bool)
    invalid_rows = 0
    total_active = 0
    for index, row in enumerate(rows):
        active = int(mask[index].sum().item())
        total_active += active
        if active <= 1:
            invalid_rows += 1
            continue
        if not mask[index, :active].all().item() or mask[index, active:].any().item():
            invalid_rows += 1
            continue
        if not token_ids[index, active:].eq(PAD_TOKEN_ID).all().item():
            invalid_rows += 1
            continue
        declared = row.get("post_bos_token_count", row.get("target_post_bos_token_count"))
        if declared is not None and int(declared) != active - 1:
            raise SupportDiagnosticError(f"{label} row {index} length disagrees with mask")
        if row.get("slot") is not None and int(row["slot"]) != index:
            raise SupportDiagnosticError(f"{label} row {index} has non-contiguous slot metadata")
    if invalid_rows:
        raise SupportDiagnosticError(f"{label} has {invalid_rows} invalid rows")
    return {
        "records": len(rows),
        "stored_width": int(token_ids.shape[1]),
        "active_tokens_including_bos": total_active,
        "post_bos_positions": int(mask[:, 1:].sum().item()),
        "bos_positions": len(rows),
        "padding_positions": int((~mask).sum().item()),
    }


def _new_cell() -> dict[str, Any]:
    # Support-only passes never load predictions, so correctness must remain
    # explicitly uncomputed rather than looking like zero accuracy.
    return {"examples": 0, "distinct_token_ids": set(), "correct": 0, "errors": 0, "scored": False}


def _finish_cells(cells: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (style, position, frequency), value in sorted(cells.items()):
        examples = int(value["examples"])
        scored = bool(value.get("scored", False))
        result.setdefault(style, {}).setdefault(position, {})[frequency] = {
            "examples": examples,
            "distinct_token_ids": len(value["distinct_token_ids"]),
            "correct": int(value.get("correct", 0)) if scored else None,
            "errors": int(value.get("errors", 0)) if scored else None,
            "token_accuracy": (
                float(value.get("correct", 0)) / examples if scored and examples else None
            ),
            "correctness_status": "computed" if scored else "not_computed",
        }
    return result


def _frequency_summary(counts: Mapping[int, int]) -> dict[str, Any]:
    bins: dict[str, dict[str, Any]] = {}
    for name, lower, upper in FREQUENCY_BINS:
        values = [int(value) for value in counts.values() if value >= lower and (upper is None or value <= upper)]
        bins[name] = {
            "distinct_token_ids": len(values),
            "token_occurrences": sum(values),
            "mean_frequency": (sum(values) / len(values)) if values else 0.0,
        }
    ordered = sorted(int(value) for value in counts.values())

    def quantile(q: float) -> float | None:
        if not ordered:
            return None
        index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
        return float(ordered[index])

    return {
        "vocab_size": VOCAB_SIZE,
        "distinct_token_ids": len(counts),
        "unseen_vocabulary_ids": VOCAB_SIZE - len(counts),
        "token_occurrences": sum(int(value) for value in counts.values()),
        "frequency_bins": bins,
        "frequency_quantiles": {
            "p50": quantile(0.50),
            "p90": quantile(0.90),
            "p99": quantile(0.99),
            "max": (float(ordered[-1]) if ordered else None),
        },
    }


def _digest_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sequence_digest(ids: Iterable[int]) -> str:
    return _digest_lines(str(int(value)) for value in ids)


def _support_for_rows(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    *,
    frequency_counts: Mapping[int, int] | None = None,
    label: str,
    expected_records: int | None = None,
    expected_post_bos: int | None = None,
    expected_width: int | None = None,
) -> dict[str, Any]:
    geometry = _validate_rows(
        token_ids,
        attention_mask,
        rows,
        label=label,
        expected_records=expected_records,
        expected_width=expected_width,
    )
    mask = attention_mask.to(dtype=torch.bool)
    counts = Counter(
        int(value)
        for value in token_ids[:, 1:][mask[:, 1:]].tolist()
    ) if frequency_counts is None else Counter({int(k): int(v) for k, v in frequency_counts.items()})
    position_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    style_position: defaultdict[tuple[str, str], int] = defaultdict(int)
    style_frequency: defaultdict[tuple[str, str], int] = defaultdict(int)
    cells: defaultdict[tuple[str, str, str], dict[str, Any]] = defaultdict(_new_cell)
    eval_token_ids: set[int] = set()
    all_token_ids: set[int] = set()
    records_by_style: Counter[str] = Counter()
    eval_records_by_style: Counter[str] = Counter()
    for index, row in enumerate(rows):
        style = _style(row)
        source_type = "controlled" if _is_synthetic(row) else "natural"
        records_by_style[style] += 1
        active = int(mask[index].sum().item())
        has_eval = False
        for position, value in enumerate(token_ids[index, 1:active].tolist(), 1):
            token = int(value)
            position_name = _position_bin(position)
            frequency_name = _frequency_bin(int(counts.get(token, 0)))
            position_counts[position_name] += 1
            style_counts[style] += 1
            source_type_counts[source_type] += 1
            style_position[(style, position_name)] += 1
            style_frequency[(style, frequency_name)] += 1
            cell = cells[(style, position_name, frequency_name)]
            cell["examples"] += 1
            cell["distinct_token_ids"].add(token)
            all_token_ids.add(token)
            if position <= EXPECTED_EVAL_LAST_POSITION:
                has_eval = True
                eval_token_ids.add(token)
        if has_eval:
            eval_records_by_style[style] += 1
    # Materialize zero cells so the joint table distinguishes no support from
    # an omitted category.  This is especially important for short prefixes
    # and controlled rows, where some style/frequency combinations are absent.
    for style in sorted(records_by_style):
        for position_name, _lower, _upper in POSITION_BINS:
            for frequency_name, _frequency_lower, _frequency_upper in FREQUENCY_BINS:
                cells.setdefault((style, position_name, frequency_name), _new_cell())
    if expected_post_bos is not None and geometry["post_bos_positions"] != expected_post_bos:
        raise SupportDiagnosticError(
            f"{label} post-BOS positions {geometry['post_bos_positions']} != {expected_post_bos}"
        )
    position_summary = {
        name: {
            "post_bos_positions": int(position_counts[name]),
            "in_evaluation_range": bool(upper <= EXPECTED_EVAL_LAST_POSITION),
            "lower": lower,
            "upper": upper,
        }
        for name, lower, upper in POSITION_BINS
    }
    style_summary: dict[str, Any] = {}
    for style in sorted(records_by_style):
        style_summary[style] = {
            "records": int(records_by_style[style]),
            "records_with_evaluation_prefix": int(eval_records_by_style[style]),
            "post_bos_positions": int(style_counts[style]),
            "evaluation_range_positions": sum(
                int(style_position[(style, name)])
                for name, _lower, upper in POSITION_BINS
                if upper <= EXPECTED_EVAL_LAST_POSITION
            ),
            "post_evaluation_positions": sum(
                int(style_position[(style, name)])
                for name, _lower, upper in POSITION_BINS
                if upper > EXPECTED_EVAL_LAST_POSITION
            ),
            "source_type": "controlled" if style.startswith("controlled_") else "natural",
            "frequency_rows": {
                name: int(style_frequency[(style, name)])
                for name, _lower, _upper in FREQUENCY_BINS
            },
            "position_rows": {
                name: int(style_position[(style, name)])
                for name, _lower, _upper in POSITION_BINS
            },
        }
    return {
        "geometry": geometry,
        "frequency_reference": "all valid post-BOS labels in this bank"
        if frequency_counts is None
        else "caller-supplied fit-bank post-BOS frequency table",
        "correctness_status": "not_computed",
        "frequency": _frequency_summary(counts),
        "position_bins": position_summary,
        "style_summary": style_summary,
        "source_type_positions": dict(sorted(source_type_counts.items())),
        "coverage": {
            "distinct_post_bos_token_ids": len(all_token_ids),
            "distinct_post_bos_token_ids_in_evaluation_range": len(eval_token_ids),
            "evaluation_range_positions": sum(
                int(position_counts[name])
                for name, _lower, upper in POSITION_BINS
                if upper <= EXPECTED_EVAL_LAST_POSITION
            ),
            "post_evaluation_positions": sum(
                int(position_counts[name])
                for name, _lower, upper in POSITION_BINS
                if upper > EXPECTED_EVAL_LAST_POSITION
            ),
        },
        "joint_style_position_frequency": _finish_cells(cells),
    }


def _replacement_support(plan_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    synthetic = [row for row in plan_rows if bool(row.get("synthetic", False))]
    # Corpus metadata stores zero-based offsets after BOS.  Keep that raw
    # coordinate and the one-based post-BOS position used by support bins
    # separate; they describe the same replacement but are not interchangeable.
    offsets: list[int] = []
    positions: list[int] = []
    replacement_ids: list[int] = []
    target_lengths: list[int] = []
    by_domain: dict[str, Any] = {}
    for row in synthetic:
        row_positions = row.get("replacement_positions", [])
        row_ids = row.get("replacement_token_ids", [])
        if not isinstance(row_positions, list) or not isinstance(row_ids, list):
            raise SupportDiagnosticError("controlled row replacement metadata is malformed")
        if len(row_positions) != len(row_ids):
            raise SupportDiagnosticError("controlled replacement positions and IDs differ")
        target_length = int(row.get("target_post_bos_token_count", -1))
        if target_length <= 0:
            raise SupportDiagnosticError("controlled row has no positive target length")
        row_offsets = [int(value) for value in row_positions]
        if any(value < 0 or value >= target_length for value in row_offsets):
            raise SupportDiagnosticError("controlled replacement offset falls outside target sequence")
        target_lengths.append(target_length)
        offsets.extend(row_offsets)
        positions.extend(value + 1 for value in row_offsets)
        replacement_ids.extend(int(value) for value in row_ids)
        domain = str(row.get("domain", "unknown"))
        item = by_domain.setdefault(domain, {"records": 0, "replacement_occurrences": 0, "positions": []})
        item["records"] += 1
        item["replacement_occurrences"] += len(row_positions)
        item["positions"].extend(value + 1 for value in row_offsets)
    position_counts = Counter(_position_bin(value) for value in positions)
    in_eval = [value for value in positions if value <= EXPECTED_EVAL_LAST_POSITION]
    return {
        "records": len(synthetic),
        "replacement_occurrences": len(positions),
        "distinct_replacement_token_ids": len(set(replacement_ids)),
        "target_post_bos_lengths": {
            "min": min(target_lengths) if target_lengths else None,
            "max": max(target_lengths) if target_lengths else None,
            "all_at_least_128": bool(target_lengths) and all(value >= 128 for value in target_lengths),
            "count_by_range": {
                name: sum(lower <= value <= upper for value in target_lengths)
                for name, lower, upper, _count in RECIPE_LENGTH_STRATA
            },
        },
        "replacement_offsets_zero_based": {
            "coordinate": "zero-based offset after BOS as stored in corpus plan",
            "min": min(offsets) if offsets else None,
            "max": max(offsets) if offsets else None,
            "all_at_least_128": bool(offsets) and all(value >= 128 for value in offsets),
            "evaluation_range_occurrences_offsets_0_126": sum(value <= EXPECTED_EVAL_LAST_POSITION - 1 for value in offsets),
            "post_evaluation_occurrences_offsets_127_190": sum(value >= EXPECTED_EVAL_LAST_POSITION for value in offsets),
        },
        "replacement_positions_after_bos_one_based": {
            "coordinate": "one-based post-BOS position used by support bins",
            "min": min(positions) if positions else None,
            "max": max(positions) if positions else None,
            "all_at_least_128": bool(positions) and all(value >= 128 for value in positions),
            "evaluation_range_occurrences_1_127": len(in_eval),
            "post_evaluation_occurrences_128_191": len(positions) - len(in_eval),
            "counts_by_position_bin": {name: int(position_counts[name]) for name, _lower, _upper in POSITION_BINS},
        },
        "by_domain": {
            domain: {
                "records": int(value["records"]),
                "replacement_occurrences": int(value["replacement_occurrences"]),
                "counts_by_position_bin": {
                    name: int(Counter(_position_bin(position) for position in value["positions"])[name])
                    for name, _lower, _upper in POSITION_BINS
                },
            }
            for domain, value in sorted(by_domain.items())
        },
        "interpretation": (
            "The current metadata has long controlled target records, but replacement offsets are "
            "not restricted to offsets >=128. Record prefix length and replacement offset are "
            "reported separately; conflating them would misstate evaluation-range support."
        ),
    }


def _verify_recorded_replacements(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    plan_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify every recorded controlled replacement against captured token IDs."""

    if token_ids.ndim != 2 or attention_mask.ndim != 2 or tuple(token_ids.shape) != tuple(attention_mask.shape):
        raise SupportDiagnosticError("fit token IDs and mask geometry is invalid for replacement verification")
    mismatches: list[dict[str, Any]] = []
    checked_records = 0
    checked_replacements = 0
    for row in plan_rows:
        if not bool(row.get("synthetic", False)):
            continue
        slot = int(row.get("slot", -1))
        if slot < 0 or slot >= token_ids.shape[0]:
            raise SupportDiagnosticError(f"controlled row slot is outside captured fit tensor: {slot}")
        active = int(attention_mask[slot].to(dtype=torch.bool).sum().item())
        offsets = row.get("replacement_positions", [])
        replacement_ids = row.get("replacement_token_ids", [])
        if not isinstance(offsets, list) or not isinstance(replacement_ids, list) or len(offsets) != len(replacement_ids):
            raise SupportDiagnosticError(f"controlled row replacement metadata is malformed at slot {slot}")
        checked_records += 1
        for offset_value, expected_value in zip(offsets, replacement_ids):
            offset = int(offset_value)
            expected = int(expected_value)
            checked_replacements += 1
            if offset < 0 or offset + 1 >= active:
                raise SupportDiagnosticError(f"recorded replacement offset is outside captured active sequence at slot {slot}")
            observed = int(token_ids[slot, offset + 1].item())
            if observed != expected:
                mismatches.append(
                    {
                        "slot": slot,
                        "record_id": str(row.get("record_id", "")),
                        "offset_zero_based_after_bos": offset,
                        "position_one_based_after_bos": offset + 1,
                        "expected_token_id": expected,
                        "observed_token_id": observed,
                    }
                )
    complete = (
        checked_records == EXPECTED_CONTROLLED_RECORDS
        and checked_replacements == EXPECTED_CONTROLLED_REPLACEMENTS
    )
    return {
        "status": "PASS" if complete and not mismatches else "FAIL",
        "checked_controlled_records": checked_records,
        "checked_replacement_occurrences": checked_replacements,
        "expected_controlled_records": EXPECTED_CONTROLLED_RECORDS,
        "expected_replacement_occurrences": EXPECTED_CONTROLLED_REPLACEMENTS,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "coordinate_binding": "corpus offset o is compared with captured token_ids[slot, o+1]",
        "captured_token_ids_used": True,
        "private_truth_accessed": False,
    }


def _validation_projection(
    *,
    state_path: Path,
    embedding_path: Path,
    validation_tensor_path: Path,
    validation_rows: Sequence[Mapping[str, Any]],
    fit_counts: Mapping[int, int],
    output_root: Path,
    method_id: str = "affine_trained_diagonal_attention128",
    batch_records: int = 1,
) -> dict[str, Any]:
    """Run a frozen decoder only on the public development validation asset."""

    if method_id != "affine_trained_diagonal_attention128":
        raise SupportDiagnosticError("TRR-0007 projection is restricted to the frozen trained-diagonal state")
    if batch_records != 1:
        raise SupportDiagnosticError("public projection uses one-record chunks for bounded CPU memory")
    from token_reconstruction.trr0005_joint_decoder import load_decoder_state

    state_path = Path(state_path).expanduser().resolve()
    embedding_path = Path(embedding_path).expanduser().resolve()
    validation_tensor_path = Path(validation_tensor_path).expanduser().resolve()
    with safe_open(str(validation_tensor_path), framework="pt", device="cpu") as handle:
        if "activations" not in set(handle.keys()):
            raise SupportDiagnosticError("validation artifact has no activations tensor")
        activations = handle.get_tensor("activations").detach().cpu().float().contiguous()
        labels = handle.get_tensor("token_ids").detach().cpu().long().contiguous()
        mask = handle.get_tensor("attention_mask").detach().cpu().bool().contiguous()
    if tuple(activations.shape[:2]) != tuple(labels.shape) or tuple(mask.shape) != tuple(labels.shape):
        raise SupportDiagnosticError("validation activation/label/mask geometry differs")
    with safe_open(str(embedding_path), framework="pt", device="cpu") as handle:
        if "embeddings" not in set(handle.keys()):
            raise SupportDiagnosticError("embedding table has no embeddings tensor")
        embedding = handle.get_tensor("embeddings").detach().cpu().float().contiguous()
    if tuple(embedding.shape) != (VOCAB_SIZE, int(activations.shape[-1])):
        raise SupportDiagnosticError("embedding table geometry does not match validation activations")
    model = load_decoder_state(
        state_path,
        method_id=method_id,
        hidden_size=int(activations.shape[-1]),
        vocabulary_size=VOCAB_SIZE,
        context_width=128,
    ).cpu().eval()
    model.requires_grad_(False)
    predictions = torch.full(labels.shape, PAD_TOKEN_ID, dtype=torch.int64)
    cells: defaultdict[tuple[str, str, str], dict[str, Any]] = defaultdict(_new_cell)
    records_correct: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(validation_rows):
            valid = mask[index]
            positions = torch.nonzero(valid, as_tuple=False).flatten()
            positions = positions[positions > 0]
            if positions.numel() == 0:
                raise SupportDiagnosticError(f"validation row {index} has no post-BOS positions")
            one_activation = activations[index : index + 1]
            one_mask = mask[index : index + 1]
            projected = model.projected_hidden(one_activation, one_mask)
            record_slots = torch.zeros_like(positions)
            logits = model.logits_from_rows(projected, record_slots, positions, embedding)
            predicted = logits.argmax(dim=-1).to(dtype=torch.int64)
            predictions[index, positions] = predicted
            correct = predicted.eq(labels[index, positions])
            records_correct.append(int(correct.all().item()))
            style = _style(row)
            for local, position in enumerate(positions.tolist()):
                token = int(labels[index, position].item())
                position_name = _position_bin(int(position))
                frequency_name = _frequency_bin(int(fit_counts.get(token, 0)))
                cell = cells[(style, position_name, frequency_name)]
                cell["examples"] += 1
                cell["distinct_token_ids"].add(token)
                hit = bool(correct[local].item())
                cell["scored"] = True
                cell["correct"] += int(hit)
                cell["errors"] += int(not hit)
    # Keep retrospective error tables structurally comparable to support-only
    # tables by materializing observed-zero style/position/frequency cells.
    for style in sorted({_style(row) for row in validation_rows}):
        for position_name, _lower, _upper in POSITION_BINS:
            for frequency_name, _frequency_lower, _frequency_upper in FREQUENCY_BINS:
                cells.setdefault((style, position_name, frequency_name), _new_cell())
    output_root = Path(output_root).expanduser().resolve()
    prediction_path = output_root / "public_validation_predictions.safetensors"
    if prediction_path.exists() or prediction_path.is_symlink():
        raise SupportDiagnosticError(f"projection prediction artifact is create-only: {prediction_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    save_file(
        {"predictions": predictions},
        str(prediction_path),
        metadata={
            "schema": PROJECTION_SCHEMA,
            "task_id": TASK_ID,
            "method_id": method_id,
            "source": "public TRR-0004 development validation only",
            "private_truth_accessed": "false",
        },
    )
    elapsed = time.perf_counter() - started
    total_examples = sum(int(cell["examples"]) for cell in cells.values())
    total_correct = sum(int(cell.get("correct", 0)) for cell in cells.values())
    return {
        "schema": PROJECTION_SCHEMA,
        "task_id": TASK_ID,
        "status": "RETROSPECTIVE_PUBLIC_DEVELOPMENT_ONLY",
        "method_id": method_id,
        "state": {"path": str(state_path), "bytes": int(state_path.stat().st_size), "sha256": file_sha256(state_path)},
        "embedding": {"path": str(embedding_path), "bytes": int(embedding_path.stat().st_size), "sha256": file_sha256(embedding_path)},
        "validation_tensor": {"path": str(validation_tensor_path), "bytes": int(validation_tensor_path.stat().st_size), "sha256": file_sha256(validation_tensor_path)},
        "geometry": {
            "records": len(validation_rows),
            "post_bos_positions": int(mask[:, 1:].sum().item()),
            "evaluation_range_positions": int(mask[:, 1 : EXPECTED_EVAL_LAST_POSITION + 1].sum().item()),
            "position_scope": "post-BOS offsets 1..191; first-127 subset reported separately",
        },
        "metrics": {
            "token_examples": total_examples,
            "correct_tokens": total_correct,
            "errors": total_examples - total_correct,
            "token_accuracy": total_correct / total_examples if total_examples else None,
            "exact_records": sum(records_correct),
            "exact_record_count": len(records_correct),
        },
        "joint_error_cells": _finish_cells(cells),
        "prediction_artifact": {
            "path": str(prediction_path),
            "bytes": int(prediction_path.stat().st_size),
            "sha256": file_sha256(prediction_path),
            "contains_truth": False,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "device": "cpu",
            "batch_records": batch_records,
            "candidate_simulations": 0,
            "public_prefix_calls": 0,
        },
        "truth_separation": {
            "public_development_labels_used_post_hoc": True,
            "private_truth_accessed": False,
            "fresh_evaluation_records_loaded": False,
            "selection_or_fit_updated": False,
        },
    }


def _hash_slot_selection(indices: Sequence[int]) -> str:
    return _digest_lines(str(int(index)) for index in indices)


def _load_frequency_reference(path: Path) -> dict[str, Any]:
    """Load the existing public TRR-0005 original/enriched frequency maps."""

    value = _json(path, label="public frequency reference")
    if value.get("schema") != "token-reconstruction.trr0005-frequency-references.v1":
        raise SupportDiagnosticError("public frequency reference schema is not TRR-0005 v1")
    references = value.get("frequency_references")
    if not isinstance(references, Mapping):
        raise SupportDiagnosticError("public frequency reference has no frequency_references map")
    try:
        enriched = {int(token_id): int(count) for token_id, count in references["enriched"].items()}
        original = {int(token_id): int(count) for token_id, count in references["original"].items()}
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise SupportDiagnosticError("public frequency reference maps are malformed") from exc
    if any(token_id < 0 or token_id >= VOCAB_SIZE for token_id in enriched):
        raise SupportDiagnosticError("public enriched frequency map contains an invalid token ID")
    if any(count <= 0 for count in enriched.values()) or any(count <= 0 for count in original.values()):
        raise SupportDiagnosticError("public frequency map contains a non-positive count")
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
        "schema": value["schema"],
        "enriched": enriched,
        "original": original,
    }


def _load_candidate_pool_frequency(path: Path) -> dict[str, Any]:
    """Load token frequencies from the permitted public fit-frequency pools."""

    value = _json(path, label="public candidate-pool frequency")
    if value.get("schema") != CANDIDATE_FREQUENCY_SCHEMA:
        raise SupportDiagnosticError("public candidate-pool frequency schema is not TRR-0007 v1")
    raw = value.get("frequency_by_token_id")
    if not isinstance(raw, Mapping):
        raise SupportDiagnosticError("public candidate-pool frequency has no frequency_by_token_id map")
    try:
        frequencies = {int(token_id): int(count) for token_id, count in raw.items()}
    except (TypeError, ValueError) as exc:
        raise SupportDiagnosticError("public candidate-pool frequency map is malformed") from exc
    if any(token_id < 0 or token_id >= VOCAB_SIZE for token_id in frequencies):
        raise SupportDiagnosticError("public candidate-pool frequency contains an invalid token ID")
    if any(count <= 0 for count in frequencies.values()):
        raise SupportDiagnosticError("public candidate-pool frequency contains a non-positive count")
    raw_special = value.get("special_token_ids_excluded_by_selection", [BOS_TOKEN_ID, PAD_TOKEN_ID])
    if not isinstance(raw_special, list):
        raise SupportDiagnosticError("public candidate-pool special-token list is malformed")
    try:
        special = sorted({int(token_id) for token_id in raw_special})
    except (TypeError, ValueError) as exc:
        raise SupportDiagnosticError("public candidate-pool special-token list is malformed") from exc
    if any(token_id < 0 or token_id >= VOCAB_SIZE for token_id in special):
        raise SupportDiagnosticError("public candidate-pool special-token list contains an invalid token ID")
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
        "schema": value["schema"],
        "frequencies": frequencies,
        "special_token_ids": special,
        "source_pools": value.get("source_pools"),
        "exclusion_binding": value.get("exclusion_binding"),
    }


def _broader_identity_pool(
    plan_rows: Sequence[Mapping[str, Any]] | None,
    frequency_reference: Mapping[str, Any],
    candidate_pool_frequency: Mapping[str, Any] | None = None,
    *,
    target_count: int = PROPOSED_CONTROLLED_TOKEN_IDS,
    baseline_count: int = EXPECTED_CONTROLLED_TOKEN_IDS,
) -> dict[str, Any]:
    """Build a current-unseen public identity pool without error labels."""

    if target_count <= baseline_count or baseline_count <= 0:
        raise SupportDiagnosticError("broader identity pool counts are invalid")
    if candidate_pool_frequency is None:
        return {
            "status": "not_computed",
            "reason": (
                "current-unseen additions require a read-only public fit-candidate frequency map; "
                "legacy-absent IDs are insufficient because they may already occur in the enriched bank"
            ),
            "reference": {
                "path": str(frequency_reference["path"]),
                "bytes": int(frequency_reference["bytes"]),
                "sha256": str(frequency_reference["sha256"]),
                "schema": str(frequency_reference["schema"]),
            },
            "private_truth_accessed": False,
        }
    enriched = {int(token_id): int(count) for token_id, count in frequency_reference["enriched"].items()}
    candidate_frequencies = {
        int(token_id): int(count)
        for token_id, count in candidate_pool_frequency["frequencies"].items()
    }
    structural = {BOS_TOKEN_ID, PAD_TOKEN_ID, *candidate_pool_frequency.get("special_token_ids", [])}
    candidates = [
        token_id
        for token_id, count in candidate_frequencies.items()
        if count > 0 and token_id not in structural and token_id not in enriched
    ]
    ordered = sorted(candidates, key=lambda token_id: (-candidate_frequencies[token_id], token_id))
    current_order: list[int] = []
    if plan_rows is not None:
        for row in plan_rows:
            if not bool(row.get("synthetic", False)):
                continue
            replacement_ids = row.get("replacement_token_ids", [])
            if not isinstance(replacement_ids, list):
                raise SupportDiagnosticError("current controlled replacement IDs are malformed")
            for value in replacement_ids:
                token_id = int(value)
                if token_id not in current_order:
                    current_order.append(token_id)
        if len(current_order) != baseline_count:
            raise SupportDiagnosticError("current controlled identity count changed")
        if any(token_id not in enriched for token_id in current_order):
            raise SupportDiagnosticError("current controlled identity is absent from current enriched references")
    additional_count = target_count - len(current_order)
    additional = ordered[:additional_count]
    if len(additional) != additional_count:
        raise SupportDiagnosticError(
            f"public fit-candidate pool has only {len(additional)} current-unseen IDs; need {additional_count}"
        )
    selected = current_order + additional
    if set(additional) & set(enriched):
        raise SupportDiagnosticError("broader additions are not current-unseen IDs")
    return {
        "status": "computed",
        "reference": {
            "enriched_fit_frequency": {
                "path": str(frequency_reference["path"]),
                "bytes": int(frequency_reference["bytes"]),
                "sha256": str(frequency_reference["sha256"]),
                "schema": str(frequency_reference["schema"]),
            },
            "candidate_pool_frequency": {
                "path": str(candidate_pool_frequency["path"]),
                "bytes": int(candidate_pool_frequency["bytes"]),
                "sha256": str(candidate_pool_frequency["sha256"]),
                "schema": str(candidate_pool_frequency["schema"]),
            },
        },
        "criteria": (
            "candidate-pool post-BOS frequency >0, absent from the current enriched fit frequency map, "
            "BOS/PAD excluded; rank by descending candidate-pool frequency then token ID; no error labels used"
        ),
        "candidate_count": len(candidates),
        "candidate_pool_distinct_token_ids": len(candidate_frequencies),
        "baseline_identity_count": len(current_order),
        "target_identity_count": target_count,
        "additional_identity_count": len(additional),
        "baseline_ids_sha256": _digest_lines(str(token_id) for token_id in current_order),
        "additional_ids_sha256": _digest_lines(str(token_id) for token_id in additional),
        "selected_ids_sha256": _digest_lines(str(token_id) for token_id in selected),
        "baseline_ids_preserved": bool(current_order) and set(current_order).issubset(selected),
        "additional_ids_currently_unseen": not (set(additional) & set(enriched)),
        "selected_token_ids": selected,
        "private_truth_accessed": False,
    }


def _planned_replacement_positions(
    source_token_ids: Sequence[int],
    *,
    target_post_bos_token_count: int,
    record_key: str,
    seed: int = 7007,
    structural_token_ids: Iterable[int] = (BOS_TOKEN_ID, PAD_TOKEN_ID),
) -> tuple[int, ...]:
    """Return the content-blind improved position plan as zero-based offsets.

    The public recipe is expressed in one-based post-BOS positions, while the
    existing corpus plan stores zero-based offsets after BOS.  A per-record
    digest rotates the evenly spaced choices within each bin so rows do not all
    receive the same positions.  Token values are consulted only to exclude
    structural BOS/PAD values, never to select rare or error-associated IDs.
    """

    if target_post_bos_token_count < 127:
        raise SupportDiagnosticError("improved position plan requires a target length covering positions 1..127")
    ids = tuple(int(value) for value in source_token_ids)
    if not ids or ids[0] != BOS_TOKEN_ID:
        raise SupportDiagnosticError("improved position plan source must begin with BOS")
    if len(ids) < target_post_bos_token_count + 1:
        raise SupportDiagnosticError("improved position plan source is shorter than its target length")
    structural = {int(value) for value in structural_token_ids}
    offsets: list[int] = []
    for name, lower, upper, quota in RECIPE_POSITION_PLAN:
        if upper > target_post_bos_token_count:
            raise SupportDiagnosticError(f"target length cannot cover position bin {name}")
        eligible = [
            position
            for position in range(lower, upper + 1)
            if ids[position] not in structural
        ]
        if len(eligible) < quota:
            raise SupportDiagnosticError(
                f"only {len(eligible)} ordinary positions are available in improved bin {name}; need {quota}"
            )
        digest = hashlib.sha256(f"{TASK_ID}|{seed}|{record_key}|{name}".encode("utf-8")).hexdigest()
        shift = int(digest[:16], 16) % len(eligible)
        chosen: list[int] = []
        for index in range(quota):
            candidate_index = (shift + int(math.floor((index + 0.5) * len(eligible) / quota))) % len(eligible)
            candidate = eligible[candidate_index]
            if candidate in chosen:
                raise SupportDiagnosticError(f"improved position plan duplicated position in bin {name}")
            chosen.append(candidate)
        offsets.extend(position - 1 for position in chosen)
    if len(offsets) != EXPECTED_CONTROLLED_REPLACEMENTS // EXPECTED_CONTROLLED_RECORDS:
        raise SupportDiagnosticError("improved position plan does not contain 30 replacements")
    if len(set(offsets)) != len(offsets):
        raise SupportDiagnosticError("improved position plan duplicated an offset across bins")
    return tuple(sorted(offsets))


def _recipe(
    fit_rows: Sequence[Mapping[str, Any]],
    *,
    plan_rows: Sequence[Mapping[str, Any]] | None = None,
    frequency_reference: Mapping[str, Any] | None = None,
    candidate_pool_frequency: Mapping[str, Any] | None = None,
    seed: int = 7007,
) -> dict[str, Any]:
    if len(fit_rows) != EXPECTED_FIT_RECORDS:
        raise SupportDiagnosticError("recipe requires the 1200-row enriched bank")
    lengths = {
        int(row.get("slot", index)): int(row.get("post_bos_token_count", row.get("target_post_bos_token_count", -1)))
        for index, row in enumerate(fit_rows)
    }
    if sorted(lengths) != list(range(EXPECTED_FIT_RECORDS)):
        raise SupportDiagnosticError("recipe requires contiguous fit slots")
    if any(value < 40 or value > 191 for value in lengths.values()):
        raise SupportDiagnosticError("fit length vector falls outside the public geometry")
    # Bind to the existing plan when available.  This preserves the current
    # natural source identities and ordered length vector; changing controlled
    # slots would confound a position-support comparison with source matching.
    selection_basis = "deterministic length-qualified fallback; no corpus plan supplied"
    current_replacement_summary: dict[str, Any] | None = None
    if plan_rows is not None:
        if len(plan_rows) != EXPECTED_FIT_RECORDS:
            raise SupportDiagnosticError("current corpus plan must contain the 1200 fit rows")
        plan_by_slot = {int(row.get("slot", -1)): row for row in plan_rows}
        if sorted(plan_by_slot) != list(range(EXPECTED_FIT_RECORDS)):
            raise SupportDiagnosticError("current corpus plan slots are not contiguous")
        for index, row in enumerate(fit_rows):
            slot = int(row.get("slot", index))
            plan_row = plan_by_slot[slot]
            fit_length = int(row.get("post_bos_token_count", row.get("target_post_bos_token_count", -1)))
            plan_length = int(plan_row.get("target_post_bos_token_count", -1))
            if fit_length != plan_length:
                raise SupportDiagnosticError(f"fit and corpus-plan lengths differ at slot {slot}")
        current_synthetic = [row for row in plan_rows if bool(row.get("synthetic", False))]
        domain_counts = Counter(str(row.get("domain", "unknown")) for row in current_synthetic)
        if len(current_synthetic) != EXPECTED_CONTROLLED_RECORDS:
            raise SupportDiagnosticError("current plan controlled record count changed")
        if domain_counts != Counter({"controlled_pile_context": 60, "controlled_finance_context": 60}):
            raise SupportDiagnosticError("current plan controlled domain counts changed")
        current_replacement_summary = _replacement_support(plan_rows)
        if current_replacement_summary["replacement_occurrences"] != EXPECTED_CONTROLLED_REPLACEMENTS:
            raise SupportDiagnosticError("current plan replacement count changed")
        if current_replacement_summary["distinct_replacement_token_ids"] != EXPECTED_CONTROLLED_TOKEN_IDS:
            raise SupportDiagnosticError("current plan controlled token-ID count changed")
        selected = sorted(int(row["slot"]) for row in current_synthetic)
        if any(lengths[index] < 128 for index in selected):
            raise SupportDiagnosticError("current controlled slots must cover one-based positions 1..127")
        selection_basis = "reuse current TRR-0005 controlled slot geometry and natural complement; regenerate controlled parent contexts, IDs, and offsets"
    else:
        def key(index: int) -> str:
            return hashlib.sha256(f"{TASK_ID}|{seed}|controlled-slot|{index}".encode("utf-8")).hexdigest()

        candidates = [index for index, length in lengths.items() if length >= 128]
        ordered = sorted(candidates, key=lambda index: (key(index), index))
        selected = sorted(ordered[:EXPECTED_CONTROLLED_RECORDS])
        if len(selected) != EXPECTED_CONTROLLED_RECORDS:
            raise SupportDiagnosticError("not enough target slots of length >=128 for recipe")
        selection_basis = "deterministic hash ordering over target slots of length >=128"
    if len(selected) != EXPECTED_CONTROLLED_RECORDS or len(set(selected)) != len(selected):
        raise SupportDiagnosticError("recipe controlled slot selection is not unique")
    selected_set = set(selected)
    natural_slots = [index for index in range(EXPECTED_FIT_RECORDS) if index not in selected_set]
    parent_capacity = {"pile": 0, "finance": 0}
    if plan_rows is not None:
        used_sources = {"pile": set(), "finance": set()}
        for row in plan_rows:
            dataset = str(row.get("dataset_key", ""))
            source_id = str(row.get("source_record_id", ""))
            if dataset not in used_sources or ":row-" not in source_id:
                continue
            try:
                used_sources[dataset].add(int(source_id.rsplit(":row-", 1)[1]))
            except ValueError:
                raise SupportDiagnosticError(f"cannot parse public source row index: {source_id}")
        parent_capacity = {
            "pile": (7000 - 2000) - len(used_sources["pile"]),
            "finance": (12000 - 2000) - len(used_sources["finance"]),
        }
    if len(natural_slots) != EXPECTED_FIT_RECORDS - EXPECTED_CONTROLLED_RECORDS:
        raise SupportDiagnosticError("recipe natural-slot complement changed")
    strata_receipt: list[dict[str, Any]] = []
    for name, lower, upper, _legacy_quota in RECIPE_LENGTH_STRATA:
        available = [index for index, length in lengths.items() if lower <= length <= upper]
        chosen = [index for index in selected if lower <= lengths[index] <= upper]
        strata_receipt.append(
            {
                "name": name,
                "lower": lower,
                "upper": upper,
                "available_fit_slots": len(available),
                "selected_controlled_slots": len(chosen),
                "selected_slots": chosen,
                "selected_post_bos_lengths": [lengths[index] for index in chosen],
            }
        )
    identity_selection = (
        _broader_identity_pool(plan_rows, frequency_reference, candidate_pool_frequency)
        if frequency_reference is not None
        else {
            "status": "not_computed",
            "reason": "pass the existing public frequency reference to materialize the broader identity pool",
            "target_identity_count": PROPOSED_CONTROLLED_TOKEN_IDS,
            "additional_identity_count": PROPOSED_ADDITIONAL_TOKEN_IDS,
            "private_truth_accessed": False,
        }
    )
    return {
        "schema": RECIPE_SCHEMA,
        "task_id": TASK_ID,
        "status": "METADATA_RECIPE_PENDING_REAL_P0_FORWARD",
        "created_utc": utc_now(),
        "purpose": (
            "Test fitting support versus decoder capacity while preserving the current enriched "
            "natural bank, total post-BOS opportunities, and deterministic optimization exposure."
        ),
        "slot_selection": {
            "policy": selection_basis,
            "current_plan_bound": plan_rows is not None,
            "controlled_slot_count": len(selected),
            "natural_slot_count": len(natural_slots),
            "controlled_target_length_requirement": "every selected slot has post-BOS length >=128",
            "selected_target_length_min": min(lengths[index] for index in selected),
            "selected_target_length_max": max(lengths[index] for index in selected),
        },
        "baseline_binding": {
            "fit_record_count": EXPECTED_FIT_RECORDS,
            "post_bos_positions": EXPECTED_POST_BOS,
            "stored_width": EXPECTED_SEQUENCE_LENGTH,
            "natural_records": EXPECTED_FIT_RECORDS - EXPECTED_CONTROLLED_RECORDS,
            "controlled_records": EXPECTED_CONTROLLED_RECORDS,
            "controlled_replacement_occurrences": EXPECTED_CONTROLLED_REPLACEMENTS,
            "controlled_token_id_target": EXPECTED_CONTROLLED_TOKEN_IDS,
            "proposed_controlled_token_id_target": PROPOSED_CONTROLLED_TOKEN_IDS,
            "proposed_additional_token_ids": PROPOSED_ADDITIONAL_TOKEN_IDS,
            "draw_schedule": {
                "batch_size": EXPECTED_DRAW_BATCH,
                "steps": EXPECTED_DRAW_STEPS,
                "draws": EXPECTED_DRAW_BATCH * EXPECTED_DRAW_STEPS,
                "position_scope": "post_bos_only",
                "schedule_identity": "same ordered length vector and same seed 4005; require byte-identical schedule receipt",
            },
        },
        "common_error_frequency_binding": {
            "status": "frozen_public_original_enriched_reference",
            "reference": (
                identity_selection.get("reference")
                if identity_selection.get("status") == "computed"
                else None
            ),
            "rule": (
                "Classify baseline and broader-arm errors with the same current original/enriched public "
                "frequency map. Do not recompute bins from each candidate bank or treat unseen-to-seen "
                "relabeling as an accuracy improvement."
            ),
            "selection_independent_of_error_labels": True,
            "private_truth_accessed": False,
        },
        "natural_component": {
            "policy": "retain the existing 1080 natural public source rows and their public rendered identities",
            "records_by_domain": {
                "alpaca_instruction": 600,
                "pile_natural": 300,
                "finance_instruction": 180,
            },
            "source_partitions": {
                "alpaca": "registered TRR-0004 fit bank only",
                "pile": {"dataset": "NeelNanda/pile-10k", "split": "train", "revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa", "allowed_rows": "[2000,7000)"},
                "finance": {"dataset": "Josephgflowers/Finance-Instruct-500k", "split": "train", "revision": "583a98fb0ec14d904e9423b671d9d0fea88891b6", "allowed_rows": "[2000,12000)"},
            },
            "assignment": (
                "when bound to the current corpus plan, keep every natural source row at its existing "
                "slot; otherwise assign only the deterministic controlled-slot complement and fail "
                "closed if a public source row cannot fill its target length"
            ),
        },
        "controlled_component": {
            "records_by_domain": {"controlled_pile_context": 60, "controlled_finance_context": 60},
            "parent_source_policy": (
                "retain all 1080 natural rows, but for the broader arm choose 120 new public Pile/Finance "
                "parent contexts from unused fit-partition rows, 60 per domain, after excluding every current "
                "TRR-0005 fit source ID and all registered public-development IDs"
            ),
            "parent_context_selection": {
                "records_by_domain": {"controlled_pile_context": 60, "controlled_finance_context": 60},
                "available_fit_rows_after_current_fit_exclusion": parent_capacity,
                "allowed_fit_partitions": {
                    "pile": {"dataset": "NeelNanda/pile-10k", "split": "train", "revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa", "half_open_range": [2000, 7000]},
                    "finance": {"dataset": "Josephgflowers/Finance-Instruct-500k", "split": "train", "revision": "583a98fb0ec14d904e9423b671d9d0fea88891b6", "half_open_range": [2000, 12000]},
                },
                "selection": (
                    "stable SHA256 ordering over dataset/revision/row keys, excluding current-fit and public-development "
                    "source IDs; scan the ordered public rows, retain rows with at least the target full-token count, "
                    "then clip each selected sequence to the exact frozen target length as in TRR-0005; fail closed "
                    "if 60 rows per domain cannot be filled"
                ),
                "length_matching": "candidate full-token count >= target_post_bos_token_count + 1; clip after BOS to target length",
                "current_fit_rows_excluded_before_length_matching": True,
                "fresh_holdout_rows_scanned": False,
                "target_weights_accessed": False,
            },
            "token_id_policy": (
                "preserve the current 2000 ordinary legacy-absent IDs and add 1600 distinct ordinary IDs from the "
                "public original/enriched frequency reference; exclude BOS/PAD and rank without error labels"
            ),
            "identity_pool": identity_selection,
            "replacement_count_per_record": 30,
            "replacement_offset_policy": {
                "coordinate": "one-based post-BOS positions in the recipe; store zero-based offset = position - 1",
                "position_bins": [
                    {"name": name, "lower": lower, "upper": upper, "per_record_quota": quota}
                    for name, lower, upper, quota in RECIPE_POSITION_PLAN
                ],
                "per_record_total": sum(quota for _name, _lower, _upper, quota in RECIPE_POSITION_PLAN),
                "all_replacements_in_evaluation_range": True,
                "selection": (
                    "for each bin, remove structural BOS/PAD positions, let N be the remaining ordered "
                    "positions, rotate by a SHA256 digest of task/seed/record/bin, choose index "
                    "floor((i+0.5)*N/quota) modulo N for i=0..quota-1, and store each chosen position minus one"
                ),
                "materialization": "run the registered real P0 forward on each complete constructed sequence; fail closed if any bin has fewer ordinary positions than its quota",
            },
            "target_length_strata": strata_receipt,
            "real_forward_requirement": {
                "required": True,
                "model": "registered public P0 checkpoint and ContiguousPublicPrefix.forward_full at cut 4",
                "one_forward_per_complete_constructed_sequence": True,
                "activation_splicing_forbidden": True,
                "target_weights_accessed": False,
            },
        },
        "matched_comparison_requirements": [
            "Keep 1200 records and 124371 post-BOS positions exactly.",
            "Keep the ordered 192-column mask and BOS/PAD geometry exactly.",
            "Use the same 3000-step x 512-draw full-vocabulary CE opportunity budget.",
            "Retain all natural text rows and current domain quotas; use new public controlled parent contexts only in the broader arm and report exact source IDs.",
            "Record current versus proposed distinct IDs, occurrences, joint style x position x frequency coverage, and repeated draws separately.",
            "Freeze both decoder choices and public validation selection before any fresh evaluation truth.",
        ],
        "selected_controlled_slot_indices": selected,
        "selected_controlled_slot_indices_sha256": _hash_slot_selection(selected),
        "natural_complement_slot_indices_sha256": _hash_slot_selection(natural_slots),
        "exclusion_policy": {
            "fresh_evaluation_population": {
                "pile_rows_reserved_and_uninspected": "[7000,10000)",
                "finance_rows_reserved_and_uninspected": "[12000,20000)",
            },
            "public_development_validation": "all registered TRR-0004 validation record IDs and sequence hashes are excluded from new fit selection and fresh evaluation",
            "current_fit_bank": "all current TRR-0005 enriched record/source/sequence identities are excluded from any new fresh panel",
            "private_truth": "never loaded",
        },
        "decision_interpretation": {
            "support_helps": "If the matched-capacity model improves on the same frozen public-development and fresh natural inputs, prioritize support before extra architecture.",
            "capacity_helps": "If the stronger positionwise inverse improves at matched support/opportunities, advance it with measured state and runtime costs.",
            "interaction": "If only the improved bank and stronger model together improve, report the interaction.",
            "neither": "If neither moves the gap under informative public fitting, propose a different inverse mechanism rather than unbounded training.",
        },
    }


def _exclusion_manifest(
    fit_tokens: torch.Tensor,
    fit_mask: torch.Tensor,
    fit_rows: Sequence[Mapping[str, Any]],
    validation_tokens: torch.Tensor | None,
    validation_mask: torch.Tensor | None,
    validation_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    def entries(tokens: torch.Tensor, mask: torch.Tensor, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        bool_mask = mask.to(dtype=torch.bool)
        for index, row in enumerate(rows):
            active = int(bool_mask[index].sum().item())
            ids = [int(value) for value in tokens[index, :active].tolist()]
            result.append(
                {
                    "slot": int(row.get("slot", index)),
                    "record_id": str(row["record_id"]),
                    "source_record_id": str(row.get("source_record_id", row["record_id"])),
                    "rendered_sha256": row.get("rendered_sha256"),
                    "sequence_sha256": _sequence_digest(ids),
                    "full_token_count": active,
                    "post_bos_token_count": active - 1,
                }
            )
        return result

    fit_entries = entries(fit_tokens, fit_mask, fit_rows)
    result: dict[str, Any] = {
        "schema": EXCLUSION_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_METADATA_EXCLUSIONS_FOR_EVALUATOR",
        "created_utc": utc_now(),
        "sequence_hash_algorithm": "SHA256 of newline-delimited decimal token IDs for active BOS-inclusive sequence",
        "fit_bank": {
            "role": "public auxiliary fit exclusion",
            "records": fit_entries,
            "record_ids_sha256": _digest_lines(entry["record_id"] for entry in fit_entries),
            "source_record_ids_sha256": _digest_lines(entry["source_record_id"] for entry in fit_entries),
            "sequence_hashes_sha256": _digest_lines(entry["sequence_sha256"] for entry in fit_entries),
            "record_count": len(fit_entries),
        },
        "reserved_future_ranges": {
            "role": "declared fresh evaluation reserve; row contents never inspected by this task",
            "pile": {"dataset": "NeelNanda/pile-10k", "split": "train", "revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa", "half_open_range": [7000, 10000]},
            "finance": {"dataset": "Josephgflowers/Finance-Instruct-500k", "split": "train", "revision": "583a98fb0ec14d904e9423b671d9d0fea88891b6", "half_open_range": [12000, 20000]},
        },
        "truth_separation": {
            "private_truth_accessed": False,
            "fresh_evaluation_truth_accessed": False,
            "target_weights_accessed": False,
            "source_text_retained": False,
        },
    }
    if validation_tokens is not None and validation_mask is not None and validation_rows is not None:
        validation_entries = entries(validation_tokens, validation_mask, validation_rows)
        result["public_development_validation"] = {
            "role": "public development exclusion",
            "records": validation_entries,
            "record_ids_sha256": _digest_lines(entry["record_id"] for entry in validation_entries),
            "source_record_ids_sha256": _digest_lines(entry["source_record_id"] for entry in validation_entries),
            "sequence_hashes_sha256": _digest_lines(entry["sequence_sha256"] for entry in validation_entries),
            "record_count": len(validation_entries),
        }
    return result


def _runtime() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": int(__import__("os").cpu_count() or 0),
        "max_rss_bytes": int(usage.ru_maxrss * 1024),
        "device": "cpu",
        "private_truth_accessed": False,
        "target_weights_accessed": False,
        "network_used": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-tensor", type=Path, required=True)
    parser.add_argument("--fit-records", type=Path, required=True)
    parser.add_argument("--corpus-plan", type=Path)
    parser.add_argument("--frequency-references", type=Path)
    parser.add_argument("--candidate-pool-frequency", type=Path)
    parser.add_argument("--validation-tensor", type=Path)
    parser.add_argument("--validation-records", type=Path)
    parser.add_argument("--embedding-table", type=Path)
    parser.add_argument("--validation-state", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7007)
    parser.add_argument("--project-validation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    if args.project_validation and (args.embedding_table is None or args.validation_state is None):
        raise SystemExit("--project-validation requires --embedding-table and --validation-state")
    if args.validation_tensor is None and args.validation_records is not None:
        raise SystemExit("--validation-records requires --validation-tensor")
    if args.validation_records is None and args.validation_tensor is not None:
        raise SystemExit("--validation-tensor requires --validation-records")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise SystemExit(f"output root is create-only: {output_root}")
    fit_tokens, fit_mask, fit_resource = _load_label_tensors(args.fit_tensor, label="fit")
    fit_rows = _records(args.fit_records, label="fit records")
    fit_summary = _support_for_rows(
        fit_tokens,
        fit_mask,
        fit_rows,
        label="enriched fit bank",
        expected_records=EXPECTED_FIT_RECORDS,
        expected_post_bos=EXPECTED_POST_BOS,
        expected_width=EXPECTED_SEQUENCE_LENGTH,
    )
    plan_summary = None
    plan_rows = None
    replacement_verification = None
    frequency_reference = None
    candidate_pool_frequency = None
    if args.frequency_references is not None:
        frequency_reference = _load_frequency_reference(args.frequency_references)
    if args.candidate_pool_frequency is not None:
        candidate_pool_frequency = _load_candidate_pool_frequency(args.candidate_pool_frequency)
    if args.corpus_plan is not None:
        plan_rows = _plan_records(args.corpus_plan)
        if len(plan_rows) != len(fit_rows):
            raise SupportDiagnosticError("corpus plan and fit records have different row counts")
        plan_summary = _replacement_support(plan_rows)
        replacement_verification = _verify_recorded_replacements(fit_tokens, fit_mask, plan_rows)
    validation_summary = None
    validation_tokens = validation_mask = None
    validation_rows = None
    validation_resource = None
    if args.validation_tensor is not None and args.validation_records is not None:
        validation_tokens, validation_mask, validation_resource = _load_label_tensors(
            args.validation_tensor,
            label="public development validation",
        )
        validation_rows = _records(args.validation_records, label="public development validation records")
        validation_summary = _support_for_rows(
            validation_tokens,
            validation_mask,
            validation_rows,
            frequency_counts=Counter(
                int(value)
                for value in fit_tokens[:, 1:][fit_mask.to(dtype=torch.bool)[:, 1:]].tolist()
            ),
            label="public development validation",
            expected_records=len(validation_rows),
            expected_width=EXPECTED_SEQUENCE_LENGTH,
        )
    recipe = _recipe(
        fit_rows,
        plan_rows=plan_rows,
        frequency_reference=frequency_reference,
        candidate_pool_frequency=candidate_pool_frequency,
        seed=args.seed,
    )
    exclusions = _exclusion_manifest(
        fit_tokens,
        fit_mask,
        fit_rows,
        validation_tokens,
        validation_mask,
        validation_rows,
    )
    # Reserve the create-only output directory before optional projection,
    # which writes its prediction artifact inside this directory.
    output_root.mkdir(parents=True, exist_ok=False)
    projection = None
    if args.project_validation:
        assert validation_tokens is not None and validation_mask is not None and validation_rows is not None
        projection = _validation_projection(
            state_path=args.validation_state,
            embedding_path=args.embedding_table,
            validation_tensor_path=args.validation_tensor,
            validation_rows=validation_rows,
            fit_counts=Counter(
                int(value)
                for value in fit_tokens[:, 1:][fit_mask.to(dtype=torch.bool)[:, 1:]].tolist()
            ),
            output_root=output_root,
        )
    support_payload = {
        "schema": SUPPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_FIT_SUPPORT_DESCRIPTIVE; NO_PRIVATE_TRUTH",
        "created_utc": utc_now(),
        "fit_resource": fit_resource,
        "fit_summary": fit_summary,
        "controlled_replacement_summary": plan_summary,
        "recorded_replacement_verification": replacement_verification,
        "public_development_validation": validation_summary,
        "recipe": {
            "path": str(output_root / "improved_public_sampling_recipe.json"),
            "schema": RECIPE_SCHEMA,
        },
        "exclusions": {
            "path": str(output_root / "public_exclusion_manifest.json"),
            "schema": EXCLUSION_SCHEMA,
        },
        "projection": projection,
        "exclusion_distinctions": {
            "bos_positions_excluded_from_supervision": int(fit_summary["geometry"]["bos_positions"]),
            "padding_positions_excluded_from_supervision": int(fit_summary["geometry"]["padding_positions"]),
            "post_bos_positions_outside_evaluation_range_128_191": int(fit_summary["coverage"]["post_evaluation_positions"]),
            "invalid_rows_excluded": 0,
            "private_truth_or_final_panel_rows_loaded": 0,
            "public_development_labels_are_descriptive_only": True,
        },
        "runtime": {"elapsed_seconds": time.perf_counter() - started, **_runtime()},
    }
    for path, payload in (
        (output_root / "support_summary.json", support_payload),
        (output_root / "improved_public_sampling_recipe.json", recipe),
        (output_root / "public_exclusion_manifest.json", exclusions),
    ):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if projection is not None:
        (output_root / "public_development_projection.json").write_text(
            json.dumps(projection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(support_payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
