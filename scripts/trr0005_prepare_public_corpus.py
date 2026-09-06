#!/usr/bin/env python3
"""Prepare the two public fitting streams frozen for TRR-0005.

The command is intentionally separate from activation capture and fitting.  It
reads only explicitly named public cache rows, computes public token-frequency
metadata, writes a padded token plan, and leaves the public-prefix forward to
the producer.  No evaluator truth, target checkpoint, or future holdout row is
used here.

The command requires ``--execute`` before it will read/tokenize caches.  This
keeps design and code review cheap while the coordinator reserves the bounded
CPU/GPU window for the actual preparation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any

from token_reconstruction.alpaca_split import historical_rendered_text
from token_reconstruction.trr0005_public_corpus import (
    ARMS,
    BOS_TOKEN_ID,
    CONTROLLED_REPLACEMENTS_PER_RECORD,
    CORPUS_SCHEMA,
    ENRICHED_ARM,
    ENRICHED_COUNTS,
    FIT_BATCH_SIZE,
    FIT_SAMPLER_SEED,
    FIT_STEPS,
    MAX_SEQUENCE_LENGTH,
    MIN_ENRICHED_DISTINCT_TOKEN_IDS,
    MIN_LEGACY_ABSENT_CONTROLLED_IDS,
    ORIGINAL_ARM,
    PAD_TOKEN_ID,
    PLAN_SCHEMA,
    POST_BOS_POSITION_COUNT,
    PREPARATION_MAX_SECONDS,
    PublicSourceRow,
    PlannedRecord,
    SOURCE_PARTITIONS,
    SOURCE_DATASETS,
    STORED_ROW_COUNT,
    TRR0005CorpusError,
    apply_replacements,
    controlled_record_id,
    coverage_contrast,
    deterministic_row_order,
    domain_length_summary,
    expected_sampler_exposure,
    length_multiset,
    length_vector_digest,
    load_trr4_length_slots,
    public_source_file_record,
    replacement_positions,
    select_public_token_ids,
    sha256_lines,
    source_record_id,
    stable_public_text_digest,
    token_frequency_summary,
    token_ids_from_encoding,
    validate_fit_only_indices,
    validate_partition_index,
    validate_planned_records,
    validate_source_row,
)


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
FINANCE_DATE_STRING = "06 Aug 2026"
DEFAULT_DESIGN = Path("experiments/TRR-0005/corpus_design.json")
DEFAULT_LENGTH_METADATA = Path("experiments/TRR-0004/alpaca_split_plan.json")
DEFAULT_MAX_SECONDS = PREPARATION_MAX_SECONDS


class PreparationError(RuntimeError):
    """Raised when public corpus preparation cannot satisfy the frozen design."""


@dataclass(frozen=True)
class _Candidate:
    """Tokenized public source row retained only until plan serialization."""

    dataset_key: str
    dataset_id: str
    split: str
    revision: str
    row_index: int
    record_id: str
    domain: str
    rendered_sha256: str
    token_ids: tuple[int, ...]

    @property
    def full_token_count(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class _Deadline:
    started: float
    maximum_seconds: float

    def check(self, phase: str) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed > self.maximum_seconds:
            raise PreparationError(
                f"public preparation exceeded {self.maximum_seconds:.1f}s during {phase}"
            )

    def elapsed(self) -> float:
        return time.monotonic() - self.started


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} must be a regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot parse {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise PreparationError(f"{label} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PreparationError(f"refusing to overwrite preparation output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit(root: Path) -> str | None:
    import subprocess

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


def _load_design(path: Path) -> Mapping[str, Any]:
    design = _load_json(path, label="TRR-0005 corpus design")
    if design.get("task_id") != "TRR-0005":
        raise PreparationError("corpus design task ID changed")
    if design.get("status") not in {"APPROVED_DESIGN_METADATA_ONLY", "APPROVED_FOR_PREPARATION"}:
        raise PreparationError("corpus design is not approved")
    geometry = design.get("geometry")
    if not isinstance(geometry, Mapping):
        raise PreparationError("corpus design has no geometry object")
    expected = {
        "record_count": 1200,
        "max_sequence_length": 192,
        "stored_rows_including_bos": STORED_ROW_COUNT,
        "post_bos_positions": POST_BOS_POSITION_COUNT,
        "cut_depth": 4,
    }
    for key, value in expected.items():
        if int(geometry.get(key, -1)) != value:
            raise PreparationError(f"corpus design geometry changed at {key}")
    return design


def _resolve_source_paths(design: Mapping[str, Any], args: argparse.Namespace) -> dict[str, list[Path]]:
    sources = design.get("public_sources")
    if not isinstance(sources, Mapping):
        raise PreparationError("corpus design has no public_sources")
    result: dict[str, list[Path]] = {}
    alpaca = Path(args.alpaca_arrow) if args.alpaca_arrow else Path(str(sources["alpaca"]["arrow_path"]))
    pile = Path(args.pile_arrow) if args.pile_arrow else Path(str(sources["pile"]["arrow_path"]))
    finance_values = args.finance_arrow or sources["finance"].get("arrow_paths")
    if not isinstance(finance_values, Sequence) or isinstance(finance_values, (str, bytes)):
        raise PreparationError("finance source must provide one or more Arrow paths")
    result["alpaca"] = [alpaca.expanduser().resolve()]
    result["pile"] = [pile.expanduser().resolve()]
    result["finance"] = [Path(str(value)).expanduser().resolve() for value in finance_values]
    return result


def _load_public_dataset(paths: Sequence[Path], *, label: str) -> Any:
    """Map public Arrow shards without invoking an online dataset loader."""

    try:
        from datasets import Dataset, concatenate_datasets

        shards = []
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise PreparationError(f"{label} Arrow shard must be a regular file: {path}")
            shards.append(Dataset.from_file(str(path)))
        if not shards:
            raise PreparationError(f"{label} has no Arrow shards")
        return shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    except PreparationError:
        raise
    except Exception as exc:  # pragma: no cover - dependency-specific details
        raise PreparationError(f"cannot map public {label} Arrow cache") from exc


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(path.expanduser().resolve()), local_files_only=True, use_fast=True
        )
    except Exception as exc:  # pragma: no cover - dependency-specific details
        raise PreparationError("public tokenizer load failed") from exc
    if int(getattr(tokenizer, "bos_token_id", -1)) != BOS_TOKEN_ID:
        raise PreparationError("public tokenizer BOS ID changed")
    return tokenizer


def _special_token_ids(tokenizer: Any) -> set[int]:
    values = {BOS_TOKEN_ID, PAD_TOKEN_ID}
    for value in getattr(tokenizer, "all_special_ids", ()) or ():
        try:
            values.add(int(value))
        except (TypeError, ValueError):
            raise PreparationError("tokenizer all_special_ids contains a non-integer")
    return values


def _tokenize_text(tokenizer: Any, rendered: str, *, add_bos_for_raw: bool = False) -> tuple[int, ...]:
    try:
        ids = token_ids_from_encoding(tokenizer(rendered, add_special_tokens=False))
    except (KeyError, TypeError, TRR0005CorpusError) as exc:
        raise PreparationError("public tokenizer returned malformed IDs") from exc
    if not ids:
        raise PreparationError("public tokenizer returned an empty sequence")
    if ids[0] != BOS_TOKEN_ID:
        if not add_bos_for_raw:
            raise PreparationError("chat-rendered public sequence did not begin with BOS")
        ids = (BOS_TOKEN_ID, *ids)
    return ids


def _finance_rendered_text(row: Mapping[str, Any], tokenizer: Any) -> str:
    """Render Finance rows with the established TRR-0004 public recipe."""

    def text_field(key: str) -> str:
        value = row.get(key, "")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise PreparationError(f"Finance public field {key!r} must be text")
        return value.strip()

    system = text_field("system") or None
    user = text_field("user")
    assistant = text_field("assistant")
    if not user:
        instruction = text_field("instruction")
        input_text = text_field("input")
        user = instruction + (("\n\n" + input_text) if input_text else "")
    if not assistant:
        assistant = text_field("output")
    if not user or not assistant:
        raise PreparationError("Finance public row has no user and assistant content")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    )
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            date_string=FINANCE_DATE_STRING,
        )
    except Exception as exc:
        raise PreparationError("Finance chat-template rendering failed") from exc
    if not isinstance(rendered, str):
        raise PreparationError("Finance chat-template rendering did not return text")
    return rendered


def _render_candidate(
    dataset_key: str,
    row: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[str, tuple[int, ...], str]:
    if dataset_key == "alpaca":
        rendered = historical_rendered_text(row, tokenizer)
        ids = _tokenize_text(tokenizer, rendered)
        return rendered, ids, "alpaca_instruction"
    if dataset_key == "pile":
        text_value = row.get("text")
        if not isinstance(text_value, str) or not text_value:
            raise PreparationError("Pile public row has no non-empty text field")
        ids = _tokenize_text(tokenizer, text_value, add_bos_for_raw=True)
        return text_value, ids, "pile_natural"
    if dataset_key == "finance":
        rendered = _finance_rendered_text(row, tokenizer)
        ids = _tokenize_text(tokenizer, rendered)
        return rendered, ids, "finance_instruction"
    raise PreparationError(f"unknown candidate dataset key: {dataset_key}")


def _candidate(
    dataset_key: str,
    dataset: Any,
    row_index: int,
    tokenizer: Any,
    *,
    deadline: _Deadline,
    excluded_ids: set[str],
    excluded_row_keys: set[tuple[str, int]],
    excluded_hashes: set[str],
    scan_stats: dict[str, Any] | None = None,
) -> _Candidate | None:
    if dataset_key in SOURCE_PARTITIONS:
        validate_partition_index(dataset_key, row_index, role="fit")
    else:
        if dataset_key not in SOURCE_DATASETS:
            raise PreparationError(f"unknown public source dataset: {dataset_key}")
        if not isinstance(row_index, int) or row_index < 0:
            raise PreparationError("public source row index must be a non-negative integer")
    source_spec = SOURCE_DATASETS[dataset_key]
    source_id = source_record_id(
        str(source_spec["dataset_id"]),
        str(source_spec["split"]),
        str(source_spec["revision"]),
        row_index,
    )
    aliases = {
        (dataset_key, row_index),
        (str(source_spec["dataset_id"]), row_index),
    }
    if source_id in excluded_ids or aliases & excluded_row_keys:
        if scan_stats is not None:
            scan_stats["rows_excluded_by_metadata"] = int(scan_stats.get("rows_excluded_by_metadata", 0)) + 1
        return None
    if scan_stats is not None:
        scan_stats["rows_dataset_accessed"] = int(scan_stats.get("rows_dataset_accessed", 0)) + 1
    try:
        row = dataset[row_index]
    except Exception as exc:
        raise PreparationError(f"cannot read public {dataset_key} row {row_index}") from exc
    rendered, ids, domain = _render_candidate(dataset_key, row, tokenizer)
    deadline.check(f"{dataset_key} tokenization")
    digest = stable_public_text_digest(rendered)
    if digest in excluded_hashes:
        if scan_stats is not None:
            scan_stats["rows_excluded_by_hash"] = int(scan_stats.get("rows_excluded_by_hash", 0)) + 1
        return None
    if len(ids) < 32 or len(ids) > MAX_SEQUENCE_LENGTH * 8:
        # Keep unusually long rows in the candidate set because a target window
        # can be cropped; reject only empty/short rows here.
        if len(ids) < 32:
            if scan_stats is not None:
                scan_stats["rows_rejected_short"] = int(scan_stats.get("rows_rejected_short", 0)) + 1
            return None
    if scan_stats is not None:
        scan_stats["rows_eligible"] = int(scan_stats.get("rows_eligible", 0)) + 1
    return _Candidate(
        dataset_key=dataset_key,
        dataset_id=str(source_spec["dataset_id"]),
        split=str(source_spec["split"]),
        revision=str(source_spec["revision"]),
        row_index=row_index,
        record_id=source_id,
        domain=domain,
        rendered_sha256=digest,
        token_ids=tuple(ids),
    )


def _scan_fit_candidates(
    dataset_key: str,
    dataset: Any,
    row_indices: Iterable[int],
    tokenizer: Any,
    *,
    deadline: _Deadline,
    excluded_ids: set[str],
    excluded_row_keys: set[tuple[str, int]],
    excluded_hashes: set[str],
    scan_stats: dict[str, Any] | None = None,
) -> list[_Candidate]:
    if dataset_key in SOURCE_PARTITIONS:
        indices = validate_fit_only_indices(dataset_key, row_indices)
    else:
        values = tuple(int(value) for value in row_indices)
        if any(value < 0 for value in values):
            raise PreparationError("public source row indices must be non-negative")
        if len(set(values)) != len(values):
            raise PreparationError(f"duplicate {dataset_key} fitting-bank row")
        indices = values
    result: list[_Candidate] = []
    if scan_stats is not None:
        scan_stats.update(
            {
                "dataset_key": dataset_key,
                "rows_partition_checked": len(indices),
                "rows_dataset_accessed": 0,
                "rows_excluded_by_metadata": 0,
                "rows_excluded_by_hash": 0,
                "rows_rejected_short": 0,
                "rows_eligible": 0,
            }
        )
    for position, row_index in enumerate(indices):
        candidate = _candidate(
            dataset_key,
            dataset,
            row_index,
            tokenizer,
            deadline=deadline,
            excluded_ids=excluded_ids,
            excluded_row_keys=excluded_row_keys,
            excluded_hashes=excluded_hashes,
            scan_stats=scan_stats,
        )
        if candidate is not None:
            result.append(candidate)
        if position % 32 == 0:
            deadline.check(f"{dataset_key} fit/frequency scan")
    if not result:
        raise PreparationError(f"no eligible public {dataset_key} fit/frequency rows")
    return result


def _public_frequency(candidates: Sequence[_Candidate], *, deadline: _Deadline) -> Counter[int]:
    counts: Counter[int] = Counter()
    for index, candidate in enumerate(candidates):
        # The slice selects post-BOS positions. Do not filter by token value:
        # an ordinary public record may contain a special ID in context, and
        # selection itself applies the declared special-ID exclusion.
        for token_id in candidate.token_ids[1:]:
            counts[int(token_id)] += 1
        if index % 64 == 0:
            deadline.check("public frequency counting")
    return counts


def _candidate_pool_summary(
    candidates: Sequence[_Candidate],
    scan_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize public fit-pool access without retaining source text."""

    token_ids = [int(token_id) for candidate in candidates for token_id in candidate.token_ids[1:]]
    source_ids = {candidate.record_id for candidate in candidates}
    return {
        "dataset_key": str(scan_stats.get("dataset_key", "")),
        "partition_role": "fit_frequency_only",
        "rows_partition_checked": int(scan_stats.get("rows_partition_checked", 0)),
        "rows_dataset_accessed": int(scan_stats.get("rows_dataset_accessed", 0)),
        "rows_eligible": int(scan_stats.get("rows_eligible", 0)),
        "rows_excluded_by_metadata": int(scan_stats.get("rows_excluded_by_metadata", 0)),
        "rows_excluded_by_hash": int(scan_stats.get("rows_excluded_by_hash", 0)),
        "rows_rejected_short": int(scan_stats.get("rows_rejected_short", 0)),
        "unique_eligible_source_ids": len(source_ids),
        "post_bos_token_occurrences_in_eligible_rows": len(token_ids),
        "unique_post_bos_token_ids_in_eligible_rows": len(set(token_ids)),
        "bos_or_padding_value_filter_applied": False,
    }


def _legacy_frequency(
    path: Path,
    *,
    deadline: _Deadline,
    report: dict[str, Any] | None = None,
) -> Counter[int]:
    """Read public TRR-0004 labels through the safetensors tensor boundary."""

    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"legacy public fit-label artifact must be a regular file: {path}")
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if "token_ids" not in keys:
                raise PreparationError("legacy fit-label artifact has no token_ids tensor")
            if "attention_mask" not in keys:
                raise PreparationError(
                    "legacy fit-label artifact has no attention_mask; valid post-BOS positions cannot be established"
                )
            token_ids = handle.get_tensor("token_ids")
            attention_mask = handle.get_tensor("attention_mask")
            if token_ids.ndim != 2 or attention_mask.ndim != 2:
                raise PreparationError("legacy fit-label and attention-mask tensors must be rank two")
            if tuple(token_ids.shape) != tuple(attention_mask.shape):
                raise PreparationError("legacy token_ids and attention_mask shapes differ")
            if token_ids.shape[0] != 1200 or token_ids.shape[1] != MAX_SEQUENCE_LENGTH:
                raise PreparationError("legacy fit-label tensor geometry changed")
            if not bool(attention_mask[:, 0].gt(0).all().item()):
                raise PreparationError("legacy attention mask has an invalid BOS column")
            if not bool(token_ids[:, 0].eq(BOS_TOKEN_ID).all().item()):
                raise PreparationError("legacy fit labels do not have the declared BOS column")
            valid_post_bos = attention_mask[:, 1:].gt(0)
            valid_count = int(valid_post_bos.sum().item())
            if valid_count != POST_BOS_POSITION_COUNT:
                raise PreparationError(
                    f"legacy attention mask exposes {valid_count} valid post-BOS positions; "
                    f"expected {POST_BOS_POSITION_COUNT}"
                )
            values = token_ids[:, 1:][valid_post_bos].reshape(-1).tolist()
            including_bos_mask = attention_mask.gt(0)
            including_bos_values = token_ids[including_bos_mask].reshape(-1).tolist()
            if report is not None:
                report.update(
                    {
                        "tensor_keys": sorted(keys),
                        "token_ids_key": "token_ids",
                        "attention_mask_key": "attention_mask",
                        "valid_post_bos_positions": valid_count,
                        "valid_positions_including_bos": int(including_bos_mask.sum().item()),
                        "distinct_post_bos_labels": len(set(int(value) for value in values)),
                        "distinct_labels_including_bos": len(
                            set(int(value) for value in including_bos_values)
                        ),
                        "bos_excluded_by_position_mask": True,
                        "values_filtered_by_token_id": False,
                    }
                )
    except PreparationError:
        raise
    except Exception as exc:  # pragma: no cover - dependency-specific details
        raise PreparationError("cannot read the public legacy fit-label artifact") from exc
    deadline.check("legacy frequency counting")
    return Counter(int(value) for value in values)


def _load_exclusion_info(paths: Sequence[Path]) -> tuple[set[str], set[tuple[str, int]], set[str]]:
    """Read only caller-named public metadata manifests for source exclusions."""

    ids: set[str] = set()
    row_keys: set[tuple[str, int]] = set()
    hashes: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            record_id = value.get("record_id")
            if isinstance(record_id, str) and record_id:
                ids.add(record_id)
                lowered = record_id.lower()
                if "pile" in lowered:
                    dataset_key = "pile"
                elif "finance" in lowered:
                    dataset_key = "finance"
                elif "alpaca" in lowered:
                    dataset_key = "alpaca"
                else:
                    dataset_key = ""
                import re

                match = re.search(r"(?:row-|pile10k-|finance-public-)(\d{1,7})", lowered)
                if dataset_key and match:
                    row_keys.add((dataset_key, int(match.group(1))))
            for key in ("text_sha256", "rendered_sha256", "content_sha256", "normalized_content_sha256"):
                digest = value.get(key)
                if isinstance(digest, str) and len(digest) >= 16:
                    hashes.add(digest)
            for key in ("row_index", "dataset_index", "raw_index", "source_index"):
                raw = value.get(key)
                if isinstance(raw, int) and raw >= 0:
                    dataset_key = str(value.get("dataset_key", value.get("dataset", ""))).lower()
                    if "pile" in dataset_key:
                        row_keys.add(("pile", int(raw)))
                    elif "finance" in dataset_key:
                        row_keys.add(("finance", int(raw)))
                    elif "alpaca" in dataset_key:
                        row_keys.add(("alpaca", int(raw)))
            for key, child in value.items():
                if key in {"source_text", "text", "token_ids", "truth", "labels"}:
                    continue
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        value = _load_json(path, label="public exclusion metadata")
        visit(value)
    return ids, row_keys, hashes


def _slot_allocation(slots: Sequence[Any], *, seed: int) -> dict[str, list[int]]:
    """Assign the frozen target slots to the five enriched source groups."""

    names = tuple(ENRICHED_COUNTS)
    order = deterministic_row_order(range(len(slots)), dataset_key="target-slots", seed=seed)
    result = {name: [] for name in names}
    # Controlled records need enough ordinary post-BOS positions for all 30
    # replacements. Prefer slots >=128, then use >=64 if the frozen
    # TRR-0004 length multiset does not contain 120 slots at 128 or longer.
    # This changes only domain-by-length composition; the global length vector
    # and ordered target slots remain untouched and are recorded in the plan.
    controlled_names = ("controlled_pile_context", "controlled_finance_context")
    controlled_count = sum(int(ENRICHED_COUNTS[name]) for name in controlled_names)
    controlled_slots: list[int] = []
    for minimum_length in (128, 64):
        eligible = [
            index
            for index in order
            if int(slots[index].post_bos_token_count) >= minimum_length
        ]
        if len(eligible) >= controlled_count:
            controlled_slots = eligible[:controlled_count]
            break
    if len(controlled_slots) != controlled_count:
        raise PreparationError(
            f"frozen target length vector has fewer than {controlled_count} slots >=64 post-BOS"
        )
    controlled_cursor = 0
    for name in controlled_names:
        count = int(ENRICHED_COUNTS[name])
        result[name].extend(controlled_slots[controlled_cursor : controlled_cursor + count])
        controlled_cursor += count
    controlled_set = set(controlled_slots)
    remaining_order = [index for index in order if index not in controlled_set]
    cursor = 0
    for name in ("alpaca_instruction", "pile_natural", "finance_instruction"):
        count = int(ENRICHED_COUNTS[name])
        result[name].extend(remaining_order[cursor : cursor + count])
        cursor += count
    if controlled_cursor != controlled_count:
        raise PreparationError("controlled source counts do not sum to the target slots")
    if cursor != len(remaining_order):
        raise PreparationError("natural source counts do not sum to the target slots")
    return result


def _assign_candidates(
    candidates: Sequence[_Candidate],
    slot_indices: Sequence[int],
    slots: Sequence[Any],
    *,
    seed: int,
) -> list[tuple[_Candidate, int]]:
    """Match public rows to target lengths without changing the target multiset."""

    if len(candidates) < len(slot_indices):
        raise PreparationError(
            f"only {len(candidates)} public candidates available for {len(slot_indices)} target slots"
        )
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.full_token_count,
            _candidate_order_key(candidate, seed=seed),
        ),
    )
    ordered_slots = sorted(
        (int(index) for index in slot_indices),
        key=lambda index: (-int(slots[index].post_bos_token_count), index),
    )
    remaining = list(ordered_candidates)
    result: list[tuple[_Candidate, int]] = []
    for slot_index in ordered_slots:
        required = int(slots[slot_index].post_bos_token_count) + 1
        chosen_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if candidate.full_token_count >= required
            ),
            None,
        )
        if chosen_index is None:
            raise PreparationError(
                f"public source rows cannot fill target slot {slot_index} ({required} full tokens)"
            )
        result.append((remaining.pop(chosen_index), slot_index))
    return result


def _candidate_order_key(candidate: _Candidate, *, seed: int) -> str:
    return hashlib.sha256(
        f"TRR-0005|{seed}|{candidate.dataset_key}|{candidate.row_index}".encode("utf-8")
    ).hexdigest()


def _original_records(slots: Sequence[Any]) -> list[PlannedRecord]:
    records = [
        PlannedRecord(
            slot=int(slot.slot),
            record_id=str(slot.legacy_record_id),
            source_record_id=str(slot.legacy_record_id),
            dataset_key="alpaca",
            domain="alpaca_instruction",
            target_post_bos_token_count=int(slot.post_bos_token_count),
            source_full_token_count=int(slot.full_token_count),
            rendered_sha256="BOUND_TO_TRR4_PUBLIC_FIT_ARTIFACT",
        )
        for slot in slots
    ]
    validate_planned_records(
        records,
        slots,
        expected_domain_counts={"alpaca_instruction": len(slots)},
    )
    return records


def _enriched_records(
    slots: Sequence[Any],
    allocation: Mapping[str, Sequence[int]],
    candidates_by_group: Mapping[str, Sequence[tuple[_Candidate, int]]],
    selected_controlled_ids: Sequence[int],
    *,
    special_token_ids: set[int],
) -> tuple[list[PlannedRecord], list[tuple[int, ...]], dict[str, Any]]:
    records: list[PlannedRecord | None] = [None] * len(slots)
    sequences: list[tuple[int, ...] | None] = [None] * len(slots)
    controlled_cursor = 0
    controlled_absent: set[int] = set()
    for group, assigned in candidates_by_group.items():
        synthetic = group.startswith("controlled_")
        for ordinal, (candidate, slot_index) in enumerate(assigned):
            target_length = int(slots[slot_index].post_bos_token_count)
            source = PublicSourceRow(
                dataset_key=candidate.dataset_key,
                dataset_id=candidate.dataset_id,
                split=candidate.split,
                revision=candidate.revision,
                row_index=candidate.row_index,
                record_id=candidate.record_id,
                domain=candidate.domain,
                rendered_sha256=candidate.rendered_sha256,
                full_token_count=candidate.full_token_count,
                token_ids=candidate.token_ids,
            )
            validate_source_row(source, target_post_bos_token_count=target_length)
            source_ids = tuple(candidate.token_ids[: target_length + 1])
            if not synthetic:
                records[slot_index] = PlannedRecord(
                    slot=slot_index,
                    record_id=candidate.record_id,
                    source_record_id=candidate.record_id,
                    dataset_key=candidate.dataset_key,
                    domain=candidate.domain,
                    target_post_bos_token_count=target_length,
                    source_full_token_count=candidate.full_token_count,
                    rendered_sha256=candidate.rendered_sha256,
                )
                sequences[slot_index] = source_ids
                continue
            positions = replacement_positions(
                source_ids,
                target_post_bos_token_count=target_length,
                count=CONTROLLED_REPLACEMENTS_PER_RECORD,
                structural_token_ids=special_token_ids,
            )
            replacement_ids = tuple(
                int(selected_controlled_ids[(controlled_cursor + index) % len(selected_controlled_ids)])
                for index in range(len(positions))
            )
            controlled_cursor += len(positions)
            controlled_absent.update(replacement_ids)
            constructed = apply_replacements(
                source_ids,
                positions,
                replacement_ids,
                target_post_bos_token_count=target_length,
                structural_token_ids=special_token_ids,
            )
            synthetic_id = controlled_record_id(candidate.record_id, ordinal)
            records[slot_index] = PlannedRecord(
                slot=slot_index,
                record_id=synthetic_id,
                source_record_id=candidate.record_id,
                dataset_key=candidate.dataset_key,
                domain=group,
                target_post_bos_token_count=target_length,
                source_full_token_count=candidate.full_token_count,
                rendered_sha256=candidate.rendered_sha256,
                synthetic=True,
                replacement_positions=tuple(positions),
                replacement_token_ids=replacement_ids,
            )
            sequences[slot_index] = constructed
    if any(record is None for record in records) or any(sequence is None for sequence in sequences):
        raise PreparationError("enriched source assignment left an unfilled target slot")
    resolved_records = [record for record in records if record is not None]
    resolved_sequences = [sequence for sequence in sequences if sequence is not None]
    controlled_lengths = [
        int(record.target_post_bos_token_count)
        for record in resolved_records
        if record.synthetic
    ]
    validate_planned_records(
        resolved_records,
        slots,
        expected_domain_counts=ENRICHED_COUNTS,
    )
    if len(resolved_sequences) != len(slots):
        raise PreparationError("constructed sequence count changed")
    return resolved_records, resolved_sequences, {
        "controlled_record_count": len(controlled_lengths),
        "controlled_replacement_occurrences": controlled_cursor,
        "controlled_ids_used": len(controlled_absent),
        "controlled_slot_minimum_post_bos": min(controlled_lengths) if controlled_lengths else None,
        "controlled_slots_all_at_least_128": all(length >= 128 for length in controlled_lengths),
        "controlled_slots_all_at_least_64": all(length >= 64 for length in controlled_lengths),
    }


def _pad_sequences(
    sequences: Sequence[Sequence[int]], *,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[list[list[int]], list[list[int]]]:
    token_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for sequence in sequences:
        values = [int(value) for value in sequence]
        if not values or values[0] != BOS_TOKEN_ID:
            raise PreparationError("constructed sequence does not begin with BOS")
        if len(values) > max_length:
            raise PreparationError(
                f"constructed sequence has {len(values)} tokens; maximum is {max_length}"
            )
        if len(values) <= 1:
            raise PreparationError("constructed sequence has no post-BOS positions")
        token_rows.append(values + [PAD_TOKEN_ID] * (max_length - len(values)))
        mask_rows.append([1] * len(values) + [0] * (max_length - len(values)))
    return token_rows, mask_rows


def _save_token_artifact(path: Path, token_rows: Sequence[Sequence[int]], mask_rows: Sequence[Sequence[int]]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PreparationError(f"refusing to overwrite token artifact: {path}")
    try:
        import torch
        from safetensors.torch import save_file

        tokens = torch.tensor(token_rows, dtype=torch.int32)
        mask = torch.tensor(mask_rows, dtype=torch.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"token_ids": tokens, "attention_mask": mask}, str(path))
    except Exception as exc:  # pragma: no cover - dependency-specific details
        raise PreparationError("cannot serialize the public constructed token artifact") from exc
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "shape": [len(token_rows), len(token_rows[0]) if token_rows else 0],
        "token_key": "token_ids",
        "mask_key": "attention_mask",
        "dtype": "torch.int32/torch.uint8",
        "content_hash": "DEFERRED_NO_GBHASH_IN_METADATA_PHASE",
    }


def _runtime_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": int(__import__("os").cpu_count() or 0),
        "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "model_loaded": False,
        "target_weights_accessed": False,
        "private_truth_accessed": False,
        "network_used": False,
    }


def _resource_usage() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        "max_rss_bytes": int(usage.ru_maxrss * 1024),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="read and tokenize the named public caches")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--trr4-lengths", type=Path, default=DEFAULT_LENGTH_METADATA)
    parser.add_argument("--alpaca-arrow", type=Path)
    parser.add_argument("--pile-arrow", type=Path)
    parser.add_argument("--finance-arrow", type=Path, action="append")
    parser.add_argument("--tokenizer", type=Path, required=False)
    parser.add_argument("--legacy-fit-labels", type=Path, required=False)
    parser.add_argument("--exclude-records", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=Path("experiments/TRR-0005/corpus"))
    parser.add_argument("--seed", type=int, default=5005)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    design_path = args.design.expanduser().resolve()
    design = _load_design(design_path)
    if args.max_seconds <= 0 or args.max_seconds > PREPARATION_MAX_SECONDS:
        raise SystemExit(f"--max-seconds must be in (0, {PREPARATION_MAX_SECONDS:g}]")
    if not args.execute:
        geometry = design["geometry"]
        print(
            json.dumps(
                {
                    "status": "DESIGN_ONLY_NO_CACHE_READ",
                    "task_id": "TRR-0005",
                    "arm_count": len(ARMS),
                    "record_count": geometry["record_count"],
                    "post_bos_positions": geometry["post_bos_positions"],
                    "execute_required": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.tokenizer is None or args.legacy_fit_labels is None:
        raise SystemExit("--execute requires --tokenizer and --legacy-fit-labels")

    deadline = _Deadline(time.monotonic(), float(args.max_seconds))
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    coverage_receipt: dict[str, Any] | None = None
    legacy_frequency_details: dict[str, Any] = {}
    phase_timings: dict[str, float] = {}
    source_scan_stats: dict[str, dict[str, Any]] = {}

    def finish_phase(name: str, started: float) -> None:
        phase_timings[name] = time.monotonic() - started

    try:
        deadline.check("startup")
        phase_start = time.monotonic()
        slots = load_trr4_length_slots(args.trr4_lengths)
        finish_phase("trr4_length_metadata", phase_start)
        phase_start = time.monotonic()
        source_paths = _resolve_source_paths(design, args)
        exclusion_ids, exclusion_rows, exclusion_hashes = _load_exclusion_info(args.exclude_records)
        tokenizer = _load_tokenizer(args.tokenizer)
        special_ids = _special_token_ids(tokenizer)
        finish_phase("public_tokenizer_and_exclusions", phase_start)
        deadline.check("public resource setup")

        phase_start = time.monotonic()
        alpaca = _load_public_dataset(source_paths["alpaca"], label="Alpaca")
        pile = _load_public_dataset(source_paths["pile"], label="Pile")
        finance = _load_public_dataset(source_paths["finance"], label="Finance")
        finish_phase("public_arrow_mapping", phase_start)
        deadline.check("public Arrow mapping")

        # The only Pile/Finance rows inspected below are the declared fit and
        # frequency ranges.  Holdout reserve ranges are never materialized here.
        pile_indices = range(
            int(SOURCE_PARTITIONS["pile"]["fit_frequency_start"]),
            int(SOURCE_PARTITIONS["pile"]["fit_frequency_stop"]),
        )
        finance_indices = range(
            int(SOURCE_PARTITIONS["finance"]["fit_frequency_start"]),
            int(SOURCE_PARTITIONS["finance"]["fit_frequency_stop"]),
        )
        phase_start = time.monotonic()
        pile_candidates = _scan_fit_candidates(
            "pile",
            pile,
            pile_indices,
            tokenizer,
            deadline=deadline,
            excluded_ids=exclusion_ids,
            excluded_row_keys=exclusion_rows,
            excluded_hashes=exclusion_hashes,
            scan_stats=source_scan_stats.setdefault("pile", {}),
        )
        finish_phase("pile_fit_frequency_tokenization", phase_start)
        phase_start = time.monotonic()
        finance_candidates = _scan_fit_candidates(
            "finance",
            finance,
            finance_indices,
            tokenizer,
            deadline=deadline,
            excluded_ids=exclusion_ids,
            excluded_row_keys=exclusion_rows,
            excluded_hashes=exclusion_hashes,
            scan_stats=source_scan_stats.setdefault("finance", {}),
        )
        finish_phase("finance_fit_frequency_tokenization", phase_start)
        deadline.check("Pile/Finance fit-frequency scans")

        # Alpaca candidates come only from the registered TRR-0004 fitting bank
        # for the primary enriched allocation.  We do not scan the future
        # footing reserve or use its contents to choose controlled IDs.
        trr4_length_slots = slots
        alpaca_indices = tuple(int(slot.legacy_row_index) for slot in trr4_length_slots)
        phase_start = time.monotonic()
        alpaca_candidates = _scan_fit_candidates(
            "alpaca",
            alpaca,
            alpaca_indices,
            tokenizer,
            deadline=deadline,
            excluded_ids=set(),
            excluded_row_keys=set(),
            excluded_hashes=set(),
            scan_stats=source_scan_stats.setdefault("alpaca", {}),
        )
        finish_phase("alpaca_fit_bank_tokenization", phase_start)
        if len(alpaca_candidates) < int(ENRICHED_COUNTS["alpaca_instruction"]):
            raise PreparationError("TRR-0004 Alpaca fitting bank has fewer than 600 usable rows")

        phase_start = time.monotonic()
        legacy_frequency = _legacy_frequency(
            args.legacy_fit_labels,
            deadline=deadline,
            report=legacy_frequency_details,
        )
        finish_phase("legacy_masked_frequency_load", phase_start)
        phase_start = time.monotonic()
        public_frequency = _public_frequency((*pile_candidates, *finance_candidates), deadline=deadline)
        finish_phase("public_fit_frequency_counting", phase_start)
        source_scan_stats["pile"] = _candidate_pool_summary(pile_candidates, source_scan_stats["pile"])
        source_scan_stats["finance"] = _candidate_pool_summary(finance_candidates, source_scan_stats["finance"])
        source_scan_stats["alpaca"] = _candidate_pool_summary(alpaca_candidates, source_scan_stats["alpaca"])
        selected_controlled_ids = select_public_token_ids(
            legacy_frequency,
            public_frequency,
            special_token_ids=special_ids,
        )
        deadline.check("controlled public token selection")

        phase_start = time.monotonic()
        allocation = _slot_allocation(slots, seed=args.seed)
        grouped_candidates: dict[str, list[tuple[_Candidate, int]]] = {}
        pile_assigned = _assign_candidates(
            pile_candidates,
            [*allocation["pile_natural"], *allocation["controlled_pile_context"]],
            slots,
            seed=args.seed,
        )
        finance_assigned = _assign_candidates(
            finance_candidates,
            [*allocation["finance_instruction"], *allocation["controlled_finance_context"]],
            slots,
            seed=args.seed,
        )
        alpaca_assigned = _assign_candidates(
            alpaca_candidates,
            allocation["alpaca_instruction"],
            slots,
            seed=args.seed,
        )
        # Partition each domain's assignments according to the slot group's
        # declared quotas, retaining the content-blind slot allocation.
        pile_controlled = set(allocation["controlled_pile_context"])
        finance_controlled = set(allocation["controlled_finance_context"])
        grouped_candidates["pile_natural"] = [item for item in pile_assigned if item[1] not in pile_controlled]
        grouped_candidates["controlled_pile_context"] = [item for item in pile_assigned if item[1] in pile_controlled]
        grouped_candidates["finance_instruction"] = [item for item in finance_assigned if item[1] not in finance_controlled]
        grouped_candidates["controlled_finance_context"] = [item for item in finance_assigned if item[1] in finance_controlled]
        grouped_candidates["alpaca_instruction"] = alpaca_assigned

        original_records = _original_records(slots)
        enriched_records, enriched_sequences, controlled_meta = _enriched_records(
            slots,
            allocation,
            grouped_candidates,
            selected_controlled_ids,
            special_token_ids=special_ids,
        )
        enriched_flat_ids = [token_id for sequence in enriched_sequences for token_id in sequence[1:]]
        enriched_summary = token_frequency_summary(
            enriched_flat_ids,
            exclude_special_values=False,
        )
        legacy_summary = token_frequency_summary(
            (token_id for token_id, count in legacy_frequency.items() for _ in range(int(count))),
            exclude_special_values=False,
        )
        coverage_receipt = coverage_contrast(
            legacy_summary,
            enriched_summary,
            selected_controlled_ids=selected_controlled_ids,
            legacy_frequency=legacy_frequency,
            minimum_distinct=MIN_ENRICHED_DISTINCT_TOKEN_IDS,
            minimum_legacy_absent=MIN_LEGACY_ABSENT_CONTROLLED_IDS,
        )
        if coverage_receipt["status"] != "PASS":
            raise PreparationError("coverage_mix_v1 failed the preregistered contrast thresholds")
        deadline.check("coverage contrast validation")
        finish_phase("slot_assignment_and_coverage_validation", phase_start)

        phase_start = time.monotonic()
        token_rows, mask_rows = _pad_sequences(enriched_sequences)
        token_artifact = _save_token_artifact(
            output_root / "coverage_mix_v1" / "constructed_public_tokens.safetensors",
            token_rows,
            mask_rows,
        )
        deadline.check("token-plan serialization")
        finish_phase("plan_and_token_artifact_serialization", phase_start)

        plan = {
            "schema": PLAN_SCHEMA,
            "task_id": "TRR-0005",
            "status": "PREPARED_PUBLIC_DATA_NO_MODEL_FORWARD",
            "generated_at_utc": _utc_now(),
            "execution": {
                "git_commit": _git_commit(root),
                "argv": [str(value) for value in (sys.argv if argv is None else [str(__file__), *argv])],
                "max_seconds": float(args.max_seconds),
                "elapsed_seconds": deadline.elapsed(),
                "runtime": _runtime_snapshot(),
                "resource_usage": _resource_usage(),
                "phase_timings_seconds": dict(phase_timings),
            },
            "preparation": {
                "status": "bounded_public_metadata_and_tokenization_complete",
                "wall_seconds": deadline.elapsed(),
                "max_seconds": float(args.max_seconds),
                "phase_timings_seconds": dict(phase_timings),
                "source_pools": dict(source_scan_stats),
                "legacy_frequency_binding": dict(legacy_frequency_details),
                "cache_access_policy": "named public Arrow/tokenizer/legacy-label paths opened read-only; no holdout rows inspected",
                "model_loaded": False,
                "public_model_forward_count": 0,
                "private_truth_accessed": False,
                "network_used": False,
                "resource_usage": _resource_usage(),
            },
            "design": {
                "schema": CORPUS_SCHEMA,
                "length_vector_digest": length_vector_digest(slots),
                "length_multiset": {str(key): int(value) for key, value in sorted(length_multiset(slots).items())},
                "record_count": len(slots),
                "stored_rows_including_bos": STORED_ROW_COUNT,
                "post_bos_positions": POST_BOS_POSITION_COUNT,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
            },
            "partitions": {
                "pile_fit_frequency": [
                    int(SOURCE_PARTITIONS["pile"]["fit_frequency_start"]),
                    int(SOURCE_PARTITIONS["pile"]["fit_frequency_stop"]),
                ],
                "pile_holdout_reserve": [
                    int(SOURCE_PARTITIONS["pile"]["holdout_reserve_start"]),
                    int(SOURCE_PARTITIONS["pile"]["holdout_reserve_stop"]),
                ],
                "finance_fit_frequency": [
                    int(SOURCE_PARTITIONS["finance"]["fit_frequency_start"]),
                    int(SOURCE_PARTITIONS["finance"]["fit_frequency_stop"]),
                ],
                "finance_holdout_reserve": [
                    int(SOURCE_PARTITIONS["finance"]["holdout_reserve_start"]),
                    int(SOURCE_PARTITIONS["finance"]["holdout_reserve_stop"]),
                ],
                "holdout_rows_inspected": False,
            },
            "public_inputs": {
                "sources": {
                    "alpaca": [public_source_file_record(path, role="public Alpaca cache") for path in source_paths["alpaca"]],
                    "pile": [public_source_file_record(path, role="public Pile cache") for path in source_paths["pile"]],
                    "finance": [public_source_file_record(path, role="public Finance cache") for path in source_paths["finance"]],
                },
                "tokenizer": public_source_file_record(args.tokenizer, role="public tokenizer directory"),
                "legacy_fit_labels": public_source_file_record(args.legacy_fit_labels, role="public TRR-0004 fit labels"),
                "exclusion_manifests": [str(Path(path).expanduser().resolve()) for path in args.exclude_records],
            },
            "arms": {
                ORIGINAL_ARM: {
                    "records": [record.as_dict() for record in original_records],
                    "reused_artifact": "TRR-0004 public activation fit artifact; no source model forward in this command",
                    "coverage": legacy_summary,
                    "coverage_reference_binding": legacy_frequency_details,
                    "domain_length": domain_length_summary(original_records),
                },
                ENRICHED_ARM: {
                    "records": [record.as_dict() for record in enriched_records],
                    "coverage": enriched_summary,
                    "coverage_contrast": coverage_receipt,
                    "domain_length": domain_length_summary(enriched_records),
                    "controlled": controlled_meta,
                    "token_artifact": token_artifact,
                    "public_forward_required": True,
                    "public_forward_rule": "run P0/public-prefix on every complete constructed sequence; never splice H rows",
                },
            },
            "controlled_token_selection": {
                "selected_token_id_count": len(selected_controlled_ids),
                "selected_token_ids": [int(value) for value in selected_controlled_ids],
                "legacy_frequency_maximum": 1,
                "selected_from": "Pile/Finance fit-frequency rows only plus public legacy fit labels",
                "special_token_ids_excluded": sorted(special_ids),
                "legacy_frequency_binding": legacy_frequency_details,
            },
            "joint_training_exposure": expected_sampler_exposure(
                post_bos_positions=POST_BOS_POSITION_COUNT,
                batch_size=FIT_BATCH_SIZE,
                steps=FIT_STEPS,
            ),
        }
        _write_json(output_root / "corpus_plan.json", plan)
        print(
            json.dumps(
                {
                    "status": plan["status"],
                    "output": str(output_root / "corpus_plan.json"),
                    "coverage": coverage_receipt,
                    "elapsed_seconds": deadline.elapsed(),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema": "token-reconstruction.trr0005-public-corpus-failure.v1",
            "task_id": "TRR-0005",
            "status": "FAILED_PUBLIC_PREPARATION",
            "generated_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": deadline.elapsed(),
            "runtime": _runtime_snapshot(),
            "coverage_contrast": coverage_receipt,
            "legacy_frequency_binding": legacy_frequency_details,
            "preserve_policy": "failed contrast receipt is retained; thresholds and source partitions are not silently downgraded",
        }
        failure_path = output_root / "failure.json"
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_json(failure_path, failure)
        raise SystemExit(f"TRR-0005 public preparation failed; preserved {failure_path}: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
