#!/usr/bin/env python3
"""Materialize the frozen TRR-0007 broader public fitting bank.

This command is the data-only handoff between the approved support recipe and
the registered TRR-0005 public-prefix producer.  It retains every natural row
and target length from the current enriched bank, chooses 60 new Pile and 60
new Finance parents from the declared fit/frequency partitions, and constructs
the 120 complete synthetic sequences with 3,600 distinct replacement IDs.

The command never loads model weights or activation tensors.  It does not scan
the reserved public holdout ranges and does not open evaluator truth.  The
producer must run a real P0/public-prefix forward over the resulting complete
token artifact before any fitting or comparison uses it.
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
from safetensors.torch import save_file

from token_reconstruction.trr0005_public_corpus import (
    BOS_TOKEN_ID,
    MAX_SEQUENCE_LENGTH,
    PAD_TOKEN_ID,
    POST_BOS_POSITION_COUNT,
    SOURCE_PARTITIONS,
    apply_replacements,
    expected_sampler_exposure,
    token_frequency_summary,
)

from trr0005_prepare_public_corpus import (  # type: ignore
    PreparationError,
    _Candidate,
    _Deadline,
    _load_public_dataset,
    _load_tokenizer,
    _scan_fit_candidates,
)
from trr0007_support_candidate_pool import (  # type: ignore
    _public_exclusions,
)
from trr0007_support_diagnostics import (  # type: ignore
    _planned_replacement_positions,
)


TASK_ID = "TRR-0007"
PLAN_SCHEMA = "token-reconstruction.trr0005-public-corpus-plan.v1"
SUPPORT_SCHEMA = "token-reconstruction.trr0007-public-broader-bank.v1"
FIT_RECORDS = 1200
CONTROLLED_RECORDS_PER_DOMAIN = 60
CONTROLLED_RECORDS = 120
REPLACEMENTS_PER_RECORD = 30
REPLACEMENTS_TOTAL = CONTROLLED_RECORDS * REPLACEMENTS_PER_RECORD
BASELINE_IDS = 2000
PROPOSED_IDS = 3600
FIT_BATCH_SIZE = 512
FIT_STEPS = 3000
CONSTRUCTOR_SEED = 7007


class ConstructionError(RuntimeError):
    """Raised when the frozen broader-bank contract cannot be materialized."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_ints(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{int(value)}\n".encode("utf-8"))
    return digest.hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_rows(rows: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(",".join(str(int(value)) for value in row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ConstructionError(f"{label} must be a regular file: {path}")
    return path


def _json(path: Path, *, label: str) -> Mapping[str, Any]:
    path = _regular(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstructionError(f"cannot parse {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ConstructionError(f"{label} must contain an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ConstructionError(f"output is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


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


def _git_dirty(root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _source_descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ConstructionError(f"source must be a regular file: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _tensor(path: Path, key: str, *, label: str) -> torch.Tensor:
    path = _regular(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in set(handle.keys()):
                raise ConstructionError(f"{label} has no {key!r} tensor")
            return handle.get_tensor(key).contiguous()
    except ConstructionError:
        raise
    except Exception as exc:
        raise ConstructionError(f"cannot load {label}: {path}") from exc


def _load_token_batch(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, ...]]]:
    tokens = _tensor(path, "token_ids", label="current enriched token artifact")
    mask = _tensor(path, "attention_mask", label="current enriched attention mask")
    if tuple(tokens.shape) != (FIT_RECORDS, MAX_SEQUENCE_LENGTH):
        raise ConstructionError("current enriched token artifact geometry changed")
    if tuple(mask.shape) != tuple(tokens.shape) or mask.dtype != torch.uint8:
        raise ConstructionError("current enriched token/mask geometry or dtype changed")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ConstructionError("current enriched token dtype changed")
    if not bool(tokens[:, 0].eq(BOS_TOKEN_ID).all().item()):
        raise ConstructionError("current enriched rows lost BOS")
    sequences: list[tuple[int, ...]] = []
    for row in range(FIT_RECORDS):
        active = int(mask[row].sum().item())
        if active <= 1 or active > MAX_SEQUENCE_LENGTH:
            raise ConstructionError(f"current enriched row {row} has invalid active length")
        if not bool(mask[row, :active].eq(1).all().item()) or not bool(mask[row, active:].eq(0).all().item()):
            raise ConstructionError("current enriched mask is not contiguous right-padding")
        if not bool(tokens[row, active:].eq(PAD_TOKEN_ID).all().item()):
            raise ConstructionError("current enriched padding values changed")
        sequences.append(tuple(int(value) for value in tokens[row, :active].tolist()))
    return tokens, mask, sequences


def _recipe_ids(recipe: Mapping[str, Any]) -> tuple[list[int], list[int], list[int]]:
    controlled = recipe.get("controlled_component")
    if not isinstance(controlled, Mapping):
        raise ConstructionError("approved recipe has no controlled component")
    identity = controlled.get("identity_pool")
    if not isinstance(identity, Mapping):
        raise ConstructionError("approved recipe has no identity pool")
    values = identity.get("selected_token_ids")
    if not isinstance(values, list):
        raise ConstructionError("approved recipe has no selected token IDs")
    selected = [int(value) for value in values]
    if len(selected) != PROPOSED_IDS or len(set(selected)) != PROPOSED_IDS:
        raise ConstructionError("approved recipe selected identity count changed")
    if any(value < 0 or value >= 128256 or value in {BOS_TOKEN_ID, PAD_TOKEN_ID} for value in selected):
        raise ConstructionError("approved recipe includes a structural or out-of-vocabulary ID")
    baseline = selected[:BASELINE_IDS]
    additions = selected[BASELINE_IDS:]
    if len(baseline) != BASELINE_IDS or len(additions) != PROPOSED_IDS - BASELINE_IDS:
        raise ConstructionError("approved recipe baseline/addition identity split changed")
    return selected, baseline, additions


def _recipe_slots(recipe: Mapping[str, Any], plan_rows: Sequence[Mapping[str, Any]]) -> list[int]:
    values = recipe.get("selected_controlled_slot_indices")
    if not isinstance(values, list):
        raise ConstructionError("approved recipe has no controlled-slot list")
    selected = [int(value) for value in values]
    if len(selected) != CONTROLLED_RECORDS or len(set(selected)) != CONTROLLED_RECORDS:
        raise ConstructionError("approved recipe controlled-slot count changed")
    expected = sorted(int(row["slot"]) for row in plan_rows if bool(row.get("synthetic", False)))
    if selected != expected:
        raise ConstructionError("approved recipe no longer binds the current controlled-slot geometry")
    if any(int(plan_rows[index]["target_post_bos_token_count"]) < 128 for index in selected):
        raise ConstructionError("controlled target length no longer covers positions 1..127")
    return selected


def _candidate_order_key(candidate: _Candidate) -> str:
    """Content-blind stable ordering frozen by the broader recipe."""

    payload = (
        f"{TASK_ID}|{CONSTRUCTOR_SEED}|{candidate.dataset_id}|{candidate.split}|"
        f"{candidate.revision}|row:{candidate.row_index}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _choose_parents(
    candidates: Sequence[_Candidate],
    slots: Sequence[int],
    plan_rows: Sequence[Mapping[str, Any]],
    *,
    group: str,
) -> list[tuple[int, _Candidate]]:
    if len(candidates) < len(slots):
        raise ConstructionError(f"not enough eligible {group} candidates for {len(slots)} parents")
    ordered = sorted(candidates, key=lambda candidate: (_candidate_order_key(candidate), candidate.row_index))
    # Match longer target slots first.  For each target, the first source in
    # the stable source order that has enough complete tokens is selected.
    ordered_slots = sorted(
        (int(slot) for slot in slots),
        key=lambda slot: (-int(plan_rows[slot]["target_post_bos_token_count"]), slot),
    )
    remaining = list(ordered)
    selected: list[tuple[int, _Candidate]] = []
    for slot in ordered_slots:
        target = int(plan_rows[slot]["target_post_bos_token_count"])
        required = target + 1
        chosen = next(
            (index for index, candidate in enumerate(remaining) if candidate.full_token_count >= required),
            None,
        )
        if chosen is None:
            raise ConstructionError(
                f"{group} candidate pool cannot fill target slot {slot} requiring {required} full tokens"
            )
        selected.append((slot, remaining.pop(chosen)))
    return selected


def _pad(sequences: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    token_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for index, sequence in enumerate(sequences):
        values = [int(value) for value in sequence]
        if not values or values[0] != BOS_TOKEN_ID:
            raise ConstructionError(f"constructed row {index} does not begin with BOS")
        if len(values) <= 1 or len(values) > MAX_SEQUENCE_LENGTH:
            raise ConstructionError(f"constructed row {index} has invalid length {len(values)}")
        token_rows.append(values + [PAD_TOKEN_ID] * (MAX_SEQUENCE_LENGTH - len(values)))
        mask_rows.append([1] * len(values) + [0] * (MAX_SEQUENCE_LENGTH - len(values)))
    return torch.tensor(token_rows, dtype=torch.int32), torch.tensor(mask_rows, dtype=torch.uint8)


def _record_sequence_hash(sequence: Sequence[int]) -> str:
    return _sha256_ints([int(value) for value in sequence])


def _domain_length_summary_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[int]] = {}
    for row in rows:
        domain = str(row.get("domain", ""))
        if not domain:
            raise ConstructionError("constructed record has no domain")
        by_domain.setdefault(domain, []).append(int(row["target_post_bos_token_count"]))
    return {
        domain: {
            "records": len(values),
            "post_bos_positions": int(sum(values)),
            "min_post_bos_length": min(values) if values else None,
            "max_post_bos_length": max(values) if values else None,
            "mean_post_bos_length": (float(sum(values)) / len(values)) if values else None,
            "length_histogram": dict(sorted(Counter(values).items())),
        }
        for domain, values in sorted(by_domain.items())
    }


def _source_row_metadata(candidate: _Candidate) -> dict[str, Any]:
    return {
        "dataset_key": candidate.dataset_key,
        "dataset_id": candidate.dataset_id,
        "split": candidate.split,
        "revision": candidate.revision,
        "row_index": int(candidate.row_index),
        "source_record_id": candidate.record_id,
        "domain": candidate.domain,
        "rendered_sha256": candidate.rendered_sha256,
        "source_full_token_count": int(candidate.full_token_count),
    }


def _exclusion_entry(row: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": source,
        "record_id": str(row.get("record_id", "")),
        "source_record_id": str(row.get("source_record_id", row.get("record_id", ""))),
    }
    for key in ("dataset_key", "domain", "rendered_sha256", "row_index", "source_index", "dataset_index"):
        if key in row and isinstance(row[key], (str, int)):
            result[key] = row[key]
    return result


def _build_exclusion_manifest(
    *,
    exclusion_paths: Sequence[Path],
    exclusion_binding: Mapping[str, Any],
    exclusion_ids: set[str],
    exclusion_rows: set[tuple[str, int]],
    exclusion_hashes: set[str],
    current_rows: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any],
    candidate_count: Mapping[str, int],
) -> dict[str, Any]:
    current = [_exclusion_entry(row, source="current_trr0005_enriched_fit") for row in current_rows]
    return {
        "schema": "token-reconstruction.trr0007-public-bank-exclusions.v1",
        "task_id": TASK_ID,
        "status": "PUBLIC_PARENT_SELECTION_EXCLUSIONS_BOUND",
        "purpose": "Exact public source and sequence exclusion receipt for the new controlled parents.",
        "metadata": [
            {
                "path": str(path.expanduser().resolve()),
                "bytes": int(path.expanduser().resolve().stat().st_size),
                "sha256": _sha256_file(path.expanduser().resolve()),
            }
            for path in exclusion_paths
        ],
        "exclusion_binding": dict(exclusion_binding),
        "current_fit_records_excluded": current,
        "current_fit_record_count": len(current),
        "selector_exclusion_sets": {
            "record_ids": sorted(str(value) for value in exclusion_ids),
            "source_row_keys": [
                {"dataset_key": str(dataset), "row_index": int(row_index)}
                for dataset, row_index in sorted(exclusion_rows)
            ],
            "opaque_sequence_or_reservation_digests": sorted(str(value) for value in exclusion_hashes),
        },
        "selector_exclusion_set_counts": {
            "record_ids": len(exclusion_ids),
            "source_row_keys": len(exclusion_rows),
            "opaque_sequence_or_reservation_digests": len(exclusion_hashes),
        },
        "opaque_p04_values_loaded_for_exclusion_only": True,
        "opaque_p04_value_count": int(exclusion_binding.get("opaque_digest_count", 0)),
        "candidate_counts_after_exclusions": {str(k): int(v) for k, v in candidate_count.items()},
        "reserved_holdout_ranges_scanned": False,
        "public_development_truth_opened": False,
        "private_truth_accessed": False,
        "source_text_retained": False,
        "recipe_binding": {
            "path": str(recipe.get("_path", "")),
            "sha256": str(recipe.get("_sha256", "")),
            "current_fit_and_public_development_excluded_before_length_matching": True,
        },
    }


def _scan_sources(
    *,
    tokenizer: Any,
    pile_arrow: Path,
    finance_arrows: Sequence[Path],
    exclusion_ids: set[str],
    exclusion_rows: set[tuple[str, int]],
    exclusion_hashes: set[str],
    deadline: _Deadline,
) -> tuple[dict[str, list[_Candidate]], dict[str, dict[str, Any]]]:
    pile = _load_public_dataset([pile_arrow], label="Pile")
    finance = _load_public_dataset(finance_arrows, label="Finance")
    candidates: dict[str, list[_Candidate]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for dataset_key, dataset, paths in (
        ("pile", pile, (pile_arrow,)),
        ("finance", finance, tuple(finance_arrows)),
    ):
        spec = SOURCE_PARTITIONS[dataset_key]
        report: dict[str, Any] = {}
        values = _scan_fit_candidates(
            dataset_key,
            dataset,
            range(int(spec["fit_frequency_start"]), int(spec["fit_frequency_stop"])),
            tokenizer,
            deadline=deadline,
            excluded_ids=exclusion_ids,
            excluded_row_keys=exclusion_rows,
            excluded_hashes=exclusion_hashes,
            scan_stats=report,
        )
        candidates[dataset_key] = values
        stats[dataset_key] = {
            **report,
            "fit_frequency_range": [int(spec["fit_frequency_start"]), int(spec["fit_frequency_stop"])],
            "holdout_range": [int(spec["holdout_reserve_start"]), int(spec["holdout_reserve_stop"])],
            "holdout_rows_scanned": False,
            "eligible_source_ids_sha256": _sha256_strings(
                [str(candidate.record_id) for candidate in sorted(values, key=lambda candidate: candidate.record_id)]
            ),
            "eligible_source_id_count": len(values),
            "source_resources": [_source_descriptor(path) for path in paths],
        }
        deadline.check(f"{dataset_key} parent scan")
    return candidates, stats


def construct(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    started_utc = _utc_now()
    started_clock = time.monotonic()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ConstructionError(f"output root is create-only and already exists: {output_root}")
    if args.max_seconds <= 0 or args.max_seconds > 300:
        raise ConstructionError("--max-seconds must be in (0, 300]")
    plan_path = _regular(args.current_plan, label="current corpus plan")
    current_plan = _json(plan_path, label="current TRR-0005 corpus plan")
    if current_plan.get("schema") != PLAN_SCHEMA or current_plan.get("task_id") != "TRR-0005":
        raise ConstructionError("current plan schema or task ID changed")
    arms = current_plan.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get("coverage_mix_v1"), Mapping):
        raise ConstructionError("current plan has no enriched coverage arm")
    current_rows_raw = arms["coverage_mix_v1"].get("records")
    if not isinstance(current_rows_raw, list) or len(current_rows_raw) != FIT_RECORDS:
        raise ConstructionError("current enriched plan must contain 1,200 records")
    current_rows = [dict(row) for row in current_rows_raw]
    if any(int(row.get("slot", -1)) != index for index, row in enumerate(current_rows)):
        raise ConstructionError("current enriched slots are not ordered 0..1199")
    current_token_path = (
        args.current_tokens.expanduser().resolve()
        if args.current_tokens is not None
        else Path(str(arms["coverage_mix_v1"]["token_artifact"]["path"])).expanduser().resolve()
    )
    current_tokens, current_mask, current_sequences = _load_token_batch(current_token_path)
    natural_slots = [index for index, row in enumerate(current_rows) if not bool(row.get("synthetic", False))]
    current_controlled_slots = [index for index, row in enumerate(current_rows) if bool(row.get("synthetic", False))]
    if len(natural_slots) != FIT_RECORDS - CONTROLLED_RECORDS or len(current_controlled_slots) != CONTROLLED_RECORDS:
        raise ConstructionError("current natural/controlled counts changed")
    recipe_path = _regular(args.recipe, label="approved TRR7 recipe")
    recipe = dict(_json(recipe_path, label="approved TRR7 recipe"))
    if recipe.get("task_id") != TASK_ID or recipe.get("status") != "METADATA_RECIPE_PENDING_REAL_P0_FORWARD":
        raise ConstructionError("approved recipe status or task ID changed")
    recipe["_path"] = str(recipe_path)
    recipe["_sha256"] = _sha256_file(recipe_path)
    selected_ids, baseline_ids, additions = _recipe_ids(recipe)
    controlled_slots = _recipe_slots(recipe, current_rows)
    if sorted(current_controlled_slots) != controlled_slots:
        raise ConstructionError("current plan controlled slots do not match the approved recipe")
    candidate_pool_path = _regular(args.candidate_pool, label="candidate frequency map")
    candidate_pool = _json(candidate_pool_path, label="candidate frequency map")
    if candidate_pool.get("schema") != "token-reconstruction.trr0007-public-fit-candidate-frequency.v1":
        raise ConstructionError("candidate frequency map schema changed")
    candidate_freq_raw = candidate_pool.get("frequency_by_token_id")
    if not isinstance(candidate_freq_raw, Mapping):
        raise ConstructionError("candidate frequency map has no token frequency mapping")
    candidate_freq = {int(key): int(value) for key, value in candidate_freq_raw.items()}
    if any(value <= 0 for value in candidate_freq.values()):
        raise ConstructionError("candidate frequency map contains non-positive counts")
    enriched_coverage = arms["coverage_mix_v1"].get("coverage")
    enriched_freq_raw = enriched_coverage.get("token_frequency_by_id") if isinstance(enriched_coverage, Mapping) else None
    if not isinstance(enriched_freq_raw, Mapping):
        raise ConstructionError("current plan has no enriched frequency reference")
    enriched_freq = {int(key): int(value) for key, value in enriched_freq_raw.items()}
    if any(token_id in enriched_freq for token_id in additions):
        raise ConstructionError("approved additions are not current-unseen in the current enriched reference")
    candidate_current_unseen = {token_id for token_id, count in candidate_freq.items() if count > 0 and token_id not in enriched_freq and token_id not in {BOS_TOKEN_ID, PAD_TOKEN_ID}}
    if not set(additions).issubset(candidate_current_unseen):
        raise ConstructionError("approved additions are absent from the candidate pool or structural")
    deadline = _Deadline(time.monotonic(), float(args.max_seconds))
    exclusion_paths = [path.expanduser().resolve() for path in args.exclude_records]
    if not any("p04" in str(path).lower() for path in exclusion_paths):
        raise ConstructionError("--exclude-records must include the approved opaque P04 exchange")
    exclusion_ids, exclusion_rows, exclusion_hashes, exclusion_binding = _public_exclusions(exclusion_paths)
    tokenizer = _load_tokenizer(args.tokenizer.expanduser().resolve())
    special_ids = {BOS_TOKEN_ID, PAD_TOKEN_ID, *[int(value) for value in getattr(tokenizer, "all_special_ids", ()) or ()]}
    candidates, scan_stats = _scan_sources(
        tokenizer=tokenizer,
        pile_arrow=args.pile_arrow,
        finance_arrows=args.finance_arrow,
        exclusion_ids=exclusion_ids,
        exclusion_rows=exclusion_rows,
        exclusion_hashes=exclusion_hashes,
        deadline=deadline,
    )
    selected_parents: dict[str, list[tuple[int, _Candidate]]] = {}
    selected_parents["controlled_pile_context"] = _choose_parents(
        candidates["pile"],
        [slot for slot in controlled_slots if current_rows[slot].get("domain") == "controlled_pile_context"],
        current_rows,
        group="Pile",
    )
    selected_parents["controlled_finance_context"] = _choose_parents(
        candidates["finance"],
        [slot for slot in controlled_slots if current_rows[slot].get("domain") == "controlled_finance_context"],
        current_rows,
        group="Finance",
    )
    deadline.check("parent selection")
    parent_by_slot = {slot: candidate for group in selected_parents.values() for slot, candidate in group}
    if set(parent_by_slot) != set(controlled_slots) or len(parent_by_slot) != CONTROLLED_RECORDS:
        raise ConstructionError("controlled parent selection did not fill all approved slots")
    sequences: list[tuple[int, ...] | None] = [None] * FIT_RECORDS
    records: list[dict[str, Any] | None] = [None] * FIT_RECORDS
    for slot in natural_slots:
        sequence = current_sequences[slot]
        current = dict(current_rows[slot])
        current["slot"] = slot
        current["synthetic"] = False
        current["target_full_token_count"] = len(sequence)
        current["target_post_bos_token_count"] = len(sequence) - 1
        records[slot] = current
        sequences[slot] = sequence
    # Keep the old TRR5 replacement traversal convention: Pile controlled
    # slots first, then Finance controlled slots, each ordered by target length
    # and slot.  The broader identity list is consumed exactly once.
    replacement_cursor = 0
    parent_rows: list[dict[str, Any]] = []
    for group in ("controlled_pile_context", "controlled_finance_context"):
        assigned = sorted(
            selected_parents[group],
            key=lambda item: (-int(current_rows[item[0]]["target_post_bos_token_count"]), item[0]),
        )
        for ordinal, (slot, candidate) in enumerate(assigned):
            target = int(current_rows[slot]["target_post_bos_token_count"])
            source_ids = tuple(candidate.token_ids[: target + 1])
            if len(source_ids) != target + 1 or source_ids[0] != BOS_TOKEN_ID:
                raise ConstructionError(f"selected {group} parent is shorter than slot {slot}")
            synthetic_id = f"{TASK_ID}/controlled-v2/{candidate.record_id}::row-{ordinal:03d}"
            offsets = _planned_replacement_positions(
                source_ids,
                target_post_bos_token_count=target,
                record_key=synthetic_id,
                seed=CONSTRUCTOR_SEED,
                structural_token_ids=special_ids,
            )
            replacement_ids = tuple(selected_ids[replacement_cursor : replacement_cursor + len(offsets)])
            if len(replacement_ids) != len(offsets):
                raise ConstructionError("broader identity list was consumed before 3,600 occurrences")
            replacement_cursor += len(offsets)
            constructed = apply_replacements(
                source_ids,
                offsets,
                replacement_ids,
                target_post_bos_token_count=target,
                structural_token_ids=special_ids,
            )
            row = {
                "record_id": synthetic_id,
                "source_record_id": candidate.record_id,
                "dataset_key": candidate.dataset_key,
                "domain": group,
                "slot": slot,
                "synthetic": True,
                "source_dataset_id": candidate.dataset_id,
                "source_split": candidate.split,
                "source_revision": candidate.revision,
                "source_row_index": int(candidate.row_index),
                "source_full_token_count": int(candidate.full_token_count),
                "target_full_token_count": target + 1,
                "target_post_bos_token_count": target,
                "rendered_sha256": candidate.rendered_sha256,
                "replacement_count": len(offsets),
                "replacement_positions": [int(value) for value in offsets],
                "replacement_positions_one_based": [int(value) + 1 for value in offsets],
                "replacement_token_ids": [int(value) for value in replacement_ids],
                "constructed_sequence_sha256": _record_sequence_hash(constructed),
            }
            if records[slot] is not None or sequences[slot] is not None:
                raise ConstructionError(f"duplicate record construction at slot {slot}")
            records[slot] = row
            sequences[slot] = constructed
            parent_rows.append(
                {
                    **_source_row_metadata(candidate),
                    "domain": group,
                    "slot": slot,
                    "synthetic_record_id": synthetic_id,
                    "target_post_bos_token_count": target,
                    "target_full_token_count": target + 1,
                    "replacement_offset_sha256": _sha256_ints(list(offsets)),
                    "constructed_sequence_sha256": _record_sequence_hash(constructed),
                }
            )
    if replacement_cursor != REPLACEMENTS_TOTAL:
        raise ConstructionError(f"constructed {replacement_cursor} replacements; expected {REPLACEMENTS_TOTAL}")
    if any(row is None for row in records) or any(sequence is None for sequence in sequences):
        raise ConstructionError("constructed bank has an unfilled slot")
    resolved_records = [row for row in records if row is not None]
    resolved_sequences = [sequence for sequence in sequences if sequence is not None]
    if len({int(value) for row in resolved_records if row.get("synthetic") for value in row["replacement_token_ids"]}) != PROPOSED_IDS:
        raise ConstructionError("broader replacement IDs are not all distinct")
    replacement_flat = [int(value) for row in resolved_records if row.get("synthetic") for value in row["replacement_token_ids"]]
    if set(replacement_flat) != set(selected_ids) or len(replacement_flat) != REPLACEMENTS_TOTAL:
        raise ConstructionError("broader replacement occurrences do not match the frozen identity list")
    token_tensor, mask_tensor = _pad(resolved_sequences)
    if int(mask_tensor[:, 1:].sum().item()) != POST_BOS_POSITION_COUNT:
        raise ConstructionError("broader bank post-BOS opportunities changed")
    current_lengths = current_mask.sum(dim=1).to(dtype=torch.int64)
    if not torch.equal(current_lengths, mask_tensor.sum(dim=1).to(dtype=torch.int64)):
        raise ConstructionError("broader bank length vector changed")
    if not torch.equal(token_tensor[natural_slots], current_tokens[natural_slots].to(dtype=torch.int32)):
        raise ConstructionError("retained natural rows differ from current token artifact")
    if not torch.equal(mask_tensor[natural_slots], current_mask[natural_slots]):
        raise ConstructionError("retained natural masks differ from current token artifact")
    broad_token_rows = token_tensor[:, 1:][mask_tensor[:, 1:].to(torch.bool)].tolist()
    natural_fixture = {
        "status": "PASS",
        "retained_records": len(natural_slots),
        "natural_slot_indices_sha256": _sha256_ints(natural_slots),
        "token_ids_sha256": _sha256_rows(token_tensor[natural_slots].tolist()),
        "attention_mask_sha256": _sha256_rows(mask_tensor[natural_slots].tolist()),
        "current_token_artifact": _source_descriptor(current_token_path),
        "token_ids_equal_current": True,
        "attention_masks_equal_current": True,
        "activation_equality": "must be checked after the real P0 forward; this CPU fixture compares only tokens and masks",
    }
    output_root.mkdir(parents=True, exist_ok=False)
    token_path = output_root / "constructed_public_tokens.safetensors"
    save_file({"token_ids": token_tensor, "attention_mask": mask_tensor}, str(token_path))
    token_entry = {
        "path": str(token_path),
        "bytes": int(token_path.stat().st_size),
        "shape": [FIT_RECORDS, MAX_SEQUENCE_LENGTH],
        "token_key": "token_ids",
        "mask_key": "attention_mask",
        "dtype": "torch.int32/torch.uint8",
        "content_hash": _sha256_file(token_path),
    }
    current_arms = dict(arms)
    current_enriched = dict(current_arms["coverage_mix_v1"])
    current_enriched["records"] = resolved_records
    current_enriched["token_artifact"] = token_entry
    current_enriched["public_forward_required"] = True
    current_enriched["public_forward_rule"] = "run registered P0/public-prefix on every complete constructed sequence; never splice H rows"
    broad_coverage = token_frequency_summary(broad_token_rows, exclude_special_values=False)
    broad_coverage["token_frequency_by_id"] = dict(sorted(Counter(int(value) for value in broad_token_rows).items()))
    current_enriched["coverage"] = broad_coverage
    current_enriched["domain_length"] = _domain_length_summary_rows(resolved_records)
    current_enriched["controlled"] = {
        "controlled_ids_used": PROPOSED_IDS,
        "controlled_token_id_target": PROPOSED_IDS,
        "controlled_record_count": CONTROLLED_RECORDS,
        "controlled_replacement_occurrences": REPLACEMENTS_TOTAL,
        "controlled_replacement_ids_sha256": _sha256_ints(selected_ids),
        "controlled_slot_minimum_post_bos": min(int(row["target_post_bos_token_count"]) for row in resolved_records if row.get("synthetic")),
        "controlled_slots_all_at_least_128": all(int(row["target_post_bos_token_count"]) >= 128 for row in resolved_records if row.get("synthetic")),
        "controlled_slots_all_at_least_64": True,
    }
    current_arms["coverage_mix_v1"] = current_enriched
    new_plan = dict(current_plan)
    new_plan["generated_at_utc"] = _utc_now()
    new_plan["execution"] = {
        "argv": [str(value) for value in sys.argv],
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_clock,
        "git_commit": _git_commit(root),
        "working_tree_dirty_at_construction": _git_dirty(root),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": int(os.cpu_count() or 0),
            "device": "cpu",
            "model_loaded": False,
            "target_weights_accessed": False,
            "private_truth_accessed": False,
            "network_used": False,
        },
        "resource_usage": {
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "user_cpu_seconds": float(resource.getrusage(resource.RUSAGE_SELF).ru_utime),
            "system_cpu_seconds": float(resource.getrusage(resource.RUSAGE_SELF).ru_stime),
        },
    }
    new_plan["arms"] = current_arms
    # Keep this legacy field at 2,000 because the registered TRR-0005
    # producer validates its historical compatibility contract.  The complete
    # 3,600-ID broader binding is explicit below and in the construction
    # receipt; all 3,600 IDs are present exactly once in the constructed rows.
    new_plan["controlled_token_selection"] = dict(current_plan.get("controlled_token_selection", {}))
    support_binding = {
        "schema": SUPPORT_SCHEMA,
        "task_id": TASK_ID,
        "recipe": _source_descriptor(recipe_path),
        "candidate_pool_frequency": _source_descriptor(candidate_pool_path),
        "current_plan": _source_descriptor(plan_path),
        "current_tokens": _source_descriptor(current_token_path),
        "selected_token_ids": selected_ids,
        "selected_token_id_count": PROPOSED_IDS,
        "baseline_token_ids": baseline_ids,
        "baseline_token_id_count": BASELINE_IDS,
        "additional_token_ids": additions,
        "additional_token_id_count": len(additions),
        "selected_token_ids_sha256": _sha256_ints(selected_ids),
        "baseline_token_ids_sha256": _sha256_ints(baseline_ids),
        "additional_token_ids_sha256": _sha256_ints(additions),
        "candidate_pool_current_unseen_candidate_count": len(candidate_current_unseen),
        "additional_ids_currently_unseen": True,
        "controlled_slot_indices": controlled_slots,
        "controlled_slot_indices_sha256": _sha256_ints(controlled_slots),
        "natural_slot_count": len(natural_slots),
        "natural_slot_indices": natural_slots,
        "natural_fixture": natural_fixture,
        "parent_selection": {
            "seed": CONSTRUCTOR_SEED,
            "ordering": "SHA256(TRR-0007|7007|dataset_id|split|revision|row:index)",
            "selection": "sort source rows by the stable key; for each target slot in descending target length then slot, choose the first remaining source with full-token count >= target+1; clip after BOS",
            "length_matching": "candidate full-token count >= target_post_bos_token_count + 1; clip after BOS to the exact frozen target length",
            "records_by_domain": {key: len(value) for key, value in selected_parents.items()},
            "rows": sorted(parent_rows, key=lambda row: int(row["slot"])),
            "source_scan": scan_stats,
        },
        "replacement_policy": {
            "per_record": REPLACEMENTS_PER_RECORD,
            "total_occurrences": REPLACEMENTS_TOTAL,
            "position_coordinate": "zero-based offsets after BOS in plan; one-based post-BOS positions also recorded",
            "position_bins_one_based": [{"name": "1-15", "lower": 1, "upper": 15, "quota": 3}, {"name": "16-39", "lower": 16, "upper": 39, "quota": 6}, {"name": "40-79", "lower": 40, "upper": 79, "quota": 9}, {"name": "80-127", "lower": 80, "upper": 127, "quota": 12}],
            "selected_ids_consumed_once": True,
            "real_p0_forward_required": True,
        },
        "matched_geometry": {
            "records": FIT_RECORDS,
            "stored_width": MAX_SEQUENCE_LENGTH,
            "post_bos_positions": int(mask_tensor[:, 1:].sum().item()),
            "length_vector_digest": str(current_plan.get("design", {}).get("length_vector_digest", "")),
            "draw_schedule": expected_sampler_exposure(post_bos_positions=POST_BOS_POSITION_COUNT, batch_size=FIT_BATCH_SIZE, steps=FIT_STEPS),
        },
        "access_contract": {
            "public_fit_partitions_only": True,
            "reserved_holdout_rows_scanned": False,
            "public_development_rows_used_for_selection": False,
            "private_truth_accessed": False,
            "target_weights_accessed": False,
            "source_text_retained": False,
        },
    }
    new_plan["trr0007_support"] = support_binding
    new_plan["public_inputs"] = dict(current_plan.get("public_inputs", {}))
    new_plan["public_inputs"]["trr0007_support"] = {
        "recipe": _source_descriptor(recipe_path),
        "candidate_pool_frequency": _source_descriptor(candidate_pool_path),
        "exclusion_manifests": [str(path) for path in exclusion_paths],
    }
    plan_out = output_root / "corpus_plan.json"
    plan_entry = _write_json(plan_out, new_plan)
    exclusion_manifest = _build_exclusion_manifest(
        exclusion_paths=exclusion_paths,
        exclusion_binding=exclusion_binding,
        exclusion_ids=exclusion_ids,
        exclusion_rows=exclusion_rows,
        exclusion_hashes=exclusion_hashes,
        current_rows=current_rows,
        recipe=recipe,
        candidate_count={key: len(value) for key, value in candidates.items()},
    )
    exclusion_entry = _write_json(output_root / "public_parent_exclusion_manifest.json", exclusion_manifest)
    parent_entry = _write_json(output_root / "selected_parent_rows.json", {"rows": sorted(parent_rows, key=lambda row: int(row["slot"]))})
    receipt = {
        "schema": SUPPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "METADATA_BANK_READY_FOR_REAL_P0_CAPTURE",
        "execution": new_plan["execution"],
        "sources": {
            "current_plan": _source_descriptor(plan_path),
            "current_tokens": _source_descriptor(current_token_path),
            "recipe": _source_descriptor(recipe_path),
            "candidate_pool_frequency": _source_descriptor(candidate_pool_path),
            "exclusion_manifests": [_source_descriptor(path) for path in exclusion_paths],
        },
        "geometry": support_binding["matched_geometry"],
        "natural_fixture": natural_fixture,
        "controlled": {
            "record_count": CONTROLLED_RECORDS,
            "replacement_occurrences": REPLACEMENTS_TOTAL,
            "distinct_replacement_ids": len(set(replacement_flat)),
            "selected_ids_consumed_once": True,
            "selected_ids_sha256": _sha256_ints(selected_ids),
            "current_unseen_additions": len(additions),
        },
        "parent_selection": support_binding["parent_selection"],
        "outputs": {
            "token_artifact": token_entry,
            "corpus_plan": plan_entry,
            "public_parent_exclusion_manifest": exclusion_entry,
            "selected_parent_rows": parent_entry,
        },
        "capture_contract": {
            "producer": "scripts/trr0005_prepare_public_activations.py",
            "mode": "capture",
            "geometry": "qualified 8 records x 192 columns, cut depth 4",
            "required": "one real P0/public-prefix forward over all 1,200 complete constructed sequences; no activation splicing",
            "activation_equality_fixture": "compare captured H and masks for all 1,080 retained natural rows against the original public bank before accepting the broader capture",
        },
        "access_contract": support_binding["access_contract"],
        "source_code": {
            "constructor": _source_descriptor(Path(__file__).resolve()),
            "diagnostics": _source_descriptor(Path(__file__).resolve().with_name("trr0007_support_diagnostics.py")),
            "candidate_pool_scanner": _source_descriptor(Path(__file__).resolve().with_name("trr0007_support_candidate_pool.py")),
        },
    }
    receipt_entry = _write_json(output_root / "bank_construction_receipt.json", receipt)
    receipt["outputs"]["bank_construction_receipt"] = receipt_entry
    # The receipt itself is create-only and was written with the output list
    # before its own descriptor was available; write a companion digest for a
    # self-binding final record rather than mutating the create-only receipt.
    binding = {
        "schema": "token-reconstruction.trr0007-public-bank-receipt-binding.v1",
        "task_id": TASK_ID,
        "receipt": receipt_entry,
        "receipt_content_excludes_self_descriptor": True,
        "outputs": receipt["outputs"],
        "status": receipt["status"],
    }
    _write_json(output_root / "bank_receipt_binding.json", binding)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-plan", type=Path, required=True)
    parser.add_argument("--current-tokens", type=Path)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pile-arrow", type=Path, required=True)
    parser.add_argument("--finance-arrow", type=Path, action="append", required=True)
    parser.add_argument("--exclude-records", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = construct(args)
    except (ConstructionError, PreparationError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": receipt["status"],
        "output_root": str(args.output_root.expanduser().resolve()),
        "token_artifact": receipt["outputs"]["token_artifact"]["path"],
        "plan": receipt["outputs"]["corpus_plan"]["path"],
        "parents": len(receipt["parent_selection"]["rows"]),
        "replacement_occurrences": receipt["controlled"]["replacement_occurrences"],
        "distinct_replacement_ids": receipt["controlled"]["distinct_replacement_ids"],
        "elapsed_seconds": receipt["execution"]["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
