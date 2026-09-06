#!/usr/bin/env python3
"""Select a frozen TRR-0006 source panel after the owner freezes its plan.

This command is deliberately separate from the count-only eligibility scan.
It refuses draft plans, validates the declared Pile/Finance ranges and clip
geometry, then reuses the trusted TRR-0005 renderer and partition guard.  The
selection artifact contains public source identities and hashes only; it does
not contain source text, token IDs, observations, targets, or truth.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_build_eligibility as eligibility
from token_reconstruction.trr0005_contract import STYLE_ORDER
from token_reconstruction.trr0005_public_corpus import (
    SOURCE_PARTITIONS,
    deterministic_row_order,
    source_record_id,
)


TASK_ID = "TRR-0006"
SELECTION_SCHEMA = "token-reconstruction.trr0006-source-selection.v1"
SELECTION_SEED = 5005
SEQUENCE_TOKENS = 128
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
CONDITION_ORDER = ("public_base", "public_lora_2601")


class SelectionError(RuntimeError):
    """Raised when a source selection cannot satisfy the frozen plan."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SelectionError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise SelectionError(f"{description} must be a JSON object")
    return dict(value)


def _reject_source_payload(value: Any, *, path: str = "plan") -> None:
    """Reject source-bearing fields in the source-free frozen plan."""

    # Frozen decision plans carry truth *gate/status* metadata.  Those control
    # fields are permitted; source identity, token, label, and evaluator-side
    # truth payloads are not.  Keep the check key-based so a prose rationale
    # mentioning "truth" or "record" does not make a valid frozen plan
    # unusable.
    allowed_control_keys = {
        "truth_gate",
        "truth_opened",
        "truth_status",
        "truth_required",
        "truth_access_status",
    }
    exact_forbidden = {
        "record_id",
        "source_record_id",
        "public_record_id",
        "source_text",
        "plaintext",
        "token_ids",
        "input_ids",
        "labels",
        "truth",
        "oracle",
        "truth_file",
        "truth_path",
        "truth_ids",
        "truth_tokens",
        "truth_labels",
        "evaluation_truth",
        "target_labels",
    }
    forbidden_fragments = (
        "source_text",
        "plaintext",
        "token_ids",
        "input_ids",
        "target_labels",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold().replace("-", "_")
            if lowered in allowed_control_keys:
                _reject_source_payload(child, path=f"{path}.{key}")
                continue
            if lowered in exact_forbidden or any(fragment in lowered for fragment in forbidden_fragments):
                raise SelectionError(f"{path}.{key} contains source/truth payload")
            if "record_id" in lowered or (lowered.startswith("truth_") and lowered not in allowed_control_keys):
                raise SelectionError(f"{path}.{key} contains source/truth payload")
            _reject_source_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_source_payload(child, path=f"{path}[{index}]")


def _plan_panel(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    panel = plan.get("panel", plan.get("fresh_evaluation", plan))
    return panel if isinstance(panel, Mapping) else plan


def _plan_records_per_domain(plan: Mapping[str, Any]) -> int:
    panel = _plan_panel(plan)
    value = panel.get("records_per_domain", panel.get("unique_source_records_per_domain"))
    if value is None:
        value = plan.get("records_per_domain")
    if isinstance(value, bool):
        raise SelectionError("frozen plan records_per_domain is not an integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError("frozen plan records_per_domain is absent") from exc
    if count <= 0:
        raise SelectionError("frozen plan records_per_domain must be positive")
    return count


def _validate_frozen_plan(path: Path) -> dict[str, Any]:
    plan = _load_json(path, description="TRR-0006 frozen selection plan")
    if plan.get("schema") != "token-reconstruction.trr0006-decision-plan.v1":
        raise SelectionError("selection plan schema changed")
    if plan.get("task_id") != TASK_ID:
        raise SelectionError("selection plan task ID changed")
    status = str(plan.get("status", ""))
    if not status.startswith("FROZEN"):
        raise SelectionError("selection requires a frozen plan; draft plan rejected")
    if plan.get("truth_opened") is True or plan.get("evaluation_truth_opened") is True:
        raise SelectionError("selection plan was written after truth access")
    if plan.get("sample_size_frozen") is False:
        raise SelectionError("selection plan does not freeze the sample size")
    _reject_source_payload(plan)
    count = _plan_records_per_domain(plan)
    panel = _plan_panel(plan)
    # These are required from the actual frozen panel/comparison objects.
    # Defaults here would let a malformed nested plan bypass a geometry or
    # population check while still looking superficially frozen.
    if panel.get("clip_tokens_including_bos") != SEQUENCE_TOKENS:
        raise SelectionError("frozen plan clip geometry changed")
    if panel.get("capture_tokens") != CAPTURE_SEQUENCE_TOKENS:
        raise SelectionError("frozen plan capture sequence changed")
    if panel.get("capture_batch_records") != CAPTURE_BATCH_RECORDS:
        raise SelectionError("frozen plan capture batch changed")
    if panel.get("selection_seed") != SELECTION_SEED:
        raise SelectionError("frozen plan selection seed changed")
    expected_ranges = {
        style: [
            int(SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
            int(SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
        ]
        for style in STYLE_ORDER
    }
    if panel.get("source_ranges_half_open") != expected_ranges:
        raise SelectionError("frozen plan source ranges changed")
    comparison = plan.get("comparison")
    conditions = comparison.get("target_conditions") if isinstance(comparison, Mapping) else None
    if conditions != list(CONDITION_ORDER):
        raise SelectionError("frozen plan target conditions changed")
    method_hash = plan.get("method_freeze_sha256")
    if not isinstance(method_hash, str) or len(method_hash) != 64:
        freeze = plan.get("method_freeze")
        if isinstance(freeze, Mapping):
            method_hash = freeze.get("sha256", freeze.get("method_freeze_sha256"))
    if not isinstance(method_hash, str) or len(method_hash) != 64:
        provenance = plan.get("provenance")
        if isinstance(provenance, Mapping):
            method_hash = provenance.get("trr5_method_freeze_sha256")
    if not isinstance(method_hash, str) or len(method_hash) != 64:
        raise SelectionError("frozen plan lacks a method-freeze SHA-256 binding")
    if any(character not in "0123456789abcdef" for character in method_hash.casefold()):
        raise SelectionError("frozen plan method-freeze hash is malformed")
    return {
        "path": str(path.expanduser().resolve()),
        "sha256": _sha256_file(path.expanduser().resolve()),
        "status": status,
        "records_per_domain": count,
        "method_freeze_sha256": method_hash,
        "value": plan,
    }


def _select_domain(
    dataset: Any,
    *,
    style: str,
    tokenizer: Any,
    records_per_domain: int,
    exclusions: Any,
    opaque: eligibility.OpaqueExclusions,
    seen_public_hashes: set[str],
    seen_final_sequences: set[str],
) -> tuple[list[Any], dict[str, int]]:
    spec = SOURCE_PARTITIONS[style]
    start = int(spec["holdout_reserve_start"])
    stop = int(spec["holdout_reserve_stop"])
    if len(dataset) < stop:
        raise SelectionError(f"{style} cache has {len(dataset)} rows; need {stop}")
    selected: list[Any] = []
    skipped = {
        "excluded_id": 0,
        "excluded_index": 0,
        "excluded_hash": 0,
        "excluded_opaque_source_hash": 0,
        "excluded_opaque_sequence_hash": 0,
        "invalid": 0,
        "duplicate_rendered_source": 0,
        "duplicate_final_sequence": 0,
    }
    order = deterministic_row_order(
        range(start, stop),
        dataset_key=f"{style}-future-holdout",
        seed=SELECTION_SEED,
    )
    for index in order:
        expected_id = source_record_id(
            str(spec["dataset_id"]),
            str(spec["split"]),
            str(spec["revision"]),
            index,
        )
        if expected_id in exclusions.ids[style]:
            skipped["excluded_id"] += 1
            continue
        if index in exclusions.indices[style]:
            skipped["excluded_index"] += 1
            continue
        row = trusted._read_reserved_row(dataset, style=style, row_index=index)
        try:
            candidate = trusted._render_row(style, row, index, tokenizer)
        except trusted.ProducerError:
            skipped["invalid"] += 1
            continue
        reason = eligibility._classify_valid_candidate(
            candidate,
            style=style,
            exclusions=exclusions,
            opaque=opaque,
            seen_public_hashes=seen_public_hashes,
            seen_final_sequences=seen_final_sequences,
        )
        if reason == "eligible":
            selected.append(candidate)
            if len(selected) == records_per_domain:
                break
        elif reason in skipped:
            skipped[reason] += 1
        else:  # pragma: no cover - defensive fail-closed branch
            raise SelectionError(f"unknown source classification: {reason}")
    if len(selected) != records_per_domain:
        raise SelectionError(
            f"{style} eligible pool yielded {len(selected)} rows; need {records_per_domain}"
        )
    if len({record.record_id for record in selected}) != records_per_domain:
        raise SelectionError(f"{style} selected source IDs are not unique")
    return selected, skipped


def _record_metadata(record: Any) -> dict[str, Any]:
    return dict(record.selection_metadata())


def _exclusion_source_descriptors(exclusions: Any) -> list[dict[str, Any]]:
    return [
        {
            key: source[key]
            for key in ("path", "available", "bytes", "sha256", "new_identity_count")
            if key in source
        }
        for source in exclusions.sources
    ]


def select_public(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    frozen = _validate_frozen_plan(args.freeze_plan)
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SelectionError(f"selection output is create-only and already exists: {output}")
    started_utc = _utc_now()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    tokenizer = trusted._load_tokenizer(tokenizer_path)
    pile_paths = tuple(path.expanduser().resolve() for path in args.pile_arrow)
    finance_paths = tuple(path.expanduser().resolve() for path in args.finance_arrow)
    datasets = {
        "pile": trusted._load_arrow_dataset(pile_paths),
        "finance": trusted._load_arrow_dataset(finance_paths),
    }
    exclusion_paths = eligibility._known_exclusion_paths(root)
    exclusion_paths.extend(path.expanduser().resolve() for path in args.exclude_source)
    exclusions = trusted._collect_exclusions(exclusion_paths)
    opaque = eligibility._load_p04_opaque_exclusions(args.p04_exchange)
    seen_public_hashes: set[str] = set()
    seen_final_sequences: set[str] = set()
    records: dict[str, list[Any]] = {}
    skipped: dict[str, dict[str, int]] = {}
    for style in STYLE_ORDER:
        records[style], skipped[style] = _select_domain(
            datasets[style],
            style=style,
            tokenizer=tokenizer,
            records_per_domain=frozen["records_per_domain"],
            exclusions=exclusions,
            opaque=opaque,
            seen_public_hashes=seen_public_hashes,
            seen_final_sequences=seen_final_sequences,
        )
    selected_metadata = {
        style: [_record_metadata(record) for record in records[style]]
        for style in STYLE_ORDER
    }
    selected_ids = {
        style: [record.record_id for record in records[style]] for style in STYLE_ORDER
    }
    selected_sequence_hashes = {
        style: [record.final_sequence_sha256 for record in records[style]]
        for style in STYLE_ORDER
    }
    ended_utc = _utc_now()
    result: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_TRR0006_SOURCE_SELECTION_NO_TRUTH",
        "selection_plan": {
            "path": frozen["path"],
            "sha256": frozen["sha256"],
            "status": frozen["status"],
            "records_per_domain": frozen["records_per_domain"],
        },
        "method_freeze_sha256": frozen["method_freeze_sha256"],
        "selection_seed": SELECTION_SEED,
        "sequence_tokens": SEQUENCE_TOKENS,
        "records_per_domain": frozen["records_per_domain"],
        "source_ranges_half_open": {
            style: [
                int(SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
                int(SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
            ]
            for style in STYLE_ORDER
        },
        "target_conditions": list(CONDITION_ORDER),
        "paired_conditions": True,
        "public_sources_frozen": {
            "pile": trusted._dataset_descriptor(pile_paths, style="pile"),
            "finance": trusted._dataset_descriptor(finance_paths, style="finance"),
            "tokenizer": trusted._tokenizer_descriptor(tokenizer_path),
        },
        "selection_rule": {
            "algorithm": "Deterministic TRR-0005 row order over each declared half-open pool; reject known public identities, P04 opaque source/129-token sequence hashes, invalid rows, and duplicate rendered sources or 128-token sequences; retain the first frozen-capacity eligible rows.",
            "identity_exclusions": True,
            "opaque_p04_hashes_applied": True,
            "source_text_or_token_ids_written": False,
            "record_ids_sha256": {
                style: _json_sha256(selected_ids[style]) for style in STYLE_ORDER
            },
            "final_sequence_sha256": {
                style: _json_sha256(selected_sequence_hashes[style])
                for style in STYLE_ORDER
            },
            "records": selected_metadata,
        },
        "selection_exclusions": {
            "sources": _exclusion_source_descriptors(exclusions),
            "identity_counts": {
                style: {
                    "ids": len(exclusions.ids[style]),
                    "hashes": len(exclusions.hashes[style]),
                    "indices": len(exclusions.indices[style]),
                }
                for style in STYLE_ORDER
            },
            "p04_exchange": opaque.exchange,
            "p04_field_summaries": opaque.fields,
            "targetfit_per_record_metadata_available": False,
        },
        "selection_diagnostics": {
            style: {
                **skipped[style],
                "selected": len(records[style]),
                "pool_size": int(SOURCE_PARTITIONS[style]["holdout_reserve_stop"])
                - int(SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
            }
            for style in STYLE_ORDER
        },
        "execution": {
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "command": list(sys.argv),
            "code_commit": eligibility._git_commit(root),
            "producer_source": {
                "path": str(Path(__file__).resolve()),
                "bytes": Path(__file__).stat().st_size,
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "trusted_trr0005_producer_source": {
                "path": str(Path(trusted.__file__).resolve()),
                "bytes": Path(trusted.__file__).stat().st_size,
                "sha256": _sha256_file(Path(trusted.__file__).resolve()),
            },
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "model_loaded": False,
            "target_loaded": False,
            "truth_created_or_opened": False,
            "selection_performed": True,
            "network_used": False,
        },
        "limitations": [
            "P04 target-fit per-record identities, source ranges, and replay sequence hashes were unavailable and therefore remain an explicit disjointness limitation.",
            "This source selection is valid only for the frozen source range and target pair named here; it does not certify unseen-source disjointness from unavailable P04 target-fit rows.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {
        "task_id": TASK_ID,
        "status": result["status"],
        "selection_plan": str(output),
        "selection_plan_sha256": _sha256_file(output),
        "records_per_domain": frozen["records_per_domain"],
        "truth_created_or_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_subparsers(dest="command", required=True).add_parser(
        "select", help="select public rows under an already frozen TRR-0006 plan"
    )
    select.add_argument("--repository-root", type=Path, default=Path("."))
    select.add_argument("--freeze-plan", type=Path, required=True)
    select.add_argument("--tokenizer", type=Path, required=True)
    select.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--exclude-source", type=Path, nargs="*", default=[])
    select.add_argument("--p04-exchange", type=Path, default=eligibility.P04_EXCHANGE_PATH)
    select.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "select":  # pragma: no cover
        raise SelectionError(f"unknown command: {args.command}")
    result = select_public(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SelectionError, eligibility.EligibilityError, trusted.ProducerError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0006 selection error: {exc}")
