#!/usr/bin/env python3
"""Bind and (after an explicit freeze) select the TRR-P06 public source panel.

The ``universe`` command is metadata-only: it freezes the dataset revisions,
candidate ranges, selection seed, and opaque exclusion catalog without reading
an Arrow row or loading a model.  The ``select`` command is intentionally
separate and requires ``--execute`` plus a ``FROZEN_SOURCE_UNIVERSE``.  It may
then render public source rows transiently to compute identity commitments,
but writes no source text, token IDs, labels, observations, or truth.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import trr0005_produce_confirmation as trusted  # noqa: E402
from token_reconstruction.trr0005_public_corpus import SOURCE_PARTITIONS, deterministic_row_order  # noqa: E402
from scripts.trr_p06.source_binding import (  # noqa: E402
    ExclusionIndex,
    SourceBindingError,
    collect_exclusions,
)


TASK_ID = "TRR-P06"
UNIVERSE_SCHEMA = "token-reconstruction.trr-p06-source-universe.v1"
SELECTION_SCHEMA = "token-reconstruction.trr-p06-source-selection.v1"
SELECTION_STATUS = "FROZEN_TRR-P06_SOURCE_SELECTION_NO_TRUTH"
SELECTION_SEED = 6206
RECORDS_PER_DOMAIN = 256
CLIP_TOKENS = 128
CAPTURE_TOKENS = 192
CONDITION_ORDER = ("public_base", "public_lora_2601")
STYLE_ORDER = ("pile", "finance")
# These ranges are a proposal based on the published P04/P05 usage.  They are
# source-universe declarations, not a claim that every row is eligible.  The
# selector still applies all bound identity hashes and records incomplete
# coverage rather than asserting global disjointness.
CANDIDATE_RANGES = {
    "pile": [0, 7000],
    "finance": [20000, 26000],
}
CANDIDATE_RANGE_RATIONALE = {
    "pile": (
        "A low-index public window is retained to provide a broad Pile domain; it may "
        "intersect published P05 fit identities, so every candidate remains subject "
        "to the bound opaque ID/hash/sequence exclusion index."
    ),
    "finance": (
        "The window starts at the published P04 Finance range stop (20000) and "
        "extends into a public-only candidate region; published identity ledgers "
        "still govern exact eligibility."
    ),
}



class PanelPreparationError(RuntimeError):
    """Raised when the P06 source universe or selection is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PanelPreparationError(f"file is unavailable: {path}")
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
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PanelPreparationError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PanelPreparationError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise PanelPreparationError(f"{description} must be a JSON object")
    return dict(value)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PanelPreparationError(f"output is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


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
    value = result.stdout.strip()
    return value or None


def _reject_payload(value: Any, *, path: str = "value") -> None:
    forbidden = {
        "source_text",
        "plaintext",
        "token_ids",
        "input_ids",
        "labels",
        "target_labels",
        "truth",
        "truth_ids",
        "truth_tokens",
        "truth_labels",
        "answers",
        "correctness",
        "private_truth",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold().replace("-", "_")
            if lowered in forbidden:
                raise PanelPreparationError(f"{path}.{key} contains source/truth payload")
            _reject_payload(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_payload(child, path=f"{path}[{index}]")


def _plan_binding(plan_path: Path) -> dict[str, Any]:
    plan = _load_json(plan_path, description="P06 plan")
    if plan.get("task_id") != TASK_ID:
        raise PanelPreparationError("P06 plan task ID changed")
    if plan.get("schema") != "token-reconstruction.trr-p06-plan.v1":
        raise PanelPreparationError("P06 plan schema changed")
    if plan.get("truth_opened") is True or plan.get("evaluation_truth_opened") is True:
        raise PanelPreparationError("P06 plan was written after truth access")
    fresh = plan.get("fresh_panel")
    if not isinstance(fresh, Mapping):
        raise PanelPreparationError("P06 plan has no fresh-panel section")
    if int(fresh.get("selection_seed", -1)) != SELECTION_SEED:
        raise PanelPreparationError("P06 selection seed differs from the proposed plan")
    if int(fresh.get("records_per_domain", -1)) != RECORDS_PER_DOMAIN:
        raise PanelPreparationError("P06 panel size differs from the proposed plan")
    if list(fresh.get("target_conditions", ())) != list(CONDITION_ORDER):
        raise PanelPreparationError("P06 target condition order differs from the proposed plan")
    _reject_payload(plan)
    return {
        "path": str(Path(plan_path).expanduser().resolve()),
        "bytes": Path(plan_path).stat().st_size,
        "sha256": _sha256_file(plan_path),
        "status": plan.get("status"),
    }


def _dataset_spec(style: str) -> dict[str, Any]:
    if style not in SOURCE_PARTITIONS:
        raise PanelPreparationError(f"unknown P06 domain: {style}")
    source = SOURCE_PARTITIONS[style]
    start, stop = CANDIDATE_RANGES[style]
    return {
        "dataset_key": style,
        "dataset_id": str(source["dataset_id"]),
        "split": str(source["split"]),
        "revision": str(source["revision"]),
        "candidate_range_half_open": [int(start), int(stop)],
        "minimum_valid_tokens_including_bos": CLIP_TOKENS,
    }


def _universe_metadata(
    *,
    root: Path,
    plan_binding: Mapping[str, Any],
    exclusions: ExclusionIndex,
) -> dict[str, Any]:
    return {
        "schema": UNIVERSE_SCHEMA,
        "task_id": TASK_ID,
        "status": "PROPOSED_BEFORE_ENUMERATION",
        "created_utc": _utc_now(),
        "provenance": {
            "repository_root": str(root),
            "plan": dict(plan_binding),
            "code_commit": _git_commit(root),
            "selection_seed": SELECTION_SEED,
        },
        "candidate_source_universe": {
            style: {**_dataset_spec(style), "rationale": CANDIDATE_RANGE_RATIONALE[style]}
            for style in STYLE_ORDER
        },
        "panel_contract": {
            "records_per_domain": RECORDS_PER_DOMAIN,
            "unique_source_records": RECORDS_PER_DOMAIN * len(STYLE_ORDER),
            "clip_tokens_including_bos": CLIP_TOKENS,
            "capture_sequence_tokens": CAPTURE_TOKENS,
            "scored_post_bos_tokens": CLIP_TOKENS - 1,
            "target_conditions": list(CONDITION_ORDER),
            "same_source_order_across_targets": True,
            "selection_is_not_started": True,
        },
        "exclusion_binding": exclusions.as_metadata(),
        "coverage_disclaimer": {
            "global_disjoint_claim": False,
            "reason": "P01-P03 and TRR-0007 opaque reservations are external inputs; any missing descriptor keeps coverage incomplete",
            "candidate_ranges_are_not_eligibility": True,
            "source_rows_or_token_values_persisted": False,
        },
        "access_boundary": {
            "source_rows_read": False,
            "tokenizer_loaded": False,
            "model_loaded": False,
            "panel_selected": False,
            "truth_opened": False,
            "trr0007_mutable_or_private_accessed": False,
            "p03_holdout_accessed": False,
        },
    }


def build_source_universe(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    plan_binding = _plan_binding(Path(args.plan))
    exclusions = collect_exclusions(
        root,
        approved_opaque_paths=tuple(args.approved_opaque or ()),
    )
    value = _universe_metadata(root=root, plan_binding=plan_binding, exclusions=exclusions)
    output = _write_create_only(Path(args.output), value)
    return {"task_id": TASK_ID, "status": value["status"], "universe": output}


def load_universe(path: Path, *, require_frozen: bool = False) -> dict[str, Any]:
    value = _load_json(path, description="P06 source universe")
    if value.get("schema") != UNIVERSE_SCHEMA or value.get("task_id") != TASK_ID:
        raise PanelPreparationError("source universe schema or task ID changed")
    status = str(value.get("status", ""))
    if require_frozen and status != "FROZEN_SOURCE_UNIVERSE":
        raise PanelPreparationError("source selection requires FROZEN_SOURCE_UNIVERSE")
    if int(value.get("provenance", {}).get("selection_seed", -1)) != SELECTION_SEED:
        raise PanelPreparationError("source universe selection seed changed")
    contract = value.get("panel_contract")
    if not isinstance(contract, Mapping):
        raise PanelPreparationError("source universe has no panel contract")
    if int(contract.get("records_per_domain", -1)) != RECORDS_PER_DOMAIN:
        raise PanelPreparationError("source universe panel count changed")
    if int(contract.get("clip_tokens_including_bos", -1)) != CLIP_TOKENS:
        raise PanelPreparationError("source universe clip length changed")
    if list(contract.get("target_conditions", ())) != list(CONDITION_ORDER):
        raise PanelPreparationError("source universe target order changed")
    sources = value.get("candidate_source_universe")
    if not isinstance(sources, Mapping):
        raise PanelPreparationError("source universe has no candidate domains")
    for style in STYLE_ORDER:
        spec = sources.get(style)
        if not isinstance(spec, Mapping) or list(spec.get("candidate_range_half_open", ())) != CANDIDATE_RANGES[style]:
            raise PanelPreparationError(f"source universe range changed for {style}")
    _reject_payload(value)
    return value


def _read_candidate_row(dataset: Any, *, style: str, row_index: int) -> Mapping[str, Any]:
    start, stop = CANDIDATE_RANGES[style]
    if not start <= row_index < stop:
        raise PanelPreparationError(f"{style} candidate row escaped declared range: {row_index}")
    try:
        row = dataset[row_index]
    except Exception as exc:
        raise PanelPreparationError(f"{style} candidate row could not be read: {row_index}") from exc
    if not isinstance(row, Mapping):
        raise PanelPreparationError(f"{style} candidate row is malformed: {row_index}")
    return row


def _selection_metadata(candidate: Any, *, style: str) -> dict[str, Any]:
    metadata = dict(candidate.selection_metadata())
    allowed = {
        "record_id",
        "public_record_sha256",
        "dataset_key",
        "dataset_id",
        "split",
        "revision",
        "row_index",
        "source_index",
        "full_token_count",
        "post_bos_token_count",
        "valid_tokens",
        "final_sequence_sha256",
    }
    if set(metadata) - allowed:
        raise PanelPreparationError("renderer emitted an unapproved source payload field")
    if metadata.get("dataset_key") != style or metadata.get("source_index") != metadata.get("row_index"):
        raise PanelPreparationError("renderer source binding changed")
    if metadata.get("valid_tokens") != CLIP_TOKENS:
        raise PanelPreparationError("candidate does not provide the fixed P06 clip")
    result = {key: metadata[key] for key in allowed}
    result["domain"] = style
    return result


def _select_records(
    *,
    universe: Mapping[str, Any],
    datasets: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    exclusion_meta = universe.get("exclusion_binding")
    if not isinstance(exclusion_meta, Mapping):
        raise PanelPreparationError("source universe has no exclusion binding")
    if exclusion_meta.get("coverage_complete") is not True:
        raise PanelPreparationError(
            "incomplete prior exclusion coverage; bind all approved opaque reservations before selection"
        )
    # Rebind the metadata paths instead of trusting counts written in the
    # universe.  The caller must provide the same catalog through the frozen
    # universe's descriptor list; actual row selection never uses a stale set.
    descriptor_paths = [
        Path(str(item["path"]))
        for item in exclusion_meta.get("descriptors", [])
        if isinstance(item, Mapping) and item.get("available") is True
    ]
    exclusions = collect_exclusions(
        Path(str(universe.get("provenance", {}).get("repository_root", "."))),
        metadata_paths=descriptor_paths,
        include_default_catalog=False,
    )
    selected: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, int] = {"scanned": 0, "invalid": 0, "excluded": 0, "duplicates": 0}
    seen_ids: set[str] = set()
    seen_rendered: set[str] = set()
    seen_sequences: set[str] = set()
    for style in STYLE_ORDER:
        rows: list[dict[str, Any]] = []
        order = deterministic_row_order(
            range(*CANDIDATE_RANGES[style]),
            dataset_key=f"trr-p06-{style}",
            seed=SELECTION_SEED,
        )
        for row_index in order:
            if len(rows) >= RECORDS_PER_DOMAIN:
                break
            stats["scanned"] += 1
            try:
                row = _read_candidate_row(datasets[style], style=style, row_index=row_index)
                candidate = trusted._render_row(style, row, row_index, tokenizer)
            except (trusted.ProducerError, PanelPreparationError):
                stats["invalid"] += 1
                continue
            metadata = _selection_metadata(candidate, style=style)
            sequence_129 = None
            if len(candidate.token_ids) >= CLIP_TOKENS + 1:
                sequence_129 = trusted._sequence_digest(candidate.token_ids[: CLIP_TOKENS + 1])
            reason = exclusions.block_reason(
                record_id=str(metadata["record_id"]),
                public_record_sha256=str(metadata["public_record_sha256"]),
                final_sequence_sha256=str(metadata["final_sequence_sha256"]),
                sequence_sha256_129=sequence_129,
                row_index=int(metadata["row_index"]),
            )
            if reason is not None:
                stats["excluded"] += 1
                continue
            if (
                metadata["record_id"] in seen_ids
                or metadata["public_record_sha256"] in seen_rendered
                or metadata["final_sequence_sha256"] in seen_sequences
            ):
                stats["duplicates"] += 1
                continue
            seen_ids.add(str(metadata["record_id"]))
            seen_rendered.add(str(metadata["public_record_sha256"]))
            seen_sequences.add(str(metadata["final_sequence_sha256"]))
            rows.append(metadata)
        if len(rows) != RECORDS_PER_DOMAIN:
            raise PanelPreparationError(
                f"{style} candidate universe has only {len(rows)} eligible rows; expected {RECORDS_PER_DOMAIN}"
            )
        selected[style] = rows
    return selected, stats


def select_panel(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise PanelPreparationError("source selection requires explicit --execute")
    root = Path(args.repository_root).expanduser().resolve()
    universe_path = Path(args.universe).expanduser().resolve()
    universe = load_universe(universe_path, require_frozen=True)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    tokenizer = trusted._load_tokenizer(tokenizer_path)
    pile_paths = tuple(Path(value).expanduser().resolve() for value in args.pile_arrow)
    finance_paths = tuple(Path(value).expanduser().resolve() for value in args.finance_arrow)
    datasets = {
        "pile": trusted._load_arrow_dataset(pile_paths),
        "finance": trusted._load_arrow_dataset(finance_paths),
    }
    selected, stats = _select_records(universe=universe, datasets=datasets, tokenizer=tokenizer)
    selection = {
        "schema": SELECTION_SCHEMA,
        "task_id": TASK_ID,
        "status": SELECTION_STATUS,
        "created_utc": _utc_now(),
        "selection_seed": SELECTION_SEED,
        "records_per_domain": RECORDS_PER_DOMAIN,
        "domains": list(STYLE_ORDER),
        "target_conditions": list(CONDITION_ORDER),
        "paired_conditions": True,
        "clip_tokens_including_bos": CLIP_TOKENS,
        "capture_sequence_tokens": CAPTURE_TOKENS,
        "scored_post_bos_tokens": CLIP_TOKENS - 1,
        "source_universe": {
            "path": str(universe_path),
            "bytes": universe_path.stat().st_size,
            "sha256": _sha256_file(universe_path),
            "catalog_sha256": universe["exclusion_binding"]["catalog_sha256"],
        },
        "public_sources_frozen": {
            "pile": trusted._dataset_descriptor(pile_paths, style="pile"),
            "finance": trusted._dataset_descriptor(finance_paths, style="finance"),
            "tokenizer": trusted._tokenizer_descriptor(tokenizer_path),
        },
        "source_ranges_half_open": dict(CANDIDATE_RANGES),
        "source_range_rationale": dict(CANDIDATE_RANGE_RATIONALE),
        "selection_rule": {
            "order": "deterministic_row_order over each declared range using trr-p06-{domain} and seed 6206",
            "records": selected,
            "source_text_or_token_ids_written": False,
            "target_labels_loaded": False,
            "truth_opened": False,
        },
        "selection_audit": stats,
        "access_boundary": {
            "source_rows_read_transiently": True,
            "source_text_written": False,
            "token_ids_written": False,
            "truth_opened": False,
            "target_weights_accessed": False,
            "candidate_simulations": 0,
        },
        "code_commit": _git_commit(root),
    }
    output = _write_create_only(Path(args.output), selection)
    return {"task_id": TASK_ID, "status": SELECTION_STATUS, "selection": output, "audit": stats}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    universe = sub.add_parser("universe", help="bind metadata without reading source rows")
    universe.add_argument("--repository-root", type=Path, default=Path("."))
    universe.add_argument("--plan", type=Path, required=True)
    universe.add_argument("--approved-opaque", type=Path, nargs="*", default=[])
    universe.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select", help="select a frozen panel after explicit authorization")
    select.add_argument("--execute", action="store_true")
    select.add_argument("--repository-root", type=Path, default=Path("."))
    select.add_argument("--universe", type=Path, required=True)
    select.add_argument("--tokenizer", type=Path, required=True)
    select.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "universe":
        result = build_source_universe(args)
    else:
        result = select_panel(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PanelPreparationError, SourceBindingError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"TRR-P06 panel preparation error: {exc}")
