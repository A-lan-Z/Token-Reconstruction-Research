#!/usr/bin/env python3
"""Prepare the TRR-P08 public source universe and (later) panel.

This adapter reuses the published P06 source renderer, exclusion index, and
deterministic row-order implementation.  ``universe`` only binds a P08 plan,
dataset ranges, seed, and approved hash-only ledgers; it never enumerates a
row.  ``freeze`` verifies those bindings and creates a frozen metadata-only
universe.  ``select`` is a separate explicit operation that may transiently
render public rows, but writes only source identities and digests.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import trr0005_produce_confirmation as trusted  # noqa: E402
from scripts.trr_p06 import prepare_public_panel as p06  # noqa: E402


TASK_ID = "TRR-P08"
PLAN_SCHEMA = "token-reconstruction.trr-p08-plan.v1"
UNIVERSE_SCHEMA = "token-reconstruction.trr-p08-source-universe.v1"
SELECTION_SCHEMA = "token-reconstruction.trr-p08-source-selection.v1"
SELECTION_STATUS = "FROZEN_TRR-P08_SOURCE_SELECTION_NO_TRUTH"
STYLE_ORDER = ("pile", "finance")
CONDITION_ORDER = ("public_base", "public_lora_2601")
RECORDS_PER_DOMAIN = 256
CLIP_TOKENS = 128
CAPTURE_TOKENS = 192


class PanelPreparationError(RuntimeError):
    """Raised when a P08 source contract is not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return p06._sha256_file(Path(path))


def _json_load(path: Path, *, label: str) -> dict[str, Any]:
    return p06._load_json(Path(path), description=label)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return p06._write_create_only(Path(path), value)


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
    try:
        p06._reject_payload(value, path=path)
    except p06.PanelPreparationError as exc:
        raise PanelPreparationError(str(exc)) from exc


def _configure_p06(*, seed: int, ranges: Mapping[str, Sequence[int]]) -> None:
    """Configure the imported P06 helpers for this isolated P08 process."""

    normalized = {}
    for style in STYLE_ORDER:
        raw = ranges.get(style)
        if not isinstance(raw, Sequence) or len(raw) != 2:
            raise PanelPreparationError(f"missing {style} half-open source range")
        start, stop = raw
        if isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int) or not isinstance(stop, int):
            raise PanelPreparationError(f"{style} source range must contain integers")
        if start < 0 or stop <= start:
            raise PanelPreparationError(f"invalid {style} half-open source range")
        normalized[style] = [int(start), int(stop)]

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PanelPreparationError("selection seed must be a non-negative integer")
    p06.TASK_ID = TASK_ID
    p06.UNIVERSE_SCHEMA = UNIVERSE_SCHEMA
    p06.SELECTION_SCHEMA = SELECTION_SCHEMA
    p06.SELECTION_STATUS = SELECTION_STATUS
    p06.SELECTION_SEED = int(seed)
    p06.RECORDS_PER_DOMAIN = RECORDS_PER_DOMAIN
    p06.CLIP_TOKENS = CLIP_TOKENS
    p06.CAPTURE_TOKENS = CAPTURE_TOKENS
    p06.CONDITION_ORDER = CONDITION_ORDER
    p06.STYLE_ORDER = STYLE_ORDER
    p06.CANDIDATE_RANGES = normalized


def _plan_binding(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).expanduser().resolve()
    plan = _json_load(plan_path, label="P08 plan")
    if plan.get("task_id") != TASK_ID or plan.get("schema") != PLAN_SCHEMA:
        raise PanelPreparationError("P08 plan schema or task ID changed")
    if plan.get("status") != "FROZEN_DESIGN_PRE_FIT":
        raise PanelPreparationError("P08 design is not frozen for source binding")
    scope = plan.get("scope")
    fresh = plan.get("fresh_evaluation")
    if not isinstance(scope, Mapping) or int(scope.get("fit_arms", -1)) != 8:
        raise PanelPreparationError("P08 plan does not bind all eight fit arms")
    if not isinstance(fresh, Mapping) or int(fresh.get("records_per_domain", -1)) != RECORDS_PER_DOMAIN:
        raise PanelPreparationError("P08 plan panel count changed")
    if list(fresh.get("target_conditions", ())) != list(CONDITION_ORDER):
        raise PanelPreparationError("P08 target condition order changed")
    _reject_payload(plan)
    return {
        "path": str(plan_path),
        "bytes": plan_path.stat().st_size,
        "sha256": _sha256_file(plan_path),
        "status": plan.get("status"),
        "fit_arms": int(scope["fit_arms"]),
        "records_per_domain": RECORDS_PER_DOMAIN,
        "target_conditions": list(CONDITION_ORDER),
    }


def _dataset_spec(style: str) -> dict[str, Any]:
    try:
        return p06._dataset_spec(style)
    except p06.PanelPreparationError as exc:
        raise PanelPreparationError(str(exc)) from exc


# Names below are the narrow helper contract consumed by the P06 capture
# primitives. They remain wrappers so P08 errors use this module's exception
# type and the P06 constants are always the ones configured from the frozen
# universe.
def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    return _json_load(path, label=description)


def _read_candidate_row(dataset: Any, *, style: str, row_index: int) -> Mapping[str, Any]:
    try:
        return p06._read_candidate_row(dataset, style=style, row_index=row_index)
    except p06.PanelPreparationError as exc:
        raise PanelPreparationError(str(exc)) from exc


def _frozen_descriptor_paths(exclusion_meta: Mapping[str, Any]) -> tuple[Path, ...]:
    try:
        return p06._frozen_descriptor_paths(exclusion_meta)
    except p06.PanelPreparationError as exc:
        raise PanelPreparationError(str(exc)) from exc


def _universe_metadata(
    *,
    root: Path,
    plan_binding: Mapping[str, Any],
    seed: int,
    ranges: Mapping[str, Sequence[int]],
    exclusions: Any,
) -> dict[str, Any]:
    _configure_p06(seed=seed, ranges=ranges)
    return {
        "schema": UNIVERSE_SCHEMA,
        "task_id": TASK_ID,
        "status": "PROPOSED_BEFORE_ENUMERATION",
        "created_utc": _utc_now(),
        "provenance": {
            "repository_root": str(root),
            "plan": dict(plan_binding),
            "code_commit": _git_commit(root),
            "selection_seed": int(seed),
        },
        "candidate_source_universe": {
            style: {
                **_dataset_spec(style),
                "rationale": "Range is a predeclared source-universe bound; eligibility still requires every frozen exclusion ledger and the fixed 128-token validity rule.",
            }
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
            "reason": "Hash-only prior ledgers are bounded metadata and do not certify universal overlap absence.",
            "candidate_ranges_are_not_eligibility": True,
            "source_rows_or_token_values_persisted": False,
        },
        "access_boundary": {
            "source_rows_read": False,
            "tokenizer_loaded": False,
            "model_loaded": False,
            "panel_selected": False,
            "truth_opened": False,
            "trr0009_mutable_or_private_accessed": False,
            "p03_holdout_accessed": False,
        },
    }


def build_source_universe(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    plan_binding = _plan_binding(Path(args.plan))
    ranges = {"pile": list(args.pile_range), "finance": list(args.finance_range)}
    _configure_p06(seed=int(args.selection_seed), ranges=ranges)
    exclusions = p06.collect_exclusions(
        root,
        approved_opaque_paths=tuple(args.approved_opaque or ()),
    )
    value = _universe_metadata(
        root=root,
        plan_binding=plan_binding,
        seed=int(args.selection_seed),
        ranges=ranges,
        exclusions=exclusions,
    )
    output = _write_create_only(Path(args.output), value)
    return {"task_id": TASK_ID, "status": value["status"], "universe": output}


def load_universe(path: Path, *, require_frozen: bool = False) -> dict[str, Any]:
    value = _json_load(Path(path), label="P08 source universe")
    if value.get("schema") != UNIVERSE_SCHEMA or value.get("task_id") != TASK_ID:
        raise PanelPreparationError("P08 source universe schema or task ID changed")
    status = str(value.get("status", ""))
    if status not in {"PROPOSED_BEFORE_ENUMERATION", "FROZEN_SOURCE_UNIVERSE"}:
        raise PanelPreparationError("P08 source universe status is invalid")
    provenance = value.get("provenance")
    sources = value.get("candidate_source_universe")
    contract = value.get("panel_contract")
    if not isinstance(provenance, Mapping) or not isinstance(sources, Mapping) or not isinstance(contract, Mapping):
        raise PanelPreparationError("P08 source universe is missing binding sections")
    seed = provenance.get("selection_seed")
    ranges = {
        style: sources.get(style, {}).get("candidate_range_half_open")
        for style in STYLE_ORDER
        if isinstance(sources.get(style), Mapping)
    }
    if set(ranges) != set(STYLE_ORDER):
        raise PanelPreparationError("P08 source universe is missing candidate domains")
    _configure_p06(seed=int(seed), ranges=ranges)
    if int(contract.get("records_per_domain", -1)) != RECORDS_PER_DOMAIN:
        raise PanelPreparationError("P08 source universe panel count changed")
    if int(contract.get("clip_tokens_including_bos", -1)) != CLIP_TOKENS or int(contract.get("capture_sequence_tokens", -1)) != CAPTURE_TOKENS:
        raise PanelPreparationError("P08 source universe geometry changed")
    if list(contract.get("target_conditions", ())) != list(CONDITION_ORDER):
        raise PanelPreparationError("P08 source universe target order changed")
    if require_frozen and status != "FROZEN_SOURCE_UNIVERSE":
        raise PanelPreparationError("P08 source selection requires FROZEN_SOURCE_UNIVERSE")
    _reject_payload(value)
    return value


def freeze_source_universe(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise PanelPreparationError("freezing the source universe requires explicit --execute")
    input_path = Path(args.universe).expanduser().resolve()
    value = load_universe(input_path, require_frozen=False)
    if value.get("status") != "PROPOSED_BEFORE_ENUMERATION":
        raise PanelPreparationError("source universe is not a proposed metadata-only universe")
    exclusion_meta = value.get("exclusion_binding")
    if not isinstance(exclusion_meta, Mapping):
        raise PanelPreparationError("source universe has no exclusion binding")
    try:
        descriptor_paths = p06._frozen_descriptor_paths(exclusion_meta)
    except p06.PanelPreparationError as exc:
        raise PanelPreparationError(str(exc)) from exc
    frozen = dict(value)
    frozen["status"] = "FROZEN_SOURCE_UNIVERSE"
    frozen["frozen_utc"] = _utc_now()
    frozen["freeze"] = {
        "source_rows_enumerated": False,
        "panel_selected": False,
        "truth_opened": False,
        "descriptor_count_reverified": len(descriptor_paths),
        "code_commit": _git_commit(Path(value["provenance"]["repository_root"]).expanduser()),
    }
    output = _write_create_only(Path(args.output), frozen)
    return {"task_id": TASK_ID, "status": frozen["status"], "universe": output}


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
    datasets = {"pile": trusted._load_arrow_dataset(pile_paths), "finance": trusted._load_arrow_dataset(finance_paths)}
    selected, stats = p06._select_records(universe=universe, datasets=datasets, tokenizer=tokenizer)
    selection = {
        "schema": SELECTION_SCHEMA,
        "task_id": TASK_ID,
        "status": SELECTION_STATUS,
        "created_utc": _utc_now(),
        "selection_seed": int(universe["provenance"]["selection_seed"]),
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
        "source_ranges_half_open": dict(p06.CANDIDATE_RANGES),
        "selection_rule": {
            "order": "reused P06 deterministic_row_order over each declared range with the frozen P08 seed",
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

    universe = sub.add_parser("universe")
    universe.add_argument("--repository-root", type=Path, default=Path("."))
    universe.add_argument("--plan", type=Path, required=True)
    universe.add_argument("--selection-seed", type=int, required=True)
    universe.add_argument("--pile-range", type=int, nargs=2, required=True)
    universe.add_argument("--finance-range", type=int, nargs=2, required=True)
    universe.add_argument("--approved-opaque", type=Path, nargs="*", default=[])
    universe.add_argument("--output", type=Path, required=True)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--execute", action="store_true")
    freeze.add_argument("--universe", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    select = sub.add_parser("select")
    select.add_argument("--execute", action="store_true")
    select.add_argument("--repository-root", type=Path, default=Path("."))
    select.add_argument("--universe", type=Path, required=True)
    select.add_argument("--tokenizer", type=Path, required=True)
    select.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "universe":
            result = build_source_universe(args)
        elif args.command == "freeze":
            result = freeze_source_universe(args)
        else:
            result = select_panel(args)
    except (PanelPreparationError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"TRR-P08 panel preparation error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
