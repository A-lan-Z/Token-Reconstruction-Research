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
from scripts.trr_p08 import source_binding as p08_binding  # noqa: E402


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

# The P06 source selector's fresh 512-record panel is not part of the
# inherited P06 published catalog. Bind this exact metadata descriptor
# explicitly before any P08 source-universe construction.
PUBLISHED_P06_SELECTION_RELATIVE = Path(
    "experiments/TRR-P06/runtime/source-selection-r1/selection.json"
)
PUBLISHED_P06_SELECTION_SHA256 = (
    "d53ed8c972ec9ec00c6490dca22a99af833ea839fa68d9c4164ce061ee893a1a"
)
APPROVED_TRR0008_OPAQUE_PATH = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/"
    "TRR-0008/experiments/TRR-0008/selection/opaque_source_sequence_reservation.json"
)
APPROVED_TRR0008_OPAQUE_SHA256 = (
    "0487b9dda91d7eb791c93e1ba704afcea22abfc21f003cad8b99984f523357a4"
)
APPROVED_TRR0008_OPAQUE_SCHEMA = (
    "token-reconstruction.trr0008-opaque-source-sequence-reservation.v1"
)
APPROVED_TRR0008_OPAQUE_COUNTS = {
    "public_record_sha256": 1408,
    "final_sequence_sha256": 1408,
}
APPROVED_TRR0007_OPAQUE_PATH = Path(
    "/tmp/trr-p06/experiments/TRR-P06/setup/approved-trr0007/"
    "p06_opaque_source_sequence_reservation.json"
)
APPROVED_TRR0007_OPAQUE_SHA256 = (
    "09e845fec244a38873c5bf127f6d984af91af503fb42d3a8411451ce41cdedf4"
)
APPROVED_TRR0007_OPAQUE_SCHEMA = (
    "token-reconstruction.trr0007-opaque-source-sequence-reservation.v1"
)
APPROVED_TRR0007_OPAQUE_COUNTS = {
    "public_record_sha256": 256,
    "final_sequence_sha256": 256,
}
APPROVED_TRR0009_OPAQUE_PATH = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/"
    "TRR-0009/experiments/TRR-0009/selection/opaque_source_sequence_reservation.json"
)
APPROVED_TRR0009_OPAQUE_SHA256 = (
    "c0a1310c2dc8198ece3eda1eb273a3ef7412a1802ab85819ed709728bf603c52"
)
APPROVED_TRR0009_OPAQUE_SCHEMA = (
    "token-reconstruction.trr0009-opaque-source-sequence-reservation.v1"
)
APPROVED_TRR0009_OPAQUE_COUNTS = {
    "public_record_sha256": 384,
    "final_sequence_sha256": 384,
}
APPROVED_TRR0009_REPLACEMENT_OPAQUE_PATH = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/"
    "TRR-0009/experiments/TRR-0009/selection_v2/opaque_source_sequence_reservation.json"
)
APPROVED_TRR0009_REPLACEMENT_OPAQUE_SHA256 = (
    "73e07e1fd2c964eface6af387229bac738b96ecc5b7e1ee3059f57f9a8c2b4f8"
)
APPROVED_TRR0009_REPLACEMENT_OPAQUE_SCHEMA = (
    "token-reconstruction.trr0009-opaque-source-sequence-reservation.v1"
)
APPROVED_TRR0009_REPLACEMENT_OPAQUE_COUNTS = {
    "public_record_sha256": 384,
    "final_sequence_sha256": 384,
}

REQUIRED_OPAQUE_LEDGER_KEYS = (
    "approved_trr0007_opaque",
    "approved_trr0008_opaque",
    "approved_trr0009_opaque",
    "approved_trr0009_replacement_opaque",
)


APPROVED_OPAQUE_SPECS = {
    APPROVED_TRR0007_OPAQUE_PATH.resolve(): {
        "key": "approved_trr0007_opaque",
        "sha256": APPROVED_TRR0007_OPAQUE_SHA256,
        "schema": APPROVED_TRR0007_OPAQUE_SCHEMA,
        "counts": APPROVED_TRR0007_OPAQUE_COUNTS,
        "label": "approved TRR-0007 opaque reservation",
    },
    APPROVED_TRR0008_OPAQUE_PATH.resolve(): {
        "key": "approved_trr0008_opaque",
        "sha256": APPROVED_TRR0008_OPAQUE_SHA256,
        "schema": APPROVED_TRR0008_OPAQUE_SCHEMA,
        "counts": APPROVED_TRR0008_OPAQUE_COUNTS,
        "label": "approved TRR-0008 opaque reservation",
    },
    APPROVED_TRR0009_OPAQUE_PATH.resolve(): {
        "key": "approved_trr0009_opaque",
        "sha256": APPROVED_TRR0009_OPAQUE_SHA256,
        "schema": APPROVED_TRR0009_OPAQUE_SCHEMA,
        "counts": APPROVED_TRR0009_OPAQUE_COUNTS,
        "label": "approved TRR-0009 opaque reservation",
    },
    APPROVED_TRR0009_REPLACEMENT_OPAQUE_PATH.resolve(): {
        "key": "approved_trr0009_replacement_opaque",
        "sha256": APPROVED_TRR0009_REPLACEMENT_OPAQUE_SHA256,
        "schema": APPROVED_TRR0009_REPLACEMENT_OPAQUE_SCHEMA,
        "counts": APPROVED_TRR0009_REPLACEMENT_OPAQUE_COUNTS,
        "label": "approved TRR-0009 replacement opaque reservation",
    },
}


def _required_opaque_ledger_contract(bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact hash-only ledger set required for repaired P08 input."""

    ledgers: dict[str, dict[str, Any]] = {}
    for key in REQUIRED_OPAQUE_LEDGER_KEYS:
        entries = bindings.get(key)
        if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], Mapping):
            raise PanelPreparationError(f"required opaque ledger binding is missing: {key}")
        item = entries[0]
        ledgers[key] = {
            "path": str(item.get("path", "")),
            "bytes": int(item.get("bytes", -1)),
            "sha256": str(item.get("sha256", "")),
            "schema": str(item.get("schema", "")),
            "counts": dict(item.get("counts", {})),
        }
    canonical = json.dumps(ledgers, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "token-reconstruction.trr-p08-required-opaque-ledger-set.v1",
        "keys": list(REQUIRED_OPAQUE_LEDGER_KEYS),
        "ledgers": ledgers,
        "set_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _validate_required_opaque_ledger_contract(value: Mapping[str, Any]) -> None:
    exclusion = value.get("exclusion_binding")
    contract = exclusion.get("required_opaque_ledger_contract") if isinstance(exclusion, Mapping) else None
    if not isinstance(contract, Mapping):
        raise PanelPreparationError(
            "required opaque ledger contract is missing; superseded pre-repair universe cannot run"
        )
    if contract.get("schema") != "token-reconstruction.trr-p08-required-opaque-ledger-set.v1":
        raise PanelPreparationError("required opaque ledger contract schema changed")
    if list(contract.get("keys", ())) != list(REQUIRED_OPAQUE_LEDGER_KEYS):
        raise PanelPreparationError("required opaque ledger set is incomplete or reordered")
    bindings = {
        key: [{
            "path": spec.get("path"),
            "bytes": spec.get("bytes"),
            "sha256": spec.get("sha256"),
            "schema": spec.get("schema"),
            "counts": spec.get("counts"),
        }]
        for key, spec in (contract.get("ledgers") or {}).items()
        if isinstance(spec, Mapping)
    }
    expected = _required_opaque_ledger_contract(bindings)
    if expected != dict(contract):
        raise PanelPreparationError("required opaque ledger contract does not match exact approved ledgers")


class PanelPreparationError(RuntimeError):
    """Raised when a P08 source contract is not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return p06._sha256_file(Path(path))


def _verify_bound_descriptor(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        actual = _sha256_file(path)
    except (OSError, PanelPreparationError, p06.PanelPreparationError) as exc:
        raise PanelPreparationError(f"{label} is unavailable: {path}") from exc
    if actual != expected_sha256:
        raise PanelPreparationError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return {
        "label": label,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def _explicit_prior_exclusion_bindings(
    root: Path,
    approved_opaque_paths: Sequence[Path | str],
    p06_selection_path: Path | str | None = None,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Verify every required prior ledger before passing it to the collector.

    The exact set is deliberately fail-closed. P06's inherited catalog had a
    stale TRR-0007 descriptor; omitting that approved export from a repaired
    P08 universe would silently permit a reserved sequence into selection.
    """

    requested_selection = (
        PUBLISHED_P06_SELECTION_RELATIVE
        if p06_selection_path is None
        else Path(p06_selection_path).expanduser()
    )
    if not requested_selection.is_absolute():
        requested_selection = root / requested_selection
    expected_selection_path = (root / PUBLISHED_P06_SELECTION_RELATIVE).resolve()
    if requested_selection.resolve() != expected_selection_path:
        raise PanelPreparationError(
            "only the approved current P06 source-selection descriptor may be bound"
        )
    p06_selection = _verify_bound_descriptor(
        requested_selection,
        PUBLISHED_P06_SELECTION_SHA256,
        label="published P06 source selection",
    )
    opaque_paths = tuple(Path(raw).expanduser().resolve() for raw in approved_opaque_paths)
    expected_paths = frozenset(APPROVED_OPAQUE_SPECS)
    if len(opaque_paths) != len(expected_paths) or frozenset(opaque_paths) != expected_paths:
        missing = sorted(str(path) for path in expected_paths.difference(opaque_paths))
        unexpected = sorted(str(path) for path in frozenset(opaque_paths).difference(expected_paths))
        raise PanelPreparationError(
            "approved opaque ledger set is incomplete or unexpected; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    opaque_bindings: dict[str, list[dict[str, Any]]] = {}
    for candidate in opaque_paths:
        spec = APPROVED_OPAQUE_SPECS[candidate]
        binding = p08_binding.bind_opaque_reservation(
            candidate,
            spec["sha256"],
            expected_schema=spec["schema"],
            expected_counts=spec["counts"],
            label=spec["label"],
        )
        opaque_bindings[spec["key"]] = [binding]
    return (
        {
            "published_p06_selection": p06_selection,
            **opaque_bindings,
        },
        (Path(p06_selection["path"]), *opaque_paths),
    )


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
    explicit_bindings, opaque_and_metadata_paths = _explicit_prior_exclusion_bindings(
        root,
        tuple(args.approved_opaque or ()),
        args.p06_selection,
    )
    exclusions = p06.collect_exclusions(
        root,
        # The P06 catalog predates the P06 fresh 512-record panel. Keep the
        # current panel's identity metadata as an explicit required descriptor.
        metadata_paths=(opaque_and_metadata_paths[0],),
        approved_opaque_paths=opaque_and_metadata_paths[1:],
    )
    value = _universe_metadata(
        root=root,
        plan_binding=plan_binding,
        seed=int(args.selection_seed),
        ranges=ranges,
        exclusions=exclusions,
    )
    value["exclusion_binding"]["explicit_prior_bindings"] = explicit_bindings
    value["exclusion_binding"]["required_opaque_ledger_contract"] = _required_opaque_ledger_contract(explicit_bindings)
    value["exclusion_binding"]["p06_selection_explicitly_bound"] = True
    value["exclusion_binding"]["approved_opaque_nested_values_contract"] = (
        "For nested reservation exports, exclusion counts are derived from "
        "hashes.<field>.values; summary mapping keys are not hash entries."
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
    if require_frozen and status != "FROZEN_SOURCE_UNIVERSE":
        raise PanelPreparationError("P08 source selection requires FROZEN_SOURCE_UNIVERSE")
    if status == "FROZEN_SOURCE_UNIVERSE":
        _validate_required_opaque_ledger_contract(value)
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
    _validate_required_opaque_ledger_contract(value)
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
    universe.add_argument(
        "--p06-selection",
        type=Path,
        default=PUBLISHED_P06_SELECTION_RELATIVE,
        help="exact approved current P06 source-selection metadata descriptor",
    )
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
