#!/usr/bin/env python3
"""Select the fresh TRR-0007 public source panel after design freeze.

This task-local adapter reuses the trusted TRR-0005 renderer and the
TRR-0006 eligibility/opaque-hash classifier, while writing TRR-0007-owned
identity-only ledgers.  It requires a frozen TRR-0007 plan and a completed
count-only eligibility projection.  It never writes source text, token IDs,
labels, observations, or private truth.  Selection is create-only.
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

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_build_eligibility as eligibility
from scripts import trr0006_select_public as trr6_selector
from scripts import trr0007_eval_contract as contract
from scripts import trr0007_bank_ledger as bank_ledger
from token_reconstruction.trr0005_contract import STYLE_ORDER


SELECTION_SCHEMA = "token-reconstruction.trr0007-source-selection.v1"
EXCLUSION_SCHEMA = "token-reconstruction.trr0007-source-exclusions.v1"
SELECTION_STATUS = "FROZEN_TRR0007_SOURCE_SELECTION_NO_TRUTH"
TASK_ROOT_RELATIVE = Path("experiments/TRR-0007")
RECORDS_PER_DOMAIN = contract.RECORDS_PER_DOMAIN
SELECTION_SEED = 5005


class SelectionError(contract.ContractError):
    """Raised when a fresh source panel cannot be selected safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SelectionError(f"file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_record(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SelectionError(f"{description} is unavailable: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _write_create_only(path: Path, value: Mapping[str, Any], *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise SelectionError(f"{description} is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return _file_record(path, description=description)


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise SelectionError(f"repository root is unavailable: {root}")
    return root


def _task_path(value: Path | str, *, root: Path, description: str) -> Path:
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    task_root = (root / TASK_ROOT_RELATIVE).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise SelectionError(f"{description} must be below {task_root}: {path}") from exc
    if path.is_symlink():
        raise SelectionError(f"{description} is a symlink: {path}")
    return path


def _load_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = contract.load_json(path, description="TRR-0007 evaluation plan")
    contract.validate_plan(plan)
    if plan.get("status") != "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION":
        raise SelectionError("source selection requires a frozen TRR-0007 evaluation plan")
    # The immutable plan deliberately carries no execution progress.  The
    # create-only selection and capture receipts below are the execution
    # status authority.
    return plan, _file_record(path, description="TRR-0007 evaluation plan")


def _load_method_freeze(
    plan: Mapping[str, Any],
    path: Path,
    *,
    root: Path,
    explicit_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    record, payload, states = contract.load_method_freeze(
        path, repository_root=root, verify_assets=True
    )
    expected = plan.get("method_freeze")
    if isinstance(expected, Mapping) and expected.get("sha256") != record["sha256"]:
        raise SelectionError("method freeze differs from the frozen design binding")
    if explicit_sha256 is not None and explicit_sha256 != record["sha256"]:
        raise SelectionError("explicit method-freeze digest differs from the ledger")
    return record, payload, states


def _validate_inventory(
    path: Path,
    *,
    final_bank: Mapping[str, Any] | None = None,
    prefix_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inventory = contract.load_json(path, description="TRR-0006 eligibility inventory")
    if inventory.get("schema") != "token-reconstruction.trr0006-eligibility-inventory.v1":
        raise SelectionError("eligibility inventory schema changed")
    if inventory.get("task_id") != "TRR-0006" or inventory.get("status") != "ELIGIBILITY_INVENTORY_COMPLETE_NO_SELECTION_NO_TRUTH":
        raise SelectionError("eligibility inventory is not a closed count-only projection")
    if inventory.get("selection_status") != "NOT_STARTED":
        raise SelectionError("eligibility inventory records selection as started")
    source_contract = inventory.get("source_contract")
    expected_ranges = {
        style: [
            int(eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
            int(eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
        ]
        for style in STYLE_ORDER
    }
    if not isinstance(source_contract, Mapping) or source_contract.get("selection_seed") != SELECTION_SEED or source_contract.get("source_ranges_half_open") != expected_ranges:
        raise SelectionError("eligibility inventory source contract changed")
    domains = inventory.get("domains")
    if not isinstance(domains, Mapping):
        raise SelectionError("eligibility inventory domains are absent")
    for style in STYLE_ORDER:
        row = domains.get(style)
        if not isinstance(row, Mapping) or int(row.get("eligible_unique", -1)) < RECORDS_PER_DOMAIN:
            raise SelectionError(f"eligibility inventory has insufficient {style} capacity")
        if row.get("source_range_half_open") != expected_ranges[style]:
            raise SelectionError(f"eligibility inventory {style} range changed")
    if final_bank is not None and inventory.get("final_bank_ledgers") != dict(final_bank):
        raise SelectionError("eligibility inventory is not bound to the reviewed final v5 bank ledgers")
    if prefix_ledger is not None and inventory.get("public_fitting_prefix_exclusions") != dict(prefix_ledger):
        raise SelectionError("eligibility inventory is not bound to the reviewed v3 fitting-prefix ledger")
    for key in ("truth_created", "truth_opened", "target_loaded", "target_labels_loaded", "private_or_truth_payload_read"):
        if inventory.get(key) is True:
            raise SelectionError(f"eligibility inventory records forbidden access: {key}")
    return inventory


def _known_exclusion_paths(root: Path) -> list[Path]:
    """Bind all accessible prior public fit/validation/evaluation ledgers."""

    paths = list(eligibility._known_exclusion_paths(root))
    # TRR-0006 introduced additional panels after the TRR-0005 default list.
    # Keep this explicit so a future private truth file cannot be swept in by
    # a broad recursive glob.
    paths.extend(
        root / relative
        for relative in (
            "experiments/TRR-0006/source_selection.json",
            "experiments/TRR-0006/duplicate_capture_exclusion.json",
            "experiments/TRR-0006/panel_capture_v1/panel.json",
            "experiments/TRR-0006/panel_capture_v1/observations.json",
            "experiments/TRR-0006/public_observations_v1/panel.json",
            "experiments/TRR-0006/public_observations_v1/observations.json",
        )
    )
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if str(resolved) not in seen:
            seen.add(str(resolved))
            result.append(resolved)
    return result


def _resolve_p04(path: Path | str, *, root: Path) -> Path:
    raw = Path(path).expanduser()
    return (raw if raw.is_absolute() else root / raw).resolve()


def _select_rows(
    *,
    datasets: Mapping[str, Any],
    tokenizer: Any,
    exclusions: Any,
    opaque: Any,
) -> tuple[dict[str, list[Any]], dict[str, dict[str, int]]]:
    selected: dict[str, list[Any]] = {}
    skipped: dict[str, dict[str, int]] = {}
    seen_public_hashes: set[str] = set()
    seen_final_sequences: set[str] = set()
    for style in STYLE_ORDER:
        rows, diagnostics = trr6_selector._select_domain(
            datasets[style],
            style=style,
            tokenizer=tokenizer,
            records_per_domain=RECORDS_PER_DOMAIN,
            exclusions=exclusions,
            opaque=opaque,
            seen_public_hashes=seen_public_hashes,
            seen_final_sequences=seen_final_sequences,
        )
        selected[style] = rows
        skipped[style] = diagnostics
    return selected, skipped


def _source_ranges() -> dict[str, list[int]]:
    return {
        style: [
            int(eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
            int(eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
        ]
        for style in STYLE_ORDER
    }


def _selection_metadata(rows: Mapping[str, Sequence[Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]], dict[str, list[str]]]:
    metadata = {style: [trr6_selector._record_metadata(row) for row in rows[style]] for style in STYLE_ORDER}
    ids = {style: [str(row["record_id"]) for row in metadata[style]] for style in STYLE_ORDER}
    sequences = {style: [str(row["final_sequence_sha256"]) for row in metadata[style]] for style in STYLE_ORDER}
    return metadata, ids, sequences


def select_public(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args.repository_root)
    try:
        final_bank = bank_ledger.load_final_bank_ledgers(
            repository_root=root,
            exclusion_manifest=args.final_bank_exclusion_manifest,
            selected_parent_rows=args.final_bank_parent_ledger,
            corpus_plan=args.final_bank_corpus_plan,
        )
        prefix_ledger = bank_ledger.load_prefix_exclusion_ledger(
            repository_root=root, path=args.public_fitting_prefix_exclusions
        )
    except bank_ledger.BankLedgerError as exc:
        raise SelectionError(str(exc)) from exc
    plan_path = Path(args.plan).expanduser().resolve()
    plan, plan_record = _load_plan(plan_path)
    inventory_path = Path(args.eligibility_inventory).expanduser().resolve()
    inventory = _validate_inventory(
        inventory_path, final_bank=final_bank, prefix_ledger=prefix_ledger
    )
    inventory_record = _file_record(inventory_path, description="eligibility inventory")
    method_freeze_record, method_freeze, method_states = _load_method_freeze(
        plan,
        Path(args.method_freeze).expanduser().resolve(),
        root=root,
        explicit_sha256=args.method_freeze_sha256,
    )
    selection_path = _task_path(args.output, root=root, description="source selection output")
    exclusions_path = _task_path(args.exclusions_output, root=root, description="exclusion output")
    if selection_path == exclusions_path:
        raise SelectionError("selection and exclusion outputs must differ")
    p04_path = _resolve_p04(args.p04_exchange, root=root)
    if p04_path.is_symlink() or not p04_path.is_file():
        raise SelectionError(f"approved P04 opaque exchange is unavailable: {p04_path}")
    # Verify the approved exchange before reading any row.  The helper returns
    # only opaque values for classification and aggregate summaries for output.
    opaque = eligibility._load_p04_opaque_exclusions(p04_path)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    pile_paths = tuple(Path(path).expanduser().resolve() for path in args.pile_arrow)
    finance_paths = tuple(Path(path).expanduser().resolve() for path in args.finance_arrow)
    tokenizer = trusted._load_tokenizer(tokenizer_path)
    datasets = {
        "pile": trusted._load_arrow_dataset(pile_paths),
        "finance": trusted._load_arrow_dataset(finance_paths),
    }
    exclusion_paths = _known_exclusion_paths(root)
    exclusion_paths.extend(Path(path).expanduser().resolve() for path in args.exclude_source)
    exclusion_paths.extend(
        Path(final_bank["files"][key]["path"])
        for key in ("exclusion_manifest", "selected_parent_rows", "corpus_plan")
    )
    exclusion_paths.append(Path(prefix_ledger["file"]["path"]))
    exclusions = trusted._collect_exclusions(exclusion_paths)
    rows, skipped = _select_rows(
        datasets=datasets,
        tokenizer=tokenizer,
        exclusions=exclusions,
        opaque=opaque,
    )
    metadata, ids, sequences = _selection_metadata(rows)
    record_id_hashes = {style: _canonical_digest(ids[style]) for style in STYLE_ORDER}
    sequence_hashes = {style: _canonical_digest(sequences[style]) for style in STYLE_ORDER}
    ended = _utc_now()
    p04_descriptor = dict(opaque.exchange)
    exclusion_sources = [
        {
            key: source[key]
            for key in ("path", "available", "bytes", "sha256", "new_identity_count")
            if key in source
        }
        for source in exclusions.sources
    ]
    exclusion_payload: dict[str, Any] = {
        "schema": EXCLUSION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "PUBLIC_IDENTITY_EXCLUSIONS_COMPLETE_NO_TRUTH",
        "sources": exclusion_sources,
        "identity_counts": {
            style: {
                "ids": len(exclusions.ids[style]),
                "hashes": len(exclusions.hashes[style]),
                "indices": len(exclusions.indices[style]),
            }
            for style in STYLE_ORDER
        },
        "p04_exchange": p04_descriptor,
        "p04_field_summaries": opaque.fields,
        "final_bank_ledgers": final_bank,
        "public_fitting_prefix_exclusions": prefix_ledger,
        "known_prior_panels": [
            "TRR-0003 fit/validation identities",
            "TRR-0004 fit/validation identities",
            "TRR-0005 fit/validation/opened-evaluation identities",
            "TRR-0006 selected 3072-source and opened-evaluation identities",
        ],
        "source_text_or_token_ids_written": False,
        "private_or_truth_payload_read": False,
        "truth_opened": False,
    }
    exclusion_record = _write_create_only(exclusions_path, exclusion_payload, description="exclusion output")
    selection_payload: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": SELECTION_STATUS,
        "plan": plan_record,
        "method_freeze": method_freeze_record,
        "method_freeze_sha256": method_freeze_record["sha256"],
        "method_freeze_state_sha256": {
            method_id: state["sha256"] for method_id, state in method_states.items()
        },
        "eligibility_inventory": inventory_record,
        "final_bank_ledgers": final_bank,
        "public_fitting_prefix_exclusions": prefix_ledger,
        "selection_seed": SELECTION_SEED,
        "records_per_domain": RECORDS_PER_DOMAIN,
        "source_ranges_half_open": _source_ranges(),
        "sequence_tokens_including_bos": contract.STORED_SEQUENCE_TOKENS,
        "capture_sequence_tokens": contract.CAPTURE_SEQUENCE_TOKENS,
        "capture_batch_records": contract.CAPTURE_BATCH_RECORDS,
        "target_conditions": list(contract.TARGET_ORDER),
        "paired_conditions": True,
        "public_sources_frozen": {
            "pile": trusted._dataset_descriptor(pile_paths, style="pile"),
            "finance": trusted._dataset_descriptor(finance_paths, style="finance"),
            "tokenizer": trusted._tokenizer_descriptor(tokenizer_path),
        },
        "selection_rule": {
            "algorithm": "Deterministic TRR-0006 future-holdout order with seed 5005; reject all known public fit/validation/opened-evaluation identities, TRR-0006 prior panel identities, approved P04 opaque hashes, invalid rows, duplicate rendered sources, and duplicate final 128-token sequences; retain the first 128 eligible rows per domain.",
            "identity_exclusions": True,
            "opaque_p04_hashes_applied": True,
            "records": metadata,
            "record_ids_sha256": record_id_hashes,
            "final_sequence_sha256": sequence_hashes,
            "source_text_or_token_ids_written": False,
        },
        "selection_exclusions": {
            "path": str(exclusions_path),
            "bytes": exclusion_record["bytes"],
            "sha256": exclusion_record["sha256"],
            "p04_exchange": p04_descriptor,
            "p04_field_summaries": opaque.fields,
            "identity_counts": exclusion_payload["identity_counts"],
            "targetfit_per_record_metadata_available": False,
        },
        "selection_diagnostics": {
            style: {
                **skipped[style],
                "selected": len(rows[style]),
                "pool_size": _source_ranges()[style][1] - _source_ranges()[style][0],
            }
            for style in STYLE_ORDER
        },
        "execution": {
            "started_utc": _utc_now(),
            "ended_utc": ended,
            "command": list(sys.argv),
            "code_commit": _git_commit(root),
            "selector_source": _file_record(Path(__file__), description="TRR-0007 selector source"),
            "trusted_trr0005_producer_source": _file_record(Path(trusted.__file__), description="trusted TRR-0005 producer source"),
            "trusted_trr0006_selector_source": _file_record(Path(trr6_selector.__file__), description="trusted TRR-0006 selector source"),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "model_loaded": False,
            "target_loaded": False,
            "truth_created_or_opened": False,
            "source_text_written": False,
            "token_ids_written": False,
            "network_used": False,
        },
        "source_text_or_target_labels": False,
        "truth_opened": False,
        "truth_created": False,
    }
    selection_record = _write_create_only(selection_path, selection_payload, description="source selection output")
    return {
        "task_id": contract.TASK_ID,
        "status": SELECTION_STATUS,
        "selection": selection_record,
        "exclusions": exclusion_record,
        "records_per_domain": RECORDS_PER_DOMAIN,
        "truth_created_or_opened": False,
    }


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--eligibility-inventory", type=Path, required=True)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--method-freeze-sha256")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    parser.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    parser.add_argument("--final-bank-exclusion-manifest", type=Path, required=True)
    parser.add_argument("--final-bank-parent-ledger", type=Path, required=True)
    parser.add_argument("--final-bank-corpus-plan", type=Path, required=True)
    parser.add_argument("--public-fitting-prefix-exclusions", type=Path, required=True)
    parser.add_argument("--exclude-source", type=Path, nargs="*", default=[])
    parser.add_argument("--p04-exchange", type=Path, default=Path("experiments/TRR-0006/coordination/p04_reservation_hashes.json"))
    parser.add_argument("--exclusions-output", type=Path, default=Path("experiments/TRR-0007/source_exclusions.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/TRR-0007/source_selection.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = select_public(args)
    except (SelectionError, contract.ContractError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-0007 source selection failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
