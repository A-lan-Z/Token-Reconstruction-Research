#!/usr/bin/env python3
"""Select the prospective TRR-0008 public panel after owner freeze.

This adapter is deliberately separate from the count-only TRR-0008 inventory.
It consumes the reviewed decision contract, the TRR-0007 method freeze, the
identity-only inventory, and the approved opaque exclusions before it reads
public Arrow rows.  It reuses the trusted TRR-0005 renderer, TRR-0006 identity
classifier conventions, TRR-0007 exclusion ledger, and deterministic natural
row order.  Selection is create-only and fail-closed: the command refuses a
draft decision contract or an unverified P06 source-hash byte convention.

The selection ledger contains only public identity metadata and hashes.  The
separate ``reserve`` command exports hash-only reservations with no record IDs,
indices, domain labels, source text, token IDs, targets, or truth.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

# Keep ``python3 scripts/trr0008_select_public.py ...`` runnable from the
# repository root without requiring a caller-specific PYTHONPATH.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_build_eligibility as eligibility
from scripts import trr0007_eval_contract as trr7_contract
from scripts import trr0007_eval_select as trr7_selector
from scripts import trr0008_plan as planning
from token_reconstruction.trr0005_contract import STYLE_ORDER
from token_reconstruction.trr0005_public_corpus import (
    SOURCE_PARTITIONS,
    deterministic_row_order,
    source_record_id,
)


TASK_ID = "TRR-0008"
SELECTION_SCHEMA = "token-reconstruction.trr0008-source-selection.v1"
EXCLUSION_SCHEMA = "token-reconstruction.trr0008-source-exclusions.v1"
RESERVATION_SCHEMA = "token-reconstruction.trr0008-opaque-source-sequence-reservation.v1"
SELECTION_STATUS = "FROZEN_TRR0008_SOURCE_SELECTION_NO_TRUTH"
EXCLUSION_STATUS = "PUBLIC_IDENTITY_EXCLUSIONS_COMPLETE_NO_TRUTH"
RESERVATION_STATUS = "READY_FOR_TRR0008_CAPTURE_HASH_ONLY"
SELECTION_SEED = 5005
SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
EXPECTED_RECORDS_BY_DOMAIN = {"pile": 384, "finance": 1024}
SOURCE_RANGES = {"pile": [7000, 10000], "finance": [12000, 20000]}
TARGET_CONDITIONS = ("public_base", "public_lora_2601")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SelectionError(RuntimeError):
    """Raised when a TRR-0008 selection or reservation cannot proceed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SelectionError(f"file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _newline_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(values) + "\n").encode("ascii"))


def _file_record(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SelectionError(f"{description} is unavailable: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SelectionError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise SelectionError(f"{description} must be a JSON object")
    return dict(value)


def _write_create_only(path: Path, value: Mapping[str, Any], *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise SelectionError(f"{description} is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:  # pragma: no cover - race-safe guard
        raise SelectionError(f"{description} is create-only and already exists: {path}") from exc
    return _file_record(path, description=description)


def _task_path(value: Path | str, *, root: Path, description: str) -> Path:
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    task_root = (root / "experiments" / "TRR-0008" / "selection").resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise SelectionError(f"{description} must be below {task_root}: {path}") from exc
    if path.is_symlink():
        raise SelectionError(f"{description} is a symlink: {path}")
    return path


def _require_hex(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise SelectionError(f"{description} must be a lowercase SHA-256")
    return value


def _validate_planning_status(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require an explicit P06 source-byte confirmation before real selection."""

    record = _file_record(path, description="TRR-0008 planning status")
    payload = _load_json(path, description="TRR-0008 planning status")
    if payload.get("task_id") != TASK_ID:
        raise SelectionError("planning status task identity changed")
    compatibility = payload.get("p06_hash_compatibility")
    if not isinstance(compatibility, Mapping):
        raise SelectionError("planning status lacks P06 hash compatibility evidence")
    status = compatibility.get("source_hash_byte_input_status")
    if status != "VERIFIED_P06_PRODUCER_CONFIRMATION":
        raise SelectionError(
            "P06 source-hash byte input is not explicitly verified; selection remains blocked "
            f"until planning_status.p06_hash_compatibility.source_hash_byte_input_status="
            "VERIFIED_P06_PRODUCER_CONFIRMATION (currently "
            f"{status!r})"
        )
    if compatibility.get("sequence_hash_algorithm", "").startswith("VERIFIED:") is not True:
        raise SelectionError("P06 H128 sequence convention is not verified in planning status")
    return record, dict(compatibility)


def _validate_decision_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path, description="TRR-0008 decision contract")
    payload = _load_json(path, description="TRR-0008 decision contract")
    if payload.get("schema") != "token-reconstruction.trr0008-decision-contract.v1":
        raise SelectionError("decision contract schema changed")
    if payload.get("task_id") != TASK_ID:
        raise SelectionError("decision contract task identity changed")
    if payload.get("status") != "FROZEN_DECISION_CONTRACT_BEFORE_SOURCE_SELECTION":
        raise SelectionError("source selection requires a frozen decision contract")
    methods = payload.get("methods")
    expected_methods = {
        "candidate": "improved_public_bank__residual_mlp512",
        "reference": "trr6__enriched_trained_diagonal_attention128",
        "credible_alternative": "current_enriched__residual_mlp512",
        "diagnostic": "improved_public_bank__trained_diagonal",
    }
    if not isinstance(methods, Mapping) or any(methods.get(k) != v for k, v in expected_methods.items()):
        raise SelectionError("decision contract method roles changed")
    panel = payload.get("panel")
    if not isinstance(panel, Mapping):
        raise SelectionError("decision contract panel is absent")
    if panel.get("finance_records_per_domain") != EXPECTED_RECORDS_BY_DOMAIN["finance"]:
        raise SelectionError("decision contract Finance count changed")
    if panel.get("pile_records_per_domain") != EXPECTED_RECORDS_BY_DOMAIN["pile"]:
        raise SelectionError("decision contract Pile count changed")
    if panel.get("natural_ranges_half_open") != SOURCE_RANGES:
        raise SelectionError("decision contract natural ranges changed")
    if panel.get("selection_seed") != SELECTION_SEED:
        raise SelectionError("decision contract selection seed changed")
    if panel.get("clip_tokens_including_bos") != SEQUENCE_TOKENS:
        raise SelectionError("decision contract stored sequence length changed")
    if panel.get("scored_post_bos_tokens") != SCORED_POST_BOS_TOKENS:
        raise SelectionError("decision contract scored length changed")
    if tuple(panel.get("target_conditions", ())) != TARGET_CONDITIONS:
        raise SelectionError("decision contract target pairing changed")
    primary = payload.get("primary")
    if not isinstance(primary, Mapping) or primary.get("cell") != "finance__public_base":
        raise SelectionError("decision contract primary cell changed")
    if primary.get("route_alpha") != 0.025 or primary.get("component_alpha") != 0.0125:
        raise SelectionError("decision contract primary confidence allocation changed")
    if primary.get("practical_margin") != 0.05:
        raise SelectionError("decision contract primary practical margin changed")
    token_endpoint = payload.get("token_endpoint")
    if (
        not isinstance(token_endpoint, Mapping)
        or token_endpoint.get("route_alpha") != 0.025
        or token_endpoint.get("practical_margin") != 0.01
    ):
        raise SelectionError("decision contract token route changed")
    safeguards = payload.get("safeguards")
    expected_cells = {
        "finance__public_base",
        "finance__public_lora_2601",
        "pile__public_base",
        "pile__public_lora_2601",
    }
    if (
        not isinstance(safeguards, Mapping)
        or safeguards.get("route_alpha") != 0.05
        or safeguards.get("exact_harm_margin") != 0.05
        or safeguards.get("token_harm_margin") != 0.01
        or set(safeguards.get("cells", ())) != expected_cells
    ):
        raise SelectionError("decision contract safeguard gates changed")
    bootstrap = payload.get("bootstrap")
    if (
        not isinstance(bootstrap, Mapping)
        or bootstrap.get("seed") != 8008
        or bootstrap.get("draws") != 10000
        or bootstrap.get("unit") != "source_record"
    ):
        raise SelectionError("decision contract bootstrap binding changed")
    cost_gate = payload.get("cost_gate")
    if (
        not isinstance(cost_gate, Mapping)
        or cost_gate.get("threshold") != 1.25
        or cost_gate.get("primary_cell") != "finance__public_base"
        or cost_gate.get("all_cells_required") is not True
        or set(cost_gate.get("cells", ())) != expected_cells
    ):
        raise SelectionError("decision contract cost gate changed")
    timing_receipt = cost_gate.get("timing_receipt")
    if (
        not isinstance(timing_receipt, Mapping)
        or timing_receipt.get("path") != "experiments/TRR-0008/timing/precision40_result.json"
        or timing_receipt.get("bytes") != 4076989
        or timing_receipt.get("sha256") != "a5d923bb9254f0ba0ec917dc6ede9e22d7b566e47e79408cf188f679c6b30c02"
        or timing_receipt.get("schema") != "token-reconstruction.trr0008-balanced-timing.v1"
        or timing_receipt.get("status") != "TIMING_COMPLETE"
        or timing_receipt.get("qualification") != "PASS"
        or timing_receipt.get("truth_opened") is not False
    ):
        raise SelectionError("decision contract timing receipt binding changed")
    timing_path = (_REPOSITORY_ROOT / str(timing_receipt["path"])).resolve()
    timing_file = _file_record(timing_path, description="bound TRR-0008 timing receipt")
    if timing_file["bytes"] != timing_receipt["bytes"] or timing_file["sha256"] != timing_receipt["sha256"]:
        raise SelectionError("bound TRR-0008 timing receipt changed")
    if payload.get("truth_and_freeze", {}).get("source_selection") not in {
        "not started by this planning artifact",
        "source selection starts only after this freeze",
        "selection authorized after this freeze",
    }:
        raise SelectionError("decision contract source-selection gate is malformed")
    for key in ("p06_underlying_provenance_opened",):
        if payload.get("provenance", {}).get(key) is True:
            raise SelectionError(f"decision contract records forbidden P06 access: {key}")
    return record, payload


def _validate_method_freeze(
    path: Path, *, root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        record, payload, states = trr7_contract.load_method_freeze(
            path, repository_root=root, verify_assets=True
        )
    except trr7_contract.ContractError as exc:
        raise SelectionError(str(exc)) from exc
    required = {
        "improved_public_bank__residual_mlp512",
        "current_enriched__residual_mlp512",
        "improved_public_bank__trained_diagonal",
    }
    if not required.issubset(set(payload.get("method_ids", ()) )):
        raise SelectionError("method freeze lacks a required TRR-0008 method state")
    retained = payload.get("retained_reference")
    if not isinstance(retained, Mapping):
        raise SelectionError("method freeze lacks retained reference binding")
    _require_hex(retained.get("sha256"), description="retained reference state")
    return record, payload, states


def _validate_inventory(
    path: Path, *, expected_counts: Mapping[str, int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path, description="TRR-0008 identity inventory")
    payload = _load_json(path, description="TRR-0008 identity inventory")
    if payload.get("schema") != "token-reconstruction.trr0008-identity-inventory.v1":
        raise SelectionError("identity inventory schema changed")
    if payload.get("task_id") != TASK_ID or payload.get("status") != "IDENTITY_INVENTORY_COMPLETE_NO_SELECTION_NO_TRUTH":
        raise SelectionError("identity inventory is not a closed count-only projection")
    if payload.get("sample_size_status") != "PROPOSED_COUNT_CHECKED_NOT_SELECTED":
        raise SelectionError("identity inventory records a changed selection status")
    if payload.get("requested_per_domain") != dict(expected_counts):
        raise SelectionError("identity inventory requested counts differ from the frozen proposal")
    source_contract = payload.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise SelectionError("identity inventory source contract is absent")
    if source_contract.get("selection_seed") != SELECTION_SEED:
        raise SelectionError("identity inventory selection seed changed")
    if source_contract.get("source_ranges_half_open") != SOURCE_RANGES:
        raise SelectionError("identity inventory source ranges changed")
    if source_contract.get("sequence_tokens_including_bos") != SEQUENCE_TOKENS or source_contract.get("scoring_post_bos_tokens") != SCORED_POST_BOS_TOKENS:
        raise SelectionError("identity inventory sequence geometry changed")
    if source_contract.get("capture_batch_records") != CAPTURE_BATCH_RECORDS or source_contract.get("capture_sequence_tokens") != CAPTURE_SEQUENCE_TOKENS:
        raise SelectionError("identity inventory capture geometry changed")
    if source_contract.get("natural_distribution_preserved") is not True:
        raise SelectionError("identity inventory does not certify natural distribution")
    domains = payload.get("domains")
    if not isinstance(domains, Mapping):
        raise SelectionError("identity inventory domains are absent")
    for style in STYLE_ORDER:
        row = domains.get(style)
        if not isinstance(row, Mapping):
            raise SelectionError(f"identity inventory lacks {style}")
        if row.get("source_range_half_open") != SOURCE_RANGES[style]:
            raise SelectionError(f"identity inventory {style} range changed")
        if int(row.get("eligible_unique", -1)) < int(expected_counts[style]):
            raise SelectionError(f"identity inventory has insufficient {style} capacity")
        capacity = row.get("capacity_for_requested_per_domain")
        if not isinstance(capacity, Mapping) or capacity.get("requested") != expected_counts[style] or capacity.get("sufficient") is not True:
            raise SelectionError(f"identity inventory {style} capacity binding changed")
    for key in ("selection_performed", "truth_created_or_opened", "model_loaded", "source_text_written", "token_ids_written"):
        if payload.get("execution", {}).get(key) is True or payload.get("exclusion_policy", {}).get(key) is True:
            raise SelectionError(f"identity inventory records forbidden state: {key}")
    p06_summary = payload.get("p06_opaque_reservation")
    if not isinstance(p06_summary, Mapping):
        raise SelectionError("identity inventory lacks P06 opaque reservation binding")
    p06_file = p06_summary.get("file")
    if not isinstance(p06_file, Mapping) or p06_file.get("sha256") != planning.P06_OPAQUE_SHA256 or p06_file.get("bytes") != planning.P06_OPAQUE_BYTES:
        raise SelectionError("identity inventory P06 opaque reservation binding changed")
    return record, payload


def _validate_trr7_file_bindings(inventory: Mapping[str, Any], *, root: Path) -> None:
    binding = inventory.get("trr0007_inventory")
    if not isinstance(binding, Mapping):
        raise SelectionError("identity inventory lacks embedded TRR-0007 binding")
    file_record = binding.get("file")
    if isinstance(file_record, Mapping):
        actual = _file_record(Path(str(file_record.get("path", ""))), description="TRR-0007 eligibility inventory")
        if actual != {"path": str(Path(str(file_record["path"])).expanduser().resolve()), "bytes": file_record.get("bytes"), "sha256": file_record.get("sha256") }:
            raise SelectionError("TRR-0007 eligibility inventory binding changed")
    final_bank = binding.get("final_bank_ledgers")
    if not isinstance(final_bank, Mapping) or final_bank.get("status") != "METADATA_BANK_READY_FOR_REAL_P0_CAPTURE":
        raise SelectionError("TRR-0007 final bank ledger binding is absent")
    files = final_bank.get("files")
    if not isinstance(files, Mapping):
        raise SelectionError("TRR-0007 final bank ledger files are absent")
    for key in ("exclusion_manifest", "selected_parent_rows", "corpus_plan"):
        descriptor = files.get(key)
        if not isinstance(descriptor, Mapping):
            raise SelectionError(f"TRR-0007 final bank {key} binding is absent")
        actual = _file_record(Path(str(descriptor.get("path", ""))), description=f"TRR-0007 final bank {key}")
        if actual["bytes"] != descriptor.get("bytes") or actual["sha256"] != descriptor.get("sha256"):
            raise SelectionError(f"TRR-0007 final bank {key} binding changed")
    prefix = binding.get("prefix_exclusions")
    if not isinstance(prefix, Mapping) or not isinstance(prefix.get("file"), Mapping):
        raise SelectionError("TRR-0007 fitting-prefix binding is absent")
    descriptor = prefix["file"]
    actual = _file_record(Path(str(descriptor.get("path", ""))), description="TRR-0007 fitting-prefix exclusions")
    if actual["bytes"] != descriptor.get("bytes") or actual["sha256"] != descriptor.get("sha256"):
        raise SelectionError("TRR-0007 fitting-prefix binding changed")


def _validate_public_inputs(
    inventory: Mapping[str, Any], *, pile_paths: Sequence[Path], finance_paths: Sequence[Path], tokenizer_path: Path
) -> dict[str, Any]:
    expected = inventory.get("public_inputs")
    if not isinstance(expected, Mapping):
        raise SelectionError("identity inventory public input bindings are absent")
    actual = {
        "pile_arrow": [_file_record(path, description="Pile Arrow input") for path in pile_paths],
        "finance_arrow": [_file_record(path, description="Finance Arrow input") for path in finance_paths],
        "tokenizer": {"path": str(Path(tokenizer_path).expanduser().resolve())},
    }
    for key in ("pile_arrow", "finance_arrow"):
        if list(expected.get(key, ())) != actual[key]:
            raise SelectionError(f"{key} input differs from the identity inventory")
    expected_tokenizer = expected.get("tokenizer")
    if not isinstance(expected_tokenizer, Mapping) or expected_tokenizer.get("path") != actual["tokenizer"]["path"]:
        raise SelectionError("tokenizer input differs from the identity inventory")
    return actual


def _known_exclusion_paths(root: Path) -> list[Path]:
    """Use the same explicit TRR-0007 public identity ledgers as inventory."""

    paths = list(trr7_selector._known_exclusion_paths(root))
    paths.extend(
        root / relative
        for relative in (
            planning.TRR7_EXCLUSIONS,
            planning.TRR7_SELECTION,
            planning.TRR7_FINAL_BANK,
            planning.TRR7_PARENT_ROWS,
            planning.TRR7_CORPUS_PLAN,
            planning.TRR7_PREFIX_EXCLUSIONS,
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


def _p04_opaque() -> tuple[eligibility.OpaqueExclusions, dict[str, Any]]:
    path = planning.P04_OPAQUE.expanduser().resolve()
    descriptor = _file_record(path, description="approved P04 opaque exchange")
    if descriptor["sha256"] != planning.P04_OPAQUE_SHA256:
        raise SelectionError("approved P04 opaque exchange hash changed")
    try:
        opaque = eligibility._load_p04_opaque_exclusions(path)
    except eligibility.EligibilityError as exc:
        raise SelectionError(str(exc)) from exc
    return opaque, descriptor


def _p06_opaque() -> tuple[dict[str, Any], frozenset[str], frozenset[str]]:
    try:
        summary, source, sequence = planning._load_p06_opaque(planning.P06_OPAQUE)
    except planning.PlanError as exc:
        raise SelectionError(str(exc)) from exc
    summary = dict(summary)
    summary["authorized_original_path"] = str(planning.P06_OPAQUE_ORIGINAL)
    summary["task_owned_copy"] = True
    return summary, source, sequence


def _p06_sequence_digest(token_ids: Sequence[int]) -> str:
    # Keep the approved P06 H128 convention explicit and independent from P04's
    # H129 matcher.  Selection never serializes these token IDs.
    import struct

    values = [int(value) for value in token_ids[:SEQUENCE_TOKENS]]
    if len(values) < SEQUENCE_TOKENS:
        raise SelectionError("candidate has fewer than 128 tokens for P06 H128 matching")
    return _sha256_bytes(struct.pack("<" + "i" * SEQUENCE_TOKENS, *values))


def _classify_candidate(
    candidate: Any,
    *,
    exclusions: Any,
    p04: eligibility.OpaqueExclusions,
    p06_source: frozenset[str],
    p06_sequence: frozenset[str],
    seen_public_hashes: set[str],
    seen_final_sequences: set[str],
) -> str:
    blocked = trusted._blocked(candidate, exclusions)
    if blocked == "public_source_id":
        return "excluded_id"
    if blocked == "public_source_index":
        return "excluded_index"
    if blocked in {"public_rendered_hash", "public_final_sequence_hash"}:
        return "excluded_hash"
    if candidate.public_record_sha256 in p04.source_hashes:
        return "excluded_p04_source_hash"
    if len(candidate.token_ids) >= SEQUENCE_TOKENS + 1:
        p04_sequence = trusted._sequence_digest(candidate.token_ids[: SEQUENCE_TOKENS + 1])
        if p04_sequence in p04.sequence_hashes_129:
            return "excluded_p04_h129_sequence_hash"
    if candidate.public_record_sha256 in p06_source:
        return "excluded_p06_source_hash"
    if _p06_sequence_digest(candidate.token_ids) in p06_sequence:
        return "excluded_p06_h128_sequence_hash"
    if candidate.public_record_sha256 in seen_public_hashes:
        return "duplicate_rendered_source"
    if candidate.final_sequence_sha256 in seen_final_sequences:
        return "duplicate_final_sequence"
    seen_public_hashes.add(candidate.public_record_sha256)
    seen_final_sequences.add(candidate.final_sequence_sha256)
    return "eligible"


def _select_domain(
    dataset: Any,
    *,
    style: str,
    tokenizer: Any,
    records_per_domain: int,
    exclusions: Any,
    p04: eligibility.OpaqueExclusions,
    p06_source: frozenset[str],
    p06_sequence: frozenset[str],
    seen_public_hashes: set[str],
    seen_final_sequences: set[str],
) -> tuple[list[Any], dict[str, int]]:
    start, stop = SOURCE_RANGES[style]
    if len(dataset) < stop:
        raise SelectionError(f"{style} cache has {len(dataset)} rows; need {stop}")
    diagnostics = {
        "excluded_id": 0,
        "excluded_index": 0,
        "excluded_hash": 0,
        "excluded_p04_source_hash": 0,
        "excluded_p04_h129_sequence_hash": 0,
        "excluded_p06_source_hash": 0,
        "excluded_p06_h128_sequence_hash": 0,
        "invalid": 0,
        "duplicate_rendered_source": 0,
        "duplicate_final_sequence": 0,
    }
    selected: list[Any] = []
    spec = SOURCE_PARTITIONS[style]
    order = deterministic_row_order(
        range(start, stop), dataset_key=f"{style}-future-holdout", seed=SELECTION_SEED
    )
    for index in order:
        expected_id = source_record_id(
            str(spec["dataset_id"]), str(spec["split"]), str(spec["revision"]), index
        )
        if expected_id in exclusions.ids[style]:
            diagnostics["excluded_id"] += 1
            continue
        if index in exclusions.indices[style]:
            diagnostics["excluded_index"] += 1
            continue
        row = trusted._read_reserved_row(dataset, style=style, row_index=index)
        try:
            candidate = trusted._render_row(style, row, index, tokenizer)
        except trusted.ProducerError:
            diagnostics["invalid"] += 1
            continue
        reason = _classify_candidate(
            candidate,
            exclusions=exclusions,
            p04=p04,
            p06_source=p06_source,
            p06_sequence=p06_sequence,
            seen_public_hashes=seen_public_hashes,
            seen_final_sequences=seen_final_sequences,
        )
        if reason == "eligible":
            selected.append(candidate)
            if len(selected) == records_per_domain:
                break
        else:
            diagnostics[reason] += 1
    if len(selected) != records_per_domain:
        raise SelectionError(
            f"{style} eligible pool yielded {len(selected)} rows; need {records_per_domain}"
        )
    if len({str(record.record_id) for record in selected}) != records_per_domain:
        raise SelectionError(f"{style} selected source IDs are not unique")
    return selected, diagnostics


def _selection_metadata(rows: Mapping[str, Sequence[Any]]) -> dict[str, list[dict[str, Any]]]:
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
    result: dict[str, list[dict[str, Any]]] = {}
    for style in STYLE_ORDER:
        values: list[dict[str, Any]] = []
        for record in rows[style]:
            metadata = dict(record.selection_metadata())
            if set(metadata) != allowed:
                raise SelectionError(f"{style} renderer metadata fields changed")
            if metadata.get("valid_tokens") != SEQUENCE_TOKENS:
                raise SelectionError(f"{style} selected clip length changed")
            for key in ("public_record_sha256", "final_sequence_sha256"):
                _require_hex(metadata.get(key), description=f"{style} {key}")
            values.append(metadata)
        result[style] = values
    all_sequences = [row["final_sequence_sha256"] for style in STYLE_ORDER for row in result[style]]
    if len(set(all_sequences)) != len(all_sequences):
        raise SelectionError("selected final H128 sequences are not unique across domains")
    return result


def _exclusion_source_descriptors(exclusions: Any) -> list[dict[str, Any]]:
    return [
        {
            key: source[key]
            for key in ("path", "available", "bytes", "sha256", "new_identity_count")
            if key in source
        }
        for source in exclusions.sources
    ]


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
    return value if value else None


def _build_exclusion_payload(
    *,
    exclusions: Any,
    p04_descriptor: Mapping[str, Any],
    p04: eligibility.OpaqueExclusions,
    p06_summary: Mapping[str, Any],
    inventory_record: Mapping[str, Any],
    decision_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": EXCLUSION_SCHEMA,
        "task_id": TASK_ID,
        "status": EXCLUSION_STATUS,
        "created_utc": _utc_now(),
        "identity_only": True,
        "sources": _exclusion_source_descriptors(exclusions),
        "identity_counts": {
            style: {
                "ids": len(exclusions.ids[style]),
                "hashes": len(exclusions.hashes[style]),
                "indices": len(exclusions.indices[style]),
            }
            for style in STYLE_ORDER
        },
        "p04_exchange": dict(p04_descriptor),
        "p04_field_summaries": dict(p04.fields),
        "p06_opaque_reservation": dict(p06_summary),
        "identity_inventory": dict(inventory_record),
        "decision_contract": dict(decision_record),
        "known_prior_panels": [
            "TRR-0003/TRR-0004/TRR-0005 known fit and validation identities",
            "TRR-0005/TRR-0006 opened public evaluation identities",
            "TRR-0007 final bank and fitting-prefix ledgers",
        ],
        "source_text_or_token_ids_written": False,
        "private_or_truth_payload_read": False,
        "truth_opened": False,
        "truth_created": False,
    }


def _build_selection_payload(
    *,
    root: Path,
    decision_record: Mapping[str, Any],
    decision: Mapping[str, Any],
    method_record: Mapping[str, Any],
    method_freeze: Mapping[str, Any],
    method_states: Mapping[str, Mapping[str, Any]],
    inventory_record: Mapping[str, Any],
    inventory: Mapping[str, Any],
    planning_record: Mapping[str, Any],
    planning_compatibility: Mapping[str, Any],
    p06_summary: Mapping[str, Any],
    p04_descriptor: Mapping[str, Any],
    exclusion_record: Mapping[str, Any],
    source_inputs: Mapping[str, Any],
    metadata: Mapping[str, Sequence[Mapping[str, Any]]],
    diagnostics: Mapping[str, Mapping[str, int]],
    selector_source: Path,
) -> dict[str, Any]:
    ids = {
        style: [str(row["record_id"]) for row in metadata[style]] for style in STYLE_ORDER
    }
    sequences = {
        style: [str(row["final_sequence_sha256"]) for row in metadata[style]]
        for style in STYLE_ORDER
    }
    compatibility = dict(planning_compatibility)
    return {
        "schema": SELECTION_SCHEMA,
        "task_id": TASK_ID,
        "status": SELECTION_STATUS,
        "created_utc": _utc_now(),
        "decision_contract": dict(decision_record),
        "decision_contract_sha256": decision_record["sha256"],
        "method_freeze": dict(method_record),
        "method_freeze_sha256": method_record["sha256"],
        "method_freeze_state_sha256": {
            method_id: state["sha256"] for method_id, state in method_states.items()
        },
        "identity_inventory": dict(inventory_record),
        "planning_status": dict(planning_record),
        "records_by_domain": dict(EXPECTED_RECORDS_BY_DOMAIN),
        "selection_seed": SELECTION_SEED,
        "source_ranges_half_open": dict(SOURCE_RANGES),
        "sequence_tokens_including_bos": SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "target_conditions": list(TARGET_CONDITIONS),
        "paired_conditions": True,
        "natural_distribution_preserved": True,
        "public_sources_frozen": {
            "pile": {"arrow_files": list(source_inputs["pile_arrow"])},
            "finance": {"arrow_files": list(source_inputs["finance_arrow"])},
            "tokenizer": dict(source_inputs["tokenizer"]),
        },
        "p06_hash_compatibility": compatibility,
        "selection_rule": {
            "algorithm": "Deterministic TRR-0005 future-holdout order via deterministic_row_order(seed=5005); reject the verified TRR-0007 public identity ledgers, approved P04 source/H129 opaque hashes, approved P06 source/H128 opaque hashes, invalid rows, duplicate rendered sources, and duplicate H128 sequences; retain the first predeclared eligible count per domain.",
            "identity_exclusions": True,
            "source_text_or_token_ids_written": False,
            "record_ids_sha256": {style: _json_digest(ids[style]) for style in STYLE_ORDER},
            "final_sequence_sha256": {style: _json_digest(sequences[style]) for style in STYLE_ORDER},
            "records": {style: list(metadata[style]) for style in STYLE_ORDER},
        },
        "selection_exclusions": {
            "path": str(exclusion_record["path"]),
            "bytes": exclusion_record["bytes"],
            "sha256": exclusion_record["sha256"],
            "p04_exchange": dict(p04_descriptor),
            "p06_opaque_reservation": dict(p06_summary),
            "targetfit_per_record_metadata_available": False,
        },
        "selection_diagnostics": {
            style: {
                **dict(diagnostics[style]),
                "selected": len(metadata[style]),
                "pool_size": SOURCE_RANGES[style][1] - SOURCE_RANGES[style][0],
            }
            for style in STYLE_ORDER
        },
        "execution": {
            "command": list(sys.argv),
            "code_commit": _git_commit(root),
            "selector_source": _file_record(selector_source, description="TRR-0008 selector source"),
            "trusted_trr0005_producer_source": _file_record(Path(trusted.__file__), description="trusted TRR-0005 producer source"),
            "trusted_trr0006_eligibility_source": _file_record(Path(eligibility.__file__), description="trusted TRR-0006 eligibility source"),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "network_used": False,
            "model_loaded": False,
            "target_loaded": False,
            "source_text_written": False,
            "token_ids_written": False,
            "truth_created_or_opened": False,
            "selection_performed": True,
        },
        "source_text_or_target_labels": False,
        "truth_opened": False,
        "truth_created": False,
    }


def select_public(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise SelectionError(f"repository root is unavailable: {root}")
    # All owner and compatibility gates happen before tokenizer/dataset access.
    decision_record, decision = _validate_decision_contract(
        (root / args.decision_contract).resolve()
        if not Path(args.decision_contract).is_absolute()
        else Path(args.decision_contract).resolve()
    )
    planning_path = (
        (root / args.planning_status).resolve()
        if not Path(args.planning_status).is_absolute()
        else Path(args.planning_status).resolve()
    )
    planning_record, _compatibility = _validate_planning_status(planning_path)
    inventory_path = (
        (root / args.inventory).resolve() if not Path(args.inventory).is_absolute() else Path(args.inventory).resolve()
    )
    inventory_record, inventory = _validate_inventory(
        inventory_path, expected_counts=EXPECTED_RECORDS_BY_DOMAIN
    )
    _validate_trr7_file_bindings(inventory, root=root)
    method_path = (
        (root / args.method_freeze).resolve()
        if not Path(args.method_freeze).is_absolute()
        else Path(args.method_freeze).resolve()
    )
    method_record, method_freeze, method_states = _validate_method_freeze(method_path, root=root)

    selection_path = _task_path(args.output, root=root, description="source selection output")
    exclusions_path = _task_path(args.exclusions_output, root=root, description="source exclusions output")
    if selection_path == exclusions_path:
        raise SelectionError("selection and exclusion outputs must differ")
    if selection_path.exists() or exclusions_path.exists():
        raise SelectionError("selection and exclusion outputs are create-only")

    pile_paths = tuple(Path(value).expanduser().resolve() for value in args.pile_arrow)
    finance_paths = tuple(Path(value).expanduser().resolve() for value in args.finance_arrow)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    source_inputs = _validate_public_inputs(
        inventory,
        pile_paths=pile_paths,
        finance_paths=finance_paths,
        tokenizer_path=tokenizer_path,
    )
    p04, p04_descriptor = _p04_opaque()
    p06_summary, p06_source, p06_sequence = _p06_opaque()

    tokenizer = trusted._load_tokenizer(tokenizer_path)
    datasets = {
        "pile": trusted._load_arrow_dataset(pile_paths),
        "finance": trusted._load_arrow_dataset(finance_paths),
    }
    known_paths = _known_exclusion_paths(root)
    exclusions = trusted._collect_exclusions(known_paths)
    seen_public_hashes: set[str] = set()
    seen_final_sequences: set[str] = set()
    selected: dict[str, list[Any]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    for style in STYLE_ORDER:
        selected[style], diagnostics[style] = _select_domain(
            datasets[style],
            style=style,
            tokenizer=tokenizer,
            records_per_domain=EXPECTED_RECORDS_BY_DOMAIN[style],
            exclusions=exclusions,
            p04=p04,
            p06_source=p06_source,
            p06_sequence=p06_sequence,
            seen_public_hashes=seen_public_hashes,
            seen_final_sequences=seen_final_sequences,
        )
    metadata = _selection_metadata(selected)
    exclusion_payload = _build_exclusion_payload(
        exclusions=exclusions,
        p04_descriptor=p04_descriptor,
        p04=p04,
        p06_summary=p06_summary,
        inventory_record=inventory_record,
        decision_record=decision_record,
    )
    exclusion_record = _write_create_only(
        exclusions_path, exclusion_payload, description="TRR-0008 source exclusions"
    )
    selection_payload = _build_selection_payload(
        root=root,
        decision_record=decision_record,
        decision=decision,
        method_record=method_record,
        method_freeze=method_freeze,
        method_states=method_states,
        inventory_record=inventory_record,
        inventory=inventory,
        planning_record=planning_record,
        planning_compatibility=_compatibility,
        p06_summary=p06_summary,
        p04_descriptor=p04_descriptor,
        exclusion_record=exclusion_record,
        source_inputs=source_inputs,
        metadata=metadata,
        diagnostics=diagnostics,
        selector_source=Path(__file__).resolve(),
    )
    selection_record = _write_create_only(
        selection_path, selection_payload, description="TRR-0008 source selection"
    )
    return {
        "task_id": TASK_ID,
        "status": SELECTION_STATUS,
        "selection": selection_record,
        "exclusions": exclusion_record,
        "records_by_domain": dict(EXPECTED_RECORDS_BY_DOMAIN),
        "truth_created_or_opened": False,
    }


def _load_selection_for_reservation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path, description="TRR-0008 source selection")
    payload = _load_json(path, description="TRR-0008 source selection")
    if payload.get("schema") != SELECTION_SCHEMA or payload.get("task_id") != TASK_ID:
        raise SelectionError("selection ledger schema or task identity changed")
    if payload.get("status") != SELECTION_STATUS:
        raise SelectionError("selection ledger is not a completed no-truth selection")
    if payload.get("source_text_or_target_labels") is not False or payload.get("truth_opened") is not False or payload.get("truth_created") is not False:
        raise SelectionError("selection ledger records forbidden source or truth access")
    rule = payload.get("selection_rule")
    if not isinstance(rule, Mapping) or rule.get("source_text_or_token_ids_written") is not False:
        raise SelectionError("selection ledger is not certified identity-only")
    rows = rule.get("records")
    if not isinstance(rows, Mapping):
        raise SelectionError("selection ledger records are absent")
    return record, payload


def _hash_summary(values: Sequence[str]) -> dict[str, Any]:
    ordered = sorted(set(values))
    return {
        "values": ordered,
        "ordered_count": len(ordered),
        "distinct_count": len(ordered),
        "ordered_newline_sha256": _newline_digest(ordered),
        "unique_set_newline_sha256": _newline_digest(ordered),
        "unique_set_canonical_json_sha256": _json_digest(ordered),
    }


def _reservation_payload(
    selection_record: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    rows = selection["selection_rule"]["records"]
    source_values: list[str] = []
    sequence_values: list[str] = []
    counts: dict[str, int] = {}
    for style, values in rows.items():
        if style not in STYLE_ORDER or not isinstance(values, list) or not values:
            raise SelectionError("selection ledger has malformed domain rows")
        counts[style] = len(values)
        for row in values:
            if not isinstance(row, Mapping):
                raise SelectionError("selection ledger row is malformed")
            source_values.append(_require_hex(row.get("public_record_sha256"), description="selected public source hash"))
            sequence_values.append(_require_hex(row.get("final_sequence_sha256"), description="selected H128 sequence hash"))
    if counts != EXPECTED_RECORDS_BY_DOMAIN:
        raise SelectionError("selection ledger counts differ from the prospective panel")
    if len(source_values) != len(set(source_values)) or len(sequence_values) != len(set(sequence_values)):
        raise SelectionError("selection ledger contains duplicate reservation hashes")
    compatibility = selection.get("p06_hash_compatibility")
    if not isinstance(compatibility, Mapping) or compatibility.get("source_hash_byte_input_status") != "VERIFIED_P06_PRODUCER_CONFIRMATION":
        raise SelectionError("selection ledger does not carry explicit P06 source-byte confirmation")
    core = {
        "schema": RESERVATION_SCHEMA,
        "hashes": {
            "public_record_sha256": sorted(source_values),
            "final_sequence_sha256": sorted(sequence_values),
        },
    }
    return {
        "schema": RESERVATION_SCHEMA,
        "task_id": TASK_ID,
        "status": RESERVATION_STATUS,
        "created_utc": _utc_now(),
        "purpose": "hash-only TRR-0008 source and H128 sequence reservation",
        "input_selection_ledger": dict(selection_record),
        "hash_conventions": {
            "public_record_sha256": "Copied from the selected public identity metadata; P06 source-byte compatibility was explicitly verified before selection.",
            "final_sequence_sha256": "SHA-256 of exactly 128 little-endian signed int32 token IDs including BOS.",
            "canonical_order": "Lexicographic order within each hash set; no domain or selection-order metadata is exported.",
        },
        "counts": {
            "public_record_sha256": len(source_values),
            "final_sequence_sha256": len(sequence_values),
        },
        "hashes": {
            "public_record_sha256": _hash_summary(source_values),
            "final_sequence_sha256": _hash_summary(sequence_values),
        },
        "reservation_digest_sha256": _json_digest(core),
        "privacy_boundary": {
            "hash_only": True,
            "contains_source_text": False,
            "contains_record_ids": False,
            "contains_source_indices": False,
            "contains_domain_or_style_labels": False,
            "contains_target_labels": False,
            "contains_token_ids": False,
            "contains_model_weights": False,
            "contains_truth": False,
        },
    }


def export_reservation(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).expanduser().resolve()
    selection_record, selection = _load_selection_for_reservation(selection_path)
    output = _task_path(args.output, root=Path(args.repository_root).expanduser().resolve(), description="opaque reservation output")
    if output.exists() or output.is_symlink():
        raise SelectionError("opaque reservation output is create-only")
    payload = _reservation_payload(selection_record, selection)
    output_record = _write_create_only(output, payload, description="TRR-0008 opaque reservation")
    return {"task_id": TASK_ID, "status": RESERVATION_STATUS, "output": output_record}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="select after the frozen decision and P06 gates")
    select.add_argument("--repository-root", type=Path, default=Path("."))
    select.add_argument("--decision-contract", type=Path, default=Path("experiments/TRR-0008/planning/decision_contract.json"))
    select.add_argument("--planning-status", type=Path, default=Path("experiments/TRR-0008/coordination/planning_status.json"))
    select.add_argument("--inventory", type=Path, default=Path("experiments/TRR-0008/planning/identity_inventory_1thread.json"))
    select.add_argument("--method-freeze", type=Path, default=Path("experiments/TRR-0007/method_freeze.json"))
    select.add_argument("--tokenizer", type=Path, required=True)
    select.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, default=Path("experiments/TRR-0008/selection/source_selection.json"))
    select.add_argument("--exclusions-output", type=Path, default=Path("experiments/TRR-0008/selection/source_exclusions.json"))
    reserve = sub.add_parser("reserve", help="export hash-only reservation from completed selection")
    reserve.add_argument("--repository-root", type=Path, default=Path("."))
    reserve.add_argument("--selection", type=Path, required=True)
    reserve.add_argument("--output", type=Path, default=Path("experiments/TRR-0008/selection/opaque_source_sequence_reservation.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        result = select_public(args)
    elif args.command == "reserve":
        result = export_reservation(args)
    else:  # pragma: no cover
        raise SelectionError(f"unknown command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SelectionError, eligibility.EligibilityError, trusted.ProducerError, trr7_contract.ContractError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0008 selection error: {exc}")
