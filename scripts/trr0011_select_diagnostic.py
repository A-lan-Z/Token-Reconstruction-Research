#!/usr/bin/env python3
"""Fail-closed TRR-0011 diagnostic-panel reservation and selector gate.

This adapter owns the small identity gate around the future Agent2 P10
reservation.  It does not choose rows, load Arrow data, render tokens, read
source text, or open truth.  A caller must supply the reviewed trusted
selector as an injected callable after this gate and an explicit root
selection authorization pass.  Keeping the sampler injected reuses the
TRR-0009/TRR-0010 selector without copying its renderer or exclusion logic.

The gate checks the proposed natural ranges, exact 128-token geometry, the
released P10 exclusion audit, hash-only reservation, and namespaced identity
ledgers covering fit-bank, checkpoint-selection, opened-evaluation, and
cross-range exclusions.  It compares record IDs and final 128-token sequence
hashes in both domain-scoped and global namespaces, so a cross-domain
duplicate cannot pass accidentally.  The trusted selector remains injected
and is callable only after the released P10 bundle passes.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import types
from typing import Any

TASK_ID = "TRR-0011"
SCHEMA = "token-reconstruction.trr0011-diagnostic-selection.v1"
PANEL_SCHEMA = "token-reconstruction.trr0011-transfer-panel.v2"
RESERVATION_SCHEMA = "token-reconstruction.trr0011-p10-opaque-reservation.v1"
RESERVATION_STATUS = "APPROVED_TRR0011_OPAQUE_RESERVATION"
P10_OPAQUE_SCHEMA = "token-reconstruction.trr-p10-opaque-source-reservation.v1"
P10_OPAQUE_STATUS = "P10_OPAQUE_SOURCE_RESERVATION_NO_SOURCE_IDS"
P10_AUDIT_SCHEMA = "token-reconstruction.trr-p10-identity-exclusion-audit.v1"
P10_AUDIT_STATUS = "COMPLETE_METADATA_ONLY_EXCLUSION_AUDIT"
P10_RELEASE_SCHEMA = "token-reconstruction.trr-p10-selection-release.v1"
P10_RELEASE_STATUS = "ROOT_RELEASED_AFTER_EXCLUSION_COVERAGE"
LEDGER_SCHEMA = "token-reconstruction.trr0011-identity-ledger.v1"
LEDGER_STATUS = "IMMUTABLE_IDENTITY_LEDGER"
READY_STATUS = "READY_FOR_ROOT_AUTHORIZED_SELECTION"
SELECTION_STATUS = "FROZEN_TRR0011_DIAGNOSTIC_SOURCE_SELECTION_NO_TRUTH"
TRUSTED_SELECTOR_STATUSES = (SELECTION_STATUS, "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH")
AGENT2_HANDOFF_SCHEMA = "token-reconstruction.trr0011-agent2-p10-handoff.v1"
AGENT2_HANDOFF_STATUS = "COMPLETE_AGENT2_P10_EXCLUSION_AND_OPAQUE_RESERVATION"
AGENT2_UNION_STATUSES = {
    "COMPLETE_METADATA_ONLY_EXCLUSION_AUDIT",
    "COMPLETE_P10_EXCLUSION_UNION",
    "COMPLETE_AGENT2_P10_EXCLUSION_UNION",
    "ROOT_RELEASED_AFTER_EXCLUSION_COVERAGE",
}
AGENT2_OPAQUE_STATUSES = {
    "P10_OPAQUE_SOURCE_RESERVATION_NO_SOURCE_IDS",
    "P10_OPAQUE_CONFIRMATION_RESERVATION_NO_SOURCE_IDS",
    "READY_FOR_TRR-P10_CAPTURE_HASH_ONLY",
    "READY_FOR_TRR_P10_CAPTURE_HASH_ONLY",
    "COMPLETE_AGENT2_P10_OPAQUE_CONFIRMATION",
}
ROOT_RELEASE_SCHEMA = "token-reconstruction.trr0011-root-selection-release.v1"
ROOT_RELEASE_STATUS = "ROOT_AUTHORIZED_TRR0011_DIAGNOSTIC_SELECTION"
AGENT2_LOADER_REQUIRED = ("source_specs", "load_bundle", "merge_bundles", "check_candidate")
AGENT2_LOADER_SELECTION_NAMES = ("select_sources", "select_candidates", "run_selection")
AGENT2_UNION_REQUIRED_SCOPES = (
    "gradient_fitting",
    "checkpoint_selection",
    "opened_evaluation",
    "duplicates",
    "cross_study",
    "trr0009",
)
AGENT2_CONFIRMATION_RECORDS = 512
STYLE_ORDER = ("finance", "pile")
SOURCE_RANGES = {"finance": [28000, 30000], "pile": [9000, 10000]}
RECORDS_BY_DOMAIN = {"finance": 32, "pile": 32}
SELECTION_SEED = 5011
SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
TARGET_VARIANTS = (
    "clean_public_base",
    "early_prefix_eps1e3",
    "early_prefix_eps1e2",
    "near_cut_prefix_eps1e3",
    "near_cut_prefix_eps1e2",
    "after_cut_suffix_eps1e2_null",
)
REQUIRED_EXCLUSION_SCOPES = (
    "fit_bank",
    "checkpoint_selection",
    "opened_evaluation",
    "cross_range",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RANGE_TEXT = re.compile(r"^\[(\d+),(\d+)\)$")
FORBIDDEN_FLAGS = (
    "truth_opened",
    "truth_created",
    "source_text_loaded",
    "source_text_written",
    "target_labels_loaded",
    "token_ids_written",
    "candidate_arrays_persisted",
    "private_or_truth_payload_read",
    "fresh_evaluation_started",
    "selection_performed",
)


class SelectionError(RuntimeError):
    """Raised when the identity-only diagnostic gate cannot pass safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SelectionError("value is not canonically JSON-serializable") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve(path: Path | str, *, root: Path, description: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate.is_symlink() or not candidate.is_file():
        raise SelectionError(f"{description} is unavailable: {candidate}")
    return candidate


def _file_record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = _resolve(path, root=root, description=description)
    try:
        relative = path.relative_to(root)
        rendered = str(relative)
        readonly = False
    except ValueError:
        rendered = str(path)
        readonly = True
    return {
        "path": rendered,
        "absolute_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "readonly": readonly,
    }


def _read_json(path: Path | str, *, root: Path, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = _file_record(Path(path), root=root, description=description)
    try:
        payload = json.loads(Path(actual["absolute_path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{description} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise SelectionError(f"{description} must be a JSON object")
    return actual, dict(payload)


def _require_hex(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise SelectionError(f"{description} must be a lowercase 64-hex SHA-256 digest")
    return value


def _truth_free(payload: Mapping[str, Any], *, description: str) -> None:
    for key in FORBIDDEN_FLAGS:
        value = payload.get(key)
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            raise SelectionError(f"{description} records forbidden state: {key}")


def _parse_range(value: Any, *, description: str) -> list[int]:
    if isinstance(value, str):
        match = RANGE_TEXT.fullmatch(value)
        if match is None:
            raise SelectionError(f"{description} is not a half-open range")
        value = [int(match.group(1)), int(match.group(2))]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 2:
        raise SelectionError(f"{description} must be [start, stop)")
    try:
        start, stop = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{description} contains a non-integer bound") from exc
    if start < 0 or stop <= start:
        raise SelectionError(f"{description} is not a positive half-open range")
    return [start, stop]


def validate_panel_spec(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the design-only panel proposal without selecting records."""

    if not isinstance(panel, Mapping) or panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != TASK_ID:
        raise SelectionError("panel schema or task identity changed")
    count = panel.get("records_per_domain")
    if count != 32 or isinstance(count, bool):
        raise SelectionError("diagnostic panel must reserve exactly 32 records per domain")
    if panel.get("same_record_order_across_targets") is not True:
        raise SelectionError("diagnostic panel must be paired across target variants")
    if tuple(panel.get("target_variants", ())) != TARGET_VARIANTS:
        raise SelectionError("diagnostic target variant order changed")
    proposal = panel.get("proposed_source_ranges_half_open")
    if not isinstance(proposal, Mapping):
        raise SelectionError("diagnostic panel source ranges are absent")
    ranges = {style: _parse_range(proposal.get(style), description=f"{style} source range") for style in STYLE_ORDER}
    if ranges != SOURCE_RANGES:
        raise SelectionError("diagnostic source ranges changed")
    if proposal.get("selection_seed") != SELECTION_SEED or proposal.get("max_records_per_domain") != 32:
        raise SelectionError("diagnostic selection seed or count changed")
    _truth_free(panel, description="diagnostic panel")
    reservation = panel.get("opaque_reservation")
    if not isinstance(reservation, Mapping):
        raise SelectionError("diagnostic panel opaque reservation block is absent")
    return {
        "schema": PANEL_SCHEMA,
        "task_id": TASK_ID,
        "records_by_domain": dict(RECORDS_BY_DOMAIN),
        "source_ranges_half_open": dict(SOURCE_RANGES),
        "selection_seed": SELECTION_SEED,
        "target_variants": list(TARGET_VARIANTS),
        "reservation_status": str(reservation.get("status", "PENDING")),
        "selection_performed": False,
    }


def _extract_rows(payload: Mapping[str, Any], *, description: str) -> dict[str, list[dict[str, Any]]]:
    rows = payload.get("records_by_domain")
    if not isinstance(rows, Mapping):
        rows = payload.get("rows_by_domain")
    if not isinstance(rows, Mapping):
        raise SelectionError(f"{description} lacks records_by_domain")
    result: dict[str, list[dict[str, Any]]] = {}
    for style in STYLE_ORDER:
        values = rows.get(style)
        if not isinstance(values, list):
            raise SelectionError(f"{description} lacks {style} identity rows")
        checked: list[dict[str, Any]] = []
        for index, item in enumerate(values):
            if not isinstance(item, Mapping):
                raise SelectionError(f"{description} {style} row {index} is malformed")
            record_id = item.get("record_id")
            sequence = item.get("final_sequence_sha256")
            if not isinstance(record_id, str) or not record_id:
                raise SelectionError(f"{description} {style} row {index} lacks record_id")
            _require_hex(sequence, description=f"{description} {style} row {index} final sequence")
            forbidden = {"text", "tokens", "token_ids", "labels", "truth"} & set(item)
            if forbidden:
                raise SelectionError(f"{description} contains forbidden identity payload: {sorted(forbidden)}")
            checked.append({"record_id": record_id, "final_sequence_sha256": sequence})
        result[style] = checked
    return result


def _identity_sets(rows: Mapping[str, Sequence[Mapping[str, Any]]], *, description: str) -> dict[str, Any]:
    scoped_ids: dict[str, set[str]] = {}
    scoped_sequences: dict[str, set[str]] = {}
    all_ids: set[str] = set()
    all_sequences: set[str] = set()
    for style in STYLE_ORDER:
        scoped_ids[style] = set()
        scoped_sequences[style] = set()
        for row in rows[style]:
            record_id = str(row["record_id"])
            sequence = str(row["final_sequence_sha256"])
            if record_id in scoped_ids[style] or sequence in scoped_sequences[style]:
                raise SelectionError(f"{description} contains a duplicate within {style}")
            if record_id in all_ids or sequence in all_sequences:
                raise SelectionError(f"{description} contains a cross-domain identity duplicate")
            scoped_ids[style].add(record_id)
            scoped_sequences[style].add(sequence)
            all_ids.add(record_id)
            all_sequences.add(sequence)
    return {
        "scoped_ids": scoped_ids,
        "scoped_sequences": scoped_sequences,
        "all_ids": all_ids,
        "all_sequences": all_sequences,
        "counts": {style: len(scoped_ids[style]) for style in STYLE_ORDER},
        "record_ids_order_digest": {style: _json_digest([row["record_id"] for row in rows[style]]) for style in STYLE_ORDER},
        "sequence_order_digest": {style: _json_digest([row["final_sequence_sha256"] for row in rows[style]]) for style in STYLE_ORDER},
    }


def validate_p10_reservation(path: Path | str, *, root: Path) -> dict[str, Any]:
    """Validate the immutable P10 identity handoff, without exposing rows."""

    record, payload = _read_json(path, root=root, description="P10 opaque reservation")
    if payload.get("schema") != RESERVATION_SCHEMA or payload.get("status") != RESERVATION_STATUS:
        raise SelectionError("P10 reservation is not the approved TRR-0011 opaque handoff")
    _truth_free(payload, description="P10 reservation")
    if payload.get("task_id") not in ("TRR-P10", "TRR-0011"):
        raise SelectionError("P10 reservation task identity changed")
    ranges = payload.get("source_ranges_half_open")
    if not isinstance(ranges, Mapping) or {style: _parse_range(ranges.get(style), description=f"P10 {style} range") for style in STYLE_ORDER} != SOURCE_RANGES:
        raise SelectionError("P10 reservation ranges do not match the frozen proposal")
    if payload.get("selection_seed") != SELECTION_SEED:
        raise SelectionError("P10 reservation selection seed changed")
    receipt = payload.get("reservation_receipt_sha256") or payload.get("receipt_sha256")
    _require_hex(receipt, description="P10 reservation receipt")
    rows = _extract_rows(payload, description="P10 reservation")
    for style in STYLE_ORDER:
        if len(rows[style]) != RECORDS_BY_DOMAIN[style]:
            raise SelectionError(f"P10 reservation {style} count is not 32")
    identity = _identity_sets(rows, description="P10 reservation")
    declared = payload.get("order_digests")
    if not isinstance(declared, Mapping):
        raise SelectionError("P10 reservation order digests are absent")
    for style in STYLE_ORDER:
        expected = {
            "record_ids_sha256": identity["record_ids_order_digest"][style],
            "final_sequence_sha256": identity["sequence_order_digest"][style],
        }
        if dict(declared.get(style, {})) != expected:
            raise SelectionError(f"P10 reservation order digest changed: {style}")
    return {
        "file": record,
        "schema": payload["schema"],
        "status": payload["status"],
        "reservation_receipt_sha256": receipt,
        "records_by_domain": dict(identity["counts"]),
        "record_ids_order_digest": dict(identity["record_ids_order_digest"]),
        "sequence_order_digest": dict(identity["sequence_order_digest"]),
        "identity": identity,
    }


def validate_exclusion_ledger(path: Path | str, *, root: Path, expected_scope: str) -> dict[str, Any]:
    """Validate one hash-bound, namespaced identity ledger."""

    if expected_scope not in REQUIRED_EXCLUSION_SCOPES:
        raise SelectionError(f"unknown exclusion scope: {expected_scope}")
    record, payload = _read_json(path, root=root, description=f"{expected_scope} exclusion ledger")
    if payload.get("schema") != LEDGER_SCHEMA or payload.get("status") != LEDGER_STATUS:
        raise SelectionError(f"{expected_scope} ledger schema/status is not immutable")
    _truth_free(payload, description=f"{expected_scope} exclusion ledger")
    if payload.get("scope") != expected_scope or not isinstance(payload.get("namespace"), str) or not payload["namespace"]:
        raise SelectionError(f"{expected_scope} ledger scope/namespace is absent")
    declared_file = payload.get("file")
    if isinstance(declared_file, Mapping):
        if declared_file.get("bytes") != record["bytes"] or declared_file.get("sha256") != record["sha256"]:
            raise SelectionError(f"{expected_scope} ledger file binding changed")
    rows = _extract_rows(payload, description=f"{expected_scope} exclusion ledger")
    identity = _identity_sets(rows, description=f"{expected_scope} exclusion ledger")
    return {
        "scope": expected_scope,
        "namespace": payload["namespace"],
        "file": record,
        "records_by_domain": dict(identity["counts"]),
        "record_ids_order_digest": dict(identity["record_ids_order_digest"]),
        "sequence_order_digest": dict(identity["sequence_order_digest"]),
        "identity": identity,
    }


def validate_zero_overlap(reservation: Mapping[str, Any], ledgers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require all required exclusion namespaces to be disjoint from P10."""

    identity = reservation.get("identity")
    if not isinstance(identity, Mapping):
        raise SelectionError("validated P10 identity is absent")
    if len(ledgers) != len(REQUIRED_EXCLUSION_SCOPES):
        raise SelectionError("all four required exclusion namespaces must be supplied")
    scopes = [str(item.get("scope")) for item in ledgers]
    if set(scopes) != set(REQUIRED_EXCLUSION_SCOPES):
        raise SelectionError(f"required exclusion scopes are missing or duplicated: {scopes}")
    namespaces = [str(item.get("namespace")) for item in ledgers]
    if len(set(namespaces)) != len(namespaces):
        raise SelectionError("exclusion ledger namespaces must be unique")
    reservation_ids = set(identity["all_ids"])
    reservation_sequences = set(identity["all_sequences"])
    overlap: dict[str, dict[str, int]] = {}
    for item in ledgers:
        ledger_identity = item.get("identity")
        if not isinstance(ledger_identity, Mapping):
            raise SelectionError(f"ledger identity is absent: {item.get('scope')}")
        ids = set(ledger_identity["all_ids"])
        sequences = set(ledger_identity["all_sequences"])
        overlap[str(item["scope"])] = {
            "record_ids": len(reservation_ids & ids),
            "final_sequence_sha256": len(reservation_sequences & sequences),
        }
        if overlap[str(item["scope"])] != {"record_ids": 0, "final_sequence_sha256": 0}:
            raise SelectionError(f"P10 reservation overlaps {item['scope']} exclusion ledger: {overlap[item['scope']]}")
    return {
        "status": "PASS_ZERO_OVERLAP",
        "reservation_records": dict(reservation["records_by_domain"]),
        "required_scopes": list(REQUIRED_EXCLUSION_SCOPES),
        "overlap_counts": overlap,
        "selection_performed": False,
        "source_text_loaded": False,
        "truth_opened": False,
    }


def build_selection_binding(
    *,
    panel_path: Path | str,
    reservation_path: Path | str,
    exclusion_paths: Mapping[str, Path | str],
    root: Path,
    output_path: Path | str,
) -> dict[str, Any]:
    """Create a hash-bound, identity-only readiness receipt.

    This is the last step before the released P10 bundle permits the trusted
    selector.  It never calls a selector or opens public rows.
    """

    panel_record, panel = _read_json(panel_path, root=root, description="TRR-0011 diagnostic panel")
    panel_summary = validate_panel_spec(panel)
    reservation = validate_p10_reservation(reservation_path, root=root)
    if set(exclusion_paths) != set(REQUIRED_EXCLUSION_SCOPES):
        raise SelectionError("exclusion_paths must bind exactly the four required scopes")
    ledgers = [validate_exclusion_ledger(exclusion_paths[scope], root=root, expected_scope=scope) for scope in REQUIRED_EXCLUSION_SCOPES]
    zero = validate_zero_overlap(reservation, ledgers)
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    evaluation_root = (root / "experiments" / TASK_ID / "evaluation").resolve()
    try:
        output.relative_to(evaluation_root)
    except ValueError as exc:
        raise SelectionError(f"selection binding must remain under {evaluation_root}") from exc
    if output.exists() or output.is_symlink():
        raise SelectionError(f"selection binding is create-only: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": READY_STATUS,
        "created_utc": _utc_now(),
        "panel": panel_summary,
        "panel_file": panel_record,
        "p10_reservation": {key: value for key, value in reservation.items() if key != "identity"},
        "exclusion_ledgers": [
            {key: value for key, value in ledger.items() if key != "identity"} for ledger in ledgers
        ],
        "zero_overlap": zero,
        "selection_authorized": False,
        "selection_performed": False,
        "source_text_loaded": False,
        "truth_opened": False,
        "target_labels_loaded": False,
    }
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    file_record = _file_record(output, root=root, description="diagnostic selection binding")
    return {"binding": file_record, "payload": payload}


def validate_p10_release(
    path: Path | str,
    *,
    root: Path,
    audit_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate the existing P10 root release, without selecting rows."""

    record, payload = _read_json(path, root=root, description="P10 selection release")
    if payload.get("schema") != P10_RELEASE_SCHEMA or payload.get("task_id") != "TRR-P10":
        raise SelectionError("P10 release schema or task identity changed")
    if payload.get("status") != P10_RELEASE_STATUS:
        raise SelectionError("P10 release is not the complete selection release")
    _truth_free(payload, description="P10 selection release")
    if payload.get("exclusion_coverage_complete") is not True:
        raise SelectionError("P10 release does not certify complete exclusion coverage")
    contract = payload.get("source_contract")
    if not isinstance(contract, Mapping):
        raise SelectionError("P10 release source contract is absent")
    if (
        contract.get("selection_seed") != SELECTION_SEED
        or contract.get("diagnostic_ranges") != SOURCE_RANGES
        or contract.get("diagnostic_max_records_per_domain") != 32
        or contract.get("stored_sequence_tokens") != SEQUENCE_TOKENS
        or contract.get("scored_post_bos_tokens") != SCORED_POST_BOS_TOKENS
        or tuple(contract.get("target_conditions", ())) != ("public_base", "public_lora_2601")
    ):
        raise SelectionError("P10 release diagnostic contract changed")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise SelectionError("P10 release bindings are absent")
    for key in ("exclusion_audit_sha256", "selector_code_sha256", "exclusion_worker_code_sha256"):
        _require_hex(bindings.get(key), description=f"P10 release {key}")
    audit_record = None
    if audit_path is not None:
        audit_record = _file_record(Path(audit_path), root=root, description="P10 exclusion audit")
        if bindings.get("exclusion_audit_sha256") != audit_record["sha256"]:
            raise SelectionError("P10 release binds a different exclusion audit")
    return {
        "file": record,
        "status": payload["status"],
        "source_contract": dict(contract),
        "bindings": dict(bindings),
        "exclusion_audit": audit_record,
    }


def validate_p10_audit(path: Path | str, *, root: Path) -> dict[str, Any]:
    """Validate the complete metadata-only P10 exclusion audit receipt."""

    record, payload = _read_json(path, root=root, description="P10 exclusion audit")
    if payload.get("schema") != P10_AUDIT_SCHEMA or payload.get("task_id") != "TRR-P10":
        raise SelectionError("P10 exclusion audit schema or task identity changed")
    if payload.get("status") != P10_AUDIT_STATUS or payload.get("coverage_complete") is not True:
        raise SelectionError("P10 exclusion audit is not complete")
    boundary = payload.get("access_boundary")
    if not isinstance(boundary, Mapping):
        raise SelectionError("P10 exclusion audit access boundary is absent")
    for key in ("source_text_read", "token_ids_read", "activation_payload_read", "truth_or_scores_read", "new_panel_selected", "p03_holdout_accessed"):
        if boundary.get(key) is True:
            raise SelectionError(f"P10 exclusion audit records forbidden access: {key}")
    gaps = payload.get("coverage_gaps", [])
    if not isinstance(gaps, list) or any(gap != "P03 sealed holdout was intentionally omitted." for gap in gaps):
        raise SelectionError("P10 exclusion audit has unresolved coverage gaps")
    if not isinstance(payload.get("source_inventory"), list) or not isinstance(payload.get("union_identity_counts"), Mapping):
        raise SelectionError("P10 exclusion audit union/source inventory is absent")
    return {
        "file": record,
        "status": payload["status"],
        "coverage_complete": True,
        "source_inventory_count": len(payload["source_inventory"]),
        "union_identity_counts": dict(payload["union_identity_counts"]),
    }


def validate_p10_opaque_reservation(path: Path | str, *, root: Path) -> dict[str, Any]:
    """Validate P10's hash-only reservation; IDs remain withheld by design."""

    record, payload = _read_json(path, root=root, description="P10 opaque reservation")
    if payload.get("schema") != P10_OPAQUE_SCHEMA or payload.get("task_id") != "TRR-P10":
        raise SelectionError("P10 opaque reservation schema or task identity changed")
    if payload.get("status") != P10_OPAQUE_STATUS:
        raise SelectionError("P10 opaque reservation is not the frozen hash-only status")
    for key in ("domain_counts_emitted", "source_ids_emitted", "source_indices_emitted", "source_payload_emitted", "token_ids_emitted", "truth_emitted"):
        if payload.get(key) is not False:
            raise SelectionError(f"P10 opaque reservation violates identity-only boundary: {key}")
    conventions = payload.get("hash_conventions")
    if not isinstance(conventions, Mapping) or conventions.get("source_hash") != "public_record_sha256" or not str(conventions.get("sequence_hash", "")).startswith("final_sequence_sha256"):
        raise SelectionError("P10 opaque reservation hash conventions changed")
    try:
        count = int(payload.get("record_count"))
    except (TypeError, ValueError) as exc:
        raise SelectionError("P10 opaque reservation record count is malformed") from exc
    if count != sum(RECORDS_BY_DOMAIN.values()):
        raise SelectionError("P10 opaque reservation count is not the 64-record diagnostic cap")
    source_hashes = payload.get("source_hashes")
    sequence_hashes = payload.get("sequence_hashes_h128")
    if not isinstance(source_hashes, list) or not isinstance(sequence_hashes, list) or len(source_hashes) != count or len(sequence_hashes) != count:
        raise SelectionError("P10 opaque reservation hash arrays are incomplete")
    source_hashes = [_require_hex(value, description="P10 source hash") for value in source_hashes]
    sequence_hashes = [_require_hex(value, description="P10 H128 sequence hash") for value in sequence_hashes]
    if len(set(source_hashes)) != count or len(set(sequence_hashes)) != count:
        raise SelectionError("P10 opaque reservation hashes are duplicated")
    return {
        "file": record,
        "status": payload["status"],
        "record_count": count,
        "source_hashes": source_hashes,
        "sequence_hashes_h128": sequence_hashes,
        "domain_counts_emitted": False,
    }


def validate_p10_release_bundle(
    *,
    release_path: Path | str,
    audit_path: Path | str,
    opaque_path: Path | str,
    root: Path,
) -> dict[str, Any]:
    """Validate the released P10 union/reservation before any selector call."""

    audit = validate_p10_audit(audit_path, root=root)
    release = validate_p10_release(release_path, root=root, audit_path=audit_path)
    opaque = validate_p10_opaque_reservation(opaque_path, root=root)
    return {
        "status": "PASS_P10_RELEASED_EXCLUSION_AND_OPAQUE_BINDINGS",
        "release": {key: value for key, value in release.items() if key != "source_contract"},
        "audit": audit,
        "opaque_reservation": opaque,
        "selection_performed": False,
        "source_text_loaded": False,
        "truth_opened": False,
    }


def run_released_selector(
    *,
    release_bundle: Mapping[str, Any],
    selector: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the existing trusted selector only after P10's release gate."""

    if release_bundle.get("status") != "PASS_P10_RELEASED_EXCLUSION_AND_OPAQUE_BINDINGS":
        raise SelectionError("P10 release bundle has not passed the complete exclusion gate")
    result = selector()
    if not isinstance(result, Mapping):
        raise SelectionError("trusted selector returned a non-object result")
    accepted = {
        SELECTION_STATUS,
        "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH",
        "P10_SOURCE_SELECTION_FROZEN_IDENTITY_ONLY_BEFORE_CAPTURE",
    }
    if result.get("status") not in accepted:
        raise SelectionError("trusted selector did not return a frozen identity-only status")
    execution = result.get("execution")
    performed = result.get("selection_performed")
    if performed is None and isinstance(execution, Mapping):
        performed = execution.get("selection_performed")
    truth_created = result.get("truth_created_or_opened")
    if truth_created is None and isinstance(execution, Mapping):
        truth_created = execution.get("truth_created_or_opened")
    if performed is not True or truth_created is not False:
        raise SelectionError("trusted selector result is not identity-only")
    return dict(result)

def _load_object(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise SelectionError(f"JSON object required: {path}")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--panel", type=Path, required=True)
    validate.add_argument("--reservation", type=Path, required=True)
    validate.add_argument("--ledger", action="append", nargs=2, metavar=("SCOPE", "PATH"), required=True)
    validate.add_argument("--root", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--panel", type=Path, required=True)
    prepare.add_argument("--reservation", type=Path, required=True)
    prepare.add_argument("--ledger", action="append", nargs=2, metavar=("SCOPE", "PATH"), required=True)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    p10 = sub.add_parser("validate-p10")
    p10.add_argument("--release", type=Path, required=True)
    p10.add_argument("--audit", type=Path, required=True)
    p10.add_argument("--opaque", type=Path, required=True)
    p10.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = args.root.expanduser().resolve()
        if args.command == "validate-p10":
            result = validate_p10_release_bundle(release_path=args.release, audit_path=args.audit, opaque_path=args.opaque, root=root)
        else:
            panel_record, panel = _read_json(args.panel, root=root, description="TRR-0011 diagnostic panel")
            panel_summary = validate_panel_spec(panel)
            reservation = validate_p10_reservation(args.reservation, root=root)
            paths = dict(args.ledger)
            if set(paths) != set(REQUIRED_EXCLUSION_SCOPES):
                raise SelectionError("exactly one ledger path is required for each exclusion scope")
            ledgers = [validate_exclusion_ledger(paths[scope], root=root, expected_scope=scope) for scope in REQUIRED_EXCLUSION_SCOPES]
            zero = validate_zero_overlap(reservation, ledgers)
            if args.command == "validate":
                result = {"schema": SCHEMA, "task_id": TASK_ID, "panel": panel_summary, "panel_file": panel_record, "zero_overlap": zero}
            else:
                result = build_selection_binding(panel_path=args.panel, reservation_path=args.reservation, exclusion_paths=paths, root=root, output_path=args.output)
    except (SelectionError, OSError, ValueError, TypeError) as exc:
        print(f"TRR-0011 diagnostic selection failed closed: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "LEDGER_SCHEMA",
    "LEDGER_STATUS",
    "P10_AUDIT_SCHEMA",
    "P10_AUDIT_STATUS",
    "P10_OPAQUE_SCHEMA",
    "P10_OPAQUE_STATUS",
    "P10_RELEASE_SCHEMA",
    "P10_RELEASE_STATUS",
    "REQUIRED_EXCLUSION_SCOPES",
    "RECORDS_BY_DOMAIN",
    "RESERVATION_SCHEMA",
    "RESERVATION_STATUS",
    "SCHEMA",
    "SELECTION_STATUS",
    "TRUSTED_SELECTOR_STATUSES",
    "SelectionError",
    "SOURCE_RANGES",
    "TARGET_VARIANTS",
    "build_selection_binding",
    "run_released_selector",
    "validate_exclusion_ledger",
    "validate_panel_spec",
    "validate_p10_audit",
    "validate_p10_opaque_reservation",
    "validate_p10_release",
    "validate_p10_release_bundle",
    "validate_p10_reservation",
    "validate_zero_overlap",
]

if __name__ == "__main__":
    raise SystemExit(main())
