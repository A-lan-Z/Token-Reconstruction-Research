#!/usr/bin/env python3
"""Truth-free TRR-0010 natural-panel selector.

This is a small parameterized wrapper around the reviewed TRR-0009 identity
selector.  It keeps the trusted renderer, row ordering, and exclusion
classifier, while binding the TRR-0010 panel (128 Finance and 128 Pile
records) and the six-contender freeze.  The selector writes identities and
hashes only; it never writes source text, token IDs, labels, or activations.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_build_eligibility as eligibility
from scripts import trr0009_plan as planning
from scripts import trr0009_select_public as trr9
from scripts import trr0010_eval_gate as gate
from token_reconstruction.trr0005_contract import STYLE_ORDER


TASK_ID = "TRR-0010"
SELECTION_SCHEMA = "token-reconstruction.trr0010-source-selection.v1"
SELECTION_STATUS = "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH"
EXCLUSION_SCHEMA = "token-reconstruction.trr0010-source-exclusions.v1"
EXCLUSION_STATUS = "TRR0010_PUBLIC_IDENTITY_EXCLUSIONS_COMPLETE_NO_TRUTH"
SELECTION_SEED = 5005
EXPECTED_RECORDS_BY_DOMAIN = {"finance": 128, "pile": 128}
SOURCE_RANGES = {"pile": [7000, 10000], "finance": [12000, 20000]}
TARGET_CONDITIONS = ("public_base", "public_lora_2601")
SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SelectionError(RuntimeError):
    """Raised when an identity-only selection cannot be frozen safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_digest(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SelectionError("value cannot be canonically encoded") from exc
    return _sha256_bytes(raw.encode("utf-8"))


def _resolve(value: Path | str, *, root: Path, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise SelectionError(f"{description} is unavailable: {path}")
    return path


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        inside = path.relative_to(root)
        del inside
        readonly = False
    except ValueError:
        readonly = True
    try:
        return gate.file_record(path, root=root, readonly=readonly)
    except gate.GateError as exc:
        raise SelectionError(str(exc)) from exc


def _json(path: Path, *, root: Path, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(path, root=root, description=description)
    try:
        payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{description} is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise SelectionError(f"{description} must be a JSON object")
    return record, dict(payload)


def _write_create_only(path: Path, payload: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    task_root = (root / "experiments" / TASK_ID / "evaluation").resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise SelectionError(f"{description} must remain below {task_root}") from exc
    if path.exists() or path.is_symlink():
        raise SelectionError(f"{description} is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise SelectionError(f"could not write {description}") from exc
    return _record(path, root=root, description=description)


def _require_hex(value: Any, *, description: str, length: int = 64) -> str:
    pattern = HEX64 if length == 64 else HEX40
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise SelectionError(f"{description} must be a lowercase hexadecimal digest")
    return value


def _truth_free(payload: Mapping[str, Any], *, description: str) -> None:
    forbidden = (
        "truth_opened", "truth_created", "source_text_written", "source_text_loaded",
        "token_ids_written", "target_labels_loaded", "candidate_arrays_persisted",
        "private_or_truth_payload_read", "fresh_evaluation_started",
    )
    for key in forbidden:
        value = payload.get(key)
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            raise SelectionError(f"{description} records forbidden access: {key}")


def _validate_design(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record, design = _json(path, root=root, description="TRR-0010 final evaluation design")
    _truth_free(design, description="final evaluation design")
    if design.get("schema") != "token-reconstruction.trr0010-final-evaluation-design.v1":
        raise SelectionError("final evaluation design schema changed")
    if design.get("task_id") != TASK_ID:
        raise SelectionError("final evaluation design task identity changed")
    if design.get("status") != "FROZEN_TRR0010_FINAL_EVALUATION_DESIGN":
        raise SelectionError("source selection requires the owner-frozen final design")
    code_commit = design.get("code_commit")
    if not isinstance(code_commit, str) or HEX40.fullmatch(code_commit) is None:
        raise SelectionError("final design must bind a full code commit")
    if design.get("method_order") != list(gate.METHOD_ORDER):
        raise SelectionError("final design method order does not freeze all six contenders")
    methods = design.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(gate.METHOD_ORDER):
        raise SelectionError("final design does not freeze all six contender rows")
    if design.get("contenders_frozen") is not True:
        raise SelectionError("final design lacks an explicit all-contender freeze marker")
    decision_rules = design.get("decision_rules")
    if (
        not isinstance(decision_rules, Mapping)
        or decision_rules.get("status") != "FROZEN"
        or not isinstance(decision_rules.get("rules"), Mapping)
        or not decision_rules["rules"]
    ):
        raise SelectionError("final design decision rules are not explicitly frozen")
    code_bindings = design.get("code_bindings")
    if not isinstance(code_bindings, Mapping) or not code_bindings:
        raise SelectionError("final design code bindings are absent")
    for name, binding in code_bindings.items():
        try:
            gate._record(binding, root=root, description=f"final design code {name}")
        except gate.GateError as exc:
            raise SelectionError(str(exc)) from exc
    # Source selection is downstream of the method freeze.  Verify every
    # state/readout binding now; roles alone are insufficient because a later
    # registration could otherwise substitute an unreviewed asset.
    for method_id in gate.METHOD_ORDER:
        row = methods.get(method_id)
        role = gate.METHOD_ROLES[method_id]
        if (
            not isinstance(row, Mapping)
            or row.get("id") != method_id
            or row.get("role") != role
            or row.get("frozen") is not True
            or row.get("state_frozen") is not True
        ):
            raise SelectionError(f"final design contender/state freeze is incomplete: {method_id}")
        try:
            gate._record(row.get("state"), root=root, description=f"final design state {method_id}")
        except gate.GateError as exc:
            raise SelectionError(str(exc)) from exc
        resources = row.get("resources")
        required = gate.METHOD_RESOURCE_REQUIREMENTS[method_id]
        if not isinstance(resources, Mapping) or not set(required).issubset(resources):
            raise SelectionError(f"final design resources are incomplete: {method_id}")
        for resource_name in required:
            try:
                gate._record(resources[resource_name], root=root, description=f"final design resource {method_id}/{resource_name}")
            except gate.GateError as exc:
                raise SelectionError(str(exc)) from exc
        loader = row.get("loader")
        if not isinstance(loader, Mapping):
            raise SelectionError(f"final design loader is absent: {method_id}")
        if method_id == gate.A1_A2_METHOD_ID:
            if loader.get("interface") != gate.A1_A2_LOADER_INTERFACE or loader.get("candidate_k") != 256:
                raise SelectionError("final design A1+A2 loader semantics changed")
        else:
            expected = {"interface": gate.LOADER_INTERFACE, "current_h_only": True, "full_vocabulary": True, "history_enabled": False, "a2_enabled": False}
            if any(loader.get(key) != value for key, value in expected.items()):
                raise SelectionError(f"final design current-H loader semantics changed: {method_id}")
    final = design.get("final_evaluation")
    if not isinstance(final, Mapping):
        raise SelectionError("final design final_evaluation section is absent")
    exclusions = final.get("exclusion_bindings")
    if not isinstance(exclusions, Mapping):
        raise SelectionError("final design exclusion bindings are absent")
    try:
        gate._record(exclusions.get("final_b1"), root=root, description="final design final B1 exclusion ledger")
    except gate.GateError as exc:
        raise SelectionError(str(exc)) from exc
    opaque = exclusions.get("approved_opaque_ledgers")
    if not isinstance(opaque, Sequence) or isinstance(opaque, (str, bytes, bytearray)) or not opaque:
        raise SelectionError("final design approved opaque exclusion ledgers are absent")
    for index, binding in enumerate(opaque):
        try:
            gate._record(binding, root=root, description=f"final design opaque exclusion ledger {index}")
        except gate.GateError as exc:
            raise SelectionError(str(exc)) from exc
    if dict(final.get("records_by_domain", {})) != dict(gate.RECORDS_BY_DOMAIN):
        raise SelectionError("final panel must contain 128 records per domain")
    if list(final.get("target_conditions", ())) != list(TARGET_CONDITIONS):
        raise SelectionError("final target pairing changed")
    if final.get("paired_sources_across_targets") is not True:
        raise SelectionError("final panel does not freeze paired source targets")
    geometry = final.get("capture_geometry")
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("batch_records") != CAPTURE_BATCH_RECORDS
        or geometry.get("sequence_tokens") != CAPTURE_SEQUENCE_TOKENS
        or geometry.get("stored_sequence_tokens") != SEQUENCE_TOKENS
        or geometry.get("retain_first_128") is not True
    ):
        raise SelectionError("final capture geometry is not the inherited B8x192 -> first-128 contract")
    return record, design


def _validate_inventory(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record, inventory = _json(path, root=root, description="TRR-0010 count-only source inventory")
    _truth_free(inventory, description="source inventory")
    if inventory.get("task_id") != "TRR-0009" or inventory.get("schema") != "token-reconstruction.trr0009-source-inventory.v1":
        raise SelectionError("source inventory schema/task identity changed")
    allowed_status = {
        "IDENTITY_INVENTORY_COMPLETE_NO_SELECTION_NO_TRUTH",
        "FROZEN_TRR0010_SOURCE_INVENTORY_NO_SELECTION_NO_TRUTH",
    }
    if inventory.get("status") not in allowed_status:
        raise SelectionError("source inventory is not a finalized count-only artifact")
    contract = inventory.get("source_contract")
    if not isinstance(contract, Mapping):
        raise SelectionError("source inventory source contract is absent")
    if contract.get("selection_seed") != SELECTION_SEED or contract.get("source_ranges_half_open") != SOURCE_RANGES:
        raise SelectionError("source inventory selection ranges or seed changed")
    if (
        contract.get("sequence_tokens_including_bos") != SEQUENCE_TOKENS
        or contract.get("scoring_post_bos_tokens") != SCORED_POST_BOS_TOKENS
        or contract.get("capture_batch_records") != CAPTURE_BATCH_RECORDS
        or contract.get("capture_sequence_tokens") != CAPTURE_SEQUENCE_TOKENS
        or contract.get("hidden_size") != HIDDEN_SIZE
        or contract.get("natural_distribution_preserved") is not True
    ):
        raise SelectionError("source inventory geometry/distribution contract changed")
    domains = inventory.get("domains")
    if not isinstance(domains, Mapping):
        raise SelectionError("source inventory domains are absent")
    for style in STYLE_ORDER:
        row = domains.get(style)
        if not isinstance(row, Mapping) or row.get("source_range_half_open") != SOURCE_RANGES[style]:
            raise SelectionError(f"source inventory {style} range changed")
        available = row.get("eligible_unique")
        if not isinstance(available, int) or available < EXPECTED_RECORDS_BY_DOMAIN[style]:
            raise SelectionError(f"source inventory {style} lacks finalized eligible capacity")
    execution = inventory.get("execution")
    if not isinstance(execution, Mapping):
        raise SelectionError("source inventory execution receipt is absent")
    for key in ("selection_performed", "truth_created_or_opened", "model_loaded", "source_text_written", "token_ids_written"):
        if execution.get(key) is True:
            raise SelectionError(f"source inventory records forbidden state: {key}")
    return record, inventory


def load_selection(path: Path, *, repository_root: Path, expected_counts: Mapping[str, int] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Load the TRR10 identity ledger used by registration and capture."""
    root = Path(repository_root).expanduser().resolve()
    record, selection = _json(path, root=root, description="TRR-0010 source selection")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("task_id") != TASK_ID or selection.get("status") != SELECTION_STATUS:
        raise SelectionError("source selection is not a frozen TRR-0010 identity-only ledger")
    _truth_free(selection, description="source selection")
    if selection.get("target_conditions") != list(TARGET_CONDITIONS) or selection.get("paired_conditions") is not True:
        raise SelectionError("source target pairing changed")
    declared = selection.get("records_by_domain")
    rule = selection.get("selection_rule")
    raw_rows = rule.get("records") if isinstance(rule, Mapping) else None
    if not isinstance(declared, Mapping) or not isinstance(raw_rows, Mapping):
        raise SelectionError("source selection rows/counts are absent")
    counts: dict[str, int] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    allowed = set(trr9._ALLOWED_ROW_FIELDS) if hasattr(trr9, "_ALLOWED_ROW_FIELDS") else {
        "record_id", "public_record_sha256", "dataset_key", "dataset_id", "split", "revision",
        "row_index", "source_index", "full_token_count", "post_bos_token_count", "valid_tokens", "final_sequence_sha256",
    }
    for style in STYLE_ORDER:
        values = raw_rows.get(style)
        try:
            count = int(declared[style])
        except (KeyError, TypeError, ValueError) as exc:
            raise SelectionError(f"source selection count is absent: {style}") from exc
        expected = int(expected_counts[style]) if expected_counts is not None else EXPECTED_RECORDS_BY_DOMAIN[style]
        if count != expected or not isinstance(values, list) or len(values) != count:
            raise SelectionError(f"source selection count changed: {style}")
        seen_ids: set[str] = set()
        seen_sequences: set[str] = set()
        checked: list[dict[str, Any]] = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping) or set(value) - allowed:
                raise SelectionError(f"source selection {style} row {index} contains unapproved payload")
            row = dict(value)
            if not isinstance(row.get("record_id"), str) or not row["record_id"] or row["record_id"] in seen_ids:
                raise SelectionError(f"source selection {style} record IDs are invalid")
            if row.get("dataset_key") != style or row.get("valid_tokens") != SEQUENCE_TOKENS:
                raise SelectionError(f"source selection {style} row binding changed")
            _require_hex(row.get("public_record_sha256"), description=f"{style} source hash")
            sequence = _require_hex(row.get("final_sequence_sha256"), description=f"{style} H128 hash")
            if sequence in seen_sequences:
                raise SelectionError(f"source selection {style} H128 sequences are duplicated")
            seen_ids.add(row["record_id"])
            seen_sequences.add(sequence)
            checked.append(row)
        rows[style] = checked
        counts[style] = count
    rule_digests = rule.get("record_ids_sha256")
    if not isinstance(rule_digests, Mapping) or set(rule_digests) != set(STYLE_ORDER):
        raise SelectionError("source selection record-order digests are absent")
    for style in STYLE_ORDER:
        expected_digest = _json_digest([row["record_id"] for row in rows[style]])
        if rule_digests[style] != expected_digest:
            raise SelectionError(f"source selection record-order digest changed: {style}")
    return selection, record, rows, counts


@contextmanager
def _trr9_geometry() -> Any:
    """Temporarily parameterize trusted TRR9 internals without global drift."""
    old_ranges = trr9.SOURCE_RANGES
    try:
        trr9.SOURCE_RANGES = dict(SOURCE_RANGES)
        yield
    finally:
        trr9.SOURCE_RANGES = old_ranges


def _extra_exclusion_paths(root: Path, values: Sequence[Path]) -> list[Path]:
    paths = list(trr9._known_exclusion_paths(root))
    seen = {str(Path(path).expanduser().resolve()) for path in paths}
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if str(path) not in seen:
            paths.append(path)
            seen.add(str(path))
    return paths


def _build_exclusions(*, design_record: Mapping[str, Any], inventory_record: Mapping[str, Any], exclusions: Any, p04_descriptor: Mapping[str, Any], p04: Any, p06_summary: Mapping[str, Any], p08_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": EXCLUSION_SCHEMA,
        "task_id": TASK_ID,
        "status": EXCLUSION_STATUS,
        "created_utc": _utc_now(),
        "identity_only": True,
        "sources": trr9._exclusion_source_descriptors(exclusions),
        "identity_counts": {style: {"ids": len(exclusions.ids[style]), "hashes": len(exclusions.hashes[style]), "indices": len(exclusions.indices[style])} for style in STYLE_ORDER},
        "p04_exchange": dict(p04_descriptor),
        "p04_field_summaries": dict(p04.fields),
        "p06_opaque_reservation": dict(p06_summary),
        "p08_opaque_reservation": dict(p08_summary),
        "identity_inventory": dict(inventory_record),
        "final_design": dict(design_record),
        "known_prior_panels": ["TRR-0003 through TRR-0009 identity ledgers and approved opaque exclusions"],
        "source_text_or_token_ids_written": False,
        "private_or_truth_payload_read": False,
        "truth_opened": False,
        "truth_created": False,
    }


def _selection_payload(*, root: Path, design_record: Mapping[str, Any], inventory_record: Mapping[str, Any], exclusion_record: Mapping[str, Any], source_inputs: Mapping[str, Any], metadata: Mapping[str, Sequence[Mapping[str, Any]]], diagnostics: Mapping[str, Mapping[str, int]], p08_summary: Mapping[str, Any], p08_intersection: Mapping[str, int]) -> dict[str, Any]:
    ids = {style: [str(row["record_id"]) for row in metadata[style]] for style in STYLE_ORDER}
    sequences = {style: [str(row["final_sequence_sha256"]) for row in metadata[style]] for style in STYLE_ORDER}
    return {
        "schema": SELECTION_SCHEMA,
        "task_id": TASK_ID,
        "status": SELECTION_STATUS,
        "created_utc": _utc_now(),
        "final_design": dict(design_record),
        "final_design_sha256": design_record["sha256"],
        "identity_inventory": dict(inventory_record),
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
        "public_sources_frozen": trr9._public_source_descriptors(source_inputs),
        "selection_rule": {
            "algorithm": "TRR-0005 deterministic future-holdout order with verified prior identity/opaque exclusions; first 128 eligible unique records per domain",
            "identity_exclusions": True,
            "source_text_or_token_ids_written": False,
            "record_ids_sha256": {style: _json_digest(ids[style]) for style in STYLE_ORDER},
            "final_sequence_sha256": {style: _json_digest(sequences[style]) for style in STYLE_ORDER},
            "records": {style: list(metadata[style]) for style in STYLE_ORDER},
        },
        "selection_exclusions": dict(exclusion_record),
        "p08_opaque_reservation": dict(p08_summary),
        "p08_postselection_intersection": dict(p08_intersection),
        "selection_diagnostics": {style: {**dict(diagnostics[style]), "selected": len(metadata[style]), "pool_size": SOURCE_RANGES[style][1] - SOURCE_RANGES[style][0]} for style in STYLE_ORDER},
        "execution": {
            "command": list(sys.argv),
            "code_commit": _git_commit(root),
            "selector_source": _record(Path(__file__), root=root, description="TRR-0010 selector source"),
            "trusted_trr0005_producer_source": _record(Path(trusted.__file__), root=root, description="trusted TRR-0005 producer source"),
            "trusted_trr0006_eligibility_source": _record(Path(eligibility.__file__), root=root, description="trusted TRR-0006 eligibility source"),
            "python": sys.executable,
            "python_version": platform.python_version(),
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


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def select_public(*, repository_root: Path, design_path: Path, inventory_path: Path, tokenizer_path: Path, pile_arrow: Sequence[Path], finance_arrow: Sequence[Path], p08_opaque: Path, output_path: Path, exclusions_output: Path, extra_exclusions: Sequence[Path] = ()) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise SelectionError(f"repository root is unavailable: {root}")
    design_record, design = _validate_design(Path(design_path).expanduser().resolve(), root=root)
    inventory_record, inventory = _validate_inventory(Path(inventory_path).expanduser().resolve(), root=root)
    trr9._validate_trr7_file_bindings(inventory, root=root)
    output_path = (root / output_path if not Path(output_path).is_absolute() else Path(output_path)).resolve()
    exclusions_output = (root / exclusions_output if not Path(exclusions_output).is_absolute() else Path(exclusions_output)).resolve()
    if output_path == exclusions_output or output_path.exists() or exclusions_output.exists():
        raise SelectionError("selection and exclusions are create-only and must differ")
    source_inputs = trr9._validate_public_inputs(inventory, pile_paths=tuple(Path(p).resolve() for p in pile_arrow), finance_paths=tuple(Path(p).resolve() for p in finance_arrow), tokenizer_path=Path(tokenizer_path).resolve())
    final_b1 = design["final_evaluation"]["exclusion_bindings"]["final_b1"]
    exclusion_values = list(extra_exclusions) + [Path(str(final_b1["path"]))]
    p04, p04_descriptor = trr9._p04_opaque()
    p06_summary, p06_source, p06_sequence = trr9._p06_opaque()
    p08_summary, p08_source, p08_sequence = trr9._p08_opaque(Path(p08_opaque).resolve(), root=root)
    tokenizer = trusted._load_tokenizer(Path(tokenizer_path).resolve())
    datasets = {"pile": trusted._load_arrow_dataset(tuple(Path(p).resolve() for p in pile_arrow)), "finance": trusted._load_arrow_dataset(tuple(Path(p).resolve() for p in finance_arrow))}
    exclusions = trusted._collect_exclusions(_extra_exclusion_paths(root, exclusion_values))
    seen_public_hashes: set[str] = set()
    seen_final_sequences: set[str] = set()
    selected: dict[str, list[Any]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    with _trr9_geometry():
        for style in STYLE_ORDER:
            selected[style], diagnostics[style] = trr9._select_domain(
                datasets[style], style=style, tokenizer=tokenizer,
                records_per_domain=EXPECTED_RECORDS_BY_DOMAIN[style], exclusions=exclusions,
                p04=p04, p06_source=p06_source, p06_sequence=p06_sequence,
                p08_source=p08_source, p08_sequence=p08_sequence,
                seen_public_hashes=seen_public_hashes, seen_final_sequences=seen_final_sequences,
            )
    metadata = trr9._selection_metadata(selected)
    p08_intersection = trr9._assert_zero_p08_intersection(metadata, p08_source=p08_source, p08_sequence=p08_sequence)
    exclusion_payload = _build_exclusions(design_record=design_record, inventory_record=inventory_record, exclusions=exclusions, p04_descriptor=p04_descriptor, p04=p04, p06_summary=p06_summary, p08_summary=p08_summary)
    exclusion_record = _write_create_only(exclusions_output, exclusion_payload, root=root, description="TRR-0010 source exclusions")
    selection_payload = _selection_payload(root=root, design_record=design_record, inventory_record=inventory_record, exclusion_record=exclusion_record, source_inputs=source_inputs, metadata=metadata, diagnostics=diagnostics, p08_summary=p08_summary, p08_intersection=p08_intersection)
    selection_record = _write_create_only(output_path, selection_payload, root=root, description="TRR-0010 source selection")
    return {"task_id": TASK_ID, "status": SELECTION_STATUS, "selection": selection_record, "exclusions": exclusion_record, "records_by_domain": dict(EXPECTED_RECORDS_BY_DOMAIN), "truth_created_or_opened": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--repository-root", type=Path, required=True)
    select.add_argument("--design", type=Path, required=True)
    select.add_argument("--inventory", type=Path, required=True)
    select.add_argument("--tokenizer", type=Path, required=True)
    select.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--p08-opaque", type=Path, required=True)
    select.add_argument("--exclusion-ledger", type=Path, action="append", default=[])
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--exclusions-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = select_public(
            repository_root=args.repository_root, design_path=args.design,
            inventory_path=args.inventory, tokenizer_path=args.tokenizer,
            pile_arrow=args.pile_arrow, finance_arrow=args.finance_arrow,
            p08_opaque=args.p08_opaque, output_path=args.output,
            exclusions_output=args.exclusions_output, extra_exclusions=args.exclusion_ledger,
        )
    except (SelectionError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-0010 selection failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
