"""Build the fail-closed TRR-0010 registration and capture wiring.

This adapter consumes only already-frozen identity/observation metadata. It
does not select rows, read public Arrow data, load a tokenizer or model, run
capture, or open truth. Selection metadata is checked through the existing
TRR-0009 identity-only loader; the eventual capture owner must package the
lower-level TRR-0009/TRR-0005 producer output under the TRR-0010 schema.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from scripts import trr0009_eval_capture as trr9_capture
from scripts import trr0010_eval_gate as gate


DESIGN_SCHEMA = "token-reconstruction.trr0010-final-evaluation-design.v1"
DESIGN_STATUS = "FROZEN_TRR0010_FINAL_EVALUATION_DESIGN"
SELECTION_BINDING_SCHEMA = "token-reconstruction.trr0010-source-selection-binding.v1"
SELECTION_BINDING_STATUS = "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH"
PANEL_SCHEMA = "token-reconstruction.trr0010-public-source-panel.v1"
PANEL_STATUS = "FROZEN_TRR0010_SOURCE_PANEL_NO_TRUTH"
OBSERVATION_MANIFEST_SCHEMA = "token-reconstruction.trr0010-public-observation-manifest.v1"
OBSERVATION_MANIFEST_STATUS = "FROZEN_TRR0010_PUBLIC_OBSERVATIONS_NO_TRUTH"
CAPTURE_SCHEMA = "token-reconstruction.trr0010-public-capture.v1"
CAPTURE_STATUS = "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH"
REGISTRATION_OUTPUT_ROOT = Path("experiments/TRR-0010/evaluation")
SELECTION_BINDING_DEFAULT = Path("experiments/TRR-0010/evaluation/source_selection_binding.json")
REGISTRATION_DEFAULT = Path("experiments/TRR-0010/evaluation/registration.json")
FREQUENCY_BANK_ORDER = ("B0", "B1")


class RegisterError(ValueError):
    """Raised when the final TRR-0010 registration cannot be built safely."""


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RegisterError(f"repository root is unavailable: {root}")
    return root


def _record(path: Path, *, root: Path, description: str, readonly: bool | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if readonly is None:
        try:
            path.relative_to(root)
            readonly = False
        except ValueError:
            readonly = True
    try:
        return gate.file_record(path, root=root, readonly=bool(readonly))
    except gate.GateError as exc:
        raise RegisterError(str(exc)) from exc


def _binding(value: Any, *, root: Path, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegisterError(f"{description} binding is malformed")
    try:
        return gate._record(value, root=root, description=description)  # noqa: SLF001
    except gate.GateError as exc:
        raise RegisterError(str(exc)) from exc


def _json(path: Path, *, root: Path, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(path, root=root, description=description)
    try:
        payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegisterError(f"{description} is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RegisterError(f"{description} must be a JSON object")
    return record, dict(payload)


def _normalize_frequency_bank(raw: Any, *, bank_id: str) -> dict[int, int]:
    """Validate one public fitting-support map without opening source data."""
    if not isinstance(raw, Mapping):
        raise RegisterError(f"frequency reference bank {bank_id!r} is not a map")
    normalized: dict[int, int] = {}
    for token, count in raw.items():
        if isinstance(token, bool) or isinstance(count, bool):
            raise RegisterError(f"frequency reference bank {bank_id!r} contains a boolean entry")
        try:
            token_id = int(token)
            value = int(count)
        except (TypeError, ValueError) as exc:
            raise RegisterError(f"frequency reference bank {bank_id!r} contains a non-integer entry") from exc
        if isinstance(token, float) and token != token_id:
            raise RegisterError(f"frequency reference bank {bank_id!r} contains a non-integral token")
        if isinstance(count, float) and count != value:
            raise RegisterError(f"frequency reference bank {bank_id!r} contains a non-integral count")
        if not 0 <= token_id < gate.VOCABULARY_SIZE or value <= 0:
            raise RegisterError(f"frequency reference bank {bank_id!r} contains an invalid token/count")
        if token_id in normalized:
            raise RegisterError(f"frequency reference bank {bank_id!r} contains duplicate token IDs")
        normalized[token_id] = value
    if not normalized:
        raise RegisterError(f"frequency reference bank {bank_id!r} is empty")
    return normalized


def _frequency_reference_bindings(
    *,
    root: Path,
    frequency_reference_path: Path | None,
    frequency_reference_paths: Mapping[str, Path] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Bind both named B0/B1 maps before the public freeze.

    The scorer consumes independent registration keys for both support maps.
    A single JSON file may contain both named maps, or callers may provide one
    file per bank. In either case the map contents are checked now and only
    file records plus canonical support-map digests are emitted.
    """
    if frequency_reference_path is not None and frequency_reference_paths is not None:
        raise RegisterError("use frequency_reference_path or frequency_reference_paths, not both")
    if frequency_reference_paths is None:
        if frequency_reference_path is None:
            raise RegisterError("both B0 and B1 frequency references are required")
        paths = {bank: Path(frequency_reference_path) for bank in FREQUENCY_BANK_ORDER}
    else:
        if set(frequency_reference_paths) != set(FREQUENCY_BANK_ORDER):
            raise RegisterError("frequency_reference_paths must bind exactly B0 and B1")
        paths = {bank: Path(frequency_reference_paths[bank]) for bank in FREQUENCY_BANK_ORDER}

    bindings: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {
        "bank_order": list(FREQUENCY_BANK_ORDER),
        "binding_names": {bank: f"frequency_reference_{bank}" for bank in FREQUENCY_BANK_ORDER},
        "all_methods_each_bank": True,
        "method_order": list(gate.METHOD_ORDER),
        "banks": {},
    }
    for bank in FREQUENCY_BANK_ORDER:
        record, payload = _json(paths[bank], root=root, description=f"frequency reference {bank}")
        _truth_free(payload, description=f"frequency reference {bank}")
        raw_references = payload.get("frequency_references")
        if not isinstance(raw_references, Mapping) or bank not in raw_references:
            raise RegisterError(f"frequency reference does not contain named {bank} support map")
        selected = raw_references[bank]
        if isinstance(selected, Mapping) and isinstance(selected.get("enriched"), Mapping):
            selected = selected["enriched"]
        elif isinstance(selected, Mapping) and isinstance(selected.get("counts"), Mapping):
            selected = selected["counts"]
        normalized = _normalize_frequency_bank(selected, bank_id=bank)
        canonical = json.dumps(
            [[int(token), int(count)] for token, count in sorted(normalized.items())],
            separators=(",", ":"),
        ).encode("utf-8")
        bindings[f"frequency_reference_{bank}"] = record
        metadata["banks"][bank] = {
            "binding_name": f"frequency_reference_{bank}",
            "file": dict(record),
            "support_token_count": len(normalized),
            "support_map_sha256": hashlib.sha256(canonical).hexdigest(),
        }
    return bindings, metadata


def _truth_free(payload: Mapping[str, Any], *, description: str) -> None:
    forbidden = (
        "truth_opened",
        "truth_created",
        "source_text_written",
        "source_text_loaded",
        "token_ids_written",
        "target_labels_loaded",
        "candidate_arrays_persisted",
        "private_or_truth_payload_read",
        "fresh_evaluation_started",
    )
    for key in forbidden:
        value = payload.get(key)
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            raise RegisterError(f"{description} records forbidden access: {key}")


def _same_record(left: Mapping[str, Any], right: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(left.get(key)) != str(right.get(key)):
            raise RegisterError(f"{description} {key} binding changed")


def _write_create_only(path: Path, payload: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to((root / "experiments" / gate.TASK_ID).resolve())
    except ValueError as exc:
        raise RegisterError(f"{description} must remain under the TRR-0010 task root") from exc
    if path.exists() or path.is_symlink():
        raise RegisterError(f"{description} is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise RegisterError(f"{description} could not be written") from exc
    return _record(path, root=root, description=description)


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RegisterError("cannot resolve current code commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RegisterError("current code commit is not a full hash")
    return value


def _require_design(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    record, design = _json(path, root=root, description="TRR-0010 final evaluation design")
    _truth_free(design, description="final evaluation design")
    if design.get("schema") != DESIGN_SCHEMA or design.get("task_id") != gate.TASK_ID:
        raise RegisterError("final evaluation design schema or task identity changed")
    code_commit = design.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40 or any(char not in "0123456789abcdef" for char in code_commit):
        raise RegisterError("final evaluation design must bind a full code commit")
    if design.get("status") != DESIGN_STATUS:
        raise RegisterError(
            "registration requires the owner-frozen TRR-0010 final evaluation design; "
            f"current status is {design.get('status')!r}"
        )
    methods = design.get("methods")
    if (
        not isinstance(methods, Mapping)
        or set(methods) != set(gate.METHOD_ORDER)
        or any(methods.get(method_id) != gate.METHOD_ROLES[method_id] for method_id in gate.METHOD_ORDER)
    ):
        raise RegisterError("final evaluation design does not freeze all six contender roles")
    if list(design.get("method_order", ())) != list(gate.METHOD_ORDER):
        raise RegisterError("final evaluation design method order changed")
    decision_rules = design.get("decision_rules")
    if (
        not isinstance(decision_rules, Mapping)
        or decision_rules.get("status") != "FROZEN"
        or not isinstance(decision_rules.get("rules"), Mapping)
        or not decision_rules["rules"]
    ):
        raise RegisterError("final evaluation decision rules are absent or not frozen")
    final = design.get("final_evaluation")
    if not isinstance(final, Mapping):
        raise RegisterError("final evaluation design section is absent")
    if dict(final.get("records_by_domain", {})) != dict(gate.RECORDS_BY_DOMAIN):
        raise RegisterError("final evaluation domain counts differ from 128 per domain")
    if list(final.get("target_conditions", ())) != list(gate.TARGET_ORDER):
        raise RegisterError("final evaluation target pairing changed")
    if final.get("paired_sources_across_targets") is not True:
        raise RegisterError("final evaluation does not freeze source pairing across P0/LoRA")
    capture = final.get("capture_geometry")
    if (
        not isinstance(capture, Mapping)
        or capture.get("batch_records") != 8
        or capture.get("sequence_tokens") != 192
        or capture.get("stored_sequence_tokens") != 128
        or capture.get("retain_first_128") is not True
    ):
        raise RegisterError("final evaluation does not freeze inherited B8x192 -> first-128 capture geometry")
    exclusions = final.get("exclusion_bindings")
    if not isinstance(exclusions, Mapping):
        raise RegisterError("final evaluation exclusion bindings are absent")
    final_b1 = _binding(exclusions.get("final_b1"), root=root, description="final B1 exclusion ledger")
    opaque = exclusions.get("approved_opaque_ledgers")
    if not isinstance(opaque, Sequence) or isinstance(opaque, (str, bytes, bytearray)) or not opaque:
        raise RegisterError("approved opaque exclusion ledgers are absent")
    opaque_records = [
        _binding(value, root=root, description=f"approved opaque exclusion ledger {index}")
        for index, value in enumerate(opaque)
    ]
    required_code = design.get("code_bindings")
    if not isinstance(required_code, Mapping) or not required_code:
        raise RegisterError("final evaluation code bindings are absent")
    code_records = {
        str(name): _binding(value, root=root, description=f"final evaluation code {name}")
        for name, value in required_code.items()
    }
    required_suffixes = {
        "scripts/trr0010_eval_gate.py",
        "scripts/trr0010_eval_runner.py",
        "scripts/trr0009_eval_capture.py",
        "scripts/trr0005_produce_confirmation.py",
        "src/token_reconstruction/public_activation.py",
        "scripts/trr0004_predict_confirmation.py",
        "scripts/trr0003_footing_compare.py",
    }
    paths = {str(Path(value["path"]).resolve()) for value in code_records.values()}
    if not all(any(path.endswith(suffix) for path in paths) for suffix in required_suffixes):
        raise RegisterError("final evaluation code bindings omit a required runner, producer, or A1+A2 source")
    return record, design, {
        "final_b1": final_b1,
        "approved_opaque_ledgers": opaque_records,
        "code_bindings": code_records,
        "decision_rules": dict(decision_rules),
        "capture_geometry": dict(capture),
    }


def _load_selection(path: Path, *, root: Path, final_b1: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        selection, selection_record, rows, counts = trr9_capture.load_selection(
            path,
            repository_root=root,
            expected_counts=gate.RECORDS_BY_DOMAIN,
        )
    except Exception as exc:
        raise RegisterError(f"TRR-0009 identity-only selection is not compatible: {exc}") from exc
    if selection.get("target_conditions") != list(gate.TARGET_ORDER) or selection.get("paired_conditions") is not True:
        raise RegisterError("selection target pairing changed")
    if dict(counts) != dict(gate.RECORDS_BY_DOMAIN):
        raise RegisterError("selection must contain exactly 128 Finance and 128 Pile records")
    selection_exclusions = selection.get("selection_exclusions")
    if not isinstance(selection_exclusions, Mapping):
        raise RegisterError("selection does not bind its final B1 exclusion ledger")
    exclusion_record = _binding(selection_exclusions, root=root, description="selection final B1 exclusions")
    _same_record(exclusion_record, final_b1, description="selection/final B1 exclusion")
    row_rule = selection.get("selection_rule")
    if not isinstance(row_rule, Mapping):
        raise RegisterError("selection rule is absent")
    record_digests = row_rule.get("record_ids_sha256")
    if not isinstance(record_digests, Mapping) or set(record_digests) != set(gate.DOMAIN_ORDER):
        raise RegisterError("selection record-order digests are absent")
    if any(not isinstance(value, str) or len(value) != 64 for value in record_digests.values()):
        raise RegisterError("selection record-order digest is malformed")
    return selection, selection_record, {
        "rows": rows,
        "counts": dict(counts),
        "record_ids_sha256": {str(k): str(v) for k, v in record_digests.items()},
    }


def _selection_binding_payload(
    *,
    selection: Mapping[str, Any],
    selection_record: Mapping[str, Any],
    selection_meta: Mapping[str, Any],
    final_b1: Mapping[str, Any],
    opaque_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SELECTION_BINDING_SCHEMA,
        "task_id": gate.TASK_ID,
        "status": SELECTION_BINDING_STATUS,
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "target_conditions": list(gate.TARGET_ORDER),
        "paired_sources_across_targets": True,
        "record_ids_sha256": dict(selection_meta["record_ids_sha256"]),
        "selection_ledger": dict(selection_record),
        "selection_ledger_schema": selection.get("schema"),
        "selection_ledger_status": selection.get("status"),
        "selection_exclusions": dict(final_b1),
        "approved_opaque_exclusion_ledgers": [dict(record) for record in opaque_records],
        "source_text_written": False,
        "source_text_loaded": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }


def _observation_bindings(
    path: Path,
    *,
    root: Path,
    selection_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observation_record, manifest = _json(path, root=root, description="TRR-0010 observation manifest")
    _truth_free(manifest, description="TRR-0010 observation manifest")
    if manifest.get("schema") != OBSERVATION_MANIFEST_SCHEMA or manifest.get("task_id") != gate.TASK_ID:
        raise RegisterError("observation manifest schema or task identity changed")
    if manifest.get("status") != OBSERVATION_MANIFEST_STATUS:
        raise RegisterError("observation manifest is not frozen before truth")
    manifest_record_ids = manifest.get("record_ids_sha256")
    if not isinstance(manifest_record_ids, Mapping) or set(manifest_record_ids) != set(gate.DOMAIN_ORDER):
        raise RegisterError("observation manifest record-order digests are absent")
    if dict(manifest.get("records_by_domain", {})) != dict(gate.RECORDS_BY_DOMAIN):
        raise RegisterError("observation domain counts changed")
    if list(manifest.get("cell_order", ())) != list(gate.CELL_ORDER):
        raise RegisterError("observation cell order changed")
    if manifest.get("selection_plan") != selection_record:
        raise RegisterError("observation manifest selection binding changed")
    cells = manifest.get("cells")
    if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes, bytearray)):
        rows = {str(row.get("cell_id")): row for row in cells if isinstance(row, Mapping)}
    elif isinstance(cells, Mapping):
        rows = {str(key): value for key, value in cells.items() if isinstance(value, Mapping)}
    else:
        raise RegisterError("observation manifest cells are absent")
    if set(rows) != set(gate.CELL_ORDER):
        raise RegisterError("observation manifest cells are incomplete or foreign")
    checked: dict[str, Any] = {}
    for cell_id in gate.CELL_ORDER:
        row = rows[cell_id]
        domain = cell_id.split("__", 1)[0]
        if row.get("cell_id") != cell_id or row.get("records") != gate.RECORDS_PER_CELL:
            raise RegisterError(f"observation cell identity/count changed: {cell_id}")
        if row.get("shape") != [gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS, gate.OBSERVATION_HIDDEN_SIZE]:
            raise RegisterError(f"observation shape changed: {cell_id}")
        if row.get("record_ids_sha256") != manifest_record_ids[domain]:
            raise RegisterError(f"observation source order changed: {cell_id}")
        observation = _binding(row.get("observation"), root=root, description=f"observation {cell_id}")
        checked[cell_id] = {
            "cell_id": cell_id,
            "records": gate.RECORDS_PER_CELL,
            "shape": list(row["shape"]),
            "record_ids_sha256": str(row["record_ids_sha256"]),
            "observation": observation,
        }
    return observation_record, manifest, checked


def _panel(
    path: Path,
    *,
    root: Path,
    selection_record: Mapping[str, Any],
    observation_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    panel_record, panel = _json(path, root=root, description="TRR-0010 source panel")
    _truth_free(panel, description="TRR-0010 source panel")
    if panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != gate.TASK_ID:
        raise RegisterError("source panel schema or task identity changed")
    if panel.get("status") != PANEL_STATUS:
        raise RegisterError("source panel is not frozen before truth")
    if dict(panel.get("records_by_domain", {})) != dict(gate.RECORDS_BY_DOMAIN):
        raise RegisterError("source panel counts changed")
    if list(panel.get("cell_order", ())) != list(gate.CELL_ORDER):
        raise RegisterError("source panel cell order changed")
    if panel.get("same_sources_across_targets") is not True or panel.get("public_material_only") is not True:
        raise RegisterError("source panel pairing/public-only boundary changed")
    if panel.get("selection_plan") != selection_record:
        raise RegisterError("source panel selection binding changed")
    if panel.get("observation_manifest") != observation_record:
        raise RegisterError("source panel observation binding changed")
    record_ids = panel.get("record_ids_sha256")
    if not isinstance(record_ids, Mapping) or set(record_ids) != set(gate.DOMAIN_ORDER):
        raise RegisterError("source panel record-order digests are absent")
    if dict(record_ids) != dict(panel.get("observation_record_ids_sha256", record_ids)):
        raise RegisterError("source panel record-order metadata is inconsistent")
    return panel_record, panel


def _capture(
    path: Path,
    *,
    root: Path,
    selection_record: Mapping[str, Any],
    observation_record: Mapping[str, Any],
    panel_record: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_record, capture = _json(path, root=root, description="TRR-0010 capture receipt")
    _truth_free(capture, description="TRR-0010 capture receipt")
    if capture.get("schema") != CAPTURE_SCHEMA or capture.get("task_id") != gate.TASK_ID:
        raise RegisterError("capture receipt schema or task identity changed")
    if capture.get("status") != CAPTURE_STATUS:
        raise RegisterError("capture receipt is not complete before truth")
    if capture.get("records_by_domain") != gate.RECORDS_BY_DOMAIN:
        raise RegisterError("capture record counts changed")
    if capture.get("selection_plan") != selection_record:
        raise RegisterError("capture selection binding changed")
    if capture.get("observations") != observation_record:
        raise RegisterError("capture observation binding changed")
    if capture.get("panel") != panel_record:
        raise RegisterError("capture panel binding changed")
    actual_geometry = capture.get("geometry")
    if not isinstance(actual_geometry, Mapping):
        raise RegisterError("capture geometry is absent")
    for key, expected in geometry.items():
        if actual_geometry.get(key) != expected:
            raise RegisterError(f"capture geometry changed: {key}")
    return capture_record, capture


def _method_rows(
    method_rows: Mapping[str, Any],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    if not isinstance(method_rows, Mapping) or set(method_rows) != set(gate.METHOD_ORDER):
        raise RegisterError("method rows do not bind all six contenders")
    result: list[dict[str, Any]] = []
    for method_id in gate.METHOD_ORDER:
        row = method_rows[method_id]
        if not isinstance(row, Mapping) or row.get("id") != method_id or row.get("role") != gate.METHOD_ROLES[method_id]:
            raise RegisterError(f"method row identity/role changed: {method_id}")
        state = _binding(row.get("state"), root=root, description=f"state {method_id}")
        loader = row.get("loader")
        if not isinstance(loader, Mapping):
            raise RegisterError(f"loader is absent: {method_id}")
        loader = dict(loader)
        resources = row.get("resources")
        if not isinstance(resources, Mapping):
            raise RegisterError(f"resources are absent: {method_id}")
        required = gate.METHOD_RESOURCE_REQUIREMENTS[method_id]
        if not set(required).issubset(resources):
            raise RegisterError(f"required state/readout resources are absent: {method_id}")
        checked_resources = {
            str(name): _binding(resources[name], root=root, description=f"{method_id} resource {name}")
            for name in required
        }
        if "base_decoder_state" in checked_resources:
            _same_record(checked_resources["base_decoder_state"], state, description=f"{method_id} base decoder/state")
        if method_id == gate.A1_A2_METHOD_ID:
            if loader.get("interface") != gate.A1_A2_LOADER_INTERFACE or loader.get("candidate_k") != 256:
                raise RegisterError("A1+A2 loader semantics are not fixed K=256")
            path_args = loader.get("path_args")
            if not isinstance(path_args, Mapping) or not all(isinstance(path_args.get(key), Mapping) for key in ("snapshot", "reference_path")):
                raise RegisterError("A1+A2 loader must bind snapshot and reference path descriptors")
        else:
            if loader.get("interface") != gate.LOADER_INTERFACE:
                raise RegisterError(f"loader interface changed: {method_id}")
            if not isinstance(loader.get("module"), str) or not isinstance(loader.get("function"), str):
                raise RegisterError(f"production loader module/function is absent: {method_id}")
            if any(loader.get(key) is not expected for key, expected in {
                "current_h_only": True,
                "full_vocabulary": True,
                "history_enabled": False,
                "a2_enabled": False,
            }.items()):
                raise RegisterError(f"current-H loader semantics changed: {method_id}")
        result.append({
            "id": method_id,
            "role": gate.METHOD_ROLES[method_id],
            "cells": list(gate.CELL_ORDER),
            "records_per_cell": {cell: gate.RECORDS_PER_CELL for cell in gate.CELL_ORDER},
            "state": state,
            "loader": loader,
            "resources": checked_resources,
        })
    return result


def build_registration(
    *,
    repository_root: Path,
    design_path: Path,
    selection_path: Path,
    panel_path: Path,
    observation_manifest_path: Path,
    capture_path: Path,
    timing_plan_path: Path,
    method_rows: Mapping[str, Any],
    frequency_reference_path: Path | None = None,
    frequency_reference_paths: Mapping[str, Path] | None = None,
    output_root: str | Path = REGISTRATION_OUTPUT_ROOT,
    output_path: Path = REGISTRATION_DEFAULT,
    selection_binding_path: Path = SELECTION_BINDING_DEFAULT,
) -> dict[str, Any]:
    root = _root(repository_root)
    design_record, design, design_meta = _require_design(Path(design_path), root=root)
    selection, selection_record, selection_meta = _load_selection(
        Path(selection_path),
        root=root,
        final_b1=design_meta["final_b1"],
    )
    selection_binding = _selection_binding_payload(
        selection=selection,
        selection_record=selection_record,
        selection_meta=selection_meta,
        final_b1=design_meta["final_b1"],
        opaque_records=design_meta["approved_opaque_ledgers"],
    )
    selection_binding_record = _write_create_only(
        Path(selection_binding_path),
        selection_binding,
        root=root,
        description="TRR-0010 source-selection binding",
    )
    observation_record, manifest, observations = _observation_bindings(
        Path(observation_manifest_path),
        root=root,
        selection_record=selection_binding_record,
    )
    panel_record, panel = _panel(
        Path(panel_path),
        root=root,
        selection_record=selection_binding_record,
        observation_record=observation_record,
    )
    capture_record, capture = _capture(
        Path(capture_path),
        root=root,
        selection_record=selection_binding_record,
        observation_record=observation_record,
        panel_record=panel_record,
        geometry=design_meta["capture_geometry"],
    )
    timing_record = _record(Path(timing_plan_path), root=root, description="TRR-0010 timing plan")
    frequency_bindings, frequency_metadata = _frequency_reference_bindings(
        root=root,
        frequency_reference_path=Path(frequency_reference_path) if frequency_reference_path is not None else None,
        frequency_reference_paths=frequency_reference_paths,
    )
    methods = _method_rows(method_rows, root=root)
    current_head = _git_head(root)
    code_records = design_meta["code_bindings"]
    declared_head = design.get("code_commit")
    if declared_head is not None and declared_head != current_head:
        raise RegisterError("final evaluation design code commit is not current")
    output = Path(output_root).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    task_root = (root / "experiments" / gate.TASK_ID).resolve()
    try:
        output.relative_to(task_root)
    except ValueError as exc:
        raise RegisterError("registration output root must be below the TRR-0010 task root") from exc
    registration = {
        "schema": gate.REGISTRATION_SCHEMA,
        "task_id": gate.TASK_ID,
        "status": gate.REGISTRATION_STATUS,
        "repository_root": str(root),
        "code_commit": current_head,
        "code_bindings": code_records,
        "contract_binding": design_record,
        "input_bindings": {
            "source_selection": selection_binding_record,
            "panel": panel_record,
            "public_observations": observation_record,
            "capture": capture_record,
            **frequency_bindings,
            "final_b1_exclusions": design_meta["final_b1"],
            **{
                f"approved_opaque_exclusion_{index}": value
                for index, value in enumerate(design_meta["approved_opaque_ledgers"])
            },
        },
        "observation_bindings": observations,
        "timing_plan": timing_record,
        "frequency_scoring": frequency_metadata,
        "method_order": list(gate.METHOD_ORDER),
        "cell_order": list(gate.CELL_ORDER),
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "geometry": dict(gate.STATIC_GEOMETRY),
        "methods": methods,
        "producer_wiring": {
            "selection_metadata_loader": "scripts.trr0009_eval_capture.load_selection",
            "capture_entrypoint": "scripts.trr0009_eval_capture.capture_public",
            "capture_prefix_producer": "scripts.trr0009_eval_capture._capture_condition_with_producer",
            "public_forward": "token_reconstruction.public_activation.capture_public_prefix",
            "batch_records": 8,
            "capture_sequence_tokens": 192,
            "stored_sequence_tokens": 128,
            "retain_first_128": True,
            "truth_curator_separate": True,
            "note": "TRR-0009 producer output must be repackaged under the TRR-0010 schema before this registration is accepted",
        },
        "decision_rules": design_meta["decision_rules"],
        "numerical_settings": dict(design.get("numerical_settings", {})),
        "resource_guard": dict(design.get("resource_guard", {})),
        "initialization_equivalence": {"status": "PASS"},
        "truth_opened": False,
        "source_text_written": False,
        "source_text_loaded": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "a2_enabled": False,
        "history_enabled": False,
    }
    registration_record = _write_create_only(
        Path(output_path),
        registration,
        root=root,
        description="TRR-0010 evaluation registration",
    )
    return registration | {"registration_file": registration_record}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True, help="JSON arguments for build_registration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RegisterError("payload must be an object")
        build_registration(**dict(payload))
    except (OSError, json.JSONDecodeError, RegisterError) as exc:
        print(f"TRR-0010 registration failed closed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
