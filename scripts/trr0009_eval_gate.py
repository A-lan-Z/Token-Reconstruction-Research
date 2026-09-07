"""Fail-closed public-output gate for TRR-0009.

The gate is deliberately truth-blind.  It validates a frozen registration,
run-manifest bindings, every prediction/timing artifact, and the current code
and input hashes before emitting a create-only public-freeze receipt.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from safetensors import safe_open

from scripts import trr0009_eval_contract as contract
from scripts import trr0009_model as model_contract


class GateError(contract.ContractError):
    """Raised when public evaluation evidence is incomplete or changed."""


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise GateError(f"repository root is unavailable: {root}")
    return root


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GateError("cannot resolve current code commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise GateError("current code commit is not a full hash")
    return value


def _record(value: Any, *, root: Path, description: str) -> dict[str, Any]:
    try:
        return contract.validate_file_record(
            value, repository_root=root, description=description, verify=True
        )
    except contract.ContractError as exc:
        raise GateError(str(exc)) from exc


def _load(path: Path, *, description: str) -> dict[str, Any]:
    try:
        return contract.load_json(path, description=description)
    except contract.ContractError as exc:
        raise GateError(str(exc)) from exc


def _same_record(actual: Mapping[str, Any], expected: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise GateError(f"{description} {key} binding changed")


def _truth_free(value: Mapping[str, Any], *, description: str) -> None:
    forbidden = (
        "truth_opened",
        "target_labels_loaded",
        "source_text_loaded",
        "source_text_written",
        "token_ids_written",
        "candidate_arrays_persisted",
    )
    if any(value.get(key) is True or value.get(key) == "true" for key in forbidden):
        raise GateError(f"{description} records forbidden truth/source access")


def _validate_public_json(binding: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    checked = _record(binding, root=root, description=description)
    value = _load(Path(checked["path"]), description=description)
    _truth_free(value, description=description)
    return checked


def _validate_observation_tensor_metadata(*, observation: Mapping[str, Any], selection: Mapping[str, Any], root: Path) -> None:
    """Verify each captured safetensor carries the frozen row and truth flags."""
    cells = contract._as_cells(observation)
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping) or not isinstance(rule.get("record_ids_sha256"), Mapping):
        raise GateError("source selection record-ID digest binding is absent")
    for cell_id in contract.CELL_ORDER:
        cell = cells[cell_id]
        style = cell_id.split("__", 1)[0]
        descriptor = cell.get("observation") if isinstance(cell.get("observation"), Mapping) else cell
        path = Path(str(descriptor.get("path"))).expanduser().resolve()
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
                keys = set(handle.keys())
        except Exception as exc:
            raise GateError(f"observation safetensor metadata is unreadable: {cell_id}") from exc
        if keys != {"activations", "attention_mask", "position_ids"}:
            raise GateError(f"observation tensor keys changed: {cell_id}")
        expected_shape = [contract.records_for_cell(observation, cell_id), contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE]
        if metadata.get("schema") != "token-reconstruction.trr0009-public-observation.v1" or metadata.get("task_id") != contract.TASK_ID or metadata.get("cell_id") != cell_id or metadata.get("shape") != json.dumps(expected_shape):
            raise GateError(f"observation safetensor identity/geometry changed: {cell_id}")
        if metadata.get("selection_plan_sha256") != str(selection.get("_record_sha256", "")):
            # The caller installs the actual selection record digest on the
            # payload copy to avoid trusting a path-only selection binding.
            raise GateError(f"observation selection binding changed: {cell_id}")
        if metadata.get("record_ids_sha256") != str(rule["record_ids_sha256"].get(style)):
            raise GateError(f"observation record-ID binding changed: {cell_id}")
        if any(metadata.get(key) != "false" for key in ("source_text_written", "token_ids_written", "target_labels_loaded", "truth_opened")):
            raise GateError(f"observation records forbidden access: {cell_id}")


def _validate_capture_inputs(*, capture_payload: Mapping[str, Any], selection_payload: Mapping[str, Any], root: Path) -> None:
    public_inputs = capture_payload.get("public_inputs")
    if not isinstance(public_inputs, Mapping):
        raise GateError("public capture input bindings are absent")
    frozen_sources = selection_payload.get("public_sources_frozen")
    if not isinstance(frozen_sources, Mapping):
        raise GateError("selection public source descriptors are absent")
    for style in ("pile", "finance"):
        actual = public_inputs.get(style)
        expected = frozen_sources.get(style)
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise GateError(f"capture {style} input descriptor is absent")
        actual_files = actual.get("arrow_files")
        expected_files = expected.get("arrow_files")
        if not isinstance(actual_files, list) or not isinstance(expected_files, list) or len(actual_files) != len(expected_files):
            raise GateError(f"capture {style} Arrow descriptor changed")
        for index, (left, right) in enumerate(zip(actual_files, expected_files)):
            _same_record(left, right, description=f"capture {style} Arrow {index}")
            _record(left, root=root, description=f"capture {style} Arrow {index}")
    actual_tokenizer = public_inputs.get("tokenizer")
    expected_tokenizer = frozen_sources.get("tokenizer")
    if not isinstance(actual_tokenizer, Mapping) or not isinstance(expected_tokenizer, Mapping) or actual_tokenizer.get("path") != expected_tokenizer.get("path"):
        raise GateError("capture tokenizer descriptor changed")
    actual_tokenizer_files = actual_tokenizer.get("files")
    expected_tokenizer_files = expected_tokenizer.get("files")
    if not isinstance(actual_tokenizer_files, Mapping) or not isinstance(expected_tokenizer_files, Mapping) or set(actual_tokenizer_files) != set(expected_tokenizer_files):
        raise GateError("capture tokenizer file binding changed")
    for name in actual_tokenizer_files:
        _same_record(actual_tokenizer_files[name], expected_tokenizer_files[name], description=f"capture tokenizer {name}")
        _record(actual_tokenizer_files[name], root=root, description=f"capture tokenizer {name}")
    model_snapshot = public_inputs.get("model_snapshot")
    if not isinstance(model_snapshot, Mapping) or not isinstance(model_snapshot.get("files"), Mapping) or not model_snapshot["files"]:
        raise GateError("capture model snapshot binding is absent")
    for name, binding in model_snapshot["files"].items():
        if not isinstance(binding, Mapping):
            raise GateError(f"capture model snapshot file binding is malformed: {name}")
        resolved = binding.get("resolved_path", binding.get("path"))
        check = dict(binding)
        check["path"] = resolved
        check.pop("resolved_path", None)
        check.pop("symlink", None)
        _record(check, root=root, description=f"capture model snapshot {name}")
    source_code = capture_payload.get("source_code")
    if not isinstance(source_code, Mapping) or not source_code:
        raise GateError("capture source-code bindings are absent")
    for name, binding in source_code.items():
        _record(binding, root=root, description=f"capture source code {name}")


def _validate_adaptable_support(*, registration: Mapping[str, Any], frequency_payload: Mapping[str, Any], root: Path) -> None:
    """Match the frozen fitting-frequency map to the adaptable state support."""
    methods = registration.get("methods")
    if not isinstance(methods, list):
        raise GateError("registration method rows are absent")
    row = next((item for item in methods if isinstance(item, Mapping) and item.get("id") == contract.ADAPTABLE_METHOD_ID), None)
    if not isinstance(row, Mapping) or not isinstance(row.get("state"), Mapping):
        raise GateError("adaptable state binding is absent")
    state_path = Path(str(row["state"]["path"])).expanduser().resolve()
    try:
        with safe_open(str(state_path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            if metadata.get("schema") != "token-reconstruction.trr0009-supported-readout.v1":
                return
            keys = set(handle.keys())
            if not {"support_ids", "support_counts"}.issubset(keys):
                raise GateError("adaptable state support tensors are absent")
            support_ids = handle.get_tensor("support_ids").to(dtype=torch.long, device="cpu").flatten().contiguous()
            support_counts = handle.get_tensor("support_counts").to(dtype=torch.long, device="cpu").flatten().contiguous()
    except GateError:
        raise
    except Exception as exc:
        if state_path.suffix == ".safetensors":
            raise GateError("adaptable state metadata/support binding is unreadable") from exc
        return
    if support_ids.numel() == 0 or support_ids.shape != support_counts.shape or support_ids.lt(0).any().item() or support_counts.le(0).any().item() or not torch.equal(support_ids, torch.sort(support_ids).values) or torch.unique(support_ids).numel() != support_ids.numel():
        raise GateError("adaptable state support geometry changed")
    raw = frequency_payload.get("frequency_references")
    raw = raw.get("enriched") if isinstance(raw, Mapping) else None
    if not isinstance(raw, Mapping):
        raise GateError("frequency reference enriched map is absent")
    pairs = sorted((int(token), int(count)) for token, count in raw.items())
    expected_ids = torch.tensor([token for token, _count in pairs], dtype=torch.long)
    expected_counts = torch.tensor([count for _token, count in pairs], dtype=torch.long)
    if not torch.equal(support_ids, expected_ids) or not torch.equal(support_counts, expected_counts):
        raise GateError(f"adaptable support/count binding differs from fitting-frequency map: state={support_ids.numel()} map={expected_ids.numel()}")
    digest = model_contract.support_digest(support_ids, support_counts)
    if metadata.get("support_digest") != digest or metadata.get("support_count") not in (None, str(int(support_ids.numel()))):
        raise GateError("adaptable support digest metadata changed")


def _validate_state_semantics(*, registration: Mapping[str, Any], root: Path) -> None:
    methods = registration.get("methods")
    if not isinstance(methods, list):
        raise GateError("registration method rows are absent")
    for row in methods:
        if not isinstance(row, Mapping) or not isinstance(row.get("state"), Mapping):
            raise GateError("registration state row is malformed")
        path = Path(str(row["state"]["path"])).expanduser().resolve()
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
        except Exception as exc:
            if path.suffix == ".safetensors":
                raise GateError(f"state metadata is unreadable: {row.get('id')}") from exc
            continue
        schema = str(metadata.get("schema", ""))
        if schema.startswith("token-reconstruction.trr0007-") or schema == "token-reconstruction.trr0009-supported-readout.v1":
            if metadata.get("current_H_only") not in (True, "true", "True") or metadata.get("full_vocabulary_cross_entropy") not in (True, "true", "True") or "current_activation_H_i_only" not in str(metadata.get("inference_contract", "")):
                raise GateError(f"state inference semantics changed: {row.get('id')}")


def _validate_bound_public_metadata(registration: Mapping[str, Any], *, root: Path, observation: Mapping[str, Any]) -> dict[str, Any]:
    selection = _validate_public_json(registration["source_selection"], root=root, description="public source selection")
    selection_payload = _load(Path(selection["path"]), description="public source selection")
    if selection_payload.get("schema") != "token-reconstruction.trr0009-source-selection.v1" or selection_payload.get("task_id") != contract.TASK_ID or selection_payload.get("status") != "FROZEN_TRR0009_SOURCE_SELECTION_NO_TRUTH":
        raise GateError("public source selection is not the frozen TRR-0009 identity ledger")
    if selection_payload.get("target_conditions") != list(contract.TARGET_ORDER) or selection_payload.get("paired_conditions") is not True:
        raise GateError("public source target pairing changed")
    if selection_payload.get("records_by_domain") != observation.get("records_by_domain"):
        raise GateError("public source selection counts differ from observations")

    capture = _validate_public_json(registration["capture_receipt"], root=root, description="public capture receipt")
    capture_payload = _load(Path(capture["path"]), description="public capture receipt")
    if capture_payload.get("schema") != "token-reconstruction.trr0009-public-capture.v1" or capture_payload.get("task_id") != contract.TASK_ID or capture_payload.get("status") != "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH":
        raise GateError("public capture receipt is not complete")
    if capture_payload.get("selection_plan") != selection:
        raise GateError("public capture selection binding changed")
    if capture_payload.get("observations") != registration.get("observation_manifest"):
        raise GateError("public capture observation binding changed")
    execution = capture_payload.get("execution")
    if not isinstance(execution, Mapping) or execution.get("truth_opened") is not False or execution.get("target_labels_loaded") is not False:
        raise GateError("public capture execution records forbidden access")
    _validate_capture_inputs(capture_payload=capture_payload, selection_payload=selection_payload, root=root)
    selection_payload = dict(selection_payload)
    selection_payload["_record_sha256"] = selection["sha256"]
    _validate_observation_tensor_metadata(observation=observation, selection=selection_payload, root=root)

    frequency = _record(registration["frequency_reference"], root=root, description="public fitting-frequency reference")
    frequency_payload = _load(Path(frequency["path"]), description="public fitting-frequency reference")
    if frequency_payload.get("schema") != "token-reconstruction.trr0005-frequency-references.v1" or frequency_payload.get("task_id") != "TRR-0005" or frequency_payload.get("status") != "PUBLIC_FITTING_FREQUENCY_REFERENCES":
        raise GateError("public fitting-frequency reference is not the frozen public map")
    maps = frequency_payload.get("frequency_references")
    metadata = frequency_payload.get("metadata")
    if not isinstance(maps, Mapping) or not isinstance(maps.get("enriched"), Mapping) or not contract.public_frequency_metadata_is_truth_free(metadata):
        raise GateError("public fitting-frequency reference records forbidden data access")
    _validate_adaptable_support(registration=registration, frequency_payload=frequency_payload, root=root)
    return {"source_selection": selection, "capture_receipt": capture, "frequency_reference": frequency}

def _validate_timing_plan(registration: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    binding = _record(registration["timing_plan"], root=root, description="timing plan")
    checked = dict(binding)
    value = _load(Path(binding["path"]), description="timing plan")
    if value.get("schema") != contract.TIMING_PLAN_SCHEMA or value.get("task_id") != contract.TASK_ID:
        raise GateError("timing plan identity/schema changed")
    if value.get("status") != "FROZEN_TIMING_PLAN_BEFORE_MEASUREMENT":
        raise GateError("timing plan is not frozen before measurement")
    _truth_free(value, description="timing plan")
    if tuple(value.get("method_order", ())) != contract.METHOD_ORDER:
        raise GateError("timing plan method order changed")
    if value.get("cell_order") != list(contract.CELL_ORDER):
        raise GateError("timing plan cell order changed")
    try:
        records = int(value["records_per_cell"])
        blocks = int(value["blocks"])
        warmups = int(value["warmup_runs"])
        measured = int(value["measured_runs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError("timing plan geometry is malformed") from exc
    if records != 32 or blocks != 40 or warmups != 1 or measured != 1:
        raise GateError("timing plan geometry or warmup/measurement settings changed")
    if value.get("timing_paths") != list(contract.METHOD_ORDER) + ["continued_fixed_readout__alias"]:
        raise GateError("timing plan path matrix changed")
    if value.get("block_cycle_length") != 10 or value.get("block_cycles") != 4:
        raise GateError("timing plan balance cycle changed")
    if float(value.get("candidate_ratio_threshold", 0.0)) != 1.25:
        raise GateError("timing plan candidate threshold changed")
    alias = value.get("alias_control")
    if not isinstance(alias, Mapping) or tuple(alias.get("band", ())) != (0.95, 1.05):
        raise GateError("timing plan alias-control band changed")
    return checked | {"value": value}


def _validate_balanced_timing(
    balanced_timing_path: Path,
    *,
    root: Path,
    registration: Mapping[str, Any],
    registration_path: Path,
    registration_record: Mapping[str, Any],
    checked_predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the separate 40-block timing receipt before truth.

    The receipt is structural/cost evidence.  An alias runtime failure or
    inconclusive CI remains a valid recorded outcome for the downstream cost
    decision; this validator only rejects changed bindings, incomplete blocks,
    or predictions that differ from the already-frozen public matrix.
    """
    timing_record = _record(
        {"path": str(Path(balanced_timing_path).expanduser().resolve()), "bytes": Path(balanced_timing_path).stat().st_size, "sha256": contract.sha256_file(Path(balanced_timing_path))},
        root=root,
        description="balanced timing receipt",
    )
    value = _load(Path(timing_record["path"]), description="balanced timing receipt")
    if value.get("schema") != "token-reconstruction.trr0009-balanced-timing.v1" or value.get("task_id") != contract.TASK_ID or value.get("status") != "TIMING_COMPLETE":
        raise GateError("balanced timing receipt identity/status changed")
    _truth_free(value, description="balanced timing receipt")
    if value.get("code_commit") != registration.get("code_commit"):
        raise GateError("balanced timing code commit differs from registration")
    _same_record(value.get("registration", {}), registration_record, description="balanced timing registration")
    _same_record(value.get("observation_manifest", {}), registration.get("observation_manifest", {}), description="balanced timing observations")
    _same_record(value.get("timing_plan", {}), registration.get("timing_plan", {}), description="balanced timing plan")
    if value.get("code_bindings") != registration.get("code_bindings"):
        raise GateError("balanced timing code bindings changed")
    if value.get("resource_guard") != registration.get("resource_guard"):
        raise GateError("balanced timing resource guard changed")
    numerical = value.get("numerical_settings")
    if not isinstance(numerical, Mapping) or numerical.get("settings") != registration.get("numerical_settings"):
        raise GateError("balanced timing numerical settings differ from runner registration")
    configuration = value.get("configuration")
    expected_configuration = {
        "records_per_cell": 32,
        "blocks": 40,
        "block_cycle_length": 10,
        "block_cycles": 4,
        "warmup_runs_per_record": 1,
        "measured_runs_per_record": 1,
        "seed": 8008,
        "maximum_seconds": 600,
        "candidate_ratio_threshold": 1.25,
        "ci_level": 0.95,
    }
    if not isinstance(configuration, Mapping) or any(configuration.get(key) != expected for key, expected in expected_configuration.items()):
        raise GateError("balanced timing geometry or CI configuration changed")
    paths = list(contract.METHOD_ORDER) + ["continued_fixed_readout__alias"]
    order_schedule = value.get("order_schedule")
    plan = _load(Path(str(registration["timing_plan"]["path"])), description="timing plan")
    if not isinstance(order_schedule, Mapping) or order_schedule.get("rows") != plan.get("schedule_rows") or order_schedule.get("sha256") != plan.get("schedule_sha256"):
        raise GateError("balanced timing order schedule changed")
    method_bindings = value.get("method_bindings")
    registration_methods = {str(row["id"]): row for row in registration.get("methods", ()) if isinstance(row, Mapping)}
    expected_bindings = {
        method_id: {"state": registration_methods[method_id]["state"], "loader": registration_methods[method_id]["loader"]}
        for method_id in contract.METHOD_ORDER
    }
    expected_bindings["continued_fixed_readout__alias"] = expected_bindings[contract.PRIMARY_CONTROL_METHOD_ID]
    if not isinstance(method_bindings, Mapping) or set(method_bindings) != set(expected_bindings):
        raise GateError("balanced timing method bindings changed")
    for method_id in paths:
        if method_bindings.get(method_id) != expected_bindings[method_id]:
            raise GateError(f"balanced timing state/loader binding changed: {method_id}")
    alias_identity = value.get("alias_execution_identity")
    if not isinstance(alias_identity, Mapping) or alias_identity.get("status") != "PASS" or alias_identity.get("exact_prediction_equivalence") is not True or alias_identity.get("identical_state_loader_binding") is not True:
        raise GateError("balanced timing alias identity/equivalence is absent")
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 40:
        raise GateError("balanced timing block count changed")
    expected_entries = {(cell_id, method_id) for cell_id in contract.CELL_ORDER for method_id in paths}
    for block_index, block in enumerate(blocks):
        if not isinstance(block, Mapping) or int(block.get("block_index", -1)) != block_index:
            raise GateError(f"balanced timing block index changed: {block_index}")
        entries = block.get("entries")
        if not isinstance(entries, list) or len(entries) != len(expected_entries):
            raise GateError(f"balanced timing block entry count changed: {block_index}")
        seen: set[tuple[str, str]] = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise GateError(f"balanced timing entry is malformed: {block_index}")
            cell_id = str(entry.get("cell_id")); method_id = str(entry.get("method_id")); pair = (cell_id, method_id)
            if pair not in expected_entries or pair in seen:
                raise GateError(f"balanced timing entry matrix changed: {block_index}/{cell_id}/{method_id}")
            seen.add(pair)
            if int(entry.get("block_index", -1)) != block_index or int(entry.get("records", -1)) != 32 or int(entry.get("warmup_runs_per_record", -1)) != 1 or int(entry.get("measured_runs_per_record", -1)) != 1:
                raise GateError(f"balanced timing entry geometry changed: {block_index}/{cell_id}/{method_id}")
            if entry.get("warmup_output_exact_match_measured") is not True or entry.get("measured_output_selected") is not True or entry.get("truth_opened") is not False or entry.get("candidate_arrays_persisted") is not False:
                raise GateError(f"balanced timing entry flags changed: {block_index}/{cell_id}/{method_id}")
            digest = entry.get("prediction_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise GateError(f"balanced timing prediction digest is malformed: {block_index}/{cell_id}/{method_id}")
            peak = entry.get("peak_memory")
            if not isinstance(peak, Mapping) or "process_max_rss_bytes" not in peak or "host_available_bytes" not in peak:
                raise GateError(f"balanced timing peak-memory receipt is incomplete: {block_index}/{cell_id}/{method_id}")
        if seen != expected_entries:
            raise GateError(f"balanced timing block matrix incomplete: {block_index}")
    try:
        from scripts import trr0009_timing as timing
        recomputed = timing.summarize_blocks(blocks)
    except Exception as exc:
        raise GateError("balanced timing summary could not be recomputed") from exc
    if value.get("summary") != recomputed:
        raise GateError("balanced timing summary changed or is not reproducible")
    output_root = Path(str(registration.get("output_root", ""))).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    expected_run_manifest_path = (output_root.resolve() / "run_manifest.json").resolve()
    expected_run_record = _record({"path": str(expected_run_manifest_path), "bytes": expected_run_manifest_path.stat().st_size, "sha256": contract.sha256_file(expected_run_manifest_path)}, root=root, description="frozen prediction run manifest for timing")
    _same_record(value.get("frozen_prediction_run_manifest", {}), expected_run_record, description="balanced timing frozen prediction run manifest")
    frozen_prefixes = value.get("frozen_prediction_prefix_digests")
    expected_prefixes: dict[str, str] = {}
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            key = f"{method_id}::{cell_id}"
            binding = checked_predictions.get(key)
            if not isinstance(binding, Mapping):
                raise GateError(f"frozen prediction binding is absent for timing: {key}")
            records = int(binding.get("records"))
            try:
                values, _metadata = contract.load_prediction_file(Path(str(binding["path"])), records=records, expected_metadata={"schema": contract.PREDICTION_SCHEMA, "task_id": contract.TASK_ID, "registration_sha256": registration_record["sha256"], "cell_id": cell_id, "method_id": method_id, "records": str(records), "truth_opened": "false", "candidate_arrays_persisted": "false"})
            except contract.ContractError as exc:
                raise GateError(f"frozen prediction could not be loaded for timing: {key}") from exc
            expected_prefixes[key] = contract.tensor_digest(values[:32].contiguous())
    if not isinstance(frozen_prefixes, Mapping) or {str(key): str(value) for key, value in frozen_prefixes.items()} != expected_prefixes:
        raise GateError("balanced timing frozen prediction-prefix bindings changed")
    for block in blocks:
        for entry in block["entries"]:
            canonical_method = contract.PRIMARY_CONTROL_METHOD_ID if entry["method_id"] == "continued_fixed_readout__alias" else str(entry["method_id"])
            key = f"{canonical_method}::{entry['cell_id']}"
            if str(entry["prediction_sha256"]) != expected_prefixes[key]:
                raise GateError(f"balanced timing prediction differs from frozen output: {block['block_index']}/{entry['method_id']}/{entry['cell_id']}")
    return timing_record


def _validate_timing_file(
    binding: Mapping[str, Any],
    *,
    root: Path,
    cell_id: str,
    method_id: str,
    records: int,
) -> dict[str, Any]:
    checked = _record(binding, root=root, description=f"timing {method_id}/{cell_id}")
    expected = contract.expected_timing_path(Path("."), cell_id=cell_id, method_id=method_id)
    # The caller checks the output-root-relative path.  This local check keeps
    # the artifact's own identity bound to its file payload.
    value = _load(Path(checked["path"]), description=f"timing {method_id}/{cell_id}")
    _truth_free(value, description=f"timing {method_id}/{cell_id}")
    if value.get("schema") != contract.TIMING_SCHEMA or value.get("task_id") != contract.TASK_ID:
        raise GateError(f"timing identity changed: {method_id}/{cell_id}")
    if value.get("method_id") != method_id or value.get("cell_id") != cell_id or int(value.get("records", -1)) != records:
        raise GateError(f"timing method/cell/record binding changed: {method_id}/{cell_id}")
    if int(value.get("warmup_runs_per_record", -1)) != 1 or int(value.get("measured_runs_per_record", -1)) != 1:
        raise GateError(f"timing warmup/measured count changed: {method_id}/{cell_id}")
    if value.get("warmup_output_exact_match_measured") is not True or value.get("measured_output_selected") is not True:
        raise GateError(f"timing output equivalence missing: {method_id}/{cell_id}")
    durations = value.get("per_record_measured_seconds")
    if not isinstance(durations, list) or len(durations) != records or any(float(x) < 0.0 for x in durations):
        raise GateError(f"timing record durations malformed: {method_id}/{cell_id}")
    for field in ("warmup_seconds_sum", "measured_seconds_sum", "model_preparation_seconds"):
        try:
            number = float(value[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise GateError(f"timing cost field is malformed: {method_id}/{cell_id}/{field}") from exc
        if number < 0.0:
            raise GateError(f"timing cost field is negative: {method_id}/{cell_id}/{field}")
    peak = value.get("peak_memory")
    if not isinstance(peak, Mapping) or "process_max_rss_bytes" not in peak or "host_available_bytes" not in peak:
        raise GateError(f"timing peak-memory receipt is incomplete: {method_id}/{cell_id}")
    artifact = value.get("prediction_artifact")
    if not isinstance(artifact, Mapping):
        raise GateError(f"timing prediction artifact is absent: {method_id}/{cell_id}")
    return checked | {"payload": value}


def _validate_prediction_file(
    binding: Mapping[str, Any],
    *,
    root: Path,
    registration_sha: str,
    cell_id: str,
    method_id: str,
    records: int,
) -> dict[str, Any]:
    checked = _record(binding, root=root, description=f"prediction {method_id}/{cell_id}")
    metadata_expected = {
        "schema": contract.PREDICTION_SCHEMA,
        "task_id": contract.TASK_ID,
        "registration_sha256": registration_sha,
        "cell_id": cell_id,
        "method_id": method_id,
        "records": str(records),
        "truth_opened": "false",
        "candidate_arrays_persisted": "false",
    }
    try:
        values, metadata = contract.load_prediction_file(
            Path(checked["path"]), records=records, expected_metadata=metadata_expected
        )
    except contract.ContractError as exc:
        raise GateError(str(exc)) from exc
    if metadata.get("geometry_json") != json.dumps(
        {"records": records, **contract.STATIC_GEOMETRY}, sort_keys=True
    ):
        raise GateError(f"prediction geometry metadata changed: {method_id}/{cell_id}")
    tensor_hash = contract.tensor_digest(values)
    return checked | {"prediction_sha256": tensor_hash, "metadata": metadata, "records": int(records)}


def _expected_key(method_id: str, cell_id: str) -> str:
    return f"{method_id}::{cell_id}"


def validate_public_outputs(
    *,
    registration_path: Path,
    run_manifest_path: Path,
    repository_root: Path,
    require_current_head: bool = True,
    balanced_timing_path: Path | None = None,
) -> dict[str, Any]:
    """Validate all public artifacts without opening truth or target labels."""

    root = _root(repository_root)
    registration_path = Path(registration_path).expanduser().resolve()
    run_manifest_path = Path(run_manifest_path).expanduser().resolve()
    registration = _load(registration_path, description="TRR-0009 registration")
    try:
        checked_registration = contract.validate_registration(
            registration, repository_root=root, verify_assets=True
        )
    except contract.ContractError as exc:
        raise GateError(str(exc)) from exc
    _validate_state_semantics(registration=registration, root=root)
    if registration.get("status") != "FROZEN_EVALUATION_REGISTRATION_BEFORE_TRUTH":
        raise GateError("registration is not frozen before truth")
    initialization = registration.get("initialization_equivalence")
    if not isinstance(initialization, Mapping) or initialization.get("status") != "PASS":
        raise GateError("initialization/harness output equivalence is missing or failed")
    if require_current_head and _git_head(root) != str(registration["code_commit"]):
        raise GateError("current code commit differs from frozen registration")

    observation = _load(Path(registration["observation_manifest"]["path"]), description="observation manifest")
    public_metadata = _validate_bound_public_metadata(registration, root=root, observation=observation)
    timing_plan = _validate_timing_plan(registration, root=root)
    registration_record = _record(
        {"path": str(registration_path), "bytes": registration_path.stat().st_size, "sha256": contract.sha256_file(registration_path)},
        root=root,
        description="registration",
    )
    registration_sha = registration_record["sha256"]
    run = _load(run_manifest_path, description="TRR-0009 run manifest")
    if run.get("schema") != contract.RUN_SCHEMA or run.get("task_id") != contract.TASK_ID:
        raise GateError("run manifest identity/schema changed")
    if run.get("status") != "PUBLIC_PREDICTIONS_COMPLETE_BEFORE_TRUTH":
        raise GateError("run manifest is not a complete pre-truth run")
    _truth_free(run, description="run manifest")
    if run.get("code_commit") != registration.get("code_commit"):
        raise GateError("run manifest code binding changed")
    nested_registration = run.get("registration")
    if not isinstance(nested_registration, Mapping):
        raise GateError("run manifest registration binding is absent")
    _same_record(nested_registration, registration_record, description="run manifest registration")
    if run.get("observation_manifest") != registration.get("observation_manifest"):
        raise GateError("run manifest observation binding changed")

    output_root = Path(str(registration.get("output_root", ""))).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    task_root = (root / "experiments" / contract.TASK_ID).resolve()
    try:
        output_root.relative_to(task_root)
    except ValueError as exc:
        raise GateError("prediction root is outside the TRR-0009 task root") from exc
    if run_manifest_path != (output_root / "run_manifest.json").resolve():
        raise GateError("run manifest is from a different prediction root")

    predictions = run.get("predictions")
    timings = run.get("timings")
    if not isinstance(predictions, Mapping) or not isinstance(timings, Mapping):
        raise GateError("run manifest prediction/timing maps are absent")
    expected_keys = {_expected_key(method_id, cell_id) for method_id in contract.METHOD_ORDER for cell_id in contract.CELL_ORDER}
    if set(predictions) != expected_keys or set(timings) != expected_keys:
        raise GateError("run manifest method-by-cell matrix is incomplete or changed")
    records_by_cell = {cell_id: contract.records_for_cell(observation, cell_id) for cell_id in contract.CELL_ORDER}
    checked_predictions: dict[str, Any] = {}
    checked_timings: dict[str, Any] = {}
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            key = _expected_key(method_id, cell_id)
            expected_prediction_path = contract.expected_prediction_path(output_root, cell_id=cell_id, method_id=method_id).resolve()
            expected_timing_path = contract.expected_timing_path(output_root, cell_id=cell_id, method_id=method_id).resolve()
            prediction_binding = predictions[key]
            timing_binding = timings[key]
            if not isinstance(prediction_binding, Mapping) or not isinstance(timing_binding, Mapping):
                raise GateError(f"run manifest artifact binding is malformed: {key}")
            if Path(str(prediction_binding.get("path", ""))).expanduser().resolve() != expected_prediction_path:
                raise GateError(f"prediction root/path changed: {key}")
            if Path(str(timing_binding.get("path", ""))).expanduser().resolve() != expected_timing_path:
                raise GateError(f"timing root/path changed: {key}")
            prediction = _validate_prediction_file(
                prediction_binding,
                root=root,
                registration_sha=registration_sha,
                cell_id=cell_id,
                method_id=method_id,
                records=records_by_cell[cell_id],
            )
            timing = _validate_timing_file(
                timing_binding,
                root=root,
                cell_id=cell_id,
                method_id=method_id,
                records=records_by_cell[cell_id],
            )
            payload_artifact = timing["payload"].get("prediction_artifact")
            if not isinstance(payload_artifact, Mapping):
                raise GateError(f"timing prediction artifact is malformed: {key}")
            _same_record(payload_artifact, prediction, description=f"timing/prediction {key}")
            if payload_artifact.get("prediction_sha256") != prediction.get("prediction_sha256"):
                raise GateError(f"timing prediction tensor binding changed: {key}")
            checked_predictions[key] = prediction
            checked_timings[key] = timing

    balanced_timing = None
    if balanced_timing_path is not None:
        balanced_timing = _validate_balanced_timing(
            balanced_timing_path,
            root=root,
            registration=registration,
            registration_path=registration_path,
            registration_record=registration_record,
            checked_predictions=checked_predictions,
        )

    freeze = {
        "schema": contract.FREEZE_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH",
        "registration": registration_record,
        "run_manifest": {"path": str(run_manifest_path), "bytes": run_manifest_path.stat().st_size, "sha256": contract.sha256_file(run_manifest_path)},
        "timing_plan": timing_plan,
        "code_commit": registration["code_commit"],
        "code_bindings": registration["code_bindings"],
        "observation_manifest": registration["observation_manifest"],
        "source_selection": registration["source_selection"],
        "capture_receipt": registration["capture_receipt"],
        "frequency_reference": public_metadata["frequency_reference"],
        "runtime_embedding": registration["runtime_embedding"],
        "output_root": str(output_root),
        "predictions": checked_predictions,
        "timings": checked_timings,
        "initialization_equivalence": initialization,
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }
    if balanced_timing is not None:
        freeze["balanced_timing"] = balanced_timing
    return freeze


def write_freeze(*, registration_path: Path, run_manifest_path: Path, output_path: Path, repository_root: Path, require_current_head: bool = True, balanced_timing_path: Path | None = None) -> dict[str, Any]:
    receipt = validate_public_outputs(
        registration_path=registration_path,
        run_manifest_path=run_manifest_path,
        repository_root=repository_root,
        require_current_head=require_current_head,
        balanced_timing_path=balanced_timing_path,
    )
    try:
        record = contract.write_create_only(output_path, receipt)
    except contract.ContractError as exc:
        raise GateError(str(exc)) from exc
    return receipt | {"freeze_receipt": record}


def validate_before_truth(*, freeze_path: Path, repository_root: Path, require_current_head: bool = True) -> dict[str, Any]:
    root = _root(repository_root)
    freeze_path = Path(freeze_path).expanduser().resolve()
    freeze = _load(freeze_path, description="public freeze receipt")
    if freeze.get("schema") != contract.FREEZE_SCHEMA or freeze.get("task_id") != contract.TASK_ID or freeze.get("status") != "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH":
        raise GateError("public freeze receipt identity/status changed")
    _truth_free(freeze, description="public freeze receipt")
    registration = _record(freeze.get("registration"), root=root, description="frozen registration")
    run = _record(freeze.get("run_manifest"), root=root, description="frozen run manifest")
    if require_current_head and _git_head(root) != str(freeze.get("code_commit")):
        raise GateError("current code commit differs from public freeze receipt")
    # Re-run the complete public gate from the bound paths.  This keeps a
    # copied, stale, or edited receipt from authorizing truth preparation.
    balanced_timing = freeze.get("balanced_timing")
    balanced_timing_path = Path(str(balanced_timing["path"])) if isinstance(balanced_timing, Mapping) and balanced_timing.get("path") else None
    refreshed = validate_public_outputs(
        registration_path=Path(registration["path"]),
        run_manifest_path=Path(run["path"]),
        repository_root=root,
        require_current_head=require_current_head,
        balanced_timing_path=balanced_timing_path,
    )
    if set(refreshed) != set(freeze):
        raise GateError("public freeze field set changed")
    for key in refreshed:
        if refreshed.get(key) != freeze.get(key):
            raise GateError(f"public freeze binding changed: {key}")
    return freeze


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--freeze-output", type=Path, required=True)
    parser.add_argument("--balanced-timing", type=Path, required=True, help="create-only balanced 40-block timing receipt")
    parser.add_argument("--allow-detached-head", action="store_true", help="only for synthetic tests")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = write_freeze(
            registration_path=args.registration,
            run_manifest_path=args.run_manifest,
            output_path=args.freeze_output,
            repository_root=args.repository_root,
            require_current_head=not args.allow_detached_head,
            balanced_timing_path=args.balanced_timing,
        )
    except Exception as exc:
        print(f"TRR-0009 public gate failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "freeze_receipt": result["freeze_receipt"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
