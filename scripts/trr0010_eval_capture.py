#!/usr/bin/env python3
"""TRR-0010 public-capture bridge.

The trusted TRR-0009 producer remains the only model/tokenizer/LoRA capture
implementation.  This module supplies the task-local metadata bridge: it
creates a short-lived TRR-0009 producer-selection ledger, invokes the trusted
producer when explicitly requested, and repackages its truth-free receipts
under the TRR-0010 schemas.  It never writes source text, token IDs, labels,
or truth.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import trr0009_eval_capture as trr9_capture
from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_register as register
from scripts import trr0010_select_public as selector


TASK_ID = "TRR-0010"
TRR9_TASK_ID = "TRR-0009"
TRR9_SELECTION_SCHEMA = "token-reconstruction.trr0009-source-selection.v1"
TRR9_SELECTION_STATUS = "FROZEN_TRR0009_SOURCE_SELECTION_NO_TRUTH"
OBSERVATION_SCHEMA = register.OBSERVATION_MANIFEST_SCHEMA
OBSERVATION_STATUS = register.OBSERVATION_MANIFEST_STATUS
PANEL_SCHEMA = register.PANEL_SCHEMA
PANEL_STATUS = register.PANEL_STATUS
CAPTURE_SCHEMA = register.CAPTURE_SCHEMA
CAPTURE_STATUS = register.CAPTURE_STATUS


class CaptureAdapterError(RuntimeError):
    """Raised when the producer-to-TRR10 bridge cannot fail closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _exception_chain(exc: BaseException) -> list[dict[str, str]]:
    """Return the complete chained exception context for fail-closed receipts."""
    chain: list[dict[str, str]] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    return chain


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CaptureAdapterError(f"repository root is unavailable: {root}")
    return root


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
        raise CaptureAdapterError(str(exc)) from exc


def _json(path: Path, *, root: Path, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(path, root=root, description=description)
    try:
        value = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureAdapterError(f"{description} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise CaptureAdapterError(f"{description} must be a JSON object")
    return record, dict(value)


def _write_create_only(path: Path, payload: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    task_root = (root / "experiments" / TASK_ID / "evaluation").resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise CaptureAdapterError(f"{description} must remain below {task_root}") from exc
    if path.exists() or path.is_symlink():
        raise CaptureAdapterError(f"{description} is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise CaptureAdapterError(f"could not write {description}") from exc
    return _record(path, root=root, description=description)


def _port_observation_tensor(
    *,
    source_path: Path,
    output_path: Path,
    metadata: Mapping[str, str],
    expected_shape: Sequence[int],
    root: Path,
    description: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy a producer observation while replacing only its file metadata.

    The trusted TRR-0009 producer writes valid public tensors with TRR-0009
    headers. TRR-0010's unchanged gate intentionally requires its own task
    schema/task identity in every tensor header, so the bridge must create a
    new file. Tensor keys, shapes, dtypes, and every tensor value are checked
    before the port is accepted; the native source descriptor remains bound
    in the output manifest for provenance.
    """
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise CaptureAdapterError(f"{description} output is create-only: {output_path}")
    try:
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != gate.OBSERVATION_KEYS:
                raise CaptureAdapterError(f"{description} tensor keys changed: {sorted(keys)}")
            source_tensors = {key: handle.get_tensor(key) for key in sorted(keys)}
            expected_headers = {
                "activations": (list(expected_shape), torch.bfloat16),
                "attention_mask": ([int(expected_shape[0]), int(expected_shape[1])], torch.uint8),
                "position_ids": ([int(expected_shape[0]), int(expected_shape[1])], torch.int64),
            }
            for key, (shape, dtype) in expected_headers.items():
                tensor = source_tensors[key]
                if list(tensor.shape) != shape or tensor.dtype != dtype:
                    raise CaptureAdapterError(
                        f"{description} tensor header changed: {key} "
                        f"shape={list(tensor.shape)!r} dtype={tensor.dtype!s}"
                    )
            source_metadata = dict(handle.metadata() or {})
    except CaptureAdapterError:
        raise
    except Exception as exc:
        raise CaptureAdapterError(f"{description} source tensor is unreadable") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_file(source_tensors, str(output_path), metadata=dict(metadata))
    except Exception as exc:
        raise CaptureAdapterError(f"{description} could not write TRR-0010 tensor port") from exc

    try:
        with safe_open(str(output_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(source_tensors):
                raise CaptureAdapterError(f"{description} output tensor keys changed")
            for key, source_tensor in source_tensors.items():
                output_tensor = handle.get_tensor(key)
                if list(output_tensor.shape) != list(source_tensor.shape) or output_tensor.dtype != source_tensor.dtype:
                    raise CaptureAdapterError(f"{description} output tensor header changed: {key}")
                if not torch.equal(output_tensor, source_tensor):
                    raise CaptureAdapterError(f"{description} output tensor values changed: {key}")
            output_metadata = dict(handle.metadata() or {})
    except CaptureAdapterError:
        raise
    except Exception as exc:
        raise CaptureAdapterError(f"{description} output tensor is unreadable") from exc

    expected_metadata = {str(key): str(value) for key, value in metadata.items()}
    if output_metadata != expected_metadata:
        raise CaptureAdapterError(f"{description} output metadata changed")
    native_record = _record(source_path, root=root, description=f"{description} native source")
    port_record = _record(output_path, root=root, description=f"{description} TRR-0010 port")
    return native_record, port_record


def _truth_free(payload: Mapping[str, Any], *, description: str) -> None:
    for key in (
        "truth_opened", "truth_created", "source_text_written", "source_text_loaded",
        "token_ids_written", "target_labels_loaded", "candidate_arrays_persisted",
        "private_or_truth_payload_read", "fresh_evaluation_started",
    ):
        value = payload.get(key)
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            raise CaptureAdapterError(f"{description} records forbidden access: {key}")


def _producer_selection_payload(selection: Mapping[str, Any], selection_record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the metadata-only ledger expected by the unchanged TRR9 producer."""
    return {
        "schema": TRR9_SELECTION_SCHEMA,
        "task_id": TRR9_TASK_ID,
        "status": TRR9_SELECTION_STATUS,
        "created_utc": _utc_now(),
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "target_conditions": list(gate.TARGET_ORDER),
        "paired_conditions": True,
        "selection_seed": selection.get("selection_seed"),
        "source_ranges_half_open": selection.get("source_ranges_half_open"),
        "sequence_tokens_including_bos": selector.SEQUENCE_TOKENS,
        "scored_post_bos_tokens": selector.SCORED_POST_BOS_TOKENS,
        "capture_batch_records": selector.CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": selector.CAPTURE_SEQUENCE_TOKENS,
        "hidden_size": selector.HIDDEN_SIZE,
        "public_sources_frozen": selection.get("public_sources_frozen"),
        "selection_rule": selection.get("selection_rule"),
        "selection_exclusions": selection.get("selection_exclusions"),
        "trr0010_selection": dict(selection_record),
        "source_text_or_target_labels": False,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "truth_opened": False,
        "truth_created": False,
    }


def write_producer_selection_bridge(*, selection_path: Path, output_path: Path, repository_root: Path) -> dict[str, Any]:
    """Create the explicit TRR9 metadata bridge consumed by its producer."""
    root = _root(repository_root)
    selection, selection_record, _rows, _counts = selector.load_selection(selection_path, repository_root=root, expected_counts=gate.RECORDS_BY_DOMAIN)
    payload = _producer_selection_payload(selection, selection_record)
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise CaptureAdapterError(f"producer selection bridge is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise CaptureAdapterError("could not write producer selection bridge") from exc
    return _record(path, root=root, description="TRR9 producer selection bridge")


def _validate_producer_receipts(*, producer_root: Path, producer_selection_record: Mapping[str, Any], selection: Mapping[str, Any], selection_record: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], ...]:
    observation_record, observation = _json(producer_root / "observations.json", root=root, description="TRR9 observation manifest")
    panel_record, panel = _json(producer_root / "panel.json", root=root, description="TRR9 source panel")
    capture_record, capture = _json(producer_root / "capture.json", root=root, description="TRR9 capture receipt")
    for name, payload in (("TRR9 observation manifest", observation), ("TRR9 panel", panel), ("TRR9 capture receipt", capture)):
        _truth_free(payload, description=name)
        if payload.get("task_id") != TRR9_TASK_ID:
            raise CaptureAdapterError(f"{name} task identity changed")
    if observation.get("schema") != trr9_capture.contract.OBSERVATION_SCHEMA or observation.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise CaptureAdapterError("TRR9 observation manifest is not complete and truth-free")
    if panel.get("schema") != trr9_capture.PANEL_SCHEMA or panel.get("status") != "FROZEN_SOURCE_PANEL_NO_TRUTH":
        raise CaptureAdapterError("TRR9 source panel is not complete and truth-free")
    if capture.get("schema") != trr9_capture.CAPTURE_SCHEMA or capture.get("status") != trr9_capture.CAPTURE_STATUS:
        raise CaptureAdapterError("TRR9 capture receipt is not complete and truth-free")
    for name, payload in (("observation", observation), ("panel", panel), ("capture", capture)):
        if payload.get("selection_plan") != producer_selection_record:
            raise CaptureAdapterError(f"{name} producer selection bridge binding changed")
    if dict(observation.get("records_by_domain", {})) != dict(gate.RECORDS_BY_DOMAIN) or dict(panel.get("records_by_domain", {})) != dict(gate.RECORDS_BY_DOMAIN) or dict(capture.get("records_by_domain", {})) != dict(gate.RECORDS_BY_DOMAIN):
        raise CaptureAdapterError("TRR9 producer counts are not 128 per domain")
    if list(observation.get("cell_order", ())) != list(gate.CELL_ORDER) or list(panel.get("cell_order", ())) != list(gate.CELL_ORDER):
        raise CaptureAdapterError("TRR9 producer cell order changed")
    record_digests = observation.get("record_ids_sha256")
    if not isinstance(record_digests, Mapping):
        # Native TRR9 receipts place this immutable source-order binding under
        # source_pairing and repeat it on every cell.  Accept that native
        # schema only when the producer explicitly attests paired ordering;
        # the cell-level checks below still bind every observation to the
        # frozen TRR10 selection.
        source_pairing = observation.get("source_pairing")
        paired = source_pairing.get("same_record_ids_across_targets") if isinstance(source_pairing, Mapping) else None
        paired_digests = source_pairing.get("record_ids_sha256") if isinstance(source_pairing, Mapping) else None
        if paired is not True or not isinstance(paired_digests, Mapping):
            raise CaptureAdapterError("TRR9 producer source-order digests are absent")
        record_digests = paired_digests
    if set(record_digests) != set(gate.DOMAIN_ORDER) or any(
        not isinstance(value, str) or len(value) != 64 for value in record_digests.values()
    ):
        raise CaptureAdapterError("TRR9 producer source-order digests are malformed")
    record_digests = {str(key): str(value) for key, value in record_digests.items()}
    # Downstream TRR10 repackaging consumes the normalized map; this updates
    # only the in-memory receipt and never rewrites the native producer file.
    observation["record_ids_sha256"] = dict(record_digests)
    selection_digests = selection.get("selection_rule", {}).get("record_ids_sha256")
    if dict(record_digests) != dict(selection_digests):
        raise CaptureAdapterError("producer source-order digest differs from TRR10 selection")
    panel_digests = panel.get("record_ids_sha256")
    if not isinstance(panel_digests, Mapping) or dict(panel_digests) != dict(record_digests):
        raise CaptureAdapterError("TRR9 panel source-order digest differs from observation manifest")
    cells = observation.get("cells")
    if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes, bytearray)):
        rows = {str(row.get("cell_id")): row for row in cells if isinstance(row, Mapping)}
    elif isinstance(cells, Mapping):
        rows = {str(key): value for key, value in cells.items() if isinstance(value, Mapping)}
    else:
        raise CaptureAdapterError("TRR9 observation cells are absent")
    if set(rows) != set(gate.CELL_ORDER):
        raise CaptureAdapterError("TRR9 observation cells are incomplete")
    for cell_id in gate.CELL_ORDER:
        row = rows[cell_id]
        domain = cell_id.split("__", 1)[0]
        if row.get("records") != gate.RECORDS_PER_CELL or row.get("record_ids_sha256") != record_digests[domain]:
            raise CaptureAdapterError(f"TRR9 observation source binding changed: {cell_id}")
        observation_descriptor = row.get("observation")
        if not isinstance(observation_descriptor, Mapping):
            raise CaptureAdapterError(f"TRR9 observation descriptor is absent: {cell_id}")
        if observation_descriptor.get("shape") != [gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS, gate.OBSERVATION_HIDDEN_SIZE]:
            raise CaptureAdapterError(f"TRR9 observation geometry changed: {cell_id}")
        actual_record = _record(
            Path(str(observation_descriptor.get("path", ""))),
            root=root,
            description=f"TRR9 observation {cell_id}",
        )
        # The producer descriptor is part of the immutable input binding. It
        # is not enough to hash whatever happens to be at the path now and
        # then write that fresh record into the TRR10 receipt: a changed H
        # payload would otherwise be silently rebound during repackaging.
        for key in ("path", "bytes", "sha256"):
            if observation_descriptor.get(key) != actual_record.get(key):
                raise CaptureAdapterError(
                    f"TRR9 observation descriptor changed at {cell_id}: {key}"
                )
    execution = capture.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("producer_semantics") != "public full forward B8x192; retain first 128 positions"
    ):
        raise CaptureAdapterError("TRR9 capture does not explicitly attest first-128 retention")
    conditions = capture.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(gate.TARGET_ORDER):
        raise CaptureAdapterError("TRR9 capture condition receipts are incomplete")
    for condition in gate.TARGET_ORDER:
        condition_receipt = conditions.get(condition)
        condition_cells = condition_receipt.get("cells") if isinstance(condition_receipt, Mapping) else None
        if not isinstance(condition_cells, Mapping):
            raise CaptureAdapterError(f"TRR9 {condition} cell receipts are absent")
        for style in gate.DOMAIN_ORDER:
            cell_id = f"{style}__{condition}"
            cell_receipt = condition_cells.get(cell_id)
            if not isinstance(cell_receipt, Mapping):
                raise CaptureAdapterError(f"TRR9 retention receipt is absent: {cell_id}")
            if cell_receipt.get("full_forward_retained_only_first_128") is not True:
                raise CaptureAdapterError(f"TRR9 producer did not attest first-128 retention: {cell_id}")
            if cell_receipt.get("capture_batch_records") != 8 or cell_receipt.get("capture_sequence_tokens") != 192 or cell_receipt.get("stored_sequence_tokens") != 128:
                raise CaptureAdapterError(f"TRR9 retention geometry changed: {cell_id}")
            if cell_receipt.get("observation") != rows[cell_id].get("observation"):
                raise CaptureAdapterError(f"TRR9 retention receipt observation binding changed: {cell_id}")
    geometry = capture.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("capture_batch_records") != 8 or geometry.get("capture_sequence_tokens") != 192 or geometry.get("stored_sequence_tokens") != 128:
        raise CaptureAdapterError("TRR9 capture geometry is not B8x192 retaining first 128")
    if panel.get("observation_manifest") != observation_record:
        raise CaptureAdapterError("TRR9 panel observation binding changed")
    if capture.get("observations") != observation_record:
        raise CaptureAdapterError("TRR9 capture observation binding changed")
    # Native TRR9 capture receipts bind observations directly and omit a
    # redundant panel child; if a panel child is present, it must still match
    # the independently validated panel receipt.
    capture_panel = capture.get("panel")
    if capture_panel is not None and capture_panel != panel_record:
        raise CaptureAdapterError("TRR9 capture panel binding changed")
    return observation_record, observation, panel_record, panel, capture_record, capture


def repackage_trr0009_capture(*, selection_path: Path, producer_root: Path, output_root: Path, repository_root: Path, producer_selection_path: Path, selection_binding_path: Path | None = None, design_path: Path | None = None) -> dict[str, Any]:
    """Port completed TRR9 producer outputs into immutable TRR10 artifacts.

    The producer's four safetensors files are copied under ``output_root``
    with TRR10 metadata headers.  The tensor keys, shapes, dtypes, and values
    are checked exactly during each copy; native descriptors remain in the
    manifest so the port cannot silently rebind captured inputs.
    """
    root = _root(repository_root)
    selection, selection_record, _rows, _counts = selector.load_selection(
        selection_path, repository_root=root, expected_counts=gate.RECORDS_BY_DOMAIN
    )
    bridge_record = _record(producer_selection_path, root=root, description="TRR9 producer selection bridge")
    if selection_binding_path is not None:
        if design_path is None:
            raise CaptureAdapterError("design_path is required when preparing a selection binding")
        selection_binding_record = register.prepare_selection_binding(
            repository_root=root,
            design_path=design_path,
            selection_path=selection_path,
            selection_binding_path=selection_binding_path,
        )
    else:
        selection_binding_record = selection_record
    observation_record, observation, panel_record, panel, capture_record, capture = _validate_producer_receipts(
        producer_root=Path(producer_root).expanduser().resolve(),
        producer_selection_record=bridge_record,
        selection=selection,
        selection_record=selection_record,
        root=root,
    )
    cells_raw = observation.get("cells")
    cells = (
        {str(row.get("cell_id")): row for row in cells_raw if isinstance(row, Mapping)}
        if isinstance(cells_raw, Sequence) and not isinstance(cells_raw, (str, bytes, bytearray))
        else {str(key): value for key, value in cells_raw.items()}
    )
    output = Path(output_root).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    port_observation_root = output / "observations"
    repack_cells: list[dict[str, Any]] = []
    port_equivalence: list[dict[str, Any]] = []
    for cell_id in gate.CELL_ORDER:
        row = cells[cell_id]
        observation_descriptor = row.get("observation")
        if not isinstance(observation_descriptor, Mapping):
            raise CaptureAdapterError(f"TRR9 observation descriptor is absent: {cell_id}")
        shape = list(observation_descriptor.get("shape", ()))
        if shape != [gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS, gate.OBSERVATION_HIDDEN_SIZE]:
            raise CaptureAdapterError(f"TRR9 observation geometry changed: {cell_id}")
        domain = cell_id.split("__", 1)[0]
        record_digest = str(row["record_ids_sha256"])
        native_path = Path(str(observation_descriptor["path"])).expanduser().resolve()
        port_path = port_observation_root / f"{cell_id}.safetensors"
        metadata = {
            "schema": gate.OBSERVATION_SCHEMA,
            "task_id": TASK_ID,
            "cell_id": cell_id,
            "records": str(gate.RECORDS_PER_CELL),
            "shape": json.dumps(shape),
            "record_ids_sha256": record_digest,
            "truth_opened": "false",
            "source_text_written": "false",
            "source_text_loaded": "false",
            "token_ids_written": "false",
            "target_labels_loaded": "false",
            "capture_batch_records": "8",
            "capture_sequence_tokens": "192",
            "selection_plan_sha256": str(selection_binding_record["sha256"]),
        }
        native_tensor_record, port_tensor_record = _port_observation_tensor(
            source_path=native_path,
            output_path=port_path,
            metadata=metadata,
            expected_shape=shape,
            root=root,
            description=f"TRR9 observation tensor {cell_id}",
        )
        if native_tensor_record != {
            key: observation_descriptor[key] for key in ("path", "bytes", "sha256")
        }:
            raise CaptureAdapterError(f"TRR9 observation descriptor changed during port: {cell_id}")
        repack_cells.append({
            "cell_id": cell_id,
            "records": gate.RECORDS_PER_CELL,
            "shape": shape,
            "record_ids_sha256": record_digest,
            "observation": port_tensor_record,
            "native_observation": native_tensor_record,
            "tensor_port": {
                "status": "PASS_EXACT_VALUES_KEYS_SHAPES_DTYPES",
                "native": native_tensor_record,
                "ported": port_tensor_record,
            },
        })
        port_equivalence.append({
            "cell_id": cell_id,
            "native": native_tensor_record,
            "ported": port_tensor_record,
            "keys": sorted(gate.OBSERVATION_KEYS),
            "shape": shape,
            "values_exact": True,
        })
    observation_payload = {
        "schema": OBSERVATION_SCHEMA,
        "task_id": TASK_ID,
        "status": OBSERVATION_STATUS,
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "cell_order": list(gate.CELL_ORDER),
        "record_ids_sha256": dict(observation["record_ids_sha256"]),
        "cells": repack_cells,
        "selection_plan": dict(selection_binding_record),
        "producer": {
            "task_id": TRR9_TASK_ID,
            "observation_manifest": observation_record,
            "selection_bridge": bridge_record,
            "native_tensor_payloads_preserved": True,
            "port_schema": gate.OBSERVATION_SCHEMA,
        },
        "native_observation_port": {
            "status": "PASS_EXACT_TENSOR_PAYLOAD_NO_TRUTH",
            "source_schema": trr9_capture.contract.OBSERVATION_SCHEMA,
            "source_task_id": TRR9_TASK_ID,
            "ported_schema": gate.OBSERVATION_SCHEMA,
            "ported_task_id": TASK_ID,
            "cells": port_equivalence,
        },
        "sequence_tokens_including_bos": gate.STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": gate.SCORED_POST_BOS_TOKENS,
        "capture_batch_records": 8,
        "capture_sequence_tokens": 192,
        "hidden_size": gate.OBSERVATION_HIDDEN_SIZE,
        "source_pairing": {"same_record_ids_across_targets": True, "record_ids_sha256": dict(observation["record_ids_sha256"])},
        "public_material_only": True,
        "source_text_loaded": False,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "truth_opened": False,
    }
    observation_out = _write_create_only(output / "observations.json", observation_payload, root=root, description="TRR10 observation manifest")
    panel_payload = {
        "schema": PANEL_SCHEMA,
        "task_id": TASK_ID,
        "status": PANEL_STATUS,
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "cell_order": list(gate.CELL_ORDER),
        "record_ids_sha256": dict(observation["record_ids_sha256"]),
        "observation_record_ids_sha256": dict(observation["record_ids_sha256"]),
        "selection_plan": dict(selection_binding_record),
        "observation_manifest": dict(observation_out),
        "producer": {"task_id": TRR9_TASK_ID, "panel": panel_record},
        "same_sources_across_targets": True,
        "public_material_only": True,
        "native_observation_port": {"status": "PASS_EXACT_TENSOR_PAYLOAD_NO_TRUTH", "manifest": dict(observation_out)},
        "truth_opened": False,
    }
    panel_out = _write_create_only(output / "panel.json", panel_payload, root=root, description="TRR10 source panel")
    capture_payload = {
        "schema": CAPTURE_SCHEMA,
        "task_id": TASK_ID,
        "status": CAPTURE_STATUS,
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "selection_plan": dict(selection_binding_record),
        "observations": dict(observation_out),
        "panel": dict(panel_out),
        "geometry": {
            "batch_records": 8,
            "sequence_tokens": 192,
            "stored_sequence_tokens": 128,
            "vocabulary_size": gate.VOCABULARY_SIZE,
            "capture_batch_records": 8,
            "capture_sequence_tokens": 192,
            "scored_post_bos_tokens": 127,
            "hidden_size": gate.OBSERVATION_HIDDEN_SIZE,
            "cells": len(gate.CELL_ORDER),
            "retain_first_128": True,
        },
        "producer": {"task_id": TRR9_TASK_ID, "capture": capture_record, "selection_bridge": bridge_record},
        "execution": {
            "adapter": "TRR10 producer-to-schema tensor port",
            "producer_semantics": "public full forward B8x192; retain first 128 positions",
            "tensor_port": "exact tensor keys/shapes/dtypes/values; TRR10 metadata headers",
            "source_text_written": False,
            "token_ids_written": False,
            "target_labels_loaded": False,
            "truth_opened": False,
        },
        "source_text_loaded": False,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "truth_opened": False,
    }
    capture_out = _write_create_only(output / "capture.json", capture_payload, root=root, description="TRR10 capture receipt")
    return {
        "task_id": TASK_ID,
        "status": CAPTURE_STATUS,
        "observation_manifest": observation_out,
        "panel": panel_out,
        "capture": capture_out,
        "producer": {"observation": observation_record, "panel": panel_record, "capture": capture_record, "selection_bridge": bridge_record},
        "tensor_port": {"status": "PASS_EXACT_VALUES_KEYS_SHAPES_DTYPES", "cells": port_equivalence},
        "truth_opened": False,
    }


def capture_public(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise CaptureAdapterError("TRR-0010 capture requires explicit --execute")
    root = _root(args.repository_root)
    bridge_record = write_producer_selection_bridge(selection_path=args.selection, output_path=args.producer_selection, repository_root=root)
    producer_args = argparse.Namespace(
        execute=True, capture="capture", repository_root=root, selection=Path(bridge_record["path"]),
        tokenizer=args.tokenizer, pile_arrow=args.pile_arrow, finance_arrow=args.finance_arrow,
        model_snapshot=args.model_snapshot, lora_config=args.lora_config, lora_update=args.lora_update,
        output_root=args.producer_output_root, device=args.device,
        # The trusted producer keeps its TRR-0009 default scope.  This
        # explicit reviewed scope is the only cross-task exception and is
        # checked again by the producer before it creates any output.
        allowed_output_root=root / "experiments" / TASK_ID / "evaluation",
    )
    try:
        trr9_capture.capture_public(producer_args)
    except Exception as exc:
        # Preserve a task-local, truth-free failure even when the producer
        # fails before it can write observations/capture.json.  The chained
        # exception is included in both the receipt and CLI exception so a
        # wrapper cannot hide the concrete producer defect.
        chain = _exception_chain(exc)
        try:
            output = Path(args.output_root).expanduser()
            if not output.is_absolute():
                output = root / output
            failure_payload = {
                "schema": "token-reconstruction.trr0010-public-capture-failure.v1",
                "task_id": TASK_ID,
                "status": "PUBLIC_OBSERVATIONS_CAPTURE_FAILED_NO_TRUTH",
                "started_utc": _utc_now(),
                "ended_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "exception_chain": chain,
                "producer_output_root": str(Path(args.producer_output_root).expanduser().resolve()),
                "producer_output_scope": str((root / "experiments" / TASK_ID / "evaluation").resolve()),
                "selection_bridge": dict(bridge_record),
                "truth_opened": False,
                "source_text_written": False,
                "token_ids_written": False,
                "target_labels_loaded": False,
            }
            _write_create_only(output / "failure.json", failure_payload, root=root, description="TRR10 capture failure receipt")
        except Exception:
            # The original producer exception remains the authoritative
            # failure when a second create-only receipt cannot be written.
            pass
        detail = json.dumps(chain, sort_keys=True)
        raise CaptureAdapterError(
            f"trusted TRR9 producer failed; no TRR10 package was written; exception_chain={detail}"
        ) from exc
    return repackage_trr0009_capture(selection_path=args.selection, producer_root=args.producer_output_root, output_root=args.output_root, repository_root=root, producer_selection_path=Path(bridge_record["path"]), selection_binding_path=args.selection_binding, design_path=args.design)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", default="capture")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--selection-binding", type=Path, required=True)
    parser.add_argument("--producer-selection", type=Path, required=True)
    parser.add_argument("--producer-output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    parser.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path)
    parser.add_argument("--lora-update", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.capture != "capture":
        print("TRR-0010 capture requires the capture command", file=sys.stderr)
        return 2
    try:
        result = capture_public(args)
    except (CaptureAdapterError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-0010 capture failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
