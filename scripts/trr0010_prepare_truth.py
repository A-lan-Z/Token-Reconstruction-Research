#!/usr/bin/env python3
"""TRR-0010 freeze-first truth curator boundary.

The public freeze is revalidated before the supplied curator callback is
allowed to materialize labels.  Only after the callback creates the sealed
sidecar is the task-local descriptor written and validated.  This module does
not select sources, route by outcomes, or open the private truth payload.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors.torch import save_file

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_capture_public as trr6_capture
from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_register as register
from scripts import trr0010_select_public as selector


TASK_ID = "TRR-0010"
TRUTH_SCHEMA = register.TRUTH_BINDING_SCHEMA
TRUTH_STATUS = register.TRUTH_BINDING_STATUS


class TruthPreparationError(RuntimeError):
    """Raised when freeze-first truth preparation cannot proceed safely."""


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise TruthPreparationError(f"repository root is unavailable: {root}")
    return root


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        path.relative_to(root)
        readonly = False
    except ValueError:
        readonly = True
    try:
        return gate.file_record(path, root=root, readonly=readonly)
    except gate.GateError as exc:
        raise TruthPreparationError(str(exc)) from exc


def _truth_sidecar_record(path: Path, *, root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise TruthPreparationError(f"curator did not create a regular truth sidecar: {path}")
    return _record(path, root=root, description="sealed TRR-0010 truth sidecar")


def write_truth_sidecar(
    tensors: Mapping[str, torch.Tensor],
    path: Path,
    *,
    repository_root: Path,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write independent, shape-checked tensors for a curator callback.

    The caller is responsible for obtaining the labels under the charter's
    truth rules.  Cloning each tensor prevents shared-storage serialization
    failures and makes the sidecar payload deterministic with respect to the
    supplied values.
    """
    root = _root(repository_root)
    expected_keys = [f"{cell}__token_ids" for cell in gate.CELL_ORDER]
    if list(tensors) != expected_keys:
        raise TruthPreparationError("truth tensor key order differs from the frozen four-cell contract")
    checked: dict[str, torch.Tensor] = {}
    for key in expected_keys:
        value = torch.as_tensor(tensors[key]).detach().cpu().contiguous().clone()
        if value.dtype not in (torch.int64, torch.int32):
            raise TruthPreparationError(f"truth tensor dtype changed: {key}")
        if tuple(value.shape) != (gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS):
            raise TruthPreparationError(f"truth tensor shape changed: {key}")
        if bool((value < 0).any().item()) or bool((value >= gate.VOCABULARY_SIZE).any().item()):
            raise TruthPreparationError(f"truth tensor contains an invalid token ID: {key}")
        checked[key] = value.to(torch.int64)
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        raise TruthPreparationError(f"truth sidecar is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_metadata = {
        "schema": "token-reconstruction.trr0010-truth-sidecar.v1",
        "task_id": TASK_ID,
        "truth_opened": "false",
        "records_by_domain": json.dumps(dict(gate.RECORDS_BY_DOMAIN), sort_keys=True, separators=(",", ":")),
        "cell_order": json.dumps(list(gate.CELL_ORDER), separators=(",", ":")),
        "sequence_tokens": str(gate.STORED_SEQUENCE_TOKENS),
        "vocabulary_size": str(gate.VOCABULARY_SIZE),
    }
    if metadata is not None:
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TruthPreparationError("truth sidecar metadata must contain string keys and values")
            if key in sidecar_metadata and sidecar_metadata[key] != value:
                raise TruthPreparationError(f"truth sidecar metadata conflicts with reserved key: {key}")
            sidecar_metadata[key] = value
    try:
        save_file(checked, str(path), metadata=sidecar_metadata)
    except Exception as exc:
        raise TruthPreparationError("could not serialize truth sidecar") from exc
    return _truth_sidecar_record(path.resolve(), root=root)


def _bound_path(value: Any, *, root: Path, description: str) -> Path:
    """Resolve a previously gate-checked file binding without accepting a new path."""
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise TruthPreparationError(f"{description} binding is absent")
    path = Path(str(value["path"])).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise TruthPreparationError(f"{description} binding is unavailable: {path}")
    return path


def _source_descriptor_paths(selection: Mapping[str, Any], *, root: Path, style: str) -> tuple[Path, ...]:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get(style) if isinstance(sources, Mapping) else None
    files = descriptor.get("arrow_files") if isinstance(descriptor, Mapping) else None
    if not isinstance(files, list) or not files:
        raise TruthPreparationError(f"frozen selection has no {style} Arrow descriptors")
    paths: list[Path] = []
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise TruthPreparationError(f"frozen selection {style} Arrow descriptor is malformed")
        path = Path(str(item["path"])).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file() or path.is_symlink():
            raise TruthPreparationError(f"frozen selection {style} Arrow input is unavailable: {path}")
        paths.append(path)
    return tuple(paths)


def _source_tokenizer_path(selection: Mapping[str, Any], *, root: Path) -> Path:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise TruthPreparationError("frozen selection tokenizer descriptor is absent")
    path = Path(str(descriptor["path"])).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_dir() or path.is_symlink():
        raise TruthPreparationError(f"frozen selection tokenizer is unavailable: {path}")
    return path


def materialize_public_truth(freeze: Mapping[str, Any], sidecar_path: Path) -> dict[str, Any]:
    """Materialize the four paired public labels from the frozen selection.

    This is the production callback used by the CLI.  The caller invokes it
    only after :func:`gate.validate_before_truth` succeeds.  It consumes the
    already frozen TRR-0010 identity ledger, verifies its Arrow/tokenizer
    descriptors through the trusted TRR6 helpers, renders rows with the
    reviewed TRR5 renderer, and writes only the four independent tensors.
    """
    root = _ROOT.resolve()
    inputs = freeze.get("input_bindings")
    if not isinstance(inputs, Mapping):
        raise TruthPreparationError("public freeze input bindings are absent")
    selection_path = _bound_path(inputs.get("source_selection"), root=root, description="frozen source selection")
    try:
        selection, selection_record, rows, counts = selector.load_selection(
            selection_path,
            repository_root=root,
            expected_counts=gate.RECORDS_BY_DOMAIN,
        )
    except Exception as exc:
        raise TruthPreparationError("frozen TRR-0010 selection could not be loaded") from exc
    if counts != dict(gate.RECORDS_BY_DOMAIN):
        raise TruthPreparationError("frozen selection counts changed before truth materialization")
    pile_paths = _source_descriptor_paths(selection, root=root, style="pile")
    finance_paths = _source_descriptor_paths(selection, root=root, style="finance")
    tokenizer_path = _source_tokenizer_path(selection, root=root)
    try:
        trr6_capture._validate_source_descriptors(
            selection,
            pile_paths=pile_paths,
            finance_paths=finance_paths,
            tokenizer_path=tokenizer_path,
        )
        tokenizer = trusted._load_tokenizer(tokenizer_path)
        datasets = {
            "pile": trusted._load_arrow_dataset(pile_paths),
            "finance": trusted._load_arrow_dataset(finance_paths),
        }
        rendered = trr6_capture._materialize_selected(rows, datasets=datasets, tokenizer=tokenizer)
    except Exception as exc:
        raise TruthPreparationError("trusted public row materialization failed") from exc

    labels: dict[str, torch.Tensor] = {}
    for style in ("pile", "finance"):
        records = rendered.get(style)
        declared = rows.get(style)
        if not isinstance(records, list) or not isinstance(declared, list) or len(records) != gate.RECORDS_BY_DOMAIN[style]:
            raise TruthPreparationError(f"materialized {style} rows do not match the frozen panel")
        actual_ids = [str(record.record_id) for record in records]
        declared_ids = [str(row["record_id"]) for row in declared]
        if actual_ids != declared_ids:
            raise TruthPreparationError(f"materialized {style} row order differs from the frozen selection")
        values = torch.tensor(
            [list(record.token_ids[: gate.STORED_SEQUENCE_TOKENS]) for record in records],
            dtype=torch.long,
        ).contiguous()
        if tuple(values.shape) != (gate.RECORDS_BY_DOMAIN[style], gate.STORED_SEQUENCE_TOKENS):
            raise TruthPreparationError(f"materialized {style} labels have the wrong shape")
        if values[:, 0].ne(gate.BOS_TOKEN_ID).any().item() or values.lt(0).any().item() or values.ge(gate.VOCABULARY_SIZE).any().item():
            raise TruthPreparationError(f"materialized {style} labels failed BOS/vocabulary checks")
        labels[style] = values

    tensors = {
        f"{cell_id}__token_ids": labels[cell_id.split("__", 1)[0]].clone()
        for cell_id in gate.CELL_ORDER
    }
    write_truth_sidecar(
        tensors,
        sidecar_path,
        repository_root=root,
        metadata={
            "selection_sha256": str(selection_record["sha256"]),
            "source_materializer": "trusted_trr0005_renderer+trr0006_selected_rows",
        },
    )
    return {
        "schema": "token-reconstruction.trr0010-public-truth-materializer.v1",
        "selection": dict(selection_record),
        "source_descriptors": dict(selection.get("public_sources_frozen", {})),
        "records_by_domain": dict(counts),
        "cell_order": list(gate.CELL_ORDER),
        "truth_tensor_keys": [f"{cell}__token_ids" for cell in gate.CELL_ORDER],
        "private_truth_opened": False,
        "source_text_loaded_for_public_label_materialization": True,
    }


def _build_descriptor(*, freeze: Mapping[str, Any], payload_record: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    observations = freeze.get("observation_bindings")
    if not isinstance(observations, Mapping):
        raise TruthPreparationError("public freeze has no observation bindings")
    cells = []
    for cell_id in gate.CELL_ORDER:
        row = observations.get(cell_id)
        if not isinstance(row, Mapping):
            raise TruthPreparationError(f"public freeze observation binding is absent: {cell_id}")
        cells.append({"cell_id": cell_id, "records": gate.RECORDS_PER_CELL, "record_ids_sha256": row.get("record_ids_sha256")})
    input_bindings = freeze.get("input_bindings")
    if not isinstance(input_bindings, Mapping):
        raise TruthPreparationError("public freeze input bindings are absent")
    return {
        "schema": TRUTH_SCHEMA,
        "task_id": TASK_ID,
        "status": TRUTH_STATUS,
        "truth_opened": False,
        "prepared_after_public_freeze": True,
        "registration": dict(freeze["registration"]),
        "run_manifest": dict(freeze["run_manifest"]),
        "contract_binding": dict(freeze["contract_binding"]),
        "input_bindings": {str(key): dict(value) for key, value in input_bindings.items() if isinstance(value, Mapping)},
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "cell_order": list(gate.CELL_ORDER),
        "target_conditions": list(gate.TARGET_ORDER),
        "labels_shared_across_target_conditions": True,
        "truth_shape": [gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS],
        "truth_tensor_keys": [f"{cell}__token_ids" for cell in gate.CELL_ORDER],
        "cells": cells,
        "truth_payload": dict(payload_record),
        "curator": {
            "freeze_revalidated_before_materialization": True,
            "descriptor_validation_after_materialization": True,
            "truth_payload_loaded_by_adapter": False,
        },
    }


def prepare_truth_after_freeze(*, freeze_path: Path, truth_sidecar_path: Path, descriptor_path: Path, repository_root: Path, materializer: Callable[[Mapping[str, Any], Path], Any], require_current_head: bool = False) -> dict[str, Any]:
    """Run the only permitted order: gate, materialize, descriptor, validate."""
    root = _root(repository_root)
    sidecar = Path(truth_sidecar_path).expanduser()
    if not sidecar.is_absolute():
        sidecar = root / sidecar
    sidecar = sidecar.resolve()
    if sidecar.exists() or sidecar.is_symlink():
        raise TruthPreparationError(f"truth sidecar is create-only: {sidecar}")
    try:
        freeze = gate.validate_before_truth(freeze_path=Path(freeze_path), repository_root=root, require_current_head=require_current_head)
    except gate.GateError as exc:
        raise TruthPreparationError(f"public freeze did not pass immediately before truth: {exc}") from exc
    if not callable(materializer):
        raise TruthPreparationError("a callable truth materializer is required")
    try:
        materializer(freeze, sidecar)
    except Exception as exc:
        raise TruthPreparationError("truth materializer failed; descriptor was not created") from exc
    payload_record = _truth_sidecar_record(sidecar, root=root)
    descriptor = _build_descriptor(freeze=freeze, payload_record=payload_record, repository_root=root)
    try:
        descriptor_record = register._write_create_only(Path(descriptor_path), descriptor, root=root, description="TRR-0010 truth descriptor")
    except register.RegisterError as exc:
        raise TruthPreparationError(str(exc)) from exc
    # This is deliberately after sidecar creation; the old runbook ordering
    # validated a descriptor before the curator was allowed to create it.
    try:
        checked = register.validate_truth_descriptor(Path(descriptor_record["path"]), repository_root=root, freeze=freeze)
    except register.RegisterError as exc:
        raise TruthPreparationError(f"created truth descriptor failed post-materialization validation: {exc}") from exc
    return {"task_id": TASK_ID, "status": TRUTH_STATUS, "descriptor": descriptor_record, "truth_payload": checked["truth_payload"], "truth_opened": False, "materializer_called_after_public_gate": True}


def _load_callable(spec: str) -> Callable[[Mapping[str, Any], Path], Any]:
    if ":" not in spec:
        raise TruthPreparationError("materializer must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise TruthPreparationError(f"cannot import truth materializer {spec}") from exc
    if not callable(function):
        raise TruthPreparationError(f"truth materializer is not callable: {spec}")
    return function


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepare", nargs="?", default="prepare")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--truth-sidecar", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument(
        "--materializer",
        default="scripts.trr0010_prepare_truth:materialize_public_truth",
        help="module:function; called only after the public freeze passes",
    )
    parser.add_argument("--require-current-head", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare != "prepare":
        print("TRR-0010 truth preparation requires the prepare command", file=sys.stderr)
        return 2
    try:
        result = prepare_truth_after_freeze(
            freeze_path=args.freeze,
            truth_sidecar_path=args.truth_sidecar,
            descriptor_path=args.descriptor,
            repository_root=args.repository_root,
            materializer=_load_callable(args.materializer),
            require_current_head=args.require_current_head,
        )
    except (TruthPreparationError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-0010 truth preparation failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
