#!/usr/bin/env python3
"""Capture task-local TRR-0007 public observations after source selection.

The adapter reuses the trusted TRR-0006 capture helpers and TRR-0005 public
producer, but writes its own TRR-0007 observation schema under the task root.
The public model and optional synthetic-LoRA target are loaded only after the
identity-only selection is frozen.  The output contains BF16 activations,
mask, and position sidecars plus hashes; it never writes source text, token
IDs, target labels, or truth.  Capture is create-only and requires
``--execute``.
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
import time
from typing import Any

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0004_produce_confirmation as trr4
from scripts import trr0006_capture_public as trr6_capture
from scripts import trr0007_eval_contract as contract
from scripts import trr0007_bank_ledger as bank_ledger
from token_reconstruction.trr0005_contract import STYLE_ORDER


SELECTION_SCHEMA = "token-reconstruction.trr0007-source-selection.v1"
SELECTION_STATUS = "FROZEN_TRR0007_SOURCE_SELECTION_NO_TRUTH"
CAPTURE_SCHEMA = "token-reconstruction.trr0007-public-capture.v1"
TASK_ROOT_RELATIVE = Path("experiments/TRR-0007")
CAPTURE_ROOT_RELATIVE = Path("experiments/TRR-0007/evaluation")


class CaptureError(contract.ContractError):
    """Raised when public observation capture cannot satisfy the contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _file_record(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"{description} is unavailable: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": contract.sha256_file(path)}


def _write_create_only(path: Path, value: Mapping[str, Any], *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise CaptureError(f"{description} is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return _file_record(path, description=description)


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CaptureError(f"repository root is unavailable: {root}")
    return root


def _task_output(value: Path | str, *, root: Path) -> Path:
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    task_root = (root / CAPTURE_ROOT_RELATIVE).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise CaptureError(f"capture output must be below {task_root}: {path}") from exc
    if path.is_symlink() or path.exists():
        raise CaptureError(f"capture output is create-only and already exists: {path}")
    return path


def _resolve_path(value: str | Path, *, root: Path) -> Path:
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else root / raw).resolve()


def _load_selection(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    selection = contract.load_json(path, description="TRR-0007 source selection")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("task_id") != contract.TASK_ID:
        raise CaptureError("source selection schema or task ID changed")
    if selection.get("status") != SELECTION_STATUS:
        raise CaptureError("source selection is not frozen")
    if selection.get("records_per_domain") != contract.RECORDS_PER_DOMAIN:
        raise CaptureError("source selection count changed")
    if selection.get("target_conditions") != list(contract.TARGET_ORDER) or selection.get("paired_conditions") is not True:
        raise CaptureError("source target pairing changed")
    for key in ("truth_opened", "truth_created", "source_text_or_target_labels"):
        if selection.get(key) is True:
            raise CaptureError(f"source selection records forbidden access: {key}")
    method_binding = selection.get("method_freeze")
    if not isinstance(method_binding, Mapping):
        raise CaptureError("source selection lacks the method-freeze file binding")
    method_hash = selection.get("method_freeze_sha256")
    if not isinstance(method_hash, str) or len(method_hash) != 64 or any(c not in "0123456789abcdef" for c in method_hash):
        raise CaptureError("source selection lacks a lowercase method-freeze SHA-256")
    if method_binding.get("sha256") != method_hash:
        raise CaptureError("source selection method-freeze descriptor disagrees with its digest")
    if repository_root is not None:
        try:
            ledger_record, _ledger, ledger_states = contract.load_method_freeze(
                Path(str(method_binding["path"])),
                repository_root=repository_root,
                verify_assets=True,
            )
        except contract.ContractError as exc:
            raise CaptureError(str(exc)) from exc
        if ledger_record != dict(method_binding):
            raise CaptureError("source selection method-freeze descriptor does not match its file")
        declared_states = selection.get("method_freeze_state_sha256")
        if declared_states != {method_id: state["sha256"] for method_id, state in ledger_states.items()}:
            raise CaptureError("source selection selected-state hashes differ from method freeze")
        final_bank = selection.get("final_bank_ledgers")
        final_files = final_bank.get("files") if isinstance(final_bank, Mapping) else None
        if not isinstance(final_files, Mapping) or not all(
            isinstance(final_files.get(key), Mapping) for key in ("exclusion_manifest", "selected_parent_rows", "corpus_plan")
        ):
            raise CaptureError("source selection lacks final v5 bank ledgers")
        try:
            verified_bank = bank_ledger.load_final_bank_ledgers(
                repository_root=repository_root,
                exclusion_manifest=Path(str(final_files["exclusion_manifest"]["path"])),
                selected_parent_rows=Path(str(final_files["selected_parent_rows"]["path"])),
                corpus_plan=Path(str(final_files["corpus_plan"]["path"])),
            )
        except bank_ledger.BankLedgerError as exc:
            raise CaptureError(str(exc)) from exc
        if verified_bank != dict(final_bank):
            raise CaptureError("source selection final v5 bank descriptor changed")
        prefix_ledger = selection.get("public_fitting_prefix_exclusions")
        prefix_file = prefix_ledger.get("file") if isinstance(prefix_ledger, Mapping) else None
        if not isinstance(prefix_file, Mapping) or not isinstance(prefix_file.get("path"), str):
            raise CaptureError("source selection lacks the reviewed v3 fitting-prefix ledger")
        try:
            verified_prefix = bank_ledger.load_prefix_exclusion_ledger(
                repository_root=repository_root, path=Path(str(prefix_file["path"]))
            )
        except bank_ledger.BankLedgerError as exc:
            raise CaptureError(str(exc)) from exc
        if verified_prefix != dict(prefix_ledger):
            raise CaptureError("source selection v3 fitting-prefix ledger descriptor changed")
    ranges = selection.get("source_ranges_half_open")
    expected_ranges = {
        style: [
            int(trr6_capture.eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
            int(trr6_capture.eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
        ]
        for style in STYLE_ORDER
    }
    if ranges != expected_ranges:
        raise CaptureError("source selection ranges changed")
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping) or rule.get("source_text_or_token_ids_written") is not False:
        raise CaptureError("source selection is not identity-only")
    raw = rule.get("records")
    if not isinstance(raw, Mapping):
        raise CaptureError("source selection records are absent")
    allowed = {
        "record_id", "public_record_sha256", "dataset_key", "dataset_id", "split",
        "revision", "row_index", "source_index", "full_token_count",
        "post_bos_token_count", "valid_tokens", "final_sequence_sha256",
    }
    rows: dict[str, list[dict[str, Any]]] = {}
    for style in STYLE_ORDER:
        values = raw.get(style)
        if not isinstance(values, list) or len(values) != contract.RECORDS_PER_DOMAIN:
            raise CaptureError(f"source selection {style} record count changed")
        rows[style] = []
        seen_ids: set[str] = set()
        seen_sequences: set[str] = set()
        for index, value in enumerate(values):
            if not isinstance(value, Mapping) or set(value) - allowed:
                raise CaptureError(f"source selection {style} row {index} contains unapproved payload")
            row = dict(value)
            record_id = row.get("record_id")
            sequence_hash = row.get("final_sequence_sha256")
            row_index = row.get("row_index")
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise CaptureError(f"source selection {style} IDs are not unique")
            if not isinstance(sequence_hash, str) or len(sequence_hash) != 64 or sequence_hash in seen_sequences:
                raise CaptureError(f"source selection {style} sequence commitments are invalid")
            if isinstance(row_index, bool) or not isinstance(row_index, int):
                raise CaptureError(f"source selection {style} row index is invalid")
            if row.get("source_index") != row_index or row.get("dataset_key") != style or row.get("valid_tokens") != contract.STORED_SEQUENCE_TOKENS:
                raise CaptureError(f"source selection {style} row binding changed")
            seen_ids.add(record_id)
            seen_sequences.add(sequence_hash)
            rows[style].append(row)
    record_digests = rule.get("record_ids_sha256")
    if not isinstance(record_digests, Mapping) or set(record_digests) != set(STYLE_ORDER):
        raise CaptureError("source selection record-order digests are absent")
    selection_record = _file_record(path, description="TRR-0007 source selection")
    return selection, selection_record, rows


def _source_paths(selection: Mapping[str, Any], style: str, *, root: Path) -> tuple[Path, ...]:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get(style) if isinstance(sources, Mapping) else None
    files = descriptor.get("arrow_files") if isinstance(descriptor, Mapping) else None
    if not isinstance(files, list) or not files:
        raise CaptureError(f"source selection has no {style} Arrow descriptor")
    paths: list[Path] = []
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise CaptureError(f"source selection {style} Arrow descriptor is malformed")
        paths.append(_resolve_path(str(item["path"]), root=root))
    return tuple(paths)


def _tokenizer_path(selection: Mapping[str, Any], *, root: Path) -> Path:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise CaptureError("source selection has no tokenizer descriptor")
    return _resolve_path(str(descriptor["path"]), root=root)


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


def _observation_manifest(
    *,
    output_root: Path,
    selection_path: Path,
    selection_record: Mapping[str, Any],
    selection: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    record_ids_sha256: Mapping[str, str],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for cell_id in contract.CELL_ORDER:
        style, condition = cell_id.split("__", 1)
        observation = observations.get(cell_id)
        if not isinstance(observation, Mapping):
            raise CaptureError(f"missing captured observation: {cell_id}")
        cells.append(
            {
                "cell_id": cell_id,
                "style": style,
                "condition": condition,
                "records": contract.RECORDS_PER_DOMAIN,
                "record_ids_sha256": record_ids_sha256[style],
                "observation": dict(observation),
            }
        )
    return {
        "schema": contract.OBSERVATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "records_per_domain": contract.RECORDS_PER_DOMAIN,
        "cell_order": list(contract.CELL_ORDER),
        "cells": cells,
        "selection_plan": dict(selection_record),
        "method_freeze_sha256": selection["method_freeze_sha256"],
        "source_ranges_half_open": selection["source_ranges_half_open"],
        "sequence_tokens_including_bos": contract.STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
        "capture_batch_records": contract.CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": contract.CAPTURE_SEQUENCE_TOKENS,
        "hidden_size": contract.HIDDEN_SIZE,
        "source_pairing": {
            "same_record_ids_across_targets": True,
            "record_ids_sha256": dict(record_ids_sha256),
        },
        "public_material_only": True,
        "source_text_loaded": False,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "truth_opened": False,
    }


def capture_public(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise CaptureError("public capture requires explicit --execute")
    root = _root(args.repository_root)
    selection_path = Path(args.selection).expanduser().resolve()
    selection, selection_record, selected_rows = _load_selection(
        selection_path, repository_root=root
    )
    output_root = _task_output(args.output_root, root=root)
    output_root.mkdir(parents=True)
    failure_path = output_root / "failure.json"
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    try:
        pile_paths = _source_paths(selection, "pile", root=root)
        finance_paths = _source_paths(selection, "finance", root=root)
        tokenizer_path = _tokenizer_path(selection, root=root)
        if args.pile_arrow:
            pile_paths = tuple(_resolve_path(path, root=root) for path in args.pile_arrow)
        if args.finance_arrow:
            finance_paths = tuple(_resolve_path(path, root=root) for path in args.finance_arrow)
        if args.tokenizer is not None:
            tokenizer_path = _resolve_path(args.tokenizer, root=root)
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
        records = trr6_capture._materialize_selected(selected_rows, datasets=datasets, tokenizer=tokenizer)
        batches = trr6_capture._batches(records)
        record_ids_sha256 = {
            style: str(selection["selection_rule"]["record_ids_sha256"][style])
            for style in STYLE_ORDER
        }
        device = trusted._device(args.device)
        model_snapshot = _resolve_path(args.model_snapshot, root=root)
        if model_snapshot.is_symlink() or not model_snapshot.is_dir():
            raise CaptureError(f"public model snapshot is unavailable: {model_snapshot}")
        lora_config_path = _resolve_path(args.lora_config, root=root) if args.lora_config is not None else None
        lora_update_path = _resolve_path(args.lora_update, root=root) if args.lora_update is not None else None
        observations: dict[str, Mapping[str, Any]] = {}
        conditions: dict[str, Any] = {}
        for condition in contract.TARGET_ORDER:
            if condition == "public_lora_2601" and (lora_config_path is None or lora_update_path is None):
                raise CaptureError("public_lora_2601 requires --lora-config and --lora-update")
            current, receipt = trr6_capture._capture_condition(
                condition=condition,
                records=records,
                batches=batches,
                model_snapshot=model_snapshot,
                lora_config_path=lora_config_path,
                lora_update=lora_update_path,
                output_root=output_root,
                records_per_domain=contract.RECORDS_PER_DOMAIN,
                record_ids_sha256=record_ids_sha256,
                selection_sha256=selection_record["sha256"],
                device=device,
            )
            observations.update(current)
            conditions[condition] = receipt
        observation_manifest = _observation_manifest(
            output_root=output_root,
            selection_path=selection_path,
            selection_record=selection_record,
            selection=selection,
            observations=observations,
            record_ids_sha256=record_ids_sha256,
        )
        observation_path = output_root / "observations.json"
        observation_record = _write_create_only(observation_path, observation_manifest, description="observation manifest")
        panel = {
            "schema": "token-reconstruction.trr0007-public-source-panel.v1",
            "task_id": contract.TASK_ID,
            "status": "FROZEN_SOURCE_PANEL_NO_TRUTH",
            "records_per_domain": contract.RECORDS_PER_DOMAIN,
            "cell_order": list(contract.CELL_ORDER),
            "record_ids_sha256": dict(record_ids_sha256),
            "selection_plan": dict(selection_record),
            "observation_manifest": dict(observation_record),
            "method_freeze_sha256": selection["method_freeze_sha256"],
            "same_sources_across_targets": True,
            "public_material_only": True,
            "truth_opened": False,
        }
        panel_record = _write_create_only(output_root / "panel.json", panel, description="source panel manifest")
        capture = {
            "schema": CAPTURE_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH",
            "selection_plan": dict(selection_record),
            "method_freeze_sha256": selection["method_freeze_sha256"],
            "source_pairing": {
                "same_record_ids_across_targets": True,
                "record_ids_sha256": dict(record_ids_sha256),
            },
            "geometry": {
                "capture_batch_records": contract.CAPTURE_BATCH_RECORDS,
                "capture_sequence_tokens": contract.CAPTURE_SEQUENCE_TOKENS,
                "stored_sequence_tokens": contract.STORED_SEQUENCE_TOKENS,
                "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
                "hidden_size": contract.HIDDEN_SIZE,
                "cells": len(contract.CELL_ORDER),
            },
            "public_inputs": {
                "pile": trusted._dataset_descriptor(pile_paths, style="pile"),
                "finance": trusted._dataset_descriptor(finance_paths, style="finance"),
                "tokenizer": trusted._tokenizer_descriptor(tokenizer_path),
                "model_snapshot": trr4._runtime_snapshot(model_snapshot),
            },
            "conditions": conditions,
            "observations": dict(observation_record),
            "panel": dict(panel_record),
            "execution": {
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started_clock,
                "command": list(sys.argv),
                "code_commit": _git_commit(root),
                "source_code": {
                    "adapter": _file_record(Path(__file__), description="TRR-0007 capture adapter"),
                    "trusted_trr0006_capture": _file_record(Path(trr6_capture.__file__), description="trusted TRR-0006 capture"),
                    "trusted_trr0005_producer": _file_record(Path(trusted.__file__), description="trusted TRR-0005 producer"),
                },
                "python": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "device": str(device),
                "model_loaded_by_producer": True,
                "target_labels_loaded": False,
                "source_text_written": False,
                "token_ids_written": False,
                "truth_opened": False,
                "network_used": False,
                "task_isolation": "TRR-0006 public producer helpers reused; all manifests and observations are written below experiments/TRR-0007/evaluation only",
            },
        }
        capture_record = _write_create_only(output_root / "capture.json", capture, description="capture receipt")
        return {
            "task_id": contract.TASK_ID,
            "status": capture["status"],
            "records_per_domain": contract.RECORDS_PER_DOMAIN,
            "observation_manifest": observation_record,
            "panel": panel_record,
            "capture": capture_record,
            "truth_opened": False,
        }
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            failure = {
                "schema": CAPTURE_SCHEMA,
                "task_id": contract.TASK_ID,
                "status": "PUBLIC_OBSERVATIONS_CAPTURE_FAILED_NO_TRUTH",
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "selection_plan": str(selection_path),
                "selection_plan_sha256": selection_record.get("sha256"),
                "truth_opened": False,
                "source_text_written": False,
                "token_ids_written": False,
            }
            _write_create_only(failure_path, failure, description="capture failure receipt")
        if isinstance(exc, CaptureError):
            raise
        raise CaptureError("public observation capture failed") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", default="capture")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--pile-arrow", type=Path, nargs="*")
    parser.add_argument("--finance-arrow", type=Path, nargs="*")
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path)
    parser.add_argument("--lora-update", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/TRR-0007/evaluation/public_observations"))
    parser.add_argument("--device", choices=("auto", "cuda"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.capture != "capture":
        print("TRR-0007 capture requires the capture command", file=sys.stderr)
        return 2
    try:
        result = capture_public(args)
    except (CaptureError, contract.ContractError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-0007 public capture failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
