#!/usr/bin/env python3
"""Capture task-local TRR-0008 public observations after source selection.

The adapter reuses the trusted TRR-0006 capture helpers and TRR-0005 public
producer, but writes its own TRR-0008 observation schema under the task root.
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
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors.torch import save_file

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0004_produce_confirmation as trr4
from scripts import trr0006_capture_public as trr6_capture
from scripts import trr0008_eval_contract as contract
from token_reconstruction.public_activation import capture_public_prefix
from token_reconstruction.trr0005_contract import STYLE_ORDER


SELECTION_SCHEMA = "token-reconstruction.trr0008-source-selection.v1"
SELECTION_STATUS = "FROZEN_TRR0008_SOURCE_SELECTION_NO_TRUTH"
CAPTURE_SCHEMA = "token-reconstruction.trr0008-public-capture.v1"
PANEL_SCHEMA = "token-reconstruction.trr0008-public-source-panel.v1"
TASK_ROOT_RELATIVE = Path("experiments/TRR-0008")
CAPTURE_ROOT_RELATIVE = Path("experiments/TRR-0008/evaluation")
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
PADDING_TOKEN_ID = 128001
_ALLOWED_ROW_FIELDS = {
    "record_id", "public_record_sha256", "dataset_key", "dataset_id", "split",
    "revision", "row_index", "source_index", "full_token_count",
    "post_bos_token_count", "valid_tokens", "final_sequence_sha256",
}


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


def _digest_record_ids(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for style in STYLE_ORDER:
        encoded = json.dumps(
            [str(row["record_id"]) for row in rows[style]],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        result[style] = hashlib.sha256(encoded).hexdigest()
    return result


def _selection_rows(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping):
        raise CaptureError("source selection lacks selection_rule")
    rows = rule.get("records", selection.get("records"))
    if not isinstance(rows, Mapping):
        raise CaptureError("source selection lacks metadata-only records")
    return rows


def _selection_counts(selection: Mapping[str, Any], rows: Mapping[str, Any]) -> dict[str, int]:
    raw = selection.get("records_by_domain", selection.get("requested_per_domain"))
    if raw is None:
        raw = {style: len(rows.get(style, ())) for style in STYLE_ORDER}
    if not isinstance(raw, Mapping):
        raise CaptureError("source selection counts are malformed")
    counts: dict[str, int] = {}
    for style in STYLE_ORDER:
        try:
            count = int(raw[style])
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureError(f"source selection count is absent: {style}") from exc
        values = rows.get(style)
        if count <= 0 or not isinstance(values, list) or len(values) != count:
            raise CaptureError(f"source selection row count changed: {style}")
        counts[style] = count
    expected = {"finance": 1024, "pile": 384}
    if counts != expected:
        raise CaptureError(f"source selection counts differ from the frozen Finance/Pile plan: {counts}")
    return counts


def _validate_method_binding(selection: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    binding = selection.get("method_freeze", selection.get("method_freeze_record"))
    declared_hash = selection.get("method_freeze_sha256")
    if not isinstance(binding, Mapping) or not isinstance(declared_hash, str):
        raise CaptureError("source selection lacks method-freeze binding")
    if binding.get("sha256") != declared_hash:
        raise CaptureError("source selection method-freeze digest disagrees with descriptor")
    checked = contract.validate_file_record(
        binding,
        repository_root=root,
        description="TRR-0008 method freeze",
        verify=True,
    )
    if checked != dict(binding):
        raise CaptureError("source selection method-freeze descriptor changed")
    return checked


def _validate_planning_bindings(selection: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    raw = selection.get("planning_bindings", selection.get("planning"))
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise CaptureError("source selection planning binding is malformed")
    result: dict[str, Any] = {}
    for label in ("decision_contract", "identity_inventory", "plan"):
        value = raw.get(label)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise CaptureError(f"planning binding is malformed: {label}")
        result[label] = contract.validate_file_record(
            value,
            repository_root=root,
            description=f"TRR-0008 planning {label}",
            verify=True,
        )
    return result


def _load_selection(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int]]:
    selection = contract.load_json(path, description="TRR-0008 source selection")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("task_id") != contract.TASK_ID:
        raise CaptureError("source selection schema or task ID changed")
    if selection.get("status") != SELECTION_STATUS:
        raise CaptureError("source selection is not frozen")
    if selection.get("target_conditions") != list(contract.TARGET_ORDER) or selection.get("paired_conditions") is not True:
        raise CaptureError("source target pairing changed")
    for key in (
        "truth_opened", "truth_created", "source_text_or_target_labels",
        "source_text_written", "token_ids_written",
    ):
        if selection.get(key) is True:
            raise CaptureError(f"source selection records forbidden access: {key}")
    _validate_method_binding(selection, root=repository_root)
    _validate_planning_bindings(selection, root=repository_root)
    expected_ranges = {
        style: [
            int(trr6_capture.eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
            int(trr6_capture.eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
        ]
        for style in STYLE_ORDER
    }
    if selection.get("source_ranges_half_open") != expected_ranges:
        raise CaptureError("source selection source ranges changed")
    raw = _selection_rows(selection)
    counts = _selection_counts(selection, raw)
    rows: dict[str, list[dict[str, Any]]] = {}
    for style in STYLE_ORDER:
        values = raw.get(style)
        rows[style] = []
        seen_ids: set[str] = set()
        seen_sequences: set[str] = set()
        start, stop = expected_ranges[style]
        spec = trr6_capture.eligibility.SOURCE_PARTITIONS[style]
        if not isinstance(values, list) or len(values) != counts[style]:
            raise CaptureError(f"source selection {style} row count changed")
        for row_number, value in enumerate(values):
            if not isinstance(value, Mapping) or set(value) - _ALLOWED_ROW_FIELDS:
                raise CaptureError(f"source selection {style} row {row_number} contains unapproved payload")
            row = dict(value)
            record_id = row.get("record_id")
            sequence_hash = row.get("final_sequence_sha256")
            index = row.get("row_index")
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise CaptureError(f"source selection {style} record IDs are invalid")
            if not isinstance(sequence_hash, str) or len(sequence_hash) != 64 or sequence_hash in seen_sequences:
                raise CaptureError(f"source selection {style} sequence commitments are invalid")
            if isinstance(index, bool) or not isinstance(index, int) or not start <= index < stop:
                raise CaptureError(f"source selection {style} row escaped the frozen source range")
            if row.get("source_index") != index or row.get("dataset_key") != style:
                raise CaptureError(f"source selection {style} source binding changed")
            if row.get("dataset_id") != spec["dataset_id"] or row.get("split") != spec["split"] or row.get("revision") != spec["revision"]:
                raise CaptureError(f"source selection {style} dataset binding changed")
            expected_id = trusted.source_record_id(
                str(spec["dataset_id"]), str(spec["split"]), str(spec["revision"]), index
            )
            if record_id != expected_id:
                raise CaptureError(f"source selection {style} record ID changed")
            if row.get("valid_tokens") != contract.STORED_SEQUENCE_TOKENS:
                raise CaptureError(f"source selection {style} clip length changed")
            full_count = row.get("full_token_count")
            post_count = row.get("post_bos_token_count")
            if (
                isinstance(full_count, bool) or isinstance(post_count, bool)
                or not isinstance(full_count, int) or not isinstance(post_count, int)
                or full_count < contract.STORED_SEQUENCE_TOKENS
                or post_count != full_count - 1
            ):
                raise CaptureError(f"source selection {style} token length metadata changed")
            for key in ("public_record_sha256", "final_sequence_sha256"):
                value_hash = row.get(key)
                if not isinstance(value_hash, str) or len(value_hash) != 64 or any(c not in "0123456789abcdef" for c in value_hash):
                    raise CaptureError(f"source selection {style} {key} is malformed")
            seen_ids.add(record_id)
            seen_sequences.add(sequence_hash)
            rows[style].append(row)
    declared = selection.get("selection_rule", {}).get("record_ids_sha256")
    if not isinstance(declared, Mapping) or dict(declared) != _digest_record_ids(rows):
        raise CaptureError("source selection record-order digest changed")
    return selection, _file_record(path, description="TRR-0008 source selection"), rows, counts

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


def _save_observation(
    path: Path,
    *,
    compact: torch.Tensor,
    batch: Any,
    cell_id: str,
    records: int,
    record_ids_sha256: str,
    selection_sha256: str,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise CaptureError(f"observation output is create-only: {path}")
    expected_shape = (records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE)
    if tuple(compact.shape) != expected_shape or compact.dtype != torch.bfloat16:
        raise CaptureError(f"{cell_id} compact observation geometry or dtype changed")
    mask = batch.attention_mask[:, : contract.STORED_SEQUENCE_TOKENS].to(torch.uint8).contiguous()
    positions = batch.position_ids[:, : contract.STORED_SEQUENCE_TOKENS].to(torch.int64).contiguous()
    expected_positions = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.int64).repeat(records, 1)
    if tuple(mask.shape) != (records, contract.STORED_SEQUENCE_TOKENS) or not bool(mask.eq(1).all().item()):
        raise CaptureError(f"{cell_id} observation mask changed")
    if not torch.equal(positions, expected_positions):
        raise CaptureError(f"{cell_id} observation positions changed")
    if not bool(torch.isfinite(compact.float()).all().item()):
        raise CaptureError(f"{cell_id} observation contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "activations": compact.detach().cpu().contiguous(),
            "attention_mask": mask,
            "position_ids": positions,
        },
        str(path),
        metadata={
            "schema": "token-reconstruction.trr0008-public-observation.v1",
            "task_id": contract.TASK_ID,
            "cell_id": cell_id,
            "shape": str(list(expected_shape)),
            "hidden_size": str(contract.HIDDEN_SIZE),
            "stored_sequence_tokens": str(contract.STORED_SEQUENCE_TOKENS),
            "scored_post_bos_tokens": str(contract.SCORED_POST_BOS_TOKENS),
            "capture_batch_records": str(CAPTURE_BATCH_RECORDS),
            "capture_sequence_tokens": str(CAPTURE_SEQUENCE_TOKENS),
            "public_full_forward": "true",
            "producer_only_lora": str(cell_id.endswith("__public_lora_2601")).lower(),
            "record_ids_sha256": record_ids_sha256,
            "selection_plan_sha256": selection_sha256,
            "source_text_written": "false",
            "token_ids_written": "false",
            "truth_opened": "false",
        },
    )
    descriptor = _file_record(path, description=f"observation {cell_id}")
    descriptor.update(
        {
            "shape": list(expected_shape),
            "stored_sequence_tokens": contract.STORED_SEQUENCE_TOKENS,
            "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
            "capture_batch_records": CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "activations_key": "activations",
            "attention_mask_key": "attention_mask",
            "position_ids_key": "position_ids",
            "public_full_forward": True,
            "producer_only_lora": cell_id.endswith("__public_lora_2601"),
        }
    )
    return descriptor


def _capture_condition(
    *,
    condition: str,
    records: Mapping[str, Sequence[Any]],
    batches: Mapping[str, Any],
    model_snapshot: Path,
    lora_config_path: Path | None,
    lora_update: Path | None,
    output_root: Path,
    counts: Mapping[str, int],
    record_ids_sha256: Mapping[str, str],
    selection_sha256: str,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        prefix, load_evidence = trusted._capture_prefix(
            condition=condition,
            model_snapshot=model_snapshot,
            lora_config_path=lora_config_path,
            lora_update=lora_update,
            device=device,
        )
    except Exception as exc:
        raise CaptureError(f"{condition} public-prefix load failed") from exc
    try:
        import trr0004_prepare_public_activations as prep
        qualification = prep._qualify_public_prefix_padding(
            prefix,
            batches["finance"],
            device=device,
            batch_size=CAPTURE_BATCH_RECORDS,
        )
        trusted._live_resource_guard(device)
        _preflight, ceiling = trusted._guard_helpers()
    except Exception as exc:
        raise CaptureError(f"{condition} largest 8x192 qualification failed") from exc
    observations: dict[str, dict[str, Any]] = {}
    receipts: dict[str, Any] = {}
    for style in STYLE_ORDER:
        cell_id = f"{style}__{condition}"
        started = time.perf_counter()
        activations: torch.Tensor | None = None
        compact: torch.Tensor | None = None
        try:
            activations = capture_public_prefix(
                prefix,
                batches[style],
                device=device,
                batch_size=CAPTURE_BATCH_RECORDS,
                resource_check=lambda: ceiling(
                    device,
                    max_reserved_gpu_bytes=trusted.MAX_RESERVED_GPU_BYTES,
                    max_host_rss_bytes=trusted.MAX_HOST_RSS_BYTES,
                ),
            )
            expected_full = (counts[style], CAPTURE_SEQUENCE_TOKENS, contract.HIDDEN_SIZE)
            if tuple(activations.shape) != expected_full:
                raise CaptureError(f"{cell_id} full capture geometry changed")
            compact = activations[:, : contract.STORED_SEQUENCE_TOKENS].contiguous()
            descriptor = _save_observation(
                output_root / "observations" / f"{cell_id}.safetensors",
                compact=compact,
                batch=batches[style],
                cell_id=cell_id,
                records=counts[style],
                record_ids_sha256=record_ids_sha256[style],
                selection_sha256=selection_sha256,
            )
            ceiling(
                device,
                max_reserved_gpu_bytes=trusted.MAX_RESERVED_GPU_BYTES,
                max_host_rss_bytes=trusted.MAX_HOST_RSS_BYTES,
            )
        except CaptureError:
            raise
        except Exception as exc:
            raise CaptureError(f"{cell_id} public-prefix capture failed") from exc
        finally:
            del compact, activations
            gc.collect()
        observations[cell_id] = descriptor
        receipts[cell_id] = {
            "seconds": time.perf_counter() - started,
            "observation": descriptor,
            "records": counts[style],
            "capture_batch_records": CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "stored_sequence_tokens": contract.STORED_SEQUENCE_TOKENS,
            "full_forward_retained_only_first_128": True,
        }
    del prefix
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return observations, {
        "condition": condition,
        "load": load_evidence,
        "qualification": qualification,
        "cells": receipts,
    }


def _observation_manifest(
    *,
    selection_record: Mapping[str, Any],
    selection: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, int],
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
                "records": counts[style],
                "record_ids_sha256": record_ids_sha256[style],
                "observation": dict(observation),
            }
        )
    return {
        "schema": contract.OBSERVATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "records_by_domain": dict(counts),
        "cell_order": list(contract.CELL_ORDER),
        "cells": cells,
        "selection_plan": dict(selection_record),
        "method_freeze_sha256": selection.get("method_freeze_sha256"),
        "source_ranges_half_open": selection["source_ranges_half_open"],
        "sequence_tokens_including_bos": contract.STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
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

def _capture_source_code(root: Path) -> dict[str, Any]:
    paths = {
        "adapter": Path(__file__).resolve(),
        "trusted_trr0006_capture": Path(trr6_capture.__file__).resolve(),
        "trusted_trr0005_producer": Path(trusted.__file__).resolve(),
        "public_activation": root / "src/token_reconstruction/public_activation.py",
        "public_prefix": root / "src/token_reconstruction/public_prefix.py",
        "public_corpus": root / "src/token_reconstruction/trr0005_public_corpus.py",
        "trr0004_capture_loader": root / "scripts/trr0004_produce_confirmation.py",
        "trr0004_resource_guard": root / "scripts/trr0004_prepare_public_activations.py",
    }
    return {name: _file_record(path, description=f"capture source {name}") for name, path in paths.items()}


def capture_public(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise CaptureError("public capture requires explicit --execute")
    root = _root(args.repository_root)
    selection_path = _resolve_path(args.selection, root=root)
    selection, selection_record, selected_rows, counts = _load_selection(
        selection_path, repository_root=root
    )
    planning_records = _validate_planning_bindings(selection, root=root)
    output_root = _task_output(args.output_root, root=root)
    output_root.mkdir(parents=True)
    failure_path = output_root / "failure.json"
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    try:
        pile_paths = tuple(
            _resolve_path(path, root=root)
            for path in (args.pile_arrow or _source_paths(selection, "pile", root=root))
        )
        finance_paths = tuple(
            _resolve_path(path, root=root)
            for path in (args.finance_arrow or _source_paths(selection, "finance", root=root))
        )
        tokenizer_path = (
            _resolve_path(args.tokenizer, root=root)
            if args.tokenizer is not None
            else _tokenizer_path(selection, root=root)
        )
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
        records = trr6_capture._materialize_selected(
            selected_rows, datasets=datasets, tokenizer=tokenizer
        )
        batches = trr6_capture._batches(records)
        record_ids_sha256 = _digest_record_ids(selected_rows)
        device = trusted._device(args.device)
        model_snapshot = _resolve_path(args.model_snapshot, root=root)
        if model_snapshot.is_symlink() or not model_snapshot.is_dir():
            raise CaptureError(f"public model snapshot is unavailable: {model_snapshot}")
        lora_config_path = (
            _resolve_path(args.lora_config, root=root) if args.lora_config is not None else None
        )
        lora_update_path = (
            _resolve_path(args.lora_update, root=root) if args.lora_update is not None else None
        )
        observations: dict[str, Mapping[str, Any]] = {}
        conditions: dict[str, Any] = {}
        for condition in contract.TARGET_ORDER:
            if condition == "public_lora_2601" and (
                lora_config_path is None or lora_update_path is None
            ):
                raise CaptureError("public_lora_2601 requires --lora-config and --lora-update")
            current, receipt = _capture_condition(
                condition=condition,
                records=records,
                batches=batches,
                model_snapshot=model_snapshot,
                lora_config_path=lora_config_path,
                lora_update=lora_update_path,
                output_root=output_root,
                counts=counts,
                record_ids_sha256=record_ids_sha256,
                selection_sha256=selection_record["sha256"],
                device=device,
            )
            observations.update(current)
            conditions[condition] = receipt
        observation_path = output_root / "observations.json"
        observation_record = _write_create_only(
            observation_path,
            _observation_manifest(
                selection_record=selection_record,
                selection=selection,
                observations=observations,
                counts=counts,
                record_ids_sha256=record_ids_sha256,
            ),
            description="TRR-0008 observation manifest",
        )
        panel_record = _write_create_only(
            output_root / "panel.json",
            {
                "schema": PANEL_SCHEMA,
                "task_id": contract.TASK_ID,
                "status": "FROZEN_SOURCE_PANEL_NO_TRUTH",
                "records_by_domain": dict(counts),
                "cell_order": list(contract.CELL_ORDER),
                "record_ids_sha256": dict(record_ids_sha256),
                "selection_plan": dict(selection_record),
                "observation_manifest": dict(observation_record),
                "method_freeze_sha256": selection.get("method_freeze_sha256"),
                "same_sources_across_targets": True,
                "public_material_only": True,
                "truth_opened": False,
            },
            description="TRR-0008 source panel",
        )
        capture = {
            "schema": CAPTURE_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH",
            "records_by_domain": dict(counts),
            "selection_plan": dict(selection_record),
            "planning_bindings": planning_records,
            "method_freeze_sha256": selection.get("method_freeze_sha256"),
            "source_pairing": {
                "same_record_ids_across_targets": True,
                "record_ids_sha256": dict(record_ids_sha256),
            },
            "geometry": {
                "capture_batch_records": CAPTURE_BATCH_RECORDS,
                "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
                **contract.STATIC_GEOMETRY,
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
                "source_code": _capture_source_code(root),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "device": str(device),
                "model_loaded_by_producer": True,
                "target_labels_loaded": False,
                "truth_opened": False,
                "source_text_written": False,
                "token_ids_written": False,
                "network_used": False,
                "task_isolation": "TRR8 adapter reuses qualified TRR6/TRR5 producer helpers",
            },
        }
        capture_record = _write_create_only(
            output_root / "capture.json", capture, description="TRR-0008 capture receipt"
        )
        return {
            "task_id": contract.TASK_ID,
            "status": capture["status"],
            "records_by_domain": dict(counts),
            "observation_manifest": observation_record,
            "panel": panel_record,
            "capture": capture_record,
            "truth_opened": False,
        }
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            try:
                _write_create_only(
                    failure_path,
                    {
                        "schema": CAPTURE_SCHEMA,
                        "task_id": contract.TASK_ID,
                        "status": "PUBLIC_OBSERVATIONS_CAPTURE_FAILED_NO_TRUTH",
                        "started_utc": started_utc,
                        "ended_utc": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "selection_plan": dict(selection_record),
                        "records_by_domain": dict(counts),
                        "truth_opened": False,
                        "source_text_written": False,
                        "token_ids_written": False,
                    },
                    description="capture failure receipt",
                )
            except Exception:
                pass
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
    parser.add_argument("--output-root", type=Path, default=Path("experiments/TRR-0008/evaluation/public_observations"))
    parser.add_argument("--device", choices=("auto", "cuda"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.capture != "capture":
        print("TRR-0008 capture requires the capture command", file=sys.stderr)
        return 2
    try:
        result = capture_public(args)
    except (CaptureError, contract.ContractError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-0008 public capture failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
