"""Truth-free TRR-0009 observation manifest/capture adapter.

The owner selection process supplies metadata-only rows.  A trusted public
producer supplies compact BF16 current-H observations for each paired cell;
this adapter validates geometry, writes create-only tensors/receipts, and
never materializes source text, token IDs, target labels, or truth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import argparse
import gc
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

import torch
from safetensors.torch import save_file

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_capture_public as trr6_capture
from scripts import trr0009_eval_contract as contract
from token_reconstruction.public_activation import capture_public_prefix

SELECTION_SCHEMA = "token-reconstruction.trr0009-source-selection.v1"
SELECTION_STATUS = "FROZEN_TRR0009_SOURCE_SELECTION_NO_TRUTH"
CAPTURE_SCHEMA = "token-reconstruction.trr0009-public-capture.v1"
CAPTURE_STATUS = "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH"
PANEL_SCHEMA = "token-reconstruction.trr0009-public-source-panel.v1"
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
_ALLOWED_ROW_FIELDS = {
    "record_id", "public_record_sha256", "dataset_key", "dataset_id", "split",
    "revision", "row_index", "source_index", "full_token_count",
    "post_bos_token_count", "valid_tokens", "final_sequence_sha256",
}


class CaptureError(contract.ContractError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CaptureError(f"repository root unavailable: {root}")
    return root


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    try:
        return contract.validate_file_record(
            {"path": str(Path(path).expanduser().resolve()), "bytes": Path(path).stat().st_size, "sha256": contract.sha256_file(Path(path))},
            repository_root=root,
            description=description,
            verify=True,
        )
    except (OSError, contract.ContractError) as exc:
        raise CaptureError(str(exc)) from exc


def _write_create_only(path: Path, value: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    try:
        record = contract.write_create_only(path, value)
    except contract.ContractError as exc:
        raise CaptureError(str(exc)) from exc
    return record


def load_selection(path: Path, *, repository_root: Path, expected_counts: Mapping[str, int] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int]]:
    root = _root(repository_root)
    path = Path(path).expanduser().resolve()
    try:
        selection = contract.load_json(path, description="TRR-0009 source selection")
    except contract.ContractError as exc:
        raise CaptureError(str(exc)) from exc
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("task_id") != contract.TASK_ID or selection.get("status") != SELECTION_STATUS:
        raise CaptureError("source selection is not a frozen TRR-0009 identity-only ledger")
    if selection.get("target_conditions") != list(contract.TARGET_ORDER) or selection.get("paired_conditions") is not True:
        raise CaptureError("source target pairing changed")
    for key in ("truth_opened", "truth_created", "source_text_or_target_labels", "source_text_written", "token_ids_written", "target_labels_loaded"):
        if selection.get(key) is True:
            raise CaptureError(f"source selection records forbidden access: {key}")
    rule = selection.get("selection_rule")
    rows_raw = rule.get("records") if isinstance(rule, Mapping) else None
    if not isinstance(rows_raw, Mapping):
        raise CaptureError("source selection lacks metadata-only rows")
    declared_counts = selection.get("records_by_domain")
    if not isinstance(declared_counts, Mapping):
        declared_counts = {style: len(rows_raw.get(style, ())) for style in contract.DOMAIN_ORDER}
    counts: dict[str, int] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for style in contract.DOMAIN_ORDER:
        values = rows_raw.get(style)
        try:
            count = int(declared_counts[style])
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureError(f"source selection count is absent: {style}") from exc
        if not isinstance(values, list) or count <= 0 or len(values) != count:
            raise CaptureError(f"source selection row count changed: {style}")
        if expected_counts is not None and count != int(expected_counts[style]):
            raise CaptureError(f"source selection count differs from frozen count: {style}")
        rows[style] = []
        seen_ids: set[str] = set()
        seen_sequences: set[str] = set()
        for index, raw in enumerate(values):
            if not isinstance(raw, Mapping) or set(raw) - _ALLOWED_ROW_FIELDS:
                raise CaptureError(f"source selection {style} row {index} contains unapproved payload")
            row = dict(raw)
            record_id = row.get("record_id")
            sequence_hash = row.get("final_sequence_sha256")
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise CaptureError(f"source selection {style} record IDs are invalid")
            if not isinstance(sequence_hash, str) or len(sequence_hash) != 64 or sequence_hash in seen_sequences:
                raise CaptureError(f"source selection {style} sequence commitments are invalid")
            if row.get("dataset_key") != style or row.get("valid_tokens") != contract.STORED_SEQUENCE_TOKENS:
                raise CaptureError(f"source selection {style} row binding changed")
            for field in ("public_record_sha256", "final_sequence_sha256"):
                value = row.get(field)
                if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise CaptureError(f"source selection {style} {field} is malformed")
            seen_ids.add(record_id); seen_sequences.add(sequence_hash); rows[style].append(row)
        counts[style] = count
    selection_record = _record(path, root=root, description="TRR-0009 source selection")
    return selection, selection_record, rows, counts


def _capture_output(path: Path, *, root: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    task_root = (root / "experiments" / contract.TASK_ID / "evaluation").resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise CaptureError(f"capture output must be below {task_root}") from exc
    if path.is_symlink() or path.exists():
        raise CaptureError(f"capture output is create-only: {path}")
    return path


def save_observation(
    path: Path,
    *,
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cell_id: str,
    selection_sha256: str,
    record_ids_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    root = _root(repository_root)
    if cell_id not in contract.CELL_ORDER:
        raise CaptureError(f"unknown observation cell: {cell_id}")
    activation = torch.as_tensor(activations).detach().cpu().contiguous()
    mask = torch.as_tensor(attention_mask).detach().cpu().contiguous()
    positions = torch.as_tensor(position_ids).detach().cpu().contiguous()
    if activation.ndim != 3 or tuple(activation.shape[1:]) != (contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE) or activation.dtype != torch.bfloat16:
        raise CaptureError(f"{cell_id} compact activation geometry/dtype changed")
    records = int(activation.shape[0])
    if records <= 0 or records % contract.CHUNK_RECORDS:
        raise CaptureError(f"{cell_id} records are not a positive multiple of the capture batch size")
    if tuple(mask.shape) != (records, contract.STORED_SEQUENCE_TOKENS) or tuple(positions.shape) != tuple(mask.shape):
        raise CaptureError(f"{cell_id} sidecar geometry changed")
    if mask.dtype not in (torch.bool, torch.uint8) or not mask.to(torch.bool).all().item():
        raise CaptureError(f"{cell_id} attention mask changed")
    expected_positions = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.long).expand(records, -1)
    if not torch.equal(positions.to(torch.long), expected_positions):
        raise CaptureError(f"{cell_id} position IDs changed")
    if not torch.isfinite(activation.float()).all().item():
        raise CaptureError(f"{cell_id} activation contains non-finite values")
    path = _capture_output(path, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"activations": activation, "attention_mask": mask.to(torch.uint8), "position_ids": positions.to(torch.int64)},
        str(path),
        metadata={
            "schema": "token-reconstruction.trr0009-public-observation.v1",
            "task_id": contract.TASK_ID,
            "cell_id": cell_id,
            "shape": json.dumps([records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE]),
            "capture_batch_records": str(CAPTURE_BATCH_RECORDS),
            "capture_sequence_tokens": str(CAPTURE_SEQUENCE_TOKENS),
            "record_ids_sha256": str(record_ids_sha256),
            "selection_plan_sha256": str(selection_sha256),
            "source_text_written": "false",
            "token_ids_written": "false",
            "target_labels_loaded": "false",
            "truth_opened": "false",
        },
    )
    descriptor = _record(path, root=root, description=f"observation {cell_id}")
    descriptor.update({
        "shape": [records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE],
        "stored_sequence_tokens": contract.STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "activations_key": "activations",
        "attention_mask_key": "attention_mask",
        "position_ids_key": "position_ids",
        "public_full_forward": True,
        "producer_only_lora": cell_id.endswith("__public_lora_2601"),
    })
    return descriptor


def build_observation_manifest(*, selection_record: Mapping[str, Any], selection: Mapping[str, Any], observations: Mapping[str, Mapping[str, Any]], records_by_domain: Mapping[str, int], record_ids_sha256: Mapping[str, str], repository_root: Path) -> dict[str, Any]:
    root = _root(repository_root)
    cells: list[dict[str, Any]] = []
    for cell_id in contract.CELL_ORDER:
        style, condition = cell_id.split("__", 1)
        observation = observations.get(cell_id)
        if not isinstance(observation, Mapping):
            raise CaptureError(f"missing captured observation: {cell_id}")
        records = int(records_by_domain[style])
        if observation.get("shape") != [records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE]:
            raise CaptureError(f"observation geometry differs from selected {cell_id}")
        # Recheck file binding after producer output is complete.
        checked = _record(Path(str(observation["path"])), root=root, description=f"observation {cell_id}")
        if checked != {key: observation.get(key) for key in ("path", "bytes", "sha256")}:
            raise CaptureError(f"observation binding changed: {cell_id}")
        cells.append({"cell_id": cell_id, "style": style, "condition": condition, "records": records, "record_ids_sha256": str(record_ids_sha256[style]), "observation": dict(observation)})
    return {
        "schema": contract.OBSERVATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "records_by_domain": {str(k): int(v) for k, v in records_by_domain.items()},
        "cell_order": list(contract.CELL_ORDER),
        "cells": cells,
        "selection_plan": dict(selection_record),
        "method_freeze_sha256": selection.get("method_freeze_sha256"),
        "sequence_tokens_including_bos": contract.STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "hidden_size": contract.HIDDEN_SIZE,
        "source_pairing": {"same_record_ids_across_targets": True, "record_ids_sha256": dict(record_ids_sha256)},
        "public_material_only": True,
        "source_text_loaded": False,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "truth_opened": False,
    }


def build_capture_receipt(*, selection_record: Mapping[str, Any], observation_record: Mapping[str, Any], observation_manifest: Mapping[str, Any], records_by_domain: Mapping[str, int], public_inputs: Mapping[str, Any], source_code: Mapping[str, Any], repository_root: Path, started_utc: str | None = None, ended_utc: str | None = None) -> dict[str, Any]:
    root = _root(repository_root)
    if observation_manifest.get("schema") != contract.OBSERVATION_SCHEMA or observation_manifest.get("truth_opened") is not False:
        raise CaptureError("observation manifest is not a complete truth-free receipt")
    return {
        "schema": CAPTURE_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": CAPTURE_STATUS,
        "records_by_domain": {str(k): int(v) for k, v in records_by_domain.items()},
        "selection_plan": dict(selection_record),
        "observations": dict(observation_record),
        "method_freeze_sha256": observation_manifest.get("method_freeze_sha256"),
        "geometry": {"capture_batch_records": CAPTURE_BATCH_RECORDS, "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS, **contract.STATIC_GEOMETRY, "cells": len(contract.CELL_ORDER)},
        "public_inputs": dict(public_inputs),
        "source_code": dict(source_code),
        "execution": {"started_utc": started_utc or _utc_now(), "ended_utc": ended_utc or _utc_now(), "python": sys.executable, "python_version": platform.python_version(), "producer_semantics": "public full forward B8x192; retain first 128 positions", "target_labels_loaded": False, "truth_opened": False, "source_text_written": False, "token_ids_written": False, "candidate_arrays_persisted": False},
        "source_text_loaded": False,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "truth_opened": False,
    }


def capture_from_observations(*, selection_path: Path, observation_tensors: Mapping[str, Mapping[str, torch.Tensor]], output_root: Path, repository_root: Path, public_inputs: Mapping[str, Any] | None = None, source_code: Mapping[str, Any] | None = None, expected_counts: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Package already-produced public tensors into a task-local capture receipt."""

    root = _root(repository_root)
    selection, selection_record, rows, counts = load_selection(selection_path, repository_root=root, expected_counts=expected_counts)
    output = _capture_output(output_root, root=root)
    output.mkdir(parents=True)
    record_ids_sha256 = {style: contract.canonical_json_digest([str(row["record_id"]) for row in rows[style]]) for style in contract.DOMAIN_ORDER}
    observations: dict[str, Mapping[str, Any]] = {}
    for cell_id in contract.CELL_ORDER:
        values = observation_tensors.get(cell_id)
        if not isinstance(values, Mapping) or not all(key in values for key in ("activations", "attention_mask", "position_ids")):
            raise CaptureError(f"producer observation tensors are incomplete: {cell_id}")
        observations[cell_id] = save_observation(output / "observations" / f"{cell_id}.safetensors", activations=values["activations"], attention_mask=values["attention_mask"], position_ids=values["position_ids"], cell_id=cell_id, selection_sha256=selection_record["sha256"], record_ids_sha256=record_ids_sha256[cell_id.split("__", 1)[0]], repository_root=root)
    observation_payload = build_observation_manifest(selection_record=selection_record, selection=selection, observations=observations, records_by_domain=counts, record_ids_sha256=record_ids_sha256, repository_root=root)
    observation_record = _write_create_only(output / "observations.json", observation_payload, root=root, description="TRR-0009 observation manifest")
    panel_record = _write_create_only(output / "panel.json", {"schema": PANEL_SCHEMA, "task_id": contract.TASK_ID, "status": "FROZEN_SOURCE_PANEL_NO_TRUTH", "records_by_domain": counts, "cell_order": list(contract.CELL_ORDER), "record_ids_sha256": record_ids_sha256, "selection_plan": dict(selection_record), "observation_manifest": dict(observation_record), "same_sources_across_targets": True, "public_material_only": True, "truth_opened": False}, root=root, description="TRR-0009 source panel")
    capture_payload = build_capture_receipt(selection_record=selection_record, observation_record=observation_record, observation_manifest=observation_payload, records_by_domain=counts, public_inputs=public_inputs or {}, source_code=source_code or {}, repository_root=root)
    capture_record = _write_create_only(output / "capture.json", capture_payload, root=root, description="TRR-0009 capture receipt")
    return {"status": CAPTURE_STATUS, "selection": selection_record, "observation_manifest": observation_record, "panel": panel_record, "capture": capture_record, "truth_opened": False}


def _resolve_path(value: Path | str, *, root: Path) -> Path:
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else root / raw).resolve()


def _digest_record_ids(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    return {
        style: contract.canonical_json_digest([str(row["record_id"]) for row in rows[style]])
        for style in contract.DOMAIN_ORDER
    }


def _source_paths(selection: Mapping[str, Any], *, style: str, root: Path) -> tuple[Path, ...]:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get(style) if isinstance(sources, Mapping) else None
    files = descriptor.get("arrow_files") if isinstance(descriptor, Mapping) else None
    if not isinstance(files, list) or not files:
        raise CaptureError(f"source selection has no {style} Arrow descriptors")
    result: list[Path] = []
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise CaptureError(f"source selection {style} Arrow descriptor is malformed")
        result.append(_resolve_path(str(item["path"]), root=root))
    return tuple(result)


def _tokenizer_path(selection: Mapping[str, Any], *, root: Path) -> Path:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise CaptureError("source selection has no tokenizer descriptor")
    return _resolve_path(str(descriptor["path"]), root=root)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _failure_diagnostics(exc: BaseException, *, root: Path, stage: str | None) -> dict[str, Any]:
    """Preserve enough context to reproduce a fail-closed producer attempt."""
    chain: list[dict[str, str]] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    return {
        "stage": stage,
        "command": list(sys.argv),
        "code_commit": _git_commit(root),
        "exception_chain": chain,
        "traceback": "".join(traceback.format_exception(exc)),
    }


def _source_code_records(root: Path) -> dict[str, dict[str, Any]]:
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
    return {name: _record(path, root=root, description=f"capture source {name}") for name, path in paths.items()}


def _capture_condition_with_producer(
    *,
    condition: str,
    records: Mapping[str, Sequence[Any]],
    batches: Mapping[str, Any],
    model_snapshot: Path,
    lora_config_path: Path | None,
    lora_update_path: Path | None,
    output_root: Path,
    counts: Mapping[str, int],
    record_ids_sha256: Mapping[str, str],
    selection_sha256: str,
    repository_root: Path,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the qualified TRR5/TRR6 public producer, then package its outputs.

    The producer reads only the already-frozen public selection and writes no
    labels or source text.  The task adapter owns the create-only TRR9 files.
    """
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
    cell_receipts: dict[str, Any] = {}
    try:
        for style in contract.DOMAIN_ORDER:
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
                expected_shape = (int(counts[style]), CAPTURE_SEQUENCE_TOKENS, contract.HIDDEN_SIZE)
                if tuple(activations.shape) != expected_shape:
                    raise CaptureError(f"{cell_id} full capture geometry changed")
                compact = activations[:, : contract.STORED_SEQUENCE_TOKENS].contiguous()
                descriptor = save_observation(
                    output_root / "observations" / f"{cell_id}.safetensors",
                    activations=compact,
                    attention_mask=batches[style].attention_mask[:, : contract.STORED_SEQUENCE_TOKENS],
                    position_ids=batches[style].position_ids[:, : contract.STORED_SEQUENCE_TOKENS],
                    cell_id=cell_id,
                    selection_sha256=selection_sha256,
                    record_ids_sha256=record_ids_sha256[style],
                    repository_root=repository_root,
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
                if compact is not None:
                    del compact
                if activations is not None:
                    del activations
                gc.collect()
            observations[cell_id] = descriptor
            cell_receipts[cell_id] = {
                "seconds": time.perf_counter() - started,
                "observation": descriptor,
                "records": int(counts[style]),
                "capture_batch_records": CAPTURE_BATCH_RECORDS,
                "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
                "stored_sequence_tokens": contract.STORED_SEQUENCE_TOKENS,
                "full_forward_retained_only_first_128": True,
            }
    finally:
        del prefix
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return observations, {"condition": condition, "load": load_evidence, "qualification": qualification, "cells": cell_receipts}


def _model_input_descriptor(model_snapshot: Path, *, root: Path) -> dict[str, Any]:
    """Record the exact model files consumed by the qualified producer."""
    descriptor: dict[str, Any] = {"path": str(model_snapshot.resolve())}
    files: dict[str, Any] = {}
    for name in ("config.json", "generation_config.json", "model.safetensors"):
        candidate = model_snapshot / name
        if candidate.resolve().is_file():
            files[name] = _record(candidate.resolve(), root=root, description=f"public model {name}")
    if not files:
        raise CaptureError("public model snapshot has no recognized files")
    descriptor["files"] = files
    return descriptor


def capture_public(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise CaptureError("public capture requires explicit --execute")
    root = _root(args.repository_root)
    selection_path = _resolve_path(args.selection, root=root)
    selection, selection_record, selected_rows, counts = load_selection(selection_path, repository_root=root)
    output_root = _capture_output(args.output_root, root=root)
    output_root.mkdir(parents=True)
    failure_path = output_root / "failure.json"
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    active_condition: str | None = None
    try:
        pile_paths = tuple(_resolve_path(path, root=root) for path in (args.pile_arrow or _source_paths(selection, style="pile", root=root)))
        finance_paths = tuple(_resolve_path(path, root=root) for path in (args.finance_arrow or _source_paths(selection, style="finance", root=root)))
        tokenizer_path = _resolve_path(args.tokenizer, root=root) if args.tokenizer is not None else _tokenizer_path(selection, root=root)
        trr6_capture._validate_source_descriptors(selection, pile_paths=pile_paths, finance_paths=finance_paths, tokenizer_path=tokenizer_path)
        tokenizer = trusted._load_tokenizer(tokenizer_path)
        datasets = {"pile": trusted._load_arrow_dataset(pile_paths), "finance": trusted._load_arrow_dataset(finance_paths)}
        records = trr6_capture._materialize_selected(selected_rows, datasets=datasets, tokenizer=tokenizer)
        batches = trr6_capture._batches(records)
        record_ids_sha256 = _digest_record_ids(selected_rows)
        device = trusted._device(args.device)
        model_snapshot = _resolve_path(args.model_snapshot, root=root)
        if model_snapshot.is_symlink() or not model_snapshot.is_dir():
            raise CaptureError(f"public model snapshot is unavailable: {model_snapshot}")
        lora_config_path = _resolve_path(args.lora_config, root=root) if args.lora_config is not None else None
        lora_update_path = _resolve_path(args.lora_update, root=root) if args.lora_update is not None else None
        observations: dict[str, Mapping[str, Any]] = {}
        conditions: dict[str, Any] = {}
        for condition in contract.TARGET_ORDER:
            active_condition = condition
            if condition == "public_lora_2601" and (lora_config_path is None or lora_update_path is None):
                raise CaptureError("public_lora_2601 requires --lora-config and --lora-update")
            current, receipt = _capture_condition_with_producer(
                condition=condition,
                records=records,
                batches=batches,
                model_snapshot=model_snapshot,
                lora_config_path=lora_config_path,
                lora_update_path=lora_update_path,
                output_root=output_root,
                counts=counts,
                record_ids_sha256=record_ids_sha256,
                selection_sha256=selection_record["sha256"],
                repository_root=root,
                device=device,
            )
            observations.update(current)
            conditions[condition] = receipt
        observation_payload = build_observation_manifest(
            selection_record=selection_record,
            selection=selection,
            observations=observations,
            records_by_domain=counts,
            record_ids_sha256=record_ids_sha256,
            repository_root=root,
        )
        observation_record = _write_create_only(output_root / "observations.json", observation_payload, root=root, description="TRR-0009 observation manifest")
        panel_record = _write_create_only(
            output_root / "panel.json",
            {"schema": PANEL_SCHEMA, "task_id": contract.TASK_ID, "status": "FROZEN_SOURCE_PANEL_NO_TRUTH", "records_by_domain": dict(counts), "cell_order": list(contract.CELL_ORDER), "record_ids_sha256": dict(record_ids_sha256), "selection_plan": dict(selection_record), "observation_manifest": dict(observation_record), "same_sources_across_targets": True, "public_material_only": True, "truth_opened": False},
            root=root,
            description="TRR-0009 source panel",
        )
        capture_payload = build_capture_receipt(
            selection_record=selection_record,
            observation_record=observation_record,
            observation_manifest=observation_payload,
            records_by_domain=counts,
            public_inputs={"pile": trusted._dataset_descriptor(pile_paths, style="pile"), "finance": trusted._dataset_descriptor(finance_paths, style="finance"), "tokenizer": trusted._tokenizer_descriptor(tokenizer_path), "model_snapshot": _model_input_descriptor(model_snapshot, root=root)},
            source_code=_source_code_records(root),
            repository_root=root,
            started_utc=started_utc,
            ended_utc=_utc_now(),
        )
        capture_payload["conditions"] = conditions
        capture_payload["execution"].update({"elapsed_seconds": time.perf_counter() - started_clock, "command": list(sys.argv), "code_commit": _git_commit(root), "device": str(device), "model_loaded_by_producer": True, "task_isolation": "TRR9 adapter reuses qualified TRR6/TRR5 producer helpers"})
        capture_record = _write_create_only(output_root / "capture.json", capture_payload, root=root, description="TRR-0009 capture receipt")
        return {"task_id": contract.TASK_ID, "status": CAPTURE_STATUS, "records_by_domain": dict(counts), "observation_manifest": observation_record, "panel": panel_record, "capture": capture_record, "truth_opened": False}
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            try:
                failure_payload = {
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
                    "failure_diagnostics": _failure_diagnostics(exc, root=root, stage=active_condition),
                }
                _write_create_only(failure_path, failure_payload, root=root, description="capture failure receipt")
            except Exception:
                pass
        if isinstance(exc, CaptureError):
            raise
        raise CaptureError("public observation capture failed") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", default="capture")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--pile-arrow", type=Path, nargs="*")
    parser.add_argument("--finance-arrow", type=Path, nargs="*")
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path)
    parser.add_argument("--lora-update", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/TRR-0009/evaluation/public_observations"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.capture != "capture":
        print("TRR-0009 capture requires the capture command", file=sys.stderr)
        return 2
    try:
        result = capture_public(args)
    except (CaptureError, contract.ContractError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-0009 public capture failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
