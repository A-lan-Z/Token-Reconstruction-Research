#!/usr/bin/env python3
"""Capture the frozen TRR-0006 public observation panel.

The source selection and this capture stage are separate.  The selection plan
must already be bound to the frozen decision plan before this command can read
any reserved source row.  Capture uses the published public-prefix path at
batch 8 by sequence 192, retains only the first 128 positions, and writes
only BF16 activations plus mask and position sidecars.  It never writes source
text, token IDs, labels, targets, or evaluator truth.

The command is intentionally create-only and requires ``--execute``.  A
failed run leaves a task-local failure receipt so a changed or partial output
cannot be mistaken for a completed panel.
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
from safetensors import safe_open
from safetensors.torch import save_file

# The trusted producer imports the TRR4 preparation modules by their script
# names.  Keep the script directory importable when this file is called as a
# path from the repository root.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts import trr0005_produce_confirmation as trusted  # noqa: E402
from scripts import trr0006_build_eligibility as eligibility  # noqa: E402
from scripts import trr0006_select_public as selector  # noqa: E402
from token_reconstruction.public_activation import (  # noqa: E402
    capture_public_prefix,
    pad_public_token_sequences,
)
from token_reconstruction.trr0005_contract import STYLE_ORDER  # noqa: E402


TASK_ID = "TRR-0006"
OBSERVATION_SCHEMA = "token-reconstruction.trr0006-public-observation-manifest.v1"
OBSERVATION_FILE_SCHEMA = "token-reconstruction.trr0006-public-observation.v1"
CAPTURE_SCHEMA = "token-reconstruction.trr0006-public-capture.v1"
PANEL_SCHEMA = "token-reconstruction.trr0006-public-source-panel.v1"
SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
PADDING_TOKEN_ID = 128001
SELECTION_SCHEMA = "token-reconstruction.trr0006-source-selection.v1"
SELECTION_STATUS = "FROZEN_TRR0006_SOURCE_SELECTION_NO_TRUTH"
CONDITION_ORDER = ("public_base", "public_lora_2601")
CELL_ORDER = (
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
)
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"


class CaptureError(RuntimeError):
    """Raised when the public capture contract cannot be satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    path = path.expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise CaptureError(f"{description} must be a JSON object")
    return dict(value)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise CaptureError(f"capture output is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return _file_record(path)


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise CaptureError(f"asset is not a regular file: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


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
    return value or None


def _source_ranges() -> dict[str, list[int]]:
    return {
        style: [
            int(eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
            int(eligibility.SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
        ]
        for style in STYLE_ORDER
    }


def _selection_rows(selection: Mapping[str, Any], *, records_per_domain: int) -> dict[str, list[dict[str, Any]]]:
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping):
        raise CaptureError("selection plan has no selection rule")
    raw = rule.get("records")
    if not isinstance(raw, Mapping):
        raise CaptureError("selection plan has no selected records")
    result: dict[str, list[dict[str, Any]]] = {}
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
    for style in STYLE_ORDER:
        rows = raw.get(style)
        if not isinstance(rows, list) or len(rows) != records_per_domain:
            raise CaptureError(f"selection plan has wrong {style} record count")
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_sequences: set[str] = set()
        start, stop = _source_ranges()[style]
        spec = eligibility.SOURCE_PARTITIONS[style]
        for row_number, value in enumerate(rows):
            if not isinstance(value, Mapping):
                raise CaptureError(f"selection plan {style} row {row_number} is malformed")
            row = dict(value)
            if set(row) - allowed:
                raise CaptureError(f"selection plan {style} row contains unapproved payload fields")
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise CaptureError(f"selection plan {style} has duplicate or empty source IDs")
            index = row.get("row_index")
            if isinstance(index, bool) or not isinstance(index, int) or not start <= index < stop:
                raise CaptureError(f"selection plan {style} row escaped the published source range")
            if row.get("source_index") != index or row.get("dataset_key") != style:
                raise CaptureError(f"selection plan {style} source index/key changed")
            if row.get("dataset_id") != spec["dataset_id"] or row.get("split") != spec["split"] or row.get("revision") != spec["revision"]:
                raise CaptureError(f"selection plan {style} dataset binding changed")
            expected_id = trusted.source_record_id(
                str(spec["dataset_id"]), str(spec["split"]), str(spec["revision"]), index
            )
            if record_id != expected_id:
                raise CaptureError(f"selection plan {style} source ID changed")
            for key in ("public_record_sha256", "final_sequence_sha256"):
                value_hash = row.get(key)
                if not isinstance(value_hash, str) or len(value_hash) != 64 or any(c not in "0123456789abcdef" for c in value_hash):
                    raise CaptureError(f"selection plan {style} {key} is malformed")
            full_count = row.get("full_token_count")
            post_count = row.get("post_bos_token_count")
            if isinstance(full_count, bool) or isinstance(post_count, bool) or not isinstance(full_count, int) or not isinstance(post_count, int) or full_count < SEQUENCE_TOKENS or post_count != full_count - 1:
                raise CaptureError(f"selection plan {style} token length metadata changed")
            if row.get("valid_tokens") != SEQUENCE_TOKENS:
                raise CaptureError(f"selection plan {style} clip length changed")
            sequence_hash = str(row["final_sequence_sha256"])
            if sequence_hash in seen_sequences:
                raise CaptureError(f"selection plan {style} has duplicate final sequences")
            seen_ids.add(record_id)
            seen_sequences.add(sequence_hash)
            normalized.append(row)
        result[style] = normalized
    all_sequences = [row["final_sequence_sha256"] for style in STYLE_ORDER for row in result[style]]
    if len(set(all_sequences)) != len(all_sequences):
        raise CaptureError("selection plan has duplicate final sequences across domains")
    return result


def _validate_frozen_selection(
    frozen_plan_path: Path,
    selection_plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    frozen = selector._validate_frozen_plan(frozen_plan_path)
    selection = _load_json(selection_plan_path, description="TRR-0006 source selection plan")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("task_id") != TASK_ID:
        raise CaptureError("selection plan schema or task ID changed")
    if selection.get("status") != SELECTION_STATUS:
        raise CaptureError("source selection is not frozen")
    if selection.get("records_per_domain") != frozen["records_per_domain"]:
        raise CaptureError("source selection count differs from frozen plan")
    if selection.get("method_freeze_sha256") != frozen["method_freeze_sha256"]:
        raise CaptureError("source selection is bound to a different method freeze")
    if selection.get("source_ranges_half_open") != _source_ranges():
        raise CaptureError("source selection ranges changed")
    if selection.get("target_conditions") != list(CONDITION_ORDER):
        raise CaptureError("source selection target conditions changed")
    if selection.get("paired_conditions") is not True:
        raise CaptureError("source selection does not declare paired target conditions")
    rows = _selection_rows(selection, records_per_domain=int(frozen["records_per_domain"]))
    return frozen, selection, rows


def _paths_from_descriptor(selection: Mapping[str, Any], style: str) -> tuple[Path, ...]:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get(style) if isinstance(sources, Mapping) else None
    files = descriptor.get("arrow_files") if isinstance(descriptor, Mapping) else None
    if not isinstance(files, list) or not files:
        raise CaptureError(f"selection plan has no {style} Arrow descriptor")
    paths: list[Path] = []
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise CaptureError(f"selection plan {style} Arrow descriptor is malformed")
        paths.append(Path(str(item["path"])).expanduser().resolve())
    return tuple(paths)


def _tokenizer_path_from_descriptor(selection: Mapping[str, Any]) -> Path:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise CaptureError("selection plan has no tokenizer descriptor")
    return Path(str(descriptor["path"])).expanduser().resolve()


def _validate_source_descriptors(
    selection: Mapping[str, Any],
    *,
    pile_paths: Sequence[Path],
    finance_paths: Sequence[Path],
    tokenizer_path: Path,
) -> None:
    actual = {
        "pile": trusted._dataset_descriptor(pile_paths, style="pile"),
        "finance": trusted._dataset_descriptor(finance_paths, style="finance"),
        "tokenizer": trusted._tokenizer_descriptor(tokenizer_path),
    }
    frozen = selection.get("public_sources_frozen")
    if not isinstance(frozen, Mapping):
        raise CaptureError("selection plan has no frozen public sources")
    for name, descriptor in actual.items():
        if dict(frozen.get(name, {})) != dict(descriptor):
            raise CaptureError(f"{name} public source differs from frozen selection input")


def _materialize_selected(
    selection_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    datasets: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, list[Any]]:
    records: dict[str, list[Any]] = {}
    for style in STYLE_ORDER:
        values: list[Any] = []
        for declared in selection_rows[style]:
            index = int(declared["row_index"])
            row = trusted._read_reserved_row(datasets[style], style=style, row_index=index)
            try:
                candidate = trusted._render_row(style, row, index, tokenizer)
            except trusted.ProducerError as exc:
                raise CaptureError(f"{style} selected row {index} no longer renders") from exc
            actual = candidate.selection_metadata()
            for key in (
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
            ):
                if str(actual.get(key)) != str(declared.get(key)):
                    raise CaptureError(f"{style} selected row {index} changed: {key}")
            if len(candidate.token_ids) < SEQUENCE_TOKENS:
                raise CaptureError(f"{style} selected row {index} is shorter than the clip")
            values.append(candidate)
        records[style] = values
    if _json_digest({style: [row.record_id for row in records[style]] for style in STYLE_ORDER}) != _json_digest({style: [row["record_id"] for row in selection_rows[style]] for style in STYLE_ORDER}):
        raise CaptureError("materialized source order differs from frozen selection")
    return records


def _batches(records: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for style in STYLE_ORDER:
        sequences = [list(record.token_ids[:SEQUENCE_TOKENS]) for record in records[style]]
        batch = pad_public_token_sequences(
            sequences,
            maximum_tokens=CAPTURE_SEQUENCE_TOKENS,
            pad_token_id=PADDING_TOKEN_ID,
            bos_token_id=BOS_TOKEN_ID,
            vocab_size=VOCAB_SIZE,
        )
        if tuple(batch.token_ids.shape) != (len(sequences), CAPTURE_SEQUENCE_TOKENS):
            raise CaptureError(f"{style} capture batch geometry changed")
        if not batch.attention_mask[:, :SEQUENCE_TOKENS].eq(1).all().item():
            raise CaptureError(f"{style} selected clip is not fully valid")
        result[style] = batch
    return result


def _source_code_records(root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "capture_producer": Path(__file__).resolve(),
        "trusted_trr0005_producer": Path(trusted.__file__).resolve(),
        "public_activation": root / "src/token_reconstruction/public_activation.py",
        "public_prefix": root / "src/token_reconstruction/public_prefix.py",
        "public_corpus": root / "src/token_reconstruction/trr0005_public_corpus.py",
        "trr0004_capture_loader": root / "scripts/trr0004_produce_confirmation.py",
        "trr0004_resource_guard": root / "scripts/trr0004_prepare_public_activations.py",
    }
    return {name: _file_record(path) for name, path in paths.items()}


def _save_observation(
    path: Path,
    *,
    compact: torch.Tensor,
    batch: Any,
    cell_id: str,
    records_per_domain: int,
    record_ids_sha256: str,
    selection_sha256: str,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise CaptureError(f"observation output is create-only: {path}")
    expected_shape = (records_per_domain, SEQUENCE_TOKENS, HIDDEN_SIZE)
    if tuple(compact.shape) != expected_shape or compact.dtype != torch.bfloat16:
        raise CaptureError(f"{cell_id} compact observation geometry or dtype changed")
    mask = batch.attention_mask[:, :SEQUENCE_TOKENS].to(torch.uint8).contiguous()
    positions = batch.position_ids[:, :SEQUENCE_TOKENS].to(torch.int64).contiguous()
    if tuple(mask.shape) != (records_per_domain, SEQUENCE_TOKENS) or not mask.eq(1).all().item():
        raise CaptureError(f"{cell_id} observation mask changed")
    if tuple(positions.shape) != (records_per_domain, SEQUENCE_TOKENS):
        raise CaptureError(f"{cell_id} observation position geometry changed")
    expected_positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.int64).repeat(records_per_domain, 1)
    if not torch.equal(positions, expected_positions):
        raise CaptureError(f"{cell_id} observation positions changed")
    if not torch.isfinite(compact.float()).all().item():
        raise CaptureError(f"{cell_id} observation contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": OBSERVATION_FILE_SCHEMA,
        "task_id": TASK_ID,
        "cell_id": cell_id,
        "shape": str(list(expected_shape)),
        "hidden_size": str(HIDDEN_SIZE),
        "stored_sequence_tokens": str(SEQUENCE_TOKENS),
        "scored_post_bos_tokens": str(SCORED_POST_BOS_TOKENS),
        "capture_batch_records": str(CAPTURE_BATCH_RECORDS),
        "capture_sequence_tokens": str(CAPTURE_SEQUENCE_TOKENS),
        "cut_depth": "4",
        "public_full_forward": "true",
        "producer_only_lora": str(cell_id.endswith("__public_lora_2601")).lower(),
        "record_ids_sha256": record_ids_sha256,
        "selection_plan_sha256": selection_sha256,
        "source_text_written": "false",
        "token_ids_written": "false",
        "truth_opened": "false",
    }
    save_file(
        {
            "activations": compact.detach().cpu().contiguous(),
            "attention_mask": mask,
            "position_ids": positions,
        },
        str(path),
        metadata=metadata,
    )
    descriptor = _file_record(path)
    descriptor.update(
        {
            "shape": list(expected_shape),
            "stored_sequence_tokens": SEQUENCE_TOKENS,
            "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
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
    records_per_domain: int,
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

        # Finance is the representative largest geometry.  The batch is
        # exactly 8x192 with future padding after the stored 128-token clip.
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
    for style in STYLE_ORDER:
        cell_id = f"{style}__{condition}"
        started = time.perf_counter()
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
            if tuple(activations.shape) != (records_per_domain, CAPTURE_SEQUENCE_TOKENS, HIDDEN_SIZE):
                raise CaptureError(f"{cell_id} full capture geometry changed")
            compact = activations[:, :SEQUENCE_TOKENS].contiguous()
            observation_path = output_root / "observations" / f"{cell_id}.safetensors"
            descriptor = _save_observation(
                observation_path,
                compact=compact,
                batch=batches[style],
                cell_id=cell_id,
                records_per_domain=records_per_domain,
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
            if "compact" in locals():
                del compact
            if "activations" in locals():
                del activations
            gc.collect()
        observations[cell_id] = descriptor
        cell_receipts[cell_id] = {
            "seconds": time.perf_counter() - started,
            "observation": descriptor,
            "capture_batch_records": CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "stored_sequence_tokens": SEQUENCE_TOKENS,
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
        "cells": cell_receipts,
    }


def _build_observation_manifest(
    *,
    output_root: Path,
    selection_path: Path,
    selection_sha256: str,
    frozen: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    record_ids_sha256: Mapping[str, str],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for cell_id in CELL_ORDER:
        style, condition = cell_id.split("__", 1)
        observation = observations.get(cell_id)
        if not isinstance(observation, Mapping):
            raise CaptureError(f"missing observation descriptor: {cell_id}")
        cells.append(
            {
                "cell_id": cell_id,
                "style": style,
                "condition": condition,
                "record_ids_sha256": record_ids_sha256[style],
                "observation": dict(observation),
            }
        )
    return {
        "schema": OBSERVATION_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "records_per_domain": int(frozen["records_per_domain"]),
        "cell_order": list(CELL_ORDER),
        "cells": cells,
        "selection_plan": {
            "path": str(selection_path),
            "bytes": int(selection_path.stat().st_size),
            "sha256": selection_sha256,
        },
        "method_freeze_sha256": frozen["method_freeze_sha256"],
        "source_ranges_half_open": _source_ranges(),
        "sequence_tokens_including_bos": SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "source_pairing": {
            "same_record_ids_across_targets": True,
            "record_ids_sha256": dict(record_ids_sha256),
        },
        "public_material_only": True,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "truth_opened": False,
    }


def _build_panel(
    *,
    selection_path: Path,
    selection_sha256: str,
    observation_path: Path,
    observation_sha256: str,
    frozen: Mapping[str, Any],
    record_ids_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": PANEL_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_SOURCE_PANEL_NO_TRUTH",
        "records_per_domain": int(frozen["records_per_domain"]),
        "source_ranges_half_open": _source_ranges(),
        "sequence_tokens_including_bos": SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "cell_order": list(CELL_ORDER),
        "target_conditions": list(CONDITION_ORDER),
        "record_ids_sha256": dict(record_ids_sha256),
        "selection_plan": {
            "path": str(selection_path),
            "bytes": int(selection_path.stat().st_size),
            "sha256": selection_sha256,
        },
        "observation_manifest": {
            "path": str(observation_path),
            "bytes": int(observation_path.stat().st_size),
            "sha256": observation_sha256,
        },
        "method_freeze_sha256": frozen["method_freeze_sha256"],
        "same_sources_across_targets": True,
        "public_material_only": True,
        "truth_opened": False,
    }


def capture_public(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise CaptureError("capture requires explicit --execute")
    root = args.repository_root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CaptureError(f"repository root is unavailable: {root}")
    frozen_path = args.frozen_plan.expanduser().resolve()
    selection_path = args.selection_plan.expanduser().resolve()
    frozen, selection, selected_rows = _validate_frozen_selection(frozen_path, selection_path)
    output_root = args.output_root.expanduser()
    if output_root.is_absolute():
        output_root = output_root.resolve()
    else:
        output_root = (root / output_root).resolve()
    try:
        output_root.relative_to(root / "experiments" / "TRR-0006")
    except ValueError as exc:
        raise CaptureError("capture output root must be under experiments/TRR-0006") from exc
    if output_root.exists() or output_root.is_symlink():
        raise CaptureError(f"capture output root is create-only and already exists: {output_root}")
    output_root.mkdir(parents=True)
    selection_sha256 = _sha256_file(selection_path)
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    failure_path = output_root / "failure.json"
    try:
        pile_paths = tuple(
            path.expanduser().resolve() for path in (args.pile_arrow or _paths_from_descriptor(selection, "pile"))
        )
        finance_paths = tuple(
            path.expanduser().resolve() for path in (args.finance_arrow or _paths_from_descriptor(selection, "finance"))
        )
        tokenizer_path = (
            args.tokenizer.expanduser().resolve()
            if args.tokenizer is not None
            else _tokenizer_path_from_descriptor(selection)
        )
        _validate_source_descriptors(
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
        records = _materialize_selected(selected_rows, datasets=datasets, tokenizer=tokenizer)
        batches = _batches(records)
        record_ids_sha256 = {
            style: _json_digest([row["record_id"] for row in selected_rows[style]])
            for style in STYLE_ORDER
        }
        device = trusted._device(args.device)
        model_snapshot = args.model_snapshot.expanduser().resolve()
        if model_snapshot.is_symlink() or not model_snapshot.is_dir():
            raise CaptureError(f"public model snapshot is unavailable: {model_snapshot}")
        # Bind the pinned public model and LoRA assets before loading either
        # target prefix.  _runtime_snapshot checks the exact published weight.
        import trr0004_produce_confirmation as trr4

        runtime_snapshot = trr4._runtime_snapshot(model_snapshot)
        lora_descriptor: dict[str, Any] | None = None
        normalized_lora: dict[str, Any] | None = None
        if args.lora_config is not None:
            lora_config_path = args.lora_config.expanduser().resolve()
            _config, normalized_lora = trr4._load_lora_config(lora_config_path)
            lora_descriptor = _file_record(lora_config_path)
            lora_descriptor["normalized"] = normalized_lora
        else:
            lora_config_path = None
        if args.lora_update is not None:
            lora_update_path = args.lora_update.expanduser().resolve()
            if lora_update_path.is_symlink() or not lora_update_path.is_file():
                raise CaptureError("public_lora_2601 update is unavailable")
            lora_update_descriptor: dict[str, Any] | None = _file_record(lora_update_path)
        else:
            lora_update_path = None
            lora_update_descriptor = None
        observations: dict[str, dict[str, Any]] = {}
        conditions: dict[str, Any] = {}
        for condition in CONDITION_ORDER:
            if condition == "public_lora_2601" and (lora_config_path is None or lora_update_path is None):
                raise CaptureError("public_lora_2601 requires --lora-config and --lora-update")
            current, receipt = _capture_condition(
                condition=condition,
                records=records,
                batches=batches,
                model_snapshot=model_snapshot,
                lora_config_path=lora_config_path,
                lora_update=lora_update_path,
                output_root=output_root,
                records_per_domain=int(frozen["records_per_domain"]),
                record_ids_sha256=record_ids_sha256,
                selection_sha256=selection_sha256,
                device=device,
            )
            observations.update(current)
            conditions[condition] = receipt
        observation_manifest = _build_observation_manifest(
            output_root=output_root,
            selection_path=selection_path,
            selection_sha256=selection_sha256,
            frozen=frozen,
            observations=observations,
            record_ids_sha256=record_ids_sha256,
        )
        observation_path = output_root / "observations.json"
        observation_record = _write_create_only(observation_path, observation_manifest)
        panel = _build_panel(
            selection_path=selection_path,
            selection_sha256=selection_sha256,
            observation_path=observation_path,
            observation_sha256=observation_record["sha256"],
            frozen=frozen,
            record_ids_sha256=record_ids_sha256,
        )
        panel_path = output_root / "panel.json"
        panel_record = _write_create_only(panel_path, panel)
        ended_utc = _utc_now()
        capture = {
            "schema": CAPTURE_SCHEMA,
            "task_id": TASK_ID,
            "status": "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH",
            "frozen_plan": {
                "path": str(frozen_path),
                "bytes": int(frozen_path.stat().st_size),
                "sha256": _sha256_file(frozen_path),
                "status": frozen["status"],
                "records_per_domain": frozen["records_per_domain"],
            },
            "selection_plan": {
                "path": str(selection_path),
                "bytes": int(selection_path.stat().st_size),
                "sha256": selection_sha256,
                "records_per_domain": frozen["records_per_domain"],
            },
            "method_freeze_sha256": frozen["method_freeze_sha256"],
            "source_pairing": {
                "same_record_ids_across_targets": True,
                "record_ids_sha256": record_ids_sha256,
            },
            "public_inputs": {
                "pile": trusted._dataset_descriptor(pile_paths, style="pile"),
                "finance": trusted._dataset_descriptor(finance_paths, style="finance"),
                "tokenizer": trusted._tokenizer_descriptor(tokenizer_path),
                "model_snapshot": runtime_snapshot,
                "lora_config": lora_descriptor,
                "lora_update": lora_update_descriptor,
            },
            "geometry": {
                "capture_batch_records": CAPTURE_BATCH_RECORDS,
                "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
                "stored_sequence_tokens": SEQUENCE_TOKENS,
                "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
                "hidden_size": HIDDEN_SIZE,
                "cells": len(CELL_ORDER),
            },
            "conditions": conditions,
            "observations": observation_record,
            "panel": panel_record,
            "execution": {
                "started_utc": started_utc,
                "ended_utc": ended_utc,
                "elapsed_seconds": time.perf_counter() - started_clock,
                "command": list(sys.argv),
                "code_commit": _git_commit(root),
                "source_code": _source_code_records(root),
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
            },
        }
        capture_record = _write_create_only(output_root / "capture.json", capture)
        return {
            "task_id": TASK_ID,
            "status": capture["status"],
            "records_per_domain": frozen["records_per_domain"],
            "observation_manifest": observation_record,
            "panel": panel_record,
            "capture": capture_record,
            "truth_opened": False,
        }
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            failure = {
                "schema": CAPTURE_SCHEMA,
                "task_id": TASK_ID,
                "status": "PUBLIC_OBSERVATIONS_CAPTURE_FAILED_NO_TRUTH",
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "frozen_plan": str(frozen_path),
                "selection_plan": str(selection_path),
                "selection_plan_sha256": selection_sha256,
                "truth_opened": False,
                "source_text_written": False,
                "token_ids_written": False,
            }
            _write_create_only(failure_path, failure)
        if isinstance(exc, CaptureError):
            raise
        raise CaptureError("public observation capture failed") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    capture = parser.add_subparsers(dest="command", required=True).add_parser(
        "capture", help="capture the already-selected public panel"
    )
    capture.add_argument("--execute", action="store_true", help="required acknowledgment for model/public-prefix work")
    capture.add_argument("--repository-root", type=Path, default=Path("."))
    capture.add_argument("--frozen-plan", type=Path, required=True)
    capture.add_argument("--selection-plan", type=Path, required=True)
    capture.add_argument("--tokenizer", type=Path)
    capture.add_argument("--pile-arrow", type=Path, nargs="*")
    capture.add_argument("--finance-arrow", type=Path, nargs="*")
    capture.add_argument("--model-snapshot", type=Path, required=True)
    capture.add_argument("--lora-config", type=Path)
    capture.add_argument("--lora-update", type=Path)
    capture.add_argument("--output-root", type=Path, required=True)
    capture.add_argument("--device", choices=("auto", "cuda"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "capture":  # pragma: no cover
        raise CaptureError(f"unknown capture command: {args.command}")
    result = capture_public(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CaptureError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"TRR-0006 capture error: {exc}")
