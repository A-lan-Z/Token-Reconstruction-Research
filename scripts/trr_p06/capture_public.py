#!/usr/bin/env python3
"""Capture a frozen TRR-P06 panel through the published public-prefix helper.

The adapter validates P06's source-universe/selection contract and then reuses
parent public model loading and full-forward capture primitives.  It writes H,
mask, and position metadata only.  Execution requires ``--execute`` and is
kept separate from source selection, fitting, prediction, and truth scoring.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import trr0005_produce_confirmation as trusted  # noqa: E402
from scripts.trr_p06 import prepare_public_panel as panel  # noqa: E402
from token_reconstruction.public_activation import capture_public_prefix, pad_public_token_sequences  # noqa: E402


TASK_ID = "TRR-P06"
SELECTION_SCHEMA = panel.SELECTION_SCHEMA
SELECTION_STATUS = panel.SELECTION_STATUS
OBSERVATION_SCHEMA = "token-reconstruction.trr-p06-public-observation-manifest.v1"
OBSERVATION_FILE_SCHEMA = "token-reconstruction.trr-p06-public-observation.v1"
CAPTURE_SCHEMA = "token-reconstruction.trr-p06-public-capture.v1"
PANEL_SCHEMA = "token-reconstruction.trr-p06-public-source-panel.v1"
SEQUENCE_TOKENS = panel.CLIP_TOKENS
SCORED_POST_BOS_TOKENS = SEQUENCE_TOKENS - 1
CAPTURE_SEQUENCE_TOKENS = panel.CAPTURE_TOKENS
CAPTURE_BATCH_RECORDS = 8
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
PADDING_TOKEN_ID = 128001
CONDITION_ORDER = panel.CONDITION_ORDER
STYLE_ORDER = panel.STYLE_ORDER
CELL_ORDER = tuple(f"{style}__{condition}" for style in STYLE_ORDER for condition in CONDITION_ORDER)
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"


class CapturePreparationError(RuntimeError):
    """Raised when the P06 capture contract is not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CapturePreparationError(f"file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


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


def _write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise CapturePreparationError(f"capture output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return _file_record(path)


def _validate_selection(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    selection = panel._load_json(path, description="P06 source selection")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("task_id") != TASK_ID:
        raise CapturePreparationError("P06 source selection schema or task ID changed")
    if selection.get("status") != SELECTION_STATUS:
        raise CapturePreparationError("P06 source selection is not frozen")
    if int(selection.get("records_per_domain", -1)) != panel.RECORDS_PER_DOMAIN:
        raise CapturePreparationError("P06 source selection count changed")
    if list(selection.get("target_conditions", ())) != list(CONDITION_ORDER):
        raise CapturePreparationError("P06 target condition order changed")
    if selection.get("paired_conditions") is not True:
        raise CapturePreparationError("P06 source selection is not paired")
    if selection.get("clip_tokens_including_bos") != SEQUENCE_TOKENS:
        raise CapturePreparationError("P06 clip geometry changed")
    if selection.get("capture_sequence_tokens") != CAPTURE_SEQUENCE_TOKENS:
        raise CapturePreparationError("P06 capture geometry changed")
    source_ranges = selection.get("source_ranges_half_open")
    if source_ranges != panel.CANDIDATE_RANGES:
        raise CapturePreparationError("P06 source ranges changed")
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping) or rule.get("source_text_or_token_ids_written") is not False:
        raise CapturePreparationError("P06 selection boundary is malformed")
    raw = rule.get("records")
    if not isinstance(raw, Mapping):
        raise CapturePreparationError("P06 selection has no records")
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
        "domain",
    }
    all_ids: set[str] = set()
    all_sequences: set[str] = set()
    for style in STYLE_ORDER:
        rows = raw.get(style)
        if not isinstance(rows, list) or len(rows) != panel.RECORDS_PER_DOMAIN:
            raise CapturePreparationError(f"P06 selection has wrong {style} count")
        normalized: list[dict[str, Any]] = []
        spec = panel._dataset_spec(style)
        for row in rows:
            if not isinstance(row, Mapping) or set(row) - allowed:
                raise CapturePreparationError("P06 selection contains an unapproved payload field")
            item = dict(row)
            if item.get("domain") != style or item.get("dataset_key") != style:
                raise CapturePreparationError("P06 selection domain binding changed")
            index = item.get("row_index")
            start, stop = panel.CANDIDATE_RANGES[style]
            if isinstance(index, bool) or not isinstance(index, int) or not start <= index < stop:
                raise CapturePreparationError(f"P06 row escaped {style} candidate range")
            if item.get("source_index") != index:
                raise CapturePreparationError("P06 source index binding changed")
            if item.get("dataset_id") != spec["dataset_id"] or item.get("split") != spec["split"] or item.get("revision") != spec["revision"]:
                raise CapturePreparationError("P06 dataset revision binding changed")
            expected_id = trusted.source_record_id(spec["dataset_id"], spec["split"], spec["revision"], index)
            if item.get("record_id") != expected_id or item["record_id"] in all_ids:
                raise CapturePreparationError("P06 source record identity changed or duplicated")
            if item.get("valid_tokens") != SEQUENCE_TOKENS:
                raise CapturePreparationError("P06 selected clip is not fully valid")
            if not isinstance(item.get("full_token_count"), int) or item["full_token_count"] < SEQUENCE_TOKENS:
                raise CapturePreparationError("P06 source length metadata is invalid")
            if item.get("post_bos_token_count") != item["full_token_count"] - 1:
                raise CapturePreparationError("P06 source length metadata is inconsistent")
            for key in ("public_record_sha256", "final_sequence_sha256"):
                value = item.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise CapturePreparationError(f"P06 {key} is malformed")
            if item["final_sequence_sha256"] in all_sequences:
                raise CapturePreparationError("P06 final sequence is duplicated")
            all_ids.add(item["record_id"])
            all_sequences.add(item["final_sequence_sha256"])
            normalized.append(item)
        result[style] = normalized
    if selection.get("access_boundary", {}).get("token_ids_written") is not False:
        raise CapturePreparationError("P06 selection claims token IDs were written")
    return selection, result


def _validate_universe_binding(universe_path: Path, selection: Mapping[str, Any]) -> dict[str, Any]:
    universe = panel.load_universe(universe_path, require_frozen=True)
    binding = selection.get("source_universe")
    if not isinstance(binding, Mapping):
        raise CapturePreparationError("P06 selection lacks source-universe binding")
    if binding.get("sha256") != _sha256_file(universe_path):
        raise CapturePreparationError("P06 source-universe hash changed")
    if binding.get("catalog_sha256") != universe.get("exclusion_binding", {}).get("catalog_sha256"):
        raise CapturePreparationError("P06 exclusion catalog binding changed")
    if universe.get("exclusion_binding", {}).get("coverage_complete") is not True:
        raise CapturePreparationError("P06 source-universe exclusion coverage is incomplete")
    return universe


def _materialize_selected(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    datasets: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, list[Any]]:
    records: dict[str, list[Any]] = {}
    for style in STYLE_ORDER:
        values: list[Any] = []
        for declared in rows[style]:
            index = int(declared["row_index"])
            row = panel._read_candidate_row(datasets[style], style=style, row_index=index)
            try:
                candidate = trusted._render_row(style, row, index, tokenizer)
            except trusted.ProducerError as exc:
                raise CapturePreparationError(f"{style} selected row {index} no longer renders") from exc
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
                    raise CapturePreparationError(f"{style} selected row {index} changed: {key}")
            values.append(candidate)
        records[style] = values
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
            raise CapturePreparationError(f"{style} capture batch geometry changed")
        if not batch.attention_mask[:, :SEQUENCE_TOKENS].eq(1).all().item():
            raise CapturePreparationError(f"{style} selected clip is not fully valid")
        result[style] = batch
    return result


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
        raise CapturePreparationError(f"observation output is create-only: {path}")
    expected_shape = (records_per_domain, SEQUENCE_TOKENS, HIDDEN_SIZE)
    if tuple(compact.shape) != expected_shape or compact.dtype != torch.bfloat16:
        raise CapturePreparationError(f"{cell_id} observation geometry or dtype changed")
    mask = batch.attention_mask[:, :SEQUENCE_TOKENS].to(torch.uint8).contiguous()
    positions = batch.position_ids[:, :SEQUENCE_TOKENS].to(torch.int64).contiguous()
    expected_positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.int64).repeat(records_per_domain, 1)
    if not mask.eq(1).all().item() or not torch.equal(positions, expected_positions):
        raise CapturePreparationError(f"{cell_id} mask/position binding changed")
    if not torch.isfinite(compact.float()).all().item():
        raise CapturePreparationError(f"{cell_id} observation is non-finite")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"activations": compact.detach().cpu().contiguous(), "attention_mask": mask, "position_ids": positions},
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
    lora_update_path: Path | None,
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
            lora_update=lora_update_path,
            device=device,
        )
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
        raise CapturePreparationError(f"{condition} public-prefix qualification failed") from exc
    observations: dict[str, dict[str, Any]] = {}
    receipts: dict[str, Any] = {}
    for style in STYLE_ORDER:
        cell_id = f"{style}__{condition}"
        started = time.perf_counter()
        activations = None
        compact = None
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
                raise CapturePreparationError(f"{cell_id} full-forward geometry changed")
            compact = activations[:, :SEQUENCE_TOKENS].contiguous()
            descriptor = _save_observation(
                output_root / "observations" / f"{cell_id}.safetensors",
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
        except CapturePreparationError:
            raise
        except Exception as exc:
            raise CapturePreparationError(f"{cell_id} public capture failed") from exc
        finally:
            del compact, activations
            gc.collect()
        observations[cell_id] = descriptor
        receipts[cell_id] = {
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
    return observations, {"condition": condition, "load": load_evidence, "qualification": qualification, "cells": receipts}


def _observation_manifest(
    *,
    selection_path: Path,
    selection_sha256: str,
    universe_path: Path,
    universe_sha256: str,
    observations: Mapping[str, Mapping[str, Any]],
    record_ids_sha256: Mapping[str, str],
) -> dict[str, Any]:
    cells = []
    for cell_id in CELL_ORDER:
        style, condition = cell_id.split("__", 1)
        if cell_id not in observations:
            raise CapturePreparationError(f"missing observation cell: {cell_id}")
        cells.append(
            {
                "cell_id": cell_id,
                "style": style,
                "condition": condition,
                "record_ids_sha256": record_ids_sha256[style],
                "observation": dict(observations[cell_id]),
            }
        )
    return {
        "schema": OBSERVATION_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "records_per_domain": panel.RECORDS_PER_DOMAIN,
        "cell_order": list(CELL_ORDER),
        "cells": cells,
        "source_universe": {"path": str(universe_path), "sha256": universe_sha256},
        "selection_plan": {"path": str(selection_path), "bytes": selection_path.stat().st_size, "sha256": selection_sha256},
        "sequence_tokens_including_bos": SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "source_pairing": {"same_record_ids_across_targets": True, "record_ids_sha256": dict(record_ids_sha256)},
        "public_material_only": True,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "truth_opened": False,
    }


def capture_public(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise CapturePreparationError("capture requires explicit --execute")
    root = Path(args.repository_root).expanduser().resolve()
    selection_path = Path(args.selection).expanduser().resolve()
    universe_path = Path(args.universe).expanduser().resolve()
    selection, selected_rows = _validate_selection(selection_path)
    universe = _validate_universe_binding(universe_path, selection)
    output_root = Path(args.output_root).expanduser()
    output_root = output_root if output_root.is_absolute() else root / output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(root / "experiments" / "TRR-P06")
    except ValueError as exc:
        raise CapturePreparationError("P06 capture output must be under experiments/TRR-P06") from exc
    if output_root.exists() or output_root.is_symlink():
        raise CapturePreparationError(f"P06 capture output is create-only: {output_root}")
    output_root.mkdir(parents=True)
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    selection_sha256 = _sha256_file(selection_path)
    universe_sha256 = _sha256_file(universe_path)
    failure_path = output_root / "failure.json"
    try:
        pile_paths = tuple(Path(value).expanduser().resolve() for value in args.pile_arrow)
        finance_paths = tuple(Path(value).expanduser().resolve() for value in args.finance_arrow)
        tokenizer_path = Path(args.tokenizer).expanduser().resolve()
        actual_sources = {
            "pile": trusted._dataset_descriptor(pile_paths, style="pile"),
            "finance": trusted._dataset_descriptor(finance_paths, style="finance"),
            "tokenizer": trusted._tokenizer_descriptor(tokenizer_path),
        }
        frozen_sources = selection.get("public_sources_frozen")
        if not isinstance(frozen_sources, Mapping):
            raise CapturePreparationError("P06 selection lacks frozen public sources")
        for key, descriptor in actual_sources.items():
            if dict(frozen_sources.get(key, {})) != dict(descriptor):
                raise CapturePreparationError(f"P06 {key} source differs from the frozen selection")
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
        model_snapshot = Path(args.model_snapshot).expanduser().resolve()
        if model_snapshot.is_symlink() or not model_snapshot.is_dir():
            raise CapturePreparationError(f"public model snapshot is unavailable: {model_snapshot}")
        import trr0004_produce_confirmation as trr4

        runtime_snapshot = trr4._runtime_snapshot(model_snapshot)
        lora_config_path = Path(args.lora_config).expanduser().resolve() if args.lora_config else None
        lora_update_path = Path(args.lora_update).expanduser().resolve() if args.lora_update else None
        lora_descriptor = None
        normalized_lora = None
        lora_update_descriptor = None
        if lora_config_path is not None:
            _config, normalized_lora = trr4._load_lora_config(lora_config_path)
            lora_descriptor = _file_record(lora_config_path)
            lora_descriptor["normalized"] = normalized_lora
        if lora_update_path is not None:
            lora_update_descriptor = _file_record(lora_update_path)
        observations: dict[str, dict[str, Any]] = {}
        conditions: dict[str, Any] = {}
        for condition in CONDITION_ORDER:
            if condition == "public_lora_2601" and (lora_config_path is None or lora_update_path is None):
                raise CapturePreparationError("public_lora_2601 requires its public config and update")
            current, receipt = _capture_condition(
                condition=condition,
                records=records,
                batches=batches,
                model_snapshot=model_snapshot,
                lora_config_path=lora_config_path,
                lora_update=lora_update_path,
                output_root=output_root,
                records_per_domain=panel.RECORDS_PER_DOMAIN,
                record_ids_sha256=record_ids_sha256,
                selection_sha256=selection_sha256,
                device=device,
            )
            observations.update(current)
            conditions[condition] = receipt
        observation_manifest = _observation_manifest(
            selection_path=selection_path,
            selection_sha256=selection_sha256,
            universe_path=universe_path,
            universe_sha256=universe_sha256,
            observations=observations,
            record_ids_sha256=record_ids_sha256,
        )
        observation_record = _write_create_only(output_root / "observations.json", observation_manifest)
        panel_manifest = {
            "schema": PANEL_SCHEMA,
            "task_id": TASK_ID,
            "status": "FROZEN_SOURCE_PANEL_NO_TRUTH",
            "records_per_domain": panel.RECORDS_PER_DOMAIN,
            "source_ranges_half_open": dict(panel.CANDIDATE_RANGES),
            "sequence_tokens_including_bos": SEQUENCE_TOKENS,
            "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
            "capture_batch_records": CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "cell_order": list(CELL_ORDER),
            "target_conditions": list(CONDITION_ORDER),
            "record_ids_sha256": dict(record_ids_sha256),
            "source_universe": {"path": str(universe_path), "sha256": universe_sha256},
            "selection_plan": {"path": str(selection_path), "sha256": selection_sha256},
            "observation_manifest": {"path": str(output_root / "observations.json"), "sha256": observation_record["sha256"]},
            "same_sources_across_targets": True,
            "public_material_only": True,
            "source_text_written": False,
            "token_ids_written": False,
            "truth_opened": False,
        }
        panel_record = _write_create_only(output_root / "panel.json", panel_manifest)
        capture = {
            "schema": CAPTURE_SCHEMA,
            "task_id": TASK_ID,
            "status": "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH",
            "source_universe": {"path": str(universe_path), "bytes": universe_path.stat().st_size, "sha256": universe_sha256},
            "selection_plan": {"path": str(selection_path), "bytes": selection_path.stat().st_size, "sha256": selection_sha256},
            "source_pairing": {"same_record_ids_across_targets": True, "record_ids_sha256": record_ids_sha256},
            "public_inputs": {
                "pile": actual_sources["pile"],
                "finance": actual_sources["finance"],
                "tokenizer": actual_sources["tokenizer"],
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
                "ended_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started_clock,
                "command": list(sys.argv),
                "code_commit": _git_commit(root),
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
        return {"task_id": TASK_ID, "status": capture["status"], "observations": observation_record, "panel": panel_record, "capture": capture_record, "truth_opened": False}
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_create_only(
                failure_path,
                {
                    "schema": CAPTURE_SCHEMA,
                    "task_id": TASK_ID,
                    "status": "PUBLIC_OBSERVATIONS_CAPTURE_FAILED_NO_TRUTH",
                    "started_utc": started_utc,
                    "ended_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "selection_plan": str(selection_path),
                    "selection_plan_sha256": selection_sha256,
                    "source_universe": str(universe_path),
                    "truth_opened": False,
                    "source_text_written": False,
                    "token_ids_written": False,
                },
            )
        if isinstance(exc, CapturePreparationError):
            raise
        raise CapturePreparationError("P06 public capture failed") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required acknowledgment for model/public-prefix work")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    parser.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path)
    parser.add_argument("--lora-update", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = capture_public(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CapturePreparationError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"TRR-P06 capture error: {exc}")
