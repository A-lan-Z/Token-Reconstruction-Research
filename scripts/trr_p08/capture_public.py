#!/usr/bin/env python3
"""Capture a frozen TRR-P08 panel through the published P06 capture helpers.

The adapter keeps P08's source-universe and selection schemas while reusing the
reviewed P06 row materialization, padding, public-prefix capture, mask checks,
and inner resource guard.  It is create-only and requires ``--execute``.
No capture is run by setup; root must provide the frozen selection and an
exclusive guarded resource window first.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts import trr0004_produce_confirmation as trr4  # noqa: E402
from scripts import trr0005_produce_confirmation as trusted  # noqa: E402
from scripts.trr_p06 import capture_public as p06_capture  # noqa: E402
from scripts.trr_p08 import prepare_public_panel as panel  # noqa: E402


TASK_ID = "TRR-P08"
SELECTION_SCHEMA = panel.SELECTION_SCHEMA
SELECTION_STATUS = panel.SELECTION_STATUS
OBSERVATION_SCHEMA = "token-reconstruction.trr-p08-public-observation-manifest.v1"
OBSERVATION_FILE_SCHEMA = "token-reconstruction.trr-p08-public-observation.v1"
CAPTURE_SCHEMA = "token-reconstruction.trr-p08-public-capture.v1"
PANEL_SCHEMA = "token-reconstruction.trr-p08-public-source-panel.v1"
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


class CapturePreparationError(RuntimeError):
    """Raised when a P08 capture contract is not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return p06_capture._sha256_file(Path(path))


def _file_record(path: Path) -> dict[str, Any]:
    return p06_capture._file_record(Path(path))


def _json_digest(value: Any) -> str:
    return p06_capture._json_digest(value)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return p06_capture._write_create_only(Path(path), value)


def _git_commit(root: Path) -> str | None:
    return p06_capture._git_commit(Path(root))


def _configure_capture(universe: Mapping[str, Any]) -> None:
    provenance = universe.get("provenance")
    sources = universe.get("candidate_source_universe")
    if not isinstance(provenance, Mapping) or not isinstance(sources, Mapping):
        raise CapturePreparationError("P08 source universe lacks capture bindings")
    ranges = {
        style: sources.get(style, {}).get("candidate_range_half_open")
        for style in STYLE_ORDER
    }
    panel._configure_p06(seed=int(provenance["selection_seed"]), ranges=ranges)

    # The P06 helpers are pure implementation primitives here; their task and
    # schema constants are rebound to P08 for the isolated create-only output.
    p06_capture.TASK_ID = TASK_ID
    p06_capture.SELECTION_SCHEMA = SELECTION_SCHEMA
    p06_capture.SELECTION_STATUS = SELECTION_STATUS
    p06_capture.OBSERVATION_SCHEMA = OBSERVATION_SCHEMA
    p06_capture.OBSERVATION_FILE_SCHEMA = OBSERVATION_FILE_SCHEMA
    p06_capture.CAPTURE_SCHEMA = CAPTURE_SCHEMA
    p06_capture.PANEL_SCHEMA = PANEL_SCHEMA
    p06_capture.SEQUENCE_TOKENS = SEQUENCE_TOKENS
    p06_capture.SCORED_POST_BOS_TOKENS = SCORED_POST_BOS_TOKENS
    p06_capture.CAPTURE_SEQUENCE_TOKENS = CAPTURE_SEQUENCE_TOKENS
    p06_capture.CAPTURE_BATCH_RECORDS = CAPTURE_BATCH_RECORDS
    p06_capture.HIDDEN_SIZE = HIDDEN_SIZE
    p06_capture.VOCAB_SIZE = VOCAB_SIZE
    p06_capture.BOS_TOKEN_ID = BOS_TOKEN_ID
    p06_capture.PADDING_TOKEN_ID = PADDING_TOKEN_ID
    p06_capture.CONDITION_ORDER = CONDITION_ORDER
    p06_capture.STYLE_ORDER = STYLE_ORDER
    p06_capture.CELL_ORDER = CELL_ORDER
    p06_capture.panel = panel


def _validate_selection(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    try:
        return p06_capture._validate_selection(Path(path))
    except p06_capture.CapturePreparationError as exc:
        raise CapturePreparationError(str(exc)) from exc


def _validate_universe_binding(universe_path: Path, selection: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return p06_capture._validate_universe_binding(Path(universe_path), selection)
    except p06_capture.CapturePreparationError as exc:
        raise CapturePreparationError(str(exc)) from exc


def _capture_public(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise CapturePreparationError("P08 capture requires explicit --execute")
    root = Path(args.repository_root).expanduser().resolve()
    universe_path = Path(args.universe).expanduser().resolve()
    universe = panel.load_universe(universe_path, require_frozen=True)
    _configure_capture(universe)
    selection_path = Path(args.selection).expanduser().resolve()
    selection, selected_rows = _validate_selection(selection_path)
    universe = _validate_universe_binding(universe_path, selection)
    output_root = Path(args.output_root).expanduser()
    output_root = output_root if output_root.is_absolute() else root / output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(root / "experiments" / "TRR-P08")
    except ValueError as exc:
        raise CapturePreparationError("P08 capture output must be under experiments/TRR-P08") from exc
    if output_root.exists() or output_root.is_symlink():
        raise CapturePreparationError(f"P08 capture output is create-only: {output_root}")
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
            raise CapturePreparationError("P08 selection lacks frozen public sources")
        for key, descriptor in actual_sources.items():
            if dict(frozen_sources.get(key, {})) != dict(descriptor):
                raise CapturePreparationError(f"P08 {key} source differs from the frozen selection")
        tokenizer = trusted._load_tokenizer(tokenizer_path)
        datasets = {
            "pile": trusted._load_arrow_dataset(pile_paths),
            "finance": trusted._load_arrow_dataset(finance_paths),
        }
        records = p06_capture._materialize_selected(selected_rows, datasets=datasets, tokenizer=tokenizer)
        batches = p06_capture._batches(records)
        record_ids_sha256 = {
            style: _json_digest([row["record_id"] for row in selected_rows[style]])
            for style in STYLE_ORDER
        }
        device = trusted._device(args.device)
        model_snapshot = Path(args.model_snapshot).expanduser().resolve()
        if model_snapshot.is_symlink() or not model_snapshot.is_dir():
            raise CapturePreparationError(f"public model snapshot is unavailable: {model_snapshot}")
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
            current, receipt = p06_capture._capture_condition(
                condition=condition,
                records=records,
                batches=batches,
                model_snapshot=model_snapshot,
                lora_config_path=lora_config_path,
                lora_update_path=lora_update_path,
                output_root=output_root,
                records_per_domain=panel.RECORDS_PER_DOMAIN,
                record_ids_sha256=record_ids_sha256,
                selection_sha256=selection_sha256,
                device=device,
            )
            observations.update(current)
            conditions[condition] = receipt

        observation_manifest = p06_capture._observation_manifest(
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
        runtime_snapshot = trr4._runtime_snapshot(model_snapshot)
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
                "resource_policy": {
                    "minimum_free_gpu_bytes": p06_capture.MIN_FREE_GPU_BYTES,
                    "maximum_reserved_gpu_bytes": p06_capture.MAX_RESERVED_GPU_BYTES,
                    "maximum_host_rss_bytes": p06_capture.MAX_HOST_RSS_BYTES,
                    "minimum_host_available_bytes": p06_capture.MIN_HOST_AVAILABLE_BYTES,
                },
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
        raise CapturePreparationError("P08 public capture failed") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required acknowledgment for public-prefix work")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _capture_public(args)
    except (CapturePreparationError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"TRR-P08 capture error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
