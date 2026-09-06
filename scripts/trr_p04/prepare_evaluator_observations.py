#!/usr/bin/env python3
"""Prepare truth-free P04 evaluator observations for both target conditions.

The default execution path captures the frozen 72-record panel with the
pinned cut-4 prefix and writes only activations, masks, and positions.  The
source text/token IDs are materialized transiently and are never serialized.
``--preflight-only`` performs the same panel/target/geometry checks without
loading a tokenizer, dataset rows, model, target update, or evaluator truth.

This runner is intentionally separate from student prediction and scoring.
The evaluator target update is supplied through a private path and is loaded
only in the capture process; it is not copied into the observation artifact.
All output paths are create-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from scripts.trr_p04 import capture_public_pools as public_capture
from scripts.trr_p04 import prepare_panel as selector
from token_reconstruction.public_activation import (
    capture_public_prefix,
    pad_public_token_sequences,
    tensor_sha256,
    validate_activation_tensor,
)
from token_reconstruction.public_prefix import ContiguousPublicPrefix
from token_reconstruction.target_update import (
    TargetLoRAConfig,
    install_target_lora,
    load_target_lora,
)


TASK_ID = "TRR-P04"
SCHEMA = "token-reconstruction.trr-p04-evaluator-observations.v1"
SELECTION_SCHEMA = "token-reconstruction.trr-p04-public-selection.v1"
TARGET_PLAN_SCHEMA = "token-reconstruction.trr-p04-evaluator-target-plan.v1"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
DEFAULT_MODEL_SNAPSHOT = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6"
)
DEFAULT_TOKENIZER = DEFAULT_MODEL_SNAPSHOT
DEFAULT_TARGET_UPDATE = Path(
    "experiments/TRR-P04/private/evaluator_target_update/"
    "p04_evaluator_target_update_v1.safetensors"
)
PR7_CAPTURE_HELPER = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/"
    "TRR-0004/scripts/trr0004_prepare_public_activations.py"
)
MAXIMUM_TOKENS = 192
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
CUT_DEPTH = 4
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
CAPTURE_BATCH_SIZE = 8
MIN_FREE_GPU_BYTES = 8 * 2**30
MAX_RESERVED_GPU_BYTES = 6 * 2**30
MAX_HOST_RSS_BYTES = 16 * 2**30
CONDITIONS = ("public_base", "p04_evaluator_target_update_v1")
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


class EvaluatorObservationError(RuntimeError):
    """Raised when evaluator observation preparation fails closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _max_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _safe_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}


def _git_head() -> str | None:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise EvaluatorObservationError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluatorObservationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluatorObservationError(f"{label} must be a JSON object")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise EvaluatorObservationError(f"{label} must be a regular file: {path}")
    return path


def _descriptor(path: Path, *, role: str, hash_file: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise EvaluatorObservationError(f"{role} is unavailable: {path}")
    value: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": int(path.stat().st_size),
    }
    if hash_file:
        value["sha256"] = _sha256_file(path)
    return value


def _row_order_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise EvaluatorObservationError("fresh panel row has no record_id")
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_selection(
    selection: Mapping[str, Any], *, selection_path: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("task_id") != TASK_ID:
        raise EvaluatorObservationError("P04 selection identity changed")
    pools = selection.get("pools")
    fresh = pools.get("fresh_evaluation") if isinstance(pools, Mapping) else None
    rows = fresh.get("records") if isinstance(fresh, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 72:
        raise EvaluatorObservationError("fresh P04 panel must contain exactly 72 records")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    lengths = set(selector.PANEL_LENGTHS)
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping):
            raise EvaluatorObservationError(f"fresh panel row {index} is malformed")
        record_id = value.get("record_id")
        style = value.get("style")
        length = value.get("length_stratum")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise EvaluatorObservationError(f"fresh panel row {index} has a duplicate/empty ID")
        if style not in selector.STYLES or length not in lengths:
            raise EvaluatorObservationError(f"fresh panel row {index} has invalid style/length")
        if any(key in value for key in ("token_ids", "input_ids", "labels", "source_text", "truth", "oracle")):
            raise EvaluatorObservationError(f"fresh panel row {index} contains private fields")
        seen.add(record_id)
        normalized.append(
            {
                "record_id": record_id,
                "style": str(style),
                "length_stratum": int(length),
                "anchor": bool(value.get("anchor", False)),
                "dataset_id": str(value.get("dataset_id", "")),
                "dataset_revision": str(value.get("dataset_revision", "")),
                "row_index": int(value.get("row_index", -1)),
                "public_record_sha256": str(value.get("public_record_sha256", "")),
                "truncated_sequence_sha256": str(value.get("truncated_sequence_sha256", "")),
                "full_token_count": int(value.get("full_token_count", -1)),
                "post_bos_token_count": int(value.get("post_bos_token_count", -1)),
            }
        )
    for style in selector.STYLES:
        for length in selector.PANEL_LENGTHS:
            cell = [row for row in normalized if row["style"] == style and row["length_stratum"] == length]
            if len(cell) != 6:
                raise EvaluatorObservationError(f"fresh panel quota changed for {style}/{length}")
    anchors = [row for row in normalized if row["anchor"]]
    if len(anchors) != 12:
        raise EvaluatorObservationError("fresh panel anchor quota changed")
    for style in selector.STYLES:
        cell = [row for row in normalized if row["style"] == style and row["length_stratum"] == 32]
        if [row["record_id"] for row in anchors if row["style"] == style] != [row["record_id"] for row in cell[:4]]:
            raise EvaluatorObservationError(f"anchor rule changed for {style}")
        if any(row["anchor"] for row in normalized if row["style"] == style and row["length_stratum"] != 32):
            raise EvaluatorObservationError(f"anchor escaped the 32-token cell for {style}")
    panel = selection.get("panel")
    if not isinstance(panel, Mapping) or panel.get("independent_source_records") != 72:
        raise EvaluatorObservationError("selection panel metadata is inconsistent")
    bound_path = (selection_path or Path("experiments/TRR-P04/setup/public_selection-r2.json")).expanduser().resolve()
    return normalized, {
        "path": str(bound_path),
        "sha256": _sha256_file(bound_path),
        "record_count": len(normalized),
        "record_order_sha256": _row_order_sha256(normalized),
        "anchor_count": len(anchors),
        "anchor_order_sha256": _row_order_sha256(anchors),
    }


def _validate_target_plan(plan: Mapping[str, Any], *, selection: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != TARGET_PLAN_SCHEMA or plan.get("task_id") != TASK_ID:
        raise EvaluatorObservationError("evaluator target plan identity changed")
    if plan.get("condition_id") != "p04_evaluator_target_update_v1":
        raise EvaluatorObservationError("evaluator target condition changed")
    base = plan.get("base_model")
    if not isinstance(base, Mapping) or base.get("id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise EvaluatorObservationError("target plan model identity changed")
    update = plan.get("update")
    expected_update = {
        "family": "evaluator_only_lora",
        "layers": [0, 1, 2, 3],
        "modules": ["q_proj", "v_proj"],
        "rank": 4,
        "alpha": 4.0,
        "initialization_seed": 20260910,
        "optimizer": "AdamW",
        "learning_rate": 0.00075,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "steps": 30,
        "record_batch_size": 8,
        "maximum_tokens_including_bos": MAXIMUM_TOKENS,
    }
    if not isinstance(update, Mapping) or any(update.get(key) != value for key, value in expected_update.items()):
        raise EvaluatorObservationError("evaluator target update configuration changed")
    corpus = plan.get("update_corpus")
    if not isinstance(corpus, Mapping) or corpus.get("dataset_id") != "HuggingFaceH4/no_robots" or corpus.get("dataset_revision") != "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b":
        raise EvaluatorObservationError("evaluator target corpus identity changed")
    source_selection = corpus.get("selection")
    if not isinstance(source_selection, Mapping) or source_selection.get("seed") != 20260910 or source_selection.get("records") != 256 or source_selection.get("selected_row_order_sha256") != "42fb5bb7dfc58dba8ccf9e3b288e787fd88cc1788e936ee934cfbbdc86de2fd2":
        raise EvaluatorObservationError("evaluator target row selection changed")
    if corpus.get("student_training_access") is not False or corpus.get("teacher_access") is not False or corpus.get("fresh_panel_access") is not False:
        raise EvaluatorObservationError("evaluator target corpus access boundary changed")
    overlap = plan.get("overlap_and_drift")
    if not isinstance(overlap, Mapping) or overlap.get("audit_status") not in ("PENDING_TARGET_PREPARATION", "PASS"):
        raise EvaluatorObservationError("evaluator target overlap audit state is invalid")
    # The target plan binds the exact frozen panel before any target artifact
    # is loaded.  No selection score or evaluation truth is consulted here.
    del selection
    return {
        "condition_id": str(plan["condition_id"]),
        "lineage_id": str(plan["lineage_id"]),
        "seed": int(update["initialization_seed"]),
        "config": {
            "layers": list(update["layers"]),
            "modules": list(update["modules"]),
            "rank": int(update["rank"]),
            "alpha": float(update["alpha"]),
        },
    }


def _safe_record_index(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the attack-facing index; source hashes/row indices stay private."""

    return [
        {
            "record_id": str(row["record_id"]),
            "style": str(row["style"]),
            "length_stratum": int(row["length_stratum"]),
            "anchor": bool(row["anchor"]),
        }
        for row in rows
    ]


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise EvaluatorObservationError(f"refusing to overwrite create-only output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_preflight(
    *,
    selection_path: Path,
    target_plan_path: Path,
    output_root: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    selection = _load_object(selection_path, label="P04 selection")
    rows, panel = _validate_selection(selection, selection_path=selection_path)
    target_plan = _load_object(target_plan_path, label="evaluator target plan")
    target = _validate_target_plan(target_plan, selection=selection)
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
            raise EvaluatorObservationError(f"preflight output must be a new empty directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    value = {
        "schema": f"{SCHEMA}-preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS_NO_MODEL_NO_TARGET_NO_TRUTH",
        "created_utc": _utc_now(),
        "selection": {**panel, "path": str(selection_path.expanduser().resolve()), "sha256": _sha256_file(selection_path)},
        "target_plan": {"path": str(target_plan_path.expanduser().resolve()), "sha256": _sha256_file(target_plan_path), **target},
        "geometry": {
            "records_per_condition": len(rows),
            "padded_tokens_including_bos": MAXIMUM_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "cut_depth": CUT_DEPTH,
            "capture_batch_size": CAPTURE_BATCH_SIZE,
            "paired_conditions": list(CONDITIONS),
            "anchor_records": 12,
            "anchor_post_bos_positions_per_target": 384,
        },
        "serialized_observation_keys": ["activations", "attention_mask", "position_ids"],
        "forbidden_serialized_fields": ["token_ids", "input_ids", "labels", "source_text", "truth", "oracle", "target_weights"],
        "access": {
            "selection_metadata_read": True,
            "source_rows_read": False,
            "model_loaded": False,
            "target_update_loaded": False,
            "evaluation_truth_opened": False,
        },
        "expected_outputs": {
            "public_base": "observations/public_base.safetensors",
            "p04_evaluator_target_update_v1": "observations/p04_evaluator_target_update_v1.safetensors",
            "index": "observation_index.json",
            "evidence": "capture_evidence.json",
        },
        "execution": {
            "argv": list(argv),
            "safe_environment": _safe_environment(),
            "git_commit": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    _write_create_only(output_root / "evaluator_capture_preflight.json", value)
    return value


def _load_pr7_capture_helper() -> Any:
    path = PR7_CAPTURE_HELPER.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise EvaluatorObservationError(f"immutable PR7 capture helper is unavailable: {path}")
    spec = importlib.util.spec_from_file_location("trr_p04_pr7_capture_helper", path)
    if spec is None or spec.loader is None:
        raise EvaluatorObservationError("cannot import immutable PR7 capture helper")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cuda_guard(device: torch.device, *, stage: str, started: float) -> dict[str, Any]:
    if device.type != "cuda":
        return {"stage": stage, "device": str(device), "status": "host_only"}
    try:
        free, total = torch.cuda.mem_get_info(device)
        reserved = int(torch.cuda.memory_reserved(device))
    except Exception as exc:
        raise EvaluatorObservationError(f"CUDA resource data unavailable at {stage}") from exc
    rss = _max_rss_bytes()
    if free < MIN_FREE_GPU_BYTES or reserved > MAX_RESERVED_GPU_BYTES or rss > MAX_HOST_RSS_BYTES:
        raise EvaluatorObservationError(
            f"resource guard failed at {stage}: free={free}, reserved={reserved}, rss={rss}"
        )
    return {
        "stage": stage,
        "device": str(device),
        "status": "PASS",
        "free_bytes": int(free),
        "total_bytes": int(total),
        "reserved_bytes": reserved,
        "rss_bytes": rss,
        "minimum_free_bytes": MIN_FREE_GPU_BYTES,
        "maximum_reserved_bytes": MAX_RESERVED_GPU_BYTES,
        "maximum_host_rss_bytes": MAX_HOST_RSS_BYTES,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def _source_paths(selection: Mapping[str, Any]) -> dict[str, tuple[Path, ...]]:
    return public_capture._source_paths(selection)


def _load_fresh_sources(selection: Mapping[str, Any]) -> dict[str, Any]:
    paths = _source_paths(selection)
    result: dict[str, Any] = {}
    try:
        from datasets import Dataset, concatenate_datasets

        for style in selector.STYLES:
            parts = [Dataset.from_file(str(path)) for path in paths[style]]
            result[style] = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    except Exception as exc:
        raise EvaluatorObservationError("unable to load pinned public source Arrow files") from exc
    return result


def _materialize_fresh(
    selection: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    datasets: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    sequences: list[list[int]] = []
    safe_rows: list[dict[str, Any]] = []
    for ordinal, declared in enumerate(rows):
        style = str(declared["style"])
        row_index = int(declared["row_index"])
        dataset = datasets.get(style)
        if dataset is None or row_index < 0 or row_index >= len(dataset):
            raise EvaluatorObservationError(f"fresh row {ordinal} source index is unavailable")
        source_row = dataset[row_index]
        candidate = selector._candidate_from_row(style, row_index, source_row, tokenizer)
        if candidate is None:
            raise EvaluatorObservationError(f"fresh row {ordinal} became invalid")
        values = public_capture._tokens_for_row(style, source_row, tokenizer)
        checks = {
            "record_id": candidate.record_id,
            "public_record_sha256": candidate.public_record_sha256,
            "truncated_sequence_sha256": candidate.truncated_sequence_sha256,
            "full_token_count": candidate.full_token_count,
            "post_bos_token_count": candidate.post_bos_token_count,
        }
        for key, actual in checks.items():
            if str(declared.get(key)) != str(actual):
                raise EvaluatorObservationError(f"fresh row binding changed for {candidate.record_id}: {key}")
        # The student/scorer contract expects exactly the declared metric
        # window.  Truncate before capture so no future activations are
        # exposed to the attack artifact; the fixed 192-token batch remains
        # the padded execution geometry.
        target_tokens = 1 + int(declared["length_stratum"])
        captured = values[:target_tokens]
        if len(captured) < 2 or captured[0] != BOS_TOKEN_ID:
            raise EvaluatorObservationError(f"fresh row {ordinal} lost BOS/current-token geometry")
        sequences.append([int(value) for value in captured])
        safe_rows.append(
            {
                "record_id": str(declared["record_id"]),
                "style": style,
                "length_stratum": int(declared["length_stratum"]),
                "anchor": bool(declared["anchor"]),
                "active_token_count": len(captured),
                "padded_tokens": MAXIMUM_TOKENS,
            }
        )
    # The materializer receives the frozen order and must not sort/adapt it.
    if [row["record_id"] for row in safe_rows] != [str(row["record_id"]) for row in rows]:
        raise EvaluatorObservationError("fresh source materialization changed panel order")
    return sequences, safe_rows


def _load_base_prefix(snapshot: Path, *, device: torch.device) -> tuple[Any, dict[str, Any]]:
    helper = _load_pr7_capture_helper()
    prefix, snapshot_evidence, model_config = helper._load_public_prefix(
        snapshot.expanduser().resolve(), device=device, cut_depth=CUT_DEPTH
    )
    return prefix, {"condition": "public_base", "snapshot": snapshot_evidence, "model_config": model_config}


def _load_target_prefix(
    snapshot: Path,
    update_path: Path,
    target_plan: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    update = target_plan["update"]
    config = TargetLoRAConfig(
        layers=tuple(int(value) for value in update["layers"]),
        modules=tuple(str(value) for value in update["modules"]),
        rank=int(update["rank"]),
        alpha=float(update["alpha"]),
        seed=int(update["initialization_seed"]),
    )
    update_path = _regular_file(update_path, label="evaluator target update")
    update_descriptor = _descriptor(update_path, role="evaluator target update", hash_file=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot.expanduser().resolve()),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        if int(model.config.hidden_size) != HIDDEN_SIZE or int(model.config.vocab_size) != VOCAB_SIZE:
            raise EvaluatorObservationError("evaluator target model geometry changed")
        model.requires_grad_(False)
        installed = install_target_lora(model, config)
        load_target_lora(installed, update_path)
        prefix = ContiguousPublicPrefix(model, cut_depth=CUT_DEPTH).to(device).eval()
    except EvaluatorObservationError:
        raise
    except Exception as exc:
        raise EvaluatorObservationError("evaluator target prefix loading failed") from exc
    return prefix, {
        "condition": "p04_evaluator_target_update_v1",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(snapshot.resolve())},
        "lora_config": {
            "layers": list(config.layers),
            "modules": list(config.modules),
            "rank": config.rank,
            "alpha": config.alpha,
            "initialization_seed": config.seed,
        },
        "target_update": update_descriptor,
        "target_weights_available_to_reconstructor": False,
    }


def _save_observation(
    path: Path,
    *,
    activations: torch.Tensor,
    batch: Any,
    condition: str,
    selection_descriptor: Mapping[str, Any],
    target_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    validate_activation_tensor(activations, batch, hidden_size=HIDDEN_SIZE)
    if tuple(activations.shape) != (72, MAXIMUM_TOKENS, HIDDEN_SIZE):
        raise EvaluatorObservationError("evaluator observation geometry changed")
    if path.exists() or path.is_symlink():
        raise EvaluatorObservationError(f"evaluator observation is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "condition": condition,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "cut_depth": str(CUT_DEPTH),
        "hidden_size": str(HIDDEN_SIZE),
        "record_count": "72",
        "maximum_tokens_including_bos": str(MAXIMUM_TOKENS),
        "activation_dtype": "bfloat16",
        "current_token_alignment": "activations[record,position] predicts the current token at that position",
        "record_order_sha256": str(selection_descriptor["record_order_sha256"]),
        "selection_sha256": str(selection_descriptor["sha256"]),
        "activations_sha256": tensor_sha256(activations),
        "attention_mask_sha256": tensor_sha256(batch.attention_mask),
        "position_ids_sha256": tensor_sha256(batch.position_ids),
        "source_text_materialized_transiently": "true",
        "source_tokens_materialized_transiently": "true",
        "source_tokens_serialized": "false",
        "evaluation_truth_opened": "false",
        "target_update_weights_serialized": "false",
        "target_lineage_id": str(target_descriptor.get("lineage_id", "public_base")),
        "forbidden_fields": "token_ids,input_ids,labels,source_text,truth,oracle,target_weights",
    }
    save_file(
        {
            "activations": activations.detach().cpu().contiguous(),
            "attention_mask": batch.attention_mask.detach().cpu().contiguous(),
            "position_ids": batch.position_ids.detach().cpu().contiguous(),
        },
        str(path),
        metadata=metadata,
    )
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "activation_tensor_sha256": tensor_sha256(activations),
        "attention_mask_sha256": tensor_sha256(batch.attention_mask),
        "position_ids_sha256": tensor_sha256(batch.position_ids),
        "shape": list(activations.shape),
        "condition": condition,
        "serialized_token_ids": False,
    }


def capture(
    *,
    selection_path: Path,
    target_plan_path: Path,
    model_snapshot: Path,
    tokenizer_path: Path,
    target_update_path: Path,
    output_root: Path,
    device: torch.device,
    argv: Sequence[str],
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    started_utc = _utc_now()
    selection = _load_object(selection_path, label="P04 selection")
    rows, panel = _validate_selection(selection, selection_path=selection_path)
    target_plan = _load_object(target_plan_path, label="evaluator target plan")
    target = _validate_target_plan(target_plan, selection=selection)
    model_snapshot = model_snapshot.expanduser().resolve()
    if not model_snapshot.is_dir() or model_snapshot.is_symlink():
        raise EvaluatorObservationError(f"model snapshot is unavailable: {model_snapshot}")
    tokenizer_path = tokenizer_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
            raise EvaluatorObservationError(f"capture output must be a new empty directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    guards: list[dict[str, Any]] = []
    guards.append(_cuda_guard(device, stage="before_source_and_model_load", started=started_perf))
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    except Exception as exc:
        raise EvaluatorObservationError("pinned tokenizer loading failed") from exc
    datasets = _load_fresh_sources(selection)
    sequences, safe_rows = _materialize_fresh(selection, rows, datasets=datasets, tokenizer=tokenizer)
    batch = pad_public_token_sequences(
        sequences,
        maximum_tokens=MAXIMUM_TOKENS,
        pad_token_id=PAD_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
    )
    selection_descriptor = {
        **panel,
        "path": str(selection_path.resolve()),
        "sha256": _sha256_file(selection_path),
    }
    target_descriptor = {"lineage_id": target["lineage_id"], "condition_id": target["condition_id"]}
    observations: dict[str, Any] = {}
    phase_records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        guards.append(_cuda_guard(device, stage=f"before_{condition}_model_load", started=started_perf))
        phase_started = time.perf_counter()
        if condition == "public_base":
            prefix, load_evidence = _load_base_prefix(model_snapshot, device=device)
        else:
            prefix, load_evidence = _load_target_prefix(
                model_snapshot,
                target_update_path,
                target_plan,
                device=device,
            )
        guards.append(_cuda_guard(device, stage=f"after_{condition}_model_load", started=started_perf))
        capture_started = time.perf_counter()
        try:
            activations = capture_public_prefix(
                prefix,
                batch,
                device=device,
                batch_size=CAPTURE_BATCH_SIZE,
                resource_check=lambda: _cuda_guard(device, stage=f"during_{condition}_capture", started=started_perf),
            )
        except Exception as exc:
            raise EvaluatorObservationError(f"{condition} evaluator observation capture failed") from exc
        artifact = _save_observation(
            output_root / "observations" / f"{condition}.safetensors",
            activations=activations,
            batch=batch,
            condition=condition,
            selection_descriptor=selection_descriptor,
            target_descriptor=target_descriptor if condition != "public_base" else {"lineage_id": "public_base"},
        )
        phase_records.append(
            {
                "condition": condition,
                "load_seconds": round(capture_started - phase_started, 6),
                "capture_seconds": round(time.perf_counter() - capture_started, 6),
                "load_evidence": load_evidence,
                "artifact": artifact,
            }
        )
        observations[condition] = artifact
        del prefix
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        guards.append(_cuda_guard(device, stage=f"after_{condition}_release", started=started_perf))
    index = {
        "schema": f"{SCHEMA}-index.v1",
        "task_id": TASK_ID,
        "status": "EVALUATOR_OBSERVATION_INDEX_READY_NO_TRUTH",
        "selection": selection_descriptor,
        "record_order_sha256": selection_descriptor["record_order_sha256"],
        "records": safe_rows,
        "conditions": list(CONDITIONS),
        "serialized_source_or_truth": False,
        "serialized_token_ids": False,
    }
    index_path = output_root / "observation_index.json"
    _write_create_only(index_path, index)
    evidence = {
        "schema": f"{SCHEMA}-capture-evidence.v1",
        "task_id": TASK_ID,
        "status": "PASS_EVALUATOR_OBSERVATIONS_NO_TRUTH",
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "selection": selection_descriptor,
        "target_plan": {
            "path": str(target_plan_path.resolve()),
            "sha256": _sha256_file(target_plan_path),
            "lineage_id": target["lineage_id"],
            "condition_id": target["condition_id"],
        },
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(model_snapshot)},
        "geometry": {
            "records": len(rows),
            "tokens_including_bos": MAXIMUM_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "cut_depth": CUT_DEPTH,
            "batch_size": CAPTURE_BATCH_SIZE,
        },
        "phases": phase_records,
        "observations": observations,
        "observation_index": {
            "path": str(index_path),
            "bytes": int(index_path.stat().st_size),
            "sha256": _sha256_file(index_path),
        },
        "guards": guards,
        "access": {
            "source_text_materialized_transiently": True,
            "source_tokens_materialized_transiently": True,
            "source_tokens_serialized": False,
            "evaluation_truth_opened": False,
            "target_update_weights_loaded_only_in_target_capture": True,
            "target_update_weights_serialized": False,
            "student_states_loaded": False,
        },
        "execution": {
            "argv": list(argv),
            "safe_environment": _safe_environment(),
            "git_commit": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "elapsed_seconds": round(time.perf_counter() - started_perf, 6),
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    _write_create_only(output_root / "capture_evidence.json", evidence)
    _write_create_only(
        output_root / "target_lineage_summary.json",
        {
            "schema": "token-reconstruction.trr-p04-target-lineage-summary.v1",
            "task_id": TASK_ID,
            "status": "CAPTURE_COMPLETE_NO_TRUTH",
            "condition_id": target["condition_id"],
            "lineage_id": target["lineage_id"],
            "target_plan": {"path": str(target_plan_path.resolve()), "sha256": _sha256_file(target_plan_path)},
            "target_update_weights_included": False,
            "evaluation_truth_opened": False,
            "activation_drift": "reported from paired observation tensors in a later diagnostic; no truth-dependent selection",
        },
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-update", type=Path, default=DEFAULT_TARGET_UPDATE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.preflight_only:
            value = build_preflight(
                selection_path=args.selection.expanduser().resolve(),
                target_plan_path=args.target_plan.expanduser().resolve(),
                output_root=args.output_root.expanduser().resolve(),
                argv=list(sys.argv if argv is None else [sys.argv[0], *argv]),
            )
        else:
            value = capture(
                selection_path=args.selection.expanduser().resolve(),
                target_plan_path=args.target_plan.expanduser().resolve(),
                model_snapshot=args.model_snapshot,
                tokenizer_path=args.tokenizer,
                target_update_path=args.target_update,
                output_root=args.output_root,
                device=torch.device(args.device),
                argv=list(sys.argv if argv is None else [sys.argv[0], *argv]),
            )
        print(json.dumps({"status": value["status"], "output_root": str(args.output_root.resolve())}, sort_keys=True))
        return 0
    except (EvaluatorObservationError, RuntimeError) as exc:
        print(f"P04 evaluator observation preparation failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
