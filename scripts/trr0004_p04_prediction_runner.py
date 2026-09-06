#!/usr/bin/env python3
"""Run the truth-free P04 student prediction/timing phase.

This implementation-owned runner consumes only the evaluator's activation
observations, the frozen public panel metadata, the selected student states,
and the fixed public normalized embedding table.  It never opens evaluator
truth, source text, or token IDs.  A cell is written only after one warmup and
three synchronized measured passes agree exactly.  The full 72-record result
is frozen per method/seed/condition; the predeclared 12-record anchor subset
is timed separately and checked against the corresponding full-panel rows.

The same executable also creates the selected-state manifest and provides a
small CPU-only smoke path.  All output directories and files are create-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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
from safetensors import safe_open

from token_reconstruction.p04_student import (
    ALL_METHODS,
    METHOD_AFFINE,
    P04StudentError,
    PREDICTION_SCHEMA,
    StudentArchitectureConfig,
    load_student_state,
    prediction_tensor,
    validate_embedding_table,
)
from token_reconstruction.public_activation import tensor_sha256


TASK_ID = "TRR-P04"
STATE_MANIFEST_SCHEMA = "token-reconstruction.trr-p04-selected-state-manifest.v1"
RUNNER_SCHEMA = "token-reconstruction.trr-p04-student-prediction-runner.v1"
CELL_SCHEMA = "token-reconstruction.trr-p04-student-prediction-cell.v1"
TIMING_SCHEMA = "token-reconstruction.trr-p04-student-prediction-timing.v1"
TIE_SCHEMA = "token-reconstruction.trr-p04-tie-diagnostics.v1"
FREEZE_SCHEMA = "token-reconstruction.trr-p04-student-prediction-freeze.v1"
OBSERVATION_SCHEMA = "token-reconstruction.trr-p04-evaluator-observations.v1"
STATE_SCHEMA = "token-reconstruction.trr-p04-student-state.v1"
SELECTION_SCHEMA = "token-reconstruction.trr-p04-public-selection.v1"
CONDITIONS = ("public_base", "p04_evaluator_target_update_v1")
METHODS = ("affine_same_data", "student_s", "student_h", "student_d")
SEEDS = (1737, 2711)
PANEL_LENGTHS = (16, 32, 64, 128)
EXPECTED_ROWS = 72
EXPECTED_SEQUENCE = 192
EXPECTED_HIDDEN = 2048
EXPECTED_VOCAB = 128256
EXPECTED_GRU_WIDTH = 256
EXPECTED_RECORD_BATCH = 8
EXPECTED_PROJECTION_CHUNK = 512
EXPECTED_WARMUPS = 1
EXPECTED_MEASUREMENTS = 3
DEFAULT_TABLE = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/"
    "TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors"
)


class PredictionRunnerError(RuntimeError):
    """Raised when a public prediction or binding contract fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _max_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  Keep this helper portable for smoke use.
    multiplier = 1024 if sys.platform.startswith("linux") else 1
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * multiplier


def _git_head() -> str | None:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _safe_environment() -> dict[str, str]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "PYTHONPATH",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def _regular_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PredictionRunnerError(f"{label} must be a regular file: {path}")
    return path


def _regular_dir(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_dir():
        raise PredictionRunnerError(f"{label} must be a regular directory: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descriptor(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PredictionRunnerError(f"output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text_create_only(path: Path, value: str) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PredictionRunnerError(f"output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionRunnerError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PredictionRunnerError(f"{label} must be a JSON object: {path}")
    return value


def record_order_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(records):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise PredictionRunnerError(f"record {index} has no record_id")
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _records_from_selection(payload: Mapping[str, Any], *, path: Path) -> list[dict[str, Any]]:
    if payload.get("schema") != SELECTION_SCHEMA or payload.get("task_id") != TASK_ID:
        raise PredictionRunnerError("P04 selection identity changed")
    pools = payload.get("pools")
    fresh = pools.get("fresh_evaluation") if isinstance(pools, Mapping) else None
    rows = fresh.get("records") if isinstance(fresh, Mapping) else None
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise PredictionRunnerError("P04 selection must contain exactly 72 fresh records")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping):
            raise PredictionRunnerError(f"selection record {index} is malformed")
        record_id = value.get("record_id")
        style = value.get("style")
        length = value.get("length_stratum")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise PredictionRunnerError(f"selection record {index} has a duplicate/empty ID")
        if not isinstance(style, str) or not style or length not in PANEL_LENGTHS:
            raise PredictionRunnerError(f"selection record {index} has invalid style/length")
        forbidden = {"token_ids", "input_ids", "labels", "source_text", "truth", "oracle"}
        if forbidden.intersection(value):
            raise PredictionRunnerError(f"selection record {index} contains source/truth fields")
        seen.add(record_id)
        result.append(
            {
                "record_id": record_id,
                "style": style,
                "length_stratum": int(length),
                "anchor": bool(value.get("anchor", False)),
            }
        )
    styles = {str(row["style"]) for row in result}
    if len(styles) != 3:
        raise PredictionRunnerError("selection must contain exactly three styles")
    for length in PANEL_LENGTHS:
        if sum(row["length_stratum"] == length for row in result) != 18:
            raise PredictionRunnerError(f"selection length quota changed for {length}")
    anchors = [row for row in result if row["anchor"]]
    if len(anchors) != 12 or any(row["length_stratum"] != 32 for row in anchors):
        raise PredictionRunnerError("selection anchor quota or length changed")
    for style in sorted(styles):
        cell = [row for row in result if row["style"] == style and row["length_stratum"] == 32]
        declared = [row["record_id"] for row in anchors if row["style"] == style]
        if declared != [row["record_id"] for row in cell[:4]]:
            raise PredictionRunnerError(f"selection anchor order changed for {style}")
    return result


def _load_selection(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _regular_file(path, label="P04 selection")
    payload = _load_json(path, label="P04 selection")
    records = _records_from_selection(payload, path=path)
    anchors = [row for row in records if row["anchor"]]
    return records, {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "record_count": len(records),
        "record_order_sha256": record_order_sha256(records),
        "anchor_count": len(anchors),
        "anchor_order_sha256": record_order_sha256(anchors),
    }


def _validate_observation_index(
    path: Path,
    *,
    selection_records: Sequence[Mapping[str, Any]],
    selection_descriptor: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _regular_file(path, label="evaluator observation index")
    payload = _load_json(path, label="evaluator observation index")
    expected_prefix = f"{OBSERVATION_SCHEMA}-index."
    if payload.get("schema") != f"{OBSERVATION_SCHEMA}-index.v1" and not str(payload.get("schema", "")).startswith(expected_prefix):
        raise PredictionRunnerError("evaluator observation index schema changed")
    if payload.get("task_id") != TASK_ID:
        raise PredictionRunnerError("evaluator observation index task changed")
    if payload.get("serialized_source_or_truth") is not False or payload.get("serialized_token_ids") is not False:
        raise PredictionRunnerError("evaluator observation index exposes source or truth")
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise PredictionRunnerError("evaluator observation index must contain 72 records")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping):
            raise PredictionRunnerError(f"observation index record {index} is malformed")
        record_id = value.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise PredictionRunnerError(f"observation index record {index} has duplicate/empty ID")
        if any(key in value for key in ("token_ids", "source_text", "truth", "labels", "oracle")):
            raise PredictionRunnerError("observation index contains source or truth fields")
        length = value.get("length_stratum")
        if length not in PANEL_LENGTHS:
            raise PredictionRunnerError(f"observation index record {record_id} has invalid length")
        normalized.append(
            {
                "record_id": record_id,
                "style": str(value.get("style", "")),
                "length_stratum": int(length),
                "anchor": bool(value.get("anchor", False)),
                "active_token_count": int(value.get("active_token_count", int(length) + 1)),
                "padded_tokens": int(value.get("padded_tokens", EXPECTED_SEQUENCE)),
            }
        )
        seen.add(record_id)
    selection_ids = [str(row["record_id"]) for row in selection_records]
    index_ids = [str(row["record_id"]) for row in normalized]
    if index_ids != selection_ids:
        raise PredictionRunnerError("observation index record order differs from frozen selection")
    for expected, actual in zip(selection_records, normalized):
        if actual["style"] != str(expected["style"]):
            raise PredictionRunnerError(f"observation style changed for {actual['record_id']}")
        if actual["length_stratum"] != int(expected["length_stratum"]):
            raise PredictionRunnerError(f"observation length changed for {actual['record_id']}")
        if actual["anchor"] != bool(expected["anchor"]):
            raise PredictionRunnerError(f"observation anchor flag changed for {actual['record_id']}")
        if actual["active_token_count"] != int(expected["length_stratum"]) + 1:
            raise PredictionRunnerError(f"observation active-token count changed for {actual['record_id']}")
        if actual["padded_tokens"] != EXPECTED_SEQUENCE:
            raise PredictionRunnerError(f"observation padded geometry changed for {actual['record_id']}")
    if payload.get("record_order_sha256") != selection_descriptor["record_order_sha256"]:
        raise PredictionRunnerError("observation index record-order hash changed")
    conditions = payload.get("conditions")
    if conditions != list(CONDITIONS):
        raise PredictionRunnerError("observation condition order changed")
    declared_selection = payload.get("selection")
    if isinstance(declared_selection, Mapping):
        if declared_selection.get("sha256") != selection_descriptor["sha256"]:
            raise PredictionRunnerError("observation index selection binding changed")
        if declared_selection.get("record_order_sha256") != selection_descriptor["record_order_sha256"]:
            raise PredictionRunnerError("observation index selection order binding changed")
    descriptor = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "record_count": len(normalized),
        "record_order_sha256": record_order_sha256(normalized),
        "conditions": list(CONDITIONS),
    }
    return payload, {"descriptor": descriptor, "records": normalized}


def _metadata(path: Path) -> dict[str, str]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            value = handle.metadata()
    except Exception as exc:
        raise PredictionRunnerError(f"cannot read safetensors metadata: {path}") from exc
    return dict(value or {})


def _validate_observation_artifact(
    path: Path,
    *,
    condition: str,
    records: Sequence[Mapping[str, Any]],
    selection_descriptor: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = _regular_file(path, label=f"{condition} evaluator observation")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            forbidden = keys.intersection({"token_ids", "input_ids", "labels", "source_text", "truth", "oracle", "target_weights"})
            if forbidden:
                raise PredictionRunnerError(f"{condition} observation contains forbidden fields: {sorted(forbidden)}")
            required = {"activations", "attention_mask", "position_ids"}
            if keys != required:
                raise PredictionRunnerError(f"{condition} observation keys changed: {sorted(keys)}")
            activations = handle.get_tensor("activations").contiguous()
            mask = handle.get_tensor("attention_mask").contiguous()
            position_ids = handle.get_tensor("position_ids").contiguous()
            metadata = dict(handle.metadata() or {})
    except PredictionRunnerError:
        raise
    except Exception as exc:
        raise PredictionRunnerError(f"cannot read {condition} evaluator observation: {path}") from exc
    if tuple(activations.shape) != (EXPECTED_ROWS, EXPECTED_SEQUENCE, EXPECTED_HIDDEN):
        raise PredictionRunnerError(f"{condition} activation geometry changed")
    if activations.dtype != torch.bfloat16:
        raise PredictionRunnerError(f"{condition} activation dtype changed: {activations.dtype}")
    if not torch.isfinite(activations).all().item():
        raise PredictionRunnerError(f"{condition} activations are non-finite")
    if tuple(mask.shape) != (EXPECTED_ROWS, EXPECTED_SEQUENCE) or mask.dtype not in (torch.bool, torch.uint8):
        raise PredictionRunnerError(f"{condition} attention-mask geometry or dtype changed")
    if tuple(position_ids.shape) != (EXPECTED_ROWS, EXPECTED_SEQUENCE) or position_ids.dtype != torch.int64:
        raise PredictionRunnerError(f"{condition} position-id geometry or dtype changed")
    # Preserve the serialized representation for the binding hash.  Setup
    # writes uint8 masks; use a separate boolean view only for geometry checks.
    # Hashing the cast view would reject an otherwise unchanged setup artifact.
    serialized_mask = mask
    mask = serialized_mask.to(dtype=torch.bool)
    if not mask[:, 0].all().item() or not mask[:, 1:].any(dim=1).all().item():
        raise PredictionRunnerError(f"{condition} observation mask lacks BOS or active positions")
    if not torch.equal(mask, mask.cumprod(dim=1).to(torch.bool)):
        raise PredictionRunnerError(f"{condition} observation mask is not right-padded")
    expected_positions = torch.arange(EXPECTED_SEQUENCE, dtype=torch.int64).expand(EXPECTED_ROWS, -1)
    for row, record in enumerate(records):
        active_count = int(mask[row].sum().item())
        if active_count != int(record["length_stratum"]) + 1:
            raise PredictionRunnerError(f"{condition} active length changed for {record['record_id']}")
        if not torch.equal(position_ids[row, :active_count], expected_positions[row, :active_count]):
            raise PredictionRunnerError(f"{condition} active position IDs changed for {record['record_id']}")
        if active_count < EXPECTED_SEQUENCE and not position_ids[row, active_count:].eq(0).all().item():
            raise PredictionRunnerError(f"{condition} padded position IDs changed for {record['record_id']}")
    if metadata.get("schema") != OBSERVATION_SCHEMA or metadata.get("task_id") != TASK_ID:
        raise PredictionRunnerError(f"{condition} observation metadata identity changed")
    if metadata.get("condition") != condition:
        raise PredictionRunnerError(f"{condition} observation metadata condition changed")
    if metadata.get("selection_sha256") != selection_descriptor["sha256"]:
        raise PredictionRunnerError(f"{condition} observation selection binding changed")
    if metadata.get("record_order_sha256") != selection_descriptor["record_order_sha256"]:
        raise PredictionRunnerError(f"{condition} observation order binding changed")
    for key, tensor in (
        ("activations_sha256", activations),
        ("attention_mask_sha256", serialized_mask),
        ("position_ids_sha256", position_ids),
    ):
        if metadata.get(key) != tensor_sha256(tensor):
            raise PredictionRunnerError(f"{condition} {key} changed")
    if metadata.get("source_tokens_serialized") != "false" or metadata.get("evaluation_truth_opened") != "false":
        raise PredictionRunnerError(f"{condition} observation access metadata changed")
    artifact = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "condition": condition,
        "shape": list(activations.shape),
        "dtype": str(activations.dtype),
        "activations_sha256": tensor_sha256(activations),
        "attention_mask_sha256": tensor_sha256(serialized_mask),
        "position_ids_sha256": tensor_sha256(position_ids),
    }
    return activations, mask, position_ids, artifact


def _observation_path(
    index_path: Path,
    index: Mapping[str, Any],
    condition: str,
    observation_root: Path | None,
) -> Path:
    # Current setup emits the index beside observations/<condition>.safetensors;
    # accept an explicit descriptor too, while keeping the path binding strict.
    for key in ("observations", "observation_files", "artifacts"):
        values = index.get(key)
        if isinstance(values, Mapping):
            row = values.get(condition)
            if isinstance(row, Mapping) and isinstance(row.get("path"), str):
                value = Path(str(row["path"])).expanduser()
                if not value.is_absolute():
                    value = index_path.parent / value
                return value.resolve()
            if isinstance(row, str):
                value = Path(row).expanduser()
                if not value.is_absolute():
                    value = index_path.parent / value
                return value.resolve()
    roots = []
    if observation_root is not None:
        roots.append(observation_root.expanduser().resolve())
    roots.append(index_path.parent.resolve())
    candidates: list[Path] = []
    for root in roots:
        candidates.extend((root / "observations" / f"{condition}.safetensors", root / f"{condition}.safetensors"))
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise PredictionRunnerError(f"cannot locate {condition} evaluator observation under {roots[0]}")


def _load_observations(
    *,
    index_path: Path,
    observation_root: Path | None,
    index: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    selection_descriptor: Mapping[str, Any],
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]], dict[str, Any]]:
    result: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = _observation_path(index_path, index, condition, observation_root)
        result[condition] = _validate_observation_artifact(
            path,
            condition=condition,
            records=records,
            selection_descriptor=selection_descriptor,
        )
    return result, {
        condition: value[3] for condition, value in result.items()
    }


def _state_metadata(path: Path) -> dict[str, str]:
    metadata = _metadata(path)
    if metadata.get("schema") != STATE_SCHEMA or metadata.get("task_id") != TASK_ID:
        raise PredictionRunnerError(f"student state identity changed: {path}")
    return metadata


def _state_row_from_paths(
    path: Path,
    *,
    method_id: str,
    seed: int,
    kind: str,
    entry: Mapping[str, Any],
    schedule_sha256: str | None,
) -> dict[str, Any]:
    path = _regular_file(path, label=f"{kind} student state")
    metadata = _state_metadata(path)
    if metadata.get("method_id") != method_id or metadata.get("seed") != str(seed):
        raise PredictionRunnerError(f"{kind} state method/seed metadata changed: {path}")
    if kind == "selected":
        selected_step: int | str = int(metadata.get("selected_step", entry.get("selected_step", -1)))
        if int(selected_step) < 0:
            raise PredictionRunnerError(f"selected state lacks selected_step: {path}")
    else:
        selected_step = metadata.get("selected_step", "final")
    try:
        architecture = json.loads(metadata["architecture_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise PredictionRunnerError(f"state architecture metadata is invalid: {path}") from exc
    expected_architecture = asdict(StudentArchitectureConfig())
    if architecture != expected_architecture:
        raise PredictionRunnerError(f"state architecture changed: {path}")
    descriptor = _descriptor(path, label=f"{kind} student state")
    result: dict[str, Any] = {
        **descriptor,
        "kind": kind,
        "evaluation_input": kind == "selected",
        "method_id": method_id,
        "seed": int(seed),
        "state_schema": metadata["schema"],
        "task_id": metadata["task_id"],
        "architecture_json": metadata["architecture_json"],
        "architecture": architecture,
        "selected_step": selected_step,
        "training_schema": metadata.get("training_schema"),
        "teacher_source": metadata.get("teacher_source"),
        "schedule_sha256": metadata.get("schedule_sha256", schedule_sha256),
    }
    selected_entry = entry.get("selected_state") if kind == "selected" else entry.get("final_state")
    if isinstance(selected_entry, Mapping):
        if selected_entry.get("bytes") != descriptor["bytes"] or selected_entry.get("sha256") != descriptor["sha256"]:
            raise PredictionRunnerError(f"training aggregate binding changed for {kind} state: {path}")
        for key in ("state_sha256", "state_bytes", "tensor_sha256"):
            if key in selected_entry:
                result[key] = selected_entry[key]
    return result


def build_state_manifest(
    *,
    training_root: Path,
    training_result_path: Path | None,
    finalization_receipt_path: Path | None,
    output_path: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    """Bind all 16 training checkpoints while exposing only eight selected states."""

    started = time.perf_counter()
    training_root = _regular_dir(training_root, label="training root")
    if training_result_path is None:
        training_result_path = training_root / "training_result.json"
    if finalization_receipt_path is None:
        finalization_receipt_path = training_root / "training_finalization_receipt.json"
    training_result_path = _regular_file(training_result_path, label="training aggregate")
    finalization_receipt_path = _regular_file(finalization_receipt_path, label="training finalization receipt")
    aggregate = _load_json(training_result_path, label="training aggregate")
    finalizer = _load_json(finalization_receipt_path, label="training finalization receipt")
    if aggregate.get("schema") != "token-reconstruction.trr-p04-training.v1" or aggregate.get("task_id") != TASK_ID:
        raise PredictionRunnerError("training aggregate identity changed")
    if finalizer.get("status") != "PASS" or finalizer.get("task_id") != TASK_ID:
        raise PredictionRunnerError("training finalization receipt is not PASS")
    results = aggregate.get("results")
    if not isinstance(results, list) or len(results) != len(SEEDS):
        raise PredictionRunnerError("training aggregate must contain both paired seeds")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping) or result.get("seed") not in SEEDS:
            raise PredictionRunnerError("training aggregate has an unexpected seed")
        seed = int(result["seed"])
        if seed in by_seed:
            raise PredictionRunnerError("training aggregate duplicates a seed")
        by_seed[seed] = result
    selected_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        result = by_seed[seed]
        arms = result.get("arms")
        if not isinstance(arms, Mapping):
            raise PredictionRunnerError(f"training aggregate lacks arms for seed {seed}")
        schedule_sha = result.get("schedule_sha256")
        for method_id in METHODS:
            entry = arms.get(method_id)
            if not isinstance(entry, Mapping):
                raise PredictionRunnerError(f"training aggregate lacks {method_id}/{seed}")
            selected = entry.get("selected_state")
            final = entry.get("final_state")
            if not isinstance(selected, Mapping) or not isinstance(final, Mapping):
                raise PredictionRunnerError(f"training aggregate lacks selected/final state for {method_id}/{seed}")
            selected_path = Path(str(selected.get("path", ""))).expanduser()
            final_path = Path(str(final.get("path", ""))).expanduser()
            selected_rows.append(
                _state_row_from_paths(
                    selected_path,
                    method_id=method_id,
                    seed=seed,
                    kind="selected",
                    entry=entry,
                    schedule_sha256=str(schedule_sha) if schedule_sha else None,
                )
            )
            final_rows.append(
                _state_row_from_paths(
                    final_path,
                    method_id=method_id,
                    seed=seed,
                    kind="final",
                    entry=entry,
                    schedule_sha256=str(schedule_sha) if schedule_sha else None,
                )
            )
    required = {(method, seed) for method in METHODS for seed in SEEDS}
    if {(row["method_id"], row["seed"]) for row in selected_rows} != required:
        raise PredictionRunnerError("selected-state manifest is incomplete")
    if {(row["method_id"], row["seed"]) for row in final_rows} != required:
        raise PredictionRunnerError("final-state manifest is incomplete")
    source_receipt_value = aggregate.get("source_receipt")
    source_receipt_path = Path(str(source_receipt_value)).expanduser() if isinstance(source_receipt_value, str) else training_root / "source_receipt.json"
    source_receipt = _descriptor(source_receipt_path, label="training source receipt")
    late_failure_value = aggregate.get("late_finalization_failure")
    late_failure_path = None
    if isinstance(late_failure_value, Mapping) and isinstance(late_failure_value.get("path"), str):
        late_failure_path = Path(str(late_failure_value["path"])).expanduser()
    elif (training_root / "late_finalization_failure.json").is_file():
        late_failure_path = training_root / "late_finalization_failure.json"
    fit_wall_seconds = 0.0
    seed_wall_seconds = 0.0
    for result in results:
        seed_wall_seconds += float(result.get("wall_seconds", 0.0))
        arms = result.get("arms")
        if isinstance(arms, Mapping):
            fit_wall_seconds += sum(float(entry.get("wall_seconds", 0.0)) for entry in arms.values() if isinstance(entry, Mapping))
    provenance: dict[str, Any] = {
        "training_result": _descriptor(training_result_path, label="training aggregate"),
        "training_finalization_receipt": _descriptor(finalization_receipt_path, label="training finalization receipt"),
        "training_source_receipt": source_receipt,
        "run_source_commit": aggregate.get("run_source_commit"),
        "finalizer_source_commit": finalizer.get("finalizer_source_commit"),
        "finalized_after_late_cli_failure": bool(aggregate.get("finalized_after_late_cli_failure", False)),
        "fit_wall_seconds_sum": fit_wall_seconds,
        "per_seed_wall_seconds_sum": seed_wall_seconds,
        "aggregate_wall_seconds": float(aggregate.get("wall_seconds", 0.0)),
        "aggregate_wall_seconds_label": "finalizer serialization time; not fit time",
        "whole_training_run_wall_seconds": None,
        "whole_training_run_wall_seconds_note": "outer process start/end were not captured by the training CLI; no value is fabricated",
    }
    if late_failure_path is not None:
        provenance["late_finalization_failure"] = _descriptor(late_failure_path, label="late finalization failure evidence")
    payload = {
        "schema": STATE_MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS_SELECTED_ONLY_BEFORE_PREDICTION",
        "created_utc": _utc_now(),
        "truth_accessed": False,
        "evaluation_state_count": len(selected_rows),
        "all_frozen_state_count": len(selected_rows) + len(final_rows),
        "evaluation_input_rule": "only rows in states are permitted as prediction inputs; excluded_final_states are bound for provenance and never evaluated",
        "states": sorted(selected_rows, key=lambda row: (int(row["seed"]), METHODS.index(str(row["method_id"])) )),
        "excluded_final_states": sorted(final_rows, key=lambda row: (int(row["seed"]), METHODS.index(str(row["method_id"])) )),
        "all_state_bindings": sorted([*selected_rows, *final_rows], key=lambda row: (str(row["kind"]), int(row["seed"]), METHODS.index(str(row["method_id"])) )),
        "architecture": asdict(StudentArchitectureConfig()),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "training_provenance": provenance,
        "execution": {
            "argv": list(argv),
            "safe_environment": _safe_environment(),
            "git_commit": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "wall_seconds": time.perf_counter() - started,
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    _write_json_create_only(output_path, payload)
    return payload


def _load_state_manifest(path: Path) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    path = _regular_file(path, label="selected-state manifest")
    payload = _load_json(path, label="selected-state manifest")
    if payload.get("schema") != STATE_MANIFEST_SCHEMA or payload.get("task_id") != TASK_ID:
        raise PredictionRunnerError("selected-state manifest identity changed")
    if payload.get("status") != "PASS_SELECTED_ONLY_BEFORE_PREDICTION" or payload.get("truth_accessed") is not False:
        raise PredictionRunnerError("selected-state manifest is not a truth-free PASS")
    rows = payload.get("states")
    final_rows = payload.get("excluded_final_states")
    all_rows = payload.get("all_state_bindings")
    if not isinstance(rows, list) or len(rows) != 8 or not isinstance(final_rows, list) or len(final_rows) != 8 or not isinstance(all_rows, list) or len(all_rows) != 16:
        raise PredictionRunnerError("selected-state manifest must bind eight selected and eight excluded final states")
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("kind") != "selected" or row.get("evaluation_input") is not True:
            raise PredictionRunnerError("selected-state manifest has a malformed evaluation state")
        method = row.get("method_id")
        seed = row.get("seed")
        identity = (str(method), int(seed)) if method in METHODS and seed in SEEDS else None
        if identity is None or identity in selected:
            raise PredictionRunnerError("selected-state manifest has an unexpected/duplicate identity")
        actual = _descriptor(Path(str(row.get("path", ""))), label="selected state")
        if actual["bytes"] != row.get("bytes") or actual["sha256"] != row.get("sha256"):
            raise PredictionRunnerError(f"selected state changed: {actual['path']}")
        metadata = _state_metadata(Path(actual["path"]))
        if metadata.get("method_id") != str(method) or metadata.get("seed") != str(seed):
            raise PredictionRunnerError(f"selected state metadata identity changed: {actual['path']}")
        selected[identity] = dict(row)
    if set(selected) != expected:
        raise PredictionRunnerError("selected-state manifest does not cover all eight identities")
    all_identities: set[tuple[str, int, str]] = set()
    manifest_base = path.parent
    for index, raw in enumerate(all_rows):
        if not isinstance(raw, Mapping):
            raise PredictionRunnerError(f"all-state descriptor {index} is malformed")
        method = raw.get("method_id")
        seed = raw.get("seed")
        kind = raw.get("kind")
        if method not in METHODS or seed not in SEEDS or kind not in ("selected", "final"):
            raise PredictionRunnerError(f"all-state descriptor {index} has an unexpected identity")
        identity = (str(method), int(seed), str(kind))
        if identity in all_identities:
            raise PredictionRunnerError("all-state provenance binding is duplicated")
        all_identities.add(identity)
        state_path = Path(str(raw.get("path", ""))).expanduser()
        if not state_path.is_absolute():
            state_path = manifest_base / state_path
        actual = _descriptor(state_path, label=f"{kind} state")
        if actual["bytes"] != raw.get("bytes") or actual["sha256"] != raw.get("sha256"):
            raise PredictionRunnerError(f"{kind} state changed: {actual['path']}")
        metadata = _state_metadata(state_path)
        if metadata.get("method_id") != str(method) or metadata.get("seed") != str(seed):
            raise PredictionRunnerError(f"{kind} state metadata identity changed: {actual['path']}")
        expected_input = kind == "selected"
        if bool(raw.get("evaluation_input")) != expected_input:
            raise PredictionRunnerError(f"{kind} state evaluation-input flag changed: {actual['path']}")
    expected_all = {(method, seed, kind) for method in METHODS for seed in SEEDS for kind in ("selected", "final")}
    if all_identities != expected_all:
        raise PredictionRunnerError("all-state provenance binding is incomplete")
    return payload, selected


def _load_embedding_table(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = _regular_file(path, label="public normalized embedding table")
    started = time.perf_counter()
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"embeddings"}:
                raise PredictionRunnerError("public embedding table keys changed")
            table = handle.get_tensor("embeddings").contiguous().float()
            metadata = dict(handle.metadata() or {})
    except PredictionRunnerError:
        raise
    except Exception as exc:
        raise PredictionRunnerError(f"cannot load public normalized embedding table: {path}") from exc
    validate_embedding_table(table, hidden_size=EXPECTED_HIDDEN, vocab_size=EXPECTED_VOCAB, require_unit_norm=True)
    return table, {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "shape": list(table.shape),
        "dtype": str(table.dtype),
        "metadata": metadata,
        "load_seconds": time.perf_counter() - started,
        "tensor_sha256": tensor_sha256(table),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prediction_digest(predictions: torch.Tensor, ties: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (predictions, ties):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _repeat_prediction(
    model: Any,
    activations: torch.Tensor,
    table: torch.Tensor,
    *,
    device: torch.device,
    valid_mask: torch.Tensor,
    warmup_repeats: int = EXPECTED_WARMUPS,
    measurement_repeats: int = EXPECTED_MEASUREMENTS,
    record_batch_size: int = EXPECTED_RECORD_BATCH,
    projection_chunk: int = EXPECTED_PROJECTION_CHUNK,
) -> dict[str, Any]:
    if warmup_repeats != EXPECTED_WARMUPS or measurement_repeats != EXPECTED_MEASUREMENTS:
        raise PredictionRunnerError("P04 timing requires exactly one warmup and three measured repeats")
    warmups: list[tuple[torch.Tensor, torch.Tensor]] = []
    warmup_seconds: list[float] = []
    for _ in range(warmup_repeats):
        _synchronize(device)
        started = time.perf_counter()
        output = prediction_tensor(
            model,
            activations,
            table,
            device=device,
            valid_mask=valid_mask,
            record_batch_size=record_batch_size,
            projection_chunk=projection_chunk,
        )
        _synchronize(device)
        warmup_seconds.append(time.perf_counter() - started)
        warmups.append((output[0].detach().cpu().contiguous(), output[1].detach().cpu().contiguous()))
    if len(warmups) != 1:
        raise PredictionRunnerError("warmup result count changed")
    baseline_predictions, baseline_ties = warmups[0]
    measured: list[tuple[torch.Tensor, torch.Tensor]] = []
    measured_seconds: list[float] = []
    for _ in range(measurement_repeats):
        _synchronize(device)
        started = time.perf_counter()
        output = prediction_tensor(
            model,
            activations,
            table,
            device=device,
            valid_mask=valid_mask,
            record_batch_size=record_batch_size,
            projection_chunk=projection_chunk,
        )
        _synchronize(device)
        measured_seconds.append(time.perf_counter() - started)
        predicted = output[0].detach().cpu().contiguous()
        ties = output[1].detach().cpu().contiguous()
        if not torch.equal(predicted, baseline_predictions) or not torch.equal(ties, baseline_ties):
            raise PredictionRunnerError("prediction changed across synchronized repeats")
        measured.append((predicted, ties))
    return {
        "predictions": baseline_predictions,
        "ties": baseline_ties,
        "warmup_seconds": warmup_seconds,
        "measured_seconds": measured_seconds,
        "repeat_count": len(measured),
        "repeated_prediction_exact": True,
        "warmup_digest": _prediction_digest(baseline_predictions, baseline_ties),
        "measured_digests": [_prediction_digest(predicted, ties) for predicted, ties in measured],
        "batch_size": int(record_batch_size),
        "projection_chunk": int(projection_chunk),
        "device": str(device),
    }


def _prediction_lines(
    predictions: torch.Tensor,
    ties: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
    seed: int,
    condition: str,
) -> tuple[str, str, dict[str, Any]]:
    if predictions.ndim != 2 or tuple(ties.shape) != tuple(predictions.shape) or predictions.shape[0] != len(records):
        raise PredictionRunnerError("prediction result geometry changed")
    lines: list[str] = []
    tie_rows: list[dict[str, Any]] = []
    total_positions = 0
    total_ties = 0
    for row, metadata in enumerate(records):
        length = int(metadata["length_stratum"])
        active_count = length + 1
        values = [int(value) for value in predictions[row, 1:active_count].tolist()]
        tie_values = [int(value) for value in ties[row, 1:active_count].tolist()]
        if len(values) != length or any(value < 0 or value >= EXPECTED_VOCAB for value in values):
            raise PredictionRunnerError(f"prediction contains an invalid token for {metadata['record_id']}")
        if len(tie_values) != length or any(value < 1 for value in tie_values):
            raise PredictionRunnerError(f"tie diagnostics are invalid for {metadata['record_id']}")
        total_positions += length
        total_ties += sum(value > 1 for value in tie_values)
        lines.append(
            json.dumps(
                {
                    "schema": PREDICTION_SCHEMA,
                    "method_id": method_id,
                    "seed": int(seed),
                    "condition": condition,
                    "record_id": str(metadata["record_id"]),
                    "predicted_token_ids": values,
                    "anchor": False,
                },
                sort_keys=True,
            )
        )
        tie_rows.append({"record_id": str(metadata["record_id"]), "tie_counts": tie_values})
    tie_payload = {
        "schema": TIE_SCHEMA,
        "task_id": TASK_ID,
        "method_id": method_id,
        "seed": int(seed),
        "condition": condition,
        "rows": tie_rows,
        "summary": {
            "scored_positions": total_positions,
            "positions_with_tie": total_ties,
            "all_tie_counts_one": total_ties == 0,
        },
    }
    return "\n".join(lines) + "\n", json.dumps(tie_payload, indent=2, sort_keys=True) + "\n", tie_payload["summary"]


def _write_cell_outputs(
    output_root: Path,
    *,
    method_id: str,
    seed: int,
    condition: str,
    predictions: torch.Tensor,
    ties: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell_dir = output_root / "predictions" / condition / f"{method_id}-seed{seed}"
    prediction_path = cell_dir / "predictions.jsonl"
    tie_path = cell_dir / "tie_diagnostics.json"
    lines, tie_json, tie_summary = _prediction_lines(
        predictions,
        ties,
        records,
        method_id=method_id,
        seed=seed,
        condition=condition,
    )
    _write_text_create_only(prediction_path, lines)
    _write_text_create_only(tie_path, tie_json)
    return {
        "path": str(prediction_path.resolve()),
        "bytes": int(prediction_path.stat().st_size),
        "sha256": _sha256_file(prediction_path),
        "rows": len(records),
        "post_bos_positions": sum(int(row["length_stratum"]) for row in records),
    }, {
        "path": str(tie_path.resolve()),
        "bytes": int(tie_path.stat().st_size),
        "sha256": _sha256_file(tie_path),
        "summary": tie_summary,
    }


def _average(values: Sequence[float]) -> float:
    if not values:
        raise PredictionRunnerError("cannot average an empty timing vector")
    return float(sum(values) / len(values))


def _anchor_binding(
    full: Mapping[str, Any],
    anchor: Mapping[str, Any],
    *,
    anchor_indices: Sequence[int],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    full_predictions = full["predictions"]
    full_ties = full["ties"]
    anchor_predictions = anchor["predictions"]
    anchor_ties = anchor["ties"]
    index_tensor = torch.tensor(list(anchor_indices), dtype=torch.long)
    expected_predictions = full_predictions.index_select(0, index_tensor)
    expected_ties = full_ties.index_select(0, index_tensor)
    if not torch.equal(expected_predictions, anchor_predictions) or not torch.equal(expected_ties, anchor_ties):
        raise PredictionRunnerError("anchor packed prediction differs from the frozen full-panel prediction")
    anchor_rows = [records[index] for index in anchor_indices]
    return {
        "record_count": len(anchor_rows),
        "record_ids": [str(row["record_id"]) for row in anchor_rows],
        "post_bos_positions": sum(int(row["length_stratum"]) for row in anchor_rows),
        "record_order_sha256": record_order_sha256(anchor_rows),
        "full_panel_slice_digest": _prediction_digest(expected_predictions, expected_ties),
        "anchor_digest": _prediction_digest(anchor_predictions, anchor_ties),
        "full_panel_slice_exact": True,
        "anchor_output_exact": True,
    }


def _timing_payload(
    *,
    method_id: str,
    seed: int,
    condition: str,
    state_row: Mapping[str, Any],
    observation: Mapping[str, Any],
    selection_descriptor: Mapping[str, Any],
    full: Mapping[str, Any],
    anchor: Mapping[str, Any],
    anchor_binding: Mapping[str, Any],
    model_load_seconds: float,
    table_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    full_count = EXPECTED_ROWS
    full_positions = int(full["predictions"].shape[0])
    # The sequence tensor includes padding; count active scored positions from
    # the exact mask binding in the caller's observation descriptor below.
    full_scored_positions = int(observation["post_bos_positions"])
    anchor_count = int(anchor["predictions"].shape[0])
    anchor_scored_positions = int(anchor_binding["post_bos_positions"])
    full_measured = [float(value) for value in full["measured_seconds"]]
    anchor_measured = [float(value) for value in anchor["measured_seconds"]]
    return {
        "schema": TIMING_SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS",
        "truth_accessed": False,
        "method_id": method_id,
        "seed": int(seed),
        "condition": condition,
        "state": {
            "method_id": state_row["method_id"],
            "seed": state_row["seed"],
            "path": state_row["path"],
            "bytes": state_row["bytes"],
            "sha256": state_row["sha256"],
            "selected_step": state_row["selected_step"],
        },
        "observation": dict(observation),
        "selection": {
            "sha256": selection_descriptor["sha256"],
            "record_order_sha256": selection_descriptor["record_order_sha256"],
        },
        "embedding_table": {
            "path": table_descriptor["path"],
            "bytes": table_descriptor["bytes"],
            "sha256": table_descriptor["sha256"],
            "shape": table_descriptor["shape"],
            "tensor_sha256": table_descriptor["tensor_sha256"],
        },
        "geometry": {
            "full_records": full_count,
            "full_tensor_shape": list(full["predictions"].shape),
            "full_scored_positions": full_scored_positions,
            "anchor_records": anchor_count,
            "anchor_tensor_shape": list(anchor["predictions"].shape),
            "anchor_scored_positions": anchor_scored_positions,
            "record_batch_size": full["batch_size"],
            "projection_chunk": full["projection_chunk"],
            "full_vocabulary": True,
        },
        "startup": {
            "state_load_seconds": float(model_load_seconds),
            "table_load_seconds": float(table_descriptor["load_seconds"]),
            "state_and_table_load_in_startup": True,
        },
        "full_panel": {
            "warmup_repeats": len(full["warmup_seconds"]),
            "warmup_seconds": list(full["warmup_seconds"]),
            "measurement_repeats": len(full_measured),
            "measurement_seconds": full_measured,
            "mean_seconds": _average(full_measured),
            "milliseconds_per_record": 1000.0 * _average(full_measured) / full_count,
            "milliseconds_per_scored_position": 1000.0 * _average(full_measured) / full_scored_positions,
            "repeated_prediction_exact": bool(full["repeated_prediction_exact"]),
            "prediction_digest": full["warmup_digest"],
            "measured_digests": full["measured_digests"],
        },
        "anchor_subset": {
            "warmup_repeats": len(anchor["warmup_seconds"]),
            "warmup_seconds": list(anchor["warmup_seconds"]),
            "measurement_repeats": len(anchor_measured),
            "measurement_seconds": anchor_measured,
            "mean_seconds": _average(anchor_measured),
            "milliseconds_per_record": 1000.0 * _average(anchor_measured) / anchor_count,
            "milliseconds_per_scored_position": 1000.0 * _average(anchor_measured) / anchor_scored_positions,
            "repeated_prediction_exact": bool(anchor["repeated_prediction_exact"]),
            "prediction_digest": anchor["warmup_digest"],
            "measured_digests": anchor["measured_digests"],
            "binding": dict(anchor_binding),
        },
        "access": {
            "uses_source_tokens": False,
            "uses_teacher_or_candidates": False,
            "uses_public_prefix": False,
            "uses_target_update_weights": False,
            "uses_evaluation_truth": False,
        },
        "resource": {
            "peak_rss_bytes": _max_rss_bytes(),
            "device": str(full["device"]),
        },
    }


def _cell_receipt(
    *,
    method_id: str,
    seed: int,
    condition: str,
    state_row: Mapping[str, Any],
    observation: Mapping[str, Any],
    selection_descriptor: Mapping[str, Any],
    output_descriptor: Mapping[str, Any],
    tie_descriptor: Mapping[str, Any],
    timing_descriptor: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CELL_SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS",
        "truth_accessed": False,
        "method_id": method_id,
        "seed": int(seed),
        "condition": condition,
        "state": dict(state_row),
        "observation": dict(observation),
        "selection": dict(selection_descriptor),
        "prediction": dict(output_descriptor),
        "tie_diagnostics": dict(tie_descriptor),
        "timing": dict(timing_descriptor),
        "timing_payload": dict(timing),
        "full_vocabulary": True,
        "uses_source_tokens": False,
        "uses_teacher_or_candidates": False,
        "uses_public_prefix": False,
        "uses_target_update_weights": False,
        "uses_evaluation_truth": False,
    }


def _write_failure(output_root: Path, exc: BaseException, *, argv: Sequence[str], started_utc: str) -> None:
    value = {
        "schema": f"{RUNNER_SCHEMA}-failure.v1",
        "task_id": TASK_ID,
        "status": "FAIL_CLOSED",
        "truth_accessed": False,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "argv": list(argv),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "git_commit": _git_head(),
        "max_rss_bytes": _max_rss_bytes(),
    }
    try:
        _write_json_create_only(output_root / "failure.json", value)
    except Exception:
        pass


def run_predictions(
    *,
    observation_index_path: Path,
    observation_root: Path | None,
    selection_path: Path,
    state_manifest_path: Path,
    embedding_table_path: Path,
    output_root: Path,
    device_name: str,
    warmup_repeats: int = EXPECTED_WARMUPS,
    measurement_repeats: int = EXPECTED_MEASUREMENTS,
    record_batch_size: int = EXPECTED_RECORD_BATCH,
    projection_chunk: int = EXPECTED_PROJECTION_CHUNK,
    threads: int = 4,
    interop_threads: int = 1,
    implementation_commit: str | None = None,
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    started_utc = _utc_now()
    started = time.perf_counter()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
            raise PredictionRunnerError(f"prediction output must be a new empty directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    if warmup_repeats != EXPECTED_WARMUPS or measurement_repeats != EXPECTED_MEASUREMENTS:
        raise PredictionRunnerError("runner requires exactly warmup=1 and measurements=3")
    if record_batch_size != EXPECTED_RECORD_BATCH or projection_chunk != EXPECTED_PROJECTION_CHUNK:
        raise PredictionRunnerError("runner geometry requires batch=8 and projection_chunk=512")
    if threads <= 0 or interop_threads <= 0:
        raise PredictionRunnerError("thread counts must be positive")
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(int(interop_threads))
    except RuntimeError:
        # A caller may have initialized the inter-op pool while importing the
        # runner.  The receipt still records the requested value.
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    if device_name == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise PredictionRunnerError("CUDA requested for student predictions but unavailable")
    device = torch.device(device_name)
    try:
        records, selection_descriptor = _load_selection(selection_path)
        index, index_info = _validate_observation_index(
            observation_index_path,
            selection_records=records,
            selection_descriptor=selection_descriptor,
        )
        observations, observation_descriptors = _load_observations(
            index_path=observation_index_path,
            observation_root=observation_root,
            index=index,
            records=records,
            selection_descriptor=selection_descriptor,
        )
        state_manifest, states = _load_state_manifest(state_manifest_path)
        table_cpu, table_descriptor = _load_embedding_table(embedding_table_path)
        table_transfer_started = time.perf_counter()
        table_device = table_cpu.to(device=device, dtype=torch.float32)
        _synchronize(device)
        table_descriptor = {**table_descriptor, "device_transfer_seconds": time.perf_counter() - table_transfer_started}
        del table_cpu
        anchor_indices = [index for index, row in enumerate(records) if bool(row["anchor"])]
        if len(anchor_indices) != 12:
            raise PredictionRunnerError("anchor subset changed")
        cells: list[dict[str, Any]] = []
        prediction_files: list[dict[str, Any]] = []
        tie_files: list[dict[str, Any]] = []
        timing_files: list[dict[str, Any]] = []
        cell_receipts: list[dict[str, Any]] = []
        # A model is loaded once per selected state and reused for both paired
        # conditions.  It is never updated during prediction.
        for seed in SEEDS:
            for method_id in METHODS:
                state_row = states[(method_id, seed)]
                state_load_started = time.perf_counter()
                model = load_student_state(
                    Path(str(state_row["path"])),
                    method_id=method_id,
                    device=device,
                    config=StudentArchitectureConfig(),
                )
                _synchronize(device)
                state_load_seconds = time.perf_counter() - state_load_started
                if str(getattr(model, "method_id", method_id)) != method_id:
                    raise PredictionRunnerError(f"loaded model method binding changed for {method_id}/{seed}")
                for condition in CONDITIONS:
                    activations, valid_mask, _position_ids, observation_artifact = observations[condition]
                    observation = {
                        **observation_artifact,
                        "record_order_sha256": selection_descriptor["record_order_sha256"],
                        "post_bos_positions": int(valid_mask[:, 1:].sum().item()),
                    }
                    full = _repeat_prediction(
                        model,
                        activations,
                        table_device,
                        device=device,
                        valid_mask=valid_mask,
                        warmup_repeats=warmup_repeats,
                        measurement_repeats=measurement_repeats,
                        record_batch_size=record_batch_size,
                        projection_chunk=projection_chunk,
                    )
                    anchor_activations = activations.index_select(0, torch.tensor(anchor_indices, dtype=torch.long)).contiguous()
                    anchor_mask = valid_mask.index_select(0, torch.tensor(anchor_indices, dtype=torch.long)).contiguous()
                    anchor = _repeat_prediction(
                        model,
                        anchor_activations,
                        table_device,
                        device=device,
                        valid_mask=anchor_mask,
                        warmup_repeats=warmup_repeats,
                        measurement_repeats=measurement_repeats,
                        record_batch_size=record_batch_size,
                        projection_chunk=projection_chunk,
                    )
                    anchor_binding = _anchor_binding(full, anchor, anchor_indices=anchor_indices, records=records)
                    output_descriptor, tie_descriptor = _write_cell_outputs(
                        output_root,
                        method_id=method_id,
                        seed=seed,
                        condition=condition,
                        predictions=full["predictions"],
                        ties=full["ties"],
                        records=records,
                    )
                    timing = _timing_payload(
                        method_id=method_id,
                        seed=seed,
                        condition=condition,
                        state_row=state_row,
                        observation=observation,
                        selection_descriptor=selection_descriptor,
                        full=full,
                        anchor=anchor,
                        anchor_binding=anchor_binding,
                        model_load_seconds=state_load_seconds,
                        table_descriptor=table_descriptor,
                    )
                    timing_path = output_root / "timing" / condition / f"{method_id}-seed{seed}.json"
                    _write_json_create_only(timing_path, timing)
                    timing_descriptor = _descriptor(timing_path, label="prediction timing receipt")
                    cell = _cell_receipt(
                        method_id=method_id,
                        seed=seed,
                        condition=condition,
                        state_row=state_row,
                        observation=observation,
                        selection_descriptor=selection_descriptor,
                        output_descriptor=output_descriptor,
                        tie_descriptor=tie_descriptor,
                        timing_descriptor=timing_descriptor,
                        timing=timing,
                    )
                    cell_path = output_root / "cells" / condition / f"{method_id}-seed{seed}.json"
                    _write_json_create_only(cell_path, cell)
                    cell_descriptor = _descriptor(cell_path, label="prediction cell receipt")
                    cell_receipts.append(cell_descriptor)
                    prediction_files.append(output_descriptor)
                    tie_files.append(tie_descriptor)
                    timing_files.append(timing_descriptor)
                    cells.append(
                        {
                            "method_id": method_id,
                            "seed": seed,
                            "condition": condition,
                            "prediction": output_descriptor,
                            "tie_diagnostics": tie_descriptor,
                            "timing": timing_descriptor,
                            "cell_receipt": cell_descriptor,
                            "repeated_prediction_exact": True,
                            "anchor_full_panel_slice_exact": True,
                        }
                    )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if len(cells) != 16:
            raise PredictionRunnerError("student prediction matrix is incomplete")
        freeze = {
            "schema": FREEZE_SCHEMA,
            "task_id": TASK_ID,
            "status": "STUDENT_PREDICTIONS_FROZEN_BEFORE_JOINT_FREEZE",
            "created_utc": _utc_now(),
            "truth_accessed": False,
            "evaluation_truth_opened": False,
            "prediction_files_rewritten": False,
            "all_predictions_repeat_exact": all(bool(row["repeated_prediction_exact"]) for row in cells),
            "all_anchor_slices_exact": all(bool(row["anchor_full_panel_slice_exact"]) for row in cells),
            "selection": selection_descriptor,
            "observation_index": index_info["descriptor"],
            "observations": observation_descriptors,
            "state_manifest": _descriptor(state_manifest_path, label="selected-state manifest"),
            "state_input_count": len(states),
            "state_input_rule": "eight selected states only; eight final states remain bound in the manifest but are excluded from evaluation",
            "embedding_table": table_descriptor,
            "required_student_groups": [
                {"method_id": method, "seed": seed, "condition": condition, "anchor": False}
                for condition in CONDITIONS
                for seed in SEEDS
                for method in METHODS
            ],
            "cells": cells,
            "prediction_files": prediction_files,
            "tie_diagnostics_files": tie_files,
            "timing_files": timing_files,
            "access": {
                "uses_source_tokens": False,
                "uses_teacher_or_candidates": False,
                "uses_public_prefix": False,
                "uses_target_update_weights": False,
                "uses_evaluation_truth": False,
            },
            "execution": {
                "argv": list(argv),
                "implementation_commit": implementation_commit or _git_head(),
                "git_commit": _git_head(),
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "safe_environment": _safe_environment(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(device),
                "threads": int(threads),
                "interop_threads": int(interop_threads),
                "wall_seconds": time.perf_counter() - started,
                "max_rss_bytes": _max_rss_bytes(),
            },
        }
        _write_json_create_only(output_root / "student_prediction_freeze.json", freeze)
        return freeze
    except BaseException as exc:
        _write_failure(output_root, exc, argv=argv, started_utc=started_utc)
        raise


def run_synthetic_smoke(output_root: Path) -> dict[str, Any]:
    """Exercise import, full-vocabulary ties, repeat equality, and JSON output."""

    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
            raise PredictionRunnerError(f"synthetic smoke output must be new and empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    from token_reconstruction.p04_student import AffineStudent

    config = StudentArchitectureConfig(hidden_size=3, vocab_size=5, gru_width=2)
    model = AffineStudent(config)
    table = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    activations = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    mask = torch.ones((1, 2), dtype=torch.bool)
    first = prediction_tensor(model, activations, table, device=torch.device("cpu"), valid_mask=mask, record_batch_size=1, projection_chunk=1)
    second = prediction_tensor(model, activations, table, device=torch.device("cpu"), valid_mask=mask, record_batch_size=1, projection_chunk=1)
    if not torch.equal(first[0], second[0]) or not torch.equal(first[1], second[1]):
        raise PredictionRunnerError("synthetic repeat changed")
    records = [{"record_id": "synthetic-0001", "style": "synthetic", "length_stratum": 1, "anchor": False}]
    prediction_text, tie_text, summary = _prediction_lines(
        first[0], first[1], records, method_id=METHOD_AFFINE, seed=1737, condition="public_base"
    )
    prediction_path = output_root / "predictions.jsonl"
    tie_path = output_root / "tie_diagnostics.json"
    _write_text_create_only(prediction_path, prediction_text)
    _write_text_create_only(tie_path, tie_text)
    payload = {
        "schema": f"{RUNNER_SCHEMA}-synthetic-smoke.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "truth_accessed": False,
        "cli_module": __name__,
        "prediction": {"path": str(prediction_path), "sha256": _sha256_file(prediction_path)},
        "tie_diagnostics": {"path": str(tie_path), "sha256": _sha256_file(tie_path)},
        "expected_post_bos_prediction": 0,
        "expected_post_bos_tie_count": 2,
        "actual_post_bos_prediction": int(first[0][0, 1].item()),
        "actual_post_bos_tie_count": int(first[1][0, 1].item()),
        "repeated_prediction_exact": True,
        "summary": summary,
    }
    if payload["actual_post_bos_prediction"] != payload["expected_post_bos_prediction"] or payload["actual_post_bos_tie_count"] != payload["expected_post_bos_tie_count"]:
        raise PredictionRunnerError("synthetic lowest-ID tie smoke failed")
    _write_json_create_only(output_root / "synthetic_smoke.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-state-manifest", action="store_true")
    mode.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--training-result", type=Path)
    parser.add_argument("--training-finalization-receipt", type=Path)
    parser.add_argument("--state-manifest-output", type=Path)
    parser.add_argument("--observation-index", type=Path)
    parser.add_argument("--observation-root", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--embedding-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-repeats", type=int, default=EXPECTED_WARMUPS)
    parser.add_argument("--measurement-repeats", type=int, default=EXPECTED_MEASUREMENTS)
    parser.add_argument("--record-batch-size", type=int, default=EXPECTED_RECORD_BATCH)
    parser.add_argument("--projection-chunk", type=int, default=EXPECTED_PROJECTION_CHUNK)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--implementation-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    effective_argv = list(sys.argv if argv is None else [sys.argv[0], *argv])
    try:
        if args.synthetic_smoke:
            value = run_synthetic_smoke(args.output_root)
        elif args.write_state_manifest:
            if args.training_root is None or args.state_manifest_output is None:
                raise PredictionRunnerError("--write-state-manifest requires --training-root and --state-manifest-output")
            value = build_state_manifest(
                training_root=args.training_root,
                training_result_path=args.training_result,
                finalization_receipt_path=args.training_finalization_receipt,
                output_path=args.state_manifest_output,
                argv=effective_argv,
            )
        else:
            required = {
                "--observation-index": args.observation_index,
                "--selection": args.selection,
                "--state-manifest": args.state_manifest,
            }
            missing = [key for key, value in required.items() if value is None]
            if missing:
                raise PredictionRunnerError(f"prediction mode requires {', '.join(missing)}")
            value = run_predictions(
                observation_index_path=args.observation_index,
                observation_root=args.observation_root,
                selection_path=args.selection,
                state_manifest_path=args.state_manifest,
                embedding_table_path=args.embedding_table,
                output_root=args.output_root,
                device_name=args.device,
                warmup_repeats=args.warmup_repeats,
                measurement_repeats=args.measurement_repeats,
                record_batch_size=args.record_batch_size,
                projection_chunk=args.projection_chunk,
                threads=args.threads,
                interop_threads=args.interop_threads,
                implementation_commit=args.implementation_commit,
                argv=effective_argv,
            )
        print(json.dumps({"status": value.get("status"), "schema": value.get("schema"), "output_root": str(args.output_root.expanduser().resolve())}, sort_keys=True))
        return 0
    except (PredictionRunnerError, P04StudentError, RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"P04 student prediction runner failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
