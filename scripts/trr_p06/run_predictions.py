#!/usr/bin/env python3
"""Run the frozen TRR-P06 activation-only prediction matrix.

The runner consumes the setup-owned public observation manifest and the six
selected P06 decoder states.  It never opens source IDs, target labels, truth,
or candidate/A2 resources.  Each state is evaluated on all four paired
observation cells with full-vocabulary readout.  Prediction files and their
timing/tie receipts are create-only; a completed run writes a source-free
student manifest that setup can combine with the separately produced A1+A2
anchor descriptor before the joint truth gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from token_reconstruction.trr0006_visibility_decoder import (  # noqa: E402
    FULL_RECORD_METHOD,
    PAST_ONLY_METHOD,
    POSITIONWISE_METHOD,
    VisibilityDecoderError,
    deterministic_top1,
    file_sha256,
    load_visibility_state,
)


TASK_ID = "TRR-P06"
OBSERVATION_SCHEMA = "token-reconstruction.trr-p06-public-observation-manifest.v1"
OBSERVATION_FILE_SCHEMA = "token-reconstruction.trr-p06-public-observation.v1"
STUDENT_SCHEMA = "token-reconstruction.trr-p06-student-prediction-manifest.v1"
RUN_SCHEMA = "token-reconstruction.trr-p06-student-prediction-run.v1"
TIMING_SCHEMA = "token-reconstruction.trr-p06-student-prediction-timing.v1"
FAILURE_SCHEMA = "token-reconstruction.trr-p06-student-prediction-failure.v1"

DOMAINS = ("pile", "finance")
TARGETS = ("public_base", "public_lora_2601")
CELL_ORDER = tuple(f"{domain}__{target}" for domain in DOMAINS for target in TARGETS)
METHOD_ORDER = (POSITIONWISE_METHOD, PAST_ONLY_METHOD, FULL_RECORD_METHOD)
SEEDS = (6106, 6107)
RECORDS_PER_DOMAIN = 256
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = -1
CAPTURE_BATCH_RECORDS = 8
PROJECTION_CHUNK = 512
WARMUP_PASSES = 1
MEASURED_PASSES = 3
DIRECT_AFFINE_SHA256 = "09c5b852373d8555b06508a79bb00c94041202702b61b121b35fa2b6f9f64e65"
EMBEDDING_BYTES = 1050673488
EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"


class PredictionError(RuntimeError):
    """Raised when the P06 prediction contract fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _resolve(path_value: str | Path, *, root: Path, description: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise PredictionError(f"{description} is unavailable: {path}")
    return path


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PredictionError(f"file is unavailable: {path}")
    record_path: str = str(path)
    if root is not None:
        try:
            record_path = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {"path": record_path, "bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise PredictionError(f"{description} must be an object")
    return dict(value)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PredictionError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PredictionError("cannot resolve executable source commit") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise PredictionError("executable source commit is not a full lowercase hash")
    return value


def _set_runtime_threads(intraop: int, interop: int) -> dict[str, Any]:
    if intraop <= 0 or interop <= 0:
        raise PredictionError("Torch thread counts must be positive")
    try:
        torch.set_num_threads(intraop)
        torch.set_num_interop_threads(interop)
    except RuntimeError as exc:
        raise PredictionError("Torch thread configuration failed") from exc
    return {"cpu_intraop_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads()}


def _available_host_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                return int(fields[1]) * 1024
    except (OSError, UnicodeError):
        pass
    raise PredictionError("host available-memory guard is unavailable")


def _guard(*, device: torch.device, started: float, max_seconds: float, min_free_gib: float, max_reserved_gib: float, max_rss_gib: float, min_host_gib: float, stage: str) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    if elapsed > max_seconds:
        raise PredictionError(f"wall-time guard failed at {stage}: {elapsed:.3f}s")
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    max_rss = int(max_rss_gib * 2**30)
    host = _available_host_bytes()
    min_host = int(min_host_gib * 2**30)
    if rss > max_rss:
        raise PredictionError(f"RSS guard failed at {stage}: {rss} > {max_rss}")
    if host < min_host:
        raise PredictionError(f"host available-memory guard failed at {stage}: {host} < {min_host}")
    result: dict[str, Any] = {
        "stage": stage,
        "elapsed_seconds": float(elapsed),
        "process_max_rss_bytes": rss,
        "host_available_bytes": host,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        reserved = int(torch.cuda.memory_reserved(device))
        allocated = int(torch.cuda.memory_allocated(device))
        if int(free) < int(min_free_gib * 2**30):
            raise PredictionError(f"GPU free-memory guard failed at {stage}: {free}")
        if reserved > int(max_reserved_gib * 2**30):
            raise PredictionError(f"GPU reserved-memory guard failed at {stage}: {reserved}")
        result["gpu"] = {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "reserved_bytes": reserved,
            "allocated_bytes": allocated,
        }
    return result


def _validate_fit_receipt(path: Path, *, root: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    receipt = _load_json(path, description="P06 main-fit receipt")
    if receipt.get("schema") != "token-reconstruction.trr-p06-main-fit.v1" or receipt.get("task_id") != TASK_ID or receipt.get("status") != "PASS":
        raise PredictionError("main-fit receipt is not a PASS P06 main receipt")
    geometry = receipt.get("geometry")
    expected_geometry = {"fit": [1200, 128, 2048], "validation": [48, 128, 2048], "record_batch_size": 8, "query_draws_per_step": 512, "steps": 3000}
    if not isinstance(geometry, Mapping) or any(geometry.get(key) != value for key, value in expected_geometry.items()):
        raise PredictionError("main-fit geometry is not the frozen P06 recipe")
    if receipt.get("selection_metric") != "validation_token_accuracy":
        raise PredictionError("main-fit selection metric is not token_accuracy")
    direct = receipt.get("direct_affine")
    if not isinstance(direct, Mapping) or direct.get("sha256") != DIRECT_AFFINE_SHA256:
        raise PredictionError("main-fit direct affine binding changed")
    methods = receipt.get("methods")
    if not isinstance(methods, list) or len(methods) != len(SEEDS) * len(METHOD_ORDER):
        raise PredictionError("main-fit receipt does not contain exactly six arms")
    states: dict[tuple[int, str], dict[str, Any]] = {}
    for row in methods:
        if not isinstance(row, Mapping):
            raise PredictionError("main-fit arm descriptor is malformed")
        method_id = row.get("method_id")
        seed = row.get("seed")
        key = (int(seed), str(method_id)) if isinstance(seed, int) and not isinstance(seed, bool) else None
        if key is None or key in states or key[0] not in SEEDS or key[1] not in METHOD_ORDER:
            raise PredictionError("main-fit arm seed/method matrix changed")
        if row.get("status") != "PASS" or int(row.get("steps", -1)) != 3000:
            raise PredictionError(f"main-fit arm is not a complete PASS: {key}")
        state = row.get("state")
        if not isinstance(state, Mapping):
            raise PredictionError(f"selected state is missing: {key}")
        raw_path = state.get("path")
        if not isinstance(raw_path, str):
            raise PredictionError(f"selected state path is missing: {key}")
        state_path = _resolve(raw_path, root=root, description=f"selected state {key}")
        actual = _file_record(state_path, root=root)
        if actual["bytes"] != state.get("bytes") or actual["sha256"] != state.get("sha256"):
            raise PredictionError(f"selected state file binding changed: {key}")
        state_sha = state.get("state_sha256")
        if not isinstance(state_sha, str) or len(state_sha) != 64:
            raise PredictionError(f"selected state tensor digest is missing: {key}")
        metadata = state.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("direct_affine_sha256") != DIRECT_AFFINE_SHA256 or metadata.get("method_id") != method_id or int(metadata.get("qkv_init_seed", -1)) != seed:
            raise PredictionError(f"selected state metadata changed: {key}")
        states[key] = {
            "path": state_path,
            "file": actual,
            "sha256": str(actual["sha256"]),
            "state_sha256": state_sha,
            "selected_step": int(state.get("selected_step", metadata.get("selected_step", -1))),
            "metadata": dict(metadata),
        }
    if set(states) != {(seed, method) for seed in SEEDS for method in METHOD_ORDER}:
        raise PredictionError("main-fit selected-state matrix is incomplete")
    return states, {"receipt": receipt, "receipt_record": _file_record(path, root=root)}


def _validate_observation_manifest(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(path, description="P06 public observation manifest")
    if manifest.get("schema") != OBSERVATION_SCHEMA or manifest.get("task_id") != TASK_ID or manifest.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise PredictionError("observation manifest is not the frozen no-truth P06 status")
    if manifest.get("records_per_domain") != RECORDS_PER_DOMAIN or manifest.get("sequence_tokens_including_bos") != SEQUENCE_TOKENS or manifest.get("scored_post_bos_tokens") != SCORED_POST_BOS or manifest.get("hidden_size") != HIDDEN_SIZE:
        raise PredictionError("observation manifest geometry changed")
    if manifest.get("source_text_written") is not False or manifest.get("token_ids_written") is not False or manifest.get("target_labels_loaded") is not False or manifest.get("truth_opened") is not False:
        raise PredictionError("observation manifest contains forbidden source or truth access")
    if manifest.get("cell_order") != list(CELL_ORDER):
        raise PredictionError("observation cell order changed")
    if manifest.get("source_pairing", {}).get("same_record_ids_across_targets") is not True:
        raise PredictionError("observation target pairing is not declared")
    cells_raw = manifest.get("cells")
    if not isinstance(cells_raw, list) or [row.get("cell_id") for row in cells_raw if isinstance(row, Mapping)] != list(CELL_ORDER):
        raise PredictionError("observation manifest cells are incomplete or reordered")
    cells: dict[str, Any] = {}
    record_ids_by_domain: dict[str, str] = {}
    for row in cells_raw:
        if not isinstance(row, Mapping):
            raise PredictionError("observation cell descriptor is malformed")
        cell_id = str(row.get("cell_id"))
        domain, target = cell_id.split("__", 1)
        if domain not in DOMAINS or target not in TARGETS:
            raise PredictionError(f"unknown P06 observation cell: {cell_id}")
        if row.get("style") != domain or row.get("condition") != target:
            raise PredictionError(f"observation cell identity changed: {cell_id}")
        record_ids_sha = row.get("record_ids_sha256")
        if not isinstance(record_ids_sha, str) or len(record_ids_sha) != 64:
            raise PredictionError(f"record identity digest is missing: {cell_id}")
        if domain in record_ids_by_domain and record_ids_by_domain[domain] != record_ids_sha:
            raise PredictionError(f"paired target record order changed: {domain}")
        record_ids_by_domain[domain] = record_ids_sha
        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise PredictionError(f"observation asset is missing: {cell_id}")
        if observation.get("schema") not in (OBSERVATION_FILE_SCHEMA, None):
            raise PredictionError(f"observation asset schema changed: {cell_id}")
        if observation.get("shape") != [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, HIDDEN_SIZE] or observation.get("stored_sequence_tokens") != SEQUENCE_TOKENS or observation.get("scored_post_bos_tokens") != SCORED_POST_BOS:
            raise PredictionError(f"observation asset geometry changed: {cell_id}")
        raw_path = observation.get("path")
        if not isinstance(raw_path, str):
            raise PredictionError(f"observation asset path is missing: {cell_id}")
        asset = _resolve(raw_path, root=root, description=f"observation {cell_id}")
        actual = _file_record(asset, root=root)
        if actual["bytes"] != observation.get("bytes") or actual["sha256"] != observation.get("sha256"):
            raise PredictionError(f"observation asset binding changed: {cell_id}")
        cells[cell_id] = {"domain": domain, "target": target, "record_ids_sha256": record_ids_sha, "observation": dict(observation), "path": asset, "file": actual}
    if set(cells) != set(CELL_ORDER):
        raise PredictionError("observation manifest cell set changed")
    return cells, {"manifest": manifest, "manifest_record": _file_record(path, root=root), "record_ids_sha256": record_ids_by_domain}


def _load_observation_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Load one source-free observation cell and validate all sidecars."""

    path = Path(str(cell["path"])).resolve()
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise PredictionError(f"observation keys changed: {path}")
            activations = handle.get_tensor("activations").contiguous()
            mask = handle.get_tensor("attention_mask").contiguous()
            positions = handle.get_tensor("position_ids").contiguous()
            metadata = dict(handle.metadata() or {})
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError(f"observation tensor load failed: {path}") from exc
    if tuple(activations.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, HIDDEN_SIZE) or activations.dtype != torch.bfloat16:
        raise PredictionError(f"observation activation geometry/dtype changed: {path}")
    if tuple(mask.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or mask.dtype not in (torch.bool, torch.uint8):
        raise PredictionError(f"observation mask geometry/dtype changed: {path}")
    if tuple(positions.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise PredictionError(f"observation positions geometry/dtype changed: {path}")
    if not torch.isfinite(activations.float()).all().item():
        raise PredictionError(f"observation activations are non-finite: {path}")
    if mask.dtype == torch.uint8 and ((mask != 0) & (mask != 1)).any().item():
        raise PredictionError(f"observation mask is not binary: {path}")
    mask = mask.to(dtype=torch.bool)
    if not mask[:, 0].all().item():
        raise PredictionError(f"observation mask is not binary or lacks BOS: {path}")
    if (mask[:, 1:] > mask[:, :-1]).any().item():
        raise PredictionError(f"observation mask is not right-padded: {path}")
    expected_positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.long).expand(RECORDS_PER_DOMAIN, -1)
    if not torch.equal(positions.to(dtype=torch.long), expected_positions):
        raise PredictionError(f"observation positions are not 0..127: {path}")
    return {
        "activations": activations,
        "mask": mask,
        "positions": positions.to(dtype=torch.long),
        "position_ids": positions.to(dtype=torch.long),
        "attention_mask_sha256": _tensor_digest(mask.to(dtype=torch.uint8)),
        "position_ids_sha256": _tensor_digest(positions.to(dtype=torch.int64)),
        "metadata": metadata,
    }


def _load_embedding(path: Path, *, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    actual = _file_record(path)
    if actual["bytes"] != EMBEDDING_BYTES or actual["sha256"] != EMBEDDING_SHA256:
        raise PredictionError("normalized public embedding table binding changed")
    started = time.perf_counter()
    try:
        table_cpu = load_file(str(path), device="cpu")
        if set(table_cpu) != {"embeddings"}:
            raise PredictionError("embedding table must contain only embeddings")
        table = table_cpu["embeddings"].contiguous()
        if table.dtype != torch.float32 or tuple(table.shape) != (VOCABULARY_SIZE, HIDDEN_SIZE) or not torch.isfinite(table).all().item():
            raise PredictionError("embedding table geometry, dtype, or finiteness changed")
        table_device = table.to(device=device, dtype=torch.float32).contiguous()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError("embedding table load failed") from exc
    finally:
        if "table_cpu" in locals():
            del table_cpu
        if "table" in locals():
            del table
        gc.collect()
    return table_device, {"file": actual, "shape": [VOCABULARY_SIZE, HIDDEN_SIZE], "dtype": "torch.float32", "load_seconds": time.perf_counter() - started}


@torch.inference_mode()
def predict_batch(model: torch.nn.Module, embedding: torch.Tensor, activations: torch.Tensor, valid_mask: torch.Tensor, *, device: torch.device, projection_chunk: int = PROJECTION_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict one record batch with chunked full-vocabulary readout."""

    if projection_chunk <= 0:
        raise PredictionError("projection chunk must be positive")
    if activations.ndim != 3 or tuple(activations.shape[1:]) != (SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise PredictionError("activation batch geometry changed")
    if valid_mask.shape != activations.shape[:2]:
        raise PredictionError("mask batch geometry changed")
    staged = activations.to(device=device, dtype=torch.float32, non_blocking=False)
    mask = valid_mask.to(device=device, dtype=torch.bool, non_blocking=False)
    projected = model.projected_hidden(staged, mask)
    ids = torch.full((activations.shape[0], SEQUENCE_TOKENS), PAD_TOKEN_ID, dtype=torch.long, device="cpu")
    ties = torch.zeros((activations.shape[0], SEQUENCE_TOKENS), dtype=torch.int64, device="cpu")
    if not mask[:, 0].all().item():
        raise PredictionError("observation batch has no valid BOS")
    # BOS is a fixed observation prefix token.  It is emitted once below and
    # must never enter the full-vocabulary readout; all model predictions are
    # strictly post-BOS valid positions.
    post_bos = mask.clone()
    post_bos[:, 0] = False
    indices = torch.nonzero(post_bos, as_tuple=False)
    for start in range(0, int(indices.shape[0]), projection_chunk):
        current = indices[start : start + projection_chunk]
        logits = model.logits_from_rows(projected, current[:, 0], current[:, 1], embedding)
        predicted, tie_count = deterministic_top1(logits)
        predicted_cpu = predicted.detach().cpu().to(dtype=torch.long)
        ties_cpu = tie_count.detach().cpu().to(dtype=torch.int64)
        ids[current[:, 0].cpu(), current[:, 1].cpu()] = predicted_cpu
        ties[current[:, 0].cpu(), current[:, 1].cpu()] = ties_cpu
    active = mask.cpu()
    ids[~active] = PAD_TOKEN_ID
    ties[~active] = 0
    ids[:, 0] = BOS_TOKEN_ID
    ties[:, 0] = 1
    if (ids[active] < 0).any().item() or (ids[active] >= VOCABULARY_SIZE).any().item() or not ids[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise PredictionError("prediction IDs failed the frozen vocabulary/BOS contract")
    return ids, ties


def _predict_cell(model: torch.nn.Module, embedding: torch.Tensor, cell: Mapping[str, Any], *, device: torch.device, started: float, max_seconds: float, min_free_gib: float, max_reserved_gib: float, max_rss_gib: float, min_host_gib: float, batch_records: int, projection_chunk: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    observation_load_started = time.perf_counter()
    loaded = _load_observation_cell(cell)
    observation_load_seconds = time.perf_counter() - observation_load_started
    activations = loaded["activations"]
    mask = loaded["mask"]
    if RECORDS_PER_DOMAIN % batch_records != 0:
        raise PredictionError("record count is not divisible by batch size")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def one_pass() -> tuple[torch.Tensor, torch.Tensor, float]:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pass_started = time.perf_counter()
        pass_ids = torch.empty((RECORDS_PER_DOMAIN, SEQUENCE_TOKENS), dtype=torch.long)
        pass_ties = torch.empty((RECORDS_PER_DOMAIN, SEQUENCE_TOKENS), dtype=torch.int64)
        for row_start in range(0, RECORDS_PER_DOMAIN, batch_records):
            row_stop = row_start + batch_records
            current_ids, current_ties = predict_batch(
                model,
                embedding,
                activations[row_start:row_stop],
                mask[row_start:row_stop],
                device=device,
                projection_chunk=projection_chunk,
            )
            pass_ids[row_start:row_stop] = current_ids
            pass_ties[row_start:row_stop] = current_ties
            _guard(device=device, started=started, max_seconds=max_seconds, min_free_gib=min_free_gib, max_reserved_gib=max_reserved_gib, max_rss_gib=max_rss_gib, min_host_gib=min_host_gib, stage=f"after_{cell['cell_id']}_rows_{row_stop}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return pass_ids, pass_ties, time.perf_counter() - pass_started

    warm_ids: torch.Tensor | None = None
    warm_ties: torch.Tensor | None = None
    warm_seconds = 0.0
    for _ in range(WARMUP_PASSES):
        warm_ids, warm_ties, elapsed = one_pass()
        warm_seconds += elapsed
    measured: list[float] = []
    selected_ids: torch.Tensor | None = None
    selected_ties: torch.Tensor | None = None
    for _ in range(MEASURED_PASSES):
        ids, ties, elapsed = one_pass()
        if selected_ids is None:
            selected_ids, selected_ties = ids, ties
        elif not torch.equal(selected_ids, ids) or not torch.equal(selected_ties, ties):
            raise PredictionError(f"repeated prediction IDs or tie counts differ for {cell['cell_id']}")
        measured.append(elapsed)
    assert warm_ids is not None and warm_ties is not None and selected_ids is not None and selected_ties is not None
    if not torch.equal(warm_ids, selected_ids) or not torch.equal(warm_ties, selected_ties):
        raise PredictionError(f"warmup and measured prediction IDs differ for {cell['cell_id']}")
    timing: dict[str, Any] = {
        "schema": TIMING_SCHEMA,
        "task_id": TASK_ID,
        "cell_id": cell["cell_id"],
        "records": RECORDS_PER_DOMAIN,
        "batch_records": batch_records,
        "projection_chunk": projection_chunk,
        "warmup_passes": WARMUP_PASSES,
        "measured_passes": MEASURED_PASSES,
        "warmup_seconds": warm_seconds,
        "measured_seconds": measured,
        "measured_mean_seconds": sum(measured) / len(measured),
        "measured_ms_per_record": 1000.0 * (sum(measured) / len(measured)) / RECORDS_PER_DOMAIN,
        "repeat_prediction_exact": True,
        "observation_load_seconds": observation_load_seconds,
        "observation_load_excluded_from_measured_interval": True,
        "measurement_includes_resource_guard_and_device_synchronization": True,
        "measured_interval_definition": "full-panel prediction passes including per-batch resource guards and CUDA synchronization; observation deserialization is timed separately",
        "peak_memory": {
            "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        },
        "attention_mask_sha256": loaded["attention_mask_sha256"],
        "position_ids_sha256": loaded["position_ids_sha256"],
    }
    return selected_ids, selected_ties, timing


def _prediction_path(output_root: Path, cell_id: str, seed: int, method_id: str) -> Path:
    domain, target = cell_id.split("__", 1)
    return output_root / "predictions" / domain / target / str(seed) / f"{method_id}.safetensors"


def _tie_path(output_root: Path, cell_id: str, seed: int, method_id: str) -> Path:
    domain, target = cell_id.split("__", 1)
    return output_root / "tie_counts" / domain / target / str(seed) / f"{method_id}.safetensors"


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(root / "experiments" / "TRR-P06")
    except ValueError as exc:
        raise PredictionError("prediction output must be task-owned under experiments/TRR-P06") from exc
    if output_root.exists() or output_root.is_symlink():
        raise PredictionError(f"prediction output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    started_clock = time.perf_counter()
    started_utc = _utc_now()
    failure_path = output_root / "failure.json"
    try:
        runtime = _set_runtime_threads(args.torch_threads, args.torch_interop_threads)
        if args.device != "cuda" or not torch.cuda.is_available():
            raise PredictionError("production P06 prediction requires CUDA")
        device = torch.device("cuda")
        fit_states, fit_evidence = _validate_fit_receipt(Path(args.fit_receipt).expanduser().resolve(), root=root)
        cells, observation_evidence = _validate_observation_manifest(Path(args.observation_manifest).expanduser().resolve(), root=root)
        guards: list[dict[str, Any]] = []
        guards.append(_guard(device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, stage="before_embedding_load"))
        embedding, embedding_evidence = _load_embedding(Path(args.embedding_path).expanduser().resolve(), device=device)
        guards.append(_guard(device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, stage="after_embedding_load"))
        state_bindings: dict[str, Any] = {}
        student_cells: dict[str, dict[str, dict[str, Any]]] = {cell_id: {str(seed): {} for seed in SEEDS} for cell_id in CELL_ORDER}
        timings: dict[str, Any] = {}
        tie_summary: dict[str, Any] = {}
        for seed in SEEDS:
            for method_id in METHOD_ORDER:
                state = fit_states[(seed, method_id)]
                key = f"{seed}::{method_id}"
                state_bindings[key] = dict(state["file"])
                try:
                    model = load_visibility_state(state["path"], method_id=method_id, hidden_size=HIDDEN_SIZE, vocabulary_size=VOCABULARY_SIZE, context_width=128, expected_sha256=state["sha256"]).to(device=device).eval()
                except (VisibilityDecoderError, OSError, RuntimeError, ValueError) as exc:
                    raise PredictionError(f"selected decoder state load failed: {key}") from exc
                model.requires_grad_(False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                guards.append(_guard(device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, stage=f"after_{key}_load"))
                try:
                    for cell_id in CELL_ORDER:
                        cell = dict(cells[cell_id])
                        cell["cell_id"] = cell_id
                        ids, ties, timing = _predict_cell(model, embedding, cell, device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, batch_records=args.batch_records, projection_chunk=args.projection_chunk)
                        prediction_path = _prediction_path(output_root, cell_id, seed, method_id)
                        ties_path = _tie_path(output_root, cell_id, seed, method_id)
                        metadata = {
                            "schema": "token-reconstruction.trr-p06-predictions.v1",
                            "task_id": TASK_ID,
                            "domain": cell["domain"],
                            "target": cell["target"],
                            "method_id": method_id,
                            "seed": str(seed),
                            "records": str(RECORDS_PER_DOMAIN),
                            "sequence_tokens": str(SEQUENCE_TOKENS),
                            "scored_post_bos_tokens": str(SCORED_POST_BOS),
                            "state_sha256": state["sha256"],
                            "observation_sha256": cell["file"]["sha256"],
                            "truth_opened": "false",
                            "candidate_arrays_persisted": "false",
                        }
                        prediction_path.parent.mkdir(parents=True, exist_ok=True)
                        ties_path.parent.mkdir(parents=True, exist_ok=True)
                        save_file({"predictions": ids.contiguous()}, str(prediction_path), metadata=metadata)
                        save_file({"tie_counts": ties.contiguous()}, str(ties_path), metadata={**metadata, "schema": "token-reconstruction.trr-p06-tie-counts.v1"})
                        prediction_file = _file_record(prediction_path, root=root)
                        ties_file = _file_record(ties_path, root=root)
                        descriptor = {
                            "task_id": TASK_ID,
                            "domain": cell["domain"],
                            "target": cell["target"],
                            "method_id": method_id,
                            "seed": seed,
                            "records": RECORDS_PER_DOMAIN,
                            "shape": [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS],
                            "sequence_tokens": SEQUENCE_TOKENS,
                            "scored_post_bos_tokens": SCORED_POST_BOS,
                            "record_ids_sha256": cell["record_ids_sha256"],
                            "attention_mask_sha256": timing["attention_mask_sha256"],
                            "position_ids_sha256": timing["position_ids_sha256"],
                            "observation_sha256": cell["file"]["sha256"],
                            "state_sha256": state["sha256"],
                            "state_tensor_sha256": state["state_sha256"],
                            "state_selected_step": state["selected_step"],
                            "prediction": prediction_file,
                            "prediction_tensor_sha256": _tensor_digest(ids),
                            "tie_counts": ties_file,
                            "tie_counts_tensor_sha256": _tensor_digest(ties),
                            "timing": timing,
                            "truth_opened": False,
                            "candidate_arrays_persisted": False,
                        }
                        student_cells[cell_id][str(seed)][method_id] = descriptor
                        timings[f"{cell_id}::{seed}::{method_id}"] = timing
                        tie_summary[f"{cell_id}::{seed}::{method_id}"] = {"max_tie_count": int(ties.max().item()), "tied_active_positions": int((ties > 1).sum().item()), "tie_counts": ties.tolist()}
                        _write_create_only(prediction_path.with_suffix(".run.json"), {"schema": TIMING_SCHEMA, "task_id": TASK_ID, "cell_id": cell_id, "seed": seed, "method_id": method_id, "prediction": prediction_file, "tie_counts": ties_file, "timing": timing, "truth_opened": False})
                finally:
                    del model
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                guards.append(_guard(device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, stage=f"after_{key}_cells"))
        for cell_id in CELL_ORDER:
            for seed in SEEDS:
                for method_id in METHOD_ORDER:
                    if method_id not in student_cells[cell_id][str(seed)]:
                        raise PredictionError(f"prediction matrix incomplete: {cell_id}/{seed}/{method_id}")
        fit_record = fit_evidence["receipt_record"]
        observation_record = observation_evidence["manifest_record"]
        manifest = {
            "schema": STUDENT_SCHEMA,
            "task_id": TASK_ID,
            "status": "STUDENT_PREDICTIONS_COMPLETE_NO_TRUTH",
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "candidate_arrays_persisted": False,
            "code_commit": _git_head(root),
            "fit_receipt": fit_record,
            "observation_manifest": observation_record,
            "fit_source_commit": fit_evidence["receipt"].get("source_commit"),
            "domains": list(DOMAINS),
            "target_conditions": list(TARGETS),
            "method_order": list(METHOD_ORDER),
            "replicate_seeds": list(SEEDS),
            "geometry": {"records_per_domain": RECORDS_PER_DOMAIN, "sequence_tokens": SEQUENCE_TOKENS, "scored_post_bos_tokens": SCORED_POST_BOS, "hidden_size": HIDDEN_SIZE, "batch_records": args.batch_records, "projection_chunk": args.projection_chunk},
            "state_bindings": state_bindings,
            "student_cells": student_cells,
            "timings": timings,
            "ties": tie_summary,
            "runtime_assets": {"normalized_public_E": embedding_evidence},
            "numerical_settings": runtime,
            "resource_policy": {"minimum_free_gpu_gib": args.minimum_free_gib, "maximum_gpu_reserved_gib": args.maximum_gpu_reserved_gib, "maximum_host_rss_gib": args.maximum_host_rss_gib, "minimum_host_available_gib": args.minimum_host_available_gib, "max_seconds": args.max_seconds},
            "resource_guards": guards,
            "predictions_count": len(CELL_ORDER) * len(SEEDS) * len(METHOD_ORDER),
            "predictions_complete": True,
            "truth_gate": "student predictions frozen before any truth access; anchor and final joint freeze are separate",
        }
        _write_create_only(output_root / "student_predictions.json", manifest)
        run_manifest = {
            "schema": RUN_SCHEMA,
            "task_id": TASK_ID,
            "status": "PUBLIC_STUDENT_PREDICTIONS_COMPLETE_NO_TRUTH",
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "student_manifest": _file_record(output_root / "student_predictions.json", root=root),
            "fit_receipt": fit_record,
            "observation_manifest": observation_record,
            "code_commit": manifest["code_commit"],
            "fit_source_commit": manifest["fit_source_commit"],
            "predictions_count": manifest["predictions_count"],
            "predictions_complete": True,
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "candidate_arrays_persisted": False,
        }
        _write_create_only(output_root / "run_manifest.json", run_manifest)
        return run_manifest
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_create_only(failure_path, {"schema": FAILURE_SCHEMA, "task_id": TASK_ID, "status": "FAILED_PRESERVED_NO_TRUTH", "started_utc": started_utc, "ended_utc": _utc_now(), "error_type": type(exc).__name__, "error": str(exc), "truth_opened": False, "source_text_loaded": False, "target_labels_loaded": False, "candidate_arrays_persisted": False})
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--batch-records", type=int, default=CAPTURE_BATCH_RECORDS)
    parser.add_argument("--projection-chunk", type=int, default=PROJECTION_CHUNK)
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=6.0)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=16.0)
    parser.add_argument("--minimum-host-available-gib", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run(args)
    except (PredictionError, OSError, RuntimeError, ValueError, VisibilityDecoderError) as exc:
        print(f"TRR-P06 prediction error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
