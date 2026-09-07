#!/usr/bin/env python3
"""Freeze the TRR-P08 activation-only prediction matrix before truth access.

The runner consumes the setup-owned source-free observation panel, the eight
selected P08 states, and the immutable public normalized embedding table.  It
uses the inherited P06 numerical boundary: H128 observations, batch-8 records,
512-row full-vocabulary chunks, deterministic lowest-ID top-1 ties, one warmup
pass, and three measured passes.  It never opens source IDs, target labels,
truth, candidate arrays, or A2 resources.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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
    PAD_TOKEN_ID,
    PAST_ONLY_METHOD,
    POSITIONWISE_METHOD,
    VisibilityDecoderError,
    deterministic_top1,
    file_sha256,
    load_visibility_state,
)

TASK_ID = "TRR-P08"
PREDICTION_SCHEMA = "token-reconstruction.trr-p08-prediction-manifest.v1"
RUN_SCHEMA = "token-reconstruction.trr-p08-prediction-run.v1"
TIMING_SCHEMA = "token-reconstruction.trr-p08-prediction-timing.v1"
FAILURE_SCHEMA = "token-reconstruction.trr-p08-prediction-failure.v1"
OBSERVATION_SCHEMAS = {
    "token-reconstruction.trr-p08-public-observation-manifest.v1",
    # Setup may retain the P06 serializer while changing only its task-owned
    # root and bindings; geometry/flags below still bind the actual P08 panel.
    "token-reconstruction.trr-p06-public-observation-manifest.v1",
}
OBSERVATION_STATUSES = {
    "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
    "FROZEN_P08_PUBLIC_OBSERVATIONS_NO_TRUTH",
}
FIT_SCHEMA = "token-reconstruction.trr-p08-staged-fit.v1"
DOMAINS = ("pile", "finance")
TARGETS = ("public_base", "public_lora_2601")
CELL_ORDER = tuple(f"{domain}__{target}" for domain in DOMAINS for target in TARGETS)
SEEDS = (6106, 6107)
METHOD_ORDER = (
    "p08_positionwise_joint",
    "p08_positionwise_staged",
    "p08_past_only_joint",
    "p08_past_only_staged",
)
BASE_METHODS = {
    "p08_positionwise_joint": POSITIONWISE_METHOD,
    "p08_positionwise_staged": POSITIONWISE_METHOD,
    "p08_past_only_joint": PAST_ONLY_METHOD,
    "p08_past_only_staged": PAST_ONLY_METHOD,
}
RECORDS_PER_DOMAIN = 256
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
CAPTURE_BATCH_RECORDS = 8
PROJECTION_CHUNK = 512
WARMUP_PASSES = 1
MEASURED_PASSES = 3
EMBEDDING_BYTES = 1050673488
EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"


class PredictionError(RuntimeError):
    """Raised when the P08 prediction contract fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


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
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PredictionError(f"file is unavailable: {path}")
    recorded = str(path)
    if root is not None:
        try:
            recorded = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {"path": recorded, "bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise PredictionError(f"{description} must be an object")
    return dict(value)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
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


def _runtime_threads(intraop: int, interop: int) -> dict[str, Any]:
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
    host = _available_host_bytes()
    if rss > int(max_rss_gib * 2**30):
        raise PredictionError(f"RSS guard failed at {stage}: {rss}")
    if host < int(min_host_gib * 2**30):
        raise PredictionError(f"host available-memory guard failed at {stage}: {host}")
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
        result["gpu"] = {"free_bytes": int(free), "total_bytes": int(total), "reserved_bytes": reserved, "allocated_bytes": allocated}
    return result


def _validate_fit_receipt(path: Path, *, root: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    receipt = _load_json(path, description="P08 main-fit receipt")
    if receipt.get("schema") != FIT_SCHEMA or receipt.get("task_id") != TASK_ID or receipt.get("status") != "PASS":
        raise PredictionError("P08 main-fit receipt is not a PASS receipt")
    geometry = receipt.get("geometry")
    expected_geometry = {
        "fit": [1200, 128, HIDDEN_SIZE],
        "validation": [48, 128, HIDDEN_SIZE],
        "record_batch_size": CAPTURE_BATCH_RECORDS,
        "position_budget": PROJECTION_CHUNK,
        "steps": 3000,
        "validation_every": 100,
    }
    if not isinstance(geometry, Mapping) or any(geometry.get(key) != value for key, value in expected_geometry.items()):
        raise PredictionError("P08 main-fit geometry is not the frozen recipe")
    if receipt.get("seeds") != list(SEEDS):
        raise PredictionError("P08 fit seed order changed")
    methods = receipt.get("methods")
    if not isinstance(methods, list) or len(methods) != len(SEEDS) * len(METHOD_ORDER):
        raise PredictionError("P08 fit receipt does not contain exactly eight arms")
    states: dict[tuple[int, str], dict[str, Any]] = {}
    for row in methods:
        if not isinstance(row, Mapping):
            raise PredictionError("P08 arm descriptor is malformed")
        method_id = row.get("arm_id")
        seed = row.get("seed")
        if not isinstance(method_id, str) or method_id not in METHOD_ORDER or isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
            raise PredictionError("P08 arm seed/method matrix changed")
        key = (seed, method_id)
        if key in states or row.get("status") != "PASS" or int(row.get("steps", -1)) != 3000:
            raise PredictionError(f"P08 arm is not a complete PASS: {key}")
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
        base_method = BASE_METHODS[method_id]
        if not isinstance(metadata, Mapping) or metadata.get("p08_method_id") != method_id or metadata.get("method_id") != base_method or int(metadata.get("qkv_init_seed", -1)) != seed:
            raise PredictionError(f"selected state metadata changed: {key}")
        states[key] = {
            "path": state_path,
            "file": actual,
            "sha256": actual["sha256"],
            "state_sha256": state_sha,
            "selected_step": int(state.get("selected_step", metadata.get("selected_step", -1))),
            "metadata": dict(metadata),
            "base_method_id": base_method,
        }
    expected = {(seed, method) for seed in SEEDS for method in METHOD_ORDER}
    if set(states) != expected:
        raise PredictionError("P08 selected-state matrix is incomplete")
    return states, {"receipt": receipt, "receipt_record": _file_record(path, root=root)}


def _validate_observation_manifest(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(path, description="P08 public observation manifest")
    if manifest.get("schema") not in OBSERVATION_SCHEMAS or manifest.get("status") not in OBSERVATION_STATUSES:
        raise PredictionError("observation manifest is not a frozen no-truth panel")
    if manifest.get("task_id") not in (TASK_ID, "TRR-P06"):
        raise PredictionError("observation manifest task identity changed")
    geometry = manifest.get("geometry")
    if isinstance(geometry, Mapping):
        if geometry.get("hidden_size", HIDDEN_SIZE) != HIDDEN_SIZE or geometry.get("sequence_tokens_including_bos", SEQUENCE_TOKENS) != SEQUENCE_TOKENS:
            raise PredictionError("observation geometry changed")
    for flag in ("source_text_written", "source_text_loaded", "token_ids_written", "target_labels_loaded", "truth_opened"):
        if flag in manifest and manifest.get(flag) is not False:
            raise PredictionError(f"observation manifest has forbidden flag: {flag}")
    if manifest.get("records_per_domain") != RECORDS_PER_DOMAIN or manifest.get("sequence_tokens_including_bos") != SEQUENCE_TOKENS or manifest.get("scored_post_bos_tokens") != SCORED_POST_BOS or manifest.get("hidden_size") != HIDDEN_SIZE:
        raise PredictionError("observation manifest H128/256 geometry changed")
    if manifest.get("cell_order") != list(CELL_ORDER):
        raise PredictionError("observation cell order changed")
    if manifest.get("source_pairing", {}).get("same_record_ids_across_targets") is not True:
        raise PredictionError("observation target pairing is not declared")
    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or [row.get("cell_id") for row in raw_cells if isinstance(row, Mapping)] != list(CELL_ORDER):
        raise PredictionError("observation cells are incomplete or reordered")
    cells: dict[str, Any] = {}
    pair_digests: dict[str, str] = {}
    for row in raw_cells:
        if not isinstance(row, Mapping):
            raise PredictionError("observation cell descriptor is malformed")
        cell_id = str(row.get("cell_id"))
        domain, target = cell_id.split("__", 1)
        if domain not in DOMAINS or target not in TARGETS or row.get("style") != domain or row.get("condition") != target:
            raise PredictionError(f"observation cell identity changed: {cell_id}")
        record_ids_sha = row.get("record_ids_sha256")
        if not isinstance(record_ids_sha, str) or len(record_ids_sha) != 64:
            raise PredictionError(f"record identity digest missing: {cell_id}")
        if domain in pair_digests and pair_digests[domain] != record_ids_sha:
            raise PredictionError(f"paired target record order changed: {domain}")
        pair_digests[domain] = record_ids_sha
        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise PredictionError(f"observation asset missing: {cell_id}")
        if observation.get("shape") != [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, HIDDEN_SIZE] or observation.get("scored_post_bos_tokens") != SCORED_POST_BOS:
            raise PredictionError(f"observation asset geometry changed: {cell_id}")
        if "stored_sequence_tokens" in observation and observation.get("stored_sequence_tokens") != SEQUENCE_TOKENS:
            raise PredictionError(f"observation stored length changed: {cell_id}")
        raw_path = observation.get("path")
        if not isinstance(raw_path, str):
            raise PredictionError(f"observation asset path missing: {cell_id}")
        asset_path = _resolve(raw_path, root=root, description=f"observation {cell_id}")
        actual = _file_record(asset_path, root=root)
        if actual["bytes"] != observation.get("bytes") or actual["sha256"] != observation.get("sha256"):
            raise PredictionError(f"observation asset binding changed: {cell_id}")
        cells[cell_id] = {"cell_id": cell_id, "domain": domain, "target": target, "record_ids_sha256": record_ids_sha, "observation": dict(observation), "path": asset_path, "file": actual}
    return cells, {"manifest": manifest, "manifest_record": _file_record(path, root=root), "record_ids_sha256": pair_digests}


def _load_observation_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(cell["path"])).resolve()
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise PredictionError(f"observation keys changed: {path}")
            activations = handle.get_tensor("activations").contiguous()
            mask = handle.get_tensor("attention_mask").contiguous()
            positions = handle.get_tensor("position_ids").contiguous()
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
    if not mask[:, 0].all().item() or (mask[:, 1:] > mask[:, :-1]).any().item():
        raise PredictionError(f"observation mask is invalid or not right-padded: {path}")
    expected_positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.long).expand(RECORDS_PER_DOMAIN, -1)
    if not torch.equal(positions.to(dtype=torch.long), expected_positions):
        raise PredictionError(f"observation positions are not 0..127: {path}")
    return {
        "activations": activations,
        "mask": mask,
        "positions": positions.to(dtype=torch.long),
        "attention_mask_sha256": _tensor_digest(mask.to(dtype=torch.uint8)),
        "position_ids_sha256": _tensor_digest(positions.to(dtype=torch.int64)),
    }


def _load_embedding(path: Path, *, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    actual = _file_record(path)
    if actual["bytes"] != EMBEDDING_BYTES or actual["sha256"] != EMBEDDING_SHA256:
        raise PredictionError("normalized public embedding table binding changed")
    started = time.perf_counter()
    table_cpu: torch.Tensor | None = None
    loaded: dict[str, torch.Tensor] | None = None
    try:
        loaded = load_file(str(path), device="cpu")
        if set(loaded) != {"embeddings"}:
            raise PredictionError("embedding table must contain only embeddings")
        table_cpu = loaded["embeddings"].contiguous()
        if table_cpu.dtype != torch.float32 or tuple(table_cpu.shape) != (VOCABULARY_SIZE, HIDDEN_SIZE) or not torch.isfinite(table_cpu).all().item():
            raise PredictionError("embedding table geometry, dtype, or finiteness changed")
        table = table_cpu.to(device=device, dtype=torch.float32).contiguous()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError("embedding table load failed") from exc
    finally:
        if loaded is not None:
            del loaded
        if table_cpu is not None and device.type == "cuda":
            del table_cpu
        gc.collect()
    return table, {"file": actual, "shape": [VOCABULARY_SIZE, HIDDEN_SIZE], "dtype": "torch.float32", "load_seconds": time.perf_counter() - started, "finite_scan": "one immutable load-time scan"}


def _validate_model_table(model: torch.nn.Module, embedding: torch.Tensor) -> None:
    if embedding.ndim != 2 or tuple(embedding.shape) != (VOCABULARY_SIZE, HIDDEN_SIZE) or embedding.dtype != torch.float32:
        raise PredictionError("embedding table geometry changed in hot path")
    if int(getattr(model, "hidden_size", -1)) != HIDDEN_SIZE or int(getattr(model, "vocabulary_size", -1)) != VOCABULARY_SIZE:
        raise PredictionError("decoder geometry changed")


@torch.inference_mode()
def predict_batch(model: torch.nn.Module, embedding: torch.Tensor, activations: torch.Tensor, valid_mask: torch.Tensor, *, device: torch.device, projection_chunk: int = PROJECTION_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the inherited P06 batch-8 projection/readout boundary."""

    if projection_chunk <= 0 or activations.ndim != 3 or tuple(activations.shape[1:]) != (SEQUENCE_TOKENS, HIDDEN_SIZE) or tuple(valid_mask.shape) != tuple(activations.shape[:2]):
        raise PredictionError("activation batch geometry changed")
    _validate_model_table(model, embedding)
    staged = activations.to(device=device, dtype=torch.float32)
    mask = valid_mask.to(device=device, dtype=torch.bool)
    projected = model.projected_hidden(staged, mask)
    ids = torch.full((activations.shape[0], SEQUENCE_TOKENS), PAD_TOKEN_ID, dtype=torch.long, device="cpu")
    ties = torch.zeros((activations.shape[0], SEQUENCE_TOKENS), dtype=torch.int64, device="cpu")
    if not mask[:, 0].all().item():
        raise PredictionError("observation batch lacks valid BOS")
    post_bos = mask.clone()
    post_bos[:, 0] = False
    indices = torch.nonzero(post_bos, as_tuple=False)
    for start in range(0, int(indices.shape[0]), projection_chunk):
        current = indices[start : start + projection_chunk]
        logits = model.logits_from_rows(projected, current[:, 0], current[:, 1], embedding)
        predicted, tie_count = deterministic_top1(logits)
        rows = current[:, 0].detach().cpu()
        positions = current[:, 1].detach().cpu()
        ids[rows, positions] = predicted.detach().cpu().to(dtype=torch.long)
        ties[rows, positions] = tie_count.detach().cpu().to(dtype=torch.int64)
    active = mask.detach().cpu()
    ids[~active] = PAD_TOKEN_ID
    ties[~active] = 0
    ids[:, 0] = BOS_TOKEN_ID
    ties[:, 0] = 1
    if (ids[active] < 0).any().item() or (ids[active] >= VOCABULARY_SIZE).any().item() or not ids[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise PredictionError("prediction IDs failed the frozen vocabulary/BOS contract")
    return ids.contiguous(), ties.contiguous()


def _peak_memory(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
    }


def _predict_cell(model: torch.nn.Module, embedding: torch.Tensor, cell: Mapping[str, Any], *, device: torch.device, started: float, max_seconds: float, min_free_gib: float, max_reserved_gib: float, max_rss_gib: float, min_host_gib: float, batch_records: int, projection_chunk: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if RECORDS_PER_DOMAIN % batch_records != 0:
        raise PredictionError("records per domain must be divisible by the common batch size")
    load_started = time.perf_counter()
    loaded = _load_observation_cell(cell)
    observation_load_seconds = time.perf_counter() - load_started
    activations = loaded["activations"]
    mask = loaded["mask"]
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
            ids, ties = predict_batch(model, embedding, activations[row_start:row_stop], mask[row_start:row_stop], device=device, projection_chunk=projection_chunk)
            pass_ids[row_start:row_stop] = ids
            pass_ties[row_start:row_stop] = ties
            _guard(device=device, started=started, max_seconds=max_seconds, min_free_gib=min_free_gib, max_reserved_gib=max_reserved_gib, max_rss_gib=max_rss_gib, min_host_gib=min_host_gib, stage=f"after_{cell['cell_id']}_rows_{row_stop}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return pass_ids, pass_ties, time.perf_counter() - pass_started

    warm_ids, warm_ties, warm_seconds = one_pass()
    measured: list[float] = []
    selected_ids: torch.Tensor | None = None
    selected_ties: torch.Tensor | None = None
    for _ in range(MEASURED_PASSES):
        ids, ties, elapsed = one_pass()
        if selected_ids is None:
            selected_ids, selected_ties = ids, ties
        elif not torch.equal(selected_ids, ids) or not torch.equal(selected_ties, ties):
            raise PredictionError(f"repeated prediction IDs or ties differ for {cell['cell_id']}")
        measured.append(elapsed)
    if not torch.equal(warm_ids, selected_ids) or not torch.equal(warm_ties, selected_ties):
        raise PredictionError(f"warmup and measured prediction IDs differ for {cell['cell_id']}")
    timing = {
        "schema": TIMING_SCHEMA,
        "task_id": TASK_ID,
        "cell_id": cell["cell_id"],
        "records": RECORDS_PER_DOMAIN,
        "batch_records": batch_records,
        "projection_chunk": projection_chunk,
        "warmup_passes": WARMUP_PASSES,
        "measured_passes": MEASURED_PASSES,
        "warmup_seconds": float(warm_seconds),
        "measured_seconds": [float(value) for value in measured],
        "measured_mean_seconds": float(sum(measured) / len(measured)),
        "measured_ms_per_record": float(1000.0 * sum(measured) / len(measured) / RECORDS_PER_DOMAIN),
        "repeat_prediction_exact": True,
        "observation_load_seconds": float(observation_load_seconds),
        "observation_load_excluded_from_measured_interval": True,
        "measurement_includes_resource_guard_and_device_synchronization": True,
        "measured_interval_definition": "full 256-record prediction passes with common batch-8/512-row readout, per-batch guards, and device synchronization; observation deserialization is timed separately",
        "peak_memory": _peak_memory(device),
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


def _asset_json(asset: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in asset.items()}


def _context_records(args: argparse.Namespace, *, root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, raw in (("p06_prediction_manifest", args.p06_context_manifest), ("trr0006_prediction_manifest", args.trr0006_context_manifest)):
        if raw is None:
            continue
        path = _resolve(raw, root=root, description=label)
        result[label] = _file_record(path, root=root)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(root / "experiments" / "TRR-P08")
    except ValueError as exc:
        raise PredictionError("prediction output must be task-owned under experiments/TRR-P08") from exc
    if output_root.exists() or output_root.is_symlink():
        raise PredictionError(f"prediction output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    started_clock = time.perf_counter()
    started_utc = _utc_now()
    failure_path = output_root / "failure.json"
    try:
        runtime = _runtime_threads(args.torch_threads, args.torch_interop_threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise PredictionError("CUDA requested but unavailable")
        device = torch.device(args.device)
        fit_states, fit_evidence = _validate_fit_receipt(Path(args.fit_receipt).expanduser().resolve(), root=root)
        cells, observation_evidence = _validate_observation_manifest(Path(args.observation_manifest).expanduser().resolve(), root=root)
        guards: list[dict[str, Any]] = []
        guards.append(_guard(device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, stage="before_embedding_load"))
        embedding, embedding_evidence = _load_embedding(Path(args.embedding_path).expanduser().resolve(), device=device)
        guards.append(_guard(device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, stage="after_embedding_load"))
        state_bindings: dict[str, Any] = {}
        prediction_cells: dict[str, dict[str, dict[str, Any]]] = {cell_id: {str(seed): {} for seed in SEEDS} for cell_id in CELL_ORDER}
        timings: dict[str, Any] = {}
        tie_summary: dict[str, Any] = {}
        for seed in SEEDS:
            for method_id in METHOD_ORDER:
                state = fit_states[(seed, method_id)]
                state_key = f"{seed}::{method_id}"
                state_bindings[state_key] = dict(state["file"])
                try:
                    model = load_visibility_state(state["path"], method_id=state["base_method_id"], hidden_size=HIDDEN_SIZE, vocabulary_size=VOCABULARY_SIZE, context_width=128, expected_sha256=state["sha256"]).to(device=device).eval()
                except (VisibilityDecoderError, OSError, RuntimeError, ValueError) as exc:
                    raise PredictionError(f"selected P08 decoder state load failed: {state_key}") from exc
                model.requires_grad_(False)
                guards.append(_guard(device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, stage=f"after_{state_key}_load"))
                try:
                    for cell_id in CELL_ORDER:
                        ids, ties, timing = _predict_cell(model, embedding, cells[cell_id], device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, batch_records=args.batch_records, projection_chunk=args.projection_chunk)
                        prediction_path = _prediction_path(output_root, cell_id, seed, method_id)
                        ties_path = _tie_path(output_root, cell_id, seed, method_id)
                        metadata = {
                            "schema": PREDICTION_SCHEMA,
                            "task_id": TASK_ID,
                            "domain": cells[cell_id]["domain"],
                            "target": cells[cell_id]["target"],
                            "method_id": method_id,
                            "base_method_id": state["base_method_id"],
                            "seed": str(seed),
                            "records": str(RECORDS_PER_DOMAIN),
                            "sequence_tokens": str(SEQUENCE_TOKENS),
                            "scored_post_bos_tokens": str(SCORED_POST_BOS),
                            "state_file_sha256": state["sha256"],
                            "observation_sha256": cells[cell_id]["file"]["sha256"],
                            "truth_opened": "false",
                            "source_text_loaded": "false",
                            "candidate_arrays_persisted": "false",
                        }
                        prediction_path.parent.mkdir(parents=True, exist_ok=True)
                        ties_path.parent.mkdir(parents=True, exist_ok=True)
                        save_file({"predictions": ids.contiguous()}, str(prediction_path), metadata=metadata)
                        save_file({"tie_counts": ties.contiguous()}, str(ties_path), metadata={**metadata, "schema": "token-reconstruction.trr-p08-tie-counts.v1"})
                        prediction_file = _file_record(prediction_path, root=root)
                        tie_file = _file_record(ties_path, root=root)
                        descriptor = {
                            "schema": "token-reconstruction.trr-p08-prediction-descriptor.v1",
                            "task_id": TASK_ID,
                            "domain": cells[cell_id]["domain"],
                            "target": cells[cell_id]["target"],
                            "method_id": method_id,
                            "base_method_id": state["base_method_id"],
                            "seed": int(seed),
                            "records": RECORDS_PER_DOMAIN,
                            "shape": [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS],
                            "sequence_tokens": SEQUENCE_TOKENS,
                            "scored_post_bos_tokens": SCORED_POST_BOS,
                            "record_ids_sha256": cells[cell_id]["record_ids_sha256"],
                            "attention_mask_sha256": timing["attention_mask_sha256"],
                            "position_ids_sha256": timing["position_ids_sha256"],
                            "observation": cells[cell_id]["file"],
                            "state": state["file"],
                            "state_tensor_sha256": state["state_sha256"],
                            "state_selected_step": state["selected_step"],
                            "prediction": prediction_file,
                            "prediction_tensor_sha256": _tensor_digest(ids),
                            "tie_counts": tie_file,
                            "tie_counts_tensor_sha256": _tensor_digest(ties),
                            "timing": timing,
                            "truth_opened": False,
                            "source_text_loaded": False,
                            "target_labels_loaded": False,
                            "candidate_arrays_persisted": False,
                        }
                        prediction_cells[cell_id][str(seed)][method_id] = descriptor
                        timings[f"{cell_id}::{seed}::{method_id}"] = timing
                        tie_summary[f"{cell_id}::{seed}::{method_id}"] = {
                            "max_tie_count": int(ties.max().item()),
                            "tied_active_positions": int((ties > 1).sum().item()),
                            "tie_tensor_sha256": _tensor_digest(ties),
                        }
                        _write_create_only(prediction_path.with_suffix(".run.json"), {"schema": TIMING_SCHEMA, "task_id": TASK_ID, "cell_id": cell_id, "seed": int(seed), "method_id": method_id, "prediction": prediction_file, "tie_counts": tie_file, "timing": timing, "truth_opened": False})
                finally:
                    del model
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                guards.append(_guard(device=device, started=started_clock, max_seconds=args.max_seconds, min_free_gib=args.minimum_free_gib, max_reserved_gib=args.maximum_gpu_reserved_gib, max_rss_gib=args.maximum_host_rss_gib, min_host_gib=args.minimum_host_available_gib, stage=f"after_{state_key}_cells"))
        for cell_id in CELL_ORDER:
            for seed in SEEDS:
                for method_id in METHOD_ORDER:
                    if method_id not in prediction_cells[cell_id][str(seed)]:
                        raise PredictionError(f"prediction matrix incomplete: {cell_id}/{seed}/{method_id}")
        context = _context_records(args, root=root)
        manifest = {
            "schema": PREDICTION_SCHEMA,
            "task_id": TASK_ID,
            "status": "FROZEN_P08_PREDICTIONS_NO_TRUTH",
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "candidate_arrays_persisted": False,
            "code_commit": _git_head(root),
            "fit_receipt": fit_evidence["receipt_record"],
            "observation_manifest": observation_evidence["manifest_record"],
            "fit_source_commit": fit_evidence["receipt"].get("source_commit"),
            "domains": list(DOMAINS),
            "target_conditions": list(TARGETS),
            "cell_order": list(CELL_ORDER),
            "method_order": list(METHOD_ORDER),
            "replicate_seeds": list(SEEDS),
            "geometry": {"records_per_domain": RECORDS_PER_DOMAIN, "sequence_tokens": SEQUENCE_TOKENS, "scored_post_bos_tokens": SCORED_POST_BOS, "hidden_size": HIDDEN_SIZE, "vocabulary_size": VOCABULARY_SIZE, "batch_records": args.batch_records, "projection_chunk": args.projection_chunk, "padding_sentinel": PAD_TOKEN_ID},
            "state_bindings": state_bindings,
            "prediction_cells": prediction_cells,
            "student_cells": prediction_cells,
            "timings": timings,
            "tie_summary": tie_summary,
            "runtime_assets": {"normalized_public_E": embedding_evidence},
            "reference_context": context,
            "numerical_settings": runtime,
            "resource_policy": {"minimum_free_gpu_gib": args.minimum_free_gib, "maximum_gpu_reserved_gib": args.maximum_gpu_reserved_gib, "maximum_host_rss_gib": args.maximum_host_rss_gib, "minimum_host_available_gib": args.minimum_host_available_gib, "max_seconds": args.max_seconds},
            "resource_guards": guards,
            "predictions_count": len(CELL_ORDER) * len(SEEDS) * len(METHOD_ORDER),
            "predictions_complete": True,
            "truth_gate": "all eight P08 states and four paired observation cells are frozen before any truth/score reader; this producer never opens truth",
        }
        _write_create_only(output_root / "p08_predictions.json", manifest)
        run_manifest = {
            "schema": RUN_SCHEMA,
            "task_id": TASK_ID,
            "status": "P08_FROZEN_PREDICTIONS_COMPLETE_NO_TRUTH",
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "prediction_manifest": _file_record(output_root / "p08_predictions.json", root=root),
            "fit_receipt": fit_evidence["receipt_record"],
            "observation_manifest": observation_evidence["manifest_record"],
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
    parser.add_argument("--p06-context-manifest", dest="p06_context_manifest", type=Path)
    parser.add_argument("--trr0006-context-manifest", dest="trr0006_context_manifest", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.batch_records != CAPTURE_BATCH_RECORDS or args.projection_chunk != PROJECTION_CHUNK:
            raise PredictionError("P08 production prediction requires batch-8 and projection-chunk-512")
        run(args)
    except (PredictionError, OSError, RuntimeError, ValueError, VisibilityDecoderError) as exc:
        print(f"TRR-P08 prediction error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
