#!/usr/bin/env python3
"""Replay the frozen TRR-P06 and TRR-0006 decoders on common inputs.

This runner owns only retrospective inference.  It reads source-free
activation observations and the already-selected decoder states, writes
create-only prediction/tie artifacts, and never opens token IDs, source text,
labels, or truth.  The two historical execution paths are intentionally kept
separate: P06 uses its published batch-8, chunked readout; TRR-0006 uses the
published native one-record full-logit call.  A small exact fixture is checked
before the new matrix so that a port cannot silently replace the native path.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Iterator

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from token_reconstruction.trr0005_joint_decoder import load_decoder_state  # noqa: E402
from token_reconstruction.trr0006_visibility_decoder import (  # noqa: E402
    BOS_TOKEN_ID,
    deterministic_top1,
    file_sha256,
    load_visibility_state,
)


TASK_ID = "TRR-P07"
SCHEMA = "token-reconstruction.trr-p07-frozen-replay.v1"
RUN_SCHEMA = "token-reconstruction.trr-p07-frozen-replay-run.v1"
FAILURE_SCHEMA = "token-reconstruction.trr-p07-frozen-replay-failure.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr-p07-predictions.v1"
TIE_SCHEMA = "token-reconstruction.trr-p07-tie-counts.v1"
CELL_ORDER = (
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
)
DOMAINS = ("pile", "finance")
TARGETS = ("public_base", "public_lora_2601")
SEEDS = (6106, 6107)
P06_METHODS = ("p06_past_only", "p06_positionwise_diagonal")
OLD_METHODS = (
    "enriched__affine_causal_h_attention128",
    "enriched__affine_trained_diagonal_attention128",
)
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
P06_RECORDS = 256
TRR0006_RECORDS = 1536
TRR0006_SUBSET_RECORDS = 256
P06_BATCH_RECORDS = 8
P06_PROJECTION_CHUNK = 512
P06_WARMUP_PASSES = 1
P06_MEASURED_PASSES = 3
OLD_WARMUP_RUNS_PER_RECORD = 1
OLD_MEASURED_RUNS_PER_RECORD = 1
EMBEDDING_BYTES = 1050673488
EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
P06_OBSERVATION_SCHEMA = "token-reconstruction.trr-p06-public-observation-manifest.v1"
OLD_OBSERVATION_SCHEMA = "token-reconstruction.trr0006-public-observation-manifest.v1"
P06_PREDICTION_SCHEMA = "token-reconstruction.trr-p06-prediction-manifest.v1"
OLD_REGISTRATION_SCHEMA = "token-reconstruction.trr0006-frozen-pair-prediction-registration.v1"
OLD_PREDICTION_DESCRIPTOR_SCHEMA = "token-reconstruction.trr0006-prediction-descriptor-manifest.v1"
PLAN_SHA256 = "a0a2339f1a4b77e02d7d1772459dc14d442a4ce24b5111a01e58622ca1ae7c3e"


class ReplayError(RuntimeError):
    """Raised when the frozen replay contract fails closed."""


@dataclass(frozen=True)
class Asset:
    path: Path
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Cell:
    panel: str
    cell_id: str
    domain: str
    target: str
    path: Path
    asset: Asset
    record_ids_sha256: str
    records: int
    subset_indices: tuple[int, ...]
    attention_mask_sha256: str | None = None
    position_ids_sha256: str | None = None


@dataclass(frozen=True)
class Method:
    key: str
    family: str
    method_id: str
    seed: int | None
    state_path: Path
    state_asset: Asset
    state_tensor_sha256: str | None
    loader: str
    execution: str
    base_method_id: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    """Hash tensor geometry, dtype, and contiguous CPU bytes."""

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_record(path: Path) -> Asset:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"asset is unavailable: {path}")
    return Asset(path=path, bytes=int(path.stat().st_size), sha256=file_sha256(path))


def _asset_from_binding(binding: Mapping[str, Any], *, root: Path, description: str) -> Asset:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ReplayError(f"{description} path is missing")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    actual = _file_record(path)
    if binding.get("bytes") is not None and int(binding["bytes"]) != actual.bytes:
        raise ReplayError(f"{description} byte binding changed: {actual.path}")
    expected_sha = binding.get("sha256")
    if not isinstance(expected_sha, str) or actual.sha256 != expected_sha:
        raise ReplayError(f"{description} hash binding changed: {actual.path}")
    return actual


def _load_json(path: Path, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ReplayError(f"{description} must be an object")
    return dict(value)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ReplayError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReplayError("cannot resolve executable source commit") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ReplayError("executable source commit is not a full lowercase hash")
    return value


def _available_host_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                return int(fields[1]) * 1024
    except (OSError, UnicodeError):
        pass
    raise ReplayError("host available-memory guard is unavailable")


def _guard(
    *,
    device: torch.device,
    started: float,
    max_seconds: float,
    min_free_gib: float,
    max_reserved_gib: float,
    max_rss_gib: float,
    min_host_gib: float,
    stage: str,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    host = _available_host_bytes()
    max_rss = int(max_rss_gib * 2**30)
    min_host = int(min_host_gib * 2**30)
    if elapsed > max_seconds:
        raise ReplayError(f"wall-time guard failed at {stage}: {elapsed:.3f}s")
    if rss > max_rss:
        raise ReplayError(f"RSS guard failed at {stage}: {rss} > {max_rss}")
    if host < min_host:
        raise ReplayError(f"host available-memory guard failed at {stage}: {host} < {min_host}")
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
            raise ReplayError(f"GPU free-memory guard failed at {stage}: {free}")
        if reserved > int(max_reserved_gib * 2**30):
            raise ReplayError(f"GPU reserved-memory guard failed at {stage}: {reserved}")
        result["gpu"] = {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "reserved_bytes": reserved,
            "allocated_bytes": allocated,
        }
    return result


def _configure_threads(intraop: int, interop: int) -> dict[str, int]:
    try:
        torch.set_num_threads(int(intraop))
        torch.set_num_interop_threads(int(interop))
    except (RuntimeError, ValueError) as exc:
        raise ReplayError("Torch thread configuration failed") from exc
    return {"cpu_intraop_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads()}


def _validate_source_free_manifest(manifest: Mapping[str, Any], *, schema: str, task_id: str, records: int, description: str) -> None:
    if manifest.get("schema") != schema or manifest.get("task_id") != task_id or manifest.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise ReplayError(f"{description} is not the frozen no-truth observation manifest")
    for key in ("source_text_written", "token_ids_written", "target_labels_loaded", "truth_opened"):
        if manifest.get(key) is not False:
            raise ReplayError(f"{description} has forbidden truth/source flag: {key}")
    if manifest.get("records_per_domain") != records or manifest.get("sequence_tokens_including_bos") != SEQUENCE_TOKENS or manifest.get("scored_post_bos_tokens") != SCORED_POST_BOS or manifest.get("hidden_size") != HIDDEN_SIZE:
        raise ReplayError(f"{description} geometry changed")
    raw_order = manifest.get("cell_order")
    if not isinstance(raw_order, list) or set(raw_order) != set(CELL_ORDER) or len(raw_order) != len(CELL_ORDER):
        raise ReplayError(f"{description} cell set changed")
    pairing = manifest.get("source_pairing")
    if not isinstance(pairing, Mapping) or pairing.get("same_record_ids_across_targets") is not True:
        raise ReplayError(f"{description} target pairing is not frozen")


def _observation_rows(
    *,
    cell: Cell,
    batch_start: int = 0,
    batch_stop: int | None = None,
    indices: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Read one bounded source-free slice, never loading a whole panel.

    ``indices`` is used for the approved evenly-spaced TRR-0006 subset.  It
    intentionally reads only those rows rather than reading a contiguous
    prefix and silently changing the frozen selection.
    """

    if indices is None:
        if batch_stop is None:
            raise ReplayError("batch_stop is required for contiguous observation reads")
        row_indices = tuple(range(batch_start, batch_stop))
    else:
        row_indices = tuple(int(index) for index in indices)
        if any(index < 0 or index >= cell.records for index in row_indices):
            raise ReplayError(f"observation row index is outside the frozen panel: {cell.cell_id}")
    if not row_indices:
        raise ReplayError(f"observation row selection is empty: {cell.cell_id}")

    try:
        with safe_open(str(cell.path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise ReplayError(f"observation keys changed: {cell.path}")
            a_slice = handle.get_slice("activations")
            m_slice = handle.get_slice("attention_mask")
            p_slice = handle.get_slice("position_ids")
            if tuple(a_slice.get_shape()) != (cell.records, SEQUENCE_TOKENS, HIDDEN_SIZE) or tuple(m_slice.get_shape()) != (cell.records, SEQUENCE_TOKENS) or tuple(p_slice.get_shape()) != (cell.records, SEQUENCE_TOKENS):
                raise ReplayError(f"observation geometry changed: {cell.cell_id}")
            contiguous = row_indices == tuple(range(row_indices[0], row_indices[-1] + 1))
            if contiguous:
                start, stop = row_indices[0], row_indices[-1] + 1
                activations = a_slice[start:stop].contiguous()
                mask = m_slice[start:stop].contiguous()
                positions = p_slice[start:stop].contiguous()
            else:
                activations = torch.cat([a_slice[index:index + 1].contiguous() for index in row_indices], dim=0)
                mask = torch.cat([m_slice[index:index + 1].contiguous() for index in row_indices], dim=0)
                positions = torch.cat([p_slice[index:index + 1].contiguous() for index in row_indices], dim=0)
    except ReplayError:
        raise
    except Exception as exc:
        raise ReplayError(f"observation slice failed: {cell.path}") from exc
    selected_records = len(row_indices)
    if activations.dtype != torch.bfloat16 or mask.dtype not in (torch.bool, torch.uint8) or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ReplayError(f"observation dtypes changed: {cell.cell_id}")
    if not torch.isfinite(activations.float()).all().item():
        raise ReplayError(f"observation activations are non-finite: {cell.cell_id}")
    if mask.dtype == torch.uint8 and ((mask != 0) & (mask != 1)).any().item():
        raise ReplayError(f"observation mask is not binary: {cell.cell_id}")
    mask = mask.to(dtype=torch.bool)
    if not mask[:, 0].all().item() or (mask[:, 1:] > mask[:, :-1]).any().item():
        raise ReplayError(f"observation mask lacks BOS or is not right-padded: {cell.cell_id}")
    expected_positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.long).expand(selected_records, -1)
    if not torch.equal(positions.to(dtype=torch.long), expected_positions):
        raise ReplayError(f"position IDs changed: {cell.cell_id}")
    sidecars = {
        "attention_mask_sha256": tensor_digest(mask.to(dtype=torch.uint8)),
        "position_ids_sha256": tensor_digest(positions.to(dtype=torch.int64)),
        "row_indices": list(row_indices),
        "row_indices_sha256": _json_digest(list(row_indices)),
    }
    return activations, mask, positions.to(dtype=torch.long), sidecars


def _manifest_cells(manifest_path: Path, *, root: Path, panel: str) -> tuple[dict[str, Cell], Asset, dict[str, Any]]:
    manifest_asset = _file_record(manifest_path)
    manifest = _load_json(manifest_path, f"{panel} observation manifest")
    if panel == "p06_panel":
        _validate_source_free_manifest(manifest, schema=P06_OBSERVATION_SCHEMA, task_id="TRR-P06", records=P06_RECORDS, description="P06 observation manifest")
        records = P06_RECORDS
    elif panel == "trr0006_subset":
        _validate_source_free_manifest(manifest, schema=OLD_OBSERVATION_SCHEMA, task_id="TRR-0006", records=TRR0006_RECORDS, description="TRR-0006 observation manifest")
        records = TRR0006_RECORDS
    else:
        raise ReplayError(f"unknown panel: {panel}")
    cells: dict[str, Cell] = {}
    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != len(CELL_ORDER) or {row.get("cell_id") for row in raw_cells if isinstance(row, Mapping)} != set(CELL_ORDER):
        raise ReplayError(f"{panel} cells are incomplete")
    rows_by_cell = {str(row.get("cell_id")): row for row in raw_cells if isinstance(row, Mapping)}
    pair_digest: dict[str, str] = {}
    for cell_id in CELL_ORDER:
        row = rows_by_cell[cell_id]
        if not isinstance(row, Mapping):
            raise ReplayError(f"{panel} cell is malformed")
        cell_id = str(row.get("cell_id"))
        domain, target = cell_id.split("__", 1)
        if domain not in DOMAINS or target not in TARGETS or row.get("style") != domain or row.get("condition") != target:
            raise ReplayError(f"{panel} cell identity changed: {cell_id}")
        record_ids_sha = row.get("record_ids_sha256")
        if not isinstance(record_ids_sha, str) or len(record_ids_sha) != 64:
            raise ReplayError(f"{panel} record identity binding missing: {cell_id}")
        if domain in pair_digest and pair_digest[domain] != record_ids_sha:
            raise ReplayError(f"{panel} target pairing changed: {domain}")
        pair_digest[domain] = record_ids_sha
        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise ReplayError(f"{panel} observation binding missing: {cell_id}")
        if observation.get("shape") != [records, SEQUENCE_TOKENS, HIDDEN_SIZE] or observation.get("scored_post_bos_tokens") != SCORED_POST_BOS:
            raise ReplayError(f"{panel} observation geometry changed: {cell_id}")
        asset = _asset_from_binding(observation, root=root, description=f"{panel} observation {cell_id}")
        subset = tuple(range(records if panel == "p06_panel" else TRR0006_SUBSET_RECORDS))
        cells[cell_id] = Cell(panel=panel, cell_id=cell_id, domain=domain, target=target, path=asset.path, asset=asset, record_ids_sha256=record_ids_sha, records=records, subset_indices=subset)
    return cells, manifest_asset, manifest


def _validate_p06_states(prediction_manifest_path: Path, *, root: Path) -> tuple[dict[tuple[int, str], Method], Asset, dict[str, Any]]:
    manifest_asset = _file_record(prediction_manifest_path)
    manifest = _load_json(prediction_manifest_path, "P06 prediction manifest")
    if manifest.get("schema") != P06_PREDICTION_SCHEMA or manifest.get("task_id") != "TRR-P06" or manifest.get("status") != "FROZEN_P06_PREDICTIONS_NO_TRUTH":
        raise ReplayError("P06 prediction manifest is not the frozen no-truth matrix")
    for key in ("truth_opened", "source_text_loaded", "target_labels_loaded", "candidate_arrays_persisted"):
        if manifest.get(key) is not False:
            raise ReplayError(f"P06 prediction manifest has forbidden flag: {key}")
    if manifest.get("replicate_seeds") != list(SEEDS) or manifest.get("method_order") != ["p06_positionwise_diagonal", "p06_past_only", "p06_full_record"]:
        raise ReplayError("P06 prediction method matrix changed")
    methods: dict[tuple[int, str], Method] = {}
    bindings = manifest.get("state_bindings")
    if not isinstance(bindings, Mapping):
        raise ReplayError("P06 state bindings are missing")
    for seed in SEEDS:
        for method_id in P06_METHODS:
            row = bindings.get(f"{seed}::{method_id}")
            if not isinstance(row, Mapping):
                raise ReplayError(f"P06 state binding missing: {seed}/{method_id}")
            path = row.get("path")
            if not isinstance(path, str):
                raise ReplayError(f"P06 state path missing: {seed}/{method_id}")
            state_path = Path(path).expanduser()
            if not state_path.is_absolute():
                state_path = root / state_path
            asset = _file_record(state_path)
            if asset.sha256 != row.get("sha256") or asset.bytes != row.get("bytes"):
                raise ReplayError(f"P06 state file binding changed: {seed}/{method_id}")
            methods[(seed, method_id)] = Method(
                key=f"{method_id}__seed{seed}",
                family="p06",
                method_id=method_id,
                seed=seed,
                state_path=asset.path,
                state_asset=asset,
                state_tensor_sha256=str(row.get("state_sha256")) if row.get("state_sha256") else None,
                loader="token_reconstruction.trr0006_visibility_decoder.load_visibility_state",
                execution="p06_batch8_chunked_full_vocab",
            )
    return methods, manifest_asset, manifest


def _validate_old_registration(registration_path: Path, *, root: Path) -> tuple[dict[str, Method], Asset, dict[str, Any]]:
    registration_asset = _file_record(registration_path)
    registration = _load_json(registration_path, "TRR-0006 prediction registration")
    if registration.get("schema") != OLD_REGISTRATION_SCHEMA or registration.get("task_id") != "TRR-0006" or registration.get("status") != "FROZEN_PREDICTION_REGISTRATION":
        raise ReplayError("TRR-0006 registration is not frozen")
    if registration.get("truth_opened") is not False or registration.get("candidate_arrays_persisted") is not False:
        raise ReplayError("TRR-0006 registration has forbidden flags")
    if registration.get("records_per_domain") != TRR0006_RECORDS or registration.get("cell_order") != list(CELL_ORDER):
        raise ReplayError("TRR-0006 registration geometry or cell order changed")
    embedding = registration.get("runtime_assets", {}).get("normalized_public_E")
    if not isinstance(embedding, Mapping) or embedding.get("sha256") != EMBEDDING_SHA256 or int(embedding.get("bytes", -1)) != EMBEDDING_BYTES:
        raise ReplayError("TRR-0006 embedding binding changed")
    methods_raw = registration.get("methods")
    if not isinstance(methods_raw, Mapping) or list(methods_raw) != list(OLD_METHODS):
        raise ReplayError("TRR-0006 retained method order changed")
    methods: dict[str, Method] = {}
    for method_id in OLD_METHODS:
        row = methods_raw.get(method_id)
        state = row.get("state") if isinstance(row, Mapping) else None
        if not isinstance(state, Mapping):
            raise ReplayError(f"TRR-0006 state binding missing: {method_id}")
        state_path_value = state.get("path")
        if not isinstance(state_path_value, str):
            raise ReplayError(f"TRR-0006 state path missing: {method_id}")
        state_path = Path(state_path_value).expanduser()
        if not state_path.is_absolute():
            state_path = root / state_path
        asset = _file_record(state_path)
        if asset.sha256 != state.get("sha256") or asset.bytes != state.get("bytes"):
            raise ReplayError(f"TRR-0006 state file binding changed: {method_id}")
        base_method = row.get("base_method_id")
        if not isinstance(base_method, str):
            raise ReplayError(f"TRR-0006 base method binding missing: {method_id}")
        methods[method_id] = Method(
            key=method_id,
            family="trr0006",
            method_id=method_id,
            seed=None,
            state_path=asset.path,
            state_asset=asset,
            state_tensor_sha256=None,
            loader="token_reconstruction.trr0005_joint_decoder.load_decoder_state",
            execution="trr0006_native_one_record_full_logits",
            base_method_id=base_method,
        )
    return methods, registration_asset, registration


def select_trr0006_subset(*, records: int, count: int = TRR0006_SUBSET_RECORDS) -> tuple[int, ...]:
    """Return the frozen, correctness-blind subset used by production replay."""

    if records != TRR0006_RECORDS or count != TRR0006_SUBSET_RECORDS:
        raise ReplayError("P07 production subset is fixed at rows 6*k, k=0..255, of each TRR-0006 domain")
    return tuple(range(0, records, records // count))


def _subset_descriptor(old_selection_path: Path, *, root: Path, indices: tuple[int, ...]) -> dict[str, Any]:
    selection_asset = _file_record(old_selection_path)
    selection = _load_json(old_selection_path, "TRR-0006 source selection")
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping) or not isinstance(rule.get("records"), Mapping):
        raise ReplayError("TRR-0006 source selection record ledger is missing")
    ordered_hashes: dict[str, list[str]] = {}
    ordered_sequences: dict[str, list[str]] = {}
    ordered_record_ids: dict[str, list[str]] = {}
    for domain in DOMAINS:
        rows = rule["records"].get(domain)
        if not isinstance(rows, list) or len(rows) != TRR0006_RECORDS:
            raise ReplayError(f"TRR-0006 source selection ledger changed: {domain}")
        selected = [rows[index] for index in indices]
        if any(not isinstance(row, Mapping) for row in selected):
            raise ReplayError(f"TRR-0006 source selection row is malformed: {domain}")
        ordered_hashes[domain] = [str(row.get("public_record_sha256")) for row in selected]
        ordered_sequences[domain] = [str(row.get("final_sequence_sha256")) for row in selected]
        ordered_record_ids[domain] = [str(row.get("record_id")) for row in selected]
        if any(len(value) != 64 for value in ordered_hashes[domain] + ordered_sequences[domain]) or any(not value for value in ordered_record_ids[domain]):
            raise ReplayError(f"TRR-0006 source selection opaque hashes are malformed: {domain}")
    def newline_digest(values: list[str]) -> str:
        return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()
    return {
        "rule": "frozen TRR-0006 source-selection rows at zero-based indices 6*k for k=0..255 per domain, before any truth or correctness access",
        "records_per_domain": TRR0006_SUBSET_RECORDS,
        "row_indices": list(indices),
        "row_indices_sha256": _json_digest(list(indices)),
        "source_selection": {"path": str(selection_asset.path), "bytes": selection_asset.bytes, "sha256": selection_asset.sha256},
        "full_panel_records_per_domain": TRR0006_RECORDS,
        "subset_record_ids_sha256": {domain: newline_digest(values) for domain, values in ordered_record_ids.items()},
        "subset_public_record_sha256": {domain: newline_digest(values) for domain, values in ordered_hashes.items()},
        "subset_final_sequence_sha256": {domain: newline_digest(values) for domain, values in ordered_sequences.items()},
        "full_panel_record_ids_sha256": dict(selection.get("selection_rule", {}).get("record_ids_sha256", {})),
        "source_text_loaded": False,
        "truth_opened": False,
    }


def _validate_plan(plan_path: Path, *, root: Path) -> tuple[Asset, dict[str, Any]]:
    """Bind the exact reviewed subset and matrix plan before loading tensors."""

    asset = _file_record(plan_path)
    if asset.sha256 != PLAN_SHA256:
        raise ReplayError(f"P07 plan hash changed: expected {PLAN_SHA256}, observed {asset.sha256}")
    plan = _load_json(plan_path, "P07 canonical plan")
    if plan.get("schema") != "token-reconstruction.trr-p07-plan.v1" or plan.get("task_id") != TASK_ID or plan.get("parent_commit") != "02c861dfbfc63e3c0b7684a48323fd476a3b268a":
        raise ReplayError("P07 canonical plan identity changed")
    panel = plan.get("panels", {}).get("trr0006_evenly_spaced_1of6")
    if not isinstance(panel, Mapping) or panel.get("records_per_domain") != TRR0006_SUBSET_RECORDS or panel.get("subset_rule") != "for each domain retain published selection rows at zero based indices 6*k for k=0..255, in their original order; no correctness, prediction, truth, or score field is consulted":
        raise ReplayError("P07 deterministic subset plan changed")
    return asset, plan


def _load_embedding(path: Path, *, device: torch.device) -> tuple[torch.Tensor, Asset, dict[str, Any]]:
    asset = _file_record(path)
    if asset.bytes != EMBEDDING_BYTES or asset.sha256 != EMBEDDING_SHA256:
        raise ReplayError("normalized public embedding table binding changed")
    started = time.perf_counter()
    try:
        table_cpu = load_file(str(asset.path), device="cpu")
        if set(table_cpu) != {"embeddings"}:
            raise ReplayError("normalized public embedding table must contain only embeddings")
        table = table_cpu["embeddings"].contiguous()
        if table.dtype != torch.float32 or tuple(table.shape) != (VOCABULARY_SIZE, HIDDEN_SIZE) or not torch.isfinite(table).all().item():
            raise ReplayError("normalized public embedding table geometry, dtype, or finiteness changed")
        result = table.to(device=device, dtype=torch.float32).contiguous()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except ReplayError:
        raise
    except Exception as exc:
        raise ReplayError("normalized public embedding table load failed") from exc
    finally:
        if "table_cpu" in locals():
            del table_cpu
        if "table" in locals():
            del table
        gc.collect()
    evidence = {"path": str(asset.path), "bytes": asset.bytes, "sha256": asset.sha256, "shape": [VOCABULARY_SIZE, HIDDEN_SIZE], "dtype": "torch.float32", "load_seconds": time.perf_counter() - started, "finite_scan": "one immutable load-time scan"}
    return result, asset, evidence


def _load_p06_model(method: Method, *, device: torch.device) -> torch.nn.Module:
    model = load_visibility_state(method.state_path, method_id=method.method_id, hidden_size=HIDDEN_SIZE, vocabulary_size=VOCABULARY_SIZE, context_width=128, expected_sha256=method.state_asset.sha256)
    return model.to(device=device).eval()


def _load_old_model(method: Method, *, device: torch.device) -> torch.nn.Module:
    if method.base_method_id is None:
        raise ReplayError(f"TRR-0006 base method is missing: {method.method_id}")
    model = load_decoder_state(method.state_path, method_id=method.base_method_id, hidden_size=HIDDEN_SIZE, vocabulary_size=VOCABULARY_SIZE, context_width=128)
    return model.to(device=device).eval()


def _validate_model_table(model: torch.nn.Module, embedding: torch.Tensor) -> None:
    if embedding.ndim != 2 or tuple(embedding.shape) != (VOCABULARY_SIZE, HIDDEN_SIZE) or embedding.dtype != torch.float32:
        raise ReplayError("embedding table geometry changed in hot path")
    if int(getattr(model, "hidden_size", -1)) != HIDDEN_SIZE or int(getattr(model, "vocabulary_size", -1)) != VOCABULARY_SIZE:
        raise ReplayError("decoder geometry changed")


@torch.inference_mode()
def predict_p06_batch(model: torch.nn.Module, embedding: torch.Tensor, activations: torch.Tensor, valid_mask: torch.Tensor, *, device: torch.device, projection_chunk: int = P06_PROJECTION_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the published P06 batch-8 projected-row readout boundary."""

    if activations.ndim != 3 or tuple(activations.shape[1:]) != (SEQUENCE_TOKENS, HIDDEN_SIZE) or valid_mask.shape != activations.shape[:2]:
        raise ReplayError("P06 activation batch geometry changed")
    _validate_model_table(model, embedding)
    staged = activations.to(device=device, dtype=torch.float32)
    mask = valid_mask.to(device=device, dtype=torch.bool)
    projected = model.projected_hidden(staged, mask)
    ids = torch.full((activations.shape[0], SEQUENCE_TOKENS), -1, dtype=torch.long, device="cpu")
    ties = torch.zeros((activations.shape[0], SEQUENCE_TOKENS), dtype=torch.int64, device="cpu")
    if not mask[:, 0].all().item():
        raise ReplayError("P06 batch lacks valid BOS")
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
    ids[~active] = -1
    ties[~active] = 0
    ids[:, 0] = BOS_TOKEN_ID
    ties[:, 0] = 1
    return ids, ties


@torch.inference_mode()
def predict_old_native_record(model: torch.nn.Module, embedding: torch.Tensor, activation: torch.Tensor, valid_mask: torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve TRR-0006's native one-record full-logit call."""

    if activation.ndim != 2 or tuple(activation.shape) != (SEQUENCE_TOKENS, HIDDEN_SIZE) or valid_mask.shape != (SEQUENCE_TOKENS,):
        raise ReplayError("TRR-0006 native record geometry changed")
    _validate_model_table(model, embedding)
    staged = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
    mask = valid_mask.to(device=device, dtype=torch.bool).unsqueeze(0)
    logits = model(staged, mask, embedding)
    if logits.ndim != 3 or tuple(logits.shape) != (1, SEQUENCE_TOKENS, VOCABULARY_SIZE) or not torch.isfinite(logits).all().item():
        raise ReplayError("TRR-0006 native decoder returned invalid logits")
    predicted, tie_count = deterministic_top1(logits[0])
    ids = predicted.detach().cpu().to(dtype=torch.long)
    ties = tie_count.detach().cpu().to(dtype=torch.int64)
    active = mask[0].detach().cpu()
    ids[~active] = -1
    ties[~active] = 0
    ids[0] = BOS_TOKEN_ID
    ties[0] = 1
    return ids.contiguous(), ties.contiguous()


def _load_stored_tensor(binding: Mapping[str, Any], *, root: Path, key: str, description: str) -> tuple[torch.Tensor, Asset]:
    asset = _asset_from_binding(binding, root=root, description=description)
    try:
        data = load_file(str(asset.path), device="cpu")
    except Exception as exc:
        raise ReplayError(f"{description} cannot be loaded") from exc
    if set(data) != {key}:
        raise ReplayError(f"{description} tensor key changed")
    return data[key].contiguous(), asset


def _stored_p06_descriptor(manifest: Mapping[str, Any], cell_id: str, seed: int, method_id: str) -> Mapping[str, Any]:
    cell = manifest.get("student_cells", {}).get(cell_id)
    row = cell.get("replicates", {}).get(str(seed), {}).get(method_id) if isinstance(cell, Mapping) else None
    if not isinstance(row, Mapping):
        raise ReplayError(f"stored P06 fixture descriptor missing: {cell_id}/{seed}/{method_id}")
    return row


def _stored_old_descriptor(manifest: Mapping[str, Any], cell_id: str, method_id: str) -> Mapping[str, Any]:
    row = manifest.get("predictions", {}).get(f"{cell_id}::{method_id}")
    if not isinstance(row, Mapping):
        raise ReplayError(f"stored TRR-0006 fixture descriptor missing: {cell_id}/{method_id}")
    return row


def _run_fixtures(*, root: Path, p06_manifest: Mapping[str, Any], old_predictions_manifest: Mapping[str, Any], p06_cells: Mapping[str, Cell], old_cells: Mapping[str, Cell], p06_methods: Mapping[tuple[int, str], Method], old_methods: Mapping[str, Method], embedding: torch.Tensor, device: torch.device, started: float, guard_args: Mapping[str, float]) -> dict[str, Any]:
    """Check all four P06 states and both old states on eight rows per cell.

    The fixture is the pre-matrix qualification gate.  GPU peak reservation is
    sampled while each model and the embedding are still resident, before
    cleanup/``empty_cache`` can erase the relevant evidence.
    """

    fixtures: dict[str, Any] = {"status": "PASS", "records_per_cell": 8, "p06": {}, "trr0006": {}, "guards": [], "truth_opened": False, "gpu_peak_reserved_bytes": None}
    fixture_rows = tuple(range(8))
    for (seed, method_id), method in p06_methods.items():
        model = _load_p06_model(method, device=device)
        peak_reserved: int | None = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            for cell_id in CELL_ORDER:
                cell = p06_cells[cell_id]
                activations, mask, _, _ = _observation_rows(cell=cell, batch_start=0, batch_stop=8)
                actual_ids, actual_ties = predict_p06_batch(model, embedding, activations, mask, device=device, projection_chunk=P06_PROJECTION_CHUNK)
                descriptor = _stored_p06_descriptor(p06_manifest, cell_id, seed, method_id)
                stored_ids, pred_asset = _load_stored_tensor(descriptor["prediction"], root=root, key="predictions", description=f"stored P06 predictions {cell_id}/{seed}/{method_id}")
                stored_ties, tie_asset = _load_stored_tensor(descriptor["tie_counts"], root=root, key="tie_counts", description=f"stored P06 ties {cell_id}/{seed}/{method_id}")
                if not torch.equal(actual_ids, stored_ids[:8]) or not torch.equal(actual_ties, stored_ties[:8]):
                    raise ReplayError(f"P06 fixture output mismatch: {cell_id}/{seed}/{method_id}")
                fixtures["p06"][f"{cell_id}::{seed}::{method_id}"] = {"status": "PASS", "prediction_asset": {"path": str(pred_asset.path), "bytes": pred_asset.bytes, "sha256": pred_asset.sha256}, "tie_asset": {"path": str(tie_asset.path), "bytes": tie_asset.bytes, "sha256": tie_asset.sha256}, "prediction_tensor_sha256": tensor_digest(actual_ids), "tie_tensor_sha256": tensor_digest(actual_ties), "row_indices": list(fixture_rows)}
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        guard = _guard(device=device, started=started, stage=f"after_fixture_{seed}_{method_id}", **guard_args)
        if peak_reserved is not None:
            guard["gpu_peak_reserved_bytes"] = peak_reserved
            fixtures["gpu_peak_reserved_bytes"] = max(int(fixtures["gpu_peak_reserved_bytes"] or 0), peak_reserved)
        fixtures["guards"].append(guard)
    for method_id, method in old_methods.items():
        model = _load_old_model(method, device=device)
        peak_reserved = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            for cell_id in CELL_ORDER:
                cell = old_cells[cell_id]
                activations, mask, _, _ = _observation_rows(cell=cell, batch_start=0, batch_stop=8)
                ids: list[torch.Tensor] = []
                ties: list[torch.Tensor] = []
                for row in range(8):
                    row_ids, row_ties = predict_old_native_record(model, embedding, activations[row], mask[row], device=device)
                    ids.append(row_ids)
                    ties.append(row_ties)
                actual_ids = torch.stack(ids)
                actual_ties = torch.stack(ties)
                descriptor = _stored_old_descriptor(old_predictions_manifest, cell_id, method_id)
                stored_ids, pred_asset = _load_stored_tensor(descriptor["prediction_artifact"], root=root, key="predictions", description=f"stored TRR-0006 predictions {cell_id}/{method_id}")
                if not torch.equal(actual_ids, stored_ids[:8]):
                    raise ReplayError(f"TRR-0006 native fixture output mismatch: {cell_id}/{method_id}")
                fixtures["trr0006"][f"{cell_id}::{method_id}"] = {"status": "PASS", "prediction_asset": {"path": str(pred_asset.path), "bytes": pred_asset.bytes, "sha256": pred_asset.sha256}, "tie_counts_persisted": False, "tie_counts_observed": {"max": int(actual_ties.max().item()), "tied_active_positions": int((actual_ties[:, 1:] > 1).sum().item())}, "prediction_tensor_sha256": tensor_digest(actual_ids), "row_indices": list(fixture_rows)}
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        guard = _guard(device=device, started=started, stage=f"after_fixture_{method_id}", **guard_args)
        if peak_reserved is not None:
            guard["gpu_peak_reserved_bytes"] = peak_reserved
            fixtures["gpu_peak_reserved_bytes"] = max(int(fixtures["gpu_peak_reserved_bytes"] or 0), peak_reserved)
        fixtures["guards"].append(guard)
    return fixtures


def _timed_p06_cell(model: torch.nn.Module, embedding: torch.Tensor, cell: Cell, *, device: torch.device, started: float, guard_args: Mapping[str, float], projection_chunk: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    load_started = time.perf_counter()
    selected_indices = cell.subset_indices
    selected_records = len(selected_indices)
    activations, mask, positions, sidecars = _observation_rows(cell=cell, indices=selected_indices)
    load_seconds = time.perf_counter() - load_started
    if selected_records % P06_BATCH_RECORDS:
        raise ReplayError("P06 record count is not divisible by batch size 8")

    def one_pass() -> tuple[torch.Tensor, torch.Tensor, float]:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pass_started = time.perf_counter()
        all_ids = torch.empty((selected_records, SEQUENCE_TOKENS), dtype=torch.long)
        all_ties = torch.empty((selected_records, SEQUENCE_TOKENS), dtype=torch.int64)
        for row_start in range(0, selected_records, P06_BATCH_RECORDS):
            row_stop = row_start + P06_BATCH_RECORDS
            ids, ties = predict_p06_batch(model, embedding, activations[row_start:row_stop], mask[row_start:row_stop], device=device, projection_chunk=projection_chunk)
            all_ids[row_start:row_stop] = ids
            all_ties[row_start:row_stop] = ties
            _guard(device=device, started=started, stage=f"after_{cell.panel}_{cell.cell_id}_{row_stop}", **guard_args)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return all_ids, all_ties, time.perf_counter() - pass_started

    warm_ids, warm_ties, warm_seconds = one_pass()
    measured: list[float] = []
    selected_ids: torch.Tensor | None = None
    selected_ties: torch.Tensor | None = None
    for _ in range(P06_MEASURED_PASSES):
        ids, ties, elapsed = one_pass()
        if selected_ids is None:
            selected_ids, selected_ties = ids, ties
        elif not torch.equal(selected_ids, ids) or not torch.equal(selected_ties, ties):
            raise ReplayError(f"P06 repeated output mismatch: {cell.cell_id}")
        measured.append(elapsed)
    assert selected_ids is not None and selected_ties is not None
    if not torch.equal(warm_ids, selected_ids) or not torch.equal(warm_ties, selected_ties):
        raise ReplayError(f"P06 warmup/measured output mismatch: {cell.cell_id}")
    timing = {"execution": "p06_batch8_chunked_full_vocab", "selected_row_indices": sidecars["row_indices"], "records": selected_records, "batch_records": P06_BATCH_RECORDS, "projection_chunk": projection_chunk, "warmup_passes": P06_WARMUP_PASSES, "measured_passes": P06_MEASURED_PASSES, "warmup_seconds": warm_seconds, "measured_seconds": measured, "measured_mean_seconds": sum(measured) / len(measured), "measured_ms_per_record": 1000 * (sum(measured) / len(measured)) / selected_records, "observation_load_seconds": load_seconds, "observation_load_excluded_from_measured_interval": True, "measurement_includes_resource_guard_and_device_synchronization": True, "repeat_prediction_exact": True, "attention_mask_sha256": sidecars["attention_mask_sha256"], "position_ids_sha256": sidecars["position_ids_sha256"], "positions_digest": tensor_digest(positions), "truth_opened": False}
    return selected_ids, selected_ties, timing


def _timed_old_cell(model: torch.nn.Module, embedding: torch.Tensor, cell: Cell, *, device: torch.device, started: float, guard_args: Mapping[str, float]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    load_started = time.perf_counter()
    selected_indices = cell.subset_indices
    selected_records = len(selected_indices)
    activations, mask, positions, sidecars = _observation_rows(cell=cell, indices=selected_indices)
    load_seconds = time.perf_counter() - load_started
    warm_seconds: list[float] = []
    measured_seconds: list[float] = []
    ids_rows: list[torch.Tensor] = []
    ties_rows: list[torch.Tensor] = []
    for row in range(selected_records):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        warm_ids, warm_ties = predict_old_native_record(model, embedding, activations[row], mask[row], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        warm_seconds.append(time.perf_counter() - start)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        ids, ties = predict_old_native_record(model, embedding, activations[row], mask[row], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        measured_seconds.append(time.perf_counter() - start)
        if not torch.equal(warm_ids, ids) or not torch.equal(warm_ties, ties):
            raise ReplayError(f"TRR-0006 warmup/measured output mismatch: {cell.cell_id}/{row}")
        ids_rows.append(ids)
        ties_rows.append(ties)
        _guard(device=device, started=started, stage=f"after_{cell.panel}_{cell.cell_id}_{row + 1}", **guard_args)
    ids = torch.stack(ids_rows)
    ties = torch.stack(ties_rows)
    timing = {"execution": "trr0006_native_one_record_full_logits", "selected_row_indices": sidecars["row_indices"], "records": selected_records, "batch_records": 1, "warmup_runs_per_record": OLD_WARMUP_RUNS_PER_RECORD, "measured_runs_per_record": OLD_MEASURED_RUNS_PER_RECORD, "warmup_seconds_sum": sum(warm_seconds), "measured_seconds_sum": sum(measured_seconds), "measured_ms_per_record": 1000 * sum(measured_seconds) / selected_records, "observation_load_seconds": load_seconds, "observation_load_excluded_from_measured_interval": True, "measurement_includes_device_synchronization": True, "repeat_prediction_exact": True, "per_record_measured_seconds": measured_seconds, "attention_mask_sha256": sidecars["attention_mask_sha256"], "position_ids_sha256": sidecars["position_ids_sha256"], "positions_digest": tensor_digest(positions), "truth_opened": False}
    return ids, ties, timing


def _prediction_path(output_root: Path, panel: str, cell_id: str, method: Method) -> Path:
    domain, target = cell_id.split("__", 1)
    seed_dir = f"seed-{method.seed}" if method.seed is not None else "retained"
    return output_root / "predictions" / panel / domain / target / seed_dir / f"{method.key}.safetensors"


def _tie_path(output_root: Path, panel: str, cell_id: str, method: Method) -> Path:
    domain, target = cell_id.split("__", 1)
    seed_dir = f"seed-{method.seed}" if method.seed is not None else "retained"
    return output_root / "tie_counts" / panel / domain / target / seed_dir / f"{method.key}.safetensors"


def _asset_json(asset: Asset) -> dict[str, Any]:
    return {"path": str(asset.path), "bytes": asset.bytes, "sha256": asset.sha256}


def _save_prediction_artifacts(*, output_root: Path, panel: str, cell: Cell, method: Method, ids: torch.Tensor, ties: torch.Tensor, timing: Mapping[str, Any], root: Path) -> dict[str, Any]:
    prediction_path = _prediction_path(output_root, panel, cell.cell_id, method)
    ties_path = _tie_path(output_root, panel, cell.cell_id, method)
    if prediction_path.exists() or ties_path.exists() or prediction_path.is_symlink() or ties_path.is_symlink():
        raise ReplayError(f"prediction artifact is not create-only: {panel}/{cell.cell_id}/{method.key}")
    metadata = {"schema": PREDICTION_SCHEMA, "task_id": TASK_ID, "panel": panel, "cell_id": cell.cell_id, "method_key": method.key, "method_id": method.method_id, "seed": "" if method.seed is None else str(method.seed), "records": str(cell.records if panel == "p06_panel" else TRR0006_SUBSET_RECORDS), "sequence_tokens": str(SEQUENCE_TOKENS), "scored_post_bos_tokens": str(SCORED_POST_BOS), "observation_sha256": cell.asset.sha256, "state_sha256": method.state_asset.sha256, "execution": method.execution, "truth_opened": "false", "source_text_loaded": "false", "target_labels_loaded": "false", "candidate_arrays_persisted": "false"}
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    ties_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"predictions": ids.contiguous()}, str(prediction_path), metadata=metadata)
    save_file({"tie_counts": ties.contiguous()}, str(ties_path), metadata={**metadata, "schema": TIE_SCHEMA})
    prediction_asset = _file_record(prediction_path)
    tie_asset = _file_record(ties_path)
    return {"schema": PREDICTION_SCHEMA, "task_id": TASK_ID, "panel": panel, "cell_id": cell.cell_id, "method_key": method.key, "method_id": method.method_id, "seed": method.seed, "records": int(ids.shape[0]), "shape": list(ids.shape), "scored_post_bos_tokens": SCORED_POST_BOS, "observation": _asset_json(cell.asset), "record_ids_sha256": cell.record_ids_sha256, "state": _asset_json(method.state_asset), "prediction": _asset_json(prediction_asset), "tie_counts": _asset_json(tie_asset), "prediction_tensor_sha256": tensor_digest(ids), "tie_counts_tensor_sha256": tensor_digest(ties), "timing": dict(timing), "tie_summary": {"max_tie_count": int(ties[:, 1:].max().item()) if ties.shape[1] > 1 else 1, "tied_active_positions": int((ties[:, 1:] > 1).sum().item()) if ties.shape[1] > 1 else 0}, "truth_opened": False, "source_text_loaded": False, "target_labels_loaded": False, "candidate_arrays_persisted": False}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(root / "experiments" / "TRR-P07")
    except ValueError as exc:
        raise ReplayError("P07 output must be task-owned under experiments/TRR-P07") from exc
    if output_root.exists() or output_root.is_symlink():
        raise ReplayError(f"P07 output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    started_utc = _utc_now()
    failure_path = output_root / "failure.json"
    try:
        runtime = _configure_threads(args.torch_threads, args.torch_interop_threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ReplayError("CUDA device requested but unavailable")
        device = torch.device(args.device)
        p06_manifest_path = Path(args.p06_prediction_manifest).expanduser().resolve()
        p06_obs_path = Path(args.p06_observation_manifest).expanduser().resolve()
        old_registration_path = Path(args.trr0006_registration).expanduser().resolve()
        old_obs_path = Path(args.trr0006_observation_manifest).expanduser().resolve()
        old_predictions_path = Path(args.trr0006_predictions).expanduser().resolve()
        old_selection_path = Path(args.trr0006_source_selection).expanduser().resolve()
        plan_asset, plan = _validate_plan(Path(args.plan).expanduser().resolve(), root=root)
        p06_methods, p06_manifest_asset, p06_manifest = _validate_p06_states(p06_manifest_path, root=root)
        old_methods, old_registration_asset, old_registration = _validate_old_registration(old_registration_path, root=root)
        p06_cells, p06_obs_asset, p06_obs_manifest = _manifest_cells(p06_obs_path, root=root, panel="p06_panel")
        old_cells_full, old_obs_asset, old_obs_manifest = _manifest_cells(old_obs_path, root=root, panel="trr0006_subset")
        subset_indices = select_trr0006_subset(records=TRR0006_RECORDS)
        old_cells = {key: Cell(**{**cell.__dict__, "subset_indices": subset_indices}) for key, cell in old_cells_full.items()}
        subset_descriptor = _subset_descriptor(old_selection_path, root=root, indices=subset_indices)
        planned_subset = plan["panels"]["trr0006_evenly_spaced_1of6"]
        for field in ("subset_record_ids_sha256", "subset_public_record_sha256", "subset_final_sequence_sha256"):
            if planned_subset.get(field) != subset_descriptor.get(field):
                raise ReplayError(f"P07 subset binding changed: {field}")
        old_predictions = _load_json(old_predictions_path, "TRR-0006 frozen prediction descriptor")
        if old_predictions.get("schema") != OLD_PREDICTION_DESCRIPTOR_SCHEMA or old_predictions.get("status") != "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH" or old_predictions.get("truth_opened") is not False:
            raise ReplayError("TRR-0006 frozen prediction descriptor is not source-free")
        embedding_binding = old_registration.get("runtime_assets", {}).get("normalized_public_E")
        if not isinstance(embedding_binding, Mapping):
            raise ReplayError("TRR-0006 embedding asset binding is missing")
        embedding_path = Path(str(embedding_binding["path"])).expanduser()
        embedding, embedding_asset, embedding_evidence = _load_embedding(embedding_path, device=device)
        guards: list[dict[str, Any]] = []
        guard_args = {"max_seconds": float(args.max_seconds), "min_free_gib": float(args.minimum_free_gib), "max_reserved_gib": float(args.maximum_gpu_reserved_gib), "max_rss_gib": float(args.maximum_host_rss_gib), "min_host_gib": float(args.minimum_host_available_gib)}
        guards.append(_guard(device=device, started=started, stage="after_embedding_load", **guard_args))
        fixtures = _run_fixtures(root=root, p06_manifest=p06_manifest, old_predictions_manifest=old_predictions, p06_cells=p06_cells, old_cells=old_cells_full, p06_methods=p06_methods, old_methods=old_methods, embedding=embedding, device=device, started=started, guard_args=guard_args)
        all_methods: list[Method] = []
        for method_id in P06_METHODS:
            for seed in SEEDS:
                all_methods.append(p06_methods[(seed, method_id)])
        all_methods.extend(old_methods[method_id] for method_id in OLD_METHODS)
        if getattr(args, "qualification_only", False):
            target = str(getattr(args, "qualification_cell", "p06_panel::pile__public_base::p06_past_only__seed6106"))
            target_parts = target.split("::")
            if len(target_parts) != 3 or target_parts[0] not in {"p06_panel", "trr0006_subset"}:
                raise ReplayError("qualification cell must be PANEL::CELL_ID::METHOD_KEY")
            qualification_panel, qualification_cell_id, qualification_method_key = target_parts
            cells_for_panel = p06_cells if qualification_panel == "p06_panel" else old_cells
            if qualification_cell_id not in cells_for_panel:
                raise ReplayError(f"qualification cell is not in the frozen cell set: {qualification_cell_id}")
            method_by_key = {method.key: method for method in all_methods}
            qualification_method = method_by_key.get(qualification_method_key)
            if qualification_method is None:
                raise ReplayError(f"qualification method is not in the frozen method set: {qualification_method_key}")
            if (qualification_panel == "p06_panel") != (qualification_method.family == "p06"):
                raise ReplayError("qualification panel and method execution family do not match")
            qualification_cell = cells_for_panel[qualification_cell_id]
            qualification_peak_reserved: int | None = None
            qualification_model = _load_p06_model(qualification_method, device=device) if qualification_method.family == "p06" else _load_old_model(qualification_method, device=device)
            qualification_model.requires_grad_(False)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            guards.append(_guard(device=device, started=started, stage="after_qualification_model_load", **guard_args))
            try:
                if qualification_method.family == "p06":
                    qualification_ids, qualification_ties, qualification_timing = _timed_p06_cell(qualification_model, embedding, qualification_cell, device=device, started=started, guard_args=guard_args, projection_chunk=P06_PROJECTION_CHUNK)
                else:
                    qualification_ids, qualification_ties, qualification_timing = _timed_old_cell(qualification_model, embedding, qualification_cell, device=device, started=started, guard_args=guard_args)
                qualification_descriptor = _save_prediction_artifacts(output_root=output_root, panel=qualification_panel, cell=qualification_cell, method=qualification_method, ids=qualification_ids, ties=qualification_ties, timing=qualification_timing, root=root)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    qualification_peak_reserved = int(torch.cuda.max_memory_reserved(device))
            finally:
                del qualification_model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            qualification_guard = _guard(device=device, started=started, stage="after_qualification_cell", **guard_args)
            if qualification_peak_reserved is not None:
                qualification_guard["gpu_peak_reserved_bytes"] = qualification_peak_reserved
            guards.append(qualification_guard)
            qualification = {
                "schema": "token-reconstruction.trr-p07-fixture-qualification.v2",
                "task_id": TASK_ID,
                "status": "P07_FIXTURE_AND_CELL_QUALIFICATION_PASS_NO_TRUTH",
                "truth_opened": False,
                "source_text_loaded": False,
                "target_labels_loaded": False,
                "candidate_arrays_persisted": False,
                "code_commit": _git_head(root),
                "parent_commit": "02c861dfbfc63e3c0b7684a48323fd476a3b268a",
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "runtime": runtime,
                "device": str(device),
                "geometry": {"sequence_tokens": SEQUENCE_TOKENS, "scored_post_bos_tokens": SCORED_POST_BOS, "hidden_size": HIDDEN_SIZE, "vocabulary_size": VOCABULARY_SIZE, "embedding": _asset_json(embedding_asset), "embedding_load": embedding_evidence},
                "source_bindings": {"canonical_plan": _asset_json(plan_asset), "p06_prediction_manifest": _asset_json(p06_manifest_asset), "trr0006_registration": _asset_json(old_registration_asset), "trr0006_predictions": _asset_json(_file_record(old_predictions_path)), "trr0006_source_selection": _asset_json(_file_record(old_selection_path))},
                "method_order": [method.key for method in all_methods],
                "fixture_gate": fixtures,
                "qualification_target": target,
                "qualification_prediction": qualification_descriptor,
                "qualification_timing": qualification_timing,
                "resource_guards": guards + list(fixtures["guards"]),
                "gpu_peak_reserved_bytes": qualification_peak_reserved,
                "qualification_gate": "All published P06 batch-8 and TRR-0006 native eight-row fixtures pass, followed by one complete selected 256-record representative cell, before the full frozen matrix.",
            }
            _write_create_only(output_root / "qualification_manifest.json", qualification)
            qualification_asset = _file_record(output_root / "qualification_manifest.json")
            run_manifest = {"schema": RUN_SCHEMA, "task_id": TASK_ID, "status": "P07_FIXTURE_AND_CELL_QUALIFICATION_COMPLETE_NO_TRUTH", "started_utc": started_utc, "ended_utc": _utc_now(), "elapsed_seconds": time.perf_counter() - started, "qualification_manifest": _asset_json(qualification_asset), "code_commit": qualification["code_commit"], "prediction_count": 1, "prediction_complete": False, "fixture_status": fixtures["status"], "qualification_target": target, "truth_opened": False, "source_text_loaded": False, "target_labels_loaded": False, "candidate_arrays_persisted": False}
            _write_create_only(output_root / "run_manifest.json", run_manifest)
            return run_manifest
        # Fixed method order is part of the cost receipt.  It is independent of
        # any truth/correctness result and therefore cannot select an arm.
        predictions: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        for method in all_methods:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model = _load_p06_model(method, device=device) if method.family == "p06" else _load_old_model(method, device=device)
            model.requires_grad_(False)
            guards.append(_guard(device=device, started=started, stage=f"after_{method.key}_load", **guard_args))
            try:
                panel_items = (("p06_panel", p06_cells), ("trr0006_subset", old_cells))
                for panel, cells in panel_items:
                    for cell_id in CELL_ORDER:
                        cell = cells[cell_id]
                        if method.family == "p06":
                            ids, ties, timing = _timed_p06_cell(model, embedding, cell, device=device, started=started, guard_args=guard_args, projection_chunk=P06_PROJECTION_CHUNK)
                        else:
                            ids, ties, timing = _timed_old_cell(model, embedding, cell, device=device, started=started, guard_args=guard_args)
                        descriptor = _save_prediction_artifacts(output_root=output_root, panel=panel, cell=cell, method=method, ids=ids, ties=ties, timing=timing, root=root)
                        predictions[f"{panel}::{cell_id}::{method.key}"] = descriptor
                        timings[f"{panel}::{cell_id}::{method.key}"] = timing
                        guards.append(_guard(device=device, started=started, stage=f"after_{panel}_{cell_id}_{method.key}", **guard_args))
            finally:
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        expected_count = len(CELL_ORDER) * (len(P06_METHODS) * len(SEEDS) + len(OLD_METHODS)) * 2
        if len(predictions) != expected_count:
            raise ReplayError(f"prediction matrix incomplete: expected {expected_count}, got {len(predictions)}")
        manifest = {"schema": SCHEMA, "task_id": TASK_ID, "status": "FROZEN_P07_PREDICTIONS_NO_TRUTH", "truth_opened": False, "source_text_loaded": False, "target_labels_loaded": False, "candidate_arrays_persisted": False, "code_commit": _git_head(root), "parent_commit": "02c861dfbfc63e3c0b7684a48323fd476a3b268a", "started_utc": started_utc, "ended_utc": _utc_now(), "elapsed_seconds": time.perf_counter() - started, "runtime": runtime, "device": str(device), "geometry": {"sequence_tokens": SEQUENCE_TOKENS, "scored_post_bos_tokens": SCORED_POST_BOS, "hidden_size": HIDDEN_SIZE, "vocabulary_size": VOCABULARY_SIZE, "embedding": _asset_json(embedding_asset), "embedding_load": embedding_evidence}, "panels": {"p06_panel": {"observation_manifest": _asset_json(p06_obs_asset), "records_per_domain": P06_RECORDS, "subset_rule": "all 256 published P06 rows per domain"}, "trr0006_subset": {"observation_manifest": _asset_json(old_obs_asset), "records_per_domain": TRR0006_SUBSET_RECORDS, "full_records_per_domain": TRR0006_RECORDS, "subset": subset_descriptor}}, "source_bindings": {"canonical_plan": _asset_json(plan_asset), "p06_prediction_manifest": _asset_json(p06_manifest_asset), "trr0006_registration": _asset_json(old_registration_asset), "trr0006_predictions": _asset_json(_file_record(old_predictions_path)), "trr0006_source_selection": _asset_json(_file_record(old_selection_path))}, "method_order": [method.key for method in all_methods], "timing_comparability": {"status": "DESCRIPTIVE_NOT_USED_FOR_DECISION", "method_order": "fixed source-order for reproducibility", "p06": "batch-8 chunked full-vocabulary throughput", "trr0006": "native one-record full-logit latency", "cross_path_pooling": False}, "methods": [{"key": method.key, "family": method.family, "method_id": method.method_id, "seed": method.seed, "state": _asset_json(method.state_asset), "state_tensor_sha256": method.state_tensor_sha256, "loader": method.loader, "execution": method.execution} for method in all_methods], "fixtures": fixtures, "predictions": predictions, "timings": timings, "resource_guards": guards, "prediction_count": len(predictions), "prediction_freeze": "all P07 predictions and tie counts are create-only and frozen before any truth/score reader", "truth_gate": "No truth, labels, source text, or token IDs were opened by this runner."}
        _write_create_only(output_root / "replay_manifest.json", manifest)
        run_manifest = {"schema": RUN_SCHEMA, "task_id": TASK_ID, "status": "P07_FROZEN_REPLAY_COMPLETE_NO_TRUTH", "started_utc": started_utc, "ended_utc": _utc_now(), "elapsed_seconds": time.perf_counter() - started, "replay_manifest": _asset_json(_file_record(output_root / "replay_manifest.json")), "code_commit": manifest["code_commit"], "prediction_count": len(predictions), "prediction_complete": True, "fixture_status": fixtures["status"], "truth_opened": False, "source_text_loaded": False, "target_labels_loaded": False, "candidate_arrays_persisted": False}
        _write_create_only(output_root / "run_manifest.json", run_manifest)
        return run_manifest
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_create_only(failure_path, {"schema": FAILURE_SCHEMA, "task_id": TASK_ID, "status": "FAILED_PRESERVED_NO_TRUTH", "started_utc": started_utc, "ended_utc": _utc_now(), "error_type": type(exc).__name__, "error": str(exc), "truth_opened": False, "source_text_loaded": False, "target_labels_loaded": False, "candidate_arrays_persisted": False})
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--p06-prediction-manifest", type=Path, required=True)
    parser.add_argument("--p06-observation-manifest", type=Path, required=True)
    parser.add_argument("--trr0006-registration", type=Path, required=True)
    parser.add_argument("--trr0006-observation-manifest", type=Path, required=True)
    parser.add_argument("--trr0006-predictions", type=Path, required=True)
    parser.add_argument("--trr0006-source-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qualification-only", action="store_true", help="run the source-bound fixtures and one complete representative cell, then exit before the full prediction matrix")
    parser.add_argument("--qualification-cell", default="p06_panel::pile__public_base::p06_past_only__seed6106", help="PANEL::CELL_ID::METHOD_KEY used with --qualification-only")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=8.0)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=16.0)
    parser.add_argument("--minimum-host-available-gib", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run(args)
    except (ReplayError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-P07 replay error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

