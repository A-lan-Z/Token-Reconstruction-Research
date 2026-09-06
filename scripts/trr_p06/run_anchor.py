#!/usr/bin/env python3
"""Run the bounded published-parent A1+A2 anchor for TRR-P06.

The anchor is a CPU-embedding port of the retained ``frozen_a1_a2_k256``
rule.  It consumes only the first 64 public-base activation rows per domain
from the P06 observation manifest.  The parent implementation still owns the
proposal and public-prefix simulation: A1 proposes the fixed top-512 list and
the A2 selector evaluates the first 256 candidates with direct cosine scores.
Candidate arrays and any source or truth payload are omitted from the output.

"CPU embedding port" identifies the inherited parent boundary precisely: the
BF16 public model embedding is normalized on CPU and compared with the pinned
normalized public table before the table is copied to the execution device.
The public four-layer prefix and the fixed selector execute on CUDA, as in the
published parent qualification.  This file is an execution adapter, not a
new A2 implementation or a student fallback.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


TASK_ID = "TRR-P06"
ANCHOR_METHOD_ID = "frozen_a1_a2_k256"
ANCHOR_STATE_SHA256 = "33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742"
REFERENCE_SHA256 = "10532a746cb8c30eb2caf338e206e1fa9d85e708d4db43a0d8fd4a2ff1a6f8bd"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
EMBEDDING_BYTES = 1050673488
ANCHOR_SUBSET = "first64_public_base"
ANCHOR_SCHEMA = "token-reconstruction.trr-p06-anchor-prediction-manifest.v1"
RUN_SCHEMA = "token-reconstruction.trr-p06-anchor-run.v1"
FAILURE_SCHEMA = "token-reconstruction.trr-p06-anchor-failure.v1"
DOMAINS = ("pile", "finance")
RECORDS_PER_DOMAIN = 256
ANCHOR_RECORDS_PER_DOMAIN = 64
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
PROPOSAL_K = 512
SELECTOR_K = 256
OBSERVATION_SCHEMA = "token-reconstruction.trr-p06-public-observation-manifest.v1"
OBSERVATION_FILE_SCHEMA = "token-reconstruction.trr-p06-public-observation.v1"


class AnchorError(RuntimeError):
    """Raised when the P06 anchor contract fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise AnchorError(f"file is unavailable: {path}")
    name = str(path)
    if root is not None:
        try:
            name = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {"path": name, "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise AnchorError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnchorError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise AnchorError(f"{description} must be an object")
    return dict(value)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise AnchorError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnchorError("unable to resolve executable source commit") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise AnchorError("executable source commit is not a full lowercase hash")
    return value


def _resolve_file(value: Any, *, root: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AnchorError(f"{description} path is malformed")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise AnchorError(f"{description} is unavailable: {path}")
    return path


def _load_selection(path: Path, *, root: Path) -> dict[str, Any]:
    selection = _load_json(path, description="P06 source selection")
    if selection.get("task_id") != TASK_ID or selection.get("schema") != "token-reconstruction.trr-p06-source-selection.v1":
        raise AnchorError("source selection schema or task ID changed")
    rows_by_domain = selection.get("selection_rule", {}).get("records")
    if not isinstance(rows_by_domain, Mapping) or set(rows_by_domain) != set(DOMAINS):
        raise AnchorError("source selection lacks both domain record lists")
    result: dict[str, Any] = {"file": _file_record(path, root=root), "record_ids": {}, "sequence_hashes": {}, "subset_ids": {}, "subset_sequence_hashes": {}}
    for domain in DOMAINS:
        rows = rows_by_domain[domain]
        if not isinstance(rows, list) or len(rows) != RECORDS_PER_DOMAIN:
            raise AnchorError(f"source selection record count changed: {domain}")
        ids: list[str] = []
        sequence_hashes: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str) or not row.get("record_id"):
                raise AnchorError(f"source selection record is malformed: {domain}/{index}")
            sequence_sha = row.get("final_sequence_sha256")
            if not isinstance(sequence_sha, str) or len(sequence_sha) != 64:
                raise AnchorError(f"source selection sequence hash is malformed: {domain}/{index}")
            ids.append(str(row["record_id"]))
            sequence_hashes.append(sequence_sha)
        if len(set(ids)) != RECORDS_PER_DOMAIN or len(set(sequence_hashes)) != RECORDS_PER_DOMAIN:
            raise AnchorError(f"source selection rows are not unique: {domain}")
        subset_ids = ids[:ANCHOR_RECORDS_PER_DOMAIN]
        result["record_ids"][domain] = ids
        result["sequence_hashes"][domain] = sequence_hashes
        result["subset_ids"][domain] = subset_ids
        result["subset_sequence_hashes"][domain] = sequence_hashes[:ANCHOR_RECORDS_PER_DOMAIN]
    result["record_ids_sha256"] = {domain: _json_digest(result["record_ids"][domain]) for domain in DOMAINS}
    result["subset_record_ids_sha256"] = {domain: _json_digest(result["subset_ids"][domain]) for domain in DOMAINS}
    return result


def _observation_descriptor(manifest: Mapping[str, Any], *, cell_id: str) -> Mapping[str, Any]:
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise AnchorError("observation manifest cells are not an ordered list")
    for cell in cells:
        if isinstance(cell, Mapping) and cell.get("cell_id") == cell_id:
            observation = cell.get("observation")
            if not isinstance(observation, Mapping):
                raise AnchorError(f"observation descriptor is missing: {cell_id}")
            return observation
    raise AnchorError(f"public-base observation cell is missing: {cell_id}")


def _load_observation(manifest_path: Path, manifest: Mapping[str, Any], *, domain: str, root: Path, expected_ids_sha: str) -> dict[str, Any]:
    if manifest.get("schema") != OBSERVATION_SCHEMA or manifest.get("task_id") != TASK_ID or manifest.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise AnchorError("observation manifest is not the frozen no-truth schema")
    descriptor = _observation_descriptor(manifest, cell_id=f"{domain}__public_base")
    if descriptor.get("record_ids_sha256") != expected_ids_sha:
        raise AnchorError(f"public-base observation source order changed: {domain}")
    observation_path = _resolve_file(descriptor.get("path"), root=root, description=f"observation {domain}")
    declared = descriptor.get("sha256")
    actual = _file_record(observation_path, root=root)
    if actual["sha256"] != declared or actual["bytes"] != descriptor.get("bytes"):
        raise AnchorError(f"observation file binding changed: {domain}")
    load_started = time.perf_counter()
    try:
        with safe_open(str(observation_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise AnchorError(f"observation tensor set changed: {domain}")
            activations = handle.get_tensor("activations")
            mask = handle.get_tensor("attention_mask")
            positions = handle.get_tensor("position_ids")
    except AnchorError:
        raise
    except Exception as exc:
        raise AnchorError(f"observation tensors are unreadable: {domain}") from exc
    if tuple(activations.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, HIDDEN_SIZE) or activations.dtype != torch.bfloat16:
        raise AnchorError(f"observation activation geometry changed: {domain}")
    if tuple(mask.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or mask.dtype not in (torch.uint8, torch.bool, torch.int8):
        raise AnchorError(f"observation mask geometry changed: {domain}")
    if tuple(positions.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS) or positions.dtype not in (torch.int64, torch.int32):
        raise AnchorError(f"observation position geometry changed: {domain}")
    mask = mask.to(torch.bool)
    positions = positions.to(torch.int64)
    if not torch.isfinite(activations.float()).all().item() or not mask[:, :SEQUENCE_TOKENS].all().item():
        raise AnchorError(f"anchor requires 128 fully valid positions: {domain}")
    expected_positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.int64).view(1, -1).expand(RECORDS_PER_DOMAIN, -1)
    if not torch.equal(positions, expected_positions):
        raise AnchorError(f"observation position IDs changed: {domain}")
    return {
        "path": observation_path,
        "file": actual,
        "activations": activations[:ANCHOR_RECORDS_PER_DOMAIN].contiguous(),
        "mask": mask[:ANCHOR_RECORDS_PER_DOMAIN].contiguous(),
        "positions": positions[:ANCHOR_RECORDS_PER_DOMAIN].contiguous(),
        "attention_mask_sha256": _tensor_digest(mask[:ANCHOR_RECORDS_PER_DOMAIN].to(torch.uint8)),
        "position_ids_sha256": _tensor_digest(positions[:ANCHOR_RECORDS_PER_DOMAIN]),
        "load_seconds": time.perf_counter() - load_started,
    }


def _normalize_prediction(prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(prediction, dtype=torch.long).detach().cpu().contiguous()
    valid = torch.as_tensor(mask, dtype=torch.bool).detach().cpu().contiguous()
    if tuple(values.shape) != (SEQUENCE_TOKENS,) or tuple(valid.shape) != (SEQUENCE_TOKENS,):
        raise AnchorError("anchor prediction row geometry changed")
    if not bool(valid[0].item()):
        raise AnchorError("anchor observation row has no valid BOS")
    output = torch.full((SEQUENCE_TOKENS,), INVALID_TOKEN_ID, dtype=torch.long)
    output[valid] = values[valid]
    output[0] = BOS_TOKEN_ID
    active = output[valid]
    if active.lt(0).any().item() or active.ge(VOCABULARY_SIZE).any().item():
        raise AnchorError("anchor emitted an invalid active token")
    if not output[~valid].eq(INVALID_TOKEN_ID).all().item():
        raise AnchorError("anchor did not preserve padding")
    return output


def _guard(device: torch.device, *, started: float, args: argparse.Namespace, stage: str) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    if elapsed > args.max_seconds:
        raise AnchorError(f"wall-time guard failed at {stage}: {elapsed:.3f}s")
    rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if rss_bytes > int(args.maximum_host_rss_gib * 2**30):
        raise AnchorError(f"host RSS guard failed at {stage}: {rss_bytes}")
    host_available = None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
            host_available = int(fields[1]) * 1024
            break
    if host_available is None or host_available < int(args.minimum_host_available_gib * 2**30):
        raise AnchorError(f"host available-memory guard failed at {stage}")
    free_gpu = reserved_gpu = None
    if device.type == "cuda":
        free_gpu, _total = torch.cuda.mem_get_info(device)
        reserved_gpu = torch.cuda.memory_reserved(device)
        if free_gpu < int(args.minimum_free_gib * 2**30):
            raise AnchorError(f"GPU free-memory guard failed at {stage}: {free_gpu}")
        if reserved_gpu > int(args.maximum_gpu_reserved_gib * 2**30):
            raise AnchorError(f"GPU reserved-memory guard failed at {stage}: {reserved_gpu}")
    return {
        "stage": stage,
        "elapsed_seconds": elapsed,
        "process_max_rss_bytes": rss_bytes,
        "host_available_bytes": host_available,
        "gpu_free_bytes": None if free_gpu is None else int(free_gpu),
        "gpu_reserved_bytes": None if reserved_gpu is None else int(reserved_gpu),
    }


def _adapter_snapshot(adapter: Any) -> dict[str, float]:
    return {
        "calls": float(getattr(adapter, "calls", 0)),
        "proposal_seconds": float(getattr(adapter, "proposal_seconds", 0.0)),
        "candidate_simulations": float(getattr(adapter, "candidate_simulations", 0)),
        "executed_candidate_simulations": float(getattr(adapter, "executed_candidate_simulations", 0)),
        "prefix_commit_tokens": float(getattr(adapter, "prefix_commit_tokens", 0)),
        "prefix_calls": float(getattr(adapter, "prefix_calls", 0)),
    }


def _adapter_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, value in after.items():
        delta = value - before[key]
        result[key] = int(delta) if key != "proposal_seconds" else float(delta)
    return result


def _run_domain(*, adapter: Any, observation: Mapping[str, Any], domain: str, device: torch.device, started: float, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    adapter.begin_cell()
    warmup_seconds = 0.0
    measured_seconds = 0.0
    row_staging_seconds = 0.0
    warmup_work = {key: 0 for key in ("calls", "candidate_simulations", "executed_candidate_simulations", "prefix_commit_tokens", "prefix_calls")}
    warmup_work["proposal_seconds"] = 0.0
    measured_work = dict(warmup_work)
    rows: list[torch.Tensor] = []
    activations = observation["activations"]
    masks = observation["mask"]
    positions = observation["positions"]
    for index in range(ANCHOR_RECORDS_PER_DOMAIN):
        stage_started = time.perf_counter()
        row_h = activations[index].to(device=device, dtype=torch.float32)
        row_mask = masks[index].to(device=device, dtype=torch.bool)
        row_positions = positions[index].to(device=device, dtype=torch.int64)
        row_staging_seconds += time.perf_counter() - stage_started
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before = _adapter_snapshot(adapter)
        t0 = time.perf_counter()
        with torch.inference_mode():
            warm = _normalize_prediction(adapter(row_h, row_mask, row_positions), masks[index])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        warmup_seconds += time.perf_counter() - t0
        after = _adapter_snapshot(adapter)
        for key, value in _adapter_delta(before, after).items():
            warmup_work[key] += value  # type: ignore[operator]
        before = after
        t0 = time.perf_counter()
        with torch.inference_mode():
            measured = _normalize_prediction(adapter(row_h, row_mask, row_positions), masks[index])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        measured_seconds += time.perf_counter() - t0
        after = _adapter_snapshot(adapter)
        for key, value in _adapter_delta(before, after).items():
            measured_work[key] += value  # type: ignore[operator]
        if not torch.equal(warm, measured):
            raise AnchorError(f"warmup/measured A1+A2 IDs differ: {domain}/{index}")
        rows.append(measured)
        _guard(device, started=started, args=args, stage=f"after_{domain}_record_{index + 1}")
    evidence = dict(adapter.evidence())
    evidence.update(
        {
            "domain": domain,
            "records": ANCHOR_RECORDS_PER_DOMAIN,
            "warmup_runs_per_record": 1,
            "measured_runs_per_record": 1,
            "warmup_seconds_sum": warmup_seconds,
            "measured_seconds_sum": measured_seconds,
            "measured_ms_per_record": 1000.0 * measured_seconds / ANCHOR_RECORDS_PER_DOMAIN,
            "observation_load_seconds": float(observation["load_seconds"]),
            "row_staging_seconds": row_staging_seconds,
            "row_staging_excluded_from_method_interval": True,
            "timed_interval_includes_adapter_cpu_gpu_staging_and_cuda_synchronization": True,
            "timed_interval_definition": "A2 adapter call from device-resident H/mask/positions through CPU proposal staging, public-prefix simulation, selector, output normalization, and explicit CUDA synchronization",
            "warmup_adapter_work": warmup_work,
            "measured_adapter_work": measured_work,
            "warmup_output_exact_match_measured": True,
            "candidate_arrays_persisted": False,
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "a2_fallback": False,
            "proposal_budget": PROPOSAL_K,
            "candidate_budget": SELECTOR_K,
            "port_label": "CPU-normalized public embedding table; published parent public-prefix A2 selector",
        }
    )
    return torch.stack(rows, dim=0), evidence


def _anchor_descriptor(*, domain: str, prediction_path: Path, root: Path, selection: Mapping[str, Any], observation: Mapping[str, Any], timing: Mapping[str, Any], state_record: Mapping[str, Any]) -> dict[str, Any]:
    subset_sha = selection["subset_record_ids_sha256"][domain]
    descriptor = {
        "task_id": TASK_ID,
        "domain": domain,
        "target": "public_base",
        "method_id": ANCHOR_METHOD_ID,
        "subset": ANCHOR_SUBSET,
        "records": ANCHOR_RECORDS_PER_DOMAIN,
        "shape": [ANCHOR_RECORDS_PER_DOMAIN, SEQUENCE_TOKENS],
        "sequence_tokens": SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS,
        "record_ids_sha256": subset_sha,
        "anchor_subset_record_ids_sha256": subset_sha,
        "anchor_subset_sequence_hashes": list(selection["subset_sequence_hashes"][domain]),
        "attention_mask_sha256": observation["attention_mask_sha256"],
        "position_ids_sha256": observation["position_ids_sha256"],
        "observation_sha256": observation["file"]["sha256"],
        "state_sha256": ANCHOR_STATE_SHA256,
        "state": dict(state_record),
        "prediction": _file_record(prediction_path, root=root),
        "timing": dict(timing),
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    return descriptor


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(root / "experiments" / "TRR-P06")
    except ValueError as exc:
        raise AnchorError("anchor output must be under experiments/TRR-P06") from exc
    if output_root.exists() or output_root.is_symlink():
        raise AnchorError(f"anchor output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    started_clock = time.perf_counter()
    started_utc = _utc_now()
    failure_path = output_root / "failure.json"
    try:
        if args.device != "cuda" or not torch.cuda.is_available():
            raise AnchorError("the published A1+A2 anchor requires CUDA for its public-prefix selector")
        device = torch.device("cuda")
        selection_path = Path(args.selection).expanduser().resolve()
        selection = _load_selection(selection_path, root=root)
        observation_manifest_path = Path(args.observation_manifest).expanduser().resolve()
        observation_manifest = _load_json(observation_manifest_path, description="P06 observation manifest")
        selection_binding = observation_manifest.get("selection_plan")
        if not isinstance(selection_binding, Mapping) or selection_binding.get("sha256") != selection["file"]["sha256"]:
            raise AnchorError("observation manifest is bound to a different source selection")
        observations: dict[str, dict[str, Any]] = {}
        for domain in DOMAINS:
            observations[domain] = _load_observation(observation_manifest_path, observation_manifest, domain=domain, root=root, expected_ids_sha=selection["record_ids_sha256"][domain])
        _guard(device, started=started_clock, args=args, stage="before_parent_resource_load")
        lens_path = Path(args.lens_path).expanduser().resolve()
        reference_path = Path(args.reference_path).expanduser().resolve()
        embedding_path = Path(args.embedding_path).expanduser().resolve()
        snapshot = Path(args.model_snapshot).expanduser().resolve()
        if _sha256_file(lens_path) != ANCHOR_STATE_SHA256:
            raise AnchorError("retained A1 state does not match the frozen anchor SHA-256")
        if _sha256_file(reference_path) != REFERENCE_SHA256:
            raise AnchorError("published parent reference implementation changed")
        embedding_record = _file_record(embedding_path)
        if embedding_record["bytes"] != EMBEDDING_BYTES or embedding_record["sha256"] != EMBEDDING_SHA256:
            raise AnchorError("normalized public embedding table does not match the frozen parent asset")
        if snapshot.name != MODEL_REVISION:
            raise AnchorError("public model snapshot revision changed")
        # Importing the published helper is deliberate: this adapter does not
        # reimplement proposal ordering, cache transitions, or A2 selection.
        import trr0004_predict_confirmation as legacy
        precut, lens, embeddings, parent_load = legacy._load_public_prefix(
            snapshot=snapshot,
            reference_path=reference_path,
            lens_path=lens_path,
            embedding_path=embedding_path,
            device=device,
        )
        policy = importlib.import_module("trr0003_footing_compare")._fixed_k256_policy()
        adapter = legacy._A2Adapter(precut=precut, lens=lens, embeddings=embeddings, device=device, policy=policy)
        adapter.method_id = ANCHOR_METHOD_ID
        state_record = _file_record(lens_path, root=root)
        descriptors: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        guards = [_guard(device, started=started_clock, args=args, stage="after_parent_resource_load")]
        for domain in DOMAINS:
            predictions, timing = _run_domain(adapter=adapter, observation=observations[domain], domain=domain, device=device, started=started_clock, args=args)
            prediction_path = output_root / "predictions" / domain / f"{ANCHOR_METHOD_ID}.safetensors"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"predictions": predictions.to(torch.int64).contiguous()}, str(prediction_path), metadata={"schema": ANCHOR_SCHEMA, "task_id": TASK_ID, "domain": domain, "target": "public_base", "method_id": ANCHOR_METHOD_ID, "subset": ANCHOR_SUBSET, "state_sha256": ANCHOR_STATE_SHA256, "truth_opened": "false", "candidate_arrays_persisted": "false"})
            descriptor = _anchor_descriptor(domain=domain, prediction_path=prediction_path, root=root, selection=selection, observation=observations[domain], timing=timing, state_record=state_record)
            descriptors[domain] = descriptor
            timings[domain] = timing
            _write_create_only(output_root / "timing" / f"{domain}.json", descriptor)
            guards.append(_guard(device, started=started_clock, args=args, stage=f"after_{domain}"))
            legacy._clear_ephemeral_a2(adapter)
        manifest = {
            "schema": ANCHOR_SCHEMA,
            "task_id": TASK_ID,
            "status": "P06_ANCHOR_PREDICTIONS_COMPLETE_NO_TRUTH",
            "method_id": ANCHOR_METHOD_ID,
            "method_rule": "retained A1 top-512 proposals scored by fixed public-prefix direct-cosine A2 K=256; candidate arrays omitted",
            "anchor_state_sha256": ANCHOR_STATE_SHA256,
            "anchor_subset_record_ids_sha256": dict(selection["subset_record_ids_sha256"]),
            "anchor_cells": descriptors,
            "selection": selection["file"],
            "observation_manifest": _file_record(observation_manifest_path, root=root),
            "runtime_assets": {
                "embedding_table": _file_record(embedding_path, root=root),
                "retained_a1_state": state_record,
                "public_model_snapshot": {"path": str(snapshot), "revision": MODEL_REVISION, "loader": "trr0004_predict_confirmation._load_public_prefix"},
                "parent_reference": _file_record(reference_path, root=root),
            },
            "parent_load": parent_load,
            "timings": timings,
            "resource_guards": guards,
            "code_commit": _git_head(root),
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "truth_opened": False,
            "source_text_loaded": False,
            "token_ids_loaded": False,
            "target_labels_loaded": False,
            "candidate_arrays_persisted": False,
        }
        _write_create_only(output_root / "anchor_predictions.json", manifest)
        _write_create_only(output_root / "run_manifest.json", {"schema": RUN_SCHEMA, "task_id": TASK_ID, "status": "PUBLIC_A1_A2_ANCHOR_COMPLETE_NO_TRUTH", "anchor_manifest": _file_record(output_root / "anchor_predictions.json", root=root), "predictions": {domain: descriptors[domain]["prediction"] for domain in DOMAINS}, "code_commit": manifest["code_commit"], "truth_opened": False, "source_text_loaded": False, "token_ids_loaded": False, "candidate_arrays_persisted": False, "elapsed_seconds": manifest["elapsed_seconds"]})
        return manifest
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_create_only(failure_path, {"schema": FAILURE_SCHEMA, "task_id": TASK_ID, "status": "FAILED_PRESERVED_NO_TRUTH", "started_utc": started_utc, "ended_utc": _utc_now(), "error_type": type(exc).__name__, "error": str(exc), "truth_opened": False, "source_text_loaded": False, "token_ids_loaded": False, "candidate_arrays_persisted": False})
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--lens-path", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
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
    except (AnchorError, OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        print(f"TRR-P06 anchor error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
