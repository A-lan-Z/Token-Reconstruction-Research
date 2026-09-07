#!/usr/bin/env python3
"""Bounded, truth-free zero-Delta deployment equivalence diagnostic.

This probe is deliberately separate from fitting, selection, and scoring.  It
uses the already-opened TRR-0009 ``public_base`` activation fixture for the
first two records of each domain and compares three readout paths:

* the selected TRR-0009 residual decoder with the immutable public ``E``;
* the TRR-0010 adapter at exact zero Delta with its materialized ``E_eff``;
* a freshly reloaded residual decoder with the serialized ``E_eff`` export.

Only aggregate equality and maximum absolute differences are written.  No
fixture labels, current-record truth, source text, or accuracy metrics are
loaded or emitted. Public fitting-support IDs/counts are read only to bind the
zero-Delta support set.
The command is intended to run under an external ``timeout`` and an exclusive
GPU lease; this module also applies fail-closed resource checks before and
after loading public tensors.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open

from scripts import trr0010_model as model_source_module
from scripts import trr0010_p09_caller as caller_source_module
from scripts.trr0010_model import (
    build_directional_from_base,
    export_effective_embedding,
    load_published_residual_state,
    support_digest,
)
from token_reconstruction import trr0007_positionwise as loader_source_module
from token_reconstruction.trr0007_positionwise import RESIDUAL_MLP_METHOD_ID, save_positionwise_state


SCHEMA = "token-reconstruction.trr0010-opened-fixture-equivalence.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0010-opened-fixture-equivalence-failure.v1"
VOCABULARY_SIZE = 128256
HIDDEN_SIZE = 2048
CONTEXT_WIDTH = 128
BOTTLENECK_SIZE = 512
EXPECTED_STATE_SHA256 = "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14"
EXPECTED_EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
EXPECTED_SUPPORT_DIGEST = "8090b9231042d6c9f9c52f3a0a9d8733b8ec1b739a4f28b12ab1057a7126969d"
EXPECTED_SUPPORT_IDS_SHA256 = "b3bef773294cbc79755852512d79c21c910dca30f000387e8c73a69c19c8f104"
EXPECTED_SUPPORT_COUNTS_SHA256 = "3ae3412ccf0061adcc0a24e3ef40ec0fbe87cd44e935bfc9e24462ae87f18825"
EXPECTED_OBSERVATION_SHA256 = {
    "pile": "5ad0fece58e4247d8fb1a1d8f5d1d5a021eb96cded0be112f0aafa9052dab89a",
    "finance": "7449bf11fd335ec8d46ca7581378b8e3a3351d7b7852561cac944c723dd643bb",
}
MIN_GPU_FREE_PRE_BYTES = 8 * 1024**3
MIN_GPU_FREE_RUNTIME_BYTES = 2 * 1024**3
MAX_GPU_RESERVED_BYTES = 4 * 1024**3
MAX_HOST_RSS_BYTES = 6 * 1024**3
MIN_HOST_AVAILABLE_BYTES = 10 * 1024**3
MIN_DISK_FREE_BYTES = 20 * 1024**3


class DiagnosticError(RuntimeError):
    """Raised for a fail-closed diagnostic precondition or mismatch."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_info(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DiagnosticError("unable to bind repository source state") from exc
    if len(head) != 40:
        raise DiagnosticError("repository HEAD is not a full commit hash")
    return {"head": head, "working_tree_status": status}


def _module_record(module: Any, *, label: str) -> dict[str, Any]:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise DiagnosticError(f"{label} does not expose an importable source path")
    path = Path(raw_path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise DiagnosticError(f"{label} source is not a regular file: {path}")
    return {"label": label, "module": str(getattr(module, "__name__", label)), "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path)}


def _runtime_source_bindings() -> dict[str, Any]:
    return {
        "model": _module_record(model_source_module, label="TRR-0010 model"),
        "loader": _module_record(loader_source_module, label="TRR-0007 positionwise loader"),
        "caller": _module_record(caller_source_module, label="TRR-P09 caller"),
    }


def _host_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    raise DiagnosticError("Linux MemAvailable is unavailable; refusing to qualify")


def _rss_bytes() -> int:
    # Linux ru_maxrss is KiB, whereas the current RSS is available in status.
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _resource_snapshot(device: torch.device, *, stage: str) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise DiagnosticError("this diagnostic requires an available CUDA device")
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "stage": stage,
        "gpu_free_bytes": int(free),
        "gpu_total_bytes": int(total),
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "host_rss_bytes": _rss_bytes(),
        "host_available_bytes": _host_available_bytes(),
    }


def _check_snapshot(snapshot: dict[str, Any], *, pre_load: bool) -> None:
    if snapshot["host_available_bytes"] < MIN_HOST_AVAILABLE_BYTES:
        raise DiagnosticError(
            f"host available memory below floor at {snapshot['stage']}: "
            f"{snapshot['host_available_bytes']} < {MIN_HOST_AVAILABLE_BYTES}"
        )
    if snapshot["host_rss_bytes"] > MAX_HOST_RSS_BYTES:
        raise DiagnosticError(
            f"host RSS above floor at {snapshot['stage']}: "
            f"{snapshot['host_rss_bytes']} > {MAX_HOST_RSS_BYTES}"
        )
    if snapshot["gpu_reserved_bytes"] > MAX_GPU_RESERVED_BYTES:
        raise DiagnosticError(
            f"GPU reserved memory above limit at {snapshot['stage']}: "
            f"{snapshot['gpu_reserved_bytes']} > {MAX_GPU_RESERVED_BYTES}"
        )
    floor = MIN_GPU_FREE_PRE_BYTES if pre_load else MIN_GPU_FREE_RUNTIME_BYTES
    if snapshot["gpu_free_bytes"] < floor:
        raise DiagnosticError(
            f"GPU free memory below {'pre-load' if pre_load else 'runtime'} floor at "
            f"{snapshot['stage']}: {snapshot['gpu_free_bytes']} < {floor}"
        )


def _path_under(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise DiagnosticError(f"{label} is not a regular file: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError:
        # Public E and the already-opened observation fixture intentionally live
        # outside this task worktree; their exact allowlisted paths are checked
        # by the caller instead of accepting arbitrary external inputs.
        pass
    return resolved


def _bound_file(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise DiagnosticError(f"{label} is unavailable or symlinked: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise DiagnosticError(f"{label} SHA-256 differs from frozen binding: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": actual}


def _load_single_tensor(path: Path, *, key: str, label: str) -> torch.Tensor:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {key}:
                raise DiagnosticError(f"{label} tensor keys differ from [{key!r}]")
            value = handle.get_tensor(key).contiguous()
    except DiagnosticError:
        raise
    except Exception as exc:
        raise DiagnosticError(f"unable to load {label}: {path}") from exc
    return value


def _selected_step(path: Path) -> int:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise DiagnosticError(f"unable to read selected state metadata: {path}") from exc
    try:
        step = int(metadata["selected_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DiagnosticError("selected state lacks an integer selected_step") from exc
    if step < 0:
        raise DiagnosticError("selected state selected_step is negative")
    return step


def _load_observation(path: Path, *, domain: str, records: int) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            expected_keys = {"activations", "attention_mask", "position_ids"}
            if set(handle.keys()) != expected_keys:
                raise DiagnosticError(f"{domain} observation keys differ from the frozen fixture")
            activation = handle.get_slice("activations")[:records].contiguous()
            mask = handle.get_slice("attention_mask")[:records].contiguous()
            positions = handle.get_slice("position_ids")[:records].contiguous()
    except DiagnosticError:
        raise
    except Exception as exc:
        raise DiagnosticError(f"unable to load {domain} public_base observation") from exc
    if tuple(activation.shape) != (records, 128, HIDDEN_SIZE):
        raise DiagnosticError(f"{domain} activation geometry changed: {tuple(activation.shape)}")
    if activation.dtype != torch.bfloat16 or not torch.isfinite(activation.float()).all().item():
        raise DiagnosticError(f"{domain} activation dtype or finiteness changed")
    if tuple(mask.shape) != (records, 128) or mask.dtype not in (torch.bool, torch.uint8):
        raise DiagnosticError(f"{domain} mask geometry or dtype changed")
    if tuple(positions.shape) != (records, 128):
        raise DiagnosticError(f"{domain} position geometry changed")
    expected_positions = torch.arange(128, dtype=torch.long).unsqueeze(0).expand(records, -1)
    if not torch.equal(positions.to(dtype=torch.long), expected_positions):
        raise DiagnosticError(f"{domain} position IDs are not the frozen 128-position clip")
    mask = mask.to(dtype=torch.bool)
    if not mask[:, 0].all().item():
        raise DiagnosticError(f"{domain} fixture has an invalid BOS mask")
    return activation, mask


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise DiagnosticError(f"comparison geometry differs: {tuple(left.shape)} vs {tuple(right.shape)}")
    difference = (left - right).abs()
    return float(difference.max().item()) if difference.numel() else 0.0


def _ensure_deadline(started: float, limit: float) -> None:
    if time.perf_counter() - started > limit:
        raise DiagnosticError(f"internal wall-time limit exceeded: {limit} seconds")


def _default_paths(root: Path) -> dict[str, Path]:
    project = root.parents[1]
    trr9 = root.parent / "TRR-0009"
    trr9_eval = trr9 / "experiments" / "TRR-0009" / "evaluation"
    return {
        "state": trr9 / "experiments" / "TRR-0009" / "training" / "run_v1" / "continued_fixed_readout" / "selected.safetensors",
        "embedding": project / "outputs" / "TRR-0003" / "track_b" / "public_fit_v2" / "public_normalized_embeddings.safetensors",
        "support_ids": trr9_eval / "support_ids.safetensors",
        "support_counts": trr9_eval / "support_counts.safetensors",
        "pile": trr9_eval / "public_observations_v2" / "observations" / "pile__public_base.safetensors",
        "finance": trr9_eval / "public_observations_v2" / "observations" / "finance__public_base.safetensors",
    }


def run_diagnostic(*, root: Path, output_root: Path, device_name: str, records: int, max_seconds: float) -> dict[str, Any]:
    started_perf = time.perf_counter()
    started_utc = _utc_now()
    if records != 2:
        raise DiagnosticError("the frozen diagnostic must use exactly two records per domain")
    if max_seconds <= 0.0:
        raise DiagnosticError("max_seconds must be positive")
    output_root = output_root.expanduser().resolve()
    task_eval_root = (root / "experiments" / "TRR-0010" / "evaluation").resolve()
    try:
        output_root.relative_to(task_eval_root)
    except ValueError as exc:
        raise DiagnosticError(f"output root must be below {task_eval_root}") from exc
    if output_root.exists():
        raise DiagnosticError(f"output root already exists (create-only): {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=False)
    paths = _default_paths(root)
    if device_name != "cuda":
        raise DiagnosticError("only --device cuda is permitted by this diagnostic contract")
    device = torch.device("cuda")
    snapshots: list[dict[str, Any]] = []

    disk = shutil.disk_usage(output_root.parent)
    if int(disk.free) < MIN_DISK_FREE_BYTES:
        raise DiagnosticError(f"disk free space below floor: {disk.free} < {MIN_DISK_FREE_BYTES}")
    if not torch.cuda.is_available():
        raise DiagnosticError("CUDA is unavailable")
    snapshots.append(_resource_snapshot(device, stage="pre_load"))
    _check_snapshot(snapshots[-1], pre_load=True)
    torch.cuda.reset_peak_memory_stats(device)

    code_info = _git_info(root)
    source_record = _bound_file(Path(__file__), expected_sha256=_sha256(Path(__file__)), label="diagnostic source")
    imported_source_records = _runtime_source_bindings()
    selected_step = _selected_step(paths["state"])
    state_record = _bound_file(paths["state"], expected_sha256=EXPECTED_STATE_SHA256, label="selected fixed state")
    embedding_record = _bound_file(paths["embedding"], expected_sha256=EXPECTED_EMBEDDING_SHA256, label="public embedding")
    support_ids_record = _bound_file(paths["support_ids"], expected_sha256=EXPECTED_SUPPORT_IDS_SHA256, label="support IDs")
    support_counts_record = _bound_file(paths["support_counts"], expected_sha256=EXPECTED_SUPPORT_COUNTS_SHA256, label="support counts")
    observation_records = {
        domain: _bound_file(paths[domain], expected_sha256=EXPECTED_OBSERVATION_SHA256[domain], label=f"{domain} public_base observation")
        for domain in ("pile", "finance")
    }
    _ensure_deadline(started_perf, max_seconds)

    support_ids = _load_single_tensor(paths["support_ids"], key="support_ids", label="support IDs")
    support_counts = _load_single_tensor(paths["support_counts"], key="support_counts", label="support counts")
    if support_ids.dtype != torch.int64 or support_counts.dtype != torch.int64:
        raise DiagnosticError("support vectors must be int64")
    if support_digest(support_ids, support_counts) != EXPECTED_SUPPORT_DIGEST:
        raise DiagnosticError("support digest differs from the bound public B0 support")
    public_embedding = _load_single_tensor(paths["embedding"], key="embeddings", label="public embedding")
    if tuple(public_embedding.shape) != (VOCABULARY_SIZE, HIDDEN_SIZE) or public_embedding.dtype != torch.float32:
        raise DiagnosticError("public embedding geometry or dtype changed")
    if not torch.isfinite(public_embedding).all().item():
        raise DiagnosticError("public embedding contains non-finite values")
    observations = {
        domain: _load_observation(paths[domain], domain=domain, records=records)
        for domain in ("pile", "finance")
    }
    snapshots.append(_resource_snapshot(device, stage="after_public_cpu_load"))
    _check_snapshot(snapshots[-1], pre_load=False)

    base = load_published_residual_state(
        paths["state"],
        hidden_size=HIDDEN_SIZE,
        vocabulary_size=VOCABULARY_SIZE,
        context_width=CONTEXT_WIDTH,
        bottleneck_size=BOTTLENECK_SIZE,
    ).eval()
    base = base.to(device)
    adapter = build_directional_from_base(base, support_ids, support_counts)
    adapter = adapter.to(device).eval()
    if not torch.equal(adapter.delta_rows, torch.zeros_like(adapter.delta_rows)):
        raise DiagnosticError("directional Delta did not initialize exactly at zero")
    embedding_gpu = public_embedding.to(device)
    adapter.bind_embedding_statistics(embedding_gpu)
    snapshots.append(_resource_snapshot(device, stage="after_model_embedding_load"))
    _check_snapshot(snapshots[-1], pre_load=False)

    initial_results: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        effective_gpu = adapter.materialize_effective_embedding(embedding_gpu)
        for domain, (activation_cpu, mask_cpu) in observations.items():
            _ensure_deadline(started_perf, max_seconds)
            activation_gpu = activation_cpu.to(device)
            mask_gpu = mask_cpu.to(device)
            base_logits = base(activation_gpu, mask_gpu, embedding_gpu)
            merged_logits = adapter.merged_forward(activation_gpu, mask_gpu, effective_gpu)
            initial_results[domain] = {
                "records": records,
                "positions": 128,
                "vocabulary": VOCABULARY_SIZE,
                "zero_delta_exact": bool(torch.equal(base_logits, merged_logits)),
                "zero_delta_max_abs": _max_abs(base_logits, merged_logits),
            }
            del activation_gpu, mask_gpu, base_logits, merged_logits
        snapshots.append(_resource_snapshot(device, stage="after_zero_delta_probe"))
        _check_snapshot(snapshots[-1], pre_load=False)
        del effective_gpu, embedding_gpu, adapter, base
        torch.cuda.empty_cache()
        snapshots.append(_resource_snapshot(device, stage="after_gpu_release_before_export"))
        _check_snapshot(snapshots[-1], pre_load=False)

    # Export through the same create-only helper used for deployment.  This is
    # CPU-side after releasing the first GPU effective-E allocation, so the
    # diagnostic never retains two full E dictionaries on the device.
    base_cpu = load_published_residual_state(
        paths["state"],
        hidden_size=HIDDEN_SIZE,
        vocabulary_size=VOCABULARY_SIZE,
        context_width=CONTEXT_WIDTH,
        bottleneck_size=BOTTLENECK_SIZE,
    ).eval()
    adapter_cpu = build_directional_from_base(base_cpu, support_ids, support_counts).eval()
    adapter_cpu.bind_embedding_statistics(public_embedding)
    export_path = output_root / "effective_readout_w.safetensors"
    export_result = export_effective_embedding(
        export_path,
        adapter_cpu,
        public_embedding,
        metadata={"diagnostic": "zero_delta_opened_fixture", "truth_opened": False},
    )
    base_only_export_path = output_root / "base_decoder_state.safetensors"
    base_only_export_result = save_positionwise_state(
        base_only_export_path,
        base_cpu,
        method_id=RESIDUAL_MLP_METHOD_ID,
        selected_step=selected_step,
        initialization="reloaded selected TRR-0009 fixed decoder",
        distribution="TRR-0010 opened-fixture deployment equivalence diagnostic",
        bottleneck_size=BOTTLENECK_SIZE,
        metadata={
            "base_state_role": "BASE_ONLY_SELECTED_DECODER",
            "directional_readout_excluded": True,
            "source_selected_state_sha256": EXPECTED_STATE_SHA256,
            "truth_opened": False,
        },
    )
    del adapter_cpu
    _ensure_deadline(started_perf, max_seconds)

    reloaded_embedding = _load_single_tensor(export_path, key="embeddings", label="serialized effective readout")
    if tuple(reloaded_embedding.shape) != (VOCABULARY_SIZE, HIDDEN_SIZE) or reloaded_embedding.dtype != torch.float32:
        raise DiagnosticError("serialized effective readout geometry or dtype changed")
    base_reloaded = load_published_residual_state(
        base_only_export_path,
        hidden_size=HIDDEN_SIZE,
        vocabulary_size=VOCABULARY_SIZE,
        context_width=CONTEXT_WIDTH,
        bottleneck_size=BOTTLENECK_SIZE,
    ).eval().to(device)
    base_original = load_published_residual_state(
        paths["state"],
        hidden_size=HIDDEN_SIZE,
        vocabulary_size=VOCABULARY_SIZE,
        context_width=CONTEXT_WIDTH,
        bottleneck_size=BOTTLENECK_SIZE,
    ).eval().to(device)
    embedding_gpu = public_embedding.to(device)
    reloaded_gpu = reloaded_embedding.to(device)
    reloaded_results: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for domain, (activation_cpu, mask_cpu) in observations.items():
            _ensure_deadline(started_perf, max_seconds)
            activation_gpu = activation_cpu.to(device)
            mask_gpu = mask_cpu.to(device)
            original_logits = base_original(activation_gpu, mask_gpu, embedding_gpu)
            base_export_logits = base_reloaded(activation_gpu, mask_gpu, embedding_gpu)
            reloaded_logits = base_reloaded(activation_gpu, mask_gpu, reloaded_gpu)
            reloaded_results[domain] = {
                "records": records,
                "positions": 128,
                "vocabulary": VOCABULARY_SIZE,
                "base_only_export_exact_vs_original": bool(torch.equal(original_logits, base_export_logits)),
                "base_only_export_max_abs_vs_original": _max_abs(original_logits, base_export_logits),
                "reloaded_effective_readout_exact": bool(torch.equal(base_export_logits, reloaded_logits)),
                "reloaded_effective_readout_max_abs": _max_abs(base_export_logits, reloaded_logits),
            }
            del activation_gpu, mask_gpu, original_logits, base_export_logits, reloaded_logits
    snapshots.append(_resource_snapshot(device, stage="after_reloaded_export_probe"))
    _check_snapshot(snapshots[-1], pre_load=False)

    elapsed = time.perf_counter() - started_perf
    aggregate = {
        "all_zero_delta_exact": all(row["zero_delta_exact"] for row in initial_results.values()),
        "all_base_only_export_exact": all(row["base_only_export_exact_vs_original"] for row in reloaded_results.values()),
        "all_reloaded_export_exact": all(row["reloaded_effective_readout_exact"] for row in reloaded_results.values()),
        "max_zero_delta_abs": max(row["zero_delta_max_abs"] for row in initial_results.values()),
        "max_base_only_export_abs": max(row["base_only_export_max_abs_vs_original"] for row in reloaded_results.values()),
        "max_reloaded_export_abs": max(row["reloaded_effective_readout_max_abs"] for row in reloaded_results.values()),
    }
    status = "PASS" if all(aggregate[key] for key in ("all_zero_delta_exact", "all_base_only_export_exact", "all_reloaded_export_exact")) else "FAIL_NONEXACT"
    result = {
        "schema": SCHEMA,
        "status": status,
        "task_id": "TRR-0010",
        "truth_opened": False,
        "accuracy_or_labels_loaded": False,
        "records_per_domain": records,
        "domains": ["pile", "finance"],
        "geometry": {"sequence_tokens": 128, "hidden_size": HIDDEN_SIZE, "vocabulary_size": VOCABULARY_SIZE},
        "state": state_record,
        "public_embedding": embedding_record,
        "support_ids": support_ids_record,
        "support_counts": support_counts_record,
        "support_digest": EXPECTED_SUPPORT_DIGEST,
        "observations": observation_records,
        "selected_step": selected_step,
        "export": export_result,
        "base_only_export": base_only_export_result,
        "initial_zero_delta": initial_results,
        "reloaded_export": reloaded_results,
        "aggregate": aggregate,
        "resource_snapshots": snapshots,
        "disk_free_bytes_at_start": int(disk.free),
        "elapsed_seconds": elapsed,
        "max_seconds": max_seconds,
        "device": str(device),
        "argv": list(sys.argv),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
            "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")},
        },
        "source": {"repository": str(root), "diagnostic": source_record, "imported": imported_source_records, **code_info},
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "PASS":
        raise DiagnosticError(f"opened-fixture equivalence was non-exact: {aggregate}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/TRR-0010/evaluation/opened_fixture_zero_delta_v1"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--records-per-domain", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.repository_root.expanduser().resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = root / output_root
    started = _utc_now()
    try:
        result = run_diagnostic(
            root=root,
            output_root=output_root,
            device_name=str(args.device),
            records=int(args.records_per_domain),
            max_seconds=float(args.max_seconds),
        )
    except Exception as exc:
        failure_root = output_root if not output_root.exists() else output_root.with_name(output_root.name + "_failure")
        failure_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": FAILURE_SCHEMA,
            "status": "FAILED",
            "task_id": "TRR-0010",
            "truth_opened": False,
            "started_utc": started,
            "ended_utc": _utc_now(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "command": sys.argv,
        }
        (failure_root / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    print(json.dumps({"status": result["status"], "output_root": str(output_root), "aggregate": result["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
