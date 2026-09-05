#!/usr/bin/env python3
"""Check faithful historical InputLens logits and ranks on public validation data.

This diagnostic is intentionally narrower than a reconstruction evaluation. It
loads the retained historical A1 checkpoint and the already prepared public
normalized embedding table, then compares this bridge with the independent
Round-001 reference implementation on the old 24-record public validation
slice. It opens no target/private truth, performs no fitting, and calls no
public prefix or candidate simulator. The output is a create-only evidence
record for orientation, normalization, dtype, scale, logit, and top-k rank
equivalence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import resource as sys_resource
import subprocess
import sys
import time
from typing import Any

from safetensors.torch import load_file
import torch

from token_reconstruction.historical_inputlens_bridge import (
    HISTORICAL_HIDDEN_SIZE,
    HistoricalInputLensError,
    binding_metadata,
    file_sha256,
    load_historical_lens_checkpoint,
    validate_normalized_embeddings,
)


TASK_ID = "TRR-0004"
SCHEMA = "token-reconstruction.trr0004-historical-inputlens-equivalence.v1"
VOCAB_SIZE = 128256
PUBLIC_VALIDATION_RECORDS = 24
DEFAULT_CHECKPOINT = Path("outputs/TRR-0002/blind/reconstructor_input/public_a1_lens.pt")
DEFAULT_OBSERVATIONS = Path(
    "outputs/TRR-0003/track_b/public_validation_slice_v2/public_validation_observations.safetensors"
)
DEFAULT_EMBEDDINGS = Path(
    "outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors"
)
DEFAULT_OUTPUT = Path(
    "experiments/TRR-0004/bridge/public_validation_equivalence.json"
)
DEFAULT_MIN_FREE_GIB = 8.0
DEFAULT_MAX_GPU_RESERVED_GIB = 4.0
DEFAULT_MAX_HOST_RSS_GIB = 16.0
DEFAULT_MAX_SECONDS = 120.0
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_OFFLINE",
    "OMP_NUM_THREADS",
    "PYTHONPATH",
    "TOKENIZERS_PARALLELISM",
    "TORCH_CUDA_ARCH_LIST",
    "TRANSFORMERS_OFFLINE",
)


class BridgeDiagnosticError(RuntimeError):
    """Raised when the no-fit bridge diagnostic cannot complete safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BridgeDiagnosticError(f"{label} must be a regular file: {path}")
    return path


def _ensure_create_only(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise BridgeDiagnosticError(f"refusing to overwrite diagnostic output: {path}")


def _create_json(path: Path, value: dict[str, Any]) -> None:
    _ensure_create_only(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise BridgeDiagnosticError(f"refusing to overwrite diagnostic output: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BridgeDiagnosticError("CUDA was requested but is unavailable")
    return device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_tensor(path: Path, *, key: str) -> torch.Tensor:
    _regular_file(path, label="tensor artifact")
    try:
        state = load_file(path, device="cpu")
    except Exception as exc:  # pragma: no cover - safetensors backend detail
        raise BridgeDiagnosticError(f"cannot load tensor artifact: {path}") from exc
    if set(state) != {key}:
        raise BridgeDiagnosticError(f"{path} must contain exactly {key!r}")
    value = state[key].contiguous()
    if not torch.isfinite(value).all().item():
        raise BridgeDiagnosticError(f"{path} contains non-finite values")
    return value


def _load_observations(path: Path) -> torch.Tensor:
    value = _load_tensor(path, key="activations")
    if value.ndim != 3 or value.shape[0] != PUBLIC_VALIDATION_RECORDS:
        raise BridgeDiagnosticError(
            "historical bridge diagnostic requires the registered 24-record validation slice"
        )
    if value.shape[1] <= 0 or value.shape[2] != HISTORICAL_HIDDEN_SIZE:
        raise BridgeDiagnosticError("validation observations must be [24,positions,2048]")
    if not value.dtype.is_floating_point:
        raise BridgeDiagnosticError("validation observations must be floating point")
    return value


def _load_normalized_embeddings(path: Path) -> torch.Tensor:
    value = _load_tensor(path, key="embeddings")
    validate_normalized_embeddings(value, vocabulary_size=VOCAB_SIZE, check_unit_norm=True)
    return value


def _load_reference(reference_path: Path, checkpoint: Path, device: torch.device) -> tuple[Any, Any]:
    spec = importlib.util.spec_from_file_location("trr0004_round001_reference", reference_path)
    if spec is None or spec.loader is None:
        raise BridgeDiagnosticError(f"cannot import reference implementation: {reference_path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses in the reference inspect sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    lens = module.load_frozen_lens(checkpoint, device=device)
    return module, lens


class _Accumulator:
    def __init__(self) -> None:
        self.elements = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.reference_sq = 0.0
        self.max_abs = 0.0

    def update(self, actual: torch.Tensor, reference: torch.Tensor) -> None:
        actual_f = actual.float()
        reference_f = reference.float()
        if actual_f.shape != reference_f.shape:
            raise BridgeDiagnosticError("bridge/reference geometry differs")
        if not torch.isfinite(actual_f).all().item() or not torch.isfinite(reference_f).all().item():
            raise BridgeDiagnosticError("bridge/reference comparison is non-finite")
        error = actual_f - reference_f
        self.elements += int(error.numel())
        self.sum_abs += float(error.abs().sum().item())
        self.sum_sq += float(error.square().sum().item())
        self.reference_sq += float(reference_f.square().sum().item())
        self.max_abs = max(self.max_abs, float(error.abs().max().item()))

    def finish(self, *, exact_equal: bool, allclose: bool) -> dict[str, Any]:
        if self.elements <= 0:
            raise BridgeDiagnosticError("empty comparison")
        return {
            "elements": self.elements,
            "max_abs": self.max_abs,
            "mean_abs": self.sum_abs / self.elements,
            "rmse": (self.sum_sq / self.elements) ** 0.5,
            "relative_l2": (self.sum_sq / max(self.reference_sq, 1e-24)) ** 0.5,
            "exact_equal": exact_equal,
            "allclose_atol_1e-6_rtol_1e-6": allclose,
        }


def _host_max_rss_bytes() -> int:
    value = int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.  The execution environment is
    # Linux, but keeping the distinction explicit makes the guard portable.
    return value if sys.platform == "darwin" else value * 1024


def _memory(device: torch.device) -> dict[str, int]:
    result = {"host_max_rss_bytes": _host_max_rss_bytes()}
    if device.type == "cuda":
        result.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return result


def _git_commit(repository_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _code_sources(repository_root: Path) -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        repository_root / "src" / "token_reconstruction" / "historical_inputlens_bridge.py",
        repository_root / "reference" / "strict_bos" / "round001_teacher.py",
    )


def _code_snapshot(repository_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _code_sources(repository_root):
        path = _regular_file(path, label="bound executable source")
        records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return records


def _safe_environment() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in SAFE_ENVIRONMENT_KEYS}


def _execution_snapshot(repository_root: Path) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(repository_root),
        "code": _code_snapshot(repository_root),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "argv": list(sys.argv),
        "environment": _safe_environment(),
    }


def _resource_policy(args: argparse.Namespace) -> dict[str, float | int | bool]:
    values = {
        "minimum_free_gpu_bytes": int(args.minimum_free_gib * 1024**3),
        "maximum_gpu_reserved_bytes": int(args.maximum_gpu_reserved_gib * 1024**3),
        "maximum_host_rss_bytes": int(args.maximum_host_rss_gib * 1024**3),
        "maximum_wall_seconds": float(args.max_seconds),
        "exclusive_gpu_required": True,
    }
    if any(value <= 0 for key, value in values.items() if key != "exclusive_gpu_required"):
        raise BridgeDiagnosticError("resource guard limits must be positive")
    return values


def _resource_preflight(device: torch.device, policy: dict[str, float | int | bool]) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "minimum_free_gpu_bytes": None,
            "free_gpu_bytes_before": None,
            "total_gpu_bytes": None,
            "status": "host_only_no_gpu_check",
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    minimum = int(policy["minimum_free_gpu_bytes"])
    if free_bytes < minimum:
        raise BridgeDiagnosticError(
            f"CUDA free memory {free_bytes} is below required {minimum} bytes"
        )
    return {
        "device": str(device),
        "minimum_free_gpu_bytes": minimum,
        "free_gpu_bytes_before": int(free_bytes),
        "total_gpu_bytes": int(total_bytes),
        "status": "pass",
    }


def _check_resource_limits(
    device: torch.device,
    policy: dict[str, float | int | bool],
    *,
    started: float,
    stage: str,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    maximum_seconds = float(policy["maximum_wall_seconds"])
    if elapsed > maximum_seconds:
        raise BridgeDiagnosticError(
            f"resource guard wall-time limit exceeded at {stage}: {elapsed:.3f}s > {maximum_seconds:.3f}s"
        )
    rss = _host_max_rss_bytes()
    maximum_rss = int(policy["maximum_host_rss_bytes"])
    if rss > maximum_rss:
        raise BridgeDiagnosticError(
            f"resource guard host RSS limit exceeded at {stage}: {rss} > {maximum_rss} bytes"
        )
    result: dict[str, Any] = {"stage": stage, "elapsed_seconds": elapsed, "host_max_rss_bytes": rss}
    if device.type == "cuda":
        reserved = int(torch.cuda.max_memory_reserved(device))
        maximum_reserved = int(policy["maximum_gpu_reserved_bytes"])
        if reserved > maximum_reserved:
            raise BridgeDiagnosticError(
                f"resource guard GPU reserved limit exceeded at {stage}: {reserved} > {maximum_reserved} bytes"
            )
        result.update(
            {
                "cuda_peak_reserved_bytes": reserved,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _regular_file(args.checkpoint, label="historical lens checkpoint")
    observations_path = _regular_file(args.observations, label="validation observations")
    embeddings_path = _regular_file(args.embeddings, label="normalized embeddings")
    if args.batch_size <= 0 or args.top_k <= 0 or args.top_k > VOCAB_SIZE:
        raise BridgeDiagnosticError("batch size and top-k must be positive and within vocabulary")

    args.repository_root = args.repository_root.resolve()
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = _utc_now()
    monotonic_started = time.perf_counter()
    policy = _resource_policy(args)
    execution_start = _execution_snapshot(args.repository_root)
    preflight = _resource_preflight(device, policy)
    guard_observations = [_check_resource_limits(
        device, policy, started=monotonic_started, stage="after_preflight"
    )]
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    bridge = load_historical_lens_checkpoint(checkpoint, device=device)
    reference_path = args.repository_root / "reference" / "strict_bos" / "round001_teacher.py"
    reference_module, reference_lens = _load_reference(reference_path, checkpoint, device)
    _sync(device)
    timings["checkpoint_load_seconds"] = time.perf_counter() - t0
    guard_observations.append(_check_resource_limits(
        device, policy, started=monotonic_started, stage="after_checkpoint_load"
    ))

    t0 = time.perf_counter()
    observations = _load_observations(observations_path)
    embeddings_cpu = _load_normalized_embeddings(embeddings_path)
    embeddings = embeddings_cpu.to(device=device)
    _sync(device)
    timings["public_input_load_seconds"] = time.perf_counter() - t0
    guard_observations.append(_check_resource_limits(
        device, policy, started=monotonic_started, stage="after_public_input_load"
    ))

    flat = observations.reshape(-1, HISTORICAL_HIDDEN_SIZE)
    projected_metrics = _Accumulator()
    logit_metrics = _Accumulator()
    rank_rows = 0
    rank_position_mismatches = 0
    rank_top1_mismatches = 0
    exact_logits = True
    allclose_logits = True
    exact_projection = True
    allclose_projection = True
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, int(flat.shape[0]), args.batch_size):
            stop = min(start + args.batch_size, int(flat.shape[0]))
            activation = flat[start:stop].to(device=device)
            bridge_projection = bridge.projected(activation)
            reference_projection = reference_lens.projected(activation)
            bridge_logits = bridge(activation, embeddings)
            reference_logits = reference_lens(activation, embeddings)
            projected_metrics.update(bridge_projection, reference_projection)
            logit_metrics.update(bridge_logits, reference_logits)
            exact_projection = exact_projection and bool(torch.equal(bridge_projection, reference_projection))
            allclose_projection = allclose_projection and bool(
                torch.allclose(bridge_projection.float(), reference_projection.float(), atol=1e-6, rtol=1e-6)
            )
            exact_logits = exact_logits and bool(torch.equal(bridge_logits, reference_logits))
            allclose_logits = allclose_logits and bool(
                torch.allclose(bridge_logits.float(), reference_logits.float(), atol=1e-6, rtol=1e-6)
            )
            bridge_rank = torch.topk(bridge_logits, k=args.top_k, dim=-1, sorted=True).indices
            reference_rank = torch.topk(reference_logits, k=args.top_k, dim=-1, sorted=True).indices
            equal = bridge_rank.eq(reference_rank)
            rank_rows += int(equal.shape[0])
            rank_position_mismatches += int((~equal).sum().item())
            rank_top1_mismatches += int((~equal[:, 0]).sum().item())
            guard_observations.append(_check_resource_limits(
                device, policy, started=monotonic_started, stage=f"after_batch_{stop}"
            ))
    _sync(device)
    timings["bridge_and_reference_seconds"] = time.perf_counter() - t0
    _sync(device)
    guard_observations.append(_check_resource_limits(
        device, policy, started=monotonic_started, stage="after_comparison"
    ))
    execution_end = _execution_snapshot(args.repository_root)
    if execution_start["git_commit"] != execution_end["git_commit"] or execution_start["code"] != execution_end["code"]:
        raise BridgeDiagnosticError("bound executable source changed during diagnostic")

    bridge_meta = binding_metadata(
        bridge,
        checkpoint_path=checkpoint,
        normalized_embedding_path=embeddings_path,
    )
    bridge_meta["reference_source"] = {
        "path": str(reference_path),
        "sha256": file_sha256(reference_path),
        "implementation": "reference.strict_bos.round001_teacher.FrozenAffineLens",
        "state_sha256": reference_module.state_sha256(
            dict(reference_lens.state_dict()), domain=b"ersoy-a1-lens-state-v1"
        ),
    }
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "diagnostic_only",
        "started_at": started,
        "finished_at": _utc_now(),
        "truth_opened": False,
        "fitting_steps": 0,
        "candidate_simulations": 0,
        "public_prefix_calls": 0,
        "execution": {
            "start": execution_start,
            "end": execution_end,
            "code_unchanged": True,
        },
        "resource_guard": {
            "policy": policy,
            "preflight": preflight,
            "observations": guard_observations,
            "final_memory": _memory(device),
        },
        "validation": {
            "records": int(observations.shape[0]),
            "positions_per_record": int(observations.shape[1]),
            "rows_compared": int(flat.shape[0]),
            "observations": {
                "path": str(observations_path),
                "sha256": file_sha256(observations_path),
                "bytes": observations_path.stat().st_size,
                "shape": list(observations.shape),
                "dtype": str(observations.dtype),
            },
            "source": "registered public 24-record development validation slice; no private or panel truth",
        },
        "fixed_settings": {
            "batch_size": args.batch_size,
            "rank_top_k": args.top_k,
            "device": str(device),
            "activation_cast": "activation.float32 inside projected()",
            "embedding_input": "pre-normalized public table; no renormalization in forward",
        },
        "method_state": bridge_meta,
        "equivalence": {
            "projection": projected_metrics.finish(
                exact_equal=exact_projection,
                allclose=allclose_projection,
            ),
            "logits": logit_metrics.finish(exact_equal=exact_logits, allclose=allclose_logits),
            "rank": {
                "k": args.top_k,
                "rows": rank_rows,
                "position_mismatches": rank_position_mismatches,
                "top1_mismatches": rank_top1_mismatches,
                "exact_equal": rank_position_mismatches == 0,
            },
        },
        "timing_seconds": timings,
        "memory": _memory(device),
        "interpretation": (
            "This control tests implementation fidelity of the retained historical A1 path. "
            "It is not a target-transfer result or a replacement claim."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=DEFAULT_MAX_GPU_RESERVED_GIB)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=DEFAULT_MAX_HOST_RSS_GIB)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        # Check before loading the external embedding table or starting CUDA
        # work; a stale output must fail closed without consuming resources.
        _ensure_create_only(args.output)
        evidence = run(args)
        _create_json(args.output, evidence)
    except (BridgeDiagnosticError, HistoricalInputLensError) as exc:
        print(f"trr0004 historical bridge diagnostic failed: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

