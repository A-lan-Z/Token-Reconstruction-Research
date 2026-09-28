"""Qualify exact CPU/GPU B1 predictions on the preserved four-record smoke.

This command is intended to run only under the existing TRR-P09 resource
watchdog.  It creates one small JSON receipt and no prediction artifact.  The
receipt is the gate accepted by ``B1PersistentAdapter(device='cuda:0')``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from scripts.trr_p12.adapter import AdapterError, compare_cpu_gpu_fixture


SCHEMA = "token-reconstruction.trr-p12-gpu-equivalence-run.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_rss_bytes() -> int | None:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None


def peak_rss_bytes() -> int | None:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform != "darwin" else value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "TRR-P12" / "gpu-equivalence-r1.json",
    )
    parser.add_argument("--watchdog-root", type=Path, required=True)
    parser.add_argument("--gpu-cost-per-hour", type=float, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.receipt_output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise AdapterError(f"GPU equivalence receipt is create-only: {output}")
    if args.gpu_cost_per_hour is not None and float(args.gpu_cost_per_hour) < 0:
        raise AdapterError("GPU cost rate must be nonnegative")
    if not torch.cuda.is_available():
        raise AdapterError("CUDA is unavailable; no equivalence attempt was run")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    started_utc = utc_now()
    started = time.perf_counter()
    status = "CPU_GPU_FIXTURE_EQUAL"
    error: str | None = None
    comparison: dict[str, Any] | None = None
    try:
        comparison = compare_cpu_gpu_fixture(
            Path(args.package_root).expanduser().resolve(),
            Path(args.observations).expanduser().resolve(),
            gpu_device="cuda:0",
        )
    except Exception as exc:
        status = "CPU_GPU_FIXTURE_FAILED"
        error = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gpu_hours = elapsed / 3600.0
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": "TRR-P12",
        "status": status,
        "created_utc": utc_now(),
        "command": list(sys.argv),
        "command_shell": " ".join(sys.argv),
        "start_utc": started_utc,
        "end_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "resource": {
            "rss_bytes": current_rss_bytes(),
            "process_peak_rss_bytes": peak_rss_bytes(),
            "cuda_device": str(device),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "cost": {
            "gpu_hours": gpu_hours,
            "gpu_rate_per_hour": args.gpu_cost_per_hour,
            "estimated_gpu_cost": None if args.gpu_cost_per_hour is None else gpu_hours * float(args.gpu_cost_per_hour),
            "currency": None,
        },
        "watchdog": {
            "implementation": "scripts/trr_p09/resource_watchdog.py",
            "output_root": str(Path(args.watchdog_root).expanduser().resolve()),
            "declared_timeout_seconds": 285.0,
            "outer_timeout_seconds": 300.0,
            "finish_bound_after_child_exit": "parent wrapper must bind finish.json after this child receipt",
        },
        "comparison": comparison,
        "error": error,
        "access_boundary": {
            "truth_opened": False,
            "source_text_loaded": False,
            "token_ids_loaded": False,
            "target_weights_loaded": False,
            "target_prefix_queried": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "CPU_GPU_FIXTURE_EQUAL":
        raise AdapterError(error or "CPU/GPU fixture equivalence failed")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run(args)
    except (AdapterError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-P12 GPU equivalence error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": payload["status"], "receipt_output": str(args.receipt_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
