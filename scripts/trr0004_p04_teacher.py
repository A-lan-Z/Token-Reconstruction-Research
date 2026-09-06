#!/usr/bin/env python3
"""Freeze the bounded privileged public-prefix teacher evidence for TRR-P04."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import torch

from token_reconstruction.p04_teacher import PUBLIC_MODEL_SPEC, P04TeacherError, qualify_teacher
from token_reconstruction.p04_training import file_sha256, load_public_pool


MIN_FREE_GPU_BYTES = 8 * 1024**3
MAX_RESERVED_GPU_BYTES = 6 * 1024**3
MAX_HOST_RSS_BYTES = 16 * 1024**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise P04TeacherError("unable to record teacher source commit") from exc


def _cuda_memory_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        raise P04TeacherError("CUDA memory snapshot requested without CUDA")
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-observations", type=Path, required=True)
    parser.add_argument("--correction-records", type=Path, required=True)
    parser.add_argument("--correction-truth", type=Path)
    parser.add_argument("--correction-mask", type=Path)
    parser.add_argument("--lens-path", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--candidate-preparation", type=Path, required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20260906)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _args()
    process_started = time.perf_counter()
    started_utc = _utc_now()
    source_commit = _source_commit()
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        # Empty means no explicit restriction; CUDA_VISIBLE_DEVICES=-1 is the
        # caller's fail-closed choice for CPU-only execution.
        pass
    import torch

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    if not torch.cuda.is_available():
        raise P04TeacherError("teacher qualification requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    preflight_gpu = _cuda_memory_snapshot()
    if preflight_gpu["free_bytes"] < MIN_FREE_GPU_BYTES:
        raise P04TeacherError(
            f"teacher requires at least {MIN_FREE_GPU_BYTES} free GPU bytes; got {preflight_gpu['free_bytes']}"
        )
    identity = json.loads(args.model_identity.read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise P04TeacherError("model identity must be a JSON object")
    correction_started = time.perf_counter()
    correction = load_public_pool(
        args.correction_observations,
        args.correction_records,
        truth_path=args.correction_truth,
        mask_path=args.correction_mask,
        embedding_vocab_size=128256,
    )
    correction_seconds = time.perf_counter() - correction_started
    teacher_started = time.perf_counter()
    result = qualify_teacher(
        correction,
        model_identity=identity,
        lens_path=args.lens_path,
        reference_path=args.reference_path,
        candidate_preparation_path=args.candidate_preparation,
        embedding_path=args.embedding_table,
        selection_path=args.selection,
        output_root=args.output_root,
        selection_seed=args.selection_seed,
    )
    teacher_seconds = time.perf_counter() - teacher_started
    post_gpu = _cuda_memory_snapshot()
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if post_gpu["free_bytes"] < MIN_FREE_GPU_BYTES:
        raise P04TeacherError(
            f"teacher ended below the {MIN_FREE_GPU_BYTES}-byte GPU free margin: {post_gpu['free_bytes']}"
        )
    if post_gpu["max_memory_reserved_bytes"] > MAX_RESERVED_GPU_BYTES:
        raise P04TeacherError(
            f"teacher peak reserved GPU memory exceeded {MAX_RESERVED_GPU_BYTES}: {post_gpu['max_memory_reserved_bytes']}"
        )
    if peak_rss_bytes > MAX_HOST_RSS_BYTES:
        raise P04TeacherError(
            f"teacher peak host RSS exceeded {MAX_HOST_RSS_BYTES}: {peak_rss_bytes}"
        )
    input_paths = {
        "correction_observations": args.correction_observations,
        "correction_records": args.correction_records,
        "lens": args.lens_path,
        "reference": args.reference_path,
        "candidate_preparation": args.candidate_preparation,
        "embedding_table": args.embedding_table,
        "model_identity": args.model_identity,
    }
    receipt = {
        "task_id": "TRR-P04",
        "schema": "token-reconstruction.trr-p04-teacher-receipt.v1",
        "status": "PASS",
        "argv": sys.argv,
        "source_commit": source_commit,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "model_spec": PUBLIC_MODEL_SPEC,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "inputs": {
            key: {"path": str(path.expanduser().resolve()), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
        },
        "phase_timing_seconds": {
            "public_pool_load": correction_seconds,
            "teacher_qualification": teacher_seconds,
            "process_total": time.perf_counter() - process_started,
        },
        "resource_guard": {
            "minimum_free_gpu_bytes": MIN_FREE_GPU_BYTES,
            "maximum_reserved_gpu_bytes": MAX_RESERVED_GPU_BYTES,
            "maximum_host_rss_bytes": MAX_HOST_RSS_BYTES,
            "preflight": preflight_gpu,
            "post": post_gpu,
            "host_max_rss_bytes": peak_rss_bytes,
            "status": "PASS",
        },
        "output": result,
        "wall_seconds": time.perf_counter() - process_started,
        "peak_rss_bytes": peak_rss_bytes,
    }
    output = args.output_root.expanduser().resolve()
    receipt_path = output / "teacher_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise P04TeacherError(f"teacher receipt is create-only: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "source_commit": source_commit, "evidence": result["evidence"], "metrics": result["metrics"], "resource_guard": receipt["resource_guard"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (P04TeacherError, RuntimeError) as exc:
        print(f"P04 teacher failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
