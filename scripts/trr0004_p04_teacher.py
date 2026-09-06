#!/usr/bin/env python3
"""Freeze the bounded privileged public-prefix teacher evidence for TRR-P04."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

from token_reconstruction.p04_teacher import PUBLIC_MODEL_SPEC, P04TeacherError, qualify_teacher
from token_reconstruction.p04_training import load_public_pool


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
    identity = json.loads(args.model_identity.read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise P04TeacherError("model identity must be a JSON object")
    correction = load_public_pool(
        args.correction_observations,
        args.correction_records,
        truth_path=args.correction_truth,
        mask_path=args.correction_mask,
        embedding_vocab_size=128256,
    )
    started = time.perf_counter()
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
    receipt = {
        "task_id": "TRR-P04",
        "schema": "token-reconstruction.trr-p04-teacher-receipt.v1",
        "status": "PASS",
        "argv": sys.argv,
        "model_spec": PUBLIC_MODEL_SPEC,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "output": result,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    output = args.output_root.expanduser().resolve()
    receipt_path = output / "teacher_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise P04TeacherError(f"teacher receipt is create-only: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "evidence": result["evidence"], "metrics": result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (P04TeacherError, RuntimeError) as exc:
        print(f"P04 teacher failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
