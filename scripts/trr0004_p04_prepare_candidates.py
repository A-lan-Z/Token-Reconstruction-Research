#!/usr/bin/env python3
"""Freeze the single PR7-affine candidate table shared by P04 teacher/H/D."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from safetensors.torch import load_file
import torch

from token_reconstruction.p04_teacher import P04TeacherError, prepare_candidate_ids
from token_reconstruction.p04_training import canonical_hash, file_sha256, combine_public_pools, load_embedding_table, load_public_pool


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("replay", "correction"):
        parser.add_argument(f"--{prefix}-observations", type=Path, required=True)
        parser.add_argument(f"--{prefix}-records", type=Path, required=True)
        parser.add_argument(f"--{prefix}-truth", type=Path)
        parser.add_argument(f"--{prefix}-mask", type=Path)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--affine-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--record-batch-size", type=int, default=8)
    parser.add_argument("--projection-chunk", type=int, default=256)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    return parser.parse_args()


def _regular(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P04TeacherError(f"{label} must be a regular file: {path}")
    return path


def _load_affine_state(path: Path) -> dict[str, torch.Tensor]:
    path = _regular(path, "affine state")
    try:
        if path.suffix == ".pt":
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            state = checkpoint.get("sd") if isinstance(checkpoint, dict) else None
        else:
            state = load_file(str(path), device="cpu")
    except Exception as exc:
        raise P04TeacherError(f"cannot load affine state: {path}") from exc
    if not isinstance(state, dict) or set(state) != {"W", "b", "s"}:
        raise P04TeacherError("affine state must contain exactly W, b, and s")
    if state["W"].shape != (2048, 2048) or state["b"].shape != (2048,) or state["s"].ndim != 0:
        raise P04TeacherError("affine state geometry does not match public hidden size")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise P04TeacherError("affine state is non-finite")
    return {key: value.float().contiguous() for key, value in state.items()}


def _set_runtime(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise P04TeacherError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(20260906)
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main() -> int:
    args = _args()
    _set_runtime(args)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise P04TeacherError(f"candidate preparation output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    replay = load_public_pool(
        args.replay_observations,
        args.replay_records,
        truth_path=args.replay_truth,
        mask_path=args.replay_mask,
        embedding_vocab_size=128256,
    )
    correction = load_public_pool(
        args.correction_observations,
        args.correction_records,
        truth_path=args.correction_truth,
        mask_path=args.correction_mask,
        embedding_vocab_size=128256,
    )
    pool = combine_public_pools(replay, correction)
    table = load_embedding_table(args.embedding_table, hidden_size=pool.hidden_size, vocab_size=128256)
    affine_state = _load_affine_state(args.affine_state)
    artifact = output_root / "candidate_preparation.safetensors"
    started = time.perf_counter()
    result = prepare_candidate_ids(
        pool,
        table,
        affine_state=affine_state,
        affine_path=args.affine_state,
        embedding_path=args.embedding_table,
        output_path=artifact,
        candidate_k=args.candidate_k,
        device=torch.device(args.device),
        record_batch_size=args.record_batch_size,
        projection_chunk=args.projection_chunk,
    )
    receipt = {
        "task_id": "TRR-P04",
        "schema": "token-reconstruction.trr-p04-candidate-preparation-receipt.v1",
        "status": "PASS",
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "pool": {
            "rows": pool.rows,
            "positions": pool.positions,
            "record_order_sha256": canonical_hash(list(pool.record_ids)),
            "observation_sha256": pool.source_sha256,
        },
        "inputs": {
            "embedding_table": {"path": str(args.embedding_table.resolve()), "sha256": file_sha256(args.embedding_table)},
            "affine_state": {"path": str(args.affine_state.resolve()), "sha256": file_sha256(args.affine_state)},
        },
        "output": result,
        "tie_policy": "descending_score_then_ascending_token_id",
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    receipt_path = output_root / "candidate_preparation_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise P04TeacherError(f"candidate preparation receipt is create-only: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (P04TeacherError, RuntimeError) as exc:
        print(f"P04 candidate preparation failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
