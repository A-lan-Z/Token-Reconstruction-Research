#!/usr/bin/env python3
"""Qualify CUDA public-E score matmul against the CPU score path.

This is a public-fixture gate for the optional geometry backend.  It consumes
one already-produced B1 ``predictions/projected/geometry`` cell and the frozen
public FP32 embedding table.  Each projected row is scored once on CPU and
once on ``cuda:0``; both argmax vectors must match the frozen prediction and
one another.  Full score panels are never written.  A mismatch is retained in
an explicit receipt and exits nonzero, so the CUDA backend cannot silently
become the production path.
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
import sys
import time
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.trr_p12.geometry_pipeline import (  # noqa: E402
    B1_BASE_S,
    DEFAULT_LOGIT_SCALE,
    DEFAULT_MIN_FREE_GPU_BYTES,
    GeometryPipelineError,
    PipelineConfig,
    ScoreBackend,
    StageInput,
    file_binding,
    load_embedding,
)

SCHEMA = "token-reconstruction.trr-p12-geometry-cuda-qualification.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * 1024 if sys.platform != "darwin" else value
    except (OSError, ValueError):
        return None


def _resources(device: torch.device) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "host_rss_bytes": _rss_bytes(),
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(
            {
                "cuda_free_bytes": int(free),
                "cuda_total_bytes": int(total),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return result


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _first_mismatch(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any] | None:
    mismatch = torch.nonzero(left.ne(right), as_tuple=False)
    if not mismatch.numel():
        return None
    index = tuple(int(item) for item in mismatch[0].tolist())
    return {"index": list(index), "left": int(left[index].item()), "right": int(right[index].item())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--logit-scale", type=float, default=DEFAULT_LOGIT_SCALE)
    parser.add_argument("--min-free-gpu-gib", type=float, default=8.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.receipt_output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise GeometryPipelineError(f"qualification receipt is create-only: {output}")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise GeometryPipelineError("CUDA is unavailable")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    preflight = _resources(device)
    minimum = int(float(args.min_free_gpu_gib) * 1024**3)
    if int(preflight["cuda_free_bytes"] or 0) < minimum:
        raise GeometryPipelineError(
            f"CUDA free-memory guard failed: {preflight['cuda_free_bytes']} < {minimum}"
        )
    config = PipelineConfig(logit_scale=float(args.logit_scale))
    stage = StageInput.from_files(0, args.predictions, args.projected, args.geometry, config=config)
    embedding = load_embedding(args.embedding, config=config)
    started_utc = utc_now()
    started = time.perf_counter()
    cpu = ScoreBackend(embedding, config=config, device="cpu")
    cuda = ScoreBackend(
        embedding,
        config=config,
        device=device,
        min_free_gpu_bytes=minimum,
    )
    rows: list[dict[str, Any]] = []
    with stage.records(config=config) as records:
        for index, slot, prediction, projected in records:
            cpu_scores = cpu.scores(projected)
            cuda_scores = cuda.scores(projected)
            cpu_argmax = torch.argmax(cpu_scores, dim=-1).to(dtype=torch.long, device="cpu")
            cuda_argmax = torch.argmax(cuda_scores, dim=-1).to(dtype=torch.long, device="cpu")
            expected = prediction[1:].to(dtype=torch.long, device="cpu")
            cpu_alignment = bool(torch.equal(cpu_argmax, expected))
            cuda_alignment = bool(torch.equal(cuda_argmax, expected))
            device_alignment = bool(torch.equal(cpu_argmax, cuda_argmax))
            difference = (cpu_scores - cuda_scores).abs()
            rows.append(
                {
                    "record_index": int(index),
                    "slot": slot,
                    "scored_rows": int(cpu_argmax.numel()),
                    "cpu_argmax_sha256": _tensor_digest(cpu_argmax),
                    "cuda_argmax_sha256": _tensor_digest(cuda_argmax),
                    "frozen_prediction_sha256": _tensor_digest(expected),
                    "cpu_matches_frozen_prediction": cpu_alignment,
                    "cuda_matches_frozen_prediction": cuda_alignment,
                    "cpu_cuda_argmax_equal": device_alignment,
                    "argmax_mismatch_cpu_vs_cuda": _first_mismatch(cpu_argmax, cuda_argmax),
                    "score_max_abs_difference": float(difference.max().item()),
                    "score_mean_abs_difference": float(difference.mean().item()),
                }
            )
    if not rows:
        raise GeometryPipelineError("public geometry fixture has no records")
    passed = all(
        bool(row["cpu_matches_frozen_prediction"])
        and bool(row["cuda_matches_frozen_prediction"])
        and bool(row["cpu_cuda_argmax_equal"])
        for row in rows
    )
    payload = {
        "schema": SCHEMA,
        "task_id": "TRR-P12",
        "status": "CUDA_SCORE_FIXTURE_EQUAL" if passed else "CUDA_SCORE_FIXTURE_MISMATCH_RETAINED",
        "created_utc": utc_now(),
        "started_utc": started_utc,
        "elapsed_seconds": time.perf_counter() - started,
        "command": "python scripts/trr_p12/qualify_geometry_cuda.py --predictions ... --projected ... --geometry ... --embedding ... --receipt-output ...",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "config": {
            "vocabulary_size": config.vocabulary_size,
            "hidden_size": config.hidden_size,
            "stored_sequence_tokens": config.stored_sequence_tokens,
            "scored_positions": config.scored_positions,
            "base_s": B1_BASE_S,
            "logit_scale": config.logit_scale,
            "logit_scale_source": "torch.float32 exp(base.s)",
        },
        "fixture": {
            "stage": stage.bindings(),
            "embedding": file_binding(args.embedding, label="public embedding table"),
        },
        "cpu_backend": cpu.metadata(),
        "cuda_backend": cuda.metadata(),
        "rows": rows,
        "verification": {
            "full_vocabulary_argmax_checked": True,
            "cpu_argmax_against_frozen_prediction": True,
            "cuda_argmax_against_frozen_prediction": True,
            "cpu_cuda_argmax_exact": passed,
            "raw_logits_retained": False,
        },
        "preflight_resources": preflight,
        "final_resources": _resources(device),
        "access_boundary": {
            "truth_opened": False,
            "source_text_loaded": False,
            "target_weights_loaded": False,
            "target_prefix_queried": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    payload["receipt_path"] = str(output)
    payload["receipt_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run(args)
    except (GeometryPipelineError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-P12 CUDA geometry qualification error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": payload["status"], "receipt_path": payload["receipt_path"]}, sort_keys=True))
    return 0 if payload["status"] == "CUDA_SCORE_FIXTURE_EQUAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
