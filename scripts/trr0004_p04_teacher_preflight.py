#!/usr/bin/env python3
"""Bounded one-cell GPU preflight for the TRR-P04 public teacher.

This command is a resource qualification only.  It consumes the already frozen
candidate table, runs one maximum-position K=32 public-prefix simulation, and
writes a JSON receipt.  It never writes teacher evidence and never opens
private evaluation truth.
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping

import torch

from token_reconstruction.p04_teacher import (
    PUBLIC_MODEL_SPEC,
    P04TeacherError,
    _build_known_public_cache,
    _centered_cosine_scores,
    _load_candidate_preparation,
)
from token_reconstruction.p04_training import (
    file_sha256,
    load_public_pool,
    tensor_sha256,
)


TASK_ID = "TRR-P04"
SCHEMA = "token-reconstruction.trr-p04-teacher-preflight.v1"
CANDIDATE_K = 32
MAX_POSITION = 191
MIN_FREE_GPU_BYTES = 8 * 1024**3
MAX_RESERVED_GPU_BYTES = 6 * 1024**3
MAX_HOST_RSS_BYTES = 16 * 1024**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise P04TeacherError("unable to record teacher preflight source commit") from exc


def _cuda_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        raise P04TeacherError("teacher preflight requires CUDA")
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _peak_rss_bytes() -> int:
    # Linux reports KiB through ru_maxrss.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P04TeacherError(f"cannot parse {label}: {path}") from exc
    if not isinstance(value, dict):
        raise P04TeacherError(f"{label} must be a JSON object")
    return value


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-observations", type=Path, required=True)
    parser.add_argument("--correction-records", type=Path, required=True)
    parser.add_argument("--lens-path", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--candidate-preparation", type=Path, required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--position", type=int, default=MAX_POSITION)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    return parser.parse_args()


def _candidate_budget(precut: Any, *, position: int) -> dict[str, int]:
    config = precut.config
    hidden_size = int(getattr(config, "hidden_size", 2048))
    layers = int(getattr(precut, "cut", len(precut.layers)))
    attention_heads = int(getattr(config, "num_attention_heads", 0))
    key_value_heads = int(getattr(config, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(config, "head_dim", 0) or (hidden_size // attention_heads))
    dtype_bytes = int(precut.embed_tokens.weight.element_size())
    # One K/V cache for each of the four public prefix layers, plus one output
    # hidden row. This is a lower-level geometry budget; the hard guard below
    # remains the authoritative measured limit.
    cache_bytes = 2 * layers * CANDIDATE_K * position * key_value_heads * head_dim * dtype_bytes
    hidden_bytes = CANDIDATE_K * hidden_size * dtype_bytes
    return {
        "max_position": int(position),
        "candidate_k": CANDIDATE_K,
        "prefix_layers": layers,
        "hidden_size": hidden_size,
        "key_value_heads": key_value_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "candidate_kv_cache_bytes": int(cache_bytes),
        "candidate_output_hidden_bytes": int(hidden_bytes),
        "declared_peak_reserved_limit_bytes": MAX_RESERVED_GPU_BYTES,
        "declared_minimum_free_gpu_bytes": MIN_FREE_GPU_BYTES,
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise P04TeacherError(f"preflight receipt is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    args = _args()
    process_started = time.perf_counter()
    started_utc = _utc_now()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_commit = _source_commit()
    input_paths = {
        "correction_observations": args.correction_observations,
        "correction_records": args.correction_records,
        "lens": args.lens_path,
        "reference": args.reference_path,
        "candidate_preparation": args.candidate_preparation,
        "embedding_table": args.embedding_table,
        "model_identity": args.model_identity,
    }
    try:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.interop_threads)
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True, warn_only=False)
        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            raise P04TeacherError("CUDA_VISIBLE_DEVICES must explicitly select the preflight GPU")
        if not torch.cuda.is_available():
            raise P04TeacherError("teacher preflight requires CUDA")
        torch.cuda.reset_peak_memory_stats()
        preflight_gpu = _cuda_snapshot()
        if preflight_gpu["free_bytes"] < MIN_FREE_GPU_BYTES:
            raise P04TeacherError(
                f"teacher preflight requires at least {MIN_FREE_GPU_BYTES} free GPU bytes; got {preflight_gpu['free_bytes']}"
            )
        identity = _json_object(args.model_identity, "model identity")
        pool_started = time.perf_counter()
        pool = load_public_pool(
            args.correction_observations,
            args.correction_records,
            embedding_vocab_size=128256,
        )
        pool_seconds = time.perf_counter() - pool_started
        if args.record_id not in pool.record_ids:
            raise P04TeacherError(f"preflight record is outside correction pool: {args.record_id}")
        pool_row = pool.record_ids.index(args.record_id)
        valid_count = int(pool.valid_mask[pool_row].sum().item())
        if args.position != MAX_POSITION:
            raise P04TeacherError(f"preflight position is fixed at {MAX_POSITION}; got {args.position}")
        if valid_count <= args.position:
            raise P04TeacherError(
                f"preflight record has only {valid_count} active positions; position {args.position} is invalid"
            )

        reference_started = time.perf_counter()
        # Importing the pinned comparator is intentionally delegated to its
        # loader, which performs the model/lens identity checks.
        module_spec = importlib.util.spec_from_file_location("trr_p04_preflight_reference", args.reference_path.expanduser().resolve())
        if module_spec is None or module_spec.loader is None:
            raise P04TeacherError(f"cannot import frozen reference: {args.reference_path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        try:
            precut, lens, normalized_embeddings, device, observed_identity = module.load_public_teacher(
                PUBLIC_MODEL_SPEC,
                identity,
                lens_path=args.lens_path.expanduser().resolve(),
            )
        except Exception as exc:
            raise P04TeacherError("frozen public teacher failed identity/load checks") from exc
        if device != torch.device("cuda"):
            raise P04TeacherError("teacher preflight loaded on an unexpected device")
        reference_seconds = time.perf_counter() - reference_started
        budget = _candidate_budget(precut, position=args.position)
        post_load_gpu = _cuda_snapshot()
        estimated_headroom = max(
            budget["candidate_kv_cache_bytes"] * 4 + budget["candidate_output_hidden_bytes"] * 8,
            1 * 1024**2,
        )
        if post_load_gpu["free_bytes"] < MIN_FREE_GPU_BYTES + estimated_headroom:
            raise P04TeacherError(
                "teacher preflight lacks declared single-cell headroom after public model load: "
                f"free={post_load_gpu['free_bytes']} required={MIN_FREE_GPU_BYTES + estimated_headroom}"
            )

        prepared_candidates, prepared_proposals, prepared_confidence, preparation_metadata = _load_candidate_preparation(
            args.candidate_preparation,
            positions=pool.positions,
            candidate_k=CANDIDATE_K,
        )
        try:
            prepared_record_ids = json.loads(preparation_metadata["pool_record_ids_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise P04TeacherError("candidate preparation lacks frozen pool record order") from exc
        if not isinstance(prepared_record_ids, list) or args.record_id not in prepared_record_ids:
            raise P04TeacherError("candidate preparation lacks preflight record")
        prepared_row = int(prepared_record_ids.index(args.record_id))
        if preparation_metadata.get("embedding_file_sha256") != file_sha256(args.embedding_table):
            raise P04TeacherError("candidate preparation embedding asset does not match preflight input")
        candidates = prepared_candidates[prepared_row, args.position].contiguous()
        proposals = prepared_proposals[prepared_row, args.position].contiguous()
        if candidates.shape != (CANDIDATE_K,) or proposals.shape != (512,):
            raise P04TeacherError("candidate preparation cell geometry changed")
        confidence = float(prepared_confidence[prepared_row, args.position].item())
        labels = pool.labels[pool_row]
        activation = pool.observations[pool_row, args.position]
        simulation_started = time.perf_counter()
        cache = _build_known_public_cache(precut, labels, position=args.position, device=device)
        simulated = module._simulate_candidates(
            precut,
            cache=cache,
            candidate_ids=candidates,
            position=args.position,
            device=device,
        )
        scores = _centered_cosine_scores(simulated, activation.to(device=device)).detach().float().cpu()
        torch.cuda.synchronize(device)
        simulation_seconds = time.perf_counter() - simulation_started
        post_cell_gpu = _cuda_snapshot()
        peak_rss_bytes = _peak_rss_bytes()
        if not torch.isfinite(scores).all().item():
            raise P04TeacherError("single-cell teacher scores are non-finite")
        if post_cell_gpu["free_bytes"] < MIN_FREE_GPU_BYTES:
            raise P04TeacherError(f"single-cell preflight ended below free-GPU margin: {post_cell_gpu['free_bytes']}")
        if post_cell_gpu["max_memory_reserved_bytes"] > MAX_RESERVED_GPU_BYTES:
            raise P04TeacherError(
                f"single-cell preflight peak reserved GPU memory exceeded {MAX_RESERVED_GPU_BYTES}: "
                f"{post_cell_gpu['max_memory_reserved_bytes']}"
            )
        if peak_rss_bytes > MAX_HOST_RSS_BYTES:
            raise P04TeacherError(f"single-cell preflight peak host RSS exceeded {MAX_HOST_RSS_BYTES}: {peak_rss_bytes}")
        target = int(labels[args.position].item())
        winner = int(scores.argmax().item())
        metrics = {
            "record_id": args.record_id,
            "position": args.position,
            "active_positions": valid_count,
            "candidate_k": CANDIDATE_K,
            "proposal_k": 512,
            "candidate_ids_sha256": tensor_sha256(candidates),
            "proposal_ids_sha256": tensor_sha256(proposals),
            "a1_confidence": confidence,
            "target_in_candidate_k32": target in set(int(value) for value in candidates.tolist()),
            "target_in_proposal_k512": target in set(int(value) for value in proposals.tolist()),
            "teacher_winner_index": winner,
            "teacher_winner_token": int(candidates[winner].item()),
            "target_token": target,
            "top_score": float(scores.max().item()),
            "score_span": float((scores.max() - scores.min()).item()),
            "score_tie_count": int(scores.eq(scores.max()).sum().item()),
            "finite_scores": True,
        }
        receipt = {
            "task_id": TASK_ID,
            "schema": SCHEMA,
            "status": "PASS",
            "scope": "single_cell_resource_preflight",
            "truth_accessed": False,
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
            "geometry_budget": {
                **budget,
                "full_pool_logits_forbidden_bytes": int(pool.post_bos_positions * 128256 * 4),
                "full_pool_logits_forbidden_shape": [int(pool.post_bos_positions), 128256],
                "estimated_single_cell_headroom_bytes": int(estimated_headroom),
            },
            "phase_timing_seconds": {
                "public_pool_load": pool_seconds,
                "public_teacher_load": reference_seconds,
                "single_cell_simulation": simulation_seconds,
                "process_total": time.perf_counter() - process_started,
            },
            "resource_guard": {
                "minimum_free_gpu_bytes": MIN_FREE_GPU_BYTES,
                "maximum_reserved_gpu_bytes": MAX_RESERVED_GPU_BYTES,
                "maximum_host_rss_bytes": MAX_HOST_RSS_BYTES,
                "preflight": preflight_gpu,
                "post_public_teacher_load": post_load_gpu,
                "post_single_cell": post_cell_gpu,
                "host_max_rss_bytes": peak_rss_bytes,
                "status": "PASS",
            },
            "metrics": metrics,
            "observed_teacher_identity": observed_identity,
            "candidate_preparation": {
                "path": str(args.candidate_preparation.expanduser().resolve()),
                "sha256": file_sha256(args.candidate_preparation),
                "proposer_id": preparation_metadata.get("proposer_id", ""),
                "tie_policy": preparation_metadata.get("tie_policy", ""),
            },
            "wall_seconds": time.perf_counter() - process_started,
            "peak_rss_bytes": peak_rss_bytes,
        }
        _write_create_only(output_root / "teacher_preflight.json", receipt)
        print(json.dumps({"status": "PASS", "output": str(output_root / "teacher_preflight.json"), "metrics": metrics, "resource_guard": receipt["resource_guard"]}, sort_keys=True))
        return 0
    except (P04TeacherError, RuntimeError, OSError, ValueError) as exc:
        failure = {
            "task_id": TASK_ID,
            "schema": "token-reconstruction.trr-p04-teacher-preflight-failure.v1",
            "status": "FAIL_CLOSED",
            "truth_accessed": False,
            "scope": "single_cell_resource_preflight",
            "source_commit": source_commit,
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "argv": sys.argv,
            "error": str(exc),
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        try:
            _write_create_only(output_root / "failure.json", failure)
        except Exception:
            pass
        print(f"P04 teacher preflight failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
