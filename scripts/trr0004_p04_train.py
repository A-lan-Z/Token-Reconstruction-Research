#!/usr/bin/env python3
"""Train the preregistered TRR-P04 affine/S/H/D public comparison.

The command has one data-loading path for all arms and seeds. It writes the
schedule and candidate identities before the first optimizer update, and each
arm freezes a selected public-validation checkpoint separately from its final
state. It never opens evaluator observations, target-update weights, or
private truth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from token_reconstruction.p04_student import (
    ALL_METHODS,
    METHOD_AFFINE,
    METHOD_D,
    METHOD_H,
    METHOD_S,
    StudentArchitectureConfig,
)
from token_reconstruction.p04_training import (
    CANDIDATE_PREPARATION_SCHEMA,
    DEFAULT_CANDIDATE_K,
    P04TrainingError,
    TrainingConfig,
    build_teacher_arrays,
    canonical_hash,
    combine_public_pools,
    file_sha256,
    load_embedding_table,
    load_public_pool,
    load_teacher_evidence,
    make_position_schedule,
    save_schedule,
    tensor_sha256,
    train_arm,
)


TASK_ID = "TRR-P04"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("replay", "correction", "validation"):
        parser.add_argument(f"--{prefix}-observations", type=Path, required=True)
        parser.add_argument(f"--{prefix}-records", type=Path, required=True)
        parser.add_argument(f"--{prefix}-truth", type=Path)
        parser.add_argument(f"--{prefix}-mask", type=Path)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--affine-state", type=Path, required=True)
    parser.add_argument("--teacher-evidence", type=Path)
    parser.add_argument("--candidate-ids", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--record-batch-size", type=int, default=8)
    parser.add_argument("--position-budget", type=int, default=512)
    parser.add_argument("--replay-fraction", type=float, default=0.75)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--arms", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _regular(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P04TrainingError(f"{label} must be a regular file: {path}")
    return path


def _load_affine_state(path: Path, *, hidden_size: int) -> dict[str, torch.Tensor]:
    path = _regular(path, "affine state")
    try:
        if path.suffix == ".pt":
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            state = checkpoint.get("sd") if isinstance(checkpoint, dict) else None
            if not isinstance(state, dict):
                raise P04TrainingError("pickled affine state lacks sd")
        else:
            state = load_file(str(path), device="cpu")
    except P04TrainingError:
        raise
    except Exception as exc:
        raise P04TrainingError(f"cannot load affine state: {path}") from exc
    if set(state) != {"W", "b", "s"}:
        raise P04TrainingError("affine state must contain exactly W, b, and s")
    if state["W"].shape != (hidden_size, hidden_size) or state["b"].shape != (hidden_size,) or state["s"].ndim != 0:
        raise P04TrainingError("affine state geometry does not match public hidden size")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise P04TrainingError("affine state is non-finite")
    return {key: value.float().contiguous() for key, value in state.items()}


def _load_candidates(path: Path, *, rows: int, positions: int, candidate_k: int) -> tuple[torch.Tensor, dict[str, str]]:
    path = _regular(path, "candidate identities")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if "candidate_ids" not in keys:
                raise P04TrainingError("candidate preparation lacks candidate_ids")
            metadata = dict(handle.metadata() or {})
            value = handle.get_tensor("candidate_ids").contiguous()
    except P04TrainingError:
        raise
    except Exception as exc:
        raise P04TrainingError(f"cannot load candidate identities: {path}") from exc
    if metadata.get("schema") != CANDIDATE_PREPARATION_SCHEMA:
        raise P04TrainingError("training requires the canonical PR7-affine candidate preparation schema")
    if metadata.get("proposer_id") != "pr7_public_affine":
        raise P04TrainingError("candidate preparation proposer identity is not the frozen PR7 affine resource")
    if metadata.get("candidate_k") != str(candidate_k) or metadata.get("proposal_k") != "512":
        raise P04TrainingError("candidate preparation budgets do not match the fixed P04 contract")
    if value.shape != (rows, positions, candidate_k):
        raise P04TrainingError("candidate artifact geometry does not match public pool")
    value = value.to(dtype=torch.int32).contiguous()
    if value.min().item() < 0 or value.max().item() >= 128256:
        raise P04TrainingError("candidate IDs are outside the public vocabulary")
    sorted_ids = value.to(torch.int64).sort(dim=-1).values
    if sorted_ids.shape[-1] > 1 and sorted_ids[..., 1:].eq(sorted_ids[..., :-1]).any().item():
        raise P04TrainingError("candidate preparation contains duplicate IDs")
    return value, metadata


def _set_runtime(args: argparse.Namespace, *, seed: int) -> None:
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise P04TrainingError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _source_receipt(args: argparse.Namespace) -> dict[str, object]:
    return {
        "task_id": TASK_ID,
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cwd": str(Path.cwd()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_files": {
            key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for key, path in {
                "replay_observations": args.replay_observations,
                "replay_records": args.replay_records,
                "correction_observations": args.correction_observations,
                "correction_records": args.correction_records,
                "validation_observations": args.validation_observations,
                "validation_records": args.validation_records,
                "embedding_table": args.embedding_table,
                "affine_state": args.affine_state,
                **({"teacher_evidence": args.teacher_evidence} if args.teacher_evidence else {}),
                **({"candidate_ids": args.candidate_ids} if args.candidate_ids else {}),
            }.items()
        },
    }


def main() -> int:
    args = _parse_args()
    seeds = args.seeds or [1737, 2711]
    if args.candidate_k != DEFAULT_CANDIDATE_K:
        raise P04TrainingError("P04 training candidate_k is fixed at 32")
    if any(method in (METHOD_H, METHOD_D) for method in args.arms) and args.candidate_ids is None:
        raise P04TrainingError("student_h/student_d require --candidate-ids from canonical preparation")
    if args.teacher_evidence and args.candidate_ids is None:
        raise P04TrainingError("teacher evidence requires --candidate-ids from canonical preparation")
    if len(seeds) != len(set(seeds)):
        raise P04TrainingError("training seeds must be unique")
    args.output_root = args.output_root.expanduser().resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.dry_run:
        raise P04TrainingError("output root must be a new or empty directory")
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Read geometry from observations before opening labels. The shared pool
    # loader itself preserves activation/mask checks before public labels.
    replay = load_public_pool(args.replay_observations, args.replay_records, truth_path=args.replay_truth, mask_path=args.replay_mask, embedding_vocab_size=128256)
    correction = load_public_pool(args.correction_observations, args.correction_records, truth_path=args.correction_truth, mask_path=args.correction_mask, embedding_vocab_size=128256)
    validation = load_public_pool(args.validation_observations, args.validation_records, truth_path=args.validation_truth, mask_path=args.validation_mask, embedding_vocab_size=128256)
    pool = combine_public_pools(replay, correction)
    if set(pool.record_ids).intersection(validation.record_ids):
        raise P04TrainingError("public training and validation records overlap")
    if (pool.positions, pool.hidden_size) != (validation.positions, validation.hidden_size):
        raise P04TrainingError("public training/validation geometries differ; setup must pad them")
    table = load_embedding_table(args.embedding_table, hidden_size=pool.hidden_size, vocab_size=128256)
    architecture = StudentArchitectureConfig(hidden_size=pool.hidden_size, vocab_size=int(table.shape[0]), gru_width=256)
    affine_state = _load_affine_state(args.affine_state, hidden_size=pool.hidden_size)
    evidence = load_teacher_evidence(args.teacher_evidence, expected_candidate_k=args.candidate_k) if args.teacher_evidence else None
    candidate_path = args.candidate_ids.expanduser().resolve() if args.candidate_ids else None
    candidate_ids = None
    candidate_metadata: dict[str, str] = {}
    if candidate_path is not None:
        candidate_ids, candidate_metadata = _load_candidates(candidate_path, rows=pool.rows, positions=pool.positions, candidate_k=args.candidate_k)
        if candidate_metadata.get("pool_record_order_sha256") != canonical_hash(list(pool.record_ids)):
            raise P04TrainingError("candidate preparation record order does not match training pool")
        if candidate_metadata.get("pool_observation_sha256") != pool.source_sha256:
            raise P04TrainingError("candidate preparation observations do not match training pool")
        if candidate_metadata.get("embedding_file_sha256") != file_sha256(args.embedding_table):
            raise P04TrainingError("candidate preparation embedding asset does not match training table")
    if evidence is not None and candidate_ids is None:
        raise P04TrainingError("teacher evidence cannot be bound without canonical candidate IDs")
    teacher_scores = teacher_mask = None
    teacher_binding: dict[str, object] = {"enabled": False}
    if evidence is not None:
        candidate_ids, teacher_scores, teacher_mask, teacher_binding = build_teacher_arrays(pool, candidate_ids, evidence)
    if METHOD_D in args.arms and evidence is None:
        raise P04TrainingError("student_d was requested without qualified teacher evidence")
    required_positions = teacher_binding.get("required_positions", {}) if isinstance(teacher_binding, dict) else {}
    if not isinstance(required_positions, dict):
        raise P04TrainingError("teacher evidence required-position map is malformed")
    source_receipt = _source_receipt(args)
    source_receipt.update({
        "replay": {"rows": replay.rows, "positions": replay.positions, "post_bos_positions": replay.post_bos_positions, "record_order_sha256": canonical_hash(list(replay.record_ids))},
        "correction": {"rows": correction.rows, "positions": correction.positions, "post_bos_positions": correction.post_bos_positions, "record_order_sha256": canonical_hash(list(correction.record_ids))},
        "validation": {"rows": validation.rows, "positions": validation.positions, "post_bos_positions": validation.post_bos_positions, "record_order_sha256": canonical_hash(list(validation.record_ids))},
        "candidate_artifact": (
            {
                "enabled": True,
                "path": str(candidate_path),
                "sha256": file_sha256(candidate_path),
                "tensor_sha256": tensor_sha256(candidate_ids),
                "candidate_k": args.candidate_k,
                "schema": candidate_metadata.get("schema"),
                "proposer_id": candidate_metadata.get("proposer_id"),
                "pool_record_order_sha256": candidate_metadata.get("pool_record_order_sha256"),
                "pool_observation_sha256": candidate_metadata.get("pool_observation_sha256"),
            }
            if candidate_path is not None
            else {"enabled": False, "reason": "affine_same_data/student_s do not consume candidate identities"}
        ),
        "teacher_binding": teacher_binding,
        "architecture": asdict(architecture),
        "training_config": vars(args),
    })
    (args.output_root / "source_receipt.json").write_text(json.dumps(source_receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "output_root": str(args.output_root), "geometry": asdict(architecture), "teacher": teacher_binding}, sort_keys=True))
        return 0
    all_results: list[dict[str, object]] = []
    config = TrainingConfig(steps=args.steps, record_batch_size=args.record_batch_size, position_budget=args.position_budget, replay_fraction=args.replay_fraction, validation_every=args.validation_every, learning_rate=args.learning_rate, weight_decay=args.weight_decay, gradient_clip_norm=args.gradient_clip_norm, projection_chunk=args.position_budget)
    for seed in seeds:
        _set_runtime(args, seed=seed)
        schedule = make_position_schedule(pool, replay_records=replay.rows, steps=config.steps, record_batch_size=config.record_batch_size, position_budget=config.position_budget, replay_fraction=config.replay_fraction, seed=seed, required_positions=required_positions)
        seed_root = args.output_root / f"seed-{seed}"
        schedule_receipt = save_schedule(seed_root / "position_schedule.safetensors", schedule, pool=pool, metadata={"source_commit": source_receipt["git_commit"]})
        seed_results: dict[str, object] = {"seed": seed, "schedule": schedule_receipt, "schedule_sha256": schedule.schedule_sha256, "arms": {}}
        for method in args.arms:
            print(f"[P04] seed={seed} arm={method} start", flush=True)
            arm_result = train_arm(method, pool=pool, validation=validation, embedding_table=table, affine_state=affine_state, schedule=schedule, candidate_ids=candidate_ids if method in (METHOD_H, METHOD_D) else None, teacher_scores=teacher_scores if method == METHOD_D else None, teacher_mask=teacher_mask if method == METHOD_D else None, sigma_q=float(teacher_binding["sigma_q"]) if method == METHOD_D and teacher_binding.get("sigma_q") is not None else None, tie_tolerance=float(teacher_binding["tie_tolerance"]) if method == METHOD_D and teacher_binding.get("tie_tolerance") is not None else None, seed=seed, config=config, architecture=architecture, device=torch.device(args.device), output_dir=seed_root / method)
            seed_results["arms"][method] = arm_result
            print(f"[P04] seed={seed} arm={method} done selected_step={arm_result['selected_step']}", flush=True)
        seed_results["wall_seconds"] = time.perf_counter() - started
        (seed_root / "seed_result.json").write_text(json.dumps(seed_results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        all_results.append(seed_results)
    summary = {"schema": TRAINING_SCHEMA, "task_id": TASK_ID, "source_receipt": str((args.output_root / "source_receipt.json").resolve()), "results": all_results, "wall_seconds": time.perf_counter() - started}
    (args.output_root / "training_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_root": str(args.output_root), "wall_seconds": summary["wall_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (P04TrainingError, P04StudentError) as exc:
        print(f"P04 training failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
