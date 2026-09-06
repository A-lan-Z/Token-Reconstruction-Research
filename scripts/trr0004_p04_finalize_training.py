#!/usr/bin/env python3
"""Finalize an already completed P04 fit matrix after an aggregate-only CLI failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


TASK_ID = "TRR-P04"
TRAINING_SCHEMA = "token-reconstruction.trr-p04-training.v1"
EXPECTED_SEEDS = (1737, 2711)
EXPECTED_ARMS = ("affine_same_data", "student_s", "student_h", "student_d")


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular artifact is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--run-source-commit", required=True)
    parser.add_argument("--late-failure", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    root = args.training_root.expanduser().resolve()
    source_path = root / "source_receipt.json"
    late_failure_path = args.late_failure.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    late_failure = json.loads(late_failure_path.read_text(encoding="utf-8"))
    if source.get("git_commit") != args.run_source_commit:
        raise RuntimeError(f"run source commit mismatch: {source.get('git_commit')} != {args.run_source_commit}")
    if late_failure.get("status") != "FAIL_CLOSED_LATE_FINALIZATION":
        raise RuntimeError("late finalization receipt does not identify the expected CLI failure")
    if late_failure.get("source_commit") != args.run_source_commit:
        raise RuntimeError("late finalization receipt source commit mismatch")
    config = source.get("training_config")
    if not isinstance(config, Mapping):
        raise RuntimeError("source receipt lacks training configuration")
    if tuple(config.get("seeds", ())) != EXPECTED_SEEDS:
        raise RuntimeError(f"training seeds changed: {config.get('seeds')}")
    if tuple(config.get("arms", ())) != EXPECTED_ARMS:
        raise RuntimeError(f"training arm order changed: {config.get('arms')}")
    if int(config.get("steps", -1)) != 3000:
        raise RuntimeError("training update budget changed")
    if int(config.get("record_batch_size", -1)) != 8 or int(config.get("position_budget", -1)) != 512:
        raise RuntimeError("training geometry changed")

    results: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        seed_root = root / f"seed-{seed}"
        seed_result_path = seed_root / "seed_result.json"
        seed_result = json.loads(seed_result_path.read_text(encoding="utf-8"))
        if int(seed_result.get("seed", -1)) != seed:
            raise RuntimeError(f"seed receipt mismatch under {seed_root}")
        if set(seed_result.get("arms", {})) != set(EXPECTED_ARMS):
            raise RuntimeError(f"arm coverage mismatch for seed {seed}")
        schedule = seed_result.get("schedule")
        if not isinstance(schedule, Mapping):
            raise RuntimeError(f"seed {seed} lacks schedule receipt")
        schedule_desc = descriptor(Path(str(schedule["path"])))
        if schedule_desc["sha256"] != schedule.get("sha256") or schedule_desc["bytes"] != schedule.get("bytes"):
            raise RuntimeError(f"schedule descriptor mismatch for seed {seed}")
        inventory.append({"seed": seed, "kind": "schedule", **schedule_desc})
        for arm in EXPECTED_ARMS:
            result = seed_result["arms"][arm]
            for state_kind in ("selected_state", "final_state"):
                state = result.get(state_kind)
                if not isinstance(state, Mapping):
                    raise RuntimeError(f"seed {seed} arm {arm} lacks {state_kind}")
                state_desc = descriptor(Path(str(state["path"])))
                if state_desc["sha256"] != state.get("sha256") or state_desc["bytes"] != state.get("bytes"):
                    raise RuntimeError(f"{state_kind} descriptor mismatch for seed {seed} arm {arm}")
                inventory.append({"seed": seed, "arm": arm, "kind": state_kind, **state_desc})
            curve_path = Path(str(result.get("learning_curve_path", "")))
            curve_desc = descriptor(curve_path)
            if curve_desc["sha256"] != result.get("learning_curve_sha256"):
                raise RuntimeError(f"learning curve descriptor mismatch for seed {seed} arm {arm}")
            inventory.append({"seed": seed, "arm": arm, "kind": "learning_curve", **curve_desc})
        results.append(seed_result)

    output = root / "training_result.json"
    receipt = root / "training_finalization_receipt.json"
    if output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise RuntimeError("training finalization is create-only")
    summary = {
        "schema": TRAINING_SCHEMA,
        "task_id": TASK_ID,
        "source_receipt": str(source_path),
        "results": results,
        "finalized_after_late_cli_failure": True,
        "late_finalization_failure": descriptor(late_failure_path),
        "run_source_commit": args.run_source_commit,
        "finalizer_source_commit": git_commit(),
        "wall_seconds": time.perf_counter() - started,
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    receipt_payload = {
        "task_id": TASK_ID,
        "schema": "token-reconstruction.trr-p04-training-finalization.v1",
        "status": "PASS",
        "run_source_commit": args.run_source_commit,
        "finalizer_source_commit": git_commit(),
        "training_root": str(root),
        "source_receipt": descriptor(source_path),
        "late_finalization_failure": descriptor(late_failure_path),
        "aggregate": descriptor(output),
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_arms": list(EXPECTED_ARMS),
        "verified_artifact_count": len(inventory),
        "verified_artifacts": inventory,
        "fit_computation_rerun": False,
        "truth_accessed": False,
        "finished_utc": utc(),
        "wall_seconds": time.perf_counter() - started,
    }
    receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "aggregate": descriptor(output), "receipt": descriptor(receipt), "verified_artifact_count": len(inventory)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"P04 training finalization failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
