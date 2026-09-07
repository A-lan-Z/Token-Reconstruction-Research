"""Fail-closed P08 score launcher for use only after root freeze approval.

This wrapper does not materialize truth. It requires the caller to provide an
already-created private truth manifest outside the repository and an immutable
joint-freeze receipt. It records wall-clock/provenance metadata outside the
repository, while score_frozen.py writes only the metrics/provenance JSON under
the task runtime. Do not run until root explicitly grants the truth gate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path("/tmp/trr-p08").resolve()
PLAN_SHA = "9d1eb8dca89c76f4064c0c380636cb9d585672f7fae796d5295d6507b788caa3"
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 8080


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def outside_repo(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(REPO)
    except ValueError:
        return
    raise SystemExit(f"{label} must be outside repository: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="explicit post-root-grant acknowledgement")
    parser.add_argument("--joint-freeze", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing score: pass --execute only after root verifies the joint freeze and truth manifest")
    if args.bootstrap_draws != DEFAULT_DRAWS or args.bootstrap_seed != DEFAULT_SEED:
        raise SystemExit("P08 bootstrap must remain 10,000 draws with seed 8080")
    freeze = args.joint_freeze.expanduser().resolve()
    truth = args.truth_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    if not freeze.is_file():
        raise SystemExit(f"joint freeze is unavailable: {freeze}")
    if not truth.is_file():
        raise SystemExit(f"truth manifest is unavailable: {truth}")
    outside_repo(truth, "truth manifest")
    outside_repo(receipt_path, "run receipt")
    try:
        output.relative_to(REPO)
    except ValueError as exc:
        raise SystemExit("score output must be task-owned under the repository") from exc
    if output.exists() or receipt_path.exists():
        raise SystemExit("score output and receipt are create-only")
    command = [
        sys.executable,
        "scripts/trr_p08/score_frozen.py",
        "--repository-root", str(REPO),
        "--freeze-receipt", str(freeze),
        "--truth-manifest", str(truth),
        "--output", str(output),
        "--expected-plan-sha256", PLAN_SHA,
        "--bootstrap-draws", str(args.bootstrap_draws),
        "--bootstrap-seed", str(args.bootstrap_seed),
    ]
    env = os.environ.copy()
    env.update({"PYTHONPATH": ".:src", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    started = utc()
    clock = time.perf_counter()
    proc = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True)
    ended = utc()
    receipt = {
        "schema": "token-reconstruction.trr-p08-score-execution.v1",
        "task_id": "TRR-P08",
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "started_utc": started,
        "ended_utc": ended,
        "elapsed_seconds": round(time.perf_counter() - clock, 6),
        "command": command,
        "environment": {"python": sys.executable, "PYTHONPATH": ".:src", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "platform": platform.platform()},
        "joint_freeze": {"path": str(freeze), "sha256": sha(freeze)},
        "truth_manifest": {"path": str(truth), "outside_repository": True, "sha256": sha(truth)},
        "output": record(output) if proc.returncode == 0 and output.is_file() else {"path": str(output)},
        "bootstrap": {"draws": args.bootstrap_draws, "seed": args.bootstrap_seed, "unit": "source-record cluster", "target_pairing": "same source-index schedule within domain"},
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "returncode": proc.returncode,
        "truth_arrays_persisted_in_repository": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path), "elapsed_seconds": receipt["elapsed_seconds"]}, sort_keys=True))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
