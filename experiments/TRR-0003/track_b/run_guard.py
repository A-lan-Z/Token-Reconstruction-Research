#!/usr/bin/env python3
"""Run one frozen Track B command while recording source hashes at both ends."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
FROZEN = (
    ROOT / "scripts/trr0003_track_b.py",
    ROOT / "src/token_reconstruction/standalone_decoder.py",
    ROOT / "src/token_reconstruction/inverse.py",
    ROOT / "src/token_reconstruction/experiment_runtime.py",
    ROOT / "experiments/TRR-0003/track_b/plan.json",
    ROOT / "experiments/TRR-0003/track_b/preflight.json",
    ROOT / "experiments/TRR-0003/track_b/prepare_validation_slice.py",
    ROOT / "experiments/TRR-0003/track_b/run_guard.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def snapshot() -> dict[str, Any]:
    return {
        "utc": now(),
        "git_commit": commit(),
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in FROZEN},
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_guard.py <command> [args ...]")
    evidence_path = Path(sys.argv[1]).resolve()
    command = sys.argv[2:]
    if evidence_path.exists() or evidence_path.is_symlink():
        raise SystemExit(f"evidence path must be create-only: {evidence_path}")
    start = snapshot()
    started = time.perf_counter()
    stdout_path = evidence_path.with_name(evidence_path.stem + ".stdout.log")
    stderr_path = evidence_path.with_name(evidence_path.stem + ".stderr.log")
    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    end = snapshot()
    payload = {
        "schema": "token-reconstruction.trr0003-track-b-run-guard.v1",
        "task_id": "TRR-0003",
        "track": "track_b",
        "command": {"argv": command, "cwd": str(ROOT)},
        "start": start,
        "end": end,
        "elapsed_seconds": time.perf_counter() - started,
        "returncode": result.returncode,
        "source_hashes_unchanged": start["source_hashes"] == end["source_hashes"],
        "git_commit_unchanged": start["git_commit"] == end["git_commit"],
        "stdout_log": {"path": str(stdout_path.relative_to(ROOT)), "sha256": sha(stdout_path), "bytes": stdout_path.stat().st_size},
        "stderr_log": {"path": str(stderr_path.relative_to(ROOT)), "sha256": sha(stderr_path), "bytes": stderr_path.stat().st_size},
        "frozen_code_edit_during_run": False,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
