#!/usr/bin/env python3
"""Run one frozen Track B command with source and resource guardrails."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_PATH = ROOT / "experiments/TRR-0003/track_b/preflight.json"
FROZEN = (
    ROOT / "scripts/trr0003_track_b.py",
    ROOT / "src/token_reconstruction/standalone_decoder.py",
    ROOT / "src/token_reconstruction/inverse.py",
    ROOT / "src/token_reconstruction/experiment_runtime.py",
    ROOT / "src/token_reconstruction/footing.py",
    ROOT / "experiments/TRR-0003/track_b/plan.json",
    ROOT / "experiments/TRR-0003/track_b/preflight.json",
    ROOT / "experiments/TRR-0003/track_b/prepare_validation_slice.py",
    ROOT / "experiments/TRR-0003/track_b/checkpoint_selection_amendment.json",
    ROOT / "experiments/TRR-0003/track_b/replay_selected.py",
    ROOT / "experiments/TRR-0003/track_b/predict_cells.py",
    ROOT / "experiments/TRR-0003/track_b/analyze_token_transfer.py",
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


def _available_host_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host MemAvailable is unavailable")


def _live_gpu() -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one exclusive GPU, observed {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"unexpected nvidia-smi geometry: {rows[0]!r}")
    try:
        total_mib, free_mib, used_mib, temperature_c, utilization_pct = (
            int(float(fields[index])) for index in range(1, 6)
        )
    except ValueError as exc:
        raise RuntimeError(f"nvidia-smi numeric fields are invalid: {rows[0]!r}") from exc
    process_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    processes = [line.strip() for line in process_output.splitlines() if line.strip()]
    return {
        "name": fields[0],
        "total_bytes": total_mib * 1024 * 1024,
        "free_bytes": free_mib * 1024 * 1024,
        "used_bytes": used_mib * 1024 * 1024,
        "temperature_c": temperature_c,
        "utilization_pct": utilization_pct,
        "compute_processes": processes,
    }


def resource_preflight() -> dict[str, Any]:
    """Require live margins for the largest declared preparation/training cell."""

    try:
        preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
        rule = preflight["qualification_rule"]
        margin = float(rule["minimum_margin_fraction"])
        preparation = preflight["public_preparation_peak_envelope"]
        decoder = preflight["predicted_peak_envelope"]
        gpu_envelope = max(
            int(preparation["gpu_envelope_bytes"]),
            int(decoder["affine_training_bytes"]),
        )
        host_envelope = int(preparation["host_envelope_bytes"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"resource preflight configuration is invalid: {exc}") from exc
    if not 0.0 <= margin < 1.0:
        raise RuntimeError("resource preflight margin is invalid")
    live_gpu = _live_gpu()
    live_host = _available_host_bytes()
    gpu_required = math.ceil(gpu_envelope / (1.0 - margin))
    host_required = math.ceil(host_envelope / (1.0 - margin))
    checks = {
        "gpu_margin_pass": live_gpu["free_bytes"] >= gpu_required,
        "host_margin_pass": live_host >= host_required,
        "thermal_pass": live_gpu["temperature_c"] < 85,
        "exclusive_gpu_pass": not live_gpu["compute_processes"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            "live resource margin failed: "
            + json.dumps(
                {
                    "checks": checks,
                    "gpu_free_bytes": live_gpu["free_bytes"],
                    "gpu_required_bytes": gpu_required,
                    "host_available_bytes": live_host,
                    "host_required_bytes": host_required,
                    "temperature_c": live_gpu["temperature_c"],
                    "compute_processes": live_gpu["compute_processes"],
                },
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "minimum_margin_fraction": margin,
        "predicted_gpu_envelope_bytes": gpu_envelope,
        "predicted_host_envelope_bytes": host_envelope,
        "required_gpu_free_bytes": gpu_required,
        "required_host_available_bytes": host_required,
        "live_gpu": live_gpu,
        "live_host_available_bytes": live_host,
        "checks": checks,
    }


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_guard.py <evidence.json> <command> [args ...]")
    evidence_path = Path(sys.argv[1]).resolve()
    command = sys.argv[2:]
    if evidence_path.exists() or evidence_path.is_symlink():
        raise SystemExit(f"evidence path must be create-only: {evidence_path}")
    stdout_path = evidence_path.with_name(evidence_path.stem + ".stdout.log")
    stderr_path = evidence_path.with_name(evidence_path.stem + ".stderr.log")
    if stdout_path.exists() or stdout_path.is_symlink() or stderr_path.exists() or stderr_path.is_symlink():
        raise SystemExit("run logs must be create-only")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    start = snapshot()
    started = time.perf_counter()
    resource: dict[str, Any]
    resource_error: str | None = None
    try:
        resource = resource_preflight()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        resource = {"status": "BLOCKED", "error": str(exc)}
        resource_error = str(exc)

    command_executed = resource_error is None
    result_returncode: int | None = None
    if command_executed:
        with stdout_path.open("x", encoding="utf-8", newline="") as stdout, stderr_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            result_returncode = result.returncode
    else:
        stdout_path.write_text("", encoding="utf-8", newline="\n")
        stderr_path.write_text(f"run guard blocked before command: {resource_error}\n", encoding="utf-8", newline="\n")
    end = snapshot()
    source_hashes_unchanged = start["source_hashes"] == end["source_hashes"]
    git_commit_unchanged = start["git_commit"] == end["git_commit"]
    integrity_pass = source_hashes_unchanged and git_commit_unchanged
    guard_passed = command_executed and result_returncode == 0 and integrity_pass and resource["status"] == "PASS"
    if guard_passed:
        returncode = 0
    elif not integrity_pass:
        returncode = 3
    elif result_returncode not in (None, 0):
        returncode = int(result_returncode)
    else:
        returncode = 3
    payload = {
        "schema": "token-reconstruction.trr0003-track-b-run-guard.v2",
        "task_id": "TRR-0003",
        "track": "track_b",
        "command": {"argv": command, "cwd": str(ROOT)},
        "start": start,
        "end": end,
        "elapsed_seconds": time.perf_counter() - started,
        "resource_preflight": resource,
        "command_executed": command_executed,
        "command_returncode": result_returncode,
        "returncode": returncode,
        "source_hashes_unchanged": source_hashes_unchanged,
        "git_commit_unchanged": git_commit_unchanged,
        "frozen_code_edit_during_run": not integrity_pass,
        "guard_passed": guard_passed,
        "stdout_log": {"path": _path_label(stdout_path), "sha256": sha(stdout_path), "bytes": stdout_path.stat().st_size},
        "stderr_log": {"path": _path_label(stderr_path), "sha256": sha(stderr_path), "bytes": stderr_path.stat().st_size},
    }
    _write_json(evidence_path, payload)
    print(json.dumps(payload, indent=2))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
