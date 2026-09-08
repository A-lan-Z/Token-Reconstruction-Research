"""External fail-closed resource watchdog for the P11 native comparator.

This helper has no torch dependency.  It polls the parent process's RSS and
NVIDIA's system view while the native adapter is inside a record loop, then
writes a create-only receipt.  On a violation it writes the receipt before
sending SIGTERM to the parent.  A parent may stop it cleanly after a cell or a
full run; a wall-time limit is optional and is never implied by default.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


def _rss_bytes(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    try:
        text = status.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1]) * 1024
                except ValueError:
                    return None
    return None


def _nvidia_rows(query: str) -> list[dict[str, str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = [item.strip() for item in query.split(",")]
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(fields):
            raise RuntimeError(f"malformed nvidia-smi row for {query}")
        rows.append(dict(zip(fields, values)))
    if not rows:
        raise RuntimeError(f"nvidia-smi returned no rows for {query}")
    return rows


def _compute_apps() -> list[dict[str, str]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip() or "no running processes" in line.casefold():
            continue
        values = [item.strip() for item in line.split(",")]
        if len(values) >= 3:
            rows.append({"pid": values[0], "process_name": values[1], "used_memory": values[2]})
    return rows


def _write(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, required=True)
    parser.add_argument("--maximum-rss-gib", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--maximum-seconds", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.parent_pid <= 0 or args.poll_seconds <= 0 or args.minimum_free_gib < 0 or args.maximum_rss_gib <= 0:
        raise SystemExit("watchdog limits are malformed")
    minimum_free = int(args.minimum_free_gib * 2**30)
    maximum_rss = int(args.maximum_rss_gib * 2**30)
    started = time.time()
    base: dict[str, Any] = {
        "schema": "token-reconstruction.trr-p11-live-watchdog.v1",
        "task_id": "TRR-P11",
        "parent_pid": args.parent_pid,
        "poll_seconds": args.poll_seconds,
        "minimum_free_gpu_bytes": minimum_free,
        "maximum_host_rss_bytes": maximum_rss,
        "maximum_seconds": args.maximum_seconds,
        "gpu_index": args.gpu_index,
    }
    while True:
        if _rss_bytes(args.parent_pid) is None:
            payload = {**base, "status": "PARENT_EXITED", "ended_unix": time.time()}
            _write(args.output, payload)
            return 0
        now = time.time()
        rss = _rss_bytes(args.parent_pid)
        try:
            gpu_rows = _nvidia_rows("index,memory.free,memory.used,temperature.gpu")
            apps = _compute_apps()
        except Exception as exc:
            payload = {**base, "status": "WATCHDOG_ERROR", "reason": str(exc), "rss_bytes": rss, "ended_unix": now}
            _write(args.output, payload)
            os.kill(args.parent_pid, signal.SIGTERM)
            return 2
        selected = [row for row in gpu_rows if int(row.get("index", -1)) == args.gpu_index]
        if len(selected) != 1:
            payload = {**base, "status": "WATCHDOG_ERROR", "reason": "requested GPU index is absent", "gpu_rows": gpu_rows, "ended_unix": now}
            _write(args.output, payload)
            os.kill(args.parent_pid, signal.SIGTERM)
            return 2
        row = selected[0]
        try:
            free = int(float(row["memory.free"]) * 1024 * 1024)
            temperature = float(row["temperature.gpu"])
        except (KeyError, TypeError, ValueError) as exc:
            payload = {**base, "status": "WATCHDOG_ERROR", "reason": "malformed GPU telemetry", "gpu_rows": gpu_rows, "ended_unix": now}
            _write(args.output, payload)
            os.kill(args.parent_pid, signal.SIGTERM)
            return 2
        other_apps = [item for item in apps if item.get("pid") != str(args.parent_pid)]
        reason = None
        if rss is None:
            reason = "parent RSS became unavailable"
        elif rss > maximum_rss:
            reason = f"host RSS exceeded limit: {rss}"
        elif free < minimum_free:
            reason = f"GPU free memory fell below limit: {free}"
        elif temperature >= 85.0:
            reason = f"GPU temperature exceeded limit: {temperature}"
        elif other_apps:
            reason = f"GPU lost exclusivity: {other_apps!r}"
        elif args.maximum_seconds is not None and now - started > args.maximum_seconds:
            reason = f"optional wall limit exceeded: {now - started:.3f}s"
        if reason is not None:
            payload = {
                **base,
                "status": "RESOURCE_VIOLATION",
                "reason": reason,
                "rss_bytes": rss,
                "gpu": row,
                "other_compute_apps": other_apps,
                "ended_unix": now,
            }
            _write(args.output, payload)
            os.kill(args.parent_pid, signal.SIGTERM)
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
