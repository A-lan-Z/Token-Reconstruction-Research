#!/usr/bin/env python3
"""Fail-closed outer launcher for one TRR-P09 fixed-control arm.

This task-local launcher reuses the process-group and host-resource helpers
from :mod:`resource_watchdog`, while adding the two arm-level bounds that the
published watchdog does not provide: free space on every output filesystem
must remain above the declared floor and the combined child/receipt output
footprint must remain below its declared cap.  The child is started in one
new process group and every failure terminates that group.

The wrapper is deliberately agnostic to the numerical child.  It records the
exact child argv and its own guard observations, and it never changes the
child's environment, batching, precision, or model code.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

try:
    from . import resource_watchdog as _watchdog
except ImportError:  # pragma: no cover - exercised when called by absolute path
    from scripts.trr_p09 import resource_watchdog as _watchdog


TASK_ID = "TRR-P09"
DEFAULT_TIMEOUT_SECONDS = 7200.0
DEFAULT_POLL_SECONDS = 0.5
DEFAULT_MAX_RSS_BYTES = 12 * 1024**3
DEFAULT_MIN_AVAILABLE_BYTES = 8 * 1024**3
DEFAULT_MIN_DISK_FREE_BYTES = 20 * 1024**3
DEFAULT_MAX_OUTPUT_BYTES = 5 * 1024**3
DEFAULT_KILL_GRACE_SECONDS = 2.0
WRAPPER_FAILURE_EXIT = _watchdog.WRAPPER_FAILURE_EXIT
TIMEOUT_EXIT = _watchdog.TIMEOUT_EXIT
POST_EXIT_RECHECK_MAX_SECONDS = _watchdog.POST_EXIT_RECHECK_MAX_SECONDS


class GuardError(RuntimeError):
    """Raised when a mandatory launch or telemetry check fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise GuardError(f"create-only receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise GuardError(f"receipt artifact is not a regular file: {path}")
    label = str(path)
    if root is not None:
        try:
            label = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {"path": label, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _create_output_root(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise GuardError(f"output directory must be create-only: {path}")
    path.mkdir(parents=True)
    return path.resolve()


def _assert_create_only_directory(path: Path, *, name: str) -> Path:
    """Validate a future child output directory without creating it."""

    if path.exists() or path.is_symlink():
        raise GuardError(f"{name} must be a new create-only path: {path}")
    parent = path.parent
    while not parent.exists():
        if parent == parent.parent:
            raise GuardError(f"cannot find existing parent for {name}: {path}")
        parent = parent.parent
    if parent.is_symlink() or not parent.is_dir():
        raise GuardError(f"{name} parent is not a regular directory: {parent}")
    return path.absolute()


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate == candidate.parent:
            raise GuardError(f"no existing filesystem path for {path}")
        candidate = candidate.parent
    if candidate.is_symlink():
        candidate = candidate.resolve()
    return candidate


def _disk_free_bytes(path: Path) -> int:
    try:
        usage = shutil.disk_usage(_nearest_existing(path))
    except OSError as exc:
        raise GuardError(f"cannot read disk free space for {path}") from exc
    free = int(usage.free)
    if free < 0:
        raise GuardError(f"disk free space is negative for {path}: {free}")
    return free


def _output_bytes(path: Path) -> int:
    """Return logical bytes below a child output root without following links.

    The output tree is mandatory telemetry while the child is live.  A
    symlink, special file, or disappearing entry is rejected instead of being
    silently skipped, so a child cannot evade the cap through filesystem
    races or redirected output.
    """

    if not path.exists():
        if path.is_symlink():
            raise GuardError(f"output root became a symlink: {path}")
        return 0
    if path.is_symlink() or not path.is_dir():
        raise GuardError(f"output root is not a regular directory: {path}")
    total = 0
    pending = [path]
    while pending:
        current = pending.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise GuardError(f"cannot scan output directory: {current}") from exc
        for entry in entries:
            try:
                if entry.is_symlink():
                    raise GuardError(f"output tree contains a symlink: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    size = int(entry.stat(follow_symlinks=False).st_size)
                    if size < 0:
                        raise GuardError(f"negative output file size: {entry.path}")
                    total += size
                else:
                    raise GuardError(f"output tree contains a special file: {entry.path}")
            except FileNotFoundError as exc:
                raise GuardError(f"output entry disappeared during scan: {entry.path}") from exc
            except OSError as exc:
                raise GuardError(f"cannot inspect output entry: {entry.path}") from exc
    return total


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    try:
        separator = argv.index("--")
    except ValueError as exc:
        raise GuardError("a literal -- must separate wrapper options from the child command") from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--child-output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-rss-bytes", type=int, default=DEFAULT_MAX_RSS_BYTES)
    parser.add_argument("--min-available-bytes", type=int, default=DEFAULT_MIN_AVAILABLE_BYTES)
    parser.add_argument("--min-disk-free-bytes", type=int, default=DEFAULT_MIN_DISK_FREE_BYTES)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument("--kill-grace-seconds", type=float, default=DEFAULT_KILL_GRACE_SECONDS)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--label", default=TASK_ID)
    parser.add_argument("--watchdog-source", type=Path, default=None)
    options = parser.parse_args(argv[:separator])
    command = argv[separator + 1 :]
    if not command:
        raise GuardError("child command after -- is empty")
    numeric = (
        options.timeout_seconds,
        options.poll_seconds,
        options.kill_grace_seconds,
        float(options.max_rss_bytes),
        float(options.min_available_bytes),
        float(options.min_disk_free_bytes),
        float(options.max_output_bytes),
    )
    if any(not math.isfinite(float(value)) for value in numeric):
        raise GuardError("all guard values must be finite")
    if options.timeout_seconds <= 0 or options.poll_seconds <= 0 or options.kill_grace_seconds < 0:
        raise GuardError("timeout, poll, and grace values are invalid")
    if any(value <= 0 for value in (
        options.max_rss_bytes,
        options.min_available_bytes,
        options.min_disk_free_bytes,
        options.max_output_bytes,
    )):
        raise GuardError("resource thresholds must be positive")
    return options, command


def _append_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _monitor_sample(
    *,
    pgid: int,
    child_output_root: Path,
    output_root: Path,
    elapsed_seconds: float,
    require_member: bool,
) -> dict[str, Any]:
    sample = _watchdog._sample(pgid, require_member=require_member, elapsed_seconds=elapsed_seconds)
    output_roots = (child_output_root, output_root)
    output_by_root = {str(path): _output_bytes(path) for path in output_roots}
    disk_by_root = {str(path): _disk_free_bytes(path) for path in output_roots}
    output_bytes = int(sum(output_by_root.values()))
    disk_free = int(min(disk_by_root.values()))
    sample.update(
        {
            "disk_free_bytes": disk_free,
            "disk_free_by_root_bytes": disk_by_root,
            "output_bytes": output_bytes,
            "output_bytes_by_root": output_by_root,
        }
    )
    return sample


def _termination_reason(sample: dict[str, Any], options: argparse.Namespace) -> str | None:
    if int(sample["group_rss_bytes"]) > int(options.max_rss_bytes):
        return "group_rss_limit_exceeded"
    if int(sample["host_mem_available_bytes"]) < int(options.min_available_bytes):
        return "host_mem_available_limit_exceeded"
    if int(sample["disk_free_bytes"]) < int(options.min_disk_free_bytes):
        return "disk_free_limit_exceeded"
    if int(sample["output_bytes"]) > int(options.max_output_bytes):
        return "output_bytes_limit_exceeded"
    return None


def _run(options: argparse.Namespace, command: list[str]) -> int:
    # The declared arm wall cap starts before output-root, binding, and telemetry preparation.
    started = time.monotonic()
    run_start_utc = _utc_now()
    output_root = _create_output_root(options.output_root.absolute())
    child_output_root = _assert_create_only_directory(options.child_output_root.absolute(), name="child output root")
    cwd = options.cwd.resolve() if options.cwd is not None else Path.cwd().resolve()
    if not cwd.is_dir():
        raise GuardError(f"child cwd is not a directory: {cwd}")
    if options.watchdog_source is not None:
        source = options.watchdog_source.resolve()
        if source.is_symlink() or not source.is_file():
            raise GuardError(f"watchdog source is not a regular file: {source}")
        watchdog_source = {"path": str(source), "sha256": _sha256_file(source), "bytes": source.stat().st_size}
    else:
        watchdog_source = None
    child_env = dict(os.environ)
    command_record = {
        "schema": "token-reconstruction.trr-p09-fixed-control-guarded-command.v1",
        "task_id": TASK_ID,
        "label": str(options.label),
        "command": command,
        "shell_command": shlex.join(command),
        "cwd": str(cwd),
        "child_output_root": str(child_output_root),
        "watchdog_source": watchdog_source,
        "thresholds": {
            "timeout_seconds": float(options.timeout_seconds),
            "poll_seconds": float(options.poll_seconds),
            "max_rss_bytes": int(options.max_rss_bytes),
            "min_available_bytes": int(options.min_available_bytes),
            "min_disk_free_bytes": int(options.min_disk_free_bytes),
            "max_output_bytes": int(options.max_output_bytes),
            "kill_grace_seconds": float(options.kill_grace_seconds),
        },
        "process_group": "new_session_start_new_session_true",
        "disk_and_output_scope": "child_output_root_plus_wrapper_output_root",
        "created_utc": _utc_now(),
    }
    command_path = output_root / "command.json"
    stdout_path = output_root / "stdout.txt"
    stderr_path = output_root / "stderr.txt"
    samples_path = output_root / "resource_samples.jsonl"
    _write_json_exclusive(command_path, command_record)
    stdout_handle = stdout_path.open("xb")
    stderr_handle = stderr_path.open("xb")
    samples_handle = samples_path.open("x", encoding="utf-8", newline="\n")

    process: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    start_utc: str | None = run_start_utc
    end_utc: str | None = None
    child_returncode: int | None = None
    termination_reason: str | None = None
    termination_actions: list[str] = []
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    initial_mem_available: int | None = None
    initial_disk_free: int | None = None

    try:
        initial_mem_available = _watchdog._read_mem_available_bytes()
        initial_disk_free = min(_disk_free_bytes(child_output_root), _disk_free_bytes(output_root))
        if initial_mem_available < int(options.min_available_bytes):
            termination_reason = "host_mem_available_limit_exceeded"
            raise GuardError(
                f"initial host MemAvailable below minimum: {initial_mem_available} < {int(options.min_available_bytes)}"
            )
        if initial_disk_free < int(options.min_disk_free_bytes):
            termination_reason = "disk_free_limit_exceeded"
            raise GuardError(
                f"initial disk free below minimum: {initial_disk_free} < {int(options.min_disk_free_bytes)}"
            )
        if time.monotonic() - started >= float(options.timeout_seconds):
            termination_reason = "declared_timeout_exceeded"
            raise GuardError("declared total wall cap elapsed before child launch")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=child_env,
            start_new_session=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        pgid = os.getpgid(process.pid)
        while True:
            elapsed = time.monotonic() - started
            leader_returncode = process.poll()
            group_live = leader_returncode is None
            sample: dict[str, Any] | None = None
            try:
                sample = _monitor_sample(
                    pgid=pgid,
                    child_output_root=child_output_root,
                    output_root=output_root,
                    elapsed_seconds=elapsed,
                    require_member=group_live,
                )
            except (_watchdog.ResourceReadError, GuardError, OSError) as exc:
                before = _watchdog._process_state_for_diagnosis(process.pid)
                recheck_seconds = min(max(0.0, float(options.kill_grace_seconds)), POST_EXIT_RECHECK_MAX_SECONDS)
                leader_returncode = _watchdog._recheck_leader_exit(process, timeout_seconds=recheck_seconds)
                elapsed = time.monotonic() - started
                if leader_returncode is not None:
                    try:
                        sample = _monitor_sample(
                            pgid=pgid,
                            child_output_root=child_output_root,
                            output_root=output_root,
                            elapsed_seconds=elapsed,
                            require_member=False,
                        )
                    except (_watchdog.ResourceReadError, GuardError, OSError) as retry_exc:
                        after = _watchdog._process_state_for_diagnosis(process.pid)
                        try:
                            live_member = _watchdog._has_live_process_group_member(pgid, ignored_pids={process.pid})
                        except _watchdog.ResourceReadError as member_exc:
                            termination_reason = "live_resource_data_unreadable"
                            errors.extend([str(retry_exc), str(member_exc)])
                            termination_actions = _watchdog._terminate_group(
                                process, pgid, options.kill_grace_seconds
                            )
                            break
                        if live_member:
                            termination_reason = "live_resource_data_unreadable"
                            errors.append(
                                f"{retry_exc}; leader_returncode={leader_returncode}; "
                                f"leader_state_after_recheck={after}"
                            )
                            termination_actions = _watchdog._terminate_group(
                                process, pgid, options.kill_grace_seconds
                            )
                            break
                        errors.append(
                            f"post_exit_resource_sample_ignored: {retry_exc}; leader_returncode={leader_returncode}; "
                            f"leader_state_before_recheck={before}; leader_state_after_recheck={after}"
                        )
                        sample = None
                else:
                    termination_reason = "live_resource_data_unreadable"
                    errors.append(
                        f"{exc}; leader_returncode=None; leader_state_before_recheck={before}; "
                        f"leader_state_after_recheck={_watchdog._process_state_for_diagnosis(process.pid)}"
                    )
                    termination_actions = _watchdog._terminate_group(process, pgid, options.kill_grace_seconds)
                    break
            if sample is not None:
                samples.append(sample)
                _append_jsonl(samples_handle, sample)
                termination_reason = _termination_reason(sample, options)
            if termination_reason is None and elapsed >= float(options.timeout_seconds):
                termination_reason = "declared_timeout_exceeded"
            if termination_reason is not None:
                termination_actions = _watchdog._terminate_group(process, pgid, options.kill_grace_seconds)
                break
            if leader_returncode is not None:
                if sample is None or not sample["group_pids"]:
                    break
            remaining = float(options.timeout_seconds) - elapsed
            if remaining <= 0:
                termination_reason = "declared_timeout_exceeded"
                termination_actions = _watchdog._terminate_group(process, pgid, options.kill_grace_seconds)
                break
            time.sleep(min(float(options.poll_seconds), remaining))
    except Exception as exc:
        termination_reason = termination_reason or "guard_exception"
        errors.append(f"{type(exc).__name__}: {exc}")
        if process is not None and process.poll() is None and pgid is not None:
            termination_actions = _watchdog._terminate_group(process, pgid, options.kill_grace_seconds)
    finally:
        if process is not None:
            try:
                if process.poll() is None and pgid is not None:
                    termination_reason = termination_reason or "guard_final_cleanup"
                    termination_actions = termination_actions or _watchdog._terminate_group(
                        process, pgid, options.kill_grace_seconds
                    )
            except Exception as exc:
                errors.append(f"final_cleanup:{type(exc).__name__}: {exc}")
            try:
                child_returncode = process.wait(timeout=max(0.1, options.kill_grace_seconds))
            except subprocess.TimeoutExpired:
                errors.append("child leader did not exit after cleanup")
        end_utc = _utc_now()
        stdout_handle.flush()
        stderr_handle.flush()
        os.fsync(stdout_handle.fileno())
        os.fsync(stderr_handle.fileno())
        samples_handle.flush()
        os.fsync(samples_handle.fileno())
        stdout_handle.close()
        stderr_handle.close()
        samples_handle.close()

    if termination_reason is not None:
        guard_status = "FAIL_CLOSED"
        wrapper_exit = TIMEOUT_EXIT if termination_reason == "declared_timeout_exceeded" else WRAPPER_FAILURE_EXIT
    elif child_returncode not in (0, None):
        guard_status = "CHILD_EXITED_NONZERO"
        wrapper_exit = _watchdog._normalise_child_exit(child_returncode)
    elif child_returncode == 0:
        guard_status = "PASS"
        wrapper_exit = 0
    else:
        guard_status = "CHILD_NOT_STARTED"
        wrapper_exit = WRAPPER_FAILURE_EXIT

    peak_rss = max((int(row["group_rss_bytes"]) for row in samples), default=0)
    min_available = min(
        (int(row["host_mem_available_bytes"]) for row in samples),
        default=initial_mem_available or 0,
    )
    min_disk_free = min((int(row["disk_free_bytes"]) for row in samples), default=initial_disk_free or 0)
    peak_output = max((int(row["output_bytes"]) for row in samples), default=0)
    guard_receipt = {
        "schema": "token-reconstruction.trr-p09-fixed-control-guard.v1",
        "task_id": TASK_ID,
        "status": guard_status,
        "thresholds": command_record["thresholds"],
        "initial_host_mem_available_bytes": initial_mem_available,
        "initial_disk_free_bytes": initial_disk_free,
        "sample_count": len(samples),
        "peak_group_rss_bytes": peak_rss,
        "minimum_sampled_host_mem_available_bytes": min_available,
        "minimum_sampled_disk_free_bytes": min_disk_free,
        "peak_sampled_output_bytes": peak_output,
        "termination_reason": termination_reason,
        "termination_actions": termination_actions,
        "errors": errors,
        "samples_path": "resource_samples.jsonl",
        "samples": samples,
        "process_group_fail_closed": True,
        "disk_and_output_enforced_during_live_process": True,
        "wall_includes_preparation": True,
    }
    guard_path = output_root / "resource_guard.json"
    _write_json_exclusive(guard_path, guard_receipt)
    elapsed_seconds = round(time.monotonic() - started, 6) if start_utc is not None else None
    time_receipt = {
        "schema": "token-reconstruction.trr-p09-fixed-control-guard-time.v1",
        "task_id": TASK_ID,
        "status": guard_status,
        "command": command,
        "cwd": str(cwd),
        "start_utc": start_utc,
        "end_utc": end_utc,
        "elapsed_seconds": elapsed_seconds,
        "timeout_seconds": float(options.timeout_seconds),
        "child_pid": process.pid if process is not None else None,
        "process_group_id": pgid,
        "child_return_code": child_returncode,
        "wrapper_exit_code": wrapper_exit,
        "termination_reason": termination_reason,
        "termination_actions": termination_actions,
        "stdout": "stdout.txt",
        "stderr": "stderr.txt",
        "resource_guard": "resource_guard.json",
    }
    time_path = output_root / "time.json"
    _write_json_exclusive(time_path, time_receipt)
    finish_path = output_root / "finish.json"
    finish = {
        "schema": "token-reconstruction.trr-p09-fixed-control-guard-finish.v1",
        "task_id": TASK_ID,
        "status": guard_status,
        "wrapper_exit_code": wrapper_exit,
        "child_return_code": child_returncode,
        "termination_reason": termination_reason,
        "command": _file_record(command_path, root=output_root),
        "stdout": _file_record(stdout_path, root=output_root),
        "stderr": _file_record(stderr_path, root=output_root),
        "samples": _file_record(samples_path, root=output_root),
        "guard": _file_record(guard_path, root=output_root),
        "time": _file_record(time_path, root=output_root),
    }
    _write_json_exclusive(finish_path, finish)
    print(
        json.dumps(
            {
                "status": guard_status,
                "wrapper_exit_code": wrapper_exit,
                "child_return_code": child_returncode,
                "termination_reason": termination_reason,
                "output_root": str(output_root),
            },
            sort_keys=True,
        )
    )
    return wrapper_exit


def main(argv: list[str] | None = None) -> int:
    options, command = _parse_args(list(sys.argv[1:] if argv is None else argv))
    return _run(options, command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GuardError, _watchdog.WatchdogError) as exc:
        print(f"TRR-P09 fixed-control guard failed before launch: {exc}", file=sys.stderr)
        raise SystemExit(WRAPPER_FAILURE_EXIT)
