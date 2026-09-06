#!/usr/bin/env python3
"""Run one command in an isolated process group with fail-closed resource guards.

The command is everything after a literal ``--``.  The wrapper starts a new
session, captures stdout/stderr, samples the complete process group and host
``MemAvailable``, and terminates the group when a declared bound is exceeded.
Resource data are treated as mandatory while the group is live: an unreadable
``/proc`` or ``/proc/meminfo`` sample also terminates the group.

The output directory is create-only and contains:

* ``command.json``: exact command, cwd, child environment, and thresholds;
* ``stdout.txt`` and ``stderr.txt``: byte-preserving child streams;
* ``resource_samples.jsonl``: periodic group RSS and host availability;
* ``resource_guard.json``: thresholds, samples, termination reason, and peak;
* ``time.json``: start/end timestamps, exit codes, and termination actions; and
* ``finish.json``: a compact receipt with hashes for the preceding artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Iterable

TASK_ID = "TRR-P06"
BYTES_PER_KIB = 1024
DEFAULT_MAX_RSS_BYTES = 8 * 1024**3
DEFAULT_MIN_AVAILABLE_BYTES = 10 * 1024**3
DEFAULT_TIMEOUT_SECONDS = 3600.0
DEFAULT_POLL_SECONDS = 0.5
DEFAULT_KILL_GRACE_SECONDS = 2.0
WRAPPER_FAILURE_EXIT = 125
TIMEOUT_EXIT = 124
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


class WatchdogError(RuntimeError):
    """Raised when the wrapper cannot safely observe or run the child."""


class ResourceReadError(WatchdogError):
    """Raised when mandatory live resource data are unavailable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise WatchdogError(f"receipt artifact is not a regular file: {path}")
    label = str(path)
    if root is not None:
        try:
            label = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {"path": label, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise WatchdogError(f"create-only receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _create_output_root(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise WatchdogError(f"output directory must be create-only: {path}")
    path.mkdir(parents=True)
    return path.resolve()


def _read_mem_available_bytes() -> int:
    path = Path("/proc/meminfo")
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ResourceReadError(f"cannot read live host memory data: {path}") from exc
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if key != "MemAvailable" or not separator:
            continue
        fields = value.strip().split()
        if not fields:
            break
        try:
            number = int(fields[0])
        except ValueError:
            break
        unit = fields[1].lower() if len(fields) > 1 else "kb"
        multiplier = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}.get(unit)
        if multiplier is None or number <= 0:
            break
        return number * multiplier
    raise ResourceReadError("live host memory data lacks a positive MemAvailable value")


def _pid_group(pid: int) -> int:
    path = Path("/proc") / str(pid) / "stat"
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ResourceReadError(f"cannot read live process stat: {path}") from exc
    # The comm field may contain spaces and ')' characters.  Everything after
    # its final ')' starts with state (field 3); pgrp is the third token there.
    try:
        fields = text.rsplit(")", 1)[1].strip().split()
        return int(fields[2])
    except (IndexError, ValueError) as exc:
        raise ResourceReadError(f"live process stat is malformed: {path}") from exc


def _process_group_members(pgid: int, *, require_member: bool) -> list[dict[str, int]]:
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError as exc:
        raise ResourceReadError("cannot enumerate live process table") from exc
    members: list[dict[str, int]] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if _pid_group(pid) != pgid:
                continue
        except ResourceReadError as exc:
            # A process can disappear between enumeration and stat.  A
            # disappearing member is safe to ignore; an unreadable leader is
            # handled by the caller through the subprocess poll state.
            if not entry.exists():
                continue
            raise exc
        status_path = entry / "status"
        try:
            status = status_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            # /proc can remove a just-exited member between directory
            # enumeration and the status read. Retry once before declaring a
            # live-resource failure; a member whose directory disappears is
            # safe to omit, while a still-live member with unreadable status
            # remains fail-closed.
            time.sleep(0.02)
            try:
                status = status_path.read_text(encoding="ascii")
            except (OSError, UnicodeError) as retry_exc:
                if not entry.exists():
                    continue
                raise ResourceReadError(f"cannot read live process RSS: {status_path}") from retry_exc
        rss_bytes: int | None = None
        for line in status.splitlines():
            if not line.startswith("VmRSS:"):
                continue
            fields = line.split()
            if len(fields) < 2:
                break
            try:
                rss_kib = int(fields[1])
            except ValueError:
                break
            if rss_kib < 0:
                break
            rss_bytes = rss_kib * BYTES_PER_KIB
            break
        if rss_bytes is None:
            # A just-exited child can remain as a zombie until Popen reaps it;
            # zombies have no live RSS to observe and must not turn a normal
            # successful child exit into a watchdog failure.  A non-zombie
            # process with missing VmRSS remains an unreadable live resource.
            stat_path = entry / "stat"
            try:
                stat_text = stat_path.read_text(encoding="ascii")
            except (OSError, UnicodeError) as exc:
                if not entry.exists():
                    continue
                raise ResourceReadError(f"cannot read live process state: {stat_path}") from exc
            try:
                state = stat_text.rsplit(")", 1)[1].strip().split()[0]
            except (IndexError, ValueError) as exc:
                raise ResourceReadError(f"live process stat is malformed: {stat_path}") from exc
            if state == "Z":
                continue
            raise ResourceReadError(f"live process RSS is missing: {status_path}")
        members.append({"pid": pid, "rss_bytes": rss_bytes})
    if require_member and not members:
        raise ResourceReadError(f"process group {pgid} has no readable live members")
    return sorted(members, key=lambda row: row["pid"])


def _sample(pgid: int, *, require_member: bool, elapsed_seconds: float) -> dict[str, Any]:
    available = _read_mem_available_bytes()
    members = _process_group_members(pgid, require_member=require_member)
    return {
        "timestamp_utc": _utc_now(),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "host_mem_available_bytes": int(available),
        "group_rss_bytes": int(sum(row["rss_bytes"] for row in members)),
        "group_pids": [int(row["pid"]) for row in members],
        "group_member_rss_bytes": {str(row["pid"]): int(row["rss_bytes"]) for row in members},
    }


def _append_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _terminate_group(process: subprocess.Popen[bytes], pgid: int, grace_seconds: float) -> list[str]:
    actions: list[str] = []
    try:
        os.killpg(pgid, signal.SIGTERM)
        actions.append("SIGTERM_process_group")
    except ProcessLookupError:
        actions.append("SIGTERM_process_group_already_gone")
    except OSError as exc:
        actions.append(f"SIGTERM_process_group_error:{type(exc).__name__}")
    try:
        process.wait(timeout=max(0.0, grace_seconds))
    except subprocess.TimeoutExpired:
        pass
    if process.poll() is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            actions.append("SIGKILL_process_group")
        except ProcessLookupError:
            actions.append("SIGKILL_process_group_already_gone")
        except OSError as exc:
            actions.append(f"SIGKILL_process_group_error:{type(exc).__name__}")
        try:
            process.wait(timeout=max(0.1, grace_seconds))
        except subprocess.TimeoutExpired:
            actions.append("leader_wait_timeout_after_SIGKILL")
    return actions


def _normalise_child_exit(returncode: int | None) -> int:
    if returncode is None:
        return WRAPPER_FAILURE_EXIT
    if returncode < 0:
        return 128 + (-returncode)
    return int(returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-rss-bytes", type=int, default=DEFAULT_MAX_RSS_BYTES)
    parser.add_argument("--min-available-bytes", type=int, default=DEFAULT_MIN_AVAILABLE_BYTES)
    parser.add_argument("--kill-grace-seconds", type=float, default=DEFAULT_KILL_GRACE_SECONDS)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--label", default=TASK_ID)
    return parser


def _parse_invocation(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    try:
        separator = argv.index("--")
    except ValueError as exc:
        # Preserve normal argparse help behavior even though the child command
        # itself must be separated by a literal --.
        if "-h" in argv or "--help" in argv:
            _parser().parse_args(argv)
        raise WatchdogError("a literal -- must separate wrapper options from the child command") from exc
    options = _parser().parse_args(argv[:separator])
    command = argv[separator + 1 :]
    if not command:
        raise WatchdogError("child command after -- is empty")
    return options, command


def main(argv: list[str] | None = None) -> int:
    options, command = _parse_invocation(list(sys.argv[1:] if argv is None else argv))
    if options.timeout_seconds <= 0 or options.poll_seconds <= 0 or options.kill_grace_seconds < 0:
        raise WatchdogError("timeout, poll, and grace values are invalid")
    if options.max_rss_bytes <= 0 or options.min_available_bytes <= 0:
        raise WatchdogError("resource thresholds must be positive")
    # Check the user-supplied final path before canonicalisation so a dangling
    # symlink cannot redirect a supposedly create-only output directory.
    if options.output_root.exists() or options.output_root.is_symlink():
        raise WatchdogError(f"output directory must be create-only: {options.output_root}")
    output_root = _create_output_root(options.output_root.absolute())
    cwd = options.cwd.resolve() if options.cwd is not None else Path.cwd().resolve()
    if not cwd.is_dir():
        raise WatchdogError(f"child cwd is not a directory: {cwd}")
    child_env = dict(os.environ)
    command_record = {
        "schema": "token-reconstruction.trr-p06-resource-watchdog-command.v1",
        "task_id": TASK_ID,
        "label": str(options.label),
        "command": command,
        "shell_command": shlex.join(command),
        "cwd": str(cwd),
        # The child inherits the full environment for reproducibility of the
        # execution, but receipts persist only this explicit safe allowlist.
        "environment": {
            key: child_env[key] for key in SAFE_ENVIRONMENT_KEYS if key in child_env
        },
        "environment_allowlist": list(SAFE_ENVIRONMENT_KEYS),
        "environment_redacted": True,
        "thresholds": {
            "timeout_seconds": float(options.timeout_seconds),
            "poll_seconds": float(options.poll_seconds),
            "max_rss_bytes": int(options.max_rss_bytes),
            "min_available_bytes": int(options.min_available_bytes),
            "kill_grace_seconds": float(options.kill_grace_seconds),
        },
        "process_group": "new_session_start_new_session_true",
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
    start_utc: str | None = None
    end_utc: str | None = None
    started = time.monotonic()
    child_returncode: int | None = None
    termination_reason: str | None = None
    termination_actions: list[str] = []
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    initial_mem_available: int | None = None

    try:
        # A host resource read is mandatory before launching the child.  This
        # prevents an unobservable job from starting under a false green state.
        initial_mem_available = _read_mem_available_bytes()
        if initial_mem_available < int(options.min_available_bytes):
            termination_reason = "host_mem_available_limit_exceeded"
            raise WatchdogError(
                "initial host MemAvailable is below the declared minimum: "
                f"available={initial_mem_available} required={int(options.min_available_bytes)}"
            )
        start_utc = _utc_now()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=child_env,
            start_new_session=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        pgid = os.getpgid(process.pid)
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            leader_returncode = process.poll()
            group_live = leader_returncode is None
            try:
                if group_live:
                    sample = _sample(pgid, require_member=True, elapsed_seconds=elapsed)
                else:
                    sample = _sample(pgid, require_member=False, elapsed_seconds=elapsed)
            except ResourceReadError as exc:
                # A child can disappear between the poll and /proc status
                # reads.  Re-poll before treating that race as an unreadable
                # live resource.  If the leader has exited, descendants are
                # still sampled with require_member=False; only a genuinely
                # live unreadable group fails closed.
                leader_returncode = process.poll()
                if leader_returncode is None:
                    # Reap a child that exited between poll and /proc reads.
                    # A zero-time wait never masks a live unreadable process.
                    try:
                        leader_returncode = process.wait(timeout=0)
                    except subprocess.TimeoutExpired:
                        pass
                if leader_returncode is not None:
                    try:
                        sample = _sample(pgid, require_member=False, elapsed_seconds=elapsed)
                    except ResourceReadError as retry_exc:
                        if "no readable live members" in str(retry_exc):
                            sample = None
                        else:
                            termination_reason = "live_resource_data_unreadable"
                            errors.append(str(retry_exc))
                            termination_actions = _terminate_group(process, pgid, options.kill_grace_seconds)
                            break
                else:
                    termination_reason = "live_resource_data_unreadable"
                    errors.append(str(exc))
                    termination_actions = _terminate_group(process, pgid, options.kill_grace_seconds)
                    break
            if sample is not None:
                samples.append(sample)
                _append_jsonl(samples_handle, sample)
                if sample["group_rss_bytes"] > int(options.max_rss_bytes):
                    termination_reason = "group_rss_limit_exceeded"
                elif sample["host_mem_available_bytes"] < int(options.min_available_bytes):
                    termination_reason = "host_mem_available_limit_exceeded"
            if termination_reason is None and elapsed >= float(options.timeout_seconds):
                termination_reason = "declared_timeout_exceeded"
            if termination_reason is not None:
                termination_actions = _terminate_group(process, pgid, options.kill_grace_seconds)
                break
            if leader_returncode is not None:
                # Keep the session alive until descendants leave the process
                # group, so a detached child cannot escape the guard.
                if sample is None or not sample["group_pids"]:
                    break
            remaining = float(options.timeout_seconds) - elapsed
            if remaining <= 0:
                termination_reason = "declared_timeout_exceeded"
                termination_actions = _terminate_group(process, pgid, options.kill_grace_seconds)
                break
            time.sleep(min(float(options.poll_seconds), remaining))
    except Exception as exc:
        termination_reason = termination_reason or "watchdog_exception"
        errors.append(f"{type(exc).__name__}: {exc}")
        if process is not None and process.poll() is None and pgid is not None:
            termination_actions = _terminate_group(process, pgid, options.kill_grace_seconds)
    finally:
        if process is not None:
            try:
                if process.poll() is None and pgid is not None:
                    termination_reason = termination_reason or "watchdog_final_cleanup"
                    termination_actions = termination_actions or _terminate_group(process, pgid, options.kill_grace_seconds)
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
        wrapper_exit = _normalise_child_exit(child_returncode)
    elif child_returncode == 0:
        guard_status = "PASS"
        wrapper_exit = 0
    else:
        guard_status = "CHILD_NOT_STARTED"
        wrapper_exit = WRAPPER_FAILURE_EXIT

    peak_rss = max((int(sample["group_rss_bytes"]) for sample in samples), default=0)
    min_available = min((int(sample["host_mem_available_bytes"]) for sample in samples), default=initial_mem_available or 0)
    guard_receipt = {
        "schema": "token-reconstruction.trr-p06-resource-watchdog-guard.v1",
        "task_id": TASK_ID,
        "status": guard_status,
        "thresholds": command_record["thresholds"],
        "initial_host_mem_available_bytes": initial_mem_available,
        "sample_count": len(samples),
        "peak_group_rss_bytes": peak_rss,
        "minimum_sampled_host_mem_available_bytes": min_available,
        "termination_reason": termination_reason,
        "termination_actions": termination_actions,
        "errors": errors,
        "samples_path": "resource_samples.jsonl",
        "samples": samples,
    }
    guard_path = output_root / "resource_guard.json"
    _write_json_exclusive(guard_path, guard_receipt)
    elapsed_seconds = None
    if start_utc is not None:
        # Monotonic elapsed is the authoritative timing measure.  This value
        # includes cleanup, while command.json retains the wall-clock start.
        elapsed_seconds = round(time.monotonic() - started, 6)
    time_receipt = {
        "schema": "token-reconstruction.trr-p06-resource-watchdog-time.v1",
        "task_id": TASK_ID,
        "status": guard_status,
        "command_receipt": "command.json",
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
        "schema": "token-reconstruction.trr-p06-resource-watchdog-finish.v1",
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
    print(json.dumps({
        "status": guard_status,
        "wrapper_exit_code": wrapper_exit,
        "child_return_code": child_returncode,
        "termination_reason": termination_reason,
        "output_root": str(output_root),
    }, sort_keys=True))
    return wrapper_exit


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WatchdogError as exc:
        print(f"TRR-P06 resource watchdog failed before launch: {exc}", file=sys.stderr)
        raise SystemExit(WRAPPER_FAILURE_EXIT)
