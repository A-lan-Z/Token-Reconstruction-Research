#!/usr/bin/env python3
"""Build or execute a guarded TRR-P09 STAGE1 capture command.

This helper deliberately does not mint authorization or watchdog receipts and
never invents lease timestamps.  In its default mode it prints the exact
child/wrapper argv and enforcement map.  ``--execute`` dispatches the already
bound P06 process-group watchdog after checking the signed executable identity
and the supplied watchdog receipt's exact command.  The child capture script
performs the final plan, input, authorization, lease, model, and inner GPU /
retained-output checks before loading a model.

The optional ``--estimate-qualification`` mode consumes a real passing
qualification receipt and an explicitly supplied fixed-overhead estimate.  It
prints a finite production-wall estimate for the root to place in a separate
capture authorization; it writes no receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


TASK_ID = "TRR-P09"
PLAN_SHA256 = "bca93d6099760791c5ecbf3be8177d250cc672c62b2177a3d72d79b934c3826c"
WRAPPER_PATH = Path(__file__).resolve().with_name("resource_watchdog.py")
WRAPPER_BYTES = 29785
WRAPPER_SHA256 = "07691ea3d8c86e2bea56a85a304ac6952ad14272a146af9cea5f0e5401a42b22"
QUALIFICATION_WALL_SECONDS = 600
PRODUCTION_WALL_SECONDS = 7200
FORWARD_BATCH_RECORDS = 8
NEW_RECORDS = 10800
PRODUCTION_FORWARD_CALLS = NEW_RECORDS // FORWARD_BATCH_RECORDS
GIB = 2**30
MAX_RSS_BYTES = 12 * GIB
MIN_HOST_AVAILABLE_RUNTIME_BYTES = 6 * GIB
MIN_HOST_AVAILABLE_PRELAUNCH_BYTES = 10 * GIB
MAX_GPU_RESERVED_BYTES = 8 * GIB
MIN_GPU_FREE_PRELAUNCH_BYTES = 8 * GIB
MIN_GPU_FREE_RUNTIME_BYTES = 2 * GIB
MAX_RETAINED_BYTES = 20 * GIB


class LaunchError(RuntimeError):
    """Raised when a launch command or receipt binding is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise LaunchError(f"required regular file is unavailable: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _require_finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LaunchError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise LaunchError(f"{label} is not finite or below its minimum")
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"{label} is not a JSON object: {path}")
    return value


def estimate_production_wall(
    qualification_path: Path,
    *,
    fixed_overhead_seconds: float,
    production_forward_calls: int = PRODUCTION_FORWARD_CALLS,
) -> dict[str, Any]:
    """Scale measured qualification forwards plus an explicit overhead estimate."""

    receipt = _load_json(qualification_path, label="qualification receipt")
    if receipt.get("status") != "QUALIFICATION_PASS":
        raise LaunchError("qualification receipt is not QUALIFICATION_PASS")
    if receipt.get("condition") != "public_base" or receipt.get("truth_opened") is not False:
        raise LaunchError("qualification receipt condition/truth boundary is invalid")
    batches = receipt.get("batches")
    if not isinstance(batches, list) or not batches or len(batches) > 3:
        raise LaunchError("qualification receipt lacks a bounded batch list")
    observed_forward_calls = 0
    future_padding_tested_batches = 0
    for index, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise LaunchError(f"qualification batch {index} is malformed")
        if batch.get("repeat_torch_equal") is not True:
            raise LaunchError(f"qualification batch {index} lacks exact repeat equivalence")
        padding_tested = batch.get("future_padding_tested")
        if not isinstance(padding_tested, bool):
            raise LaunchError(f"qualification batch {index} lacks future-padding test metadata")
        expected_calls = 3 if padding_tested else 2
        if padding_tested:
            if batch.get("future_padding_active_torch_equal") is not True:
                raise LaunchError(f"qualification batch {index} lacks exact future-padding equivalence")
            future_padding_tested_batches += 1
        elif batch.get("future_padding_active_torch_equal") is not None:
            raise LaunchError(f"qualification batch {index} has inconsistent future-padding metadata")
        if int(batch.get("forward_call_count", -1)) != expected_calls:
            raise LaunchError(f"qualification batch {index} has an incorrect forward call count")
        observed_forward_calls += expected_calls
    if future_padding_tested_batches < 1:
        raise LaunchError("qualification receipt lacks a future-padding representative")
    if int(receipt.get("future_padding_tested_batches", -1)) != future_padding_tested_batches:
        raise LaunchError("qualification receipt future-padding count is inconsistent")
    if int(receipt.get("forward_call_count", -1)) != observed_forward_calls:
        raise LaunchError("qualification receipt forward call count is inconsistent")
    elapsed = _require_finite(receipt.get("elapsed_seconds"), label="qualification elapsed_seconds", minimum=0.0)
    if elapsed <= 0.0:
        raise LaunchError("qualification elapsed_seconds must be positive")
    overhead = _require_finite(fixed_overhead_seconds, label="fixed overhead seconds", minimum=0.0)
    qualification_forward_calls = observed_forward_calls
    scaled_forward_seconds = elapsed * (float(production_forward_calls) / float(qualification_forward_calls))
    estimated = scaled_forward_seconds + overhead
    if not math.isfinite(estimated):
        raise LaunchError("production wall estimate is not finite")
    qualification_record = _file_record(qualification_path)
    return {
        "schema": "token-reconstruction.trr-p09-stage1-production-extrapolation.v1",
        "qualification_receipt": qualification_record,
        "qualification_receipt_sha256": qualification_record["sha256"],
        "qualification_elapsed_seconds": elapsed,
        "qualification_batch_count": len(batches),
        "qualification_forward_calls": qualification_forward_calls,
        "production_forward_calls": int(production_forward_calls),
        "forward_scaled_seconds": scaled_forward_seconds,
        "fixed_overhead_seconds": overhead,
        "fixed_overhead_scope": "model_load plus input_validation plus streamed_write/hash/synchronization overhead supplied by root",
        "estimated_wall_seconds": estimated,
        "production_wall_seconds_cap": PRODUCTION_WALL_SECONDS,
        "within_production_wall_cap": estimated <= PRODUCTION_WALL_SECONDS,
        "truth_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-qualification", type=Path)
    parser.add_argument("--fixed-overhead-seconds", type=float)
    parser.add_argument("--mode", choices=("qualify", "capture"))
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--stage1-plan", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--watchdog-output-root", type=Path)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--watchdog-receipt", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--wrapper-path", type=Path, default=WRAPPER_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--plan-sha256", default=PLAN_SHA256)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--kill-grace-seconds", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true", help="dispatch the external watchdog; default only prints the command")
    return parser


def _require_launch_args(args: argparse.Namespace) -> None:
    required = (
        "mode",
        "input_manifest",
        "stage1_plan",
        "output_root",
        "watchdog_output_root",
        "model_snapshot",
        "authorization_receipt",
        "watchdog_receipt",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise LaunchError("launch mode lacks required arguments: " + ", ".join(missing))
    if args.plan_sha256 != PLAN_SHA256:
        raise LaunchError("--plan-sha256 must equal the signed Stage1 plan SHA")
    repo = args.repository_root.expanduser().resolve()
    if not repo.is_dir():
        raise LaunchError(f"repository root is unavailable: {repo}")
    for name in ("input_manifest", "stage1_plan", "model_snapshot"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file() and name != "model_snapshot":
            raise LaunchError(f"launch input is unavailable: {path}")
    if not args.model_snapshot.expanduser().resolve().is_dir():
        raise LaunchError(f"model snapshot directory is unavailable: {args.model_snapshot}")
    if args.output_root.expanduser().resolve() == args.watchdog_output_root.expanduser().resolve():
        raise LaunchError("capture and watchdog output roots must be distinct")
    if args.poll_seconds <= 0 or args.kill_grace_seconds < 0:
        raise LaunchError("watchdog poll/grace values are invalid")


def _child_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/trr_p09/stage1_public_capture.py",
        "--mode", str(args.mode),
        "--input-manifest", str(args.input_manifest.expanduser().resolve()),
        "--stage1-plan", str(args.stage1_plan.expanduser().resolve()),
        "--output-root", str(args.output_root.expanduser().resolve()),
        "--model-snapshot", str(args.model_snapshot.expanduser().resolve()),
        "--device", str(args.device),
        "--plan-sha256", str(args.plan_sha256),
        "--authorization-receipt", str(args.authorization_receipt.expanduser().resolve()),
        "--watchdog-receipt", str(args.watchdog_receipt.expanduser().resolve()),
    ]


def build_launch(args: argparse.Namespace) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    _require_launch_args(args)
    wrapper = args.wrapper_path.expanduser().resolve()
    wrapper_record = _file_record(wrapper)
    if wrapper_record["bytes"] != WRAPPER_BYTES or wrapper_record["sha256"] != WRAPPER_SHA256:
        raise LaunchError("watchdog executable does not match the approved reusable P06 wrapper")
    timeout = QUALIFICATION_WALL_SECONDS if args.mode == "qualify" else PRODUCTION_WALL_SECONDS
    child = _child_command(args)
    watchdog = [
        sys.executable,
        str(wrapper),
        "--output-root", str(args.watchdog_output_root.expanduser().resolve()),
        "--timeout-seconds", str(timeout),
        "--poll-seconds", str(args.poll_seconds),
        "--max-rss-bytes", str(MAX_RSS_BYTES),
        "--min-available-bytes", str(MIN_HOST_AVAILABLE_RUNTIME_BYTES),
        "--kill-grace-seconds", str(args.kill_grace_seconds),
        "--cwd", str(args.repository_root.expanduser().resolve()),
        "--label", f"TRR-P09-stage1-{args.mode}",
        "--",
        *child,
    ]
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONPATH": f"{args.repository_root.expanduser().resolve() / 'src'}:{args.repository_root.expanduser().resolve()}",
    })
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    safe_env = {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PYTHONPATH", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE") if key in env}
    metadata = {
        "schema": "token-reconstruction.trr-p09-stage1-guarded-launch.v1",
        "task_id": TASK_ID,
        "mode": args.mode,
        "plan_sha256": args.plan_sha256,
        "child_command": child,
        "watchdog_command": watchdog,
        "wrapper": wrapper_record,
        "wrapper_enforcement": {
            "wall_seconds_cap": timeout,
            "host_rss_bytes_max": MAX_RSS_BYTES,
            "host_available_bytes_min": MIN_HOST_AVAILABLE_RUNTIME_BYTES,
            "process_group_fail_closed": True,
            "gpu_reserved_or_free": False,
            "retained_output_bytes": False,
        },
        "child_inner_enforcement": {
            "gpu_reserved_bytes_max": MAX_GPU_RESERVED_BYTES,
            "gpu_free_before_bytes_min": MIN_GPU_FREE_PRELAUNCH_BYTES,
            "gpu_free_during_bytes_min": MIN_GPU_FREE_RUNTIME_BYTES,
            "host_available_before_bytes_min": MIN_HOST_AVAILABLE_PRELAUNCH_BYTES,
            "retained_capture_bytes_max": MAX_RETAINED_BYTES,
            "filesystem_free_bytes_min": 20 * GIB,
        },
        "environment_allowlist": safe_env,
        "authorization_receipt": str(args.authorization_receipt.expanduser().resolve()),
        "watchdog_receipt": str(args.watchdog_receipt.expanduser().resolve()),
        "receipt_policy": "supplied root receipts are checked by the child; this helper creates none",
    }
    return watchdog, env, metadata


def _check_supplied_watchdog_receipt(path: Path, watchdog_command: Sequence[str], *, mode: str) -> None:
    value = _load_json(path, label="supplied watchdog receipt")
    command = value.get("command")
    expected = list(watchdog_command)
    if isinstance(command, list):
        actual = [str(item) for item in command]
    elif isinstance(command, str):
        actual = shlex.split(command)
    else:
        raise LaunchError("supplied watchdog receipt command is absent")
    if actual != expected:
        raise LaunchError("supplied watchdog receipt command differs from the guarded launch")
    if value.get("mode") != mode or value.get("condition") != "public_base":
        raise LaunchError("supplied watchdog receipt mode/condition differs")
    if value.get("schema") != "token-reconstruction.trr-p09-stage1-capture-watchdog.v1":
        raise LaunchError("supplied watchdog receipt schema differs")


def _run_estimate(args: argparse.Namespace) -> int:
    if args.fixed_overhead_seconds is None:
        raise LaunchError("--fixed-overhead-seconds is required with --estimate-qualification")
    estimate = estimate_production_wall(
        args.estimate_qualification.expanduser().resolve(),
        fixed_overhead_seconds=args.fixed_overhead_seconds,
    )
    print(json.dumps(estimate, indent=2, sort_keys=True, allow_nan=False))
    if not estimate["within_production_wall_cap"]:
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.estimate_qualification is not None:
            return _run_estimate(args)
        watchdog_command, env, metadata = build_launch(args)
        print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))
        if not args.execute:
            return 0
        for path in (args.authorization_receipt, args.watchdog_receipt):
            if not path.expanduser().resolve().is_file():
                raise LaunchError(f"--execute requires receipt file: {path}")
        if args.output_root.expanduser().resolve().exists() or args.watchdog_output_root.expanduser().resolve().exists():
            raise LaunchError("--execute requires create-only output roots")
        _check_supplied_watchdog_receipt(args.watchdog_receipt.expanduser().resolve(), watchdog_command, mode=args.mode)
        completed = subprocess.run(watchdog_command, cwd=str(args.repository_root.expanduser().resolve()), env=env, check=False)
        return int(completed.returncode)
    except LaunchError as exc:
        print(f"TRR-P09 guarded launch refused: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
