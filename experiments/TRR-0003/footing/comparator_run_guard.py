#!/usr/bin/env python3
"""Fail-closed resource and source guard for the TRR-0003 comparator.

The comparator runs one largest-cell qualification before the complete matrix.
The matrix guard accepts a qualification receipt only when its measured peak is
inside the declared envelope and leaves the declared free-memory margin.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_PATH = ROOT / "experiments/TRR-0003/footing/comparator_preflight.json"
DEFAULT_REFERENCE = ROOT / (
    "outputs/TRR-0002/configuration-search/fresh-blind-code/reference/strict_bos/"
    "round001_teacher.py"
)
DEFAULT_LENS = ROOT / "outputs/TRR-0002/blind/reconstructor_input/public_a1_lens.pt"
DEFAULT_INVERSE = ROOT / "outputs/TRR-0001/reconstructor_public/inverses/cut4.safetensors"
FROZEN = (
    ROOT / "scripts/trr0003_footing_compare.py",
    ROOT / "src/token_reconstruction/a1a2_configuration_search.py",
    ROOT / "src/token_reconstruction/component_crossover.py",
    ROOT / "src/token_reconstruction/experiment_runtime.py",
    ROOT / "src/token_reconstruction/inverse.py",
    ROOT / "src/token_reconstruction/footing.py",
    ROOT / "experiments/TRR-0003/footing/plan.json",
    ROOT / "experiments/TRR-0003/footing/panel.json",
    ROOT / "experiments/TRR-0003/footing/comparator_preflight.json",
    ROOT / "experiments/TRR-0003/footing/comparator_run_guard.py",
    DEFAULT_REFERENCE,
    DEFAULT_LENS,
    DEFAULT_INVERSE,
)


class GuardError(RuntimeError):
    """Raised when a comparator guard contract cannot be verified."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise GuardError(f"frozen path unavailable or symlinked: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()
    if len(value) != 40:
        raise GuardError("executable commit is not a full hash")
    return value


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
    raise GuardError("host MemAvailable is unavailable")


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
        raise GuardError(f"expected one exclusive GPU, observed {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6:
        raise GuardError(f"unexpected nvidia-smi geometry: {rows[0]!r}")
    try:
        total_mib, free_mib, used_mib, temperature_c, utilization_pct = (
            int(float(fields[index])) for index in range(1, 6)
        )
    except ValueError as exc:
        raise GuardError(f"nvidia-smi numeric fields are invalid: {rows[0]!r}") from exc
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


def _preflight_config() -> dict[str, Any]:
    try:
        value = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError(f"comparator preflight is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise GuardError("comparator preflight root is not an object")
    return value


def resource_preflight() -> dict[str, Any]:
    config = _preflight_config()
    try:
        envelope = config["resource_envelope"]
        gpu_envelope = int(envelope["gpu_envelope_bytes"])
        host_envelope = int(envelope["host_envelope_bytes"])
        margin = float(envelope["minimum_margin_fraction"])
        minimum_free = int(envelope["minimum_free_before_load_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GuardError(f"resource envelope is malformed: {exc}") from exc
    if gpu_envelope <= 0 or host_envelope <= 0 or not 0.0 <= margin < 1.0:
        raise GuardError("resource envelope has invalid values")
    live_gpu = _live_gpu()
    live_host = _available_host_bytes()
    gpu_required = max(minimum_free, math.ceil(gpu_envelope / (1.0 - margin)))
    host_required = math.ceil(host_envelope / (1.0 - margin))
    checks = {
        "gpu_margin_pass": live_gpu["free_bytes"] >= gpu_required,
        "host_margin_pass": live_host >= host_required,
        "thermal_pass": live_gpu["temperature_c"] < 85,
        "exclusive_gpu_pass": not live_gpu["compute_processes"],
    }
    if not all(checks.values()):
        raise GuardError(
            "live comparator resource margin failed: "
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
        "minimum_free_before_load_bytes": minimum_free,
        "predicted_gpu_envelope_bytes": gpu_envelope,
        "predicted_host_envelope_bytes": host_envelope,
        "required_gpu_free_bytes": gpu_required,
        "required_host_available_bytes": host_required,
        "live_gpu": live_gpu,
        "live_host_available_bytes": live_host,
        "checks": checks,
    }


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GuardError(f"evidence path unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError(f"evidence JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise GuardError(f"evidence JSON root is not an object: {path}")
    return value


def _peak_from_value(value: Any, peaks: list[tuple[int, int]]) -> None:
    if isinstance(value, Mapping):
        allocated = value.get("cuda_peak_allocated_bytes")
        reserved = value.get("cuda_peak_reserved_bytes")
        if isinstance(allocated, (int, float)) and isinstance(reserved, (int, float)):
            peaks.append((int(allocated), int(reserved)))
        for nested in value.values():
            _peak_from_value(nested, peaks)
    elif isinstance(value, list):
        for nested in value:
            _peak_from_value(nested, peaks)


def measured_peak(path: Path) -> dict[str, Any]:
    evidence = _json(path)
    peaks: list[tuple[int, int]] = []
    _peak_from_value(evidence, peaks)
    if not peaks:
        raise GuardError(f"measured evidence has no CUDA peak memory: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "cuda_peak_allocated_bytes": max(item[0] for item in peaks),
        "cuda_peak_reserved_bytes": max(item[1] for item in peaks),
        "observations": len(peaks),
    }


def _qualification_check(path: Path, current: dict[str, Any]) -> dict[str, Any]:
    receipt = _json(path)
    if not receipt.get("guard_passed"):
        raise GuardError("qualification guard did not pass")
    if receipt.get("git_commit_unchanged") is not True or receipt.get("source_hashes_unchanged") is not True:
        raise GuardError("qualification guard has an integrity failure")
    end = receipt.get("end")
    if not isinstance(end, Mapping) or end.get("git_commit") != current["git_commit"]:
        raise GuardError("qualification and main matrix commits differ")
    if dict(end.get("source_hashes", {})) != current["source_hashes"]:
        raise GuardError("qualification and main matrix source hashes differ")
    peak = receipt.get("measured_peak")
    if not isinstance(peak, Mapping):
        raise GuardError("qualification guard has no measured peak")
    config = _preflight_config()
    rule = config["qualification_rule"]
    qualification = receipt.get("qualification")
    if not isinstance(qualification, Mapping):
        raise GuardError("qualification guard has no geometry declaration")
    if qualification.get("cell_id") != rule["cell_id"]:
        raise GuardError("qualification cell differs from preflight")
    if qualification.get("record_batch_size") != rule["record_batch_size"]:
        raise GuardError("qualification record batch differs from preflight")
    if qualification.get("candidate_budget") != rule["candidate_budget"]:
        raise GuardError("qualification candidate budget differs from preflight")
    if qualification.get("reference_record_batch_size") != rule["batch_equivalence_reference_size"]:
        raise GuardError("qualification reference record batch differs from preflight")
    measured_path = ROOT / str(peak.get("path", ""))
    measured_evidence = _json(measured_path)
    if measured_evidence.get("status") != "QUALIFICATION_ONLY_PREDICTIONS_COMPLETE":
        raise GuardError("qualification output status is not complete")
    if measured_evidence.get("selected_cell_id") != rule["cell_id"]:
        raise GuardError("qualification output cell differs from preflight")
    timing = measured_evidence.get("timing")
    a2_rows = timing.get("frozen_a1_a2_k256") if isinstance(timing, Mapping) else None
    if not isinstance(a2_rows, list) or len(a2_rows) != 1:
        raise GuardError("qualification output lacks one A1+A2 timing row")
    equivalence = a2_rows[0].get("batch_equivalence")
    if not isinstance(equivalence, Mapping) or equivalence.get("verified") is not True:
        raise GuardError("qualification record-batch equivalence was not verified")
    if int(equivalence.get("reference_record_batch_size", -1)) != rule["batch_equivalence_reference_size"]:
        raise GuardError("qualification equivalence reference batch differs from preflight")
    if int(peak["cuda_peak_reserved_bytes"]) > int(config["resource_envelope"]["gpu_envelope_bytes"]):
        raise GuardError("qualification reserved peak exceeds declared envelope")
    start_free = int(receipt["resource_preflight"]["live_gpu"]["free_bytes"])
    remaining_fraction = (start_free - int(peak["cuda_peak_reserved_bytes"])) / start_free
    required_fraction = float(rule["require_remaining_free_fraction"])
    if remaining_fraction < required_fraction:
        raise GuardError(
            f"qualification leaves insufficient measured GPU margin: {remaining_fraction:.6f} < {required_fraction:.6f}"
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "git_commit": current["git_commit"],
        "measured_peak": dict(peak),
        "qualification_start_free_bytes": start_free,
        "remaining_free_fraction": remaining_fraction,
        "required_remaining_free_fraction": required_fraction,
    }


def _label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise GuardError(f"guard evidence must be create-only: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _parse_argv(argv: list[str]) -> tuple[Path, str, Path | None, Path | None, list[str]]:
    if len(argv) < 3:
        raise GuardError(
            "usage: comparator_run_guard.py <evidence.json> --mode qualification|main "
            "[--measured-evidence path] [--require-qualification path] -- command ..."
        )
    evidence = Path(argv[0]).resolve()
    mode = "main"
    measured: Path | None = None
    qualification: Path | None = None
    index = 1
    while index < len(argv) and argv[index] != "--":
        option = argv[index]
        if option == "--mode" and index + 1 < len(argv):
            mode = argv[index + 1]
            index += 2
        elif option == "--measured-evidence" and index + 1 < len(argv):
            measured = Path(argv[index + 1]).resolve()
            index += 2
        elif option == "--require-qualification" and index + 1 < len(argv):
            qualification = Path(argv[index + 1]).resolve()
            index += 2
        else:
            raise GuardError(f"unknown or incomplete guard option: {option}")
    if index >= len(argv) - 1:
        raise GuardError("guard command is missing after --")
    command = argv[index + 1 :]
    if mode not in {"qualification", "main"}:
        raise GuardError(f"unknown guard mode: {mode}")
    if mode == "qualification" and measured is None:
        raise GuardError("qualification mode requires --measured-evidence")
    if mode == "main" and qualification is None:
        raise GuardError("main mode requires --require-qualification")
    return evidence, mode, measured, qualification, command


def main(argv: list[str] | None = None) -> int:
    try:
        evidence_path, mode, measured_path, qualification_path, command = _parse_argv(
            list(sys.argv[1:] if argv is None else argv)
        )
        stdout_path = evidence_path.with_name(evidence_path.stem + ".stdout.log")
        stderr_path = evidence_path.with_name(evidence_path.stem + ".stderr.log")
        if stdout_path.exists() or stdout_path.is_symlink() or stderr_path.exists() or stderr_path.is_symlink():
            raise GuardError("guard logs must be create-only")
        start = snapshot()
        started = time.perf_counter()
        resource: dict[str, Any]
        resource_error: str | None = None
        try:
            resource = resource_preflight()
        except (GuardError, OSError, ValueError, subprocess.CalledProcessError) as exc:
            resource = {"status": "BLOCKED", "error": str(exc)}
            resource_error = str(exc)
        qualification: dict[str, Any] | None = None
        if resource_error is None and mode == "main":
            try:
                qualification = _qualification_check(qualification_path, start)
            except (GuardError, OSError, ValueError, json.JSONDecodeError) as exc:
                resource_error = str(exc)
        command_executed = resource_error is None
        command_returncode: int | None = None
        if command_executed:
            with stdout_path.open("x", encoding="utf-8", newline="") as stdout, stderr_path.open(
                "x", encoding="utf-8", newline="\n"
            ) as stderr:
                result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
                command_returncode = result.returncode
        else:
            stdout_path.write_text("", encoding="utf-8", newline="\n")
            stderr_path.write_text(
                f"comparator guard blocked before command: {resource_error}\n",
                encoding="utf-8",
                newline="\n",
            )
        measured: dict[str, Any] | None = None
        measured_error: str | None = None
        if measured_path is not None and command_executed and command_returncode == 0:
            try:
                measured = measured_peak(measured_path)
                measured_output = _json(measured_path)
                if mode == "qualification":
                    declaration = measured_output.get("qualification")
                    config = _preflight_config()
                    rule = config["qualification_rule"]
                    if measured_output.get("status") != "QUALIFICATION_ONLY_PREDICTIONS_COMPLETE":
                        raise GuardError("qualification output status is not complete")
                    if not isinstance(declaration, Mapping):
                        raise GuardError("qualification output has no qualification declaration")
                    qualification = {
                        "cell_id": declaration.get("cell_id"),
                        "candidate_budget": declaration.get("candidate_budget"),
                        "record_batch_size": declaration.get("record_batch_size"),
                        "reference_record_batch_size": declaration.get("reference_record_batch_size"),
                    }
                    if qualification["cell_id"] != rule["cell_id"]:
                        raise GuardError("qualification output cell differs from preflight")
                    if qualification["candidate_budget"] != rule["candidate_budget"]:
                        raise GuardError("qualification output candidate budget differs from preflight")
                    if qualification["record_batch_size"] != rule["record_batch_size"]:
                        raise GuardError("qualification output record batch differs from preflight")
                    if qualification["reference_record_batch_size"] != rule["batch_equivalence_reference_size"]:
                        raise GuardError("qualification output reference batch differs from preflight")
                if int(measured["cuda_peak_reserved_bytes"]) > int(_preflight_config()["resource_envelope"]["gpu_envelope_bytes"]):
                    measured_error = "measured reserved peak exceeds declared GPU envelope"
            except (GuardError, OSError, ValueError, json.JSONDecodeError) as exc:
                measured_error = str(exc)
        end = snapshot()
        source_hashes_unchanged = start["source_hashes"] == end["source_hashes"]
        git_commit_unchanged = start["git_commit"] == end["git_commit"]
        integrity_pass = source_hashes_unchanged and git_commit_unchanged
        guard_passed = (
            command_executed
            and command_returncode == 0
            and integrity_pass
            and resource["status"] == "PASS"
            and measured_error is None
            and (mode != "main" or qualification is not None)
        )
        if guard_passed:
            returncode = 0
        elif not integrity_pass:
            returncode = 3
        elif command_returncode not in (None, 0):
            returncode = int(command_returncode)
        else:
            returncode = 3
        payload: dict[str, Any] = {
            "schema": "token-reconstruction.trr0003-footing-comparator-run-guard.v1",
            "task_id": "TRR-0003",
            "track": "footing_comparator",
            "mode": mode,
            "command": {"argv": command, "cwd": str(ROOT)},
            "start": start,
            "end": end,
            "elapsed_seconds": time.perf_counter() - started,
            "resource_preflight": resource,
            "qualification": qualification,
            "qualification_error": resource_error if mode == "main" else None,
            "command_executed": command_executed,
            "command_returncode": command_returncode,
            "measured_peak": measured,
            "measured_error": measured_error,
            "returncode": returncode,
            "source_hashes_unchanged": source_hashes_unchanged,
            "git_commit_unchanged": git_commit_unchanged,
            "frozen_code_edit_during_run": not integrity_pass,
            "guard_passed": guard_passed,
            "stdout_log": {"path": _label(stdout_path), "sha256": sha(stdout_path), "bytes": stdout_path.stat().st_size},
            "stderr_log": {"path": _label(stderr_path), "sha256": sha(stderr_path), "bytes": stderr_path.stat().st_size},
        }
        _write_json(evidence_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return returncode
    except (GuardError, OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SystemExit(f"TRR-0003 comparator guard error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
