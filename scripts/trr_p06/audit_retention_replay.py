#!/usr/bin/env python3
"""Audit the logging-only P06 capacity replay against capacity-r1.

This reads only compact probe receipts, ledgers, schedules, learning curves,
and retained per-position prediction JSON.  It never opens observations,
source text, target truth outside the already recorded public-fit ledger, or
model/state tensors.  The output is create-only and records exact metrics and
container-versus-semantic schedule equivalence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

from safetensors import safe_open


SCHEMA = "token-reconstruction.trr-p06-capacity-replay-equivalence.v1"
METHODS = ("p06_positionwise_diagonal", "p06_past_only", "p06_full_record")


class AuditError(RuntimeError):
    """Raised when replay equivalence or artifact integrity fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON: {path}") from exc


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _tensor_digest(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _schedule(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise AuditError(f"safetensors file is truncated: {path}")
    header_length = struct.unpack("<Q", raw[:8])[0]
    header_end = 8 + int(header_length)
    if header_end > len(raw):
        raise AuditError(f"safetensors header exceeds file: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = dict(handle.metadata() or {})
        tensors = {key: _tensor_digest(handle.get_tensor(key)) for key in keys}
    return {
        "path": str(path),
        "bytes": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "header_length": int(header_length),
        "header_sha256": hashlib.sha256(raw[8:header_end]).hexdigest(),
        "data_sha256": hashlib.sha256(raw[header_end:]).hexdigest(),
        "keys": keys,
        "metadata": metadata,
        "tensor_digests": tensors,
    }


def _watchdog_cost(path: Path) -> dict[str, Any]:
    time_record = _json(path / "time.json")
    guard_record = _json(path / "resource_guard.json")
    finish = _json(path / "finish.json")
    return {
        "status": time_record.get("status"),
        "wrapper_exit_code": time_record.get("wrapper_exit_code"),
        "child_return_code": time_record.get("child_return_code"),
        "elapsed_seconds": time_record.get("elapsed_seconds"),
        "start_utc": time_record.get("start_utc"),
        "end_utc": time_record.get("end_utc"),
        "termination_reason": time_record.get("termination_reason"),
        "guard_status": guard_record.get("status"),
        "finish_sha256": _sha256(path / "finish.json"),
        "finish_record": finish,
    }


def _method_map(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    methods = receipt.get("methods")
    if not isinstance(methods, list):
        raise AuditError("probe receipt methods are not a list")
    result = {str(row["method_id"]): row for row in methods if isinstance(row, Mapping) and "method_id" in row}
    if tuple(result) != METHODS:
        raise AuditError(f"probe method order changed: {tuple(result)}")
    return result


def _check_prediction_rows(
    *,
    ledger: list[Mapping[str, Any]],
    prediction_path: Path,
    method_id: str,
    expected_initial: Mapping[str, Any],
    expected_final: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _json(prediction_path)
    if payload.get("schema") != "token-reconstruction.trr-p06-capacity-probe-predictions.v1":
        raise AuditError(f"prediction schema changed: {prediction_path}")
    if payload.get("method_id") != method_id or payload.get("row_count") != len(ledger):
        raise AuditError(f"prediction identity/count changed: {prediction_path}")
    initial = payload.get("initial_predictions")
    final = payload.get("final_predictions")
    if not isinstance(initial, list) or not isinstance(final, list) or len(initial) != len(ledger) or len(final) != len(ledger):
        raise AuditError(f"prediction rows are incomplete: {prediction_path}")
    ledger_keys = [(int(row["record_index"]), int(row["position"])) for row in ledger]
    initial_keys = [(int(row["record_index"]), int(row["position"])) for row in initial]
    final_keys = [(int(row["record_index"]), int(row["position"])) for row in final]
    if sorted(initial_keys) != sorted(ledger_keys) or sorted(final_keys) != sorted(ledger_keys):
        raise AuditError(f"prediction ledger positions changed: {prediction_path}")
    ledger_by_key = {(int(row["record_index"]), int(row["position"])): row for row in ledger}
    for retained, label in ((initial, "initial"), (final, "final")):
        for retained_row in retained:
            key = (int(retained_row["record_index"]), int(retained_row["position"]))
            source_row = ledger_by_key[key]
            if int(retained_row["target"]) != int(source_row["target"]):
                raise AuditError(f"{label} target differs from public-fit ledger: {prediction_path}")
            if label == "initial":
                if int(retained_row["prediction"]) != int(source_row["initial_prediction"]):
                    raise AuditError(f"initial prediction differs from ledger: {prediction_path}")
                if int(retained_row["tie_count"]) != int(source_row["initial_tie_count"]):
                    raise AuditError(f"initial tie count differs from ledger: {prediction_path}")
            if bool(retained_row["correct"]) != (int(retained_row["prediction"]) == int(retained_row["target"])):
                raise AuditError(f"{label} correctness flag is inconsistent: {prediction_path}")
    initial_correct = sum(bool(row["correct"]) for row in initial)
    final_correct = sum(bool(row["correct"]) for row in final)
    if initial_correct != int(expected_initial["correct"]) or final_correct != int(expected_final["correct"]):
        raise AuditError(f"prediction correctness counts differ from receipt: {prediction_path}")
    return {
        "path": str(prediction_path),
        "sha256": _sha256(prediction_path),
        "row_count": len(initial),
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "ledger_initial_ids_match": True,
        "prediction_row_order": "record-major evaluation order; key set matches public-fit ledger",
        "targets_match_public_fit_ledger": True,
    }


def audit(original_root: Path, replay_root: Path, output: Path) -> dict[str, Any]:
    original_root = original_root.expanduser().resolve()
    replay_root = replay_root.expanduser().resolve()
    output = output.expanduser().resolve()
    original_receipt_path = original_root / "capacity_probe_receipt.json"
    replay_receipt_path = replay_root / "capacity_probe_receipt.json"
    original_receipt = _json(original_receipt_path)
    replay_receipt = _json(replay_receipt_path)
    if original_receipt.get("status") != "PASS" or replay_receipt.get("status") != "PASS":
        raise AuditError("both capacity probe receipts must be PASS")
    if replay_receipt.get("probe_prediction_retention", {}).get("enabled") is not True:
        raise AuditError("replay receipt does not declare prediction retention")
    original_methods = _method_map(original_receipt)
    replay_methods = _method_map(replay_receipt)
    original_ledger = _json(original_root / "capacity_error_ledger.json")["rows"]
    replay_ledger = _json(replay_root / "capacity_error_ledger.json")["rows"]
    if original_ledger != replay_ledger:
        raise AuditError("capacity error ledgers differ")
    original_ledger_file_sha = _sha256(original_root / "capacity_error_ledger.json")
    replay_ledger_file_sha = _sha256(replay_root / "capacity_error_ledger.json")
    if original_ledger_file_sha != replay_ledger_file_sha:
        raise AuditError("capacity error ledger file hashes differ")

    original_schedule = _schedule(original_root / "capacity_schedule.safetensors")
    replay_schedule = _schedule(replay_root / "capacity_schedule.safetensors")
    schedule_semantic_equal = (
        original_schedule["keys"] == replay_schedule["keys"]
        and original_schedule["metadata"] == replay_schedule["metadata"]
        and original_schedule["tensor_digests"] == replay_schedule["tensor_digests"]
        and original_schedule["data_sha256"] == replay_schedule["data_sha256"]
    )
    if not schedule_semantic_equal:
        raise AuditError("capacity schedule tensor/header semantics differ")
    if original_schedule["file_sha256"] == replay_schedule["file_sha256"]:
        raise AuditError("expected replay schedule container hash difference was absent")

    method_results: list[dict[str, Any]] = []
    for method_id in METHODS:
        old = original_methods[method_id]
        new = replay_methods[method_id]
        curve_old = original_root / f"{method_id}.learning_curve.json"
        curve_new = replay_root / f"{method_id}.learning_curve.json"
        if _sha256(curve_old) != _sha256(curve_new):
            raise AuditError(f"learning curve differs: {method_id}")
        if old.get("initial_state_sha256") != new.get("initial_state_sha256") or old.get("final_state_sha256") != new.get("final_state_sha256"):
            raise AuditError(f"state hash differs: {method_id}")
        if old.get("initial_metrics") != new.get("initial_metrics") or old.get("final_metrics") != new.get("final_metrics"):
            raise AuditError(f"aggregate metrics differ: {method_id}")
        prediction = _check_prediction_rows(
            ledger=original_ledger,
            prediction_path=replay_root / f"{method_id}.probe_predictions.json",
            method_id=method_id,
            expected_initial=old["initial_metrics"],
            expected_final=old["final_metrics"],
        )
        method_results.append(
            {
                "method_id": method_id,
                "status_equal": old.get("status") == new.get("status") == "PASS",
                "initial_state_sha256": new.get("initial_state_sha256"),
                "final_state_sha256": new.get("final_state_sha256"),
                "initial_metrics_equal": True,
                "final_metrics_equal": True,
                "learning_curve_sha256": _sha256(curve_new),
                "prediction_file": prediction,
            }
        )

    original_watchdog = original_root.parent / "watchdog-capacity-r1"
    replay_watchdog = replay_root.parent / "watchdog-capacity-retention-replay-r1"
    original_arm_seconds = sum(float(row["arm_wall_seconds"]) for row in original_methods.values())
    replay_arm_seconds = sum(float(row["arm_wall_seconds"]) for row in replay_methods.values())
    result = {
        "schema": SCHEMA,
        "task_id": "TRR-P06",
        "status": "PASS",
        "created_utc": _utc_now(),
        "command": [
            "python3",
            "scripts/trr_p06/audit_retention_replay.py",
            "--original-root",
            str(original_root),
            "--replay-root",
            str(replay_root),
            "--output",
            str(output),
        ],
        "original": {
            "root": str(original_root),
            "receipt_sha256": _sha256(original_receipt_path),
            "source_commit": original_receipt.get("source_commit"),
            "watchdog": _watchdog_cost(original_watchdog),
            "arm_wall_seconds_sum": original_arm_seconds,
        },
        "replay": {
            "root": str(replay_root),
            "receipt_sha256": _sha256(replay_receipt_path),
            "source_commit": replay_receipt.get("source_commit"),
            "watchdog": _watchdog_cost(replay_watchdog),
            "arm_wall_seconds_sum": replay_arm_seconds,
        },
        "ledger": {
            "row_count": len(original_ledger),
            "file_sha256_equal": True,
            "sha256": original_ledger_file_sha,
        },
        "schedule": {
            "semantic_schedule_sha256": original_receipt["schedule"]["schedule_sha256"],
            "semantic_schedule_sha256_equal": original_receipt["schedule"]["schedule_sha256"] == replay_receipt["schedule"]["schedule_sha256"],
            "tensor_values_equal": True,
            "metadata_equal": True,
            "data_region_sha256_equal": True,
            "container_file_sha256_equal": False,
            "original_file_sha256": original_schedule["file_sha256"],
            "replay_file_sha256": replay_schedule["file_sha256"],
            "original_header_sha256": original_schedule["header_sha256"],
            "replay_header_sha256": replay_schedule["header_sha256"],
            "difference_scope": "safetensors header JSON metadata ordering; tensor data region is byte-identical",
        },
        "methods": method_results,
        "replay_cost": {
            "retention_files": 3,
            "retained_rows_per_method": len(original_ledger),
            "retained_rows_total": len(original_ledger) * len(METHODS) * 2,
            "additional_optimizer_updates": 0,
            "additional_fit_choices": 0,
            "fresh_truth_or_source_access": False,
        },
    }
    if output.exists() or output.is_symlink():
        raise AuditError(f"output is create-only: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit(args.original_root, args.replay_root, args.output)
    except (AuditError, OSError, KeyError, ValueError) as exc:
        print(f"TRR-P06 retention audit failed: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
