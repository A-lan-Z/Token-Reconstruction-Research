#!/usr/bin/env python3
"""Complete one missing registration byte count in a preserved run manifest.

The TRR-0008 runner wrote a registration binding containing ``path`` and
``sha256`` but omitted ``bytes``.  This helper preserves that original file and
writes a new metadata-completed manifest with only ``registration.bytes``
added.  It never loads predictions, observations, source rows, labels, or
truth.  A separate create-only receipt records the exact semantic diff and
why the public gate failed.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


TASK_ID = "TRR-0008"
RUN_SCHEMA = "token-reconstruction.trr0008-prediction-run.v1"
REPAIR_SCHEMA = "token-reconstruction.trr0008-run-manifest-metadata-completion.v1"
REPAIR_STATUS = "RUN_MANIFEST_METADATA_COMPLETION_COMPLETE_NO_TRUTH"
EXPECTED_GATE_REASON = "run registration binding changed: bytes"


class RepairError(RuntimeError):
    """Raised when metadata completion cannot be proven to be lossless."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RepairError(f"{description} is unavailable or a symlink: {path}")
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _resolve(value: Path, *, root: Path, description: str) -> Path:
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if path.is_symlink():
        raise RepairError(f"{description} is a symlink: {path}")
    return path


def _task_output(value: Path, *, root: Path, description: str) -> Path:
    path = _resolve(value, root=root, description=description)
    task_root = (root / "experiments" / TASK_ID).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise RepairError(f"{description} must be below {task_root}: {path}") from exc
    if path.exists() or path.is_symlink():
        raise RepairError(f"{description} is create-only and already exists: {path}")
    return path


def _load_json(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    record = _record(path, description=description)
    try:
        value = json.loads(Path(record["path"]).read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RepairError(f"{description} must be a JSON object")
    return value, Path(record["path"]).read_bytes()


def _write_create_only(path: Path, value: Mapping[str, Any], *, description: str) -> None:
    if path.exists() or path.is_symlink():
        raise RepairError(f"{description} is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:  # pragma: no cover - race-safe guard
        raise RepairError(f"{description} is create-only and already exists: {path}") from exc


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RepairError("cannot resolve helper execution commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RepairError("helper execution commit is not a full hash")
    return value


def _semantic_diff(before: Any, after: Any, *, path: str = "") -> list[dict[str, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else str(key)
            if key not in before:
                differences.append(
                    {"path": child, "operation": "add", "before": "<missing>", "after": after[key]}
                )
            elif key not in after:
                differences.append(
                    {"path": child, "operation": "remove", "before": before[key], "after": "<missing>"}
                )
            else:
                differences.extend(_semantic_diff(before[key], after[key], path=child))
        return differences
    if before != after:
        return [{"path": path, "operation": "replace", "before": before, "after": after}]
    return []


def complete_metadata(
    *,
    repository_root: Path,
    original_run_manifest: Path,
    registration: Path,
    output: Path,
    receipt: Path,
    gate_failure: str,
) -> dict[str, Any]:
    started_utc = _utc_now()
    root = Path(repository_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RepairError(f"repository root is unavailable: {root}")
    original_path = _resolve(original_run_manifest, root=root, description="original run manifest")
    registration_path = _resolve(registration, root=root, description="registration")
    output_path = _task_output(output, root=root, description="metadata-completed run manifest")
    receipt_path = _task_output(receipt, root=root, description="metadata-completion receipt")
    if output_path == receipt_path:
        raise RepairError("metadata-completed manifest and receipt paths must differ")
    if output_path == original_path or receipt_path == original_path:
        raise RepairError("original run manifest must remain untouched")
    if receipt_path == registration_path:
        raise RepairError("registration must remain untouched")
    if not gate_failure or EXPECTED_GATE_REASON not in gate_failure:
        raise RepairError(
            "gate_failure must preserve the exact public-gate reason: "
            f"{EXPECTED_GATE_REASON}"
        )

    original, original_bytes = _load_json(original_path, description="original run manifest")
    registration_record = _record(registration_path, description="actual registration")
    registration_doc, _registration_bytes = _load_json(
        registration_path, description="actual registration"
    )
    if registration_doc.get("task_id") not in (None, TASK_ID):
        raise RepairError("actual registration task identity changed")
    for key in ("truth_opened", "source_text_or_target_labels", "candidate_arrays_persisted"):
        if registration_doc.get(key) is True:
            raise RepairError(f"actual registration records forbidden state: {key}")
    original_registration = original.get("registration")
    if not isinstance(original_registration, Mapping):
        raise RepairError("original run manifest registration binding is absent")
    if set(original_registration) != {"path", "sha256"}:
        raise RepairError(
            "original registration binding must contain exactly path and sha256 before completion"
        )
    declared_registration_path = _resolve(
        Path(str(original_registration["path"])),
        root=root,
        description="original registration binding",
    )
    if declared_registration_path != registration_path:
        raise RepairError("original run manifest points to a different registration")
    if str(original_registration.get("sha256")) != registration_record["sha256"]:
        raise RepairError("original run manifest registration hash differs from actual registration")
    if original.get("schema") != RUN_SCHEMA or original.get("task_id") != TASK_ID:
        raise RepairError("original run manifest schema or task identity changed")
    for key in ("truth_opened", "candidate_arrays_persisted"):
        if original.get(key) is True:
            raise RepairError(f"original run manifest records forbidden state: {key}")

    completed = json.loads(original_bytes.decode("utf-8"))
    completed_registration = dict(completed["registration"])
    completed_registration["bytes"] = registration_record["bytes"]
    completed["registration"] = completed_registration
    differences = _semantic_diff(original, completed)
    expected_difference = [
        {
            "path": "registration.bytes",
            "operation": "add",
            "before": "<missing>",
            "after": registration_record["bytes"],
        }
    ]
    if differences != expected_difference:
        raise RepairError(f"metadata completion changed more than registration.bytes: {differences!r}")

    helper_record = _record(Path(__file__), description="metadata-completion helper")
    _write_create_only(
        output_path,
        completed,
        description="metadata-completed run manifest",
    )
    completed_record = _record(output_path, description="metadata-completed run manifest")
    receipt_value = {
        "schema": REPAIR_SCHEMA,
        "task_id": TASK_ID,
        "status": REPAIR_STATUS,
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "candidate_arrays_persisted": False,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "original_run_manifest": _record(original_path, description="original run manifest"),
        "metadata_completed_run_manifest": completed_record,
        "registration": registration_record,
        "semantic_diff": differences,
        "helper_code": helper_record,
        "gate_failure": {
            "action": "public_prediction_gate",
            "message": gate_failure,
            "reason": EXPECTED_GATE_REASON,
        },
        "execution": {
            "command": list(sys.argv),
            "code_commit": _git_head(root),
            "repository_root": str(root),
            "network_used": False,
            "truth_opened": False,
            "source_text_written": False,
            "token_ids_written": False,
        },
    }
    _write_create_only(
        receipt_path,
        receipt_value,
        description="metadata-completion receipt",
    )
    return receipt_value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--original-run-manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gate-failure", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = complete_metadata(
            repository_root=args.repository_root,
            original_run_manifest=args.original_run_manifest,
            registration=args.registration,
            output=args.output,
            receipt=args.receipt,
            gate_failure=args.gate_failure,
        )
    except (RepairError, OSError, ValueError, TypeError) as exc:
        print(f"TRR-0008 run-manifest metadata completion failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
