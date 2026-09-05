#!/usr/bin/env python3
"""Freeze the TRR-0005 public matrix before opening evaluator truth.

This task-local adapter carries the strong TRR-0004 gate forward while using
the prospective TRR-0005 4-cell × 8-method contract.  It validates every
prediction and timing receipt, then delegates byte hashing and create-only
receipt writing to :mod:`token_reconstruction.freeze`.  The truth sidecar is
only represented by its path at freeze time and is never loaded here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from token_reconstruction.freeze import (
    FreezeError,
    create_freeze_receipt,
    verify_freeze_receipt,
)
from token_reconstruction.footing import FootingError, file_record, sha256_file
from token_reconstruction.trr0005_contract import (
    ContractError,
    EXPECTED_CELL_IDS,
    METHOD_IDS,
    TASK_ID,
    validate_complete_public_matrix,
)


SCHEMA = "token-reconstruction.trr0005-confirmation-freeze.v1"


class ConfirmationFreezeError(ContractError):
    """Raised when the public TRR-0005 matrix cannot be frozen."""


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConfirmationFreezeError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmationFreezeError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ConfirmationFreezeError(f"{description} must be a JSON object")
    return value


def discover_prediction_receipts(output_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Discover task-local ``*.run.json`` receipts without opening truth."""

    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    timings: dict[tuple[str, str], dict[str, Any]] = {}
    if output_root.is_symlink() or not output_root.is_dir():
        raise ConfirmationFreezeError(f"prediction output root is unavailable: {output_root}")
    for path in sorted(output_root.rglob("*.run.json")):
        if path.is_symlink() or not path.is_file():
            continue
        value = _load_json(path, description="prediction receipt")
        cell_id = value.get("cell_id")
        method_id = value.get("method_id")
        if not isinstance(cell_id, str) or not isinstance(method_id, str):
            raise ConfirmationFreezeError(f"prediction receipt lacks cell/method binding: {path}")
        key = (cell_id, method_id)
        if key in predictions:
            raise ConfirmationFreezeError(f"duplicate prediction receipt: {cell_id}/{method_id}")
        value["receipt_path"] = str(path)
        predictions[key] = value
        timings[key] = value
    return predictions, timings


def freeze_public_matrix(
    *,
    root: Path,
    panel: Mapping[str, Any],
    registration: Mapping[str, Any],
    prediction_descriptors: Mapping[tuple[str, str], Mapping[str, Any]],
    timing_descriptors: Mapping[tuple[str, str], Mapping[str, Any]],
    frozen_root: Path | None = None,
    plan_path: Path | None = None,
    receipt_path: Path | None = None,
    panel_path: Path | None = None,
    registration_path: Path | None = None,
    public_validation_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all public receipts and optionally write a byte-bound receipt."""

    gate = validate_complete_public_matrix(
        panel,
        registration,
        prediction_descriptors,
        timing_descriptors=timing_descriptors,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_MATRIX_VALIDATED_NO_TRUTH_OPENED",
        "public_gate": gate,
        "truth_opened": False,
        "method_ids": list(METHOD_IDS),
        "cells": list(EXPECTED_CELL_IDS),
    }
    if frozen_root is None or plan_path is None or receipt_path is None:
        return result
    root = root.expanduser().resolve()
    frozen_root = frozen_root.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    if panel_path is None or registration_path is None:
        raise ConfirmationFreezeError(
            "panel_path and registration_path are required to bind the frozen receipt"
        )
    panel_path = panel_path.expanduser().resolve()
    registration_path = registration_path.expanduser().resolve()
    try:
        panel_record = file_record(panel_path, repository_root=root)
        registration_record = file_record(registration_path, repository_root=root)
        plan_record = file_record(plan_path, repository_root=root)
    except (FootingError, OSError, ValueError) as exc:
        raise ConfirmationFreezeError("freeze input file binding is unavailable") from exc
    try:
        panel_file = _load_json(panel_path, description="panel")
        registration_file = _load_json(registration_path, description="registration")
    except ConfirmationFreezeError:
        raise
    if panel_file != dict(panel) or registration_file != dict(registration):
        raise ConfirmationFreezeError("freeze input file content differs from supplied mappings")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ConfirmationFreezeError(f"freeze receipt is create-only: {receipt_path}")
    code_commit = registration.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise ConfirmationFreezeError("frozen registration has no full code commit")
    metadata = {
        "task_id": TASK_ID,
        "contract": "token-reconstruction.trr0005-contract.v1",
        "method_ids": list(METHOD_IDS),
        "panel_sha256": panel_record["sha256"],
        "selection_plan_sha256": plan_record["sha256"],
        "registration_sha256": registration_record["sha256"],
        "code_commit": code_commit,
        "public_gate": gate,
        "truth_opened": False,
        "candidate_output": {
            "frozen_a1_a2_k256": "omitted_after_decision",
        },
        "timing_contract": {
            "warmup_runs_per_record": 1,
            "measured_runs_per_record": 1,
            "warmup_output_exact_match_measured": True,
        },
    }
    if public_validation_selection is not None:
        metadata["public_validation_selection"] = dict(public_validation_selection)
    try:
        payload = create_freeze_receipt(
            repository_root=root,
            frozen_root=frozen_root,
            plan_path=plan_path,
            receipt_path=receipt_path,
            preregistration_commit=code_commit,
            created_utc=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )
        verify_freeze_receipt(receipt_path, repository_root=root)
    except FreezeError as exc:
        raise ConfirmationFreezeError(str(exc)) from exc
    result.update(
        {
            "status": "PUBLIC_MATRIX_FROZEN_NO_TRUTH_OPENED",
            "receipt": file_record(receipt_path, repository_root=root),
            "frozen_root": str(frozen_root.relative_to(root).as_posix()),
            "frozen_entries": len(payload["entries"]),
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.expanduser().resolve()
    panel = _load_json(args.panel.expanduser().resolve(), description="panel")
    registration = _load_json(args.registration.expanduser().resolve(), description="registration")
    predictions, timings = discover_prediction_receipts(args.output_root.expanduser().resolve())
    plan = _load_json(args.plan.expanduser().resolve(), description="selection plan")
    selection = plan.get("public_validation_selection")
    result = freeze_public_matrix(
        root=root,
        panel=panel,
        registration=registration,
        prediction_descriptors=predictions,
        timing_descriptors=timings,
        frozen_root=args.output_root,
        plan_path=args.plan,
        receipt_path=args.receipt,
        panel_path=args.panel,
        registration_path=args.registration,
        public_validation_selection=selection if isinstance(selection, Mapping) else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfirmationFreezeError, FreezeError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0005 freeze error: {exc}") from exc

