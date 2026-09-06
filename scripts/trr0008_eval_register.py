"""Bind the frozen TRR-0008 methods to public observations and runtime E.

This binder reads only metadata and state files.  It does not inspect source
text, token labels, or a truth sidecar.  The parent TRR-0007 method-freeze
ledger remains the authority for the selected state hashes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from scripts import trr0008_eval_contract as contract


class RegisterError(contract.ContractError):
    pass


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RegisterError(f"repository root is unavailable: {root}")
    return root


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RegisterError("cannot resolve registration commit") from exc
    if contract._COMMIT.fullmatch(value) is None:
        raise RegisterError("registration commit is not a full hash")
    return value


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        return contract.validate_file_record(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": contract.sha256_file(path),
            },
            repository_root=root,
            description=description,
            verify=True,
        )
    except (OSError, contract.ContractError) as exc:
        raise RegisterError(f"cannot bind {description}: {path}") from exc


def _load_method_freeze(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(path, root=root, description="TRR-0007 method freeze")
    freeze = contract.load_json(path, description="TRR-0007 method freeze")
    if freeze.get("task_id") != "TRR-0007" or freeze.get("status") != "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION":
        raise RegisterError("parent method freeze is not the reviewed pre-truth freeze")
    if freeze.get("truth_opened") is True or freeze.get("target_loaded") is True:
        raise RegisterError("parent method freeze records forbidden truth access")
    if not isinstance(freeze.get("state_bindings"), Mapping):
        raise RegisterError("parent method freeze lacks state bindings")
    return record, freeze


def _state_binding(freeze: Mapping[str, Any], method_id: str) -> Mapping[str, Any]:
    if method_id == contract.REFERENCE_METHOD_ID:
        state = freeze.get("retained_reference")
    else:
        row = freeze.get("state_bindings", {}).get(method_id)
        state = row.get("state") if isinstance(row, Mapping) else None
    if not isinstance(state, Mapping):
        raise RegisterError(f"state binding missing for {method_id}")
    return state


def _method_row(
    method_id: str,
    *,
    state: Mapping[str, Any],
    root: Path,
    freeze_sha256: str,
    records_per_cell: Mapping[str, int],
) -> dict[str, Any]:
    if method_id not in (*contract.METHOD_ORDER, contract.TIMING_CONTROL_METHOD_ID):
        raise RegisterError(f"unknown TRR-0008 method: {method_id}")
    checked_state = contract.validate_file_record(
        state,
        repository_root=root,
        description=f"{method_id} state",
        verify=True,
    )
    if method_id == contract.REFERENCE_METHOD_ID:
        loader = contract.REFERENCE_LOADER
        role = "reference"
    else:
        loader = dict(contract.POSITIONWISE_LOADER) | {
            "kwargs": dict(contract.POSITIONWISE_LOADER["kwargs"])
            | {"method_id": contract.METHOD_MODEL_IDS[method_id]},
        }
        role = "candidate" if method_id == contract.PRIMARY_METHOD_ID else "control"
    return {
        "id": method_id,
        "role": role,
        "kind": "decoder",
        "support": "current_enriched" if method_id == contract.REFERENCE_METHOD_ID else contract.METHOD_SUPPORT[method_id],
        "capacity": "trained_diagonal" if method_id == contract.REFERENCE_METHOD_ID else contract.METHOD_CAPACITY[method_id],
        "cells": list(contract.CELL_ORDER),
        "records_per_cell": dict(records_per_cell),
        "candidate_policy": "forbidden",
        "state": checked_state,
        "loader": loader,
        "method_freeze_sha256": freeze_sha256,
    }


def build_registration(
    *,
    repository_root: Path,
    method_freeze_path: Path,
    observation_manifest_path: Path,
    output_root: str,
    plan_path: Path | None = None,
    timing_plan_path: Path | None = None,
    timing_receipt_path: Path | None = None,
    include_code_bindings: bool = True,
) -> dict[str, Any]:
    root = _root(repository_root)
    if (timing_plan_path is None) != (timing_receipt_path is None):
        raise RegisterError("timing plan and final timing receipt must be supplied together")
    freeze_record, freeze = _load_method_freeze(method_freeze_path, root=root)
    observations_record = _record(
        observation_manifest_path,
        root=root,
        description="TRR-0008 public observation manifest",
    )
    observations = contract.load_json(observation_manifest_path, description="TRR-0008 public observations")
    checked_observations = contract.validate_observation_manifest(
        observations,
        repository_root=root,
        verify_assets=True,
    )
    records_by_domain = {
        str(domain): int(count)
        for domain, count in checked_observations["records_by_domain"].items()
    }
    records_per_cell = {
        cell_id: records_by_domain[cell_id.split("__", 1)[0]]
        for cell_id in contract.CELL_ORDER
    }
    runtime_e = freeze.get("runtime_embedding")
    if not isinstance(runtime_e, Mapping):
        raise RegisterError("parent method freeze lacks normalized public E")
    e_record = contract.validate_file_record(
        runtime_e,
        repository_root=root,
        description="normalized public E",
        verify=True,
    )
    if e_record["sha256"] != contract.PUBLIC_E_SHA256:
        raise RegisterError("normalized public E differs from the reviewed public table")
    methods = [
        _method_row(
            method_id,
            state=_state_binding(freeze, method_id),
            root=root,
            freeze_sha256=freeze_record["sha256"],
            records_per_cell=records_per_cell,
        )
        for method_id in contract.METHOD_ORDER
    ]
    # The alias is recorded only to make timing validation reproducible.  It is
    # deliberately absent from ``method_ids`` and from the scoring matrix.
    alias = _method_row(
        contract.TIMING_CONTROL_METHOD_ID,
        state=_state_binding(freeze, contract.TIMING_CONTROL_METHOD_ID),
        root=root,
        freeze_sha256=freeze_record["sha256"],
        records_per_cell=records_per_cell,
    )
    output = Path(output_root).expanduser()
    if output.is_absolute():
        resolved_output = output.resolve()
    else:
        resolved_output = (root / output).resolve()
    task_root = (root / "experiments" / "TRR-0008").resolve()
    try:
        resolved_output.relative_to(task_root)
    except ValueError as exc:
        raise RegisterError(f"output root must be below {task_root}") from exc
    if resolved_output.is_symlink():
        raise RegisterError("output root is a symlink")

    code_bindings: list[dict[str, Any]] = []
    if include_code_bindings:
        for relative in (
            "scripts/trr0008_eval_contract.py",
            "scripts/trr0008_eval_register.py",
            "scripts/trr0008_eval_runner.py",
            "scripts/trr0008_eval_timing.py",
            "scripts/trr0008_eval_gate.py",
            "scripts/trr0008_eval_capture.py",
            "scripts/trr0008_eval_truth.py",
            "scripts/trr0008_score.py",
            "src/token_reconstruction/trr0005_joint_decoder.py",
            "src/token_reconstruction/trr0007_positionwise.py",
        ):
            path = root / relative
            if not path.is_file():
                raise RegisterError(f"required code binding is unavailable: {relative}")
            code_bindings.append(
                _record(path, root=root, description=f"code binding {relative}")
                | {"role": relative, "path": relative}
            )
    plan_record = None
    if plan_path is not None:
        plan_record = _record(plan_path, root=root, description="TRR-0008 prospective plan")
        plan = contract.load_json(plan_path, description="TRR-0008 prospective plan")
        if plan.get("task_id") != contract.TASK_ID:
            raise RegisterError("prospective plan task identity changed")
        planned_counts = plan.get("records_by_domain")
        if planned_counts is not None and dict(planned_counts) != records_by_domain:
            raise RegisterError("prospective plan record counts do not match observations")
    timing_plan_record = None
    timing_receipt_record = None
    if timing_plan_path is not None:
        timing_plan_record = _record(timing_plan_path, root=root, description="TRR-0008 timing plan")
        timing_plan = contract.load_json(timing_plan_path, description="TRR-0008 timing plan")
        if timing_plan.get("task_id") != contract.TASK_ID or timing_plan.get("schema") != "token-reconstruction.trr0008-timing-plan.v1":
            raise RegisterError("timing plan identity changed")
    if timing_receipt_path is not None:
        timing_receipt_record = _record(timing_receipt_path, root=root, description="TRR-0008 timing receipt")
        timing_receipt = contract.load_json(timing_receipt_path, description="TRR-0008 timing receipt")
        if (
            timing_receipt.get("task_id") != contract.TASK_ID
            or timing_receipt.get("schema") != "token-reconstruction.trr0008-balanced-timing.v1"
            or timing_receipt.get("status") != "TIMING_COMPLETE"
            or timing_receipt.get("truth_opened") is not False
            or timing_receipt.get("equivalence", {}).get("status") != "PASS"
            or timing_receipt.get("configuration", {}).get("blocks") != 40
        ):
            raise RegisterError("timing receipt is not the final truth-free precision40 qualification")

    registration: dict[str, Any] = {
        "schema": contract.REGISTRATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_METHOD_AND_INPUT_BINDING_NO_TRUTH",
        "repository_root": str(root),
        "code_commit": _git_head(root),
        "method_order": list(contract.METHOD_ORDER),
        "method_ids": list(contract.METHOD_ORDER),
        "timing_control": alias,
        "method_freeze": freeze_record,
        "method_freeze_state_sha256": {
            method_id: _state_binding(freeze, method_id)["sha256"]
            for method_id in (*contract.METHOD_ORDER, contract.TIMING_CONTROL_METHOD_ID)
        },
        "plan": plan_record,
        "timing_plan": timing_plan_record,
        "timing_receipt": timing_receipt_record,
        "observation_manifest": observations_record,
        "observation_schema": checked_observations.get("schema"),
        "cell_order": list(contract.CELL_ORDER),
        "records_by_domain": records_by_domain,
        "geometry": dict(contract.STATIC_GEOMETRY),
        "methods": methods,
        "runtime_assets": {"normalized_public_E": e_record},
        "output_root": str(resolved_output),
        "numerical_settings": dict(contract.NUMERICAL_SETTINGS),
        "resource_guard": dict(contract.RESOURCE_GUARD),
        "code_bindings": code_bindings,
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "candidate_arrays_persisted": False,
    }
    contract.validate_registration(registration)
    return registration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--timing-plan", type=Path)
    parser.add_argument("--timing-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = build_registration(
            repository_root=args.repository_root,
            method_freeze_path=args.method_freeze,
            observation_manifest_path=args.observations,
            output_root=args.output_root,
            plan_path=args.plan,
            timing_plan_path=args.timing_plan,
            timing_receipt_path=args.timing_receipt,
        )
        contract.write_create_only(args.output, value)
    except (RegisterError, contract.ContractError) as exc:
        print(f"TRR-0008 registration failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
