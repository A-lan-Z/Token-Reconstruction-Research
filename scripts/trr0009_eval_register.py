"""Build a complete, truth-free TRR-0009 evaluation registration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from scripts import trr0009_eval_contract as contract


class RegisterError(contract.ContractError):
    pass


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RegisterError(f"repository root is unavailable: {root}")
    return root


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RegisterError("cannot resolve registration code commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RegisterError("registration code commit is not a full hash")
    return value


def _public_json(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    record = _record(path, root=root, description=description)
    try:
        value = contract.load_json(path, description=description)
    except contract.ContractError as exc:
        raise RegisterError(str(exc)) from exc
    for key in ("truth_opened", "target_labels_loaded", "source_text_loaded", "source_text_written", "token_ids_written", "candidate_arrays_persisted"):
        if value.get(key) is True:
            raise RegisterError(f"{description} records forbidden access: {key}")
    return record


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    try:
        return contract.validate_file_record(
            {"path": str(Path(path).expanduser().resolve()), "bytes": Path(path).stat().st_size, "sha256": contract.sha256_file(Path(path))},
            repository_root=root,
            description=description,
            verify=True,
        )
    except (OSError, contract.ContractError) as exc:
        raise RegisterError(str(exc)) from exc


def _normalize_loader(loader: Mapping[str, Any], *, root: Path, method_id: str) -> dict[str, Any]:
    result = dict(loader)
    for key in ("path_args", "tensor_args"):
        values = result.get(key, {})
        if not isinstance(values, Mapping):
            raise RegisterError(f"{method_id} loader {key} is malformed")
        result[key] = {
            str(arg): _record(Path(str(binding["path"])), root=root, description=f"{method_id} loader {key}.{arg}")
            if isinstance(binding, Mapping) and "path" in binding
            else binding
            for arg, binding in values.items()
        }
    return result


def build_registration(
    *,
    repository_root: Path,
    plan_path: Path,
    observation_manifest_path: Path,
    source_selection_path: Path,
    capture_receipt_path: Path,
    frequency_reference_path: Path,
    embedding_path: Path,
    timing_plan_path: Path,
    method_rows: Mapping[str, Mapping[str, Any]],
    records_by_domain: Mapping[str, int],
    code_bindings: Sequence[Path | Mapping[str, Any]],
    output_root: str,
    output_path: Path,
    initialization_equivalence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _root(repository_root)
    try:
        plan = contract.load_json(plan_path, description="TRR-0009 evaluation plan")
        observation = contract.load_json(observation_manifest_path, description="TRR-0009 observation manifest")
        timing_plan = contract.load_json(timing_plan_path, description="TRR-0009 timing plan")
    except contract.ContractError as exc:
        raise RegisterError(str(exc)) from exc
    if plan.get("task_id") != contract.TASK_ID or plan.get("truth_opened") is True:
        raise RegisterError("evaluation plan is not a truth-free TRR-0009 plan")
    if timing_plan.get("schema") != contract.TIMING_PLAN_SCHEMA or timing_plan.get("task_id") != contract.TASK_ID:
        raise RegisterError("timing plan schema or task identity changed")
    if timing_plan.get("status") != "FROZEN_TIMING_PLAN_BEFORE_MEASUREMENT" or timing_plan.get("truth_opened") is True:
        raise RegisterError("timing plan was not frozen before measurement")
    try:
        checked_observation = contract.validate_observation_manifest(observation, repository_root=root, verify_assets=True)
    except contract.ContractError as exc:
        raise RegisterError(str(exc)) from exc
    if set(records_by_domain) != set(contract.DOMAIN_ORDER):
        raise RegisterError("records_by_domain keys changed")
    expected_records = {cell: int(records_by_domain[cell.split("__", 1)[0]]) for cell in contract.CELL_ORDER}
    if any(value <= 0 for value in expected_records.values()):
        raise RegisterError("record counts must be positive")
    if {cell: contract.records_for_cell(checked_observation, cell) for cell in contract.CELL_ORDER} != expected_records:
        raise RegisterError("observation counts do not match registration counts")
    rows: list[dict[str, Any]] = []
    for method_id in contract.METHOD_ORDER:
        row = method_rows.get(method_id)
        if not isinstance(row, Mapping):
            raise RegisterError(f"method row is absent: {method_id}")
        if row.get("id") != method_id or row.get("role") != contract.METHOD_ROLES[method_id]:
            raise RegisterError(f"method role changed: {method_id}")
        state_path = Path(str(row.get("state_path", row.get("state", {}).get("path", "")))).expanduser()
        if not state_path.is_absolute():
            state_path = root / state_path
        state = _record(state_path, root=root, description=f"{method_id} state")
        loader = row.get("loader")
        if not isinstance(loader, Mapping):
            raise RegisterError(f"loader is absent: {method_id}")
        rows.append({
            "id": method_id,
            "role": contract.METHOD_ROLES[method_id],
            "kind": str(row.get("kind", "decoder")),
            "cells": list(contract.CELL_ORDER),
            "records_per_cell": expected_records,
            "state": state,
            "loader": _normalize_loader(loader, root=root, method_id=method_id),
            "fit_cost": row.get("fit_cost"),
            "deployment_cost": row.get("deployment_cost"),
        })
    binding_values: list[dict[str, Any]] = []
    for index, value in enumerate(code_bindings):
        if isinstance(value, Mapping):
            path_value = value.get("path")
        else:
            path_value = value
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = root / path
        binding_values.append(_record(path, root=root, description=f"code binding {index}"))
    output = Path(output_root).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    task_root = (root / "experiments" / contract.TASK_ID).resolve()
    try:
        output.relative_to(task_root)
    except ValueError as exc:
        raise RegisterError("output root must be below the TRR-0009 task root") from exc
    registration: dict[str, Any] = {
        "schema": contract.REGISTRATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_EVALUATION_REGISTRATION_BEFORE_TRUTH",
        "repository_root": str(root),
        "code_commit": _git_head(root),
        "code_bindings": binding_values,
        "plan": _record(Path(plan_path), root=root, description="evaluation plan"),
        "observation_manifest": _record(Path(observation_manifest_path), root=root, description="observation manifest"),
        "source_selection": _public_json(Path(source_selection_path), root=root, description="source selection"),
        "capture_receipt": _public_json(Path(capture_receipt_path), root=root, description="capture receipt"),
        "frequency_reference": _record(Path(frequency_reference_path), root=root, description="public fitting-frequency reference"),
        "runtime_embedding": _record(Path(embedding_path), root=root, description="normalized public embedding"),
        "timing_plan": _record(Path(timing_plan_path), root=root, description="timing plan"),
        "cell_order": list(contract.CELL_ORDER),
        "method_order": list(contract.METHOD_ORDER),
        "methods": rows,
        "records_by_domain": {str(key): int(value) for key, value in records_by_domain.items()},
        "geometry": dict(contract.STATIC_GEOMETRY),
        "numerical_settings": dict(contract.NUMERICAL_SETTINGS),
        "resource_guard": dict(contract.RESOURCE_GUARD),
        "output_root": str(output),
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "candidate_arrays_persisted": False,
        "a2_enabled": False,
        "history_enabled": False,
        "initialization_equivalence": dict(initialization_equivalence or {}),
    }
    try:
        record = contract.write_create_only(Path(output_path), registration)
        registration["registration_file"] = record
        return registration | {"registration_sha256": record["sha256"]}
    except contract.ContractError as exc:
        raise RegisterError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True, help="JSON payload containing build_registration arguments")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.payload.read_text())
    build_registration(**payload, output_path=args.registration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
