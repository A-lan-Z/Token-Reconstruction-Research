"""Executable wrapper around the concrete TRR-0010 provider and fit entrypoint.

This CLI wires the existing A2 lease/resource guard and
``trr0010_p09_provider.build_inputs`` API to
``trr0010_directional_fit.run_directional_arms``.  It does not provide a
fallback loader or schedule.  The provider must expose the arm-aware handoff
fields documented in ``production_fit_entrypoint_v1.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

from trr0010_directional_fit import bind_concrete_provider, run_directional_arm, run_directional_arms
from trr0010_p09_qualifier import enforce_resource_guard, resource_snapshot, validate_exclusive_lease


class ProductionCLIError(RuntimeError):
    """Raised before a production fit when lease/provider inputs are invalid."""


DIAGNOSTIC_SCHEMA = "token-reconstruction.trr-p09-fixed-diagnostic-binding.v1"
DIAGNOSTIC_TASK_ID = "TRR-P09"
DIAGNOSTIC_STATUS = "PASS_FIXED_DIAGNOSTIC64_BOUND_PUBLIC_INPUTS_ONLY"
DIAGNOSTIC_FILE_SHA256 = "8b899bc168b55570e4cfd526ff4e8656b70050e840386cba9390ad6adb1f54c3"
DIAGNOSTIC_BANK_DIGESTS = {
    "B0": "2dba4fdb243f742496cc010985820205a6d36f8c29b7ed3058369322d181f907",
    "B1": "670624a8ef439fe11b250f1ba432d87922d6ff53da38ff8514203880237b6933",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProductionCLIError(f"cannot hash diagnostic binding: {path}") from exc
    return digest.hexdigest()


def _load_diagnostic_binding(path: Path) -> dict[str, Any]:
    """Load only the approved public 64-row diagnostic binding.

    The rows themselves are deliberately omitted from the normalized handoff:
    the provider owns loading the bound public source and maps diagnostics by
    ``global_row``.  The descriptor's bytes/hash and both declared index
    digests remain part of the fit receipt.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ProductionCLIError(f"diagnostic binding is not a file: {path}")
    try:
        byte_count = int(path.stat().st_size)
    except OSError as exc:
        raise ProductionCLIError(f"cannot stat diagnostic binding: {path}") from exc
    file_sha256 = _sha256_file(path)
    if file_sha256 != DIAGNOSTIC_FILE_SHA256:
        raise ProductionCLIError("diagnostic binding file SHA-256 differs from approved public receipt")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionCLIError(f"cannot load diagnostic binding JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProductionCLIError("diagnostic binding must be a JSON object")
    if payload.get("schema") != DIAGNOSTIC_SCHEMA or payload.get("task_id") != DIAGNOSTIC_TASK_ID:
        raise ProductionCLIError("diagnostic binding schema/task identity is not the approved P09 binding")
    if payload.get("status") != DIAGNOSTIC_STATUS:
        raise ProductionCLIError("diagnostic binding is not the approved PASS public-only receipt")
    banks = payload.get("banks")
    if not isinstance(banks, dict) or set(banks) != set(DIAGNOSTIC_BANK_DIGESTS):
        raise ProductionCLIError("diagnostic binding must contain exactly B0 and B1 banks")
    normalized_banks: dict[str, dict[str, Any]] = {}
    for bank, expected_digest in DIAGNOSTIC_BANK_DIGESTS.items():
        entry = banks.get(bank)
        if not isinstance(entry, dict):
            raise ProductionCLIError(f"diagnostic binding {bank} entry is missing")
        if entry.get("bank") != bank:
            raise ProductionCLIError(f"diagnostic binding {bank} bank label is inconsistent")
        if entry.get("record_count") != 64:
            raise ProductionCLIError(f"diagnostic binding {bank} must contain 64 records")
        indices = entry.get("global_indices")
        if (
            not isinstance(indices, list)
            or len(indices) != 64
            or any(isinstance(value, bool) or not isinstance(value, int) for value in indices)
            or indices != sorted(set(indices))
        ):
            raise ProductionCLIError(
                f"diagnostic binding {bank} global indices must be 64 sorted unique integers"
            )
        if entry.get("global_indices_sha256") != expected_digest:
            raise ProductionCLIError(f"diagnostic binding {bank} index digest differs from approved receipt")
        normalized_banks[bank] = {
            "record_count": 64,
            "global_indices_sha256": expected_digest,
            "mapping": "global_row",
            "rows_order": "descriptive_rank_only_do_not_zip",
        }
    verification = payload.get("verification")
    if not isinstance(verification, dict) or verification.get("evaluation_truth_opened") is not False:
        raise ProductionCLIError("diagnostic binding does not prove that evaluation truth stayed closed")
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "task_id": DIAGNOSTIC_TASK_ID,
        "status": DIAGNOSTIC_STATUS,
        "path": str(path),
        "bytes": byte_count,
        "sha256": file_sha256,
        "fit_record_count": 64,
        "full_bank_endpoint_steps": [0, 13000],
        "selection_isolated": True,
        "global_row_mapping": "global_row; descriptive rows are not zipped",
        "banks": normalized_banks,
        "rule_seed": 4010,
        "evaluation_truth_opened": False,
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionCLIError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ProductionCLIError(f"{label} must contain a JSON object: {path}")
    return value


def _load_callable(spec: str) -> Any:
    if ":" not in spec:
        raise ProductionCLIError("provider must use MODULE:CALLABLE syntax")
    module_name, function_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise ProductionCLIError(f"cannot load provider {spec!r}") from exc
    if not callable(function):
        raise ProductionCLIError(f"provider {spec!r} is not callable")
    return function


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    lease_receipt = _load_json(Path(args.lease), label="lease receipt")
    dry_cpu = bool(getattr(args, "configuration_dry_run", False)) and str(getattr(args, "device", "")) == "cpu"
    if dry_cpu:
        # This branch validates only serialized metadata and intentionally does
        # not claim a compute lease; no model, bank payload, or update is
        # touched by configuration_dry_run.
        lease_caps = {"device": "cpu", "mode": "configuration_dry_run"}
    else:
        lease_caps = validate_exclusive_lease(lease_receipt)
    device = torch.device(str(lease_caps["device"]))
    if args.device is not None and str(device) != str(args.device):
        raise ProductionCLIError("--device differs from the device in the exclusive lease")
    requested_arm = str(getattr(args, "arm", "both"))
    if requested_arm not in {"both", "current_directional", "expanded_directional"}:
        raise ProductionCLIError("--arm must be both, current_directional, or expanded_directional")
    binding_receipts: dict[str, dict[str, Any]] = {}
    binding_paths = {
        "current_directional": getattr(args, "binding_current", None),
        "expanded_directional": getattr(args, "binding_expanded", None),
    }
    selected_arms = ("current_directional", "expanded_directional") if requested_arm == "both" else (requested_arm,)
    for arm_name in selected_arms:
        binding_path = binding_paths[arm_name]
        if binding_path is None:
            flag = "--binding-current" if arm_name == "current_directional" else "--binding-expanded"
            raise ProductionCLIError(f"{flag} is required for --arm {arm_name}")
        binding_receipts[arm_name] = _load_json(Path(binding_path), label=f"{arm_name} binding receipt")
    diagnostic_binding = _load_diagnostic_binding(Path(args.diagnostic_binding))
    provider = _load_callable(args.provider)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if bool(getattr(args, "configuration_dry_run", False)):
        module_name = getattr(provider, "__module__", "")
        module = sys.modules.get(module_name)
        if module is None and module_name:
            module = importlib.import_module(module_name)
        dry_run = getattr(module, "configuration_dry_run", None) if module is not None else None
        if not callable(dry_run):
            raise ProductionCLIError("provider does not expose configuration_dry_run")
        result = dry_run(
            binding_receipts=binding_receipts,
            diagnostic_binding=diagnostic_binding,
            lease_caps=lease_caps,
            device=device,
            output_root=output_root,
            arm_name=None if requested_arm == "both" else requested_arm,
        )
        receipt_path = output_root / "configuration_dry_run.json"
        if receipt_path.exists():
            raise ProductionCLIError(f"configuration dry-run is create-only: {receipt_path}")
        output_root.mkdir(parents=True, exist_ok=True)
        receipt = {**result, "command": list(sys.argv), "truth_opened": False, "model_allocated": False, "updates": False}
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return receipt
    started = time.perf_counter()
    guard_checks: list[dict[str, Any]] = []

    def guard(stage: str) -> None:
        snapshot = resource_snapshot(device=device, output_root=output_root)
        enforce_resource_guard(snapshot, lease_caps, started=started)
        guard_checks.append({"stage": str(stage), "snapshot": snapshot})

    guard("before_provider")
    build_inputs = bind_concrete_provider(
        provider,
        binding_receipts=binding_receipts,
        lease_caps=lease_caps,
        device=device,
        preparation_guard=guard,
        diagnostic_binding=diagnostic_binding,
    )

    def guarded_builder(**kwargs: Any) -> dict[str, Any]:
        value = build_inputs(**kwargs)
        if not isinstance(value, dict):
            value = dict(value)
        if value.get("diagnostic_binding") is None:
            value["diagnostic_binding"] = diagnostic_binding
        provider_guard = value.get("resource_guard_callback")

        def runtime_guard(stage: str) -> None:
            guard(f"runtime:{stage}")
            if callable(provider_guard):
                provider_guard(stage)

        value["resource_guard_callback"] = runtime_guard
        return value

    if requested_arm == "both":
        result = run_directional_arms(
            guarded_builder,
            output_root=output_root,
            deadline_seconds=float(args.deadline_seconds),
            command=list(sys.argv),
        )
    else:
        result = run_directional_arm(
            guarded_builder,
            arm_name=requested_arm,
            output_root=output_root,
            deadline_seconds=float(args.deadline_seconds),
            command=list(sys.argv),
        )
    guard("after_fit")
    guard_receipt = {
        "schema": "token-reconstruction.trr0010-production-resource-guard.v1",
        "task_id": "TRR-0010",
        "status": "PASS",
        "device": str(device),
        "lease_caps": dict(lease_caps),
        "checks": guard_checks,
        "diagnostic_binding": {
            "path": diagnostic_binding["path"],
            "bytes": diagnostic_binding["bytes"],
            "sha256": diagnostic_binding["sha256"],
            "banks": diagnostic_binding["banks"],
        },
        "truth_opened": False,
        "command": list(sys.argv),
    }
    guard_path = output_root / "resource_guard_receipt.json"
    if guard_path.exists():
        raise ProductionCLIError(f"resource guard receipt is create-only: {guard_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    guard_path.write_text(json.dumps(guard_receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    result["guard_checks"] = guard_checks
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="trr0010_production_provider:build_inputs")
    parser.add_argument("--arm", choices=("both", "current_directional", "expanded_directional"), default="both")
    parser.add_argument("--binding-current", type=Path)
    parser.add_argument("--binding-expanded", type=Path)
    parser.add_argument("--diagnostic-binding", required=True, type=Path)
    parser.add_argument("--lease", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--deadline-seconds", type=float, default=7200.0)
    parser.add_argument("--configuration-dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_cli(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
