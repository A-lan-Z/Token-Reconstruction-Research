#!/usr/bin/env python3
"""Assemble the owner-frozen, truth-free TRR-0010 design.

The assembler is deliberately small: it binds already reviewed method rows,
rules, source inventories, and opaque identity ledgers into the exact design
consumed by ``trr0010_select_public`` and ``trr0010_eval_register``.  It never
selects records, loads a tokenizer/model, reads observations, or opens truth.
The final B1 ledger is a required argument so an absent or provisional ledger
cannot be silently substituted.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_runner as runner


METHOD_ORDER = list(gate.METHOD_ORDER)
REQUIRED_CODE = (
    "scripts/trr0010_eval_gate.py",
    "scripts/trr0010_eval_runner.py",
    "scripts/trr0010_p09_watchdog.py",
    "scripts/trr0009_eval_capture.py",
    "scripts/trr0005_produce_confirmation.py",
    "src/token_reconstruction/public_activation.py",
    "src/token_reconstruction/trr0005_joint_decoder.py",
    "src/token_reconstruction/trr0007_positionwise.py",
    "scripts/trr0004_predict_confirmation.py",
    "scripts/trr0003_footing_compare.py",
    "scripts/trr0010_eval_register.py",
    "scripts/trr0010_select_public.py",
    "scripts/trr0010_eval_capture.py",
    "scripts/trr0010_p09_fixed_loader.py",
    "scripts/trr0010_prepare_truth.py",
    "scripts/trr0010_score.py",
    "scripts/trr0010_score_cli.py",
    "scripts/trr0010_cost_evidence.py",
    "scripts/trr0010_eval_assemble_methods.py",
    "scripts/trr0010_freeze_design.py",
)
DEFAULT_ASSEMBLY = Path("experiments/TRR-0010/setup/final_method_row_assembly_v2.json")
DEFAULT_RULES = Path("experiments/TRR-0010/planning/final_scoring_rules_binding_v1.json")
DEFAULT_OUTPUT = Path("experiments/TRR-0010/evaluation/final_evaluation_design.json")
DEFAULT_OPAQUE = (
    Path("/tmp/trr-p04/experiments/TRR-P04/coordination/reservation_hashes.json"),
    Path("experiments/TRR-0008/planning/approved_opaque/p06_opaque_source_sequence_reservation.json"),
    Path("experiments/TRR-0008/selection/source_exclusions.json"),
    Path("experiments/TRR-0010/planning/count_scan/p08_union_sanitized.json"),
)


def _root(value: Path | str) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"repository root is unavailable: {root}")
    return root


def _path(value: Path | str, *, root: Path) -> Path:
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _record(value: Path | str, *, root: Path, description: str) -> dict[str, Any]:
    path = _path(value, root=root)
    if not path.is_file():
        raise ValueError(f"{description} is unavailable: {path}")
    readonly = False
    try:
        path.relative_to(root)
    except ValueError:
        readonly = True
    return gate.file_record(path, root=root, readonly=readonly)


def _json(value: Path | str, *, root: Path, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(value, root=root, description=description)
    try:
        payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must be a JSON object")
    return record, dict(payload)


def _git_head(root: Path) -> str:
    value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("current HEAD is not a full lowercase commit hash")
    return value


def _contract_rules(rules: Mapping[str, Any], *, rules_record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the frozen scoring contract without rewriting threshold prose."""
    statistical = rules.get("statistical_procedure")
    useful = rules.get("useful_outcome_proposal")
    cost = rules.get("safeguards_and_cost")
    constants = rules.get("referenced_constants")
    if not all(isinstance(x, Mapping) for x in (statistical, useful, cost, constants)):
        raise ValueError("scoring-rules binding lacks its frozen sections")
    geometry = constants.get("training_geometry_and_schedule")
    if not isinstance(geometry, Mapping):
        raise ValueError("scoring-rules binding lacks training geometry")
    return {
        "status": "FROZEN",
        "source_binding": dict(rules_record),
        "source_status": rules.get("status"),
        "rules": {
            "statistical_procedure": deepcopy(dict(statistical)),
            "useful_outcome_proposal": deepcopy(dict(useful)),
            "safeguards_and_cost": deepcopy(dict(cost)),
            "training_geometry_and_schedule": deepcopy(dict(geometry)),
            "method_order": list(METHOD_ORDER),
            "paired_cells": list(gate.CELL_ORDER),
            "post_score_changes_forbidden": True,
        },
    }


def _code_bindings(*, root: Path) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for relative in REQUIRED_CODE:
        bindings[relative] = _record(relative, root=root, description=f"source binding {relative}")
    return bindings


def _contract_bindings(*, root: Path, assembly: Path, rules: Path) -> dict[str, Any]:
    paths = {
        "method_row_assembly": assembly,
        "scoring_rules": rules,
        "shared_stage3_contract": Path("experiments/TRR-0010/planning/shared_stage3_contract_v1.json"),
        "shared_contract_v4": Path("experiments/TRR-0010/planning/shared_contract.proposal.json"),
        "source_inventory": Path("experiments/TRR-0010/planning/count_scan/inventory_7000_trr9_p08_union.json"),
        "combined_exclusion_receipt": Path("experiments/TRR-0010/planning/count_scan/combined_exclusion_receipt.json"),
        "qualified_bank_binding": Path("experiments/TRR-0010/setup/qualified_bank_manifest_binding_r1.json"),
        "support_binding": Path("experiments/TRR-0010/setup/public_support_binding_r1.json"),
        "capture_binding": Path("experiments/TRR-0010/setup/stage1_capture_binding_r1.json"),
        "capture_receipt": Path("experiments/TRR-0010/setup/capture_receipt_r2.json"),
        "validation_audit": Path("experiments/TRR-0010/setup/public_validation_r1_audit.json"),
        "schedule_binding": Path("experiments/TRR-0010/setup/common_schedules_r1.json"),
        "a1_a2_fixture": Path("experiments/TRR-0010/evaluation/a1_a2_opened_fixture_equivalence_v3/result.json"),
        "execution_commands": Path("experiments/TRR-0010/setup/final_evaluation_commands_v2.json"),
    }
    return {name: _record(path, root=root, description=f"contract binding {name}") for name, path in paths.items()}


def _resource_guard(execution: Mapping[str, Any]) -> dict[str, Any]:
    # These are the runner's registered whole-matrix caps.  Do not copy the
    # rejected per-method 900/2400-second proposal into this design.
    capture_policy = execution.get("capture", {}).get("resource_policy") if isinstance(execution.get("capture"), Mapping) else None
    prediction_policy = execution.get("prediction", {}).get("resource_policy") if isinstance(execution.get("prediction"), Mapping) else None
    if not isinstance(capture_policy, Mapping) or not isinstance(prediction_policy, Mapping):
        raise ValueError("reviewed execution manifest lacks capture/prediction guard policies")
    return {
        "inference": dict(runner.REGISTERED_INFERENCE_CAPS),
        "capture": deepcopy(dict(capture_policy)),
        "prediction": deepcopy(dict(prediction_policy)),
        "fit": {
            "whole_arm_max_seconds": 7200,
            "fixed_cuda_reserved_limit_bytes": 8 * 2**30,
            "directional_cuda_reserved_limit_bytes": 10 * 2**30,
            "gpu_free_floor_bytes_runtime": 2 * 2**30,
            "host_available_floor_bytes_runtime": 8 * 2**30,
            "host_rss_limit_bytes": 12 * 2**30,
            "disk_free_floor_bytes": 20 * 2**30,
            "output_bytes_limit": 5 * 2**30,
        },
        "failure_policy": "fail closed; include preparation, inference, serialization and I/O; preserve a failure receipt",
    }


def assemble(
    *,
    root: Path,
    assembly_path: Path,
    rules_path: Path,
    b1_ledger_path: Path,
    opaque_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    assembly_record, assembly = _json(assembly_path, root=root, description="method-row assembly")
    rules_record, rules = _json(rules_path, root=root, description="frozen scoring-rules binding")
    execution_record, execution = _json(Path("experiments/TRR-0010/setup/final_evaluation_commands_v2.json"), root=root, description="reviewed final-evaluation command manifest")
    if assembly.get("task_id") != "TRR-0010" or not isinstance(assembly.get("methods"), Mapping):
        raise ValueError("method-row assembly is not a TRR-0010 assembly")
    if list(assembly.get("method_order", ())) != METHOD_ORDER:
        raise ValueError("method-row assembly order differs from the gate")
    methods = deepcopy(dict(assembly["methods"]))
    if set(methods) != set(METHOD_ORDER):
        raise ValueError("method-row assembly does not contain all six contenders")
    for method_id in METHOD_ORDER:
        row = methods[method_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"method row is malformed: {method_id}")
        row["frozen"] = True
        row["state_frozen"] = True
        row["freeze_status"] = "FROZEN_PRE_SOURCE_SELECTION"
        row["decision_role"] = gate.METHOD_ROLES[method_id]

    final_b1 = _record(b1_ledger_path, root=root, description="final B1 exclusion ledger")
    opaque = [_record(path, root=root, description=f"approved opaque ledger {index}") for index, path in enumerate(opaque_paths)]
    inventory = _record("experiments/TRR-0010/planning/count_scan/inventory_7000_trr9_p08_union.json", root=root, description="source inventory")
    exclusion_receipt = _record("experiments/TRR-0010/planning/count_scan/combined_exclusion_receipt.json", root=root, description="combined exclusion receipt")

    contract_bindings = _contract_bindings(root=root, assembly=assembly_path, rules=rules_path)
    code_bindings = _code_bindings(root=root)
    design = {
        "schema": "token-reconstruction.trr0010-final-evaluation-design.v1",
        "task_id": "TRR-0010",
        "status": "FROZEN_TRR0010_FINAL_EVALUATION_DESIGN",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "code_commit": _git_head(root),
        "method_order": METHOD_ORDER,
        "contenders_frozen": True,
        "methods": methods,
        "code_bindings": code_bindings,
        "contract_bindings": contract_bindings,
        "decision_rules": _contract_rules(rules, rules_record=rules_record),
        "resource_guard": _resource_guard(execution),
        "numerical_settings": {
            "fit_context_width": 192,
            "natural_validation_context_width": 128,
            "batch_records": 8,
            "position_draws_per_update": 512,
            "seed": 4010,
            "optimizer": "AdamW",
            "decoder_learning_rate": 2e-4,
            "directional_learning_rate": 1e-4,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "foreach": False,
            "checkpoint_grid": [0, 1000, 2000, 4000, 8000, 12000, 13000],
            "exact_update_count": 13000,
        },
        "final_evaluation": {
            "exclusion_bindings": {
                "final_b1": {**final_b1, "role": "final_b1_public_identity_exclusion_ledger"},
                "approved_opaque_ledgers": [
                    {**record, "role": "approved_opaque_identity_ledger"} for record in opaque
                ],
            },
            "source_inventory": inventory,
            "combined_exclusion_receipt": exclusion_receipt,
            "source_ranges_half_open": {"finance": [12000, 20000], "pile": [7000, 10000]},
            "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
            "target_conditions": list(gate.TARGET_ORDER),
            "paired_sources_across_targets": True,
            "capture_geometry": {
                "batch_records": 8,
                "sequence_tokens": 192,
                "stored_sequence_tokens": 128,
                "retain_first_128": True,
                "hidden_size": 2048,
                "vocabulary_size": 128256,
                "scored_post_bos_tokens": 127,
            },
            "selection_plan": {
                "selection_seed": 5005,
                "identity_only": True,
                "selection_after_design_freeze": True,
                "design": "experiments/TRR-0010/evaluation/final_evaluation_design.json",
                "output": "experiments/TRR-0010/evaluation/source_selection.json",
                "exclusions_output": "experiments/TRR-0010/evaluation/source_exclusions.json",
            },
            "capture_plan": {
                "watchdog_root": "experiments/TRR-0010/evaluation/public_capture_watchdog_r1",
                "producer_output_root": "experiments/TRR-0010/evaluation/public_capture_watchdog_r1/producer_capture",
                "observations_root": "experiments/TRR-0010/evaluation/public_capture_watchdog_r1/observations_v1",
                "observation_manifest": "experiments/TRR-0010/evaluation/public_capture_watchdog_r1/observations_v1/observations.json",
                "panel": "experiments/TRR-0010/evaluation/public_capture_watchdog_r1/observations_v1/panel.json",
                "capture_receipt": "experiments/TRR-0010/evaluation/public_capture_watchdog_r1/observations_v1/capture.json",
                "requires_first_128_retention_marker": True,
            },
            "registration_plan": {
                "output": "experiments/TRR-0010/evaluation/registration.json",
                "frequency_reference_B0": "experiments/TRR-0010/evaluation/frequency_reference_B0.json",
                "frequency_reference_B1": "experiments/TRR-0010/evaluation/frequency_reference_B1.json",
                "observation_manifest": "experiments/TRR-0010/evaluation/public_capture_watchdog_r1/observations_v1/observations.json",
                "panel": "experiments/TRR-0010/evaluation/public_capture_watchdog_r1/observations_v1/panel.json",
                "capture": "experiments/TRR-0010/evaluation/public_capture_watchdog_r1/observations_v1/capture.json",
                "timing_plan": "experiments/TRR-0010/evaluation/timing_plan.json",
                "prediction_root": "experiments/TRR-0010/evaluation/public_prediction_watchdog_r1",
                "requires_all_24_predictions_before_truth": True,
            },
            "opened_fixture_equivalence": contract_bindings["a1_a2_fixture"],
        },
        "truth_boundary": {
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "source_text_written": False,
            "token_ids_written": False,
            "candidate_arrays_persisted": False,
            "source_selection_created": False,
            "public_predictions_created": False,
            "truth_creation_authorized_only_after_public_freeze": True,
        },
        "assembly_evidence": {
            "method_row_assembly": assembly_record,
            "scoring_rules_binding": rules_record,
            "execution_commands": execution_record,
            "final_b1_ledger_required_input": final_b1,
            "approved_opaque_ledger_count": len(opaque),
        },
    }
    output = _path(output_path, root=root)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output is create-only: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(design, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return _record(output, root=root, description="frozen TRR-0010 design") | {"design": design}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--assembly", type=Path, default=DEFAULT_ASSEMBLY)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--b1-exclusion-ledger", type=Path, required=True)
    parser.add_argument("--opaque-ledger", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = assemble(
            root=_root(args.repository_root),
            assembly_path=_path(args.assembly, root=_root(args.repository_root)),
            rules_path=_path(args.rules, root=_root(args.repository_root)),
            b1_ledger_path=_path(args.b1_exclusion_ledger, root=_root(args.repository_root)),
            opaque_paths=[_path(x, root=_root(args.repository_root)) for x in (args.opaque_ledger or DEFAULT_OPAQUE)],
            output_path=args.output,
        )
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError) as exc:
        print(f"TRR-0010 design assembly failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ("path", "bytes", "sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
