"""Create a create-only TRR-0007 frozen evaluation registration.

This binder runs after the reviewed plan, public exclusion/source selection,
fresh source-free observations, and all selected student states are frozen.
It records every executable asset and exact hash needed by the runner and
public gate; it never reads target labels or private truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from scripts import trr0007_eval_contract as contract
from scripts import trr0007_bank_ledger as bank_ledger


class RegisterError(contract.ContractError):
    """Raised when registration cannot be made immutable."""


def _root(value: Path) -> Path:
    result = Path(value).expanduser().resolve()
    if result.is_symlink() or not result.is_dir():
        raise RegisterError(f"repository root is unavailable: {result}")
    return result


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RegisterError("cannot resolve registration commit") from exc
    if not contract._COMMIT.fullmatch(value):
        raise RegisterError("registration commit is not a full hash")
    return value


def _file_record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
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


def _mapping_record(value: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    try:
        return contract.validate_file_record(
            value, repository_root=root, description=description, verify=True
        )
    except contract.ContractError as exc:
        raise RegisterError(str(exc)) from exc


def _load_bound_json(path: Path, *, root: Path, schema: str, status: str, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path, root=root, description=description)
    try:
        payload = contract.load_json(Path(record["path"]), description=description)
    except contract.ContractError as exc:
        raise RegisterError(str(exc)) from exc
    if payload.get("schema") != schema or payload.get("task_id") != contract.TASK_ID:
        raise RegisterError(f"{description} schema or task ID changed")
    if payload.get("status") != status:
        raise RegisterError(f"{description} is not complete")
    return record, payload


def _verify_execution_receipts(
    *,
    root: Path,
    method_freeze: Mapping[str, Any],
    method_freeze_record: Mapping[str, Any],
    source_record: Mapping[str, Any],
    exclusion_record: Mapping[str, Any],
    observation_record: Mapping[str, Any],
    capture_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = Path(str(source_record["path"]))
    selection = contract.load_json(selection_path, description="public source selection")
    final_bank = selection.get("final_bank_ledgers")
    final_files = final_bank.get("files") if isinstance(final_bank, Mapping) else None
    if not isinstance(final_files, Mapping) or not all(
        isinstance(final_files.get(key), Mapping) for key in ("exclusion_manifest", "selected_parent_rows", "corpus_plan")
    ):
        raise RegisterError("source selection lacks final v5 bank ledgers")
    try:
        verified_bank = bank_ledger.load_final_bank_ledgers(
            repository_root=root,
            exclusion_manifest=Path(str(final_files["exclusion_manifest"]["path"])),
            selected_parent_rows=Path(str(final_files["selected_parent_rows"]["path"])),
            corpus_plan=Path(str(final_files["corpus_plan"]["path"])),
        )
    except bank_ledger.BankLedgerError as exc:
        raise RegisterError(str(exc)) from exc
    if verified_bank != dict(final_bank):
        raise RegisterError("source selection final v5 bank descriptor changed")
    prefix_ledger = selection.get("public_fitting_prefix_exclusions")
    prefix_file = prefix_ledger.get("file") if isinstance(prefix_ledger, Mapping) else None
    if not isinstance(prefix_file, Mapping) or not isinstance(prefix_file.get("path"), str):
        raise RegisterError("source selection lacks the reviewed v3 fitting-prefix ledger")
    try:
        verified_prefix = bank_ledger.load_prefix_exclusion_ledger(
            repository_root=root, path=Path(str(prefix_file["path"]))
        )
    except bank_ledger.BankLedgerError as exc:
        raise RegisterError(str(exc)) from exc
    if verified_prefix != dict(prefix_ledger):
        raise RegisterError("source selection v3 fitting-prefix ledger descriptor changed")
    if selection.get("method_freeze_sha256") != method_freeze_record["sha256"]:
        raise RegisterError("source selection is bound to a different method freeze")
    if selection.get("method_freeze") != dict(method_freeze_record):
        raise RegisterError("source selection method-freeze descriptor changed")
    exclusions = selection.get("selection_exclusions")
    if not isinstance(exclusions, Mapping) or exclusions.get("sha256") != exclusion_record["sha256"]:
        raise RegisterError("source selection exclusion receipt is not bound")
    observation = contract.load_json(Path(str(observation_record["path"])), description="public observation manifest")
    if observation.get("method_freeze_sha256") != method_freeze_record["sha256"]:
        raise RegisterError("public observations are bound to a different method freeze")
    if not isinstance(observation.get("selection_plan"), Mapping) or observation["selection_plan"].get("sha256") != source_record["sha256"]:
        raise RegisterError("public observation manifest is not bound to source selection")
    capture = contract.load_json(Path(str(capture_record["path"])), description="public capture receipt")
    if not isinstance(capture.get("selection_plan"), Mapping) or capture["selection_plan"].get("sha256") != source_record["sha256"]:
        raise RegisterError("capture receipt is not bound to source selection")
    if not isinstance(capture.get("observations"), Mapping) or capture["observations"].get("sha256") != observation_record["sha256"]:
        raise RegisterError("capture receipt is not bound to observation manifest")
    if capture.get("method_freeze_sha256") != method_freeze_record["sha256"]:
        raise RegisterError("capture receipt is bound to a different method freeze")
    if capture.get("truth_opened") is True:
        raise RegisterError("capture receipt records truth access")
    execution = capture.get("execution")
    if not isinstance(execution, Mapping) or execution.get("truth_opened") is not False:
        raise RegisterError("capture receipt records truth access")
    return selection, capture


def _code_bindings(root: Path) -> list[dict[str, Any]]:
    result = []
    for role, relative in contract.CODE_BINDING_SPECS:
        result.append(_file_record(root / relative, root=root, description=f"code binding {role}") | {"role": role, "path": relative})
    return result


def _student_row(
    method_id: str,
    state: Mapping[str, Any],
    *,
    root: Path,
    method_freeze_sha256: str,
) -> dict[str, Any]:
    if method_id not in contract.STUDENT_METHOD_IDS:
        raise RegisterError(f"unknown student method: {method_id}")
    model_id = contract.STUDENT_METHOD_MODEL_IDS[method_id]
    state_record = _mapping_record(state, root=root, description=f"{method_id} state")
    return {
        "id": method_id,
        "role": "student",
        "kind": "decoder",
        "support": contract.STUDENT_SUPPORT[method_id],
        "capacity": contract.STUDENT_CAPACITY[method_id],
        "cells": list(contract.CELL_ORDER),
        "records_per_cell": contract.RECORDS_PER_DOMAIN,
        "candidate_policy": "forbidden",
        "state": state_record,
        "method_freeze_sha256": method_freeze_sha256,
        "loader": {
            "module": "token_reconstruction.trr0007_positionwise",
            "function": "load_positionwise_model_state",
            "kwargs": {
                "method_id": model_id,
                "hidden_size": contract.HIDDEN_SIZE,
                "vocabulary_size": contract.VOCAB_SIZE,
                "context_width": contract.STORED_SEQUENCE_TOKENS,
            },
        },
    }


def build_registration(
    *,
    repository_root: Path,
    plan_path: Path,
    source_selection_path: Path,
    exclusion_manifest_path: Path,
    observation_manifest_path: Path,
    capture_receipt_path: Path,
    method_freeze_path: Path,
    state_paths: Mapping[str, Path] | None,
    frequency_reference_path: Path,
    public_model_snapshot: Path,
    lens_path: Path,
    reference_path: Path,
    output_root: str,
    fit_costs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _root(repository_root)
    plan_path = Path(plan_path).expanduser().resolve()
    source_selection_path = Path(source_selection_path).expanduser().resolve()
    exclusion_manifest_path = Path(exclusion_manifest_path).expanduser().resolve()
    observation_manifest_path = Path(observation_manifest_path).expanduser().resolve()
    plan = contract.load_json(plan_path, description="evaluation plan")
    contract.validate_plan(plan)
    if plan.get("status") != "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION":
        raise RegisterError("plan must be frozen before registration")
    plan_record = _file_record(plan_path, root=root, description="evaluation plan")
    source_record = _file_record(source_selection_path, root=root, description="public source selection")
    exclusion_record = _file_record(exclusion_manifest_path, root=root, description="public exclusion manifest")
    observation_record = _file_record(observation_manifest_path, root=root, description="public observation manifest")
    capture_record = _file_record(capture_receipt_path, root=root, description="public capture receipt")
    method_freeze_record, method_freeze, frozen_states = contract.load_method_freeze(
        method_freeze_path, repository_root=root, verify_assets=True
    )
    frequency_reference_record = _file_record(
        frequency_reference_path, root=root, description="public frequency reference"
    )
    frequency_reference = contract.load_json(
        Path(frequency_reference_record["path"]), description="public frequency reference"
    )
    if (
        frequency_reference.get("schema") != contract.FREQUENCY_REFERENCE_SCHEMA
        or frequency_reference.get("task_id") != "TRR-0005"
        or frequency_reference.get("status") != "PUBLIC_FITTING_FREQUENCY_REFERENCES"
        or not isinstance(frequency_reference.get("frequency_references"), Mapping)
        or not isinstance(frequency_reference["frequency_references"].get("enriched"), Mapping)
    ):
        raise RegisterError("public frequency reference schema or enriched map changed")
    if plan_record["sha256"] != contract.sha256_file(plan_path):
        raise RegisterError("plan hash changed while registering")
    if state_paths is not None and set(state_paths) != set(contract.STUDENT_METHOD_IDS):
        raise RegisterError("all four student states are required when supplied")
    if state_paths is None:
        state_paths = {method_id: Path(frozen_states[method_id]["path"]) for method_id in contract.STUDENT_METHOD_IDS}
    bound_states: dict[str, dict[str, Any]] = {}
    for method_id in contract.STUDENT_METHOD_IDS:
        supplied = _file_record(state_paths[method_id], root=root, description=f"{method_id} state")
        if supplied != frozen_states[method_id]:
            raise RegisterError(f"{method_id} state does not match the frozen method ledger")
        bound_states[method_id] = supplied
    selection, _capture = _verify_execution_receipts(
        root=root,
        method_freeze=method_freeze,
        method_freeze_record=method_freeze_record,
        source_record=source_record,
        exclusion_record=exclusion_record,
        observation_record=observation_record,
        capture_record=capture_record,
    )
    reference_state = _mapping_record(
        {
            "path": contract.REFERENCE_STATE_PATH,
            "bytes": contract.REFERENCE_STATE_BYTES,
            "sha256": contract.REFERENCE_STATE_SHA256,
        },
        root=root,
        description="retained reference state",
    )
    # File verification resolves the path, while the registration contract
    # deliberately binds this retained legacy asset to its declared relative
    # spelling (or to the monkeypatched spelling used by synthetic tests).
    reference_state["path"] = contract.REFERENCE_STATE_PATH
    rows: list[dict[str, Any]] = [
        {
            "id": contract.REFERENCE_METHOD_ID,
            "role": "reference",
            "kind": "decoder",
            "support": "current_enriched",
            "capacity": "trained_diagonal",
            "cells": list(contract.CELL_ORDER),
            "records_per_cell": contract.RECORDS_PER_DOMAIN,
            "candidate_policy": "forbidden",
            "state": reference_state,
            "loader": {
                "module": "token_reconstruction.trr0005_joint_decoder",
                "function": "load_decoder_state",
                "kwargs": {
                    "method_id": "affine_trained_diagonal_attention128",
                    "hidden_size": contract.HIDDEN_SIZE,
                    "vocabulary_size": contract.VOCAB_SIZE,
                    "context_width": contract.STORED_SEQUENCE_TOKENS,
                },
            },
        }
    ]
    for method_id in contract.STUDENT_METHOD_IDS:
        row = _student_row(
            method_id,
            bound_states[method_id],
            root=root,
            method_freeze_sha256=method_freeze_record["sha256"],
        )
        if fit_costs is not None:
            fit_cost = fit_costs.get(method_id)
        else:
            bindings = method_freeze.get("state_bindings")
            binding = bindings.get(method_id) if isinstance(bindings, Mapping) else None
            fit_cost = binding.get("fit_cost") if isinstance(binding, Mapping) else None
        if fit_cost is not None:
            row["fit_cost"] = fit_cost
        rows.append(row)
    rows.append(
        {
            "id": contract.ANCHOR_METHOD_ID,
            "role": "anchor",
            "kind": "a1_a2",
            "cells": list(contract.BASE_CELL_ORDER),
            "records_per_cell": contract.ANCHOR_RECORDS_PER_DOMAIN,
            "proposal_budget": contract.A2_PROPOSAL_K,
            "candidate_budget": contract.A2_K,
            "candidate_policy": "output_only",
            "adapter": {
                "kind": "legacy_trr0003_a1_a2_p0",
                "selection_policy": "fixed_k256_direct_cosine",
                "proposal_max_k": contract.A2_PROPOSAL_K,
                "proposal_chunk": 256,
            },
        }
    )
    runtime_e = _mapping_record(
        {
            "path": contract.PUBLIC_E_PATH,
            "bytes": contract.PUBLIC_E_BYTES,
            "sha256": contract.PUBLIC_E_SHA256,
        },
        root=root,
        description="normalized public E",
    ) | {
        "shape": [contract.VOCAB_SIZE, contract.HIDDEN_SIZE],
        "dtype": "torch.float32",
    }
    snapshot = Path(public_model_snapshot).expanduser().resolve()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RegisterError(f"public model snapshot is unavailable: {snapshot}")
    lens_record = _file_record(Path(lens_path), root=root, description="A1 lens")
    prefix_record = _file_record(Path(reference_path), root=root, description="A1+A2 reference")
    registration: dict[str, Any] = {
        "schema": contract.REGISTRATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_EVALUATION_REGISTRATION",
        "records_per_domain": contract.RECORDS_PER_DOMAIN,
        "anchor_records_per_domain": contract.ANCHOR_RECORDS_PER_DOMAIN,
        "cell_order": list(contract.CELL_ORDER),
        "method_ids": list(contract.METHOD_ORDER),
        "plan": plan_record,
        "plan_sha256": plan_record["sha256"],
        "source_selection": source_record,
        "exclusion_manifest": exclusion_record,
        "observation_manifest": observation_record,
        "capture_receipt": capture_record,
        "method_freeze": method_freeze_record,
        "method_freeze_state_sha256": {
            method_id: state["sha256"] for method_id, state in frozen_states.items()
        },
        "final_bank_ledgers": dict(selection["final_bank_ledgers"]),
        "public_fitting_prefix_exclusions": dict(selection["public_fitting_prefix_exclusions"]),
        "frequency_reference": frequency_reference_record,
        "code_commit": _git_head(root),
        "truth_opened": False,
        "truth_created": False,
        "candidate_arrays_persisted": False,
        "source_text_or_target_labels": False,
        "geometry": {
            "capture_batch_records": contract.CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": contract.CAPTURE_SEQUENCE_TOKENS,
            "stored_sequence_tokens": contract.STORED_SEQUENCE_TOKENS,
            "scored_sequence_tokens": contract.SCORED_SEQUENCE_TOKENS,
            "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
            "hidden_size": contract.HIDDEN_SIZE,
            "vocabulary_size": contract.VOCAB_SIZE,
            "chunk_records": contract.CAPTURE_BATCH_RECORDS,
        },
        "methods": rows,
        "runtime_assets": {
            "normalized_public_E": runtime_e,
            "a1_a2": {
                "public_model_snapshot": {
                    "path": str(snapshot),
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "revision": "9213176726f574b556790deb65791e0c5aa438b6",
                    "local_files_only": True,
                },
                "lens": lens_record,
                "reference": prefix_record,
            },
        },
        "output_root": output_root,
        "timing_contract": {
            "warmup_runs_per_record": 1,
            "measured_runs_per_record": 1,
            "repeat_integrity": "Require warmup and measured predicted IDs to match exactly",
        },
        "resource_guard": {
            "minimum_free_gpu_bytes": contract.MIN_FREE_GPU_BYTES,
            "maximum_reserved_gpu_bytes": contract.MAX_RESERVED_GPU_BYTES,
            "maximum_rss_bytes": contract.MAX_RSS_BYTES,
            "minimum_host_available_bytes": contract.MIN_HOST_AVAILABLE_BYTES,
            "maximum_seconds": contract.MAX_SECONDS,
        },
        "numerical_settings": dict(contract.NUMERICAL_SETTINGS),
        "code_bindings": _code_bindings(root),
    }
    contract.validate_registration(registration)
    _manifest, _parsed, _actual = contract.load_observation_manifest(
        registration, repository_root=root, verify_assets=True
    )
    return registration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--frequency-reference", type=Path, default=Path(contract.FREQUENCY_REFERENCE_PATH))
    parser.add_argument("--state", action="append", default=[], metavar="METHOD_ID=PATH")
    parser.add_argument("--public-model-snapshot", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        states: dict[str, Path] = {}
        for item in args.state:
            if "=" not in item:
                raise RegisterError("--state must use METHOD_ID=PATH")
            method, path = item.split("=", 1)
            if method in states:
                raise RegisterError(f"duplicate state method: {method}")
            states[method] = Path(path)
        value = build_registration(
            repository_root=args.repository_root,
            plan_path=args.plan,
            source_selection_path=args.source_selection,
            exclusion_manifest_path=args.exclusions,
            observation_manifest_path=args.observations,
            capture_receipt_path=args.capture_receipt,
            method_freeze_path=args.method_freeze,
            state_paths=states or None,
            frequency_reference_path=args.frequency_reference,
            public_model_snapshot=args.public_model_snapshot,
            lens_path=args.lens,
            reference_path=args.reference,
            output_root=args.output_root,
        )
        contract.write_create_only(args.output, value)
    except (RegisterError, contract.ContractError) as exc:
        print(f"TRR-0007 registration failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
