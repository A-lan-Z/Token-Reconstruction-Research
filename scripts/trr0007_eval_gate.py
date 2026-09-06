"""Freeze and revalidate the complete TRR-0007 public matrix.

The freeze command is a public-only gate. It validates every decoder artifact
and bounded anchor artifact, all one-warmup/one-measured timing receipts, source
selection/exclusion bindings, and the source-free observation manifest before
writing an immutable receipt. The pre-truth validator repeats these checks and
inspects only a private truth binding header; it never stats, hashes, or opens
the sidecar. The scorer is the sole caller that may read the sidecar after
this gate succeeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from scripts import trr0007_eval_contract as contract
from scripts import trr0007_bank_ledger as bank_ledger


class GateError(contract.ContractError):
    """Raised when the complete public gate cannot be proven."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _root(value: Path) -> Path:
    result = Path(value).expanduser().resolve()
    if result.is_symlink() or not result.is_dir():
        raise GateError(f"repository root is unavailable: {result}")
    return result


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _outside(path: Path, directory: Path, *, description: str) -> None:
    if _inside(path, directory):
        raise GateError(f"{description} must be outside {directory}: {path}")


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
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
        raise GateError(f"unable to bind {description}: {path}") from exc


def _bound_file(value: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    try:
        return contract.validate_file_record(
            value, repository_root=root, description=description, verify=True
        )
    except contract.ContractError as exc:
        raise GateError(str(exc)) from exc


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GateError("cannot resolve current commit") from exc
    if not contract._COMMIT.fullmatch(value):
        raise GateError("current commit is not a full hash")
    return value


def _load_plan(
    registration: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _bound_file(registration["plan"], root=root, description="evaluation plan")
    if binding["sha256"] != registration["plan_sha256"]:
        raise GateError("registration plan binding does not match plan_sha256")
    plan = contract.load_json(Path(binding["path"]), description="evaluation plan")
    contract.validate_plan(plan)
    if plan.get("status") != "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION":
        raise GateError("evaluation plan must be frozen before the public gate")
    # Execution progress is bound by the source/capture receipts below.  The
    # design plan remains immutable, so its status flags stay false.
    return plan, binding


def _validate_public_metadata(
    registration: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    selection_binding = _bound_file(
        registration["source_selection"], root=root, description="public source selection"
    )
    exclusion_binding = _bound_file(
        registration["exclusion_manifest"], root=root, description="public exclusion manifest"
    )
    observation_binding = _bound_file(
        registration["observation_manifest"], root=root, description="public observation manifest"
    )
    capture_binding = _bound_file(
        registration["capture_receipt"], root=root, description="public capture receipt"
    )
    method_freeze_binding = _bound_file(
        registration["method_freeze"], root=root, description="method freeze"
    )
    try:
        method_record, _method_payload, method_states = contract.load_method_freeze(
            Path(method_freeze_binding["path"]),
            repository_root=root,
            verify_assets=True,
        )
    except contract.ContractError as exc:
        raise GateError(str(exc)) from exc
    if method_record != method_freeze_binding:
        raise GateError("method-freeze registration binding changed")
    expected_state_hashes = registration.get("method_freeze_state_sha256")
    if expected_state_hashes != {
        method_id: state["sha256"] for method_id, state in method_states.items()
    }:
        raise GateError("method-freeze selected-state hashes changed")
    frequency_binding = _bound_file(
        registration["frequency_reference"], root=root, description="public frequency reference"
    )
    frequency_payload = _load_json(
        Path(frequency_binding["path"]), description="public frequency reference"
    )
    frequency_maps = frequency_payload.get("frequency_references")
    if (
        frequency_payload.get("schema") != contract.FREQUENCY_REFERENCE_SCHEMA
        or frequency_payload.get("task_id") != "TRR-0005"
        or frequency_payload.get("status") != "PUBLIC_FITTING_FREQUENCY_REFERENCES"
        or not isinstance(frequency_maps, Mapping)
        or not isinstance(frequency_maps.get("enriched"), Mapping)
    ):
        raise GateError("public frequency reference is not the frozen enriched fitting map")
    selection = _load_json(
        Path(selection_binding["path"]), description="public source selection"
    )
    if (
        selection.get("schema") != contract.SOURCE_SELECTION_SCHEMA
        or selection.get("task_id") != contract.TASK_ID
        or selection.get("status") != contract.SOURCE_SELECTION_STATUS
    ):
        raise GateError("public source selection receipt is not frozen")
    if selection.get("records_per_domain") != contract.RECORDS_PER_DOMAIN:
        raise GateError("public source selection record count changed")
    if selection.get("method_freeze_sha256") != method_freeze_binding["sha256"]:
        raise GateError("public source selection method-freeze binding changed")
    if selection.get("method_freeze") != method_freeze_binding:
        raise GateError("public source selection method-freeze descriptor changed")
    if not isinstance(selection.get("selection_exclusions"), Mapping) or selection["selection_exclusions"].get("sha256") != exclusion_binding["sha256"]:
        raise GateError("public source selection exclusion binding changed")
    final_bank = selection.get("final_bank_ledgers")
    final_files = final_bank.get("files") if isinstance(final_bank, Mapping) else None
    if not isinstance(final_files, Mapping) or not all(
        isinstance(final_files.get(key), Mapping) for key in ("exclusion_manifest", "selected_parent_rows", "corpus_plan")
    ):
        raise GateError("public source selection lacks final v5 bank ledgers")
    try:
        verified_bank = bank_ledger.load_final_bank_ledgers(
            repository_root=root,
            exclusion_manifest=Path(str(final_files["exclusion_manifest"]["path"])),
            selected_parent_rows=Path(str(final_files["selected_parent_rows"]["path"])),
            corpus_plan=Path(str(final_files["corpus_plan"]["path"])),
        )
    except bank_ledger.BankLedgerError as exc:
        raise GateError(str(exc)) from exc
    if verified_bank != dict(final_bank):
        raise GateError("public source selection final v5 bank descriptor changed")
    prefix_ledger = selection.get("public_fitting_prefix_exclusions")
    prefix_file = prefix_ledger.get("file") if isinstance(prefix_ledger, Mapping) else None
    if not isinstance(prefix_file, Mapping) or not isinstance(prefix_file.get("path"), str):
        raise GateError("public source selection lacks the reviewed v3 fitting-prefix ledger")
    try:
        verified_prefix = bank_ledger.load_prefix_exclusion_ledger(
            repository_root=root, path=Path(str(prefix_file["path"]))
        )
    except bank_ledger.BankLedgerError as exc:
        raise GateError(str(exc)) from exc
    if verified_prefix != dict(prefix_ledger):
        raise GateError("public source selection v3 fitting-prefix ledger descriptor changed")
    if registration.get("public_fitting_prefix_exclusions") != dict(prefix_ledger):
        raise GateError("registration v3 fitting-prefix ledger binding changed")
    exclusion = _load_json(
        Path(exclusion_binding["path"]), description="public exclusion manifest"
    )
    if exclusion.get("task_id") != contract.TASK_ID or exclusion.get("status") != "PUBLIC_IDENTITY_EXCLUSIONS_COMPLETE_NO_TRUTH":
        raise GateError("public exclusion receipt is not closed")
    capture = _load_json(
        Path(capture_binding["path"]), description="public capture receipt"
    )
    if (
        capture.get("schema") != contract.CAPTURE_SCHEMA
        or capture.get("task_id") != contract.TASK_ID
        or capture.get("status") != contract.CAPTURE_STATUS
    ):
        raise GateError("public capture receipt is not complete")
    if capture.get("method_freeze_sha256") != method_freeze_binding["sha256"]:
        raise GateError("public capture method-freeze binding changed")
    if not isinstance(capture.get("selection_plan"), Mapping) or capture["selection_plan"].get("sha256") != selection_binding["sha256"]:
        raise GateError("public capture selection binding changed")
    if not isinstance(capture.get("observations"), Mapping) or capture["observations"].get("sha256") != observation_binding["sha256"]:
        raise GateError("public capture observation binding changed")
    if capture.get("truth_opened") is not False:
        raise GateError("public capture receipt records truth access")
    observation = _load_json(
        Path(observation_binding["path"]), description="public observation manifest"
    )
    if observation.get("method_freeze_sha256") != method_freeze_binding["sha256"]:
        raise GateError("public observation method-freeze binding changed")
    if not isinstance(observation.get("selection_plan"), Mapping) or observation["selection_plan"].get("sha256") != selection_binding["sha256"]:
        raise GateError("public observation selection binding changed")
    for field, payload in (
        ("public source selection", selection),
        ("public exclusion manifest", exclusion),
        ("public capture receipt", capture),
        ("public observation manifest", observation),
    ):
        for key in (
            "truth_opened",
            "truth_created",
            "truth_created_or_opened",
            "target_loaded",
            "target_labels_loaded",
            "source_text_loaded",
            "source_text_written",
            "private_or_truth_payload_read",
        ):
            if payload.get(key) is True:
                raise GateError(f"{field} records forbidden payload access: {key}")
    result.update(
        {
            "source_selection": selection_binding,
            "exclusion_manifest": exclusion_binding,
            "observation_manifest": observation_binding,
            "capture_receipt": capture_binding,
            "method_freeze": method_freeze_binding,
            "frequency_reference": frequency_binding,
            "final_bank_ledgers": dict(final_bank),
            "public_fitting_prefix_exclusions": dict(prefix_ledger),
        }
    )
    return result


def _resolve_output_root(registration: Mapping[str, Any], root: Path) -> Path:
    raw = Path(str(registration["output_root"])).expanduser()
    if not raw.is_absolute():
        raw = root / raw
    output = raw.resolve()
    task_root = (root / "experiments" / "TRR-0007").resolve()
    if not _inside(output, task_root):
        raise GateError(f"prediction output is outside task root: {output}")
    if output.is_symlink() or not output.is_dir():
        raise GateError(f"prediction output root is unavailable: {output}")
    return output


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        return contract.load_json(path, description=description)
    except contract.ContractError as exc:
        raise GateError(str(exc)) from exc


def _finite_nonnegative(values: Any, *, description: str, length: int) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise GateError(f"{description} must have {length} entries")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise GateError(f"{description} contains a non-number")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise GateError(f"{description} contains a non-finite/negative value")
        result.append(value)
    return result


def _validate_timing(
    path: Path,
    *,
    method_id: str,
    cell_id: str,
    records: int,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    timing = _load_json(path, description=f"timing {method_id}/{cell_id}")
    if timing.get("schema") != contract.TIMING_SCHEMA or timing.get("task_id") != contract.TASK_ID:
        raise GateError(f"timing identity changed: {method_id}/{cell_id}")
    if timing.get("method_id") != method_id or timing.get("cell_id") != cell_id:
        raise GateError(f"timing method/cell changed: {method_id}/{cell_id}")
    if timing.get("records") != records:
        raise GateError(f"timing record count changed: {method_id}/{cell_id}")
    if timing.get("warmup_runs_per_record") != 1 or timing.get("measured_runs_per_record") != 1:
        raise GateError(f"timing repeat count changed: {method_id}/{cell_id}")
    if timing.get("warmup_output_exact_match_measured") is not True or timing.get("measured_output_selected") is not True:
        raise GateError(f"timing repeat integrity is incomplete: {method_id}/{cell_id}")
    _finite_nonnegative(
        timing.get("per_record_measured_seconds"),
        description=f"timing per-record values {method_id}/{cell_id}",
        length=records,
    )
    for key in ("warmup_seconds_sum", "measured_seconds_sum", "model_preparation_seconds"):
        if key in timing:
            _finite_nonnegative([timing[key]], description=f"timing {key}", length=1)
    artifact = timing.get("prediction_artifact")
    if not isinstance(artifact, Mapping):
        raise GateError(f"timing prediction artifact is absent: {method_id}/{cell_id}")
    if artifact.get("sha256") != prediction["artifact"]["sha256"]:
        raise GateError(f"timing/prediction artifact hash differs: {method_id}/{cell_id}")
    if timing.get("truth_opened") is not False or timing.get("candidate_arrays_persisted") is not False:
        raise GateError(f"timing flags are open: {method_id}/{cell_id}")
    if method_id == contract.ANCHOR_METHOD_ID:
        a2 = timing.get("a2")
        if not isinstance(a2, Mapping):
            raise GateError("A1+A2 timing evidence is absent")
        if a2.get("proposal_budget") != contract.A2_PROPOSAL_K or a2.get("candidate_budget") != contract.A2_K:
            raise GateError("A1+A2 budget evidence changed")
        if a2.get("a2_fallback") is not False or a2.get("candidate_output") != "output_only; no candidate tensors persisted":
            raise GateError("A1+A2 output policy changed")
        if timing.get("a1_prediction_sha256") != prediction.get("a1_prediction_sha256"):
            raise GateError("A1 diagnostic hash is not bound to the anchor artifact")
    return timing


def validate_public_matrix(
    *,
    registration: Mapping[str, Any],
    repository_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    root = _root(repository_root)
    registration = contract.validate_registration(registration)
    if _git_head(root) != registration["code_commit"]:
        raise GateError("registration code_commit does not match current executable commit")
    for row in registration["code_bindings"]:
        _bound_file(row, root=root, description=f"code binding {row['role']}")
    _load_plan(registration, root=root)
    public_meta = _validate_public_metadata(registration, root=root)
    _manifest, observations, observation_record = contract.load_observation_manifest(
        registration, repository_root=root, verify_assets=True
    )
    output = _resolve_output_root(registration, root) if output_root is None else Path(output_root).expanduser().resolve()
    if output != _resolve_output_root(registration, root):
        raise GateError("supplied output root differs from registration")
    required = {"registration.json", "predictions.json", "timings.json", "run_manifest.json"}
    present = {path.name for path in output.iterdir() if path.is_file()}
    missing = sorted(required - present)
    if missing:
        raise GateError(f"prediction output is incomplete: missing={missing!r}")
    if (output / "failure.json").exists():
        raise GateError("prediction output contains a failure receipt")
    expected_artifact_paths = {
        contract.expected_prediction_path(output, cell_id=cell, method_id=method)
        for method in contract.METHOD_ORDER
        for cell in contract.expected_method_cells(method)
    }
    expected_artifact_paths |= {path.with_suffix(".run.json") for path in expected_artifact_paths}
    unexpected_tensor_files = [
        path for path in output.rglob("*.safetensors")
        if path not in expected_artifact_paths
    ]
    if unexpected_tensor_files:
        raise GateError(f"prediction output contains unregistered tensor artifacts: {unexpected_tensor_files!r}")
    registration_copy = output / "registration.json"
    if contract.sha256_file(registration_copy) != contract.sha256_file(Path(registration["_path"])):
        raise GateError("embedded registration differs from supplied registration")
    predictions_doc = _load_json(output / "predictions.json", description="prediction descriptor")
    timings_doc = _load_json(output / "timings.json", description="timing descriptor")
    run_doc = _load_json(output / "run_manifest.json", description="prediction run manifest")
    for doc, schema, label in (
        (predictions_doc, contract.PREDICTION_SCHEMA, "prediction descriptor"),
        (timings_doc, contract.TIMING_SCHEMA, "timing descriptor"),
    ):
        if doc.get("schema") != schema or doc.get("task_id") != contract.TASK_ID:
            raise GateError(f"{label} identity changed")
        if doc.get("registration_sha256") != registration["registration_sha256"]:
            raise GateError(f"{label} registration binding changed")
        if doc.get("truth_opened") is not False:
            raise GateError(f"{label} truth flag is open")
    if run_doc.get("schema") != contract.RUN_SCHEMA or run_doc.get("task_id") != contract.TASK_ID:
        raise GateError("run manifest identity changed")
    if run_doc.get("status") != "COMPLETE_PUBLIC_PREDICTIONS_NO_TRUTH" or run_doc.get("truth_opened") is not False:
        raise GateError("run manifest is incomplete or open")
    expected_entries = sum(len(contract.expected_method_cells(method)) for method in contract.METHOD_ORDER)
    if predictions_doc.get("entries") is None or timings_doc.get("entries") is None:
        raise GateError("prediction/timing descriptor entries are absent")
    expected_keys = {
        f"{method}::{cell}"
        for method in contract.METHOD_ORDER
        for cell in contract.expected_method_cells(method)
    }
    if set(predictions_doc["entries"]) != expected_keys or set(timings_doc["entries"]) != expected_keys:
        raise GateError("prediction/timing descriptor matrix is incomplete")
    validated: list[dict[str, Any]] = []
    validated_timings: list[dict[str, Any]] = []
    for method in contract.METHOD_ORDER:
        records = contract.ANCHOR_RECORDS_PER_DOMAIN if method == contract.ANCHOR_METHOD_ID else contract.RECORDS_PER_DOMAIN
        for cell_id in contract.expected_method_cells(method):
            cell = observations["cells"][cell_id]
            prediction_path = contract.expected_prediction_path(output, cell_id=cell_id, method_id=method)
            try:
                prediction = contract.validate_prediction_artifact(
                    prediction_path,
                    registration=registration,
                    cell=cell,
                    method_id=method,
                    records=records,
                )
            except contract.ContractError as exc:
                raise GateError(str(exc)) from exc
            timing_path = contract.expected_timing_path(output, cell_id=cell_id, method_id=method)
            timing = _validate_timing(
                timing_path,
                method_id=method,
                cell_id=cell_id,
                records=records,
                prediction=prediction,
            )
            key = f"{method}::{cell_id}"
            if predictions_doc["entries"][key].get("prediction_sha256") != prediction["prediction_sha256"]:
                raise GateError(f"prediction descriptor hash changed: {key}")
            if method == contract.ANCHOR_METHOD_ID and predictions_doc["entries"][key].get("a1_prediction_sha256") != prediction["a1_prediction_sha256"]:
                raise GateError(f"A1 descriptor hash changed: {key}")
            if timings_doc["entries"][key].get("prediction_artifact", {}).get("sha256") != prediction["artifact"]["sha256"]:
                raise GateError(f"timing descriptor hash changed: {key}")
            validated.append(prediction)
            validated_timings.append({
                "method_id": method,
                "cell_id": cell_id,
                "path": str(timing_path),
                "sha256": contract.sha256_file(timing_path),
            })
    if len(validated) != expected_entries or len(validated_timings) != expected_entries:
        raise GateError("validated artifact count is incomplete")
    return {
        "registration": _record(Path(registration["_path"]), root=root, description="registration"),
        "plan": _bound_file(registration["plan"], root=root, description="evaluation plan"),
        "observation_manifest": observation_record,
        "public_metadata": public_meta,
        "output_root": str(output),
        "predictions": validated,
        "timings": validated_timings,
        "expected_artifact_count": expected_entries,
        "truth_opened": False,
    }


def freeze_public_matrix(
    *,
    registration_path: Path,
    repository_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    root = _root(repository_root)
    registration_path = Path(registration_path).expanduser().resolve()
    registration = contract.load_registration(registration_path)
    output = _resolve_output_root(registration, root)
    receipt_path = Path(receipt_path).expanduser().resolve()
    _outside(receipt_path, output, description="freeze receipt")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise GateError(f"freeze receipt is create-only: {receipt_path}")
    evidence = validate_public_matrix(registration=registration, repository_root=root, output_root=output)
    receipt = {
        "schema": contract.FREEZE_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_COMPLETE_PUBLIC_MATRIX_NO_TRUTH",
        "created_utc": _utc_now(),
        "registration": evidence["registration"],
        "registration_sha256": registration["registration_sha256"],
        "plan": evidence["plan"],
        "source_selection": evidence["public_metadata"]["source_selection"],
        "exclusion_manifest": evidence["public_metadata"]["exclusion_manifest"],
        "capture_receipt": evidence["public_metadata"]["capture_receipt"],
        "observation_manifest": evidence["observation_manifest"],
        "method_freeze": evidence["public_metadata"]["method_freeze"],
        "frequency_reference": evidence["public_metadata"]["frequency_reference"],
        "final_bank_ledgers": evidence["public_metadata"]["final_bank_ledgers"],
        "output_root": evidence["output_root"],
        "prediction_artifacts": evidence["predictions"],
        "timing_artifacts": evidence["timings"],
        "expected_artifact_count": evidence["expected_artifact_count"],
        "truth_opened": False,
        "private_truth_payload_persisted": False,
    }
    contract.write_create_only(receipt_path, receipt)
    return receipt


def _validate_truth_header(
    path: Path,
    *,
    repository_root: Path,
    output_root: Path,
    observations: Mapping[str, Any],
) -> dict[str, Any]:
    # Read only the binding JSON. The sidecar is neither opened nor stat'ed
    # until the scorer calls its post-gate loader.
    header = contract.load_json(path, description="private truth binding")
    if header.get("schema") != "token-reconstruction.trr0007-truth-binding.v1" or header.get("task_id") != contract.TASK_ID:
        raise GateError("private truth binding identity changed")
    if header.get("truth_opened") is not False:
        raise GateError("private truth binding is already open")
    sidecar = header.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise GateError("private truth sidecar descriptor is absent")
    sidecar_path = sidecar.get("path")
    if not isinstance(sidecar_path, str) or not Path(sidecar_path).expanduser().is_absolute():
        raise GateError("private truth sidecar path must be absolute")
    sidecar_path = Path(sidecar_path).expanduser().resolve(strict=False)
    _outside(sidecar_path, output_root, description="private truth sidecar")
    _outside(sidecar_path, repository_root, description="private truth sidecar")
    if isinstance(sidecar.get("bytes"), bool) or not isinstance(sidecar.get("bytes"), int) or sidecar["bytes"] <= 0:
        raise GateError("private truth sidecar byte count is invalid")
    if not isinstance(sidecar.get("sha256"), str) or contract._SHA256.fullmatch(sidecar["sha256"]) is None:
        raise GateError("private truth sidecar digest is invalid")
    cells = header.get("cells")
    if not isinstance(cells, list) or [row.get("cell_id") for row in cells if isinstance(row, Mapping)] != list(contract.CELL_ORDER):
        raise GateError("private truth binding cell order changed")
    for row in cells:
        if not isinstance(row, Mapping) or row.get("records") != contract.RECORDS_PER_DOMAIN:
            raise GateError("private truth binding geometry changed")
        cell_id = row.get("cell_id")
        if cell_id not in observations["cells"]:
            raise GateError("private truth binding references an unknown cell")
        if row.get("record_ids_sha256") != observations["cells"][cell_id]["record_ids_sha256"]:
            raise GateError(f"private truth record order changed: {cell_id}")
    return {
        "schema": header["schema"],
        "task_id": header["task_id"],
        "truth_opened": False,
        "sidecar": dict(sidecar),
        "cells": [dict(row) for row in cells],
    }


def validate_before_truth(
    *,
    receipt_path: Path,
    registration_path: Path,
    repository_root: Path,
    truth_binding_path: Path | None = None,
) -> dict[str, Any]:
    root = _root(repository_root)
    registration_path = Path(registration_path).expanduser().resolve()
    registration = contract.load_registration(registration_path)
    receipt_path = Path(receipt_path).expanduser().resolve()
    receipt = _load_json(receipt_path, description="TRR-0007 freeze receipt")
    if receipt.get("schema") != contract.FREEZE_SCHEMA or receipt.get("task_id") != contract.TASK_ID:
        raise GateError("freeze receipt identity changed")
    if receipt.get("status") != "FROZEN_COMPLETE_PUBLIC_MATRIX_NO_TRUTH" or receipt.get("truth_opened") is not False:
        raise GateError("freeze receipt is not closed")
    if receipt.get("registration_sha256") != registration["registration_sha256"]:
        raise GateError("freeze receipt registration changed")
    output = _resolve_output_root(registration, root)
    evidence = validate_public_matrix(registration=registration, repository_root=root, output_root=output)
    if receipt.get("expected_artifact_count") != evidence["expected_artifact_count"]:
        raise GateError("freeze receipt artifact count changed")
    for key in (
        "plan",
        "source_selection",
        "exclusion_manifest",
        "capture_receipt",
        "observation_manifest",
        "method_freeze",
        "frequency_reference",
        "final_bank_ledgers",
    ):
        expected = evidence.get(key)
        if key in ("source_selection", "exclusion_manifest", "capture_receipt", "method_freeze", "frequency_reference"):
            expected = evidence["public_metadata"][key]
            if not isinstance(receipt.get(key), Mapping) or receipt[key].get("sha256") != expected.get("sha256"):
                raise GateError(f"freeze receipt binding changed: {key}")
        elif key == "final_bank_ledgers":
            expected = evidence["public_metadata"][key]
            if receipt.get(key) != expected:
                raise GateError(f"freeze receipt binding changed: {key}")
        elif not isinstance(receipt.get(key), Mapping) or receipt[key].get("sha256") != expected.get("sha256"):
            raise GateError(f"freeze receipt binding changed: {key}")
    if truth_binding_path is not None:
        truth_header = _validate_truth_header(
            Path(truth_binding_path).expanduser().resolve(),
            repository_root=root,
            output_root=output,
            observations=contract.load_observation_manifest(
                registration, repository_root=root, verify_assets=True
            )[1],
        )
    else:
        truth_header = None
    return {
        "schema": contract.FREEZE_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "PUBLIC_MATRIX_VERIFIED_NO_TRUTH_OPENED",
        "receipt": {"path": str(receipt_path), "bytes": int(receipt_path.stat().st_size), "sha256": contract.sha256_file(receipt_path)},
        "registration": registration,
        "truth_binding_header": truth_header,
        "truth_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--registration", type=Path, required=True)
    freeze.add_argument("--repository-root", type=Path, default=Path("."))
    freeze.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_public_matrix(
                registration_path=args.registration,
                repository_root=args.repository_root,
                receipt_path=args.receipt,
            )
        else:
            raise GateError(f"unknown command {args.command}")
    except (GateError, contract.ContractError) as exc:
        print(f"TRR-0007 public gate failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
