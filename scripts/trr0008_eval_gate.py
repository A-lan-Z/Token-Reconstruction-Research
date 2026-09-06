"""Fail-closed public prediction and pre-truth gates for TRR-0008.

The first gate checks the complete four-method by four-cell public matrix and
its timing receipts, including the hashes recorded by the runner.  The
metadata-only pre-truth gate calls that public gate again and validates only a
truth binding header.  It never opens, hashes, or stats the private sidecar;
the scorer owns the first sidecar read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from scripts import trr0008_eval_contract as contract


class GateError(contract.ContractError):
    pass


def _record(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise GateError(f"{description} is unavailable: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": contract.sha256_file(path),
    }


def _same_record(actual: Mapping[str, Any], expected: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise GateError(f"{description} binding changed: {key}")


def _load_timing(path: Path) -> dict[str, Any]:
    """Load only the owner-bound final precision40 timing receipt."""

    value = contract.load_json(path, description="TRR-0008 timing receipt")
    if value.get("task_id") != contract.TASK_ID:
        raise GateError("timing receipt task identity changed")
    if (
        value.get("truth_opened") is True
        or value.get("source_text_or_target_labels") is True
        or value.get("candidate_arrays_persisted") is True
    ):
        raise GateError("timing receipt records forbidden truth/source access")
    if value.get("schema") != "token-reconstruction.trr0008-balanced-timing.v1":
        raise GateError("only the canonical balanced timing receipt is accepted")
    if value.get("status") != "TIMING_COMPLETE":
        raise GateError("canonical timing receipt is incomplete")
    configuration = value.get("configuration")
    if not isinstance(configuration, Mapping) or int(configuration.get("blocks", -1)) != 40:
        raise GateError("timing receipt is not the frozen 40-block qualification")
    if value.get("equivalence", {}).get("status") != "PASS":
        raise GateError("canonical timing output equivalence did not pass")
    summary = value.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("qualification"), Mapping):
        raise GateError("canonical timing qualification is absent")
    return value


def _registration_and_observation(
    *,
    registration_path: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).expanduser().resolve()
    registration = contract.load_registration(
        registration_path, repository_root=root, verify_assets=True
    )
    registration_record = _record(registration_path, description="TRR8 registration")
    registration["registration_sha256"] = registration_record["sha256"]
    observation_binding = registration["observation_manifest"]
    observation_record = contract.validate_file_record(
        observation_binding,
        repository_root=root,
        description="TRR8 observation manifest",
        verify=True,
    )
    if observation_record != dict(observation_binding):
        raise GateError("registration observation binding changed")
    observation = contract.validate_observation_manifest(
        contract.load_json(Path(observation_record["path"]), description="public observations"),
        repository_root=root,
        verify_assets=True,
    )
    embedding = registration["runtime_assets"]["normalized_public_E"]
    checked_embedding = contract.validate_file_record(
        embedding,
        repository_root=root,
        description="normalized public E",
        verify=True,
    )
    if checked_embedding != dict(embedding):
        raise GateError("registration normalized E binding changed")
    return registration, registration_record, observation


def validate_public_outputs(
    *,
    registration_path: Path,
    repository_root: Path,
    run_manifest_path: Path,
    timing_receipt_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    registration, registration_record, observation = _registration_and_observation(
        registration_path=registration_path,
        repository_root=root,
    )
    run_record = _record(run_manifest_path, description="TRR8 prediction run manifest")
    run = contract.load_json(run_manifest_path, description="TRR8 prediction run")
    if run.get("schema") != contract.RUN_SCHEMA or run.get("task_id") != contract.TASK_ID:
        raise GateError("run manifest identity changed")
    if run.get("truth_opened") is True or run.get("candidate_arrays_persisted") is True:
        raise GateError("run manifest records truth/candidate arrays")
    run_registration = run.get("registration")
    if not isinstance(run_registration, Mapping):
        raise GateError("run manifest registration binding is absent")
    _same_record(run_registration, registration_record, description="run registration")
    run_observation = run.get("observation_manifest")
    if not isinstance(run_observation, Mapping):
        raise GateError("run manifest observation binding is absent")
    _same_record(run_observation, registration["observation_manifest"], description="run observation")
    if run.get("code_commit") != registration.get("code_commit"):
        raise GateError("run manifest code commit differs from registration")

    output_root = Path(str(registration["output_root"])).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    prediction_rows = run.get("predictions")
    timing_rows = run.get("timings")
    expected_keys = {
        f"{method_id}::{cell_id}"
        for method_id in contract.METHOD_ORDER
        for cell_id in contract.CELL_ORDER
    }
    if not isinstance(prediction_rows, Mapping) or set(prediction_rows) != expected_keys:
        raise GateError("run prediction matrix is incomplete")
    if not isinstance(timing_rows, Mapping) or set(timing_rows) != expected_keys:
        raise GateError("run timing matrix is incomplete")

    artifacts: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            key = f"{method_id}::{cell_id}"
            records = contract.records_for_cell(observation, cell_id)
            path = contract.expected_prediction_path(
                output_root, cell_id=cell_id, method_id=method_id
            )
            artifact = contract.validate_prediction_artifact(
                path,
                registration=registration,
                cell_id=cell_id,
                method_id=method_id,
                records=records,
                verify_hash=True,
            )
            declared_artifact = prediction_rows[key].get("prediction_artifact") if isinstance(prediction_rows[key], Mapping) else None
            if not isinstance(declared_artifact, Mapping):
                declared_artifact = prediction_rows[key]
            _same_record(artifact["artifact"], declared_artifact, description=f"prediction {key}")
            if str(prediction_rows[key].get("prediction_sha256")) != artifact["prediction_sha256"]:
                raise GateError(f"prediction tensor digest changed: {key}")
            if int(prediction_rows[key].get("records", -1)) != records:
                raise GateError(f"prediction denominator changed: {key}")
            timing_path = contract.expected_timing_path(
                output_root, cell_id=cell_id, method_id=method_id
            )
            timing = contract.load_json(timing_path, description=f"timing {key}")
            if timing.get("schema") != contract.TIMING_SCHEMA or timing.get("task_id") != contract.TASK_ID:
                raise GateError(f"per-cell timing identity changed: {key}")
            if timing.get("method_id") != method_id or timing.get("cell_id") != cell_id:
                raise GateError(f"per-cell timing method/cell changed: {key}")
            if timing.get("records") != records or timing.get("truth_opened") is True:
                raise GateError(f"per-cell timing denominator/truth flag changed: {key}")
            if timing.get("warmup_output_exact_match_measured") is not True:
                raise GateError(f"per-cell warmup equivalence is absent: {key}")
            timing_record = _record(timing_path, description=f"timing {key}")
            declared_timing = timing_rows[key]
            if not isinstance(declared_timing, Mapping):
                raise GateError(f"run timing binding is malformed: {key}")
            for field in ("schema", "task_id", "method_id", "cell_id", "records", "warmup_output_exact_match_measured"):
                if declared_timing.get(field) != timing.get(field):
                    raise GateError(f"run timing metadata changed: {key}/{field}")
            prediction_binding = timing.get("prediction_artifact")
            if not isinstance(prediction_binding, Mapping):
                raise GateError(f"timing prediction binding is absent: {key}")
            _same_record(artifact["artifact"], prediction_binding, description=f"timing prediction {key}")
            artifacts[key] = artifact
            timings[key] = {
                "path": str(timing_path),
                "bytes": timing_record["bytes"],
                "sha256": timing_record["sha256"],
                "measured_seconds_sum": timing.get("measured_seconds_sum"),
            }

    timing_receipt = None
    timing_binding = None
    if timing_receipt_path is not None:
        timing_receipt = _load_timing(timing_receipt_path)
        timing_binding = _record(timing_receipt_path, description="canonical timing receipt")
        registration_timing = registration.get("timing_receipt")
        if isinstance(registration_timing, Mapping):
            _same_record(timing_binding, registration_timing, description="registration timing receipt")
    return {
        "schema": "token-reconstruction.trr0008-public-freeze-receipt.v1",
        "task_id": contract.TASK_ID,
        "status": "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH",
        "registration": registration_record,
        "run_manifest": run_record,
        "observation_manifest": registration["observation_manifest"],
        "predictions": artifacts,
        "timings": timings,
        "timing_receipt": timing_binding,
        "timing_receipt_payload": timing_receipt,
        "truth_opened": False,
        "candidate_arrays_persisted": False,
        "records_by_domain": registration["records_by_domain"],
    }


def _outside_truth_path(path_value: Any, *, root: Path, output_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise GateError("truth sidecar path is absent")
    path = Path(path_value).expanduser().resolve(strict=False)
    for directory, label in ((root, "repository"), (output_root, "prediction output")):
        try:
            path.relative_to(directory.resolve())
        except ValueError:
            continue
        raise GateError(f"truth sidecar is inside {label}: {path}")
    if path.is_symlink():
        raise GateError("truth sidecar path is a symlink")
    return path


def validate_before_truth(
    *,
    receipt_path: Path,
    registration_path: Path,
    repository_root: Path,
    truth_binding_path: Path,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    receipt_record = _record(receipt_path, description="public freeze receipt")
    receipt = contract.load_json(receipt_path, description="public freeze receipt")
    if receipt.get("schema") != "token-reconstruction.trr0008-public-freeze-receipt.v1":
        raise GateError("public freeze receipt schema changed")
    if receipt.get("status") != "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH" or receipt.get("truth_opened") is not False:
        raise GateError("public freeze receipt is not closed before truth")
    registration_record = _record(registration_path, description="TRR8 registration")
    declared_registration = receipt.get("registration")
    if not isinstance(declared_registration, Mapping):
        raise GateError("public freeze registration binding is absent")
    _same_record(registration_record, declared_registration, description="freeze registration")
    run_binding = receipt.get("run_manifest")
    if not isinstance(run_binding, Mapping):
        raise GateError("public freeze run binding is absent")
    timing_binding = receipt.get("timing_receipt")
    timing_path = Path(str(timing_binding["path"])) if isinstance(timing_binding, Mapping) else None
    public = validate_public_outputs(
        registration_path=registration_path,
        repository_root=root,
        run_manifest_path=Path(str(run_binding["path"])),
        timing_receipt_path=timing_path,
    )
    _same_record(_record(receipt_path, description="public freeze receipt"), receipt_record, description="freeze receipt")
    if public["run_manifest"] != dict(run_binding):
        raise GateError("freeze receipt run binding differs from the revalidated run")
    for key in ("predictions", "timings", "observation_manifest", "timing_receipt"):
        if public.get(key) != receipt.get(key):
            raise GateError(f"freeze receipt changed after public revalidation: {key}")
    header = contract.load_json(truth_binding_path, description="truth binding header")
    if header.get("schema") != "token-reconstruction.trr0008-truth-binding.v1" or header.get("task_id") != contract.TASK_ID:
        raise GateError("truth binding identity changed")
    if header.get("truth_opened") is not False or header.get("prepared_after_public_gate") is not True:
        raise GateError("truth binding is not closed before truth")
    if header.get("registration") != dict(declared_registration):
        raise GateError("truth binding registration changed")
    if header.get("receipt") != receipt_record:
        raise GateError("truth binding receipt changed")
    observation = header.get("observation_manifest")
    registration, _registration_record, observation_doc = _registration_and_observation(
        registration_path=registration_path,
        repository_root=root,
    )
    if observation != registration.get("observation_manifest"):
        raise GateError("truth binding observation changed")
    counts = header.get("records_by_domain")
    if counts != registration.get("records_by_domain"):
        raise GateError("truth binding record counts changed")
    output_root = Path(str(registration["output_root"])).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    cells = header.get("cells")
    if not isinstance(cells, list) or {
        str(row.get("cell_id")) for row in cells if isinstance(row, Mapping)
    } != set(contract.CELL_ORDER):
        raise GateError("truth binding cell matrix changed")
    for row in cells:
        if not isinstance(row, Mapping):
            raise GateError("truth binding cell is malformed")
        expected = int(registration["records_by_domain"][str(row["cell_id"]).split("__", 1)[0]])
        if int(row.get("records", -1)) != expected:
            raise GateError("truth binding cell denominator changed")
        cell_id = str(row["cell_id"])
        actual_digest = contract._as_cells(observation_doc)[cell_id].get("record_ids_sha256")
        if row.get("record_ids_sha256") != actual_digest:
            raise GateError("truth binding record digest differs from observations")
    _outside_truth_path(
        header.get("sidecar", {}).get("path") if isinstance(header.get("sidecar"), Mapping) else None,
        root=root,
        output_root=output_root.resolve(),
    )
    return {
        "status": "PUBLIC_MATRIX_VERIFIED_NO_TRUTH_OPENED",
        "task_id": contract.TASK_ID,
        "truth_opened": False,
        "public_freeze": {
            "path": str(Path(receipt_path).resolve()),
            "sha256": receipt_record["sha256"],
        },
        "truth_binding": {
            "path": str(Path(truth_binding_path).resolve()),
            "sha256": contract.sha256_file(truth_binding_path),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--timing-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = validate_public_outputs(
            registration_path=args.registration,
            repository_root=args.repository_root,
            run_manifest_path=args.run_manifest,
            timing_receipt_path=args.timing_receipt,
        )
        contract.write_create_only(args.output, value)
    except (GateError, contract.ContractError, OSError) as exc:
        print(f"TRR-0008 gate failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": value["status"], "prediction_count": len(value["predictions"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
