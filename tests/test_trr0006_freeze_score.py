"""Integration tests for the executable TRR-0006 public/truth boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts import trr0006_freeze_score as driver

ROOT = Path(__file__).resolve().parents[1]


def _load_fixture_helpers():
    path = ROOT / "tests" / "test_trr0006_pair_contract.py"
    spec = importlib.util.spec_from_file_location("trr0006_pair_contract_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    display = str(path.resolve())
    if root is not None:
        display = path.resolve().relative_to(root.resolve()).as_posix()
    return {"path": display, "bytes": path.stat().st_size, "sha256": _sha(path)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _prepare_main_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Adapt the real runner-schema fixture with task-6 bindings and labels."""

    helpers = _load_fixture_helpers()
    fixture = helpers._actual_runner_fixture(tmp_path, monkeypatch, records=8)
    root = fixture["root"]
    task_root = root / "experiments" / "TRR-0006"
    output = fixture["output"]
    # The driver uses package imports while the existing fixture loads the
    # same files under short names.  Point it at the fixture's real gate and
    # scorer so all eight runner descriptors are exercised unchanged.
    monkeypatch.setattr(driver, "freeze", helpers.freeze)
    monkeypatch.setattr(driver, "scorer", helpers.score)
    monkeypatch.setattr(driver, "_git_head", lambda _root: "b" * 40)

    ids = {domain: [f"{domain}-source-{i}" for i in range(8)] for domain in ("pile", "finance")}
    observation_path = fixture["observation_manifest"]
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    for cell in observation["cells"]:
        cell["record_ids_sha256"] = _json_digest(ids[cell["style"]])
    _write_json(observation_path, observation)
    observation_record = _record(observation_path, root=root)

    selection_path = task_root / "source_selection.json"
    selection = {
        "schema": driver.SOURCE_SELECTION_SCHEMA,
        "task_id": driver.TASK_ID,
        "status": driver.SOURCE_SELECTION_STATUS,
        "records_per_domain": 8,
        "paired_conditions": True,
        "selection_rule": {
            "source_text_or_token_ids_written": False,
            "record_ids_sha256": {domain: _json_digest(values) for domain, values in ids.items()},
            "records": {domain: [{"record_id": value} for value in values] for domain, values in ids.items()},
        },
    }
    _write_json(selection_path, selection)
    selection_record = _record(selection_path, root=root)

    plan_path = task_root / "decision_plan.json"
    plan = {
        "schema": "token-reconstruction.trr0006-decision-plan.v1",
        "task_id": driver.TASK_ID,
        "status": "FROZEN_MAIN_TEST_PLAN",
        "sample_size_frozen": True,
        "panel": {
            "records_per_domain": 8,
            "unique_sources_total": 16,
            "record_condition_evaluations_per_method": 32,
            "clip_tokens_including_bos": 128,
            "scored_post_bos_tokens": 127,
        },
        "comparison": {
            "cells": list(driver.freeze.CELL_ORDER),
            "method_order": list(driver.contract.METHOD_IDS),
        },
        "truth_opened": False,
    }
    _write_json(plan_path, plan)
    plan_record = _record(plan_path, root=root)

    registration_path = fixture["registration"]
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    registration["observation_manifest"] = observation_record
    registration["source_selection"] = selection_record
    registration["source_record_ids_sha256"] = {domain: _json_digest(values) for domain, values in ids.items()}
    registration["decision_plan"] = plan_record
    registration["decision_plan_sha256"] = plan_record["sha256"]
    execution_files = []
    for role, relative in driver.EXECUTION_BINDING_SPECS:
        code_path = root / relative
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(f"# test executable: {role}\n", encoding="utf-8")
        row = _record(code_path, root=root)
        row["role"] = role
        execution_files.append(row)
    registration["execution_binding"] = {
        "schema": driver.EXECUTION_BINDING_SCHEMA,
        "code_commit": "b" * 40,
        "files": execution_files,
    }
    # Rebind prediction metadata to the newly written public manifest and
    # registration.  The prediction IDs and timing rows remain unchanged.
    registration_path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registration_record = _record(registration_path)
    predictions_path = output / "predictions.json"
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    timings_path = output / "timings.json"
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    for key, row in predictions["predictions"].items():
        row["registration_sha256"] = registration_record["sha256"]
        artifact = Path(row["prediction_artifact"]["path"])
        with safe_open(str(artifact), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor("predictions")
            metadata = dict(handle.metadata() or {})
        metadata["registration_sha256"] = registration_record["sha256"]
        metadata["observation_manifest_sha256"] = observation_record["sha256"]
        save_file({"predictions": tensor}, str(artifact), metadata=metadata)
        row["prediction_artifact"] = _record(artifact)
        row["prediction_sha256"] = driver.contract.tensor_digest(tensor)
        timings["timings"][key]["prediction_artifact"] = _record(artifact)
        timings["timings"][key]["prediction_sha256"] = row["prediction_sha256"]
    predictions["registration_sha256"] = registration_record["sha256"]
    timings["registration_sha256"] = registration_record["sha256"]
    _write_json(predictions_path, predictions)
    _write_json(timings_path, timings)
    run_manifest_path = output / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["registration"] = registration_record
    run_manifest["observation_manifest"] = observation_record
    _write_json(run_manifest_path, run_manifest)

    # The receipt is deliberately created only after every public descriptor
    # has the final registration/observation hashes.
    receipt = task_root / "freeze_receipt.json"
    helpers.freeze.freeze_matrix(
        repository_root=root,
        registration_path=registration_path,
        receipt_path=receipt,
        observation_manifest_path=observation_path,
        plan_path=plan_path,
    )

    truth_path = tmp_path.parent / f"truth-{tmp_path.name}.safetensors"
    labels = torch.ones((8, 128), dtype=torch.int64)
    labels[:, 0] = driver.contract.BOS_TOKEN_ID
    save_file(
        {"pile__token_ids": labels, "finance__token_ids": labels.clone()},
        str(truth_path),
        metadata={
            "schema": driver.TRUTH_SIDECAR_SCHEMA,
            "task_id": driver.TASK_ID,
            "truth_opened": "false",
            "decision_plan_sha256": str(plan_record["sha256"]),
            "source_selection_sha256": str(selection_record["sha256"]),
            "observation_sha256": json.dumps(
                {cell["cell_id"]: cell["observation"]["sha256"] for cell in observation["cells"]},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "record_ids_sha256": json.dumps(
                {domain: _json_digest(values) for domain, values in ids.items()},
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    )
    binding_path = tmp_path.parent / f"truth-{tmp_path.name}.binding.json"
    binding = {
        "schema": driver.TRUTH_BINDING_SCHEMA,
        "task_id": driver.TASK_ID,
        "status": driver.TRUTH_READY_STATUS,
        "truth_file": _record(truth_path),
        "decision_plan": plan_record,
        "source_selection": selection_record,
        "observation_sha256": {cell["cell_id"]: cell["observation"]["sha256"] for cell in observation["cells"]},
        "record_ids_sha256": {domain: _json_digest(values) for domain, values in ids.items()},
        "truth_tensor_keys": list(driver.EXPECTED_TRUTH_KEYS),
        "truth_opened": False,
        "reconstruction_root_contains_truth": False,
    }
    _write_json(binding_path, binding)
    return {
        **fixture,
        "helpers": helpers,
        "plan": plan_path,
        "selection": selection_path,
        "registration_record": registration_record,
        "binding": binding_path,
        "truth": truth_path,
        "ids": ids,
    }


def _run_fixture(driver_fixture, *, result_name: str = "result.json"):
    root = driver_fixture["root"]
    task_root = root / "experiments" / "TRR-0006"
    return driver.run(
        repository_root=root,
        plan_path=driver_fixture["plan"],
        registration_path=driver_fixture["registration"],
        source_selection_path=driver_fixture["selection"],
        observation_manifest_path=driver_fixture["observation_manifest"],
        freeze_receipt_path=driver_fixture["receipt"],
        truth_binding_path=driver_fixture["binding"],
        truth_path=driver_fixture["truth"],
        result_path=task_root / result_name,
        report_path=task_root / "report.md",
        manifest_path=task_root / "manifest.json",
        execution_receipt_path=task_root / "execution_receipt.json",
    )


def test_executable_driver_opens_private_truth_once_after_real_public_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _prepare_main_fixture(tmp_path, monkeypatch)
    opened: list[Path] = []
    original_safe_open = driver.safe_open

    def spy_safe_open(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return original_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(driver, "safe_open", spy_safe_open)
    result = _run_fixture(fixture)
    assert result["status"] == "TRR6_EXECUTED_SCORED_AFTER_PUBLIC_FREEZE"
    assert opened.count(fixture["truth"].resolve()) == 1
    assert opened[-1] == fixture["truth"].resolve()
    receipt = json.loads((fixture["root"] / "experiments/TRR-0006/execution_receipt.json").read_text())
    assert receipt["truth_binding_recorded_before_gate"]["truth_payload_read_before_gate"] is False
    assert receipt["truth_verified_after_public_gate"]["truth_file"]["sha256"] == _sha(fixture["truth"])
    assert receipt["truth_opened_once"] is True
    assert (fixture["root"] / "experiments/TRR-0006/result.json").exists()
    assert (fixture["root"] / "experiments/TRR-0006/manifest.json").exists()


def test_invalid_public_receipt_fails_before_truth_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _prepare_main_fixture(tmp_path, monkeypatch)
    receipt_path = fixture["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "BROKEN_PUBLIC_RECEIPT"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    opened: list[Path] = []
    original_safe_open = driver.safe_open

    def spy_safe_open(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return original_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(driver, "safe_open", spy_safe_open)
    with pytest.raises(driver.freeze.FreezePairError):
        _run_fixture(fixture)
    assert fixture["truth"].resolve() not in opened
    assert not (fixture["root"] / "experiments/TRR-0006/result.json").exists()
