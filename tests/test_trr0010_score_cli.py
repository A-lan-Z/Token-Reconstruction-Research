"""Synthetic fail-closed tests for the TRR-0010 score CLI wrapper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_register as register
from scripts import trr0010_score_cli as score_cli
from tests.test_trr0010_score import _freeze_fixture, _patch_frequency_binding
from tests.test_trr0010_eval_gate import _fixture


@pytest.fixture(autouse=True)
def _small_synthetic_observation_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "OBSERVATION_HIDDEN_SIZE", 2)


def _write_descriptor(
    *,
    fixture: dict[str, Any],
    freeze: dict[str, Any],
    descriptor_path: Path,
    payload_path: Path,
    mutate: dict[str, Any] | None = None,
) -> Path:
    values = torch.zeros((gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS), dtype=torch.long)
    values[:, 0] = gate.BOS_TOKEN_ID
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({f"{cell}__token_ids": values.clone() for cell in gate.CELL_ORDER}, str(payload_path))
    payload = gate.file_record(payload_path, root=fixture["root"])
    descriptor = {
        "schema": register.TRUTH_BINDING_SCHEMA,
        "task_id": gate.TASK_ID,
        "status": register.TRUTH_BINDING_STATUS,
        "truth_opened": False,
        "prepared_after_public_freeze": True,
        "registration": freeze["registration"],
        "run_manifest": freeze["run_manifest"],
        "contract_binding": freeze["contract_binding"],
        "input_bindings": freeze["input_bindings"],
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "cell_order": list(gate.CELL_ORDER),
        "target_conditions": list(gate.TARGET_ORDER),
        "labels_shared_across_target_conditions": True,
        "truth_shape": [gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS],
        "truth_tensor_keys": [f"{cell}__token_ids" for cell in gate.CELL_ORDER],
        "cells": [
            {
                "cell_id": cell,
                "records": gate.RECORDS_PER_CELL,
                "record_ids_sha256": freeze["observation_bindings"][cell]["record_ids_sha256"],
            }
            for cell in gate.CELL_ORDER
        ],
        "truth_payload": payload,
    }
    if mutate:
        for key, value in mutate.items():
            descriptor[key] = value
    descriptor_path.write_text(json.dumps(descriptor, sort_keys=True) + "\n", encoding="utf-8")
    return descriptor_path


def _frequency_paths(fixture: dict[str, Any]) -> dict[str, Path]:
    return {
        "B0": fixture["inputs"]["frequency_reference_B0"],
        "B1": fixture["inputs"]["frequency_reference_B1"],
    }


def _prepared_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    fixture = _fixture(tmp_path)
    # The small public-gate fixture predates the TRR-0010 truth-header
    # validator and therefore starts with only generic source/observation
    # inputs. Add truth-free panel/capture records through the same
    # registration path used in production so the test exercises the real
    # six-input descriptor contract rather than bypassing it.
    registration = json.loads(fixture["registration"].read_text(encoding="utf-8"))
    run = json.loads(fixture["run"].read_text(encoding="utf-8"))
    for name in ("panel", "capture"):
        path = fixture["root"] / "assets" / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": f"synthetic.{name}.v1",
                    "task_id": gate.TASK_ID,
                    "truth_opened": False,
                    "source_text_written": False,
                    "source_text_loaded": False,
                    "token_ids_written": False,
                    "target_labels_loaded": False,
                    "candidate_arrays_persisted": False,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record = gate.file_record(path, root=fixture["root"])
        registration["input_bindings"][name] = record
        run["input_bindings"][name] = record
    fixture["registration"].write_text(json.dumps(registration, sort_keys=True) + "\n", encoding="utf-8")
    registration_record = gate.file_record(fixture["registration"], root=fixture["root"])
    run["registration"] = registration_record
    fixture["run"].write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")
    # Add B0/B1 after the panel/capture records so the frequency helper can
    # rewrite every prediction metadata record against the final registration
    # hash rather than leaving a stale hash in the synthetic artifacts.
    _patch_frequency_binding(fixture)
    _freeze_fixture(fixture)
    freeze = gate.validate_before_truth(freeze_path=fixture["freeze"], repository_root=fixture["root"])
    descriptor_path = fixture["task_root"] / "truth_descriptor.json"
    _write_descriptor(
        fixture=fixture,
        freeze=freeze,
        descriptor_path=descriptor_path,
        payload_path=fixture["task_root"] / "sealed_truth.safetensors",
    )
    return fixture, freeze, descriptor_path


def test_descriptor_rejection_precedes_truth_payload_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, freeze, descriptor_path = _prepared_fixture(tmp_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["cells"][0]["record_ids_sha256"] = "d" * 64
    descriptor_path.write_text(json.dumps(descriptor, sort_keys=True) + "\n", encoding="utf-8")
    opened: list[str] = []
    monkeypatch.setattr(score_cli, "_load_sealed_truth", lambda *args, **kwargs: opened.append("payload") or {})

    with pytest.raises(register.RegisterError, match="source order"):
        score_cli.score_from_truth_descriptor(
            freeze_path=fixture["freeze"],
            truth_descriptor_path=descriptor_path,
            repository_root=fixture["root"],
            frequency_reference_paths=_frequency_paths(fixture),
            output_path=fixture["task_root"] / "score.json",
            bootstrap_draws=10,
        )
    assert opened == []


def test_public_gate_rejection_precedes_descriptor_payload_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, freeze, descriptor_path = _prepared_fixture(tmp_path)
    fixture["inputs"]["source_selection"].write_text("changed after freeze\n", encoding="utf-8")
    opened: list[str] = []
    monkeypatch.setattr(score_cli, "_load_sealed_truth", lambda *args, **kwargs: opened.append("payload") or {})

    with pytest.raises(gate.GateError, match="input source_selection hash or size changed"):
        score_cli.score_from_truth_descriptor(
            freeze_path=fixture["freeze"],
            truth_descriptor_path=descriptor_path,
            repository_root=fixture["root"],
            frequency_reference_paths=_frequency_paths(fixture),
            output_path=fixture["task_root"] / "score.json",
            bootstrap_draws=10,
        )
    assert opened == []


def test_valid_descriptor_opens_sealed_payload_only_after_public_gate(tmp_path: Path) -> None:
    fixture, freeze, descriptor_path = _prepared_fixture(tmp_path)
    result = score_cli.score_from_truth_descriptor(
        freeze_path=fixture["freeze"],
        truth_descriptor_path=descriptor_path,
        repository_root=fixture["root"],
        frequency_reference_paths=_frequency_paths(fixture),
        output_path=fixture["task_root"] / "score.json",
        bootstrap_draws=10,
    )
    assert result["status"] == "SCORE_COMPLETE_AFTER_PUBLIC_FREEZE"
    assert result["truth_opened"] is True
    assert result["truth_loader_invocations"] == 1
    assert result["truth_descriptor"]["sha256"] == gate.sha256_file(descriptor_path)
    receipt_record = result["boundary_receipt"]
    receipt_path = Path(receipt_record["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == score_cli.BOUNDARY_RECEIPT_STATUS
    assert receipt["score_artifact"]["sha256"] == result["score_artifact"]["sha256"]
    assert receipt["truth_descriptor"]["sha256"] == result["truth_descriptor"]["sha256"]
    assert receipt["truth_payload"]["sha256"] == result["truth_payload_binding"]["sha256"]
    assert receipt["score_artifact_rewritten"] is False
    saved_score = json.loads(Path(result["score_artifact"]["path"]).read_text(encoding="utf-8"))
    assert "truth_descriptor" not in saved_score
