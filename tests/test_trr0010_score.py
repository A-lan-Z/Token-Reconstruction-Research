"""Truth-gated synthetic tests for the TRR-0010 scorer adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts import trr0010_eval_gate as gate
from scripts import trr0010_score as score
from tests.test_trr0010_eval_gate import _fixture, _rewrite_prediction


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _frequency_payload(bank: str, count: int) -> dict[str, Any]:
    return {
        "schema": "token-reconstruction.trr0010-frequency-reference.v1",
        "task_id": gate.TASK_ID,
        "frequency_references": {bank: {"0": count}},
        "truth_opened": False,
        "source_text_written": False,
        "source_text_loaded": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }


def _patch_frequency_binding(fixture: dict[str, Any]) -> None:
    """Bind independent synthetic B0/B1 support maps before freezing."""

    assets = fixture["root"] / "assets"
    frequency_b0_path = assets / "frequency_reference_B0.json"
    frequency_b1_path = assets / "frequency_reference_B1.json"
    _write_json(frequency_b0_path, _frequency_payload("B0", 3))
    _write_json(frequency_b1_path, _frequency_payload("B1", 30))
    fixture["inputs"]["frequency_reference_B0"] = frequency_b0_path
    fixture["inputs"]["frequency_reference_B1"] = frequency_b1_path
    registration = json.loads(fixture["registration"].read_text(encoding="utf-8"))
    registration["input_bindings"]["frequency_reference_B0"] = gate.file_record(
        frequency_b0_path, root=fixture["root"]
    )
    registration["input_bindings"]["frequency_reference_B1"] = gate.file_record(
        frequency_b1_path, root=fixture["root"]
    )
    # Retain the original generic binding for older registration readers; the
    # scorer requires and consumes the canonical B0/B1 bindings above.
    _write_json(fixture["registration"], registration)
    registration_record = gate.file_record(fixture["registration"], root=fixture["root"])

    run = json.loads(fixture["run"].read_text(encoding="utf-8"))
    run["registration"] = registration_record
    run["input_bindings"] = registration["input_bindings"]
    for method_id in gate.METHOD_ORDER:
        for cell_id in gate.CELL_ORDER:
            key = f"{method_id}::{cell_id}"
            prediction_path = Path(run["predictions"][key]["path"])
            with safe_open(str(prediction_path), framework="pt", device="cpu") as handle:
                values = handle.get_tensor("predictions").detach().cpu().contiguous()
            metadata = {
                "schema": gate.PREDICTION_SCHEMA,
                "task_id": gate.TASK_ID,
                "registration_sha256": registration_record["sha256"],
                "method_id": method_id,
                "cell_id": cell_id,
                "records": str(gate.RECORDS_PER_CELL),
                "truth_opened": "false",
                "candidate_arrays_persisted": "false",
                "geometry_json": json.dumps(
                    {"records": gate.RECORDS_PER_CELL, **gate.STATIC_GEOMETRY},
                    sort_keys=True,
                ),
            }
            save_file({"predictions": values}, str(prediction_path), metadata=metadata)
            prediction_record = gate.file_record(prediction_path, root=fixture["root"])
            prediction_record["prediction_sha256"] = gate.tensor_digest(values)
            run["predictions"][key] = prediction_record

            timing_path = Path(run["timings"][key]["path"])
            timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
            timing_payload["registration_sha256"] = registration_record["sha256"]
            timing_payload["prediction_artifact"] = prediction_record
            _write_json(timing_path, timing_payload)
            run["timings"][key] = gate.file_record(timing_path, root=fixture["root"])
    _write_json(fixture["run"], run)

def _freeze_fixture(fixture: dict[str, Any]) -> None:
    gate.write_freeze(
        registration_path=fixture["registration"],
        run_manifest_path=fixture["run"],
        output_path=fixture["freeze"],
        repository_root=fixture["root"],
    )


def _truth_fixture() -> dict[str, torch.Tensor]:
    values = torch.zeros(
        (gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS), dtype=torch.long
    )
    values[:, 0] = gate.BOS_TOKEN_ID
    return {cell: values.clone() for cell in gate.CELL_ORDER}


def test_incomplete_prediction_rejected_before_truth_opener(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _patch_frequency_binding(fixture)
    _freeze_fixture(fixture)
    run = json.loads(fixture["run"].read_text(encoding="utf-8"))
    run["predictions"].pop(f"{gate.METHOD_ORDER[0]}::{gate.CELL_ORDER[0]}")
    _write_json(fixture["run"], run)
    opened: list[str] = []

    def truth_loader() -> dict[str, torch.Tensor]:
        opened.append("truth")
        return _truth_fixture()

    with pytest.raises((gate.GateError, score.ScoreError)):
        score.score_after_gate(
            freeze_path=fixture["freeze"],
            repository_root=fixture["root"],
            truth_loader=truth_loader,
        )
    assert opened == []


def test_changed_bound_resource_rejected_before_truth_opener(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _patch_frequency_binding(fixture)
    _freeze_fixture(fixture)
    fixture["states"][gate.CURRENT_FIXED_METHOD_ID].write_bytes(b"changed after freeze\n")
    opened: list[str] = []

    def truth_loader() -> dict[str, torch.Tensor]:
        opened.append("truth")
        return _truth_fixture()

    with pytest.raises((gate.GateError, score.ScoreError)):
        score.score_after_gate(
            freeze_path=fixture["freeze"],
            repository_root=fixture["root"],
            truth_loader=truth_loader,
        )
    assert opened == []


def test_valid_matrix_opens_truth_once_and_reports_paired_metrics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _patch_frequency_binding(fixture)
    candidate = torch.zeros(
        (gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS), dtype=torch.long
    )
    candidate[:, 0] = gate.BOS_TOKEN_ID
    candidate[0, 1] = 1
    _rewrite_prediction(
        fixture,
        f"{gate.EXPANDED_DIRECTIONAL_METHOD_ID}::{gate.CELL_ORDER[0]}",
        candidate,
    )
    _freeze_fixture(fixture)
    opened: list[str] = []
    output_path = fixture["task_root"] / "evaluation" / "score.json"

    def truth_loader() -> dict[str, torch.Tensor]:
        opened.append("truth")
        return _truth_fixture()

    result = score.score_after_gate(
        freeze_path=fixture["freeze"],
        repository_root=fixture["root"],
        truth_loader=truth_loader,
        frequency_reference_paths={
            "B0": fixture["inputs"]["frequency_reference_B0"],
            "B1": fixture["inputs"]["frequency_reference_B1"],
        },
        output_path=output_path,
        bootstrap_draws=20,
    )
    assert opened == ["truth"]
    assert result["status"] == score.SCORE_STATUS
    assert result["truth_loader_invocations"] == 1
    assert result["method_order"] == list(gate.METHOD_ORDER)
    assert result["cell_order"] == list(gate.CELL_ORDER)
    assert set(result["cell_scores"]) == set(gate.METHOD_ORDER)
    assert set(result["contrasts"]) == set(score.CONTRASTS)
    contrast = result["contrasts"]["directional_expanded"][gate.CELL_ORDER[0]]
    assert contrast["exact"]["benefit"]["losses"] == 1
    assert contrast["token"]["benefit"]["candidate_correct"] < contrast["token"]["benefit"]["control_correct"]
    assert output_path.is_file()
    serialized = output_path.read_text(encoding="utf-8")
    assert "token_correct_by_record" not in serialized
    assert result["truth_opened"] is True
    assert set(result["remaining_error_reduction"]) == set(score.ERROR_REDUCTION_ROUTES)
    assert set(result["gap_closure"]) == {"directional_candidate", "data_expanded_fixed"}
    assert set(result["interaction"]) == set(gate.CELL_ORDER)
    assert result["decision_readout"]["thresholds_applied"] is True
    assert result["decision_readout"]["status"] == "CHECKS_COMPUTED_NO_POOLED_DECISION"
    assert result["decision_readout"]["cost_gate"]["status"] == "UNKNOWN"

    streams = result["bootstrap"]["index_streams"]
    assert set(streams) == {"finance", "pile"}
    for method in gate.METHOD_ORDER:
        for cell in gate.CELL_ORDER:
            domain = cell.split("__", 1)[0]
            for bank in ("B0", "B1"):
                interval = result["cell_scores"][method][cell]["frequency_strata"][bank]["seen_1_4"]["interval"]
                assert interval["resample_index_sha256"] == streams[domain]["sha256"]
            for contrast in result["contrasts"].values():
                token_interval = contrast[cell]["token"]["benefit"]
                assert token_interval["resample_index_sha256"] == streams[domain]["sha256"]
                for bank in ("B0", "B1"):
                    freq_interval = contrast[cell]["frequency"][bank]["seen_1_4"]["interval"]
                    assert freq_interval["resample_index_sha256"] == streams[domain]["sha256"]
            for route in result["gap_closure"].values():
                assert route[cell]["token"]["resample_index_sha256"] == streams[domain]["sha256"]
            assert result["interaction"][cell]["token"]["benefit"]["resample_index_sha256"] == streams[domain]["sha256"]


def test_truth_target_pair_mismatch_is_rejected_after_truth_open(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _patch_frequency_binding(fixture)
    _freeze_fixture(fixture)
    truth = _truth_fixture()
    truth["finance__public_lora_2601"] = truth["finance__public_lora_2601"].clone()
    truth["finance__public_lora_2601"][0, 1] = 1
    opened: list[str] = []

    def truth_loader() -> dict[str, torch.Tensor]:
        opened.append("truth")
        return truth

    with pytest.raises(score.ScoreError, match="source-paired and identical"):
        score.score_after_gate(
            freeze_path=fixture["freeze"],
            repository_root=fixture["root"],
            truth_loader=truth_loader,
            bootstrap_draws=20,
        )
    assert opened == ["truth"]


def test_named_frequency_banks_are_kept_distinct() -> None:
    payload = {
        "frequency_references": {
            "B0": {"0": 3},
            "B1": {"0": 30, "1": 5},
        }
    }
    banks = score._extract_frequency_banks(payload)
    assert set(banks) == {"B0", "B1"}
    assert len(banks["B0"]) == 1
    assert len(banks["B1"]) == 2
    with pytest.raises(score.ScoreError, match="multiple frequency banks"):
        score._resolve_method_frequency_banks(banks, None)
