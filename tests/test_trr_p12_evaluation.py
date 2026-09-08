"""CPU synthetic tests for the pretruth P12 evaluator."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.trr_p12 import analysis
from scripts.trr_p12.evaluation import EvaluationError, freeze_evaluation, score_evaluation


def _binding(path: Path, *, key: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if key is not None:
        value["key"] = key
    return value


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    code = tmp_path / "decoder.py"
    code.write_text("# frozen synthetic decoder\n", encoding="utf-8")
    observation = tmp_path / "observation.json"
    observation.write_text('{"sanitized": true}\n', encoding="utf-8")
    projected = tmp_path / "projected.safetensors"
    save_file({"projected_hidden": torch.zeros((2, 2, 3), dtype=torch.float32)}, str(projected))
    predictions = {
        0: torch.tensor([[0, 1, 1], [0, 2, 2]], dtype=torch.long),
        64: torch.tensor([[0, 1, 1], [0, 2, 2]], dtype=torch.long),
        128: torch.tensor([[0, 1, 2], [0, 3, 2]], dtype=torch.long),
        256: torch.tensor([[0, 1, 1], [0, 2, 2]], dtype=torch.long),
    }
    prediction_paths: dict[int, Path] = {}
    for stage, prediction in predictions.items():
        path = tmp_path / f"prediction-{stage}.safetensors"
        save_file({"expanded_fixed": prediction}, str(path))
        prediction_paths[stage] = path
    forecast = tmp_path / "forecast.json"
    rows = []
    for index in range(2):
        rows.append({
            "record_index": index,
            "slot": f"record/{index}",
            "valid": [True, True],
            "predicted_strict_change_by_stage": {
                "128": [False, False] if index == 0 else [True, False],
                "256": [False, False],
            },
        })
    forecast.write_text(json.dumps({"schema": "token-reconstruction.trr-p12-direction-forecast-artifact.v1", "rows": rows}) + "\n", encoding="utf-8")
    source_hashes = ["1" * 64, "2" * 64]
    source_order = {"hashes": source_hashes, "sha256": analysis.source_order_digest(source_hashes)}
    stages: dict[str, object] = {}
    for stage, prediction_path in prediction_paths.items():
        stages[str(stage)] = {
            "record_count": 2,
            "artifacts": {
                "observation": _binding(observation),
                "prediction": _binding(prediction_path, key="expanded_fixed"),
                "projected": _binding(projected, key="projected_hidden"),
            },
            "costs": {"elapsed_seconds": 0.01, "rss_bytes": 1},
            "code": {"pipeline": _binding(code)},
        }
    descriptor = {
        "schema": "token-reconstruction.trr-p12-evaluation-freeze.v1",
        "status": "FREEZE_COMPLETE_BEFORE_TRUTH",
        "truth_opened": False,
        "source_text_loaded": False,
        "target_weights_loaded": False,
        "source_orders": {"toy": source_order},
        "methods": {"B1": {"domains": {"toy": {"forecast": _binding(forecast), "stages": stages}}}},
    }
    descriptor_path = tmp_path / "freeze-descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    truth_path = tmp_path / "truth.safetensors"
    truth = torch.tensor([[0, 1, 2], [0, 2, 2]], dtype=torch.long)
    save_file({f"truth-{stage}": truth.clone() for stage in (0, 64, 128, 256)}, str(truth_path))
    return descriptor_path, descriptor, truth_path


def test_freeze_rejects_missing_stage_and_wrong_hash(tmp_path: Path) -> None:
    descriptor_path, descriptor, _ = _fixture(tmp_path)
    missing = copy.deepcopy(descriptor)
    del missing["methods"]["B1"]["domains"]["toy"]["stages"]["64"]
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps(missing) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="stages must be"):
        freeze_evaluation(missing_path, tmp_path / "missing-receipt.json", strict_record_counts=False)

    bad = copy.deepcopy(descriptor)
    bad["methods"]["B1"]["domains"]["toy"]["stages"]["0"]["artifacts"]["prediction"]["sha256"] = "0" * 64
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="sha256 changed"):
        freeze_evaluation(bad_path, tmp_path / "bad-receipt.json", strict_record_counts=False)


def test_score_reports_paired_counts_and_second_half_forecast(tmp_path: Path) -> None:
    descriptor_path, _, truth_path = _fixture(tmp_path)
    freeze_path = tmp_path / "freeze-receipt.json"
    freeze = freeze_evaluation(descriptor_path, freeze_path, strict_record_counts=False)
    truth_descriptor = {
        "schema": "token-reconstruction.trr-p12-evaluation-truth.v1",
        "status": "TRUTH_READY_AFTER_FREEZE",
        "truth_opened": True,
        "freeze_receipt_sha256": freeze["receipt_sha256"],
        "cells": [
            {
                "method": "B1",
                "domain": "toy",
                "stage": stage,
                "source_order_sha256": freeze["methods"]["B1"]["domains"]["toy"]["source_order"]["sha256"],
                "truth": _binding(truth_path, key=f"truth-{stage}"),
            }
            for stage in (0, 64, 128, 256)
        ],
    }
    truth_descriptor_path = tmp_path / "truth-descriptor.json"
    truth_descriptor_path.write_text(json.dumps(truth_descriptor, indent=2) + "\n", encoding="utf-8")
    output = tmp_path / "score.json"
    result = score_evaluation(
        freeze_path,
        truth_descriptor_path,
        output,
        strict_record_counts=False,
        bootstrap_draws=20,
    )
    stage128 = result["methods"]["B1"]["domains"]["toy"]["stages"]["128"]
    assert stage128["paired"]["broken_tokens"] == 1
    assert stage128["paired"]["improved_tokens"] == 1
    assert stage128["stage_score"]["token_errors"] == 1
    forecast = result["methods"]["B1"]["domains"]["toy"]["forecast"]["stages"]["128"]
    primary = forecast["primary_second64"]["primary_baseline_correct_only"]
    assert primary["eligible_tokens"] == 2
    assert primary["broken_tokens"] == 1
    assert primary["true_positive"] == 1
    assert primary["false_negative"] == 0
    assert forecast["descriptive_full128"]["record_count"] == 2
    assert result["truth_opened"] is True
    assert json.loads(output.read_text())["status"] == "SCORED_AFTER_FREEZE"


def test_strict_freeze_rejects_partial_domain_or_missing_a1(tmp_path: Path) -> None:
    descriptor_path, _, _ = _fixture(tmp_path)
    with pytest.raises(EvaluationError, match="both B1 and A1"):
        freeze_evaluation(descriptor_path, tmp_path / "strict-receipt.json")


def test_freeze_rejects_public_prediction_wrong_record_count(tmp_path: Path) -> None:
    descriptor_path, descriptor, _ = _fixture(tmp_path)
    wrong = tmp_path / "wrong-count.safetensors"
    save_file({"expanded_fixed": torch.zeros((3, 3), dtype=torch.long)}, str(wrong))
    descriptor = copy.deepcopy(descriptor)
    descriptor["methods"]["B1"]["domains"]["toy"]["stages"]["0"]["artifacts"]["prediction"] = _binding(wrong, key="expanded_fixed")
    wrong_descriptor = tmp_path / "wrong-count-descriptor.json"
    wrong_descriptor.write_text(json.dumps(descriptor) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="record count"):
        freeze_evaluation(wrong_descriptor, tmp_path / "wrong-count-receipt.json", strict_record_counts=False)


def test_a1_binds_candidate_receipt_without_fake_projected_or_forecast(tmp_path: Path) -> None:
    descriptor_path, descriptor, _ = _fixture(tmp_path)
    candidate_receipt = tmp_path / "candidate-receipt.json"
    candidate_receipt.write_text('{"candidate": true}\n', encoding="utf-8")
    a1_stages = {}
    for stage in (0, 256):
        b1_stage = descriptor["methods"]["B1"]["domains"]["toy"]["stages"][str(stage)]
        a1_stages[str(stage)] = {
            "record_count": 2,
            "artifacts": {
                "observation": b1_stage["artifacts"]["observation"],
                "prediction": b1_stage["artifacts"]["prediction"],
                "candidate_receipt": _binding(candidate_receipt),
            },
            "costs": {"elapsed_seconds": 0.01},
            "code": b1_stage["code"],
        }
    descriptor["methods"]["A1+A2"] = {"domains": {"toy": {"source_order": descriptor["source_orders"]["toy"], "stages": a1_stages}}}
    descriptor_path.write_text(json.dumps(descriptor) + "\n", encoding="utf-8")
    receipt = freeze_evaluation(descriptor_path, tmp_path / "a1-receipt.json", strict_record_counts=False)
    assert set(receipt["methods"]["A1+A2"]["domains"]["toy"]["stages"]["0"]["artifacts"]) == {"observation", "prediction", "candidate_receipt"}

    bad = copy.deepcopy(descriptor)
    del bad["methods"]["A1+A2"]["domains"]["toy"]["stages"]["0"]["artifacts"]["candidate_receipt"]
    bad_path = tmp_path / "a1-missing-candidate.json"
    bad_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="candidate receipt"):
        freeze_evaluation(bad_path, tmp_path / "a1-bad-receipt.json", strict_record_counts=False)
