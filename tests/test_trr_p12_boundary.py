"""CPU-only synthetic tests for the TRR-P12 boundary/forecast design."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from scripts.trr_p12.adapter import AdapterError, infer_b1_one_record
from scripts.trr_p12.analysis import paired_stage_analysis, json_ready, source_order_digest
from scripts.trr_p12.boundary import analyze_score_pair, geometric_boundary_metrics
from scripts.trr_p12.predict import direction_forecast, evaluate_forecast_against_truth


def _scores(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_full_vocabulary_competitor_proves_strict_crossing() -> None:
    baseline = _scores([[1.0, 0.9, 0.8, -4.0, -5.0, -6.0]])
    updated = _scores([[0.7, 0.6, 1.2, -4.0, -5.0, -6.0]])
    result = analyze_score_pair(baseline, updated)
    assert result.baseline_winner.tolist() == [0]
    assert result.baseline_runner_up.tolist() == [1]
    assert result.updated_winner.tolist() == [2]
    assert result.transition_competitor.tolist() == [2]
    assert result.transition_margin_before.item() == pytest.approx(0.2)
    # The baseline winner remains the argmax, so the transition competitor's
    # baseline score must be taken from the full vocabulary, not runner-up 1.
    assert result.baseline_competitor_score.item() == pytest.approx(0.8)
    assert result.transition_margin_after.item() == pytest.approx(-0.5)
    assert result.strict_crossing.tolist() == [True]


def test_large_unsigned_movement_without_crossing_is_not_a_crossing() -> None:
    baseline = _scores([[1.0, 0.8, 0.0, -1.0]])
    updated = _scores([[51.0, 50.8, 0.0, -100.0]])
    result = analyze_score_pair(baseline, updated)
    assert result.argmax_changed.tolist() == [False]
    assert result.strict_crossing.tolist() == [False]
    assert result.unsigned_pair_movement.item() > 90.0
    assert result.signed_displacement.item() == pytest.approx(-0.0, abs=1e-6)


def test_tie_transition_is_separate_from_strict_crossing() -> None:
    baseline = _scores([[0.2, 0.9, 1.0, -2.0]])
    updated = _scores([[0.2, 1.0, 1.0, -2.0]])
    result = analyze_score_pair(baseline, updated)
    assert result.baseline_winner.tolist() == [2]
    assert result.updated_winner.tolist() == [1]
    assert result.argmax_changed.tolist() == [True]
    assert result.tie_transition.tolist() == [True]
    assert result.strict_crossing.tolist() == [False]


def test_geometric_signed_distance_matches_score_margin() -> None:
    embedding = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    base_z = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    update_z = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)
    scale = 2.0
    baseline = scale * (base_z @ embedding.T)
    updated = scale * (update_z @ embedding.T)
    result = analyze_score_pair(baseline, updated)
    geometry = geometric_boundary_metrics(result, base_z, update_z, embedding, logit_scale=scale)
    assert geometry["signed_distance_before"].item() == pytest.approx(1.0)
    assert geometry["signed_distance_after"].item() == pytest.approx(-1.0)
    assert geometry["signed_distance_displacement"].item() == pytest.approx(-2.0)
    assert geometry["strict_geometric_crossing"].tolist() == [True]
    assert geometry["score_margin_reconstruction_error_max"] < 1e-5


def test_direction_forecast_uses_only_stage_zero_and_64() -> None:
    stage0 = _scores([
        [1.0, 0.9, 0.0, -1.0],
        [1.0, 0.9, 0.0, -1.0],
    ])
    stage64 = _scores([
        [0.8, 0.95, 0.0, -1.0],
        [0.98, 0.91, 0.0, -1.0],
    ])
    forecast = direction_forecast(stage0, stage64)
    assert forecast.forecast_margin[128][0].item() < 0.0
    assert forecast.predicted_strict_change_by_stage[128].tolist() == [True, False]
    assert forecast.predicted_strict_change_by_stage[256].tolist() == [True, True]
    assert forecast.predicted_strict_change.tolist() == [True, True]
    assert forecast.predicted_argmax_change.tolist() == [True, True]
    assert forecast.summary()["forecast_stages"] == [128, 256]


def test_forecast_evaluation_masks_records_and_aligns_post_bos_rows() -> None:
    # Row 0 is predicted to cross at both later stages.  Row 1 crosses only
    # at 256, which catches accidental use of the union risk for a 128 table.
    stage0 = torch.tensor(
        [
            [[1.0, 0.9], [1.0, 0.9], [1.0, 0.9]],
            [[1.0, 0.9], [1.0, 0.9], [1.0, 0.9]],
        ],
        dtype=torch.float32,
    )
    stage64 = torch.tensor(
        [
            [[0.8, 0.95], [0.8, 0.95], [0.8, 0.95]],
            [[0.98, 0.91], [0.98, 0.91], [0.98, 0.91]],
        ],
        dtype=torch.float32,
    )
    forecast = direction_forecast(stage0, stage64)
    assert forecast.predicted_strict_change_by_stage[128].tolist() == [True] * 3 + [False] * 3
    assert forecast.predicted_strict_change_by_stage[256].tolist() == [True] * 6
    baseline = torch.tensor([[128000, 0, 0, 0], [128000, 1, 1, 1]], dtype=torch.long)
    truth = torch.tensor([[128000, 0, 9, 0], [128000, 1, 1, 1]], dtype=torch.long)
    later128 = torch.tensor([[128000, 9, 9, 0], [128000, 1, 1, 1]], dtype=torch.long)
    later256 = torch.tensor([[128000, 0, 9, 0], [128000, 1, 9, 1]], dtype=torch.long)
    masked = evaluate_forecast_against_truth(
        forecast,
        baseline,
        later128,
        truth,
        record_mask=torch.tensor([True, False]),
        later_stage=128,
    )
    assert masked["later_stage"] == 128
    assert masked["all_valid"]["rows"] == 3
    assert masked["all_valid"]["baseline_correct_rows"] == 2
    assert masked["all_valid"]["broken_rows"] == 1
    assert masked["all_valid"]["predicted_strict_change_rows"] == 3
    assert masked["all_valid"]["true_positive"] == 1
    assert masked["all_valid"]["false_positive"] == 2
    assert masked["all_valid"]["true_negative"] == 0
    assert masked["all_valid"]["false_negative"] == 0
    assert masked["baseline_correct_only"]["rows"] == 2
    assert masked["baseline_correct_only"]["broken_rows"] == 1
    assert masked["baseline_correct_only"]["true_positive"] == 1
    assert masked["baseline_correct_only"]["false_positive"] == 1
    assert sum(masked["all_valid"][key] for key in ("true_positive", "false_positive", "true_negative", "false_negative")) == masked["all_valid"]["rows"]
    later = evaluate_forecast_against_truth(forecast, baseline, later256, truth, later_stage=256)
    assert later["later_stage"] == 256
    assert later["all_valid"]["rows"] == 6
    assert later["all_valid"]["broken_rows"] == 1
    assert later["all_valid"]["predicted_strict_change_rows"] == 6
    assert later["all_valid"]["true_positive"] == 1
    assert later["baseline_correct_only"]["rows"] == 5
    assert later["baseline_correct_only"]["broken_rows"] == 1


def test_source_paired_analysis_reports_broken_improved_and_exact_units() -> None:
    truth = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    baseline = torch.tensor([[0, 1, 9, 4], [0, 1, 2, 3]], dtype=torch.long)
    stage = torch.tensor([[0, 9, 2, 3], [0, 1, 2, 9]], dtype=torch.long)
    result = paired_stage_analysis(
        baseline,
        stage,
        truth,
        source_ids=["a", "b"],
        domain="synthetic",
        stage=128,
        bootstrap_draws=100,
    )
    assert result["paired"]["broken_tokens"] == 2
    assert result["paired"]["improved_tokens"] == 2
    assert result["paired"]["broken_exact_records"] == 1
    assert result["paired"]["improved_exact_records"] == 0
    assert result["source_order_sha256"] == source_order_digest(["a", "b"])
    assert json_ready(result)["schema"].endswith("source-paired-analysis.v1")


def test_adapter_rejects_non_cpu_before_loading_package() -> None:
    with pytest.raises(AdapterError, match="CPU-only"):
        infer_b1_one_record(Path("/does/not/matter"), device="cuda")


@pytest.mark.skipif(os.environ.get("TRR_P12_RUN_FROZEN_PACKAGE_SMOKE") != "1", reason="explicit frozen-package CPU smoke")
def test_frozen_package_one_record_geometry_smoke() -> None:
    package = Path("../TRR-P11/restore-runtime/actual-r2").resolve()
    result = infer_b1_one_record(package)
    assert result.geometry()["projected_hidden_shape"] == [127, 2048]
    assert result.geometry()["logits_shape"] == [127, 128256]
    assert result.geometry()["prediction_shape"] == [128]
    assert result.state_sha256 == "088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706"
    expected_path = package / "smoke" / "expected_predictions.safetensors"
    with safe_open(str(expected_path), framework="pt", device="cpu") as handle:
        expected = handle.get_tensor("expanded_fixed")[0]
    assert torch.equal(result.prediction, expected)
