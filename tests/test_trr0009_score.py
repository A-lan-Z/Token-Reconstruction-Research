from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from scripts import trr0009_decision as decision
from scripts import trr0009_eval_contract as contract
from scripts import trr0009_score as score


def _matrix(records: int = 32):
    truth = {}
    predictions = {method: {} for method in contract.METHOD_ORDER}
    base = torch.full((records, contract.STORED_SEQUENCE_TOKENS), contract.BOS_TOKEN_ID, dtype=torch.long)
    base[:, 1:] = 1
    for cell in contract.CELL_ORDER:
        truth[cell] = base.clone()
        for method in contract.METHOD_ORDER:
            predictions[method][cell] = base.clone()
    predictions[contract.PRIMARY_METHOD_ID][contract.CELL_ORDER[0]][0, 1] = 2
    predictions[contract.PRIMARY_CONTROL_METHOD_ID][contract.CELL_ORDER[0]][1, 1] = 2
    return predictions, truth


def test_cp_uses_two_component_tails_and_exact_endpoints() -> None:
    all_success = score.clopper_pearson(32, 32, alpha=0.025)
    all_failure = score.clopper_pearson(0, 32, alpha=0.025)
    assert all_success["lower"] > 0.0
    assert all_success["upper"] == 1.0
    assert all_failure["lower"] == 0.0
    assert all_failure["upper"] < 1.0
    assert all_success["component_alpha"] == pytest.approx(0.0125)
    from scipy.stats import beta
    assert all_success["lower"] == pytest.approx(float(beta.ppf(0.0125, 32, 1)))


def test_frequency_summary_is_exposure_weighted_and_unknown_below_32_sources() -> None:
    prediction = torch.full((40, 128), contract.BOS_TOKEN_ID, dtype=torch.long)
    truth = prediction.clone()
    truth[:, 1:] = 1
    prediction[:, 1:] = 1
    prediction[:10, 1:3] = 2
    counts = {1: 1}
    cell = score._cell_score(prediction, truth, frequency_counts=counts)
    summary = score._serialize_frequency_summary(cell, seed=1, draws=100)
    row = summary["seen_1_4"]
    assert row["correct_tokens"] == 40 * 127 - 20
    assert row["exposed_tokens"] == 40 * 127
    assert row["token_accuracy"] == pytest.approx((40 * 127 - 20) / (40 * 127))
    assert row["interval"]["status"] == "COMPUTED"

    small = score._cell_score(prediction[:31], truth[:31], frequency_counts=counts)
    small_summary = score._serialize_frequency_summary(small, seed=1, draws=10)
    assert small_summary["seen_1_4"]["interval"]["status"] == "UNKNOWN"


def test_score_predictions_emits_all_token_harm_interval_for_decision() -> None:
    predictions, truth = _matrix(records=32)
    result = score.score_predictions(predictions, truth, bootstrap_draws=10000)
    row = result["contrasts"]["candidate_vs_fixed"][contract.CELL_ORDER[0]]
    assert row["token_net_bootstrap"]["one_sided_alpha"] == pytest.approx(0.025)
    assert row["token_net_bootstrap_harm"]["one_sided_alpha"] == pytest.approx(0.05)
    assert row["token_net_bootstrap_harm"]["seed"] == row["token_net_bootstrap"]["seed"]
    assert row["token_net_bootstrap_harm"]["draws"] == row["token_net_bootstrap"]["draws"]

    decision_contract = json.loads((Path(__file__).resolve().parents[1] / "experiments/TRR-0009/planning/decision_contract.json").read_text())
    cost = {
        "qualification": {
            "decision": "PASS",
            "candidate_method_id": contract.PRIMARY_METHOD_ID,
            "denominator_method_id": contract.PRIMARY_CONTROL_METHOD_ID,
            "threshold": 1.25,
            "per_cell": {cell: {"ci_lower": 0.9, "ci_upper": 1.1} for cell in contract.CELL_ORDER},
        },
        "alias_control": {"decision": "PASS", "runtime_order_control_valid": True},
        "cost_evidence": {},
    }
    aggregated = decision.aggregate_decision(result, cost, decision_contract)
    assert aggregated["gates"]["token_harm"]["status"] != decision.UNKNOWN
    assert all(row["status"] != decision.UNKNOWN for row in aggregated["gates"]["token_harm"]["by_cell"].values())


def test_score_matrix_keeps_all_target_conditions_cell_local() -> None:
    predictions, truth = _matrix()
    result = score.score_predictions(predictions, truth, frequency_counts={1: 3}, bootstrap_draws=100)
    assert set(result["cell_order"]) == set(contract.CELL_ORDER)
    assert set(result["contrasts"]["candidate_vs_fixed"]) == set(contract.CELL_ORDER)
    assert "pooled" not in str(result).lower()
    assert result["contrasts"]["candidate_vs_fixed"][contract.CELL_ORDER[0]]["exact_gains"] == 1
    assert result["contrasts"]["candidate_vs_fixed"][contract.CELL_ORDER[0]]["exact_losses"] == 1
