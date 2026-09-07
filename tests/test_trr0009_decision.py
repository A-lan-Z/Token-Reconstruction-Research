from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import trr0009_decision as decision
from scripts import trr0009_eval_contract as eval_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "experiments/TRR-0009/planning/decision_contract.json").read_text())


def _score(*, exact_point: float = 0.50, exact_lower: float = 0.01, token_point: float = 0.003, token_lower: float = 0.001, rare_lower: float = 0.0, rare_support: int = 64) -> dict:
    contrasts = {}
    for cell in eval_contract.CELL_ORDER:
        contrasts[cell] = {
            "records": 64,
            "exact_net_rate": exact_point if cell == "finance__public_base" else 0.0,
            "exact_gains": 32 if cell == "finance__public_base" else 32,
            "exact_losses": 0,
            "exact_net_cp_lower_bound": exact_lower if cell == "finance__public_base" else 0.0,
            "token_net_rate": token_point if cell == "finance__public_base" else 0.0,
            "token_net_bootstrap": {
                "status": "COMPUTED",
                "lower": token_lower if cell == "finance__public_base" else 0.0,
                "upper": 0.01,
                "exposed_source_records": 64,
                "exposed_tokens": 64 * 127,
                "unit": "source_record",
                "one_sided_alpha": 0.025,
            },
            "token_net_bootstrap_harm": {
                "status": "COMPUTED",
                "lower": token_lower if cell == "finance__public_base" else 0.0,
                "upper": 0.01,
                "one_sided_alpha": 0.05,
                "exposed_source_records": 64,
                "exposed_tokens": 64 * 127,
                "unit": "source_record",
            },
        }
    rare_rows = {}
    for cell in eval_contract.CELL_ORDER:
        for bin_name in ("0", "1-4"):
            rare_rows[f"{cell}::{bin_name}"] = {
                "cell_id": cell,
                "stratum": bin_name,
                "interval": {
                    "status": "COMPUTED",
                    "lower": rare_lower,
                    "upper": 0.01,
                    "exposed_source_records": rare_support,
                    "exposed_tokens": rare_support * 10,
                    "unit": "source_record",
                    "one_sided_alpha": 0.05,
                },
            }
    return {
        "schema": "token-reconstruction.trr0009-score.v1",
        "task_id": "TRR-0009",
        "status": "SCORE_COMPLETE_AFTER_PUBLIC_FREEZE",
        "method_order": list(eval_contract.METHOD_ORDER),
        "cell_order": list(eval_contract.CELL_ORDER),
        "primary": {"candidate": eval_contract.PRIMARY_METHOD_ID, "control": eval_contract.PRIMARY_CONTROL_METHOD_ID},
        "contrasts": {"candidate_vs_fixed": contrasts},
        "rare_token_safeguard": {
            "all_eight_gates_required": True,
            "rows": rare_rows,
        },
        "confidence": {
            "exact_composite_alpha": 0.025,
            "exact_component_alpha": 0.0125,
            "token_one_sided_alpha": 0.025,
        },
        "bootstrap": {"seed": 9009, "draws": 10000, "unit": "source_record"},
        "truth_opened": True,
        "source_text_or_target_labels": False,
    }


def _cost(*, alias_decision: str = "PASS", qualification_decision: str = "PASS", upper: float = 1.10, lower: float = 0.90) -> dict:
    return {
        "qualification": {
            "decision": qualification_decision,
            "candidate_method_id": eval_contract.PRIMARY_METHOD_ID,
            "denominator_method_id": eval_contract.PRIMARY_CONTROL_METHOD_ID,
            "threshold": 1.25,
            "per_cell": {
                cell: {"ci_lower": lower, "ci_upper": upper}
                for cell in eval_contract.CELL_ORDER
            },
        },
        "alias_control": {
            "decision": alias_decision,
            "runtime_order_control_valid": alias_decision == "PASS",
        },
        "cost_evidence": {"model_fit_and_deployment": {}},
    }


def test_all_declared_gates_pass_without_pooling() -> None:
    result = decision.aggregate_decision(_score(), _cost(), CONTRACT)
    assert result["status"] == decision.PASS
    assert result["outcome"] == "USEFUL_MECHANISM_SCREENING_SIGNAL"
    assert result["gates"]["primary_useful"]["status"] == decision.PASS
    assert result["gates"]["exact_harm"]["status"] == decision.PASS
    assert result["gates"]["token_harm"]["status"] == decision.PASS
    assert result["gates"]["rare_absent"]["status"] == decision.PASS
    assert result["gates"]["cost"]["status"] == decision.PASS
    assert set(result["gates"]["harm"]["by_cell"]) == set(eval_contract.CELL_ORDER)
    assert len(result["gates"]["rare_absent"]["by_gate"]) == 8


def test_primary_useful_or_route_and_harm_fail_are_separate() -> None:
    score = _score(exact_point=0.01, exact_lower=-0.06, token_point=0.003, token_lower=0.001)
    score["contrasts"]["candidate_vs_fixed"]["finance__public_base"]["exact_net_cp_lower_bound_by_alpha"] = {"0.05": -0.06}
    result = decision.aggregate_decision(score, _cost(), CONTRACT)
    assert result["gates"]["primary_useful"]["status"] == decision.PASS
    assert result["gates"]["primary_useful"]["routes"]["exact"]["status"] == decision.FAIL
    assert result["gates"]["primary_useful"]["routes"]["token"]["status"] == decision.PASS
    assert result["gates"]["exact_harm"]["status"] == decision.FAIL
    assert result["status"] == decision.FAIL


def test_positive_evidence_below_useful_floor_is_reported_separately() -> None:
    score = _score(exact_point=0.01, exact_lower=0.005, token_point=0.001, token_lower=0.0005)
    result = decision.aggregate_decision(score, _cost(), CONTRACT)
    routes = result["gates"]["primary_useful"]["routes"]
    assert result["gates"]["primary_useful"]["status"] == decision.FAIL
    assert routes["exact"]["status"] == decision.FAIL
    assert routes["exact"]["positive_status"] == decision.PASS
    assert routes["token"]["status"] == decision.FAIL
    assert routes["token"]["positive_status"] == decision.PASS


def test_underpowered_rare_gate_and_unqualified_alias_are_unknown() -> None:
    score = _score(rare_support=31)
    result = decision.aggregate_decision(score, _cost(alias_decision="INCONCLUSIVE"), CONTRACT)
    assert result["gates"]["rare_absent"]["status"] == decision.UNKNOWN
    assert result["gates"]["cost"]["status"] == decision.UNKNOWN
    assert result["status"] == decision.UNKNOWN
    assert result["action"] == "RETAIN_REFERENCE_AND_LABEL_ADAPTATION_QUESTION_UNRESOLVED"


def test_cost_failure_requires_alias_qualified_all_cell_bounds() -> None:
    result = decision.aggregate_decision(_score(), _cost(upper=1.40, lower=1.30, qualification_decision="FAIL"), CONTRACT)
    assert result["gates"]["cost"]["status"] == decision.FAIL
    assert result["status"] == decision.FAIL


def test_missing_cell_or_interval_fails_closed() -> None:
    score = _score()
    del score["contrasts"]["candidate_vs_fixed"][eval_contract.CELL_ORDER[0]]
    with pytest.raises(decision.DecisionError, match="every unpooled cell"):
        decision.aggregate_decision(score, _cost(), CONTRACT)

    score = _score()
    score["contrasts"]["candidate_vs_fixed"]["pile__public_base"]["token_net_bootstrap"] = {"status": "UNKNOWN"}
    del score["contrasts"]["candidate_vs_fixed"]["pile__public_base"]["token_net_bootstrap_harm"]
    result = decision.aggregate_decision(score, _cost(), CONTRACT)
    assert result["gates"]["token_harm"]["status"] == decision.UNKNOWN
    assert result["status"] == decision.UNKNOWN


def test_contract_cell_order_is_not_a_hidden_pooling_rule() -> None:
    contract = copy.deepcopy(CONTRACT)
    contract["safeguards"]["cells"] = list(reversed(contract["safeguards"]["cells"]))
    contract["rare_token_safeguard"]["cells"] = list(reversed(contract["rare_token_safeguard"]["cells"]))
    contract["cost_gate"]["cells"] = list(reversed(contract["cost_gate"]["cells"]))
    result = decision.aggregate_decision(_score(), _cost(), contract)
    assert result["status"] == decision.PASS


def test_primary_alpha_does_not_satisfy_wider_safeguard_alpha() -> None:
    score = _score()
    for row in score["contrasts"]["candidate_vs_fixed"].values():
        del row["token_net_bootstrap_harm"]
    result = decision.aggregate_decision(score, _cost(), CONTRACT)
    assert result["gates"]["primary_useful"]["status"] == decision.PASS
    assert result["gates"]["token_harm"]["status"] == decision.UNKNOWN
    assert result["status"] == decision.UNKNOWN
