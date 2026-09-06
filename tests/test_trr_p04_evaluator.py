"""Lightweight checks for the evaluator-only P04 preparation contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.trr_p04.native_anchor_runner import build_preflight as build_anchor_preflight
from scripts.trr_p04.prepare_evaluator_observations import (
    EvaluatorObservationError,
    build_preflight as build_observation_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "experiments/TRR-P04/setup/public_selection-r2.json"
TARGET_PLAN = ROOT / "experiments/TRR-P04/setup/evaluator_target_plan.json"


def test_evaluator_preflight_binds_frozen_panel_without_model(tmp_path: Path) -> None:
    output = tmp_path / "observation-preflight"
    receipt = build_observation_preflight(
        selection_path=SELECTION,
        target_plan_path=TARGET_PLAN,
        output_root=output,
        argv=["prepare_evaluator_observations.py", "--preflight-only"],
    )
    assert receipt["status"] == "PASS_NO_MODEL_NO_TARGET_NO_TRUTH"
    assert receipt["selection"]["record_count"] == 72
    assert receipt["selection"]["anchor_count"] == 12
    assert receipt["access"]["model_loaded"] is False
    assert receipt["access"]["target_update_loaded"] is False
    saved = json.loads((output / "evaluator_capture_preflight.json").read_text(encoding="utf-8"))
    assert saved["target_plan"]["seed"] == 20260910
    assert "token_ids" in saved["forbidden_serialized_fields"]


def test_native_anchor_preflight_binds_separate_384_position_denominator(tmp_path: Path) -> None:
    receipt = build_anchor_preflight(
        selection_path=SELECTION,
        target_plan_path=TARGET_PLAN,
        output_root=tmp_path / "anchor-preflight",
        argv=["native_anchor_runner.py", "--preflight-only"],
    )
    assert receipt["status"] == "PASS_NO_MODEL_NO_TARGET_NO_TRUTH"
    assert receipt["anchor"]["record_count"] == 12
    assert receipt["anchor"]["scored_positions_per_target"] == 384
    assert receipt["algorithm"]["expected_candidate_simulations"] == 98304
    assert receipt["anchor"]["denominator_separate"] is True


def test_evaluator_plan_rejects_modified_target_seed(tmp_path: Path) -> None:
    plan = json.loads(TARGET_PLAN.read_text(encoding="utf-8"))
    plan["update"]["initialization_seed"] = 2711
    changed = tmp_path / "changed-plan.json"
    changed.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(EvaluatorObservationError, match="configuration changed"):
        build_observation_preflight(
            selection_path=SELECTION,
            target_plan_path=changed,
            output_root=tmp_path / "rejected",
            argv=["prepare_evaluator_observations.py", "--preflight-only"],
        )
