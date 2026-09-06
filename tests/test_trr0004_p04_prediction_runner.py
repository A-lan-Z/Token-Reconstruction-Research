from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.trr0004_p04_prediction_runner import (
    METHODS,
    SEEDS,
    _load_selection,
    _load_state_manifest,
    run_synthetic_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "experiments/TRR-P04/setup/public_selection-r2.json"
STATE_MANIFEST = ROOT / "experiments/TRR-P04/runtime/training-r1/selected_state_manifest.json"


def test_prediction_runner_synthetic_cli_path_serializes_lowest_id_ties(tmp_path: Path) -> None:
    result = run_synthetic_smoke(tmp_path / "smoke")
    assert result["status"] == "PASS"
    assert result["truth_accessed"] is False
    assert result["repeated_prediction_exact"] is True
    assert result["actual_post_bos_prediction"] == 0
    assert result["actual_post_bos_tie_count"] == 2
    prediction = json.loads((tmp_path / "smoke" / "predictions.jsonl").read_text(encoding="utf-8"))
    ties = json.loads((tmp_path / "smoke" / "tie_diagnostics.json").read_text(encoding="utf-8"))
    assert prediction["predicted_token_ids"] == [0]
    assert ties["rows"][0]["tie_counts"] == [2]
    assert ties["summary"]["positions_with_tie"] == 1


def test_real_selection_and_selected_manifest_bind_the_eight_evaluation_states() -> None:
    records, descriptor = _load_selection(SELECTION)
    assert len(records) == 72
    assert descriptor["anchor_count"] == 12
    payload, states = _load_state_manifest(STATE_MANIFEST)
    assert payload["truth_accessed"] is False
    assert payload["evaluation_state_count"] == 8
    assert payload["all_frozen_state_count"] == 16
    assert set(states) == {(method, seed) for method in METHODS for seed in SEEDS}
    assert all(row["evaluation_input"] is True for row in states.values())
    assert len(payload["excluded_final_states"]) == 8
    assert payload["training_provenance"]["finalized_after_late_cli_failure"] is True
