from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.trr0004_p04_prediction_runner import (
    METHODS,
    SEEDS,
    _load_observations,
    _load_selection,
    _load_state_manifest,
    _validate_observation_index,
    run_synthetic_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "experiments/TRR-P04/setup/public_selection-r2.json"
STATE_MANIFEST = ROOT / "experiments/TRR-P04/runtime/training-r1/selected_state_manifest.json"
OBSERVATION_ROOT = ROOT / "experiments/TRR-P04/runtime/evaluator-observations-r1"
OBSERVATION_INDEX = OBSERVATION_ROOT / "observation_index.json"


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


def test_real_observation_masks_bind_in_serialized_uint8_representation() -> None:
    if not OBSERVATION_INDEX.is_file():
        import pytest

        pytest.skip("setup-owned evaluator observations are unavailable")
    records, selection_descriptor = _load_selection(SELECTION)
    index, _ = _validate_observation_index(
        OBSERVATION_INDEX,
        selection_records=records,
        selection_descriptor=selection_descriptor,
    )
    observations, descriptors = _load_observations(
        index_path=OBSERVATION_INDEX,
        observation_root=OBSERVATION_ROOT,
        index=index,
        records=records,
        selection_descriptor=selection_descriptor,
    )
    assert set(observations) == {"public_base", "p04_evaluator_target_update_v1"}
    assert all(value[1].dtype == torch.bool for value in observations.values())
    assert all(
        descriptor["attention_mask_sha256"] == "854efd9f11edec5c584e908ef33ed6ba6899ce65bde6c8421c6ccd4dc0dc11de"
        for descriptor in descriptors.values()
    )
