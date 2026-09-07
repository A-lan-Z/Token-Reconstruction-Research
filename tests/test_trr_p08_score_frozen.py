from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.trr_p08 import score_frozen


METHODS = (
    "p08_positionwise_joint",
    "p08_positionwise_staged",
    "p08_past_only_joint",
    "p08_past_only_staged",
)
SEEDS = (6106, 6107)


def _truth(records: int = 4) -> np.ndarray:
    truth = np.zeros((records, 128), dtype=np.int64)
    truth[:, 0] = 128000
    truth[:, 1:] = 2000 + np.arange(127, dtype=np.int64)
    return truth


def _cells() -> dict[str, dict[str, object]]:
    truth = _truth()
    ids = [f"source-{i}" for i in range(len(truth))]
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        predictions[method] = {}
        for seed in SEEDS:
            value = truth.copy()
            if method.endswith("staged"):
                value[0, 1] += 1
            if method.endswith("joint"):
                value[1, 1] += 1
            if seed == 6107 and method.startswith("p08_past"):
                value[2, 1] += 1
            predictions[method][str(seed)] = value
    return {
        f"{domain}/{target}": {
            "domain": domain,
            "target": target,
            "record_ids": ids,
            "truth": truth.copy(),
            "predictions": copy.deepcopy(predictions),
        }
        for domain in ("pile", "finance")
        for target in ("public_base", "public_lora_2601")
    }


def test_score_arrays_scores_all_cells_and_keeps_targets_separate() -> None:
    result = score_frozen.score_arrays(_cells(), bootstrap_draws=32, bootstrap_seed=8080)
    assert result["schema"] == score_frozen.SCORE_SCHEMA
    assert result["status"] == "TRR-P08_SCORED_AFTER_JOINT_FREEZE"
    assert set(result["cells"]) == {
        "pile/public_base",
        "pile/public_lora_2601",
        "finance/public_base",
        "finance/public_lora_2601",
    }
    assert result["cells"]["pile/public_base"]["records"] == 4
    assert result["cells"]["pile/public_base"]["scores"][METHODS[0]]["6106"]["metrics"]["scored_tokens"] == 4 * 127
    assert result["bootstrap"]["draws"] == 32
    assert result["bootstrap"]["cells"]["pile/public_base"]["schedule_sha256"] == result["bootstrap"]["cells"]["pile/public_lora_2601"]["schedule_sha256"]
    assert result["truth_payload_persisted"] is False


def test_score_arrays_rejects_missing_arm_before_scoring() -> None:
    cells = _cells()
    del cells["pile/public_base"]["predictions"][METHODS[-1]]
    with pytest.raises(score_frozen.P08ScoreError, match="method matrix is incomplete"):
        score_frozen.score_arrays(cells, bootstrap_draws=8)


def test_score_arrays_rejects_target_source_order_change() -> None:
    cells = _cells()
    cell = cells["finance/public_lora_2601"]
    cell["record_ids"] = list(reversed(cell["record_ids"]))
    with pytest.raises(score_frozen.P08ScoreError, match="source-record order"):
        score_frozen.score_arrays(cells, bootstrap_draws=8)
