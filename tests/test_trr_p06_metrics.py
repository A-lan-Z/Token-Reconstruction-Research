from __future__ import annotations

import numpy as np
import pytest

from token_reconstruction.trr_p06_metrics import (
    CONTRASTS,
    P06MetricsError,
    paired_cluster_bootstrap,
    paired_metrics,
    score_method,
)


def _unequal_length_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """Return five records with one full clip and four unequal short clips."""

    records = 5
    lengths = (1, 3, 7, 16, 127)  # post-BOS scored positions
    record_ids = tuple(f"source-{index}" for index in range(records))
    truth = np.zeros((records, 128), dtype=np.int64)
    truth[:, 0] = 128000
    for row in range(records):
        truth[row, 1:] = 1000 + np.arange(127)
    mask = np.zeros((records, 128), dtype=bool)
    for row, length in enumerate(lengths):
        mask[row, : length + 1] = True

    left = truth.copy()
    right = truth.copy()
    # One gain, one loss, and a larger pair of asymmetric records make the
    # micro token delta and the unweighted record mean visibly different.
    right[0, 1] += 1                 # left gain
    left[1, 1] += 1                  # right gain / left loss
    right[2, 1:8] += 1               # seven left gains
    left[3, 1:17] += 1               # sixteen right gains
    right[4, 1:21] += 1              # full-row left exact, right non-exact
    return left, right, truth, record_ids, mask


def test_unequal_length_pair_metrics_keep_micro_denominator_and_exact_subset() -> None:
    left, right, truth, record_ids, mask = _unequal_length_fixture()
    left_score = score_method(
        left,
        truth,
        record_ids=record_ids,
        attention_mask=mask,
        position_ids=np.tile(np.arange(128), (len(record_ids), 1)),
        method_id="p06_full_record",
    )
    right_score = score_method(
        right,
        truth,
        record_ids=record_ids,
        attention_mask=mask,
        position_ids=np.tile(np.arange(128), (len(record_ids), 1)),
        method_id="p06_past_only",
    )
    comparison = paired_metrics(
        left,
        right,
        truth,
        left_method="p06_full_record",
        right_method="p06_past_only",
        record_ids=record_ids,
        attention_mask=mask,
        contrast_id="full_minus_past",
    )

    left_metrics = left_score["metrics"]
    assert left_metrics["scored_tokens"] == 154
    assert left_metrics["correct_tokens"] == 137
    assert left_metrics["exact_records"] == 1
    assert left_metrics["exact_denominator"] == 1
    assert left_metrics["token_accuracy"] == pytest.approx(137 / 154)
    record_mean = np.mean(
        [row["token_accuracy"] for row in left_score["per_record"] if row["token_accuracy"] is not None]
    )
    assert left_metrics["token_accuracy"] != pytest.approx(record_mean)
    assert left_metrics["macro_records"] == 5
    assert left_metrics["macro_token_accuracy"] == pytest.approx(record_mean)

    metrics = comparison["metrics"]
    assert metrics["scored_tokens"] == 154
    assert metrics["token_gains"] == 28
    assert metrics["token_losses"] == 17
    assert metrics["token_delta_pp"] == pytest.approx(100 * 11 / 154)
    expected_macro_delta = 100 * np.mean([1 / 1, -1 / 3, 7 / 7, -16 / 16, 20 / 127])
    assert metrics["macro_token_delta_pp"] == pytest.approx(expected_macro_delta)
    assert metrics["exact_denominator"] == 1
    assert metrics["left_exact_records"] == 1
    assert metrics["right_exact_records"] == 0
    assert metrics["exact_delta_pp"] == pytest.approx(100.0)
    assert comparison["position_metrics"]["early"]["scored_tokens"] == 41
    assert comparison["per_record"][1]["token_delta"] == -1


def _full_clip_score(
    method_id: str,
    record_ids: tuple[str, ...],
    truth: np.ndarray,
    wrong_spans: dict[int, tuple[int, int]],
) -> dict[str, object]:
    predictions = truth.copy()
    for row, (start, stop) in wrong_spans.items():
        predictions[row, start:stop] += 1
    mask = np.ones_like(truth, dtype=bool)
    return score_method(
        predictions,
        truth,
        record_ids=record_ids,
        attention_mask=mask,
        method_id=method_id,
    )


def _bootstrap_cells(*, mismatched_target_order: bool = False) -> dict[str, dict[str, object]]:
    records = 4
    record_ids = tuple(f"source-{index}" for index in range(records))
    truth = np.zeros((records, 128), dtype=np.int64)
    truth[:, 0] = 128000
    truth[:, 1:] = 2000 + np.arange(127)
    methods = {
        "p06_positionwise_diagonal": {0: (1, 4)},
        "p06_past_only": {1: (1, 7)},
        "p06_full_record": {2: (1, 2)},
    }
    target_cells: dict[str, dict[str, object]] = {}
    for target, target_methods in (
        ("public_base", methods),
        (
            "public_lora_2601",
            {
                "p06_positionwise_diagonal": {0: (1, 4)},
                "p06_past_only": {1: (1, 2)},
                "p06_full_record": {2: (1, 2)},
            },
        ),
    ):
        ids = tuple(reversed(record_ids)) if (target == "public_lora_2601" and mismatched_target_order) else record_ids
        target_cells[target] = {
            "domain": "pile",
            "target": target,
            "methods": {
                method: _full_clip_score(method, ids, truth, spans)
                for method, spans in target_methods.items()
            },
        }
    return {
        "pile__public_base": target_cells["public_base"],
        "pile__public_lora_2601": target_cells["public_lora_2601"],
    }


def test_cluster_bootstrap_is_seeded_and_reuses_paired_domain_schedule() -> None:
    cells = _bootstrap_cells()
    first = paired_cluster_bootstrap(cells, draws=256, seed=6306)
    second = paired_cluster_bootstrap(cells, draws=256, seed=6306)
    assert first == second
    assert first["draws"] == 256
    assert first["seed"] == 6306
    assert first["unit"] == "source-record cluster"
    domain = first["domains"]["pile"]
    assert domain["schedule_shared_across_targets"] is True
    assert domain["schedule_shape"] == [256, 4]
    assert domain["target_conditions"] == ["public_base", "public_lora_2601"]
    for target in domain["target_conditions"]:
        contrasts = domain["targets"][target]["contrasts"]
        assert set(contrasts) == set(CONTRASTS)
        assert contrasts["full_minus_past"]["records"] == 4
        assert contrasts["full_minus_past"]["token_delta_ci95_percentile_pp"]
        assert contrasts["full_minus_past"]["macro_token_delta_ci95_percentile_pp"]
    assert (
        domain["targets"]["public_base"]["contrasts"]["full_minus_past"]["point"]["token_delta_pp"]
        != domain["targets"]["public_lora_2601"]["contrasts"]["full_minus_past"]["point"]["token_delta_pp"]
    )


def test_cluster_bootstrap_rejects_changed_source_record_order() -> None:
    with pytest.raises(P06MetricsError, match="source-record order"):
        paired_cluster_bootstrap(_bootstrap_cells(mismatched_target_order=True), draws=32, seed=6306)
