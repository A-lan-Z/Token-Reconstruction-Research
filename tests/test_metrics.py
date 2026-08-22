"""Unit tests for preregistered TRR-0001 metric primitives."""

from __future__ import annotations

import pytest

from token_reconstruction.metrics import (
    bootstrap_mean,
    correct_prefix_length,
    frequency_bin,
    record_metrics,
    token_group,
)


def test_record_metrics_cover_prefix_candidates_and_no_abstention() -> None:
    truth = list(range(39))
    prediction = truth.copy()
    prediction[3] = 999
    candidates = [[value, 500] for value in truth]
    value = record_metrics(prediction, truth, candidates)
    assert value["correct_tokens"] == 38
    assert value["correct_prefix_length"] == 3
    assert value["first_error_position"] == 4
    assert value["coverage"] == 1.0
    assert value["top16_recall"] == 1.0
    assert value["conditional_selection_accuracy"] == pytest.approx(38 / 39)


def test_record_metric_geometry_and_bootstrap_are_fail_closed() -> None:
    with pytest.raises(ValueError):
        correct_prefix_length([1], [1, 2])
    with pytest.raises(ValueError):
        record_metrics([1], [1], [[1]])
    first = bootstrap_mean([0.0, 1.0, 1.0], draws=100, seed=1732)
    second = bootstrap_mean([0.0, 1.0, 1.0], draws=100, seed=1732)
    assert first == second
    assert first["estimate"] == pytest.approx(2 / 3)


def test_frequency_and_token_groups_follow_preregistered_bins() -> None:
    assert [frequency_bin(value) for value in (0, 1, 4, 5, 19, 20)] == [
        "unseen",
        "1-4",
        "1-4",
        "5-19",
        "5-19",
        "20-or-more",
    ]
    assert token_group("123", "123") == "numeric"
    assert token_group(",", ",") == "punctuation"
    assert token_group("Ġword", " word") == "whitespace_prefixed"
    assert token_group("word", "word") == "other"
