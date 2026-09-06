"""Focused tests for the preregistered TRR-P03 paired statistics."""

from __future__ import annotations

import pytest

from token_reconstruction.trr_p03.statistics import (
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    StatisticsError,
    paired_record_statistics,
)


def _row(
    record_id: str,
    length: int,
    projected_correct: int,
    a1_correct: int,
    *,
    projected_correctness: list[bool] | None = None,
    a1_correctness: list[bool] | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "length": length,
        "projected_correct": projected_correct,
        "a1_correct": a1_correct,
        "projected_exact": projected_correct == length,
        "a1_exact": a1_correct == length,
        **(
            {
                "projected_correctness": projected_correctness,
                "a1_correctness": a1_correctness,
            }
            if projected_correctness is not None or a1_correctness is not None
            else {}
        ),
    }


def _varied_panel() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for length in (16, 39, 64, 128):
        for index in range(6):
            a1 = length // 2 + index
            projected = a1 + ((index % 3) - 1)
            rows.append(_row(f"r-{length}-{index}", length, projected, a1))
    return rows


def test_unequal_lengths_keep_micro_and_macro_estimands_distinct() -> None:
    rows = [
        _row(f"short-{index}", 16, 16, 0) for index in range(6)
    ] + [
        _row(f"long-{index}", 128, 64, 0) for index in range(6)
    ]

    stats = paired_record_statistics(rows, draws=1_000, seed=17)

    # Micro weighting gives the 16-token records less influence than the
    # 128-token records, whereas macro weighting gives both strata equal
    # record weight: 480 / 864 versus (1 + .5) / 2.
    assert stats["token"]["delta"] == pytest.approx(480 / 864)
    assert stats["macro"]["delta"] == pytest.approx(0.75)
    assert stats["token"]["delta"] != pytest.approx(stats["macro"]["delta"])
    assert stats["strata"] == {"16": 6, "128": 6}
    assert stats["gate_ready"]["projected_exact_records"] == 6
    assert stats["gate_ready"]["a1_exact_records"] == 0


def test_length_stratified_bootstrap_is_seeded_and_paired() -> None:
    rows = _varied_panel()
    first = paired_record_statistics(rows)
    second = paired_record_statistics(list(reversed(rows)))

    # Sorting by (length, record_id) makes the CI reproducible even if the
    # caller serializes the same panel in another order.
    assert first == second
    assert first["bootstrap_config"] == {
        "draws": DEFAULT_BOOTSTRAP_DRAWS,
        "seed": DEFAULT_BOOTSTRAP_SEED,
        "unit": "paired_record_cluster",
        "stratified_by": "length",
    }
    for family in ("token", "macro", "exact_record"):
        bootstrap = first[family]["bootstrap"]
        assert bootstrap["draws"] == DEFAULT_BOOTSTRAP_DRAWS
        assert bootstrap["seed"] == DEFAULT_BOOTSTRAP_SEED
        assert bootstrap["unit"] == "paired_record_cluster"
        assert bootstrap["stratified_by"] == "length"
        assert bootstrap["strata"] == {"16": 6, "39": 6, "64": 6, "128": 6}
        assert bootstrap["ci95_percentile"][0] <= bootstrap["estimate"]
        assert bootstrap["estimate"] <= bootstrap["ci95_percentile"][1]


def test_count_changes_and_position_changes_are_reported_separately() -> None:
    rows: list[dict[str, object]] = []
    for length in (4, 8):
        for index in range(6):
            # Same count but different positions: one position-level gain and
            # one position-level loss per record.
            projected_vector = [True, False, True, False] + [False] * (length - 4)
            a1_vector = [True, True, False, False] + [False] * (length - 4)
            rows.append(
                _row(
                    f"v-{length}-{index}",
                    length,
                    2,
                    2,
                    projected_correctness=projected_vector,
                    a1_correctness=a1_vector,
                )
            )

    stats = paired_record_statistics(rows, draws=100, seed=23)

    assert stats["token"]["correct_count_changes"] == {
        "gain_records": 0,
        "tie_records": 12,
        "loss_records": 0,
        "gain_tokens": 0,
        "loss_tokens": 0,
        "net_tokens": 0,
    }
    assert stats["token_position_changes"] == {
        "available": True,
        "gain_tokens": 12,
        "tie_tokens": 48,
        "loss_tokens": 12,
        "regression_tokens": 12,
        "both_correct_tokens": 12,
        "both_wrong_tokens": 36,
        "net_tokens": 0,
    }


def test_invalid_geometry_and_stratum_size_fail_closed() -> None:
    with pytest.raises(StatisticsError, match="expected 6"):
        paired_record_statistics([_row("one", 16, 1, 1)])
    with pytest.raises(StatisticsError, match="disagrees"):
        paired_record_statistics(
            [
                {
                    **_row("one", 16, 1, 1),
                    "projected_exact": True,
                }
            ],
            records_per_stratum=1,
        )
