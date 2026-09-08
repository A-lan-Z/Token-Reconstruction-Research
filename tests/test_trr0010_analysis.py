from __future__ import annotations

import pytest

from trr0010_analysis import (
    AnalysisError,
    PERCENTILE_CONVENTION,
    TRR0010_BENEFIT_ALPHA,
    TRR0010_HARM_ALPHA,
    bootstrap_ratio,
    bootstrap_token_delta,
    paired_exact_cp,
    paired_token_delta,
)


def test_paired_exact_cp_uses_route_specific_component_tail() -> None:
    candidate = [True, True, False, False]
    control = [False, True, False, True]
    benefit = paired_exact_cp(candidate, control, alpha=TRR0010_BENEFIT_ALPHA)
    harm = paired_exact_cp(candidate, control, alpha=TRR0010_HARM_ALPHA)

    from scipy.stats import beta

    benefit_component = TRR0010_BENEFIT_ALPHA / 2.0
    expected_benefit_lower = float(beta.ppf(benefit_component, 1, 4) - beta.ppf(1.0 - benefit_component, 2, 3))
    assert benefit["method"] == "paired_discordance_clopper_pearson"
    assert benefit["bootstrap_used"] is False
    assert benefit["gains"] == 1 and benefit["losses"] == 1 and benefit["ties"] == 2
    assert benefit["component_alpha"] == pytest.approx(benefit_component)
    assert benefit["lower"] == pytest.approx(expected_benefit_lower)
    assert harm["component_alpha"] == pytest.approx(TRR0010_HARM_ALPHA / 2.0)
    assert harm["lower"] > benefit["lower"]


def test_paired_exact_cp_rejects_unpaired_inputs() -> None:
    with pytest.raises(AnalysisError, match="different record counts"):
        paired_exact_cp([True, False], [True])


def test_token_delta_zero_variance_is_unknown() -> None:
    result = bootstrap_token_delta([0.0, 0.0, 0.0, 0.0], draws=32)
    assert result["status"] == "UNKNOWN"
    assert result["reason"] == "zero_observed_difference_variance"
    assert "lower" not in result and "upper" not in result
    assert result["zero_variance_unknown"] is True


def test_paired_token_delta_is_source_record_paired_and_reproducible() -> None:
    kwargs = {"seed": 10010, "draws": 128, "one_sided_alpha": 0.025}
    first = paired_token_delta(
        [90, 80, 100, 70],
        [80, 80, 90, 70],
        [100, 100, 100, 100],
        **kwargs,
    )
    second = paired_token_delta(
        [90, 80, 100, 70],
        [80, 80, 90, 70],
        [100, 100, 100, 100],
        **kwargs,
    )
    assert first == second
    assert first["status"] == "COMPUTED"
    assert first["unit"] == "source_record"
    assert first["percentile_convention"] == PERCENTILE_CONVENTION
    assert first["point"] == pytest.approx(0.05)
    assert first["lower"] <= first["point"] <= first["upper"]


def test_ratio_interval_computes_only_when_every_draw_has_positive_denominator() -> None:
    result = bootstrap_ratio([2, 4, 1, 3], [10, 10, 10, 10], draws=128)
    assert result["status"] == "COMPUTED"
    assert result["point"] == pytest.approx(0.25)
    assert result["nonpositive_bootstrap_draws"] == 0
    assert result["lower"] <= result["point"] <= result["upper"]


def test_ratio_nonpositive_bootstrap_draw_fails_closed_without_dropping_draw() -> None:
    # The observed denominator is positive, but repeated source-record draws
    # can select the negative contribution twice.  The whole route is UNKNOWN.
    result = bootstrap_ratio([0, 1], [-1, 3], seed=10010, draws=128)
    assert result["status"] == "UNKNOWN"
    assert result["reason"] == "nonpositive_bootstrap_denominator"
    assert result["nonpositive_bootstrap_draws"] > 0
    assert "lower" not in result and "upper" not in result


def test_ratio_observed_nonpositive_denominator_is_unknown() -> None:
    result = bootstrap_ratio([1, 2], [0, 0], draws=16)
    assert result["status"] == "UNKNOWN"
    assert result["reason"] == "nonpositive_observed_denominator"


def test_ratio_and_token_inputs_validate_record_geometry() -> None:
    with pytest.raises(AnalysisError, match="different record counts"):
        paired_token_delta([1, 2], [1], [2, 2])
    with pytest.raises(AnalysisError, match="record counts differ"):
        bootstrap_ratio([1, 2], [1])
    with pytest.raises(AnalysisError, match="exceed exposure"):
        paired_token_delta([3], [1], [2])


@pytest.mark.parametrize("bad_outcome", ["False", None, float("nan"), 2, -1, 1.0])
def test_exact_outcomes_reject_silent_coercion(bad_outcome: object) -> None:
    with pytest.raises(AnalysisError, match="boolean or binary integer"):
        paired_exact_cp([bad_outcome], [False])


def test_exact_outcomes_accept_binary_numpy_scalars() -> None:
    np = pytest.importorskip("numpy")
    result = paired_exact_cp([np.bool_(True), np.int64(0)], [np.int64(0), np.bool_(False)])
    assert result["gains"] == 1
    assert result["losses"] == 0


def test_fixed_width_token_delta_rejects_variable_exposure() -> None:
    with pytest.raises(AnalysisError, match="equal exposures.*bootstrap_ratio"):
        paired_token_delta([8, 9], [7, 8], [10, 20])
