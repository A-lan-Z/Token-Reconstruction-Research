"""Focused tests for the TRR-0008 prospective planning calculations."""

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from scipy.stats import binom

from scripts import trr0008_plan as plan


def test_clopper_pearson_bounds_have_expected_order_and_endpoints():
    for n in (1, 8, 32, 128):
        lower, upper, engine = plan._cp_arrays(n, alpha=0.025)
        assert engine == "scipy.stats.beta.ppf_clopper_pearson"
        empirical = np.arange(n + 1, dtype=float) / n
        assert upper[0] > 0.0
        assert lower[-1] < 1.0
        assert np.all(lower <= empirical)
        assert np.all(empirical <= upper)
        assert np.all(lower <= upper)


def test_conditional_power_matches_exhaustive_small_multinomial():
    n = 8
    q = 0.15
    effect = 0.08
    margin = 0.05
    alpha = 0.025
    calculated, _ = plan._exact_multinomial_power(
        n=n, true_effect=effect, margin=margin, discordance_rate=q, alpha=alpha
    )
    lower, upper, _ = plan._cp_arrays(n, alpha=alpha)
    p_gain = (q + effect) / 2.0
    p_loss = (q - effect) / 2.0
    p_zero = 1.0 - p_gain - p_loss
    expected = 0.0
    for gains in range(n + 1):
        for losses in range(n - gains + 1):
            probability = (
                math.factorial(n)
                / (
                    math.factorial(gains)
                    * math.factorial(losses)
                    * math.factorial(n - gains - losses)
                )
                * p_gain**gains
                * p_loss**losses
                * p_zero ** (n - gains - losses)
            )
            if lower[gains] - upper[losses] >= margin:
                expected += probability
    assert calculated == pytest.approx(expected, abs=1e-12)


def test_null_positive_power_is_conservative():
    for n in (8, 32, 128):
        power, _ = plan._exact_multinomial_power(
            n=n, true_effect=0.0, margin=0.0, discordance_rate=0.15, alpha=0.025
        )
        assert power <= 0.025 + 1e-12


def test_p06_loader_is_allowlisted_and_opaque():
    summary, source, sequence = plan._load_p06_opaque()
    assert summary["file"]["sha256"] == plan.P06_OPAQUE_SHA256
    assert summary["file"]["bytes"] == plan.P06_OPAQUE_BYTES
    assert len(source) == 512
    assert len(sequence) == 512
    assert summary["privacy"]["suitable_for_identity_exclusion_only"] is True
    assert summary["underlying_provenance_opened"] is False
