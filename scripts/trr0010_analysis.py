"""Source-paired TRR-0010 analysis primitives.

This module contains only numerical summaries over already materialized,
source-paired records.  It never opens source data, model state, predictions,
or truth.  The constants mirror the committed TRR-0010 shared contract:
source-record bootstrap seed 10010, 10,000 draws, benefit alpha 0.025, and
harm alpha 0.05.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
import numbers
import random
from typing import Any

TRR0010_BOOTSTRAP_SEED = 10010
TRR0010_BOOTSTRAP_DRAWS = 10_000
TRR0010_BENEFIT_ALPHA = 0.025
TRR0010_HARM_ALPHA = 0.05
PERCENTILE_CONVENTION = (
    "ascending order statistics: lower index=floor(alpha * draws); "
    "upper index=ceil((1-alpha) * draws)-1"
)


class AnalysisError(ValueError):
    """Raised when source-paired analysis inputs violate the contract."""


def _finite_values(values: Sequence[float], *, label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise AnalysisError(f"{label} must contain at least one source record")
    if any(not math.isfinite(value) for value in result):
        raise AnalysisError(f"{label} contains a non-finite value")
    return result


def _paired_bools(values: Sequence[bool], *, label: str) -> list[bool]:
    """Accept only booleans or binary integer scalars, without coercion."""

    result: list[bool] = []
    for value in values:
        scalar: object = value
        if not isinstance(scalar, (bool, numbers.Integral)):
            # NumPy bool_ exposes item() as a Python bool but is not an
            # Integral subclass.  Do not import NumPy just for this boundary.
            item = getattr(scalar, "item", None)
            if callable(item):
                scalar = item()
        if isinstance(scalar, bool):
            bit = int(scalar)
        elif isinstance(scalar, numbers.Integral):
            bit = int(scalar)
        else:
            raise AnalysisError(f"{label} must contain only boolean or binary integer outcomes")
        if bit not in (0, 1):
            raise AnalysisError(f"{label} must contain only boolean or binary integer outcomes")
        result.append(bool(bit))
    if not result:
        raise AnalysisError(f"{label} must contain at least one source record")
    return result


def _validate_alpha(alpha: float, *, label: str = "alpha") -> float:
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise AnalysisError(f"{label} must lie strictly between zero and one")
    return value


def _validate_bootstrap(*, seed: int, draws: int, one_sided_alpha: float) -> float:
    if int(draws) <= 0:
        raise AnalysisError("bootstrap draws must be positive")
    return _validate_alpha(one_sided_alpha, label="one-sided alpha")


def _cp_lower(successes: int, trials: int, component_alpha: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        raise AnalysisError("invalid CP successes/trials")
    if successes == 0:
        return 0.0
    try:
        from scipy.stats import beta
    except ImportError as exc:  # pragma: no cover - environment contract
        raise AnalysisError("scipy is required for exact CP bounds") from exc
    return float(beta.ppf(component_alpha, successes, trials - successes + 1))


def _cp_upper(successes: int, trials: int, component_alpha: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        raise AnalysisError("invalid CP successes/trials")
    if successes == trials:
        return 1.0
    try:
        from scipy.stats import beta
    except ImportError as exc:  # pragma: no cover - environment contract
        raise AnalysisError("scipy is required for exact CP bounds") from exc
    return float(beta.ppf(1.0 - component_alpha, successes + 1, trials - successes))


def paired_exact_cp(
    candidate_exact: Sequence[bool],
    control_exact: Sequence[bool],
    *,
    alpha: float = TRR0010_BENEFIT_ALPHA,
) -> dict[str, Any]:
    """Return paired-discordance CP bounds for candidate-minus-control exactness.

    ``alpha`` is the composite one-sided route alpha.  Its component tail is
    ``alpha / 2`` for the gain and loss CP bounds, so the net lower bound is
    L_CP(gains) - U_CP(losses), as required by the shared contract.
    """

    left = _paired_bools(candidate_exact, label="candidate exact outcomes")
    right = _paired_bools(control_exact, label="control exact outcomes")
    if len(left) != len(right):
        raise AnalysisError("paired exact outcomes have different record counts")
    route_alpha = _validate_alpha(alpha)
    component_alpha = route_alpha / 2.0
    gains = sum(candidate and not control for candidate, control in zip(left, right))
    losses = sum(control and not candidate for candidate, control in zip(left, right))
    records = len(left)
    gain_lower = _cp_lower(gains, records, component_alpha)
    gain_upper = _cp_upper(gains, records, component_alpha)
    loss_lower = _cp_lower(losses, records, component_alpha)
    loss_upper = _cp_upper(losses, records, component_alpha)
    return {
        "status": "COMPUTED",
        "method": "paired_discordance_clopper_pearson",
        "unit": "source_record_exact_outcome",
        "records": records,
        "gains": gains,
        "losses": losses,
        "ties": records - gains - losses,
        "point": float(gains - losses) / float(records),
        "lower": float(gain_lower - loss_upper),
        "upper": float(gain_upper - loss_lower),
        "alpha": route_alpha,
        "component_alpha": component_alpha,
        "bootstrap_used": False,
    }


def _percentile_interval(values: Sequence[float], *, one_sided_alpha: float) -> tuple[float, float]:
    ordered = sorted(float(value) for value in values)
    lower_index = max(0, min(len(ordered) - 1, int(math.floor(one_sided_alpha * len(ordered)))))
    upper_index = max(
        0,
        min(len(ordered) - 1, int(math.ceil((1.0 - one_sided_alpha) * len(ordered)) - 1)),
    )
    return float(ordered[lower_index]), float(ordered[upper_index])


def _bootstrap_indices(*, records: int, seed: int, draws: int):
    rng = random.Random(int(seed))
    for _ in range(int(draws)):
        yield [rng.randrange(records) for _ in range(records)]


def bootstrap_token_delta(
    record_deltas: Sequence[float],
    *,
    seed: int = TRR0010_BOOTSTRAP_SEED,
    draws: int = TRR0010_BOOTSTRAP_DRAWS,
    one_sided_alpha: float = TRR0010_BENEFIT_ALPHA,
) -> dict[str, Any]:
    """Bootstrap source-record token deltas with fail-closed zero variance.

    A zero observed-difference variance is UNKNOWN for the route or safeguard;
    a degenerate interval is never treated as evidence of no harm.
    """

    values = _finite_values(record_deltas, label="record token deltas")
    tail = _validate_bootstrap(seed=seed, draws=draws, one_sided_alpha=one_sided_alpha)
    point = float(sum(values) / len(values))
    result: dict[str, Any] = {
        "status": "UNKNOWN" if len(set(values)) == 1 else "COMPUTED",
        "unit": "source_record",
        "records": len(values),
        "point": point,
        "draws": int(draws),
        "seed": int(seed),
        "one_sided_alpha": tail,
        "percentile_convention": PERCENTILE_CONVENTION,
        "zero_variance_unknown": len(set(values)) == 1,
    }
    if result["status"] == "UNKNOWN":
        result["reason"] = "zero_observed_difference_variance"
        return result
    estimates = [
        float(sum(values[index] for index in indices) / len(values))
        for indices in _bootstrap_indices(records=len(values), seed=seed, draws=draws)
    ]
    result["lower"], result["upper"] = _percentile_interval(estimates, one_sided_alpha=tail)
    return result


def paired_token_delta(
    candidate_correct: Sequence[int | float],
    control_correct: Sequence[int | float],
    exposed_tokens: Sequence[int | float] | None = None,
    *,
    seed: int = TRR0010_BOOTSTRAP_SEED,
    draws: int = TRR0010_BOOTSTRAP_DRAWS,
    one_sided_alpha: float = TRR0010_BENEFIT_ALPHA,
) -> dict[str, Any]:
    """Compute per-source token-accuracy deltas and bootstrap them.

    This is appropriate for fixed-width full-panel cells.  Variable-exposure
    strata should use :func:`bootstrap_ratio` so each draw sums token counts
    and exposure counts rather than averaging record accuracies.
    """

    candidate = _finite_values(candidate_correct, label="candidate correct counts")
    control = _finite_values(control_correct, label="control correct counts")
    if len(candidate) != len(control):
        raise AnalysisError("paired token counts have different record counts")
    if exposed_tokens is None:
        exposed = [1.0] * len(candidate)
    else:
        exposed = _finite_values(exposed_tokens, label="exposed token counts")
        if len(exposed) != len(candidate):
            raise AnalysisError("exposure counts have different record counts")
    if any(exposure <= 0.0 for exposure in exposed):
        raise AnalysisError("each source record must expose positive token count")
    if len(set(exposed)) != 1:
        raise AnalysisError(
            "paired_token_delta requires equal exposures; use bootstrap_ratio for variable exposure strata"
        )
    if any(value < 0.0 or value > exposure for value, exposure in zip(candidate, exposed)):
        raise AnalysisError("candidate correct counts exceed exposure")
    if any(value < 0.0 or value > exposure for value, exposure in zip(control, exposed)):
        raise AnalysisError("control correct counts exceed exposure")
    result = bootstrap_token_delta(
        [(left - right) / exposure_count for left, right, exposure_count in zip(candidate, control, exposed)],
        seed=seed,
        draws=draws,
        one_sided_alpha=one_sided_alpha,
    )
    result.update({
        "candidate_correct": float(sum(candidate)),
        "control_correct": float(sum(control)),
        "exposed_tokens": float(sum(exposed)),
    })
    return result


def bootstrap_ratio(
    numerators: Sequence[int | float],
    denominators: Sequence[int | float],
    *,
    seed: int = TRR0010_BOOTSTRAP_SEED,
    draws: int = TRR0010_BOOTSTRAP_DRAWS,
    one_sided_alpha: float = TRR0010_BENEFIT_ALPHA,
) -> dict[str, Any]:
    """Bootstrap a source-record ratio with whole-route denominator failure.

    Every bootstrap draw must have a strictly positive denominator.  If any
    draw has a nonpositive denominator, the complete ratio interval is
    UNKNOWN; no draw is dropped and no nonpositive denominator is treated as
    zero.  This covers remaining-error reduction and A1+A2 gap closure.
    This generic ratio primitive is not itself a weighted rare-token harm
    gate; such a wrapper must also apply the token zero-variance UNKNOWN rule.
    """

    numerator = _finite_values(numerators, label="ratio numerators")
    denominator = _finite_values(denominators, label="ratio denominators")
    if len(numerator) != len(denominator):
        raise AnalysisError("ratio numerator/denominator record counts differ")
    tail = _validate_bootstrap(seed=seed, draws=draws, one_sided_alpha=one_sided_alpha)
    records = len(numerator)
    observed_denominator = float(sum(denominator))
    base: dict[str, Any] = {
        "unit": "source_record",
        "records": records,
        "draws": int(draws),
        "seed": int(seed),
        "one_sided_alpha": tail,
        "percentile_convention": PERCENTILE_CONVENTION,
        "observed_numerator": float(sum(numerator)),
        "observed_denominator": observed_denominator,
    }
    if observed_denominator <= 0.0:
        return {
            **base,
            "status": "UNKNOWN",
            "reason": "nonpositive_observed_denominator",
        }
    point = float(sum(numerator) / observed_denominator)
    estimates: list[float] = []
    nonpositive_draws = 0
    for indices in _bootstrap_indices(records=records, seed=seed, draws=draws):
        sampled_denominator = sum(denominator[index] for index in indices)
        if sampled_denominator <= 0.0:
            nonpositive_draws += 1
            continue
        sampled_numerator = sum(numerator[index] for index in indices)
        estimates.append(float(sampled_numerator / sampled_denominator))
    if nonpositive_draws:
        return {
            **base,
            "status": "UNKNOWN",
            "reason": "nonpositive_bootstrap_denominator",
            "nonpositive_bootstrap_draws": nonpositive_draws,
            "point": point,
        }
    if len(estimates) != int(draws):  # defensive; every draw must be retained
        raise AnalysisError("ratio bootstrap draw accounting changed")
    lower, upper = _percentile_interval(estimates, one_sided_alpha=tail)
    return {
        **base,
        "status": "COMPUTED",
        "point": point,
        "lower": lower,
        "upper": upper,
        "nonpositive_bootstrap_draws": 0,
    }


__all__ = [
    "AnalysisError",
    "TRR0010_BOOTSTRAP_SEED",
    "TRR0010_BOOTSTRAP_DRAWS",
    "TRR0010_BENEFIT_ALPHA",
    "TRR0010_HARM_ALPHA",
    "PERCENTILE_CONVENTION",
    "paired_exact_cp",
    "bootstrap_token_delta",
    "paired_token_delta",
    "bootstrap_ratio",
]
