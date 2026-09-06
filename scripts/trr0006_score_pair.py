#!/usr/bin/env python3
"""Task-local metrics and report extraction for the TRR-0006 pair.

The scorer has one contrast only: enriched causal attention versus the
enriched trained diagonal (positionwise) control.  It is intentionally
independent of the TRR-0005 eight-method scorer.  Its public functions accept
predictions and truth only after a caller has completed the public gate from
``trr0006_freeze_pair``; ``score_with_truth_loader`` makes that ordering
executable in tests and in the eventual runner.

Exact-record uncertainty uses the registered paired discordances and
one-sided Clopper-Pearson marginal bounds.  Token uncertainty uses a paired
source-record bootstrap with the same deterministic schedule for both target
conditions within a domain.  Bootstrap endpoints are explicitly labelled
approximate; point estimates and descriptive intervals do not establish a
practical claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any

import numpy as np

try:  # Package import used by the executable driver.
    from scripts.trr0006_freeze_pair import CELL_ORDER, METHOD_ORDER
except ModuleNotFoundError:  # Direct script/import fallback.
    from trr0006_freeze_pair import CELL_ORDER, METHOD_ORDER


TASK_ID = "TRR-0006"
SCHEMA = "token-reconstruction.trr0006-pair-score.v1"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = -1
VOCAB_SIZE = 128256
SEQUENCE_TOKENS = 128
POST_BOS_TOKENS = 127
DEFAULT_BOOTSTRAP_DRAWS = 10000
DEFAULT_BOOTSTRAP_SEED = 5005
TOKEN_TAIL_ALPHA = 0.05 / 16
EXACT_TAIL_ALPHA = 0.05 / 32
TOKEN_MARGIN_PP = 0.5
EXACT_MARGIN_PP = 5.0
TOKEN_HARM_MARGIN_PP = 0.5
EXACT_HARM_MARGIN_PP = 5.0


class PairScoreError(ValueError):
    """Raised when a post-gate TRR6 score input is incomplete or malformed."""


def _array(value: Any, *, name: str, dtype: Any | None = None) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result = np.asarray(value)
    except Exception as exc:  # pragma: no cover - backend-specific conversion errors
        raise PairScoreError(f"{name} is not array-like") from exc
    if dtype is not None:
        try:
            result = result.astype(dtype, copy=False)
        except (TypeError, ValueError) as exc:
            raise PairScoreError(f"{name} has an incompatible dtype") from exc
    return np.ascontiguousarray(result)


def _validate_record_ids(record_ids: Sequence[str], records: int, *, name: str) -> tuple[str, ...]:
    if len(record_ids) != records:
        raise PairScoreError(f"{name} record ID count changed")
    values = tuple(record_ids)
    if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != records:
        raise PairScoreError(f"{name} record IDs are not unique nonempty strings")
    return values


def _right_padded_mask(mask: np.ndarray, *, name: str) -> None:
    if mask.ndim != 2 or mask.shape[1] != SEQUENCE_TOKENS:
        raise PairScoreError(f"{name} mask geometry changed")
    if not np.issubdtype(mask.dtype, np.bool_) and not np.isin(mask, [0, 1]).all():
        raise PairScoreError(f"{name} mask is not binary")
    mask = mask.astype(bool, copy=False)
    if not mask[:, 0].all() or (mask[:, 1:] > mask[:, :-1]).any():
        raise PairScoreError(f"{name} mask is not BOS/right-padded")
    # TRR6 declares a 128-token clip and exactly 127 post-BOS scores.  A
    # shorter padded row is a different estimand and therefore fails closed.
    if not mask.all():
        raise PairScoreError(f"{name} clip is not exactly 127 post-BOS tokens")


def _validate_positions(position_ids: Any | None, mask: np.ndarray, *, name: str) -> None:
    if position_ids is None:
        return
    positions = _array(position_ids, name=f"{name} positions", dtype=np.int64)
    expected = np.arange(SEQUENCE_TOKENS, dtype=np.int64)[None, :]
    expected = np.broadcast_to(expected, mask.shape)
    if positions.shape != mask.shape or not np.array_equal(positions, expected):
        raise PairScoreError(f"{name} positions disagree with the 128-token clip")


def _validate_ids(ids: np.ndarray, mask: np.ndarray, *, name: str, require_bos: bool) -> None:
    if ids.ndim != 2 or ids.shape != mask.shape:
        raise PairScoreError(f"{name} geometry changed")
    if not np.issubdtype(ids.dtype, np.integer):
        raise PairScoreError(f"{name} IDs are not integer")
    if require_bos and not np.all(ids[:, 0] == BOS_TOKEN_ID):
        raise PairScoreError(f"{name} BOS changed")
    active = mask.astype(bool, copy=False)
    if np.any(ids[active] < 0) or np.any(ids[active] >= VOCAB_SIZE):
        raise PairScoreError(f"{name} active ID is outside the vocabulary")
    if np.any(ids[~active] != PAD_TOKEN_ID):
        raise PairScoreError(f"{name} padding ID is not -1")


def score_cell(
    *,
    predictions: Any,
    truth: Any,
    attention_mask: Any,
    record_ids: Sequence[str],
    method_id: str,
    position_ids: Any | None = None,
) -> dict[str, Any]:
    """Score one method on one post-gate four-cell input."""

    if method_id not in METHOD_ORDER:
        raise PairScoreError(f"unknown TRR6 method: {method_id}")
    mask = _array(attention_mask, name="attention_mask")
    if mask.ndim != 2 or mask.shape[1] != SEQUENCE_TOKENS:
        raise PairScoreError("attention mask geometry changed")
    records = mask.shape[0]
    _right_padded_mask(mask, name="attention_mask")
    ids = _array(predictions, name=f"{method_id} predictions")
    labels = _array(truth, name="truth", dtype=np.int64)
    _validate_ids(ids, mask, name=f"{method_id} predictions", require_bos=True)
    if labels.shape != mask.shape:
        raise PairScoreError("truth geometry changed")
    if not np.issubdtype(labels.dtype, np.integer):
        raise PairScoreError("truth IDs are not integer")
    _validate_ids(labels, mask, name="truth", require_bos=True)
    _validate_positions(position_ids, mask, name="attention_mask")
    ids_tuple = _validate_record_ids(record_ids, records, name="cell")
    scored = mask.astype(bool, copy=True)
    scored[:, 0] = False
    correct = ids == labels
    per_record: list[dict[str, Any]] = []
    for index, record_id in enumerate(ids_tuple):
        correct_tokens = int(np.count_nonzero(correct[index, scored[index]]))
        scored_tokens = int(np.count_nonzero(scored[index]))
        per_record.append(
            {
                "record_id": record_id,
                "correct_tokens": correct_tokens,
                "scored_tokens": scored_tokens,
                "token_accuracy": correct_tokens / scored_tokens,
                "exact_record": bool(correct_tokens == POST_BOS_TOKENS),
            }
        )
    total_correct = int(sum(row["correct_tokens"] for row in per_record))
    total_scored = int(sum(row["scored_tokens"] for row in per_record))
    exact_records = int(sum(bool(row["exact_record"]) for row in per_record))
    return {
        "task_id": TASK_ID,
        "cell_id": None,
        "method_id": method_id,
        "records": records,
        "clip_tokens_including_bos": SEQUENCE_TOKENS,
        "scored_post_bos_tokens": POST_BOS_TOKENS,
        "metrics": {
            "scored_tokens": total_scored,
            "correct_tokens": total_correct,
            "token_accuracy": total_correct / total_scored,
            "exact_records": exact_records,
            "exact_record_rate": exact_records / records,
        },
        "per_record": per_record,
    }


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, name: str) -> list[dict[str, Any]]:
    if not rows:
        raise PairScoreError(f"{name} is empty")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise PairScoreError(f"{name} contains a malformed row")
        record_id = row.get("record_id")
        correct = row.get("correct_tokens")
        scored = row.get("scored_tokens")
        exact = row.get("exact_record")
        if not isinstance(record_id, str) or not record_id or record_id in ids:
            raise PairScoreError(f"{name} record IDs are not unique")
        if isinstance(correct, bool) or isinstance(scored, bool) or not isinstance(correct, (int, np.integer)) or not isinstance(scored, (int, np.integer)):
            raise PairScoreError(f"{name} counts are not integers")
        correct = int(correct)
        scored = int(scored)
        if scored != POST_BOS_TOKENS or correct < 0 or correct > scored:
            raise PairScoreError(f"{name} counts do not describe 127 post-BOS tokens")
        if not isinstance(exact, (bool, np.bool_)):
            raise PairScoreError(f"{name} exact flag is not boolean")
        expected_exact = correct == POST_BOS_TOKENS
        if bool(exact) != expected_exact:
            raise PairScoreError(f"{name} exact flag disagrees with counts")
        ids.add(record_id)
        result.append(
            {
                "record_id": record_id,
                "correct_tokens": correct,
                "scored_tokens": scored,
                "token_accuracy": correct / scored,
                "exact_record": expected_exact,
            }
        )
    return result


def make_resample_schedule(records: int, *, draws: int, seed: int) -> np.ndarray:
    """Create the shared source-record bootstrap schedule for one domain."""

    if records <= 0 or draws <= 0:
        raise PairScoreError("bootstrap records/draws must be positive")
    if not isinstance(seed, (int, np.integer)):
        raise PairScoreError("bootstrap seed must be integer")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, records, size=(int(draws), records), dtype=np.int64)


def _bootstrap_from_schedule(
    causal: Sequence[Mapping[str, Any]],
    diagonal: Sequence[Mapping[str, Any]],
    schedule: np.ndarray,
    *,
    tail_alpha: float,
) -> dict[str, Any]:
    left = _validate_rows(causal, name="causal rows")
    right = _validate_rows(diagonal, name="diagonal rows")
    if [row["record_id"] for row in left] != [row["record_id"] for row in right]:
        raise PairScoreError("paired source IDs changed between methods")
    if not isinstance(schedule, np.ndarray) or schedule.ndim != 2 or schedule.shape[1] != len(left):
        raise PairScoreError("bootstrap schedule geometry changed")
    if schedule.size and (schedule.min() < 0 or schedule.max() >= len(left)):
        raise PairScoreError("bootstrap schedule indexes outside source records")
    if not 0.0 < tail_alpha < 1.0:
        raise PairScoreError("bootstrap tail alpha must be in (0,1)")
    left_correct = np.asarray([row["correct_tokens"] for row in left], dtype=np.float64)
    right_correct = np.asarray([row["correct_tokens"] for row in right], dtype=np.float64)
    scored = np.asarray([row["scored_tokens"] for row in left], dtype=np.float64)
    left_exact = np.asarray([row["exact_record"] for row in left], dtype=np.float64)
    right_exact = np.asarray([row["exact_record"] for row in right], dtype=np.float64)
    beneficial = (left_exact == 1.0) & (right_exact == 0.0)
    harmful = (left_exact == 0.0) & (right_exact == 1.0)
    deltas: list[np.ndarray] = []
    exact_deltas: list[np.ndarray] = []
    beneficial_rates: list[np.ndarray] = []
    harmful_rates: list[np.ndarray] = []
    # Chunking bounds peak planning memory while consuming exactly the same
    # schedule array for every target/method comparison.
    for start in range(0, schedule.shape[0], 256):
        indices = schedule[start : start + 256]
        denom = scored[indices].sum(axis=1)
        left_rate = left_correct[indices].sum(axis=1) / denom
        right_rate = right_correct[indices].sum(axis=1) / denom
        deltas.append(left_rate - right_rate)
        exact_deltas.append(left_exact[indices].mean(axis=1) - right_exact[indices].mean(axis=1))
        beneficial_rates.append(beneficial[indices].mean(axis=1))
        harmful_rates.append(harmful[indices].mean(axis=1))
    delta_draws = np.concatenate(deltas)
    exact_delta_draws = np.concatenate(exact_deltas)
    beneficial_draws = np.concatenate(beneficial_rates)
    harmful_draws = np.concatenate(harmful_rates)
    q = lambda values, probability: float(np.quantile(values, probability))
    left_correct_total = float(left_correct.sum())
    right_correct_total = float(right_correct.sum())
    total_scored = float(scored.sum())
    left_exact_total = float(left_exact.sum())
    right_exact_total = float(right_exact.sum())
    return {
        "unit": "paired natural source record; micro correct/scored ratio per draw",
        "records": len(left),
        "draws": int(schedule.shape[0]),
        "schedule_shared": True,
        "left_estimate_pp": 100.0 * left_correct_total / total_scored,
        "right_estimate_pp": 100.0 * right_correct_total / total_scored,
        "delta_estimate_pp": 100.0 * (left_correct_total - right_correct_total) / total_scored,
        "delta_ci95_percentile_pp": [q(delta_draws, 0.025) * 100.0, q(delta_draws, 0.975) * 100.0],
        "lower_tail_alpha": float(tail_alpha),
        "upper_tail_alpha": float(tail_alpha),
        "delta_lower_practical_bound_pp": q(delta_draws, tail_alpha) * 100.0,
        "delta_upper_practical_bound_pp": q(delta_draws, 1.0 - tail_alpha) * 100.0,
        "exact_delta_estimate_pp": 100.0 * (left_exact_total - right_exact_total) / len(left),
        "exact_delta_ci95_percentile_pp": [q(exact_delta_draws, 0.025) * 100.0, q(exact_delta_draws, 0.975) * 100.0],
        "beneficial_rate_ci95_percentile_pp": [q(beneficial_draws, 0.025) * 100.0, q(beneficial_draws, 0.975) * 100.0],
        "harmful_rate_ci95_percentile_pp": [q(harmful_draws, 0.025) * 100.0, q(harmful_draws, 0.975) * 100.0],
        "coverage": "Approximate bootstrap coverage; these are descriptive/planning bounds, not finite-sample exact guarantees.",
    }


def paired_token_bootstrap(
    causal: Sequence[Mapping[str, Any]],
    diagonal: Sequence[Mapping[str, Any]],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    tail_alpha: float = TOKEN_TAIL_ALPHA,
    schedule: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return token and exact descriptive paired bootstrap quantities."""

    n = len(causal)
    if schedule is None:
        schedule = make_resample_schedule(n, draws=draws, seed=seed)
    elif draws != schedule.shape[0]:
        raise PairScoreError("draw count disagrees with supplied source schedule")
    result = _bootstrap_from_schedule(causal, diagonal, schedule, tail_alpha=tail_alpha)
    result["seed"] = int(seed)
    result["schedule_seed"] = int(seed)
    return result


def _log_binomial_term(n: int, k: int, p: float) -> float:
    if p == 0.0:
        return 0.0 if k == 0 else -math.inf
    if p == 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def _binomial_cdf(n: int, successes: int, p: float) -> float:
    if n < 0 or successes < 0 or successes > n:
        raise PairScoreError("invalid binomial count")
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if successes >= n else 0.0
    terms = [_log_binomial_term(n, k, p) for k in range(successes + 1)]
    maximum = max(terms)
    if maximum == -math.inf:
        return 0.0
    total = math.exp(maximum) * sum(math.exp(term - maximum) for term in terms)
    return min(1.0, max(0.0, total))


def cp_upper_rate(*, records: int, successes: int, tail_alpha: float) -> float:
    """One-sided Clopper-Pearson upper endpoint."""

    if records <= 0 or successes < 0 or successes > records or not 0.0 < tail_alpha < 1.0:
        raise PairScoreError("invalid upper CP arguments")
    if successes == records:
        return 1.0
    low, high = successes / records, 1.0
    for _ in range(90):
        mid = (low + high) / 2.0
        if _binomial_cdf(records, successes, mid) > tail_alpha:
            low = mid
        else:
            high = mid
    return high


def cp_lower_rate(*, records: int, successes: int, tail_alpha: float) -> float:
    """One-sided Clopper-Pearson lower endpoint."""

    if records <= 0 or successes < 0 or successes > records or not 0.0 < tail_alpha < 1.0:
        raise PairScoreError("invalid lower CP arguments")
    if successes == 0:
        return 0.0
    low, high = 0.0, successes / records
    for _ in range(90):
        mid = (low + high) / 2.0
        # P[X >= successes] = 1 - P[X <= successes - 1].
        tail = 1.0 - _binomial_cdf(records, successes - 1, mid)
        if tail < tail_alpha:
            low = mid
        else:
            high = mid
    return low


def exact_discordance_categories(causal: Sequence[Mapping[str, Any]], diagonal: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count mutually exclusive both/gain/loss/neither exact events."""

    left = _validate_rows(causal, name="causal rows")
    right = _validate_rows(diagonal, name="diagonal rows")
    if [row["record_id"] for row in left] != [row["record_id"] for row in right]:
        raise PairScoreError("paired source IDs changed between exact methods")
    both = causal_only = positionwise_only = neither = 0
    for left_row, right_row in zip(left, right):
        left_exact = bool(left_row["exact_record"])
        right_exact = bool(right_row["exact_record"])
        if left_exact and right_exact:
            both += 1
        elif left_exact:
            causal_only += 1
        elif right_exact:
            positionwise_only += 1
        else:
            neither += 1
    records = len(left)
    if both + causal_only + positionwise_only + neither != records:
        raise PairScoreError("exact event categories do not partition source records")
    return {
        "records": records,
        "both_correct": both,
        "causal_only": causal_only,
        "positionwise_only": positionwise_only,
        "neither_correct": neither,
        "gain": causal_only,
        "loss": positionwise_only,
        "point_delta_pp": 100.0 * (causal_only - positionwise_only) / records,
    }


def exact_net_bounds(
    *,
    records: int,
    gain: int,
    loss: int,
    tail_alpha: float = EXACT_TAIL_ALPHA,
    margin_pp: float = EXACT_MARGIN_PP,
) -> dict[str, Any]:
    """Return CP lower/upper bounds for ``U(g)-L(h)`` and ``L(g)-U(h)``."""

    if records <= 0 or gain < 0 or loss < 0 or gain + loss > records:
        raise PairScoreError("invalid exact discordance counts")
    gain_lower = cp_lower_rate(records=records, successes=gain, tail_alpha=tail_alpha)
    gain_upper = cp_upper_rate(records=records, successes=gain, tail_alpha=tail_alpha)
    loss_lower = cp_lower_rate(records=records, successes=loss, tail_alpha=tail_alpha)
    loss_upper = cp_upper_rate(records=records, successes=loss, tail_alpha=tail_alpha)
    net_lower = gain_lower - loss_upper
    net_upper = gain_upper - loss_lower
    return {
        "records": records,
        "gain": gain,
        "loss": loss,
        "tail_alpha_each": float(tail_alpha),
        "gain_lower_pp": gain_lower * 100.0,
        "gain_upper_pp": gain_upper * 100.0,
        "loss_lower_pp": loss_lower * 100.0,
        "loss_upper_pp": loss_upper * 100.0,
        "lower_practical_bound_pp": net_lower * 100.0,
        "upper_practical_bound_pp": net_upper * 100.0,
        "formula_lower": "L_CP(g) - U_CP(h)",
        "formula_upper": "U_CP(g) - L_CP(h)",
        "cp_marginal_coverage": "Exact one-sided marginal guarantee for each CP endpoint; the retained alpha allocation is conservative for the directional family.",
        "practical_margin_pp": float(margin_pp),
        "support": bool(net_lower * 100.0 >= margin_pp),
        "exclude": bool(net_upper * 100.0 < margin_pp),
        "harm_not_excluded": bool(net_lower * 100.0 < -margin_pp),
        "material_harm_evidence": bool(net_upper * 100.0 < -margin_pp),
    }


def paired_comparison(
    causal: Sequence[Mapping[str, Any]],
    diagonal: Sequence[Mapping[str, Any]],
    *,
    cell_id: str | None = None,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    schedule: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare causal and diagonal rows with all registered report fields."""

    categories = exact_discordance_categories(causal, diagonal)
    if schedule is None:
        schedule = make_resample_schedule(categories["records"], draws=draws, seed=seed)
    bootstrap = paired_token_bootstrap(
        causal,
        diagonal,
        draws=schedule.shape[0],
        seed=seed,
        tail_alpha=TOKEN_TAIL_ALPHA,
        schedule=schedule,
    )
    exact = exact_net_bounds(
        records=categories["records"],
        gain=categories["gain"],
        loss=categories["loss"],
    )
    causal_rows = _validate_rows(causal, name="causal rows")
    diagonal_rows = _validate_rows(diagonal, name="diagonal rows")
    causal_exact = int(sum(row["exact_record"] for row in causal_rows))
    diagonal_exact = int(sum(row["exact_record"] for row in diagonal_rows))
    return {
        "task_id": TASK_ID,
        "cell_id": cell_id,
        "contrast": "enriched causal versus enriched trained diagonal positionwise control",
        "records": categories["records"],
        "exact_categories": categories,
        "absolute_exact_recovery": {
            "causal_exact_records": causal_exact,
            "diagonal_exact_records": diagonal_exact,
            "causal_rate_pp": 100.0 * causal_exact / categories["records"],
            "diagonal_rate_pp": 100.0 * diagonal_exact / categories["records"],
            "definition": "all 127 scored post-BOS tokens in the declared 128-token clip",
        },
        "token": bootstrap,
        "exact": {
            **exact,
            "point_delta_pp": categories["point_delta_pp"],
            "descriptive_bootstrap_ci95_pp": bootstrap["exact_delta_ci95_percentile_pp"],
        },
        "endpoint_status": {
            "token_support": bool(bootstrap["delta_lower_practical_bound_pp"] >= TOKEN_MARGIN_PP),
            "token_exclusion": bool(bootstrap["delta_upper_practical_bound_pp"] < TOKEN_MARGIN_PP),
            "token_harm_not_excluded": bool(bootstrap["delta_lower_practical_bound_pp"] < -TOKEN_HARM_MARGIN_PP),
            "token_material_harm_evidence": bool(bootstrap["delta_upper_practical_bound_pp"] < -TOKEN_HARM_MARGIN_PP),
            "exact_support": bool(exact["lower_practical_bound_pp"] >= EXACT_MARGIN_PP),
            "exact_exclusion": bool(exact["upper_practical_bound_pp"] < EXACT_MARGIN_PP),
            "exact_harm_not_excluded": bool(exact["lower_practical_bound_pp"] < -EXACT_HARM_MARGIN_PP),
            "exact_material_harm_evidence": bool(exact["upper_practical_bound_pp"] < -EXACT_HARM_MARGIN_PP),
        },
    }


def _cell_outcome_rows(comparisons: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell_id in CELL_ORDER:
        result = comparisons[cell_id]
        token = result["token"]
        exact = result["exact"]
        rows.extend(
            [
                {
                    "cell_id": cell_id,
                    "outcome": "token",
                    "lower_bound_pp": float(token["delta_lower_practical_bound_pp"]),
                    "upper_bound_pp": float(token["delta_upper_practical_bound_pp"]),
                    "benefit_margin_pp": TOKEN_MARGIN_PP,
                    "harm_margin_pp": TOKEN_HARM_MARGIN_PP,
                    "coverage": "bootstrap_approximate",
                },
                {
                    "cell_id": cell_id,
                    "outcome": "exact",
                    "lower_bound_pp": float(exact["lower_practical_bound_pp"]),
                    "upper_bound_pp": float(exact["upper_practical_bound_pp"]),
                    "benefit_margin_pp": EXACT_MARGIN_PP,
                    "harm_margin_pp": EXACT_HARM_MARGIN_PP,
                    "coverage": "clopper_pearson_exact_marginal",
                },
            ]
        )
    return rows


def classify_matrix(
    comparisons: Mapping[str, Mapping[str, Any]],
    *,
    runtime_ratio: float | None = None,
    training_ratio: float | None = None,
) -> dict[str, Any]:
    """Apply the predeclared cell-level support/exclusion rules.

    Support is scoped: at least one of the eight cell/outcome lower bounds
    reaches its practical benefit margin, while every cell/outcome lower bound
    excludes the corresponding harm margin.  Exclusion is stricter: both
    token and exact upper bounds must be below their margins in all four cells.
    A causal-worse result can therefore support the positionwise default; a
    lower bound below a harm margin alone is only ``harm_not_excluded``.
    """

    if set(comparisons) != set(CELL_ORDER):
        raise PairScoreError("matrix comparison results are not exactly four cells")
    endpoint_rows = _cell_outcome_rows(comparisons)
    supporting = [
        row for row in endpoint_rows if row["lower_bound_pp"] >= row["benefit_margin_pp"]
    ]
    harm_exceptions = [
        row for row in endpoint_rows if row["lower_bound_pp"] < -row["harm_margin_pp"]
    ]
    material_harm = [
        row for row in endpoint_rows if row["upper_bound_pp"] < -row["harm_margin_pp"]
    ]
    exclusion_failures = [
        row for row in endpoint_rows if row["upper_bound_pp"] >= row["benefit_margin_pp"]
    ]
    quality_support = bool(supporting and not harm_exceptions)
    quality_exclusion = not exclusion_failures
    if quality_support:
        decision = "context_gain_supported"
    elif quality_exclusion:
        decision = "positionwise_default"
    else:
        decision = "inconclusive"
    if material_harm:
        harm_status = "evidence_of_material_harm"
    elif harm_exceptions:
        harm_status = "harm_not_excluded"
    else:
        harm_status = "harm_excluded"
    cost: dict[str, Any] = {
        "runtime_ratio": runtime_ratio,
        "runtime_budget": 1.25,
        "runtime_qualified": runtime_ratio is not None and math.isfinite(runtime_ratio) and runtime_ratio <= 1.25,
        "training_ratio": training_ratio,
        "training_budget": 2.0,
        "training_qualified": training_ratio is not None and math.isfinite(training_ratio) and training_ratio <= 2.0,
        "status": "qualified" if (
            runtime_ratio is not None
            and training_ratio is not None
            and math.isfinite(runtime_ratio)
            and math.isfinite(training_ratio)
            and runtime_ratio <= 1.25
            and training_ratio <= 2.0
        ) else "unqualified_or_unavailable",
        "interpretation": "Cost qualification is reported separately from quality inference; no new training is implied.",
    }
    return {
        "decision": decision,
        "quality_support": quality_support,
        "quality_exclusion": quality_exclusion,
        "supporting_endpoints": supporting,
        "harm_exceptions": harm_exceptions,
        "material_harm_endpoints": material_harm,
        "harm_status": harm_status,
        "exclusion_failures": exclusion_failures,
        "directional_family": {
            "cells": 4,
            "outcomes": ["token", "exact"],
            "directions": ["lower_support_or_harm", "upper_exclusion"],
            "endpoint_count": 16,
            "token_tail_alpha": TOKEN_TAIL_ALPHA,
            "exact_cp_tail_alpha_each": EXACT_TAIL_ALPHA,
            "duplicate_contrasts": 0,
        },
        "cost_qualification": cost,
    }


def score_matrix(
    *,
    predictions: Mapping[tuple[str, str], Any],
    truth: Mapping[str, Any],
    attention_masks: Mapping[str, Any],
    record_ids: Mapping[str, Sequence[str]],
    position_ids: Mapping[str, Any] | None = None,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    runtime_ratio: float | None = None,
    training_ratio: float | None = None,
) -> dict[str, Any]:
    """Score all eight method/cell outputs after the public gate."""

    expected = {(cell, method) for cell in CELL_ORDER for method in METHOD_ORDER}
    if set(predictions) != expected:
        raise PairScoreError("prediction matrix is incomplete")
    if set(truth) != set(CELL_ORDER) or set(attention_masks) != set(CELL_ORDER) or set(record_ids) != set(CELL_ORDER):
        raise PairScoreError("truth/geometry matrix is incomplete")
    if position_ids is not None and set(position_ids) != set(CELL_ORDER):
        raise PairScoreError("position matrix is incomplete")
    for domain in ("pile", "finance"):
        base_cell = f"{domain}__public_base"
        lora_cell = f"{domain}__public_lora_2601"
        if tuple(record_ids[base_cell]) != tuple(record_ids[lora_cell]):
            raise PairScoreError(f"paired source IDs changed between target conditions: {domain}")
    scored: dict[tuple[str, str], dict[str, Any]] = {}
    for cell_id in CELL_ORDER:
        for method_id in METHOD_ORDER:
            row = score_cell(
                predictions=predictions[(cell_id, method_id)],
                truth=truth[cell_id],
                attention_mask=attention_masks[cell_id],
                record_ids=record_ids[cell_id],
                method_id=method_id,
                position_ids=position_ids[cell_id] if position_ids is not None else None,
            )
            row["cell_id"] = cell_id
            scored[(cell_id, method_id)] = row
    comparisons: dict[str, dict[str, Any]] = {}
    # A domain seed is intentionally reused for its two target conditions;
    # this produces the same source-resample schedule across targets.  A
    # target product is never used as a joint probability claim.
    domain_seed = {"pile": int(bootstrap_seed), "finance": int(bootstrap_seed) + 1}
    for cell_id in CELL_ORDER:
        domain = cell_id.split("__", 1)[0]
        causal_rows = scored[(cell_id, METHOD_ORDER[0])]["per_record"]
        diagonal_rows = scored[(cell_id, METHOD_ORDER[1])]["per_record"]
        schedule = make_resample_schedule(len(causal_rows), draws=bootstrap_draws, seed=domain_seed[domain])
        comparisons[cell_id] = paired_comparison(
            causal_rows,
            diagonal_rows,
            cell_id=cell_id,
            draws=bootstrap_draws,
            seed=domain_seed[domain],
            schedule=schedule,
        )
    matrix_decision = classify_matrix(
        comparisons,
        runtime_ratio=runtime_ratio,
        training_ratio=training_ratio,
    )
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "TRR6_SCORED_AFTER_PUBLIC_GATE",
        "claim_scope": "Two frozen enriched methods; four domain/target cells; no pooled universal claim.",
        "matrix": {
            "cells": list(CELL_ORDER),
            "methods": list(METHOD_ORDER),
            "records_per_cell": {cell: len(scored[(cell, METHOD_ORDER[0])]["per_record"]) for cell in CELL_ORDER},
            "sequence_tokens": SEQUENCE_TOKENS,
            "scored_post_bos_tokens": POST_BOS_TOKENS,
            "source_pairing": "same source IDs across target conditions within each domain",
        },
        "bootstrap": {
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
            "unit": "paired source record",
            "same_schedule_across_targets": True,
            "tail_alpha": TOKEN_TAIL_ALPHA,
            "coverage": "approximate",
        },
        "exact_uncertainty": {
            "tail_alpha_each": EXACT_TAIL_ALPHA,
            "bound": "U_CP(p_gain)-L_CP(p_loss) for upper; L_CP(p_gain)-U_CP(p_loss) for lower",
            "coverage": "exact marginal CP endpoint guarantee",
        },
        "cells": {cell: comparisons[cell] for cell in CELL_ORDER},
        "method_scores": {
            f"{cell}__{method}": scored[(cell, method)] for cell in CELL_ORDER for method in METHOD_ORDER
        },
        "decision": matrix_decision,
    }


def score_with_truth_loader(
    *,
    public_gate: Callable[[], Mapping[str, Any]],
    truth_loader: Callable[[], Mapping[str, Any]],
    score_after_truth: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Invoke the truth loader only after a complete public gate succeeds.

    ``score_after_truth`` receives the gate receipt and loaded truth, allowing
    the runner to load predictions only after the gate and to call
    :func:`score_matrix`.  The callback is not invoked when the gate raises or
    does not explicitly report ``verified_before_truth``.
    """

    try:
        gate = public_gate()
    except Exception as exc:
        raise PairScoreError("public gate failed before truth loader") from exc
    if not isinstance(gate, Mapping) or gate.get("verified_before_truth") is not True or gate.get("truth_opened") is not False:
        raise PairScoreError("public gate did not return a verified closed receipt")
    truth = truth_loader()
    if not isinstance(truth, Mapping):
        raise PairScoreError("truth loader did not return a mapping")
    result = score_after_truth(gate, truth)
    if not isinstance(result, Mapping):
        raise PairScoreError("post-truth scorer did not return a mapping")
    return dict(result)


def extract_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a compact report while retaining every required cell field."""

    if result.get("schema") != SCHEMA or result.get("task_id") != TASK_ID:
        raise PairScoreError("score result schema/task binding changed")
    cells = result.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(CELL_ORDER):
        raise PairScoreError("score result cell matrix is incomplete")
    rows: list[dict[str, Any]] = []
    for cell_id in CELL_ORDER:
        cell = cells[cell_id]
        categories = cell["exact_categories"]
        absolute = cell["absolute_exact_recovery"]
        token = cell["token"]
        exact = cell["exact"]
        rows.append(
            {
                "cell_id": cell_id,
                "both_correct": categories["both_correct"],
                "causal_only": categories["causal_only"],
                "positionwise_only": categories["positionwise_only"],
                "neither_correct": categories["neither_correct"],
                "causal_absolute_exact_rate_pp": absolute["causal_rate_pp"],
                "positionwise_absolute_exact_rate_pp": absolute["diagonal_rate_pp"],
                "token_delta_pp": token["delta_estimate_pp"],
                "token_ci95_pp": token["delta_ci95_percentile_pp"],
                "token_lower_practical_bound_pp": token["delta_lower_practical_bound_pp"],
                "token_upper_practical_bound_pp": token["delta_upper_practical_bound_pp"],
                "exact_delta_pp": exact["point_delta_pp"],
                "exact_ci95_pp": exact["descriptive_bootstrap_ci95_pp"],
                "exact_lower_practical_bound_pp": exact["lower_practical_bound_pp"],
                "exact_upper_practical_bound_pp": exact["upper_practical_bound_pp"],
            }
        )
    return {
        "schema": "token-reconstruction.trr0006-pair-report.v1",
        "task_id": TASK_ID,
        "decision": result["decision"],
        "rows": rows,
        "directional_family": result["decision"]["directional_family"],
        "cost_qualification": result["decision"]["cost_qualification"],
        "scope": result["claim_scope"],
    }


def render_report(result: Mapping[str, Any]) -> str:
    report = extract_report(result)
    decision = report["decision"]
    lines = [
        "# TRR-0006 paired context report",
        "",
        f"Decision: **{decision['decision']}**; harm status: **{decision['harm_status']}**.",
        "",
        "Bounds are causal minus trained diagonal. Bootstrap endpoints have approximate coverage; CP endpoints have exact marginal one-sided coverage.",
        "",
        "| Cell | Both | Causal only | Positionwise only | Neither | Causal exact % | Diagonal exact % | Token Δ pp [95%] | Token practical [L,U] pp | Exact Δ pp [95%] | Exact practical [L,U] pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {cell} | {both_correct} | {causal_only} | {positionwise_only} | {neither_correct} | {causal_absolute_exact_rate_pp:.3f} | {positionwise_absolute_exact_rate_pp:.3f} | {token_delta_pp:.3f} [{t0:.3f},{t1:.3f}] | [{tl:.3f},{tu:.3f}] | {exact_delta_pp:.3f} [{e0:.3f},{e1:.3f}] | [{el:.3f},{eu:.3f}] |".format(
                cell=row["cell_id"],
                t0=row["token_ci95_pp"][0],
                t1=row["token_ci95_pp"][1],
                e0=row["exact_ci95_pp"][0],
                e1=row["exact_ci95_pp"][1],
                tl=row["token_lower_practical_bound_pp"],
                tu=row["token_upper_practical_bound_pp"],
                el=row["exact_lower_practical_bound_pp"],
                eu=row["exact_upper_practical_bound_pp"],
                **row,
            )
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "BOS_TOKEN_ID",
    "CELL_ORDER",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "EXACT_TAIL_ALPHA",
    "METHOD_ORDER",
    "PairScoreError",
    "TOKEN_TAIL_ALPHA",
    "classify_matrix",
    "cp_lower_rate",
    "cp_upper_rate",
    "exact_discordance_categories",
    "exact_net_bounds",
    "extract_report",
    "make_resample_schedule",
    "paired_comparison",
    "paired_token_bootstrap",
    "render_report",
    "score_cell",
    "score_matrix",
    "score_with_truth_loader",
]
