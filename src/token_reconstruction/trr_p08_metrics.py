"""Pure TRR-P08 scoring, replicate aggregation, and paired interaction metrics.

The module accepts already materialized prediction/truth arrays and score rows.
It does not load observations, truth, checkpoints, or prediction artifacts.  A
caller must complete the P08 joint-freeze gate before passing truth here.

The two fit seeds are replicate measurements of each source record.  All
correctness counts, exact indicators, and paired gain/loss counts are averaged
within source before source-record bootstrap resampling.  The primary
interaction is computed from four arm contrasts::

    (past_staged - positionwise_staged)
      - (past_joint - positionwise_joint)

Token and exact denominators remain source-level quantities; no token IDs or
replicate seeds are averaged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np

TASK_ID = "TRR-P08"
SEQUENCE_TOKENS = 128
BOS_TOKEN_ID = 128000
VOCABULARY_SIZE = 128256
POST_BOS_POSITIONS = tuple(range(1, SEQUENCE_TOKENS))
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 8080
REPLICATE_SEEDS = (6106, 6107)
DOMAINS = ("pile", "finance")
TARGETS = ("public_base", "public_lora_2601")
METHOD_ORDER = (
    "p08_positionwise_joint",
    "p08_positionwise_staged",
    "p08_past_only_joint",
    "p08_past_only_staged",
)
INTERACTION_METHODS = {
    "positionwise_joint": "p08_positionwise_joint",
    "positionwise_staged": "p08_positionwise_staged",
    "past_joint": "p08_past_only_joint",
    "past_staged": "p08_past_only_staged",
}
POSITION_BINS = {
    "early": tuple(range(1, 16)),
    "early_middle": tuple(range(16, 40)),
    "late_middle": tuple(range(40, 80)),
    "near_end": tuple(range(80, 128)),
}


class P08MetricsError(ValueError):
    """Raised when a P08 metric input violates the frozen geometry."""


def _array(value: Any, *, name: str) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(value))
    except Exception as exc:  # pragma: no cover - backend-specific conversion
        raise P08MetricsError(f"{name} is not array-like") from exc


def _record_ids(record_ids: Sequence[str] | None, records: int) -> tuple[str, ...]:
    if record_ids is None:
        return tuple(f"record-{index}" for index in range(records))
    if len(record_ids) != records:
        raise P08MetricsError("record ID count differs from prediction rows")
    values = tuple(record_ids)
    if any(not isinstance(value, str) or not value for value in values):
        raise P08MetricsError("record IDs must be nonempty strings")
    if len(set(values)) != records:
        raise P08MetricsError("record IDs must be unique")
    return values


def _matrix(value: Any, *, name: str, records: int | None = None) -> np.ndarray:
    result = _array(value, name=name)
    if result.ndim != 2 or result.shape[1] != SEQUENCE_TOKENS:
        raise P08MetricsError(f"{name} must have shape [records, 128]")
    if records is not None and result.shape[0] != records:
        raise P08MetricsError(f"{name} record count differs from predictions")
    if not np.issubdtype(result.dtype, np.integer):
        raise P08MetricsError(f"{name} must contain integer token IDs")
    return result.astype(np.int64, copy=False)


def _mask(value: Any | None, records: int) -> np.ndarray:
    if value is None:
        result = np.ones((records, SEQUENCE_TOKENS), dtype=bool)
    else:
        result = _array(value, name="attention_mask")
        if result.shape != (records, SEQUENCE_TOKENS):
            raise P08MetricsError("attention_mask must have shape [records, 128]")
        if not np.issubdtype(result.dtype, np.bool_) and not np.isin(result, [0, 1]).all():
            raise P08MetricsError("attention_mask must be binary")
        result = result.astype(bool, copy=False)
    if records and not result[:, 0].all():
        raise P08MetricsError("every record must contain BOS")
    if (result[:, 1:] > result[:, :-1]).any():
        raise P08MetricsError("attention_mask must be right-padded")
    return result


def _positions(value: Any | None, records: int) -> None:
    if value is None:
        return
    result = _array(value, name="position_ids")
    if result.shape != (records, SEQUENCE_TOKENS) or not np.issubdtype(result.dtype, np.integer):
        raise P08MetricsError("position_ids must have integer shape [records, 128]")
    expected = np.broadcast_to(np.arange(SEQUENCE_TOKENS, dtype=np.int64), result.shape)
    if not np.array_equal(result.astype(np.int64, copy=False), expected):
        raise P08MetricsError("position_ids must be 0..127 in every row")


def _validate_ids(values: np.ndarray, mask: np.ndarray, *, name: str) -> None:
    if values.shape != mask.shape:
        raise P08MetricsError(f"{name} and attention_mask geometry differs")
    if not np.all(values[:, 0] == BOS_TOKEN_ID):
        raise P08MetricsError(f"{name} BOS differs from {BOS_TOKEN_ID}")
    active = values[mask]
    if np.any(active < 0) or np.any(active >= VOCABULARY_SIZE):
        raise P08MetricsError(f"{name} active IDs leave the public vocabulary")


def _position_metrics(correct: np.ndarray, active: np.ndarray) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, positions in POSITION_BINS.items():
        indices = np.asarray([position - 1 for position in positions], dtype=np.int64)
        valid = active[:, indices]
        values = correct[:, indices]
        scored = int(valid.sum())
        correct_tokens = int((values & valid).sum())
        result[name] = {
            "positions": list(positions),
            "scored_tokens": scored,
            "correct_tokens": correct_tokens,
            "token_accuracy": correct_tokens / scored if scored else None,
        }
    return result


def score_method(
    predictions: Any,
    truth: Any,
    *,
    record_ids: Sequence[str] | None = None,
    attention_mask: Any | None = None,
    position_ids: Any | None = None,
    method_id: str | None = None,
) -> dict[str, Any]:
    """Score one frozen method on valid post-BOS positions.

    Short/padded records remain in token metrics and are excluded from the
    exact-clip denominator.  All paired methods must supply identical masks.
    """

    if method_id is not None and method_id not in METHOD_ORDER:
        raise P08MetricsError(f"unknown P08 method: {method_id}")
    prediction_array = _matrix(predictions, name="predictions")
    records = int(prediction_array.shape[0])
    if records <= 0:
        raise P08MetricsError("at least one record is required")
    truth_array = _matrix(truth, name="truth", records=records)
    mask = _mask(attention_mask, records)
    _positions(position_ids, records)
    _validate_ids(prediction_array, mask, name="predictions")
    _validate_ids(truth_array, mask, name="truth")
    ids = _record_ids(record_ids, records)

    active = mask.copy()
    active[:, 0] = False
    correct = (prediction_array == truth_array) & active
    scored_per_record = active.sum(axis=1).astype(np.int64)
    correct_per_record = correct.sum(axis=1).astype(np.int64)
    exact_eligible = scored_per_record == len(POST_BOS_POSITIONS)
    exact_record = (correct_per_record == scored_per_record) & exact_eligible
    nonempty = scored_per_record > 0
    total_scored = int(scored_per_record.sum())
    total_correct = int(correct_per_record.sum())
    exact_denominator = int(exact_eligible.sum())
    exact_records = int(exact_record.sum())
    per_record: list[dict[str, Any]] = []
    for index, record_id in enumerate(ids):
        per_record.append(
            {
                "record_id": record_id,
                "correct_tokens": int(correct_per_record[index]),
                "scored_tokens": int(scored_per_record[index]),
                "token_accuracy": (
                    float(correct_per_record[index] / scored_per_record[index])
                    if scored_per_record[index]
                    else None
                ),
                "exact_eligible": bool(exact_eligible[index]),
                "exact_record": bool(exact_record[index]) if exact_eligible[index] else None,
                "correctness": [bool(item) for item in correct[index, 1:]],
                "valid_post_bos": [bool(item) for item in active[index, 1:]],
            }
        )
    return {
        "task_id": TASK_ID,
        "method_id": method_id,
        "records": records,
        "record_ids": list(ids),
        "clip_tokens_including_bos": SEQUENCE_TOKENS,
        "scored_post_bos_tokens": len(POST_BOS_POSITIONS),
        "metrics": {
            "scored_tokens": total_scored,
            "correct_tokens": total_correct,
            "token_accuracy": total_correct / total_scored if total_scored else None,
            "macro_records": int(nonempty.sum()),
            "macro_token_accuracy": (
                float((correct_per_record[nonempty] / scored_per_record[nonempty]).mean())
                if nonempty.any()
                else None
            ),
            "exact_records": exact_records,
            "exact_denominator": exact_denominator,
            "exact_record_rate": exact_records / exact_denominator if exact_denominator else None,
        },
        "position_metrics": _position_metrics(correct, active),
        "per_record": per_record,
    }


def _bool_array(rows: Sequence[Mapping[str, Any]], key: str, *, width: int) -> np.ndarray:
    values = np.asarray([row[key] for row in rows], dtype=bool)
    if values.ndim != 2 or values.shape[1] != width:
        raise P08MetricsError(f"score rows have invalid {key} vectors")
    return values


def _validated_score(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P08MetricsError(f"{name} score is not an object")
    records = value.get("records")
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise P08MetricsError(f"{name} score has invalid record count")
    ids = value.get("record_ids")
    if not isinstance(ids, list):
        raise P08MetricsError(f"{name} score has no record IDs")
    normalized_ids = _record_ids(ids, records)
    rows = value.get("per_record")
    if not isinstance(rows, list) or len(rows) != records:
        raise P08MetricsError(f"{name} score has invalid per-record rows")
    for row, expected_id in zip(rows, normalized_ids):
        if not isinstance(row, Mapping) or row.get("record_id") != expected_id:
            raise P08MetricsError(f"{name} score record order changed")
        for key in ("correct_tokens", "scored_tokens"):
            if not isinstance(row.get(key), (int, np.integer)):
                raise P08MetricsError(f"{name} score lacks integer {key}")
        if not 0 <= int(row["correct_tokens"]) <= int(row["scored_tokens"]):
            raise P08MetricsError(f"{name} score counts are invalid")
        if not isinstance(row.get("exact_eligible"), (bool, np.bool_)):
            raise P08MetricsError(f"{name} score lacks exact eligibility")
        if bool(row["exact_eligible"]) and not isinstance(row.get("exact_record"), (bool, np.bool_)):
            raise P08MetricsError(f"{name} score lacks exact flag")
        correctness = row.get("correctness")
        valid = row.get("valid_post_bos")
        if (
            not isinstance(correctness, (list, tuple))
            or not isinstance(valid, (list, tuple))
            or len(correctness) != len(POST_BOS_POSITIONS)
            or len(valid) != len(POST_BOS_POSITIONS)
            or any(not isinstance(item, (bool, np.bool_)) for item in correctness)
            or any(not isinstance(item, (bool, np.bool_)) for item in valid)
        ):
            raise P08MetricsError(f"{name} score lacks fixed-width correctness/mask vectors")
        correctness_array = np.asarray(correctness, dtype=bool)
        valid_array = np.asarray(valid, dtype=bool)
        if np.any(correctness_array & ~valid_array):
            raise P08MetricsError(f"{name} marks a padded token correct")
        if int(valid_array.sum()) != int(row["scored_tokens"]):
            raise P08MetricsError(f"{name} mask/count mismatch")
        if int((correctness_array & valid_array).sum()) != int(row["correct_tokens"]):
            raise P08MetricsError(f"{name} correctness/count mismatch")
        eligible = int(row["scored_tokens"]) == len(POST_BOS_POSITIONS)
        if bool(row["exact_eligible"]) != eligible:
            raise P08MetricsError(f"{name} exact eligibility disagrees with denominator")
        if eligible and bool(row["exact_record"]) != bool(correctness_array.all()):
            raise P08MetricsError(f"{name} exact flag disagrees with correctness")
    return dict(value)


def _paired_arrays(left: Mapping[str, Any], right: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    left = _validated_score(left, name=f"{name}.left")
    right = _validated_score(right, name=f"{name}.right")
    if left["records"] != right["records"] or left["record_ids"] != right["record_ids"]:
        raise P08MetricsError(f"{name} methods do not have identical source-record order")
    lrows, rrows = left["per_record"], right["per_record"]
    lcorrect = _bool_array(lrows, "correctness", width=len(POST_BOS_POSITIONS))
    rcorrect = _bool_array(rrows, "correctness", width=len(POST_BOS_POSITIONS))
    valid = _bool_array(lrows, "valid_post_bos", width=len(POST_BOS_POSITIONS))
    rvalid = _bool_array(rrows, "valid_post_bos", width=len(POST_BOS_POSITIONS))
    if not np.array_equal(valid, rvalid):
        raise P08MetricsError(f"{name} methods have different valid-position masks")
    scored = np.asarray([int(row["scored_tokens"]) for row in lrows], dtype=np.int64)
    rscored = np.asarray([int(row["scored_tokens"]) for row in rrows], dtype=np.int64)
    if not np.array_equal(scored, rscored):
        raise P08MetricsError(f"{name} methods have different metric denominators")
    eligible = np.asarray([bool(row["exact_eligible"]) for row in lrows], dtype=bool)
    reeligible = np.asarray([bool(row["exact_eligible"]) for row in rrows], dtype=bool)
    if not np.array_equal(eligible, reeligible):
        raise P08MetricsError(f"{name} methods have different exact eligibility")
    lexact = np.asarray([bool(row["exact_record"]) if row["exact_eligible"] else False for row in lrows], dtype=bool)
    rexact = np.asarray([bool(row["exact_record"]) if row["exact_eligible"] else False for row in rrows], dtype=bool)
    gains = lcorrect & ~rcorrect & valid
    losses = ~lcorrect & rcorrect & valid
    both = lcorrect & rcorrect & valid
    neither = ~lcorrect & ~rcorrect & valid
    return {
        "left_method": left.get("method_id"),
        "right_method": right.get("method_id"),
        "record_ids": list(left["record_ids"]),
        "records": int(left["records"]),
        "scored": scored,
        "valid": valid,
        "left_correctness": lcorrect,
        "right_correctness": rcorrect,
        "left_correct": lcorrect.sum(axis=1).astype(np.int64),
        "right_correct": rcorrect.sum(axis=1).astype(np.int64),
        "gains": gains.sum(axis=1).astype(np.int64),
        "losses": losses.sum(axis=1).astype(np.int64),
        "both": both.sum(axis=1).astype(np.int64),
        "neither": neither.sum(axis=1).astype(np.int64),
        "eligible": eligible,
        "left_exact": lexact,
        "right_exact": rexact,
    }


def paired_metrics_from_scores(
    left_score: Mapping[str, Any],
    right_score: Mapping[str, Any],
    *,
    contrast_id: str | None = None,
) -> dict[str, Any]:
    """Return paired token/exact gains and losses for one replicate."""

    arrays = _paired_arrays(left_score, right_score, name="paired")
    scored = arrays["scored"]
    valid = arrays["valid"]
    eligible = arrays["eligible"]
    per_record = []
    for i, record_id in enumerate(arrays["record_ids"]):
        per_record.append(
            {
                "record_id": record_id,
                "scored_tokens": int(scored[i]),
                "left_correct_tokens": int(arrays["left_correct"][i]),
                "right_correct_tokens": int(arrays["right_correct"][i]),
                "token_gains": int(arrays["gains"][i]),
                "token_losses": int(arrays["losses"][i]),
                "token_both_correct": int(arrays["both"][i]),
                "token_neither_correct": int(arrays["neither"][i]),
                "exact_eligible": bool(eligible[i]),
                "left_exact_record": bool(arrays["left_exact"][i]) if eligible[i] else None,
                "right_exact_record": bool(arrays["right_exact"][i]) if eligible[i] else None,
                "token_delta": int(arrays["left_correct"][i] - arrays["right_correct"][i]),
            }
        )
    denominator = int(scored.sum())
    exact_denominator = int(eligible.sum())
    left_hits = int(arrays["left_correct"].sum())
    right_hits = int(arrays["right_correct"].sum())
    left_exact = int((arrays["left_exact"] & eligible).sum())
    right_exact = int((arrays["right_exact"] & eligible).sum())
    return {
        "task_id": TASK_ID,
        "contrast_id": contrast_id,
        "left_method": arrays["left_method"],
        "right_method": arrays["right_method"],
        "record_ids": arrays["record_ids"],
        "records": arrays["records"],
        "metrics": {
            "scored_tokens": denominator,
            "left_correct_tokens": left_hits,
            "right_correct_tokens": right_hits,
            "left_token_accuracy": left_hits / denominator if denominator else None,
            "right_token_accuracy": right_hits / denominator if denominator else None,
            "token_delta_pp": 100.0 * (left_hits - right_hits) / denominator if denominator else None,
            "macro_token_delta_pp": (
                100.0 * np.mean(
                    (arrays["left_correct"][scored > 0] - arrays["right_correct"][scored > 0])
                    / scored[scored > 0]
                )
                if np.any(scored > 0)
                else None
            ),
            "token_gains": int(arrays["gains"].sum()),
            "token_losses": int(arrays["losses"].sum()),
            "token_both_correct": int(arrays["both"].sum()),
            "token_neither_correct": int(arrays["neither"].sum()),
            "exact_denominator": exact_denominator,
            "left_exact_records": left_exact,
            "right_exact_records": right_exact,
            "exact_delta_pp": 100.0 * (left_exact - right_exact) / exact_denominator if exact_denominator else None,
        },
        "per_record": per_record,
    }


def _mean(values: Sequence[Any]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_replicate_comparisons(
    comparisons: Mapping[str | int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Average paired correctness/gain/loss counts within each source."""

    if not comparisons:
        raise P08MetricsError("at least one replicate comparison is required")
    first = next(iter(comparisons.values()))
    record_ids = list(first["record_ids"])
    records = int(first["records"])
    for comparison in comparisons.values():
        if int(comparison["records"]) != records or list(comparison["record_ids"]) != record_ids:
            raise P08MetricsError("replicate comparison source records differ")
    per_record = []
    for i, record_id in enumerate(record_ids):
        rows = [comparison["per_record"][i] for comparison in comparisons.values()]
        scored = [int(row["scored_tokens"]) for row in rows]
        eligible = [bool(row["exact_eligible"]) for row in rows]
        if len(set(scored)) != 1 or len(set(eligible)) != 1:
            raise P08MetricsError(f"replicate denominator differs for source record {record_id}")
        row: dict[str, Any] = {
            "record_id": record_id,
            "scored_tokens": scored[0],
            "exact_eligible": eligible[0],
        }
        for key in (
            "left_correct_tokens", "right_correct_tokens", "token_gains", "token_losses",
            "token_both_correct", "token_neither_correct", "token_delta",
        ):
            row[key] = _mean([item[key] for item in rows])
        if eligible[0]:
            row["left_exact_record"] = _mean([float(item["left_exact_record"]) for item in rows])
            row["right_exact_record"] = _mean([float(item["right_exact_record"]) for item in rows])
        else:
            row["left_exact_record"] = None
            row["right_exact_record"] = None
        per_record.append(row)
    scored_array = np.asarray([row["scored_tokens"] for row in per_record], dtype=np.float64)
    left_array = np.asarray([row["left_correct_tokens"] for row in per_record], dtype=np.float64)
    right_array = np.asarray([row["right_correct_tokens"] for row in per_record], dtype=np.float64)
    eligible_array = np.asarray([row["exact_eligible"] for row in per_record], dtype=bool)
    left_exact = np.asarray([float(row["left_exact_record"]) if row["exact_eligible"] else 0.0 for row in per_record])
    right_exact = np.asarray([float(row["right_exact_record"]) if row["exact_eligible"] else 0.0 for row in per_record])
    denominator = float(scored_array.sum())
    exact_denominator = int(eligible_array.sum())
    return {
        "task_id": TASK_ID,
        "contrast_id": first.get("contrast_id"),
        "left_method": first.get("left_method"),
        "right_method": first.get("right_method"),
        "record_ids": record_ids,
        "records": records,
        "replicate_ids": [str(key) for key in comparisons],
        "replicate_count": len(comparisons),
        "aggregation": "mean correctness, exact, and paired gain/loss counts within source; seeds are not bootstrap units",
        "metrics": {
            "scored_tokens": int(denominator),
            "left_correct_tokens": float(left_array.sum()),
            "right_correct_tokens": float(right_array.sum()),
            "left_token_accuracy": float(left_array.sum() / denominator) if denominator else None,
            "right_token_accuracy": float(right_array.sum() / denominator) if denominator else None,
            "token_delta_pp": 100.0 * float((left_array - right_array).sum()) / denominator if denominator else None,
            "macro_token_delta_pp": (
                100.0 * float(np.mean((left_array[scored_array > 0] - right_array[scored_array > 0]) / scored_array[scored_array > 0]))
                if np.any(scored_array > 0)
                else None
            ),
            "token_gains": float(sum(row["token_gains"] for row in per_record)),
            "token_losses": float(sum(row["token_losses"] for row in per_record)),
            "token_both_correct": float(sum(row["token_both_correct"] for row in per_record)),
            "token_neither_correct": float(sum(row["token_neither_correct"] for row in per_record)),
            "exact_denominator": exact_denominator,
            "left_exact_records": float(left_exact.sum()),
            "right_exact_records": float(right_exact.sum()),
            "exact_delta_pp": 100.0 * float((left_exact - right_exact).sum()) / exact_denominator if exact_denominator else None,
        },
        "per_record": per_record,
    }


def _validate_replicates(scores_by_seed: Mapping[str | int, Mapping[str, Any]], *, name: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(scores_by_seed, Mapping) or not scores_by_seed:
        raise P08MetricsError(f"{name} has no replicate scores")
    normalized: dict[str, Mapping[str, Any]] = {}
    for raw_seed, score in scores_by_seed.items():
        seed = str(raw_seed)
        if seed in normalized:
            raise P08MetricsError(f"{name} repeats seed {seed}")
        normalized[seed] = _validated_score(score, name=f"{name}/{seed}")
    expected = {str(seed) for seed in REPLICATE_SEEDS}
    if set(normalized) != expected:
        raise P08MetricsError(f"{name} must contain exactly seeds {REPLICATE_SEEDS}")
    return normalized


def _replicate_comparison(
    method_scores: Mapping[str, Mapping[str | int, Mapping[str, Any]]],
    *,
    left_method: str,
    right_method: str,
    contrast_id: str,
) -> dict[str, Any]:
    left = _validate_replicates(method_scores[left_method], name=left_method)
    right = _validate_replicates(method_scores[right_method], name=right_method)
    per_seed: dict[str, Mapping[str, Any]] = {}
    for seed in (str(value) for value in REPLICATE_SEEDS):
        per_seed[seed] = paired_metrics_from_scores(left[seed], right[seed], contrast_id=contrast_id)
    return aggregate_replicate_comparisons(per_seed)


def _interaction_rows(
    staged: Mapping[str, Any],
    joint: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if staged["record_ids"] != joint["record_ids"]:
        raise P08MetricsError("staged and joint contrast source order differs")
    rows = []
    for srow, jrow in zip(staged["per_record"], joint["per_record"]):
        if srow["scored_tokens"] != jrow["scored_tokens"] or srow["exact_eligible"] != jrow["exact_eligible"]:
            raise P08MetricsError("staged and joint contrast denominators differ")
        rows.append(
            {
                "record_id": srow["record_id"],
                "scored_tokens": int(srow["scored_tokens"]),
                "exact_eligible": bool(srow["exact_eligible"]),
                "past_staged_token_delta": float(srow["token_delta"]),
                "past_joint_token_delta": float(jrow["token_delta"]),
                "interaction_token_delta": float(srow["token_delta"] - jrow["token_delta"]),
                "past_staged_token_gains": float(srow["token_gains"]),
                "past_staged_token_losses": float(srow["token_losses"]),
                "past_joint_token_gains": float(jrow["token_gains"]),
                "past_joint_token_losses": float(jrow["token_losses"]),
                "past_staged_exact_delta": (
                    float(srow["left_exact_record"] - srow["right_exact_record"])
                    if srow["exact_eligible"]
                    else None
                ),
                "past_joint_exact_delta": (
                    float(jrow["left_exact_record"] - jrow["right_exact_record"])
                    if jrow["exact_eligible"]
                    else None
                ),
                "interaction_exact_delta": (
                    float(
                        (srow["left_exact_record"] - srow["right_exact_record"])
                        - (jrow["left_exact_record"] - jrow["right_exact_record"])
                    )
                    if srow["exact_eligible"]
                    else None
                ),
            }
        )
    return rows


def _interaction_summary(rows: Sequence[Mapping[str, Any]], *, schedule: np.ndarray | None = None) -> dict[str, Any]:
    if not rows:
        raise P08MetricsError("interaction has no source rows")
    scored = np.asarray([float(row["scored_tokens"]) for row in rows], dtype=np.float64)
    token_delta = np.asarray([float(row["interaction_token_delta"]) for row in rows], dtype=np.float64)
    exact_eligible = np.asarray([bool(row["exact_eligible"]) for row in rows], dtype=bool)
    exact_delta = np.asarray(
        [float(row["interaction_exact_delta"]) if row["exact_eligible"] else 0.0 for row in rows],
        dtype=np.float64,
    )
    if schedule is None:
        schedule = np.arange(len(rows), dtype=np.int64)[None, :]
    schedule = np.asarray(schedule, dtype=np.int64)
    if schedule.ndim != 2 or schedule.shape[1] != len(rows):
        raise P08MetricsError("bootstrap schedule and interaction rows differ")
    draw_token_denominator = scored[schedule].sum(axis=1)
    draw_token = np.divide(
        token_delta[schedule].sum(axis=1),
        draw_token_denominator,
        out=np.full(schedule.shape[0], np.nan),
        where=draw_token_denominator > 0,
    ) * 100.0
    per_record_token = np.divide(token_delta, scored, out=np.full(scored.shape, np.nan), where=scored > 0)
    draw_macro_count = np.isfinite(per_record_token)[schedule].sum(axis=1)
    draw_macro = np.divide(
        np.nansum(per_record_token[schedule], axis=1),
        draw_macro_count,
        out=np.full(schedule.shape[0], np.nan),
        where=draw_macro_count > 0,
    ) * 100.0
    draw_exact_denominator = exact_eligible[schedule].sum(axis=1)
    draw_exact = np.divide(
        exact_delta[schedule].sum(axis=1),
        draw_exact_denominator,
        out=np.full(schedule.shape[0], np.nan),
        where=draw_exact_denominator > 0,
    ) * 100.0

    def percentile(values: np.ndarray) -> list[float | None]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return [None, None]
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    point_denominator = float(scored.sum())
    point_exact_denominator = int(exact_eligible.sum())
    return {
        "records": len(rows),
        "scored_tokens": int(point_denominator),
        "exact_denominator": point_exact_denominator,
        "point": {
            "token_delta_pp": float(100.0 * token_delta.sum() / point_denominator) if point_denominator else None,
            "macro_token_delta_pp": (
                float(100.0 * np.nanmean(per_record_token)) if np.isfinite(per_record_token).any() else None
            ),
            "exact_delta_pp": float(100.0 * exact_delta.sum() / point_exact_denominator) if point_exact_denominator else None,
        },
        "ci95_percentile": {
            "token_delta_pp": percentile(draw_token),
            "macro_token_delta_pp": percentile(draw_macro),
            "exact_delta_pp": percentile(draw_exact),
        },
        "draws_with_exact_observation": int(np.isfinite(draw_exact).sum()),
        "schedule_sha256": _schedule_digest(schedule),
    }


def interaction_from_replicates(
    method_scores: Mapping[str, Mapping[str | int, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compute the P08 primary interaction after within-source replication."""

    missing = [method for method in METHOD_ORDER if method not in method_scores]
    if missing:
        raise P08MetricsError(f"method matrix lacks {missing}")
    staged = _replicate_comparison(
        method_scores,
        left_method=INTERACTION_METHODS["past_staged"],
        right_method=INTERACTION_METHODS["positionwise_staged"],
        contrast_id="past_staged_minus_positionwise_staged",
    )
    joint = _replicate_comparison(
        method_scores,
        left_method=INTERACTION_METHODS["past_joint"],
        right_method=INTERACTION_METHODS["positionwise_joint"],
        contrast_id="past_joint_minus_positionwise_joint",
    )
    rows = _interaction_rows(staged, joint)
    return {
        "task_id": TASK_ID,
        "contrast_id": "staged_interaction_minus_joint_interaction",
        "record_ids": list(staged["record_ids"]),
        "records": len(rows),
        "replicate_ids": list(staged["replicate_ids"]),
        "replicate_count": staged["replicate_count"],
        "aggregation": "average each seed's paired correctness/exact/gain/loss counts within source before the four-arm interaction",
        "pairwise": {
            "past_staged_minus_positionwise_staged": staged,
            "past_joint_minus_positionwise_joint": joint,
        },
        "per_record": rows,
    }


def make_bootstrap_schedule(records: int, *, draws: int, seed: int) -> np.ndarray:
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise P08MetricsError("bootstrap record count must be positive")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise P08MetricsError("bootstrap draw count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise P08MetricsError("bootstrap seed must be an integer")
    return np.asarray(np.random.default_rng(int(seed)).integers(0, records, size=(draws, records), dtype=np.int64))


def _schedule_digest(schedule: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(schedule.shape), "dtype": str(schedule.dtype)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(schedule).tobytes(order="C"))
    return digest.hexdigest()


def bootstrap_interaction(
    interaction: Mapping[str, Any],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    schedule: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return deterministic paired source-cluster CI for one domain/target."""

    rows = interaction.get("per_record")
    if not isinstance(rows, list) or not rows:
        raise P08MetricsError("interaction has no per-record rows")
    if schedule is None:
        schedule = make_bootstrap_schedule(len(rows), draws=draws, seed=seed)
    summary = _interaction_summary(rows, schedule=np.asarray(schedule))
    summary.update({"draws": int(np.asarray(schedule).shape[0]), "seed": int(seed), "unit": "source-record cluster"})
    return summary


def paired_cluster_bootstrap(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap P08 interaction cells with one schedule per domain.

    ``cells`` contains one ``method_scores`` matrix per domain/target.  The
    source-ID order must match across targets within a domain, so the same
    generated source indices are reused for both target conditions.
    """

    if not isinstance(cells, Mapping) or not cells:
        raise P08MetricsError("P08 cells are empty")
    normalized: dict[str, Mapping[str, Any]] = {}
    expected_pairs: set[tuple[str, str]] = set()
    for cell_id, cell in cells.items():
        if not isinstance(cell_id, str) or not isinstance(cell, Mapping):
            raise P08MetricsError("P08 cell is malformed")
        for key in ("domain", "target", "method_scores"):
            if key not in cell:
                raise P08MetricsError(f"{cell_id} lacks {key}")
        domain, target = str(cell["domain"]), str(cell["target"])
        if domain not in DOMAINS or target not in TARGETS:
            raise P08MetricsError(f"{cell_id} has an unregistered domain/target")
        pair = (domain, target)
        if pair in expected_pairs:
            raise P08MetricsError(f"duplicate P08 cell {pair}")
        expected_pairs.add(pair)
        normalized[cell_id] = cell
    expected = {(domain, target) for domain in DOMAINS for target in TARGETS}
    if expected_pairs != expected:
        raise P08MetricsError("P08 requires exactly four domain/target cells")

    rng = np.random.default_rng(int(seed))
    schedules: dict[str, np.ndarray] = {}
    result_cells: dict[str, Any] = {}
    for domain in DOMAINS:
        domain_cells = [cell for cell in normalized.values() if cell["domain"] == domain]
        source_ids: list[str] | None = None
        records: int | None = None
        interactions: list[tuple[str, Mapping[str, Any]]] = []
        for cell_id, cell in normalized.items():
            if cell["domain"] != domain:
                continue
            interaction = interaction_from_replicates(cell["method_scores"])
            ids = list(interaction["record_ids"])
            if source_ids is None:
                source_ids, records = ids, len(ids)
            elif ids != source_ids:
                raise P08MetricsError(f"source-record order differs across {domain} targets")
            interactions.append((cell_id, interaction))
        assert records is not None and source_ids is not None
        schedule = np.asarray(rng.integers(0, records, size=(draws, records), dtype=np.int64))
        schedules[domain] = schedule
        digest = _schedule_digest(schedule)
        for cell_id, interaction in interactions:
            result_cells[cell_id] = {
                "domain": domain,
                "target": normalized[cell_id]["target"],
                "records": records,
                "source_record_ids": source_ids,
                "schedule_sha256": digest,
                "interaction": interaction,
                "bootstrap": bootstrap_interaction(interaction, schedule=schedule, seed=seed, draws=draws),
            }
    return {
        "schema": "token-reconstruction.trr-p08-paired-bootstrap.v1",
        "task_id": TASK_ID,
        "draws": int(draws),
        "seed": int(seed),
        "unit": "source-record cluster",
        "target_resampling": "one source-index schedule reused across paired target conditions within each domain",
        "replicate_aggregation": "fractional correctness/exact/gain/loss counts averaged within source before bootstrap",
        "cells": result_cells,
        "schedule_sha256_by_domain": {domain: _schedule_digest(schedule) for domain, schedule in schedules.items()},
    }


def _ci_pair(summary: Mapping[str, Any], metric: str) -> tuple[float | None, float | None]:
    values = summary.get("ci95_percentile", {}).get(metric)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return (None, None)
    return (values[0], values[1])


def _support_metric(summary: Mapping[str, Any], *, metric: str, margin: float) -> bool:
    point = summary.get("point", {}).get(metric)
    low, _ = _ci_pair(summary, metric)
    return point is not None and point >= margin and low is not None and low > 0


def _harm_metric(summary: Mapping[str, Any], *, metric: str, margin: float) -> bool:
    point = summary.get("point", {}).get(metric)
    _, high = _ci_pair(summary, metric)
    return point is not None and point <= -margin and high is not None and high < 0


def classify_interaction_gate(
    domain_summaries: Mapping[str, Mapping[str, Any]],
    *,
    seed_interactions: Mapping[str, Sequence[float | Mapping[str, float | None] | None]] | None = None,
    token_margin: float = 0.5,
    exact_margin: float = 5.0,
) -> dict[str, Any]:
    """Apply P08 support, ruled-out, then inconclusive gates.

    A seed-direction check is verdict-specific: support rejects a materially
    negative seed, while ruling out benefit rejects a seed at or above the
    positive practical margin.  Tiny mixed signs alone do not force the
    inconclusive label when the practical ruled-out gate is met.
    """

    if set(domain_summaries) != set(DOMAINS):
        raise P08MetricsError(f"gate requires summaries for {DOMAINS}")
    seed_interactions = seed_interactions or {}
    support_cells: dict[str, bool] = {}
    harm_cells: dict[str, bool] = {}
    ruled_out_cells: dict[str, bool] = {}
    for domain, summary in domain_summaries.items():
        token_support = _support_metric(summary, metric="token_delta_pp", margin=token_margin)
        exact_support = _support_metric(summary, metric="exact_delta_pp", margin=exact_margin)
        support_cells[domain] = token_support or exact_support
        harm_cells[domain] = _harm_metric(summary, metric="token_delta_pp", margin=token_margin) or _harm_metric(summary, metric="exact_delta_pp", margin=exact_margin)
        token_ci = _ci_pair(summary, "token_delta_pp")
        exact_ci = _ci_pair(summary, "exact_delta_pp")
        ruled_out_cells[domain] = (
            token_ci[1] is not None
            and exact_ci[1] is not None
            and token_ci[1] < token_margin
            and exact_ci[1] < exact_margin
        )
    support_seed_ok: dict[str, bool] = {}
    ruled_out_seed_ok: dict[str, bool] = {}
    for domain in DOMAINS:
        token_values: list[float] = []
        exact_values: list[float] = []
        for raw_value in seed_interactions.get(domain, ()):
            if raw_value is None:
                continue
            if isinstance(raw_value, Mapping):
                token_value = raw_value.get("token_delta_pp")
                exact_value = raw_value.get("exact_delta_pp")
                if token_value is not None:
                    token_values.append(float(token_value))
                if exact_value is not None:
                    exact_values.append(float(exact_value))
            else:
                token_values.append(float(raw_value))
        # Support rejects a materially negative seed in either reported
        # metric.  Ruling out useful benefit rejects a seed at or above the
        # positive practical margin in either metric.
        support_seed_ok[domain] = all(value > -token_margin for value in token_values) and all(
            value > -exact_margin for value in exact_values
        )
        ruled_out_seed_ok[domain] = all(value < token_margin for value in token_values) and all(
            value < exact_margin for value in exact_values
        )
    support = all(support_cells.values()) and not any(harm_cells.values()) and all(support_seed_ok.values())
    ruled_out = all(ruled_out_cells.values()) and all(ruled_out_seed_ok.values())
    if support:
        disposition = "CONTEXTUAL_STAGING_SUPPORT"
    elif ruled_out:
        disposition = "USEFUL_CONTEXTUAL_BENEFIT_RULED_OUT"
    else:
        disposition = "INCONCLUSIVE"
    return {
        "disposition": disposition,
        "gate_order": ["support", "useful_benefit_ruled_out", "inconclusive"],
        "support_cells": support_cells,
        "harm_cells": harm_cells,
        "ruled_out_cells": ruled_out_cells,
        "support_seed_ok": support_seed_ok,
        "ruled_out_seed_ok": ruled_out_seed_ok,
        "token_margin_pp": float(token_margin),
        "exact_margin_pp": float(exact_margin),
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DOMAINS",
    "INTERACTION_METHODS",
    "METHOD_ORDER",
    "P08MetricsError",
    "POSITION_BINS",
    "REPLICATE_SEEDS",
    "TARGETS",
    "aggregate_replicate_comparisons",
    "bootstrap_interaction",
    "classify_interaction_gate",
    "interaction_from_replicates",
    "make_bootstrap_schedule",
    "paired_cluster_bootstrap",
    "paired_metrics_from_scores",
    "score_method",
]
