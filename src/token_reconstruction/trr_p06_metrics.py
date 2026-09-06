"""Pure TRR-P06 scoring and paired source-record uncertainty helpers.

The module contains no file or truth loader.  Callers provide frozen prediction
and label arrays after the pre-truth gate.  It keeps the three visibility arms
and their registered contrasts explicit, scores only valid post-BOS positions,
and reuses one source-record bootstrap index schedule across paired target
conditions within each domain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np


TASK_ID = "TRR-P06"
SEQUENCE_TOKENS = 128
BOS_TOKEN_ID = 128000
VOCABULARY_SIZE = 128256
POST_BOS_POSITIONS = tuple(range(1, SEQUENCE_TOKENS))
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 6306
TRAINING_REPLICATE_SEEDS = (6106, 6107)

METHOD_ORDER = (
    "p06_positionwise_diagonal",
    "p06_past_only",
    "p06_full_record",
)
CONTRASTS = {
    "full_minus_past": ("p06_full_record", "p06_past_only"),
    "past_minus_positionwise": ("p06_past_only", "p06_positionwise_diagonal"),
    "full_minus_positionwise": ("p06_full_record", "p06_positionwise_diagonal"),
}
POSITION_BINS = {
    "early": tuple(range(1, 16)),
    "early_middle": tuple(range(16, 40)),
    "late_middle": tuple(range(40, 80)),
    "near_end": tuple(range(80, 128)),
}


class P06MetricsError(ValueError):
    """Raised when a P06 metric input violates the frozen geometry."""


def _array(value: Any, *, name: str, dtype: Any | None = None) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result = np.asarray(value)
    except Exception as exc:  # pragma: no cover - backend-specific conversion
        raise P06MetricsError(f"{name} is not array-like") from exc
    if dtype is not None:
        try:
            result = result.astype(dtype, copy=False)
        except (TypeError, ValueError) as exc:
            raise P06MetricsError(f"{name} has an incompatible dtype") from exc
    return np.ascontiguousarray(result)


def _record_ids(record_ids: Sequence[str] | None, records: int) -> tuple[str, ...]:
    if record_ids is None:
        return tuple(f"record-{index}" for index in range(records))
    if len(record_ids) != records:
        raise P06MetricsError("record ID count differs from prediction rows")
    values = tuple(record_ids)
    if any(not isinstance(value, str) or not value for value in values):
        raise P06MetricsError("record IDs must be nonempty strings")
    if len(set(values)) != records:
        raise P06MetricsError("record IDs must be unique")
    return values


def _validate_matrix(value: Any, *, name: str, records: int | None = None) -> np.ndarray:
    result = _array(value, name=name)
    if result.ndim != 2 or result.shape[1] != SEQUENCE_TOKENS:
        raise P06MetricsError(f"{name} must have shape [records, 128]")
    if records is not None and result.shape[0] != records:
        raise P06MetricsError(f"{name} record count differs from predictions")
    if not np.issubdtype(result.dtype, np.integer):
        raise P06MetricsError(f"{name} must contain integer IDs")
    return result


def _validate_mask(value: Any | None, records: int) -> np.ndarray:
    if value is None:
        return np.ones((records, SEQUENCE_TOKENS), dtype=bool)
    mask = _array(value, name="attention_mask")
    if mask.ndim != 2 or mask.shape != (records, SEQUENCE_TOKENS):
        raise P06MetricsError("attention_mask must have shape [records, 128]")
    if not np.issubdtype(mask.dtype, np.bool_) and not np.isin(mask, [0, 1]).all():
        raise P06MetricsError("attention_mask must be binary")
    mask = mask.astype(bool, copy=False)
    if not mask[:, 0].all():
        raise P06MetricsError("every record must contain BOS")
    # Once a row is padded, it remains padded.  Invalid activations therefore
    # cannot become an accidental later observation or metric denominator.
    if (mask[:, 1:] > mask[:, :-1]).any():
        raise P06MetricsError("attention_mask must be right padded")
    return mask


def _validate_positions(value: Any | None, records: int) -> None:
    if value is None:
        return
    positions = _array(value, name="position_ids")
    if not np.issubdtype(positions.dtype, np.integer):
        raise P06MetricsError("position_ids must contain integer positions")
    positions = positions.astype(np.int64, copy=False)
    expected = np.broadcast_to(np.arange(SEQUENCE_TOKENS, dtype=np.int64), positions.shape)
    if positions.shape != (records, SEQUENCE_TOKENS) or not np.array_equal(positions, expected):
        raise P06MetricsError("position_ids must be 0..127 in every row")


def _validate_active_ids(ids: np.ndarray, mask: np.ndarray, *, name: str) -> None:
    if not np.all(ids[:, 0] == BOS_TOKEN_ID):
        raise P06MetricsError(f"{name} BOS differs from {BOS_TOKEN_ID}")
    active = mask
    if np.any(ids[active] < 0) or np.any(ids[active] >= VOCABULARY_SIZE):
        raise P06MetricsError(f"{name} active IDs leave the public vocabulary")


def _position_metrics(correct: np.ndarray, active: np.ndarray) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, positions in POSITION_BINS.items():
        index = np.asarray(positions, dtype=np.int64)
        valid = active[:, index]
        values = correct[:, index]
        scored = int(valid.sum())
        hit = int((values & valid).sum())
        result[name] = {
            "positions": list(positions),
            "scored_tokens": scored,
            "correct_tokens": hit,
            "token_accuracy": (hit / scored) if scored else None,
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
    """Score one frozen method on a matrix of 128-token records.

    ``truth`` is an explicit post-gate argument; this function performs no
    resource loading.  Exact clips are eligible only when all 127 post-BOS
    positions are valid.  Shorter rows remain available for descriptive token
    and position-bin metrics but never enter the exact denominator.
    """

    if method_id is not None and method_id not in METHOD_ORDER:
        raise P06MetricsError(f"unknown P06 method: {method_id}")
    prediction_array = _validate_matrix(predictions, name="predictions")
    records = int(prediction_array.shape[0])
    if records <= 0:
        raise P06MetricsError("at least one record is required")
    truth_array = _validate_matrix(truth, name="truth", records=records)
    mask = _validate_mask(attention_mask, records)
    _validate_positions(position_ids, records)
    _validate_active_ids(prediction_array, mask, name="predictions")
    _validate_active_ids(truth_array, mask, name="truth")
    ids = _record_ids(record_ids, records)

    active = mask.copy()
    active[:, 0] = False
    correct = prediction_array == truth_array
    correct &= active
    scored_per_record = active.sum(axis=1).astype(np.int64)
    correct_per_record = correct.sum(axis=1).astype(np.int64)
    exact_eligible = scored_per_record == len(POST_BOS_POSITIONS)
    exact_values = (correct_per_record == scored_per_record) & exact_eligible
    total_scored = int(scored_per_record.sum())
    total_correct = int(correct_per_record.sum())
    nonempty = scored_per_record > 0
    macro_rates = correct_per_record[nonempty] / scored_per_record[nonempty]
    macro_records = int(nonempty.sum())
    exact_denominator = int(exact_eligible.sum())
    exact_records = int(exact_values.sum())
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
                "exact_record": (bool(exact_values[index]) if exact_eligible[index] else None),
                # Fixed-width vectors make paired gain/loss computation
                # deterministic while the companion validity vector excludes
                # padding from every contrast.
                "correctness": [bool(value) for value in correct[index, 1:]],
                "valid_post_bos": [bool(value) for value in active[index, 1:]],
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
            "token_accuracy": (total_correct / total_scored) if total_scored else None,
            "macro_records": macro_records,
            "macro_token_accuracy": (float(macro_rates.mean()) if macro_records else None),
            "exact_records": exact_records,
            "exact_denominator": exact_denominator,
            "exact_record_rate": (exact_records / exact_denominator) if exact_denominator else None,
        },
        "position_metrics": _position_metrics(correct, active),
        "per_record": per_record,
    }


def _validated_score(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P06MetricsError(f"{name} score is not an object")
    records = value.get("records")
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise P06MetricsError(f"{name} score has invalid record count")
    ids = value.get("record_ids")
    if not isinstance(ids, list):
        raise P06MetricsError(f"{name} score has no record IDs")
    normalized_ids = _record_ids(ids, records)
    rows = value.get("per_record")
    if not isinstance(rows, list) or len(rows) != records:
        raise P06MetricsError(f"{name} score has invalid per-record rows")
    for row, expected_id in zip(rows, normalized_ids):
        if not isinstance(row, Mapping) or row.get("record_id") != expected_id:
            raise P06MetricsError(f"{name} score record order changed")
        if not isinstance(row.get("correct_tokens"), (int, np.integer)):
            raise P06MetricsError(f"{name} score lacks correct token counts")
        if not isinstance(row.get("scored_tokens"), (int, np.integer)):
            raise P06MetricsError(f"{name} score lacks scored token counts")
        if row.get("scored_tokens", 0) < 0 or row.get("correct_tokens", 0) < 0 or row.get("correct_tokens", 0) > row.get("scored_tokens", 0):
            raise P06MetricsError(f"{name} score counts are invalid")
        if not isinstance(row.get("exact_eligible"), (bool, np.bool_)):
            raise P06MetricsError(f"{name} score lacks exact eligibility")
        if row.get("exact_eligible") and not isinstance(row.get("exact_record"), (bool, np.bool_)):
            raise P06MetricsError(f"{name} score lacks exact flag")
        correctness = row.get("correctness")
        valid_post_bos = row.get("valid_post_bos")
        if (
            not isinstance(correctness, (list, tuple))
            or not isinstance(valid_post_bos, (list, tuple))
            or len(correctness) != len(POST_BOS_POSITIONS)
            or len(valid_post_bos) != len(POST_BOS_POSITIONS)
            or any(not isinstance(item, (bool, np.bool_)) for item in correctness)
            or any(not isinstance(item, (bool, np.bool_)) for item in valid_post_bos)
        ):
            raise P06MetricsError(f"{name} score lacks fixed-width correctness/mask vectors")
        correctness_array = np.asarray(correctness, dtype=bool)
        valid_array = np.asarray(valid_post_bos, dtype=bool)
        if bool(np.any(correctness_array & ~valid_array)):
            raise P06MetricsError(f"{name} score marks a padded token correct")
        if int(valid_array.sum()) != int(row["scored_tokens"]):
            raise P06MetricsError(f"{name} score mask/count mismatch")
        if int((correctness_array & valid_array).sum()) != int(row["correct_tokens"]):
            raise P06MetricsError(f"{name} score correctness/count mismatch")
        expected_exact_eligible = int(row["scored_tokens"]) == len(POST_BOS_POSITIONS)
        if bool(row["exact_eligible"]) != expected_exact_eligible:
            raise P06MetricsError(f"{name} score exact eligibility disagrees with its denominator")
        if expected_exact_eligible and bool(row["exact_record"]) != bool(correctness_array.all()):
            raise P06MetricsError(f"{name} score exact flag disagrees with correctness vector")
    return dict(value)


def paired_metrics_from_scores(
    left_score: Mapping[str, Any],
    right_score: Mapping[str, Any],
    *,
    contrast_id: str | None = None,
) -> dict[str, Any]:
    """Compare two already-scored methods on identical source rows."""

    left = _validated_score(left_score, name="left")
    right = _validated_score(right_score, name="right")
    if left["records"] != right["records"] or left["record_ids"] != right["record_ids"]:
        raise P06MetricsError("paired methods do not have identical source-record order")
    left_rows = left["per_record"]
    right_rows = right["per_record"]
    records = int(left["records"])
    left_correct = np.asarray([int(row["correct_tokens"]) for row in left_rows], dtype=np.int64)
    right_correct = np.asarray([int(row["correct_tokens"]) for row in right_rows], dtype=np.int64)
    scored = np.asarray([int(row["scored_tokens"]) for row in left_rows], dtype=np.int64)
    right_scored = np.asarray([int(row["scored_tokens"]) for row in right_rows], dtype=np.int64)
    if not np.array_equal(scored, right_scored):
        raise P06MetricsError("paired methods have different metric denominators")
    left_exact_eligible = np.asarray([bool(row["exact_eligible"]) for row in left_rows])
    right_exact_eligible = np.asarray([bool(row["exact_eligible"]) for row in right_rows])
    if not np.array_equal(left_exact_eligible, right_exact_eligible):
        raise P06MetricsError("paired methods have different exact eligibility")
    left_exact = np.asarray([bool(row["exact_record"]) if row["exact_eligible"] else False for row in left_rows])
    right_exact = np.asarray([bool(row["exact_record"]) if row["exact_eligible"] else False for row in right_rows])

    left_correctness = np.asarray([row["correctness"] for row in left_rows], dtype=bool)
    right_correctness = np.asarray([row["correctness"] for row in right_rows], dtype=bool)
    left_valid = np.asarray([row["valid_post_bos"] for row in left_rows], dtype=bool)
    right_valid = np.asarray([row["valid_post_bos"] for row in right_rows], dtype=bool)
    if left_correctness.shape != (records, len(POST_BOS_POSITIONS)) or right_correctness.shape != left_correctness.shape:
        raise P06MetricsError("paired correctness vectors must be [records, 127]")
    if not np.array_equal(left_valid, right_valid):
        raise P06MetricsError("paired methods have different valid-position masks")
    valid = left_valid
    gains = left_correctness & ~right_correctness & valid
    losses = ~left_correctness & right_correctness & valid
    both = left_correctness & right_correctness & valid
    neither = ~left_correctness & ~right_correctness & valid
    exact_eligible_count = int(left_exact_eligible.sum())
    exact_both = int((left_exact & right_exact & left_exact_eligible).sum())
    exact_left_only = int((left_exact & ~right_exact & left_exact_eligible).sum())
    exact_right_only = int((~left_exact & right_exact & left_exact_eligible).sum())
    exact_neither = int((~left_exact & ~right_exact & left_exact_eligible).sum())

    per_record: list[dict[str, Any]] = []
    for index, record_id in enumerate(left["record_ids"]):
        record_valid = valid[index]
        per_record.append(
            {
                "record_id": record_id,
                "scored_tokens": int(record_valid.sum()),
                "left_correct_tokens": int((left_correctness[index] & record_valid).sum()),
                "right_correct_tokens": int((right_correctness[index] & record_valid).sum()),
                "token_delta": int((gains[index] & record_valid).sum() - (losses[index] & record_valid).sum()),
                "token_gains": int(gains[index].sum()),
                "token_losses": int(losses[index].sum()),
                "exact_eligible": bool(left_exact_eligible[index]),
                "left_exact_record": (bool(left_exact[index]) if left_exact_eligible[index] else None),
                "right_exact_record": (bool(right_exact[index]) if right_exact_eligible[index] else None),
            }
        )

    position_metrics: dict[str, dict[str, Any]] = {}
    for name, positions in POSITION_BINS.items():
        local = np.asarray([position - 1 for position in positions], dtype=np.int64)
        local_valid = valid[:, local]
        local_gains = gains[:, local]
        local_losses = losses[:, local]
        local_both = both[:, local]
        local_neither = neither[:, local]
        left_hits = (left_correctness[:, local] & local_valid).sum()
        right_hits = (right_correctness[:, local] & local_valid).sum()
        denominator = int(local_valid.sum())
        position_metrics[name] = {
            "positions": list(positions),
            "scored_tokens": denominator,
            "left_correct_tokens": int(left_hits),
            "right_correct_tokens": int(right_hits),
            "token_gains": int(local_gains.sum()),
            "token_losses": int(local_losses.sum()),
            "both_correct": int(local_both.sum()),
            "neither_correct": int(local_neither.sum()),
            "left_token_accuracy": (float(left_hits / denominator) if denominator else None),
            "right_token_accuracy": (float(right_hits / denominator) if denominator else None),
            "delta_pp": (float(100.0 * (left_hits - right_hits) / denominator) if denominator else None),
        }

    left_total = int((left_correctness & valid).sum())
    right_total = int((right_correctness & valid).sum())
    denominator = int(valid.sum())
    macro_valid = scored > 0
    macro_left = np.divide(
        left_correct, scored, out=np.full(scored.shape, np.nan, dtype=np.float64), where=macro_valid
    )
    macro_right = np.divide(
        right_correct, scored, out=np.full(scored.shape, np.nan, dtype=np.float64), where=macro_valid
    )
    macro_records = int(macro_valid.sum())
    return {
        "task_id": TASK_ID,
        "contrast_id": contrast_id,
        "left_method": left.get("method_id"),
        "right_method": right.get("method_id"),
        "record_ids": list(left["record_ids"]),
        "records": records,
        "metrics": {
            "scored_tokens": denominator,
            "left_correct_tokens": left_total,
            "right_correct_tokens": right_total,
            "left_token_accuracy": (left_total / denominator) if denominator else None,
            "right_token_accuracy": (right_total / denominator) if denominator else None,
            "token_delta_pp": (100.0 * (left_total - right_total) / denominator) if denominator else None,
            "macro_records": macro_records,
            "left_macro_token_accuracy": (float(macro_left[macro_valid].mean()) if macro_records else None),
            "right_macro_token_accuracy": (float(macro_right[macro_valid].mean()) if macro_records else None),
            "macro_token_delta_pp": (float(100.0 * (macro_left[macro_valid] - macro_right[macro_valid]).mean()) if macro_records else None),
            "token_gains": int(gains.sum()),
            "token_losses": int(losses.sum()),
            "token_both_correct": int(both.sum()),
            "token_neither_correct": int(neither.sum()),
            "exact_denominator": exact_eligible_count,
            "left_exact_records": int((left_exact & left_exact_eligible).sum()),
            "right_exact_records": int((right_exact & right_exact_eligible).sum()),
            "exact_delta_pp": (
                100.0 * (int((left_exact & left_exact_eligible).sum()) - int((right_exact & right_exact_eligible).sum())) / exact_eligible_count
                if exact_eligible_count
                else None
            ),
            "exact_both": exact_both,
            "exact_left_only": exact_left_only,
            "exact_right_only": exact_right_only,
            "exact_neither": exact_neither,
        },
        "position_metrics": position_metrics,
        "per_record": per_record,
    }


def paired_metrics(
    left_predictions: Any,
    right_predictions: Any,
    truth: Any,
    *,
    left_method: str,
    right_method: str,
    record_ids: Sequence[str] | None = None,
    attention_mask: Any | None = None,
    position_ids: Any | None = None,
    contrast_id: str | None = None,
) -> dict[str, Any]:
    """Score and compare two methods without loading any artifacts."""

    left = score_method(
        left_predictions,
        truth,
        record_ids=record_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        method_id=left_method,
    )
    right = score_method(
        right_predictions,
        truth,
        record_ids=record_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        method_id=right_method,
    )
    return paired_metrics_from_scores(left, right, contrast_id=contrast_id)


def make_bootstrap_schedule(records: int, *, draws: int, seed: int) -> np.ndarray:
    """Return a deterministic source-record resampling schedule."""

    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise P06MetricsError("bootstrap record count must be positive")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise P06MetricsError("bootstrap draw count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise P06MetricsError("bootstrap seed must be an integer")
    generator = np.random.default_rng(int(seed))
    return np.asarray(generator.integers(0, records, size=(draws, records), dtype=np.int64), order="C")


def _schedule_digest(schedule: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(schedule.shape), "dtype": str(schedule.dtype)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(schedule).tobytes(order="C"))
    return digest.hexdigest()


def _bootstrap_summary(
    rows: Sequence[Mapping[str, Any]],
    schedule: np.ndarray,
) -> dict[str, Any]:
    if schedule.ndim != 2 or schedule.shape[1] != len(rows):
        raise P06MetricsError("bootstrap row/schedule geometry differs")
    left_correct = np.asarray([float(row["left_correct_tokens"]) for row in rows], dtype=np.float64)
    right_correct = np.asarray([float(row["right_correct_tokens"]) for row in rows], dtype=np.float64)
    scored = np.asarray([float(row["scored_tokens"]) for row in rows], dtype=np.float64)
    left_gains = np.asarray([float(row["token_gains"]) for row in rows], dtype=np.float64)
    left_losses = np.asarray([float(row["token_losses"]) for row in rows], dtype=np.float64)
    exact_eligible = np.asarray([bool(row["exact_eligible"]) for row in rows])
    right_exact = np.asarray([float(row["right_exact_record"]) if row["exact_eligible"] else 0.0 for row in rows], dtype=np.float64)
    left_exact = np.asarray([float(row["left_exact_record"]) if row["exact_eligible"] else 0.0 for row in rows], dtype=np.float64)

    denominator = scored[schedule].sum(axis=1)
    token_delta = np.divide(
        left_correct[schedule].sum(axis=1) - right_correct[schedule].sum(axis=1),
        denominator,
        out=np.full(schedule.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    per_record_macro_delta = np.divide(
        left_correct - right_correct,
        scored,
        out=np.full(scored.shape, np.nan, dtype=np.float64),
        where=scored > 0,
    )
    macro_draw_counts = np.isfinite(per_record_macro_delta)[schedule].sum(axis=1)
    macro_delta = np.divide(
        np.nansum(per_record_macro_delta[schedule], axis=1),
        macro_draw_counts,
        out=np.full(schedule.shape[0], np.nan, dtype=np.float64),
        where=macro_draw_counts > 0,
    )
    token_gain_rate = np.divide(
        left_gains[schedule].sum(axis=1),
        denominator,
        out=np.full(schedule.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    token_loss_rate = np.divide(
        left_losses[schedule].sum(axis=1),
        denominator,
        out=np.full(schedule.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    exact_denominator = exact_eligible[schedule].sum(axis=1)
    # Primary P06 rows are all exact eligible.  A conditional calculation keeps
    # the helper honest for public padding/ends diagnostics as well.
    exact_delta = np.divide(
        left_exact[schedule].sum(axis=1) - right_exact[schedule].sum(axis=1),
        exact_denominator,
        out=np.full(schedule.shape[0], np.nan, dtype=np.float64),
        where=exact_denominator > 0,
    )

    def percentile(values: np.ndarray) -> list[float | None]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return [None, None]
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    def percentile_pp(values: np.ndarray) -> list[float | None]:
        return [None if value is None else 100.0 * value for value in percentile(values)]

    point_denominator = float(scored.sum())
    point_exact_denominator = int(exact_eligible.sum())
    point = {
        "token_delta_pp": float(100.0 * (left_correct.sum() - right_correct.sum()) / point_denominator) if point_denominator else None,
        "macro_token_delta_pp": (float(100.0 * np.nanmean(per_record_macro_delta)) if np.isfinite(per_record_macro_delta).any() else None),
        "token_gain_rate_pp": float(100.0 * left_gains.sum() / point_denominator) if point_denominator else None,
        "token_loss_rate_pp": float(100.0 * left_losses.sum() / point_denominator) if point_denominator else None,
        "exact_delta_pp": float(100.0 * (left_exact.sum() - right_exact.sum()) / point_exact_denominator) if point_exact_denominator else None,
    }
    return {
        "records": len(rows),
        "scored_tokens": int(scored.sum()),
        "exact_denominator": point_exact_denominator,
        "point": point,
        "token_delta_ci95_percentile_pp": percentile_pp(token_delta),
        "macro_token_delta_ci95_percentile_pp": percentile_pp(macro_delta),
        "token_gain_rate_ci95_percentile_pp": percentile_pp(token_gain_rate),
        "token_loss_rate_ci95_percentile_pp": percentile_pp(token_loss_rate),
        "exact_delta_ci95_percentile_pp": percentile_pp(exact_delta),
        "draws_with_exact_observation": int(np.isfinite(exact_delta).sum()),
    }


def _normalise_replicate_scores(
    cell_id: str,
    raw: Mapping[str, Any],
    *,
    contrasts: Mapping[str, tuple[str, str]],
) -> dict[int, dict[str, dict[str, Any]]]:
    """Validate the two registered fit replicates for one target cell."""

    replicates = raw.get("replicates")
    if not isinstance(replicates, Mapping):
        raise P06MetricsError(
            f"{cell_id} must provide score replicates for seeds {TRAINING_REPLICATE_SEEDS}; "
            "a single methods score is not a registered P06 result"
        )
    normalized: dict[int, dict[str, dict[str, Any]]] = {}
    for raw_seed, seed_methods in replicates.items():
        if isinstance(raw_seed, bool):
            raise P06MetricsError(f"{cell_id} has a boolean replicate seed")
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise P06MetricsError(f"{cell_id} has a non-integer replicate seed") from exc
        if seed in normalized:
            raise P06MetricsError(f"{cell_id} repeats replicate seed {seed}")
        if not isinstance(seed_methods, Mapping):
            raise P06MetricsError(f"{cell_id}/{seed} replicate methods are malformed")
        methods: dict[str, dict[str, Any]] = {}
        for method_id in METHOD_ORDER:
            if method_id not in seed_methods:
                raise P06MetricsError(f"{cell_id}/{seed} lacks {method_id}")
            methods[method_id] = _validated_score(
                seed_methods[method_id], name=f"{cell_id}/{seed}/{method_id}"
            )
        # Also validate custom contrast method IDs before any score is used.
        for contrast_id, (left_method, right_method) in contrasts.items():
            if left_method not in methods or right_method not in methods:
                raise P06MetricsError(
                    f"contrast {contrast_id} references an unregistered method"
                )
        normalized[seed] = methods
    expected = set(TRAINING_REPLICATE_SEEDS)
    if set(normalized) != expected:
        raise P06MetricsError(
            f"{cell_id} must contain exactly replicate seeds {TRAINING_REPLICATE_SEEDS}"
        )
    source_ids: list[str] | None = None
    for seed in TRAINING_REPLICATE_SEEDS:
        for method_id in METHOD_ORDER:
            ids = normalized[seed][method_id]["record_ids"]
            if source_ids is None:
                source_ids = list(ids)
            elif ids != source_ids:
                raise P06MetricsError(
                    f"source-record order differs within {cell_id}/{seed}/{method_id}"
                )
    return normalized


def _mean_numeric(values: Sequence[Any]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _aggregate_seed_comparisons(
    comparisons: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Average paired *metrics* within each source record across two seeds.

    This intentionally consumes paired count/discordance summaries rather than
    token IDs or correctness vectors.  Seeds therefore remain replicate
    measurements of each source record, never extra bootstrap clusters.
    """

    seeds = tuple(sorted(comparisons))
    if seeds != tuple(sorted(TRAINING_REPLICATE_SEEDS)):
        raise P06MetricsError("replicate comparison set is not the registered two-seed set")
    first = comparisons[seeds[0]]
    record_ids = list(first["record_ids"])
    records = int(first["records"])
    for seed in seeds[1:]:
        comparison = comparisons[seed]
        if comparison["records"] != records or comparison["record_ids"] != record_ids:
            raise P06MetricsError("replicate comparison source records differ")

    per_record: list[dict[str, Any]] = []
    for index, record_id in enumerate(record_ids):
        rows = [comparisons[seed]["per_record"][index] for seed in seeds]
        scored = [int(row["scored_tokens"]) for row in rows]
        exact_eligible = [bool(row["exact_eligible"]) for row in rows]
        if len(set(scored)) != 1 or len(set(exact_eligible)) != 1:
            raise P06MetricsError(
                f"replicate denominator/eligibility differs for source record {record_id}"
            )
        left_exact = (
            _mean_numeric([float(row["left_exact_record"]) for row in rows])
            if exact_eligible[0]
            else None
        )
        right_exact = (
            _mean_numeric([float(row["right_exact_record"]) for row in rows])
            if exact_eligible[0]
            else None
        )
        token_gains = _mean_numeric([row["token_gains"] for row in rows])
        token_losses = _mean_numeric([row["token_losses"] for row in rows])
        per_record.append(
            {
                "record_id": record_id,
                "scored_tokens": scored[0],
                "left_correct_tokens": _mean_numeric(
                    [row["left_correct_tokens"] for row in rows]
                ),
                "right_correct_tokens": _mean_numeric(
                    [row["right_correct_tokens"] for row in rows]
                ),
                "token_delta": token_gains - token_losses,
                "token_gains": token_gains,
                "token_losses": token_losses,
                "exact_eligible": exact_eligible[0],
                "left_exact_record": left_exact,
                "right_exact_record": right_exact,
            }
        )

    scored = np.asarray([row["scored_tokens"] for row in per_record], dtype=np.float64)
    left_correct = np.asarray(
        [row["left_correct_tokens"] for row in per_record], dtype=np.float64
    )
    right_correct = np.asarray(
        [row["right_correct_tokens"] for row in per_record], dtype=np.float64
    )
    gains = np.asarray([row["token_gains"] for row in per_record], dtype=np.float64)
    losses = np.asarray([row["token_losses"] for row in per_record], dtype=np.float64)
    exact_eligible = np.asarray(
        [bool(row["exact_eligible"]) for row in per_record], dtype=bool
    )
    left_exact = np.asarray(
        [float(row["left_exact_record"]) if row["exact_eligible"] else 0.0 for row in per_record],
        dtype=np.float64,
    )
    right_exact = np.asarray(
        [float(row["right_exact_record"]) if row["exact_eligible"] else 0.0 for row in per_record],
        dtype=np.float64,
    )
    denominator = float(scored.sum())
    macro_valid = scored > 0
    left_macro = np.divide(
        left_correct, scored, out=np.full(scored.shape, np.nan), where=macro_valid
    )
    right_macro = np.divide(
        right_correct, scored, out=np.full(scored.shape, np.nan), where=macro_valid
    )
    exact_denominator = int(exact_eligible.sum())
    left_exact_total = float(left_exact.sum())
    right_exact_total = float(right_exact.sum())
    exact_both = _mean_numeric(
        [comparison["metrics"]["exact_both"] for comparison in comparisons.values()]
    )
    exact_left_only = _mean_numeric(
        [comparison["metrics"]["exact_left_only"] for comparison in comparisons.values()]
    )
    exact_right_only = _mean_numeric(
        [comparison["metrics"]["exact_right_only"] for comparison in comparisons.values()]
    )
    exact_neither = _mean_numeric(
        [comparison["metrics"]["exact_neither"] for comparison in comparisons.values()]
    )
    metrics = {
        "scored_tokens": int(denominator),
        "left_correct_tokens": float(left_correct.sum()),
        "right_correct_tokens": float(right_correct.sum()),
        "left_token_accuracy": (
            float(left_correct.sum() / denominator) if denominator else None
        ),
        "right_token_accuracy": (
            float(right_correct.sum() / denominator) if denominator else None
        ),
        "token_delta_pp": (
            float(100.0 * (left_correct.sum() - right_correct.sum()) / denominator)
            if denominator
            else None
        ),
        "macro_records": int(macro_valid.sum()),
        "left_macro_token_accuracy": (
            float(np.nanmean(left_macro)) if macro_valid.any() else None
        ),
        "right_macro_token_accuracy": (
            float(np.nanmean(right_macro)) if macro_valid.any() else None
        ),
        "macro_token_delta_pp": (
            float(100.0 * np.nanmean(left_macro - right_macro))
            if macro_valid.any()
            else None
        ),
        "token_gains": float(gains.sum()),
        "token_losses": float(losses.sum()),
        "token_both_correct": _mean_numeric(
            [comparison["metrics"]["token_both_correct"] for comparison in comparisons.values()]
        ),
        "token_neither_correct": _mean_numeric(
            [comparison["metrics"]["token_neither_correct"] for comparison in comparisons.values()]
        ),
        "exact_denominator": exact_denominator,
        "left_exact_records": left_exact_total,
        "right_exact_records": right_exact_total,
        "exact_delta_pp": (
            float(100.0 * (left_exact_total - right_exact_total) / exact_denominator)
            if exact_denominator
            else None
        ),
        "exact_both": exact_both,
        "exact_left_only": exact_left_only,
        "exact_right_only": exact_right_only,
        "exact_neither": exact_neither,
    }

    position_metrics: dict[str, dict[str, Any]] = {}
    for position_name in POSITION_BINS:
        bins = [comparison["position_metrics"][position_name] for comparison in comparisons.values()]
        scored_bin = int(bins[0]["scored_tokens"])
        if any(int(item["scored_tokens"]) != scored_bin for item in bins):
            raise P06MetricsError(f"replicate position denominator differs in {position_name}")
        left_bin = _mean_numeric([item["left_correct_tokens"] for item in bins])
        right_bin = _mean_numeric([item["right_correct_tokens"] for item in bins])
        position_metrics[position_name] = {
            "positions": list(POSITION_BINS[position_name]),
            "scored_tokens": scored_bin,
            "left_correct_tokens": left_bin,
            "right_correct_tokens": right_bin,
            "token_gains": _mean_numeric([item["token_gains"] for item in bins]),
            "token_losses": _mean_numeric([item["token_losses"] for item in bins]),
            "both_correct": _mean_numeric([item["both_correct"] for item in bins]),
            "neither_correct": _mean_numeric([item["neither_correct"] for item in bins]),
            "left_token_accuracy": (left_bin / scored_bin if scored_bin else None),
            "right_token_accuracy": (right_bin / scored_bin if scored_bin else None),
            "delta_pp": (
                100.0 * (left_bin - right_bin) / scored_bin if scored_bin else None
            ),
        }
    return {
        "task_id": TASK_ID,
        "contrast_id": first.get("contrast_id"),
        "left_method": first.get("left_method"),
        "right_method": first.get("right_method"),
        "record_ids": record_ids,
        "records": records,
        "replicate_ids": [str(seed) for seed in seeds],
        "replicate_count": len(seeds),
        "aggregation": "mean within each source record; replicates are not bootstrap records",
        "metrics": metrics,
        "position_metrics": position_metrics,
        "per_record": per_record,
    }


def paired_cluster_bootstrap(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    contrasts: Mapping[str, tuple[str, str]] = CONTRASTS,
) -> dict[str, Any]:
    """Bootstrap registered contrasts after averaging the two fit replicates.

    ``cells`` maps one target cell per domain/target to ``replicates`` keyed by
    seeds 6106 and 6107.  Every seed has one score for each of the three arms.
    Seed summaries are retained, but their per-source counts/discordances are
    averaged before one shared source-record schedule is applied to both target
    conditions.  No token IDs or seed predictions are averaged here.
    """

    if not isinstance(cells, Mapping) or not cells:
        raise P06MetricsError("bootstrap cells are empty")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise P06MetricsError("bootstrap draw count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise P06MetricsError("bootstrap seed must be an integer")

    normalized: dict[str, dict[str, Any]] = {}
    for cell_id, raw in cells.items():
        if not isinstance(cell_id, str) or not cell_id or not isinstance(raw, Mapping):
            raise P06MetricsError("bootstrap cell is malformed")
        domain = raw.get("domain")
        target = raw.get("target")
        if (
            not isinstance(domain, str)
            or not domain
            or not isinstance(target, str)
            or not target
        ):
            raise P06MetricsError(f"bootstrap cell lacks domain/target: {cell_id}")
        normalized[cell_id] = {
            "domain": domain,
            "target": target,
            "replicates": _normalise_replicate_scores(
                cell_id, raw, contrasts=contrasts
            ),
        }

    rng = np.random.default_rng(int(seed))
    by_domain: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for cell_id in sorted(normalized):
        cell = normalized[cell_id]
        by_domain.setdefault(cell["domain"], []).append((cell_id, cell))

    domain_results: dict[str, Any] = {}
    for domain in sorted(by_domain):
        domain_cells = by_domain[domain]
        target_seen: dict[str, dict[str, Any]] = {}
        source_ids: list[str] | None = None
        records: int | None = None
        for cell_id, cell in domain_cells:
            target = cell["target"]
            if target in target_seen:
                raise P06MetricsError(f"duplicate target cell in domain {domain}: {target}")
            target_seen[target] = cell
            for replicate in TRAINING_REPLICATE_SEEDS:
                for method_id in METHOD_ORDER:
                    score = cell["replicates"][replicate][method_id]
                    ids = list(score["record_ids"])
                    if source_ids is None:
                        source_ids = ids
                        records = len(ids)
                    elif ids != source_ids:
                        raise P06MetricsError(
                            f"source-record order differs across paired cells: {cell_id}/{replicate}/{method_id}"
                        )
        if source_ids is None or records is None:
            raise P06MetricsError(f"domain {domain} has no score cells")
        expected_targets = {"public_base", "public_lora_2601"}
        if set(target_seen) != expected_targets:
            raise P06MetricsError(
                f"domain {domain} must contain exactly targets {sorted(expected_targets)}"
            )
        schedule = np.asarray(
            rng.integers(0, records, size=(draws, records), dtype=np.int64), order="C"
        )
        schedule_digest = _schedule_digest(schedule)
        target_results: dict[str, Any] = {}
        for target, cell in sorted(target_seen.items()):
            seed_comparisons: dict[str, dict[str, dict[str, Any]]] = {}
            for replicate in TRAINING_REPLICATE_SEEDS:
                methods = cell["replicates"][replicate]
                seed_comparisons[str(replicate)] = {}
                for contrast_id, (left_method, right_method) in contrasts.items():
                    seed_comparisons[str(replicate)][contrast_id] = paired_metrics_from_scores(
                        methods[left_method], methods[right_method], contrast_id=contrast_id
                    )
            contrast_results: dict[str, Any] = {}
            for contrast_id in contrasts:
                comparisons = {
                    int(replicate): seed_comparisons[str(replicate)][contrast_id]
                    for replicate in TRAINING_REPLICATE_SEEDS
                }
                aggregate = _aggregate_seed_comparisons(comparisons)
                contrast_results[contrast_id] = {
                    "left_method": aggregate["left_method"],
                    "right_method": aggregate["right_method"],
                    "replicate_ids": aggregate["replicate_ids"],
                    "replicate_count": aggregate["replicate_count"],
                    "aggregation": aggregate["aggregation"],
                    "per_seed": {
                        replicate: seed_comparisons[replicate][contrast_id]
                        for replicate in sorted(seed_comparisons)
                    },
                    "replicate_averaged": aggregate,
                    "schedule_sha256": schedule_digest,
                    **_bootstrap_summary(aggregate["per_record"], schedule),
                }
            target_results[target] = {
                "replicate_ids": [str(seed) for seed in TRAINING_REPLICATE_SEEDS],
                "replicate_count": len(TRAINING_REPLICATE_SEEDS),
                "schedule_sha256": schedule_digest,
                "contrasts": contrast_results,
            }
        domain_results[domain] = {
            "records": records,
            "source_record_ids": source_ids,
            "target_conditions": sorted(target_seen),
            "schedule_shared_across_targets": True,
            "schedule_shape": list(schedule.shape),
            "schedule_sha256": schedule_digest,
            "replicate_ids": [str(seed) for seed in TRAINING_REPLICATE_SEEDS],
            "replicate_count": len(TRAINING_REPLICATE_SEEDS),
            "targets": target_results,
        }
    return {
        "schema": "token-reconstruction.trr-p06-paired-bootstrap.v2",
        "task_id": TASK_ID,
        "draws": int(draws),
        "seed": int(seed),
        "training_replicate_seeds": list(TRAINING_REPLICATE_SEEDS),
        "unit": "source-record cluster",
        "strata": "domain-only",
        "target_resampling": "one source-index schedule reused across target conditions within each domain",
        "replicate_aggregation": "mean per source record before bootstrap; seeds are not independent records",
        "domains": domain_results,
        "contrasts": {name: list(pair) for name, pair in contrasts.items()},
    }


__all__ = [
    "BOS_TOKEN_ID",
    "CONTRASTS",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "METHOD_ORDER",
    "P06MetricsError",
    "POSITION_BINS",
    "SEQUENCE_TOKENS",
    "TRAINING_REPLICATE_SEEDS",
    "make_bootstrap_schedule",
    "paired_cluster_bootstrap",
    "paired_metrics",
    "paired_metrics_from_scores",
    "score_method",
]
