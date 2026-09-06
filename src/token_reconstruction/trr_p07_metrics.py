"""Pure TRR-P07 scoring, replicate aggregation, and paired uncertainty.

The module deliberately accepts already materialized prediction/truth arrays and
per-method score records.  It does not load files or select rows.  A caller must
complete the P07 prediction freeze before passing truth to these functions.

P06's two fit seeds are replicate measurements of each source record.  Their
correctness and paired joint-event counts are averaged *within the source*
first; the resulting fractional source rows are then used by one shared
source-record bootstrap schedule for both paired target conditions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np

TASK_ID = "TRR-P07"
SEQUENCE_TOKENS = 128
BOS_TOKEN_ID = 128000
POST_BOS_POSITIONS = tuple(range(1, SEQUENCE_TOKENS))
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 7007
P06_REPLICATE_SEEDS = (6106, 6107)
RETAINED_REPLICATE = "retained"
DOMAINS = ("pile", "finance")
TARGETS = ("public_base", "public_lora_2601")
PANELS = ("p06_panel", "trr0006_subset")
CONTRASTS: dict[str, tuple[str, str]] = {
    "past_minus_reference": (
        "p06_past_only",
        "trr0006_positionwise_reference",
    ),
    "past_minus_diagonal": ("p06_past_only", "p06_positionwise_diagonal"),
    "diagonal_minus_reference": (
        "p06_positionwise_diagonal",
        "trr0006_positionwise_reference",
    ),
    "past_minus_causal": ("p06_past_only", "trr0006_causal_enriched"),
}


class P07MetricsError(ValueError):
    """Raised when a P07 scoring input violates the frozen contract."""


def _array(value: Any, *, name: str, dtype: Any | None = None) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result = np.asarray(value)
        if dtype is not None:
            result = result.astype(dtype, copy=False)
    except Exception as exc:  # pragma: no cover - backend-specific conversion
        raise P07MetricsError(f"{name} is not array-like") from exc
    return np.ascontiguousarray(result)


def _record_ids(record_ids: Sequence[str] | None, records: int) -> tuple[str, ...]:
    if record_ids is None:
        return tuple(f"record-{index}" for index in range(records))
    if len(record_ids) != records:
        raise P07MetricsError("record ID count differs from prediction rows")
    values = tuple(record_ids)
    if any(not isinstance(value, str) or not value for value in values):
        raise P07MetricsError("record IDs must be nonempty strings")
    if len(set(values)) != records:
        raise P07MetricsError("record IDs must be unique")
    return values


def _matrix(value: Any, *, name: str, records: int | None = None) -> np.ndarray:
    result = _array(value, name=name)
    if result.ndim != 2 or result.shape[1] != SEQUENCE_TOKENS:
        raise P07MetricsError(f"{name} must have shape [records, 128]")
    if records is not None and result.shape[0] != records:
        raise P07MetricsError(f"{name} record count differs from predictions")
    if not np.issubdtype(result.dtype, np.integer):
        raise P07MetricsError(f"{name} must contain integer token IDs")
    return result


def _mask(value: Any | None, records: int) -> np.ndarray:
    if value is None:
        result = np.ones((records, SEQUENCE_TOKENS), dtype=bool)
    else:
        result = _array(value, name="attention_mask")
        if result.shape != (records, SEQUENCE_TOKENS):
            raise P07MetricsError("attention_mask must have shape [records, 128]")
        if not np.issubdtype(result.dtype, np.bool_) and not np.isin(result, [0, 1]).all():
            raise P07MetricsError("attention_mask must be binary")
        result = result.astype(bool, copy=False)
    if not result[:, 0].all():
        raise P07MetricsError("every record must contain BOS")
    if (result[:, 1:] > result[:, :-1]).any():
        raise P07MetricsError("attention_mask must be right-padded")
    return result


def _positions(value: Any | None, records: int) -> None:
    if value is None:
        return
    result = _array(value, name="position_ids")
    if result.shape != (records, SEQUENCE_TOKENS) or not np.issubdtype(result.dtype, np.integer):
        raise P07MetricsError("position_ids must have integer shape [records, 128]")
    expected = np.broadcast_to(np.arange(SEQUENCE_TOKENS, dtype=np.int64), result.shape)
    if not np.array_equal(result.astype(np.int64, copy=False), expected):
        raise P07MetricsError("position_ids must be 0..127 in every row")


def score_method(
    predictions: Any,
    truth: Any,
    *,
    record_ids: Sequence[str] | None = None,
    attention_mask: Any | None = None,
    position_ids: Any | None = None,
    method_id: str | None = None,
) -> dict[str, Any]:
    """Score one frozen method on all valid post-BOS positions."""

    prediction_array = _matrix(predictions, name="predictions")
    records = int(prediction_array.shape[0])
    if records <= 0:
        raise P07MetricsError("at least one record is required")
    truth_array = _matrix(truth, name="truth", records=records)
    valid = _mask(attention_mask, records)
    _positions(position_ids, records)
    if (prediction_array[valid] < 0).any() or (truth_array[valid] < 0).any():
        raise P07MetricsError("active token IDs must be nonnegative")
    ids = _record_ids(record_ids, records)

    post_valid = valid.copy()
    post_valid[:, 0] = False
    correct = (prediction_array == truth_array) & post_valid
    scored_per_record = post_valid.sum(axis=1).astype(np.int64)
    correct_per_record = correct.sum(axis=1).astype(np.int64)
    exact_eligible = scored_per_record == len(POST_BOS_POSITIONS)
    exact_record = correct_per_record == scored_per_record

    per_record: list[dict[str, Any]] = []
    for index, record_id in enumerate(ids):
        per_record.append(
            {
                "record_id": record_id,
                "scored_tokens": int(scored_per_record[index]),
                "correct_tokens": int(correct_per_record[index]),
                "token_accuracy": (
                    float(correct_per_record[index] / scored_per_record[index])
                    if scored_per_record[index]
                    else None
                ),
                "exact_eligible": bool(exact_eligible[index]),
                "exact_record": bool(exact_record[index]) if exact_eligible[index] else None,
                "correctness": [bool(item) for item in correct[index, 1:]],
                "valid_post_bos": [bool(item) for item in post_valid[index, 1:]],
            }
        )
    total_scored = int(scored_per_record.sum())
    total_correct = int(correct_per_record.sum())
    exact_denominator = int(exact_eligible.sum())
    exact_records = int((exact_record & exact_eligible).sum())
    return {
        "task_id": TASK_ID,
        "method_id": method_id,
        "records": records,
        "record_ids": list(ids),
        "scored_post_bos_tokens": len(POST_BOS_POSITIONS),
        "metrics": {
            "scored_tokens": total_scored,
            "correct_tokens": total_correct,
            "token_accuracy": total_correct / total_scored if total_scored else None,
            "exact_denominator": exact_denominator,
            "exact_records": exact_records,
            "exact_record_rate": exact_records / exact_denominator if exact_denominator else None,
        },
        "per_record": per_record,
    }


def _validated_score(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P07MetricsError(f"{name} score is not an object")
    records = value.get("records")
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise P07MetricsError(f"{name} score has invalid record count")
    ids = value.get("record_ids")
    if not isinstance(ids, list):
        raise P07MetricsError(f"{name} score has no record IDs")
    normalized_ids = _record_ids(ids, records)
    rows = value.get("per_record")
    if not isinstance(rows, list) or len(rows) != records:
        raise P07MetricsError(f"{name} score has invalid per-record rows")
    for row, expected_id in zip(rows, normalized_ids):
        if not isinstance(row, Mapping) or row.get("record_id") != expected_id:
            raise P07MetricsError(f"{name} score record order changed")
        for key in ("correct_tokens", "scored_tokens"):
            if not isinstance(row.get(key), (int, np.integer)):
                raise P07MetricsError(f"{name} score lacks integer {key}")
        if not (0 <= row["correct_tokens"] <= row["scored_tokens"]):
            raise P07MetricsError(f"{name} score counts are invalid")
        if not isinstance(row.get("exact_eligible"), (bool, np.bool_)):
            raise P07MetricsError(f"{name} score lacks exact eligibility")
        if row["exact_eligible"] and not isinstance(row.get("exact_record"), (bool, np.bool_)):
            raise P07MetricsError(f"{name} score lacks exact flag")
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
            raise P07MetricsError(f"{name} score lacks fixed-width correctness vectors")
        correctness_array = np.asarray(correctness, dtype=bool)
        valid_array = np.asarray(valid, dtype=bool)
        if np.any(correctness_array & ~valid_array):
            raise P07MetricsError(f"{name} marks a padded token correct")
        if int(valid_array.sum()) != int(row["scored_tokens"]):
            raise P07MetricsError(f"{name} mask/count mismatch")
        if int((correctness_array & valid_array).sum()) != int(row["correct_tokens"]):
            raise P07MetricsError(f"{name} correctness/count mismatch")
        eligible = int(row["scored_tokens"]) == len(POST_BOS_POSITIONS)
        if bool(row["exact_eligible"]) != eligible:
            raise P07MetricsError(f"{name} exact eligibility disagrees with denominator")
        if eligible and bool(row["exact_record"]) != bool(correctness_array.all()):
            raise P07MetricsError(f"{name} exact flag disagrees with correctness")
    return dict(value)


def paired_metrics_from_scores(
    left_score: Mapping[str, Any],
    right_score: Mapping[str, Any],
    *,
    contrast_id: str | None = None,
) -> dict[str, Any]:
    """Return integer per-seed paired token/exact counts for two methods."""

    left = _validated_score(left_score, name="left")
    right = _validated_score(right_score, name="right")
    if left["records"] != right["records"] or left["record_ids"] != right["record_ids"]:
        raise P07MetricsError("paired methods do not have identical source-record order")
    records = int(left["records"])
    lrows, rrows = left["per_record"], right["per_record"]
    lcorrect = np.asarray([row["correctness"] for row in lrows], dtype=bool)
    rcorrect = np.asarray([row["correctness"] for row in rrows], dtype=bool)
    valid = np.asarray([row["valid_post_bos"] for row in lrows], dtype=bool)
    rvalid = np.asarray([row["valid_post_bos"] for row in rrows], dtype=bool)
    if not np.array_equal(valid, rvalid):
        raise P07MetricsError("paired methods have different valid-position masks")
    lscored = np.asarray([row["scored_tokens"] for row in lrows], dtype=np.int64)
    rscored = np.asarray([row["scored_tokens"] for row in rrows], dtype=np.int64)
    if not np.array_equal(lscored, rscored):
        raise P07MetricsError("paired methods have different metric denominators")
    eligible = np.asarray([bool(row["exact_eligible"]) for row in lrows], dtype=bool)
    religible = np.asarray([bool(row["exact_eligible"]) for row in rrows], dtype=bool)
    if not np.array_equal(eligible, religible):
        raise P07MetricsError("paired methods have different exact eligibility")
    lexact = np.asarray([bool(row["exact_record"]) if row["exact_eligible"] else False for row in lrows])
    rexact = np.asarray([bool(row["exact_record"]) if row["exact_eligible"] else False for row in rrows])
    gains = lcorrect & ~rcorrect & valid
    losses = ~lcorrect & rcorrect & valid
    both = lcorrect & rcorrect & valid
    neither = ~lcorrect & ~rcorrect & valid
    per_record: list[dict[str, Any]] = []
    for i, record_id in enumerate(left["record_ids"]):
        per_record.append(
            {
                "record_id": record_id,
                "scored_tokens": int(valid[i].sum()),
                "left_correct_tokens": int((lcorrect[i] & valid[i]).sum()),
                "right_correct_tokens": int((rcorrect[i] & valid[i]).sum()),
                "token_gains": int(gains[i].sum()),
                "token_losses": int(losses[i].sum()),
                "token_both_correct": int(both[i].sum()),
                "token_neither_correct": int(neither[i].sum()),
                "exact_eligible": bool(eligible[i]),
                "left_exact_record": bool(lexact[i]) if eligible[i] else None,
                "right_exact_record": bool(rexact[i]) if eligible[i] else None,
                "token_delta": int(gains[i].sum() - losses[i].sum()),
            }
        )
    exact_both = int((lexact & rexact & eligible).sum())
    exact_left_only = int((lexact & ~rexact & eligible).sum())
    exact_right_only = int((~lexact & rexact & eligible).sum())
    exact_neither = int((~lexact & ~rexact & eligible).sum())
    denominator = int(valid.sum())
    left_hits = int((lcorrect & valid).sum())
    right_hits = int((rcorrect & valid).sum())
    exact_denominator = int(eligible.sum())
    left_exact_records = int((lexact & eligible).sum())
    right_exact_records = int((rexact & eligible).sum())
    return {
        "task_id": TASK_ID,
        "contrast_id": contrast_id,
        "left_method": left.get("method_id"),
        "right_method": right.get("method_id"),
        "record_ids": list(left["record_ids"]),
        "records": records,
        "metrics": {
            "scored_tokens": denominator,
            "left_correct_tokens": left_hits,
            "right_correct_tokens": right_hits,
            "left_token_accuracy": left_hits / denominator if denominator else None,
            "right_token_accuracy": right_hits / denominator if denominator else None,
            "token_delta_pp": 100.0 * (left_hits - right_hits) / denominator if denominator else None,
            "token_gains": int(gains.sum()),
            "token_losses": int(losses.sum()),
            "token_both_correct": int(both.sum()),
            "token_neither_correct": int(neither.sum()),
            "exact_denominator": exact_denominator,
            "left_exact_records": left_exact_records,
            "right_exact_records": right_exact_records,
            "exact_delta_pp": 100.0 * (left_exact_records - right_exact_records) / exact_denominator if exact_denominator else None,
            "exact_both": exact_both,
            "exact_left_only": exact_left_only,
            "exact_right_only": exact_right_only,
            "exact_neither": exact_neither,
        },
        "per_record": per_record,
    }


def _mean(values: Sequence[Any]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_replicate_comparisons(
    comparisons: Mapping[str | int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Average joint counts within each source across supplied fit replicates."""

    if not comparisons:
        raise P07MetricsError("at least one replicate comparison is required")
    labels = tuple(str(key) for key in comparisons)
    first = next(iter(comparisons.values()))
    record_ids = list(first["record_ids"])
    records = int(first["records"])
    for comparison in comparisons.values():
        if int(comparison["records"]) != records or list(comparison["record_ids"]) != record_ids:
            raise P07MetricsError("replicate comparison source records differ")
    per_record: list[dict[str, Any]] = []
    for index, record_id in enumerate(record_ids):
        rows = [comparison["per_record"][index] for comparison in comparisons.values()]
        scored = [int(row["scored_tokens"]) for row in rows]
        eligible = [bool(row["exact_eligible"]) for row in rows]
        if len(set(scored)) != 1 or len(set(eligible)) != 1:
            raise P07MetricsError(f"replicate denominator differs for source record {record_id}")
        row: dict[str, Any] = {
            "record_id": record_id,
            "scored_tokens": scored[0],
            "exact_eligible": eligible[0],
        }
        for key in (
            "left_correct_tokens",
            "right_correct_tokens",
            "token_gains",
            "token_losses",
            "token_both_correct",
            "token_neither_correct",
            "token_delta",
        ):
            row[key] = _mean([item[key] for item in rows])
        if eligible[0]:
            row["left_exact_record"] = _mean([float(item["left_exact_record"]) for item in rows])
            row["right_exact_record"] = _mean([float(item["right_exact_record"]) for item in rows])
        else:
            row["left_exact_record"] = None
            row["right_exact_record"] = None
        per_record.append(row)
    scored = np.asarray([row["scored_tokens"] for row in per_record], dtype=np.float64)
    left = np.asarray([row["left_correct_tokens"] for row in per_record], dtype=np.float64)
    right = np.asarray([row["right_correct_tokens"] for row in per_record], dtype=np.float64)
    gains = np.asarray([row["token_gains"] for row in per_record], dtype=np.float64)
    losses = np.asarray([row["token_losses"] for row in per_record], dtype=np.float64)
    eligible = np.asarray([row["exact_eligible"] for row in per_record], dtype=bool)
    lexact = np.asarray([float(row["left_exact_record"]) if row["exact_eligible"] else 0.0 for row in per_record])
    rexact = np.asarray([float(row["right_exact_record"]) if row["exact_eligible"] else 0.0 for row in per_record])
    denominator = float(scored.sum())
    exact_denominator = int(eligible.sum())
    metrics = {
        "scored_tokens": int(denominator),
        "left_correct_tokens": float(left.sum()),
        "right_correct_tokens": float(right.sum()),
        "left_token_accuracy": float(left.sum() / denominator) if denominator else None,
        "right_token_accuracy": float(right.sum() / denominator) if denominator else None,
        "token_delta_pp": float(100.0 * (left.sum() - right.sum()) / denominator) if denominator else None,
        "token_gains": float(gains.sum()),
        "token_losses": float(losses.sum()),
        "token_both_correct": float(sum(row["token_both_correct"] for row in per_record)),
        "token_neither_correct": float(sum(row["token_neither_correct"] for row in per_record)),
        "exact_denominator": exact_denominator,
        "left_exact_records": float(lexact.sum()),
        "right_exact_records": float(rexact.sum()),
        "exact_delta_pp": float(100.0 * (lexact.sum() - rexact.sum()) / exact_denominator) if exact_denominator else None,
        "exact_both": _mean([comparison["metrics"]["exact_both"] for comparison in comparisons.values()]),
        "exact_left_only": _mean([comparison["metrics"]["exact_left_only"] for comparison in comparisons.values()]),
        "exact_right_only": _mean([comparison["metrics"]["exact_right_only"] for comparison in comparisons.values()]),
        "exact_neither": _mean([comparison["metrics"]["exact_neither"] for comparison in comparisons.values()]),
    }
    return {
        "task_id": TASK_ID,
        "contrast_id": first.get("contrast_id"),
        "left_method": first.get("left_method"),
        "right_method": first.get("right_method"),
        "record_ids": record_ids,
        "records": records,
        "replicate_ids": list(labels),
        "replicate_count": len(labels),
        "aggregation": "mean joint correctness counts within each source record; replicates are not bootstrap units",
        "metrics": metrics,
        "per_record": per_record,
    }


def make_bootstrap_schedule(records: int, *, draws: int, seed: int) -> np.ndarray:
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise P07MetricsError("bootstrap record count must be positive")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise P07MetricsError("bootstrap draw count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise P07MetricsError("bootstrap seed must be an integer")
    return np.asarray(np.random.default_rng(int(seed)).integers(0, records, size=(draws, records), dtype=np.int64))


def _schedule_digest(schedule: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(schedule.shape), "dtype": str(schedule.dtype)}, sort_keys=True, separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(schedule).tobytes())
    return digest.hexdigest()


def bootstrap_summary(per_record: Sequence[Mapping[str, Any]], schedule: np.ndarray) -> dict[str, Any]:
    """Compute fractional source-cluster point estimates and percentile CIs."""

    if schedule.ndim != 2 or schedule.shape[1] != len(per_record):
        raise P07MetricsError("bootstrap schedule and per-record geometry differ")
    scored = np.asarray([float(row["scored_tokens"]) for row in per_record])
    left = np.asarray([float(row["left_correct_tokens"]) for row in per_record])
    right = np.asarray([float(row["right_correct_tokens"]) for row in per_record])
    gains = np.asarray([float(row["token_gains"]) for row in per_record])
    losses = np.asarray([float(row["token_losses"]) for row in per_record])
    eligible = np.asarray([bool(row["exact_eligible"]) for row in per_record])
    lexact = np.asarray([float(row["left_exact_record"]) if row["exact_eligible"] else 0.0 for row in per_record])
    rexact = np.asarray([float(row["right_exact_record"]) if row["exact_eligible"] else 0.0 for row in per_record])
    denominator = scored[schedule].sum(axis=1)
    token_delta = np.divide((left - right)[schedule].sum(axis=1), denominator, out=np.full(len(schedule), np.nan), where=denominator > 0) * 100.0
    token_gain = np.divide(gains[schedule].sum(axis=1), denominator, out=np.full(len(schedule), np.nan), where=denominator > 0) * 100.0
    token_loss = np.divide(losses[schedule].sum(axis=1), denominator, out=np.full(len(schedule), np.nan), where=denominator > 0) * 100.0
    exact_denominator = eligible[schedule].sum(axis=1)
    exact_delta = np.divide((lexact - rexact)[schedule].sum(axis=1), exact_denominator, out=np.full(len(schedule), np.nan), where=exact_denominator > 0) * 100.0

    def ci(values: np.ndarray) -> list[float | None]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return [None, None]
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    point_denominator = float(scored.sum())
    point_exact_denominator = int(eligible.sum())
    return {
        "records": len(per_record),
        "scored_tokens": int(point_denominator),
        "exact_denominator": point_exact_denominator,
        "point": {
            "token_delta_pp": float(100.0 * (left.sum() - right.sum()) / point_denominator) if point_denominator else None,
            "token_gain_rate_pp": float(100.0 * gains.sum() / point_denominator) if point_denominator else None,
            "token_loss_rate_pp": float(100.0 * losses.sum() / point_denominator) if point_denominator else None,
            "exact_delta_pp": float(100.0 * (lexact.sum() - rexact.sum()) / point_exact_denominator) if point_exact_denominator else None,
        },
        "ci95_percentile": {
            "token_delta_pp": ci(token_delta),
            "token_gain_rate_pp": ci(token_gain),
            "token_loss_rate_pp": ci(token_loss),
            "exact_delta_pp": ci(exact_delta),
        },
        "draws_with_exact_observation": int(np.isfinite(exact_delta).sum()),
        "schedule_sha256": _schedule_digest(schedule),
    }


def _method_replicates(scores: Mapping[str, Any], method: str) -> dict[str, Mapping[str, Any]]:
    value = scores.get(method)
    if not isinstance(value, Mapping) or not value:
        raise P07MetricsError(f"missing replicate scores for {method}")
    return {str(key): score for key, score in value.items()}


def _contrast_comparisons(scores: Mapping[str, Any], left_method: str, right_method: str) -> dict[str, dict[str, Any]]:
    left = _method_replicates(scores, left_method)
    right = _method_replicates(scores, right_method)
    if set(left) == set(right):
        pairs = [(key, left[key], right[key]) for key in sorted(left)]
    elif set(right) == {RETAINED_REPLICATE}:
        pairs = [(key, left[key], right[RETAINED_REPLICATE]) for key in sorted(left)]
    elif set(left) == {RETAINED_REPLICATE}:
        pairs = [(key, left[RETAINED_REPLICATE], right[key]) for key in sorted(right)]
    else:
        raise P07MetricsError(f"replicate keys cannot be paired for {left_method} versus {right_method}")
    return {
        key: paired_metrics_from_scores(left_score, right_score, contrast_id=f"{left_method}_minus_{right_method}")
        for key, left_score, right_score in pairs
    }


def paired_cluster_bootstrap(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    contrasts: Mapping[str, tuple[str, str]] = CONTRASTS,
) -> dict[str, Any]:
    """Score four contrasts over eight cells with shared source resampling."""

    if not isinstance(cells, Mapping) or not cells:
        raise P07MetricsError("P07 cells are empty")
    normalized: dict[str, Mapping[str, Any]] = {}
    for cell_id, cell in cells.items():
        if not isinstance(cell_id, str) or not isinstance(cell, Mapping):
            raise P07MetricsError("P07 cell is malformed")
        for key in ("panel", "domain", "target", "scores"):
            if key not in cell:
                raise P07MetricsError(f"{cell_id} lacks {key}")
        if cell["panel"] not in PANELS or cell["domain"] not in DOMAINS or cell["target"] not in TARGETS:
            raise P07MetricsError(f"{cell_id} has an unregistered panel/domain/target")
        normalized[cell_id] = cell
    expected = {(panel, domain, target) for panel in PANELS for domain in DOMAINS for target in TARGETS}
    actual = {(str(cell["panel"]), str(cell["domain"]), str(cell["target"])) for cell in normalized.values()}
    if actual != expected:
        raise P07MetricsError("P07 requires exactly eight panel/domain/target cells")

    rng = np.random.default_rng(int(seed))
    result_cells: dict[str, Any] = {}
    schedules: dict[tuple[str, str], np.ndarray] = {}
    for panel in PANELS:
        for domain in DOMAINS:
            target_cells = [cell for cell in normalized.values() if cell["panel"] == panel and cell["domain"] == domain]
            source_ids: list[str] | None = None
            records: int | None = None
            for cell in target_cells:
                score_maps = cell["scores"]
                if not isinstance(score_maps, Mapping):
                    raise P07MetricsError(f"{cell['panel']}/{domain}/{cell['target']} scores are malformed")
                for method in set(sum(([pair[0], pair[1]] for pair in contrasts.values()), [])):
                    for score in _method_replicates(score_maps, method).values():
                        checked = _validated_score(score, name=f"{panel}/{domain}/{cell['target']}/{method}")
                        ids = list(checked["record_ids"])
                        if source_ids is None:
                            source_ids, records = ids, len(ids)
                        elif ids != source_ids:
                            raise P07MetricsError(f"source-record order differs in {panel}/{domain}")
            assert source_ids is not None and records is not None
            schedule = np.asarray(rng.integers(0, records, size=(draws, records), dtype=np.int64))
            schedules[(panel, domain)] = schedule
            schedule_digest = _schedule_digest(schedule)
            for cell in target_cells:
                cell_id = next(key for key, value in normalized.items() if value is cell)
                contrast_results: dict[str, Any] = {}
                for contrast_id, (left_method, right_method) in contrasts.items():
                    per_seed = _contrast_comparisons(cell["scores"], left_method, right_method)
                    aggregate = aggregate_replicate_comparisons(per_seed)
                    contrast_results[contrast_id] = {
                        "left_method": left_method,
                        "right_method": right_method,
                        "per_seed": per_seed,
                        "replicate_averaged": aggregate,
                        "bootstrap": bootstrap_summary(aggregate["per_record"], schedule),
                        "schedule_sha256": schedule_digest,
                    }
                result_cells[cell_id] = {
                    "panel": panel,
                    "domain": domain,
                    "target": cell["target"],
                    "records": records,
                    "source_record_ids": source_ids,
                    "schedule_sha256": schedule_digest,
                    "contrasts": contrast_results,
                }
    return {
        "schema": "token-reconstruction.trr-p07-paired-bootstrap.v1",
        "task_id": TASK_ID,
        "draws": int(draws),
        "seed": int(seed),
        "unit": "source-record cluster",
        "target_resampling": "one source-index schedule reused across both paired targets within each panel/domain",
        "replicate_aggregation": "fractional joint counts averaged within source before bootstrap; seeds are not independent records",
        "cells": result_cells,
        "contrasts": {name: list(pair) for name, pair in contrasts.items()},
    }


def _cell_status(summary: Mapping[str, Any]) -> str:
    point = summary["point"]
    ci = summary["ci95_percentile"]
    token = point.get("token_delta_pp")
    exact = point.get("exact_delta_pp")
    token_ci = ci.get("token_delta_pp")
    exact_ci = ci.get("exact_delta_pp")
    token_support = token is not None and token >= 1.0 and token_ci[0] is not None and token_ci[0] > 0
    exact_support = exact is not None and exact >= 5.0 and exact_ci[0] is not None and exact_ci[0] > 0
    token_harm = token is not None and token <= -1.0 and token_ci[1] is not None and token_ci[1] < 0
    exact_harm = exact is not None and exact <= -5.0 and exact_ci[1] is not None and exact_ci[1] < 0
    if token_harm or exact_harm:
        return "harm"
    if (token_support or exact_support) and not (token_harm or exact_harm):
        return "support"
    return "uncertain"


def coherent_contrast(result: Mapping[str, Any], contrast_id: str) -> dict[str, Any]:
    """Apply the frozen cellwise support/harm/non-reversal interpretation."""

    cells = result.get("cells")
    if not isinstance(cells, Mapping):
        raise P07MetricsError("bootstrap result lacks cells")
    selected = {key: value for key, value in cells.items() if contrast_id in value.get("contrasts", {})}
    if len(selected) != 8:
        raise P07MetricsError(f"contrast {contrast_id} does not cover eight cells")
    statuses = {key: _cell_status(value["contrasts"][contrast_id]["bootstrap"]) for key, value in selected.items()}
    no_harm = all(status != "harm" for status in statuses.values())
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for value in selected.values():
        groups.setdefault((str(value["domain"]), str(value["target"])), []).append(value)
    covered = all(any(statuses[next(key for key, item in selected.items() if item is cell)] == "support" for cell in group) for group in groups.values())
    panel_support = {panel: any(statuses[key] == "support" and value["panel"] == panel for key, value in selected.items()) for panel in PANELS}
    no_material_reversal = True
    for group in groups.values():
        token_points = [value["contrasts"][contrast_id]["bootstrap"]["point"]["token_delta_pp"] for value in group]
        exact_points = [value["contrasts"][contrast_id]["bootstrap"]["point"]["exact_delta_pp"] for value in group]
        if max(token_points) >= 1.0 and min(token_points) <= -1.0:
            no_material_reversal = False
        if max(exact_points) >= 5.0 and min(exact_points) <= -5.0:
            no_material_reversal = False
    coherent = no_harm and covered and all(panel_support.values()) and no_material_reversal
    return {"contrast_id": contrast_id, "cell_status": statuses, "no_harm": no_harm, "domain_target_coverage": covered, "panel_support": panel_support, "no_material_reversal": no_material_reversal, "coherent": coherent}


def classify_gate(result: Mapping[str, Any]) -> dict[str, Any]:
    required = ("past_minus_reference", "diagonal_minus_reference", "past_minus_diagonal")
    checks = {contrast: coherent_contrast(result, contrast) for contrast in required}
    if checks["past_minus_reference"]["coherent"] and checks["diagonal_minus_reference"]["coherent"]:
        disposition = "BOTH_NEW_IMPROVE_FITTING_DEPENDENT"
    elif checks["past_minus_reference"]["coherent"]:
        disposition = "PAST_CANDIDATE_FRESH_CONFIRMATION_ONLY"
    elif checks["past_minus_diagonal"]["coherent"]:
        disposition = "LOCAL_CONTROL_IMPROVEMENT_ONLY"
    else:
        disposition = "PANEL_DEPENDENT_OR_UNCERTAIN"
    return {"disposition": disposition, "checks": checks, "automatic_follow_on": False}


__all__ = [
    "CONTRASTS",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "PANELS",
    "P07MetricsError",
    "P06_REPLICATE_SEEDS",
    "TARGETS",
    "aggregate_replicate_comparisons",
    "bootstrap_summary",
    "classify_gate",
    "coherent_contrast",
    "make_bootstrap_schedule",
    "paired_cluster_bootstrap",
    "paired_metrics_from_scores",
    "score_method",
]
