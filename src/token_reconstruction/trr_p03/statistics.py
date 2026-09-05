"""Deterministic paired-record statistics for the TRR-P03 panel.

The P03 panel has unequal record lengths, so a record-level mean and a
token-level (micro) accuracy answer different questions.  This module keeps
those estimands separate and uses the same length-stratified resample for
both methods in every bootstrap draw.  The default draw count and seed are
the preregistered Stage 1 values.

The minimum input is one row per common record with the following fields:
``record_id``, ``length`` (post-BOS scored-token count), ``projected_correct``,
``a1_correct``, ``projected_exact``, and ``a1_exact``.  Optional paired
correctness vectors can be supplied when position-level gains and losses are
needed.  Counts alone cannot identify whether the methods were correct at the
same positions, so the module reports count changes separately from position
changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np


STATISTICS_SCHEMA = "token-reconstruction.trr-p03-paired-record-statistics.v1"
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_905


class StatisticsError(ValueError):
    """Raised when paired-record statistics cannot be computed safely."""


@dataclass(frozen=True)
class PairedRecord:
    """The per-record inputs required by :func:`paired_record_statistics`.

    ``length`` is the number of post-BOS positions scored.  The optional
    correctness vectors must use that same geometry and are only needed for
    position-level token gains/losses.
    """

    record_id: str
    length: int
    projected_correct: int
    a1_correct: int
    projected_exact: bool
    a1_exact: bool
    projected_correctness: tuple[bool, ...] | None = None
    a1_correctness: tuple[bool, ...] | None = None


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise StatisticsError(f"{name} must be an integer")
    return int(value)


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise StatisticsError(f"{name} must be boolean")
    return bool(value)


def _correctness_vector(value: Any, *, name: str) -> tuple[bool, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise StatisticsError(f"{name} must be a boolean sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise StatisticsError(f"{name} must be a boolean sequence") from exc
    return tuple(_boolean(item, name=f"{name}[{index}]") for index, item in enumerate(values))


def _pick(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    joined = ", ".join(names)
    raise StatisticsError(f"record is missing one of: {joined}")


def _normalise_record(value: PairedRecord | Mapping[str, Any]) -> PairedRecord:
    if isinstance(value, PairedRecord):
        record_id = value.record_id
        length = value.length
        projected_correct = value.projected_correct
        a1_correct = value.a1_correct
        projected_exact = value.projected_exact
        a1_exact = value.a1_exact
        projected_correctness = value.projected_correctness
        a1_correctness = value.a1_correctness
    elif isinstance(value, Mapping):
        record_id = value.get("record_id")
        length = _pick(value, "length", "scored_tokens")
        projected_correct = _pick(value, "projected_correct", "projected_correct_tokens")
        a1_correct = _pick(value, "a1_correct", "a1_correct_tokens")
        projected_exact = _pick(
            value,
            "projected_exact",
            "projected_exact_sequence_match",
        )
        a1_exact = _pick(value, "a1_exact", "a1_exact_sequence_match")
        projected_correctness = value.get("projected_correctness")
        a1_correctness = value.get("a1_correctness")
    else:
        raise StatisticsError("each paired record must be a PairedRecord or mapping")

    if not isinstance(record_id, str) or not record_id:
        raise StatisticsError("record_id must be a non-empty string")
    length = _integer(length, name=f"length for {record_id}")
    projected_correct = _integer(projected_correct, name=f"projected_correct for {record_id}")
    a1_correct = _integer(a1_correct, name=f"a1_correct for {record_id}")
    projected_exact = _boolean(projected_exact, name=f"projected_exact for {record_id}")
    a1_exact = _boolean(a1_exact, name=f"a1_exact for {record_id}")
    projected_correctness = _correctness_vector(
        projected_correctness,
        name=f"projected_correctness for {record_id}",
    )
    a1_correctness = _correctness_vector(a1_correctness, name=f"a1_correctness for {record_id}")

    if length <= 0:
        raise StatisticsError(f"length for {record_id} must be positive")
    for method, correct in (("projected", projected_correct), ("a1", a1_correct)):
        if correct < 0 or correct > length:
            raise StatisticsError(f"{method}_correct for {record_id} is outside [0, length]")
    if projected_exact != (projected_correct == length):
        raise StatisticsError(f"projected_exact disagrees with projected_correct for {record_id}")
    if a1_exact != (a1_correct == length):
        raise StatisticsError(f"a1_exact disagrees with a1_correct for {record_id}")

    vectors = (projected_correctness, a1_correctness)
    if (vectors[0] is None) != (vectors[1] is None):
        raise StatisticsError(
            f"both correctness vectors are required together for {record_id}"
        )
    if projected_correctness is not None and a1_correctness is not None:
        if len(projected_correctness) != length or len(a1_correctness) != length:
            raise StatisticsError(f"correctness-vector length disagrees for {record_id}")
        if sum(projected_correctness) != projected_correct:
            raise StatisticsError(f"projected correctness vector disagrees for {record_id}")
        if sum(a1_correctness) != a1_correct:
            raise StatisticsError(f"A1 correctness vector disagrees for {record_id}")

    return PairedRecord(
        record_id=record_id,
        length=length,
        projected_correct=projected_correct,
        a1_correct=a1_correct,
        projected_exact=projected_exact,
        a1_exact=a1_exact,
        projected_correctness=projected_correctness,
        a1_correctness=a1_correctness,
    )


def _normalise_records(
    records: Sequence[PairedRecord | Mapping[str, Any]],
) -> list[PairedRecord]:
    if isinstance(records, (str, bytes)):
        raise StatisticsError("records must be a sequence of paired records")
    try:
        values = list(records)
    except TypeError as exc:
        raise StatisticsError("records must be a sequence of paired records") from exc
    if not values:
        raise StatisticsError("records cannot be empty")
    normalised = [_normalise_record(value) for value in values]
    record_ids = [record.record_id for record in normalised]
    if len(set(record_ids)) != len(record_ids):
        raise StatisticsError("record_id values must be unique")
    # The ordering is part of the deterministic resampling contract.  It
    # keeps a caller's dictionary/list ordering from changing the CI stream.
    return sorted(normalised, key=lambda record: (record.length, record.record_id))


def _validate_sampling(draws: int, records_per_stratum: int | None) -> tuple[int, int | None]:
    draws = _integer(draws, name="draws")
    if draws <= 0:
        raise StatisticsError("draws must be positive")
    if records_per_stratum is not None:
        records_per_stratum = _integer(records_per_stratum, name="records_per_stratum")
        if records_per_stratum <= 0:
            raise StatisticsError("records_per_stratum must be positive")
    return draws, records_per_stratum


def _bootstrap_interval(
    values: np.ndarray,
    *,
    estimate: float,
    draws: int,
    seed: int,
    strata: dict[str, int],
) -> dict[str, Any]:
    low, high = np.quantile(values, [0.025, 0.975])
    ci = [float(low), float(high)]
    return {
        "estimate": float(estimate),
        "estimate_pp": float(estimate * 100.0),
        "ci95_percentile": ci,
        "ci95_percentile_pp": [float(low * 100.0), float(high * 100.0)],
        "draws": int(draws),
        "seed": int(seed),
        "unit": "paired_record_cluster",
        "stratified_by": "length",
        "strata": dict(strata),
    }


def _stratified_bootstrap(
    records_by_length: dict[int, list[PairedRecord]],
    *,
    draws: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Return matched bootstrap intervals for micro, macro, and exact deltas."""

    generator = np.random.default_rng(seed)
    record_count = sum(len(group) for group in records_by_length.values())
    total_length = sum(length * len(group) for length, group in records_by_length.items())
    token_numerator = np.zeros(draws, dtype=np.float64)
    macro_numerator = np.zeros(draws, dtype=np.float64)
    exact_numerator = np.zeros(draws, dtype=np.float64)

    for length in sorted(records_by_length):
        group = records_by_length[length]
        indices = generator.integers(0, len(group), size=(draws, len(group)))
        count_delta = np.asarray(
            [record.projected_correct - record.a1_correct for record in group],
            dtype=np.float64,
        )
        accuracy_delta = count_delta / float(length)
        exact_delta = np.asarray(
            [int(record.projected_exact) - int(record.a1_exact) for record in group],
            dtype=np.float64,
        )
        token_numerator += count_delta[indices].sum(axis=1)
        macro_numerator += accuracy_delta[indices].sum(axis=1)
        exact_numerator += exact_delta[indices].sum(axis=1)

    strata = {str(length): len(records_by_length[length]) for length in sorted(records_by_length)}
    point_token = 0.0
    point_macro = 0.0
    point_exact = 0.0
    for length in sorted(records_by_length):
        group = records_by_length[length]
        point_token += sum(
            record.projected_correct - record.a1_correct for record in group
        )
        point_macro += sum(
            (record.projected_correct - record.a1_correct) / float(length)
            for record in group
        )
        point_exact += sum(
            int(record.projected_exact) - int(record.a1_exact) for record in group
        )
    return {
        "token": _bootstrap_interval(
            token_numerator / float(total_length),
            estimate=float(point_token / total_length),
            draws=draws,
            seed=seed,
            strata=strata,
        ),
        "macro": _bootstrap_interval(
            macro_numerator / float(record_count),
            estimate=float(point_macro / record_count),
            draws=draws,
            seed=seed,
            strata=strata,
        ),
        "exact_record": _bootstrap_interval(
            exact_numerator / float(record_count),
            estimate=float(point_exact / record_count),
            draws=draws,
            seed=seed,
            strata=strata,
        ),
    }


def _change_counts(records: Sequence[PairedRecord]) -> dict[str, int]:
    count_deltas = [record.projected_correct - record.a1_correct for record in records]
    return {
        "gain_records": int(sum(delta > 0 for delta in count_deltas)),
        "tie_records": int(sum(delta == 0 for delta in count_deltas)),
        "loss_records": int(sum(delta < 0 for delta in count_deltas)),
        "gain_tokens": int(sum(max(delta, 0) for delta in count_deltas)),
        "loss_tokens": int(sum(max(-delta, 0) for delta in count_deltas)),
        "net_tokens": int(sum(count_deltas)),
    }


def _exact_change_counts(records: Sequence[PairedRecord]) -> dict[str, int]:
    deltas = [int(record.projected_exact) - int(record.a1_exact) for record in records]
    return {
        "gain_records": int(sum(delta > 0 for delta in deltas)),
        "tie_records": int(sum(delta == 0 for delta in deltas)),
        "loss_records": int(sum(delta < 0 for delta in deltas)),
        "net_records": int(sum(deltas)),
    }


def _position_change_counts(records: Sequence[PairedRecord]) -> dict[str, Any]:
    vectors_available = all(
        record.projected_correctness is not None and record.a1_correctness is not None
        for record in records
    )
    if not vectors_available:
        return {
            "available": False,
            "gain_tokens": None,
            "tie_tokens": None,
            "loss_tokens": None,
            "regression_tokens": None,
            "both_correct_tokens": None,
            "both_wrong_tokens": None,
            "net_tokens": None,
        }

    gain = tie = loss = 0
    both_correct = both_wrong = 0
    for record in records:
        assert record.projected_correctness is not None
        assert record.a1_correctness is not None
        projected = np.asarray(record.projected_correctness, dtype=bool)
        a1 = np.asarray(record.a1_correctness, dtype=bool)
        gain += int(np.logical_and(projected, np.logical_not(a1)).sum())
        loss += int(np.logical_and(a1, np.logical_not(projected)).sum())
        tie += int(np.equal(projected, a1).sum())
        both_correct += int(np.logical_and(projected, a1).sum())
        both_wrong += int(np.logical_and(np.logical_not(projected), np.logical_not(a1)).sum())
    return {
        "available": True,
        "gain_tokens": gain,
        "tie_tokens": tie,
        "loss_tokens": loss,
        "regression_tokens": loss,
        "both_correct_tokens": both_correct,
        "both_wrong_tokens": both_wrong,
        "net_tokens": gain - loss,
    }


def paired_record_statistics(
    records: Sequence[PairedRecord | Mapping[str, Any]],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    records_per_stratum: int | None = 6,
) -> dict[str, Any]:
    """Compute paired P03 accuracy, exactness, and uncertainty statistics.

    Bootstrap draws sample records with replacement *within each length
    stratum*, preserving the observed number of records in every stratum and
    using one shared sampled record set for projected and A1.  ``token`` is
    the primary micro-token delta; ``macro`` gives an equal-record-weighted
    accuracy delta; and ``exact_record`` gives the exact-record-rate delta.

    ``gain_tokens``/``loss_tokens`` under ``correct_count_changes`` are
    positive and negative changes in each record's correct-count total.  They
    are not position-level gains.  Position-level values are present under
    ``token_position_changes`` only when both correctness vectors were passed.
    """

    draws, records_per_stratum = _validate_sampling(draws, records_per_stratum)
    seed = _integer(seed, name="seed")
    normalised = _normalise_records(records)
    records_by_length: dict[int, list[PairedRecord]] = {}
    for record in normalised:
        records_by_length.setdefault(record.length, []).append(record)
    if records_per_stratum is not None:
        for length, group in records_by_length.items():
            if len(group) != records_per_stratum:
                raise StatisticsError(
                    f"length stratum {length} has {len(group)} records; "
                    f"expected {records_per_stratum}"
                )

    record_count = len(normalised)
    total_length = sum(record.length for record in normalised)
    projected_correct = sum(record.projected_correct for record in normalised)
    a1_correct = sum(record.a1_correct for record in normalised)
    projected_accuracy = projected_correct / float(total_length)
    a1_accuracy = a1_correct / float(total_length)
    token_delta = (projected_correct - a1_correct) / float(total_length)

    per_record_deltas = np.asarray(
        [
            (record.projected_correct - record.a1_correct) / float(record.length)
            for record in normalised
        ],
        dtype=np.float64,
    )
    projected_record_accuracies = np.asarray(
        [record.projected_correct / float(record.length) for record in normalised],
        dtype=np.float64,
    )
    a1_record_accuracies = np.asarray(
        [record.a1_correct / float(record.length) for record in normalised],
        dtype=np.float64,
    )
    macro_projected_accuracy = float(projected_record_accuracies.mean())
    macro_a1_accuracy = float(a1_record_accuracies.mean())
    macro_delta = float(per_record_deltas.mean())
    projected_exact_records = sum(int(record.projected_exact) for record in normalised)
    a1_exact_records = sum(int(record.a1_exact) for record in normalised)
    exact_delta = (projected_exact_records - a1_exact_records) / float(record_count)

    bootstrap = _stratified_bootstrap(
        records_by_length,
        draws=draws,
        seed=seed,
    )
    count_changes = _change_counts(normalised)
    exact_changes = _exact_change_counts(normalised)
    position_changes = _position_change_counts(normalised)
    strata = {str(length): len(group) for length, group in sorted(records_by_length.items())}

    token_result = {
        "projected_correct_tokens": int(projected_correct),
        "a1_correct_tokens": int(a1_correct),
        "scored_tokens": int(total_length),
        "projected_accuracy": float(projected_accuracy),
        "a1_accuracy": float(a1_accuracy),
        "delta": float(token_delta),
        "delta_pp": float(token_delta * 100.0),
        "bootstrap": bootstrap["token"],
        "correct_count_changes": count_changes,
    }
    macro_result = {
        "records": int(record_count),
        "projected_accuracy": macro_projected_accuracy,
        "a1_accuracy": macro_a1_accuracy,
        "delta": macro_delta,
        "delta_pp": float(macro_delta * 100.0),
        "median_record_delta": float(np.median(per_record_deltas)),
        "worst_record_delta": float(per_record_deltas.min()),
        "best_record_delta": float(per_record_deltas.max()),
        "bootstrap": bootstrap["macro"],
        "record_changes": {
            "gain_records": count_changes["gain_records"],
            "tie_records": count_changes["tie_records"],
            "loss_records": count_changes["loss_records"],
        },
    }
    exact_result = {
        "projected_exact_records": int(projected_exact_records),
        "a1_exact_records": int(a1_exact_records),
        "records": int(record_count),
        "projected_rate": float(projected_exact_records / record_count),
        "a1_rate": float(a1_exact_records / record_count),
        "delta": float(exact_delta),
        "delta_pp": float(exact_delta * 100.0),
        "bootstrap": bootstrap["exact_record"],
        "record_changes": exact_changes,
    }

    return {
        "schema": STATISTICS_SCHEMA,
        "records": int(record_count),
        "strata": strata,
        "bootstrap_config": {
            "draws": int(draws),
            "seed": int(seed),
            "unit": "paired_record_cluster",
            "stratified_by": "length",
        },
        "token": token_result,
        "macro": macro_result,
        "exact_record": exact_result,
        "token_position_changes": position_changes,
        # Gate consumers can use these stable aliases without depending on
        # the nested presentation of the three metric families above.
        "gate_ready": {
            "token_delta_pp": float(token_delta * 100.0),
            "token_ci95_pp": bootstrap["token"]["ci95_percentile_pp"],
            "token_ci95_pp_lower": float(bootstrap["token"]["ci95_percentile_pp"][0]),
            "projected_exact_records": int(projected_exact_records),
            "a1_exact_records": int(a1_exact_records),
            "projected_exact_records_not_lower": bool(
                projected_exact_records >= a1_exact_records
            ),
            "macro_delta_pp": float(macro_delta * 100.0),
            "exact_record_delta_pp": float(exact_delta * 100.0),
            "exact_record_ci95_pp": bootstrap["exact_record"]["ci95_percentile_pp"],
        },
    }


# Short aliases make migration from an early P03 runner harmless while
# retaining one canonical implementation and schema.
compute_paired_statistics = paired_record_statistics
bootstrap_paired_records = paired_record_statistics


__all__ = [
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "PairedRecord",
    "STATISTICS_SCHEMA",
    "StatisticsError",
    "bootstrap_paired_records",
    "compute_paired_statistics",
    "paired_record_statistics",
]
