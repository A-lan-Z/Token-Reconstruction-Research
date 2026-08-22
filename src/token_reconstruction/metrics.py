"""Metric and record-level bootstrap primitives for TRR-0001."""

from __future__ import annotations

import math
import statistics
import unicodedata
from typing import Any, Sequence

import numpy as np


def correct_prefix_length(prediction: Sequence[int], truth: Sequence[int]) -> int:
    if len(prediction) != len(truth):
        raise ValueError("prediction and truth lengths differ")
    length = 0
    for predicted, expected in zip(prediction, truth):
        if int(predicted) != int(expected):
            break
        length += 1
    return length


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile input is invalid")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean(
    values: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or draws <= 0:
        raise ValueError("bootstrap inputs are empty")
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, array.size, size=(draws, array.size))
    estimates = array[samples].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "ci95_percentile": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "draws": draws,
        "seed": seed,
        "unit": "record",
    }


def record_metrics(
    prediction: Sequence[int],
    truth: Sequence[int],
    candidates: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(prediction) != 39 or len(truth) != 39 or len(candidates) != 39:
        raise ValueError("TRR-0001 record geometry changed")
    correct = [int(left) == int(right) for left, right in zip(prediction, truth)]
    in_candidates = [
        int(expected) in {int(value) for value in proposed}
        for expected, proposed in zip(truth, candidates)
    ]
    prefix = correct_prefix_length(prediction, truth)
    first_error = None if prefix == len(truth) else prefix + 1
    conditional_denominator = sum(in_candidates)
    conditional_numerator = sum(
        int(ok and available) for ok, available in zip(correct, in_candidates)
    )
    return {
        "correct_tokens": sum(correct),
        "token_accuracy": sum(correct) / len(correct),
        "exact_sequence_match": all(correct),
        "correct_prefix_length": prefix,
        "first_error_position": first_error,
        "coverage": 1.0,
        "selective_accuracy": sum(correct) / len(correct),
        "top16_true_token_count": conditional_denominator,
        "top16_recall": conditional_denominator / len(truth),
        "conditional_correct_count": conditional_numerator,
        "conditional_selection_accuracy": (
            conditional_numerator / conditional_denominator
            if conditional_denominator
            else None
        ),
        "correctness": correct,
        "true_in_top16": in_candidates,
    }


def frequency_bin(count: int) -> str:
    if count < 0:
        raise ValueError("frequency cannot be negative")
    if count == 0:
        return "unseen"
    if count <= 4:
        return "1-4"
    if count <= 19:
        return "5-19"
    return "20-or-more"


def token_group(raw_token: str, decoded: str) -> str:
    stripped = "".join(character for character in decoded if not character.isspace())
    has_digit = any(character.isdecimal() for character in stripped)
    nondigits = [character for character in stripped if not character.isdecimal()]
    if has_digit and all(
        unicodedata.category(character).startswith("P") for character in nondigits
    ):
        return "numeric"
    if stripped and all(
        unicodedata.category(character).startswith("P") for character in stripped
    ):
        return "punctuation"
    if (decoded and decoded[0].isspace()) or raw_token.startswith("Ġ"):
        return "whitespace_prefixed"
    return "other"


def summarize_numeric(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }
