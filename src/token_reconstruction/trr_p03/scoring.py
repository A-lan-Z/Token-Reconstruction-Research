"""Post-freeze TRR-P03 scoring.

Prediction rows are validated, and all declared method/record coverage is
checked, before this module opens the evaluator-only truth sidecar.  Numeric
outputs are written before their file records are constructed; a failed plot
or serialization step therefore cannot discard a completed score computation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors import safe_open

from .io import (
    BOS_TOKEN_ID,
    P03IOError,
    TRUTH_SCHEMA,
    VOCAB_SIZE,
    create_only_directory,
    file_record,
    read_json,
    read_jsonl,
    verify_freeze_receipt,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from .statistics import paired_record_statistics


SCORING_SCHEMA = "token-reconstruction.trr-p03-score.v2"
PAIRED_SCHEMA = "token-reconstruction.trr-p03-paired-statistics.v2"
JOINT_VALIDATION_SCHEMA = "token-reconstruction.trr-p03-stage1-joint-validation.v1"
JOINT_VALIDATION_STATUS = "VALIDATED"
JOINT_VALIDATION_MARKER = "STAGE1_JOINT_VALIDATION_PASS"
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_905
BASE_METHODS = frozenset(
    {
        "raw_boundary.cosine",
        "projected_boundary.cosine",
        "historical_a1.cosine",
    }
)
A2_METHOD = "historical_a1_a2_anchor.cosine"


class ScoringError(RuntimeError):
    """Raised when post-freeze truth or prediction rows cannot be joined."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class _PreparedBundle:
    root: Path
    freeze: dict[str, Any]
    rows_path: Path
    rows: list[dict[str, Any]]
    grouped: dict[str, dict[str, dict[str, Any]]]
    methods: tuple[str, ...]
    record_ids: tuple[str, ...]
    anchor_record_ids: tuple[str, ...]


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ScoringError(f"{name} must be finite")
    return result


def _record_score(
    prediction: Sequence[int],
    truth: Sequence[int],
    *,
    declared_sequence_length: int | None = None,
) -> dict[str, Any]:
    if isinstance(prediction, (str, bytes)) or isinstance(truth, (str, bytes)):
        raise ScoringError("prediction/truth sequences must be integer sequences")
    try:
        predicted = [int(value) for value in prediction]
        expected = [int(value) for value in truth]
    except (TypeError, ValueError) as exc:
        raise ScoringError("prediction/truth sequences contain non-integer values") from exc
    if len(predicted) != len(expected) or len(expected) < 2:
        raise ScoringError("prediction/truth sequence geometry differs")
    if declared_sequence_length is not None and int(declared_sequence_length) != len(predicted):
        raise ScoringError("prediction sequence_length disagrees with token list")
    if expected[0] != BOS_TOKEN_ID:
        raise ScoringError("truth does not begin with the declared BOS")
    if predicted[0] != BOS_TOKEN_ID:
        raise ScoringError("prediction does not begin with the declared BOS")
    if any(value < 0 or value >= VOCAB_SIZE for value in expected):
        raise ScoringError("truth token is outside the pinned vocabulary")
    if any(value < 0 or value >= VOCAB_SIZE for value in predicted):
        raise ScoringError("prediction token is outside the pinned vocabulary")
    correctness = [left == right for left, right in zip(predicted[1:], expected[1:])]
    prefix = 0
    for ok in correctness:
        if not ok:
            break
        prefix += 1
    first_error = None if prefix == len(correctness) else prefix + 1
    correct = int(sum(correctness))
    scored = len(correctness)
    return {
        "correct_tokens": correct,
        "scored_tokens": scored,
        "token_accuracy": float(correct / scored),
        "exact_sequence_match": bool(all(correctness)),
        "correct_prefix_length": int(prefix),
        "first_error_position": first_error,
        "coverage": 1.0,
        "selective_accuracy": float(correct / scored),
        "correctness": [bool(value) for value in correctness],
    }


def _bootstrap(values: Sequence[float], *, draws: int, seed: int) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or values_array.size <= 0 or draws <= 0:
        raise ScoringError("paired bootstrap inputs are invalid")
    generator = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    remaining = int(draws)
    while remaining:
        count = min(1024, remaining)
        indices = generator.integers(0, values_array.size, size=(count, values_array.size))
        samples.append(values_array[indices].mean(axis=1))
        remaining -= count
    sampled = np.concatenate(samples)
    return {
        "estimate": float(values_array.mean()),
        "ci95_percentile": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "draws": int(draws),
        "seed": int(seed),
        "unit": "paired_record_cluster",
    }


def _truth_rows_jsonl(path: Path) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for row in read_jsonl(path):
        if set(row) != {"record_id", "token_ids"}:
            raise ScoringError("truth JSONL row fields changed")
        record_id = row.get("record_id")
        token_ids = row.get("token_ids")
        if not isinstance(record_id, str) or not record_id or record_id in result:
            raise ScoringError("truth record IDs are missing or duplicated")
        if not isinstance(token_ids, list) or len(token_ids) < 2:
            raise ScoringError("truth token sequence is invalid")
        result[record_id] = [int(value) for value in token_ids]
    if not result:
        raise ScoringError("truth sidecar is empty")
    return result


def _truth_rows_safetensors(path: Path, truth_index_path: Path | None) -> dict[str, list[int]]:
    values: dict[str, list[int]] = {}
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata and metadata.get("schema") not in {
                TRUTH_SCHEMA,
                "token-reconstruction.trr-p03-private-truth.v1",
            }:
                raise ScoringError("truth safetensors schema changed")
            for key in handle.keys():
                tensor = handle.get_tensor(key).reshape(-1).long()
                values[key] = [int(value) for value in tensor.tolist()]
    except ScoringError:
        raise
    except Exception as exc:
        raise ScoringError(f"truth safetensors is invalid: {path}") from exc
    if not values:
        raise ScoringError("truth safetensors is empty")
    if truth_index_path is None:
        result = {
            key.removesuffix("__input_ids").replace("_", "-"): token_ids
            for key, token_ids in values.items()
        }
    else:
        index = read_json(truth_index_path)
        rows = index.get("records") if isinstance(index, Mapping) else None
        if not isinstance(rows, list):
            raise ScoringError("truth index records are missing")
        result = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ScoringError("truth index row is invalid")
            record_id, key = row.get("record_id"), row.get("tensor_key")
            if not isinstance(record_id, str) or not isinstance(key, str) or key not in values:
                raise ScoringError("truth index does not match truth tensors")
            if record_id in result:
                raise ScoringError("truth index IDs are duplicated")
            result[record_id] = values[key]
    for record_id, token_ids in result.items():
        if len(token_ids) < 2 or token_ids[0] != BOS_TOKEN_ID:
            raise ScoringError(f"truth sequence is invalid for {record_id}")
    return result


def _load_truth(path: Path, truth_index_path: Path | None = None) -> dict[str, list[int]]:
    """Open evaluator truth; callers must invoke only after the freeze gate."""

    if path.is_symlink() or not path.is_file():
        raise ScoringError(f"truth sidecar is missing: {path}")
    try:
        if path.suffix.lower() in {".jsonl", ".json"}:
            return _truth_rows_jsonl(path)
        return _truth_rows_safetensors(path, truth_index_path)
    except ScoringError:
        raise
    except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
        raise ScoringError(f"truth sidecar is invalid: {path}") from exc


def _validate_prediction_row(row: Mapping[str, Any]) -> tuple[str, str, int, list[int]]:
    required = {"record_id", "method", "prediction_tokens", "sequence_length", "truth_opened"}
    allowed = required | {
        "top1_tie_count",
        "top1_scores",
        "top1_runner_margins",
        "score_units",
        "observation_sha256",
    }
    if set(row) - allowed:
        raise ScoringError("prediction row exposes unexpected fields")
    if not required.issubset(row) or row.get("truth_opened") is not False:
        raise ScoringError("prediction row is malformed or truth-aware")
    method = row.get("method")
    record_id = row.get("record_id")
    tokens = row.get("prediction_tokens")
    if not isinstance(method, str) or not method or not isinstance(record_id, str) or not record_id:
        raise ScoringError("prediction row method or record ID is invalid")
    if not isinstance(row.get("sequence_length"), int) or isinstance(row.get("sequence_length"), bool):
        raise ScoringError("prediction sequence_length is invalid")
    sequence_length = int(row["sequence_length"])
    if sequence_length < 2 or not isinstance(tokens, list) or len(tokens) != sequence_length:
        raise ScoringError("prediction token geometry is invalid")
    try:
        token_values = [int(value) for value in tokens]
    except (TypeError, ValueError) as exc:
        raise ScoringError("prediction tokens are not integers") from exc
    if token_values[0] != BOS_TOKEN_ID:
        raise ScoringError("prediction sequence BOS changed")
    for key in ("top1_tie_count", "top1_scores", "top1_runner_margins"):
        if key in row:
            values = row[key]
            expected = sequence_length - 1
            if not isinstance(values, list) or len(values) != expected:
                raise ScoringError(f"prediction diagnostic geometry changed: {key}")
            if key == "top1_tie_count":
                if any(isinstance(value, bool) or not isinstance(value, int) or int(value) < 1 for value in values):
                    raise ScoringError("prediction tie counts are invalid")
            else:
                for index, value in enumerate(values):
                    _finite_float(value, name=f"{key}[{index}]")
    return method, record_id, sequence_length, token_values


def _prepare_prediction_bundle(root: Path) -> _PreparedBundle:
    root = root.resolve()
    try:
        freeze = verify_freeze_receipt(root)
    except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
        raise ScoringError(f"frozen prediction bundle failed verification: {root}") from exc
    rows_path = root / "predictions.jsonl"
    if not rows_path.is_file():
        raise ScoringError("frozen prediction rows are missing")
    frozen_paths = {str(entry["path"]) for entry in freeze["entries"] if isinstance(entry, Mapping) and "path" in entry}
    if "predictions.jsonl" not in frozen_paths:
        raise ScoringError("prediction rows were not included in the freeze")

    try:
        rows = read_jsonl(rows_path)
    except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
        raise ScoringError(f"frozen prediction rows are invalid: {rows_path}") from exc
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    sequence_lengths: dict[str, int] = {}
    for row in rows:
        method, record_id, sequence_length, _ = _validate_prediction_row(row)
        if record_id in grouped[method]:
            raise ScoringError("duplicate method/record prediction row")
        prior_length = sequence_lengths.get(record_id)
        if prior_length is not None and prior_length != sequence_length:
            raise ScoringError("method/record sequence lengths differ")
        sequence_lengths[record_id] = sequence_length
        grouped[method][record_id] = dict(row)
    if not grouped:
        raise ScoringError("prediction rows are empty")

    metadata = freeze.get("metadata") if isinstance(freeze.get("metadata"), Mapping) else {}
    declared_methods = metadata.get("methods")
    if declared_methods is not None:
        if not isinstance(declared_methods, list) or any(not isinstance(value, str) for value in declared_methods):
            raise ScoringError("freeze method declaration is malformed")
        if len(set(declared_methods)) != len(declared_methods) or set(declared_methods) != set(grouped):
            raise ScoringError("prediction methods differ from the frozen declaration")
    methods = tuple(sorted(grouped))

    declared_record_ids = metadata.get("record_ids")
    if declared_record_ids is not None:
        if not isinstance(declared_record_ids, list) or any(not isinstance(value, str) for value in declared_record_ids):
            raise ScoringError("freeze record declaration is malformed")
        if len(set(declared_record_ids)) != len(declared_record_ids):
            raise ScoringError("freeze record declaration contains duplicates")
        expected_ids = tuple(declared_record_ids)
    else:
        expected_ids = tuple(sorted(sequence_lengths))
    declared_count = metadata.get("records")
    if declared_count is not None and (not isinstance(declared_count, int) or int(declared_count) != len(expected_ids)):
        raise ScoringError("prediction record count differs from the frozen declaration")

    base_present = BASE_METHODS.intersection(grouped)
    if base_present:
        if base_present != BASE_METHODS:
            raise ScoringError("the frozen base-method matrix is incomplete")
        base_sets = {method: set(grouped[method]) for method in BASE_METHODS}
        if any(ids != set(expected_ids) for ids in base_sets.values()):
            raise ScoringError("base methods do not cover the same frozen records")
    if A2_METHOD in grouped:
        raw_anchors = metadata.get("anchor_record_ids")
        if not isinstance(raw_anchors, list) or not raw_anchors or any(not isinstance(value, str) for value in raw_anchors):
            raise ScoringError("A1+A2 anchor IDs are missing from the freeze declaration")
        anchor_ids = tuple(raw_anchors)
        if len(set(anchor_ids)) != len(anchor_ids) or set(grouped[A2_METHOD]) != set(anchor_ids):
            raise ScoringError("A1+A2 anchor coverage differs from the frozen declaration")
    else:
        anchor_ids = ()
    return _PreparedBundle(
        root=root,
        freeze=freeze,
        rows_path=rows_path,
        rows=rows,
        grouped={method: dict(values) for method, values in grouped.items()},
        methods=methods,
        record_ids=expected_ids,
        anchor_record_ids=anchor_ids,
    )


def verify_prediction_bundles(
    prediction_roots: Sequence[Path],
) -> list[dict[str, Any]]:
    """Run all public freeze/coverage checks without opening any truth."""

    if not prediction_roots:
        raise ScoringError("at least one prediction bundle is required")
    prepared = [_prepare_prediction_bundle(path) for path in prediction_roots]
    if len(prepared) > 1:
        first = prepared[0]
        for other in prepared[1:]:
            if set(first.methods) != set(other.methods):
                raise ScoringError("paired target bundles declare different methods")
            common_methods = BASE_METHODS.intersection(first.methods)
            for method in common_methods:
                if set(first.grouped[method]) != set(other.grouped[method]):
                    raise ScoringError("paired target bundles have different base record coverage")
            if first.anchor_record_ids != other.anchor_record_ids:
                raise ScoringError("paired target bundles have different anchor declarations")
    return [
        {
            "root": str(item.root),
            "methods": list(item.methods),
            "records": len(item.record_ids),
            "anchor_records": list(item.anchor_record_ids),
            "truth_opened": False,
        }
        for item in prepared
    ]


def _score_prepared(
    prepared: _PreparedBundle,
    truth: Mapping[str, Sequence[int]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    per_record: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    scores_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    total_truth_tokens = sum(max(len(values) - 1, 0) for values in truth.values())
    for method in prepared.methods:
        method_rows = prepared.grouped[method]
        method_scores: dict[str, dict[str, Any]] = {}
        for record_id in sorted(method_rows):
            if record_id not in truth:
                raise ScoringError(f"prediction has no matching truth record: {record_id}")
            row = method_rows[record_id]
            score = _record_score(
                row["prediction_tokens"],
                truth[record_id],
                declared_sequence_length=int(row["sequence_length"]),
            )
            record_result: dict[str, Any] = {
                "method": method,
                "record_id": record_id,
                "sequence_length": int(row["sequence_length"]),
                "length": int(score["scored_tokens"]),
                **score,
                "truth_opened": True,
            }
            if "top1_tie_count" in row:
                tie_values = [int(value) for value in row["top1_tie_count"]]
                record_result["top1_tie_count"] = tie_values
                record_result["top1_tie_positions"] = int(sum(value > 1 for value in tie_values))
            method_scores[record_id] = record_result
            per_record.append(record_result)
        scores_by_method[method] = method_scores
        correct = sum(int(item["correct_tokens"]) for item in method_scores.values())
        scored = sum(int(item["scored_tokens"]) for item in method_scores.values())
        exact = sum(int(item["exact_sequence_match"]) for item in method_scores.values())
        coverage_tokens = float(scored / total_truth_tokens) if total_truth_tokens else 0.0
        accuracies = [float(item["token_accuracy"]) for item in method_scores.values()]
        summaries[method] = {
            "method": method,
            "records": len(method_scores),
            "truth_records": len(truth),
            "scored_tokens": scored,
            "correct_tokens": correct,
            "token_accuracy": float(correct / scored) if scored else 0.0,
            "exact_records": exact,
            "exact_record_rate": float(exact / len(method_scores)) if method_scores else 0.0,
            "coverage": coverage_tokens,
            "selective_accuracy": float(correct / scored) if scored else 0.0,
            "top1_tie_positions": sum(int(item.get("top1_tie_positions", 0)) for item in method_scores.values()),
            "record_accuracy_mean": float(np.mean(accuracies)) if accuracies else 0.0,
            "record_accuracy_min": float(min(accuracies)) if accuracies else 0.0,
            "record_accuracy_max": float(max(accuracies)) if accuracies else 0.0,
        }
    return per_record, summaries, scores_by_method


def _generic_pairs(
    scores_by_method: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(scores_by_method), 2):
        common = sorted(set(scores_by_method[left]) & set(scores_by_method[right]))
        if not common:
            continue
        left_rows = [scores_by_method[left][record_id] for record_id in common]
        right_rows = [scores_by_method[right][record_id] for record_id in common]
        record_delta = [
            float(left_row["token_accuracy"] - right_row["token_accuracy"])
            for left_row, right_row in zip(left_rows, right_rows)
        ]
        left_correctness = [value for row in left_rows for value in row["correctness"]]
        right_correctness = [value for row in right_rows for value in row["correctness"]]
        gains = sum(bool(l) and not bool(r) for l, r in zip(left_correctness, right_correctness))
        regressions = sum(bool(r) and not bool(l) for l, r in zip(left_correctness, right_correctness))
        both_correct = sum(bool(l) and bool(r) for l, r in zip(left_correctness, right_correctness))
        both_wrong = sum(not bool(l) and not bool(r) for l, r in zip(left_correctness, right_correctness))
        scored = sum(int(row["scored_tokens"]) for row in left_rows)
        pairs.append(
            {
                "left_method": left,
                "right_method": right,
                "records": len(common),
                "record_ids": common,
                "mean_record_accuracy_delta": float(np.mean(record_delta)),
                "token_correct_delta": int(sum(row["correct_tokens"] for row in left_rows) - sum(row["correct_tokens"] for row in right_rows)),
                "token_accuracy_delta": float((sum(row["correct_tokens"] for row in left_rows) - sum(row["correct_tokens"] for row in right_rows)) / scored),
                "token_position_changes": {
                    "gain_tokens": int(gains),
                    "regression_tokens": int(regressions),
                    "both_correct_tokens": int(both_correct),
                    "both_wrong_tokens": int(both_wrong),
                },
                "bootstrap": _bootstrap(record_delta, draws=draws, seed=seed),
            }
        )
    return pairs


def _design_stats(
    scores_by_method: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    draws: int,
    seed: int,
    records_per_stratum: int | None,
) -> dict[str, Any] | None:
    projected_name = "projected_boundary.cosine"
    a1_name = "historical_a1.cosine"
    if projected_name not in scores_by_method or a1_name not in scores_by_method:
        return None
    common = sorted(set(scores_by_method[projected_name]) & set(scores_by_method[a1_name]))
    rows: list[dict[str, Any]] = []
    for record_id in common:
        projected = scores_by_method[projected_name][record_id]
        a1 = scores_by_method[a1_name][record_id]
        rows.append(
            {
                "record_id": record_id,
                "length": int(projected["scored_tokens"]),
                "projected_correct": int(projected["correct_tokens"]),
                "a1_correct": int(a1["correct_tokens"]),
                "projected_exact": bool(projected["exact_sequence_match"]),
                "a1_exact": bool(a1["exact_sequence_match"]),
                "projected_correctness": list(projected["correctness"]),
                "a1_correctness": list(a1["correctness"]),
            }
        )
    if not rows:
        return None
    result = paired_record_statistics(
        rows,
        draws=draws,
        seed=seed,
        records_per_stratum=records_per_stratum,
    )
    # Keep the design module's count/tie/loss names and add the explicit
    # position contingency requested by the Stage-1 report. These values are
    # calculated from saved correctness vectors, never from count deltas.
    projected_values = [
        value for row in rows for value in row["projected_correctness"]
    ]
    a1_values = [value for row in rows for value in row["a1_correctness"]]
    gain = sum(
        bool(left) and not bool(right)
        for left, right in zip(projected_values, a1_values)
    )
    regression = sum(
        bool(right) and not bool(left)
        for left, right in zip(projected_values, a1_values)
    )
    both_correct = sum(
        bool(left) and bool(right)
        for left, right in zip(projected_values, a1_values)
    )
    both_wrong = sum(
        not bool(left) and not bool(right)
        for left, right in zip(projected_values, a1_values)
    )
    result["token_position_changes"].update(
        {
            "regression_tokens": int(regression),
            "both_correct_tokens": int(both_correct),
            "both_wrong_tokens": int(both_wrong),
            "gain_tokens": int(gain),
        }
    )
    return result


def _load_pre_score_receipt(
    path: Path,
    *,
    expected_implementation_commit: str | None,
    prepared_items: Sequence[_PreparedBundle],
) -> dict[str, Any]:
    """Check and bind the strict Stage-1 receipt before opening truth."""

    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ScoringError(f"pre-score validation receipt is missing: {path}")
    try:
        value = read_json(path)
    except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
        raise ScoringError(f"pre-score validation receipt is invalid: {path}") from exc
    if not isinstance(value, Mapping):
        raise ScoringError("pre-score validation receipt is not an object")
    if value.get("schema") != JOINT_VALIDATION_SCHEMA:
        raise ScoringError("pre-score validation receipt schema is not the strict Stage-1 schema")
    if value.get("task_id") != "TRR-P03" or value.get("truth_opened") is not False:
        raise ScoringError("pre-score validation receipt is not truth-free")
    if value.get("status") != JOINT_VALIDATION_STATUS or value.get("validation") != JOINT_VALIDATION_MARKER:
        raise ScoringError("pre-score validation receipt is not a strict Stage-1 PASS")
    receipt_commit = value.get("implementation_commit")
    if not isinstance(receipt_commit, str) or not receipt_commit:
        raise ScoringError("pre-score receipt implementation commit is missing")
    if expected_implementation_commit is not None and receipt_commit != expected_implementation_commit:
        raise ScoringError("pre-score receipt implementation commit differs")

    prerequisite = value.get("score_prerequisite")
    if not isinstance(prerequisite, Mapping):
        raise ScoringError("strict pre-score receipt prerequisite is missing")
    if prerequisite.get("paired_prediction_root_required") is not True:
        raise ScoringError("strict pre-score receipt does not require the paired prediction root")
    if prerequisite.get("allow_unequal_strata") is not False:
        raise ScoringError("strict pre-score receipt permits unequal strata")
    if prerequisite.get("truth_read_after_this_receipt") is not True:
        raise ScoringError("strict pre-score receipt truth ordering is missing")

    predictions = value.get("predictions")
    if not isinstance(predictions, Mapping) or set(predictions) != {"bundle-a", "bundle-b"}:
        raise ScoringError("strict pre-score receipt must bind both prediction bundles")
    if len(prepared_items) != 2:
        raise ScoringError("strict pre-score receipt requires primary and paired prediction roots")

    for bundle_id, prepared in zip(("bundle-a", "bundle-b"), prepared_items, strict=True):
        declared = predictions.get(bundle_id)
        if not isinstance(declared, Mapping):
            raise ScoringError(f"strict pre-score receipt is missing {bundle_id} prediction metadata")
        declared_root = declared.get("root")
        if not isinstance(declared_root, str) or not declared_root:
            raise ScoringError(f"strict pre-score receipt {bundle_id} root is missing")
        declared_root_path = Path(declared_root)
        if declared_root_path.is_symlink() or declared_root_path.resolve() != prepared.root:
            raise ScoringError(f"strict pre-score receipt {bundle_id} root differs from supplied root")

        freeze_declared = declared.get("freeze")
        if not isinstance(freeze_declared, Mapping):
            raise ScoringError(f"strict pre-score receipt {bundle_id} freeze record is missing")
        freeze_path = prepared.root / "freeze_receipt.json"
        try:
            actual_freeze = file_record(freeze_path)
        except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
            raise ScoringError(f"strict pre-score receipt {bundle_id} freeze artifact is missing") from exc
        declared_freeze_path = freeze_declared.get("path")
        if not isinstance(declared_freeze_path, str) or not declared_freeze_path:
            raise ScoringError(f"strict pre-score receipt {bundle_id} freeze path is missing")
        declared_freeze_path_obj = Path(declared_freeze_path)
        if declared_freeze_path_obj.is_symlink() or declared_freeze_path_obj.resolve() != freeze_path.resolve():
            raise ScoringError(f"strict pre-score receipt {bundle_id} freeze path differs from supplied root")
        if declared_freeze_path_obj.is_absolute():
            declared_freeze_path_value = str(declared_freeze_path_obj.resolve())
        else:
            declared_freeze_path_value = str(declared_freeze_path_obj.resolve())
        if declared_freeze_path_value != actual_freeze["path"]:
            raise ScoringError(f"strict pre-score receipt {bundle_id} freeze path is not canonical")
        if freeze_declared.get("bytes") != actual_freeze["bytes"] or freeze_declared.get("sha256") != actual_freeze["sha256"]:
            raise ScoringError(f"strict pre-score receipt {bundle_id} freeze hash differs from supplied root")

        freeze_metadata = (
            prepared.freeze.get("metadata")
            if isinstance(prepared.freeze.get("metadata"), Mapping)
            else {}
        )
        for key, expected in (
            ("plan_sha256", prepared.freeze.get("plan_sha256")),
            ("implementation_commit", prepared.freeze.get("implementation_commit")),
            ("methods", freeze_metadata.get("methods", list(prepared.methods))),
            ("record_order", freeze_metadata.get("record_ids", list(prepared.record_ids))),
            ("anchor_record_ids", freeze_metadata.get("anchor_record_ids", list(prepared.anchor_record_ids))),
        ):
            if key in declared and declared.get(key) != expected:
                raise ScoringError(f"strict pre-score receipt {bundle_id} {key} differs from supplied root")

    return dict(value)


def score_prediction_bundle(
    *,
    prediction_root: Path,
    truth_path: Path,
    output_root: Path,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    truth_index_path: Path | None = None,
    paired_prediction_root: Path | None = None,
    records_per_stratum: int | None = 6,
    pre_score_receipt_path: Path | None = None,
    implementation_commit: str | None = None,
) -> dict[str, Any]:
    """Verify frozen bundle(s), then open truth once and score every bundle.

    The first bundle remains the backward-compatible primary result.  When a
    paired root is supplied, both roots are scored against the same truth and
    ``per_record.jsonl``, ``metrics.json``, and ``paired_statistics.json`` carry
    a separate target-bundle result for each root.
    """

    started = time.perf_counter()
    started_utc = _utc_now()
    prediction_root = prediction_root.resolve()
    truth_path = truth_path.resolve()
    paired_root = paired_prediction_root.resolve() if paired_prediction_root else None
    if truth_path.is_relative_to(prediction_root) or (
        paired_root is not None and truth_path.is_relative_to(paired_root)
    ):
        raise ScoringError("truth sidecar must remain outside the frozen prediction roots")

    # This is the complete public gate for the generic scorer.  The strict
    # Stage-1 validator is owned by the design agent and can be supplied as a
    # pre-score PASS receipt below; no truth loader runs before this block.
    prepared_items = [_prepare_prediction_bundle(prediction_root)]
    if paired_root is not None:
        prepared_items.append(_prepare_prediction_bundle(paired_root))
    verify_prediction_bundles([item.root for item in prepared_items])
    pre_score_receipt = (
        _load_pre_score_receipt(
            pre_score_receipt_path,
            expected_implementation_commit=implementation_commit,
            prepared_items=prepared_items,
        )
        if pre_score_receipt_path is not None
        else None
    )

    # Create the score directory and its truth-free gate record before opening
    # evaluator truth.  Numeric files are created only after truth is read.
    out = create_only_directory(output_root.resolve())
    gate_path = out / "pre_score_gate.json"
    write_json_exclusive(
        gate_path,
        {
            "schema": "token-reconstruction.trr-p03-pre-score-gate.v1",
            "task_id": "TRR-P03",
            "status": "PASS",
            "truth_opened": False,
            "created_utc": _utc_now(),
            "prediction_bundles": [str(item.root) for item in prepared_items],
            "bundle_coverage": [
                {
                    "methods": list(item.methods),
                    "records": list(item.record_ids),
                    "anchor_records": list(item.anchor_record_ids),
                }
                for item in prepared_items
            ],
            "strict_validator_receipt": (
                file_record(pre_score_receipt_path.resolve())
                if pre_score_receipt_path is not None
                else None
            ),
        },
    )
    gate_path.chmod(0o444)

    # Truth is opened exactly once, after every supplied public bundle and the
    # optional strict receipt have passed their pre-score checks.
    pretruth_validation_seconds = float(time.perf_counter() - started)
    truth_open_started = time.perf_counter()
    truth_open_started_utc = _utc_now()
    truth = _load_truth(
        truth_path,
        truth_index_path.resolve() if truth_index_path else None,
    )
    truth_opened_utc = _utc_now()
    truth_opening_seconds = float(time.perf_counter() - truth_open_started)

    bundle_results: list[dict[str, Any]] = []
    all_per_record: list[dict[str, Any]] = []
    for index, prepared in enumerate(prepared_items):
        role = "primary" if index == 0 else f"paired_{index}"
        per_record, summaries, scores_by_method = _score_prepared(prepared, truth)
        for row in per_record:
            row = dict(row)
            row["target_bundle"] = role
            all_per_record.append(row)
        generic_pairs = _generic_pairs(
            scores_by_method,
            draws=int(bootstrap_draws),
            seed=int(bootstrap_seed),
        )
        design = _design_stats(
            scores_by_method,
            draws=int(bootstrap_draws),
            seed=int(bootstrap_seed),
            records_per_stratum=records_per_stratum,
        )
        bundle_results.append(
            {
                "target_bundle": role,
                "root": str(prepared.root),
                "methods": list(prepared.methods),
                "records": len(prepared.record_ids),
                "anchor_records": list(prepared.anchor_record_ids),
                "summaries": summaries,
                "pairs": generic_pairs,
                "projected_vs_a1": design,
            }
        )

    per_record_path = out / "per_record.jsonl"
    paired_path = out / "paired_statistics.json"
    evidence_path = out / "scoring_evidence.json"
    metrics_path = out / "metrics.json"
    # Write all numeric results before constructing their file records.  A
    # plotting/presentation failure after this point cannot discard scores.
    write_jsonl_exclusive(per_record_path, all_per_record)
    primary_result = bundle_results[0]
    write_json_exclusive(
        paired_path,
        {
            "schema": PAIRED_SCHEMA,
            "task_id": "TRR-P03",
            "truth_opened": True,
            "bootstrap_draws": int(bootstrap_draws),
            "bootstrap_seed": int(bootstrap_seed),
            "records_per_stratum": records_per_stratum,
            # Backward-compatible primary fields.
            "pairs": primary_result["pairs"],
            "projected_vs_a1": primary_result["projected_vs_a1"],
            "bundle_results": bundle_results,
            "paired_bundle_verified_before_truth": paired_root is not None,
        },
    )
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr-p03-scoring-evidence.v1",
            "task_id": "TRR-P03",
            "status": "SCORED_AFTER_PREDICTION_FREEZE",
            "truth_opened": True,
            "started_utc": started_utc,
            "truth_open_started_utc": truth_open_started_utc,
            "truth_opened_utc": truth_opened_utc,
            "ended_utc": _utc_now(),
            "command": {"argv": [str(value) for value in sys.argv], "cwd": os.getcwd()},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "implementation_commit": implementation_commit,
            },
            "prediction_bundles": [file_record(item.root / "freeze_receipt.json") for item in prepared_items],
            "pre_score_gate": file_record(gate_path),
            "strict_validator_receipt": (
                file_record(pre_score_receipt_path.resolve())
                if pre_score_receipt_path is not None
                else None
            ),
            "truth": file_record(truth_path),
            "truth_index": file_record(truth_index_path.resolve()) if truth_index_path else None,
            "output_numeric": {
                "per_record": file_record(per_record_path),
                "paired_statistics": file_record(paired_path),
            },
            "phases": {
                "pretruth_validation_seconds": pretruth_validation_seconds,
                "truth_opening_seconds": truth_opening_seconds,
                "total_seconds": float(time.perf_counter() - started),
            },
        },
    )
    metrics = {
        "schema": SCORING_SCHEMA,
        "task_id": "TRR-P03",
        "status": "SCORED_AFTER_PREDICTION_FREEZE",
        "truth_opened": True,
        "prediction_freeze": file_record(prediction_root / "freeze_receipt.json"),
        "paired_prediction_freeze": file_record(paired_root / "freeze_receipt.json") if paired_root else None,
        "truth": file_record(truth_path),
        "methods": list(prepared_items[0].methods),
        # Backward-compatible primary summaries plus complete paired results.
        "summaries": primary_result["summaries"],
        "bundle_summaries": {
            str(item["target_bundle"]): item["summaries"] for item in bundle_results
        },
        "paired_stats": file_record(paired_path),
        "per_record": file_record(per_record_path),
        "scoring_evidence": file_record(evidence_path),
        "pre_score_gate": file_record(gate_path),
        "strict_validator_receipt": (
            file_record(pre_score_receipt_path.resolve())
            if pre_score_receipt_path is not None
            else None
        ),
        "truth_records": len(truth),
        "target_bundles": [item["target_bundle"] for item in bundle_results],
    }
    write_json_exclusive(metrics_path, metrics)
    return {
        "metrics": file_record(metrics_path),
        "per_record": file_record(per_record_path),
        "paired_statistics": file_record(paired_path),
        "scoring_evidence": file_record(evidence_path),
        "pre_score_gate": file_record(gate_path),
        "strict_validator_receipt": (
            file_record(pre_score_receipt_path.resolve())
            if pre_score_receipt_path is not None
            else None
        ),
        "methods": list(prepared_items[0].methods),
        "target_bundles": [item["target_bundle"] for item in bundle_results],
        "records": len(truth),
        "truth_opened_after_freeze_verification": True,
    }


__all__ = [
    "A2_METHOD",
    "BASE_METHODS",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "PAIRED_SCHEMA",
    "SCORING_SCHEMA",
    "ScoringError",
    "score_prediction_bundle",
    "verify_prediction_bundles",
]
