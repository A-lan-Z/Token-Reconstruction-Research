#!/usr/bin/env python3
"""Truth-gated scorer for frozen P04 prediction JSONL files.

The scorer first validates the public panel, freeze receipt, prediction-file
bindings, expected methods/seeds, and unrestricted prediction geometry.  Only
after that gate passes does it read the two evaluator-private truth files.  It
never changes predictions and never returns token truth in its score artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


TASK_ID = "TRR-P04"
SELECTION_SCHEMA = "token-reconstruction.trr-p04-public-selection.v1"
SCORE_SCHEMA = "token-reconstruction.trr-p04-score.v1"
FREEZE_SCHEMA = "token-reconstruction.trr-p04-freeze.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr-p04-predictions.v1"
BOS_TOKEN_ID = 128000
VOCAB_SIZE = 128256
DEFAULT_CONDITIONS = ("public_base", "p04_evaluator_target_update_v1")
DEFAULT_METHODS = ("affine_same_data", "student_s", "student_h", "student_d")
DEFAULT_SEEDS = (1737, 2711)
PANEL_LENGTHS = (16, 32, 64, 128)
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


class ScoreError(ValueError):
    """Raised when a frozen prediction or truth contract is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _safe_environment() -> dict[str, str]:
    import os

    return {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ScoreError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ScoreError(f"{description} must be an object")
    return value


def _read_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ScoreError(f"{description} is unavailable: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ScoreError(f"{description} line {line_number} is invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ScoreError(f"{description} line {line_number} is not an object")
                rows.append(value)
    except OSError as exc:
        raise ScoreError(f"unable to read {description}: {path}") from exc
    return rows


def _descriptor(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ScoreError(f"prediction file is unavailable: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _resolve_descriptor_path(value: Any, *, base: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ScoreError(f"{description} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _validate_panel(panel: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if panel.get("schema") != SELECTION_SCHEMA or panel.get("task_id") != TASK_ID:
        raise ScoreError("P04 panel identity changed")
    fresh = panel.get("pools", {}).get("fresh_evaluation") if isinstance(panel.get("pools"), Mapping) else None
    records = fresh.get("records") if isinstance(fresh, Mapping) else None
    if not isinstance(records, list):
        nested = panel.get("panel")
        records = nested.get("records") if isinstance(nested, Mapping) else None
    if not isinstance(records, list) or len(records) != 72:
        raise ScoreError("P04 panel must contain exactly 72 fresh records")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ScoreError(f"panel record {index} is malformed")
        record_id = row.get("record_id")
        style = row.get("style")
        length = row.get("length_stratum")
        if not isinstance(record_id, str) or not record_id:
            raise ScoreError(f"panel record {index} has no record_id")
        if record_id in result:
            raise ScoreError(f"panel record ID is duplicated: {record_id}")
        if not isinstance(style, str) or not style:
            raise ScoreError(f"panel record {index} has no style")
        if length not in PANEL_LENGTHS:
            raise ScoreError(f"panel record {index} has invalid length stratum")
        forbidden = {"token_ids", "input_ids", "labels", "source_text", "truth", "oracle"}
        if forbidden.intersection(row):
            raise ScoreError(f"panel record {index} contains private source/truth fields")
        result[record_id] = {
            "record_id": record_id,
            "style": style,
            "length_stratum": int(length),
            "anchor": bool(row.get("anchor", False)),
        }
    if len({row["style"] for row in result.values()}) < 3:
        raise ScoreError("P04 panel does not contain three styles")
    for length in PANEL_LENGTHS:
        if sum(row["length_stratum"] == length for row in result.values()) != 18:
            raise ScoreError(f"P04 panel length quota changed for {length}")
    anchors = [row for row in result.values() if row["anchor"]]
    if len(anchors) != 12:
        raise ScoreError("P04 panel anchor quota changed")
    return result


def _required_groups() -> tuple[tuple[str, int | None, str, bool], ...]:
    """Return the exact eight state identities per target plus two anchors."""

    return tuple(
        [
            (method, seed, condition, False)
            for condition in DEFAULT_CONDITIONS
            for seed in DEFAULT_SEEDS
            for method in DEFAULT_METHODS
        ]
        + [("native_a1_a2", None, condition, True) for condition in DEFAULT_CONDITIONS]
    )


def _expected_groups(freeze: Mapping[str, Any]) -> tuple[tuple[str, int | None, str, bool], ...]:
    required = _required_groups()
    raw = freeze.get("prediction_groups")
    if raw is None:
        return required
    if not isinstance(raw, list) or not raw:
        raise ScoreError("freeze prediction_groups is malformed")
    parsed: list[tuple[str, int | None, str, bool]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ScoreError(f"freeze prediction group {index} is malformed")
        method = row.get("method_id")
        seed = row.get("seed")
        condition = row.get("condition")
        anchor = bool(row.get("anchor", False))
        if not isinstance(method, str) or not method or not isinstance(condition, str) or not condition:
            raise ScoreError(f"freeze prediction group {index} has invalid identity")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ScoreError(f"freeze prediction group {index} has invalid seed")
        group = (method, seed, condition, anchor)
        if group in parsed:
            raise ScoreError("freeze prediction groups are duplicated")
        parsed.append(group)
    if len(parsed) != len(required) or set(parsed) != set(required):
        raise ScoreError("freeze prediction_groups must contain all eight states per target and both native anchors")
    return required


def _validate_freeze(
    freeze: Mapping[str, Any],
    *,
    freeze_base: Path,
    panel_path: Path,
    panel_sha256: str,
    prediction_files: Sequence[Path],
    prediction_descriptors: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, int | None, str, bool], ...]:
    if freeze.get("task_id") != TASK_ID or freeze.get("schema") not in (FREEZE_SCHEMA, None):
        raise ScoreError("P04 freeze identity changed")
    if freeze.get("status") not in ("FROZEN_BEFORE_TRUTH", "FROZEN"):
        raise ScoreError("P04 predictions are not frozen before truth")
    for key in ("panel_frozen", "predictions_frozen", "all_states_frozen", "truth_open_allowed"):
        if freeze.get(key) is not True:
            raise ScoreError(f"P04 freeze is missing required flag {key}")
    declared_panel = freeze.get("panel")
    if isinstance(declared_panel, Mapping):
        if declared_panel.get("path") not in (str(panel_path), panel_path.name, str(panel_path.resolve())):
            raise ScoreError("freeze panel path binding changed")
        if declared_panel.get("sha256") != panel_sha256:
            raise ScoreError("freeze panel hash binding changed")
    elif freeze.get("panel_sha256") != panel_sha256:
        raise ScoreError("freeze panel hash binding is absent or changed")
    declared_files = freeze.get("prediction_files")
    if not isinstance(declared_files, list) or len(declared_files) != len(prediction_files):
        raise ScoreError("freeze prediction-file bindings are incomplete")
    declared_by_path: dict[str, Mapping[str, Any]] = {}
    for row in declared_files:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ScoreError("freeze prediction-file descriptor is malformed")
        declared_by_path[row["path"]] = row
    for path, actual in zip(prediction_files, prediction_descriptors):
        candidates = (str(path), str(path.resolve()), path.name)
        declared = next((declared_by_path.get(candidate) for candidate in candidates if candidate in declared_by_path), None)
        if declared is None:
            raise ScoreError(f"prediction file is not bound by freeze: {path}")
        if declared.get("bytes") != actual["bytes"] or declared.get("sha256") != actual["sha256"]:
            raise ScoreError(f"prediction file hash or size changed: {path}")
    required_states = {(method, seed) for seed in DEFAULT_SEEDS for method in DEFAULT_METHODS}
    state_rows = freeze.get("state_files")
    if not isinstance(state_rows, list) or len(state_rows) != len(required_states):
        raise ScoreError("freeze must bind all eight student/reference state files")
    seen_states: set[tuple[str, int]] = set()
    for index, row in enumerate(state_rows):
        if not isinstance(row, Mapping):
            raise ScoreError(f"freeze state descriptor {index} is malformed")
        method = row.get("method_id")
        seed = row.get("seed")
        if method not in DEFAULT_METHODS or seed not in DEFAULT_SEEDS:
            raise ScoreError(f"freeze state descriptor {index} has an unexpected method or seed")
        identity = (str(method), int(seed))
        if identity in seen_states:
            raise ScoreError("freeze state descriptors are duplicated")
        seen_states.add(identity)
        path = _resolve_descriptor_path(row.get("path"), base=freeze_base, description="student state")
        actual = _descriptor(path)
        if row.get("bytes") != actual["bytes"] or row.get("sha256") != actual["sha256"]:
            raise ScoreError(f"student state hash or size changed: {path}")
    if seen_states != required_states:
        raise ScoreError("freeze state descriptors do not cover every method and seed")
    return _expected_groups(freeze)


def _prediction_row(
    row: Mapping[str, Any],
    *,
    panel: Mapping[str, Mapping[str, Any]],
    description: str,
) -> tuple[tuple[str, int | None, str, bool], str, list[int]]:
    allowed = {"schema", "method_id", "seed", "condition", "record_id", "predicted_token_ids", "anchor"}
    if set(row) - allowed:
        raise ScoreError(f"{description} contains unapproved fields")
    method = row.get("method_id")
    seed = row.get("seed")
    condition = row.get("condition")
    record_id = row.get("record_id")
    anchor = bool(row.get("anchor", False))
    if row.get("schema") not in (None, PREDICTION_SCHEMA):
        raise ScoreError(f"{description} schema changed")
    if not isinstance(method, str) or not method or not isinstance(condition, str) or not condition:
        raise ScoreError(f"{description} has invalid method/condition")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ScoreError(f"{description} has invalid seed")
    if not isinstance(record_id, str) or record_id not in panel:
        raise ScoreError(f"{description} references an unknown panel record")
    values = row.get("predicted_token_ids")
    if not isinstance(values, list) or not values:
        raise ScoreError(f"{description} has no predicted_token_ids")
    predicted: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= VOCAB_SIZE:
            raise ScoreError(f"{description} contains an invalid predicted token ID")
        predicted.append(int(value))
    length = int(panel[record_id]["length_stratum"])
    if len(predicted) != length:
        raise ScoreError(f"{description} prediction length is {len(predicted)}, expected {length}")
    if anchor and not bool(panel[record_id]["anchor"]):
        raise ScoreError(f"{description} anchor group references a non-anchor record")
    return (method, seed, condition, anchor), record_id, predicted


def _load_predictions(
    paths: Sequence[Path], *, panel: Mapping[str, Mapping[str, Any]], expected: Sequence[tuple[str, int | None, str, bool]]
) -> dict[tuple[str, int | None, str, bool], dict[str, list[int]]]:
    expected_set = set(expected)
    result: dict[tuple[str, int | None, str, bool], dict[str, list[int]]] = defaultdict(dict)
    for path in paths:
        for line_number, row in enumerate(_read_jsonl(path, description=f"prediction file {path}"), start=1):
            group, record_id, predicted = _prediction_row(
                row,
                panel=panel,
                description=f"prediction file {path} line {line_number}",
            )
            if group not in expected_set:
                raise ScoreError(f"prediction group is not frozen: {group}")
            if record_id in result[group]:
                raise ScoreError(f"prediction record is duplicated in group {group}: {record_id}")
            result[group][record_id] = predicted
    for group in expected:
        if group not in result:
            raise ScoreError(f"frozen prediction group is missing: {group}")
        if set(result[group]) != set(panel if not group[3] else {key: value for key, value in panel.items() if value["anchor"]}):
            raise ScoreError(f"prediction group does not cover its required panel records: {group}")
    return dict(result)


def _truth_paths(
    freeze: Mapping[str, Any], *, truth_dir: Path, conditions: Iterable[str]
) -> dict[str, Path]:
    declared = freeze.get("truth_files")
    result: dict[str, Path] = {}
    if declared is not None:
        if not isinstance(declared, list):
            raise ScoreError("freeze truth_files is malformed")
        for row in declared:
            if not isinstance(row, Mapping) or not isinstance(row.get("condition"), str):
                raise ScoreError("freeze truth-file descriptor is malformed")
            result[row["condition"]] = _resolve_descriptor_path(
                row.get("path"), base=truth_dir, description="truth file"
            )
    for condition in conditions:
        result.setdefault(condition, (truth_dir / f"{condition}.jsonl").resolve())
    return result


def _load_truth_after_gate(
    truth_paths: Mapping[str, Path], *, panel: Mapping[str, Mapping[str, Any]], conditions: Iterable[str]
) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    for condition in conditions:
        path = truth_paths[condition]
        rows = _read_jsonl(path, description=f"private truth for {condition}")
        local: dict[str, list[int]] = {}
        for line_number, row in enumerate(rows, start=1):
            if set(row) != {"record_id", "token_ids"}:
                raise ScoreError(f"private truth {condition} line {line_number} has the wrong schema")
            record_id = row.get("record_id")
            values = row.get("token_ids")
            if not isinstance(record_id, str) or record_id not in panel:
                raise ScoreError(f"private truth {condition} line {line_number} has an unknown record")
            if record_id in local:
                raise ScoreError(f"private truth {condition} duplicates {record_id}")
            if not isinstance(values, list):
                raise ScoreError(f"private truth {condition} line {line_number} has no token list")
            length = int(panel[record_id]["length_stratum"])
            if len(values) != length:
                raise ScoreError(f"private truth {condition} length changed for {record_id}")
            truth: list[int] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= VOCAB_SIZE:
                    raise ScoreError(f"private truth {condition} has an invalid token ID")
                truth.append(int(value))
            local[record_id] = truth
        if set(local) != set(panel):
            raise ScoreError(f"private truth {condition} does not cover the exact fresh panel")
        result[condition] = local
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean_difference(
    left: Mapping[str, float], right: Mapping[str, float], *, seed: int, samples: int = 2000
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ScoreError("paired bootstrap has no common source records")
    observed = sum(left[key] - right[key] for key in ids) / len(ids)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        draws.append(sum(left[key] - right[key] for key in sample) / len(sample))
    return {
        "cluster_unit": "source_record_id",
        "clusters": len(ids),
        "kind": "mean_record_accuracy_difference_secondary",
        "observed_mean_record_accuracy_difference": observed,
        "bootstrap_samples": samples,
        "seed": seed,
        "ci_95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "paired_target_rows_kept_together": True,
    }


def _bootstrap_token_weighted_difference(
    strata: Mapping[str, Sequence[tuple[int, int, int]]],
    *,
    seed: int = 20260908,
    samples: int = 10000,
) -> dict[str, Any]:
    """Bootstrap a token-accuracy difference, stratified by style/length cell.

    Each tuple is ``(left_correct_across_seeds, right_correct_across_seeds,
    scored_tokens_across_seeds)`` for one source record.  Both training seeds
    therefore remain paired inside the source cluster and each resample keeps
    the twelve style/length strata represented.
    """

    if not strata or any(not rows for rows in strata.values()):
        raise ScoreError("primary bootstrap requires all style/length strata")
    if samples <= 0 or seed < 0:
        raise ScoreError("primary bootstrap configuration is invalid")
    rows = [tuple(row) for values in strata.values() for row in values]
    observed_numerator = sum(left - right for left, right, _ in rows)
    observed_denominator = sum(tokens for _, _, tokens in rows)
    if observed_denominator <= 0:
        raise ScoreError("primary bootstrap has no scored tokens")
    observed = observed_numerator / observed_denominator
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        numerator = 0
        denominator = 0
        for values in strata.values():
            for _ in values:
                left, right, tokens = values[rng.randrange(len(values))]
                numerator += left - right
                denominator += tokens
        draws.append(numerator / denominator)
    return {
        "kind": "token_weighted_accuracy_difference",
        "cluster_unit": "source_record_id",
        "strata": len(strata),
        "clusters": len(rows),
        "observed_token_accuracy_difference": observed,
        "bootstrap_samples": samples,
        "seed": seed,
        "ci_95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "seed_pairing": list(DEFAULT_SEEDS),
        "paired_target_rows_kept_together": True,
    }


def _primary_pairwise(
    groups: Mapping[tuple[str, int | None, str, bool], Mapping[str, list[int]]],
    truths: Mapping[str, Mapping[str, list[int]]],
    panel: Mapping[str, Mapping[str, Any]],
    expected: Sequence[tuple[str, int | None, str, bool]],
) -> list[dict[str, Any]]:
    """Compute whole-target D/H/S/affine comparisons with paired seeds."""

    pairs = (
        ("student_d", "student_h", "D", "H"),
        ("student_d", "student_s", "D", "S"),
        ("student_h", "student_s", "H", "S"),
        ("student_s", "affine_same_data", "S", "affine"),
    )
    result: list[dict[str, Any]] = []
    styles = sorted({row["style"] for row in panel.values()})
    for condition in DEFAULT_CONDITIONS:
        for left_method, right_method, left_label, right_label in pairs:
            strata: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
            for style in styles:
                for length in PANEL_LENGTHS:
                    cell = [
                        record_id
                        for record_id, row in panel.items()
                        if row["style"] == style and row["length_stratum"] == length
                    ]
                    for record_id in cell:
                        left_correct = 0
                        right_correct = 0
                        for seed in DEFAULT_SEEDS:
                            left_group = (left_method, seed, condition, False)
                            right_group = (right_method, seed, condition, False)
                            if left_group not in groups or right_group not in groups:
                                raise ScoreError(f"primary comparison group is missing for {condition}")
                            expected_tokens = truths[condition][record_id]
                            left_correct += sum(
                                int(a == b)
                                for a, b in zip(groups[left_group][record_id], expected_tokens)
                            )
                            right_correct += sum(
                                int(a == b)
                                for a, b in zip(groups[right_group][record_id], expected_tokens)
                            )
                        strata[f"{style}__length{length}"].append(
                            (left_correct, right_correct, 2 * length)
                        )
            result.append(
                {
                    "condition": condition,
                    "left": left_label,
                    "right": right_label,
                    "left_method_id": left_method,
                    "right_method_id": right_method,
                    "scope": "whole_target_descriptive_with_stratified_record_cluster_bootstrap",
                    "bootstrap": _bootstrap_token_weighted_difference(strata),
                }
            )
    return result


def _anchor_table(
    groups: Mapping[tuple[str, int | None, str, bool], Mapping[str, list[int]]],
    truths: Mapping[str, Mapping[str, list[int]]],
    panel: Mapping[str, Mapping[str, Any]],
    expected: Sequence[tuple[str, int | None, str, bool]],
) -> list[dict[str, Any]]:
    """Return an explicit per-target A1+A2-anchor denominator table."""

    result: list[dict[str, Any]] = []
    anchor_ids = {record_id for record_id, row in panel.items() if row["anchor"]}
    for condition in DEFAULT_CONDITIONS:
        entries: list[dict[str, Any]] = []
        group_order = [
            group
            for group in expected
            if group[2] == condition and (group[3] or (group[0] in DEFAULT_METHODS and group[1] in DEFAULT_SEEDS))
        ]
        for group in group_order:
            if group not in groups:
                raise ScoreError(f"anchor table group is missing: {group}")
            selected = {record_id: panel[record_id] for record_id in anchor_ids}
            selected_predictions = {record_id: groups[group][record_id] for record_id in anchor_ids}
            summary, _ = _score_group(selected_predictions, truths[condition], selected, anchor_only=True)
            entries.append(
                {
                    "method_id": group[0],
                    "seed": group[1],
                    "native_anchor": group[3],
                    "records": summary["records"],
                    "scored_tokens": summary["scored_tokens"],
                    "correct_tokens": summary["correct_tokens"],
                    "token_accuracy": summary["token_accuracy"],
                    "exact_records": summary["exact_records"],
                }
            )
        result.append(
            {
                "condition": condition,
                "anchor_records": len(anchor_ids),
                "scored_tokens_per_group": len(anchor_ids) * 32,
                "denominator_scope": "12 source records in the predeclared 32-token anchor; separate from full panel",
                "methods": entries,
            }
        )
    return result


def _score_group(
    predictions: Mapping[str, list[int]], truth: Mapping[str, list[int]], panel: Mapping[str, Mapping[str, Any]], *, anchor_only: bool = False
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    selected_ids = [record_id for record_id, row in panel.items() if (not anchor_only or row["anchor"])]
    if set(predictions) != set(selected_ids):
        raise ScoreError("prediction/truth group coverage changed during scoring")
    total = 0
    correct = 0
    exact = 0
    per_record: dict[str, float] = {}
    cells: dict[str, dict[str, Any]] = {}
    for record_id in sorted(selected_ids):
        predicted = predictions[record_id]
        expected = truth[record_id]
        row = panel[record_id]
        matches = sum(int(a == b) for a, b in zip(predicted, expected))
        count = len(expected)
        total += count
        correct += matches
        exact += int(matches == count)
        per_record[record_id] = matches / count
        cell_id = f"{row['style']}__length{row['length_stratum']}"
        cell = cells.setdefault(cell_id, {"style": row["style"], "length_stratum": row["length_stratum"], "records": 0, "scored_tokens": 0, "correct_tokens": 0, "exact_records": 0})
        cell["records"] += 1
        cell["scored_tokens"] += count
        cell["correct_tokens"] += matches
        cell["exact_records"] += int(matches == count)
    for cell in cells.values():
        cell["token_accuracy"] = cell["correct_tokens"] / cell["scored_tokens"]
    return {
        "records": len(selected_ids),
        "scored_tokens": total,
        "correct_tokens": correct,
        "token_accuracy": correct / total if total else 0.0,
        "exact_records": exact,
        "exact_record_rate": exact / len(selected_ids) if selected_ids else 0.0,
        "cells": dict(sorted(cells.items())),
    }, {"all": per_record}


def _pairwise(
    groups: Mapping[tuple[str, int | None, str, bool], Mapping[str, list[int]]],
    truths: Mapping[str, Mapping[str, list[int]]],
    panel: Mapping[str, Mapping[str, Any]],
    expected: Sequence[tuple[str, int | None, str, bool]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    pairs = (("student_d", "student_h", "D", "H"), ("student_d", "student_s", "D", "S"), ("student_h", "student_s", "H", "S"), ("student_s", "affine_same_data", "S", "affine"))
    styles = sorted({row["style"] for row in panel.values()})
    for condition in sorted({group[2] for group in expected if not group[3]}):
        for seed in DEFAULT_SEEDS:
            available = {group[0]: group for group in expected if group[1] == seed and group[2] == condition and not group[3]}
            for style in styles:
                for length in PANEL_LENGTHS:
                    record_ids = [record_id for record_id, row in panel.items() if row["style"] == style and row["length_stratum"] == length]
                    if not record_ids:
                        continue
                    per_method: dict[str, dict[str, float]] = {}
                    for method, group in available.items():
                        if group not in groups:
                            continue
                        per_method[method] = {
                            record_id: sum(int(a == b) for a, b in zip(groups[group][record_id], truths[condition][record_id])) / length
                            for record_id in record_ids
                        }
                    for left_method, right_method, left_label, right_label in pairs:
                        if left_method not in per_method or right_method not in per_method:
                            continue
                        result.append(
                            {
                                "condition": condition,
                                "seed": seed,
                                "style": style,
                                "length_stratum": length,
                                "left": left_label,
                                "right": right_label,
                                "left_method_id": left_method,
                                "right_method_id": right_method,
                                "bootstrap": _bootstrap_mean_difference(
                                    per_method[left_method], per_method[right_method], seed=2711
                                ),
                            }
                        )
    return result


def score_predictions(
    *,
    panel_path: Path,
    freeze_path: Path,
    prediction_paths: Sequence[Path],
    truth_dir: Path,
    output_path: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    started = time.monotonic()
    panel = _load_json(panel_path, description="P04 public panel")
    panel_records = _validate_panel(panel)
    panel_sha = _sha256_file(panel_path)
    freeze = _load_json(freeze_path, description="P04 freeze receipt")
    descriptors = [_descriptor(path) for path in prediction_paths]
    expected = _validate_freeze(
        freeze,
        freeze_base=freeze_path.parent.resolve(),
        panel_path=panel_path.resolve(),
        panel_sha256=panel_sha,
        prediction_files=prediction_paths,
        prediction_descriptors=descriptors,
    )
    groups = _load_predictions(prediction_paths, panel=panel_records, expected=expected)
    # This is the truth-opening gate.  No private path is opened before every
    # public binding and every frozen prediction group above has validated.
    conditions = sorted({group[2] for group in expected})
    truth_paths = _truth_paths(freeze, truth_dir=truth_dir.expanduser().resolve(), conditions=conditions)
    truths = _load_truth_after_gate(truth_paths, panel=panel_records, conditions=conditions)
    summaries: dict[str, Any] = {}
    group_per_record: dict[tuple[str, int | None, str, bool], dict[str, dict[str, float]]] = {}
    for group in expected:
        method, seed, condition, anchor = group
        summary, per_record = _score_group(groups[group], truths[condition], panel_records, anchor_only=anchor)
        key = f"{method}__seed{seed if seed is not None else 'none'}__{condition}{'__anchor' if anchor else ''}"
        summaries[key] = {
            "method_id": method,
            "seed": seed,
            "condition": condition,
            "anchor": anchor,
            "metrics": summary,
        }
        group_per_record[group] = per_record
    pairwise = _pairwise(groups, truths, panel_records, expected)
    primary_pairwise = _primary_pairwise(groups, truths, panel_records, expected)
    anchor_table = _anchor_table(groups, truths, panel_records, expected)
    score = {
        "schema": SCORE_SCHEMA,
        "task_id": TASK_ID,
        "status": "SCORED_AFTER_TRUTH_GATE",
        "created_utc": _utc_now(),
        "panel": {"path": str(panel_path), "bytes": panel_path.stat().st_size, "sha256": panel_sha},
        "freeze": {"path": str(freeze_path), "bytes": freeze_path.stat().st_size, "sha256": _sha256_file(freeze_path)},
        "prediction_files": descriptors,
        "truth_gate": {
            "verified_before_truth": True,
            "truth_opened_after_gate": True,
            "prediction_files_read_before_truth": True,
            "prediction_files_rewritten": False,
        },
        "truth_files": {
            condition: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for condition, path in truth_paths.items()
        },
        "summaries": summaries,
        "pairwise_record_cluster_bootstrap": pairwise,
        "primary_target_pairwise_token_bootstrap": primary_pairwise,
        "native_anchor_table": anchor_table,
        "uncertainty_contract": {
            "cluster_unit": "source_record_id",
            "paired_target_conditions_are_not_independent_clusters": True,
            "pooled_overall_inference": False,
            "per_target_style_length_required": True,
        },
        "execution": {
            "argv": list(argv),
            "python": sys.executable,
            "started_utc": None,
            "ended_utc": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "max_rss_bytes": _max_rss_bytes(),
            "safe_environment": _safe_environment(),
            "truth_accessed": True,
        },
    }
    score["execution"]["started_utc"] = score["created_utc"]
    if output_path.exists() or output_path.is_symlink():
        raise ScoreError(f"refusing to overwrite score output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return score


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path, action="append", required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        score_predictions(
            panel_path=args.panel.expanduser().resolve(),
            freeze_path=args.freeze.expanduser().resolve(),
            prediction_paths=[path.expanduser().resolve() for path in args.prediction_file],
            truth_dir=args.truth_dir,
            output_path=args.output.expanduser().resolve(),
            argv=list(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]),
        )
    except ScoreError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"output": str(args.output.resolve()), "status": "SCORED_AFTER_TRUTH_GATE"}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

