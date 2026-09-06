#!/usr/bin/env python3
"""Derive compact Stage-1 reporting summaries from frozen score artifacts.

This report-only command reads the already scored Stage-1 JSONL truth and the
frozen prediction/score artifacts.  It does not rerun inference or scoring,
does not write inside either prediction root or the score root, and rejects
Stage-2/holdout paths.  Numeric JSON/CSV summaries are written before the
optional static figure so a plotting failure cannot erase the report data.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
for _path in (_SOURCE_ROOT, _SOURCE_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from token_reconstruction.trr_p03.io import (  # noqa: E402
    BOS_TOKEN_ID,
    P03IOError,
    create_only_directory,
    file_record,
    read_json,
    read_jsonl,
    verify_freeze_receipt,
    write_json_exclusive,
)


TASK_ID = "TRR-P03"
SUPPLEMENT_SCHEMA = "token-reconstruction.trr-p03-stage1-score-supplement.v1"
BUNDLES = ("primary", "paired_1")
BASE_METHODS = (
    "raw_boundary.cosine",
    "projected_boundary.cosine",
    "historical_a1.cosine",
)
A2_METHOD = "historical_a1_a2_anchor.cosine"
METHODS = BASE_METHODS + (A2_METHOD,)
ANCHOR_IDS = (
    "p03-s1-r0007",
    "p03-s1-r0009",
    "p03-s1-r0011",
    "p03-s1-r0012",
)
EXPECTED_IDS = tuple(f"p03-s1-r{index:04d}" for index in range(1, 25))
EXPECTED_LENGTHS = (16, 39, 64, 128)
POSITION_BINS = (
    ("1-16", 1, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-96", 65, 96),
    ("97-128", 97, 128),
)


class SupplementError(RuntimeError):
    """Raised when frozen Stage-1 report inputs are incomplete or mutable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SupplementError(message)


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    _require(not path.is_symlink() and path.is_file(), f"{label} is not a regular file: {path}")
    return path


def _read_stage1_truth(path: Path) -> dict[str, list[int]]:
    path = _regular_file(path, "Stage-1 truth")
    lower_parts = {part.lower() for part in path.parts}
    _require("stage1" in lower_parts, "truth path is not explicitly under Stage 1")
    _require("stage2" not in lower_parts and "holdout" not in lower_parts, "Stage-2/holdout truth is forbidden")
    try:
        rows = read_jsonl(path)
    except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
        raise SupplementError("Stage-1 truth JSONL is invalid") from exc
    result: dict[str, list[int]] = {}
    for row in rows:
        _require(set(row) == {"record_id", "token_ids"}, "truth JSONL fields changed")
        record_id = row.get("record_id")
        token_ids = row.get("token_ids")
        _require(isinstance(record_id, str) and record_id in EXPECTED_IDS, "truth record ID is invalid")
        _require(record_id not in result, "truth record IDs are duplicated")
        _require(isinstance(token_ids, list) and len(token_ids) >= 2, "truth sequence is invalid")
        values = [int(value) for value in token_ids]
        _require(values[0] == BOS_TOKEN_ID, f"truth BOS changed for {record_id}")
        _require(all(0 <= value < 128256 for value in values), f"truth vocabulary changed for {record_id}")
        result[record_id] = values
    _require(tuple(result) == EXPECTED_IDS, "truth record order or coverage changed")
    return result


def _read_panel(path: Path) -> dict[str, dict[str, Any]]:
    path = _regular_file(path, "Stage-1 evaluator panel")
    try:
        panel = read_json(path)
    except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
        raise SupplementError("Stage-1 evaluator panel is invalid") from exc
    _require(isinstance(panel, Mapping), "evaluator panel is not an object")
    _require(panel.get("truth_opened") is False, "evaluator panel is truth-opened")
    rows = panel.get("records")
    _require(isinstance(rows, list) and len(rows) == len(EXPECTED_IDS), "evaluator panel record count changed")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "evaluator panel row is malformed")
        record_id = row.get("record_id")
        _require(isinstance(record_id, str) and record_id in EXPECTED_IDS, "evaluator panel record ID is invalid")
        _require(record_id not in result, "evaluator panel IDs are duplicated")
        stage = str(row.get("stage", ""))
        _require(stage in {"s1", "stage1"}, f"panel row is outside Stage 1: {record_id}")
        length = int(row.get("scored_tokens", -1))
        _require(length in EXPECTED_LENGTHS, f"panel length is undeclared for {record_id}")
        style = row.get("style")
        _require(isinstance(style, str) and style, f"panel style is missing for {record_id}")
        result[record_id] = {"length": length, "style": style}
    _require(tuple(result) == EXPECTED_IDS, "evaluator panel order or coverage changed")
    return result


def _read_score_rows(path: Path, panel: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = _regular_file(path, "per-record score file")
    try:
        rows = read_jsonl(path)
    except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
        raise SupplementError("per-record score JSONL is invalid") from exc
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in rows:
        _require(isinstance(raw, Mapping), "per-record score row is malformed")
        bundle = raw.get("target_bundle")
        method = raw.get("method")
        record_id = raw.get("record_id")
        _require(bundle in BUNDLES, f"unexpected target bundle in score rows: {bundle}")
        _require(method in METHODS, f"unexpected method in score rows: {method}")
        _require(isinstance(record_id, str) and record_id in panel, "score record ID is invalid")
        _require(raw.get("truth_opened") is True, "score row was not truth-scored")
        length = int(raw.get("length", -1))
        _require(length == int(panel[record_id]["length"]), f"score length changed for {record_id}")
        correctness = raw.get("correctness")
        _require(isinstance(correctness, list) and len(correctness) == length, f"score correctness geometry changed for {record_id}")
        _require(all(isinstance(value, bool) for value in correctness), f"score correctness values changed for {record_id}")
        correct = int(raw.get("correct_tokens", -1))
        _require(correct == sum(correctness), f"score count disagrees with correctness for {record_id}")
        ties = raw.get("top1_tie_count")
        _require(isinstance(ties, list) and len(ties) == length, f"score tie geometry changed for {record_id}")
        _require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 1 for value in ties), f"score tie values changed for {record_id}")
        key = (str(bundle), str(method), record_id)
        _require(key not in result, f"duplicate score row: {key}")
        result[key] = {
            "target_bundle": str(bundle),
            "method": str(method),
            "record_id": record_id,
            "length": length,
            "style": str(panel[record_id]["style"]),
            "correctness": [bool(value) for value in correctness],
            "correct_tokens": correct,
            "exact": bool(raw.get("exact_sequence_match")),
            "tie_counts": [int(value) for value in ties],
        }
    for bundle in BUNDLES:
        for method in METHODS:
            expected = ANCHOR_IDS if method == A2_METHOD else EXPECTED_IDS
            actual = tuple(record_id for b, m, record_id in result if b == bundle and m == method)
            _require(set(actual) == set(expected), f"score coverage changed for {bundle}/{method}")
    return result


def _read_prediction_rows(
    root: Path,
    *,
    expected_bundle: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    root = root.resolve()
    try:
        freeze = verify_freeze_receipt(root)
    except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
        raise SupplementError(f"prediction freeze failed: {root}") from exc
    _require(freeze.get("truth_opened") is False, f"prediction root is truth-opened: {root}")
    rows_path = _regular_file(root / "predictions.jsonl", f"{expected_bundle} prediction rows")
    try:
        rows = read_jsonl(rows_path)
    except (P03IOError, OSError, ValueError, TypeError, KeyError) as exc:
        raise SupplementError(f"prediction rows are invalid: {rows_path}") from exc
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        method = raw.get("method")
        record_id = raw.get("record_id")
        _require(method in METHODS, f"unexpected prediction method: {method}")
        _require(isinstance(record_id, str) and record_id in EXPECTED_IDS, "prediction record ID is invalid")
        tokens = raw.get("prediction_tokens")
        _require(isinstance(tokens, list) and len(tokens) >= 2, "prediction token geometry is invalid")
        ties = raw.get("top1_tie_count")
        _require(isinstance(ties, list) and len(ties) == len(tokens) - 1, "prediction tie geometry is invalid")
        _require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 1 for value in ties), "prediction tie values are invalid")
        key = (str(method), record_id)
        _require(key not in result, f"duplicate prediction row: {key}")
        result[key] = {
            "tokens": [int(value) for value in tokens],
            "score_units": str(raw.get("score_units", "unspecified")),
            "tie_counts": [int(value) for value in ties],
        }
    expected_count = len(EXPECTED_IDS) * 3 + len(ANCHOR_IDS)
    _require(len(result) == expected_count, f"prediction row count changed for {expected_bundle}")
    return result, {
        "root": str(root),
        "freeze_receipt": file_record(root / "freeze_receipt.json"),
        "prediction_rows": file_record(rows_path),
    }


def _record_summary(rows: Iterable[Mapping[str, Any]], *, selected_positions: Iterable[int] | None = None) -> dict[str, Any]:
    values = list(rows)
    positions = None if selected_positions is None else tuple(int(value) for value in selected_positions)
    correct = 0
    scored = 0
    ties = 0
    exact_records = 0
    for row in values:
        correctness = row["correctness"]
        tie_counts = row["tie_counts"]
        chosen = range(len(correctness)) if positions is None else (index - 1 for index in positions if 1 <= index <= len(correctness))
        selected = [index for index in chosen if 0 <= index < len(correctness)]
        correct += sum(bool(correctness[index]) for index in selected)
        scored += len(selected)
        ties += sum(int(tie_counts[index]) > 1 for index in selected)
        if positions is None:
            exact_records += int(bool(row["exact"]))
    return {
        "records": len(values),
        "scored_tokens": int(scored),
        "correct_tokens": int(correct),
        "token_accuracy": float(correct / scored) if scored else 0.0,
        "exact_records": int(exact_records) if positions is None else None,
        "exact_record_rate": float(exact_records / len(values)) if values and positions is None else None,
        "tie_positions": int(ties),
    }


def _position_summary(rows: Iterable[Mapping[str, Any]], position: int) -> dict[str, Any]:
    return _record_summary(rows, selected_positions=(position,))


def _pair_summary(
    projected: Mapping[str, Mapping[str, Any]],
    a1: Mapping[str, Mapping[str, Any]],
    record_ids: Iterable[str],
    *,
    selected_positions: Iterable[int] | None = None,
) -> dict[str, Any]:
    ids = [record_id for record_id in record_ids if record_id in projected and record_id in a1]
    positions = None if selected_positions is None else tuple(int(value) for value in selected_positions)
    projected_correct = a1_correct = gain = regression = both_correct = both_wrong = equal = scored = 0
    for record_id in ids:
        p = projected[record_id]["correctness"]
        a = a1[record_id]["correctness"]
        chosen = range(len(p)) if positions is None else (index - 1 for index in positions if 1 <= index <= len(p))
        for index in chosen:
            if not (0 <= index < len(p) and index < len(a)):
                continue
            pv, av = bool(p[index]), bool(a[index])
            projected_correct += int(pv)
            a1_correct += int(av)
            gain += int(pv and not av)
            regression += int(av and not pv)
            both_correct += int(pv and av)
            both_wrong += int(not pv and not av)
            equal += int(pv == av)
            scored += 1
    return {
        "records": len(ids),
        "scored_tokens": int(scored),
        "projected_correct_tokens": int(projected_correct),
        "a1_correct_tokens": int(a1_correct),
        "token_delta": int(projected_correct - a1_correct),
        "projected_accuracy": float(projected_correct / scored) if scored else 0.0,
        "a1_accuracy": float(a1_correct / scored) if scored else 0.0,
        "gain_tokens": int(gain),
        "regression_tokens": int(regression),
        "both_correct_tokens": int(both_correct),
        "both_wrong_tokens": int(both_wrong),
        "equal_tokens": int(equal),
    }


def _method_scopes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_length = {
        str(length): _record_summary([row for row in rows if row["length"] == length])
        for length in EXPECTED_LENGTHS
    }
    by_position_bin: dict[str, Any] = {}
    for label, start, end in POSITION_BINS:
        by_position_bin[label] = _record_summary(rows, selected_positions=range(start, end + 1))
    by_position = {
        str(position): _position_summary(rows, position)
        for position in range(1, max(row["length"] for row in rows) + 1)
    }
    return {
        "overall": _record_summary(rows),
        "by_length": by_length,
        "by_position_bin": by_position_bin,
        "by_position": by_position,
    }


def _pair_scopes(
    projected: Mapping[str, Mapping[str, Any]],
    a1: Mapping[str, Mapping[str, Any]],
    record_ids: list[str],
) -> dict[str, Any]:
    by_length = {
        str(length): _pair_summary(
            projected,
            a1,
            [record_id for record_id in record_ids if projected[record_id]["length"] == length],
        )
        for length in EXPECTED_LENGTHS
    }
    by_position_bin = {
        label: _pair_summary(projected, a1, record_ids, selected_positions=range(start, end + 1))
        for label, start, end in POSITION_BINS
    }
    max_length = max(projected[record_id]["length"] for record_id in record_ids)
    by_position = {
        str(position): _pair_summary(projected, a1, record_ids, selected_positions=(position,))
        for position in range(1, max_length + 1)
    }
    return {
        "overall": _pair_summary(projected, a1, record_ids),
        "by_length": by_length,
        "by_position_bin": by_position_bin,
        "by_position": by_position,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "target_bundle",
        "scope_type",
        "scope",
        "method",
        "records",
        "scored_tokens",
        "correct_tokens",
        "token_accuracy",
        "exact_records",
        "exact_record_rate",
        "tie_positions",
        "projected_correct_tokens",
        "a1_correct_tokens",
        "token_delta",
        "projected_accuracy",
        "a1_accuracy",
        "gain_tokens",
        "regression_tokens",
        "both_correct_tokens",
        "both_wrong_tokens",
        "equal_tokens",
    ]
    if path.exists() or path.is_symlink():
        raise SupplementError(f"supplement CSV already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
        handle.flush()
        os.fsync(handle.fileno())


def _flatten_csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle, value in summary["bundles"].items():
        methods = value["methods"]
        for method, scopes in methods.items():
            for scope_type, scope_values in (
                ("overall", {"overall": scopes["overall"]}),
                ("length", scopes["by_length"]),
                ("position_bin", scopes["by_position_bin"]),
                ("position", scopes["by_position"]),
                ("style", scopes["by_style"]),
            ):
                for scope, metrics in scope_values.items():
                    rows.append({"target_bundle": bundle, "scope_type": scope_type, "scope": scope, "method": method, **metrics})
        for scope_type, scope_values in (
            ("paired_overall", {"overall": value["projected_vs_a1"]["overall"]}),
            ("paired_length", value["projected_vs_a1"]["by_length"]),
            ("paired_position_bin", value["projected_vs_a1"]["by_position_bin"]),
            ("paired_position", value["projected_vs_a1"]["by_position"]),
        ):
            for scope, metrics in scope_values.items():
                rows.append({"target_bundle": bundle, "scope_type": scope_type, "scope": scope, "method": "projected_vs_a1", **metrics})
    return rows


def _plot(summary: Mapping[str, Any], output: Path) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"status": "FAILED", "error": f"matplotlib unavailable: {exc.__class__.__name__}: {exc}"}

    labels = ["raw boundary", "projected", "native A1"]
    methods = list(BASE_METHODS)
    colors = ["#8c8c8c", "#4477aa", "#228833"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    try:
        for axis, bundle, title in zip(axes, BUNDLES, ("Matched public", "Shifted full SFT"), strict=True):
            values = [100.0 * float(summary["bundles"][bundle]["methods"][method]["overall"]["token_accuracy"]) for method in methods]
            bars = axis.bar(range(3), values, color=colors, width=0.65)
            anchor = summary["bundles"][bundle]["methods"][A2_METHOD]["overall"]
            anchor_value = 100.0 * float(anchor["token_accuracy"])
            anchor_bar = axis.bar([3], [anchor_value], color="#aa3377", width=0.65, hatch="//")
            axis.set_xticks(range(4), labels + ["A1+A2\nanchor"])
            axis.set_ylim(0.0, 108.0)
            axis.set_yticks((0, 20, 40, 60, 80, 100))
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            for bar, value in zip((*bars, *anchor_bar), (*values, anchor_value), strict=True):
                axis.text(bar.get_x() + bar.get_width() / 2.0, min(value + 2.0, 98.0), f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
            axis.text(3, 8, "4 records\n156 tokens", ha="center", va="center", fontsize=7.5, color="#552244")
        axes[0].set_ylabel("Token accuracy (%)")
        figure.suptitle("TRR-P03 Stage 1 reconstruction accuracy", fontsize=12, y=0.99)
        figure.text(0.5, 0.945, "Full panel: 24 records, 1,482 tokens per target; A1+A2 anchor: 4 records, 156 tokens", ha="center", va="center", fontsize=8.5)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        figure.savefig(output, dpi=220, bbox_inches="tight")
    except Exception as exc:  # pragma: no cover - environment-specific
        plt.close(figure)
        return {"status": "FAILED", "error": f"plot failed: {exc.__class__.__name__}: {exc}"}
    plt.close(figure)
    return {"status": "PASS", "path": str(output), "bytes": output.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--paired-prediction-root", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    started_utc = _utc_now()
    score_root = args.score_root.resolve()
    panel_path = _regular_file(args.panel, "Stage-1 evaluator panel")
    panel = _read_panel(panel_path)
    truth = _read_stage1_truth(args.truth)
    score_rows = _read_score_rows(score_root / "per_record.jsonl", panel)
    prediction_data: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    prediction_inputs: dict[str, Any] = {}
    for bundle, root in zip(BUNDLES, (args.prediction_root, args.paired_prediction_root), strict=True):
        prediction_data[bundle], prediction_inputs[bundle] = _read_prediction_rows(root, expected_bundle=bundle)

    for bundle in BUNDLES:
        for method in METHODS:
            expected_ids = ANCHOR_IDS if method == A2_METHOD else EXPECTED_IDS
            for record_id in expected_ids:
                score = score_rows[(bundle, method, record_id)]
                prediction = prediction_data[bundle][(method, record_id)]
                _require(len(prediction["tokens"]) == score["length"] + 1, f"prediction/score length differs for {bundle}/{method}/{record_id}")
                _require(prediction["tokens"][0] == BOS_TOKEN_ID, f"prediction BOS changed for {bundle}/{method}/{record_id}")
                _require(prediction["tie_counts"] == score["tie_counts"], f"prediction/score tie diagnostics differ for {bundle}/{method}/{record_id}")
                _require(len(prediction["tokens"]) == len(truth[record_id]), f"prediction/truth length differs for {record_id}")

    tie_by_method: dict[str, dict[str, int]] = {}
    for bundle in BUNDLES:
        tie_by_method[bundle] = {
            method: int(sum(sum(1 for value in score_rows[(bundle, method, record_id)]["tie_counts"] if value > 1) for record_id in (ANCHOR_IDS if method == A2_METHOD else EXPECTED_IDS)))
            for method in METHODS
        }
    _require(all(value == 0 for bundle in tie_by_method.values() for value in bundle.values()), "nonzero top-1 ties found")

    summary: dict[str, Any] = {
        "schema": SUPPLEMENT_SCHEMA,
        "task_id": TASK_ID,
        "status": "DERIVED_FROM_FROZEN_STAGE1_SCORE",
        "truth_opened": True,
        "stage2_opened": False,
        "created_utc": started_utc,
        "position_bins": [{"label": label, "start": start, "end": end} for label, start, end in POSITION_BINS],
        "panel": {
            "records": len(panel),
            "record_order": list(EXPECTED_IDS),
            "styles": {style: sum(1 for value in panel.values() if value["style"] == style) for style in sorted({value["style"] for value in panel.values()})},
            "join": "evaluator_panel_metadata_only",
        },
        "distinct_scored_token_id_count": {
            "truth_scored_positions": int(sum(len(values) - 1 for values in truth.values())),
            "truth_scored_token_ids": int(len({value for values in truth.values() for value in values[1:]})),
            "prediction_scored_token_ids": {},
        },
        "tie_counts": {"all_zero": True, "positions_by_bundle_method": tie_by_method},
        "bundles": {},
    }

    for bundle in BUNDLES:
        method_rows: dict[str, list[dict[str, Any]]] = {}
        method_summaries: dict[str, Any] = {}
        for method in METHODS:
            expected_ids = ANCHOR_IDS if method == A2_METHOD else list(EXPECTED_IDS)
            rows = [score_rows[(bundle, method, record_id)] for record_id in expected_ids]
            method_rows[method] = rows
            scopes = _method_scopes(rows)
            by_style = {
                style: _record_summary([row for row in rows if row["style"] == style])
                for style in sorted({row["style"] for row in rows})
            }
            scopes["by_style"] = by_style
            prediction_ids = {
                value
                for record_id in expected_ids
                for value in prediction_data[bundle][(method, record_id)]["tokens"][1:]
            }
            scopes["score_units"] = sorted({prediction_data[bundle][(method, record_id)]["score_units"] for record_id in expected_ids})
            scopes["distinct_scored_token_ids"] = int(len(prediction_ids))
            method_summaries[method] = scopes
            summary["distinct_scored_token_id_count"]["prediction_scored_token_ids"].setdefault(bundle, {})[method] = int(len(prediction_ids))
        projected = {row["record_id"]: row for row in method_rows["projected_boundary.cosine"]}
        a1 = {row["record_id"]: row for row in method_rows["historical_a1.cosine"]}
        summary["bundles"][bundle] = {
            "methods": method_summaries,
            "projected_vs_a1": _pair_scopes(projected, a1, list(EXPECTED_IDS)),
        }

    output_root = create_only_directory(args.output_root.resolve())
    summary_path = output_root / "stratified_summary.json"
    csv_path = output_root / "stratified_summary.csv"
    write_json_exclusive(summary_path, summary)
    _write_csv(csv_path, _flatten_csv_rows(summary))

    figure_path = output_root / "accuracy_by_bundle.png"
    plot = _plot(summary, figure_path)
    ended_utc = _utc_now()
    evidence_path = output_root / "supplement_evidence.json"
    evidence = {
        "schema": "token-reconstruction.trr-p03-stage1-score-supplement-evidence.v1",
        "task_id": TASK_ID,
        "status": "REPORT_SUPPLEMENT_COMPLETE" if plot["status"] == "PASS" else "REPORT_SUMMARY_COMPLETE_PLOT_FAILED",
        "truth_opened": True,
        "stage2_opened": False,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "command": {"argv": [str(value) for value in sys.argv], "cwd": os.getcwd()},
        "implementation_commit": args.implementation_commit,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "inputs": {
            "score_root": str(score_root),
            "per_record": file_record(score_root / "per_record.jsonl"),
            "panel": file_record(panel_path),
            "truth": {"stage": "stage1", "records": len(truth)},
            "prediction_bundles": prediction_inputs,
        },
        "phases": {
            "input_and_summary_seconds": float(time.perf_counter() - started),
            "plot_status": plot,
        },
        "outputs": {
            "stratified_summary": file_record(summary_path),
            "stratified_csv": file_record(csv_path),
            "figure": plot,
        },
    }
    write_json_exclusive(evidence_path, evidence)
    evidence_path.chmod(0o444)
    summary_path.chmod(0o444)
    csv_path.chmod(0o444)
    if plot["status"] == "PASS":
        figure_path.chmod(0o444)
    print(json.dumps({"status": evidence["status"], "output_root": str(output_root), "summary": str(summary_path), "csv": str(csv_path), "figure": plot}, sort_keys=True))
    return 0 if plot["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SupplementError, P03IOError) as exc:
        print(f"TRR-P03 score supplement failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
