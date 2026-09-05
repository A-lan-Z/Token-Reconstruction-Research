#!/usr/bin/env python3
"""Compute paired method differences from an immutable TRR-0004 score result.

This post-score analysis deliberately runs after the scorer's truth gate.  It
uses only the scorer's per-record JSON output, compares each requested method
to the fresh no-vocabulary-bias affine baseline within each style/condition,
and applies the same fixed paired-record bootstrap as the registered scorer.
It does not refit, retune, load model resources, or open a truth sidecar.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


TASK_ID = "TRR-0004"
SCHEMA = "token-reconstruction.trr0004-postscore-method-differences.v1"
BOOTSTRAP_SEED = 4004
BOOTSTRAP_DRAWS = 2000
BASELINE_METHOD = "historical_affine_ce_no_vocab_bias"
COMPARISON_METHODS = (
    "causal_h_attention128",
    "positionwise_mlp256",
    "historical_alpaca_a1",
)
EXPECTED_STYLES = ("pile", "finance")
EXPECTED_CONDITIONS = ("public_base", "public_lora_2601")


class PostscoreAnalysisError(RuntimeError):
    """Raised when the immutable score result is incomplete or malformed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PostscoreAnalysisError(f"score result is unavailable: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_score(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PostscoreAnalysisError(f"score result is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostscoreAnalysisError(f"score result is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PostscoreAnalysisError("score result must be a JSON object")
    if value.get("status") != "FRESH_CONFIRMATION_SCORED_AFTER_COMPLETE_PUBLIC_GATE":
        raise PostscoreAnalysisError("post-score analysis requires a completed gated score result")
    gate = value.get("truth_gate")
    if not isinstance(gate, Mapping) or gate.get("truth_opened_after_gate") is not True:
        raise PostscoreAnalysisError("score result does not attest to truth opening after the gate")
    cells = value.get("cells")
    if not isinstance(cells, Mapping) or not cells:
        raise PostscoreAnalysisError("score result has no per-cell records")
    return value


def _bootstrap(values: list[float]) -> dict[str, Any]:
    if not values:
        raise PostscoreAnalysisError("cannot bootstrap an empty paired comparison")
    data = np.asarray(values, dtype=np.float64)
    if not np.isfinite(data).all():
        raise PostscoreAnalysisError("paired method differences contain non-finite values")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, data.size, size=(BOOTSTRAP_DRAWS, data.size))
    means = data[indices].mean(axis=1)
    return {
        "seed": BOOTSTRAP_SEED,
        "draws": BOOTSTRAP_DRAWS,
        "unit": "paired record",
        "delta_estimate": float(data.mean()),
        "delta_median": float(np.median(data)),
        "delta_std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _index_records(score: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    cells = score["cells"]
    indexed: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for cell_key, cell in cells.items():
        if not isinstance(cell, Mapping):
            raise PostscoreAnalysisError(f"cell entry is not an object: {cell_key}")
        style = cell.get("style")
        condition = cell.get("condition")
        method_id = cell.get("method_id")
        if not isinstance(style, str) or not isinstance(condition, str) or not isinstance(method_id, str):
            raise PostscoreAnalysisError(f"cell entry lacks style/condition/method_id: {cell_key}")
        records = cell.get("per_record")
        if not isinstance(records, list) or not records:
            raise PostscoreAnalysisError(f"cell lacks per_record metrics: {cell_key}")
        key = (style, condition, method_id)
        if key in indexed:
            raise PostscoreAnalysisError(f"duplicate cell metrics: {key}")
        rows: dict[str, dict[str, Any]] = {}
        for row in records:
            if not isinstance(row, Mapping):
                raise PostscoreAnalysisError(f"non-object per-record metric: {cell_key}")
            record_id = row.get("record_id")
            accuracy = row.get("token_accuracy")
            correct = row.get("correct_tokens")
            scored = row.get("scored_tokens")
            if not isinstance(record_id, str) or not isinstance(accuracy, (int, float)):
                raise PostscoreAnalysisError(f"invalid per-record metric in {cell_key}")
            if record_id in rows:
                raise PostscoreAnalysisError(f"duplicate record ID in {cell_key}: {record_id}")
            if not np.isfinite(float(accuracy)):
                raise PostscoreAnalysisError(f"non-finite token accuracy in {cell_key}")
            rows[record_id] = {
                "record_id": record_id,
                "token_accuracy": float(accuracy),
                "correct_tokens": int(correct) if isinstance(correct, int) else None,
                "scored_tokens": int(scored) if isinstance(scored, int) else None,
                "exact_record": bool(row.get("exact_record")),
            }
        indexed[key] = rows
    return indexed


def analyze(score: Mapping[str, Any]) -> dict[str, Any]:
    indexed = _index_records(score)
    comparisons: dict[str, Any] = {}
    for style in EXPECTED_STYLES:
        for condition in EXPECTED_CONDITIONS:
            base_key = (style, condition, BASELINE_METHOD)
            base = indexed.get(base_key)
            if base is None:
                raise PostscoreAnalysisError(f"missing baseline cell: {base_key}")
            for method_id in COMPARISON_METHODS:
                method_key = (style, condition, method_id)
                method = indexed.get(method_key)
                if method is None:
                    raise PostscoreAnalysisError(f"missing comparison cell: {method_key}")
                if list(base) != list(method):
                    raise PostscoreAnalysisError(f"record pairing changed for {style}/{condition}/{method_id}")
                deltas: list[float] = []
                correct_deltas: list[int] = []
                exact_deltas: list[int] = []
                rows: list[dict[str, Any]] = []
                for record_id, base_row in base.items():
                    method_row = method[record_id]
                    delta = method_row["token_accuracy"] - base_row["token_accuracy"]
                    deltas.append(float(delta))
                    if method_row["correct_tokens"] is not None and base_row["correct_tokens"] is not None:
                        correct_deltas.append(method_row["correct_tokens"] - base_row["correct_tokens"])
                    exact_deltas.append(int(method_row["exact_record"]) - int(base_row["exact_record"]))
                    rows.append({
                        "record_id": record_id,
                        "baseline_token_accuracy": base_row["token_accuracy"],
                        "method_token_accuracy": method_row["token_accuracy"],
                        "token_accuracy_delta": float(delta),
                        "baseline_exact_record": base_row["exact_record"],
                        "method_exact_record": method_row["exact_record"],
                    })
                key = f"{style}__{condition}__{method_id}__vs__{BASELINE_METHOD}"
                comparisons[key] = {
                    "style": style,
                    "condition": condition,
                    "baseline_method": BASELINE_METHOD,
                    "method_id": method_id,
                    "records": len(rows),
                    "mean_correct_tokens_delta": (float(np.mean(correct_deltas)) if correct_deltas else None),
                    "total_correct_tokens_delta": (int(sum(correct_deltas)) if correct_deltas else None),
                    "total_exact_record_delta": int(sum(exact_deltas)),
                    "paired_record_differences": rows,
                    "bootstrap": _bootstrap(deltas),
                }
    return comparisons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc)
    source_path = Path(__file__).resolve()
    input_path = args.score_result.resolve()
    source_start = sha256_file(source_path)
    input_start = file_record(input_path)
    score = load_score(input_path)
    comparisons = analyze(score)
    input_end = file_record(input_path)
    source_end = sha256_file(source_path)
    if input_start != input_end:
        raise PostscoreAnalysisError("score result changed during analysis")
    if source_start != source_end:
        raise PostscoreAnalysisError("analysis source changed during analysis")
    result = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "POSTSCORE_METHOD_COMPARISONS_COMPLETE",
        "claim_scope": "post-score paired method analysis; no refit, retuning, or independent confirmation claim",
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "command": {"argv": [str(value) for value in sys.argv], "cwd": str(Path.cwd())},
        "source": {"path": str(source_path), "bytes": source_path.stat().st_size, "sha256_start": source_start, "sha256_end": source_end},
        "input_score_result": {"before": input_start, "after": input_end},
        "truth_policy": "reads only the already-scored per-record JSON after the registered scorer truth gate; does not load a sidecar",
        "baseline_method": BASELINE_METHOD,
        "comparison_methods": list(COMPARISON_METHODS),
        "styles": list(EXPECTED_STYLES),
        "conditions": list(EXPECTED_CONDITIONS),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "unit": "paired record within each style/condition"},
        "comparisons": comparisons,
    }
    if args.output.exists() or args.output.is_symlink():
        raise PostscoreAnalysisError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "comparisons": len(comparisons)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PostscoreAnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
