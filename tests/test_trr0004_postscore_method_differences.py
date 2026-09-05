from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import trr0004_postscore_method_differences as analysis


METHODS = (
    analysis.BASELINE_METHOD,
    "causal_h_attention128",
    "positionwise_mlp256",
    "historical_alpaca_a1",
    "frozen_a1_a2_k256",
)


def _score_fixture() -> dict:
    cells = {}
    for style in analysis.EXPECTED_STYLES:
        for condition in analysis.EXPECTED_CONDITIONS:
            for method_id in METHODS:
                rows = []
                for index, record_id in enumerate((f"{style}-r0", f"{style}-r1", f"{style}-r2")):
                    base = (0.5, 0.25, 1.0)[index]
                    delta = {
                        analysis.BASELINE_METHOD: 0.0,
                        "causal_h_attention128": (0.25, 0.0, -0.25)[index],
                        "positionwise_mlp256": (0.0, 0.0, 0.0)[index],
                        "historical_alpaca_a1": (-0.25, 0.0, 0.0)[index],
                        "frozen_a1_a2_k256": (0.0, 0.0, 0.0)[index],
                    }[method_id]
                    accuracy = base + delta
                    rows.append(
                        {
                            "record_id": record_id,
                            "scored_tokens": 4,
                            "correct_tokens": int(round(accuracy * 4)),
                            "token_accuracy": accuracy,
                            "exact_record": accuracy == 1.0,
                        }
                    )
                cell_id = f"{style}__{condition}__{method_id}"
                cells[cell_id] = {
                    "cell_id": cell_id,
                    "style": style,
                    "condition": condition,
                    "method_id": method_id,
                    "per_record": rows,
                }
    return {
        "status": "FRESH_CONFIRMATION_SCORED_AFTER_COMPLETE_PUBLIC_GATE",
        "truth_gate": {"truth_opened_after_gate": True},
        "cells": cells,
    }


def test_method_differences_are_paired_and_deterministic() -> None:
    first = analysis.analyze(_score_fixture())
    second = analysis.analyze(_score_fixture())
    key = "pile__public_base__causal_h_attention128__vs__historical_affine_ce_no_vocab_bias"
    assert len(first) == 12
    assert first == second
    assert first[key]["total_correct_tokens_delta"] == 0
    assert first[key]["bootstrap"]["seed"] == 4004
    assert first[key]["bootstrap"]["draws"] == 2000
    assert first[key]["bootstrap"]["delta_estimate"] == pytest.approx(0.0)


def test_cli_binds_immutable_score_result(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    output_path = tmp_path / "postscore.json"
    score_path.write_text(json.dumps(_score_fixture()) + "\n")
    assert analysis.main(["--score-result", str(score_path), "--output", str(output_path)]) == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "POSTSCORE_METHOD_COMPARISONS_COMPLETE"
    assert result["input_score_result"]["before"] == result["input_score_result"]["after"]
    assert result["source"]["sha256_start"] == result["source"]["sha256_end"]
    assert len(result["comparisons"]) == 12
