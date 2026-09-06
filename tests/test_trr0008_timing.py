from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import statistics

import pytest
from scipy.stats import t as scipy_t
import torch

from scripts import trr0008_timing as timing
from token_reconstruction.trr0007_positionwise import build_current_positionwise


def _fake_blocks(*, alias_ratio: float = 1.0, candidate_ratio: float = 1.1):
    blocks = []
    for block_index in range(10):
        entries = []
        for cell_id in timing.contract.CELL_ORDER:
            for order_index, method_id in enumerate(timing.METHOD_IDS):
                if method_id == timing.REFERENCE_METHOD_ID:
                    seconds = 1.0
                elif method_id == "current_enriched__trained_diagonal":
                    seconds = alias_ratio
                elif method_id == timing.CANDIDATE_METHOD_ID:
                    seconds = candidate_ratio
                else:
                    seconds = 1.0
                entries.append(
                    {
                        "method_id": method_id,
                        "cell_id": cell_id,
                        "order_index": order_index,
                        "measured_seconds_sum": seconds,
                        "per_record_measured_seconds": [seconds],
                    }
                )
        blocks.append({"block_index": block_index, "entries": entries})
    return blocks


def test_schedule_is_exactly_balanced_per_cell() -> None:
    orders, digest = timing._schedule_rows(
        method_ids=timing.METHOD_IDS,
        cell_ids=timing.contract.CELL_ORDER,
        blocks=timing.DEFAULT_BLOCKS,
        seed=timing.DEFAULT_SEED,
    )
    assert len(orders) == 10 * len(timing.contract.CELL_ORDER)
    assert len(digest) == 64
    for cell_id in timing.contract.CELL_ORDER:
        rows = [row for row in orders if row["cell_id"] == cell_id]
        assert len(rows) == 10
        for position in range(len(timing.METHOD_IDS)):
            counts = Counter(row["order"][position] for row in rows)
            assert counts == Counter({method_id: 2 for method_id in timing.METHOD_IDS})
        first = rows[0]["order"]
        assert rows[5]["order"] == list(reversed(first))


def test_followup_schedule_repeats_exact_ten_block_cycle() -> None:
    orders, digest = timing._schedule_rows(
        method_ids=timing.METHOD_IDS,
        cell_ids=timing.contract.CELL_ORDER,
        blocks=timing.FOLLOWUP_BLOCKS,
        seed=timing.DEFAULT_SEED,
    )
    assert len(orders) == timing.FOLLOWUP_BLOCKS * len(timing.contract.CELL_ORDER)
    assert len(digest) == 64
    for cell_id in timing.contract.CELL_ORDER:
        rows = [row for row in orders if row["cell_id"] == cell_id]
        assert len(rows) == timing.FOLLOWUP_BLOCKS
        cycle = [tuple(row["order"]) for row in rows[: timing.DEFAULT_BLOCKS]]
        for cycle_index in range(4):
            start = cycle_index * timing.DEFAULT_BLOCKS
            assert [tuple(row["order"]) for row in rows[start : start + timing.DEFAULT_BLOCKS]] == cycle
        for position in range(len(timing.METHOD_IDS)):
            counts = Counter(row["order"][position] for row in rows)
            assert counts == Counter({method_id: 8 for method_id in timing.METHOD_IDS})


def test_run_rejects_unregistered_block_count() -> None:
    config = timing.TimingConfig(
        repository_root=Path("."),
        trr7_root=Path("."),
        output_path=Path("/tmp/trr0008-unused.json"),
        device="cpu",
        blocks=20,
    )
    with pytest.raises(timing.TimingError, match="registered timing supports exactly"):
        timing.run(config)


def test_followup_ratio_ci_uses_exact_student_t_df39() -> None:
    values = [1.0 + 0.01 * (index % 5) for index in range(timing.FOLLOWUP_BLOCKS)]
    result = timing._ratio_summary(values)
    critical = float(scipy_t.ppf(0.975, 39))
    expected_margin = critical * statistics.stdev(values) / math.sqrt(timing.FOLLOWUP_BLOCKS)
    assert result["degrees_of_freedom"] == 39
    assert result["critical_value"] == pytest.approx(critical, abs=1e-12)
    assert result["ci_lower"] == pytest.approx(result["mean_ratio"] - expected_margin, abs=1e-12)
    assert result["ci_upper"] == pytest.approx(result["mean_ratio"] + expected_margin, abs=1e-12)


def test_ratio_summary_has_fixed_threshold_decisions() -> None:
    assert timing._ratio_summary([1.0] * 10)["decision"] == "PASS"
    assert timing._ratio_summary([1.4] * 10)["decision"] == "FAIL"
    assert timing._ratio_summary([1.0] * 9 + [2.0])["decision"] == "INCONCLUSIVE"


def test_candidate_qualification_is_per_cell_and_requires_alias_pass() -> None:
    result = timing._summarize_blocks(_fake_blocks())
    assert result["qualification"]["decision"] == "PASS"
    assert result["qualification"]["raw_per_cell_decision"] == "PASS"
    assert result["alias_control"]["decision"] == "PASS"
    assert set(result["qualification"]["per_cell"]) == set(timing.contract.CELL_ORDER)
    assert result["comparisons_vs_reference"][timing.CANDIDATE_METHOD_ID]["pooled_used_for_qualification"] is False


def test_alias_persistent_deviation_fails_candidate_cost_claim() -> None:
    result = timing._summarize_blocks(_fake_blocks(alias_ratio=1.06))
    assert result["alias_control"]["decision"] == "FAIL"
    assert result["alias_control"]["invalid_persistent_deviation_cells"]
    assert result["qualification"]["decision"] == "INVALID_ALIAS_CONTROL"
    assert result["qualification"]["measurement_valid"] is False
    assert result["qualification"]["cost_failure_demonstrated"] is False


def test_candidate_budget_failure_requires_valid_alias_control() -> None:
    result = timing._summarize_blocks(_fake_blocks(candidate_ratio=1.4))
    assert result["alias_control"]["decision"] == "PASS"
    assert result["qualification"]["decision"] == "FAIL"
    assert result["qualification"]["measurement_valid"] is True
    assert result["qualification"]["cost_failure_demonstrated"] is True


def test_alias_wide_ci_is_inconclusive_and_blocks_pass() -> None:
    blocks = _fake_blocks()
    # Vary the alias in a way that leaves its CI outside the fully-contained
    # band criterion while keeping the candidate/reference cells constant.
    for block in blocks:
        value = 0.9 if int(block["block_index"]) % 2 else 1.1
        for entry in block["entries"]:
            if entry["method_id"] == "current_enriched__trained_diagonal":
                entry["measured_seconds_sum"] = value
                entry["per_record_measured_seconds"] = [value]
    result = timing._summarize_blocks(blocks)
    assert result["alias_control"]["decision"] == "INCONCLUSIVE"
    assert result["qualification"]["decision"] == "INCONCLUSIVE"


def test_alias_inconclusive_blocks_candidate_failure_claim() -> None:
    blocks = _fake_blocks(candidate_ratio=1.4)
    for block in blocks:
        value = 0.9 if int(block["block_index"]) % 2 else 1.1
        for entry in block["entries"]:
            if entry["method_id"] == "current_enriched__trained_diagonal":
                entry["measured_seconds_sum"] = value
                entry["per_record_measured_seconds"] = [value]
    result = timing._summarize_blocks(blocks)
    assert result["alias_control"]["decision"] == "INCONCLUSIVE"
    assert result["qualification"]["raw_per_cell_decision"] == "FAIL"
    assert result["qualification"]["decision"] == "INCONCLUSIVE"
    assert result["qualification"]["measurement_valid"] is False
    assert result["qualification"]["cost_failure_demonstrated"] is False


def test_alias_execution_identity_checks_class_and_tensors() -> None:
    reference = build_current_positionwise(hidden_size=4, vocabulary_size=7, context_width=2)
    alias = build_current_positionwise(hidden_size=4, vocabulary_size=7, context_width=2)
    alias.load_state_dict(reference.state_dict(), strict=True)
    evidence = {
        timing.REFERENCE_METHOD_ID: {"state": {"sha256": "a" * 64}},
        "current_enriched__trained_diagonal": {"state": {"sha256": "a" * 64}},
    }
    receipt = timing._verify_alias_execution_identity(
        {timing.REFERENCE_METHOD_ID: reference, "current_enriched__trained_diagonal": alias},
        evidence,
    )
    assert receipt["class_exact"] is True
    assert receipt["state_tensors_exact"] is True


def test_alias_execution_identity_rejects_weight_change() -> None:
    reference = build_current_positionwise(hidden_size=4, vocabulary_size=7, context_width=2)
    alias = build_current_positionwise(hidden_size=4, vocabulary_size=7, context_width=2)
    with torch.no_grad():
        alias.W[0, 0] += 1.0
    evidence = {
        timing.REFERENCE_METHOD_ID: {"state": {"sha256": "a" * 64}},
        "current_enriched__trained_diagonal": {"state": {"sha256": "b" * 64}},
    }
    with pytest.raises(timing.TimingError, match="identical-weight"):
        timing._verify_alias_execution_identity(
            {timing.REFERENCE_METHOD_ID: reference, "current_enriched__trained_diagonal": alias},
            evidence,
        )
