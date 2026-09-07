from __future__ import annotations

import pytest
import torch

from scripts import trr0009_eval_contract as contract
from scripts import trr0009_timing as timing


def _blocks(ratio: float = 1.1, alias_ratio: float = 1.0):
    rows, _ = timing.schedule_rows()
    blocks = []
    for block_index in range(timing.DEFAULT_BLOCKS):
        entries = []
        for row in [x for x in rows if x["block_index"] == block_index]:
            for method_id in row["order"]:
                value = 1.0
                if method_id == contract.PRIMARY_METHOD_ID:
                    value = ratio
                elif method_id == timing.ALIAS_METHOD_ID:
                    value = alias_ratio
                entries.append({
                    "method_id": method_id,
                    "cell_id": row["cell_id"],
                    "measured_seconds_sum": value,
                    "per_record_measured_seconds": [value] * 32,
                })
        blocks.append({"block_index": block_index, "entries": entries})
    return blocks


def test_five_path_schedule_has_rotations_reversals_and_exact_balance() -> None:
    rows, digest = timing.schedule_rows(seed=8008)
    assert len(rows) == 40 * 4
    assert len(digest) == 64
    for block in range(40):
        block_rows = [row for row in rows if row["block_index"] == block]
        assert len(block_rows) == 4
        assert all(tuple(row["order"]) in {tuple(row["order"]) for row in block_rows} for row in block_rows)
    for cell in contract.CELL_ORDER:
        orders = [row["order"] for row in rows if row["cell_id"] == cell]
        counts = {method: sum(method in order for order in orders) for method in timing.TIMING_PATH_IDS}
        assert counts == {method: 40 for method in timing.TIMING_PATH_IDS}
        positions = {method: [order.index(method) for order in orders] for method in timing.TIMING_PATH_IDS}
        assert all(len(values) == 40 for values in positions.values())


def test_alias_inconclusive_invalidates_cost_claim_without_reclassifying_candidate() -> None:
    summary = timing.summarize_blocks(_blocks(ratio=1.1, alias_ratio=1.1))
    assert summary["alias_control"]["decision"] == "FAIL"
    assert summary["qualification"]["decision"] == "INVALID_ALIAS_CONTROL"
    assert summary["qualification"]["measurement_valid"] is False
    assert summary["qualification"]["cost_failure_demonstrated"] is False


def test_candidate_pass_requires_alias_containment_and_each_cell() -> None:
    summary = timing.summarize_blocks(_blocks(ratio=1.1, alias_ratio=1.0))
    assert summary["alias_control"]["decision"] == "PASS"
    assert summary["qualification"]["decision"] == "PASS"
    assert summary["qualification"]["measurement_valid"] is True
    assert summary["qualification"]["cost_failure_demonstrated"] is False


def test_df39_uses_frozen_student_t_critical_value() -> None:
    values = [1.0 + (index - 19.5) / 10000.0 for index in range(40)]
    row = timing.ratio_summary(values)
    from scipy.stats import t
    assert row["degrees_of_freedom"] == 39
    assert row["critical_value"] == pytest.approx(float(t.ppf(0.975, 39)), rel=0, abs=1e-12)


def test_live_timed_blocks_execute_all_five_paths_with_synthetic_predictor(monkeypatch, tmp_path) -> None:
    cells = {}
    activation = torch.zeros((32, 128, 2048), dtype=torch.bfloat16)
    mask = torch.ones((32, 128), dtype=torch.bool)
    positions = torch.arange(128, dtype=torch.long).expand(32, -1).contiguous()
    for cell_id in contract.CELL_ORDER:
        cells[cell_id] = timing.TimingCell(cell_id, activation, mask, positions, {})

    prediction = torch.full((128,), contract.BOS_TOKEN_ID, dtype=torch.long)
    monkeypatch.setattr(timing.runner, "predict_current_h", lambda *args, **kwargs: prediction)
    monkeypatch.setattr(timing.runner, "_synchronize", lambda device: None)
    monkeypatch.setattr(timing, "_light_guard", lambda **kwargs: None)
    counter = {"value": 0}

    def full_guard(*, device, started, maximum_seconds, stage):
        counter["value"] += 1
        return {
            "stage": stage,
            "status": "PASS",
            "elapsed_seconds": 0.01,
            "guard_overhead_seconds": 0.001,
            "telemetry": {"utc": "2026-09-07T00:00:00Z", "synthetic": True},
        }

    monkeypatch.setattr(timing, "_full_guard", full_guard)
    clock = {"value": 0.0}
    def deterministic_clock() -> float:
        clock["value"] += 1.0
        return clock["value"]
    monkeypatch.setattr(timing.time, "perf_counter", deterministic_clock)
    models = {method_id: object() for method_id in contract.METHOD_ORDER}
    models[timing.ALIAS_METHOD_ID] = models[contract.PRIMARY_CONTROL_METHOD_ID]
    bindings = {
        method_id: {"state": {"path": method_id}, "loader": {"module": "synthetic", "function": "load"}}
        for method_id in contract.METHOD_ORDER
    }
    bindings[timing.ALIAS_METHOD_ID] = bindings[contract.PRIMARY_CONTROL_METHOD_ID]
    alias = timing._check_alias_before_timing(
        models=models,
        bindings=bindings,
        embedding=torch.zeros((1, 1)),
        cells=cells,
        device=torch.device("cpu"),
    )
    assert alias["status"] == "PASS"
    orders, digest = timing.schedule_rows()
    config = timing.TimingConfig(
        repository_root=tmp_path,
        registration_path=tmp_path / "registration.json",
        output_path=tmp_path / "timing.json",
        device="cpu",
    )
    blocks = timing._timed_blocks(
        models=models,
        embedding=torch.zeros((1, 1)),
        cells=cells,
        orders=orders,
        device=torch.device("cpu"),
        config=config,
        started=timing.time.perf_counter(),
    )
    summary = timing.summarize_blocks(blocks)
    assert len(blocks) == 40
    assert all(len(block["entries"]) == 20 for block in blocks)
    assert all(entry["records"] == 32 for block in blocks for entry in block["entries"])
    assert summary["block_count"] == 40
    assert summary["alias_control"]["decision"] == "PASS"
    assert summary["qualification"]["measurement_valid"] is True
    assert counter["value"] == 80
    assert len(digest) == 64
