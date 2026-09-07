"""Focused tests for the truth-free TRR-0010 fit table assembler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import trr0010_fit_diagnostics_table as table


FIXED = table.DEFAULT_FIXED_TABLE
STATE_BINDING = table.DEFAULT_FIXED_STATE_BINDING


def _directional_receipt(path: Path, *, arm: str = "current_directional", truth_opened: bool = False) -> Path:
    digest = "a" * 64
    points = [
        {
            "step": step,
            "validation": {"domain_balanced_token_accuracy": 0.8 + step / 100000.0},
            "train": None,
            "validation_seconds": 0.1,
            "state_sha256": digest,
        }
        for step in table.CHECKPOINT_GRID
    ]
    diagnostics = []
    for step in table.CHECKPOINT_GRID:
        diagnostics.append(
            {
                "step": step,
                "fit_records": 64,
                "fit_records_sha256": digest,
                "full_bank": None if step not in (0, 13000) else {"endpoint": "start" if step == 0 else "end", "row_count": 1200},
                "selection_metric_untouched": True,
                "elapsed_seconds": 0.01,
            }
        )
    exposure = {
        "seed": 4010,
        "steps": 13000,
        "draws_per_step": 512,
        "total_draws": 6_656_000,
        "used_replacement_steps": 4,
        "unique_record_position_pairs": 123,
        "repeated_record_position_pairs": 456,
        "max_exposures_per_record_position": 19,
        "schedule_semantic_sha256": digest,
    }
    runner = {
        "schema": "token-reconstruction.trr-p09-fixed-control-runner.v1",
        "status": "COMPLETED",
        "schedule": {"seed": 4010, "steps": 13000, "semantic_sha256": digest, "exposure": exposure},
        "checkpoints": list(table.CHECKPOINT_GRID),
        "selected_step": 8000,
        "selected_state_sha256": digest,
        "checkpoint_state_bindings": [],
        "learning_curve": points,
        "timing": {
            "whole_wall_seconds": 10.0,
            "stream_load_seconds": 2.0,
            "optimizer_update_seconds": 7.0,
            "validation_seconds": 1.0,
        },
    }
    arm_record = {
        "arm_name": arm,
        "bank_role": "B0",
        "status": "COMPLETED",
        "truth_opened": truth_opened,
        "support_token_ids": 17,
        "diagnostic_events": diagnostics,
        "runner_result": runner,
        "timing": {
            "provider_preparation_seconds": 1.0,
            "runner_call_seconds": 10.0,
            "restore_export_seconds": 0.2,
            "base_decoder_export_seconds": 0.3,
            "wrapper_seconds": 10.5,
        },
        "selected_checkpoint": {"path": "checkpoint.safetensors", "bytes": 12, "sha256": digest},
        "base_decoder_state": {"path": "base.safetensors", "bytes": 13, "sha256": digest},
        "effective_readout": {"path": "w.safetensors", "bytes": 14, "sha256": digest},
    }
    payload = {
        "schema": "token-reconstruction.trr0010-directional-fit.v1",
        "task_id": table.TASK_ID,
        "status": "FIT_COMPLETE",
        "arms": {arm: arm_record},
        "truth_opened": truth_opened,
        "elapsed_seconds": 12.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pending_output_binds_fixed_v2_rows_and_leaves_directionals_pending() -> None:
    result = table.assemble_table(fixed_table_path=FIXED, fixed_state_binding_path=STATE_BINDING)

    assert result["status"] == "FIXED_ARMS_BOUND_DIRECTIONAL_ARMS_PENDING"
    assert result["truth_opened"] is False
    assert result["source_bindings"]["fixed_fit_table_v2"]["sha256"] == table.EXPECTED_FIXED_TABLE_SHA256
    assert [row["step"] for row in result["arms"]["current_fixed"]["checkpoint_curve"]] == list(table.CHECKPOINT_GRID)
    assert result["arms"]["current_fixed"]["selected_step"] == 8000
    assert result["arms"]["expanded_fixed"]["exposure"]["actual_sampled_unique_record_position_pairs"] == 1_237_403
    assert result["arms"]["current_directional"]["status"] == "PENDING_MISSING_RECEIPT"
    assert result["arms"]["expanded_directional"]["status"] == "PENDING_MISSING_RECEIPT"


def test_directional_wrapper_is_normalized_without_truth(tmp_path: Path) -> None:
    receipt = _directional_receipt(tmp_path / "run_receipt.json")
    result = table.assemble_table(
        fixed_table_path=FIXED,
        current_directional_path=receipt,
        fixed_state_binding_path=STATE_BINDING,
    )

    arm = result["arms"]["current_directional"]
    assert arm["status"] == "COMPLETE_DIRECTIONAL_RECEIPT"
    assert arm["truth_opened"] is False
    assert [point["step"] for point in arm["checkpoint_curve"]] == list(table.CHECKPOINT_GRID)
    assert len(arm["diagnostics"]) == 7
    assert arm["diagnostics"][0]["full_bank_endpoint"]["endpoint"] == "start"
    assert arm["diagnostics"][-1]["full_bank_endpoint"]["endpoint"] == "end"
    assert arm["exposure"]["support_token_ids"] == 17
    assert arm["cost"]["nonoverlap_wall_cost"]["sum_explicit_components_seconds"] == pytest.approx(11.5)
    assert arm["selected_state"]["deployed_base_plus_effective_readout_bytes"] == 27
    assert result["arms"]["expanded_directional"]["status"] == "PENDING_MISSING_RECEIPT"


def test_directional_truth_receipt_is_rejected(tmp_path: Path) -> None:
    receipt = _directional_receipt(tmp_path / "truth_receipt.json", truth_opened=True)
    with pytest.raises(table.FitTableError, match="opened truth"):
        table.assemble_table(fixed_table_path=FIXED, current_directional_path=receipt)


def test_raw_runner_result_is_accepted_without_inventing_postfit_metadata(tmp_path: Path) -> None:
    wrapper = _directional_receipt(tmp_path / "wrapper.json")
    payload = json.loads(wrapper.read_text(encoding="utf-8"))
    raw = payload["arms"]["current_directional"]["runner_result"]
    raw_path = tmp_path / "raw_runner_result.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = table.assemble_table(fixed_table_path=FIXED, current_directional_path=raw_path)
    arm = result["arms"]["current_directional"]
    assert arm["status"] == "TRAINING_COMPLETE_DEPLOYMENT_EXPORT_PENDING"
    assert arm["fit_started"] is True
    assert arm["checkpoint_curve"][-1]["step"] == 13000
    assert arm["diagnostics"] == []
    assert arm["selected_step"] == 8000
    assert arm["selected_state"] is None
    assert arm["exposure"]["support_token_ids"] is None
    assert "full_and_frozen64_diagnostics" in arm["source"]["missing_fields"]
    assert "selected_state_footprints" in arm["source"]["missing_fields"]
    assert result["status"] == "TRAINING_COMPLETE_DEPLOYMENT_EXPORT_PENDING"
