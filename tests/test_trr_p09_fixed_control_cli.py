"""Model-free end-to-end smoke tests for the P09 fixed-control CLI."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.trr_p09.fixed_control_cli import (
    FixedControlCLIError,
    _ValidationCropLoader,
    main,
)
from scripts.trr_p09.prepare_streamed_bank import StreamBatch


def test_synthetic_cli_serializes_schedule_states_and_domain_selection(tmp_path: Path) -> None:
    output_root = tmp_path / "synthetic"
    assert main(["--synthetic", "--output-root", str(output_root), "--steps", "2"]) == 0

    receipt = json.loads((output_root / "receipt.json").read_text(encoding="utf-8"))
    schedule = json.loads((output_root / "schedule_receipt.json").read_text(encoding="utf-8"))
    state_paths = sorted((output_root / "states").glob("checkpoint_step_*.safetensors"))

    assert receipt["status"] == "COMPLETED"
    assert receipt["caller_schema"] == "token-reconstruction.trr-p09-fixed-control-caller.v1"
    assert receipt["selection"] == {
        "metric": "domain_balanced_token_accuracy",
        "rule": "earliest strict maximum; step zero eligible",
        "selected_step": 0,
    }
    assert receipt["cost"]["updates"] == 2
    assert receipt["cost"]["total_position_draws"] == 6
    assert [point["step"] for point in receipt["learning_curve"]] == [0, 1, 2]
    assert schedule["status"] == "DERIVED_BEFORE_MODEL_LOAD"
    assert schedule["semantic_sha256"] == receipt["exposure"]["schedule_semantic_sha256"]
    assert len(state_paths) == 3
    assert all(path.stat().st_size > 0 for path in state_paths)


def test_h128_validation_crop_preserves_rows_masks_and_positions() -> None:
    class FullWidthLoader:
        geometry = SimpleNamespace(sequence_tokens=192)

        def get_records(self, global_indices):
            rows = tuple(int(row) for row in global_indices)
            activations = torch.zeros(len(rows), 192, 3, dtype=torch.bfloat16)
            token_ids = torch.arange(192, dtype=torch.long).expand(len(rows), -1).clone()
            attention_mask = torch.ones(len(rows), 192, dtype=torch.bool)
            attention_mask[0, 100:] = False
            position_ids = torch.where(
                attention_mask,
                torch.arange(192, dtype=torch.long).expand(len(rows), -1),
                torch.zeros(len(rows), 192, dtype=torch.long),
            )
            return StreamBatch(
                activations=activations,
                token_ids=token_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                global_rows=rows,
                record_ids=tuple(f"r-{row}" for row in rows),
                sequence_ids=tuple(f"s-{row}" for row in rows),
            )

    batch = _ValidationCropLoader(FullWidthLoader(), sequence_tokens=128).get_records((7, 2))
    assert tuple(batch.activations.shape) == (2, 128, 3)
    assert batch.global_rows == (7, 2)
    assert batch.record_ids == ("r-7", "r-2")
    assert int(batch.attention_mask[0].sum()) == 100
    assert int(batch.position_ids[0, 100]) == 0
    assert int(batch.position_ids[1, 127]) == 127


def test_synthetic_output_is_create_only(tmp_path: Path) -> None:
    output_root = tmp_path / "synthetic"
    assert main(["--synthetic", "--output-root", str(output_root), "--steps", "1"]) == 0
    with pytest.raises(FixedControlCLIError, match="create-only"):
        main(["--synthetic", "--output-root", str(output_root), "--steps", "1"])


def test_production_fails_closed_before_model_load_without_signed_gates(tmp_path: Path) -> None:
    with pytest.raises(FixedControlCLIError, match="lacks required arguments"):
        main(["--output-root", str(tmp_path / "production")])
