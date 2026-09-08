"""CPU synthetic tests for the staged TRR-P12 geometry pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.trr_p12.geometry_pipeline import (
    GeometryPipelineError,
    PipelineConfig,
    StageInput,
    run_pipeline,
)


def _write_stage(root: Path, stage: int, projected: torch.Tensor, embedding: torch.Tensor, config: PipelineConfig, *, corrupt_prediction: bool = False) -> tuple[Path, Path, Path]:
    scores = projected @ embedding.T * config.logit_scale
    predictions = torch.empty((projected.shape[0], config.stored_sequence_tokens), dtype=torch.long)
    predictions[:, 0] = config.bos_token_id
    predictions[:, 1:] = torch.argmax(scores, dim=-1)
    if corrupt_prediction:
        predictions[0, 1] = (int(predictions[0, 1]) + 1) % config.vocabulary_size
    pred_path = root / f"stage{stage}.predictions.safetensors"
    projected_path = root / f"stage{stage}.projected.safetensors"
    geometry_path = root / f"stage{stage}.geometry.json"
    save_file({"expanded_fixed": predictions}, str(pred_path))
    save_file({"projected_hidden": projected.float()}, str(projected_path))
    slots = [f"record/{index:03d}" for index in range(projected.shape[0])]
    geometry = {
        "schema": "token-reconstruction.trr-p12-b1-compact-geometry.v1",
        "record_count": projected.shape[0],
        "record_order": slots,
        "prediction_output_geometry": list(predictions.shape),
        "projected_output_geometry": list(projected.shape),
        "rows": [{"slot": slot} for slot in slots],
    }
    geometry_path.write_text(json.dumps(geometry, indent=2) + "\n")
    return pred_path, projected_path, geometry_path


def _stage_inputs(root: Path, embedding: torch.Tensor, config: PipelineConfig, *, corrupt_stage64: bool = False) -> tuple[dict[int, StageInput], Path]:
    embedding_path = root / "embedding.safetensors"
    save_file({"embeddings": embedding}, str(embedding_path))
    stage_features = {
        0: torch.tensor([[[1.0, 0.0, 0.0]] * 3, [[1.0, 0.0, 0.0]] * 3]),
        64: torch.tensor([[[-1.0, 0.0, 0.0]] * 3, [[-1.0, 0.0, 0.0]] * 3]),
        128: torch.tensor([[[-1.0, 0.0, 0.0]] * 3, [[-1.0, 0.0, 0.0]] * 3]),
        256: torch.tensor([[[1.0, 0.0, 0.0]] * 3, [[1.0, 0.0, 0.0]] * 3]),
    }
    inputs: dict[int, StageInput] = {}
    for stage, projected in stage_features.items():
        paths = _write_stage(root, stage, projected, embedding, config, corrupt_prediction=corrupt_stage64 and stage == 64)
        inputs[stage] = StageInput.from_files(stage, *paths, config=config)
    return inputs, embedding_path


def test_pipeline_freezes_forecast_then_reports_signed_actual_pairs(tmp_path: Path) -> None:
    config = PipelineConfig(vocabulary_size=4, hidden_size=3, stored_sequence_tokens=4, bos_token_id=0, logit_scale=2.0)
    embedding = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    stages, embedding_path = _stage_inputs(tmp_path, embedding, config)
    receipt = run_pipeline(
        stages,
        embedding_path,
        forecast_output=tmp_path / "forecast.json",
        private_output=tmp_path / "private.json",
        public_output=tmp_path / "public.json",
        receipt_output=tmp_path / "receipt.json",
        config=config,
    )
    forecast = json.loads((tmp_path / "forecast.json").read_text())
    public = json.loads((tmp_path / "public.json").read_text())
    assert forecast["status"] == "FORECAST_FROZEN_BEFORE_LATER_FEATURE_LOAD"
    assert forecast["later_stages_not_loaded"] == [128, 256]
    assert forecast["aggregate"]["stages"]["128"]["strict_predicted_changes"] == 6
    assert forecast["aggregate"]["stages"]["256"]["strict_predicted_changes"] == 6
    assert public["actual_pairs"]["64"]["strict_crossings"] == 6
    assert public["actual_pairs"]["64"]["strict_geometric_crossings"] == 6
    assert public["actual_pairs"]["128"]["strict_crossings"] == 6
    assert public["actual_pairs"]["128"]["strict_geometric_crossings"] == 6
    assert public["actual_pairs"]["256"]["strict_crossings"] == 0
    assert public["actual_pairs"]["256"]["strict_geometric_crossings"] == 0
    assert receipt["verification"]["full_vocabulary_argmax_checked"] is True
    assert receipt["verification"]["prediction_alignment_stages"] == [0, 64, 128, 256]
    assert receipt["verification"]["raw_logits_retained"] is False


def test_pipeline_rejects_frozen_prediction_argmax_mismatch(tmp_path: Path) -> None:
    config = PipelineConfig(vocabulary_size=4, hidden_size=3, stored_sequence_tokens=4, bos_token_id=0, logit_scale=2.0)
    embedding = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    stages, embedding_path = _stage_inputs(tmp_path, embedding, config, corrupt_stage64=True)
    with pytest.raises(GeometryPipelineError, match="argmax mismatch"):
        run_pipeline(
            stages,
            embedding_path,
            forecast_output=tmp_path / "forecast.json",
            private_output=tmp_path / "private.json",
            public_output=tmp_path / "public.json",
            receipt_output=tmp_path / "receipt.json",
            config=config,
        )


def test_score_backend_cpu_returns_compact_scores_and_metadata() -> None:
    from scripts.trr_p12.geometry_pipeline import ScoreBackend, scores_from_projected

    config = PipelineConfig(vocabulary_size=4, hidden_size=3, stored_sequence_tokens=4, bos_token_id=0, logit_scale=2.0)
    embedding = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    projected = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float32)
    backend = ScoreBackend(embedding, config=config, device="cpu")
    actual = backend.scores(projected)
    expected = scores_from_projected(projected, embedding, config=config)
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)
    assert backend.metadata()["scores_returned_to"] == "cpu"
    assert backend.metadata()["full_vocabulary_argmax_checked_after_device_copy"] is True


def test_score_backend_rejects_unregistered_cuda_device() -> None:
    from scripts.trr_p12.geometry_pipeline import GeometryPipelineError, ScoreBackend

    config = PipelineConfig(vocabulary_size=2, hidden_size=1, stored_sequence_tokens=2, bos_token_id=0)
    with pytest.raises(GeometryPipelineError, match="only cuda:0"):
        ScoreBackend(torch.ones((2, 1)), config=config, device="cuda:1")
