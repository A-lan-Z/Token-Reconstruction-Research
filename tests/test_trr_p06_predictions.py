from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.trr_p06 import run_predictions as runner
from token_reconstruction.trr0006_visibility_decoder import build_visibility_decoder


def _tiny_model(method: str):
    direct = {
        "W": torch.eye(4, dtype=torch.float32),
        "b": torch.zeros(4, dtype=torch.float32),
        "s": torch.tensor(0.0, dtype=torch.float32),
    }
    return build_visibility_decoder(
        method,
        hidden_size=4,
        vocabulary_size=7,
        context_width=2,
        qkv_seed=1737,
        direct_state=direct,
    ).eval()


def test_predict_batch_cross_chunk_has_lowest_id_ties_and_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "SEQUENCE_TOKENS", 5)
    monkeypatch.setattr(runner, "HIDDEN_SIZE", 4)
    monkeypatch.setattr(runner, "VOCABULARY_SIZE", 7)
    monkeypatch.setattr(runner, "BOS_TOKEN_ID", 5)
    model = _tiny_model("p06_full_record")
    activations = torch.randn(2, 5, 4, dtype=torch.float32).to(dtype=torch.bfloat16)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    # A zero table creates a seven-way score tie.  The decoder must use the
    # lowest vocabulary ID, and chunking at two rows must not change it.
    embedding = torch.zeros(7, 4, dtype=torch.float32)
    ids_small, ties_small = runner.predict_batch(
        model,
        embedding,
        activations,
        valid,
        device=torch.device("cpu"),
        projection_chunk=2,
    )
    ids_large, ties_large = runner.predict_batch(
        model,
        embedding,
        activations,
        valid,
        device=torch.device("cpu"),
        projection_chunk=64,
    )
    assert torch.equal(ids_small, ids_large)
    assert torch.equal(ties_small, ties_large)
    assert ids_small.tolist() == [[5, 0, 0, 0, 0], [5, 0, 0, -1, -1]]
    assert ties_small.tolist() == [[1, 7, 7, 7, 7], [1, 7, 7, 0, 0]]


def test_observation_loader_validates_sidecars_and_rejects_nonbinary_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "RECORDS_PER_DOMAIN", 2)
    monkeypatch.setattr(runner, "SEQUENCE_TOKENS", 4)
    monkeypatch.setattr(runner, "HIDDEN_SIZE", 3)
    valid_path = tmp_path / "valid.safetensors"
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.uint8)
    positions = torch.arange(4, dtype=torch.int64).repeat(2, 1)
    save_file(
        {
            "activations": torch.randn(2, 4, 3, dtype=torch.bfloat16),
            "attention_mask": mask,
            "position_ids": positions,
        },
        str(valid_path),
    )
    loaded = runner._load_observation_cell({"path": valid_path})
    assert loaded["activations"].shape == (2, 4, 3)
    assert loaded["mask"].dtype == torch.bool
    assert loaded["mask"].tolist() == [[True, True, True, False], [True, True, False, False]]
    assert loaded["position_ids"].tolist() == positions.tolist()

    bad_path = tmp_path / "bad.safetensors"
    bad_mask = mask.clone()
    bad_mask[0, 1] = 2
    save_file(
        {
            "activations": torch.randn(2, 4, 3, dtype=torch.bfloat16),
            "attention_mask": bad_mask,
            "position_ids": positions,
        },
        str(bad_path),
    )
    with pytest.raises(runner.PredictionError, match="not binary"):
        runner._load_observation_cell({"path": bad_path})


def test_prediction_manifest_fragment_keeps_truth_closed_and_join_fields() -> None:
    # Keep this small contract assertion model-free: the fields consumed by the
    # joint scorer are explicit and the runner's public statuses remain closed.
    assert runner.STUDENT_SCHEMA == "token-reconstruction.trr-p06-student-prediction-manifest.v1"
    assert runner.CELL_ORDER == (
        "pile__public_base",
        "pile__public_lora_2601",
        "finance__public_base",
        "finance__public_lora_2601",
    )
    assert runner.METHOD_ORDER == (
        "p06_positionwise_diagonal",
        "p06_past_only",
        "p06_full_record",
    )
    assert runner.WARMUP_PASSES == 1
    assert runner.MEASURED_PASSES == 3
    assert runner.PAD_TOKEN_ID == -1
