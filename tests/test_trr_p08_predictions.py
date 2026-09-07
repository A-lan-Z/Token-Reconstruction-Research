from __future__ import annotations

from pathlib import Path
import time

from safetensors.torch import save_file
import torch

from scripts.trr_p08 import run_predictions as runner


class _ToyModel:
    hidden_size = 3
    vocabulary_size = 5

    def projected_hidden(self, activation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return activation.float()

    def logits_from_rows(
        self,
        projected: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros((rows.numel(), self.vocabulary_size), dtype=torch.float32)


def test_predict_batch_uses_lowest_id_ties_and_public_padding(monkeypatch):
    monkeypatch.setattr(runner, "SEQUENCE_TOKENS", 4)
    monkeypatch.setattr(runner, "HIDDEN_SIZE", 3)
    monkeypatch.setattr(runner, "VOCABULARY_SIZE", 5)
    monkeypatch.setattr(runner, "BOS_TOKEN_ID", 4)
    activations = torch.zeros((2, 4, 3), dtype=torch.bfloat16)
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    ids, ties = runner.predict_batch(
        _ToyModel(),
        torch.zeros((5, 3), dtype=torch.float32),
        activations,
        valid,
        device=torch.device("cpu"),
        projection_chunk=2,
    )
    assert ids.tolist() == [[4, 0, 0, 0], [4, 0, runner.PAD_TOKEN_ID, runner.PAD_TOKEN_ID]]
    assert ties.tolist() == [[1, 5, 5, 5], [1, 5, 0, 0]]


def test_observation_loader_validates_masks_positions_and_dtype(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner, "RECORDS_PER_DOMAIN", 2)
    monkeypatch.setattr(runner, "SEQUENCE_TOKENS", 4)
    monkeypatch.setattr(runner, "HIDDEN_SIZE", 3)
    path = tmp_path / "observation.safetensors"
    save_file(
        {
            "activations": torch.zeros((2, 4, 3), dtype=torch.bfloat16),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
            "position_ids": torch.arange(4, dtype=torch.int64).repeat(2, 1),
        },
        str(path),
    )
    loaded = runner._load_observation_cell({"path": path})
    assert loaded["activations"].shape == (2, 4, 3)
    assert loaded["mask"].tolist() == [[True, True, True, True], [True, True, False, False]]
    assert len(loaded["attention_mask_sha256"]) == 64
    assert len(loaded["position_ids_sha256"]) == 64


def test_cell_timing_has_warmup_measured_repeat_and_load_boundary(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner, "RECORDS_PER_DOMAIN", 2)
    monkeypatch.setattr(runner, "SEQUENCE_TOKENS", 4)
    monkeypatch.setattr(runner, "HIDDEN_SIZE", 3)
    monkeypatch.setattr(runner, "VOCABULARY_SIZE", 5)
    monkeypatch.setattr(runner, "BOS_TOKEN_ID", 4)
    path = tmp_path / "observation.safetensors"
    save_file(
        {
            "activations": torch.zeros((2, 4, 3), dtype=torch.bfloat16),
            "attention_mask": torch.ones((2, 4), dtype=torch.bool),
            "position_ids": torch.arange(4, dtype=torch.int64).repeat(2, 1),
        },
        str(path),
    )
    ids, ties, timing = runner._predict_cell(
        _ToyModel(),
        torch.zeros((5, 3), dtype=torch.float32),
        {"cell_id": "pile__public_base", "path": path},
        device=torch.device("cpu"),
        started=time.perf_counter(),
        max_seconds=60.0,
        min_free_gib=0.001,
        max_reserved_gib=1.0,
        max_rss_gib=16.0,
        min_host_gib=0.001,
        batch_records=1,
        projection_chunk=2,
    )
    assert ids.shape == (2, 4)
    assert ties.shape == (2, 4)
    assert timing["warmup_passes"] == 1
    assert timing["measured_passes"] == 3
    assert len(timing["measured_seconds"]) == 3
    assert timing["repeat_prediction_exact"] is True
    assert timing["observation_load_excluded_from_measured_interval"] is True
