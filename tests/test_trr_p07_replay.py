from __future__ import annotations

from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts.trr_p07 import run_replay as runner


def _cell(path: Path, *, records: int, rows: tuple[int, ...]) -> runner.Cell:
    asset = runner._file_record(path)
    return runner.Cell(
        panel="synthetic",
        cell_id="pile__public_base",
        domain="pile",
        target="public_base",
        path=asset.path,
        asset=asset,
        record_ids_sha256="a" * 64,
        records=records,
        subset_indices=rows,
    )


def test_approved_subset_is_evenly_spaced_and_plan_hash_bound() -> None:
    indices = runner.select_trr0006_subset(records=runner.TRR0006_RECORDS)
    assert indices[:5] == (0, 6, 12, 18, 24)
    assert indices[-1] == 1530
    assert len(indices) == 256
    asset, plan = runner._validate_plan(Path("experiments/TRR-P07/plan.json"), root=Path.cwd())
    assert asset.sha256 == runner.PLAN_SHA256
    assert plan["panels"]["trr0006_evenly_spaced_1of6"]["records_per_domain"] == 256


def test_observation_rows_reads_selected_indices_not_contiguous_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records, positions, hidden = 12, 128, 4
    activation = torch.stack(
        [torch.full((positions, hidden), float(row), dtype=torch.float32) for row in range(records)]
    ).to(dtype=torch.bfloat16)
    mask = torch.ones((records, positions), dtype=torch.bool)
    position_ids = torch.arange(positions, dtype=torch.int64).repeat(records, 1)
    path = tmp_path / "obs.safetensors"
    save_file({"activations": activation, "attention_mask": mask, "position_ids": position_ids}, str(path))
    cell = _cell(path, records=records, rows=(0, 6, 11))
    monkeypatch.setattr(runner, "HIDDEN_SIZE", hidden)

    actual, actual_mask, actual_positions, evidence = runner._observation_rows(cell=cell, indices=(0, 6, 11))

    assert actual[:, 0, 0].tolist() == [0.0, 6.0, 11.0]
    assert actual_mask.shape == (3, positions)
    assert actual_positions.tolist() == position_ids[[0, 6, 11]].tolist()
    assert evidence["row_indices"] == [0, 6, 11]


def test_both_execution_paths_use_lowest_id_ties_and_fixed_bos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "SEQUENCE_TOKENS", 4)
    monkeypatch.setattr(runner, "HIDDEN_SIZE", 3)
    monkeypatch.setattr(runner, "VOCABULARY_SIZE", 5)
    monkeypatch.setattr(runner, "BOS_TOKEN_ID", 4)

    class ToyModel:
        hidden_size = 3
        vocabulary_size = 5

        def projected_hidden(self, activation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return activation.float()

        def logits_from_rows(self, projected: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
            return torch.zeros((rows.numel(), 5), dtype=torch.float32)

        def __call__(self, activation: torch.Tensor, mask: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
            return torch.zeros((1, 4, 5), dtype=torch.float32)

    model = ToyModel()
    embedding = torch.zeros((5, 3), dtype=torch.float32)
    activations = torch.zeros((2, 4, 3), dtype=torch.bfloat16)
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    batch_ids, batch_ties = runner.predict_p06_batch(model, embedding, activations, valid, device=torch.device("cpu"), projection_chunk=2)
    native_ids, native_ties = runner.predict_old_native_record(model, embedding, activations[0], valid[0], device=torch.device("cpu"))

    assert batch_ids.tolist() == [[4, 0, 0, 0], [4, 0, -1, -1]]
    assert batch_ties.tolist() == [[1, 5, 5, 5], [1, 5, 0, 0]]
    assert native_ids.tolist() == [4, 0, 0, 0]
    assert native_ties.tolist() == [1, 5, 5, 5]


def test_prediction_and_tie_serialization_is_create_only(tmp_path: Path) -> None:
    observation = tmp_path / "observation.safetensors"
    save_file(
        {
            "activations": torch.zeros((2, 4, 3), dtype=torch.bfloat16),
            "attention_mask": torch.ones((2, 4), dtype=torch.bool),
            "position_ids": torch.arange(4, dtype=torch.int64).repeat(2, 1),
        },
        str(observation),
    )
    state = tmp_path / "state.safetensors"
    state.write_bytes(b"state")
    obs_asset = runner._file_record(observation)
    state_asset = runner._file_record(state)
    cell = runner.Cell("synthetic", "pile__public_base", "pile", "public_base", observation, obs_asset, "a" * 64, 2, (0, 1))
    method = runner.Method("toy__seed1", "p06", "p06_past_only", 1, state, state_asset, None, "toy", "p06_batch8_chunked_full_vocab")
    ids = torch.tensor([[runner.BOS_TOKEN_ID, 0, 1, 2], [runner.BOS_TOKEN_ID, 2, 3, 4]], dtype=torch.long)
    ties = torch.ones_like(ids, dtype=torch.int64)

    descriptor = runner._save_prediction_artifacts(
        output_root=tmp_path / "out",
        panel="p06_panel",
        cell=cell,
        method=method,
        ids=ids,
        ties=ties,
        timing={"execution": "synthetic", "truth_opened": False},
        root=tmp_path,
    )

    assert descriptor["truth_opened"] is False
    with safe_open(str(tmp_path / "out/predictions/p06_panel/pile/public_base/seed-1/toy__seed1.safetensors"), framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {"predictions"}
        assert handle.metadata()["truth_opened"] == "false"
    with pytest.raises(runner.ReplayError, match="create-only"):
        runner._save_prediction_artifacts(
            output_root=tmp_path / "out",
            panel="p06_panel",
            cell=cell,
            method=method,
            ids=ids,
            ties=ties,
            timing={"execution": "synthetic", "truth_opened": False},
            root=tmp_path,
        )
