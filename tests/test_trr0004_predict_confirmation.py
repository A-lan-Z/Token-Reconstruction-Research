from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import trr0004_fresh_confirmation as fc
import trr0004_predict_confirmation as runner


def _cell(tmp_path: Path, *, records: int = 2, sequence_tokens: int = 3) -> fc.FreshCell:
    mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0]][:records], dtype=torch.long
    )
    positions = torch.tensor(
        [[0, 1, 2], [0, 1, 1]][:records], dtype=torch.long
    )
    return fc.FreshCell(
        cell_id="finance__public_base",
        style="finance",
        condition="public_base",
        record_ids=tuple(f"r{index}" for index in range(records)),
        activations=torch.zeros(records, sequence_tokens, fc.HIDDEN_SIZE, dtype=torch.bfloat16),
        attention_mask=mask,
        position_ids=positions,
        observation_path=tmp_path / "observation.safetensors",
        observation_sha256="0" * 64,
    )


def test_batch_validation_accepts_ids_normalized_inside_callback(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    normalized = torch.tensor(
        [[fc.BOS_TOKEN_ID, 18, 19], [fc.BOS_TOKEN_ID, 21, fc.INVALID_TOKEN_ID]],
        dtype=torch.long,
    )
    assert runner._validate_normalized_batch_prediction(normalized, cell) is None


def test_batch_validation_rejects_ids_not_normalized_in_callback(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    raw = torch.tensor([[fc.BOS_TOKEN_ID, -2, 19], [fc.BOS_TOKEN_ID, 21, fc.INVALID_TOKEN_ID]], dtype=torch.long)
    with pytest.raises(runner.PredictionRunnerError, match="invalid active token"):
        runner._validate_normalized_batch_prediction(raw, cell)


class _FakeLens(torch.nn.Module):
    def forward(self, activation: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        del embeddings
        # A deterministic four-class stand-in lets this test exercise the
        # direct top-1 path without materializing or inspecting top-k ranks.
        result = torch.zeros((activation.shape[0], 4), dtype=torch.float32, device=activation.device)
        result[:, 2] = 4.0
        return result


def test_a1_adapter_is_direct_top1_and_has_no_candidate_output() -> None:
    adapter = runner._A1Adapter(
        lens=_FakeLens(),
        embeddings=torch.zeros(4, fc.HIDDEN_SIZE, dtype=torch.float32),
    )
    row_h = torch.zeros(3, fc.HIDDEN_SIZE, dtype=torch.float32)
    prediction = adapter(row_h, torch.tensor([1, 1, 0]), torch.tensor([0, 1, 1]))
    assert prediction.tolist() == [fc.BOS_TOKEN_ID, 2, fc.INVALID_TOKEN_ID]
    assert adapter.evidence()["candidate_output"].startswith("forbidden")
    assert adapter.evidence()["candidate_simulations"] == 0


def test_write_prediction_serializes_a2_candidates_and_binding(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    path = tmp_path / "predictions.safetensors"
    predictions = torch.tensor(
        [[fc.BOS_TOKEN_ID, 2, 3], [fc.BOS_TOKEN_ID, 4, fc.INVALID_TOKEN_ID]], dtype=torch.long
    )
    candidates = torch.ones(2, 3, 4, dtype=torch.long)
    candidates[1, 2] = fc.INVALID_TOKEN_ID
    scores = torch.zeros(2, 3, 4, dtype=torch.float32)
    scores[1, 2] = float("-inf")
    binding = {"method_id": runner.M_A2, "method_rule": "fixture"}
    runner._write_prediction(
        path=path,
        cell=cell,
        method_id=runner.M_A2,
        predictions=predictions,
        candidates=candidates,
        candidate_scores=scores,
        binding=binding,
        panel_sha256="1" * 64,
        selection_plan_sha256="2" * 64,
    )
    tensors = load_file(str(path), device="cpu")
    assert set(tensors) == {"predictions", "candidates", "candidate_scores"}
    assert tensors["candidates"][:, 0].eq(fc.BOS_TOKEN_ID).all()
    assert tensors["candidates"][1, 2].eq(fc.INVALID_TOKEN_ID).all()
    # Ensure metadata is JSON and contains the complete binding rather than an
    # unbound method name shortcut.
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert json.loads(metadata["binding_json"]) == binding
    assert json.loads(metadata["geometry_json"])["sequence_tokens"] == 3


def test_selected_adapter_loader_does_not_load_other_method_states(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    registration = {"bindings": {method_id: {} for method_id in runner.EXPECTED_METHOD_IDS}}

    def fake_single_state(binding, *, method_id: str, root: Path) -> Path:
        del binding, root
        calls.append(("state", method_id))
        path = tmp_path / f"{method_id}.state"
        path.write_bytes(method_id.encode("ascii"))
        return path

    def fake_affine(*args, **kwargs):
        del args, kwargs
        calls.append(("decoder", runner.M_AFFINE))
        return torch.nn.Identity()

    monkeypatch.setattr(runner, "_single_state", fake_single_state)
    monkeypatch.setattr(runner, "load_historical_affine_ce", fake_affine)
    adapters = runner._load_method_adapters(
        method_id=runner.M_AFFINE,
        registration=registration,
        root=tmp_path,
        precut=None,
        lens=None,
        # Loader construction does not need a full public table; geometry is
        # validated at the resource boundary before this function is called.
        embeddings=torch.zeros(1, 1),
        device=torch.device("cpu"),
    )

    assert set(adapters) == {runner.M_AFFINE}
    assert calls == [("state", runner.M_AFFINE), ("decoder", runner.M_AFFINE)]


def test_normalized_predictor_applies_bos_and_padding_inside_callback(tmp_path: Path) -> None:
    cell = _cell(tmp_path)

    def raw_predictor(row_h: torch.Tensor, row_mask: torch.Tensor, row_positions: torch.Tensor) -> torch.Tensor:
        del row_h, row_mask, row_positions
        return torch.tensor([77, 9, 42], dtype=torch.long)

    predictor = runner._normalized_predictor(raw_predictor, cell)
    output = predictor(
        torch.zeros(3, runner.fc.HIDDEN_SIZE),
        torch.tensor([1, 1, 0]),
        torch.tensor([0, 1, 1]),
    )
    assert output.tolist() == [runner.fc.BOS_TOKEN_ID, 9, runner.fc.INVALID_TOKEN_ID]

def test_timing_summary_retains_raw_per_record_repeat_receipts(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    records = [
        {
            "record_index": 0,
            "warmup_runs": 1,
            "warmup_seconds": [0.01],
            "measured_runs": 3,
            "measured_seconds": [0.02, 0.03, 0.04],
            "repeated_prediction_exact": True,
            "mismatch_runs": [],
        },
        {
            "record_index": 1,
            "warmup_runs": 1,
            "warmup_seconds": [0.05],
            "measured_runs": 3,
            "measured_seconds": [0.06, 0.07, 0.08],
            "repeated_prediction_exact": True,
            "mismatch_runs": [],
        },
    ]
    adapter = type("Adapter", (), {"method_id": runner.M_AFFINE, "evidence": lambda self: {}})()
    summary = runner._timing_summary(
        {
            "warmup_runs": 1,
            "measured_runs": 3,
            "records": records,
            "total_elapsed_seconds": 0.36,
        },
        adapter=adapter,
        cell=cell,
        path=tmp_path / "cell.run.json",
        root=tmp_path,
        peak={"process_max_rss_bytes": 10, "cuda_peak_allocated_bytes": 20, "cuda_peak_reserved_bytes": 30},
    )
    assert summary["per_record_timing_records"] == records
    assert summary["per_record_measured_mean_seconds"] == pytest.approx([0.03, 0.07])


def test_peak_memory_envelope_includes_cold_and_cell_peaks() -> None:
    assert runner._peak_memory_envelope(
        [
            {"process_max_rss_bytes": 100, "cuda_peak_allocated_bytes": 200, "cuda_peak_reserved_bytes": None},
            {"process_max_rss_bytes": 90, "cuda_peak_allocated_bytes": 250, "cuda_peak_reserved_bytes": 275},
            {"process_max_rss_bytes": 125, "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": 300},
        ]
    ) == {
        "process_max_rss_bytes": 125,
        "cuda_peak_allocated_bytes": 250,
        "cuda_peak_reserved_bytes": 300,
    }

