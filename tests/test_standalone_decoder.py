from __future__ import annotations

import pytest
import torch
from torch import nn

from token_reconstruction.standalone_decoder import (
    DecoderTrainingConfig,
    StandaloneDecoderError,
    TiedAffineTokenDecoder,
    _evaluate_decoder,
    complete_prediction_check,
    normalized_embedding_table,
    prediction_tensor,
    train_token_decoder,
    validate_embedding_table,
)


class _FixedLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, activation: torch.Tensor, embedding_table: torch.Tensor) -> torch.Tensor:
        del embedding_table
        return activation + self.anchor * 0.0


def test_decoder_curve_uses_weighted_partial_batch_metrics() -> None:
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.25],
        ]
    )
    labels = torch.tensor([0, 1, 2, 3, 0])
    loss, accuracy = _evaluate_decoder(
        _FixedLogits(), logits, labels, torch.eye(4), batch_size=3
    )
    assert loss == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())
    assert accuracy == 4 / 5


def test_tied_decoder_trains_and_evaluates_on_a_separate_device_safe_curve() -> None:
    torch.manual_seed(11)
    table = normalized_embedding_table(torch.randn(7, 4))
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    activations = table.index_select(0, labels) + 0.02 * torch.randn(10, 4)
    model, evidence = train_token_decoder(
        TiedAffineTokenDecoder(4, 7),
        activations[:7],
        labels[:7],
        table,
        config=DecoderTrainingConfig(
            steps=2,
            batch_size=3,
            learning_rate=1e-2,
            log_every=1,
            seed=13,
        ),
        device=torch.device("cpu"),
        eval_sets={"public_validation": (activations[7:], labels[7:])},
    )
    assert evidence["examples"] == 7
    assert len(evidence["learning_curve"]) == 2
    assert "public_validation_token_accuracy" in evidence["learning_curve"][-1]
    assert all(torch.isfinite(value).all().item() for value in model.parameters())


def test_prediction_is_direct_and_complete_without_candidate_state() -> None:
    table = normalized_embedding_table(torch.eye(5))
    model = TiedAffineTokenDecoder(5, 5)
    activation = table[[3, 1, 4]]
    prediction = prediction_tensor(
        model, activation, table, device=torch.device("cpu"), batch_size=2
    )
    assert prediction.shape == (3,)
    complete_prediction_check(
        {"tied_affine_token_ce": prediction.view(1, 3)},
        expected_methods=("tied_affine_token_ce",),
        expected_shape=(1, 3),
        vocab_size=5,
    )
    with torch.no_grad():
        invalid = prediction.clone()
        invalid[0] = 5
    with pytest.raises(StandaloneDecoderError):
        complete_prediction_check(
            {"tied_affine_token_ce": invalid.view(1, 3)},
            expected_methods=("tied_affine_token_ce",),
            expected_shape=(1, 3),
            vocab_size=5,
        )


def test_embedding_validation_is_explicit_and_fails_closed() -> None:
    table = torch.eye(4)
    validate_embedding_table(table, hidden_size=4, vocab_size=4)
    bad = table.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(StandaloneDecoderError, match="non-finite"):
        validate_embedding_table(bad, hidden_size=4, vocab_size=4)


def test_track_b_prepare_wires_public_dataset_and_model(monkeypatch, tmp_path) -> None:
    import argparse
    from types import SimpleNamespace

    import trr0003_track_b as runner

    source_plan = tmp_path / "plan.json"
    source_plan.write_text("{}\n")
    output_root = tmp_path / "prepared"
    calls: dict[str, object] = {}

    class _FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self._embedding = SimpleNamespace(weight=torch.ones(3, 3))

        def get_input_embeddings(self):
            return self._embedding

    fake_model = _FakeModel()
    fake_tokenizer = SimpleNamespace()
    fake_dataset = object()
    records = [
        {
            "record_id": f"fit-{index}",
            "dataset_index": index,
            "text_sha256": f"hash-{index}",
            "token_ids": [128000] + [index % 17] * 39,
        }
        for index in range(128)
    ]

    def fake_model_loader():
        calls["model"] = True
        return fake_tokenizer, fake_dataset, fake_model

    def fake_records(plan, split, *, tokenizer, dataset):
        calls["split"] = (split, tokenizer is fake_tokenizer, dataset is fake_dataset)
        return records

    def fake_capture(model, rows, batch_size):
        calls["capture"] = (model is fake_model, len(rows), batch_size)
        return torch.zeros((128, 40, 2048), dtype=torch.bfloat16)

    written: list[Path] = []

    def fake_save_file(tensors, path, metadata=None):
        del tensors, metadata
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock safetensors")
        written.append(path)

    monkeypatch.setattr(runner, "_model", fake_model_loader)
    monkeypatch.setattr(runner, "records_for_split", fake_records)
    monkeypatch.setattr(runner, "_capture_cut4", fake_capture)
    monkeypatch.setattr(runner, "normalized_embedding_table", lambda value: torch.zeros((1, 1)))
    monkeypatch.setattr(runner, "validate_embedding_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "save_file", fake_save_file)
    monkeypatch.setattr(runner, "seed_everything", lambda seed: None)
    monkeypatch.setattr(runner, "peak_memory", lambda: {})

    args = argparse.Namespace(
        source_plan=source_plan,
        output_root=output_root,
        record_batch_size=8,
    )
    assert runner._prepare(args) == 0
    assert calls["model"] is True
    assert calls["split"] == ("inverse_train", True, True)
    assert calls["capture"] == (True, 128, 8)
    assert {path.name for path in written} == {
        "fit_observations.safetensors",
        "fit_truth.safetensors",
        "public_normalized_embeddings.safetensors",
    }
    assert (output_root / "fit_records.json").is_file()
    assert (output_root / "prepare_evidence.json").is_file()
