from __future__ import annotations

import torch
import pytest

from token_reconstruction.p04_student import (
    AffineStudent,
    GRUAffineStudent,
    METHOD_D,
    METHOD_H,
    METHOD_S,
    P04StudentError,
    StudentArchitectureConfig,
    _deterministic_lowest_id,
    build_student,
    load_student_state,
    prediction_tensor,
    save_student_state,
)


def _small_config() -> StudentArchitectureConfig:
    return StudentArchitectureConfig(hidden_size=4, vocab_size=7, gru_width=3)


def test_all_arms_have_trainable_affine_and_common_zero_residual() -> None:
    config = _small_config()
    affine = AffineStudent(config)
    activation = torch.randn(2, 5, 4)
    for method in (METHOD_S, METHOD_H, METHOD_D):
        model = build_student(method, config=config)
        assert model.affine.linear.weight.requires_grad
        assert model.affine.linear.bias.requires_grad
        assert model.affine.log_scale.requires_grad
        assert torch.count_nonzero(model.gru_up.weight).item() == 0
        assert torch.count_nonzero(model.gru_up.bias).item() == 0
        assert torch.allclose(model.projected_hidden(activation), affine.projected_hidden(activation))


def test_prediction_chunk_crossing_and_right_padding() -> None:
    config = _small_config()
    model = GRUAffineStudent(config)
    table = torch.nn.functional.normalize(torch.randn(7, 4), dim=-1)
    token_ids = torch.tensor([[2, 1, 4, 0, 0], [5, 6, 3, 2, 1]])
    activations = table[token_ids]
    valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    predictions, ties = prediction_tensor(
        model,
        activations,
        table,
        device=torch.device("cpu"),
        valid_mask=valid,
        record_batch_size=2,
        projection_chunk=3,
    )
    assert predictions.shape == (2, 5)
    assert torch.equal(predictions[0, :3], token_ids[0, :3])
    assert torch.equal(predictions[1], token_ids[1])
    assert torch.equal(predictions[0, 3:], torch.tensor([-1, -1], dtype=torch.int32))
    assert torch.equal(ties[0, 3:], torch.zeros(2, dtype=torch.int32))
    with pytest.raises(P04StudentError, match="right-padded"):
        prediction_tensor(model, activations, table, device=torch.device("cpu"), valid_mask=valid.roll(1, 1))


def test_lowest_id_ties_are_explicit_and_chunk_independent() -> None:
    logits = torch.tensor([[0.0, 2.0, 2.0, 1.0], [4.0, 4.0, 4.0, 0.0]])
    ids, counts = _deterministic_lowest_id(logits)
    assert torch.equal(ids, torch.tensor([1, 0]))
    assert torch.equal(counts, torch.tensor([2, 3], dtype=torch.int32))


def test_state_round_trip_preserves_method_and_predictions(tmp_path) -> None:
    config = _small_config()
    model = build_student(METHOD_H, config=config)
    state_path = tmp_path / "student.safetensors"
    receipt = save_student_state(model, state_path, method_id=METHOD_H, seed=1737, config=config)
    assert receipt["sha256"]
    loaded = load_student_state(state_path, method_id=METHOD_H, device=torch.device("cpu"), config=config)
    table = torch.nn.functional.normalize(torch.randn(7, 4), dim=-1)
    activations = table[torch.tensor([[1, 2, 3], [4, 5, 6]])]
    expected, _ = prediction_tensor(model, activations, table, device=torch.device("cpu"), projection_chunk=2)
    actual, _ = prediction_tensor(loaded, activations, table, device=torch.device("cpu"), projection_chunk=2)
    assert torch.equal(actual, expected)
    with pytest.raises(P04StudentError):
        load_student_state(state_path, method_id=METHOD_D, device=torch.device("cpu"), config=config)
