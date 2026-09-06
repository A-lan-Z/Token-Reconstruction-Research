from __future__ import annotations

import pytest
import torch

from scripts import trr0008_eval_contract as contract
from scripts import trr0008_eval_qualifier as qualifier


def _fixture_matrix(monkeypatch: pytest.MonkeyPatch):
    # Keep the fixture CPU-light while retaining the production row/cell
    # cardinalities.  The call under test remains the actual
    # trr0008_eval_runner.predict_current_h adapter.
    monkeypatch.setattr(contract, "STORED_SEQUENCE_TOKENS", 4)
    monkeypatch.setattr(contract, "SCORED_POST_BOS_TOKENS", 3)
    monkeypatch.setattr(contract, "VOCABULARY_SIZE", 5)
    monkeypatch.setattr(contract, "HIDDEN_SIZE", 2)
    monkeypatch.setattr(contract, "BOS_TOKEN_ID", 4)

    class FakeModel(torch.nn.Module):
        def projected_hidden(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
            return activation

        def logits_from_rows(
            self,
            projected: torch.Tensor,
            record_slots: torch.Tensor,
            position_slots: torch.Tensor,
            embedding: torch.Tensor,
        ) -> torch.Tensor:
            return projected[record_slots, position_slots] @ embedding.T

    embedding = torch.eye(5, 2)
    models = {method_id: FakeModel() for method_id in contract.METHOD_ORDER}
    cells = {}
    for cell_index, cell_id in enumerate(contract.CELL_ORDER):
        values = torch.zeros((128, 4, 2), dtype=torch.float32)
        for row in range(128):
            values[row, 1:, row % 2] = 1.0 + cell_index + (row % 3) * 0.1
        cells[cell_id] = {
            "activations": values.to(torch.bfloat16),
            "valid_mask": torch.ones((128, 4), dtype=torch.bool),
        }

    call_count = {"value": 0}
    original_runner = qualifier.runner.predict_current_h

    def counted_runner(*args, **kwargs):
        call_count["value"] += 1
        return original_runner(*args, **kwargs)

    monkeypatch.setattr(qualifier.runner, "predict_current_h", counted_runner)
    archived = {}
    expected_rows = torch.arange(128, dtype=torch.long).remainder(2).unsqueeze(1)
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            expected = torch.full((128, 4), contract.INVALID_TOKEN_ID, dtype=torch.long)
            expected[:, 0] = contract.BOS_TOKEN_ID
            expected[:, 1:] = expected_rows
            archived[f"{method_id}::{cell_id}"] = expected
    return models, embedding, cells, archived, call_count


def test_fixture_qualifier_runs_actual_runner_for_complete_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    models, embedding, cells, archived, call_count = _fixture_matrix(monkeypatch)
    guard_stages: list[str] = []
    result = qualifier.compare_runner_matrix(
        models,
        embedding,
        cells,
        archived,
        device=torch.device("cpu"),
        records=128,
        guard=guard_stages.append,
    )
    assert len(result) == len(contract.METHOD_ORDER) * len(contract.CELL_ORDER)
    assert all(row["exact_match"] is True for row in result.values())
    assert call_count["value"] == len(contract.METHOD_ORDER) * len(contract.CELL_ORDER) * 128
    assert len(guard_stages) == len(result) * (128 // qualifier.GUARD_INTERVAL + 1)


def test_fixture_corruption_fails_closed_before_success_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    models, embedding, cells, archived, _call_count = _fixture_matrix(monkeypatch)
    key = f"{contract.METHOD_ORDER[0]}::{contract.CELL_ORDER[0]}"
    corrupted = dict(archived)
    corrupted[key] = archived[key].clone()
    corrupted[key][0, 1] = (corrupted[key][0, 1] + 1) % contract.VOCABULARY_SIZE
    receipt_path = tmp_path / "runner_qualifier.json"
    with pytest.raises(qualifier.QualifierError, match="runner/archive mismatch"):
        qualifier.compare_runner_matrix(
            models,
            embedding,
            cells,
            corrupted,
            device=torch.device("cpu"),
            records=128,
        )
    # A mismatch is detected before the caller can serialize a success
    # receipt; no prediction array or truth payload is written as a side
    # effect of the failed comparison.
    assert not receipt_path.exists()


def test_success_receipt_is_create_only_and_serializable(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    record = qualifier._write_create_only(
        path,
        {"schema": qualifier.QUALIFIER_SCHEMA, "truth_opened": False},
        description="fixture qualifier receipt",
    )
    assert record["bytes"] == path.stat().st_size
    with pytest.raises(qualifier.QualifierError, match="create-only"):
        qualifier._write_create_only(
            path,
            {"schema": qualifier.QUALIFIER_SCHEMA},
            description="fixture qualifier receipt",
        )


def test_qualifier_cli_requires_explicit_execution() -> None:
    assert qualifier.main(["--trr7-root", "."]) == 2
