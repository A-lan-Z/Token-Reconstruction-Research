from __future__ import annotations

import pytest
import torch

from token_reconstruction.access import AccessContractError, BoundaryObservation
from token_reconstruction.io import ObservationIOError, load_observation, save_observation


def observation() -> BoundaryObservation:
    return BoundaryObservation(
        activation=torch.randn(2, 3, 4, dtype=torch.float32),
        attention_mask=torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.long),
        cut_depth=4,
        source_id="record-0001",
        metadata={"tensor_layout": "batch-sequence-hidden"},
    )


def test_create_only_round_trip(tmp_path) -> None:
    original = observation()
    path = tmp_path / "observation.safetensors"
    digest = save_observation(original, path)
    loaded = load_observation(path)

    assert len(digest) == 64
    assert torch.equal(loaded.activation, original.activation)
    assert torch.equal(loaded.attention_mask, original.attention_mask)
    assert torch.equal(loaded.position_ids, original.position_ids)
    assert loaded.cut_depth == original.cut_depth
    assert loaded.source_id == original.source_id
    assert loaded.metadata == original.metadata

    with pytest.raises(ObservationIOError, match="already exists"):
        save_observation(original, path)


@pytest.mark.parametrize(
    "metadata",
    [
        {"true_token_ids": [1, 2, 3]},
        {"nested": {"target_text": "hidden"}},
        {"evaluation": [{"oracle_score": 1.0}]},
    ],
)
def test_prohibited_truth_metadata_is_rejected(metadata) -> None:
    candidate = observation()
    candidate = BoundaryObservation(
        activation=candidate.activation,
        attention_mask=candidate.attention_mask,
        position_ids=candidate.position_ids,
        cut_depth=candidate.cut_depth,
        source_id=candidate.source_id,
        metadata=metadata,
    )
    with pytest.raises(AccessContractError, match="prohibited per-record truth"):
        candidate.validate()


def test_shape_mismatch_is_rejected() -> None:
    candidate = observation()
    candidate = BoundaryObservation(
        activation=candidate.activation,
        attention_mask=torch.ones(2, 2, dtype=torch.long),
        position_ids=candidate.position_ids,
        cut_depth=candidate.cut_depth,
        source_id=candidate.source_id,
    )
    with pytest.raises(AccessContractError, match="attention_mask"):
        candidate.validate()
