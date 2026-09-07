"""CPU checks for the strict P09 fixed-state deployment loader."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from scripts.trr0010_p09_fixed_loader import P09FixedLoaderError, load_p09_fixed_state


STATE = Path("/tmp/trr-p09-runtime/fixed-control-b0-r3/states/checkpoint_step_008000.safetensors")
KWARGS = {
    "expected_state_sha256": "d2477cdf11422cb2d028196d72775825246f65f7e07394952375b3929627475a",
    "expected_selected_step": 8000,
    "expected_bank_manifest_sha256": "aefaa5f47aa1042dffa7210767ed4a0be1fd848c373f8cd6de1ad4c739035a31",
    "expected_fit_manifest_sha256": "aefaa5f47aa1042dffa7210767ed4a0be1fd848c373f8cd6de1ad4c739035a31",
    "expected_schedule_semantic_sha256": "9aaad9c030f2f9b801f91f956c97f858966580edd21229cebb350dcde358f2c3",
    "expected_base_state_sha256": "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14",
    "expected_embedding_sha256": "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1",
    "expected_runner_state_sha256": "58aeaf1b3e16c760cc3bbaccd8413a5f2674fd07f59e4f527a53b8bf3ca0243a",
}


@pytest.mark.skipif(not STATE.is_file(), reason="A2 B0 selected P09 state is unavailable")
def test_load_p09_state_matches_every_serialized_tensor_exactly() -> None:
    raw = load_file(str(STATE), device="cpu")
    model = load_p09_fixed_state(STATE, **KWARGS)
    loaded = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    assert set(loaded) == set(raw)
    assert model.training is False
    assert getattr(model, "_trr_p09_state_sha256") == KWARGS["expected_state_sha256"]
    for name in sorted(raw):
        assert torch.equal(loaded[name], raw[name]), name


@pytest.mark.skipif(not STATE.is_file(), reason="A2 B0 selected P09 state is unavailable")
def test_load_p09_state_rejects_metadata_substitution() -> None:
    wrong = dict(KWARGS)
    wrong["expected_selected_step"] = 7999
    with pytest.raises(P09FixedLoaderError, match="selected_step"):
        load_p09_fixed_state(STATE, **wrong)
