import json
from pathlib import Path

import pytest
import torch

from scripts import trr0009_loader_qualify as qualify


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments/TRR-0009/evaluation/trr8_loader_equivalence_fixture.json"
FREEZE = ROOT / "experiments/TRR-0009/training/method_freeze.json"


def test_fixture_is_bounded_and_truth_free():
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["schema"] == "token-reconstruction.trr0009-trr8-loader-equivalence-fixture.v1"
    assert fixture["records_to_check_per_cell"] == 8
    assert fixture["target_labels_loaded"] is False
    assert fixture["source_text_loaded"] is False
    assert fixture["truth_opened"] is False
    assert fixture["cell_order"] == [
        "pile__public_base",
        "pile__public_lora_2601",
        "finance__public_base",
        "finance__public_lora_2601",
    ]
    assert set(fixture["records_by_domain"]) == {"finance", "pile"}
    for cell in fixture["cell_order"]:
        assert cell in fixture["methods"]["unchanged_anchor"]["predictions"]
        assert cell in fixture["methods"]["published_reference"]["predictions"]
        assert fixture["methods"]["unchanged_anchor"]["predictions"][cell]["records"] >= fixture["records_to_check_per_cell"]
    assert set(fixture["methods"]) == {"unchanged_anchor", "published_reference"}


def test_logits_id_projection_rejects_non_b1_geometry():
    with pytest.raises(qualify.QualificationError, match="full forward geometry"):
        qualify._prediction_from_logits(
            torch.zeros((1, 1, 1), dtype=torch.float32),
            torch.ones((128,), dtype=torch.bool),
        )


def test_method_freeze_exposes_selected_fixed_and_adapt_bindings():
    freeze = json.loads(FREEZE.read_text())
    methods = freeze["state_bindings"]
    for method_id in ("continued_fixed_readout", "continued_adaptable_readout"):
        row = methods[method_id]
        assert row["selected_step"] == 400
        assert row["loader"]
        state = row["state"]
        assert Path(state["path"]).is_file()
        assert len(state["sha256"]) == 64
    assert methods["continued_adaptable_readout"]["loader"]["tensor_args"]
