from __future__ import annotations

import pytest
import torch

import scripts.trr0012_transfer_analysis as analysis
from scripts.trr0012_transfer_analysis import (
    TransferAnalysisError,
    validate_truth_order_declarations,
)


def test_truth_order_binding_rejects_swapped_domain_digests() -> None:
    expected = {
        "panel_sha256": "p" * 64,
        "domain_record_order_sha256": {
            "finance": "f" * 64,
            "pile": "b" * 64,
        },
    }
    manifest = {
        "panel_sha256": expected["panel_sha256"],
        "domain_record_order_sha256": dict(expected["domain_record_order_sha256"]),
    }
    assert validate_truth_order_declarations(
        manifest,
        expected_panel_sha256=expected["panel_sha256"],
        expected_domain_record_order_sha256=expected["domain_record_order_sha256"],
    ) == expected

    swapped = {
        "panel_sha256": expected["panel_sha256"],
        "domain_record_order_sha256": {
            "finance": "b" * 64,
            "pile": "f" * 64,
        },
    }
    with pytest.raises(TransferAnalysisError, match="ordered-ID digests"):
        validate_truth_order_declarations(
            swapped,
            expected_panel_sha256=expected["panel_sha256"],
            expected_domain_record_order_sha256=expected["domain_record_order_sha256"],
        )


def test_loaded_geometry_adapter_flattens_record_position_readout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis, "RECORDS_PER_DOMAIN", 2)
    monkeypatch.setattr(analysis, "SCORED_POST_BOS_TOKENS", 3)
    monkeypatch.setattr(analysis, "HIDDEN_SIZE", 4)

    activation = torch.ones((2, 3, 4), dtype=torch.float32)
    changed_activation = activation + 0.1
    projected = torch.zeros((2, 3, 4), dtype=torch.float32)
    projected[..., 0] = 1.0
    readout = torch.zeros((2, 3, 2, 4), dtype=torch.float32)
    readout[..., 0, 0] = 1.0
    readout[..., 1, 1] = 1.0
    logits = torch.zeros((2, 3, 2), dtype=torch.float32)
    logits[..., 0] = 2.0
    ids = torch.zeros((2, 3, 2), dtype=torch.long)
    ids[..., 0] = 4
    ids[..., 1] = 5

    def cell(activation_value: torch.Tensor, geometry_readout: torch.Tensor) -> dict[str, object]:
        return {
            "activation": activation_value,
            "geometry": {
                "projected_features": projected,
                "top_runner_logits": logits,
                "top_runner_ids": ids,
                "top_runner_readout": geometry_readout,
                "logit_scale": torch.tensor(2.0),
            },
            "observation": {"sha256": "a" * 64},
            "geometry_binding": {"sha256": "b" * 64},
        }

    result = analysis._summarize_loaded_cell_geometry(  # noqa: SLF001
        cell(activation, readout),
        cell(changed_activation, readout.clone()),
        method_id="expanded_fixed",
        variant_id="early_prefix_eps1e3",
        domain="finance",
        parameter_binding={"actual_delta_l2": None, "actual_base_l2": None},
    )

    assert result["rows"] == 6
    provenance = result["geometry_provenance"]
    assert provenance["readout_shape_before_adapter"] == [2, 3, 2, 4]
    assert provenance["readout_shape_after_adapter"] == [6, 2, 4]
    assert provenance["readout_adapter"] == "reshape_records_positions_to_rows"
