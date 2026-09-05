from __future__ import annotations

import json
from pathlib import Path

import torch

from trr0004_affine_coverage_diagnostic import _coverage_json_path, _coverage_metrics


def test_coverage_metrics_uses_full_fit_frequencies_and_style_groups() -> None:
    predictions = torch.tensor([1, 2, 2, 4], dtype=torch.int32)
    labels = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    groups = ("alpaca", "alpaca", "pile", "pile")
    fit_frequencies = torch.tensor([0, 1, 2, 0, 5], dtype=torch.long)
    validation_frequencies = fit_frequencies.index_select(0, labels)

    result = _coverage_metrics(
        predictions, labels, groups, validation_frequencies, fit_frequencies
    )

    assert result["overall"]["rows"] == 4
    assert result["overall"]["correct"] == 3
    assert result["by_frequency_bucket"]["unseen_0"]["rows"] == 1
    assert result["by_validation_group"]["alpaca"]["overall"]["correct"] == 2
    assert result["by_validation_group"]["pile"]["overall"]["correct"] == 1
    assert result["fit_label_coverage"]["distinct_labels_seen"] == 3
    assert result["fit_label_coverage"]["distinct_validation_labels_unseen"] == 1


def test_actual_coverage_evidence_path_field_is_json_serializable(tmp_path: Path) -> None:
    """The evidence field used by ``run`` must not leak a ``Path`` object."""

    selected_path = tmp_path / "selected.safetensors"
    payload = {
        "provenance": {
            "selected_state": _coverage_json_path(selected_path),
        },
        "binding": {"method_id": "historical_affine_ce_no_vocab_bias"},
    }
    encoded = json.dumps(payload, allow_nan=False)
    assert str(selected_path) in encoded
