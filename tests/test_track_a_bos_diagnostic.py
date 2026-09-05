from __future__ import annotations

import importlib

import pytest
import torch


DIAGNOSTIC = importlib.import_module("trr0003_track_a_bos_diagnostic")


def test_selection_is_bounded_and_distinct() -> None:
    assert DIAGNOSTIC._select_indices([0]) == (0,)
    assert DIAGNOSTIC._select_indices([0, 23]) == (0, 23)
    with pytest.raises(DIAGNOSTIC.DiagnosticError, match="one or two"):
        DIAGNOSTIC._select_indices([])
    with pytest.raises(DIAGNOSTIC.DiagnosticError, match="one or two"):
        DIAGNOSTIC._select_indices([0, 1, 2])
    with pytest.raises(DIAGNOSTIC.DiagnosticError, match="distinct"):
        DIAGNOSTIC._select_indices([1, 1])
    with pytest.raises(DIAGNOSTIC.DiagnosticError, match="outside"):
        DIAGNOSTIC._select_indices([24])


def test_metrics_separate_bos_and_post_bos() -> None:
    reference = torch.ones(4, 3)
    actual = reference.clone()
    actual[0, 0] = 2.0
    metrics = DIAGNOSTIC._per_position_metrics(actual, reference)
    assert metrics["all_positions"]["max_abs"] == pytest.approx(1.0)
    assert metrics["bos_position"]["max_abs"] == pytest.approx(1.0)
    assert metrics["post_bos"]["max_abs"] == pytest.approx(0.0)
    assert metrics["post_bos"]["relative_l2"] == pytest.approx(0.0)


def test_metrics_reject_nonfinite_values() -> None:
    with pytest.raises(DIAGNOSTIC.DiagnosticError, match="non-finite"):
        DIAGNOSTIC._metrics(torch.tensor([[float("nan")]]), torch.ones(1, 1))
