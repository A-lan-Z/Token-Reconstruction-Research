from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_PATH = Path("experiments/TRR-0003/track_b/predict_cells.py").resolve()
_SPEC = spec_from_file_location("trr0003_track_b_predict_cells", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _cell(mask: list[list[int]]) -> SimpleNamespace:
    value = torch.tensor(mask, dtype=torch.long)
    return SimpleNamespace(
        cell_id="synthetic",
        records=value.shape[0],
        sequence_tokens=value.shape[1],
        attention_mask=value,
    )


def test_full_predictions_marks_padding_and_keeps_bos() -> None:
    cell = _cell([[1, 1, 0, 0], [1, 1, 1, 0]])
    flat = torch.tensor([11, 12, 13, 21, 22, 23], dtype=torch.int32)

    actual = _MODULE._full_predictions(cell, flat)

    assert actual.tolist() == [
        [128000, 11, -1, -1],
        [128000, 21, 22, -1],
    ]


def test_full_predictions_rejects_padded_bos() -> None:
    cell = _cell([[0, 1, 1]])
    flat = torch.tensor([11, 12], dtype=torch.int32)

    with pytest.raises(_MODULE.FootingError, match="BOS"):
        _MODULE._full_predictions(cell, flat)
