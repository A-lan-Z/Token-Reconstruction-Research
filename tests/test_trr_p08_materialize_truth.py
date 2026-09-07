from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.trr_p08 import materialize_truth


def _sequence_digest(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.int32).tobytes(order="C")).hexdigest()


def test_truth_adapter_requires_explicit_execute_before_any_input(tmp_path: Path) -> None:
    args = Namespace(execute=False, repository_root=tmp_path, joint_freeze=tmp_path / "missing.json", output_dir=tmp_path / "out", expected_plan_sha256="a" * 64, source_selection=None, universe=None, tokenizer=None, pile_arrow=None, finance_arrow=None)
    with pytest.raises(materialize_truth.TruthMaterializationError, match="explicit --execute"):
        materialize_truth.prepare_truth(args)
    assert not (tmp_path / "out").exists()


def test_truth_adapter_hashes_synthetic_records_with_frozen_order() -> None:
    arrays: dict[str, np.ndarray] = {}
    expected: dict[str, list[str]] = {}
    records: dict[str, list[object]] = {}
    for domain, offset in (("pile", 1000), ("finance", 2000)):
        value = np.zeros((256, 128), dtype=np.int32)
        value[:, 0] = materialize_truth.BOS_TOKEN_ID
        for row in range(256):
            value[row, 1:] = offset + row * 127 + np.arange(127, dtype=np.int32)
        arrays[domain] = value
        expected[domain] = [_sequence_digest(row) for row in value]
        records[domain] = [SimpleNamespace(token_ids=row.tolist(), record_id=f"{domain}-{index}") for index, row in enumerate(value)]
    got, observed = materialize_truth._arrays_and_hashes(records, expected=expected)
    assert set(got) == {"pile", "finance"}
    assert observed == expected
    np.testing.assert_array_equal(got["pile"], arrays["pile"])


def test_truth_adapter_rejects_wrong_synthetic_sequence_fingerprint() -> None:
    value = np.zeros((256, 128), dtype=np.int32)
    value[:, 0] = materialize_truth.BOS_TOKEN_ID
    records = {domain: [SimpleNamespace(token_ids=row.tolist(), record_id=f"{domain}-{index}") for index, row in enumerate(value)] for domain in ("pile", "finance")}
    expected = {domain: [_sequence_digest(row) for row in value] for domain in records}
    records["finance"][0].token_ids[1] = 99
    with pytest.raises(materialize_truth.TruthMaterializationError, match="sequence fingerprints"):
        materialize_truth._arrays_and_hashes(records, expected=expected)
