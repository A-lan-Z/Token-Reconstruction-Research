"""Focused synthetic checks for the deferred TRR-P06 truth boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.trr_p06 import materialize_truth as truth


def _records(domain: str) -> list[SimpleNamespace]:
    rows: list[SimpleNamespace] = []
    for index in range(truth.scorer.RECORDS_PER_DOMAIN):
        values = np.arange(truth.SEQUENCE_TOKENS, dtype=np.int32) + index + (1000 if domain == "finance" else 0)
        values[0] = truth.BOS_TOKEN_ID
        rows.append(SimpleNamespace(record_id=f"{domain}-{index}", token_ids=tuple(int(value) for value in values)))
    return rows


def test_arrays_and_hashes_bind_frozen_sequence_order() -> None:
    records = {domain: _records(domain) for domain in truth.DOMAINS}
    expected: dict[str, list[str]] = {}
    for domain in truth.DOMAINS:
        expected[domain] = [
            __import__("hashlib").sha256(np.asarray(row.token_ids, dtype=np.int32).tobytes(order="C")).hexdigest()
            for row in records[domain]
        ]

    arrays, observed = truth._arrays_and_hashes(records, expected=expected)

    assert set(arrays) == set(truth.DOMAINS)
    assert all(array.shape == (truth.scorer.RECORDS_PER_DOMAIN, truth.SEQUENCE_TOKENS) for array in arrays.values())
    assert all(array.dtype == np.int32 for array in arrays.values())
    assert observed == expected


def test_arrays_and_hashes_reject_sequence_reordering() -> None:
    records = {domain: _records(domain) for domain in truth.DOMAINS}
    expected: dict[str, list[str]] = {domain: ["0" * 64] * truth.scorer.RECORDS_PER_DOMAIN for domain in truth.DOMAINS}

    with pytest.raises(truth.TruthMaterializationError, match="sequence fingerprints"):
        truth._arrays_and_hashes(records, expected=expected)
