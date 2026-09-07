from __future__ import annotations

import hashlib
import json

import pytest

from scripts.trr_p08.source_binding import (
    SourceBindingError,
    bind_opaque_reservation,
    reject_payload_keys,
    verify_descriptor,
)


def _write_json(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_descriptor_hash_binding_fails_closed(tmp_path):
    path = tmp_path / "descriptor.json"
    expected = _write_json(path, {"schema": "test.v1", "status": "metadata"})
    ref = verify_descriptor(path, expected)
    assert ref["sha256"] == expected
    with pytest.raises(SourceBindingError, match="SHA256 mismatch"):
        verify_descriptor(path, "0" * 64)


def test_payload_keys_are_rejected_without_loading_source_values():
    with pytest.raises(SourceBindingError, match="source_text"):
        reject_payload_keys({"metadata": {"source_text": "must not persist"}})
    with pytest.raises(SourceBindingError, match="token_ids"):
        reject_payload_keys({"token_ids": [1, 2, 3]})


def test_hash_only_opaque_reservation_contract(tmp_path):
    path = tmp_path / "reservation.json"
    value = {
        "schema": "token-reconstruction.test-opaque.v1",
        "privacy_boundary": {
            "hash_only": True,
            "contains_model_weights": False,
            "contains_record_ids": False,
            "contains_source_indices": False,
            "contains_source_text": False,
            "contains_target_labels": False,
            "contains_token_ids": False,
            "contains_truth": False,
        },
        "hashes": {
            "public_record_sha256": ["a" * 64, "b" * 64],
            "final_sequence_sha256": ["c" * 64],
        },
    }
    expected = _write_json(path, value)
    bound = bind_opaque_reservation(
        path,
        expected,
        expected_schema="token-reconstruction.test-opaque.v1",
        expected_counts={"public_record_sha256": 2, "final_sequence_sha256": 1},
    )
    assert bound["hash_only"] is True
    assert bound["counts"] == {"public_record_sha256": 2, "final_sequence_sha256": 1}

    value["privacy_boundary"]["contains_token_ids"] = True
    _write_json(path, value)
    with pytest.raises(SourceBindingError, match="privacy boundary"):
        bind_opaque_reservation(
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_schema="token-reconstruction.test-opaque.v1",
            expected_counts={"public_record_sha256": 2, "final_sequence_sha256": 1},
        )


def test_opaque_count_mismatch_fails_closed(tmp_path):
    path = tmp_path / "reservation.json"
    value = {
        "schema": "token-reconstruction.test-opaque.v1",
        "privacy_boundary": {"hash_only": True},
        "hashes": {"public_record_sha256": ["a" * 64]},
    }
    expected = _write_json(path, value)
    with pytest.raises(SourceBindingError, match="count mismatch"):
        bind_opaque_reservation(
            path,
            expected,
            expected_schema="token-reconstruction.test-opaque.v1",
            expected_counts={"public_record_sha256": 2},
        )
