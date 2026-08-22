"""Focused fail-closed tests for the TRR-0001-R1 blind interface."""

from __future__ import annotations

import copy

import pytest

from token_reconstruction.blind_commitment import (
    BlindProtocolError,
    OBSERVATION_INDEX_SCHEMA,
    SANITIZED_CONFIG_SCHEMA,
    opaque_record_ids,
    private_selection_document,
    public_commitment,
    reveal_document,
    select_private_records,
    validate_observation_index,
    validate_public_commitment,
    validate_sanitized_config,
    verify_reveal,
)


REVISION = "a" * 40


class DummyTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        assert add_special_tokens is False
        index = int(text.split()[-1])
        return {"input_ids": [1000 + index + offset for offset in range(45)]}


def private_records() -> list[dict]:
    tokenizer = DummyTokenizer()
    rows = [
        (index, f"row {index}", tokenizer(f"row {index}", add_special_tokens=False)["input_ids"])
        for index in range(80)
    ]
    return select_private_records(
        key=bytes(range(32)),
        dataset_revision=REVISION,
        rows=rows,
        excluded_indices={1, 3, 5},
    )


def observation_index() -> dict:
    entries = []
    for condition in ("matched_public", "unavailable_target_lora"):
        for cut in (0, 4, 8):
            entries.append(
                {
                    "condition": condition,
                    "cut_depth": cut,
                    "path": f"observations/{condition}_cut{cut}.safetensors",
                    "bytes": 1,
                    "sha256": "b" * 64,
                }
            )
    return {
        "schema": OBSERVATION_INDEX_SCHEMA,
        "records": [{"record_id": value} for value in opaque_record_ids()],
        "entries": entries,
        "source_material_included": False,
    }


def sanitized_config() -> dict:
    return {
        "schema": SANITIZED_CONFIG_SCHEMA,
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R1",
        "model": {
            "id": "meta-llama/Llama-3.2-1B-Instruct",
            "revision": "9" * 40,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
        },
        "observation_index": {
            "path": "observation_index.json",
            "bytes": 1,
            "sha256": "c" * 64,
        },
        "inverse_states": [
            {"cut_depth": 4, "path": "inverses/cut4.safetensors", "bytes": 1, "sha256": "d" * 64},
            {"cut_depth": 8, "path": "inverses/cut8.safetensors", "bytes": 1, "sha256": "e" * 64},
        ],
        "record_order": opaque_record_ids(),
        "condition_order": ["matched_public", "unavailable_target_lora"],
        "cut_order": [0, 4, 8],
        "geometry": {
            "records": 64,
            "sequence_tokens": 40,
            "scored_tokens_per_record": 39,
            "hidden_size": 2048,
            "candidate_budget": 16,
        },
        "methods": ["direct_inverse", "causal_public_surrogate_search"],
        "execution": {
            "seed": 1729,
            "stopping": "all 39 scored positions",
            "abstention": "none",
            "score_batch_size": 64,
            "causal_record_batch_size": 16,
        },
        "access_contract": "process-enforced isolation manifest required",
        "truth_or_source_inputs": 0,
    }


def test_hiding_commitment_and_post_freeze_reveal_verification() -> None:
    key = bytes(range(32))
    records = private_records()
    public = public_commitment(
        key=key,
        records=records,
        dataset_id="public/data",
        dataset_revision=REVISION,
        created_utc="2026-08-22T00:00:00Z",
    )
    validate_public_commitment(public)
    serialized = repr(public)
    for forbidden in ("dataset_index", "text_sha256", "token_ids", key.hex()):
        assert forbidden not in serialized
    private = private_selection_document(
        key=key, records=records, created_utc="2026-08-22T00:00:00Z"
    )
    reveal = reveal_document(private, revealed_utc="2026-08-22T01:00:00Z")
    result = verify_reveal(
        public=public,
        reveal=reveal,
        dataset_revision=REVISION,
        excluded_indices={1, 3, 5},
        dataset_rows=[f"row {index}" for index in range(80)],
        tokenizer=DummyTokenizer(),
    )
    assert result["verified"] is True
    assert result["disjoint_from_original_records"] is True
    tampered = copy.deepcopy(reveal)
    tampered["records"][0]["token_ids"][1] += 1
    with pytest.raises(BlindProtocolError, match="commitment"):
        verify_reveal(
            public=public,
            reveal=tampered,
            dataset_revision=REVISION,
            excluded_indices={1, 3, 5},
            dataset_rows=[f"row {index}" for index in range(80)],
            tokenizer=DummyTokenizer(),
        )


def test_observation_schema_allows_only_opaque_ids() -> None:
    value = observation_index()
    validate_observation_index(value)
    exposed = copy.deepcopy(value)
    exposed["records"][0]["dataset_index"] = 7
    with pytest.raises(BlindProtocolError):
        validate_observation_index(exposed)
    derived = copy.deepcopy(value)
    derived["records"][0]["record_id"] = "blind-deadbeef"
    with pytest.raises(BlindProtocolError, match="opaque"):
        validate_observation_index(derived)


def test_sanitized_config_is_strict_and_rejects_nested_source_fields() -> None:
    value = sanitized_config()
    validate_sanitized_config(value)
    extra = copy.deepcopy(value)
    extra["debug"] = {}
    with pytest.raises(BlindProtocolError, match="fields changed"):
        validate_sanitized_config(extra)
    leaked = copy.deepcopy(value)
    leaked["execution"]["source_token_ids"] = [1, 2]
    with pytest.raises(BlindProtocolError):
        validate_sanitized_config(leaked)
    reordered = copy.deepcopy(value)
    reordered["record_order"][0], reordered["record_order"][1] = (
        reordered["record_order"][1],
        reordered["record_order"][0],
    )
    with pytest.raises(BlindProtocolError, match="order"):
        validate_sanitized_config(reordered)


def test_selection_is_keyed_deterministic_and_excludes_declared_rows() -> None:
    records = private_records()
    repeated = private_records()
    assert records == repeated
    assert len(records) == 64
    assert {row["dataset_index"] for row in records}.isdisjoint({1, 3, 5})
    assert [row["record_id"] for row in records] == opaque_record_ids()
