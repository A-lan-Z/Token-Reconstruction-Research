from __future__ import annotations

import json
from pathlib import Path

import pytest

import trr0004_produce_confirmation as producer
import trr0004_fresh_confirmation as fc


class FakeTokenizer:
    pad_token_id = None
    bos_token_id = fc.BOS_TOKEN_ID

    def convert_tokens_to_ids(self, value: str) -> int:
        assert value == "<|end_of_text|>"
        return fc.PAD_TOKEN_ID

    def convert_ids_to_tokens(self, value: int) -> str:
        assert value == fc.PAD_TOKEN_ID
        return "<|end_of_text|>"

    def __call__(self, value: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        offset = int(value.rsplit("-", 1)[-1])
        return {"input_ids": [((offset + index) % 10_000) + 1 for index in range(150)]}

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool, date_string: str):
        assert tokenize is True and add_generation_prompt is False
        assert date_string == producer.DATE_STRING
        user = messages[-2]["content"]
        offset = int(user.rsplit("-", 1)[-1])
        return [fc.BOS_TOKEN_ID] + [((offset + index) % 10_000) + 1 for index in range(150)]


def _empty_exclusions() -> producer.ExclusionSets:
    return producer.ExclusionSets(
        ids={style: set() for style in fc.STYLE_ORDER},
        hashes={style: set() for style in fc.STYLE_ORDER},
        indices={style: set() for style in fc.STYLE_ORDER},
        sources=[],
    )


def test_selection_skips_known_indices_and_deduplicates_final_sequences() -> None:
    tokenizer = FakeTokenizer()
    pile = [{"text": f"pile-{index}"} for index in range(8)]
    exclusions = _empty_exclusions()
    exclusions.indices["pile"].add(1)
    seen: set[str] = set()
    selected, skipped = producer._select_style_records(
        pile,
        style="pile",
        tokenizer=tokenizer,
        exclusions=exclusions,
        records=3,
        seen_truncated_sequences=seen,
    )
    assert [row.raw_index for row in selected] == [0, 2, 3]
    assert skipped["excluded"] == 1
    assert len({row.truncated_sequence_sha256 for row in selected}) == 3
    assert all(set(row.panel_metadata(sequence_tokens=40)) == {
        "record_id", "public_record_sha256", "raw_index", "source_index", "valid_tokens"
    } for row in selected)


def test_exclusion_scan_does_not_descend_into_sensitive_payload(tmp_path: Path) -> None:
    source = tmp_path / "public_metadata.json"
    source.write_text(json.dumps({
        "dataset": "NeelNanda/pile-10k",
        "records": [{
            "record_id": "pile10k-00001-deadbeefdeadbeef",
            "text_sha256": "a" * 64,
            "raw_index": 1,
            "token_ids": [{"record_id": "pile10k-99999-badbadbadbadbadb", "raw_index": 999}],
        }],
    }) + "\n", encoding="utf-8")
    exclusions = producer._collect_exclusions([source])
    assert "pile10k-00001-deadbeefdeadbeef" in exclusions.ids["pile"]
    assert 1 in exclusions.indices["pile"]
    assert "pile10k-99999-badbadbadbadbadb" not in exclusions.ids["pile"]
    assert 999 not in exclusions.indices["pile"]


def test_method_freeze_requires_exact_registered_order(tmp_path: Path) -> None:
    marker = tmp_path / "freeze.json"
    marker.write_text(json.dumps({"status": "FROZEN_METHOD_STATES", "method_ids": list(fc.METHOD_IDS)}) + "\n")
    assert producer._validate_method_freeze(marker)["status"] == "FROZEN_METHOD_STATES"
    marker.write_text(json.dumps({"status": "FROZEN", "method_ids": list(reversed(fc.METHOD_IDS))}) + "\n")
    with pytest.raises(producer.ProducerError, match="exact five"):
        producer._validate_method_freeze(marker)


def test_private_truth_destination_is_create_only(tmp_path: Path) -> None:
    destination = producer._require_external_destination(tmp_path / "private" / "truth.safetensors", description="truth")
    assert not destination.exists()
    destination.write_bytes(b"private")
    with pytest.raises(producer.ProducerError, match="create-only"):
        producer._require_external_destination(destination, description="truth")
