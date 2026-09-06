from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_build_eligibility as producer
from token_reconstruction.trr0005_contract import BOS_TOKEN_ID


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _opaque_info(values: list[str], *, available: bool = True) -> dict[str, object]:
    ordered = list(values)
    unique = sorted(set(values))
    return {
        "available": available,
        "source_field": "test",
        "ordered_count": len(ordered),
        "distinct_count": len(unique),
        "ordered_values": ordered,
        "unique_values": unique,
        "ordered_canonical_json_sha256": _digest(
            json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ),
        "ordered_newline_sha256": _digest(("\n".join(ordered) + "\n").encode()),
        "unique_set_canonical_json_sha256": _digest(
            json.dumps(unique, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ),
    }


def _candidate(*, token_count: int = 129) -> trusted.FreshRecord:
    token_ids = (BOS_TOKEN_ID,) + tuple(range(1, token_count))
    return trusted.FreshRecord(
        style="pile",
        dataset_key="pile",
        dataset_id="dataset",
        split="train",
        revision="revision",
        row_index=7000,
        record_id="pile-row",
        public_record_sha256="a" * 64,
        token_ids=token_ids,
        final_sequence_sha256="b" * 64,
    )


def test_opaque_field_digest_conventions_are_checked() -> None:
    values = ["a" * 64, "b" * 64, "a" * 64]
    parsed, summary = producer._opaque_field_values(_opaque_info(values), label="test")
    assert parsed == {"a" * 64, "b" * 64}
    assert summary["applied_distinct_count"] == 2

    bad = _opaque_info(values)
    bad["ordered_newline_sha256"] = "c" * 64
    with pytest.raises(producer.EligibilityError, match="ordered_newline_sha256"):
        producer._opaque_field_values(bad, label="test")


def test_p04_source_and_129_sequence_hashes_are_applied_before_deduplication() -> None:
    candidate = _candidate()
    sequence_hash = trusted._sequence_digest(candidate.token_ids[:129])
    empty = trusted.ExclusionSets(
        ids={style: set() for style in producer.STYLE_ORDER},
        hashes={style: set() for style in producer.STYLE_ORDER},
        indices={style: set() for style in producer.STYLE_ORDER},
        sources=[],
    )
    opaque = producer.OpaqueExclusions(
        source_hashes=frozenset({candidate.public_record_sha256}),
        sequence_hashes_129=frozenset({sequence_hash}),
        fields={},
        exchange={},
    )
    assert (
        producer._classify_valid_candidate(
            candidate,
            style="pile",
            exclusions=empty,
            opaque=opaque,
            seen_public_hashes=set(),
            seen_final_sequences=set(),
        )
        == "excluded_opaque_source_hash"
    )
    opaque = producer.OpaqueExclusions(
        source_hashes=frozenset(),
        sequence_hashes_129=frozenset({sequence_hash}),
        fields={},
        exchange={},
    )
    assert (
        producer._classify_valid_candidate(
            candidate,
            style="pile",
            exclusions=empty,
            opaque=opaque,
            seen_public_hashes=set(),
            seen_final_sequences=set(),
        )
        == "excluded_opaque_sequence_hash"
    )


def test_domain_counts_emit_commitments_without_raw_identity() -> None:
    stats = producer.DomainCounts("pile", 7000, 10000, 3000)
    stats.valid_rows = 1
    stats.eligible_unique = 1
    stats.valid_identity_commitments.append(
        {
            "record_id_sha256": "c" * 64,
            "public_record_sha256": "d" * 64,
            "final_sequence_sha256": "e" * 64,
        }
    )
    value = stats.as_dict(requested_per_domain=1536)
    encoded = json.dumps(value, sort_keys=True)
    assert "record_id" not in encoded
    assert "token_ids" not in encoded
    assert value["capacity_for_requested_per_domain"]["sufficient"] is False


def test_p04_descriptor_rejects_changed_exchange(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "exchange.json"
    path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(producer, "P04_EXCHANGE_SHA256", "0" * 64)
    with pytest.raises(producer.EligibilityError, match="hash changed"):
        producer._describe_p04_exchange(path)
