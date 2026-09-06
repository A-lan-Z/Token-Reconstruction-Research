"""Synthetic and fail-closed checks for the TRR-0008 selector adapter."""

from types import SimpleNamespace

import pytest

from scripts import trr0008_select_public as selector


def test_selection_refuses_explicit_draft_decision_fixture(tmp_path):
    import json

    source = selector.Path("experiments/TRR-0008/planning/decision_contract.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["status"] = "PROSPECTIVE_DRAFT_PENDING_OWNER_FREEZE"
    candidate = tmp_path / "decision_contract.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="frozen decision contract"):
        selector._validate_decision_contract(candidate)


def test_real_p06_loader_uses_task_owned_exact_copy():
    summary, source, sequence = selector._p06_opaque()
    assert summary["file"]["path"].endswith(
        "experiments/TRR-0008/planning/approved_opaque/p06_opaque_source_sequence_reservation.json"
    )
    assert summary["file"]["bytes"] == selector.planning.P06_OPAQUE_BYTES
    assert summary["file"]["sha256"] == selector.planning.P06_OPAQUE_SHA256
    assert summary["authorized_original_path"].endswith(
        "p06_opaque_source_sequence_reservation.json"
    )
    assert len(source) == 512
    assert len(sequence) == 512


def test_p06_source_byte_gate_refuses_explicit_pending_fixture(tmp_path):
    import json

    source = selector.Path("experiments/TRR-0008/coordination/planning_status.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["p06_hash_compatibility"]["source_hash_byte_input_status"] = "PENDING_P06_PRODUCER_CONFIRMATION"
    candidate = tmp_path / "planning_status.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="source-hash byte input"):
        selector._validate_planning_status(candidate)


def test_selector_classifies_h128_opaque_sequence_without_serializing_ids():
    tokens = tuple(range(selector.SEQUENCE_TOKENS))
    candidate = SimpleNamespace(
        style="pile",
        record_id="synthetic",
        row_index=7000,
        token_ids=tokens,
        public_record_sha256="a" * 64,
        final_sequence_sha256="b" * 64,
    )
    exclusions = SimpleNamespace(
        ids={"pile": set(), "finance": set()},
        hashes={"pile": set(), "finance": set()},
        indices={"pile": set(), "finance": set()},
    )
    p04 = SimpleNamespace(source_hashes=frozenset(), sequence_hashes_129=frozenset())
    reason = selector._classify_candidate(
        candidate,
        exclusions=exclusions,
        p04=p04,
        p06_source=frozenset(),
        p06_sequence=frozenset({selector._p06_sequence_digest(tokens)}),
        seen_public_hashes=set(),
        seen_final_sequences=set(),
    )
    assert reason == "excluded_p06_h128_sequence_hash"


def test_selection_metadata_is_capture_compatible_and_source_free():
    def row(record_id: str, source: str, sequence: str, index: int):
        return SimpleNamespace(
            selection_metadata=lambda: {
                "record_id": record_id,
                "public_record_sha256": source,
                "dataset_key": "pile",
                "dataset_id": "NeelNanda/pile-10k",
                "split": "train",
                "revision": "r" * 40,
                "row_index": index,
                "source_index": index,
                "full_token_count": 128,
                "post_bos_token_count": 127,
                "valid_tokens": 128,
                "final_sequence_sha256": sequence,
            }
        )

    metadata = selector._selection_metadata(
        {
            "pile": [row("p", "a" * 64, "b" * 64, 7000)],
            "finance": [row("f", "c" * 64, "d" * 64, 12000)],
        }
    )
    assert set(metadata["pile"][0]) == {
        "record_id",
        "public_record_sha256",
        "dataset_key",
        "dataset_id",
        "split",
        "revision",
        "row_index",
        "source_index",
        "full_token_count",
        "post_bos_token_count",
        "valid_tokens",
        "final_sequence_sha256",
    }
    assert "source_text" not in metadata["pile"][0]
    assert "token_ids" not in metadata["pile"][0]


def test_reservation_is_hash_only_and_domain_free(monkeypatch):
    monkeypatch.setattr(selector, "EXPECTED_RECORDS_BY_DOMAIN", {"pile": 2, "finance": 3})
    records = {
        "pile": [
            {"public_record_sha256": f"{i:064x}", "final_sequence_sha256": f"{i + 10:064x}"}
            for i in range(2)
        ],
        "finance": [
            {"public_record_sha256": f"{i + 20:064x}", "final_sequence_sha256": f"{i + 30:064x}"}
            for i in range(3)
        ],
    }
    selection = {
        "schema": selector.SELECTION_SCHEMA,
        "task_id": selector.TASK_ID,
        "status": selector.SELECTION_STATUS,
        "source_text_or_target_labels": False,
        "truth_opened": False,
        "truth_created": False,
        "p06_hash_compatibility": {
            "source_hash_byte_input_status": "VERIFIED_P06_PRODUCER_CONFIRMATION"
        },
        "selection_rule": {
            "source_text_or_token_ids_written": False,
            "records": records,
        },
    }
    payload = selector._reservation_payload(
        {"path": "selection.json", "bytes": 1, "sha256": "e" * 64}, selection
    )
    assert payload["counts"] == {"public_record_sha256": 5, "final_sequence_sha256": 5}
    assert payload["privacy_boundary"]["hash_only"] is True
    assert set(payload["hashes"]) == {"public_record_sha256", "final_sequence_sha256"}
    assert "pile" not in payload["hashes"]
    assert "finance" not in payload["hashes"]
    assert payload["privacy_boundary"]["contains_record_ids"] is False
    assert payload["privacy_boundary"]["contains_token_ids"] is False


def test_frozen_contract_requires_explicit_numeric_gates(tmp_path):
    import json

    source = selector.Path("experiments/TRR-0008/planning/decision_contract.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["status"] = "FROZEN_DECISION_CONTRACT_BEFORE_SOURCE_SELECTION"
    del payload["primary"]["component_alpha"]
    candidate = tmp_path / "decision_contract.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="primary confidence allocation"):
        selector._validate_decision_contract(candidate)
