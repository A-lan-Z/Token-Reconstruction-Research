"""Synthetic fail-closed checks for the TRR-0009 selector adapter."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import trr0009_select_public as selector
from scripts import trr0009_plan as planning


def test_draft_contract_refuses_real_contract_fixture(tmp_path):
    payload = json.loads(
        selector.Path("experiments/TRR-0009/planning/decision_contract.json").read_text(
            encoding="utf-8"
        )
    )
    payload["status"] = "PROSPECTIVE_DRAFT_PENDING_OWNER_FREEZE"
    candidate = tmp_path / "decision_contract.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="frozen decision contract"):
        selector._validate_decision_contract(candidate)


def test_finalized_contract_fixture_passes_contract_gate(tmp_path):
    payload = json.loads(
        selector.Path("experiments/TRR-0009/planning/decision_contract.json").read_text(
            encoding="utf-8"
        )
    )
    payload["status"] = "FROZEN_DECISION_CONTRACT_BEFORE_SOURCE_SELECTION"
    candidate = tmp_path / "decision_contract.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    record, actual = selector._validate_decision_contract(candidate)
    assert record["bytes"] == candidate.stat().st_size
    assert actual["methods"]["continued_adaptable_readout"].endswith("gain_bias_readout")


def test_projection_inventory_refuses_selection_gate(tmp_path):
    payload = json.loads(
        selector.Path("experiments/TRR-0009/planning/source_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "IDENTITY_CAPACITY_PROJECTION_PENDING_FINAL_COUNT_ONLY_SCAN"
    candidate = tmp_path / "source_inventory.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="finalized count-only inventory"):
        selector._validate_planning_status(candidate)


def test_h128_opaque_sequence_classifier_keeps_ids_out_of_metadata():
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


def test_selection_metadata_is_source_free_and_capture_compatible():
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
    assert metadata["pile"][0]["valid_tokens"] == 128
    assert "source_text" not in metadata["pile"][0]
    assert "token_ids" not in metadata["pile"][0]


def test_reservation_is_hash_only_and_uses_trr9_counts(monkeypatch):
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
    assert payload["task_id"] == "TRR-0009"
    assert payload["counts"] == {"public_record_sha256": 5, "final_sequence_sha256": 5}
    assert payload["privacy_boundary"]["hash_only"] is True
    assert "pile" not in payload["hashes"]
    assert "finance" not in payload["hashes"]


def test_final_inventory_refuses_draft_before_source_access(tmp_path):
    # Clone the live contract into a state-independent draft fixture.  The
    # repository contract is frozen now, so this test must not depend on it
    # remaining draft forever.
    contract_payload = json.loads((selector.Path("experiments/TRR-0009/planning/decision_contract.json")).read_text())
    contract_payload["status"] = "PROSPECTIVE_DRAFT_PENDING_OWNER_FREEZE"
    decision_contract = tmp_path / "decision_contract.json"
    decision_contract.write_text(json.dumps(contract_payload), encoding="utf-8")
    args = SimpleNamespace(
        repository_root=selector.Path("."),
        output=tmp_path / "source_inventory_final.json",
        decision_contract=decision_contract,
        method_freeze=tmp_path / "method_freeze.json",
        trr7_method_freeze=selector.Path("experiments/TRR-0007/method_freeze.json"),
        tokenizer=tmp_path / "missing-tokenizer",
        pile_arrow=[],
        finance_arrow=[],
        exclude_source=[],
        p08_opaque=None,
        no_p08_reservation=True,
    )
    with pytest.raises(planning.PlanError, match="owner-frozen decision contract"):
        planning._final_inventory(args)
    assert not args.output.exists()


def test_final_inventory_requires_explicit_p08_resolution():
    args = SimpleNamespace(p08_opaque=None, no_p08_reservation=False)
    with pytest.raises(planning.PlanError, match="exactly one of --p08-opaque or --no-p08-reservation"):
        planning._load_p08_option(args)


def test_generic_opaque_loader_accepts_approved_h128_shape_without_values_in_summary():
    summary, source_hashes, sequence_hashes = planning._load_generic_opaque_reservation(
        planning.P06_OPAQUE, label="approved-p06-test"
    )
    assert summary["source_hash_count"] == 512
    assert summary["sequence_hash_count"] == 512
    assert len(source_hashes) == len(sequence_hashes) == 512
    assert "values" not in summary


def test_task_method_freeze_missing_file_refuses(tmp_path):
    with pytest.raises(selector.SelectionError, match="selected-method freeze is unavailable"):
        selector._validate_task_method_freeze(
            tmp_path / "method_freeze.json",
            root=tmp_path,
            decision_record={"sha256": "a" * 64},
        )


def test_task_method_freeze_unfrozen_fixture_refuses(tmp_path):
    payload = {
        "schema": selector.TASK_METHOD_FREEZE_SCHEMA,
        "task_id": "TRR-0009",
        "status": "DRAFT_SELECTED_METHODS",
        "source_selection_started": False,
        "truth_opened": False,
        "private_or_truth_payload_read": False,
        "fresh_evaluation_started": False,
        "state_selection_frozen": False,
    }
    path = tmp_path / "method_freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="not frozen"):
        selector._validate_task_method_freeze(
            path, root=tmp_path, decision_record={"sha256": "a" * 64}
        )


def test_task_method_freeze_frozen_fixture_still_requires_actual_states(tmp_path):
    payload = {
        "schema": selector.TASK_METHOD_FREEZE_SCHEMA,
        "task_id": "TRR-0009",
        "status": selector.TASK_METHOD_FREEZE_STATUS,
        "source_selection_started": False,
        "truth_opened": False,
        "private_or_truth_payload_read": False,
        "fresh_evaluation_started": False,
        "state_selection_frozen": True,
        "decision_contract": {"sha256": "a" * 64},
        "decision_rules": {
            "decision_contract_sha256": "a" * 64,
            "selection_metric": "earliest maximum public validation style-balanced token accuracy including step zero",
            "validation_interval": 100,
            "step_zero_eligible": True,
            "same_selection_opportunities": True,
        },
        "method_order": list(selector.TASK_METHOD_ORDER),
        "state_bindings": {},
        "source_code": [],
        "code_commit": "b" * 40,
    }
    path = tmp_path / "method_freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="lacks all actual method states"):
        selector._validate_task_method_freeze(
            path, root=tmp_path, decision_record={"sha256": "a" * 64}
        )
