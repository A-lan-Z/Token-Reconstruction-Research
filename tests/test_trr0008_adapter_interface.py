"""Synthetic selector-to-evaluation adapter compatibility checks.

These tests deliberately use descriptor and identity metadata only.  They do
not open Arrow rows, load a model, create a selection, capture activations, or
prepare truth.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import trr0008_select_public as selector
from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_capture_public as trr6_capture
from scripts import trr0008_eval_capture as capture
from scripts import trr0008_eval_truth as truth


def _synthetic_source_inputs() -> dict[str, object]:
    return {
        "pile_arrow": [
            {"path": "/synthetic/pile.arrow", "bytes": 11, "sha256": "a" * 64}
        ],
        "finance_arrow": [
            {"path": "/synthetic/finance.arrow", "bytes": 13, "sha256": "b" * 64}
        ],
        "tokenizer": {"path": "/synthetic/tokenizer"},
    }


def test_selector_descriptors_match_qualified_capture_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    token_descriptor = {
        "path": "/synthetic/tokenizer",
        "files": [{"path": "/synthetic/tokenizer/tokenizer.json", "bytes": 17, "sha256": "c" * 64}],
    }
    monkeypatch.setattr(selector.trusted, "_tokenizer_descriptor", lambda path: token_descriptor)
    frozen = selector._public_source_descriptors(_synthetic_source_inputs())

    assert set(frozen) == {"pile", "finance", "tokenizer"}
    for style in ("pile", "finance"):
        assert set(frozen[style]) == {
            "dataset_key",
            "dataset_id",
            "split",
            "revision",
            "arrow_files",
            "reserved_holdout",
        }
        assert frozen[style]["dataset_key"] == style
        assert frozen[style]["arrow_files"] == _synthetic_source_inputs()[f"{style}_arrow"]
        assert frozen[style]["reserved_holdout"] == dict(selector.SOURCE_PARTITIONS[style])
    assert frozen["tokenizer"] == token_descriptor

    monkeypatch.setattr(
        trr6_capture.trusted,
        "_dataset_descriptor",
        lambda paths, *, style: frozen[style],
    )
    monkeypatch.setattr(trr6_capture.trusted, "_tokenizer_descriptor", lambda path: token_descriptor)
    trr6_capture._validate_source_descriptors(
        {"public_sources_frozen": frozen},
        pile_paths=(Path("/synthetic/pile.arrow"),),
        finance_paths=(Path("/synthetic/finance.arrow"),),
        tokenizer_path=Path("/synthetic/tokenizer"),
    )


def test_local_source_hash_rules_match_reported_p06_convention() -> None:
    text = "  original source \n"
    assert trusted._sha256_bytes(text.encode("utf-8")) == hashlib.sha256(text.encode("utf-8")).hexdigest()

    system, user, assistant = trusted._finance_fields(
        {
            "system": "  ",
            "user": "",
            "instruction": " instruction ",
            "input": " input ",
            "assistant": "",
            "output": " output ",
        }
    )
    assert system is None
    assert user == "instruction\n\ninput"
    assert assistant == "output"
    canonical = trusted._canonical_json([system, user, assistant]).encode("utf-8")
    assert trusted._sha256_bytes(canonical) == hashlib.sha256(canonical).hexdigest()


def test_capture_accepts_selector_planning_binding_shape(tmp_path: Path) -> None:
    bindings: dict[str, dict[str, object]] = {}
    for label, content in {
        "decision_contract": b"decision",
        "identity_inventory": b"inventory",
        "plan": b"plan",
    }.items():
        path = tmp_path / f"{label}.json"
        path.write_bytes(content)
        bindings[label] = {
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    checked = capture._validate_planning_bindings(
        {"planning_bindings": bindings}, root=tmp_path
    )
    assert checked == bindings


def test_truth_reuses_capture_materialization_and_checks_record_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = {"pile": [{"record_id": "pile-record"}], "finance": [{"record_id": "finance-record"}]}
    records = {
        "pile": [SimpleNamespace(record_id="pile-record")],
        "finance": [SimpleNamespace(record_id="finance-record")],
    }
    selection_record = {"path": "selection.json", "bytes": 1, "sha256": "d" * 64}
    monkeypatch.setattr(
        truth.trr8_capture,
        "_load_selection",
        lambda path, *, repository_root: ({"synthetic": True}, selection_record, rows, {"pile": 1, "finance": 1}),
    )
    monkeypatch.setattr(truth.trr6_capture, "_validate_source_descriptors", lambda *args, **kwargs: None)
    monkeypatch.setattr(truth.trusted, "_load_tokenizer", lambda path: object())
    monkeypatch.setattr(truth.trusted, "_load_arrow_dataset", lambda paths: object())
    monkeypatch.setattr(
        truth.trr6_capture,
        "_materialize_selected",
        lambda selection_rows, *, datasets, tokenizer: records,
    )

    selection, checked_record, checked_rows, checked_records, counts = truth._load_inputs(
        root=tmp_path,
        selection_path=tmp_path / "selection.json",
        tokenizer_path=tmp_path / "tokenizer",
        pile_paths=(tmp_path / "pile.arrow",),
        finance_paths=(tmp_path / "finance.arrow",),
    )
    assert selection == {"synthetic": True}
    assert checked_record == selection_record
    assert checked_rows == rows
    assert checked_records == records
    assert counts == {"pile": 1, "finance": 1}


def test_truth_rejects_materialized_order_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = {"pile": [{"record_id": "pile-record"}], "finance": [{"record_id": "finance-record"}]}
    selection_record = {"path": "selection.json", "bytes": 1, "sha256": "d" * 64}
    monkeypatch.setattr(
        truth.trr8_capture,
        "_load_selection",
        lambda path, *, repository_root: ({"synthetic": True}, selection_record, rows, {"pile": 1, "finance": 1}),
    )
    monkeypatch.setattr(truth.trr6_capture, "_validate_source_descriptors", lambda *args, **kwargs: None)
    monkeypatch.setattr(truth.trusted, "_load_tokenizer", lambda path: object())
    monkeypatch.setattr(truth.trusted, "_load_arrow_dataset", lambda paths: object())
    monkeypatch.setattr(
        truth.trr6_capture,
        "_materialize_selected",
        lambda selection_rows, *, datasets, tokenizer: {
            "pile": [SimpleNamespace(record_id="different-pile-record")],
            "finance": [SimpleNamespace(record_id="finance-record")],
        },
    )

    with pytest.raises(truth.TruthError, match="materialized truth row order"):
        truth._load_inputs(
            root=tmp_path,
            selection_path=tmp_path / "selection.json",
            tokenizer_path=tmp_path / "tokenizer",
            pile_paths=(tmp_path / "pile.arrow",),
            finance_paths=(tmp_path / "finance.arrow",),
        )
