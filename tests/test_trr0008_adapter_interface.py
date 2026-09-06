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
import torch

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


def test_capture_entry_resolves_all_explicit_paths_without_public_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise capture_public's real entry path with all loaders/capture stubbed.

    This is a path and create-only smoke test.  It deliberately does not open
    Arrow data, a tokenizer, model weights, source text, labels, or a GPU.
    """
    (tmp_path / "model").mkdir()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    selection = {
        "source_ranges_half_open": {
            "pile": [7000, 10000],
            "finance": [12000, 20000],
        },
    }
    selection_record = {
        "path": str(selection_path),
        "bytes": selection_path.stat().st_size,
        "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    }
    selected_rows = {
        "pile": [{"record_id": "pile-synthetic"}],
        "finance": [{"record_id": "finance-synthetic"}],
    }
    counts = {"pile": 1, "finance": 1}
    monkeypatch.setattr(
        capture,
        "_load_selection",
        lambda path, *, repository_root: (selection, selection_record, selected_rows, counts),
    )
    monkeypatch.setattr(capture.trr6_capture, "_validate_source_descriptors", lambda *a, **k: None)
    monkeypatch.setattr(capture.trusted, "_load_tokenizer", lambda path: object())
    monkeypatch.setattr(capture.trusted, "_load_arrow_dataset", lambda paths: object())
    monkeypatch.setattr(
        capture.trr6_capture,
        "_materialize_selected",
        lambda rows, *, datasets, tokenizer: rows,
    )
    monkeypatch.setattr(
        capture.trr6_capture,
        "_batches",
        lambda records: {"pile": "pile-batch", "finance": "finance-batch"},
    )
    monkeypatch.setattr(capture.trusted, "_device", lambda value: torch.device("cpu"))
    monkeypatch.setattr(
        capture.trusted,
        "_dataset_descriptor",
        lambda paths, *, style: {"style": style, "paths": [str(path) for path in paths]},
    )
    monkeypatch.setattr(
        capture.trusted,
        "_tokenizer_descriptor",
        lambda path: {"path": str(path)},
    )
    monkeypatch.setattr(capture.trr4, "_runtime_snapshot", lambda path: {"path": str(path)})
    monkeypatch.setattr(capture, "_capture_source_code", lambda root: {})

    def fake_capture_condition(**kwargs):
        condition = kwargs["condition"]
        observations = {
            f"{style}__{condition}": {"path": f"synthetic-{style}-{condition}.safetensors"}
            for style in capture.STYLE_ORDER
        }
        return observations, {"condition": condition, "synthetic": True}

    monkeypatch.setattr(capture, "_capture_condition", fake_capture_condition)
    args = SimpleNamespace(
        execute=True,
        repository_root=tmp_path,
        selection=selection_path,
        pile_arrow=[tmp_path / "pile.arrow"],
        finance_arrow=[tmp_path / "finance.arrow"],
        tokenizer=tmp_path / "tokenizer",
        model_snapshot=tmp_path / "model",
        lora_config=tmp_path / "lora-config.json",
        lora_update=tmp_path / "lora-update.safetensors",
        output_root=tmp_path / "experiments/TRR-0008/evaluation/capture-smoke",
        device="cuda",
    )

    result = capture.capture_public(args)

    assert result["status"] == "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH"
    assert result["truth_opened"] is False
    output_root = args.output_root
    assert (output_root / "observations.json").is_file()
    assert (output_root / "panel.json").is_file()
    assert (output_root / "capture.json").is_file()
    assert not (output_root / "failure.json").exists()
