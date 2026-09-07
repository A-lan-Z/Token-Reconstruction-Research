"""Synthetic tests for the TRR-0009 public source-evidence adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0009_eval_contract as contract
from scripts import trr0009_eval_truth as truth
from scripts import trr0009_eval_truth_source_compat as compat


def _selection(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    snapshot = tmp_path / "tokenizer-snapshot"
    snapshot.mkdir()
    blob_dir = tmp_path / "tokenizer-blobs"
    blob_dir.mkdir()
    for name, payload in (
        ("tokenizer.json", b"tokenizer"),
        ("tokenizer_config.json", b"config"),
        ("special_tokens_map.json", b"special"),
    ):
        blob = blob_dir / name
        blob.write_bytes(payload)
        (snapshot / name).symlink_to(blob)
    tokenizer = trusted._tokenizer_descriptor(snapshot)

    pile = tmp_path / "pile.arrow"
    finance = tmp_path / "finance.arrow"
    pile.write_bytes(b"pile-arrow")
    finance.write_bytes(b"finance-arrow")
    pile_descriptor = trusted._dataset_descriptor((pile,), style="pile")
    finance_descriptor = trusted._dataset_descriptor((finance,), style="finance")
    selection: dict[str, object] = {
        "public_sources_frozen": {
            "tokenizer": tokenizer,
            "pile": pile_descriptor,
            "finance": finance_descriptor,
        },
        "records_by_domain": {"pile": 1, "finance": 1},
    }
    return selection, snapshot, pile, finance


def _evidence(selection: dict[str, object], snapshot: Path, pile: Path, finance: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return compat.source_evidence_compat(
        selection,
        root=root,
        tokenizer_path=snapshot,
        pile_paths=(pile,),
        finance_paths=(finance,),
    )


def test_tiny_source_evidence_round_trip_binds_directory_components_and_arrows(tmp_path: Path) -> None:
    selection, snapshot, pile, finance = _selection(tmp_path)
    evidence = _evidence(selection, snapshot, pile, finance)
    header = {"source_evidence": evidence}
    assert header["source_evidence"] == _evidence(selection, snapshot, pile, finance)
    assert evidence["schema"] == compat.SOURCE_EVIDENCE_SCHEMA
    assert evidence["records_by_domain"] == {"pile": 1, "finance": 1}
    assert evidence["adapter_source"]["sha256"] == compat._adapter_source_record(root=Path(__file__).resolve().parents[1])["sha256"]
    assert set(evidence["tokenizer"]["files"]) == {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    assert all(item["symlink"] is True for item in evidence["tokenizer"]["files"].values())
    assert len(evidence["pile_arrow"]) == 1 and len(evidence["finance_arrow"]) == 1


def test_changed_tokenizer_component_fails_closed(tmp_path: Path) -> None:
    selection, snapshot, pile, finance = _selection(tmp_path)
    _evidence(selection, snapshot, pile, finance)
    (snapshot / "tokenizer.json").resolve().write_bytes(b"changed-tokenizer")
    with pytest.raises(compat.TruthSourceCompatibilityError, match="tokenizer"):
        _evidence(selection, snapshot, pile, finance)


def test_changed_tokenizer_directory_path_fails_closed(tmp_path: Path) -> None:
    selection, snapshot, pile, finance = _selection(tmp_path)
    other = tmp_path / "other-snapshot"
    other.mkdir()
    with pytest.raises(compat.TruthSourceCompatibilityError, match="directory differs"):
        _evidence(selection, other, pile, finance)


def test_changed_arrow_path_or_bytes_fails_closed(tmp_path: Path) -> None:
    selection, snapshot, pile, finance = _selection(tmp_path)
    _evidence(selection, snapshot, pile, finance)
    changed = tmp_path / "changed.arrow"
    changed.write_bytes(b"changed-arrow")
    with pytest.raises(compat.TruthSourceCompatibilityError, match="pile"):
        _evidence(selection, snapshot, changed, finance)
    pile.write_bytes(b"changed-payload")
    with pytest.raises(compat.TruthSourceCompatibilityError, match="pile"):
        _evidence(selection, snapshot, pile, finance)


def test_source_evidence_patch_is_scoped_and_restored(tmp_path: Path) -> None:
    original = truth._source_evidence
    with compat._patched_source_evidence():
        assert truth._source_evidence is compat.source_evidence_compat
    assert truth._source_evidence is original
    with pytest.raises(RuntimeError):
        with compat._patched_source_evidence():
            raise RuntimeError("synthetic source-evidence failure")
    assert truth._source_evidence is original



def test_original_materializer_merge_and_header_validation_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the original merge and header comparison with tiny fixtures."""
    selection, snapshot, pile, finance = _selection(tmp_path)
    rows = {"pile": [{"record_id": "pile-0"}], "finance": [{"record_id": "finance-0"}]}

    class Record:
        def __init__(self, record_id: str) -> None:
            self.record_id = record_id
            self.token_ids = [contract.BOS_TOKEN_ID] + [1] * (contract.STORED_SEQUENCE_TOKENS - 1)

    monkeypatch.setattr(truth.trr6_capture, "_validate_source_descriptors", lambda *args, **kwargs: None)
    monkeypatch.setattr(truth.trusted, "_load_tokenizer", lambda path: object())
    monkeypatch.setattr(truth.trusted, "_load_arrow_dataset", lambda paths: object())
    monkeypatch.setattr(
        truth.trr6_capture,
        "_materialize_selected",
        lambda selected, datasets, tokenizer: {
            "pile": [Record("pile-0")],
            "finance": [Record("finance-0")],
        },
    )
    root = Path(__file__).resolve().parents[1]
    with compat._patched_source_evidence():
        tensors, source_evidence = truth._materialize_truth(
            selection=selection,
            rows=rows,
            root=root,
            tokenizer_path=snapshot,
            pile_paths=(pile,),
            finance_paths=(finance,),
        )
    assert source_evidence["records_by_domain"] == {"pile": 1, "finance": 1}
    assert source_evidence["schema"] == compat.SOURCE_EVIDENCE_SCHEMA

    sidecar_path = tmp_path / "truth-sidecar.safetensors"
    with pytest.raises(RuntimeError, match="share memory"):
        truth.save_file(tensors, str(tmp_path / "raw-shared.safetensors"))
    with compat._patched_truth_serialization():
        truth.save_file(
            tensors,
            str(tmp_path / "alias-safe.safetensors"),
            metadata={"schema": "synthetic"},
        )
    with safe_open(str(tmp_path / "alias-safe.safetensors"), framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(truth.TRUTH_SIDECAR_KEYS)
        assert all(handle.get_tensor(key).equal(tensors[key]) for key in truth.TRUTH_SIDECAR_KEYS)

    metadata = {
        "schema": truth.TRUTH_SIDECAR_SCHEMA,
        "task_id": contract.TASK_ID,
        "truth_opened": "false",
        "registration_sha256": "a" * 64,
        "source_selection_sha256": "b" * 64,
        "capture_receipt_sha256": "c" * 64,
        "observation_manifest_sha256": "d" * 64,
        "observation_record_ids_sha256": json.dumps({"pile": "p" * 64, "finance": "f" * 64}, sort_keys=True, separators=(",", ":")),
        "records_by_domain": json.dumps({"pile": 1, "finance": 1}, sort_keys=True, separators=(",", ":")),
        "sequence_tokens": str(contract.STORED_SEQUENCE_TOKENS),
        "scored_post_bos_tokens": str(contract.SCORED_POST_BOS_TOKENS),
        "target_model_or_target_labels_loaded": "false",
        "source_text_loaded_for_label_materialization": "true",
    }
    save_file({key: value.clone() for key, value in tensors.items()}, str(sidecar_path), metadata=metadata)

    def record(path: Path, *, root: Path, description: str) -> dict[str, object]:
        path = Path(path).resolve()
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    registration_path = tmp_path / "registration.json"
    selection_path = tmp_path / "selection.json"
    observation_path = tmp_path / "observation.json"
    capture_path = tmp_path / "capture.json"
    for path in (registration_path, selection_path, observation_path, capture_path):
        path.write_text("{}", encoding="utf-8")
    registration_record = {"path": str(registration_path), "bytes": 1, "sha256": "a" * 64}
    selection_record = {"path": str(selection_path), "bytes": 1, "sha256": "b" * 64}
    observation_record = {"path": str(observation_path), "bytes": 1, "sha256": "d" * 64}
    capture_record = {"path": str(capture_path), "bytes": 1, "sha256": "c" * 64}
    observation = {
        "cells": {
            cell: {"record_ids_sha256": "p" * 64 if cell.startswith("pile") else "f" * 64}
            for cell in contract.CELL_ORDER
        }
    }
    registration = {
        "source_selection": selection_record,
        "observation_manifest": observation_record,
        "capture_receipt": capture_record,
        "records_by_domain": {"pile": 1, "finance": 1},
        "output_root": str(tmp_path / "prediction-output"),
    }
    freeze = {"registration": registration_record}
    freeze_record = {"path": str(tmp_path / "freeze.json"), "bytes": 1, "sha256": "e" * 64}
    sidecar_record = {"path": str(sidecar_path), "bytes": sidecar_path.stat().st_size, "sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest()}
    header = {
        "schema": truth.TRUTH_BINDING_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": truth.TRUTH_STATUS,
        "truth_opened": False,
        "prepared_after_public_gate": True,
        "registration": registration_record,
        "receipt": freeze_record,
        "source_selection": selection_record,
        "observation_manifest": observation_record,
        "capture_receipt": capture_record,
        "records_by_domain": {"pile": 1, "finance": 1},
        "cell_order": list(contract.CELL_ORDER),
        "target_conditions": list(contract.TARGET_ORDER),
        "cells": [{"cell_id": cell} for cell in contract.CELL_ORDER],
        "sidecar": sidecar_record,
        "source_evidence": source_evidence,
    }
    header_path = tmp_path / "truth-binding.json"
    header_path.write_text(json.dumps(header), encoding="utf-8")
    original_load_json = truth._load_json

    def load_json(path: Path, *, description: str) -> dict[str, object]:
        resolved = Path(path).resolve()
        if resolved == observation_path.resolve():
            return observation
        if resolved == selection_path.resolve():
            return selection
        return original_load_json(path, description=description)

    monkeypatch.setattr(truth, "_load_json", load_json)
    monkeypatch.setattr(truth, "_record", record)
    monkeypatch.setattr(truth, "_tokenizer_path", lambda selection, root: snapshot)
    monkeypatch.setattr(truth, "_source_paths", lambda selection, root, style: (pile,) if style == "pile" else (finance,))
    with compat._patched_source_evidence():
        checked_header, checked_sidecar = truth._validate_sidecar_header(
            freeze=freeze,
            registration=registration,
            freeze_record=freeze_record,
            header_path=header_path,
            repository_root=root,
            open_tensor_data=False,
        )
    assert checked_header["source_evidence"] == source_evidence
    assert checked_sidecar["header"]["path"] == str(header_path.resolve())
    assert checked_sidecar["sidecar"]["path"] == str(sidecar_path.resolve())
