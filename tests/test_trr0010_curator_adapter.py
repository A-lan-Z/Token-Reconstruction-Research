from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_PATH = _ROOT / "experiments" / "TRR-0010" / "setup" / "trr0010_curator_adapter.py"
_SPEC = importlib.util.spec_from_file_location("trr0010_curator_adapter", _ADAPTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapter)


def _record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _fixture(tmp_path: Path) -> tuple[dict, dict]:
    ledger_path = tmp_path / "selection.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema": "token-reconstruction.trr0010-source-selection.v1",
                "task_id": "TRR-0010",
                "status": "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH",
                "truth_opened": False,
                "truth_created": False,
                "records_by_domain": {"finance": 128, "pile": 128},
                "target_conditions": ["public_base", "public_lora_2601"],
            }
        )
        + "\n"
    )
    wrapper_path = tmp_path / "source_selection_binding.json"
    wrapper_path.write_text(
        json.dumps(
            {
                "schema": "token-reconstruction.trr0010-source-selection-binding.v1",
                "task_id": "TRR-0010",
                "status": "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH",
                "source_text_loaded": False,
                "target_labels_loaded": False,
                "truth_opened": False,
                "selection_ledger": _record(ledger_path),
                "selection_ledger_schema": "token-reconstruction.trr0010-source-selection.v1",
                "selection_ledger_status": "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH",
            }
        )
        + "\n"
    )
    freeze = {"input_bindings": {"source_selection": _record(wrapper_path)}}
    return freeze, _record(ledger_path)


def test_selection_binding_chain_validates_wrapper_and_nested_ledger(tmp_path: Path) -> None:
    freeze, ledger_record = _fixture(tmp_path)
    checked = adapter.validate_selection_binding_chain(freeze, repository_root=tmp_path)
    assert checked["ledger_record"] == ledger_record
    assert checked["wrapper"]["selection_ledger"]["sha256"] == ledger_record["sha256"]


def test_delegate_freeze_changes_only_source_selection_binding(tmp_path: Path) -> None:
    freeze, ledger_record = _fixture(tmp_path)
    checked = adapter.validate_selection_binding_chain(freeze, repository_root=tmp_path)
    delegated = adapter.build_delegate_freeze(freeze, checked)
    assert freeze["input_bindings"]["source_selection"] != ledger_record
    assert delegated["input_bindings"]["source_selection"] == ledger_record
    assert freeze["input_bindings"]["source_selection"]["bytes"] != ledger_record["bytes"] or freeze["input_bindings"]["source_selection"]["sha256"] != ledger_record["sha256"]


def test_selection_binding_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    freeze, _ = _fixture(tmp_path)
    freeze["input_bindings"]["source_selection"]["sha256"] = "0" * 64
    with pytest.raises(adapter.CuratorAdapterError):
        adapter.validate_selection_binding_chain(freeze, repository_root=tmp_path)
