"""Synthetic fail-closed tests for the TRR-0010 registration adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_register as register


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_draft_design_is_rejected_before_selection_or_capture(tmp_path: Path) -> None:
    design = _write(
        tmp_path / "design.json",
        {
            "schema": register.DESIGN_SCHEMA,
            "task_id": gate.TASK_ID,
            "status": "PROPOSAL_V4_READY_FOR_ROOT_REVIEW",
            "code_commit": "a" * 40,
        },
    )
    with pytest.raises(register.RegisterError, match="owner-frozen"):
        register._require_design(design, root=tmp_path)


def test_frozen_design_requires_all_six_contenders(tmp_path: Path) -> None:
    design = _write(
        tmp_path / "design.json",
        {
            "schema": register.DESIGN_SCHEMA,
            "task_id": gate.TASK_ID,
            "status": register.DESIGN_STATUS,
            "code_commit": "a" * 40,
            "methods": {
                method_id: gate.METHOD_ROLES[method_id]
                for method_id in gate.METHOD_ORDER[:-1]
            },
        },
    )
    with pytest.raises(register.RegisterError, match="all six contender"):
        register._require_design(design, root=tmp_path)


def test_producer_wiring_constants_are_inherited_geometry() -> None:
    assert register.gate.RECORDS_BY_DOMAIN == {"finance": 128, "pile": 128}
    assert register.gate.CELL_ORDER == (
        "pile__public_base",
        "pile__public_lora_2601",
        "finance__public_base",
        "finance__public_lora_2601",
    )
    assert register.gate.STORED_SEQUENCE_TOKENS == 128
    assert register.gate.OBSERVATION_HIDDEN_SIZE == 2048


def test_frequency_reference_binding_freezes_both_named_banks(tmp_path: Path) -> None:
    frequency_path = _write(
        tmp_path / "frequency.json",
        {
            "schema": "token-reconstruction.trr0010-frequency-reference.v1",
            "task_id": gate.TASK_ID,
            "frequency_references": {
                "B0": {"0": 3, "7": 2},
                "B1": {"0": 30, "7": 20, "9": 1},
            },
            "truth_opened": False,
        },
    )
    bindings, metadata = register._frequency_reference_bindings(
        root=tmp_path,
        frequency_reference_path=frequency_path,
        frequency_reference_paths=None,
    )
    assert set(bindings) == {"frequency_reference_B0", "frequency_reference_B1"}
    assert metadata["bank_order"] == ["B0", "B1"]
    assert metadata["all_methods_each_bank"] is True
    assert metadata["banks"]["B0"]["support_token_count"] == 2
    assert metadata["banks"]["B1"]["support_token_count"] == 3


def test_frequency_reference_binding_rejects_missing_bank(tmp_path: Path) -> None:
    frequency_path = _write(
        tmp_path / "frequency.json",
        {
            "frequency_references": {"B0": {"0": 1}},
            "truth_opened": False,
        },
    )
    with pytest.raises(register.RegisterError, match="named B1"):
        register._frequency_reference_bindings(
            root=tmp_path,
            frequency_reference_path=frequency_path,
            frequency_reference_paths=None,
        )
