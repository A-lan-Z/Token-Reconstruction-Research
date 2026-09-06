from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import trr0006_select_public as selector


def _plan(*, status: str = "FROZEN_BEFORE_NEW_SOURCE_SELECTION") -> dict[str, object]:
    return {
        "schema": "token-reconstruction.trr0006-decision-plan.v1",
        "task_id": "TRR-0006",
        "status": status,
        "sample_size_frozen": True,
        "panel": {
            "records_per_domain": 1536,
            "clip_tokens_including_bos": 128,
            "capture_batch_records": 8,
            "capture_tokens": 192,
            "selection_seed": 5005,
            "source_ranges_half_open": {"pile": [7000, 10000], "finance": [12000, 20000]},
        },
        "comparison": {
            "target_conditions": ["public_base", "public_lora_2601"],
        },
        "method_freeze_sha256": "a" * 64,
        "truth_gate": {
            "truth_status": "NOT_OPENED",
            "required_before_truth": "complete public matrix gate",
        },
    }


def test_frozen_plan_accepts_control_metadata_without_source_payload(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(_plan()) + "\n", encoding="utf-8")

    result = selector._validate_frozen_plan(path)

    assert result["status"] == "FROZEN_BEFORE_NEW_SOURCE_SELECTION"
    assert result["records_per_domain"] == 1536
    assert result["method_freeze_sha256"] == "a" * 64


def test_frozen_plan_reads_method_binding_from_provenance(tmp_path: Path) -> None:
    value = _plan()
    value.pop("method_freeze_sha256")
    value["provenance"] = {"trr5_method_freeze_sha256": "b" * 64}
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    result = selector._validate_frozen_plan(path)

    assert result["method_freeze_sha256"] == "b" * 64


def test_wrong_nested_population_metadata_is_rejected(tmp_path: Path) -> None:
    value = _plan()
    value["panel"]["source_ranges_half_open"] = {"pile": [0, 3000], "finance": [12000, 20000]}
    path = tmp_path / "wrong-range.json"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(selector.SelectionError, match="source ranges"):
        selector._validate_frozen_plan(path)


def test_draft_plan_is_rejected_before_source_selection(tmp_path: Path) -> None:
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(_plan(status="READY_FOR_ROOT_FREEZE")) + "\n", encoding="utf-8")

    with pytest.raises(selector.SelectionError, match="frozen plan"):
        selector._validate_frozen_plan(path)


def test_frozen_plan_rejects_source_or_truth_payload_fields(tmp_path: Path) -> None:
    for key in ("source_text", "record_id", "truth_file"):
        value = _plan()
        value[key] = "payload"
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

        with pytest.raises(selector.SelectionError, match="payload"):
            selector._validate_frozen_plan(path)
