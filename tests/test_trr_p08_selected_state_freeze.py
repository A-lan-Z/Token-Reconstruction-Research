from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.trr_p08 import freeze_matrix
from scripts.trr_p08 import freeze_selected_states as selected


def _record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_selected_state_freeze_binds_eight_states_without_payload_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    root.mkdir()
    plan_record = {"path": "experiments/TRR-P08/plan.json", "bytes": 1, "sha256": "a" * 64}
    qualification_record = {"path": "runtime/qualification.json", "bytes": 1, "sha256": "b" * 64}
    fit_record = {"path": "runtime/main_fit_receipt.json", "bytes": 1, "sha256": "c" * 64}
    states = {}
    for seed in freeze_matrix.SEEDS:
        for method in freeze_matrix.METHOD_ORDER:
            path = root / f"runtime/{seed}-{method}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"state-{seed}-{method}".encode())
            record = _record(path)
            states[(seed, method)] = {
                "seed": seed,
                "arm_id": method,
                "selected_step": 100,
                "schedule_sha256": "d" * 64,
                "selected": record,
                "selected_tensor_sha256": "e" * 64,
            }

    monkeypatch.setattr(selected.freeze_matrix, "_validate_plan", lambda *args, **kwargs: plan_record)
    monkeypatch.setattr(
        selected.freeze_matrix,
        "_validate_fit_receipt",
        lambda *args, **kwargs: {
            "source_commit": "f" * 40,
            "receipt_record": fit_record,
            "qualification": qualification_record,
            "states": states,
        },
    )
    output = root / "experiments/TRR-P08/runtime/selected-state-freeze.json"
    receipt = selected.assemble_selected_state_freeze(
        repository_root=root,
        plan_path=root / "plan.json",
        fit_receipt_path=root / "fit.json",
        output_path=output,
        expected_plan_sha256="a" * 64,
        expected_source_commit="f" * 40,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema"] == selected.SCHEMA
    assert saved["status"] == selected.STATUS
    assert saved["truth_opened"] is False
    assert saved["state_payloads_loaded"] is False
    assert saved["selected_state_count"] == 8
    assert set(saved["selected_states"]) == {f"{seed}::{method}" for seed in freeze_matrix.SEEDS for method in freeze_matrix.METHOD_ORDER}
    assert saved["source_commit"] == "f" * 40
