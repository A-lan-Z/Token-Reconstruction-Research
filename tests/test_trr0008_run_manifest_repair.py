from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import trr0008_run_manifest_repair as repair


def _write_json(path: Path, value: dict[str, object]) -> bytes:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return payload.encode("utf-8")


def test_metadata_completion_preserves_original_and_adds_only_registration_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task_root = tmp_path / "experiments" / "TRR-0008"
    registration_path = task_root / "evaluation" / "registration_v1.json"
    registration_bytes = _write_json(
        registration_path,
        {
            "schema": "token-reconstruction.trr0008-frozen-evaluation-registration.v1",
            "task_id": "TRR-0008",
            "truth_opened": False,
            "source_text_or_target_labels": False,
            "candidate_arrays_persisted": False,
        },
    )
    registration_sha = hashlib.sha256(registration_bytes).hexdigest()
    original_path = task_root / "evaluation" / "predictions_v1" / "run_manifest.json"
    original_bytes = _write_json(
        original_path,
        {
            "schema": repair.RUN_SCHEMA,
            "task_id": "TRR-0008",
            "registration": {
                "path": str(registration_path),
                "sha256": registration_sha,
            },
            "truth_opened": False,
            "candidate_arrays_persisted": False,
            "predictions": {"synthetic": {"sha256": "p" * 64}},
            "timings": {"synthetic": {"truth_opened": False}},
        },
    )
    before = original_path.read_bytes()
    output_path = original_path.with_name("run_manifest.metadata_completed.json")
    receipt_path = original_path.with_name("run_manifest.metadata_completed.receipt.json")
    message = "TRR-0008 gate failed closed: run registration binding changed: bytes"
    monkeypatch.setattr(repair, "_git_head", lambda root: "a" * 40)

    result = repair.complete_metadata(
        repository_root=tmp_path,
        original_run_manifest=original_path,
        registration=registration_path,
        output=output_path,
        receipt=receipt_path,
        gate_failure=message,
    )

    assert original_path.read_bytes() == before == original_bytes
    completed = json.loads(output_path.read_text(encoding="utf-8"))
    assert completed["registration"] == {
        "path": str(registration_path.resolve()),
        "sha256": registration_sha,
        "bytes": len(registration_bytes),
    }
    assert result["semantic_diff"] == [
        {
            "path": "registration.bytes",
            "operation": "add",
            "before": "<missing>",
            "after": len(registration_bytes),
        }
    ]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["gate_failure"]["message"] == message
    assert receipt["gate_failure"]["reason"] == repair.EXPECTED_GATE_REASON
    assert receipt["original_run_manifest"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert receipt["metadata_completed_run_manifest"]["sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert receipt["helper_code"]["sha256"] == hashlib.sha256(Path(repair.__file__).read_bytes()).hexdigest()
    assert receipt["truth_opened"] is False
    assert receipt["execution"]["code_commit"] == "a" * 40


def test_metadata_completion_rejects_nonminimal_registration_binding(tmp_path: Path) -> None:
    task_root = tmp_path / "experiments" / "TRR-0008"
    registration_path = task_root / "registration.json"
    registration_bytes = _write_json(
        registration_path,
        {"task_id": "TRR-0008", "truth_opened": False},
    )
    registration_sha = hashlib.sha256(registration_bytes).hexdigest()
    original_path = task_root / "run_manifest.json"
    _write_json(
        original_path,
        {
            "schema": repair.RUN_SCHEMA,
            "task_id": "TRR-0008",
            "registration": {
                "path": str(registration_path),
                "sha256": registration_sha,
                "bytes": 1,
            },
            "truth_opened": False,
            "candidate_arrays_persisted": False,
        },
    )

    with pytest.raises(repair.RepairError, match="exactly path and sha256"):
        repair.complete_metadata(
            repository_root=tmp_path,
            original_run_manifest=original_path,
            registration=registration_path,
            output=task_root / "completed.json",
            receipt=task_root / "receipt.json",
            gate_failure="TRR-0008 gate failed closed: run registration binding changed: bytes",
        )
