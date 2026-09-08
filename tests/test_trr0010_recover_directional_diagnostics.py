"""CPU-only tests for retained TRR-0010 diagnostic recovery planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.trr0010_recover_directional_diagnostics import (
    CHECKPOINT_STEPS,
    RecoveryError,
    build_recovery_plan,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw_receipt(tmp_path: Path) -> tuple[dict[str, object], Path]:
    bindings = []
    curve = []
    selected_state = None
    for step in CHECKPOINT_STEPS:
        payload = f"checkpoint-{step}".encode("ascii")
        checkpoint_path = tmp_path / f"checkpoint_{step}.bin"
        checkpoint_path.write_bytes(payload)
        state_digest = _sha(f"state-{step}".encode("ascii"))
        metadata = {
            "schema": "token-reconstruction.trr0010-directional-readout.v1",
            "method_id": "trr0010_supported_token_directional_readout",
            "selected_step": step,
            "serialization_only": True,
            "optimizer_state_external": True,
            "runner_point_state_sha256": state_digest,
            "current_H_only": True,
            "full_vocabulary_cross_entropy": True,
            "p09_contract_sha256": "a" * 64,
            "p09_bank_sha256": "b" * 64,
            "p09_schedule_sha256": "c" * 64,
            "support_count": 17,
            "support_digest": "d" * 64,
        }
        descriptor = {
            "path": str(checkpoint_path),
            "bytes": len(payload),
            "sha256": _sha(payload),
            "state_bytes": 4,
            "state_tensor_digest": state_digest,
            "metadata": metadata,
        }
        bindings.append(
            {
                "checkpoint": descriptor,
                "optimizer_state_external": True,
                "resume_requires_separate_optimizer_artifact": True,
            }
        )
        if step == 0:
            selected_state = state_digest
        curve.append(
            {
                "step": step,
                "validation": {
                    "domain_balanced_token_accuracy": 0.9 if step == 0 else 0.8,
                    "correct_tokens": 9,
                    "token_rows": 10,
                    "label_join_sha256": "e" * 64,
                },
            }
        )
    raw = {
        "schema": "token-reconstruction.trr-p09-fixed-control-run.v1",
        "status": "COMPLETED",
        "checkpoints": list(CHECKPOINT_STEPS),
        "selected_step": 0,
        "selected_state_sha256": selected_state,
        "checkpoint_state_bindings": bindings,
        "learning_curve": curve,
        "schedule": {
            "steps": 13000,
            "seed": 4010,
            "semantic_sha256": "f" * 64,
            "exposure": {"draws_per_step": 512},
        },
    }
    raw_path = tmp_path / "raw_runner_result.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    return raw, raw_path


def test_plan_resolves_nested_selected_checkpoint_and_endpoints(tmp_path: Path) -> None:
    raw, raw_path = _raw_receipt(tmp_path)
    plan = build_recovery_plan(raw, raw_path=raw_path, arm_name="current_directional")
    assert plan["status"] == "PLAN_ONLY"
    assert plan["completed_fit"]["selected_step"] == 0
    assert plan["completed_fit"]["selected_checkpoint"]["metadata_selected_step"] == 0
    assert plan["completed_fit"]["runner_point_state_sha256"] == raw["selected_state_sha256"]
    assert plan["recovery_scopes"]["frozen64"]["steps"] == list(CHECKPOINT_STEPS)
    assert plan["recovery_scopes"]["full_bank"]["steps"] == [0, 13000]
    assert plan["recovery_policy"]["new_fit_updates"] is False


def test_plan_rejects_top_level_only_selected_step(tmp_path: Path) -> None:
    raw, raw_path = _raw_receipt(tmp_path)
    raw["checkpoint_state_bindings"][0]["checkpoint"]["metadata"].pop("selected_step")
    with pytest.raises(RecoveryError, match="metadata.selected_step"):
        build_recovery_plan(raw, raw_path=raw_path, arm_name="current_directional")


def test_plan_can_verify_checkpoint_bytes_and_sha(tmp_path: Path) -> None:
    raw, raw_path = _raw_receipt(tmp_path)
    plan = build_recovery_plan(raw, raw_path=raw_path, arm_name="expanded_directional", verify_files=True)
    assert plan["bank_role"] == "B1"
    assert all(item["actual_sha256"] == item["sha256"] for item in plan["checkpoints"])
