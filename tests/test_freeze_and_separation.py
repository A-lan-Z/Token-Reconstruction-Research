from __future__ import annotations

import json
from pathlib import Path

import pytest

from token_reconstruction.freeze import (
    FreezeError,
    create_freeze_receipt,
    require_truth_open_allowed,
    verify_freeze_receipt,
)
from token_reconstruction.separation import (
    ReconstructionInputs,
    SeparationError,
    reconstruction_input_fields,
)


PREREGISTRATION_COMMIT = "a" * 40


def _fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    plan = root / "plan.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text('{"truth_opened":false}\n', encoding="utf-8")
    frozen = root / "bundle"
    frozen.mkdir()
    (frozen / "outputs.jsonl").write_text('{"prediction":[1]}\n', encoding="utf-8")
    receipt = root / "receipt.json"
    truth = root / "private" / "answers.json"
    truth.parent.mkdir()
    truth.write_text('{"tokens":[1]}\n', encoding="utf-8")
    return plan, frozen, receipt, truth


def test_reconstruction_interface_has_no_private_fields(tmp_path: Path) -> None:
    plan, frozen, _, _ = _fixture(tmp_path)
    index = tmp_path / "observations.json"
    index.write_text("{}\n", encoding="utf-8")
    inverse = tmp_path / "inverse"
    inverse.mkdir()
    inputs = ReconstructionInputs(
        observation_index=index,
        inverse_directory=inverse,
        plan_path=plan,
        output_directory=tmp_path / "new-output",
        model_id="public/model",
        model_revision="b" * 40,
    )
    inputs.validate()
    assert reconstruction_input_fields() == (
        "observation_index",
        "inverse_directory",
        "plan_path",
        "output_directory",
        "model_id",
        "model_revision",
    )
    assert not any("truth" in name or "token" in name for name in reconstruction_input_fields())

    with pytest.raises(SeparationError, match="evaluator-private"):
        ReconstructionInputs(
            observation_index=tmp_path / "truth" / "observations.json",
            inverse_directory=inverse,
            plan_path=plan,
            output_directory=tmp_path / "out",
            model_id="public/model",
            model_revision="b" * 40,
        ).validate()


def test_truth_gate_fails_before_freeze_and_after_mutation(tmp_path: Path) -> None:
    plan, frozen, receipt, truth = _fixture(tmp_path)
    with pytest.raises(FreezeError, match="freeze receipt"):
        require_truth_open_allowed(
            receipt_path=receipt, repository_root=tmp_path, truth_path=truth
        )

    create_freeze_receipt(
        repository_root=tmp_path,
        frozen_root=frozen,
        plan_path=plan,
        receipt_path=receipt,
        preregistration_commit=PREREGISTRATION_COMMIT,
        created_utc="2026-08-22T00:00:00Z",
    )
    require_truth_open_allowed(
        receipt_path=receipt, repository_root=tmp_path, truth_path=truth
    )

    output = frozen / "outputs.jsonl"
    output.chmod(0o644)
    output.write_text('{"prediction":[2]}\n', encoding="utf-8")
    with pytest.raises(FreezeError, match="hash changed"):
        verify_freeze_receipt(receipt, repository_root=tmp_path)


def test_receipt_is_deterministic_for_equal_bytes(tmp_path: Path) -> None:
    receipts = []
    for name in ("one", "two"):
        root = tmp_path / name
        plan, frozen, receipt, _ = _fixture(root)
        payload = create_freeze_receipt(
            repository_root=root,
            frozen_root=frozen,
            plan_path=plan,
            receipt_path=receipt,
            preregistration_commit=PREREGISTRATION_COMMIT,
            created_utc="2026-08-22T00:00:00Z",
            metadata={"record_order_sha256": "c" * 64},
        )
        receipts.append(payload)
    assert receipts[0] == receipts[1]
    assert json.dumps(receipts[0], sort_keys=True) == json.dumps(receipts[1], sort_keys=True)



def test_public_unavailable_target_condition_is_not_private_path(
    tmp_path: Path,
) -> None:
    plan, frozen, receipt, _ = _fixture(tmp_path)
    observation = frozen / "unavailable_target_lora_cut0.safetensors"
    observation.write_bytes(b"public activation observation")
    payload = create_freeze_receipt(
        repository_root=tmp_path,
        frozen_root=frozen,
        plan_path=plan,
        receipt_path=receipt,
        preregistration_commit=PREREGISTRATION_COMMIT,
        created_utc="2026-08-22T00:00:00Z",
    )
    assert any(
        entry["path"].endswith("unavailable_target_lora_cut0.safetensors")
        for entry in payload["entries"]
    )


def test_private_target_lora_path_is_rejected(tmp_path: Path) -> None:
    plan, frozen, receipt, _ = _fixture(tmp_path)
    private = frozen / "target_lora.safetensors"
    private.write_bytes(b"private target")
    with pytest.raises(FreezeError, match="prohibited private artifact"):
        create_freeze_receipt(
            repository_root=tmp_path,
            frozen_root=frozen,
            plan_path=plan,
            receipt_path=receipt,
            preregistration_commit=PREREGISTRATION_COMMIT,
            created_utc="2026-08-22T00:00:00Z",
        )
