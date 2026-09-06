from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.trr_p04 import freeze_predictions as freezer


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments" / "TRR-P04" / "runtime"
PANEL = REPO / "experiments" / "TRR-P04" / "setup" / "public_selection-r2.json"
STUDENT_FREEZE = RUNTIME / "student-predictions-r3" / "student_prediction_freeze.json"
OBSERVATION_INDEX = RUNTIME / "evaluator-observations-r1" / "observation_index.json"
STATE_MANIFEST = RUNTIME / "training-r1" / "selected_state_manifest-r3.json"
ANCHOR_RECEIPTS = [
    RUNTIME / "native-anchor-public-r2" / "native_anchor_receipt.json",
    RUNTIME / "native-anchor-target-r2" / "native_anchor_receipt.json",
]


def _inputs() -> dict[str, Any]:
    required = [PANEL, STUDENT_FREEZE, OBSERVATION_INDEX, STATE_MANIFEST, *ANCHOR_RECEIPTS]
    if not all(path.is_file() for path in required):
        pytest.skip("the frozen public no-truth P04 runtime receipts are unavailable")
    student = json.loads(STUDENT_FREEZE.read_text(encoding="utf-8"))
    student_paths = [Path(row["path"]) for row in student["prediction_files"]]
    anchor_paths = [
        Path(json.loads(path.read_text(encoding="utf-8"))["prediction"]["path"])
        for path in ANCHOR_RECEIPTS
    ]
    return {
        "panel_path": PANEL,
        "prediction_paths": student_paths,
        "anchor_prediction_paths": anchor_paths,
        "state_manifest_path": STATE_MANIFEST,
        "truth_dir": RUNTIME / "truth-never-opened",
        "student_prediction_freeze_path": STUDENT_FREEZE,
        "observation_index_path": OBSERVATION_INDEX,
        "native_anchor_receipt_paths": ANCHOR_RECEIPTS,
    }


def _build(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    kwargs = _inputs()
    kwargs.update(overrides)
    kwargs.update(output_path=tmp_path / "joint-freeze.json", argv=["pytest", "joint-freeze"])
    return freezer.build_freeze(**kwargs)


def test_joint_freeze_binds_actual_public_matrix_without_truth(tmp_path: Path) -> None:
    frozen = _build(tmp_path)

    assert frozen["status"] == "FROZEN_BEFORE_TRUTH"
    assert frozen["truth_accessed"] is False
    assert frozen["provenance"]["status"] == "JOINT_PUBLIC_BINDINGS_VALIDATED_BEFORE_TRUTH"
    assert frozen["provenance"]["student_cell_receipt_count"] == 16
    assert frozen["provenance"]["native_anchor_receipt_count"] == 2
    assert frozen["observation_index"]["record_count"] == 72
    assert frozen["observation_index"]["conditions"] == [
        "public_base",
        "p04_evaluator_target_update_v1",
    ]
    assert (tmp_path / "joint-freeze.json").is_file()


def test_joint_freeze_rejects_missing_native_anchor_arm_before_output(tmp_path: Path) -> None:
    kwargs = _inputs()
    kwargs["native_anchor_receipt_paths"] = kwargs["native_anchor_receipt_paths"][:1]
    with pytest.raises(freezer.FreezeError, match="one native anchor receipt"):
        freezer.build_freeze(
            **kwargs,
            output_path=tmp_path / "joint-freeze.json",
            argv=["pytest", "missing-anchor"],
        )
    assert not (tmp_path / "joint-freeze.json").exists()


def test_joint_freeze_rejects_observation_geometry_mismatch_before_output(tmp_path: Path) -> None:
    kwargs = _inputs()
    mutated = tmp_path / "student_prediction_freeze-mutated.json"
    student = json.loads(STUDENT_FREEZE.read_text(encoding="utf-8"))
    student["observations"]["p04_evaluator_target_update_v1"]["attention_mask_sha256"] = "0" * 64
    mutated.write_text(json.dumps(student) + "\n", encoding="utf-8")

    kwargs["student_prediction_freeze_path"] = mutated
    with pytest.raises(freezer.FreezeError, match="attention_mask_sha256|mask/position geometry|digest"):
        freezer.build_freeze(
            **kwargs,
            output_path=tmp_path / "joint-freeze.json",
            argv=["pytest", "geometry-mismatch"],
        )
    assert not (tmp_path / "joint-freeze.json").exists()


def test_joint_freeze_rejects_missing_student_method_before_output(tmp_path: Path) -> None:
    kwargs = _inputs()
    kwargs["prediction_paths"] = kwargs["prediction_paths"][:-1]

    with pytest.raises(freezer.FreezeError, match="prediction group is missing"):
        freezer.build_freeze(
            **kwargs,
            output_path=tmp_path / "joint-freeze.json",
            argv=["pytest", "missing-method"],
        )
    assert not (tmp_path / "joint-freeze.json").exists()
