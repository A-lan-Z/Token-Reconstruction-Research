from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.trr_p04 import materialize_evaluator_truth as materializer


def _rows() -> list[dict[str, object]]:
    return [
        {
            "record_id": f"{style}-l{length}-r{offset}",
            "style": style,
            "length_stratum": length,
            "anchor": length == 32 and offset < 4,
        }
        for style in ("pile_plain", "finance_chat", "alpaca_instruction")
        for length in (16, 32, 64, 128)
        for offset in range(6)
    ]


def _sequences(rows: list[dict[str, object]]) -> list[list[int]]:
    return [
        [materializer.evaluator.BOS_TOKEN_ID]
        + [index + 1] * int(row["length_stratum"])
        for index, row in enumerate(rows)
    ]


def _freeze(path: Path, selection_path: Path, truth_dir: Path, *, status: str = "FROZEN_BEFORE_TRUTH") -> None:
    selection_sha = materializer._sha256_file(selection_path)
    path.write_text(
        json.dumps(
            {
                "schema": materializer.FREEZE_SCHEMA,
                "task_id": materializer.TASK_ID,
                "status": status,
                "panel_frozen": True,
                "predictions_frozen": True,
                "all_states_frozen": True,
                "truth_open_allowed": True,
                "truth_accessed": False,
                "panel": {"path": str(selection_path), "sha256": selection_sha},
                "truth_files": [
                    {
                        "condition": condition,
                        "path": str(truth_dir / f"{condition}.jsonl"),
                    }
                    for condition in materializer.CONDITIONS
                ],
                "truth_boundary": {
                    "prediction_and_panel_validation_completed_before_truth": True,
                    "truth_rows_not_loaded_by_freezer": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_truth_rows_strip_bos_and_preserve_frozen_record_order() -> None:
    rows = _rows()
    result = materializer._truth_rows(rows, _sequences(rows))

    assert len(result) == 72
    assert [row["record_id"] for row in result] == [row["record_id"] for row in rows]
    assert all(set(row) == {"record_id", "token_ids"} for row in result)
    assert all(len(row["token_ids"]) == int(rows[index]["length_stratum"]) for index, row in enumerate(result))
    assert all(tokens and materializer.evaluator.BOS_TOKEN_ID not in tokens for tokens in (row["token_ids"] for row in result))


def test_truth_files_are_separate_and_create_only(tmp_path: Path) -> None:
    rows = _rows()
    truth_dir = tmp_path / "private-truth"
    descriptors = materializer._write_truth_files(truth_dir, rows=rows, sequences=_sequences(rows))

    assert set(descriptors) == set(materializer.CONDITIONS)
    public_lines = (truth_dir / "public_base.jsonl").read_text(encoding="utf-8").splitlines()
    target_lines = (truth_dir / "p04_evaluator_target_update_v1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(public_lines) == len(target_lines) == 72
    assert public_lines == target_lines
    assert all(set(json.loads(line)) == {"record_id", "token_ids"} for line in public_lines)
    with pytest.raises(materializer.TruthMaterializationError, match="new empty directory"):
        materializer._write_truth_files(truth_dir, rows=rows, sequences=_sequences(rows))


def test_authorization_rejects_unfrozen_receipt_before_panel_hash(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    _freeze(freeze, selection, tmp_path / "truth", status="PENDING")
    with pytest.raises(materializer.TruthMaterializationError, match="FROZEN_BEFORE_TRUTH"):
        materializer._validate_freeze_authorization(
            freeze,
            selection_path=selection,
            truth_dir=tmp_path / "truth",
        )


def test_authorization_rejects_repository_truth_output(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    truth_dir = materializer.REPOSITORY_ROOT / "experiments" / "TRR-P04" / "private" / "test-truth-output"
    _freeze(freeze, selection, truth_dir)
    with pytest.raises(materializer.TruthMaterializationError, match="outside the repository"):
        materializer._validate_freeze_authorization(
            freeze,
            selection_path=selection,
            truth_dir=truth_dir,
        )
