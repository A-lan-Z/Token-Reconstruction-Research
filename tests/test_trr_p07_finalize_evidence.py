from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.trr_p07 import finalize_evidence as finalizer


def _record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_selected_rows_summary_binds_both_frozen_subset_rules() -> None:
    p06 = finalizer._selected_rows_summary("p06_panel", list(range(256)))
    old = finalizer._selected_rows_summary("trr0006_subset", list(range(0, 1536, 6)))

    assert p06["rule"] == "contiguous_zero_through_255"
    assert old["rule"] == "published_trr0006_rows_6k_k0_through_255"
    assert p06["count"] == old["count"] == 256
    assert p06["last"] == 255
    assert old["last"] == 1530


def test_selected_rows_summary_rejects_contiguous_old_subset() -> None:
    with pytest.raises(finalizer.FinalizationError, match="selected-row rule changed"):
        finalizer._selected_rows_summary("trr0006_subset", list(range(256)))


def test_score_validation_accepts_registered_disposition(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    score_path.write_text(
        json.dumps(
            {
                "schema": "token-reconstruction.trr-p07-score.v1",
                "task_id": "TRR-P07",
                "status": "TRR-P07_SCORED_AFTER_PREDICTION_FREEZE",
                "truth_opened": True,
                "truth_payload_persisted": False,
                "prediction_freeze": {"sha256": "a" * 64},
                "gate": {"disposition": "PANEL_DEPENDENT_OR_UNCERTAIN"},
            }
        ),
        encoding="utf-8",
    )

    score, record = finalizer._validate_score(tmp_path, score_path, {"sha256": "a" * 64})

    assert score["gate"]["disposition"] == "PANEL_DEPENDENT_OR_UNCERTAIN"
    assert record["sha256"] == hashlib.sha256(score_path.read_bytes()).hexdigest()


def test_score_execution_receipt_binds_output_bytes_and_hash(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    output.write_text("{}\n", encoding="utf-8")
    score_record = finalizer._actual_record(tmp_path, output, description="synthetic score")
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(
        json.dumps(
            {
                "task_id": "TRR-P07",
                "status": "COMPLETE_RETROSPECTIVE_SCORED_AFTER_FREEZE",
                "code_commit": "b" * 40,
                "output": score_record,
                "exit_code": 0,
                "elapsed_seconds": None,
                "timing_note": "Scoring elapsed time was not instrumented.",
            }
        ),
        encoding="utf-8",
    )

    execution, _ = finalizer._validate_score_execution(tmp_path, execution_path, score_record)

    assert execution["output"]["sha256"] == score_record["sha256"]


def test_task_path_guard_rejects_global_state_location(tmp_path: Path) -> None:
    with pytest.raises(finalizer.FinalizationError, match="under experiments/TRR-P07"):
        finalizer._require_task_path(tmp_path, tmp_path / "coordination" / "STATE.json", description="P07 manifest")


def test_parser_defaults_to_authoritative_scored_r2() -> None:
    args = finalizer._parser().parse_args([])
    assert args.score.as_posix().endswith("experiments/TRR-P07/runtime/scored-r2/results.json")
    assert args.score_execution.as_posix().endswith("experiments/TRR-P07/runtime/scored-r2/execution.json")
