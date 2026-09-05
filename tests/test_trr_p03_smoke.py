"""Model-free TRR-P03 contract and freeze-to-score smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from shutil import copytree

import pytest
import torch
from safetensors.torch import save_file

from token_reconstruction.trr_p03.io import (
    BOS_TOKEN_ID,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    OBSERVATION_INDEX_SCHEMA,
    PREDICTION_SCHEMA,
    file_record,
    freeze_prediction_bundle,
    load_index_and_observations,
    save_observation_bundle,
    sha256_file,
    validate_observation_index,
    write_freeze_receipt,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from token_reconstruction.trr_p03.ranking import rank_queries
from token_reconstruction.trr_p03 import scoring as scoring_module
from token_reconstruction.trr_p03.scoring import ScoringError, score_prediction_bundle


def test_chunked_ranking_resolves_ties_across_candidate_chunks() -> None:
    prototypes = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
    )
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    split = rank_queries(queries, prototypes, query_chunk_size=1, prototype_chunk_size=2)
    whole = rank_queries(queries, prototypes, query_chunk_size=2, prototype_chunk_size=5)

    assert split.top1_ids.tolist() == [0, 1]
    assert split.runner_up_ids.tolist() == [3, 0]
    assert split.top1_tie_count.tolist() == [2, 1]
    assert split.margins.tolist() == pytest.approx([0.0, 1.0])
    assert torch.equal(split.top1_ids, whole.top1_ids)
    assert torch.equal(split.runner_up_ids, whole.runner_up_ids)
    assert torch.equal(split.top1_tie_count, whole.top1_tie_count)


def test_grouped_observation_roundtrip_exposes_no_style_or_truth(tmp_path: Path) -> None:
    public = tmp_path / "public"
    observations = public / "observations" / "bundle-a"
    observations.mkdir(parents=True)
    sequence = 4
    count = 2
    activations = torch.ones((count, sequence, HIDDEN_SIZE), dtype=torch.bfloat16)
    masks = torch.ones((count, sequence), dtype=torch.int64)
    positions = torch.arange(sequence, dtype=torch.int64).view(1, -1).expand(count, -1)
    relative = Path("observations/bundle-a/stage1_len3.safetensors")
    artifact = public / relative
    digest = save_observation_bundle(
        activations=activations,
        attention_mask=masks,
        position_ids=positions,
        path=artifact,
        bundle_id="bundle-a",
        stage="stage1",
        record_ids=["r0", "r1"],
    )
    index = {
        "schema": OBSERVATION_INDEX_SCHEMA,
        "task_id": "TRR-P03",
        "truth_opened": False,
        "source_truth_included": False,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "cut_depth": CUT_DEPTH,
        "bos_token_id": BOS_TOKEN_ID,
        "bundle_id": "bundle-a",
        "bundles": [
            {
                "bundle_id": "bundle-a",
                "stage": "stage1",
                "scored_tokens": 3,
                "sequence_length": sequence,
                "record_ids": ["r0", "r1"],
                "relative_path": relative.as_posix(),
                "keys": {
                    "activations": "activations",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                },
                "expected_shapes": {
                    "activations": [count, sequence, HIDDEN_SIZE],
                    "attention_mask": [count, sequence],
                    "position_ids": [count, sequence],
                },
                "bytes": artifact.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    index_path = public / "observation_index.json"
    write_json_exclusive(index_path, index)
    records = validate_observation_index(index)
    assert [record["record_id"] for record in records] == ["r0", "r1"]
    assert all("style" not in record for record in records)
    _, loaded_records, loaded = load_index_and_observations(index_path)
    assert [record["record_id"] for record in loaded_records] == ["r0", "r1"]
    assert len(loaded) == 2
    assert all(observation.source_id in {"r0", "r1"} for observation in loaded)


def _write_frozen_prediction_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    record_ids = ["r0", "r1"]
    truth_rows = {
        "r0": [BOS_TOKEN_ID, 10, 11, 12],
        "r1": [BOS_TOKEN_ID, 20, 21, 22],
    }
    methods = [
        "raw_boundary.cosine",
        "projected_boundary.cosine",
        "historical_a1.cosine",
    ]
    rows: list[dict[str, object]] = []
    tensors: dict[str, torch.Tensor] = {}
    for method in methods:
        field = method.replace(".", "_")
        values = []
        for index, record_id in enumerate(record_ids):
            truth = truth_rows[record_id]
            predicted = truth if method != "historical_a1.cosine" or index == 0 else [BOS_TOKEN_ID, 20, 0, 22]
            values.append(predicted)
            rows.append(
                {
                    "record_id": record_id,
                    "method": method,
                    "sequence_length": len(predicted),
                    "prediction_tokens": predicted,
                    "top1_tie_count": [1, 1, 1],
                    "top1_scores": [0.9, 0.8, 0.7],
                    "top1_runner_margins": [0.1, 0.1, 0.1],
                    "truth_opened": False,
                }
            )
        tensors[field] = torch.tensor(values, dtype=torch.int32)
    prediction_path = prediction_root / "predictions.safetensors"
    save_file(
        tensors,
        prediction_path,
        metadata={"schema": PREDICTION_SCHEMA, "task_id": "TRR-P03", "truth_opened": "false"},
    )
    rows_path = prediction_root / "predictions.jsonl"
    write_jsonl_exclusive(rows_path, rows)
    preflight_path = prediction_root / "preflight.json"
    write_json_exclusive(preflight_path, {"truth_opened": False, "methods": methods})
    evidence_path = prediction_root / "evidence.json"
    write_json_exclusive(evidence_path, {"truth_opened": False, "status": "ready"})
    progress_path = prediction_root / "phase_progress.jsonl"
    write_jsonl_exclusive(progress_path, [{"truth_opened": False, "event": "ready"}])
    freeze = freeze_prediction_bundle(
        root=prediction_root,
        plan_hash="a" * 64,
        implementation_commit="synthetic",
        artifacts=[prediction_path, rows_path, preflight_path, evidence_path, progress_path],
        metadata={
            "methods": methods,
            "records": len(record_ids),
            "record_ids": record_ids,
        },
    )
    write_freeze_receipt(prediction_root, freeze)
    truth_path = tmp_path / "private_truth.jsonl"
    write_jsonl_exclusive(
        truth_path,
        [{"record_id": record_id, "token_ids": token_ids} for record_id, token_ids in truth_rows.items()],
    )
    return prediction_root, truth_path, prediction_path


def test_synthetic_prediction_freeze_score_roundtrip_writes_numeric_files_first(tmp_path: Path) -> None:
    prediction_root, truth_path, _ = _write_frozen_prediction_fixture(tmp_path)
    output_root = tmp_path / "scores"
    result = score_prediction_bundle(
        prediction_root=prediction_root,
        truth_path=truth_path,
        output_root=output_root,
        bootstrap_draws=100,
        bootstrap_seed=20260905,
        records_per_stratum=2,
    )
    assert set(result) >= {"metrics", "per_record", "paired_statistics"}
    assert (output_root / "metrics.json").is_file()
    assert (output_root / "per_record.jsonl").is_file()
    assert (output_root / "paired_statistics.json").is_file()
    metrics = json.loads((output_root / "metrics.json").read_text())
    paired = json.loads((output_root / "paired_statistics.json").read_text())
    assert metrics["truth_opened"] is True
    assert metrics["summaries"]["projected_boundary.cosine"]["token_accuracy"] == pytest.approx(1.0)
    assert metrics["summaries"]["historical_a1.cosine"]["token_accuracy"] == pytest.approx(5 / 6)
    assert paired["projected_vs_a1"]["token_position_changes"]["gain_tokens"] == 1
    assert paired["projected_vs_a1"]["token_position_changes"]["regression_tokens"] == 0


def test_strict_receipt_binds_roots_and_rejects_freeze_hash_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, truth_path, _ = _write_frozen_prediction_fixture(tmp_path)
    paired = tmp_path / "predictions-paired"
    copytree(primary, paired)
    receipt_path = tmp_path / "joint_validation_receipt.json"
    methods = [
        "raw_boundary.cosine",
        "projected_boundary.cosine",
        "historical_a1.cosine",
    ]

    def receipt_for(left: Path, right: Path) -> dict[str, object]:
        return {
            "schema": "token-reconstruction.trr-p03-stage1-joint-validation.v1",
            "task_id": "TRR-P03",
            "status": "VALIDATED",
            "validation": "STAGE1_JOINT_VALIDATION_PASS",
            "truth_opened": False,
            "implementation_commit": "synthetic",
            "predictions": {
                "bundle-a": {
                    "root": str(left.resolve()),
                    "freeze": file_record(left / "freeze_receipt.json"),
                    "plan_sha256": "a" * 64,
                    "implementation_commit": "synthetic",
                    "methods": methods,
                    "record_order": ["r0", "r1"],
                    "anchor_record_ids": [],
                },
                "bundle-b": {
                    "root": str(right.resolve()),
                    "freeze": file_record(right / "freeze_receipt.json"),
                    "plan_sha256": "a" * 64,
                    "implementation_commit": "synthetic",
                    "methods": methods,
                    "record_order": ["r0", "r1"],
                    "anchor_record_ids": [],
                },
            },
            "score_prerequisite": {
                "paired_prediction_root_required": True,
                "allow_unequal_strata": False,
                "truth_read_after_this_receipt": True,
            },
        }

    write_json_exclusive(receipt_path, receipt_for(primary, paired))
    result = score_prediction_bundle(
        prediction_root=primary,
        paired_prediction_root=paired,
        truth_path=truth_path,
        output_root=tmp_path / "valid-score",
        bootstrap_draws=8,
        bootstrap_seed=20260905,
        records_per_stratum=2,
        pre_score_receipt_path=receipt_path,
        implementation_commit="synthetic",
    )
    assert result["target_bundles"] == ["primary", "paired_1"]

    bad_payload = receipt_for(primary, paired)
    bad_payload["predictions"]["bundle-b"]["freeze"]["sha256"] = "0" * 64
    bad_receipt = tmp_path / "bad-joint-validation-receipt.json"
    write_json_exclusive(bad_receipt, bad_payload)

    def truth_must_not_open(*args: object, **kwargs: object) -> dict[str, list[int]]:
        raise AssertionError("truth opened before strict receipt binding")

    monkeypatch.setattr(scoring_module, "_load_truth", truth_must_not_open)
    with pytest.raises(ScoringError, match="freeze hash"):
        score_prediction_bundle(
            prediction_root=primary,
            paired_prediction_root=paired,
            truth_path=truth_path,
            output_root=tmp_path / "bad-score",
            bootstrap_draws=8,
            bootstrap_seed=20260905,
            records_per_stratum=2,
            pre_score_receipt_path=bad_receipt,
            implementation_commit="synthetic",
        )
    assert not (tmp_path / "bad-score").exists()


def test_paired_prediction_freeze_scores_both_bundles_before_truth(tmp_path: Path) -> None:
    primary, truth_path, _ = _write_frozen_prediction_fixture(tmp_path)
    paired = tmp_path / "predictions-paired"
    copytree(primary, paired)
    output_root = tmp_path / "paired-scores"

    result = score_prediction_bundle(
        prediction_root=primary,
        paired_prediction_root=paired,
        truth_path=truth_path,
        output_root=output_root,
        bootstrap_draws=32,
        bootstrap_seed=20260905,
        records_per_stratum=2,
    )

    metrics = json.loads((output_root / "metrics.json").read_text())
    paired_stats = json.loads((output_root / "paired_statistics.json").read_text())
    scored_rows = (output_root / "per_record.jsonl").read_text().splitlines()
    assert result["target_bundles"] == ["primary", "paired_1"]
    assert metrics["target_bundles"] == ["primary", "paired_1"]
    assert len(metrics["bundle_summaries"]) == 2
    assert len(scored_rows) == 12
    assert paired_stats["paired_bundle_verified_before_truth"] is True
    assert json.loads((output_root / "pre_score_gate.json").read_text())["truth_opened"] is False


def test_p03_cli_modules_import_and_serialize_help(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("generate_observations.py", "prepare_projected.py", "reconstruct.py", "score.py"):
        completed = subprocess.run(
            [sys.executable, str(root / "scripts" / "trr_p03" / name), "--help"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()

def test_frozen_evaluator_panel_parser_is_truth_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real frozen panel can be checked without loading its sidecar."""

    panel_path = Path("experiments/TRR-P03/setup/panel-20260906-frozen/stage1/evaluator_panel.json")
    if not panel_path.is_file():
        pytest.skip("frozen setup panel is unavailable in this checkout")
    from scripts.trr_p03 import generate_observations as generator

    def fail_if_opened(*args: object, **kwargs: object) -> dict[str, list[int]]:
        raise AssertionError("metadata-only panel parsing opened private truth")

    monkeypatch.setattr(generator, "_load_truth", fail_if_opened)
    panel, rows, truth_path = generator._read_panel(
        panel_path,
        truth_path=None,
        truth_index_path=None,
        stage="stage1",
        open_truth=False,
    )
    assert panel["truth_opened"] is False
    assert len(rows) == 24
    assert [row["record_id"] for row in rows[:2]] == ["p03-s1-r0001", "p03-s1-r0002"]
    assert rows[-1]["record_id"] == "p03-s1-r0024"
    assert all("style" not in row and "token_ids" in row for row in rows)
    assert truth_path is not None and truth_path.name == "private_truth.jsonl"

