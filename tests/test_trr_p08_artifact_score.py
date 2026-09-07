from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.trr_p08 import freeze_matrix, score_frozen
from test_trr_p08_freeze_matrix import _fixture as make_freeze_fixture


DOMAINS = ("pile", "finance")


def _record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _sequence_digest(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.int32).tobytes(order="C")).hexdigest()


def _truth() -> np.ndarray:
    value = np.zeros((256, 128), dtype=np.int64)
    value[:, 0] = score_frozen.BOS_TOKEN_ID
    for row in range(256):
        value[row, 1:] = 1000 + row * 127 + np.arange(127, dtype=np.int64)
    return value


def _artifact_fixture(tmp_path: Path) -> dict[str, Path | str]:
    fixture = make_freeze_fixture(tmp_path)
    root = Path(fixture["root"])
    truth = _truth()
    record_ids = {domain: [f"{domain}-{i}" for i in range(256)] for domain in DOMAINS}
    sequence_hashes = {domain: [_sequence_digest(row) for row in truth] for domain in DOMAINS}

    # Convert each synthetic prediction payload to a valid integer artifact and
    # refresh only the prediction descriptors' immutable file records.
    prediction_manifest_path = Path(fixture["predictions"])
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    for cell_id in freeze_matrix.CELL_ORDER:
        domain = cell_id.split("__", 1)[0]
        for seed in freeze_matrix.SEEDS:
            for method in freeze_matrix.METHOD_ORDER:
                descriptor = prediction_manifest["student_cells"][cell_id][str(seed)][method]
                path = Path(descriptor["prediction"]["path"])
                npy_path = path.with_suffix(".npy")
                path.rename(npy_path)
                np.save(npy_path, truth, allow_pickle=False)
                descriptor["prediction"] = _record(npy_path)
                descriptor["record_ids_sha256"] = _digest(record_ids[domain])
    selection_path = root / "runtime/source-selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({
        "schema": "token-reconstruction.trr-p08-source-selection.v1",
        "task_id": "TRR-P08",
        "status": "FROZEN_TRR-P08_SOURCE_SELECTION_NO_TRUTH",
        "records_per_domain": 256,
        "selection_rule": {"records": {
            domain: [{"record_id": record_ids[domain][i], "final_sequence_sha256": sequence_hashes[domain][i]} for i in range(256)]
            for domain in DOMAINS
        }},
        "access_boundary": {"truth_opened": False, "source_text_written": False, "token_ids_written": False},
    }, sort_keys=True), encoding="utf-8")
    selection_record = _record(selection_path)

    observation_path = Path(fixture["observations"])
    observation_manifest = json.loads(observation_path.read_text(encoding="utf-8"))
    observation_manifest["selection_plan"] = selection_record
    observation_manifest["source_pairing"]["record_ids_sha256"] = {domain: _digest(record_ids[domain]) for domain in DOMAINS}
    for row in observation_manifest["cells"]:
        row["record_ids_sha256"] = _digest(record_ids[row["style"]])
    observation_path.write_text(json.dumps(observation_manifest, sort_keys=True), encoding="utf-8")
    observation_record = _record(observation_path)
    prediction_manifest["observation_manifest"] = observation_record
    prediction_manifest_path.write_text(json.dumps(prediction_manifest, sort_keys=True), encoding="utf-8")

    freeze_path = Path(fixture["output"])
    assembled = freeze_matrix.assemble_joint_freeze(
        repository_root=root,
        plan_path=Path(fixture["plan"]),
        approval_path=Path(fixture["approval"]),
        fit_receipt_path=Path(fixture["fit"]),
        prediction_manifest_path=prediction_manifest_path,
        observation_manifest_path=observation_path,
        output_path=freeze_path,
        expected_plan_sha256=str(fixture["plan_sha"]),
    )
    truth_paths: dict[str, dict[str, object]] = {}
    for domain in DOMAINS:
        path = root / f"runtime/truth-{domain}.npy"
        np.save(path, truth, allow_pickle=False)
        truth_paths[domain] = _record(path)
    truth_manifest_path = root / "runtime/truth.manifest.json"
    truth_manifest_path.write_text(json.dumps({
        "schema": score_frozen.TRUTH_SCHEMA,
        "task_id": "TRR-P08",
        "status": score_frozen.TRUTH_STATUS,
        "joint_freeze_sha256": assembled["output"]["sha256"],
        "source_selection_sha256": selection_record["sha256"],
        "observation_manifest_sha256": observation_record["sha256"],
        "prediction_manifest_sha256": _record(prediction_manifest_path)["sha256"],
        "record_ids_sha256": {domain: _digest(record_ids[domain]) for domain in DOMAINS},
        "final_sequence_sha256": sequence_hashes,
        "records_per_domain": 256,
        "sequence_tokens_including_bos": 128,
        "scored_post_bos_tokens": 127,
        "vocabulary_size": 128256,
        "target_conditions": list(freeze_matrix.TARGETS),
        "source_text_persisted": False,
        "target_labels_loaded": False,
        "model_loaded": False,
        "truth_opened": True,
        "truth_materialized": True,
        "domains": truth_paths,
    }, sort_keys=True), encoding="utf-8")
    return {"root": root, "freeze": freeze_path, "truth": truth_manifest_path, "plan_sha": str(fixture["plan_sha"]), "truth_array": root / "runtime/truth-pile.npy"}


def test_artifact_runner_revalidates_freeze_then_scores_and_writes_no_truth(tmp_path: Path) -> None:
    fixture = _artifact_fixture(tmp_path)
    output = Path(fixture["root"]) / "runtime/score.json"
    result = score_frozen.run(
        repository_root=Path(fixture["root"]),
        freeze_receipt_path=Path(fixture["freeze"]),
        truth_manifest_path=Path(fixture["truth"]),
        output_path=output,
        expected_plan_sha256=str(fixture["plan_sha"]),
        bootstrap_draws=16,
        bootstrap_seed=8080,
    )
    assert result["status"] == "TRR-P08_SCORED_AFTER_JOINT_FREEZE"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["truth_opened"] is True
    assert payload["truth_payload_persisted"] is False
    assert payload["provenance"]["scored_after_joint_freeze"] is True
    assert all("truth" not in cell for cell in payload["cells"].values())


def test_artifact_runner_rejects_reordered_truth_after_hash_refresh(tmp_path: Path) -> None:
    fixture = _artifact_fixture(tmp_path)
    truth_path = Path(fixture["truth_array"])
    truth = np.load(truth_path, allow_pickle=False)
    np.save(truth_path, truth[::-1], allow_pickle=False)
    manifest_path = Path(fixture["truth"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["domains"]["pile"] = _record(truth_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(score_frozen.P08ScoreError, match="sequence fingerprints"):
        score_frozen.run(
            repository_root=Path(fixture["root"]),
            freeze_receipt_path=Path(fixture["freeze"]),
            truth_manifest_path=manifest_path,
            output_path=Path(fixture["root"]) / "runtime/bad-score.json",
            expected_plan_sha256=str(fixture["plan_sha"]),
            bootstrap_draws=8,
        )


def test_artifact_runner_rejects_tampered_prediction_before_truth(tmp_path: Path) -> None:
    fixture = _artifact_fixture(tmp_path)
    freeze = json.loads(Path(fixture["freeze"]).read_text(encoding="utf-8"))
    descriptor = freeze["descriptors"][f"pile__public_base::{freeze_matrix.SEEDS[0]}::{freeze_matrix.METHOD_ORDER[0]}"]
    prediction_path = Path(descriptor["prediction"]["path"])
    prediction_path.write_bytes(b"changed")
    with pytest.raises(score_frozen.P08ScoreError, match="binding changed"):
        score_frozen.run(
            repository_root=Path(fixture["root"]),
            freeze_receipt_path=Path(fixture["freeze"]),
            truth_manifest_path=Path(fixture["truth"]),
            output_path=Path(fixture["root"]) / "runtime/bad-score.json",
            expected_plan_sha256=str(fixture["plan_sha"]),
            bootstrap_draws=8,
        )


def test_artifact_cli_runs_the_same_create_only_loader(tmp_path: Path) -> None:
    fixture = _artifact_fixture(tmp_path)
    output = Path(fixture["root"]) / "runtime/cli-score.json"
    code = score_frozen.main([
        "--repository-root", str(fixture["root"]),
        "--freeze-receipt", str(fixture["freeze"]),
        "--truth-manifest", str(fixture["truth"]),
        "--output", str(output),
        "--expected-plan-sha256", str(fixture["plan_sha"]),
        "--bootstrap-draws", "8",
        "--bootstrap-seed", "8080",
    ])
    assert code == 0
    assert output.is_file()
