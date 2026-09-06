from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from token_reconstruction.p04_student import METHOD_H, StudentArchitectureConfig
from token_reconstruction.p04_training import (
    TrainingConfig,
    generate_candidate_ids,
    load_public_pool,
    make_position_schedule,
    train_arm,
)


def _write_pool(tmp_path, name: str, rows: int, *, offset: int) -> tuple[object, object]:
    table = torch.nn.functional.normalize(torch.randn(7, 4, generator=torch.Generator().manual_seed(100 + offset)), dim=-1)
    ids = torch.arange(rows * 6).reshape(rows, 6) % 7
    ids[:, 0] = 0
    observations = table[ids].clone()
    mask = torch.ones((rows, 6), dtype=torch.bool)
    artifact = tmp_path / f"{name}.safetensors"
    save_file({"activations": observations, "token_ids": ids.to(torch.int64), "attention_mask": mask}, str(artifact))
    records = tmp_path / f"{name}.json"
    records.write_text(json.dumps({"records": [{"record_id": f"{name}-{offset + i}", "style": "synthetic", "sequence_length": 6} for i in range(rows)]}) + "\n")
    return artifact, records


def test_public_schedule_candidate_and_train_smoke(tmp_path) -> None:
    replay_artifact, replay_records = _write_pool(tmp_path, "replay", 4, offset=0)
    correction_artifact, correction_records = _write_pool(tmp_path, "correction", 2, offset=10)
    validation_artifact, validation_records = _write_pool(tmp_path, "validation", 2, offset=20)
    replay = load_public_pool(replay_artifact, replay_records, embedding_vocab_size=7, bos_token_id=0)
    correction = load_public_pool(correction_artifact, correction_records, embedding_vocab_size=7, bos_token_id=0)
    validation = load_public_pool(validation_artifact, validation_records, embedding_vocab_size=7, bos_token_id=0)
    table = torch.nn.functional.normalize(torch.randn(7, 4, generator=torch.Generator().manual_seed(300)), dim=-1)
    architecture = StudentArchitectureConfig(hidden_size=4, vocab_size=7, gru_width=3)
    from token_reconstruction.p04_training import combine_public_pools

    pool = combine_public_pools(replay, correction)
    candidates = generate_candidate_ids(pool, table, affine_state=None, config=architecture, device=torch.device("cpu"), candidate_k=3, record_batch_size=2, projection_chunk=4)
    assert candidates.shape == (6, 6, 3)
    schedule = make_position_schedule(pool, replay_records=4, steps=2, record_batch_size=4, position_budget=8, seed=1737)
    assert schedule.selected_mask.sum(dim=(1, 2)).le(8).all()
    output = tmp_path / "arm"
    result = train_arm(
        METHOD_H,
        pool=pool,
        validation=validation,
        embedding_table=table,
        affine_state=None,
        schedule=schedule,
        candidate_ids=candidates,
        teacher_scores=None,
        teacher_mask=None,
        sigma_q=None,
        tie_tolerance=None,
        seed=1737,
        config=TrainingConfig(steps=2, record_batch_size=4, position_budget=8, validation_every=1, projection_chunk=4),
        architecture=architecture,
        device=torch.device("cpu"),
        output_dir=output,
    )
    assert result["selected_step"] in (0, 1, 2)
    assert (output / "selected.safetensors").is_file()
    assert (output / "final.safetensors").is_file()
    assert (output / "learning_curve.json").is_file()


def test_validation_regression_keeps_step_zero_checkpoint(tmp_path, monkeypatch) -> None:
    import token_reconstruction.p04_training as training

    replay_artifact, replay_records = _write_pool(tmp_path, "reg-replay", 4, offset=40)
    correction_artifact, correction_records = _write_pool(tmp_path, "reg-correction", 2, offset=50)
    validation_artifact, validation_records = _write_pool(tmp_path, "reg-validation", 2, offset=60)
    replay = load_public_pool(replay_artifact, replay_records, embedding_vocab_size=7, bos_token_id=0)
    correction = load_public_pool(correction_artifact, correction_records, embedding_vocab_size=7, bos_token_id=0)
    validation = load_public_pool(validation_artifact, validation_records, embedding_vocab_size=7, bos_token_id=0)
    pool = training.combine_public_pools(replay, correction)
    table = torch.nn.functional.normalize(torch.randn(7, 4, generator=torch.Generator().manual_seed(301)), dim=-1)
    architecture = __import__("token_reconstruction.p04_student", fromlist=["StudentArchitectureConfig"]).StudentArchitectureConfig(hidden_size=4, vocab_size=7, gru_width=3)
    candidates = training.generate_candidate_ids(pool, table, affine_state=None, config=architecture, device=torch.device("cpu"), candidate_k=3, record_batch_size=2, projection_chunk=4)
    schedule = training.make_position_schedule(pool, replay_records=4, steps=1, record_batch_size=4, position_budget=8, seed=1737)
    calls = {"count": 0}

    def fake_evaluate(*args, **kwargs):
        del args, kwargs
        calls["count"] += 1
        metric = 0.9 if calls["count"] == 1 else 0.1
        return {"style_balanced_token_accuracy": metric, "predictions": torch.zeros((2, 6), dtype=torch.int64), "tie_counts": torch.zeros((2, 6), dtype=torch.int32)}

    monkeypatch.setattr(training, "evaluate_public", fake_evaluate)
    result = training.train_arm(
        "student_h",
        pool=pool,
        validation=validation,
        embedding_table=table,
        affine_state=None,
        schedule=schedule,
        candidate_ids=candidates,
        teacher_scores=None,
        teacher_mask=None,
        sigma_q=None,
        tie_tolerance=None,
        seed=1737,
        config=training.TrainingConfig(steps=1, record_batch_size=4, position_budget=8, validation_every=1, projection_chunk=4),
        architecture=architecture,
        device=torch.device("cpu"),
        output_dir=tmp_path / "regression-arm",
    )
    assert result["selected_step"] == 0


def test_truth_free_prediction_cli_round_trip(tmp_path) -> None:
    import os
    import subprocess
    from token_reconstruction.p04_student import METHOD_S, StudentArchitectureConfig, build_student, save_student_state

    config = StudentArchitectureConfig(hidden_size=4, vocab_size=7, gru_width=256)
    model = build_student(METHOD_S, config=config)
    table = torch.nn.functional.normalize(torch.randn(7, 4, generator=torch.Generator().manual_seed(302)), dim=-1)
    token_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 2, 1]])
    observations = table[token_ids]
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    observation_path = tmp_path / "observations.safetensors"
    save_file({"activations": observations, "attention_mask": mask}, str(observation_path))
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps({"records": [{"record_id": "eval-0", "style": "synthetic"}, {"record_id": "eval-1", "style": "synthetic"}]}) + "\n")
    embedding_path = tmp_path / "embeddings.safetensors"
    save_file({"embeddings": table}, str(embedding_path))
    state_path = tmp_path / "state.safetensors"
    save_student_state(model, state_path, method_id=METHOD_S, seed=1737, config=config)
    output_path = tmp_path / "predictions.jsonl"
    ties_path = tmp_path / "ties.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str((__import__("pathlib").Path.cwd() / "src").resolve())
    completed = subprocess.run(
        [
            "python3", "scripts/trr0004_p04_predict.py", "--observations", str(observation_path), "--records", str(records_path), "--state", str(state_path), "--method-id", METHOD_S, "--seed", "1737", "--condition", "public_base", "--embedding-table", str(embedding_path), "--output", str(output_path), "--tie-output", str(ties_path), "--device", "cpu", "--threads", "1", "--interop-threads", "1", "--projection-chunk", "3",
        ],
        cwd=str(__import__("pathlib").Path.cwd()),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["schema"] == "token-reconstruction.trr-p04-predictions.v1" for row in rows)
    tie_payload = json.loads(ties_path.read_text())
    assert tie_payload["rows"][0]["record_id"] == "eval-0"


def test_teacher_candidate_identity_fixture_binds_exact_shared_rows(tmp_path) -> None:
    import pytest
    import token_reconstruction.p04_training as training

    replay_artifact, replay_records = _write_pool(tmp_path, "bind-replay", 4, offset=70)
    correction_artifact, correction_records = _write_pool(tmp_path, "bind-correction", 2, offset=80)
    replay = load_public_pool(replay_artifact, replay_records, embedding_vocab_size=7, bos_token_id=0)
    correction = load_public_pool(correction_artifact, correction_records, embedding_vocab_size=7, bos_token_id=0)
    pool = training.combine_public_pools(replay, correction)
    table = torch.nn.functional.normalize(torch.randn(7, 4, generator=torch.Generator().manual_seed(303)), dim=-1)
    architecture = StudentArchitectureConfig(hidden_size=4, vocab_size=7, gru_width=3)
    candidates = training.generate_candidate_ids(
        pool,
        table,
        affine_state=None,
        config=architecture,
        device=torch.device("cpu"),
        candidate_k=3,
        record_batch_size=2,
        projection_chunk=4,
    )
    evidence = training.TeacherEvidence(
        candidate_ids=candidates[0, 1].reshape(1, 3).to(torch.int64),
        teacher_scores=torch.tensor([[0.8, 0.3, -0.1]], dtype=torch.float32),
        record_ids=(pool.record_ids[0],),
        positions=(1,),
        row_kind=("fixture",),
        sigma_q=1.0,
        tie_tolerance=0.01,
        source_path="fixture",
        source_sha256="fixture",
        metadata={"schema": training.TEACHER_EVIDENCE_SCHEMA},
    )
    bound, scores, mask, binding = training.build_teacher_arrays(pool, candidates, evidence)
    assert torch.equal(bound, candidates)
    assert mask[0, 1].item() is True
    assert torch.equal(scores[0, 1], evidence.teacher_scores[0])
    assert binding["required_positions"] == {pool.record_ids[0]: [1]}

    mismatched = candidates.clone()
    row = mismatched[0, 1]
    replacement = next(token for token in range(7) if token not in set(row.tolist()))
    mismatched[0, 1, 0] = replacement
    with pytest.raises(training.P04TrainingError, match="candidate identity mismatch"):
        training.build_teacher_arrays(pool, mismatched, evidence)
