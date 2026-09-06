from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from token_reconstruction.p04_student import (
    METHOD_H,
    StudentArchitectureConfig,
    build_student,
)
from token_reconstruction.p04_training import PublicPool, TeacherEvidence
from token_reconstruction.p05_diagnostics import (
    _row_rank_metric,
    _top2_metrics,
    forward_state,
    gradient_cell,
)


def _evidence(*, candidate_ids: torch.Tensor, record_ids: tuple[str, ...], positions: tuple[int, ...]) -> TeacherEvidence:
    rows = int(candidate_ids.shape[0])
    return TeacherEvidence(
        candidate_ids=candidate_ids.to(dtype=torch.int64),
        teacher_scores=torch.arange(rows * 32, 0, -1, dtype=torch.float32).reshape(rows, 32),
        record_ids=record_ids,
        positions=positions,
        row_kind=tuple("difficult_a1_error" for _ in range(rows)),
        sigma_q=1.0,
        tie_tolerance=0.001,
        source_path="synthetic-teacher.safetensors",
        source_sha256="synthetic",
        metadata={},
    )



def test_top1_uses_lowest_token_id_on_exact_tie() -> None:
    logits = torch.tensor([[1.0, 4.0, 7.0, 2.0, 7.0, 7.0]], dtype=torch.float32)
    top1, tie_count, margin, _gold = _top2_metrics(logits, torch.tensor([2]))
    assert int(top1.item()) == 2
    assert int(tie_count.item()) == 3
    assert float(margin.item()) == pytest.approx(0.0)


def test_rank_metric_gathers_noncontiguous_token_ids() -> None:
    # The candidate columns are 9, 2, 7.  Their full-vocabulary scores are
    # deliberately different from logits columns 0, 1, 2.
    logits = torch.zeros((1, 10), dtype=torch.float32)
    logits[0, 9] = 3.0
    logits[0, 2] = 1.0
    logits[0, 7] = -1.0
    candidate_ids = torch.tensor([[9, 2, 7]], dtype=torch.int64)
    teacher_scores = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([8], dtype=torch.int64)
    rows, aggregate = _row_rank_metric(
        logits,
        labels,
        candidate_ids,
        teacher_scores,
        sigma_q=1.0,
        tie_tolerance=0.001,
    )
    assert rows[0]["pair_order_agree"] == 2
    assert rows[0]["rank_pairs"] == 2
    assert aggregate["pair_order_agreement"] == 1.0
    assert aggregate["rank_pairs"] == 2


def test_forward_state_uses_original_noncontiguous_pool_rows() -> None:
    config = StudentArchitectureConfig(hidden_size=2, vocab_size=10, gru_width=2, initial_logit_scale=1.0)
    model = build_student(METHOD_H, config=config)
    # GRU output projection is zero initialized, so this is the affine path.
    observations = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    labels = torch.zeros((3, 4), dtype=torch.long)
    labels[0, 2] = 1
    labels[2, 1] = 1
    valid = torch.ones((3, 4), dtype=torch.bool)
    pool = PublicPool(
        observations=observations,
        labels=labels,
        valid_mask=valid,
        record_ids=("r0", "r1", "r2"),
        styles=("a", "b", "c"),
        source_path="synthetic-observations.safetensors",
        source_sha256="synthetic",
        records_path="synthetic-records.json",
        records_sha256="synthetic",
    )
    table = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]] + [[0.5, 0.5]] * 6), dim=-1)
    sample = {
        "forward": {
            "rows": [
                {"sample_kind": "control", "teacher_kind": None, "teacher_row": None, "record_id": "r2", "pool_row": 2, "position": 1},
                {"sample_kind": "control", "teacher_kind": None, "teacher_row": None, "record_id": "r0", "pool_row": 0, "position": 2},
            ]
        }
    }
    evidence = TeacherEvidence(
        candidate_ids=torch.empty((0, 32), dtype=torch.int64),
        teacher_scores=torch.empty((0, 32), dtype=torch.float32),
        record_ids=(),
        positions=(),
        row_kind=(),
        sigma_q=1.0,
        tie_tolerance=0.001,
        source_path="synthetic-teacher.safetensors",
        source_sha256="synthetic",
        metadata={},
    )
    rows, summary = forward_state(model, pool, sample, evidence, table, device=torch.device("cpu"), state_id="synthetic", record_batch_size=2, projection_chunk=1)
    assert [row["pool_row"] for row in rows] == [2, 0]
    assert [row["label"] for row in rows] == [1, 1]
    assert all(row["tie_count"] >= 1 for row in rows)
    assert summary["total_rows"] == 2


def test_h_checkpoint_reports_counterfactual_rank_without_update() -> None:
    config = StudentArchitectureConfig(hidden_size=2, vocab_size=40, gru_width=2, initial_logit_scale=1.0)
    model = build_student(METHOD_H, config=config)
    observations = torch.randn((8, 3, 2), generator=torch.Generator().manual_seed(7))
    labels = torch.zeros((8, 3), dtype=torch.long)
    valid = torch.ones((8, 3), dtype=torch.bool)
    combined = PublicPool(
        observations=observations,
        labels=labels,
        valid_mask=valid,
        record_ids=tuple(f"r{i}" for i in range(8)),
        styles=tuple("public" for _ in range(8)),
        source_path="synthetic-combined.safetensors",
        source_sha256="synthetic",
        records_path="synthetic-records.json",
        records_sha256="synthetic",
    )
    base_candidates = torch.arange(32, dtype=torch.int64)
    candidates = base_candidates.reshape(1, 1, 32).expand(8, 3, 32).clone()
    teacher = _evidence(candidate_ids=base_candidates.reshape(1, 32), record_ids=("r0",), positions=(1,))
    batch = {
        "seed": 1737,
        "step": 0,
        "record_indices": list(range(8)),
        "records": [
            {"local_row": i, "pool_row": i, "record_id": f"r{i}", "selected_positions": [1]}
            for i in range(8)
        ],
        "teacher_kind_counts": {"difficult_a1_error": 1},
    }
    table = F.normalize(torch.randn((40, 2), generator=torch.Generator().manual_seed(8)), dim=-1)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    cell = gradient_cell(
        model,
        METHOD_H,
        combined,
        batch,
        candidates,
        {(0, 1): 0},
        teacher,
        table,
        device=torch.device("cpu"),
        state_id="synthetic-h",
    )
    after = model.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert cell["teacher_rows"] == 1
    assert cell["losses"]["rank"] > 0.0
    assert cell["losses"]["actual_total"] == pytest.approx(cell["losses"]["ce"] + 0.25 * cell["losses"]["hard"], rel=1e-7, abs=1e-7)
    assert cell["losses"]["hypothetical_d_total"] > cell["losses"]["actual_total"]
    assert cell["optimizer_step_called"] is False
    assert cell["parameter_update_applied"] is False
    assert cell["gradient_norms"]["clip_factor"] <= 1.0


def _write_synthetic_pool(root: Path, prefix: str, rows: int) -> tuple[Path, Path]:
    positions = 192
    observations = torch.zeros((rows, positions, 2), dtype=torch.float32)
    labels = torch.ones((rows, positions), dtype=torch.long)
    labels[:, 0] = 128000
    valid = torch.ones((rows, positions), dtype=torch.bool)
    observation_path = root / f"{prefix}-observations.safetensors"
    records_path = root / f"{prefix}-records.json"
    save_file(
        {"activations": observations, "token_ids": labels, "attention_mask": valid},
        str(observation_path),
    )
    records_path.write_text(
        json.dumps({"records": [{"record_id": f"{prefix}-{row:04d}", "sequence_length": positions, "style": "synthetic"} for row in range(rows)]}),
        encoding="utf-8",
    )
    return observation_path, records_path


def test_cli_sample_freezes_serialized_ledger(tmp_path: Path) -> None:
    correction_observations, correction_records = _write_synthetic_pool(tmp_path, "correction", 384)
    replay_observations, replay_records = _write_synthetic_pool(tmp_path, "replay", 6)
    evidence_rows = [
        {"record_id": f"correction-{row:04d}", "position": 1, "kind": "difficult_a1_error" if row < 256 else "uniform_audit"}
        for row in range(384)
    ]
    evidence_path = tmp_path / "teacher.safetensors"
    save_file(
        {
            "candidate_ids": torch.arange(32, dtype=torch.int64).repeat(384, 1),
            "teacher_scores": torch.arange(32, 0, -1, dtype=torch.float32).repeat(384, 1),
        },
        str(evidence_path),
        metadata={"rows_json": json.dumps(evidence_rows), "sigma_q": "1.0", "tie_tolerance": "0.001"},
    )
    schedule_paths: list[Path] = []
    record_indices = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long).expand(3000, -1).clone()
    selected_mask = torch.zeros((3000, 8, 192), dtype=torch.bool)
    selected_mask[:, :, 1] = True
    for seed in (1737, 2711):
        path = tmp_path / f"schedule-{seed}.safetensors"
        save_file({"record_indices": record_indices, "selected_mask": selected_mask}, str(path))
        schedule_paths.append(path)
    output = tmp_path / "sample-index.json"
    script = Path(__file__).parents[1] / "scripts" / "trr0004_p05_diagnostic.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "sample",
            "--correction-observations", str(correction_observations),
            "--correction-records", str(correction_records),
            "--replay-observations", str(replay_observations),
            "--replay-records", str(replay_records),
            "--teacher-evidence", str(evidence_path),
            "--schedule", f"1737={schedule_paths[0]}",
            "--schedule", f"2711={schedule_paths[1]}",
            "--output", str(output),
        ],
        cwd=str(Path(__file__).parents[1]),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    cli_payload = json.loads(result.stdout)
    sample = json.loads(output.read_text(encoding="utf-8"))
    assert cli_payload["status"] == "PASS"
    assert sample["status"] == "PUBLIC_SAMPLE_FROZEN_NO_MODEL_LOADED"
    assert sample["forward"]["total_count"] == 768
    assert sample["forward"]["teacher_partition"] == {"difficult_a1_error": 256, "uniform_audit": 128}
    assert sample["gradient"]["batch_count"] == 8
