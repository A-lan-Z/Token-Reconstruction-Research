from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from token_reconstruction.trr0005_joint_decoder import (
    PublicJointData,
    build_position_schedule,
)
from token_reconstruction.trr0006_visibility_decoder import (
    FULL_RECORD_METHOD,
    load_visibility_state,
)
import trr0006_fit_visibility as runner


def _tiny_data() -> tuple[PublicJointData, dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6106)
    records, positions, hidden, vocab = 8, 6, 8, 13
    observations = torch.randn(records, positions, hidden, generator=generator)
    truth = torch.randint(vocab, (records, positions), generator=generator, dtype=torch.long)
    truth[:, 0] = 0
    mask = torch.ones(records, positions, dtype=torch.bool)
    table = F.normalize(torch.randn(vocab, hidden, generator=generator), dim=-1)
    direct = {
        "W": torch.eye(hidden, dtype=torch.float32),
        "b": torch.zeros(hidden, dtype=torch.float32),
        "s": torch.tensor(2.25, dtype=torch.float32),
    }
    data = PublicJointData(
        fit_observations=observations,
        fit_truth=truth,
        fit_valid_mask=mask,
        fit_record_ids=tuple(f"fit-{index}" for index in range(records)),
        validation_observations=observations.clone(),
        validation_truth=truth.clone(),
        validation_valid_mask=mask.clone(),
        validation_record_ids=tuple(f"val-{index}" for index in range(records)),
        validation_groups=tuple("synthetic" for _ in range(records)),
        embedding_table=table,
        metadata={},
    )
    return data, direct


def _guard_args() -> SimpleNamespace:
    return SimpleNamespace(
        maximum_host_rss_gib=16.0,
        minimum_host_available_gib=0.1,
        minimum_free_gib=0.1,
        maximum_gpu_reserved_gib=6.0,
        direct_affine_sha256=runner.DIRECT_AFFINE_SHA256,
    )


def test_probe_schedule_repeats_only_ledger_errors_and_is_seed_bound() -> None:
    ledger = []
    for record in range(8):
        for position in range(1, 33):
            ledger.append(
                {
                    "record_index": record,
                    "record_id": f"fit-{record}",
                    "position": position,
                    "bin": "1-15" if position <= 15 else "16-39",
                }
            )
    first = runner.build_probe_schedule(ledger, steps=3, record_batch_size=2, position_budget=4, seed=6106)
    second = runner.build_probe_schedule(ledger, steps=3, record_batch_size=2, position_budget=4, seed=6106)
    assert torch.equal(first.batch_record_indices, second.batch_record_indices)
    assert torch.equal(first.draw_record_slots, second.draw_record_slots)
    assert torch.equal(first.draw_position_slots, second.draw_position_slots)
    allowed = {
        record: set(range(1, 33))
        for record in range(8)
    }
    for step in range(first.steps):
        batch = first.batch_record_indices[step].tolist()
        for local_slot, position in zip(
            first.draw_record_slots[step].tolist(), first.draw_position_slots[step].tolist()
        ):
            assert 0 <= local_slot < len(batch)
            assert position in allowed[int(batch[local_slot])]
    assert bool(first.used_replacement.all().item())


def test_runner_main_arm_trains_saves_and_loads_selected_state(tmp_path: Path, monkeypatch) -> None:
    data, direct = _tiny_data()
    monkeypatch.setattr(runner, "HIDDEN_SIZE", 8)
    monkeypatch.setattr(runner, "VOCABULARY_SIZE", 13)
    monkeypatch.setattr(runner, "CONTEXT_WIDTH", 4)
    monkeypatch.setattr(runner, "RECORD_BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "POSITION_BUDGET", 4)
    monkeypatch.setattr(runner, "MAIN_STEPS", 2)
    monkeypatch.setattr(runner, "VALIDATION_EVERY", 1)
    schedule = build_position_schedule(
        data.fit_valid_mask,
        steps=2,
        record_batch_size=2,
        position_budget=4,
        seed=6106,
    )
    schedule_record = {"schedule_sha256": "synthetic-schedule"}
    output = tmp_path / "arm"
    output.mkdir()
    guards: list[dict[str, object]] = []
    result = runner._train_main_arm(
        FULL_RECORD_METHOD,
        6106,
        data,
        direct,
        data.embedding_table,
        schedule,
        schedule_record,
        args=_guard_args(),
        device=torch.device("cpu"),
        output_dir=output,
        deadline=None,
        guards=guards,
    )
    assert result["status"] == "PASS"
    assert result["selected_step"] in {0, 1, 2}
    assert result["state"]["metadata"]["direct_affine_sha256"] == runner.DIRECT_AFFINE_SHA256
    loaded = load_visibility_state(
        output / "selected.safetensors",
        method_id=FULL_RECORD_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
    )
    assert loaded.method_id == FULL_RECORD_METHOD
    assert (output / "learning_curve.json").is_file()
    assert guards
