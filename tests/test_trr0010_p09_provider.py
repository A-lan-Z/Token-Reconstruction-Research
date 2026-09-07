from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from token_reconstruction.trr0007_positionwise import build_residual_mlp512
from trr0010_model import build_directional_from_base
import trr0010_p09_provider as provider


def _digest(seed: int, steps: list[dict[str, object]]) -> str:
    payload = {"seed": int(seed), "steps": steps}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class _Step:
    def __init__(self, *, step, batch_global_rows, draw_record_slots, draw_position_slots, used_replacement):
        self.step = int(step)
        self.batch_global_rows = tuple(batch_global_rows)
        self.draw_record_slots = tuple(draw_record_slots)
        self.draw_position_slots = tuple(draw_position_slots)
        self.used_replacement = bool(used_replacement)

    def as_dict(self):
        return {
            "step": self.step,
            "batch_global_rows": list(self.batch_global_rows),
            "draw_record_slots": list(self.draw_record_slots),
            "draw_position_slots": list(self.draw_position_slots),
            "used_replacement": self.used_replacement,
        }

    def validate(self, *, record_batch_size, position_budget, sequence_tokens):
        assert len(self.batch_global_rows) == record_batch_size
        assert len(self.draw_record_slots) == position_budget
        assert len(self.draw_position_slots) == position_budget
        assert all(0 <= value < record_batch_size for value in self.draw_record_slots)
        assert all(0 < value < sequence_tokens for value in self.draw_position_slots)


class _Plan:
    def __init__(self, seed, steps):
        self.seed = int(seed)
        self.steps = tuple(steps)
        self.semantic_sha256 = _digest(self.seed, [step.as_dict() for step in self.steps])

    @classmethod
    def from_steps(cls, *, seed, steps):
        return cls(seed, steps)

    def validate(self, **kwargs):
        assert tuple(step.step for step in self.steps) == tuple(range(len(self.steps)))
        for step in self.steps:
            step.validate(**kwargs)

    def exposure_summary(self):
        return {"steps": len(self.steps), "draws_per_step": len(self.steps[0].draw_position_slots)}


def _runner() -> SimpleNamespace:
    return SimpleNamespace(ScheduleStep=_Step, SchedulePlan=_Plan)


def test_deserialize_schedule_preserves_order_and_digest(tmp_path):
    batch_rows = torch.tensor([[4, 9], [2, 7]], dtype=torch.int64)
    draw_records = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int64)
    draw_positions = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64)
    steps = [
        {
            "step": index,
            "batch_global_rows": batch_rows[index].tolist(),
            "draw_record_slots": draw_records[index].tolist(),
            "draw_position_slots": draw_positions[index].tolist(),
            "used_replacement": False,
        }
        for index in range(2)
    ]
    digest = _digest(4005, steps)
    path = tmp_path / "schedule.safetensors"
    save_file(
        {
            "batch_record_indices": batch_rows,
            "draw_position_slots": draw_positions,
            "draw_record_slots": draw_records,
            "used_replacement": torch.zeros(2, dtype=torch.bool),
        },
        str(path),
        metadata={"seed": "4005", "steps": "2"},
    )
    actual, receipt = provider.deserialize_schedule(
        path,
        runner=_runner(),
        expected={"seed": 4005, "steps": 2, "semantic_sha256": digest},
        sequence_tokens=8,
        batch_records=2,
        position_budget=3,
    )
    assert [step.batch_global_rows for step in actual] == [(4, 9), (2, 7)]
    assert receipt["semantic_sha256"] == digest


def test_deserialize_schedule_rejects_binding_digest(tmp_path):
    path = tmp_path / "schedule.safetensors"
    save_file(
        {
            "batch_record_indices": torch.tensor([[0, 1]], dtype=torch.int64),
            "draw_position_slots": torch.tensor([[1, 2]], dtype=torch.int64),
            "draw_record_slots": torch.tensor([[0, 1]], dtype=torch.int64),
            "used_replacement": torch.zeros(1, dtype=torch.bool),
        },
        str(path),
    )
    with pytest.raises(provider.ProviderError, match="semantic digest"):
        provider.deserialize_schedule(
            path,
            runner=_runner(),
            expected={"seed": 4005, "steps": 1, "semantic_sha256": "0" * 64},
            sequence_tokens=4,
            batch_records=2,
            position_budget=2,
        )


class _FixtureSource:
    def __init__(self, batch):
        self.batch = batch

    def batch_for_global_rows(self, rows):
        assert tuple(rows) == (11, 17)
        return self.batch


def test_export_probe_reloads_base_and_effective_readout_exactly(tmp_path):
    torch.manual_seed(1010)
    base = build_residual_mlp512(hidden_size=8, vocabulary_size=17, context_width=4, bottleneck_size=3, seed=4005)
    embedding = torch.nn.functional.normalize(torch.randn(17, 8), dim=-1)
    hook = build_directional_from_base(base, torch.tensor([1, 4, 8]), torch.tensor([1, 4, 16]))
    hook.bind_embedding_statistics(embedding)
    with torch.no_grad():
        hook.delta_rows[0, 0] = 0.025
        hook.delta_rows[1, 3] = -0.015
    batch = SimpleNamespace(
        activations=torch.randn(2, 4, 8, dtype=torch.float32),
        attention_mask=torch.ones(2, 4, dtype=torch.bool),
        token_ids=torch.zeros(2, 4, dtype=torch.long),
        position_ids=torch.arange(4, dtype=torch.long).expand(2, -1),
        global_rows=(11, 17),
    )
    def descriptor(path, status):
        raw = path.read_bytes()
        return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "status": status}
    base_binding_path = tmp_path / "base-binding.bin"
    bank_binding_path = tmp_path / "bank-binding.bin"
    base_binding_path.write_bytes(b"base")
    bank_binding_path.write_bytes(b"bank")
    receipt = {
        "settings": {"probe_steps": 2},
        "artifacts": {
            "base_state": descriptor(base_binding_path, "SELECTED"),
            "bank_manifest": descriptor(bank_binding_path, "VERIFIED"),
        },
    }
    callback = provider._checkpoint_export_factory(
        receipt=receipt,
        contract={"schema": "synthetic"},
        embedding=embedding,
        fixture_source=_FixtureSource(batch),
        fixture_rows=(11, 17),
    )
    result = callback(SimpleNamespace(decoder=base, hook=hook), tmp_path / "out")
    assert result["status"] == "PROBE_EXPORTED_RELOADED_EXACT"
    assert result["reload_check"]["exact_logits"] is True
    assert result["reload_check"]["exact_argmax"] is True
    assert result["reload_check"]["max_abs"] == 0.0
