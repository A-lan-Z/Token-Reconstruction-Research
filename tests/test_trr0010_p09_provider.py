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
    assert result["training_checkpoint"]["path"].endswith("training_checkpoint_step_000002.safetensors")
    assert result["training_checkpoint"]["bytes"] > 0


def test_manifest_validation_views_gather_interleaved_rows_from_separate_files(tmp_path):
    """Validation must preserve declared domain order across separately bound files."""

    record_count = 6
    sequence_tokens = 4
    hidden_size = 2

    observations = torch.stack(
        [torch.full((sequence_tokens, hidden_size), float(10 + row)) for row in range(record_count)]
    )
    token_ids = torch.stack(
        [torch.full((sequence_tokens,), 100 + row, dtype=torch.long) for row in range(record_count)]
    )
    masks = torch.ones(record_count, sequence_tokens, dtype=torch.bool)
    paths_and_keys = (
        ("h.safetensors", "h", observations),
        ("labels.safetensors", "labels", token_ids),
        ("mask.safetensors", "mask", masks),
    )

    def descriptor(relative_name, tensor_key, tensor):
        path = tmp_path / relative_name
        save_file({tensor_key: tensor}, str(path))
        raw = path.read_bytes()
        return {
            "path": relative_name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "tensor_key": tensor_key,
            "shape": list(tensor.shape),
        }

    resources = {
        "validation_observations": descriptor(*paths_and_keys[0][:2], paths_and_keys[0][2]),
        "validation_truth": descriptor(*paths_and_keys[1][:2], paths_and_keys[1][2]),
        "validation_valid_mask": descriptor(*paths_and_keys[2][:2], paths_and_keys[2][2]),
    }
    manifest_payload = {
        "resources": resources,
        "validation_grouping": {
            "record_count": record_count,
            "groups_in_record_order": ["Finance", "Pile", "Finance", "Pile", "Finance", "Pile"],
            "post_bos_positions_by_style": {"Finance": 9, "Pile": 9},
        },
    }
    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")
    manifest_raw = manifest_path.read_bytes()
    manifest_descriptor = {
        "path": str(manifest_path),
        "bytes": len(manifest_raw),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }

    views = provider._manifest_validation_views(
        manifest_descriptor,
        expected_tokens=sequence_tokens,
        expected_batch=1,
    )
    assert {domain: len(view.record_indices) for domain, view in views.items()} == {
        "Finance": 3,
        "Pile": 3,
    }
    for domain, expected_rows in {"Finance": (0, 2, 4), "Pile": (1, 3, 5)}.items():
        batches = list(views[domain].batches())
        assert [batch.global_rows[0] for batch in batches] == list(expected_rows)
        for batch in batches:
            row = batch.global_rows[0]
            assert torch.equal(batch.activations[0], observations[row])
            assert torch.equal(batch.token_ids[0], token_ids[row])
            assert torch.equal(batch.attention_mask[0], masks[row])
            assert torch.equal(batch.position_ids[0], torch.arange(sequence_tokens))


def test_p09_validation_manifest_maps_global_rows_to_domain_local_payloads(tmp_path):
    """The P09 join must not use Pile global rows as local label indices."""

    sequence_tokens = 3

    def write_tensor_file(name, tensors):
        path = tmp_path / name
        save_file(tensors, str(path))
        raw = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    finance_h = write_tensor_file(
        "finance_h.safetensors",
        {"activations": torch.stack([
            torch.full((sequence_tokens, 2), 10.0),
            torch.full((sequence_tokens, 2), 11.0),
        ])},
    )
    pile_h = write_tensor_file(
        "pile_h.safetensors",
        {"activations": torch.stack([
            torch.full((sequence_tokens, 2), 20.0),
            torch.full((sequence_tokens, 2), 21.0),
        ])},
    )
    finance_labels = write_tensor_file(
        "finance_labels.safetensors",
        {
            "token_ids": torch.tensor([[100, 101, 102], [110, 111, 112]], dtype=torch.int32),
            "attention_mask": torch.ones(2, sequence_tokens, dtype=torch.uint8),
            "position_ids": torch.arange(sequence_tokens, dtype=torch.long).expand(2, -1).contiguous(),
        },
    )
    pile_labels = write_tensor_file(
        "pile_labels.safetensors",
        {
            "token_ids": torch.tensor([[200, 201, 202], [210, 211, 212]], dtype=torch.int32),
            "attention_mask": torch.ones(2, sequence_tokens, dtype=torch.uint8),
            "position_ids": torch.arange(sequence_tokens, dtype=torch.long).expand(2, -1).contiguous(),
        },
    )

    def h_entry(domain, global_row, observation_row, record_id, descriptor):
        return {
            "domain": domain,
            "global_row": global_row,
            "observation_row": observation_row,
            "record_id": record_id,
            "h_path": descriptor["path"],
            "h_bytes": descriptor["bytes"],
            "h_sha256": descriptor["sha256"],
            "activations_key": "activations",
            "attention_mask_key": "attention_mask",
            "position_ids_key": "position_ids",
        }

    rows = [
        {"domain": "Finance", "global_row": 0, "record_id": "finance-0"},
        {"domain": "Finance", "global_row": 1, "record_id": "finance-1"},
        {"domain": "Pile", "global_row": 2, "record_id": "pile-0"},
        {"domain": "Pile", "global_row": 3, "record_id": "pile-1"},
    ]
    rows_path = tmp_path / "validation_rows.json"
    rows_path.write_text(json.dumps({"rows": rows}, sort_keys=True), encoding="utf-8")
    rows_raw = rows_path.read_bytes()
    rows_descriptor = {
        "path": str(rows_path),
        "bytes": len(rows_raw),
        "sha256": hashlib.sha256(rows_raw).hexdigest(),
    }
    manifest_payload = {
        "schema": "token-reconstruction.trr-p09-public-validation-preparation.v1",
        "task_id": "TRR-P09",
        "truth_opened": False,
        "domains": ["Finance", "Pile"],
        "hidden_size": 2,
        "sequence_tokens_including_bos": sequence_tokens,
        "record_count": 4,
        "records_by_domain": {"Finance": 2, "Pile": 2},
        "observation_h_join": [
            h_entry("Finance", 0, 0, "finance-0", finance_h),
            h_entry("Finance", 1, 1, "finance-1", finance_h),
            h_entry("Pile", 2, 0, "pile-0", pile_h),
            h_entry("Pile", 3, 1, "pile-1", pile_h),
        ],
        "label_join": {
            "rows_by_domain": {"Finance": [0, 1], "Pile": [2, 3]},
        },
        "payloads": {
            "Finance": {"file": finance_labels, "shape": [2, sequence_tokens], "tensor_keys": ["token_ids", "attention_mask", "position_ids"]},
            "Pile": {"file": pile_labels, "shape": [2, sequence_tokens], "tensor_keys": ["token_ids", "attention_mask", "position_ids"]},
        },
        "rows": rows_descriptor,
    }
    manifest_path = tmp_path / "p09_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")
    manifest_raw = manifest_path.read_bytes()
    manifest_descriptor = {
        "path": str(manifest_path),
        "bytes": len(manifest_raw),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }

    views = provider._manifest_validation_views(
        manifest_descriptor,
        expected_tokens=sequence_tokens,
        expected_batch=2,
    )
    finance = list(views["Finance"].batches())
    pile = list(views["Pile"].batches())
    assert [batch.global_rows for batch in finance] == [(0, 1)]
    assert [batch.global_rows for batch in pile] == [(2, 3)]
    assert torch.equal(finance[0].activations[:, 0, 0], torch.tensor([10.0, 11.0]))
    assert torch.equal(pile[0].activations[:, 0, 0], torch.tensor([20.0, 21.0]))
    assert torch.equal(finance[0].token_ids[:, 0], torch.tensor([100, 110]))
    assert torch.equal(pile[0].token_ids[:, 0], torch.tensor([200, 210]))
    assert torch.equal(pile[0].position_ids, torch.arange(sequence_tokens).expand(2, -1))
