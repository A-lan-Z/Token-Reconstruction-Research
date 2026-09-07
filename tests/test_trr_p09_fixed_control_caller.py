"""Synthetic caller plumbing tests; no public assets, model, truth, or GPU."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from torch import nn

from scripts.trr_p09.fixed_control_caller import (
    DEFAULT_VALIDATION_DOMAINS,
    FixedControlCallerError,
    build_fixed_control_receipt,
    fixed_control_cost_summary,
    inherited_schedule_steps,
    join_public_validation_labels,
    make_domain_validation_callback,
    make_fixed_checkpoint_callback,
    materialize_schedule_plan,
    signed_p09_checkpoint_grid,
)
from scripts.trr_p09.fixed_control_runner import FixedControlRunnerError, SchedulePlan, validate_batch
from token_reconstruction.trr_p09_fixed_control_adapter import (
    AssetBinding,
    BankContract,
    FixedPublicReadoutHook,
)
from token_reconstruction.trr0005_joint_decoder import build_position_schedule


_DIGEST = "a" * 64


def _asset(label: str) -> AssetBinding:
    return AssetBinding(label=label, path=f"/synthetic/{label}", bytes=11, sha256=_DIGEST)


def _bank() -> BankContract:
    return BankContract(
        fit_manifest=_asset("fit-manifest"),
        validation_manifest=_asset("validation-manifest"),
        embedding=_asset("embedding"),
        schedule=_asset("schedule"),
        fit_shape=(12, 192, 4),
        validation_shape=(6, 128, 4),
        fit_mask_shape=(12, 192),
        validation_mask_shape=(6, 128),
        fit_post_bos_rows=120,
        fit_supported_token_count=9,
        schedule_semantic_sha256=_DIGEST,
    )


def test_validation_label_join_partitions_and_callback_preserves_equal_domains() -> None:
    join = join_public_validation_labels(
        [
            {"global_row": 10, "record_id": "f-0", "domain": "Finance"},
            {"global_row": 11, "record_id": "p-0", "domain": "Pile"},
            {"global_row": 12, "record_id": "f-1", "domain": "Finance"},
            {"global_row": 13, "record_id": "p-1", "domain": "Pile"},
        ]
    )
    assert join.required_domains == DEFAULT_VALIDATION_DOMAINS
    assert join.rows_by_domain == {"Finance": (10, 12), "Pile": (11, 13)}
    calls: list[tuple[int, str, tuple[int, ...]]] = []

    def factory(step: int, domain: str, rows: tuple[int, ...]):
        calls.append((step, domain, rows))
        return [domain]

    def evaluate(batches):
        domain = next(iter(batches))
        if domain == "Finance":
            return {"token_rows": 10, "correct_tokens": 2, "token_accuracy": 0.2}
        return {"token_rows": 2, "correct_tokens": 2, "token_accuracy": 1.0}

    callback = make_domain_validation_callback(join, factory)
    result = callback(0, evaluate)
    assert result["domain_balanced_token_accuracy"] == pytest.approx(0.6)
    assert result["token_accuracy"] == pytest.approx(4 / 12)
    assert result["label_join_sha256"] == join.semantic_sha256
    assert calls == [(0, "Finance", (10, 12)), (0, "Pile", (11, 13))]


def test_validation_label_join_rejects_duplicate_or_missing_domain() -> None:
    with pytest.raises(FixedControlCallerError, match="duplicated"):
        join_public_validation_labels(
            [
                {"global_row": 1, "record_id": "same", "domain": "Finance"},
                {"global_row": 1, "record_id": "other", "domain": "Pile"},
            ]
        )
    with pytest.raises(FixedControlCallerError, match="no rows"):
        join_public_validation_labels(
            [{"global_row": 1, "record_id": "only", "domain": "Finance"}]
        )


def test_signed_checkpoint_grid_is_integer_deterministic_and_binds_position_budget() -> None:
    assert signed_p09_checkpoint_grid(124_371) == (0, 1000, 2000, 4000, 8000, 12000)
    assert signed_p09_checkpoint_grid(2_000_001)[-1] == 20_000
    with pytest.raises(FixedControlCallerError, match="512-position"):
        signed_p09_checkpoint_grid(100, position_budget=256)


def test_schedule_matches_inherited_torch_generator_and_plan_digest() -> None:
    valid = torch.ones(10, 9, dtype=torch.bool)
    valid[0, 7:] = False
    valid[4, 4:] = False
    expected = build_position_schedule(
        valid,
        steps=4,
        record_batch_size=3,
        position_budget=5,
        seed=4005,
    )
    actual = tuple(
        inherited_schedule_steps(
            valid,
            tuple(range(10)),
            steps=4,
            seed=4005,
            record_batch_size=3,
            position_budget=5,
        )
    )
    assert [step.batch_global_rows for step in actual] == [
        tuple(int(v) for v in row.tolist()) for row in expected.batch_record_indices
    ]
    assert [step.draw_record_slots for step in actual] == [
        tuple(int(v) for v in row.tolist()) for row in expected.draw_record_slots
    ]
    assert [step.draw_position_slots for step in actual] == [
        tuple(int(v) for v in row.tolist()) for row in expected.draw_position_slots
    ]
    assert [step.used_replacement for step in actual] == [
        bool(v) for v in expected.used_replacement.tolist()
    ]
    plan = materialize_schedule_plan(
        valid,
        tuple(range(10)),
        steps=4,
        seed=4005,
        record_batch_size=3,
        position_budget=5,
    )
    assert isinstance(plan, SchedulePlan)
    assert plan.exposure_summary()["total_draws"] == 20


def test_runner_accepts_zero_position_ids_only_in_inactive_padding() -> None:
    from tests.test_trr_p09_fixed_control_runner import _batch

    batch = _batch(rows=(0, 1), sequence_tokens=5)
    batch.attention_mask[0, 3:] = False
    batch.position_ids[0, 3:] = 0
    validate_batch(
        batch,
        expected_global_rows=(0, 1),
        expected_sequence_tokens=5,
        expected_hidden_size=3,
        expected_batch_records=2,
        expected_activation_dtype=torch.bfloat16,
    )
    batch.position_ids[0, 3] = 3
    with pytest.raises(FixedControlRunnerError, match="active-prefix/zero-padding"):
        validate_batch(
            batch,
            expected_global_rows=(0, 1),
            expected_sequence_tokens=5,
            expected_hidden_size=3,
            expected_batch_records=2,
            expected_activation_dtype=torch.bfloat16,
        )


class _TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))


def _training_result() -> dict:
    return {
        "status": "COMPLETED",
        "selected_step": 1,
        "selected_state_sha256": _DIGEST,
        "schedule": {
            "seed": 4005,
            "steps": 1,
            "semantic_sha256": _DIGEST,
            "exposure": {"draws_per_step": 4, "total_draws": 4},
        },
        "timing": {
            "whole_wall_seconds": 2.0,
            "stream_load_seconds": 0.3,
            "optimizer_update_seconds": 1.2,
            "validation_seconds": 0.5,
            "timing_boundary": "synthetic",
        },
        "learning_curve": [
            {"step": 0, "validation": {"token_rows": 2}},
            {"step": 1, "validation": {"token_rows": 2}},
        ],
        "checkpoint_state_bindings": [{"step": 0}, {"step": 1}],
    }


def test_fixed_checkpoint_serializer_is_create_only_and_bank_bound(tmp_path: Path) -> None:
    decoder = _TinyDecoder()
    hook = FixedPublicReadoutHook(method_id="continued_fixed_readout", embedding_sha256=_DIGEST)
    callback = make_fixed_checkpoint_callback(
        output_root=tmp_path / "states",
        bank_contract=_bank(),
        bank_manifest_sha256="b" * 64,
        base_state_sha256="c" * 64,
        fit_manifest_sha256="d" * 64,
    )
    binding = callback({"step": 0}, decoder, hook)
    checkpoint = binding["checkpoint"]
    assert checkpoint["sha256"] != ""
    assert binding["bank_manifest_sha256"] == "b" * 64
    with safe_open(checkpoint["path"], framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    assert metadata["schema"] == "token-reconstruction.trr-p09-fixed-state.v1"
    assert metadata["bank_manifest_sha256"] == "b" * 64
    with pytest.raises(FixedControlCallerError, match="already exists"):
        callback({"step": 0}, decoder, hook)


def test_fixed_receipt_contains_curve_exposure_cost_and_selected_state(tmp_path: Path) -> None:
    result = _training_result()
    cost = fixed_control_cost_summary(result)
    assert cost["total_position_draws"] == 4
    assert cost["optimizer_steps"] == 1
    receipt = build_fixed_control_receipt(
        training_result=result,
        source_commit="e" * 40,
        command=("python", "fixed_fit.py"),
        started_utc="2026-09-07T00:00:00Z",
        finished_utc="2026-09-07T00:00:02Z",
        environment={"torch": "synthetic"},
        assets={"bank": {"sha256": "b" * 64}},
        training_contract={"steps": 1},
        resource_peak={"host_rss_bytes": 1},
        bank_manifest_sha256="b" * 64,
        state={"selected_checkpoint_sha256": "c" * 64},
    )
    assert receipt["selection"]["metric"] == "domain_balanced_token_accuracy"
    assert receipt["state"]["selected_step"] == 1
    assert receipt["exposure"]["total_draws"] == 4
    output = tmp_path / "run.json"
    from scripts.trr_p09.fixed_control_caller import write_fixed_control_receipt

    write_fixed_control_receipt(output, receipt)
    assert output.exists()
    with pytest.raises(FixedControlCallerError, match="already exists"):
        write_fixed_control_receipt(output, receipt)
