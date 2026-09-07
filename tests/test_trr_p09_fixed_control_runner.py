"""Synthetic tests for the P09 streamed fixed-control runner.

No public bank, checkpoint, model snapshot, source text, truth, or GPU is
opened.  The tests exercise schedule/loader identity, step-zero selection,
H128 validation, and create-only receipts with tiny tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scripts.trr_p09.fixed_control_runner import (
    FixedControlRunnerError,
    RunnerConfig,
    SchedulePlan,
    ScheduleStep,
    build_run_receipt,
    evaluate_batches,
    select_earliest_maximum,
    train_one_step,
    write_create_only_json,
)
from token_reconstruction.trr_p09_fixed_control_adapter import FixedPublicReadoutHook


_DIGEST = "a" * 64


@dataclass
class Batch:
    activations: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    global_rows: tuple[int, ...]


class Source:
    def __init__(self, batch: Batch) -> None:
        self.batch = batch

    def batch_for_global_rows(self, global_rows):
        expected = tuple(int(row) for row in global_rows)
        if expected != self.batch.global_rows:
            raise FixedControlRunnerError("synthetic row mismatch")
        return Batch(
            activations=self.batch.activations.clone(),
            token_ids=self.batch.token_ids.clone(),
            attention_mask=self.batch.attention_mask.clone(),
            position_ids=self.batch.position_ids.clone(),
            global_rows=self.batch.global_rows,
        )


class TinyDecoder(nn.Module):
    def __init__(self, hidden_size: int = 3) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def projected_hidden(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        projected = F.normalize(self.projection(activation), dim=-1)
        return torch.where(valid_mask.unsqueeze(-1), projected, torch.zeros_like(projected))

    def logits_from_rows(
        self,
        projected_hidden: torch.Tensor,
        record_slots: torch.Tensor,
        position_slots: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        rows = projected_hidden[record_slots, position_slots]
        return rows @ embedding_table.transpose(0, 1) * self.logit_scale


def _batch(*, rows: tuple[int, ...], sequence_tokens: int = 4) -> Batch:
    records = len(rows)
    activations = torch.arange(records * sequence_tokens * 3, dtype=torch.float32).reshape(
        records, sequence_tokens, 3
    ).to(dtype=torch.bfloat16)
    token_ids = torch.arange(records * sequence_tokens, dtype=torch.long).reshape(
        records, sequence_tokens
    ) % 7
    attention_mask = torch.ones(records, sequence_tokens, dtype=torch.bool)
    position_ids = torch.arange(sequence_tokens, dtype=torch.long).expand(records, -1).clone()
    return Batch(activations, token_ids, attention_mask, position_ids, rows)


def _config() -> RunnerConfig:
    return RunnerConfig(
        steps=1,
        record_batch_size=2,
        position_budget=3,
        validation_every=1,
        selection_metric="token_accuracy",
        seed=13,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        train_sequence_tokens=4,
        hidden_size=3,
        expected_activation_dtype=torch.bfloat16.__str__(),
    )


def test_schedule_exposure_digest_and_repeated_pairs_are_structured() -> None:
    steps = (
        ScheduleStep(0, (2, 5), (0, 1, 0), (1, 2, 1), False),
        ScheduleStep(1, (2, 5), (1, 0, 1), (2, 1, 2), False),
    )
    plan = SchedulePlan.from_steps(seed=13, steps=steps)
    plan.validate(record_batch_size=2, position_budget=3, sequence_tokens=4)
    exposure = plan.exposure_summary()
    assert exposure["total_draws"] == 6
    assert exposure["unique_record_position_pairs"] == 2
    assert exposure["repeated_record_position_pairs"] == 2
    assert exposure["max_exposures_per_record_position"] == 3


def test_same_batch_streaming_matches_inherited_one_step_update() -> None:
    batch = _batch(rows=(2, 5))
    source_a = Source(batch)
    source_b = Source(batch)
    decoder_a = TinyDecoder()
    decoder_b = TinyDecoder()
    decoder_b.load_state_dict(decoder_a.state_dict())
    hook_a = FixedPublicReadoutHook(method_id="fixed-a", embedding_sha256=_DIGEST)
    hook_b = FixedPublicReadoutHook(method_id="fixed-b", embedding_sha256=_DIGEST)
    embedding = torch.randn(7, 3)
    optimizer_a = torch.optim.AdamW(decoder_a.parameters(), lr=0.01)
    optimizer_b = torch.optim.AdamW(decoder_b.parameters(), lr=0.01)
    step = ScheduleStep(0, (2, 5), (0, 1, 0), (1, 2, 3), False)
    config = _config()
    result_a = train_one_step(
        decoder_a,
        hook_a,
        source_a,
        step,
        optimizer=optimizer_a,
        embedding=embedding,
        config=config,
        activation_dtype=torch.bfloat16,
    )
    result_b = train_one_step(
        decoder_b,
        hook_b,
        source_b,
        step,
        optimizer=optimizer_b,
        embedding=embedding,
        config=config,
        activation_dtype=torch.bfloat16,
    )
    assert result_a.state_sha256 == result_b.state_sha256
    assert result_a.token_rows == result_b.token_rows == 3
    assert result_a.correct_tokens == result_b.correct_tokens
    assert result_a.loss == pytest.approx(result_b.loss, abs=1e-7)
    assert result_a.gradient_norm == pytest.approx(result_b.gradient_norm, abs=1e-7)
    for left, right in zip(decoder_a.parameters(), decoder_b.parameters(), strict=True):
        assert torch.equal(left, right)


def test_h128_validation_is_independent_from_fit_width() -> None:
    decoder = TinyDecoder()
    hook = FixedPublicReadoutHook(method_id="fixed", embedding_sha256=_DIGEST)
    batch = _batch(rows=(0, 1), sequence_tokens=128)
    result = evaluate_batches(
        decoder,
        hook,
        [batch],
        embedding=torch.randn(7, 3),
        expected_sequence_tokens=128,
        expected_hidden_size=3,
        expected_batch_records=2,
        position_budget=17,
        activation_dtype=torch.bfloat16,
    )
    assert result["token_rows"] == 2 * 127
    assert result["compute_base_logits"] is True


def test_earliest_maximum_keeps_step_zero_eligible() -> None:
    selected = select_earliest_maximum(
        [
            {"step": 0, "validation": {"token_accuracy": 0.40}},
            {"step": 500, "validation": {"token_accuracy": 0.80}},
            {"step": 1000, "validation": {"token_accuracy": 0.80}},
        ],
        metric="token_accuracy",
    )
    assert selected["step"] == 500
    with pytest.raises(FixedControlRunnerError, match="step zero"):
        select_earliest_maximum([{"step": 500, "token_accuracy": 1.0}], metric="token_accuracy")


def test_create_only_receipt_contains_cost_state_and_curve(tmp_path: Path) -> None:
    receipt = build_run_receipt(
        source_commit="source-commit",
        command=["python", "runner.py", "--mode", "synthetic"],
        started_utc="2026-09-07T00:00:00Z",
        finished_utc="2026-09-07T00:00:01Z",
        environment={"device": "cpu"},
        assets={"embedding": {"sha256": _DIGEST}},
        training_contract={"steps": 1, "record_batch_size": 2},
        timing={"preparation_seconds": 0.1, "fit_seconds": 0.2, "validation_seconds": 0.3},
        resource_peak={"host_rss_bytes": 123},
        exposure={"total_draws": 3},
        state={"selected_step": 0, "sha256": _DIGEST},
        learning_curve=[{"step": 0, "token_accuracy": 0.5}],
        status="SYNTHETIC_ONLY",
    )
    output = tmp_path / "receipt.json"
    write_create_only_json(output, receipt)
    assert output.exists()
    assert '"learning_curve"' in output.read_text(encoding="utf-8")
    with pytest.raises(FixedControlRunnerError, match="create-only"):
        write_create_only_json(output, receipt)
