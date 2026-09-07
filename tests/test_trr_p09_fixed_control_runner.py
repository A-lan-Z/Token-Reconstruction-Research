"""Synthetic tests for the P09 streamed fixed-control runner.

No public bank, checkpoint, model snapshot, source text, truth, or GPU is
opened.  The tests exercise schedule/loader identity, step-zero selection,
H128 validation, and create-only receipts with tiny tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scripts.trr_p09.fixed_control_runner import (
    DOMAIN_BALANCED_SELECTION_METRIC,
    FixedControlRunnerError,
    RandomAccessLoaderSource,
    RunnerConfig,
    SchedulePlan,
    aggregate_domain_validation,
    ScheduleStep,
    build_run_receipt,
    evaluate_batches,
    method_state_digest,
    run_training,
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
        record_state_digest=True,
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
        record_state_digest=True,
    )
    assert result_a.state_sha256 == result_b.state_sha256 == method_state_digest(decoder_a, hook_a)
    assert result_a.token_rows == result_b.token_rows == 3
    assert result_a.correct_tokens == result_b.correct_tokens
    assert result_a.loss == pytest.approx(result_b.loss, abs=1e-7)
    assert result_a.gradient_norm == pytest.approx(result_b.gradient_norm, abs=1e-7)
    for left, right in zip(decoder_a.parameters(), decoder_b.parameters(), strict=True):
        assert torch.equal(left, right)


def test_random_access_source_preserves_cross_shard_order_and_duplicates() -> None:
    class RandomLoader:
        def __init__(self) -> None:
            self.rows = {
                row: _batch(rows=(row,)).activations[0].clone()
                for row in range(6)
            }

        def get_records(self, global_indices):
            requested = tuple(int(row) for row in global_indices)
            return Batch(
                activations=torch.stack([self.rows[row] for row in requested]),
                token_ids=torch.zeros(len(requested), 4, dtype=torch.long),
                attention_mask=torch.ones(len(requested), 4, dtype=torch.bool),
                position_ids=torch.arange(4, dtype=torch.long).expand(len(requested), -1).clone(),
                global_rows=requested,
            )

    source = RandomAccessLoaderSource(RandomLoader())
    batch = source.batch_for_global_rows((5, 0, 5, 2))
    assert batch.global_rows == (5, 0, 5, 2)
    assert torch.equal(batch.activations[0], batch.activations[2])
    with pytest.raises(FixedControlRunnerError, match="global rows"):
        source.batch_for_global_rows((-1, 0))


class _DirectionalTrainableHook(nn.Module):
    method_id = "synthetic_directional"
    readout_mode = "token_direction"

    def __init__(self, vocabulary_size: int = 7, hidden_size: int = 3) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(vocabulary_size, hidden_size))

    def score_rows(self, query_rows, logit_scale, embedding, *, base_logits=None):
        del base_logits
        return (query_rows @ (embedding + self.delta).transpose(0, 1)) * logit_scale

    def loss_terms(self, logits, target_ids):
        total = F.cross_entropy(logits, target_ids)
        return {"cross_entropy": total, "total": total}

    def optimizer_param_groups(self, decoder, *, base_learning_rate):
        return (
            {"name": "decoder", "params": list(decoder.parameters()), "lr": base_learning_rate},
            {"name": "directions", "params": [self.delta], "lr": base_learning_rate},
        )

    def metadata(self):
        return {"method_id": self.method_id}


def test_train_step_clips_and_hashes_decoder_and_directional_parameters() -> None:
    decoder = TinyDecoder()
    hook = _DirectionalTrainableHook()
    source = Source(_batch(rows=(2, 5)))
    step = ScheduleStep(0, (2, 5), (0, 1, 0), (1, 2, 3), False)
    config = _config()
    optimizer = torch.optim.AdamW(
        hook.optimizer_param_groups(decoder, base_learning_rate=config.learning_rate),
        weight_decay=0.0,
    )
    before = hook.delta.detach().clone()
    result = train_one_step(
        decoder,
        hook,
        source,
        step,
        optimizer=optimizer,
        embedding=torch.randn(7, 3),
        config=config,
        activation_dtype=torch.bfloat16,
        compute_base_logits=False,
        record_state_digest=True,
    )
    assert result.state_sha256 == method_state_digest(decoder, hook)
    assert not torch.equal(before, hook.delta.detach())


def test_train_step_rejects_optimizer_omitting_directional_parameters() -> None:
    decoder = TinyDecoder()
    hook = _DirectionalTrainableHook()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    with pytest.raises(FixedControlRunnerError, match="omits trainable"):
        train_one_step(
            decoder,
            hook,
            Source(_batch(rows=(2, 5))),
            ScheduleStep(0, (2, 5), (0, 1, 0), (1, 2, 3), False),
            optimizer=optimizer,
            embedding=torch.randn(7, 3),
            config=_config(),
            activation_dtype=torch.bfloat16,
        )


def test_train_step_rejects_unowned_trainable_optimizer_parameters() -> None:
    decoder = TinyDecoder()
    hook = FixedPublicReadoutHook(method_id="fixed", embedding_sha256=_DIGEST)
    foreign = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW(
        [*decoder.parameters(), foreign],
        lr=0.01,
    )
    with pytest.raises(FixedControlRunnerError, match="outside decoder/readout"):
        train_one_step(
            decoder,
            hook,
            Source(_batch(rows=(2, 5))),
            ScheduleStep(0, (2, 5), (0, 1, 0), (1, 2, 3), False),
            optimizer=optimizer,
            embedding=torch.randn(7, 3),
            config=_config(),
            activation_dtype=torch.bfloat16,
        )


def test_train_step_rejects_a_masked_sampled_position() -> None:
    decoder = TinyDecoder()
    hook = FixedPublicReadoutHook(method_id="fixed", embedding_sha256=_DIGEST)
    batch = _batch(rows=(2, 5))
    batch.attention_mask[0, 1] = False
    batch.position_ids[0, 1] = 0
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    with pytest.raises(FixedControlRunnerError, match="masked/invalid"):
        train_one_step(
            decoder,
            hook,
            Source(batch),
            ScheduleStep(0, (2, 5), (0, 1, 0), (1, 2, 3), False),
            optimizer=optimizer,
            embedding=torch.randn(7, 3),
            config=_config(),
            activation_dtype=torch.bfloat16,
        )


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




def test_equal_domain_validation_changes_pooled_ranking_and_keeps_step_zero_ties() -> None:
    aggregate = aggregate_domain_validation(
        {
            "Finance": {"token_rows": 256, "correct_tokens": 128, "token_accuracy": 0.5},
            "Pile": {"token_rows": 64, "correct_tokens": 64, "token_accuracy": 1.0},
        }
    )
    assert aggregate["token_accuracy"] == pytest.approx(0.6)
    assert aggregate[DOMAIN_BALANCED_SELECTION_METRIC] == pytest.approx(0.75)

    points = [
        {
            "step": 0,
            "validation": {
                "token_accuracy": 0.60,
                DOMAIN_BALANCED_SELECTION_METRIC: 0.75,
            },
        },
        {
            "step": 500,
            "validation": {
                "token_accuracy": 0.62,
                DOMAIN_BALANCED_SELECTION_METRIC: 0.75,
            },
        },
    ]
    assert select_earliest_maximum(points, metric="token_accuracy")["step"] == 500
    assert select_earliest_maximum(points, metric=DOMAIN_BALANCED_SELECTION_METRIC)["step"] == 0


def test_domain_validation_rejects_missing_or_extra_domains() -> None:
    metrics = {"token_rows": 4, "correct_tokens": 2, "token_accuracy": 0.5}
    with pytest.raises(FixedControlRunnerError, match="exactly"):
        aggregate_domain_validation({"Finance": metrics})
    with pytest.raises(FixedControlRunnerError, match="exactly"):
        aggregate_domain_validation({"Finance": metrics, "Pile": metrics, "Other": metrics})


def test_run_training_domain_callback_reuses_per_domain_evaluator_and_serializer_cannot_inject_metric() -> None:
    decoder = TinyDecoder()
    hook = FixedPublicReadoutHook(method_id="fixed", embedding_sha256=_DIGEST)
    source = Source(_batch(rows=(2, 5)))
    embedding = torch.randn(7, 3)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    step = ScheduleStep(0, (2, 5), (0, 1, 0), (1, 2, 3), False)
    plan = SchedulePlan.from_steps(seed=13, steps=(step,))
    config = replace(_config(), selection_metric=DOMAIN_BALANCED_SELECTION_METRIC)
    observed: dict[int, float] = {}

    def validation_callback(step_index, evaluate):
        per_domain = {
            "Finance": evaluate([_batch(rows=(10, 11))]),
            "Pile": evaluate([_batch(rows=(12, 13))]),
        }
        aggregate = aggregate_domain_validation(per_domain)
        observed[step_index] = aggregate[DOMAIN_BALANCED_SELECTION_METRIC]
        return aggregate

    def checkpoint_serializer(point, _decoder, _hook):
        # This mutates only the deep-copied callback view.  The live point used
        # by selection must retain the callback's canonical score.
        point["validation"][DOMAIN_BALANCED_SELECTION_METRIC] = -1.0
        return {"serialized": True}

    result = run_training(
        decoder,
        hook,
        source,
        (step,),
        schedule_steps_count=1,
        schedule_seed=plan.seed,
        schedule_semantic_sha256=plan.semantic_sha256,
        schedule_exposure=plan.exposure_summary(),
        optimizer=optimizer,
        embedding=embedding,
        config=config,
        validation_callback=validation_callback,
        validation_sequence_tokens=4,
        validation_batch_records=2,
        validation_activation_dtype=torch.bfloat16,
        training_activation_dtype=torch.bfloat16,
        checkpoint_steps=(0, 1),
        checkpoint_callback=checkpoint_serializer,
    )
    assert set(observed) == {0, 1}
    for point in result["learning_curve"]:
        assert point["validation"][DOMAIN_BALANCED_SELECTION_METRIC] == pytest.approx(
            observed[point["step"]]
        )
    assert result["checkpoint_state_bindings"] == [{"serialized": True}, {"serialized": True}]


def test_run_training_domain_metric_fails_closed_without_callback() -> None:
    decoder = TinyDecoder()
    hook = FixedPublicReadoutHook(method_id="fixed", embedding_sha256=_DIGEST)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    step = ScheduleStep(0, (2, 5), (0, 1, 0), (1, 2, 3), False)
    plan = SchedulePlan.from_steps(seed=13, steps=(step,))
    with pytest.raises(FixedControlRunnerError, match="explicit validation callback"):
        run_training(
            decoder,
            hook,
            Source(_batch(rows=(2, 5))),
            (step,),
            schedule_steps_count=1,
            schedule_seed=plan.seed,
            schedule_semantic_sha256=plan.semantic_sha256,
            schedule_exposure=plan.exposure_summary(),
            optimizer=optimizer,
            embedding=torch.randn(7, 3),
            config=replace(_config(), selection_metric=DOMAIN_BALANCED_SELECTION_METRIC),
            validation_batches=lambda _step: [_batch(rows=(2, 5))],
            validation_sequence_tokens=4,
            validation_batch_records=2,
            validation_activation_dtype=torch.bfloat16,
            training_activation_dtype=torch.bfloat16,
            checkpoint_steps=(0, 1),
        )

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
