from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from token_reconstruction.trr0005_joint_decoder import (
    AFFINE_METHOD,
    CAUSAL_ATTENTION_METHOD,
    DIAGONAL_ATTENTION_METHOD,
    BOS_TOKEN_ID,
    JointDecoderError,
    build_decoder,
    build_position_schedule,
    checkpoint_steps,
    evaluate_dataset,
    load_public_joint_data,
    schedule_metadata,
    train_step,
)
import trr0005_fit_joint_decoders as runner


def _inputs(*, records: int = 2, positions: int = 6, hidden: int = 8, vocab: int = 17):
    torch.manual_seed(5001)
    activation = torch.randn(records, positions, hidden)
    truth = torch.randint(0, vocab, (records, positions), dtype=torch.long)
    truth[:, 0] = BOS_TOKEN_ID
    mask = torch.ones(records, positions, dtype=torch.bool)
    table = F.normalize(torch.randn(vocab, hidden), dim=-1)
    return activation, truth, mask, table


def test_initialization_is_identity_affine_and_deterministic_qkv() -> None:
    first = build_decoder(
        CAUSAL_ATTENTION_METHOD,
        hidden_size=8,
        vocabulary_size=17,
        seed=4005,
    )
    second = build_decoder(
        CAUSAL_ATTENTION_METHOD,
        hidden_size=8,
        vocabulary_size=17,
        seed=4005,
    )
    assert torch.equal(first.W, torch.eye(8))
    assert torch.equal(first.b, torch.zeros(8))
    assert torch.equal(first.s, torch.tensor(3.0))
    assert torch.equal(first.query.weight, second.query.weight)
    assert torch.equal(first.key.weight, second.key.weight)
    assert torch.equal(first.value.weight, second.value.weight)
    assert torch.equal(first.output.weight, torch.zeros_like(first.output.weight))
    assert torch.equal(first.output.bias, torch.zeros_like(first.output.bias))


@pytest.mark.parametrize("method_id", [AFFINE_METHOD, CAUSAL_ATTENTION_METHOD, DIAGONAL_ATTENTION_METHOD])
def test_zero_output_contextual_initialization_matches_affine(method_id: str) -> None:
    activation, _, mask, table = _inputs()
    affine = build_decoder(AFFINE_METHOD, hidden_size=8, vocabulary_size=17)
    model = build_decoder(method_id, hidden_size=8, vocabulary_size=17)
    with torch.inference_mode():
        expected = affine(activation, mask, table)
        actual = model(activation, mask, table)
    assert torch.equal(actual, expected)


def test_causal_attention_cannot_read_future_activation() -> None:
    activation, _, mask, table = _inputs(records=1, positions=7)
    model = build_decoder(CAUSAL_ATTENTION_METHOD, hidden_size=8, vocabulary_size=17)
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.05)
        model.output.bias.normal_(0.0, 0.05)
    changed = activation.clone()
    changed[:, 4:, :] += 17.0
    with torch.inference_mode():
        before = model(activation, mask, table)
        after = model(changed, mask, table)
    assert torch.equal(before[:, :4], after[:, :4])
    assert not torch.equal(before[:, 4:], after[:, 4:])


def test_diagonal_attention_reads_current_h_but_no_earlier_h() -> None:
    activation, _, mask, table = _inputs(records=1, positions=7)
    model = build_decoder(DIAGONAL_ATTENTION_METHOD, hidden_size=8, vocabulary_size=17)
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.05)
        model.output.bias.normal_(0.0, 0.05)
    changed = activation.clone()
    changed[:, 1, :] += 17.0
    with torch.inference_mode():
        before = model(activation, mask, table)
        after = model(changed, mask, table)
    assert torch.equal(before[:, 2:], after[:, 2:])
    assert not torch.equal(before[:, 1], after[:, 1])


def test_diagonal_qk_gradients_are_zero_but_value_output_can_train() -> None:
    activation, truth, mask, table = _inputs(records=1, positions=6, hidden=8, vocab=17)
    model = build_decoder(DIAGONAL_ATTENTION_METHOD, hidden_size=8, vocabulary_size=17)
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.05)
        model.output.bias.normal_(0.0, 0.05)
    selected = mask.clone()
    selected[:, 0] = False
    logits = model.selected_logits(activation, mask, selected, table)
    loss = F.cross_entropy(logits, truth[selected])
    loss.backward()
    assert model.query.weight.grad is not None
    assert model.key.weight.grad is not None
    assert torch.equal(model.query.weight.grad, torch.zeros_like(model.query.weight.grad))
    assert torch.equal(model.key.weight.grad, torch.zeros_like(model.key.weight.grad))
    assert float(model.value.weight.grad.norm()) > 0.0
    assert float(model.output.weight.grad.norm()) > 0.0


def test_joint_attention_base_parameters_receive_gradient_and_update() -> None:
    activation, truth, mask, table = _inputs(records=2, positions=6, hidden=8, vocab=17)
    schedule = build_position_schedule(mask, steps=1, record_batch_size=2, position_budget=4, seed=4005)
    model = build_decoder(CAUSAL_ATTENTION_METHOD, hidden_size=8, vocabulary_size=17)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    before = model.W.detach().clone()
    point = train_step(
        model,
        activation,
        truth,
        mask,
        table,
        schedule,
        0,
        device=torch.device("cpu"),
        optimizer=optimizer,
        gradient_clip_norm=1.0,
    )
    assert point["draws"] == 4
    assert model.W.grad is not None
    assert float(model.W.grad.norm()) > 0.0
    assert not torch.equal(before, model.W.detach())


def test_schedule_has_exact_draw_count_and_records_replacement() -> None:
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    schedule = build_position_schedule(
        mask,
        steps=3,
        record_batch_size=2,
        position_budget=5,
        seed=4005,
    )
    assert schedule.total_draws == 15
    metadata = schedule_metadata(schedule)
    assert metadata["expected_total_draws"] == 15
    assert metadata["unique_draws_within_step"] <= 15
    assert metadata["repeated_draws_within_step"] == metadata["replacement_repeated_draws"]
    assert bool(schedule.used_replacement.all().item())
    assert schedule.draw_position_slots.ne(0).all().item()


def test_schedule_is_seed_bound() -> None:
    mask = torch.ones(8, 192, dtype=torch.bool)
    first = build_position_schedule(mask, steps=4, record_batch_size=8, position_budget=512, seed=4005)
    second = build_position_schedule(mask, steps=4, record_batch_size=8, position_budget=512, seed=4005)
    assert torch.equal(first.batch_record_indices, second.batch_record_indices)
    assert torch.equal(first.draw_record_slots, second.draw_record_slots)
    assert torch.equal(first.draw_position_slots, second.draw_position_slots)


def test_checkpoint_schedule_contains_required_early_and_regular_points() -> None:
    assert checkpoint_steps(3000) == tuple([0, 25, 50, 75, 100, 150, 200] + list(range(300, 3001, 100)))


def test_runner_rejects_recipe_changes() -> None:
    args = runner._parser().parse_args(
        [
            "--original-manifest", "original.json",
            "--enriched-manifest", "enriched.json",
            "--output-root", "out",
            "--learning-rate", "0.0005",
        ]
    )
    with pytest.raises(runner.JointFitRunnerError, match="learning rate"):
        runner._validate_args(args)


def test_resource_preflight_records_largest_geometry() -> None:
    result = runner.resource_preflight(
        hidden_size=2048,
        vocabulary_size=128256,
        sequence_length=192,
        record_batch_size=8,
        position_budget=512,
        context_width=128,
    )
    assert result["geometry"] == {
        "hidden_size": 2048,
        "vocabulary_size": 128256,
        "sequence_length": 192,
        "record_batch_size": 8,
        "position_budget": 512,
        "context_width": 128,
    }
    assert result["bytes"]["selected_logits_fp32"] == 512 * 128256 * 4
    assert result["bytes"]["conservative_envelope"] > result["bytes"]["raw_sum"]
    assert result["bytes"]["measured_v1_qualification_peak"] == 2_942_304_256
    assert result["bytes"]["measured_qualification_floor"] == 4_413_456_384
    assert result["bytes"]["conservative_envelope"] == 4_413_456_384
    assert result["forecast_basis"]["measured_floor_source"].endswith(
        "joint_qualification_v1/failure.json"
    )


def test_evaluator_bookkeeping_preserves_histogram_and_exact_counts() -> None:
    torch.manual_seed(5017)
    model = build_decoder(AFFINE_METHOD, hidden_size=4, vocabulary_size=7)
    observations = torch.randn(2, 5, 4)
    truth = torch.tensor(
        [[BOS_TOKEN_ID, 1, 1, 2, 3], [BOS_TOKEN_ID, 1, 2, 2, 0]],
        dtype=torch.long,
    )
    valid_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]],
        dtype=torch.bool,
    )
    embedding = torch.randn(7, 4)
    reference = truth[:, 1:][valid_mask[:, 1:]]
    metrics = evaluate_dataset(
        model,
        observations,
        truth,
        valid_mask,
        embedding,
        ("original", "enriched"),
        device=torch.device("cpu"),
        record_batch_size=1,
        position_budget=2,
        frequency_reference=reference,
    )
    assert metrics["token_rows"] == 7
    assert metrics["record_token_rows"] == [4, 3]
    assert metrics["exact_records"] == 0
    assert metrics["projection_calls"] == 4
    assert metrics["projection_rows"] == 7
    assert sum(item["rows"] for item in metrics["frequency_bucket_metrics"].values()) == 7
    assert sum(item["rows"] for item in metrics["position_bucket_metrics"].values()) == 7
    assert sum(
        item["rows"]
        for frequency in metrics["frequency_by_position_bucket"].values()
        for item in frequency.values()
    ) == 7
    assert metrics["token_accuracy"] == pytest.approx(2 / 7)


def test_qualification_cli_is_bounded_and_does_not_require_retained_state() -> None:
    args = runner._parser().parse_args(
        [
            "--original-manifest", "original.json",
            "--enriched-manifest", "enriched.json",
            "--output-root", "out",
            "--qualification-only",
            "--qualification-steps", "2",
        ]
    )
    runner._validate_args(args)
    assert args.qualification_only is True
    assert args.preflight_only is False
    assert args.qualification_steps == 2
    assert args.retained_affine_state is None


def test_sampler_receipt_rejects_cross_distribution_mask_or_draw_changes() -> None:
    original_mask = torch.ones(2, 5, dtype=torch.bool)
    enriched_mask = original_mask.clone()
    enriched_mask[1, 4] = False
    original_schedule = build_position_schedule(
        original_mask, steps=2, record_batch_size=2, position_budget=3, seed=4005
    )
    enriched_schedule = build_position_schedule(
        enriched_mask, steps=2, record_batch_size=2, position_budget=3, seed=4005
    )
    original_data = SimpleNamespace(
        fit_valid_mask=original_mask,
        fit_observations=torch.zeros(2, 5, 4),
        fit_record_ids=("original-0", "original-1"),
    )
    enriched_data = SimpleNamespace(
        fit_valid_mask=enriched_mask,
        fit_observations=torch.zeros(2, 5, 4),
        fit_record_ids=("enriched-0", "enriched-1"),
    )
    original_receipt = runner._sampler_receipt("original", original_data, original_schedule)
    enriched_receipt = runner._sampler_receipt("enriched", enriched_data, enriched_schedule)
    with pytest.raises(runner.JointFitRunnerError, match="sampler mask/vector/draw mismatch"):
        runner._compare_sampler_receipts([original_receipt, enriched_receipt])


def _write_split_manifest(
    root: Path,
    *,
    prefix: str,
    observations: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    record_ids: list[str],
    embedding: torch.Tensor,
) -> Path:
    artifact = root / f"{prefix}.safetensors"
    save_file(
        {
            "activations": observations,
            "token_ids": truth,
            "attention_mask": mask.to(torch.uint8),
            "embeddings": embedding,
        },
        str(artifact),
    )
    records = root / f"{prefix}_records.json"
    records.write_text(
        json.dumps({"records": [{"record_id": rid, "style": "pile"} for rid in record_ids]}),
        encoding="utf-8",
    )
    resources = {
        f"{prefix}_observations": {"path": artifact.name, "tensor_key": "activations"},
        f"{prefix}_truth": {"path": artifact.name, "tensor_key": "token_ids"},
        f"{prefix}_valid_mask": {"path": artifact.name, "tensor_key": "attention_mask"},
        f"{prefix}_records": {"path": records.name},
        "embedding_table": {"path": artifact.name, "tensor_key": "embeddings"},
    }
    manifest = root / f"{prefix}_manifest.json"
    manifest.write_text(
        json.dumps({"schema": "token-reconstruction.trr0005-public-fit-data.v1", "resources": resources}),
        encoding="utf-8",
    )
    return manifest


def test_public_loader_preserves_current_token_alignment_and_split(tmp_path: Path) -> None:
    hidden, vocab, positions = 4, 11, 5
    fit_x = torch.randn(2, positions, hidden)
    fit_y = torch.tensor([[BOS_TOKEN_ID, 1, 2, 3, 4], [BOS_TOKEN_ID, 5, 6, 7, 8]], dtype=torch.int32)
    fit_m = torch.ones(2, positions, dtype=torch.bool)
    val_x = torch.randn(1, positions, hidden)
    val_y = torch.tensor([[BOS_TOKEN_ID, 9, 10, 1, 2]], dtype=torch.int32)
    val_m = torch.ones(1, positions, dtype=torch.bool)
    table = F.normalize(torch.randn(vocab, hidden), dim=-1)
    fit_manifest = _write_split_manifest(
        tmp_path, prefix="fit", observations=fit_x, truth=fit_y, mask=fit_m,
        record_ids=["fit-0", "fit-1"], embedding=table,
    )
    val_manifest = _write_split_manifest(
        tmp_path, prefix="validation", observations=val_x, truth=val_y, mask=val_m,
        record_ids=["val-0"], embedding=table,
    )
    data = load_public_joint_data(fit_manifest, val_manifest)
    assert data.fit_record_ids == ("fit-0", "fit-1")
    assert data.validation_record_ids == ("val-0",)
    assert data.validation_groups == ("pile",)
    assert data.fit_truth[0, 1].item() == 1
    assert data.fit_truth[0, 0].item() == BOS_TOKEN_ID


def test_public_loader_rejects_fit_validation_overlap(tmp_path: Path) -> None:
    hidden, vocab, positions = 4, 11, 5
    table = F.normalize(torch.randn(vocab, hidden), dim=-1)
    x = torch.randn(1, positions, hidden)
    y = torch.tensor([[BOS_TOKEN_ID, 1, 2, 3, 4]], dtype=torch.int32)
    mask = torch.ones(1, positions, dtype=torch.bool)
    fit_manifest = _write_split_manifest(
        tmp_path, prefix="fit", observations=x, truth=y, mask=mask,
        record_ids=["same"], embedding=table,
    )
    val_manifest = _write_split_manifest(
        tmp_path, prefix="validation", observations=x, truth=y, mask=mask,
        record_ids=["same"], embedding=table,
    )
    with pytest.raises(JointDecoderError, match="overlap"):
        load_public_joint_data(fit_manifest, val_manifest)
