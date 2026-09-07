from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.trr_p08 import fit_staged as runner
from token_reconstruction.trr0005_joint_decoder import PublicJointData, build_position_schedule


def _small_models():
    return (
        runner.build_arm_model(
            runner.ARM_BY_ID[runner.P08_POSITIONWISE_JOINT],
            seed=6106,
            hidden_size=8,
            vocabulary_size=19,
            context_width=4,
        ),
        runner.build_arm_model(
            runner.ARM_BY_ID[runner.P08_PAST_ONLY_JOINT],
            seed=6106,
            hidden_size=8,
            vocabulary_size=19,
            context_width=4,
        ),
    )


def _small_batch(records: int = 10, positions: int = 6, hidden: int = 8, vocab: int = 19):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(8012)
    observations = torch.randn(records, positions, hidden, generator=generator)
    truth = torch.randint(vocab, (records, positions), generator=generator)
    valid = torch.ones(records, positions, dtype=torch.bool)
    table = F.normalize(torch.randn(vocab, hidden, generator=generator), dim=-1)
    return observations, truth, valid, table


def test_standard_initialization_is_shared_across_visibility_arms():
    positionwise, past = _small_models()
    assert positionwise.direct_init_label == "p08_standard_identity_affine"
    assert past.direct_init_label == "p08_standard_identity_affine"
    assert set(positionwise.state_dict()) == set(past.state_dict())
    for name, value in positionwise.state_dict().items():
        assert torch.equal(value, past.state_dict()[name]), name


def test_training_phase_freezes_only_the_added_correction_path():
    positionwise, _ = _small_models()
    runner.set_training_phase(positionwise, "affine_only")
    assert all(getattr(positionwise, name).requires_grad for name in ("W", "b", "s"))
    assert all(
        not parameter.requires_grad
        for name, parameter in positionwise.named_parameters()
        if name.split(".", 1)[0] in {"query", "key", "value", "output"}
    )
    assert torch.equal(positionwise.output.weight, torch.zeros_like(positionwise.output.weight))
    assert torch.equal(positionwise.output.bias, torch.zeros_like(positionwise.output.bias))
    runner.set_training_phase(positionwise, "full")
    assert all(parameter.requires_grad for parameter in positionwise.parameters())


def test_shared_schedule_and_phase_boundary_are_deterministic():
    valid = torch.ones(10, 6, dtype=torch.bool)
    first = build_position_schedule(valid, steps=4, record_batch_size=4, position_budget=8, seed=6106)
    second = build_position_schedule(valid, steps=4, record_batch_size=4, position_budget=8, seed=6106)
    assert runner.schedule_digest(first) == runner.schedule_digest(second)
    assert torch.equal(first.batch_record_indices, second.batch_record_indices)
    assert torch.equal(first.draw_position_slots, second.draw_position_slots)
    staged = runner.ARM_BY_ID[runner.P08_POSITIONWISE_STAGED]
    joint = runner.ARM_BY_ID[runner.P08_POSITIONWISE_JOINT]
    assert runner._phase_for_update(staged, 0) == "affine_only"
    assert runner._phase_for_update(staged, runner.STAGED_AFFINE_STEPS - 1) == "affine_only"
    assert runner._phase_for_update(staged, runner.STAGED_AFFINE_STEPS) == "full"
    assert runner._phase_for_update(joint, 0) == "full"


def test_train_step_keeps_correction_parameters_out_of_affine_phase():
    observations, truth, valid, table = _small_batch()
    model, _ = _small_models()
    schedule = build_position_schedule(valid, steps=2, record_batch_size=4, position_budget=8, seed=6106)
    optimizer = torch.optim.AdamW(model.parameters(), lr=runner.LEARNING_RATE, weight_decay=runner.WEIGHT_DECAY)
    runner.set_training_phase(model, "affine_only")
    affine_result = runner.train_step(
        model,
        observations,
        truth,
        valid,
        table,
        schedule,
        0,
        device=torch.device("cpu"),
        optimizer=optimizer,
        gradient_clip_norm=runner.GRADIENT_CLIP_NORM,
    )
    assert affine_result["draws"] == 8
    assert model.output.weight not in optimizer.state
    assert model.output.weight.grad is None
    runner.set_training_phase(model, "full")
    full_result = runner.train_step(
        model,
        observations,
        truth,
        valid,
        table,
        schedule,
        1,
        device=torch.device("cpu"),
        optimizer=optimizer,
        gradient_clip_norm=runner.GRADIENT_CLIP_NORM,
    )
    assert full_result["draws"] == 8
    assert model.output.weight in optimizer.state
    assert torch.isfinite(model.output.weight).all()


def test_row_prediction_uses_original_record_indices_for_sparse_cohort():
    observations, truth, valid, table = _small_batch(records=4)
    model, _ = _small_models()
    with torch.inference_mode():
        hidden = model.direct_pre_normalized_hidden(observations, valid)
        rows = torch.tensor([1, 3], dtype=torch.long)
        positions = torch.tensor([2, 3], dtype=torch.long)
        predicted = model.logits_from_rows(
            F.normalize(hidden, dim=-1), rows, positions, table
        ).argmax(dim=-1).tolist()
    # Make the requested sparse rows correct while the local slots that the
    # old implementation used would be wrong.
    truth[1, 2] = predicted[0]
    truth[0, 2] = (predicted[0] + 1) % table.shape[0]
    truth[3, 3] = predicted[1]
    truth[1, 3] = (predicted[1] + 1) % table.shape[0]
    data = PublicJointData(
        fit_observations=observations,
        fit_truth=truth,
        fit_valid_mask=valid,
        fit_record_ids=("r0", "r1", "r2", "r3"),
        validation_observations=observations[:1],
        validation_truth=truth[:1],
        validation_valid_mask=valid[:1],
        validation_record_ids=("v0",),
        validation_groups=("g",),
        embedding_table=table,
        metadata={},
    )
    result = runner._row_predictions(
        model,
        data,
        table,
        [
            {"record_index": 1, "record_id": "r1", "position": 2, "bin": "1-15"},
            {"record_index": 3, "record_id": "r3", "position": 3, "bin": "1-15"},
        ],
        split="fit",
        device=torch.device("cpu"),
        direct_only=True,
    )
    assert [row["correct"] for row in result] == [True, True]


def test_row_prediction_chunks_cohorts_larger_than_readout_budget():
    observations, truth, valid, table = _small_batch(records=4)
    model, _ = _small_models()
    rows = [
        {"record_index": 1, "record_id": "r1", "position": 2, "bin": "1-15"}
        for _ in range(runner.POSITION_BUDGET + 17)
    ]
    data = PublicJointData(
        fit_observations=observations,
        fit_truth=truth,
        fit_valid_mask=valid,
        fit_record_ids=("r0", "r1", "r2", "r3"),
        validation_observations=observations[:1],
        validation_truth=truth[:1],
        validation_valid_mask=valid[:1],
        validation_record_ids=("v0",),
        validation_groups=("g",),
        embedding_table=table,
        metadata={},
    )
    result = runner._row_predictions(
        model,
        data,
        table,
        rows,
        split="fit",
        device=torch.device("cpu"),
        direct_only=True,
    )
    assert len(result) == runner.POSITION_BUDGET + 17


def test_diagnostic_guard_callback_covers_batches_and_chunks():
    observations, truth, valid, table = _small_batch(records=4)
    model, _ = _small_models()
    data = PublicJointData(
        fit_observations=observations,
        fit_truth=truth,
        fit_valid_mask=valid,
        fit_record_ids=("r0", "r1", "r2", "r3"),
        validation_observations=observations,
        validation_truth=truth,
        validation_valid_mask=valid,
        validation_record_ids=("v0", "v1", "v2", "v3"),
        validation_groups=("g",) * 4,
        embedding_table=table,
        metadata={},
    )
    events: list[str] = []
    runner._row_predictions(
        model,
        data,
        table,
        [{"record_index": 1, "record_id": "r1", "position": 2, "bin": "1-15"}],
        split="fit",
        device=torch.device("cpu"),
        direct_only=True,
        guard_callback=events.append,
    )
    assert any(":before_batch:" in event for event in events)
    assert any(":before_chunk:" in event for event in events)
    assert any(":after_chunk:" in event for event in events)
    cohort_events: list[str] = []
    runner.select_transition_error_cohort(
        model,
        data,
        table,
        split="validation",
        device=torch.device("cpu"),
        per_bin=1,
        guard_callback=cohort_events.append,
    )
    assert any(event.startswith("transition_cohort:validation:before_batch:") for event in cohort_events)
    assert any(event.startswith("transition_cohort:validation:after_chunk:") for event in cohort_events)


def test_validation_transition_cohort_keeps_all_errors_and_denominator():
    observations, truth, valid, table = _small_batch(records=4)
    model, _ = _small_models()
    with torch.inference_mode():
        hidden = F.normalize(model.direct_pre_normalized_hidden(observations, valid), dim=-1)
        record_slots = torch.arange(observations.shape[0]).repeat_interleave(observations.shape[1] - 1)
        position_slots = torch.arange(1, observations.shape[1]).repeat(observations.shape[0])
        predictions = model.logits_from_rows(hidden, record_slots, position_slots, table).argmax(dim=-1)
    truth[:, 1:] = (predictions.reshape(observations.shape[0], observations.shape[1] - 1) + 1) % table.shape[0]
    data = PublicJointData(
        fit_observations=observations,
        fit_truth=truth,
        fit_valid_mask=valid,
        fit_record_ids=("r0", "r1", "r2", "r3"),
        validation_observations=observations,
        validation_truth=truth,
        validation_valid_mask=valid,
        validation_record_ids=("v0", "v1", "v2", "v3"),
        validation_groups=("g",) * 4,
        embedding_table=table,
        metadata={},
    )
    cohort = runner.select_transition_error_cohort(
        model,
        data,
        table,
        split="validation",
        device=torch.device("cpu"),
        per_bin=1,
    )
    assert cohort["selection_mode"] == "all_transition_errors"
    assert cohort["valid_post_bos_count"] == 20
    assert cohort["transition_error_count"] == 20
    assert cohort["count"] == 20
    assert cohort["transition_token_accuracy"] == 0.0


def test_transition_artifact_is_truth_free_and_correction_summary_is_paired():
    cohorts = {
        "validation": {
            "count": 2,
            "rows": [
                {"record_index": 1, "record_id": "v1", "position": 2, "bin": "1-15"},
                {"record_index": 2, "record_id": "v2", "position": 18, "bin": "16-39"},
            ],
            "cohort_sha256": "a" * 64,
        },
        "fit": {
            "count": 1,
            "rows": [{"record_index": 3, "record_id": "f3", "position": 80, "bin": "80-127"}],
            "cohort_sha256": "b" * 64,
        },
    }
    artifact = runner._cohort_artifact_payload(
        seed=6106,
        arm_id=runner.P08_POSITIONWISE_STAGED,
        transition_step=runner.STAGED_AFFINE_STEPS,
        transition_state_sha256="c" * 64,
        cohorts=cohorts,
    )
    encoded = str(artifact)
    assert "truth" not in encoded.lower() or artifact["selection"]["truth_used_only_for_error_selection"] is True
    assert "prediction" not in encoded.lower() or artifact["selection"]["rows_expose_no_truth_or_predictions"] is True
    assert artifact["cohorts"]["validation"]["rows"][0]["record_id"] == "v1"
    summary = runner.correction_summary(
        [
            {"record_index": 1, "position": 2, "correct": False},
            {"record_index": 2, "position": 3, "correct": True},
            {"record_index": 4, "position": 5, "correct": False},
        ],
        [
            {"record_index": 1, "position": 2, "correct": True},
            {"record_index": 2, "position": 3, "correct": False},
            {"record_index": 4, "position": 5, "correct": False},
        ],
    )
    assert summary == {
        "rows": 3,
        "affine_correct": 1,
        "full_correct": 1,
        "both_correct": 0,
        "both_wrong": 1,
        "correction_gain": 1,
        "correction_regression": 1,
    }
