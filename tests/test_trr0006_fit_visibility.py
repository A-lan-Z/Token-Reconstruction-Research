from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
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
        selection_metric="token_accuracy",
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


def test_probe_prediction_retention_is_logging_only(tmp_path: Path) -> None:
    data, direct = _tiny_data()
    model = runner.build_visibility_decoder(
        runner.POSITIONWISE_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
        qkv_seed=runner.PROBE_SEED,
        direct_state=direct,
        direct_init_label="competent_public_affine",
    )
    ledger = [
        {
            "record_index": record,
            "record_id": f"fit-{record}",
            "position": position,
            "bin": "1-15",
        }
        for record, position in ((0, 1), (0, 2), (2, 1), (4, 3), (6, 4))
    ]
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    aggregate = runner._evaluate_error_ledger(
        model, data, data.embedding_table, ledger, device=torch.device("cpu")
    )
    retained = runner._evaluate_error_ledger(
        model,
        data,
        data.embedding_table,
        ledger,
        device=torch.device("cpu"),
        retain_predictions=True,
    )
    retained_without_rows = {key: value for key, value in retained.items() if key != "prediction_rows"}
    assert retained_without_rows == aggregate
    for name, value in model.state_dict().items():
        assert torch.equal(value, before_state[name])
    rows = retained["prediction_rows"]
    assert len(rows) == len(ledger)
    assert [(row["record_index"], row["position"]) for row in rows] == [
        (row["record_index"], row["position"]) for row in ledger
    ]
    assert all(0 <= row["prediction"] < 13 and row["tie_count"] >= 1 for row in rows)

    output = tmp_path / "probe_predictions.json"
    runner._write_probe_prediction_snapshot(
        output,
        method_id=runner.POSITIONWISE_METHOD,
        initial_metrics=aggregate,
        final_metrics=aggregate,
        initial_rows=rows,
        final_rows=rows,
    )
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == runner.PROBE_PREDICTION_SCHEMA
    assert payload["retention"].startswith("logging-only")
    assert payload["row_count"] == len(ledger)
    assert payload["initial_predictions"] == payload["final_predictions"]


def test_probe_prediction_retention_preserves_arm_state_hash(tmp_path: Path, monkeypatch) -> None:
    data, direct = _tiny_data()
    monkeypatch.setattr(runner, "HIDDEN_SIZE", 8)
    monkeypatch.setattr(runner, "VOCABULARY_SIZE", 13)
    monkeypatch.setattr(runner, "CONTEXT_WIDTH", 4)
    monkeypatch.setattr(runner, "RECORD_BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "POSITION_BUDGET", 4)
    monkeypatch.setattr(runner, "PROBE_STEPS", 1)
    ledger = [
        {
            "record_index": record,
            "record_id": f"fit-{record}",
            "position": position,
            "bin": "1-15",
        }
        for record in range(8)
        for position in (1, 2)
    ]
    schedule = runner.PositionSchedule(
        batch_record_indices=torch.tensor([[0, 1]], dtype=torch.long),
        draw_record_slots=torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
        draw_position_slots=torch.tensor([[1, 2, 1, 2]], dtype=torch.long),
        eligible_counts=torch.tensor([4], dtype=torch.int32),
        used_replacement=torch.tensor([True], dtype=torch.bool),
        seed=runner.PROBE_SEED,
        position_budget=4,
        record_batch_size=2,
    )
    plain_args = _guard_args()
    plain_args.retain_probe_predictions = False
    retained_args = _guard_args()
    retained_args.retain_probe_predictions = True
    plain = runner._run_probe_arm(
        runner.POSITIONWISE_METHOD,
        data,
        direct,
        data.embedding_table,
        schedule,
        ledger,
        args=plain_args,
        device=torch.device("cpu"),
        output_root=tmp_path / "plain",
        deadline=None,
        guards=[],
    )
    retained = runner._run_probe_arm(
        runner.POSITIONWISE_METHOD,
        data,
        direct,
        data.embedding_table,
        schedule,
        ledger,
        args=retained_args,
        device=torch.device("cpu"),
        output_root=tmp_path / "retained",
        deadline=None,
        guards=[],
    )
    assert plain["initial_state_sha256"] == retained["initial_state_sha256"]
    assert plain["final_state_sha256"] == retained["final_state_sha256"]
    assert plain["initial_metrics"] == retained["initial_metrics"]
    assert plain["final_metrics"] == retained["final_metrics"]
    assert retained["probe_predictions"]["row_count"] == len(ledger)


def test_probe_prediction_retention_rejects_reordered_snapshots(tmp_path: Path) -> None:
    row = {"record_index": 0, "record_id": "fit-0", "position": 1, "bin": "1-15"}
    with pytest.raises(runner.VisibilityFitError, match="ledger order"):
        runner._write_probe_prediction_snapshot(
            tmp_path / "mismatch.json",
            method_id=runner.POSITIONWISE_METHOD,
            initial_metrics={"correct": 0},
            final_metrics={"correct": 0},
            initial_rows=[row],
            final_rows=[{**row, "position": 2}],
        )


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
    assert result["selection_metric"] == "validation_token_accuracy"
    assert result["state"]["metadata"]["direct_affine_sha256"] == runner.DIRECT_AFFINE_SHA256
    loaded = load_visibility_state(
        output / "selected.safetensors",
        method_id=FULL_RECORD_METHOD,
        hidden_size=8,
        vocabulary_size=13,
        context_width=4,
    )
    assert loaded.method_id == FULL_RECORD_METHOD
    assert result["state"]["metadata"]["selection_metric"] == "validation_token_accuracy"
    assert (output / "learning_curve.json").is_file()
    assert guards


def test_runner_qualification_runs_disposable_full_record_backward_cell(tmp_path: Path, monkeypatch) -> None:
    data, direct = _tiny_data()
    monkeypatch.setattr(runner, "HIDDEN_SIZE", 8)
    monkeypatch.setattr(runner, "VOCABULARY_SIZE", 13)
    monkeypatch.setattr(runner, "CONTEXT_WIDTH", 4)
    monkeypatch.setattr(runner, "DECLARED_SEQUENCE_LENGTH", 6)
    monkeypatch.setattr(runner, "RECORD_BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "POSITION_BUDGET", 4)
    monkeypatch.setattr(runner, "QUALIFICATION_STEPS", 2)

    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text("{}\n", encoding="utf-8")
    fit_manifest = tmp_path / "fit.json"
    fit_manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "qualification"
    args = SimpleNamespace(
        output_root=output,
        repository_root=tmp_path,
        device="cpu",
        max_seconds=60.0,
        direct_affine_state=tmp_path / "direct.safetensors",
        direct_affine_sha256=runner.DIRECT_AFFINE_SHA256,
        preflight_receipt=preflight_path,
        fit_manifest=fit_manifest,
        validation_manifest=None,
        embedding_path=None,
        maximum_host_rss_gib=16.0,
        minimum_host_available_gib=0.1,
        minimum_free_gib=0.1,
        maximum_gpu_reserved_gib=6.0,
        torch_threads=1,
        torch_interop_threads=1,
    )
    monkeypatch.setattr(
        runner,
        "_direct_state",
        lambda _args: (
            direct,
            {
                "path": "synthetic-direct.safetensors",
                "bytes": 1,
                "sha256": runner.DIRECT_AFFINE_SHA256,
                "state_sha256": "synthetic-state",
                "initialization": "competent_public_affine",
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "_validate_preflight_receipt",
        lambda *args, **kwargs: {"status": "SOURCE_ONLY_PREFLIGHT_PASS"},
    )
    data_receipt = {
        "fit_manifest": {"path": "fit", "bytes": 1, "sha256": "fit"},
        "validation_manifest": {"path": "fit", "bytes": 1, "sha256": "fit"},
        "embedding_table": {"path": "embedding", "bytes": 1, "sha256": "embedding"},
        "fit_geometry": [8, 6, 8],
        "validation_geometry": [8, 6, 8],
    }
    monkeypatch.setattr(
        runner,
        "_load_cropped_public_data",
        lambda *args, **kwargs: (data, data_receipt),
    )

    receipt = runner.run_qualification(args)

    assert receipt["status"] == "PASS"
    assert receipt["method_id"] == FULL_RECORD_METHOD
    assert receipt["geometry"]["record_batch_size"] == 2
    assert receipt["geometry"]["sequence_length"] == 6
    assert receipt["geometry"]["query_draws_per_step"] == 4
    assert receipt["all_parameters_trainable"] is True
    assert receipt["optimizer_state_parameter_count"] == receipt["parameter_tensor_count"]
    assert receipt["output_residual_became_active"] is True
    assert receipt["q_path_became_active"] is True
    assert receipt["state_role"].startswith("disposable qualification")
    assert (output / "qualification_receipt.json").is_file()
    assert (output / "qualification_schedule.safetensors").is_file()
    assert not (output / "selected.safetensors").exists()


def test_qualification_receipt_binds_geometry_and_affine_before_execution(tmp_path: Path) -> None:
    fit_manifest = tmp_path / "fit.json"
    fit_manifest.write_text("{}\n", encoding="utf-8")
    expected_file = runner._file_record(fit_manifest)
    receipt_path = tmp_path / "qualification_receipt.json"
    receipt_path.write_text(
        __import__("json").dumps(
            {
                "schema": runner.QUALIFICATION_SCHEMA,
                "status": "PASS",
                "method_id": FULL_RECORD_METHOD,
                "updates": 2,
                "all_parameters_trainable": True,
                "finite_parameters": True,
                "output_residual_became_active": True,
                "q_path_became_active": True,
                "parameter_tensor_count": 11,
                "optimizer_state_parameter_count": 11,
                "direct_affine": {"sha256": runner.DIRECT_AFFINE_SHA256},
                "data": {
                    "fit_manifest": expected_file,
                    "validation_manifest": expected_file,
                    "fit_geometry": [1200, 128, 2048],
                },
                "geometry": {
                    "record_batch_size": 8,
                    "sequence_length": 128,
                    "query_draws_per_step": 512,
                },
            }
        ),
        encoding="utf-8",
    )

    validated = runner._validate_qualification_receipt(
        receipt_path,
        direct_hash=runner.DIRECT_AFFINE_SHA256,
        fit_manifest=fit_manifest,
        validation_manifest=None,
    )
    assert validated["status"] == "PASS"

    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8").replace(
            runner.DIRECT_AFFINE_SHA256, "0" * 64
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.VisibilityFitError, match="direct-affine hash"):
        runner._validate_qualification_receipt(
            receipt_path,
            direct_hash=runner.DIRECT_AFFINE_SHA256,
            fit_manifest=fit_manifest,
            validation_manifest=None,
        )
