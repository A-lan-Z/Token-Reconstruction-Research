from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import trr0010_directional_fit as fit


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        steps=fit.TRAINING_STEPS,
        record_batch_size=fit.EXPECTED_BATCH_RECORDS,
        position_budget=fit.EXPECTED_POSITION_BUDGET,
        train_sequence_tokens=fit.EXPECTED_FIT_TOKENS,
        hidden_size=fit.EXPECTED_HIDDEN_SIZE,
        selection_metric=fit.EXPECTED_SELECTION_METRIC,
        learning_rate=fit.EXPECTED_BASE_LEARNING_RATE,
        weight_decay=fit.EXPECTED_WEIGHT_DECAY,
        gradient_clip_norm=fit.EXPECTED_GRADIENT_CLIP_NORM,
    )


def _prepared(arm_name: str, *, schedule_sha: str = "a" * 64) -> dict:
    bank_role = fit.ARM_TO_BANK[arm_name]
    runner = SimpleNamespace(run_training=lambda *args, **kwargs: None)
    parameter = nn.Parameter(torch.zeros(()))
    delta_parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter], "lr": fit.EXPECTED_BASE_LEARNING_RATE, "name": "decoder"},
            {"params": [delta_parameter], "lr": fit.EXPECTED_DIRECTIONAL_LEARNING_RATE, "name": "directional_delta"},
        ],
        weight_decay=fit.EXPECTED_WEIGHT_DECAY,
        foreach=False,
    )
    bindings = SimpleNamespace(
        metadata=lambda: {"arm": arm_name, "contract": "c" * 64},
        source_binding_digest=lambda: "d" * 64,
    )
    runtime = SimpleNamespace(
        decoder=nn.Linear(1, 1, bias=False),
        hook=object(),
        optimizer=optimizer,
        bindings=bindings,
    )
    return {
        "arm_name": arm_name,
        "bank_role": bank_role,
        "runner": runner,
        "runtime": runtime,
        "config": _config(),
        "training_activation_dtype": torch.bfloat16,
        "validation_activation_dtype": torch.bfloat16,
        "validation_sequence_tokens": fit.EXPECTED_VALIDATION_TOKENS,
        "validation_batch_records": 8,
        "validation_callback": lambda *_args: {},
        "source": object(),
        "base_state": {"sha256": "e" * 64},
        "fit_manifest": {"sha256": "f" * 64},
        "schedule_steps": tuple(range(fit.TRAINING_STEPS)),
        "schedule_steps_count": fit.TRAINING_STEPS,
        "schedule_seed": 4010,
        "schedule_semantic_sha256": "b" * 64,
        "schedule_exposure": {"steps": fit.TRAINING_STEPS, "draws_per_step": 512},
        "schedule_binding": {"sha256": schedule_sha, "bytes": 10},
        "control_schedule_binding": {"sha256": schedule_sha, "bytes": 10},
        "diagnostic_binding": {
            "fit_record_count": 64,
            "full_bank_endpoint_steps": [0, fit.TRAINING_STEPS],
            "selection_isolated": True,
        },
        "diagnostic_callback": lambda point, decoder, hook: {
            "fit_diagnostics": {"records": [{"row": index} for index in range(64)]},
            **({"full_bank": {"endpoint": "start", "rows": 100}} if int(point["step"]) == 0 else {}),
            **({"full_bank": {"endpoint": "end", "rows": 100}} if int(point["step"]) == fit.TRAINING_STEPS else {}),
        },
        "public_embedding": torch.zeros(2, 2),
    }


def test_handoff_binds_exact_grid_and_64_diagnostics() -> None:
    prepared = _prepared("current_directional")
    result = fit.validate_fit_inputs(prepared, arm_name="current_directional")
    assert result["schedule"]["seed"] == 4010
    assert result["diagnostics"]["fit_record_count"] == 64
    assert result["diagnostics"]["full_bank_endpoint_steps"] == [0, fit.TRAINING_STEPS]


def test_handoff_rejects_schedule_not_equal_to_fixed_control() -> None:
    prepared = _prepared("expanded_directional")
    prepared["control_schedule_binding"] = {"sha256": "c" * 64, "bytes": 10}
    with pytest.raises(fit.DirectionalFitError, match="schedule bytes differ"):
        fit.validate_fit_inputs(prepared, arm_name="expanded_directional")


def test_handoff_rejects_wrong_diagnostic_count() -> None:
    prepared = _prepared("current_directional")
    prepared["diagnostic_binding"] = {
        "fit_record_count": 63,
        "full_bank_endpoint_steps": [0, fit.TRAINING_STEPS],
        "selection_isolated": True,
    }
    with pytest.raises(fit.DirectionalFitError, match="64 fit records"):
        fit.validate_fit_inputs(prepared, arm_name="current_directional")


def test_a2_nested_fitting_diagnostics_are_normalized_without_selection_metric() -> None:
    fixed = {"scope": "frozen64", "row_count": 64, "batch_count": 8, "position_chunk_count": 1, "full_vocab_logits_rows": 8128}
    start = {"scope": "full_bank", "row_count": 1200, "batch_count": 150, "position_chunk_count": 1, "full_vocab_logits_rows": 152400}
    summary = fit._diagnostic_summary(
        {"fitting_diagnostics": {"selection_metric_untouched": True, "scopes": [fixed, start]}},
        step=0,
        started=0.0,
        arm_name="current_directional",
    )
    assert summary["fit_records"] == 64
    assert summary["full_bank"]["endpoint"] == "start"
    assert summary["selection_metric_untouched"] is True


def test_run_forwards_exact_grid_scheduler_and_runs_arms_sequentially(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    factory_calls: list[str] = []

    def factory(*, arm_name, bank_role, checkpoint_steps, output_root):
        assert tuple(checkpoint_steps) == fit.CHECKPOINT_STEPS
        assert bank_role == fit.ARM_TO_BANK[arm_name]
        factory_calls.append(arm_name)
        return _prepared(arm_name)

    def fake_validate(*args, **kwargs):
        del args, kwargs

    def fake_checkpoint_callback(**kwargs):
        del kwargs

        def callback(point, decoder, hook):
            del decoder, hook
            step = int(point["step"])
            return {
                "checkpoint": {
                    "metadata": {
                        "selected_step": step,
                        "runner_point_state_sha256": "d" * 64,
                    },
                    "path": f"/tmp/{step}.safetensors",
                    "bytes": 10,
                    "sha256": "a" * 64,
                }
            }

        return callback

    def fake_restore(*, selected_step, **kwargs):
        del kwargs
        return {
            "selected_step": selected_step,
            "checkpoint_path": f"/tmp/{selected_step}.safetensors",
            "checkpoint_bytes": 10,
            "checkpoint_sha256": "a" * 64,
            "export": {"path": f"/tmp/{selected_step}-W.safetensors", "bytes": 11, "sha256": "b" * 64},
        }

    def fake_base_export(*, path, selected_step, **kwargs):
        del kwargs
        return {"path": str(path), "bytes": 12, "sha256": "c" * 64, "selected_step": selected_step}

    monkeypatch.setattr(fit, "validate_optimizer_configuration", fake_validate)
    monkeypatch.setattr(fit, "make_serialization_only_checkpoint_callback", fake_checkpoint_callback)
    monkeypatch.setattr(fit, "restore_selected_and_export", fake_restore)
    monkeypatch.setattr(fit, "export_selected_base_decoder_state", fake_base_export)

    def run_training(decoder, hook, source, schedule_steps, **kwargs):
        del decoder, hook, source, schedule_steps
        assert kwargs["schedule_steps_count"] == fit.TRAINING_STEPS
        assert tuple(kwargs["checkpoint_steps"]) == fit.CHECKPOINT_STEPS
        assert kwargs["compute_base_logits"] is False
        assert kwargs["scheduler"].T_max == fit.TRAINING_STEPS
        assert tuple(kwargs["scheduler"].get_last_lr()) == (
            fit.EXPECTED_BASE_LEARNING_RATE,
            fit.EXPECTED_DIRECTIONAL_LEARNING_RATE,
        )
        calls.append(kwargs)
        callback = kwargs["checkpoint_callback"]
        bindings = [callback({"step": step}, object(), object()) for step in fit.CHECKPOINT_STEPS]
        return {
            "status": "COMPLETED",
            "selected_step": 4000,
            "selected_state_sha256": "d" * 64,
            "checkpoints": list(fit.CHECKPOINT_STEPS),
            "checkpoint_state_bindings": bindings,
            "timing": {"whole_wall_seconds": 1.0},
        }

    # The two factory-created runner objects share this synthetic method, as a
    # real capacity factory would bind the imported A2 runner module.
    original_factory = factory

    def factory_with_runner(*, arm_name, bank_role, checkpoint_steps, output_root):
        prepared = original_factory(
            arm_name=arm_name,
            bank_role=bank_role,
            checkpoint_steps=checkpoint_steps,
            output_root=output_root,
        )
        prepared["runner"].run_training = run_training
        return prepared

    receipt = fit.run_directional_arms(factory_with_runner, output_root=tmp_path / "fit")
    assert receipt["status"] == "FIT_COMPLETE"
    assert factory_calls == ["current_directional", "expanded_directional"]
    assert len(calls) == 2
    assert (tmp_path / "fit" / "run_receipt.json").is_file()
    for arm_name in fit.REQUIRED_ARMS:
        arm = receipt["arms"][arm_name]
        assert arm["runner_result"]["status"] == "COMPLETED"
        assert arm["timing"]["runner_call_seconds"] >= 0.0
        assert arm["timing"]["restore_export_seconds"] >= 0.0
        assert arm["timing"]["base_decoder_export_seconds"] >= 0.0
        assert arm["timing"]["provider_preparation_included"] is False
        assert [event["step"] for event in arm["diagnostic_events"]] == list(fit.CHECKPOINT_STEPS)
        assert {event["full_bank"]["endpoint"] for event in arm["diagnostic_events"] if event["full_bank"]} == {"start", "end"}
        assert all("records" not in event for event in arm["diagnostic_events"])
        raw_artifacts = arm["diagnostic_raw_artifacts"]
        assert [item["step"] for item in raw_artifacts] == list(fit.CHECKPOINT_STEPS)
        for item in raw_artifacts:
            raw_path = Path(item["path"])
            assert raw_path.is_file()
            assert item["bytes"] == raw_path.stat().st_size
            assert item["sha256"] == __import__("hashlib").sha256(raw_path.read_bytes()).hexdigest()


def test_selected_binding_rejects_incomplete_runner_grid() -> None:
    result = {
        "status": "COMPLETED",
        "selected_step": 4000,
        "selected_state_sha256": "a" * 64,
        "checkpoints": list(fit.CHECKPOINT_STEPS[:-1]),
        "checkpoint_state_bindings": [],
    }
    with pytest.raises(fit.DirectionalFitError, match="checkpoint grid"):
        fit._selected_checkpoint(result, arm_name="current_directional")


def test_concrete_provider_adapter_preserves_existing_four_argument_loader_api() -> None:
    seen = {}

    def provider(binding_receipt, lease_caps, device, preparation_guard):
        seen.update({"binding": binding_receipt, "lease": lease_caps, "device": device})
        preparation_guard("synthetic")
        return {"arm_name": "current_directional", "bank_role": "current"}

    guard_calls: list[str] = []
    builder = fit.bind_concrete_provider(
        provider,
        binding_receipts={"current_directional": {"sha256": "a" * 64}},
        lease_caps={"device": "cpu"},
        device=torch.device("cpu"),
        preparation_guard=guard_calls.append,
    )
    result = builder(
        arm_name="current_directional",
        bank_role="current",
        checkpoint_steps=fit.CHECKPOINT_STEPS,
        output_root=Path("/tmp/synthetic-fit"),
    )
    assert result["bank_role"] == "current"
    assert seen["binding"]["sha256"] == "a" * 64
    assert seen["lease"]["device"] == "cpu"
    assert seen["device"] == torch.device("cpu")
    assert guard_calls == ["synthetic"]


def test_run_single_arm_writes_arm_scoped_receipt(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_fit(inputs, *, arm_name, output_root, deadline_seconds, arm_output_root=None):
        calls.append({"inputs": inputs, "arm_name": arm_name, "output_root": output_root, "arm_output_root": arm_output_root, "deadline_seconds": deadline_seconds})
        return {
            "arm_name": arm_name,
            "bank_role": fit.ARM_TO_BANK[arm_name],
            "status": "COMPLETED",
            "truth_opened": False,
        }

    monkeypatch.setattr(fit, "fit_one_arm", fake_fit)

    def builder(**kwargs):
        assert kwargs["arm_name"] == "current_directional"
        assert kwargs["bank_role"] == "B0"
        assert tuple(kwargs["checkpoint_steps"]) == fit.CHECKPOINT_STEPS
        return {"synthetic": True}

    output = tmp_path / "current"
    receipt = fit.run_directional_arm(
        builder,
        arm_name="current_directional",
        output_root=output,
        deadline_seconds=7200.0,
        command=["fit", "--arm", "current_directional"],
    )
    assert receipt["status"] == "FIT_COMPLETE"
    assert receipt["arm"] == "current_directional"
    assert set(receipt["arms"]) == {"current_directional"}
    assert calls[0]["deadline_seconds"] == 7200.0
    assert calls[0]["arm_output_root"] == output
    assert json.loads((output / "run_receipt.json").read_text())["arm"] == "current_directional"


def test_selected_binding_accepts_actual_runner_receipt() -> None:
    receipt_path = Path("experiments/TRR-0010/training/directional_fit/current_directional_watchdog_r2/fit/raw_runner_result.json")
    if not receipt_path.is_file():
        pytest.skip("authorized current-arm raw runner receipt is not present")
    result = json.loads(receipt_path.read_text(encoding="utf-8"))
    selected = fit._selected_checkpoint(result, arm_name="current_directional")
    assert selected["metadata"]["selected_step"] == result["selected_step"]
    assert selected["metadata"]["runner_point_state_sha256"] == result["selected_state_sha256"]


def test_selected_binding_rejects_mismatched_runner_point_state() -> None:
    result = {
        "status": "COMPLETED",
        "selected_step": 0,
        "selected_state_sha256": "a" * 64,
        "checkpoints": list(fit.CHECKPOINT_STEPS),
        "checkpoint_state_bindings": [{
            "checkpoint": {
                "metadata": {"selected_step": 0, "runner_point_state_sha256": "b" * 64},
                "path": "/tmp/0.safetensors", "bytes": 10, "sha256": "c" * 64,
            }
        }],
    }
    with pytest.raises(fit.DirectionalFitError, match="does not match runner selected state"):
        fit._selected_checkpoint(result, arm_name="current_directional")
