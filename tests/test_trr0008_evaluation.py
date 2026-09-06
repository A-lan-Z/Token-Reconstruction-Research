from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from scripts import trr0008_eval_contract as contract
from scripts import trr0008_eval_gate as gate
from scripts import trr0008_eval_runner as runner
from scripts import trr0008_eval_timing as timing
from scripts import trr0008_score as scorer


def _ids(records: int, *, token: int = 7) -> torch.Tensor:
    value = torch.full((records, contract.STORED_SEQUENCE_TOKENS), token, dtype=torch.long)
    value[:, 0] = contract.BOS_TOKEN_ID
    return value


def test_scientific_method_set_excludes_alias_and_a2() -> None:
    assert contract.METHOD_ORDER == (
        contract.REFERENCE_METHOD_ID,
        contract.CURRENT_RESIDUAL_METHOD_ID,
        contract.IMPROVED_RESIDUAL_METHOD_ID,
        contract.IMPROVED_DIAGONAL_METHOD_ID,
    )
    assert contract.TIMING_CONTROL_METHOD_ID not in contract.METHOD_ORDER
    assert contract.NO_A2_METHOD_ID not in contract.METHOD_ORDER
    assert contract.TIMING_METHOD_ORDER == (*contract.METHOD_ORDER, contract.TIMING_CONTROL_METHOD_ID)


def test_fresh_record_counts_are_bound_per_domain() -> None:
    manifest = {
        "cells": [
            {"cell_id": "pile__public_base", "records": 384},
            {"cell_id": "pile__public_lora_2601", "records": 384},
            {"cell_id": "finance__public_base", "records": 1024},
            {"cell_id": "finance__public_lora_2601", "records": 1024},
        ],
        "records_by_domain": {"pile": 384, "finance": 1024},
    }
    assert contract.records_for_cell(manifest, "pile__public_base") == 384
    assert contract.records_for_cell(manifest, "finance__public_lora_2601") == 1024


def test_timing_plan_has_balanced_method_positions() -> None:
    plan = timing.build_order_schedule(blocks=10, records_per_domain=128, records_per_block_cell=32, seed=8008)
    timing.validate_order_schedule(plan)
    counts = {method: [0] * len(contract.TIMING_METHOD_ORDER) for method in contract.TIMING_METHOD_ORDER}
    for entry in plan["record_schedule"]:
        for position, method in enumerate(entry["method_order"]):
            counts[method][position] += 1
    assert all(len(set(values)) == 1 for values in counts.values())
    assert all(sum(values) == 40 for values in counts.values())


def test_timing_ratio_summary_and_threshold() -> None:
    blocks = [
        {"method_cell_seconds": {cell: {contract.REFERENCE_METHOD_ID: 1.0, contract.PRIMARY_METHOD_ID: 1.1} for cell in contract.CELL_ORDER}},
        {"method_cell_seconds": {cell: {contract.REFERENCE_METHOD_ID: 2.0, contract.PRIMARY_METHOD_ID: 2.2} for cell in contract.CELL_ORDER}},
    ]
    result = timing.summarize_block_ratios(blocks)
    assert result["mean_ratio"] == 1.1
    assert result["decision"] == "passes"


def test_clopper_pearson_tail_directions_and_endpoints() -> None:
    interval = scorer.clopper_pearson(10, 20)
    assert interval["lower"] < 0.5 < interval["upper"]
    assert scorer.clopper_pearson(0, 20)["lower"] == 0.0
    assert scorer.clopper_pearson(20, 20)["upper"] == 1.0
    assert scorer.clopper_pearson(0, 20)["upper"] < 1.0
    assert scorer.clopper_pearson(20, 20)["lower"] > 0.0


def test_score_reports_paired_gains_and_losses_without_factorial_family() -> None:
    records = 4
    truth = {domain: _ids(records) for domain in contract.DOMAIN_ORDER}
    predictions: dict[str, dict[str, torch.Tensor]] = {}
    for method in contract.METHOD_ORDER:
        predictions[method] = {cell: _ids(records) for cell in contract.CELL_ORDER}
    candidate = predictions[contract.PRIMARY_METHOD_ID]["finance__public_base"].clone()
    candidate[0, 1] = 7
    predictions[contract.PRIMARY_METHOD_ID]["finance__public_base"] = candidate
    # Make one reference error so the candidate has one paired exact gain.
    reference = predictions[contract.REFERENCE_METHOD_ID]["finance__public_base"].clone()
    reference[0, 1] = 9
    predictions[contract.REFERENCE_METHOD_ID]["finance__public_base"] = reference
    result = scorer.score_predictions(predictions, truth, bootstrap_draws=100)
    assert result["method_order"] == list(contract.METHOD_ORDER)
    assert "factorial_contrasts" not in result
    contrast = result["contrasts_vs_reference"][contract.PRIMARY_METHOD_ID]["finance__public_base"]
    assert contrast["exact_gains"] == 1
    assert contrast["exact_losses"] == 0
    assert contrast["exact_gain_rate_cp"]["upper"] > contrast["exact_gain_rate_cp"]["lower"]


def test_decision_refuses_unfrozen_thresholds() -> None:
    result = {"contrasts_vs_reference": {contract.PRIMARY_METHOD_ID: {}}}
    decision = scorer.decide(result, scorer.proposed_decision_contract())
    assert decision["status"] == "BLOCKED_UNTIL_DECISION_CONTRACT_FREEZE"
    assert decision["promotion"] == "retain_reference"


def test_current_h_runner_adapter_is_positionwise(monkeypatch) -> None:
    from scripts import trr0008_eval_runner as runner

    monkeypatch.setattr(contract, "STORED_SEQUENCE_TOKENS", 4)
    monkeypatch.setattr(contract, "SCORED_POST_BOS_TOKENS", 3)
    monkeypatch.setattr(contract, "VOCABULARY_SIZE", 5)
    monkeypatch.setattr(contract, "HIDDEN_SIZE", 2)
    monkeypatch.setattr(contract, "BOS_TOKEN_ID", 4)

    class FakeModel(torch.nn.Module):
        hidden_size = 2
        vocabulary_size = 5

        def projected_hidden(self, activation, valid_mask):
            return activation.float()

        def logits_from_rows(self, projected, record_slots, position_slots, embedding):
            rows = projected[record_slots, position_slots]
            return rows @ embedding.T

    model = FakeModel()
    embedding = torch.eye(5, 2)
    activation = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    mask = torch.ones(4, dtype=torch.bool)
    first = runner.predict_current_h(model, embedding, activation, mask, device=torch.device("cpu"))
    changed_other_position = activation.clone()
    changed_other_position[2] = torch.tensor([99.0, -99.0])
    second = runner.predict_current_h(model, embedding, changed_other_position, mask, device=torch.device("cpu"))
    assert first[1].item() == second[1].item()
    assert first[0].item() == contract.BOS_TOKEN_ID
    assert first.shape == (4,)


def _file_record(path: Path) -> dict[str, object]:
    value = path.read_bytes()
    return {"path": str(path), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def test_public_gate_accepts_complete_synthetic_matrix(tmp_path: Path) -> None:
    records = 2
    observation_file = tmp_path / "observation.bin"
    observation_file.write_bytes(b"public-observation-placeholder")
    cells = []
    for cell_id in contract.CELL_ORDER:
        cells.append(
            {
                "cell_id": cell_id,
                "records": records,
                "observation": _file_record(observation_file)
                | {
                    "shape": [records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE],
                    "activations_key": "activations",
                    "attention_mask_key": "attention_mask",
                    "position_ids_key": "position_ids",
                },
            }
        )
    observations = {
        "schema": contract.OBSERVATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "truth_opened": False,
        "target_labels_loaded": False,
        "source_text_loaded": False,
        "source_text_written": False,
        "records_by_domain": {"pile": records, "finance": records},
        "cells": cells,
    }
    observation_path = tmp_path / "observations.json"
    observation_path.write_text(json.dumps(observations), encoding="utf-8")
    fake_state = _file_record(observation_file)
    methods = []
    for method_id in contract.METHOD_ORDER:
        if method_id == contract.REFERENCE_METHOD_ID:
            loader = contract.REFERENCE_LOADER
        else:
            loader = dict(contract.POSITIONWISE_LOADER) | {
                "kwargs": dict(contract.POSITIONWISE_LOADER["kwargs"])
                | {"method_id": contract.METHOD_MODEL_IDS[method_id]}
            }
        methods.append(
            {
                "id": method_id,
                "cells": list(contract.CELL_ORDER),
                "records_per_cell": {cell: records for cell in contract.CELL_ORDER},
                "state": fake_state,
                "loader": loader,
            }
        )
    alias = dict(methods[3]) | {
        "id": contract.TIMING_CONTROL_METHOD_ID,
        "loader": dict(contract.POSITIONWISE_LOADER)
        | {"kwargs": dict(contract.POSITIONWISE_LOADER["kwargs"]) | {"method_id": contract.METHOD_MODEL_IDS[contract.TIMING_CONTROL_METHOD_ID]}},
    }
    output_root = tmp_path / "predictions"
    registration = {
        "schema": contract.REGISTRATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "method_order": list(contract.METHOD_ORDER),
        "method_ids": list(contract.METHOD_ORDER),
        "cell_order": list(contract.CELL_ORDER),
        "records_by_domain": {"pile": records, "finance": records},
        "geometry": dict(contract.STATIC_GEOMETRY),
        "methods": methods,
        "timing_control": alias,
        "observation_manifest": _file_record(observation_path),
        "runtime_assets": {"normalized_public_E": fake_state},
        "output_root": str(output_root),
        "repository_root": str(tmp_path),
        "code_commit": "a" * 40,
        "truth_opened": False,
        "source_text_or_target_labels": False,
    }
    contract.validate_registration(registration)
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    registration_sha = contract.sha256_file(registration_path)
    predictions = _ids(records)
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            path = output_root / cell_id.split("__", 1)[0] / cell_id.split("__", 1)[1] / f"{method_id}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"predictions": predictions},
                str(path),
                metadata={
                    "schema": contract.PREDICTION_SCHEMA,
                    "task_id": contract.TASK_ID,
                    "registration_sha256": registration_sha,
                    "cell_id": cell_id,
                    "method_id": method_id,
                    "records": str(records),
                    "geometry_json": json.dumps({"records": records, **contract.STATIC_GEOMETRY}, sort_keys=True),
                    "truth_opened": "false",
                    "candidate_arrays_persisted": "false",
                },
            )
            path.with_suffix(".run.json").write_text(
                json.dumps(
                    {
                        "schema": contract.TIMING_SCHEMA,
                        "task_id": contract.TASK_ID,
                        "method_id": method_id,
                        "cell_id": cell_id,
                        "records": records,
                        "truth_opened": False,
                        "warmup_output_exact_match_measured": True,
                        "prediction_artifact": _file_record(path),
                    }
                ),
                encoding="utf-8",
            )
    run_predictions = {}
    run_timings = {}
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            key = f"{method_id}::{cell_id}"
            path = output_root / cell_id.split("__", 1)[0] / cell_id.split("__", 1)[1] / f"{method_id}.safetensors"
            run_predictions[key] = _file_record(path) | {
                "prediction_sha256": contract.tensor_digest(predictions),
                "records": records,
            }
            run_timings[key] = json.loads(path.with_suffix(".run.json").read_text(encoding="utf-8"))
    run_path = tmp_path / "run_manifest.json"
    run_path.write_text(
        json.dumps(
            {
                "schema": contract.RUN_SCHEMA,
                "task_id": contract.TASK_ID,
                "registration": _file_record(registration_path),
                "code_commit": "a" * 40,
                "observation_manifest": _file_record(observation_path),
                "predictions": run_predictions,
                "timings": run_timings,
                "truth_opened": False,
                "candidate_arrays_persisted": False,
            }
        ),
        encoding="utf-8",
    )
    checked = gate.validate_public_outputs(
        registration_path=registration_path,
        repository_root=tmp_path,
        run_manifest_path=run_path,
    )
    assert checked["status"] == "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH"
    assert len(checked["predictions"]) == 16


def test_record_bootstrap_uses_declared_one_sided_tails() -> None:
    lower_95 = scorer._bootstrap_interval(
        [0.0, 0.2, 0.8, 1.0], seed=8008, draws=200, one_sided_alpha=0.05
    )
    lower_975 = scorer._bootstrap_interval(
        [0.0, 0.2, 0.8, 1.0], seed=8008, draws=200, one_sided_alpha=0.025
    )
    assert lower_95["unit"] == "source_record"
    assert lower_95["one_sided_confidence"] == 0.95
    assert lower_975["one_sided_confidence"] == 0.975
    assert lower_975["lower"] <= lower_95["lower"]
    assert lower_975["upper"] >= lower_95["upper"]


def test_score_json_is_serializable_and_binds_all_confidence_routes() -> None:
    records = 3
    truth = {domain: _ids(records) for domain in contract.DOMAIN_ORDER}
    predictions = {
        method: {cell: _ids(records) for cell in contract.CELL_ORDER}
        for method in contract.METHOD_ORDER
    }
    result = scorer.score_predictions(
        predictions,
        truth,
        bootstrap_seed=8008,
        bootstrap_draws=40,
        alpha_component=0.0125,
        safeguard_alpha_component=0.025,
        primary_token_alpha=0.025,
        safeguard_token_alpha=0.05,
    )
    json.dumps(result, allow_nan=False)
    assert result["bootstrap"] == {"seed": 8008, "draws": 40, "unit": "source_record"}
    assert result["confidence"] == {
        "primary_exact_component_alpha": 0.0125,
        "safeguard_exact_component_alpha": 0.025,
        "primary_token_one_sided_alpha": 0.025,
        "safeguard_token_one_sided_alpha": 0.05,
    }
    contrast = result["contrasts_vs_reference"][contract.PRIMARY_METHOD_ID][contract.CELL_ORDER[0]]
    assert contrast["token_net_bootstrap_975"]["one_sided_alpha"] == 0.025
    assert contrast["token_net_bootstrap_95"]["one_sided_alpha"] == 0.05


def test_timing_inconclusive_is_not_cost_failure() -> None:
    timing_doc = json.loads(
        Path("experiments/TRR-0008/timing/precision40_result.json").read_text(encoding="utf-8")
    )
    params = scorer._decision_parameters(
        json.loads(Path("experiments/TRR-0008/planning/decision_contract.json").read_text(encoding="utf-8"))
    )
    timing_doc["summary"]["qualification"]["decision"] = "INCONCLUSIVE"
    timing_doc["summary"]["qualification"]["measurement_valid"] = False
    status, _ = scorer._timing_decision(
        timing_doc,
        primary_cell=params["cost_primary_cell"],
        cost_cells=params["cost_cells"],
        threshold=params["cost_threshold"],
    )
    assert status == "COST_EVIDENCE_INCONCLUSIVE"
    timing_doc["summary"]["qualification"]["decision"] = "FAIL"
    timing_doc["summary"]["qualification"]["measurement_valid"] = True
    timing_doc["summary"]["qualification"]["cost_failure_demonstrated"] = True
    status, _ = scorer._timing_decision(
        timing_doc,
        primary_cell=params["cost_primary_cell"],
        cost_cells=params["cost_cells"],
        threshold=params["cost_threshold"],
    )
    assert status == "COST_GATE_FAILED"


def test_decision_contract_structured_confidence_binding_is_checked() -> None:
    decision_contract = json.loads(
        Path("experiments/TRR-0008/planning/decision_contract.json").read_text(encoding="utf-8")
    )
    scorer._decision_parameters(decision_contract)
    decision_contract["confidence"]["safeguard"]["token_bootstrap_lower_tail_alpha"] = 0.025
    try:
        scorer._decision_parameters(decision_contract)
    except scorer.ScoreError as exc:
        assert "safeguard token bootstrap" in str(exc)
    else:
        raise AssertionError("changed confidence binding unexpectedly accepted")


def test_runner_resource_guard_fails_closed_on_missing_host_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_rss_bytes", lambda: None)
    with pytest.raises(runner.RunnerError, match="RSS telemetry unavailable"):
        runner._guard(
            device=torch.device("cpu"),
            guard=contract.RESOURCE_GUARD,
            started=__import__("time").perf_counter(),
            stage="fixture",
        )


def test_runner_numerics_fails_closed_on_interop_setting_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original = torch.set_num_interop_threads
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _value: (_ for _ in ()).throw(RuntimeError("locked")))
    with pytest.raises(runner.RunnerError, match="inter-op"):
        runner._configure_numerics(contract.NUMERICAL_SETTINGS)
    monkeypatch.setattr(torch, "set_num_interop_threads", original)
