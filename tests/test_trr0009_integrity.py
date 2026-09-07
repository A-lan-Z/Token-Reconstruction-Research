from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts import trr0009_eval_contract as contract
from scripts import trr0009_eval_gate as gate
from scripts import trr0009_eval_register as register
from scripts import trr0009_eval_runner as runner
from scripts import trr0009_eval_truth as truth_adapter
from scripts import trr0009_model as model_contract


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, public_fitting_tensor_metadata: bool = False, records: int = 8):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    task = repo / "experiments" / contract.TASK_ID
    task.mkdir(parents=True)
    fixed_head = "a" * 40
    monkeypatch.setattr(register, "_git_head", lambda root: fixed_head)
    monkeypatch.setattr(gate, "_git_head", lambda root: fixed_head)

    def file_record(path: Path, data: bytes) -> dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": contract.sha256_file(path)}

    def json_record(path: Path, payload: dict[str, object]) -> dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return file_record(path, path.read_bytes())

    plan_record = json_record(task / "plan.json", {"schema": "synthetic-plan", "task_id": contract.TASK_ID, "truth_opened": False})
    tokenizer_dir = repo / "tokenizer"
    tokenizer_file = file_record(tokenizer_dir / "tokenizer.json", b"synthetic tokenizer")
    pile_arrow = file_record(repo / "pile.arrow", b"synthetic pile arrow")
    finance_arrow = file_record(repo / "finance.arrow", b"synthetic finance arrow")
    rows = {
        style: [{"record_id": f"{style}-{index}", "public_record_sha256": f"{index + 1:064x}", "dataset_key": style, "valid_tokens": contract.STORED_SEQUENCE_TOKENS, "final_sequence_sha256": f"{index + 101:064x}"} for index in range(records)]
        for style in contract.DOMAIN_ORDER
    }
    selection_payload = {"schema": "token-reconstruction.trr0009-source-selection.v1", "task_id": contract.TASK_ID, "status": "FROZEN_TRR0009_SOURCE_SELECTION_NO_TRUTH", "records_by_domain": {"pile": records, "finance": records}, "target_conditions": list(contract.TARGET_ORDER), "paired_conditions": True, "selection_rule": {"record_ids_sha256": {style: contract.canonical_json_digest([row["record_id"] for row in rows[style]]) for style in contract.DOMAIN_ORDER}, "records": rows}, "public_sources_frozen": {"pile": {"arrow_files": [pile_arrow]}, "finance": {"arrow_files": [finance_arrow]}, "tokenizer": {"path": str(tokenizer_dir.resolve()), "files": {"tokenizer.json": tokenizer_file}}}, "truth_opened": False, "source_text_loaded": False, "source_text_written": False, "token_ids_written": False, "target_labels_loaded": False, "candidate_arrays_persisted": False}
    selection_record = json_record(task / "selection.json", selection_payload)
    frequency_record = json_record(repo / "frequency.json", {"schema": "token-reconstruction.trr0005-frequency-references.v1", "task_id": "TRR-0005", "status": "PUBLIC_FITTING_FREQUENCY_REFERENCES", "frequency_references": {"enriched": {"1": 3}, "original": {"1": 3}}, "metadata": {"truth_accessed": False, "fresh_data_accessed": False, "tensor_data_loaded": public_fitting_tensor_metadata, "public_fitting_tensors_loaded": public_fitting_tensor_metadata}})
    embedding_record = file_record(repo / "embedding.bin", b"embedding")
    timing_plan = {
        "schema": contract.TIMING_PLAN_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_TIMING_PLAN_BEFORE_MEASUREMENT",
        "method_order": list(contract.METHOD_ORDER),
        "cell_order": list(contract.CELL_ORDER),
        "records_per_cell": 32,
        "blocks": 40,
        "timing_paths": list(contract.METHOD_ORDER) + ["continued_fixed_readout__alias"],
        "block_cycle_length": 10,
        "block_cycles": 4,
        "warmup_runs": 1,
        "measured_runs": 1,
        "candidate_ratio_threshold": 1.25,
        "alias_control": {"band": [0.95, 1.05]},
        "truth_opened": False,
    }
    from scripts import trr0009_timing as timing_module
    timing_rows, timing_schedule_sha256 = timing_module.schedule_rows()
    timing_plan.update({"generated_utc": "2026-09-07T00:00:00Z", "seed": 8008, "ci_level": 0.95, "maximum_seconds": 600, "schedule_sha256": timing_schedule_sha256, "schedule_rows": timing_rows})
    timing_plan_record = json_record(task / "timing_plan.json", timing_plan)

    observations_dir = task / "capture"
    observations_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    records_by_domain = {"pile": records, "finance": records}
    activations = torch.zeros((records, 128, 2048), dtype=torch.bfloat16)
    mask = torch.ones((records, 128), dtype=torch.uint8)
    positions = torch.arange(128, dtype=torch.long).expand(records, -1).contiguous()
    for cell_id in contract.CELL_ORDER:
        obs_path = observations_dir / f"{cell_id}.safetensors"
        style = cell_id.split("__", 1)[0]
        save_file({"activations": activations, "attention_mask": mask, "position_ids": positions}, str(obs_path), metadata={"schema": "token-reconstruction.trr0009-public-observation.v1", "task_id": contract.TASK_ID, "cell_id": cell_id, "shape": json.dumps([records, 128, 2048]), "selection_plan_sha256": selection_record["sha256"], "record_ids_sha256": selection_payload["selection_rule"]["record_ids_sha256"][style], "source_text_written": "false", "token_ids_written": "false", "target_labels_loaded": "false", "truth_opened": "false"})
        obs = {"path": str(obs_path.resolve()), "bytes": obs_path.stat().st_size, "sha256": contract.sha256_file(obs_path), "shape": [records, 128, 2048], "activations_key": "activations", "attention_mask_key": "attention_mask", "position_ids_key": "position_ids"}
        style, condition = cell_id.split("__", 1)
        cells.append({"cell_id": cell_id, "style": style, "condition": condition, "records": records, "observation": obs})
    observation_record = json_record(task / "observations.json", {"schema": contract.OBSERVATION_SCHEMA, "task_id": contract.TASK_ID, "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH", "records_by_domain": records_by_domain, "cell_order": list(contract.CELL_ORDER), "cells": cells, "truth_opened": False})
    model_dir = repo / "model-dir"
    model_file = file_record(model_dir / "model.json", b"synthetic model")
    source_code_file = file_record(task / "capture_adapter.py", b"# synthetic capture adapter")
    capture_record = json_record(task / "capture.json", {"schema": "token-reconstruction.trr0009-public-capture.v1", "task_id": contract.TASK_ID, "status": "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH", "selection_plan": selection_record, "observations": observation_record, "public_inputs": {"pile": {"arrow_files": [pile_arrow]}, "finance": {"arrow_files": [finance_arrow]}, "tokenizer": {"path": str(tokenizer_dir.resolve()), "files": {"tokenizer.json": tokenizer_file}}, "model_snapshot": {"path": str(model_dir.resolve()), "files": {"model.json": model_file}}}, "source_code": {"adapter": source_code_file}, "execution": {"truth_opened": False, "target_labels_loaded": False}, "truth_opened": False, "source_text_loaded": False, "source_text_written": False, "token_ids_written": False, "target_labels_loaded": False, "token_ids_written": False, "candidate_arrays_persisted": False})

    methods = {}
    code_bindings = []
    for index, method_id in enumerate(contract.METHOD_ORDER):
        state_path = task / "states" / f"{method_id}.bin"
        if method_id == contract.ADAPTABLE_METHOD_ID:
            state_path = state_path.with_suffix(".safetensors")
            support_ids = torch.tensor([1], dtype=torch.long)
            support_counts = torch.tensor([3], dtype=torch.long)
            save_file({"support_ids": support_ids, "support_counts": support_counts}, str(state_path), metadata={"schema": "token-reconstruction.trr0009-supported-readout.v1", "support_digest": model_contract.support_digest(support_ids, support_counts), "support_count": "1", "current_H_only": "true", "full_vocabulary_cross_entropy": "true", "inference_contract": "current_activation_H_i_only; supported-row calibrated full-vocabulary E"})
            state_record = {"path": str(state_path.resolve()), "bytes": state_path.stat().st_size, "sha256": contract.sha256_file(state_path)}
        else:
            state_record = file_record(state_path, f"state-{index}".encode())
        methods[method_id] = {"id": method_id, "role": contract.METHOD_ROLES[method_id], "state_path": state_record["path"], "loader": {"module": "synthetic", "function": "load", "interface": contract.LOADER_INTERFACE, "current_h_only": True, "full_vocabulary": True, "history_enabled": False, "a2_enabled": False, "kwargs": {}, "path_args": {}, "tensor_args": {}}}
    for index in range(2):
        code_bindings.append(file_record(task / "code" / f"binding{index}.py", f"# binding {index}".encode()))

    registration_path = task / "evaluation" / "registration.json"
    registration = register.build_registration(
        repository_root=repo,
        plan_path=Path(plan_record["path"]),
        observation_manifest_path=Path(observation_record["path"]),
        source_selection_path=Path(selection_record["path"]),
        capture_receipt_path=Path(capture_record["path"]),
        frequency_reference_path=Path(frequency_record["path"]),
        embedding_path=Path(embedding_record["path"]),
        timing_plan_path=Path(timing_plan_record["path"]),
        method_rows=methods,
        records_by_domain=records_by_domain,
        code_bindings=[Path(row["path"]) for row in code_bindings],
        output_root="experiments/TRR-0009/evaluation/predictions",
        output_path=registration_path,
        initialization_equivalence={"status": "PASS", "scope": "synthetic fixture"},
    )
    registration_sha = registration["registration_sha256"]
    output_root = Path(registration["output_root"])
    predictions = {}
    timings = {}
    base = torch.full((records, 128), contract.BOS_TOKEN_ID, dtype=torch.long)
    base[:, 1:] = 1
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            path = contract.expected_prediction_path(output_root, cell_id=cell_id, method_id=method_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"predictions": base}, str(path), metadata={"schema": contract.PREDICTION_SCHEMA, "task_id": contract.TASK_ID, "registration_sha256": registration_sha, "cell_id": cell_id, "method_id": method_id, "records": str(records), "geometry_json": json.dumps({"records": records, **contract.STATIC_GEOMETRY}, sort_keys=True), "truth_opened": "false", "candidate_arrays_persisted": "false"})
            prediction = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": contract.sha256_file(path), "prediction_sha256": contract.tensor_digest(base), "records": records}
            timing_path = contract.expected_timing_path(output_root, cell_id=cell_id, method_id=method_id)
            timing_payload = {"schema": contract.TIMING_SCHEMA, "task_id": contract.TASK_ID, "method_id": method_id, "cell_id": cell_id, "records": records, "warmup_runs_per_record": 1, "measured_runs_per_record": 1, "warmup_output_exact_match_measured": True, "measured_output_selected": True, "warmup_seconds_sum": 0.008, "measured_seconds_sum": 0.008, "model_preparation_seconds": 0.001, "peak_memory": {"process_max_rss_bytes": 1, "host_available_bytes": 1}, "per_record_measured_seconds": [0.001] * records, "prediction_artifact": prediction, "truth_opened": False, "candidate_arrays_persisted": False}
            timing_path.write_text(json.dumps(timing_payload, sort_keys=True), encoding="utf-8")
            timing_record = {"path": str(timing_path.resolve()), "bytes": timing_path.stat().st_size, "sha256": contract.sha256_file(timing_path)}
            key = f"{method_id}::{cell_id}"
            predictions[key] = prediction
            timings[key] = timing_record
    run_path = output_root / "run_manifest.json"
    run_payload = {"schema": contract.RUN_SCHEMA, "task_id": contract.TASK_ID, "status": "PUBLIC_PREDICTIONS_COMPLETE_BEFORE_TRUTH", "code_commit": fixed_head, "registration": {"path": str(registration_path.resolve()), "bytes": registration_path.stat().st_size, "sha256": contract.sha256_file(registration_path)}, "observation_manifest": registration["observation_manifest"], "predictions": predictions, "timings": timings, "model_startup": {method: {"model_preparation_seconds": 0.001} for method in contract.METHOD_ORDER}, "truth_opened": False, "candidate_arrays_persisted": False}
    run_path.write_text(json.dumps(run_payload, sort_keys=True), encoding="utf-8")
    return {"repo": repo, "registration_path": registration_path, "run_path": run_path, "frequency_path": Path(frequency_record["path"]), "base": base, "run_payload": run_payload, "registration": registration}


def test_public_fitting_tensor_metadata_remains_truth_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch, public_fitting_tensor_metadata=True)
    receipt = gate.validate_public_outputs(
        registration_path=fixture["registration_path"],
        run_manifest_path=fixture["run_path"],
        repository_root=fixture["repo"],
    )
    assert receipt["status"] == "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH"


def test_public_gate_success_and_missing_registration_bytes_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = gate.validate_public_outputs(registration_path=fixture["registration_path"], run_manifest_path=fixture["run_path"], repository_root=fixture["repo"])
    assert receipt["status"] == "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH"
    bad = json.loads(fixture["run_path"].read_text())
    bad["registration"].pop("bytes")
    fixture["run_path"].write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(gate.GateError, match="bytes"):
        gate.validate_public_outputs(registration_path=fixture["registration_path"], run_manifest_path=fixture["run_path"], repository_root=fixture["repo"])


def test_prediction_corruption_is_rejected_before_truth_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    target_key = f"{contract.PRIMARY_METHOD_ID}::{contract.CELL_ORDER[0]}"
    path = Path(fixture["run_payload"]["predictions"][target_key]["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    calls = {"count": 0}
    def loader():
        calls["count"] += 1
        return {cell: fixture["base"].clone() for cell in contract.CELL_ORDER}
    with pytest.raises(gate.GateError):
        gate.validate_public_outputs(registration_path=fixture["registration_path"], run_manifest_path=fixture["run_path"], repository_root=fixture["repo"])
    assert calls["count"] == 0


def test_post_gate_truth_adapter_opens_synthetic_truth_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    freeze_path = fixture["repo"] / "experiments" / contract.TASK_ID / "evaluation" / "freeze.json"
    gate.write_freeze(registration_path=fixture["registration_path"], run_manifest_path=fixture["run_path"], output_path=freeze_path, repository_root=fixture["repo"])
    calls = {"count": 0}
    def loader():
        calls["count"] += 1
        return {cell: fixture["base"].clone() for cell in contract.CELL_ORDER}
    out = fixture["repo"] / "experiments" / contract.TASK_ID / "evaluation" / "score.json"
    result = truth_adapter.score_after_gate(freeze_path=freeze_path, repository_root=fixture["repo"], truth_loader=loader, frequency_reference_path=fixture["frequency_path"], output_path=out, truth_binding={"schema": truth_adapter.TRUTH_BINDING_SCHEMA, "status": truth_adapter.TRUTH_STATUS, "truth_opened": False}, bootstrap_draws=10)
    assert result["status"] == "SCORE_COMPLETE_AFTER_PUBLIC_FREEZE"
    assert result["truth_opened"] is True
    assert calls["count"] == 1


def test_different_prediction_root_and_altered_state_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    bad = json.loads(fixture["run_path"].read_text())
    key = f"{contract.PRIMARY_METHOD_ID}::{contract.CELL_ORDER[0]}"
    bad["predictions"][key]["path"] = str(fixture["repo"] / "other-root" / "prediction.safetensors")
    fixture["run_path"].write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(gate.GateError, match="prediction root/path"):
        gate.validate_public_outputs(registration_path=fixture["registration_path"], run_manifest_path=fixture["run_path"], repository_root=fixture["repo"])

    fixture = _fixture(tmp_path / "state", monkeypatch)
    state_path = Path(fixture["registration"]["methods"][0]["state"]["path"])
    state_path.write_bytes(state_path.read_bytes() + b"changed")
    with pytest.raises(gate.GateError, match="hash or size changed"):
        gate.validate_public_outputs(registration_path=fixture["registration_path"], run_manifest_path=fixture["run_path"], repository_root=fixture["repo"])


def test_pretruth_gate_rejects_altered_authoritative_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    freeze_path = fixture["repo"] / "experiments" / contract.TASK_ID / "evaluation" / "freeze.json"
    gate.write_freeze(registration_path=fixture["registration_path"], run_manifest_path=fixture["run_path"], output_path=freeze_path, repository_root=fixture["repo"])
    freeze = json.loads(freeze_path.read_text())
    key = f"{contract.PRIMARY_METHOD_ID}::{contract.CELL_ORDER[0]}"
    freeze["predictions"][key]["prediction_sha256"] = "0" * 64
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(gate.GateError, match="public freeze binding changed"):
        gate.validate_before_truth(freeze_path=freeze_path, repository_root=fixture["repo"])



def test_cpu_runner_entrypoint_writes_complete_matrix_and_gate_accepts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real runner boundary with only model/embedding calls stubbed.

    The fixture still traverses registered observation files, create-only
    prediction/timing writers, registration recheck, and the public gate. The
    model and 1 GB embedding are replaced so this checks startup/data/shape
    interfaces without a GPU or fresh source/truth access.
    """
    fixture = _fixture(tmp_path, monkeypatch)
    output_root = Path(fixture["registration"]["output_root"])
    for path in sorted(output_root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()

    fixed_head = "a" * 40
    monkeypatch.setattr(runner, "_git_head", lambda root: fixed_head)
    monkeypatch.setattr(runner, "_configure_numerics", lambda settings: {"settings": dict(settings)})
    monkeypatch.setattr(runner, "_guard", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_synchronize", lambda device: None)
    monkeypatch.setattr(runner, "_rss_bytes", lambda: 1)
    monkeypatch.setattr(runner, "_host_available_bytes", lambda: 1)
    monkeypatch.setattr(
        runner,
        "_load_embedding",
        lambda registration, root, device: (torch.zeros((1, 1), dtype=torch.float32), {"synthetic": True}),
    )

    class DummyDecoder(torch.nn.Module):
        hidden_size = contract.HIDDEN_SIZE
        vocabulary_size = contract.VOCABULARY_SIZE

    def load_decoder(row, *, root, device):
        return DummyDecoder(), {"method_id": row["id"], "synthetic": True}

    monkeypatch.setattr(runner, "_load_decoder", load_decoder)
    expected = fixture["base"]
    monkeypatch.setattr(
        runner,
        "predict_current_h",
        lambda model, embedding, activation, valid_mask, *, device: expected[0].clone(),
    )

    code = runner.main([
        "--repository-root", str(fixture["repo"]),
        "--registration", str(fixture["registration_path"]),
        "--device", "cpu",
    ])
    assert code == 0
    run_path = output_root / "run_manifest.json"
    result = json.loads(run_path.read_text())
    assert result["status"] == "PUBLIC_PREDICTIONS_COMPLETE_BEFORE_TRUTH"
    assert len(result["predictions"]) == len(contract.METHOD_ORDER) * len(contract.CELL_ORDER)
    assert len(result["timings"]) == len(result["predictions"])
    assert result["truth_opened"] is False
    gate_receipt = gate.validate_public_outputs(
        registration_path=fixture["registration_path"],
        run_manifest_path=run_path,
        repository_root=fixture["repo"],
    )
    assert gate_receipt["status"] == "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH"


def test_balanced_timing_receipt_binds_frozen_prefixes_and_rejects_digest_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate the real balanced receipt path against the public run matrix."""
    from scripts import trr0009_timing as timing

    fixture = _fixture(tmp_path, monkeypatch, records=32)
    registration = fixture["registration"]
    rows, schedule_sha256 = timing.schedule_rows()
    prefix_digest = contract.tensor_digest(fixture["base"][:32].contiguous())
    blocks = []
    for block_index in range(timing.DEFAULT_BLOCKS):
        cell_orders = [row for row in rows if int(row["block_index"]) == block_index]
        entries = []
        for row in cell_orders:
            for order_index, method_id in enumerate(row["order"]):
                entries.append({
                    "method_id": method_id,
                    "cell_id": row["cell_id"],
                    "block_index": block_index,
                    "order_index": order_index,
                    "records": 32,
                    "warmup_runs_per_record": 1,
                    "measured_runs_per_record": 1,
                    "warmup_seconds_sum": 0.032,
                    "measured_seconds_sum": 0.032,
                    "per_record_measured_seconds": [0.001] * 32,
                    "warmup_variability": timing.method_variability([0.001] * 32),
                    "measured_variability": timing.method_variability([0.001] * 32),
                    "warmup_output_exact_match_measured": True,
                    "measured_output_selected": True,
                    "prediction_sha256": prefix_digest,
                    "peak_memory": {"process_max_rss_bytes": 1, "host_available_bytes": 1, "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None},
                    "timed_interval": "synchronized BF16 current-H staging -> FP32 decoder -> full-vocabulary argmax -> CPU IDs",
                    "truth_opened": False,
                    "candidate_arrays_persisted": False,
                })
        blocks.append({
            "block_index": block_index,
            "order_by_cell": {row["cell_id"]: list(row["order"]) for row in cell_orders},
            "started_utc": "2026-09-07T00:00:00Z",
            "ended_utc": "2026-09-07T00:00:01Z",
            "wall_seconds": 1.0,
            "telemetry_before": {"utc": "2026-09-07T00:00:00Z"},
            "telemetry_after": {"utc": "2026-09-07T00:00:01Z"},
            "resource_guard_before": {"status": "PASS"},
            "resource_guard_after": {"status": "PASS"},
            "resource_guard_overhead_seconds": 0.0,
            "entries": entries,
        })

    method_bindings = {
        method_id: {"state": row["state"], "loader": row["loader"]}
        for method_id, row in ((str(row["id"]), row) for row in registration["methods"])
    }
    method_bindings[timing.ALIAS_METHOD_ID] = method_bindings[contract.PRIMARY_CONTROL_METHOD_ID]
    prefix_digests = {f"{method_id}::{cell_id}": prefix_digest for method_id in contract.METHOD_ORDER for cell_id in contract.CELL_ORDER}
    run_path = fixture["run_path"]
    run_record = {"path": str(run_path.resolve()), "bytes": run_path.stat().st_size, "sha256": contract.sha256_file(run_path)}
    receipt = {
        "schema": timing.SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "TIMING_COMPLETE",
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "candidate_arrays_persisted": False,
        "code_commit": registration["code_commit"],
        "registration": registration["registration_file"],
        "observation_manifest": registration["observation_manifest"],
        "timing_plan": registration["timing_plan"],
        "code_bindings": registration["code_bindings"],
        "resource_guard": registration["resource_guard"],
        "numerical_settings": {"settings": registration["numerical_settings"]},
        "configuration": {"device": "cpu", "records_per_cell": 32, "blocks": 40, "block_cycle_length": 10, "block_cycles": 4, "warmup_runs_per_record": 1, "measured_runs_per_record": 1, "seed": 8008, "maximum_seconds": 600, "candidate_ratio_threshold": 1.25, "ci_level": 0.95},
        "method_bindings": method_bindings,
        "alias_execution_identity": {"status": "PASS", "exact_prediction_equivalence": True, "identical_state_loader_binding": True},
        "order_schedule": {"rows": rows, "sha256": schedule_sha256},
        "blocks": blocks,
        "summary": timing.summarize_blocks(blocks),
        "frozen_prediction_run_manifest": run_record,
        "frozen_prediction_prefix_digests": prefix_digests,
    }
    timing_path = fixture["repo"] / "experiments" / contract.TASK_ID / "evaluation" / "timing" / "result.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    public = gate.validate_public_outputs(
        registration_path=fixture["registration_path"],
        run_manifest_path=fixture["run_path"],
        repository_root=fixture["repo"],
        balanced_timing_path=timing_path,
    )
    assert public["status"] == "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH"
    assert public["balanced_timing"]["sha256"] == contract.sha256_file(timing_path)

    corrupted = json.loads(timing_path.read_text())
    corrupted["blocks"][0]["entries"][0]["prediction_sha256"] = "0" * 64
    timing_path.write_text(json.dumps(corrupted, sort_keys=True), encoding="utf-8")
    with pytest.raises(gate.GateError, match="summary changed|prediction differs"):
        gate.validate_public_outputs(
            registration_path=fixture["registration_path"],
            run_manifest_path=fixture["run_path"],
            repository_root=fixture["repo"],
            balanced_timing_path=timing_path,
        )
