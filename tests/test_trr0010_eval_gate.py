"""Synthetic, truth-blind tests for the TRR-0010 public gate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from scripts import trr0010_eval_gate as gate


def _json_file(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _small_synthetic_observation_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a test-only hidden width while retaining production H=2048 in the gate."""
    monkeypatch.setattr(gate, "OBSERVATION_HIDDEN_SIZE", 2)


def _public_payload(schema: str, *, record_ids: dict[str, str] | None = None) -> dict[str, Any]:
    payload = {
        "schema": schema,
        "task_id": gate.TASK_ID,
        "truth_opened": False,
        "source_text_written": False,
        "source_text_loaded": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    if record_ids is not None:
        payload["record_ids_sha256"] = dict(record_ids)
    return payload


def _fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path
    task_root = root / "experiments" / gate.TASK_ID
    output_root = task_root / "evaluation"
    output_root.mkdir(parents=True)
    assets = root / "assets"
    assets.mkdir()

    contract_path = _json_file(assets / "contract.json", _public_payload("contract"))
    selection_record_ids = {"pile": "b" * 64, "finance": "c" * 64}
    input_paths = {
        name: _json_file(
            assets / f"{name}.json",
            _public_payload(
                f"{name}.v1",
                record_ids=selection_record_ids if name == "source_selection" else None,
            ),
        )
        for name in ("source_selection", "public_observations", "frequency_reference")
    }
    observation_paths: dict[str, Path] = {}
    observation_bindings: dict[str, dict[str, Any]] = {}
    for cell_id in gate.CELL_ORDER:
        domain = cell_id.split("__", 1)[0]
        shape = [gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS, gate.OBSERVATION_HIDDEN_SIZE]
        observation_path = assets / f"{cell_id}.safetensors"
        observation_path.parent.mkdir(parents=True, exist_ok=True)
        observation_metadata = {
            "schema": gate.OBSERVATION_SCHEMA,
            "task_id": gate.TASK_ID,
            "cell_id": cell_id,
            "records": str(gate.RECORDS_PER_CELL),
            "shape": json.dumps(shape),
            "record_ids_sha256": selection_record_ids[domain],
            "truth_opened": "false",
            "source_text_written": "false",
            "token_ids_written": "false",
            "target_labels_loaded": "false",
        }
        save_file(
            {
                "activations": torch.zeros(shape, dtype=torch.bfloat16),
                "attention_mask": torch.ones((shape[0], shape[1]), dtype=torch.uint8),
                "position_ids": torch.arange(shape[1], dtype=torch.int64).repeat(shape[0], 1),
            },
            str(observation_path),
            metadata=observation_metadata,
        )
        observation_paths[cell_id] = observation_path
        observation_bindings[cell_id] = {
            "cell_id": cell_id,
            "records": gate.RECORDS_PER_CELL,
            "shape": shape,
            "record_ids_sha256": selection_record_ids[domain],
            "observation": gate.file_record(observation_path, root=root),
        }
    timing_path = _json_file(assets / "timing_plan.json", _public_payload("timing_plan.v1"))
    code_paths = {
        name: (assets / f"{name}.py")
        for name in ("gate", "runner")
    }
    for path in code_paths.values():
        path.write_text("# synthetic public code binding\n", encoding="utf-8")
    state_paths = {}
    for method_id in gate.METHOD_ORDER:
        path = assets / f"{method_id}.state"
        path.write_bytes(f"synthetic state for {method_id}\n".encode("utf-8"))
        state_paths[method_id] = path
    public_embedding_path = assets / "public_embedding_table.asset"
    public_embedding_path.write_bytes(b"synthetic public embedding E\n")
    a1_a2_resource_paths = {"public_embedding_table": public_embedding_path}
    for name in ("retained_a1_lens", "public_p0_prefix"):
        path = assets / f"{name}.asset"
        path.write_bytes(f"synthetic A1+A2 {name}\n".encode("utf-8"))
        a1_a2_resource_paths[name] = path
    method_resource_paths: dict[str, dict[str, Path]] = {}
    for method_id in gate.METHOD_ORDER:
        requirements = gate.METHOD_RESOURCE_REQUIREMENTS[method_id]
        resources: dict[str, Path] = {}
        for name in requirements:
            if name == "public_embedding_table":
                resources[name] = public_embedding_path
            elif name == "base_decoder_state":
                resources[name] = state_paths[method_id]
            else:
                path = assets / f"{method_id}.{name}.asset"
                path.write_bytes(f"synthetic {method_id} {name}\n".encode("utf-8"))
                resources[name] = path
        method_resource_paths[method_id] = resources

    registration_path = output_root / "registration.json"
    registration = {
        "schema": gate.REGISTRATION_SCHEMA,
        "task_id": gate.TASK_ID,
        "status": gate.REGISTRATION_STATUS,
        "method_order": list(gate.METHOD_ORDER),
        "cell_order": list(gate.CELL_ORDER),
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "geometry": dict(gate.STATIC_GEOMETRY),
        "code_commit": "a" * 40,
        "output_root": str(output_root),
        "contract_binding": gate.file_record(contract_path, root=root),
        "input_bindings": {
            name: gate.file_record(path, root=root)
            for name, path in input_paths.items()
        },
        "observation_bindings": observation_bindings,
        "timing_plan": gate.file_record(timing_path, root=root),
        "code_bindings": {
            name: gate.file_record(path, root=root)
            for name, path in code_paths.items()
        },
        "initialization_equivalence": {"status": "PASS"},
        "truth_opened": False,
        "source_text_written": False,
        "source_text_loaded": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "methods": [
            {
                "id": method_id,
                "role": gate.METHOD_ROLES[method_id],
                "cells": list(gate.CELL_ORDER),
                "records_per_cell": {cell: gate.RECORDS_PER_CELL for cell in gate.CELL_ORDER},
                "state": gate.file_record(state_paths[method_id], root=root),
                "loader": (
                    {
                        "interface": gate.A1_A2_LOADER_INTERFACE,
                        "current_h_only": False,
                        "full_vocabulary": True,
                        "history_enabled": False,
                        "a2_enabled": True,
                        "reconstructed_prefix": True,
                        "candidate_k": 256,
                    }
                    if method_id == gate.A1_A2_METHOD_ID
                    else {
                        "interface": gate.LOADER_INTERFACE,
                        "current_h_only": True,
                        "full_vocabulary": True,
                        "history_enabled": False,
                        "a2_enabled": False,
                    }
                ),
                **(
                    {
                        "resources": {
                            name: gate.file_record(path, root=root)
                            for name, path in method_resource_paths[method_id].items()
                        }
                    }
                ),
            }
            for method_id in gate.METHOD_ORDER
        ],
    }
    _json_file(registration_path, registration)
    registration_record = gate.file_record(registration_path, root=root)

    predictions: dict[str, dict[str, Any]] = {}
    timings: dict[str, dict[str, Any]] = {}
    values = torch.zeros((gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS), dtype=torch.long)
    values[:, 0] = gate.BOS_TOKEN_ID
    for method_id in gate.METHOD_ORDER:
        for cell_id in gate.CELL_ORDER:
            style, condition = cell_id.split("__", 1)
            prediction_path = output_root / "predictions" / style / condition / f"{method_id}.safetensors"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "schema": gate.PREDICTION_SCHEMA,
                "task_id": gate.TASK_ID,
                "registration_sha256": registration_record["sha256"],
                "method_id": method_id,
                "cell_id": cell_id,
                "records": str(gate.RECORDS_PER_CELL),
                "truth_opened": "false",
                "candidate_arrays_persisted": "false",
                "geometry_json": json.dumps(
                    {"records": gate.RECORDS_PER_CELL, **gate.STATIC_GEOMETRY},
                    sort_keys=True,
                ),
            }
            save_file({"predictions": values.clone()}, str(prediction_path), metadata=metadata)
            prediction_record = gate.file_record(prediction_path, root=root)
            prediction_record["prediction_sha256"] = gate.tensor_digest(values)
            key = f"{method_id}::{cell_id}"
            predictions[key] = dict(prediction_record)

            timing_path = output_root / "timings" / style / condition / f"{method_id}.run.json"
            timing_payload = {
                "schema": gate.TIMING_SCHEMA,
                "task_id": gate.TASK_ID,
                "method_id": method_id,
                "cell_id": cell_id,
                "records": gate.RECORDS_PER_CELL,
                "registration_sha256": registration_record["sha256"],
                "warmup_runs_per_record": 1,
                "measured_runs_per_record": 1,
                "warmup_output_exact_match_measured": True,
                "measured_output_selected": True,
                "per_record_measured_seconds": [0.001] * gate.RECORDS_PER_CELL,
                "warmup_seconds_sum": 0.1,
                "measured_seconds_sum": 0.128,
                "model_preparation_seconds": 0.01,
                "peak_memory": {
                    "process_max_rss_bytes": 1000000,
                    "host_available_bytes": 1000000000,
                },
                "prediction_artifact": prediction_record,
                "truth_opened": False,
                "source_text_written": False,
                "source_text_loaded": False,
                "token_ids_written": False,
                "target_labels_loaded": False,
                "candidate_arrays_persisted": False,
            }
            _json_file(timing_path, timing_payload)
            timings[key] = gate.file_record(timing_path, root=root)

    run_path = output_root / "run_manifest.json"
    run = {
        "schema": gate.RUN_SCHEMA,
        "task_id": gate.TASK_ID,
        "status": gate.RUN_STATUS,
        "code_commit": registration["code_commit"],
        "registration": registration_record,
        "input_bindings": registration["input_bindings"],
        "observation_bindings": registration["observation_bindings"],
        "code_bindings": registration["code_bindings"],
        "timing_plan": registration["timing_plan"],
        "predictions": predictions,
        "timings": timings,
        "truth_opened": False,
        "source_text_written": False,
        "source_text_loaded": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    _json_file(run_path, run)
    freeze_path = task_root / "public_freeze.json"
    return {
        "root": root,
        "task_root": task_root,
        "output_root": output_root,
        "registration": registration_path,
        "run": run_path,
        "freeze": freeze_path,
        "inputs": input_paths,
        "states": state_paths,
        "code": code_paths,
        "observations": observation_paths,
        "a1_a2_resources": a1_a2_resource_paths,
        "method_resources": method_resource_paths,
    }


def _gate_then_open(fixture: dict[str, Any], opened: list[str]) -> dict[str, Any]:
    result = gate.validate_public_outputs(
        registration_path=fixture["registration"],
        run_manifest_path=fixture["run"],
        repository_root=fixture["root"],
        require_current_head=False,
    )
    opened.append("truth-opener-would-run-here")
    return result




def _rewrite_prediction(fixture: dict[str, Any], key: str, values: torch.Tensor) -> None:
    run = json.loads(fixture["run"].read_text(encoding="utf-8"))
    method_id, cell_id = key.split("::", 1)
    prediction_path = Path(run["predictions"][key]["path"])
    registration_sha = gate.file_record(fixture["registration"], root=fixture["root"])["sha256"]
    metadata = {
        "schema": gate.PREDICTION_SCHEMA,
        "task_id": gate.TASK_ID,
        "registration_sha256": registration_sha,
        "method_id": method_id,
        "cell_id": cell_id,
        "records": str(gate.RECORDS_PER_CELL),
        "truth_opened": "false",
        "candidate_arrays_persisted": "false",
        "geometry_json": json.dumps(
            {"records": gate.RECORDS_PER_CELL, **gate.STATIC_GEOMETRY},
            sort_keys=True,
        ),
    }
    save_file({"predictions": values}, str(prediction_path), metadata=metadata)
    prediction_record = gate.file_record(prediction_path, root=fixture["root"])
    prediction_record["prediction_sha256"] = gate.tensor_digest(values)
    run["predictions"][key] = prediction_record
    timing_path = Path(run["timings"][key]["path"])
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    timing_payload["prediction_artifact"] = prediction_record
    _json_file(timing_path, timing_payload)
    run["timings"][key] = gate.file_record(timing_path, root=fixture["root"])
    _json_file(fixture["run"], run)

def test_complete_six_method_matrix_passes_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    opened: list[str] = []
    freeze = _gate_then_open(fixture, opened)
    assert opened == ["truth-opener-would-run-here"]
    assert freeze["status"] == gate.FREEZE_STATUS
    assert freeze["method_order"] == list(gate.METHOD_ORDER)
    assert len(freeze["predictions"]) == 24
    assert len(freeze["timings"]) == 24

    gate.write_freeze(
        registration_path=fixture["registration"],
        run_manifest_path=fixture["run"],
        output_path=fixture["freeze"],
        repository_root=fixture["root"],
    )
    assert gate.validate_before_truth(freeze_path=fixture["freeze"], repository_root=fixture["root"])["status"] == gate.FREEZE_STATUS


def test_missing_prediction_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["output_root"].joinpath("predictions", "pile", "public_base", f"{gate.METHOD_ORDER[0]}.safetensors").unlink()
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="prediction"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_duplicate_prediction_path_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run = json.loads(fixture["run"].read_text(encoding="utf-8"))
    first = f"{gate.METHOD_ORDER[0]}::{gate.CELL_ORDER[0]}"
    second = f"{gate.METHOD_ORDER[1]}::{gate.CELL_ORDER[0]}"
    run["predictions"][second] = dict(run["predictions"][first])
    fixture["run"].write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="prediction path changed"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_foreign_prediction_key_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run = json.loads(fixture["run"].read_text(encoding="utf-8"))
    key = f"{gate.METHOD_ORDER[0]}::{gate.CELL_ORDER[0]}"
    run["predictions"]["foreign::pile__public_base"] = run["predictions"].pop(key)
    fixture["run"].write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="incomplete, duplicated, or foreign"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_changed_input_binding_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["inputs"]["source_selection"].write_text("changed after registration\n", encoding="utf-8")
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="input source_selection hash or size changed"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_changed_state_binding_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["states"][gate.METHOD_ORDER[2]].write_bytes(b"changed state\n")
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="state current_directional hash or size changed"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_changed_code_binding_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["code"]["runner"].write_text("changed runner implementation\n", encoding="utf-8")
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="code runner hash or size changed"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_freeze_revalidation_rejects_changed_binding_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    gate.write_freeze(
        registration_path=fixture["registration"],
        run_manifest_path=fixture["run"],
        output_path=fixture["freeze"],
        repository_root=fixture["root"],
    )
    fixture["inputs"]["public_observations"].write_text("changed after freeze\n", encoding="utf-8")
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="input public_observations hash or size changed"):
        try:
            gate.validate_before_truth(freeze_path=fixture["freeze"], repository_root=fixture["root"])
        except gate.GateError:
            # The test models the caller's ordering: the truth opener is after
            # validation, and therefore cannot run on this failure.
            raise
        opened.append("truth-opener-would-run-here")
    assert opened == []


def test_native_a1_a2_requires_prefix_resources(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registration = json.loads(fixture["registration"].read_text(encoding="utf-8"))
    row = next(row for row in registration["methods"] if row["id"] == gate.A1_A2_METHOD_ID)
    row["resources"].pop("public_p0_prefix")
    _json_file(fixture["registration"], registration)
    opened: list[str] = []
    with pytest.raises(gate.GateError, match=r"A1\+A2 E/lens/public-prefix resources"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_float_prediction_rejected_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    key = f"{gate.METHOD_ORDER[0]}::{gate.CELL_ORDER[0]}"
    values = torch.zeros((gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS), dtype=torch.float32)
    values[:, 0] = gate.BOS_TOKEN_ID
    _rewrite_prediction(fixture, key, values)
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="signed integer"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_incomplete_prediction_suffix_rejected_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    key = f"{gate.METHOD_ORDER[0]}::{gate.CELL_ORDER[0]}"
    values = torch.zeros((gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS), dtype=torch.long)
    values[:, 0] = gate.BOS_TOKEN_ID
    values[:, -1] = gate.INVALID_TOKEN_ID
    _rewrite_prediction(fixture, key, values)
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="incomplete or invalid"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_nonfinite_timing_rejected_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    key = f"{gate.METHOD_ORDER[0]}::{gate.CELL_ORDER[0]}"
    run = json.loads(fixture["run"].read_text(encoding="utf-8"))
    timing_path = Path(run["timings"][key]["path"])
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    timing_payload["per_record_measured_seconds"][0] = float("nan")
    _json_file(timing_path, timing_payload)
    run["timings"][key] = gate.file_record(timing_path, root=fixture["root"])
    _json_file(fixture["run"], run)
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="not finite and nonnegative"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_changed_observation_order_digest_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registration = json.loads(fixture["registration"].read_text(encoding="utf-8"))
    cell = gate.CELL_ORDER[0]
    registration["observation_bindings"][cell]["record_ids_sha256"] = "d" * 64
    _json_file(fixture["registration"], registration)
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="record-ID order differs"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_changed_observation_payload_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    observation = fixture["observations"][gate.CELL_ORDER[0]]
    observation.write_bytes(observation.read_bytes() + b"changed")
    opened: list[str] = []
    with pytest.raises(gate.GateError, match="registration observation .* hash or size changed"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_external_hashed_readonly_resource_is_allowed_but_unmarked_is_rejected(tmp_path: Path) -> None:
    shared = tmp_path.parent / f"trr0010-shared-{tmp_path.name}"
    shared.write_bytes(b"shared readonly public resource\n")
    binding = {
        "path": str(shared),
        "bytes": shared.stat().st_size,
        "sha256": gate.sha256_file(shared),
        "readonly": True,
    }
    checked = gate._record(binding, root=tmp_path, description="shared resource")
    assert checked["readonly"] is True
    with pytest.raises(gate.GateError, match="external path must be explicitly readonly"):
        gate._record({key: value for key, value in binding.items() if key != "readonly"}, root=tmp_path, description="shared resource")


def test_direct_method_requires_hash_bound_public_embedding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registration = json.loads(fixture["registration"].read_text(encoding="utf-8"))
    row = next(row for row in registration["methods"] if row["id"] == gate.CURRENT_FIXED_METHOD_ID)
    row["resources"].pop("public_embedding_table")
    _json_file(fixture["registration"], registration)
    opened: list[str] = []
    with pytest.raises(gate.GateError, match=r"current_fixed public E and base/decoder state resources are incomplete"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_changed_directional_effective_w_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["method_resources"][gate.CURRENT_DIRECTIONAL_METHOD_ID]["effective_readout_w"].write_bytes(
        b"changed exported effective W\n"
    )
    opened: list[str] = []
    with pytest.raises(gate.GateError, match=r"current_directional resource effective_readout_w hash or size changed"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_changed_direct_public_e_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["method_resources"][gate.CURRENT_FIXED_METHOD_ID]["public_embedding_table"].write_bytes(
        b"changed public E\n"
    )
    opened: list[str] = []
    with pytest.raises(gate.GateError, match=r"unchanged_shared_start resource public_embedding_table hash or size changed"):
        _gate_then_open(fixture, opened)
    assert opened == []


def test_observation_header_geometry_is_checked_without_materializing_h(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cell_id = gate.CELL_ORDER[0]
    observation_path = fixture["observations"][cell_id]
    shape = [gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS, gate.OBSERVATION_HIDDEN_SIZE + 1]
    metadata = {
        "schema": gate.OBSERVATION_SCHEMA,
        "task_id": gate.TASK_ID,
        "cell_id": cell_id,
        "records": str(gate.RECORDS_PER_CELL),
        # Deliberately retain the registered metadata shape: only the tensor
        # header is changed, so the header-level check is exercised directly.
        "shape": json.dumps([gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS, gate.OBSERVATION_HIDDEN_SIZE]),
        "record_ids_sha256": "b" * 64,
        "truth_opened": "false",
        "source_text_written": "false",
        "token_ids_written": "false",
        "target_labels_loaded": "false",
    }
    save_file(
        {
            "activations": torch.zeros(shape, dtype=torch.bfloat16),
            "attention_mask": torch.ones((shape[0], shape[1]), dtype=torch.uint8),
            "position_ids": torch.arange(shape[1], dtype=torch.int64).repeat(shape[0], 1),
        },
        str(observation_path),
        metadata=metadata,
    )
    registration = json.loads(fixture["registration"].read_text(encoding="utf-8"))
    registration["observation_bindings"][cell_id]["observation"] = gate.file_record(observation_path, root=fixture["root"])
    selection_payload = json.loads(fixture["inputs"]["source_selection"].read_text(encoding="utf-8"))
    with pytest.raises(gate.GateError, match=r"observation tensor header geometry changed: .*activations"):
        gate._validate_observation_bindings(
            registration["observation_bindings"],
            selection_payload=selection_payload,
            root=fixture["root"],
            description="synthetic",
        )


def test_observation_metadata_shape_mismatch_fails_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cell_id = gate.CELL_ORDER[0]
    observation_path = fixture["observations"][cell_id]
    metadata = {
        "schema": gate.OBSERVATION_SCHEMA,
        "task_id": gate.TASK_ID,
        "cell_id": cell_id,
        "records": str(gate.RECORDS_PER_CELL),
        "shape": json.dumps([gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS, gate.OBSERVATION_HIDDEN_SIZE + 1]),
        "record_ids_sha256": "b" * 64,
        "truth_opened": "false",
        "source_text_written": "false",
        "token_ids_written": "false",
        "target_labels_loaded": "false",
    }
    shape = [gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS, gate.OBSERVATION_HIDDEN_SIZE]
    save_file(
        {
            "activations": torch.zeros(shape, dtype=torch.bfloat16),
            "attention_mask": torch.ones((shape[0], shape[1]), dtype=torch.uint8),
            "position_ids": torch.arange(shape[1], dtype=torch.int64).repeat(shape[0], 1),
        },
        str(observation_path),
        metadata=metadata,
    )
    registration = json.loads(fixture["registration"].read_text(encoding="utf-8"))
    registration["observation_bindings"][cell_id]["observation"] = gate.file_record(observation_path, root=fixture["root"])
    selection_payload = json.loads(fixture["inputs"]["source_selection"].read_text(encoding="utf-8"))
    with pytest.raises(gate.GateError, match=r"observation metadata changed: .*shape"):
        gate._validate_observation_bindings(
            registration["observation_bindings"],
            selection_payload=selection_payload,
            root=fixture["root"],
            description="synthetic",
        )
