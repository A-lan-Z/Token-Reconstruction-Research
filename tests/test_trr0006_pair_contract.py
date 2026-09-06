"""Focused TRR6 gate, scoring, and decision-rule tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts import trr0006_prediction_contract as contract


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


freeze = _load("trr0006_freeze_pair", SCRIPTS / "trr0006_freeze_pair.py")
score = _load("trr0006_score_pair", SCRIPTS / "trr0006_score_pair.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _actual_runner_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, records: int = 8):
    """Build the runner's real registration/manifest/prediction schemas."""

    root = tmp_path
    task_root = root / "experiments" / "TRR-0006"
    output = task_root / "predictions"
    output.mkdir(parents=True)
    code_bindings: list[dict[str, object]] = []
    for role, relative_path in contract.CODE_BINDING_SPECS:
        code_file = root / relative_path
        code_file.parent.mkdir(parents=True, exist_ok=True)
        code_file.write_text(f"# executable binding: {role}\n", encoding="utf-8")
        row = _record(code_file)
        row.update({"role": role, "path": relative_path})
        code_bindings.append(row)
    embedding_file = root / "normalized-E.bin"
    embedding_file.write_bytes(b"normalized public E fixture")
    monkeypatch.setattr(contract, "NORMALIZED_PUBLIC_E_BYTES", embedding_file.stat().st_size)
    monkeypatch.setattr(contract, "NORMALIZED_PUBLIC_E_SHA256", _sha(embedding_file))
    state_bindings: dict[str, dict[str, object]] = {}
    for index, method in enumerate(contract.METHOD_IDS):
        state = root / f"state-{index}.bin"
        state.write_bytes(f"state-{index}".encode())
        state_bindings[method] = {
            "path": str(state),
            "bytes": state.stat().st_size,
            "sha256": _sha(state),
            "source_commit": "a" * 40,
            "attention_mode": "causal" if index == 0 else "diagonal",
            "attention_score_mode": "cosine_scale4" if index == 0 else "dot_product",
            "selected_step": "1900" if index == 0 else "1600",
        }
    monkeypatch.setattr(contract, "PUBLISHED_STATE_BINDINGS", state_bindings)
    monkeypatch.setattr(freeze, "_git_head", lambda _root: "b" * 40)

    observations: list[dict[str, object]] = []
    for cell_id in contract.CELL_ORDER:
        style, condition = cell_id.split("__", 1)
        obs_path = task_root / f"{cell_id}.safetensors"
        activations = torch.zeros((records, 128, 2048), dtype=torch.bfloat16)
        mask = torch.ones((records, 128), dtype=torch.uint8)
        positions = torch.arange(128, dtype=torch.long).repeat(records, 1)
        save_file(
            {"activations": activations, "attention_mask": mask, "position_ids": positions},
            str(obs_path),
        )
        observations.append(
            {
                "cell_id": cell_id,
                "style": style,
                "condition": condition,
                "record_ids_sha256": hashlib.sha256(f"{style}-ids".encode()).hexdigest(),
                "observation": {
                    **_record(obs_path),
                    "shape": [records, 128, 2048],
                    "stored_sequence_tokens": 128,
                    "scored_post_bos_tokens": 127,
                    "capture_batch_records": 8,
                    "capture_sequence_tokens": 192,
                    "activations_key": "activations",
                    "attention_mask_key": "attention_mask",
                    "position_ids_key": "position_ids",
                    "public_full_forward": True,
                    "producer_only_lora": condition == "public_lora_2601",
                },
            }
        )
    observation_manifest = task_root / "observations.json"
    _write_json(
        observation_manifest,
        {
            "schema": contract.OBSERVATION_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
            "records_per_domain": records,
            "cell_order": list(contract.CELL_ORDER),
            "cells": observations,
        },
    )
    observation_manifest_record = _record(observation_manifest)
    registration = {
        "schema": contract.REGISTRATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_PREDICTION_REGISTRATION",
        "code_commit": "b" * 40,
        "records_per_domain": records,
        "cell_order": list(contract.CELL_ORDER),
        "method_ids": list(contract.METHOD_IDS),
        "geometry": {
            "capture_batch_records": 8,
            "capture_sequence_tokens": 192,
            "stored_sequence_tokens": 128,
            "scored_sequence_tokens": 128,
            "scored_post_bos_tokens": 127,
            "hidden_size": 2048,
            "chunk_records": 8,
        },
        "runtime_assets": {
            "normalized_public_E": {
                **_record(embedding_file),
                "shape": [contract.VOCAB_SIZE, contract.HIDDEN_SIZE],
                "dtype": "torch.float32",
            }
        },
        "methods": {
            method: {
                "base_method_id": contract.BASE_METHOD_IDS[method],
                "decision_rule": contract.METHOD_RULES[method],
                "state": state_bindings[method],
            }
            for method in contract.METHOD_IDS
        },
        "observation_manifest": observation_manifest_record,
        "output_root": str(output.relative_to(root)),
        "timing_contract": {
            "warmup_runs_per_record": 1,
            "measured_runs_per_record": 1,
            "repeat_integrity": "Require warmup and measured predicted IDs to match exactly",
        },
        "resource_guard": {
            "minimum_free_gpu_bytes": contract.MIN_FREE_GPU_BYTES,
            "maximum_reserved_gpu_bytes": contract.MAX_RESERVED_GPU_BYTES,
            "maximum_rss_bytes": contract.MAX_RSS_BYTES,
            "minimum_host_available_bytes": contract.MIN_HOST_AVAILABLE_BYTES,
            "maximum_seconds": 300,
        },
        "numerical_settings": dict(contract.NUMERICAL_SETTINGS),
        "code_bindings": code_bindings,
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }
    registration_path = task_root / "registration.json"
    _write_json(registration_path, registration)
    registration_record = _record(registration_path)

    prediction_rows: dict[str, object] = {}
    timing_rows: dict[str, object] = {}
    for index, (cell_id, method_id) in enumerate(freeze.EXPECTED_KEYS):
        key = f"{cell_id}::{method_id}"
        predictions = torch.ones((records, 128), dtype=torch.long)
        predictions[:, 0] = contract.BOS_TOKEN_ID
        prediction_path = output / cell_id.split("__", 1)[0] / cell_id.split("__", 1)[1] / f"{method_id}.safetensors"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_metadata = {
            "schema": contract.PREDICTION_SCHEMA,
            "task_id": contract.TASK_ID,
            "registration_sha256": registration_record["sha256"],
            "observation_manifest_sha256": observation_manifest_record["sha256"],
            "observation_sha256": observations[contract.CELL_ORDER.index(cell_id)]["observation"]["sha256"],
            "cell_id": cell_id,
            "method_id": method_id,
            "records": str(records),
            "sequence_tokens": "128",
            "capture_sequence_tokens": "192",
            "hidden_size": "2048",
            "candidate_arrays_persisted": "false",
            "truth_opened": "false",
        }
        save_file({"predictions": predictions}, str(prediction_path), metadata=prediction_metadata)
        prediction_rows[key] = {
            "schema": contract.PREDICTION_SCHEMA,
            "task_id": contract.TASK_ID,
            "cell_id": cell_id,
            "method_id": method_id,
            "records": records,
            "shape": [records, 128],
            "prediction_artifact": _record(prediction_path),
            "prediction_sha256": contract.tensor_digest(predictions),
            "observation": observations[contract.CELL_ORDER.index(cell_id)]["observation"],
            "state": state_bindings[method_id],
            "registration_sha256": registration_record["sha256"],
            "truth_opened": False,
            "candidate_arrays_persisted": False,
        }
        timing_rows[key] = {
            "schema": contract.TIMING_SCHEMA,
            "task_id": contract.TASK_ID,
            "cell_id": cell_id,
            "method_id": method_id,
            "records": records,
            "warmup_runs_per_record": 1,
            "measured_runs_per_record": 1,
            "warmup_seconds_sum": 0.01 + index,
            "measured_seconds_sum": 0.02 + index,
            "timed_interval_total_seconds": 0.03 + 2 * index,
            "per_record_measured_seconds": [0.001] * records,
            "warmup_output_exact_match_measured": True,
            "measured_output_selected": True,
            "steady_interval": "public fixture",
            "chunk_records": 8,
            "chunks": records // 8,
            "load_seconds_separate": 0.01,
            "peak_memory": {"process_max_rss_bytes": 1},
            "prediction_artifact": _record(prediction_path),
            "prediction_sha256": contract.tensor_digest(predictions),
            "truth_opened": False,
        }
    predictions_path = output / "predictions.json"
    timings_path = output / "timings.json"
    _write_json(
        predictions_path,
        {
            "schema": "token-reconstruction.trr0006-prediction-descriptor-manifest.v1",
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH",
            "registration_sha256": registration_record["sha256"],
            "records_per_domain": records,
            "cell_order": list(contract.CELL_ORDER),
            "method_ids": list(contract.METHOD_IDS),
            "predictions": prediction_rows,
            "truth_opened": False,
        },
    )
    _write_json(
        timings_path,
        {
            "schema": "token-reconstruction.trr0006-timing-descriptor-manifest.v1",
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_TIMINGS_COMPLETE_NO_TRUTH",
            "registration_sha256": registration_record["sha256"],
            "records_per_domain": records,
            "cell_order": list(contract.CELL_ORDER),
            "method_ids": list(contract.METHOD_IDS),
            "timings": timing_rows,
            "truth_opened": False,
        },
    )
    _write_json(
        output / "run_manifest.json",
        {
            "schema": contract.RUN_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH",
            "registration": registration_record,
            "observation_manifest": observation_manifest_record,
            "code_commit": "b" * 40,
            "code_bindings": code_bindings,
            "runtime_assets": registration["runtime_assets"],
            "geometry": registration["geometry"],
            "records_per_domain": records,
            "cell_order": list(contract.CELL_ORDER),
            "method_ids": list(contract.METHOD_IDS),
            "predictions_count": 8,
            "timings_count": 8,
            "predictions_complete": True,
            "timing_decisions_complete": True,
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "candidate_arrays_persisted": False,
        },
    )
    return {
        "root": root,
        "output": output,
        "registration": registration_path,
        "observation_manifest": observation_manifest,
        "receipt": task_root / "freeze_receipt.json",
        "truth": tmp_path / "sealed" / "truth.safetensors",
    }


def test_actual_runner_schema_freezes_and_rechecks_all_eight_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _actual_runner_fixture(tmp_path, monkeypatch)
    receipt = freeze.freeze_matrix(
        repository_root=fixture["root"],
        registration_path=fixture["registration"],
        receipt_path=fixture["receipt"],
        observation_manifest_path=fixture["observation_manifest"],
    )
    assert receipt["status"] == freeze.RECEIPT_STATUS
    gate = freeze.validate_before_truth(
        repository_root=fixture["root"],
        registration_path=fixture["registration"],
        receipt_path=fixture["receipt"],
        observation_manifest_path=fixture["observation_manifest"],
        truth_path=fixture["truth"],
    )
    assert gate["verified_before_truth"] is True
    assert gate["truth_opened"] is False
    assert gate["entry_count"] == 8
    assert set(gate["prediction_tensors"]) == set(freeze.EXPECTED_KEYS)


def test_malformed_prediction_tensor_fails_before_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _actual_runner_fixture(tmp_path, monkeypatch)
    freeze.freeze_matrix(
        repository_root=fixture["root"],
        registration_path=fixture["registration"],
        receipt_path=fixture["receipt"],
        observation_manifest_path=fixture["observation_manifest"],
    )
    prediction_path = next((fixture["output"]).rglob("*.safetensors"))
    bad = torch.ones((8, 127), dtype=torch.long)
    bad[:, 0] = contract.BOS_TOKEN_ID
    with safe_open(str(prediction_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    save_file({"predictions": bad}, str(prediction_path), metadata=metadata)
    # Rebind the public descriptors to the changed file so the gate reaches
    # the safetensor geometry validator rather than stopping at the file hash.
    prediction_manifest = fixture["output"] / "predictions.json"
    prediction_value = json.loads(prediction_manifest.read_text(encoding="utf-8"))
    timing_manifest = fixture["output"] / "timings.json"
    timing_value = json.loads(timing_manifest.read_text(encoding="utf-8"))
    changed_key = None
    for key, row in prediction_value["predictions"].items():
        if Path(row["prediction_artifact"]["path"]).resolve() == prediction_path.resolve():
            changed_key = key
            row["prediction_artifact"] = _record(prediction_path)
            row["prediction_sha256"] = contract.tensor_digest(bad)
            timing_value["timings"][key]["prediction_artifact"] = _record(prediction_path)
            timing_value["timings"][key]["prediction_sha256"] = contract.tensor_digest(bad)
            break
    assert changed_key is not None
    _write_json(prediction_manifest, prediction_value)
    _write_json(timing_manifest, timing_value)
    called: list[str] = []

    def truth_loader():
        called.append("truth")
        return {}

    with pytest.raises(score.PairScoreError, match="public gate"):
        score.score_with_truth_loader(
            public_gate=lambda: freeze.validate_before_truth(
                repository_root=fixture["root"],
                registration_path=fixture["registration"],
                receipt_path=fixture["receipt"],
                observation_manifest_path=fixture["observation_manifest"],
                truth_path=fixture["truth"],
            ),
            truth_loader=truth_loader,
            score_after_truth=lambda _gate, truth: truth,
        )
    assert called == []


def test_incomplete_prediction_manifest_fails_before_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _actual_runner_fixture(tmp_path, monkeypatch)
    predictions_path = fixture["output"] / "predictions.json"
    value = json.loads(predictions_path.read_text(encoding="utf-8"))
    value["predictions"].pop(next(iter(value["predictions"])))
    predictions_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(freeze.FreezePairError, match="incomplete"):
        freeze.freeze_matrix(
            repository_root=fixture["root"],
            registration_path=fixture["registration"],
            receipt_path=fixture["receipt"],
            observation_manifest_path=fixture["observation_manifest"],
        )
    assert not fixture["receipt"].exists()


def _post_bos_fixture(records: int = 4):
    labels = np.ones((records, 128), dtype=np.int64)
    labels[:, 0] = score.BOS_TOKEN_ID
    labels[:, 1:] = np.arange(1, 128, dtype=np.int64)
    causal = labels.copy()
    diagonal = labels.copy()
    diagonal[1, 1] = 1000
    causal[2, 1] = 1001
    diagonal[3, 1] = 1002
    causal[3, 2] = 1003
    mask = np.ones_like(labels, dtype=bool)
    ids = tuple(f"source-{i}" for i in range(records))
    return labels, causal, diagonal, mask, ids


def test_score_reports_four_exact_categories_and_absolute_rates() -> None:
    labels, causal, diagonal, mask, ids = _post_bos_fixture()
    left = score.score_cell(
        predictions=causal,
        truth=labels,
        attention_mask=mask,
        record_ids=ids,
        method_id=score.METHOD_ORDER[0],
        position_ids=np.broadcast_to(np.arange(128), mask.shape),
    )
    right = score.score_cell(
        predictions=diagonal,
        truth=labels,
        attention_mask=mask,
        record_ids=ids,
        method_id=score.METHOD_ORDER[1],
    )
    comparison = score.paired_comparison(left["per_record"], right["per_record"], cell_id="finance__public_base", draws=80, seed=5005)
    categories = comparison["exact_categories"]
    assert categories["both_correct"] == 1
    assert categories["causal_only"] == 1
    assert categories["positionwise_only"] == 1
    assert categories["neither_correct"] == 1
    assert sum(categories[key] for key in ("both_correct", "causal_only", "positionwise_only", "neither_correct")) == 4
    assert comparison["absolute_exact_recovery"]["causal_rate_pp"] == 50.0
    assert comparison["absolute_exact_recovery"]["diagonal_rate_pp"] == 50.0


def test_score_matrix_extracts_complete_four_cell_report() -> None:
    labels, causal, diagonal, mask, ids = _post_bos_fixture()
    predictions: dict[tuple[str, str], np.ndarray] = {}
    truth: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    record_ids: dict[str, tuple[str, ...]] = {}
    positions: dict[str, np.ndarray] = {}
    for cell_id in score.CELL_ORDER:
        predictions[(cell_id, score.METHOD_ORDER[0])] = causal
        predictions[(cell_id, score.METHOD_ORDER[1])] = diagonal
        truth[cell_id] = labels
        masks[cell_id] = mask
        record_ids[cell_id] = ids
        positions[cell_id] = np.broadcast_to(np.arange(128), mask.shape)
    result = score.score_matrix(
        predictions=predictions,
        truth=truth,
        attention_masks=masks,
        record_ids=record_ids,
        position_ids=positions,
        bootstrap_draws=24,
        bootstrap_seed=5005,
    )
    report = score.extract_report(result)
    assert len(report["rows"]) == 4
    assert set(report["rows"][0]) >= {
        "both_correct", "causal_only", "positionwise_only", "neither_correct",
        "causal_absolute_exact_rate_pp", "positionwise_absolute_exact_rate_pp",
        "token_delta_pp", "token_lower_practical_bound_pp",
        "exact_delta_pp", "exact_lower_practical_bound_pp",
    }
    assert report["directional_family"]["endpoint_count"] == 16
    rendered = score.render_report(result)
    assert "Token practical" in rendered
    assert "Exact practical" in rendered


def test_registered_exact_bound_reproduces_1024_finance_checkpoint() -> None:
    bounds = score.exact_net_bounds(records=1024, gain=40, loss=32)
    assert bounds["gain_upper_pp"] == pytest.approx(6.0376778, abs=1e-6)
    assert bounds["loss_lower_pp"] == pytest.approx(1.7519513, abs=1e-6)
    assert bounds["upper_practical_bound_pp"] == pytest.approx(4.2857265, abs=1e-6)
    assert bounds["cp_marginal_coverage"].startswith("Exact")


def _bound_comparison(token_lower: float, token_upper: float, exact_lower: float, exact_upper: float):
    return {
        "token": {"delta_lower_practical_bound_pp": token_lower, "delta_upper_practical_bound_pp": token_upper},
        "exact": {"lower_practical_bound_pp": exact_lower, "upper_practical_bound_pp": exact_upper},
    }


def test_support_is_scoped_to_one_endpoint_and_requires_harm_exclusion() -> None:
    comparisons = {
        cell: _bound_comparison(0.6 if cell == "finance__public_base" else 0.0, 1.0, 0.0, 4.0)
        for cell in score.CELL_ORDER
    }
    decision = score.classify_matrix(comparisons)
    assert decision["decision"] == "context_gain_supported"
    assert decision["quality_support"] is True
    assert decision["quality_exclusion"] is False
    assert decision["supporting_endpoints"][0]["cell_id"] == "finance__public_base"


def test_exclusion_requires_both_outcomes_in_all_cells_and_allows_causal_harm() -> None:
    comparisons = {cell: _bound_comparison(-20.0, 0.4, -20.0, 4.0) for cell in score.CELL_ORDER}
    decision = score.classify_matrix(comparisons)
    assert decision["decision"] == "positionwise_default"
    assert decision["quality_exclusion"] is True
    assert decision["harm_status"] == "harm_not_excluded"


def test_ambiguous_bound_is_inconclusive() -> None:
    comparisons = {cell: _bound_comparison(0.0, 0.6, 0.0, 5.1) for cell in score.CELL_ORDER}
    decision = score.classify_matrix(comparisons)
    assert decision["decision"] == "inconclusive"
    assert decision["harm_status"] == "harm_excluded"


def test_truth_loader_is_never_called_when_public_gate_fails() -> None:
    called: list[str] = []

    def failing_gate():
        raise RuntimeError("missing timing")

    def truth_loader():
        called.append("truth")
        return {}

    with pytest.raises(score.PairScoreError, match="public gate"):
        score.score_with_truth_loader(
            public_gate=failing_gate,
            truth_loader=truth_loader,
            score_after_truth=lambda _gate, truth: truth,
        )
    assert called == []


def test_truth_loader_runs_only_after_verified_gate() -> None:
    order: list[str] = []

    def gate():
        order.append("gate")
        return {"verified_before_truth": True, "truth_opened": False}

    def truth_loader():
        order.append("truth")
        return {"cell": "sealed"}

    result = score.score_with_truth_loader(
        public_gate=gate,
        truth_loader=truth_loader,
        score_after_truth=lambda _gate, truth: {"truth": truth},
    )
    assert order == ["gate", "truth"]
    assert result["truth"]["cell"] == "sealed"
