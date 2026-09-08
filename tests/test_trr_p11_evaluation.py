"""Synthetic-only tests for the TRR-P11 evaluation adapter."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.trr_p11 import evaluation as ev


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_json(path: Path, payload: dict) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return _record(path)


def _save(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    return _record(path)


def _prediction(rows: int, value: int) -> torch.Tensor:
    result = torch.full((rows, ev.STORED_SEQUENCE_TOKENS), value, dtype=torch.int64)
    result[:, 0] = ev.BOS_TOKEN_ID
    return result


def _truth_tensor(rows: int = ev.RECORDS_PER_DOMAIN) -> torch.Tensor:
    result = _prediction(rows, 1)
    result[:, 1] = torch.arange(rows, dtype=torch.int64) + 1
    return result


def _fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path
    cells = list(ev.CELL_ORDER)
    selection_record_ids = {
        domain: [f"{domain}-record-{i}" for i in range(ev.RECORDS_PER_DOMAIN)]
        for domain in ev.DOMAIN_ORDER
    }
    selection_h128 = {
        domain: [ev._h128_sequence_digest(row.tolist()) for row in _truth_tensor()]
        for domain in ev.DOMAIN_ORDER
    }
    selection_ids = {domain: ev._json_digest(selection_record_ids[domain]) for domain in ev.DOMAIN_ORDER}
    selection_path = root / "selection.json"
    _write_json(
        selection_path,
        {
            "schema": "token-reconstruction.trr-p11-source-selection.v1",
            "task_id": ev.TASK_ID,
            "status": "FROZEN_TRR-P11_SOURCE_SELECTION_NO_TRUTH",
            "records_by_domain": {domain: ev.RECORDS_PER_DOMAIN for domain in ev.DOMAIN_ORDER},
            "selection_rule": {
                "record_ids_sha256": selection_ids,
                "h128_sequence_sha256": {domain: ev._json_digest(selection_h128[domain]) for domain in ev.DOMAIN_ORDER},
                "records": {
                    domain: [
                        {"record_id": selection_record_ids[domain][index], "final_sequence_sha256": selection_h128[domain][index], "h128_sequence_sha256": selection_h128[domain][index]}
                        for index in range(ev.RECORDS_PER_DOMAIN)
                    ]
                    for domain in ev.DOMAIN_ORDER
                },
            },
            "truth_opened": False,
            "p03_holdout_accessed": False,
        },
    )
    observations: dict[str, dict[str, object]] = {}
    capture_cells = []
    for cell in cells:
        domain = cell.split("__", 1)[0]
        observation_path = root / "observations" / f"{cell}.safetensors"
        observation_path.parent.mkdir(parents=True, exist_ok=True)
        activations = torch.arange(256 * 2 * 3, dtype=torch.float32).reshape(256, 2, 3)
        masks = torch.ones((256, 2), dtype=torch.uint8)
        positions = torch.arange(2, dtype=torch.int64).expand(256, -1).contiguous()
        observation = _save(observation_path, {"activations": activations, "attention_mask": masks, "position_ids": positions})
        raw_sha = {
            "activations": ev.p10.tensor_digest(activations),
            "attention_mask": ev.p10.tensor_digest(masks),
            "position_ids": ev.p10.tensor_digest(positions),
        }
        normalized_sha = {
            "activations": raw_sha["activations"],
            "attention_mask": ev.p10.tensor_digest(masks.to(torch.bool)),
            "position_ids": raw_sha["position_ids"],
        }
        capture_order = selection_record_ids[domain]
        record_order = [f"record/{index:03d}" for index in range(ev.RECORDS_PER_DOMAIN)]
        observation.update({
            "capture_tensor_sha256": raw_sha,
            "normalized_tensor_sha256": normalized_sha,
            "tensor_sha256": normalized_sha,
            "capture_record_order": capture_order,
            "capture_record_order_sha256": ev._json_digest(capture_order),
            "record_order": record_order,
            "record_order_sha256": ev._json_digest(record_order),
        })
        observations[cell] = observation
        capture_cells.append({"cell_id": cell, "records": ev.RECORDS_PER_DOMAIN, "record_ids_sha256": selection_ids[domain], "observation": observation})
    capture_path = root / "capture.json"
    _write_json(
        capture_path,
        {
            "schema": "token-reconstruction.trr-p11-public-capture.v1",
            "task_id": ev.TASK_ID,
            "status": "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH",
            "cells": capture_cells,
            "truth_opened": False,
            "source_text_written": False,
            "token_ids_written": False,
            "p03_holdout_accessed": False,
        },
    )
    state_paths: dict[str, Path] = {}
    states: dict[str, dict[str, object]] = {}
    for method, label in (("new_current_fixed_B0", "B0"), ("new_expanded_fixed_B1", "B1"), (ev.COMPARATOR_METHOD, "A1")):
        state_path = root / "states" / f"{label}.safetensors"
        _save(state_path, {"state": torch.tensor([len(label)], dtype=torch.float32)})
        state_paths[method] = state_path
        states[method] = {
            "state_id": f"synthetic/{label}",
            "model_id": f"synthetic-model/{label}",
            "selected_step": 1,
            "file": _record(state_path),
        }
    deployment_paths: dict[str, Path] = {}
    for name in ("public_readout", "loader_code", "decoder_code", "package_cli", "package_manifest", "frozen_config", "selection_receipt", "smoke_input", "smoke_expected"):
        path = root / "restore-assets" / (name + ".bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic restore asset " + name).encode())
        deployment_paths[name] = path
    restore_assets: dict[str, object] = {}
    for name, path in {"current_fixed": state_paths["new_current_fixed_B0"], "expanded_fixed": state_paths["new_expanded_fixed_B1"], **deployment_paths}.items():
        rec = _record(path)
        metadata = {}
        if name == "current_fixed":
            metadata = {"bank": "B0", "model_id": states["new_current_fixed_B0"]["model_id"], "selected_step": 1}
        elif name == "expanded_fixed":
            metadata = {"bank": "B1", "model_id": states["new_expanded_fixed_B1"]["model_id"], "selected_step": 1}
        restore_assets[name] = {"relative_path": str(path.relative_to(root)), "copies": {"primary": rec, "secondary": rec}, "metadata": metadata}
    loaded_assets = []
    for name, path in {"current_fixed": state_paths["new_current_fixed_B0"], "expanded_fixed": state_paths["new_expanded_fixed_B1"], "public_readout": deployment_paths["public_readout"], "loader_code": deployment_paths["loader_code"], "decoder_code": deployment_paths["decoder_code"], "package_cli": deployment_paths["package_cli"]}.items():
        rec = _record(path)
        loaded_assets.append({"path": str(path), "bytes": rec["bytes"], "sha256": rec["sha256"], "role": name})
    restore_manifest_path = root / "restore-source-manifest.json"
    restore_manifest_path.write_text("synthetic restore descriptor\n", encoding="utf-8")
    restore_manifest_record = _record(restore_manifest_path)
    restore_path = root / "restore.json"
    _write_json(
        restore_path,
        {
            "schema": "token-reconstruction.trr-p11-restore-gate.v1",
            "task_id": ev.TASK_ID,
            "status": ev.RESTORE_STATUS,
            "independent_evaluation_truth_opened": False,
            "training_worktree_import": False,
            "source_boundary": "secondary",
            "retrieval": {"source_boundary": "secondary"},
            "restore_manifest": restore_manifest_record,
            "p03_holdout_accessed": False,
            "assets": restore_assets,
            "tensor_identity": {name: {"file_sha256": restore_assets[name]["copies"]["primary"]["sha256"]} for name in ev.RESTORE_TENSOR_ASSETS},
            "consumer_receipt": {
                "loaded_file_bindings": loaded_assets,
                "loaded_code_paths": [str(deployment_paths[name]) for name in ("loader_code", "decoder_code", "package_cli")],
                "training_worktree_import": False,
                "temporary_dependency": False,
            },
        },
    )
    outputs: dict[str, dict[str, object]] = {}
    methods: dict[str, object] = {}
    for method, tensor_key, value in (
        ("new_current_fixed_B0", "current_fixed", 1),
        ("new_expanded_fixed_B1", "expanded_fixed", 2),
    ):
        cells_payload: dict[str, object] = {}
        for cell in cells:
            output_path = root / "predictions" / f"{cell}.safetensors"
            tensors = {"current_fixed": _prediction(ev.RECORDS_PER_DOMAIN, 1), "expanded_fixed": _prediction(ev.RECORDS_PER_DOMAIN, 2)}
            output_record = _save(output_path, tensors)
            receipt_path = root / "predictions" / f"{cell}.receipt.json"
            receipt = _write_json(
                receipt_path,
                {
                    "schema": ev.PREDICTION_RECEIPT_SCHEMA,
                    "task_id": "TRR-0012",
                    "status": "PREDICTIONS_GENERATED_AFTER_SELECTION",
                    "complete_before_smoke": True,
                    "smoke_used_for_selection": False,
                    "independent_evaluation_truth_opened": False,
                    "truth_opened": False,
                    "source_text_loaded": False,
                    "token_ids_loaded": False,
                    "output": output_record,
                    "observations": observations[cell],
                    "readout": {
                        **restore_assets["public_readout"]["copies"]["primary"],
                        "tensor_key": "embeddings",
                    },
                    "package_descriptor_sha256": restore_assets["package_manifest"]["copies"]["primary"]["sha256"],
                    "methods": {
                        "current_fixed": {"state_file_binding": states["new_current_fixed_B0"]["file"]},
                        "expanded_fixed": {"state_file_binding": states["new_expanded_fixed_B1"]["file"]},
                    },
                    "runtime": {"dependencies": {"torch": "synthetic"}},
                },
            )
            cells_payload[cell] = {
                "file": output_record,
                "tensor_key": tensor_key,
                "tensor_sha256": ev.p10.tensor_digest(tensors[tensor_key]),
                "records": ev.RECORDS_PER_DOMAIN,
                "receipt": receipt,
            }
        methods[method] = {"state": states[method], "cells": cells_payload}
    comparator_cells: dict[str, object] = {}
    for cell in cells:
        output_path = root / "predictions" / f"{cell}-a1.safetensors"
        tensor = _prediction(ev.COMPARATOR_RECORDS_PER_DOMAIN, 3)
        output_record = _save(output_path, {"predictions": tensor})
        receipt_path = root / "predictions" / f"{cell}-a1.receipt.json"
        receipt = _write_json(
            receipt_path,
            {
                "schema": "synthetic-a1-receipt.v1",
                "task_id": "TRR-A1",
                "status": "NATIVE_PREDICTIONS_FROZEN_NO_TRUTH",
                "truth_opened": False,
                "source_text_loaded": False,
                "token_ids_loaded": False,
                "output": output_record,
                "observations": observations[cell],
                "method": {"state_file_binding": states[ev.COMPARATOR_METHOD]["file"]},
            },
        )
        comparator_cells[cell] = {
            "file": output_record,
            "tensor_key": "predictions",
            "tensor_sha256": ev.p10.tensor_digest(tensor),
            "records": ev.COMPARATOR_RECORDS_PER_DOMAIN,
            "subset": {"parent_cell": cell, "indices": list(range(ev.COMPARATOR_RECORDS_PER_DOMAIN)), "indices_sha256": ev.p10.indices_digest(list(range(ev.COMPARATOR_RECORDS_PER_DOMAIN)),), "records": ev.COMPARATOR_RECORDS_PER_DOMAIN},
            "receipt_method_key": "predictions",
            "receipt": receipt,
        }
    trace_path = root / "traces" / "run.json"
    trace = _write_json(trace_path, {"schema": "synthetic-trace", "truth_opened": False, "p03_holdout_accessed": False, "candidate_arrays_persisted": True})
    cost_path = root / "costs" / "run.json"
    cost = _write_json(cost_path, {"schema": "synthetic-cost", "truth_opened": False, "p03_holdout_accessed": False, "cost_scope": "synthetic"})
    source_order = {
        cell: {"record_ids_sha256": selection_ids[cell.split("__", 1)[0]], "record_order_sha256": observations[cell]["record_order_sha256"]}
        for cell in cells
    }
    manifest = {
        "schema": ev.MANIFEST_SCHEMA,
        "task_id": ev.TASK_ID,
        "status": "PREDICTIONS_READY_NO_TRUTH",
        "selection": _record(selection_path),
        "capture": _record(capture_path),
        "restore": _record(restore_path),
        "observations": observations,
        "source_order": source_order,
        "methods": methods,
        "comparator": {"status": "READY", "state": states[ev.COMPARATOR_METHOD], "cells": comparator_cells},
        "trace_files": [trace],
        "cost_files": [cost],
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "p03_holdout_accessed": False,
    }
    manifest_path = root / "evaluation.json"
    _write_json(manifest_path, manifest)
    return {"root": root, "manifest": manifest_path, "manifest_payload": manifest, "states": states, "observations": observations}


def test_freeze_accepts_agent1_output_keys_and_full_matrix(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = ev.freeze_predictions(manifest_path=fixture["manifest"], output_path=tmp_path / "freeze.json", repository_root=tmp_path)
    assert result["status"] == "P11_PREDICTIONS_FROZEN_BEFORE_TRUTH"
    payload = json.loads((tmp_path / "freeze.json").read_text(encoding="utf-8"))
    assert payload["methods"]["new_current_fixed_B0"]["cells"][ev.CELL_ORDER[0]]["tensor_key"] == "current_fixed"
    assert payload["methods"]["new_expanded_fixed_B1"]["cells"][ev.CELL_ORDER[0]]["tensor_key"] == "expanded_fixed"
    assert payload["comparator"]["status"] == "READY"


def test_freeze_rejects_agent1_singleton_key_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = copy.deepcopy(fixture["manifest_payload"])
    manifest["methods"]["new_current_fixed_B0"]["cells"][ev.CELL_ORDER[0]]["tensor_key"] = "predictions"
    bad = tmp_path / "bad.json"
    _write_json(bad, manifest)
    with pytest.raises(ev.EvaluationError, match="output key changed"):
        ev.freeze_predictions(manifest_path=bad, output_path=tmp_path / "freeze.json", repository_root=tmp_path)


def test_primary_prediction_rejects_extra_tensor_keys(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cell = ev.CELL_ORDER[0]
    output_binding = fixture["manifest_payload"]["methods"]["new_current_fixed_B0"]["cells"][cell]["file"]
    output_path = Path(output_binding["path"])
    tensors = {
        "current_fixed": _prediction(ev.RECORDS_PER_DOMAIN, 1),
        "expanded_fixed": _prediction(ev.RECORDS_PER_DOMAIN, 2),
        "unexpected": _prediction(ev.RECORDS_PER_DOMAIN, 3),
    }
    _save(output_path, tensors)
    fixture["manifest_payload"]["methods"]["new_current_fixed_B0"]["cells"][cell]["file"] = _record(output_path)
    bad = tmp_path / "bad-output-keys.json"
    _write_json(bad, fixture["manifest_payload"])
    with pytest.raises(ev.EvaluationError, match="primary prediction tensor keys changed"):
        ev.freeze_predictions(manifest_path=bad, output_path=tmp_path / "freeze.json", repository_root=tmp_path)


def test_freeze_requires_explicit_comparator_or_registered_blocker(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = copy.deepcopy(fixture["manifest_payload"])
    manifest["comparator"] = {"status": "MISSING"}
    bad = tmp_path / "missing.json"
    _write_json(bad, manifest)
    with pytest.raises(ev.EvaluationError, match="neither complete nor explicitly blocked"):
        ev.freeze_predictions(manifest_path=bad, output_path=tmp_path / "missing-freeze.json", repository_root=tmp_path)

    manifest["comparator"] = {"status": "BLOCKED_TECHNICAL", "preregistered": True, "blocker_id": "A1-ENV-001", "reason": "synthetic blocker"}
    partial = tmp_path / "partial.json"
    _write_json(partial, manifest)
    result = ev.freeze_predictions(manifest_path=partial, output_path=tmp_path / "partial-freeze.json", repository_root=tmp_path)
    assert result["status"] == "QUALIFIED_PARTIAL_A1_BLOCKER"
    scorer_source = Path(ev.__file__).resolve().parents[2] / ev.SCORER_RELATIVE_PATH
    scorer_target = tmp_path / ev.SCORER_RELATIVE_PATH
    scorer_target.parent.mkdir(parents=True, exist_ok=True)
    scorer_target.write_bytes(scorer_source.read_bytes())
    freeze_record = _record(tmp_path / "partial-freeze.json")
    truth_path = tmp_path / "partial-truth.safetensors"
    _save(truth_path, {f"{cell}__token_ids": _truth_tensor() for cell in ev.CELL_ORDER}, metadata={"freeze_sha256": freeze_record["sha256"], "truth_opened": "true"})
    descriptor_path = tmp_path / "partial-truth.json"
    _write_json(descriptor_path, {"schema": ev.TRUTH_SCHEMA, "task_id": ev.TASK_ID, "status": "TRUTH_PREPARED_AFTER_PREDICTION_FREEZE", "freeze": freeze_record, "sidecar": _record(truth_path), "predictions_frozen_before_truth": True, "p03_holdout_accessed": False})
    truth, truth_info = ev.load_truth_after_freeze(freeze_path=tmp_path / "partial-freeze.json", truth_descriptor_path=descriptor_path, repository_root=tmp_path)
    assert truth_info["a1_comparator_available"] is False
    score = ev.score_after_truth(freeze_path=tmp_path / "partial-freeze.json", truth_descriptor_path=descriptor_path, output_path=tmp_path / "partial-score.json", repository_root=tmp_path)
    assert score["status"] == "SCORE_COMPLETE_AFTER_TRUTH_QUALIFIED_PARTIAL_A1_BLOCKER"
    score_payload = json.loads((tmp_path / "partial-score.json").read_text(encoding="utf-8"))
    assert score_payload["a1_diagnostics"]["status"] == "UNAVAILABLE"
    assert all("new_B1_minus_a1_a2_first128" not in item for item in score_payload["cells"].values())


def test_frozen_revalidation_rejects_mutated_prediction_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    ev.freeze_predictions(manifest_path=fixture["manifest"], output_path=freeze_path, repository_root=tmp_path)
    output = Path(fixture["manifest_payload"]["methods"]["new_current_fixed_B0"]["cells"][ev.CELL_ORDER[0]]["file"]["path"])
    output.write_bytes(b"mutated-after-freeze")
    with pytest.raises(ev.EvaluationError):
        ev._load_frozen(freeze_path, root=tmp_path)


def test_score_uses_p10_inventory_and_registered_scorer_on_synthetic_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    scorer_source = Path(ev.__file__).resolve().parents[2] / ev.SCORER_RELATIVE_PATH
    scorer_target = tmp_path / ev.SCORER_RELATIVE_PATH
    scorer_target.parent.mkdir(parents=True, exist_ok=True)
    scorer_target.write_bytes(scorer_source.read_bytes())
    freeze_path = tmp_path / "freeze.json"
    ev.freeze_predictions(manifest_path=fixture["manifest"], output_path=freeze_path, repository_root=tmp_path)
    freeze_record = _record(freeze_path)
    truth_path = tmp_path / "truth.safetensors"
    truth_tensors = {f"{cell}__token_ids": _truth_tensor() for cell in ev.CELL_ORDER}
    _save(truth_path, truth_tensors, metadata={"freeze_sha256": freeze_record["sha256"], "truth_opened": "true"})
    descriptor_path = tmp_path / "truth.json"
    _write_json(
        descriptor_path,
        {
            "schema": ev.TRUTH_SCHEMA,
            "task_id": ev.TASK_ID,
            "status": "TRUTH_PREPARED_AFTER_PREDICTION_FREEZE",
            "freeze": freeze_record,
            "sidecar": _record(truth_path),
            "predictions_frozen_before_truth": True,
            "p03_holdout_accessed": False,
        },
    )
    result = ev.score_after_truth(freeze_path=freeze_path, truth_descriptor_path=descriptor_path, output_path=tmp_path / "score.json", repository_root=tmp_path)
    assert result["status"] == "SCORE_COMPLETE_AFTER_TRUTH"
    payload = json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))
    assert set(payload["cells"]) == set(ev.CELL_ORDER)
    assert payload["scorer"]["bootstrap_seed"] == 9009
    assert payload["p10_inventory"]["status"] == "SCORED_AFTER_PUBLIC_FREEZE"
    assert payload["a1_diagnostics"]["status"] == "AVAILABLE"
    assert all("new_B1_minus_a1_a2_first128" in item for item in payload["cells"].values())
    assert all(len(item["position_strata"]) == ev.SCORED_POST_BOS_TOKENS for item in payload["cells"].values())



def test_restore_state_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = copy.deepcopy(fixture["manifest_payload"])
    restore_path = Path(manifest["restore"]["path"])
    restore_payload = json.loads(restore_path.read_text(encoding="utf-8"))
    restore_payload["assets"]["current_fixed"]["metadata"]["model_id"] = "synthetic-model/wrong"
    restore_path.write_text(json.dumps(restore_payload, sort_keys=True), encoding="utf-8")
    manifest["restore"] = _record(restore_path)
    bad = tmp_path / "bad-restore.json"
    _write_json(bad, manifest)
    with pytest.raises(ev.EvaluationError, match="restore state model identity differs"):
        ev.freeze_predictions(manifest_path=bad, output_path=tmp_path / "freeze.json", repository_root=tmp_path)


def test_manifest_rejects_source_or_target_labels_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = copy.deepcopy(fixture["manifest_payload"])
    manifest["source_text_or_target_labels"] = True
    bad = tmp_path / "bad-source-boundary.json"
    _write_json(bad, manifest)
    with pytest.raises(ev.EvaluationError, match="source text or target labels"):
        ev.freeze_predictions(manifest_path=bad, output_path=tmp_path / "freeze.json", repository_root=tmp_path)


def test_restore_receipt_requires_explicit_clean_runtime_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = copy.deepcopy(fixture["manifest_payload"])
    restore_path = Path(manifest["restore"]["path"])
    restore_payload = json.loads(restore_path.read_text(encoding="utf-8"))
    del restore_payload["consumer_receipt"]["temporary_dependency"]
    restore_path.write_text(json.dumps(restore_payload, sort_keys=True), encoding="utf-8")
    manifest["restore"] = _record(restore_path)
    bad = tmp_path / "bad-boundary.json"
    _write_json(bad, manifest)
    with pytest.raises(ev.EvaluationError, match="clean-runtime boundary"):
        ev.freeze_predictions(manifest_path=bad, output_path=tmp_path / "freeze.json", repository_root=tmp_path)


def test_prediction_receipt_requires_explicit_observation_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = copy.deepcopy(fixture["manifest_payload"])
    cell = ev.CELL_ORDER[0]
    receipt_binding = manifest["methods"]["new_current_fixed_B0"]["cells"][cell]["receipt"]
    receipt_path = Path(receipt_binding["path"])
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    del receipt_payload["observations"]
    receipt_path.write_text(json.dumps(receipt_payload, sort_keys=True), encoding="utf-8")
    manifest["methods"]["new_current_fixed_B0"]["cells"][cell]["receipt"] = _record(receipt_path)
    bad = tmp_path / "bad-observation-receipt.json"
    _write_json(bad, manifest)
    with pytest.raises(ev.EvaluationError, match="observation binding is absent"):
        ev.freeze_predictions(manifest_path=bad, output_path=tmp_path / "freeze.json", repository_root=tmp_path)



def test_recovered_public_support_binding_reports_frozen_strata(tmp_path: Path) -> None:
    repository_root = Path(ev.__file__).resolve().parents[2]
    support_path = repository_root / "experiments/TRR-P11/evaluation/support-binding-r1.json"
    support = json.loads(support_path.read_text(encoding="utf-8"))
    for key in ("histogram", "common_frequency", "contract", "execution", "frequency_recovery_execution", "frequency_recovery_receipt"):
        if isinstance(support.get(key), dict):
            support[key]["path"] = str(repository_root / support[key]["path"])
    support = ev._validate_support(support, root=repository_root)
    assert support["status"] == "AVAILABLE_PUBLIC_FIT_SUPPORT"
    assert {name: support["fit_bank_counts"][name]["records"] for name in ("B0", "B1", "B1_additions")} == {"B0": 1200, "B1": 12000, "B1_additions": 10800}

    observation_path = tmp_path / "observation.safetensors"
    save_file(
        {
            "activations": torch.zeros((ev.RECORDS_PER_DOMAIN, ev.STORED_SEQUENCE_TOKENS, 2), dtype=torch.float32),
            "attention_mask": torch.ones((ev.RECORDS_PER_DOMAIN, ev.STORED_SEQUENCE_TOKENS), dtype=torch.uint8),
            "position_ids": torch.arange(ev.STORED_SEQUENCE_TOKENS, dtype=torch.int64).expand(ev.RECORDS_PER_DOMAIN, -1).contiguous(),
        },
        str(observation_path),
    )
    observation = _record(observation_path)
    truth = {cell: _truth_tensor() for cell in ev.CELL_ORDER}
    predictions: dict[str, torch.Tensor] = {}
    for cell in ev.CELL_ORDER:
        predictions[f"new_current_fixed_B0::{cell}"] = truth[cell].clone()
        predictions[f"new_expanded_fixed_B1::{cell}"] = truth[cell].clone()
        predictions[f"{ev.COMPARATOR_METHOD}::{cell}"] = truth[cell][:ev.COMPARATOR_RECORDS_PER_DOMAIN].clone()
    frozen = ev.FrozenEvaluation(
        freeze_path=tmp_path / "freeze.json",
        freeze_record={},
        payload={"observations": {cell: observation for cell in ev.CELL_ORDER}},
        predictions=predictions,
        prediction_bindings={},
        states={},
        records_by_cell={cell: ev.RECORDS_PER_DOMAIN for cell in ev.CELL_ORDER},
        subsets={},
    )
    report = ev._build_support_strata(frozen, truth, support=support, root=tmp_path)
    assert report["status"] == "AVAILABLE_PUBLIC_FIT_SUPPORT"
    for cell in ev.CELL_ORDER:
        cell_report = report["cells"][cell]
        assert cell_report["denominator_tokens"] == ev.RECORDS_PER_DOMAIN * ev.SCORED_POST_BOS_TOKENS
        assert cell_report["denominator_by_position"]["128-191"] == 0
        assert cell_report["methods"]["new_current_fixed_B0"]["total_tokens"] == cell_report["denominator_tokens"]
        assert cell_report["methods"][ev.COMPARATOR_METHOD]["records"] == ev.COMPARATOR_RECORDS_PER_DOMAIN


def test_actual_a1_trace_receipt_binds_tensor_identity_and_reaches_p10_inventory(tmp_path: Path) -> None:
    """Exercise the native A1 receipt/trace shape and P10 diagnostic handoff."""
    fixture = _fixture(tmp_path)
    cell = ev.CELL_ORDER[0]
    selection_sha256 = fixture["manifest_payload"]["selection"]["sha256"]
    candidates = torch.zeros((ev.COMPARATOR_RECORDS_PER_DOMAIN, ev.STORED_SEQUENCE_TOKENS, 512), dtype=torch.int64)
    candidate_scores = torch.zeros_like(candidates, dtype=torch.float32)
    trace_path = tmp_path / "a1" / "traces" / f"{cell}.safetensors"
    trace_record = _save(
        trace_path,
        {"candidates": candidates, "candidate_scores": candidate_scores},
        metadata={
            "schema": ev.A1_TRACE_SCHEMA,
            "task_id": ev.TASK_ID,
            "method_id": ev.COMPARATOR_METHOD,
            "cell_id": cell,
            "records": str(ev.COMPARATOR_RECORDS_PER_DOMAIN),
            "stored_sequence_tokens": str(ev.STORED_SEQUENCE_TOKENS),
            "proposal_budget": str(ev.A1_TRACE_PROPOSAL_BUDGET),
            "candidate_budget": str(ev.A1_TRACE_CANDIDATE_BUDGET),
            "selection_sha256": str(selection_sha256),
            "truth_opened": "false",
        },
    )
    cost_record = _write_json(
        tmp_path / "a1" / "costs" / f"{cell}.json",
        {"schema": "token-reconstruction.trr-p11-a1-a2-cost.v1", "task_id": ev.TASK_ID, "truth_opened": False, "p03_holdout_accessed": False},
    )
    prediction = _prediction(ev.COMPARATOR_RECORDS_PER_DOMAIN, 3)
    output_record = _save(
        tmp_path / "a1" / "predictions" / f"{cell}.safetensors",
        {"predictions": prediction},
        metadata={
            "schema": "token-reconstruction.trr-p11-a1-a2-prediction.v1",
            "task_id": ev.TASK_ID,
            "method_id": ev.COMPARATOR_METHOD,
            "cell_id": cell,
            "prediction_tensor_sha256": ev.p10.tensor_digest(prediction),
        },
    )
    receipt_record = _write_json(
        tmp_path / "a1" / "predictions" / f"{cell}.receipt.json",
        {
            "schema": ev.A1_PREDICTION_RECEIPT_SCHEMA,
            "task_id": ev.TASK_ID,
            "status": "A1_A2_K256_PREDICTIONS_COMPLETE_NO_TRUTH",
            "selection_sha256": selection_sha256,
            "output": output_record,
            "observations": fixture["observations"][cell],
            "methods": {
                "a1_a2_k256": {
                    "state_file_binding": fixture["states"][ev.COMPARATOR_METHOD]["file"],
                    "tensor_sha256": ev.p10.tensor_digest(prediction),
                    "shape": list(prediction.shape),
                    "dtype": str(prediction.dtype),
                }
            },
            "trace": trace_record,
            "cost": cost_record,
            "tensor_sha256": ev.p10.tensor_digest(prediction),
            "candidate_arrays_persisted": True,
            "source_text_loaded": False,
            "token_ids_loaded": False,
            "target_labels_loaded": False,
            "truth_opened": False,
            "p03_holdout_accessed": False,
        },
    )
    binding = {
        "file": output_record,
        "tensor_key": "predictions",
        "tensor_sha256": ev.p10.tensor_digest(prediction),
        "records": ev.COMPARATOR_RECORDS_PER_DOMAIN,
        "trace": trace_record,
        "cost": cost_record,
        "receipt": receipt_record,
    }
    normalized, tensor = ev._validate_prediction(
        ev.COMPARATOR_METHOD,
        cell,
        binding,
        root=tmp_path,
        state=fixture["states"][ev.COMPARATOR_METHOD],
        observation=fixture["observations"][cell],
        restore_links={},
        selection_sha256=selection_sha256,
    )
    assert tuple(normalized["trace"]["tensor_shapes"]["candidates"]) == (128, 128, 512)
    assert normalized["trace"]["tensor_dtypes"]["candidate_scores"] == "torch.float32"
    assert torch.equal(tensor, prediction)

    predictions: dict[str, torch.Tensor] = {}
    prediction_bindings: dict[str, dict[str, object]] = {}
    subsets: dict[str, tuple[int, ...] | None] = {}
    for method in ("new_expanded_fixed_B1", "new_current_fixed_B0", ev.COMPARATOR_METHOD):
        rows = ev.COMPARATOR_RECORDS_PER_DOMAIN if method == ev.COMPARATOR_METHOD else ev.RECORDS_PER_DOMAIN
        for item in ev.CELL_ORDER:
            key = f"{method}::{item}"
            predictions[key] = _prediction(rows, 0)
            prediction_bindings[key] = {"file": output_record}
            subsets[key] = tuple(range(rows)) if method == ev.COMPARATOR_METHOD else None
    prediction_bindings[f"{ev.COMPARATOR_METHOD}::{cell}"] = normalized
    frozen = ev.FrozenEvaluation(
        freeze_path=tmp_path / "freeze.json",
        freeze_record={},
        payload={"trace_files": []},
        predictions=predictions,
        prediction_bindings=prediction_bindings,
        states={},
        records_by_cell={item: ev.RECORDS_PER_DOMAIN for item in ev.CELL_ORDER},
        subsets=subsets,
    )
    p10_frozen = ev._as_p10_frozen(frozen)
    assert any(key.endswith("::candidate_trace") for key in p10_frozen.trace_records)
    truth = {item: _prediction(ev.RECORDS_PER_DOMAIN, 0) for item in ev.CELL_ORDER}
    inventory = ev.p10.build_error_inventory(p10_frozen, truth)
    diagnostics = inventory["cells"][cell]["rank_and_proposal_diagnostics"]["a1_proposal"]
    assert diagnostics["status"] == "AVAILABLE_BOUND_TRACE_OPAQUE"
