from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.trr_p08 import freeze_matrix as freeze


METHODS = freeze.METHOD_ORDER
SEEDS = freeze.SEEDS


def _write(path: Path, payload: bytes = b"fixture") -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(payload).hexdigest()}


def _json(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return _record(path)


def _record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    root = tmp_path / "repo"
    root.mkdir()
    plan = root / "experiments/TRR-P08/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"task_id": "TRR-P08", "status": "FROZEN_DESIGN_PRE_FIT", "truth_opened": False, "fresh_selection_started": False}), encoding="utf-8")
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    approval = root / "experiments/TRR-P08/review/root-design-approval.json"
    approval.parent.mkdir(parents=True)
    approval.write_text(json.dumps({"task_id": "TRR-P08", "status": "ROOT_DESIGN_APPROVED_PRE_FIT", "plan_sha256": plan_sha}), encoding="utf-8")

    qualification_path = root / "runtime/qualification/qualification.json"
    qualification_path.parent.mkdir(parents=True)
    qualification_path.write_text(json.dumps({
        "schema": freeze.QUALIFICATION_SCHEMA,
        "task_id": "TRR-P08",
        "status": "PASS",
        "states_retained": False,
        "command": ["python", "qualify"],
    }), encoding="utf-8")
    qualification = _record(qualification_path)

    state_records: dict[tuple[int, str], tuple[dict[str, object], dict[str, object], str, str]] = {}
    methods: list[dict[str, object]] = []
    for seed in SEEDS:
        for method_index, method in enumerate(METHODS):
            selected_path = root / f"runtime/fit/seed-{seed}/{method}/selected.safetensors"
            final_path = root / f"runtime/fit/seed-{seed}/{method}/final.safetensors"
            selected = _write(selected_path, f"selected-{seed}-{method}".encode())
            final = _write(final_path, f"final-{seed}-{method}".encode())
            selected_tensor = hashlib.sha256(f"selected-tensor-{seed}-{method}".encode()).hexdigest()
            final_tensor = hashlib.sha256(f"final-tensor-{seed}-{method}".encode()).hexdigest()
            state_records[(seed, method)] = (selected, final, selected_tensor, final_tensor)
            methods.append({
                "arm_id": method,
                "seed": seed,
                "status": "PASS",
                "steps": 3000,
                "final_step": 3000,
                "selected_step": 100 + method_index,
                "schedule_sha256": "a" * 64,
                "state": {**selected, "state_sha256": selected_tensor},
                "final_state": {**final, "state_sha256": final_tensor},
            })
    fit_path = root / "runtime/fit/main_fit_receipt.json"
    fit = {
        "schema": freeze.FIT_SCHEMA,
        "task_id": "TRR-P08",
        "status": "PASS",
        "source_commit": "1" * 40,
        "command": ["python", "fit", "--mode", "main"],
        "geometry": {"fit": [1200, 128, 2048], "validation": [48, 128, 2048], "record_batch_size": 8, "position_budget": 512, "steps": 3000, "validation_every": 100},
        "seeds": list(SEEDS),
        "initialization": {"W": "identity", "b": "zeros", "s": 3.0},
        "runtime_components": {"source_token_access": False, "target_truth_access": False, "candidate_simulations": 0, "a2_student": False},
        "qualification_receipt": {**qualification, "status": "PASS"},
        "methods": methods,
    }
    fit_path.write_text(json.dumps(fit, sort_keys=True), encoding="utf-8")
    fit_record = _record(fit_path)

    observation_records: dict[str, dict[str, object]] = {}
    observation_cells: list[dict[str, object]] = []
    record_ids = {domain: ("b" if domain == "pile" else "c") * 64 for domain in freeze.DOMAINS}
    for cell_id in freeze.CELL_ORDER:
        domain, target = cell_id.split("__", 1)
        asset_path = root / f"runtime/observations/{cell_id}.safetensors"
        asset = _write(asset_path, f"observation-{cell_id}".encode())
        observation = {**asset, "shape": [256, 128, 2048], "stored_sequence_tokens": 128, "scored_post_bos_tokens": 127}
        observation_records[cell_id] = observation
        observation_cells.append({"cell_id": cell_id, "style": domain, "condition": target, "record_ids_sha256": record_ids[domain], "observation": observation})
    observation_path = root / "runtime/observations.json"
    observation_path.write_text(json.dumps({
        "schema": next(iter(freeze.OBSERVATION_SCHEMAS)),
        "task_id": "TRR-P08",
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "records_per_domain": 256,
        "cell_order": list(freeze.CELL_ORDER),
        "cells": observation_cells,
        "sequence_tokens_including_bos": 128,
        "scored_post_bos_tokens": 127,
        "hidden_size": 2048,
        "source_pairing": {"same_record_ids_across_targets": True, "record_ids_sha256": record_ids},
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "truth_opened": False,
    }, sort_keys=True), encoding="utf-8")
    observation_record = _record(observation_path)

    prediction_cells: dict[str, dict[str, dict[str, object]]] = {}
    state_bindings: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        for method in METHODS:
            selected = state_records[(seed, method)][0]
            state_bindings[f"{seed}::{method}"] = selected
    for cell_id in freeze.CELL_ORDER:
        domain, target = cell_id.split("__", 1)
        prediction_cells[cell_id] = {}
        for seed in SEEDS:
            prediction_cells[cell_id][str(seed)] = {}
            for method in METHODS:
                prediction_path = root / f"runtime/predictions/{cell_id}/{seed}/{method}.safetensors"
                tie_path = root / f"runtime/ties/{cell_id}/{seed}/{method}.safetensors"
                prediction = _write(prediction_path, f"prediction-{cell_id}-{seed}-{method}".encode())
                ties = _write(tie_path, f"ties-{cell_id}-{seed}-{method}".encode())
                selected = state_records[(seed, method)]
                prediction_cells[cell_id][str(seed)][method] = {
                    "schema": "token-reconstruction.trr-p08-prediction-descriptor.v1",
                    "task_id": "TRR-P08",
                    "domain": domain,
                    "target": target,
                    "method_id": method,
                    "seed": seed,
                    "records": 256,
                    "shape": [256, 128],
                    "sequence_tokens": 128,
                    "scored_post_bos_tokens": 127,
                    "record_ids_sha256": record_ids[domain],
                    "observation": observation_records[cell_id],
                    "state": selected[0],
                    "state_tensor_sha256": selected[2],
                    "timing": {"repeat_prediction_exact": True, "measured_passes": 3, "attention_mask_sha256": "d" * 64, "position_ids_sha256": "e" * 64},
                    "prediction": prediction,
                    "tie_counts": ties,
                    "truth_opened": False,
                    "source_text_loaded": False,
                    "target_labels_loaded": False,
                    "candidate_arrays_persisted": False,
                }
    prediction_path = root / "runtime/predictions/p08_predictions.json"
    prediction_path.write_text(json.dumps({
        "schema": "token-reconstruction.trr-p08-prediction-manifest.v1",
        "task_id": "TRR-P08",
        "status": "FROZEN_P08_PREDICTIONS_NO_TRUTH",
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "code_commit": "2" * 40,
        "command": ["python", "scripts/trr_p08/run_predictions.py", "--device", "cuda"],
        "fit_receipt": fit_record,
        "observation_manifest": observation_record,
        "fit_source_commit": "1" * 40,
        "domains": list(freeze.DOMAINS),
        "target_conditions": list(freeze.TARGETS),
        "cell_order": list(freeze.CELL_ORDER),
        "method_order": list(METHODS),
        "replicate_seeds": list(SEEDS),
        "geometry": {"records_per_domain": 256, "sequence_tokens": 128, "scored_post_bos_tokens": 127, "hidden_size": 2048, "batch_records": 8, "projection_chunk": 512},
        "state_bindings": state_bindings,
        "student_cells": prediction_cells,
        "predictions_count": 32,
        "predictions_complete": True,
    }, sort_keys=True), encoding="utf-8")
    return {"root": root, "plan": plan, "approval": approval, "fit": fit_path, "observations": observation_path, "predictions": prediction_path, "output": root / "runtime/joint-freeze.json", "plan_sha": plan_sha}


def test_joint_freeze_binds_all_metadata_without_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = freeze.assemble_joint_freeze(
        repository_root=fixture["root"], plan_path=fixture["plan"], approval_path=fixture["approval"], fit_receipt_path=fixture["fit"], prediction_manifest_path=fixture["predictions"], observation_manifest_path=fixture["observations"], output_path=fixture["output"], expected_plan_sha256=fixture["plan_sha"],
    )
    assert receipt["status"] == "JOINT_FREEZE_VALIDATED_NO_TRUTH"
    assert receipt["truth_opened"] is False
    assert receipt["prediction_count"] == 32
    assert fixture["output"].exists()


def test_joint_freeze_rejects_prediction_file_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["predictions"].read_text(encoding="utf-8"))
    descriptor = payload["student_cells"]["pile__public_base"]["6106"][METHODS[0]]
    Path(descriptor["prediction"]["path"]).write_bytes(b"mutated")
    with pytest.raises(freeze.P08FreezeError, match="prediction"):
        freeze.validate_prediction_freeze(repository_root=fixture["root"], plan_path=fixture["plan"], approval_path=fixture["approval"], fit_receipt_path=fixture["fit"], prediction_manifest_path=fixture["predictions"], observation_manifest_path=fixture["observations"], expected_plan_sha256=fixture["plan_sha"])


def test_joint_freeze_rejects_truth_flag_and_missing_cell(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["predictions"].read_text(encoding="utf-8"))
    payload["truth_opened"] = True
    fixture["predictions"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(freeze.P08FreezeError, match="forbidden flag"):
        freeze.validate_prediction_freeze(repository_root=fixture["root"], plan_path=fixture["plan"], approval_path=fixture["approval"], fit_receipt_path=fixture["fit"], prediction_manifest_path=fixture["predictions"], observation_manifest_path=fixture["observations"], expected_plan_sha256=fixture["plan_sha"])
