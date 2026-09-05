from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts import trr0005_freeze_confirmation as freeze
from scripts import trr0005_predict_confirmation as predict
from scripts import trr0005_score_confirmation as score
from token_reconstruction import trr0005_contract as contract
from token_reconstruction.footing import external_file_record, file_record, sha256_file
from token_reconstruction.freeze import create_freeze_receipt


def _registration() -> dict:
    bindings = {
        method_id: {
            "code_commit": "a" * 40,
            "state_sha256": "b" * 64,
            "state_path": f"experiments/TRR-0005/states/{method_id}.safetensors",
        }
        for method_id in contract.METHOD_IDS
    }
    return contract.build_registration(
        status="FROZEN_METHOD_REGISTRATION",
        code_commit="a" * 40,
        state_bindings=bindings,
    )


def _panel() -> dict:
    cells = {}
    for style in contract.STYLE_ORDER:
        ids = [f"{style}-holdout-{index:03d}" for index in range(contract.RECORDS_PER_DOMAIN)]
        records = [
            {
                "record_id": record_id,
                "public_record_sha256": "c" * 64,
                "raw_index": index,
                "source_index": index,
                "valid_tokens": contract.SEQUENCE_TOKENS,
            }
            for index, record_id in enumerate(ids)
        ]
        for condition in contract.CONDITION_ORDER:
            cell_id = f"{style}__{condition}"
            cells[cell_id] = {
                "cell_id": cell_id,
                "style": style,
                "condition": condition,
                "records": copy.deepcopy(records),
            }
    return {
        "schema": contract.PANEL_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_FRESH_CONFIRMATION_PANEL",
        "sequence_tokens": contract.SEQUENCE_TOKENS,
        "records_per_domain": contract.RECORDS_PER_DOMAIN,
        "cells": cells,
    }


def _descriptor(cell_id: str, method_id: str, *, candidates: bool = False) -> dict:
    value = {
        "schema": contract.PREDICTION_SCHEMA,
        "task_id": contract.TASK_ID,
        "cell_id": cell_id,
        "method_id": method_id,
        "shape": [contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS],
        "candidate_policy": contract.CANDIDATE_POLICIES[method_id],
        "candidate_arrays_present": candidates,
        "warmup_runs_per_record": 1,
        "measured_runs_per_record": 1,
        "warmup_output_exact_match_measured": True,
        "measured_output_selected": True,
    }
    if contract.CANDIDATE_POLICIES[method_id] == "output_only":
        value["candidate_output"] = "omitted_after_decision"
    value["predictions"] = torch.full(
        (contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS),
        contract.INVALID_TOKEN_ID,
        dtype=torch.long,
    )
    value["predictions"][:, 0] = contract.BOS_TOKEN_ID
    return value


def _matrix_descriptors() -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    predictions = {}
    timings = {}
    for cell_id in contract.EXPECTED_CELL_IDS:
        for method_id in contract.METHOD_IDS:
            value = _descriptor(cell_id, method_id)
            predictions[(cell_id, method_id)] = value
            timings[(cell_id, method_id)] = {
                "warmup_runs_per_record": 1,
                "measured_runs_per_record": 1,
                "warmup_output_exact_match_measured": True,
            }
    return predictions, timings


def _selection() -> dict:
    return {
        "schema": score.PUBLIC_SELECTION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_PUBLIC_VALIDATION_SELECTION",
        "selection_stage": "public_validation_before_fresh_evaluation",
        "truth_accessed": False,
        "distributions": {
            "original": {
                "selected_method_id": "original__joint_full_affine",
                "candidate_method_ids": [
                    "original__joint_full_affine",
                    "original__affine_trained_diagonal_attention128",
                ],
            },
            "enriched": {
                "selected_method_id": "enriched__affine_trained_diagonal_attention128",
                "candidate_method_ids": [
                    "enriched__joint_full_affine",
                    "enriched__affine_trained_diagonal_attention128",
                ],
            },
        },
    }


def _cell_inputs() -> dict[str, dict]:
    result = {}
    for cell_id in contract.EXPECTED_CELL_IDS:
        style, condition = cell_id.split("__", 1)
        record_ids = [f"{style}-holdout-{index:03d}" for index in range(contract.RECORDS_PER_DOMAIN)]
        truth = torch.full(
            (contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS),
            contract.PAD_TOKEN_ID,
            dtype=torch.long,
        )
        truth[:, 0] = contract.BOS_TOKEN_ID
        mask = torch.zeros((contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS), dtype=torch.bool)
        mask[:, :contract.SEQUENCE_TOKENS] = True
        positions = mask.to(torch.long).cumsum(1).sub(1).clamp_min(0)
        result[cell_id] = {
            "record_ids": record_ids,
            "truth": truth,
            "attention_mask": mask,
            "position_ids": positions,
        }
    return result



def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _executable_fixture(tmp_path: Path, *, bad: str | None = None) -> dict:
    """Build a small real-file matrix; no private truth is read by setup."""

    root = tmp_path
    panel = _panel()
    panel_path = root / "panel.json"
    _write_json(panel_path, panel)
    selection = _selection()
    plan_path = root / "selection_plan.json"
    _write_json(plan_path, {"public_validation_selection": selection})
    code_path = root / "code.py"
    code_path.write_text("# frozen synthetic executable binding\n", encoding="utf-8")
    runtime_root = root / "runtime"
    embedding_path = runtime_root / "normalized_embeddings.safetensors"
    p0_checkpoint_path = runtime_root / "public_p0.safetensors"
    p0_config_path = runtime_root / "public_p0_config.json"
    runtime_root.mkdir(parents=True, exist_ok=True)
    embedding_path.write_bytes(b"synthetic normalized public E\n")
    p0_checkpoint_path.write_bytes(b"synthetic public P0 checkpoint\n")
    p0_config_path.write_bytes(b'{"cut_depth":4}\n')
    runtime_assets = {
        "public_embedding_table": external_file_record(embedding_path),
    }
    a2_runtime_assets = {
        **runtime_assets,
        "public_prefix_checkpoint": external_file_record(p0_checkpoint_path),
        "public_prefix_config": external_file_record(p0_config_path),
    }
    bindings = {}
    for method_id in contract.METHOD_IDS:
        state_path = root / "states" / f"{method_id}.bin"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(f"state:{method_id}\n".encode())
        state_record = file_record(state_path, repository_root=root)
        code_record = file_record(code_path, repository_root=root)
        bindings[method_id] = {
            "state_path": state_record["path"],
            "state_bytes": state_record["bytes"],
            "state_sha256": state_record["sha256"],
            "code": [code_record],
            "code_commit": "a" * 40,
            "runtime_assets": dict(
                a2_runtime_assets if method_id == "frozen_a1_a2_k256" else runtime_assets
            ),
        }
    registration = contract.build_registration(
        status="FROZEN_METHOD_REGISTRATION",
        code_commit="a" * 40,
        state_bindings=bindings,
    )
    registration_path = root / "registration.json"
    _write_json(registration_path, registration)

    mask = torch.ones((contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS), dtype=torch.long)
    positions = mask.cumsum(1).sub(1)
    observations = {}
    for cell_id in contract.EXPECTED_CELL_IDS:
        obs_path = root / "observations" / f"{cell_id}.json"
        _write_json(
            obs_path,
            {
                "schema": "synthetic-public-observation.v1",
                "shape": [contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS, 2048],
                "attention_mask": mask.tolist(),
                "position_ids": positions.tolist(),
            },
        )
        record = file_record(obs_path, repository_root=root)
        observations[cell_id] = {
            **record,
            "shape": [contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS, 2048],
            "attention_mask": mask.tolist(),
            "position_ids": positions.tolist(),
        }

    output_root = root / "frozen_predictions"
    predictions, timings = _matrix_descriptors()
    panel_sha = sha256_file(panel_path)
    plan_sha = sha256_file(plan_path)
    for cell_id in contract.EXPECTED_CELL_IDS:
        for method_id in contract.METHOD_IDS:
            value = torch.zeros(
                (contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS), dtype=torch.long
            )
            value[:, 0] = contract.BOS_TOKEN_ID
            path = output_root / cell_id.split("__", 1)[0] / cell_id.split("__", 1)[1] / f"{method_id}.safetensors"
            artifact = predict.write_prediction_artifact(
                path,
                cell_id=cell_id,
                method_id=method_id,
                predictions=value,
                binding=bindings[method_id],
                panel_sha256=panel_sha,
                selection_plan_sha256=plan_sha,
                observation_sha256=observations[cell_id]["sha256"],
                repository_root=root,
            )
            predictions[(cell_id, method_id)]["prediction_artifact"] = artifact

    # Invalid tensor cases are created before the receipt so the receipt still
    # proves the exact bytes that the pretruth tensor validator must reject.
    if bad in {"out_of_range", "shape", "missing_tensor"}:
        key = next(iter(predictions))
        descriptor = predictions[key]
        artifact_path = root / descriptor["prediction_artifact"]["path"]
        with safe_open(artifact_path, framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        if bad == "out_of_range":
            invalid = torch.zeros((contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS), dtype=torch.long)
            invalid[:, 0] = contract.BOS_TOKEN_ID
            invalid[:, 1] = 128256
            tensors = {"predictions": invalid}
        elif bad == "shape":
            invalid = torch.zeros((contract.RECORDS_PER_DOMAIN - 1, contract.SEQUENCE_TOKENS), dtype=torch.long)
            invalid[:, 0] = contract.BOS_TOKEN_ID
            tensors = {"predictions": invalid}
        else:
            tensors = {"unexpected": torch.zeros((1,), dtype=torch.long)}
        save_file(tensors, str(artifact_path), metadata=metadata)
        descriptor["prediction_artifact"] = file_record(artifact_path, repository_root=root)

    truth_path = root / "private_truth.sidecar"
    truth_path.write_bytes(b"private truth placeholder\n")
    receipt_path = root / "freeze_receipt.json"
    metadata = {
        "task_id": contract.TASK_ID,
        "panel_sha256": panel_sha,
        "selection_plan_sha256": plan_sha,
        "registration_sha256": sha256_file(registration_path),
        "method_ids": list(contract.METHOD_IDS),
        "truth_opened": False,
        "public_validation_selection": selection,
    }
    create_freeze_receipt(
        repository_root=root,
        frozen_root=output_root,
        plan_path=plan_path,
        receipt_path=receipt_path,
        preregistration_commit="a" * 40,
        created_utc="2026-09-06T00:00:00+00:00",
        metadata=metadata,
    )
    if bad == "missing":
        key = next(iter(predictions))
        (root / predictions[key]["prediction_artifact"]["path"]).unlink()
    elif bad == "truncated":
        key = next(iter(predictions))
        path = root / predictions[key]["prediction_artifact"]["path"]
        os.chmod(path, 0o644)
        path.write_bytes(b"truncated")
    elif bad == "stale_state":
        state_path = root / bindings[contract.METHOD_IDS[0]]["state_path"]
        state_path.write_bytes(b"stale-state\n")
    elif bad == "stale_code":
        code_path.write_bytes(b"stale-code\n")
    elif bad == "stale_embedding":
        embedding_path.write_bytes(b"stale normalized public E\n")
    return {
        "root": root,
        "panel": panel,
        "registration": registration,
        "predictions": predictions,
        "timings": timings,
        "selection": selection,
        "observations": observations,
        "panel_path": panel_path,
        "registration_path": registration_path,
        "plan_path": plan_path,
        "output_root": output_root,
        "receipt_path": receipt_path,
        "truth_path": truth_path,
    }


def _truth_cli_fixture(tmp_path: Path, *, row_order: bool = False) -> dict:
    method_freeze_sha = "d" * 64
    panel = _panel()
    panel["method_freeze_sha256"] = method_freeze_sha
    for cell_id in contract.EXPECTED_CELL_IDS:
        observation_path = tmp_path / "observations" / f"{cell_id}.bin"
        observation_path.parent.mkdir(parents=True, exist_ok=True)
        observation_path.write_bytes(f"public observation {cell_id}\n".encode())
        panel["cells"][cell_id]["observation"] = file_record(
            observation_path,
            repository_root=tmp_path,
        )
    panel_path = tmp_path / "panel.json"
    _write_json(panel_path, panel)
    plan_path = tmp_path / "selection_plan.json"
    _write_json(
        plan_path,
        {
            "public_validation_selection": _selection(),
            "method_freeze_sha256": method_freeze_sha,
        },
    )

    ids_by_style = {
        style: [
            row["record_id"]
            for row in panel["cells"][f"{style}__{contract.CONDITION_ORDER[0]}"]["records"]
        ]
        for style in contract.STYLE_ORDER
    }
    tensors = {}
    for cell_id in contract.EXPECTED_CELL_IDS:
        truth = torch.full(
            (contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS),
            contract.PAD_TOKEN_ID,
            dtype=torch.int64,
        )
        truth[:, 0] = contract.BOS_TOKEN_ID
        tensors[f"{cell_id}__token_ids"] = truth
        tensors[f"{cell_id}__attention_mask"] = torch.ones_like(truth, dtype=torch.uint8)
        tensors[f"{cell_id}__position_ids"] = torch.arange(
            contract.SEQUENCE_TOKENS, dtype=torch.int64
        ).repeat(contract.RECORDS_PER_DOMAIN, 1)
    sidecar_path = tmp_path / ("truth_row_order_bad.safetensors" if row_order else "truth.safetensors")
    header_ids = {
        style: list(reversed(ids)) if row_order and style == "pile" else ids
        for style, ids in ids_by_style.items()
    }
    metadata = {
        "schema": score.TRUTH_MANIFEST_SCHEMA,
        "task_id": contract.TASK_ID,
        "panel_sha256": sha256_file(panel_path),
        "selection_plan_sha256": sha256_file(plan_path),
        "method_freeze_sha256": method_freeze_sha,
        "record_ids_pile": json.dumps(header_ids["pile"], sort_keys=True, separators=(",", ":")),
        "record_ids_finance": json.dumps(header_ids["finance"], sort_keys=True, separators=(",", ":")),
        "truth_source": "synthetic evaluator sidecar",
        "truth_opened": "false",
    }
    save_file(tensors, str(sidecar_path), metadata=metadata)
    truth_keys = sorted(tensors)
    truth_manifest = {
        "schema": score.TRUTH_MANIFEST_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": score.TRUTH_MANIFEST_STATUS,
        "truth_file": external_file_record(sidecar_path),
        "panel": external_file_record(panel_path),
        "selection_plan": external_file_record(plan_path),
        "method_freeze_sha256": method_freeze_sha,
        "observation_sha256": {
            cell_id: panel["cells"][cell_id]["observation"]["sha256"]
            for cell_id in contract.EXPECTED_CELL_IDS
        },
        "record_ids_sha256": {
            style: score._cli_json_sha256(ids)
            for style, ids in ids_by_style.items()
        },
        "cell_order": list(contract.EXPECTED_CELL_IDS),
        "truth_tensor_keys": truth_keys,
        "truth_opened": False,
        "reconstruction_root_contains_truth": False,
    }
    output_root = tmp_path / "frozen_predictions"
    binding_path = output_root / "evaluator_binding.json"
    _write_json(binding_path, truth_manifest)
    receipt_path = tmp_path / "freeze_receipt.json"
    create_freeze_receipt(
        repository_root=tmp_path,
        frozen_root=output_root,
        plan_path=plan_path,
        receipt_path=receipt_path,
        preregistration_commit="a" * 40,
        created_utc="2026-09-06T00:00:00+00:00",
        metadata={"task_id": contract.TASK_ID, "truth_opened": False},
    )
    return {
        "panel": panel,
        "panel_path": panel_path,
        "selection_plan": {
            "public_validation_selection": _selection(),
            "method_freeze_sha256": method_freeze_sha,
        },
        "selection_plan_path": plan_path,
        "truth_path": sidecar_path,
        "truth_binding_path": binding_path,
        "truth_manifest": truth_manifest,
        "receipt_path": receipt_path,
        "output_root": output_root,
    }


def test_cli_truth_binding_validates_and_loads_bound_sidecar(tmp_path: Path) -> None:
    fixture = _truth_cli_fixture(tmp_path)
    result = score._cli_truth_cells(
        fixture["truth_path"],
        panel=fixture["panel"],
        selection_plan=fixture["selection_plan"],
        truth_binding_path=fixture["truth_binding_path"],
        panel_path=fixture["panel_path"],
        selection_plan_path=fixture["selection_plan_path"],
        receipt_path=fixture["receipt_path"],
        output_root=fixture["output_root"],
        repository_root=tmp_path,
    )
    assert set(result) == set(contract.EXPECTED_CELL_IDS)
    assert tuple(result[contract.EXPECTED_CELL_IDS[0]]["record_ids"]) == tuple(
        row["record_id"]
        for row in fixture["panel"]["cells"][contract.EXPECTED_CELL_IDS[0]]["records"]
    )


@pytest.mark.parametrize("bad", ["wrong_sidecar", "row_order"])
def test_cli_truth_binding_rejects_wrong_sidecar_or_row_order(tmp_path: Path, bad: str) -> None:
    fixture = _truth_cli_fixture(tmp_path, row_order=bad == "row_order")
    truth_path = fixture["truth_path"]
    if bad == "wrong_sidecar":
        wrong_path = tmp_path / "wrong_same_shape.safetensors"
        wrong_path.write_bytes(truth_path.read_bytes())
        truth_path = wrong_path
    with pytest.raises(score.ConfirmationScoreError, match="path differs|row order"):
        score._cli_truth_cells(
            truth_path,
            panel=fixture["panel"],
            selection_plan=fixture["selection_plan"],
            truth_binding_path=fixture["truth_binding_path"],
            panel_path=fixture["panel_path"],
            selection_plan_path=fixture["selection_plan_path"],
            receipt_path=fixture["receipt_path"],
            output_root=fixture["output_root"],
            repository_root=tmp_path,
        )


def test_cli_truth_binding_rejects_unfrozen_descriptor(tmp_path: Path) -> None:
    fixture = _truth_cli_fixture(tmp_path)
    unfrozen = tmp_path / "unfrozen_evaluator_binding.json"
    unfrozen.write_bytes(fixture["truth_binding_path"].read_bytes())
    with pytest.raises(score.ConfirmationScoreError, match="inside the frozen prediction root"):
        score._cli_truth_cells(
            fixture["truth_path"],
            panel=fixture["panel"],
            selection_plan=fixture["selection_plan"],
            truth_binding_path=unfrozen,
            panel_path=fixture["panel_path"],
            selection_plan_path=fixture["selection_plan_path"],
            receipt_path=fixture["receipt_path"],
            output_root=fixture["output_root"],
            repository_root=tmp_path,
        )


def test_cli_prediction_descriptor_requires_driver_attestations(tmp_path: Path) -> None:
    entries = [
        {
            "cell_id": cell_id,
            "method_id": method_id,
            "prediction_path": "placeholder.safetensors",
        }
        for cell_id in contract.EXPECTED_CELL_IDS
        for method_id in contract.METHOD_IDS
    ]
    manifest = tmp_path / "predictions.json"
    _write_json(manifest, {"entries": entries})
    descriptors = score._cli_prediction_descriptors(
        manifest,
        output_root=tmp_path / "frozen",
        repository_root=tmp_path,
    )
    sample = descriptors[(contract.EXPECTED_CELL_IDS[0], contract.METHOD_IDS[0])]
    assert "warmup_output_exact_match_measured" not in sample
    assert "measured_output_selected" not in sample
    with pytest.raises(contract.ContractError, match="warmup/measured|measured output"):
        contract.validate_prediction_descriptor(
            sample,
            cell_id=contract.EXPECTED_CELL_IDS[0],
            method_id=contract.METHOD_IDS[0],
        )


def _run_executable_fixture(fixture: dict) -> None:
    calls = []

    def truth_loader():
        calls.append(True)
        return _cell_inputs()

    with pytest.raises(score.PretruthGateError):
        score.score_with_truth_loader(
            panel=fixture["panel"],
            registration=fixture["registration"],
            prediction_descriptors=fixture["predictions"],
            timing_descriptors=fixture["timings"],
            public_validation_selection=fixture["selection"],
            repository_root=fixture["root"],
            receipt_path=fixture["receipt_path"],
            truth_path=fixture["truth_path"],
            output_root=fixture["output_root"],
            panel_path=fixture["panel_path"],
            registration_path=fixture["registration_path"],
            selection_plan_path=fixture["plan_path"],
            observation_descriptors=fixture["observations"],
            truth_loader=truth_loader,
            bootstrap_draws=2,
        )
    assert calls == []


def test_frozen_selection_drives_best_positionwise_baseline() -> None:
    pairs = score._declared_comparison_pairs(_selection())
    by_label = {label: (baseline, method) for label, baseline, method in pairs}
    assert by_label["enriched__causal_vs_best_positionwise"][0] == (
        "enriched__affine_trained_diagonal_attention128"
    )
    with pytest.raises(score.ConfirmationScoreError, match="selection"):
        score._declared_comparison_pairs(None)


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "truncated",
        "out_of_range",
        "shape",
        "missing_tensor",
        "stale_state",
        "stale_code",
        "stale_embedding",
    ],
)
def test_executable_pretruth_gate_rejects_bad_public_evidence(tmp_path: Path, bad: str) -> None:
    _run_executable_fixture(_executable_fixture(tmp_path, bad=bad))


def test_executable_pretruth_gate_opens_truth_only_after_all_files_pass(tmp_path: Path) -> None:
    fixture = _executable_fixture(tmp_path)
    calls = []

    def truth_loader():
        calls.append(True)
        return _cell_inputs()

    result = score.score_with_truth_loader(
        panel=fixture["panel"],
        registration=fixture["registration"],
        prediction_descriptors=fixture["predictions"],
        timing_descriptors=fixture["timings"],
        public_validation_selection=fixture["selection"],
        repository_root=fixture["root"],
        receipt_path=fixture["receipt_path"],
        truth_path=fixture["truth_path"],
        output_root=fixture["output_root"],
        panel_path=fixture["panel_path"],
        registration_path=fixture["registration_path"],
        selection_plan_path=fixture["plan_path"],
        observation_descriptors=fixture["observations"],
        truth_loader=truth_loader,
        bootstrap_draws=2,
    )
    assert calls == [True]
    assert result["truth_gate"]["executable_public_gate"]["prediction_artifact_count"] == 32


def test_contract_has_two_banks_and_exactly_eight_methods() -> None:
    assert tuple(contract.FIT_BANKS) == ("original", "enriched")
    assert all(bank["records"] == 1200 for bank in contract.FIT_BANKS.values())
    assert all(bank["post_bos_positions"] == 124371 for bank in contract.FIT_BANKS.values())
    assert len(contract.METHOD_IDS) == 8
    assert len(contract.EXPECTED_CELL_IDS) == 4
    ledger = contract.build_preplanned_ledger()
    assert ledger["status"] == "PREPLANNED_NO_HOLDOUT_SELECTED"
    assert "record_id" not in str(ledger)
    assert ledger["holdout"]["records_per_domain"] == 128


def test_sorted_json_round_trip_accepts_contract_maps_and_preserves_lists() -> None:
    registration = json.loads(json.dumps(_registration(), sort_keys=True))
    panel = json.loads(json.dumps(_panel(), sort_keys=True))

    contract.validate_registration(registration, require_frozen=True)
    by_domain = contract.validate_panel_descriptor(panel)

    assert tuple(registration["method_ids"]) == contract.METHOD_IDS
    assert tuple(row["id"] for row in registration["methods"]) == contract.METHOD_IDS
    assert set(registration["state_bindings"]) == set(contract.METHOD_IDS)
    assert set(panel["cells"]) == set(contract.EXPECTED_CELL_IDS)
    assert tuple(by_domain) == contract.STYLE_ORDER


def test_complete_public_gate_requires_all_32_outputs_and_1_plus_1_timing() -> None:
    panel = _panel()
    registration = _registration()
    predictions, timings = _matrix_descriptors()
    gate = contract.validate_complete_public_matrix(
        panel,
        registration,
        predictions,
        timing_descriptors=timings,
    )
    assert gate["prediction_artifacts"] == 32
    assert gate["timing_receipts"] == 32
    with pytest.raises(contract.ContractError, match="incomplete"):
        contract.validate_complete_public_matrix(
            panel,
            registration,
            {key: value for key, value in predictions.items() if key != next(iter(predictions))},
            timing_descriptors=timings,
        )
    bad_timing = copy.deepcopy(timings)
    bad_timing[next(iter(bad_timing))]["measured_runs_per_record"] = 3
    with pytest.raises(contract.ContractError, match="timing count changed"):
        contract.validate_complete_public_matrix(panel, registration, predictions, timing_descriptors=bad_timing)


def test_a2_candidate_arrays_are_rejected_when_output_only() -> None:
    cell_id = contract.EXPECTED_CELL_IDS[0]
    method_id = "frozen_a1_a2_k256"
    with pytest.raises(contract.ContractError, match="must not be persisted"):
        contract.validate_prediction_descriptor(
            _descriptor(cell_id, method_id, candidates=True),
            cell_id=cell_id,
            method_id=method_id,
        )


def test_truth_loader_is_not_called_on_failed_public_gate() -> None:
    panel = _panel()
    registration = _registration()
    predictions, timings = _matrix_descriptors()
    del predictions[next(iter(predictions))]
    calls = []

    def truth_loader():
        calls.append(True)
        return _cell_inputs()

    with pytest.raises(contract.ContractError, match="incomplete"):
        score.score_with_truth_loader(
            panel=panel,
            registration=registration,
            prediction_descriptors=predictions,
            timing_descriptors=timings,
            truth_loader=truth_loader,
            bootstrap_draws=8,
        )
    assert calls == []


def test_zero_discordance_has_positive_familywise_exact_bound() -> None:
    value = score.exact_net_benefit_bound(beneficial=0, harmful=0, records=128)
    assert value["zero_discordance_is_not_no_effect"] is True
    assert value["net_upper_pp"] == pytest.approx(4.922726535, abs=1e-8)
    assert value["net_upper_pp"] > 0.0
    simple = score.exact_beneficial_discordance_bound(
        beneficial=0,
        harmful=0,
        records=128,
    )
    assert simple["beneficial_upper_pp"] == pytest.approx(2.313, abs=0.01)


def test_paired_bootstrap_uses_micro_correct_scored_ratio() -> None:
    left = []
    right = []
    for index in range(contract.RECORDS_PER_DOMAIN):
        left.append(
            {
                "record_id": f"r{index}",
                "correct_tokens": 2 if index == 0 else 1,
                "scored_tokens": 2 if index == 0 else 100,
                "token_accuracy": (1.0 if index == 0 else 0.01),
                "exact_record": False,
            }
        )
        right.append(
            {
                "record_id": f"r{index}",
                "correct_tokens": 0,
                "scored_tokens": 2 if index == 0 else 100,
                "token_accuracy": 0.0,
                "exact_record": False,
            }
        )
    result = score.paired_token_bootstrap(left, right, draws=32, seed=5005)
    assert result["left_estimate"] == pytest.approx(129 / 12702)
    assert result["right_estimate"] == 0.0
    assert result["seed"] == 5005


def test_score_matrix_keeps_cells_separate_and_emits_joint_diagnostics() -> None:
    panel = _panel()
    registration = _registration()
    predictions, timings = _matrix_descriptors()
    cells = _cell_inputs()
    # Give each method a valid active output equal to truth.  Padded rows were
    # already marked -1 by the descriptor fixture.
    for cell_id in contract.EXPECTED_CELL_IDS:
        truth = cells[cell_id]["truth"]
        active = cells[cell_id]["attention_mask"]
        for method_id in contract.METHOD_IDS:
            predictions[(cell_id, method_id)]["predictions"][active] = truth[active]
    counts = {
        "original": {contract.PAD_TOKEN_ID: 1},
        "enriched": {contract.PAD_TOKEN_ID: 2},
    }
    result = score.score_matrix(
        panel=panel,
        registration=registration,
        prediction_descriptors=predictions,
        timing_descriptors=timings,
        cell_inputs=cells,
        frequency_counts=counts,
        bootstrap_draws=8,
        bootstrap_seed=5005,
        public_validation_selection=_selection(),
    )
    assert result["matrix"]["pooled_headline"] is False
    assert len(result["cells_results"]) == 32
    row = result["cells_results"]["pile__public_base__enriched__affine_causal_h_attention128"]
    assert row["domain"] == "pile"
    assert row["joint_frequency_position"]["domain"] == "pile"
    assert len(row["joint_frequency_position"]["rows"]) == 16
    assert set(row["frequency_references"]) == {"original", "enriched"}
    primary = result["method_comparisons"][
        "finance__public_base__enriched__causal_vs_diagonal"
    ]
    assert primary["token_bootstrap"]["upper_tail_alpha"] == pytest.approx(0.05 / 16)
    assert primary["exact_net_benefit_bound"]["tail_alpha_each"] == pytest.approx(0.05 / 32)


def test_freeze_adapter_returns_pretruth_status_without_files() -> None:
    panel = _panel()
    registration = _registration()
    predictions, timings = _matrix_descriptors()
    result = freeze.freeze_public_matrix(
        root=Path("."),
        panel=panel,
        registration=registration,
        prediction_descriptors=predictions,
        timing_descriptors=timings,
    )
    assert result["status"] == "PUBLIC_MATRIX_VALIDATED_NO_TRUTH_OPENED"
    assert result["truth_opened"] is False


def test_predictor_warmup_and_measured_ids_are_exactly_paired() -> None:
    from scripts import trr0005_predict_confirmation as predict

    records = list(range(contract.RECORDS_PER_DOMAIN))

    def predict_one(record: int):
        value = torch.full((contract.SEQUENCE_TOKENS,), contract.INVALID_TOKEN_ID, dtype=torch.long)
        value[0] = contract.BOS_TOKEN_ID
        value[1 : 1 + 4 + (record % 3)] = torch.arange(4 + (record % 3), dtype=torch.long)
        return value

    outputs, timing = predict.run_warmed_prediction(
        method_id=contract.METHOD_IDS[2],
        records=records,
        predict_one=predict_one,
    )
    assert tuple(outputs.shape) == (contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS)
    assert timing["warmup_runs_per_record"] == 1
    assert timing["measured_runs_per_record"] == 1
    assert timing["warmup_output_exact_match_measured"] is True


def test_predictor_rejects_warmup_measured_drift() -> None:
    from scripts import trr0005_predict_confirmation as predict

    calls = {"count": 0}

    def predict_one(_record: int):
        calls["count"] += 1
        value = torch.full((contract.SEQUENCE_TOKENS,), contract.INVALID_TOKEN_ID, dtype=torch.long)
        value[0] = contract.BOS_TOKEN_ID
        value[1] = 1 if calls["count"] == 1 else 2
        return value

    with pytest.raises(predict.PredictionError, match="differ"):
        predict.run_warmed_prediction(
            method_id=contract.METHOD_IDS[2],
            records=list(range(contract.RECORDS_PER_DOMAIN)),
            predict_one=predict_one,
        )


def test_a2_prediction_descriptor_retains_counts_but_omits_arrays() -> None:
    from scripts import trr0005_predict_confirmation as predict

    predictions = torch.full(
        (contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS),
        contract.INVALID_TOKEN_ID,
        dtype=torch.long,
    )
    predictions[:, 0] = contract.BOS_TOKEN_ID
    predictions[:, 1] = 1
    timing = {
        "warmup_runs_per_record": 1,
        "measured_runs_per_record": 1,
        "warmup_output_exact_match_measured": True,
        "measured_output_selected": True,
    }
    descriptor = predict.prediction_descriptor(
        cell_id=contract.EXPECTED_CELL_IDS[0],
        method_id="frozen_a1_a2_k256",
        predictions=predictions,
        timing=timing,
        candidate_budget=256,
        public_prefix_calls=128,
        candidate_simulations=32768,
    )
    assert descriptor["candidate_arrays_present"] is False
    assert descriptor["candidate_output"] == "omitted_after_decision"
    assert descriptor["candidate_budget"] == 256
    assert descriptor["candidate_simulations"] == 32768
    assert "predictions" in descriptor


def test_canonical_state_path_matches_parallel_fit_layout() -> None:
    from scripts import trr0005_predict_confirmation as predict

    path = predict.canonical_state_path(
        Path("fit-output"),
        distribution="enriched",
        method_id="affine_causal_h_attention128",
    )
    assert path.as_posix() == "fit-output/enriched/affine_causal_h_attention128/selected.safetensors"


def test_predictor_rejects_interior_padding() -> None:
    from scripts import trr0005_predict_confirmation as predict

    value = torch.full((contract.SEQUENCE_TOKENS,), contract.INVALID_TOKEN_ID, dtype=torch.long)
    value[0] = contract.BOS_TOKEN_ID
    value[1] = 1
    value[3] = 2
    with pytest.raises(predict.PredictionError, match="non-contiguous"):
        predict._ids(value, description="synthetic")
