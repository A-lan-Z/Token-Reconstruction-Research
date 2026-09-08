"""CPU-only synthetic tests for the TRR-P10 execution boundary."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.trr_p10 import execution


RECORDS = 4
A1_RECORDS = 2
A1_INDICES = (0, 2)


def _write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return execution.file_record(path, base=path.parent, description=str(path))


def _binding(path: Path, *, base: Path) -> dict:
    return execution.file_record(path, base=base, description=str(path))


def _make_fixture(tmp_path: Path) -> dict:
    root = tmp_path / "repo"
    task = root / "experiments" / "TRR-P10"
    task.mkdir(parents=True)
    cells = list(execution.CELL_ORDER)
    methods = list(execution.METHOD_ORDER)
    records_by_cell = {cell: RECORDS for cell in cells}

    truth_by_cell: dict[str, torch.Tensor] = {}
    predictions: dict[str, dict] = {}
    output = task / "predictions"
    for cell_index, cell in enumerate(cells):
        truth = torch.full((RECORDS, execution.STORED_SEQUENCE_TOKENS), 1, dtype=torch.long)
        truth[:, 0] = execution.BOS_TOKEN_ID
        truth_by_cell[cell] = truth
        for method in methods:
            method_records = A1_RECORDS if method == "frozen_a1_a2_k256" else RECORDS
            values = truth.index_select(0, torch.tensor(A1_INDICES, dtype=torch.long)).clone() if method == "frozen_a1_a2_k256" else truth.clone()
            # Make one distinct category in each cell while retaining valid IDs.
            if method == "expanded_fixed":
                values[0, 1] = 2
            elif method == "current_fixed":
                values[0, 2] = 2
            elif method == "frozen_a1_a2_k256":
                values[0, 3] = 2
            path = output / cell.split("__", 1)[0] / cell.split("__", 1)[1] / f"{method}.safetensors"
            metadata = {
                "schema": "token-reconstruction.trr-p10-prediction.v1",
                "task_id": execution.TASK_ID,
                "method_id": method,
                "cell_id": cell,
                "records": str(method_records),
                "geometry_json": json.dumps(
                    {
                        "bos_token_id": execution.BOS_TOKEN_ID,
                        "records": method_records,
                        "scored_post_bos_tokens": execution.SCORED_POST_BOS_TOKENS,
                        "stored_sequence_tokens": execution.STORED_SEQUENCE_TOKENS,
                        "vocabulary_size": execution.VOCABULARY_SIZE,
                    },
                    sort_keys=True,
                ),
                "truth_opened": "false",
                "candidate_arrays_persisted": "false",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"predictions": values}, str(path), metadata=metadata)
            record = _binding(path, base=root)
            record["prediction_sha256"] = execution.tensor_digest(values)
            record["records"] = method_records
            if method == "frozen_a1_a2_k256":
                record["source_subset"] = {
                    "parent_cell": cell,
                    "indices": list(A1_INDICES),
                    "indices_sha256": execution.indices_digest(list(A1_INDICES)),
                    "records": A1_RECORDS,
                    "rule": "first_frozen_panel_order",
                }
            predictions[f"{method}::{cell}"] = record

    registration_path = task / "registration.json"
    registration_payload = {
        "schema": "token-reconstruction.trr-p10-registration.v1",
        "task_id": execution.TASK_ID,
        "status": "FROZEN_EVALUATION_REGISTRATION_BEFORE_TRUTH",
        "method_order": methods,
        "cell_order": cells,
        "records_by_domain": {"pile": RECORDS, "finance": RECORDS},
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }
    registration_record = _write_json(registration_path, registration_payload)

    run_path = task / "run_manifest.json"
    run_payload = {
        "schema": "token-reconstruction.trr-p10-prediction-run.v1",
        "task_id": execution.TASK_ID,
        "status": "PUBLIC_PREDICTIONS_COMPLETE_BEFORE_TRUTH",
        "predictions": predictions,
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }
    run_record = _write_json(run_path, run_payload)

    freeze_path = task / "public_freeze.json"
    freeze_payload = {
        "schema": "token-reconstruction.trr-p10-public-freeze.v1",
        "task_id": execution.TASK_ID,
        "status": "P10_PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH",
        "method_order": methods,
        "cell_order": cells,
        "records_by_domain": {"pile": RECORDS, "finance": RECORDS},
        "registration": registration_record,
        "run_manifest": run_record,
        "predictions": predictions,
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }
    _write_json(freeze_path, freeze_payload)
    freeze_record = _binding(freeze_path, base=root)

    truth_path = task / "truth.safetensors"
    truth_metadata = {
        "schema": "token-reconstruction.trr-p10-truth-sidecar.v1",
        "task_id": execution.TASK_ID,
        "truth_opened": "false",
    }
    save_file(
        {f"{cell}__token_ids": value for cell, value in truth_by_cell.items()},
        str(truth_path),
        metadata=truth_metadata,
    )
    truth_record = _binding(truth_path, base=root)
    descriptor_path = task / "truth_descriptor.json"
    descriptor_payload = {
        "schema": "token-reconstruction.trr-p10-truth-binding.v1",
        "task_id": execution.TASK_ID,
        "status": "TRR-P10_TRUTH_PREPARED_AFTER_PUBLIC_FREEZE",
        "prepared_after_public_freeze": True,
        "public_freeze": freeze_record,
        "registration": registration_record,
        "truth_payload": truth_record,
        "truth_tensor_keys": [f"{cell}__token_ids" for cell in cells],
        "records_by_domain": {"pile": RECORDS, "finance": RECORDS},
        "truth_opened": False,
    }
    _write_json(descriptor_path, descriptor_payload)
    return {
        "root": root,
        "freeze_path": freeze_path,
        "freeze_record": freeze_record,
        "registration_path": registration_path,
        "truth_path": truth_path,
        "descriptor_path": descriptor_path,
        "truth_by_cell": truth_by_cell,
    }


def test_validate_frozen_package_checks_hashes_and_matrix(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    frozen = execution.validate_frozen_package(
        fixture["freeze_path"],
        repository_root=fixture["root"],
        registration_path=fixture["registration_path"],
    )
    assert frozen.methods == tuple(execution.METHOD_ORDER)
    assert frozen.records_by_method_cell[f"frozen_a1_a2_k256::{execution.CELL_ORDER[0]}"] == A1_RECORDS
    assert frozen.subset_indices_by_method_cell[f"frozen_a1_a2_k256::{execution.CELL_ORDER[0]}"] == A1_INDICES
    assert set(frozen.predictions) == {
        f"{method}::{cell}"
        for method in execution.METHOD_ORDER
        for cell in execution.CELL_ORDER
    }
    prediction = next(iter(frozen.prediction_records.values()))["path"]
    Path(prediction).write_bytes(Path(prediction).read_bytes() + b"tamper")
    with pytest.raises(execution.ExecutionError, match="binding changed"):
        execution.validate_frozen_package(
            fixture["freeze_path"],
            repository_root=fixture["root"],
            registration_path=fixture["registration_path"],
        )


def test_explicit_registration_must_match_freeze_binding(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    alternate = fixture["root"] / "experiments" / "TRR-P10" / "alternate-registration.json"
    _write_json(alternate, {"schema": "wrong-registration", "truth_opened": False})
    with pytest.raises(execution.ExecutionError, match="frozen package/registration"):
        execution.validate_frozen_package(
            fixture["freeze_path"],
            repository_root=fixture["root"],
            registration_path=alternate,
        )


def test_truth_load_requires_matching_postfreeze_descriptor(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    frozen = execution.validate_frozen_package(fixture["freeze_path"], repository_root=fixture["root"])
    values, binding = execution.load_truth_after_freeze(
        frozen,
        fixture["descriptor_path"],
        truth_path=fixture["truth_path"],
        repository_root=fixture["root"],
    )
    assert set(values) == set(execution.CELL_ORDER)
    assert binding["status"] == "TRUTH_LOADED_AFTER_PUBLIC_FREEZE"
    assert all(tuple(value.shape) == (RECORDS, execution.STORED_SEQUENCE_TOKENS) for value in values.values())

    descriptor = json.loads(fixture["descriptor_path"].read_text(encoding="utf-8"))
    descriptor["public_freeze"]["sha256"] = "0" * 64
    fixture["descriptor_path"].write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(execution.ExecutionError, match="truth descriptor/freeze sha256"):
        execution.load_truth_after_freeze(frozen, fixture["descriptor_path"], repository_root=fixture["root"])


def test_error_inventory_keeps_pair_categories_and_no_rank_inference(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    frozen = execution.validate_frozen_package(fixture["freeze_path"], repository_root=fixture["root"])
    truth, truth_binding = execution.load_truth_after_freeze(
        frozen,
        fixture["descriptor_path"],
        repository_root=fixture["root"],
    )
    inventory = execution.build_error_inventory(frozen, truth, truth_binding=truth_binding)
    row = inventory["cells"][execution.CELL_ORDER[0]]
    pair = row["pairwise"]["expanded_fixed_vs_a1_a2"]["tokens"]
    assert pair["denominator"] == A1_RECORDS * execution.SCORED_POST_BOS_TOKENS
    assert row["pairwise"]["expanded_fixed_vs_current_fixed"]["tokens"]["denominator"] == RECORDS * execution.SCORED_POST_BOS_TOKENS
    assert row["method_summary"]["frozen_a1_a2_k256"]["records"] == A1_RECORDS
    assert row["pairwise"]["expanded_fixed_vs_a1_a2"]["alignment"]["mode"] == "right_method_parent_subset"
    assert pair["left_only_correct"] == 1
    assert pair["right_only_correct"] == 1
    assert row["rank_and_proposal_diagnostics"]["a1_proposal"]["status"] == "UNAVAILABLE"
    assert row["rank_and_proposal_diagnostics"]["a2_decoder_ranking"]["status"] == "UNAVAILABLE"
    assert inventory["decomposition_policy"]["final_prediction_mismatch_is_not_a_candidate_rank"] is True


def test_reduced_a1_subset_digest_is_fail_closed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    freeze = json.loads(fixture["freeze_path"].read_text(encoding="utf-8"))
    binding = freeze["predictions"]["frozen_a1_a2_k256::pile__public_base"]
    binding["source_subset"]["indices_sha256"] = "0" * 64
    fixture["freeze_path"].write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(execution.ExecutionError, match="source subset index digest"):
        execution.validate_frozen_package(fixture["freeze_path"], repository_root=fixture["root"])


def test_pretruth_flags_and_create_only_inventory_are_fail_closed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    freeze = json.loads(fixture["freeze_path"].read_text(encoding="utf-8"))
    freeze["truth_opened"] = True
    fixture["freeze_path"].write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(execution.ExecutionError, match="forbidden access"):
        execution.validate_frozen_package(fixture["freeze_path"], repository_root=fixture["root"])

    fixture = _make_fixture(tmp_path / "write")
    frozen = execution.validate_frozen_package(fixture["freeze_path"], repository_root=fixture["root"])
    truth, binding = execution.load_truth_after_freeze(frozen, fixture["descriptor_path"], repository_root=fixture["root"])
    inventory = execution.build_error_inventory(frozen, truth, truth_binding=binding)
    output = fixture["root"] / "experiments" / "TRR-P10" / "inventory.json"
    record = execution.write_inventory(output, inventory)
    assert record["sha256"] == execution.sha256_file(output)
    with pytest.raises(execution.ExecutionError, match="overwrite"):
        execution.write_inventory(output, inventory)
