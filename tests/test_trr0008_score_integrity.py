from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from scripts import trr0008_eval_contract as contract
from scripts import trr0008_eval_gate as gate
from scripts import trr0008_score as scorer


def _ids(records: int, *, token: int = 7) -> torch.Tensor:
    values = torch.full((records, contract.STORED_SEQUENCE_TOKENS), token, dtype=torch.long)
    values[:, 0] = contract.BOS_TOKEN_ID
    return values


def _file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _file_record(path)


def _decision_contract(timing_record: dict[str, Any]) -> dict[str, Any]:
    primary = contract.PRIMARY_METHOD_ID
    reference = contract.REFERENCE_METHOD_ID
    current = contract.CURRENT_RESIDUAL_METHOD_ID
    diagnostic = contract.IMPROVED_DIAGONAL_METHOD_ID
    cells = list(contract.CELL_ORDER)
    return {
        "schema": "token-reconstruction.trr0008-decision-contract.v1",
        "task_id": contract.TASK_ID,
        "status": "FROZEN_DECISION_CONTRACT_BEFORE_SOURCE_SELECTION",
        "methods": {
            "candidate": primary,
            "credible_alternative": current,
            "diagnostic": diagnostic,
            "reference": reference,
        },
        "primary": {
            "cell": "finance__public_base",
            "route_alpha": 0.025,
            "component_alpha": 0.0125,
            "practical_margin": 0.05,
        },
        "token_endpoint": {
            "route_alpha": 0.025,
            "practical_margin": 0.01,
        },
        "safeguards": {
            "route_alpha": 0.05,
            "exact_harm_margin": 0.05,
            "token_harm_margin": 0.01,
            "cells": cells,
            "primary_harm": {
                "required": True,
                "cell": "finance__public_base",
                "alpha": 0.05,
                "exact_lower_bound_minimum": -0.05,
                "token_lower_bound_minimum": -0.01,
            },
        },
        "confidence": {
            "primary_quality": {
                "route_alpha": 0.025,
                "exact_cp_component_alpha": 0.0125,
                "token_bootstrap_lower_tail_alpha": 0.025,
            },
            "safeguard": {
                "overall_alpha": 0.05,
                "exact_cp_component_alpha": 0.025,
                "token_bootstrap_lower_tail_alpha": 0.05,
            },
        },
        "advance_rule": {"cost_threshold": 1.25},
        "bootstrap": {"seed": 8008, "draws": 8, "unit": "source_record"},
        "cost_gate": {
            "threshold": 1.25,
            "primary_cell": "finance__public_base",
            "cells": cells,
            "all_cells_required": True,
            "timing_receipt": {
                **timing_record,
                "schema": "token-reconstruction.trr0008-balanced-timing.v1",
                "status": "TIMING_COMPLETE",
                "qualification": "PASS",
                "truth_opened": False,
            },
        },
    }


def _build_fixture(tmp_path: Path) -> dict[str, Any]:
    records = 2
    observation_file = tmp_path / "public_observation.bin"
    observation_file.write_bytes(b"synthetic-public-observation")
    observation_record = _file_record(observation_file)
    digest_by_domain = {domain: f"{domain[0]}" * 64 for domain in contract.DOMAIN_ORDER}
    cells = []
    for cell_id in contract.CELL_ORDER:
        domain = cell_id.split("__", 1)[0]
        cells.append(
            {
                "cell_id": cell_id,
                "records": records,
                "record_ids_sha256": digest_by_domain[domain],
                "observation": observation_record
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
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "truth_opened": False,
        "target_labels_loaded": False,
        "source_text_loaded": False,
        "source_text_written": False,
        "candidate_arrays_persisted": False,
        "records_by_domain": {domain: records for domain in contract.DOMAIN_ORDER},
        "cell_order": list(contract.CELL_ORDER),
        "cells": cells,
    }
    observation_path = tmp_path / "observations.json"
    observation_record = _write_json(observation_path, observations)

    fake_state_path = tmp_path / "frozen_state.bin"
    fake_state_path.write_bytes(b"synthetic-frozen-state")
    fake_state = _file_record(fake_state_path)
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
    alias = dict(methods[-1]) | {
        "id": contract.TIMING_CONTROL_METHOD_ID,
        "loader": dict(contract.POSITIONWISE_LOADER)
        | {
            "kwargs": dict(contract.POSITIONWISE_LOADER["kwargs"])
            | {"method_id": contract.METHOD_MODEL_IDS[contract.TIMING_CONTROL_METHOD_ID]}
        },
    }

    timing_path = tmp_path / "timing.json"
    timing = {
        "schema": "token-reconstruction.trr0008-balanced-timing.v1",
        "task_id": contract.TASK_ID,
        "status": "TIMING_COMPLETE",
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "candidate_arrays_persisted": False,
        "configuration": {"blocks": 40, "threshold": 1.25},
        "equivalence": {"status": "PASS"},
        "summary": {
            "qualification": {
                "decision": "PASS",
                "measurement_valid": True,
                "threshold": 1.25,
                "per_cell": {
                    cell: {"decision": "PASS", "ci_upper": 1.0}
                    for cell in contract.CELL_ORDER
                },
            }
        },
    }
    _write_json(timing_path, timing)
    timing_record = _file_record(timing_path)

    output_root = tmp_path / "predictions"
    registration = {
        "schema": contract.REGISTRATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_METHOD_AND_INPUT_BINDING_NO_TRUTH",
        "repository_root": str(tmp_path),
        "code_commit": "a" * 40,
        "method_order": list(contract.METHOD_ORDER),
        "method_ids": list(contract.METHOD_ORDER),
        "cell_order": list(contract.CELL_ORDER),
        "records_by_domain": {domain: records for domain in contract.DOMAIN_ORDER},
        "geometry": dict(contract.STATIC_GEOMETRY),
        "methods": methods,
        "timing_control": alias,
        "timing_receipt": timing_record,
        "observation_manifest": observation_record,
        "runtime_assets": {"normalized_public_E": fake_state},
        "output_root": str(output_root),
        "numerical_settings": dict(contract.NUMERICAL_SETTINGS),
        "resource_guard": dict(contract.RESOURCE_GUARD),
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "candidate_arrays_persisted": False,
    }
    contract.validate_registration(registration)
    registration_path = tmp_path / "registration.json"
    registration_record = _write_json(registration_path, registration)
    registration_sha = registration_record["sha256"]

    predictions = _ids(records)
    run_predictions: dict[str, Any] = {}
    run_timings: dict[str, Any] = {}
    for method_id in contract.METHOD_ORDER:
        for cell_id in contract.CELL_ORDER:
            path = contract.expected_prediction_path(output_root, cell_id=cell_id, method_id=method_id)
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
                    "geometry_json": json.dumps(
                        {"records": records, **contract.STATIC_GEOMETRY}, sort_keys=True
                    ),
                    "truth_opened": "false",
                    "candidate_arrays_persisted": "false",
                },
            )
            artifact = _file_record(path)
            timing_row = {
                "schema": contract.TIMING_SCHEMA,
                "task_id": contract.TASK_ID,
                "method_id": method_id,
                "cell_id": cell_id,
                "records": records,
                "truth_opened": False,
                "warmup_output_exact_match_measured": True,
                "prediction_artifact": artifact,
            }
            _write_json(path.with_suffix(".run.json"), timing_row)
            key = f"{method_id}::{cell_id}"
            run_predictions[key] = artifact | {
                "prediction_sha256": contract.tensor_digest(predictions),
                "records": records,
            }
            run_timings[key] = timing_row

    run_path = tmp_path / "run_manifest.json"
    run_manifest = {
        "schema": contract.RUN_SCHEMA,
        "task_id": contract.TASK_ID,
        "registration": registration_record,
        "code_commit": registration["code_commit"],
        "observation_manifest": observation_record,
        "predictions": run_predictions,
        "timings": run_timings,
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }
    run_record = _write_json(run_path, run_manifest)

    timing_record = _file_record(timing_path)
    public = gate.validate_public_outputs(
        registration_path=registration_path,
        repository_root=tmp_path,
        run_manifest_path=run_path,
        timing_receipt_path=timing_path,
    )
    freeze_path = tmp_path / "freeze_receipt.json"
    freeze_receipt = dict(public)
    freeze_record = _write_json(freeze_path, freeze_receipt)

    source_selection_path = tmp_path / "selection_metadata.json"
    source_selection_record = _write_json(source_selection_path, {"source_free": True})

    truth_path = tmp_path.parent / f"{tmp_path.name}-truth.safetensors"
    truth = {f"{domain}__token_ids": predictions.clone() for domain in contract.DOMAIN_ORDER}
    save_file(
        truth,
        str(truth_path),
        metadata={
            "schema": "token-reconstruction.trr0008-truth-sidecar.v1",
            "task_id": contract.TASK_ID,
            "truth_opened": "false",
            "registration_sha256": registration_sha,
            "source_selection_sha256": source_selection_record["sha256"],
            "observation_record_ids_sha256": json.dumps(
                digest_by_domain, sort_keys=True, separators=(",", ":")
            ),
            "records_by_domain": json.dumps(
                {domain: records for domain in contract.DOMAIN_ORDER},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "sequence_tokens": str(contract.STORED_SEQUENCE_TOKENS),
            "scored_post_bos_tokens": str(contract.SCORED_POST_BOS_TOKENS),
            "labels_shared_across_target_conditions": "true",
            "source_text_loaded_for_label_materialization": "true",
            "target_model_or_target_labels_loaded": "false",
        },
    )
    sidecar_record = _file_record(truth_path)
    truth_binding_path = tmp_path / "truth_binding.json"
    truth_binding = {
        "schema": "token-reconstruction.trr0008-truth-binding.v1",
        "task_id": contract.TASK_ID,
        "status": "TRR0008_TRUTH_PREPARED_AFTER_PUBLIC_GATE",
        "truth_opened": False,
        "prepared_after_public_gate": True,
        "registration": registration_record,
        "receipt": freeze_record,
        "source_selection": source_selection_record,
        "observation_manifest": observation_record,
        "sidecar": sidecar_record,
        "records_by_domain": {domain: records for domain in contract.DOMAIN_ORDER},
        "sequence_tokens_including_bos": contract.STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
        "cell_order": list(contract.CELL_ORDER),
        "target_conditions": list(contract.TARGET_ORDER),
        "labels_shared_across_target_conditions": True,
        "cells": [
            {
                "cell_id": cell_id,
                "records": records,
                "record_ids_sha256": digest_by_domain[cell_id.split("__", 1)[0]],
            }
            for cell_id in contract.CELL_ORDER
        ],
    }
    truth_binding_record = _write_json(truth_binding_path, truth_binding)

    decision_contract_path = tmp_path / "decision_contract.json"
    decision_record = _write_json(
        decision_contract_path,
        _decision_contract(timing_record),
    )
    # The synthetic header is checked once before the scorer tests. This call
    # reads metadata only; the private sidecar is opened only by scorer.main.
    pretruth = gate.validate_before_truth(
        receipt_path=freeze_path,
        registration_path=registration_path,
        repository_root=tmp_path,
        truth_binding_path=truth_binding_path,
    )
    assert pretruth["truth_opened"] is False
    return {
        "root": tmp_path,
        "registration": registration_path,
        "run": run_path,
        "freeze": freeze_path,
        "binding": truth_binding_path,
        "truth": truth_path,
        "output_root": output_root,
        "result": tmp_path / "score.json",
        "decision": decision_contract_path,
        "timing": timing_path,
        "timing_record": timing_record,
        "sidecar_record": sidecar_record,
        "decision_record": decision_record,
        "truth_binding_record": truth_binding_record,
        "freeze_record": freeze_record,
        "run_record": run_record,
    }


def _argv(fixture: dict[str, Any], *, predictions_root: Path | None = None, result: Path | None = None) -> list[str]:
    return [
        "--repository-root",
        str(fixture["root"]),
        "--predictions-root",
        str(predictions_root or fixture["output_root"]),
        "--truth",
        str(fixture["truth"]),
        "--registration",
        str(fixture["registration"]),
        "--run-manifest",
        str(fixture["run"]),
        "--freeze-receipt",
        str(fixture["freeze"]),
        "--truth-binding",
        str(fixture["binding"]),
        "--result",
        str(result or fixture["result"]),
        "--decision-contract",
        str(fixture["decision"]),
    ]


def _assert_rejected_before_truth(
    monkeypatch: pytest.MonkeyPatch, fixture: dict[str, Any], argv: list[str]
) -> None:
    truth_path = Path(fixture["truth"]).resolve()
    opened: list[str] = []
    original = scorer.safe_open

    def sentinel(path: str, *args: Any, **kwargs: Any):
        if Path(path).expanduser().resolve() == truth_path:
            opened.append(str(path))
            raise AssertionError("private truth sidecar was opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(scorer, "safe_open", sentinel)
    assert scorer.main(argv) == 2
    assert opened == []


def test_scorer_cli_serializes_complete_matrix_and_binds_timing_truth(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    assert scorer.main(_argv(fixture)) == 0
    result = json.loads(Path(fixture["result"]).read_text(encoding="utf-8"))
    assert result["schema"] == contract.SCORE_SCHEMA
    assert result["truth_opened"] is True
    assert result["timing_receipt"] == fixture["timing_record"]
    assert result["decision"]["cost_status"] == "COST_PASS"
    assert result["decision_contract"]["sha256"] == fixture["decision_record"]["sha256"]
    assert fixture["sidecar_record"]["sha256"] == json.loads(
        Path(fixture["binding"]).read_text(encoding="utf-8")
    )["sidecar"]["sha256"]


def test_missing_prediction_is_rejected_before_private_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    missing = contract.expected_prediction_path(
        fixture["output_root"],
        cell_id=contract.CELL_ORDER[0],
        method_id=contract.METHOD_ORDER[0],
    )
    missing.unlink()
    _assert_rejected_before_truth(monkeypatch, fixture, _argv(fixture, result=tmp_path / "missing.json"))


def test_corrupted_prediction_is_rejected_before_private_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    corrupt = contract.expected_prediction_path(
        fixture["output_root"],
        cell_id=contract.CELL_ORDER[0],
        method_id=contract.METHOD_ORDER[0],
    )
    save_file({"predictions": _ids(2, token=8)}, str(corrupt), metadata={"tampered": "true"})
    _assert_rejected_before_truth(monkeypatch, fixture, _argv(fixture, result=tmp_path / "corrupt.json"))


def test_different_prediction_root_is_rejected_before_private_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    alternate_root = tmp_path / "alternate-predictions"
    _assert_rejected_before_truth(
        monkeypatch,
        fixture,
        _argv(fixture, predictions_root=alternate_root, result=tmp_path / "alternate.json"),
    )


def test_altered_frozen_timing_binding_is_rejected_before_private_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    freeze_path = Path(fixture["freeze"])
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["timing_receipt"] = dict(freeze["timing_receipt"]) | {"sha256": "0" * 64}
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _assert_rejected_before_truth(monkeypatch, fixture, _argv(fixture, result=tmp_path / "altered.json"))
