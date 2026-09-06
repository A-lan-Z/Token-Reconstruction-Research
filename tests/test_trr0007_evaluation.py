from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pytest
from safetensors.torch import save_file
import torch

from scripts import trr0007_bank_ledger as bank_ledger
from scripts import trr0007_eval_contract as contract
from scripts import trr0007_eval_gate as gate
from scripts import trr0007_eval_register as register
from scripts import trr0007_eval_runner as runner
from scripts import trr0007_score as scorer


def _struct(path: str = "asset.bin", *, digest: str = "a" * 64) -> dict[str, object]:
    return {"path": path, "bytes": 1, "sha256": digest}


def _registration() -> dict[str, object]:
    methods: list[dict[str, object]] = [
        {
            "id": contract.REFERENCE_METHOD_ID,
            "role": "reference",
            "kind": "decoder",
            "support": "current_enriched",
            "capacity": "trained_diagonal",
            "cells": list(contract.CELL_ORDER),
            "records_per_cell": 128,
            "candidate_policy": "forbidden",
            "state": {
                "path": contract.REFERENCE_STATE_PATH,
                "bytes": contract.REFERENCE_STATE_BYTES,
                "sha256": contract.REFERENCE_STATE_SHA256,
            },
            "loader": {
                "module": "token_reconstruction.trr0005_joint_decoder",
                "function": "load_decoder_state",
                "kwargs": {
                    "method_id": "affine_trained_diagonal_attention128",
                    "hidden_size": 2048,
                    "vocabulary_size": 128256,
                    "context_width": 128,
                },
            },
        }
    ]
    for method in contract.STUDENT_METHOD_IDS:
        methods.append(
            {
                "id": method,
                "role": "student",
                "kind": "decoder",
                "support": contract.STUDENT_SUPPORT[method],
                "capacity": contract.STUDENT_CAPACITY[method],
                "cells": list(contract.CELL_ORDER),
                "records_per_cell": 128,
                "candidate_policy": "forbidden",
                "state": _struct(method + ".safetensors"),
                "method_freeze_sha256": "d" * 64,
                "loader": {
                    "module": "token_reconstruction.trr0007_positionwise",
                    "function": "load_positionwise_model_state",
                    "kwargs": {
                        "method_id": contract.STUDENT_METHOD_MODEL_IDS[method],
                        "hidden_size": 2048,
                        "vocabulary_size": 128256,
                        "context_width": 128,
                    },
                },
            }
        )
    methods.append(
        {
            "id": contract.ANCHOR_METHOD_ID,
            "role": "anchor",
            "kind": "a1_a2",
            "cells": list(contract.BASE_CELL_ORDER),
            "records_per_cell": 32,
            "proposal_budget": 512,
            "candidate_budget": 256,
            "candidate_policy": "output_only",
            "adapter": {
                "kind": "legacy_trr0003_a1_a2_p0",
                "selection_policy": "fixed_k256_direct_cosine",
                "proposal_max_k": 512,
                "proposal_chunk": 256,
            },
        }
    )
    return {
        "schema": contract.REGISTRATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_EVALUATION_REGISTRATION",
        "records_per_domain": 128,
        "anchor_records_per_domain": 32,
        "cell_order": list(contract.CELL_ORDER),
        "method_ids": list(contract.METHOD_ORDER),
        "plan_sha256": "b" * 64,
        "plan": _struct("plan.json", digest="b" * 64),
        "code_commit": "c" * 40,
        "truth_opened": False,
        "truth_created": False,
        "candidate_arrays_persisted": False,
        "geometry": {
            "capture_batch_records": 8,
            "capture_sequence_tokens": 192,
            "stored_sequence_tokens": 128,
            "scored_sequence_tokens": 128,
            "scored_post_bos_tokens": 127,
            "hidden_size": 2048,
            "vocabulary_size": 128256,
            "chunk_records": 8,
        },
        "methods": methods,
        "runtime_assets": {
            "normalized_public_E": {
                "path": contract.PUBLIC_E_PATH,
                "bytes": contract.PUBLIC_E_BYTES,
                "sha256": contract.PUBLIC_E_SHA256,
                "shape": [128256, 2048],
                "dtype": "torch.float32",
            },
            "a1_a2": {
                "public_model_snapshot": {"path": "/tmp/model-snapshot"},
                "lens": _struct("lens.pt"),
                "reference": _struct("reference.py"),
            },
        },
        "observation_manifest": _struct("observations.json", digest="d" * 64),
        "capture_receipt": _struct("capture.json", digest="e" * 64),
        "method_freeze": _struct("method_freeze.json", digest="d" * 64),
        "method_freeze_state_sha256": {method: "a" * 64 for method in contract.STUDENT_METHOD_IDS},
        "final_bank_ledgers": {
            "schema": "token-reconstruction.trr0007-final-bank-ledger.v1",
            "task_id": contract.TASK_ID,
            "status": "METADATA_BANK_READY_FOR_REAL_P0_CAPTURE",
            "files": {
                "exclusion_manifest": _struct("bank-exclusions.json"),
                "selected_parent_rows": _struct("bank-parents.json"),
                "corpus_plan": _struct("bank-plan.json"),
            },
            "exclusion_set_counts": {"record_ids": 2449, "source_row_keys": 1848, "opaque_sequence_or_reservation_digests": 4073},
            "selected_parent_rows": {"rows": 120, "rows_by_domain": {"controlled_pile_context": 60, "controlled_finance_context": 60}},
            "source_and_sequence_ledgers_verified": True,
        },
        "frequency_reference": _struct("frequency_references_v1.json", digest="g" * 64),
        "source_selection": _struct("selection.json", digest="e" * 64),
        "exclusion_manifest": _struct("exclusions.json", digest="f" * 64),
        "output_root": "experiments/TRR-0007/evaluation/predictions",
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
            "maximum_seconds": 1800,
        },
        "numerical_settings": dict(contract.NUMERICAL_SETTINGS),
        "code_bindings": [
            {"role": role, "path": path, "bytes": 1, "sha256": "1" * 64}
            for role, path in contract.CODE_BINDING_SPECS
        ],
        "source_text_or_target_labels": False,
    }


def test_plan_is_complete_draft_with_correct_denominators() -> None:
    plan = contract.load_json(Path("experiments/TRR-0007/evaluation_plan.json"), description="plan")
    parsed = contract.validate_plan(plan)
    assert parsed["status"] in {
        "DRAFT_PENDING_ROOT_REVIEW",
        "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION",
    }
    assert parsed["a1_a2_anchor"]["denominator"]["total_token_positions"] == 8128
    assert parsed["uncertainty_and_multiplicity"]["bootstrap_draws"] == 50000
    assert "0.05/64" in parsed["uncertainty_and_multiplicity"]["token_tail_alpha"]


def test_plan_rejects_wrong_anchor_target_or_count() -> None:
    plan = contract.load_json(Path("experiments/TRR-0007/evaluation_plan.json"), description="plan")
    plan["a1_a2_anchor"]["target_conditions"] = ["public_lora_2601"]
    with pytest.raises(contract.ContractError, match="anchor target"):
        contract.validate_plan(plan)


def test_registration_rejects_candidate_policy_change() -> None:
    value = _registration()
    value["methods"][-1]["candidate_policy"] = "required"
    with pytest.raises(contract.ContractError, match="output policy"):
        contract.validate_registration(value)


def test_registration_rejects_student_loader_outside_allowlist() -> None:
    value = _registration()
    value["methods"][1]["loader"]["module"] = "evil.module"
    with pytest.raises(contract.ContractError, match="allowlisted"):
        contract.validate_registration(value)


def test_normalize_prediction_sets_bos_and_suffix_padding() -> None:
    raw = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    # Normalize is intentionally fixed at 128 positions; test the generic
    # prediction validator independently on the compact fixture.
    out = torch.full((1, 128), contract.INVALID_TOKEN_ID, dtype=torch.long)
    out[0, 0] = contract.BOS_TOKEN_ID
    out[0, 1] = 4
    assert contract.validate_prediction_tensor(out, records=1).shape == (1, 128)
    assert contract.normalize_prediction(
        torch.cat((raw, torch.zeros(124, dtype=torch.long))),
        torch.tensor([1, 1, 0, 0] + [0] * 124, dtype=torch.bool),
    )[0].item() == contract.BOS_TOKEN_ID


def test_prediction_validator_rejects_non_suffix_invalid() -> None:
    value = torch.full((1, 128), contract.INVALID_TOKEN_ID, dtype=torch.long)
    value[0, 0] = contract.BOS_TOKEN_ID
    value[0, 2] = 4
    with pytest.raises(contract.ContractError, match="suffix"):
        contract.validate_prediction_tensor(value, records=1)


def test_observation_chunks_reject_token_payload_and_bad_positions(tmp_path: Path) -> None:
    activations = torch.zeros((8, 128, 2048), dtype=torch.bfloat16)
    mask = torch.ones((8, 128), dtype=torch.uint8)
    positions = torch.arange(128, dtype=torch.long).repeat(8, 1)
    path = tmp_path / "obs.safetensors"
    save_file(
        {"activations": activations, "attention_mask": mask, "position_ids": positions},
        str(path),
    )
    cell = {"cell_id": "pile__public_base", "observation": {"path": str(path)}}
    chunks = list(runner._iter_observation_chunks(cell, records=8, chunk_records=8))
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 8)]
    token_path = tmp_path / "token.safetensors"
    save_file(
        {
            "activations": activations,
            "attention_mask": mask,
            "position_ids": positions,
            "token_ids": torch.zeros((8, 128), dtype=torch.int32),
        },
        str(token_path),
    )
    with pytest.raises(runner.RunnerError, match="tensor keys"):
        list(runner._iter_observation_chunks(
            {"cell_id": "pile__public_base", "observation": {"path": str(token_path)}},
            records=8,
            chunk_records=8,
        ))


def test_score_metrics_use_127_post_bos_positions() -> None:
    truth = torch.full((2, 128), 9, dtype=torch.long)
    truth[:, 0] = contract.BOS_TOKEN_ID
    pred = truth.clone()
    pred[0, 1] = 8
    pred[1, 127] = 8
    metrics = scorer._score_predictions(pred, truth)
    assert metrics["token_positions"] == 254
    assert metrics["token_errors"] == 2
    assert metrics["exact_records"] == 0
    assert metrics["exact_definition"] == "all 127 post-BOS positions; BOS is a fixed known diagnostic"
    assert metrics["first_error_position"] == [1, 127]
    bos_only = truth.clone()
    bos_only[0, 0] = 123
    bos_metrics = scorer._score_predictions(bos_only, truth)
    assert bos_metrics["exact_records"] == 2
    assert bos_metrics["bos_fixed_records"] == 1


def test_exact_bounds_keep_nonzero_upper_bound_for_zero_gains() -> None:
    result = scorer.clopper_pearson_gain_loss(0, 0, 32)
    assert result["lower"] < 0.0
    assert result["upper"] > 0.0


def test_harm_direction_uses_upper_for_evidence_and_lower_for_exclusion() -> None:
    contrast = {
        "token": {"point_pp": -2.0, "primary_lower": -0.015, "primary_upper": -0.02},
        "exact": {"point_pp": -6.0, "lower": -0.06, "upper": -0.06},
    }
    decision = scorer._contrast_decision(contrast)
    assert decision["token_material_harm_evidenced"] is True
    assert decision["token_harm_excluded"] is False
    assert decision["exact_material_harm_evidenced"] is True
    assert decision["exact_harm_excluded"] is False
    safe = {
        "token": {"point_pp": -0.2, "primary_lower": -0.005, "primary_upper": 0.002},
        "exact": {"point_pp": -0.2, "lower": -0.005, "upper": 0.002},
    }
    safe_decision = scorer._contrast_decision(safe)
    assert safe_decision["token_material_harm_evidenced"] is False
    assert safe_decision["token_harm_excluded"] is True


def test_gate_truth_header_does_not_require_sidecar_to_exist(tmp_path: Path) -> None:
    binding = {
        "schema": "token-reconstruction.trr0007-truth-binding.v1",
        "task_id": contract.TASK_ID,
        "truth_opened": False,
        "sidecar": {"path": str(tmp_path / "private.safetensors"), "bytes": 7, "sha256": "a" * 64},
        "cells": [
            {"cell_id": cell, "records": 128, "record_ids_sha256": "b" * 64}
            for cell in contract.CELL_ORDER
        ],
    }
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding))
    observations = {
        "cells": {
            cell: {"record_ids_sha256": "b" * 64}
            for cell in contract.CELL_ORDER
        }
    }
    result = gate._validate_truth_header(
        path,
        repository_root=tmp_path / "repo",
        output_root=tmp_path / "repo" / "out",
        observations=observations,
    )
    assert result["truth_opened"] is False
    assert not (tmp_path / "private.safetensors").exists()


def _metric(token_accuracy: float, exact_records: int = 0, records: int = 32) -> dict[str, object]:
    exact = np.zeros(records, dtype=np.bool_)
    exact[:exact_records] = True
    return {
        "per_record_token_accuracy": np.full(records, token_accuracy, dtype=np.float64),
        "per_record_exact": exact,
    }


def test_factorial_builder_uses_exact_four_primary_edges() -> None:
    values = {
        contract.REFERENCE_METHOD_ID: _metric(0.45, 2),
        "current_enriched__trained_diagonal": _metric(0.50, 4),
        "current_enriched__residual_mlp512": _metric(0.60, 8),
        "improved_public_bank__trained_diagonal": _metric(0.70, 12),
        "improved_public_bank__residual_mlp512": _metric(0.80, 16),
    }
    cell_results = {method: {"pile__public_base": metric} for method, metric in values.items()}
    result = scorer._build_factorial_contrasts(
        cell_results, "pile__public_base", draws=128, seed=5007
    )
    primary = [name for name, _, _ in scorer.PRIMARY_FACTORIAL_EDGES]
    assert list(result)[:4] == primary
    assert set(primary).isdisjoint({"interaction_detail", "improved_residual_vs_reference_endpoint"})
    assert all(result[name]["scope"] == "primary direct factorial edge" for name in primary)
    assert all("primary_lower" in result[name]["contrast"]["token"] for name in primary)
    assert result["support_at_trained_diagonal"]["contrast"]["token"]["point_pp"] == pytest.approx(20.0)
    assert result["support_at_residual_mlp512"]["contrast"]["token"]["point_pp"] == pytest.approx(20.0)
    assert result["capacity_on_current_enriched"]["contrast"]["token"]["point_pp"] == pytest.approx(10.0)
    assert result["capacity_on_improved_public_bank"]["contrast"]["token"]["point_pp"] == pytest.approx(10.0)
    assert "primary_lower" not in result["interaction_detail"]["contrast"]["token"]
    assert result["improved_residual_vs_reference_endpoint"]["contrast"]["exact"]["interval_scope"].startswith("descriptive")


def test_anchor_builder_covers_five_decoders_with_explicit_32_record_gaps() -> None:
    a1 = _metric(0.30, 1)
    a2 = _metric(0.40, 2)
    decoder_first32 = {method: _metric(0.50, 3) for method in scorer.DECODER_METHODS}
    result = scorer._build_anchor_comparisons(
        a1, a2, decoder_first32, draws=128, seed=5007
    )
    assert set(result["paired_student_vs_a2"]) == set(scorer.DECODER_METHODS)
    for row in result["paired_student_vs_a2"].values():
        assert row["records_per_domain"] == 32
        assert row["post_bos_token_denominator"] == 32 * 127
        assert row["exact_record_denominator"] == 32
        assert row["contrast"]["token"]["records"] == 32
        assert "primary_lower" not in row["contrast"]["token"]
    assert result["a2_minus_a1"]["exact_record_denominator"] == 32



def test_synthetic_registration_binds_path_states_and_gate_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the registration-to-gate receipt chain with tiny public files."""

    repo = tmp_path / "repo"
    repo.mkdir()
    task_root = repo / "experiments" / contract.TASK_ID
    task_root.mkdir(parents=True)

    def write_bytes(path: Path, value: bytes = b"x") -> dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": contract.sha256_file(path),
        }

    def write_json(path: Path, value: dict[str, object]) -> dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return write_bytes(path, path.read_bytes())

    plan_path = task_root / "evaluation_plan.json"
    plan_record = write_json(
        plan_path,
        {
            "schema": contract.PLAN_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION",
        },
    )
    monkeypatch.setattr(contract, "validate_plan", lambda value: dict(value))

    method_states: dict[str, dict[str, object]] = {}
    state_bindings: dict[str, dict[str, object]] = {}
    for index, method_id in enumerate(contract.STUDENT_METHOD_IDS):
        state_path = task_root / "states" / f"state_{index}.bin"
        state_record = write_bytes(state_path, f"state-{index}".encode())
        method_states[method_id] = state_record
        state_bindings[method_id] = {"state": state_record, "state_sha256": state_record["sha256"]}
    method_freeze_path = task_root / "method_freeze.json"
    method_freeze_payload = {
        "schema": contract.METHOD_FREEZE_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": contract.METHOD_FREEZE_STATUS,
        "method_ids": list(contract.STUDENT_METHOD_IDS),
        "state_bindings": state_bindings,
        "truth_opened": False,
        "fresh_evaluation_started": False,
        "source_accessed": False,
        "target_loaded": False,
        "target_labels_loaded": False,
        "private_or_truth_payload_read": False,
    }
    method_freeze_record = write_json(method_freeze_path, method_freeze_payload)

    frequency_path = repo / contract.FREQUENCY_REFERENCE_PATH
    frequency_record = write_json(
        frequency_path,
        {
            "schema": contract.FREQUENCY_REFERENCE_SCHEMA,
            "task_id": "TRR-0005",
            "status": "PUBLIC_FITTING_FREQUENCY_REFERENCES",
            "frequency_references": {"enriched": {"1": 3}, "original": {"1": 2}},
        },
    )

    source_path = task_root / "source_selection.json"
    exclusion_path = task_root / "source_exclusions.json"
    observation_path = task_root / "observations.json"
    capture_path = task_root / "capture.json"
    exclusion_payload = {
        "schema": "synthetic-exclusion",
        "task_id": contract.TASK_ID,
        "status": "PUBLIC_IDENTITY_EXCLUSIONS_COMPLETE_NO_TRUTH",
    }
    exclusion_record = write_json(exclusion_path, exclusion_payload)
    fake_bank = {
        "schema": "token-reconstruction.trr0007-final-bank-ledger.v1",
        "task_id": contract.TASK_ID,
        "status": "METADATA_BANK_READY_FOR_REAL_P0_CAPTURE",
        "files": {
            "exclusion_manifest": exclusion_record,
            "selected_parent_rows": exclusion_record,
            "corpus_plan": exclusion_record,
        },
        "exclusion_set_counts": {"record_ids": 2449, "source_row_keys": 1848, "opaque_sequence_or_reservation_digests": 4073},
        "selected_parent_rows": {"rows": 120, "rows_by_domain": {"controlled_pile_context": 60, "controlled_finance_context": 60}},
        "source_and_sequence_ledgers_verified": True,
    }
    fake_prefix = {
        "schema": bank_ledger.PREFIX_LEDGER_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": bank_ledger.PREFIX_LEDGER_STATUS,
        "file": exclusion_record,
        "counts": dict(bank_ledger.EXPECTED_PREFIX_COUNTS),
        "sequence_convention": {
            "hash_key": "final_sequence_sha256",
            "hash_algorithm": "SHA-256",
            "prefix_tokens_including_bos": 128,
            "active_rows_only": True,
        },
        "collector_styles": ["pile", "finance"],
        "all_bank_all_style_union": True,
    }
    source_payload = {
        "schema": contract.SOURCE_SELECTION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": contract.SOURCE_SELECTION_STATUS,
        "records_per_domain": contract.RECORDS_PER_DOMAIN,
        "method_freeze": method_freeze_record,
        "method_freeze_sha256": method_freeze_record["sha256"],
        "method_freeze_state_sha256": {method: state["sha256"] for method, state in method_states.items()},
        "final_bank_ledgers": fake_bank,
        "public_fitting_prefix_exclusions": fake_prefix,
        "selection_exclusions": exclusion_record,
        "truth_opened": False,
    }
    source_record = write_json(source_path, source_payload)
    observation_payload = {
        "schema": "synthetic-observation",
        "task_id": contract.TASK_ID,
        "method_freeze_sha256": method_freeze_record["sha256"],
        "selection_plan": source_record,
        "truth_opened": False,
    }
    observation_record = write_json(observation_path, observation_payload)
    capture_payload = {
        "schema": contract.CAPTURE_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": contract.CAPTURE_STATUS,
        "method_freeze_sha256": method_freeze_record["sha256"],
        "selection_plan": source_record,
        "observations": observation_record,
        "execution": {
            "code_commit": "c" * 40,
            "device": "cpu",
            "model_loaded_by_producer": True,
            "network_used": False,
            "started_utc": "2026-09-06T00:00:00Z",
            "ended_utc": "2026-09-06T00:00:01Z",
            "truth_opened": False,
            "target_labels_loaded": False,
            "source_text_written": False,
            "token_ids_written": False,
        },
    }
    capture_record = write_json(capture_path, capture_payload)

    reference_state_path = repo / "reference.bin"
    public_e_path = repo / "public_E.bin"
    lens_path = repo / "lens.bin"
    prefix_path = repo / "prefix.py"
    reference_state = write_bytes(reference_state_path)
    public_e = write_bytes(public_e_path)
    lens = write_bytes(lens_path)
    prefix = write_bytes(prefix_path, b"prefix")
    snapshot = repo / "model_snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(contract, "REFERENCE_STATE_PATH", str(reference_state_path.resolve()))
    monkeypatch.setattr(contract, "REFERENCE_STATE_BYTES", reference_state["bytes"])
    monkeypatch.setattr(contract, "REFERENCE_STATE_SHA256", reference_state["sha256"])
    monkeypatch.setattr(contract, "PUBLIC_E_PATH", str(public_e_path.resolve()))
    monkeypatch.setattr(contract, "PUBLIC_E_BYTES", public_e["bytes"])
    monkeypatch.setattr(contract, "PUBLIC_E_SHA256", public_e["sha256"])
    monkeypatch.setattr(register, "_git_head", lambda root: "c" * 40)
    monkeypatch.setattr(
        register,
        "_code_bindings",
        lambda root: [
            {"role": role, "path": path, "bytes": 1, "sha256": "1" * 64}
            for role, path in contract.CODE_BINDING_SPECS
        ],
    )
    monkeypatch.setattr(contract, "validate_registration", lambda value: dict(value))
    monkeypatch.setattr(contract, "load_observation_manifest", lambda *args, **kwargs: ({}, {}, {}))
    monkeypatch.setattr(register.bank_ledger, "load_final_bank_ledgers", lambda **kwargs: fake_bank)
    monkeypatch.setattr(gate.bank_ledger, "load_final_bank_ledgers", lambda **kwargs: fake_bank)
    monkeypatch.setattr(register.bank_ledger, "load_prefix_exclusion_ledger", lambda **kwargs: fake_prefix)
    monkeypatch.setattr(gate.bank_ledger, "load_prefix_exclusion_ledger", lambda **kwargs: fake_prefix)

    registration = register.build_registration(
        repository_root=repo,
        plan_path=plan_path,
        source_selection_path=source_path,
        exclusion_manifest_path=exclusion_path,
        observation_manifest_path=observation_path,
        capture_receipt_path=capture_path,
        method_freeze_path=method_freeze_path,
        state_paths={method: Path(record["path"]) for method, record in method_states.items()},
        frequency_reference_path=frequency_path,
        public_model_snapshot=snapshot,
        lens_path=lens_path,
        reference_path=prefix_path,
        output_root="experiments/TRR-0007/predictions",
    )
    student_rows = {
        row["id"]: row for row in registration["methods"] if row["role"] == "student"
    }
    assert set(student_rows) == set(contract.STUDENT_METHOD_IDS)
    for method, row in student_rows.items():
        assert row["state"]["path"] == method_states[method]["path"]
        assert row["state"]["sha256"] == method_states[method]["sha256"]
        assert row["method_freeze_sha256"] == method_freeze_record["sha256"]
    assert registration["frequency_reference"] == frequency_record

    public_metadata = gate._validate_public_metadata(registration, root=repo)
    assert public_metadata["method_freeze"]["sha256"] == method_freeze_record["sha256"]
    assert public_metadata["frequency_reference"]["sha256"] == frequency_record["sha256"]

    invalid_captures = [
        (
            "top-level truth flag",
            {**capture_payload, "truth_opened": True},
        ),
        (
            "nested truth flag",
            {
                **capture_payload,
                "execution": {**capture_payload["execution"], "truth_opened": True},
            },
        ),
        (
            "missing execution mapping",
            {key: value for key, value in capture_payload.items() if key != "execution"},
        ),
        (
            "non-mapping execution",
            {**capture_payload, "execution": False},
        ),
    ]
    for _description, invalid_capture in invalid_captures:
        invalid_record = write_json(capture_path, invalid_capture)
        with pytest.raises(register.RegisterError, match="records truth access"):
            register._verify_execution_receipts(
                root=repo,
                method_freeze=method_freeze_payload,
                method_freeze_record=method_freeze_record,
                source_record=source_record,
                exclusion_record=exclusion_record,
                observation_record=observation_record,
                capture_record=invalid_record,
            )
        invalid_registration = dict(registration)
        invalid_registration["capture_receipt"] = invalid_record
        with pytest.raises(gate.GateError, match="records truth access"):
            gate._validate_public_metadata(invalid_registration, root=repo)
    write_json(capture_path, capture_payload)



def test_frequency_error_diagnostic_uses_common_enriched_bins_without_extra_truth_read() -> None:
    truth = {
        domain: torch.cat(
            [
                torch.full((128, 1), contract.BOS_TOKEN_ID, dtype=torch.long),
                torch.ones((128, 127), dtype=torch.long),
            ],
            dim=1,
        )
        for domain in contract.DOMAIN_ORDER
    }
    tensors: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for method in contract.METHOD_ORDER:
        tensors[method] = {}
        for cell in contract.expected_method_cells(method):
            records = contract.ANCHOR_RECORDS_PER_DOMAIN if method == contract.ANCHOR_METHOD_ID else contract.RECORDS_PER_DOMAIN
            pred = truth[cell.split("__", 1)[0]][:records].clone()
            if method == contract.STUDENT_METHOD_IDS[0]:
                pred[:, 1] = 2
            values: dict[str, torch.Tensor] = {"predictions": pred}
            if method == contract.ANCHOR_METHOD_ID:
                values["a1_predictions"] = pred.clone()
            tensors[method][cell] = values
    diagnostic = scorer._frequency_error_diagnostic(
        tensors,
        truth,
        frequency_reference={
            "binding": {"path": "frequency_references_v1.json", "bytes": 1, "sha256": "a" * 64},
            "map_name": "enriched",
            "counts": {1: 3},
            "schema": contract.FREQUENCY_REFERENCE_SCHEMA,
            "counting_scope": "synthetic",
        },
    )
    assert diagnostic["status"] == "DESCRIPTIVE_SAME_PASS_AGGREGATION"
    assert diagnostic["frequency_reference"]["map_name"] == "enriched"
    row = next(
        item for item in diagnostic["rows"]
        if item["method_id"] == contract.STUDENT_METHOD_IDS[0]
        and item["cell_id"] == "pile__public_base"
        and item["prefix_bin"] == "1-15"
        and item["frequency_bin"] == "seen_2_4"
    )
    assert row["token_positions"] == contract.RECORDS_PER_DOMAIN * 15
    assert row["token_errors"] == contract.RECORDS_PER_DOMAIN



def test_prefix_ledger_binds_v3_all_style_union_and_rejects_intermediate() -> None:
    binding = bank_ledger.load_prefix_exclusion_ledger(
        repository_root=Path("."),
        path=Path(bank_ledger.PREFIX_LEDGER_V3_RELATIVE_PATH),
    )
    assert binding["file"]["sha256"] == bank_ledger.PREFIX_LEDGER_V3_SHA256
    assert binding["counts"]["collector_hashes_by_fresh_style"] == {
        "pile": 470,
        "finance": 470,
    }
    assert binding["all_bank_all_style_union"] is True
    with pytest.raises(bank_ledger.BankLedgerError, match="reviewed final v5"):
        bank_ledger.load_prefix_exclusion_ledger(
            repository_root=Path("."),
            path=Path("experiments/TRR-0007/support/public_fit_prefix_exclusions_v2.json"),
        )


def test_final_bank_ledger_binds_v5_and_rejects_stale_v3() -> None:
    final = Path("experiments/TRR-0007/support/broader_bank_v5")
    binding = bank_ledger.load_final_bank_ledgers(
        repository_root=Path("."),
        exclusion_manifest=final / "public_parent_exclusion_manifest.json",
        selected_parent_rows=final / "selected_parent_rows.json",
        corpus_plan=final / "corpus_plan.json",
    )
    assert binding["exclusion_set_counts"] == {
        "record_ids": 2449,
        "source_row_keys": 1848,
        "opaque_sequence_or_reservation_digests": 4073,
    }
    stale = Path("experiments/TRR-0007/support/broader_bank_v3")
    with pytest.raises(bank_ledger.BankLedgerError, match="reviewed final v5"):
        bank_ledger.load_final_bank_ledgers(
            repository_root=Path("."),
            exclusion_manifest=stale / "public_parent_exclusion_manifest.json",
            selected_parent_rows=stale / "selected_parent_rows.json",
            corpus_plan=stale / "corpus_plan.json",
        )
