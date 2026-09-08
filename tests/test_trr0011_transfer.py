from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from trr0011_transfer import (
    TransferDiagnosticError,
    evaluator_only_error_inventory,
    evaluator_only_fixed_auc,
    evaluator_only_transfer_inventory,
    run_registered_transfer_predictions,
    summarize_transfer,
    validate_after_cut_null,
    validate_panel_descriptor,
    validate_transfer_observation_manifest,
    validate_transfer_prediction_matrix,
    validate_variant_plan,
)


def _variant_plan() -> dict:
    variants = [
        {"variant_id": "clean", "role": "clean_public_target", "update_location": "none"},
    ]
    for role, prefix, layer, seed in [
        ("early_prefix_perturbation", "early", 1, 1101),
        ("near_cut_prefix_perturbation", "near_cut", 3, 1101),
    ]:
        for index, amplitude in enumerate((1e-3, 1e-2)):
            variants.append({
                "variant_id": f"{prefix}_{index}",
                "role": role,
                "update_location": {"layer_index": layer, "relative_parameter_l2": amplitude, "recipe_seed": seed + index},
                "artificial": True,
            })
    variants.append({
        "variant_id": "after_cut",
        "role": "after_cut_suffix_null_control",
        "update_location": {"layer_index": 4, "relative_parameter_l2": 1e-2, "recipe_seed": 1103},
        "expected_activation_equivalence": True,
        "artificial": True,
    })
    variants.append({
        "variant_id": "synthetic_adapted",
        "role": "historical_trained_benchmark_target_condition",
        "target_condition": "public_lora_2601_historical_trained_benchmark",
        "artificial": False,
        "available_for_run": True,
        "independent_target": False,
        "availability_reason": "published read-only calibration asset is bound; it is not an independent adaptation",
        "update_binding": {"path": "synthetic_lora.safetensors", "bytes": 1, "sha256": "a" * 64, "readonly": True},
    })
    return {
        "schema": "token-reconstruction.trr0011-controlled-target-transfer.v1",
        "task_id": "TRR-0011",
        "cut_depth": 4,
        "geometry": {
            "hidden_size": 2048,
            "vocabulary_size": 128256,
            "stored_sequence_tokens": 128,
            "scored_post_bos_tokens": 127,
            "bos_token_id": 128000,
        },
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "variants": variants,
    }


def test_variant_plan_requires_all_declared_roles_and_fixed_geometry() -> None:
    result = validate_variant_plan(_variant_plan())
    assert result["truth_free"] is True
    assert result["cut_depth"] == 4
    assert set(result["roles"]) == {
        "clean_public_target",
        "early_prefix_perturbation",
        "near_cut_prefix_perturbation",
        "after_cut_suffix_null_control",
        "historical_trained_benchmark_target_condition",
    }


def test_variant_plan_rejects_truth_flags_and_bad_near_cut_location() -> None:
    bad = _variant_plan()
    bad["target_labels_loaded"] = True
    with pytest.raises(TransferDiagnosticError, match="target_labels_loaded"):
        validate_variant_plan(bad)

    bad = _variant_plan()
    bad["variants"][3]["update_location"]["layer_index"] = 2
    with pytest.raises(TransferDiagnosticError, match="near-cut"):
        validate_variant_plan(bad)


def test_panel_descriptor_requires_opaque_reservation_without_selecting() -> None:
    panel = {
        "records_per_domain": 32,
        "same_record_order_across_targets": True,
        "opaque_reservation": {"receipt_sha256": "a" * 64},
        "selection_performed": False,
        "truth_opened": False,
    }
    assert validate_panel_descriptor(panel)["records_per_domain"] == 32
    panel["selection_performed"] = True
    with pytest.raises(TransferDiagnosticError, match="source selection"):
        validate_panel_descriptor(panel)


def test_margin_geometry_is_readout_consistent_and_keeps_parameter_distance_separate() -> None:
    # Two clean rows, each represented by two post-BOS positions.  The clean
    # predicted/runner readout rows differ by sqrt(2), and scale=2.
    clean_h = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]])
    changed_h = clean_h + 0.5
    clean_u = torch.nn.functional.normalize(clean_h, dim=-1)
    changed_u = torch.nn.functional.normalize(clean_h + 0.1, dim=-1)
    readout_pairs = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    result = summarize_transfer(
        clean_h,
        changed_h,
        clean_u,
        changed_u,
        torch.full((4,), 2.0),
        torch.full((4,), 0.0),
        readout_pairs,
        logit_scale=2.0,
        clean_top_ids=torch.tensor([[10, 11], [10, 11]]),
        changed_top_ids=torch.tensor([[10, 12], [10, 11]]),
        parameter_delta_l2=3.0,
        parameter_base_l2=12.0,
    )
    assert result["truth_free"] is True
    assert result["prediction_change"]["changed_rows"] == 1
    assert result["parameter_distance"]["relative_l2"] == pytest.approx(0.25)
    expected_distance = 1.0 / math.sqrt(2.0)
    assert result["metrics"]["clean_hyperplane_distance"]["median"] == pytest.approx(expected_distance)
    assert result["margin_geometry"]["threshold_fitted"] is False


def test_after_cut_null_requires_exact_activation_and_prediction_equality() -> None:
    clean = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    prediction = torch.tensor([[128000, 4], [128000, 5]])
    result = validate_after_cut_null(clean, clean.clone(), prediction, prediction.clone())
    assert result == {"status": "PASS", "activation_exact_equal": True, "prediction_exact_equal": True}
    with pytest.raises(TransferDiagnosticError, match="cut activation"):
        validate_after_cut_null(clean, clean + 1e-6)
    with pytest.raises(TransferDiagnosticError, match="prediction"):
        validate_after_cut_null(clean, clean, prediction, prediction + 1)


def test_error_inventory_is_explicitly_truth_dependent_and_paired() -> None:
    candidate = torch.tensor([[128000, 3, 4, 5]])
    comparator = torch.tensor([[128000, 9, 4, 8]])
    truth = torch.tensor([[128000, 3, 7, 8]])
    result = evaluator_only_error_inventory(candidate, comparator, truth)
    assert result["truth_dependent"] is True
    assert result["deployable"] is False
    assert result["both_correct"] == 1
    assert result["only_candidate_correct"] == 1
    assert result["only_comparator_correct"] == 1
    assert result["both_wrong"] == 1


def test_transfer_rejects_nonfinite_or_unsorted_clean_margin_inputs() -> None:
    base = torch.zeros((1, 2), dtype=torch.float32)
    pairs = torch.zeros((1, 2, 2), dtype=torch.float32)
    with pytest.raises(TransferDiagnosticError, match="non-finite"):
        summarize_transfer(base, base, base, base, [float("nan")], [0.0], pairs, logit_scale=1.0)
    with pytest.raises(TransferDiagnosticError, match="sorted"):
        summarize_transfer(base, base, base, base, [0.0], [1.0], pairs, logit_scale=1.0)


def _file_binding(path: Path, *, readonly: bool = False) -> dict:
    payload = path.read_bytes()
    result = {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    if readonly:
        result["readonly"] = True
    return result


def _transfer_manifest(tmp_path: Path) -> tuple[dict, Path]:
    observation_path = tmp_path / "observation.safetensors"
    activations = torch.zeros((32, 128, 2048), dtype=torch.bfloat16)
    mask = torch.ones((32, 128), dtype=torch.uint8)
    positions = torch.arange(128, dtype=torch.int64).unsqueeze(0).expand(32, -1).contiguous()
    save_file({"activations": activations, "attention_mask": mask, "position_ids": positions}, str(observation_path))
    code_path = tmp_path / "runner.py"
    state_path = tmp_path / "decoder.safetensors"
    embedding_path = tmp_path / "embedding.safetensors"
    loader_path = tmp_path / "loader.py"
    for path, payload in ((code_path, b"# public runner\n"), (state_path, b"public decoder state\n"), (embedding_path, b"public E binding\n"), (loader_path, b"# P09 loader\n")):
        path.write_bytes(payload)
    order = {"finance": "f" * 64, "pile": "b" * 64}
    variants = {"clean": {}}
    for domain in order:
        variants["clean"][domain] = _file_binding(observation_path) | {
            "records": 32,
            "record_ids_sha256": order[domain],
        }
    manifest = {
        "schema": "token-reconstruction.trr0011-transfer-public-inference.v1",
        "task_id": "TRR-0011",
        "status": "PUBLIC_TRANSFER_INPUTS_VALIDATED_BEFORE_TRUTH",
        "method_id": "current_fixed",
        "variant_ids": ["clean"],
        "source_order_sha256": order,
        "code_bindings": {"runner": _file_binding(code_path)},
        "state_bindings": {"decoder": _file_binding(state_path)},
        "decoder_resources": {
            "embedding": _file_binding(embedding_path),
            "state": _file_binding(state_path),
            "loader": _file_binding(loader_path) | {"module": "scripts.trr0010_p09_fixed_loader"},
        },
        "observation_bindings": variants,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    return manifest, observation_path


def test_transfer_manifest_reads_headers_and_rejects_truth_before_runner(tmp_path: Path) -> None:
    manifest, _observation_path = _transfer_manifest(tmp_path)
    checked = validate_transfer_observation_manifest(manifest, repository_root=tmp_path)
    assert checked["status"] == "PUBLIC_TRANSFER_INPUTS_VALIDATED_BEFORE_TRUTH"
    assert checked["observation_bindings"]["clean"]["finance"]["dtype"] == "BF16"
    bad = dict(manifest)
    bad["truth_opened"] = True
    with pytest.raises(TransferDiagnosticError, match="truth_opened"):
        validate_transfer_observation_manifest(bad, repository_root=tmp_path)


def test_transfer_manifest_rejects_header_mismatch_before_decoder_load(tmp_path: Path) -> None:
    manifest, observation_path = _transfer_manifest(tmp_path)
    replacement = torch.zeros((32, 128, 2047), dtype=torch.bfloat16)
    mask = torch.ones((32, 128), dtype=torch.uint8)
    positions = torch.arange(128, dtype=torch.int64).unsqueeze(0).expand(32, -1).contiguous()
    save_file({"activations": replacement, "attention_mask": mask, "position_ids": positions}, str(observation_path))
    with pytest.raises(TransferDiagnosticError, match="bytes or SHA-256 changed"):
        validate_transfer_observation_manifest(manifest, repository_root=tmp_path)


def test_transfer_bridge_validates_then_calls_existing_runner_once_per_cell(tmp_path: Path) -> None:
    manifest, _observation_path = _transfer_manifest(tmp_path)
    manifest_path = tmp_path / "transfer.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registration_path = tmp_path / "registration.json"
    registration_path.write_text("{}", encoding="utf-8")
    calls: list[str] = []

    class FakeRunner:
        @staticmethod
        def _load_registration(path, *, root, require_current_head, require_runtime_sources):
            calls.append("registration")
            return ({"methods": [{"id": "current_fixed"}]}, {"code_bindings": {}}, {"path": str(path), "bytes": 2, "sha256": "r" * 64})

        @staticmethod
        def _load_embedding(registration, *, root, device):
            calls.append("embedding")
            return torch.zeros((1, 1)), {"loader": "synthetic-test"}

        @staticmethod
        def _load_method(method_id, row, *, root, device, embedding, method_factory, code_bindings, allow_materialization):
            calls.append("method")
            return SimpleNamespace(adapter=object())

        @staticmethod
        def _run_cell(*, adapter, cell, records, hidden_size, device, method_id, guard_callback):
            calls.append(f"cell:{cell['observation']['domain'] if 'domain' in cell['observation'] else 'unknown'}")
            values = torch.full((records, 128), 7, dtype=torch.int64)
            values[:, 0] = 128000
            return values, {"measured_seconds_sum": 0.0}

    # Add the domain field expected by the fake call receipt without changing
    # the production validator's source-order or header checks.
    for domain_binding in manifest["observation_bindings"]["clean"].values():
        domain_binding["domain"] = "synthetic"
    result = run_registered_transfer_predictions(
        manifest_path,
        registration_path=registration_path,
        repository_root=tmp_path,
        output_root=tmp_path / "experiments" / "TRR-0011" / "predictions",
        require_current_head=False,
        runner_module=FakeRunner,
        write_geometry=False,
    )
    assert result["truth_opened"] is False
    assert len(result["predictions"]) == 2
    assert calls.count("registration") == 2
    assert calls.count("method") == 1


def test_transfer_inventory_names_baseline_wrong_broken_and_improved() -> None:
    result = evaluator_only_transfer_inventory(
        torch.tensor([[128000, 1, 2, 3, 4]]),
        torch.tensor([[128000, 9, 7, 3, 8]]),
        torch.tensor([[128000, 1, 7, 3, 9]]),
    )
    assert result["truth_dependent"] is True
    assert result["baseline_wrong"] == 2
    assert result["broken"] == 1
    assert result["improved"] == 1


def test_direct_decoder_geometry_uses_loaded_model_not_supplied_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    import trr0011_transfer as transfer

    monkeypatch.setattr(transfer, "STORED_SEQUENCE_TOKENS", 3)
    monkeypatch.setattr(transfer, "SCORED_POST_BOS_TOKENS", 2)
    monkeypatch.setattr(transfer, "VOCABULARY_SIZE", 5)
    monkeypatch.setattr(transfer, "HIDDEN_SIZE", 2)

    class TinyDecoder(torch.nn.Module):
        def projected_hidden(self, activation, mask):
            del mask
            return torch.nn.functional.normalize(activation, dim=-1)

        def logits_from_rows(self, projected, record_slots, position_slots, embedding):
            return projected[record_slots, position_slots] @ embedding.transpose(0, 1)

    embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.5, 0.5]])
    adapter = SimpleNamespace(model=TinyDecoder(), embedding=embedding, device=torch.device("cpu"))
    geometry = transfer._decoder_geometry_for_record(
        adapter,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        torch.ones(3, dtype=torch.bool),
        torch.arange(3),
    )
    assert tuple(geometry["projected_features"].shape) == (2, 2)
    assert tuple(geometry["top_runner_logits"].shape) == (2, 2)
    assert tuple(geometry["top_runner_ids"].shape) == (2, 2)
    assert tuple(geometry["top_runner_readout"].shape) == (2, 2, 2)
    assert geometry["top_runner_ids"][0, 0].item() == 1


def test_transfer_matrix_gate_rejects_partial_old_shape_before_truth(tmp_path: Path) -> None:
    manifest, _observation_path = _transfer_manifest(tmp_path)
    input_path = tmp_path / "transfer_input.json"
    input_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    panel_path = tmp_path / "panel.json"
    plan_path.write_text(json.dumps(_variant_plan()), encoding="utf-8")
    panel_path.write_text(json.dumps({
        "records_per_domain": 32,
        "same_record_order_across_targets": True,
        "opaque_reservation": {"receipt_sha256": "a" * 64},
        "selection_performed": False,
        "truth_opened": False,
    }), encoding="utf-8")
    matrix = {
        "schema": "token-reconstruction.trr0011-transfer-public-prediction-matrix.v1",
        "task_id": "TRR-0011",
        "status": "PUBLIC_TRANSFER_PREDICTIONS_COMPLETE_BEFORE_TRUTH",
        "method_ids": ["expanded_fixed", "current_fixed"],
        "variant_ids": ["clean"],
        "domain_order": ["finance", "pile"],
        "records_per_domain": 32,
        "input_manifest": _file_binding(input_path),
        "variant_plan": _file_binding(plan_path),
        "panel_descriptor": _file_binding(panel_path),
        "predictions": {},
        "decoder_geometry": {},
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    with pytest.raises(TransferDiagnosticError, match="bound frozen plan"):
        validate_transfer_prediction_matrix(matrix, repository_root=tmp_path)


def test_fixed_auc_is_predeclared_distortion_direction_and_no_threshold() -> None:
    result = evaluator_only_fixed_auc(
        torch.tensor([0.9, 0.0, 0.1]),
        torch.tensor([1, 2, 3]),
        torch.tensor([9, 2, 7]),
        torch.tensor([1, 2, 7]),
        score_name="raw_activation_l2",
    )
    assert result["truth_dependent"] is True
    assert result["positive_label"] == "broken"
    assert result["threshold_fitted"] is False
    assert result["risk_set"] == "baseline_correct"
    assert result["n_broken"] == 1
    assert result["n_remaining_correct"] == 1
    assert result["n_improved_excluded"] == 1
    assert result["auc"] == pytest.approx(1.0)


def test_fixed_auc_is_unknown_for_near_zero_risk_set_spread_and_excludes_improvements() -> None:
    result = evaluator_only_fixed_auc(
        torch.tensor([1e-13, 1e-13, 0.5]),
        torch.tensor([1, 2, 3]),
        torch.tensor([9, 2, 7]),
        torch.tensor([1, 2, 7]),
        score_name="parameter_relative_l2",
    )
    assert result["status"] == "UNKNOWN"
    assert result["risk_set"] == "baseline_correct"
    assert result["n_broken"] == 1
    assert result["n_remaining_correct"] == 1
    assert result["n_improved_excluded"] == 1
    assert "no resolved variation" in result["reason"]
