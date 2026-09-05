from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import trr0004_fresh_confirmation as fc
from token_reconstruction.footing import external_file_record, file_record, sha256_file
from token_reconstruction.freeze import create_freeze_receipt


COMMIT = "a" * 40


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _asset(path: Path, *, root: Path) -> dict[str, object]:
    return {
        **file_record(path, repository_root=root),
        "tensor_key": "activations",
        "row_indices": list(range(fc.RECORDS_PER_STYLE)),
    }


def _panel_fixture(tmp_path: Path) -> tuple[Path, dict, Path, Path]:
    root = tmp_path
    selection_plan = root / "selection_plan.json"
    _write_json(selection_plan, {"status": "frozen public selection fixture", "records_selected": 32})
    plan_sha = sha256_file(selection_plan)
    observation_root = root / "observations"
    observation_root.mkdir()
    records_by_style: dict[str, list[dict[str, object]]] = {}
    cells: list[dict[str, object]] = []
    for style_index, style in enumerate(fc.STYLE_ORDER):
        sequence_tokens = fc.SEQUENCE_TOKENS[style]
        record_rows = [
            {
                "record_id": f"trr4-{style}-{index:02d}",
                "public_record_sha256": f"{style_index * 100 + index + 1:064x}",
                "raw_index": index + style_index * 1000,
                "valid_tokens": sequence_tokens - (index % 5) if style == "finance" else sequence_tokens,
            }
            for index in range(fc.RECORDS_PER_STYLE)
        ]
        records_by_style[style] = record_rows
        mask_rows: list[list[int]] = []
        position_rows: list[list[int]] = []
        for index in range(fc.RECORDS_PER_STYLE):
            valid = sequence_tokens if style == "pile" else sequence_tokens - (index % 5)
            mask = [1] * valid + [0] * (sequence_tokens - valid)
            position = list(range(valid)) + [max(valid - 1, 0)] * (sequence_tokens - valid)
            mask_rows.append(mask)
            position_rows.append(position)
        for condition_index, condition in enumerate(fc.CONDITION_ORDER):
            observation_path = observation_root / f"{style}_{condition}.safetensors"
            value = torch.full(
                (fc.RECORDS_PER_STYLE, sequence_tokens, fc.HIDDEN_SIZE),
                float(condition_index),
                dtype=torch.bfloat16,
            )
            save_file({"activations": value}, str(observation_path))
            cells.append(
                {
                    "id": f"{style}__{condition}",
                    "style": style,
                    "condition": condition,
                    "shift_role": (
                        "matched_public_control"
                        if condition == "public_base"
                        else "single_public_shift_diagnostic"
                    ),
                    "records": record_rows,
                    "attention_mask": mask_rows,
                    "position_ids": position_rows,
                    "observation": _asset(observation_path, root=root),
                    "geometry": {
                        "records": fc.RECORDS_PER_STYLE,
                        "sequence_tokens": sequence_tokens,
                        "hidden_size": fc.HIDDEN_SIZE,
                        "cut_depth": fc.CUT_DEPTH,
                    },
                }
            )
    panel = {
        "schema": fc.PANEL_SCHEMA,
        "task_id": fc.TASK_ID,
        "status": "FROZEN_FRESH_CONFIRMATION_PANEL",
        "panel_id": "fixture",
        "source_material_included": False,
        "selection_plan_sha256": plan_sha,
        "observation_generation": {
            "path": "public_prefix.forward_full",
            "same_public_prefix_path": True,
            "batch_size": 8,
            "sequence_tokens": 192,
        },
        "model": {"id": fc.MODEL_ID, "revision": fc.MODEL_REVISION},
        "cut_depth": fc.CUT_DEPTH,
        "hidden_size": fc.HIDDEN_SIZE,
        "styles": [
            {
                "id": "pile",
                "records": fc.RECORDS_PER_STYLE,
                "sequence_tokens": fc.SEQUENCE_TOKENS["pile"],
                "hidden_size": fc.HIDDEN_SIZE,
                "input_style": "plain Pile text",
            },
            {
                "id": "finance",
                "records": fc.RECORDS_PER_STYLE,
                "sequence_tokens": fc.SEQUENCE_TOKENS["finance"],
                "hidden_size": fc.HIDDEN_SIZE,
                "input_style": "Finance chat-template rendering",
            },
        ],
        "conditions": [
            {
                "id": "public_base",
                "weights_available_to_reconstructor": True,
                "role": "matched public control",
            },
            {
                "id": "public_lora_2601",
                "weights_available_to_reconstructor": False,
                "role": "one synthetic target-shift diagnostic",
            },
        ],
        "cells": cells,
    }
    panel_path = root / "panel.json"
    _write_json(panel_path, panel)
    return panel_path, fc.load_fresh_panel(panel_path, repository_root=root), root, selection_plan


def _method_setup(panel_path: Path, root: Path, *, methods: tuple[str, ...] = fc.METHOD_IDS):
    code = root / "method_adapter.py"
    code.write_text("def predict(*args):\n    return None\n", encoding="utf-8")
    bindings: dict[str, dict] = {}
    runtime_assets = {}
    for role in fc.RUNTIME_ASSET_ROLES:
        path = root / f"{role}.public"
        path.write_bytes(role.encode("ascii"))
        runtime_assets[role] = path
    for index, method in enumerate(methods):
        state = root / f"{method}.state"
        state.write_bytes(f"state-{index}".encode())
        config = root / f"{method}.config.json"
        _write_json(config, {"method_id": method, "version": 1})
        bindings[method] = fc.make_confirmation_binding(
            panel_path=panel_path,
            repository_root=root,
            method_id=method,
            method_rule=next(row["rule"] for row in fc.METHOD_SPECS if row["id"] == method),
            method_state_paths=[state],
            method_config_paths=[config],
            code_paths=[code],
            code_commit=COMMIT,
            runtime_asset_paths=runtime_assets,
        )
    return bindings


def _write_predictions(panel_path: Path, panel: dict, root: Path, bindings: dict[str, dict]):
    output = root / "predictions"
    output.mkdir(exist_ok=True)
    panel_sha = sha256_file(panel_path)
    cells = fc.load_fresh_cells(panel, repository_root=root)
    for cell in cells:
        active = cell.attention_mask.to(torch.bool)
        for method in bindings:
            prediction = torch.full(cell.attention_mask.shape, fc.INVALID_TOKEN_ID, dtype=torch.int64)
            prediction[active] = 1
            prediction[:, 0] = fc.BOS_TOKEN_ID
            tensors: dict[str, torch.Tensor] = {"predictions": prediction}
            if method == "frozen_a1_a2_k256":
                candidates = torch.full((*prediction.shape, 4), fc.INVALID_TOKEN_ID, dtype=torch.int64)
                candidates[active] = 2
                scores = torch.full(candidates.shape, float("-inf"), dtype=torch.float32)
                scores[active] = 0.0
                tensors.update({"candidates": candidates, "candidate_scores": scores})
            metadata = {
                "schema": fc.PREDICTION_SCHEMA,
                "task_id": fc.TASK_ID,
                "panel_sha256": panel_sha,
                "selection_plan_sha256": panel["selection_plan_sha256"],
                "observation_sha256": cell.observation_sha256,
                "cell_id": cell.cell_id,
                "style": cell.style,
                "condition": cell.condition,
                "method_id": method,
                "geometry_json": json.dumps(
                    {
                        "records": fc.RECORDS_PER_STYLE,
                        "sequence_tokens": cell.sequence_tokens,
                        "hidden_size": fc.HIDDEN_SIZE,
                        "cut_depth": fc.CUT_DEPTH,
                    },
                    sort_keys=True,
                ),
                "binding_json": json.dumps(bindings[method], sort_keys=True),
            }
            path = fc.expected_prediction_path(output, cell=cell, method_id=method)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(path), metadata=metadata)
    return output


def _registration(panel_path: Path, selection_plan: Path, root: Path, bindings: dict[str, dict]) -> Path:
    registration = {
        "schema": fc.REGISTRATION_SCHEMA,
        "task_id": fc.TASK_ID,
        "status": "FROZEN_METHOD_REGISTRATION",
        "panel": file_record(panel_path, repository_root=root),
        "selection_plan": file_record(selection_plan, repository_root=root),
        "method_ids": list(bindings),
        "methods": [
            {
                "id": method,
                "track": fc._METHOD_TRACKS[method],
                "candidate_policy": fc.CANDIDATE_POLICIES[method],
                "rule": next(row["rule"] for row in fc.METHOD_SPECS if row["id"] == method),
                "binding": binding,
            }
            for method, binding in bindings.items()
        ],
    }
    path = root / "registration.json"
    _write_json(path, registration)
    return path


def _truth_binding(panel_path: Path, panel: dict, root: Path, selection_plan: Path) -> tuple[dict, Path]:
    cells = fc.load_fresh_cells(panel, repository_root=root)
    preparation_path = root / "truth_preparation.json"
    _write_json(preparation_path, {"role": "public-label preparation", "truth_opened": False})
    sidecar_path = root / "private_truth" / "answers.safetensors"
    sidecar_path.parent.mkdir()
    truth: dict[str, torch.Tensor] = {}
    for cell in cells:
        value = torch.full(cell.attention_mask.shape, fc.PAD_TOKEN_ID, dtype=torch.int64)
        active = cell.attention_mask.to(torch.bool)
        value[active] = 1
        value[:, 0] = fc.BOS_TOKEN_ID
        truth[cell.cell_id] = value
    placeholder = {"path": str(sidecar_path.resolve()), "bytes": 0, "sha256": "0" * 64}
    binding = fc.build_confirmation_truth_binding(
        panel_sha256=sha256_file(panel_path),
        selection_plan_sha256=sha256_file(selection_plan),
        cells=cells,
        truth=truth,
        preparation=external_file_record(preparation_path),
        sidecar=placeholder,
    )
    fc.write_confirmation_truth_sidecar(sidecar_path, cells=cells, truth=truth, binding=binding)
    binding["sidecar"] = external_file_record(sidecar_path)
    return binding, sidecar_path


def _evaluation_fixture(tmp_path: Path):
    panel_path, panel, root, selection_plan = _panel_fixture(tmp_path)
    bindings = _method_setup(panel_path, root)
    output = _write_predictions(panel_path, panel, root, bindings)
    registration = _registration(panel_path, selection_plan, root, bindings)
    truth_binding, truth_path = _truth_binding(panel_path, panel, root, selection_plan)
    freeze_plan = root / "freeze_plan.json"
    _write_json(freeze_plan, {"task_id": fc.TASK_ID, "status": "freeze fixture"})
    receipt = root / "freeze_receipt.json"
    create_freeze_receipt(
        repository_root=root,
        frozen_root=output,
        plan_path=freeze_plan,
        receipt_path=receipt,
        preregistration_commit=COMMIT,
        created_utc="2026-09-05T00:00:00Z",
        metadata={
            "task_id": fc.TASK_ID,
            "panel_sha256": sha256_file(panel_path),
            "selection_plan_sha256": sha256_file(selection_plan),
            "method_ids": list(bindings),
            "registration_sha256": sha256_file(registration),
            "truth_binding": truth_binding,
        },
    )
    return {
        "panel_path": panel_path,
        "panel": panel,
        "root": root,
        "selection_plan": selection_plan,
        "bindings": bindings,
        "output": output,
        "registration": registration,
        "truth_binding": truth_binding,
        "truth_path": truth_path,
        "receipt": receipt,
    }


def _gate_kwargs(bundle: dict) -> dict:
    return {
        "receipt_path": bundle["receipt"],
        "repository_root": bundle["root"],
        "truth_path": bundle["truth_path"],
        "output_root": bundle["output"],
        "panel_path": bundle["panel_path"],
        "selection_plan_path": bundle["selection_plan"],
        "registration_path": bundle["registration"],
        "method_ids": tuple(bundle["bindings"]),
        "expected_bindings": bundle["bindings"],
        "candidate_policies": fc.CANDIDATE_POLICIES,
        "truth_binding": bundle["truth_binding"],
    }


def test_prospective_plan_has_fixed_rule_without_selected_records(tmp_path: Path) -> None:
    plan = fc.build_prospective_plan(repository_root=tmp_path, generated_at_utc="2026-09-05T00:00:00Z")
    assert plan["schema"] == fc.PLAN_SCHEMA
    assert plan["status"] == "PROSPECTIVE_SELECTION_RULE_NO_RECORDS_SELECTED"
    assert plan["selection_rule"]["record_ids_selected"] is None
    assert plan["selection_rule"]["record_hashes_selected"] is None
    assert [(row["id"], row["records"], row["sequence_tokens"]) for row in plan["styles"]] == [
        ("pile", 16, 40),
        ("finance", 16, 128),
    ]
    assert tuple(row["id"] for row in plan["methods_prospective"]) == fc.METHOD_IDS
    assert plan["execution"]["model_loaded"] is False
    assert plan["execution"]["truth_opened"] is False
    assert plan["coverage_analysis"]["token_frequency_bins"] == ["0", "1-4", "5-19", "20+"]
    assert plan["coverage_analysis"]["position_bins_post_bos"] == ["1-15", "16-39", "40-79", "80+"]


def test_panel_pairs_public_records_masks_and_observations(tmp_path: Path) -> None:
    panel_path, panel, root, _ = _panel_fixture(tmp_path)
    cells = fc.load_fresh_cells(panel, repository_root=root)
    assert len(cells) == 4
    for style in fc.STYLE_ORDER:
        base = next(cell for cell in cells if cell.cell_id == f"{style}__public_base")
        shifted = next(cell for cell in cells if cell.cell_id == f"{style}__public_lora_2601")
        assert base.record_ids == shifted.record_ids
        assert torch.equal(base.attention_mask, shifted.attention_mask)
        assert torch.equal(base.position_ids, shifted.position_ids)
        assert base.observation_path != shifted.observation_path
    assert panel["selection_plan_sha256"] == sha256_file(root / "selection_plan.json")
    assert panel_path.is_file()


def test_prediction_set_requires_all_methods_and_rejects_oov_or_extra(tmp_path: Path) -> None:
    bundle = _evaluation_fixture(tmp_path)
    fc.validate_complete_confirmation_predictions(
        bundle["output"],
        panel_path=bundle["panel_path"],
        repository_root=bundle["root"],
        method_ids=tuple(bundle["bindings"]),
        expected_bindings=bundle["bindings"],
        candidate_policies=fc.CANDIDATE_POLICIES,
    )
    extra = bundle["output"] / "extra.safetensors"
    save_file({"predictions": torch.zeros((1, 1), dtype=torch.int64)}, str(extra))
    with pytest.raises(fc.ConfirmationError, match="incomplete"):
        fc.validate_complete_confirmation_predictions(
            bundle["output"],
            panel_path=bundle["panel_path"],
            repository_root=bundle["root"],
            method_ids=tuple(bundle["bindings"]),
            expected_bindings=bundle["bindings"],
            candidate_policies=fc.CANDIDATE_POLICIES,
        )
    extra.unlink()

    cell = fc.load_fresh_cells(bundle["panel"], repository_root=bundle["root"])[0]
    bad_path = fc.expected_prediction_path(bundle["output"], cell=cell, method_id="historical_affine_ce_no_vocab_bias")
    bad_path.chmod(0o644)
    prediction = torch.full(cell.attention_mask.shape, fc.INVALID_TOKEN_ID, dtype=torch.int64)
    prediction[cell.attention_mask.to(torch.bool)] = fc.VOCAB_SIZE
    prediction[:, 0] = fc.BOS_TOKEN_ID
    binding = bundle["bindings"]["historical_affine_ce_no_vocab_bias"]
    save_file(
        {"predictions": prediction},
        str(bad_path),
        metadata={
            "schema": fc.PREDICTION_SCHEMA,
            "task_id": fc.TASK_ID,
            "panel_sha256": sha256_file(bundle["panel_path"]),
            "selection_plan_sha256": bundle["panel"]["selection_plan_sha256"],
            "observation_sha256": cell.observation_sha256,
            "cell_id": cell.cell_id,
            "style": cell.style,
            "condition": cell.condition,
            "method_id": "historical_affine_ce_no_vocab_bias",
            "geometry_json": json.dumps({"records": 16, "sequence_tokens": cell.sequence_tokens, "hidden_size": 2048, "cut_depth": 4}, sort_keys=True),
            "binding_json": json.dumps(binding, sort_keys=True),
        },
    )
    with pytest.raises(fc.ConfirmationError, match="out of vocabulary|invalid"):
        fc.validate_complete_confirmation_predictions(
            bundle["output"],
            panel_path=bundle["panel_path"],
            repository_root=bundle["root"],
            method_ids=tuple(bundle["bindings"]),
            expected_bindings=bundle["bindings"],
            candidate_policies=fc.CANDIDATE_POLICIES,
        )


def test_binding_mutation_is_rejected(tmp_path: Path) -> None:
    bundle = _evaluation_fixture(tmp_path)
    config = bundle["root"] / "historical_affine_ce_no_vocab_bias.config.json"
    config.write_text('{"method_id":"historical_affine_ce_no_vocab_bias","version":2}\n', encoding="utf-8")
    with pytest.raises(fc.ConfirmationError, match="binding|changed"):
        fc.validate_complete_confirmation_predictions(
            bundle["output"],
            panel_path=bundle["panel_path"],
            repository_root=bundle["root"],
            method_ids=tuple(bundle["bindings"]),
            expected_bindings=bundle["bindings"],
            candidate_policies=fc.CANDIDATE_POLICIES,
        )


def test_truth_loader_runs_only_after_complete_public_gate(tmp_path: Path) -> None:
    bundle = _evaluation_fixture(tmp_path)
    calls: list[Path] = []

    def loader(path: Path):
        calls.append(path)
        return "private truth"

    receipt, value = fc.open_truth_after_confirmation_gate(
        truth_loader=loader,
        gate_kwargs=_gate_kwargs(bundle),
    )
    assert receipt["metadata"]["task_id"] == fc.TASK_ID
    assert value == "private truth"
    assert calls == [bundle["truth_path"]]

    missing = fc.expected_prediction_path(
        bundle["output"],
        cell=fc.load_fresh_cells(bundle["panel"], repository_root=bundle["root"])[0],
        method_id="historical_alpaca_a1",
    )
    missing.chmod(0o644)
    missing.unlink()
    with pytest.raises(fc.ConfirmationError, match="freeze|incomplete|unavailable"):
        fc.open_truth_after_confirmation_gate(
            truth_loader=loader,
            gate_kwargs=_gate_kwargs(bundle),
        )
    assert calls == [bundle["truth_path"]]


def test_warmed_timing_checks_three_repeated_predictions(tmp_path: Path) -> None:
    observations = torch.zeros((2, 4, fc.HIDDEN_SIZE), dtype=torch.float32)
    mask = torch.ones((2, 4), dtype=torch.long)
    positions = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    calls = 0

    def predictor(hidden: torch.Tensor, _mask: torch.Tensor, _positions: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        token = 1 if calls not in (3, 4) else 2
        return torch.full((hidden.shape[0],), token, dtype=torch.long, device=hidden.device)

    predictions, timing = fc.run_warmed_prediction(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        predictor=predictor,
    )
    assert tuple(predictions.shape) == (2, 4)
    assert calls == 8
    assert timing["warmup_runs"] == 1
    assert timing["measured_runs"] == 3
    assert timing["records"][0]["repeated_prediction_exact"] is False
    assert timing["records"][0]["mismatch_runs"] == [2, 3]
    assert timing["records"][1]["repeated_prediction_exact"] is True
    assert timing["cold_costs_separate"] is True


def test_truth_gate_rejects_subset_method_set_before_truth(tmp_path: Path) -> None:
    bundle = _evaluation_fixture(tmp_path)
    calls: list[Path] = []

    def loader(path: Path):
        calls.append(path)
        return "private truth"

    kwargs = _gate_kwargs(bundle)
    subset = tuple(fc.METHOD_IDS[:-1])
    kwargs["method_ids"] = subset
    kwargs["expected_bindings"] = {method: bundle["bindings"][method] for method in subset}
    kwargs["candidate_policies"] = {method: fc.CANDIDATE_POLICIES[method] for method in subset}
    with pytest.raises(fc.ConfirmationError, match="registered five-method set|registration"):
        fc.open_truth_after_confirmation_gate(truth_loader=loader, gate_kwargs=kwargs)
    assert calls == []


def test_truth_gate_rejects_altered_registered_policy_before_truth(tmp_path: Path) -> None:
    bundle = _evaluation_fixture(tmp_path)
    registration = json.loads(bundle["registration"].read_text(encoding="utf-8"))
    registration["methods"][0]["candidate_policy"] = "required"
    _write_json(bundle["registration"], registration)
    calls: list[Path] = []

    def loader(path: Path):
        calls.append(path)
        return "private truth"

    with pytest.raises(fc.ConfirmationError, match="candidate policy|registration"):
        fc.open_truth_after_confirmation_gate(
            truth_loader=loader,
            gate_kwargs=_gate_kwargs(bundle),
        )
    assert calls == []


def test_tampered_truth_is_rejected_after_complete_prediction_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _evaluation_fixture(tmp_path)
    calls: list[Path] = []
    prediction_gate_calls: list[bool] = []
    original = fc.validate_complete_confirmation_predictions

    def wrapped(*args, **kwargs):
        prediction_gate_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(fc, "validate_complete_confirmation_predictions", wrapped)

    def loader(path: Path):
        calls.append(path)
        return "private truth"

    fc.open_truth_after_confirmation_gate(
        truth_loader=loader,
        gate_kwargs=_gate_kwargs(bundle),
    )
    assert prediction_gate_calls == [True]
    assert calls == [bundle["truth_path"]]

    bundle["truth_path"].write_bytes(bundle["truth_path"].read_bytes() + b"tamper")
    with pytest.raises(fc.ConfirmationError, match="sidecar hash|size|changed"):
        fc.open_truth_after_confirmation_gate(
            truth_loader=loader,
            gate_kwargs=_gate_kwargs(bundle),
        )
    assert len(prediction_gate_calls) == 2
    assert calls == [bundle["truth_path"]]
