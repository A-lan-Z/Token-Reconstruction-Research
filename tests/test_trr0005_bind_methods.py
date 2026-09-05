from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import trr0005_bind_methods as binder
from token_reconstruction import trr0005_contract as contract
from token_reconstruction.footing import file_record


COMMIT = "a" * 40


def _assets(tmp_path: Path, *, separate_causal: bool = False) -> dict:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    decision = root / "experiments/TRR-0005/decision_plan.json"
    decision.parent.mkdir(parents=True)
    decision.write_text(
        json.dumps({"schema": "decision", "task_id": contract.TASK_ID}) + "\n",
        encoding="utf-8",
    )
    lens = root / "experiments/TRR-0004/evidence/comparators/public_a1_lens.pt"
    lens.parent.mkdir(parents=True)
    lens.write_bytes(b"retained lens")
    code = root / "scripts/trr0005_run_predictions.py"
    code.parent.mkdir(parents=True)
    code.write_bytes(b"driver source")
    for distribution in ("original", "enriched"):
        for state in (
            "joint_full_affine",
            "affine_causal_h_attention128",
            "affine_trained_diagonal_attention128",
        ):
            fit = root / "experiments/TRR-0005/joint_fit_v1" / distribution / state
            fit.mkdir(parents=True)
            (fit / "selected.safetensors").write_bytes(
                f"{distribution}:{state}".encode()
            )
    fit_methods = {}
    for distribution in ("original", "enriched"):
        methods = {}
        for state, score, step in (
            ("joint_full_affine", 0.90, 1500),
            ("affine_causal_h_attention128", 0.89, 1300),
            ("affine_trained_diagonal_attention128", 0.91, 1600),
        ):
            curve = (
                root
                / "experiments/TRR-0005/joint_fit_v1"
                / distribution
                / state
                / "learning_curve.json"
            )
            curve.write_text(json.dumps({"state": state}) + "\n", encoding="utf-8")
            curve_record = file_record(curve, repository_root=root)
            methods[state] = {
                "canonical_method_id": f"{distribution}__{state}",
                "best_validation_style_balanced_token_accuracy": score,
                "selected_step": step,
                "curve": {
                    "path": str(curve.resolve()),
                    "bytes": curve_record["bytes"],
                    "sha256": curve_record["sha256"],
                },
            }
        fit_methods[distribution] = {"methods": methods}
    fit_evidence = root / "experiments/TRR-0005/joint_fit_v1/run_evidence.json"
    fit_evidence.write_text(
        json.dumps(
            {
                "schema": "token-reconstruction.trr0005-joint-fit.v1",
                "task_id": contract.TASK_ID,
                "status": "JOINT_FIT_COMPLETE_NO_FINAL_EVALUATION",
                "final_holdout_loaded": False,
                "current_evaluator_truth_accessed": False,
                "distributions": fit_methods,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    causal_root = root / "experiments/TRR-0005/joint_fit_qknorm_v1"
    if separate_causal:
        for distribution in ("original", "enriched"):
            state = causal_root / distribution / "affine_causal_h_attention128"
            state.mkdir(parents=True)
            (state / "selected.safetensors").write_bytes(
                f"repaired:{distribution}".encode()
            )
    amendment = root / "experiments/TRR-0005/qk_amendment.json"
    amendment.write_text(
        json.dumps(
            {
                "schema": "token-reconstruction.trr0005-public-development-amendment.v1",
                "task_id": contract.TASK_ID,
                "status": "DECLARED_BEFORE_REPAIR_FITS",
                "truth_accessed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    external = tmp_path / "external"
    external.mkdir()
    embedding = external / "embedding.safetensors"
    checkpoint = external / "p0.safetensors"
    config = external / "p0-config.json"
    embedding.write_bytes(b"E")
    checkpoint.write_bytes(b"P0")
    config.write_bytes(b"{}")
    return {
        "root": root,
        "decision": decision,
        "lens": lens,
        "code": code,
        "fit": root / "experiments/TRR-0005/joint_fit_v1",
        "causal": causal_root,
        "amendment": amendment,
        "embedding": embedding,
        "checkpoint": checkpoint,
        "config": config,
    }


def _build(tmp_path: Path, *, separate_causal: bool = False) -> tuple[dict, dict]:
    a = _assets(tmp_path, separate_causal=separate_causal)
    freeze = a["root"] / "experiments/TRR-0005/method_freeze.json"
    selection = a["root"] / "experiments/TRR-0005/public_validation_selection.json"
    kwargs = dict(
        repository_root=a["root"],
        decision_plan_path=a["decision"],
        output_freeze=freeze,
        output_selection=selection,
        fit_root=a["fit"],
        lens_path=a["lens"],
        embedding_path=a["embedding"],
        p0_checkpoint=a["checkpoint"],
        p0_config=a["config"],
        code_commit=COMMIT,
        code_paths=[a["code"]],
    )
    if separate_causal:
        kwargs.update(
            causal_fit_root=a["causal"],
            attention_amendment_path=a["amendment"],
        )
    result = binder.build_method_freeze(**kwargs)
    return a, result


def _panel_and_plan(a: dict, freeze_result: dict) -> tuple[Path, Path]:
    root = a["root"]
    freeze_path = root / "experiments/TRR-0005/method_freeze.json"
    freeze_digest = freeze_result["method_freeze"]["sha256"]
    selection = json.loads(
        (root / "experiments/TRR-0005/public_validation_selection.json").read_text()
    )
    plan = root / "experiments/TRR-0005/selection_plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "token-reconstruction.trr0005-fresh-source-selection.v1",
                "task_id": contract.TASK_ID,
                "status": "FROZEN_FRESH_SOURCE_SELECTION_NO_TRUTH",
                "method_freeze_sha256": freeze_digest,
                "public_validation_selection": selection,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan_record = file_record(plan, repository_root=root)
    cells = {}
    for style in contract.STYLE_ORDER:
        records = [
            {
                "record_id": f"{style}-{i}",
                "public_record_sha256": f"{i + 1:064x}",
                "raw_index": 7000 + i,
                "source_index": 7000 + i,
                "valid_tokens": 128,
            }
            for i in range(contract.RECORDS_PER_DOMAIN)
        ]
        for condition in contract.CONDITION_ORDER:
            cell_id = f"{style}__{condition}"
            cells[cell_id] = {
                "cell_id": cell_id,
                "style": style,
                "condition": condition,
                "records": copy.deepcopy(records),
            }
    panel = root / "experiments/TRR-0005/panel.json"
    panel.write_text(
        json.dumps(
            {
                "schema": contract.PANEL_SCHEMA,
                "task_id": contract.TASK_ID,
                "status": "FROZEN_FRESH_CONFIRMATION_PANEL",
                "sequence_tokens": 128,
                "records_per_domain": 128,
                "cells": cells,
                "selection_plan": {
                    "path": str(plan.resolve()),
                    "bytes": plan_record["bytes"],
                    "sha256": plan_record["sha256"],
                },
                "method_freeze_sha256": freeze_digest,
                "public_validation_selection": selection,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return panel, plan


def test_source_freeze_contains_exact_selection_and_runtime_roles(tmp_path):
    a, result = _build(tmp_path)
    freeze = result
    assert freeze["status"] == binder.METHOD_FREEZE_STATUS
    assert freeze["method_ids"] == list(contract.METHOD_IDS)
    assert freeze["public_validation_selection"]["decision_plan_sha256"]
    for method_id in contract.METHOD_IDS:
        binding = freeze["state_bindings"][method_id]
        assert len(binding["method_state"]) == 1
        assert binding["method_state"][0]["sha256"] == binding["state_sha256"]
        assert binding["method_config"]
        assert binding["code"]
        assert binding["code_commit"] == COMMIT
        roles = set(binding["runtime_assets"])
        expected = (
            {"public_embedding_table", "public_prefix_checkpoint", "public_prefix_config"}
            if method_id == binder.A2_METHOD_ID
            else {"public_embedding_table"}
        )
        assert roles == expected
    selection = json.loads(
        (a["root"] / "experiments/TRR-0005/public_validation_selection.json").read_text()
    )
    original = selection["distributions"]["original"]
    assert original["candidate_method_ids"] == [
        "original__joint_full_affine",
        "original__affine_trained_diagonal_attention128",
    ]
    assert original["selected_method_id"] == (
        "original__affine_trained_diagonal_attention128"
    )
    assert original["selected_score"] == 0.91
    assert original["selected_step"] == 1600
    assert selection["fit_evidence"]["path"].endswith(
        "experiments/TRR-0005/joint_fit_v1/run_evidence.json"
    )
    assert original["candidates"][1]["curve_file"]["path"].endswith(
        "original/affine_trained_diagonal_attention128/learning_curve.json"
    )


def test_producer_validator_accepts_source_free_marker_and_rejects_panel_payload(tmp_path):
    a, result = _build(tmp_path)
    marker = a["root"] / "experiments/TRR-0005/method_freeze.json"
    from scripts import trr0005_produce_confirmation as producer

    accepted = producer._validate_preselection(
        marker, decision_plan_path=a["decision"]
    )
    assert accepted["method_ids"] == list(contract.METHOD_IDS)
    value = json.loads(marker.read_text())
    value["panel_sha256"] = "b" * 64
    bad = a["root"] / "experiments/TRR-0005/bad_freeze.json"
    bad.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(producer.ProducerError, match="fresh source or panel"):
        producer._validate_preselection(bad)


def test_panel_registration_adds_descriptors_without_changing_frozen_bindings(tmp_path):
    a, freeze = _build(tmp_path)
    panel, plan = _panel_and_plan(a, freeze)
    output = a["root"] / "experiments/TRR-0005/registration.json"
    result = binder.bind_panel_registration(
        repository_root=a["root"],
        method_freeze_path=a["root"] / "experiments/TRR-0005/method_freeze.json",
        panel_path=panel,
        selection_plan_path=plan,
        public_validation_selection_path=(
            a["root"] / "experiments/TRR-0005/public_validation_selection.json"
        ),
        decision_plan_path=a["decision"],
        output_registration=output,
    )
    registration = json.loads(output.read_text())
    assert registration["status"] == binder.REGISTRATION_STATUS
    assert registration["panel"]["sha256"] == file_record(panel, repository_root=a["root"])["sha256"]
    assert registration["selection_plan"]["sha256"] == file_record(plan, repository_root=a["root"])["sha256"]
    assert registration["method_freeze_sha256"] == freeze["method_freeze"]["sha256"]
    contract.validate_registration(registration, require_frozen=True)
    for method_id in contract.METHOD_IDS:
        before = freeze["state_bindings"][method_id]
        after = registration["state_bindings"][method_id]
        panel_descriptor = after.pop("panel")
        assert panel_descriptor == registration["panel"]
        assert after == before
    assert result["registration"]["sha256"] == file_record(
        output, repository_root=a["root"]
    )["sha256"]


def test_panel_registration_rejects_changed_selection_or_record_order(tmp_path):
    a, freeze = _build(tmp_path)
    panel, plan = _panel_and_plan(a, freeze)
    panel_value = json.loads(panel.read_text())
    panel_value["cells"]["pile__public_base"]["records"][0]["record_id"] = "wrong"
    panel.write_text(json.dumps(panel_value), encoding="utf-8")
    with pytest.raises(binder.MethodBindingError):
        binder.bind_panel_registration(
            repository_root=a["root"],
            method_freeze_path=a["root"] / "experiments/TRR-0005/method_freeze.json",
            panel_path=panel,
            selection_plan_path=plan,
            output_registration=a["root"] / "registration.json",
        )


def test_separate_causal_root_requires_and_binds_public_amendment(tmp_path):
    a, freeze = _build(tmp_path, separate_causal=True)
    for method_id in (
        "original__affine_causal_h_attention128",
        "enriched__affine_causal_h_attention128",
    ):
        configs = freeze["state_bindings"][method_id]["method_config"]
        assert configs[-1]["path"].endswith("qk_amendment.json")
        assert freeze["state_bindings"][method_id]["method_state"][0]["path"].startswith(
            "experiments/TRR-0005/joint_fit_qknorm_v1/"
        )
    missing = _assets(tmp_path / "missing_amendment", separate_causal=True)
    with pytest.raises(binder.MethodBindingError, match="attention amendment"):
        binder.build_method_freeze(
            repository_root=missing["root"],
            decision_plan_path=missing["decision"],
            output_freeze=missing["root"] / "method_freeze.json",
            output_selection=missing["root"] / "selection.json",
            fit_root=missing["fit"],
            causal_fit_root=missing["causal"],
            lens_path=missing["lens"],
            embedding_path=missing["embedding"],
            p0_checkpoint=missing["checkpoint"],
            p0_config=missing["config"],
            code_commit=COMMIT,
            code_paths=[missing["code"]],
        )
