from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from token_reconstruction.footing import (
    BOS_TOKEN_ID,
    CONDITION_ORDER,
    REGISTRATION_SCHEMA,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PANEL_SCHEMA,
    PREDICTION_SCHEMA,
    STYLE_ORDER,
    FootingError,
    build_truth_binding,
    external_file_record,
    expected_cell_ids,
    expected_prediction_path,
    file_record,
    load_all_cells,
    load_method_registry,
    load_panel,
    make_binding,
    sha256_file,
    truth_sidecar_metadata,
    validate_before_truth,
    validate_truth_sidecar,
    validate_complete_prediction_set,
    validate_prediction_artifact,
)
from token_reconstruction.freeze import FreezeError, create_freeze_receipt, verify_freeze_receipt


COMMIT = "a" * 40
METHODS = ("method_a", "method_b", "method_c")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _public_asset(path: Path, *, root: Path) -> dict[str, object]:
    return {
        **file_record(path, repository_root=root),
        "tensor_key": "activations",
        "row_indices": list(range(8)),
    }


def _panel_fixture(tmp_path: Path) -> tuple[Path, dict, Path]:
    root = tmp_path
    obs_root = root / "observations"
    obs_root.mkdir()
    pile = torch.zeros((32, 40, HIDDEN_SIZE), dtype=torch.bfloat16)
    finance = torch.zeros((32, 128, HIDDEN_SIZE), dtype=torch.bfloat16)
    pile_path = obs_root / "pile.safetensors"
    finance_path = obs_root / "finance.safetensors"
    save_file({"activations": pile}, str(pile_path))
    save_file({"activations": finance}, str(finance_path))

    pile_ids = [f"pile-{index}" for index in range(8)]
    finance_ids = [f"finance-{index}" for index in range(8)]
    pile_records = [
        {"record_id": value, "public_record_sha256": f"{index:064x}"}
        for index, value in enumerate(pile_ids)
    ]
    finance_records = [
        {
            "record_id": value,
            "public_record_sha256": f"{index + 8:064x}",
            "tokenized_record_sha256": f"{index + 16:064x}",
            "raw_index": index,
            "valid_tokens": 6 + index,
        }
        for index, value in enumerate(finance_ids)
    ]
    pile_mask = [[1] * 40 for _ in range(8)]
    pile_positions = [list(range(40)) for _ in range(8)]
    finance_mask = []
    finance_positions = []
    for index in range(8):
        valid = 6 + index
        finance_mask.append([1] * valid + [0] * (128 - valid))
        finance_positions.append(list(range(valid)) + [valid - 1] * (128 - valid))

    def cell(style: str, condition: str, records, mask, positions, asset, sequence_tokens: int):
        return {
            "id": f"{style}__{condition}",
            "style": style,
            "condition": condition,
            "shift_role": (
                "matched_public_control"
                if condition == "public_base"
                else "single_public_shift_diagnostic"
            ),
            "records": records,
            "attention_mask": mask,
            "position_ids": positions,
            "observation": asset,
            "geometry": {
                "records": 8,
                "sequence_tokens": sequence_tokens,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
            },
        }

    panel = {
        "schema": PANEL_SCHEMA,
        "task_id": "TRR-0003",
        "status": "RETROSPECTIVE_DEVELOPMENT_PANEL",
        "panel_id": "fixture",
        "source_material_included": False,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "cut_depth": CUT_DEPTH,
        "hidden_size": HIDDEN_SIZE,
        "styles": [
            {"id": "pile", "records": 8, "sequence_tokens": 40, "hidden_size": HIDDEN_SIZE},
            {"id": "finance", "records": 8, "sequence_tokens": 128, "hidden_size": HIDDEN_SIZE},
        ],
        "conditions": [
            {"id": "public_base", "role": "matched_public_control"},
            {"id": "public_lora_2601", "role": "one synthetic public target shift diagnostic"},
        ],
        "cells": [
            cell("pile", condition, pile_records, pile_mask, pile_positions, _public_asset(pile_path, root=root), 40)
            for condition in CONDITION_ORDER
        ]
        + [
            cell("finance", condition, finance_records, finance_mask, finance_positions, _public_asset(finance_path, root=root), 128)
            for condition in CONDITION_ORDER
        ],
        "method_output_contract": {
            "method_ids": list(METHODS),
            "all_cells_required_before_evaluation": True,
        },
        "canonical_status": {"new_track_a_methods": "NOT_RUN", "new_track_b_methods": "NOT_RUN"},
    }
    panel_path = root / "panel.json"
    _write_json(panel_path, panel)
    loaded = load_panel(panel_path, repository_root=root)
    return panel_path, loaded, root


def _prediction_set(
    *, panel_path: Path, panel: dict, root: Path
) -> tuple[Path, dict[str, dict], dict]:
    output = root / "predictions"
    output.mkdir(exist_ok=True)
    code = root / "runner.py"
    code.write_text("print('fixture')\n", encoding="utf-8")
    bindings: dict[str, dict] = {}
    for method in METHODS:
        state = root / f"{method}.safetensors"
        state.write_bytes(method.encode())
        bindings[method] = make_binding(
            panel_path=panel_path,
            repository_root=root,
            method_state_paths=[state],
            code_paths=[code],
            code_commit=COMMIT,
        )
    panel_sha = sha256_file(panel_path)
    for cell in load_all_cells(panel, repository_root=root):
        for method in METHODS:
            path = expected_prediction_path(output, cell=cell, method_id=method)
            path.parent.mkdir(parents=True, exist_ok=True)
            prediction = torch.full(cell.attention_mask.shape, -1, dtype=torch.int64)
            active = cell.attention_mask.to(torch.bool)
            prediction[active] = 1
            prediction[:, 0] = BOS_TOKEN_ID
            metadata = {
                "schema": PREDICTION_SCHEMA,
                "task_id": "TRR-0003",
                "panel_sha256": panel_sha,
                "cell_id": cell.cell_id,
                "style": cell.style,
                "condition": cell.condition,
                "method_id": method,
                "geometry_json": json.dumps(
                    {"records": 8, "sequence_tokens": cell.sequence_tokens, "hidden_size": HIDDEN_SIZE, "cut_depth": CUT_DEPTH},
                    sort_keys=True,
                ),
                "binding_json": json.dumps(bindings[method], sort_keys=True),
            }
            save_file({"predictions": prediction}, str(path), metadata=metadata)
    return output, bindings, {"panel_sha256": panel_sha, "method_ids": list(METHODS)}


def test_paired_panel_load_and_complete_prediction_set(tmp_path: Path) -> None:
    panel_path, panel, root = _panel_fixture(tmp_path)
    output, bindings, _ = _prediction_set(panel_path=panel_path, panel=panel, root=root)
    validated = validate_complete_prediction_set(
        output,
        panel=panel,
        panel_path=panel_path,
        repository_root=root,
        method_ids=METHODS,
        expected_bindings=bindings,
    )
    assert len(validated) == 4 * len(METHODS)
    assert expected_cell_ids() == tuple(cell.cell_id for cell in load_all_cells(panel, repository_root=root))


def test_prediction_set_rejects_missing_extra_and_oov(tmp_path: Path) -> None:
    panel_path, panel, root = _panel_fixture(tmp_path)
    output, bindings, _ = _prediction_set(panel_path=panel_path, panel=panel, root=root)
    missing = expected_prediction_path(
        output,
        cell=load_all_cells(panel, repository_root=root)[0],
        method_id=METHODS[0],
    )
    missing.unlink()
    with pytest.raises(FootingError, match="unavailable|incomplete"):
        validate_complete_prediction_set(
            output, panel=panel, panel_path=panel_path, repository_root=root,
            method_ids=METHODS, expected_bindings=bindings,
        )

    # Recreate the missing artifact, then add an unregistered safetensors file.
    _prediction_set(panel_path=panel_path, panel=panel, root=root)
    extra = output / "extra.safetensors"
    save_file({"predictions": torch.zeros((1, 1), dtype=torch.int64)}, str(extra))
    with pytest.raises(FootingError, match="incomplete"):
        validate_complete_prediction_set(
            output, panel=panel, panel_path=panel_path, repository_root=root,
            method_ids=METHODS, expected_bindings=bindings,
        )

    # A complete artifact with an out-of-vocabulary active prediction is also invalid.
    _prediction_set(panel_path=panel_path, panel=panel, root=root)
    cell = load_all_cells(panel, repository_root=root)[0]
    path = expected_prediction_path(output, cell=cell, method_id=METHODS[0])
    prediction = torch.full(cell.attention_mask.shape, -1, dtype=torch.int64)
    prediction[cell.attention_mask.to(torch.bool)] = 128256
    prediction[:, 0] = BOS_TOKEN_ID
    metadata = {
        "schema": PREDICTION_SCHEMA,
        "task_id": "TRR-0003",
        "panel_sha256": sha256_file(panel_path),
        "cell_id": cell.cell_id,
        "style": cell.style,
        "condition": cell.condition,
        "method_id": METHODS[0],
        "geometry_json": json.dumps({"records": 8, "sequence_tokens": cell.sequence_tokens, "hidden_size": HIDDEN_SIZE, "cut_depth": CUT_DEPTH}, sort_keys=True),
        "binding_json": json.dumps(bindings[METHODS[0]], sort_keys=True),
    }
    save_file({"predictions": prediction}, str(path), metadata=metadata)
    with pytest.raises(FootingError, match="out-of-vocabulary"):
        validate_prediction_artifact(
            path, cell=cell, panel_sha256=sha256_file(panel_path),
            expected_method_id=METHODS[0], expected_binding=bindings[METHODS[0]],
        )


def test_freeze_verifier_rejects_unlisted_and_traversal_entries(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    frozen = tmp_path / "bundle"
    frozen.mkdir()
    artifact = frozen / "predictions.json"
    artifact.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    create_freeze_receipt(
        repository_root=tmp_path,
        frozen_root=frozen,
        plan_path=plan,
        receipt_path=receipt,
        preregistration_commit=COMMIT,
        created_utc="2026-09-05T00:00:00Z",
    )
    (frozen / "unlisted.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FreezeError, match="artifact set changed"):
        verify_freeze_receipt(receipt, repository_root=tmp_path)

    receipt.chmod(0o644)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["entries"][0]["path"] = "../plan.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FreezeError, match="unsafe"):
        verify_freeze_receipt(receipt, repository_root=tmp_path)


def test_truth_gate_binds_receipt_output_panel_and_state(tmp_path: Path) -> None:
    panel_path, panel, root = _panel_fixture(tmp_path)
    output, bindings, metadata = _prediction_set(panel_path=panel_path, panel=panel, root=root)
    plan = root / "plan.json"
    plan.write_text("{\"task_id\":\"TRR-0003\"}\n", encoding="utf-8")
    receipt = root / "receipt.json"
    truth = root / "private" / "answers.safetensors"
    truth.parent.mkdir()
    truth.write_bytes(b"private fixture")
    create_freeze_receipt(
        repository_root=root,
        frozen_root=output,
        plan_path=plan,
        receipt_path=receipt,
        preregistration_commit=COMMIT,
        created_utc="2026-09-05T00:00:00Z",
        metadata=metadata,
    )
    validate_before_truth(
        receipt_path=receipt,
        repository_root=root,
        truth_path=truth,
        output_root=output,
        panel=panel,
        panel_path=panel_path,
        method_ids=METHODS,
        expected_bindings=bindings,
    )

    state_path = root / "method_a.safetensors"
    state_path.write_bytes(b"changed state")
    with pytest.raises(FootingError, match="hash or size changed|binding"):
        validate_before_truth(
            receipt_path=receipt,
            repository_root=root,
            truth_path=truth,
            output_root=output,
            panel=panel,
            panel_path=panel_path,
            method_ids=METHODS,
            expected_bindings=bindings,
        )

    with pytest.raises(FootingError, match="output root"):
        validate_before_truth(
            receipt_path=receipt,
            repository_root=root,
            truth_path=truth,
            output_root=root / "other-output",
            panel=panel,
            panel_path=panel_path,
            method_ids=METHODS,
            expected_bindings=bindings,
        )

    panel_path.chmod(0o644)
    panel_path.write_text(panel_path.read_text(encoding="utf-8").replace('"fixture"', '"changed"'), encoding="utf-8")
    with pytest.raises(FootingError, match="panel"):
        validate_before_truth(
            receipt_path=receipt,
            repository_root=root,
            truth_path=truth,
            output_root=output,
            panel=panel,
            panel_path=panel_path,
            method_ids=METHODS,
            expected_bindings=bindings,
        )


def test_comparator_score_does_not_call_truth_loader_on_gate_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    import sys

    panel_path, panel, root = _panel_fixture(tmp_path)
    output, bindings, metadata = _prediction_set(panel_path=panel_path, panel=panel, root=root)
    (output / "bindings.json").write_text(json.dumps(bindings, sort_keys=True) + "\n", encoding="utf-8")
    plan = root / "plan.json"
    plan.write_text("{\"task_id\":\"TRR-0003\"}\n", encoding="utf-8")
    receipt = root / "receipt.json"
    truth = root / "private" / "truth.safetensors"
    truth.parent.mkdir()
    truth.write_bytes(b"private fixture")
    create_freeze_receipt(
        repository_root=root,
        frozen_root=output,
        plan_path=plan,
        receipt_path=receipt,
        preregistration_commit=COMMIT,
        created_utc="2026-09-05T00:00:00Z",
        metadata=metadata,
    )
    expected = expected_prediction_path(
        output,
        cell=load_all_cells(panel, repository_root=root)[0],
        method_id=METHODS[0],
    )
    expected.unlink()

    module_path = Path("scripts/trr0003_footing_compare.py").resolve()
    spec = importlib.util.spec_from_file_location("trr0003_footing_compare_test", module_path)
    assert spec is not None and spec.loader is not None
    compare = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compare
    spec.loader.exec_module(compare)
    monkeypatch.setattr(compare, "METHOD_IDS", METHODS)
    called = False

    def fail_truth(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("truth loader was called before the gate")

    monkeypatch.setattr(compare, "_truth_for_cells", fail_truth)
    args = type("Args", (), {
        "repository_root": root,
        "output": output,
        "panel": panel_path,
        "receipt": receipt,
        "truth": truth,
        "result": root / "result.json",
    })()
    with pytest.raises((FootingError, compare.ComparatorError), match="incomplete|unavailable|freeze"):
        compare.score(args)
    assert called is False


def _truth_sidecar_fixture(panel_path: Path, panel: dict, root: Path):
    """Create a deterministic private-sidecar fixture and its commitment."""
    cells = load_all_cells(panel, repository_root=root)
    prep = root / "private" / "truth_preparation.json"
    prep.parent.mkdir(exist_ok=True)
    _write_json(prep, {"task_id": "TRR-0003", "role": "private sidecar preparation", "version": 1})
    prep_record = external_file_record(prep)
    sidecar = root / "private" / "panel_truth.safetensors"
    truth = {}
    tensors = {}
    for cell in cells:
        value = torch.ones(cell.attention_mask.shape, dtype=torch.int32)
        value[:, 0] = BOS_TOKEN_ID
        truth[cell.cell_id] = value
        tensors[f"{cell.cell_id}__token_ids"] = value
        tensors[f"{cell.cell_id}__attention_mask"] = cell.attention_mask.to(torch.int32)
        tensors[f"{cell.cell_id}__position_ids"] = cell.position_ids.to(torch.int32)
    placeholder = {"path": str(sidecar.resolve()), "bytes": 0, "sha256": "0" * 64}
    binding = build_truth_binding(
        panel_sha256=sha256_file(panel_path),
        cells=cells,
        truth=truth,
        preparation=prep_record,
        sidecar=placeholder,
    )
    save_file(tensors, str(sidecar), metadata=truth_sidecar_metadata(binding))
    # The sidecar metadata intentionally omits its own hash.  Record the
    # committed bytes after the single write; rewriting a safetensors file can
    # change metadata ordering even when its logical metadata is unchanged.
    binding["sidecar"] = external_file_record(sidecar)
    assert external_file_record(sidecar) == binding["sidecar"]
    return binding, truth, sidecar


def _registry_fixture(
    *,
    panel_path: Path,
    output: Path,
    root: Path,
    bindings: dict[str, dict],
    truth_binding: dict | None = None,
    methods: tuple[str, ...] = METHODS,
) -> Path:
    rows = []
    tracks = ("comparator", "track_a", "track_b")
    for index, method in enumerate(methods):
        rows.append(
            {
                "id": method,
                "track": tracks[index % len(tracks)],
                "candidate_policy": "optional",
                "binding": bindings[method],
                "bundle": {"layout": "canonical", "root": output.relative_to(root).as_posix()},
            }
        )
    payload = {
        "schema": REGISTRATION_SCHEMA,
        "task_id": "TRR-0003",
        "status": "EXPLORATORY_METHOD_BUNDLE_REGISTRATION",
        "panel": file_record(panel_path, repository_root=root),
        "method_ids": list(methods),
        "methods": rows,
    }
    if truth_binding is not None:
        payload["truth_binding"] = truth_binding
    path = root / "method_registry.json"
    _write_json(path, payload)
    return path


def test_candidate_policy_allows_standalone_without_candidates(tmp_path: Path) -> None:
    panel_path, panel, root = _panel_fixture(tmp_path)
    output, bindings, _ = _prediction_set(panel_path=panel_path, panel=panel, root=root)
    policies = {method: "optional" for method in METHODS}
    validate_complete_prediction_set(
        output,
        panel=panel,
        panel_path=panel_path,
        repository_root=root,
        method_ids=METHODS,
        expected_bindings=bindings,
        candidate_policies=policies,
    )
    required = dict(policies)
    required[METHODS[0]] = "required"
    with pytest.raises(FootingError, match="candidate tensors are required"):
        validate_complete_prediction_set(
            output,
            panel=panel,
            panel_path=panel_path,
            repository_root=root,
            method_ids=METHODS,
            expected_bindings=bindings,
            candidate_policies=required,
        )


def test_registered_merge_validates_all_bundles_and_writes_common_matrix(tmp_path: Path) -> None:
    import importlib.util
    import sys

    panel_path, panel, root = _panel_fixture(tmp_path)
    source, bindings, _ = _prediction_set(panel_path=panel_path, panel=panel, root=root)
    registration = _registry_fixture(
        panel_path=panel_path,
        output=source,
        root=root,
        bindings=bindings,
    )
    module_path = Path("scripts/trr0003_footing_compare.py").resolve()
    spec = importlib.util.spec_from_file_location("trr0003_footing_compare_merge_test", module_path)
    assert spec is not None and spec.loader is not None
    compare = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compare
    spec.loader.exec_module(compare)
    merged = root / "merged"
    args = type("Args", (), {"repository_root": root, "panel": panel_path, "registration": registration, "output": merged})()
    assert compare.merge(args) == 0
    assert sha256_file(merged / "registration.json") == sha256_file(registration)
    embedded = load_method_registry(
        merged / "registration.json",
        repository_root=root,
        panel=panel,
        panel_path=panel_path,
        require_truth_binding=False,
    )
    validate_complete_prediction_set(
        merged,
        panel=panel,
        panel_path=panel_path,
        repository_root=root,
        method_ids=embedded["method_ids"],
        expected_bindings=embedded["bindings"],
        candidate_policies=embedded["candidate_policies"],
    )
    assert (merged / "merge_evidence.json").is_file()


def test_truth_sidecar_commitment_checks_paired_masks_and_token_order(tmp_path: Path) -> None:
    panel_path, panel, root = _panel_fixture(tmp_path)
    binding, truth, sidecar = _truth_sidecar_fixture(panel_path, panel, root)
    cells = load_all_cells(panel, repository_root=root)
    loaded = validate_truth_sidecar(sidecar, cells=cells, truth_binding=binding)
    assert all(torch.equal(loaded[cell.cell_id], truth[cell.cell_id]) for cell in cells)

    # Preserve metadata and token digests, but alter a paired public mask.  The
    # updated sidecar hash proves that the test reaches the post-gate pairing
    # check rather than failing merely on stale file bytes.
    bad_mask = cells[0].attention_mask.to(torch.int32).clone()
    bad_mask[0, 1] = 0
    tensors = {}
    for cell in cells:
        tensors[f"{cell.cell_id}__token_ids"] = truth[cell.cell_id]
        tensors[f"{cell.cell_id}__attention_mask"] = (
            bad_mask if cell.cell_id == cells[0].cell_id else cell.attention_mask.to(torch.int32)
        )
        tensors[f"{cell.cell_id}__position_ids"] = cell.position_ids.to(torch.int32)
    save_file(tensors, str(sidecar), metadata=truth_sidecar_metadata(binding))
    bad_binding = dict(binding)
    bad_binding["sidecar"] = external_file_record(sidecar)
    with pytest.raises(FootingError, match="truth mask pairing changed"):
        validate_truth_sidecar(sidecar, cells=cells, truth_binding=bad_binding)


def test_common_registered_gate_blocks_incomplete_other_track_before_truth_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import sys

    panel_path, panel, root = _panel_fixture(tmp_path)
    output, bindings, _ = _prediction_set(panel_path=panel_path, panel=panel, root=root)
    binding, _, truth = _truth_sidecar_fixture(panel_path, panel, root)
    extra_method = "other_track_missing"
    extra_state = root / f"{extra_method}.safetensors"
    extra_state.write_bytes(b"extra")
    code = root / "runner.py"
    full_bindings = dict(bindings)
    full_bindings[extra_method] = make_binding(
        panel_path=panel_path,
        repository_root=root,
        method_state_paths=[extra_state],
        code_paths=[code],
        code_commit=COMMIT,
    )
    methods = METHODS + (extra_method,)
    registration = _registry_fixture(
        panel_path=panel_path,
        output=output,
        root=root,
        bindings=full_bindings,
        truth_binding=binding,
        methods=methods,
    )
    # The embedded registration is part of the frozen common root.  Its source
    # root is the same root, so the fourth method is deliberately missing from
    # the Cartesian prediction set.
    (output / "bindings.json").write_text(json.dumps(full_bindings, sort_keys=True) + "\n", encoding="utf-8")
    shutil = __import__("shutil")
    shutil.copyfile(registration, output / "registration.json")
    plan = root / "plan.json"
    plan.write_text("{\"task_id\":\"TRR-0003\"}\n", encoding="utf-8")
    receipt = root / "receipt.json"
    create_freeze_receipt(
        repository_root=root,
        frozen_root=output,
        plan_path=plan,
        receipt_path=receipt,
        preregistration_commit=COMMIT,
        created_utc="2026-09-05T00:00:00Z",
        metadata={
            "task_id": "TRR-0003",
            "panel_sha256": sha256_file(panel_path),
            "method_ids": list(methods),
            "output_contract": "style__condition__method Cartesian complete",
            "truth_opened": False,
            "registration_sha256": sha256_file(output / "registration.json"),
            "candidate_policies": {method: "optional" for method in methods},
            "truth_binding": binding,
        },
    )
    module_path = Path("scripts/trr0003_footing_compare.py").resolve()
    spec = importlib.util.spec_from_file_location("trr0003_footing_compare_incomplete_test", module_path)
    assert spec is not None and spec.loader is not None
    compare = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compare
    spec.loader.exec_module(compare)
    called = False

    def fail_truth(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("truth loader was called before complete cross-track validation")

    monkeypatch.setattr(compare, "_truth_for_cells", fail_truth)
    args = type("Args", (), {
        "repository_root": root,
        "output": output,
        "panel": panel_path,
        "receipt": receipt,
        "registration": None,
        "truth": truth,
        "result": root / "result.json",
    })()
    with pytest.raises((FootingError, compare.ComparatorError), match="incomplete|unavailable"):
        compare.score(args)
    assert called is False
