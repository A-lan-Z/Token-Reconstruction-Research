"""Metadata-only guards for the TRR-0012 deployment package."""
from __future__ import annotations

import copy
import importlib.util
import os
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("trr0012_package_under_test", ROOT / "scripts" / "trr0012_package.py")
assert _SPEC is not None and _SPEC.loader is not None
_PACKAGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PACKAGE)

CONTRACT_PATH = ROOT / "experiments" / "TRR-0012" / "package" / "contract_v1.json"
PACKAGE_ROOT = ROOT / "outputs" / "TRR-0012" / "model-package"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_draft_contract_validates_without_model_or_tensor_load() -> None:
    report = _PACKAGE.validate_manifest(CONTRACT_PATH, package_root=PACKAGE_ROOT)
    assert report["status"] == "DRAFT_PRE_SELECTION"
    assert report["smoke_fixture"]["sha256"] == "652a67744f2dabe2ca53198f1b7fddde6cb36d414280a1a0d95d866d1e5cc840"
    assert [row["slot"] for row in report["smoke_records"]] == list(_PACKAGE.SMOKE_RECORD_ORDER)


def test_component_omission_fails_closed(tmp_path: Path) -> None:
    payload = _contract()
    payload["required_components"].pop("expanded_fixed")
    with pytest.raises(_PACKAGE.PackageError, match="required component inventory"):
        _PACKAGE.validate_manifest(_write_manifest(tmp_path, payload), package_root=PACKAGE_ROOT)


def test_component_path_escape_fails_closed(tmp_path: Path) -> None:
    payload = _contract()
    payload["required_components"]["loader_code"]["path"] = "../loader.py"
    with pytest.raises(_PACKAGE.PackageError, match="package-relative"):
        _PACKAGE.validate_manifest(_write_manifest(tmp_path, payload), package_root=PACKAGE_ROOT)


def test_incomplete_two_method_smoke_output_fails_closed(tmp_path: Path) -> None:
    payload = _contract()
    payload["smoke"]["expected_outputs"] = {"finance/public_base/000": {}}
    with pytest.raises(_PACKAGE.PackageError, match="all four ordered records"):
        _PACKAGE.validate_manifest(_write_manifest(tmp_path, payload), package_root=PACKAGE_ROOT)


def test_smoke_output_length_and_bos_are_checked(tmp_path: Path) -> None:
    payload = _contract()
    outputs = {}
    for slot in _PACKAGE.SMOKE_RECORD_ORDER:
        outputs[slot] = {
            method: {"token_ids": [128000] * 128, "tensor_sha256": "0" * 64}
            for method in _PACKAGE.PACKAGE_METHODS
        }
    outputs["pile/public_base/001"]["expanded_fixed"]["token_ids"] = [128000] * 127
    payload["smoke"]["expected_outputs"] = outputs
    with pytest.raises(_PACKAGE.PackageError, match="exactly 128"):
        _PACKAGE.validate_manifest(_write_manifest(tmp_path, payload), package_root=PACKAGE_ROOT)


def test_smoke_output_non_bos_is_rejected(tmp_path: Path) -> None:
    payload = _contract()
    outputs = {}
    for slot in _PACKAGE.SMOKE_RECORD_ORDER:
        outputs[slot] = {
            method: {"token_ids": [128000] * 128, "tensor_sha256": "0" * 64}
            for method in _PACKAGE.PACKAGE_METHODS
        }
    outputs["finance/public_base/000"]["current_fixed"]["token_ids"][0] = 1
    payload["smoke"]["expected_outputs"] = outputs
    with pytest.raises(_PACKAGE.PackageError, match="fixed BOS"):
        _PACKAGE.validate_manifest(_write_manifest(tmp_path, payload), package_root=PACKAGE_ROOT)


def test_symlinked_smoke_ancestor_is_rejected(tmp_path: Path) -> None:
    package_copy = tmp_path / "package"
    shutil.copytree(PACKAGE_ROOT, package_copy)
    (package_copy / "link").symlink_to(package_copy / "smoke", target_is_directory=True)
    payload = _contract()
    payload["required_components"]["smoke_input"]["path"] = "link/public_base_first2.safetensors"
    payload["smoke"]["fixture"]["path"] = "link/public_base_first2.safetensors"
    manifest = _write_manifest(tmp_path, payload)
    with pytest.raises(_PACKAGE.PackageError, match="symlink"):
        _PACKAGE.validate_manifest(manifest, package_root=package_copy)

def test_clean_consumer_provenance_is_package_bound() -> None:
    # Imports only the vendored modules; this deliberately does not load E or
    # any selected state, so it remains a metadata/resource preflight check.
    _PACKAGE._package_loader(PACKAGE_ROOT)
    modules = _PACKAGE._loaded_decoder_modules(PACKAGE_ROOT)
    names = {row["module"] for row in modules}
    assert "_trr0012_vendored_p09_loader" in names
    assert "token_reconstruction.trr0007_positionwise" in names
    assert all(row["relative_path"].startswith("code/") for row in modules)
    runtime = _PACKAGE._runtime_provenance(
        PACKAGE_ROOT,
        device=_PACKAGE.torch.device("cpu"),
        imported_modules=modules,
    )
    assert runtime["dependencies"] == {
        "numpy": "1.26.4",
        "safetensors": "0.7.0",
        "torch": "2.10.0",
    }
    assert runtime["imported_modules"] == modules
    assert runtime["runtime_descriptor"]["relative_path"] == "config/runtime.json"



def _timing_elapsed_values(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_seconds") and isinstance(child, (int, float)):
                yield float(child)
            yield from _timing_elapsed_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _timing_elapsed_values(child)


def test_package_timing_patch_preserves_prediction_call_order() -> None:
    import ast
    import hashlib

    source_path = ROOT / "scripts" / "trr0012_package.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    predict_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "predict_package")
    method_loops = [
        node
        for node in ast.walk(predict_fn)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "method_id"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "PACKAGE_METHODS"
    ]
    assert len(method_loops) == 1
    calls = sorted(
        (
            node
            for node in ast.walk(predict_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_predict_row_package"
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert len(calls) == 2
    first_activation = calls[0].args[2]
    remaining_activation = calls[1].args[2]
    assert isinstance(first_activation, ast.Subscript)
    assert isinstance(first_activation.slice, ast.Constant)
    assert first_activation.slice.value == 0
    assert isinstance(remaining_activation, ast.Subscript)
    assert isinstance(remaining_activation.slice, ast.Name)
    assert remaining_activation.slice.id == "row"
    assert "shared_observation_readout_load" in source
    assert "first_record_prediction" in source
    assert "remaining_record_predictions" in source
    assert "NOT_APPLICABLE" in source
    original = ROOT / "package" / "reference_code" / "trr0012_package_original_908392.py"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == (
        "9083926a0c369e3ca4361108bea043fbda79f619ff0fca2ce76b780d11bf0109"
    )


def test_package_smoke_outputs_and_timing_after_b1(tmp_path: Path) -> None:
    if os.environ.get("TRR0012_RUN_PACKAGE_SMOKE") != "1":
        pytest.skip("deferred until B1 completion and explicit short test window")
    required = (
        PACKAGE_ROOT / "receipts" / "selection_complete_before_smoke.json",
        PACKAGE_ROOT / "states" / "current_fixed.safetensors",
        PACKAGE_ROOT / "states" / "expanded_fixed.safetensors",
        PACKAGE_ROOT / "smoke" / "expected_predictions.safetensors",
    )
    if not all(path.is_file() and not path.is_symlink() for path in required):
        pytest.skip("post-B1 selected package fixture is not complete")

    device = os.environ.get("TRR0012_PACKAGE_TEST_DEVICE", "cpu")
    output = tmp_path / "smoke_predictions.safetensors"
    receipt = tmp_path / "smoke_predictions.receipt.json"
    result = _PACKAGE.predict_package(
        package_root=PACKAGE_ROOT,
        observations=PACKAGE_ROOT / "smoke" / "public_base_first2.safetensors",
        output_path=output,
        receipt_path=receipt,
        device_name=device,
    )

    with _PACKAGE.safe_open(str(required[-1]), framework="pt", device="cpu") as expected_handle:
        with _PACKAGE.safe_open(str(output), framework="pt", device="cpu") as actual_handle:
            assert set(actual_handle.keys()) == set(_PACKAGE.PACKAGE_METHODS)
            for method_id in _PACKAGE.PACKAGE_METHODS:
                assert _PACKAGE.torch.equal(
                    actual_handle.get_tensor(method_id), expected_handle.get_tensor(method_id)
                )

    timing = result["timing"]
    assert timing["method_order"] == list(_PACKAGE.PACKAGE_METHODS)
    assert timing["first_record_phase_label"].startswith("first record")
    assert timing["remaining_record_phase_label"].startswith("subsequent-record throughput")
    assert timing["shared_readout_allocation"]["attribution"].startswith("shared_")
    assert all(value >= 0.0 for value in _timing_elapsed_values(timing))
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["timing"]["file_io"]["receipt_write_passes"] == 2
    assert receipt_payload["timing"]["methods"].keys() == result["timing"]["methods"].keys()

    one_row = PACKAGE_ROOT / "smoke" / "public_base_first1.safetensors"
    if one_row.is_file() and not one_row.is_symlink():
        one_output = tmp_path / "one_row_predictions.safetensors"
        one_receipt = tmp_path / "one_row_predictions.receipt.json"
        one_result = _PACKAGE.predict_package(
            package_root=PACKAGE_ROOT,
            observations=one_row,
            output_path=one_output,
            receipt_path=one_receipt,
            device_name=device,
        )
        for method_id in _PACKAGE.PACKAGE_METHODS:
            assert one_result["timing"]["methods"][method_id]["remaining_record_predictions"]["status"] == "NOT_APPLICABLE"
