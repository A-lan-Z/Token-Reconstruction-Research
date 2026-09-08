"""Metadata-only guards for the TRR-0012 deployment package."""
from __future__ import annotations

import copy
import importlib.util
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

