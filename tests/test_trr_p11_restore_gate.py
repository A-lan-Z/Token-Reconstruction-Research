"""CPU-free metadata tests for the TRR-P11 restore boundary."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.trr_p11 import restore_gate as gate


def _write(root: Path, relative: str, data: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _record(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": gate.sha256_file(path),
    }


def _identity_payload(file_sha: str, *, label: str) -> dict:
    digest = hashlib.sha256(f"tensor:{label}".encode("utf-8")).hexdigest()
    return {
        "schema": gate.TENSOR_SCHEMA,
        "file_sha256": file_sha,
        "tensor_keys_sorted": ["tensor"],
        "tensors": {
            "tensor": {
                "dtype": "torch.uint8",
                "shape": [1],
                "sha256": digest,
            }
        },
    }


def _make_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    # The production gate rejects temporary roots.  The fixture explicitly
    # disables that production-only path check while retaining all binding
    # checks; production calls never pass this monkeypatch.
    monkeypatch.setattr(gate, "_TEMP_PARTS", set())
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir()
    secondary.mkdir()

    asset_bytes = {
        "current_fixed": b"state-current-b0",
        "expanded_fixed": b"state-expanded-b1",
        "public_readout": b"public-readout",
        "loader_code": b"loader-code",
        "decoder_code": b"decoder-code",
        "package_manifest": b"package-manifest",
        "frozen_config": b"frozen-config",
        "selection_receipt": b"",
        "smoke_input": b"smoke-input",
        "smoke_expected": b"smoke-expected",
    }
    relative_paths = {
        "current_fixed": "states/current_fixed.safetensors",
        "expanded_fixed": "states/expanded_fixed.safetensors",
        "public_readout": "readout/public_e.safetensors",
        "loader_code": "code/loader.py",
        "decoder_code": "code/decoder.py",
        "package_manifest": "package/manifest.json",
        "frozen_config": "package/frozen_config.json",
        "selection_receipt": "package/selection_receipt.json",
        "smoke_input": "smoke/smoke_input.safetensors",
        "smoke_expected": "smoke/smoke_expected.safetensors",
    }

    states: dict[str, str] = {}
    for name in ("current_fixed", "expanded_fixed", "public_readout"):
        for root in (primary, secondary):
            path = _write(root, relative_paths[name], asset_bytes[name])
            states.setdefault(name, gate.sha256_file(path))

    for name in (
        "loader_code",
        "decoder_code",
        "package_manifest",
        "frozen_config",
        "smoke_input",
        "smoke_expected",
    ):
        for root in (primary, secondary):
            _write(root, relative_paths[name], asset_bytes[name])

    receipt = {
        "schema": "token-reconstruction.trr-p11-selection-receipt.v1",
        "task_id": gate.TASK_ID,
        "status": "SELECTION_COMPLETE_BEFORE_SMOKE",
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "development_labels_used": True,
        "selected_methods": {
            "current_fixed": {
                "bank": "B0",
                "model_id": "new-b0",
                "state_sha256": states["current_fixed"],
            },
            "expanded_fixed": {
                "bank": "B1",
                "model_id": "new-b1",
                "state_sha256": states["expanded_fixed"],
            },
        },
    }
    receipt_bytes = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    asset_bytes["selection_receipt"] = receipt_bytes
    for root in (primary, secondary):
        _write(root, relative_paths["selection_receipt"], receipt_bytes)

    tensor_identity_paths: dict[str, str] = {}
    for name in ("current_fixed", "expanded_fixed", "public_readout"):
        identity_rel = f"identity/{name}.json"
        tensor_identity_paths[name] = identity_rel
        payload = _identity_payload(states[name], label=name)
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        for root in (primary, secondary):
            _write(root, identity_rel, payload_bytes)

    assets: dict[str, dict] = {}
    for name, relative in relative_paths.items():
        copies = {
            "primary": _record(primary / relative),
            "secondary": _record(secondary / relative),
        }
        item = {
            "relative_path": relative,
            "copies": copies,
        }
        if name in gate.METHOD_BANKS:
            item.update(
                {
                    "bank": gate.METHOD_BANKS[name],
                    "selected_step": 8000 if name == "current_fixed" else 13000,
                    "model_id": "new-b0" if name == "current_fixed" else "new-b1",
                }
            )
        if name in tensor_identity_paths:
            identity_rel = tensor_identity_paths[name]
            item["tensor_identity"] = {
                "relative_path": identity_rel,
                "copies": {
                    "primary": _record(primary / identity_rel),
                    "secondary": _record(secondary / identity_rel),
                },
            }
        assets[name] = item

    manifest_payload = {
        "schema": gate.SCHEMA,
        "task_id": gate.TASK_ID,
        "status": "RESTORE_PACKAGE_CANDIDATE",
        "package_id": "test-package",
        "selection": {
            "complete_before_smoke": True,
            "smoke_used_for_selection": False,
            "independent_evaluation_truth_opened": False,
            "development_labels_used": True,
            "checkpoint_reselection_after_smoke": False,
        },
        "boundaries": {
            "primary": {
                "boundary_id": "test-primary",
                "kind": "test",
                "root": str(primary),
                "st_dev": primary.stat().st_dev,
                "mount_root": str(primary),
            },
            "secondary": {
                "boundary_id": "test-secondary",
                "kind": "test",
                "root": str(secondary),
                "st_dev": secondary.stat().st_dev,
                "mount_root": str(secondary),
            },
        },
        "assets": assets,
        "smoke": {
            "truth_free": True,
            "record_order": list(gate.SMOKE_RECORD_ORDER),
            "methods": list(gate.SMOKE_METHODS),
            "input_asset": "smoke_input",
            "expected_asset": "smoke_expected",
            "selection_receipt": {
                "complete_before_smoke": True,
                "smoke_used_for_selection": False,
                "independent_evaluation_truth_opened": False,
            },
            "input_tensor_digests": {
                "activations": "1" * 64,
                "attention_mask": "4" * 64,
                "position_ids": "5" * 64,
            },
            "prediction_tensor_digests": {
                "current_fixed": "2" * 64,
                "expanded_fixed": "3" * 64,
            },
        },
    }
    manifest_path = tmp_path / "restore_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path, manifest_payload


def _validate(path: Path) -> dict:
    return gate.validate_restore_manifest(
        path,
        require_windows_boundary=False,
        require_clean_root=False,
        require_distinct_devices=False,
    )


def test_metadata_gate_validates_two_equal_copies_and_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, _ = _make_fixture(tmp_path, monkeypatch)
    report = _validate(manifest_path)
    assert report["status"] == "PASS_METADATA_BOUNDARIES"
    assert report["boundaries"]["primary"]["boundary_id"] == "test-primary"
    assert report["assets"]["current_fixed"]["copies"]["primary"]["sha256"]
    assert report["tensor_identity"]["public_readout"]["payload"]["tensors"]["tensor"]["shape"] == [1]
    assert report["tensor_files_opened"] is False
    assert report["smoke_files_opened"] is False


def test_gate_rejects_changed_copy_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, payload = _make_fixture(tmp_path, monkeypatch)
    payload["assets"]["current_fixed"]["copies"]["secondary"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(gate.RestoreGateError, match="changed"):
        _validate(manifest_path)


def test_gate_rejects_relative_path_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, payload = _make_fixture(tmp_path, monkeypatch)
    payload["assets"]["current_fixed"]["relative_path"] = "../outside.safetensors"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(gate.RestoreGateError, match="escapes"):
        _validate(manifest_path)


def test_gate_requires_explicit_closed_evaluation_truth_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, payload = _make_fixture(tmp_path, monkeypatch)
    payload["selection"]["independent_evaluation_truth_opened"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(gate.RestoreGateError, match="prohibited access"):
        _validate(manifest_path)


def test_gate_requires_selection_receipt_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, payload = _make_fixture(tmp_path, monkeypatch)
    del payload["assets"]["selection_receipt"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(gate.RestoreGateError, match="required assets"):
        _validate(manifest_path)


def test_strict_gate_requires_windows_boundary_and_distinct_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(gate.RestoreGateError, match="kind"):
        gate.validate_restore_manifest(
            manifest_path,
            require_windows_boundary=True,
            require_clean_root=False,
            require_distinct_devices=False,
        )
    with pytest.raises(gate.RestoreGateError, match="failure boundary"):
        gate.validate_restore_manifest(
            manifest_path,
            require_windows_boundary=False,
            require_clean_root=False,
            require_distinct_devices=True,
        )


def test_safe_relative_rejects_windows_absolute_and_parent_paths() -> None:
    with pytest.raises(gate.RestoreGateError, match="absolute"):
        gate._safe_relative("C:/outside/file", description="fixture")
    with pytest.raises(gate.RestoreGateError, match="escapes"):
        gate._safe_relative("a/../../outside", description="fixture")
