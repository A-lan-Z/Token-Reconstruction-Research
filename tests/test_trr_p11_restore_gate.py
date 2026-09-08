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
        "package_cli": b"package-cli",
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
        "package_cli": "code/trr0012_package.py",
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
        "package_cli",
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
        "complete_before_smoke": True,
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "development_labels_used": True,
        "selected_methods": {
            "current_fixed": {
                "bank": "B0",
                "model_id": "new-b0",
                "selected_step": 8000,
                "state_sha256": states["current_fixed"],
            },
            "expanded_fixed": {
                "bank": "B1",
                "model_id": "new-b1",
                "selected_step": 13000,
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
            "record_bindings": [
                {
                    "record_id": record_id,
                    "source_identity_sha256": "6" * 64,
                    "activation_slice_sha256": "7" * 64,
                    "attention_mask_slice_sha256": "8" * 64,
                    "position_ids_slice_sha256": "9" * 64,
                }
                for record_id in gate.SMOKE_RECORD_ORDER
            ],
            "input_asset": "smoke_input",
            "expected_asset": "smoke_expected",
            "input_key_layout": "domain_prefixed",
            "input_tensor_keys": {
                "activations": ["finance__activations", "pile__activations"],
                "attention_mask": ["finance__attention_mask", "pile__attention_mask"],
                "position_ids": ["finance__position_ids", "pile__position_ids"],
            },
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
        "consumer": {
            "entrypoint_asset": "package_cli",
            "python": "/usr/bin/python3",
            "receipt_schema": "token-reconstruction.trr0012-prediction-receipt.v1",
            "receipt_task_id": "TRR-0012",
            "observation_asset": "smoke_input",
            "output_relative_path": "runtime/smoke_restored.safetensors",
            "receipt_relative_path": "runtime/smoke_restored.receipt.json",
            "package_root_arg": "--package-root",
            "observations_arg": "--observations",
            "output_arg": "--output",
            "device_arg": "--device",
            "device": "cpu",
            "numerical_settings": {
                "argmax_dtype": "float32",
                "device": "cpu",
                "preserve_bos": True,
                "projection_dtype": "float32",
                "vocabulary_size": 128256,
            },
            "numerical_settings_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "argmax_dtype": "float32",
                        "device": "cpu",
                        "preserve_bos": True,
                        "projection_dtype": "float32",
                        "vocabulary_size": 128256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "code_relative_roots": ["code"],
            "timeout_seconds": 30,
            "dependency_id": "synthetic-dependencies",
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


def _save_safetensors(path: Path, tensors: dict[str, object]) -> Path:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path))
    return path


def _write_tensor_identity(path: Path, identity_path: Path) -> dict[str, dict]:
    from safetensors import safe_open

    tensors: dict[str, dict] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            value = handle.get_tensor(key)
            tensors[key] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": gate.tensor_digest(value),
            }
    payload = {
        "schema": gate.TENSOR_SCHEMA,
        "file_sha256": gate.sha256_file(path),
        "tensor_keys_sorted": sorted(tensors),
        "tensors": tensors,
    }
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def _smoke_descriptor_for_tensors(tensors: dict[str, object]) -> dict:
    from torch import cat

    groups = {
        "activations": [tensors["finance__activations"], tensors["pile__activations"]],
        "attention_mask": [tensors["finance__attention_mask"], tensors["pile__attention_mask"]],
        "position_ids": [tensors["finance__position_ids"], tensors["pile__position_ids"]],
    }
    merged = {key: cat(values, dim=0).contiguous() for key, values in groups.items()}
    bindings = []
    for index, record_id in enumerate(gate.SMOKE_RECORD_ORDER):
        bindings.append(
            {
                "record_id": record_id,
                "source_identity_sha256": ("a" * 64),
                "activation_slice_sha256": gate.tensor_digest(merged["activations"][index]),
                "attention_mask_slice_sha256": gate.tensor_digest(merged["attention_mask"][index]),
                "position_ids_slice_sha256": gate.tensor_digest(merged["position_ids"][index]),
            }
        )
    return {
        "truth_free": True,
        "record_order": list(gate.SMOKE_RECORD_ORDER),
        "methods": list(gate.SMOKE_METHODS),
        "input_asset": "smoke_input",
        "expected_asset": "smoke_expected",
        "input_key_layout": "domain_prefixed",
        "input_tensor_keys": {
            "activations": ["finance__activations", "pile__activations"],
            "attention_mask": ["finance__attention_mask", "pile__attention_mask"],
            "position_ids": ["finance__position_ids", "pile__position_ids"],
        },
        "record_bindings": bindings,
        "selection_receipt": {
            "complete_before_smoke": True,
            "smoke_used_for_selection": False,
            "independent_evaluation_truth_opened": False,
        },
        "input_tensor_digests": {key: gate.tensor_digest(value) for key, value in merged.items()},
        "prediction_tensor_digests": {"current_fixed": "b" * 64, "expanded_fixed": "c" * 64},
    }


def test_tensor_identity_checks_digest_shape_and_dtype(tmp_path: Path) -> None:
    import torch

    tensor_path = _save_safetensors(
        tmp_path / "state.safetensors",
        {"tensor": torch.tensor([1.0, 2.0], dtype=torch.float32)},
    )
    identity_path = tmp_path / "state.json"
    _write_tensor_identity(tensor_path, identity_path)
    receipt = gate.verify_tensor_identity(
        tensor_path,
        identity_path,
        expected_file_sha256=gate.sha256_file(tensor_path),
    )
    assert receipt["verified"] is True

    wrong_shape = json.loads(identity_path.read_text(encoding="utf-8"))
    wrong_shape["tensors"]["tensor"]["shape"] = [2, 1]
    identity_path.write_text(json.dumps(wrong_shape), encoding="utf-8")
    with pytest.raises(gate.RestoreGateError, match="shape/dtype"):
        gate.verify_tensor_identity(
            tensor_path,
            identity_path,
            expected_file_sha256=gate.sha256_file(tensor_path),
        )

    changed_path = _save_safetensors(
        tmp_path / "changed.safetensors",
        {"tensor": torch.tensor([1.0, 3.0], dtype=torch.float32)},
    )
    changed_identity = tmp_path / "changed.json"
    changed_payload = _write_tensor_identity(changed_path, changed_identity)
    changed_payload["tensors"]["tensor"]["sha256"] = "0" * 64
    changed_identity.write_text(json.dumps(changed_payload), encoding="utf-8")
    with pytest.raises(gate.RestoreGateError, match="tensor digest"):
        gate.verify_tensor_identity(
            changed_path,
            changed_identity,
            expected_file_sha256=gate.sha256_file(changed_path),
        )


def test_prefixed_smoke_identity_checks_aggregate_and_rows(tmp_path: Path) -> None:
    import torch

    tensors = {
        "finance__activations": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        "finance__attention_mask": torch.ones((2, 3), dtype=torch.bool),
        "finance__position_ids": torch.arange(3, dtype=torch.int64).repeat(2, 1),
        "pile__activations": torch.arange(24, 48, dtype=torch.float32).reshape(2, 3, 4),
        "pile__attention_mask": torch.ones((2, 3), dtype=torch.bool),
        "pile__position_ids": torch.arange(3, dtype=torch.int64).repeat(2, 1),
    }
    path = _save_safetensors(tmp_path / "smoke.safetensors", tensors)
    smoke = _smoke_descriptor_for_tensors(tensors)
    receipt = gate.verify_smoke_input_identity(path, smoke)
    assert receipt["verified"] is True
    assert receipt["key_layout"] == "domain_prefixed"
    assert receipt["record_count"] == 4

    changed = dict(tensors)
    changed["pile__activations"] = tensors["pile__activations"].clone()
    changed["pile__activations"][0, 0, 0] += 1
    changed_path = _save_safetensors(tmp_path / "smoke_changed.safetensors", changed)
    with pytest.raises(gate.RestoreGateError, match="digest"):
        gate.verify_smoke_input_identity(changed_path, smoke)


def test_prediction_identity_rejects_changed_method_tensor(tmp_path: Path) -> None:
    import torch

    expected = {
        "current_fixed": torch.tensor([[128000, 4, 5], [128000, 6, 7]], dtype=torch.int64),
        "expanded_fixed": torch.tensor([[128000, 8, 9], [128000, 10, 11]], dtype=torch.int64),
    }
    packaged = _save_safetensors(tmp_path / "packaged.safetensors", expected)
    restored = _save_safetensors(tmp_path / "restored.safetensors", expected)
    digests = {key: gate.tensor_digest(value) for key, value in expected.items()}
    receipt = gate.compare_smoke_prediction_files(
        packaged,
        restored,
        expected_tensor_digests=digests,
    )
    assert receipt["verified"] is True

    changed = dict(expected)
    changed["expanded_fixed"] = expected["expanded_fixed"].clone()
    changed["expanded_fixed"][1, 2] += 1
    changed_path = _save_safetensors(tmp_path / "restored_changed.safetensors", changed)
    with pytest.raises(gate.RestoreGateError, match="IDs differ"):
        gate.compare_smoke_prediction_files(packaged, changed_path, expected_tensor_digests=digests)


def test_consumer_receipt_binds_actual_clean_assets_and_rejects_unbound_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    manifest_path, _ = _make_fixture(tmp_path, monkeypatch)
    report = _validate(manifest_path)
    clean = tmp_path / "clean"
    primary = Path(report["boundaries"]["primary"]["root"])
    for name, asset in report["assets"].items():
        source = primary / asset["relative_path"]
        target = clean / asset["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name, identity in report["tensor_identity"].items():
        source = primary / identity["relative_path"]
        target = clean / identity["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    output = clean / "runtime" / "smoke_restored.safetensors"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"synthetic-predictions")
    receipt_path = clean / "runtime" / "smoke_restored.receipt.json"
    code_names = ("package_cli", "loader_code", "decoder_code")
    resource_names = ("current_fixed", "expanded_fixed", "public_readout", "frozen_config", "smoke_input")
    loaded_names = code_names + resource_names
    loaded_paths = {
        name: str((clean / report["assets"][name]["relative_path"]).resolve())
        for name in loaded_names
    }
    file_bindings = []
    for name in loaded_names:
        path = Path(loaded_paths[name])
        file_bindings.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": gate.sha256_file(path),
                "role": name,
            }
        )
    def binding_for(name: str) -> dict[str, object]:
        path = Path(loaded_paths[name])
        return {
            "relative_path": report["assets"][name]["relative_path"],
            "loaded_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": gate.sha256_file(path),
        }

    receipt = {
        "schema": gate.CONSUMER_RECEIPT_SCHEMA,
        "task_id": gate.CONSUMER_TASK_ID,
        "status": "PREDICTIONS_GENERATED_AFTER_SELECTION",
        "complete_before_smoke": True,
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "truth_opened": False,
        "source_text_loaded": False,
        "token_ids_loaded": False,
        "output": {
            "path": "runtime/smoke_restored.safetensors",
            "bytes": output.stat().st_size,
            "sha256": gate.sha256_file(output),
        },
        "observations": {
            "path": report["assets"]["smoke_input"]["relative_path"],
            "bytes": Path(loaded_paths["smoke_input"]).stat().st_size,
            "sha256": gate.sha256_file(Path(loaded_paths["smoke_input"])),
            "tensor_sha256": report["smoke"]["input_tensor_digests"],
            "record_order": report["smoke"]["record_order"],
        },
        "methods": {
            "current_fixed": {"state_file_binding": binding_for("current_fixed")},
            "expanded_fixed": {"state_file_binding": binding_for("expanded_fixed")},
        },
        "readout": {"file_binding": binding_for("public_readout")},
        "runtime": {
            "python": "3.12.3",
            "python_implementation": "CPython",
            "python_executable": "/usr/bin/python3",
            "dependencies": {
                "numpy": "1.26.4",
                "safetensors": "0.7.0",
                "torch": "2.10.0",
            },
            "device": "cpu",
            "numeric_profile": "qualified FP32 decoder",
            "imported_modules": [
                {**binding_for("loader_code"), "module": "synthetic_loader"},
                {**binding_for("decoder_code"), "module": "synthetic_decoder"},
            ],
        },
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    checked = gate._verify_consumer_receipt(receipt_path, report, clean, output)
    assert checked["verified"] is True
    assert checked["dependency_versions"]["torch"] == "2.10.0"

    unbound = clean / "code" / "unbound.py"
    unbound.write_bytes(b"unbound")
    bad_receipt = copy.deepcopy(receipt)
    bad_receipt["runtime"]["imported_modules"].append(
        {
            "relative_path": "code/unbound.py",
            "loaded_path": str(unbound),
            "bytes": unbound.stat().st_size,
            "sha256": gate.sha256_file(unbound),
            "module": "unbound",
        }
    )
    bad_receipt_path = clean / "runtime" / "bad.receipt.json"
    bad_receipt_path.write_text(json.dumps(bad_receipt), encoding="utf-8")
    with pytest.raises(gate.RestoreGateError, match="not a bound bundle asset"):
        gate._verify_consumer_receipt(bad_receipt_path, report, clean, output)


def _make_end_to_end_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    import shutil
    import torch

    monkeypatch.setattr(gate, "_TEMP_PARTS", set())
    primary = tmp_path / "e2e-primary"
    secondary = tmp_path / "e2e-secondary"
    clean = tmp_path / "e2e-clean"
    primary.mkdir()
    secondary.mkdir()

    asset_paths = {
        "current_fixed": "states/current_fixed.safetensors",
        "expanded_fixed": "states/expanded_fixed.safetensors",
        "public_readout": "readout/public_e.safetensors",
        "loader_code": "code/loader.py",
        "decoder_code": "code/decoder.py",
        "package_cli": "code/trr0012_package.py",
        "package_manifest": "package/manifest.json",
        "frozen_config": "package/frozen_config.json",
        "selection_receipt": "package/selection_receipt.json",
        "smoke_input": "smoke/smoke_input.safetensors",
        "smoke_expected": "smoke/smoke_expected.safetensors",
    }
    tensors = {
        "finance__activations": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        "finance__attention_mask": torch.ones((2, 3), dtype=torch.bool),
        "finance__position_ids": torch.arange(3, dtype=torch.int64).repeat(2, 1),
        "pile__activations": torch.arange(24, 48, dtype=torch.float32).reshape(2, 3, 4),
        "pile__attention_mask": torch.ones((2, 3), dtype=torch.bool),
        "pile__position_ids": torch.arange(3, dtype=torch.int64).repeat(2, 1),
    }
    _save_safetensors(primary / asset_paths["current_fixed"], {"tensor": torch.tensor([1.0, 2.0])})
    _save_safetensors(primary / asset_paths["expanded_fixed"], {"tensor": torch.tensor([3.0, 4.0])})
    _save_safetensors(primary / asset_paths["public_readout"], {"embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]])})
    _save_safetensors(primary / asset_paths["smoke_input"], tensors)
    smoke = _smoke_descriptor_for_tensors(tensors)
    expected_predictions = {
        "current_fixed": torch.tensor(
            [[128000, 1, 2], [128000, 1, 2], [128000, 1, 2], [128000, 1, 2]], dtype=torch.int64
        ),
        "expanded_fixed": torch.tensor(
            [[128000, 3, 4], [128000, 3, 4], [128000, 3, 4], [128000, 3, 4]], dtype=torch.int64
        ),
    }
    _save_safetensors(primary / asset_paths["smoke_expected"], expected_predictions)

    package_cli = '''\
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value):
    value = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def file_binding(root, path):
    path = Path(path).resolve()
    return {"relative_path": path.relative_to(root).as_posix(), "loaded_path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.command != "predict":
        raise SystemExit(2)
    root = Path(args.package_root).resolve()
    observations = Path(args.observations).resolve()
    output = Path(args.output).resolve()
    with safe_open(str(observations), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {
            "finance__activations", "finance__attention_mask", "finance__position_ids",
            "pile__activations", "pile__attention_mask", "pile__position_ids",
        }:
            raise RuntimeError("unexpected smoke keys")
        activations = torch.cat([handle.get_tensor("finance__activations"), handle.get_tensor("pile__activations")], dim=0).contiguous()
        masks = torch.cat([handle.get_tensor("finance__attention_mask"), handle.get_tensor("pile__attention_mask")], dim=0).contiguous()
        positions = torch.cat([handle.get_tensor("finance__position_ids"), handle.get_tensor("pile__position_ids")], dim=0).contiguous()
        rows = int(activations.shape[0])
        observation_tensor_sha = {
            "activations": tensor_digest(activations),
            "attention_mask": tensor_digest(masks),
            "position_ids": tensor_digest(positions),
        }
    for relative in (
        "code/trr0012_package.py", "code/loader.py", "code/decoder.py",
    ):
        (root / relative).read_bytes()
    for relative in (
        "states/current_fixed.safetensors", "states/expanded_fixed.safetensors",
        "readout/public_e.safetensors", "package/frozen_config.json",
        "package/manifest.json", "package/selection_receipt.json",
        "smoke/smoke_input.safetensors",
    ):
        (root / relative).read_bytes()
    predictions = {
        "current_fixed": torch.tensor([[128000, 1, 2]] * rows, dtype=torch.int64),
        "expanded_fixed": torch.tensor([[128000, 3, 4]] * rows, dtype=torch.int64),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(predictions, str(output))
    code_paths = [root / relative for relative in (
        "code/trr0012_package.py", "code/loader.py", "code/decoder.py",
    )]
    resource_paths = [root / relative for relative in (
        "states/current_fixed.safetensors", "states/expanded_fixed.safetensors",
        "readout/public_e.safetensors", "package/frozen_config.json", "smoke/smoke_input.safetensors",
    )]
    state_bindings = {
        "current_fixed": file_binding(root, resource_paths[0]),
        "expanded_fixed": file_binding(root, resource_paths[1]),
    }
    receipt = {
        "schema": "token-reconstruction.trr0012-prediction-receipt.v1",
        "task_id": "TRR-0012",
        "status": "PREDICTIONS_GENERATED_AFTER_SELECTION",
        "complete_before_smoke": True,
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "truth_opened": False,
        "source_text_loaded": False,
        "token_ids_loaded": False,
        "output": {"path": output.relative_to(root).as_posix(), "bytes": output.stat().st_size, "sha256": sha256_file(output)},
        "observations": {
            "path": observations.relative_to(root).as_posix(),
            "bytes": observations.stat().st_size,
            "sha256": sha256_file(observations),
            "tensor_sha256": observation_tensor_sha,
            "record_order": [
                "finance/public_base/000", "finance/public_base/001",
                "pile/public_base/000", "pile/public_base/001",
            ],
        },
        "methods": {
            "current_fixed": {"state_file_binding": state_bindings["current_fixed"]},
            "expanded_fixed": {"state_file_binding": state_bindings["expanded_fixed"]},
        },
        "readout": {"file_binding": file_binding(root, resource_paths[2])},
        "runtime": {
            "python": "3.12.3",
            "python_implementation": "CPython",
            "python_executable": sys.executable,
            "dependencies": {"numpy": "1.26.4", "safetensors": "0.7.0", "torch": "2.10.0"},
            "device": args.device,
            "numeric_profile": "qualified FP32 decoder",
            "imported_modules": [
                {**file_binding(root, code_paths[1]), "module": "synthetic_loader"},
                {**file_binding(root, code_paths[2]), "module": "synthetic_decoder"},
            ],
        },
    }
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
'''
    _write(primary, asset_paths["package_cli"], package_cli.encode())
    _write(primary, asset_paths["loader_code"], b"# synthetic loader\n")
    _write(primary, asset_paths["decoder_code"], b"# synthetic decoder\n")
    _write(primary, asset_paths["package_manifest"], b"{\"schema\":\"synthetic-package\"}\n")
    _write(primary, asset_paths["frozen_config"], b"{\"device\":\"cpu\"}\n")

    states = {
        "current_fixed": gate.sha256_file(primary / asset_paths["current_fixed"]),
        "expanded_fixed": gate.sha256_file(primary / asset_paths["expanded_fixed"]),
    }
    selection_receipt = {
        "schema": gate.SELECTION_RECEIPT_SCHEMA,
        "task_id": gate.TASK_ID,
        "status": "SELECTION_COMPLETE_BEFORE_SMOKE",
        "complete_before_smoke": True,
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "development_labels_used": True,
        "selected_methods": {
            "current_fixed": {"bank": "B0", "model_id": "TRR-0012/current_fixed", "selected_step": 8, "state_sha256": states["current_fixed"]},
            "expanded_fixed": {"bank": "B1", "model_id": "TRR-0012/expanded_fixed", "selected_step": 13, "state_sha256": states["expanded_fixed"]},
        },
    }
    _write(primary, asset_paths["selection_receipt"], (json.dumps(selection_receipt, sort_keys=True) + "\n").encode())
    tensor_identity_paths = {}
    for name in ("current_fixed", "expanded_fixed", "public_readout"):
        relative = f"identity/{name}.json"
        tensor_identity_paths[name] = relative
        _write_tensor_identity(primary / asset_paths[name], primary / relative)
    shutil.copytree(primary, secondary, dirs_exist_ok=True)

    assets = {}
    for name, relative in asset_paths.items():
        item = {
            "relative_path": relative,
            "copies": {
                "primary": _record(primary / relative),
                "secondary": _record(secondary / relative),
            },
        }
        if name in gate.METHOD_BANKS:
            item.update({
                "bank": gate.METHOD_BANKS[name],
                "selected_step": 8 if name == "current_fixed" else 13,
                "model_id": "TRR-0012/current_fixed" if name == "current_fixed" else "TRR-0012/expanded_fixed",
            })
        if name in tensor_identity_paths:
            identity_relative = tensor_identity_paths[name]
            item["tensor_identity"] = {
                "relative_path": identity_relative,
                "copies": {
                    "primary": _record(primary / identity_relative),
                    "secondary": _record(secondary / identity_relative),
                },
            }
        assets[name] = item
    numerical_settings = {
        "argmax_dtype": "int64",
        "device": "cpu",
        "preserve_bos": True,
        "projection_dtype": "float32",
        "vocabulary_size": 128256,
    }
    manifest = {
        "schema": gate.SCHEMA,
        "task_id": gate.TASK_ID,
        "status": "RESTORE_PACKAGE_CANDIDATE",
        "package_id": "synthetic-e2e",
        "selection": {
            "complete_before_smoke": True,
            "smoke_used_for_selection": False,
            "independent_evaluation_truth_opened": False,
            "development_labels_used": True,
            "checkpoint_reselection_after_smoke": False,
        },
        "boundaries": {
            "primary": {"boundary_id": "synthetic-primary", "kind": "test", "root": str(primary), "st_dev": primary.stat().st_dev, "mount_root": str(primary)},
            "secondary": {"boundary_id": "synthetic-secondary", "kind": "test", "root": str(secondary), "st_dev": secondary.stat().st_dev, "mount_root": str(secondary)},
        },
        "assets": assets,
        "smoke": {
            **smoke,
            "prediction_tensor_digests": {key: gate.tensor_digest(value) for key, value in expected_predictions.items()},
        },
        "consumer": {
            "entrypoint_asset": "package_cli",
            "python": "/usr/bin/python3",
            "receipt_schema": gate.CONSUMER_RECEIPT_SCHEMA,
            "receipt_task_id": gate.CONSUMER_TASK_ID,
            "observation_asset": "smoke_input",
            "output_relative_path": "runtime/smoke_restored.safetensors",
            "receipt_relative_path": "runtime/smoke_restored.receipt.json",
            "package_root_arg": "--package-root",
            "observations_arg": "--observations",
            "output_arg": "--output",
            "device_arg": "--device",
            "device": "cpu",
            "numerical_settings": numerical_settings,
            "numerical_settings_sha256": hashlib.sha256(json.dumps(numerical_settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "code_relative_roots": ["code"],
            "timeout_seconds": 30,
            "dependency_id": "synthetic-e2e-dependencies",
        },
        "clean_runtime": {"root": str(clean), "training_worktree": False, "temporary": False},
    }
    manifest_path = tmp_path / "e2e-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, clean


def test_restore_and_run_smoke_end_to_end_synthetic_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, clean = _make_end_to_end_bundle(tmp_path, monkeypatch)
    receipt = gate.restore_and_run_smoke(
        manifest_path,
        clean,
        require_windows_boundary=False,
        require_distinct_devices=False,
    )
    assert receipt["status"] == "PASS_RESTORED_SMOKE"
    assert receipt["retrieval"]["source_boundary"] == "secondary"
    assert receipt["tensor_identity"]["current_fixed"]["verified"] is True
    assert receipt["smoke_input"]["key_layout"] == "domain_prefixed"
    assert receipt["smoke_comparison"]["verified"] is True
    assert receipt["consumer_receipt"]["verified"] is True
