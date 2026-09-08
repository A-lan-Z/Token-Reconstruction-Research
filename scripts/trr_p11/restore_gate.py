"""Fail-closed TRR-P11 package restore gate.

The gate is intentionally separate from fitting, panel selection, truth loading,
and scoring.  It validates persistent artifact copies and their metadata.  The
optional safetensors checks are lazy so a metadata-only validation does not
import a model runtime or consume the shared accelerator/CPU window.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import os
import subprocess
import time
from typing import Any

SCHEMA = "token-reconstruction.trr-p11-restore-gate.v1"
TENSOR_SCHEMA = "token-reconstruction.trr-p11-tensor-identity.v1"
SELECTION_RECEIPT_SCHEMA = "token-reconstruction.trr-p11-selection-receipt.v1"
SELECTION_RECEIPT_SCHEMAS = {
    SELECTION_RECEIPT_SCHEMA,
    "token-reconstruction.trr0012-selection-receipt.v1",
}
TASK_ID = "TRR-P11"
PRIMARY_KIND = "wsl_persistent"
SECONDARY_KIND = "windows_persistent"
BOUNDARY_NAMES = ("primary", "secondary")
METHOD_BANKS = {"current_fixed": "B0", "expanded_fixed": "B1"}
REQUIRED_ASSETS = (
    "current_fixed",
    "expanded_fixed",
    "public_readout",
    "loader_code",
    "decoder_code",
    "package_cli",
    "package_manifest",
    "frozen_config",
    "selection_receipt",
    "smoke_input",
    "smoke_expected",
)
CONSUMER_PYTHON = "/usr/bin/python3"
CONSUMER_RECEIPT_SCHEMA = "token-reconstruction.trr0012-prediction-receipt.v1"
CONSUMER_TASK_ID = "TRR-0012"
TENSOR_ASSETS = ("current_fixed", "expanded_fixed", "public_readout")
SMOKE_RECORD_ORDER = (
    "finance/public_base/000",
    "finance/public_base/001",
    "pile/public_base/000",
    "pile/public_base/001",
)
SMOKE_METHODS = ("current_fixed", "expanded_fixed")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_TRUE_KEYS = {
    "independent_evaluation_truth_opened",
    "smoke_used_for_selection",
    "private_truth_loaded",
    "scorer_opened",
}
_TEMP_PARTS = {"tmp", "temp", "temporary"}
_TRAINING_WORKTREE_PARTS = {"TRR-0011", "TRR-0012", "TRR-0013"}
_SMOKE_INPUT_KEYS = ("activations", "attention_mask", "position_ids")
_SMOKE_INPUT_LAYOUTS = {
    "standard": {
        "activations": ("activations",),
        "attention_mask": ("attention_mask",),
        "position_ids": ("position_ids",),
    },
    "domain_prefixed": {
        "activations": ("finance__activations", "pile__activations"),
        "attention_mask": ("finance__attention_mask", "pile__attention_mask"),
        "position_ids": ("finance__position_ids", "pile__position_ids"),
    },
}


class RestoreGateError(ValueError):
    """Raised when a restore contract fails closed."""


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RestoreGateError(f"asset is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Public hash helper used when a packager creates a binding."""
    return _sha256_file(Path(path).expanduser().resolve())


def _walk_forbidden_flags(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_TRUE_KEYS and _is_true(item):
                raise RestoreGateError(f"{location}.{key_text} records prohibited access")
            _walk_forbidden_flags(item, location=f"{location}.{key_text}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _walk_forbidden_flags(item, location=f"{location}[{index}]")


def _safe_relative(value: Any, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RestoreGateError(f"{description} relative_path is absent")
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or re.match(r"^[A-Za-z]:/", normalized):
        raise RestoreGateError(f"{description} relative_path is absolute")
    parts = tuple(part for part in pure.parts if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise RestoreGateError(f"{description} relative_path escapes its copy root")
    if any("\x00" in part for part in parts):
        raise RestoreGateError(f"{description} relative_path contains NUL")
    return Path(*parts)


def _resolved_path(value: Any, *, description: str, directory: bool | None = None) -> Path:
    if not isinstance(value, (str, Path)) or not value:
        raise RestoreGateError(f"{description} path is absent")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise RestoreGateError(f"{description} path must be absolute")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RestoreGateError(f"{description} traverses a symlink: {current}")
    resolved = raw.resolve(strict=False)
    if directory is True and resolved.exists() and not resolved.is_dir():
        raise RestoreGateError(f"{description} is not a directory: {resolved}")
    if directory is False and (not resolved.exists() or not resolved.is_file()):
        raise RestoreGateError(f"{description} is not a file: {resolved}")
    return resolved


def _under(path: Path, root: Path, *, description: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RestoreGateError(f"{description} is outside its declared bundle root") from exc


def _root_record(
    name: str,
    descriptor: Mapping[str, Any],
    *,
    require_windows_boundary: bool,
) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise RestoreGateError(f"boundary {name} is malformed")
    expected_kind = PRIMARY_KIND if name == "primary" else SECONDARY_KIND
    kind = descriptor.get("kind")
    if kind != expected_kind and require_windows_boundary:
        raise RestoreGateError(f"boundary {name} kind is not {expected_kind}")
    root = _resolved_path(descriptor.get("root"), description=f"boundary {name} root", directory=True)
    if not root.exists():
        raise RestoreGateError(f"boundary {name} root is unavailable: {root}")
    parts = set(root.parts)
    if any(part.lower() in _TEMP_PARTS for part in root.parts):
        raise RestoreGateError(f"boundary {name} root is temporary: {root}")
    if name == "secondary" and require_windows_boundary:
        windows_root = Path("/mnt/c").resolve()
        _under(root, windows_root, description="secondary boundary root")
    mount_root = _resolved_path(
        descriptor.get("mount_root"),
        description=f"boundary {name} mount_root",
        directory=True,
    )
    if name == "secondary" and require_windows_boundary and mount_root != Path("/mnt/c").resolve():
        raise RestoreGateError("secondary mount_root must be the /mnt/c Windows mount")
    _under(root, mount_root, description=f"boundary {name} root/mount provenance")
    root_stat = root.stat()
    mount_stat = mount_root.stat()
    actual_st_dev = int(root_stat.st_dev)
    if int(mount_stat.st_dev) != actual_st_dev:
        raise RestoreGateError(f"boundary {name} mount device differs from root")
    if "st_dev" not in descriptor:
        raise RestoreGateError(f"boundary {name} st_dev provenance is absent")
    try:
        declared_st_dev = int(descriptor["st_dev"])
    except (TypeError, ValueError) as exc:
        raise RestoreGateError(f"boundary {name} st_dev provenance is malformed") from exc
    if declared_st_dev != actual_st_dev:
        raise RestoreGateError(
            f"boundary {name} st_dev changed: {actual_st_dev}/{declared_st_dev}"
        )
    boundary_id = descriptor.get("boundary_id")
    if not isinstance(boundary_id, str) or not boundary_id:
        raise RestoreGateError(f"boundary {name} boundary_id is absent")
    return {
        "name": name,
        "boundary_id": boundary_id,
        "kind": kind,
        "root": str(root),
        "st_dev": actual_st_dev,
        "mount_root": str(mount_root),
        "parts": sorted(parts),
    }


def _binding_record(
    path: Path,
    binding: Mapping[str, Any],
    *,
    description: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RestoreGateError(f"{description} is not a regular file: {path}")
    actual = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }
    try:
        declared_bytes = int(binding["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RestoreGateError(f"{description} byte binding is malformed") from exc
    declared_sha = binding.get("sha256")
    if declared_bytes != actual["bytes"] or declared_sha != actual["sha256"]:
        raise RestoreGateError(
            f"{description} changed: bytes {actual['bytes']}/{declared_bytes}, "
            f"sha {actual['sha256']}/{declared_sha}"
        )
    if _SHA256.fullmatch(str(declared_sha)) is None:
        raise RestoreGateError(f"{description} SHA-256 binding is malformed")
    declared_path = binding.get("path")
    if declared_path is not None:
        resolved_declared = _resolved_path(declared_path, description=f"{description} binding")
        if resolved_declared != path:
            raise RestoreGateError(f"{description} path binding changed")
    return actual


def _asset_file(
    asset_name: str,
    asset: Mapping[str, Any],
    *,
    boundary: Mapping[str, Any],
    boundary_name: str,
) -> dict[str, Any]:
    relative = _safe_relative(asset.get("relative_path"), description=f"asset {asset_name}")
    root = Path(str(boundary["root"]))
    path = root / relative
    _under(path, root, description=f"asset {asset_name} {boundary_name}")
    path = _resolved_path(path, description=f"asset {asset_name} {boundary_name}", directory=False)
    _under(path, root, description=f"asset {asset_name} {boundary_name}")
    copies = asset.get("copies")
    if not isinstance(copies, Mapping):
        raise RestoreGateError(f"asset {asset_name} copies are absent")
    binding = copies.get(boundary_name)
    if not isinstance(binding, Mapping):
        raise RestoreGateError(f"asset {asset_name} lacks {boundary_name} binding")
    actual = _binding_record(path, binding, description=f"asset {asset_name} {boundary_name}")
    if binding.get("relative_path") is not None:
        declared_relative = _safe_relative(
            binding["relative_path"],
            description=f"asset {asset_name} {boundary_name}",
        )
        if declared_relative != relative:
            raise RestoreGateError(f"asset {asset_name} relative path differs across bindings")
    return actual


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RestoreGateError(f"{description} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RestoreGateError(f"{description} must be a JSON object")
    return value


def _validate_tensor_identity_sidecar(
    asset_name: str,
    asset: Mapping[str, Any],
    *,
    boundaries: Mapping[str, Mapping[str, Any]],
    state_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    identity = asset.get("tensor_identity")
    if not isinstance(identity, Mapping):
        raise RestoreGateError(f"asset {asset_name} lacks tensor_identity")
    relative = _safe_relative(
        identity.get("relative_path"),
        description=f"asset {asset_name} tensor_identity",
    )
    copies = identity.get("copies")
    if not isinstance(copies, Mapping):
        raise RestoreGateError(f"asset {asset_name} tensor_identity copies are absent")
    checked: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for boundary_name in BOUNDARY_NAMES:
        boundary = boundaries[boundary_name]
        root = Path(str(boundary["root"]))
        path = _resolved_path(
            root / relative,
            description=f"asset {asset_name} tensor_identity {boundary_name}",
            directory=False,
        )
        _under(path, root, description=f"asset {asset_name} tensor_identity {boundary_name}")
        binding = copies.get(boundary_name)
        if not isinstance(binding, Mapping):
            raise RestoreGateError(
                f"asset {asset_name} tensor_identity lacks {boundary_name} binding"
            )
        checked[boundary_name] = _binding_record(
            path,
            binding,
            description=f"asset {asset_name} tensor_identity {boundary_name}",
        )
        payload = _read_json(path, description=f"asset {asset_name} tensor_identity {boundary_name}")
        if payload.get("schema") != TENSOR_SCHEMA:
            raise RestoreGateError(f"asset {asset_name} tensor identity schema changed")
        if payload.get("file_sha256") != state_records[boundary_name]["sha256"]:
            raise RestoreGateError(
                f"asset {asset_name} tensor identity does not bind the selected file"
            )
        tensors = payload.get("tensors")
        if not isinstance(tensors, Mapping) or not tensors:
            raise RestoreGateError(f"asset {asset_name} tensor inventory is empty")
        tensor_keys_sorted = payload.get("tensor_keys_sorted")
        if tensor_keys_sorted != sorted(tensors):
            raise RestoreGateError(f"asset {asset_name} tensor names are not sorted")
        normalized: dict[str, Any] = {}
        for key, entry in tensors.items():
            if not isinstance(key, str) or not key or not isinstance(entry, Mapping):
                raise RestoreGateError(f"asset {asset_name} tensor inventory is malformed")
            dtype = entry.get("dtype")
            shape = entry.get("shape")
            digest = entry.get("sha256")
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or any(not isinstance(dim, int) or dim < 0 for dim in shape)
                or _SHA256.fullmatch(str(digest)) is None
            ):
                raise RestoreGateError(f"asset {asset_name} tensor identity is malformed: {key}")
            normalized[key] = {
                "dtype": dtype,
                "shape": list(shape),
                "sha256": str(digest),
            }
        payloads[boundary_name] = {
            "schema": payload["schema"],
            "file_sha256": payload["file_sha256"],
            "tensors": normalized,
        }
    if payloads["primary"] != payloads["secondary"]:
        raise RestoreGateError(f"asset {asset_name} tensor inventories differ across copies")
    return {"relative_path": str(relative), "copies": checked, "payload": payloads["primary"]}


def _validate_smoke(manifest: Mapping[str, Any], assets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    smoke = manifest.get("smoke")
    if not isinstance(smoke, Mapping):
        raise RestoreGateError("smoke descriptor is absent")
    if smoke.get("truth_free") is not True:
        raise RestoreGateError("smoke is not explicitly truth-free")
    if tuple(smoke.get("record_order", ())) != SMOKE_RECORD_ORDER:
        raise RestoreGateError("smoke record order changed")
    if tuple(smoke.get("methods", ())) != SMOKE_METHODS:
        raise RestoreGateError("smoke method order changed")
    if smoke.get("input_asset") != "smoke_input" or smoke.get("expected_asset") != "smoke_expected":
        raise RestoreGateError("smoke asset roles changed")
    receipt = smoke.get("selection_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("complete_before_smoke") is not True:
        raise RestoreGateError("smoke was not bound after complete checkpoint selection")
    if receipt.get("smoke_used_for_selection") is not False:
        raise RestoreGateError("smoke selection leakage is not explicitly false")
    if receipt.get("independent_evaluation_truth_opened") is not False:
        raise RestoreGateError("smoke evaluation truth boundary is not explicitly closed")
    record_bindings = smoke.get("record_bindings")
    if not isinstance(record_bindings, list) or len(record_bindings) != len(SMOKE_RECORD_ORDER):
        raise RestoreGateError("smoke record identity bindings are absent")
    if [item.get("record_id") if isinstance(item, Mapping) else None for item in record_bindings] != list(SMOKE_RECORD_ORDER):
        raise RestoreGateError("smoke record identity order changed")
    for index, binding in enumerate(record_bindings):
        if not isinstance(binding, Mapping):
            raise RestoreGateError(f"smoke record identity is malformed: {index}")
        for key in (
            "record_id",
            "source_identity_sha256",
            "activation_slice_sha256",
            "attention_mask_slice_sha256",
            "position_ids_slice_sha256",
        ):
            if key != "record_id" and _SHA256.fullmatch(str(binding.get(key))) is None:
                raise RestoreGateError(f"smoke record identity digest is malformed: {index}/{key}")
    input_key_layout = smoke.get("input_key_layout")
    input_tensor_keys = smoke.get("input_tensor_keys")
    if not isinstance(input_key_layout, str) or input_key_layout not in _SMOKE_INPUT_LAYOUTS:
        raise RestoreGateError("smoke input key layout is absent or unsupported")
    expected_key_layout = _SMOKE_INPUT_LAYOUTS[input_key_layout]
    if not isinstance(input_tensor_keys, Mapping):
        raise RestoreGateError("smoke input tensor-key mapping is absent")
    normalized_input_keys: dict[str, list[str]] = {}
    for canonical in _SMOKE_INPUT_KEYS:
        values = input_tensor_keys.get(canonical)
        if not isinstance(values, list) or tuple(values) != expected_key_layout[canonical]:
            raise RestoreGateError(f"smoke input tensor-key mapping changed: {canonical}")
        normalized_input_keys[canonical] = list(values)
    flattened_keys = [key for canonical in _SMOKE_INPUT_KEYS for key in normalized_input_keys[canonical]]
    if len(set(flattened_keys)) != len(flattened_keys):
        raise RestoreGateError("smoke input tensor-key mapping is ambiguous")
    for name in ("input_tensor_digests", "prediction_tensor_digests"):
        digests = smoke.get(name)
        if not isinstance(digests, Mapping) or not digests:
            raise RestoreGateError(f"smoke {name} are absent")
        if name == "prediction_tensor_digests" and set(digests) != set(SMOKE_METHODS):
            raise RestoreGateError("smoke prediction digest methods changed")
        if name == "input_tensor_digests" and set(digests) != set(_SMOKE_INPUT_KEYS):
            raise RestoreGateError("smoke input geometry digests are incomplete")
        for key, digest in digests.items():
            if not isinstance(key, str) or _SHA256.fullmatch(str(digest)) is None:
                raise RestoreGateError(f"smoke {name} contains malformed digest: {key}")
    for name in ("smoke_input", "smoke_expected"):
        if name not in assets:
            raise RestoreGateError(f"smoke asset is absent: {name}")
    return {
        "truth_free": True,
        "record_order": list(SMOKE_RECORD_ORDER),
        "methods": list(SMOKE_METHODS),
        "record_bindings": [dict(item) for item in record_bindings],
        "input_asset": "smoke_input",
        "expected_asset": "smoke_expected",
        "input_key_layout": input_key_layout,
        "input_tensor_keys": normalized_input_keys,
        "input_tensor_digests": dict(smoke["input_tensor_digests"]),
        "prediction_tensor_digests": dict(smoke["prediction_tensor_digests"]),
    }



def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RestoreGateError("canonical JSON value is not serializable") from exc


def _validate_consumer(
    manifest: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    consumer = manifest.get("consumer")
    if not isinstance(consumer, Mapping):
        raise RestoreGateError("consumer descriptor is absent")
    entrypoint_asset = consumer.get("entrypoint_asset")
    if entrypoint_asset != "package_cli" or entrypoint_asset not in assets:
        raise RestoreGateError("consumer package CLI asset is not bound")
    if consumer.get("python") != CONSUMER_PYTHON:
        raise RestoreGateError("consumer Python executable is not the pinned interpreter")
    if consumer.get("receipt_schema") != CONSUMER_RECEIPT_SCHEMA:
        raise RestoreGateError("consumer receipt schema is not the pinned package schema")
    if consumer.get("receipt_task_id") != CONSUMER_TASK_ID:
        raise RestoreGateError("consumer receipt task identity is not the pinned package task")
    if consumer.get("observation_asset") != "smoke_input":
        raise RestoreGateError("consumer observation asset is not the fixed smoke input")
    output_relative = _safe_relative(
        consumer.get("output_relative_path"),
        description="consumer output",
    )
    receipt_relative = _safe_relative(
        consumer.get("receipt_relative_path"),
        description="consumer receipt",
    )
    if not str(output_relative).startswith("runtime/"):
        raise RestoreGateError("consumer output must be under runtime/")
    if not str(receipt_relative).startswith("runtime/"):
        raise RestoreGateError("consumer receipt must be under runtime/")
    expected_receipt_relative = output_relative.with_suffix(".receipt.json")
    if receipt_relative != expected_receipt_relative:
        raise RestoreGateError("consumer receipt is not the CLI sibling receipt")
    all_bundle_paths = {
        str(_safe_relative(asset["relative_path"], description=f"asset {name}"))
        for name, asset in assets.items()
    }
    for name, asset in assets.items():
        if "tensor_identity" in asset:
            identity = asset["tensor_identity"]
            if isinstance(identity, Mapping):
                all_bundle_paths.add(
                    str(_safe_relative(identity["relative_path"], description=f"asset {name} tensor identity"))
                )
    if str(output_relative) in all_bundle_paths or str(receipt_relative) in all_bundle_paths:
        raise RestoreGateError("consumer runtime output collides with a bundle asset")
    for key, expected in {
        "package_root_arg": "--package-root",
        "observations_arg": "--observations",
        "output_arg": "--output",
        "device_arg": "--device",
    }.items():
        if consumer.get(key) != expected:
            raise RestoreGateError(f"consumer argument binding changed: {key}")
    device = consumer.get("device")
    if not isinstance(device, str) or not device or device.startswith("-"):
        raise RestoreGateError("consumer device binding is malformed")
    settings = consumer.get("numerical_settings")
    if not isinstance(settings, Mapping) or settings.get("device") != device:
        raise RestoreGateError("consumer numerical device settings are absent or changed")
    settings_sha = consumer.get("numerical_settings_sha256")
    if _SHA256.fullmatch(str(settings_sha)) is None:
        raise RestoreGateError("consumer numerical settings hash is malformed")
    if hashlib.sha256(_canonical_json(dict(settings))).hexdigest() != settings_sha:
        raise RestoreGateError("consumer numerical settings hash changed")
    code_roots = consumer.get("code_relative_roots", ["code"])
    if not isinstance(code_roots, list) or not code_roots:
        raise RestoreGateError("consumer code roots are absent")
    normalized_roots = [
        str(_safe_relative(item, description="consumer code root"))
        for item in code_roots
    ]
    try:
        timeout_seconds = int(consumer["timeout_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RestoreGateError("consumer timeout is malformed") from exc
    if timeout_seconds <= 0 or timeout_seconds > 3600:
        raise RestoreGateError("consumer timeout is outside the bounded range")
    dependency_id = consumer.get("dependency_id")
    if not isinstance(dependency_id, str) or not dependency_id:
        raise RestoreGateError("consumer dependency identity is absent")
    return {
        "entrypoint_asset": entrypoint_asset,
        "python": CONSUMER_PYTHON,
        "receipt_schema": CONSUMER_RECEIPT_SCHEMA,
        "receipt_task_id": CONSUMER_TASK_ID,
        "observation_asset": "smoke_input",
        "output_relative_path": str(output_relative),
        "receipt_relative_path": str(receipt_relative),
        "package_root_arg": "--package-root",
        "observations_arg": "--observations",
        "output_arg": "--output",
        "device_arg": "--device",
        "device": device,
        "numerical_settings": dict(settings),
        "numerical_settings_sha256": settings_sha,
        "code_relative_roots": normalized_roots,
        "timeout_seconds": timeout_seconds,
        "dependency_id": dependency_id,
    }

def _validate_clean_root(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RestoreGateError("clean_runtime descriptor is absent")
    if value.get("training_worktree") is not False or value.get("temporary") is not False:
        raise RestoreGateError("clean runtime is marked as training or temporary")
    root = _resolved_path(value.get("root"), description="clean runtime root", directory=True)
    for part in root.parts:
        if part.lower() in _TEMP_PARTS:
            raise RestoreGateError(f"clean runtime root is temporary: {root}")
    # A clean runtime may live in the P11 worktree, but not in another agent's
    # training worktree.  The path is only a runtime target; it is not an input
    # dependency until a restore operation populates it.
    if any(part in _TRAINING_WORKTREE_PARTS for part in root.parts):
        raise RestoreGateError(f"clean runtime root is a training worktree: {root}")
    return {"root": str(root), "training_worktree": False, "temporary": False}


def validate_restore_manifest(
    manifest_path: Path,
    *,
    require_windows_boundary: bool = True,
    require_clean_root: bool = True,
    require_distinct_devices: bool = True,
) -> dict[str, Any]:
    """Validate persistent copies and return a create-only verification report.

    This function reads only the descriptor, file metadata, and tensor-identity
    JSON sidecars. It does not open a checkpoint tensor, import a model, read
    smoke tensors, select records, or read truth.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_path, description="restore manifest")
    if manifest.get("schema") != SCHEMA or manifest.get("task_id") != TASK_ID:
        raise RestoreGateError("restore manifest schema or task identity changed")
    if manifest.get("status") not in {"RESTORE_PACKAGE_CANDIDATE", "RESTORE_PACKAGE_READY"}:
        raise RestoreGateError("restore manifest status is not a restore candidate")
    _walk_forbidden_flags(manifest, location="manifest")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise RestoreGateError("selection descriptor is absent")
    required_selection = {
        "complete_before_smoke": True,
        "smoke_used_for_selection": False,
        "independent_evaluation_truth_opened": False,
        "checkpoint_reselection_after_smoke": False,
    }
    for key, expected in required_selection.items():
        if selection.get(key) is not expected:
            raise RestoreGateError(f"selection binding changed: {key}")
    if not isinstance(selection.get("development_labels_used"), bool):
        raise RestoreGateError("selection development_labels_used must be explicit")
    boundaries_payload = manifest.get("boundaries")
    if not isinstance(boundaries_payload, Mapping):
        raise RestoreGateError("boundary descriptors are absent")
    boundaries: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for name in BOUNDARY_NAMES:
        record = _root_record(
            name,
            boundaries_payload.get(name),
            require_windows_boundary=require_windows_boundary,
        )
        if record["boundary_id"] in seen_ids:
            raise RestoreGateError("boundary IDs are not unique")
        seen_ids.add(record["boundary_id"])
        boundaries[name] = record
    if require_distinct_devices and boundaries["primary"]["st_dev"] == boundaries["secondary"]["st_dev"]:
        raise RestoreGateError("copy roots do not cross a filesystem failure boundary")
    assets_payload = manifest.get("assets")
    if not isinstance(assets_payload, Mapping):
        raise RestoreGateError("asset descriptors are absent")
    missing = [name for name in REQUIRED_ASSETS if name not in assets_payload]
    if missing:
        raise RestoreGateError(f"required assets are absent: {', '.join(missing)}")
    assets: dict[str, Any] = {}
    # Every declared asset is copied and verified. Required roles establish the
    # minimum package; extra bundled helper/config files cannot bypass hashing.
    for name, asset in assets_payload.items():
        if not isinstance(name, str) or not name:
            raise RestoreGateError("asset name is malformed")
        if not isinstance(asset, Mapping):
            raise RestoreGateError(f"asset {name} is malformed")
        if not isinstance(asset.get("copies"), Mapping):
            raise RestoreGateError(f"asset {name} copies are absent")
        records = {
            boundary_name: _asset_file(
                name,
                asset,
                boundary=boundaries[boundary_name],
                boundary_name=boundary_name,
            )
            for boundary_name in BOUNDARY_NAMES
        }
        if records["primary"]["bytes"] != records["secondary"]["bytes"]:
            raise RestoreGateError(f"asset {name} byte counts differ across copies")
        if records["primary"]["sha256"] != records["secondary"]["sha256"]:
            raise RestoreGateError(f"asset {name} hashes differ across copies")
        if name in METHOD_BANKS:
            if asset.get("bank") != METHOD_BANKS[name]:
                raise RestoreGateError(f"asset {name} bank binding changed")
            try:
                selected_step = int(asset["selected_step"])
                if selected_step < 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError) as exc:
                raise RestoreGateError(f"asset {name} selected_step is malformed") from exc
            if not isinstance(asset.get("model_id"), str) or not asset["model_id"]:
                raise RestoreGateError(f"asset {name} model_id is absent")
            if not str(asset["relative_path"]).endswith(".safetensors"):
                raise RestoreGateError(f"asset {name} is not a safetensors state")
        assets[name] = {
            "relative_path": str(_safe_relative(asset["relative_path"], description=f"asset {name}")),
            "copies": records,
            "metadata": {
                key: asset[key]
                for key in ("bank", "selected_step", "model_id")
                if key in asset
            },
        }
    selection_receipt_asset = assets_payload["selection_receipt"]
    selection_records = assets["selection_receipt"]["copies"]
    selection_payloads: dict[str, dict[str, Any]] = {}
    selection_relative = _safe_relative(
        selection_receipt_asset["relative_path"],
        description="asset selection_receipt",
    )
    for boundary_name in BOUNDARY_NAMES:
        boundary_root = Path(boundaries[boundary_name]["root"])
        selection_path = _resolved_path(
            boundary_root / selection_relative,
            description=f"selection receipt {boundary_name}",
            directory=False,
        )
        _under(selection_path, boundary_root, description=f"selection receipt {boundary_name}")
        selection_payloads[boundary_name] = _read_json(
            selection_path,
            description=f"selection receipt {boundary_name}",
        )
        receipt_payload = selection_payloads[boundary_name]
        receipt_schema = receipt_payload.get("schema")
        expected_receipt_task = (
            TASK_ID
            if receipt_schema == SELECTION_RECEIPT_SCHEMA
            else "TRR-0012"
        )
        if (
            receipt_schema not in SELECTION_RECEIPT_SCHEMAS
            or receipt_payload.get("task_id") != expected_receipt_task
        ):
            raise RestoreGateError("selection receipt schema or task identity changed")
        if receipt_payload.get("status") != "SELECTION_COMPLETE_BEFORE_SMOKE":
            raise RestoreGateError("selection receipt status is not complete-before-smoke")
        if receipt_payload.get("smoke_used_for_selection") is not False:
            raise RestoreGateError("selection receipt records smoke leakage")
        if receipt_payload.get("independent_evaluation_truth_opened") is not False:
            raise RestoreGateError("selection receipt records opened evaluation truth")
        if not isinstance(receipt_payload.get("development_labels_used"), bool):
            raise RestoreGateError("selection receipt development_labels_used is absent")
        if receipt_payload["development_labels_used"] != selection["development_labels_used"]:
            raise RestoreGateError("selection development-label provenance differs")
        selected_methods = receipt_payload.get("selected_methods")
        if not isinstance(selected_methods, Mapping):
            raise RestoreGateError("selection receipt selected methods are absent")
        for method_name in METHOD_BANKS:
            selected = selected_methods.get(method_name)
            if not isinstance(selected, Mapping):
                raise RestoreGateError(f"selection receipt omits {method_name}")
            if selected.get("state_sha256") != assets[method_name]["copies"][boundary_name]["sha256"]:
                raise RestoreGateError(f"selection receipt state hash differs: {method_name}")
            if selected.get("bank") != assets[method_name]["metadata"].get("bank"):
                raise RestoreGateError(f"selection receipt bank differs: {method_name}")
            if selected.get("model_id") != assets[method_name]["metadata"].get("model_id"):
                raise RestoreGateError(f"selection receipt model differs: {method_name}")
            try:
                receipt_step = int(selected.get("selected_step", -1))
            except (TypeError, ValueError) as exc:
                raise RestoreGateError(f"selection receipt step is malformed: {method_name}") from exc
            if receipt_step != assets[method_name]["metadata"].get("selected_step"):
                raise RestoreGateError(f"selection receipt step differs: {method_name}")
    if selection_payloads["primary"] != selection_payloads["secondary"]:
        raise RestoreGateError("selection receipt differs across copies")
    tensor_reports: dict[str, Any] = {}
    for name in TENSOR_ASSETS:
        if not isinstance(assets_payload.get(name), Mapping):
            raise RestoreGateError(f"tensor asset is absent: {name}")
    for name, asset in assets_payload.items():
        if "tensor_identity" in asset:
            tensor_reports[name] = _validate_tensor_identity_sidecar(
                name,
                asset,
                boundaries=boundaries,
                state_records=assets[name]["copies"],
            )
    for name in TENSOR_ASSETS:
        if name not in tensor_reports:
            raise RestoreGateError(f"tensor identity is absent: {name}")
    smoke_report = _validate_smoke(manifest, assets_payload)
    consumer_report = _validate_consumer(manifest, assets_payload)
    clean_report = _validate_clean_root(manifest.get("clean_runtime")) if require_clean_root else None
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS_METADATA_BOUNDARIES",
        "manifest": str(manifest_path),
        "package_id": manifest.get("package_id"),
        "boundaries": boundaries,
        "assets": assets,
        "tensor_identity": tensor_reports,
        "selection_receipt": {
            "copies": selection_records,
            "payload": selection_payloads["primary"],
        },
        "smoke": smoke_report,
        "consumer": consumer_report,
        "clean_runtime": clean_report,
        "tensor_files_opened": False,
        "smoke_files_opened": False,
        "independent_evaluation_truth_opened": False,
    }


def _ensure_clean_destination(
    destination: Path,
    *,
    source_roots: Sequence[Path] = (),
    require_empty: bool = False,
) -> Path:
    raw = Path(destination).expanduser()
    if not raw.is_absolute():
        raise RestoreGateError("clean runtime destination must be absolute")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RestoreGateError(f"clean runtime traverses a symlink: {current}")
    resolved = raw.resolve(strict=False)
    for part in resolved.parts:
        if part.lower() in _TEMP_PARTS:
            raise RestoreGateError(f"clean runtime destination is temporary: {resolved}")
    if any(part in _TRAINING_WORKTREE_PARTS for part in resolved.parts):
        raise RestoreGateError(f"clean runtime destination is a training worktree: {resolved}")
    for source_root in source_roots:
        source_root = source_root.resolve()
        try:
            resolved.relative_to(source_root)
        except ValueError:
            continue
        raise RestoreGateError("clean runtime is inside a deployment source root")
    if resolved.exists():
        if not resolved.is_dir() or resolved.is_symlink():
            raise RestoreGateError("clean runtime destination is not a real directory")
        if require_empty and any(resolved.iterdir()):
            raise RestoreGateError("clean runtime destination is not create-only/empty")
    else:
        parent = resolved.parent
        if not parent.exists() or parent.is_symlink():
            raise RestoreGateError("clean runtime parent is unavailable or symlinked")
    return resolved


def _asset_relative_paths(report: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, asset in report["assets"].items():
        paths[name] = Path(str(asset["relative_path"]))
    for name, identity in report["tensor_identity"].items():
        paths[f"{name}.tensor_identity"] = Path(str(identity["relative_path"]))
    return paths


def _clean_asset_binding(
    clean_root: Path,
    relative: Path,
    expected: Mapping[str, Any],
    *,
    description: str,
) -> dict[str, Any]:
    path = _resolved_path(clean_root / relative, description=description, directory=False)
    _under(path, clean_root, description=description)
    # The source report's binding includes its source absolute path.  A clean
    # runtime must verify the same bytes/hash at a different absolute path.
    portable_expected = {
        key: value
        for key, value in expected.items()
        if key in {"bytes", "sha256"}
    }
    return _binding_record(path, portable_expected, description=description)


def verify_clean_runtime(
    manifest_path: Path,
    clean_root: Path,
    *,
    require_windows_boundary: bool = True,
    require_distinct_devices: bool = True,
) -> dict[str, Any]:
    """Verify files retrieved into a clean Agent-2 runtime directory.

    The returned receipt records the actual paths loaded by the consumer.  It
    does not import training modules, fetch missing files, open smoke tensors,
    or run a model.
    """
    report = validate_restore_manifest(
        manifest_path,
        require_windows_boundary=require_windows_boundary,
        require_clean_root=False,
        require_distinct_devices=require_distinct_devices,
    )
    destination = _ensure_clean_destination(
        Path(clean_root),
        source_roots=tuple(Path(item["root"]) for item in report["boundaries"].values()),
    )
    expected_clean = _read_json(
        Path(manifest_path).expanduser().resolve(),
        description="restore manifest",
    ).get("clean_runtime")
    if isinstance(expected_clean, Mapping) and expected_clean.get("root") is not None:
        declared_root = _resolved_path(
            expected_clean["root"],
            description="declared clean runtime root",
            directory=True,
        )
        if declared_root != destination:
            raise RestoreGateError("clean runtime destination differs from manifest")
    checked_assets: dict[str, Any] = {}
    loaded_paths: list[str] = []
    for name, asset in report["assets"].items():
        relative = Path(str(asset["relative_path"]))
        expected = asset["copies"]["primary"]
        checked = _clean_asset_binding(
            destination,
            relative,
            expected,
            description=f"clean asset {name}",
        )
        checked_assets[name] = checked
        loaded_paths.append(checked["path"])
    for name, identity in report["tensor_identity"].items():
        relative = Path(str(identity["relative_path"]))
        expected = identity["copies"]["primary"]
        checked = _clean_asset_binding(
            destination,
            relative,
            expected,
            description=f"clean tensor identity {name}",
        )
        checked_assets[f"{name}.tensor_identity"] = checked
        loaded_paths.append(checked["path"])
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS_CLEAN_RUNTIME_RETRIEVED",
        "clean_root": str(destination),
        "loaded_asset_paths": loaded_paths,
        "assets": checked_assets,
        "source_boundaries": report["boundaries"],
        "tensor_files_opened": False,
        "smoke_files_opened": False,
        "independent_evaluation_truth_opened": False,
    }


def materialize_clean_runtime(
    manifest_path: Path,
    clean_root: Path,
    *,
    source_boundary: str = "secondary",
    require_windows_boundary: bool = True,
    require_distinct_devices: bool = True,
) -> dict[str, Any]:
    """Copy one verified package boundary into a new clean runtime directory.

    The default source is the independently mounted secondary copy. The
    destination must be absent or empty. This function copies only the
    hash-bound deployment bundle and never searches for or downloads assets.
    """
    report = validate_restore_manifest(
        manifest_path,
        require_windows_boundary=require_windows_boundary,
        require_clean_root=False,
        require_distinct_devices=require_distinct_devices,
    )
    if source_boundary != "secondary":
        raise RestoreGateError("clean retrieval must use the verified secondary boundary")
    source_root = Path(report["boundaries"][source_boundary]["root"])
    destination = _ensure_clean_destination(
        Path(clean_root),
        source_roots=tuple(Path(item["root"]) for item in report["boundaries"].values()),
        require_empty=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    for relative in _asset_relative_paths(report).values():
        source = _resolved_path(
            source_root / relative,
            description=f"source bundle {relative}",
            directory=False,
        )
        target = destination / relative
        _under(target, destination, description=f"clean bundle {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise RestoreGateError(f"clean bundle target already exists: {target}")
        shutil.copy2(source, target)
    receipt = verify_clean_runtime(
        manifest_path,
        destination,
        require_windows_boundary=require_windows_boundary,
        require_distinct_devices=require_distinct_devices,
    )
    receipt["source_boundary"] = source_boundary
    receipt["source_root"] = str(source_root)
    return receipt


def consumer_environment(
    clean_root: Path,
    *,
    code_relative_roots: Sequence[str] = ("code",),
    offline_cache_relative: str = "offline-cache",
    dependency_id: str | None = None,
) -> dict[str, str]:
    """Build a controlled offline consumer environment.

    The returned environment intentionally does not inherit PYTHONPATH or
    network/model-cache settings.  The caller records this receipt and starts
    the consumer from an isolated current directory.
    """
    root = _ensure_clean_destination(Path(clean_root))
    code_paths: list[str] = []
    for relative in code_relative_roots:
        safe = _safe_relative(relative, description="consumer code root")
        code_root = _resolved_path(root / safe, description="consumer code root", directory=True)
        _under(code_root, root, description="consumer code root")
        code_paths.append(str(code_root))
    cache = root / _safe_relative(offline_cache_relative, description="offline cache")
    env = {
        "PYTHONPATH": os.pathsep.join(code_paths),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": str(cache),
    }
    if dependency_id is not None:
        env["TRR_P11_DEPENDENCY_ID"] = str(dependency_id)
    return env


def _clean_runtime_file(report: Mapping[str, Any], clean_root: Path, asset_name: str) -> Path:
    asset = report["assets"].get(asset_name)
    if not isinstance(asset, Mapping):
        raise RestoreGateError(f"clean asset is absent: {asset_name}")
    return _resolved_path(
        clean_root / Path(str(asset["relative_path"])),
        description=f"clean asset {asset_name}",
        directory=False,
    )


def _verify_clean_tensor_inventories(
    report: Mapping[str, Any],
    clean_root: Path,
) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for name, identity in report["tensor_identity"].items():
        tensor_path = _clean_runtime_file(report, clean_root, name)
        identity_path = _resolved_path(
            clean_root / Path(str(identity["relative_path"])),
            description=f"clean tensor identity {name}",
            directory=False,
        )
        receipts[name] = verify_tensor_identity(
            tensor_path,
            identity_path,
            expected_file_sha256=report["assets"][name]["copies"]["primary"]["sha256"],
        )
    return receipts


def verify_smoke_input_identity(
    observations_path: Path,
    smoke: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the bound smoke tensors and each opaque row/slice digest."""
    observations_path = _resolved_path(
        observations_path,
        description="smoke input",
        directory=False,
    )
    expected = smoke.get("input_tensor_digests")
    record_bindings = smoke.get("record_bindings")
    layout_name = smoke.get("input_key_layout")
    input_tensor_keys = smoke.get("input_tensor_keys")
    if (
        not isinstance(expected, Mapping)
        or set(expected) != set(_SMOKE_INPUT_KEYS)
        or not isinstance(record_bindings, list)
        or len(record_bindings) != len(SMOKE_RECORD_ORDER)
        or not isinstance(layout_name, str)
        or layout_name not in _SMOKE_INPUT_LAYOUTS
        or not isinstance(input_tensor_keys, Mapping)
    ):
        raise RestoreGateError("smoke input digest or key bindings are absent")
    expected_layout = _SMOKE_INPUT_LAYOUTS[layout_name]
    normalized_keys: dict[str, list[str]] = {}
    for canonical in _SMOKE_INPUT_KEYS:
        values = input_tensor_keys.get(canonical)
        if not isinstance(values, list) or tuple(values) != expected_layout[canonical]:
            raise RestoreGateError(f"smoke input tensor-key mapping changed: {canonical}")
        normalized_keys[canonical] = list(values)
    flattened_keys = [key for canonical in _SMOKE_INPUT_KEYS for key in normalized_keys[canonical]]
    if len(set(flattened_keys)) != len(flattened_keys):
        raise RestoreGateError("smoke input tensor-key mapping is ambiguous")
    try:
        from safetensors import safe_open
        import torch
    except ImportError as exc:
        raise RestoreGateError("safetensors/torch are unavailable for smoke input verification") from exc
    forbidden_fragments = ("token", "label", "truth", "source", "answer")
    actual_digests: dict[str, str] = {}
    slice_digests: list[dict[str, str]] = []
    with safe_open(str(observations_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        expected_keys = set(flattened_keys)
        if keys != expected_keys:
            raise RestoreGateError(
                f"smoke input tensor keys differ: expected {sorted(expected_keys)}, got {sorted(keys)}"
            )
        if any(any(fragment in key.lower() for fragment in forbidden_fragments) for key in keys):
            raise RestoreGateError("smoke input exposes a forbidden truth/source tensor")
        pieces: dict[str, list[Any]] = {
            canonical: [handle.get_tensor(key) for key in normalized_keys[canonical]]
            for canonical in _SMOKE_INPUT_KEYS
        }
        values: dict[str, Any] = {}
        for canonical in _SMOKE_INPUT_KEYS:
            try:
                values[canonical] = (
                    pieces[canonical][0]
                    if len(pieces[canonical]) == 1
                    else torch.cat(pieces[canonical], dim=0).contiguous()
                )
            except (RuntimeError, TypeError) as exc:
                raise RestoreGateError(f"smoke input tensor groups cannot be concatenated: {canonical}") from exc
            if values[canonical].ndim < 1 or int(values[canonical].shape[0]) != len(record_bindings):
                raise RestoreGateError(f"smoke input row geometry differs: {canonical}")
        if layout_name == "domain_prefixed":
            offset = 0
            for domain, activation_piece in zip(("finance", "pile"), pieces["activations"]):
                count = int(activation_piece.shape[0])
                for binding in record_bindings[offset : offset + count]:
                    if not isinstance(binding, Mapping) or not str(binding.get("record_id", "")).startswith(f"{domain}/"):
                        raise RestoreGateError("smoke input domain/key order differs from record bindings")
                offset += count
            if offset != len(record_bindings):
                raise RestoreGateError("smoke input domain row count differs from record bindings")
        for key in _SMOKE_INPUT_KEYS:
            actual = tensor_digest(values[key])
            if actual != expected[key]:
                raise RestoreGateError(f"smoke input tensor digest differs: {key}")
            actual_digests[key] = actual
        for index, binding in enumerate(record_bindings):
            if not isinstance(binding, Mapping):
                raise RestoreGateError(f"smoke input record binding is malformed: {index}")
            row_values: dict[str, str] = {}
            for tensor_key, binding_key in (
                ("activations", "activation_slice_sha256"),
                ("attention_mask", "attention_mask_slice_sha256"),
                ("position_ids", "position_ids_slice_sha256"),
            ):
                actual = tensor_digest(values[tensor_key][index])
                if actual != binding.get(binding_key):
                    raise RestoreGateError(
                        f"smoke input row digest differs: {index}/{tensor_key}"
                    )
                row_values[binding_key] = actual
            slice_digests.append(row_values)
    return {
        "verified": True,
        "key_layout": layout_name,
        "tensor_keys": normalized_keys,
        "tensor_digests": actual_digests,
        "record_count": len(slice_digests),
        "slice_digests": slice_digests,
    }

def _consumer_command(
    report: Mapping[str, Any],
    clean_root: Path,
) -> tuple[list[str], Path, Path, Path]:
    consumer = report["consumer"]
    entrypoint_asset = str(consumer["entrypoint_asset"])
    entrypoint = _clean_runtime_file(report, clean_root, entrypoint_asset)
    observations = _clean_runtime_file(report, clean_root, str(consumer["observation_asset"]))
    output_relative = _safe_relative(
        consumer["output_relative_path"],
        description="consumer output",
    )
    output = (clean_root / output_relative).resolve(strict=False)
    receipt = (
        clean_root
        / _safe_relative(consumer["receipt_relative_path"], description="consumer receipt")
    ).resolve(strict=False)
    _under(entrypoint, clean_root, description="consumer entrypoint")
    _under(observations, clean_root, description="consumer observations")
    _under(output, clean_root, description="consumer output")
    _under(receipt, clean_root, description="consumer receipt")
    if output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise RestoreGateError("consumer output or sibling receipt already exists; runtime is create-only")
    if output.parent.exists() and any(output.parent.iterdir()):
        raise RestoreGateError("consumer runtime output directory is not empty")
    if receipt.parent.exists() and any(receipt.parent.iterdir()):
        raise RestoreGateError("consumer receipt directory is not empty")
    command = [
        str(consumer["python"]),
        str(entrypoint),
        "predict",
        str(consumer["package_root_arg"]),
        str(clean_root),
        str(consumer["observations_arg"]),
        str(observations),
        str(consumer["output_arg"]),
        str(output),
        str(consumer["device_arg"]),
        str(consumer["device"]),
    ]
    return command, output, observations, receipt

def _subprocess_environment(report: Mapping[str, Any], clean_root: Path) -> dict[str, str]:
    consumer = report["consumer"]
    blocked = {
        "PYTHONPATH",
        "PYTHONHOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_ENDPOINT",
    }
    env = {key: value for key, value in os.environ.items() if key not in blocked}
    env.update(
        consumer_environment(
            clean_root,
            code_relative_roots=consumer["code_relative_roots"],
            dependency_id=consumer["dependency_id"],
        )
    )
    # Keep the interpreter's explicitly pinned installed dependencies available.
    # PYTHONPATH remains controlled to clean bundled code above; provenance checks
    # below reject any task code imported outside the hash-bound runtime.
    return env


def _verify_consumer_receipt(
    receipt_path: Path,
    report: Mapping[str, Any],
    clean_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify Agent1's sibling prediction receipt and actual load provenance."""
    receipt_path = _resolved_path(
        receipt_path,
        description="consumer receipt",
        directory=False,
    )
    output_path = _resolved_path(output_path, description="consumer output", directory=False)
    payload = _read_json(receipt_path, description="consumer receipt")
    if payload.get("schema") != CONSUMER_RECEIPT_SCHEMA or payload.get("task_id") != CONSUMER_TASK_ID:
        raise RestoreGateError("consumer receipt schema or task identity changed")
    if payload.get("status") != "PREDICTIONS_GENERATED_AFTER_SELECTION":
        raise RestoreGateError("consumer receipt status is not post-selection prediction")
    for key in ("complete_before_smoke", "smoke_used_for_selection", "independent_evaluation_truth_opened"):
        expected = True if key == "complete_before_smoke" else False
        if payload.get(key) is not expected:
            raise RestoreGateError(f"consumer receipt selection boundary changed: {key}")
    for key in ("truth_opened", "source_text_loaded", "token_ids_loaded"):
        if payload.get(key) is not False:
            raise RestoreGateError(f"consumer receipt truth boundary is not explicitly closed: {key}")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RestoreGateError("consumer receipt runtime provenance is absent")
    if runtime.get("device") != report["consumer"]["device"]:
        raise RestoreGateError("consumer receipt device differs")
    if runtime.get("numeric_profile") != "qualified FP32 decoder":
        raise RestoreGateError("consumer receipt numeric profile changed")
    dependencies = runtime.get("dependencies")
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise RestoreGateError("consumer receipt dependency versions are absent")
    if any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in dependencies.items()):
        raise RestoreGateError("consumer receipt dependency versions are malformed")
    executable = runtime.get("python_executable")
    if not isinstance(executable, str) or not executable:
        raise RestoreGateError("consumer receipt Python executable is absent")
    actual_executable = Path(executable).expanduser().resolve(strict=False)
    expected_executable = Path(report["consumer"]["python"]).expanduser().resolve(strict=False)
    if not actual_executable.is_file() or actual_executable != expected_executable:
        raise RestoreGateError("consumer receipt Python executable differs")

    declared_bundle: dict[str, str] = {}
    for name, asset in report["assets"].items():
        declared_bundle[
            str(_resolved_path(clean_root / Path(str(asset["relative_path"])), description=f"clean asset {name}", directory=False))
        ] = name
    for name, identity in report["tensor_identity"].items():
        declared_bundle[
            str(_resolved_path(clean_root / Path(str(identity["relative_path"])), description=f"clean tensor identity {name}", directory=False))
        ] = f"{name}.tensor_identity"

    def resolve_value(value: Any, *, description: str, must_be_bundle: bool) -> Path:
        if not isinstance(value, str) or not value:
            raise RestoreGateError(f"consumer receipt {description} path is malformed")
        path = Path(value)
        if not path.is_absolute():
            path = clean_root / path
        resolved = _resolved_path(path, description=f"consumer receipt {description}", directory=False)
        _under(resolved, clean_root, description=f"consumer receipt {description}")
        if any(part in _TRAINING_WORKTREE_PARTS for part in resolved.parts):
            raise RestoreGateError("consumer receipt references a training worktree")
        if any(part.lower() in _TEMP_PARTS for part in resolved.parts):
            raise RestoreGateError("consumer receipt references a temporary path")
        if must_be_bundle and str(resolved) not in declared_bundle:
            raise RestoreGateError(f"consumer receipt {description} is not a bound bundle asset")
        return resolved

    def check_binding(binding: Any, expected_path: Path, *, description: str, must_be_bundle: bool) -> dict[str, Any]:
        if not isinstance(binding, Mapping):
            raise RestoreGateError(f"consumer receipt {description} file binding is absent")
        expected_path = _resolved_path(expected_path, description=description, directory=False)
        if must_be_bundle and str(expected_path) not in declared_bundle:
            raise RestoreGateError(f"consumer receipt {description} is not a bound bundle asset")
        path_value = binding.get("loaded_path", binding.get("path"))
        if path_value is not None:
            loaded_path = resolve_value(path_value, description=description, must_be_bundle=must_be_bundle)
            if loaded_path != expected_path:
                raise RestoreGateError(f"consumer receipt {description} path differs")
        relative_value = binding.get("relative_path")
        if relative_value is not None:
            relative = _safe_relative(relative_value, description=f"consumer receipt {description}")
            if _resolved_path(clean_root / relative, description=f"consumer receipt {description}", directory=False) != expected_path:
                raise RestoreGateError(f"consumer receipt {description} relative path differs")
        try:
            expected_bytes = int(binding["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RestoreGateError(f"consumer receipt {description} byte binding is malformed") from exc
        expected_sha = binding.get("sha256")
        if _SHA256.fullmatch(str(expected_sha)) is None:
            raise RestoreGateError(f"consumer receipt {description} hash binding is malformed")
        return _binding_record(
            expected_path,
            {"bytes": expected_bytes, "sha256": expected_sha},
            description=f"consumer receipt {description}",
        )

    output_binding = payload.get("output")
    if not isinstance(output_binding, Mapping):
        raise RestoreGateError("consumer receipt output binding is absent")
    check_binding(output_binding, output_path, description="output", must_be_bundle=False)
    observations_binding = payload.get("observations")
    observation_path = _clean_runtime_file(report, clean_root, "smoke_input")
    check_binding(observations_binding, observation_path, description="observations", must_be_bundle=True)
    if observations_binding.get("tensor_sha256") != report["smoke"]["input_tensor_digests"]:
        raise RestoreGateError("consumer receipt observation tensor digests differ")
    if observations_binding.get("record_order") != report["smoke"]["record_order"]:
        raise RestoreGateError("consumer receipt observation order differs")

    methods = payload.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(SMOKE_METHODS):
        raise RestoreGateError("consumer receipt method bindings are incomplete")
    loaded_resources: set[str] = {str(observation_path)}
    checked_bindings: dict[str, dict[str, Any]] = {
        str(observation_path): {
            "path": str(observation_path),
            "bytes": observation_path.stat().st_size,
            "sha256": _sha256_file(observation_path),
            "role": "smoke_input",
        }
    }
    for method_name in SMOKE_METHODS:
        method = methods[method_name]
        state_path = _clean_runtime_file(report, clean_root, method_name)
        checked = check_binding(
            method.get("state_file_binding"),
            state_path,
            description=f"{method_name} state",
            must_be_bundle=True,
        )
        loaded_resources.add(str(state_path))
        checked_bindings[str(state_path)] = {**checked, "role": method_name}
    readout_path = _clean_runtime_file(report, clean_root, "public_readout")
    readout = payload.get("readout")
    if not isinstance(readout, Mapping):
        raise RestoreGateError("consumer receipt readout binding is absent")
    checked = check_binding(readout.get("file_binding"), readout_path, description="public readout", must_be_bundle=True)
    loaded_resources.add(str(readout_path))
    checked_bindings[str(readout_path)] = {**checked, "role": "public_readout"}

    imported_modules = runtime.get("imported_modules")
    if not isinstance(imported_modules, list) or not imported_modules:
        raise RestoreGateError("consumer receipt imported module bindings are absent")
    package_cli_path = _clean_runtime_file(report, clean_root, "package_cli")
    package_cli_binding = {
        "path": str(package_cli_path),
        "bytes": report["assets"]["package_cli"]["copies"]["primary"]["bytes"],
        "sha256": report["assets"]["package_cli"]["copies"]["primary"]["sha256"],
    }
    checked = check_binding(package_cli_binding, package_cli_path, description="package CLI", must_be_bundle=True)
    loaded_code: set[str] = {str(package_cli_path)}
    checked_bindings[str(package_cli_path)] = {**checked, "role": "package_cli"}
    for index, module in enumerate(imported_modules):
        if not isinstance(module, Mapping):
            raise RestoreGateError(f"consumer receipt imported module is malformed: {index}")
        module_name = module.get("module")
        if not isinstance(module_name, str) or not module_name:
            raise RestoreGateError(f"consumer receipt imported module name is absent: {index}")
        module_path = resolve_value(module.get("loaded_path"), description=f"imported module {module_name}", must_be_bundle=True)
        checked = check_binding(module, module_path, description=f"imported module {module_name}", must_be_bundle=True)
        loaded_code.add(str(module_path))
        checked_bindings[str(module_path)] = {**checked, "role": f"module:{module_name}"}
    for name in ("loader_code", "decoder_code"):
        expected = _clean_runtime_file(report, clean_root, name)
        if str(expected) not in loaded_code:
            raise RestoreGateError(f"consumer receipt omits imported code: {name}")
    runtime_descriptor = runtime.get("runtime_descriptor")
    if runtime_descriptor is not None:
        descriptor_path_value = runtime_descriptor.get("loaded_path") if isinstance(runtime_descriptor, Mapping) else None
        if descriptor_path_value is None:
            descriptor_path_value = runtime_descriptor.get("relative_path") if isinstance(runtime_descriptor, Mapping) else None
        if descriptor_path_value is not None:
            descriptor_path = resolve_value(descriptor_path_value, description="runtime descriptor", must_be_bundle=True)
            checked = check_binding(runtime_descriptor, descriptor_path, description="runtime descriptor", must_be_bundle=True)
            loaded_resources.add(str(descriptor_path))
            checked_bindings[str(descriptor_path)] = {**checked, "role": "runtime_descriptor"}
    return {
        "verified": True,
        "schema": payload["schema"],
        "package_root": str(clean_root),
        "loaded_code_paths": sorted(loaded_code),
        "loaded_resource_paths": sorted(loaded_resources),
        "loaded_file_bindings": [checked_bindings[key] for key in sorted(checked_bindings)],
        "dependency_id": report["consumer"]["dependency_id"],
        "dependency_versions": dict(dependencies),
        "device": runtime["device"],
        "numeric_profile": runtime["numeric_profile"],
        "training_worktree_import": False,
        "temporary_dependency": False,
    }

def run_restored_smoke(
    manifest_path: Path,
    clean_root: Path,
    *,
    verify_tensors: bool = True,
    require_windows_boundary: bool = True,
    require_distinct_devices: bool = True,
) -> dict[str, Any]:
    """Restore from a verified clean bundle and run the bound smoke CLI.

    The caller must grant the model execution window explicitly. This function
    retrieves no files and uses no training checkout; it only runs the pinned
    bundled entrypoint against the copied smoke fixture.
    """
    report = validate_restore_manifest(
        manifest_path,
        require_windows_boundary=require_windows_boundary,
        require_clean_root=False,
        require_distinct_devices=require_distinct_devices,
    )
    clean = verify_clean_runtime(
        manifest_path,
        clean_root,
        require_windows_boundary=require_windows_boundary,
        require_distinct_devices=require_distinct_devices,
    )
    destination = Path(clean["clean_root"])
    if not verify_tensors:
        raise RestoreGateError("full tensor verification is mandatory for restored smoke")
    tensor_receipts = _verify_clean_tensor_inventories(report, destination)
    command, output, observations, receipt = _consumer_command(report, destination)
    smoke_input_receipt = verify_smoke_input_identity(observations, report["smoke"])
    env = _subprocess_environment(report, destination)
    start_wall = time.time()
    start_monotonic = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=destination,
            env=env,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(report["consumer"]["timeout_seconds"]),
        )
    except subprocess.TimeoutExpired as exc:
        raise RestoreGateError("consumer smoke timed out") from exc
    elapsed = time.monotonic() - start_monotonic
    if completed.returncode != 0:
        raise RestoreGateError(
            f"consumer smoke failed with exit {completed.returncode}: "
            f"{completed.stderr[-500:]}"
        )
    if not output.exists() or output.is_symlink() or not output.is_file():
        raise RestoreGateError("consumer smoke output is missing or symlinked")
    if not receipt.exists() or receipt.is_symlink() or not receipt.is_file():
        raise RestoreGateError("consumer sibling receipt is missing or symlinked")
    consumer_receipt = _verify_consumer_receipt(receipt, report, destination, output)
    post_execution_clean = verify_clean_runtime(
        manifest_path,
        destination,
        require_windows_boundary=require_windows_boundary,
        require_distinct_devices=require_distinct_devices,
    )
    expected = _clean_runtime_file(report, destination, "smoke_expected")
    comparison = compare_smoke_prediction_files(
        expected,
        output,
        expected_tensor_digests=report["smoke"]["prediction_tensor_digests"],
    )
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS_RESTORED_SMOKE",
        "clean_runtime": str(destination),
        "command": command,
        "cwd": str(destination),
        "device": report["consumer"]["device"],
        "dependency_id": report["consumer"]["dependency_id"],
        "package_id": report.get("package_id"),
        "boundaries": report["boundaries"],
        "assets": report["assets"],
        "restore_manifest": _binding_record(
            Path(manifest_path).expanduser().resolve(),
            {"bytes": Path(manifest_path).expanduser().resolve().stat().st_size, "sha256": _sha256_file(Path(manifest_path).expanduser().resolve())},
            description="restore manifest receipt",
        ),
        "observations": str(observations),
        "output": _binding_record(
            output,
            {"bytes": output.stat().st_size, "sha256": _sha256_file(output)},
            description="restored smoke output",
        ),
        "packaged_expected": _binding_record(
            expected,
            {"bytes": expected.stat().st_size, "sha256": _sha256_file(expected)},
            description="packaged smoke output",
        ),
        "tensor_identity": tensor_receipts,
        "smoke_input": smoke_input_receipt,
        "consumer_receipt": consumer_receipt,
        "post_execution_clean_runtime": post_execution_clean,
        "smoke_comparison": comparison,
        "started_unix": start_wall,
        "elapsed_seconds": elapsed,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "return_code": completed.returncode,
        "offline": True,
        "training_worktree_import": consumer_receipt["training_worktree_import"],
        "independent_evaluation_truth_opened": False,
    }


def restore_and_run_smoke(
    manifest_path: Path,
    clean_root: Path,
    *,
    require_windows_boundary: bool = True,
    require_distinct_devices: bool = True,
) -> dict[str, Any]:
    """Copy from the secondary boundary, verify tensors, and run exact smoke."""
    retrieval = materialize_clean_runtime(
        manifest_path,
        clean_root,
        source_boundary="secondary",
        require_windows_boundary=require_windows_boundary,
        require_distinct_devices=require_distinct_devices,
    )
    smoke = run_restored_smoke(
        manifest_path,
        clean_root,
        verify_tensors=True,
        require_windows_boundary=require_windows_boundary,
        require_distinct_devices=require_distinct_devices,
    )
    smoke["retrieval"] = retrieval
    smoke["source_boundary"] = retrieval["source_boundary"]
    return smoke


def tensor_digest(value: Any) -> str:
    """Hash shape, dtype, and contiguous CPU bytes for one tensor.

    This is the canonical identity algorithm for P11 tensor sidecars and smoke
    prediction receipts.
    """
    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def verify_tensor_identity(
    tensor_path: Path,
    identity_path: Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Open one safetensors file and verify every declared tensor digest.

    This is an explicit post-copy action. It is not called by metadata-only
    validation and does not run a model or load truth.
    """
    tensor_path = _resolved_path(tensor_path, description="tensor file", directory=False)
    identity_path = _resolved_path(identity_path, description="tensor identity", directory=False)
    actual_file_sha = _sha256_file(tensor_path)
    if actual_file_sha != expected_file_sha256:
        raise RestoreGateError("tensor file hash differs before tensor verification")
    payload = _read_json(identity_path, description="tensor identity")
    if payload.get("schema") != TENSOR_SCHEMA or payload.get("file_sha256") != actual_file_sha:
        raise RestoreGateError("tensor identity file binding changed")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise RestoreGateError("tensor identity inventory is empty")
    if payload.get("tensor_keys_sorted") != sorted(tensors):
        raise RestoreGateError("tensor identity names are not sorted")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RestoreGateError("safetensors is unavailable for tensor verification") from exc
    seen: set[str] = set()
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        if actual_keys != set(tensors):
            raise RestoreGateError("tensor key inventory differs")
        for key, expected in tensors.items():
            value = handle.get_tensor(key)
            seen.add(key)
            if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                raise RestoreGateError(f"tensor shape/dtype differs: {key}")
            if tensor_digest(value) != expected["sha256"]:
                raise RestoreGateError(f"tensor digest differs: {key}")
    return {
        "file_sha256": actual_file_sha,
        "tensor_count": len(seen),
        "tensor_keys": sorted(seen),
        "verified": True,
    }


def compare_smoke_prediction_files(
    packaged_path: Path,
    restored_path: Path,
    *,
    expected_tensor_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare ordered prediction tensors emitted by packaged/restored runners."""
    packaged_path = _resolved_path(packaged_path, description="packaged smoke output", directory=False)
    restored_path = _resolved_path(restored_path, description="restored smoke output", directory=False)
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RestoreGateError("safetensors is unavailable for smoke verification") from exc
    with safe_open(str(packaged_path), framework="pt", device="cpu") as left, safe_open(
        str(restored_path), framework="pt", device="cpu"
    ) as right:
        left_keys = list(left.keys())
        right_keys = list(right.keys())
        if left_keys != right_keys:
            raise RestoreGateError("smoke prediction key order differs")
        digests: dict[str, str] = {}
        for key in left_keys:
            left_value = left.get_tensor(key)
            right_value = right.get_tensor(key)
            if list(left_value.shape) != list(right_value.shape) or str(left_value.dtype) != str(right_value.dtype):
                raise RestoreGateError(f"smoke prediction geometry differs: {key}")
            left_digest = tensor_digest(left_value)
            right_digest = tensor_digest(right_value)
            if left_digest != right_digest:
                raise RestoreGateError(f"smoke prediction IDs differ: {key}")
            digests[key] = left_digest
    if expected_tensor_digests is not None:
        for key, expected in expected_tensor_digests.items():
            if digests.get(key) != expected:
                raise RestoreGateError(f"smoke expected digest differs: {key}")
    return {"verified": True, "keys": left_keys, "tensor_digests": digests}


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Validate or restore a TRR-P11 persistent package")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--restore-smoke", action="store_true")
    parser.add_argument("--clean-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--allow-test-boundaries", action="store_true")
    parser.add_argument("--no-clean-root", action="store_true")
    args = parser.parse_args()
    try:
        require_windows = not args.allow_test_boundaries
        require_devices = not args.allow_test_boundaries
        if args.restore_smoke:
            if args.clean_root is None:
                raise RestoreGateError("--clean-root is required with --restore-smoke")
            if args.no_clean_root:
                raise RestoreGateError("--no-clean-root cannot be used with --restore-smoke")
            report = restore_and_run_smoke(
                args.manifest,
                args.clean_root,
                require_windows_boundary=require_windows,
                require_distinct_devices=require_devices,
            )
        else:
            report = validate_restore_manifest(
                args.manifest,
                require_windows_boundary=require_windows,
                require_clean_root=not args.no_clean_root,
                require_distinct_devices=require_devices,
            )
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.receipt is not None:
            receipt = Path(args.receipt).expanduser()
            if not receipt.is_absolute():
                raise RestoreGateError("--receipt must be absolute")
            if receipt.exists() or receipt.is_symlink():
                raise RestoreGateError("receipt path already exists; receipts are create-only")
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except RestoreGateError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
