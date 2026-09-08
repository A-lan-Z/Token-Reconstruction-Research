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
from typing import Any

SCHEMA = "token-reconstruction.trr-p11-restore-gate.v1"
TENSOR_SCHEMA = "token-reconstruction.trr-p11-tensor-identity.v1"
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
    "package_manifest",
    "frozen_config",
    "selection_receipt",
    "smoke_input",
    "smoke_expected",
)
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
    for name in ("input_tensor_digests", "prediction_tensor_digests"):
        digests = smoke.get(name)
        if not isinstance(digests, Mapping) or not digests:
            raise RestoreGateError(f"smoke {name} are absent")
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
        "input_asset": "smoke_input",
        "expected_asset": "smoke_expected",
        "input_tensor_digests": dict(smoke["input_tensor_digests"]),
        "prediction_tensor_digests": dict(smoke["prediction_tensor_digests"]),
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
    for name in REQUIRED_ASSETS:
        asset = assets_payload[name]
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
                if int(asset["selected_step"]) < 0:
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
        if receipt_payload.get("status") != "SELECTION_COMPLETE_BEFORE_SMOKE":
            raise RestoreGateError("selection receipt status is not complete-before-smoke")
        if receipt_payload.get("smoke_used_for_selection") is not False:
            raise RestoreGateError("selection receipt records smoke leakage")
        if receipt_payload.get("independent_evaluation_truth_opened") is not False:
            raise RestoreGateError("selection receipt records opened evaluation truth")
        if not isinstance(receipt_payload.get("development_labels_used"), bool):
            raise RestoreGateError("selection receipt development_labels_used is absent")
        selected_methods = receipt_payload.get("selected_methods")
        if not isinstance(selected_methods, Mapping):
            raise RestoreGateError("selection receipt selected methods are absent")
        for method_name in METHOD_BANKS:
            selected = selected_methods.get(method_name)
            if not isinstance(selected, Mapping):
                raise RestoreGateError(f"selection receipt omits {method_name}")
            if selected.get("state_sha256") != assets[method_name]["copies"][boundary_name]["sha256"]:
                raise RestoreGateError(f"selection receipt state hash differs: {method_name}")
    if selection_payloads["primary"] != selection_payloads["secondary"]:
        raise RestoreGateError("selection receipt differs across copies")
    tensor_reports: dict[str, Any] = {}
    for name in TENSOR_ASSETS:
        tensor_reports[name] = _validate_tensor_identity_sidecar(
            name,
            assets_payload[name],
            boundaries=boundaries,
            state_records=assets[name]["copies"],
        )
    smoke_report = _validate_smoke(manifest, assets_payload)
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
    if source_boundary not in BOUNDARY_NAMES:
        raise RestoreGateError(f"unknown source boundary: {source_boundary}")
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
    parser = argparse.ArgumentParser(description="Validate a TRR-P11 persistent restore package")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-test-boundaries", action="store_true")
    parser.add_argument("--no-clean-root", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_restore_manifest(
            args.manifest,
            require_windows_boundary=not args.allow_test_boundaries,
            require_clean_root=not args.no_clean_root,
        )
    except RestoreGateError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
