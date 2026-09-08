"""TRR-0012 transfer adapter for the frozen expanded fixed decoder.

This adapter binds the actual TRR-0012 consumer package to the existing
truth-free TRR-0011 transfer bridge.  It does not select sources, load target
text/weights, open truth, or alter decoder/loss code.  The historical bridge
continues to own the transfer geometry and pairwise top-vs-runner diagnostic.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows has no resource module.
    _resource = None
import shlex
import sys
import time
from types import ModuleType
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts import trr0011_transfer as transfer


TASK_ID = "TRR-0012"
PACKAGE_SCHEMA = "token-reconstruction.trr0012-consumer-manifest.v1"
PACKAGE_STATUS = "FROZEN_SELECTED_PACKAGE"
PACKAGE_MANIFEST_SHA256 = "868ea07fec21643c387119df416e72aeaf932964566d4b84ad4fa45b5dc16d31"
PACKAGE_CODE_SHA256 = "5023a9e7cb19611952c0ce16a440ec9fde6b6444fd74aaf74a7efbb5c671c13b"
EXPANDED_METHOD = "expanded_fixed"
EXPANDED_STATE_SHA256 = "088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706"
EXPANDED_RUNNER_SHA256 = "209155048df936359870a1419803800d7bad4ab72a46e09bf3287648079964da"
EXPANDED_SCHEDULE_SHA256 = "8acdb2c4f8e5afba546ad01cbe0adae340eb841c8e3322c919e976434d29ede5"
BANK_FIT_SHA256 = "13c7442b481ba6bdaad2db14e3fd0b2a4300efefc48653dc6955e937d091c2d2"
BASE_STATE_SHA256 = "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14"
EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
CUDA_MIN_FREE_BYTES = 11 * 1024**3
CUDA_RESERVED_CEILING_BYTES = 8 * 1024**3
HOST_RSS_CAP_BYTES = 12 * 1024**3
BRIDGE_FEATURE_RTOL = 1e-5
BRIDGE_FEATURE_ATOL = 1e-6
BRIDGE_LOGIT_RTOL = 1e-5
BRIDGE_LOGIT_ATOL = 1e-6


class TransferAdapterError(RuntimeError):
    """Raised when the frozen actual package or transfer adapter is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, root: Path, readonly: bool = True) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TransferAdapterError(f"regular file required: {path}")
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise TransferAdapterError(f"adapter file is outside repository root: {path}") from exc
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if readonly:
        result["readonly"] = True
    return result


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransferAdapterError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TransferAdapterError(f"JSON object required: {path}")
    return value


def _package_path(package_root: Path, value: Any, *, label: str) -> Path:
    if isinstance(value, Mapping):
        value = value.get("path", value.get("relative_path"))
    if not isinstance(value, str) or not value:
        raise TransferAdapterError(f"{label} path is absent")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise TransferAdapterError(f"{label} path is not package-relative")
    resolved = (Path(package_root).resolve() / path).resolve()
    try:
        resolved.relative_to(Path(package_root).resolve())
    except ValueError as exc:
        raise TransferAdapterError(f"{label} escapes package root") from exc
    return resolved


def _import_package_code(package_root: Path) -> ModuleType:
    code_path = _package_path(package_root, "code/trr0012_package.py", label="package producer")
    if _sha256(code_path) != PACKAGE_CODE_SHA256:
        raise TransferAdapterError("actual package producer hash changed")
    module_name = "_trr0012_package_actual_5023a9e7"
    existing = __import__("sys").modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, code_path)
    if spec is None or spec.loader is None:
        raise TransferAdapterError("cannot import actual package producer")
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        __import__("sys").modules.pop(module_name, None)
        raise TransferAdapterError("actual package producer import failed") from exc
    return module


def _check_bound_file(package_root: Path, binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = _package_path(package_root, binding, label=label)
    actual = {
        "path": path,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    for key in ("bytes", "sha256"):
        if binding.get(key) != actual[key]:
            raise TransferAdapterError(f"{label} {key} changed")
    return actual


def bind_actual_package(package_root: Path, *, repository_root: Path) -> dict[str, Any]:
    """Validate and expose the frozen actual package's expanded method binding."""

    package_root = Path(package_root).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    manifest_path = package_root / "package_manifest.json"
    manifest_record = _record(manifest_path, root=repository_root)
    if manifest_record["sha256"] != PACKAGE_MANIFEST_SHA256:
        raise TransferAdapterError("actual package manifest hash changed")
    manifest = _json(manifest_path)
    if manifest.get("schema") != PACKAGE_SCHEMA or manifest.get("task_id") != TASK_ID:
        raise TransferAdapterError("actual package schema or task identity changed")
    if manifest.get("status") != PACKAGE_STATUS or manifest.get("immutable_after_selection") is not True:
        raise TransferAdapterError("actual package is not frozen and immutable")
    if manifest.get("consumer_paths_package_relative") is not True:
        raise TransferAdapterError("actual package consumer path boundary changed")
    methods = manifest.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {"current_fixed", EXPANDED_METHOD}:
        raise TransferAdapterError("actual package fixed-method binding changed")
    decoder = manifest.get("decoder")
    if not isinstance(decoder, Mapping) or decoder.get("package_cli_sha256") != PACKAGE_CODE_SHA256:
        raise TransferAdapterError("actual package decoder producer binding changed")
    producer_record = _record(
        _package_path(package_root, "code/trr0012_package.py", label="package producer"),
        root=repository_root,
    )
    expanded = methods.get(EXPANDED_METHOD)
    if not isinstance(expanded, Mapping):
        raise TransferAdapterError("expanded package method is absent")
    state_binding = expanded.get("state")
    readout_binding = expanded.get("public_readout") or manifest.get("readout")
    if not isinstance(state_binding, Mapping) or not isinstance(readout_binding, Mapping):
        raise TransferAdapterError("expanded package state/readout bindings are incomplete")
    state_actual = _check_bound_file(package_root, state_binding, label="expanded package state")
    readout_actual = _check_bound_file(package_root, readout_binding, label="expanded package readout")
    if state_actual["sha256"] != EXPANDED_STATE_SHA256:
        raise TransferAdapterError("expanded package state is not the frozen B1 state")
    if readout_actual["sha256"] != EMBEDDING_SHA256:
        raise TransferAdapterError("expanded package readout hash changed")
    loader_kwargs = expanded.get("loader_kwargs")
    if not isinstance(loader_kwargs, Mapping):
        loader_kwargs = expanded.get("loader", {}).get("kwargs") if isinstance(expanded.get("loader"), Mapping) else None
    if not isinstance(loader_kwargs, Mapping):
        raise TransferAdapterError("expanded package loader kwargs are absent")
    required_loader = {
        "expected_state_sha256": EXPANDED_STATE_SHA256,
        "expected_runner_state_sha256": EXPANDED_RUNNER_SHA256,
        "expected_schedule_semantic_sha256": EXPANDED_SCHEDULE_SHA256,
        "expected_bank_manifest_sha256": BANK_FIT_SHA256,
        "expected_fit_manifest_sha256": BANK_FIT_SHA256,
        "expected_base_state_sha256": BASE_STATE_SHA256,
        "expected_embedding_sha256": EMBEDDING_SHA256,
        "expected_selected_step": 13000,
    }
    for key, expected in required_loader.items():
        if loader_kwargs.get(key) != expected:
            raise TransferAdapterError(f"expanded loader binding changed: {key}")
    return {
        "schema": PACKAGE_SCHEMA,
        "status": PACKAGE_STATUS,
        "package_root": str(package_root),
        "manifest": manifest_record,
        "producer": producer_record,
        "method_id": EXPANDED_METHOD,
        "state": {
            "path": str(state_actual["path"]),
            "bytes": state_actual["bytes"],
            "sha256": state_actual["sha256"],
            "readonly": True,
        },
        "readout": {
            "path": str(readout_actual["path"]),
            "bytes": readout_actual["bytes"],
            "sha256": readout_actual["sha256"],
            "readonly": True,
        },
        "loader_kwargs": dict(loader_kwargs),
        "selected_step": 13000,
        "truth_boundary": dict(manifest.get("truth_boundary", {})),
    }


def _current_runtime_bindings(repository_root: Path) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).resolve()
    paths = {
        "scripts.trr0010_eval_runner": root / "scripts" / "trr0010_eval_runner.py",
        "scripts.trr0010_eval_gate": root / "scripts" / "trr0010_eval_gate.py",
        "scripts.trr0010_p09_fixed_loader": root / "scripts" / "trr0010_p09_fixed_loader.py",
        "token_reconstruction.trr0007_positionwise": root / "src" / "token_reconstruction" / "trr0007_positionwise.py",
    }
    return {name: _record(path, root=root) for name, path in paths.items()}


def _local_b1_registration(package: Mapping[str, Any], *, repository_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an in-memory one-method runner binding from the actual package."""

    root = Path(repository_root).resolve()
    code_bindings = _current_runtime_bindings(root)
    state = dict(package["state"])
    readout = dict(package["readout"])
    row = {
        "id": EXPANDED_METHOD,
        "state": state,
        "resources": {
            "base_decoder_state": state,
            "public_embedding_table": readout,
        },
        "loader": {
            "module": "scripts.trr0010_p09_fixed_loader",
            "function": "load_p09_fixed_state",
            "interface": "trr0010.current_h.full_vocabulary.v1",
            "current_h_only": True,
            "full_vocabulary": True,
            "history_enabled": False,
            "a2_enabled": False,
            "path_args": {},
            "tensor_args": {},
            "kwargs": dict(package["loader_kwargs"]),
        },
    }
    registration = {
        "schema": "token-reconstruction.trr0012-transfer-local-registration.v1",
        "task_id": TASK_ID,
        "status": "FROZEN_ACTUAL_PACKAGE_BINDING_BEFORE_TRANSFER_TRUTH",
        "runtime_embedding": readout,
        "methods": [row],
        "code_bindings": code_bindings,
        "method_selection": [EXPANDED_METHOD],
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    return registration, {"code_bindings": code_bindings}


def _write_create_only(path: Path, value: Mapping[str, Any], *, task_root: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    task_root = Path(task_root).expanduser().resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise TransferAdapterError(f"output is outside task root: {path}") from exc
    if path.exists() or path.is_symlink():
        raise TransferAdapterError(f"output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def qualify_expanded_smoke(
    *,
    package_root: Path,
    repository_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Qualify B1 features/logits/predictions on the four bundled opened rows."""

    root = Path(repository_root).resolve()
    package = bind_actual_package(package_root, repository_root=root)
    package_module = _import_package_code(Path(package["package_root"]))
    descriptor, descriptor_path = package_module._load_package_descriptor(Path(package["package_root"]))
    fixture_path = _package_path(Path(package["package_root"]), descriptor["smoke"]["input_path"], label="smoke fixture")
    expected_path = _package_path(Path(package["package_root"]), descriptor["smoke"]["expected_path"], label="smoke expected predictions")
    activations, masks, positions, slots = package_module._observation_batch(fixture_path)
    if int(activations.shape[0]) != 4 or list(slots) != list(package_module.SMOKE_RECORD_ORDER):
        raise TransferAdapterError("bundled smoke fixture order or size changed")
    readout, readout_path = package_module._load_readout(Path(package["package_root"]), descriptor)
    device = torch.device("cpu")
    model, loaded_state_sha, state_path = package_module._load_package_method(
        Path(package["package_root"]), descriptor, EXPANDED_METHOD, device=device, readout=readout
    )
    if loaded_state_sha != EXPANDED_STATE_SHA256 or _sha256(state_path) != EXPANDED_STATE_SHA256:
        raise TransferAdapterError("ordinary package loader did not load the frozen B1 state")
    with safe_open(str(expected_path), framework="pt", device="cpu") as handle:
        expected_predictions = handle.get_tensor(EXPANDED_METHOD).detach().cpu().contiguous()
    expected_manifest = descriptor.get("smoke", {}).get("expected_outputs")
    if tuple(expected_predictions.shape) != (4, 128):
        raise TransferAdapterError("smoke expected B1 prediction geometry changed")
    records: list[dict[str, Any]] = []
    prediction_rows: list[torch.Tensor] = []
    for index, slot in enumerate(slots):
        activation = activations[index].to(device=device, dtype=torch.float32).unsqueeze(0)
        mask = masks[index].to(device=device, dtype=torch.bool).unsqueeze(0)
        position = positions[index].to(device=device, dtype=torch.long)
        with torch.inference_mode():
            projected = model.projected_hidden(activation, mask)
            rows = torch.zeros_like(position[1:])
            logits = model.logits_from_rows(projected, rows, position[1:], readout)
            prediction = torch.empty(128, dtype=torch.long)
            prediction[0] = package_module.BOS_TOKEN_ID
            prediction[1:] = logits.argmax(dim=-1).to(device="cpu", dtype=torch.long)
        if tuple(projected.shape) != (1, 128, 2048) or not bool(torch.isfinite(projected.float()).all().item()):
            raise TransferAdapterError(f"B1 projected feature geometry/finiteness changed: {slot}")
        features = projected[0, 1:].detach().cpu().contiguous()
        if tuple(logits.shape) != (127, 128256) or not bool(torch.isfinite(logits.float()).all().item()):
            raise TransferAdapterError(f"B1 logits geometry/finiteness changed: {slot}")
        norms = torch.linalg.vector_norm(features.float(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=2e-3, atol=2e-3):
            raise TransferAdapterError(f"B1 projected features are not normalized: {slot}")
        if not torch.equal(prediction, expected_predictions[index]):
            raise TransferAdapterError(f"B1 prediction differs from frozen smoke output: {slot}")
        if isinstance(expected_manifest, Mapping):
            expected_slot = expected_manifest.get(slot, {}).get(EXPANDED_METHOD) if isinstance(expected_manifest.get(slot), Mapping) else None
            if isinstance(expected_slot, Mapping) and prediction.tolist() != expected_slot.get("token_ids"):
                raise TransferAdapterError(f"B1 prediction differs from manifest smoke identity: {slot}")
        prediction_rows.append(prediction)
        records.append(
            {
                "slot": slot,
                "feature_shape": list(features.shape),
                "feature_dtype": str(features.dtype),
                "feature_tensor_sha256": package_module.tensor_digest(features),
                "logit_shape": list(logits.shape),
                "logit_dtype": str(logits.dtype),
                "logit_tensor_sha256": package_module.tensor_digest(logits),
                "prediction_tensor_sha256": package_module.tensor_digest(prediction),
                "prediction_matches_frozen_smoke": True,
                "top_class_rule": "full_vocabulary_argmax",
                "nearest_boundary_claim": False,
            }
        )
    predictions = torch.stack(prediction_rows, dim=0).contiguous()
    receipt = {
        "schema": "token-reconstruction.trr0012-expanded-transfer-qualification.v1",
        "task_id": TASK_ID,
        "status": "PASS_FROZEN_B1_SMOKE_FEATURE_LOGIT_PREDICTION_EQUIVALENCE",
        "method_id": EXPANDED_METHOD,
        "scope": "four already-opened package smoke records; no target capture/source selection/truth",
        "package": {
            "root": package["package_root"],
            "manifest": package["manifest"],
            "manifest_sha256": package["manifest"]["sha256"],
            "producer": package["producer"],
            "producer_sha256": package["producer"]["sha256"],
            "descriptor": {"path": str(descriptor_path), "sha256": _sha256(descriptor_path)},
            "state": {"path": str(state_path), "sha256": _sha256(state_path), "selected_step": 13000},
            "readout": {"path": str(readout_path), "sha256": _sha256(readout_path)},
            "expected_predictions": {"path": str(expected_path), "sha256": _sha256(expected_path)},
            "loaded_state_sha256": loaded_state_sha,
        },
        "observations": {
            "path": str(fixture_path),
            "sha256": _sha256(fixture_path),
            "records": len(slots),
            "record_order": list(slots),
            "truth_opened": False,
            "source_text_loaded": False,
            "token_ids_loaded": False,
        },
        "checks": {
            "features_finite_normalized": True,
            "logits_finite_full_vocabulary": True,
            "prediction_argmax_matches_logits": True,
            "prediction_matches_frozen_smoke": True,
            "pairwise_top_runner_only": True,
            "global_nearest_boundary_claim": False,
            "deployable_confidence_certificate": False,
        },
        "prediction_tensor_sha256": package_module.tensor_digest(predictions),
        "records_detail": records,
        "created_at_unix": time.time(),
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    return receipt | {"receipt_record": _write_create_only(receipt_path, receipt, task_root=root / "experiments" / TASK_ID)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_memory_snapshot() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) >= 2:
                    result["process_rss_bytes"] = int(fields[1]) * 1024
                break
    except (OSError, UnicodeError, ValueError, IndexError):
        pass
    if _resource is not None:
        try:
            usage = _resource.getrusage(_resource.RUSAGE_SELF)
            unit = 1024 if sys.platform.startswith("linux") else 1
            result["process_peak_rss_bytes"] = int(usage.ru_maxrss * unit)
        except (OSError, ValueError):
            pass
    return result


def _cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_numerical_settings() -> dict[str, Any]:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _materialize_bridge_smoke_input(
    *,
    package_module: ModuleType,
    fixture_path: Path,
    output_path: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    if output_path.exists() or output_path.is_symlink():
        raise TransferAdapterError(f"bridge smoke input is create-only: {output_path}")
    activations, masks, positions, slots = package_module._observation_batch(fixture_path)
    if int(activations.shape[0]) != 4 or list(slots) != list(package_module.SMOKE_RECORD_ORDER):
        raise TransferAdapterError("bridge smoke fixture must contain the four fixed opened rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "activations": activations.contiguous(),
            "attention_mask": masks.to(dtype=torch.bool).contiguous(),
            "position_ids": positions.to(dtype=torch.int64).contiguous(),
        },
        str(output_path),
        metadata={
            "schema": "token-reconstruction.trr0012-bridge-qualification-input.v1",
            "task_id": TASK_ID,
            "record_order": json.dumps(list(slots), separators=(",", ":")),
            "truth_opened": "false",
            "source_text_loaded": "false",
            "token_ids_loaded": "false",
        },
    )
    return _record(output_path, root=repository_root), list(slots), activations, masks, positions


def _model_feature_logit_prediction(
    model: torch.nn.Module,
    readout: torch.Tensor,
    activation: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    *,
    device: torch.device,
    bos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        staged = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
        staged_mask = mask.to(device=device, dtype=torch.bool).unsqueeze(0)
        staged_positions = positions.to(device=device, dtype=torch.long)
        try:
            projected = model.projected_hidden(staged, staged_mask)
            rows = torch.zeros_like(staged_positions[1:])
            logits = model.logits_from_rows(projected, rows, staged_positions[1:], readout)
        except AttributeError:
            logits_full = model(staged, staged_mask, readout)
            logits = logits_full[0, 1:]
        if tuple(projected.shape) != (1, transfer.STORED_SEQUENCE_TOKENS, transfer.HIDDEN_SIZE):
            raise TransferAdapterError("changed-path projected feature geometry changed")
        if tuple(logits.shape) != (transfer.SCORED_POST_BOS_TOKENS, transfer.VOCABULARY_SIZE):
            raise TransferAdapterError("changed-path logits geometry changed")
        if not bool(torch.isfinite(projected.float()).all().item()) or not bool(torch.isfinite(logits.float()).all().item()):
            raise TransferAdapterError("changed-path features or logits are non-finite")
        features_cpu = projected[0, 1:].detach().cpu().contiguous()
        logits_cpu = logits.detach().cpu().contiguous()
        prediction = torch.empty(transfer.STORED_SEQUENCE_TOKENS, dtype=torch.long)
        prediction[0] = int(bos_token_id)
        prediction[1:] = logits_cpu.argmax(dim=-1).to(dtype=torch.long)
    return features_cpu, logits_cpu, prediction


def _compare_cuda_tensors(
    package_value: torch.Tensor,
    production_value: torch.Tensor,
    *,
    label: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = package_value.detach().cpu().float()
    right = production_value.detach().cpu().float()
    if tuple(left.shape) != tuple(right.shape):
        raise TransferAdapterError(f"{label} geometry differs: {tuple(left.shape)} != {tuple(right.shape)}")
    delta = (left - right).abs()
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    max_rel = float((delta / right.abs().clamp_min(atol)).max().item()) if delta.numel() else 0.0
    if not torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
        raise TransferAdapterError(
            f"{label} differs between package and production paths: max_abs={max_abs} max_rel={max_rel}"
        )
    return {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "max_abs_difference": max_abs,
        "max_relative_difference": max_rel,
        "rtol": rtol,
        "atol": atol,
        "equivalent": True,
    }


def qualify_expanded_bridge_smoke(
    *,
    package_root: Path,
    repository_root: Path,
    receipt_path: Path,
    device_name: str = "cuda",
    command: str | None = None,
) -> dict[str, Any]:
    """Compare package and production registration/loader paths on four smoke rows."""

    root = Path(repository_root).resolve()
    started_utc = _utc_now()
    started = time.perf_counter()
    if not str(device_name).startswith("cuda"):
        raise TransferAdapterError("changed-path qualification requires CUDA; CPU v1 is already preserved separately")
    if not torch.cuda.is_available():
        raise TransferAdapterError("changed-path qualification requires an available CUDA device")
    device = torch.device(device_name)
    free_before, total_before = torch.cuda.mem_get_info(device)
    if int(free_before) < CUDA_MIN_FREE_BYTES:
        raise TransferAdapterError(
            f"CUDA preflight free memory is below 11 GiB: {int(free_before)} bytes"
        )
    torch.cuda.reset_peak_memory_stats(device)
    numerical_settings = _cuda_numerical_settings()
    package = bind_actual_package(package_root, repository_root=root)
    package_module = _import_package_code(Path(package["package_root"]))
    descriptor, descriptor_path = package_module._load_package_descriptor(Path(package["package_root"]))
    fixture_path = _package_path(Path(package["package_root"]), descriptor["smoke"]["input_path"], label="smoke fixture")
    expected_path = _package_path(Path(package["package_root"]), descriptor["smoke"]["expected_path"], label="smoke expected predictions")
    bridge_input_path = Path(receipt_path).expanduser().resolve().parent / "qualification_b1_bridge_smoke_input.safetensors"
    input_record, slots, activations, masks, positions = _materialize_bridge_smoke_input(
        package_module=package_module,
        fixture_path=fixture_path,
        output_path=bridge_input_path,
        repository_root=root,
    )
    with safe_open(str(expected_path), framework="pt", device="cpu") as handle:
        expected_predictions = handle.get_tensor(EXPANDED_METHOD).detach().cpu().contiguous()
    if tuple(expected_predictions.shape) != (4, transfer.STORED_SEQUENCE_TOKENS):
        raise TransferAdapterError("frozen expected smoke prediction geometry changed")

    package_started = time.perf_counter()
    package_readout, package_readout_path = package_module._load_readout(Path(package["package_root"]), descriptor)
    package_readout_device = package_readout.to(device=device).contiguous()
    package_model, package_state_sha, package_state_path = package_module._load_package_method(
        Path(package["package_root"]), descriptor, EXPANDED_METHOD, device=device, readout=package_readout_device
    )
    _sync_cuda(device)
    package_rows: list[dict[str, torch.Tensor]] = []
    for index, slot in enumerate(slots):
        features, logits, prediction = _model_feature_logit_prediction(
            package_model,
            package_readout_device,
            activations[index],
            masks[index],
            positions[index],
            device=device,
            bos_token_id=package_module.BOS_TOKEN_ID,
        )
        if not torch.equal(prediction, expected_predictions[index]):
            raise TransferAdapterError(f"package CUDA prediction differs from frozen smoke output: {slot}")
        package_rows.append({"features": features, "logits": logits, "prediction": prediction})
    _sync_cuda(device)
    package_elapsed = time.perf_counter() - package_started
    package_memory = _cuda_memory_snapshot(device) | _process_memory_snapshot()
    del package_model, package_readout_device, package_readout
    torch.cuda.empty_cache()
    _sync_cuda(device)

    production_started = time.perf_counter()
    runner_module = __import__("scripts.trr0010_eval_runner", fromlist=["*"])
    registration, checked_registration = _local_b1_registration(package, repository_root=root)
    source_check = transfer._validate_historical_runner_sources(
        registration, repository_root=root, runner_module=runner_module
    )
    production_embedding, embedding_evidence = runner_module._load_embedding(
        registration, root=root, device=device
    )
    loaded = runner_module._load_method(
        EXPANDED_METHOD,
        registration["methods"][0],
        root=root,
        device=device,
        embedding=production_embedding,
        method_factory=None,
        code_bindings=checked_registration["code_bindings"],
        allow_materialization=False,
    )
    _sync_cuda(device)
    production_rows: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        features, logits, prediction = _model_feature_logit_prediction(
            loaded.adapter.model,
            loaded.adapter.embedding,
            activations[index],
            masks[index],
            positions[index],
            device=device,
            bos_token_id=transfer.BOS_TOKEN_ID,
        )
        adapter_prediction = loaded.adapter(activations[index], masks[index], positions[index])
        adapter_prediction = torch.as_tensor(adapter_prediction).detach().cpu().to(dtype=torch.long).contiguous()
        if not torch.equal(prediction, adapter_prediction):
            raise TransferAdapterError(f"production adapter call differs from direct production logits: {slot}")
        feature_check = _compare_cuda_tensors(
            package_rows[index]["features"], features,
            label=f"projected features {slot}", rtol=BRIDGE_FEATURE_RTOL, atol=BRIDGE_FEATURE_ATOL,
        )
        logit_check = _compare_cuda_tensors(
            package_rows[index]["logits"], logits,
            label=f"full logits {slot}", rtol=BRIDGE_LOGIT_RTOL, atol=BRIDGE_LOGIT_ATOL,
        )
        if not torch.equal(package_rows[index]["prediction"], prediction) or not torch.equal(prediction, adapter_prediction):
            raise TransferAdapterError(f"production prediction differs from package path: {slot}")
        production_rows.append(
            {
                "slot": slot,
                "features": feature_check,
                "logits": logit_check,
                "prediction_shape": list(prediction.shape),
                "prediction_tensor_sha256": package_module.tensor_digest(prediction),
                "package_prediction_tensor_sha256": package_module.tensor_digest(package_rows[index]["prediction"]),
                "adapter_prediction_exact": True,
                "prediction_equivalent": True,
            }
        )
    _sync_cuda(device)
    production_elapsed = time.perf_counter() - production_started
    production_memory = _cuda_memory_snapshot(device) | _process_memory_snapshot()
    final_memory = _cuda_memory_snapshot(device) | _process_memory_snapshot()
    peak_reserved = int(final_memory.get("peak_reserved_bytes", 0))
    peak_rss = int(final_memory.get("process_peak_rss_bytes", 0))
    if peak_reserved > CUDA_RESERVED_CEILING_BYTES:
        raise TransferAdapterError(f"CUDA reserved peak exceeded 8 GiB: {peak_reserved} bytes")
    if peak_rss > HOST_RSS_CAP_BYTES:
        raise TransferAdapterError(f"process RSS peak exceeded 12 GiB: {peak_rss} bytes")
    finished_utc = _utc_now()
    receipt = {
        "schema": "token-reconstruction.trr0012-expanded-transfer-bridge-cuda-qualification.v1",
        "task_id": TASK_ID,
        "status": "PASS_PRODUCTION_BRIDGE_PACKAGE_CUDA_EQUIVALENCE",
        "method_id": EXPANDED_METHOD,
        "command": command or shlex.join([sys.executable, *sys.argv]),
        "scope": "four already-opened package smoke records; no target capture/source selection/truth",
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "wall_seconds": float(time.perf_counter() - started),
        "device": {
            "requested": str(device_name),
            "resolved": str(device),
            "name": torch.cuda.get_device_name(device),
            "index": int(device.index if device.index is not None else torch.cuda.current_device()),
            "total_memory_bytes": int(total_before),
            "preflight_free_bytes": int(free_before),
            "preflight_min_free_bytes": CUDA_MIN_FREE_BYTES,
        },
        "numerical_settings": numerical_settings,
        "source_bindings": {
            "actual_package_manifest": package["manifest"],
            "actual_package_producer": package["producer"],
            "package_descriptor": {"path": str(descriptor_path), "sha256": _sha256(descriptor_path)},
            "package_loader": {"path": str(_package_path(Path(package["package_root"]), "code/trr0010_p09_fixed_loader.py", label="package loader")), "sha256": _sha256(_package_path(Path(package["package_root"]), "code/trr0010_p09_fixed_loader.py", label="package loader"))},
            "package_decoder": {"path": str(_package_path(Path(package["package_root"]), "code/token_reconstruction/trr0007_positionwise.py", label="package decoder")), "sha256": _sha256(_package_path(Path(package["package_root"]), "code/token_reconstruction/trr0007_positionwise.py", label="package decoder"))},
            "production_runner": _record(root / "scripts" / "trr0010_eval_runner.py", root=root),
            "production_loader": _record(root / "scripts" / "trr0010_p09_fixed_loader.py", root=root),
            "production_decoder": _record(root / "src" / "token_reconstruction" / "trr0007_positionwise.py", root=root),
            "transfer_adapter": _record(Path(__file__).resolve(), root=root),
        },
        "package_binding": {
            "state": {"path": str(package_state_path), "sha256": package_state_sha, "selected_step": 13000},
            "readout": {"path": str(package_readout_path), "sha256": _sha256(package_readout_path)},
            "loaded_state_sha256": package_state_sha,
            "production_loader_state_sha256": package["state"]["sha256"],
        },
        "observations": {
            "source_fixture": {"path": str(fixture_path), "sha256": _sha256(fixture_path)},
            "bridge_input": input_record,
            "records": len(slots),
            "record_order": list(slots),
            "truth_opened": False,
            "source_text_loaded": False,
            "token_ids_loaded": False,
        },
        "phases": {
            "package_load_and_four_rows": {"elapsed_seconds": float(package_elapsed), "memory": package_memory},
            "production_registration_loader_and_four_rows": {"elapsed_seconds": float(production_elapsed), "memory": production_memory, "embedding": embedding_evidence, "source_check": source_check},
        },
        "memory": {
            "peak_gpu_allocated_bytes": int(final_memory.get("peak_allocated_bytes", 0)),
            "peak_gpu_reserved_bytes": peak_reserved,
            "process_rss_bytes": int(final_memory.get("process_rss_bytes", 0)),
            "process_peak_rss_bytes": peak_rss,
            "reserved_ceiling_bytes": CUDA_RESERVED_CEILING_BYTES,
            "host_rss_cap_bytes": HOST_RSS_CAP_BYTES,
        },
        "equivalence": {
            "feature_rtol": BRIDGE_FEATURE_RTOL,
            "feature_atol": BRIDGE_FEATURE_ATOL,
            "logit_rtol": BRIDGE_LOGIT_RTOL,
            "logit_atol": BRIDGE_LOGIT_ATOL,
            "rows": production_rows,
            "all_features_equivalent": True,
            "all_full_logits_equivalent": True,
            "all_predictions_equivalent": True,
            "production_adapter_call_equivalent": True,
            "pairwise_top_runner_only": True,
            "global_nearest_boundary_claim": False,
            "deployable_confidence_certificate": False,
        },
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    return receipt | {"receipt_record": _write_create_only(receipt_path, receipt, task_root=root / "experiments" / TASK_ID)}


def _external_readonly_record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    """Bind an explicitly supplied sanitized file outside this worktree."""

    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise TransferAdapterError(f"{description} must be a regular file: {path}")
    return transfer._transfer_file_record(  # noqa: SLF001
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "readonly": True,
        },
        root=root,
        description=description,
    )


def _adapt_external_capture_bindings(capture: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Mark only explicitly bound external sanitized observations read-only.

    The evaluator manifest predates this repository boundary marker.  Adding
    the marker to an in-memory copy preserves every declared observation path,
    byte count, and SHA-256 while allowing the existing read-only checker to
    validate the external files.
    """

    adapted = json.loads(json.dumps(dict(capture)))
    observations = adapted.get("observations")
    if not isinstance(observations, Mapping):
        raise TransferAdapterError("capture manifest observations are absent")
    for variant_id, domain_bindings in observations.items():
        if not isinstance(domain_bindings, Mapping):
            raise TransferAdapterError(f"capture observation domains are malformed: {variant_id}")
        for domain, binding in domain_bindings.items():
            if not isinstance(binding, dict):
                raise TransferAdapterError(f"capture observation binding is malformed: {variant_id}/{domain}")
            raw_path = binding.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise TransferAdapterError(f"capture observation path is absent: {variant_id}/{domain}")
            path = Path(raw_path).expanduser().resolve()
            try:
                path.relative_to(root)
            except ValueError:
                if binding.get("readonly") not in (None, True):
                    raise TransferAdapterError(
                        f"external capture observation is explicitly non-readonly: {variant_id}/{domain}"
                    )
                binding["readonly"] = True
    return adapted


def _build_transfer_manifest(
    *,
    capture_path: Path,
    package: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    capture = _adapt_external_capture_bindings(_json(capture_path), root=root)
    code = _current_runtime_bindings(root)
    code["scripts.trr0011_transfer"] = _record(root / "scripts" / "trr0011_transfer.py", root=root)
    code["scripts.trr0012_transfer"] = _record(Path(__file__).resolve(), root=root)
    code["scripts.trr0012_package"] = dict(package["producer"])
    manifest = transfer.build_transfer_manifest_from_capture(
        capture,
        repository_root=root,
        method_id=EXPANDED_METHOD,
        code_bindings=code,
        state_bindings={"decoder_state": dict(package["state"])},
        decoder_resources={
            "embedding": dict(package["readout"]),
            "state": dict(package["state"]),
            "loader": code["scripts.trr0010_p09_fixed_loader"],
        },
    )
    return manifest | {
        "capture_manifest_source": _external_readonly_record(
            capture_path,
            root=root,
            description="sanitized capture manifest",
        ),
        "package_actual": package["manifest"],
        "package_producer": package["producer"],
        "method_selection": [EXPANDED_METHOD],
        "diagnostic_scope": "pairwise top-vs-runner diagnostic only",
        "nearest_boundary_claim": False,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }


def _matrix_resource_guard(
    *,
    runner_module: ModuleType,
    device: torch.device,
    output_root: Path,
    started: float,
) -> tuple[Callable[[str], None], dict[str, Any]]:
    """Reuse the reviewed native runner guard for the one-process fixed matrix.

    The capture policy's one-variant-per-process rule applies to public target
    capture.  The decoder bridge intentionally loads the frozen fixed decoder
    once and evaluates all seven variants across both domains in one
    restart-safe process.  The native runner's whole-matrix guard is therefore
    used at the same record/cell boundaries, with the fixed 8 GiB GPU cap.
    """

    registered_caps = dict(runner_module.REGISTERED_INFERENCE_CAPS)
    registered_caps["cuda_reserved_limit_bytes"] = int(
        registered_caps["fixed_cuda_reserved_limit_bytes"]
    )
    state: dict[str, Any] = {
        "checks": 0,
        "guard_overhead_seconds": 0.0,
        "first_snapshot": None,
        "last_snapshot": None,
    }

    def check(stage: str) -> None:
        guard_started = time.perf_counter()
        try:
            snapshot = runner_module._resource_snapshot(  # noqa: SLF001
                device=device,
                output_root=output_root,
            )
            runner_module._enforce_resource_guard(  # noqa: SLF001
                snapshot,
                registered_caps,
                started=started,
                stage=stage,
                require_gpu=device.type == "cuda",
                gpu_peak_field="max_reserved_bytes",
            )
            if state["first_snapshot"] is None:
                state["first_snapshot"] = dict(snapshot)
            state["last_snapshot"] = dict(snapshot)
        finally:
            state["checks"] += 1
            state["guard_overhead_seconds"] += time.perf_counter() - guard_started

    state["registered_caps"] = registered_caps
    state["scope"] = (
        "existing trr0010 live host/GPU/disk checks run outside timed decoder "
        "intervals; the outer process watchdog owns wall time; one process "
        "covers expanded_fixed across seven variants and two domains"
    )
    return check, state


def run_expanded_transfer_matrix(
    *,
    capture_manifest_path: Path,
    variant_plan_path: Path,
    panel_descriptor_path: Path,
    package_root: Path,
    repository_root: Path,
    output_root: Path,
    device_name: str = "cuda",
    method_ids: Sequence[str] = (EXPANDED_METHOD,),
) -> dict[str, Any]:
    """Run the approved transfer bridge for the selected frozen method subset."""

    selected = tuple(str(value) for value in method_ids)
    if selected != (EXPANDED_METHOD,):
        raise TransferAdapterError("TRR-0012 transfer execution is restricted to expanded_fixed")
    root = Path(repository_root).resolve()
    task_root = root / "experiments" / TASK_ID
    package = bind_actual_package(package_root, repository_root=root)
    plan_path = Path(variant_plan_path).resolve()
    panel_path = Path(panel_descriptor_path).resolve()
    plan = _json(plan_path)
    panel = _json(panel_path)
    transfer.validate_variant_plan(plan)
    transfer.validate_panel_descriptor(panel, strict_reservation=True)
    capture_path = Path(capture_manifest_path).resolve()
    manifest = _build_transfer_manifest(capture_path=capture_path, package=package, repository_root=root)
    manifest_path = Path(output_root).resolve() / "transfer_input_expanded_fixed.json"
    manifest_record = _write_create_only(manifest_path, manifest, task_root=task_root)
    checked = transfer.validate_transfer_observation_manifest(manifest, repository_root=root)
    runner_module = __import__("scripts.trr0010_eval_runner", fromlist=["*"])
    registration, checked_registration = _local_b1_registration(package, repository_root=root)
    source_check = transfer._validate_historical_runner_sources(registration, repository_root=root, runner_module=runner_module)
    device = torch.device(device_name)
    output_root = Path(output_root).resolve()
    matrix_started = time.perf_counter()
    guard_check, guard_state = _matrix_resource_guard(
        runner_module=runner_module,
        device=device,
        output_root=output_root,
        started=matrix_started,
    )
    guard_check("initial")
    embedding, embedding_evidence = runner_module._load_embedding(registration, root=root, device=device)
    loaded = runner_module._load_method(
        EXPANDED_METHOD,
        registration["methods"][0],
        root=root,
        device=device,
        embedding=embedding,
        method_factory=None,
        code_bindings=checked_registration["code_bindings"],
        allow_materialization=False,
    )
    guard_check("after_expanded_fixed_load")
    predictions: dict[str, Any] = {}
    geometries: dict[str, Any] = {}
    for variant_id in checked["variant_ids"]:
        for domain, binding in checked["observation_bindings"][variant_id].items():
            cell = {"observation": binding}
            records = int(binding["records"])
            values, timing = runner_module._run_cell(
                adapter=loaded.adapter,
                cell=cell,
                records=records,
                hidden_size=transfer.HIDDEN_SIZE,
                device=device,
                method_id=EXPANDED_METHOD,
                guard_callback=lambda stage, variant_id=variant_id, domain=domain: guard_check(
                    f"{EXPANDED_METHOD}/{variant_id}/{domain}/{stage}"
                ),
            )
            base = output_root / EXPANDED_METHOD / variant_id / domain
            prediction_record = transfer._write_transfer_prediction(
                base / "predictions.safetensors",
                values,
                root=root,
                method_id=EXPANDED_METHOD,
                variant_id=variant_id,
                domain=domain,
                records=records,
                observation_sha256=str(binding["sha256"]),
                output_task_root=task_root,
            ) | {"timing": timing}
            geometry = transfer._collect_decoder_geometry(
                loaded.adapter,
                cell=cell,
                records=records,
                hidden_size=transfer.HIDDEN_SIZE,
                runner_module=runner_module,
            )
            geometry_record = transfer._write_transfer_geometry(
                base / "decoder_geometry.safetensors",
                geometry,
                root=root,
                method_id=EXPANDED_METHOD,
                variant_id=variant_id,
                domain=domain,
                records=records,
                observation_sha256=str(binding["sha256"]),
                logit_scale=transfer._adapter_logit_scale(loaded.adapter),
                output_task_root=task_root,
            )
            key = f"{EXPANDED_METHOD}/{variant_id}/{domain}"
            predictions[key] = {name: prediction_record[name] for name in ("path", "bytes", "sha256", "prediction_sha256")}
            geometries[key] = {name: geometry_record[name] for name in ("path", "bytes", "sha256", "tensor_digests")}
            guard_check(f"{EXPANDED_METHOD}/{variant_id}/{domain}/after_write")
    matrix = {
        "schema": transfer.TRANSFER_MATRIX_SCHEMA,
        "task_id": transfer.TASK_ID,
        "status": transfer.TRANSFER_MATRIX_STATUS,
        "method_ids": [EXPANDED_METHOD],
        "variant_ids": list(checked["variant_ids"]),
        "domain_order": list(transfer.TRANSFER_DOMAIN_ORDER),
        "records_per_domain": transfer.TRANSFER_RECORDS_PER_DOMAIN,
        "cell_count": len(predictions),
        "input_manifest": manifest_record,
        "capture_manifest": _external_readonly_record(
            capture_path,
            root=root,
            description="sanitized capture manifest",
        ),
        "actual_package": package["manifest"],
        "variant_plan": _record(plan_path, root=root),
        "panel_descriptor": _record(panel_path, root=root),
        "code_bindings": _current_runtime_bindings(root),
        "method_selection": [EXPANDED_METHOD],
        "diagnostic_scope": "pairwise top-vs-runner diagnostic only",
        "nearest_boundary_claim": False,
        "predictions": predictions,
        "decoder_geometry": geometries,
        "resource_guard": {
            key: value
            for key, value in guard_state.items()
            if key in {"registered_caps", "checks", "guard_overhead_seconds", "first_snapshot", "last_snapshot", "scope"}
        },
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "private_or_truth_payload_read": False,
        "fresh_evaluation_started": False,
    }
    matrix_record = _write_create_only(Path(output_root) / "prediction_matrix_expanded_fixed.json", matrix, task_root=task_root)
    checked_matrix = transfer.validate_transfer_prediction_matrix(
        matrix,
        repository_root=root,
        expected_method_ids=(EXPANDED_METHOD,),
    )
    return {
        "status": transfer.TRANSFER_MATRIX_STATUS,
        "method_ids": [EXPANDED_METHOD],
        "matrix": checked_matrix,
        "matrix_record": matrix_record,
        "source_check": source_check,
        "embedding": embedding_evidence,
        "truth_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    qualify = sub.add_parser("qualify-smoke")
    qualify.add_argument("--package-root", type=Path, required=True)
    qualify.add_argument("--repository-root", type=Path, default=Path("."))
    qualify.add_argument("--receipt", type=Path, required=True)
    bridge = sub.add_parser("qualify-bridge-smoke")
    bridge.add_argument("--package-root", type=Path, required=True)
    bridge.add_argument("--repository-root", type=Path, default=Path("."))
    bridge.add_argument("--receipt", type=Path, required=True)
    bridge.add_argument("--device", default="cuda")
    matrix = sub.add_parser("run-matrix")
    matrix.add_argument("--package-root", type=Path, required=True)
    matrix.add_argument("--capture-manifest", type=Path, required=True)
    matrix.add_argument("--variant-plan", type=Path, required=True)
    matrix.add_argument("--panel", type=Path, required=True)
    matrix.add_argument("--repository-root", type=Path, default=Path("."))
    matrix.add_argument("--output-root", type=Path, required=True)
    matrix.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "qualify-smoke":
            result = qualify_expanded_smoke(
                package_root=args.package_root,
                repository_root=args.repository_root,
                receipt_path=args.receipt,
            )
        elif args.command == "qualify-bridge-smoke":
            result = qualify_expanded_bridge_smoke(
                package_root=args.package_root,
                repository_root=args.repository_root,
                receipt_path=args.receipt,
                device_name=args.device,
                command=shlex.join([sys.executable, *sys.argv]),
            )
        else:
            result = run_expanded_transfer_matrix(
                capture_manifest_path=args.capture_manifest,
                variant_plan_path=args.variant_plan,
                panel_descriptor_path=args.panel,
                package_root=args.package_root,
                repository_root=args.repository_root,
                output_root=args.output_root,
                device_name=args.device,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (TransferAdapterError, transfer.TransferDiagnosticError, RuntimeError, OSError) as exc:
        print(f"TRR0012 transfer adapter error: {exc}")
        return 2


__all__ = [
    "EXPANDED_METHOD",
    "TransferAdapterError",
    "bind_actual_package",
    "qualify_expanded_bridge_smoke",
    "qualify_expanded_smoke",
    "run_expanded_transfer_matrix",
]


if __name__ == "__main__":
    raise SystemExit(main())
