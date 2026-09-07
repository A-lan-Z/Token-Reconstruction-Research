"""Fail-closed TRR-0010 directional largest-cell qualifier.

This entrypoint is a bounded resource probe, not a fitting runner.  An A2
provider supplies the already-reviewed decoder, streamed random-access source,
serialized schedule steps, and shared runner module.  This module verifies
immutable bindings and lease caps, builds a zero-initialized directional arm,
checks exact zero-delta equivalence, then calls A2 ``run_training`` for a
small predeclared validation-aware discarded probe.  It never retains a selected checkpoint,
opens evaluation truth, or retains a contender.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch
from torch import nn

from token_reconstruction.trr0007_positionwise import ResidualMLPPositionwiseDecoder
from trr0010_model import (
    DEFAULT_ANCHOR_STRENGTH,
    DEFAULT_DIRECTIONAL_LEARNING_RATE,
    DirectionalTokenReadout,
    build_directional_from_base,
)
from trr0010_p09_caller import P09CallerError, validate_optimizer_configuration


TASK_ID = "TRR-0010"
QUALIFIER_SCHEMA = "token-reconstruction.trr0010-directional-qualifier.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0010-directional-qualifier-failure.v1"
REQUIRED_ARTIFACTS = {
    "contract": "FROZEN",
    "bank_manifest": "VERIFIED",
    "schedule": "FROZEN",
    "base_state": "SELECTED",
    "public_embedding": "PUBLIC_NORMALIZED",
    "support_ids": "VERIFIED",
    "support_counts": "VERIFIED",
}
REQUIRED_A2_SOURCES = (
    "src/token_reconstruction/trr_p09_fixed_control_adapter.py",
    "scripts/trr_p09/fixed_control_runner.py",
    "scripts/trr_p09/prepare_streamed_bank.py",
)
ValidationCallback = Callable[[int, Callable[[Iterable[Any]], Mapping[str, Any]]], Mapping[str, Any]]
DOMAIN_BALANCED_SELECTION_METRIC = "domain_balanced_token_accuracy"


REQUIRED_SETTINGS = (
    "hidden_size",
    "vocabulary_size",
    "sequence_tokens",
    "batch_records",
    "position_budget",
    "base_learning_rate",
    "directional_learning_rate",
    "weight_decay",
    "anchor_strength",
    "probe_steps",
    "compute_base_logits",
)


class QualificationError(RuntimeError):
    """Raised when the bounded qualifier cannot certify a safe probe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"bound file is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_descriptor(value: Mapping[str, Any], *, label: str, expected_status: str | None = None) -> dict[str, Any]:
    if expected_status is not None and value.get("status") != expected_status:
        raise QualificationError(f"{label} status must be {expected_status!r}")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise QualificationError(f"{label} path is missing")
    try:
        expected_bytes = int(value["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError(f"{label} byte binding is malformed") from exc
    expected_sha = value.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise QualificationError(f"{label} SHA-256 binding is malformed")
    path = Path(raw_path).expanduser().resolve()
    actual_bytes = int(path.stat().st_size) if path.is_file() and not path.is_symlink() else -1
    actual_sha = _sha256_file(path) if actual_bytes >= 0 else ""
    if actual_bytes != expected_bytes or actual_sha != expected_sha:
        raise QualificationError(f"{label} changed after its immutable binding")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha, "status": value.get("status")}


def validate_qualification_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate static artifacts/settings before a model or bank is loaded."""

    if manifest.get("schema") != "token-reconstruction.trr0010-qualifier-bindings.v1":
        raise QualificationError("qualifier binding schema differs")
    if manifest.get("task_id") != TASK_ID or manifest.get("finalized") is not True:
        raise QualificationError("qualifier bindings are not finalized for this task")
    if manifest.get("status") != "READY_FOR_QUALIFICATION":
        raise QualificationError("qualifier bindings are not READY_FOR_QUALIFICATION")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(REQUIRED_ARTIFACTS):
        raise QualificationError("qualifier artifact binding set is incomplete")
    verified_artifacts = {
        role: _verify_descriptor(value, label=role, expected_status=REQUIRED_ARTIFACTS[role])
        for role, value in artifacts.items()
        if isinstance(value, Mapping)
    }
    if set(verified_artifacts) != set(REQUIRED_ARTIFACTS):
        raise QualificationError("qualifier artifact descriptors are malformed")

    sources = manifest.get("a2_sources")
    if not isinstance(sources, Mapping) or set(sources) != set(REQUIRED_A2_SOURCES):
        raise QualificationError("qualifier A2 source binding set is incomplete")
    verified_sources: dict[str, dict[str, Any]] = {}
    for relative_path, value in sources.items():
        if not isinstance(value, Mapping):
            raise QualificationError(f"A2 source binding is malformed: {relative_path}")
        commit = value.get("commit")
        if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise QualificationError(f"A2 source commit is malformed: {relative_path}")
        verified = _verify_descriptor(value, label=f"A2 source {relative_path}")
        verified["commit"] = commit
        verified_sources[relative_path] = verified

    settings = manifest.get("settings")
    if not isinstance(settings, Mapping) or any(key not in settings for key in REQUIRED_SETTINGS):
        raise QualificationError("qualifier numerical settings are incomplete")
    try:
        positive_ints = ("hidden_size", "vocabulary_size", "sequence_tokens", "batch_records", "position_budget", "probe_steps")
        for key in positive_ints:
            if int(settings[key]) <= 0:
                raise QualificationError(f"qualifier setting {key} must be positive")
        positive_floats = ("base_learning_rate", "directional_learning_rate")
        for key in positive_floats:
            value = float(settings[key])
            if not torch.isfinite(torch.tensor(value)).item() or value <= 0.0:
                raise QualificationError(f"qualifier setting {key} must be finite and positive")
        weight_decay = float(settings["weight_decay"])
        anchor_strength = float(settings["anchor_strength"])
        if not torch.isfinite(torch.tensor(weight_decay)).item() or weight_decay < 0.0:
            raise QualificationError("qualifier weight decay is invalid")
        if not torch.isfinite(torch.tensor(anchor_strength)).item() or anchor_strength < 0.0:
            raise QualificationError("qualifier anchor strength is invalid")
    except (TypeError, ValueError) as exc:
        raise QualificationError("qualifier numerical settings are malformed") from exc
    if bool(settings["compute_base_logits"]):
        raise QualificationError("merged primary qualifier must set compute_base_logits=false")
    if float(settings["directional_learning_rate"]) != DEFAULT_DIRECTIONAL_LEARNING_RATE:
        raise QualificationError("directional learning rate differs from the frozen model default")
    schedule = manifest.get("schedule")
    if not isinstance(schedule, Mapping):
        raise QualificationError("qualification schedule metadata is missing")
    for key in ("seed", "steps", "semantic_sha256", "exposure"):
        if key not in schedule:
            raise QualificationError(f"qualification schedule metadata is incomplete: {key}")
    if int(schedule["steps"]) != int(settings["probe_steps"]):
        raise QualificationError("qualification schedule steps differ from probe_steps")
    semantic = schedule["semantic_sha256"]
    if not isinstance(semantic, str) or len(semantic) != 64 or any(c not in "0123456789abcdef" for c in semantic):
        raise QualificationError("qualification schedule semantic digest is malformed")
    validation_geometry = manifest.get("validation_geometry")
    if not isinstance(validation_geometry, Mapping):
        raise QualificationError("qualification validation geometry is missing")
    for key in ("sequence_tokens", "batch_records", "activation_dtype"):
        if key not in validation_geometry:
            raise QualificationError(f"qualification validation geometry is incomplete: {key}")
    if int(validation_geometry["sequence_tokens"]) <= 1 or int(validation_geometry["batch_records"]) <= 0:
        raise QualificationError("qualification validation geometry is invalid")
    if not isinstance(validation_geometry["activation_dtype"], str) or not validation_geometry["activation_dtype"]:
        raise QualificationError("qualification validation activation dtype is malformed")
    return {
        "artifacts": verified_artifacts,
        "a2_sources": verified_sources,
        "settings": {str(key): value for key, value in settings.items()},
        "schedule": {str(key): value for key, value in schedule.items()},
        "validation_geometry": {str(key): value for key, value in validation_geometry.items()},
        "binding_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def validate_exclusive_lease(lease: Mapping[str, Any]) -> dict[str, Any]:
    """Validate explicit compute ownership/caps; no defaults are allowed."""

    if lease.get("schema") != "token-reconstruction.trr0010-qualifier-lease.v1":
        raise QualificationError("qualifier lease schema differs")
    if lease.get("task_id") != TASK_ID or lease.get("status") != "GRANTED":
        raise QualificationError("qualifier lease is not GRANTED to TRR-0010")
    if lease.get("exclusive") is not True:
        raise QualificationError("qualifier requires an exclusive lease")
    device = lease.get("device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise QualificationError("qualifier lease must name an explicit CUDA device")
    required = (
        "max_seconds",
        "gpu_reserved_limit_bytes",
        "gpu_free_floor_bytes",
        "host_rss_limit_bytes",
        "host_available_floor_bytes",
        "disk_free_floor_bytes",
    )
    caps: dict[str, Any] = {}
    for key in required:
        try:
            value = int(lease[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise QualificationError(f"qualifier lease cap is malformed: {key}") from exc
        if value <= 0:
            raise QualificationError(f"qualifier lease cap must be positive: {key}")
        caps[key] = value
    caps["device"] = device
    caps["owner"] = str(lease.get("owner", ""))
    return caps


def _meminfo_bytes(name: str) -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, raw = line.partition(":")
        if key != name or not separator:
            continue
        fields = raw.strip().split()
        try:
            value = int(fields[0])
        except (IndexError, ValueError):
            return None
        return value * 1024 if len(fields) > 1 and fields[1].lower() == "kb" else value
    return None


def _current_rss_bytes() -> int | None:
    try:
        lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, raw = line.partition(":")
        if key == "VmRSS" and separator:
            fields = raw.strip().split()
            try:
                value = int(fields[0])
            except (IndexError, ValueError):
                return None
            return value * 1024 if len(fields) > 1 and fields[1].lower() == "kb" else value
    return None


def resource_snapshot(*, device: torch.device, output_root: Path) -> dict[str, Any]:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise QualificationError("CUDA became unavailable during qualifier")
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        gpu = {
            "available": True,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    else:
        gpu = {"available": False}
    disk = shutil.disk_usage(Path(output_root).expanduser().resolve().parent)
    return {
        "utc": _utc_now(),
        "host_available_bytes": _meminfo_bytes("MemAvailable"),
        "host_rss_bytes": _current_rss_bytes(),
        "disk_free_bytes": int(disk.free),
        "gpu": gpu,
    }


def enforce_resource_guard(snapshot: Mapping[str, Any], caps: Mapping[str, Any], *, started: float) -> None:
    if time.perf_counter() - started > int(caps["max_seconds"]):
        raise QualificationError("qualifier wall-time cap exceeded")
    host_available = snapshot.get("host_available_bytes")
    host_rss = snapshot.get("host_rss_bytes")
    if not isinstance(host_available, int) or host_available < int(caps["host_available_floor_bytes"]):
        raise QualificationError("host available-memory cap failed")
    if not isinstance(host_rss, int) or host_rss > int(caps["host_rss_limit_bytes"]):
        raise QualificationError("host RSS cap failed")
    if int(snapshot.get("disk_free_bytes", -1)) < int(caps["disk_free_floor_bytes"]):
        raise QualificationError("disk-free cap failed")
    gpu = snapshot.get("gpu")
    if not isinstance(gpu, Mapping) or gpu.get("available") is not True:
        raise QualificationError("GPU telemetry is unavailable")
    if int(gpu.get("free_bytes", -1)) < int(caps["gpu_free_floor_bytes"]):
        raise QualificationError("GPU free-memory cap failed")
    if int(gpu.get("max_reserved_bytes", -1)) > int(caps["gpu_reserved_limit_bytes"]):
        raise QualificationError("GPU reserved-memory cap failed")


def _module_device(module: nn.Module) -> torch.device:
    devices = {parameter.device for parameter in module.parameters()}
    if len(devices) != 1:
        raise QualificationError("decoder parameters do not occupy one device")
    return next(iter(devices))


@dataclass
class QualificationRuntime:
    decoder: ResidualMLPPositionwiseDecoder
    hook: DirectionalTokenReadout
    optimizer: torch.optim.Optimizer


def build_qualification_runtime(
    *,
    base_decoder: ResidualMLPPositionwiseDecoder,
    support_ids: torch.Tensor,
    support_counts: torch.Tensor,
    public_embedding: torch.Tensor,
    settings: Mapping[str, Any],
) -> QualificationRuntime:
    """Build the selected base plus zero Delta without choosing a contender."""

    if not isinstance(base_decoder, ResidualMLPPositionwiseDecoder):
        raise QualificationError("qualifier base must be the selected residual MLP decoder")
    device = _module_device(base_decoder)
    if public_embedding.device != device:
        raise QualificationError("public embedding must already be on decoder device")
    hook = build_directional_from_base(
        base_decoder,
        support_ids,
        support_counts,
        anchor_strength=float(settings["anchor_strength"]),
    )
    hook.to(device)
    hook.bind_embedding_statistics(public_embedding)
    if not bool(torch.equal(hook.delta_rows, torch.zeros_like(hook.delta_rows))):
        raise QualificationError("directional Delta did not initialize to exact zero")
    groups = hook.optimizer_param_groups(
        base_decoder,
        base_learning_rate=float(settings["base_learning_rate"]),
    )
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=float(settings["weight_decay"]),
        foreach=False,
    )
    validate_optimizer_configuration(
        optimizer,
        base_decoder,
        hook,
        expected_weight_decay=float(settings["weight_decay"]),
        expected_base_learning_rate=float(settings["base_learning_rate"]),
    )
    return QualificationRuntime(base_decoder, hook, optimizer)


def verify_zero_delta_equivalence(
    *,
    runtime: QualificationRuntime,
    runner: Any,
    batch: Any,
    schedule_step: Any,
    embedding: torch.Tensor,
    config: Any,
) -> dict[str, Any]:
    """Compare fixed and directional logits on one identical public batch."""

    if not callable(getattr(runner, "shared_decoder_rows", None)):
        raise QualificationError("A2 runner does not expose shared_decoder_rows")
    if not bool(torch.equal(runtime.hook.delta_rows, torch.zeros_like(runtime.hook.delta_rows))):
        raise QualificationError("zero-equivalence probe requires an untouched Delta")
    device = _module_device(runtime.decoder)
    activation = batch.activations.to(device=device, dtype=torch.float32)
    valid_mask = batch.attention_mask.to(device=device, dtype=torch.bool)
    token_ids = batch.token_ids.to(device=device, dtype=torch.long)
    record_slots = torch.tensor(schedule_step.draw_record_slots, device=device, dtype=torch.long)
    position_slots = torch.tensor(schedule_step.draw_position_slots, device=device, dtype=torch.long)
    if not bool(valid_mask[record_slots, position_slots].all().item()):
        raise QualificationError("zero-equivalence probe sampled an invalid position")
    target_ids = token_ids[record_slots, position_slots]
    with torch.inference_mode():
        projected = runtime.decoder.projected_hidden(activation, valid_mask)
        fixed_logits = runtime.decoder.logits_from_rows(
            projected, record_slots, position_slots, embedding
        )
        _base_logits, directional_logits, _losses = runner.shared_decoder_rows(
            runtime.decoder,
            runtime.hook,
            activation,
            valid_mask,
            record_slots,
            position_slots,
            embedding,
            target_ids,
            compute_base_logits=False,
        )
    if not torch.equal(fixed_logits, directional_logits):
        difference = (fixed_logits - directional_logits).abs()
        raise QualificationError(
            f"zero Delta changed primary logits: max_abs={float(difference.max().cpu())}"
        )
    return {
        "rows": int(target_ids.numel()),
        "logits_shape": list(fixed_logits.shape),
        "exact_logits": True,
        "exact_argmax": bool(torch.equal(fixed_logits.argmax(-1), directional_logits.argmax(-1))),
        "compute_base_logits": False,
        "config_hidden_size": int(config.hidden_size),
    }


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += int(value.numel()) * int(value.element_size())
    return total


def _write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise QualificationError(f"qualifier receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def qualify_discarded_updates(
    *,
    binding_receipt: Mapping[str, Any],
    lease_caps: Mapping[str, Any],
    runtime: QualificationRuntime,
    runner: Any,
    source: Any,
    schedule_steps: Iterable[Any],
    embedding: torch.Tensor,
    config: Any,
    validation_batches: Callable[[int], Iterable[Any]] | None,
    validation_sequence_tokens: int,
    validation_batch_records: int,
    validation_activation_dtype: torch.dtype | None,
    training_activation_dtype: torch.dtype | None,
    output_root: Path,
    checkpoint_export: Callable[[QualificationRuntime, Path], Mapping[str, Any]] | None = None,
    validation_callback: ValidationCallback | None = None,
    started: float | None = None,
    started_utc: str | None = None,
    initial_guard_checks: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run a fixed validation-aware warmup probe and retain only evidence.

    ``runner.run_training`` remains the sole update/validation implementation.
    Its selected point is explicitly discarded; the probe cannot create a
    scientific contender or alter the later frozen schedule.
    """

    device = _module_device(runtime.decoder)
    if str(device) != str(lease_caps["device"]):
        raise QualificationError(f"runtime device {device} differs from lease {lease_caps['device']}")
    if not callable(getattr(runner, "run_training", None)):
        raise QualificationError("A2 runner does not expose run_training")
    if validation_callback is None and not callable(validation_batches):
        raise QualificationError("qualifier requires the shared validation-batch factory without a domain callback")
    if validation_callback is not None and not callable(validation_callback):
        raise QualificationError("qualifier domain validation callback is not callable")
    output_root = Path(output_root).expanduser().resolve()
    output_path = output_root / "qualification.json"
    failure_path = output_root / "failure.json"
    started = time.perf_counter() if started is None else float(started)
    started_utc = _utc_now() if started_utc is None else str(started_utc)
    guard_checks: list[dict[str, Any]] = [dict(check) for check in initial_guard_checks]
    try:
        before = resource_snapshot(device=device, output_root=output_root)
        enforce_resource_guard(before, lease_caps, started=started)
        guard_checks.append({"stage": "before_model_probe", "snapshot": before})
        probe_steps = int(binding_receipt["settings"]["probe_steps"])
        steps = tuple(itertools.islice(schedule_steps, probe_steps))
        if len(steps) != probe_steps:
            raise QualificationError("schedule ended before the declared probe step count")
        if tuple(int(step.step) for step in steps) != tuple(range(len(steps))):
            raise QualificationError("probe steps are not contiguous from zero")
        if int(config.steps) != probe_steps:
            raise QualificationError("A2 probe config.steps must equal the bound probe_steps")
        first_batch = source.batch_for_global_rows(steps[0].batch_global_rows)
        validate_batch = getattr(runner, "validate_batch", None)
        if not callable(validate_batch):
            raise QualificationError("A2 runner does not expose validate_batch")
        validate_batch(
            first_batch,
            expected_global_rows=steps[0].batch_global_rows,
            expected_sequence_tokens=int(config.train_sequence_tokens),
            expected_hidden_size=int(config.hidden_size),
            expected_batch_records=int(config.record_batch_size),
            expected_activation_dtype=training_activation_dtype,
        )
        zero = verify_zero_delta_equivalence(
            runtime=runtime,
            runner=runner,
            batch=first_batch,
            schedule_step=steps[0],
            embedding=embedding,
            config=config,
        )
        guard_checks.append({"stage": "after_zero_equivalence", "snapshot": resource_snapshot(device=device, output_root=output_root)})

        def checkpoint_guard(point: Mapping[str, Any], decoder: nn.Module, hook: DirectionalTokenReadout) -> None:
            del decoder, hook
            snapshot = resource_snapshot(device=device, output_root=output_root)
            enforce_resource_guard(snapshot, lease_caps, started=started)
            guard_checks.append({"stage": f"after_validation_step_{int(point['step'])}", "snapshot": snapshot})
            return None

        remaining_seconds = float(lease_caps["max_seconds"]) - (time.perf_counter() - started)
        if remaining_seconds <= 0.0:
            raise QualificationError("qualifier wall-time cap exhausted before A2 run")
        run_receipt = runner.run_training(
            runtime.decoder,
            runtime.hook,
            source,
            steps,
            schedule_steps_count=probe_steps,
            schedule_seed=int(binding_receipt["schedule"]["seed"]),
            schedule_semantic_sha256=str(binding_receipt["schedule"]["semantic_sha256"]),
            schedule_exposure=dict(binding_receipt["schedule"]["exposure"]),
            optimizer=runtime.optimizer,
            embedding=embedding,
            config=config,
            validation_batches=validation_batches,
            validation_callback=validation_callback,
            validation_sequence_tokens=int(validation_sequence_tokens),
            validation_batch_records=int(validation_batch_records),
            validation_activation_dtype=validation_activation_dtype,
            training_activation_dtype=training_activation_dtype,
            checkpoint_steps=(0, probe_steps),
            scheduler=None,
            compute_base_logits=False,
            checkpoint_callback=checkpoint_guard,
            deadline_seconds=remaining_seconds,
        )
        optimizer_bytes = _optimizer_state_bytes(runtime.optimizer)
        if optimizer_bytes <= 0:
            raise QualificationError("warmup did not allocate Adam state")
        export_receipt: Mapping[str, Any]
        if checkpoint_export is None:
            export_receipt = {"status": "NOT_RUN", "reason": "caller did not bind export probe"}
        else:
            io_started = time.perf_counter()
            export_receipt = dict(checkpoint_export(runtime, output_root))
            export_receipt = {**export_receipt, "wall_seconds": time.perf_counter() - io_started}
            snapshot = resource_snapshot(device=device, output_root=output_root)
            enforce_resource_guard(snapshot, lease_caps, started=started)
            guard_checks.append({"stage": "after_checkpoint_export_probe", "snapshot": snapshot})
        finished_utc = _utc_now()
        final = resource_snapshot(device=device, output_root=output_root)
        enforce_resource_guard(final, lease_caps, started=started)
        result = {
            "schema": QUALIFIER_SCHEMA,
            "task_id": TASK_ID,
            "status": (
                "QUALIFICATION_PASS"
                if checkpoint_export is not None
                else "QUALIFICATION_PARTIAL_NO_CHECKPOINT_EXPORT"
            ),
            "qualification_complete": checkpoint_export is not None,
            "preparation_guarded": True,
            "discarded_updates": True,
            "contender_selection": False,
            "retained_fitted_arm": False,
            "binding": dict(binding_receipt),
            "lease_caps": dict(lease_caps),
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "wall_seconds": time.perf_counter() - started,
            "zero_delta_equivalence": zero,
            "run_timing": dict(run_receipt.get("timing", {})),
            "runner_deadline_seconds": remaining_seconds,
            "selection_metric": _validate_validation_contract(config, validation_callback),
            "learning_curve_discarded": list(run_receipt.get("learning_curve", [])),
            "selected_step_discarded": run_receipt.get("selected_step"),
            "optimizer_state_bytes": optimizer_bytes,
            "checkpoint_export": export_receipt,
            "resource_guard_checks": guard_checks,
            "final_snapshot": final,
        }
        return {**result, "receipt": _write_create_only(output_path, result)}
    except Exception as exc:
        failure = {
            "schema": FAILURE_SCHEMA,
            "task_id": TASK_ID,
            "status": "QUALIFICATION_FAILED_CLOSED",
            "discarded_updates": True,
            "contender_selection": False,
            "retained_fitted_arm": False,
            "binding": dict(binding_receipt),
            "lease_caps": dict(lease_caps),
            "started_utc": started_utc,
            "finished_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "resource_guard_checks": guard_checks,
        }
        if not failure_path.exists():
            failure["receipt"] = _write_create_only(failure_path, failure)
        raise


def _verify_provider_modules(inputs: Mapping[str, Any], binding_receipt: Mapping[str, Any]) -> None:
    expected_paths = {
        "adapter_module": REQUIRED_A2_SOURCES[0],
        "runner_module": REQUIRED_A2_SOURCES[1],
        "loader_module": REQUIRED_A2_SOURCES[2],
    }
    for key, relative_path in expected_paths.items():
        module = inputs.get(key)
        actual_raw = getattr(module, "__file__", None)
        expected_raw = binding_receipt["a2_sources"][relative_path]["path"]
        if not isinstance(actual_raw, str) or Path(actual_raw).expanduser().resolve() != Path(expected_raw).resolve():
            raise QualificationError(f"provider {key} is not the hash-bound imported module")


def _validate_validation_contract(config: Any, validation_callback: Any) -> str | None:
    """Require the current A2 domain-balanced callback when that metric is bound."""

    selection_metric = getattr(config, "selection_metric", None)
    if selection_metric is None and isinstance(config, Mapping):
        selection_metric = config.get("selection_metric")
    if selection_metric == DOMAIN_BALANCED_SELECTION_METRIC and not callable(validation_callback):
        raise QualificationError(
            "domain-balanced selection requires the provider's explicit validation_callback"
        )
    if selection_metric != DOMAIN_BALANCED_SELECTION_METRIC and validation_callback is not None:
        raise QualificationError("validation_callback is only valid for domain-balanced selection")
    return None if selection_metric is None else str(selection_metric)


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise QualificationError(f"{label} must contain an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--provider", required=True, help="A2-owned MODULE:CALLABLE input factory")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    failure_path = output_root / "failure.json"
    started = time.perf_counter()
    started_utc = _utc_now()
    preparation_checks: list[dict[str, Any]] = []
    binding_receipt: Mapping[str, Any] | None = None
    lease_caps: Mapping[str, Any] | None = None
    try:
        manifest = _load_json(args.bindings, label="qualifier bindings")
        lease = _load_json(args.lease, label="qualifier lease")
        binding_receipt = validate_qualification_bindings(manifest)
        lease_caps = validate_exclusive_lease(lease)
        device = torch.device(lease_caps["device"])

        # The lease and wall deadline cover preparation as well as updates.
        # Reset CUDA peaks only after recording the pre-provider baseline; all
        # provider/model/equivalence/export peaks are then retained together.
        before_provider = resource_snapshot(device=device, output_root=output_root)
        enforce_resource_guard(before_provider, lease_caps, started=started)
        preparation_checks.append({"stage": "before_provider_load", "snapshot": before_provider})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        provider_guard_calls = 0

        def preparation_guard(stage: str) -> None:
            nonlocal provider_guard_calls
            provider_guard_calls += 1
            snapshot = resource_snapshot(device=device, output_root=output_root)
            enforce_resource_guard(snapshot, lease_caps, started=started)
            preparation_checks.append({"stage": f"provider_{stage}", "snapshot": snapshot})

        provider = load_provider(args.provider)
        # Providers must call preparation_guard around their own E/model and
        # loader phases.  A three-argument provider is rejected deliberately:
        # it cannot expose a fail-closed preparation deadline.
        inputs = provider(binding_receipt, lease_caps, device, preparation_guard)
        if not isinstance(inputs, Mapping):
            raise QualificationError("A2 provider must return a mapping")
        if provider_guard_calls == 0:
            raise QualificationError("A2 provider did not call preparation_guard during its load phases")
        _verify_provider_modules(inputs, binding_receipt)
        preparation_guard("after_import_and_provider_load")
        validation_geometry = binding_receipt["validation_geometry"]
        if int(inputs["validation_sequence_tokens"]) != int(validation_geometry["sequence_tokens"]):
            raise QualificationError("provider validation sequence geometry differs from binding")
        if int(inputs["validation_batch_records"]) != int(validation_geometry["batch_records"]):
            raise QualificationError("provider validation batch geometry differs from binding")
        config = inputs["config"]
        validation_callback = inputs.get("validation_callback")
        _validate_validation_contract(config, validation_callback)
        runtime = build_qualification_runtime(
            base_decoder=inputs["base_decoder"],
            support_ids=inputs["support_ids"],
            support_counts=inputs["support_counts"],
            public_embedding=inputs["public_embedding"],
            settings=binding_receipt["settings"],
        )
        preparation_guard("after_runtime_build")
        result = qualify_discarded_updates(
            binding_receipt=binding_receipt,
            lease_caps=lease_caps,
            runtime=runtime,
            runner=inputs["runner"],
            source=inputs["source"],
            schedule_steps=inputs["schedule_steps"],
            embedding=inputs["public_embedding"],
            config=config,
            validation_batches=inputs.get("validation_batches"),
            validation_callback=validation_callback,
            validation_sequence_tokens=int(inputs["validation_sequence_tokens"]),
            validation_batch_records=int(inputs["validation_batch_records"]),
            validation_activation_dtype=inputs.get("validation_activation_dtype"),
            training_activation_dtype=inputs.get("training_activation_dtype"),
            output_root=output_root,
            checkpoint_export=inputs.get("checkpoint_export"),
            started=started,
            started_utc=started_utc,
            initial_guard_checks=preparation_checks,
        )
        print(json.dumps({"status": result["status"], "receipt": result["receipt"]}, sort_keys=True))
        return 0
    except Exception as exc:
        if not failure_path.exists():
            try:
                _write_create_only(
                    failure_path,
                    {
                        "schema": FAILURE_SCHEMA,
                        "task_id": TASK_ID,
                        "status": "QUALIFICATION_FAILED_CLOSED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "binding_path": str(Path(args.bindings).expanduser().resolve()),
                        "lease_path": str(Path(args.lease).expanduser().resolve()),
                        "started_utc": started_utc,
                        "finished_utc": _utc_now(),
                        "wall_seconds": time.perf_counter() - started,
                        "binding": None if binding_receipt is None else dict(binding_receipt),
                        "lease_caps": None if lease_caps is None else dict(lease_caps),
                        "resource_guard_checks": preparation_checks,
                        "discarded_updates": True,
                        "contender_selection": False,
                        "retained_fitted_arm": False,
                    },
                )
            except Exception:
                pass
        raise

def load_provider(spec: str) -> Callable[..., Mapping[str, Any]]:
    """Load the future A2-owned provider without copying its loader/trainer."""

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise QualificationError("provider must be MODULE:CALLABLE")
    module = importlib.import_module(module_name)
    provider = getattr(module, attribute, None)
    if not callable(provider):
        raise QualificationError(f"provider is not callable: {spec}")
    return provider


__all__ = [
    "FAILURE_SCHEMA",
    "QualificationError",
    "QualificationRuntime",
    "QUALIFIER_SCHEMA",
    "build_qualification_runtime",
    "enforce_resource_guard",
    "load_provider",
    "main",
    "qualify_discarded_updates",
    "resource_snapshot",
    "validate_exclusive_lease",
    "validate_qualification_bindings",
    "verify_zero_delta_equivalence",
]


if __name__ == "__main__":
    raise SystemExit(main())
