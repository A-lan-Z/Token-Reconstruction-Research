"""Thin TRR-0010 production directional-fit entrypoint.

A2 owns public-bank loading, the immutable schedule, validation, checkpoint
selection, resource guards, and the update loop.  This task-local module only
reuses the concrete provider mapping, constructs the directional runtime with
``trr0010_p09_caller``, installs the frozen cosine scheduler, wraps the A2
checkpoint callback with the contract's fit diagnostics, and exports the
selected state.  It does not sample, select records, or open evaluation truth.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

from trr0010_p09_caller import (
    P09CallerError,
    export_selected_base_decoder_state,
    make_serialization_only_checkpoint_callback,
    prepare_directional_runtime,
    restore_selected_and_export,
    validate_optimizer_configuration,
)

TASK_ID = "TRR-0010"
FIT_SCHEMA = "token-reconstruction.trr0010-directional-fit.v1"
FIT_FAILURE_SCHEMA = "token-reconstruction.trr0010-directional-fit-failure.v1"
CHECKPOINT_STEPS = (0, 1000, 2000, 4000, 8000, 12000, 13000)
TRAINING_STEPS = 13000
REQUIRED_ARMS = ("current_directional", "expanded_directional")
ARM_TO_BANK = {"current_directional": "current", "expanded_directional": "expanded"}
EXPECTED_SELECTION_METRIC = "domain_balanced_token_accuracy"
EXPECTED_BATCH_RECORDS = 8
EXPECTED_POSITION_BUDGET = 512
EXPECTED_FIT_TOKENS = 192
EXPECTED_VALIDATION_TOKENS = 128
EXPECTED_HIDDEN_SIZE = 2048
EXPECTED_BASE_LEARNING_RATE = 2.0e-4
EXPECTED_DIRECTIONAL_LEARNING_RATE = 1.0e-4
EXPECTED_WEIGHT_DECAY = 0.0
EXPECTED_GRADIENT_CLIP_NORM = 1.0
EXPECTED_FIT_DIAGNOSTIC_RECORDS = 64


class DirectionalFitError(RuntimeError):
    """Raised when the concrete public provider handoff is incomplete."""


def _canonical_sha(value: Any, *, label: str) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise DirectionalFitError(f"{label} is not serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise DirectionalFitError(f"{label} must be a lowercase SHA-256")
    return value


def _value(config: Any, key: str, *, label: str) -> Any:
    if isinstance(config, Mapping):
        if key not in config:
            raise DirectionalFitError(f"{label} is missing {key!r}")
        return config[key]
    try:
        return getattr(config, key)
    except AttributeError as exc:
        raise DirectionalFitError(f"{label} is missing {key!r}") from exc


def _int(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise DirectionalFitError(f"{label} is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DirectionalFitError(f"{label} is not an integer") from exc
    if positive and result <= 0:
        raise DirectionalFitError(f"{label} must be positive")
    return result


def _float(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise DirectionalFitError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DirectionalFitError(f"{label} is not numeric") from exc
    if not torch.isfinite(torch.tensor(result)).item() or (nonnegative and result < 0):
        raise DirectionalFitError(f"{label} must be finite and non-negative")
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectionalFitError(f"{label} must be a mapping")
    return value


def _is_bf16(value: Any) -> bool:
    return value is torch.bfloat16 or str(value).lower() in {"torch.bfloat16", "bfloat16", "bf16"}


def _validate_schedule(inputs: Mapping[str, Any], *, arm_name: str) -> dict[str, Any]:
    count = _int(inputs.get("schedule_steps_count"), label=f"{arm_name}.schedule_steps_count", positive=True)
    if count != TRAINING_STEPS:
        raise DirectionalFitError(f"{arm_name} schedule must contain exactly {TRAINING_STEPS} updates")
    schedule = _mapping(inputs.get("schedule_binding"), label=f"{arm_name}.schedule_binding")
    control = _mapping(inputs.get("control_schedule_binding"), label=f"{arm_name}.control_schedule_binding")
    schedule_sha = _sha(schedule.get("sha256"), label=f"{arm_name}.schedule_binding.sha256")
    control_sha = _sha(control.get("sha256"), label=f"{arm_name}.control_schedule_binding.sha256")
    if schedule_sha != control_sha:
        raise DirectionalFitError(f"{arm_name} schedule bytes differ from fixed-control schedule")
    semantic = _sha(inputs.get("schedule_semantic_sha256"), label=f"{arm_name}.schedule_semantic_sha256")
    seed = _int(inputs.get("schedule_seed"), label=f"{arm_name}.schedule_seed")
    exposure = _mapping(inputs.get("schedule_exposure"), label=f"{arm_name}.schedule_exposure")
    steps = inputs.get("schedule_steps")
    if steps is None or isinstance(steps, (str, bytes)):
        raise DirectionalFitError(f"{arm_name}.schedule_steps is missing")
    if hasattr(steps, "__len__") and len(steps) != TRAINING_STEPS:
        raise DirectionalFitError(f"{arm_name}.schedule_steps length differs from {TRAINING_STEPS}")
    return {
        "file_sha256": schedule_sha,
        "control_file_sha256": control_sha,
        "semantic_sha256": semantic,
        "seed": seed,
        "exposure_sha256": _canonical_sha(exposure, label=f"{arm_name}.schedule_exposure"),
    }


def _validate_diagnostic_binding(inputs: Mapping[str, Any], *, arm_name: str) -> dict[str, Any]:
    """Check the small provider contract for callback-owned diagnostics."""

    binding = _mapping(inputs.get("diagnostic_binding"), label=f"{arm_name}.diagnostic_binding")
    if _int(binding.get("fit_record_count"), label=f"{arm_name}.diagnostic_binding.fit_record_count", positive=True) != EXPECTED_FIT_DIAGNOSTIC_RECORDS:
        raise DirectionalFitError(f"{arm_name} diagnostic binding must declare 64 fit records")
    endpoints = binding.get("full_bank_endpoint_steps")
    if not isinstance(endpoints, Sequence) or tuple(int(v) for v in endpoints) != (0, TRAINING_STEPS):
        raise DirectionalFitError(f"{arm_name} full-bank diagnostics must be bound at steps 0 and 13000")
    if binding.get("selection_isolated") is not True:
        raise DirectionalFitError(f"{arm_name} diagnostic output must be isolated from checkpoint selection")
    callback = inputs.get("diagnostic_callback")
    if not callable(callback):
        raise DirectionalFitError(f"{arm_name}.diagnostic_callback is missing")
    return {
        "binding_sha256": _canonical_sha(binding, label=f"{arm_name}.diagnostic_binding"),
        "fit_record_count": EXPECTED_FIT_DIAGNOSTIC_RECORDS,
        "full_bank_endpoint_steps": [0, TRAINING_STEPS],
        "selection_isolated": True,
    }


def validate_fit_inputs(inputs: Mapping[str, Any], *, arm_name: str) -> dict[str, Any]:
    """Validate the concrete provider mapping before runtime construction."""

    if arm_name not in REQUIRED_ARMS:
        raise DirectionalFitError(f"unknown arm {arm_name!r}")
    if not isinstance(inputs, Mapping):
        raise DirectionalFitError(f"{arm_name} provider result must be a mapping")
    if inputs.get("arm_name", arm_name) != arm_name:
        raise DirectionalFitError(f"{arm_name} provider returned a different arm")
    if inputs.get("bank_role") != ARM_TO_BANK[arm_name]:
        raise DirectionalFitError(f"{arm_name} provider bank role is not {ARM_TO_BANK[arm_name]!r}")
    runner = inputs.get("runner")
    if runner is None or not callable(getattr(runner, "run_training", None)):
        raise DirectionalFitError(f"{arm_name} A2 runner is missing")
    config = inputs.get("config")
    if config is None:
        raise DirectionalFitError(f"{arm_name} runner config is missing")
    expected_ints = {
        "steps": TRAINING_STEPS,
        "record_batch_size": EXPECTED_BATCH_RECORDS,
        "position_budget": EXPECTED_POSITION_BUDGET,
        "train_sequence_tokens": EXPECTED_FIT_TOKENS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
    }
    for key, expected in expected_ints.items():
        if _int(_value(config, key, label=f"{arm_name}.config"), label=f"{arm_name}.config.{key}", positive=True) != expected:
            raise DirectionalFitError(f"{arm_name}.config.{key} differs from the frozen fit geometry")
    if _value(config, "selection_metric", label=f"{arm_name}.config") != EXPECTED_SELECTION_METRIC:
        raise DirectionalFitError(f"{arm_name} selection metric is not domain-balanced")
    if _float(_value(config, "learning_rate", label=f"{arm_name}.config"), label=f"{arm_name}.config.learning_rate") != EXPECTED_BASE_LEARNING_RATE:
        raise DirectionalFitError(f"{arm_name} base learning rate differs from 2e-4")
    if _float(_value(config, "weight_decay", label=f"{arm_name}.config"), label=f"{arm_name}.config.weight_decay", nonnegative=True) != EXPECTED_WEIGHT_DECAY:
        raise DirectionalFitError(f"{arm_name} weight decay differs from zero")
    if _float(_value(config, "gradient_clip_norm", label=f"{arm_name}.config"), label=f"{arm_name}.config.gradient_clip_norm") != EXPECTED_GRADIENT_CLIP_NORM:
        raise DirectionalFitError(f"{arm_name} gradient clip differs from one")
    if not _is_bf16(inputs.get("training_activation_dtype")) or not _is_bf16(inputs.get("validation_activation_dtype")):
        raise DirectionalFitError(f"{arm_name} activation storage must be BF16")
    if _int(inputs.get("validation_sequence_tokens"), label=f"{arm_name}.validation_sequence_tokens", positive=True) != EXPECTED_VALIDATION_TOKENS:
        raise DirectionalFitError(f"{arm_name} validation width must be 128")
    if _int(inputs.get("validation_batch_records"), label=f"{arm_name}.validation_batch_records", positive=True) <= 0:
        raise DirectionalFitError(f"{arm_name} validation batch size is invalid")
    if not callable(inputs.get("validation_callback")) or inputs.get("source") is None:
        raise DirectionalFitError(f"{arm_name} source/validation callback is incomplete")
    for key in ("base_state", "fit_manifest"):
        value = _mapping(inputs.get(key), label=f"{arm_name}.{key}")
        _sha(value.get("sha256"), label=f"{arm_name}.{key}.sha256")
    runtime = inputs.get("runtime")
    if runtime is None:
        for key in ("protocol", "base_decoder", "support_ids", "support_counts", "public_embedding", "bindings", "source_paths"):
            if inputs.get(key) is None:
                raise DirectionalFitError(f"{arm_name}.{key} is missing for caller runtime construction")
    else:
        for key in ("decoder", "hook", "optimizer"):
            if not hasattr(runtime, key):
                raise DirectionalFitError(f"{arm_name}.runtime.{key} is missing")
    return {
        "schedule": _validate_schedule(inputs, arm_name=arm_name),
        "diagnostics": _validate_diagnostic_binding(inputs, arm_name=arm_name),
    }


def _diagnostic_summary(raw: Mapping[str, Any], *, step: int, started: float, arm_name: str) -> dict[str, Any]:
    """Normalize the approved A2 nested fitting-diagnostic callback output.

    A2's ``make_fitting_metric_callback`` returns scopes under
    ``fitting_diagnostics``.  The legacy synthetic callback used in this
    module returns ``fit_diagnostics``; retaining that narrow branch keeps the
    CPU integration tests independent of public assets while production uses
    the A2 helper unchanged.
    """

    nested = raw.get("fitting_diagnostics")
    if nested is not None:
        diagnostics = _mapping(nested, label=f"{arm_name} fitting diagnostics at step {step}")
        if diagnostics.get("selection_metric_untouched") is not True:
            raise DirectionalFitError(f"{arm_name} fitting diagnostics can not alter selection")
        scopes = diagnostics.get("scopes")
        if isinstance(scopes, (str, bytes)) or not isinstance(scopes, Sequence):
            raise DirectionalFitError(f"{arm_name} fitting diagnostic scopes are missing at step {step}")
        frozen = [scope for scope in scopes if isinstance(scope, Mapping) and scope.get("scope") == "frozen64"]
        full = [scope for scope in scopes if isinstance(scope, Mapping) and scope.get("scope") == "full_bank"]
        if len(frozen) != 1 or _int(frozen[0].get("row_count"), label=f"{arm_name}.frozen64.row_count", positive=True) != EXPECTED_FIT_DIAGNOSTIC_RECORDS:
            raise DirectionalFitError(f"{arm_name} fitting diagnostic must contain exactly 64 frozen rows at step {step}")
        expected_endpoint = "start" if step == 0 else "end" if step == TRAINING_STEPS else None
        if expected_endpoint is None and full:
            raise DirectionalFitError(f"{arm_name} full-bank diagnostics are only allowed at steps 0 and 13000")
        if expected_endpoint is not None and len(full) != 1:
            raise DirectionalFitError(f"{arm_name} full-bank diagnostic is missing at step {step}")
        full_summary = None
        if full:
            full_rows = _int(full[0].get("row_count"), label=f"{arm_name}.full_bank.row_count", positive=True)
            full_summary = {
                "endpoint": expected_endpoint,
                "row_count": full_rows,
                "sha256": _canonical_sha(full[0], label=f"{arm_name}.full_bank.{expected_endpoint}"),
            }
        return {
            "step": step,
            "fit_records": EXPECTED_FIT_DIAGNOSTIC_RECORDS,
            "fit_records_sha256": _canonical_sha(frozen[0], label=f"{arm_name}.fitting_diagnostics.frozen64.{step}"),
            "full_bank": full_summary,
            "selection_metric_untouched": True,
            "elapsed_seconds": time.perf_counter() - started,
        }

    fit = _mapping(raw.get("fit_diagnostics"), label=f"{arm_name} fit diagnostics at step {step}")
    records = fit.get("records", fit.get("items"))
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or len(records) != EXPECTED_FIT_DIAGNOSTIC_RECORDS:
        raise DirectionalFitError(f"{arm_name} diagnostic callback returned fewer/more than 64 fit records at step {step}")
    endpoint = None
    if step in (0, TRAINING_STEPS):
        full = _mapping(raw.get("full_bank"), label=f"{arm_name} full-bank diagnostics at step {step}")
        expected = "start" if step == 0 else "end"
        if full.get("endpoint") != expected:
            raise DirectionalFitError(f"{arm_name} full-bank endpoint at step {step} is not {expected!r}")
        endpoint = {"endpoint": expected, "sha256": _canonical_sha(full, label=f"{arm_name}.full_bank.{expected}")}
    elif raw.get("full_bank") is not None:
        raise DirectionalFitError(f"{arm_name} full-bank diagnostics are only allowed at steps 0 and 13000")
    return {
        "step": step,
        "fit_records": EXPECTED_FIT_DIAGNOSTIC_RECORDS,
        "fit_records_sha256": _canonical_sha(records, label=f"{arm_name}.fit_diagnostics.records.{step}"),
        "full_bank": endpoint,
        "elapsed_seconds": time.perf_counter() - started,
    }

def _selected_checkpoint(result: Mapping[str, Any], *, arm_name: str) -> Mapping[str, Any]:
    if result.get("status") != "COMPLETED":
        raise DirectionalFitError(f"{arm_name} runner did not complete")
    selected = _int(result.get("selected_step"), label=f"{arm_name}.selected_step")
    if selected not in CHECKPOINT_STEPS:
        raise DirectionalFitError(f"{arm_name} selected step is outside the frozen grid")
    if tuple(int(v) for v in result.get("checkpoints", ())) != CHECKPOINT_STEPS:
        raise DirectionalFitError(f"{arm_name} checkpoint grid differs from the frozen grid")
    _sha(result.get("selected_state_sha256"), label=f"{arm_name}.selected_state_sha256")
    values = result.get("checkpoint_state_bindings")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise DirectionalFitError(f"{arm_name} checkpoint bindings are missing")
    matches = []
    for value in values:
        if not isinstance(value, Mapping) or not isinstance(value.get("checkpoint"), Mapping):
            continue
        checkpoint = value["checkpoint"]
        try:
            checkpoint_step = int(checkpoint.get("selected_step", checkpoint.get("step", -1)))
        except (TypeError, ValueError):
            continue
        if checkpoint_step == selected:
            matches.append(checkpoint)
    if len(matches) != 1 or not isinstance(matches[0].get("path"), str):
        raise DirectionalFitError(f"{arm_name} selected checkpoint binding is missing or ambiguous")
    _sha(matches[0].get("sha256"), label=f"{arm_name}.selected_checkpoint.sha256")
    if _int(matches[0].get("bytes"), label=f"{arm_name}.selected_checkpoint.bytes", positive=True) <= 0:
        raise DirectionalFitError(f"{arm_name} selected checkpoint byte binding is invalid")
    return matches[0]


def fit_one_arm(inputs: Mapping[str, Any], *, arm_name: str, output_root: Path, deadline_seconds: float | None = None) -> dict[str, Any]:
    validation = validate_fit_inputs(inputs, arm_name=arm_name)
    runtime = inputs.get("runtime")
    if runtime is None:
        runtime = prepare_directional_runtime(
            protocol=inputs["protocol"],
            base_decoder=inputs["base_decoder"],
            support_ids=inputs["support_ids"],
            support_counts=inputs["support_counts"],
            public_embedding=inputs["public_embedding"],
            bindings=inputs["bindings"],
            source_paths=inputs["source_paths"],
            base_learning_rate=EXPECTED_BASE_LEARNING_RATE,
            weight_decay=EXPECTED_WEIGHT_DECAY,
            artifact_root=inputs.get("artifact_root"),
        )
    public_embedding = inputs.get("public_embedding", getattr(runtime, "public_embedding", None))
    if not isinstance(public_embedding, torch.Tensor):
        raise DirectionalFitError(f"{arm_name} public embedding is missing")
    try:
        validate_optimizer_configuration(
            runtime.optimizer,
            runtime.decoder,
            runtime.hook,
            expected_weight_decay=EXPECTED_WEIGHT_DECAY,
            expected_current_learning_rates=(EXPECTED_BASE_LEARNING_RATE, EXPECTED_DIRECTIONAL_LEARNING_RATE),
        )
    except (P09CallerError, AttributeError, TypeError, ValueError) as exc:
        raise DirectionalFitError(f"{arm_name} optimizer groups/rates are not frozen") from exc
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(runtime.optimizer, T_max=TRAINING_STEPS)
    if tuple(float(v) for v in scheduler.get_last_lr()) != (EXPECTED_BASE_LEARNING_RATE, EXPECTED_DIRECTIONAL_LEARNING_RATE):
        raise DirectionalFitError(f"{arm_name} cosine scheduler does not preserve the 2e-4/1e-4 rates")
    arm_root = Path(output_root) / arm_name
    serial_callback = make_serialization_only_checkpoint_callback(
        output_root=arm_root / "checkpoints",
        runtime=runtime,
        base_state=inputs["base_state"],
        fit_manifest=inputs["fit_manifest"],
    )
    diagnostic_callback = inputs["diagnostic_callback"]
    diagnostic_events: list[dict[str, Any]] = []

    def checkpoint_callback(point: Mapping[str, Any], decoder: Any, hook: Any) -> Mapping[str, Any]:
        step = _int(point.get("step"), label=f"{arm_name}.checkpoint.step")
        started = time.perf_counter()
        raw = diagnostic_callback(point, decoder, hook)
        if not isinstance(raw, Mapping):
            raise DirectionalFitError(f"{arm_name} diagnostic callback must return a mapping")
        diagnostic_events.append(_diagnostic_summary(raw, step=step, started=started, arm_name=arm_name))
        # The callback returns only the serialization binding.  Diagnostic
        # summaries never enter the runner point and cannot influence selection.
        return serial_callback(point, decoder, hook)

    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "schedule_steps_count": TRAINING_STEPS,
        "schedule_seed": _int(inputs["schedule_seed"], label=f"{arm_name}.schedule_seed"),
        "schedule_semantic_sha256": inputs["schedule_semantic_sha256"],
        "schedule_exposure": dict(inputs["schedule_exposure"]),
        "optimizer": runtime.optimizer,
        "embedding": public_embedding,
        "config": inputs["config"],
        "validation_batches": inputs.get("validation_batches"),
        "validation_callback": inputs["validation_callback"],
        "validation_sequence_tokens": _int(inputs["validation_sequence_tokens"], label=f"{arm_name}.validation_sequence_tokens"),
        "validation_batch_records": _int(inputs["validation_batch_records"], label=f"{arm_name}.validation_batch_records"),
        "validation_activation_dtype": inputs["validation_activation_dtype"],
        "training_activation_dtype": inputs["training_activation_dtype"],
        "checkpoint_steps": CHECKPOINT_STEPS,
        "scheduler": scheduler,
        "compute_base_logits": False,
        "checkpoint_callback": checkpoint_callback,
    }
    if deadline_seconds is not None:
        kwargs["deadline_seconds"] = float(deadline_seconds)
    if callable(inputs.get("resource_guard_callback")):
        kwargs["resource_guard_callback"] = inputs["resource_guard_callback"]
    result = inputs["runner"].run_training(runtime.decoder, runtime.hook, inputs["source"], inputs["schedule_steps"], **kwargs)
    selected = _selected_checkpoint(result, arm_name=arm_name)
    selected_step = _int(result["selected_step"], label=f"{arm_name}.selected_step")
    restore = restore_selected_and_export(
        checkpoint_path=Path(str(selected["path"])),
        expected_checkpoint=selected,
        export_path=arm_root / "effective_readout_w.safetensors",
        runtime=runtime,
        public_embedding=public_embedding,
        selected_step=selected_step,
    )
    base = export_selected_base_decoder_state(
        path=arm_root / "base_decoder_state.safetensors",
        runtime=runtime,
        selected_receipt=restore,
        selected_step=selected_step,
        metadata={"arm_name": arm_name, "bank_role": ARM_TO_BANK[arm_name]},
    )
    if tuple(event["step"] for event in diagnostic_events) != CHECKPOINT_STEPS:
        raise DirectionalFitError(f"{arm_name} diagnostics did not run at every checkpoint")
    if {event["full_bank"]["endpoint"] for event in diagnostic_events if event["full_bank"]} != {"start", "end"}:
        raise DirectionalFitError(f"{arm_name} full-bank diagnostics did not cover start/end")
    binding_summary = getattr(runtime, "bindings", None)
    runtime_binding = None
    if binding_summary is not None and callable(getattr(binding_summary, "metadata", None)):
        runtime_binding = {"metadata_sha256": _canonical_sha(binding_summary.metadata(), label=f"{arm_name}.bindings")}
        digest = getattr(binding_summary, "source_binding_digest", None)
        if callable(digest):
            runtime_binding["source_binding_digest"] = _sha(digest(), label=f"{arm_name}.source_binding_digest")
    provider_receipt = inputs.get("provider_receipt")
    provider_summary = None if provider_receipt is None else {"sha256": _canonical_sha(provider_receipt, label=f"{arm_name}.provider_receipt")}
    return {
        "arm_name": arm_name,
        "bank_role": ARM_TO_BANK[arm_name],
        "status": result["status"],
        "selected_step": selected_step,
        "selected_state_sha256": result["selected_state_sha256"],
        "checkpoints": list(CHECKPOINT_STEPS),
        "schedule": validation["schedule"],
        "diagnostic_binding": validation["diagnostics"],
        "diagnostic_events": diagnostic_events,
        "timing": {**dict(result.get("timing", {})), "wrapper_seconds": time.perf_counter() - started},
        "runtime_binding": runtime_binding,
        "provider_receipt": provider_summary,
        "selected_checkpoint": {"path": str(selected["path"]), "bytes": int(selected.get("bytes", 0)), "sha256": selected["sha256"]},
        "effective_readout": {"path": restore["export"]["path"], "bytes": restore["export"]["bytes"], "sha256": restore["export"]["sha256"]},
        "base_decoder_state": {"path": base["path"], "bytes": base["bytes"], "sha256": base["sha256"]},
        "optimizer": {"groups": ["decoder", "directional_delta"], "base_lr": EXPECTED_BASE_LEARNING_RATE, "directional_lr": EXPECTED_DIRECTIONAL_LEARNING_RATE, "weight_decay": EXPECTED_WEIGHT_DECAY, "foreach": False, "scheduler": "CosineAnnealingLR", "scheduler_t_max": TRAINING_STEPS},
        "truth_opened": False,
    }


def bind_concrete_provider(
    provider_build_inputs: Callable[..., Mapping[str, Any]],
    *,
    binding_receipts: Mapping[str, Mapping[str, Any]],
    lease_caps: Mapping[str, Any],
    device: torch.device,
    preparation_guard: Callable[[str], None],
    diagnostic_binding: Mapping[str, Any] | None = None,
) -> Callable[..., Mapping[str, Any]]:
    """Bind the existing provider API for the two production bank roles.

    The current provider API is ``build_inputs(binding_receipt, lease_caps,
    device, preparation_guard)``.  The corrected production provider may add
    the named ``arm_name``, ``bank_role``, ``checkpoint_steps``, and
    ``output_root`` fields; this adapter forwards those only when explicitly
    declared.  A provider that cannot distinguish B0 from the expanded bank
    is rejected by :func:`validate_fit_inputs` rather than reused silently.
    """

    signature = inspect.signature(provider_build_inputs)
    parameters = signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())

    def build_inputs(*, arm_name: str, bank_role: str, checkpoint_steps: Sequence[int], output_root: Path, **extra: Any) -> Mapping[str, Any]:
        if arm_name not in binding_receipts:
            raise DirectionalFitError(f"no binding receipt is supplied for {arm_name}")
        optional = {
            "arm_name": arm_name,
            "bank_role": bank_role,
            "checkpoint_steps": tuple(checkpoint_steps),
            "output_root": Path(output_root),
            **extra,
        }
        if diagnostic_binding is not None:
            optional["diagnostic_binding"] = diagnostic_binding
        if not accepts_kwargs:
            optional = {key: value for key, value in optional.items() if key in parameters}
        return provider_build_inputs(
            binding_receipts[arm_name],
            lease_caps,
            device,
            preparation_guard,
            **optional,
        )

    return build_inputs


def run_directional_arms(build_inputs: Callable[..., Mapping[str, Any]], *, output_root: Path, factory_config: Mapping[str, Any] | None = None, deadline_seconds: float | None = None, command: Sequence[str] | None = None) -> dict[str, Any]:
    """Call the concrete capacity provider and fit both arms sequentially."""

    output_root = Path(output_root).expanduser().resolve()
    receipt_path = output_root / "run_receipt.json"
    if receipt_path.exists():
        raise DirectionalFitError(f"production fit is create-only: {receipt_path}")
    completed: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        signature = inspect.signature(build_inputs)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        for arm_name in REQUIRED_ARMS:
            kwargs: dict[str, Any] = {"arm_name": arm_name, "bank_role": ARM_TO_BANK[arm_name], "checkpoint_steps": CHECKPOINT_STEPS, "output_root": output_root / arm_name}
            if factory_config is not None:
                kwargs["factory_config"] = factory_config
            if not accepts_kwargs:
                kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
            if "arm_name" not in kwargs and "bank_role" not in kwargs:
                raise DirectionalFitError("provider must expose arm_name/bank_role for current and expanded banks")
            inputs = build_inputs(**kwargs)
            completed[arm_name] = fit_one_arm(inputs, arm_name=arm_name, output_root=output_root, deadline_seconds=deadline_seconds)
            del inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        failure = output_root / "failure.json"
        if not failure.exists():
            failure.write_text(json.dumps({"schema": FIT_FAILURE_SCHEMA, "task_id": TASK_ID, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "completed_arms": sorted(completed), "command": list(command or sys.argv), "truth_opened": False}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        raise
    receipt = {"schema": FIT_SCHEMA, "task_id": TASK_ID, "status": "FIT_COMPLETE", "checkpoint_steps": list(CHECKPOINT_STEPS), "training_steps": TRAINING_STEPS, "arms": completed, "command": list(command or sys.argv), "elapsed_seconds": time.perf_counter() - started, "truth_opened": False}
    output_root.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt


__all__ = [
    "ARM_TO_BANK",
    "CHECKPOINT_STEPS",
    "DirectionalFitError",
    "EXPECTED_FIT_DIAGNOSTIC_RECORDS",
    "REQUIRED_ARMS",
    "TRAINING_STEPS",
    "bind_concrete_provider",
    "fit_one_arm",
    "run_directional_arms",
    "validate_fit_inputs",
]
