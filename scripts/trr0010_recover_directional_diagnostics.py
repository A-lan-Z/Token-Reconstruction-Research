"""Recover retained TRR-0010 fitting diagnostics without refitting.

The directional runner completed the public fit before its post-fit export
adapter failed.  This helper consumes that immutable runner receipt and the
seven serialized checkpoints, then (only when an explicit CUDA lease is
supplied) re-evaluates the existing public diagnostic callback.  It never
updates parameters, selects a new checkpoint, opens evaluation truth, or
creates a contender.  Plan mode is CPU-only metadata validation and is the
safe default.

The helper deliberately does not repair the production runner.  A later
export-only recovery can use the nested checkpoint descriptor recorded by
``checkpoint_state_bindings[*].checkpoint``; the selected step is read from
``checkpoint.metadata.selected_step``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any


TASK_ID = "TRR-0010"
RECOVERY_SCHEMA = "token-reconstruction.trr0010-retained-diagnostic-recovery.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0010-retained-diagnostic-recovery-failure.v1"
CHECKPOINT_STEPS = (0, 1000, 2000, 4000, 8000, 12000, 13000)
FULL_BANK_STEPS = (0, 13000)
EXPECTED_FROZEN_ROWS = 64
ARM_TO_BANK = {"current_directional": "B0", "expanded_directional": "B1"}


class RecoveryError(RuntimeError):
    """Raised when retained diagnostics cannot be recovered fail-closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_path(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RecoveryError(f"{label} path is missing")
    path = Path(raw).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"{label} is not a regular file: {path}")
    return path


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_path(str(path), label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must contain a JSON object")
    return value


def _int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise RecoveryError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryError(f"{label} must be an integer") from exc
    return result


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RecoveryError(f"{label} must be a lowercase SHA-256")
    return value


def _descriptor_summary(checkpoint: Mapping[str, Any], *, step: int, verify_file: bool) -> dict[str, Any]:
    path = _regular_path(checkpoint.get("path"), label=f"checkpoint step {step}")
    expected_bytes = _int(checkpoint.get("bytes"), label=f"checkpoint step {step}.bytes")
    if expected_bytes <= 0:
        raise RecoveryError(f"checkpoint step {step} has invalid bytes")
    expected_sha = _sha(checkpoint.get("sha256"), label=f"checkpoint step {step}.sha256")
    actual_bytes = int(path.stat().st_size)
    actual_sha = None
    if verify_file:
        actual_sha = _sha256_file(path)
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise RecoveryError(f"checkpoint step {step} changed after the runner receipt")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RecoveryError(f"checkpoint step {step} metadata is missing")
    if _int(metadata.get("selected_step"), label=f"checkpoint step {step}.metadata.selected_step") != step:
        raise RecoveryError(f"checkpoint step {step} metadata selected_step differs")
    if metadata.get("serialization_only") is not True:
        raise RecoveryError(f"checkpoint step {step} is not serialization-only")
    if metadata.get("optimizer_state_external") is not True:
        raise RecoveryError(f"checkpoint step {step} does not bind external optimizer state")
    state_digest = _sha(checkpoint.get("state_tensor_digest"), label=f"checkpoint step {step}.state_tensor_digest")
    return {
        "step": step,
        "path": str(path),
        "bytes": expected_bytes,
        "sha256": expected_sha,
        "actual_bytes": actual_bytes,
        "actual_sha256": actual_sha,
        "state_bytes": _int(checkpoint.get("state_bytes"), label=f"checkpoint step {step}.state_bytes"),
        "state_tensor_digest": state_digest,
        "metadata_selected_step": _int(metadata.get("selected_step"), label=f"checkpoint step {step}.metadata.selected_step"),
        "metadata": {
            "schema": metadata.get("schema"),
            "method_id": metadata.get("method_id"),
            "current_H_only": metadata.get("current_H_only"),
            "full_vocabulary_cross_entropy": metadata.get("full_vocabulary_cross_entropy"),
            "p09_contract_sha256": metadata.get("p09_contract_sha256"),
            "p09_bank_sha256": metadata.get("p09_bank_sha256"),
            "p09_schedule_sha256": metadata.get("p09_schedule_sha256"),
            "runner_point_state_sha256": metadata.get("runner_point_state_sha256"),
            "support_count": metadata.get("support_count"),
            "support_digest": metadata.get("support_digest"),
        },
    }


def build_recovery_plan(raw: Mapping[str, Any], *, raw_path: Path, arm_name: str, verify_files: bool = False) -> dict[str, Any]:
    """Validate a completed runner receipt and return a no-execution plan."""

    if arm_name not in ARM_TO_BANK:
        raise RecoveryError(f"unknown directional arm: {arm_name}")
    if raw.get("status") != "COMPLETED":
        raise RecoveryError("retained runner receipt is not COMPLETED")
    if raw.get("schema") != "token-reconstruction.trr-p09-fixed-control-run.v1":
        raise RecoveryError("retained runner schema differs")
    if tuple(_int(value, label="checkpoint grid item") for value in raw.get("checkpoints", ())) != CHECKPOINT_STEPS:
        raise RecoveryError("retained checkpoint grid differs from the frozen seven-point grid")
    selected_step = _int(raw.get("selected_step"), label="runner selected_step")
    if selected_step not in CHECKPOINT_STEPS:
        raise RecoveryError("runner selected_step is outside the frozen grid")
    selected_state_sha = _sha(raw.get("selected_state_sha256"), label="runner selected_state_sha256")

    bindings = raw.get("checkpoint_state_bindings")
    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence) or len(bindings) != len(CHECKPOINT_STEPS):
        raise RecoveryError("runner checkpoint_state_bindings does not cover exactly seven checkpoints")
    checkpoints: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for expected_step, binding in zip(CHECKPOINT_STEPS, bindings):
        if not isinstance(binding, Mapping) or not isinstance(binding.get("checkpoint"), Mapping):
            raise RecoveryError(f"checkpoint binding at step {expected_step} is malformed")
        descriptor = _descriptor_summary(binding["checkpoint"], step=expected_step, verify_file=verify_files)
        descriptor["optimizer_state_external"] = binding.get("optimizer_state_external") is True
        descriptor["resume_requires_separate_optimizer_artifact"] = binding.get("resume_requires_separate_optimizer_artifact") is True
        if not descriptor["optimizer_state_external"] or not descriptor["resume_requires_separate_optimizer_artifact"]:
            raise RecoveryError(f"checkpoint step {expected_step} lacks external optimizer-state binding")
        checkpoints.append(descriptor)
        if expected_step == selected_step:
            selected.append(descriptor)
    if len(selected) != 1:
        raise RecoveryError("runner selected checkpoint is missing or ambiguous")
    selected_descriptor = selected[0]
    runner_point_state = selected_descriptor["metadata"].get("runner_point_state_sha256")
    if runner_point_state != selected_state_sha:
        raise RecoveryError("selected checkpoint runner point state differs from retained selected state")
    # The selected checkpoint metadata is the authoritative nested descriptor;
    # the raw runner's selected state must still agree with its state binding.
    raw_curve = raw.get("learning_curve")
    if isinstance(raw_curve, (str, bytes)) or not isinstance(raw_curve, Sequence) or len(raw_curve) != len(CHECKPOINT_STEPS):
        raise RecoveryError("runner learning curve does not cover the frozen grid")
    curve: list[dict[str, Any]] = []
    for expected_step, row in zip(CHECKPOINT_STEPS, raw_curve):
        if not isinstance(row, Mapping) or _int(row.get("step"), label="learning_curve.step") != expected_step:
            raise RecoveryError("learning curve steps differ from the checkpoint grid")
        validation = row.get("validation")
        if not isinstance(validation, Mapping):
            raise RecoveryError(f"learning curve validation is missing at step {expected_step}")
        metric = validation.get("domain_balanced_token_accuracy")
        if not isinstance(metric, (int, float)) or isinstance(metric, bool):
            raise RecoveryError(f"selection metric is missing at step {expected_step}")
        curve.append({
            "step": expected_step,
            "selection_metric": float(metric),
            "correct_tokens": validation.get("correct_tokens"),
            "token_rows": validation.get("token_rows"),
            "label_join_sha256": validation.get("label_join_sha256"),
        })
    best = max(curve, key=lambda item: (item["selection_metric"], -item["step"]))
    if best["step"] != selected_step:
        raise RecoveryError("retained selected_step is not the earliest maximum validation metric")

    schedule = raw.get("schedule")
    if not isinstance(schedule, Mapping):
        raise RecoveryError("runner schedule is missing")
    schedule_summary = {
        "steps": _int(schedule.get("steps"), label="schedule.steps"),
        "seed": _int(schedule.get("seed"), label="schedule.seed"),
        "semantic_sha256": _sha(schedule.get("semantic_sha256"), label="schedule.semantic_sha256"),
        "exposure": dict(schedule.get("exposure", {})) if isinstance(schedule.get("exposure"), Mapping) else None,
    }
    if schedule_summary["steps"] != CHECKPOINT_STEPS[-1] or schedule_summary["exposure"] is None:
        raise RecoveryError("runner schedule does not bind the completed 13,000-step run")

    raw_path = Path(raw_path).expanduser().resolve()
    return {
        "schema": RECOVERY_SCHEMA,
        "task_id": TASK_ID,
        "status": "PLAN_ONLY",
        "arm_name": arm_name,
        "bank_role": ARM_TO_BANK[arm_name],
        "source_runner_receipt": {
            "path": str(raw_path),
            "bytes": int(raw_path.stat().st_size),
            "sha256": _sha256_file(raw_path),
            "status": raw.get("status"),
            "schema": raw.get("schema"),
        },
        "completed_fit": {
            "training_steps": CHECKPOINT_STEPS[-1],
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "schedule": schedule_summary,
            "selected_step": selected_step,
            "selected_state_sha256": selected_state_sha,
            "selected_checkpoint": selected_descriptor,
            "selection_metric": "domain_balanced_token_accuracy",
            "validation_curve": curve,
            "earliest_maximum_confirmed": True,
            "runner_point_state_sha256": runner_point_state,
        },
        "recovery_scopes": {
            "frozen64": {"steps": list(CHECKPOINT_STEPS), "row_count": EXPECTED_FROZEN_ROWS},
            "full_bank": {"steps": list(FULL_BANK_STEPS), "endpoint_labels": {"0": "start", "13000": "end"}},
        },
        "recovery_policy": {
            "recomputed_from_retained_checkpoints": True,
            "new_fit_updates": False,
            "selection_tuning": False,
            "selection_rule_unchanged": True,
            "original_fit_cost_charged": True,
            "truth_opened": False,
            "source_text_loaded": False,
            "diagnostic_detail_originally_retained": False,
            "diagnostic_limitation": (
                "The failed post-fit export occurred before raw diagnostic sidecars were serialized; "
                "the retained runner receipt contains the validation curve and checkpoint summaries, "
                "but not the original frozen64/full-bank fitting metric payloads."
            ),
            "execution_requires_explicit_exclusive_cuda_lease": True,
        },
        "checkpoints": checkpoints,
    }


def _load_callable(spec: str) -> Any:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise RecoveryError("provider must be MODULE:CALLABLE")
    module = importlib.import_module(module_name)
    value = getattr(module, attribute, None)
    if not callable(value):
        raise RecoveryError(f"provider is not callable: {spec}")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RecoveryError(f"recovery output is create-only: {path}")
    path.write_text(json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validate_scope_payload(raw: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    if set(raw).intersection({"selection_metric", "domain_balanced_token_accuracy"}):
        raise RecoveryError(f"diagnostic callback rewrote selection at step {step}")
    fitting = raw.get("fitting_diagnostics")
    if not isinstance(fitting, Mapping) or fitting.get("selection_metric_untouched") is not True:
        raise RecoveryError(f"diagnostic payload is not selection-isolated at step {step}")
    scopes = fitting.get("scopes")
    if isinstance(scopes, (str, bytes)) or not isinstance(scopes, Sequence):
        raise RecoveryError(f"diagnostic scopes are missing at step {step}")
    frozen = [scope for scope in scopes if isinstance(scope, Mapping) and scope.get("scope") == "frozen64"]
    full = [scope for scope in scopes if isinstance(scope, Mapping) and scope.get("scope") == "full_bank"]
    if len(frozen) != 1 or _int(frozen[0].get("row_count"), label=f"frozen64 row_count at step {step}") != EXPECTED_FROZEN_ROWS:
        raise RecoveryError(f"diagnostic frozen64 scope is invalid at step {step}")
    if step in FULL_BANK_STEPS:
        if len(full) != 1:
            raise RecoveryError(f"diagnostic full-bank scope is missing at endpoint {step}")
    elif full:
        raise RecoveryError(f"diagnostic full-bank scope appeared at non-endpoint {step}")
    return {"step": step, "frozen64": dict(frozen[0]), "full_bank": None if not full else dict(full[0])}


def execute_recovery(
    *,
    plan: Mapping[str, Any],
    binding_path: Path,
    diagnostic_binding_path: Path,
    lease_path: Path,
    provider_spec: str,
    output_root: Path,
) -> dict[str, Any]:
    """Recompute only the registered diagnostics under a fresh explicit lease."""

    # Imports are intentionally delayed: plan mode must remain CPU-only and
    # must not initialize CUDA or load public tensors.
    import torch
    from safetensors.torch import load_file

    from trr0010_directional_fit_cli import _load_diagnostic_binding
    from trr0010_p09_qualifier import enforce_resource_guard, resource_snapshot, validate_exclusive_lease

    from trr0010_directional_fit import CHECKPOINT_STEPS

    if plan.get("status") != "PLAN_ONLY" or plan.get("recovery_policy", {}).get("truth_opened") is not False:
        raise RecoveryError("recovery plan is not an unopened PLAN_ONLY receipt")
    lease = _load_json(lease_path, label="recovery lease")
    lease_caps = validate_exclusive_lease(lease)
    device = torch.device(str(lease_caps["device"]))
    if device.type != "cuda":
        raise RecoveryError("retained diagnostic recovery requires the explicit CUDA lease device")
    binding = _load_json(binding_path, label="directional binding")
    diagnostic_binding = _load_diagnostic_binding(Path(diagnostic_binding_path))
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RecoveryError(f"recovery output must be a new empty directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    guard_checks: list[dict[str, Any]] = []

    def guard(stage: str) -> None:
        snapshot = resource_snapshot(device=device, output_root=output_root)
        enforce_resource_guard(snapshot, lease_caps, started=started)
        guard_checks.append({"stage": stage, "snapshot": snapshot})

    try:
        guard("before_provider")
        provider = _load_callable(provider_spec)
        arm_name = str(plan["arm_name"])
        inputs = provider(
            binding,
            lease_caps,
            device,
            guard,
            arm_name=arm_name,
            bank_role=ARM_TO_BANK[arm_name],
            checkpoint_steps=CHECKPOINT_STEPS,
            output_root=output_root,
            diagnostic_binding=diagnostic_binding,
        )
        if not isinstance(inputs, Mapping):
            raise RecoveryError("provider returned a non-mapping")
        runtime = inputs.get("runtime")
        diagnostic_callback = inputs.get("diagnostic_callback")
        if runtime is None or not callable(diagnostic_callback):
            raise RecoveryError("provider did not return runtime and diagnostic_callback")
        decoder = runtime.decoder
        hook = runtime.hook
        decoder.eval()
        hook.eval()
        guard("after_provider")

        per_step: list[dict[str, Any]] = []
        for descriptor in plan["checkpoints"]:
            step = _int(descriptor.get("step"), label="recovery checkpoint step")
            if step not in CHECKPOINT_STEPS:
                raise RecoveryError(f"recovery checkpoint step {step} is outside the frozen grid")
            path = _regular_path(descriptor.get("path"), label=f"recovery checkpoint {step}")
            if int(path.stat().st_size) != int(descriptor["bytes"]):
                raise RecoveryError(f"recovery checkpoint {step} byte binding changed")
            actual_sha = _sha256_file(path)
            if actual_sha != descriptor["sha256"]:
                raise RecoveryError(f"recovery checkpoint {step} SHA-256 binding changed")
            guard(f"before_checkpoint_{step}")
            step_started = time.perf_counter()
            state = load_file(str(path), device=str(device))
            try:
                hook.load_state_dict(state, strict=True)
            except Exception as exc:
                raise RecoveryError(f"retained checkpoint {step} does not load into the provider runtime") from exc
            with torch.no_grad():
                raw = diagnostic_callback(
                    {"step": step, "state_sha256": descriptor["state_tensor_digest"]},
                    decoder,
                    hook,
                )
            if not isinstance(raw, Mapping):
                raise RecoveryError(f"diagnostic callback returned a non-mapping at step {step}")
            normalized = _validate_scope_payload(raw, step=step)
            artifact_path = output_root / "diagnostics" / f"checkpoint_step_{step:06d}.json"
            _write_create_only(
                artifact_path,
                {
                    "schema": "token-reconstruction.trr0010-recomputed-fitting-diagnostic.v1",
                    "task_id": TASK_ID,
                    "arm_name": arm_name,
                    "step": step,
                    "recomputed": True,
                    "selection_isolated": True,
                    "source_checkpoint": dict(descriptor),
                    "diagnostic": dict(raw),
                },
            )
            per_step.append({
                "step": step,
                "path": str(artifact_path),
                "bytes": int(artifact_path.stat().st_size),
                "sha256": _sha256_file(artifact_path),
                "scopes": normalized,
                "elapsed_seconds": time.perf_counter() - step_started,
            })
            del state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            guard(f"after_checkpoint_{step}")
        guard("after_recovery")
        result = {
            "schema": RECOVERY_SCHEMA,
            "task_id": TASK_ID,
            "status": "RECOVERY_COMPLETE",
            "arm_name": arm_name,
            "bank_role": ARM_TO_BANK[arm_name],
            "recomputed": True,
            "new_fit_updates": False,
            "selection_rule_unchanged": True,
            "selected_step": plan["completed_fit"]["selected_step"],
            "selected_step_source": "retained_runner_receipt",
            "original_fit_cost_charged": True,
            "truth_opened": False,
            "source_text_loaded": False,
            "diagnostic_originally_retained": False,
            "diagnostics": per_step,
            "resource_guard_checks": guard_checks,
            "timing": {"recovery_wall_seconds": time.perf_counter() - started},
            "command": list(sys.argv),
        }
        _write_create_only(output_root / "recovery_receipt.json", result)
        return result
    except Exception as exc:
        failure = {
            "schema": FAILURE_SCHEMA,
            "task_id": TASK_ID,
            "status": "RECOVERY_FAILED_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "command": list(sys.argv),
            "resource_guard_checks": guard_checks,
            "truth_opened": False,
            "new_fit_updates": False,
        }
        failure_path = output_root / "failure.json"
        if not failure_path.exists():
            _write_create_only(failure_path, failure)
        raise


def _write_plan(path: Path, plan: Mapping[str, Any]) -> None:
    _write_create_only(path, plan)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-runner-result", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=tuple(ARM_TO_BANK))
    parser.add_argument("--output", required=True, type=Path, help="Create-only plan or recovery receipt path")
    parser.add_argument("--execute", action="store_true", help="Run retained public diagnostics under the explicit lease")
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--diagnostic-binding", type=Path)
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--provider", default="trr0010_production_provider:build_inputs")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    raw = _load_json(args.raw_runner_result, label="raw runner result")
    plan = build_recovery_plan(raw, raw_path=args.raw_runner_result, arm_name=args.arm, verify_files=False)
    if not args.execute:
        _write_plan(args.output, plan)
        return 0
    if args.binding is None or args.diagnostic_binding is None or args.lease is None or args.output_root is None:
        raise RecoveryError("--execute requires --binding, --diagnostic-binding, --lease, and --output-root")
    result = execute_recovery(
        plan=plan,
        binding_path=args.binding,
        diagnostic_binding_path=args.diagnostic_binding,
        lease_path=args.lease,
        provider_spec=args.provider,
        output_root=args.output_root,
    )
    _write_create_only(args.output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARM_TO_BANK",
    "CHECKPOINT_STEPS",
    "FULL_BANK_STEPS",
    "RecoveryError",
    "build_recovery_plan",
    "execute_recovery",
    "main",
]
