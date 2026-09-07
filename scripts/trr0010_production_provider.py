"""Concrete TRR-0010 directional-fit provider.

This module is the production handoff between the task-owned directional
wrapper and the published P09 infrastructure.  It does not implement a
trainer or sampler.  It verifies one explicitly labelled B0 or B1 binding,
loads the existing streamed source/schedule/caller helpers, constructs the
full 13,000-update ``RunnerConfig`` and directional runtime, and installs the
A2 fitting-diagnostic callback.  ``configuration_dry_run`` performs the same
metadata checks for both arms without opening model/embedding/activation
payloads or allocating a model.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors import safe_open

from trr0010_directional_fit import (
    ARM_TO_BANK,
    CHECKPOINT_STEPS,
    EXPECTED_BATCH_RECORDS,
    EXPECTED_DIRECTIONAL_LEARNING_RATE,
    EXPECTED_FIT_TOKENS,
    EXPECTED_GRADIENT_CLIP_NORM,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_POSITION_BUDGET,
    EXPECTED_SELECTION_METRIC,
    EXPECTED_VALIDATION_TOKENS,
    EXPECTED_BASE_LEARNING_RATE,
    EXPECTED_WEIGHT_DECAY,
    TRAINING_STEPS,
)
from trr0010_p09_caller import (
    A2SourceBinding,
    FrozenArtifact,
    ProductionBindings,
    ResourceQualification,
    prepare_directional_runtime,
)
from trr0010_p09_provider import (
    ProviderError,
    _artifact,
    _contract_settings,
    _load_a2_modules,
    _load_embedding,
    _load_support,
    _metadata_int,
    _read_json,
    _verify_descriptor,
    deserialize_schedule,
    _ValidationView,
    _validation_views,
)


TASK_ID = "TRR-0010"
PRODUCTION_SCHEMA = "token-reconstruction.trr0010-directional-production-binding.v1"
SIGNED_STAGE3_CONTRACT_SCHEMA = "token-reconstruction.trr0010-shared-stage3-contract.v1"
SIGNED_STAGE3_CONTRACT_SHA256 = "23e4bde4475082cc004e7ef787c22fed6301275ce2ff23a0396d975b439faf9d"
SIGNED_STAGE3_COUNTERSIGNATURE_SHA256 = "a99d31991880be095a21cf7bef119adbceb14c51e82c35f2df108c4164467ab9"
SIGNED_STAGE3_COUNTERSIGNATURE_STATUS = "AGENT2_COUNTERSIGNED_ROOT_REVIEW_PASS_AGENT1_GO_PENDING"
MEASURED_QUALIFICATION_RECEIPT_SHA256 = "58594a477733c2f1ce21d356ec9acead92de6001b2c31bd41f635213700bca66"
MEASURED_RESOURCE_GUARD_SHA256 = "c14d22414ae4feac0f7c021d3278bb5fbfe5baa90c6d5eeb9e47b3f172042893"
MEASURED_GPU_PEAK_RESERVED_BYTES = 7600078848
MEASURED_HOST_PEAK_RSS_BYTES = 4276609024
MEASURED_MIN_HOST_AVAILABLE_BYTES = 18282762240
MEASURED_QUALIFICATION_WALL_SECONDS = 26.84726572499494
EXPECTED_START_STATE_SHA256 = "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14"
EXPECTED_START_SELECTED_STEP = 400
REQUIRED_A2_SOURCE_PATHS = (
    "src/token_reconstruction/trr_p09_fixed_control_adapter.py",
    "scripts/trr_p09/fixed_control_runner.py",
    "scripts/trr_p09/prepare_streamed_bank.py",
)
HELPER_SOURCE_PATHS = (
    "scripts/trr_p09/fixed_control_caller.py",
    "scripts/trr_p09/fixed_control_cli.py",
    "scripts/trr_p09/public_validation_loader.py",
)
REQUIRED_BANKS = {"current_directional": "B0", "expanded_directional": "B1"}


class ProductionProviderError(ProviderError):
    """Raised when a final arm binding is incomplete or changed."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionProviderError(f"{label} must be a mapping")
    return value


def _int(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ProductionProviderError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ProductionProviderError(f"{label} must be an integer") from exc
    if positive and result <= 0:
        raise ProductionProviderError(f"{label} must be positive")
    return result


def _float(value: Any, *, label: str, expected: float | None = None, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ProductionProviderError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProductionProviderError(f"{label} must be numeric") from exc
    if not torch.isfinite(torch.tensor(result)).item() or (nonnegative and result < 0.0):
        raise ProductionProviderError(f"{label} must be finite and non-negative")
    if expected is not None and result != expected:
        raise ProductionProviderError(f"{label} differs from frozen value {expected}")
    return result


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ProductionProviderError(f"{label} must be a lowercase SHA-256")
    return value


def _descriptor(receipt: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    artifacts = _mapping(receipt.get("artifacts"), label="artifacts")
    value = artifacts.get(role)
    if not isinstance(value, Mapping):
        raise ProductionProviderError(f"artifact descriptor is missing: {role}")
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ProductionProviderError(f"{role} path is missing")
    _int(value.get("bytes"), label=f"{role}.bytes", positive=True)
    _sha(value.get("sha256"), label=f"{role}.sha256")
    return value


def _helper_descriptor(receipt: Mapping[str, Any], relative: str) -> Mapping[str, Any]:
    sources = receipt.get("a2_sources")
    helpers = receipt.get("a2_helpers")
    candidates: list[Any] = []
    if isinstance(sources, Mapping):
        candidates.append(sources.get(relative))
    if isinstance(helpers, Mapping):
        candidates.append(helpers.get(relative))
    for value in candidates:
        if isinstance(value, Mapping):
            return value
    raise ProductionProviderError(f"A2 helper source binding is missing: {relative}")


def _declared_bank(receipt: Mapping[str, Any], *, arm_name: str, bank_role: str) -> str:
    expected = REQUIRED_BANKS.get(arm_name)
    if expected is None or bank_role != expected:
        raise ProductionProviderError(f"{arm_name} must use explicit bank role {expected}")
    declared = receipt.get("bank_role")
    if declared is None:
        bank = receipt.get("bank")
        if isinstance(bank, Mapping):
            declared = bank.get("role", bank.get("bank"))
        elif isinstance(bank, str):
            declared = bank
    if declared != expected:
        raise ProductionProviderError(f"binding does not declare {expected} for {arm_name}")
    declared_arm = receipt.get("arm_name")
    if declared_arm is not None and declared_arm != arm_name:
        raise ProductionProviderError(f"binding arm_name differs for {arm_name}")
    return expected


def _settings_and_training(receipt: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    settings = _mapping(receipt.get("settings"), label="settings")
    training = receipt.get("training")
    if not isinstance(training, Mapping):
        contract = receipt.get("contract")
        if isinstance(contract, Mapping):
            training = contract.get("training")
    if not isinstance(training, Mapping):
        contract_descriptor = _descriptor(receipt, "contract")
        contract = _read_json(Path(str(contract_descriptor["path"])), label="contract")
        training = _contract_settings(contract, receipt)
    return settings, _mapping(training, label="training")


def _validate_fixed_settings(receipt: Mapping[str, Any], *, arm_name: str, bank_role: str) -> dict[str, Any]:
    _declared_bank(receipt, arm_name=arm_name, bank_role=bank_role)
    settings, training = _settings_and_training(receipt)
    combined = dict(settings)
    combined.update({key: value for key, value in training.items() if key not in combined})
    expected_ints = {
        "steps": TRAINING_STEPS,
        "probe_steps": TRAINING_STEPS,
        "batch_records": EXPECTED_BATCH_RECORDS,
        "record_batch_size": EXPECTED_BATCH_RECORDS,
        "position_budget": EXPECTED_POSITION_BUDGET,
        "sequence_tokens": EXPECTED_FIT_TOKENS,
        "train_sequence_tokens": EXPECTED_FIT_TOKENS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "vocabulary_size": 128256,
        "validation_sequence_tokens": EXPECTED_VALIDATION_TOKENS,
    }
    for key, expected in expected_ints.items():
        if key in combined and _int(combined[key], label=f"{arm_name}.{key}") != expected:
            raise ProductionProviderError(f"{arm_name}.{key} differs from the frozen production geometry")
    if "steps" not in combined and "probe_steps" not in combined:
        raise ProductionProviderError(f"{arm_name} binding does not declare 13000 updates")
    _float(combined.get("base_learning_rate", combined.get("learning_rate")), label=f"{arm_name}.base_learning_rate", expected=EXPECTED_BASE_LEARNING_RATE)
    _float(combined.get("directional_learning_rate"), label=f"{arm_name}.directional_learning_rate", expected=EXPECTED_DIRECTIONAL_LEARNING_RATE)
    _float(combined.get("weight_decay"), label=f"{arm_name}.weight_decay", expected=EXPECTED_WEIGHT_DECAY, nonnegative=True)
    _float(combined.get("gradient_clip_norm"), label=f"{arm_name}.gradient_clip_norm", expected=EXPECTED_GRADIENT_CLIP_NORM, nonnegative=True)
    groups = combined.get("optimizer_groups")
    if list(groups or ()) != ["decoder", "directional_delta"]:
        raise ProductionProviderError(f"{arm_name} optimizer groups differ from decoder/directional_delta")
    if combined.get("optimizer", "AdamW") != "AdamW" or combined.get("optimizer_foreach", False) is not False:
        raise ProductionProviderError(f"{arm_name} optimizer policy is not AdamW foreach=False")
    metric = training.get("selection_metric", combined.get("selection_metric", EXPECTED_SELECTION_METRIC))
    if metric != EXPECTED_SELECTION_METRIC:
        raise ProductionProviderError(f"{arm_name} selection metric is not domain-balanced")
    validation_every = _int(training.get("validation_every", combined.get("validation_every", 1000)), label=f"{arm_name}.validation_every", positive=True)
    return {
        "steps": TRAINING_STEPS,
        "record_batch_size": EXPECTED_BATCH_RECORDS,
        "position_budget": EXPECTED_POSITION_BUDGET,
        "sequence_tokens": EXPECTED_FIT_TOKENS,
        "validation_sequence_tokens": EXPECTED_VALIDATION_TOKENS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "vocabulary_size": 128256,
        "base_learning_rate": EXPECTED_BASE_LEARNING_RATE,
        "directional_learning_rate": EXPECTED_DIRECTIONAL_LEARNING_RATE,
        "weight_decay": EXPECTED_WEIGHT_DECAY,
        "gradient_clip_norm": EXPECTED_GRADIENT_CLIP_NORM,
        "validation_every": validation_every,
        "selection_metric": EXPECTED_SELECTION_METRIC,
    }


def _validate_signed_stage3_contract(
    receipt: Mapping[str, Any],
    *,
    expected_contract_sha256: str = SIGNED_STAGE3_CONTRACT_SHA256,
    expected_countersignature_sha256: str = SIGNED_STAGE3_COUNTERSIGNATURE_SHA256,
) -> Mapping[str, Any]:
    """Validate the immutable stage-3 contract and its exact countersignature.

    The production path never treats a filename suffix or a self-declared
    status as evidence of signing.  It hashes both approved JSON artifacts,
    checks the contract schema/task/status, and verifies that the external
    countersignature names this exact contract hash.  Tests may inject their
    own expected hashes into this pure metadata validator; ``build_inputs``
    always uses the frozen defaults above.
    """
    contract_descriptor = _descriptor(receipt, "contract")
    contract_path = Path(str(contract_descriptor["path"])).expanduser()
    try:
        contract_bytes = contract_path.read_bytes()
    except OSError as exc:
        raise ProductionProviderError("signed stage-3 contract cannot be read") from exc
    if len(contract_bytes) != _int(contract_descriptor.get("bytes"), label="contract.bytes"):
        raise ProductionProviderError("signed stage-3 contract byte count differs from its binding")
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    if contract_sha != _sha(contract_descriptor.get("sha256"), label="contract.sha256"):
        raise ProductionProviderError("signed stage-3 contract bytes differ from its binding")
    if contract_sha != expected_contract_sha256:
        raise ProductionProviderError("signed stage-3 contract SHA differs from approved exact hash")
    try:
        contract = json.loads(contract_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionProviderError("signed stage-3 contract is not valid JSON") from exc
    if not isinstance(contract, Mapping):
        raise ProductionProviderError("signed stage-3 contract must be a JSON object")
    if contract.get("schema") != SIGNED_STAGE3_CONTRACT_SCHEMA or contract.get("task_id") != TASK_ID:
        raise ProductionProviderError("signed stage-3 contract schema/task identity differs")
    if contract.get("status") != "FROZEN_STAGE3_AGREED_PENDING_EXACT_HASH_COUNTERSIGNATURE":
        raise ProductionProviderError("signed stage-3 contract status is not the approved frozen status")
    counter_descriptor = receipt.get("contract_countersignature")
    if not isinstance(counter_descriptor, Mapping):
        raise ProductionProviderError("stage-3 contract countersignature binding is missing")
    counter_path_value = counter_descriptor.get("path")
    if not isinstance(counter_path_value, str) or not counter_path_value:
        raise ProductionProviderError("stage-3 contract countersignature path is missing")
    counter_path = Path(counter_path_value).expanduser()
    try:
        counter_bytes = counter_path.read_bytes()
    except OSError as exc:
        raise ProductionProviderError("stage-3 contract countersignature cannot be read") from exc
    if len(counter_bytes) != _int(counter_descriptor.get("bytes"), label="contract_countersignature.bytes"):
        raise ProductionProviderError("stage-3 contract countersignature byte count differs from its binding")
    counter_sha = hashlib.sha256(counter_bytes).hexdigest()
    if counter_sha != _sha(counter_descriptor.get("sha256"), label="contract_countersignature.sha256"):
        raise ProductionProviderError("stage-3 contract countersignature bytes differ from its binding")
    if counter_sha != expected_countersignature_sha256:
        raise ProductionProviderError("stage-3 contract countersignature SHA differs from approved exact hash")
    try:
        countersignature = json.loads(counter_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionProviderError("stage-3 contract countersignature is not valid JSON") from exc
    if not isinstance(countersignature, Mapping):
        raise ProductionProviderError("stage-3 contract countersignature must be a JSON object")
    if countersignature.get("status") != SIGNED_STAGE3_COUNTERSIGNATURE_STATUS:
        raise ProductionProviderError("stage-3 contract countersignature status differs")
    copy = _mapping(countersignature.get("contract_copy"), label="contract_copy")
    if copy.get("sha256") != contract_sha or copy.get("source_sha256") != contract_sha:
        raise ProductionProviderError("stage-3 countersignature does not name this exact contract hash")
    signature = _mapping(countersignature.get("counter_signature"), label="counter_signature")
    if signature.get("exact_contract_hash_agreed") is not True:
        raise ProductionProviderError("stage-3 countersignature lacks exact-contract-hash agreement")
    return contract


def _bound_file_descriptor(
    descriptor: Mapping[str, Any],
    *,
    label: str,
    expected_sha256: str,
) -> None:
    """Verify a create-only evidence file without opening model/input payloads."""
    path_value = descriptor.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ProductionProviderError(f"{label}.path is missing")
    path = Path(path_value).expanduser()
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProductionProviderError(f"{label} cannot be read") from exc
    if len(payload) != _int(descriptor.get("bytes"), label=f"{label}.bytes"):
        raise ProductionProviderError(f"{label} byte count differs from its binding")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _sha(descriptor.get("sha256"), label=f"{label}.sha256"):
        raise ProductionProviderError(f"{label} bytes differ from its binding")
    if digest != expected_sha256:
        raise ProductionProviderError(f"{label} SHA differs from approved measured receipt")


def _validate_resource_qualification(
    receipt: Mapping[str, Any],
    *,
    signed_contract: Mapping[str, Any],
    expected_qualification_sha256: str = MEASURED_QUALIFICATION_RECEIPT_SHA256,
    expected_resource_guard_sha256: str = MEASURED_RESOURCE_GUARD_SHA256,
) -> dict[str, Any]:
    """Validate measured qualification separately from signed production caps."""
    value = _mapping(receipt.get("resource_qualification"), label="resource_qualification")
    if value.get("status") != "PASS":
        raise ProductionProviderError("resource qualification is not PASS")
    qualification_receipt = _mapping(value.get("qualification_receipt"), label="qualification_receipt")
    guard_receipt = _mapping(value.get("resource_guard_receipt"), label="resource_guard_receipt")
    _bound_file_descriptor(
        qualification_receipt,
        label="qualification_receipt",
        expected_sha256=expected_qualification_sha256,
    )
    _bound_file_descriptor(
        guard_receipt,
        label="resource_guard_receipt",
        expected_sha256=expected_resource_guard_sha256,
    )
    measured = {
        "gpu_peak_reserved_bytes": _int(value.get("gpu_peak_reserved_bytes"), label="resource_qualification.gpu_peak_reserved_bytes", positive=True),
        "host_peak_rss_bytes": _int(value.get("host_peak_rss_bytes"), label="resource_qualification.host_peak_rss_bytes", positive=True),
        "host_available_bytes": _int(value.get("host_available_bytes"), label="resource_qualification.host_available_bytes", positive=True),
    }
    if expected_qualification_sha256 == MEASURED_QUALIFICATION_RECEIPT_SHA256:
        if measured["gpu_peak_reserved_bytes"] != MEASURED_GPU_PEAK_RESERVED_BYTES:
            raise ProductionProviderError("measured GPU qualification peak differs from receipt")
        if measured["host_peak_rss_bytes"] <= 0 or measured["host_available_bytes"] != MEASURED_MIN_HOST_AVAILABLE_BYTES:
            raise ProductionProviderError("measured host qualification values differ from receipt")
        if _float(value.get("wall_seconds"), label="resource_qualification.wall_seconds", nonnegative=True) != MEASURED_QUALIFICATION_WALL_SECONDS:
            raise ProductionProviderError("measured qualification wall time differs from receipt")
    caps = _mapping(value.get("production_caps"), label="resource_qualification.production_caps")
    contract_caps = _mapping(signed_contract.get("resource_caps"), label="resource_caps")
    per_arm = _mapping(contract_caps.get("per_arm"), label="resource_caps.per_arm")
    expected_caps = {
        "gpu_reserved_limit_bytes": per_arm.get("cuda_reserved_bytes_directional"),
        "host_rss_limit_bytes": per_arm.get("host_rss_bytes"),
        "host_available_floor_bytes": per_arm.get("host_available_floor_bytes"),
        "wall_limit_seconds": per_arm.get("max_seconds_including_preparation_diagnostics_checkpoint_export"),
        "gpu_free_floor_bytes": per_arm.get("gpu_free_floor_bytes"),
        "disk_free_floor_bytes": per_arm.get("disk_free_floor_bytes"),
        "output_bytes_limit": per_arm.get("output_bytes_limit"),
    }
    for field, expected in expected_caps.items():
        if field not in caps:
            raise ProductionProviderError(f"resource_qualification.production_caps is missing {field}")
        if field.endswith("_seconds"):
            actual_value = _float(caps.get(field), label=f"resource_qualification.production_caps.{field}", nonnegative=True)
            if actual_value != float(expected):
                raise ProductionProviderError(f"resource cap {field} differs from signed contract")
        else:
            if _int(caps.get(field), label=f"resource_qualification.production_caps.{field}") != _int(expected, label=f"signed resource cap {field}"):
                raise ProductionProviderError(f"resource cap {field} differs from signed contract")
    flat_cap_fields = {
        "gpu_reserved_limit_bytes": "gpu_reserved_limit_bytes",
        "host_rss_limit_bytes": "host_rss_limit_bytes",
        "host_available_floor_bytes": "host_available_floor_bytes",
        "wall_limit_seconds": "wall_limit_seconds",
    }
    for flat_field, cap_field in flat_cap_fields.items():
        if flat_field not in value:
            raise ProductionProviderError(f"resource_qualification is missing {flat_field}")
        if flat_field.endswith("_seconds"):
            flat_value = _float(value.get(flat_field), label=f"resource_qualification.{flat_field}", nonnegative=True)
            cap_value = _float(caps.get(cap_field), label=f"resource_qualification.production_caps.{cap_field}", nonnegative=True)
        else:
            flat_value = _int(value.get(flat_field), label=f"resource_qualification.{flat_field}")
            cap_value = _int(caps.get(cap_field), label=f"resource_qualification.production_caps.{cap_field}")
        if flat_value != cap_value:
            raise ProductionProviderError(f"resource qualification {flat_field} differs from production cap")
    # The measured qualifier had its own 900-second lease; production's 7,200
    # second cap is intentionally sourced from the signed contract above.
    return {
        "status": "PASS",
        "qualification_receipt": dict(qualification_receipt),
        "resource_guard_receipt": dict(guard_receipt),
        "measured": measured,
        "production_caps": dict(caps),
    }


def _validate_schedule_metadata(
    receipt: Mapping[str, Any],
    *,
    arm_name: str,
    signed_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    schedule = _mapping(receipt.get("schedule"), label=f"{arm_name}.schedule")
    if _int(schedule.get("steps"), label=f"{arm_name}.schedule.steps") != TRAINING_STEPS:
        raise ProductionProviderError(f"{arm_name} schedule must declare 13000 steps")
    if "probe_steps" in schedule and _int(schedule["probe_steps"], label=f"{arm_name}.schedule.probe_steps") != TRAINING_STEPS:
        raise ProductionProviderError(f"{arm_name} schedule probe_steps is qualification-only")
    seed = _int(schedule.get("seed"), label=f"{arm_name}.schedule.seed")
    semantic = _sha(schedule.get("semantic_sha256"), label=f"{arm_name}.schedule.semantic_sha256")
    exposure = _mapping(schedule.get("exposure"), label=f"{arm_name}.schedule.exposure")
    if _int(exposure.get("draws_per_step"), label=f"{arm_name}.schedule.exposure.draws_per_step", positive=True) != 512:
        raise ProductionProviderError(f"{arm_name} schedule exposure must contain 512 draws per step")
    schedule_artifact = _descriptor(receipt, "schedule")
    schedule_sha = _sha(schedule_artifact.get("sha256"), label=f"{arm_name}.schedule artifact SHA")
    bank_role = str(receipt.get("bank_role", ""))
    contract = signed_contract if signed_contract is not None else _validate_signed_stage3_contract(receipt)
    fixed = _mapping(contract.get("fixed_control_configuration"), label="fixed_control_configuration")
    contract_schedule_value = fixed.get(f"{bank_role}_schedule")
    if not isinstance(contract_schedule_value, Mapping):
        raise ProductionProviderError(
            f"{arm_name} signed contract lacks fixed_control_configuration.{bank_role}_schedule"
        )
    contract_schedule: Mapping[str, Any] = contract_schedule_value
    expected_path = contract_schedule.get("path")
    schedule_path = Path(str(schedule_artifact["path"])).expanduser().resolve()
    if not isinstance(expected_path, str) or Path(expected_path).expanduser().resolve() != schedule_path:
        raise ProductionProviderError(f"{arm_name} schedule path differs from signed {bank_role} schedule")
    for field, actual in (
        ("bytes", schedule_artifact.get("bytes")),
        ("sha256", schedule_sha),
        ("semantic_sha256", semantic),
        ("seed", seed),
        ("steps", int(schedule.get("steps"))),
    ):
        expected = contract_schedule.get(field)
        if field in {"bytes", "seed", "steps"}:
            try:
                matches = int(actual) == int(expected)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise ProductionProviderError(
                f"{arm_name} {field} differs from signed {bank_role} schedule"
            )
    control = receipt.get("control_schedule")
    if control is None:
        artifacts = _mapping(receipt.get("artifacts"), label="artifacts")
        control = artifacts.get("control_schedule")
    if not isinstance(control, Mapping):
        raise ProductionProviderError(f"{arm_name} fixed-control schedule binding is missing")
    control_sha = _sha(control.get("sha256"), label=f"{arm_name}.control_schedule.sha256")
    if control_sha != schedule_sha:
        raise ProductionProviderError(f"{arm_name} schedule bytes differ from fixed-control schedule")
    return {
        "steps": TRAINING_STEPS,
        "seed": seed,
        "semantic_sha256": semantic,
        "exposure": dict(exposure),
        "schedule_binding": dict(schedule_artifact),
        "control_schedule_binding": dict(control),
        "contract_schedule_binding": dict(contract_schedule) if contract_schedule is not None else None,
    }


def _validate_diagnostic(diagnostic_binding: Mapping[str, Any], *, bank_role: str) -> dict[str, Any]:
    binding = _mapping(diagnostic_binding, label="diagnostic_binding")
    if _int(binding.get("fit_record_count"), label="diagnostic_binding.fit_record_count", positive=True) != 64:
        raise ProductionProviderError("diagnostic binding must contain exactly 64 rows")
    if list(binding.get("full_bank_endpoint_steps", ())) != [0, TRAINING_STEPS] or binding.get("selection_isolated") is not True:
        raise ProductionProviderError("diagnostic binding endpoint or selection isolation differs")
    path = binding.get("path")
    if not isinstance(path, str) or not Path(path).is_file():
        raise ProductionProviderError("diagnostic binding path is unavailable")
    bank = binding.get("banks")
    if not isinstance(bank, Mapping) or not isinstance(bank.get(bank_role), Mapping):
        raise ProductionProviderError(f"diagnostic binding lacks {bank_role}")
    return {
        "schema": binding.get("schema"),
        "path": str(Path(path).expanduser().resolve()),
        "bytes": _int(binding.get("bytes"), label="diagnostic_binding.bytes", positive=True),
        "sha256": _sha(binding.get("sha256"), label="diagnostic_binding.sha256"),
        "fit_record_count": 64,
        "full_bank_endpoint_steps": [0, TRAINING_STEPS],
        "selection_isolated": True,
        "bank_role": bank_role,
        "bank_index_sha256": _sha(_mapping(bank[bank_role], label=f"diagnostic_binding.{bank_role}").get("global_indices_sha256"), label=f"diagnostic_binding.{bank_role}.global_indices_sha256"),
    }


def _validate_artifact_declarations(receipt: Mapping[str, Any], *, arm_name: str) -> dict[str, Mapping[str, Any]]:
    required = ("contract", "bank_manifest", "schedule", "base_state", "public_embedding", "support_ids", "support_counts")
    result = {role: _descriptor(receipt, role) for role in required}
    fit = receipt.get("fit_manifest")
    if fit is None:
        fit = result["bank_manifest"]
    if not isinstance(fit, Mapping):
        raise ProductionProviderError(f"{arm_name}.fit_manifest is missing")
    _sha(fit.get("sha256"), label=f"{arm_name}.fit_manifest.sha256")
    result["fit_manifest"] = fit
    start = receipt.get("start_identity_guard")
    if isinstance(start, Mapping):
        if start.get("sha256") != EXPECTED_START_STATE_SHA256 or _int(start.get("selected_step"), label=f"{arm_name}.start_identity_guard.selected_step") != EXPECTED_START_SELECTED_STEP:
            raise ProductionProviderError(f"{arm_name} starting checkpoint differs from shared step-400 state")
    base = result["base_state"]
    if base.get("sha256") != EXPECTED_START_STATE_SHA256:
        raise ProductionProviderError(f"{arm_name} base state differs from shared continued-fixed step-400 state")
    if _int(base.get("selected_step"), label=f"{arm_name}.base_state.selected_step") != EXPECTED_START_SELECTED_STEP:
        raise ProductionProviderError(f"{arm_name} base state selected step differs from 400")
    return result


def _declared_support(
    receipt: Mapping[str, Any],
    *,
    bank_role: str,
) -> dict[str, Any] | None:
    """Return the bank-specific support binding from the frozen contract.

    The qualification-only contract predates the per-bank binding and carries
    one expanded-bank count. It remains usable for the metadata/CPU assembly
    smoke only. A stage-3 production contract must carry a separate B0/B1
    support count and digest so B0 cannot accidentally consume B1's 45,631
    rows.
    """

    declared: Any = receipt.get("support_by_bank")
    if not isinstance(declared, Mapping):
        direct = receipt.get("directional_readout")
        if isinstance(direct, Mapping):
            declared = direct.get("support_by_bank")
    contract_descriptor = _descriptor(receipt, "contract")
    contract_path = Path(str(contract_descriptor["path"]))
    # Synthetic/configuration-only tests may bind an opaque contract token;
    # only JSON contract artifacts can carry the stage-3 per-bank map.
    # Real JSON artifacts remain fail-closed if malformed.
    if contract_path.suffix.lower() != ".json":
        return None
    contract = _read_json(contract_path, label="production contract")
    if not isinstance(declared, Mapping):
        direct = contract.get("directional_readout")
        if isinstance(direct, Mapping):
            declared = direct.get("support_by_bank")
    if not isinstance(declared, Mapping):
        if contract.get("schema") == "token-reconstruction.trr0010-shared-stage3-contract.v1":
            raise ProductionProviderError(
                f"stage-3 contract must bind directional_readout.support_by_bank for {bank_role}"
            )
        return None
    value = declared.get(bank_role)
    if not isinstance(value, Mapping):
        raise ProductionProviderError(f"support_by_bank is missing {bank_role}")
    count = _int(value.get("support_count"), label=f"support_by_bank.{bank_role}.support_count", positive=True)
    digest = _sha(value.get("support_digest"), label=f"support_by_bank.{bank_role}.support_digest")
    return {
        "bank_role": bank_role,
        "support_count": count,
        "support_digest": digest,
        "post_bos_positions": _int(
            value.get("post_bos_positions"),
            label=f"support_by_bank.{bank_role}.post_bos_positions",
            positive=True,
        ) if value.get("post_bos_positions") is not None else None,
    }


def _source_bindings(receipt: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    sources = _mapping(receipt.get("a2_sources"), label="a2_sources")
    required: dict[str, Mapping[str, Any]] = {}
    for relative in REQUIRED_A2_SOURCE_PATHS:
        value = sources.get(relative)
        if not isinstance(value, Mapping):
            raise ProductionProviderError(f"A2 source binding is missing: {relative}")
        required[relative] = value
    helpers: dict[str, Mapping[str, Any]] = {}
    for relative in HELPER_SOURCE_PATHS:
        helpers[relative] = _helper_descriptor(receipt, relative)
    return required, helpers


def _configuration(
    binding: Mapping[str, Any],
    *,
    arm_name: str,
    bank_role: str,
    diagnostic_binding: Mapping[str, Any],
    expected_contract_sha256: str = SIGNED_STAGE3_CONTRACT_SHA256,
    expected_countersignature_sha256: str = SIGNED_STAGE3_COUNTERSIGNATURE_SHA256,
    expected_qualification_sha256: str = MEASURED_QUALIFICATION_RECEIPT_SHA256,
    expected_resource_guard_sha256: str = MEASURED_RESOURCE_GUARD_SHA256,
) -> dict[str, Any]:
    signed_contract = _validate_signed_stage3_contract(
        binding,
        expected_contract_sha256=expected_contract_sha256,
        expected_countersignature_sha256=expected_countersignature_sha256,
    )
    settings = _validate_fixed_settings(binding, arm_name=arm_name, bank_role=bank_role)
    schedule = _validate_schedule_metadata(binding, arm_name=arm_name, signed_contract=signed_contract)
    diagnostic = _validate_diagnostic(diagnostic_binding, bank_role=bank_role)
    artifacts = _validate_artifact_declarations(binding, arm_name=arm_name)
    support = _declared_support(binding, bank_role=bank_role)
    sources, helpers = _source_bindings(binding)
    validation_geometry = _mapping(binding.get("validation_geometry"), label=f"{arm_name}.validation_geometry")
    if _int(validation_geometry.get("record_count"), label=f"{arm_name}.validation_geometry.record_count") != 384:
        raise ProductionProviderError(f"{arm_name} validation geometry must contain 384 rows")
    if _int(validation_geometry.get("sequence_tokens"), label=f"{arm_name}.validation_geometry.sequence_tokens") != EXPECTED_VALIDATION_TOKENS:
        raise ProductionProviderError(f"{arm_name} validation width differs from 128")
    resource = _validate_resource_qualification(
        binding,
        signed_contract=signed_contract,
        expected_qualification_sha256=expected_qualification_sha256,
        expected_resource_guard_sha256=expected_resource_guard_sha256,
    )
    return {
        "arm_name": arm_name,
        "bank_role": bank_role,
        "settings": settings,
        "schedule": schedule,
        "diagnostic": diagnostic,
        "support": support,
        "artifacts": {key: {"path": value.get("path"), "bytes": value.get("bytes"), "sha256": value.get("sha256")} for key, value in artifacts.items()},
        "a2_sources": {key: {"path": value.get("path"), "bytes": value.get("bytes"), "sha256": value.get("sha256"), "commit": value.get("commit")} for key, value in sources.items()},
        "a2_helpers": {key: {"path": value.get("path"), "bytes": value.get("bytes"), "sha256": value.get("sha256"), "commit": value.get("commit")} for key, value in helpers.items()},
        "validation": {"record_count": 384, "sequence_tokens": EXPECTED_VALIDATION_TOKENS},
        "resource_qualification": resource,
        "model_allocated": False,
        "updates": False,
        "truth_opened": False,
    }


def configuration_dry_run(
    *,
    binding_receipts: Mapping[str, Mapping[str, Any]],
    diagnostic_binding: Mapping[str, Any],
    lease_caps: Mapping[str, Any] | None = None,
    device: torch.device | None = None,
    output_root: Path | None = None,
    arm_name: str | None = None,
    expected_contract_sha256: str = SIGNED_STAGE3_CONTRACT_SHA256,
    expected_countersignature_sha256: str = SIGNED_STAGE3_COUNTERSIGNATURE_SHA256,
    expected_qualification_sha256: str = MEASURED_QUALIFICATION_RECEIPT_SHA256,
    expected_resource_guard_sha256: str = MEASURED_RESOURCE_GUARD_SHA256,
) -> dict[str, Any]:
    """Validate both explicit arm bindings without model/payload allocation."""

    del lease_caps, device, output_root
    arms: dict[str, Any] = {}
    selected = REQUIRED_BANKS if arm_name in (None, "both") else {arm_name: REQUIRED_BANKS.get(arm_name)}
    if any(role is None for role in selected.values()):
        raise ProductionProviderError(f"unknown dry-run arm: {arm_name}")
    for selected_arm, bank_role in selected.items():
        receipt = binding_receipts.get(selected_arm)
        if not isinstance(receipt, Mapping):
            raise ProductionProviderError(f"binding receipt is missing: {selected_arm}")
        arms[selected_arm] = _configuration(
            receipt,
            arm_name=selected_arm,
            bank_role=str(bank_role),
            diagnostic_binding=diagnostic_binding,
            expected_contract_sha256=expected_contract_sha256,
            expected_countersignature_sha256=expected_countersignature_sha256,
            expected_qualification_sha256=expected_qualification_sha256,
            expected_resource_guard_sha256=expected_resource_guard_sha256,
        )
    return {
        "schema": PRODUCTION_SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS_CONFIGURATION_DRY_RUN",
        "arms": arms,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "schedule_steps_count": TRAINING_STEPS,
        "model_allocated": False,
        "updates": False,
        "truth_opened": False,
    }


def _load_bound_helpers(receipt: Mapping[str, Any], runner: Any, loader_module: Any) -> tuple[Any, Any, Any]:
    """Import hash-bound A2 caller/CLI/validation modules under their package."""

    _, helpers = _source_bindings(receipt)
    helper_paths = {relative: _verify_descriptor(descriptor, label=f"A2 helper {relative}") for relative, descriptor in helpers.items()}
    package_root = helper_paths[HELPER_SOURCE_PATHS[0]].parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    # The helper modules use relative imports.  Point those imports at the
    # already hash-bound runner/stream loader rather than an ambient copy.
    sys.modules["scripts.trr_p09.fixed_control_runner"] = runner
    sys.modules["scripts.trr_p09.prepare_streamed_bank"] = loader_module
    importlib.invalidate_caches()
    caller = importlib.import_module("scripts.trr_p09.fixed_control_caller")
    fixed_cli = importlib.import_module("scripts.trr_p09.fixed_control_cli")
    public_loader = importlib.import_module("scripts.trr_p09.public_validation_loader")
    for relative, module in ((HELPER_SOURCE_PATHS[0], caller), (HELPER_SOURCE_PATHS[1], fixed_cli), (HELPER_SOURCE_PATHS[2], public_loader)):
        if Path(str(getattr(module, "__file__", ""))).resolve() != helper_paths[relative]:
            raise ProductionProviderError(f"imported A2 helper path differs: {relative}")
    return caller, fixed_cli, public_loader


def _resource_qualification(binding: Mapping[str, Any]) -> ResourceQualification:
    value = _mapping(binding.get("resource_qualification"), label="resource_qualification")
    fields = (
        "gpu_peak_reserved_bytes", "gpu_reserved_limit_bytes", "host_peak_rss_bytes",
        "host_rss_limit_bytes", "host_available_bytes", "host_available_floor_bytes",
    )
    return ResourceQualification(
        status=str(value.get("status")),
        **{field: _int(value.get(field), label=f"resource_qualification.{field}") for field in fields},
        wall_seconds=float(value.get("wall_seconds")),
        wall_limit_seconds=float(value.get("wall_limit_seconds")),
    )


def _production_bindings(binding: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> tuple[ProductionBindings, dict[str, str]]:
    required_sources, helpers = _source_bindings(binding)
    def frozen(role: str, status: str, label: str) -> FrozenArtifact:
        value = artifacts[role]
        return FrozenArtifact(label=label, path=str(value["path"]), bytes=_int(value["bytes"], label=f"{role}.bytes"), sha256=str(value["sha256"]), status=status)
    source_objects = [
        A2SourceBinding(path=relative, commit=str(value.get("commit")), sha256=str(value.get("sha256")))
        for relative, value in required_sources.items()
    ]
    source_objects.extend(
        A2SourceBinding(path=relative, commit=str(value.get("commit")), sha256=str(value.get("sha256")))
        for relative, value in helpers.items()
    )
    bindings = ProductionBindings(
        contract=frozen("contract", "FROZEN", "TRR-0010 frozen contract"),
        bank_manifest=frozen("bank_manifest", "VERIFIED", "TRR-0010 bank manifest"),
        schedule=frozen("schedule", "FROZEN", "TRR-0010 common schedule"),
        resource_qualification=_resource_qualification(binding),
        a2_sources=tuple(source_objects),
    )
    bindings.validate()
    source_paths = {relative: str(value["path"]) for relative, value in required_sources.items()}
    return bindings, source_paths


def _validation_callback(public_loader: Any, caller: Any, runner: Any) -> tuple[Any, Any]:
    labels = public_loader.label_join
    def factory(_step: int, _domain: str, rows: tuple[int, ...]) -> list[Any]:
        if len(rows) % EXPECTED_BATCH_RECORDS:
            raise ProductionProviderError("validation domain is not batch aligned")
        return [public_loader.get_records(rows[index:index + EXPECTED_BATCH_RECORDS]) for index in range(0, len(rows), EXPECTED_BATCH_RECORDS)]
    callback = caller.make_domain_validation_callback(labels, factory)
    return callback, labels


def build_inputs(
    binding_receipt: Mapping[str, Any],
    lease_caps: Mapping[str, Any],
    device: torch.device,
    preparation_guard: Callable[[str], None],
    *,
    arm_name: str | None = None,
    bank_role: str | None = None,
    checkpoint_steps: Sequence[int] = CHECKPOINT_STEPS,
    output_root: Path | None = None,
    diagnostic_binding: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build one actual current-B0 or expanded-B1 directional runtime."""

    if arm_name not in REQUIRED_BANKS or bank_role != REQUIRED_BANKS[arm_name]:
        raise ProductionProviderError("build_inputs requires explicit current_directional/B0 or expanded_directional/B1")
    if tuple(int(step) for step in checkpoint_steps) != CHECKPOINT_STEPS:
        raise ProductionProviderError("checkpoint grid differs from frozen production grid")
    if diagnostic_binding is None:
        raise ProductionProviderError("diagnostic_binding is required")
    config_summary = _configuration(binding_receipt, arm_name=arm_name, bank_role=bank_role, diagnostic_binding=diagnostic_binding)
    artifacts = _validate_artifact_declarations(binding_receipt, arm_name=arm_name)
    schedule_summary = _validate_schedule_metadata(binding_receipt, arm_name=arm_name)
    diagnostic_summary = _validate_diagnostic(diagnostic_binding, bank_role=bank_role)
    preparation_guard("before_a2_import")
    a2_adapter, runner, loader_module, b0_module = _load_a2_modules(binding_receipt)
    if b0_module is None:
        raise ProductionProviderError("B0 loader source is required")
    caller, fixed_cli, public_validation_module = _load_bound_helpers(binding_receipt, runner, loader_module)
    preparation_guard("after_a2_import")
    bindings, source_paths = _production_bindings(binding_receipt, artifacts)
    contract_path, _ = _artifact(binding_receipt, "contract")
    contract = _read_json(contract_path, label="production contract")
    settings, training = _settings_and_training(binding_receipt)
    frozen_settings = _mapping(config_summary.get("settings"), label=f"{arm_name}.validated_settings")
    hidden_size = int(frozen_settings["hidden_size"])
    vocabulary_size = int(frozen_settings["vocabulary_size"])
    base_path, base_binding = _artifact(binding_receipt, "base_state")
    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    context_width = _metadata_int(metadata, "context_width", "sequence_tokens")
    bottleneck_size = _metadata_int(metadata, "bottleneck_size")
    if context_width != EXPECTED_VALIDATION_TOKENS or bottleneck_size is None:
        raise ProductionProviderError("base state context/bottleneck metadata differs")
    preparation_guard("before_base_load")
    from trr0010_model import load_positionwise_model_state
    base_decoder = load_positionwise_model_state(
        base_path,
        method_id=str(base_binding.get("method_id", "trr0007_residual_mlp512")),
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
        bottleneck_size=bottleneck_size,
    ).to(device)
    preparation_guard("after_base_load")
    embedding_path, embedding_binding = _artifact(binding_receipt, "public_embedding")
    public_embedding = _load_embedding(embedding_path, device=device, expected_shape=(vocabulary_size, hidden_size))
    support_ids_path, _ = _artifact(binding_receipt, "support_ids")
    support_counts_path, _ = _artifact(binding_receipt, "support_counts")
    support_ids = _load_support(support_ids_path, key="support_ids")
    support_counts = _load_support(support_counts_path, key="support_counts")
    support_expectation = config_summary.get("support")
    from trr0010_model import support_digest
    actual_support_digest = support_digest(support_ids, support_counts)
    if isinstance(support_expectation, Mapping):
        expected_count = _int(support_expectation.get("support_count"), label="support_by_bank.support_count", positive=True)
        if int(support_ids.numel()) != expected_count or int(support_counts.numel()) != expected_count:
            raise ProductionProviderError(
                f"loaded {bank_role} support vectors do not match the bank-specific contract count"
            )
        if actual_support_digest != str(support_expectation["support_digest"]):
            raise ProductionProviderError(f"loaded {bank_role} support vectors differ from the frozen digest")
    preparation_guard("after_public_state_load")
    b0_path = _verify_descriptor(binding_receipt["b0_binding"], label="B0 immutable binding")
    bank_path, bank_binding = _artifact(binding_receipt, "bank_manifest")
    if bank_role == "B0":
        bank_loader = b0_module.B0ImmutableLoader(b0_path, device="cpu")
    else:
        bank_loader = b0_module.CombinedB0StreamedBankLoader(b0_path, bank_path, device="cpu")
    source = runner.RandomAccessLoaderSource(bank_loader)
    schedule_path, _ = _artifact(binding_receipt, "schedule")
    schedule_steps, schedule_receipt = deserialize_schedule(
        schedule_path,
        runner=runner,
        expected=binding_receipt["schedule"],
        sequence_tokens=EXPECTED_FIT_TOKENS,
        batch_records=EXPECTED_BATCH_RECORDS,
        position_budget=EXPECTED_POSITION_BUDGET,
    )
    if len(schedule_steps) != TRAINING_STEPS:
        raise ProductionProviderError("deserialized schedule does not contain 13000 steps")
    preparation_guard("after_bank_schedule_load")
    validation_geometry = _mapping(binding_receipt.get("validation_geometry"), label="validation_geometry")
    validation_manifest = validation_geometry.get("manifest")
    validation_rows = validation_geometry.get("rows")
    if not isinstance(validation_manifest, Mapping) or not isinstance(validation_rows, Mapping):
        raise ProductionProviderError("public validation manifest/rows bindings are missing")
    validation_manifest_path = _verify_descriptor(validation_manifest, label="public validation manifest")
    validation_rows_path = _verify_descriptor(validation_rows, label="public validation rows")
    public_validation = public_validation_module.PublicValidationLoader(validation_manifest_path, validation_rows_path)
    validation_callback, label_join = _validation_callback(public_validation, caller, runner)
    preparation_guard("after_public_validation_load")
    raw_diag, diagnostic_rows = fixed_cli._load_fixed_diagnostic_rows(
        Path(str(diagnostic_binding["path"])),
        bank=bank_role,
        row_limit=int(bank_loader.global_row_stop),
        expected_sha256=str(diagnostic_binding["sha256"]),
    )
    fixed_cli._validate_fixed_diagnostic_rows(raw_diag, diagnostic_rows, bank_loader)
    fit_source = source
    diagnostic_callback = caller.make_fitting_metric_callback(
        source=fit_source,
        embedding=public_embedding,
        fixed_rows=diagnostic_rows,
        full_bank_rows=tuple(range(int(bank_loader.global_row_stop))),
        final_step=TRAINING_STEPS,
        expected_sequence_tokens=EXPECTED_FIT_TOKENS,
        expected_hidden_size=EXPECTED_HIDDEN_SIZE,
        record_batch_size=EXPECTED_BATCH_RECORDS,
        position_budget=EXPECTED_POSITION_BUDGET,
        activation_dtype=torch.bfloat16,
        compute_base_logits=False,
        fixed_row_count=64,
    )
    preparation_guard("after_diagnostic_binding")
    config = runner.RunnerConfig(
        steps=TRAINING_STEPS,
        record_batch_size=EXPECTED_BATCH_RECORDS,
        position_budget=EXPECTED_POSITION_BUDGET,
        validation_every=int(frozen_settings["validation_every"]),
        selection_metric=EXPECTED_SELECTION_METRIC,
        seed=int(schedule_summary["seed"]),
        learning_rate=EXPECTED_BASE_LEARNING_RATE,
        weight_decay=EXPECTED_WEIGHT_DECAY,
        gradient_clip_norm=EXPECTED_GRADIENT_CLIP_NORM,
        train_sequence_tokens=EXPECTED_FIT_TOKENS,
        hidden_size=EXPECTED_HIDDEN_SIZE,
        expected_activation_dtype="torch.bfloat16",
    )
    config.validate()
    runtime = prepare_directional_runtime(
        protocol=a2_adapter,
        base_decoder=base_decoder,
        support_ids=support_ids,
        support_counts=support_counts,
        public_embedding=public_embedding,
        bindings=bindings,
        source_paths=source_paths,
        base_learning_rate=EXPECTED_BASE_LEARNING_RATE,
        weight_decay=EXPECTED_WEIGHT_DECAY,
        artifact_root=None,
    )
    fit_manifest = dict(artifacts["fit_manifest"])
    provider_receipt = {
        "schema": "token-reconstruction.trr0010-production-provider.v1",
        "task_id": TASK_ID,
        "arm_name": arm_name,
        "bank_role": bank_role,
        "start_state_sha256": EXPECTED_START_STATE_SHA256,
        "start_selected_step": EXPECTED_START_SELECTED_STEP,
        "schedule": schedule_summary,
        "schedule_receipt_sha256": schedule_receipt.get("semantic_sha256", schedule_receipt.get("sha256")),
        "production_bindings_sha256": bindings.source_binding_digest(),
        "validation_label_join_sha256": label_join.semantic_sha256,
        "diagnostic": diagnostic_summary,
        "support": {
            "count": int(support_ids.numel()),
            "digest": actual_support_digest,
            "contract": dict(support_expectation) if isinstance(support_expectation, Mapping) else None,
        },
        "fit_record_count": 64,
        "full_bank_row_count": int(bank_loader.global_row_stop),
        "truth_opened": False,
    }
    return {
        "arm_name": arm_name,
        "bank_role": bank_role,
        "runner": runner,
        "adapter_module": a2_adapter,
        "runner_module": runner,
        "loader_module": loader_module,
        "combined_loader_module": b0_module,
        "base_decoder": base_decoder,
        "public_embedding": public_embedding,
        "support_ids": support_ids,
        "support_counts": support_counts,
        "source": source,
        "schedule_steps": schedule_steps,
        "schedule_steps_count": TRAINING_STEPS,
        "schedule_seed": int(schedule_summary["seed"]),
        "schedule_semantic_sha256": str(schedule_summary["semantic_sha256"]),
        "schedule_exposure": dict(schedule_summary["exposure"]),
        "schedule_binding": dict(schedule_summary["schedule_binding"]),
        "control_schedule_binding": dict(schedule_summary["control_schedule_binding"]),
        "config": config,
        "validation_callback": validation_callback,
        "validation_sequence_tokens": EXPECTED_VALIDATION_TOKENS,
        "validation_batch_records": EXPECTED_BATCH_RECORDS,
        "validation_activation_dtype": torch.bfloat16,
        "training_activation_dtype": torch.bfloat16,
        "diagnostic_binding": dict(diagnostic_summary),
        "diagnostic_callback": diagnostic_callback,
        "runtime": runtime,
        "base_state": dict(artifacts["base_state"]),
        "fit_manifest": fit_manifest,
        "provider_receipt": provider_receipt,
        "resource_guard_callback": None,
        "artifact_root": None,
        "output_root": str(output_root) if output_root is not None else None,
    }


__all__ = ["PRODUCTION_SCHEMA", "ProductionProviderError", "build_inputs", "configuration_dry_run"]
