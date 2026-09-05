"""Task-local TRR-0005 contracts and prospective evaluation ledger.

The TRR-0004 confirmation helpers are intentionally bound to four cells and
five methods.  This module keeps the small amount of TRR-0005-specific shape
and method metadata in one place so the producer, predictor, freeze adapter,
and scorer cannot silently drift apart.  It contains no dataset loader, model
loader, holdout selector, or truth reader.

The prospective ledger is useful before the reserved source pools are opened:
it records the planned ranges and geometry but deliberately contains no row
IDs, token IDs, source text, or target labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any


TASK_ID = "TRR-0005"
CONTRACT_SCHEMA = "token-reconstruction.trr0005-contract.v1"
LEDGER_SCHEMA = "token-reconstruction.trr0005-preplanned-ledger.v1"
REGISTRATION_SCHEMA = "token-reconstruction.trr0005-confirmation-registration.v1"
PANEL_SCHEMA = "token-reconstruction.trr0005-fresh-confirmation-panel.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0005-fresh-confirmation-prediction.v1"

STYLE_ORDER = ("pile", "finance")
CONDITION_ORDER = ("public_base", "public_lora_2601")
RECORDS_PER_DOMAIN = 128
SEQUENCE_TOKENS = 128
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
INVALID_TOKEN_ID = -1

# The names are intentionally explicit about the fitting distribution.  A
# state from one distribution must never be compared as though it had been
# fitted on the other distribution.
DISTRIBUTION_ORDER = ("original", "enriched")
DISTRIBUTION_CONTRACT_IDS = {"original": "original_like_alpaca_v1", "enriched": "coverage_mix_v1"}
STATE_ORDER = (
    "joint_full_affine",
    "affine_causal_h_attention128",
    "affine_trained_diagonal_attention128",
)


def distribution_state_id(distribution: str, state: str) -> str:
    """Return the canonical method ID for one fitted distribution/state."""

    if distribution not in DISTRIBUTION_ORDER:
        raise ValueError(f"unknown TRR-0005 fitting distribution: {distribution}")
    if state not in STATE_ORDER:
        raise ValueError(f"unknown TRR-0005 decoder state: {state}")
    return f"{distribution}__{state}"


def _new_method_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = [
        {
            "id": "historical_alpaca_a1",
            "track": "anchor",
            "distribution": "retained_trr0004_alpaca",
            "state": "retained_historical_a1",
            "candidate_policy": "forbidden",
            "rule": "retained standalone historical A1 direct argmax; top-k is diagnostic only",
        },
        {
            "id": "frozen_a1_a2_k256",
            "track": "anchor",
            "distribution": "retained_trr0004_alpaca",
            "state": "retained_a1_a2_k256_cpu_embedding_port",
            # The K256 proposal is used internally to choose the output.  The
            # large candidate tensor is intentionally not persisted.
            "candidate_policy": "output_only",
            "rule": "retained A1 proposals scored by fixed public-prefix A2 K256; candidate arrays omitted after decision",
        },
    ]
    state_rules = {
        "joint_full_affine": "joint full-sequence affine decoder fitted on the declared public distribution",
        "affine_causal_h_attention128": "joint affine path plus zero-initialized causal H_0..H_i attention correction",
        "affine_trained_diagonal_attention128": "joint affine path plus trained current-position-only diagonal attention correction",
    }
    for distribution in DISTRIBUTION_ORDER:
        for state in STATE_ORDER:
            specs.append(
                {
                    "id": distribution_state_id(distribution, state),
                    "track": "coverage_by_context",
                    "distribution": distribution,
                    "state": state,
                    "candidate_policy": "forbidden",
                    "rule": state_rules[state],
                }
            )
    return tuple(specs)


METHOD_SPECS = _new_method_specs()
METHOD_IDS = tuple(spec["id"] for spec in METHOD_SPECS)
CANDIDATE_POLICIES = {spec["id"]: spec["candidate_policy"] for spec in METHOD_SPECS}
METHOD_SPEC_BY_ID = {spec["id"]: spec for spec in METHOD_SPECS}

EXPECTED_CELL_IDS = tuple(
    f"{style}__{condition}"
    for style in STYLE_ORDER
    for condition in CONDITION_ORDER
)

FIT_BANKS: dict[str, dict[str, Any]] = {
    "original": {
        "records": 1200,
        "sequence_tokens": 192,
        "post_bos_positions": 124371,
        "composition": {"alpaca": 1200},
        "source_contract_id": "original_like_alpaca_v1",
        "source_role": "TRR-0004 old Alpaca public fit bank",
    },
    "enriched": {
        "records": 1200,
        "sequence_tokens": 192,
        "post_bos_positions": 124371,
        "composition": {
            "alpaca_instruction": 600,
            "pile_natural": 300,
            "finance_instruction": 180,
            "controlled_pile_context": 60,
            "controlled_finance_context": 60,
        },
        "source_contract_id": "coverage_mix_v1",
        "source_role": "coverage_mix_v1 with actual public-model forwards",
    },
}

RESERVED_SOURCE_POOLS: dict[str, dict[str, Any]] = {
    "pile": {
        "train_frequency": {"start": 2000, "stop": 7000, "stop_inclusive": False},
        "future_holdout": {"start": 7000, "stop": 10000, "stop_inclusive": False},
    },
    "finance": {
        "train_frequency": {"start": 2000, "stop": 12000, "stop_inclusive": False},
        "future_holdout": {"start": 12000, "stop": 20000, "stop_inclusive": False},
    },
}

FREQUENCY_BINS = (
    ("0", 0, 0),
    ("1-4", 1, 4),
    ("5-19", 5, 19),
    ("20+", 20, None),
)
POSITION_BINS = (
    ("1-15", 1, 15),
    ("16-39", 16, 39),
    ("40-79", 40, 79),
    ("80+", 80, None),
)

TRAINING_CONTRACT: dict[str, Any] = {
    "steps": 3000,
    "seed": 4005,
    "loss_draws_per_step": 512,
    "loss_draws_scope": "post-BOS rows, sampled by one shared deterministic schedule for every arm",
    "fit_geometry": [1200, 192, 2048],
    "matching_length_multiset": True,
    "same_optimizer_and_update_budget": True,
    "correction_split": "joint training from one identity affine path; any optional public correction diagnostic is disjoint and shared across context arms",
    "base_error_diagnostic": "record base error/loss on the declared public fit stream before interpreting context gains",
    "gradient_clip_norm": 1.0,
    "truth_accessed": False,
}

TIMING_CONTRACT: dict[str, Any] = {
    "warmup_runs_per_record": 1,
    "measured_runs_per_record": 1,
    "warmup_output_must_equal_measured": True,
    "steady_interval": "one record activation transfer, method call, and predicted IDs transfer; cold loads excluded",
    "a2_k": 256,
    "a2_candidate_output": "omitted_after_decision",
}


class ContractError(ValueError):
    """Raised when a TRR-0005 descriptor violates the frozen contract."""


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def safe_method_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ContractError("method ID is unsafe")
    return value


def safe_relative_path(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{description} path is absent")
    if "\\" in value:
        raise ContractError(f"{description} path must use POSIX separators")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise ContractError(f"{description} path is unsafe: {value}")
    return value


def valid_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(f"{description} must be a lowercase SHA-256 digest")
    return value


def expected_cell_id(style: str, condition: str) -> str:
    if style not in STYLE_ORDER or condition not in CONDITION_ORDER:
        raise ContractError(f"unknown cell: {style}/{condition}")
    return f"{style}__{condition}"


def _reject_private_keys(value: Any, *, path: str = "descriptor") -> None:
    """Reject source/token/truth payload fields from public descriptors."""

    forbidden = (
        "oracle",
        "token_ids",
        "input_ids",
        "labels",
        "source_text",
        "plaintext",
        "target_tokens",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} keys must be strings")
            lowered = key.casefold().replace("-", "_")
            if any(fragment in lowered for fragment in forbidden):
                raise ContractError(f"{path}.{key} is private source/evaluator state")
            _reject_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private_keys(child, path=f"{path}[{index}]")


def validate_method_ids(method_ids: Sequence[str]) -> tuple[str, ...]:
    observed = tuple(method_ids)
    if observed != METHOD_IDS:
        raise ContractError(
            "TRR-0005 method order changed: "
            f"expected {METHOD_IDS!r}, observed {observed!r}"
        )
    return observed


def validate_registration(registration: Mapping[str, Any], *, require_frozen: bool = False) -> None:
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise ContractError("TRR-0005 registration schema changed")
    if registration.get("task_id") != TASK_ID:
        raise ContractError("TRR-0005 registration task ID changed")
    status = registration.get("status")
    allowed_status = {"PLANNED_METHOD_REGISTRATION", "FROZEN_METHOD_REGISTRATION"}
    if status not in allowed_status or (require_frozen and status != "FROZEN_METHOD_REGISTRATION"):
        raise ContractError("method registration is not in the required frozen state")
    if status == "FROZEN_METHOD_REGISTRATION":
        if re.fullmatch(r"[0-9a-f]{40}", str(registration.get("code_commit", ""))) is None:
            raise ContractError("frozen method registration has no full code commit")
        bindings = registration.get("state_bindings")
        if not isinstance(bindings, Mapping) or set(bindings) != set(METHOD_IDS):
            raise ContractError("frozen method registration has incomplete state bindings")
        for method_id in METHOD_IDS:
            value = bindings[method_id]
            if not isinstance(value, Mapping) or value.get("status") == "PENDING_STATE":
                raise ContractError("frozen method registration contains pending state bindings")
    validate_method_ids(registration.get("method_ids", ()))
    policies = registration.get("candidate_policies")
    if policies != CANDIDATE_POLICIES:
        raise ContractError("candidate output policies changed")
    methods = registration.get("methods")
    if not isinstance(methods, list) or tuple(row.get("id") for row in methods if isinstance(row, Mapping)) != METHOD_IDS:
        raise ContractError("method specification list changed")
    for row, expected in zip(methods, METHOD_SPECS):
        if not isinstance(row, Mapping):
            raise ContractError("method specification is malformed")
        for key in ("id", "track", "distribution", "state", "candidate_policy", "rule"):
            if row.get(key) != expected[key]:
                raise ContractError(f"method specification changed for {expected['id']}: {key}")
    contract = registration.get("training_contract")
    if contract != TRAINING_CONTRACT:
        raise ContractError("training contract changed")
    timing = registration.get("timing_contract")
    if timing != TIMING_CONTRACT:
        raise ContractError("timing contract changed")


def build_registration(
    *,
    status: str = "PLANNED_METHOD_REGISTRATION",
    code_commit: str | None = None,
    state_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic method registration without loading any state."""

    if status not in {"PLANNED_METHOD_REGISTRATION", "FROZEN_METHOD_REGISTRATION"}:
        raise ContractError(f"invalid registration status: {status}")
    if status == "FROZEN_METHOD_REGISTRATION":
        if not isinstance(code_commit, str) or re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
            raise ContractError("a frozen registration needs a full code commit")
        if not isinstance(state_bindings, Mapping) or set(state_bindings) != set(METHOD_IDS):
            raise ContractError("a frozen registration needs one binding per method in order")
    bindings: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        value = state_bindings.get(method_id) if isinstance(state_bindings, Mapping) else None
        if value is not None:
            if not isinstance(value, Mapping):
                raise ContractError(f"state binding is malformed: {method_id}")
            bindings[method_id] = dict(value)
        else:
            bindings[method_id] = {"status": "PENDING_STATE"}
    registration: dict[str, Any] = {
        "schema": REGISTRATION_SCHEMA,
        "task_id": TASK_ID,
        "status": status,
        "method_ids": list(METHOD_IDS),
        "methods": [dict(row) for row in METHOD_SPECS],
        "candidate_policies": dict(CANDIDATE_POLICIES),
        "training_contract": dict(TRAINING_CONTRACT),
        "timing_contract": dict(TIMING_CONTRACT),
        "state_bindings": bindings,
        "code_commit": code_commit,
        "truth_opened": False,
    }
    registration["contract_sha256"] = _sha256_json(
        {
            "method_ids": registration["method_ids"],
            "methods": registration["methods"],
            "candidate_policies": registration["candidate_policies"],
            "training_contract": registration["training_contract"],
            "timing_contract": registration["timing_contract"],
        }
    )
    validate_registration(registration)
    return registration


def build_preplanned_ledger(*, registration_sha256: str | None = None) -> dict[str, Any]:
    """Return the no-holdout-ID TRR-0005 plan used before source opening."""

    if registration_sha256 is not None:
        valid_sha256(registration_sha256, description="registration")
    ledger: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "task_id": TASK_ID,
        "status": "PREPLANNED_NO_HOLDOUT_SELECTED",
        "selection_order": [
            "freeze method IDs, rules, and training contract",
            "freeze all six fitted state bindings",
            "reserve and then select 128 paired public records per domain",
            "capture public_base and synthetic public_lora_2601 observations",
            "prepare truth sidecar outside the frozen reconstruction root",
        ],
        "fit_banks": {
            distribution: dict(bank)
            for distribution, bank in FIT_BANKS.items()
        },
        "fit_bank_invariants": {
            "bank_count": 2,
            "records_per_bank": 1200,
            "sequence_tokens": 192,
            "post_bos_positions_per_bank": 124371,
            "matching_length_multiset_required": True,
            "original_source": "original_like_alpaca_v1: old TRR-0004 Alpaca public fit bank",
            "coverage_mix_composition": "600 Alpaca + 300 Pile + 180 Finance + 60 controlled Pile + 60 controlled Finance records",
            "controlled_rows_require_actual_public_model_forward": True,
            "joint_initialization_no_frozen_memorized_base": True,
            "optional_correction_diagnostic_must_be_disjoint": True,
            "optional_correction_diagnostic_equal_between_context_arms": True,
        },
        "reserved_source_pools": {
            "reserved_before_scanning": True,
            "contents_opened": False,
            "ranges": {
                domain: {role: dict(bounds) for role, bounds in pools.items()}
                for domain, pools in RESERVED_SOURCE_POOLS.items()
            },
        },
        "development": {
            "source": "experiments/TRR-0004 mixed48 public validation",
            "records": 48,
            "role": "development checkpoint selection only",
            "holdout_reuse_forbidden": True,
        },
        "holdout": {
            "records_per_domain": RECORDS_PER_DOMAIN,
            "domains": list(STYLE_ORDER),
            "conditions": list(CONDITION_ORDER),
            "cells": len(EXPECTED_CELL_IDS),
            "sequence_tokens": SEQUENCE_TOKENS,
            "tokens_per_declared_record_including_bos": SEQUENCE_TOKENS,
            "unique_sources_total": 256,
            "selection_seed": 5005,
            "pairing": "same public record IDs across public_base and public_lora_2601 within each domain",
            "pooled_headline": False,
            "selection_status": "deferred_until_method_and_state_choices_freeze",
        },
        "methods": {
            "method_ids": list(METHOD_IDS),
            "anchor_count": 2,
            "new_state_count": 6,
            "new_state_matrix": {
                distribution: [distribution_state_id(distribution, state) for state in STATE_ORDER]
                for distribution in DISTRIBUTION_ORDER
            },
        },
        "training": {
            **dict(TRAINING_CONTRACT),
            "total_post_bos_supervision_draws": 3000 * 512,
        },
        "inference": dict(TIMING_CONTRACT),
        "diagnostics": {
            "frequency_bins": [name for name, _lo, _hi in FREQUENCY_BINS],
            "position_bins": [name for name, _lo, _hi in POSITION_BINS],
            "joint_key": "domain/style × target condition × method × frequency bin × position bin",
            "distribution_directory_ids": {"original": "original", "enriched": "enriched"},
            "distribution_source_contract_ids": dict(DISTRIBUTION_CONTRACT_IDS),
            "paired_token_unit": "complete source record; bootstrap recomputes micro correct/scored ratio",
            "paired_exact_unit": "complete source record; one-sided CP gain/loss bound",
            "descriptive_bootstrap_draws": 10000,
            "bootstrap_seed": 5005,
            "primary_contrasts": [
                "coverage_mix_v1 causal vs trained diagonal",
                "coverage_mix_v1 causal vs public-validation-selected best positionwise",
            ],
            "token_family_tail_alpha": 0.05 / 16,
            "exact_family_tail_alpha_each": 0.05 / 32,
        },
        "truth_policy": {
            "selection_uses_truth": False,
            "prediction_uses_truth": False,
            "truth_read_requires_complete_public_matrix": True,
            "truth_sidecar_outside_frozen_root": True,
        },
        "registration_sha256": registration_sha256,
    }
    _reject_private_keys(ledger)
    return ledger


def _record_ids_from_cell(cell: Mapping[str, Any], *, cell_id: str) -> tuple[str, ...]:
    records = cell.get("records")
    if not isinstance(records, list) or len(records) != RECORDS_PER_DOMAIN:
        raise ContractError(f"{cell_id} must contain exactly {RECORDS_PER_DOMAIN} records")
    result: list[str] = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ContractError(f"{cell_id} record {index} is malformed")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ContractError(f"{cell_id} record {index} has no public ID")
        valid_sha256(row.get("public_record_sha256"), description=f"{cell_id} record {index}")
        if set(row) - {"record_id", "public_record_sha256", "raw_index", "source_index", "valid_tokens"}:
            raise ContractError(f"{cell_id} record {index} contains unapproved fields")
        for key in ("raw_index", "source_index", "valid_tokens"):
            if key in row and (not isinstance(row[key], int) or row[key] < 0):
                raise ContractError(f"{cell_id} record {index} has invalid {key}")
        result.append(record_id)
    if len(set(result)) != len(result):
        raise ContractError(f"{cell_id} contains duplicate public IDs")
    return tuple(result)


def validate_panel_descriptor(panel: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Validate source-free four-cell metadata and target pairing."""

    _reject_private_keys(panel)
    if panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != TASK_ID:
        raise ContractError("TRR-0005 panel identity changed")
    if panel.get("status") != "FROZEN_FRESH_CONFIRMATION_PANEL":
        raise ContractError("TRR-0005 panel is not frozen")
    if panel.get("sequence_tokens") != SEQUENCE_TOKENS or panel.get("records_per_domain") != RECORDS_PER_DOMAIN:
        raise ContractError("TRR-0005 panel geometry changed")
    cells = panel.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(EXPECTED_CELL_IDS):
        raise ContractError("TRR-0005 panel must contain exactly four cells")
    by_domain: dict[str, tuple[str, ...]] = {}
    for style in STYLE_ORDER:
        for condition in CONDITION_ORDER:
            cell_id = expected_cell_id(style, condition)
            cell = cells.get(cell_id)
            if not isinstance(cell, Mapping):
                raise ContractError(f"panel cell is malformed: {cell_id}")
            if cell.get("cell_id") != cell_id or cell.get("style") != style or cell.get("condition") != condition:
                raise ContractError(f"panel cell identity changed: {cell_id}")
            record_ids = _record_ids_from_cell(cell, cell_id=cell_id)
            if condition == CONDITION_ORDER[0]:
                by_domain[style] = record_ids
            elif by_domain[style] != record_ids:
                raise ContractError(f"target pairing changed for {style}")
    return by_domain


def validate_prediction_descriptor(
    descriptor: Mapping[str, Any],
    *,
    cell_id: str,
    method_id: str,
    records: int = RECORDS_PER_DOMAIN,
    sequence_tokens: int = SEQUENCE_TOKENS,
) -> None:
    """Validate a prediction metadata receipt before truth can be opened."""

    safe_method_id(method_id)
    if method_id not in METHOD_SPEC_BY_ID:
        raise ContractError(f"unknown method: {method_id}")
    if cell_id not in EXPECTED_CELL_IDS:
        raise ContractError(f"unknown cell: {cell_id}")
    if descriptor.get("schema") != PREDICTION_SCHEMA or descriptor.get("task_id") != TASK_ID:
        raise ContractError(f"prediction identity changed: {cell_id}/{method_id}")
    if descriptor.get("cell_id") != cell_id or descriptor.get("method_id") != method_id:
        raise ContractError(f"prediction binding changed: {cell_id}/{method_id}")
    if descriptor.get("shape") not in ([records, sequence_tokens], (records, sequence_tokens)):
        raise ContractError(f"prediction geometry changed: {cell_id}/{method_id}")
    policy = CANDIDATE_POLICIES[method_id]
    if descriptor.get("candidate_policy") != policy:
        raise ContractError(f"candidate policy changed: {cell_id}/{method_id}")
    has_candidates = bool(descriptor.get("candidate_arrays_present", False))
    if policy in {"forbidden", "output_only"} and has_candidates:
        raise ContractError(f"candidate arrays must not be persisted: {cell_id}/{method_id}")
    if policy == "output_only" and descriptor.get("candidate_output") != "omitted_after_decision":
        raise ContractError(f"A2 output-only omission is not attested: {cell_id}/{method_id}")
    if descriptor.get("warmup_runs_per_record") != TIMING_CONTRACT["warmup_runs_per_record"]:
        raise ContractError(f"warmup timing contract changed: {cell_id}/{method_id}")
    if descriptor.get("measured_runs_per_record") != TIMING_CONTRACT["measured_runs_per_record"]:
        raise ContractError(f"measured timing contract changed: {cell_id}/{method_id}")
    if descriptor.get("warmup_output_exact_match_measured") is not True:
        raise ContractError(f"warmup/measured output stability is missing: {cell_id}/{method_id}")
    if descriptor.get("measured_output_selected") is not True:
        raise ContractError(f"measured output selection is missing: {cell_id}/{method_id}")


def validate_complete_public_matrix(
    panel: Mapping[str, Any],
    registration: Mapping[str, Any],
    prediction_descriptors: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    timing_descriptors: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail closed unless all 32 public prediction/timing receipts are present."""

    by_domain = validate_panel_descriptor(panel)
    validate_registration(registration, require_frozen=True)
    expected = {(cell_id, method_id) for cell_id in EXPECTED_CELL_IDS for method_id in METHOD_IDS}
    observed = set(prediction_descriptors)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ContractError(f"public matrix is incomplete: missing={missing!r} extra={extra!r}")
    for cell_id, method_id in sorted(expected):
        validate_prediction_descriptor(
            prediction_descriptors[(cell_id, method_id)],
            cell_id=cell_id,
            method_id=method_id,
        )
    if timing_descriptors is None or set(timing_descriptors) != expected:
        raise ContractError("timing receipt matrix is incomplete")
    for cell_id, method_id in sorted(expected):
        timing = timing_descriptors[(cell_id, method_id)]
        if timing.get("warmup_runs_per_record") != 1 or timing.get("measured_runs_per_record") != 1:
            raise ContractError(f"timing count changed: {cell_id}/{method_id}")
        if timing.get("warmup_output_exact_match_measured") is not True:
            raise ContractError(f"timing output stability missing: {cell_id}/{method_id}")
    return {
        "task_id": TASK_ID,
        "panel_status": panel["status"],
        "domains": list(by_domain),
        "cells": len(EXPECTED_CELL_IDS),
        "methods": len(METHOD_IDS),
        "prediction_artifacts": len(expected),
        "timing_receipts": len(timing_descriptors),
        "truth_opened": False,
    }


__all__ = [
    "BOS_TOKEN_ID",
    "CANDIDATE_POLICIES",
    "CONDITION_ORDER",
    "CONTRACT_SCHEMA",
    "ContractError",
    "DISTRIBUTION_CONTRACT_IDS",
    "DISTRIBUTION_ORDER",
    "EXPECTED_CELL_IDS",
    "FIT_BANKS",
    "FREQUENCY_BINS",
    "INVALID_TOKEN_ID",
    "METHOD_IDS",
    "METHOD_SPECS",
    "PAD_TOKEN_ID",
    "PANEL_SCHEMA",
    "POSITION_BINS",
    "PREDICTION_SCHEMA",
    "RECORDS_PER_DOMAIN",
    "REGISTRATION_SCHEMA",
    "RESERVED_SOURCE_POOLS",
    "SEQUENCE_TOKENS",
    "STATE_ORDER",
    "STYLE_ORDER",
    "TASK_ID",
    "TIMING_CONTRACT",
    "TRAINING_CONTRACT",
    "build_preplanned_ledger",
    "build_registration",
    "distribution_state_id",
    "expected_cell_id",
    "safe_method_id",
    "safe_relative_path",
    "valid_sha256",
    "validate_complete_public_matrix",
    "validate_method_ids",
    "validate_panel_descriptor",
    "validate_prediction_descriptor",
    "validate_registration",
]
