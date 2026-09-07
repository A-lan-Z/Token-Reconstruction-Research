#!/usr/bin/env python3
"""Aggregate the prospective TRR-0009 decision from sealed score/cost JSON.

This module is deliberately downstream of the owner-gated scorer and timing
receipt.  It does not open truth, predictions, source rows, or model files.  It
only consumes the per-cell ``candidate_vs_fixed`` score rows, the eight rare
frequency-bin rows, and the timing qualification/alias-control payload.

The contract is the only source of thresholds.  The primary decision is an OR
over the predeclared Finance public-base exact and token routes.  Exact and
token harm safeguards are evaluated separately in all four cells.  Rare and
absent bins remain ``UNKNOWN`` when the scorer did not compute a bound or when
the exposed-source floor is not met.  A timing result is usable only after its
fixed-method alias has qualified in every cell.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import trr0009_eval_contract as eval_contract


DECISION_SCHEMA = "token-reconstruction.trr0009-decision.v1"
PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
EXPECTED_SCORE_SCHEMA = "token-reconstruction.trr0009-score.v1"
EXPECTED_SCORE_STATUS = "SCORE_COMPLETE_AFTER_PUBLIC_FREEZE"


class DecisionError(ValueError):
    """Raised when a score, cost receipt, or contract is malformed."""


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionError(f"{description} must be an object")
    return value


def _finite_number(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionError(f"{description} is not numeric") from exc
    if not math.isfinite(result):
        raise DecisionError(f"{description} is not finite")
    return result


def _optional_number(value: Any, description: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _same_float(left: Any, right: Any, description: str) -> None:
    left_value = _finite_number(left, description)
    right_value = _finite_number(right, description)
    if not math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-12):
        raise DecisionError(f"{description} changed: {left_value} != {right_value}")


def _status_all(rows: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = [str(row.get("status", UNKNOWN)) for row in rows.values()]
    if any(status == FAIL for status in statuses):
        return FAIL
    if any(status != PASS for status in statuses):
        return UNKNOWN
    return PASS


def _or_status(rows: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = [str(row.get("status", UNKNOWN)) for row in rows.values()]
    if any(status == PASS for status in statuses):
        return PASS
    if any(status == UNKNOWN for status in statuses):
        return UNKNOWN
    return FAIL


def _contract_rules(decision_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed interface and extract only declared thresholds."""

    payload = _mapping(decision_contract, "decision contract")
    if payload.get("schema") != "token-reconstruction.trr0009-decision-contract.v1":
        raise DecisionError("decision contract schema changed")
    if payload.get("task_id") != eval_contract.TASK_ID:
        raise DecisionError("decision contract task identity changed")

    expected_cells = tuple(eval_contract.CELL_ORDER)
    primary = _mapping(payload.get("primary"), "primary contract")
    if primary.get("cell") != "finance__public_base":
        raise DecisionError("primary decision cell changed")
    if primary.get("contrast") != "continued_adaptable_readout minus continued_fixed_readout":
        raise DecisionError("primary decision contrast changed")
    exact_floor = _finite_number(primary.get("exact_point_floor"), "primary exact point floor")
    token_floor = _finite_number(primary.get("token_point_floor"), "primary token point floor")
    component_alpha = _finite_number(primary.get("component_alpha"), "primary exact component alpha")
    route_alpha = _finite_number(primary.get("route_alpha"), "primary route alpha")
    if exact_floor < 0.0 or token_floor < 0.0 or not 0.0 < component_alpha < 1.0 or not 0.0 < route_alpha < 1.0:
        raise DecisionError("primary decision thresholds are malformed")
    if not math.isclose(component_alpha, route_alpha / 2.0, rel_tol=0.0, abs_tol=1e-12):
        raise DecisionError("primary exact component alpha does not match its route alpha")
    if tuple(primary.get("routes", ())) != ("exact_record_recovery", "post_bos_token_accuracy"):
        raise DecisionError("primary decision routes changed")

    safeguards = _mapping(payload.get("safeguards"), "harm safeguards")
    if safeguards.get("all_four_cells_required") is not True or set(safeguards.get("cells", ())) != set(expected_cells):
        raise DecisionError("harm safeguard cell scope changed")
    exact_harm_margin = _finite_number(safeguards.get("exact_harm_margin"), "exact harm margin")
    token_harm_margin = _finite_number(safeguards.get("token_harm_margin"), "token harm margin")
    harm_alpha = _finite_number(safeguards.get("route_alpha"), "harm route alpha")
    if exact_harm_margin < 0.0 or token_harm_margin < 0.0 or not 0.0 < harm_alpha < 1.0:
        raise DecisionError("harm safeguard thresholds are malformed")

    rare = _mapping(payload.get("rare_token_safeguard"), "rare-token safeguard")
    if rare.get("all_eight_gates_required") is not True or rare.get("all_four_cells_required") is not True:
        raise DecisionError("rare-token safeguard scope changed")
    if set(rare.get("cells", ())) != set(expected_cells):
        raise DecisionError("rare-token safeguard cells changed")
    rare_bins = tuple(str(x) for x in rare.get("target_bins", ()))
    if len(rare_bins) != 2 or not rare_bins:
        raise DecisionError("rare-token safeguard bins changed")
    rare_margin = _finite_number(rare.get("harm_margin"), "rare-token harm margin")
    rare_support = int(rare.get("minimum_exposed_sources"))
    if rare_margin < 0.0 or rare_support <= 0:
        raise DecisionError("rare-token safeguard thresholds are malformed")

    cost_gate = _mapping(payload.get("cost_gate"), "cost gate")
    if cost_gate.get("all_cells_required") is not True or set(cost_gate.get("cells", ())) != set(expected_cells):
        raise DecisionError("cost gate cell scope changed")
    cost_threshold = _finite_number(cost_gate.get("threshold"), "cost threshold")
    if cost_threshold <= 0.0:
        raise DecisionError("cost threshold is malformed")

    bootstrap = _mapping(payload.get("bootstrap"), "bootstrap contract")
    bootstrap_seed = int(bootstrap.get("seed"))
    bootstrap_draws = int(bootstrap.get("draws"))
    if bootstrap_draws <= 0 or bootstrap.get("unit") != "source_record":
        raise DecisionError("bootstrap contract changed")

    return {
        "cells": expected_cells,
        "primary_cell": str(primary["cell"]),
        "candidate_method": eval_contract.PRIMARY_METHOD_ID,
        "control_method": eval_contract.PRIMARY_CONTROL_METHOD_ID,
        "exact_floor": exact_floor,
        "token_floor": token_floor,
        "component_alpha": component_alpha,
        "route_alpha": route_alpha,
        "exact_harm_margin": exact_harm_margin,
        "token_harm_margin": token_harm_margin,
        "harm_alpha": harm_alpha,
        "rare_bins": rare_bins,
        "rare_margin": rare_margin,
        "rare_support": rare_support,
        "cost_threshold": cost_threshold,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws": bootstrap_draws,
    }


def _validate_score(score: Mapping[str, Any], rules: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _mapping(score, "score")
    if payload.get("schema") != EXPECTED_SCORE_SCHEMA or payload.get("task_id") != eval_contract.TASK_ID:
        raise DecisionError("score schema or task identity changed")
    if payload.get("status") not in (None, EXPECTED_SCORE_STATUS):
        raise DecisionError("score is not a completed post-freeze score")
    if payload.get("truth_opened") is not True:
        raise DecisionError("score does not certify truth was opened after the public freeze")
    if tuple(payload.get("cell_order", ())) != tuple(rules["cells"]):
        raise DecisionError("score cell order changed")
    primary = _mapping(payload.get("primary"), "score primary binding")
    if primary.get("candidate") != rules["candidate_method"] or primary.get("control") != rules["control_method"]:
        raise DecisionError("score primary method binding changed")
    contrasts = _mapping(payload.get("contrasts"), "score contrasts")
    primary_contrast = _mapping(contrasts.get("candidate_vs_fixed"), "candidate_vs_fixed contrast")
    if set(primary_contrast) != set(rules["cells"]):
        raise DecisionError("candidate_vs_fixed must contain every unpooled cell")

    confidence = _mapping(payload.get("confidence"), "score confidence settings")
    primary_component = confidence.get("exact_component_alpha", confidence.get("primary_exact_component_alpha"))
    primary_composite = confidence.get("exact_composite_alpha")
    if primary_composite is None and primary_component is not None:
        primary_composite = 2.0 * _finite_number(primary_component, "score primary exact component alpha")
    primary_token_alpha = confidence.get("token_one_sided_alpha", confidence.get("primary_token_one_sided_alpha"))
    _same_float(primary_composite, rules["route_alpha"], "score exact composite alpha")
    _same_float(primary_component, rules["component_alpha"], "score exact component alpha")
    _same_float(primary_token_alpha, rules["route_alpha"], "score token alpha")
    safeguard_component = confidence.get("safeguard_exact_component_alpha")
    if safeguard_component is not None:
        _same_float(safeguard_component, rules["harm_alpha"] / 2.0, "score safeguard exact component alpha")
    safeguard_token_alpha = confidence.get("safeguard_token_one_sided_alpha")
    if safeguard_token_alpha is not None:
        _same_float(safeguard_token_alpha, rules["harm_alpha"], "score safeguard token alpha")
    bootstrap = _mapping(payload.get("bootstrap"), "score bootstrap settings")
    if int(bootstrap.get("seed")) != int(rules["bootstrap_seed"]) or int(bootstrap.get("draws")) != int(rules["bootstrap_draws"]):
        raise DecisionError("score bootstrap seed/draws changed")
    if bootstrap.get("unit") != "source_record":
        raise DecisionError("score bootstrap unit changed")
    rare = _mapping(payload.get("rare_token_safeguard"), "score rare-token safeguard")
    if rare.get("all_eight_gates_required") is not True:
        raise DecisionError("score does not report all eight rare-token gates")
    if set(_mapping(rare.get("rows"), "score rare-token rows")) != {f"{cell}::{bin_name}" for cell in rules["cells"] for bin_name in rules["rare_bins"]}:
        raise DecisionError("score rare-token rows are not the declared eight unpooled gates")
    return payload


def _interval_at_alpha(
    row: Mapping[str, Any],
    *,
    field_names: Sequence[str],
    by_alpha_names: Sequence[str],
    alpha: float,
) -> Mapping[str, Any] | None:
    """Find a serialized interval whose declared one-sided alpha matches."""

    def matches(candidate: Mapping[str, Any]) -> bool:
        value = _optional_number(candidate.get("one_sided_alpha"), "interval alpha")
        return value is not None and math.isclose(value, float(alpha), rel_tol=0.0, abs_tol=1e-12)

    for name in field_names:
        candidate = row.get(name)
        if isinstance(candidate, Mapping) and matches(candidate):
            return candidate
    for name in by_alpha_names:
        mapping = row.get(name)
        if not isinstance(mapping, Mapping):
            continue
        for key, candidate in mapping.items():
            if not isinstance(candidate, Mapping):
                continue
            key_value = _optional_number(key, "interval alpha key")
            if key_value is not None and math.isclose(key_value, float(alpha), rel_tol=0.0, abs_tol=1e-12):
                return dict(candidate) | {"one_sided_alpha": key_value}
    return None


def _route_result(
    *,
    name: str,
    point: Any,
    lower: Any,
    floor: float,
    interval_status: str | None = None,
    interval_alpha: Any = None,
    expected_alpha: float | None = None,
) -> dict[str, Any]:
    point_value = _optional_number(point, f"{name} point estimate")
    lower_value = _optional_number(lower, f"{name} lower bound")
    row: dict[str, Any] = {
        "status": UNKNOWN,
        "positive_status": UNKNOWN,
        "point": point_value,
        "lower_bound": lower_value,
        "point_floor": float(floor),
    }
    if expected_alpha is not None:
        row["one_sided_alpha"] = _optional_number(interval_alpha, f"{name} interval alpha")
    if point_value is None:
        row["reason"] = "missing_or_nonfinite_point_estimate"
        return row
    below_floor = point_value < float(floor)
    if interval_status is not None and interval_status != "COMPUTED":
        row["status"] = FAIL if below_floor else UNKNOWN
        row["reason"] = "point_below_useful_floor" if below_floor else f"interval_{interval_status.casefold()}"
        return row
    if expected_alpha is not None:
        alpha_value = _optional_number(interval_alpha, f"{name} interval alpha")
        if alpha_value is None or not math.isclose(alpha_value, float(expected_alpha), rel_tol=0.0, abs_tol=1e-12):
            row["status"] = FAIL if below_floor else UNKNOWN
            row["reason"] = "point_below_useful_floor" if below_floor else "interval_alpha_mismatch"
            return row
    if lower_value is None:
        row["status"] = FAIL if below_floor else UNKNOWN
        row["reason"] = "point_below_useful_floor" if below_floor else "missing_or_nonfinite_lower_bound"
        return row
    row["positive_status"] = PASS if lower_value > 0.0 else FAIL
    if below_floor:
        row["status"] = FAIL
        row["reason"] = "point_below_useful_floor"
        return row
    row["status"] = row["positive_status"]
    row["reason"] = "point_and_lower_bound_pass" if row["status"] == PASS else "lower_bound_not_positive"
    return row

def _exact_bound_at_alpha(row: Mapping[str, Any], alpha: float) -> tuple[float | None, str]:
    """Return a paired CP gain-minus-loss lower bound at the requested alpha.

    Primary exact bounds are already serialized by the scorer.  Safeguards use
    a distinct, wider composite alpha and therefore must be recomputed from
    the paired gain/loss counts (or supplied in an explicitly alpha-keyed
    score field).  Falling back to the primary bound would silently weaken the
    declared safeguard and is intentionally rejected as UNKNOWN.
    """

    by_alpha = row.get("exact_net_cp_lower_bound_by_alpha")
    if isinstance(by_alpha, Mapping):
        for key, value in by_alpha.items():
            key_value = _optional_number(key, "exact bound alpha key")
            if key_value is None or not math.isclose(key_value, float(alpha), rel_tol=0.0, abs_tol=1e-12):
                continue
            candidate = value.get("lower") if isinstance(value, Mapping) else value
            parsed = _optional_number(candidate, "exact safeguard lower bound")
            return parsed, "serialized_alpha_bound" if parsed is not None else "malformed_alpha_bound"
    for name in ("exact_net_cp_lower_bound_95", "exact_harm_net_cp_lower_bound"):
        if name in row:
            candidate = row[name]
            candidate = candidate.get("lower") if isinstance(candidate, Mapping) else candidate
            parsed = _optional_number(candidate, "exact safeguard lower bound")
            return parsed, "serialized_safeguard_bound" if parsed is not None else "malformed_safeguard_bound"

    try:
        records = int(row.get("records"))
        gains = int(row.get("exact_gains"))
        losses = int(row.get("exact_losses"))
    except (TypeError, ValueError):
        return None, "paired_counts_missing"
    if records <= 0 or gains < 0 or losses < 0 or gains + losses > records:
        return None, "paired_counts_malformed"
    try:
        from scipy.stats import beta
        component = float(alpha) / 2.0
        gain_lower = 0.0 if gains == 0 else float(beta.ppf(component, gains, records - gains + 1))
        loss_upper = 1.0 if losses == records else float(beta.ppf(1.0 - component, losses + 1, records - losses))
    except (ImportError, TypeError, ValueError, ZeroDivisionError):
        return None, "exact_cp_unavailable"
    bound = gain_lower - loss_upper
    return (bound, "recomputed_from_paired_counts") if math.isfinite(bound) else (None, "exact_cp_nonfinite")

def _primary_routes(score: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(_mapping(score["contrasts"], "score contrasts")["candidate_vs_fixed"][rules["primary_cell"]], "primary score row")
    token_interval = _interval_at_alpha(
        row,
        field_names=("token_net_bootstrap_975", "token_net_bootstrap_primary", "token_net_bootstrap"),
        by_alpha_names=("token_net_bootstrap_by_alpha",),
        alpha=float(rules["route_alpha"]),
    )
    exact = _route_result(
        name="exact",
        point=row.get("exact_net_rate"),
        lower=row.get("exact_net_cp_lower_bound"),
        floor=float(rules["exact_floor"]),
    )
    token = _route_result(
        name="token",
        point=row.get("token_net_rate"),
        lower=token_interval.get("lower") if token_interval is not None else None,
        floor=float(rules["token_floor"]),
        interval_status=str(token_interval.get("status", UNKNOWN)) if token_interval is not None else UNKNOWN,
        interval_alpha=token_interval.get("one_sided_alpha") if token_interval is not None else None,
        expected_alpha=float(rules["route_alpha"]),
    )
    routes = {"exact": exact, "token": token}
    return {
        "cell": rules["primary_cell"],
        "candidate": rules["candidate_method"],
        "control": rules["control_method"],
        "routes": routes,
        "status": _or_status(routes),
    }

def _bound_gate(*, bound: Any, minimum: float, description: str) -> dict[str, Any]:
    value = _optional_number(bound, description)
    if value is None:
        return {"status": UNKNOWN, "lower_bound": None, "minimum": float(minimum), "reason": "missing_or_nonfinite_lower_bound"}
    return {
        "status": PASS if value >= float(minimum) else FAIL,
        "lower_bound": value,
        "minimum": float(minimum),
        "reason": "bound_meets_harm_limit" if value >= float(minimum) else "bound_below_harm_limit",
    }


def _harm_gates(score: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    contrasts = _mapping(score["contrasts"], "score contrasts")
    primary = _mapping(contrasts["candidate_vs_fixed"], "candidate_vs_fixed contrast")
    exact: dict[str, Any] = {}
    token: dict[str, Any] = {}
    combined: dict[str, Any] = {}
    for cell in rules["cells"]:
        row = _mapping(primary[cell], f"candidate_vs_fixed {cell}")
        exact_lower, exact_source = _exact_bound_at_alpha(row, float(rules["harm_alpha"]))
        exact[cell] = _bound_gate(
            bound=exact_lower,
            minimum=-float(rules["exact_harm_margin"]),
            description=f"exact harm lower bound {cell}",
        )
        exact[cell]["alpha"] = float(rules["harm_alpha"])
        exact[cell]["bound_source"] = exact_source
        interval = _interval_at_alpha(
            row,
            field_names=("token_net_bootstrap_harm", "token_net_bootstrap_95", "token_net_bootstrap_safeguard"),
            by_alpha_names=("token_net_bootstrap_by_alpha",),
            alpha=float(rules["harm_alpha"]),
        )
        if interval is None or interval.get("status") != "COMPUTED":
            token[cell] = {
                "status": UNKNOWN,
                "lower_bound": _optional_number(interval.get("lower") if isinstance(interval, Mapping) else None, f"token harm lower bound {cell}"),
                "minimum": -float(rules["token_harm_margin"]),
                "alpha": float(rules["harm_alpha"]),
                "reason": "token_safeguard_interval_unavailable_or_alpha_mismatch",
            }
        else:
            token[cell] = _bound_gate(
                bound=interval.get("lower"),
                minimum=-float(rules["token_harm_margin"]),
                description=f"token harm lower bound {cell}",
            )
            token[cell]["alpha"] = float(rules["harm_alpha"])
        pair = {"exact": exact[cell], "token": token[cell]}
        combined[cell] = {"status": _status_all(pair), "exact": exact[cell], "token": token[cell]}
    return {
        "exact": {"status": _status_all(exact), "by_cell": exact, "minimum": -float(rules["exact_harm_margin"]), "alpha": float(rules["harm_alpha"])},
        "token": {"status": _status_all(token), "by_cell": token, "minimum": -float(rules["token_harm_margin"]), "alpha": float(rules["harm_alpha"])},
        "by_cell": combined,
        "status": _status_all(combined),
    }

def _rare_gates(score: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    rare = _mapping(score["rare_token_safeguard"], "score rare-token safeguard")
    rows = _mapping(rare.get("rows"), "score rare-token rows")
    by_gate: dict[str, Any] = {}
    for cell in rules["cells"]:
        for bin_name in rules["rare_bins"]:
            key = f"{cell}::{bin_name}"
            row = _mapping(rows[key], f"rare-token gate {key}")
            interval = _interval_at_alpha(
                row,
                field_names=("interval_95", "token_net_bootstrap_95", "interval_safeguard", "interval"),
                by_alpha_names=("interval_by_alpha",),
                alpha=float(rules["harm_alpha"]),
            )
            if interval is None or interval.get("status") != "COMPUTED":
                by_gate[key] = {
                    "status": UNKNOWN,
                    "cell": cell,
                    "frequency_bin": bin_name,
                    "minimum": -float(rules["rare_margin"]),
                    "minimum_exposed_source_records": int(rules["rare_support"]),
                    "exposed_source_records": interval.get("exposed_source_records") if isinstance(interval, Mapping) else None,
                    "lower_bound": _optional_number(interval.get("lower") if isinstance(interval, Mapping) else None, f"rare lower bound {key}"),
                    "alpha": float(rules["harm_alpha"]),
                    "reason": "interval_unavailable_or_alpha_mismatch",
                }
                continue
            try:
                exposed = int(interval.get("exposed_source_records"))
            except (TypeError, ValueError):
                exposed = -1
            lower = _optional_number(interval.get("lower"), f"rare lower bound {key}")
            if exposed < int(rules["rare_support"]):
                status = UNKNOWN
                reason = "fewer_than_minimum_exposed_source_records"
            elif lower is None:
                status = UNKNOWN
                reason = "missing_or_nonfinite_lower_bound"
            elif lower < -float(rules["rare_margin"]):
                status = FAIL
                reason = "bound_below_harm_limit"
            else:
                status = PASS
                reason = "bound_meets_harm_limit"
            by_gate[key] = {
                "status": status,
                "cell": cell,
                "frequency_bin": bin_name,
                "minimum": -float(rules["rare_margin"]),
                "minimum_exposed_source_records": int(rules["rare_support"]),
                "exposed_source_records": exposed if exposed >= 0 else None,
                "lower_bound": lower,
                "alpha": float(rules["harm_alpha"]),
                "reason": reason,
            }
    return {
        "status": _status_all(by_gate),
        "by_gate": by_gate,
        "minimum": -float(rules["rare_margin"]),
        "minimum_exposed_source_records": int(rules["rare_support"]),
        "alpha": float(rules["harm_alpha"]),
    }

def _timing_payload(cost: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _mapping(cost, "cost receipt")
    for key in ("timing", "timing_qualification", "cost_gate"):
        if isinstance(payload.get(key), Mapping):
            candidate = payload[key]
            if "qualification" in candidate or "alias_control" in candidate:
                return candidate
    return payload


def _cost_gate(cost: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    payload = _timing_payload(cost)
    qualification = payload.get("qualification")
    alias = payload.get("alias_control")
    if not isinstance(qualification, Mapping) or not isinstance(alias, Mapping):
        return {"status": UNKNOWN, "reason": "timing_qualification_or_alias_receipt_missing"}
    if qualification.get("candidate_method_id") != rules["candidate_method"] or qualification.get("denominator_method_id") != rules["control_method"]:
        raise DecisionError("cost receipt method roles changed")
    _same_float(qualification.get("threshold"), rules["cost_threshold"], "cost threshold")
    alias_decision = str(alias.get("decision", UNKNOWN))
    if alias_decision != PASS or alias.get("runtime_order_control_valid") is not True:
        return {
            "status": UNKNOWN,
            "reason": "alias_control_not_qualified",
            "alias_decision": alias_decision,
            "alias_control": dict(alias),
        }
    per_cell = _mapping(qualification.get("per_cell"), "cost qualification per-cell rows")
    if set(per_cell) != set(rules["cells"]):
        raise DecisionError("cost qualification must contain every unpooled cell")
    by_cell: dict[str, Any] = {}
    for cell in rules["cells"]:
        row = _mapping(per_cell[cell], f"cost qualification {cell}")
        upper = _optional_number(row.get("ci_upper"), f"cost upper confidence bound {cell}")
        lower = _optional_number(row.get("ci_lower"), f"cost lower confidence bound {cell}")
        if upper is None:
            by_cell[cell] = {"status": UNKNOWN, "ci_lower": lower, "ci_upper": upper, "threshold": float(rules["cost_threshold"]), "reason": "missing_or_nonfinite_ci"}
        elif upper <= float(rules["cost_threshold"]):
            by_cell[cell] = {"status": PASS, "ci_lower": lower, "ci_upper": upper, "threshold": float(rules["cost_threshold"]), "reason": "upper_ci_meets_cost_limit"}
        elif lower is not None and lower > float(rules["cost_threshold"]):
            by_cell[cell] = {"status": FAIL, "ci_lower": lower, "ci_upper": upper, "threshold": float(rules["cost_threshold"]), "reason": "lower_ci_exceeds_cost_limit"}
        else:
            by_cell[cell] = {"status": UNKNOWN, "ci_lower": lower, "ci_upper": upper, "threshold": float(rules["cost_threshold"]), "reason": "ci_straddles_cost_limit"}
    status = _status_all(by_cell)
    declared = str(qualification.get("decision", UNKNOWN))
    if declared == FAIL and status == PASS:
        raise DecisionError("cost receipt decision contradicts its per-cell bounds")
    if declared == "INVALID_ALIAS_CONTROL":
        status = UNKNOWN
    elif declared == "INCONCLUSIVE" and status == PASS:
        status = UNKNOWN
    return {
        "status": status,
        "by_cell": by_cell,
        "alias_control": dict(alias),
        "qualification_decision": declared,
        "candidate_method": rules["candidate_method"],
        "control_method": rules["control_method"],
        "threshold": float(rules["cost_threshold"]),
        "cost_evidence_reported": bool(payload.get("cost_evidence") is not None or cost.get("cost_evidence") is not None),
        "training_cost_threshold_applied": False,
    }


def aggregate_decision(
    score: Mapping[str, Any],
    cost: Mapping[str, Any],
    decision_contract: Mapping[str, Any],
    *,
    contract_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic prospective outcome without opening data."""

    rules = _contract_rules(decision_contract)
    score_payload = _validate_score(score, rules)
    primary = _primary_routes(score_payload, rules)
    harm = _harm_gates(score_payload, rules)
    rare = _rare_gates(score_payload, rules)
    cost_gate = _cost_gate(cost, rules)
    gates = {
        "primary_useful": primary,
        "exact_harm": harm["exact"],
        "token_harm": harm["token"],
        "harm": {"status": harm["status"], "by_cell": harm["by_cell"]},
        "rare_absent": rare,
        "cost": cost_gate,
    }
    statuses = [primary["status"], harm["status"], rare["status"], cost_gate["status"]]
    if any(status == FAIL for status in statuses):
        status = FAIL
        outcome = "NO_USEFUL_PILOT_SIGNAL"
        action = "RETAIN_REFERENCE_AND_REPORT_FAILED_GATE"
    elif any(status == UNKNOWN for status in statuses):
        status = UNKNOWN
        outcome = "UNRESOLVED"
        action = "RETAIN_REFERENCE_AND_LABEL_ADAPTATION_QUESTION_UNRESOLVED"
    else:
        status = PASS
        outcome = "USEFUL_MECHANISM_SCREENING_SIGNAL"
        action = "REPORT_MECHANISM_SCREENING_ONLY_NO_AUTOMATIC_REPLACEMENT"
    result: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "task_id": eval_contract.TASK_ID,
        "status": status,
        "outcome": outcome,
        "action": action,
        "primary_cell": rules["primary_cell"],
        "candidate_method": rules["candidate_method"],
        "control_method": rules["control_method"],
        "gates": gates,
        "contract_thresholds": {
            "primary_exact_point_floor": float(rules["exact_floor"]),
            "primary_token_point_floor": float(rules["token_floor"]),
            "primary_route_alpha": float(rules["route_alpha"]),
            "primary_exact_component_alpha": float(rules["component_alpha"]),
            "exact_harm_minimum": -float(rules["exact_harm_margin"]),
            "token_harm_minimum": -float(rules["token_harm_margin"]),
            "rare_harm_minimum": -float(rules["rare_margin"]),
            "rare_minimum_exposed_source_records": int(rules["rare_support"]),
            "cost_ratio_maximum": float(rules["cost_threshold"]),
        },
        "scope": {
            "cells": list(rules["cells"]),
            "rare_bins": list(rules["rare_bins"]),
            "paired_unit": "source_record",
            "target_conditions_pooled": False,
            "automatic_replacement": False,
            "automatic_retraining_or_sample_expansion": False,
        },
        "score_status": score_payload.get("status"),
        "truth_opened": True,
        "source_text_or_target_labels": False,
    }
    if contract_record is not None:
        result["decision_contract"] = dict(contract_record)
    return result


# Friendly aliases for callers that use the verb from the surrounding tools.
evaluate_decision = aggregate_decision
decide = aggregate_decision


def _file_record(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise DecisionError(f"{description} is unavailable: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionError(f"{description} is invalid JSON: {path}") from exc
    return dict(_mapping(value, description))


def _write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise DecisionError(f"refusing to overwrite decision artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return _file_record(path, description="decision artifact")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True, help="completed TRR-0009 score JSON")
    parser.add_argument("--cost", type=Path, required=True, help="timing qualification/cost JSON")
    parser.add_argument("--decision-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract_path = Path(args.decision_contract).expanduser().resolve()
    contract_payload = _load_json(contract_path, description="decision contract")
    contract_record = _file_record(contract_path, description="decision contract")
    score = _load_json(Path(args.score), description="score")
    cost = _load_json(Path(args.cost), description="cost receipt")
    result = aggregate_decision(score, cost, contract_payload, contract_record=contract_record)
    result["score_artifact"] = _file_record(Path(args.score), description="score artifact")
    result["cost_artifact"] = _file_record(Path(args.cost), description="cost artifact")
    result["decision_artifact"] = _write_create_only(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
