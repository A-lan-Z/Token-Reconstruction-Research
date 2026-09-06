"""Decision-focused TRR-0008 scorer.

The public functions in this module accept already materialized prediction
and truth tensors; they do not discover, select, or open a truth sidecar on
their own.  The command-line scorer is intended for the one explicit score
pass after the root-owned freeze gate.  The primary comparison is the
improved-bank residual candidate versus the retained reference on Finance
public-base exact clip recovery.  Shifted Finance and Pile cells are paired
safeguards; current residual and improved trained-diagonal contrasts are
reported as descriptive controls.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import torch
from safetensors import safe_open

from scripts import trr0008_eval_contract as contract
from scripts import trr0008_eval_gate as gate


class ScoreError(contract.ContractError):
    pass


def _cp_lower(successes: int, trials: int, alpha_component: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials or not 0 < alpha_component < 1:
        raise ScoreError("invalid Clopper-Pearson inputs")
    if successes == 0:
        return 0.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(alpha_component, successes, trials - successes + 1))
    except ImportError as exc:
        raise ScoreError("scipy is required for exact confidence bounds") from exc


def _cp_upper(successes: int, trials: int, alpha_component: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials or not 0 < alpha_component < 1:
        raise ScoreError("invalid Clopper-Pearson inputs")
    if successes == trials:
        return 1.0
    try:
        from scipy.stats import beta
        # The upper tail is 1-alpha, not alpha.  This direction is tested.
        return float(beta.ppf(1.0 - alpha_component, successes + 1, trials - successes))
    except ImportError as exc:
        raise ScoreError("scipy is required for exact confidence bounds") from exc


def clopper_pearson(successes: int, trials: int, *, alpha_component: float = 0.0125) -> dict[str, float]:
    """Return the declared two-sided bound using one-sided component alpha."""

    return {
        "lower": _cp_lower(successes, trials, alpha_component),
        "upper": _cp_upper(successes, trials, alpha_component),
        "alpha_component": float(alpha_component),
        "successes": int(successes),
        "trials": int(trials),
    }


def _bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int,
    draws: int,
    one_sided_alpha: float,
) -> dict[str, float]:
    """Record-bootstrap quantiles for a declared one-sided confidence bound."""

    if not values:
        raise ScoreError("cannot bootstrap an empty record set")
    if draws <= 0 or not 0 < one_sided_alpha < 1:
        raise ScoreError("invalid bootstrap settings")
    rng = random.Random(int(seed))
    n = len(values)
    estimates: list[float] = []
    for _ in range(int(draws)):
        estimates.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    lower_index = max(
        0,
        min(len(estimates) - 1, int(math.floor(one_sided_alpha * len(estimates)))),
    )
    upper_index = max(
        0,
        min(
            len(estimates) - 1,
            int(math.ceil((1.0 - one_sided_alpha) * len(estimates)) - 1),
        ),
    )
    return {
        "lower": float(estimates[lower_index]),
        "upper": float(estimates[upper_index]),
        "draws": int(draws),
        "seed": int(seed),
        "one_sided_alpha": float(one_sided_alpha),
        "one_sided_confidence": float(1.0 - one_sided_alpha),
        "unit": "source_record",
    }

def _as_prediction_map(predictions: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    """Accept nested method->cell or flat ``method::cell`` prediction maps."""

    result: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in predictions.items():
        if isinstance(value, Mapping) and "::" not in str(key):
            method_id = str(key)
            result[method_id] = {str(cell): torch.as_tensor(tensor) for cell, tensor in value.items()}
        else:
            if "::" not in str(key):
                raise ScoreError(f"prediction key is not method::cell: {key}")
            method_id, cell_id = str(key).split("::", 1)
            result.setdefault(method_id, {})[cell_id] = torch.as_tensor(value)
    return result


def _truth_for_cell(truth: Mapping[str, Any] | torch.Tensor, cell_id: str) -> torch.Tensor:
    if isinstance(truth, Mapping):
        if cell_id in truth:
            return torch.as_tensor(truth[cell_id])
        domain = cell_id.split("__", 1)[0]
        if domain in truth:
            return torch.as_tensor(truth[domain])
        raise ScoreError(f"truth tensor is absent for {cell_id}")
    return torch.as_tensor(truth)


def _cell_score(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    prediction = torch.as_tensor(prediction, dtype=torch.long).detach().cpu().contiguous()
    truth = torch.as_tensor(truth, dtype=torch.long).detach().cpu().contiguous()
    if prediction.ndim != 2 or truth.shape != prediction.shape or prediction.shape[1] != contract.STORED_SEQUENCE_TOKENS:
        raise ScoreError("prediction/truth geometry changed")
    contract.validate_prediction_tensor(prediction, records=int(prediction.shape[0]))
    if truth[:, 0].ne(contract.BOS_TOKEN_ID).any().item():
        raise ScoreError("truth BOS column changed")
    post_prediction = prediction[:, 1:]
    post_truth = truth[:, 1:]
    token_correct = post_prediction.eq(post_truth)
    record_exact = token_correct.all(dim=1)
    token_total = int(token_correct.numel())
    token_correct_count = int(token_correct.sum().item())
    exact_count = int(record_exact.sum().item())
    return {
        "records": int(prediction.shape[0]),
        "scored_post_bos_tokens": int(prediction.shape[0] * contract.SCORED_POST_BOS_TOKENS),
        "token_correct": token_correct_count,
        "token_errors": token_total - token_correct_count,
        "token_accuracy": token_correct_count / token_total,
        "exact_correct": exact_count,
        "exact_errors": int(prediction.shape[0] - exact_count),
        "exact_rate": exact_count / int(prediction.shape[0]),
        "record_exact_mask": record_exact,
        "token_correct_mask": token_correct,
        # Token positions are nested within source records.  Any paired token
        # interval therefore resamples these per-record means, never flattened
        # positions treated as independent observations.
        "record_token_accuracy": token_correct.to(torch.float32).mean(dim=1),
    }


def paired_contrast(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    bootstrap_seed: int = 8008,
    bootstrap_draws: int = 10000,
    alpha_component: float = 0.0125,
    safeguard_alpha_component: float | None = None,
    primary_token_alpha: float = 0.025,
    safeguard_token_alpha: float = 0.05,
) -> dict[str, Any]:
    candidate_exact = torch.as_tensor(candidate["record_exact_mask"], dtype=torch.bool)
    reference_exact = torch.as_tensor(reference["record_exact_mask"], dtype=torch.bool)
    candidate_tokens = torch.as_tensor(candidate["token_correct_mask"], dtype=torch.bool)
    reference_tokens = torch.as_tensor(reference["token_correct_mask"], dtype=torch.bool)
    if candidate_exact.shape != reference_exact.shape or candidate_tokens.shape != reference_tokens.shape:
        raise ScoreError("paired contrast record/token geometry changed")
    if candidate_tokens.ndim != 2 or candidate_tokens.shape[0] != candidate_exact.numel():
        raise ScoreError("paired token mask is not record-major")
    exact_delta = candidate_exact.to(torch.int8) - reference_exact.to(torch.int8)
    token_position_delta = candidate_tokens.to(torch.int8) - reference_tokens.to(torch.int8)
    candidate_record_tokens = candidate.get("record_token_accuracy")
    reference_record_tokens = reference.get("record_token_accuracy")
    if candidate_record_tokens is None:
        candidate_record_tokens = candidate_tokens.to(torch.float32).mean(dim=1)
    if reference_record_tokens is None:
        reference_record_tokens = reference_tokens.to(torch.float32).mean(dim=1)
    token_delta = (
        torch.as_tensor(candidate_record_tokens, dtype=torch.float32)
        - torch.as_tensor(reference_record_tokens, dtype=torch.float32)
    ).reshape(-1)
    if token_delta.numel() != exact_delta.numel():
        raise ScoreError("record-level token deltas changed")
    exact_values = exact_delta.tolist()
    token_values = token_delta.tolist()
    gains = int((exact_delta > 0).sum().item())
    losses = int((exact_delta < 0).sum().item())
    record_count = int(candidate_exact.numel())
    token_positions = int(token_position_delta.numel())
    exact_net = (gains - losses) / record_count
    token_net = float(token_delta.mean().item())
    gain_cp = clopper_pearson(gains, record_count, alpha_component=alpha_component)
    loss_cp = clopper_pearson(losses, record_count, alpha_component=alpha_component)
    safeguard_alpha = (
        float(safeguard_alpha_component)
        if safeguard_alpha_component is not None
        else float(alpha_component) * 2.0
    )
    if not 0.0 < safeguard_alpha < 1.0:
        raise ScoreError("invalid safeguard exact component alpha")
    gain_cp_95 = clopper_pearson(gains, record_count, alpha_component=safeguard_alpha)
    loss_cp_95 = clopper_pearson(losses, record_count, alpha_component=safeguard_alpha)
    return {
        "records": record_count,
        "tokens": token_positions,
        "bootstrap_unit": "source_record",
        "exact_gains": gains,
        "exact_losses": losses,
        "exact_ties": record_count - gains - losses,
        "exact_net_rate": exact_net,
        "token_net_rate": token_net,
        "token_improved": int((token_position_delta > 0).sum().item()),
        "token_harmed": int((token_position_delta < 0).sum().item()),
        "record_token_delta_mean": token_net,
        "exact_net_bootstrap_95": _bootstrap_interval(
            exact_values, seed=bootstrap_seed, draws=bootstrap_draws, one_sided_alpha=0.05
        ),
        "token_net_bootstrap_95": _bootstrap_interval(
            token_values,
            seed=bootstrap_seed + 1,
            draws=bootstrap_draws,
            one_sided_alpha=float(safeguard_token_alpha),
        ),
        "token_net_bootstrap_975": _bootstrap_interval(
            token_values,
            seed=bootstrap_seed + 2,
            draws=bootstrap_draws,
            one_sided_alpha=float(primary_token_alpha),
        ),
        "exact_gain_rate_cp": gain_cp,
        "exact_loss_rate_cp": loss_cp,
        "exact_gain_rate_cp_95": gain_cp_95,
        "exact_loss_rate_cp_95": loss_cp_95,
        # Conservative paired-discordance lower bounds: lower gain rate minus
        # upper loss rate.  The primary route uses alpha/2=.0125; safeguards
        # use one-sided 95% component alpha=.025.
        "exact_net_cp_lower_bound": gain_cp["lower"] - loss_cp["upper"],
        "exact_net_cp_lower_bound_95": gain_cp_95["lower"] - loss_cp_95["upper"],
        "exact_net_cp_alpha_component": float(alpha_component),
        "exact_net_cp_safeguard_alpha_component": safeguard_alpha,
        "token_bootstrap_primary_one_sided_alpha": float(primary_token_alpha),
        "token_bootstrap_safeguard_one_sided_alpha": float(safeguard_token_alpha),
    }

def score_predictions(
    predictions: Mapping[str, Any],
    truth: Mapping[str, Any] | torch.Tensor,
    *,
    bootstrap_seed: int = 8008,
    bootstrap_draws: int = 10000,
    alpha_component: float = 0.0125,
    safeguard_alpha_component: float | None = None,
    primary_token_alpha: float = 0.025,
    safeguard_token_alpha: float = 0.05,
) -> dict[str, Any]:
    """Score a complete frozen matrix after the root-owned truth gate.

    The CLI supplies every confidence allocation from the owner-frozen
    decision contract. Defaults remain only for direct unit-level callers.
    """

    matrix = _as_prediction_map(predictions)
    if set(matrix) != set(contract.METHOD_ORDER):
        raise ScoreError("score matrix must contain exactly the four scientific methods")
    cell_scores: dict[str, dict[str, Any]] = {}
    for method_id in contract.METHOD_ORDER:
        if set(matrix[method_id]) != set(contract.CELL_ORDER):
            raise ScoreError(f"score matrix cells changed for {method_id}")
        cell_scores[method_id] = {}
        for cell_id in contract.CELL_ORDER:
            cell_scores[method_id][cell_id] = _cell_score(matrix[method_id][cell_id], _truth_for_cell(truth, cell_id))
    contrasts: dict[str, dict[str, Any]] = {}
    for candidate_id in (
        contract.PRIMARY_METHOD_ID,
        contract.CURRENT_RESIDUAL_METHOD_ID,
        contract.IMPROVED_DIAGONAL_METHOD_ID,
    ):
        contrasts[candidate_id] = {}
        for index, cell_id in enumerate(contract.CELL_ORDER):
            contrasts[candidate_id][cell_id] = paired_contrast(
                cell_scores[candidate_id][cell_id],
                cell_scores[contract.REFERENCE_METHOD_ID][cell_id],
                bootstrap_seed=bootstrap_seed + index,
                bootstrap_draws=bootstrap_draws,
                alpha_component=alpha_component,
                safeguard_alpha_component=safeguard_alpha_component,
                primary_token_alpha=primary_token_alpha,
                safeguard_token_alpha=safeguard_token_alpha,
            )

    def direct(left_id: str, right_id: str, seed_offset: int) -> dict[str, Any]:
        return {
            cell_id: paired_contrast(
                cell_scores[left_id][cell_id],
                cell_scores[right_id][cell_id],
                bootstrap_seed=bootstrap_seed + seed_offset + index,
                bootstrap_draws=bootstrap_draws,
                alpha_component=alpha_component,
                safeguard_alpha_component=safeguard_alpha_component,
                primary_token_alpha=primary_token_alpha,
                safeguard_token_alpha=safeguard_token_alpha,
            )
            for index, cell_id in enumerate(contract.CELL_ORDER)
        }

    direct_contrasts = {
        "candidate_vs_current_residual": direct(
            contract.PRIMARY_METHOD_ID, contract.CURRENT_RESIDUAL_METHOD_ID, 100
        ),
        "candidate_vs_improved_diagonal": direct(
            contract.PRIMARY_METHOD_ID, contract.IMPROVED_DIAGONAL_METHOD_ID, 200
        ),
    }
    # Tensor masks are consumed above for paired contrasts but are not dumped
    # into the JSON result. Keep compact counts and rates so the result is
    # directly serializable and still auditable.
    public_cell_scores: dict[str, dict[str, Any]] = {}
    for method_id, cells in cell_scores.items():
        public_cell_scores[method_id] = {}
        for cell_id, value in cells.items():
            public_cell_scores[method_id][cell_id] = {
                key: item
                for key, item in value.items()
                if not torch.is_tensor(item)
            }
            public_cell_scores[method_id][cell_id].update(
                {
                    "record_exact_correct": int(value["exact_correct"]),
                    "record_exact_errors": int(value["exact_errors"]),
                    "token_correct": int(value["token_correct"]),
                    "token_errors": int(value["token_errors"]),
                }
            )
    return {
        "schema": contract.SCORE_SCHEMA,
        "task_id": contract.TASK_ID,
        "method_order": list(contract.METHOD_ORDER),
        "primary_method": contract.PRIMARY_METHOD_ID,
        "cell_scores": public_cell_scores,
        "contrasts_vs_reference": contrasts,
        "direct_contrasts": direct_contrasts,
        "bootstrap": {"seed": bootstrap_seed, "draws": bootstrap_draws, "unit": "source_record"},
        "confidence": {
            "primary_exact_component_alpha": float(alpha_component),
            "safeguard_exact_component_alpha": float(
                safeguard_alpha_component if safeguard_alpha_component is not None else float(alpha_component) * 2.0
            ),
            "primary_token_one_sided_alpha": float(primary_token_alpha),
            "safeguard_token_one_sided_alpha": float(safeguard_token_alpha),
        },
        "alpha_component": alpha_component,
        "truth_opened": True,
    }


def proposed_decision_contract() -> dict[str, Any]:
    """Return only a pointer to the owner-frozen nested planning contract.

    Execution must load experiments/TRR-0008/planning/decision_contract.json;
    this helper deliberately contains no duplicate thresholds.
    """

    return {
        "schema": "token-reconstruction.trr0008-decision-contract.v1",
        "task_id": contract.TASK_ID,
        "status": "PROSPECTIVE_DRAFT_PENDING_OWNER_FREEZE",
        "source": "experiments/TRR-0008/planning/decision_contract.json",
    }


def _numeric(mapping: Mapping[str, Any], *keys: str) -> float | int | str | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ScoreError(f"decision contract field is not numeric: {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoreError(f"decision contract field is not numeric: {label}") from exc
    if not math.isfinite(result):
        raise ScoreError(f"decision contract field is not finite: {label}")
    return result


def _same_float(actual: Any, expected: Any, *, label: str) -> float:
    actual_value = _finite_float(actual, label=label)
    expected_value = _finite_float(expected, label=label)
    if actual_value != expected_value:
        raise ScoreError(f"decision contract binding changed: {label}")
    return actual_value


def _decision_parameters(decision_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and expose only fields from the single owner-frozen contract."""

    if decision_contract.get("schema") != "token-reconstruction.trr0008-decision-contract.v1":
        raise ScoreError("decision contract schema changed")
    methods = decision_contract.get("methods")
    expected_methods = {
        "candidate": contract.PRIMARY_METHOD_ID,
        "credible_alternative": contract.CURRENT_RESIDUAL_METHOD_ID,
        "diagnostic": contract.IMPROVED_DIAGONAL_METHOD_ID,
        "reference": contract.REFERENCE_METHOD_ID,
    }
    if not isinstance(methods, Mapping) or any(
        methods.get(key) != value for key, value in expected_methods.items()
    ):
        raise ScoreError("decision contract method roles changed")

    primary = decision_contract.get("primary")
    token_endpoint = decision_contract.get("token_endpoint")
    safeguards = decision_contract.get("safeguards")
    confidence = decision_contract.get("confidence")
    advance = decision_contract.get("advance_rule")
    bootstrap = decision_contract.get("bootstrap")
    cost_gate = decision_contract.get("cost_gate")
    if not all(
        isinstance(value, Mapping)
        for value in (primary, token_endpoint, safeguards, confidence, advance, bootstrap, cost_gate)
    ):
        raise ScoreError("nested decision contract is incomplete")
    primary_quality = confidence.get("primary_quality")
    safeguard_quality = confidence.get("safeguard")
    primary_harm = safeguards.get("primary_harm")
    timing_binding = cost_gate.get("timing_receipt")
    if not isinstance(primary_quality, Mapping) or not isinstance(safeguard_quality, Mapping):
        raise ScoreError("decision confidence contract is incomplete")
    if not isinstance(primary_harm, Mapping) or not isinstance(timing_binding, Mapping):
        raise ScoreError("decision safeguard/cost binding is incomplete")

    primary_cell = primary.get("cell")
    if primary_cell != "finance__public_base":
        raise ScoreError("primary cell binding changed")
    primary_route_alpha = _same_float(
        primary.get("route_alpha"), primary_quality.get("route_alpha"), label="primary route alpha"
    )
    primary_component_alpha = _same_float(
        primary.get("component_alpha"),
        primary_quality.get("exact_cp_component_alpha"),
        label="primary exact CP component alpha",
    )
    if primary_component_alpha != primary_route_alpha / 2.0:
        raise ScoreError("primary exact CP component allocation changed")
    primary_token_alpha = _same_float(
        token_endpoint.get("route_alpha"),
        primary_quality.get("token_bootstrap_lower_tail_alpha"),
        label="primary token bootstrap lower tail alpha",
    )
    if primary_token_alpha != primary_route_alpha:
        raise ScoreError("primary token/quality route alpha differs")

    safeguard_route_alpha = _same_float(
        safeguards.get("route_alpha"),
        safeguard_quality.get("overall_alpha"),
        label="safeguard route alpha",
    )
    safeguard_component_alpha = _same_float(
        safeguard_quality.get("exact_cp_component_alpha"),
        safeguard_route_alpha / 2.0,
        label="safeguard exact CP component alpha",
    )
    safeguard_token_alpha = _same_float(
        safeguard_quality.get("token_bootstrap_lower_tail_alpha"),
        safeguard_route_alpha,
        label="safeguard token bootstrap lower tail alpha",
    )
    exact_margin = _finite_float(primary.get("practical_margin"), label="primary exact practical margin")
    token_margin = _finite_float(token_endpoint.get("practical_margin"), label="primary token practical margin")
    exact_harm = _finite_float(safeguards.get("exact_harm_margin"), label="safeguard exact harm margin")
    token_harm = _finite_float(safeguards.get("token_harm_margin"), label="safeguard token harm margin")
    if exact_margin < 0.0 or token_margin < 0.0 or exact_harm < 0.0 or token_harm < 0.0:
        raise ScoreError("decision margins must be non-negative magnitudes")
    if primary_harm.get("required") is not True or primary_harm.get("cell") != primary_cell:
        raise ScoreError("primary harm safeguard binding changed")
    _same_float(primary_harm.get("alpha"), safeguard_route_alpha, label="primary harm alpha")
    _same_float(
        primary_harm.get("exact_lower_bound_minimum"),
        -exact_harm,
        label="primary exact harm minimum",
    )
    _same_float(
        primary_harm.get("token_lower_bound_minimum"),
        -token_harm,
        label="primary token harm minimum",
    )
    safeguard_cells = safeguards.get("cells")
    if not isinstance(safeguard_cells, Sequence) or isinstance(safeguard_cells, (str, bytes)):
        raise ScoreError("safeguard cell set is absent")
    if set(safeguard_cells) != set(contract.CELL_ORDER):
        raise ScoreError("safeguard cell set changed")

    bootstrap_seed = bootstrap.get("seed")
    bootstrap_draws = bootstrap.get("draws")
    bootstrap_unit = bootstrap.get("unit")
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool) or bootstrap_seed < 0:
        raise ScoreError("bootstrap seed binding is malformed")
    if not isinstance(bootstrap_draws, int) or isinstance(bootstrap_draws, bool) or bootstrap_draws <= 0:
        raise ScoreError("bootstrap draw binding is malformed")
    if bootstrap_unit != "source_record":
        raise ScoreError("bootstrap unit binding changed")

    cost_threshold = _finite_float(cost_gate.get("threshold"), label="timing threshold")
    cost_primary_cell = cost_gate.get("primary_cell")
    cost_cells = cost_gate.get("cells")
    if cost_primary_cell != primary_cell or cost_gate.get("all_cells_required") is not True:
        raise ScoreError("cost gate primary/all-cell binding changed")
    if not isinstance(cost_cells, Sequence) or isinstance(cost_cells, (str, bytes)):
        raise ScoreError("cost gate cell set is absent")
    if set(cost_cells) != set(contract.CELL_ORDER):
        raise ScoreError("cost gate cell set changed")
    if cost_threshold <= 0.0:
        raise ScoreError("timing threshold must be positive")
    timing_path = timing_binding.get("path")
    timing_sha = timing_binding.get("sha256")
    if not isinstance(timing_path, str) or not timing_path:
        raise ScoreError("canonical timing path is absent")
    if not isinstance(timing_sha, str) or contract._SHA256.fullmatch(timing_sha) is None:
        raise ScoreError("canonical timing hash is malformed")
    try:
        timing_bytes = int(timing_binding.get("bytes"))
    except (TypeError, ValueError) as exc:
        raise ScoreError("canonical timing byte count is malformed") from exc
    if timing_bytes <= 0:
        raise ScoreError("canonical timing byte count is non-positive")
    if timing_binding.get("schema") != "token-reconstruction.trr0008-balanced-timing.v1":
        raise ScoreError("canonical timing schema changed")
    if timing_binding.get("status") != "TIMING_COMPLETE" or timing_binding.get("qualification") != "PASS":
        raise ScoreError("canonical timing qualification binding changed")
    if timing_binding.get("truth_opened") is not False:
        raise ScoreError("canonical timing truth binding changed")
    advance_threshold = _finite_float(advance.get("cost_threshold"), label="advance cost threshold")
    if advance_threshold != cost_threshold:
        raise ScoreError("advance/cost timing threshold differs")
    return {
        "primary_cell": str(primary_cell),
        "safeguard_cells": tuple(str(cell) for cell in safeguard_cells),
        "exact_practical_margin": exact_margin,
        "token_practical_margin": token_margin,
        "exact_harm_margin": exact_harm,
        "token_harm_margin": token_harm,
        "primary_component_alpha": primary_component_alpha,
        "safeguard_component_alpha": safeguard_component_alpha,
        "primary_token_alpha": primary_token_alpha,
        "safeguard_token_alpha": safeguard_token_alpha,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_draws": int(bootstrap_draws),
        "cost_primary_cell": str(cost_primary_cell),
        "cost_cells": tuple(str(cell) for cell in cost_cells),
        "cost_threshold": cost_threshold,
        "timing_binding": dict(timing_binding),
    }

def _timing_decision(
    timing: Mapping[str, Any] | None,
    *,
    primary_cell: str,
    cost_cells: Sequence[str],
    threshold: float,
) -> tuple[str, dict[str, Any] | None]:
    """Apply the bound final timing receipt without treating inconclusive data as failure."""

    if timing is None:
        return "COST_EVIDENCE_MISSING", None
    if timing.get("task_id") != contract.TASK_ID:
        return "COST_EVIDENCE_INVALID", None
    if timing.get("schema") != "token-reconstruction.trr0008-balanced-timing.v1":
        return "COST_EVIDENCE_INVALID", None
    if timing.get("status") != "TIMING_COMPLETE" or timing.get("truth_opened") is True:
        return "COST_EVIDENCE_INVALID", None
    configuration = timing.get("configuration")
    if not isinstance(configuration, Mapping) or int(configuration.get("blocks", -1)) != 40:
        return "COST_EVIDENCE_INVALID", None
    try:
        if float(configuration.get("threshold")) != float(threshold):
            return "COST_EVIDENCE_INVALID", None
    except (TypeError, ValueError):
        return "COST_EVIDENCE_INVALID", None
    equivalence = timing.get("equivalence")
    if not isinstance(equivalence, Mapping) or equivalence.get("status") != "PASS":
        return "COST_EVIDENCE_INVALID", None
    summary = timing.get("summary")
    qualification = summary.get("qualification") if isinstance(summary, Mapping) else None
    if not isinstance(qualification, Mapping):
        return "COST_EVIDENCE_INVALID", None
    if qualification.get("measurement_valid") is True and qualification.get("decision") == "FAIL":
        if qualification.get("cost_failure_demonstrated") is True:
            return "COST_GATE_FAILED", dict(qualification)
        return "COST_EVIDENCE_INCONCLUSIVE", dict(qualification)
    if qualification.get("decision") != "PASS" or qualification.get("measurement_valid") is not True:
        # Alias-control INCONCLUSIVE or any other non-PASS result is evidence
        # insufficiency, not a demonstrated candidate cost failure.
        return "COST_EVIDENCE_INCONCLUSIVE", dict(qualification)
    per_cell = qualification.get("per_cell")
    if primary_cell not in cost_cells:
        return "COST_EVIDENCE_INVALID", dict(qualification)
    if not isinstance(per_cell, Mapping) or set(per_cell) != set(cost_cells):
        return "COST_EVIDENCE_INVALID", dict(qualification)
    try:
        if float(qualification.get("threshold")) != float(threshold):
            return "COST_EVIDENCE_INVALID", dict(qualification)
    except (TypeError, ValueError):
        return "COST_EVIDENCE_INVALID", dict(qualification)
    failed = []
    for cell_id in cost_cells:
        row = per_cell[cell_id]
        if not isinstance(row, Mapping) or row.get("decision") != "PASS":
            failed.append(cell_id)
            continue
        try:
            if float(row["ci_upper"]) > threshold:
                failed.append(cell_id)
        except (KeyError, TypeError, ValueError):
            return "COST_EVIDENCE_INVALID", dict(qualification)
    if failed:
        # A per-cell ratio failure is a genuine cost gate failure only when
        # the timing qualification itself is valid.
        return "COST_GATE_FAILED", {"qualification": qualification, "failed_cells": failed}
    return "COST_PASS", dict(qualification)

def decide(
    score: Mapping[str, Any],
    decision_contract: Mapping[str, Any],
    *,
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the owner-frozen nested plan; never infer gates from point estimates."""

    if decision_contract.get("status") != "FROZEN_DECISION_CONTRACT_BEFORE_SOURCE_SELECTION":
        return {
            "status": "BLOCKED_UNTIL_DECISION_CONTRACT_FREEZE",
            "promotion": "retain_reference",
        }
    parameters = _decision_parameters(decision_contract)
    bootstrap = score.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or (
        bootstrap.get("seed") != parameters["bootstrap_seed"]
        or bootstrap.get("draws") != parameters["bootstrap_draws"]
        or bootstrap.get("unit") != "source_record"
    ):
        raise ScoreError("score bootstrap binding differs from the frozen decision contract")
    score_confidence = score.get("confidence")
    if not isinstance(score_confidence, Mapping):
        raise ScoreError("score confidence binding is absent")
    expected_confidence = {
        "primary_exact_component_alpha": parameters["primary_component_alpha"],
        "safeguard_exact_component_alpha": parameters["safeguard_component_alpha"],
        "primary_token_one_sided_alpha": parameters["primary_token_alpha"],
        "safeguard_token_one_sided_alpha": parameters["safeguard_token_alpha"],
    }
    for key, expected in expected_confidence.items():
        try:
            observed = float(score_confidence[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScoreError(f"score confidence binding is malformed: {key}") from exc
        if observed != float(expected):
            raise ScoreError(f"score confidence binding differs from the frozen decision contract: {key}")
    contrasts = score.get("contrasts_vs_reference")
    if not isinstance(contrasts, Mapping):
        raise ScoreError("score lacks reference contrasts")
    primary = contrasts.get(contract.PRIMARY_METHOD_ID, {}).get(parameters["primary_cell"])
    if not isinstance(primary, Mapping):
        raise ScoreError("score lacks the primary candidate/reference contrast")
    exact_lcb = float(primary["exact_net_cp_lower_bound"])
    token_lcb = float(primary["token_net_bootstrap_975"]["lower"])
    reliable_positive = exact_lcb > 0.0 or token_lcb > 0.0
    practical_exact = exact_lcb >= parameters["exact_practical_margin"]
    practical_token = token_lcb >= parameters["token_practical_margin"]
    practical = practical_exact or practical_token

    safeguard_rows = [
        contrasts[contract.PRIMARY_METHOD_ID][cell_id]
        for cell_id in parameters["safeguard_cells"]
    ]
    safeguard_failures = []
    for cell_id, row in zip(parameters["safeguard_cells"], safeguard_rows):
        exact_lower = float(row["exact_net_cp_lower_bound_95"])
        token_lower = float(row["token_net_bootstrap_95"]["lower"])
        if exact_lower < -parameters["exact_harm_margin"] or token_lower < -parameters["token_harm_margin"]:
            safeguard_failures.append(
                {
                    "cell_id": cell_id,
                    "exact_lower": exact_lower,
                    "token_lower": token_lower,
                }
            )
    safeguard = not safeguard_failures
    cost_status, timing_summary = _timing_decision(
        timing,
        primary_cell=parameters["cost_primary_cell"],
        cost_cells=parameters["cost_cells"],
        threshold=parameters["cost_threshold"],
    )
    if cost_status == "COST_EVIDENCE_MISSING":
        status = cost_status
    elif cost_status == "COST_EVIDENCE_INVALID":
        status = cost_status
    elif cost_status == "COST_EVIDENCE_INCONCLUSIVE":
        status = cost_status
    elif cost_status == "COST_GATE_FAILED":
        status = cost_status
    elif not reliable_positive:
        status = "UNRESOLVED_REFERENCE_RETAINED"
    elif not practical:
        status = "RELIABLE_BUT_PRACTICAL_MAGNITUDE_UNCERTAIN"
    elif not safeguard:
        status = "NO_PROMOTION_SAFEGUARD_FAILURE"
    else:
        status = "PROMOTE_PRIMARY_CANDIDATE"
    return {
        "status": status,
        "promotion": "promote_candidate" if status == "PROMOTE_PRIMARY_CANDIDATE" else "retain_reference",
        "primary": {
            "cell": parameters["primary_cell"],
            "exact_lcb": exact_lcb,
            "token_lcb": token_lcb,
            "reliably_positive": reliable_positive,
            "practical_exact": practical_exact,
            "practical_token": practical_token,
            "practical": practical,
        },
        "safeguards_pass": safeguard,
        "safeguard_failures": safeguard_failures,
        "primary_harm_safeguard": {
            "cell": parameters["primary_cell"],
            "exact_lower_bound": float(contrasts[contract.PRIMARY_METHOD_ID][parameters["primary_cell"]]["exact_net_cp_lower_bound_95"]),
            "token_lower_bound": float(contrasts[contract.PRIMARY_METHOD_ID][parameters["primary_cell"]]["token_net_bootstrap_95"]["lower"]),
            "exact_minimum": -parameters["exact_harm_margin"],
            "token_minimum": -parameters["token_harm_margin"],
            "passed": not any(row["cell_id"] == parameters["primary_cell"] for row in safeguard_failures),
        },
        "cost_status": cost_status,
        "timing": timing_summary,
    }

def _load_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise ScoreError(f"tensor key {key!r} is absent: {path}")
        return handle.get_tensor(key)


def _validate_truth_sidecar_metadata(
    truth_path: Path,
    *,
    header: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> set[str]:
    """Check sidecar metadata before loading any label tensor."""

    try:
        with safe_open(str(truth_path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            keys = set(handle.keys())
    except Exception as exc:
        raise ScoreError("truth sidecar is not a readable safetensors file") from exc
    if metadata.get("schema") != "token-reconstruction.trr0008-truth-sidecar.v1":
        raise ScoreError("truth sidecar schema changed")
    if metadata.get("task_id") != contract.TASK_ID or metadata.get("truth_opened") != "false":
        raise ScoreError("truth sidecar truth/task metadata changed")
    if metadata.get("registration_sha256") != registration.get("registration_sha256"):
        raise ScoreError("truth sidecar registration binding changed")
    source_selection = header.get("source_selection")
    if not isinstance(source_selection, Mapping) or metadata.get("source_selection_sha256") != source_selection.get("sha256"):
        raise ScoreError("truth sidecar source-selection binding changed")
    cells = header.get("cells")
    if not isinstance(cells, list):
        raise ScoreError("truth binding cell metadata is absent")
    digest_by_domain: dict[str, str] = {}
    for row in cells:
        if not isinstance(row, Mapping):
            raise ScoreError("truth binding cell metadata is malformed")
        cell_id = str(row.get("cell_id"))
        domain = cell_id.split("__", 1)[0]
        digest = row.get("record_ids_sha256")
        if domain not in contract.DOMAIN_ORDER or not isinstance(digest, str):
            raise ScoreError("truth binding record digest metadata is malformed")
        previous = digest_by_domain.setdefault(domain, digest)
        if previous != digest:
            raise ScoreError("truth binding target record digests disagree")
    try:
        observed_digests = json.loads(str(metadata.get("observation_record_ids_sha256")))
        observed_counts = json.loads(str(metadata.get("records_by_domain")))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ScoreError("truth sidecar observation/count metadata is malformed") from exc
    if observed_digests != digest_by_domain or observed_counts != header.get("records_by_domain"):
        raise ScoreError("truth sidecar observation/count binding changed")
    if metadata.get("sequence_tokens") != str(contract.STORED_SEQUENCE_TOKENS):
        raise ScoreError("truth sidecar sequence geometry metadata changed")
    if metadata.get("scored_post_bos_tokens") != str(contract.SCORED_POST_BOS_TOKENS):
        raise ScoreError("truth sidecar scored geometry metadata changed")
    if metadata.get("target_model_or_target_labels_loaded") != "false":
        raise ScoreError("truth sidecar target-label metadata changed")
    return keys


def _load_predictions(root: Path, output_root: Path) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for method_id in contract.METHOD_ORDER:
        result[method_id] = {}
        for cell_id in contract.CELL_ORDER:
            path = output_root / cell_id.split("__", 1)[0] / cell_id.split("__", 1)[1] / f"{method_id}.safetensors"
            result[method_id][cell_id] = _load_tensor(path, "predictions")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True, help="truth sidecar bound by --truth-binding")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--truth-binding", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--decision-contract", type=Path, required=True)
    parser.add_argument("--timing-receipt", type=Path)
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=None,
        help="optional assertion; the frozen decision contract supplies the value",
    )
    return parser


def _resolve_under_root(value: Path, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _assert_bound_record(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    description: str,
) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise ScoreError(f"{description} binding differs: {key}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path(args.repository_root).expanduser().resolve()
        decision_contract_path = _resolve_under_root(args.decision_contract, root=root)
        decision_contract = contract.load_json(
            decision_contract_path, description="owner-frozen TRR8 decision contract"
        )
        # This contract check is deliberately before the public revalidation and
        # before any private truth sidecar can be opened.  A draft or malformed
        # contract therefore cannot silently produce a score artifact.
        if decision_contract.get("status") != "FROZEN_DECISION_CONTRACT_BEFORE_SOURCE_SELECTION":
            raise ScoreError("scoring requires an owner-frozen decision contract")
        parameters = _decision_parameters(decision_contract)
        if args.bootstrap_draws is not None and args.bootstrap_draws != parameters["bootstrap_draws"]:
            raise ScoreError("--bootstrap-draws differs from the frozen decision contract")

        decision_contract_record = {
            "path": str(decision_contract_path),
            "bytes": int(decision_contract_path.stat().st_size),
            "sha256": contract.sha256_file(decision_contract_path),
        }
        bound_timing = contract.validate_file_record(
            parameters["timing_binding"],
            repository_root=root,
            description="canonical timing receipt",
            verify=True,
        )
        timing_path = Path(bound_timing["path"])
        if args.timing_receipt is not None:
            supplied_timing_path = _resolve_under_root(args.timing_receipt, root=root)
            if supplied_timing_path != timing_path:
                raise ScoreError("timing receipt differs from the frozen decision contract")
        timing = gate._load_timing(timing_path)

        freeze_path = _resolve_under_root(args.freeze_receipt, root=root)
        registration_path = _resolve_under_root(args.registration, root=root)
        run_path = _resolve_under_root(args.run_manifest, root=root)
        truth_binding_path = _resolve_under_root(args.truth_binding, root=root)
        receipt_doc = contract.load_json(freeze_path, description="public freeze receipt")
        freeze_timing = receipt_doc.get("timing_receipt")
        if not isinstance(freeze_timing, Mapping):
            raise ScoreError("public freeze receipt has no canonical timing binding")
        freeze_timing_record = contract.validate_file_record(
            freeze_timing,
            repository_root=root,
            description="public freeze timing receipt",
            verify=True,
        )
        _assert_bound_record(
            freeze_timing_record,
            bound_timing,
            description="freeze/decision timing",
        )

        # This is the sole authorization boundary before the private sidecar
        # is opened. It re-runs the complete public gate and reads only the
        # truth binding header as metadata.
        pretruth = gate.validate_before_truth(
            receipt_path=freeze_path,
            registration_path=registration_path,
            repository_root=root,
            truth_binding_path=truth_binding_path,
        )
        header = contract.load_json(truth_binding_path, description="TRR8 truth binding header")
        sidecar = header.get("sidecar")
        truth_path = _resolve_under_root(args.truth, root=root)
        if not isinstance(sidecar, Mapping) or Path(str(sidecar.get("path"))).expanduser().resolve() != truth_path:
            raise ScoreError("truth sidecar does not match the metadata-only truth binding")
        registration_doc = contract.load_registration(
            registration_path,
            repository_root=root,
            verify_assets=False,
        )
        registration_doc["registration_sha256"] = contract.sha256_file(registration_path)
        expected_output_root = _resolve_under_root(Path(str(registration_doc["output_root"])), root=root)
        requested_output_root = _resolve_under_root(args.predictions_root, root=root)
        if requested_output_root != expected_output_root:
            raise ScoreError("predictions root differs from the verified registration output root")
        declared_run = receipt_doc.get("run_manifest")
        actual_run = contract.validate_file_record(
            {"path": str(run_path), "bytes": int(run_path.stat().st_size), "sha256": contract.sha256_file(run_path)},
            repository_root=root,
            description="requested prediction run manifest",
            verify=True,
        )
        if not isinstance(declared_run, Mapping):
            raise ScoreError("public freeze run binding is absent")
        _assert_bound_record(actual_run, declared_run, description="requested/frozen run manifest")
        # Validate the sidecar identity after the public gate and before any
        # tensor labels are read. This catches replacement/truncation.
        checked_sidecar = contract.validate_file_record(
            sidecar,
            repository_root=root,
            description="truth sidecar",
            verify=True,
        )
        _assert_bound_record(checked_sidecar, sidecar, description="truth sidecar")
        expected_truth_keys = _validate_truth_sidecar_metadata(
            truth_path, header=header, registration=registration_doc
        )
        predictions = _load_predictions(root, expected_output_root)
        truth: dict[str, torch.Tensor] = {}
        # This is the sole explicit truth-opening operation in the scorer.
        with safe_open(str(truth_path), framework="pt", device="cpu") as handle:
            expected_keys = {f"{domain}__token_ids" for domain in contract.DOMAIN_ORDER}
            if expected_truth_keys != expected_keys:
                raise ScoreError("truth tensor key matrix changed")
            for domain in contract.DOMAIN_ORDER:
                key = f"{domain}__token_ids"
                truth[domain] = handle.get_tensor(key)
        result = score_predictions(
            predictions,
            truth,
            bootstrap_seed=parameters["bootstrap_seed"],
            bootstrap_draws=parameters["bootstrap_draws"],
            alpha_component=parameters["primary_component_alpha"],
            safeguard_alpha_component=parameters["safeguard_component_alpha"],
            primary_token_alpha=parameters["primary_token_alpha"],
            safeguard_token_alpha=parameters["safeguard_token_alpha"],
        )
        result["public_freeze"] = pretruth["public_freeze"]
        result["truth_binding"] = pretruth["truth_binding"]
        result["decision_contract"] = decision_contract_record
        result["timing_receipt"] = bound_timing
        result["decision"] = decide(result, decision_contract, timing=timing)
        # Serialize before the create-only write so tensor masks or another
        # accidental non-JSON value cannot appear after private truth access.
        json.dumps(result, sort_keys=True, allow_nan=False)
        contract.write_create_only(args.result, result)
    except (ScoreError, gate.GateError, contract.ContractError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"TRR-0008 score failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "SCORED", "result": str(args.result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
