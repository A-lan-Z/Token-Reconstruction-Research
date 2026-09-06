#!/usr/bin/env python3
"""Task-local TRR-0005 freeze-gated scoring adapter.

The adapter keeps the TRR-0004 public/truth separation but removes the
TRR-0004-specific five-method and 16-record assumptions.  It exposes pure
metric functions for synthetic tests and a small in-memory matrix scorer for
the later producer/predictor.  A caller that owns a truth sidecar should use
``score_with_truth_loader``: the loader is invoked only after the complete
four-cell-by-eight-method public gate and timing receipts have passed.

No holdout selector, model loader, or public source scanner lives here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Callable

import numpy as np
from safetensors import safe_open
import torch

from token_reconstruction.footing import (
    FootingError,
    external_file_record,
    file_record,
    validate_binding,
)
from token_reconstruction.freeze import (
    FreezeError,
    require_truth_open_allowed,
    verify_freeze_receipt,
)

from token_reconstruction.trr0005_contract import (
    BOS_TOKEN_ID,
    CANDIDATE_POLICIES,
    CONDITION_ORDER,
    ContractError,
    EXPECTED_CELL_IDS,
    FREQUENCY_BINS,
    INVALID_TOKEN_ID,
    METHOD_IDS,
    METHOD_SPEC_BY_ID,
    FIT_BANKS,
    POSITION_BINS,
    PREDICTION_SCHEMA,
    RECORDS_PER_DOMAIN,
    SEQUENCE_TOKENS,
    STYLE_ORDER,
    TASK_ID,
    TIMING_CONTRACT,
    validate_complete_public_matrix,
)


SCHEMA = "token-reconstruction.trr0005-confirmation-score.v1"
FREEZE_SCHEMA = "token-reconstruction.trr0005-confirmation-freeze.v1"
DEFAULT_BOOTSTRAP_SEED = 5005
DEFAULT_BOOTSTRAP_DRAWS = 10000
EXACT_CONFIDENCE = 0.95
EXACT_FAMILY_TAIL_ALPHA = 0.05 / 32
TOKEN_FAMILY_TAIL_ALPHA = 0.05 / 16
VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
CUT_DEPTH = 4
PUBLIC_SELECTION_SCHEMA = "token-reconstruction.trr0005-public-validation-selection.v1"
RUNTIME_EMBEDDING_ROLE = "public_embedding_table"
RUNTIME_P0_ROLES = ("public_prefix_checkpoint", "public_prefix_config")
A2_METHOD_ID = "frozen_a1_a2_k256"
TRUTH_MANIFEST_SCHEMA = "token-reconstruction.trr0005-fresh-confirmation-truth-preparation.v1"
TRUTH_MANIFEST_STATUS = "PUBLIC_TRUTH_PREPARED_OUTSIDE_RECONSTRUCTION_ROOT"


class ConfirmationScoreError(ContractError):
    """Raised when TRR-0005 scoring inputs are incomplete or inconsistent."""


class PretruthGateError(ConfirmationScoreError):
    """Raised when executable public evidence is not bound before truth."""


@dataclass(frozen=True)
class CellInput:
    """Truth and geometry for one cell, supplied only after the public gate."""

    style: str
    condition: str
    record_ids: tuple[str, ...]
    truth: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

    @property
    def cell_id(self) -> str:
        return f"{self.style}__{self.condition}"


def _tensor(value: Any, *, dtype: torch.dtype, description: str) -> torch.Tensor:
    try:
        result = torch.as_tensor(value, dtype=dtype).contiguous().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfirmationScoreError(f"{description} is not a tensor-like value") from exc
    return result


def _validate_cell_input(cell: CellInput) -> None:
    if cell.style not in STYLE_ORDER or cell.condition not in CONDITION_ORDER:
        raise ConfirmationScoreError(f"unknown cell: {cell.cell_id}")
    if len(cell.record_ids) != RECORDS_PER_DOMAIN:
        raise ConfirmationScoreError(
            f"{cell.cell_id} needs exactly {RECORDS_PER_DOMAIN} record IDs"
        )
    if len(set(cell.record_ids)) != len(cell.record_ids) or any(
        not isinstance(value, str) or not value for value in cell.record_ids
    ):
        raise ConfirmationScoreError(f"{cell.cell_id} record IDs are not unique public IDs")
    for name, value in (
        ("truth", cell.truth),
        ("attention_mask", cell.attention_mask),
        ("position_ids", cell.position_ids),
    ):
        if value.ndim != 2 or tuple(value.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
            raise ConfirmationScoreError(f"{cell.cell_id} {name} geometry changed")
    mask = cell.attention_mask.to(torch.bool)
    if not mask[:, 0].all().item() or (mask[:, 1:] > mask[:, :-1]).any().item():
        raise ConfirmationScoreError(f"{cell.cell_id} mask is not BOS/right-padded")
    expected_positions = mask.to(torch.long).cumsum(1).sub(1).clamp_min(0)
    if not torch.equal(cell.position_ids.to(torch.long), expected_positions):
        raise ConfirmationScoreError(f"{cell.cell_id} positions disagree with mask")
    if not torch.isfinite(cell.truth.to(torch.float32)).all().item():
        raise ConfirmationScoreError(f"{cell.cell_id} truth contains non-finite values")
    scored = mask.clone()
    scored[:, 0] = False
    if cell.truth[scored].lt(0).any().item() or cell.truth[scored].ge(128256).any().item():
        raise ConfirmationScoreError(f"{cell.cell_id} truth label is outside vocabulary")


def _normalise_prediction(predictions: Any, *, cell: CellInput, method_id: str) -> torch.Tensor:
    value = _tensor(predictions, dtype=torch.long, description="predictions")
    if tuple(value.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
        raise ConfirmationScoreError(f"{cell.cell_id}/{method_id} prediction geometry changed")
    active = cell.attention_mask.to(torch.bool)
    if not torch.equal(value[:, 0], torch.full((RECORDS_PER_DOMAIN,), BOS_TOKEN_ID, dtype=torch.long)):
        raise ConfirmationScoreError(f"{cell.cell_id}/{method_id} predictions do not retain BOS")
    # Padding can be INVALID_TOKEN_ID; scored positions must be vocabulary IDs.
    scored = active.clone()
    scored[:, 0] = False
    if value[scored].lt(0).any().item() or value[scored].ge(128256).any().item():
        raise ConfirmationScoreError(f"{cell.cell_id}/{method_id} scored prediction is outside vocabulary")
    if value[~active].ne(INVALID_TOKEN_ID).any().item():
        raise ConfirmationScoreError(f"{cell.cell_id}/{method_id} padding prediction must be -1")
    return value


def _frequency_tensor(frequencies: Any, *, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(frequencies, Mapping):
        values = [frequencies.get(int(label), 0) for label in labels.reshape(-1).tolist()]
        result = torch.tensor(values, dtype=torch.long)
    else:
        table = _tensor(frequencies, dtype=torch.long, description="frequency reference").reshape(-1)
        if labels.numel() and labels.max().item() >= table.numel():
            raise ConfirmationScoreError("frequency reference does not cover truth labels")
        result = table.index_select(0, labels.reshape(-1)).reshape(labels.shape)
    if result.lt(0).any().item():
        raise ConfirmationScoreError("frequency reference contains negative counts")
    return result


def _bin_index(values: torch.Tensor, bins: Sequence[tuple[str, int, int | None]]) -> torch.Tensor:
    result = torch.full(values.shape, -1, dtype=torch.long)
    for index, (_name, lower, upper) in enumerate(bins):
        selected = values.ge(lower)
        if upper is not None:
            selected &= values.le(upper)
        result[selected] = index
    if result.eq(-1).any().item():
        raise ConfirmationScoreError("diagnostic bins do not cover all scored rows")
    return result


def _group_metrics(
    correct: torch.Tensor,
    groups: torch.Tensor,
    bins: Sequence[tuple[str, int, int | None]],
    *,
    labels: torch.Tensor | None = None,
) -> dict[str, dict[str, Any]]:
    correct = correct.to(torch.bool).reshape(-1).cpu()
    groups = groups.to(torch.long).reshape(-1).cpu()
    if correct.shape != groups.shape:
        raise ConfirmationScoreError("diagnostic rows do not agree")
    result: dict[str, dict[str, Any]] = {}
    for index, (name, lower, upper) in enumerate(bins):
        selected = groups.eq(index)
        examples = int(selected.sum().item())
        correct_count = int(correct[selected].sum().item())
        row: dict[str, Any] = {
            "lower": lower,
            "upper": upper,
            "examples": examples,
            "correct_tokens": correct_count,
            "token_accuracy": correct_count / examples if examples else None,
        }
        if labels is not None:
            row["scored_tokens"] = examples
        result[name] = row
    return result


def _joint_frequency_position(
    *,
    correct: torch.Tensor,
    frequencies: torch.Tensor,
    positions: torch.Tensor,
    style: str,
    condition: str,
    method_id: str,
) -> dict[str, Any]:
    frequency_index = _bin_index(frequencies, FREQUENCY_BINS)
    position_index = _bin_index(positions, POSITION_BINS)
    rows: dict[str, Any] = {}
    for f_index, (f_name, f_lower, f_upper) in enumerate(FREQUENCY_BINS):
        for p_index, (p_name, p_lower, p_upper) in enumerate(POSITION_BINS):
            selected = frequency_index.eq(f_index) & position_index.eq(p_index)
            examples = int(selected.sum().item())
            correct_count = int(correct[selected].sum().item())
            key = f"{f_name}__{p_name}"
            rows[key] = {
                "style": style,
                "domain": style,
                "condition": condition,
                "method_id": method_id,
                "frequency_bin": f_name,
                "frequency_lower": f_lower,
                "frequency_upper": f_upper,
                "position_bin": p_name,
                "position_lower": p_lower,
                "position_upper": p_upper,
                "examples": examples,
                "scored_tokens": examples,
                "correct_tokens": correct_count,
                "token_accuracy": correct_count / examples if examples else None,
            }
    return {
        "style": style,
        "domain": style,
        "condition": condition,
        "method_id": method_id,
        "frequency_reference": "caller_supplied",
        "rows": rows,
    }


def score_cell(
    *,
    predictions: Any,
    cell: CellInput,
    method_id: str,
    frequency_counts: Any | None = None,
    frequency_reference_id: str | None = None,
) -> dict[str, Any]:
    """Score one post-gate cell and return micro, exact, and joint diagnostics."""

    if method_id not in METHOD_SPEC_BY_ID:
        raise ConfirmationScoreError(f"unknown method: {method_id}")
    _validate_cell_input(cell)
    prediction_tensor = _normalise_prediction(predictions, cell=cell, method_id=method_id)
    mask = cell.attention_mask.to(torch.bool)
    scored = mask.clone()
    scored[:, 0] = False
    truth = cell.truth.to(torch.long)
    correct = prediction_tensor.eq(truth)
    correct_scored = correct[scored]
    scored_count = int(scored.sum().item())
    correct_count = int(correct_scored.sum().item())
    per_record: list[dict[str, Any]] = []
    for index, record_id in enumerate(cell.record_ids):
        record_mask = scored[index]
        record_correct = int(correct[index][record_mask].sum().item())
        record_scored = int(record_mask.sum().item())
        per_record.append(
            {
                "record_id": record_id,
                "correct_tokens": record_correct,
                "scored_tokens": record_scored,
                "token_accuracy": record_correct / record_scored if record_scored else None,
                "exact_record": bool(record_scored > 0 and record_correct == record_scored),
            }
        )
    result: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "style": cell.style,
        "domain": cell.style,
        "condition": cell.condition,
        "method_id": method_id,
        "distribution": METHOD_SPEC_BY_ID[method_id]["distribution"],
        "metrics": {
            "scored_tokens": scored_count,
            "correct_tokens": correct_count,
            "token_accuracy": correct_count / scored_count if scored_count else None,
            "records": len(per_record),
            "exact_records": sum(bool(row["exact_record"]) for row in per_record),
            "exact_record_rate": (
                sum(bool(row["exact_record"]) for row in per_record) / len(per_record)
                if per_record
                else None
            ),
        },
        "per_record": per_record,
        "position_bins": _group_metrics(
            correct_scored,
            _bin_index(cell.position_ids[scored].to(torch.long), POSITION_BINS),
            POSITION_BINS,
        ),
    }
    if frequency_counts is not None:
        labels = truth[scored]
        frequencies = _frequency_tensor(frequency_counts, labels=labels)
        result["frequency_reference_id"] = frequency_reference_id or "caller_supplied"
        result["frequency_bins"] = _group_metrics(
            correct_scored,
            _bin_index(frequencies, FREQUENCY_BINS),
            FREQUENCY_BINS,
        )
        result["joint_frequency_position"] = _joint_frequency_position(
            correct=correct_scored,
            frequencies=frequencies,
            positions=cell.position_ids[scored].to(torch.long),
            style=cell.style,
            condition=cell.condition,
            method_id=method_id,
        )
        result["joint_frequency_position"]["frequency_reference"] = frequency_reference_id or "caller_supplied"
    else:
        result["frequency_reference_id"] = None
        result["frequency_bins"] = None
        result["joint_frequency_position"] = None
    return result


def _valid_record_rows(rows: Sequence[Mapping[str, Any]], *, name: str) -> list[dict[str, Any]]:
    if len(rows) != RECORDS_PER_DOMAIN:
        raise ConfirmationScoreError(f"{name} needs exactly {RECORDS_PER_DOMAIN} records")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for row in rows:
        record_id = row.get("record_id")
        correct = row.get("correct_tokens")
        scored = row.get("scored_tokens")
        accuracy = row.get("token_accuracy")
        if not isinstance(record_id, str) or not record_id or record_id in ids:
            raise ConfirmationScoreError(f"{name} has duplicate or invalid record IDs")
        if not isinstance(correct, int) or not isinstance(scored, int) or scored <= 0 or correct < 0 or correct > scored:
            raise ConfirmationScoreError(f"{name} has invalid correct/scored counts")
        if not isinstance(accuracy, (float, int)) or not math.isfinite(float(accuracy)):
            raise ConfirmationScoreError(f"{name} has non-finite token accuracy")
        if not math.isclose(float(accuracy), correct / scored, rel_tol=0.0, abs_tol=1e-12):
            raise ConfirmationScoreError(f"{name} token accuracy does not match counts")
        ids.add(record_id)
        result.append(
            {
                "record_id": record_id,
                "correct_tokens": correct,
                "scored_tokens": scored,
                "token_accuracy": float(accuracy),
                "exact_record": bool(row.get("exact_record")),
            }
        )
    return result


def paired_token_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    tail_alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster-bootstrap micro accuracy by resampling complete source records.

    The same ``seed`` and record ordering give every comparison the same
    source-resample schedule.  ``tail_alpha`` additionally records the upper
    quantile used for practical-benefit exclusion; the central 95% interval
    remains descriptive.
    """

    left_rows = _valid_record_rows(left, name="left paired records")
    right_rows = _valid_record_rows(right, name="right paired records")
    if [row["record_id"] for row in left_rows] != [row["record_id"] for row in right_rows]:
        raise ConfirmationScoreError("paired bootstrap record IDs changed")
    if draws <= 0:
        raise ConfirmationScoreError("bootstrap draws must be positive")
    if not 0.0 < tail_alpha < 1.0:
        raise ConfirmationScoreError("bootstrap tail alpha must be in (0,1)")
    left_correct = np.asarray([row["correct_tokens"] for row in left_rows], dtype=np.float64)
    left_scored = np.asarray([row["scored_tokens"] for row in left_rows], dtype=np.float64)
    right_correct = np.asarray([row["correct_tokens"] for row in right_rows], dtype=np.float64)
    right_scored = np.asarray([row["scored_tokens"] for row in right_rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left_rows), size=(draws, len(left_rows)))
    left_draw = left_correct[indices].sum(axis=1) / left_scored[indices].sum(axis=1)
    right_draw = right_correct[indices].sum(axis=1) / right_scored[indices].sum(axis=1)
    delta_draw = left_draw - right_draw
    left_estimate = float(left_correct.sum() / left_scored.sum())
    right_estimate = float(right_correct.sum() / right_scored.sum())
    return {
        "unit": "paired source record; micro correct/scored ratio per draw",
        "records": len(left_rows),
        "draws": int(draws),
        "seed": int(seed),
        "left_estimate": left_estimate,
        "right_estimate": right_estimate,
        "delta_estimate": left_estimate - right_estimate,
        "delta_ci95_percentile": [
            float(np.quantile(delta_draw, 0.025)),
            float(np.quantile(delta_draw, 0.975)),
        ],
        "upper_tail_alpha": float(tail_alpha),
        "delta_upper_bound": float(np.quantile(delta_draw, 1.0 - tail_alpha)),
        "left_ci95_percentile": [
            float(np.quantile(left_draw, 0.025)),
            float(np.quantile(left_draw, 0.975)),
        ],
        "right_ci95_percentile": [
            float(np.quantile(right_draw, 0.025)),
            float(np.quantile(right_draw, 0.975)),
        ],
    }


def _binomial_cdf(n: int, successes: int, probability: float) -> float:
    """Stable lower-tail binomial CDF for the small exact panel bound."""

    if n < 0 or successes < 0 or successes > n:
        raise ConfirmationScoreError("invalid binomial CDF counts")
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if successes >= n else 0.0
    log_probability = math.log(probability)
    log_failure = math.log1p(-probability)
    terms: list[float] = []
    for k in range(successes + 1):
        log_term = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * log_probability
            + (n - k) * log_failure
        )
        terms.append(log_term)
    maximum = max(terms)
    return min(1.0, max(0.0, math.exp(maximum) * math.fsum(math.exp(value - maximum) for value in terms)))


def _cp_upper_rate(*, records: int, successes: int, tail_alpha: float) -> float:
    """One-sided Clopper-Pearson upper endpoint for a binomial rate."""

    if records <= 0 or successes < 0 or successes > records:
        raise ConfirmationScoreError("invalid binomial counts")
    if not 0.0 < tail_alpha < 1.0:
        raise ConfirmationScoreError("binomial tail alpha must be in (0,1)")
    if successes == records:
        return 1.0
    if successes == 0:
        return 1.0 - tail_alpha ** (1.0 / records)
    low, high = successes / records, 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(records, successes, midpoint) > tail_alpha:
            low = midpoint
        else:
            high = midpoint
    return high


def _cp_lower_rate(*, records: int, successes: int, tail_alpha: float) -> float:
    """One-sided Clopper-Pearson lower endpoint for a binomial rate."""

    if records <= 0 or successes < 0 or successes > records:
        raise ConfirmationScoreError("invalid binomial counts")
    if not 0.0 < tail_alpha < 1.0:
        raise ConfirmationScoreError("binomial tail alpha must be in (0,1)")
    if successes == 0:
        return 0.0
    if successes == records:
        return tail_alpha ** (1.0 / records)
    # P[X >= successes | p] = alpha is equivalent to the upper endpoint for
    # the failure rate after reflecting successes and failures.
    return 1.0 - _cp_upper_rate(
        records=records,
        successes=records - successes,
        tail_alpha=tail_alpha,
    )


def exact_beneficial_discordance_bound(
    *,
    beneficial: int,
    harmful: int,
    records: int,
    confidence: float = EXACT_CONFIDENCE,
    practical_margin_pp: float | None = None,
) -> dict[str, Any]:
    """Return a conservative one-sided exact upper bound on beneficial rate.

    ``beneficial`` counts records exact under the method and not exact under
    the baseline.  The denominator is all paired records, while harmful
    discordances are ignored when bounding the beneficial rate; this is
    intentionally conservative.  In particular, zero discordances produce a
    positive finite-sample upper bound rather than an asserted zero effect.
    """

    if records <= 0 or beneficial < 0 or harmful < 0 or beneficial + harmful > records:
        raise ConfirmationScoreError("invalid exact-discordance counts")
    if not 0.0 < confidence < 1.0:
        raise ConfirmationScoreError("confidence must be in (0,1)")
    alpha = 1.0 - confidence
    upper = _cp_upper_rate(records=records, successes=beneficial, tail_alpha=alpha)
    beneficial_rate = beneficial / records
    harmful_rate = harmful / records
    net_rate = (beneficial - harmful) / records
    result: dict[str, Any] = {
        "unit": "paired source record",
        "records": int(records),
        "beneficial_discordances": int(beneficial),
        "harmful_discordances": int(harmful),
        "discordances": int(beneficial + harmful),
        "beneficial_rate": beneficial_rate,
        "harmful_rate": harmful_rate,
        "net_exact_rate_delta": net_rate,
        "confidence": confidence,
        "method": "one-sided Clopper-Pearson upper bound on beneficial rate; harmful discordances treated as zero benefit",
        "beneficial_upper_rate": float(upper),
        "beneficial_upper_pp": float(100.0 * upper),
        "zero_discordance_is_not_no_effect": bool(beneficial + harmful == 0),
        "interpretation": (
            "zero discordances still permit a positive finite-sample benefit; this bound does not establish equivalence"
            if beneficial + harmful == 0
            else "upper bound concerns beneficial discordance rate and does not erase harmful regressions"
        ),
    }
    if practical_margin_pp is not None:
        if not math.isfinite(float(practical_margin_pp)) or practical_margin_pp < 0:
            raise ConfirmationScoreError("practical margin must be finite and non-negative")
        result["practical_margin_pp"] = float(practical_margin_pp)
        result["upper_bound_below_practical_margin"] = float(100.0 * upper) <= float(practical_margin_pp)
    return result


def exact_net_benefit_bound(
    *,
    beneficial: int,
    harmful: int,
    records: int,
    tail_alpha: float = EXACT_FAMILY_TAIL_ALPHA,
    practical_margin_pp: float | None = None,
) -> dict[str, Any]:
    """Bound paired exact-record improvement with separate gain/loss tails.

    The interval is conservative: it upper-bounds the gain discordance rate
    and lower-bounds the loss discordance rate independently, then subtracts
    the latter from the former.  ``tail_alpha`` is set to 0.05/32 for the
    declared family of two primary contrasts × two domains × two targets ×
    token/exact endpoints.
    """

    if records <= 0 or beneficial < 0 or harmful < 0 or beneficial + harmful > records:
        raise ConfirmationScoreError("invalid exact-discordance counts")
    if not 0.0 < tail_alpha < 1.0:
        raise ConfirmationScoreError("exact bound tail alpha must be in (0,1)")
    gain_upper = _cp_upper_rate(
        records=records,
        successes=beneficial,
        tail_alpha=tail_alpha,
    )
    loss_lower = _cp_lower_rate(
        records=records,
        successes=harmful,
        tail_alpha=tail_alpha,
    )
    gain_rate = beneficial / records
    loss_rate = harmful / records
    net_rate = gain_rate - loss_rate
    result: dict[str, Any] = {
        "unit": "paired source record",
        "records": int(records),
        "beneficial_discordances": int(beneficial),
        "harmful_discordances": int(harmful),
        "discordances": int(beneficial + harmful),
        "beneficial_rate": gain_rate,
        "harmful_rate": loss_rate,
        "net_exact_rate_delta": net_rate,
        "tail_alpha_each": float(tail_alpha),
        # Two tails belong to one endpoint.  Keep that allocation separate
        # from the declared 0.05 family across the 16 practical endpoints.
        "endpoint_pair_alpha": float(2.0 * tail_alpha),
        "familywise_alpha": (
            0.05 if math.isclose(tail_alpha, EXACT_FAMILY_TAIL_ALPHA, rel_tol=0.0, abs_tol=1e-15) else None
        ),
        "method": "U(p_gain) - L(p_loss), one-sided Clopper-Pearson tails; zero gains retain a positive upper bound",
        "gain_upper_rate": float(gain_upper),
        "loss_lower_rate": float(loss_lower),
        "net_upper_rate": float(gain_upper - loss_lower),
        "net_lower_rate": float(
            _cp_lower_rate(records=records, successes=beneficial, tail_alpha=tail_alpha)
            - _cp_upper_rate(records=records, successes=harmful, tail_alpha=tail_alpha)
        ),
        "gain_upper_pp": float(100.0 * gain_upper),
        "loss_lower_pp": float(100.0 * loss_lower),
        "net_upper_pp": float(100.0 * (gain_upper - loss_lower)),
        "zero_discordance_is_not_no_effect": bool(beneficial + harmful == 0),
    }
    if practical_margin_pp is not None:
        if not math.isfinite(float(practical_margin_pp)) or practical_margin_pp < 0:
            raise ConfirmationScoreError("practical margin must be finite and non-negative")
        result["practical_margin_pp"] = float(practical_margin_pp)
        result["net_upper_below_practical_margin"] = result["net_upper_pp"] <= float(practical_margin_pp)
    return result


def paired_method_comparison(
    baseline: Sequence[Mapping[str, Any]],
    method: Sequence[Mapping[str, Any]],
    *,
    baseline_method_id: str,
    method_id: str,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    token_tail_alpha: float = 0.05,
    exact_tail_alpha: float = EXACT_FAMILY_TAIL_ALPHA,
    exact_practical_margin_pp: float | None = None,
) -> dict[str, Any]:
    """Compare two methods on the same source records with gains/regressions."""

    pseudo_methods = {"public_base", "public_lora_2601"}
    if (
        baseline_method_id not in METHOD_SPEC_BY_ID
        and baseline_method_id not in pseudo_methods
    ) or (
        method_id not in METHOD_SPEC_BY_ID
        and method_id not in pseudo_methods
    ):
        raise ConfirmationScoreError("unknown method in paired comparison")
    base_rows = _valid_record_rows(baseline, name="baseline paired records")
    method_rows = _valid_record_rows(method, name="method paired records")
    if [row["record_id"] for row in base_rows] != [row["record_id"] for row in method_rows]:
        raise ConfirmationScoreError("paired method record IDs changed")
    rows: list[dict[str, Any]] = []
    beneficial = 0
    harmful = 0
    gains = losses = ties = 0
    for base_row, method_row in zip(base_rows, method_rows):
        token_delta = method_row["token_accuracy"] - base_row["token_accuracy"]
        correct_delta = method_row["correct_tokens"] - base_row["correct_tokens"]
        if correct_delta > 0:
            gains += 1
        elif correct_delta < 0:
            losses += 1
        else:
            ties += 1
        base_exact = bool(base_row["exact_record"])
        method_exact = bool(method_row["exact_record"])
        is_beneficial = method_exact and not base_exact
        is_harmful = base_exact and not method_exact
        beneficial += int(is_beneficial)
        harmful += int(is_harmful)
        rows.append(
            {
                "record_id": base_row["record_id"],
                "baseline": base_row,
                "method": method_row,
                "token_accuracy_delta": float(token_delta),
                "correct_tokens_delta": int(correct_delta),
                "beneficial_exact": is_beneficial,
                "harmful_exact": is_harmful,
            }
        )
    base_correct = sum(row["correct_tokens"] for row in base_rows)
    method_correct = sum(row["correct_tokens"] for row in method_rows)
    base_scored = sum(row["scored_tokens"] for row in base_rows)
    method_scored = sum(row["scored_tokens"] for row in method_rows)
    result: dict[str, Any] = {
        "baseline_method_id": baseline_method_id,
        "method_id": method_id,
        "records": len(rows),
        "baseline_correct_tokens": base_correct,
        "method_correct_tokens": method_correct,
        "total_correct_tokens_delta": method_correct - base_correct,
        "baseline_token_accuracy": base_correct / base_scored,
        "method_token_accuracy": method_correct / method_scored,
        "micro_token_accuracy_delta": method_correct / method_scored - base_correct / base_scored,
        "mean_record_token_accuracy_delta": float(
            np.mean([row["token_accuracy_delta"] for row in rows])
        ),
        "records_with_any_token_gain": gains,
        "records_with_any_token_loss": losses,
        "records_with_token_tie": ties,
        "baseline_exact_records": sum(bool(row["exact_record"]) for row in base_rows),
        "method_exact_records": sum(bool(row["exact_record"]) for row in method_rows),
        "exact_record_delta": sum(bool(row["exact_record"]) for row in method_rows)
        - sum(bool(row["exact_record"]) for row in base_rows),
        "gains_and_regressions": {
            "beneficial_exact_records": beneficial,
            "harmful_exact_records": harmful,
            "net_exact_discordance": beneficial - harmful,
        },
        "paired_record_differences": rows,
        "token_bootstrap": paired_token_bootstrap(
            method_rows,
            base_rows,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
            tail_alpha=token_tail_alpha,
        ),
        "exact_beneficial_bound": exact_beneficial_discordance_bound(
            beneficial=beneficial,
            harmful=harmful,
            records=len(rows),
            confidence=1.0 - exact_tail_alpha,
            practical_margin_pp=exact_practical_margin_pp,
        ),
        "exact_net_benefit_bound": exact_net_benefit_bound(
            beneficial=beneficial,
            harmful=harmful,
            records=len(rows),
            tail_alpha=exact_tail_alpha,
            practical_margin_pp=exact_practical_margin_pp,
        ),
    }
    return result


def _cell_from_mapping(value: Mapping[str, Any], *, cell_id: str) -> CellInput:
    style, condition = cell_id.split("__", 1)
    ids = value.get("record_ids")
    if not isinstance(ids, (list, tuple)):
        raise ConfirmationScoreError(f"{cell_id} has no record_ids")
    return CellInput(
        style=style,
        condition=condition,
        record_ids=tuple(ids),
        truth=_tensor(value.get("truth"), dtype=torch.long, description=f"{cell_id} truth"),
        attention_mask=_tensor(value.get("attention_mask"), dtype=torch.bool, description=f"{cell_id} mask"),
        position_ids=_tensor(value.get("position_ids"), dtype=torch.long, description=f"{cell_id} positions"),
    )


_POSITIONWISE_SELECTION_SUFFIXES = (
    "joint_full_affine",
    "affine_trained_diagonal_attention128",
)
_DISTRIBUTION_ALIASES = {
    "original": "original",
    "enriched": "enriched",
    "original_like_alpaca_v1": "original",
    "coverage_mix_v1": "enriched",
}
_SELECTION_METHOD_ALIASES = {
    "affine": "joint_full_affine",
    "joint_affine": "joint_full_affine",
    "diagonal": "affine_trained_diagonal_attention128",
    "trained_diagonal": "affine_trained_diagonal_attention128",
}


def _selection_method_id(distribution: str, value: Any) -> str:
    """Normalize one frozen public-selection method and reject context arms."""

    if isinstance(value, Mapping):
        value = value.get("selected_method_id", value.get("selected_method"))
    if not isinstance(value, str):
        raise ConfirmationScoreError(
            f"public-validation selection is missing for {distribution}"
        )
    value = _SELECTION_METHOD_ALIASES.get(value, value)
    if value in _POSITIONWISE_SELECTION_SUFFIXES:
        value = f"{distribution}__{value}"
    expected = {f"{distribution}__{suffix}" for suffix in _POSITIONWISE_SELECTION_SUFFIXES}
    if value not in expected:
        raise ConfirmationScoreError(
            f"public-validation selection for {distribution} must choose affine or diagonal"
        )
    return value


def validate_public_validation_selection(
    selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate a frozen affine-versus-diagonal public selection.

    The selected arm is supplied by the public-validation stage and is bound
    before fresh truth is available.  This function intentionally never looks
    at final-cell metrics, so a scorer cannot silently reselect a baseline.
    """

    if not isinstance(selection, Mapping):
        raise ConfirmationScoreError(
            "frozen public-validation selection is required before scoring"
        )
    if selection.get("schema") not in (None, PUBLIC_SELECTION_SCHEMA):
        raise ConfirmationScoreError("public-validation selection schema changed")
    if selection.get("task_id") not in (None, TASK_ID):
        raise ConfirmationScoreError("public-validation selection task ID changed")
    if selection.get("status") != "FROZEN_PUBLIC_VALIDATION_SELECTION":
        raise ConfirmationScoreError("public-validation selection is not frozen")
    stage = selection.get("selection_stage")
    if stage is not None and stage != "public_validation_before_fresh_evaluation":
        raise ConfirmationScoreError("public-validation selection stage changed")
    if selection.get("truth_accessed") not in (None, False):
        raise ConfirmationScoreError("public-validation selection accessed truth")
    if selection.get("fresh_evaluation_accessed") not in (None, False):
        raise ConfirmationScoreError("public-validation selection was made after fresh evaluation")
    distributions = selection.get("distributions", selection.get("by_distribution"))
    if not isinstance(distributions, Mapping):
        raise ConfirmationScoreError(
            "public-validation selection must cover original and enriched distributions"
        )
    distribution_rows: dict[str, Mapping[str, Any]] = {}
    for raw_distribution, row in distributions.items():
        distribution = _DISTRIBUTION_ALIASES.get(raw_distribution)
        if distribution in ("original", "enriched"):
            if distribution in distribution_rows or not isinstance(row, Mapping):
                raise ConfirmationScoreError("public-validation distribution rows are duplicated or malformed")
            distribution_rows[distribution] = row
    if set(distribution_rows) != {"original", "enriched"}:
        raise ConfirmationScoreError(
            "public-validation selection must cover original and enriched distributions"
        )
    normalized: dict[str, str] = {}
    for distribution in ("original", "enriched"):
        row = distribution_rows[distribution]
        if not isinstance(row, Mapping):
            raise ConfirmationScoreError(
                f"public-validation selection row is malformed: {distribution}"
            )
        candidates = row.get("candidate_method_ids", row.get("candidates"))
        expected_candidates = {
            f"{distribution}__{suffix}" for suffix in _POSITIONWISE_SELECTION_SUFFIXES
        }
        if candidates is not None:
            if not isinstance(candidates, (list, tuple, set)):
                raise ConfirmationScoreError(
                    f"public-validation candidate arms changed: {distribution}"
                )
            try:
                observed_candidates = {
                    _selection_method_id(distribution, candidate) for candidate in candidates
                }
            except ConfirmationScoreError as exc:
                raise ConfirmationScoreError(
                    f"public-validation candidate arms changed: {distribution}"
                ) from exc
            if observed_candidates != expected_candidates:
                raise ConfirmationScoreError(
                    f"public-validation candidate arms changed: {distribution}"
                )
        normalized[distribution] = _selection_method_id(distribution, row)
    return {
        "schema": selection.get("schema", PUBLIC_SELECTION_SCHEMA),
        "task_id": selection.get("task_id", TASK_ID),
        "status": selection["status"],
        "selection_stage": stage or "public_validation_before_fresh_evaluation",
        "selected_method_ids": dict(normalized),
        "selection_plan_sha256": selection.get("selection_plan_sha256"),
        "validation_panel_sha256": selection.get("validation_panel_sha256"),
    }


def _declared_comparison_pairs(
    public_validation_selection: Mapping[str, Any] | None,
) -> tuple[tuple[str, str, str], ...]:
    """Return predeclared comparisons using the frozen selected baseline."""

    if isinstance(public_validation_selection, Mapping) and isinstance(
        public_validation_selection.get("selected_method_ids"), Mapping
    ) and "distributions" not in public_validation_selection and "by_distribution" not in public_validation_selection:
        selected = dict(public_validation_selection["selected_method_ids"])
        if set(selected) != {"original", "enriched"}:
            raise ConfirmationScoreError("normalized public-validation selection is incomplete")
        selected = {
            distribution: _selection_method_id(distribution, method_id)
            for distribution, method_id in selected.items()
        }
    else:
        selected = validate_public_validation_selection(public_validation_selection)["selected_method_ids"]
    pairs: list[tuple[str, str, str]] = []
    for distribution in ("original", "enriched"):
        best_positionwise = selected[distribution]
        causal = f"{distribution}__affine_causal_h_attention128"
        diagonal = f"{distribution}__affine_trained_diagonal_attention128"
        pairs.extend(
            (
                (f"{distribution}__causal_vs_best_positionwise", best_positionwise, causal),
                (f"{distribution}__causal_vs_diagonal", diagonal, causal),
            )
        )
        if distribution == "enriched":
            for state in (
                "joint_full_affine",
                "affine_causal_h_attention128",
                "affine_trained_diagonal_attention128",
            ):
                pairs.append(
                    (
                        f"coverage__{state}__enriched_vs_original",
                        f"original__{state}",
                        f"enriched__{state}",
                    )
                )
    pairs.extend(
        (
            (
                "anchor__historical_a1_vs_original_affine",
                "original__joint_full_affine",
                "historical_alpaca_a1",
            ),
            (
                "anchor__a2_vs_original_affine",
                "original__joint_full_affine",
                "frozen_a1_a2_k256",
            ),
        )
    )
    return tuple(pairs)



def _json_object_value(value: Any, *, description: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise PretruthGateError(f"{description} is absent")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PretruthGateError(f"{description} is invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise PretruthGateError(f"{description} must be an object")
    return dict(decoded)


def _asset_descriptor(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    description: str,
    allow_external: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Rehash one TRR4-style path/bytes/SHA descriptor.

    This is intentionally the same path and byte binding used by the proven
    TRR4 footing gate.  Extra role metadata is retained by callers, while the
    three integrity fields are compared to the current file record.
    """

    if not isinstance(descriptor, Mapping):
        raise PretruthGateError(f"{description} descriptor is absent")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise PretruthGateError(f"{description} path is absent")
    try:
        supplied = Path(raw_path).expanduser()
        if supplied.is_absolute():
            if supplied.is_symlink():
                raise PretruthGateError(f"{description} must not be a symbolic link")
            if not allow_external:
                try:
                    supplied.resolve().relative_to(root.resolve())
                except ValueError as exc:
                    raise PretruthGateError(
                        f"{description} is outside the repository root"
                    ) from exc
            path = supplied.resolve()
            current = external_file_record(path)
        else:
            candidate = PurePosixPath(raw_path)
            if (
                candidate.is_absolute()
                or not candidate.parts
                or any(part in ("", ".", "..") for part in candidate.parts)
                or candidate.as_posix() != raw_path
            ):
                raise PretruthGateError(f"{description} path is unsafe: {raw_path}")
            unresolved = root.resolve() / raw_path
            if unresolved.is_symlink():
                raise PretruthGateError(f"{description} must not be a symbolic link")
            path = unresolved.resolve()
            path.relative_to(root.resolve())
            current = file_record(path, repository_root=root.resolve())
    except PretruthGateError:
        raise
    except (FootingError, OSError, ValueError) as exc:
        raise PretruthGateError(f"{description} is unavailable: {raw_path}") from exc
    for key in ("path", "bytes", "sha256"):
        expected = descriptor.get(key)
        if key == "path" and supplied.is_absolute():
            matches = str(path) == str(expected) or str(path) == str(Path(str(expected)).expanduser().resolve())
        else:
            matches = expected == current[key]
        if not matches:
            raise PretruthGateError(f"{description} hash or path binding changed: {path}")
    if not isinstance(descriptor.get("bytes"), int) or descriptor["bytes"] < 0:
        raise PretruthGateError(f"{description} byte count is invalid")
    if not isinstance(descriptor.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is None:
        raise PretruthGateError(f"{description} SHA-256 binding is invalid")
    return path, dict(current)


def _json_file_bound(
    path: Path,
    expected: Mapping[str, Any],
    *,
    root: Path,
    description: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = expected.get("file") if isinstance(expected.get("file"), Mapping) else expected
    actual_path, record = _asset_descriptor(
        descriptor,
        root=root,
        description=description,
    )
    try:
        value = json.loads(actual_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PretruthGateError(f"{description} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise PretruthGateError(f"{description} must contain a JSON object")
    if dict(value) != dict(expected):
        raise PretruthGateError(f"{description} content binding changed")
    return dict(value), record


def _prediction_asset_from_descriptor(
    descriptor: Mapping[str, Any], *, description: str
) -> dict[str, Any]:
    for key in ("prediction_artifact", "artifact", "prediction_file"):
        value = descriptor.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("path"), str):
            return dict(value)
    if isinstance(descriptor.get("prediction_path"), str):
        result = {
            "path": descriptor["prediction_path"],
            "bytes": descriptor.get("prediction_bytes"),
            "sha256": descriptor.get("prediction_sha256"),
        }
        return result
    raise PretruthGateError(
        f"{description} has no bound prediction artifact (path/bytes/sha256)"
    )


def _observation_asset_from_descriptor(
    descriptor: Mapping[str, Any], *, description: str
) -> dict[str, Any]:
    for key in ("observation", "observation_artifact", "artifact", "file"):
        value = descriptor.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("path"), str):
            return dict(value)
    if isinstance(descriptor.get("path"), str):
        return dict(descriptor)
    raise PretruthGateError(f"{description} has no bound observation artifact")


def _tensor_from_public_geometry(value: Any, *, dtype: torch.dtype, description: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=dtype).contiguous().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PretruthGateError(f"{description} is malformed") from exc
    if tuple(tensor.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
        raise PretruthGateError(f"{description} geometry changed")
    return tensor


def _validate_public_geometry(
    *,
    cell_id: str,
    mask_value: Any,
    position_value: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _tensor_from_public_geometry(
        mask_value,
        dtype=torch.long,
        description=f"{cell_id} observation mask",
    )
    positions = _tensor_from_public_geometry(
        position_value,
        dtype=torch.long,
        description=f"{cell_id} observation positions",
    )
    if mask.lt(0).any().item() or mask.gt(1).any().item():
        raise PretruthGateError(f"{cell_id} observation mask is not binary")
    bool_mask = mask.to(torch.bool)
    if not bool_mask[:, 0].all().item() or (bool_mask[:, 1:] > bool_mask[:, :-1]).any().item():
        raise PretruthGateError(f"{cell_id} observation mask is not BOS/right-padded")
    expected_positions = bool_mask.to(torch.long).cumsum(1).sub(1).clamp_min(0)
    if not torch.equal(positions, expected_positions):
        raise PretruthGateError(f"{cell_id} observation positions disagree with mask")
    return bool_mask, positions


def _panel_cell(panel: Mapping[str, Any], cell_id: str) -> Mapping[str, Any] | None:
    cells = panel.get("cells")
    if isinstance(cells, Mapping):
        value = cells.get(cell_id)
        return value if isinstance(value, Mapping) else None
    if isinstance(cells, list):
        for value in cells:
            if isinstance(value, Mapping) and value.get("id", value.get("cell_id")) == cell_id:
                return value
    return None


def _geometry_values(
    panel: Mapping[str, Any],
    cell_id: str,
    descriptor: Mapping[str, Any],
) -> tuple[Any, Any]:
    candidates: list[Mapping[str, Any]] = [descriptor]
    cell = _panel_cell(panel, cell_id)
    if cell is not None:
        candidates.append(cell)
    for source in candidates:
        mask = source.get("attention_mask", source.get("mask"))
        positions = source.get("position_ids", source.get("positions"))
        if mask is not None and positions is not None:
            return mask, positions
        geometry = source.get("geometry")
        if isinstance(geometry, Mapping):
            mask = geometry.get("attention_mask", geometry.get("mask"))
            positions = geometry.get("position_ids", geometry.get("positions"))
            if mask is not None and positions is not None:
                return mask, positions
    raise PretruthGateError(
        f"{cell_id} has no public mask/position geometry bound to its observation"
    )


def _validate_observation_artifact(
    *,
    cell_id: str,
    descriptor: Mapping[str, Any],
    root: Path,
    mask: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, Any]:
    asset = _observation_asset_from_descriptor(descriptor, description=f"observation {cell_id}")
    path, record = _asset_descriptor(
        asset,
        root=root,
        description=f"observation {cell_id}",
        allow_external=True,
    )
    declared_shape = descriptor.get("shape", asset.get("shape"))
    expected_hidden = int(descriptor.get("hidden_size", asset.get("hidden_size", HIDDEN_SIZE)))
    expected_shape = (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, expected_hidden)
    try:
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if "activations" not in keys:
                    raise PretruthGateError(f"observation activations are absent: {cell_id}")
                shape = tuple(int(value) for value in handle.get_slice("activations").get_shape())
                if shape != expected_shape:
                    raise PretruthGateError(f"observation activation geometry changed: {cell_id}")
                if "attention_mask" in keys:
                    observed_mask = handle.get_tensor("attention_mask").to(torch.long)
                    if not torch.equal(observed_mask, mask.to(torch.long)):
                        raise PretruthGateError(f"observation mask binding changed: {cell_id}")
                if "position_ids" in keys:
                    observed_positions = handle.get_tensor("position_ids").to(torch.long)
                    if not torch.equal(observed_positions, positions):
                        raise PretruthGateError(f"observation position binding changed: {cell_id}")
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PretruthGateError(f"observation manifest is unreadable: {cell_id}") from exc
            if not isinstance(payload, Mapping):
                raise PretruthGateError(f"observation manifest is malformed: {cell_id}")
            shape = tuple(payload.get("shape", ()))
            if shape != expected_shape:
                raise PretruthGateError(f"observation manifest geometry changed: {cell_id}")
            if "attention_mask" in payload and torch.as_tensor(payload["attention_mask"], dtype=torch.long).shape == mask.shape:
                if not torch.equal(torch.as_tensor(payload["attention_mask"], dtype=torch.long), mask.to(torch.long)):
                    raise PretruthGateError(f"observation manifest mask changed: {cell_id}")
            if "position_ids" in payload and not torch.equal(torch.as_tensor(payload["position_ids"], dtype=torch.long), positions):
                raise PretruthGateError(f"observation manifest positions changed: {cell_id}")
        elif declared_shape is not None and tuple(declared_shape) != expected_shape:
            raise PretruthGateError(f"observation declared geometry changed: {cell_id}")
    except PretruthGateError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise PretruthGateError(f"observation artifact is unreadable: {cell_id}") from exc
    # Rehash after opening, as in the TRR4 executable gate.
    _, after = _asset_descriptor(
        asset,
        root=root,
        description=f"observation {cell_id}",
        allow_external=True,
    )
    if after != record:
        raise PretruthGateError(f"observation changed while validating: {cell_id}")
    return {
        "cell_id": cell_id,
        "path": record["path"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "shape": list(expected_shape),
    }


def _bound_method_assets(
    *,
    registration: Mapping[str, Any],
    method_id: str,
    root: Path,
    panel_record: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = registration.get("state_bindings")
    if not isinstance(bindings, Mapping) or method_id not in bindings:
        raise PretruthGateError(f"state/code binding is absent: {method_id}")
    entry = bindings[method_id]
    if not isinstance(entry, Mapping):
        raise PretruthGateError(f"state/code binding is malformed: {method_id}")
    binding = entry.get("binding") if isinstance(entry.get("binding"), Mapping) else entry
    binding = dict(binding)
    expected_commit = registration.get("code_commit")
    if isinstance(expected_commit, str) and binding.get("code_commit") not in (None, expected_commit):
        raise PretruthGateError(f"code commit binding changed: {method_id}")
    if isinstance(binding.get("panel"), Mapping):
        try:
            _asset_descriptor(binding["panel"], root=root, description=f"bound panel {method_id}")
        except PretruthGateError:
            raise
        if dict(binding["panel"]) != dict(panel_record):
            raise PretruthGateError(f"panel binding changed: {method_id}")
    # The full TRR4 binding is preferred when present.  It checks exact group
    # names and rehashes every state/code byte through the shared footing code.
    if all(key in binding for key in ("panel", "method_state", "code")):
        try:
            validate_binding(binding, binding, repository_root=root.resolve())
        except (FootingError, OSError, ValueError) as exc:
            raise PretruthGateError(f"state/code binding changed: {method_id}") from exc
        _validate_runtime_assets(binding, method_id=method_id)
        return binding

    state = binding.get("state")
    if not isinstance(state, Mapping):
        state_path = binding.get("state_path")
        state = {
            "path": state_path,
            "bytes": binding.get("state_bytes"),
            "sha256": binding.get("state_sha256"),
        }
    _asset_descriptor(state, root=root, description=f"state {method_id}")
    code = binding.get("code", binding.get("code_artifacts"))
    if code is None and isinstance(binding.get("code_path"), str):
        code = [{
            "path": binding["code_path"],
            "bytes": binding.get("code_bytes"),
            "sha256": binding.get("code_sha256"),
        }]
    if isinstance(code, Mapping):
        code = [code]
    if not isinstance(code, (list, tuple)) or not code:
        raise PretruthGateError(f"code binding is absent: {method_id}")
    for index, descriptor in enumerate(code):
        _asset_descriptor(descriptor, root=root, description=f"code {method_id}[{index}]")
    if binding.get("code_commit") is None and not isinstance(expected_commit, str):
        raise PretruthGateError(f"code commit binding is absent: {method_id}")
    _validate_runtime_assets(binding, method_id=method_id)
    return binding


def _validate_runtime_assets(
    binding: Mapping[str, Any],
    *,
    method_id: str,
) -> dict[str, dict[str, Any]]:
    """Rehash the shared E resource and A2-only public P0 assets.

    The TRR4 runtime contract uses absolute descriptors because these public
    resources may live in the shared external cache.  Standalone methods bind
    only the normalized public embedding table; requiring the P0 checkpoint or
    config for them would make the common gate depend on an unused comparator
    resource.
    """

    expected_roles = (
        (RUNTIME_EMBEDDING_ROLE, *RUNTIME_P0_ROLES)
        if method_id == A2_METHOD_ID
        else (RUNTIME_EMBEDDING_ROLE,)
    )
    assets = binding.get("runtime_assets")
    if not isinstance(assets, Mapping) or set(assets) != set(expected_roles):
        raise PretruthGateError(
            f"runtime assets for {method_id} must contain exactly {expected_roles!r}"
        )
    records: dict[str, dict[str, Any]] = {}
    for role in expected_roles:
        descriptor = assets.get(role)
        if not isinstance(descriptor, Mapping):
            raise PretruthGateError(f"runtime asset is malformed: {method_id}/{role}")
        raw_path = descriptor.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise PretruthGateError(
                f"runtime asset path must be absolute: {method_id}/{role}"
            )
        path = Path(raw_path).expanduser()
        if path.is_symlink() or not path.is_file():
            raise PretruthGateError(f"runtime asset is unavailable: {method_id}/{role}")
        try:
            current = external_file_record(path)
        except (FootingError, OSError, ValueError) as exc:
            raise PretruthGateError(
                f"runtime asset is unavailable: {method_id}/{role}"
            ) from exc
        if dict(descriptor) != current:
            raise PretruthGateError(
                f"runtime asset binding changed: {method_id}/{role}"
            )
        records[role] = current
    return records


def _prediction_tensor_from_artifact(
    *,
    path: Path,
    descriptor: Mapping[str, Any],
    cell_id: str,
    method_id: str,
    panel_sha256: str,
    selection_plan_sha256: str,
    observation_sha256: str,
    expected_binding: Mapping[str, Any],
    mask: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            required = {
                "schema", "task_id", "panel_sha256", "selection_plan_sha256",
                "observation_sha256", "cell_id", "style", "condition",
                "method_id", "geometry_json", "binding_json",
            }
            if not isinstance(metadata, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in metadata.items()
            ) or not required.issubset(metadata):
                raise PretruthGateError(f"prediction metadata is incomplete: {cell_id}/{method_id}")
            if metadata["schema"] != PREDICTION_SCHEMA or metadata["task_id"] != TASK_ID:
                raise PretruthGateError(f"prediction artifact identity changed: {cell_id}/{method_id}")
            if metadata["panel_sha256"] != panel_sha256 or metadata["selection_plan_sha256"] != selection_plan_sha256:
                raise PretruthGateError(f"prediction panel/selection binding changed: {cell_id}/{method_id}")
            if metadata["observation_sha256"] != observation_sha256:
                raise PretruthGateError(f"prediction observation binding changed: {cell_id}/{method_id}")
            style, condition = cell_id.split("__", 1)
            if metadata["cell_id"] != cell_id or metadata["style"] != style or metadata["condition"] != condition:
                raise PretruthGateError(f"prediction cell binding changed: {cell_id}/{method_id}")
            if metadata["method_id"] != method_id:
                raise PretruthGateError(f"prediction method binding changed: {cell_id}/{method_id}")
            expected_policy = CANDIDATE_POLICIES[method_id]
            if metadata.get("candidate_policy") != expected_policy:
                raise PretruthGateError(f"prediction candidate policy changed: {cell_id}/{method_id}")
            expected_candidate_output = (
                "omitted_after_decision" if expected_policy == "output_only" else "forbidden"
            )
            if metadata.get("candidate_output") != expected_candidate_output:
                raise PretruthGateError(f"prediction candidate output policy changed: {cell_id}/{method_id}")
            geometry = _json_object_value(metadata["geometry_json"], description="prediction geometry")
            if geometry != {
                "records": RECORDS_PER_DOMAIN,
                "sequence_tokens": SEQUENCE_TOKENS,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
            }:
                raise PretruthGateError(f"prediction geometry binding changed: {cell_id}/{method_id}")
            artifact_binding = _json_object_value(metadata["binding_json"], description="prediction binding")
            if dict(artifact_binding) != dict(expected_binding):
                raise PretruthGateError(f"prediction state/code binding changed: {cell_id}/{method_id}")
            keys = set(handle.keys())
            if keys != {"predictions"}:
                raise PretruthGateError(
                    f"prediction tensor fields changed or candidates persisted: {cell_id}/{method_id}"
                )
            predictions = handle.get_tensor("predictions").contiguous().cpu()
            if predictions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise PretruthGateError(f"prediction IDs are not integer: {cell_id}/{method_id}")
            if tuple(predictions.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
                raise PretruthGateError(f"prediction shape changed: {cell_id}/{method_id}")
            if predictions[:, 0].ne(BOS_TOKEN_ID).any().item():
                raise PretruthGateError(f"prediction BOS changed: {cell_id}/{method_id}")
            active = mask.to(torch.bool)
            scored = active.clone()
            scored[:, 0] = False
            if predictions[scored].lt(0).any().item() or predictions[scored].ge(VOCAB_SIZE).any().item():
                raise PretruthGateError(f"prediction ID range changed: {cell_id}/{method_id}")
            if predictions[~active].ne(INVALID_TOKEN_ID).any().item():
                raise PretruthGateError(f"prediction padding changed: {cell_id}/{method_id}")
            if positions.shape != mask.shape:
                raise PretruthGateError(f"prediction geometry mask changed: {cell_id}/{method_id}")
    except PretruthGateError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise PretruthGateError(f"prediction artifact is unreadable: {cell_id}/{method_id}") from exc
    return predictions.to(torch.long), {
        "cell_id": cell_id,
        "method_id": method_id,
        "path": str(path),
        "tensor_fields": ["predictions"],
        "shape": list(predictions.shape),
    }


def _frozen_receipt_metadata(
    *,
    receipt: Mapping[str, Any],
    root: Path,
    output_root: Path,
    panel_record: Mapping[str, Any],
    selection_plan_record: Mapping[str, Any],
    registration_record: Mapping[str, Any],
    selection: Mapping[str, Any],
    expected_code_commit: str | None = None,
) -> None:
    try:
        output_relative = output_root.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PretruthGateError("prediction output root is outside repository root") from exc
    if receipt.get("frozen_root") != output_relative:
        raise PretruthGateError("freeze receipt does not bind the requested prediction root")
    receipt_plan = receipt.get("plan")
    if not isinstance(receipt_plan, Mapping) or dict(receipt_plan) != dict(selection_plan_record):
        raise PretruthGateError("freeze receipt does not bind the requested selection plan")
    metadata = receipt.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PretruthGateError("freeze receipt metadata is absent")
    if metadata.get("task_id") != TASK_ID:
        raise PretruthGateError("freeze receipt task binding changed")
    checks = {
        "panel_sha256": panel_record["sha256"],
        "selection_plan_sha256": selection_plan_record["sha256"],
        "registration_sha256": registration_record["sha256"],
        "method_ids": list(METHOD_IDS),
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise PretruthGateError(f"freeze receipt {key} binding changed")
    if metadata.get("truth_opened") is not False:
        raise PretruthGateError("freeze receipt was created after truth opened")
    preregistration_commit = receipt.get("preregistration_commit")
    # The commit is the executable code binding used by every method state.
    # Registration validation below checks the file bytes; this catches a
    # receipt copied from a different checkout before any truth read.
    if preregistration_commit != str(metadata.get("code_commit", preregistration_commit)):
        raise PretruthGateError("freeze receipt code commit metadata is inconsistent")
    if expected_code_commit is not None and preregistration_commit != expected_code_commit:
        raise PretruthGateError("freeze receipt executable commit binding changed")
    recorded_selection = metadata.get("public_validation_selection")
    if recorded_selection is not None:
        if validate_public_validation_selection(recorded_selection) != validate_public_validation_selection(selection):
            raise PretruthGateError("freeze receipt public selection changed")


def validate_before_truth(
    *,
    panel: Mapping[str, Any],
    registration: Mapping[str, Any],
    prediction_descriptors: Mapping[tuple[str, str], Mapping[str, Any]],
    timing_descriptors: Mapping[tuple[str, str], Mapping[str, Any]],
    public_validation_selection: Mapping[str, Any] | None,
    repository_root: Path | str,
    receipt_path: Path | str,
    truth_path: Path | str,
    output_root: Path | str,
    panel_path: Path | str,
    registration_path: Path | str,
    selection_plan_path: Path | str,
    observation_descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run all executable public checks before a caller may invoke truth.

    The order follows the TRR4 gate: freeze receipt first, then bound panel,
    plan, registration, observations, state/code assets, and every prediction
    tensor.  The returned tensors are safe to score after the caller opens the
    private sidecar; this function itself never reads truth content.
    """

    normalized_selection = validate_public_validation_selection(public_validation_selection)
    root = Path(repository_root).expanduser().resolve()
    receipt_path = Path(receipt_path).expanduser().resolve()
    truth_path = Path(truth_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    panel_path = Path(panel_path).expanduser().resolve()
    registration_path = Path(registration_path).expanduser().resolve()
    selection_plan_path = Path(selection_plan_path).expanduser().resolve()
    try:
        receipt = require_truth_open_allowed(
            receipt_path=receipt_path,
            repository_root=root,
            truth_path=truth_path,
        )
    except (FreezeError, OSError, ValueError) as exc:
        raise PretruthGateError(f"freeze receipt rejected: {exc}") from exc
    try:
        panel_asset = file_record(panel_path, repository_root=root)
    except (FootingError, OSError, ValueError) as exc:
        raise PretruthGateError("confirmation panel file is unavailable") from exc
    panel_actual_path, panel_record = _asset_descriptor(
        panel_asset,
        root=root,
        description="confirmation panel",
    )
    try:
        panel_file = json.loads(panel_actual_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PretruthGateError("confirmation panel file is unreadable") from exc
    if not isinstance(panel_file, Mapping) or dict(panel_file) != dict(panel):
        raise PretruthGateError("confirmation panel content binding changed")
    _, panel_after_read = _asset_descriptor(
        panel_asset,
        root=root,
        description="confirmation panel",
    )
    if panel_after_read != panel_record:
        raise PretruthGateError("confirmation panel changed while validating")
    try:
        registration_asset = file_record(registration_path, repository_root=root)
    except (FootingError, OSError, ValueError) as exc:
        raise PretruthGateError("confirmation registration file is unavailable") from exc
    registration_actual_path, registration_record = _asset_descriptor(
        registration_asset,
        root=root,
        description="confirmation registration",
    )
    try:
        registration_file = json.loads(registration_actual_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PretruthGateError("confirmation registration file is unreadable") from exc
    if not isinstance(registration_file, Mapping) or dict(registration_file) != dict(registration):
        raise PretruthGateError("confirmation registration content binding changed")
    _, registration_after_read = _asset_descriptor(
        registration_asset,
        root=root,
        description="confirmation registration",
    )
    if registration_after_read != registration_record:
        raise PretruthGateError("confirmation registration changed while validating")
    try:
        plan_asset = file_record(selection_plan_path, repository_root=root)
    except (FootingError, OSError, ValueError) as exc:
        raise PretruthGateError("public selection plan is unavailable") from exc
    plan_actual_path, selection_plan_record = _asset_descriptor(
        plan_asset,
        root=root,
        description="public selection plan",
    )
    try:
        plan_file = json.loads(plan_actual_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PretruthGateError("public selection plan is unreadable") from exc
    _, selection_plan_after_read = _asset_descriptor(
        plan_asset,
        root=root,
        description="public selection plan",
    )
    if selection_plan_after_read != selection_plan_record:
        raise PretruthGateError("public selection plan changed while validating")
    if isinstance(plan_file, Mapping) and isinstance(plan_file.get("public_validation_selection"), Mapping):
        plan_selection = validate_public_validation_selection(plan_file["public_validation_selection"])
        if plan_selection != normalized_selection:
            raise PretruthGateError("public-validation selection plan binding changed")
    if normalized_selection.get("selection_plan_sha256") not in (None, selection_plan_record["sha256"]):
        raise PretruthGateError("public-validation selection plan hash changed")
    _frozen_receipt_metadata(
        receipt=receipt,
        root=root,
        output_root=output_root,
        panel_record=panel_record,
        selection_plan_record=selection_plan_record,
        registration_record=registration_record,
        selection=public_validation_selection or normalized_selection,
        expected_code_commit=(
            str(registration.get("code_commit"))
            if isinstance(registration.get("code_commit"), str)
            else None
        ),
    )
    if set(observation_descriptors) != set(EXPECTED_CELL_IDS):
        raise PretruthGateError("observation bindings are incomplete")
    observation_records: dict[str, dict[str, Any]] = {}
    geometry: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for cell_id in EXPECTED_CELL_IDS:
        descriptor = observation_descriptors[cell_id]
        mask_value, positions_value = _geometry_values(panel, cell_id, descriptor)
        mask, positions = _validate_public_geometry(
            cell_id=cell_id,
            mask_value=mask_value,
            position_value=positions_value,
        )
        observation_records[cell_id] = _validate_observation_artifact(
            cell_id=cell_id,
            descriptor=descriptor,
            root=root,
            mask=mask,
            positions=positions,
        )
        geometry[cell_id] = (mask, positions)
    method_bindings: dict[str, dict[str, Any]] = {}
    for method_id in METHOD_IDS:
        method_bindings[method_id] = _bound_method_assets(
            registration=registration,
            method_id=method_id,
            root=root,
            panel_record=panel_record,
        )
    expected_keys = {(cell_id, method_id) for cell_id in EXPECTED_CELL_IDS for method_id in METHOD_IDS}
    if set(prediction_descriptors) != expected_keys:
        raise PretruthGateError("prediction descriptor matrix is incomplete")
    if set(timing_descriptors) != expected_keys:
        raise PretruthGateError("timing descriptor matrix is incomplete")
    expected_paths: set[Path] = set()
    prediction_tensors: dict[tuple[str, str], torch.Tensor] = {}
    validated_artifacts: list[dict[str, Any]] = []
    for cell_id, method_id in sorted(expected_keys):
        timing = timing_descriptors[(cell_id, method_id)]
        if timing.get("warmup_runs_per_record") != 1 or timing.get("measured_runs_per_record") != 1 or timing.get("warmup_output_exact_match_measured") is not True:
            raise PretruthGateError(f"timing contract changed: {cell_id}/{method_id}")
        descriptor = prediction_descriptors[(cell_id, method_id)]
        asset = _prediction_asset_from_descriptor(
            descriptor,
            description=f"prediction {cell_id}/{method_id}",
        )
        path, record = _asset_descriptor(
            asset,
            root=root,
            description=f"prediction {cell_id}/{method_id}",
        )
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise PretruthGateError(f"prediction escaped frozen root: {path}") from exc
        expected_paths.add(path)
        mask, positions = geometry[cell_id]
        prediction, summary = _prediction_tensor_from_artifact(
            path=path,
            descriptor=descriptor,
            cell_id=cell_id,
            method_id=method_id,
            panel_sha256=panel_record["sha256"],
            selection_plan_sha256=selection_plan_record["sha256"],
            observation_sha256=observation_records[cell_id]["sha256"],
            expected_binding=method_bindings[method_id],
            mask=mask,
            positions=positions,
        )
        prediction_after = _asset_descriptor(
            asset,
            root=root,
            description=f"prediction {cell_id}/{method_id}",
        )[1]
        if prediction_after != record:
            raise PretruthGateError(f"prediction changed while validating: {cell_id}/{method_id}")
        prediction_tensors[(cell_id, method_id)] = prediction
        summary.update({"bytes": record["bytes"], "sha256": record["sha256"]})
        validated_artifacts.append(summary)
    actual_paths = {
        path.resolve()
        for path in output_root.rglob("*.safetensors")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise PretruthGateError(
            f"prediction artifact set is incomplete: missing={missing!r} extra={extra!r}"
        )
    # Close the small state/code TOCTOU window between binding validation and
    # the last prediction read, matching the TRR4 gate's rehash-before-truth
    # discipline.
    for method_id in METHOD_IDS:
        _bound_method_assets(
            registration=registration,
            method_id=method_id,
            root=root,
            panel_record=panel_record,
        )
    return {
        "task_id": TASK_ID,
        "status": "COMPLETE_PUBLIC_MATRIX_VERIFIED_NO_TRUTH_OPENED",
        "verified_before_truth": True,
        "receipt": dict(receipt),
        "panel": panel_record,
        "registration": registration_record,
        "selection_plan": selection_plan_record,
        "public_validation_selection": normalized_selection,
        "observations": observation_records,
        "method_bindings": {key: dict(value) for key, value in method_bindings.items()},
        "prediction_artifacts": validated_artifacts,
        "prediction_artifact_count": len(validated_artifacts),
        "timing_receipt_count": len(expected_keys),
        "prediction_tensors": prediction_tensors,
    }

def score_matrix(
    *,
    panel: Mapping[str, Any],
    registration: Mapping[str, Any],
    prediction_descriptors: Mapping[tuple[str, str], Mapping[str, Any]],
    cell_inputs: Mapping[str, Mapping[str, Any]],
    timing_descriptors: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    frequency_counts: Mapping[str, Any] | None = None,
    frequency_reference_ids: Mapping[str, str] | None = None,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    exact_practical_margin_pp: float | None = None,
    public_validation_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score all cells after validating the complete public matrix."""

    # Baseline selection is a frozen public-validation input, never inferred
    # from final-cell truth.  Validate it before touching any cell input.
    normalized_selection = validate_public_validation_selection(public_validation_selection)

    # This is deliberately the first operation.  ``cell_inputs`` contains
    # truth tensors but callers must construct it only after their sidecar
    # loader returns; ``score_with_truth_loader`` below enforces that order.
    gate = validate_complete_public_matrix(
        panel,
        registration,
        prediction_descriptors,
        timing_descriptors=timing_descriptors,
    )
    if set(cell_inputs) != set(EXPECTED_CELL_IDS):
        raise ConfirmationScoreError("truth/geometry input does not contain exactly four cells")
    cells = {cell_id: _cell_from_mapping(cell_inputs[cell_id], cell_id=cell_id) for cell_id in EXPECTED_CELL_IDS}
    rows: dict[str, Any] = {}
    per_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for cell_id in EXPECTED_CELL_IDS:
        cell = cells[cell_id]
        for method_id in METHOD_IDS:
            descriptor = prediction_descriptors[(cell_id, method_id)]
            if "predictions" not in descriptor:
                raise ConfirmationScoreError(f"prediction tensor missing: {cell_id}/{method_id}")
            spec = METHOD_SPEC_BY_ID[method_id]
            reference_key = spec["distribution"] if spec["distribution"] in FIT_BANKS else "original"
            counts = frequency_counts.get(reference_key) if frequency_counts is not None else None
            reference_id = frequency_reference_ids.get(reference_key) if frequency_reference_ids else reference_key
            scored = score_cell(
                predictions=descriptor["predictions"],
                cell=cell,
                method_id=method_id,
                frequency_counts=counts,
                frequency_reference_id=reference_id,
            )
            # Keep both public fitting-frequency references for every
            # contender.  The method's own fitting distribution remains the
            # headline reference, while the alternate map exposes label
            # migration under one unchanged bin definition.
            frequency_reference_rows: dict[str, Any] = {}
            if frequency_counts is not None:
                for reference_key in ("original", "enriched"):
                    reference_counts = frequency_counts.get(reference_key)
                    if reference_counts is None:
                        continue
                    reference_result = score_cell(
                        predictions=descriptor["predictions"],
                        cell=cell,
                        method_id=method_id,
                        frequency_counts=reference_counts,
                        frequency_reference_id=(
                            frequency_reference_ids.get(reference_key)
                            if frequency_reference_ids
                            else reference_key
                        ),
                    )
                    frequency_reference_rows[reference_key] = {
                        "frequency_reference_id": reference_result["frequency_reference_id"],
                        "frequency_bins": reference_result["frequency_bins"],
                        "joint_frequency_position": reference_result["joint_frequency_position"],
                    }
            scored["frequency_references"] = frequency_reference_rows
            scored["prediction_descriptor"] = {
                key: value
                for key, value in descriptor.items()
                if key != "predictions"
            }
            rows[f"{cell_id}__{method_id}"] = scored
            per_key[(cell.style, cell.condition, method_id)] = scored["per_record"]

    paired_targets: dict[str, Any] = {}
    for style in STYLE_ORDER:
        for method_id in METHOD_IDS:
            base = per_key[(style, "public_base", method_id)]
            shifted = per_key[(style, "public_lora_2601", method_id)]
            paired_targets[f"{style}__{method_id}"] = paired_method_comparison(
                base,
                shifted,
                baseline_method_id="public_base",
                method_id="public_lora_2601",
                bootstrap_draws=bootstrap_draws,
                bootstrap_seed=bootstrap_seed,
                token_tail_alpha=0.05,
                exact_tail_alpha=0.05,
                exact_practical_margin_pp=exact_practical_margin_pp,
            )

    method_comparisons: dict[str, Any] = {}
    primary_labels = {
        "enriched__causal_vs_best_positionwise",
        "enriched__causal_vs_diagonal",
    }
    for label, baseline_id, method_id in _declared_comparison_pairs(normalized_selection):
        for style in STYLE_ORDER:
            for condition in CONDITION_ORDER:
                base = per_key[(style, condition, baseline_id)]
                method = per_key[(style, condition, method_id)]
                method_comparisons[f"{style}__{condition}__{label}"] = paired_method_comparison(
                    base,
                    method,
                    baseline_method_id=baseline_id,
                    method_id=method_id,
                    bootstrap_draws=bootstrap_draws,
                    bootstrap_seed=bootstrap_seed,
                    token_tail_alpha=(TOKEN_FAMILY_TAIL_ALPHA if label in primary_labels else 0.05),
                    exact_tail_alpha=(EXACT_FAMILY_TAIL_ALPHA if label in primary_labels else 0.05),
                    exact_practical_margin_pp=exact_practical_margin_pp,
                )

    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "FRESH_CONFIRMATION_SCORED_AFTER_COMPLETE_PUBLIC_GATE",
        "claim_scope": "TRR-0005 exploratory four-cell matrix; no pooled headline and no replacement claim",
        "truth_gate": {
            **gate,
            "verified_before_truth": True,
            "truth_opened_after_gate": True,
        },
        "public_validation_selection": normalized_selection,
        "matrix": {
            "domains": list(STYLE_ORDER),
            "conditions": list(CONDITION_ORDER),
            "cells": list(EXPECTED_CELL_IDS),
            "method_ids": list(METHOD_IDS),
            "cell_count": len(EXPECTED_CELL_IDS),
            "method_count": len(METHOD_IDS),
            "pooled_headline": False,
        },
        "bootstrap": {
            "seed": bootstrap_seed,
            "draws": bootstrap_draws,
            "unit": "complete paired source record; micro correct/scored ratio",
        },
        "paired_target_comparisons": paired_targets,
        "method_comparisons": method_comparisons,
        "cells_results": rows,
        "diagnostic_contract": {
            "frequency_bins": [name for name, _lo, _hi in FREQUENCY_BINS],
            "position_bins": [name for name, _lo, _hi in POSITION_BINS],
            "joint_frequency_position_domain": True,
            "frequency_references_reported": ["original", "enriched"],
            "exact_uncertainty": "one-sided finite-sample beneficial-discordance upper bound; zero discordance is not equivalence",
            "primary_contrasts": [
                "enriched__causal_vs_diagonal",
                "enriched__causal_vs_best_positionwise",
            ],
            "familywise_token_tail_alpha": TOKEN_FAMILY_TAIL_ALPHA,
            "familywise_exact_tail_alpha_each": EXACT_FAMILY_TAIL_ALPHA,
        },
        "runtime_contract": dict(TIMING_CONTRACT),
    }


def score_with_truth_loader(
    *,
    panel: Mapping[str, Any],
    registration: Mapping[str, Any],
    prediction_descriptors: Mapping[tuple[str, str], Mapping[str, Any]],
    truth_loader: Callable[[], Mapping[str, Mapping[str, Any]]],
    timing_descriptors: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    frequency_counts: Mapping[str, Any] | None = None,
    frequency_reference_ids: Mapping[str, str] | None = None,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    exact_practical_margin_pp: float | None = None,
    public_validation_selection: Mapping[str, Any] | None = None,
    repository_root: Path | str | None = None,
    receipt_path: Path | str | None = None,
    truth_path: Path | str | None = None,
    output_root: Path | str | None = None,
    panel_path: Path | str | None = None,
    registration_path: Path | str | None = None,
    selection_plan_path: Path | str | None = None,
    observation_descriptors: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Open truth only after complete executable public evidence is verified.

    The lightweight metadata gate remains first so malformed/incomplete
    matrices fail without touching any file or invoking the loader.  A caller
    that wants to cross the truth boundary must then provide the frozen
    receipt, bound JSON inputs, public observations, and actual prediction
    artifacts.  Prediction tensors are loaded by the gate and only copied into
    scoring descriptors after ``truth_loader`` returns.
    """

    metadata_gate = validate_complete_public_matrix(
        panel,
        registration,
        prediction_descriptors,
        timing_descriptors=timing_descriptors,
    )
    if public_validation_selection is None:
        raise PretruthGateError(
            "frozen public-validation selection is required before truth"
        )
    if timing_descriptors is None:
        raise PretruthGateError("timing receipt matrix is required before truth")
    required_paths = {
        "repository_root": repository_root,
        "receipt_path": receipt_path,
        "truth_path": truth_path,
        "output_root": output_root,
        "panel_path": panel_path,
        "registration_path": registration_path,
        "selection_plan_path": selection_plan_path,
    }
    missing_paths = [name for name, value in required_paths.items() if value is None]
    if missing_paths:
        raise PretruthGateError(
            "executable truth gate paths are absent: " + ", ".join(missing_paths)
        )
    if observation_descriptors is None:
        raise PretruthGateError("observation bindings are required before truth")
    gate = validate_before_truth(
        panel=panel,
        registration=registration,
        prediction_descriptors=prediction_descriptors,
        timing_descriptors=timing_descriptors,
        public_validation_selection=public_validation_selection,
        repository_root=repository_root,
        receipt_path=receipt_path,
        truth_path=truth_path,
        output_root=output_root,
        panel_path=panel_path,
        registration_path=registration_path,
        selection_plan_path=selection_plan_path,
        observation_descriptors=observation_descriptors,
    )
    truth = truth_loader()
    if not isinstance(truth, Mapping):
        raise ConfirmationScoreError("truth loader did not return a cell mapping")
    # All file-backed prediction tensors were validated before the loader.  A
    # new mapping prevents a caller's JSON receipt from replacing the frozen
    # tensor after truth is opened.
    loaded_descriptors = {
        key: {
            **dict(descriptor),
            "predictions": gate["prediction_tensors"][key],
        }
        for key, descriptor in prediction_descriptors.items()
    }
    public_gate = {
        key: value
        for key, value in gate.items()
        if key != "prediction_tensors"
    }
    result = score_matrix(
        panel=panel,
        registration=registration,
        prediction_descriptors=loaded_descriptors,
        cell_inputs=truth,
        timing_descriptors=timing_descriptors,
        frequency_counts=frequency_counts,
        frequency_reference_ids=frequency_reference_ids,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        exact_practical_margin_pp=exact_practical_margin_pp,
        public_validation_selection=public_validation_selection,
    )
    result["truth_gate"]["initial_metadata_gate"] = metadata_gate
    result["truth_gate"]["executable_public_gate"] = public_gate
    return result


def _cli_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PretruthGateError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PretruthGateError(f"{description} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise PretruthGateError(f"{description} must be a JSON object")
    return dict(value)


def _cli_matrix_entries(
    value: Mapping[str, Any],
    *,
    kind: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Normalize a compact list or nested mapping of matrix descriptors."""

    payload: Any = value.get(kind, value.get("entries", value))
    entries: dict[tuple[str, str], dict[str, Any]] = {}

    def add(cell_id: Any, method_id: Any, row: Any) -> None:
        if cell_id not in EXPECTED_CELL_IDS or method_id not in METHOD_IDS:
            raise PretruthGateError(f"{kind} has an unknown cell/method")
        if not isinstance(row, Mapping):
            raise PretruthGateError(f"{kind} descriptor is malformed")
        descriptor = row.get("descriptor")
        if not isinstance(descriptor, Mapping):
            descriptor = row
        key = (str(cell_id), str(method_id))
        if key in entries:
            raise PretruthGateError(f"{kind} contains duplicate cell/method entries")
        entries[key] = dict(descriptor)

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, Mapping):
                raise PretruthGateError(f"{kind} entry is malformed")
            add(row.get("cell_id"), row.get("method_id"), row)
    elif isinstance(payload, Mapping):
        # Preferred compact form: {"cell_id::method_id": descriptor}.
        for raw_key, row in payload.items():
            if raw_key in EXPECTED_CELL_IDS and isinstance(row, Mapping):
                for method_id, method_row in row.items():
                    add(raw_key, method_id, method_row)
                continue
            if isinstance(raw_key, str) and "::" in raw_key:
                cell_id, method_id = raw_key.split("::", 1)
                add(cell_id, method_id, row)
                continue
            if raw_key in {"schema", "task_id", "status", "entries", kind}:
                continue
            # A single descriptor object is accepted only when it carries its
            # own identity; this keeps malformed matrix maps fail-closed.
            if raw_key in {"cell_id", "method_id"}:
                add(payload.get("cell_id"), payload.get("method_id"), payload)
                break
            raise PretruthGateError(
                f"{kind} keys must be cell::method or canonical cell IDs"
            )
    else:
        raise PretruthGateError(f"{kind} must contain a list or mapping")
    expected = {(cell_id, method_id) for cell_id in EXPECTED_CELL_IDS for method_id in METHOD_IDS}
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise PretruthGateError(f"{kind} matrix is incomplete: missing={missing!r} extra={extra!r}")
    return entries


def _cli_prediction_descriptors(
    path: Path,
    *,
    output_root: Path,
    repository_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    value = _cli_json(path, description="prediction descriptor manifest")
    entries = _cli_matrix_entries(value, kind="predictions")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for (cell_id, method_id), row in entries.items():
        style, condition = cell_id.split("__", 1)
        descriptor = dict(row)
        descriptor.setdefault("schema", PREDICTION_SCHEMA)
        descriptor.setdefault("task_id", TASK_ID)
        descriptor.setdefault("cell_id", cell_id)
        descriptor.setdefault("method_id", method_id)
        descriptor.setdefault("shape", [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS])
        descriptor.setdefault("candidate_policy", CANDIDATE_POLICIES[method_id])
        descriptor.setdefault("candidate_arrays_present", False)
        if CANDIDATE_POLICIES[method_id] == "output_only":
            descriptor.setdefault("candidate_output", "omitted_after_decision")
        descriptor.setdefault("warmup_runs_per_record", 1)
        descriptor.setdefault("measured_runs_per_record", 1)
        if not any(
            isinstance(descriptor.get(key), Mapping)
            and isinstance(descriptor[key].get("path"), str)
            for key in ("prediction_artifact", "artifact", "prediction_file")
        ) and not isinstance(descriptor.get("prediction_path"), str):
            artifact_path = (
                output_root / style / condition / f"{method_id}.safetensors"
            )
            try:
                artifact = file_record(artifact_path, repository_root=repository_root)
            except (FootingError, OSError, ValueError) as exc:
                raise PretruthGateError(
                    f"prediction artifact is absent: {cell_id}/{method_id}"
                ) from exc
            descriptor["prediction_artifact"] = artifact
        result[(cell_id, method_id)] = descriptor
    return result


def _cli_timing_descriptors(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    value = _cli_json(path, description="timing descriptor manifest")
    entries = _cli_matrix_entries(value, kind="timings")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, row in entries.items():
        timing = row.get("timing")
        if isinstance(timing, Mapping):
            timing = dict(timing)
        else:
            timing = dict(row)
        result[key] = timing
    return result


def _cli_observations(path: Path) -> dict[str, Mapping[str, Any]]:
    value = _cli_json(path, description="observation descriptor manifest")
    payload = value.get("observations", value)
    if not isinstance(payload, Mapping):
        raise PretruthGateError("observation manifest must contain a mapping")
    result = {
        str(cell_id): dict(row)
        for cell_id, row in payload.items()
        if isinstance(row, Mapping)
    }
    if set(result) != set(EXPECTED_CELL_IDS):
        raise PretruthGateError("observation descriptor manifest is incomplete")
    return result


def _cli_frequency_counts(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = _cli_json(path, description="frequency reference manifest")
    payload = value.get("frequency_references", value.get("frequencies", value))
    if not isinstance(payload, Mapping):
        raise ConfirmationScoreError("frequency manifest must contain a mapping")
    result: dict[str, Any] = {}
    for reference in ("original", "enriched"):
        counts = payload.get(reference)
        if counts is None:
            continue
        if isinstance(counts, Mapping):
            normalized: dict[int, int] = {}
            for raw_key, raw_count in counts.items():
                try:
                    token = int(raw_key)
                except (TypeError, ValueError) as exc:
                    raise ConfirmationScoreError(
                        f"frequency token key is invalid: {reference}/{raw_key!r}"
                    ) from exc
                if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                    raise ConfirmationScoreError(
                        f"frequency count is invalid: {reference}/{raw_key!r}"
                    )
                normalized[token] = raw_count
            result[reference] = normalized
        elif isinstance(counts, list):
            result[reference] = counts
        else:
            raise ConfirmationScoreError(f"frequency reference is malformed: {reference}")
    if set(result) != {"original", "enriched"}:
        raise ConfirmationScoreError("frequency manifest must provide original and enriched references")
    return result


def _cli_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cli_truth_binding_manifest(
    path: Path,
    *,
    truth_path: Path,
    panel: Mapping[str, Any],
    selection_plan: Mapping[str, Any],
    panel_path: Path,
    selection_plan_path: Path,
    receipt_path: Path,
    output_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate the frozen producer descriptor after the public gate only.

    The producer writes this label-free descriptor before truth is opened.  It
    must then be copied into the frozen prediction root so the core freeze
    receipt covers its bytes.  Resolve and verify that receipt entry before
    touching the private sidecar.
    """

    root = repository_root.expanduser().resolve()
    binding_path = path.expanduser().resolve()
    frozen_root = output_root.expanduser().resolve()
    if binding_path.is_symlink() or not binding_path.is_file():
        raise ConfirmationScoreError("frozen truth binding descriptor is unavailable")
    try:
        binding_path.relative_to(frozen_root)
    except ValueError as exc:
        raise ConfirmationScoreError(
            "truth binding descriptor must be inside the frozen prediction root"
        ) from exc
    try:
        receipt = verify_freeze_receipt(
            receipt_path.expanduser().resolve(), repository_root=root
        )
    except (FreezeError, OSError, ValueError) as exc:
        raise ConfirmationScoreError(
            "freeze receipt does not validate the truth binding descriptor"
        ) from exc
    try:
        receipt_root = (root / str(receipt["frozen_root"])).resolve()
        if receipt_root != frozen_root:
            raise ConfirmationScoreError(
                "freeze receipt root differs from the requested prediction root"
            )
        descriptor_record = file_record(binding_path, repository_root=root)
    except (FootingError, OSError, ValueError) as exc:
        raise ConfirmationScoreError(
            "frozen truth binding descriptor cannot be recorded"
        ) from exc
    relative = descriptor_record["path"]
    entries = [
        entry
        for entry in receipt.get("entries", [])
        if isinstance(entry, Mapping) and entry.get("path") == relative
    ]
    if len(entries) != 1 or dict(entries[0]) != dict(descriptor_record):
        raise ConfirmationScoreError(
            "freeze receipt does not bind the supplied truth descriptor"
        )

    binding = _cli_json(path, description="truth binding manifest")
    if binding.get("schema") != TRUTH_MANIFEST_SCHEMA or binding.get("task_id") != TASK_ID:
        raise ConfirmationScoreError("truth binding manifest identity changed")
    if binding.get("status") != TRUTH_MANIFEST_STATUS:
        raise ConfirmationScoreError("truth binding manifest is not prepared")
    truth_file = binding.get("truth_file")
    if not isinstance(truth_file, Mapping):
        raise ConfirmationScoreError("truth binding has no sidecar descriptor")
    declared_truth_path = truth_file.get("path")
    if not isinstance(declared_truth_path, str) or not Path(declared_truth_path).is_absolute():
        raise ConfirmationScoreError("truth sidecar descriptor path is not absolute")
    if Path(declared_truth_path).expanduser().resolve() != truth_path.expanduser().resolve():
        raise ConfirmationScoreError("truth sidecar path differs from its binding manifest")
    if not isinstance(truth_file.get("bytes"), int) or truth_file["bytes"] < 0:
        raise ConfirmationScoreError("truth sidecar byte count is invalid")
    if not isinstance(truth_file.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", truth_file["sha256"]) is None:
        raise ConfirmationScoreError("truth sidecar SHA-256 binding is invalid")
    if binding.get("cell_order") != list(EXPECTED_CELL_IDS):
        raise ConfirmationScoreError("truth sidecar cell order changed")
    expected_keys = sorted(
        f"{cell_id}__{suffix}"
        for cell_id in EXPECTED_CELL_IDS
        for suffix in ("token_ids", "attention_mask", "position_ids")
    )
    if binding.get("truth_tensor_keys") != expected_keys:
        raise ConfirmationScoreError("truth sidecar tensor key binding changed")
    if binding.get("truth_opened") is not False:
        raise ConfirmationScoreError("truth binding manifest was written after truth opened")
    if binding.get("reconstruction_root_contains_truth") is not False:
        raise ConfirmationScoreError("truth sidecar is inside the reconstruction root")

    # Producer truth manifests use absolute external-style records for all
    # three bound files, including the public panel and selection plan.
    try:
        panel_record = external_file_record(panel_path)
        plan_record = external_file_record(selection_plan_path)
    except (FootingError, OSError, ValueError) as exc:
        raise ConfirmationScoreError("truth binding public file is unavailable") from exc
    declared_panel = binding.get("panel")
    declared_plan = binding.get("selection_plan")
    if not isinstance(declared_panel, Mapping) or dict(declared_panel) != panel_record:
        raise ConfirmationScoreError("truth sidecar is bound to a different panel")
    if not isinstance(declared_plan, Mapping) or dict(declared_plan) != plan_record:
        raise ConfirmationScoreError("truth sidecar is bound to a different selection plan")

    cells = panel.get("cells")
    if not isinstance(cells, Mapping):
        raise ConfirmationScoreError("panel cells are unavailable for truth binding")
    expected_observations: dict[str, str] = {}
    expected_ids: dict[str, list[str]] = {}
    for cell_id in EXPECTED_CELL_IDS:
        cell = cells.get(cell_id)
        if not isinstance(cell, Mapping):
            raise ConfirmationScoreError(f"panel cell is absent for truth binding: {cell_id}")
        records = cell.get("records")
        if not isinstance(records, list) or len(records) != RECORDS_PER_DOMAIN:
            raise ConfirmationScoreError(f"panel records are malformed for truth binding: {cell_id}")
        ids = [row.get("record_id") for row in records if isinstance(row, Mapping)]
        if len(ids) != RECORDS_PER_DOMAIN or any(not isinstance(value, str) or not value for value in ids):
            raise ConfirmationScoreError(f"panel record IDs are malformed for truth binding: {cell_id}")
        expected_ids[cell_id] = ids
        observation = cell.get("observation")
        if not isinstance(observation, Mapping) or not isinstance(observation.get("sha256"), str):
            raise ConfirmationScoreError(f"panel observation binding is absent: {cell_id}")
        expected_observations[cell_id] = str(observation["sha256"])
    if binding.get("observation_sha256") != expected_observations:
        raise ConfirmationScoreError("truth sidecar observation bindings changed")
    expected_record_digests = {
        style: _cli_json_sha256(expected_ids[f"{style}__{CONDITION_ORDER[0]}"])
        for style in STYLE_ORDER
    }
    if binding.get("record_ids_sha256") != expected_record_digests:
        raise ConfirmationScoreError("truth sidecar record-ID binding changed")
    method_freeze_sha = binding.get("method_freeze_sha256")
    if not isinstance(method_freeze_sha, str) or re.fullmatch(r"[0-9a-f]{64}", method_freeze_sha) is None:
        raise ConfirmationScoreError("truth sidecar method-freeze binding is invalid")
    try:
        bound_selection_plan = _cli_json(
            selection_plan_path, description="selection plan for truth binding"
        )
    except PretruthGateError as exc:
        raise ConfirmationScoreError(
            "selection plan for truth binding is unavailable"
        ) from exc
    if dict(bound_selection_plan) != dict(selection_plan):
        raise ConfirmationScoreError(
            "selection plan content changed before truth binding"
        )
    if bound_selection_plan.get("method_freeze_sha256") != method_freeze_sha:
        raise ConfirmationScoreError(
            "truth sidecar method-freeze binding differs from the selection plan"
        )
    panel_method_freeze = panel.get("method_freeze_sha256")
    if panel_method_freeze is not None and panel_method_freeze != method_freeze_sha:
        raise ConfirmationScoreError(
            "truth sidecar method-freeze binding differs from the panel"
        )
    return dict(binding)


def _cli_truth_header_metadata(
    metadata: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    expected_ids: Mapping[str, Sequence[str]],
) -> None:
    """Check producer truth sidecar identity and ordered row header."""

    if metadata.get("schema") != TRUTH_MANIFEST_SCHEMA or metadata.get("task_id") != TASK_ID:
        raise ConfirmationScoreError("truth sidecar header identity changed")
    panel = binding["panel"]
    plan = binding["selection_plan"]
    if metadata.get("panel_sha256") != panel.get("sha256"):
        raise ConfirmationScoreError("truth sidecar header panel binding changed")
    if metadata.get("selection_plan_sha256") != plan.get("sha256"):
        raise ConfirmationScoreError("truth sidecar header selection-plan binding changed")
    if metadata.get("method_freeze_sha256") != binding.get("method_freeze_sha256"):
        raise ConfirmationScoreError("truth sidecar header method-freeze binding changed")
    for style in STYLE_ORDER:
        raw = metadata.get(f"record_ids_{style}")
        if not isinstance(raw, str):
            raise ConfirmationScoreError(f"truth sidecar header record IDs are absent: {style}")
        try:
            observed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfirmationScoreError(f"truth sidecar header record IDs are invalid: {style}") from exc
        if observed != list(expected_ids[f"{style}__{CONDITION_ORDER[0]}"]):
            raise ConfirmationScoreError(f"truth sidecar row order differs from panel: {style}")


def _cli_truth_cells(
    path: Path,
    *,
    panel: Mapping[str, Any],
    selection_plan: Mapping[str, Any],
    truth_binding_path: Path,
    panel_path: Path,
    selection_plan_path: Path,
    receipt_path: Path,
    output_root: Path,
    repository_root: Path,
) -> dict[str, Mapping[str, Any]]:
    """Validate and read the producer sidecar; called only after the gate."""

    binding = _cli_truth_binding_manifest(
        truth_binding_path,
        truth_path=path,
        panel=panel,
        selection_plan=selection_plan,
        panel_path=panel_path,
        selection_plan_path=selection_plan_path,
        receipt_path=receipt_path,
        output_root=output_root,
        repository_root=repository_root,
    )
    # This is the first private-sidecar hash/open operation in the CLI path.
    try:
        truth_record = external_file_record(path)
    except (FootingError, OSError, ValueError) as exc:
        raise ConfirmationScoreError("truth sidecar is unavailable") from exc
    if dict(binding["truth_file"]) != truth_record:
        raise ConfirmationScoreError("truth sidecar bytes or hash differ from its binding manifest")

    cells = panel.get("cells")
    if not isinstance(cells, Mapping):
        raise ConfirmationScoreError("panel cells are unavailable for truth loading")
    expected_ids = {
        cell_id: [row["record_id"] for row in cells[cell_id]["records"]]
        for cell_id in EXPECTED_CELL_IDS
    }
    expected_keys = {
        f"{cell_id}__{suffix}"
        for cell_id in EXPECTED_CELL_IDS
        for suffix in ("token_ids", "attention_mask", "position_ids")
    }
    result: dict[str, Mapping[str, Any]] = {}
    if path.suffix.casefold() == ".safetensors":
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
                _cli_truth_header_metadata(metadata, binding=binding, expected_ids=expected_ids)
                if set(handle.keys()) != expected_keys:
                    raise ConfirmationScoreError("truth sidecar tensor set changed")
                for cell_id in EXPECTED_CELL_IDS:
                    result[cell_id] = {
                        "record_ids": list(expected_ids[cell_id]),
                        "truth": handle.get_tensor(f"{cell_id}__token_ids").to(torch.long).contiguous(),
                        "attention_mask": handle.get_tensor(
                            f"{cell_id}__attention_mask"
                        ).to(torch.bool).contiguous(),
                        "position_ids": handle.get_tensor(
                            f"{cell_id}__position_ids"
                        ).to(torch.long).contiguous(),
                    }
        except ConfirmationScoreError:
            raise
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise ConfirmationScoreError("truth sidecar is unreadable") from exc
        return result

    payload = _cli_json(path, description="truth sidecar")
    metadata = payload.get("metadata", payload)
    if not isinstance(metadata, Mapping):
        raise ConfirmationScoreError("truth sidecar header is malformed")
    _cli_truth_header_metadata(metadata, binding=binding, expected_ids=expected_ids)
    cells_payload = payload.get("cells")
    if not isinstance(cells_payload, Mapping):
        raise ConfirmationScoreError("truth sidecar cells are malformed")
    for cell_id in EXPECTED_CELL_IDS:
        row = cells_payload.get(cell_id)
        if not isinstance(row, Mapping):
            raise ConfirmationScoreError(f"truth sidecar cell is absent: {cell_id}")
        if row.get("record_ids") != expected_ids[cell_id]:
            raise ConfirmationScoreError(f"truth sidecar row order differs from panel: {cell_id}")
        truth = row.get("truth", row.get("token_ids"))
        mask = row.get("attention_mask", row.get("mask"))
        positions = row.get("position_ids", row.get("positions"))
        if truth is None or mask is None or positions is None:
            raise ConfirmationScoreError(f"truth sidecar geometry is incomplete: {cell_id}")
        result[cell_id] = {
            "record_ids": list(expected_ids[cell_id]),
            "truth": truth,
            "attention_mask": mask,
            "position_ids": positions,
        }
    return result


def _cli_write_result(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ConfirmationScoreError(f"score result is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ConfirmationScoreError("score result is not JSON serializable") from exc
    path.write_text(encoded, encoding="utf-8")


def _cli_score(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    panel_path = args.panel.expanduser().resolve()
    registration_path = args.registration.expanduser().resolve()
    selection_plan_path = args.selection_plan.expanduser().resolve()
    observations_path = args.observations.expanduser().resolve()
    truth_path = args.truth.expanduser().resolve()
    truth_binding_path = args.truth_binding.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    panel = _cli_json(panel_path, description="panel")
    registration = _cli_json(registration_path, description="registration")
    selection_plan = _cli_json(selection_plan_path, description="selection plan")
    selection = selection_plan.get("public_validation_selection", selection_plan)
    if not isinstance(selection, Mapping):
        raise PretruthGateError("selection plan has no frozen public selection")
    observations = _cli_observations(observations_path)
    predictions = _cli_prediction_descriptors(
        args.predictions.expanduser().resolve(),
        output_root=output_root,
        repository_root=root,
    )
    timings = _cli_timing_descriptors(args.timings.expanduser().resolve())
    frequencies = _cli_frequency_counts(
        args.frequency_manifest.expanduser().resolve()
        if args.frequency_manifest is not None
        else None
    )
    try:
        result_path.resolve().relative_to(output_root.resolve())
    except ValueError:
        pass
    else:
        raise ConfirmationScoreError("score result must be outside the frozen prediction root")
    result = score_with_truth_loader(
        panel=panel,
        registration=registration,
        prediction_descriptors=predictions,
        timing_descriptors=timings,
        public_validation_selection=selection,
        repository_root=root,
        receipt_path=receipt_path,
        truth_path=truth_path,
        output_root=output_root,
        panel_path=panel_path,
        registration_path=registration_path,
        selection_plan_path=selection_plan_path,
        observation_descriptors=observations,
        frequency_counts=frequencies,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
        truth_loader=lambda: _cli_truth_cells(
            truth_path,
            panel=panel,
            selection_plan=selection_plan,
            truth_binding_path=truth_binding_path,
            panel_path=panel_path,
            selection_plan_path=selection_plan_path,
            receipt_path=receipt_path,
            output_root=output_root,
            repository_root=root,
        ),
    )
    _cli_write_result(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(result_path),
                "truth_opened_after_gate": True,
                "prediction_artifact_count": result["truth_gate"]["executable_public_gate"][
                    "prediction_artifact_count"
                ],
            },
            sort_keys=True,
        )
    )
    return result


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the frozen TRR-0005 matrix after the executable public gate.")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--truth-binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--frequency-manifest", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    if args.bootstrap_draws <= 0:
        raise SystemExit("TRR-0005 score error: bootstrap draws must be positive")
    try:
        _cli_score(args)
    except (ConfirmationScoreError, FreezeError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0005 score error: {exc}") from exc
    return 0


__all__ = [
    "CellInput",
    "ConfirmationScoreError",
    "PretruthGateError",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "EXACT_CONFIDENCE",
    "paired_method_comparison",
    "paired_token_bootstrap",
    "exact_beneficial_discordance_bound",
    "exact_net_benefit_bound",
    "validate_public_validation_selection",
    "validate_before_truth",
    "RUNTIME_EMBEDDING_ROLE",
    "RUNTIME_P0_ROLES",
    "score_cell",
    "score_matrix",
    "score_with_truth_loader",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
