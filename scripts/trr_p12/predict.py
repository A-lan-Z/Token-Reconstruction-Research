"""Frozen early-direction extrapolation predictor for TRR-P12.

The predictor is chosen without later-stage scores or labels.  It receives only
stage-0 and stage-64 full-vocabulary decoder scores and extrapolates
``u_t = u_0 + (t / 64) * (u_64 - u_0)`` for the preregistered forecast stages
128 and 256.  For each row it computes the full-vocabulary margin of the
stage-0 winner against its strongest estimated competitor.  Risk is the
negative minimum of those two forecast margins; a strict predicted change is
``risk > 0`` and a zero risk is reported as a tie boundary separately.

This forecast is a prospective score-only diagnostic.  It is not fitted to
truth, does not inspect stage-128/256 scores, and is not the same quantity as
the post-freeze actual crossing analysis in ``boundary.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from scripts.trr_p12.boundary import BoundaryError


SCHEMA = "token-reconstruction.trr-p12-direction-forecast.v1"
EARLY_STAGE = 64
FORECAST_STAGES = (128, 256)


class PredictorError(ValueError):
    """Raised when score geometry or truth-scoring inputs are malformed."""


def _scores(value: torch.Tensor, *, label: str) -> torch.Tensor:
    result = torch.as_tensor(value).detach()
    if result.ndim < 2 or int(result.shape[-1]) < 2:
        raise PredictorError(f"{label} must have shape [...,vocabulary>=2]")
    if not result.dtype.is_floating_point or not bool(torch.isfinite(result).all().item()):
        raise PredictorError(f"{label} must be finite floating point")
    return result.float()


def _winner_and_competitor(scores: torch.Tensor, winner: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = scores.reshape(-1, int(scores.shape[-1]))
    winner_flat = winner.reshape(-1).to(dtype=torch.long)
    rows = torch.arange(int(flat.shape[0]), dtype=torch.long)
    argmax = torch.argmax(flat, dim=-1).to(dtype=torch.long)
    masked = flat.clone()
    masked[rows, winner_flat] = float("-inf")
    competitor = torch.argmax(masked, dim=-1).to(dtype=torch.long)
    return argmax.reshape(scores.shape[:-1]), competitor.reshape(scores.shape[:-1])


def _valid_mask(value: torch.Tensor | None, *, shape: tuple[int, ...]) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool)
    result = torch.as_tensor(value)
    if tuple(result.shape) != shape:
        raise PredictorError("valid mask geometry differs from score rows")
    if result.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise PredictorError("valid mask must be boolean or integer")
    return result.to(device="cpu", dtype=torch.bool)


@dataclass(frozen=True)
class DirectionForecast:
    """Compact outputs from the stage-0/64 extrapolation."""

    row_shape: tuple[int, ...]
    valid: torch.Tensor
    baseline_winner: torch.Tensor
    forecast_competitor: dict[int, torch.Tensor]
    forecast_winner: dict[int, torch.Tensor]
    forecast_margin: dict[int, torch.Tensor]
    risk_by_stage: dict[int, torch.Tensor]
    risk: torch.Tensor
    predicted_strict_change_by_stage: dict[int, torch.Tensor]
    predicted_tie_by_stage: dict[int, torch.Tensor]
    predicted_argmax_change_by_stage: dict[int, torch.Tensor]
    predicted_strict_change: torch.Tensor
    predicted_tie: torch.Tensor
    predicted_argmax_change: torch.Tensor

    def _valid(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(-1)[self.valid.reshape(-1)]

    def summary(self) -> dict[str, Any]:
        n = int(self.valid.sum().item())
        if n <= 0:
            raise PredictorError("direction forecast has no valid rows")
        def count(value: torch.Tensor) -> int:
            return int(self._valid(value.to(dtype=torch.bool)).sum().item())
        return {
            "schema": SCHEMA,
            "row_shape": list(self.row_shape),
            "rows": n,
            "forecast_stages": list(FORECAST_STAGES),
            "strict_predicted_changes_by_stage": {str(stage): count(value) for stage, value in self.predicted_strict_change_by_stage.items()},
            "predicted_ties_by_stage": {str(stage): count(value) for stage, value in self.predicted_tie_by_stage.items()},
            "argmax_changes_with_tie_rule_by_stage": {str(stage): count(value) for stage, value in self.predicted_argmax_change_by_stage.items()},
            "risk_mean_by_stage": {str(stage): float(self._valid(value.float()).mean().item()) for stage, value in self.risk_by_stage.items()},
            "risk_max_by_stage": {str(stage): float(self._valid(value.float()).max().item()) for stage, value in self.risk_by_stage.items()},
            "strict_predicted_changes_union": count(self.predicted_strict_change),
            "predicted_ties_union": count(self.predicted_tie),
            "argmax_changes_with_tie_rule_union": count(self.predicted_argmax_change),
            "strict_change_rate_union": float(count(self.predicted_strict_change)) / n,
            "tie_rate_union": float(count(self.predicted_tie)) / n,
            "risk_mean": float(self._valid(self.risk.float()).mean().item()),
            "risk_max": float(self._valid(self.risk.float()).max().item()),
            "decision_rule": {
                "forecast": "u_t = u_0 + (t/64) * (u_64-u_0)",
                "margin": "score of stage-0 winner minus the maximum estimated score over all other vocabulary IDs",
                "per_stage_strict_prediction": "forecast_margin[t] < 0 for the matching later stage t",
                "per_stage_tie": "forecast_margin[t] == 0 for the matching later stage t",
                "union_risk": "-min(forecast_margin[128], forecast_margin[256]); descriptive union only",
                "union_strict_prediction": "per-stage strict prediction at either forecast stage",
                "argmax_prediction": "torch.argmax on each estimated full-vocabulary row, including its tie rule",
            },
            "limitations": "No stage-128/256 scores or truth are used to construct this forecast; it is checked prospectively against later stages after output freeze.",
        }

    def compact(self) -> dict[str, Any]:
        def ids(value: torch.Tensor) -> list[int]:
            return [int(item) for item in value.reshape(-1).tolist()]
        def vals(value: torch.Tensor) -> list[float]:
            return [float(item) for item in value.reshape(-1).tolist()]
        return {
            "schema": SCHEMA,
            "row_shape": list(self.row_shape),
            "valid": [bool(item) for item in self.valid.reshape(-1).tolist()],
            "baseline_winner": ids(self.baseline_winner),
            "forecast_competitor": {str(stage): ids(value) for stage, value in self.forecast_competitor.items()},
            "forecast_winner": {str(stage): ids(value) for stage, value in self.forecast_winner.items()},
            "forecast_margin": {str(stage): vals(value) for stage, value in self.forecast_margin.items()},
            "risk_by_stage": {str(stage): vals(value) for stage, value in self.risk_by_stage.items()},
            "risk": vals(self.risk),
            "predicted_strict_change_by_stage": {str(stage): [bool(item) for item in value.reshape(-1).tolist()] for stage, value in self.predicted_strict_change_by_stage.items()},
            "predicted_tie_by_stage": {str(stage): [bool(item) for item in value.reshape(-1).tolist()] for stage, value in self.predicted_tie_by_stage.items()},
            "predicted_argmax_change_by_stage": {str(stage): [bool(item) for item in value.reshape(-1).tolist()] for stage, value in self.predicted_argmax_change_by_stage.items()},
            "predicted_strict_change_union": [bool(item) for item in self.predicted_strict_change.reshape(-1).tolist()],
            "predicted_tie_union": [bool(item) for item in self.predicted_tie.reshape(-1).tolist()],
            "predicted_argmax_change_union": [bool(item) for item in self.predicted_argmax_change.reshape(-1).tolist()],
        }


def direction_forecast(
    stage0_scores: torch.Tensor,
    stage64_scores: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> DirectionForecast:
    """Forecast later margins from stage-0 and stage-64 scores only."""

    base = _scores(stage0_scores, label="stage-0 scores")
    early = _scores(stage64_scores, label="stage-64 scores")
    if tuple(base.shape) != tuple(early.shape):
        raise PredictorError("stage-0 and stage-64 score geometry differs")
    row_shape = tuple(int(item) for item in base.shape[:-1])
    valid = _valid_mask(valid_mask, shape=row_shape)
    flat = base.reshape(-1, int(base.shape[-1]))
    baseline_winner = torch.argmax(flat, dim=-1).reshape(row_shape).to(dtype=torch.long)
    margins: dict[int, torch.Tensor] = {}
    competitors: dict[int, torch.Tensor] = {}
    winners: dict[int, torch.Tensor] = {}
    for stage in FORECAST_STAGES:
        estimate = base + (float(stage) / float(EARLY_STAGE)) * (early - base)
        winner, competitor = _winner_and_competitor(estimate, baseline_winner)
        estimate_flat = estimate.reshape(-1, int(estimate.shape[-1]))
        winner_flat = baseline_winner.reshape(-1)
        competitor_flat = competitor.reshape(-1)
        rows = torch.arange(int(estimate_flat.shape[0]), dtype=torch.long)
        margin = estimate_flat[rows, winner_flat] - estimate_flat[rows, competitor_flat]
        margins[stage] = margin.reshape(row_shape)
        competitors[stage] = competitor
        winners[stage] = winner
    risk_by_stage = {stage: -margin for stage, margin in margins.items()}
    strict_by_stage = {stage: margin.lt(0.0) for stage, margin in margins.items()}
    tie_by_stage = {stage: margin.eq(0.0) for stage, margin in margins.items()}
    argmax_by_stage = {stage: winners[stage].ne(baseline_winner) for stage in FORECAST_STAGES}
    minimum = torch.minimum(margins[128], margins[256])
    risk = -minimum
    strict = strict_by_stage[128] | strict_by_stage[256]
    tie = tie_by_stage[128] | tie_by_stage[256]
    argmax_change = argmax_by_stage[128] | argmax_by_stage[256]
    return DirectionForecast(
        row_shape=row_shape,
        valid=valid,
        baseline_winner=baseline_winner.reshape(-1),
        forecast_competitor={stage: value.reshape(-1) for stage, value in competitors.items()},
        forecast_winner={stage: value.reshape(-1) for stage, value in winners.items()},
        forecast_margin={stage: value.reshape(-1) for stage, value in margins.items()},
        risk_by_stage={stage: value.reshape(-1) for stage, value in risk_by_stage.items()},
        risk=risk.reshape(-1),
        predicted_strict_change_by_stage={stage: value.reshape(-1) for stage, value in strict_by_stage.items()},
        predicted_tie_by_stage={stage: value.reshape(-1) for stage, value in tie_by_stage.items()},
        predicted_argmax_change_by_stage={stage: value.reshape(-1) for stage, value in argmax_by_stage.items()},
        predicted_strict_change=strict.reshape(-1),
        predicted_tie=tie.reshape(-1),
        predicted_argmax_change=argmax_change.reshape(-1),
    )


def evaluate_forecast_against_truth(
    forecast: DirectionForecast,
    baseline_prediction: torch.Tensor,
    later_prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    record_mask: torch.Tensor | None = None,
    later_stage: int = 128,
    bos_index: int = 0,
) -> dict[str, Any]:
    """Score forecast changes against later correctness after truth opening.

    The returned conditional confusion table is restricted to positions that
    were correct at baseline, where the later error is the preregistered
    ``broken`` outcome.  The all-valid table is retained as a descriptive
    check and does not alter the frozen forecast.
    """

    baseline = torch.as_tensor(baseline_prediction).to(dtype=torch.long)
    later = torch.as_tensor(later_prediction).to(dtype=torch.long)
    target = torch.as_tensor(truth).to(dtype=torch.long)
    if baseline.ndim != 2 or tuple(later.shape) != tuple(baseline.shape) or tuple(target.shape) != tuple(baseline.shape):
        raise PredictorError("prediction/truth geometry differs")
    if int(later_stage) not in FORECAST_STAGES:
        raise PredictorError(f"later stage must be one of {FORECAST_STAGES}")
    if tuple(forecast.row_shape) == tuple(baseline.shape):
        offset = 0
    elif tuple(forecast.row_shape) == (int(baseline.shape[0]), int(baseline.shape[1]) - 1):
        offset = 1
    else:
        raise PredictorError("forecast rows do not align with full-sequence or post-BOS predictions")
    if not 0 <= int(bos_index) < int(baseline.shape[1]):
        raise PredictorError("BOS index is outside predictions")
    baseline = baseline[:, offset:]
    later = later[:, offset:]
    target = target[:, offset:]
    valid = forecast.valid.reshape(baseline.shape).clone()
    if offset == 0:
        valid[:, int(bos_index)] = False
    if record_mask is not None:
        rows = torch.as_tensor(record_mask).to(dtype=torch.bool).reshape(-1)
        if int(rows.numel()) != int(baseline.shape[0]):
            raise PredictorError("record holdout mask is not aligned")
        valid &= rows[:, None]
    baseline_correct = baseline.eq(target)
    broken = baseline_correct & ~later.eq(target) & valid
    predicted = forecast.predicted_strict_change_by_stage[int(later_stage)].reshape(baseline.shape)
    predicted_tie = forecast.predicted_tie_by_stage[int(later_stage)].reshape(baseline.shape)
    actual = broken
    def table(mask: torch.Tensor) -> dict[str, int]:
        eligible = mask
        target_value = actual & eligible
        guess_value = predicted & eligible
        return {
            "rows": int(mask.sum().item()),
            "baseline_correct_rows": int((baseline_correct & eligible).sum().item()),
            "broken_rows": int(target_value.sum().item()),
            "predicted_strict_change_rows": int(guess_value.sum().item()),
            "true_positive": int((target_value & guess_value & eligible).sum().item()),
            "false_positive": int((~target_value & guess_value & eligible).sum().item()),
            "true_negative": int((~target_value & ~guess_value & eligible).sum().item()),
            "false_negative": int((target_value & ~guess_value & eligible).sum().item()),
            "predicted_tie_rows": int((predicted_tie & eligible).sum().item()),
        }
    return {
        "schema": "token-reconstruction.trr-p12-direction-forecast-evaluation.v1",
        "all_valid": table(valid),
        "baseline_correct_only": table(valid & baseline_correct),
        "later_stage": int(later_stage),
        "record_mask_applied": record_mask is not None,
        "truth_use": "post-freeze scoring only; truth does not alter forecast outputs",
    }


__all__ = [
    "EARLY_STAGE",
    "FORECAST_STAGES",
    "DirectionForecast",
    "PredictorError",
    "SCHEMA",
    "direction_forecast",
    "evaluate_forecast_against_truth",
]
