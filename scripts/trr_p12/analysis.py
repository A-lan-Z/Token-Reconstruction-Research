"""Source-paired TRR-P12 stage scoring and uncertainty wrappers.

This wrapper keeps P12's variable domain/stage matrix separate from P11's
four-cell fixed schema while reusing the validated TRR-0010 scorer primitives.
It is a post-freeze numerical scorer: it accepts predictions and truth that
have already passed the caller's freeze/truth gate and performs no model or
source loading.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from scripts.trr_p11.scorer import trr0010_analysis as scorer


SCHEMA = "token-reconstruction.trr-p12-source-paired-analysis.v1"
DEFAULT_BOOTSTRAP_SEED = 9009
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_ONE_SIDED_ALPHA = 0.025


class AnalysisError(ValueError):
    """Raised when paired stage inputs are not aligned or finite."""


def source_order_digest(source_ids: Sequence[str]) -> str:
    values = [str(item) for item in source_ids]
    if not values or len(values) != len(set(values)):
        raise AnalysisError("source IDs must be non-empty and unique")
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _predictions(value: torch.Tensor, *, label: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if result.ndim != 2 or int(result.shape[0]) <= 0 or int(result.shape[1]) <= 1:
        raise AnalysisError(f"{label} must have shape [records,positions>1]")
    if result.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise AnalysisError(f"{label} must be integer token IDs")
    return result.to(device="cpu", dtype=torch.long)


def _truth(value: torch.Tensor, *, shape: tuple[int, int]) -> torch.Tensor:
    result = _predictions(value, label="truth")
    if tuple(result.shape) != shape:
        raise AnalysisError("truth geometry differs from predictions")
    return result


def _mask(value: torch.Tensor | None, *, shape: tuple[int, int], bos_index: int) -> torch.Tensor:
    if value is None:
        result = torch.ones(shape, dtype=torch.bool)
    else:
        result = torch.as_tensor(value)
        if tuple(result.shape) != shape:
            raise AnalysisError("valid mask geometry differs from predictions")
        if result.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise AnalysisError("valid mask must be boolean or integer")
        result = result.to(device="cpu", dtype=torch.bool)
    if not 0 <= int(bos_index) < shape[1]:
        raise AnalysisError("BOS index is outside prediction geometry")
    result = result.clone()
    result[:, int(bos_index)] = False
    return result


def _source_ids(value: Sequence[str] | None, *, rows: int) -> tuple[str, ...] | None:
    if value is None:
        return None
    result = tuple(str(item) for item in value)
    if len(result) != rows or len(set(result)) != rows:
        raise AnalysisError("source IDs must be unique and align with prediction rows")
    return result


def score_method(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    bos_index: int = 0,
) -> dict[str, Any]:
    """Score absolute post-BOS accuracy and exact clips per source record."""

    pred = _predictions(prediction, label="prediction")
    target = _truth(truth, shape=tuple(pred.shape))
    valid = _mask(valid_mask, shape=tuple(pred.shape), bos_index=bos_index)
    correct = pred.eq(target) & valid
    exposure = valid.sum(dim=1).to(dtype=torch.long)
    if bool(exposure.le(0).any().item()):
        raise AnalysisError("every source record must expose at least one scored position")
    exact = (correct | ~valid).all(dim=1)
    counts = correct.sum(dim=1).to(dtype=torch.long)
    return {
        "records": int(pred.shape[0]),
        "positions": int(pred.shape[1]),
        "scored_tokens": int(exposure.sum().item()),
        "correct_tokens": int(counts.sum().item()),
        "token_errors": int(exposure.sum().item() - counts.sum().item()),
        "token_accuracy": float(counts.sum().item()) / float(exposure.sum().item()),
        "exact_records": int(exact.sum().item()),
        "exact_rate": float(exact.float().mean().item()),
        "correct_by_record": counts,
        "exposure_by_record": exposure,
        "exact_by_record": exact,
        "correct_mask": correct,
        "valid_mask": valid,
    }


def _intervals(
    *,
    stage_correct: torch.Tensor,
    base_correct: torch.Tensor,
    exposure: torch.Tensor,
    stage_exact: torch.Tensor,
    base_exact: torch.Tensor,
    broken: torch.Tensor,
    improved: torch.Tensor,
    seed: int,
    draws: int,
    one_sided_alpha: float,
) -> dict[str, Any]:
    exposure_values = exposure.tolist()
    stage_values = stage_correct.tolist()
    base_values = base_correct.tolist()
    broken_values = broken.sum(dim=1).tolist()
    improved_values = improved.sum(dim=1).tolist()
    base_error_values = [int(exposure.item()) - int(value) for exposure, value in zip(exposure, base_correct)]
    return {
        "token_delta": scorer.paired_token_delta(
            stage_values,
            base_values,
            exposure_values,
            seed=seed,
            draws=draws,
            one_sided_alpha=one_sided_alpha,
        ),
        "exact_delta": scorer.paired_exact_cp(
            stage_exact.tolist(),
            base_exact.tolist(),
            alpha=one_sided_alpha,
        ),
        "broken_per_exposed_token": scorer.bootstrap_ratio(
            broken_values,
            exposure_values,
            seed=seed,
            draws=draws,
            one_sided_alpha=one_sided_alpha,
        ),
        "improved_per_exposed_token": scorer.bootstrap_ratio(
            improved_values,
            exposure_values,
            seed=seed,
            draws=draws,
            one_sided_alpha=one_sided_alpha,
        ),
        "broken_given_baseline_correct": scorer.bootstrap_ratio(
            broken_values,
            base_values,
            seed=seed,
            draws=draws,
            one_sided_alpha=one_sided_alpha,
        ),
        "improved_given_baseline_error": scorer.bootstrap_ratio(
            improved_values,
            base_error_values,
            seed=seed,
            draws=draws,
            one_sided_alpha=one_sided_alpha,
        ),
    }


def paired_stage_analysis(
    baseline_prediction: torch.Tensor,
    stage_prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    bos_index: int = 0,
    source_ids: Sequence[str] | None = None,
    domain: str | None = None,
    stage: str | int | None = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    one_sided_alpha: float = DEFAULT_ONE_SIDED_ALPHA,
) -> dict[str, Any]:
    """Score paired baseline/stage outcomes and source-paired intervals."""

    baseline = _predictions(baseline_prediction, label="baseline prediction")
    stage_value = _predictions(stage_prediction, label="stage prediction")
    if tuple(stage_value.shape) != tuple(baseline.shape):
        raise AnalysisError("baseline and stage prediction geometry differs")
    target = _truth(truth, shape=tuple(baseline.shape))
    valid = _mask(valid_mask, shape=tuple(baseline.shape), bos_index=bos_index)
    ids = _source_ids(source_ids, rows=int(baseline.shape[0]))
    base_correct_mask = baseline.eq(target) & valid
    stage_correct_mask = stage_value.eq(target) & valid
    broken = base_correct_mask & ~stage_correct_mask
    improved = ~base_correct_mask & stage_correct_mask
    retained_correct = base_correct_mask & stage_correct_mask
    base_score = score_method(baseline, target, valid_mask=valid, bos_index=bos_index)
    stage_score = score_method(stage_value, target, valid_mask=valid, bos_index=bos_index)
    intervals = _intervals(
        stage_correct=stage_score["correct_by_record"],
        base_correct=base_score["correct_by_record"],
        exposure=stage_score["exposure_by_record"],
        stage_exact=stage_score["exact_by_record"],
        base_exact=base_score["exact_by_record"],
        broken=broken,
        improved=improved,
        seed=int(bootstrap_seed),
        draws=int(bootstrap_draws),
        one_sided_alpha=float(one_sided_alpha),
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "domain": domain,
        "stage": None if stage is None else str(stage),
        "records": int(baseline.shape[0]),
        "positions": int(baseline.shape[1]),
        "source_order_sha256": None if ids is None else source_order_digest(ids),
        "baseline": {key: value for key, value in base_score.items() if not isinstance(value, torch.Tensor)},
        "stage_score": {key: value for key, value in stage_score.items() if not isinstance(value, torch.Tensor)},
        "paired": {
            "broken_tokens": int(broken.sum().item()),
            "improved_tokens": int(improved.sum().item()),
            "retained_correct_tokens": int(retained_correct.sum().item()),
            "baseline_correct_tokens": int(base_correct_mask.sum().item()),
            "stage_correct_tokens": int(stage_correct_mask.sum().item()),
            "baseline_error_tokens": int((valid & ~base_correct_mask).sum().item()),
            "stage_error_tokens": int((valid & ~stage_correct_mask).sum().item()),
            "broken_exact_records": int((base_score["exact_by_record"] & ~stage_score["exact_by_record"]).sum().item()),
            "improved_exact_records": int((~base_score["exact_by_record"] & stage_score["exact_by_record"]).sum().item()),
            "unchanged_exact_records": int((base_score["exact_by_record"] == stage_score["exact_by_record"]).sum().item()),
        },
        "intervals": intervals,
        "uncertainty": {
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_draws": int(bootstrap_draws),
            "one_sided_alpha": float(one_sided_alpha),
            "interpretation": "all intervals are source-record paired; exact records use paired discordance Clopper-Pearson and token quantities use the validated source-record bootstrap",
        },
        "_raw": {
            "baseline_correct_by_record": base_score["correct_by_record"],
            "stage_correct_by_record": stage_score["correct_by_record"],
            "exposure_by_record": stage_score["exposure_by_record"],
            "baseline_exact_by_record": base_score["exact_by_record"],
            "stage_exact_by_record": stage_score["exact_by_record"],
            "broken_by_token": broken,
            "improved_by_token": improved,
        },
    }
    return result


def absolute_stage_summary(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    bos_index: int = 0,
    domain: str | None = None,
    stage: str | int | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready absolute score for a named domain/stage."""

    score = score_method(prediction, truth, valid_mask=valid_mask, bos_index=bos_index)
    return {
        "schema": SCHEMA,
        "domain": domain,
        "stage": None if stage is None else str(stage),
        "score": {key: value for key, value in score.items() if not isinstance(value, torch.Tensor)},
    }


def json_ready(value: Any) -> Any:
    """Convert an analysis result's private tensors into JSON-safe values."""

    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return [bool(item) for item in value.reshape(-1).tolist()]
        return [item.item() for item in value.reshape(-1).tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items() if key != "_raw"}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


__all__ = [
    "AnalysisError",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_ONE_SIDED_ALPHA",
    "SCHEMA",
    "absolute_stage_summary",
    "json_ready",
    "paired_stage_analysis",
    "score_method",
    "source_order_digest",
]
