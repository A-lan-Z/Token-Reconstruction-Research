"""Post hoc diagnostics for the TRR-0004 standalone decoder gap.

This module deliberately evaluates already-fitted states.  It does not train a
decoder, call a public model, generate candidates, or inspect target/private
truth.  The labels accepted here are the public fitting labels (for token
coverage statistics) and the disjoint public validation labels (for the
diagnostic metrics).

The main ablation is the fitted vocabulary bias.  Scale and output
normalization controls are included because they help determine whether a
change in the CE decoder is a calibration effect or a change in the learned
activation map.  Removing a bias from a fitted checkpoint is explicitly a
post hoc diagnostic; it is not equivalent to fitting a no-bias model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .inverse import ResidualAffineInverse
from .standalone_decoder import ResidualMLPTokenDecoder, TiedAffineTokenDecoder


BIAS_MODES = ("original", "zero")
FREQUENCY_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("unseen", 0, 0),
    ("singleton", 1, 1),
    ("rare_2_4", 2, 4),
    ("low_5_9", 5, 9),
    ("medium_10_49", 10, 49),
    ("frequent_50_plus", 50, None),
)


@dataclass(frozen=True)
class DiagnosticVariant:
    """One fixed post hoc evaluation control.

    ``scale`` is ``None`` for the model's frozen native scale.  All scales are
    positive, so a scale-only comparison with zero bias has an invariant
    argmax; the runner records that expected relationship rather than treating
    it as a new decoder.
    """

    variant_id: str
    bias_mode: Literal["original", "zero"] = "original"
    scale: float | None = None
    normalize_output: bool = True

    def validate(self) -> None:
        if not self.variant_id:
            raise ValueError("diagnostic variant needs an identifier")
        if self.bias_mode not in BIAS_MODES:
            raise ValueError(f"unknown bias mode: {self.bias_mode}")
        if self.scale is not None and (not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("diagnostic scale must be finite and positive")


DEFAULT_VARIANTS: tuple[DiagnosticVariant, ...] = (
    DiagnosticVariant("original", bias_mode="original"),
    DiagnosticVariant("vocab_bias_disabled", bias_mode="zero"),
    DiagnosticVariant("original_bias_scale_1", bias_mode="original", scale=1.0),
    DiagnosticVariant("no_bias_scale_1", bias_mode="zero", scale=1.0),
    DiagnosticVariant(
        "no_bias_output_normalization_disabled",
        bias_mode="zero",
        normalize_output=False,
    ),
)


def _as_cpu_long(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must contain integer token IDs")
    return value.detach().to(device="cpu", dtype=torch.long).contiguous()


def flatten_public_records(
    observations: torch.Tensor,
    truth: torch.Tensor,
    *,
    bos_token_id: int = 128000,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Flatten post-BOS validation observations and public labels.

    The returned record shape is retained so exact-record metrics can be
    calculated without writing per-position prediction artifacts.
    """

    if observations.ndim != 3 or observations.shape[0] <= 0 or observations.shape[1] <= 1:
        raise ValueError("observations must have shape [records, positions>1, hidden]")
    if truth.ndim != 2 or tuple(truth.shape) != tuple(observations.shape[:2]):
        raise ValueError("truth geometry must match observations")
    if not observations.dtype.is_floating_point or not torch.isfinite(observations).all().item():
        raise ValueError("observations must be finite floating point")
    truth_cpu = _as_cpu_long(truth, name="truth")
    if truth_cpu[:, 0].ne(bos_token_id).any().item():
        raise ValueError("public truth rows must begin with the declared BOS token")
    return (
        observations[:, 1:, :].reshape(-1, observations.shape[-1]).contiguous(),
        truth_cpu[:, 1:].reshape(-1).contiguous(),
        (int(observations.shape[0]), int(observations.shape[1] - 1)),
    )


def flatten_public_labels(truth: torch.Tensor, *, bos_token_id: int = 128000) -> torch.Tensor:
    """Return post-BOS public fitting labels for coverage counting."""

    if truth.ndim != 2 or truth.shape[0] <= 0 or truth.shape[1] <= 1:
        raise ValueError("public fitting truth must have shape [records, positions>1]")
    truth_cpu = _as_cpu_long(truth, name="fit_truth")
    if truth_cpu[:, 0].ne(bos_token_id).any().item():
        raise ValueError("public fitting truth rows must begin with the declared BOS token")
    return truth_cpu[:, 1:].reshape(-1).contiguous()


def token_frequency(fit_labels: torch.Tensor, *, vocab_size: int) -> torch.Tensor:
    """Count public fitting token IDs, excluding BOS positions."""

    labels = _as_cpu_long(fit_labels, name="fit_labels").reshape(-1)
    if vocab_size <= 0 or labels.numel() == 0:
        raise ValueError("fit labels and vocabulary must be non-empty")
    if labels.lt(0).any().item() or labels.ge(vocab_size).any().item():
        raise ValueError("fit labels contain an out-of-range token ID")
    return torch.bincount(labels, minlength=vocab_size).to(torch.long)


def _frequency_index(counts: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    counts = counts.detach().to(device="cpu", dtype=torch.long)
    ids = _as_cpu_long(token_ids, name="token_ids")
    if ids.lt(0).any().item() or ids.ge(counts.shape[0]).any().item():
        raise ValueError("token IDs are outside the frequency table")
    values = counts.index_select(0, ids)
    result = torch.full_like(values, fill_value=-1)
    for index, (_name, lower, upper) in enumerate(FREQUENCY_BINS):
        mask = values.ge(lower)
        if upper is not None:
            mask &= values.le(upper)
        result[mask] = index
    if result.eq(-1).any().item():
        raise ValueError("frequency bins do not cover all counts")
    return result


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().reshape(-1).cpu()
    count = int(values.numel())
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95], dtype=values.dtype))
    return {
        "count": count,
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p05": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "max": float(values.max().item()),
    }


def _metrics(
    predictions: torch.Tensor,
    ranks: torch.Tensor,
    labels: torch.Tensor,
    *,
    record_shape: tuple[int, int],
) -> dict[str, Any]:
    predictions = _as_cpu_long(predictions, name="predictions").reshape(-1)
    labels = _as_cpu_long(labels, name="labels").reshape(-1)
    ranks = ranks.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if predictions.shape != labels.shape or ranks.shape != labels.shape:
        raise ValueError("prediction, rank, and label rows must agree")
    records, positions = record_shape
    if predictions.numel() != records * positions:
        raise ValueError("prediction rows do not match record geometry")
    correct = predictions.eq(labels)
    exact = correct.reshape(records, positions).all(dim=1)
    return {
        "records": records,
        "positions_per_record": positions,
        "examples": int(labels.numel()),
        "token_accuracy": float(correct.float().mean().item()),
        "correct_tokens": int(correct.sum().item()),
        "exact_records": int(exact.sum().item()),
        "mean_true_rank": float(ranks.mean().item()),
        "median_true_rank": float(ranks.median().item()),
        "p90_true_rank": float(torch.quantile(ranks, torch.tensor(0.90)).item()),
        "top1": float(correct.float().mean().item()),
        "top5": float(ranks.le(5).float().mean().item()),
        "top10": float(ranks.le(10).float().mean().item()),
        "top50": float(ranks.le(50).float().mean().item()),
    }


def _group_metrics(
    predictions: torch.Tensor,
    ranks: torch.Tensor,
    labels: torch.Tensor,
    group_index: torch.Tensor,
    *,
    names: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    predictions = _as_cpu_long(predictions, name="predictions")
    labels = _as_cpu_long(labels, name="labels")
    ranks = ranks.detach().to(device="cpu", dtype=torch.float32)
    group_index = _as_cpu_long(group_index, name="group_index")
    if not (predictions.shape == labels.shape == ranks.shape == group_index.shape):
        raise ValueError("group diagnostic rows must agree")
    result: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        mask = group_index.eq(index)
        correct = predictions[mask].eq(labels[mask])
        result[name] = {
            "examples": int(mask.sum().item()),
            "token_accuracy": float(correct.float().mean().item()) if mask.any() else None,
            "mean_true_rank": float(ranks[mask].mean().item()) if mask.any() else None,
            "median_true_rank": float(ranks[mask].median().item()) if mask.any() else None,
            "correct_tokens": int(correct.sum().item()),
        }
    return result


def _raw_transformed(model: nn.Module, activation: torch.Tensor) -> torch.Tensor:
    """Reproduce a frozen decoder's pre-normalization residual output."""

    value = activation.float()
    if isinstance(model, TiedAffineTokenDecoder):
        return value + model.residual(value)
    if isinstance(model, ResidualMLPTokenDecoder):
        return value + model.up(F.gelu(model.down(value)))
    if isinstance(model, ResidualAffineInverse):
        return value + model.residual(value)
    raise TypeError(f"unsupported decoder type for post hoc diagnostic: {type(model)!r}")


def _native_bias(model: nn.Module, *, vocab_size: int, device: torch.device) -> torch.Tensor:
    value = getattr(model, "classifier_bias", None)
    if value is None:
        return torch.zeros(vocab_size, device=device, dtype=torch.float32)
    if value.ndim != 1 or value.shape[0] != vocab_size:
        raise ValueError("decoder vocabulary bias geometry changed")
    return value.detach().to(device=device, dtype=torch.float32)


def _variant_logits(
    model: nn.Module,
    activation: torch.Tensor,
    embedding_table: torch.Tensor,
    variant: DiagnosticVariant,
) -> torch.Tensor:
    variant.validate()
    native_scale = float(getattr(model, "logit_scale", 1.0))
    scale = native_scale if variant.scale is None else float(variant.scale)
    raw = _raw_transformed(model, activation)
    query = F.normalize(raw, dim=-1) if variant.normalize_output else raw
    if not torch.isfinite(query).all().item():
        raise ValueError("decoder post hoc query is non-finite")
    bias = _native_bias(model, vocab_size=int(embedding_table.shape[0]), device=query.device)
    if variant.bias_mode == "zero":
        bias = torch.zeros_like(bias)
    return query @ embedding_table.transpose(0, 1) * scale + bias


def _rank_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute competition rank (one plus the number of strictly larger logits)."""

    true_score = logits.gather(1, labels.reshape(-1, 1)).squeeze(1)
    return (logits > true_score.unsqueeze(1)).sum(dim=1).to(torch.long) + 1


def _evaluate_variant(
    model: nn.Module,
    observations: torch.Tensor,
    labels: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    variant: DiagnosticVariant,
    batch_size: int,
    record_shape: tuple[int, int],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("diagnostic batch size must be positive")
    if observations.ndim != 2 or labels.ndim != 1 or observations.shape[0] != labels.shape[0]:
        raise ValueError("validation observations and labels must be matching matrices")
    if embedding_table.ndim != 2 or observations.shape[1] != embedding_table.shape[1]:
        raise ValueError("embedding and observation hidden sizes do not agree")
    device = next(model.parameters()).device
    x = observations.to(device=device, dtype=torch.float32)
    y = labels.to(device=device, dtype=torch.long)
    embeddings = embedding_table.to(device=device, dtype=torch.float32)
    predictions: list[torch.Tensor] = []
    ranks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), batch_size):
            stop = min(start + batch_size, int(x.shape[0]))
            logits = _variant_logits(model, x[start:stop], embeddings, variant)
            predictions.append(logits.argmax(dim=-1).to(device="cpu", dtype=torch.long))
            ranks.append(_rank_from_logits(logits, y[start:stop]).to(device="cpu"))
    if not predictions:
        raise ValueError("diagnostic validation set is empty")
    pred = torch.cat(predictions).contiguous()
    rank = torch.cat(ranks).contiguous()
    return _metrics(pred, rank, labels, record_shape=record_shape), pred, rank


def _bias_diagnostics(
    model: nn.Module,
    *,
    validation_labels: torch.Tensor,
    baseline_predictions: torch.Tensor,
    baseline_ranks: torch.Tensor,
    counts: torch.Tensor,
    frequency_index: torch.Tensor,
) -> dict[str, Any]:
    bias = getattr(model, "classifier_bias", None)
    if bias is None:
        return {"present": False, "reason": "angular control has no vocabulary bias"}
    values = bias.detach().float().cpu().reshape(-1)
    vocab_size = int(values.shape[0])
    if counts.shape != (vocab_size,):
        raise ValueError("frequency table does not match vocabulary bias")
    result: dict[str, Any] = {
        "present": True,
        "vocab_size": vocab_size,
        "all_tokens": _summary(values),
        "by_frequency_bin": {},
        "validation_true": _summary(values.index_select(0, validation_labels)),
        "validation_predicted_original": _summary(values.index_select(0, baseline_predictions)),
        "validation_true_correct": _summary(
            values.index_select(0, validation_labels)[baseline_predictions.eq(validation_labels)]
        ),
        "validation_true_error": _summary(
            values.index_select(0, validation_labels)[baseline_predictions.ne(validation_labels)]
        ),
        "true_token_rank": _summary(baseline_ranks),
    }
    for _index, (name, lower, upper) in enumerate(FREQUENCY_BINS):
        # ``counts`` is indexed by vocabulary token, while
        # ``frequency_index`` is indexed by validation row. Keep the two
        # spaces separate here: this table describes the fitted vocabulary.
        count_values = counts
        mask = count_values.ge(lower)
        if upper is not None:
            mask &= count_values.le(upper)
        result["by_frequency_bin"][name] = {
            "token_count": int(mask.sum().item()),
            "fit_label_count": int(counts[mask].sum().item()),
            "bias": _summary(values[mask]),
        }
    result["validation_by_frequency_bin"] = _group_metrics(
        baseline_predictions,
        baseline_ranks,
        validation_labels,
        frequency_index,
        names=tuple(name for name, _lower, _upper in FREQUENCY_BINS),
    )
    return result


def _comparison(
    baseline_predictions: torch.Tensor,
    baseline_ranks: torch.Tensor,
    baseline_metrics: Mapping[str, Any],
    predictions: torch.Tensor,
    ranks: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    baseline_correct = baseline_predictions.eq(labels)
    current_correct = predictions.eq(labels)
    changed = predictions.ne(baseline_predictions)
    return {
        "prediction_changed_examples": int(changed.sum().item()),
        "prediction_changed_fraction": float(changed.float().mean().item()),
        "changed_to_correct": int((changed & ~baseline_correct & current_correct).sum().item()),
        "changed_from_correct": int((changed & baseline_correct & ~current_correct).sum().item()),
        "accuracy_delta": float(current_correct.float().mean().item() - baseline_metrics["token_accuracy"]),
        "mean_rank_delta": float((ranks.float() - baseline_ranks.float()).mean().item()),
        "better_rank_examples": int((ranks < baseline_ranks).sum().item()),
        "worse_rank_examples": int((ranks > baseline_ranks).sum().item()),
        "same_rank_examples": int((ranks == baseline_ranks).sum().item()),
    }


def diagnose_model(
    model: nn.Module,
    *,
    method_id: str,
    validation_observations: torch.Tensor,
    validation_labels: torch.Tensor,
    fit_labels: torch.Tensor,
    embedding_table: torch.Tensor,
    record_shape: tuple[int, int],
    batch_size: int = 128,
    variants: tuple[DiagnosticVariant, ...] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    """Evaluate a frozen decoder and post hoc controls on public validation.

    ``fit_labels`` must come from public fitting records and
    ``validation_labels`` from the disjoint public validation records.  The
    function never sees panel/private labels and all variants use one direct
    full-vocabulary projection; there is no candidate or A2 fallback.
    """

    if not variants or len({variant.variant_id for variant in variants}) != len(variants):
        raise ValueError("diagnostic variants must be non-empty and uniquely named")
    for variant in variants:
        variant.validate()
    validation_labels = _as_cpu_long(validation_labels, name="validation_labels").reshape(-1)
    fit_labels = _as_cpu_long(fit_labels, name="fit_labels").reshape(-1)
    if validation_observations.ndim != 2 or validation_observations.shape[0] != validation_labels.shape[0]:
        raise ValueError("validation observations and labels have different row counts")
    vocab_size = int(embedding_table.shape[0])
    counts = token_frequency(fit_labels, vocab_size=vocab_size)
    if validation_labels.lt(0).any().item() or validation_labels.ge(vocab_size).any().item():
        raise ValueError("validation labels contain an out-of-range token ID")
    validation_frequency = _frequency_index(counts, validation_labels)
    native_scale = float(getattr(model, "logit_scale", 1.0))
    variant_results: dict[str, Any] = {}
    prediction_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for variant in variants:
        metrics, predictions, ranks = _evaluate_variant(
            model,
            validation_observations,
            validation_labels,
            embedding_table,
            variant=variant,
            batch_size=batch_size,
            record_shape=record_shape,
        )
        metrics["seen_token_accuracy"] = _group_metrics(
            predictions,
            ranks,
            validation_labels,
            (counts.index_select(0, validation_labels) == 0).to(dtype=torch.long),
            names=("seen", "unseen"),
        )
        metrics["frequency_bin_metrics"] = _group_metrics(
            predictions,
            ranks,
            validation_labels,
            validation_frequency,
            names=tuple(name for name, _lower, _upper in FREQUENCY_BINS),
        )
        variant_results[variant.variant_id] = {
            "bias_mode": variant.bias_mode,
            "scale": native_scale if variant.scale is None else variant.scale,
            "scale_is_native": variant.scale is None,
            "normalize_output": variant.normalize_output,
            "metrics": metrics,
        }
        prediction_cache[variant.variant_id] = (predictions, ranks)
    baseline_id = variants[0].variant_id
    baseline_predictions, baseline_ranks = prediction_cache[baseline_id]
    comparisons: dict[str, Any] = {}
    for variant in variants:
        predictions, ranks = prediction_cache[variant.variant_id]
        comparisons[variant.variant_id] = _comparison(
            baseline_predictions,
            baseline_ranks,
            variant_results[baseline_id]["metrics"],
            predictions,
            ranks,
            validation_labels,
        )
    bias = _bias_diagnostics(
        model,
        validation_labels=validation_labels,
        baseline_predictions=baseline_predictions,
        baseline_ranks=baseline_ranks,
        counts=counts,
        frequency_index=validation_frequency,
    )
    pairwise: dict[str, Any] = {}
    for left_id, (left_predictions, left_ranks) in prediction_cache.items():
        for right_id, (right_predictions, right_ranks) in prediction_cache.items():
            if left_id >= right_id:
                continue
            pair_key = f"{left_id}__vs__{right_id}"
            pairwise[pair_key] = {
                "prediction_changed_examples": int(left_predictions.ne(right_predictions).sum().item()),
                "same_rank_examples": int(left_ranks.eq(right_ranks).sum().item()),
                "mean_rank_delta_left_minus_right": float(
                    (left_ranks.float() - right_ranks.float()).mean().item()
                ),
            }
    return {
        "method_id": method_id,
        "native_logit_scale": native_scale,
        "variant_order": [variant.variant_id for variant in variants],
        "baseline_variant": baseline_id,
        "variants": variant_results,
        "comparisons_to_baseline": comparisons,
        "pairwise_variant_comparisons": pairwise,
        "vocabulary_coverage": {
            "fit_examples": int(fit_labels.numel()),
            "fit_unique_tokens": int(counts.gt(0).sum().item()),
            "vocab_size": vocab_size,
            "validation_examples": int(validation_labels.numel()),
            "validation_unique_tokens": int(validation_labels.unique().numel()),
            "validation_seen_examples": int(counts.index_select(0, validation_labels).gt(0).sum().item()),
            "validation_unseen_examples": int(counts.index_select(0, validation_labels).eq(0).sum().item()),
            "frequency_bins": {
                name: {
                    "lower_fit_count": lower,
                    "upper_fit_count": upper,
                    "validation_examples": int(validation_frequency.eq(index).sum().item()),
                }
                for index, (name, lower, upper) in enumerate(FREQUENCY_BINS)
            },
        },
        "bias_diagnostics": bias,
        "interpretation_limits": [
            "These are post hoc controls on a fitted checkpoint; zeroing vocabulary bias is not a no-bias fit.",
            "Public validation labels are auxiliary development labels and are not an independent confirmation set.",
            "All accuracy and rank values come from one direct full-vocabulary projection with no A2 fallback.",
        ],
    }


def expected_scale_invariance(
    original: Mapping[str, Any], scaled: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a machine-readable check for a zero-bias positive-scale pair."""

    if original.get("bias_mode") != "zero" or scaled.get("bias_mode") != "zero":
        return {"applicable": False, "reason": "both variants must disable bias"}
    if not bool(original.get("normalize_output")) or not bool(scaled.get("normalize_output")):
        return {"applicable": False, "reason": "both variants must use output normalization"}
    original_scale = float(original["scale"])
    scaled_scale = float(scaled["scale"])
    if original_scale <= 0 or scaled_scale <= 0:
        return {"applicable": False, "reason": "scales must be positive"}
    return {
        "applicable": True,
        "expected_argmax_invariant": True,
        "scale_ratio": scaled_scale / original_scale,
    }

