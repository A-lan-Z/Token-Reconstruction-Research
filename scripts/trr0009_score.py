"""Truth-gated, per-cell TRR-0009 scoring primitives.

The public functions consume materialized predictions and truth tensors supplied
by the owner-gated score entry point.  They never discover or open private
truth on their own.  Every endpoint and uncertainty result remains within its
single domain/target cell; target conditions are never pooled.
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

from scripts import trr0009_eval_contract as contract

SCORE_SCHEMA = "token-reconstruction.trr0009-score.v1"
FIT_FREQUENCY_BINS = (
    "unseen_0",
    "seen_1_4",
    "seen_5_9",
    "seen_10_49",
    "seen_50_plus",
)


class ScoreError(contract.ContractError):
    pass


def _cp_lower(successes: int, trials: int, alpha_component: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials or not 0.0 < alpha_component < 1.0:
        raise ScoreError("invalid Clopper-Pearson lower-bound inputs")
    if successes == 0:
        return 0.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(alpha_component, successes, trials - successes + 1))
    except ImportError as exc:
        raise ScoreError("scipy is required for exact confidence bounds") from exc


def _cp_upper(successes: int, trials: int, alpha_component: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials or not 0.0 < alpha_component < 1.0:
        raise ScoreError("invalid Clopper-Pearson upper-bound inputs")
    if successes == trials:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(1.0 - alpha_component, successes + 1, trials - successes))
    except ImportError as exc:
        raise ScoreError("scipy is required for exact confidence bounds") from exc


def clopper_pearson(successes: int, trials: int, *, alpha: float = 0.025) -> dict[str, float | int]:
    """Two-sided CP interval with two component tails alpha/2."""

    if not 0.0 < float(alpha) < 1.0:
        raise ScoreError("composite CP alpha must be in (0,1)")
    component = float(alpha) / 2.0
    return {
        "lower": _cp_lower(int(successes), int(trials), component),
        "upper": _cp_upper(int(successes), int(trials), component),
        "alpha": float(alpha),
        "component_alpha": component,
        "successes": int(successes),
        "trials": int(trials),
    }


def _bootstrap_interval(values: Sequence[float], *, seed: int, draws: int, one_sided_alpha: float) -> dict[str, Any]:
    if not values:
        raise ScoreError("cannot bootstrap an empty record set")
    if draws <= 0 or not 0.0 < one_sided_alpha < 1.0:
        raise ScoreError("invalid bootstrap settings")
    rng = random.Random(int(seed))
    n = len(values)
    estimates = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(int(draws))]
    estimates.sort()
    lower_index = max(0, min(len(estimates) - 1, int(math.floor(float(one_sided_alpha) * len(estimates)))))
    upper_index = max(0, min(len(estimates) - 1, int(math.ceil((1.0 - float(one_sided_alpha)) * len(estimates)) - 1)))
    return {
        "status": "COMPUTED",
        "lower": float(estimates[lower_index]),
        "upper": float(estimates[upper_index]),
        "draws": int(draws),
        "seed": int(seed),
        "one_sided_alpha": float(one_sided_alpha),
        "one_sided_confidence": float(1.0 - one_sided_alpha),
        "unit": "source_record",
    }


def _bootstrap_ratio(
    record_correct: Sequence[int],
    record_exposed: Sequence[int],
    *,
    seed: int,
    draws: int,
    one_sided_alpha: float,
) -> dict[str, Any]:
    if len(record_correct) != len(record_exposed) or not record_correct:
        raise ScoreError("ratio bootstrap record geometry changed")
    if any(int(c) < 0 or int(e) < 0 or int(c) > int(e) for c, e in zip(record_correct, record_exposed)):
        raise ScoreError("ratio bootstrap counts are invalid")
    total_exposed = sum(int(x) for x in record_exposed)
    if total_exposed <= 0:
        return {"status": "NOT_COMPUTED", "reason": "zero_exposed_tokens", "unit": "source_record"}
    exposed_sources = sum(int(x) > 0 for x in record_exposed)
    if exposed_sources < 32:
        return {
            "status": "UNKNOWN",
            "reason": "fewer_than_32_exposed_source_records",
            "exposed_source_records": exposed_sources,
            "exposed_tokens": total_exposed,
            "unit": "source_record",
        }
    rng = random.Random(int(seed))
    n = len(record_correct)
    estimates: list[float] = []
    for _ in range(int(draws)):
        indices = [rng.randrange(n) for _ in range(n)]
        denominator = sum(int(record_exposed[i]) for i in indices)
        numerator = sum(int(record_correct[i]) for i in indices)
        if denominator:
            estimates.append(float(numerator) / float(denominator))
    if not estimates:
        return {"status": "UNKNOWN", "reason": "bootstrap_zero_exposure", "unit": "source_record"}
    estimates.sort()
    lower_index = max(0, min(len(estimates) - 1, int(math.floor(float(one_sided_alpha) * len(estimates)))))
    upper_index = max(0, min(len(estimates) - 1, int(math.ceil((1.0 - float(one_sided_alpha)) * len(estimates)) - 1)))
    return {
        "status": "COMPUTED",
        "estimate": float(sum(int(x) for x in record_correct) / total_exposed),
        "lower": float(estimates[lower_index]),
        "upper": float(estimates[upper_index]),
        "draws": int(draws),
        "seed": int(seed),
        "one_sided_alpha": float(one_sided_alpha),
        "one_sided_confidence": float(1.0 - one_sided_alpha),
        "unit": "source_record",
        "exposed_source_records": exposed_sources,
        "exposed_tokens": total_exposed,
    }


def frequency_bin(count: int) -> str:
    value = int(count)
    if value <= 0:
        return "unseen_0"
    if value <= 4:
        return "seen_1_4"
    if value <= 9:
        return "seen_5_9"
    if value <= 49:
        return "seen_10_49"
    return "seen_50_plus"


def normalize_frequency_reference(value: Mapping[str, Any]) -> dict[int, int]:
    """Extract a sparse public fit-frequency map; no truth/source loading."""

    raw: Any = value
    if isinstance(value.get("frequency_references"), Mapping):
        raw = value["frequency_references"].get("enriched")
    if not isinstance(raw, Mapping):
        raise ScoreError("public fitting-frequency reference lacks an enriched map")
    result: dict[int, int] = {}
    for token, count in raw.items():
        try:
            token_id = int(token)
            frequency = int(count)
        except (TypeError, ValueError) as exc:
            raise ScoreError("fitting-frequency map contains a non-integer entry") from exc
        if not 0 <= token_id < contract.VOCABULARY_SIZE or frequency <= 0:
            raise ScoreError("fitting-frequency map entry is out of range")
        result[token_id] = frequency
    return result


def _as_prediction_map(predictions: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
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
        raise ScoreError(f"truth tensor is absent for {cell_id}")
    # A single tensor is useful only to a direct unit caller.  Full scoring
    # still validates every cell independently and never pools conditions.
    return torch.as_tensor(truth)


def _cell_score(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    frequency_counts: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    prediction = torch.as_tensor(prediction, dtype=torch.long).detach().cpu().contiguous()
    truth = torch.as_tensor(truth, dtype=torch.long).detach().cpu().contiguous()
    if prediction.ndim != 2 or truth.shape != prediction.shape or prediction.shape[1] != contract.STORED_SEQUENCE_TOKENS:
        raise ScoreError("prediction/truth geometry changed")
    contract.validate_prediction_tensor(prediction, records=int(prediction.shape[0]))
    if truth[:, 0].ne(contract.BOS_TOKEN_ID).any().item():
        raise ScoreError("truth BOS column changed")
    if truth[:, 1:].lt(0).any().item() or truth[:, 1:].ge(contract.VOCABULARY_SIZE).any().item():
        raise ScoreError("truth contains invalid active token IDs")
    token_correct = prediction[:, 1:].eq(truth[:, 1:])
    record_exact = token_correct.all(dim=1)
    token_total = int(token_correct.numel())
    record_token_accuracy = token_correct.to(torch.float32).mean(dim=1)
    result: dict[str, Any] = {
        "records": int(prediction.shape[0]),
        "scored_post_bos_tokens": token_total,
        "token_correct": int(token_correct.sum().item()),
        "token_errors": token_total - int(token_correct.sum().item()),
        "token_accuracy": float(token_correct.to(torch.float32).mean().item()),
        "exact_correct": int(record_exact.sum().item()),
        "exact_errors": int(prediction.shape[0] - int(record_exact.sum().item())),
        "exact_rate": float(record_exact.to(torch.float32).mean().item()),
        "record_exact_mask": record_exact,
        "token_correct_mask": token_correct,
        "record_token_accuracy": record_token_accuracy,
    }
    if frequency_counts is None:
        return result
    counts = torch.zeros_like(truth[:, 1:], dtype=torch.long)
    for token_id, count in frequency_counts.items():
        counts[truth[:, 1:] == int(token_id)] = int(count)
    strata: dict[str, dict[str, Any]] = {}
    for stratum in FIT_FREQUENCY_BINS:
        exposed = torch.zeros_like(token_correct, dtype=torch.bool)
        for row in range(int(truth.shape[0])):
            for position in range(contract.SCORED_POST_BOS_TOKENS):
                if frequency_bin(int(counts[row, position])) == stratum:
                    exposed[row, position] = True
        record_exposed = exposed.sum(dim=1).to(torch.long)
        record_correct = (token_correct & exposed).sum(dim=1).to(torch.long)
        strata[stratum] = {
            "record_correct": record_correct,
            "record_exposed": record_exposed,
            "correct_tokens": int(record_correct.sum().item()),
            "exposed_tokens": int(record_exposed.sum().item()),
            "exposed_source_records": int((record_exposed > 0).sum().item()),
        }
    result["frequency_strata"] = strata
    return result


def _serialize_frequency_summary(
    score: Mapping[str, Any], *, seed: int, draws: int, one_sided_alpha: float = 0.05
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index, stratum in enumerate(FIT_FREQUENCY_BINS):
        row = score["frequency_strata"][stratum]
        ratio = _bootstrap_ratio(
            row["record_correct"].tolist(), row["record_exposed"].tolist(),
            seed=seed + index, draws=draws, one_sided_alpha=one_sided_alpha,
        )
        rows[stratum] = {
            "correct_tokens": int(row["correct_tokens"]),
            "exposed_tokens": int(row["exposed_tokens"]),
            "exposed_source_records": int(row["exposed_source_records"]),
            "token_accuracy": (
                float(row["correct_tokens"]) / float(row["exposed_tokens"])
                if int(row["exposed_tokens"]) else None
            ),
            "interval": ratio,
        }
    return rows


def paired_contrast(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    bootstrap_seed: int = 9009,
    bootstrap_draws: int = 10000,
    exact_alpha: float = 0.025,
    token_one_sided_alpha: float = 0.025,
    frequency_one_sided_alpha: float = 0.05,
    safeguard_token_one_sided_alpha: float = 0.05,
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
    candidate_record_tokens = torch.as_tensor(candidate.get("record_token_accuracy", candidate_tokens.to(torch.float32).mean(dim=1)), dtype=torch.float32).reshape(-1)
    reference_record_tokens = torch.as_tensor(reference.get("record_token_accuracy", reference_tokens.to(torch.float32).mean(dim=1)), dtype=torch.float32).reshape(-1)
    token_delta = candidate_record_tokens - reference_record_tokens
    if token_delta.numel() != exact_delta.numel():
        raise ScoreError("record-level token deltas changed")
    records = int(exact_delta.numel())
    gains = int((exact_delta > 0).sum().item())
    losses = int((exact_delta < 0).sum().item())
    gain_rate = clopper_pearson(gains, records, alpha=exact_alpha)
    loss_rate = clopper_pearson(losses, records, alpha=exact_alpha)
    result: dict[str, Any] = {
        "records": records,
        "exact_gains": gains,
        "exact_losses": losses,
        "exact_ties": records - gains - losses,
        "exact_net_rate": float((gains - losses) / records),
        "exact_net_cp_lower_bound": float(gain_rate["lower"] - loss_rate["upper"]),
        "exact_gain_rate_cp": gain_rate,
        "exact_loss_rate_cp": loss_rate,
        "token_net_rate": float(token_delta.mean().item()),
        # Keep the primary .025 interval and emit the wider .05 harm
        # safeguard from the identical seeded bootstrap draws.  The decision
        # layer must never reuse the primary interval for the safeguard.
        "token_net_bootstrap": _bootstrap_interval(token_delta.tolist(), seed=bootstrap_seed, draws=bootstrap_draws, one_sided_alpha=token_one_sided_alpha),
        "token_net_bootstrap_harm": _bootstrap_interval(token_delta.tolist(), seed=bootstrap_seed, draws=bootstrap_draws, one_sided_alpha=safeguard_token_one_sided_alpha),
        "bootstrap_unit": "source_record",
        "exact_alpha": float(exact_alpha),
        "token_one_sided_alpha": float(token_one_sided_alpha),
        "safeguard_token_one_sided_alpha": float(safeguard_token_one_sided_alpha),
    }
    if "frequency_strata" in candidate and "frequency_strata" in reference:
        freq: dict[str, Any] = {}
        for index, stratum in enumerate(FIT_FREQUENCY_BINS):
            left = candidate["frequency_strata"][stratum]
            right = reference["frequency_strata"][stratum]
            left_exposed = torch.as_tensor(left["record_exposed"], dtype=torch.long).reshape(-1)
            right_exposed = torch.as_tensor(right["record_exposed"], dtype=torch.long).reshape(-1)
            left_correct = torch.as_tensor(left["record_correct"], dtype=torch.long).reshape(-1)
            right_correct = torch.as_tensor(right["record_correct"], dtype=torch.long).reshape(-1)
            if not torch.equal(left_exposed, right_exposed):
                raise ScoreError(f"frequency exposure differs between paired methods: {stratum}")
            if left_correct.numel() != records:
                raise ScoreError(f"frequency record geometry changed: {stratum}")
            exposed = int(left_exposed.sum().item())
            exposed_sources = int((left_exposed > 0).sum().item())
            if exposed <= 0:
                interval: dict[str, Any] = {"status": "NOT_COMPUTED", "reason": "zero_exposed_tokens", "unit": "source_record"}
            elif exposed_sources < 32:
                interval = {"status": "UNKNOWN", "reason": "fewer_than_32_exposed_source_records", "exposed_source_records": exposed_sources, "exposed_tokens": exposed, "unit": "source_record"}
            else:
                delta_estimates: list[float] = []
                rng = random.Random(int(bootstrap_seed + 1000 + index))
                n = records
                for _ in range(int(bootstrap_draws)):
                    indices = [rng.randrange(n) for _ in range(n)]
                    denom = sum(int(left_exposed[i]) for i in indices)
                    if denom:
                        delta_estimates.append(float(sum(int(left_correct[i]) - int(right_correct[i]) for i in indices)) / float(denom))
                delta_estimates.sort()
                lower_index = max(0, min(len(delta_estimates) - 1, int(math.floor(frequency_one_sided_alpha * len(delta_estimates)))))
                upper_index = max(0, min(len(delta_estimates) - 1, int(math.ceil((1.0 - frequency_one_sided_alpha) * len(delta_estimates)) - 1)))
                interval = {
                    "status": "COMPUTED",
                    "estimate": float((int(left_correct.sum().item()) - int(right_correct.sum().item())) / exposed),
                    "lower": float(delta_estimates[lower_index]),
                    "upper": float(delta_estimates[upper_index]),
                    "draws": int(bootstrap_draws),
                    "seed": int(bootstrap_seed + 1000 + index),
                    "one_sided_alpha": float(frequency_one_sided_alpha),
                    "unit": "source_record",
                    "exposed_source_records": exposed_sources,
                    "exposed_tokens": exposed,
                }
            freq[stratum] = {
                "correct_delta_tokens": int(left_correct.sum().item() - right_correct.sum().item()),
                "exposed_tokens": exposed,
                "exposed_source_records": exposed_sources,
                "interval": interval,
            }
        result["frequency_contrasts"] = freq
    return result


def score_predictions(
    predictions: Mapping[str, Any],
    truth: Mapping[str, Any] | torch.Tensor,
    *,
    frequency_counts: Mapping[int, int] | None = None,
    bootstrap_seed: int = 9009,
    bootstrap_draws: int = 10000,
    exact_alpha: float = 0.025,
    token_one_sided_alpha: float = 0.025,
    safeguard_token_one_sided_alpha: float = 0.05,
) -> dict[str, Any]:
    if not 0.0 < float(safeguard_token_one_sided_alpha) < 1.0:
        raise ScoreError("safeguard token alpha must be in (0,1)")
    matrix = _as_prediction_map(predictions)
    if set(matrix) != set(contract.METHOD_ORDER):
        raise ScoreError("score matrix must contain exactly the frozen four methods")
    raw_scores: dict[str, dict[str, dict[str, Any]]] = {}
    for method_id in contract.METHOD_ORDER:
        if set(matrix[method_id]) != set(contract.CELL_ORDER):
            raise ScoreError(f"score matrix cells changed for {method_id}")
        raw_scores[method_id] = {}
        for cell_id in contract.CELL_ORDER:
            raw_scores[method_id][cell_id] = _cell_score(
                matrix[method_id][cell_id], _truth_for_cell(truth, cell_id), frequency_counts=frequency_counts
            )
    serialized_scores: dict[str, dict[str, Any]] = {}
    for method_index, method_id in enumerate(contract.METHOD_ORDER):
        serialized_scores[method_id] = {}
        for cell_index, cell_id in enumerate(contract.CELL_ORDER):
            score = raw_scores[method_id][cell_id]
            row = {key: value for key, value in score.items() if not torch.is_tensor(value) and key != "frequency_strata"}
            if frequency_counts is not None:
                row["frequency_strata"] = _serialize_frequency_summary(score, seed=bootstrap_seed + method_index * 100 + cell_index * 10, draws=bootstrap_draws)
            serialized_scores[method_id][cell_id] = row
    def contrast(left_id: str, right_id: str, seed_offset: int) -> dict[str, Any]:
        return {
            cell_id: paired_contrast(
                raw_scores[left_id][cell_id], raw_scores[right_id][cell_id],
                bootstrap_seed=bootstrap_seed + seed_offset + index,
                bootstrap_draws=bootstrap_draws, exact_alpha=exact_alpha,
                token_one_sided_alpha=token_one_sided_alpha,
                frequency_one_sided_alpha=safeguard_token_one_sided_alpha,
                safeguard_token_one_sided_alpha=safeguard_token_one_sided_alpha,
            )
            for index, cell_id in enumerate(contract.CELL_ORDER)
        }
    primary_contrast = contrast(contract.PRIMARY_METHOD_ID, contract.PRIMARY_CONTROL_METHOD_ID, 300)
    rare_rows: dict[str, dict[str, Any]] = {}
    rare_failures: list[dict[str, Any]] = []
    rare_unknown: list[dict[str, Any]] = []
    for cell_id in contract.CELL_ORDER:
        for stratum, gate_bin in (("unseen_0", "0"), ("seen_1_4", "1-4")):
            interval = primary_contrast[cell_id].get("frequency_contrasts", {}).get(stratum, {}).get("interval", {"status": "NOT_COMPUTED"})
            status = str(interval.get("status"))
            row = {"cell_id": cell_id, "stratum": gate_bin, "frequency_bin": stratum, "harm_margin": 0.01, "interval": interval}
            rare_rows[f"{cell_id}::{gate_bin}"] = row
            if status == "COMPUTED" and float(interval.get("lower", -1.0)) < -0.01:
                rare_failures.append(row)
            elif status != "COMPUTED":
                rare_unknown.append(row)
    rare_status = "FAIL" if rare_failures else ("UNKNOWN" if rare_unknown else "PASS")
    return {
        "schema": SCORE_SCHEMA,
        "task_id": contract.TASK_ID,
        "method_order": list(contract.METHOD_ORDER),
        "cell_order": list(contract.CELL_ORDER),
        "primary": {"candidate": contract.PRIMARY_METHOD_ID, "control": contract.PRIMARY_CONTROL_METHOD_ID},
        "cell_scores": serialized_scores,
        "contrasts": {
            "candidate_vs_fixed": primary_contrast,
            "candidate_vs_unchanged": contrast(contract.PRIMARY_METHOD_ID, contract.UNCHANGED_METHOD_ID, 100),
            "candidate_vs_published_reference": contrast(contract.PRIMARY_METHOD_ID, contract.PUBLISHED_REFERENCE_METHOD_ID, 200),
        },
        "rare_token_safeguard": {"frequency_bins": ["0", "1-4"], "harm_margin": 0.01, "minimum_exposed_source_records": 32, "all_cells_required": True, "all_four_cells_required": True, "all_eight_gates_required": True, "status": rare_status, "rows": rare_rows, "failures": rare_failures, "unknown": rare_unknown},
        "confidence": {"exact_composite_alpha": float(exact_alpha), "exact_component_alpha": float(exact_alpha / 2.0), "token_one_sided_alpha": float(token_one_sided_alpha), "safeguard_token_one_sided_alpha": float(safeguard_token_one_sided_alpha)},
        "bootstrap": {"seed": int(bootstrap_seed), "draws": int(bootstrap_draws), "unit": "source_record"},
        "frequency_reference_bound": frequency_counts is not None,
        "truth_opened": True,
        "candidate_arrays_persisted": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print("TRR-0009 scorer CLI requires the owner-gated truth adapter; no direct truth opening is permitted.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
