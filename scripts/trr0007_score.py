"""Score TRR-0007 only after the complete public matrix is frozen.

The scorer first calls the public gate, then reads one metadata-only truth
binding header, and only then opens the two-domain truth sidecar exactly once.
It reports all methods and cells separately, the four direct factorial
edges, descriptive reference/interaction contrasts, a bounded first-32
A1-to-A2 anchor with decoder-vs-A2 gaps, cost denominators, and
preregistered point-versus-margin decisions.  No token labels are copied
into result artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from safetensors import safe_open
import torch
from scipy.stats import beta

from scripts import trr0007_eval_contract as contract
from scripts import trr0007_eval_gate as gate
from scripts import trr0007_support_diagnostics as support


class ScoreError(contract.ContractError):
    """Raised when scoring inputs or post-gate truth access is invalid."""


BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 5007
TOKEN_TAIL_ALPHA = 0.05 / 64.0
EXACT_COMPONENT_ALPHA = 0.05 / 128.0
USEFUL_TOKEN_PP = 1.0
USEFUL_EXACT_PP = 5.0

# These are the only corrected primary comparisons.  Each tuple is
# (reported edge, left method, right method), with left-minus-right as the
# improvement direction.
PRIMARY_FACTORIAL_EDGES = (
    (
        "support_at_trained_diagonal",
        "improved_public_bank__trained_diagonal",
        "current_enriched__trained_diagonal",
    ),
    (
        "support_at_residual_mlp512",
        "improved_public_bank__residual_mlp512",
        "current_enriched__residual_mlp512",
    ),
    (
        "capacity_on_current_enriched",
        "current_enriched__residual_mlp512",
        "current_enriched__trained_diagonal",
    ),
    (
        "capacity_on_improved_public_bank",
        "improved_public_bank__residual_mlp512",
        "improved_public_bank__trained_diagonal",
    ),
)
DECODER_METHODS = (contract.REFERENCE_METHOD_ID, *contract.STUDENT_METHOD_IDS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        return contract.load_json(Path(path).expanduser().resolve(), description=description)
    except contract.ContractError as exc:
        raise ScoreError(str(exc)) from exc


def _cp_lower(successes: int, trials: int, alpha: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials or not 0 < alpha < 1:
        raise ScoreError("invalid Clopper-Pearson inputs")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def _cp_upper(successes: int, trials: int, alpha: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials or not 0 < alpha < 1:
        raise ScoreError("invalid Clopper-Pearson inputs")
    if successes == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def clopper_pearson_gain_loss(
    gains: int,
    losses: int,
    trials: int,
    *,
    component_alpha: float = EXACT_COMPONENT_ALPHA,
) -> dict[str, Any]:
    """Bound paired exact-record gain minus loss with corrected directions."""

    if gains < 0 or losses < 0 or gains + losses > trials:
        raise ScoreError("paired exact discordances are invalid")
    gain_lower = _cp_lower(gains, trials, component_alpha)
    gain_upper = _cp_upper(gains, trials, component_alpha)
    loss_lower = _cp_lower(losses, trials, component_alpha)
    loss_upper = _cp_upper(losses, trials, component_alpha)
    return {
        "gains": int(gains),
        "losses": int(losses),
        "trials": int(trials),
        "point": float((gains - losses) / trials),
        "gain_lower": gain_lower,
        "gain_upper": gain_upper,
        "loss_lower": loss_lower,
        "loss_upper": loss_upper,
        "lower": float(gain_lower - loss_upper),
        "upper": float(gain_upper - loss_lower),
        "component_alpha": float(component_alpha),
        "interpretation": "material harm is evidenced when upper < negative margin; harm is excluded only when lower >= negative margin",
    }


def _bootstrap_samples(
    left: np.ndarray,
    right: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or left.size <= 0:
        raise ScoreError("paired bootstrap vectors are invalid")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ScoreError("paired bootstrap vectors contain non-finite values")
    if draws <= 0:
        raise ScoreError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    # Keep the same index matrix for every paired contrast invocation.
    indices = rng.integers(0, left.size, size=(draws, left.size))
    return (left[indices] - right[indices]).mean(axis=1)


def paired_contrast(
    left_token_accuracy: np.ndarray,
    right_token_accuracy: np.ndarray,
    left_exact: np.ndarray,
    right_exact: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
    primary: bool = True,
    exact_cp: bool = True,
) -> dict[str, Any]:
    """Return point estimates, descriptive 95% intervals, and one-sided bounds."""

    left_token_accuracy = np.asarray(left_token_accuracy, dtype=np.float64)
    right_token_accuracy = np.asarray(right_token_accuracy, dtype=np.float64)
    left_exact_raw = np.asarray(left_exact)
    right_exact_raw = np.asarray(right_exact)
    if left_token_accuracy.shape != right_token_accuracy.shape or left_token_accuracy.ndim != 1:
        raise ScoreError("paired token arrays differ")
    if left_exact_raw.shape != right_exact_raw.shape or left_exact_raw.shape != left_token_accuracy.shape:
        raise ScoreError("paired exact arrays differ")
    if exact_cp:
        left_exact = left_exact_raw.astype(np.bool_)
        right_exact = right_exact_raw.astype(np.bool_)
    else:
        left_exact = left_exact_raw.astype(np.float64)
        right_exact = right_exact_raw.astype(np.float64)
    token_boot = _bootstrap_samples(
        left_token_accuracy,
        right_token_accuracy,
        draws=draws,
        seed=seed,
    )
    delta_token = float(left_token_accuracy.mean() - right_token_accuracy.mean())
    token_result: dict[str, Any] = {
        "point": delta_token,
        "point_pp": 100.0 * delta_token,
        "ci95": [float(np.quantile(token_boot, 0.025)), float(np.quantile(token_boot, 0.975))],
        "records": int(left_token_accuracy.size),
        "draws": int(draws),
        "seed": int(seed),
        "resampling_unit": "source record",
    }
    if primary:
        token_result["primary_lower"] = float(np.quantile(token_boot, TOKEN_TAIL_ALPHA))
        token_result["primary_upper"] = float(np.quantile(token_boot, 1.0 - TOKEN_TAIL_ALPHA))
        token_result["tail_alpha"] = TOKEN_TAIL_ALPHA
    exact_gain = np.logical_and(left_exact, np.logical_not(right_exact)) if exact_cp else None
    exact_loss = np.logical_and(np.logical_not(left_exact), right_exact) if exact_cp else None
    exact_boot = _bootstrap_samples(
        left_exact.astype(np.float64),
        right_exact.astype(np.float64),
        draws=draws,
        seed=seed,
    )
    exact_result: dict[str, Any] = {
        "point": float(left_exact.mean() - right_exact.mean()),
        "point_pp": float(100.0 * (left_exact.mean() - right_exact.mean())),
        "ci95": [float(np.quantile(exact_boot, 0.025)), float(np.quantile(exact_boot, 0.975))],
        "records": int(left_exact.size),
        "draws": int(draws),
        "seed": int(seed),
        "resampling_unit": "source record",
    }
    if exact_cp:
        assert exact_gain is not None and exact_loss is not None
        # Primary direct edges receive the preregistered family correction.
        # Secondary direct comparisons retain a labelled descriptive 95% CP
        # interval, so no corrected bound is mistaken for a primary result.
        component_alpha = EXACT_COMPONENT_ALPHA if primary else 0.025
        exact_result.update(
            clopper_pearson_gain_loss(
                int(exact_gain.sum()),
                int(exact_loss.sum()),
                int(left_exact.size),
                component_alpha=component_alpha,
            )
        )
        exact_result["interval_scope"] = (
            "primary corrected one-sided gain/loss components"
            if primary else "descriptive 95% gain/loss components; no multiplicity correction"
        )
    elif primary:
        exact_result["primary_lower"] = float(np.quantile(exact_boot, TOKEN_TAIL_ALPHA))
        exact_result["primary_upper"] = float(np.quantile(exact_boot, 1.0 - TOKEN_TAIL_ALPHA))
        exact_result["uncertainty"] = "bootstrap interaction contrast; CP gain/loss applies only to direct binary comparisons"
    return {"token": token_result, "exact": exact_result}


def _contrast_for_metrics(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    seed: int,
    draws: int,
    primary: bool,
    exact_cp: bool = True,
) -> dict[str, Any]:
    return paired_contrast(
        left["per_record_token_accuracy"], right["per_record_token_accuracy"],
        left["per_record_exact"], right["per_record_exact"],
        seed=seed, draws=draws, primary=primary, exact_cp=exact_cp,
    )


def _build_factorial_contrasts(
    cell_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cell_id: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Build the four primary edges plus labelled descriptive diagnostics."""

    def metrics(method: str) -> Mapping[str, Any]:
        return cell_results[method][cell_id]

    result: dict[str, Any] = {}
    for label, left_method, right_method in PRIMARY_FACTORIAL_EDGES:
        contrast = _contrast_for_metrics(
            metrics(left_method), metrics(right_method),
            seed=seed, draws=draws, primary=True, exact_cp=True,
        )
        result[label] = {
            "contrast": contrast,
            "decision": _contrast_decision(contrast),
            "scope": "primary direct factorial edge",
        }
    current_diag = metrics("current_enriched__trained_diagonal")
    current_res = metrics("current_enriched__residual_mlp512")
    improved_diag = metrics("improved_public_bank__trained_diagonal")
    improved_res = metrics("improved_public_bank__residual_mlp512")
    ref = metrics(contract.REFERENCE_METHOD_ID)
    interaction = paired_contrast(
        improved_res["per_record_token_accuracy"] - improved_diag["per_record_token_accuracy"],
        current_res["per_record_token_accuracy"] - current_diag["per_record_token_accuracy"],
        improved_res["per_record_exact"].astype(np.float64) - improved_diag["per_record_exact"].astype(np.float64),
        current_res["per_record_exact"].astype(np.float64) - current_diag["per_record_exact"].astype(np.float64),
        seed=seed, draws=draws, primary=False, exact_cp=False,
    )
    result["interaction_detail"] = {
        "contrast": interaction,
        "scope": "descriptive 95% interaction detail; outside primary family",
    }
    result["improved_residual_vs_reference_endpoint"] = {
        "contrast": _contrast_for_metrics(
            improved_res, ref, seed=seed, draws=draws, primary=False, exact_cp=True,
        ),
        "scope": "descriptive 95% endpoint comparison; outside primary family",
    }
    result["interpretation"] = "The four named direct edges are the preregistered primary family; interaction and reference endpoint details are descriptive."
    return result


def _build_anchor_comparisons(
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    decoder_first32: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Build descriptive first-32 decoder/reference versus A2 anchor gaps."""

    decoder_vs_a2: dict[str, Any] = {}
    for method in DECODER_METHODS:
        decoder_vs_a2[method] = {
            "contrast": _contrast_for_metrics(
                decoder_first32[method], a2,
                seed=seed, draws=draws, primary=False, exact_cp=True,
            ),
            "records_per_domain": contract.ANCHOR_RECORDS_PER_DOMAIN,
            "post_bos_token_denominator": contract.ANCHOR_RECORDS_PER_DOMAIN * contract.SCORED_POST_BOS_TOKENS,
            "exact_record_denominator": contract.ANCHOR_RECORDS_PER_DOMAIN,
            "scope": "descriptive 95% paired decoder-vs-A2 gap; outside primary family",
        }
    return {
        "paired_student_vs_a2": decoder_vs_a2,
        "a2_minus_a1": {
            "contrast": _contrast_for_metrics(
                a2, a1, seed=seed, draws=draws, primary=False, exact_cp=True,
            ),
            "records_per_domain": contract.ANCHOR_RECORDS_PER_DOMAIN,
            "post_bos_token_denominator": contract.ANCHOR_RECORDS_PER_DOMAIN * contract.SCORED_POST_BOS_TOKENS,
            "exact_record_denominator": contract.ANCHOR_RECORDS_PER_DOMAIN,
            "scope": "descriptive 95% A1-to-A2 anchor gap; outside primary family",
        },
    }


def _load_frequency_reference(
    registration: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Load the one immutable public fitting-frequency map used by the diagnostic."""

    binding = registration.get("frequency_reference")
    if not isinstance(binding, Mapping):
        raise ScoreError("registration lacks the public frequency reference binding")
    try:
        record = contract.validate_file_record(
            binding,
            repository_root=repository_root,
            description="public frequency reference",
            verify=True,
        )
        value = contract.load_json(Path(record["path"]), description="public frequency reference")
    except contract.ContractError as exc:
        raise ScoreError(str(exc)) from exc
    maps = value.get("frequency_references")
    enriched = maps.get("enriched") if isinstance(maps, Mapping) else None
    if (
        value.get("schema") != contract.FREQUENCY_REFERENCE_SCHEMA
        or value.get("task_id") != "TRR-0005"
        or value.get("status") != "PUBLIC_FITTING_FREQUENCY_REFERENCES"
        or not isinstance(enriched, Mapping)
    ):
        raise ScoreError("public frequency reference enriched map is invalid")
    try:
        counts = {int(token_id): int(count) for token_id, count in enriched.items()}
    except (TypeError, ValueError) as exc:
        raise ScoreError("public enriched frequency map is malformed") from exc
    if any(token_id < 0 or token_id >= contract.VOCAB_SIZE for token_id in counts):
        raise ScoreError("public enriched frequency map contains an invalid token ID")
    if any(count <= 0 for count in counts.values()):
        raise ScoreError("public enriched frequency map contains a non-positive count")
    return {
        "binding": record,
        "map_name": "enriched",
        "counts": counts,
        "schema": value["schema"],
        "counting_scope": "TRR-0005 public enriched fitting positions; sparse missing IDs are frequency zero",
    }


def _frequency_bin(frequency: int) -> str:
    value = int(frequency)
    for name, lower, upper in support.FREQUENCY_BINS:
        if value >= lower and (upper is None or value <= upper):
            return name
    raise ScoreError(f"frequency has no declared bin: {value}")


def _prefix_bin(position: int) -> str:
    value = int(position)
    for name, lower, upper in support.POSITION_BINS:
        if lower <= value <= upper:
            if upper > contract.SCORED_POST_BOS_TOKENS:
                break
            return name
    raise ScoreError(f"scored post-BOS position has no prefix bin: {value}")


def _frequency_error_diagnostic(
    tensors: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    truth_by_domain: Mapping[str, torch.Tensor],
    *,
    frequency_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate token errors by cell, decoder, prefix bin, and baseline frequency.

    This is descriptive only.  It consumes the labels already opened by the
    scorer's single truth-sidecar read and performs no selection or fitting.
    """

    counts = frequency_reference.get("counts")
    if not isinstance(counts, Mapping):
        raise ScoreError("frequency diagnostic map is unavailable")
    bins: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    decoder_specs: list[tuple[str, str]] = []
    for method in DECODER_METHODS:
        decoder_specs.append((method, "a2" if method == contract.ANCHOR_METHOD_ID else "decoder"))
        if method == contract.ANCHOR_METHOD_ID:
            decoder_specs.append((f"{method}__a1", "a1_diagnostic"))
    for method, method_role in decoder_specs:
        base_method = contract.ANCHOR_METHOD_ID if method_role == "a1_diagnostic" else method
        for cell_id in contract.expected_method_cells(base_method):
            domain, target = cell_id.split("__", 1)
            truth = torch.as_tensor(truth_by_domain[domain][:contract.ANCHOR_RECORDS_PER_DOMAIN if method_role == "a1_diagnostic" or base_method == contract.ANCHOR_METHOD_ID else contract.RECORDS_PER_DOMAIN], dtype=torch.long).cpu()
            values = tensors[base_method][cell_id]
            prediction = values["a1_predictions"] if method_role == "a1_diagnostic" else values["predictions"]
            prediction = torch.as_tensor(prediction[: truth.shape[0]], dtype=torch.long).cpu()
            if tuple(prediction.shape) != tuple(truth.shape):
                raise ScoreError(f"frequency diagnostic geometry differs: {method}/{cell_id}")
            correct = prediction[:, 1:].eq(truth[:, 1:]).numpy()
            labels = truth[:, 1:].numpy()
            for position_index in range(labels.shape[1]):
                position = position_index + 1
                prefix = _prefix_bin(position)
                for record_index in range(labels.shape[0]):
                    token_id = int(labels[record_index, position_index])
                    frequency = int(counts.get(token_id, 0))
                    frequency_name = _frequency_bin(frequency)
                    key = (cell_id, domain, target, method, prefix, frequency_name)
                    cell = bins.setdefault(
                        key,
                        {
                            "cell_id": cell_id,
                            "domain": domain,
                            "target": target,
                            "method_id": method,
                            "method_role": method_role,
                            "prefix_bin": prefix,
                            "frequency_bin": frequency_name,
                            "token_positions": 0,
                            "correct_tokens": 0,
                            "token_errors": 0,
                            "distinct_truth_token_ids": set(),
                        },
                    )
                    cell["token_positions"] += 1
                    cell["correct_tokens"] += int(correct[record_index, position_index])
                    cell["token_errors"] += int(not correct[record_index, position_index])
                    cell["distinct_truth_token_ids"].add(token_id)
    rows: list[dict[str, Any]] = []
    for key in sorted(bins):
        row = dict(bins[key])
        positions = int(row.pop("token_positions"))
        correct_tokens = int(row.pop("correct_tokens"))
        errors = int(row.pop("token_errors"))
        distinct = row.pop("distinct_truth_token_ids")
        row.update(
            {
                "token_positions": positions,
                "correct_tokens": correct_tokens,
                "token_errors": errors,
                "token_accuracy": (float(correct_tokens / positions) if positions else None),
                "distinct_truth_token_ids": len(distinct),
                "denominator_definition": "all valid labels in the scored 127 post-BOS positions; no truth payload is persisted",
            }
        )
        rows.append(row)
    return {
        "status": "DESCRIPTIVE_SAME_PASS_AGGREGATION",
        "scope": "domain x target x one-based post-BOS prefix bin x common baseline fitting-frequency bin",
        "primary_or_secondary": "secondary descriptive diagnostic; outside all formal primary bounds",
        "frequency_reference": {
            "binding": dict(frequency_reference["binding"]),
            "map_name": str(frequency_reference["map_name"]),
            "schema": str(frequency_reference["schema"]),
            "counting_scope": str(frequency_reference["counting_scope"]),
        },
        "frequency_bins": [
            {"name": name, "lower": lower, "upper": upper}
            for name, lower, upper in support.FREQUENCY_BINS
        ],
        "prefix_bins": [
            {"name": name, "lower": lower, "upper": upper}
            for name, lower, upper in support.POSITION_BINS
            if upper <= contract.SCORED_POST_BOS_TOKENS
        ],
        "rows": rows,
        "row_count": len(rows),
    }


def _score_predictions(predictions: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    pred = torch.as_tensor(predictions, dtype=torch.long).detach().cpu().contiguous()
    target = torch.as_tensor(truth, dtype=torch.long).detach().cpu().contiguous()
    if pred.ndim != 2 or target.shape != pred.shape or pred.shape[1] != contract.STORED_SEQUENCE_TOKENS:
        raise ScoreError("prediction/truth geometry differs")
    post = pred[:, 1:].eq(target[:, 1:])
    per_record_token_accuracy = post.to(torch.float64).mean(dim=1)
    # BOS is fixed by the public interface and is reported as a diagnostic;
    # exact recovery is defined over all 127 post-BOS positions.
    bos_fixed = pred[:, 0].eq(target[:, 0])
    exact = post.all(dim=1)
    first_error: list[int | None] = []
    for row in range(pred.shape[0]):
        wrong = (~post[row]).nonzero(as_tuple=False).flatten()
        first_error.append(None if wrong.numel() == 0 else int(wrong[0].item()) + 1)
    correct = int(post.sum().item())
    total = int(post.numel())
    exact_count = int(exact.sum().item())
    return {
        "records": int(pred.shape[0]),
        "post_bos_positions_per_record": contract.SCORED_POST_BOS_TOKENS,
        "token_positions": total,
        "correct_tokens": correct,
        "token_errors": total - correct,
        "token_accuracy": float(correct / total),
        "bos_fixed_records": int(bos_fixed.sum().item()),
        "exact_definition": "all 127 post-BOS positions; BOS is a fixed known diagnostic",
        "exact_records": exact_count,
        "exact_record_rate": float(exact_count / pred.shape[0]),
        "per_record_token_accuracy": per_record_token_accuracy.numpy(),
        "per_record_exact": exact.numpy().astype(np.bool_),
        "first_error_position": first_error,
    }


def _public_truth_sidecar(
    header: Mapping[str, Any],
    *,
    repository_root: Path,
    output_root: Path,
) -> dict[str, torch.Tensor]:
    """Open the sidecar once, after gate validation, and return two tensors."""

    sidecar = header.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise ScoreError("truth sidecar descriptor is absent")
    path = Path(str(sidecar["path"])).expanduser().resolve()
    gate._outside(path, output_root, description="private truth sidecar")
    gate._outside(path, repository_root, description="private truth sidecar")
    if path.is_symlink() or not path.is_file():
        raise ScoreError("private truth sidecar is unavailable after the public gate")
    actual_bytes = int(path.stat().st_size)
    actual_sha = contract.sha256_file(path)
    if actual_bytes != sidecar.get("bytes") or actual_sha != sidecar.get("sha256"):
        raise ScoreError("private truth sidecar changed after the public gate")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            expected_keys = {"pile__token_ids", "finance__token_ids"}
            if set(handle.keys()) != expected_keys:
                raise ScoreError("private truth sidecar tensor keys changed")
            if metadata.get("schema") != "token-reconstruction.trr0007-truth-sidecar.v1":
                raise ScoreError("private truth sidecar schema changed")
            if metadata.get("task_id") != contract.TASK_ID:
                raise ScoreError("private truth sidecar task ID changed")
            if metadata.get("truth_opened") not in (None, "false", "False"):
                raise ScoreError("private truth sidecar truth flag is open")
            result = {
                cell_id: handle.get_tensor(f"{cell_id}__token_ids").detach().cpu().to(torch.long).contiguous()
                for cell_id in contract.DOMAIN_ORDER
            }
    except ScoreError:
        raise
    except Exception as exc:
        raise ScoreError("private truth sidecar could not be read") from exc
    for domain, value in result.items():
        if tuple(value.shape) != (contract.RECORDS_PER_DOMAIN, contract.STORED_SEQUENCE_TOKENS):
            raise ScoreError(f"private truth geometry changed: {domain}")
        if int(value[:, 0].ne(contract.BOS_TOKEN_ID).sum().item()) != 0:
            raise ScoreError(f"private truth BOS changed: {domain}")
        if value.lt(0).any().item() or value.ge(contract.VOCAB_SIZE).any().item():
            raise ScoreError(f"private truth ID range changed: {domain}")
    return result


def _load_prediction_tensors(
    output_root: Path,
    *,
    registration: Mapping[str, Any],
    observations: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    result: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for method in contract.METHOD_ORDER:
        records = contract.ANCHOR_RECORDS_PER_DOMAIN if method == contract.ANCHOR_METHOD_ID else contract.RECORDS_PER_DOMAIN
        result[method] = {}
        for cell_id in contract.expected_method_cells(method):
            cell = observations["cells"][cell_id]
            path = contract.expected_prediction_path(output_root, cell_id=cell_id, method_id=method)
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    values = {"predictions": handle.get_tensor("predictions").detach().cpu().to(torch.long).contiguous()}
                    if method == contract.ANCHOR_METHOD_ID:
                        values["a1_predictions"] = handle.get_tensor("a1_predictions").detach().cpu().to(torch.long).contiguous()
                contract.validate_prediction_artifact(
                    path,
                    registration=registration,
                    cell=cell,
                    method_id=method,
                    records=records,
                )
            except contract.ContractError as exc:
                raise ScoreError(str(exc)) from exc
            except Exception as exc:
                raise ScoreError(f"prediction artifact could not be read: {path}") from exc
            result[method][cell_id] = values
    return result


def _cell_metric(
    predictions: Mapping[str, torch.Tensor],
    truth: torch.Tensor,
    *,
    records: int,
) -> dict[str, Any]:
    return _score_predictions(predictions["predictions"][:records], truth[:records])


def _fit_cost(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("fit_cost")
    return value if isinstance(value, Mapping) else None


def _cost_summary(
    registration: Mapping[str, Any],
    timings: Mapping[str, Any],
) -> dict[str, Any]:
    by_method: dict[str, Any] = {}
    for row in registration["methods"]:
        method = str(row["id"])
        entries = [
            timings[f"{method}::{cell}"]
            for cell in contract.expected_method_cells(method)
            if f"{method}::{cell}" in timings
        ]
        measured = [float(item["measured_seconds_sum"]) for item in entries]
        prep = [float(item["model_preparation_seconds"]) for item in entries if "model_preparation_seconds" in item]
        if method == contract.REFERENCE_METHOD_ID:
            by_method[method] = {
                "measured_seconds_by_cell": measured,
                "preparation_seconds": prep,
                "preparation_ratio": None,
                "ratio_denominator": "reused retained reference; no ratio is defined",
            }
            continue
        ref_entries = []
        for cell in contract.expected_method_cells(method):
            ref = timings.get(f"{contract.REFERENCE_METHOD_ID}::{cell}")
            student = timings.get(f"{method}::{cell}")
            if isinstance(ref, Mapping) and isinstance(student, Mapping):
                student_time = float(student["measured_seconds_sum"])
                student_records = int(student.get("records", contract.RECORDS_PER_DOMAIN))
                if method == contract.ANCHOR_METHOD_ID:
                    per_record = ref.get("per_record_measured_seconds")
                    if not isinstance(per_record, list) or len(per_record) < contract.ANCHOR_RECORDS_PER_DOMAIN:
                        raise ScoreError("reference timing lacks the first-32 per-record denominator for A1+A2")
                    reference_records = contract.ANCHOR_RECORDS_PER_DOMAIN
                    ref_time = sum(float(value) for value in per_record[:reference_records])
                    denominator_rule = "sum of reference per_record_measured_seconds for the matching first 32 public-base records"
                else:
                    reference_records = contract.RECORDS_PER_DOMAIN
                    ref_time = float(ref["measured_seconds_sum"])
                    denominator_rule = "same-cell reference measured_seconds_sum for all 128 records"
                ref_entries.append({
                    "cell_id": cell,
                    "student_records": student_records,
                    "reference_records": reference_records,
                    "student_measured_seconds": student_time,
                    "reference_measured_seconds": ref_time,
                    "ratio": None if ref_time == 0.0 else student_time / ref_time,
                    "zero_reference_denominator": ref_time == 0.0,
                    "denominator_rule": denominator_rule,
                })
        fit_cost = next((_fit_cost(row) for _ in registration["methods"] if row["id"] == method), None)
        by_method[method] = {
            "measured_seconds_by_cell": measured,
            "preparation_seconds": prep,
            "runtime_ratio_vs_reference": ref_entries,
            "fit_cost": fit_cost,
            "training_draws_per_arm": contract.FIT_TRAINING_DRAWS,
            "training_post_bos_positions": contract.FIT_POST_BOS_POSITIONS,
            "preparation_ratio_rule": "method-specific student preparation divided by same-cell reference only when reference denominator is nonzero; reused reference preparation has no ratio",
        }
    return {
        "methods": by_method,
        "runtime_budget_ratio": 1.25,
        "training_or_preparation_budget_ratio": 2.0,
        "qualification_rule": "A quality point estimate is cost-qualified only with a known ratio within budget; unavailable fit/prep cost remains unqualified and is reported.",
    }


def _contrast_decision(contrast: Mapping[str, Any]) -> dict[str, Any]:
    token = contrast["token"]
    exact = contrast["exact"]
    token_point_useful = float(token["point_pp"]) >= USEFUL_TOKEN_PP
    exact_point_useful = float(exact["point_pp"]) >= USEFUL_EXACT_PP
    token_margin = float(token.get("primary_lower", float("-inf"))) * 100.0 >= USEFUL_TOKEN_PP
    exact_margin = float(exact.get("lower", exact.get("primary_lower", float("-inf")))) * 100.0 >= USEFUL_EXACT_PP
    token_harm_evidenced = float(token.get("primary_upper", float("inf"))) * 100.0 < -USEFUL_TOKEN_PP
    exact_harm_evidenced = float(exact.get("upper", float("inf"))) * 100.0 < -USEFUL_EXACT_PP
    token_harm_excluded = float(token.get("primary_lower", float("-inf"))) * 100.0 >= -USEFUL_TOKEN_PP
    exact_harm_excluded = float(exact.get("lower", exact.get("primary_lower", float("-inf")))) * 100.0 >= -USEFUL_EXACT_PP
    return {
        "exploratory_useful_point": bool(token_point_useful or exact_point_useful),
        "token_point_useful": bool(token_point_useful),
        "exact_point_useful": bool(exact_point_useful),
        "token_margin_exceeded": bool(token_margin),
        "exact_margin_exceeded": bool(exact_margin),
        "token_material_harm_evidenced": bool(token_harm_evidenced),
        "exact_material_harm_evidenced": bool(exact_harm_evidenced),
        "token_harm_excluded": bool(token_harm_excluded),
        "exact_harm_excluded": bool(exact_harm_excluded),
        "interpretation": "point estimates are exploratory-useful; margin confidence requires the corrected one-sided bound",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def score_after_gate(
    *,
    receipt_path: Path,
    registration_path: Path,
    truth_binding_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    pretruth = gate.validate_before_truth(
        receipt_path=receipt_path,
        registration_path=registration_path,
        repository_root=root,
        truth_binding_path=truth_binding_path,
    )
    header = pretruth.get("truth_binding_header")
    if not isinstance(header, Mapping):
        raise ScoreError("public gate did not return a truth binding header")
    registration = pretruth["registration"]
    output_root = Path(str(registration["output_root"])).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    frequency_reference = _load_frequency_reference(registration, repository_root=root)
    _manifest, observations, _observation_record = contract.load_observation_manifest(
        registration, repository_root=root, verify_assets=True
    )
    # Truth is first opened here, after the public gate and never earlier.
    truth_by_domain = _public_truth_sidecar(header, repository_root=root, output_root=output_root)
    tensors = _load_prediction_tensors(output_root, registration=registration, observations=observations)
    cell_results: dict[str, dict[str, Any]] = {}
    for method in contract.METHOD_ORDER:
        cell_results[method] = {}
        records = contract.ANCHOR_RECORDS_PER_DOMAIN if method == contract.ANCHOR_METHOD_ID else contract.RECORDS_PER_DOMAIN
        for cell_id in contract.expected_method_cells(method):
            domain = cell_id.split("__", 1)[0]
            target = truth_by_domain[domain][:records]
            cell_results[method][cell_id] = _cell_metric(tensors[method][cell_id], target, records=records)
    paired_reference: dict[str, Any] = {}
    for method in contract.STUDENT_METHOD_IDS:
        paired_reference[method] = {}
        for cell_id in contract.CELL_ORDER:
            student = cell_results[method][cell_id]
            reference = cell_results[contract.REFERENCE_METHOD_ID][cell_id]
            contrast = paired_contrast(
                student["per_record_token_accuracy"],
                reference["per_record_token_accuracy"],
                student["per_record_exact"],
                reference["per_record_exact"],
                seed=BOOTSTRAP_SEED,
                draws=BOOTSTRAP_DRAWS,
                primary=False,
            )
            paired_reference[method][cell_id] = {
                "contrast": contrast,
                "decision_scope": "descriptive 95% student-vs-reference comparison; outside primary family",
            }

    factorial: dict[str, Any] = {
        cell_id: _build_factorial_contrasts(cell_results, cell_id)
        for cell_id in contract.CELL_ORDER
    }
    anchor: dict[str, Any] = {}
    for cell_id in contract.BASE_CELL_ORDER:
        domain = cell_id.split("__", 1)[0]
        truth_anchor = truth_by_domain[domain][:contract.ANCHOR_RECORDS_PER_DOMAIN]
        a2 = cell_results[contract.ANCHOR_METHOD_ID][cell_id]
        a1 = _score_predictions(
            tensors[contract.ANCHOR_METHOD_ID][cell_id]["a1_predictions"],
            truth_anchor,
        )
        decoder_first32: dict[str, Any] = {}
        for method in DECODER_METHODS:
            decoder_first32[method] = _score_predictions(
                tensors[method][cell_id]["predictions"][:contract.ANCHOR_RECORDS_PER_DOMAIN],
                truth_anchor,
            )
        anchor_comparisons = _build_anchor_comparisons(a1, a2, decoder_first32)
        anchor[cell_id] = {
            "records_per_domain": contract.ANCHOR_RECORDS_PER_DOMAIN,
            "post_bos_token_denominator": contract.ANCHOR_RECORDS_PER_DOMAIN * contract.SCORED_POST_BOS_TOKENS,
            "exact_record_denominator": contract.ANCHOR_RECORDS_PER_DOMAIN,
            "same_first32_public_base_records": True,
            "decoder_first32": decoder_first32,
            **anchor_comparisons,
            "a1": a1,
            "a1_a2": a2,
        }
    frequency_error_diagnostic = _frequency_error_diagnostic(
        tensors,
        truth_by_domain,
        frequency_reference=frequency_reference,
    )
    # Remove arrays before serializing; they are computation-only, not truth output.
    clean_cells = _jsonable(cell_results)
    for method in clean_cells.values():
        for metrics in method.values():
            metrics.pop("per_record_token_accuracy", None)
            metrics.pop("per_record_exact", None)
    clean_anchor = _jsonable(anchor)
    for item in clean_anchor.values():
        for key in ("a1", "a1_a2"):
            item[key].pop("per_record_token_accuracy", None)
            item[key].pop("per_record_exact", None)
        for metrics in item["decoder_first32"].values():
            metrics.pop("per_record_token_accuracy", None)
            metrics.pop("per_record_exact", None)
    timings = _load_json(output_root / "timings.json", description="timing descriptor")
    result = {
        "schema": contract.SCORE_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "SCORED_AFTER_COMPLETE_PUBLIC_GATE",
        "scored_utc": _utc_now(),
        "gate": {
            "receipt": pretruth["receipt"],
            "truth_binding_header": {
                "schema": header["schema"],
                "task_id": header["task_id"],
                "sidecar": dict(header["sidecar"]),
            },
            "truth_opened_once": True,
            "verified_before_truth": True,
        },
        "panel": {
            "records_per_domain": contract.RECORDS_PER_DOMAIN,
            "post_bos_tokens_per_record": contract.SCORED_POST_BOS_TOKENS,
            "cells": list(contract.CELL_ORDER),
            "truth_records_used": contract.RECORDS_PER_DOMAIN,
        },
        "cell_metrics": clean_cells,
        "paired_student_vs_reference": _jsonable(paired_reference),
        "factorial_contrasts": _jsonable(factorial),
        "bounded_a1_a2_anchor": clean_anchor,
        "frequency_error_diagnostic": _jsonable(frequency_error_diagnostic),
        "cost": _cost_summary(registration, timings["entries"]),
        "uncertainty": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "token_tail_alpha": TOKEN_TAIL_ALPHA,
            "exact_component_alpha": EXACT_COMPONENT_ALPHA,
            "primary_family": "4 direct factorial edges x 4 domain/target cells x 2 outcomes x 2 directions = 64 directional bounds",
            "primary_edges": [
                "support_at_trained_diagonal",
                "support_at_residual_mlp512",
                "capacity_on_current_enriched",
                "capacity_on_improved_public_bank",
            ],
            "secondary": "student-vs-reference, interaction detail, reference endpoint, decoder-vs-A2, and A1+A2 anchor intervals are descriptive 95% intervals outside the primary family",
        },
        "truth_opened_once": True,
        "private_truth_payload_persisted": False,
    }
    return _jsonable(result)


def write_score_outputs(
    result: Mapping[str, Any],
    *,
    repository_root: Path,
    result_path: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    result_path = Path(result_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()
    payload = dict(result)
    contract.write_create_only(result_path, payload)
    result_record = {
        "path": str(result_path),
        "bytes": int(result_path.stat().st_size),
        "sha256": contract.sha256_file(result_path),
    }
    # Keep the structured handoff small and hash-bind the complete result.
    manifest = {
        "schema": "token-reconstruction.trr0007-manifest.v1",
        "task_id": contract.TASK_ID,
        "status": "COMPLETE_EXPLORATORY_HANDOFF",
        "result": result_record,
        "receipt": result.get("gate", {}).get("receipt"),
        "truth_opened_once": True,
        "private_truth_payload_persisted": False,
        "replay": {
            "runner": "scripts/trr0007_eval_runner.py",
            "gate": "scripts/trr0007_eval_gate.py",
            "scorer": "scripts/trr0007_score.py",
            "score_result": str(result_path),
        },
    }
    contract.write_create_only(manifest_path, manifest)
    report_lines = [
        "# TRR-0007 exploratory evaluation",
        "",
        "The natural panel contains 128 records per domain and four paired public target cells. Metrics use all 127 post-BOS positions per record; BOS is a fixed known diagnostic. The bounded P0 A1+A2 anchor and every retained decoder/reference use the same first 32 public-base records per domain (4,064 post-BOS token positions and 32 exact records per domain), with paired decoder-vs-A2 gaps.",
        "",
        "Predictions passed the complete public matrix gate before the truth sidecar was opened once. Point estimates are exploratory-useful at the preregistered 1 token-point or 5 exact-point thresholds; one-sided margin evidence uses the corrected multiplicity tails and harm directions recorded in the structured result.",
        "",
        f"Structured result: {result_path}",
        f"Manifest: {manifest_path}",
        "",
        "This phase is exploratory and does not establish dual-benchmark completeness or a canonical replacement method.",
    ]
    if report_path.exists() or report_path.is_symlink():
        raise ScoreError(f"refusing to overwrite report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(report_lines) + "\n")
    return {
        "result": result_record,
        "manifest": {
            "path": str(manifest_path),
            "bytes": int(manifest_path.stat().st_size),
            "sha256": contract.sha256_file(manifest_path),
        },
        "report": {
            "path": str(report_path),
            "bytes": int(report_path.stat().st_size),
            "sha256": contract.sha256_file(report_path),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--truth-binding", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--result", type=Path, default=Path("experiments/TRR-0007/scored/result.json"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/TRR-0007/manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("coordination/results/TRR-0007.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = score_after_gate(
            receipt_path=args.receipt,
            registration_path=args.registration,
            truth_binding_path=args.truth_binding,
            repository_root=args.repository_root,
        )
        outputs = write_score_outputs(
            result,
            repository_root=args.repository_root,
            result_path=args.result,
            manifest_path=args.manifest,
            report_path=args.report,
        )
    except (ScoreError, gate.GateError, contract.ContractError) as exc:
        print(f"TRR-0007 scoring failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"result": result, "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
