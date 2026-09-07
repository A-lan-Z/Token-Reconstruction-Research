"""Truth-gated TRR-0010 scorer adapter.

The public freeze is revalidated before the supplied truth loader is called.
Predictions and the public fitting-frequency reference are materialized before
that call; after it returns, scoring uses only the in-memory paired tensors and
the source-paired analysis primitives. This module does not discover source
records, load a model, reconstruct candidates, or write prediction arrays.
"""
from __future__ import annotations

from array import array
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
from safetensors import safe_open

from scripts import trr0010_analysis as analysis
from scripts import trr0010_eval_gate as gate


TASK_ID = gate.TASK_ID
SCORE_SCHEMA = "token-reconstruction.trr0010-score.v1"
SCORE_STATUS = "SCORE_COMPLETE_AFTER_PUBLIC_FREEZE"

FREQUENCY_BINS = ("unseen_0", "seen_1_4", "seen_5_9", "seen_10_49", "seen_50_plus")
DOMAIN_ORDER = ("finance", "pile")
DOMAIN_BY_CELL = {cell: cell.split("__", 1)[0] for cell in gate.CELL_ORDER}

# These are the predeclared factor and anchor contrasts. No post-score control
# selection is performed by this adapter. Frequency strata are contrasted only
# when both arms use the same frozen fitting-frequency bank.
CONTRASTS = {
    "directional_current": (gate.CURRENT_DIRECTIONAL_METHOD_ID, gate.CURRENT_FIXED_METHOD_ID),
    "directional_expanded": (gate.EXPANDED_DIRECTIONAL_METHOD_ID, gate.EXPANDED_FIXED_METHOD_ID),
    "data_fixed": (gate.EXPANDED_FIXED_METHOD_ID, gate.CURRENT_FIXED_METHOD_ID),
    "data_directional": (gate.EXPANDED_DIRECTIONAL_METHOD_ID, gate.CURRENT_DIRECTIONAL_METHOD_ID),
    "data_expanded_fixed_vs_unchanged": (gate.EXPANDED_FIXED_METHOD_ID, gate.UNCHANGED_METHOD_ID),
    "combined_candidate": (gate.EXPANDED_DIRECTIONAL_METHOD_ID, gate.UNCHANGED_METHOD_ID),
    # Kept as an explicit compatibility alias for earlier TRR-0010 consumers.
    "combined_vs_unchanged": (gate.EXPANDED_DIRECTIONAL_METHOD_ID, gate.UNCHANGED_METHOD_ID),
    "a1_a2_vs_unchanged": (gate.A1_A2_METHOD_ID, gate.UNCHANGED_METHOD_ID),
}

# Remaining-error reduction is a ratio route, so every denominator is kept
# separate and fail-closed. These are the v4 useful routes, plus the explicit
# A1+A2 anchor report and compatibility alias.
ERROR_REDUCTION_ROUTES = {
    "directional_expanded": (gate.EXPANDED_DIRECTIONAL_METHOD_ID, gate.EXPANDED_FIXED_METHOD_ID),
    "directional_current": (gate.EXPANDED_DIRECTIONAL_METHOD_ID, gate.CURRENT_FIXED_METHOD_ID),
    "data_fixed": (gate.EXPANDED_FIXED_METHOD_ID, gate.CURRENT_FIXED_METHOD_ID),
    "data_directional": (gate.EXPANDED_DIRECTIONAL_METHOD_ID, gate.CURRENT_DIRECTIONAL_METHOD_ID),
    "data_expanded_fixed_vs_unchanged": (gate.EXPANDED_FIXED_METHOD_ID, gate.UNCHANGED_METHOD_ID),
    "combined_candidate": (gate.EXPANDED_DIRECTIONAL_METHOD_ID, gate.UNCHANGED_METHOD_ID),
    "combined_vs_unchanged": (gate.EXPANDED_DIRECTIONAL_METHOD_ID, gate.UNCHANGED_METHOD_ID),
    "a1_a2_vs_unchanged": (gate.A1_A2_METHOD_ID, gate.UNCHANGED_METHOD_ID),
}

# Values are copied from the frozen v4 contract. They are serialized with the
# checks so the planner can distinguish an UNKNOWN route from a failed one.
PRACTICAL_THRESHOLDS = {
    "token_accuracy_floor": 0.005,
    "exact_point_floor": 0.05,
    "remaining_error_reduction_fraction": 0.20,
    "gap_visibility_token": 0.01,
    "gap_visibility_exact": 0.05,
    "gap_minimum_fraction": 0.25,
    "harm_token_lower": -0.005,
    "harm_exact_lower": -0.03,
}


class ScoreError(ValueError):
    """Raised when public score inputs or the truth result violate the contract."""


class _BootstrapPlan:
    """One deterministic source-record resampling stream per domain.

    The same tensor of indices is reused by every method, target condition,
    contrast, frequency stratum, error-reduction ratio, and gap-closure ratio
    within a domain. Keeping the plan in a compact tensor avoids regenerating
    different paired samples for otherwise identical source records.
    """

    def __init__(self, *, seed: int, draws: int, records_by_domain: Mapping[str, int]) -> None:
        if int(draws) <= 0:
            raise ScoreError("bootstrap draws must be positive")
        self.seed = int(seed)
        self.draws = int(draws)
        self.records_by_domain = {str(domain): int(records) for domain, records in records_by_domain.items()}
        self._indices: dict[str, torch.Tensor] = {}
        self._seeds: dict[str, int] = {}
        self._digests: dict[str, str] = {}
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            if domain not in self.records_by_domain or self.records_by_domain[domain] <= 0:
                raise ScoreError(f"bootstrap record count is missing for {domain}")
            records = self.records_by_domain[domain]
            # A documented, stable domain offset gives each domain one stream
            # while preserving the single contract seed as the root seed.
            domain_seed = self.seed + domain_index
            rng = random.Random(domain_seed)
            packed = array("I", (rng.randrange(records) for _ in range(self.draws * records)))
            self._indices[domain] = torch.tensor(list(packed), dtype=torch.long).reshape(self.draws, records)
            self._seeds[domain] = domain_seed
            self._digests[domain] = hashlib.sha256(packed.tobytes()).hexdigest()

    def indices(self, domain: str) -> torch.Tensor:
        try:
            return self._indices[domain]
        except KeyError as exc:
            raise ScoreError(f"unknown bootstrap domain: {domain}") from exc

    def digest(self, domain: str) -> str:
        try:
            return self._digests[domain]
        except KeyError as exc:
            raise ScoreError(f"unknown bootstrap domain: {domain}") from exc

    def seed_for(self, domain: str) -> int:
        try:
            return self._seeds[domain]
        except KeyError as exc:
            raise ScoreError(f"unknown bootstrap domain: {domain}") from exc

    def metadata(self) -> dict[str, Any]:
        return {
            domain: {
                "seed": self.seed_for(domain),
                "records": self.records_by_domain[domain],
                "draws": self.draws,
                "sha256": self.digest(domain),
            }
            for domain in DOMAIN_ORDER
        }


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ScoreError(f"repository root is unavailable: {root}")
    return root


def _truth_free(payload: Mapping[str, Any], *, description: str) -> None:
    for key in (
        "truth_opened",
        "source_text_written",
        "source_text_loaded",
        "token_ids_written",
        "target_labels_loaded",
        "candidate_arrays_persisted",
    ):
        value = payload.get(key)
        if value is True or value == "true":
            raise ScoreError(f"{description} records truth/source access")


def _json_file(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ScoreError(f"{description} is not readable JSON") from exc
    if not isinstance(payload, Mapping):
        raise ScoreError(f"{description} is not a JSON object")
    result = dict(payload)
    _truth_free(result, description=description)
    return result


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise ScoreError(f"file is unavailable: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": gate.sha256_file(path)}


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _mapping_digest(values: Mapping[int, int]) -> str:
    canonical = json.dumps(
        [[int(token), int(count)] for token, count in sorted(values.items())],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalize_frequency_map(raw: Any, *, bank_id: str) -> dict[int, int]:
    if not isinstance(raw, Mapping):
        raise ScoreError(f"frequency reference bank {bank_id!r} is not a map")
    normalized: dict[int, int] = {}
    for token, count in raw.items():
        try:
            token_id = int(token)
            value = int(count)
        except (TypeError, ValueError) as exc:
            raise ScoreError(f"frequency bank {bank_id!r} contains a non-integer entry") from exc
        if not 0 <= token_id < gate.VOCABULARY_SIZE or value <= 0:
            raise ScoreError(f"frequency bank {bank_id!r} contains an invalid token/count")
        normalized[token_id] = value
    if not normalized:
        raise ScoreError(f"frequency bank {bank_id!r} is empty")
    return normalized


def _extract_frequency_banks(payload: Mapping[str, Any]) -> dict[str, dict[int, int]]:
    """Extract explicitly named B0/B1 (or legacy enriched) support maps."""

    banks: dict[str, dict[int, int]] = {}
    raw = payload.get("frequency_references")
    declared_bank = payload.get("frequency_bank", payload.get("bank_id", payload.get("reference_id")))
    declared_bank_text = str(declared_bank) if declared_bank is not None else None
    if isinstance(raw, Mapping):
        # Legacy payload: {"enriched": {token: count}}.
        if raw and all(not isinstance(value, Mapping) for value in raw.values()):
            raise ScoreError("frequency_references is not a named bank map")
        for name, value in raw.items():
            bank_id = str(name)
            selected = value
            if isinstance(value, Mapping):
                if isinstance(value.get("enriched"), Mapping):
                    selected = value["enriched"]
                elif isinstance(value.get("counts"), Mapping):
                    selected = value["counts"]
            if isinstance(selected, Mapping):
                banks[bank_id] = _normalize_frequency_map(selected, bank_id=bank_id)
    if not banks:
        counts = payload.get("counts")
        if isinstance(counts, Mapping):
            bank_id = declared_bank_text or "unlabelled"
            banks[bank_id] = _normalize_frequency_map(counts, bank_id=bank_id)
    if not banks:
        raise ScoreError("frequency reference lacks named public fitting-support maps")
    if declared_bank_text is not None and declared_bank_text not in banks and len(banks) == 1:
        only_id, only_map = next(iter(banks.items()))
        banks = {declared_bank_text: only_map}
    return banks


def _frequency_binding_name(bank: str, bindings: Mapping[str, Any]) -> str:
    canonical = f"frequency_reference_{bank}"
    lower = canonical.lower()
    for name in (canonical, lower):
        if isinstance(bindings.get(name), Mapping):
            return name
    raise ScoreError(
        "frozen registration must bind frequency_reference_B0 and frequency_reference_B1 "
        "before truth"
    )


def _load_bound_frequency_references(
    freeze: Mapping[str, Any],
    *,
    frequency_reference_paths: Mapping[str, Path] | None,
    frequency_reference_path: Path | None,
    frequency_counts: Mapping[int, int] | None,
    frequency_counts_by_bank: Mapping[str, Mapping[int, int]] | None,
) -> tuple[dict[str, dict[int, int]], dict[str, Any]]:
    """Load both frozen B0/B1 fitting-support maps before truth access.

    The registration binds each bank independently, even when both bindings
    point to one JSON file containing named B0/B1 maps. Runtime callers may
    supply paths only to assert the already-frozen binding; they cannot choose
    an arbitrary support map.
    """

    bindings = freeze.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise ScoreError("frozen input bindings are absent")
    binding_names = {bank: _frequency_binding_name(bank, bindings) for bank in ("B0", "B1")}
    if frequency_reference_paths is not None and set(frequency_reference_paths) != {"B0", "B1"}:
        raise ScoreError("frequency_reference_paths must bind exactly B0 and B1")
    if frequency_counts_by_bank is not None and set(frequency_counts_by_bank) != {"B0", "B1"}:
        raise ScoreError("frequency_counts_by_bank must bind exactly B0 and B1")
    if frequency_reference_path is not None and frequency_reference_paths is not None:
        raise ScoreError("use frequency_reference_paths or legacy frequency_reference_path, not both")

    banks: dict[str, dict[int, int]] = {}
    records: dict[str, dict[str, Any]] = {}
    for bank, binding_name in binding_names.items():
        declared = bindings[binding_name]
        declared_path = Path(str(declared["path"])).expanduser().resolve()
        if frequency_reference_paths is not None:
            path = Path(frequency_reference_paths[bank]).expanduser().resolve()
        elif frequency_reference_path is not None:
            path = Path(frequency_reference_path).expanduser().resolve()
        else:
            path = declared_path
        if path != declared_path:
            raise ScoreError(f"frequency reference {bank} differs from frozen input binding")
        actual = _file_record(path)
        for key in ("path", "bytes", "sha256"):
            if actual[key] != declared.get(key):
                raise ScoreError(f"frequency reference {bank} binding changed: {key}")
        payload = _json_file(path, description=f"frequency reference {bank}")
        named = _extract_frequency_banks(payload)
        if bank not in named:
            raise ScoreError(f"frequency reference file does not contain frozen bank {bank}")
        banks[bank] = named[bank]
        record = dict(actual)
        record.update(
            {
                "bank_id": bank,
                "support_token_count": len(banks[bank]),
                "support_map_sha256": _mapping_digest(banks[bank]),
                "binding_name": binding_name,
            }
        )
        records[bank] = record

    if frequency_counts_by_bank is not None:
        for bank, supplied_raw in frequency_counts_by_bank.items():
            supplied = _normalize_frequency_map(supplied_raw, bank_id=bank)
            if supplied != banks[bank]:
                raise ScoreError(f"supplied frequency counts differ from frozen {bank} reference")
    if frequency_counts is not None:
        # Legacy direct counts are accepted only when both frozen maps are
        # exactly the same; otherwise a bank-specific mapping is required.
        supplied = _normalize_frequency_map(frequency_counts, bank_id="legacy")
        if any(supplied != banks[bank] for bank in ("B0", "B1")):
            raise ScoreError("supplied frequency counts differ from frozen B0/B1 references")

    return banks, {"banks": records, "bank_ids": ["B0", "B1"]}


# Retain this validator for callers that inspect named support maps directly.
def _resolve_method_frequency_banks(
    banks: Mapping[str, Mapping[int, int]],
    requested: Mapping[str, str] | None,
) -> dict[str, str]:
    if requested is None:
        if len(banks) != 1:
            raise ScoreError("multiple frequency banks require all methods to be scored under each frozen bank")
        only = next(iter(banks))
        return {method: only for method in gate.METHOD_ORDER}
    if set(requested) != set(gate.METHOD_ORDER):
        raise ScoreError("method-to-frequency-bank binding is incomplete or foreign")
    result = {str(method): str(bank) for method, bank in requested.items()}
    unknown = sorted(set(result.values()) - set(banks))
    if unknown:
        raise ScoreError(f"method-to-frequency-bank binding names unknown banks: {unknown}")
    return result

def _load_predictions(freeze: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    bindings = freeze.get("predictions")
    if not isinstance(bindings, Mapping):
        raise ScoreError("frozen prediction matrix is absent")
    expected = {f"{method}::{cell}" for method in gate.METHOD_ORDER for cell in gate.CELL_ORDER}
    if set(bindings) != expected:
        raise ScoreError("frozen prediction matrix is incomplete or contains foreign keys")
    result: dict[str, dict[str, torch.Tensor]] = {method: {} for method in gate.METHOD_ORDER}
    for method in gate.METHOD_ORDER:
        for cell in gate.CELL_ORDER:
            key = f"{method}::{cell}"
            binding = bindings[key]
            if not isinstance(binding, Mapping):
                raise ScoreError(f"prediction binding is malformed: {key}")
            path = Path(str(binding.get("path"))).expanduser().resolve()
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != {"predictions"}:
                        raise ScoreError(f"prediction tensor keys changed: {key}")
                    values = handle.get_tensor("predictions").detach().cpu().contiguous()
            except ScoreError:
                raise
            except Exception as exc:
                raise ScoreError(f"prediction artifact is unreadable: {key}") from exc
            if not _is_integer_dtype(values.dtype):
                raise ScoreError(f"prediction dtype is not integer: {key}")
            if tuple(values.shape) != (gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS):
                raise ScoreError(f"prediction geometry changed: {key}")
            if values[:, 0].ne(gate.BOS_TOKEN_ID).any().item():
                raise ScoreError(f"prediction BOS column changed: {key}")
            if values.lt(0).any().item() or values.ge(gate.VOCABULARY_SIZE).any().item():
                raise ScoreError(f"prediction contains invalid token IDs: {key}")
            digest = gate.tensor_digest(values)
            if digest != binding.get("prediction_sha256"):
                raise ScoreError(f"prediction tensor digest changed: {key}")
            result[method][cell] = values.to(dtype=torch.long)
    return result


def _load_truth(truth_value: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(truth_value, Mapping) or set(truth_value) != set(gate.CELL_ORDER):
        raise ScoreError("truth loader must return exactly the frozen four cells")
    result: dict[str, torch.Tensor] = {}
    for cell in gate.CELL_ORDER:
        values = torch.as_tensor(truth_value[cell]).detach().cpu().contiguous()
        if not _is_integer_dtype(values.dtype):
            raise ScoreError(f"truth dtype is not integer: {cell}")
        if tuple(values.shape) != (gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS):
            raise ScoreError(f"truth geometry changed: {cell}")
        if values[:, 0].ne(gate.BOS_TOKEN_ID).any().item():
            raise ScoreError(f"truth BOS column changed: {cell}")
        if values[:, 1:].lt(0).any().item() or values[:, 1:].ge(gate.VOCABULARY_SIZE).any().item():
            raise ScoreError(f"truth contains invalid token IDs: {cell}")
        result[cell] = values.to(dtype=torch.long)
    paired_targets: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        base = result[f"{domain}__public_base"]
        lora = result[f"{domain}__public_lora_2601"]
        if not torch.equal(base, lora):
            raise ScoreError(f"truth target cells are not source-paired and identical: {domain}")
        paired_targets[domain] = {
            "records": int(base.shape[0]),
            "tensor_sha256": gate.tensor_digest(base),
            "target_cells": [f"{domain}__public_base", f"{domain}__public_lora_2601"],
        }
    return result, paired_targets


def _frequency_bin(count: int) -> str:
    if count <= 0:
        return "unseen_0"
    if count <= 4:
        return "seen_1_4"
    if count <= 9:
        return "seen_5_9"
    if count <= 49:
        return "seen_10_49"
    return "seen_50_plus"


def _cell_raw(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    frequency_counts: Mapping[int, int],
) -> dict[str, Any]:
    correct = prediction[:, 1:].eq(truth[:, 1:])
    exact = correct.all(dim=1)
    strata: dict[str, dict[str, list[int]]] = {}
    bins = [[_frequency_bin(int(frequency_counts.get(int(token), 0))) for token in row] for row in truth[:, 1:]]
    for name in FREQUENCY_BINS:
        exposed = [sum(item == name for item in row) for row in bins]
        correct_by_record = [
            int(sum(bool(ok) for ok, item in zip(ok_row.tolist(), bin_row) if item == name))
            for ok_row, bin_row in zip(correct, bins)
        ]
        strata[name] = {"correct": correct_by_record, "exposed": exposed}
    return {
        "exact": [bool(value) for value in exact.tolist()],
        "token_correct": [int(value) for value in correct.sum(dim=1).tolist()],
        "strata": strata,
        "records": int(prediction.shape[0]),
        "scored_post_bos_tokens": int(correct.numel()),
        "token_correct_total": int(correct.sum().item()),
        "exact_correct": int(exact.sum().item()),
    }


def _percentile_bounds(values: torch.Tensor, *, one_sided_alpha: float) -> tuple[float, float]:
    if values.numel() <= 0:
        raise ScoreError("bootstrap produced no estimates")
    ordered = torch.sort(values.to(dtype=torch.float64).reshape(-1)).values
    count = int(ordered.numel())
    lower_index = max(0, min(count - 1, int(math.floor(float(one_sided_alpha) * count))))
    upper_index = max(0, min(count - 1, int(math.ceil((1.0 - float(one_sided_alpha)) * count) - 1)))
    return float(ordered[lower_index].item()), float(ordered[upper_index].item())


def _shared_delta(
    values: Sequence[float],
    *,
    plan: _BootstrapPlan,
    domain: str,
    one_sided_alpha: float,
) -> dict[str, Any]:
    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if tensor.numel() <= 0:
        raise ScoreError("source-record delta is empty")
    if not torch.isfinite(tensor).all().item():
        raise ScoreError("source-record delta contains non-finite values")
    point = float(tensor.mean().item())
    zero_variance = bool(torch.all(tensor == tensor[0]).item())
    result: dict[str, Any] = {
        "status": "UNKNOWN" if zero_variance else "COMPUTED",
        "unit": "source_record",
        "records": int(tensor.numel()),
        "point": point,
        "draws": plan.draws,
        "seed": plan.seed_for(domain),
        "one_sided_alpha": float(one_sided_alpha),
        "percentile_convention": analysis.PERCENTILE_CONVENTION,
        "zero_variance_unknown": zero_variance,
        "resample_index_sha256": plan.digest(domain),
    }
    if zero_variance:
        result["reason"] = "zero_observed_difference_variance"
        return result
    estimates = tensor[plan.indices(domain)].mean(dim=1)
    result["lower"], result["upper"] = _percentile_bounds(estimates, one_sided_alpha=one_sided_alpha)
    return result


def _shared_ratio(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    plan: _BootstrapPlan,
    domain: str,
    one_sided_alpha: float,
) -> dict[str, Any]:
    numerator = torch.as_tensor(list(numerators), dtype=torch.float64)
    denominator = torch.as_tensor(list(denominators), dtype=torch.float64)
    if numerator.numel() <= 0 or numerator.numel() != denominator.numel():
        raise ScoreError("ratio record geometry changed")
    if not torch.isfinite(numerator).all().item() or not torch.isfinite(denominator).all().item():
        raise ScoreError("ratio inputs contain non-finite values")
    observed_denominator = float(denominator.sum().item())
    base: dict[str, Any] = {
        "unit": "source_record",
        "records": int(numerator.numel()),
        "draws": plan.draws,
        "seed": plan.seed_for(domain),
        "one_sided_alpha": float(one_sided_alpha),
        "percentile_convention": analysis.PERCENTILE_CONVENTION,
        "observed_numerator": float(numerator.sum().item()),
        "observed_denominator": observed_denominator,
        "resample_index_sha256": plan.digest(domain),
    }
    if observed_denominator <= 0.0:
        return {**base, "status": "UNKNOWN", "reason": "nonpositive_observed_denominator"}
    sampled_denominator = denominator[plan.indices(domain)].sum(dim=1)
    bad = sampled_denominator <= 0.0
    if bool(bad.any().item()):
        return {
            **base,
            "status": "UNKNOWN",
            "reason": "nonpositive_bootstrap_denominator",
            "nonpositive_bootstrap_draws": int(bad.sum().item()),
            "point": float(numerator.sum().item() / observed_denominator),
        }
    sampled_numerator = numerator[plan.indices(domain)].sum(dim=1)
    estimates = sampled_numerator / sampled_denominator
    lower, upper = _percentile_bounds(estimates, one_sided_alpha=one_sided_alpha)
    return {
        **base,
        "status": "COMPUTED",
        "point": float(numerator.sum().item() / observed_denominator),
        "lower": lower,
        "upper": upper,
        "nonpositive_bootstrap_draws": 0,
    }


def _shared_token_delta(
    candidate_correct: Sequence[int],
    control_correct: Sequence[int],
    *,
    plan: _BootstrapPlan,
    domain: str,
    one_sided_alpha: float,
) -> dict[str, Any]:
    if len(candidate_correct) != len(control_correct):
        raise ScoreError("paired token counts have different record counts")
    exposures = gate.SCORED_POST_BOS_TOKENS
    if any(value < 0 or value > exposures for value in candidate_correct) or any(value < 0 or value > exposures for value in control_correct):
        raise ScoreError("paired token counts exceed fixed exposure")
    result = _shared_delta(
        [(float(left) - float(right)) / float(exposures) for left, right in zip(candidate_correct, control_correct)],
        plan=plan,
        domain=domain,
        one_sided_alpha=one_sided_alpha,
    )
    result.update(
        {
            "candidate_correct": float(sum(candidate_correct)),
            "control_correct": float(sum(control_correct)),
            "exposed_tokens": float(exposures * len(candidate_correct)),
        }
    )
    return result


def _unknown_interval(reason: str, *, plan: _BootstrapPlan, domain: str, exposed_source_records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "UNKNOWN",
        "reason": reason,
        "resample_index_sha256": plan.digest(domain),
    }
    if exposed_source_records is not None:
        result["exposed_source_records"] = exposed_source_records
    return result


def _stratum_summary(raw: Mapping[str, Any], *, plan: _BootstrapPlan, domain: str) -> dict[str, Any]:
    correct = [int(value) for value in raw["correct"]]
    exposed = [int(value) for value in raw["exposed"]]
    total_exposed = sum(exposed)
    exposed_sources = sum(value > 0 for value in exposed)
    row: dict[str, Any] = {
        "correct_tokens": sum(correct),
        "exposed_tokens": total_exposed,
        "exposed_source_records": exposed_sources,
        "token_accuracy": (float(sum(correct)) / float(total_exposed)) if total_exposed else None,
    }
    if total_exposed <= 0:
        row["interval"] = _unknown_interval("zero_exposed_tokens", plan=plan, domain=domain)
    elif exposed_sources < 32:
        row["interval"] = _unknown_interval(
            "fewer_than_32_exposed_source_records", plan=plan, domain=domain, exposed_source_records=exposed_sources
        )
    else:
        row["interval"] = _shared_ratio(
            correct,
            exposed,
            plan=plan,
            domain=domain,
            one_sided_alpha=analysis.TRR0010_HARM_ALPHA,
        )
    return row


def _frequency_contrast(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    plan: _BootstrapPlan,
    domain: str,
) -> dict[str, Any]:
    left_exposed = [int(value) for value in left["exposed"]]
    right_exposed = [int(value) for value in right["exposed"]]
    if left_exposed != right_exposed:
        raise ScoreError("paired frequency exposure differs between methods")
    exposed = left_exposed
    exposed_sources = sum(value > 0 for value in exposed)
    delta = [int(a) - int(b) for a, b in zip(left["correct"], right["correct"])]
    total_exposed = sum(exposed)
    result: dict[str, Any] = {
        "correct_delta_tokens": sum(delta),
        "exposed_tokens": total_exposed,
        "exposed_source_records": exposed_sources,
    }
    if total_exposed <= 0:
        result["interval"] = _unknown_interval("zero_exposed_tokens", plan=plan, domain=domain)
    elif exposed_sources < 32:
        result["interval"] = _unknown_interval(
            "fewer_than_32_exposed_source_records", plan=plan, domain=domain, exposed_source_records=exposed_sources
        )
    else:
        ratios = [float(d) / float(e) for d, e in zip(delta, exposed) if e > 0]
        if ratios and len(set(ratios)) == 1:
            result["interval"] = _unknown_interval("zero_observed_difference_variance", plan=plan, domain=domain)
            result["interval"]["point"] = float(sum(delta)) / float(total_exposed)
        else:
            result["interval"] = _shared_ratio(
                delta,
                exposed,
                plan=plan,
                domain=domain,
                one_sided_alpha=analysis.TRR0010_HARM_ALPHA,
            )
    return result


def _contrast(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    frequency_banks: Mapping[str, Mapping[int, int]],
    plan: _BootstrapPlan,
    domain: str,
) -> dict[str, Any]:
    records = int(left["records"])
    if records != int(right["records"]):
        raise ScoreError("paired contrast record counts differ")
    # Every method is scored against both frozen maps. This keeps data-factor
    # contrasts on the same exposure bins even when the arms used different
    # fitting banks.
    frequency = {
        bank_id: {
            name: _frequency_contrast(
                left["strata_by_bank"][bank_id][name],
                right["strata_by_bank"][bank_id][name],
                plan=plan,
                domain=domain,
            )
            for name in FREQUENCY_BINS
        }
        for bank_id in ("B0", "B1")
    }
    return {
        "records": records,
        "frequency_banks": {"available": ["B0", "B1"], "comparable_within_each_bank": True},
        "exact": {
            "benefit": analysis.paired_exact_cp(left["exact"], right["exact"], alpha=analysis.TRR0010_BENEFIT_ALPHA),
            "harm": analysis.paired_exact_cp(left["exact"], right["exact"], alpha=analysis.TRR0010_HARM_ALPHA),
        },
        "token": {
            "benefit": _shared_token_delta(
                left["token_correct"], right["token_correct"], plan=plan, domain=domain,
                one_sided_alpha=analysis.TRR0010_BENEFIT_ALPHA,
            ),
            "harm": _shared_token_delta(
                left["token_correct"], right["token_correct"], plan=plan, domain=domain,
                one_sided_alpha=analysis.TRR0010_HARM_ALPHA,
            ),
        },
        "frequency": frequency,
    }

def _remaining_error_reduction(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    candidate_method: str,
    control_method: str,
    plan: _BootstrapPlan,
    domain: str,
) -> dict[str, Any]:
    """Compute control-error reduction for one predeclared route."""

    candidate_token_errors = [gate.SCORED_POST_BOS_TOKENS - int(value) for value in candidate["token_correct"]]
    control_token_errors = [gate.SCORED_POST_BOS_TOKENS - int(value) for value in control["token_correct"]]
    candidate_exact_errors = [0 if bool(value) else 1 for value in candidate["exact"]]
    control_exact_errors = [0 if bool(value) else 1 for value in control["exact"]]
    return {
        "candidate_method": candidate_method,
        "control_method": control_method,
        "token": _shared_ratio(
            [float(c) - float(k) for c, k in zip(control_token_errors, candidate_token_errors)],
            control_token_errors,
            plan=plan,
            domain=domain,
            one_sided_alpha=analysis.TRR0010_BENEFIT_ALPHA,
        ),
        "exact": _shared_ratio(
            [float(c) - float(k) for c, k in zip(control_exact_errors, candidate_exact_errors)],
            control_exact_errors,
            plan=plan,
            domain=domain,
            one_sided_alpha=analysis.TRR0010_BENEFIT_ALPHA,
        ),
    }

def _gap_closure(
    candidate: Mapping[str, Any],
    anchor: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    candidate_method: str,
    plan: _BootstrapPlan,
    domain: str,
) -> dict[str, Any]:
    candidate_token = [float(value) / float(gate.SCORED_POST_BOS_TOKENS) for value in candidate["token_correct"]]
    anchor_token = [float(value) / float(gate.SCORED_POST_BOS_TOKENS) for value in anchor["token_correct"]]
    baseline_token = [float(value) / float(gate.SCORED_POST_BOS_TOKENS) for value in baseline["token_correct"]]
    candidate_exact = [1.0 if value else 0.0 for value in candidate["exact"]]
    anchor_exact = [1.0 if value else 0.0 for value in anchor["exact"]]
    baseline_exact = [1.0 if value else 0.0 for value in baseline["exact"]]
    return {
        "candidate_method": candidate_method,
        "anchor_method": gate.A1_A2_METHOD_ID,
        "baseline_method": gate.UNCHANGED_METHOD_ID,
        "formula": "(candidate-baseline)/(a1_a2-baseline), per source-record bootstrap draw",
        "token": _shared_ratio(
            [c - b for c, b in zip(candidate_token, baseline_token)],
            [a - b for a, b in zip(anchor_token, baseline_token)],
            plan=plan,
            domain=domain,
            one_sided_alpha=analysis.TRR0010_BENEFIT_ALPHA,
        ),
        "exact": _shared_ratio(
            [c - b for c, b in zip(candidate_exact, baseline_exact)],
            [a - b for a, b in zip(anchor_exact, baseline_exact)],
            plan=plan,
            domain=domain,
            one_sided_alpha=analysis.TRR0010_BENEFIT_ALPHA,
        ),
    }


def _interaction(
    expanded_directional: Mapping[str, Any],
    expanded_fixed: Mapping[str, Any],
    current_directional: Mapping[str, Any],
    current_fixed: Mapping[str, Any],
    *,
    plan: _BootstrapPlan,
    domain: str,
) -> dict[str, Any]:
    exact_delta = [
        (int(bool(ed)) - int(bool(ef))) - (int(bool(cd)) - int(bool(cf)))
        for ed, ef, cd, cf in zip(
            expanded_directional["exact"], expanded_fixed["exact"], current_directional["exact"], current_fixed["exact"]
        )
    ]
    token_delta = [
        ((float(ed) - float(ef)) - (float(cd) - float(cf))) / float(gate.SCORED_POST_BOS_TOKENS)
        for ed, ef, cd, cf in zip(
            expanded_directional["token_correct"], expanded_fixed["token_correct"], current_directional["token_correct"], current_fixed["token_correct"]
        )
    ]
    return {
        "definition": "(expanded_directional-expanded_fixed)-(current_directional-current_fixed)",
        "records": int(expanded_directional["records"]),
        "exact": {
            "benefit": _shared_delta(exact_delta, plan=plan, domain=domain, one_sided_alpha=analysis.TRR0010_BENEFIT_ALPHA),
            "harm": _shared_delta(exact_delta, plan=plan, domain=domain, one_sided_alpha=analysis.TRR0010_HARM_ALPHA),
        },
        "token": {
            "benefit": _shared_delta(token_delta, plan=plan, domain=domain, one_sided_alpha=analysis.TRR0010_BENEFIT_ALPHA),
            "harm": _shared_delta(token_delta, plan=plan, domain=domain, one_sided_alpha=analysis.TRR0010_HARM_ALPHA),
        },
    }


def _check_interval(interval: Mapping[str, Any], predicate: Callable[[float], bool]) -> dict[str, Any]:
    if interval.get("status") != "COMPUTED":
        return {"status": "UNKNOWN", "reason": interval.get("reason", "interval_not_computed")}
    value = interval.get("lower")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return {"status": "UNKNOWN", "reason": "interval_lower_bound_missing"}
    return {"status": "PASS" if predicate(float(value)) else "FAIL", "lower": float(value)}


def _combine_statuses(rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(row.get("status")) for row in rows}
    if "FAIL" in statuses:
        return "FAIL"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    return "PASS"


def _absolute_route_check(contrast: Mapping[str, Any]) -> dict[str, Any]:
    token = contrast["token"]
    exact = contrast["exact"]
    checks = {
        "token_point_floor": {
            "status": "PASS" if float(token["benefit"]["point"]) >= PRACTICAL_THRESHOLDS["token_accuracy_floor"] else "FAIL",
            "point": float(token["benefit"]["point"]),
            "threshold": PRACTICAL_THRESHOLDS["token_accuracy_floor"],
        },
        "token_positive_lower": _check_interval(token["benefit"], lambda value: value > 0.0),
        "token_harm_safeguard": _check_interval(token["harm"], lambda value: value >= PRACTICAL_THRESHOLDS["harm_token_lower"]),
        "exact_point_floor": {
            "status": "PASS" if float(exact["benefit"]["point"]) >= PRACTICAL_THRESHOLDS["exact_point_floor"] else "FAIL",
            "point": float(exact["benefit"]["point"]),
            "threshold": PRACTICAL_THRESHOLDS["exact_point_floor"],
        },
        "exact_positive_lower": _check_interval(exact["benefit"], lambda value: value > 0.0),
        "exact_harm_safeguard": _check_interval(exact["harm"], lambda value: value >= PRACTICAL_THRESHOLDS["harm_exact_lower"]),
    }
    checks["status"] = _combine_statuses(list(checks.values()))
    return checks


def _ratio_check(interval: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    if interval.get("status") != "COMPUTED":
        return {"status": "UNKNOWN", "reason": interval.get("reason", "interval_not_computed"), "threshold": threshold}
    point = interval.get("point")
    lower = interval.get("lower")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (point, lower)):
        return {"status": "UNKNOWN", "reason": "ratio_bounds_missing", "threshold": threshold}
    return {
        "status": "PASS" if float(point) >= threshold and float(lower) >= threshold else "FAIL",
        "point": float(point),
        "lower": float(lower),
        "threshold": threshold,
    }


def _decision_readout(
    contrasts: Mapping[str, Mapping[str, Any]],
    remaining_error_reduction: Mapping[str, Mapping[str, Any]],
    gap_closure: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Serialize per-cell v4 checks without pooling or choosing a winner."""

    directional_cells: dict[str, Any] = {}
    data_cells: dict[str, Any] = {}
    for cell in gate.CELL_ORDER:
        directional_expanded = _absolute_route_check(contrasts["directional_expanded"][cell])
        directional_current = _absolute_route_check(contrasts["directional_current"][cell])
        data_fixed = _absolute_route_check(contrasts["data_fixed"][cell])
        data_expanded_unchanged = _absolute_route_check(contrasts["data_expanded_fixed_vs_unchanged"][cell])
        directional_gap = gap_closure["directional_candidate"][cell]
        data_gap = gap_closure["data_expanded_fixed"][cell]
        directional_gap_checks = {
            "token_visibility": {
                "status": "PASS" if float(contrasts["combined_candidate"][cell]["token"]["benefit"]["point"]) >= PRACTICAL_THRESHOLDS["gap_visibility_token"] else "FAIL",
                "point": float(contrasts["combined_candidate"][cell]["token"]["benefit"]["point"]),
                "threshold": PRACTICAL_THRESHOLDS["gap_visibility_token"],
            },
            "exact_visibility": {
                "status": "PASS" if float(contrasts["combined_candidate"][cell]["exact"]["benefit"]["point"]) >= PRACTICAL_THRESHOLDS["gap_visibility_exact"] else "FAIL",
                "point": float(contrasts["combined_candidate"][cell]["exact"]["benefit"]["point"]),
                "threshold": PRACTICAL_THRESHOLDS["gap_visibility_exact"],
            },
            "token_fraction": _ratio_check(directional_gap["token"], PRACTICAL_THRESHOLDS["gap_minimum_fraction"]),
            "exact_fraction": _ratio_check(directional_gap["exact"], PRACTICAL_THRESHOLDS["gap_minimum_fraction"]),
        }
        directional_gap_checks["status"] = _combine_statuses(list(directional_gap_checks.values()))
        data_gap_checks = {
            "token_visibility": {
                "status": "PASS" if float(contrasts["data_expanded_fixed_vs_unchanged"][cell]["token"]["benefit"]["point"]) >= PRACTICAL_THRESHOLDS["gap_visibility_token"] else "FAIL",
                "point": float(contrasts["data_expanded_fixed_vs_unchanged"][cell]["token"]["benefit"]["point"]),
                "threshold": PRACTICAL_THRESHOLDS["gap_visibility_token"],
            },
            "exact_visibility": {
                "status": "PASS" if float(contrasts["data_expanded_fixed_vs_unchanged"][cell]["exact"]["benefit"]["point"]) >= PRACTICAL_THRESHOLDS["gap_visibility_exact"] else "FAIL",
                "point": float(contrasts["data_expanded_fixed_vs_unchanged"][cell]["exact"]["benefit"]["point"]),
                "threshold": PRACTICAL_THRESHOLDS["gap_visibility_exact"],
            },
            "token_fraction": _ratio_check(data_gap["token"], PRACTICAL_THRESHOLDS["gap_minimum_fraction"]),
            "exact_fraction": _ratio_check(data_gap["exact"], PRACTICAL_THRESHOLDS["gap_minimum_fraction"]),
        }
        data_gap_checks["status"] = _combine_statuses(list(data_gap_checks.values()))
        directional_cells[cell] = {
            "expanded_directional_vs_expanded_fixed": directional_expanded,
            "expanded_directional_vs_current_fixed": directional_current,
            "gap_closure": directional_gap_checks,
            "status": _combine_statuses([directional_expanded, directional_current, directional_gap_checks]),
        }
        data_cells[cell] = {
            "expanded_fixed_vs_current_fixed": data_fixed,
            "expanded_fixed_vs_unchanged": data_expanded_unchanged,
            "gap_closure": data_gap_checks,
            "status": _combine_statuses([data_fixed, data_expanded_unchanged, data_gap_checks]),
        }

    def by_domain(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for domain in DOMAIN_ORDER:
            selected = [row for cell, row in rows.items() if DOMAIN_BY_CELL[cell] == domain]
            result[domain] = {"status": _combine_statuses(selected), "target_cells": [cell for cell in rows if DOMAIN_BY_CELL[cell] == domain]}
        return result

    return {
        "status": "CHECKS_COMPUTED_NO_POOLED_DECISION",
        "thresholds_applied": True,
        "thresholds": dict(PRACTICAL_THRESHOLDS),
        "directional": {"by_cell": directional_cells, "by_domain": by_domain(directional_cells)},
        "data": {"by_cell": data_cells, "by_domain": by_domain(data_cells)},
        "remaining_error_reduction": {
            route: {
                cell: {
                    "token": _ratio_check(row["token"], PRACTICAL_THRESHOLDS["remaining_error_reduction_fraction"]),
                    "exact": _ratio_check(row["exact"], PRACTICAL_THRESHOLDS["remaining_error_reduction_fraction"]),
                }
                for cell, row in rows.items()
            }
            for route, rows in remaining_error_reduction.items()
        },
        "cost_gate": {"status": "UNKNOWN", "reason": "timing and deployment cost evidence is consumed by the planner"},
        "rare_absent_strata": "DIAGNOSTIC_ONLY",
    }


def _score_loaded(
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
    truth: Mapping[str, torch.Tensor],
    frequency_banks: Mapping[str, Mapping[int, int]],
    *,
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    if set(frequency_banks) != {"B0", "B1"}:
        raise ScoreError("score requires both frozen B0 and B1 frequency references")
    plan = _BootstrapPlan(
        seed=bootstrap_seed,
        draws=bootstrap_draws,
        records_by_domain={domain: gate.RECORDS_BY_DOMAIN[domain] for domain in DOMAIN_ORDER},
    )
    raw: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in gate.METHOD_ORDER}
    cell_scores: dict[str, dict[str, Any]] = {method: {} for method in gate.METHOD_ORDER}
    for method in gate.METHOD_ORDER:
        if set(predictions.get(method, {})) != set(gate.CELL_ORDER):
            raise ScoreError(f"prediction matrix cells changed: {method}")
        for cell in gate.CELL_ORDER:
            domain = DOMAIN_BY_CELL[cell]
            items_by_bank = {
                bank_id: _cell_raw(predictions[method][cell], truth[cell], counts)
                for bank_id, counts in frequency_banks.items()
            }
            base_item = next(iter(items_by_bank.values()))
            item = dict(base_item)
            item["strata_by_bank"] = {bank: row["strata"] for bank, row in items_by_bank.items()}
            raw[method][cell] = item
            token_total = item["scored_post_bos_tokens"]
            records = item["records"]
            cell_scores[method][cell] = {
                "records": records,
                "scored_post_bos_tokens": token_total,
                "scored_domain": domain,
                "token_correct": item["token_correct_total"],
                "token_errors": token_total - item["token_correct_total"],
                "token_accuracy": float(item["token_correct_total"]) / float(token_total),
                "exact_correct": item["exact_correct"],
                "exact_errors": records - item["exact_correct"],
                "exact_rate": float(item["exact_correct"]) / float(records),
                "frequency_strata": {
                    bank_id: {
                        name: _stratum_summary(items_by_bank[bank_id]["strata"][name], plan=plan, domain=domain)
                        for name in FREQUENCY_BINS
                    }
                    for bank_id in ("B0", "B1")
                },
            }

    contrasts: dict[str, dict[str, Any]] = {}
    for name, (left_id, right_id) in CONTRASTS.items():
        contrasts[name] = {
            cell: _contrast(
                raw[left_id][cell],
                raw[right_id][cell],
                frequency_banks=frequency_banks,
                plan=plan,
                domain=DOMAIN_BY_CELL[cell],
            )
            for cell in gate.CELL_ORDER
        }

    remaining_error_reduction = {
        route: {
            cell: _remaining_error_reduction(
                raw[left_id][cell],
                raw[right_id][cell],
                candidate_method=left_id,
                control_method=right_id,
                plan=plan,
                domain=DOMAIN_BY_CELL[cell],
            )
            for cell in gate.CELL_ORDER
        }
        for route, (left_id, right_id) in ERROR_REDUCTION_ROUTES.items()
    }
    gap_closure = {
        "directional_candidate": {
            cell: _gap_closure(
                raw[gate.EXPANDED_DIRECTIONAL_METHOD_ID][cell],
                raw[gate.A1_A2_METHOD_ID][cell],
                raw[gate.UNCHANGED_METHOD_ID][cell],
                candidate_method=gate.EXPANDED_DIRECTIONAL_METHOD_ID,
                plan=plan,
                domain=DOMAIN_BY_CELL[cell],
            )
            for cell in gate.CELL_ORDER
        },
        "data_expanded_fixed": {
            cell: _gap_closure(
                raw[gate.EXPANDED_FIXED_METHOD_ID][cell],
                raw[gate.A1_A2_METHOD_ID][cell],
                raw[gate.UNCHANGED_METHOD_ID][cell],
                candidate_method=gate.EXPANDED_FIXED_METHOD_ID,
                plan=plan,
                domain=DOMAIN_BY_CELL[cell],
            )
            for cell in gate.CELL_ORDER
        },
    }
    interaction = {
        cell: _interaction(
            raw[gate.EXPANDED_DIRECTIONAL_METHOD_ID][cell],
            raw[gate.EXPANDED_FIXED_METHOD_ID][cell],
            raw[gate.CURRENT_DIRECTIONAL_METHOD_ID][cell],
            raw[gate.CURRENT_FIXED_METHOD_ID][cell],
            plan=plan,
            domain=DOMAIN_BY_CELL[cell],
        )
        for cell in gate.CELL_ORDER
    }
    decision_readout = _decision_readout(contrasts, remaining_error_reduction, gap_closure)
    return {
        "schema": SCORE_SCHEMA,
        "task_id": TASK_ID,
        "status": SCORE_STATUS,
        "method_order": list(gate.METHOD_ORDER),
        "cell_order": list(gate.CELL_ORDER),
        "method_roles": dict(gate.METHOD_ROLES),
        "frequency_reference_ids": ["B0", "B1"],
        "frequency_scoring": "all_methods_and_all_contrasts_under_each_frozen_bank",
        "cell_scores": cell_scores,
        "contrasts": contrasts,
        "remaining_error_reduction": remaining_error_reduction,
        "gap_closure": gap_closure,
        "interaction": interaction,
        "frequency_bins": list(FREQUENCY_BINS),
        "bootstrap": {
            "unit": "source_record",
            "seed": int(bootstrap_seed),
            "draws": int(bootstrap_draws),
            "benefit_alpha": analysis.TRR0010_BENEFIT_ALPHA,
            "harm_alpha": analysis.TRR0010_HARM_ALPHA,
            "percentile_convention": analysis.PERCENTILE_CONVENTION,
            "domains_and_cells_separate": True,
            "shared_source_indices_across_methods_and_targets": True,
            "index_streams": plan.metadata(),
        },
        "decision_readout": decision_readout,
    }

def _write_create_only(path: Path, payload: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    path = path.resolve()
    task_root = (repository_root / "experiments" / TASK_ID).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise ScoreError("score output must be under the task root") from exc
    if path.exists() or path.is_symlink():
        raise ScoreError(f"score output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return _file_record(path)


def score_after_gate(
    *,
    freeze_path: Path,
    repository_root: Path,
    truth_loader: Callable[[], Mapping[str, Any]],
    frequency_reference_path: Path | None = None,
    frequency_reference_paths: Mapping[str, Path] | None = None,
    frequency_counts: Mapping[int, int] | None = None,
    frequency_counts_by_bank: Mapping[str, Mapping[int, int]] | None = None,
    output_path: Path | None = None,
    bootstrap_seed: int = analysis.TRR0010_BOOTSTRAP_SEED,
    bootstrap_draws: int = analysis.TRR0010_BOOTSTRAP_DRAWS,
    require_current_head: bool = False,
) -> dict[str, Any]:
    """Validate public outputs, then open truth once and score paired cells."""

    root = _root(repository_root)
    if not callable(truth_loader):
        raise ScoreError("truth_loader must be callable")
    if int(bootstrap_draws) <= 0:
        raise ScoreError("bootstrap draws must be positive")

    # This is the only public gate call and precedes every truth-loader call.
    freeze = gate.validate_before_truth(
        freeze_path=freeze_path,
        repository_root=root,
        require_current_head=require_current_head,
    )
    bound_frequency, frequency_record = _load_bound_frequency_references(
        freeze,
        frequency_reference_paths=frequency_reference_paths,
        frequency_reference_path=frequency_reference_path,
        frequency_counts=frequency_counts,
        frequency_counts_by_bank=frequency_counts_by_bank,
    )
    predictions = _load_predictions(freeze)

    # The truth loader is deliberately invoked exactly once, after all public
    # prediction/resource checks. No loader/model/source calls occur below.
    truth, truth_pairing = _load_truth(truth_loader())
    result = _score_loaded(
        predictions,
        truth,
        bound_frequency,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_draws=int(bootstrap_draws),
    )
    freeze_record = _file_record(Path(freeze_path).expanduser().resolve())
    result.update(
        {
            "freeze_receipt": freeze_record,
            "public_gate": {
                "status": "PASS",
                "code_commit": freeze["code_commit"],
                "registration": freeze["registration"],
                "run_manifest": freeze["run_manifest"],
            },
            "frequency_reference": frequency_record,
            "truth_pairing": truth_pairing,
            "truth_opened": True,
            "truth_loader_invocations": 1,
            "target_labels_loaded": True,
            "source_text_or_model_access": False,
            "candidate_arrays_persisted": False,
        }
    )
    if output_path is not None:
        result["score_artifact"] = _write_create_only(output_path, result, repository_root=root)
    return result


__all__ = [
    "CONTRASTS",
    "DOMAIN_ORDER",
    "ERROR_REDUCTION_ROUTES",
    "FREQUENCY_BINS",
    "SCORE_SCHEMA",
    "SCORE_STATUS",
    "PRACTICAL_THRESHOLDS",
    "ScoreError",
    "score_after_gate",
]
