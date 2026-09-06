#!/usr/bin/env python3
"""Prospective TRR-0008 decision planning and identity-only inventory.

The ``power`` command writes the frozen planning assumptions and a light-weight
paired-discordance sensitivity table.  The ``inventory`` command is a
count-only audit of the already-established TRR-0007 natural source ranges.
It reuses the trusted renderer only transiently: source text and token IDs are
never written, and the output contains aggregate counts and identity
commitments.  Inventory does not select rows, capture activations, create
truth, or load a target model.

The P06 input is deliberately allow-listed to one byte-identical task-owned
copy of the approved opaque export.  Its source and H128 int32 sequence hash
values are used only as identity exclusions; no P06 source/provenance/results
artifact is opened.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

# Keep the documented ``python3 scripts/trr0008_plan.py ...`` command runnable
# without requiring callers to set PYTHONPATH.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


TASK_ID = "TRR-0008"
PLAN_SCHEMA = "token-reconstruction.trr0008-prospective-plan.v1"
POWER_SCHEMA = "token-reconstruction.trr0008-power-analysis.v1"
INVENTORY_SCHEMA = "token-reconstruction.trr0008-identity-inventory.v1"

STYLE_ORDER = ("pile", "finance")
SOURCE_RANGES = {"finance": [12000, 20000], "pile": [7000, 10000]}
SELECTION_SEED = 5005
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048

TRR7_ELIGIBILITY = Path("experiments/TRR-0007/selection/eligibility_inventory.json")
TRR7_EXCLUSIONS = Path("experiments/TRR-0007/selection/source_exclusions.json")
TRR7_SELECTION = Path("experiments/TRR-0007/selection/source_selection.json")
TRR7_FINAL_BANK = Path(
    "experiments/TRR-0007/support/broader_bank_v5/public_parent_exclusion_manifest.json"
)
TRR7_PARENT_ROWS = Path(
    "experiments/TRR-0007/support/broader_bank_v5/selected_parent_rows.json"
)
TRR7_CORPUS_PLAN = Path("experiments/TRR-0007/support/broader_bank_v5/corpus_plan.json")
TRR7_PREFIX_EXCLUSIONS = Path(
    "experiments/TRR-0007/support/public_fit_prefix_exclusions_v3.json"
)
P06_OPAQUE_ORIGINAL = Path(
    "/tmp/trr-p06/experiments/TRR-P06/setup/p06_opaque_source_sequence_reservation.json"
)
P06_OPAQUE = _REPOSITORY_ROOT / "experiments/TRR-0008/planning/approved_opaque/p06_opaque_source_sequence_reservation.json"
P06_OPAQUE_SHA256 = "231f924c774ace135e5870232b3aa276b9d565a927fc18ec5e64667688843f70"
P06_OPAQUE_BYTES = 80243
P04_OPAQUE = Path("/tmp/trr-p04/experiments/TRR-P04/coordination/reservation_hashes.json")
P04_OPAQUE_SHA256 = "98f8dfcab0977b4bcafa47d97a86a410ab37359b897b9b553746afa7df5c7904"


class PlanError(RuntimeError):
    """Raised for a failed closed planning or inventory check."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _file_descriptor(path: Path, *, expected_sha256: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PlanError(f"required metadata file is unavailable: {path}")
    actual_sha256 = _sha256_file(path)
    actual_bytes = int(path.stat().st_size)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise PlanError(
            f"metadata hash changed for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise PlanError(
            f"metadata size changed for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha256}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PlanError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise PlanError(f"{description} must be a JSON object")
    return dict(value)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PlanError(f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return _file_descriptor(path)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_quantile(probability: float) -> float:
    """Acklam-free bisection quantile for the fallback planning values."""

    if not 0.0 < probability < 1.0:
        raise ValueError("normal quantile requires a probability in (0,1)")
    lo, hi = -9.0, 9.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _normal_cdf(mid) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _normal_power(*, n: int, true_effect: float, margin: float, discordance_rate: float, alpha: float = 0.05) -> float:
    """Normal fallback for environments without SciPy (never the preferred path)."""

    variance = max(discordance_rate - true_effect * true_effect, 1e-15)
    z_alpha = _normal_quantile(1.0 - alpha)
    return _normal_cdf((true_effect - margin) * math.sqrt(n / variance) - z_alpha)


def _cp_arrays(n: int, *, alpha: float) -> tuple[Any, Any, str]:
    """Return one-sided Clopper-Pearson bounds for Bin(n,p).

    The lower bound is used for gains and the upper bound for losses.  Splitting
    alpha across the two component bounds gives a conservative one-sided
    (1-alpha) lower bound for p_gain - p_loss.
    """

    try:
        import numpy as np
        from scipy.stats import beta as beta_distribution
    except ImportError as exc:
        raise PlanError(
            "SciPy is required for the claimed exact CP planning power; install the pinned planning environment or do not report power"
        ) from exc
    component_alpha = alpha / 2.0
    counts = np.arange(n + 1, dtype=int)
    lower = np.zeros(n + 1, dtype=float)
    upper = np.ones(n + 1, dtype=float)
    if n > 0:
        lower[1:] = beta_distribution.ppf(component_alpha, counts[1:], n - counts[1:] + 1)
        upper[:-1] = beta_distribution.ppf(1.0 - component_alpha, counts[:-1] + 1, n - counts[:-1])
    return lower, upper, "scipy.stats.beta.ppf_clopper_pearson"


def _exact_multinomial_power(*, n: int, true_effect: float, margin: float, discordance_rate: float, alpha: float) -> tuple[float, str]:
    """Power of the exact CP gain-minus-loss lower-bound decision.

    Counts are generated under a prospective multinomial model with
    p_gain=(q+d)/2 and p_loss=(q-d)/2.  The conditional decomposition
    G~Bin(n,p_gain), L|G~Bin(n-G,p_loss/(1-p_gain)) avoids a large joint grid.
    """

    lower, upper, engine = _cp_arrays(n, alpha=alpha)
    if engine != "scipy.stats.beta.ppf_clopper_pearson":
        raise PlanError("exact CP power engine is unavailable")
    import numpy as np
    from scipy.stats import binom

    p_gain = (discordance_rate + true_effect) / 2.0
    p_loss = (discordance_rate - true_effect) / 2.0
    if not 0.0 <= p_loss <= p_gain <= 1.0:
        raise ValueError("invalid multinomial planning probabilities")
    g_values = np.arange(n + 1, dtype=int)
    g_probability = binom.pmf(g_values, n, p_gain)
    total = 0.0
    for g, weight in zip(g_values.tolist(), g_probability.tolist()):
        if weight == 0.0:
            continue
        maximum_loss = int(np.searchsorted(upper, lower[g] - margin, side="right") - 1)
        remaining = n - g
        maximum_loss = min(maximum_loss, remaining)
        if maximum_loss < 0:
            continue
        if remaining == 0:
            conditional = 1.0
        else:
            conditional_probability = p_loss / (1.0 - p_gain)
            conditional = float(binom.cdf(maximum_loss, remaining, conditional_probability))
        total += weight * conditional
    return float(total), engine


def _paired_power_rows(*, sample_sizes: Sequence[int], effects: Sequence[float], q: float, alpha: float, domain: str) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    engine = ""
    for n in sample_sizes:
        for effect in effects:
            positive, used_engine = _exact_multinomial_power(n=n, true_effect=effect, margin=0.0, discordance_rate=q, alpha=alpha)
            practical, used_engine_2 = _exact_multinomial_power(n=n, true_effect=effect, margin=0.05, discordance_rate=q, alpha=alpha)
            engine = used_engine if used_engine == used_engine_2 else f"{used_engine}+{used_engine_2}"
            rows.append(
                {
                    "domain": domain,
                    "records_per_domain": int(n),
                    "true_effect_pp": round(100.0 * effect, 8),
                    "positive_margin_pp": 0.0,
                    "practical_exact_margin_pp": 5.0,
                    "positive_power": positive,
                    "practical_exact_power": practical,
                }
            )
    return rows, engine


def _decision_contract(*, output: Path, power_path: Path, finance_n: int, pile_n: int) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema": "token-reconstruction.trr0008-decision-contract.v1",
        "task_id": TASK_ID,
        "status": "PROSPECTIVE_DRAFT_PENDING_OWNER_FREEZE",
        "created_utc": _utc_now(),
        "methods": {
            "candidate": "improved_public_bank__residual_mlp512",
            "reference": "trr6__enriched_trained_diagonal_attention128",
            "credible_alternative": "current_enriched__residual_mlp512",
            "diagnostic": "improved_public_bank__trained_diagonal",
            "a1_a2_fresh_run": False,
            "a1_a2_reason": "TRR-0007's bounded anchor already quantifies the remaining A1+A2 gap; this task selects among frozen standalone checkpoints.",
        },
        "panel": {
            "finance_records_per_domain": finance_n,
            "pile_records_per_domain": pile_n,
            "target_conditions": ["public_base", "public_lora_2601"],
            "pairing": "identical source records are paired across public_base and public_lora_2601; target conditions are not independent sources",
            "clip_tokens_including_bos": SEQUENCE_TOKENS,
            "scored_post_bos_tokens": SCORED_POST_BOS,
            "natural_ranges_half_open": SOURCE_RANGES,
            "selection_seed": SELECTION_SEED,
        },
        "primary": {
            "cell": "finance__public_base",
            "endpoint": "record_exact_recovery",
            "contrast": "candidate minus frozen reference",
            "route_alpha": 0.025,
            "positive_rule": "one-sided 97.5% paired exact lower confidence bound > 0 percentage points",
            "practical_rule": "one-sided 97.5% paired exact lower confidence bound >= +5 percentage points",
            "interval": "exact paired discordance: lower Clopper-Pearson bound for gains minus upper Clopper-Pearson bound for losses, with route alpha/2 allocated to each component",
            "record_unit": "one 128-token including-BOS source row; exact means all 127 post-BOS positions correct",
        },
        "token_endpoint": {
            "endpoint": "post_bos_token_accuracy",
            "route_alpha": 0.025,
            "practical_rule": "one-sided 97.5% paired record-level lower confidence bound >= +1 percentage point",
            "positive_rule": "one-sided 97.5% paired record-level lower confidence bound > 0 percentage points",
            "interval": "paired record bootstrap, fixed seed declared by evaluator before truth; token positions are nested within records and are not independent records",
            "role": "second predeclared quality route for practical advance; exact and token routes share a split alpha family and remain separately labelled",
        },
        "safeguards": {
            "cells": ["finance__public_base", "finance__public_lora_2601", "pile__public_base", "pile__public_lora_2601"],
            "endpoint_rules": {
                "exact": "in every safeguard cell, the one-sided 95% paired exact lower bound must be >= -5 percentage points (material exact harm excluded)",
                "token": "in every safeguard cell, the one-sided 95% paired token lower bound must be >= -1 percentage point (material token harm excluded)",
            },
            "interpretation": "A safeguard lower bound that remains below its harm margin leaves deployment unresolved; a point estimate alone cannot clear the gate.",
        },
        "advance_rule": {
            "quality": "primary exact practical lower bound >= +5 percentage points OR primary token practical lower bound >= +1 percentage point; exact and token quality routes each use one-sided alpha 0.025, and the report must identify which route passed",
            "positive_vs_practical": "A lower bound >0 but below the practical margin is positive evidence only and does not establish a practical replacement claim.",
            "cost": "candidate warmed runtime/reference warmed runtime <= 1.25 under the same synchronized boundary on the primary cell; report all paired LoRA/Pile cells and fail deployment if any declared cost gate fails",
            "cost_threshold": 1.25,
            "memory": "peak memory and deployed footprint are reported; no unregistered memory threshold is introduced",
            "inconclusive": "If quality, safeguard, or cost bounds do not clear their predeclared gates, retain the frozen reference and label the practical decision unresolved.",
        },
        "roles": {
            "finance_public_base": "primary deployment decision (P0 exact endpoint)",
            "finance_public_lora": "paired target-shift replication and harm safeguard; not pooled with P0",
            "pile_public_base": "natural-text generalization safeguard",
            "pile_public_lora": "paired natural-text target-shift safeguard",
            "current_residual": "credible alternative, descriptive comparison only",
            "improved_diagonal": "same-bank rescue diagnostic, descriptive only",
        },
        "multiplicity": {
            "primary_family": "one candidate/reference Finance P0 contrast with two predeclared quality routes (exact and token), alpha 0.025 per route; no 64-direction family",
            "secondary": "token endpoint, paired target/domain safeguards, current residual, and improved diagonal are predeclared descriptive/gate reports; no pooled winner",
            "pooling": "do not pool domains, target conditions, methods, or token positions into a single overall score",
        },
        "truth_and_freeze": {
            "before_truth": "freeze source identities, observations, all four method outputs, timing blocks, and validation receipts",
            "after_truth": "truth is used only to score frozen predictions; no refit, routing, model swap, sample expansion, or timing change",
            "source_selection": "not started by this planning artifact",
        },
        "provenance": {
            "trr7_method_freeze": "experiments/TRR-0007/method_freeze.json",
            "trr7_score": "experiments/TRR-0007/scored/result.json",
            "power_analysis": str(power_path),
            "p06_underlying_provenance_opened": False,
        },
    }
    contract["artifact"] = _write_create_only(output, contract)
    return contract


def _power_artifact(*, output: Path, decision_output: Path, finance_n: int = 1024, pile_n: int = 384) -> dict[str, Any]:
    pilot_n = 128
    pilot_gains = 13
    pilot_losses = 6
    pilot_q = (pilot_gains + pilot_losses) / pilot_n
    pilot_effect = (pilot_gains - pilot_losses) / pilot_n
    effects = (0.005, 0.02, pilot_effect, 0.08, 0.10)
    finance_sizes = (512, 1024, 1536, 2048)
    pile_sizes = (256, 384, 512)
    alpha = 0.025
    finance_rows, finance_engine = _paired_power_rows(sample_sizes=finance_sizes, effects=effects, q=pilot_q, alpha=alpha, domain="finance")
    pile_rows, pile_engine = _paired_power_rows(sample_sizes=pile_sizes, effects=effects, q=pilot_q, alpha=alpha, domain="pile")
    payload: dict[str, Any] = {
        "schema": POWER_SCHEMA,
        "task_id": TASK_ID,
        "status": "PROSPECTIVE_EXACT_CP_POWER_CALCULATION_COMPLETE",
        "created_utc": _utc_now(),
        "endpoint": {
            "primary": "finance__public_base exact-record paired candidate-minus-reference",
            "scored_post_bos_tokens_per_record": SCORED_POST_BOS,
            "record_unit": "matched natural source record; public_base and public_lora_2601 are paired target conditions",
            "positive_margin_pp": 0.0,
            "practical_exact_margin_pp": 5.0,
            "practical_token_margin_pp": 1.0,
            "quality_route_alpha": 0.025,
            "final_interval": "exact paired discordance CP gain-minus-loss lower bound for exact records; paired record bootstrap for token accuracy; alpha 0.025 per quality route",
        },
        "pilot": {
            "records": pilot_n,
            "exact_gains": pilot_gains,
            "exact_losses": pilot_losses,
            "net_effect_pp": 100.0 * pilot_effect,
            "total_discordance_pp": 100.0 * pilot_q,
            "source": "TRR-0007 paired_student_vs_reference finance__public_base",
        },
        "model": {
            "distribution": "Y in {-1,0,+1} per matched record; q=P(Y!=0) fixed at pilot 19/128",
            "q": pilot_q,
            "quality_route_alpha": alpha,
            "component_alpha": alpha / 2.0,
            "formula": "p_gain=(q+d)/2, p_loss=(q-d)/2; exact CP lower(p_gain)-upper(p_loss) >= margin",
            "engine": {"finance": finance_engine, "pile": pile_engine},
            "caveat": "Prospective multinomial sensitivity is conditional on pilot discordance q and does not assume future source outcomes; it is not a guarantee and is not used to alter a frozen sample after truth.",
        },
        "candidate_sample_sizes": {
            "finance": list(finance_sizes),
            "pile": list(pile_sizes),
        },
        "selected_sample_size_proposal": {
            "finance_records_per_domain": finance_n,
            "pile_records_per_domain": pile_n,
            "reason": "finance uses a moderate decision panel with ample inventory; pile uses a bounded natural-text safeguard panel and is fixed only after the identity inventory confirms capacity",
            "status": "PROPOSAL_PENDING_INVENTORY_AND_OWNER_FREEZE",
        },
        "sensitivity_effects_pp": [100.0 * effect for effect in effects],
        "rows": finance_rows + pile_rows,
        "interpretation": {
            "near_boundary": "The 5.47-point pilot effect is close to the +5-point practical margin; even Finance n=2048 has low power to prove a lower bound above +5 at that effect. Do not promise threshold confirmation near the boundary.",
            "smaller_effects": "The 0.5-point and 2-point rows make clear that a bounded study is not powered to certify tiny practical changes; an inconclusive result retains the reference.",
            "positive_vs_practical": "A positive lower bound above 0 is reported separately from a practical lower bound at +5 exact points or +1 token point; an exact/token route that clears only 0 is positive evidence, not a practical replacement claim.",
            "bounded_stopping": "No outcome-driven expansion or partial-score stopping is allowed; source availability may reduce a predeclared safeguard size only before selection and must be owner-frozen.",
            "alternative_interpretation": "Report candidate-versus-current-residual explicitly; if the candidate is inferior to current_residual on a declared cell, state that result rather than calling it a global winner.",
        },
    }
    payload["artifact"] = _write_create_only(output, payload)
    contract = _decision_contract(output=decision_output, power_path=output, finance_n=finance_n, pile_n=pile_n)
    payload["decision_contract"] = contract
    return payload

def _validate_trr7_inventory(root: Path) -> dict[str, Any]:
    path = (root / TRR7_ELIGIBILITY).resolve()
    inventory = _load_json(path, description="TRR-0007 eligibility inventory")
    if inventory.get("status") != "ELIGIBILITY_INVENTORY_COMPLETE_NO_SELECTION_NO_TRUTH":
        raise PlanError("TRR-0007 eligibility inventory is not a closed count-only projection")
    if inventory.get("selection_status") != "NOT_STARTED":
        raise PlanError("TRR-0007 inventory records source selection as started")
    contract = inventory.get("source_contract")
    if not isinstance(contract, Mapping) or contract.get("source_ranges_half_open") != SOURCE_RANGES:
        raise PlanError("TRR-0007 established natural source ranges changed")
    for key in ("private_or_truth_payload_read", "truth_created", "truth_opened"):
        if inventory.get(key) is True:
            raise PlanError(f"TRR-0007 inventory records forbidden access: {key}")
    return {
        "file": _file_descriptor(path),
        "status": inventory.get("status"),
        "source_ranges_half_open": SOURCE_RANGES,
        "domains": {
            style: {
                "eligible_unique_before_p06": int(inventory["domains"][style]["eligible_unique"]),
                "range": inventory["domains"][style]["source_range_half_open"],
                "p04_opaque_source_excluded": int(inventory["domains"][style].get("excluded_opaque_source_hash", 0)),
                "p04_opaque_sequence_excluded": int(inventory["domains"][style].get("excluded_opaque_sequence_hash", 0)),
            }
            for style in STYLE_ORDER
        },
        "identity_counts": inventory.get("exclusion_policy", {}).get("identity_counts"),
        "final_bank_ledgers": inventory.get("final_bank_ledgers"),
        "prefix_exclusions": inventory.get("public_fitting_prefix_exclusions"),
    }


def _load_p06_opaque(path: Path = P06_OPAQUE) -> tuple[dict[str, Any], frozenset[str], frozenset[str]]:
    descriptor = _file_descriptor(path, expected_sha256=P06_OPAQUE_SHA256, expected_bytes=P06_OPAQUE_BYTES)
    payload = _load_json(path, description="authorized P06 opaque source reservation")
    if payload.get("schema") != "token-reconstruction.trr-p06-opaque-source-sequence-reservation.v1":
        raise PlanError("P06 opaque reservation schema changed")
    if payload.get("status") != "OPAQUE_HASH_RESERVATION_FOR_FUTURE_EXCLUSION":
        raise PlanError("P06 opaque reservation status is not exclusion-only")
    if payload.get("privacy") != {
        "labels_or_answers_present": False,
        "record_ids_present": False,
        "row_indices_present": False,
        "source_text_present": False,
        "suitable_for_identity_exclusion_only": True,
        "token_ids_present": False,
    }:
        raise PlanError("P06 opaque reservation privacy boundary changed")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or counts.get("public_record_sha256") != 512 or counts.get("final_sequence_sha256") != 512:
        raise PlanError("P06 opaque reservation must contain exactly 512 source and 512 H128 sequence identities")
    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping):
        raise PlanError("P06 opaque reservation hashes are absent")
    source = hashes.get("public_record_sha256")
    sequence = hashes.get("final_sequence_sha256")
    if not isinstance(source, Mapping) or not isinstance(sequence, Mapping):
        raise PlanError("P06 opaque reservation hash summaries are malformed")
    source_values = source.get("values")
    sequence_values = sequence.get("values")
    if not isinstance(source_values, list) or len(source_values) != 512:
        raise PlanError("P06 source hash array length changed")
    if not isinstance(sequence_values, list) or len(sequence_values) != 512:
        raise PlanError("P06 H128 sequence hash array length changed")
    for label, values in (("source", source_values), ("sequence", sequence_values)):
        if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()) for value in values):
            raise PlanError(f"P06 {label} hash array contains a malformed digest")
        if len(set(values)) != 512:
            raise PlanError(f"P06 {label} hash array is not unique")
    summary = {
        "file": descriptor,
        "task_id": payload.get("task_id"),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "records_per_domain": payload.get("source_selection", {}).get("records_per_domain"),
        "records_total": payload.get("source_selection", {}).get("records_total"),
        "selection_seed": payload.get("source_selection", {}).get("selection_seed"),
        "source_hash_count": len(source_values),
        "sequence_hash_count": len(sequence_values),
        "source_ordered_newline_sha256": source.get("ordered_newline_sha256"),
        "source_unique_set_canonical_json_sha256": source.get("unique_set_canonical_json_sha256"),
        "sequence_ordered_newline_sha256": sequence.get("ordered_newline_sha256"),
        "sequence_unique_set_canonical_json_sha256": sequence.get("unique_set_canonical_json_sha256"),
        "privacy": payload.get("privacy"),
        "underlying_provenance_opened": False,
        "underlying_results_opened": False,
        "underlying_holdout_opened": False,
    }
    return summary, frozenset(source_values), frozenset(sequence_values)


def _p06_sequence_digest(token_ids: Sequence[int]) -> str:
    # P06 declares exactly 128 int32 IDs including BOS.  Keep the convention
    # separate from TRR-0007's P04 129-token truncated hash.
    import struct

    values = [int(value) for value in token_ids[:SEQUENCE_TOKENS]]
    if len(values) < SEQUENCE_TOKENS:
        raise PlanError("candidate has fewer than 128 tokens for P06 H128 matching")
    return _sha256_bytes(struct.pack("<" + "i" * len(values), *values))


def _identity_commitment(record: Any) -> dict[str, str]:
    return {
        "record_id_sha256": _sha256_bytes(str(record.record_id).encode("utf-8")),
        "public_record_sha256": str(record.public_record_sha256),
        "h128_sequence_sha256": _p06_sequence_digest(record.token_ids),
        "final_sequence_sha256": str(record.final_sequence_sha256),
    }


@dataclass
class _Counts:
    style: str
    start: int
    stop: int
    scanned: int = 0
    valid: int = 0
    eligible: int = 0
    invalid: int = 0
    excluded_id: int = 0
    excluded_index: int = 0
    excluded_hash: int = 0
    excluded_p04_source: int = 0
    excluded_p04_sequence: int = 0
    excluded_p06_source: int = 0
    excluded_p06_sequence: int = 0
    duplicate_source: int = 0
    duplicate_sequence: int = 0
    p06_source_or_sequence_union: int = 0
    valid_commitments: list[dict[str, str]] | None = None
    eligible_commitments: list[dict[str, str]] | None = None

    def __post_init__(self) -> None:
        self.valid_commitments = []
        self.eligible_commitments = []


def _commitment_digest(values: Sequence[Mapping[str, str]]) -> str:
    canonical = "\n".join(_canonical_json(dict(value)) for value in sorted((dict(v) for v in values), key=_canonical_json))
    return _sha256_bytes((canonical + "\n").encode("utf-8"))


def _inventory(args: argparse.Namespace) -> dict[str, Any]:
    """Run the count-only renderer audit after the owner permits the scan."""

    root = Path(args.repository_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise PlanError(f"inventory output already exists: {output}")
    # Imports are delayed so `power` remains a light CPU operation and never
    # initializes the model/tokenizer stack.
    from scripts import trr0005_produce_confirmation as trusted
    from scripts import trr0006_build_eligibility as eligibility
    from scripts import trr0007_eval_select as trr7_selector
    from token_reconstruction.trr0005_public_corpus import deterministic_row_order, source_record_id

    trr7 = _validate_trr7_inventory(root)
    p06_summary, p06_source, p06_sequence = _load_p06_opaque()
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    tokenizer = trusted._load_tokenizer(tokenizer_path)
    pile_paths = tuple(Path(value).expanduser().resolve() for value in args.pile_arrow)
    finance_paths = tuple(Path(value).expanduser().resolve() for value in args.finance_arrow)
    datasets = {
        "pile": trusted._load_arrow_dataset(pile_paths),
        "finance": trusted._load_arrow_dataset(finance_paths),
    }
    # Reuse the exact established TRR-0007 exclusion ledger and explicitly
    # bind the known TRR-0007 final bank/prefix ledgers.  No private P06 source
    # or result path is accepted by this command.
    # Start with the complete explicit TRR-0007 selector ledger (fit,
    # validation, prior opened public panels, and TRR-0006 additions), then
    # bind the reviewed TRR-0007 exclusion/range ledgers.  This avoids relying
    # on the aggregate source_exclusions receipt alone, which has counts but
    # intentionally does not embed every prior identity value.
    known = list(trr7_selector._known_exclusion_paths(root))
    known.extend(
        [
            root / TRR7_EXCLUSIONS,
            root / TRR7_SELECTION,
            root / TRR7_FINAL_BANK,
            root / TRR7_PARENT_ROWS,
            root / TRR7_CORPUS_PLAN,
            root / TRR7_PREFIX_EXCLUSIONS,
        ]
    )
    known.extend(Path(value).expanduser().resolve() for value in args.exclude_source)
    exclusions = trusted._collect_exclusions(known)
    # P04 is the approved aggregate exchange already applied by TRR-0007.
    # Re-apply its opaque identity sets explicitly for this audit; no P04
    # source, provenance, target update, result, or holdout payload is opened.
    p04_path = P04_OPAQUE.expanduser().resolve()
    if (
        p04_path.is_symlink()
        or not p04_path.is_file()
        or _sha256_file(p04_path) != P04_OPAQUE_SHA256
    ):
        raise PlanError("approved P04 opaque exchange is unavailable or has changed")
    p04 = eligibility._load_p04_opaque_exclusions(p04_path)
    seen_public_hashes: set[str] = set()
    seen_final_sequences: set[str] = set()
    domains: dict[str, _Counts] = {}
    for style in STYLE_ORDER:
        start, stop = SOURCE_RANGES[style]
        dataset = datasets[style]
        if len(dataset) < stop:
            raise PlanError(f"{style} cache has {len(dataset)} rows; need {stop}")
        counts = _Counts(style=style, start=start, stop=stop)
        spec = eligibility.SOURCE_PARTITIONS[style]
        order = deterministic_row_order(range(start, stop), dataset_key=f"{style}-future-holdout", seed=SELECTION_SEED)
        for index in order:
            counts.scanned += 1
            expected_id = source_record_id(str(spec["dataset_id"]), str(spec["split"]), str(spec["revision"]), index)
            if expected_id in exclusions.ids[style]:
                counts.excluded_id += 1
                continue
            if index in exclusions.indices[style]:
                counts.excluded_index += 1
                continue
            row = trusted._read_reserved_row(dataset, style=style, row_index=index)
            try:
                candidate = trusted._render_row(style, row, index, tokenizer)
            except trusted.ProducerError:
                counts.invalid += 1
                continue
            counts.valid += 1
            commitment = _identity_commitment(candidate)
            counts.valid_commitments.append(commitment)
            if candidate.public_record_sha256 in exclusions.hashes[style]:
                counts.excluded_hash += 1
                continue
            # Existing TRR-0007/P04 sequence identities use the candidate's
            # final_sequence_sha256 plus the trusted opaque matcher.  We call
            # only the public metadata helper; no row payload is emitted.
            if candidate.final_sequence_sha256 in exclusions.hashes[style]:
                counts.excluded_hash += 1
                continue
            if candidate.public_record_sha256 in p04.source_hashes:
                counts.excluded_p04_source += 1
                continue
            # P04 uses the established 129-ID (BOS + 128 post-BOS) digest;
            # P06's separately declared H128 digest is handled below.
            if len(candidate.token_ids) >= SEQUENCE_TOKENS + 1:
                p04_sequence = trusted._sequence_digest(
                    candidate.token_ids[: SEQUENCE_TOKENS + 1]
                )
                if p04_sequence in p04.sequence_hashes_129:
                    counts.excluded_p04_sequence += 1
                    continue
            if candidate.public_record_sha256 in p06_source:
                counts.excluded_p06_source += 1
                counts.p06_source_or_sequence_union += 1
                continue
            h128 = commitment["h128_sequence_sha256"]
            if h128 in p06_sequence:
                counts.excluded_p06_sequence += 1
                counts.p06_source_or_sequence_union += 1
                continue
            if candidate.public_record_sha256 in seen_public_hashes:
                counts.duplicate_source += 1
                continue
            if candidate.final_sequence_sha256 in seen_final_sequences:
                counts.duplicate_sequence += 1
                continue
            seen_public_hashes.add(candidate.public_record_sha256)
            seen_final_sequences.add(candidate.final_sequence_sha256)
            counts.eligible += 1
            counts.eligible_commitments.append(commitment)
        domains[style] = counts

    requested_by_domain = {
        "finance": int(args.finance_records),
        "pile": int(args.pile_records),
    }
    for style, counts in domains.items():
        requested = requested_by_domain[style]
        if counts.eligible < requested:
            raise PlanError(f"{style} has only {counts.eligible} eligible records; requested {requested}")
    result: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "task_id": TASK_ID,
        "status": "IDENTITY_INVENTORY_COMPLETE_NO_SELECTION_NO_TRUTH",
        "created_utc": _utc_now(),
        "sample_size_status": "PROPOSED_COUNT_CHECKED_NOT_SELECTED",
        "requested_per_domain": requested_by_domain,
        "source_contract": {
            "source_ranges_half_open": SOURCE_RANGES,
            "selection_seed": SELECTION_SEED,
            "sequence_tokens_including_bos": SEQUENCE_TOKENS,
            "scoring_post_bos_tokens": SCORED_POST_BOS,
            "capture_batch_records": CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "natural_distribution_preserved": True,
            "pairing": "Future selection, if separately authorized after design freeze, pairs each source across public_base and public_lora_2601; this inventory emits no rows.",
        },
        "trr0007_inventory": trr7,
        "p06_opaque_reservation": p06_summary,
        "domains": {
            style: {
                "source_range_half_open": [counts.start, counts.stop],
                "scanned_rows": counts.scanned,
                "valid_rows": counts.valid,
                "invalid_rows": counts.invalid,
                "excluded_id": counts.excluded_id,
                "excluded_index": counts.excluded_index,
                "excluded_hash": counts.excluded_hash,
                "excluded_p04_source_hash": counts.excluded_p04_source,
                "excluded_p04_h129_sequence_hash": counts.excluded_p04_sequence,
                "excluded_p06_source_hash": counts.excluded_p06_source,
                "excluded_p06_h128_sequence_hash": counts.excluded_p06_sequence,
                "excluded_p06_union": counts.p06_source_or_sequence_union,
                "duplicate_rendered_source": counts.duplicate_source,
                "duplicate_final_sequence": counts.duplicate_sequence,
                "eligible_unique": counts.eligible,
                "capacity_for_requested_per_domain": {
                    "requested": requested_by_domain[style],
                    "sufficient": counts.eligible >= requested_by_domain[style],
                    "surplus_or_shortfall": counts.eligible - requested_by_domain[style],
                },
                "valid_identity_commitment_sha256": _commitment_digest(counts.valid_commitments),
                "eligible_identity_commitment_sha256": _commitment_digest(counts.eligible_commitments),
            }
            for style, counts in domains.items()
        },
        "exclusion_policy": {
            "identity_only": True,
            "trr0007_known_ledgers_bound": True,
            "p04_opaque_export": str(P04_OPAQUE),
            "p04_source_and_h129_hashes_only": True,
            "p06_opaque_export": str(P06_OPAQUE),
            "p06_source_and_h128_hashes_only": True,
            "source_text_written": False,
            "token_ids_written": False,
            "selection_performed": False,
            "truth_created_or_opened": False,
            "model_loaded": False,
        },
        "totals": {
            "unique_source_records_scanned": sum(c.scanned for c in domains.values()),
            "eligible_unique_across_domains": sum(c.eligible for c in domains.values()),
            "p04_source_excluded": sum(c.excluded_p04_source for c in domains.values()),
            "p04_h129_sequence_excluded": sum(c.excluded_p04_sequence for c in domains.values()),
            "p06_source_excluded": sum(c.excluded_p06_source for c in domains.values()),
            "p06_h128_sequence_excluded": sum(c.excluded_p06_sequence for c in domains.values()),
            "p06_union_excluded": sum(c.p06_source_or_sequence_union for c in domains.values()),
        },
        "public_inputs": {
            "pile_arrow": [_file_descriptor(path) for path in pile_paths],
            "finance_arrow": [_file_descriptor(path) for path in finance_paths],
            "tokenizer": {"path": str(tokenizer_path)},
        },
        "execution": {
            "command": list(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "network_used": False,
            "target_loaded": False,
            "truth_created_or_opened": False,
            "selection_performed": False,
        },
        "limitations": [
            "Counts are an identity-only capacity audit; no source row is selected, frozen, observed, or truth-bound.",
            "P06 source and H128 sequence hash sets are opaque and are not joined to one another; both sets are conservatively applied as a union exclusion.",
            "P04 target-fit per-record disjointness remains unavailable under the approved aggregate exchange and is not inferred.",
        ],
    }
    result["artifact"] = _write_create_only(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    power = sub.add_parser("power", help="write prospective paired-discordance power analysis")
    power.add_argument("--output", type=Path, required=True)
    power.add_argument("--decision-output", type=Path, required=True)
    power.add_argument("--finance-records", type=int, default=1024)
    power.add_argument("--pile-records", type=int, default=384)
    inventory = sub.add_parser("inventory", help="scan natural ranges for identity-only capacity")
    inventory.add_argument("--repository-root", type=Path, default=Path("."))
    inventory.add_argument("--tokenizer", type=Path, required=True)
    inventory.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    inventory.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    inventory.add_argument("--exclude-source", type=Path, nargs="*", default=[])
    inventory.add_argument("--finance-records", type=int, default=1024)
    inventory.add_argument("--pile-records", type=int, default=384)
    inventory.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "power":
        if args.finance_records <= 0 or args.pile_records <= 0:
            raise PlanError("records per domain must be positive")
        result = _power_artifact(
            output=args.output,
            decision_output=args.decision_output,
            finance_n=args.finance_records,
            pile_n=args.pile_records,
        )
    elif args.command == "inventory":
        if args.finance_records <= 0 or args.pile_records <= 0:
            raise PlanError("requested records per domain must be positive")
        result = _inventory(args)
    else:  # pragma: no cover
        raise PlanError(f"unknown command: {args.command}")
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "artifact"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PlanError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0008 planning error: {exc}")
