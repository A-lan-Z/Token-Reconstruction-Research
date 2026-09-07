#!/usr/bin/env python3
"""Prospective TRR-0009 planning and identity-only inventory.

This helper writes the light exact paired-discordance sensitivity table and a
metadata-only capacity projection derived from the closed TRR-0008 inventory
and selection ledgers.  The deferred ``inventory-final`` command is a
count-only public-range scan: it transiently renders rows to apply the
validated identity exclusions, writes aggregate counts/commitments only, and
never selects a row, loads a model, opens truth, or writes source text/token
IDs.  The real selector is ``scripts/trr0009_select_public.py`` and remains
gated by the owner-frozen contract, the finalized inventory, and the
preselection TRR-0009 selected-method freeze.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

TASK_ID = "TRR-0009"
POWER_SCHEMA = "token-reconstruction.trr0009-power-analysis.v1"
INVENTORY_SCHEMA = "token-reconstruction.trr0009-source-inventory.v1"
STYLE_ORDER = ("pile", "finance")
SOURCE_RANGES = {"finance": [12000, 20000], "pile": [7000, 10000]}
SELECTION_SEED = 5005
SEQUENCE_TOKENS = 128
SCORED_POST_BOS = 127
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
EXPECTED_RECORDS_BY_DOMAIN = {"finance": 256, "pile": 128}
VOCAB_SIZE = 128256

TRR7_ELIGIBILITY = Path("experiments/TRR-0007/selection/eligibility_inventory.json")
TRR7_EXCLUSIONS = Path("experiments/TRR-0007/selection/source_exclusions.json")
TRR7_SELECTION = Path("experiments/TRR-0007/selection/source_selection.json")
TRR7_FINAL_BANK = Path("experiments/TRR-0007/support/broader_bank_v5/public_parent_exclusion_manifest.json")
TRR7_PARENT_ROWS = Path("experiments/TRR-0007/support/broader_bank_v5/selected_parent_rows.json")
TRR7_CORPUS_PLAN = Path("experiments/TRR-0007/support/broader_bank_v5/corpus_plan.json")
TRR7_PREFIX_EXCLUSIONS = Path("experiments/TRR-0007/support/public_fit_prefix_exclusions_v3.json")
TRR8_INVENTORY = Path("experiments/TRR-0008/planning/identity_inventory_1thread.json")
TRR8_EXCLUSIONS = Path("experiments/TRR-0008/selection/source_exclusions.json")
TRR8_SELECTION = Path("experiments/TRR-0008/selection/source_selection.json")
TRR8_RESERVATION = Path("experiments/TRR-0008/selection/opaque_source_sequence_reservation.json")
P06_OPAQUE_ORIGINAL = Path("/tmp/trr-p06/experiments/TRR-P06/setup/p06_opaque_source_sequence_reservation.json")
# The approved P06 copy is inherited read-only from TRR-0008.  TRR-0009 does
# not copy or inspect any other P06 artifact.
P06_OPAQUE = _REPOSITORY_ROOT / "experiments/TRR-0008/planning/approved_opaque/p06_opaque_source_sequence_reservation.json"
P06_OPAQUE_SHA256 = "231f924c774ace135e5870232b3aa276b9d565a927fc18ec5e64667688843f70"
P06_OPAQUE_BYTES = 80243
P04_OPAQUE = Path("/tmp/trr-p04/experiments/TRR-P04/coordination/reservation_hashes.json")
P04_OPAQUE_SHA256 = "98f8dfcab0977b4bcafa47d97a86a410ab37359b897b9b553746afa7df5c7904"
P07_REPLY = Path("experiments/TRR-0009/coordination/p07_resource_exclusion_reply.json")
FREQUENCY_REFERENCE = Path("experiments/TRR-0005/frequency_references_v1.json")
IMPROVED_CORPUS_PLAN = Path("experiments/TRR-0007/support/broader_bank_v5/corpus_plan.json")
CURRENT_STATE = Path("experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_residual_mlp512/selected.safetensors")
CURRENT_STATE_SHA256 = "2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8"
CURRENT_STATE_BYTES = 29390628
CURRENT_METHOD_ID = "trr0007_current_enriched__residual_mlp512"
# The improved public bank remains the fitting/frequency bank.  The common
# prefit state is the published current-enriched residual selected at step
# 2100 because the improved-bank residual had already reached its fitting-bank
# ceiling; this is one prefit choice, not a sweep.


class PlanError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_descriptor(path: Path, *, expected_sha256: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PlanError(f"required metadata file is unavailable: {path}")
    actual_bytes = int(path.stat().st_size)
    actual_sha256 = _sha256_file(path)
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise PlanError(f"metadata byte count changed for {path}: expected {expected_bytes}, got {actual_bytes}")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise PlanError(f"metadata hash changed for {path}: expected {expected_sha256}, got {actual_sha256}")
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_p06_opaque(path: Path = P06_OPAQUE) -> tuple[dict[str, Any], frozenset[str], frozenset[str]]:
    """Load only the approved opaque P06 hash export."""

    descriptor = _file_descriptor(path, expected_sha256=P06_OPAQUE_SHA256, expected_bytes=P06_OPAQUE_BYTES)
    payload = _load_json(path, description="approved P06 opaque reservation")
    if payload.get("schema") != "token-reconstruction.trr-p06-opaque-source-sequence-reservation.v1":
        raise PlanError("P06 opaque reservation schema changed")
    if payload.get("status") != "OPAQUE_HASH_RESERVATION_FOR_FUTURE_EXCLUSION":
        raise PlanError("P06 opaque reservation is not exclusion-only")
    privacy = payload.get("privacy")
    expected_privacy = {
        "labels_or_answers_present": False,
        "record_ids_present": False,
        "row_indices_present": False,
        "source_text_present": False,
        "suitable_for_identity_exclusion_only": True,
        "token_ids_present": False,
    }
    if privacy != expected_privacy:
        raise PlanError("P06 opaque privacy boundary changed")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or counts.get("public_record_sha256") != 512 or counts.get("final_sequence_sha256") != 512:
        raise PlanError("P06 opaque export must contain 512 source and 512 H128 hashes")
    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping):
        raise PlanError("P06 opaque hash summaries are absent")
    source_summary = hashes.get("public_record_sha256")
    sequence_summary = hashes.get("final_sequence_sha256")
    if not isinstance(source_summary, Mapping) or not isinstance(sequence_summary, Mapping):
        raise PlanError("P06 opaque hash summaries are malformed")
    source_values = source_summary.get("values")
    sequence_values = sequence_summary.get("values")
    if not isinstance(source_values, list) or len(source_values) != 512 or len(set(source_values)) != 512:
        raise PlanError("P06 source hash values are malformed")
    if not isinstance(sequence_values, list) or len(sequence_values) != 512 or len(set(sequence_values)) != 512:
        raise PlanError("P06 sequence hash values are malformed")
    for label, values in (("source", source_values), ("sequence", sequence_values)):
        if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in values):
            raise PlanError(f"P06 {label} hash values are malformed")
    summary = {
        "file": descriptor,
        "authorized_original_path": str(P06_OPAQUE_ORIGINAL),
        "task_id": payload.get("task_id"),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "source_hash_count": len(source_values),
        "sequence_hash_count": len(sequence_values),
        "source_ordered_newline_sha256": source_summary.get("ordered_newline_sha256"),
        "source_unique_set_canonical_json_sha256": source_summary.get("unique_set_canonical_json_sha256"),
        "sequence_ordered_newline_sha256": sequence_summary.get("ordered_newline_sha256"),
        "sequence_unique_set_canonical_json_sha256": sequence_summary.get("unique_set_canonical_json_sha256"),
        "privacy": privacy,
        "underlying_provenance_opened": False,
        "underlying_results_opened": False,
        "underlying_holdout_opened": False,
    }
    return summary, frozenset(source_values), frozenset(sequence_values)


def _cp_arrays(n: int, *, alpha: float):
    try:
        import numpy as np
        from scipy.stats import beta
    except ImportError as exc:
        raise PlanError("SciPy is required for exact prospective CP power") from exc
    component_alpha = alpha / 2.0
    counts = np.arange(n + 1, dtype=int)
    lower = np.zeros(n + 1, dtype=float)
    upper = np.ones(n + 1, dtype=float)
    if n:
        lower[1:] = beta.ppf(component_alpha, counts[1:], n - counts[1:] + 1)
        # CP upper tail is beta.ppf(1-alpha_component), not beta.ppf(alpha).
        upper[:-1] = beta.ppf(1.0 - component_alpha, counts[:-1] + 1, n - counts[:-1])
    return lower, upper, "scipy.stats.beta.ppf_clopper_pearson"


def exact_cp_power(*, n: int, true_effect: float, margin: float, discordance_rate: float, alpha: float = 0.025) -> tuple[float, str]:
    """Prospective exact power for CP lower(gains)-upper(losses) >= margin."""

    lower, upper, engine = _cp_arrays(n, alpha=alpha)
    import numpy as np
    from scipy.stats import binom
    p_gain = (discordance_rate + true_effect) / 2.0
    p_loss = (discordance_rate - true_effect) / 2.0
    if not 0.0 <= p_loss <= p_gain <= 1.0:
        raise ValueError("invalid paired multinomial planning probabilities")
    total = 0.0
    g_values = np.arange(n + 1, dtype=int)
    for gain, probability in zip(g_values.tolist(), binom.pmf(g_values, n, p_gain).tolist()):
        if probability == 0.0:
            continue
        maximum_loss = int(np.searchsorted(upper, lower[gain] - margin, side="right") - 1)
        remaining = n - gain
        maximum_loss = min(maximum_loss, remaining)
        if maximum_loss < 0:
            continue
        conditional = 1.0 if remaining == 0 else float(
            binom.cdf(maximum_loss, remaining, p_loss / (1.0 - p_gain))
        )
        total += probability * conditional
    return float(total), engine


def exact_cp_joint_power(*, n: int, true_effect: float, point_floor: float, discordance_rate: float, alpha: float = 0.025) -> tuple[float, str]:
    """Power for a point-floor plus positive conservative CP lower bound."""

    lower, upper, engine = _cp_arrays(n, alpha=alpha)
    import numpy as np
    from scipy.stats import binom
    p_gain = (discordance_rate + true_effect) / 2.0
    p_loss = (discordance_rate - true_effect) / 2.0
    if not 0.0 <= p_loss <= p_gain <= 1.0:
        raise ValueError("invalid paired multinomial planning probabilities")
    total = 0.0
    minimum_net = math.ceil(n * point_floor - 1e-12)
    g_values = np.arange(n + 1, dtype=int)
    for gain, probability in zip(g_values.tolist(), binom.pmf(g_values, n, p_gain).tolist()):
        if probability == 0.0:
            continue
        # CP lower(gain)-upper(loss) > 0 and (gain-loss)/n >= point_floor.
        cp_max_loss = int(np.searchsorted(upper, lower[gain], side="left") - 1)
        point_max_loss = gain - minimum_net
        maximum_loss = min(cp_max_loss, point_max_loss, n - gain)
        if maximum_loss < 0:
            continue
        remaining = n - gain
        conditional = 1.0 if remaining == 0 else float(
            binom.cdf(maximum_loss, remaining, p_loss / (1.0 - p_gain))
        )
        total += probability * conditional
    return float(total), engine


def _power_artifact() -> dict[str, Any]:
    q = 19.0 / 128.0
    effects = (0.005, 0.01, 0.02, 0.0322265625, 0.05, 0.0546875, 0.08, 0.10)
    alpha = 0.025
    exact_point_floor = 0.02
    rows: list[dict[str, Any]] = []
    engine = ""
    for domain, sizes in (("finance", (128, 256, 384, 512)), ("pile", (128, 256))):
        for n in sizes:
            for effect in effects:
                positive, used = exact_cp_power(n=n, true_effect=effect, margin=0.0, discordance_rate=q, alpha=alpha)
                useful, used2 = exact_cp_joint_power(n=n, true_effect=effect, point_floor=exact_point_floor, discordance_rate=q, alpha=alpha)
                engine = used if used == used2 else f"{used}+{used2}"
                rows.append({
                    "domain": domain,
                    "records": n,
                    "true_effect_pp": round(effect * 100.0, 8),
                    "positive_margin_pp": 0.0,
                    "useful_exact_point_floor_pp": exact_point_floor * 100.0,
                    "positive_cp_power": positive,
                    "useful_exact_power": useful,
                    "useful_power_definition": "P(point estimate >= +2pp AND CP lower(gains)-upper(losses) > 0); exact CP tails use component alpha 0.0125",
                })
    return {
        "schema": POWER_SCHEMA,
        "task_id": TASK_ID,
        "status": "PROSPECTIVE_EXACT_CP_POWER_CALCULATION_COMPLETE",
        "created_utc": _utc_now(),
        "endpoint": {
            "primary": "Finance public_base candidate minus continued fixed-readout control",
            "record_unit": "matched natural source record; target conditions share source records",
            "scored_post_bos_tokens_per_record": SCORED_POST_BOS,
            "positive_rule": "one-sided 97.5% CP gain-minus-loss lower bound > 0",
            "useful_exact_rule": "point difference >= +2 percentage points AND one-sided 97.5% CP gain-minus-loss lower bound > 0; component tails alpha 0.0125",
            "useful_token_rule": "point difference >= +0.25 percentage points AND one-sided 97.5% paired source-record bootstrap lower bound > 0; power is not claimed before token discordance is observed",
            "route_alpha": alpha,
        },
        "pilot_model": {
            "discordance_rate_q": q,
            "source": "TRR-0007 Finance public_base paired candidate-versus-retained-reference pilot (13 gains, 6 losses, n=128); used only as a sensitivity model",
            "formula": "p_gain=(q+d)/2; p_loss=(q-d)/2; G and L follow the paired multinomial model",
            "caveat": "Candidate-versus-fixed discordance may differ. These rows are conditional planning sensitivities, not guarantees and cannot trigger sample expansion or a claim after truth.",
        },
        "sample_size_options": {
            "finance_records": [128, 256, 384, 512],
            "pile_records": [128, 256],
            "proposal": {"finance": 256, "pile": 128, "status": "PROPOSAL_PENDING_OWNER_FREEZE"},
        },
        "rows": rows,
        "engine": engine,
        "interpretation": {
            "positive_vs_useful": "Positive evidence above zero is reported separately from the +2-point exact and +0.25-point token point floors; neither floor is a replacement-promotion claim.",
            "near_boundary": "At this bounded sample, exact lower-bound confirmation near +2 points has low conditional power under the pilot q; an inconclusive result retains the existing reference and does not imply adaptation failure.",
            "rare_ids": "Rare/absent ID strata are not powered by this aggregate source-record table; their gate remains UNKNOWN when the fresh panel has fewer than 32 source records in a stratum.",
            "multiplicity": "The exact and token routes are the two predeclared primary routes; domains, target conditions, and rare strata remain separate safeguard reports and are never pooled.",
        },
    }


def _descriptor_if_exists(path: Path) -> dict[str, Any]:
    try:
        return _file_descriptor(path)
    except PlanError:
        return {"path": str(path.expanduser().resolve()), "available": False}


def _build_inventory() -> dict[str, Any]:
    prior = _load_json(TRR8_INVENTORY, description="TRR-0008 identity inventory")
    selection = _load_json(TRR8_SELECTION, description="TRR-0008 source selection")
    exclusions = _load_json(TRR8_EXCLUSIONS, description="TRR-0008 source exclusions")
    if prior.get("status") != "IDENTITY_INVENTORY_COMPLETE_NO_SELECTION_NO_TRUTH":
        raise PlanError("TRR-0008 inventory is not a closed identity-only projection")
    if selection.get("status") != "FROZEN_TRR0008_SOURCE_SELECTION_NO_TRUTH":
        raise PlanError("TRR-0008 selection is not a no-truth ledger")
    if selection.get("truth_opened") is not False or selection.get("truth_created") is not False:
        raise PlanError("TRR-0008 selection records forbidden truth access")
    selected_counts = {style: len(selection.get("selection_rule", {}).get("records", {}).get(style, [])) for style in STYLE_ORDER}
    domains: dict[str, Any] = {}
    for style in STYLE_ORDER:
        base = prior["domains"][style]
        remaining = int(base["eligible_unique"]) - selected_counts[style]
        domains[style] = {
            "style": style,
            "source_range_half_open": list(SOURCE_RANGES[style]),
            "prior_trr0008_eligible_unique": int(base["eligible_unique"]),
            "excluded_trr0008_selected_records": selected_counts[style],
            "projected_unique_after_known_ledgers_and_trr0008": remaining,
            "requested_for_trr0009": EXPECTED_RECORDS_BY_DOMAIN[style],
            "projected_surplus_or_shortfall": remaining - EXPECTED_RECORDS_BY_DOMAIN[style],
            "capacity_sufficient_projection": remaining >= EXPECTED_RECORDS_BY_DOMAIN[style],
            "derivation": "Projection only: subtracts frozen TRR-0008 selected record count from the prior eligible count; H128 duplicate effects under the union of all TRR-0009 exclusions have not been rescanned.",
        }
    p06_summary, _, _ = _load_p06_opaque()
    compatibility = selection.get("p06_hash_compatibility", {})
    return {
        "schema": INVENTORY_SCHEMA,
        "task_id": TASK_ID,
        "status": "IDENTITY_CAPACITY_PROJECTION_PENDING_FINAL_COUNT_ONLY_SCAN",
        "created_utc": _utc_now(),
        "sample_size_status": "PROPOSED_COUNT_CHECKED_NOT_SELECTED",
        "requested_per_domain": dict(EXPECTED_RECORDS_BY_DOMAIN),
        "domains": domains,
        "source_contract": {
            "source_ranges_half_open": dict(SOURCE_RANGES),
            "selection_seed": SELECTION_SEED,
            "sequence_tokens_including_bos": SEQUENCE_TOKENS,
            "scoring_post_bos_tokens": SCORED_POST_BOS,
            "capture_batch_records": CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "natural_distribution_preserved": True,
            "pairing": "Future owner-authorized selection pairs each natural source across public_base and public_lora_2601; this artifact contains no rows.",
        },
        "prior_fitting_bank": {
            "starting_method": CURRENT_METHOD_ID,
            "state": {
                "path": str(CURRENT_STATE),
                "bytes": CURRENT_STATE_BYTES,
                "sha256": CURRENT_STATE_SHA256,
                "selected_step": 2100,
            },
            "starting_state_role": "Common current_enriched state; improved_public_bank remains the fitting and frequency reference.",
            "starting_state_rationale": "The improved-bank residual reached 100% on its 124371-position fitting bank; use the published competent current-bank residual as the one prefit choice, not a sweep.",
            "schedule": {"seed": 4005, "steps": 3000, "batch_records": 8, "post_bos_positions": 124371, "schedule_sha256": "5a2daa0087b1877bb5f9be4bd59ef201a4fa6478fcd5b16a1b88808963eab472"},
            "frequency_artifact": _descriptor_if_exists(IMPROVED_CORPUS_PLAN),
            "frequency_strata": {
                "vocab_size": VOCAB_SIZE,
                "distinct_supported_ids": 17126,
                "unseen_ids": 111130,
                "0": {"distinct_ids": 111130, "token_occurrences": 0},
                "1-4": {"distinct_ids": 14257, "token_occurrences": 22783},
                "5-9": {"distinct_ids": 1587, "token_occurrences": 10165},
                "10-49": {"distinct_ids": 1061, "token_occurrences": 19463},
                "50+": {"distinct_ids": 221, "token_occurrences": 71960},
                "post_bos_token_occurrences": 124371,
                "source": "published TRR-0007 broader_bank_v5 corpus_plan coverage_mix_v1 exact post-BOS frequency buckets",
            },
            "legacy_frequency_reference": _descriptor_if_exists(FREQUENCY_REFERENCE),
        },
        "known_exclusion_chain": {
            "trr0007_ledgers": [_descriptor_if_exists(_REPOSITORY_ROOT / p) for p in (TRR7_ELIGIBILITY, TRR7_EXCLUSIONS, TRR7_SELECTION, TRR7_FINAL_BANK, TRR7_PARENT_ROWS, TRR7_CORPUS_PLAN, TRR7_PREFIX_EXCLUSIONS)],
            "trr0008_inventory": _descriptor_if_exists(TRR8_INVENTORY),
            "trr0008_exclusions": _descriptor_if_exists(TRR8_EXCLUSIONS),
            "trr0008_selection": _descriptor_if_exists(TRR8_SELECTION),
            "trr0008_reservation": _descriptor_if_exists(TRR8_RESERVATION),
            "p07_resource_reply": _descriptor_if_exists(_REPOSITORY_ROOT / P07_REPLY),
            "interpretation": "TRR7 ledgers cover prior fit/dev/opened public identities; TRR8 selection adds 1,024 Finance and 384 Pile source identities; P07 reports no new identities and reuses already-opened panels.",
        },
        "p06_opaque_reservation": p06_summary,
        "p06_hash_compatibility": dict(compatibility),
        "public_inputs": dict(prior.get("public_inputs", {})),
        "trr0007_inventory": dict(prior.get("trr0007_inventory", {})),
        "trr0008_bindings": {
            "inventory_sha256": _file_descriptor(TRR8_INVENTORY)["sha256"],
            "exclusions_sha256": _file_descriptor(TRR8_EXCLUSIONS)["sha256"],
            "selection_sha256": _file_descriptor(TRR8_SELECTION)["sha256"],
            "selected_counts": selected_counts,
        },
        "execution": {
            "new_source_scan": False,
            "projection_only": True,
            "final_count_only_scan_required": True,
            "selection_performed": False,
            "model_loaded": False,
            "source_text_written": False,
            "token_ids_written": False,
            "truth_created_or_opened": False,
            "private_or_holdout_opened": False,
            "network_used": False,
        },
        "exclusion_policy": {
            "identity_only": True,
            "all_known_fit_validation_opened_eval_ledgers_applied": True,
            "trr8_panel_excluded": True,
            "p04_opaque_identity_only": True,
            "p06_opaque_identity_only": True,
            "p07_new_reservation_required": False,
            "p08_opaque_reservation_status": "PENDING_ROOT_COORDINATION_NOT_READ",
            "p03_holdout_opened": False,
            "p07_workspace_or_results_opened": False,
            "source_text_or_target_labels_written": False,
        },
        "limitations": [
            "This is a metadata-derived capacity projection, not a final TRR-0009 count-only scan. A fresh identity-only scan must reapply the latest approved opaque exclusions and H128 duplicate checks before source selection.",
            "The P06 source and H128 sequence sets remain opaque and are used only as identity exclusions; no provenance, holdout, or result payload was opened.",
            "Rare/absent token strata are supported only where the future panel contains enough source records; a sparse stratum is UNKNOWN rather than a zero-harm pass.",
            "Capacity does not imply fresh source selection or truth access; both remain owner-authorized post-freeze actions.",
        ],
    }



def _load_generic_opaque_reservation(
    path: Path, *, label: str
) -> tuple[dict[str, Any], frozenset[str], frozenset[str]]:
    """Load an approved hash-only reservation without opening provenance.

    P08 may provide the original P06-shaped reservation or the producer's
    sanitized hash-exchange schema.  Both branches accept only source/H128
    hash values and retain no labels, row indices, source text, or token IDs.
    """

    descriptor = _file_descriptor(path)
    payload = _load_json(path, description=f"{label} opaque reservation")
    schema = payload.get("schema")
    sanitized_p08 = schema == "token-reconstruction.trr-p08-sanitized-opaque-hashes.v1"
    if sanitized_p08:
        if label != "p08" or payload.get("task_id") != "TRR-P08":
            raise PlanError(f"{label} sanitized opaque reservation task identity changed")
        if payload.get("status") != "READY_FOR_HASH_ONLY_EXCHANGE":
            raise PlanError(f"{label} sanitized opaque reservation status is not exclusion-only")
        expected_top_keys = {
            "created_utc", "hash_conventions", "hashes", "privacy_boundary",
            "purpose", "recipe", "schema", "status", "task_id",
        }
        if set(payload) != expected_top_keys:
            raise PlanError(f"{label} sanitized opaque reservation schema changed")
        expected_recipe = {
            "canonical_order": "lexicographic order within each hash field; no domain, row, record, source, selection, or target metadata is exported",
            "hash_fields": ["public_record_sha256", "final_sequence_sha256"],
            "name": "H128 hash-only source reservation",
            "sequence_rule": "canonical 128-token including-BOS final-sequence SHA-256 fingerprint",
        }
        if payload.get("purpose") != "hash-only identity and sequence exclusion exchange" or payload.get("recipe") != expected_recipe:
            raise PlanError(f"{label} sanitized opaque reservation recipe changed")
        privacy = {
            "contains_domain_or_style_labels": False,
            "contains_model_weights": False,
            "contains_record_ids": False,
            "contains_source_indices": False,
            "contains_source_text": False,
            "contains_target_labels": False,
            "contains_token_ids": False,
            "contains_truth": False,
            "hash_only": True,
        }
        if payload.get("privacy_boundary") != privacy:
            raise PlanError(f"{label} sanitized opaque reservation privacy boundary changed")
        expected_conventions = {
            "canonical_order": "lexicographic order within each hash field; no domain, row, record, source, selection, or target metadata is exported",
            "final_sequence_sha256": "SHA-256 fingerprint of the canonical H128 final sequence including BOS",
            "public_record_sha256": "SHA-256 fingerprint of the rendered public record",
        }
        conventions = payload.get("hash_conventions")
        if conventions != expected_conventions:
            raise PlanError(f"{label} sanitized opaque reservation hash convention changed")
    else:
        if not isinstance(schema, str) or not schema.endswith(
            "opaque-source-sequence-reservation.v1"
        ):
            raise PlanError(f"{label} opaque reservation schema is not hash-only")
        if payload.get("status") != "OPAQUE_HASH_RESERVATION_FOR_FUTURE_EXCLUSION":
            raise PlanError(f"{label} opaque reservation status is not exclusion-only")
        expected_privacy = {
            "labels_or_answers_present": False,
            "record_ids_present": False,
            "row_indices_present": False,
            "source_text_present": False,
            "suitable_for_identity_exclusion_only": True,
            "token_ids_present": False,
        }
        if payload.get("privacy") != expected_privacy:
            raise PlanError(f"{label} opaque reservation privacy boundary changed")
        privacy = expected_privacy
        conventions = payload.get("hash_conventions")
        if not isinstance(conventions, Mapping):
            raise PlanError(f"{label} opaque reservation hash conventions are absent")
        sequence_convention = conventions.get("final_sequence_sha256")
        if not isinstance(sequence_convention, str):
            raise PlanError(f"{label} opaque reservation sequence convention is absent")
        normalized_convention = sequence_convention.lower().replace(" ", "")
        if "128-token" not in normalized_convention or "including-bos" not in normalized_convention:
            raise PlanError(f"{label} opaque reservation is not an H128 including-BOS export")

    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping):
        raise PlanError(f"{label} opaque reservation hashes are absent")
    source = hashes.get("public_record_sha256")
    sequence = hashes.get("final_sequence_sha256")
    if not isinstance(source, Mapping) or not isinstance(sequence, Mapping):
        raise PlanError(f"{label} opaque reservation hash summaries are malformed")
    source_values = source.get("values")
    sequence_values = sequence.get("values")
    if sanitized_p08:
        expected_source_count = source.get("ordered_count")
        expected_sequence_count = sequence.get("ordered_count")
        if (
            source.get("distinct_count") != expected_source_count
            or sequence.get("distinct_count") != expected_sequence_count
        ):
            raise PlanError(f"{label} sanitized opaque reservation counts are not distinct")
    else:
        counts = payload.get("counts")
        if not isinstance(counts, Mapping):
            raise PlanError(f"{label} opaque reservation counts are absent")
        expected_source_count = counts.get("public_record_sha256")
        expected_sequence_count = counts.get("final_sequence_sha256")
    if (
        not isinstance(source_values, list)
        or not isinstance(sequence_values, list)
        or not isinstance(expected_source_count, int)
        or not isinstance(expected_sequence_count, int)
        or expected_source_count != len(source_values)
        or expected_sequence_count != len(sequence_values)
        or len(source_values) == 0
        or len(sequence_values) == 0
        or len(set(source_values)) != len(source_values)
        or len(set(sequence_values)) != len(sequence_values)
    ):
        raise PlanError(f"{label} opaque reservation hash counts or uniqueness changed")
    for name, values in (("source", source_values), ("sequence", sequence_values)):
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in values
        ):
            raise PlanError(f"{label} opaque {name} hash values are malformed")
    summary = {
        "label": label,
        "file": descriptor,
        "task_id": payload.get("task_id"),
        "schema": schema,
        "status": payload.get("status"),
        "source_hash_count": len(source_values),
        "sequence_hash_count": len(sequence_values),
        "hash_conventions": {
            "public_record_sha256": conventions.get("public_record_sha256"),
            "final_sequence_sha256": conventions.get("final_sequence_sha256"),
        },
        "privacy": dict(privacy),
        "underlying_provenance_opened": False,
        "underlying_results_opened": False,
        "underlying_holdout_opened": False,
    }
    if sanitized_p08:
        summary["recipe"] = dict(payload["recipe"])
    return summary, frozenset(source_values), frozenset(sequence_values)


def _h128_sequence_digest(token_ids: Sequence[int]) -> str:
    import struct

    values = [int(value) for value in token_ids[:SEQUENCE_TOKENS]
    ]
    if len(values) != SEQUENCE_TOKENS:
        raise PlanError("candidate has fewer than 128 tokens for opaque H128 matching")
    return hashlib.sha256(struct.pack("<" + "i" * SEQUENCE_TOKENS, *values)).hexdigest()


def _identity_commitment(candidate: Any, *, h128_sequence_sha256: str) -> dict[str, str]:
    return {
        "record_id_sha256": hashlib.sha256(str(candidate.record_id).encode("utf-8")).hexdigest(),
        "public_record_sha256": str(candidate.public_record_sha256),
        "h128_sequence_sha256": h128_sequence_sha256,
        "final_sequence_sha256": str(candidate.final_sequence_sha256),
    }


def _commitment_digest(values: Sequence[Mapping[str, str]]) -> str:
    canonical = "\n".join(
        _canonical_json(dict(value))
        for value in sorted((dict(value) for value in values), key=_canonical_json)
    )
    return hashlib.sha256((canonical + "\n").encode("utf-8")).hexdigest()


def _validate_final_scan_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_descriptor(path)
    payload = _load_json(path, description="TRR-0009 decision contract")
    if payload.get("schema") != "token-reconstruction.trr0009-decision-contract.v1":
        raise PlanError("TRR-0009 decision contract schema changed")
    if payload.get("task_id") != TASK_ID:
        raise PlanError("TRR-0009 decision contract task identity changed")
    if payload.get("status") != "FROZEN_DECISION_CONTRACT_BEFORE_SOURCE_SELECTION":
        raise PlanError("final count-only scan requires an owner-frozen decision contract")
    methods = payload.get("methods")
    if not isinstance(methods, Mapping) or methods.get("unchanged_anchor") != "trr0009_unchanged_current_enriched__residual_mlp512":
        raise PlanError("decision contract does not bind the approved current-bank unchanged arm")
    panel = payload.get("panel")
    if not isinstance(panel, Mapping) or panel.get("finance_records_per_domain") != EXPECTED_RECORDS_BY_DOMAIN["finance"] or panel.get("pile_records_per_domain") != EXPECTED_RECORDS_BY_DOMAIN["pile"]:
        raise PlanError("decision contract panel counts are not frozen to TRR-0009")
    if panel.get("natural_ranges_half_open") != SOURCE_RANGES or panel.get("selection_seed") != SELECTION_SEED:
        raise PlanError("decision contract natural range or selection seed changed")
    starting = payload.get("starting_state")
    if not isinstance(starting, Mapping) or starting.get("sha256") != CURRENT_STATE_SHA256 or starting.get("bytes") != CURRENT_STATE_BYTES or starting.get("path") != str(CURRENT_STATE):
        raise PlanError("decision contract starting state is not the approved current-bank checkpoint")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or any(provenance.get(key) is True for key in ("p03_holdout_opened", "p06_underlying_provenance_opened", "p07_workspace_or_results_opened")):
        raise PlanError("decision contract records forbidden access")
    return record, payload


def _validate_final_scan_trr7_method_freeze(path: Path) -> dict[str, Any]:
    record = _file_descriptor(path)
    payload = _load_json(path, description="TRR-0007 method freeze")
    if payload.get("status") != "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION":
        raise PlanError("TRR-0007 method freeze is not closed before source scanning")
    if any(payload.get(key) is True for key in ("truth_opened", "fresh_evaluation_started", "source_accessed", "target_loaded", "target_labels_loaded", "private_or_truth_payload_read")):
        raise PlanError("TRR-0007 method freeze records forbidden access")
    bindings = payload.get("state_bindings")
    current = bindings.get("current_enriched__residual_mlp512") if isinstance(bindings, Mapping) else None
    state = current.get("state") if isinstance(current, Mapping) else None
    if not isinstance(state, Mapping) or state.get("bytes") != CURRENT_STATE_BYTES or state.get("sha256") != CURRENT_STATE_SHA256:
        raise PlanError("TRR-0007 method freeze current-bank state binding changed")
    return {"file": record, "status": payload.get("status"), "current_state": dict(state)}



def _validate_final_scan_task_method_freeze(
    path: Path, *, root: Path, decision_record: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the same TRR-0009 selected-method gate used by selection."""

    from scripts import trr0009_select_public as selector

    try:
        record, _payload, states = selector._validate_task_method_freeze(
            path, root=root, decision_record=decision_record
        )
    except selector.SelectionError as exc:
        raise PlanError(str(exc)) from exc
    return {
        "file": record,
        "state_sha256": {method_id: state["sha256"] for method_id, state in states.items()},
    }

def _load_p08_option(args: argparse.Namespace) -> tuple[dict[str, Any], frozenset[str], frozenset[str]]:
    if (args.p08_opaque is None) == (not args.no_p08_reservation):
        raise PlanError("inventory-final requires exactly one of --p08-opaque or --no-p08-reservation")
    if args.no_p08_reservation:
        return {
            "label": "p08",
            "status": "NO_NEW_P08_RESERVATION_DECLARED",
            "file": None,
            "source_hash_count": 0,
            "sequence_hash_count": 0,
            "underlying_provenance_opened": False,
            "underlying_results_opened": False,
            "underlying_holdout_opened": False,
        }, frozenset(), frozenset()
    path = Path(args.p08_opaque).expanduser().resolve()
    allowed_roots = [
        (_REPOSITORY_ROOT / "experiments" / "TRR-0009").resolve(),
        Path("/tmp/trr-p08").resolve(),
    ]
    if not any(path == allowed or allowed in path.parents for allowed in allowed_roots):
        raise PlanError("P08 opaque reservation must be task-owned or under /tmp/trr-p08")
    return _load_generic_opaque_reservation(path, label="p08")


def _final_inventory(args: argparse.Namespace) -> dict[str, Any]:
    """Run the deferred count-only identity scan after owner freeze."""

    root = Path(args.repository_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise PlanError(f"final inventory output already exists: {output}")
    decision_path = Path(args.decision_contract).expanduser()
    decision_path = (decision_path if decision_path.is_absolute() else root / decision_path).resolve()
    method_path = Path(args.method_freeze).expanduser()
    method_path = (method_path if method_path.is_absolute() else root / method_path).resolve()
    trr7_method_path = Path(
        getattr(args, "trr7_method_freeze", Path("experiments/TRR-0007/method_freeze.json"))
    ).expanduser()
    trr7_method_path = (trr7_method_path if trr7_method_path.is_absolute() else root / trr7_method_path).resolve()
    decision_record, decision = _validate_final_scan_contract(decision_path)
    method_record = _validate_final_scan_task_method_freeze(
        method_path, root=root, decision_record=decision_record
    )
    trr7_method_record = _validate_final_scan_trr7_method_freeze(trr7_method_path)
    planning_path = (root / Path("experiments/TRR-0009/planning/source_inventory.json")).resolve()
    planning_projection = _load_json(planning_path, description="TRR-0009 planning inventory projection")
    if planning_projection.get("status") != "IDENTITY_CAPACITY_PROJECTION_PENDING_FINAL_COUNT_ONLY_SCAN":
        raise PlanError("final inventory requires the preserved projection as its planning input")

    # Imports are delayed until this explicit command so power/projection stay
    # metadata-only and never initialize a tokenizer or Arrow reader.
    from scripts import trr0005_produce_confirmation as trusted
    from scripts import trr0006_build_eligibility as eligibility
    from scripts import trr0007_eval_select as trr7_selector
    from token_reconstruction.trr0005_public_corpus import SOURCE_PARTITIONS, deterministic_row_order, source_record_id

    p06_summary, p06_source, p06_sequence = _load_p06_opaque()
    p08_summary, p08_source, p08_sequence = _load_p08_option(args)
    p04_path = P04_OPAQUE.expanduser().resolve()
    p04_descriptor = _file_descriptor(p04_path, expected_sha256=P04_OPAQUE_SHA256)
    p04 = eligibility._load_p04_opaque_exclusions(p04_path)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    tokenizer = trusted._load_tokenizer(tokenizer_path)
    pile_paths = tuple(Path(value).expanduser().resolve() for value in args.pile_arrow)
    finance_paths = tuple(Path(value).expanduser().resolve() for value in args.finance_arrow)
    datasets = {
        "pile": trusted._load_arrow_dataset(pile_paths),
        "finance": trusted._load_arrow_dataset(finance_paths),
    }
    known = list(trr7_selector._known_exclusion_paths(root))
    known.extend(
        root / relative
        for relative in (
            TRR7_EXCLUSIONS,
            TRR7_SELECTION,
            TRR7_FINAL_BANK,
            TRR7_PARENT_ROWS,
            TRR7_CORPUS_PLAN,
            TRR7_PREFIX_EXCLUSIONS,
            TRR8_EXCLUSIONS,
            TRR8_SELECTION,
        )
    )
    known.extend(Path(value).expanduser().resolve() for value in args.exclude_source)
    exclusions = trusted._collect_exclusions(known)
    opaque_sets = (("p06", p06_source, p06_sequence), ("p08", p08_source, p08_sequence))
    domains: dict[str, dict[str, Any]] = {}
    seen_public_hashes: set[str] = set()
    seen_final_sequences: set[str] = set()
    for style in STYLE_ORDER:
        start, stop = SOURCE_RANGES[style]
        dataset = datasets[style]
        if len(dataset) < stop:
            raise PlanError(f"{style} cache has {len(dataset)} rows; need {stop}")
        counts: dict[str, Any] = {
            "style": style,
            "source_range_half_open": [start, stop],
            "scanned_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "excluded_id": 0,
            "excluded_index": 0,
            "excluded_hash": 0,
            "excluded_p04_source_hash": 0,
            "excluded_p04_h129_sequence_hash": 0,
            "excluded_p06_source_hash": 0,
            "excluded_p06_h128_sequence_hash": 0,
            "excluded_p08_source_hash": 0,
            "excluded_p08_h128_sequence_hash": 0,
            "duplicate_rendered_source": 0,
            "duplicate_final_sequence": 0,
            "valid_identity_commitments": [],
            "eligible_identity_commitments": [],
            "eligible_unique": 0,
        }
        spec = SOURCE_PARTITIONS[style]
        order = deterministic_row_order(range(start, stop), dataset_key=f"{style}-future-holdout", seed=SELECTION_SEED)
        for index in order:
            counts["scanned_rows"] += 1
            expected_id = source_record_id(str(spec["dataset_id"]), str(spec["split"]), str(spec["revision"]), index)
            if expected_id in exclusions.ids[style]:
                counts["excluded_id"] += 1
                continue
            if index in exclusions.indices[style]:
                counts["excluded_index"] += 1
                continue
            row = trusted._read_reserved_row(datasets[style], style=style, row_index=index)
            try:
                candidate = trusted._render_row(style, row, index, tokenizer)
            except trusted.ProducerError:
                counts["invalid_rows"] += 1
                continue
            counts["valid_rows"] += 1
            h128 = _h128_sequence_digest(candidate.token_ids)
            commitment = _identity_commitment(candidate, h128_sequence_sha256=h128)
            counts["valid_identity_commitments"].append(commitment)
            reason = trusted._blocked(candidate, exclusions)
            if reason == "public_source_id":
                counts["excluded_id"] += 1
                continue
            if reason == "public_source_index":
                counts["excluded_index"] += 1
                continue
            if reason in {"public_rendered_hash", "public_final_sequence_hash"}:
                counts["excluded_hash"] += 1
                continue
            if candidate.public_record_sha256 in p04.source_hashes:
                counts["excluded_p04_source_hash"] += 1
                continue
            if len(candidate.token_ids) >= SEQUENCE_TOKENS + 1 and trusted._sequence_digest(candidate.token_ids[: SEQUENCE_TOKENS + 1]) in p04.sequence_hashes_129:
                counts["excluded_p04_h129_sequence_hash"] += 1
                continue
            opaque_reason = None
            for label, source_hashes, sequence_hashes in opaque_sets:
                if candidate.public_record_sha256 in source_hashes:
                    opaque_reason = f"excluded_{label}_source_hash"
                    break
                if h128 in sequence_hashes:
                    opaque_reason = f"excluded_{label}_h128_sequence_hash"
                    break
            if opaque_reason is not None:
                counts[opaque_reason] += 1
                continue
            if candidate.public_record_sha256 in seen_public_hashes:
                counts["duplicate_rendered_source"] += 1
                continue
            if candidate.final_sequence_sha256 in seen_final_sequences:
                counts["duplicate_final_sequence"] += 1
                continue
            seen_public_hashes.add(candidate.public_record_sha256)
            seen_final_sequences.add(candidate.final_sequence_sha256)
            counts["eligible_unique"] += 1
            counts["eligible_identity_commitments"].append(commitment)
        counts["valid_identity_commitment_sha256"] = _commitment_digest(counts.pop("valid_identity_commitments"))
        counts["eligible_identity_commitment_sha256"] = _commitment_digest(counts.pop("eligible_identity_commitments"))
        requested = EXPECTED_RECORDS_BY_DOMAIN[style]
        counts["capacity_for_requested_per_domain"] = {
            "requested": requested,
            "sufficient": counts["eligible_unique"] >= requested,
            "surplus_or_shortfall": counts["eligible_unique"] - requested,
        }
        domains[style] = counts

    result = dict(planning_projection)
    result.update({
        "status": "IDENTITY_INVENTORY_COMPLETE_NO_SELECTION_NO_TRUTH",
        "created_utc": _utc_now(),
        "sample_size_status": "PROPOSED_COUNT_CHECKED_NOT_SELECTED",
        "requested_per_domain": dict(EXPECTED_RECORDS_BY_DOMAIN),
        "domains": domains,
        "p08_opaque_reservation": p08_summary,
        "public_inputs": {
            "pile_arrow": [_file_descriptor(path) for path in pile_paths],
            "finance_arrow": [_file_descriptor(path) for path in finance_paths],
            "tokenizer": {"path": str(tokenizer_path)},
        },
        "final_scan": {
            "planning_projection": _file_descriptor(planning_path),
            "decision_contract": decision_record,
            "trr0009_method_freeze": method_record["file"],
            "trr0009_method_freeze_state_sha256": dict(method_record["state_sha256"]),
            "trr0007_method_freeze": trr7_method_record["file"],
            "p04_opaque_exchange": p04_descriptor,
            "p06_opaque_reservation": p06_summary["file"],
            "p08_reservation_status": p08_summary.get("status"),
            "selection_started": False,
            "truth_opened": False,
        },
        "execution": {
            "new_source_scan": True,
            "projection_only": False,
            "final_count_only_scan_required": False,
            "source_rows_transiently_read": True,
            "token_ids_transiently_derived": True,
            "selection_performed": False,
            "model_loaded": False,
            "source_text_written": False,
            "token_ids_written": False,
            "truth_created_or_opened": False,
            "private_or_holdout_opened": False,
            "network_used": False,
        },
        "exclusion_policy": {
            "identity_only": True,
            "all_known_fit_validation_opened_eval_ledgers_applied": True,
            "trr8_panel_excluded": True,
            "p04_opaque_identity_only": True,
            "p06_opaque_identity_only": True,
            "p08_opaque_identity_only": p08_summary.get("status") != "NO_NEW_P08_RESERVATION_DECLARED",
            "p08_new_reservation_status": p08_summary.get("status"),
            "p03_holdout_opened": False,
            "p07_workspace_or_results_opened": False,
            "source_text_or_target_labels_written": False,
        },
        "limitations": [
            "Counts are an identity-only capacity audit; no source row is selected, frozen, observed, or truth-bound.",
            "Public source rows and H128 token sequences are rendered transiently only to apply identity exclusions; aggregate counts and commitment digests are retained, never source payloads.",
            "Opaque source and H128 sequence sets are applied as a conservative union; no provenance, holdout, or result payload is opened.",
            "A capacity shortfall leaves the fixed panel unresolved and cannot trigger automatic sample expansion.",
        ],
    })
    result["p04_opaque_exchange"] = {"file": p04_descriptor, "identity_only": True}
    result["p08_opaque_reservation"] = p08_summary
    return {"result": result, "output": _write(output, result)}

def _write(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PlanError(f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return _file_descriptor(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    power = sub.add_parser("power", help="write prospective paired-discordance power analysis")
    power.add_argument("--output", type=Path, required=True)
    inventory = sub.add_parser("inventory", help="write the metadata-only capacity projection")
    inventory.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("inventory-final", help="run the deferred count-only public identity scan")
    final.add_argument("--repository-root", type=Path, default=Path("."))
    final.add_argument("--decision-contract", type=Path, default=Path("experiments/TRR-0009/planning/decision_contract.json"))
    final.add_argument("--method-freeze", type=Path, default=Path("experiments/TRR-0009/training/method_freeze.json"), help="frozen TRR-0009 selected-method state binding")
    final.add_argument("--trr7-method-freeze", type=Path, default=Path("experiments/TRR-0007/method_freeze.json"), help="inherited TRR-0007 method freeze")
    final.add_argument("--tokenizer", type=Path, required=True)
    final.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    final.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    final.add_argument("--exclude-source", type=Path, nargs="*", default=[])
    final.add_argument("--p08-opaque", type=Path)
    final.add_argument("--no-p08-reservation", action="store_true")
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "power":
        result = _power_artifact()
        record = _write(args.output, result)
        print(json.dumps({"task_id": TASK_ID, "output": record, "status": result["status"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "inventory":
        result = _build_inventory()
        record = _write(args.output, result)
        print(json.dumps({"task_id": TASK_ID, "output": record, "status": result["status"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "inventory-final":
        result = _final_inventory(args)
        print(json.dumps({"task_id": TASK_ID, "output": result["output"], "status": result["result"]["status"]}, indent=2, sort_keys=True))
        return 0
    raise PlanError(f"unknown planning command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PlanError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0009 planning error: {exc}")
