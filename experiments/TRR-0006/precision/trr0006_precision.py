#!/usr/bin/env python3
"""Bounded, read-only TRR-0006 precision preflight.

This script reads only the published TRR-0005 decision plan, scorer source,
and truth-gated published result.  It does not load predictions, source truth,
model state, or private sidecars.  Exact-record planning probabilities use the
registered one-sided Clopper-Pearson endpoint and a multinomial model for the
mutually exclusive beneficial/harmful/neutral discordance events.  A small
9-category simulation uses the published per-record cross-target pattern only
as a labeled dependence-shaped scenario; Frechet bounds are reported without
assuming target independence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


ALPHA_EXACT = 0.05 / 32.0
ALPHA_TOKEN = 0.05 / 16.0
EXACT_MARGIN = 0.05
TOKEN_MARGIN = 0.005
DEFAULT_SIM_DRAWS = 20_000
DEFAULT_SEED = 6006
STATUSES = ("G", "L", "N")
CATEGORIES = tuple(a + b for a in STATUSES for b in STATUSES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cp_upper(n: int, k: int, alpha: float = ALPHA_EXACT) -> float:
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    return float(stats.beta.ppf(1.0 - alpha, k + 1, n - k))


def cp_lower(n: int, k: int, alpha: float = ALPHA_EXACT) -> float:
    if k == 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)
    return float(stats.beta.ppf(alpha, k, n - k + 1))


def exact_net_upper(n: int, g: int, h: int) -> float:
    return cp_upper(n, g) - cp_lower(n, h)


def exact_net_lower(n: int, g: int, h: int) -> float:
    return cp_lower(n, g) - cp_upper(n, h)


def _valid_rates(pg: float, ph: float) -> None:
    if not (0.0 <= pg <= 1.0 and 0.0 <= ph <= 1.0 and pg + ph <= 1.0):
        raise ValueError(f"invalid mutually-exclusive rates pg={pg}, ph={ph}")


def exact_multinomial_probabilities(
    n: int, pg: float, ph: float, margin: float = EXACT_MARGIN
) -> dict[str, float]:
    """Exact probabilities under Multinomial(n, pg, ph, 1-pg-ph).

    G is a beneficial exact discordance, L is a harmful exact discordance.
    Conditional on G=g, H is binomial with n-g trials and probability
    ph/(1-pg), so monotone CP thresholds reduce the calculation to O(n).
    """

    _valid_rates(pg, ph)
    upper = np.fromiter((cp_upper(n, g) for g in range(n + 1)), float)
    lower = np.fromiter((cp_lower(n, h) for h in range(n + 1)), float)
    g_pmf = stats.binom.pmf(np.arange(n + 1), n, pg)
    q_h_given_not_g = ph / (1.0 - pg) if pg < 1.0 else 0.0
    p_exclude = 0.0
    p_point_useful = 0.0
    p_lower_support = 0.0
    min_point_delta = math.ceil(n * margin - 1e-15)
    for g, p_g in enumerate(g_pmf):
        if p_g == 0.0:
            continue
        max_h = n - g

        # Registered exclusion event: U_CP(g) - L_CP(h) <= margin.
        h_candidates = np.flatnonzero(lower[: max_h + 1] >= upper[g] - margin)
        if h_candidates.size:
            h_min = int(h_candidates[0])
            p_exclude += p_g * float(stats.binom.sf(h_min - 1, max_h, q_h_given_not_g))

        # Descriptive candidate-useful point estimate: (g-h)/n >= margin.
        h_max = min(max_h, g - min_point_delta)
        if h_max >= 0:
            p_point_useful += p_g * float(stats.binom.cdf(h_max, max_h, q_h_given_not_g))

        # Diagnostic only: lower CP net endpoint >= margin. This is not a
        # registered support criterion, but shows how conservative the tail is.
        h_candidates = np.flatnonzero(upper[: max_h + 1] <= lower[g] - margin)
        if h_candidates.size:
            h_max_support = int(h_candidates[-1])
            p_lower_support += p_g * float(
                stats.binom.cdf(h_max_support, max_h, q_h_given_not_g)
            )

    return {
        "exclude_exact_margin_probability": float(p_exclude),
        "point_estimate_reaches_exact_margin_probability": float(p_point_useful),
        "lower_endpoint_exceeds_exact_margin_probability_diagnostic": float(p_lower_support),
    }


def expected_bound(n: int, pg: float, ph: float) -> dict[str, Any]:
    # Planning plug-in counts use the nearest integer counts to n*p. This is
    # a descriptive calculation, not a decision probability.
    g = int(round(n * pg))
    h = int(round(n * ph))
    if g + h > n:
        h = n - g
    return {
        "expected_counts_round": {"beneficial": g, "harmful": h},
        "beneficial_rate_pp": 100.0 * g / n,
        "harmful_rate_pp": 100.0 * h / n,
        "net_point_pp": 100.0 * (g - h) / n,
        "gain_upper_pp": 100.0 * cp_upper(n, g),
        "loss_lower_pp": 100.0 * cp_lower(n, h),
        "net_upper_pp": 100.0 * exact_net_upper(n, g, h),
        "net_lower_pp": 100.0 * exact_net_lower(n, g, h),
    }


def extract_published(result_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(result_path.read_text())
    relevant: dict[str, Any] = {}
    target_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for domain in ("pile", "finance"):
        target_rows[domain] = {}
        for condition in ("public_base", "public_lora_2601"):
            key = f"{domain}__{condition}__enriched__causal_vs_diagonal"
            comp = result["method_comparisons"][key]
            rows = comp["paired_record_differences"]
            if len(rows) != 128:
                raise ValueError(f"{key} expected 128 rows, got {len(rows)}")
            target_rows[domain][condition] = {
                "record_ids": [row["record_id"] for row in rows],
                "rows": rows,
                "g": int(comp["gains_and_regressions"]["beneficial_exact_records"]),
                "h": int(comp["gains_and_regressions"]["harmful_exact_records"]),
                "token_delta_counts": dict(
                    Counter(int(row["correct_tokens_delta"]) for row in rows)
                ),
                "token_delta_mean_tokens": float(
                    statistics.mean(int(row["correct_tokens_delta"]) for row in rows)
                ),
                "token_delta_sd_tokens": float(
                    statistics.stdev(int(row["correct_tokens_delta"]) for row in rows)
                ),
            }
        ids0 = target_rows[domain]["public_base"]["record_ids"]
        ids1 = target_rows[domain]["public_lora_2601"]["record_ids"]
        if ids0 != ids1:
            raise ValueError(f"target pairing changed for {domain}")
        categories = []
        for row0, row1 in zip(
            target_rows[domain]["public_base"]["rows"],
            target_rows[domain]["public_lora_2601"]["rows"],
        ):
            s0 = "G" if row0["beneficial_exact"] else "L" if row0["harmful_exact"] else "N"
            s1 = "G" if row1["beneficial_exact"] else "L" if row1["harmful_exact"] else "N"
            categories.append(s0 + s1)
        category_counts = Counter(categories)
        target_rows[domain]["joint_categories"] = {
            category: int(category_counts.get(category, 0)) for category in CATEGORIES
        }
        relevant[domain] = {
            "paired_exact_counts": {
                condition: {
                    "records": 128,
                    "beneficial": target_rows[domain][condition]["g"],
                    "harmful": target_rows[domain][condition]["h"],
                }
                for condition in ("public_base", "public_lora_2601")
            },
            "joint_target_category_counts": target_rows[domain]["joint_categories"],
            "token_delta_summary": {
                condition: {
                    "mean_delta_tokens_per_record": target_rows[domain][condition][
                        "token_delta_mean_tokens"
                    ],
                    "sd_delta_tokens_per_record": target_rows[domain][condition][
                        "token_delta_sd_tokens"
                    ],
                    "counts": target_rows[domain][condition]["token_delta_counts"],
                }
                for condition in ("public_base", "public_lora_2601")
            },
        }
    return relevant, target_rows


def target_dependence_simulation(
    category_counts: dict[str, int],
    n: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    counts = np.asarray([category_counts[c] for c in CATEGORIES], dtype=float)
    probs = counts / counts.sum()
    rng = np.random.default_rng(seed)
    simulated = rng.multinomial(n, probs, size=draws)
    first_g = simulated[:, [i for i, c in enumerate(CATEGORIES) if c[0] == "G"]].sum(axis=1)
    first_h = simulated[:, [i for i, c in enumerate(CATEGORIES) if c[0] == "L"]].sum(axis=1)
    second_g = simulated[:, [i for i, c in enumerate(CATEGORIES) if c[1] == "G"]].sum(axis=1)
    second_h = simulated[:, [i for i, c in enumerate(CATEGORIES) if c[1] == "L"]].sum(axis=1)
    first_exclude = np.fromiter(
        (exact_net_upper(n, int(g), int(h)) <= EXACT_MARGIN for g, h in zip(first_g, first_h)),
        dtype=bool,
        count=draws,
    )
    second_exclude = np.fromiter(
        (exact_net_upper(n, int(g), int(h)) <= EXACT_MARGIN for g, h in zip(second_g, second_h)),
        dtype=bool,
        count=draws,
    )
    first_point = (first_g - first_h) >= math.ceil(n * EXACT_MARGIN - 1e-15)
    second_point = (second_g - second_h) >= math.ceil(n * EXACT_MARGIN - 1e-15)
    joint_exclude = first_exclude & second_exclude
    joint_point = first_point & second_point
    q0 = float(first_exclude.mean())
    q1 = float(second_exclude.mean())
    q_joint = float(joint_exclude.mean())
    return {
        "n": n,
        "draws": draws,
        "seed": seed,
        "marginal_exclude_probability_mc": [q0, q1],
        "joint_both_exclude_probability_mc": q_joint,
        "joint_both_exclude_mc_se": math.sqrt(q_joint * (1.0 - q_joint) / draws),
        "frechet_joint_exclude_lower": max(0.0, q0 + q1 - 1.0),
        "frechet_joint_exclude_upper": min(q0, q1),
        "marginal_point_useful_probability_mc": [float(first_point.mean()), float(second_point.mean())],
        "joint_both_point_useful_probability_mc": float(joint_point.mean()),
        "joint_both_point_useful_mc_se": math.sqrt(
            float(joint_point.mean()) * (1.0 - float(joint_point.mean())) / draws
        ),
        "category_probabilities": {c: float(p) for c, p in zip(CATEGORIES, probs)},
        "dependence_note": (
            "Published per-record 3x3 exact-event pattern used as a TRR-0005-shaped planning scenario; "
            "this is not a guarantee for the new natural panel. Frechet bounds require no target-independence assumption."
        ),
    }


def token_normal_sensitivity(
    token_sd_tokens_per_record: float,
    n: int,
    means_pp: tuple[float, ...] = (0.0, 0.2, 0.5, 1.0),
) -> list[dict[str, float]]:
    """Approximate registered bootstrap upper endpoint via normal radius.

    The published result stores only per-record correct-token counts, so this
    compact planning diagnostic models the paired record delta as Normal with
    the observed SD.  It is not a replacement for the preregistered bootstrap
    once new predictions exist.
    """

    se_pp = 100.0 * token_sd_tokens_per_record / (127.0 * math.sqrt(n))
    z = float(stats.norm.ppf(1.0 - ALPHA_TOKEN))
    radius_pp = z * se_pp
    out = []
    for mean_pp in means_pp:
        upper_mean_pp = mean_pp + radius_pp
        p_point = float(1.0 - stats.norm.cdf((TOKEN_MARGIN * 100.0 - mean_pp) / se_pp))
        p_exclude = float(
            stats.norm.cdf((TOKEN_MARGIN * 100.0 - radius_pp - mean_pp) / se_pp)
        )
        out.append(
            {
                "true_mean_delta_pp": mean_pp,
                "record_delta_se_pp": se_pp,
                "normal_bootstrap_radius_pp": radius_pp,
                "expected_upper_endpoint_pp": upper_mean_pp,
                "point_estimate_reaches_token_margin_probability": p_point,
                "upper_endpoint_below_token_margin_probability": p_exclude,
            }
        )
    return out


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trr5-root",
        type=Path,
        default=Path("/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005"),
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/trr0006_precision"))
    parser.add_argument("--draws", type=int, default=DEFAULT_SIM_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.draws <= 0:
        raise SystemExit("--draws must be positive")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    trr5_root = args.trr5_root.resolve()
    plan_path = trr5_root / "experiments/TRR-0005/decision_plan.json"
    result_path = trr5_root / "experiments/TRR-0005/fresh_confirmation_v1/result.json"
    scorer_path = trr5_root / "scripts/trr0005_score_confirmation.py"
    plan = json.loads(plan_path.read_text())
    result = json.loads(result_path.read_text())
    relevant, rows = extract_published(result_path)

    # Check the published scorer constants/source binding used for this run.
    scorer_text = scorer_path.read_text()
    scorer_constants = {
        "exact_family_tail_alpha_literal": "EXACT_FAMILY_TAIL_ALPHA = 0.05 / 32" in scorer_text,
        "token_family_tail_alpha_literal": "TOKEN_FAMILY_TAIL_ALPHA = 0.05 / 16" in scorer_text,
        "exact_net_formula_present": '"U(p_gain) - L(p_loss)' in scorer_text,
    }
    if not all(scorer_constants.values()):
        raise RuntimeError(f"published scorer binding check failed: {scorer_constants}")

    scenario_rates = {
        "null_no_discordance": (0.0, 0.0),
        "pile_like_observed": (1.0 / 128.0, 1.0 / 128.0),
        "finance_p0_observed": (5.0 / 128.0, 4.0 / 128.0),
        "finance_lora_observed": (5.0 / 128.0, 6.0 / 128.0),
        "moderate_net_plus_2_low": (0.04, 0.02),
        "moderate_net_plus_2": (0.06, 0.04),
        "useful_net_plus_5_low_harm": (0.08, 0.03),
        "useful_net_plus_5_balanced": (0.10, 0.05),
        "balanced_zero_net_high_discordance": (0.05, 0.05),
    }
    exact_sensitivity: dict[str, Any] = {}
    for name, (pg, ph) in scenario_rates.items():
        exact_sensitivity[name] = {
            "true_beneficial_rate": pg,
            "true_harmful_rate": ph,
            "true_net_rate": pg - ph,
            "n128": {
                **expected_bound(128, pg, ph),
                **exact_multinomial_probabilities(128, pg, ph),
            },
            "n1024": {
                **expected_bound(1024, pg, ph),
                **exact_multinomial_probabilities(1024, pg, ph),
            },
        }

    dependence = {
        domain: target_dependence_simulation(
            relevant[domain]["joint_target_category_counts"],
            n=1024,
            draws=args.draws,
            seed=args.seed + (0 if domain == "pile" else 1),
        )
        for domain in ("pile", "finance")
    }
    # Exact per-target probabilities for the observed-rate scenario, followed
    # by target-dependence-free Frechet bounds. These are the key planning
    # values; the 9-category MC above is an optional shape-sensitive estimate.
    observed_target_power: dict[str, Any] = {}
    for domain in ("pile", "finance"):
        q = {}
        for condition in ("public_base", "public_lora_2601"):
            c = relevant[domain]["paired_exact_counts"][condition]
            q[condition] = exact_multinomial_probabilities(
                1024,
                c["beneficial"] / c["records"],
                c["harmful"] / c["records"],
            )
        q0 = q["public_base"]["exclude_exact_margin_probability"]
        q1 = q["public_lora_2601"]["exclude_exact_margin_probability"]
        observed_target_power[domain] = {
            "per_target_exact_exclusion_probability": q,
            "joint_target_exact_exclusion_frechet_lower": max(0.0, q0 + q1 - 1.0),
            "joint_target_exact_exclusion_frechet_upper": min(q0, q1),
            "joint_target_note": "No independence assumption across P0 and synthetic-LoRA; same source IDs are paired.",
        }

    token = {}
    for domain in ("pile", "finance"):
        token[domain] = {}
        for condition in ("public_base", "public_lora_2601"):
            s = relevant[domain]["token_delta_summary"][condition]["sd_delta_tokens_per_record"]
            token[domain][condition] = {
                "observed_sd_tokens_per_record": s,
                "n128": token_normal_sensitivity(s, 128),
                "n1024": token_normal_sensitivity(s, 1024),
            }

    elapsed = time.perf_counter() - t0
    output = {
        "schema": "token-reconstruction.trr0006-precision-preflight.v1",
        "task_id": "TRR-0006",
        "status": "PROVISIONAL_ADEQUACY_PRETRUTH_NO_REGISTRATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "computation": {
            "runtime_seconds": elapsed,
            "cpu_only": True,
            "requested_simulation_draws": args.draws,
            "seed": args.seed,
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
            "platform": platform.platform(),
        },
        "source_binding": {
            "trr5_root": str(trr5_root),
            "trr5_git_head": git_head(trr5_root),
            "decision_plan_path": str(plan_path),
            "decision_plan_sha256": sha256_file(plan_path),
            "published_result_path": str(result_path),
            "published_result_sha256": sha256_file(result_path),
            "scorer_path": str(scorer_path),
            "scorer_sha256": sha256_file(scorer_path),
            "parent_commit_requested": "3a7e8f579e713c3e41d02639237042ca26fd019b",
        },
        "registered_rule": {
            "token_useful_margin_pp": 0.5,
            "exact_useful_margin_pp": 5.0,
            "exact_cp_tail_alpha_each": ALPHA_EXACT,
            "token_bootstrap_tail_alpha": ALPHA_TOKEN,
            "exact_bound": "U_CP(p_gain) - L_CP(p_loss)",
            "decision_plan_practical_margins": plan["practical_margins"],
            "decision_plan_uncertainty": plan["uncertainty"],
            "scorer_binding_checks": scorer_constants,
        },
        "published_observed_events": relevant,
        "exact_rate_sensitivity": exact_sensitivity,
        "observed_rate_target_dependence": observed_target_power,
        "target_dependence_9category_simulation": dependence,
        "token_margin_normal_diagnostic": token,
        "limitations": [
            "All rates are planning scenarios; no new truth or model evaluation was run.",
            "Exact per-target probabilities assume IID mutually exclusive G/L/N records.",
            "Cross-target joint probability is bounded by Frechet inequalities; the 9-category simulation is only a TRR-0005-shaped sensitivity scenario.",
            "Token planning uses an observed per-record delta SD and a normal approximation to the registered source bootstrap; it is not a replacement for the frozen bootstrap scorer.",
            "The 1024-record plug-in count is a descriptive expected-count calculation, not a decision probability.",
        ],
    }
    json_path = out / "precision_preflight.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    # Human-readable compact receipt for parent review.
    lines = [
        "# TRR-0006 precision preflight (provisional)",
        "",
        "Status: CPU-only, read-only planning; no new truth/model/fits and no registration.",
        "",
        f"Exact rule: B = U_CP(g;n,{ALPHA_EXACT:.7f}) - L_CP(h;n,{ALPHA_EXACT:.7f}); exclusion event B <= 5 pp.",
        "Token rule: registered source-bootstrap upper tail alpha = 0.05/16 = 0.003125; margin = 0.5 pp.",
        "",
        "## Exact endpoint sensitivity at n=1024 per target",
        "",
        "| Scenario (pg, ph) | P(exclude 5pp) | P(point net >=5pp) | plug-in net upper pp |",
        "|---|---:|---:|---:|",
    ]
    for name, row in exact_sensitivity.items():
        r = row["n1024"]
        lines.append(
            f"| {name} ({row['true_beneficial_rate']:.4f}, {row['true_harmful_rate']:.4f}) | "
            f"{r['exclude_exact_margin_probability']:.6f} | "
            f"{r['point_estimate_reaches_exact_margin_probability']:.6f} | "
            f"{r['net_upper_pp']:.3f} |"
        )
    lines += [
        "",
        "## Observed-rate target dependence",
        "",
        "| Domain | P0 exclusion | LoRA exclusion | target-joint Frechet range | 9-category paired-shape MC |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in ("pile", "finance"):
        d = observed_target_power[domain]
        q0 = d["per_target_exact_exclusion_probability"]["public_base"]["exclude_exact_margin_probability"]
        q1 = d["per_target_exact_exclusion_probability"]["public_lora_2601"]["exclude_exact_margin_probability"]
        dep = dependence[domain]
        lines.append(
            f"| {domain} | {q0:.6f} | {q1:.6f} | "
            f"[{d['joint_target_exact_exclusion_frechet_lower']:.6f}, {d['joint_target_exact_exclusion_frechet_upper']:.6f}] | "
            f"{dep['joint_both_exclude_probability_mc']:.6f} ± {1.96*dep['joint_both_exclude_mc_se']:.6f} |"
        )
    lines += [
        "",
        "The TRR-0005-shaped Finance joint estimate is close to the no-independence Frechet interval; it is a scenario, not a guarantee for the new panel. Pile is effectively certain under its very low observed discordance rates. Across all four cells, a target-independent product would be only a labeled assumption; without independence, use the corresponding Frechet bounds.",
        "",
        "## Provisional adequacy recommendation",
        "",
        "The 1024-per-domain plan is adequate for the registered exact-margin exclusion if rates remain near the observed Pile/Finance discordance rates: per-target exclusion probabilities are effectively 1.000 (Pile), 0.805 (Finance P0), and 0.981 (Finance LoRA), with a dependence-aware Finance target-joint range of about 0.785–0.805. It is not guaranteed to resolve a small positive exact effect: at true net +2 pp, exclusion is about 0.14 for a (6%,4%) rate pair; at true net +5 pp, exclusion is approximately 3e-5 while the point estimate reaches +5 pp only about 0.49 of the time. Keep the recommendation provisional until the remaining packet tail provides the final registration/selection instruction.",
        "",
        f"Artifact: `{json_path}`; elapsed CPU time {elapsed:.3f}s; simulation draws per domain {args.draws}.",
    ]
    md_path = out / "precision_preflight.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)
    print(f"elapsed_seconds={elapsed:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
