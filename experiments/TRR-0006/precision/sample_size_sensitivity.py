#!/usr/bin/env python3
"""Bounded CPU sample-size sensitivity for the registered TRR-0005 exact bound.

The calculation is pre-truth and reads only the published TRR-0005 result.
For each mutually exclusive exact event (beneficial, harmful, neutral), the
study count is multinomial.  The event probability is evaluated exactly by
conditioning on the beneficial count.  No target-independence assumption is
made in this script; target dependence is handled separately by the parent
preflight's paired-category simulation/Frechet bounds.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


ALPHA = 0.05 / 32.0
MARGIN = 0.05
SCENARIOS = {
    "finance_p0_observed": (5 / 128, 4 / 128),
    "finance_lora_observed": (5 / 128, 6 / 128),
    # Higher total discordance while retaining plausible positive/neutral net
    # effects; these are sensitivity cases, not new claims about Finance.
    "higher_discordance_net_plus_3": (0.08, 0.05),
    "higher_discordance_net_plus_5": (0.10, 0.05),
    "higher_discordance_net_plus_5b": (0.12, 0.07),
    "higher_discordance_zero_net": (0.10, 0.10),
}
SAMPLE_SIZES = (128, 768, 1024, 1536, 2048)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cp_upper(n: int, k: int) -> float:
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - ALPHA ** (1.0 / n)
    return float(stats.beta.ppf(1 - ALPHA, k + 1, n - k))


def cp_lower(n: int, k: int) -> float:
    if k == 0:
        return 0.0
    if k == n:
        return ALPHA ** (1.0 / n)
    return float(stats.beta.ppf(ALPHA, k, n - k + 1))


def exact_probabilities(n: int, pg: float, ph: float) -> dict[str, float]:
    """Return exclusion and point-useful probabilities exactly under a 3-cell multinomial."""
    if pg < 0 or ph < 0 or pg + ph > 1:
        raise ValueError((pg, ph))
    upper = np.array([cp_upper(n, g) for g in range(n + 1)])
    lower = np.array([cp_lower(n, h) for h in range(n + 1)])
    g_pmf = stats.binom.pmf(np.arange(n + 1), n, pg)
    conditional_h = ph / (1 - pg) if pg < 1 else 0.0
    p_exclude = 0.0
    p_point_useful = 0.0
    useful_count = math.ceil(n * MARGIN - 1e-15)
    for g, p_g in enumerate(g_pmf):
        if p_g == 0:
            continue
        max_h = n - g
        # U(g)-L(h) <= 5 percentage points iff h reaches this monotone cutoff.
        candidates = np.flatnonzero(lower[: max_h + 1] >= upper[g] - MARGIN)
        if candidates.size:
            p_exclude += p_g * stats.binom.sf(int(candidates[0]) - 1, max_h, conditional_h)
        # Point estimate net exact discordance >= 5 percentage points.
        h_max = min(max_h, g - useful_count)
        if h_max >= 0:
            p_point_useful += p_g * stats.binom.cdf(h_max, max_h, conditional_h)
    return {
        "exclude_probability": float(p_exclude),
        "point_net_at_least_5pp_probability": float(p_point_useful),
    }


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def main() -> int:
    t0 = time.perf_counter()
    root = Path(__file__).resolve().parents[3]
    # parents[3] is the TRR-0006 worktree; published TRR-0005 is sibling.
    trr5 = root.parent / "TRR-0005"
    result_path = trr5 / "experiments/TRR-0005/fresh_confirmation_v1/result.json"
    plan_path = trr5 / "experiments/TRR-0005/decision_plan.json"
    result = json.loads(result_path.read_text())
    plan = json.loads(plan_path.read_text())
    published = {}
    for condition in ("public_base", "public_lora_2601"):
        key = f"finance__{condition}__enriched__causal_vs_diagonal"
        comp = result["method_comparisons"][key]
        published[condition] = {
            "records": comp["records"],
            "beneficial": comp["gains_and_regressions"]["beneficial_exact_records"],
            "harmful": comp["gains_and_regressions"]["harmful_exact_records"],
        }
    if published["public_base"] != {"records": 128, "beneficial": 5, "harmful": 4}:
        raise RuntimeError(f"unexpected published Finance P0 counts: {published}")
    if published["public_lora_2601"] != {"records": 128, "beneficial": 5, "harmful": 6}:
        raise RuntimeError(f"unexpected published Finance LoRA counts: {published}")

    table = {}
    for name, (pg, ph) in SCENARIOS.items():
        table[name] = {
            "true_beneficial_rate": pg,
            "true_harmful_rate": ph,
            "true_net_rate": pg - ph,
            "sample_sizes": {},
        }
        for n in SAMPLE_SIZES:
            g = round(n * pg)
            h = round(n * ph)
            h = min(h, n - g)
            table[name]["sample_sizes"][str(n)] = {
                "expected_count_round": {"beneficial": g, "harmful": h},
                "plug_in_net_upper_pp": 100 * (cp_upper(n, g) - cp_lower(n, h)),
                **exact_probabilities(n, pg, ph),
            }

    elapsed = time.perf_counter() - t0
    output = {
        "schema": "token-reconstruction.trr0006-sample-size-sensitivity.v1",
        "task_id": "TRR-0006",
        "status": "PROVISIONAL_PRETRUTH_NO_REGISTRATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": {
            "exact_margin_pp": 5.0,
            "tail_alpha_each": ALPHA,
            "bound": "U_CP(p_gain)-L_CP(p_loss)",
            "sample_sizes": list(SAMPLE_SIZES),
        },
        "source_binding": {
            "trr5_root": str(trr5),
            "trr5_git_head": git_head(trr5),
            "decision_plan_path": str(plan_path),
            "decision_plan_sha256": sha256_file(plan_path),
            "published_result_path": str(result_path),
            "published_result_sha256": sha256_file(result_path),
            "parent_commit_requested": "3a7e8f579e713c3e41d02639237042ca26fd019b",
        },
        "published_finance_counts": published,
        "scenarios": table,
        "computation": {
            "cpu_only": True,
            "single_thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            "runtime_seconds": elapsed,
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "limitations": [
            "Each target's G/L/N events are modeled as IID mutually exclusive multinomial categories.",
            "Target dependence is not collapsed into a product; joint target claims require the parent paired-pattern simulation or Frechet bounds.",
            "Plug-in bounds use rounded expected counts and are descriptive, not decision probabilities.",
            "No new truth, model state, fitting, selection, or registration was performed.",
        ],
    }
    out = root / "experiments/TRR-0006/precision/sample_size_sensitivity.json"
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# TRR-0006 bounded exact sample-size sensitivity",
        "",
        "Pre-truth CPU-only planning under the registered exact bound `U_CP(g)-L_CP(h)`, with one-sided tail alpha 0.05/32 = 0.0015625 and exclusion margin 5 pp.",
        "",
        "| Scenario | n | plug-in B upper pp | P(exclude 5 pp) | P(point net >= 5 pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in table.items():
        for n in SAMPLE_SIZES:
            s = row["sample_sizes"][str(n)]
            lines.append(
                f"| {name} ({row['true_beneficial_rate']:.3f},{row['true_harmful_rate']:.3f}) | {n} | "
                f"{s['plug_in_net_upper_pp']:.3f} | {s['exclude_probability']:.6f} | "
                f"{s['point_net_at_least_5pp_probability']:.6f} |"
            )
    lines += [
        "",
        "The observed Finance P0 rates yield about 0.805 exclusion probability at n=1024 (0.571 at n=768; 0.977 at n=1536), while the observed LoRA rates yield about 0.981 at n=1024 (0.892 at n=768; 0.9998 at n=1536). Higher-discordance positive-net scenarios are much less likely to be excluded: at true net +3 pp (8%,5%), the n=1024 exclusion probability is about 0.011; at true net +5 pp (10%,5%), it is about 0.00003. These are sensitivity probabilities, not a registration decision.",
        "",
        f"Artifact: `{out}`; elapsed CPU time {elapsed:.3f}s.",
    ]
    md = out.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    print(out)
    print(md)
    print(f"elapsed_seconds={elapsed:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
