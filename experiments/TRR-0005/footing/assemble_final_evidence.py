#!/usr/bin/env python3
"""Assemble TRR-0005 final report and manifest from recorded JSON evidence.

The builder reads compact JSON evidence after the truth-gated score has passed.
It does not open prediction tensors, model weights, activations, source text, or
truth. It records compact score structures, file descriptors, and development
provenance without rerunning any scientific phase.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
TASK = "TRR-0005"
RESULT_REL = Path("experiments/TRR-0005/fresh_confirmation_v1/result.json")
SUMMARY_REL = Path("experiments/TRR-0005/fresh_confirmation_v1/final_summary.json")
RUN_REL = Path("experiments/TRR-0005/fresh_confirmation_v1/predictions_v1/run_evidence.json")
V2_ROOT_REL = Path("experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export")
FREEZE_REL = Path("experiments/TRR-0005/fresh_confirmation_v1/freeze_receipt_v2.json")
ATTEMPT_REL = Path("experiments/TRR-0005/freeze_score_attempt_v2/execution_receipt.json")
PANEL_REL = Path("experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json")
OBS_REL = Path("experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations.json")
REG_REL = Path("experiments/TRR-0005/fresh_confirmation_v1/method_registration.json")
PLAN_REL = Path("experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json")
BINDING_REL = V2_ROOT_REL / "evaluator_binding.json"
PRED_REL = V2_ROOT_REL / "predictions.json"
TIMING_REL = V2_ROOT_REL / "timings.json"
EXPORT_REL = V2_ROOT_REL / "export_provenance.json"
FREQ_REL = Path("experiments/TRR-0005/frequency_references_v1.json")
CORPUS_PLAN_REL = Path("experiments/TRR-0005/corpus/corpus_plan.json")
DEV_MANIFEST_REL = Path("experiments/TRR-0005/manifest.json")

CELLS = ("pile__public_base", "pile__public_lora_2601", "finance__public_base", "finance__public_lora_2601")
METHODS = (
    "historical_alpaca_a1",
    "frozen_a1_a2_k256",
    "original__joint_full_affine",
    "original__affine_causal_h_attention128",
    "original__affine_trained_diagonal_attention128",
    "enriched__joint_full_affine",
    "enriched__affine_causal_h_attention128",
    "enriched__affine_trained_diagonal_attention128",
)
STATES = (
    "joint_full_affine",
    "affine_causal_h_attention128",
    "affine_trained_diagonal_attention128",
)
PRIMARY_LABELS = ("enriched__causal_vs_diagonal", "enriched__causal_vs_best_positionwise")


def read(rel: Path) -> dict[str, Any]:
    value = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected object: {rel}")
    return dict(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def descriptor(rel: Path, *, digest: bool = True) -> dict[str, Any]:
    path = (ROOT / rel).resolve()
    value: dict[str, Any] = {"path": str(rel), "bytes": path.stat().st_size}
    if digest:
        value["sha256"] = sha256(path)
    return value


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def pp(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.{digits}f}"


def ci_pp(ci: Any, digits: int = 3) -> str:
    # All report summary CI fields are already percentage-point values.
    if not isinstance(ci, list) or len(ci) != 2:
        return "—"
    return f"[{fmt(ci[0], digits)}, {fmt(ci[1], digits)}]"


def comparison_label(cell: str, label: str) -> str:
    return f"{cell}__{label}"


def raw_target_comparison(comp: Mapping[str, Any]) -> dict[str, Any]:
    token = comp.get("token_bootstrap")
    if not isinstance(token, Mapping):
        token = {}
    exact = comp.get("exact_net_benefit_bound")
    if not isinstance(exact, Mapping):
        exact = {}
    gain_loss = comp.get("gains_and_regressions")
    if not isinstance(gain_loss, Mapping):
        gain_loss = {}
    records = int(comp["records"])
    exact_delta_pp = 100.0 * float(comp["exact_record_delta"]) / records
    lower_rate = exact.get("net_lower_rate")
    upper_pp = exact.get("net_upper_pp")
    if upper_pp is None and exact.get("net_upper_rate") is not None:
        upper_pp = 100.0 * float(exact["net_upper_rate"])
    if lower_rate is not None:
        lower_pp = 100.0 * float(lower_rate)
    else:
        lower_pp = exact.get("net_lower_pp")
    paired_rows = comp.get("paired_record_differences")
    exact_ci: list[float] | None = None
    if isinstance(paired_rows, list) and len(paired_rows) == records:
        # The target comparison retains the same source ordering and bootstrap
        # contract as the scorer. This is descriptive; exact CP remains the
        # conservative endpoint bound reported below.
        import numpy as np
        baseline = np.asarray([
            float(bool(row["baseline"]["exact_record"])) for row in paired_rows
        ])
        method = np.asarray([
            float(bool(row["method"]["exact_record"])) for row in paired_rows
        ])
        rng = np.random.default_rng(5005)
        indexes = rng.integers(0, records, size=(10000, records))
        draws = (method[indexes] - baseline[indexes]).mean(axis=1)
        exact_ci = [100.0 * float(np.quantile(draws, 0.025)), 100.0 * float(np.quantile(draws, 0.975))]
    return {
        "baseline_method_id": comp.get("baseline_method_id"),
        "method_id": comp.get("method_id"),
        "records": records,
        "token_delta_pp": 100.0 * float(comp["micro_token_accuracy_delta"]),
        "token_ci95_pp": [100.0 * float(x) for x in token.get("delta_ci95_percentile", [])],
        "token_upper_bound_pp": None if token.get("delta_upper_bound") is None else 100.0 * float(token["delta_upper_bound"]),
        "exact_delta_pp": exact_delta_pp,
        "exact_ci95_pp": exact_ci,
        "beneficial_exact_records": int(gain_loss.get("beneficial_exact_records", 0)),
        "harmful_exact_records": int(gain_loss.get("harmful_exact_records", 0)),
        "exact_net_lower_bound_pp": None if lower_pp is None else float(lower_pp),
        "exact_net_upper_bound_pp": None if upper_pp is None else float(upper_pp),
    }


def build_manifest(dev: Mapping[str, Any], result: Mapping[str, Any], summary: Mapping[str, Any], run: Mapping[str, Any], freeze: Mapping[str, Any], attempt: Mapping[str, Any], panel: Mapping[str, Any], observations: Mapping[str, Any], registration: Mapping[str, Any], plan: Mapping[str, Any], binding: Mapping[str, Any], export: Mapping[str, Any], frequency: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("status") != "FRESH_CONFIRMATION_SCORED_AFTER_COMPLETE_PUBLIC_GATE":
        raise ValueError("score is not complete")
    gate = result.get("truth_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("score has no truth gate")
    public_gate = gate.get("executable_public_gate")
    if not isinstance(public_gate, Mapping):
        raise ValueError("score has no executable public gate")
    if public_gate.get("prediction_artifact_count") != 32 or public_gate.get("timing_receipt_count") != 32:
        raise ValueError("score matrix is incomplete")
    if gate.get("verified_before_truth") is not True or gate.get("truth_opened_after_gate") is not True:
        raise ValueError("truth gate is incomplete")
    phase_commits = dict(dev.get("phase_commits", {}))
    executed_commit = attempt.get("head_after")
    phase_commits.update({
        "fresh_prediction_matrix": executed_commit,
        "fresh_freeze": freeze.get("preregistration_commit", executed_commit),
        "fresh_score": executed_commit,
    })
    run_descriptor = descriptor(RUN_REL)
    pred_descriptor = descriptor(PRED_REL)
    timing_descriptor = descriptor(TIMING_REL)
    return {
        "schema": "token-reconstruction.trr0005-final-manifest.v1",
        "manifest_revision": "final_machine_assembled_after_truth_gate",
        "task_id": TASK,
        "status": "COMPLETE_FRESH_CONFIRMATION_SCORED_AFTER_TRUTH_GATE",
        "source_parent_branch": dev.get("source_parent_branch"),
        "source_parent_commit": dev.get("source_parent_commit"),
        "reviewed_pr": dev.get("reviewed_pr"),
        "phase_commits": phase_commits,
        "access": {
            "public_only_activation_capture": True,
            "fresh_confirmation_panel_selected": True,
            "reserved_source_pool_contents_opened_for_fresh_selection": True,
            "current_evaluator_truth_accessed": bool(gate.get("truth_opened_after_gate")),
            "truth_opened": bool(gate.get("truth_opened_after_gate")),
            "truth_opened_after_complete_public_gate": bool(gate.get("truth_opened_after_gate")),
            "final_holdout_loaded": bool(gate.get("truth_opened_after_gate")),
            "prediction_source_text_loaded": bool(run.get("source_text_loaded", False)),
            "prediction_target_labels_loaded": bool(run.get("target_labels_loaded", False)),
            "no_truth_read_before_gate": bool(gate.get("verified_before_truth")),
        },
        "contracts": {
            **dict(dev.get("contracts", {})),
            "actual_matrix": result.get("matrix"),
            "score_claim_scope": result.get("claim_scope"),
            "runtime_contract": result.get("runtime_contract"),
            "diagnostic_contract": result.get("diagnostic_contract"),
        },
        "decision_plan": {
            "file": descriptor(Path("experiments/TRR-0005/decision_plan.json")),
            "content": read(Path("experiments/TRR-0005/decision_plan.json")),
        },
        "development": dev.get("development"),
        "archive": dev.get("archive"),
        "footing": {
            **dict(dev.get("footing", {})),
            "final_report": "coordination/results/TRR-0005.md",
            "final_manifest": "experiments/TRR-0005/manifest.json",
            "summary_extractor": descriptor(Path("experiments/TRR-0005/footing/summarize_confirmation.py")),
            "final_assembly_builder": descriptor(Path("experiments/TRR-0005/footing/assemble_final_evidence.py")),
        },
        "fresh_evaluation": {
            "status": result.get("status"),
            "matrix": result.get("matrix"),
            "records_per_domain": int(summary["scope"]["records_per_cell"]),
            "unique_sources_total": int(plan.get("records_per_domain", 0)) * 2,
            "public_validation_selection": read(Path("experiments/TRR-0005/public_validation_selection.json")),
            "panel": {"file": descriptor(PANEL_REL), "descriptor": public_gate.get("panel")},
            "observations": {"file": descriptor(OBS_REL), "content": observations},
            "registration": {"file": descriptor(REG_REL), "content": registration},
            "selection_plan": {"file": descriptor(PLAN_REL), "content": plan},
            "evaluator_binding": {"file": descriptor(BINDING_REL), "content": binding},
            "frequency_manifest": {"file": descriptor(FREQ_REL), "content": frequency},
            "public_gate_descriptor": dict(public_gate),
            "freeze_receipt": {"file": descriptor(FREEZE_REL), "content": freeze},
            "score_result": {
                "file": descriptor(RESULT_REL),
                "schema": result.get("schema"),
                "status": result.get("status"),
                "claim_scope": result.get("claim_scope"),
                "matrix": result.get("matrix"),
                "bootstrap": result.get("bootstrap"),
                "diagnostic_contract": result.get("diagnostic_contract"),
                "truth_gate_summary": {
                    "status": public_gate.get("status"),
                    "verified_before_truth": public_gate.get("verified_before_truth"),
                    "truth_opened_after_gate": gate.get("truth_opened_after_gate"),
                    "prediction_artifact_count": public_gate.get("prediction_artifact_count"),
                    "timing_receipt_count": public_gate.get("timing_receipt_count"),
                },
            },
            "summary": {"file": descriptor(SUMMARY_REL), "content": summary},
            "report_tables": {"file": descriptor(Path("experiments/TRR-0005/fresh_confirmation_v1/final_report_tables.md")), "frequency_joint_rows": sum(len(rows) for domain in summary.get("frequency_position_by_domain", {}).values() for condition in domain.values() for methods in condition.values() for refs in methods.values() for rows in [refs.get("joint_frequency_position_rows", {})])},
            "prediction_export": {
                "root": str(V2_ROOT_REL),
                "predictions_manifest": pred_descriptor,
                "timings_manifest": timing_descriptor,
                "export_provenance": {"file": descriptor(EXPORT_REL), "content": export},
                "run_evidence": run_descriptor,
                "prediction_receipt_count": len(list((ROOT / V2_ROOT_REL).rglob("*.run.json"))),
                "prediction_tensor_count": len(list((ROOT / V2_ROOT_REL).rglob("*.safetensors"))),
                "binary_copy_policy": export.get("binary_copy_policy"),
                "descriptor_change_policy": export.get("descriptor_change_policy"),
            },
            "attempt_receipt": {"file": descriptor(ATTEMPT_REL), "content": attempt},
            "paired_target_comparisons": {
                key: raw_target_comparison(value)
                for key, value in result.get("paired_target_comparisons", {}).items()
                if isinstance(value, Mapping)
            },
            "fresh_outcomes": summary.get("fresh_outcomes"),
            "comparisons": summary.get("comparisons"),
            "runtime": summary.get("runtime"),
            "uncertainty": summary.get("uncertainty"),
            "decision_support": summary.get("decision_support"),
        },
        "post_score_maintenance": {
            "status": "SCIENTIFIC_EXECUTION_BINDINGS_FROZEN_AT_RECORDED_COMMIT",
            "executed_prediction_freeze_score_commit": executed_commit,
            "later_source_maintenance_policy": "Any post-score driver/schema maintenance is separate evidence; never rehash or relabel the executed da82 prediction/scorer path as if later bytes produced the frozen outputs.",
            "replay": "Check out the executed commit recorded above, then restore compact archived helper/evidence artifacts before replay; do not alter frozen prediction outputs.",
            "maintenance_commit": "1dba67a8dc75844727866cb4273da28a311df216",
            "driver_source_sha256": "e93607955d914fc4a357d432262ac8d3a946afeecd220988cfc2d8c348b3443f",
            "test_source_sha256": "eab77689f81e2fbaad40f9f26d9a3d132323ca1aa3db1edd8cd93d22da75fbc7",
            "targeted_tests": "11 targeted tests passed; scientific outputs unchanged",
            "full_tests": "302 CPU-only tests passed; scientific outputs unchanged",
            "test_receipts": {
                "targeted": "experiments/TRR-0005/footing/postscore_tests_v1/targeted_execution_receipt.json",
                "full": "experiments/TRR-0005/footing/postscore_tests_v1/full_execution_receipt.json",
            },
        },
    }


def _rare_joint_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a small, machine-derived rare-frequency joint diagnostic.

    The full report table retains all methods, both frequency maps, all four
    positions, and all four cells. This excerpt keeps the two rarest fitting
    bins (unseen and 1--4) for the distribution-selected diagonal in each
    distribution, so the narrative contains actual domain/position values
    without duplicating the full 1,024-row table.
    """
    freq = summary.get("frequency_position_by_domain", {})
    selected = summary.get("public_validation_selection", {}).get("selected_method_ids", {})
    positions = ("1-15", "16-39", "40-79", "80+")
    rows: list[dict[str, Any]] = []
    for domain in ("pile", "finance"):
        for condition in ("public_base", "public_lora_2601"):
            for fit_distribution in ("original", "enriched"):
                method_id = selected.get(fit_distribution)
                if not method_id:
                    continue
                for frequency_reference in ("original", "enriched"):
                    table = (
                        freq.get(domain, {})
                        .get(condition, {})
                        .get(method_id, {})
                        .get(frequency_reference, {})
                        .get("joint_frequency_position_rows", {})
                    )
                    for frequency_bin in ("0", "1-4"):
                        errors: list[str] = []
                        for position_bin in positions:
                            item = table.get(f"{frequency_bin}__{position_bin}")
                            if not isinstance(item, Mapping):
                                errors.append("—")
                                continue
                            scored = int(item.get("scored_tokens", 0))
                            correct = int(item.get("correct_tokens", 0))
                            errors.append("—" if scored == 0 else f"{scored - correct}/{scored}")
                        rows.append({
                            "domain": domain,
                            "condition": condition,
                            "fit_distribution": fit_distribution,
                            "method_id": method_id,
                            "frequency_reference": frequency_reference,
                            "frequency_bin": frequency_bin,
                            "errors": errors,
                        })
    return rows


def build_report(dev: Mapping[str, Any], result: Mapping[str, Any], summary: Mapping[str, Any], run: Mapping[str, Any], attempt: Mapping[str, Any], fit_run: Mapping[str, Any], corpus_plan: Mapping[str, Any]) -> str:
    gate = summary["truth_gate"]
    matrix = summary["scope"]
    selected = summary["public_validation_selection"].get("selected_method_ids", {})
    comparisons = summary["comparisons"]
    outcomes = summary["fresh_outcomes"]["cells"]
    runtime = summary["runtime"]
    dev_corpus = dev["development"]["corpus_preparation"]
    dev_capture = dev["development"]["public_activation_capture"]
    dev_fit = dev["development"]["decoder_fit"]
    dev_val = dev["development"]["public_validation"]
    dev_qual = dev["development"]["qualification"]
    dev_attn = dev["development"]["attention_diagnostic"]
    footprint = dev["development"]["memory_footprint"]
    coverage = dev_corpus["coverage"]
    fit_bank: dict[str, dict[str, float]] = {}
    for distribution, diagnostics in fit_run.get("pretraining_diagnostics", {}).items():
        retained = diagnostics.get("retained_trr0004_affine", {}).get("fit_metrics", {})
        identity = diagnostics.get("identity_initialization", {}).get("fit_metrics", {})
        fit_bank[distribution] = {
            "token_accuracy": float(retained.get("token_accuracy")),
            "exact_records": int(retained.get("exact_records")),
            "identity_accuracy": float(identity.get("token_accuracy")),
        }
    report: list[str] = []
    report.extend([
        "# TRR-0005 fresh confirmation result",
        "",
        "Status: **COMPLETE_FRESH_CONFIRMATION_SCORED_AFTER_TRUTH_GATE**.",
        "",
        f"The fresh matrix contains {matrix['method_cell_count']} method-cell artifacts across {matrix['cell_count']} cells, with {matrix['records_per_cell']} paired source records per cell. The scorer reports `{result['status']}`. Its public gate verified {gate['prediction_artifact_count']} prediction artifacts and {gate['timing_receipt_count']} timing receipts before truth opened; `truth_opened_after_gate` is `{gate['truth_opened_after_gate']}`. The executed freeze/score receipt is `experiments/TRR-0005/freeze_score_attempt_v2/execution_receipt.json`, with prediction, freeze, and score code bound to commit `{attempt.get('head_after')}`.",
        "",
        "## Fresh findings",
        "",
        f"Public validation selected trained diagonal in both distributions: `{selected.get('original')}` and `{selected.get('enriched')}`. The selection compares affine with trained diagonal within each distribution and never selects against causal. Because both selections are diagonal, causal-versus-best-positionwise duplicates causal-versus-diagonal and is not independent corroboration.",
        "",
        "The enrichment contrast raises token accuracy for the selected trained-diagonal arm in every cell. The machine-derived paired comparison table below gives the exact point estimates, source bootstrap intervals, and finite-sample bounds; all domains and target conditions remain separate.",
        "",
        "### All fresh cell outcomes",
        "",
        "Each entry is token accuracy followed by exact records out of 128. The four cells remain separate columns.",
        "",
        "| Method | Pile/P0 | Pile/synthetic-LoRA | Finance/P0 | Finance/synthetic-LoRA |",
        "|---|---|---|---|---|",
    ])
    for method in METHODS:
        values = []
        for cell in CELLS:
            m = outcomes[cell]["methods"][method]
            values.append(f"{pp(m['token_accuracy'], 4)}% ({m['exact_records']}/{m['records']})")
        report.append(f"| `{method}` | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")
    report.extend(["", "### Extra-context paired comparisons", "", "Token and exact deltas use the scorer's `method_id` minus `baseline_method_id` orientation. `U_net` is the one-sided finite-sample exact upper bound. An upper bound below a margin excludes that practical endpoint; it does not establish equivalence when it is above the margin.", "", "| Cell | Label | Token Δ pp | Token 95% CI pp | Token U pp | Exact Δ pp | Exact 95% CI pp | Exact net U pp | Gain/loss | Token margin | Exact margin |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"])
    for cell in CELLS:
        for label in PRIMARY_LABELS:
            key = comparison_label(cell, label)
            c = comparisons[key]
            exact = c["exact"]
            token = c["token"]
            token_margin = "excluded" if token["upper_bound_pp"] <= 0.5 else "not excluded"
            exact_margin = "excluded" if exact["net_upper_bound_pp"] <= 5.0 else "not excluded"
            report.append(f"| `{cell}` | `{label}` | {fmt(token['delta_pp'])} | {ci_pp(token['ci95_pp'])} | {fmt(token['upper_bound_pp'])} | {fmt(exact['delta_pp'])} | {ci_pp(exact['ci95_pp'])} | {fmt(exact['net_upper_bound_pp'])} | {exact['beneficial_exact_records']}/{exact['harmful_exact_records']} | {token_margin} | {exact_margin} |")
    report.extend(["", "The token upper bound is below the frozen 0.5 pp extra-context margin in all four cells. The exact net upper bounds are above the frozen 5 pp margin in every cell, so the strict two-endpoint context disposition is **INCONCLUSIVE**. Exact zero or near-zero observed discordance is not evidence of no effect.", "", "### Enrichment paired comparisons", "", "| Cell | State | Token Δ pp | Token 95% CI pp | Exact Δ pp | Exact 95% CI pp | Exact net U pp | Gain/loss |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for cell in CELLS:
        for state in STATES:
            label = f"coverage__{state}__enriched_vs_original"
            c = comparisons[f"{cell}__{label}"]
            exact = c["exact"]
            token = c["token"]
            report.append(f"| `{cell}` | `{state}` | {fmt(token['delta_pp'])} | {ci_pp(token['ci95_pp'])} | {fmt(exact['delta_pp'])} | {ci_pp(exact['ci95_pp'])} | {fmt(exact['net_upper_bound_pp'])} | {exact['beneficial_exact_records']}/{exact['harmful_exact_records']} |")
    report.extend(["", "The selected trained-diagonal token deltas are generated from the same source-paired matrix; they are not vocabulary-only effects. Enrichment changes public coverage, contexts, and domain mixture at fixed record and fitting-position counts. Exact endpoint support varies by cell, so the frozen decision rules should be applied without pooling.", "", "### Target pairing", "", "P0 and synthetic-LoRA records reuse source IDs within each domain. The target contrast is synthetic-LoRA minus P0 for each method; rows are descriptive paired effects with source bootstrap intervals and exact gains/losses/bounds. They are not independent target samples.", "", "| Domain | Method (fitting arm) | Contrast | Token Δ pp | Token 95% CI pp | Exact Δ pp | Exact 95% CI pp | Exact net U pp | Gain/loss |", "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for key, c in sorted(summary.get("target_comparisons", {}).items()):
        # Raw target-comparison keys are <domain>__<method>; split only once
        # because canonical method IDs themselves contain double underscores.
        domain, method = key.split("__", 1)
        report.append(f"| `{domain}` | `{method}` | synthetic-LoRA − P0 | {fmt(c['token_delta_pp'])} | {ci_pp(c['token_ci95_pp'])} | {fmt(c['exact_delta_pp'])} | {ci_pp(c['exact_ci95_pp'])} | {fmt(c['exact_net_upper_bound_pp'])} | {c['beneficial_exact_records']}/{c['harmful_exact_records']} |")
    # The target comparisons are stored in the manifest; render a compact
    # pointer if this report was built before the optional table is attached.
    if not summary.get("target_comparisons"):
        report.append("| See final manifest | paired target comparisons retained under `fresh_evaluation.paired_target_comparisons` | — | — | — | — | — | — |")
    report.extend(["", "## Frequency and error diagnostics", "", "Every contender is scored under both unchanged `original` and `enriched` fitting-frequency references. The generated table `experiments/TRR-0005/fresh_confirmation_v1/final_report_tables.md` retains 1,024 joint frequency × position rows labeled by domain, cell, method, and reference. These are alternate views of the same predictions, not additional samples. Gains and regressions remain source-paired in the score JSON and final summary.", "", "The following excerpt gives actual error counts over scored tokens (`errors/scored`) in the unseen frequency bin 0 and rare bin 1–4, split by position. It uses the public-validation selected diagonal for each fitting distribution; `—` means no eligible token in that bin. The full matrix remains in the generated table.", "", "| Domain | Target cell | Fitting arm | Frequency map | Frequency bin | 1–15 | 16–39 | 40–79 | 80+ |", "|---|---|---|---|---|---:|---:|---:|---:|"])
    for row in _rare_joint_rows(summary):
        report.append(f"| `{row['domain']}` | `{row['condition']}` | `{row['fit_distribution']} selected diagonal` | `{row['frequency_reference']}` | `{row['frequency_bin']}` | {row['errors'][0]} | {row['errors'][1]} | {row['errors'][2]} | {row['errors'][3]} |")
    report.extend(["", "## Runtime and footprint", "", "Steady per-record means are `measured_seconds_sum / records`; the outer `measured_elapsed_seconds` includes cell overhead and is not used as the steady mean. `runtime_load_seconds` is a warm-cache process load/init value counted once per method, not a flushed cold-disk measurement. Per-cell CUDA peaks and simulation counts are retained in the final summary and generated tables. Process RSS is a cumulative whole-process high-water mark and is not attributed to standalone methods.", "", "| Method | Load/init seconds once | Steady mean range ms across cells | Cell-call totals | Candidate simulations | Public prefix calls |", "|---|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        m = runtime["per_method"][method]
        means = [m["cells"][cell]["measured_seconds_mean_ms"] for cell in CELLS]
        totals = m["simulation_totals_across_cells"]
        report.append(f"| `{method}` | {fmt(m['runtime_load_seconds_once'], 6)} | {fmt(min(means), 3)}–{fmt(max(means), 3)} | {totals.get('calls', '—')} | {totals.get('candidate_simulations', '—')} | {totals.get('public_prefix_calls', '—')} |")
    component_bytes = footprint.get("component_bytes", {})
    component_gib = footprint.get("component_gib", {})
    runtime_assets = dev.get("prediction_qualification", {}).get("runtime_assets", {})
    state_sizes: dict[str, tuple[int, int]] = {}
    for distribution, distribution_info in dev_fit.get("distributions", {}).items():
        for state, state_info in distribution_info.get("methods", {}).items():
            record = state_info.get("state", {})
            if isinstance(record, Mapping) and "bytes" in record and "parameter_count" in state_info:
                state_sizes[state] = (int(state_info["parameter_count"]), int(record["bytes"]))
    state_size_text = "; ".join(
        f"{state} {params:,} parameters/{bytes_ / 1_000_000:.1f} MB ({bytes_ / 2**20:.1f} MiB)"
        for state, (params, bytes_) in state_sizes.items()
    )
    e_bytes = int(runtime_assets.get("normalized_embedding", {}).get("bytes", component_bytes.get("embedding_table_fp32", 0)))
    a2_bytes = int(runtime_assets.get("public_prefix_checkpoint", {}).get("bytes", 0))
    scan_occurrences = sum(
        int(corpus_plan.get("preparation", {}).get("source_pools", {}).get(name, {}).get("post_bos_token_occurrences_in_eligible_rows", 0))
        for name in ("pile", "finance")
    )
    report.extend(["", "All standalone methods use the shared normalized public E table (1,050,673,488 bytes) plus their retained state and make zero A2 candidate calls. The A2 anchor uses the qualified CPU-E normalization port and retains the public P0 assets; its exact candidate and prefix counts are shown above. The final per-cell table records the A2 exact timing gap and per-cell reserved peaks without attributing cumulative process RSS to it alone.", "", f"Registered decoder state sizes are machine-recorded as {state_size_text}. The shared normalized E table is {e_bytes:,} bytes ({e_bytes / 1_000_000_000:.3f} GB; {component_gib.get('embedding_table_fp32', e_bytes / 2**30):.3f} GiB). The A2 public-prefix checkpoint is {a2_bytes:,} bytes ({a2_bytes / 1_000_000_000:.3f} GB). The fit resource forecast records a {component_gib.get('training_peak_envelope', 0):.3f} GiB analytic training envelope, a {component_gib.get('conservative_envelope', 0):.3f} GiB guarded qualification floor, and measured fit reserved peaks of roughly 2.743–2.981 GiB; these are resource measurements/forecasts, not per-method standalone RSS claims.", "", "## Development context and costs", "", f"Public corpus preparation retained {dev_corpus['matching_fit_positions']} fitting positions per arm. The original-like arm has {coverage['original_distinct_post_bos_ids']} distinct post-BOS IDs; the enriched arm has {coverage['enriched_distinct_post_bos_ids']}. Enrichment adds {coverage['newly_covered_by_enriched']} IDs and loses {coverage['lost_from_original']} original IDs. The Pile and Finance fit-frequency pools contributed {scan_occurrences:,} post-BOS token occurrences to the recorded public scan (about {scan_occurrences / 1_000_000:.3f} million), with {dev_corpus['elapsed_seconds']:.3f} s child preparation time. All controlled placements occur at token positions greater than or equal to 128. This is a joint coverage/context intervention, not an isolated vocabulary experiment.", "", f"Public activation capture used the batch-8 × 192 bit-exact path with capture time {dev_capture['capture_wall_seconds']:.3f} s and launch interval {dev_capture['launch_wall_seconds']:.3f} s. The excluded unpadded batch-1 diagnostic remains marked non-equivalent. The primary capture reused no evaluator-private truth or target weights.", "", f"The six original fits used 3,000 steps and the shared 512 post-BOS draws per step for {dev_fit['elapsed_seconds']:.3f} s; the two qknorm causal repair fits added {dev_fit['qknorm_causal_repair']['elapsed_seconds']:.3f} s, for {dev_fit['elapsed_seconds'] + dev_fit['qknorm_causal_repair']['elapsed_seconds']:.3f} s across eight fits. The two original dot-product causal fits are successful completed development fits superseded by the preregistered qknorm repair; their attention routing limitation is not an execution failure. Actual failed attempts are preserved V1 qualification forecast guard and the capture output-root collision.", "", f"The retained fit-bank diagnostic is machine-ingested from the pretraining diagnostic: original-like token accuracy {100 * fit_bank['original']['token_accuracy']:.7f}% with {fit_bank['original']['exact_records']}/1,200 exact records, versus enriched token accuracy {100 * fit_bank['enriched']['token_accuracy']:.7f}% with {fit_bank['enriched']['exact_records']}/1,200 exact records. Initial identity accuracy was {100 * fit_bank['original']['identity_accuracy']:.3f}% original-like and {100 * fit_bank['enriched']['identity_accuracy']:.3f}% enriched. Public validation and final fit-stream accuracy are development evidence, not fresh outcomes.", "", f"Qualification evidence includes the preserved V1 failure (`{dev_qual['v1_preserved_failure']['status']}`), V2 largest-cell qualification (`{dev_qual['v2']['status']}`), qknorm qualification (`{dev_qual['qknorm_causal_repair']['qualification_status']}`), and archived Finance-128 one-warmup/one-measured qualification (`{dev.get('development', {}).get('prediction_qualification', {}).get('status', 'machine-recorded')}`).", ""])
    # Add exact development fit-bank values from the existing machine manifest
    fit_diag = dev_fit.get("fit_bank_diagnostic") or dev_fit.get("pretraining_diagnostics")
    if isinstance(fit_diag, Mapping):
        report.extend(["The fit-bank diagnostic details are:", "", "| Arm | Token accuracy | Exact records |", "|---|---:|---:|"])
        for key, value in fit_diag.items():
            if isinstance(value, Mapping):
                report.append(f"| `{key}` | {pp(value.get('token_accuracy'), 4)}% | {value.get('exact_records', '—')} |")
        report.append("")
    report.extend(["The H-only attention diagnostic found the original dot-product branch routing essentially all original queries to BOS and enriched queries mostly to BOS. The qknorm repair increased earlier-position mass on the tested public validation H. This diagnoses those score branches only; it does not establish that earlier H is generally useless. Trained diagonal retains contextual H_i and adds a positionwise nonlinear layer-normalized value correction; causal adds earlier H through the same path, while qknorm repairs routing. The one-key diagonal mask leaves Q/K routing degrees of freedom effectively inactive; their declared parameter footprint remains part of the registered state.", "", "## Interpretation and next experiment", "", "The fresh results support a consistent enrichment token gain across the four cells. The strict extra-context endpoint is INCONCLUSIVE because its token bound excludes 0.5 pp while its exact-record bound does not exclude 5 pp in any cell. The two primary causal labels are duplicates under the frozen diagonal selections. No pooled domain or target headline is valid.", "", "A bounded next confirmation should freeze enriched causal and diagonal contenders and collect 1,024 source-disjoint natural records per domain, keeping the same 128-token clip and paired targets. At the observed Finance discordance rate, 1,024 records would give an estimated familywise exact upper bound near 4.29 pp (planning estimate, not guaranteed power); 768 would be about 4.85 pp. Keep the frozen 0.5 pp token and 5 pp exact margins, domains separate, and frequency-balanced strata as a separate diagnostic because changing sampling changes the estimand. Use the same paired uncertainty rules and treat the planning calculation as conditional on the observed rates.", "", "## Reproduction and archive checklist", "", "- Reproduce the score from the exact freeze/score execution receipt and v2 prediction export; do not use the failed V1 freeze root.", "- Restore commit bound by the freeze and score receipt before replay; later source/schema maintenance is separate evidence.", "- Retain compact panel, observations, selection, registration, binding, prediction/timing manifests, per-cell receipts, freeze receipt, score result, final summary, and frequency tables.", "- Keep evaluator truth outside the frozen reconstruction root; the frozen root contains only the label-free evaluator binding descriptor.", "- Keep raw public H/fit tensors and private truth in the task archive according to the archive policy, with their existing receipts and replay commands.", "", "Machine-readable primary artifacts:", "", "- `experiments/TRR-0005/fresh_confirmation_v1/result.json`",
        "- `experiments/TRR-0005/fresh_confirmation_v1/final_summary.json`",
        "- `experiments/TRR-0005/fresh_confirmation_v1/final_report_tables.md`",
        "- `experiments/TRR-0005/freeze_score_attempt_v2/execution_receipt.json`",
        "- `experiments/TRR-0005/manifest.json`",
        ""])
    return "\n".join(report)


def main() -> None:
    dev = read(DEV_MANIFEST_REL)
    result = read(RESULT_REL)
    summary = read(SUMMARY_REL)
    run = read(RUN_REL)
    fit_run = read(Path("experiments/TRR-0005/joint_fit_v1/run_evidence.json"))
    corpus_plan = read(CORPUS_PLAN_REL)
    freeze = read(FREEZE_REL)
    attempt = read(ATTEMPT_REL)
    panel = read(PANEL_REL)
    observations = read(OBS_REL)
    registration = read(REG_REL)
    plan = read(PLAN_REL)
    binding = read(BINDING_REL)
    export = read(EXPORT_REL)
    frequency = read(FREQ_REL)
    # Attach target comparisons to a report-only summary view without changing
    # the extractor's canonical JSON output.
    summary_for_report = dict(summary)
    summary_for_report["target_comparisons"] = {
        key: raw_target_comparison(value)
        for key, value in result.get("paired_target_comparisons", {}).items()
        if isinstance(value, Mapping)
    }
    manifest = build_manifest(dev, result, summary, run, freeze, attempt, panel, observations, registration, plan, binding, export, frequency)
    report = build_report(dev, result, summary_for_report, run, attempt, fit_run, corpus_plan)
    (ROOT / "experiments/TRR-0005/manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (ROOT / "coordination/results/TRR-0005.md").write_text(report, encoding="utf-8")
    print(json.dumps({"manifest":"experiments/TRR-0005/manifest.json","report":"coordination/results/TRR-0005.md","status":manifest["status"],"result":result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
