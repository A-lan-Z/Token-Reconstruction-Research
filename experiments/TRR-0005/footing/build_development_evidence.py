#!/usr/bin/env python3
"""Build TRR-0005's development-only report and manifest from compact receipts.

This builder deliberately reads JSON metadata and learning-curve summaries only.
It does not open safetensors, model weights, source pools, or evaluator truth.
The generated report remains IN_PROGRESS until the frozen fresh-evaluation matrix
and truth-gated score are available.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "coordination/results/TRR-0005.md"
MANIFEST = ROOT / "experiments/TRR-0005/manifest.json"
DOC = ROOT / "experiments/TRR-0005/footing/development_reproduction.md"


def read(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text())


def relpath(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return str(Path(value).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return value


def compact_ref(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, val in value.items():
        if key == "path":
            out[key] = relpath(val)
        elif key in {"bytes", "sha256", "state_sha256", "dtype", "shape", "tensor_key", "points", "selected_step"}:
            out[key] = val
    return out


def pct(value: float, digits: int = 4) -> str:
    return f"{100.0 * value:.{digits}f}%"


def pp(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f} pp"


def sec(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} s"


def gib(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} GiB"


def bytes_gib(value: int) -> float:
    return value / float(1024**3)


def curve_summary(
    fit: dict[str, Any],
    distribution: str,
    method: str,
    fit_root: str = "joint_fit_v1",
) -> dict[str, Any]:
    curve_path = f"experiments/TRR-0005/{fit_root}/{distribution}/{method}/learning_curve.json"
    curve_file = read(curve_path)
    rows = curve_file["curve"]
    metric = curve_file["selection_metric"]
    metric_key = metric.removeprefix("validation_")
    selected = min(rows, key=lambda row: (-row["validation"][metric_key], row["step"]))
    step0 = rows[0]
    fit_method = fit["distributions"][distribution]["methods"][method]
    return {
        "distribution": distribution,
        "method_id": method,
        "canonical_method_id": curve_file["canonical_method_id"],
        "selection_metric": metric,
        "selection_rule": curve_file["selection_rule"],
        "curve_points": len(rows),
        "selected_step": selected["step"],
        "best_style_balanced_token_accuracy": selected["validation"][metric_key],
        "best_token_accuracy": selected["validation"]["token_accuracy"],
        "step0_style_balanced_token_accuracy": step0["validation"][metric_key],
        "step0_token_accuracy": step0["validation"]["token_accuracy"],
        "style_balanced_headroom_pp": 100.0 * (
            selected["validation"][metric_key] - step0["validation"][metric_key]
        ),
        "validation_token_rows": selected["validation"]["token_rows"],
        "validation_exact_records": selected["validation"]["exact_records"],
        "arm_wall_seconds": fit_method["arm_wall_seconds"],
        "optimization_update_seconds": fit_method["optimization_update_seconds"],
        "selection_validation_seconds": fit_method["selection_validation_seconds"],
        "final_fit_diagnostic_seconds": fit_method["final_fit_diagnostic_seconds"],
        "parameter_count": fit_method["parameter_count"],
        "curve": compact_ref(fit_method["curve"]),
        "state": compact_ref(fit_method["state"]),
        "peak_memory": fit_method["peak_memory"],
    }


def command_for(rel: str, key: str) -> str:
    obj = read(rel)
    value: Any = obj
    for part in key.split("."):
        value = value[part]
    return shlex.join(str(x) for x in value)


def build() -> tuple[dict[str, Any], str, str]:
    setup = read("experiments/TRR-0005/footing/setup_preflight.json")
    corpus_design = read("experiments/TRR-0005/corpus_design.json")
    corpus = read("experiments/TRR-0005/corpus/corpus_plan.json")
    corpus_run = read("experiments/TRR-0005/corpus_run/run_receipt.json")
    capture = read("experiments/TRR-0005/public_activation_v1/capture_manifest_receipt.json")
    launch = read("experiments/TRR-0005/public_activation_v1/launch.json")
    launch_preflight = read("experiments/TRR-0005/public_activation_v1/launch_preflight.json")
    fit_design = read("experiments/TRR-0005/joint_fit_design.json")
    fit = read("experiments/TRR-0005/joint_fit_v1/run_evidence.json")
    fit_memory = read("experiments/TRR-0005/joint_fit_v1/memory_preflight.json")
    qknorm_fit = read("experiments/TRR-0005/joint_fit_qknorm_v1/run_evidence.json")
    qual_v1 = read("experiments/TRR-0005/joint_qualification_v1/failure.json")
    qual_v2 = read("experiments/TRR-0005/joint_qualification_v2/qualification_evidence.json")
    qknorm_qual = read("experiments/TRR-0005/joint_qualification_qknorm_v1/qualification_evidence.json")
    qknorm_amendment = read("experiments/TRR-0005/qk_score_repair_amendment_v1.json")
    prediction_qualification = read("experiments/TRR-0005/prediction_qualification_v1/qualification.json")
    attention = read("experiments/TRR-0005/attention_diagnostic.json")
    attention_exec = read("experiments/TRR-0005/attention_diagnostic_execution.json")
    attention_qknorm = read("experiments/TRR-0005/attention_diagnostic_qknorm_v1.json")
    attention_qknorm_exec = read("experiments/TRR-0005/attention_diagnostic_qknorm_v1_execution.json")
    decision = read("experiments/TRR-0005/decision_plan.json")
    ledger = read("experiments/TRR-0005/footing/preplanned_ledger.json")
    adapter_status = read("experiments/TRR-0005/footing/adapter_status.json")

    distributions = ["original", "enriched"]
    methods = [
        "joint_full_affine",
        "affine_causal_h_attention128",
        "affine_trained_diagonal_attention128",
    ]
    curves = {
        distribution: {
            method: curve_summary(fit, distribution, method) for method in methods
        }
        for distribution in distributions
    }
    qknorm_curves = {
        distribution: {
            "affine_causal_h_attention128": curve_summary(
                qknorm_fit,
                distribution,
                "affine_causal_h_attention128",
                fit_root="joint_fit_qknorm_v1",
            )
        }
        for distribution in distributions
    }
    positionwise = {}
    for distribution in distributions:
        contenders = [
            qknorm_curves[distribution]["affine_causal_h_attention128"],
            curves[distribution]["affine_trained_diagonal_attention128"],
        ]
        positionwise[distribution] = min(
            contenders,
            key=lambda row: (-row["best_style_balanced_token_accuracy"], row["selected_step"]),
        )

    cov = capture["coverage_diagnostics"]
    coverage_arm = corpus["arms"]["coverage_mix_v1"]
    original_arm = corpus["arms"]["original_like_alpaca_v1"]
    corpus_exec = corpus["execution"]
    cap_exec = capture["execution"]
    cap_out = capture["outputs"]
    qualification = capture["capture"]["qualification"]
    setup_gpu = setup["resource_snapshot"]["gpu"]
    setup_host = setup["resource_snapshot"]["host_memory"]

    artifact_paths = {
        "setup_preflight": "experiments/TRR-0005/footing/setup_preflight.json",
        "adapter_status": "experiments/TRR-0005/footing/adapter_status.json",
        "preplanned_ledger": "experiments/TRR-0005/footing/preplanned_ledger.json",
        "runnable_interface": "experiments/TRR-0005/footing/runnable_interface.md",
        "corpus_design": "experiments/TRR-0005/corpus_design.json",
        "corpus_plan": "experiments/TRR-0005/corpus/corpus_plan.json",
        "corpus_run_receipt": "experiments/TRR-0005/corpus_run/run_receipt.json",
        "corpus_preflight": "experiments/TRR-0005/corpus_run/preflight.json",
        "capture_receipt": "experiments/TRR-0005/public_activation_v1/capture_manifest_receipt.json",
        "capture_raw_receipt": "experiments/TRR-0005/public_activation_v1/capture_manifest_receipt.raw.json",
        "capture_launch": "experiments/TRR-0005/public_activation_v1/launch.json",
        "capture_launch_preflight": "experiments/TRR-0005/public_activation_v1/launch_preflight.json",
        "original_public_manifest": "experiments/TRR-0005/public_activation_v1/original_manifest.json",
        "enriched_public_manifest": "experiments/TRR-0005/public_activation_v1/enriched_manifest.json",
        "original_fit_records": "experiments/TRR-0005/public_activation_v1/original_fit_records.json",
        "enriched_fit_records": "experiments/TRR-0005/public_activation_v1/enriched_fit_records.json",
        "joint_fit_design": "experiments/TRR-0005/joint_fit_design.json",
        "joint_fit_interface": "experiments/TRR-0005/joint_fit_interface.md",
        "joint_fit_run_evidence": "experiments/TRR-0005/joint_fit_v1/run_evidence.json",
        "joint_fit_memory_preflight": "experiments/TRR-0005/joint_fit_v1/memory_preflight.json",
        "qknorm_fit_run_evidence": "experiments/TRR-0005/joint_fit_qknorm_v1/run_evidence.json",
        "prediction_qualification": "experiments/TRR-0005/prediction_qualification_v1/qualification.json",
        "qualification_v1_failure": "experiments/TRR-0005/joint_qualification_v1/failure.json",
        "qualification_v2": "experiments/TRR-0005/joint_qualification_v2/qualification_evidence.json",
        "qknorm_amendment": "experiments/TRR-0005/qk_score_repair_amendment_v1.json",
        "qknorm_qualification": "experiments/TRR-0005/joint_qualification_qknorm_v1/qualification_evidence.json",
        "qknorm_attention_result": "experiments/TRR-0005/attention_diagnostic_qknorm_v1.json",
        "qknorm_attention_execution": "experiments/TRR-0005/attention_diagnostic_qknorm_v1_execution.json",
        "attention_result": "experiments/TRR-0005/attention_diagnostic.json",
        "attention_execution": "experiments/TRR-0005/attention_diagnostic_execution.json",
        "decision_plan": "experiments/TRR-0005/decision_plan.json",
        "report": "coordination/results/TRR-0005.md",
        "manifest": "experiments/TRR-0005/manifest.json",
        "reproduction_doc": "experiments/TRR-0005/footing/development_reproduction.md",
        "report_builder": "experiments/TRR-0005/footing/build_development_evidence.py",
    }

    output_artifacts = {
        key: compact_ref(cap_out[key])
        for key in ("original_manifest", "original_records", "enriched_manifest", "enriched_records", "enriched_artifact")
    }
    fit_artifacts = {
        distribution: {
            method: {
                "curve": row["curve"],
                "state": row["state"],
                "selected_step": row["selected_step"],
            }
            for method, row in curves[distribution].items()
        }
        for distribution in distributions
    }
    qknorm_fit_artifacts = {
        distribution: {
            method: {
                "curve": row["curve"],
                "state": row["state"],
                "selected_step": row["selected_step"],
            }
            for method, row in qknorm_curves[distribution].items()
        }
        for distribution in distributions
    }

    phase_commits = {
        "reviewed_source_parent": setup["source_parent_commit"],
        "corpus_preparation": corpus_exec["git_commit"],
        "public_activation_capture": cap_exec["git_commit"],
        "public_activation_launch": launch["source_commit"],
        "joint_qualification_v1_preserved_failure": qual_v1["git_commit"],
        "joint_qualification_v2": qual_v2["git_commit"],
        "joint_fit": fit["git_commit"],
        "qknorm_qualification": qknorm_qual["git_commit"],
        "qknorm_fit": qknorm_fit["git_commit"],
        "prediction_qualification": prediction_qualification["git_commit"],
        "attention_diagnostic": attention_exec["git_commit"],
        "qknorm_attention_diagnostic": attention_qknorm_exec["git_commit"],
    }

    manifest = {
        "schema": "token-reconstruction.trr0005-footing-manifest.v1",
        "manifest_revision": 2,
        "task_id": "TRR-0005",
        "status": "IN_PROGRESS_PUBLIC_DEVELOPMENT_FRESH_EVALUATION_PENDING",
        "source_parent_branch": setup["source_parent_branch"],
        "source_parent_commit": setup["source_parent_commit"],
        "reviewed_pr": setup["reviewed_pr"],
        "contracts": {
            "method_count": ledger["methods"]["anchor_count"] + ledger["methods"]["new_state_count"],
            "cell_count": ledger["holdout"]["cells"],
            "records_per_domain": ledger["holdout"]["records_per_domain"],
            "fresh_prediction_shape": [ledger["holdout"]["records_per_domain"], ledger["holdout"]["sequence_tokens"]],
            "fit_geometry": ledger["training"]["fit_geometry"],
            "frequency_references": ["original", "enriched"],
            "selection": "frozen public-validation affine-versus-diagonal per distribution",
            "timing": {
                "warmup_calls_per_record": ledger["inference"]["warmup_runs_per_record"],
                "measured_calls_per_record": ledger["inference"]["measured_runs_per_record"],
            },
            "exact_uncertainty": decision["uncertainty"]["exact_upper_bound"],
        },
        "corpus_preparation": {
            "commit": corpus_exec["git_commit"],
            "cpu_seconds": corpus_exec["elapsed_seconds"],
            "original_distinct_post_bos_ids": cov["original_like_distinct_post_bos"],
            "enriched_distinct_post_bos_ids": cov["enriched_distinct_post_bos"],
            "matching_fit_positions": corpus["design"]["post_bos_positions"],
            "fresh_holdout_selected": False,
        },
        "validation": {
            "task_focused_tests": "38 passed (recorded before the current exclusive window)",
            "scorer_cli_help_and_source_compile": "pass",
            "negative_truth_sentinel_cases": adapter_status["validation"]["negative_truth_sentinel_cases"] + ["wrong_sidecar", "row_order", "unfrozen_descriptor"],
        },
        "heavy_execution_started": True,
        "model_or_external_large_asset_loaded": True,
        "holdout_selected": False,
        "truth_opened": False,
        "phase_commits": phase_commits,
        "access": {
            "public_only_corpus_preparation": corpus["preparation"]["private_truth_accessed"] is False,
            "public_only_activation_capture": capture["access_contract"]["evaluator_private_truth_accessed"] is False,
            "current_evaluator_truth_accessed": fit["current_evaluator_truth_accessed"],
            "final_holdout_loaded": fit["final_holdout_loaded"],
            "fresh_holdout_selected": False,
            "reserved_source_pool_contents_opened": ledger["reserved_source_pools"]["contents_opened"],
            "truth_opened": False,
        },
        "development": {
            "corpus_preparation": {
                "status": corpus["status"],
                "execution_status": corpus_run["status"],
                "elapsed_seconds": corpus_exec["elapsed_seconds"],
                "receipt_elapsed_seconds": corpus_run["elapsed_seconds"],
                "max_rss_bytes": corpus_exec["resource_usage"]["max_rss_bytes"],
                "user_cpu_seconds": corpus_exec["resource_usage"]["user_cpu_seconds"],
                "system_cpu_seconds": corpus_exec["resource_usage"]["system_cpu_seconds"],
                "fit_records_per_arm": {name: len(arm["records"]) for name, arm in corpus["arms"].items()},
                "matching_fit_positions": corpus["design"]["post_bos_positions"],
                "source_pool_contents_scanned": False,
                "phase_timings_seconds": corpus_exec["phase_timings_seconds"],
                "coverage": {
                    "original_distinct_post_bos_ids": cov["original_like_distinct_post_bos"],
                    "enriched_distinct_post_bos_ids": cov["enriched_distinct_post_bos"],
                    "overlap_distinct_post_bos_ids": cov["original_enriched_overlap_distinct_post_bos"],
                    "newly_covered_by_enriched": cov["newly_covered_by_enriched_distinct_post_bos"],
                    "lost_from_original": cov["lost_from_original_distinct_post_bos"],
                    "controlled_ids": coverage_arm["controlled"]["controlled_ids_used"],
                    "controlled_records": coverage_arm["controlled"]["controlled_record_count"],
                    "controlled_replacement_occurrences": coverage_arm["controlled"]["controlled_replacement_occurrences"],
                    "minimum_distinct_ids": coverage_arm["coverage_contrast"]["minimum_distinct_token_ids"],
                    "minimum_legacy_absent_controlled_ids": coverage_arm["coverage_contrast"]["minimum_legacy_absent_controlled_ids"],
                    "coverage_contrast_status": coverage_arm["coverage_contrast"]["status"],
                },
                "arm_composition": {
                    "original_like_alpaca_v1": original_arm["domain_length"],
                    "coverage_mix_v1": coverage_arm["domain_length"],
                },
            },
            "public_activation_capture": {
                "status": capture["status"],
                "capture_wall_seconds": capture["capture"]["wall_seconds"],
                "launch_wall_seconds": launch["wall_seconds"],
                "geometry": capture["geometry"],
                "fit_post_bos_positions": capture["geometry"]["post_bos_fit"],
                "validation_post_bos_positions": capture["geometry"]["post_bos_validation"],
                "resource_preflight": capture["capture"]["resource_preflight"],
                "peak_memory": capture["capture"]["peak_memory"],
                "primary_geometry_qualification": qualification["primary_geometry"],
                "excluded_unpadded_batch1_diagnostic": qualification["unpadded_batch1_diagnostic"],
                "output_artifacts": output_artifacts,
                "access_contract": capture["access_contract"],
            },
            "decoder_fit": {
                "status": fit["status"],
                "elapsed_seconds": fit["elapsed_seconds"],
                "started_utc": fit["started_utc"],
                "ended_utc": fit["ended_utc"],
                "settings": fit["fixed_settings"],
                "distributions": {
                    distribution: {
                        "contract_distribution_id": fit["distributions"][distribution]["contract_distribution_id"],
                        "fit_record_count": fit["distributions"][distribution]["fit_record_count"],
                        "fit_post_bos_positions": fit["distributions"][distribution]["fit_post_bos_positions"],
                        "validation_record_count": fit["distributions"][distribution]["validation_record_count"],
                        "validation_post_bos_positions": fit["distributions"][distribution]["validation_post_bos_positions"],
                        "load_seconds": fit["distributions"][distribution]["load_seconds"],
                        "preparation_timing": fit["distributions"][distribution]["preparation_timing"],
                        "methods": curves[distribution],
                    }
                    for distribution in distributions
                },
                "fit_artifacts": fit_artifacts,
                "qknorm_causal_repair": {
                    "status": qknorm_fit["status"],
                    "elapsed_seconds": qknorm_fit["elapsed_seconds"],
                    "started_utc": qknorm_fit["started_utc"],
                    "ended_utc": qknorm_fit["ended_utc"],
                    "settings": qknorm_fit["fixed_settings"],
                    "distributions": {
                        distribution: {
                            "contract_distribution_id": qknorm_fit["distributions"][distribution]["contract_distribution_id"],
                            "fit_record_count": qknorm_fit["distributions"][distribution]["fit_record_count"],
                            "fit_post_bos_positions": qknorm_fit["distributions"][distribution]["fit_post_bos_positions"],
                            "load_seconds": qknorm_fit["distributions"][distribution]["load_seconds"],
                            "preparation_timing": qknorm_fit["distributions"][distribution]["preparation_timing"],
                            "methods": qknorm_curves[distribution],
                        }
                        for distribution in distributions
                    },
                    "fit_artifacts": qknorm_fit_artifacts,
                },
            },
            "public_validation": {
                "panel_records": fit["distributions"]["original"]["validation_record_count"],
                "post_bos_rows": fit["distributions"]["original"]["validation_post_bos_positions"],
                "selection_metric": "validation_style_balanced_token_accuracy",
                "step0": {
                    "style_balanced_token_accuracy": curves["original"][methods[0]]["step0_style_balanced_token_accuracy"],
                    "token_accuracy": curves["original"][methods[0]]["step0_token_accuracy"],
                },
                "six_fit_curves": curves,
                "qknorm_causal_curves": qknorm_curves,
                "current_positionwise_leaders": {
                    distribution: {
                        "method_id": row["method_id"],
                        "selected_step": row["selected_step"],
                        "best_style_balanced_token_accuracy": row["best_style_balanced_token_accuracy"],
                    }
                    for distribution, row in positionwise.items()
                },
                "interpretation": "Development-only reused public panel; trained diagonal remains the current best positionwise arm after comparing against the completed qknorm causal repair, but the method choice is not frozen until the two-stage freeze and panel registration complete. The two distribution selections are not independent evidence.",
            },
            "qualification": {
                "v1_preserved_failure": {
                    "status": qual_v1["status"],
                    "error": qual_v1["error"],
                    "elapsed_seconds": qual_v1["elapsed_seconds"],
                    "git_commit": qual_v1["git_commit"],
                },
                "v2": {
                    "status": qual_v2["status"],
                    "total_wall_seconds": qual_v2["total_qualification_wall_seconds"],
                    "git_commit": qual_v2["git_commit"],
                    "distributions": {
                        distribution: {
                            "status": qual_v2["distributions"][distribution]["status"],
                            "total_wall_seconds": qual_v2["distributions"][distribution]["total_qualification_wall_seconds"],
                            "measured_device_peak_bytes": qual_v2["distributions"][distribution]["forecast_comparison"]["measured_device_peak_bytes"],
                            "measured_device_peak_gib": qual_v2["distributions"][distribution]["forecast_comparison"]["measured_device_peak_gib"],
                            "host_rss_bytes": qual_v2["distributions"][distribution]["peak_memory"]["process_max_rss_bytes"],
                        }
                        for distribution in distributions
                    },
                },
                "qknorm_causal_repair": {
                    "amendment_status": qknorm_amendment["status"],
                    "qualification_status": qknorm_qual["status"],
                    "total_wall_seconds": qknorm_qual["total_qualification_wall_seconds"],
                    "git_commit": qknorm_qual["git_commit"],
                    "score_mode": qknorm_amendment["repair"]["attention_score_mode"],
                    "full_fit_status": qknorm_fit["status"],
                    "fit_elapsed_seconds": qknorm_fit["elapsed_seconds"],
                    "fit_distributions": {
                        distribution: {
                            "selected_step": qknorm_curves[distribution]["affine_causal_h_attention128"]["selected_step"],
                            "best_style_balanced_token_accuracy": qknorm_curves[distribution]["affine_causal_h_attention128"]["best_style_balanced_token_accuracy"],
                            "state": qknorm_curves[distribution]["affine_causal_h_attention128"]["state"],
                        }
                        for distribution in distributions
                    },
                    "qualification_distributions": {
                        distribution: {
                            "status": qknorm_qual["distributions"][distribution]["status"],
                            "total_wall_seconds": qknorm_qual["distributions"][distribution]["total_qualification_wall_seconds"],
                            "measured_device_peak_bytes": qknorm_qual["distributions"][distribution]["forecast_comparison"]["measured_device_peak_bytes"],
                            "measured_device_peak_gib": qknorm_qual["distributions"][distribution]["forecast_comparison"]["measured_device_peak_gib"],
                        }
                        for distribution in distributions
                    },
                },
            },
            "prediction_qualification": {
                "status": prediction_qualification["status"],
                "git_commit": prediction_qualification["git_commit"],
                "elapsed_seconds": prediction_qualification["elapsed_seconds"],
                "method_count": prediction_qualification["method_count"],
                "joint_method_count": prediction_qualification["joint_method_count"],
                "record": prediction_qualification["record"],
                "warmup_runs_per_record": prediction_qualification["warmup_runs_per_record"],
                "measured_runs_per_record": prediction_qualification["measured_runs_per_record"],
                "warmup_measured_ids_exact": prediction_qualification["warmup_measured_ids_exact"],
                "archived_a1_a2_ids_exact": prediction_qualification["archived_a1_a2_ids_exact"],
                "candidate_arrays_persisted": prediction_qualification["candidate_arrays_persisted"],
                "fresh_panel_loaded": prediction_qualification["fresh_panel_loaded"],
                "target_labels_loaded": prediction_qualification["target_labels_loaded"],
                "truth_opened": prediction_qualification["truth_opened"],
                "future_activation_reads": prediction_qualification["future_activation_reads"],
                "driver": compact_ref(prediction_qualification["driver"]),
                "runtime_assets": {key: relpath(value) for key, value in prediction_qualification["runtime_assets"].items()},
                "methods": {
                    method_id: {
                        "prediction_shape": method["prediction_shape"],
                        "prediction_sha256": method["prediction_sha256"],
                        "timing": {
                            "warmup_seconds": method["timing"]["warmup_seconds"],
                            "measured_seconds": method["timing"]["measured_seconds"],
                            "timed_interval_total_seconds": method["timing"]["timed_interval_total_seconds"],
                            "warmup_output_exact_match_measured": method["timing"]["warmup_output_exact_match_measured"],
                            "measured_output_selected": method["timing"]["measured_output_selected"],
                        },
                        "peak_memory": method["peak_memory"],
                    }
                    for method_id, method in prediction_qualification["methods"].items()
                },
            },
            "attention_diagnostic": {
                "status": attention["status"],
                "execution_status": attention_exec["status"],
                "git_commit": attention_exec["git_commit"],
                "elapsed_wall_seconds": attention_exec["elapsed_wall_seconds"],
                "child_max_rss_kib": attention_exec["child_max_rss_kib"],
                "truth_accessed": attention_exec["truth_accessed"],
                "state_mutated": attention["state_mutated"],
                "states": {
                    distribution: {
                        "state_file": compact_ref(attention["states"][distribution]["state_file"]),
                        "overall": attention["states"][distribution]["summary"]["overall"],
                    }
                    for distribution in distributions
                },
                "interpretation": "The tested original causal state routes essentially all mass to BOS; enriched retains 3.055% average mass on earlier non-BOS positions. This diagnoses the tested dot-product causal branch only and does not establish that earlier H is generally useless; qknorm repair states remain separate contenders.",
                "qknorm_repair": {
                    "status": attention_qknorm["status"],
                    "execution_status": attention_qknorm_exec["status"],
                    "git_commit": attention_qknorm_exec["git_commit"],
                    "elapsed_wall_seconds": attention_qknorm_exec["launcher_wall_seconds"],
                    "truth_accessed": attention_qknorm_exec["truth_accessed"],
                    "state_mutated": attention_qknorm["state_mutated"],
                    "states": {
                        distribution: {
                            "state_file": compact_ref(attention_qknorm["states"][distribution]["state_file"]),
                            "overall": attention_qknorm["states"][distribution]["summary"]["overall"],
                        }
                        for distribution in distributions
                    },
                },
            },
            "memory_footprint": {
                "geometry": fit_memory["geometry"],
                "component_bytes": fit_memory["bytes"],
                "component_gib": fit_memory["gib"],
                "forecast_basis": fit_memory["forecast_basis"],
                "resource_policy": fit_memory["resource_policy"],
                "measured_fit_peaks": {
                    distribution: {
                        method: {
                            "cuda_peak_allocated_bytes": curves[distribution][method]["peak_memory"]["cuda_peak_allocated_bytes"],
                            "cuda_peak_reserved_bytes": curves[distribution][method]["peak_memory"]["cuda_peak_reserved_bytes"],
                            "process_max_rss_bytes": curves[distribution][method]["peak_memory"]["process_max_rss_bytes"],
                        }
                        for method in methods
                    }
                    for distribution in distributions
                },
                "qknorm_fit_peaks": {
                    distribution: {
                        "cuda_peak_allocated_bytes": qknorm_curves[distribution]["affine_causal_h_attention128"]["peak_memory"]["cuda_peak_allocated_bytes"],
                        "cuda_peak_reserved_bytes": qknorm_curves[distribution]["affine_causal_h_attention128"]["peak_memory"]["cuda_peak_reserved_bytes"],
                        "process_max_rss_bytes": qknorm_curves[distribution]["affine_causal_h_attention128"]["peak_memory"]["process_max_rss_bytes"],
                    }
                    for distribution in distributions
                },
                "setup_snapshot": {
                    "gpu_name": setup_gpu["name"],
                    "memory_free_mib": setup_gpu["memory_free_mib"],
                    "memory_total_mib": setup_gpu["memory_total_mib"],
                    "memory_free_gib": setup_gpu["memory_free_mib"] / 1024.0,
                    "host_available_bytes": setup_host["available_bytes"],
                    "captured_utc": setup["resource_snapshot"]["captured_utc"],
                },
            },
        },
        "footing": {
            "artifacts": artifact_paths,
            "adapters": adapter_status["adapters"],
            "contracts": {
                "method_count_planned": ledger["methods"]["anchor_count"] + ledger["methods"]["new_state_count"],
                "development_fit_state_count": ledger["methods"]["new_state_count"],
                "fresh_cell_count": ledger["holdout"]["cells"],
                "records_per_domain": ledger["holdout"]["records_per_domain"],
                "fresh_prediction_shape": [ledger["holdout"]["records_per_domain"], ledger["holdout"]["sequence_tokens"]],
                "fit_geometry": ledger["training"]["fit_geometry"],
                "frequency_references": ["original", "enriched"],
                "timing": {
                    "warmup_calls_per_record": ledger["inference"]["warmup_runs_per_record"],
                    "measured_calls_per_record": ledger["inference"]["measured_runs_per_record"],
                },
                "exact_uncertainty": decision["uncertainty"]["exact_upper_bound"],
            },
            "validation_recorded_before_current_fit_window": {
                "task_focused_suite": "38 passed",
                "scorer_cli_help_and_source_compile": "pass",
                "negative_truth_sentinel_cases": adapter_status["validation"]["negative_truth_sentinel_cases"] + ["wrong_sidecar", "row_order", "unfrozen_descriptor"],
            },
        },
        "planned_fresh_evaluation": {
            "status": "PENDING_METHOD_FREEZE_AND_PANEL_CAPTURE",
            "domains": decision["fresh_evaluation"]["domains"],
            "target_conditions": decision["fresh_evaluation"]["target_conditions"],
            "records_per_domain": decision["fresh_evaluation"]["unique_source_records_per_domain"],
            "unique_sources_total": decision["fresh_evaluation"]["unique_sources_total"],
            "cells": ledger["holdout"]["cells"],
            "method_cell_artifact_count": (ledger["methods"]["anchor_count"] + ledger["methods"]["new_state_count"]) * ledger["holdout"]["cells"],
            "truth_gate": decision["fresh_evaluation"]["truth_gate"],
            "selection_status": ledger["holdout"]["selection_status"],
            "source_pool_ranges": ledger["reserved_source_pools"]["ranges"],
            "uncertainty": decision["uncertainty"],
            "practical_margins": decision["practical_margins"],
        },
        "archive": {
            "retain_compact": [
                "all phase receipts and preflight JSON",
                "corpus manifests and coverage diagnostics",
                "public validation curves, selected states, schedules, and binding metadata",
                "qknorm repair curves, selected causal states, attention diagnostic, and receipts",
                "method freeze, panel registration, prediction receipts, timing, truth-binding descriptor, freeze receipt, and score output once created",
            ],
            "retain_local_raw": [
                "raw public activation H and fit tensors needed for reproduction",
                "all task-owned selected states and prediction artifacts",
                "private truth sidecar and its external evaluator receipt after the gate",
            ],
            "compact_handoff_policy": "Do not automatically add the largest raw H or fit tensors to the final Git handoff; retain their paths, bytes, hashes, and reproduction commands in receipts.",
            "truth_sidecar_policy": "Keep evaluator truth outside reconstruction/frozen-public roots. Copy only the label-free evaluator binding descriptor into the frozen output root and have the freeze receipt cover its exact path, bytes, and SHA before loading the sidecar.",
            "replay_policy": "Replay requires checking out the exact executable phase commit recorded in the receipt, then restoring compact evidence from the archive.",
        },
    }

    memory = fit_memory["bytes"]
    report_lines = [
        "# TRR-0005 development footing and gate handoff",
        "",
        "Status: **IN_PROGRESS**. Public corpus preparation, public activation capture, six original decoder fits, the two qknorm causal fits, and bounded public diagnostics are recorded; the fresh holdout remains unselected and evaluator truth remains unopened.",
        "",
        f"The task uses reviewed TRR-0004 source parent `{setup['source_parent_commit']}` on `{setup['source_parent_branch']}` (PR {setup['reviewed_pr']}). The public development work binds original-like data to `{fit['distributions']['original']['contract_distribution_id']}` and enriched data to `{fit['distributions']['enriched']['contract_distribution_id']}`. The qknorm causal repair is now fit and qualified at its recorded commit; the final eight-method matrix remains pending two-stage method freeze, panel-bound registration, fresh capture, and the pre-truth gate.",
        "",
        "## Development findings",
        "",
        "The corpus preparation held 1,200 records and 124,371 post-BOS fitting positions per arm. The original-like bank contains "
        f"{cov['original_like_distinct_post_bos']} distinct post-BOS token IDs; the enriched bank contains {cov['enriched_distinct_post_bos']}. "
        f"Their overlap is {cov['original_enriched_overlap_distinct_post_bos']}; enrichment adds {cov['newly_covered_by_enriched_distinct_post_bos']} IDs and loses {cov['lost_from_original_distinct_post_bos']} original IDs under the fixed comparison. "
        f"The controlled supplement uses {coverage_arm['controlled']['controlled_ids_used']} selected IDs in {coverage_arm['controlled']['controlled_record_count']} records with {coverage_arm['controlled']['controlled_replacement_occurrences']} replacement occurrences. These are descriptive coverage diagnostics, not a fresh holdout selection.",
        "",
        f"Preparation took {sec(corpus_exec['elapsed_seconds'])} in the child receipt ({sec(corpus_run['elapsed_seconds'])} including launch/receipt overhead), with max RSS {corpus_exec['resource_usage']['max_rss_bytes'] / (1024**3):.3f} GiB. No public model forward or private truth access occurred in this phase. The phase commit is `{corpus_exec['git_commit']}`.",
        "",
        f"The public activation capture used the fixed batch-8 × 192 path at cut 4. Its authoritative capture interval was {sec(capture['capture']['wall_seconds'])}; the full launch interval was {sec(launch['wall_seconds'])}. The capture receipt records {capture['geometry']['original_fit']} and {capture['geometry']['enriched_fit']} fit tensors, a common validation geometry of {capture['geometry']['common_validation']}, and matching length vectors. The primary active outputs were bit-exact (`torch.equal`, maximum absolute difference 0); {qualification['primary_geometry']['changed_future_pad_tokens']} future padding tokens changed. The excluded unpadded batch-1 diagnostic compared {qualification['unpadded_batch1_diagnostic']['compared_active_values']} active values, had maximum absolute difference {qualification['unpadded_batch1_diagnostic']['maximum_absolute_difference']} and relative L2 {qualification['unpadded_batch1_diagnostic']['relative_l2']:.10f}, and was marked `{qualification['unpadded_batch1_diagnostic']['status']}`. The capture commit is `{cap_exec['git_commit']}`.",
        "",
        "The six original decoder arms were trained for 3,000 steps with seed 4005 and 512 post-BOS draws per step under the shared schedule. The run consumed "
        f"{fit['elapsed_seconds']:.3f} seconds wall time; the two qknorm causal repair arms added {qknorm_fit['elapsed_seconds']:.3f} seconds. Public validation is a reused 48-record Alpaca/Pile panel (3,133 post-BOS rows), and the step-0 baseline is "
        f"{pct(curves['original']['joint_full_affine']['step0_style_balanced_token_accuracy'])} style-balanced token accuracy ({pct(curves['original']['joint_full_affine']['step0_token_accuracy'])} token accuracy). The six-fit receipt commit is `{fit['git_commit']}` and the qknorm receipt commit is `{qknorm_fit['git_commit']}`. The values below are development selection evidence; fit-stream metrics and these public curves are not fresh outcomes.",
        "",
        "| Distribution | Method | Selected step | Public style-balanced accuracy | Token accuracy | Style-balanced headroom | Arm wall time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for distribution in distributions:
        for method in methods:
            row = curves[distribution][method]
            report_lines.append(
                f"| {distribution} | `{method}` | {row['selected_step']} | {pct(row['best_style_balanced_token_accuracy'])} | {pct(row['best_token_accuracy'])} | {pp(row['style_balanced_headroom_pp'] / 100.0)} | {sec(row['arm_wall_seconds'])} |"
            )
    report_lines += [
        "",
        "| Distribution | Method | Selected step | Public style-balanced accuracy | Token accuracy | Style-balanced headroom | Arm wall time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for distribution in distributions:
        row = qknorm_curves[distribution]["affine_causal_h_attention128"]
        report_lines.append(
            f"| {distribution} | `qknorm/{row['method_id']}` | {row['selected_step']} | {pct(row['best_style_balanced_token_accuracy'])} | {pct(row['best_token_accuracy'])} | {pp(row['style_balanced_headroom_pp'] / 100.0)} | {sec(row['arm_wall_seconds'])} |"
        )
    report_lines += [
        "",
        "The qknorm causal repair used the predeclared cosine-scale-4 score rule. Its public selections were original step "
        f"{qknorm_curves['original']['affine_causal_h_attention128']['selected_step']} at {pct(qknorm_curves['original']['affine_causal_h_attention128']['best_style_balanced_token_accuracy'])} and enriched step "
        f"{qknorm_curves['enriched']['affine_causal_h_attention128']['selected_step']} at {pct(qknorm_curves['enriched']['affine_causal_h_attention128']['best_style_balanced_token_accuracy'])}. "
        "The current positionwise leaders remain trained diagonal after comparing against these repaired causal curves: original step "
        f"{positionwise['original']['selected_step']} at {pct(positionwise['original']['best_style_balanced_token_accuracy'])}, and enriched step {positionwise['enriched']['selected_step']} at {pct(positionwise['enriched']['best_style_balanced_token_accuracy'])}. "
        "These are public-panel observations rather than the final frozen choice, and the two distribution selections are not independent evidence.",
        "",
        f"The preserved V1 largest-cell qualification failed after {qual_v1['elapsed_seconds']:.3f} seconds because measured device peak {qual_v1['error'].split('>')[0].split(':')[-1].strip()} exceeded the analytic forecast; the exact failure is retained in `experiments/TRR-0005/joint_qualification_v1/failure.json`. V2 then qualified both distributions under the conservative envelope {gib(fit_memory['gib']['conservative_envelope'])} ({fit_memory['bytes']['conservative_envelope']} bytes), with total qualification wall time {sec(qual_v2['total_qualification_wall_seconds'])}. The qknorm score-rule repair passed its two-step public qualification in {sec(qknorm_qual['total_qualification_wall_seconds'])} and its two full public fits in {sec(qknorm_fit['elapsed_seconds'])}; no fresh evaluator resource was loaded.",
        "",
        "The public H-only attention diagnostics found a routing limitation in the original dot-product causal state. Original validation queries assigned "
        f"{pct(attention['states']['original']['summary']['overall']['average_bos_mass'], 3)} average mass to BOS, {attention['states']['original']['summary']['overall']['average_current_position_mass']:.3e} to the current position, "
        f"{attention['states']['original']['summary']['overall']['average_earlier_position_mass']:.3e} to earlier positions, and {attention['states']['original']['summary']['overall']['average_entropy_nats']:.3e} nats entropy; enriched queries assigned "
        f"{pct(attention['states']['enriched']['summary']['overall']['average_bos_mass'], 3)} to BOS, {attention['states']['enriched']['summary']['overall']['average_current_position_mass']:.3e} to the current position, and "
        f"{pct(attention['states']['enriched']['summary']['overall']['average_earlier_position_mass'], 3)} to earlier non-BOS positions. "
        f"The qknorm diagnostic raised current-position mass to {pct(attention_qknorm['states']['original']['summary']['overall']['average_current_position_mass'], 3)} original and {pct(attention_qknorm['states']['enriched']['summary']['overall']['average_current_position_mass'], 3)} enriched, and earlier-position mass to {pct(attention_qknorm['states']['original']['summary']['overall']['average_earlier_position_mass'], 3)} original and {pct(attention_qknorm['states']['enriched']['summary']['overall']['average_earlier_position_mass'], 3)} enriched, with entropy {attention_qknorm['states']['original']['summary']['overall']['average_entropy_nats']:.3f} and {attention_qknorm['states']['enriched']['summary']['overall']['average_entropy_nats']:.3f} nats. Self-mass above 0.99 was zero in all four states. These diagnostics test the learned score branches only; they do not show that earlier H is generally uninformative.",
        "",
        "An archived Finance-128 runtime qualification completed at commit "
        f"`{prediction_qualification['git_commit']}` in {sec(prediction_qualification['elapsed_seconds'])}. All {prediction_qualification['method_count']} methods produced one warmup and one measured output with exact ID stability; the archived A2 anchor measured {sec(prediction_qualification['methods']['frozen_a1_a2_k256']['timing']['measured_seconds'])} after a {sec(prediction_qualification['methods']['frozen_a1_a2_k256']['timing']['warmup_seconds'])} warmup. It loaded no fresh panel, target labels, evaluator truth, or future activations, so this is runtime qualification evidence rather than a fresh outcome.",
        "",
        "## Resource and integrity accounting",
        "",
        "The phase commits are machine-ingested in the manifest: reviewed parent "
        f"`{setup['source_parent_commit']}`, corpus `{corpus_exec['git_commit']}`, capture `{cap_exec['git_commit']}`, six-fit `{fit['git_commit']}`, qknorm `{qknorm_fit['git_commit']}`, inference qualification `{prediction_qualification['git_commit']}`, and attention diagnostics `{attention_exec['git_commit']}`/`{attention_qknorm_exec['git_commit']}`.",
        "",
        "The setup snapshot recorded "
        f"{setup_gpu['memory_free_mib']} MiB ({setup_gpu['memory_free_mib'] / 1024.0:.3f} GiB) free on the {setup_gpu['name']} with {setup_gpu['memory_total_mib']} MiB total. "
        f"The capture measured {capture['capture']['peak_memory']['cuda_peak_allocated_bytes']} allocated and {capture['capture']['peak_memory']['cuda_peak_reserved_bytes']} reserved device bytes, with host max RSS {capture['capture']['peak_memory']['host_max_rss_bytes']} bytes. The six-fit and qknorm causal arms recorded their per-arm peaks in the manifest. "
        "The fit preflight records the following component accounting:",
        "",
        "| Component | Bytes | GiB |",
        "|---|---:|---:|",
    ]
    memory_keys = [
        "embedding_table_fp32",
        "activation_batch_fp32",
        "adamw_m_v",
        "gradient_buffer",
        "selected_logits_fp32",
        "selected_logits_backward_workspace_fp32",
        "attention_scores_fp32",
        "hidden_workspace_envelope",
        "max_model_parameters_fp32",
        "raw_sum",
        "training_workspace_envelope",
        "training_peak_envelope",
        "validation_workspace_envelope",
        "validation_peak_envelope",
        "analytic_conservative_envelope",
        "measured_v1_qualification_peak",
        "measured_qualification_floor",
        "conservative_envelope",
    ]
    for key in memory_keys:
        report_lines.append(f"| `{key}` | {memory[key]} | {bytes_gib(memory[key]):.6f} |")
    report_lines += [
        "",
        f"The forecast uses the worst-case training-backward stage with AdamW and sequential arm residency. The preserved V1 measured peak is {gib(bytes_gib(memory['measured_v1_qualification_peak']))}; the 1.5× empirical floor is {gib(bytes_gib(memory['measured_qualification_floor']))}. These are auditable estimates plus live guards, not a capacity guarantee.",
        "",
        "The executable footing validates frozen receipts and bindings before any truth loader: prediction tensors, shapes, integer/range/BOS/right-padding coverage, observations, state/code/config bindings, ordered records, sidecar identity, panel/selection/observation linkage, and one warmup plus one measured timing record per prediction. The task-local focused suite and CLI/source checks were run before the current fit window; negative sentinels include missing, truncated, out-of-range, shape, missing-tensor, stale-state, stale-code, wrong-sidecar, row-order, and unfrozen-descriptor cases.",
        "",
        "## Fresh evaluation remains pending",
        "",
        f"The declared fresh matrix has {decision['fresh_evaluation']['unique_source_records_per_domain']} paired public sources per domain across {len(decision['fresh_evaluation']['domains'])} domains and {len(decision['fresh_evaluation']['target_conditions'])} target conditions, giving {decision['fresh_evaluation']['unique_sources_total']} unique sources and {ledger['holdout']['cells']} cells. The qknorm causal states are now available for the final frozen eight-method registration, but all {ledger['methods']['anchor_count'] + ledger['methods']['new_state_count']} methods must still produce complete, immutable prediction and timing artifacts before truth opens. The report will keep Pile and Finance, base and synthetic-LoRA, frequency × position × domain errors, gains, regressions, and paired uncertainty separate; it will not pool domains or treat paired targets as independent.",
        "",
        f"The declared exact-record analysis uses one-sided Clopper–Pearson tails alpha 0.05/32 for the gain/loss bound, so zero beneficial discordances at n={decision['fresh_evaluation']['unique_source_records_per_domain']} still has a conservative upper net-benefit bound of about 4.93 percentage points. Descriptive paired source bootstrap uses {decision['uncertainty']['bootstrap_draws']} resamples with seed {decision['uncertainty']['bootstrap_seed']}; practical margins and the final decision remain pending fresh observations.",
        "",
        "Reproduction and archive handling are documented in `experiments/TRR-0005/footing/development_reproduction.md`. The machine-ingested manifest is `experiments/TRR-0005/manifest.json`; both remain explicitly IN_PROGRESS until root completes freeze, fresh capture, prediction gating, truth-gated scoring, and final decision assembly.",
        "",
    ]
    report = "\n".join(report_lines)

    corpus_cmd = command_for("experiments/TRR-0005/corpus_run/run_receipt.json", "argv")
    capture_cmd = command_for("experiments/TRR-0005/public_activation_v1/launch.json", "command")
    attention_cmd = command_for("experiments/TRR-0005/attention_diagnostic_execution.json", "argv")
    attention_qknorm_cmd = command_for("experiments/TRR-0005/attention_diagnostic_qknorm_v1_execution.json", "argv")
    doc = f"""# TRR-0005 development reproduction and archive checklist

This document records completed public development evidence. It does not select or inspect the fresh holdout or open evaluator truth. The report and manifest builder reads compact JSON receipts and learning curves only.

## Regenerate the task-local report and manifest

From the TRR-0005 worktree:

```bash
.venv-trr0005/bin/python experiments/TRR-0005/footing/build_development_evidence.py
```

The builder writes `coordination/results/TRR-0005.md` and `experiments/TRR-0005/manifest.json`. It does not read safetensors or model weights.

## Recorded development commands

These commands are copied from the corresponding execution receipts. They are historical evidence; rerun them only under the root coordinator's resource and data-access decisions.

Corpus preparation (`experiments/TRR-0005/corpus_run/run_receipt.json`):

```bash
{corpus_cmd}
```

Public activation capture (`experiments/TRR-0005/public_activation_v1/launch.json`):

```bash
{capture_cmd}
```

The capture receipt records the fixed batch-8 × 192 bit-exact path. The unpadded batch-1 diagnostic is retained as excluded numerical evidence and must not be substituted for the captured path.

Joint-fit interface and qualifier templates are in `experiments/TRR-0005/joint_fit_interface.md`; the completed original six-fit receipt is `experiments/TRR-0005/joint_fit_v1/run_evidence.json`, and the completed qknorm repair receipt is `experiments/TRR-0005/joint_fit_qknorm_v1/run_evidence.json`. Their receipts and curves are the source of all development selection/timing values in the generated manifest. The qknorm repair uses the predeclared cosine-scale-4 score rule and supplies the two repaired causal contenders for the eventual method freeze.

Public H-only attention diagnostic (`experiments/TRR-0005/attention_diagnostic_execution.json`):

```bash
{attention_cmd}
```

The qknorm public H-only diagnostic (`experiments/TRR-0005/attention_diagnostic_qknorm_v1_execution.json`) was run with:

```bash
{attention_qknorm_cmd}
```

## Compact evidence to retain

- Setup/preflight, corpus design/plan/run, source-pool reservations, and coverage diagnostics.
- Public capture raw/normalized receipts, launch/preflight, manifests, record metadata, source bindings, output bytes/SHA, and the excluded equivalence diagnostic.
- Joint-fit design, memory preflight, sampler receipts/schedules, pretraining diagnostics, learning curves, selected states, arm timing, and both original/qknorm `run_evidence.json` files.
- Preserved V1 failure, successful V2 qualification, qknorm amendment/qualification and fit, the archived Finance-128 prediction qualification, and both attention diagnostic/result/receipt sets.
- Frozen method selection, method registration, panel descriptor, observation/prediction/timing receipts, truth-binding descriptor, freeze receipt, and scorer output after fresh evaluation exists.

Retain the task-owned raw H and fit tensors locally when needed for reproduction. The compact Git handoff should carry receipts, metadata, selected states, predictions, and hashes, while leaving the largest raw tensors in the archive with their recorded paths and reproduction commands. Keep the evaluator truth sidecar outside reconstruction and frozen-public roots; copy only the label-free binding descriptor into the frozen output root, and require the freeze receipt to cover its exact path, bytes, and SHA before the scorer loads truth.

For replay, check out the exact phase commit recorded in the relevant receipt, then restore compact evidence from the archive. Do not treat public-panel curves or fit-stream metrics as fresh holdout outcomes.
"""
    return manifest, report, doc


def main() -> None:
    manifest, report, doc = build()
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(report)
    DOC.write_text(doc)


if __name__ == "__main__":
    main()
