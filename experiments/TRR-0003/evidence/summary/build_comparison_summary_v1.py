#!/usr/bin/env python3
"""Build a source-bound retrospective TRR-0003 comparison and Track-A figure.

This script only reads frozen score/runtime evidence and writes separate
comparison-summary artifacts.  It deliberately ignores each score cell's
historical timing_path because those merged paths are not the executed raw
sources; timing is rebound to the actual per-cell evidence files below.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "experiments" / "TRR-0003"
SUMMARY_PATH = TASK_DIR / "comparison_summary_v1.json"
MARKDOWN_PATH = TASK_DIR / "comparison_summary_v1.md"
FIGURE_PNG = TASK_DIR / "comparison_summary_v1.png"
FIGURE_SVG = TASK_DIR / "comparison_summary_v1.svg"

CELL_ORDER = [
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
]
CELL_LABELS = {
    "pile__public_base": "Pile / base",
    "pile__public_lora_2601": "Pile / shifted",
    "finance__public_base": "Finance / base",
    "finance__public_lora_2601": "Finance / shifted",
}
ITERATIONS = [0, 1, 2, 4, 8, 16, 32]
A_PREFIX = "checkpoint_reverse_fixed_point_euclidean_k16"
A_METHODS = [f"{A_PREFIX}_i{i:03d}" for i in ITERATIONS]
B_METHODS = [
    "angular_inverse_control",
    "tied_affine_token_ce",
    "residual_mlp256_token_ce",
]
COMPARATOR_METHODS = [
    "historical_alpaca_a1",
    "frozen_a1_a2_k256",
    "direct_inverse",
]
METHOD_ORDER = A_METHODS + B_METHODS + COMPARATOR_METHODS
TRACK_BY_METHOD = {
    **{method: "track_a" for method in A_METHODS},
    **{method: "track_b" for method in B_METHODS},
    **{method: "comparator" for method in COMPARATOR_METHODS},
}
METHOD_LABELS = {
    A_METHODS[0]: "checkpoint reverse fixed point / i000",
    A_METHODS[1]: "checkpoint reverse fixed point / i001",
    A_METHODS[2]: "checkpoint reverse fixed point / i002",
    A_METHODS[3]: "checkpoint reverse fixed point / i004",
    A_METHODS[4]: "checkpoint reverse fixed point / i008",
    A_METHODS[5]: "checkpoint reverse fixed point / i016",
    A_METHODS[6]: "checkpoint reverse fixed point / i032",
    "angular_inverse_control": "angular inverse control",
    "tied_affine_token_ce": "tied affine token CE",
    "residual_mlp256_token_ce": "residual MLP-256 token CE",
    "historical_alpaca_a1": "historical Alpaca A1",
    "frozen_a1_a2_k256": "frozen A1+A2 K256",
    "direct_inverse": "direct inverse",
}

SCORE_REL = Path("experiments/TRR-0003/evidence/common_score_v2.json")
RUNTIME_REL = Path("experiments/TRR-0003/track_a/runtime_analysis.json")
TRACK_B_REL = Path("outputs/TRR-0003/track_b/panel_selected_v1/prediction_evidence.json")
TRACK_B_INVENTORY_REL = Path("experiments/TRR-0003/track_b/inventory_v1.json")
COMPARATOR_RUN_REL = Path("outputs/TRR-0003/footing/comparator_matrix_v2/run_evidence.json")
COMPARATOR_BINDINGS_REL = Path("outputs/TRR-0003/footing/comparator_matrix_v2/bindings.json")
FREEZE_REL = Path("experiments/TRR-0003/footing/common_matrix_v2.freeze.json")
PANEL_REL = Path("experiments/TRR-0003/footing/panel.json")
ARCHIVE_ROOT_REL = Path("experiments/TRR-0003/evidence/comparison_sources_v1")

_hash_cache: dict[str, dict[str, Any]] = {}
_doc_cache: dict[str, Any] = {}


def actual_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: str | Path) -> Any:
    p = actual_path(path)
    key = str(p.resolve())
    if key not in _doc_cache:
        _doc_cache[key] = json.loads(p.read_text())
    return _doc_cache[key]


def file_record(path: str | Path, declared_sha256: str | None = None) -> dict[str, Any]:
    p = actual_path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    key = str(p.resolve())
    if key not in _hash_cache:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        _hash_cache[key] = {
            "path": relative_path(p),
            "bytes": p.stat().st_size,
            "sha256": digest,
        }
    result = dict(_hash_cache[key])
    if declared_sha256 is not None:
        result["declared_sha256"] = declared_sha256
        result["hash_matches_declared"] = declared_sha256 == result["sha256"]
    return result


def archived_timing_path(path: str | Path) -> Path:
    """Return the tracked byte-identical copy path for an ignored output."""
    return ROOT / ARCHIVE_ROOT_REL / relative_path(actual_path(path))


def timing_file_record(path: str | Path, declared_sha256: str | None = None) -> dict[str, Any]:
    record = file_record(path, declared_sha256)
    archive = archived_timing_path(path)
    if not archive.is_file():
        raise FileNotFoundError(f"tracked timing archive missing for {path}: {archive}")
    archive_record = file_record(archive)
    if archive_record["bytes"] != record["bytes"] or archive_record["sha256"] != record["sha256"]:
        raise AssertionError(f"timing archive differs from executed source: {archive}")
    record["archived_path"] = archive_record["path"]
    record["archived_sha256"] = archive_record["sha256"]
    return record


def json_pointer_key(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def rounded_or_none(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    return round(float(value), digits)


def memory_copy(value: Any) -> Any:
    if value is None:
        return None
    return json.loads(json.dumps(value))


def score_metric(score_cell: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(score_cell["metrics"])
    # Retain the exact score values; no recomputation from rounded displays.
    return metrics


def main() -> None:
    score_path = actual_path(SCORE_REL)
    runtime_path = actual_path(RUNTIME_REL)
    track_b_path = actual_path(TRACK_B_REL)
    track_b_inventory_path = actual_path(TRACK_B_INVENTORY_REL)
    comparator_run_path = actual_path(COMPARATOR_RUN_REL)
    comparator_bindings_path = actual_path(COMPARATOR_BINDINGS_REL)
    freeze_path = actual_path(FREEZE_REL)
    panel_path = actual_path(PANEL_REL)
    generator_path = actual_path(__file__)

    score = load_json(score_path)
    runtime = load_json(runtime_path)
    track_b = load_json(track_b_path)
    track_b_inventory = load_json(track_b_inventory_path)
    comparator_run = load_json(comparator_run_path)
    comparator_bindings = load_json(comparator_bindings_path)
    freeze = load_json(freeze_path)
    panel = load_json(panel_path)

    if score.get("method_ids") != METHOD_ORDER:
        raise AssertionError(f"score method order changed: {score.get('method_ids')}")
    if set(score.get("cells", {})) != {
        f"{cell}__{method}" for cell in CELL_ORDER for method in METHOD_ORDER
    }:
        raise AssertionError("common score does not contain exactly the expected 52 Cartesian cells")
    if list(runtime.get("cells", {})) != [
        "finance__public_base",
        "finance__public_lora_2601",
        "pile__public_base",
        "pile__public_lora_2601",
    ]:
        raise AssertionError("unexpected Track A runtime-analysis cell set/order")

    top_sources = {
        "common_score_v2": file_record(score_path),
        "track_a_runtime_analysis": file_record(runtime_path),
        "track_b_prediction_evidence": timing_file_record(track_b_path),
        "track_b_inventory": file_record(track_b_inventory_path),
        "comparator_run_evidence": timing_file_record(comparator_run_path),
        "comparator_bindings": file_record(comparator_bindings_path),
        "freeze_receipt": file_record(freeze_path),
        "panel": file_record(panel_path),
        "comparison_generator": file_record(generator_path),
    }

    # Rebind Track-A timing to the raw per-cell evidence.  The runtime-analysis
    # summary is kept as a second source and used for a value-equality check.
    a_raw: dict[str, dict[str, Any]] = {}
    a_timing_records: dict[str, dict[str, Any]] = {}
    a_runtime_cells = runtime["cells"]
    for cell_id in CELL_ORDER:
        cell_summary = a_runtime_cells[cell_id]
        raw_path = actual_path(cell_summary["evidence_path"])
        raw = load_json(raw_path)
        raw_record = timing_file_record(raw_path, cell_summary.get("evidence_sha256"))
        a_raw[cell_id] = raw
        a_timing_records[cell_id] = raw_record
        if raw.get("truth_opened") is not False:
            raise AssertionError(f"Track A raw evidence opened truth: {cell_id}")
        raw_iterations = [entry["iterations"] for entry in raw["iterations"]]
        if raw_iterations != ITERATIONS:
            raise AssertionError(f"Track A iteration ladder changed for {cell_id}: {raw_iterations}")

    b_rows: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for method_id, records in track_b["cells"].items():
        for index, record in enumerate(records):
            b_rows[(record["cell_id"], method_id)] = (index, record)
    if set(b_rows) != {(cell, method) for cell in CELL_ORDER for method in B_METHODS}:
        raise AssertionError("Track B panel timing cells are incomplete")
    if track_b.get("runtime", {}).get("truth_opened") is not False:
        raise AssertionError("Track B prediction evidence opened truth")
    b_selected_states = {
        method_id: track_b_inventory["states"][method_id]["selected"]
        for method_id in B_METHODS
    }
    embedding_meta = track_b_inventory["fixed_runtime_footprint"]["normalized_public_embedding_table"]
    if not actual_path(embedding_meta["path"]).is_file():
        raise AssertionError(f"Track B shared embedding asset is missing: {embedding_meta['path']}")

    comparator_entries: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for method_id in COMPARATOR_METHODS:
        entries = comparator_run["timing"][method_id]
        for index, entry in enumerate(entries):
            comparator_entries[(entry["cell_id"], method_id)] = (index, entry)
    if set(comparator_entries) != {(cell, method) for cell in CELL_ORDER for method in COMPARATOR_METHODS}:
        raise AssertionError("comparator run evidence timing cells are incomplete")

    # Comparator evidence files also carry a per-cell safetensors artifact. That
    # artifact is the prediction output (its size scales with the number of
    # scored positions), not the learned method state. Bind retained state to
    # the fixed resources in the comparator's method-state receipt and retain
    # prediction artifacts in a separate catalog below.
    if set(comparator_bindings) != set(COMPARATOR_METHODS):
        raise AssertionError("comparator method-state binding set changed")
    comparator_state_records: dict[str, dict[str, Any]] = {}
    comparator_state_roles = {
        "historical_alpaca_a1": "comparator historical Alpaca A1 lens state used by historical A1 and A1+A2",
        "frozen_a1_a2_k256": "comparator historical Alpaca A1 lens state used by historical A1 and A1+A2",
        "direct_inverse": "comparator direct inverse state",
    }
    for method_id in COMPARATOR_METHODS:
        method_binding = comparator_bindings[method_id]
        method_states = method_binding.get("method_state")
        if not isinstance(method_states, list) or len(method_states) != 1:
            raise AssertionError(f"expected one fixed method-state resource for {method_id}")
        state_meta = method_states[0]
        state_record = file_record(state_meta["path"], state_meta.get("sha256"))
        if state_record["bytes"] != state_meta.get("bytes"):
            raise AssertionError(f"comparator state byte binding mismatch: {method_id}")
        if state_record["sha256"] != state_meta.get("sha256"):
            raise AssertionError(f"comparator state hash binding mismatch: {method_id}")
        comparator_state_records[method_id] = state_record

    timing_catalog: list[dict[str, Any]] = []
    timing_catalog_by_path: dict[str, str] = {}
    retained_state_catalog: list[dict[str, Any]] = []
    retained_state_catalog_by_path: dict[str, str] = {}
    prediction_artifact_catalog: list[dict[str, Any]] = []
    prediction_artifact_catalog_by_path: dict[str, str] = {}

    def register_timing(record: dict[str, Any], pointer: str) -> str:
        path = record["path"]
        if path not in timing_catalog_by_path:
            source_id = f"T{len(timing_catalog) + 1:02d}"
            timing_catalog_by_path[path] = source_id
            timing_catalog.append({
                "source_id": source_id,
                "path": path,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "declared_sha256": record.get("declared_sha256"),
                "hash_matches_declared": record.get("hash_matches_declared"),
                "archived_path": record.get("archived_path"),
                "archived_sha256": record.get("archived_sha256"),
                "json_pointers": [],
            })
        source_id = timing_catalog_by_path[path]
        entry = next(item for item in timing_catalog if item["source_id"] == source_id)
        if pointer not in entry["json_pointers"]:
            entry["json_pointers"].append(pointer)
        return source_id

    def register_state(record: dict[str, Any], role: str, cell_id: str, method_id: str) -> str:
        path = record["path"]
        if path not in retained_state_catalog_by_path:
            source_id = f"S{len(retained_state_catalog) + 1:02d}"
            retained_state_catalog_by_path[path] = source_id
            retained_state_catalog.append({
                "source_id": source_id,
                "path": path,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "declared_sha256": record.get("declared_sha256"),
                "hash_matches_declared": record.get("hash_matches_declared"),
                "role": role,
                "usages": [],
            })
        source_id = retained_state_catalog_by_path[path]
        entry = next(item for item in retained_state_catalog if item["source_id"] == source_id)
        usage = {"cell_id": cell_id, "method_id": method_id}
        if usage not in entry["usages"]:
            entry["usages"].append(usage)
        return source_id

    def register_prediction_artifact(
        record: dict[str, Any], cell_id: str, method_id: str, role: str
    ) -> str:
        path = record["path"]
        if path not in prediction_artifact_catalog_by_path:
            source_id = f"P{len(prediction_artifact_catalog) + 1:02d}"
            prediction_artifact_catalog_by_path[path] = source_id
            prediction_artifact_catalog.append({
                "source_id": source_id,
                "path": path,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "declared_sha256": record.get("declared_sha256"),
                "hash_matches_declared": record.get("hash_matches_declared"),
                "role": role,
                "usages": [],
            })
        source_id = prediction_artifact_catalog_by_path[path]
        entry = next(item for item in prediction_artifact_catalog if item["source_id"] == source_id)
        usage = {"cell_id": cell_id, "method_id": method_id}
        if usage not in entry["usages"]:
            entry["usages"].append(usage)
        return source_id

    rows: list[dict[str, Any]] = []
    for cell_id in CELL_ORDER:
        style, condition = cell_id.split("__", 1)
        for method_id in METHOD_ORDER:
            score_key = f"{cell_id}__{method_id}"
            score_cell = score["cells"][score_key]
            metrics = score_metric(score_cell)
            track = TRACK_BY_METHOD[method_id]
            state_record: dict[str, Any] | None = None
            state_role: str | None = None
            row: dict[str, Any] = {
                "cell_id": cell_id,
                "style": style,
                "condition": condition,
                "method_id": method_id,
                "method_label": METHOD_LABELS[method_id],
                "track": track,
                "score_source_json_pointer": f"/cells/{json_pointer_key(score_key)}/metrics",
                "score_timing_path_claimed": score_cell.get("timing_path"),
                "metrics": metrics,
            }
            prediction_artifact_record: dict[str, Any] | None = None
            prediction_artifact_source_id: str | None = None

            if track == "track_a":
                iteration = int(method_id.rsplit("_i", 1)[1])
                raw = a_raw[cell_id]
                raw_index = next(index for index, item in enumerate(raw["iterations"]) if item["iterations"] == iteration)
                aggregate = raw["iterations"][raw_index]["aggregate"]
                runtime_item = next(
                    item for item in a_runtime_cells[cell_id]["iteration_summary"] if item["iteration"] == iteration
                )
                for field in ("inference_seconds", "public_prefix_layer_evaluations", "branch_forward_calls", "cycle_forward_passes"):
                    if aggregate.get(field) != runtime_item.get(field):
                        if not math.isclose(float(aggregate.get(field)), float(runtime_item.get(field)), rel_tol=0, abs_tol=1e-9):
                            raise AssertionError(f"Track A runtime mismatch {cell_id} i{iteration}: {field}")
                state_meta = raw["iterations"][raw_index]["artifact"]
                prediction_artifact_record = file_record(state_meta["path"], state_meta.get("sha256"))
                prediction_artifact_source_id = register_prediction_artifact(
                    prediction_artifact_record,
                    cell_id,
                    method_id,
                    "Track-A fixed-point iteration prediction/diagnostic artifact",
                )
                state_record = None
                state_role = None
                full_prefix_layer_evaluations = (
                    int(aggregate["cycle_forward_passes"]) * int(raw["algorithm"]["cut_depth"])
                )
                pointer = f"/iterations/{raw_index}/aggregate"
                source_id = register_timing(a_timing_records[cell_id], pointer)
                cost = {
                    "branch_evaluation_calls": aggregate.get("branch_forward_calls"),
                    "full_cycle_forward_passes": aggregate.get("cycle_forward_passes"),
                    "full_prefix_layer_evaluations": full_prefix_layer_evaluations,
                    "legacy_public_prefix_layer_evaluations": aggregate.get("public_prefix_layer_evaluations"),
                    "public_prefix_layer_evaluations_include_cycle_forwards": raw["cost_contract"].get(
                        "public_prefix_layer_evaluations_include_cycle_forwards"
                    ),
                    "public_prefix_calls": None,
                    "candidate_simulations": aggregate.get("candidate_prefix_simulations"),
                    "logical_candidate_simulations": None,
                    "executed_candidate_simulations": None,
                    "prefix_commit_tokens": None,
                    "separation_note": (
                        "branch_evaluation_calls are reverse branch evaluations; full_cycle_forward_passes and "
                        "full_prefix_layer_evaluations are the public-model cycle work and are kept separate"
                    ),
                }
                row["timing"] = {
                    "timing_source_id": source_id,
                    "timing_source_path": a_timing_records[cell_id]["path"],
                    "timing_source_sha256": a_timing_records[cell_id]["sha256"],
                    "timing_source_archived_path": a_timing_records[cell_id]["archived_path"],
                    "timing_source_archived_sha256": a_timing_records[cell_id]["archived_sha256"],
                    "timing_source_json_pointer": pointer,
                    "runtime_analysis_source_path": top_sources["track_a_runtime_analysis"]["path"],
                    "runtime_analysis_source_sha256": top_sources["track_a_runtime_analysis"]["sha256"],
                    "runtime_analysis_json_pointer": f"/cells/{json_pointer_key(cell_id)}/iteration_summary/{raw_index}",
                    "inference_seconds": aggregate.get("inference_seconds"),
                    "inference_seconds_semantics": "raw Track A cell elapsed time for this fixed-point ladder iteration",
                    "first_inference_seconds": None,
                    "warm_inference_seconds": None,
                    "warm_inference_seconds_semantics": None,
                    "warm_repeat_exact": None,
                    "peak_memory": memory_copy(raw.get("memory", {}).get("method_peak_after_reset")),
                    "peak_memory_scope": "measured Track-A process peak after method reset",
                    "preparation_peak_memory": memory_copy(raw.get("memory", {}).get("preparation_peak")),
                    "record_batch_size": raw["cost_contract"].get("record_batch_size"),
                    "cost_events": cost,
                    "legacy_public_prefix_layer_evaluations": aggregate.get("public_prefix_layer_evaluations"),
                    "full_prefix_layer_definition": "cycle_forward_passes multiplied by cut_depth",
                    "residual": {
                        "continuous_relative_l2": aggregate.get("mean_continuous_cycle_relative_l2_scored"),
                        "discrete_relative_l2": aggregate.get("mean_discrete_cycle_relative_l2_scored"),
                    },
                }
            elif track == "track_b":
                b_index, b_record = b_rows[(cell_id, method_id)]
                selected_meta = b_selected_states[method_id]
                state_record = file_record(selected_meta["path"], selected_meta.get("sha256"))
                state_role = "Track-B selected standalone decoder state"
                pointer = f"/cells/{json_pointer_key(method_id)}/{b_index}"
                source_id = register_timing(top_sources["track_b_prediction_evidence"], pointer)
                runtime = track_b["runtime"]
                cost = {
                    "branch_evaluation_calls": None,
                    "full_cycle_forward_passes": None,
                    "full_prefix_layer_evaluations": None,
                    "public_prefix_layer_evaluations_include_cycle_forwards": None,
                    "public_prefix_calls": runtime.get("public_prefix_calls", 0),
                    "candidate_simulations": runtime.get("candidate_simulations", 0),
                    "logical_candidate_simulations": None,
                    "executed_candidate_simulations": None,
                    "prefix_commit_tokens": 0,
                    "separation_note": "standalone direct argmax decoder; no branch or public-prefix events were executed",
                }
                row["timing"] = {
                    "timing_source_id": source_id,
                    "timing_source_path": top_sources["track_b_prediction_evidence"]["path"],
                    "timing_source_sha256": top_sources["track_b_prediction_evidence"]["sha256"],
                    "timing_source_archived_path": top_sources["track_b_prediction_evidence"]["archived_path"],
                    "timing_source_archived_sha256": top_sources["track_b_prediction_evidence"]["archived_sha256"],
                    "timing_source_json_pointer": pointer,
                    "inference_seconds": b_record.get("first_inference_seconds"),
                    "inference_seconds_semantics": "first per-cell prediction interval from the frozen standalone adapter",
                    "first_inference_seconds": b_record.get("first_inference_seconds"),
                    "warm_inference_seconds": b_record.get("warm_inference_seconds"),
                    "warm_inference_seconds_semantics": "exact same-cell warm repeat",
                    "warm_repeat_exact": b_record.get("warm_repeat_exact"),
                    "peak_memory": memory_copy(b_record.get("peak_memory")),
                    "peak_memory_scope": "measured standalone decoder process peak",
                    "preparation_peak_memory": None,
                    "record_batch_size": b_record.get("record_batch_size"),
                    "cost_events": cost,
                    "startup_seconds": {
                        "embedding_load_validation": runtime.get("embedding_load_validation_seconds"),
                        "embedding_hash": runtime.get("embedding_hash_seconds"),
                        "embedding_transfer": runtime.get("embedding_transfer_seconds"),
                    },
                }
            else:
                comparator_index, run_entry = comparator_entries[(cell_id, method_id)]
                style = score_cell["style"]
                condition = score_cell["condition"]
                raw_path = ROOT / "outputs" / "TRR-0003" / "footing" / "comparator_matrix_v2" / style / condition / f"{method_id}.evidence.json"
                raw = load_json(raw_path)
                raw_method = raw["method"]
                if raw_method.get("cell_id") != cell_id or raw_method.get("method_id") != method_id:
                    raise AssertionError(f"comparator raw evidence binding mismatch: {raw_path}")
                if not math.isclose(float(raw_method["elapsed_seconds"]), float(run_entry["elapsed_seconds"]), rel_tol=0, abs_tol=1e-9):
                    raise AssertionError(f"comparator raw/run timing mismatch: {raw_path}")
                prediction_artifact_record = file_record(raw_method["artifact"], raw.get("artifact_sha256"))
                prediction_artifact_source_id = register_prediction_artifact(
                    prediction_artifact_record,
                    cell_id,
                    method_id,
                    "comparator prediction artifact",
                )
                state_record = comparator_state_records[method_id]
                state_role = comparator_state_roles[method_id]
                pointer = "/method"
                raw_record = timing_file_record(raw_path)
                source_id = register_timing(raw_record, pointer)
                cost = {
                    "branch_evaluation_calls": None,
                    "full_cycle_forward_passes": None,
                    "full_prefix_layer_evaluations": None,
                    "public_prefix_layer_evaluations_include_cycle_forwards": None,
                    "public_prefix_calls": raw_method.get("public_prefix_calls"),
                    "candidate_simulations": raw_method.get("candidate_simulations"),
                    "logical_candidate_simulations": raw_method.get("logical_candidate_simulations"),
                    "executed_candidate_simulations": raw_method.get("executed_candidate_simulations"),
                    "prefix_commit_tokens": raw_method.get("prefix_commit_tokens"),
                    "separation_note": "comparator prefix calls and candidate simulations are reported separately from branch/full-cycle events",
                }
                row["timing"] = {
                    "timing_source_id": source_id,
                    "timing_source_path": raw_record["path"],
                    "timing_source_sha256": raw_record["sha256"],
                    "timing_source_archived_path": raw_record["archived_path"],
                    "timing_source_archived_sha256": raw_record["archived_sha256"],
                    "timing_source_json_pointer": pointer,
                    "run_evidence_source_path": top_sources["comparator_run_evidence"]["path"],
                    "run_evidence_source_sha256": top_sources["comparator_run_evidence"]["sha256"],
                    "run_evidence_json_pointer": f"/timing/{json_pointer_key(method_id)}/{comparator_index}",
                    "inference_seconds": raw_method.get("elapsed_seconds"),
                    "inference_seconds_semantics": "raw comparator cell elapsed_seconds",
                    "first_inference_seconds": raw_method.get("elapsed_seconds") if raw_method.get("cold_start") else None,
                    "warm_inference_seconds": raw_method.get("elapsed_seconds") if raw_method.get("cold_start") is False else None,
                    "warm_inference_seconds_semantics": "subsequent cell in resident comparator process; not an exact repeat",
                    "warm_repeat_exact": None,
                    "cold_start": raw_method.get("cold_start"),
                    "peak_memory": memory_copy(raw_method.get("peak_memory")),
                    "peak_memory_scope": "measured comparator process peak including common loaded model/harness resources; not a minimal method-only footprint",
                    "preparation_peak_memory": None,
                    "record_batch_size": raw_method.get("record_batch_size"),
                    "cost_events": cost,
                    "runtime_components": {
                        "proposal_seconds": raw_method.get("proposal_seconds"),
                        "per_record_seconds": raw_method.get("per_record_seconds"),
                        "per_scored_token_seconds": raw_method.get("per_scored_token_seconds"),
                        "rule": raw_method.get("rule"),
                        "candidate_budget": raw_method.get("candidate_budget"),
                        "active_tokens": raw_method.get("active_tokens"),
                        "records": raw_method.get("records"),
                        "scored_tokens": raw_method.get("scored_tokens"),
                    },
                }
            if state_record is not None or state_role is not None:
                state_source_id = register_state(state_record, state_role, cell_id, method_id)
                row["retained_state"] = {
                    "source_id": state_source_id,
                    "path": state_record["path"],
                    "bytes": state_record["bytes"],
                    "sha256": state_record["sha256"],
                    "declared_sha256": state_record.get("declared_sha256"),
                    "hash_matches_declared": state_record.get("hash_matches_declared"),
                    "role": state_role,
                }
            elif track != "track_a":
                raise AssertionError(f"missing retained-state binding for {cell_id}/{method_id}")
            if prediction_artifact_record is not None:
                row["prediction_artifact"] = {
                    "source_id": prediction_artifact_source_id,
                    "path": prediction_artifact_record["path"],
                    "bytes": prediction_artifact_record["bytes"],
                    "sha256": prediction_artifact_record["sha256"],
                    "declared_sha256": prediction_artifact_record.get("declared_sha256"),
                    "hash_matches_declared": prediction_artifact_record.get("hash_matches_declared"),
                    "role": (
                        "Track-A fixed-point iteration prediction/diagnostic artifact; excluded from retained method-state accounting"
                        if track == "track_a"
                        else "comparator prediction artifact; excluded from retained method-state accounting"
                    ),
                }
            rows.append(row)

    if len(rows) != 52 or len({(r["cell_id"], r["method_id"]) for r in rows}) != 52:
        raise AssertionError("summary row set is not exactly 13 methods x 4 cells")

    # Build the Track-A plot data from the same exact rows and raw runtime data.
    row_by_pair = {(r["cell_id"], r["method_id"]): r for r in rows}
    plot_data: dict[str, Any] = {"iterations": ITERATIONS, "cells": {}}
    for cell_id in CELL_ORDER:
        cell_plot: dict[str, list[Any]] = {
            "token_accuracy": [],
            "continuous_residual": [],
            "discrete_residual": [],
            "inference_seconds": [],
            "branch_forward_calls": [],
            "full_cycle_forward_passes": [],
            "full_prefix_layer_evaluations": [],
        }
        for iteration in ITERATIONS:
            method_id = f"{A_PREFIX}_i{iteration:03d}"
            row = row_by_pair[(cell_id, method_id)]
            timing = row["timing"]
            cost = timing["cost_events"]
            cell_plot["token_accuracy"].append(row["metrics"]["token_accuracy"])
            cell_plot["continuous_residual"].append(timing["residual"]["continuous_relative_l2"])
            cell_plot["discrete_residual"].append(timing["residual"]["discrete_relative_l2"])
            cell_plot["inference_seconds"].append(timing["inference_seconds"])
            cell_plot["branch_forward_calls"].append(cost["branch_evaluation_calls"])
            cell_plot["full_cycle_forward_passes"].append(cost["full_cycle_forward_passes"])
            cell_plot["full_prefix_layer_evaluations"].append(cost["full_prefix_layer_evaluations"])
        plot_data["cells"][cell_id] = cell_plot

    # A compact standard static scientific figure.  Means/ranges in residual and
    # cost panels are unweighted across the four paired cells; accuracy remains
    # per-cell so target/style transfer is visible.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "pile__public_base": "#1f77b4",
        "pile__public_lora_2601": "#2ca02c",
        "finance__public_base": "#d62728",
        "finance__public_lora_2601": "#9467bd",
    }
    x = ITERATIONS
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 13.0), sharex=True, constrained_layout=False)
    ax_acc, ax_res, ax_cost = axes
    for cell_id in CELL_ORDER:
        p = plot_data["cells"][cell_id]
        color = colors[cell_id]
        label = CELL_LABELS[cell_id]
        ax_acc.plot(x, [100.0 * y for y in p["token_accuracy"]], marker="o", color=color, label=label)
    ax_acc.set_ylabel("Token accuracy (%)")
    ax_acc.set_ylim(bottom=0)
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(ncol=2, frameon=False, loc="best")

    for key, label, linestyle, color in [
        ("continuous_residual", "continuous cycle residual", "-", "#111111"),
        ("discrete_residual", "discrete cycle residual", "--", "#e67e22"),
    ]:
        values = [[plot_data["cells"][cell][key][i] for cell in CELL_ORDER] for i in range(len(x))]
        means = [mean(v) for v in values]
        lows = [min(v) for v in values]
        highs = [max(v) for v in values]
        ax_res.plot(x, means, marker="o", linestyle=linestyle, color=color, label=label)
        ax_res.fill_between(x, lows, highs, color=color, alpha=0.10)
    ax_res.set_ylabel("Mean relative L2 residual")
    ax_res.grid(True, alpha=0.25)
    ax_res.legend(frameon=False, loc="best")

    seconds = [mean([plot_data["cells"][cell]["inference_seconds"][i] for cell in CELL_ORDER]) for i in range(len(x))]
    seconds_low = [min(plot_data["cells"][cell]["inference_seconds"][i] for cell in CELL_ORDER) for i in range(len(x))]
    seconds_high = [max(plot_data["cells"][cell]["inference_seconds"][i] for cell in CELL_ORDER) for i in range(len(x))]
    ax_cost.plot(x, seconds, marker="o", color="#111111", label="cell inference seconds (mean)")
    ax_cost.fill_between(x, seconds_low, seconds_high, color="#111111", alpha=0.10, label="cell-time range")
    ax_cost.set_ylabel("Inference seconds")
    ax_cost.grid(True, alpha=0.25)
    ax_cost_right = ax_cost.twinx()
    layers = [mean([plot_data["cells"][cell]["full_prefix_layer_evaluations"][i] for cell in CELL_ORDER]) for i in range(len(x))]
    branches = [mean([plot_data["cells"][cell]["branch_forward_calls"][i] for cell in CELL_ORDER]) for i in range(len(x))]
    ax_cost_right.plot(x, layers, marker="s", linestyle="-.", color="#1f77b4", label="full prefix layer evaluations")
    ax_cost_right.plot(x, branches, marker="^", linestyle=":", color="#d62728", label="branch evaluation calls")
    ax_cost_right.set_ylabel("Evaluation events (mean per cell)")
    handles1, labels1 = ax_cost.get_legend_handles_labels()
    handles2, labels2 = ax_cost_right.get_legend_handles_labels()
    ax_cost.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    ax_cost.set_xlabel("Fixed-point iterations")
    ax_cost.set_xticks(x)

    fig.suptitle(
        "TRR-0003 Track A checkpoint-only ladder: accuracy, residual, and cost",
        fontsize=14,
    )
    fig.text(
        0.01,
        0.012,
        "Residual/cost bands are unweighted across the four paired cells. Branch calls are reverse evaluations; full prefix layers are cycle forwards × cut depth. Legacy raw event sums remain in JSON.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.09, top=0.93, hspace=0.28)
    fig.savefig(FIGURE_PNG, dpi=220)
    fig.savefig(FIGURE_SVG)
    plt.close(fig)

    figure_records = {
        "png": file_record(FIGURE_PNG),
        "svg": file_record(FIGURE_SVG),
    }

    aggregate: dict[str, Any] = {}
    for track in ("track_a", "track_b", "comparator"):
        track_rows = [row for row in rows if row["track"] == track]
        token_values = [row["metrics"]["token_accuracy"] for row in track_rows]
        aggregate[track] = {
            "rows": len(track_rows),
            "token_accuracy_mean_unweighted": mean(token_values),
            "token_accuracy_mean_interpretation": "descriptive unweighted mean across four cells; not pooled token evidence",
            "token_accuracy_min": min(token_values),
            "token_accuracy_max": max(token_values),
            "exact_records": sum(row["metrics"]["exact_records"] for row in track_rows),
            "records": sum(row["metrics"]["records"] for row in track_rows),
        }

    generated_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary: dict[str, Any] = {
        "schema": "token-reconstruction.trr0003-cross-track-comparison.v1",
        "task_id": "TRR-0003",
        "created_utc": generated_utc,
        "status": "RETROSPECTIVE_DEVELOPMENT_PANEL_SCORED_AFTER_FREEZE",
        "comparison_scope": {
            "rows": 52,
            "methods": 13,
            "cells": 4,
            "truth_opened_for_scoring": True,
            "predictions_were_complete_before_truth_open": True,
            "canonical_dual_benchmark_complete": False,
            "score_source_is_preserved": True,
        },
        "method_order": METHOD_ORDER,
        "cell_order": CELL_ORDER,
        "method_tracks": TRACK_BY_METHOD,
        "source_artifacts": top_sources,
        "timing_source_policy": {
            "score_timing_path_used": False,
            "reason": "common_score_v2 timing_path values point into a merged-root path that is absent for this run",
            "actual_timing_binding": "each row cites the executed Track-A raw cell evidence, Track-B prediction evidence, or comparator raw cell evidence with bytes and SHA-256",
            "tracked_timing_archive_root": relative_path(ROOT / ARCHIVE_ROOT_REL),
            "branch_vs_full_prefix": "branch evaluation calls, full cycle forwards, derived full prefix layer evaluations, legacy heterogeneous event sums, public prefix calls, and candidate simulations remain separate fields",
            "full_prefix_layer_definition": "Track A full_prefix_layer_evaluations is cycle_forward_passes multiplied by cut_depth; raw public_prefix_layer_evaluations is retained as legacy heterogeneous event sum",
        },
        "timing_sources": timing_catalog,
        "retained_state_sources": retained_state_catalog,
        "prediction_artifact_sources": prediction_artifact_catalog,
        "retained_state_convention": {
            "row_state_binding": "rows with deployed learned method state carry the actual serialized resource; Track-A iteration artifacts are prediction/diagnostic outputs and have no fitted retained state",
            "state_bytes_exclude_model": True,
            "peak_memory_is_reported_separately": True,
            "prediction_artifacts_are_separate": True,
            "prediction_artifact_definition": "Track-A iteration diagnostics and per-cell comparator predictions; these are outputs, not retained method states and are excluded from State bytes",
            "track_b_shared_embedding": {
                "path": embedding_meta["path"],
                "bytes": embedding_meta["bytes"],
                "sha256": embedding_meta["sha256"],
                "source_inventory_path": top_sources["track_b_inventory"]["path"],
                "source_inventory_sha256": top_sources["track_b_inventory"]["sha256"],
            },
            "track_a_loaded_model_resource_bytes_by_cell": {
                cell: {
                    "retained_loaded_parameter_bytes": a_raw[cell]["preparation"].get("retained_loaded_parameter_bytes"),
                    "retained_model_resource_bytes": a_raw[cell]["preparation"].get("retained_model_resource_bytes"),
                }
                for cell in CELL_ORDER
            },
        },
        "aggregate": aggregate,
        "track_b_runtime_startup": track_b.get("runtime"),
        "comparator_process_timing": {
            "source_path": top_sources["comparator_run_evidence"]["path"],
            "source_sha256": top_sources["comparator_run_evidence"]["sha256"],
            "model_preparation_seconds": comparator_run.get("model", {}).get("model_preparation_seconds"),
            "model_load_wall_seconds": comparator_run.get("model", {}).get("model_load_wall_seconds"),
            "record_batch_size": comparator_run.get("record_batch_size"),
        },
        "track_a_plot_data": plot_data,
        "figure": figure_records,
        "rows": rows,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")

    def fmt(value: Any, digits: int = 6) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    def gib(value: Any) -> str:
        if value is None:
            return "—"
        return f"{float(value) / (1024**3):.2f}"

    b_rows_summary = [row for row in rows if row["track"] == "track_b"]
    a_rows_summary = [row for row in rows if row["track"] == "track_a"]
    c_rows_summary = [row for row in rows if row["track"] == "comparator"]
    direct_rows_summary = [row for row in c_rows_summary if row["method_id"] == "direct_inverse"]
    a1a2_rows_summary = [row for row in c_rows_summary if row["method_id"] == "frozen_a1_a2_k256"]
    b_min, b_max = min(r["metrics"]["token_accuracy"] for r in b_rows_summary), max(r["metrics"]["token_accuracy"] for r in b_rows_summary)
    a_min, a_max = min(r["metrics"]["token_accuracy"] for r in a_rows_summary), max(r["metrics"]["token_accuracy"] for r in a_rows_summary)
    direct_min, direct_max = min(r["metrics"]["token_accuracy"] for r in direct_rows_summary), max(r["metrics"]["token_accuracy"] for r in direct_rows_summary)
    a1a2_min, a1a2_max = min(r["metrics"]["token_accuracy"] for r in a1a2_rows_summary), max(r["metrics"]["token_accuracy"] for r in a1a2_rows_summary)
    a_i0 = [row_by_pair[(cell, A_METHODS[0])]["metrics"]["token_accuracy"] for cell in CELL_ORDER]
    a_i32 = [row_by_pair[(cell, A_METHODS[-1])]["metrics"]["token_accuracy"] for cell in CELL_ORDER]
    a_cost0 = [plot_data["cells"][cell]["inference_seconds"][0] for cell in CELL_ORDER]
    a_cost32 = [plot_data["cells"][cell]["inference_seconds"][-1] for cell in CELL_ORDER]
    a_branch32 = [plot_data["cells"][cell]["branch_forward_calls"][-1] for cell in CELL_ORDER]

    md: list[str] = []
    md.extend([
        "# TRR-0003 cross-track comparison summary",
        "",
        "This retrospective summary binds all 13 methods across the four shared cells (52 scored rows) to the frozen `common_score_v2.json` metrics and to the raw timing evidence that actually executed. It is a pilot comparison on the development panel; the canonical dual-benchmark requirement remains incomplete.",
        "",
        "## What we learned",
        "",
        f"The no-A2 Track B methods reached {b_min:.4f}–{b_max:.4f} token accuracy across the four cells, while the direct inverse comparator reached {direct_min:.4f}–{direct_max:.4f}. None of the three selected standalone methods produced a completely reconstructed record in these eight-record cells (16 distinct records per target condition across the two styles); their small public-only states do not remove the transfer gap. Historical A1 reached {min(r['metrics']['token_accuracy'] for r in c_rows_summary if r['method_id'] == 'historical_alpaca_a1'):.4f}–{max(r['metrics']['token_accuracy'] for r in c_rows_summary if r['method_id'] == 'historical_alpaca_a1'):.4f}, and frozen A1+A2 reached {a1a2_min:.4f}–{a1a2_max:.4f} at the cost of public-prefix candidate search.",
        "",
        f"Track A's checkpoint-only fixed-point ladder reached only {a_min:.4f}–{a_max:.4f} token accuracy. Increasing the ladder from zero to 32 iterations did not improve this panel: the descriptive unweighted mean across the four cell values changed from {mean(a_i0):.4f} to {mean(a_i32):.4f}, while mean cell inference time changed from {mean(a_cost0):.3f}s to {mean(a_cost32):.3f}s and branch evaluation calls rose to {mean(a_branch32):.0f} per cell. Residual and token accuracy therefore do not move together reliably; the figure keeps them as separate diagnostics.",
        "",
        "The Track B decoder timings are direct prediction intervals with exact warm repeats and zero candidate simulations/public-prefix calls. Track A's branch evaluation calls are separate from its full cycle forwards and full prefix layer evaluations. Comparator A1+A2 public-prefix calls and candidate simulations are reported separately again; no cost field is inferred from a score-side timing path.",
        "",
        "## Scope and source binding",
        "",
        f"Score source: `{top_sources['common_score_v2']['path']}` (SHA-256 `{top_sources['common_score_v2']['sha256']}`). Its 52 predictions were complete before truth scoring, and this file was read without modification. Every table row below points to a timing source ID; the source catalog gives the actual path, byte count, SHA-256, and JSON pointer. The obsolete `timing_path` values in the score cells are retained in JSON as claims for audit but were not used as timing evidence. Comparator retained states are bound to `{top_sources['comparator_bindings']['path']}` (SHA-256 `{top_sources['comparator_bindings']['sha256']}`), rather than to the per-cell prediction files; Track-A iteration files are likewise cataloged as diagnostics, not state.",
        "",
        f"Figure: `{relative_path(FIGURE_PNG)}` (SHA-256 `{figure_records['png']['sha256']}`) and `{relative_path(FIGURE_SVG)}` (SHA-256 `{figure_records['svg']['sha256']}`). Residual and cost bands are unweighted across the four paired cells; accuracy is shown per cell.",
        "",
        "## All 52 scored rows",
        "",
        "`Inference s` is the raw cell elapsed time for Track A/comparator rows and the first prediction interval for Track B. `Warm/subsequent s` is an exact same-cell repeat for Track B; for comparator rows it is the elapsed time of a later cell in the same resident process, not an exact repeat. `Full layers` is the derived Track-A full-prefix count (`cycle_forward_passes × cut_depth`); the legacy raw event sum is retained in JSON. `State bytes` is the actual serialized method-state artifact bound to that row; comparator prediction outputs are labeled separately in the `prediction` column and catalog. The fixed model and Track-B embedding are accounted separately below.",
        "",
        "| cell | method | track | token acc. | exact records | inference s | warm/subsequent s | branch evals | full cycles | full layers | public prefix calls | candidate sims | peak alloc GiB | peak reserved GiB | state bytes | prediction | timing |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for row in rows:
        timing = row["timing"]
        cost = timing["cost_events"]
        peak = timing.get("peak_memory") or {}
        exact = f"{row['metrics']['exact_records']}/{row['metrics']['records']}"
        md.append(
            "| "
            + " | ".join([
                CELL_LABELS[row["cell_id"]],
                row["method_label"],
                row["track"],
                fmt(row["metrics"]["token_accuracy"]),
                exact,
                fmt(timing.get("inference_seconds")),
                fmt(timing.get("warm_inference_seconds")),
                fmt(cost.get("branch_evaluation_calls"), 0),
                fmt(cost.get("full_cycle_forward_passes"), 0),
                fmt(cost.get("full_prefix_layer_evaluations"), 0),
                fmt(cost.get("public_prefix_calls"), 0),
                fmt(cost.get("candidate_simulations"), 0),
                gib(peak.get("cuda_peak_allocated_bytes")),
                gib(peak.get("cuda_peak_reserved_bytes")),
                fmt((row.get("retained_state") or {}).get("bytes"), 0),
                row.get("prediction_artifact", {}).get("source_id", "—"),
                timing["timing_source_id"],
            ])
            + " |"
        )
    md.extend([
        "",
        "## Timing-source catalog",
        "",
        "| source | actual path | tracked archive path | bytes | SHA-256 | JSON pointers used |",
        "| --- | --- | --- | ---: | --- | --- |",
    ])
    for source in timing_catalog:
        pointers = ", ".join(f"`{p}`" for p in source["json_pointers"])
        md.append(
            f"| {source['source_id']} | `{source['path']}` | `{source.get('archived_path')}` | {source['bytes']:,} | `{source['sha256']}` | {pointers} |"
        )
    md.extend([
        "",
        "## Retained-state catalog",
        "",
        "`State bytes` binds only to serialized resources retained by the deployed method. Track-A iteration `.safetensors` and comparator per-cell `.safetensors` files are prediction/diagnostic outputs; they are cataloged separately and excluded from retained-state accounting.",
        "",
        "| source | role | actual path | bytes | SHA-256 | usages |",
        "| --- | --- | --- | ---: | --- | --- |",
    ])
    for source in retained_state_catalog:
        usages = ", ".join(f"{u['cell_id']}/{u['method_id']}" for u in source["usages"])
        md.append(
            f"| {source['source_id']} | {source['role']} | `{source['path']}` | {source['bytes']:,} | `{source['sha256']}` | {usages} |"
        )
    md.extend([
        "",
        "## Comparator prediction-artifact catalog",
        "",
        "These files preserve Track-A iteration diagnostics and comparator predictions used for the frozen scoring inputs. They are evidence outputs, not method state; their bytes must not be interpreted as a retained footprint.",
        "",
        "| source | role | actual path | bytes | SHA-256 | usages |",
        "| --- | --- | --- | ---: | --- | --- |",
    ])
    for source in prediction_artifact_catalog:
        usages = ", ".join(f"{u['cell_id']}/{u['method_id']}" for u in source["usages"])
        md.append(
            f"| {source['source_id']} | {source['role']} | `{source['path']}` | {source['bytes']:,} | `{source['sha256']}` | {usages} |"
        )
    md.extend([
        "",
        "## Retained-state convention and standalone A1 components",
        "",
        "Each row's `retained_state` JSON object binds the serialized state actually loaded for that method/cell or Track-A iteration, with bytes and SHA-256. These bytes exclude the loaded model; peak allocated/reserved memory is reported in the row separately. Track B additionally retains the fixed normalized public embedding table at `outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors` (1,050,673,488 FP32 bytes), whose bound hash and inventory source are recorded in JSON. Comparator peak memory is a measured resident process/harness peak that includes common loaded model resources; it must not be read as a minimal method-only footprint.",
        "",
        "| cell | historical A1 proposal s | per-record s | per-token s | cold start | public prefix calls | candidate sims | state bytes |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ])
    for cell_id in CELL_ORDER:
        a1_row = row_by_pair[(cell_id, "historical_alpaca_a1")]
        comp = a1_row["timing"]
        components = comp.get("runtime_components", {})
        md.append(
            "| " + " | ".join([
                CELL_LABELS[cell_id],
                fmt(components.get("proposal_seconds")),
                fmt(components.get("per_record_seconds")),
                fmt(components.get("per_scored_token_seconds")),
                "yes" if comp.get("cold_start") else "no",
                fmt(comp["cost_events"].get("public_prefix_calls"), 0),
                fmt(comp["cost_events"].get("candidate_simulations"), 0),
                fmt((a1_row.get("retained_state") or {}).get("bytes"), 0),
            ]) + " |"
        )
    md.extend([
        "",
        "Track B's selected decoder states are compact, but all three deployment paths retain the fixed normalized public embedding table (1,050,673,488 FP32 bytes); that table dominates their retained footprint. Track B preparation, training, replay, and diagnostic costs remain in `experiments/TRR-0003/track_b/report_fragment.md`, whose phase table deliberately does not sum unlike guard scopes into one aggregate.",
        "",
        "Track A residual/cost values come from the four raw `evidence.json` files and are also cross-checked against `track_a/runtime_analysis.json`. The summary's `full layers` is the derived full-prefix count (`cycle_forward_passes × cut_depth`, 64 in this run); the raw `public_prefix_layer_evaluations` field, which mixes branch and cycle work, is retained in JSON as `legacy_public_prefix_layer_evaluations`. Comparator per-cell elapsed times come from the 12 raw `*.evidence.json` files and are cross-checked against `run_evidence.json`; A1+A2 counts are actual recorded public-prefix/candidate events. No new model execution was performed while generating this summary.",
        "",
        "These findings do not promote any method as a replacement. The next decision should test broader public token coverage for the standalone decoder and independently confirm whether the no-A2 transfer ceiling moves; direction should change if diverse held-out and shifted panels remain near this ceiling or the fixed embedding table is unacceptable.",
        "",
    ])
    MARKDOWN_PATH.write_text("\n".join(md))

    print(json.dumps({
        "summary": relative_path(SUMMARY_PATH),
        "summary_bytes": SUMMARY_PATH.stat().st_size,
        "markdown": relative_path(MARKDOWN_PATH),
        "markdown_bytes": MARKDOWN_PATH.stat().st_size,
        "figure_png": figure_records["png"],
        "figure_svg": figure_records["svg"],
        "rows": len(rows),
        "timing_sources": len(timing_catalog),
        "source_hash_checks": {
            source["path"]: source.get("hash_matches_declared")
            for source in timing_catalog
            if source.get("declared_sha256") is not None
        },
    }, indent=2))


if __name__ == "__main__":
    main()
