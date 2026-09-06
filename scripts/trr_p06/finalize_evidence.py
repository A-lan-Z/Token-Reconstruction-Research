#!/usr/bin/env python3
"""Prepare the final TRR-P06 state, manifest, and publication inventory.

The default mode is a read-only plan.  ``--execute`` together with
``--allow-authoritative-write`` is required before this utility can update the
P06 task-local state and manifest.  It never opens raw observations or evaluator truth. It reads JSON/Markdown
metadata and stream-hashes bounded selected-state and prediction artifacts without
loading tensors.  Large observation
files and private truth are represented by already-declared metadata only.

The write order is state -> inventory -> manifest.  State contains paths but no
inventory hash; inventory contains a state hash but excludes itself and the
manifest hash; the manifest receives the final inventory hash.  This keeps the
three metadata files acyclic while preserving the prior files in Git history.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


TASK_ID = "TRR-P06"
PARENT_COMMIT = "f10f8ba438973b3cb260d41707fbb14293db9cd3"
PARENT_PR = 11
BRANCH = "task/TRR-P06"
MAX_PUBLISHED_FILE_BYTES = 100 * 1024 * 1024


class FinalizationError(RuntimeError):
    """Raised when the final metadata update cannot be made safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinalizationError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise FinalizationError(f"{description} must be a JSON object")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.finalize.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FinalizationError(f"temporary metadata path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _resolve(root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _declared_record(
    root: Path,
    declared: Mapping[str, Any],
    *,
    description: str,
    verify_hash: bool,
    allow_missing: bool = False,
) -> dict[str, Any]:
    raw_path = declared.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise FinalizationError(f"{description} path is missing")
    path = _resolve(root, raw_path)
    record = {
        "path": _display_path(root, path),
        "bytes": declared.get("bytes"),
        "sha256": declared.get("sha256"),
    }
    if not isinstance(record["bytes"], int) or record["bytes"] < 0:
        raise FinalizationError(f"{description} byte count is malformed")
    if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64:
        raise FinalizationError(f"{description} SHA-256 is malformed")
    if not path.exists():
        if allow_missing:
            record["availability"] = "declared_external_or_local_missing"
            return record
        raise FinalizationError(f"{description} is unavailable: {path}")
    if path.is_symlink() or not path.is_file():
        raise FinalizationError(f"{description} is not a regular file: {path}")
    actual_bytes = int(path.stat().st_size)
    if actual_bytes != record["bytes"]:
        raise FinalizationError(f"{description} byte count changed: {path}")
    if verify_hash:
        actual_sha = _sha256(path)
        if actual_sha != record["sha256"]:
            raise FinalizationError(f"{description} hash changed: {path}")
    else:
        record["hash_verification"] = "declared_receipt_hash_only"
    record["availability"] = "present"
    return record


def _actual_record(root: Path, raw: str | Path, *, description: str) -> dict[str, Any]:
    path = _resolve(root, raw)
    if path.is_symlink() or not path.is_file():
        raise FinalizationError(f"{description} is unavailable: {path}")
    return {
        "path": _display_path(root, path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "availability": "present",
    }


def _ref_record(root: Path, raw: str | Path, *, description: str) -> dict[str, Any]:
    """Hash a compact task artifact; reject an unexpectedly huge publication file."""

    path = _resolve(root, raw)
    if path.is_file() and path.stat().st_size > MAX_PUBLISHED_FILE_BYTES:
        raise FinalizationError(
            f"{description} exceeds the publication limit; use a declared local-only record: {path}"
        )
    return _actual_record(root, path, description=description)


def _merge_phase_flags(target: dict[str, Any]) -> None:
    target["phase_completion"] = {
        "panel_512_selected": True,
        "capture_complete": True,
        "student_fits_complete": True,
        "student_predictions_complete": True,
        "anchor_complete": True,
        "joint_freeze_complete": True,
        "evaluator_truth_access_after_freeze": True,
        "scoring_complete": True,
    }
    target["panel_512_selected"] = True
    target["capture_complete"] = True
    target["student_complete"] = True
    target["student_predictions_complete"] = True
    target["anchor_complete"] = True
    target["scoring_complete"] = True
    target["evaluator_truth_access_after_freeze"] = True


def _artifact_paths() -> dict[str, str]:
    return {
        "control_packet": "coordination/requests/TRR-P06.md",
        "plan": "experiments/TRR-P06/plan.json",
        "source_universe": "experiments/TRR-P06/setup/source_universe.frozen.json",
        "source_selection": "experiments/TRR-P06/runtime/source-selection-r1/selection.json",
        "preflight": "experiments/TRR-P06/runtime/preflight-r2/preflight.json",
        "qualification": "experiments/TRR-P06/runtime/qualification-r2/qualification_receipt.json",
        "capacity_original": "experiments/TRR-P06/runtime/capacity-r1/capacity_probe_receipt.json",
        "capacity_retention": "experiments/TRR-P06/runtime/capacity-retention-replay-r1/capacity_probe_receipt.json",
        "retention_equivalence": "experiments/TRR-P06/runtime/capacity-retention-replay-r1/retention_equivalence.json",
        "retention_watchdog": "experiments/TRR-P06/runtime/watchdog-capacity-retention-replay-r1/finish.json",
        "main_fit": "experiments/TRR-P06/runtime/main-r1/main_fit_receipt.json",
        "main_watchdog": "experiments/TRR-P06/runtime/watchdog-main-r1/finish.json",
        "capture": "experiments/TRR-P06/runtime/public-capture-r2/capture.json",
        "capture_panel": "experiments/TRR-P06/runtime/public-capture-r2/panel.json",
        "capture_observation_manifest": "experiments/TRR-P06/runtime/public-capture-r2/observations.json",
        "capture_postflight": "experiments/TRR-P06/setup/capture-postflight-r2.json",
        "capture_publication_entry": "experiments/TRR-P06/setup/public-capture-publication-entry-r2.json",
        "capture_watchdog": "experiments/TRR-P06/runtime/watchdog-public-capture-r2/finish.json",
        "capture_failure_r1": "experiments/TRR-P06/runtime/public-capture-r1/failure.json",
        "capture_launch_rejection": "experiments/TRR-P06/runtime/public-capture-r2-launch-rejection.json",
        "student_manifest": "experiments/TRR-P06/runtime/prediction_manifest.json",
        "student_run_manifest": "experiments/TRR-P06/runtime/predictions-r1/run_manifest.json",
        "student_watchdog": "experiments/TRR-P06/runtime/watchdog-predictions-r1/finish.json",
        "anchor_run_manifest": "experiments/TRR-P06/runtime/anchor-r2/run_manifest.json",
        "anchor_predictions": "experiments/TRR-P06/runtime/anchor-r2/anchor_predictions.json",
        "anchor_equivalence": "experiments/TRR-P06/runtime/anchor-r2/retry_equivalence.json",
        "anchor_watchdog": "experiments/TRR-P06/runtime/watchdog-anchor-r2/finish.json",
        "joint_freeze": "experiments/TRR-P06/runtime/freeze_receipt.json",
        "scored_results": "experiments/TRR-P06/runtime/scored-r1/results.json",
        "cost_summary": "experiments/TRR-P06/runtime/report-cost-summary.json",
        "pretruth_approval": "experiments/TRR-P06/runtime/approval_manifest.pretruth.json",
        "pre_adapter_predictions": "experiments/TRR-P06/runtime/prediction_manifest.pre-adapter.json",
        "pre_adapter_freeze": "experiments/TRR-P06/runtime/freeze_receipt.pre-adapter.json",
        "report": "coordination/results/TRR-P06.md",
    }


def _validate_inputs(root: Path, paths: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for name, raw in paths.items():
        refs[name] = _ref_record(root, raw, description=name)
    return refs


def _raw_observation_metadata(root: Path, publication_entry: Mapping[str, Any]) -> dict[str, Any]:
    raw = publication_entry.get("raw_observations")
    if not isinstance(raw, Mapping):
        raise FinalizationError("capture publication entry has no raw-observation metadata")
    files = raw.get("files")
    if not isinstance(files, list) or len(files) != 4:
        raise FinalizationError("capture publication entry must list four raw observation cells")
    result_files: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, Mapping):
            raise FinalizationError("raw observation entry is malformed")
        record = dict(item)
        record["publication"] = "EXCLUDE_LARGE_LOCAL_ONLY"
        record["availability"] = "metadata_declared_local_artifact"
        result_files.append(record)
    return {
        "publication": "EXCLUDE_LARGE_LOCAL_ONLY",
        "reason": str(raw.get("reason", "raw observation payloads are outside the publication allowlist")),
        "files": result_files,
    }


def _private_truth_metadata(score: Mapping[str, Any]) -> dict[str, Any]:
    provenance = score.get("provenance")
    if not isinstance(provenance, Mapping):
        raise FinalizationError("scored result has no provenance")
    truth_manifest = provenance.get("truth_manifest")
    truth_files = provenance.get("truth_files")
    if not isinstance(truth_manifest, Mapping) or not isinstance(truth_files, Mapping):
        raise FinalizationError("scored result has incomplete private-truth provenance")
    return {
        "publication": "EXCLUDE_PRIVATE_TRUTH_OUTSIDE_GIT",
        "scored_after_joint_freeze": provenance.get("scored_after_joint_freeze") is True,
        "truth_manifest": dict(truth_manifest),
        "truth_files": {str(key): dict(value) for key, value in truth_files.items() if isinstance(value, Mapping)},
        "content_persisted_in_repository": False,
    }


def _prediction_files(root: Path, prediction: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    student: dict[str, dict[str, Any]] = {}
    cells = prediction.get("student_cells")
    if not isinstance(cells, Mapping):
        raise FinalizationError("student prediction matrix is missing")
    for cell_id, cell in cells.items():
        if not isinstance(cell, Mapping):
            raise FinalizationError(f"student cell is malformed: {cell_id}")
        replicates = cell.get("replicates")
        if not isinstance(replicates, Mapping):
            raise FinalizationError(f"student cell replicates are missing: {cell_id}")
        for seed, methods in replicates.items():
            if not isinstance(methods, Mapping):
                raise FinalizationError(f"student methods are malformed: {cell_id}/{seed}")
            for method_id, descriptor in methods.items():
                if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("prediction"), Mapping):
                    raise FinalizationError(f"student prediction descriptor is malformed: {cell_id}/{seed}/{method_id}")
                declared = dict(descriptor["prediction"])
                key = str(declared.get("path"))
                if key not in student:
                    student[key] = _declared_record(
                        root,
                        declared,
                        description=f"student prediction {cell_id}/{seed}/{method_id}",
                        verify_hash=True,
                    )
    anchor: dict[str, dict[str, Any]] = {}
    anchors = prediction.get("anchor_cells")
    if isinstance(anchors, Mapping):
        for domain, descriptor in anchors.items():
            if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("prediction"), Mapping):
                raise FinalizationError(f"anchor prediction descriptor is malformed: {domain}")
            declared = dict(descriptor["prediction"])
            key = str(declared.get("path"))
            if key not in anchor:
                anchor[key] = _declared_record(
                    root,
                    declared,
                    description=f"anchor prediction {domain}",
                    verify_hash=True,
                )
    return {"student": list(student.values()), "anchor": list(anchor.values())}


def _selected_states(root: Path, prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    bindings = prediction.get("state_bindings")
    if not isinstance(bindings, Mapping) or len(bindings) != 6:
        raise FinalizationError("prediction manifest must bind six selected student states")
    states: list[dict[str, Any]] = []
    for key, declared in sorted(bindings.items()):
        if not isinstance(declared, Mapping):
            raise FinalizationError(f"state binding is malformed: {key}")
        states.append({
            "binding": key,
            "file": _declared_record(root, declared, description=f"selected state {key}", verify_hash=True),
        })
    return states


def _build_inventory(
    *,
    root: Path,
    refs: Mapping[str, Mapping[str, Any]],
    state_record: Mapping[str, Any],
    prediction: Mapping[str, Any],
    score: Mapping[str, Any],
    publication_entry: Mapping[str, Any],
    inventory_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    raw_obs = _raw_observation_metadata(root, publication_entry)
    prediction_files = _prediction_files(root, prediction)
    states = _selected_states(root, prediction)
    gate = score.get("gate")
    if not isinstance(gate, Mapping) or gate.get("decision") != "QUALIFIED_NEGATIVE_RETAIN_PAST_ONLY":
        raise FinalizationError("scored result does not carry the registered negative decision")
    return {
        "schema": "token-reconstruction.trr-p06-artifact-inventory.v1",
        "task_id": TASK_ID,
        "status": "FINAL_PUBLICATION_INVENTORY_METADATA_ONLY",
        "generated_utc": _utc_now(),
        "parent": {
            "commit": PARENT_COMMIT,
            "pr": PARENT_PR,
            "pr_state": "OPEN_UNMERGED",
            "branch": BRANCH,
        },
        "decision": gate["decision"],
        "phase_completion": {
            "panel_512_selected": True,
            "capture_complete": True,
            "student_fits_complete": True,
            "student_predictions_complete": True,
            "anchor_complete": True,
            "joint_freeze_complete": True,
            "evaluator_truth_access_after_freeze": True,
            "scoring_complete": True,
        },
        "publication": {
            "status": "PENDING_ROOT_PUBLICATION",
            "merge_allowed": False,
            "global_state_modified": False,
            "active_method_registry_modified": False,
            "other_agent_workspace_accessed": False,
            "manifest_path": _display_path(root, manifest_path),
            "inventory_path_excluded_from_own_artifact_list": True,
            "manifest_hash_excluded_to_avoid_cycle": True,
        },
        "state": {
            "path": _display_path(root, root / "coordination/parallel/TRR-P06.json"),
            "record": dict(state_record),
        },
        "references": {name: dict(value) for name, value in refs.items()},
        "selected_states": {
            "count": len(states),
            "tracked_publication_role": "selected student states",
            "files": states,
        },
        "prediction_files": prediction_files,
        "raw_observations": raw_obs,
        "private_truth": _private_truth_metadata(score),
        "access_boundary": {
            "source_text_or_token_values_in_inventory": False,
            "raw_observation_payloads_embedded": False,
            "truth_payload_embedded": False,
            "truth_opened_after_joint_freeze": True,
            "scored_results_payload_embedded": False,
        },
        "limitations": [
            "The four H128 observation tensors remain local-only because each exceeds the 100 MB publication limit; metadata hashes are retained.",
            "The accepted capture watchdog receipt is FAIL_CLOSED from a post-exit /proc race; no clean-watchdog claim is made.",
            "Private evaluator truth remains outside Git; only declared hashes and paths from the scoring receipt are retained.",
            "This inventory does not assert universal source disjointness beyond the frozen exclusion audit.",
        ],
        "excluded_from_inventory": [
            _display_path(root, inventory_path),
            _display_path(root, manifest_path),
            "coordination/STATE.json",
            "global active registry",
            "raw observation tensors",
            "private evaluator truth payloads",
            "model, tokenizer, Arrow, and external embedding assets",
        ],
    }


def _update_state(state: dict[str, Any], *, now: str, inventory_path: Path, manifest_path: Path) -> dict[str, Any]:
    if state.get("task_id") != TASK_ID:
        raise FinalizationError("task state has the wrong task ID")
    state = json.loads(json.dumps(state))
    _merge_phase_flags(state)
    state["status"] = "SCIENTIFIC_COMPLETE_PUBLICATION_PENDING"
    state["updated_utc"] = now
    state["records_selected"] = True
    state["truth_opened"] = True
    state["gpu_used"] = True
    state["panel_selected"] = True
    state["access_and_integrity"] = dict(state.get("access_and_integrity", {}))
    state["access_and_integrity"].update({
        "fresh_evaluation_truth_opened": True,
        "global_state_modified": False,
        "active_method_registry_modified": False,
        "agent_one_workspace_or_science_accessed": False,
        "p03_holdout_accessed": False,
        "p04_holdout_or_private_truth_accessed": False,
        "truth_opened": True,
    })
    state["publication"] = {
        "status": "PENDING_ROOT_PUBLICATION",
        "merge_allowed": False,
        "parent_pr": PARENT_PR,
        "parent_pr_state": "OPEN_UNMERGED",
        "branch": BRANCH,
        "inventory_path": _display_path(inventory_path.parent.parent.parent, inventory_path),
        "manifest_path": _display_path(manifest_path.parent.parent.parent, manifest_path),
        "global_state_unchanged": True,
        "registry_unchanged": True,
    }
    state.setdefault("setup_artifacts", {})["finalization_updater"] = "scripts/trr_p06/finalize_evidence.py"
    state["setup_artifacts"]["artifact_inventory"] = _display_path(inventory_path.parent.parent.parent, inventory_path)
    state["setup_artifacts"]["report"] = "coordination/results/TRR-P06.md"
    state["continuation"] = {
        "decision": "QUALIFIED_NEGATIVE_RETAIN_PAST_ONLY",
        "no_more_experiments": True,
        "distinct_hypothesis_required_for_future_work": True,
    }
    return state


def _update_manifest(
    manifest: dict[str, Any],
    *,
    now: str,
    refs: Mapping[str, Mapping[str, Any]],
    inventory_record: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("task_id") != TASK_ID:
        raise FinalizationError("task manifest has the wrong task ID")
    manifest = json.loads(json.dumps(manifest))
    _merge_phase_flags(manifest)
    manifest["status"] = "SCIENTIFIC_COMPLETE_PUBLICATION_PENDING"
    manifest["updated_utc"] = now
    manifest["records_selected"] = True
    manifest["truth_opened"] = True
    manifest["decision"] = score["gate"]["decision"]
    manifest["scientific_result"] = {
        "score_status": score.get("status"),
        "decision": score["gate"]["decision"],
        "harm_cleared": score["gate"].get("harm_cleared"),
        "scored_after_joint_freeze": score.get("provenance", {}).get("scored_after_joint_freeze") is True,
        "claim_scope": score.get("claim_scope"),
    }
    manifest["publication"] = {
        "status": "PENDING_ROOT_PUBLICATION",
        "merge_allowed": False,
        "parent_pr": PARENT_PR,
        "parent_pr_state": "OPEN_UNMERGED",
        "branch": BRANCH,
        "global_state_unchanged": True,
        "registry_unchanged": True,
    }
    manifest["artifact_inventory"] = dict(inventory_record)
    manifest["final_evidence_refs"] = {name: dict(value) for name, value in refs.items()}
    manifest.setdefault("task_files", {})["finalization_updater"] = {
        "path": "scripts/trr_p06/finalize_evidence.py",
        "status": "DEFERRED_REVIEWED_UPDATER",
    }
    manifest["task_files"]["report"] = dict(refs["report"])
    manifest["task_files"]["scored_results"] = dict(refs["scored_results"])
    manifest["task_files"]["joint_freeze"] = dict(refs["joint_freeze"])
    return manifest


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    state_path = _resolve(root, args.state)
    manifest_path = _resolve(root, args.manifest)
    inventory_path = _resolve(root, args.inventory)
    if state_path == (root / "coordination/STATE.json").resolve():
        raise FinalizationError("global coordination STATE is outside P06 updater scope")
    if state_path == manifest_path or state_path == inventory_path or manifest_path == inventory_path:
        raise FinalizationError("state, manifest, and inventory paths must be distinct")
    state = _load_json(state_path, description="P06 task state")
    manifest = _load_json(manifest_path, description="P06 task manifest")
    paths = _artifact_paths()
    refs = _validate_inputs(root, paths)
    prediction = _load_json(_resolve(root, paths["student_manifest"]), description="student prediction manifest")
    score = _load_json(_resolve(root, paths["scored_results"]), description="scored result")
    publication_entry = _load_json(_resolve(root, paths["capture_publication_entry"]), description="capture publication entry")
    now = args.updated_utc or _utc_now()
    state_history = state.get("historical_setup_snapshot", state)
    manifest_history = manifest.get("historical_setup_snapshot", manifest)
    state = {"task_id": TASK_ID, "schema": "token-reconstruction.parallel-state.v1", "branch": BRANCH,
             "parent": state.get("parent", {}), "control_packet": state.get("control_packet", {}),
             "historical_setup_snapshot": state_history}
    manifest = {"task_id": TASK_ID, "schema": "token-reconstruction.trr-p06-manifest.v2",
                "parent_commit": PARENT_COMMIT, "historical_setup_snapshot": manifest_history}
    state_candidate = _update_state(state, now=now, inventory_path=inventory_path, manifest_path=manifest_path)
    state_candidate["pending"] = ["publish unmerged PR"]
    state_candidate["access_and_integrity"].update({
        "panel_selected": True, "source_rows_or_plaintext_read": True, "source_token_values_read": True,
        "source_access_scope": "trusted public source producer and evaluator only; deployed student BOS-only",
        "new_data_or_target_preparation_started": True, "optimizer_steps_started": True})
    state_candidate["resource_gate"] = {
        "status": "COMPLETE_RELEASED", "gpu_jobs_started": True, "heavy_cpu_jobs_started": True,
        "paid_compute": False, "qualification_source_commit": "925759bfbf57f4167ec6feabb1512fad47bd28d0",
        "qualification_receipt": refs["qualification"], "capture_watchdog_exception": refs["capture_postflight"]}
    state_candidate["asset_inventory"] = {
        "fresh_p06_records_selected": 512, "source_universe": refs["source_universe"],
        "source_selection": refs["source_selection"], "universal_coverage_complete": False}
    state_candidate["plan_review"] = {"status": "FROZEN_EXECUTED_SCORED", "plan": refs["plan"]}
    state_record = {
        "path": _display_path(root, state_path),
        "bytes": None,
        "sha256": None,
        "written_first": True,
    }
    inventory_candidate = _build_inventory(
        root=root,
        refs=refs,
        state_record=state_record,
        prediction=prediction,
        score=score,
        publication_entry=publication_entry,
        inventory_path=inventory_path,
        manifest_path=manifest_path,
    )
    # The final inventory will receive the state hash after state is written;
    # this dry plan intentionally leaves both values null.
    manifest_candidate = _update_manifest(
        manifest,
        now=now,
        refs=refs,
        inventory_record={"path": _display_path(root, inventory_path), "bytes": None, "sha256": None},
        score=score,
    )
    manifest_candidate["task_files"]["finalization_updater"]["status"] = "ROOT_REVIEWED_EXECUTED"
    if args.publication_receipt:
        receipt = _load_json(_resolve(root, args.publication_receipt), description="publication receipt")
        if (receipt.get("task_id") != TASK_ID or receipt.get("state") != "open" or receipt.get("merged") is not False
                or receipt.get("base") != "task/TRR-0006" or receipt.get("base_sha") != PARENT_COMMIT
                or receipt.get("head") != BRANCH):
            raise FinalizationError("publication receipt does not match required unmerged P06 parent")
        pubref = _ref_record(root, args.publication_receipt, description="publication receipt")
        for candidate in (state_candidate, inventory_candidate, manifest_candidate):
            candidate["publication"].update(status="PUBLISHED_OPEN_UNMERGED", receipt=pubref, url=receipt["url"])
        state_candidate["status"] = manifest_candidate["status"] = "COMPLETE_OPEN_UNMERGED"
        state_candidate["pending"] = []
    return {
        "task_id": TASK_ID,
        "mode": "EXECUTE" if args.execute else "READ_ONLY_PLAN",
        "write_authorized": bool(args.execute and args.allow_authoritative_write),
        "write_order": [
            _display_path(root, state_path),
            _display_path(root, inventory_path),
            _display_path(root, manifest_path),
        ],
        "state_candidate": state_candidate,
        "inventory_candidate": inventory_candidate,
        "manifest_candidate": manifest_candidate,
        "large_observation_payloads_excluded": True,
        "truth_payloads_excluded": True,
    }


def execute_plan(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not args.execute or not args.allow_authoritative_write:
        return {"status": "READ_ONLY_PLAN", "write_authorized": False, "write_order": plan["write_order"]}
    root = Path(args.repository_root).expanduser().resolve()
    state_path = _resolve(root, args.state)
    manifest_path = _resolve(root, args.manifest)
    inventory_path = _resolve(root, args.inventory)
    if inventory_path.is_symlink():
        raise FinalizationError("inventory cannot be a symlink")
    if inventory_path.exists() and not args.publication_receipt:
        raise FinalizationError("existing inventory refresh requires publication receipt")
    state_candidate = dict(plan["state_candidate"])
    inventory_candidate = dict(plan["inventory_candidate"])
    manifest_candidate = dict(plan["manifest_candidate"])

    # state -> inventory -> manifest: no inventory hash is stored in state.
    _write_json(state_path, state_candidate)
    state_record = _actual_record(root, state_path, description="updated P06 task state")
    inventory_candidate["state"]["record"] = state_record
    _write_json(inventory_path, inventory_candidate)
    inventory_record = _actual_record(root, inventory_path, description="final P06 artifact inventory")
    manifest_candidate["artifact_inventory"] = inventory_record
    _write_json(manifest_path, manifest_candidate)
    return {
        "status": state_candidate["status"],
        "state": state_record,
        "inventory": inventory_record,
        "manifest": _actual_record(root, manifest_path, description="updated P06 task manifest"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--state", type=Path, default=Path("coordination/parallel/TRR-P06.json"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/TRR-P06/manifest.json"))
    parser.add_argument("--inventory", type=Path, default=Path("experiments/TRR-P06/publication-inventory.json"))
    parser.add_argument("--publication-receipt", type=Path)
    parser.add_argument("--updated-utc", help="deterministic timestamp for an authorized update")
    parser.add_argument("--execute", action="store_true", help="build and apply the metadata update")
    parser.add_argument(
        "--allow-authoritative-write",
        action="store_true",
        help="explicit second acknowledgement required for task-local state/manifest mutation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_plan(args)
        result = execute_plan(args, plan)
    except (FinalizationError, OSError, ValueError) as exc:
        print(f"TRR-P06 finalization failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
