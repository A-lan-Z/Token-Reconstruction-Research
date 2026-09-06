#!/usr/bin/env python3
"""Build and optionally apply the bounded TRR-P07 evidence finalization.

The default is a read-only plan.  Applying the update requires both
``--execute`` and ``--allow-authoritative-write``.  The helper reads only
metadata and compact prediction/tie artifacts; it never opens raw activation
observations or evaluator truth.  It requires a completed score receipt and
report before it can construct final candidates.

The write order is state -> inventory -> manifest.  State contains no
inventory hash, the inventory contains the written state hash but excludes its
own and the manifest hash, and the manifest receives the final inventory hash.
This keeps the authoritative metadata acyclic.  Existing setup state and
manifest are retained under ``historical_setup_snapshot``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

TASK_ID = "TRR-P07"
BRANCH = "task/TRR-P07"
PARENT_BRANCH = "task/TRR-P06"
PARENT_COMMIT = "02c861dfbfc63e3c0b7684a48323fd476a3b268a"
PLAN_SHA256 = "a0a2339f1a4b77e02d7d1772459dc14d442a4ce24b5111a01e58622ca1ae7c3e"
PACKET_SHA256 = "a0a9f02f4410f6833b8bb6caaba1b599a528d97935bd2ebd353ee470142b3dd7"
MAX_PUBLISHED_FILE_BYTES = 100 * 1024 * 1024
CELL_IDS = (
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
)
P06_METHOD_PREFIXES = ("p06_past_only__seed", "p06_positionwise_diagonal__seed")


class FinalizationError(RuntimeError):
    """Raised when finalization cannot prove the required metadata boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FinalizationError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise FinalizationError(f"{description} must be a JSON object")
    return dict(value)


def _resolve(root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _actual_record(root: Path, raw: str | Path, *, description: str, max_bytes: int | None = MAX_PUBLISHED_FILE_BYTES) -> dict[str, Any]:
    path = _resolve(root, raw)
    if path.is_symlink() or not path.is_file():
        raise FinalizationError(f"{description} is unavailable or not a regular file: {path}")
    size = int(path.stat().st_size)
    if max_bytes is not None and size > max_bytes:
        raise FinalizationError(f"{description} exceeds the publication artifact limit: {path}")
    return {"path": _display_path(root, path), "bytes": size, "sha256": _sha256(path)}


def _declared_record(root: Path, declared: Mapping[str, Any], *, description: str, max_bytes: int | None = MAX_PUBLISHED_FILE_BYTES) -> dict[str, Any]:
    raw_path = declared.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise FinalizationError(f"{description} path is missing")
    actual = _actual_record(root, raw_path, description=description, max_bytes=max_bytes)
    if declared.get("bytes") is not None and int(declared["bytes"]) != actual["bytes"]:
        raise FinalizationError(f"{description} byte binding changed: {actual['path']}")
    expected_sha = declared.get("sha256")
    if not isinstance(expected_sha, str) or expected_sha != actual["sha256"]:
        raise FinalizationError(f"{description} SHA-256 binding changed: {actual['path']}")
    return actual


def _ref(root: Path, path: Path, *, description: str) -> dict[str, Any]:
    return _actual_record(root, path, description=description)


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinalizationError("cannot resolve the finalizer source commit") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise FinalizationError("git HEAD is not a full lowercase commit hash")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.is_symlink():
        raise FinalizationError(f"refusing to write through symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.finalize.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FinalizationError(f"temporary finalization path already exists: {temporary}")
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


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _require_task_path(root: Path, path: Path, *, description: str) -> None:
    task_root = (root / "experiments" / TASK_ID).resolve()
    if path.resolve() != task_root and task_root not in path.resolve().parents:
        raise FinalizationError(f"{description} must remain under experiments/{TASK_ID}: {path}")


def _validate_replay(root: Path, replay_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    replay_record = _ref(root, replay_path, description="P07 replay manifest")
    replay = _load_json(replay_path, description="P07 replay manifest")
    if replay.get("schema") != "token-reconstruction.trr-p07-frozen-replay.v1" or replay.get("task_id") != TASK_ID:
        raise FinalizationError("replay manifest schema or task ID changed")
    if replay.get("status") != "FROZEN_P07_PREDICTIONS_NO_TRUTH" or replay.get("prediction_count") != 48:
        raise FinalizationError("replay manifest is not the complete frozen 48-cell matrix")
    for flag in ("truth_opened", "source_text_loaded", "target_labels_loaded", "candidate_arrays_persisted"):
        if replay.get(flag) is not False:
            raise FinalizationError(f"replay manifest has forbidden flag: {flag}")
    fixtures = replay.get("fixtures")
    if not isinstance(fixtures, Mapping) or fixtures.get("status") != "PASS" or fixtures.get("truth_opened") is not False:
        raise FinalizationError("replay fixture gate is not PASS")
    predictions = replay.get("predictions")
    if not isinstance(predictions, Mapping) or len(predictions) != 48:
        raise FinalizationError("replay prediction descriptor matrix is incomplete")
    return replay, replay_record


def _validate_score(root: Path, score_path: Path, replay_record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    score_record = _ref(root, score_path, description="P07 score result")
    score = _load_json(score_path, description="P07 score result")
    if score.get("schema") != "token-reconstruction.trr-p07-score.v1" or score.get("task_id") != TASK_ID:
        raise FinalizationError("score result schema or task ID changed")
    if score.get("status") != "TRR-P07_SCORED_AFTER_PREDICTION_FREEZE" or score.get("truth_opened") is not True:
        raise FinalizationError("score result is not a completed post-freeze score")
    if score.get("truth_payload_persisted") is not False:
        raise FinalizationError("score result claims persisted truth payload")
    freeze = score.get("prediction_freeze")
    if not isinstance(freeze, Mapping) or freeze.get("sha256") != replay_record["sha256"]:
        raise FinalizationError("score result is not bound to the frozen replay manifest")
    gate = score.get("gate")
    if not isinstance(gate, Mapping) or not isinstance(gate.get("decision") or gate.get("disposition"), str):
        raise FinalizationError("score result has no registered gate disposition")
    return score, score_record


def _validate_score_execution(root: Path, execution_path: Path, score_record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    execution_record = _ref(root, execution_path, description="P07 score execution receipt")
    execution = _load_json(execution_path, description="P07 score execution receipt")
    if execution.get("task_id") != TASK_ID or execution.get("status") != "COMPLETE_RETROSPECTIVE_SCORED_AFTER_FREEZE":
        raise FinalizationError("score execution receipt is incomplete")
    if execution.get("exit_code") != 0 or not isinstance(execution.get("code_commit"), str):
        raise FinalizationError("score execution receipt has no successful code binding")
    output = execution.get("output")
    if not isinstance(output, Mapping):
        raise FinalizationError("score execution receipt has no output binding")
    actual_output = _declared_record(root, output, description="score execution output")
    if actual_output != dict(score_record):
        raise FinalizationError("score execution output is not byte/hash bound to the score result")
    if execution.get("elapsed_seconds") is not None and not isinstance(execution.get("elapsed_seconds"), (int, float)):
        raise FinalizationError("score execution elapsed_seconds is malformed")
    if not isinstance(execution.get("timing_note"), str) or not execution["timing_note"].strip():
        raise FinalizationError("score execution must state its timing provenance")
    return execution, execution_record


def _validate_watchdog(root: Path, path: Path, *, description: str) -> dict[str, Any]:
    record = _ref(root, path, description=description)
    value = _load_json(path, description=description)
    if value.get("status") != "PASS" or value.get("child_return_code") != 0:
        raise FinalizationError(f"{description} did not PASS")
    return record


def _validate_audit(root: Path, audit_path: Path, replay_record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_record = _ref(root, audit_path, description="P07 metadata audit")
    audit = _load_json(audit_path, description="P07 metadata audit")
    if audit.get("schema") != "token-reconstruction.trr-p07-replay-metadata-audit.v1" or audit.get("status") != "PASS_METADATA_ONLY":
        raise FinalizationError("metadata audit is not the final PASS receipt")
    checks = audit.get("checks")
    if not isinstance(checks, Mapping) or not checks or any(value is not True for value in checks.values()):
        raise FinalizationError("metadata audit has a failed check")
    replay = audit.get("replay")
    if not isinstance(replay, Mapping) or replay.get("manifest", {}).get("sha256") != replay_record["sha256"]:
        raise FinalizationError("metadata audit is not bound to this replay")
    return audit, audit_record


def _validate_approvals(root: Path, *, plan_path: Path, approval_path: Path, release_path: Path, qualification_path: Path, qualification_watchdog_path: Path, freeze_path: Path) -> dict[str, dict[str, Any]]:
    plan = _ref(root, plan_path, description="canonical P07 plan")
    if plan["sha256"] != PLAN_SHA256:
        raise FinalizationError("canonical P07 plan hash changed")
    approval = _load_json(approval_path, description="root plan approval")
    if approval.get("task_id") != TASK_ID or approval.get("status") != "ROOT_APPROVED_BEFORE_NEW_PREDICTIONS_OR_SCORING" or approval.get("parent_commit") != PARENT_COMMIT:
        raise FinalizationError("root plan approval is not bound to P07")
    if approval.get("plan", {}).get("sha256") != PLAN_SHA256:
        raise FinalizationError("root plan approval does not bind the canonical plan")
    release = _load_json(release_path, description="root matrix release")
    if release.get("task_id") != TASK_ID or release.get("status") != "ROOT_RELEASED_FROZEN_MATRIX" or release.get("full_matrix_state_cells") != 48:
        raise FinalizationError("root matrix release is incomplete")
    qualification = _load_json(qualification_path, description="P07 qualification manifest")
    if qualification.get("status") != "P07_FIXTURE_AND_CELL_QUALIFICATION_PASS_NO_TRUTH" or qualification.get("truth_opened") is not False:
        raise FinalizationError("qualification manifest is not the accepted no-truth PASS")
    watchdog = _load_json(qualification_watchdog_path, description="qualification watchdog finish")
    if watchdog.get("status") != "PASS" or watchdog.get("child_return_code") != 0:
        raise FinalizationError("qualification watchdog did not PASS")
    freeze = _load_json(freeze_path, description="P07 joint freeze")
    if freeze.get("task_id") != TASK_ID or freeze.get("status") != "JOINT_FREEZE_VALIDATED_NO_TRUTH" or freeze.get("truth_opened") is not False or freeze.get("descriptor_count") != 48:
        raise FinalizationError("joint freeze receipt is incomplete")
    records = {
        "plan": plan,
        "approval": _ref(root, approval_path, description="root plan approval"),
        "matrix_release": _ref(root, release_path, description="root matrix release"),
        "qualification": _ref(root, qualification_path, description="qualification manifest"),
        "qualification_watchdog": _ref(root, qualification_watchdog_path, description="qualification watchdog"),
        "joint_freeze": _ref(root, freeze_path, description="joint freeze receipt"),
    }
    return records


def _selected_rows_summary(panel: str, selected: Any) -> dict[str, Any]:
    if not isinstance(selected, list) or len(selected) != 256 or any(isinstance(value, bool) or not isinstance(value, int) for value in selected):
        raise FinalizationError(f"{panel} selected-row metadata is malformed")
    expected = list(range(256)) if panel == "p06_panel" else list(range(0, 1536, 6))
    if selected != expected:
        raise FinalizationError(f"{panel} selected-row rule changed")
    encoded = json.dumps(selected, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "rule": "contiguous_zero_through_255" if panel == "p06_panel" else "published_trr0006_rows_6k_k0_through_255",
        "count": 256,
        "first": selected[0],
        "last": selected[-1],
        "indices_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _prediction_inventory(root: Path, replay: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = replay.get("predictions")
    if not isinstance(raw, Mapping) or len(raw) != 48:
        raise FinalizationError("replay prediction descriptors are incomplete")
    entries: list[dict[str, Any]] = []
    state_records: dict[str, dict[str, Any]] = {}
    for key in sorted(raw):
        parts = key.split("::")
        if len(parts) != 3:
            raise FinalizationError(f"prediction descriptor key is malformed: {key}")
        panel, cell_id, method = parts
        descriptor = raw[key]
        if not isinstance(descriptor, Mapping):
            raise FinalizationError(f"prediction descriptor is malformed: {key}")
        prediction = descriptor.get("prediction")
        ties = descriptor.get("tie_counts")
        state = descriptor.get("state")
        if not isinstance(prediction, Mapping) or not isinstance(ties, Mapping) or not isinstance(state, Mapping):
            raise FinalizationError(f"prediction/tie/state bindings are incomplete: {key}")
        pred_record = _declared_record(root, prediction, description=f"prediction {key}")
        tie_record = _declared_record(root, ties, description=f"tie counts {key}")
        state_record = _declared_record(root, state, description=f"state {key}")
        state_records[state_record["sha256"]] = state_record
        timing = descriptor.get("timing")
        if not isinstance(timing, Mapping):
            raise FinalizationError(f"timing is missing: {key}")
        selected = _selected_rows_summary(panel, timing.get("selected_row_indices"))
        shape = descriptor.get("shape")
        if descriptor.get("records") != 256 or shape != [256, 128]:
            raise FinalizationError(f"prediction geometry changed: {key}")
        expected_execution = "p06_batch8_chunked_full_vocab" if method.startswith(P06_METHOD_PREFIXES) else "trr0006_native_one_record_full_logits"
        if timing.get("execution") != expected_execution:
            raise FinalizationError(f"execution path changed: {key}")
        entries.append({
            "key": key,
            "panel": panel,
            "cell_id": cell_id,
            "method": method,
            "records": 256,
            "shape": shape,
            "selected_rows": selected,
            "prediction": pred_record,
            "tie_counts": tie_record,
            "state": state_record,
            "prediction_tensor_sha256": descriptor.get("prediction_tensor_sha256"),
            "tie_counts_tensor_sha256": descriptor.get("tie_counts_tensor_sha256"),
            "tie_summary": descriptor.get("tie_summary"),
            "timing": {
                key: timing.get(key)
                for key in ("execution", "batch_records", "projection_chunk", "measured_mean_seconds", "measured_seconds_sum", "measured_ms_per_record", "observation_load_seconds", "repeat_prediction_exact")
                if key in timing
            },
        })
    return {"count": len(entries), "states": list(state_records.values()), "files": entries}, entries


def _failure_records(root: Path, scoring_attempt: Path | None) -> list[dict[str, Any]]:
    if scoring_attempt is None or not scoring_attempt.exists():
        return []
    record = _ref(root, scoring_attempt, description="excluded scoring attempt")
    value = _load_json(scoring_attempt, description="excluded scoring attempt")
    return [{
        "record": record,
        "status": value.get("status"),
        "error": value.get("error"),
        "exit_code": value.get("exit_code"),
        "truth_arrays_opened": value.get("truth_arrays_opened"),
        "scored_output_created": value.get("scored_output_created"),
        "disposition": "excluded_failure_preserved_before_authoritative_score",
    }]


def _receipt_info(root: Path, path: Path | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    receipt = _load_json(path, description="publication receipt")
    if receipt.get("task_id") != TASK_ID or receipt.get("merged") is True:
        raise FinalizationError("publication receipt is not an unmerged P07 receipt")
    if receipt.get("base_sha") not in (None, PARENT_COMMIT) or receipt.get("base") not in (None, PARENT_BRANCH) or receipt.get("head") not in (None, BRANCH):
        raise FinalizationError("publication receipt branch binding changed")
    record = _ref(root, path, description="publication receipt")
    return receipt, record


def _publication(root: Path, inventory_path: Path, manifest_path: Path, receipt: Mapping[str, Any] | None, receipt_record: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "PUBLISHED_OPEN_UNMERGED" if receipt is not None else "PENDING_ROOT_PUBLICATION",
        "merge_allowed": False,
        "pr_status": "OPEN_UNMERGED" if receipt is not None else "NOT_CREATED",
        "branch": BRANCH,
        "parent_branch": PARENT_BRANCH,
        "parent_commit": PARENT_COMMIT,
        "inventory_path": _display_path(root, inventory_path),
        "manifest_path": _display_path(root, manifest_path),
        "global_state_unchanged": True,
        "active_method_registry_unchanged": True,
    }
    if receipt is not None:
        result["receipt"] = dict(receipt_record or {})
        if isinstance(receipt.get("url"), str):
            result["url"] = receipt["url"]
    return result


def _build_candidates(*, root: Path, state_path: Path, manifest_path: Path, inventory_path: Path, replay_path: Path, replay_run_path: Path, replay_watchdog_path: Path, score_path: Path, score_execution_path: Path, score_review_path: Path | None, score_review_report_path: Path | None, checkpoint_differences_json_path: Path, checkpoint_differences_md_path: Path, report_path: Path, audit_path: Path, plan_path: Path, approval_path: Path, release_path: Path, qualification_path: Path, qualification_watchdog_path: Path, freeze_path: Path, scoring_attempt: Path | None, publication_receipt: Path | None, updated_utc: str | None, authoritative_write_requested: bool) -> dict[str, Any]:
    if state_path.resolve() == (root / "coordination/STATE.json").resolve():
        raise FinalizationError("global coordination STATE is outside P07 finalization scope")
    _require_task_path(root, manifest_path, description="P07 manifest")
    _require_task_path(root, inventory_path, description="P07 inventory")
    for path, description in ((state_path, "P07 parallel state"), (manifest_path, "P07 manifest"), (inventory_path, "P07 inventory")):
        if path.is_symlink():
            raise FinalizationError(f"{description} cannot be a symlink: {path}")
    old_state = _load_json(state_path, description="existing P07 parallel state")
    old_manifest = _load_json(manifest_path, description="existing P07 manifest")
    replay, replay_record = _validate_replay(root, replay_path)
    replay_run_record = _ref(root, replay_run_path, description="P07 replay run manifest")
    replay_watchdog_record = _validate_watchdog(root, replay_watchdog_path, description="P07 replay watchdog")
    score, score_record = _validate_score(root, score_path, replay_record)
    score_execution, score_execution_record = _validate_score_execution(root, score_execution_path, score_record)
    audit, audit_record = _validate_audit(root, audit_path, replay_record)
    approval_refs = _validate_approvals(root, plan_path=plan_path, approval_path=approval_path, release_path=release_path, qualification_path=qualification_path, qualification_watchdog_path=qualification_watchdog_path, freeze_path=freeze_path)
    report_record = _ref(root, report_path, description="P07 final report")
    report_text = report_path.read_text(encoding="utf-8")
    if not report_text.strip() or "PENDING_PREDICTION_FREEZE_AND_SCORE" in report_text or "Status: **PENDING" in report_text:
        raise FinalizationError("P07 final report is still a setup placeholder")
    checkpoint_differences_json_record = _ref(root, checkpoint_differences_json_path, description="checkpoint differences JSON")
    checkpoint_differences_md_record = _ref(root, checkpoint_differences_md_path, description="checkpoint differences report")
    score_review_record = None
    if score_review_path is not None:
        score_review_record = _ref(root, score_review_path, description="independent score review JSON")
    score_review_report_record = None
    if score_review_report_path is not None:
        score_review_report_record = _ref(root, score_review_report_path, description="independent score review report")
    receipt, receipt_record = _receipt_info(root, publication_receipt)
    prediction_inventory, prediction_entries = _prediction_inventory(root, replay)
    failures = _failure_records(root, scoring_attempt)
    now = updated_utc or _utc_now()
    history_state = old_state.get("historical_setup_snapshot", old_state)
    history_manifest = old_manifest.get("historical_setup_snapshot", old_manifest)
    finalizer_record = _actual_record(root, Path(__file__), description="P07 finalizer source")
    source_commits = {
        "parent_commit": PARENT_COMMIT,
        "replay_code_commit": replay.get("code_commit"),
        "qualification_code_commit": _load_json(qualification_path, description="qualification manifest").get("code_commit"),
        "score_code_commit": score_execution.get("code_commit"),
        "finalizer_current_head": _git_head(root),
    }
    refs = {
        "control_packet": _ref(root, root / "coordination/requests/TRR-P07.md", description="P07 control packet"),
        "canonical_plan": approval_refs["plan"],
        "root_plan_approval": approval_refs["approval"],
        "root_matrix_release": approval_refs["matrix_release"],
        "qualification": approval_refs["qualification"],
        "qualification_watchdog": approval_refs["qualification_watchdog"],
        "joint_freeze": approval_refs["joint_freeze"],
        "replay_manifest": replay_record,
        "replay_run_manifest": replay_run_record,
        "replay_watchdog": replay_watchdog_record,
        "replay_audit": audit_record,
        "score_result": score_record,
        "score_execution": score_execution_record,
        "checkpoint_differences_json": checkpoint_differences_json_record,
        "checkpoint_differences_report": checkpoint_differences_md_record,
        "report": report_record,
        "finalizer_source": finalizer_record,
    }
    if scoring_attempt is not None and scoring_attempt.exists():
        refs["excluded_scoring_attempt"] = _ref(root, scoring_attempt, description="excluded scoring attempt")
    if score_review_record is not None:
        refs["independent_score_review"] = score_review_record
    if score_review_report_record is not None:
        refs["independent_score_review_report"] = score_review_report_record
    publication = _publication(root, inventory_path, manifest_path, receipt, receipt_record)
    gate = _copy_json(score["gate"])
    gate_status = gate.get("decision") or gate.get("disposition")
    audit_costs = _copy_json(audit.get("costs", {}))
    audit_resources = _copy_json(audit.get("resources", {}))
    phase_completion = {
        "prediction_replay_complete": True,
        "joint_freeze_complete": True,
        "scoring_complete": True,
        "report_complete": True,
    }
    access = _copy_json(old_state.get("access_and_integrity", {}))
    access.update({
        "gpu_jobs_started": True,
        "metadata_only_setup": False,
        "fit_or_optimizer_steps_started": False,
        "fresh_records_selected": False,
        "truth_accessed_by_p07": True,
        "truth_accessed_after_joint_freeze": True,
        "truth_payload_persisted_in_repository": False,
        "source_text_or_token_values_loaded_by_decoder": False,
        "global_state_modified": False,
        "active_method_registry_modified": False,
        "p03_holdout_accessed": False,
        "trr0007_or_unpublished_data_accessed": False,
    })
    setup_artifacts = _copy_json(old_state.get("setup_artifacts", {}))
    setup_artifacts.update({
        "finalizer_source": {"path": finalizer_record["path"], "status": "EXECUTED" if authoritative_write_requested else "READY_FOR_AUTHORITATIVE_EXECUTION"},
        "final_report": report_record,
        "score_result": score_record,
        "metadata_audit": audit_record,
    })
    state_candidate = {
        "task_id": TASK_ID,
        "schema": "token-reconstruction.trr-p07-state.v1",
        "branch": BRANCH,
        "parent": _copy_json(old_state.get("parent", {"branch": PARENT_BRANCH, "commit": PARENT_COMMIT})),
        "control_packet": _copy_json(old_state.get("control_packet", {})),
        "historical_setup_snapshot": _copy_json(history_state),
        "status": "COMPLETE_OPEN_UNMERGED" if receipt is not None else "SCIENTIFIC_COMPLETE_PUBLICATION_PENDING",
        "updated_utc": now,
        "phase_completion": phase_completion,
        "access_and_integrity": access,
        "comparison_plan": {**_copy_json(old_state.get("comparison_plan", {})), "status": "FROZEN_EXECUTED_SCORED", "record_count_per_panel_domain": 256},
        "selection": {**_copy_json(old_state.get("selection", {})), "trr0006_subset_inference_rows_used": True, "subset_rule_verified_against_score": True},
        "truth_status": "SCORED_AFTER_JOINT_FREEZE_PRIVATE_PAYLOADS_OUTSIDE_REPOSITORY",
        "scientific_result": {"score_status": score["status"], "gate": gate, "claim_scope": score.get("claim_scope"), "gate_status": gate_status, "score_result": score_record, "report": report_record},
        "source_commits": source_commits,
        "costs": {"audit": audit_costs, "resources": audit_resources, "timing_use": "descriptive native-path costs only; not a cross-path performance recommendation"},
        "failures": failures,
        "provenance": refs,
        "setup_artifacts": setup_artifacts,
        "publication": publication,
        "pending": [] if receipt is not None else ["root PR publication review"],
        "asset_inventory": {"path": _display_path(root, inventory_path), "status": "FINAL_METADATA_ONLY_INVENTORY_WRITTEN_AFTER_SCORE"},
    }
    inventory_candidate = {
        "schema": "token-reconstruction.trr-p07-artifact-inventory.v1",
        "task_id": TASK_ID,
        "status": "FINAL_PUBLICATION_INVENTORY_METADATA_ONLY",
        "generated_utc": now,
        "parent": {"branch": PARENT_BRANCH, "commit": PARENT_COMMIT},
        "publication": {**publication, "inventory_path_excluded_from_own_artifact_list": True, "manifest_hash_excluded_to_avoid_cycle": True},
        "state": {"path": _display_path(root, state_path), "record": {"path": _display_path(root, state_path), "bytes": None, "sha256": None}},
        "references": refs,
        "source_commits": source_commits,
        "prediction_files": prediction_inventory,
        "costs": audit_costs,
        "resources": audit_resources,
        "score_summary": {"status": score["status"], "gate": gate, "truth_opened": True, "truth_payload_persisted": False},
        "failures": failures,
        "access_boundary": {"raw_observations_embedded": False, "private_truth_embedded": False, "source_text_or_token_values_embedded": False, "score_payload_embedded": False},
        "setup_history": {"state_snapshot_retained": True, "manifest_snapshot_retained": True, "setup_artifacts_not_rewritten": True},
        "excluded_from_inventory": [_display_path(root, inventory_path), _display_path(root, manifest_path), "coordination/STATE.json", "raw H128 observation tensors", "private evaluator truth payloads", "large external model/embedding assets"],
    }
    manifest_candidate = {
        "task_id": TASK_ID,
        "schema": "token-reconstruction.trr-p07-manifest.v2",
        "branch": BRANCH,
        "parent": _copy_json(old_manifest.get("parent", {"branch": PARENT_BRANCH, "commit": PARENT_COMMIT})),
        "control_packet": _copy_json(old_manifest.get("control_packet", {})),
        "historical_setup_snapshot": _copy_json(history_manifest),
        "status": "COMPLETE_OPEN_UNMERGED" if receipt is not None else "SCIENTIFIC_COMPLETE_PUBLICATION_PENDING",
        "updated_utc": now,
        "phase_completion": phase_completion,
        "canonical_plan": refs["canonical_plan"],
        "scientific_scope": {**_copy_json(old_manifest.get("scientific_scope", {})), "status": "SCORED_AFTER_JOINT_FREEZE", "decision_status": gate_status, "truth_status": "SCORED_AFTER_JOINT_FREEZE_PRIVATE_PAYLOADS_OUTSIDE_REPOSITORY"},
        "access_and_integrity": access,
        "scientific_result": {"score_status": score["status"], "gate": gate, "claim_scope": score.get("claim_scope"), "scored_after_joint_freeze": True},
        "source_commits": source_commits,
        "costs": {"audit": audit_costs, "resources": audit_resources, "timing_use": "descriptive native-path costs only; not a cross-path performance recommendation"},
        "failures": failures,
        "final_evidence_refs": refs,
        "artifact_inventory": {"path": _display_path(root, inventory_path), "bytes": None, "sha256": None},
        "publication": publication,
        "limitations": ["Exploratory retrospective comparison under the frozen P07 matrix; gate outcome does not automatically promote a method.", "Raw activation observations and private truth remain outside the repository and are represented by metadata references only.", "P06 batch and retained native timing boundaries are descriptive and are not pooled into a performance recommendation.", "No fitting, fresh records, target preparation, or P03/TRR-0007 holdout access occurred in P07."],
        "setup_history": {"state_snapshot_retained": True, "manifest_snapshot_retained": True, "setup_artifacts_not_rewritten": True},
        "task_files": {"finalizer": finalizer_record, "report": report_record, "score": score_record, "score_execution": score_execution_record, "replay": replay_record, "audit": audit_record},
    }
    return {"write_order": [_display_path(root, state_path), _display_path(root, inventory_path), _display_path(root, manifest_path)], "state_path": state_path, "inventory_path": inventory_path, "manifest_path": manifest_path, "state_candidate": state_candidate, "inventory_candidate": inventory_candidate, "manifest_candidate": manifest_candidate, "score": score, "replay": replay}


def _execute(root: Path, plan: Mapping[str, Any], *, allow: bool, execute: bool, publication_receipt: Path | None) -> dict[str, Any]:
    if not execute or not allow:
        return {"status": "READ_ONLY_PLAN", "write_authorized": False, "write_order": plan["write_order"]}
    inventory_path = Path(plan["inventory_path"])
    if inventory_path.exists() and publication_receipt is None:
        raise FinalizationError("refreshing an existing inventory requires --publication-receipt")
    state_path = Path(plan["state_path"]); manifest_path = Path(plan["manifest_path"])
    _write_json(state_path, plan["state_candidate"])
    state_record = _actual_record(root, state_path, description="updated P07 state")
    inventory_candidate = _copy_json(plan["inventory_candidate"])
    inventory_candidate["state"]["record"] = state_record
    _write_json(inventory_path, inventory_candidate)
    inventory_record = _actual_record(root, inventory_path, description="P07 final inventory")
    manifest_candidate = _copy_json(plan["manifest_candidate"])
    manifest_candidate["artifact_inventory"] = inventory_record
    _write_json(manifest_path, manifest_candidate)
    return {"status": plan["state_candidate"]["status"], "write_authorized": True, "state": state_record, "inventory": inventory_record, "manifest": _actual_record(root, manifest_path, description="updated P07 manifest")}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--state", type=Path, default=Path("coordination/parallel/TRR-P07.json"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/TRR-P07/manifest.json"))
    parser.add_argument("--inventory", type=Path, default=Path("experiments/TRR-P07/publication-inventory.json"))
    parser.add_argument("--replay-manifest", type=Path, default=Path("experiments/TRR-P07/runtime/replay-r1/replay_manifest.json"))
    parser.add_argument("--replay-run", type=Path, default=Path("experiments/TRR-P07/runtime/replay-r1/run_manifest.json"))
    parser.add_argument("--replay-watchdog", type=Path, default=Path("experiments/TRR-P07/runtime/watchdog-replay-r1/finish.json"))
    parser.add_argument("--score", type=Path, default=Path("experiments/TRR-P07/runtime/scored-r2/results.json"))
    parser.add_argument("--score-execution", type=Path, default=Path("experiments/TRR-P07/runtime/scored-r2/execution.json"))
    parser.add_argument("--score-review", type=Path, default=Path("experiments/TRR-P07/review/independent-score-review.json"), help="independent score review metadata")
    parser.add_argument("--score-review-report", type=Path, default=Path("experiments/TRR-P07/review/independent-score-review.md"), help="independent score review report")
    parser.add_argument("--checkpoint-differences-json", type=Path, default=Path("experiments/TRR-P07/review/checkpoint-differences.json"))
    parser.add_argument("--checkpoint-differences-report", type=Path, default=Path("experiments/TRR-P07/review/checkpoint-differences.md"))
    parser.add_argument("--report", type=Path, default=Path("coordination/results/TRR-P07.md"))
    parser.add_argument("--audit", type=Path, default=Path("experiments/TRR-P07/runtime/replay-r1/metadata_audit_final.json"))
    parser.add_argument("--plan", type=Path, default=Path("experiments/TRR-P07/plan.json"))
    parser.add_argument("--plan-approval", type=Path, default=Path("experiments/TRR-P07/review/root-plan-approval.json"))
    parser.add_argument("--matrix-release", type=Path, default=Path("experiments/TRR-P07/review/root-matrix-release.json"))
    parser.add_argument("--qualification", type=Path, default=Path("experiments/TRR-P07/runtime/qualification-r1/qualification_manifest.json"))
    parser.add_argument("--qualification-watchdog", type=Path, default=Path("experiments/TRR-P07/runtime/watchdog-qualification-r1/finish.json"))
    parser.add_argument("--joint-freeze", type=Path, default=Path("experiments/TRR-P07/runtime/root-joint-freeze.json"))
    parser.add_argument("--scoring-attempt", type=Path, default=Path("experiments/TRR-P07/runtime/scoring-attempt-r1.json"))
    parser.add_argument("--publication-receipt", type=Path)
    parser.add_argument("--updated-utc")
    parser.add_argument("--execute", action="store_true", help="apply the state -> inventory -> manifest update")
    parser.add_argument("--allow-authoritative-write", action="store_true", help="explicit acknowledgement for task-local state/manifest mutation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repository_root).expanduser().resolve()
    try:
        paths = {name: _resolve(root, value) for name, value in {
            "state": args.state, "manifest": args.manifest, "inventory": args.inventory, "replay": args.replay_manifest, "replay_run": args.replay_run, "replay_watchdog": args.replay_watchdog, "score": args.score, "score_execution": args.score_execution, "checkpoint_differences_json": args.checkpoint_differences_json, "checkpoint_differences_md": args.checkpoint_differences_report, "report": args.report, "audit": args.audit, "plan": args.plan, "approval": args.plan_approval, "release": args.matrix_release, "qualification": args.qualification, "qualification_watchdog": args.qualification_watchdog, "freeze": args.joint_freeze, "scoring_attempt": args.scoring_attempt,
        }.items()}
        receipt = _resolve(root, args.publication_receipt) if args.publication_receipt is not None else None
        score_review = _resolve(root, args.score_review) if args.score_review is not None else None
        score_review_report = _resolve(root, args.score_review_report) if args.score_review_report is not None else None
        plan = _build_candidates(root=root, state_path=paths["state"], manifest_path=paths["manifest"], inventory_path=paths["inventory"], replay_path=paths["replay"], replay_run_path=paths["replay_run"], replay_watchdog_path=paths["replay_watchdog"], score_path=paths["score"], score_execution_path=paths["score_execution"], score_review_path=score_review, score_review_report_path=score_review_report, checkpoint_differences_json_path=paths["checkpoint_differences_json"], checkpoint_differences_md_path=paths["checkpoint_differences_md"], report_path=paths["report"], audit_path=paths["audit"], plan_path=paths["plan"], approval_path=paths["approval"], release_path=paths["release"], qualification_path=paths["qualification"], qualification_watchdog_path=paths["qualification_watchdog"], freeze_path=paths["freeze"], scoring_attempt=paths["scoring_attempt"], publication_receipt=receipt, updated_utc=args.updated_utc, authoritative_write_requested=args.execute and args.allow_authoritative_write)
        result = _execute(root, plan, allow=args.allow_authoritative_write, execute=args.execute, publication_receipt=receipt)
    except (FinalizationError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"TRR-P07 finalization failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
