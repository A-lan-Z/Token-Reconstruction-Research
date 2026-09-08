#!/usr/bin/env python3
"""Create the private, create-only TRR-P11 evidence preservation copy.

The copy contains frozen receipts, predictions, A1 traces/costs, transfer
geometry and aggregates, the completed truth/score bundle, and the exact
task-local scripts.  Large regenerable observation/model arrays are omitted
deliberately and listed in the receipt.  This script never parses tensor
payloads and never prints payload values.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = Path("/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-P11")
MIN_FREE_AFTER = 10 * 1024**3
CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                return h.hexdigest()
            h.update(block)


def add_file(files: dict[str, tuple[Path, str]], rel: str, role: str) -> None:
    src = ROOT / rel
    if not src.is_file() or src.is_symlink():
        raise FileNotFoundError(f"required regular file missing: {rel}")
    files.setdefault(rel, (src, role))


def add_tree(files: dict[str, tuple[Path, str]], rel: str, role: str, *,
             suffixes: set[str] | None = None,
             exclude_suffixes: set[str] | None = None) -> None:
    base = ROOT / rel
    if not base.is_dir():
        raise FileNotFoundError(f"required directory missing: {rel}")
    for src in sorted(base.rglob("*")):
        if not src.is_file() or src.is_symlink():
            continue
        if "__pycache__" in src.parts:
            continue
        if suffixes is not None and src.suffix not in suffixes:
            continue
        if exclude_suffixes is not None and src.suffix in exclude_suffixes:
            continue
        add_file(files, src.relative_to(ROOT).as_posix(), role)


def add_glob(files: dict[str, tuple[Path, str]], rel_dir: str, pattern: str,
             role: str) -> None:
    base = ROOT / rel_dir
    if not base.is_dir():
        raise FileNotFoundError(f"required directory missing: {rel_dir}")
    matches = [p for p in base.glob(pattern) if p.is_file() and not p.is_symlink()]
    if not matches:
        raise FileNotFoundError(f"required glob has no files: {rel_dir}/{pattern}")
    for src in matches:
        add_file(files, src.relative_to(ROOT).as_posix(), role)


def disk_free(path: Path) -> int:
    return shutil.disk_usage(path).free


def copy_exclusive(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IWUSR | stat.S_IRUSR
    fd = os.open(dst, flags, mode)
    try:
        with os.fdopen(fd, "wb") as out, src.open("rb") as inp:
            while True:
                block = inp.read(CHUNK)
                if not block:
                    break
                out.write(block)
    except Exception:
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
        raise
    shutil.copystat(src, dst, follow_symlinks=False)


def build_selection() -> tuple[dict[str, tuple[Path, str]], list[dict[str, str]]]:
    files: dict[str, tuple[Path, str]] = {}
    omitted: list[dict[str, str]] = []

    # Frozen evaluation/selection and current status snapshots.
    for rel in [
        "experiments/TRR-P11/backup/private_backup_r1.py",
        "experiments/TRR-P11/manifest.json",
        "experiments/TRR-P11/final-evidence.json",
        "experiments/TRR-P11/phase-status-r1.json",
        "experiments/TRR-P11/phase-status-r2.json",
        "experiments/TRR-P11/preparation-attempt-inventory-r1.json",
        "experiments/TRR-P11/replication-provenance-r1.json",
        "experiments/TRR-P11/evidence-packaging-inventory-r1.json",
        "experiments/TRR-P11/capture/resource_plan_v1.json",
        "experiments/TRR-P11/capture/validation-r1.json",
        "experiments/TRR-P11/evaluation/actual-evaluation-setup-r1.json",
        "experiments/TRR-P11/evaluation/execution-tests-r13.json",
        "experiments/TRR-P11/evaluation/static-validation-r1.json",
        "experiments/TRR-P11/evaluation/support-binding-r1.json",
        "experiments/TRR-P11/selector/command-r3.json",
        "experiments/TRR-P11/selector/public_source_inputs_r1.json",
        "experiments/TRR-P11/selector/validation-r2.json",
        "experiments/TRR-P11/restore/restore-execution-actual-r2-attempt3.json",
        "experiments/TRR-P11/restore/restore_manifest_actual_r2.json",
        "experiments/TRR-P11/restore/restore_receipt_actual_r2.json",
        "experiments/TRR-P11/restore/resource_plan.md",
        "experiments/TRR-P11/transfer/analysis_execution_command_r2.json",
        "experiments/TRR-P11/transfer/analysis_summary_r2.json",
        "experiments/TRR-P11/transfer/analysis_summary_r3.json",
        "experiments/TRR-P11/transfer/capture_preflight_r4.json",
        "experiments/TRR-P11/transfer/opaque64_reservation_r5.json",
        "experiments/TRR-P11/exclusions/canonical_pass_binding_r1.json",
        "experiments/TRR-P11/exclusions/content_release_review_r1.json",
        "experiments/TRR-P11/exclusions/focused_test_receipt_r1.json",
        "experiments/TRR-P11/exclusions/identity_union_export_r14.json",
        "experiments/TRR-P11/exclusions/recovery_identity_audit_r17.json",
        "experiments/TRR-P11/exclusions/recovery_validation_r2.json",
        "experiments/TRR-P11/exclusions/replication_inputs_binding_r1.json",
        "experiments/TRR-P11/exclusions/root_selection_release_r1.json",
        "experiments/TRR-P11/exclusions/original_payload_recovery_handoff_r1.json",
        "experiments/TRR-P11/exclusions/original_fit_1200_identity_r1.json",
        "experiments/TRR-P11/exclusions/validation_48_identity_r1.json",
        "experiments/TRR-P11/exclusions/trr0001_trr0002_h40_recovery_handoff_r1.json",
        "coordination/STATE.json",
        "coordination/parallel/TRR-P11.json",
        "coordination/results/TRR-P11.md",
        "coordination/results/TRR-P11-replication-provenance.md",
    ]:
        add_file(files, rel, "frozen_receipt_or_status_snapshot")

    add_file(
        files,
        "outputs/TRR-P11/private-evaluation/selection/source_selection.json",
        "frozen_source_selection",
    )

    # Evaluation manifest/freeze, completed truth and score, including the
    # private truth sidecar and producer receipt now that truth is materialized.
    add_tree(
        files,
        "outputs/TRR-P11/private-evaluation/evaluation",
        "evaluation_manifest_freeze_truth_score",
        exclude_suffixes={".pyc"},
    )

    # B0/B1 restored prediction outputs and their per-cell receipts.  The
    # model/readout arrays beside them are intentionally excluded below.
    add_tree(
        files,
        "restore-runtime/actual-r2/runtime/predictions",
        "b0_b1_prediction_outputs_and_receipts",
    )
    for rel in [
        "restore-runtime/actual-r2/package_manifest.json",
        "restore-runtime/actual-r2/receipts/selection_complete_before_smoke.json",
        "restore-runtime/actual-r2/verification/original_smoke_predictions.receipt.json",
        "restore-runtime/actual-r2/verification/smoke_equivalence.json",
        "restore-runtime/actual-r2/smoke/expected_predictions.receipt.json",
        "restore-runtime/actual-r2/runtime/smoke_predictions.receipt.json",
    ]:
        add_file(files, rel, "b0_b1_restore_receipt")
    add_tree(files, "restore-runtime/actual-r2/code", "restored_runtime_exact_code")
    add_tree(files, "restore-runtime/actual-r2/reference_code", "restored_runtime_exact_code")
    add_tree(files, "restore-runtime/actual-r2/config", "restored_runtime_exact_config")
    add_tree(files, "restore-runtime/actual-r2/identity", "restored_runtime_identity_descriptors")

    # A1 predictions, frozen traces, costs, and every bounded attempt/guard
    # receipt.  No model activations outside the frozen traces are selected.
    for rel in [
        "outputs/TRR-P11/private-evaluation/a1_a2-execution-r1",
        "outputs/TRR-P11/private-evaluation/a1_a2-execution-r2",
        "outputs/TRR-P11/private-evaluation/a1_a2-execution-r3",
        "outputs/TRR-P11/private-evaluation/a1_a2-r1",
    ]:
        add_tree(files, rel, "a1_predictions_traces_costs_and_attempt_receipts")
    add_glob(
        files,
        "outputs/TRR-P11/private-evaluation",
        "a1_a2-*.json",
        "a1_preflight_and_guard_receipts",
    )

    # Transfer predictions/geometry, aggregate summaries, failures, and
    # selection/capture receipts.  Truth tensors for the transfer analysis
    # are already materialized and are retained privately here.
    add_tree(files, "outputs/TRR-P11/private-evaluation/transfer-capture", "transfer_geometry_and_receipts")
    add_tree(files, "outputs/TRR-P11/private-evaluation/transfer-analysis", "transfer_aggregate_failure_truth_receipts")
    add_tree(files, "outputs/TRR-P11/private-evaluation/transfer-selection-r1", "transfer_selection_attempt_receipts")
    add_tree(files, "outputs/TRR-P11/private-evaluation/transfer-selection-r2", "transfer_selection_receipts")

    # Confirmation capture receipts are useful to bind the source-selection
    # and primary pipeline.  The large observation arrays are regenerable and
    # intentionally omitted from this compact private copy.
    for rel in [
        "outputs/TRR-P11/private-evaluation/capture/preflight-r1.json",
        "outputs/TRR-P11/private-evaluation/capture/preflight-r2.json",
        "outputs/TRR-P11/private-evaluation/capture/confirmation-r1/failure.json",
        "outputs/TRR-P11/private-evaluation/capture/confirmation-r2/capture.json",
        "outputs/TRR-P11/private-evaluation/capture/confirmation-r2/observations.json",
        "outputs/TRR-P11/private-evaluation/capture/confirmation-r2/panel.json",
        "outputs/TRR-P11/private-evaluation/capture/public-r1/capture.json",
        "outputs/TRR-P11/private-evaluation/capture/public-r1/observations.json",
        "outputs/TRR-P11/private-evaluation/capture/public-r1/panel.json",
    ]:
        add_file(files, rel, "capture_receipt_without_large_observation_arrays")

    # Primary receipts and the exact executed P11 scripts.  Keep the
    # intentionally unexecuted prospective draft out of the backup.
    add_tree(files, "outputs/TRR-P11/private-evaluation/primary", "primary_prediction_receipts")
    add_tree(files, "scripts/trr_p11", "executed_task_script", suffixes={".py"})
    prospective = ROOT / "scripts/trr_p11/prospective_pipeline.py"
    if prospective.exists():
        files.pop("scripts/trr_p11/prospective_pipeline.py", None)
        omitted.append({
            "path": "scripts/trr_p11/prospective_pipeline.py",
            "reason": "UNEXECUTED_UNVALIDATED_DRAFT_OMITTED",
            "bytes": str(prospective.stat().st_size),
        })

    # The following arrays remain task-local or are already backed up in the
    # separate model package.  Keep the omission list explicit and sized.
    omission_roots = [
        ("outputs/TRR-P11/private-evaluation/capture", "large_confirmation_observation_arrays_regenerable"),
        ("restore-runtime/actual-r2/input", "large_restore_observation_arrays_regenerable"),
        ("restore-runtime/actual-r2/states", "model_package_already_backed_up_separately"),
        ("restore-runtime/actual-r2/readout", "model_package_already_backed_up_separately"),
        ("restore-runtime/actual-r2/smoke", "model_package_already_backed_up_separately"),
        ("restore-runtime/actual-r2/verification", "model_package_already_backed_up_separately"),
    ]
    for rel, reason in omission_roots:
        base = ROOT / rel
        if not base.exists():
            continue
        for src in sorted(base.rglob("*")):
            if src.is_file() and not src.is_symlink() and "__pycache__" not in src.parts:
                omitted.append({
                    "path": src.relative_to(ROOT).as_posix(),
                    "reason": reason,
                    "bytes": str(src.stat().st_size),
                })

    # Superseded intermediate exclusion/audit receipts are preserved by the
    # final audit/union bindings above and are not needed in the compact copy.
    exclusion_base = ROOT / "experiments/TRR-P11/exclusions"
    final_names = {Path(rel).name for rel in files if rel.startswith("experiments/TRR-P11/exclusions/")}
    for src in sorted(exclusion_base.iterdir()):
        if src.is_file() and src.name not in final_names and "__pycache__" not in src.parts:
            omitted.append({
                "path": src.relative_to(ROOT).as_posix(),
                "reason": "SUPERSEDED_INTERMEDIATE_EXCLUSION_EVIDENCE_RETAINED_IN_WORKTREE",
                "bytes": str(src.stat().st_size),
            })

    return files, omitted


def main() -> int:
    started = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if TARGET.exists():
        raise SystemExit(f"create-only refusal: backup target already exists: {TARGET}")

    files, omitted = build_selection()
    total = sum(src.stat().st_size for src, _role in files.values())
    free_before = disk_free(TARGET.parent)
    if free_before - total < MIN_FREE_AFTER:
        raise SystemExit(
            f"free-space guard failed: before={free_before} selected={total} "
            f"minimum_after={MIN_FREE_AFTER}"
        )

    # Create the destination only after the create-only and capacity guards.
    TARGET.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for rel, (src, role) in sorted(files.items()):
        before_stat = src.stat()
        src_sha = sha256_file(src)
        after_stat = src.stat()
        if (before_stat.st_size, before_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
            raise RuntimeError(f"source changed while hashing: {rel}")
        dst = TARGET / rel
        copy_exclusive(src, dst)
        dst_stat = dst.stat()
        dst_sha = sha256_file(dst)
        if dst_stat.st_size != before_stat.st_size or dst_sha != src_sha:
            raise RuntimeError(f"copy verification failed: {rel}")
        records.append({
            "path": rel,
            "role": role,
            "bytes": before_stat.st_size,
            "sha256": src_sha,
        })

    free_after = disk_free(TARGET.parent)
    if free_after < MIN_FREE_AFTER:
        raise RuntimeError(
            f"post-copy free-space guard failed: free_after={free_after} "
            f"minimum_after={MIN_FREE_AFTER}"
        )
    receipt = {
        "schema": "token-reconstruction.trr-p11-private-backup-receipt.v1",
        "task_id": "TRR-P11",
        "status": "PRIVATE_CREATE_ONLY_COPY_VERIFIED",
        "target": str(TARGET),
        "created_utc": started_utc,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_root": str(ROOT),
        "selection_count": len(records),
        "selection_bytes": sum(int(r["bytes"]) for r in records),
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
        "minimum_free_after_bytes": MIN_FREE_AFTER,
        "hash_verification": "source_sha256_equals_destination_sha256_for_every_file",
        "payload_values_printed": False,
        "remote_upload": False,
        "source_raw_values_printed": False,
        "files": records,
        "omitted": omitted,
        "omitted_bytes": sum(int(x["bytes"]) for x in omitted),
        "score_and_truth_state": "truth_and_score_artifacts_present_at_copy_time;_future_mutations_require_a_new_create_only_backup",
        "model_package_note": "model package arrays are omitted because they are already backed up separately",
    }
    receipt_path = TARGET / "backup_receipt.json"
    with receipt_path.open("x") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")
    receipt["backup_receipt_sha256"] = sha256_file(receipt_path)
    # Rewrite is safe only for the receipt just created; no evidence file is
    # overwritten.  Use a second create-only sidecar for the self-hash.
    sidecar = TARGET / "backup_receipt_self_hash.json"
    with sidecar.open("x") as fh:
        json.dump({"backup_receipt": "backup_receipt.json", "sha256": receipt["backup_receipt_sha256"]}, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # Keep a terse machine-readable completion line; no content or payload
    # values are printed.
    print(json.dumps({
        "status": receipt["status"],
        "selection_count": receipt["selection_count"],
        "selection_bytes": receipt["selection_bytes"],
        "omitted_count": len(omitted),
        "omitted_bytes": receipt["omitted_bytes"],
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
        "backup_receipt_sha256": receipt["backup_receipt_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
