#!/usr/bin/env python3
"""Build the TRR-0001-R1 structured evidence handoff and relay state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CHARTER_SHA256 = "ab0fbe9dfad39eddee48c14f4cb8201f8c3f02d1c58668d8a8e59be5a250700d"
REQUEST_SHA256 = "a86e9342df732291052cc8b69599217140ee7bba81a79351dd70f026a6f2360c"
PLAN_SHA256 = "59944cb1e01ec2e88e04109e46db0088eece74500fe599eb6a62ead038b6fe14"
PREREGISTRATION_COMMIT = "4169e4444dd99ebdc81a9adbbb9610dd8afc9555"
IMPLEMENTATION_COMMIT = "970fdf6c65883bb93878c8458141f1e0bcef3085"
AUDIT_CODE_COMMIT = "c2145618fff977c0da864b87324f4e7967792246"
PREVIOUS_REVIEWED_HEAD = "2a89a578921afc48dc3b54c65a2c39c35c422134"
PACKET_ID = "TRR-PACKET-TRR-0001-REVIEW-R1-20260822-1E1E1C00AFB4A8BC58B157BB"
METHODS = ("direct_inverse", "causal_public_surrogate_search")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--updated-utc", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(root: Path, path: Path | str) -> dict[str, Any]:
    value = Path(path)
    absolute = value if value.is_absolute() else root / value
    return {
        "path": absolute.relative_to(root).as_posix(),
        "bytes": absolute.stat().st_size,
        "sha256": sha256(absolute),
    }


def gnu_time_command(path: Path) -> str:
    prefix = 'Command being timed: "'
    first = path.read_text(encoding="utf-8").splitlines()[0].lstrip()
    if not first.startswith(prefix) or not first.endswith('"'):
        raise RuntimeError(f"GNU-time command is malformed: {path}")
    return first[len(prefix) : -1]


def write_json(path: Path, value: Any, *, create_only: bool = False) -> None:
    if create_only and path.exists():
        raise RuntimeError(f"create-only output exists: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    revision = root / "experiments/TRR-0001/revision-r1"
    manifest_path = revision / "manifest.json"
    parent_path = root / "experiments/TRR-0001/manifest.json"
    state_path = root / "coordination/STATE.json"

    request = root / "coordination/requests/TRR-0001-R1.md"
    charter = root / "RESEARCH_CHARTER.md"
    plan_path = revision / "plan.json"
    if sha256(request) != REQUEST_SHA256 or request.stat().st_size != 26821:
        raise RuntimeError("verbatim R1 request identity changed")
    if sha256(charter) != CHARTER_SHA256 or sha256(plan_path) != PLAN_SHA256:
        raise RuntimeError("charter or committed R1 plan identity changed")

    plan = load_json(plan_path)
    commitment = load_json(revision / "selection_commitment.json")
    receipt = load_json(revision / "freeze_receipt.json")
    reveal = load_json(revision / "selection_reveal.json")
    selection_verification = load_json(revision / "selection_verification.json")
    metrics = load_json(revision / "metrics.json")
    score = load_json(revision / "score_evidence.json")
    audit_preparation = load_json(revision / "original_noninterference_preparation.json")
    audit = load_json(revision / "original_noninterference_audit.json")
    final_validation = load_json(revision / "final_validation.json")
    preregistration_attempts = load_json(revision / "preregistration/attempts.json")

    if commitment["commitment"] != plan["data"]["selection"]["commitment_digest"]:
        raise RuntimeError("selection commitment differs from preregistration")
    if receipt["preregistration_commit"] != PREREGISTRATION_COMMIT:
        raise RuntimeError("freeze receipt preregistration identity changed")
    if selection_verification["result"] != "PASS_SELECTION_REVEALED_AFTER_VERIFIED_FREEZE":
        raise RuntimeError("selection verification is not passing")
    if final_validation["result"] != "PASS_ALL_R1_INTEGRITY_CHECKS":
        raise RuntimeError("final R1 validator is not passing")
    if audit["result"] != "PASS_ORIGINAL_IDENTIFIER_NONINTERFERENCE":
        raise RuntimeError("original-run noninterference audit is not passing")

    frozen_root = root / receipt["frozen_root"]
    reconstructor_evidence = {
        method: load_json(frozen_root / method / "reconstructor_evidence.json")
        for method in METHODS
    }
    access_manifests = {
        method: load_json(frozen_root / method / "access_manifest.json")
        for method in METHODS
    }
    access_records = {
        method: file_record(root, frozen_root / method / "access_manifest.json")
        for method in METHODS
    }
    for method, value in access_manifests.items():
        if (
            value["result"] != "PASS_FAIL_CLOSED_ACCESS_BOUNDARY"
            or len(value["denial_probes"]) != 7
            or value["network"]["default_route_present"] is not False
        ):
            raise RuntimeError(f"{method} isolation evidence is not passing")

    revision_files = []
    for path in sorted(revision.rglob("*")):
        if path.is_file() and path != manifest_path:
            revision_files.append(file_record(root, path))

    local_artifacts = {
        "evaluator_evidence": file_record(
            root, "outputs/TRR-0001-R1/clean/evaluator_private/evaluator_evidence.json"
        ),
        "blind_truth": file_record(
            root, "outputs/TRR-0001-R1/clean/evaluator_private/blind_truth.jsonl"
        ),
        "private_selection": file_record(
            root, "outputs/TRR-0001-R1/evaluator_private/fresh_selection_private.json"
        ),
        "frequency_counts": file_record(
            root,
            "outputs/TRR-0001-R1/clean/evaluator_private/auxiliary_frequency_counts.json",
        ),
        "frozen_entries": receipt["entries"],
        "retention": "local raw artifacts retained at the recorded paths; frozen entries are read-only",
    }

    exact_commands = {
        name: gnu_time_command(revision / f"{name}-time.txt")
        for name in (
            "evaluator",
            "direct",
            "causal",
            "freeze",
            "reveal",
            "score",
            "noninterference-prepare",
            "noninterference-rerun",
            "noninterference-compare",
            "pycompile",
            "focused-tests",
            "full-tests",
            "final-validation",
        )
    }

    access_summary = {}
    for method in METHODS:
        value = access_manifests[method]
        evidence = reconstructor_evidence[method]
        access_summary[method] = {
            "manifest": access_records[method],
            "result": value["result"],
            "started_utc": value["started_utc"],
            "ended_utc": value["ended_utc"],
            "exit_status": value["exit_status"],
            "identity": value["identity"],
            "namespaces": value["namespaces"],
            "mounts": {
                "read_only": value["mounts"]["read_only"],
                "writable": value["mounts"]["writable"],
                "mountinfo_sha256": value["mounts"]["mountinfo_sha256"],
            },
            "environment": value["environment"],
            "denial_probes": value["denial_probes"],
            "network": value["network"],
            "inner_command": evidence["command"],
            "outer_command": exact_commands["direct" if method == METHODS[0] else "causal"],
            "cost": metrics["cost"]["methods"][method],
            "fresh_training_steps": evidence["fresh_training_steps"],
            "fresh_adaptation_steps": evidence["fresh_adaptation_steps"],
        }

    manifest = {
        "schema": "token-reconstruction.trr0001-r1-manifest.v1",
        "task": {
            "id": "TRR-0001",
            "revision_id": "TRR-0001-R1",
            "title": "Repair the blind evaluation and rerun TRR-0001 cleanly",
            "status": "AWAITING_CHATGPT_REVIEW",
            "branch": "task/TRR-0001",
            "pull_request": 2,
            "merge_authorization": "NOT_GRANTED",
            "merged": False,
            "next_task_started": False,
        },
        "authority": {
            "charter": file_record(root, charter),
            "charter_expected_sha256": CHARTER_SHA256,
            "packet_id": PACKET_ID,
            "request": file_record(root, request),
            "request_line_endings": "CRLF",
            "previous_reviewed_head": PREVIOUS_REVIEWED_HEAD,
            "disposition": "REVISE",
        },
        "commits": {
            "clean_preregistration": PREREGISTRATION_COMMIT,
            "fixed_reconstruction_implementation": IMPLEMENTATION_COMMIT,
            "original_noninterference_audit_code": AUDIT_CODE_COMMIT,
            "r1_scientific_evidence": "PENDING_METADATA_BINDING",
        },
        "original_run": {
            "disposition": "ACCESS_INTERFACE_NONCOMPLIANT_ORIGINAL_RUN",
            "accepted_blind_claim_basis": False,
            "artifacts_preserved": True,
            "permitted_uses": [
                "descriptive evidence",
                "implementation debugging",
                "post-hoc source-identifier noninterference audit",
                "comparison with the clean confirmatory run",
            ],
        },
        "clean_preregistration": {
            "plan": file_record(root, plan_path),
            "resources": plan["resources"],
            "fixed_method_state": plan["fixed_method_state"],
            "baseline_families": plan["baseline_families"],
            "cuts": plan["cut_depths"],
            "conditions": plan["conditions"],
            "data": plan["data"],
            "metrics": plan["metrics"],
            "statistics": plan["statistics"],
            "claim_scope": plan["claim_scope"],
        },
        "selection_protocol": {
            "public_commitment": file_record(root, revision / "selection_commitment.json"),
            "commitment_digest": commitment["commitment"],
            "pre_freeze_selection_key_disclosed": commitment["selection_key_disclosed"],
            "pre_freeze_source_identity_disclosed": commitment["source_identity_disclosed"],
            "post_freeze_reveal": file_record(root, revision / "selection_reveal.json"),
            "reveal_verification": file_record(root, revision / "selection_verification.json"),
            "revealed_records": len(reveal["records"]),
            "disjoint_from_original_records": selection_verification["disjoint_from_original_records"],
            "token_ids_disclosed_in_mapping_reveal": selection_verification[
                "token_ids_disclosed_in_mapping_reveal"
            ],
            "truth_sidecar_read_during_reveal": selection_verification["truth_sidecar_read"],
            "result": selection_verification["result"],
        },
        "access_isolation": {
            "contract": plan["access_isolation"],
            "processes": access_summary,
            "separate_sequential_processes": True,
            "namespace_inode_note": "Linux reused lifetime-scoped namespace inode numbers after the first process exited; no simultaneous distinct-inode claim is made.",
            "public_source_resolving_fields": 0,
            "target_prefix_calls": metrics["cost"]["target_prefix_calls"],
        },
        "freeze_reveal_truth_chronology": {
            "freeze_receipt": file_record(root, revision / "freeze_receipt.json"),
            "freeze_created_utc": receipt["created_utc"],
            "frozen_root": receipt["frozen_root"],
            "frozen_entries": len(receipt["entries"]),
            "frozen_bytes": sum(entry["bytes"] for entry in receipt["entries"]),
            "runtime_identity_sha256": receipt["metadata"]["runtime_identity_sha256"],
            "selection_revealed_at_freeze": receipt["metadata"]["selection_revealed"],
            "truth_opened_at_freeze": receipt["metadata"]["truth_opened"],
            "freeze_verification": file_record(root, revision / "freeze-verification.txt"),
            "reveal_created_utc": reveal["revealed_utc"],
            "score_started_utc": score["started_utc"],
            "score_ended_utc": score["ended_utc"],
            "truth_gate": score["truth_gate"],
            "frozen_outputs_revised_after_receipt": False,
        },
        "measurements": {
            "metrics": file_record(root, revision / "metrics.json"),
            "per_record_metrics": file_record(root, revision / "per_record_metrics.jsonl"),
            "diagnostics": file_record(root, revision / "diagnostics.json"),
            "summary_csv": file_record(root, revision / "summary.csv"),
            "accuracy_plot": file_record(root, revision / "accuracy_plot.svg"),
            "records": metrics["records"],
            "scored_tokens_per_arm": metrics["scored_tokens_per_arm"],
            "primary": metrics["primary"],
            "paired_method_comparisons": metrics["paired_method_comparisons"],
            "paired_target_surrogate_mismatch": metrics["paired_target_surrogate_mismatch"],
            "method_specific_cost": metrics["cost"],
            "claim_scope": metrics["claim_scope"],
        },
        "posthoc_original_noninterference_audit": {
            "kind": audit["audit_kind"],
            "preparation": file_record(
                root, revision / "original_noninterference_preparation.json"
            ),
            "audit": file_record(root, revision / "original_noninterference_audit.json"),
            "audit_code_commit": audit["audit_code_commit"],
            "held_fixed": audit_preparation["held_fixed"],
            "identifier_substitution": audit_preparation["identifier_substitution"],
            "comparisons": audit["comparisons"],
            "result": audit["result"],
            "interpretation": audit["interpretation"],
            "repairs_original_access_interface": False,
        },
        "commands": exact_commands,
        "artifacts": {
            "committed_revision_files": revision_files,
            "local_raw_artifacts": local_artifacts,
        },
        "attempts_and_deviations": {
            "preregistration_and_boundary": preregistration_attempts,
            "freeze_attempt_1": {
                "result": "failed before frozen root or receipt due to an invalid namespace-inode reuse assertion",
                "output": file_record(root, revision / "freeze-attempt1-output.txt"),
                "time": file_record(root, revision / "freeze-attempt1-time.txt"),
            },
            "freeze_attempt_2": {
                "result": "false-positive private-path rejection; no receipt; incomplete bundle retained separately",
                "output": file_record(root, revision / "freeze-attempt2-output.txt"),
                "time": file_record(root, revision / "freeze-attempt2-time.txt"),
                "retained_path": "outputs/TRR-0001-R1/failed_freeze_attempt2_frozen_bundle",
            },
            "final_validation_attempt_1": {
                "result": "caught a documentation-only preregistration plan hash transcription error",
                "output": file_record(root, revision / "final-validation-attempt1-output.txt"),
                "time": file_record(root, revision / "final-validation-attempt1-time.txt"),
            },
            "final_validation_attempt_2": {
                "result": "caught a validator-only isolation JSON field-name mismatch",
                "output": file_record(root, revision / "final-validation-attempt2-output.txt"),
                "time": file_record(root, revision / "final-validation-attempt2-time.txt"),
            },
        },
        "validation": {
            "source_compilation": {
                "exit_status": 0,
                "output": file_record(root, revision / "pycompile-output.txt"),
                "time": file_record(root, revision / "pycompile-time.txt"),
            },
            "focused_access_separation_tests": {
                "exit_status": 0,
                "summary": (revision / "focused-tests-output.txt").read_text().strip(),
                "output": file_record(root, revision / "focused-tests-output.txt"),
                "time": file_record(root, revision / "focused-tests-time.txt"),
            },
            "full_source_checkout_tests": {
                "exit_status": 0,
                "summary": (revision / "full-tests-output.txt").read_text().strip(),
                "output": file_record(root, revision / "full-tests-output.txt"),
                "time": file_record(root, revision / "full-tests-time.txt"),
            },
            "final_integrity": {
                "exit_status": final_validation["exit_status"],
                "record": file_record(root, revision / "final_validation.json"),
                "output": file_record(root, revision / "final-validation-output.txt"),
                "time": file_record(root, revision / "final-validation-time.txt"),
                "result": final_validation["result"],
                "row_counts": final_validation["row_counts"],
                "tracked_json_files_parsed_before_this_manifest": final_validation[
                    "tracked_json_files_parsed"
                ],
            },
            "remaining_delivery_checks": [
                "parse every final committed JSON file",
                "validate final patch whitespace excluding only verbatim CRLF packets",
                "compare changed files from the previous reviewed head to the R1 evidence commit",
            ],
        },
        "conclusions": {
            "supported": [
                "causal public-surrogate selection improves token accuracy over direct inversion at cuts 4 and 8 in the pinned clean condition",
                "recoverability decreases with cut depth",
                "proposal recall is the dominant primary bottleneck",
            ],
            "not_supported": [
                "reliable exact non-embedding sequence recovery",
                "a general target-prefix mismatch degradation",
                "generalization beyond the pinned model, dataset, target update, cuts, and candidate budget",
            ],
            "recommended_next_action": "A future fresh study may test public-only proposal recall and the budget/quality/cost frontier; it is proposed only and is not authorized or started.",
        },
    }

    write_json(manifest_path, manifest, create_only=True)
    manifest_record = file_record(root, manifest_path)

    parent = load_json(parent_path)
    parent["revision_r1"] = {
        "id": "TRR-0001-R1",
        "status": "AWAITING_CHATGPT_REVIEW",
        "disposition": "REVISE",
        "packet_id": PACKET_ID,
        "reviewed_head": PREVIOUS_REVIEWED_HEAD,
        "reviewed_pull_request": 2,
        "merge_authorization": "NOT_GRANTED",
        "request": file_record(root, request),
        "original_run": manifest["original_run"],
        "clean_preregistration_commit": PREREGISTRATION_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "selection_commitment": manifest["selection_protocol"]["public_commitment"],
        "selection_reveal": manifest["selection_protocol"]["post_freeze_reveal"],
        "freeze_receipt": manifest["freeze_reveal_truth_chronology"]["freeze_receipt"],
        "isolation": {
            "result": "PASS_FAIL_CLOSED_ACCESS_BOUNDARY",
            "manifests": access_records,
            "denial_probes_per_process": 7,
            "network_default_routes": 0,
        },
        "clean_confirmatory_run": {
            "status": "COMPLETE",
            "fresh_records": metrics["records"],
            "scored_tokens_per_arm": metrics["scored_tokens_per_arm"],
            "primary": metrics["primary"],
            "cost": metrics["cost"]["methods"],
            "next_task_started": False,
        },
        "original_noninterference_audit": {
            "result": audit["result"],
            "code_commit": AUDIT_CODE_COMMIT,
            "record": manifest["posthoc_original_noninterference_audit"]["audit"],
            "repairs_original_access_interface": False,
        },
        "validation": {
            "focused_tests": manifest["validation"]["focused_access_separation_tests"]["summary"],
            "full_tests": manifest["validation"]["full_source_checkout_tests"]["summary"],
            "final_integrity": final_validation["result"],
        },
        "dedicated_manifest": manifest_record,
        "r1_scientific_evidence_commit": "PENDING_METADATA_BINDING",
    }
    parent["task"]["status"] = "revision_r1_awaiting_review"
    parent["task"]["next_task_started"] = False
    write_json(parent_path, parent)

    state = load_json(state_path)
    expected_state = {
        "active_task": "TRR-0001",
        "branch": "task/TRR-0001",
        "pull_request": 2,
        "revision_id": "TRR-0001-R1",
    }
    for key, expected in expected_state.items():
        if state.get(key) != expected:
            raise RuntimeError(f"STATE {key} changed: {state.get(key)!r}")
    state["status"] = "AWAITING_CHATGPT_REVIEW"
    state["updated_utc"] = args.updated_utc
    write_json(state_path, state)

    print(
        json.dumps(
            {
                "status": "R1_MANIFEST_AND_STATE_BUILT",
                "dedicated_manifest": manifest_record,
                "state": "AWAITING_CHATGPT_REVIEW",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
