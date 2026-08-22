#!/usr/bin/env python3
"""Create the TRR-0001-R1 clean confirmation plan before clean outputs exist."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from token_reconstruction.blind_commitment import (
    reject_source_metadata,
    validate_public_commitment,
)
from token_reconstruction.experiment_runtime import (
    file_record,
    load_json,
    utc_now,
    write_json_exclusive,
)


TARGET_LORA_SHA256 = "34d92f1e664236bfa1990b10148e8ad52c60b16e72ed0ff4c7eb7da8d15019f6"
INVERSE_SHA256 = {
    "4": "9e2487f85057748130bf87b2aad0a883f3c36dfc052eefd83c0f5c35497a24e3",
    "8": "ac8871f1fa0d40664c5d9d94343ef560832477aade3574978a1c6b572df01e80",
}
PACKET_ID = "TRR-PACKET-TRR-0001-REVIEW-R1-20260822-1E1E1C00AFB4A8BC58B157BB"
PREVIOUS_HEAD = "2a89a578921afc48dc3b54c65a2c39c35c422134"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--original-plan", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--revision-request", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--clean-output-root", type=Path, required=True)
    parser.add_argument("--created-utc")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    if args.plan.exists() or args.plan.is_symlink():
        raise RuntimeError("revision plan is create-only")
    if args.clean_output_root.exists() or args.clean_output_root.is_symlink():
        raise RuntimeError("clean output root must not exist before preregistration")
    original = load_json(args.original_plan)
    commitment = load_json(args.commitment)
    validate_public_commitment(commitment)
    plan = copy.deepcopy(original)
    add_special_tokens = plan["resources"]["tokenizer"].pop(
        "add_special_tokens_for_source_text"
    )
    plan["resources"]["tokenizer"]["add_special_tokens"] = add_special_tokens
    plan["schema"] = "token-reconstruction.trr0001-r1.preregistration.v1"
    plan["created_utc"] = args.created_utc or utc_now()
    plan["status"] = "COMMITTED_BEFORE_CLEAN_CONFIRMATORY_OUTPUTS"
    plan["truth_opened"] = False
    plan["revision"] = {
        "revision_id": "TRR-0001-R1",
        "packet_id": PACKET_ID,
        "request": file_record(args.revision_request, root=root),
        "previous_reviewed_head": PREVIOUS_HEAD,
        "purpose": "access-compliant clean confirmation with a fresh disjoint split",
        "scientific_method_change": False,
    }
    plan["original_run_disposition"] = {
        "label": "ACCESS_INTERFACE_NONCOMPLIANT_ORIGINAL_RUN",
        "accepted_blind_claim": False,
        "retained_uses": [
            "descriptive evidence",
            "debugging evidence",
            "post-hoc source-identifier noninterference audit",
        ],
        "prohibited_use": "accepted blind confirmatory evidence",
    }
    plan["data"]["selection"] = {
        "kind": "cryptographically_hiding_evaluator_private_selection",
        "public_commitment": file_record(args.commitment, root=root),
        "commitment_digest": commitment["commitment"],
        "opaque_record_order": commitment["opaque_record_order"],
        "record_count": 64,
        "eligibility": commitment["eligibility"],
        "disjointness": commitment["disjointness"],
        "source_identity_available_to_reconstructor": False,
        "key_or_seed_available_to_reconstructor": False,
        "post_freeze_reveal_required": True,
    }
    plan["seeds"].pop("record_selection", None)
    plan["seeds"]["fresh_record_selection"] = (
        "evaluator-private cryptographic 256-bit key; value withheld until post-freeze reveal"
    )
    plan["target_prefix_update"]["artifact"] = {
        "path": "outputs/TRR-0001/evaluator_private/target_lora.safetensors",
        "sha256": TARGET_LORA_SHA256,
        "reused_without_training": True,
        "reconstructor_access": False,
    }
    plan["target_prefix_update"]["decision_time"] = (
        "fixed by the original preregistration and R1 packet; exact retained artifact reused"
    )
    plan["fixed_method_state"] = {
        "target_lora_sha256": TARGET_LORA_SHA256,
        "inverse_cut4_sha256": INVERSE_SHA256["4"],
        "inverse_cut8_sha256": INVERSE_SHA256["8"],
        "training_or_adaptation_in_clean_run": False,
    }
    plan["baseline_families"]["causal_public_surrogate_search"]["commit"] = (
        "highest cosine score; frozen proposal rank then token-ID ascending tie break"
    )
    plan["access_isolation"] = {
        "mechanism": (
            "unshare user, mount, network, and PID namespaces; minimal chroot; "
            "read-only binds for exact code, sanitized input, public inverse state, "
            "pinned public model/tokenizer and Python runtime; tmpfs temporary space; "
            "one empty writable method output"
        ),
        "separate_processes": [
            "direct_inverse",
            "causal_public_surrogate_search",
        ],
        "network": "new namespace with no interface route; denial probe required",
        "workspace": "repository and host workspace absent from chroot",
        "dataset": "dataset package/cache absent from chroot",
        "evaluator_private": "private mapping, truth, target LoRA, and evaluator evidence absent",
        "environment": "cleared and replaced with an allowlisted offline environment",
        "permissions": "public inputs and runtime read-only; output is the only writable bind",
        "access_manifest": "frozen before selection reveal or truth opening",
        "denial_probes": [
            "repository root",
            "dataset cache and dataset package",
            "fresh private mapping",
            "clean truth",
            "target LoRA",
            "original truth",
            "network route",
        ],
        "failure_policy": "any schema, mount, permission, access-probe, or network-probe failure exits nonzero",
    }
    plan["execution_order"] = {
        "pre_freeze": [
            "commit this revision plan and public hiding commitment",
            "generate evaluator-only fresh observations and sanitized reconstruction inputs",
            "validate strict schemas and verify exact retained state hashes",
            "export exact reconstruction code and public runtime dependencies",
            "run access probes and direct method in its isolated process",
            "run access probes and causal method in its separate isolated process",
            "freeze sanitized inputs, method outputs, routes, configuration, and access manifests",
            "verify freeze receipt and isolation evidence",
        ],
        "post_freeze_pre_truth": [
            "reveal selection key and private mapping",
            "verify HMAC commitment, keyed ordering, eligibility, opaque order, and disjointness",
            "open clean truth only through the combined freeze, access, and commitment gate",
        ],
        "post_truth": [
            "score frozen direct and causal outputs without revision",
            "run preregistered diagnostics",
            "perform the post-hoc original-run opaque-ID noninterference audit",
        ],
    }
    plan["freeze_contract"] = {
        "frozen_root": "outputs/TRR-0001-R1/clean/frozen_bundle",
        "receipt_path": "experiments/TRR-0001/revision-r1/freeze_receipt.json",
        "required_before_reveal": True,
        "required_before_truth": True,
        "hashes": [
            "sanitized observation index and reconstruction configuration",
            "all boundary observation tensors",
            "direct and causal candidates, scores, predictions, queries, routes, timings, and counters",
            "isolation identity, mount, permission, environment, and denial-probe manifests",
        ],
        "mutation_policy": "create-only files made read-only; any byte, path, schema, or access mismatch fails closed",
    }
    plan["cost_accounting"] = {
        "separation": "direct and causal execute in distinct fresh processes with reset CUDA counters",
        "direct": "proposal and total runtime separately per arm, record, and scored token",
        "causal": "proposal, causal selection, and total runtime separately per arm, record, and scored token",
        "memory": "pre-method preparation and reset per-method CUDA peaks plus process max RSS",
        "evaluations": "full-vocabulary embedding comparisons and candidate simulations counted exactly",
        "training": "zero fresh training/adaptation; exact persisted state bytes and hashes recorded",
        "complexity": "source lines and executable statement counts for method-common and method-specific code",
    }
    plan["artifacts"] = {
        "uncommitted_private_root": "outputs/TRR-0001-R1",
        "committed_revision_root": "experiments/TRR-0001/revision-r1",
        "result": "coordination/results/TRR-0001.md",
        "manifest": "experiments/TRR-0001/manifest.json",
        "retention": "retain all local raw artifacts through review; commit hashes, byte counts, commands, outputs, statuses, and required compact evidence",
    }
    plan["deviation_policy"]["current_deviations"] = [
        {
            "from_original_run": "source-resolvable IDs and same-workspace interface",
            "required_revision": "fresh opaque committed split and process-enforced access boundary",
            "scientific_method_effect": "none",
        }
    ]
    reject_source_metadata(plan)
    write_json_exclusive(args.plan, plan)
    print(
        {
            "status": "revision_preregistered_before_clean_outputs",
            "plan": str(args.plan),
            "commitment": commitment["commitment"],
            "clean_output_root_absent": not args.clean_output_root.exists(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
