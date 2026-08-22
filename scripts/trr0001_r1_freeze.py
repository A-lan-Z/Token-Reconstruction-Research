#!/usr/bin/env python3
"""Validate, copy, and immutably freeze the TRR-0001-R1 clean interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

from safetensors.torch import load_file
import torch

from token_reconstruction.blind_commitment import (
    canonical_bytes,
    reject_source_metadata,
    validate_observation_index,
    validate_sanitized_config,
)
from token_reconstruction.experiment_runtime import (
    file_record,
    load_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from token_reconstruction.freeze import (
    create_freeze_receipt,
    verify_freeze_receipt,
)
from token_reconstruction.isolation import validate_isolation_manifest


METHODS = ("direct_inverse", "causal_public_surrogate_search")
ROW_KEYS = {
    "condition",
    "cut_depth",
    "record_index",
    "record_id",
    "method",
    "prediction_tokens",
    "candidate_ids",
    "proposal_scores",
    "selection_scores",
    "candidate_budget",
    "scored_tokens",
    "abstained_tokens",
    "proposal_amortized_seconds",
    "selection_amortized_seconds",
    "total_amortized_seconds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-input", type=Path, required=True)
    parser.add_argument("--direct-output", type=Path, required=True)
    parser.add_argument("--causal-output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-repository", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--created-utc")
    return parser.parse_args()


def regular_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_dir():
        raise RuntimeError(f"{label} must be a regular directory")
    return resolved


def tree_manifest(root: Path, *, hash_contents: bool = True) -> dict[str, Any]:
    root = regular_directory(root, "tree")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        resolved = path.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"tree member is not a regular file: {relative}")
        entry = {
            "path": relative,
            "bytes": resolved.stat().st_size,
            "mode": oct(path.lstat().st_mode & 0o777),
            "symlink": path.is_symlink(),
        }
        if path.is_symlink():
            entry["link_target"] = os.readlink(path)
        if hash_contents:
            entry["sha256"] = sha256_file(resolved)
        entries.append(entry)
    if not entries:
        raise RuntimeError("tree manifest is empty")
    return {
        "root": str(root),
        "entries": entries,
        "aggregate_sha256": hashlib.sha256(canonical_bytes(entries)).hexdigest(),
    }


def validate_rows(path: Path, *, method: str, record_order: list[str]) -> list[dict]:
    rows = read_jsonl(path)
    if len(rows) != 384:
        raise RuntimeError(f"{method} row coverage changed")
    expected_keys = [
        (condition, cut, record_index)
        for condition in ("matched_public", "unavailable_target_lora")
        for cut in (0, 4, 8)
        for record_index in range(64)
    ]
    observed_keys = []
    for row in rows:
        if set(row) != ROW_KEYS:
            raise RuntimeError(f"{method} reconstruction fields changed")
        if row["method"] != method:
            raise RuntimeError(f"{method} row has the wrong method identity")
        record_index = int(row["record_index"])
        if row["record_id"] != record_order[record_index]:
            raise RuntimeError(f"{method} opaque record order changed")
        if row["candidate_budget"] != 16 or row["scored_tokens"] != 39:
            raise RuntimeError(f"{method} frozen budget changed")
        if row["abstained_tokens"] != 0:
            raise RuntimeError(f"{method} unexpectedly abstained")
        if len(row["prediction_tokens"]) != 39:
            raise RuntimeError(f"{method} prediction geometry changed")
        if len(row["candidate_ids"]) != 39 or any(
            len(values) != 16 for values in row["candidate_ids"]
        ):
            raise RuntimeError(f"{method} candidate geometry changed")
        if len(row["proposal_scores"]) != 39 or any(
            len(values) != 16 for values in row["proposal_scores"]
        ):
            raise RuntimeError(f"{method} proposal score geometry changed")
        if len(row["selection_scores"]) != 39 or any(
            len(values) != 16 for values in row["selection_scores"]
        ):
            raise RuntimeError(f"{method} selection score geometry changed")
        observed_keys.append(
            (row["condition"], int(row["cut_depth"]), record_index)
        )
    if observed_keys != expected_keys:
        raise RuntimeError(f"{method} arm/order routing changed")
    return rows


def validate_method_evidence(path: Path, *, method: str) -> dict[str, Any]:
    value = load_json(path)
    if (
        value.get("schema")
        != "token-reconstruction.trr0001-r1-reconstructor-evidence.v1"
        or value.get("method") != method
        or value.get("exit_status") != 0
    ):
        raise RuntimeError(f"{method} evidence identity changed")
    if value.get("access_manifest_verified_before_inputs") is not True:
        raise RuntimeError(f"{method} did not validate access before inputs")
    if value.get("fresh_training_steps") != 0 or value.get("fresh_adaptation_steps") != 0:
        raise RuntimeError(f"{method} performed fresh adaptation")
    expected_comparisons = 2496 * 128256 * 6
    expected_simulations = (
        2496 * 16 * 6 if method == "causal_public_surrogate_search" else 0
    )
    if value.get("embedding_comparisons") != expected_comparisons:
        raise RuntimeError(f"{method} embedding-comparison count changed")
    if value.get("candidate_simulations") != expected_simulations:
        raise RuntimeError(f"{method} candidate-simulation count changed")
    timings = value.get("arm_timings")
    if not isinstance(timings, list) or len(timings) != 6:
        raise RuntimeError(f"{method} arm timing coverage changed")
    for timing in timings:
        required = {
            "proposal_seconds",
            "selection_seconds",
            "compute_seconds",
            "end_to_end_arm_seconds",
            "proposal_seconds_per_record",
            "selection_seconds_per_record",
            "total_seconds_per_record",
            "proposal_seconds_per_scored_token",
            "selection_seconds_per_scored_token",
            "total_seconds_per_scored_token",
            "embedding_comparisons",
            "candidate_simulations",
        }
        if not required.issubset(timing):
            raise RuntimeError(f"{method} timing fields are incomplete")
        if timing["proposal_seconds"] <= 0 or timing["compute_seconds"] <= 0:
            raise RuntimeError(f"{method} proposal timing is not positive")
        if method == "direct_inverse" and timing["selection_seconds"] != 0.0:
            raise RuntimeError("direct selection timing must be zero")
        if method == "causal_public_surrogate_search" and timing["selection_seconds"] <= 0:
            raise RuntimeError("causal selection timing must be positive")
    memory = value.get("memory", {})
    if not {"preparation_peak", "method_peak_after_cuda_reset"}.issubset(memory):
        raise RuntimeError(f"{method} separate memory evidence is absent")
    if not value.get("implementation_complexity"):
        raise RuntimeError(f"{method} implementation complexity is absent")
    return value


def binary_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"frozen destination is not create-only: {destination}")
    shutil.copytree(source, destination, symlinks=False, copy_function=shutil.copy2)


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    if len(args.preregistration_commit) != 40 or len(args.implementation_commit) != 40:
        raise RuntimeError("full commit identities are required")
    public_root = regular_directory(args.public_input, "public input")
    direct_root = regular_directory(args.direct_output, "direct output")
    causal_root = regular_directory(args.causal_output, "causal output")
    code_root = regular_directory(args.code_root, "code export")
    model_repository = regular_directory(args.model_repository, "model repository")
    if args.frozen_root.exists() or args.receipt.exists():
        raise RuntimeError("freeze outputs are create-only")

    config = load_json(public_root / "sanitized_config.json")
    index = load_json(public_root / "observation_index.json")
    validate_sanitized_config(config)
    validate_observation_index(index)
    reject_source_metadata(config)
    reject_source_metadata(index)
    record_order = config["record_order"]

    direct_access = load_json(direct_root / "access_manifest.json")
    causal_access = load_json(causal_root / "access_manifest.json")
    validate_isolation_manifest(direct_access, method=METHODS[0])
    validate_isolation_manifest(causal_access, method=METHODS[1])
    if (
        direct_access["started_utc"] == causal_access["started_utc"]
        or direct_access["method"] != METHODS[0]
        or causal_access["method"] != METHODS[1]
    ):
        raise RuntimeError("method-specific process invocation evidence changed")
    namespace_inode_reuse = {
        namespace: (
            direct_access["namespaces"][namespace]
            == causal_access["namespaces"][namespace]
        )
        for namespace in ("user", "mount", "network", "pid")
    }

    direct_rows = validate_rows(
        direct_root / "reconstructions.jsonl",
        method=METHODS[0],
        record_order=record_order,
    )
    causal_rows = validate_rows(
        causal_root / "reconstructions.jsonl",
        method=METHODS[1],
        record_order=record_order,
    )
    for direct, causal in zip(direct_rows, causal_rows):
        if direct["candidate_ids"] != causal["candidate_ids"]:
            raise RuntimeError("candidate proposals depend on the method process")
        if direct["proposal_scores"] != causal["proposal_scores"]:
            raise RuntimeError("proposal scores depend on the method process")
        if direct["prediction_tokens"] != [
            candidates[0] for candidates in direct["candidate_ids"]
        ]:
            raise RuntimeError("direct output is not frozen proposal top-1")
        if any(
            predicted not in candidates
            for predicted, candidates in zip(
                causal["prediction_tokens"], causal["candidate_ids"]
            )
        ):
            raise RuntimeError("causal output escaped its frozen candidate budget")

    direct_queries = load_file(direct_root / "queries.safetensors", device="cpu")
    causal_queries = load_file(causal_root / "queries.safetensors", device="cpu")
    if set(direct_queries) != set(causal_queries) or any(
        not torch.equal(direct_queries[key], causal_queries[key])
        for key in direct_queries
    ):
        raise RuntimeError("recomputed method query tensors differ")

    direct_evidence = validate_method_evidence(
        direct_root / "reconstructor_evidence.json", method=METHODS[0]
    )
    causal_evidence = validate_method_evidence(
        causal_root / "reconstructor_evidence.json", method=METHODS[1]
    )

    frozen_root = args.frozen_root.resolve()
    frozen_root.mkdir(parents=True)
    copy_tree(public_root, frozen_root / "reconstructor_input")
    copy_tree(direct_root, frozen_root / METHODS[0])
    copy_tree(causal_root, frozen_root / METHODS[1])

    code = tree_manifest(code_root)
    snapshot = model_repository / "snapshots" / "9213176726f574b556790deb65791e0c5aa438b6"
    model = tree_manifest(snapshot)
    runtime = {
        "isolation_kind": "native Linux user/mount/network/PID namespaces with pivot_root; no container image",
        "container_image": None,
        "container_image_not_applicable": True,
        "implementation_commit": args.implementation_commit,
        "kernel": platform.uname()._asdict(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "binaries": [
            binary_record(Path(path))
            for path in (
                "/usr/bin/unshare",
                "/usr/sbin/pivot_root",
                "/usr/bin/mount",
                "/usr/bin/bash",
                "/usr/bin/python3.12",
            )
        ],
        "launcher": file_record(
            repository_root / "scripts/trr0001_r1_isolate.sh",
            root=repository_root,
        ),
        "code_export": code,
        "model_snapshot": model,
        "input_tree": tree_manifest(public_root),
    }
    runtime["runtime_identity_sha256"] = hashlib.sha256(
        canonical_bytes(runtime)
    ).hexdigest()
    runtime_path = frozen_root / "runtime_identity.json"
    write_json_exclusive(runtime_path, runtime)

    validation = {
        "schema": "token-reconstruction.trr0001-r1-prefreeze-validation.v1",
        "created_utc": args.created_utc or utc_now(),
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R1",
        "preregistration_commit": args.preregistration_commit,
        "implementation_commit": args.implementation_commit,
        "sanitized_interface": {
            "validated": True,
            "opaque_records": 64,
            "observation_entries": 6,
            "source_identity_fields": 0,
            "source_text_or_token_fields": 0,
        },
        "isolated_processes": {
            "validated": True,
            "separate_sequential_process_invocations": True,
            "namespace_inode_reuse_after_sequential_teardown": namespace_inode_reuse,
            "namespace_inode_numbers_are_lifetime_scoped": True,
            "direct": direct_access["namespaces"],
            "causal": causal_access["namespaces"],
            "denial_probes_per_process": 7,
            "network_default_routes": 0,
        },
        "cross_method_independence": {
            "rows_compared": 384,
            "candidate_ids_identical": True,
            "proposal_scores_identical": True,
            "query_tensors_identical": True,
        },
        "method_cost_evidence": {
            "direct": {
                "embedding_comparisons": direct_evidence["embedding_comparisons"],
                "candidate_simulations": direct_evidence["candidate_simulations"],
                "method_compute_seconds": direct_evidence["method_compute_seconds"],
                "memory": direct_evidence["memory"],
                "complexity": direct_evidence["implementation_complexity"],
            },
            "causal": {
                "embedding_comparisons": causal_evidence["embedding_comparisons"],
                "candidate_simulations": causal_evidence["candidate_simulations"],
                "method_compute_seconds": causal_evidence["method_compute_seconds"],
                "memory": causal_evidence["memory"],
                "complexity": causal_evidence["implementation_complexity"],
            },
        },
        "truth_or_correctness_opened": False,
        "selection_mapping_or_key_opened": False,
        "result": "PASS_READY_TO_FREEZE",
    }
    validation_path = frozen_root / "prefreeze_validation.json"
    write_json_exclusive(validation_path, validation)

    receipt = create_freeze_receipt(
        repository_root=repository_root,
        frozen_root=frozen_root,
        plan_path=args.plan.resolve(),
        receipt_path=args.receipt.resolve(),
        preregistration_commit=args.preregistration_commit,
        created_utc=args.created_utc or utc_now(),
        metadata={
            "task_id": "TRR-0001",
            "revision_id": "TRR-0001-R1",
            "implementation_commit": args.implementation_commit,
            "methods": list(METHODS),
            "runtime_identity_sha256": runtime["runtime_identity_sha256"],
            "access_manifests_verified": True,
            "selection_revealed": False,
            "truth_opened": False,
        },
    )
    verified = verify_freeze_receipt(
        args.receipt.resolve(), repository_root=repository_root
    )
    if verified != receipt:
        raise RuntimeError("create-time and verification-time freeze receipts differ")
    print(
        {
            "status": "clean_outputs_frozen_and_verified",
            "entries": len(receipt["entries"]),
            "runtime_identity_sha256": runtime["runtime_identity_sha256"],
            "cross_method_rows": 384,
            "truth_opened": False,
            "selection_revealed": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
