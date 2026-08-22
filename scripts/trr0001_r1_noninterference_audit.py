#!/usr/bin/env python3
"""Post-truth identifier noninterference audit of the original TRR-0001 run."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
import torch

from token_reconstruction.experiment_runtime import (
    command_record,
    file_record,
    load_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json_exclusive,
)


SCIENCE_FIELDS = (
    "candidate_ids",
    "direct_candidate_scores",
    "causal_candidate_scores",
    "direct_tokens",
    "causal_tokens",
    "candidate_budget",
    "scored_tokens",
    "abstained_tokens",
)
ROUTE_FIELDS = (
    "condition_order",
    "cut_order",
    "methods",
    "candidate_budget",
    "stopping",
    "target_prefix_calls",
    "truth_or_correctness_inputs",
    "implementation",
)
COUNT_FIELDS = (
    "blind_records",
    "scored_tokens_per_arm",
    "arms",
    "direct_embedding_comparisons",
    "causal_candidate_simulations",
    "public_surrogate_model_loads",
    "target_prefix_calls",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repository-root", type=Path, required=True)
    prepare.add_argument("--original-index", type=Path, required=True)
    prepare.add_argument("--original-frozen-root", type=Path, required=True)
    prepare.add_argument("--audit-index", type=Path, required=True)
    prepare.add_argument("--evidence", type=Path, required=True)
    prepare.add_argument("--audit-code-commit", required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--repository-root", type=Path, required=True)
    compare.add_argument("--preparation", type=Path, required=True)
    compare.add_argument("--original-frozen-root", type=Path, required=True)
    compare.add_argument("--audit-output-root", type=Path, required=True)
    compare.add_argument("--evidence", type=Path, required=True)
    compare.add_argument("--audit-code-commit", required=True)
    return value


def required_original_files(root: Path) -> dict[str, dict[str, Any]]:
    return {
        name: file_record(root / relative)
        for name, relative in {
            "reconstructions": "reconstructions.jsonl",
            "queries": "queries.safetensors",
            "route": "route.json",
            "evidence": "reconstructor_evidence.json",
        }.items()
    }


def prepare(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.resolve()
    if len(args.audit_code_commit) != 40:
        raise RuntimeError("full audit code commit is required")
    original_index = args.original_index.resolve()
    original_root = args.original_frozen_root.resolve()
    if args.audit_index.exists() or args.evidence.exists():
        raise RuntimeError("audit preparation outputs are create-only")
    index = load_json(original_index)
    if (
        index.get("schema") != "token-reconstruction.observation-index.v1"
        or index.get("source_tokens_or_text_included") is not False
        or len(index.get("records", [])) != 64
        or len(index.get("entries", [])) != 6
    ):
        raise RuntimeError("original observation interface changed")

    audit = copy.deepcopy(index)
    original_to_opaque = []
    for ordinal, (source, destination) in enumerate(
        zip(index["records"], audit["records"]), 1
    ):
        if set(source) != {"record_id", "dataset_index", "text_sha256"}:
            raise RuntimeError("original source-identifier fields changed")
        opaque_id = f"original-audit-opaque-{ordinal:06d}"
        destination.clear()
        destination.update(
            {
                "record_id": opaque_id,
                "dataset_index": f"opaque-index-{ordinal:06d}",
                "text_sha256": f"opaque-fingerprint-{ordinal:06d}",
            }
        )
        original_to_opaque.append(
            {"original_record_id": source["record_id"], "opaque_record_id": opaque_id}
        )

    observations = []
    for source, destination in zip(index["entries"], audit["entries"]):
        observation = (original_index.parent / source["path"]).resolve()
        if (
            not observation.is_file()
            or sha256_file(observation) != source["artifact"]["sha256"]
        ):
            raise RuntimeError("original activation observation changed")
        destination["path"] = str(observation)
        observations.append(file_record(observation))
    write_json_exclusive(args.audit_index, audit)

    evidence = {
        "schema": "token-reconstruction.trr0001-r1-original-noninterference-preparation.v1",
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R1",
        "audit_kind": "post-hoc noninterference audit; not confirmatory evidence",
        "command": command_record(),
        "created_utc": utc_now(),
        "exit_status": 0,
        "audit_code_commit": args.audit_code_commit,
        "original_index": file_record(original_index, root=repository_root),
        "audit_index": file_record(args.audit_index, root=repository_root),
        "identifier_substitution": {
            "records": 64,
            "fields_replaced": ["record_id", "dataset_index", "text_sha256"],
            "source_identifiers_retained_in_audit_index": 0,
            "mapping": original_to_opaque,
        },
        "held_fixed": {
            "activation_observations": observations,
            "original_frozen_files_before_rerun": required_original_files(
                original_root
            ),
        },
        "truth_used_for_reconstruction": False,
        "result": "PASS_OPAQUE_AUDIT_INPUT_PREPARED",
    }
    write_json_exclusive(args.evidence, evidence)
    print(
        {
            "status": "opaque_original_audit_input_prepared",
            "records": 64,
            "source_identifiers_retained": 0,
            "audit_index": str(args.audit_index),
        }
    )
    return 0


def method_state_signature(route: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": Path(entry["path"]).name,
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for entry in route["method_state"]
    ]


def compare(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.resolve()
    if len(args.audit_code_commit) != 40 or args.evidence.exists():
        raise RuntimeError("compare output is not create-only or commit is incomplete")
    preparation = load_json(args.preparation)
    if (
        preparation.get("result") != "PASS_OPAQUE_AUDIT_INPUT_PREPARED"
        or preparation.get("audit_code_commit") != args.audit_code_commit
    ):
        raise RuntimeError("audit preparation identity changed")
    original_root = args.original_frozen_root.resolve()
    audit_root = args.audit_output_root.resolve()
    original_files_after = required_original_files(original_root)
    if original_files_after != preparation["held_fixed"][
        "original_frozen_files_before_rerun"
    ]:
        raise RuntimeError("original frozen artifacts changed during audit")

    original_rows = read_jsonl(original_root / "reconstructions.jsonl")
    audit_rows = read_jsonl(audit_root / "reconstructions.jsonl")
    mapping = preparation["identifier_substitution"]["mapping"]
    if len(original_rows) != 384 or len(audit_rows) != 384:
        raise RuntimeError("audit reconstruction coverage changed")
    timing_rows = []
    for original, audit in zip(original_rows, audit_rows):
        for identity in ("condition", "cut_depth", "record_index"):
            if original[identity] != audit[identity]:
                raise RuntimeError("audit arm or record routing changed")
        record_index = int(original["record_index"])
        expected = mapping[record_index]
        if (
            original["record_id"] != expected["original_record_id"]
            or audit["record_id"] != expected["opaque_record_id"]
        ):
            raise RuntimeError("identifier substitution did not propagate by order")
        for field in SCIENCE_FIELDS:
            if original[field] != audit[field]:
                raise RuntimeError(f"identifier affected frozen field: {field}")
        timing_rows.append(
            {
                "condition": original["condition"],
                "cut_depth": original["cut_depth"],
                "record_index": record_index,
                "original_direct_seconds": original["direct_amortized_seconds"],
                "audit_direct_seconds": audit["direct_amortized_seconds"],
                "original_causal_seconds": original["causal_amortized_seconds"],
                "audit_causal_seconds": audit["causal_amortized_seconds"],
            }
        )

    original_queries = load_file(original_root / "queries.safetensors", device="cpu")
    audit_queries = load_file(audit_root / "queries.safetensors", device="cpu")
    if set(original_queries) != set(audit_queries) or any(
        not torch.equal(original_queries[key], audit_queries[key])
        for key in original_queries
    ):
        raise RuntimeError("identifier substitution changed frozen query tensors")

    original_route = load_json(original_root / "route.json")
    audit_route = load_json(audit_root / "route.json")
    for field in ROUTE_FIELDS:
        if original_route[field] != audit_route[field]:
            raise RuntimeError(f"identifier affected route field: {field}")
    if audit_route["record_order"] != [
        entry["opaque_record_id"] for entry in mapping
    ]:
        raise RuntimeError("opaque record routing order changed")
    if method_state_signature(original_route) != method_state_signature(audit_route):
        raise RuntimeError("identifier substitution changed persisted method state")

    original_evidence = load_json(original_root / "reconstructor_evidence.json")
    audit_evidence = load_json(audit_root / "reconstructor_evidence.json")
    for field in COUNT_FIELDS:
        if original_evidence[field] != audit_evidence[field]:
            raise RuntimeError(f"identifier affected evidence count: {field}")

    arm_timings = []
    for original, audit in zip(
        original_evidence["arm_timings"], audit_evidence["arm_timings"]
    ):
        if (original["condition"], original["cut_depth"]) != (
            audit["condition"],
            audit["cut_depth"],
        ):
            raise RuntimeError("timing arm order changed")
        arm_timings.append(
            {
                "condition": original["condition"],
                "cut_depth": original["cut_depth"],
                "original": original,
                "opaque_rerun": audit,
                "numeric_equality_expected": False,
                "reason": "wall-clock GPU timings are not deterministic across reruns",
            }
        )

    result = {
        "schema": "token-reconstruction.trr0001-r1-original-noninterference-audit.v1",
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R1",
        "audit_kind": "post-hoc noninterference audit; not confirmatory evidence",
        "command": command_record(),
        "created_utc": utc_now(),
        "exit_status": 0,
        "audit_code_commit": args.audit_code_commit,
        "original_implementation": file_record(
            repository_root / "scripts/trr0001_reconstruct.py",
            root=repository_root,
        ),
        "preparation": file_record(args.preparation, root=repository_root),
        "original_frozen": original_files_after,
        "opaque_rerun": required_original_files(audit_root),
        "comparisons": {
            "rows": 384,
            "candidate_lists": 384 * 39,
            "candidate_values": 384 * 39 * 16,
            "candidate_ids_identical": True,
            "candidate_scores_identical": True,
            "direct_predictions_identical": True,
            "causal_predictions_identical": True,
            "queries_bit_identical": True,
            "routing_identical_except_declared_identifier_substitution": True,
            "stopping_identical": True,
            "method_state_identical": True,
            "operation_counts_identical": True,
            "truth_or_correctness_inputs": 0,
        },
        "timing": {
            "deterministic_route_and_accounting_fields_identical": True,
            "wall_clock_numeric_equality_expected": False,
            "arms": arm_timings,
            "per_record_values": timing_rows,
        },
        "interpretation": (
            "Replacing every original record ID, dataset row number, and text "
            "fingerprint with opaque sentinels did not change any candidate, "
            "score, prediction, query, route, stopping decision, method state, "
            "or operation count. This post-hoc result does not repair or replace "
            "the fresh clean blind R1 run."
        ),
        "result": "PASS_ORIGINAL_IDENTIFIER_NONINTERFERENCE",
    }
    write_json_exclusive(args.evidence, result)
    print(
        {
            "status": result["result"],
            "rows": 384,
            "candidate_values": 384 * 39 * 16,
            "queries_bit_identical": True,
        }
    )
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.operation == "prepare":
        return prepare(args)
    if args.operation == "compare":
        return compare(args)
    raise RuntimeError("unknown operation")


if __name__ == "__main__":
    raise SystemExit(main())
