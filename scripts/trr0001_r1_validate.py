#!/usr/bin/env python3
"""Independent final integrity validation for the TRR-0001-R1 clean run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from typing import Any

from safetensors.torch import load_file
import torch

from token_reconstruction.blind_commitment import (
    validate_observation_index,
    validate_sanitized_config,
)
from token_reconstruction.experiment_runtime import (
    command_record,
    file_record,
    load_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from token_reconstruction.freeze import require_truth_open_allowed
from token_reconstruction.isolation import validate_isolation_manifest
from token_reconstruction.metrics import record_metrics


CHARTER_SHA256 = "ab0fbe9dfad39eddee48c14f4cb8201f8c3f02d1c58668d8a8e59be5a250700d"
REQUEST_SHA256 = "a86e9342df732291052cc8b69599217140ee7bba81a79351dd70f026a6f2360c"
PLAN_SHA256 = "59944cb1e01ec2e88e04109e46db0088eece74500fe599eb6a62ead038b6fe14"
PREREGISTRATION_COMMIT = "4169e4444dd99ebdc81a9adbbb9610dd8afc9555"
IMPLEMENTATION_COMMIT = "970fdf6c65883bb93878c8458141f1e0bcef3085"
TARGET_LORA_SHA256 = "34d92f1e664236bfa1990b10148e8ad52c60b16e72ed0ff4c7eb7da8d15019f6"
INVERSE_SHA256 = {
    4: "9e2487f85057748130bf87b2aad0a883f3c36dfc052eefd83c0f5c35497a24e3",
    8: "ac8871f1fa0d40664c5d9d94343ef560832477aade3574978a1c6b572df01e80",
}
METHODS = ("direct_inverse", "causal_public_surrogate_search")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tracked_json_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "*.json"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    values = [
        root / value.decode("utf-8")
        for value in result.stdout.split(b"\0")
        if value
    ]
    if not values:
        raise RuntimeError("tracked JSON enumeration is empty")
    return sorted(values)


def compare_rows(
    *,
    direct_rows: list[dict[str, Any]],
    causal_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> dict[str, int]:
    if len(direct_rows) != 384 or len(causal_rows) != 384:
        raise RuntimeError("frozen reconstruction coverage changed")
    truth = {row["record_id"]: row["token_ids"][1:] for row in truth_rows}
    if len(truth) != 64:
        raise RuntimeError("truth identifiers are absent or duplicated")
    direct_by_key = {}
    causal_by_key = {}
    for direct, causal in zip(direct_rows, causal_rows):
        key = (direct["condition"], direct["cut_depth"], direct["record_index"])
        if key != (causal["condition"], causal["cut_depth"], causal["record_index"]):
            raise RuntimeError("cross-method row order changed")
        if (
            direct["record_id"] != causal["record_id"]
            or direct["candidate_ids"] != causal["candidate_ids"]
            or direct["proposal_scores"] != causal["proposal_scores"]
        ):
            raise RuntimeError("cross-method candidate interface changed")
        if direct["prediction_tokens"] != [
            candidates[0] for candidates in direct["candidate_ids"]
        ]:
            raise RuntimeError("direct prediction is not proposal top-1")
        if any(
            token not in candidates
            for token, candidates in zip(
                causal["prediction_tokens"], causal["candidate_ids"]
            )
        ):
            raise RuntimeError("causal prediction escaped candidate budget")
        direct_by_key[key] = direct
        causal_by_key[key] = causal
    if len(direct_by_key) != 384 or len(causal_by_key) != 384:
        raise RuntimeError("frozen arm keys are absent or duplicated")

    inconsistencies = 0
    if len(metric_rows) != 768:
        raise RuntimeError("per-record metric coverage changed")
    for row in metric_rows:
        key = (row["condition"], row["cut_depth"], row["record_index"])
        frozen = (
            direct_by_key[key]
            if row["method"] == METHODS[0]
            else causal_by_key[key]
        )
        prediction = frozen["prediction_tokens"]
        candidates = frozen["candidate_ids"]
        measured = record_metrics(prediction, truth[row["record_id"]], candidates)
        if (
            row["correct_tokens"] != measured["correct_tokens"]
            or row["exact_sequence_match"] != measured["exact_sequence_match"]
            or row["true_in_top16"] != measured["true_in_top16"]
            or row["true_in_top16"]
            != [rank <= 16 for rank in row["true_token_ranks"]]
        ):
            inconsistencies += 1
    if inconsistencies:
        raise RuntimeError("frozen outputs and committed metrics differ")
    return {
        "direct_rows": len(direct_rows),
        "causal_rows": len(causal_rows),
        "metric_rows": len(metric_rows),
        "metric_inconsistencies": inconsistencies,
    }


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    if args.output.exists():
        raise RuntimeError("validation output is create-only")
    receipt_path = root / "experiments/TRR-0001/revision-r1/freeze_receipt.json"
    truth_path = (
        root
        / "outputs/TRR-0001-R1/clean/evaluator_private/blind_truth.jsonl"
    )
    receipt = require_truth_open_allowed(
        receipt_path=receipt_path,
        repository_root=root,
        truth_path=truth_path,
    )
    metadata = receipt["metadata"]
    if (
        receipt["preregistration_commit"] != PREREGISTRATION_COMMIT
        or metadata["implementation_commit"] != IMPLEMENTATION_COMMIT
        or metadata["access_manifests_verified"] is not True
        or metadata["selection_revealed"] is not False
        or metadata["truth_opened"] is not False
        or len(receipt["entries"]) != 22
    ):
        raise RuntimeError("freeze receipt identity or gate metadata changed")
    for entry in receipt["entries"]:
        path = root / entry["path"]
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"frozen artifact remains writable: {path}")

    request = root / "coordination/requests/TRR-0001-R1.md"
    request_bytes = request.read_bytes()
    if (
        len(request_bytes) != 26821
        or hashlib.sha256(request_bytes).hexdigest() != REQUEST_SHA256
        or request_bytes.count(b"\r\n") == 0
        or request_bytes.replace(b"\r\n", b"").count(b"\n") != 0
    ):
        raise RuntimeError("verbatim R1 request bytes changed")
    charter = root / "RESEARCH_CHARTER.md"
    plan = root / "experiments/TRR-0001/revision-r1/plan.json"
    if sha256_file(charter) != CHARTER_SHA256 or sha256_file(plan) != PLAN_SHA256:
        raise RuntimeError("charter or preregistered plan changed")

    frozen = root / receipt["frozen_root"]
    config = load_json(frozen / "reconstructor_input/sanitized_config.json")
    observation_index = load_json(
        frozen / "reconstructor_input/observation_index.json"
    )
    validate_sanitized_config(config)
    validate_observation_index(observation_index)
    if config["truth_or_source_inputs"] != 0:
        raise RuntimeError("sanitized configuration declares a private input")
    for cut, expected in INVERSE_SHA256.items():
        inverse = frozen / f"reconstructor_input/inverses/cut{cut}.safetensors"
        if sha256_file(inverse) != expected:
            raise RuntimeError(f"frozen cut-{cut} inverse changed")

    access = {}
    for method in METHODS:
        value = load_json(frozen / method / "access_manifest.json")
        validate_isolation_manifest(value, method=method)
        if (
            len(value["denial_probes"]) != 7
            or value["network"]["default_route_present"] is not False
        ):
            raise RuntimeError(f"{method} access-denial evidence changed")
        access[method] = value
    if access[METHODS[0]]["started_utc"] == access[METHODS[1]]["started_utc"]:
        raise RuntimeError("separate process invocation timestamps changed")

    reveal_path = root / "experiments/TRR-0001/revision-r1/selection_reveal.json"
    verification = load_json(
        root / "experiments/TRR-0001/revision-r1/selection_verification.json"
    )
    reveal = load_json(reveal_path)
    if (
        verification["result"]
        != "PASS_SELECTION_REVEALED_AFTER_VERIFIED_FREEZE"
        or verification["truth_sidecar_read"] is not False
        or verification["token_ids_disclosed_in_mapping_reveal"] is not False
        or any(
            set(row) != {"record_id", "dataset_index", "text_sha256"}
            for row in reveal["records"]
        )
    ):
        raise RuntimeError("post-freeze selection reveal evidence changed")

    truth_rows = read_jsonl(truth_path)
    metric_rows = read_jsonl(
        root / "experiments/TRR-0001/revision-r1/per_record_metrics.jsonl"
    )
    row_counts = compare_rows(
        direct_rows=read_jsonl(
            frozen / "direct_inverse/reconstructions.jsonl"
        ),
        causal_rows=read_jsonl(
            frozen
            / "causal_public_surrogate_search/reconstructions.jsonl"
        ),
        truth_rows=truth_rows,
        metric_rows=metric_rows,
    )

    direct_queries = load_file(
        frozen / "direct_inverse/queries.safetensors", device="cpu"
    )
    causal_queries = load_file(
        frozen / "causal_public_surrogate_search/queries.safetensors",
        device="cpu",
    )
    expected_query_keys = {
        f"{condition}.cut{cut}"
        for condition in ("matched_public", "unavailable_target_lora")
        for cut in (0, 4, 8)
    }
    if (
        set(direct_queries) != expected_query_keys
        or set(causal_queries) != expected_query_keys
        or any(
            tuple(value.shape) != (64, 39, 2048)
            for value in direct_queries.values()
        )
        or any(
            not torch.equal(direct_queries[key], causal_queries[key])
            for key in expected_query_keys
        )
    ):
        raise RuntimeError("frozen query identity or geometry changed")

    metrics = load_json(
        root / "experiments/TRR-0001/revision-r1/metrics.json"
    )
    primary = metrics["primary"]
    if (
        metrics["records"] != 64
        or metrics["scored_tokens_per_arm"] != 2496
        or primary["direct_inverse"] != 0.6470352564102564
        or primary["causal_public_surrogate_search"] != 0.8397435897435898
        or primary["causal_minus_direct"]["disposition"]
        != "causal_supported_improvement"
        or metrics["cost"]["target_prefix_calls"] != 0
    ):
        raise RuntimeError("primary result, coverage, or target-call count changed")
    audit = load_json(
        root
        / "experiments/TRR-0001/revision-r1/original_noninterference_audit.json"
    )
    if (
        audit["result"] != "PASS_ORIGINAL_IDENTIFIER_NONINTERFERENCE"
        or not all(
            value is True
            for key, value in audit["comparisons"].items()
            if key.endswith("_identical")
        )
    ):
        raise RuntimeError("original noninterference audit changed")

    target_lora = (
        root / "outputs/TRR-0001/evaluator_private/target_lora.safetensors"
    )
    if sha256_file(target_lora) != TARGET_LORA_SHA256:
        raise RuntimeError("fixed target LoRA changed")
    json_paths = tracked_json_paths(root)
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))

    result = {
        "schema": "token-reconstruction.trr0001-r1-final-validation.v1",
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R1",
        "command": command_record(),
        "created_utc": utc_now(),
        "exit_status": 0,
        "result": "PASS_ALL_R1_INTEGRITY_CHECKS",
        "charter_sha256": sha256_file(charter),
        "request": file_record(request, root=root),
        "plan": file_record(plan, root=root),
        "freeze_receipt": file_record(receipt_path, root=root),
        "frozen_entries": len(receipt["entries"]),
        "frozen_bytes": sum(entry["bytes"] for entry in receipt["entries"]),
        "access_manifests": {
            method: file_record(
                frozen / method / "access_manifest.json", root=root
            )
            for method in METHODS
        },
        "denial_probes_per_process": 7,
        "network_default_routes": 0,
        "truth_records_opened_after_all_gates": len(truth_rows),
        "row_counts": row_counts,
        "query_arms": len(expected_query_keys),
        "query_geometry": [64, 39, 2048],
        "primary": primary,
        "target_prefix_calls": 0,
        "target_lora_sha256": sha256_file(target_lora),
        "inverse_sha256": {str(key): value for key, value in INVERSE_SHA256.items()},
        "original_noninterference_audit": file_record(
            root
            / "experiments/TRR-0001/revision-r1/original_noninterference_audit.json",
            root=root,
        ),
        "tracked_json_files_parsed": len(json_paths),
    }
    write_json_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
