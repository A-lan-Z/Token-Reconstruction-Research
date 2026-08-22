#!/usr/bin/env python3
"""Independent post-freeze integrity validation for TRR-0001."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat

from safetensors.torch import load_file

from token_reconstruction.experiment_runtime import read_jsonl, sha256_file
from token_reconstruction.freeze import require_truth_open_allowed


CHARTER_SHA256 = "ab0fbe9dfad39eddee48c14f4cb8201f8c3f02d1c58668d8a8e59be5a250700d"
REQUEST_SHA256 = "e12296750323ecd8d90955f052f6ca2a1140e2e6452dc3cedadf4cb636daca6e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    root = parse_args().repository_root.resolve()
    receipt_path = root / "experiments/TRR-0001/freeze_receipt.json"
    truth_path = root / "outputs/TRR-0001/evaluator_private/blind_truth.jsonl"
    receipt = require_truth_open_allowed(
        receipt_path=receipt_path,
        repository_root=root,
        truth_path=truth_path,
    )
    if receipt["preregistration_commit"] != "6648b322bcaaa32ed225c962bb874105f2c98d48":
        raise RuntimeError("preregistration commit changed")
    for entry in receipt["entries"]:
        path = root / entry["path"]
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"frozen artifact remains writable: {path}")

    request = root / "coordination/requests/TRR-0001.md"
    request_bytes = request.read_bytes()
    if len(request_bytes) != 24375 or hashlib.sha256(request_bytes).hexdigest() != REQUEST_SHA256:
        raise RuntimeError("incoming packet bytes changed")
    if request_bytes.count(b"\r\n") == 0 or request_bytes.replace(b"\r\n", b"").count(b"\n"):
        raise RuntimeError("incoming packet line endings changed")
    charter = root / "RESEARCH_CHARTER.md"
    if sha256_file(charter) != CHARTER_SHA256:
        raise RuntimeError("authoritative charter changed")

    frozen = root / receipt["frozen_root"]
    reconstruction_rows = read_jsonl(frozen / "reconstructions.jsonl")
    truth_rows = read_jsonl(truth_path)
    per_record_rows = read_jsonl(
        root / "experiments/TRR-0001/per_record_metrics.jsonl"
    )
    if len(reconstruction_rows) != 384 or len(truth_rows) != 64 or len(per_record_rows) != 768:
        raise RuntimeError("record evidence coverage changed")

    reconstructions = {
        (row["condition"], row["cut_depth"], row["record_index"]): row
        for row in reconstruction_rows
    }
    truth = {row["record_id"]: row["token_ids"][1:] for row in truth_rows}
    if len(reconstructions) != 384 or len(truth) != 64:
        raise RuntimeError("record identifiers are duplicated")

    inconsistencies = 0
    causal_outside_candidates = 0
    direct_not_first = 0
    for row in per_record_rows:
        frozen_row = reconstructions[
            (row["condition"], row["cut_depth"], row["record_index"])
        ]
        expected = truth[row["record_id"]]
        if row["true_in_top16"] != [rank <= 16 for rank in row["true_token_ranks"]]:
            inconsistencies += 1
        if frozen_row["direct_tokens"] != [
            candidates[0] for candidates in frozen_row["candidate_ids"]
        ]:
            direct_not_first += 1
        for token, candidates in zip(
            frozen_row["causal_tokens"], frozen_row["candidate_ids"]
        ):
            if token not in candidates:
                causal_outside_candidates += 1
        prediction = (
            frozen_row["direct_tokens"]
            if row["method"] == "direct_inverse"
            else frozen_row["causal_tokens"]
        )
        if row["correct_tokens"] != sum(
            int(left == right) for left, right in zip(prediction, expected)
        ):
            inconsistencies += 1
    if inconsistencies or causal_outside_candidates or direct_not_first:
        raise RuntimeError("frozen candidates, ranks, or metrics are inconsistent")

    query_state = load_file(frozen / "queries.safetensors", device="cpu")
    expected_keys = {
        f"{condition}.cut{cut}"
        for condition in ("matched_public", "unavailable_target_lora")
        for cut in (0, 4, 8)
    }
    if set(query_state) != expected_keys:
        raise RuntimeError("query arms changed")
    if any(tuple(value.shape) != (64, 39, 2048) for value in query_state.values()):
        raise RuntimeError("query geometry changed")

    metrics = json.loads(
        (root / "experiments/TRR-0001/metrics.json").read_text(encoding="utf-8")
    )
    if metrics["records"] != 64 or metrics["scored_tokens_per_arm"] != 2496:
        raise RuntimeError("aggregate metric coverage changed")
    cut0 = [
        row
        for row in metrics["aggregates"]
        if row["cut_depth"] == 0
    ]
    if any(row["token_accuracy"] != 1.0 for row in cut0):
        raise RuntimeError("embedding-boundary sanity condition failed")
    evidence = json.loads(
        (frozen / "reconstructor_evidence.json").read_text(encoding="utf-8")
    )
    if evidence["target_prefix_calls"] != 0 or evidence["causal_candidate_simulations"] != 239616:
        raise RuntimeError("target access or candidate budget changed")

    json_paths = sorted(
        list((root / "experiments/TRR-0001").glob("*.json"))
        + [root / "coordination/STATE.json"]
    )
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))

    print(
        json.dumps(
            {
                "status": "all_integrity_checks_passed",
                "exit_status": 0,
                "receipt_sha256": sha256_file(receipt_path),
                "frozen_entries": len(receipt["entries"]),
                "frozen_bytes": sum(entry["bytes"] for entry in receipt["entries"]),
                "reconstruction_rows": len(reconstruction_rows),
                "per_record_metric_rows": len(per_record_rows),
                "truth_records_opened_after_gate": len(truth_rows),
                "scored_tokens_per_arm": 2496,
                "candidate_rank_consistency_failures": inconsistencies,
                "causal_outputs_outside_frozen_candidates": causal_outside_candidates,
                "target_prefix_calls": evidence["target_prefix_calls"],
                "charter_sha256": sha256_file(charter),
                "request_sha256": sha256_file(request),
                "json_files_parsed": len(json_paths),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
