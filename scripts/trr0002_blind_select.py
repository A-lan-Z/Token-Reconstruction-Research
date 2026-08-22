#!/usr/bin/env python3
"""Create the evaluator-private TRR-0002 confirmation split and public commitment."""

from __future__ import annotations

import argparse
from pathlib import Path
import secrets
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from token_reconstruction.blind_commitment import (
    commitment_digest,
    select_private_records,
)
from token_reconstruction.experiment_runtime import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    load_json,
    utc_now,
    write_json_exclusive,
)


PUBLIC_SCHEMA = "token-reconstruction.trr0002-selection-commitment.v1"
PRIVATE_SCHEMA = "token-reconstruction.trr0002-private-selection.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--previous-private-selection", type=Path, required=True)
    parser.add_argument("--calibration-records", type=Path, required=True)
    parser.add_argument("--public-commitment", type=Path, required=True)
    parser.add_argument("--private-selection", type=Path, required=True)
    parser.add_argument("--created-utc")
    return parser.parse_args()


def original_indices(plan: dict[str, Any]) -> set[int]:
    splits = plan["data"]["selection"]["splits"]
    expected = {
        "target_update_train": 64,
        "inverse_train": 128,
        "development": 32,
        "blind_evaluation": 64,
    }
    if set(splits) != set(expected):
        raise RuntimeError("original split declarations changed")
    result: list[int] = []
    for name, count in expected.items():
        rows = splits[name]["records"]
        if len(rows) != count:
            raise RuntimeError(f"original split geometry changed: {name}")
        result.extend(int(row["index"]) for row in rows)
    if len(result) != 288 or len(set(result)) != 288:
        raise RuntimeError("original records overlap or are duplicated")
    return set(result)


def previous_indices(selection: dict[str, Any]) -> set[int]:
    rows = selection.get("records")
    if not isinstance(rows, list) or len(rows) != 64:
        raise RuntimeError("previous fresh selection geometry changed")
    result = {int(row["dataset_index"]) for row in rows}
    if len(result) != 64:
        raise RuntimeError("previous fresh selection contains duplicate records")
    return result


def calibration_indices(records: dict[str, Any]) -> set[int]:
    if records.get("disjoint") is not True:
        raise RuntimeError("public calibration disjointness is absent")
    rows = [*records["development"], *records["update_train"]]
    result = {int(row["dataset_index"]) for row in rows}
    if len(result) != 96:
        raise RuntimeError("public calibration record geometry changed")
    return result


def main() -> int:
    args = parse_args()
    if args.public_commitment.exists() or args.private_selection.exists():
        raise RuntimeError("fresh selection artifacts are create-only")
    original = original_indices(load_json(args.source_plan))
    previous = previous_indices(load_json(args.previous_private_selection))
    calibration = calibration_indices(load_json(args.calibration_records))
    if not calibration.issubset(original):
        raise RuntimeError("public calibration records unexpectedly escaped original splits")
    if original & previous:
        raise RuntimeError("previous clean split is not disjoint from original splits")
    excluded = original | previous | calibration

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    rows = (
        (
            index,
            str(dataset[index]["text"]),
            tokenizer(str(dataset[index]["text"]), add_special_tokens=False)[
                "input_ids"
            ],
        )
        for index in range(len(dataset))
    )
    created_utc = args.created_utc or utc_now()
    key = secrets.token_bytes(32)
    selected_legacy = select_private_records(
        key=key,
        dataset_revision=DATASET_REVISION,
        rows=rows,
        excluded_indices=excluded,
    )
    selected = selected_legacy
    if {int(row["dataset_index"]) for row in selected} & excluded:
        raise RuntimeError("new blind selection overlaps an excluded record")
    opaque_order = [row["record_id"] for row in selected]
    expected_order = [f"blind-r1-{position:06d}" for position in range(1, 65)]
    if opaque_order != expected_order:
        raise RuntimeError("TRR-0002 opaque record order changed")
    digest = commitment_digest(key, selected)
    public = {
        "schema": PUBLIC_SCHEMA,
        "task_id": "TRR-0002",
        "phase_id": "TRR-0002-FRESH-BLIND-CONFIRMATION",
        "created_utc": created_utc,
        "scheme": "HMAC-SHA256 over canonical private mapping with an evaluator-private 256-bit key",
        "commitment": digest,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "train",
        },
        "selection_algorithm": "eligible non-excluded rows ordered by HMAC-SHA256 of revision, row number, and text digest; first 64",
        "eligibility": "at least 39 source tokens; prepend exactly one declared BOS",
        "disjointness": {
            "original_trr0001_records": len(original),
            "previous_fresh_trr0001_r1_records": len(previous),
            "public_calibration_records": len(calibration),
            "unique_excluded_records": len(excluded),
        },
        "record_count": 64,
        "opaque_record_order": opaque_order,
        "source_identity_disclosed": False,
        "selection_key_disclosed": False,
        "reveal_gate": "only after calibrated predictions, confidence, route, sanitized configuration, code, and access evidence are frozen and verified",
    }
    private = {
        "schema": PRIVATE_SCHEMA,
        "task_id": "TRR-0002",
        "created_utc": created_utc,
        "selection_key_hex": key.hex(),
        "records": selected,
    }
    args.private_selection.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(args.private_selection, private)
    args.private_selection.chmod(0o600)
    args.public_commitment.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(args.public_commitment, public)
    print(
        {
            "status": "TRR0002_FRESH_SELECTION_COMMITTED_WITHOUT_SOURCE_DISCLOSURE",
            "schema": PUBLIC_SCHEMA,
            "records": len(selected),
            "excluded_original": len(original),
            "excluded_previous_fresh": len(previous),
            "excluded_total": len(excluded),
            "public_commitment": str(args.public_commitment),
            "private_mode": oct(args.private_selection.stat().st_mode & 0o777),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
