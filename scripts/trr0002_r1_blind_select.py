#!/usr/bin/env python3
"""Create a new disjoint evaluator-private blind split for the frozen R1 winner."""

from __future__ import annotations

import argparse
from pathlib import Path
import secrets
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from token_reconstruction.blind_commitment import commitment_digest, select_private_records
from token_reconstruction.experiment_runtime import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    load_json,
    utc_now,
    write_json_exclusive,
)


PUBLIC_SCHEMA = "token-reconstruction.trr0002-owner-r1-selection-commitment.v1"
PRIVATE_SCHEMA = "token-reconstruction.trr0002-owner-r1-private-selection.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--previous-trr0001-private-selection", type=Path, required=True)
    parser.add_argument(
        "--previous-trr0002-private-selection",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--public-calibration-records", type=Path, required=True)
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
    values: list[int] = []
    for name, count in expected.items():
        rows = splits[name]["records"]
        if len(rows) != count:
            raise RuntimeError(f"original split geometry changed: {name}")
        values.extend(int(row["index"]) for row in rows)
    if len(values) != 288 or len(set(values)) != 288:
        raise RuntimeError("original records overlap or are duplicated")
    return set(values)


def prior_indices(selection: dict[str, Any], *, label: str) -> set[int]:
    rows = selection.get("records")
    if not isinstance(rows, list) or len(rows) != 64:
        raise RuntimeError(f"{label} geometry changed")
    values = {int(row["dataset_index"]) for row in rows}
    if len(values) != 64:
        raise RuntimeError(f"{label} contains duplicate records")
    return values


def calibration_indices(records: dict[str, Any]) -> set[int]:
    if records.get("disjoint") is not True:
        raise RuntimeError("public calibration disjointness is absent")
    rows = [*records["development"], *records["update_train"]]
    values = {int(row["dataset_index"]) for row in rows}
    if len(values) != 96:
        raise RuntimeError("public calibration record geometry changed")
    return values


def main() -> int:
    args = parse_args()
    if args.public_commitment.exists() or args.private_selection.exists():
        raise RuntimeError("new blind selection artifacts are create-only")
    original = original_indices(load_json(args.source_plan))
    previous_trr0001 = prior_indices(
        load_json(args.previous_trr0001_private_selection), label="TRR-0001-R1 blind selection"
    )
    previous_trr0002_sets = [
        prior_indices(load_json(path), label=f"TRR-0002 prior blind selection {index}")
        for index, path in enumerate(args.previous_trr0002_private_selection, start=1)
    ]
    previous_trr0002 = set().union(*previous_trr0002_sets)
    calibration = calibration_indices(load_json(args.public_calibration_records))
    if not calibration.issubset(original):
        raise RuntimeError("public Pile development records escaped the original selection")
    if original & previous_trr0001 or original & previous_trr0002:
        raise RuntimeError("prior evaluation selections are not mutually disjoint")
    excluded = original | previous_trr0001 | previous_trr0002 | calibration

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    rows = (
        (
            index,
            str(dataset[index]["text"]),
            tokenizer(str(dataset[index]["text"]), add_special_tokens=False)["input_ids"],
        )
        for index in range(len(dataset))
    )
    created_utc = args.created_utc or utc_now()
    key = secrets.token_bytes(32)
    selected = select_private_records(
        key=key,
        dataset_revision=DATASET_REVISION,
        rows=rows,
        excluded_indices=excluded,
    )
    if len(selected) != 64 or {int(row["dataset_index"]) for row in selected} & excluded:
        raise RuntimeError("new blind selection overlaps an excluded record")
    opaque_order = [row["record_id"] for row in selected]
    expected_order = [f"blind-r1-{position:06d}" for position in range(1, 65)]
    if opaque_order != expected_order:
        raise RuntimeError("new opaque record order changed")
    digest = commitment_digest(key, selected)
    public = {
        "schema": PUBLIC_SCHEMA,
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R1",
        "phase_id": "TRR-0002-OWNER-R1-FRESH-BLIND-CONFIRMATION",
        "created_utc": created_utc,
        "scheme": "HMAC-SHA256 over canonical private mapping with an evaluator-private 256-bit key",
        "commitment": digest,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "train"},
        "selection_algorithm": "eligible non-excluded rows ordered by HMAC-SHA256 of revision, row number, and text digest; first 64",
        "eligibility": "at least 39 source tokens; prepend exactly one declared BOS",
        "disjointness": {
            "original_trr0001_records": len(original),
            "previous_fresh_trr0001_r1_records": len(previous_trr0001),
            "previous_fresh_trr0002_records": len(previous_trr0002),
            "previous_trr0002_selection_files": len(previous_trr0002_sets),
            "public_pile_development_records": len(calibration),
            "unique_excluded_records": len(excluded),
            "selected_overlap": 0,
        },
        "record_count": 64,
        "opaque_record_order": opaque_order,
        "source_identity_disclosed": False,
        "selection_key_disclosed": False,
        "reveal_gate": "only after the frozen winner predictions, routes, code, sanitized configuration, and access evidence are frozen and verified",
    }
    private = {
        "schema": PRIVATE_SCHEMA,
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R1",
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
            "status": "OWNER_R1_FRESH_SELECTION_COMMITTED_WITHOUT_SOURCE_DISCLOSURE",
            "records": 64,
            "excluded_total": len(excluded),
            "selected_overlap": 0,
            "public_commitment": str(args.public_commitment),
            "private_mode": oct(args.private_selection.stat().st_mode & 0o777),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
