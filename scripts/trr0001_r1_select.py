#!/usr/bin/env python3
"""Create the evaluator-private fresh split and its hiding public commitment."""

from __future__ import annotations

import argparse
from pathlib import Path
import secrets

from datasets import load_dataset
from transformers import AutoTokenizer

from token_reconstruction.blind_commitment import (
    private_selection_document,
    public_commitment,
    validate_public_commitment,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-plan", type=Path, required=True)
    parser.add_argument("--public-commitment", type=Path, required=True)
    parser.add_argument("--private-selection", type=Path, required=True)
    parser.add_argument("--created-utc")
    return parser.parse_args()


def original_excluded_indices(plan: dict) -> set[int]:
    try:
        splits = plan["data"]["selection"]["splits"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("original split declarations are unavailable") from exc
    required = {
        "target_update_train": 64,
        "inverse_train": 128,
        "development": 32,
        "blind_evaluation": 64,
    }
    if set(splits) != set(required):
        raise RuntimeError("original split set changed")
    indices: list[int] = []
    for name, expected_count in required.items():
        records = splits[name].get("records")
        if not isinstance(records, list) or len(records) != expected_count:
            raise RuntimeError(f"original split geometry changed: {name}")
        indices.extend(int(row["index"]) for row in records)
    if len(indices) != 288 or len(set(indices)) != 288:
        raise RuntimeError("original source records are absent, duplicated, or overlapping")
    return set(indices)


def main() -> int:
    args = parse_args()
    if args.public_commitment.exists() or args.private_selection.exists():
        raise RuntimeError("selection artifacts are create-only")
    created_utc = args.created_utc or utc_now()
    original = load_json(args.original_plan)
    excluded = original_excluded_indices(original)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    rows = (
        (
            index,
            str(dataset[index]["text"]),
            tokenizer(
                str(dataset[index]["text"]), add_special_tokens=False
            )["input_ids"],
        )
        for index in range(len(dataset))
    )
    from token_reconstruction.blind_commitment import select_private_records

    key = secrets.token_bytes(32)
    records = select_private_records(
        key=key,
        dataset_revision=DATASET_REVISION,
        rows=rows,
        excluded_indices=excluded,
    )
    public = public_commitment(
        key=key,
        records=records,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        created_utc=created_utc,
    )
    validate_public_commitment(public)
    private = private_selection_document(
        key=key, records=records, created_utc=created_utc
    )
    write_json_exclusive(args.private_selection, private)
    args.private_selection.chmod(0o600)
    write_json_exclusive(args.public_commitment, public)
    print(
        {
            "status": "fresh_selection_committed_without_source_disclosure",
            "records": len(records),
            "excluded_original_records": len(excluded),
            "public_commitment": str(args.public_commitment),
            "private_selection_mode": oct(args.private_selection.stat().st_mode & 0o777),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
