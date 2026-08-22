#!/usr/bin/env python3
"""Post-freeze selection-key/mapping reveal for TRR-0001-R1."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from token_reconstruction.blind_commitment import (
    reveal_document,
    validate_public_commitment,
    verify_reveal,
)
from token_reconstruction.experiment_runtime import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    command_record,
    file_record,
    load_json,
    utc_now,
    write_json_exclusive,
)
from token_reconstruction.freeze import verify_freeze_receipt
from token_reconstruction.isolation import validate_isolation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--public-commitment", type=Path, required=True)
    parser.add_argument("--private-selection", type=Path, required=True)
    parser.add_argument("--original-plan", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--revealed-utc")
    return parser.parse_args()


def excluded_indices(original: dict) -> set[int]:
    splits = original["data"]["selection"]["splits"]
    required = {
        "target_update_train": 64,
        "inverse_train": 128,
        "development": 32,
        "blind_evaluation": 64,
    }
    if set(splits) != set(required):
        raise RuntimeError("original split set changed")
    values = [
        int(row["index"])
        for name, expected in required.items()
        for row in splits[name]["records"]
        if len(splits[name]["records"]) == expected
    ]
    if len(values) != 288 or len(set(values)) != 288:
        raise RuntimeError("original exclusion union changed")
    return set(values)


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    if args.reveal.exists() or args.verification.exists():
        raise RuntimeError("reveal outputs are create-only")
    receipt = verify_freeze_receipt(args.receipt, repository_root=root)
    if (
        receipt.get("metadata", {}).get("selection_revealed") is not False
        or receipt.get("metadata", {}).get("truth_opened") is not False
        or receipt.get("metadata", {}).get("access_manifests_verified") is not True
    ):
        raise RuntimeError("freeze receipt is not the pre-reveal access gate")
    frozen_root = root / receipt["frozen_root"]
    direct_access = load_json(
        frozen_root / "direct_inverse" / "access_manifest.json"
    )
    causal_access = load_json(
        frozen_root
        / "causal_public_surrogate_search"
        / "access_manifest.json"
    )
    validate_isolation_manifest(direct_access, method="direct_inverse")
    validate_isolation_manifest(
        causal_access, method="causal_public_surrogate_search"
    )

    public = load_json(args.public_commitment)
    validate_public_commitment(public)
    private = load_json(args.private_selection)
    revealed_utc = args.revealed_utc or utc_now()
    reveal = reveal_document(private, revealed_utc=revealed_utc)
    if any("token_ids" in record for record in reveal["records"]):
        raise RuntimeError("mapping reveal must precede truth and exclude token IDs")
    write_json_exclusive(args.reveal, reveal)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    dataset = load_dataset(
        DATASET_ID, revision=DATASET_REVISION, split="train"
    )
    original = load_json(args.original_plan)
    excluded = excluded_indices(original)
    result = verify_reveal(
        public=public,
        reveal=reveal,
        dataset_revision=DATASET_REVISION,
        excluded_indices=excluded,
        dataset_rows=[str(dataset[index]["text"]) for index in range(len(dataset))],
        tokenizer=tokenizer,
    )
    if not all(
        result[key]
        for key in (
            "verified",
            "disjoint_from_original_records",
            "opaque_order_verified",
            "eligibility_verified",
        )
    ):
        raise RuntimeError("selection reveal verification did not pass")
    write_json_exclusive(
        args.verification,
        {
            "schema": "token-reconstruction.trr0001-r1-selection-verification.v1",
            "task_id": "TRR-0001",
            "revision_id": "TRR-0001-R1",
            "command": command_record(),
            "revealed_utc": revealed_utc,
            "exit_status": 0,
            "freeze_receipt": file_record(args.receipt, root=root),
            "preregistration_commit": receipt["preregistration_commit"],
            "access_manifests_reverified_before_private_mapping_read": True,
            "public_commitment": file_record(args.public_commitment, root=root),
            "reveal": file_record(args.reveal, root=root),
            "commitment": public["commitment"],
            "records": result["records"],
            "original_excluded_records": len(excluded),
            "disjoint_from_original_records": True,
            "opaque_order_verified": True,
            "eligibility_verified": True,
            "selection_algorithm_verified": True,
            "token_ids_disclosed_in_mapping_reveal": False,
            "truth_sidecar_read": False,
            "result": "PASS_SELECTION_REVEALED_AFTER_VERIFIED_FREEZE",
        },
    )
    print(
        {
            "status": "selection_mapping_revealed_and_verified",
            "records": result["records"],
            "commitment": public["commitment"],
            "disjoint": True,
            "truth_sidecar_read": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
