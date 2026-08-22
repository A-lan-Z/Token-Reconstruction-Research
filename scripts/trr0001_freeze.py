#!/usr/bin/env python3
"""Create and independently verify the immutable TRR-0001 freeze receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from token_reconstruction.experiment_runtime import (
    command_record,
    file_record,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from token_reconstruction.freeze import (
    create_freeze_receipt,
    verify_freeze_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--validation-record", type=Path, required=True)
    parser.add_argument("--preregistration-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created = utc_now()
    payload = create_freeze_receipt(
        repository_root=args.repository_root,
        frozen_root=args.frozen_root,
        plan_path=args.plan,
        receipt_path=args.receipt,
        preregistration_commit=args.preregistration_commit,
        created_utc=created,
        metadata={
            "task_id": "TRR-0001",
            "truth_opened": False,
            "command": command_record(),
            "contract": "outputs, candidates, queries, method state, configuration, order, routing, and timings",
        },
    )
    verified = verify_freeze_receipt(
        args.receipt, repository_root=args.repository_root
    )
    if verified != payload:
        raise RuntimeError("freeze receipt verification changed parsed payload")
    write_json_exclusive(
        args.validation_record,
        {
            "schema": "token-reconstruction.freeze-validation.v1",
            "task_id": "TRR-0001",
            "verified_utc": utc_now(),
            "exit_status": 0,
            "entry_count": len(verified["entries"]),
            "frozen_root": verified["frozen_root"],
            "preregistration_commit": verified["preregistration_commit"],
            "plan": verified["plan"],
            "receipt": file_record(args.receipt, root=args.repository_root),
            "receipt_sha256": sha256_file(args.receipt),
            "truth_opened": False,
        },
    )
    print(
        {
            "status": "freeze_verified",
            "entries": len(verified["entries"]),
            "receipt_sha256": sha256_file(args.receipt),
            "truth_opened": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
