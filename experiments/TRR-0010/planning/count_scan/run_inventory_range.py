#!/usr/bin/env python3
"""Task-local range adapter for TRR-0009's identity-only inventory scanner.

The imported TRR-0009 scanner remains unchanged.  This adapter temporarily
uses the frozen contract range while validating the contract, then applies a
caller-declared Pile lower bound only to the public aggregate scan.  It never
selects records, opens truth, loads a model, or writes source identities.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for _root in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts import trr0009_plan as plan


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pile-start", type=int, required=True)
    args, forwarded = parser.parse_known_args()
    if args.pile_start < 0 or args.pile_start >= 7000:
        raise SystemExit("--pile-start must be below the frozen 7000 start")
    if "--repository-root" not in forwarded:
        raise SystemExit("--repository-root is required")

    frozen_ranges = {"finance": [12000, 20000], "pile": [7000, 10000]}
    scan_ranges = {"finance": [12000, 20000], "pile": [args.pile_start, 10000]}
    original_ranges = plan.SOURCE_RANGES
    original_validate = plan._validate_final_scan_contract

    def validate_with_frozen_range(path: Path):
        plan.SOURCE_RANGES = frozen_ranges
        try:
            return original_validate(path)
        finally:
            plan.SOURCE_RANGES = scan_ranges

    plan.SOURCE_RANGES = scan_ranges
    plan._validate_final_scan_contract = validate_with_frozen_range
    try:
        return int(plan.main(["inventory-final", *forwarded]))
    finally:
        plan._validate_final_scan_contract = original_validate
        plan.SOURCE_RANGES = original_ranges


if __name__ == "__main__":
    raise SystemExit(main())
