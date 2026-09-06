#!/usr/bin/env python3
"""Open Stage-1 truth only after verifying frozen TRR-P03 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
for _path in (_SOURCE_ROOT, _SOURCE_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from token_reconstruction.trr_p03.scoring import (
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    ScoringError,
    score_prediction_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--paired-prediction-root", type=Path, default=None)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--truth-index", type=Path, default=None)
    parser.add_argument("--pre-score-receipt", type=Path, default=None)
    parser.add_argument("--implementation-commit", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--records-per-stratum", type=int, default=6)
    parser.add_argument("--allow-unequal-strata", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_draws <= 0 or args.bootstrap_seed < 0:
        raise ScoringError("bootstrap draws must be positive and seed non-negative")
    records_per_stratum: int | None = None if args.allow_unequal_strata else args.records_per_stratum
    if records_per_stratum is not None and records_per_stratum <= 0:
        raise ScoringError("records per stratum must be positive")
    result = score_prediction_bundle(
        prediction_root=args.prediction_root,
        paired_prediction_root=args.paired_prediction_root,
        truth_path=args.truth,
        truth_index_path=args.truth_index,
        output_root=args.output_root,
        pre_score_receipt_path=args.pre_score_receipt,
        implementation_commit=args.implementation_commit,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
        records_per_stratum=records_per_stratum,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScoringError as exc:
        print(f"TRR-P03 scoring failed: {exc}")
        raise SystemExit(2)
