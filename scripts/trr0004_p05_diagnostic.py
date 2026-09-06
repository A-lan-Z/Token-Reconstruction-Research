#!/usr/bin/env python3
"""Prepare and run the bounded, truth-free TRR-P05 diagnostics (canonical P05 runner).

``sample`` only loads public pool metadata/tensors and the cached public
teacher rows; it writes the frozen sample ledger and never loads a model.
``diagnose`` consumes that ledger and stored P04 checkpoints.  It performs
forward/backward calculations without an optimizer and writes create-only
receipts.  It never opens evaluator truth or target data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from token_reconstruction.p05_diagnostics import (
    P05DiagnosticError,
    prepare_sample_from_paths,
    parse_schedule_args,
    run_diagnostics,
)


def _common_public(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--correction-observations", type=Path, required=True)
    parser.add_argument("--correction-records", type=Path, required=True)
    parser.add_argument("--replay-observations", type=Path, required=True)
    parser.add_argument("--replay-records", type=Path, required=True)
    parser.add_argument("--teacher-evidence", type=Path, required=True)
    parser.add_argument("--schedule", action="append", required=True, metavar="SEED=PATH", help="one P04 schedule for each seed (1737=PATH and 2711=PATH)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="freeze the public sample ledger without a model")
    _common_public(sample)
    sample.add_argument("--output", type=Path, required=True)
    diagnose = sub.add_parser("diagnose", help="run truth-free forward/backward diagnostics")
    _common_public(diagnose)
    diagnose.add_argument("--sample-index", type=Path, required=True)
    diagnose.add_argument("--candidate-preparation", type=Path, required=True)
    diagnose.add_argument("--embedding-table", type=Path, required=True)
    diagnose.add_argument("--state-manifest", type=Path, required=True)
    diagnose.add_argument("--affine-initial", type=Path, required=True)
    diagnose.add_argument("--output-root", type=Path, required=True)
    diagnose.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    diagnose.add_argument("--mode", choices=("qualify", "full"), default="full")
    diagnose.add_argument("--threads", type=int, default=4)
    diagnose.add_argument("--interop-threads", type=int, default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    schedules = parse_schedule_args(args.schedule)
    if args.command == "sample":
        sample = prepare_sample_from_paths(
            correction_observations=args.correction_observations,
            correction_records=args.correction_records,
            replay_observations=args.replay_observations,
            replay_records=args.replay_records,
            teacher_evidence_path=args.teacher_evidence,
            schedule_paths=schedules,
            output=args.output,
        )
        print(json.dumps({"status": "PASS", "sample_index": str(args.output.expanduser().resolve()), "selection_sha256": sample["selection_sha256"], "forward_count": sample["forward"]["total_count"], "gradient_batch_count": sample["gradient"]["batch_count"]}, sort_keys=True))
        return 0
    receipt = run_diagnostics(
        sample_path=args.sample_index,
        correction_observations=args.correction_observations,
        correction_records=args.correction_records,
        replay_observations=args.replay_observations,
        replay_records=args.replay_records,
        teacher_evidence_path=args.teacher_evidence,
        candidate_preparation=args.candidate_preparation,
        embedding_table_path=args.embedding_table,
        state_manifest_path=args.state_manifest,
        affine_initial_path=args.affine_initial,
        schedule_paths=schedules,
        output_root=args.output_root,
        device=__import__("torch").device(args.device),
        mode=args.mode,
        threads=args.threads,
        interop_threads=args.interop_threads,
    )
    print(json.dumps({"status": receipt["status"], "mode": receipt["mode"], "output_root": str(args.output_root.expanduser().resolve()), "forward_states": len(receipt["states"]["forward"]), "gradient_cells": len(receipt["states"]["gradient"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (P05DiagnosticError, RuntimeError) as exc:
        print(f"P05 diagnostic failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
