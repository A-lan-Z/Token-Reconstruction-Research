"""CLI for the truth-gated TRR-P10 paired error inventory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from .execution import (
        ExecutionError,
        build_error_inventory,
        load_truth_after_freeze,
        validate_frozen_package,
        write_inventory,
    )
except ImportError:  # direct ``python scripts/trr_p10/evaluate.py`` invocation
    from scripts.trr_p10.execution import (
        ExecutionError,
        build_error_inventory,
        load_truth_after_freeze,
        validate_frozen_package,
        write_inventory,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True, help="hash-bound public freeze/package JSON")
    parser.add_argument("--registration", type=Path, default=None, help="optional explicit registration JSON")
    parser.add_argument("--truth-descriptor", type=Path, required=True, help="post-freeze truth descriptor JSON")
    parser.add_argument("--truth", type=Path, default=None, help="optional explicit truth sidecar path")
    parser.add_argument("--root", type=Path, default=None, help="repository root used to resolve relative bindings")
    parser.add_argument("--output", type=Path, required=True, help="create-only paired inventory JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        frozen = validate_frozen_package(
            args.freeze,
            repository_root=args.root,
            registration_path=args.registration,
        )
        truth, truth_binding = load_truth_after_freeze(
            frozen,
            args.truth_descriptor,
            truth_path=args.truth,
            repository_root=args.root,
        )
        inventory = build_error_inventory(frozen, truth, truth_binding=truth_binding)
        output = write_inventory(args.output, inventory)
    except ExecutionError as exc:
        print(f"TRR-P10 evaluation failed closed: {exc}", file=sys.stderr)
        return 2
    print(output["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
