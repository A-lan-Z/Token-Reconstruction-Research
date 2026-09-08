#!/usr/bin/env python3
"""Emit one TRR-0012 fixed-arm plan without loading tensors or fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.trr0012.fixed_control_wrapper import build_run_plan, write_run_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("current_fixed_replication_1", "expanded_fixed_replication_1"), required=True)
    parser.add_argument("--actual-valid-positions", type=int, required=True)
    parser.add_argument("--artifacts-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifacts = json.loads(args.artifacts_json.read_text(encoding="utf-8"))
    if not isinstance(artifacts, dict):
        raise SystemExit("artifact descriptor must be a JSON object")
    plan = build_run_plan(
        arm=args.arm,
        actual_valid_positions=args.actual_valid_positions,
        artifacts=artifacts,
        output_root=_REPOSITORY_ROOT / "outputs" / "TRR-0012",
    )
    path = write_run_plan(plan, args.output)
    print(json.dumps({"status": "PLAN_WRITTEN", "arm": args.arm, "path": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
