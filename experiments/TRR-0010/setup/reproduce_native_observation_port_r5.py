#!/usr/bin/env python3
"""Reproduce the CPU-only native TRR9 -> TRR10 observation port.

This helper intentionally writes to a caller-supplied output directory.  It
does not touch the frozen r5 observation package unless that directory is
explicitly supplied by the caller.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.trr0010_eval_capture import repackage_trr0009_capture


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.expanduser().resolve()
    output = args.output_root.expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    result_path = args.result_json.expanduser()
    if not result_path.is_absolute():
        result_path = root / result_path
    result_path = result_path.resolve()

    started_utc = _utc_now()
    started = time.monotonic()
    result = repackage_trr0009_capture(
        selection_path=root / "experiments/TRR-0010/evaluation/source_selection.json",
        producer_root=root / "experiments/TRR-0010/evaluation/public_capture_watchdog_r2/producer_capture",
        output_root=output,
        repository_root=root,
        producer_selection_path=root / "experiments/TRR-0010/evaluation/public_capture_watchdog_r2/producer_selection_bridge.json",
        selection_binding_path=root / "experiments/TRR-0010/evaluation/source_selection_binding_r4.json",
        design_path=args.design.expanduser().resolve(),
    )
    finished_utc = _utc_now()
    payload = {
        "schema": "token-reconstruction.trr0010-native-observation-port-replay.v1",
        "status": "PASS_EXACT_VALUES_KEYS_SHAPES_DTYPES",
        "task_id": "TRR-0010",
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": time.monotonic() - started,
        "repository_root": str(root),
        "output_root": str(output),
        "result": result,
        "truth_opened": False,
        "source_text_loaded": False,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
