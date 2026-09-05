#!/usr/bin/env python3
"""Run the predeclared first-post-BOS full-vocabulary identity diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from safetensors import safe_open
import torch

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from token_reconstruction.trr_p01 import PrototypeTable  # noqa: E402
from common import (  # noqa: E402
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    TASK_ID,
    VOCAB_SIZE,
    file_record,
    load_json,
    require_create_only_file,
    sha256_file,
    validate_public_plan,
    write_json_exclusive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, default=None)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--prototype-chunk-size", type=int, default=8192)
    return parser.parse_args()


def _load_qualification(build_root: Path) -> tuple[dict[str, Any], torch.Tensor, Path, Path]:
    report_path = build_root / "qualification.json"
    chosen_path = build_root / "qualification_chosen.safetensors"
    report = load_json(report_path)
    if report.get("schema") != "token-reconstruction.trr-p01-qualification.v1" or report.get("truth_opened") is not False:
        raise RuntimeError("qualification report schema or truth state changed")
    ids = report.get("probe_token_ids")
    if not isinstance(ids, list) or len(ids) != 256:
        raise RuntimeError("qualification probes are missing")
    if chosen_path.is_symlink() or not chosen_path.is_file():
        raise RuntimeError("chosen qualification output is unavailable")
    with safe_open(chosen_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"prototype_outputs"}:
            raise RuntimeError("qualification output fields changed")
        metadata = handle.metadata() or {}
        if metadata.get("schema") != "token-reconstruction.trr-p01-qualification-output.v1" or metadata.get("truth_opened") != "false":
            raise RuntimeError("qualification output truth state changed")
        values = handle.get_tensor("prototype_outputs")
    if values.dtype != torch.bfloat16 or tuple(values.shape) != (256, HIDDEN_SIZE):
        raise RuntimeError("qualification output geometry or dtype changed")
    return report, values, report_path, chosen_path


def main() -> int:
    args = parse_args()
    plan = load_json(args.plan)
    validate_public_plan(plan)
    build_root = args.build_root.resolve()
    report, queries, qualification_path, chosen_path = _load_qualification(build_root)
    probe_ids = [int(value) for value in report["probe_token_ids"]]
    prototype_path = (args.prototype or (build_root / "boundary_prototypes.safetensors")).resolve()
    table = PrototypeTable.load(
        prototype_path,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_cut_depth=CUT_DEPTH,
        expected_vocab_size=VOCAB_SIZE,
        expected_hidden_size=HIDDEN_SIZE,
    )
    if args.query_chunk_size <= 0 or args.prototype_chunk_size <= 0:
        raise RuntimeError("lookup chunks must be positive")
    arms: list[dict[str, Any]] = []
    for metric in ("cosine", "l2"):
        nearest = table.nearest(
            queries,
            metric=metric,
            query_chunk_size=args.query_chunk_size,
            prototype_chunk_size=args.prototype_chunk_size,
        )
        predicted = [int(value) for value in nearest.predictions.tolist()]
        mismatches = []
        for position, (expected, actual, score, margin) in enumerate(
            zip(probe_ids, predicted, nearest.scores.tolist(), nearest.margins.tolist()), 1
        ):
            if expected != actual:
                mismatches.append(
                    {
                        "probe_index": position,
                        "expected_token_id": expected,
                        "predicted_token_id": actual,
                        "score": float(score),
                        "margin": float(margin),
                        "diagnosis": "exact_score_tie_or_collision" if float(margin) == 0.0 else "nearest_nonidentity",
                    }
                )
        arms.append(
            {
                "metric": metric,
                "predictions": predicted,
                "expected": probe_ids,
                "identity_count": sum(left == right for left, right in zip(predicted, probe_ids)),
                "probe_count": len(probe_ids),
                "mismatches": mismatches,
                "status": "PASS" if not mismatches else "DIAGNOSTIC_NONIDENTITY",
            }
        )
    status = "PASS" if all(arm["status"] == "PASS" for arm in arms) else "DIAGNOSTIC_NONIDENTITY"
    require_create_only_file(args.output.resolve())
    write_json_exclusive(
        args.output.resolve(),
        {
            "schema": "token-reconstruction.trr-p01-bos-identity.v1",
            "task_id": TASK_ID,
            "truth_opened": False,
            "status": status,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "cut_depth": CUT_DEPTH, "vocab_size": VOCAB_SIZE, "hidden_size": HIDDEN_SIZE},
            "qualification": file_record(qualification_path),
            "qualification_output": file_record(chosen_path),
            "prototype": file_record(prototype_path),
            "query_chunk_size": args.query_chunk_size,
            "prototype_chunk_size": args.prototype_chunk_size,
            "arms": arms,
            "diagnostic_rule": "a nonidentity with zero runner-up margin is reported as an exact tie/collision; no truth is consulted",
        },
    )
    print({"status": status, "output": str(args.output.resolve())})
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
