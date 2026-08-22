#!/usr/bin/env python3
"""Fail-closed validation for the TRR-0001-R2 dual-benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_HASHES = {
    "RESEARCH_CHARTER.md": "ab0fbe9dfad39eddee48c14f4cb8201f8c3f02d1c58668d8a8e59be5a250700d",
    "coordination/requests/TRR-0001-R2.md": "a7baf9ee25604f14535a986357dc56ddc0efa599bb3e0b880ac3fd2aad7bfb7e",
    "research/DUAL_BENCHMARK_PROTOCOL.md": "98380e7298d0720cd9ef12358c651a3fcb419b029703975a87e14a3093ec6d33",
    "research/dual_benchmark_registry.json": "dfbf5f4c6129f1213337c2b8eabaa6b13c7313ef1b429db4fccf861a0a27038e",
    "experiments/TRR-0001/revision-r2/dual_benchmark_matrix.json": "5594b3a4e3af6e27d28c8f0267f7d71c0d8f3715672a36202909be41e9a0dd66",
}

EXPECTED_COUNTS = {
    ("clean-pile-lora-64x40", "direct_inverse_k16"): {
        "records": 64,
        "scored_tokens": 2496,
        "covered_tokens": 2496,
        "correct_tokens": 1615,
        "candidate_hits": 2113,
        "exact_records": 0,
    },
    ("clean-pile-lora-64x40", "causal_public_surrogate_k16"): {
        "records": 64,
        "scored_tokens": 2496,
        "covered_tokens": 2496,
        "correct_tokens": 2096,
        "candidate_hits": 2113,
        "exact_records": 0,
    },
    ("clean-pile-lora-64x40", "strict_bos_adaptive_a1_a2"): {
        "records": 64,
        "scored_tokens": 2496,
        "covered_tokens": 331,
        "correct_tokens": 331,
        "candidate_hits": 2492,
        "exact_records": 0,
    },
    ("historical-finance-strict-bos-128x128", "direct_inverse_k16"): {
        "records": 128,
        "scored_tokens": 13990,
        "covered_tokens": 13990,
        "correct_tokens": 8534,
        "candidate_hits": 11422,
        "exact_records": 0,
    },
    ("historical-finance-strict-bos-128x128", "causal_public_surrogate_k16"): {
        "records": 128,
        "scored_tokens": 13990,
        "covered_tokens": 13990,
        "correct_tokens": 11349,
        "candidate_hits": 11422,
        "exact_records": 0,
    },
    ("historical-finance-strict-bos-128x128", "strict_bos_adaptive_a1_a2"): {
        "records": 128,
        "scored_tokens": 13990,
        "covered_tokens": 13744,
        "correct_tokens": 13741,
        "candidate_hits": 13980,
        "exact_records": 122,
    },
}

EXPECTED_PREDICTION = {
    "bytes": 40257376,
    "sha256": "fd35ee3dbddbe38d8d6a3d5877911cef532b53a61ecd1c685569bd13c4f8f65d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate(root: Path, prediction_path: Path) -> dict[str, Any]:
    verified_hashes: dict[str, dict[str, Any]] = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = root / relative
        require(path.is_file(), f"missing required file: {relative}")
        actual = sha256(path)
        require(actual == expected, f"SHA-256 mismatch for {relative}: {actual}")
        verified_hashes[relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }

    registry = load_json(root / "research/dual_benchmark_registry.json")
    setup_ids = [entry["id"] for entry in registry["setups"]]
    method_ids = [entry["id"] for entry in registry["methods"]]
    require(len(setup_ids) == len(set(setup_ids)), "duplicate setup ID")
    require(len(method_ids) == len(set(method_ids)), "duplicate method ID")
    cartesian = set(itertools.product(setup_ids, method_ids))
    registered = {
        (entry["setup_id"], entry["method_id"])
        for entry in registry["required_cells"]
    }
    require(len(registry["required_cells"]) == len(registered), "duplicate required cell")
    require(registered == cartesian, "registry required_cells is not setups x methods")
    require(registry["cross_setup_pooling"] is False, "cross-setup pooling must be false")

    matrix = load_json(root / "experiments/TRR-0001/revision-r2/dual_benchmark_matrix.json")
    require(matrix["task_id"] == "TRR-0001", "wrong task ID")
    require(matrix["revision_id"] == "TRR-0001-R2", "wrong revision ID")
    require(matrix["matrix_complete"] is True, "matrix does not claim completeness")
    require(matrix["status"] == "RETROSPECTIVE_COMPLETE_MATRIX", "wrong matrix status")
    require(matrix["setup_order"] == setup_ids, "matrix setup order differs from registry")
    require(matrix["method_order"] == method_ids, "matrix method order differs from registry")

    actual_cells = {
        (setup_id, method_id)
        for setup_id, methods in matrix["matrix"].items()
        for method_id in methods
    }
    require(actual_cells == cartesian, "matrix cells are not the registered cartesian product")
    require(len(actual_cells) == 6, "matrix must contain exactly six cells")

    summary: dict[str, dict[str, Any]] = {}
    for cell, expected_counts in EXPECTED_COUNTS.items():
        setup_id, method_id = cell
        entry = matrix["matrix"][setup_id][method_id]
        require("failed" not in entry["status"], f"failed matrix cell: {cell}")
        metrics = entry["metrics"]
        for key, expected in expected_counts.items():
            require(metrics[key] == expected, f"{cell} {key}: {metrics[key]} != {expected}")
        require(len(entry["per_record"]) == expected_counts["records"], f"{cell} record count")
        require(
            sum(row["scored_tokens"] for row in entry["per_record"])
            == expected_counts["scored_tokens"],
            f"{cell} per-record scored-token total",
        )
        summary.setdefault(setup_id, {})[method_id] = {
            key: metrics[key]
            for key in (
                "token_accuracy",
                "coverage",
                "selective_accuracy",
                "candidate_recall",
                "exact_records",
                "records",
            )
        }

    clean_gate = matrix["semantic_checks"]["clean_native_prediction_reproduction"]
    require(clean_gate["direct_prediction_mismatches"] == 0, "clean direct mismatch")
    require(clean_gate["causal_prediction_mismatches"] == 0, "clean causal mismatch")
    historical_gate = matrix["semantic_checks"][
        "historical_strict_native_aggregate_reproduction"
    ]
    require(all(historical_gate.values()), "historical strict semantic gate failed")

    require(prediction_path.is_file(), f"missing prediction artifact: {prediction_path}")
    prediction_hash = sha256(prediction_path)
    prediction_bytes = prediction_path.stat().st_size
    require(prediction_bytes == EXPECTED_PREDICTION["bytes"], "prediction byte count mismatch")
    require(prediction_hash == EXPECTED_PREDICTION["sha256"], "prediction SHA-256 mismatch")
    recorded_prediction = matrix["artifacts"]["prediction_freeze"]
    require(recorded_prediction["bytes"] == prediction_bytes, "recorded prediction bytes mismatch")
    require(recorded_prediction["sha256"] == prediction_hash, "recorded prediction hash mismatch")

    return {
        "schema": "token-reconstruction.trr0001-r2-matrix-validation.v1",
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R2",
        "status": "PASS_ALL_R2_MATRIX_CHECKS",
        "checked_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "registered_setups": setup_ids,
        "registered_methods": method_ids,
        "required_cells": len(cartesian),
        "matrix_cells": len(actual_cells),
        "semantic_checks": matrix["semantic_checks"],
        "metrics": summary,
        "verified_hashes": verified_hashes,
        "prediction_artifact": {
            "path": str(prediction_path),
            "bytes": prediction_bytes,
            "sha256": prediction_hash,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--prediction-artifact",
        type=Path,
        default=Path("outputs/TRR-0001-R2/dual-benchmark/predictions.safetensors"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    prediction_path = args.prediction_artifact
    if not prediction_path.is_absolute():
        prediction_path = root / prediction_path

    try:
        evidence = validate(root, prediction_path)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"FAIL_R2_MATRIX_VALIDATION: {exc}")
        return 1

    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
