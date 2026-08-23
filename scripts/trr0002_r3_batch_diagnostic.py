#!/usr/bin/env python3
"""Compare two prediction archives exactly without opening target truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def equal_nan_mask(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    equal = left.eq(right)
    if left.is_floating_point():
        equal = equal.logical_or(torch.isnan(left).logical_and(torch.isnan(right)))
    return equal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_path = args.reference_directory / "predictions.safetensors"
    candidate_path = args.candidate_directory / "predictions.safetensors"
    reference_evidence_path = args.reference_directory / "evidence.json"
    candidate_evidence_path = args.candidate_directory / "evidence.json"
    reference = load_file(str(reference_path), device="cpu")
    candidate = load_file(str(candidate_path), device="cpu")
    reference_evidence = load_json(reference_evidence_path)
    candidate_evidence = load_json(candidate_evidence_path)
    if set(reference) != set(candidate):
        raise RuntimeError("prediction archives have different tensor registries")

    tensors: dict[str, Any] = {}
    total_differences = 0
    for key in sorted(reference):
        left = reference[key]
        right = candidate[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"tensor geometry differs for {key}")
        equal = equal_nan_mask(left, right)
        differences = int(equal.logical_not().sum().item())
        total_differences += differences
        tensors[key] = {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "differing_entries_equal_nan": differences,
            "all_entries_equal_nan": differences == 0,
        }

    root = args.repository_root.resolve(strict=True)
    execution_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema": "token-reconstruction.trr0002-owner-r3-batch-diagnostic.v1",
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R3",
        "status": (
            "PASS"
            if total_differences == 0
            else "NON_INVARIANT_RERUN_ENTIRE_MATRIX"
        ),
        "execution_commit": execution_commit,
        "command": ["python3", *sys.argv],
        "target_or_truth_accessed": False,
        "reference": {
            "record_batch_size": reference_evidence.get("record_batch_size"),
            "predictions": file_record(reference_path),
            "evidence": file_record(reference_evidence_path),
        },
        "candidate": {
            "record_batch_size": candidate_evidence.get("record_batch_size"),
            "predictions": file_record(candidate_path),
            "evidence": file_record(candidate_evidence_path),
        },
        "tensor_count": len(tensors),
        "total_differing_entries_equal_nan": total_differences,
        "tensors": tensors,
        "resolution": (
            "Rerun every final matrix cell at common record batch size four."
            if total_differences
            else "No batch-dependent prediction difference detected."
        ),
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "tensor_count": payload["tensor_count"],
                "differences": total_differences,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
