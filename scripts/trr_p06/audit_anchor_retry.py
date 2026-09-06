#!/usr/bin/env python3
"""Audit the completed TRR-P06 anchor retry against its preserved failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import torch
from safetensors import safe_open


SCHEMA = "token-reconstruction.trr-p06-anchor-retry-equivalence.v1"
METHOD_FILE = "frozen_a1_a2_k256.safetensors"


class AuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = raw[8 : 8 + header_length]
    data = raw[8 + header_length :]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = dict(handle.metadata() or {})
        values = handle.get_tensor("predictions").detach().cpu().contiguous()
    return {
        "path": str(path),
        "bytes": len(raw),
        "file_sha256": _sha256(path),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "header_length": int(header_length),
        "keys": keys,
        "metadata": metadata,
        "prediction_shape": list(values.shape),
        "prediction_dtype": str(values.dtype),
        "prediction_tensor_sha256": hashlib.sha256(values.numpy().tobytes(order="C")).hexdigest(),
        "prediction_tensor": values,
    }


def audit(failed_root: Path, retry_root: Path, output: Path) -> dict[str, Any]:
    failed_root = failed_root.expanduser().resolve()
    retry_root = retry_root.expanduser().resolve()
    output = output.expanduser().resolve()
    failed_file = failed_root / "predictions" / "pile" / METHOD_FILE
    retry_file = retry_root / "predictions" / "pile" / METHOD_FILE
    if not failed_file.is_file() or not retry_file.is_file():
        raise AuditError("both pile prediction files are required")
    failed = _prediction(failed_file)
    retry = _prediction(retry_file)
    for key in ("keys", "metadata", "prediction_shape", "prediction_dtype", "data_sha256", "prediction_tensor_sha256"):
        if failed[key] != retry[key]:
            raise AuditError(f"anchor retry semantic mismatch in {key}")
    failure = _json(failed_root / "failure.json")
    manifest = _json(retry_root / "anchor_predictions.json")
    result = {
        "schema": SCHEMA,
        "task_id": "TRR-P06",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": [
            "python3",
            "scripts/trr_p06/audit_anchor_retry.py",
            "--failed-root",
            str(failed_root),
            "--retry-root",
            str(retry_root),
            "--output",
            str(output),
        ],
        "failed_r1": {
            "status": failure.get("status"),
            "error_type": failure.get("error_type"),
            "error": failure.get("error"),
            "prediction": {key: value for key, value in failed.items() if key != "prediction_tensor"},
        },
        "successful_r2": {
            "status": manifest.get("status"),
            "code_commit": manifest.get("code_commit"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "prediction": {key: value for key, value in retry.items() if key != "prediction_tensor"},
        },
        "equivalence": {
            "keys_equal": True,
            "metadata_equal": True,
            "shape_equal": True,
            "dtype_equal": True,
            "prediction_tensor_equal": bool(torch.equal(failed["prediction_tensor"], retry["prediction_tensor"])),
            "prediction_tensor_sha256_equal": True,
            "data_region_sha256_equal": True,
            "container_file_sha256_equal": False,
            "difference_scope": "safetensors header JSON metadata ordering; prediction tensor bytes are identical",
            "truth_opened": False,
            "source_text_loaded": False,
            "candidate_arrays_persisted": False,
        },
    }
    if output.exists() or output.is_symlink():
        raise AuditError(f"output is create-only: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-root", type=Path, required=True)
    parser.add_argument("--retry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit(args.failed_root, args.retry_root, args.output)
    except (AuditError, OSError, KeyError, ValueError) as exc:
        print(f"TRR-P06 anchor retry audit failed: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
