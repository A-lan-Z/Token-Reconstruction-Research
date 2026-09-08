#!/usr/bin/env python3
"""Bounded public token/mask identity recovery for an existing fit payload.

Only ``token_ids`` and ``attention_mask`` are read from the safetensors file.
The output is an opaque identity export; token values, source text, activations,
labels, and truth are never serialized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import time
from typing import Any

from safetensors import safe_open

BOS = 128000
TASK_ID = "TRR-P11"
MAX_RSS_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 12 * 1024**3


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack("<" + "i" * len(ids), *ids)).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise RuntimeError("tensor row is not list-like")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--expected-metadata-sha256", required=True)
    args = parser.parse_args()
    started = utc_now()
    t0 = time.monotonic()
    payload = args.payload.resolve()
    metadata_path = args.metadata.resolve()
    identity_output = args.identity_output.resolve()
    receipt_output = args.receipt_output.resolve()
    for path in (identity_output, receipt_output):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to overwrite create-only output: {path}")
    available = mem_available()
    if available < MIN_FREE_BYTES:
        raise RuntimeError(f"host MemAvailable {available} is below {MIN_FREE_BYTES}")
    if not payload.is_file() or payload.is_symlink() or not metadata_path.is_file() or metadata_path.is_symlink():
        raise RuntimeError("public payload or metadata is unavailable")
    payload_sha = sha_file(payload)
    metadata_sha = sha_file(metadata_path)
    if payload_sha != args.expected_payload_sha256 or metadata_sha != args.expected_metadata_sha256:
        raise RuntimeError("public payload or metadata SHA-256 changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = metadata.get("records") if isinstance(metadata, dict) else metadata
    if not isinstance(rows, list):
        raise RuntimeError("metadata has no records list")
    identity_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    short_rows = 0
    h128_rows = 0
    h129_rows = 0
    with safe_open(str(payload), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if not {"token_ids", "attention_mask"}.issubset(keys):
            raise RuntimeError("payload lacks token_ids/attention_mask")
        token_tensor = handle.get_tensor("token_ids")
        mask_tensor = handle.get_tensor("attention_mask")
        if tuple(token_tensor.shape) != tuple(mask_tensor.shape) or len(tuple(token_tensor.shape)) != 2:
            raise RuntimeError("token/mask tensor geometry changed")
        token_rows = token_tensor.tolist()
        mask_rows = mask_tensor.tolist()
    if len(rows) != len(token_rows):
        raise RuntimeError("metadata and token payload row counts differ")
    for index, (metadata_row, raw_ids, raw_mask) in enumerate(zip(rows, token_rows, mask_rows)):
        if not isinstance(metadata_row, dict) or not isinstance(metadata_row.get("record_id"), str) or not metadata_row["record_id"]:
            raise RuntimeError(f"metadata row {index} has no record_id")
        ids = [int(value) for value in as_list(raw_ids)]
        mask = [int(value) for value in as_list(raw_mask)]
        if len(ids) != len(mask) or not ids or any(value not in (0, 1) for value in mask):
            raise RuntimeError(f"row {index} mask geometry is malformed")
        active = sum(mask)
        if any(mask[pos] == 0 and any(mask[pos + 1 :]) for pos in range(len(mask))):
            raise RuntimeError(f"row {index} mask is not a prefix")
        if ids[0] != BOS:
            raise RuntimeError(f"row {index} BOS identity changed")
        active_ids = ids[:active]
        if len(active_ids) != active:
            raise RuntimeError(f"row {index} active token geometry changed")
        expected_full = metadata_row.get("full_token_count")
        if isinstance(expected_full, int) and expected_full != active:
            mismatches.append({"row_index": index, "field": "full_token_count", "expected": expected_full, "actual": active})
        expected_post = metadata_row.get("post_bos_token_count")
        if isinstance(expected_post, int) and expected_post != active - 1:
            mismatches.append({"row_index": index, "field": "post_bos_token_count", "expected": expected_post, "actual": active - 1})
        h128 = sha_ids(active_ids[:128]) if active >= 128 else None
        h129 = sha_ids(active_ids[:129]) if active >= 129 else None
        if h128 is None:
            short_rows += 1
        else:
            h128_rows += 1
        if h129 is not None:
            h129_rows += 1
        identity_rows.append({
            "record_id": metadata_row["record_id"],
            "rendered_sha256": metadata_row.get("rendered_sha256"),
            "h128_sequence_sha256": h128,
            "h129_sequence_sha256": h129,
            "dataset_key": metadata_row.get("dataset_key") or metadata_row.get("domain"),
            "dataset_id": metadata_row.get("dataset_id"),
            "split": metadata_row.get("split") or "train",
            "revision": metadata_row.get("revision"),
            "source_index": metadata_row.get("source_row_index") if isinstance(metadata_row.get("source_row_index"), int) else metadata_row.get("row_index"),
            "full_token_count": active,
            "post_bos_token_count": active - 1,
        })
    status = "PASS_PUBLIC_TOKEN_IDENTITY_RECOVERY" if not mismatches else "FAIL_PUBLIC_TOKEN_IDENTITY_RECOVERY"
    ended = utc_now()
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    receipt = {
        "schema": "token-reconstruction.trr-p11-public-token-identity-recovery.v1",
        "task_id": TASK_ID,
        "source_label": args.source_label,
        "status": status,
        "started_utc": started,
        "ended_utc": ended,
        "elapsed_seconds": time.monotonic() - t0,
        "resource": {
            "threads": 1,
            "timeout_seconds": 240,
            "host_available_bytes_at_start": available,
            "host_minimum_bytes": MIN_FREE_BYTES,
            "max_rss_bytes": max_rss,
            "max_rss_limit_bytes": MAX_RSS_BYTES,
            "model_loaded": False,
            "gpu_used": False,
        },
        "payload": {"path": str(payload), "bytes": payload.stat().st_size, "sha256": payload_sha, "tensors_read": ["token_ids", "attention_mask"], "tensors_not_read": ["activations", "position_ids", "post_bos_selector_large", "post_bos_selector_small"]},
        "metadata": {"path": str(metadata_path), "bytes": metadata_path.stat().st_size, "sha256": metadata_sha, "records": len(rows)},
        "coverage": {"rows_seen": len(rows), "h128_rows": h128_rows, "h129_rows": h129_rows, "short_rows_h128_inapplicable": short_rows},
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "access_boundary": {"source_text_read": False, "source_text_serialized": False, "source_tokens_serialized": False, "token_values_emitted": False, "activations_read": False, "model_loaded": False, "gpu_used": False, "evaluation_truth_opened": False, "p03_holdout_accessed": False, "new_selection_started": False},
    }
    if mismatches:
        raise RuntimeError(json.dumps(receipt, sort_keys=True))
    identity = {
        "schema": "token-reconstruction.trr-p11-public-token-identity-rows.v1",
        "task_id": TASK_ID,
        "source_label": args.source_label,
        "status": status,
        "payload_sha256": payload_sha,
        "metadata_sha256": metadata_sha,
        "records": identity_rows,
        "record_count": len(identity_rows),
        "contains_source_text": False,
        "contains_token_ids": False,
        "contains_truth": False,
        "access_boundary": receipt["access_boundary"],
    }
    identity_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    identity_output.write_bytes(canonical(identity))
    receipt["identity_export"] = {"path": str(identity_output), "bytes": identity_output.stat().st_size, "sha256": sha_file(identity_output)}
    receipt_output.write_bytes(canonical(receipt))
    print(json.dumps({"status": status, "rows": len(rows), "h128_rows": h128_rows, "short_rows": short_rows, "identity_output": str(identity_output), "receipt_output": str(receipt_output), "max_rss_bytes": max_rss}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
