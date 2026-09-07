#!/usr/bin/env python3
"""Bounded synthetic I/O qualification for one P09 64-row shard.

This benchmark never opens a public source, model, checkpoint, evaluation
truth, or P09 capture artifact.  It writes one deterministic synthetic
safetensors payload, measures serialization/hash/header/full-read I/O, records
a conservative 169-shard extrapolation, and removes the synthetic payload on
success.  A failed run leaves its scratch directory as evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TASK_ID = "TRR-P09"
SEED = 20260907
RECORDS = 64
SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
SHARDS = 169
RETAINED_CAP_GIB = 20.0
CONSERVATIVE_FACTOR = 2.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filesystem_free(path: Path) -> int:
    return int(shutil.disk_usage(Path(path).resolve()).free)


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"


def build_tensors() -> dict[str, torch.Tensor]:
    # Avoid random state and large temporary arange tensors.  The first H
    # channel carries a deterministic row/position marker; all remaining H
    # values are zero, which is sufficient for serialization and read I/O.
    activations = torch.zeros((RECORDS, SEQUENCE_TOKENS, HIDDEN_SIZE), dtype=torch.bfloat16)
    row_markers = torch.arange(RECORDS, dtype=torch.bfloat16).view(RECORDS, 1)
    position_markers = torch.arange(SEQUENCE_TOKENS, dtype=torch.bfloat16).view(1, SEQUENCE_TOKENS)
    activations[:, :, 0] = row_markers + position_markers
    attention_mask = torch.ones((RECORDS, SEQUENCE_TOKENS), dtype=torch.uint8)
    # Keep a small amount of inactive padding to exercise the P09 convention.
    lengths = torch.tensor([(SEQUENCE_TOKENS - (index % 7)) for index in range(RECORDS)], dtype=torch.long)
    positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.int64).expand(RECORDS, -1).clone()
    for index, length in enumerate(lengths.tolist()):
        attention_mask[index, length:] = 0
        positions[index, length:] = 0
    token_ids = torch.zeros((RECORDS, SEQUENCE_TOKENS), dtype=torch.int32)
    token_ids[:, 0] = 128000
    token_ids[:, 1:] = torch.arange(1, SEQUENCE_TOKENS, dtype=torch.int32).view(1, -1)
    return {
        "activations": activations,
        "attention_mask": attention_mask,
        "position_ids": positions,
        "token_ids": token_ids,
    }


def run(root: Path, scratch_root: Path, receipt_path: Path) -> dict[str, object]:
    root = Path(root).resolve()
    scratch_root = Path(scratch_root).resolve()
    shard_root = scratch_root / "synthetic-io-qualification-r1"
    payload_path = shard_root / "shard-000000" / "payload.safetensors"
    command = [
        sys.executable,
        "scripts/trr_p09/synthetic_io_qualification.py",
        "--scratch-root",
        str(scratch_root),
        "--receipt",
        str(receipt_path),
    ]
    started = utc_now()
    free_before = filesystem_free(scratch_root.parent)
    result: dict[str, object] = {
        "schema": "token-reconstruction.trr-p09-synthetic-io-qualification.v1",
        "task_id": TASK_ID,
        "status": "FAILED",
        "started_utc": started,
        "finished_utc": None,
        "seed": SEED,
        "command": command,
        "cwd": str(root),
        "source_commit": git_value(root, "rev-parse", "HEAD"),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "geometry": {
            "records": RECORDS,
            "sequence_tokens": SEQUENCE_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "hidden_dtype": "torch.bfloat16",
            "payload_keys": ["activations", "attention_mask", "position_ids", "token_ids"],
        },
        "filesystem": {"scratch_parent": str(scratch_root.parent), "free_before_bytes": free_before},
        "observations": {},
        "extrapolation": {},
        "truth_boundary": {
            "synthetic_only": True,
            "model_loaded": False,
            "public_source_opened": False,
            "public_forward_started": False,
            "activation_bank_opened": False,
            "evaluation_truth_opened": False,
        },
        "cleanup": {"payload_removed": False, "failure_scratch_retained": True},
    }
    tensors = build_tensors()
    try:
        shard_root.mkdir(parents=True, exist_ok=False)
        payload_path.parent.mkdir(parents=True, exist_ok=False)
        t0 = time.perf_counter()
        save_file(tensors, str(payload_path), metadata={"schema": "synthetic-p09-shard", "seed": str(SEED)})
        serialize_seconds = time.perf_counter() - t0
        payload_bytes = int(payload_path.stat().st_size)

        t0 = time.perf_counter()
        payload_sha = sha256_file(payload_path)
        hash_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            headers = {
                key: {"shape": list(handle.get_slice(key).get_shape()), "dtype": str(handle.get_slice(key).get_dtype())}
                for key in keys
            }
        header_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        loaded: dict[str, torch.Tensor] = {}
        with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
            for key in keys:
                loaded[key] = handle.get_slice(key)[:].contiguous()
        read_seconds = time.perf_counter() - t0
        read_bytes = sum(int(value.numel() * value.element_size()) for value in loaded.values())
        if not torch.equal(loaded["attention_mask"], tensors["attention_mask"]):
            raise RuntimeError("synthetic mask readback mismatch")
        if not torch.equal(loaded["position_ids"], tensors["position_ids"]):
            raise RuntimeError("synthetic position readback mismatch")
        if not torch.equal(loaded["token_ids"], tensors["token_ids"]):
            raise RuntimeError("synthetic token readback mismatch")
        if not torch.equal(loaded["activations"], tensors["activations"]):
            raise RuntimeError("synthetic activation readback mismatch")

        shard_total = payload_bytes * SHARDS
        conservative_total = int(shard_total * CONSERVATIVE_FACTOR)
        cap_bytes = int(RETAINED_CAP_GIB * 2**30)
        result["observations"] = {
            "payload_path": str(payload_path),
            "payload_bytes": payload_bytes,
            "payload_sha256": payload_sha,
            "serialized_seconds": serialize_seconds,
            "sha256_seconds": hash_seconds,
            "header_seconds": header_seconds,
            "full_read_seconds": read_seconds,
            "read_bytes": read_bytes,
            "keys": keys,
            "headers": headers,
            "readback_equal": True,
        }
        result["extrapolation"] = {
            "shards": SHARDS,
            "raw_payload_bytes": shard_total,
            "raw_payload_gib": shard_total / 2**30,
            "raw_payload_decimal_gb": shard_total / 10**9,
            "conservative_factor": CONSERVATIVE_FACTOR,
            "conservative_retained_bytes": conservative_total,
            "conservative_retained_gib": conservative_total / 2**30,
            "retained_cap_bytes": cap_bytes,
            "retained_cap_gib": RETAINED_CAP_GIB,
            "cap_margin_gib": (cap_bytes - conservative_total) / 2**30,
            "cap_check": conservative_total < cap_bytes,
            "time_seconds_if_serial": {
                "serialize": serialize_seconds * SHARDS,
                "sha256": hash_seconds * SHARDS,
                "header": header_seconds * SHARDS,
                "full_read": read_seconds * SHARDS,
            },
            "interpretation": "synthetic serialization/hash/read timing only; excludes public-model forward and capture scheduling",
        }
        result["status"] = "PASS_SYNTHETIC_IO_ONLY"
        result["cleanup"] = {"payload_removed": True, "failure_scratch_retained": False}
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["cleanup"] = {"payload_removed": False, "failure_scratch_retained": True}
    finally:
        result["filesystem"]["free_after_bytes"] = filesystem_free(scratch_root.parent)
        result["finished_utc"] = utc_now()
        receipt_path = Path(receipt_path).resolve()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if result["status"] == "PASS_SYNTHETIC_IO_ONLY":
            shutil.rmtree(shard_root, ignore_errors=False)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp/trr-p09-runtime"))
    parser.add_argument("--receipt", type=Path, default=Path("experiments/TRR-P09/review/synthetic-io-qualification-r1.json"))
    args = parser.parse_args(argv)
    result = run(Path.cwd(), args.scratch_root, args.receipt)
    print(json.dumps({"status": result["status"], "receipt": str(args.receipt), "observations": result.get("observations", {}), "extrapolation": result.get("extrapolation", {})}, sort_keys=True))
    return 0 if result["status"] == "PASS_SYNTHETIC_IO_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())

