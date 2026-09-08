#!/usr/bin/env python3
"""Recover opaque identities for the predeclared P04 no_robots target rows.

This follows the retained P04 evaluator-target producer exactly for the public
row order and rendering recipe.  The pinned public Arrow rows and tokenizer are
read transiently; only rendered/hash/geometry metadata is written.  No model,
activation, label, evaluator truth, or P03 holdout is opened.
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
import subprocess
import time
from typing import Any, Mapping

from datasets import Dataset
from transformers import AutoTokenizer

TASK_ID = "TRR-P11"
ROOT = Path(__file__).resolve().parents[2]
BOS = 128000
DATASET_ID = "HuggingFaceH4/no_robots"
DATASET_REVISION = "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
ARROW = Path("/home/alanz/.cache/huggingface/datasets/HuggingFaceH4___no_robots/default/0.0.0/e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b/no_robots-train.arrow")
ARROW_BYTES = 16_503_208
ARROW_SHA256 = "5a9193e927d899d167fd40553d0b403499f5f9cf9a9254db19399a4d0b3550fb"
DATASET_INFO = ARROW.parent / "dataset_info.json"
DATASET_INFO_SHA256 = "5b18eaa76054da66839dd9c4bab7069504ea8072f917948d9adc1e00aec84b7d"
TOKENIZER = Path("/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6")
TOKENIZER_FILES = {
    "tokenizer.json": ("79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4", 9_085_657),
    "tokenizer_config.json": ("9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e", 54_528),
    "special_tokens_map.json": ("6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec", 296),
}
PLAN_COMMIT = "1aefc307ebdd4cd5002ac6ac0cdc5a1fc696aa68"
PLAN_PATH = "experiments/TRR-P04/setup/evaluator_target_plan.json"
PLAN_SHA256 = "55f5cc5ecc90599d8983ea1fa23d81a5a062178fd5dde6c9d19359c9fcc54fc2"
PRODUCER_COMMIT = "63d4016f2e555e8460ea566cf967237dc94fb0c3"
PRODUCER_SHA256 = "bbf4fe9ac443f49f11c12f4a2fd3a5e5eef2cd67bf46573384eb13ffb2cbbd04"
SEED = 20260910
TARGET_ROWS = 256
TARGET_ORDER_SHA256 = "42fb5bb7dfc58dba8ccf9e3b288e787fd88cc1788e936ee934cfbbdc86de2fd2"
DATE_STRING = "06 Aug 2026"
MAX_RSS_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 12 * 1024**3


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_ids(ids: list[int]) -> str:
    return sha_bytes(struct.pack("<" + "i" * len(ids), *ids))


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_object(commit: str, relative_path: str) -> bytes:
    try:
        completed = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relative_path}"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"retained Git object unavailable: {commit}:{relative_path}") from exc
    return completed.stdout


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def target_indices(size: int) -> list[int]:
    if size != 9500:
        raise RuntimeError(f"no_robots source row count changed: {size}")
    return sorted(range(size), key=lambda index: (sha_bytes(f"TRR-P04|target-update-v1|row:{index}|seed:{SEED}".encode("utf-8")), index))[:TARGET_ROWS]


def digest_indices(indices: list[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(str(index).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def messages(row: Mapping[str, Any], index: int) -> list[dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"no_robots row {index} has no messages list")
    result: list[dict[str, str]] = []
    for message_index, item in enumerate(raw):
        if not isinstance(item, Mapping) or not isinstance(item.get("role"), str) or not isinstance(item.get("content"), str) or not item["role"] or not item["content"]:
            raise RuntimeError(f"no_robots row {index} message {message_index} is malformed")
        result.append({"role": item["role"], "content": item["content"]})
    return result


def token_list(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value:
        raise RuntimeError("tokenizer returned no token IDs")
    return [int(token) for token in value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    identity_output = args.identity_output.resolve()
    receipt_output = args.receipt_output.resolve()
    if identity_output.exists() or identity_output.is_symlink() or receipt_output.exists() or receipt_output.is_symlink():
        raise RuntimeError("refusing to overwrite create-only output")
    started = utc_now()
    t0 = time.monotonic()
    available = mem_available()
    if available < MIN_FREE_BYTES:
        raise RuntimeError(f"host MemAvailable {available} is below {MIN_FREE_BYTES}")
    if not ARROW.is_file() or ARROW.is_symlink() or ARROW.stat().st_size != ARROW_BYTES or sha_file(ARROW) != ARROW_SHA256:
        raise RuntimeError("pinned no_robots Arrow changed")
    if not DATASET_INFO.is_file() or DATASET_INFO.is_symlink() or sha_file(DATASET_INFO) != DATASET_INFO_SHA256:
        raise RuntimeError("pinned no_robots dataset info changed")
    plan_bytes = git_object(PLAN_COMMIT, PLAN_PATH)
    if sha_bytes(plan_bytes) != PLAN_SHA256:
        raise RuntimeError("retained P04 target plan bytes changed")
    producer_bytes = git_object(PRODUCER_COMMIT, "scripts/trr_p04/prepare_evaluator_target.py")
    if sha_bytes(producer_bytes) != PRODUCER_SHA256:
        raise RuntimeError("retained P04 target producer bytes changed")
    asset_checks: dict[str, Any] = {"arrow": {"path": str(ARROW), "bytes": ARROW_BYTES, "sha256": ARROW_SHA256}, "dataset_info": {"path": str(DATASET_INFO), "bytes": DATASET_INFO.stat().st_size, "sha256": DATASET_INFO_SHA256}, "tokenizer": {}}
    for name, (expected_sha, expected_bytes) in TOKENIZER_FILES.items():
        path = TOKENIZER / name
        if not path.is_file() or path.stat().st_size != expected_bytes or sha_file(path) != expected_sha:
            raise RuntimeError(f"pinned tokenizer file changed: {path}")
        asset_checks["tokenizer"][name] = {"path": str(path), "bytes": expected_bytes, "sha256": expected_sha}
    dataset = Dataset.from_file(str(ARROW))
    indices = target_indices(len(dataset))
    if digest_indices(indices) != TARGET_ORDER_SHA256:
        raise RuntimeError("target row-order commitment changed")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    if int(tokenizer.bos_token_id) != BOS:
        raise RuntimeError("tokenizer BOS changed")
    identity_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    h128_rows = 0
    h129_rows = 0
    short_rows = 0
    for index in indices:
        row = dataset[int(index)]
        msgs = messages(row, int(index))
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, date_string=DATE_STRING)
        ids = token_list(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, date_string=DATE_STRING))
        if ids[0] != BOS or len(ids) < 2:
            raise RuntimeError(f"target row {index} lost BOS or is too short")
        rendered_sha = sha_bytes(rendered.encode("utf-8"))
        h128 = sha_ids(ids[:128]) if len(ids) >= 128 else None
        h129 = sha_ids(ids[:129]) if len(ids) >= 129 else None
        if h128 is None:
            short_rows += 1
        else:
            h128_rows += 1
        if h129 is not None:
            h129_rows += 1
        identity_rows.append({
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "split": "train",
            "row_index": int(index),
            "record_id": f"no-robots-train-{int(index):05d}-{rendered_sha[:16]}",
            "rendered_sha256": rendered_sha,
            "h128_sequence_sha256": h128,
            "h129_sequence_sha256": h129,
            "rendered_char_count": len(rendered),
            "full_token_count": len(ids),
            "post_bos_token_count": len(ids) - 1,
        })
    status = "PASS_P04_TARGETFIT_EXACT_RENDERED_H129_H128_RECOVERY" if not mismatches else "FAIL_P04_TARGETFIT_EXACT_RECOVERY"
    ended = utc_now()
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    receipt = {
        "schema": "token-reconstruction.trr-p11-p04-targetfit-identity-recovery.v1",
        "task_id": TASK_ID,
        "status": status,
        "started_utc": started,
        "ended_utc": ended,
        "elapsed_seconds": time.monotonic() - t0,
        "resource": {"threads": 1, "timeout_seconds": 240, "host_available_bytes_at_start": available, "host_minimum_bytes": MIN_FREE_BYTES, "max_rss_bytes": max_rss, "max_rss_limit_bytes": MAX_RSS_BYTES, "model_loaded": False, "gpu_used": False},
        "producer": {"commit": PRODUCER_COMMIT, "sha256": PRODUCER_SHA256, "plan_commit": PLAN_COMMIT, "plan_path": PLAN_PATH, "plan_sha256": PLAN_SHA256, "date_string": DATE_STRING, "git_objects_verified": True},
        "dataset": {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION, "split": "train", "source_rows": len(dataset), "selected_rows": len(indices), "selected_row_order_sha256": TARGET_ORDER_SHA256},
        "assets": asset_checks,
        "coverage": {"selected_rows": len(identity_rows), "h128_rows": h128_rows, "h129_rows": h129_rows, "short_rows_h128_inapplicable": short_rows},
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "access_boundary": {"source_text_materialized_transiently": True, "source_text_serialized": False, "source_tokens_materialized_transiently": True, "source_tokens_serialized": False, "token_values_emitted": False, "labels_read": False, "model_loaded": False, "gpu_used": False, "evaluation_truth_opened": False, "target_update_opened": False, "p03_holdout_accessed": False, "new_selection_started": False},
    }
    if mismatches:
        raise RuntimeError(json.dumps(receipt, sort_keys=True))
    identity = {
        "schema": "token-reconstruction.trr-p11-p04-targetfit-identity-rows.v1",
        "task_id": TASK_ID,
        "status": status,
        "producer_commit": PRODUCER_COMMIT,
        "producer_sha256": PRODUCER_SHA256,
        "plan_commit": PLAN_COMMIT,
        "plan_sha256": PLAN_SHA256,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "selected_row_order_sha256": TARGET_ORDER_SHA256,
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
    print(json.dumps({"status": status, "selected_rows": len(identity_rows), "h128_rows": h128_rows, "short_rows": short_rows, "identity_output": str(identity_output), "receipt_output": str(receipt_output), "max_rss_bytes": max_rss}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
