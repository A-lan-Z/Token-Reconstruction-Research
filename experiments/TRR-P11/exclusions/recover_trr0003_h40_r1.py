#!/usr/bin/env python3
"""Recover TRR-0003 inverse-train H40 identities from pinned public inputs.

The public text is read transiently only to reproduce the already committed
TRR-0001 inverse_train renderer. The output contains no text or token values.
No model, activation, truth, or evaluation artifact is loaded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time

TASK_ID = "TRR-P11"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
BOS = 128000
ROWS = 128
TOKENS = 40
MAX_RSS = 2 * 1024**3
MIN_FREE = 12 * 1024**3
ROOT = Path(__file__).resolve().parents[3]
FIT_METADATA = ROOT / "experiments/TRR-0003/evidence/control/track_b_fit_records.json"
TRR1_PLAN = ROOT / "experiments/TRR-0001/plan.json"
RESOURCE_MANIFEST = ROOT / "experiments/TRR-0003/evidence/control/public_resource_manifest.json"
OUTPUT = ROOT / "experiments/TRR-P11/exclusions/trr0003_h40_identity_rows_r1.json"
RECEIPT = ROOT / "experiments/TRR-P11/exclusions/trr0003_h40_recovery_r1.json"
EXPECTED_FIT_SHA = "7aee0f6cb452bb1df401c920ca2a628d32fb204d144d72e4419fc8bd34a3a08e"
EXPECTED_PLAN_SHA = "b498a5db5b14ae8dde19f3ae4f519f86fdf0a67572a8f78747f7921e4f9e7269"
EXPECTED_MANIFEST_SHA = "9fd0ab2a786dd7884c72dcb0d724ba10dc3912518c05cff7a69267132beefb94"


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def available() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def h40(ids: list[int]) -> str:
    if len(ids) != TOKENS:
        raise RuntimeError("H40 geometry changed")
    return hashlib.sha256(struct.pack("<" + "i" * TOKENS, *ids)).hexdigest()


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink() or RECEIPT.exists() or RECEIPT.is_symlink():
        raise RuntimeError("create-only output already exists")
    start = utc()
    t0 = time.monotonic()
    free = available()
    if free < MIN_FREE:
        raise RuntimeError(f"MemAvailable {free} below {MIN_FREE}")
    for path, expected in ((FIT_METADATA, EXPECTED_FIT_SHA), (TRR1_PLAN, EXPECTED_PLAN_SHA), (RESOURCE_MANIFEST, EXPECTED_MANIFEST_SHA)):
        if not path.is_file() or path.is_symlink() or sha_file(path) != expected:
            raise RuntimeError(f"pinned metadata changed: {path}")
    fit = json.loads(FIT_METADATA.read_text(encoding="utf-8"))
    plan = json.loads(TRR1_PLAN.read_text(encoding="utf-8"))
    manifest = json.loads(RESOURCE_MANIFEST.read_text(encoding="utf-8"))
    rows = fit.get("records")
    declared = plan["data"]["selection"]["splits"]["inverse_train"]["records"]
    if fit.get("split") != "inverse_train" or len(rows) != ROWS or len(declared) != ROWS:
        raise RuntimeError("TRR-0003/TRR-0001 inverse_train geometry changed")
    snapshot = Path(manifest["snapshot_path"])
    expected_files = {item["path"]: item["sha256"] for item in manifest["files"]}
    checked_files = {}
    for relative, expected in expected_files.items():
        path = snapshot / relative
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"pinned tokenizer/resource file changed: {path}")
        checked_files[relative] = {"path": str(path), "bytes": path.stat().st_size, "sha256": expected}

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
    if tokenizer.bos_token_id != BOS:
        raise RuntimeError("pinned tokenizer BOS changed")
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    identity = []
    mismatches = []
    for index, (actual, expected) in enumerate(zip(rows, declared)):
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            raise RuntimeError(f"row {index} malformed")
        for actual_field, expected_field in (("record_id", "record_id"), ("dataset_index", "index"), ("text_sha256", "text_sha256")):
            if actual.get(actual_field) != expected.get(expected_field):
                mismatches.append({"row": index, "field": actual_field})
        source_index = int(actual["dataset_index"])
        text = dataset[source_index]["text"]
        rendered = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if rendered != actual["text_sha256"]:
            mismatches.append({"row": index, "field": "text_sha256"})
        source_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(source_ids) < TOKENS - 1:
            mismatches.append({"row": index, "field": "eligibility"})
            continue
        ids = [BOS, *[int(value) for value in source_ids[: TOKENS - 1]]]
        identity.append({
            "record_id": actual["record_id"],
            "rendered_sha256": rendered,
            "trr0002_h40_token_ids_sha256": h40(ids),
            "dataset_id": DATASET_ID,
            "split": "train",
            "revision": DATASET_REVISION,
            "source_index": source_index,
            "full_token_count": TOKENS,
            "post_bos_token_count": TOKENS - 1,
        })
    if mismatches or len(identity) != ROWS:
        raise RuntimeError(json.dumps({"mismatch_count": len(mismatches), "rows": len(identity)}, sort_keys=True))
    identity_payload = {
        "schema": "token-reconstruction.trr-p11-public-h40-identity-rows.v1",
        "task_id": TASK_ID,
        "source_label": "trr0003_fit_records_h40_public_rerender",
        "status": "PASS_TRR0003_PUBLIC_H40_RECOVERY",
        "producer": {"fit_metadata_sha256": EXPECTED_FIT_SHA, "trr0001_plan_sha256": EXPECTED_PLAN_SHA, "dataset_id": DATASET_ID, "revision": DATASET_REVISION, "split": "inverse_train", "sequence_convention": "first 40 BOS-inclusive IDs, SHA-256 little-endian signed-int32 bytes"},
        "records": identity,
        "record_count": len(identity),
        "contains_source_text": False,
        "contains_token_ids": False,
        "contains_truth": False,
        "access_boundary": {"source_text_read_transiently": True, "source_text_serialized": False, "token_values_emitted": False, "model_loaded": False, "activations_read": False, "truth_opened": False, "gpu_used": False, "p03_holdout_accessed": False, "new_selection_started": False},
    }
    identity_bytes = canonical(identity_payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        handle.write(identity_bytes)
    end = utc()
    receipt_payload = {
        "schema": "token-reconstruction.trr-p11-trr0003-h40-recovery.v1",
        "task_id": TASK_ID,
        "status": "PASS_TRR0003_PUBLIC_H40_RECOVERY",
        "command": {"argv": sys.argv, "cwd": str(Path.cwd()), "environment": {key: __import__("os").environ.get(key) for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TOKENIZERS_PARALLELISM", "NUMEXPR_NUM_THREADS")}},
        "started_utc": start,
        "ended_utc": end,
        "elapsed_seconds": time.monotonic() - t0,
        "resource": {"threads": 1, "timeout_seconds": 120, "host_available_bytes_at_start": free, "host_minimum_bytes": MIN_FREE, "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "max_rss_limit_bytes": MAX_RSS, "model_loaded": False, "activations_read": False, "truth_opened": False, "gpu_used": False},
        "inputs": {"fit_metadata": {"path": str(FIT_METADATA), "sha256": EXPECTED_FIT_SHA, "rows": ROWS}, "trr0001_plan": {"path": str(TRR1_PLAN), "sha256": EXPECTED_PLAN_SHA, "rows": ROWS}, "resource_manifest": {"path": str(RESOURCE_MANIFEST), "sha256": EXPECTED_MANIFEST_SHA}, "tokenizer_snapshot_files": checked_files, "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "train"}},
        "coverage": {"rows_seen": ROWS, "h40_rows": ROWS, "h128_rows": 0, "h129_rows": 0, "mismatch_count": 0},
        "output": {"path": str(OUTPUT), "bytes": len(identity_bytes), "sha256": hashlib.sha256(identity_bytes).hexdigest()},
        "access_boundary": identity_payload["access_boundary"],
    }
    with RECEIPT.open("xb") as handle:
        handle.write(canonical(receipt_payload))
    print(json.dumps({"status": receipt_payload["status"], "rows": ROWS, "h40_rows": ROWS, "identity_sha256": receipt_payload["output"]["sha256"], "max_rss_bytes": receipt_payload["resource"]["max_rss_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
