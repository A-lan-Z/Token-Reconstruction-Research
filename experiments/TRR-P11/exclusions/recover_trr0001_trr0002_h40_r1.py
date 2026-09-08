#!/usr/bin/env python3
"""Recover TRR1/TRR2 public 40-token sequence identities.

This is a bounded, hash-only recovery of already used public records.  TRR1
is reproduced with its committed Pile/tokenizer rule from the pinned public
dataset and the published plan/reveal metadata.  TRR2 Pile rows are checked
against the exact public record ledger and the same pinned producer rule, then
hashed under TRR2's distinct int64 tensor-header convention.  Token values and
source text are transient only; no model, activation, truth, or new selection
is accessed or written.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

TASK_ID = "TRR-P11"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
BOS = 128000
H40 = 40
MAX_RSS_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 12 * 1024**3

ROOT = Path(__file__).resolve().parents[3]
DATASET_ARROW = Path(
    "/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/"
    "0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"
)
DATASET_INFO = DATASET_ARROW.with_name("dataset_info.json")
TOKENIZER_SNAPSHOT = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6"
)

TRR1_PLAN = ROOT / "experiments/TRR-0001/plan.json"
TRR1_MANIFEST = ROOT / "experiments/TRR-0001/manifest.json"
TRR1_R1_PLAN = ROOT / "experiments/TRR-0001/revision-r1/plan.json"
TRR1_R1_MANIFEST = ROOT / "experiments/TRR-0001/revision-r1/manifest.json"
TRR1_R1_REVEAL = ROOT / "experiments/TRR-0001/revision-r1/selection_reveal.json"
TRR2_PILE = ROOT / "experiments/TRR-0002/configuration-search/public-pile/records.json"
TRR2_WINNER = ROOT / "experiments/TRR-0002/configuration-search/causal-selection/winner.json"
TRR2_FRESH_INDEX = ROOT / "experiments/TRR-0002/configuration-search/fresh-blind/observation-index.json"

TRR1_OUTPUT = ROOT / "experiments/TRR-P11/exclusions/trr0001_h40_identity_rows_r1.json"
TRR1_RECEIPT = ROOT / "experiments/TRR-P11/exclusions/trr0001_h40_recovery_r1.json"
TRR2_OUTPUT = ROOT / "experiments/TRR-P11/exclusions/trr0002_h40_identity_rows_r1.json"
TRR2_RECEIPT = ROOT / "experiments/TRR-P11/exclusions/trr0002_h40_recovery_r1.json"
HANDOFF = ROOT / "experiments/TRR-P11/exclusions/trr0001_trr0002_h40_recovery_handoff_r1.json"

EXPECTED_SHA = {
    TRR1_PLAN: "b498a5db5b14ae8dde19f3ae4f519f86fdf0a67572a8f78747f7921e4f9e7269",
    TRR1_MANIFEST: "80b6e7bae818729e06685b329f7270524623cb365817f8f318a3dac2bdac4dc0",
    TRR1_R1_PLAN: "59944cb1e01ec2e88e04109e46db0088eece74500fe599eb6a62ead038b6fe14",
    TRR1_R1_MANIFEST: "1abdfe97ca9066f00ae30bfe3519da612bb288bb0358a904663c0518e3808c72",
    TRR1_R1_REVEAL: "d96cae6d6c9beb29d99be34ecb597f4986f9e107212657f4ad98268676251b41",
    TRR2_PILE: "37be9bb6d621873047a05544e4e2d8ed7c07f1160ab878093ad912306ae1dd7f",
    TRR2_WINNER: "a75a5220647b0dea019cabb09be7c99a82f0d8142bf5476ffff6be5099fcb4f3",
    TRR2_FRESH_INDEX: "df7248f4d6f6566bc9d9b687c8da10a84ca08cb94bc4f23a9449b5e83ee103d2",
    DATASET_ARROW: "77ddf02e2a69373a944bc8bc8ac8f7b9926f5c62203d727341a24d709bf81113",
    DATASET_INFO: "6f76b8c5908b60192866afdcf2ff773bb877ae7c8240d9007e841633af21ba0e",
    TOKENIZER_SNAPSHOT / "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    TOKENIZER_SNAPSHOT / "tokenizer_config.json": "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
    TOKENIZER_SNAPSHOT / "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
}
EXPECTED_COUNTS = {
    "target_update_train": 64,
    "inverse_train": 128,
    "development": 32,
    "blind_evaluation": 64,
}


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def verify_inputs() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_SHA.items():
        if not path.is_file():
            raise RuntimeError(f"pinned input unavailable: {path}")
        actual = sha_file(path)
        if actual != expected:
            raise RuntimeError(f"pinned input changed: {path}: {actual} != {expected}")
        records[str(path)] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    return records


def h40_trr0001(ids: Iterable[int]) -> str:
    values = [int(value) for value in ids]
    if len(values) != H40:
        raise RuntimeError("TRR1 H40 geometry changed")
    if any(value < -(2**31) or value >= 2**31 for value in values):
        raise RuntimeError("TRR1 H40 ID is outside signed int32")
    return hashlib.sha256(struct.pack("<" + "i" * H40, *values)).hexdigest()


def h40_trr0002(ids: Iterable[int]) -> str:
    values = [int(value) for value in ids]
    if len(values) != H40:
        raise RuntimeError("TRR2 H40 geometry changed")
    if any(value < -(2**63) or value >= 2**63 for value in values):
        raise RuntimeError("TRR2 H40 ID is outside signed int64")
    header = json.dumps(
        {"dtype": "torch.int64", "shape": [H40]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = struct.pack("<" + "q" * H40, *values)
    return hashlib.sha256(header + payload).hexdigest()


def render_trr0001_row(
    *,
    row: Mapping[str, Any],
    dataset: Any,
    tokenizer: Any,
    source_group: str,
) -> dict[str, Any]:
    index = int(row["index"])
    record_id = str(row["record_id"])
    declared_text_sha = str(row["text_sha256"])
    text = str(dataset[index]["text"])
    actual_text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual_text_sha != declared_text_sha:
        raise RuntimeError(f"TRR1 text hash changed: {source_group}/{record_id}")
    source_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(source_ids) < H40 - 1:
        raise RuntimeError(f"TRR1 record became shorter than 39 source tokens: {record_id}")
    ids = [BOS, *[int(value) for value in source_ids[: H40 - 1]]]
    if ids[0] != BOS:
        raise RuntimeError(f"TRR1 BOS changed: {record_id}")
    return {
        "source_group": source_group,
        "record_id": record_id,
        "rendered_sha256": actual_text_sha,
        "h40_sequence_sha256": h40_trr0001(ids),
        "dataset_id": DATASET_ID,
        "split": "train",
        "revision": DATASET_REVISION,
        "source_index": index,
        "full_token_count": H40,
        "post_bos_token_count": H40 - 1,
    }


def recover_trr0001(dataset: Any, tokenizer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_json(TRR1_PLAN)
    selection = plan["data"]["selection"]["splits"]
    rows: list[dict[str, Any]] = []
    by_index: dict[int, str] = {}
    group_counts: dict[str, int] = {}
    for group, expected_count in EXPECTED_COUNTS.items():
        declared = selection[group]["records"]
        if not isinstance(declared, list) or len(declared) != expected_count:
            raise RuntimeError(f"TRR1 plan geometry changed: {group}")
        group_counts[group] = len(declared)
        for row in declared:
            index = int(row["index"])
            if index in by_index:
                raise RuntimeError(f"TRR1 plan index reused: {index}")
            by_index[index] = group
            rows.append(render_trr0001_row(row=row, dataset=dataset, tokenizer=tokenizer, source_group=group))
    reveal = load_json(TRR1_R1_REVEAL)
    reveal_rows = reveal.get("records")
    if not isinstance(reveal_rows, list) or len(reveal_rows) != 64:
        raise RuntimeError("TRR1 R1 reveal geometry changed")
    fresh_rows: list[dict[str, Any]] = []
    fresh_indices: set[int] = set()
    for row in reveal_rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("TRR1 R1 reveal row malformed")
        index = int(row["dataset_index"])
        if index in by_index or index in fresh_indices:
            raise RuntimeError(f"TRR1 R1 fresh row overlaps retained TRR1 source: {index}")
        fresh_indices.add(index)
        fresh_rows.append(
            render_trr0001_row(
                row={"index": index, "record_id": row["record_id"], "text_sha256": row["text_sha256"]},
                dataset=dataset,
                tokenizer=tokenizer,
                source_group="revision_r1_blind_evaluation",
            )
        )
    payload = {
        "schema": "token-reconstruction.trr-p11-trr0001-h40-identity-rows.v1",
        "task_id": TASK_ID,
        "source_label": "trr0001_plan_and_revision_r1_reveal_public_rerender",
        "status": "PASS_TRR0001_PUBLIC_H40_RECOVERY",
        "producer": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "train",
            "bos_token_id": BOS,
            "source_tokenization": "pinned tokenizer, add_special_tokens=False",
            "sequence_convention": "first 40 BOS-inclusive IDs, SHA-256 of little-endian signed-int32 bytes",
            "source_text_transient_only": True,
        },
        "groups": group_counts | {"revision_r1_blind_evaluation": len(fresh_rows)},
        "records": rows + fresh_rows,
        "record_count": len(rows) + len(fresh_rows),
        "contains_source_text": False,
        "contains_token_ids": False,
        "contains_truth": False,
        "access_boundary": {
            "source_text_read_transiently": True,
            "source_text_serialized": False,
            "source_tokens_serialized": False,
            "token_values_emitted": False,
            "model_loaded": False,
            "activations_read": False,
            "truth_opened": False,
            "gpu_used": False,
            "p03_holdout_accessed": False,
            "new_selection_started": False,
        },
    }
    return payload, {
        "rows_by_group": group_counts | {"revision_r1_blind_evaluation": len(fresh_rows)},
        "plan_indices": len(rows),
        "fresh_indices": len(fresh_rows),
        "mismatch_count": 0,
    }


def recover_trr0002(dataset: Any, tokenizer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = load_json(TRR2_PILE)
    rows: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    for group, expected_count in (("development", 32), ("update_train", 64)):
        declared = ledger.get(group)
        if not isinstance(declared, list) or len(declared) != expected_count:
            raise RuntimeError(f"TRR2 Pile geometry changed: {group}")
        group_counts[group] = len(declared)
        for row_number, row in enumerate(declared):
            if not isinstance(row, Mapping):
                raise RuntimeError(f"TRR2 Pile row malformed: {group}/{row_number}")
            tokens = row.get("token_ids")
            if (
                not isinstance(tokens, list)
                or len(tokens) != H40
                or any(not isinstance(value, int) or isinstance(value, bool) for value in tokens)
                or int(tokens[0]) != BOS
            ):
                raise RuntimeError(f"TRR2 Pile H40 payload malformed: {group}/{row_number}")
            index = int(row["dataset_index"])
            text = str(dataset[index]["text"])
            text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_sha != row["text_sha256"]:
                raise RuntimeError(f"TRR2 Pile text hash changed: {group}/{row_number}")
            source_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            expected_tokens = [BOS, *[int(value) for value in source_ids[: H40 - 1]]]
            if expected_tokens != [int(value) for value in tokens]:
                raise RuntimeError(f"TRR2 Pile producer token IDs changed: {group}/{row_number}")
            rows.append(
                {
                    "source_group": group,
                    "record_id": str(row["record_id"]),
                    "rendered_sha256": str(row["text_sha256"]),
                    "trr0002_h40_token_ids_sha256": h40_trr0002(tokens),
                    "dataset_id": DATASET_ID,
                    "split": "train",
                    "revision": DATASET_REVISION,
                    "source_index": index,
                    "full_token_count": H40,
                    "post_bos_token_count": H40 - 1,
                }
            )
    payload = {
        "schema": "token-reconstruction.trr-p11-trr0002-h40-identity-rows.v1",
        "task_id": TASK_ID,
        "source_label": "trr0002_public_pile_records_verified_against_pinned_producer",
        "status": "PASS_TRR0002_PUBLIC_PILE_H40_RECOVERY",
        "producer": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "train",
            "bos_token_id": BOS,
            "source_tokenization": "pinned tokenizer, add_special_tokens=False",
            "sequence_convention": "first 40 BOS-inclusive IDs; SHA-256 of canonical JSON tensor header {dtype: torch.int64, shape: [40]} followed by little-endian signed-int64 bytes",
            "source_text_transient_only": True,
        },
        "groups": group_counts,
        "records": rows,
        "record_count": len(rows),
        "contains_source_text": False,
        "contains_token_ids": False,
        "contains_truth": False,
        "access_boundary": {
            "source_text_read_transiently": True,
            "source_text_serialized": False,
            "source_tokens_serialized": False,
            "token_values_emitted": False,
            "model_loaded": False,
            "activations_read": False,
            "truth_opened": False,
            "gpu_used": False,
            "p03_holdout_accessed": False,
            "new_selection_started": False,
        },
    }
    return payload, {
        "rows_by_group": group_counts,
        "producer_token_matches": len(rows),
        "mismatch_count": 0,
    }


def write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical(value)
    with path.open("xb") as handle:
        handle.write(encoded)
    return {"path": str(path), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def main() -> int:
    outputs = (TRR1_OUTPUT, TRR1_RECEIPT, TRR2_OUTPUT, TRR2_RECEIPT, HANDOFF)
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise RuntimeError("H40 recovery outputs are create-only")
    started = utc()
    t0 = time.monotonic()
    available = available_bytes()
    if available < MIN_FREE_BYTES:
        raise RuntimeError(f"MemAvailable {available} below {MIN_FREE_BYTES}")
    input_records = verify_inputs()
    from datasets import Dataset
    from transformers import AutoTokenizer

    dataset = Dataset.from_file(str(DATASET_ARROW))
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_SNAPSHOT), local_files_only=True, use_fast=True)
    if tokenizer.bos_token_id != BOS:
        raise RuntimeError("pinned tokenizer BOS changed")
    trr1_payload, trr1_coverage = recover_trr0001(dataset, tokenizer)
    trr2_payload, trr2_coverage = recover_trr0002(dataset, tokenizer)
    trr1_record = write_exclusive(TRR1_OUTPUT, trr1_payload)
    trr2_record = write_exclusive(TRR2_OUTPUT, trr2_payload)
    common_boundary = trr1_payload["access_boundary"]
    ended = utc()
    resource_payload = {
        "threads": 1,
        "timeout_seconds": 120,
        "host_available_bytes_at_start": available,
        "host_minimum_bytes": MIN_FREE_BYTES,
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "max_rss_limit_bytes": MAX_RSS_BYTES,
        "model_loaded": False,
        "activations_read": False,
        "truth_opened": False,
        "gpu_used": False,
    }
    common = {
        "task_id": TASK_ID,
        "started_utc": started,
        "ended_utc": ended,
        "elapsed_seconds": time.monotonic() - t0,
        "command": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "git_head": git_head(),
            "environment": {
                key: os.environ.get(key)
                for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS")
            },
        },
        "resource": resource_payload,
        "inputs": input_records,
        "access_boundary": common_boundary,
    }
    trr1_receipt = {
        "schema": "token-reconstruction.trr-p11-trr0001-h40-recovery.v1",
        **common,
        "status": trr1_payload["status"],
        "source_label": trr1_payload["source_label"],
        "coverage": trr1_coverage,
        "output": trr1_record,
    }
    trr2_receipt = {
        "schema": "token-reconstruction.trr-p11-trr0002-h40-recovery.v1",
        **common,
        "status": trr2_payload["status"],
        "source_label": trr2_payload["source_label"],
        "coverage": trr2_coverage,
        "output": trr2_record,
    }
    trr1_receipt_record = write_exclusive(TRR1_RECEIPT, trr1_receipt)
    trr2_receipt_record = write_exclusive(TRR2_RECEIPT, trr2_receipt)
    handoff = {
        "schema": "token-reconstruction.trr-p11-h40-recovery-handoff.v1",
        "task_id": TASK_ID,
        "status": "PASS_TRR0001_AND_TRR0002_PUBLIC_H40_RECOVERY",
        "records": {
            "trr0001_plan_and_revision_r1": trr1_record,
            "trr0002_public_pile": trr2_record,
        },
        "receipts": {
            "trr0001": trr1_receipt_record,
            "trr0002": trr2_receipt_record,
        },
        "provenance_bindings": {
            "trr0001_plan": input_records[str(TRR1_PLAN)],
            "trr0001_manifest": input_records[str(TRR1_MANIFEST)],
            "trr0001_revision_r1_plan": input_records[str(TRR1_R1_PLAN)],
            "trr0001_revision_r1_manifest": input_records[str(TRR1_R1_MANIFEST)],
            "trr0001_revision_r1_reveal": input_records[str(TRR1_R1_REVEAL)],
            "trr0002_configuration_winner": input_records[str(TRR2_WINNER)],
            "trr0002_fresh_observation_index": input_records[str(TRR2_FRESH_INDEX)],
            "trr0002_public_pile_records": input_records[str(TRR2_PILE)],
        },
        "access_boundary": common_boundary,
        "integration_note": "Exports are opaque row-level identity hashes only. TRR1 uses raw signed-int32 H40; TRR2 Pile uses the distinct torch.int64 tensor-header plus signed-int64 H40 convention. No audit integration or shared exclusion code was modified.",
    }
    handoff_record = write_exclusive(HANDOFF, handoff)
    print(json.dumps({
        "status": handoff["status"],
        "trr0001_rows": trr1_payload["record_count"],
        "trr0002_rows": trr2_payload["record_count"],
        "trr0001_sha256": trr1_record["sha256"],
        "trr0002_sha256": trr2_record["sha256"],
        "handoff_sha256": handoff_record["sha256"],
        "max_rss_bytes": resource_payload["max_rss_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
