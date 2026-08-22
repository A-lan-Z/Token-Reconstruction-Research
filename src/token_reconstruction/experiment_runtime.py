"""Shared deterministic runtime utilities for the TRR-0001 experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


TASK_ID = "TRR-0001"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
BOS_TOKEN_ID = 128000
SEQUENCE_TOKENS = 40
SCORED_TOKENS = 39
CUT_DEPTHS = (0, 4, 8)
CONDITIONS = ("matched_public", "unavailable_target_lora")


class ExperimentError(RuntimeError):
    """Raised when frozen experimental assumptions are violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentError(f"artifact must be a regular file: {path}")
    label = path.resolve().relative_to(root.resolve()).as_posix() if root else str(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ExperimentError(f"JSON input must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"invalid JSON: {path}") from exc


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentError(f"JSONL input must be a regular file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExperimentError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ExperimentError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory() -> dict[str, int]:
    result = {"process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    if torch.cuda.is_available():
        result.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
    return result


class PhaseTimer:
    """CUDA-synchronized wall timer with UTC boundaries."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def measure(self, name: str):
        return _MeasuredPhase(self, name)


class _MeasuredPhase:
    def __init__(self, owner: PhaseTimer, name: str) -> None:
        self.owner = owner
        self.name = name

    def __enter__(self):
        synchronize()
        self.started_utc = utc_now()
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        synchronize()
        self.owner.records.append(
            {
                "phase": self.name,
                "started_utc": self.started_utc,
                "ended_utc": utc_now(),
                "elapsed_seconds": time.perf_counter() - self.started,
                "exit_status": 0 if exc_type is None else 1,
            }
        )
        return False


def require_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "task_id": TASK_ID,
        "status": "COMMITTED_BEFORE_BLIND_TRUTH",
        "truth_opened": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ExperimentError(f"plan field {key} changed")
    model = plan.get("resources", {}).get("model", {})
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ExperimentError("pinned model identity changed")
    if tuple(plan.get("cut_depths", {}).get("evaluated", ())) != CUT_DEPTHS:
        raise ExperimentError("cut depths changed")
    if plan.get("data", {}).get("blind_scored_tokens") != 64 * SCORED_TOKENS:
        raise ExperimentError("blind token budget changed")


def load_resources():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    dataset = load_dataset(
        DATASET_ID, revision=DATASET_REVISION, split="train"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if not torch.cuda.is_available():
        raise ExperimentError("TRR-0001 execution requires the audited CUDA device")
    model = model.to(torch.device("cuda")).eval()
    if tokenizer.bos_token_id != BOS_TOKEN_ID:
        raise ExperimentError("declared BOS token changed")
    if model.config.hidden_size != 2048 or model.config.vocab_size != 128256:
        raise ExperimentError("model geometry changed")
    return tokenizer, dataset, model


def records_for_split(
    plan: Mapping[str, Any],
    split: str,
    *,
    tokenizer: Any,
    dataset: Any,
) -> list[dict[str, Any]]:
    try:
        declared = plan["data"]["selection"]["splits"][split]["records"]
    except (KeyError, TypeError) as exc:
        raise ExperimentError(f"split missing from plan: {split}") from exc
    result: list[dict[str, Any]] = []
    for item in declared:
        index = int(item["index"])
        text = dataset[index]["text"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != item["text_sha256"]:
            raise ExperimentError(f"public text hash changed for {item['record_id']}")
        source = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(source) < SCORED_TOKENS:
            raise ExperimentError(f"record became ineligible: {item['record_id']}")
        token_ids = [BOS_TOKEN_ID] + [int(value) for value in source[:SCORED_TOKENS]]
        result.append(
            {
                "record_id": item["record_id"],
                "dataset_index": index,
                "text_sha256": digest,
                "token_ids": token_ids,
            }
        )
    return result


def require_create_only_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ExperimentError(f"output must be create-only: {path}")
    path.mkdir(parents=True)


def command_record() -> dict[str, Any]:
    return {
        "argv": [str(value) for value in sys.argv],
        "cwd": os.getcwd(),
        "environment": {
            key: os.environ.get(key)
            for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH")
        },
    }
