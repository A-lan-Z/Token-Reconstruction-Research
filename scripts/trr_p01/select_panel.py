#!/usr/bin/env python3
"""Select and freeze the evaluator-private sixteen-record diagnostic panel.

The selection is public-data preparation performed before target observations.
The generated directory is private evaluator state; no file from it is copied
into an attack interface.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
import random
import sys
from typing import Any

from safetensors.torch import save_file
import torch

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from common import (  # noqa: E402
    BOS_TOKEN_ID,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    SEQUENCE_TOKENS,
    SCORED_TOKENS,
    TASK_ID,
    artifact_entry,
    command_record,
    file_record,
    load_json,
    read_jsonl,
    require_create_only_directory,
    require_create_only_file,
    sha256_file,
    utc_now,
    validate_public_plan,
    write_json_exclusive,
)


STYLE_COUNTS = {
    "prose": 4,
    "code": 4,
    "numeric_plus_punctuation": 4,
    "unicode_plus_instruction": 4,
}
STYLE_ORDER = ("code", "numeric_plus_punctuation", "unicode_plus_instruction", "prose")
DATASET_CACHE_DEFAULT = Path(
    "/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/"
    "127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"
)
PRIVATE_TRUTH_SCHEMA = "token-reconstruction.trr-p01-private-truth.v1"
PANEL_SCHEMA = "token-reconstruction.trr-p01-panel.v1"
SELECTION_SCHEMA = "token-reconstruction.trr-p01-panel-selection.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--permutation-seed", type=int, default=314159)
    return parser.parse_args()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _style(text: str) -> str | None:
    """Apply the frozen ordered style predicates used by the plan."""

    if not isinstance(text, str) or not text.strip():
        return None
    sample = text[:4000]
    lower = sample.lower()
    code_hits = sum(
        bool(re.search(pattern, sample, flags=re.MULTILINE))
        for pattern in (
            r"\b(def|class|import|from|return|for|while)\s+[A-Za-z_]",
            r"[{};]",
            r"=>|::|#include|\bSELECT\s+.+\s+FROM\b",
            r"```|</?[A-Za-z][^>]*>",
        )
    )
    if code_hits >= 2:
        return "code"

    numeric = sum(character.isdecimal() for character in sample)
    punctuation = sum(
        character in "!?,.;:/\\()[]{}<>+-=*%$#@&_\"'`"
        for character in sample
    )
    if numeric >= 3 or (punctuation >= 8 and punctuation >= len(sample) * 0.06):
        return "numeric_plus_punctuation"

    instruction = bool(
        re.search(
            r"\b(translate|summarize|explain|write|list|instruction|question|answer|task|please)\b",
            lower,
        )
    )
    non_ascii = any(ord(character) > 127 for character in sample)
    if non_ascii or instruction:
        return "unicode_plus_instruction"
    return "prose"


def _load_dataset(path: Path | None) -> Any:
    from datasets import Dataset, load_dataset

    if path is not None:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("dataset Arrow path must be a regular file")
        return Dataset.from_file(str(path))
    try:
        return load_dataset(
            DATASET_ID,
            revision=DATASET_REVISION,
            split="train",
            download_mode="reuse_dataset_if_exists",
        )
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        if DATASET_CACHE_DEFAULT.is_file():
            return Dataset.from_file(str(DATASET_CACHE_DEFAULT))
        raise RuntimeError("pinned local Pile dataset could not be loaded") from exc


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    if tokenizer.bos_token_id != BOS_TOKEN_ID:
        raise RuntimeError("tokenizer BOS identity changed")
    return tokenizer


def _select_rows(dataset: Any, tokenizer: Any, *, seed: int) -> list[dict[str, Any]]:
    if seed != 314159:
        raise RuntimeError("only the frozen panel permutation seed is accepted")
    permutation = list(range(len(dataset)))
    random.Random(seed).shuffle(permutation)
    selected: dict[str, dict[str, Any]] = {}
    for dataset_index in permutation:
        row = dataset[int(dataset_index)]
        text = row.get("text") if isinstance(row, dict) else None
        style = _style(text)
        if style is None or style not in STYLE_COUNTS or style in selected and len(selected[style]["rows"]) >= STYLE_COUNTS[style]:
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < SCORED_TOKENS:
            continue
        selected.setdefault(style, {"rows": []})["rows"].append(
            {
                "dataset_index": int(dataset_index),
                "text_sha256": _sha_text(text),
                "token_ids": [BOS_TOKEN_ID] + [int(value) for value in token_ids[:SCORED_TOKENS]],
                "source_token_count": len(token_ids),
            }
        )
        if all(len(selected.get(name, {}).get("rows", [])) == count for name, count in STYLE_COUNTS.items()):
            break
    if any(len(selected.get(name, {}).get("rows", [])) != count for name, count in STYLE_COUNTS.items()):
        counts = {name: len(selected.get(name, {}).get("rows", [])) for name in STYLE_COUNTS}
        raise RuntimeError(f"fixed panel strata could not be filled: {counts}")
    result: list[dict[str, Any]] = []
    ordinal = 1
    for style in ("prose", "code", "numeric_plus_punctuation", "unicode_plus_instruction"):
        for row in selected[style]["rows"]:
            if len(row["token_ids"]) != SEQUENCE_TOKENS:
                raise RuntimeError("selected panel geometry changed")
            result.append(
                {
                    "record_id": f"p01-r{ordinal:04d}",
                    "style": style,
                    **row,
                }
            )
            ordinal += 1
    return result


def main() -> int:
    args = parse_args()
    plan = load_json(args.plan)
    validate_public_plan(plan)
    if args.permutation_seed != int(plan["panel"]["source_selection"]["permutation_seed"]):
        raise RuntimeError("panel permutation differs from frozen plan")
    root = require_create_only_directory(args.output_root.resolve())
    started_utc = utc_now()
    dataset_path = args.dataset_path or DATASET_CACHE_DEFAULT
    dataset = _load_dataset(args.dataset_path)
    tokenizer = _load_tokenizer()
    rows = _select_rows(dataset, tokenizer, seed=args.permutation_seed)
    if [row["record_id"] for row in rows] != [f"p01-r{i:04d}" for i in range(1, 17)]:
        raise RuntimeError("opaque panel IDs are not contiguous")
    if {style: sum(row["style"] == style for row in rows) for style in STYLE_COUNTS} != STYLE_COUNTS:
        raise RuntimeError("panel style counts changed")

    truth = torch.tensor([row["token_ids"] for row in rows], dtype=torch.int64).contiguous()
    truth_path = root / "private_truth.safetensors"
    require_create_only_file(truth_path)
    save_file(
        {"input_ids": truth},
        truth_path,
        metadata={
            "schema": PRIVATE_TRUTH_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": "false",
            "source_truth_included": "true",
        },
    )
    manifest_path = root / "panel_manifest.json"
    manifest = {
        "schema": PANEL_SCHEMA,
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_truth_included": True,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "path": str(dataset_path),
            "path_sha256": sha256_file(dataset_path) if dataset_path.is_file() else None,
        },
        "tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION, "bos_token_id": BOS_TOKEN_ID},
        "selection": {
            "permutation_seed": args.permutation_seed,
            "style_order": ["prose", "code", "numeric_plus_punctuation", "unicode_plus_instruction"],
            "style_counts": STYLE_COUNTS,
            "eligibility": "at least 39 tokens without added special tokens",
            "construction": "prepend BOS 128000; retain first 39 source tokens; no padding",
        },
        "geometry": {
            "records": 16,
            "sequence_tokens": SEQUENCE_TOKENS,
            "scored_tokens": SCORED_TOKENS,
        },
        "record_order": [row["record_id"] for row in rows],
        "records": [
            {
                "record_id": row["record_id"],
                "style": row["style"],
                "dataset_index": row["dataset_index"],
                "text_sha256": row["text_sha256"],
                "source_token_count": row["source_token_count"],
            }
            for row in rows
        ],
        "private_truth": artifact_entry(truth_path, relative_to=root),
    }
    write_json_exclusive(manifest_path, manifest)
    evidence_path = root / "selection_evidence.json"
    write_json_exclusive(
        evidence_path,
        {
            "schema": SELECTION_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": False,
            "source_truth_included": True,
            "status": "PANEL_FROZEN_BEFORE_TARGET_OBSERVATIONS",
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "command": command_record(torch.device("cpu")),
            "plan": file_record(args.plan),
            "panel_manifest": file_record(manifest_path),
            "private_truth": file_record(truth_path),
            "record_order": manifest["record_order"],
            "style_counts": STYLE_COUNTS,
        },
    )
    print({"status": "PANEL_FROZEN_BEFORE_TARGET_OBSERVATIONS", "manifest": str(manifest_path), "truth": str(truth_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
