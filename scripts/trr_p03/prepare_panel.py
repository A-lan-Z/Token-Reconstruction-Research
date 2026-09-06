#!/usr/bin/env python3
"""Prepare the TRR-P03 natural panel and its evaluator-only truth sidecar.

This is a source-side preparation program.  It may inspect public dataset text
and metadata before the panel is frozen, but it never loads a model or emits
source plaintext into an observation interface.  The selector is deliberately
create-only: a frozen output directory cannot be silently rewritten.

The public attack input is represented by ``interface.json`` and the
subsequent evaluator-written observation indexes.  It contains opaque record
IDs, geometry, and tensor-key conventions only.  Stage-specific JSONL truth
and source maps are evaluator-side files and must stay outside a reconstruction
process.  Stage-2 truth is prepared in a separate sealed directory but is not
opened or scored by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "TRR-P03"
PARENT_COMMIT = "7956b4357d076abce3ccfc407d3fcac832fd34f6"
BOS_TOKEN_ID = 128000
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
TARGET_ID = "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"
TARGET_REVISION = "7fa9d06a59246629244cdd3b6b92e4fc756baa0f"
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
CUT_DEPTH = 4
LENGTHS = (16, 39, 64, 128)
RECORDS_PER_LENGTH = 6
PER_STYLE_PER_LENGTH = 2
STAGE1_SEED = 20260906
STAGE2_SEED = 20260907
STYLE_ORDER = ("coding", "question_answer", "creative_generation")
STYLE_CATEGORIES = {
    "coding": ("Coding",),
    "question_answer": ("Open QA", "Closed QA", "Classify", "Extract"),
    "creative_generation": (
        "Generation",
        "Brainstorm",
        "Chat",
        "Rewrite",
        "Summarize",
    ),
}

DATASET_ID = "HuggingFaceH4/no_robots"
DATASET_REVISION = "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
DATASET_ARROW_DEFAULT = Path(
    "/home/alanz/.cache/huggingface/datasets/HuggingFaceH4___no_robots/default/0.0.0/"
    "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b/no_robots-train.arrow"
)
BASE_SNAPSHOT_DEFAULT = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6"
)
TARGET_SNAPSHOT_DEFAULT = Path(
    "/home/alanz/.cache/huggingface/hub/models--Vikhrmodels--Vikhr-Llama-3.2-1B-Instruct/"
    "snapshots/7fa9d06a59246629244cdd3b6b92e4fc756baa0f"
)
P02_EXCLUSION_DEFAULT = Path("experiments/TRR-P02/setup/public-diagnostic-exclusion.final.json")

BOUNDARY_PROTOTYPES = Path(
    "/tmp/trr-p01/experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors"
)
BOUNDARY_PROTOTYPES_SHA256 = "51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3"
BOUNDARY_PROTOTYPES_BYTES = 525337024
HISTORICAL_LENS = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/"
    "TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt"
)
HISTORICAL_LENS_SHA256 = "33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742"
HISTORICAL_LENS_BYTES = 16787653

BASE_CONFIG_SHA256 = "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"
BASE_TOKENIZER_CONFIG_SHA256 = "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e"
BASE_WEIGHTS_SHA256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"
BASE_WEIGHTS_BYTES = 2471645608
TARGET_CONFIG_SHA256 = "7510055506497971937a3b247c853e664fdc1b1bbeece4cafc03107fa5e6fae7"
TARGET_TOKENIZER_CONFIG_SHA256 = "cb0ba7a62825d3621cc1bd87430e17744ed32dd31b1b43143e3dc7aa4ba3b9ce"
TARGET_WEIGHTS_SHA256 = "389c73748a00a8a006a4a4a26fa473319676c25672aa188f8337981cd0cc8850"
TARGET_WEIGHTS_BYTES = 2471645464

SCHEMA_PANEL = "token-reconstruction.trr-p03-natural-panel.v1"
SCHEMA_TRUTH = "token-reconstruction.trr-p03-private-truth.v1"
SCHEMA_TRUTH_INDEX = "token-reconstruction.trr-p03-private-truth-index.v1"
SCHEMA_OBSERVATIONS = "token-reconstruction.trr-p03-observation-index-template.v1"
SCHEMA_AUDIT = "token-reconstruction.trr-p03-selection-audit.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_ids(values: Iterable[int]) -> str:
    encoded = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular local file: {path}")


def require_readable_file(path: Path, label: str) -> None:
    """Require a readable file while allowing Hugging Face cache symlinks."""

    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")


def require_new_dir(path: Path) -> Path:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing frozen directory: {path}")
    path.mkdir(parents=True)
    return path


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.relative_to(root)) if root is not None else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_ARROW_DEFAULT)
    parser.add_argument("--base-snapshot", type=Path, default=BASE_SNAPSHOT_DEFAULT)
    parser.add_argument("--target-snapshot", type=Path, default=TARGET_SNAPSHOT_DEFAULT)
    parser.add_argument("--p02-exclusion", type=Path, default=P02_EXCLUSION_DEFAULT)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="TRR repository root containing the public prior records",
    )
    return parser.parse_args()


def _load_dataset(path: Path) -> Any:
    require_regular_file(path, "dataset Arrow path")
    try:
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("datasets is required for public panel preparation") from exc
    return Dataset.from_file(str(path))


def _load_tokenizer(snapshot: Path) -> Any:
    if not snapshot.is_dir():
        raise RuntimeError(f"tokenizer snapshot is missing: {snapshot}")
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("transformers is required for public panel preparation") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if int(tokenizer.bos_token_id) != BOS_TOKEN_ID:
        raise RuntimeError(
            f"pinned tokenizer BOS changed: expected {BOS_TOKEN_ID}, got {tokenizer.bos_token_id}"
        )
    return tokenizer


def _load_json(path: Path) -> dict[str, Any]:
    require_regular_file(path, "JSON input")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON input must be an object: {path}")
    return value


def _p02_tuples(
    path: Path,
) -> tuple[dict[tuple[tuple[int, ...], int, int], list[str]], dict[str, Any]]:
    data = _load_json(path)
    cases = data.get("cases")
    if data.get("case_status") != "EXECUTED_RUN4_PUBLIC_ONLY_TRUTH_FREE":
        raise RuntimeError("P02 exclusion is not the executed truth-free public artifact")
    if not isinstance(cases, list) or len(cases) != 51:
        raise RuntimeError("P02 exclusion must contain exactly 51 cases")
    tuples: dict[tuple[tuple[int, ...], int, int], list[str]] = defaultdict(list)
    for case in cases:
        if not isinstance(case, dict):
            raise RuntimeError("P02 case is not an object")
        sequence = tuple(int(value) for value in case["sequence_token_ids"])
        positions = tuple(int(value) for value in case["position_ids"])
        endpoint_ids = tuple(int(value) for value in case["endpoint_token_ids"])
        if sequence[0] != BOS_TOKEN_ID:
            raise RuntimeError("P02 tuple does not begin with the pinned BOS")
        if positions != tuple(range(len(sequence))):
            raise RuntimeError("P02 position IDs are not contiguous from BOS")
        if not endpoint_ids or endpoint_ids[-1] != sequence[-1]:
            raise RuntimeError("P02 endpoint ID disagrees with sequence tuple")
        endpoint_position = positions[-1]
        key = (sequence, endpoint_position, endpoint_ids[-1])
        tuples[key].append(str(case["case_id"]))
    return dict(tuples), {
        "path": str(path),
        "sha256": sha256_file(path),
        "case_count": len(cases),
        "unique_sequence_count": len({key[0] for key in tuples}),
        "position_endpoint_key_count": len(tuples),
        "global_token_type_exclusion": False,
        "token_ids_are_global_exclusions": False,
    }


def _prefix_collisions(
    sequence: list[int],
    p02: dict[Any, list[str]],
) -> list[dict[str, Any]]:
    """Return every P02 exact tuple occurring at a scored prefix.

    Position zero is BOS.  A scored position is consequently one-based in the
    returned audit and is the endpoint's position ID in the evaluator input.
    """

    collisions: list[dict[str, Any]] = []
    for scored_position in range(1, len(sequence)):
        prefix = tuple(sequence[: scored_position + 1])
        # Current P02 records are keyed by the full sequence, endpoint
        # position, and endpoint ID.  The fallback keeps this helper useful
        # for tiny synthetic tests that provide the historical sequence-only
        # map; production selection always uses the exact three-part key.
        case_ids = p02.get((prefix, scored_position, int(sequence[scored_position])))
        if case_ids is None:
            case_ids = p02.get(prefix)
        if not case_ids:
            continue
        for case_id in case_ids:
            collisions.append(
                {
                    "case_id": case_id,
                    "scored_position": scored_position,
                    "endpoint_id": int(sequence[scored_position]),
                    "prefix_length": len(prefix),
                }
            )
    return collisions


def _style_for_category(category: Any) -> str | None:
    if not isinstance(category, str):
        return None
    for style in STYLE_ORDER:
        if category in STYLE_CATEGORIES[style]:
            return style
    return None


def _walk_prior_records(value: Any, *, source_path: Path) -> list[dict[str, Any]]:
    """Extract public prior IDs/hashes from old manifests without source text."""

    found: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            text_hash = node.get("text_sha256") or node.get("source_sha256")
            record_id = node.get("record_id") or node.get("opaque_id")
            if isinstance(text_hash, str) and isinstance(record_id, str):
                index = node.get("dataset_index", node.get("index"))
                found.append(
                    {
                        "record_id": record_id,
                        "dataset_index": int(index) if isinstance(index, int) else None,
                        "text_sha256": text_hash,
                        "source_manifest": str(source_path),
                    }
                )
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in found:
        key = (row["source_manifest"], row["record_id"], row["text_sha256"])
        unique[key] = row
    return list(unique.values())


def _prior_manifest_paths(repo_root: Path) -> list[Path]:
    return [
        repo_root / "experiments/TRR-P01/runtime/panel-20260905/panel_manifest.json",
        repo_root / "experiments/TRR-0001/manifest.json",
        repo_root / "experiments/TRR-0002/configuration-search/public-pile/records.json",
        repo_root / "experiments/TRR-0002/strict-surrogate-heavy/selection-reveal.json",
    ]


def _prior_exclusion_audit(repo_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in _prior_manifest_paths(repo_root):
        if not path.is_file():
            manifests.append({"path": str(path), "available": False})
            continue
        value = _load_json(path)
        extracted = _walk_prior_records(value, source_path=path)
        records.extend(extracted)
        manifests.append(
            {
                "path": str(path),
                "available": True,
                "sha256": sha256_file(path),
                "extracted_record_count": len(extracted),
            }
        )

    commitments: list[dict[str, Any]] = []
    for path in (
        repo_root / "experiments/TRR-0002/blind/selection_commitment.json",
        repo_root / "experiments/TRR-0002/configuration-search/fresh-blind/selection_commitment.json",
    ):
        if not path.is_file():
            commitments.append({"path": str(path), "available": False})
            continue
        value = _load_json(path)
        commitments.append(
            {
                "path": str(path),
                "available": True,
                "sha256": sha256_file(path),
                "record_count": value.get("record_count"),
                "source_identity_disclosed": value.get("source_identity_disclosed"),
                "selection_key_disclosed": value.get("selection_key_disclosed"),
                "opaque_record_order": value.get("opaque_record_order", []),
                "audit_limit": "opaque commitment has no recoverable source hashes; no claim of non-overlap with these hidden rows",
            }
        )

    unique_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        unique_by_hash[row["text_sha256"]].append(row)
    known_indices = sorted(
        {
            int(row["dataset_index"])
            for row in records
            if isinstance(row.get("dataset_index"), int)
        }
    )
    return {
        "known_opened_manifests": manifests,
        "known_opened_record_count": len(records),
        "known_opened_unique_text_hash_count": len(unique_by_hash),
        "known_opened_records": sorted(
            records, key=lambda row: (row["source_manifest"], row["record_id"], row["text_sha256"])
        ),
        "known_opened_dataset_indices": known_indices,
        "known_opened_dataset_identities": [
            "NeelNanda/pile-10k@127bfedcd5047750df5ccf3a12979a47bfa0bafa"
        ],
        "known_opened_text_hashes": sorted(unique_by_hash),
        "opaque_commitments": commitments,
        "rule": "reject exact text-hash matches and any known prior source row; reject whole candidate records on P02 exact-prefix collisions; never globally ban token IDs",
        "cross_dataset_limit": "hidden blind commitment mappings remain unaudited because their source identities are intentionally withheld",
    }


def _selection_key(stage: str, length: int, style: str, dataset_index: int, seed: int) -> str:
    raw = f"TRR-P03|{DATASET_REVISION}|{stage}|{length}|{style}|{dataset_index}|{seed}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _row_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is empty or not a string")
    return prompt


def _select_panel(
    dataset: Any,
    tokenizer: Any,
    *,
    p02: dict[tuple[int, ...], list[str]],
    prior_hashes: set[str],
    prior_indices: set[int],
    stage: str,
    seed: int,
    excluded_indices: set[int],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_indices = set(excluded_indices)
    rejections: list[dict[str, Any]] = []

    for length in LENGTHS:
        for style in STYLE_ORDER:
            candidates: list[tuple[str, int]] = []
            for dataset_index in range(len(dataset)):
                if dataset_index in used_indices:
                    continue
                row = dataset[int(dataset_index)]
                if _style_for_category(row.get("category")) != style:
                    continue
                candidates.append(
                    (
                        _selection_key(stage, length, style, dataset_index, seed),
                        int(dataset_index),
                    )
                )
            candidates.sort(key=lambda item: (item[0], item[1]))
            accepted_for_slot = 0
            for selection_key, dataset_index in candidates:
                if accepted_for_slot >= PER_STYLE_PER_LENGTH:
                    break
                row = dataset[int(dataset_index)]
                try:
                    prompt = _row_prompt(row)
                except ValueError as exc:
                    rejections.append(
                        {
                            "stage": stage,
                            "length": length,
                            "style": style,
                            "dataset_index": dataset_index,
                            "reason": "empty_prompt",
                            "detail": str(exc),
                        }
                    )
                    continue
                text_hash = sha256_text(prompt)
                if text_hash in prior_hashes:
                    rejections.append(
                        {
                            "stage": stage,
                            "length": length,
                            "style": style,
                            "dataset_index": dataset_index,
                            "text_sha256": text_hash,
                            "reason": "known_prior_text_hash",
                        }
                    )
                    continue
                if dataset_index in prior_indices:
                    rejections.append(
                        {
                            "stage": stage,
                            "length": length,
                            "style": style,
                            "dataset_index": dataset_index,
                            "text_sha256": text_hash,
                            "reason": "known_prior_dataset_index",
                        }
                    )
                    continue
                token_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
                if len(token_ids) < length:
                    rejections.append(
                        {
                            "stage": stage,
                            "length": length,
                            "style": style,
                            "dataset_index": dataset_index,
                            "text_sha256": text_hash,
                            "source_token_count": len(token_ids),
                            "reason": "too_short",
                        }
                    )
                    continue
                sequence = [BOS_TOKEN_ID] + token_ids[:length]
                collisions = _prefix_collisions(sequence, p02)
                if collisions:
                    rejections.append(
                        {
                            "stage": stage,
                            "length": length,
                            "style": style,
                            "dataset_index": dataset_index,
                            "text_sha256": text_hash,
                            "reason": "p02_exact_prefix_collision",
                            "collisions": collisions,
                        }
                    )
                    continue
                record_ordinal = len(selected) + 1
                # Stage-local IDs are opaque to the attack process.  Their
                # deterministic length-major order leaves the fixed within-stratum
                # indices [0, 2, 4, 5] as the predeclared A1/A2 anchor.
                record_id = f"p03-{stage}-r{record_ordinal:04d}"
                selected.append(
                    {
                        "record_id": record_id,
                        "stage": stage,
                        "scored_tokens": length,
                        "sequence_length": length + 1,
                        "style": style,
                        "category": str(row.get("category")),
                        "dataset_index": dataset_index,
                        "prompt_id": str(row.get("prompt_id", "")),
                        "source_text_sha256": text_hash,
                        "source_token_count": len(token_ids),
                        "selection_key": selection_key,
                        "input_ids": sequence,
                    }
                )
                used_indices.add(dataset_index)
                accepted_for_slot += 1
            if accepted_for_slot != PER_STYLE_PER_LENGTH:
                counts = Counter(
                    row.get("category")
                    for row in (dataset[i] for i in range(len(dataset)))
                    if _style_for_category(row.get("category")) == style
                )
                raise RuntimeError(
                    f"could not fill {stage} length={length} style={style}: "
                    f"accepted={accepted_for_slot}, candidates={len(candidates)}, metadata_counts={dict(counts)}"
                )

    # Reset the ordinal to the stage-local length-major order.  The temporary
    # append order is already length-major, but this assertion protects a
    # future selector edit from silently moving the A1/A2 anchor.
    expected = [f"p03-{stage}-r{i:04d}" for i in range(1, len(selected) + 1)]
    if [row["record_id"] for row in selected] != expected:
        raise RuntimeError("opaque stage record IDs are not contiguous")
    audit["rejections"].extend(rejections)
    audit["stage_counts"][stage] = {
        "records": len(selected),
        "by_length": {str(length): sum(row["scored_tokens"] == length for row in selected) for length in LENGTHS},
        "by_style": {style: sum(row["style"] == style for row in selected) for style in STYLE_ORDER},
        "rejected_candidates": len(rejections),
    }
    return selected


def _asset_discovery(
    *,
    dataset_path: Path,
    p02_path: Path,
    base_snapshot: Path,
    target_snapshot: Path,
    prior_audit: dict[str, Any],
) -> dict[str, Any]:
    def snapshot_record(
        path: Path,
        *,
        model_id: str,
        revision: str,
        config_sha: str,
        tokenizer_sha: str,
        weights_sha: str,
        weights_bytes: int,
    ) -> dict[str, Any]:
        config = path / "config.json"
        tokenizer_config = path / "tokenizer_config.json"
        require_readable_file(config, f"{model_id} config")
        require_readable_file(tokenizer_config, f"{model_id} tokenizer config")
        if sha256_file(config) != config_sha:
            raise RuntimeError(f"{model_id} config hash changed")
        if sha256_file(tokenizer_config) != tokenizer_sha:
            raise RuntimeError(f"{model_id} tokenizer config hash changed")
        weight_files = [
            item
            for item in path.iterdir()
            if item.is_file() and item.name.endswith((".safetensors", ".bin", ".pth"))
        ]
        if not weight_files:
            raise RuntimeError(f"no local weight file found in {path}")
        total_bytes = sum(item.stat().st_size for item in weight_files)
        if total_bytes != weights_bytes:
            raise RuntimeError(
                f"{model_id} weight bytes changed: expected {weights_bytes}, got {total_bytes}"
            )
        return {
            "id": model_id,
            "revision": revision,
            "snapshot": str(path),
            "config": {"path": str(config), "sha256": config_sha, "bytes": config.stat().st_size},
            "tokenizer_config": {
                "path": str(tokenizer_config),
                "sha256": tokenizer_sha,
                "bytes": tokenizer_config.stat().st_size,
            },
            "weight_files": [
                {"name": item.name, "bytes": item.stat().st_size} for item in sorted(weight_files)
            ],
            "weights": {
                "sha256": weights_sha,
                "bytes": weights_bytes,
                "hash_basis": "verified pinned local snapshot blob; no model load or rehash during preparation",
            },
            "model_loaded": False,
        }

    for path, label in ((BOUNDARY_PROTOTYPES, "boundary prototypes"), (HISTORICAL_LENS, "historical lens")):
        require_readable_file(path, label)
    if BOUNDARY_PROTOTYPES.stat().st_size != BOUNDARY_PROTOTYPES_BYTES:
        raise RuntimeError("boundary prototype byte size changed")
    if HISTORICAL_LENS.stat().st_size != HISTORICAL_LENS_BYTES:
        raise RuntimeError("historical lens byte size changed")
    return {
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "arrow": file_record(dataset_path),
            "columns_used": ["prompt", "prompt_id", "category"],
            "columns_ignored": ["messages"],
            "source_text_field": "prompt",
            "construction": "source prompt as-is; add BOS 128000; crop first requested post-BOS tokens; no instruction or Unicode wrapper",
        },
        "tokenizer": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot": str(base_snapshot),
            "bos_token_id": BOS_TOKEN_ID,
            "add_special_tokens": False,
        },
        "base_target": snapshot_record(
            base_snapshot,
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            config_sha=BASE_CONFIG_SHA256,
            tokenizer_sha=BASE_TOKENIZER_CONFIG_SHA256,
            weights_sha=BASE_WEIGHTS_SHA256,
            weights_bytes=BASE_WEIGHTS_BYTES,
        ),
        "shifted_target": snapshot_record(
            target_snapshot,
            model_id=TARGET_ID,
            revision=TARGET_REVISION,
            config_sha=TARGET_CONFIG_SHA256,
            tokenizer_sha=TARGET_TOKENIZER_CONFIG_SHA256,
            weights_sha=TARGET_WEIGHTS_SHA256,
            weights_bytes=TARGET_WEIGHTS_BYTES,
        ),
        "historical_assets": {
            "boundary_prototypes": {
                "path": str(BOUNDARY_PROTOTYPES),
                "sha256": BOUNDARY_PROTOTYPES_SHA256,
                "bytes": BOUNDARY_PROTOTYPES_BYTES,
                "mode": "read_only",
                "construction": "public base prefix [BOS,v] at post-BOS position 1",
            },
            "historical_affine_lens": {
                "path": str(HISTORICAL_LENS),
                "sha256": HISTORICAL_LENS_SHA256,
                "bytes": HISTORICAL_LENS_BYTES,
                "mode": "read_only",
                "provenance": "public Alpaca fitted comparator; no TRR-P03 refit",
            },
        },
        "p02_exclusion": {"path": str(p02_path), "sha256": sha256_file(p02_path)},
        "prior_exclusion_audit_summary": {
            "known_opened_record_count": prior_audit["known_opened_record_count"],
            "known_opened_unique_text_hash_count": prior_audit["known_opened_unique_text_hash_count"],
            "opaque_commitment_count": len(prior_audit["opaque_commitments"]),
        },
        "preparation_limits": {
            "model_loaded": False,
            "forward_calls": 0,
            "gpu_used": False,
            "full_table_loaded": False,
            "resource_hold": {
                "source_thread_id": "01a07029-b0e2-71d2-af08-ecd9aacc645f",
                "captured_utc": utc_now(),
                "status": "HEAVY_COMPUTE_HELD_PENDING_ROOT_GRANT",
                "reservation_context": "TRR-0004 paired observation preparation and sequential timing runs",
            },
        },
    }


def _record_public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": row["record_id"],
        "stage": row["stage"],
        "scored_tokens": row["scored_tokens"],
        "sequence_length": row["sequence_length"],
        "style": row["style"],
        "category": row["category"],
        "source_token_count": row["source_token_count"],
        "source_text_sha256": row["source_text_sha256"],
        "dataset_index": row["dataset_index"],
        "prompt_id": row["prompt_id"],
    }


def _observation_index(stage1: list[dict[str, Any]], stage2: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = stage1 + stage2
    bundles: list[dict[str, Any]] = []
    for bundle_id in ("bundle-a", "bundle-b"):
        for stage_name, rows in (("stage1", stage1), ("stage2_holdout", stage2)):
            for length in LENGTHS:
                group = [row for row in rows if row["scored_tokens"] == length]
                if len(group) != RECORDS_PER_LENGTH:
                    raise RuntimeError("observation group cardinality changed")
                slots = length + 1
                bundles.append(
                    {
                        "bundle_id": bundle_id,
                        "stage": stage_name,
                        "scored_tokens": length,
                        "sequence_length": slots,
                        "record_ids": [row["record_id"] for row in group],
                        "relative_path": f"observations/{bundle_id}/{stage_name}_len{length}.safetensors",
                        "keys": {
                            "activations": "activations",
                            "attention_mask": "attention_mask",
                            "position_ids": "position_ids",
                        },
                        "expected_shapes": {
                            "activations": [RECORDS_PER_LENGTH, slots, HIDDEN_SIZE],
                            "attention_mask": [RECORDS_PER_LENGTH, slots],
                            "position_ids": [RECORDS_PER_LENGTH, slots],
                        },
                        "activation_definition": "post-cut residual stream at cut_depth=4, one hidden vector per sequence slot",
                        "status": "PENDING_EVALUATOR_GENERATION",
                    }
                )
    return {
        "schema": SCHEMA_OBSERVATIONS,
        "task_id": TASK_ID,
        "status": "PREDECLARED_BEFORE_TARGET_OBSERVATIONS",
        "truth_opened": False,
        "source_truth_included": False,
        "attack_input_must_exclude": [
            "source plaintext",
            "source token IDs",
            "dataset index",
            "prompt ID",
            "source text hash",
            "style and category labels",
            "target condition labels and model paths",
            "truth and correctness signal",
            "P01/P02 opened outputs",
        ],
        "public_inputs": [
            "activation tensor",
            "attention mask",
            "position IDs",
            "cut depth",
            "opaque bundle ID",
            "opaque record ID",
            "public readout asset IDs and hashes",
            "frozen method configuration",
        ],
        "geometry": {
            "bos_token_id": BOS_TOKEN_ID,
            "cut_depth": CUT_DEPTH,
            "hidden_size": HIDDEN_SIZE,
            "lengths_scored_post_bos": list(LENGTHS),
            "records_per_length_stage1": RECORDS_PER_LENGTH,
            "stage1_records": len(stage1),
            "stage2_holdout_records": len(stage2),
        },
        "record_order": [row["record_id"] for row in all_rows],
        "bundles": bundles,
        "pairing_rule": "bundle-a and bundle-b use identical opaque record order, sequence geometry, masks, and positions; only evaluator target weights differ",
        "stage2_rule": "stage2_holdout bundles remain unopened/unscored until the predeclared Stage-1 gate advances and compactness constants are frozen",
    }


def _truth_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_TRUTH_INDEX,
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_truth_included": True,
        "status": "SEALED_EVALUATOR_ONLY",
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "records": [
            {
                "record_id": row["record_id"],
                "stage": row["stage"],
                "scored_tokens": row["scored_tokens"],
                "sequence_length": row["sequence_length"],
                "style": row["style"],
                "category": row["category"],
                "dataset_index": row["dataset_index"],
                "prompt_id": row["prompt_id"],
                "source_text_sha256": row["source_text_sha256"],
                "source_token_count": row["source_token_count"],
                "input_ids_sha256": sha256_ids(row["input_ids"]),
            }
            for row in rows
        ],
        "opening_rule": "A scorer may open this sidecar only after all Stage-1 predictions and prediction hashes are frozen; Stage-2 rows remain sealed until the predeclared Stage-1 disposition",
    }


def _plan_input(
    *,
    asset_discovery: dict[str, Any],
    stage1: list[dict[str, Any]],
    stage2: list[dict[str, Any]],
    observation_index: dict[str, Any],
    p02_info: dict[str, Any],
    prior_audit: dict[str, Any],
) -> dict[str, Any]:
    length39 = [
        row for row in stage1 if row["scored_tokens"] == 39
    ]
    anchor_ids = [length39[index]["record_id"] for index in (0, 2, 4, 5)]
    return {
        "schema": "token-reconstruction.trr-p03-setup-plan-input.v1",
        "task_id": TASK_ID,
        "status": "DRAFT_INPUT_FOR_ROOT_FREEZE",
        "truth_opened": False,
        "parent": {"commit": PARENT_COMMIT, "task_id": "TRR-P02"},
        "panel": {
            "stage1_records": len(stage1),
            "stage2_holdout_records": len(stage2),
            "lengths_post_bos": list(LENGTHS),
            "records_per_length": RECORDS_PER_LENGTH,
            "scored_tokens_per_target_stage1": sum(row["scored_tokens"] for row in stage1),
            "per_length_style_quota": {style: PER_STYLE_PER_LENGTH for style in STYLE_ORDER},
            "style_order": list(STYLE_ORDER),
            "a1_a2_anchor_record_ids": anchor_ids,
            "a1_a2_anchor_scored_tokens": 39,
            "construction": "cached public prompts as-is, BOS plus first N post-BOS tokens, no wrappers or score-adaptive filtering",
        },
        "selection": {
            "stage1_seed": STAGE1_SEED,
            "stage2_seed": STAGE2_SEED,
            "ordering": "sha256(TRR-P03|dataset_revision|stage|length|style|dataset_index|seed), then dataset index",
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "source_field": "prompt",
            "metadata_style_field": "category",
            "style_categories": {style: list(categories) for style, categories in STYLE_CATEGORIES.items()},
            "exact_exclusions": {
                "p02_tuple_count": p02_info["case_count"],
                "p02_unique_sequence_count": p02_info["unique_sequence_count"],
                "whole_record_on_prefix_collision": True,
                "global_token_id_ban": False,
                "known_prior_exact_text_hash": True,
                "known_prior_dataset_index": "only when prior dataset identity exactly matches; inactive for no_robots versus prior Pile records",
                "hidden_commitment_mapping_available": False,
            },
            "prior_audit_counts": {
                "known_opened_records": prior_audit["known_opened_record_count"],
                "known_opened_unique_text_hashes": prior_audit["known_opened_unique_text_hash_count"],
                "same_dataset_index_exclusion": False,
            },
        },
        "target_conditions_evaluator_only": {
            "bundle-a": {
                "condition_id": "matched_public",
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
            },
            "bundle-b": {
                "condition_id": "shifted_full_sft",
                "model_id": TARGET_ID,
                "revision": TARGET_REVISION,
                "historical_p01_label": "shifted_target_lora",
                "provenance_correction": "cached resource is a full public Vikhr SFT checkpoint; do not label the resource itself as LoRA",
            },
        },
        "observation_interface": {
            "index_schema": observation_index["schema"],
            "activation_key": "activations",
            "attention_mask_key": "attention_mask",
            "position_ids_key": "position_ids",
            "cut_depth": CUT_DEPTH,
            "bundle_pairing": observation_index["pairing_rule"],
            "truth_separation": "stage1/private_truth.jsonl and stage2_holdout/private_truth.jsonl are evaluator-only; observation index has no source IDs, hashes, or token IDs",
        },
        "assets": {
            "base_snapshot": asset_discovery["base_target"],
            "shifted_snapshot": asset_discovery["shifted_target"],
            "boundary_prototypes": asset_discovery["historical_assets"]["boundary_prototypes"],
            "historical_affine_lens": asset_discovery["historical_assets"]["historical_affine_lens"],
        },
        "resource_boundary": {
            "full_model_or_table_run": False,
            "status": "HELD_PENDING_EXPLICIT_ROOT_GRANT",
            "no_truth_scoring": True,
        },
    }


def _write_truth_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            # This is the evaluator-only contract consumed by the scorer.
            # No source text or metadata is placed in the truth rows.
            handle.write(
                json.dumps(
                    {"record_id": row["record_id"], "token_ids": row["input_ids"]},
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _evaluator_panel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "token-reconstruction.trr-p03-setup-panel.v1",
        "task_id": TASK_ID,
        "status": "PANEL_FROZEN_BEFORE_TARGET_OBSERVATIONS",
        "truth_opened": False,
        "source_truth_included": True,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION, "bos_token_id": BOS_TOKEN_ID},
        "records": [
            {
                "record_id": row["record_id"],
                "stage": row["stage"],
                "style": row["style"],
                "category": row["category"],
                "scored_tokens": row["scored_tokens"],
                "sequence_length": row["sequence_length"],
                "token_ids": row["input_ids"],
                "source_token_count": row["source_token_count"],
                "source_text_sha256": row["source_text_sha256"],
                "dataset_index": row["dataset_index"],
                "prompt_id": row["prompt_id"],
            }
            for row in rows
        ],
        "a2_anchor_record_ids": [
            length39[index]["record_id"] for index in (0, 2, 4, 5)
        ] if (length39 := [
            row for row in rows
            if row["stage"] == "s1" and row["scored_tokens"] == 39
        ]) else [],
        "usage": "evaluator-only; never copy this file into a reconstruction input root",
    }


def _interface_contract(
    stage1: list[dict[str, Any]], stage2: list[dict[str, Any]]
) -> dict[str, Any]:
    records = stage1 + stage2
    return {
        "schema": "token-reconstruction.trr-p03-preparation-interface.v1",
        "task_id": TASK_ID,
        "status": "DRAFT_INPUT_FOR_ROOT_FREEZE",
        "truth_opened": False,
        "source_truth_included": False,
        "source_side_artifacts": {
            "evaluator_panel": {
                "stage1": "stage1/evaluator_panel.json",
                "stage2_holdout": "stage2_holdout/evaluator_panel.json",
            },
            "private_truth": {
                "stage1": "stage1/private_truth.jsonl",
                "stage2_holdout": "stage2_holdout/private_truth.jsonl",
            },
            "truth_index": {
                "stage1": "stage1/truth_index.json",
                "stage2_holdout": "stage2_holdout/truth_index.json",
            },
            "rule": "source-side artifacts remain evaluator-only and are never mounted in the reconstruction process",
        },
        "runtime_observation_index_contract": {
            "schema": "token-reconstruction.trr-p03-observation-index.v1",
            "top_level": {
                "truth_opened": False,
                "source_truth_included": False,
                "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
                "cut_depth": CUT_DEPTH,
                "bos_token_id": BOS_TOKEN_ID,
            },
            "per_record_fields": {
                "required": [
                    "record_id",
                    "sequence_length",
                    "path",
                    "bytes",
                    "sha256",
                    "shape",
                    "dtype",
                    "mask_digest",
                    "position_digest",
                ],
                "optional": [],
                "forbidden": [
                    "token_ids",
                    "input_ids",
                    "dataset_index",
                    "prompt_id",
                    "source_text_sha256",
                    "category",
                    "style",
                    "target_condition",
                    "truth",
                ],
                "shape": [1, "sequence_length", HIDDEN_SIZE],
                "tensor_keys": ["activation", "attention_mask", "position_ids"],
            },
            "observation_artifact": {
                "schema": "token-reconstruction.boundary-observation.v1",
                "activation_dtype": "bfloat16",
                "activation_shape": [1, "sequence_length", HIDDEN_SIZE],
                "attention_mask_shape": [1, "sequence_length"],
                "position_ids_shape": [1, "sequence_length"],
                "positions": "0..sequence_length-1; BOS occupies position 0 and is not scored",
            },
        },
        "records": [
            {
                "record_id": row["record_id"],
                "stage": row["stage"],
                "scored_tokens": row["scored_tokens"],
                "sequence_length": row["sequence_length"],
            }
            for row in records
        ],
        "bundle_pairing": {
            "opaque_bundle_ids": ["bundle-a", "bundle-b"],
            "mapping_is_evaluator_only": True,
            "record_order_identical": True,
            "masks_positions_identical": True,
            "bundle_a_condition": "matched_public (evaluator-only label)",
            "bundle_b_condition": "shifted_full_sft (evaluator-only label; historical P01 label shifted_target_lora)",
            "shifted_target_required_for_stage1": True,
        },
        "grouping": {
            "per_stage_lengths": list(LENGTHS),
            "records_per_length": RECORDS_PER_LENGTH,
            "recommended_observation_path": "observations/{opaque_bundle_id}/{record_id}.safetensors",
            "one_observation_index_per_bundle": True,
        },
        "freeze_order": [
            "freeze plan, assets, method source, panel selection, and target-condition map",
            "generate both target bundles with source-aware evaluator only",
            "publish/hash observation indexes and prediction archives before opening Stage-1 truth",
            "open private_truth.jsonl once for complete frozen Stage-1 matrix; keep Stage-2 sealed until gate disposition",
        ],
        "stage2": {
            "record_ids": [row["record_id"] for row in stage2],
            "truth_status": "sealed_unopened_until_stage1_gate_and_compactness_freeze",
        },
    }



def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    root = require_new_dir(output_root)
    started_utc = utc_now()
    dataset_path = args.dataset_path.resolve()
    p02_path = args.p02_exclusion.resolve()
    base_snapshot = args.base_snapshot.resolve()
    target_snapshot = args.target_snapshot.resolve()
    repo_root = args.repo_root.resolve()

    p02, p02_info = _p02_tuples(p02_path)
    prior_audit = _prior_exclusion_audit(repo_root)
    asset_discovery = _asset_discovery(
        dataset_path=dataset_path,
        p02_path=p02_path,
        base_snapshot=base_snapshot,
        target_snapshot=target_snapshot,
        prior_audit=prior_audit,
    )
    dataset = _load_dataset(dataset_path)
    tokenizer = _load_tokenizer(base_snapshot)

    selection_audit: dict[str, Any] = {
        "schema": SCHEMA_AUDIT,
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_truth_included": True,
        "status": "PANEL_FROZEN_BEFORE_TARGET_OBSERVATIONS",
        "started_utc": started_utc,
        "rejections": [],
        "stage_counts": {},
        "p02": p02_info,
        "prior": {
            "known_opened_records": prior_audit["known_opened_record_count"],
            "known_opened_unique_text_hashes": prior_audit["known_opened_unique_text_hash_count"],
            "opaque_commitments": prior_audit["opaque_commitments"],
        },
        "selection_rule": "reject candidate if any source hash/index is known prior or if any sequence prefix exactly equals a P02 tuple; skip whole candidate record; token IDs remain eligible elsewhere",
    }
    prior_hashes = set(prior_audit["known_opened_text_hashes"])
    current_identity = f"{DATASET_ID}@{DATASET_REVISION}"
    prior_indices = (
        set(prior_audit["known_opened_dataset_indices"])
        if current_identity in set(prior_audit.get("known_opened_dataset_identities", []))
        else set()
    )
    selection_audit["prior"]["same_dataset_index_exclusion_active"] = bool(prior_indices)
    selection_audit["prior"]["current_dataset_identity"] = current_identity
    stage1 = _select_panel(
        dataset,
        tokenizer,
        p02=p02,
        prior_hashes=prior_hashes,
        prior_indices=prior_indices,
        stage="s1",
        seed=STAGE1_SEED,
        excluded_indices=set(),
        audit=selection_audit,
    )
    stage1_indices = {row["dataset_index"] for row in stage1}
    stage2 = _select_panel(
        dataset,
        tokenizer,
        p02=p02,
        prior_hashes=prior_hashes,
        prior_indices=prior_indices,
        stage="s2",
        seed=STAGE2_SEED,
        excluded_indices=stage1_indices,
        audit=selection_audit,
    )
    if stage1_indices.intersection(row["dataset_index"] for row in stage2):
        raise RuntimeError("Stage-1 and Stage-2 source rows overlap")
    if len(stage1) != 24 or len(stage2) != 24:
        raise RuntimeError("stage cardinality changed")
    if sum(row["scored_tokens"] for row in stage1) != 1482:
        raise RuntimeError("Stage-1 scored-token total changed")

    all_rows = stage1 + stage2
    manifest = {
        "schema": SCHEMA_PANEL,
        "task_id": TASK_ID,
        "status": "PANEL_FROZEN_BEFORE_TARGET_OBSERVATIONS",
        "truth_opened": False,
        "source_truth_included": True,
        "dataset": asset_discovery["dataset"],
        "tokenizer": asset_discovery["tokenizer"],
        "selection": {
            "stage1_seed": STAGE1_SEED,
            "stage2_seed": STAGE2_SEED,
            "styles": list(STYLE_ORDER),
            "style_categories": {style: list(categories) for style, categories in STYLE_CATEGORIES.items()},
            "records_per_style_per_length": PER_STYLE_PER_LENGTH,
            "lengths_post_bos": list(LENGTHS),
            "construction": "BOS 128000 plus first N source prompt token IDs; no source transformation",
            "p02_exact_prefix_filter": True,
            "global_token_id_exclusion": False,
        },
        "stage1": {
            "panel_id": "trr-p03-natural-stage1",
            "records": len(stage1),
            "scored_tokens_total_per_target": sum(row["scored_tokens"] for row in stage1),
            "record_order": [row["record_id"] for row in stage1],
        },
        "stage2_holdout": {
            "panel_id": "trr-p03-natural-stage2-holdout",
            "records": len(stage2),
            "scored_tokens_total_per_target": sum(row["scored_tokens"] for row in stage2),
            "record_order": [row["record_id"] for row in stage2],
            "truth_status": "sealed_unopened_until_stage1_gate",
        },
        "a1_a2_anchor": {
            "record_ids": [
                row["record_id"]
                for index, row in enumerate(
                    [item for item in stage1 if item["scored_tokens"] == 39]
                )
                if index in (0, 2, 4, 5)
            ],
            "within_length39_zero_based_indices": [0, 2, 4, 5],
            "scored_tokens": 39,
            "sequence_slots_including_bos": 40,
            "selection": "first four records in the predeclared length-39 Stage-1 stratum, before observations or truth scoring",
        },
        "records": [_record_public(row) for row in all_rows],
        "private_truth": {
            "stage1_path": "stage1/private_truth.jsonl",
            "stage2_holdout_path": "stage2_holdout/private_truth.jsonl",
            "row_schema": "{record_id, token_ids}",
            "source_plaintext": "not stored",
        },
        "observation_index": "observation_index.json",
    }
    observation_index = _observation_index(stage1, stage2)
    plan_input = _plan_input(
        asset_discovery=asset_discovery,
        stage1=stage1,
        stage2=stage2,
        observation_index=observation_index,
        p02_info=p02_info,
        prior_audit=prior_audit,
    )

    stage1_root = root / "stage1"
    stage2_root = root / "stage2_holdout"
    stage1_root.mkdir()
    stage2_root.mkdir()
    truth_stage1_path = stage1_root / "private_truth.jsonl"
    truth_stage2_path = stage2_root / "private_truth.jsonl"
    _write_truth_jsonl(truth_stage1_path, stage1)
    _write_truth_jsonl(truth_stage2_path, stage2)
    truth_index_stage1_path = stage1_root / "truth_index.json"
    truth_index_stage2_path = stage2_root / "truth_index.json"
    write_json_new(truth_index_stage1_path, _truth_index(stage1))
    write_json_new(truth_index_stage2_path, _truth_index(stage2))
    evaluator_panel_stage1_path = stage1_root / "evaluator_panel.json"
    evaluator_panel_stage2_path = stage2_root / "evaluator_panel.json"
    write_json_new(evaluator_panel_stage1_path, _evaluator_panel(stage1))
    write_json_new(evaluator_panel_stage2_path, _evaluator_panel(stage2))
    manifest_path = root / "panel_manifest.json"
    write_json_new(manifest_path, manifest)
    observation_path = root / "observation_index.json"
    write_json_new(observation_path, observation_index)
    interface_path = root / "interface.json"
    write_json_new(interface_path, _interface_contract(stage1, stage2))
    plan_path = root / "plan-input.json"
    write_json_new(plan_path, plan_input)

    prior_path = root / "prior-exclusion-audit.json"
    write_json_new(prior_path, prior_audit)
    asset_path = root / "resource-discovery.json"
    write_json_new(asset_path, asset_discovery)
    audit_path = root / "selection-audit.json"
    selection_audit["ended_utc"] = utc_now()
    selection_audit["selected"] = {
        "stage1_record_ids": [row["record_id"] for row in stage1],
        "stage2_record_ids": [row["record_id"] for row in stage2],
        "stage1_dataset_indices": [row["dataset_index"] for row in stage1],
        "stage2_dataset_indices": [row["dataset_index"] for row in stage2],
        "stage1_text_hashes": [row["source_text_sha256"] for row in stage1],
        "stage2_text_hashes": [row["source_text_sha256"] for row in stage2],
    }
    selection_audit["collision_rejection_count"] = sum(
        item.get("reason") == "p02_exact_prefix_collision" for item in selection_audit["rejections"]
    )
    selection_audit["rejection_counts_by_reason"] = dict(
        Counter(item.get("reason") for item in selection_audit["rejections"])
    )
    write_json_new(audit_path, selection_audit)

    hold_path = root / "resource-hold-receipt.json"
    write_json_new(
        hold_path,
        {
            "schema": "token-reconstruction.trr-p03-resource-hold-receipt.v1",
            "task_id": TASK_ID,
            "status": "HEAVY_COMPUTE_HELD_PENDING_EXPLICIT_ROOT_GRANT",
            "source_thread_id": "01a07029-b0e2-71d2-af08-ecd9aacc645f",
            "captured_utc": utc_now(),
            "reservation_context": "TRR-0004 paired observation preparation and five sequential isolated timing runs",
            "work_performed": "dataset metadata and tokenizer-only panel preparation; no model load, forward pass, GPU use, or full boundary-table read",
            "model_loaded": False,
            "forward_calls": 0,
            "gpu_used": False,
        },
    )
    deviation_path = root / "routing-deviation.json"
    write_json_new(
        deviation_path,
        {
            "schema": "token-reconstruction.trr-p03-setup-deviation.v1",
            "task_id": TASK_ID,
            "kind": "accidental_plan_message_routing",
            "occurred_utc": started_utc,
            "wrong_thread_id": "01a07029-b0e2-71d2-af08-ecd9aacc645f",
            "correct_root_thread_id": "01a07061-9c25-7e13-bcac-12d57c41c666",
            "content_scope": "P03 planning/interface only; no truth, source plaintext, record IDs, scores, or private data",
            "disposition": "TRR-0004 reported its methods/records were already frozen and will not use the plan; no scientific output was received from that thread",
        },
    )
    env_path = root / "environment.json"
    write_json_new(
        env_path,
        {
            "schema": "token-reconstruction.trr-p03-preparation-environment.v1",
            "task_id": TASK_ID,
            "captured_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": platform.node(),
            "cwd": str(Path.cwd()),
            "environment_flags": {
                name: os.environ.get(name)
                for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
                if name in os.environ
            },
            "model_loaded": False,
            "full_table_loaded": False,
            "truth_opened": False,
        },
    )

    artifact_records = {}
    for path in (
        truth_stage1_path,
        truth_stage2_path,
        truth_index_stage1_path,
        truth_index_stage2_path,
        evaluator_panel_stage1_path,
        evaluator_panel_stage2_path,
        manifest_path,
        observation_path,
        interface_path,
        plan_path,
        prior_path,
        asset_path,
        audit_path,
        hold_path,
        deviation_path,
        env_path,
    ):
        artifact_records[path.relative_to(root).as_posix()] = file_record(path, root=root)
    receipt_path = root / "setup-receipt.json"
    write_json_new(
        receipt_path,
        {
            "schema": "token-reconstruction.trr-p03-setup-receipt.v1",
            "task_id": TASK_ID,
            "status": "SETUP_READY_FOR_ROOT_REVIEW",
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "parent_commit": PARENT_COMMIT,
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "stage1_records": len(stage1),
            "stage2_records": len(stage2),
            "stage1_scored_tokens_per_target": sum(row["scored_tokens"] for row in stage1),
            "a1_a2_anchor_record_ids": manifest["a1_a2_anchor"]["record_ids"],
            "p02_collision_rejections": selection_audit["collision_rejection_count"],
            "rejection_counts_by_reason": selection_audit["rejection_counts_by_reason"],
            "truth_opened": False,
            "model_loaded": False,
            "artifact_records": artifact_records,
            "truth_files": {
                "stage1": file_record(truth_stage1_path, root=root),
                "stage2_holdout": file_record(truth_stage2_path, root=root),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "SETUP_READY_FOR_ROOT_REVIEW",
                "output_root": str(root),
                "stage1_records": len(stage1),
                "stage2_records": len(stage2),
                "anchor_ids": manifest["a1_a2_anchor"]["record_ids"],
                "p02_collision_rejections": selection_audit["collision_rejection_count"],
                "rejection_counts": selection_audit["rejection_counts_by_reason"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
