#!/usr/bin/env python3
"""Prepare the public P04 pools and fresh panel without loading a model.

The selector reads only public Arrow rows and the pinned tokenizer.  It writes
record identities, source/rendering hashes, geometry, and provenance; source
text and token IDs are transient and are never written to the selection file.
The PR7 1,200-record Alpaca fit stream is referenced as an immutable replay
pool.  Correction, validation, and fresh-panel records are selected by one
deterministic order after explicit public identity and cross-pool exclusions.

This command does not capture activations, generate teacher scores, load target
weights, or open evaluator truth.  It is safe to run before any model-resource
reservation.  Dataset Arrow files are described by size and known parent
metadata hashes; the selector deliberately does not hash large source files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


TASK_ID = "TRR-P04"
SCHEMA = "token-reconstruction.trr-p04-public-selection.v1"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
TOKENIZER_SNAPSHOT = (
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6"
)
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
MAX_PANEL_POST_BOS = 128
PANEL_LENGTHS = (16, 32, 64, 128)
STYLES = ("pile_plain", "finance_chat", "alpaca_instruction")
STYLE_DATASET = {
    "pile_plain": "NeelNanda/pile-10k",
    "finance_chat": "Josephgflowers/Finance-Instruct-500k",
    "alpaca_instruction": "tatsu-lab/alpaca",
}
STYLE_REVISION = {
    "pile_plain": "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
    "finance_chat": "583a98fb0ec14d904e9423b671d9d0fea88891b6",
    "alpaca_instruction": "dce01c9b08f87459cf36a430d809084718273017",
}
STYLE_SOURCE_HASHES = {
    "pile10k-train.arrow": "77ddf02e2a69373a944bc8bc8ac8f7b9926f5c62203d727341a24d709bf81113",
    "finance-instruct-500k-train-00000-of-00002.arrow": "b49ca0980a0b02fecbef2220eee0ef5d3c3c893ae42b4e1910edec993c3d164e",
    "finance-instruct-500k-train-00001-of-00002.arrow": "ce4b0786646cd68561da736f145fd5df7ba2f4e754e0caa3ae646d6be9900bd3",
    "alpaca-train.arrow": "f45103036ed651f4c06d0a3c3e0fb7d53acb3074ed5c8e804a69c1efc1cea794",
}
DEFAULT_ARROWS = {
    "pile_plain": (
        "/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/"
        "127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow",
    ),
    "finance_chat": (
        "/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/"
        "583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00000-of-00002.arrow",
        "/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/"
        "583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00001-of-00002.arrow",
    ),
    "alpaca_instruction": (
        "/home/alanz/.cache/huggingface/datasets/tatsu-lab___alpaca/default/0.0.0/"
        "dce01c9b08f87459cf36a430d809084718273017/alpaca-train.arrow",
    ),
}
DEFAULT_EXCLUSIONS = (
    "experiments/TRR-0004/fit/affine_fit_records.json",
    "experiments/TRR-0004/fit/affine_validation_records.json",
    "experiments/TRR-0004/fresh_confirmation_v1/panel_capture/panel.json",
    "experiments/TRR-0004/fresh_confirmation_v1/selection_plan.json",
)
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)
FIT_RECORD_COUNT = 1200
CORRECTION_RECORD_COUNT = 256
VALIDATION_RECORD_COUNT = 192
PANEL_RECORD_COUNT = 72
PANEL_RECORDS_PER_STYLE = 24
RECORDS_PER_LENGTH = 6
ANCHOR_RECORDS_PER_STYLE = 4


class PreparationError(ValueError):
    """Raised for a failed-closed public selection or metadata contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def digest_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sequence_sha256(token_ids: Sequence[int]) -> str:
    """Hash a bounded int32 sequence without retaining it in metadata."""

    try:
        payload = struct.pack("<" + "i" * len(token_ids), *(int(v) for v in token_ids))
    except (struct.error, TypeError, ValueError) as exc:
        raise PreparationError("token sequence cannot be represented as int32") from exc
    return sha256_bytes(payload)


def balanced_counts(total: int, parts: int) -> list[int]:
    if total <= 0 or parts <= 0:
        raise ValueError("total and parts must be positive")
    quotient, remainder = divmod(total, parts)
    return [quotient + int(index < remainder) for index in range(parts)]


def seeded_indices(size: int, *, style: str, seed: int) -> list[int]:
    """Return a cross-version deterministic pseudo-random row order."""

    if size < 0 or seed < 0 or not style:
        raise ValueError("size, style, and non-negative seed are required")
    return sorted(
        range(size),
        key=lambda index: (
            sha256_bytes(f"TRR-P04|{style}|row:{index}|seed:{seed}".encode("utf-8")),
            index,
        ),
    )


def _safe_environment() -> dict[str, str]:
    import os

    return {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _max_rss_bytes() -> int:
    # Linux reports KiB; keep this receipt task-local and scalar.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _tokenizer_ids(value: Any) -> list[int]:
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PreparationError("tokenizer did not return a one-dimensional ID sequence")
    result: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int):
            raise PreparationError("tokenizer returned a non-integer token ID")
        if token < 0 or token >= 128256:
            raise PreparationError(f"tokenizer returned out-of-vocabulary ID {token}")
        result.append(int(token))
    return result


def _text_value(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise PreparationError(f"public row field {key!r} is not text")
    return value


def _finance_fields(row: Mapping[str, Any]) -> tuple[str | None, str, str]:
    system = _text_value(row, "system").strip() or None
    user = _text_value(row, "user").strip()
    assistant = _text_value(row, "assistant").strip()
    if not user:
        instruction = _text_value(row, "instruction").strip()
        input_text = _text_value(row, "input").strip()
        user = instruction + (("\n\n" + input_text) if input_text else "")
    if not assistant:
        assistant = _text_value(row, "output").strip()
    return system, user, assistant


def _candidate_from_row(style: str, row_index: int, row: Mapping[str, Any], tokenizer: Any) -> "PublicCandidate | None":
    revision = STYLE_REVISION[style]
    if style == "pile_plain":
        text = _text_value(row, "text")
        source_hash = sha256_bytes(text.encode("utf-8"))
        token_ids = [BOS_TOKEN_ID, *_tokenizer_ids(tokenizer(text, add_special_tokens=False))]
        if len(token_ids) < 2:
            return None
        record_id = f"pile10k-{row_index:05d}-{source_hash[:16]}"
        rendered_char_count = len(text)
    elif style == "finance_chat":
        system, user, assistant = _finance_fields(row)
        if not user or not assistant:
            return None
        content = json.dumps(
            [system, user, assistant], ensure_ascii=False, separators=(",", ":")
        )
        source_hash = sha256_bytes(content.encode("utf-8"))
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
        try:
            tokenized = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                date_string="06 Aug 2026",
            )
        except Exception as exc:
            raise PreparationError(f"Finance row {row_index} chat rendering failed") from exc
        token_ids = _tokenizer_ids(tokenized)
        if not token_ids or token_ids[0] != BOS_TOKEN_ID:
            raise PreparationError(f"Finance row {row_index} lost the declared BOS token")
        record_id = f"finance-public-{row_index:06d}-{source_hash[:16]}"
        rendered_char_count = len(content)
    elif style == "alpaca_instruction":
        from token_reconstruction.alpaca_split import historical_rendered_text

        rendered = historical_rendered_text(row, tokenizer)
        source_hash = sha256_bytes(rendered.encode("utf-8"))
        token_ids = _tokenizer_ids(tokenizer(rendered, add_special_tokens=False))
        if not token_ids or token_ids[0] != BOS_TOKEN_ID:
            raise PreparationError(f"Alpaca row {row_index} lost the declared BOS token")
        record_id = (
            f"tatsu-lab/alpaca/train@{revision}:row-{row_index:05d}"
        )
        rendered_char_count = len(rendered)
    else:  # pragma: no cover - constants are validated by callers
        raise PreparationError(f"unknown public style: {style}")
    return PublicCandidate(
        style=style,
        dataset_id=STYLE_DATASET[style],
        dataset_revision=revision,
        row_index=row_index,
        record_id=record_id,
        public_record_sha256=source_hash,
        truncated_sequence_sha256=sequence_sha256(token_ids[: 1 + MAX_PANEL_POST_BOS]),
        rendered_char_count=rendered_char_count,
        full_token_count=len(token_ids),
        post_bos_token_count=len(token_ids) - 1,
    )


@dataclass(frozen=True)
class PublicCandidate:
    style: str
    dataset_id: str
    dataset_revision: str
    row_index: int
    record_id: str
    public_record_sha256: str
    truncated_sequence_sha256: str
    rendered_char_count: int
    full_token_count: int
    post_bos_token_count: int

    def metadata(self, *, pool: str, length_stratum: int | None = None, anchor: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "pool": pool,
            "style": self.style,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "record_id": self.record_id,
            "row_index": self.row_index,
            "public_record_sha256": self.public_record_sha256,
            "truncated_sequence_sha256": self.truncated_sequence_sha256,
            "rendered_char_count": self.rendered_char_count,
            "full_token_count": self.full_token_count,
            "post_bos_token_count": self.post_bos_token_count,
        }
        if length_stratum is not None:
            row["length_stratum"] = length_stratum
            row["anchor"] = bool(anchor)
        return row


@dataclass
class Exclusions:
    ids: set[str]
    hashes: set[str]
    indices: dict[str, set[int]]
    sources: list[dict[str, Any]] = field(default_factory=list)


_SENSITIVE_KEYS = {
    "token_ids",
    "input_ids",
    "labels",
    "source_text",
    "truth",
    "oracle",
    "target_weights",
}
_HASH_KEYS = {
    "public_record_sha256",
    "rendered_sha256",
    "text_sha256",
    "content_sha256",
    "record_hash",
    "sequence_sha256",
    "truncated_sequence_sha256",
}
_INDEX_KEYS = {"row_index", "raw_index", "source_index", "dataset_index", "index"}


def _style_hint(value: Any) -> str | None:
    lowered = str(value or "").casefold()
    if "pile" in lowered:
        return "pile_plain"
    if "finance" in lowered:
        return "finance_chat"
    if "alpaca" in lowered:
        return "alpaca_instruction"
    return None


def _id_style(value: str) -> str | None:
    if value.startswith("pile10k-"):
        return "pile_plain"
    if value.startswith("finance-public-"):
        return "finance_chat"
    if value.startswith("tatsu-lab/alpaca/") or value.startswith("alpaca-public-"):
        return "alpaca_instruction"
    return None


def _scan_exclusion(value: Any, *, hint: str | None, result: Exclusions) -> None:
    if isinstance(value, Mapping):
        local_style = _style_hint(value.get("dataset")) or _style_hint(value.get("style")) or _style_hint(hint)
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            lowered = key.casefold().replace("-", "_")
            if lowered in _SENSITIVE_KEYS or any(fragment in lowered for fragment in _SENSITIVE_KEYS):
                continue
            if lowered == "record_id" and isinstance(child, str):
                style = _id_style(child) or local_style
                if style is not None:
                    result.ids.add(child)
            elif lowered in _HASH_KEYS and isinstance(child, str) and len(child) == 64:
                try:
                    int(child, 16)
                except ValueError:
                    pass
                else:
                    result.hashes.add(child)
            elif lowered in _INDEX_KEYS and isinstance(child, int) and not isinstance(child, bool):
                if local_style is not None and child >= 0:
                    result.indices.setdefault(local_style, set()).add(int(child))
            _scan_exclusion(child, hint=local_style or hint, result=result)
    elif isinstance(value, list):
        for child in value:
            _scan_exclusion(child, hint=hint, result=result)


def collect_exclusions(paths: Sequence[Path]) -> Exclusions:
    result = Exclusions(ids=set(), hashes=set(), indices={}, sources=[])
    for path in paths:
        resolved = path.expanduser().resolve()
        descriptor: dict[str, Any] = {
            "path": str(path.expanduser()),
            "resolved_path": str(resolved),
            "available": False,
            "known_exact_ids": False,
        }
        if not resolved.is_file() or resolved.is_symlink():
            result.sources.append(descriptor)
            continue
        descriptor.update(
            {
                "available": True,
                "bytes": int(resolved.stat().st_size),
                "sha256": sha256_file(resolved),
            }
        )
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationError(f"public exclusion metadata is invalid: {resolved}") from exc
        before = len(result.ids)
        _scan_exclusion(value, hint=str(resolved), result=result)
        descriptor["known_exact_ids"] = len(result.ids) > before
        descriptor["cumulative_identity_counts"] = {
            "ids": len(result.ids),
            "hashes": len(result.hashes),
            "pile_indices": len(result.indices.get("pile_plain", set())),
            "finance_indices": len(result.indices.get("finance_chat", set())),
            "alpaca_indices": len(result.indices.get("alpaca_instruction", set())),
        }
        result.sources.append(descriptor)
    return result


def _source_descriptor(style: str, paths: Sequence[Path]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for original in paths:
        path = original.expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise PreparationError(f"public Arrow source is unavailable: {original}")
        file_row: dict[str, Any] = {
            "path": str(original.expanduser()),
            "resolved_path": str(path),
            "bytes": int(path.stat().st_size),
            "hash_status": "pinned_parent_metadata_if_available",
        }
        known = STYLE_SOURCE_HASHES.get(path.name)
        if known is not None:
            file_row["sha256"] = known
        else:
            file_row["sha256"] = None
        files.append(file_row)
    return {
        "id": STYLE_DATASET[style],
        "split": "train",
        "revision": STYLE_REVISION[style],
        "arrow_files": files,
        "large_source_hashing_by_selector": False,
    }


def _tokenizer_descriptor(tokenizer_path: Path) -> dict[str, Any]:
    path = tokenizer_path.expanduser().resolve()
    if not path.is_dir() or path.is_symlink():
        raise PreparationError(f"tokenizer snapshot is unavailable: {tokenizer_path}")
    files: list[dict[str, Any]] = []
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        candidate = path / name
        if candidate.is_file() and not candidate.is_symlink():
            files.append(
                {
                    "name": name,
                    "bytes": int(candidate.stat().st_size),
                    "sha256": sha256_file(candidate),
                }
            )
    if not files:
        raise PreparationError("tokenizer snapshot has no local metadata files")
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot": str(path),
        "bos_token_id": BOS_TOKEN_ID,
        "padding_token_id": PAD_TOKEN_ID,
        "files": files,
    }


def _load_arrow(paths: Sequence[Path]) -> Any:
    try:
        from datasets import Dataset, concatenate_datasets
    except Exception as exc:  # pragma: no cover - environment dependency
        raise PreparationError("datasets is required for public Arrow selection") from exc
    datasets = []
    for original in paths:
        path = original.expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise PreparationError(f"public Arrow source is unavailable: {original}")
        try:
            datasets.append(Dataset.from_file(str(path)))
        except Exception as exc:
            raise PreparationError(f"unable to open public Arrow source: {path}") from exc
    if not datasets:
        raise PreparationError("at least one public Arrow source is required")
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def _fit_replay_descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PreparationError(f"PR7 fit-record metadata is unavailable: {path}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError("PR7 fit-record metadata is invalid JSON") from exc
    rows = value.get("records") if isinstance(value, Mapping) else None
    if not isinstance(rows, list) or len(rows) != FIT_RECORD_COUNT:
        raise PreparationError(f"PR7 fit replay must contain exactly {FIT_RECORD_COUNT} records")
    ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str):
            raise PreparationError(f"PR7 fit replay row {index} has no record_id")
        ids.append(str(row["record_id"]))
    if len(set(ids)) != len(ids):
        raise PreparationError("PR7 fit replay record IDs are not distinct")
    return {
        "role": "immutable PR7 public fit/replay pool",
        "record_count": len(ids),
        "record_ids_sha256": digest_lines(ids),
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
        "labels_are_public": True,
        "target_condition": "public_base",
    }


def _excluded(candidate: PublicCandidate, exclusions: Exclusions, seen_hashes: set[str], seen_sequences: set[str]) -> str | None:
    if candidate.record_id in exclusions.ids:
        return "explicit_record_id"
    if candidate.public_record_sha256 in exclusions.hashes:
        return "explicit_content_hash"
    if candidate.row_index in exclusions.indices.get(candidate.style, set()):
        return "explicit_source_index"
    if candidate.public_record_sha256 in seen_hashes:
        return "cross_pool_content_hash"
    if candidate.truncated_sequence_sha256 in seen_sequences:
        return "cross_pool_truncated_sequence_hash"
    return None


def _select_style(
    style: str,
    dataset: Any,
    tokenizer: Any,
    *,
    seed: int,
    exclusions: Exclusions,
    seen_hashes: set[str],
    seen_sequences: set[str],
    correction_count: int,
    validation_count: int,
    panel_count: int,
) -> tuple[dict[str, list[PublicCandidate]], dict[str, int]]:
    required = correction_count + validation_count + panel_count
    accepted: list[PublicCandidate] = []
    skipped = {
        "invalid": 0,
        "too_short": 0,
        "excluded": 0,
        "duplicate_content": 0,
        "duplicate_sequence": 0,
    }
    for row_index in seeded_indices(len(dataset), style=style, seed=seed):
        row = dataset[row_index]
        if not isinstance(row, Mapping):
            skipped["invalid"] += 1
            continue
        try:
            candidate = _candidate_from_row(style, row_index, row, tokenizer)
        except PreparationError:
            skipped["invalid"] += 1
            continue
        if candidate is None:
            skipped["invalid"] += 1
            continue
        if candidate.post_bos_token_count < MAX_PANEL_POST_BOS:
            skipped["too_short"] += 1
            continue
        reason = _excluded(candidate, exclusions, seen_hashes, seen_sequences)
        if reason is not None:
            skipped["excluded"] += int(reason.startswith("explicit_"))
            skipped["duplicate_content"] += int(reason == "cross_pool_content_hash")
            skipped["duplicate_sequence"] += int(reason == "cross_pool_truncated_sequence_hash")
            continue
        accepted.append(candidate)
        seen_hashes.add(candidate.public_record_sha256)
        seen_sequences.add(candidate.truncated_sequence_sha256)
        if len(accepted) >= required:
            break
    if len(accepted) != required:
        raise PreparationError(f"{style} has only {len(accepted)} eligible records; need {required}")
    return {
        "correction": accepted[:correction_count],
        "validation": accepted[correction_count : correction_count + validation_count],
        "panel": accepted[correction_count + validation_count :],
    }, skipped


def _panel_rows(records: Sequence[PublicCandidate]) -> tuple[list[dict[str, Any]], list[str]]:
    if len(records) != PANEL_RECORDS_PER_STYLE:
        raise PreparationError("panel style quota changed")
    result: list[dict[str, Any]] = []
    anchors: list[str] = []
    for index, candidate in enumerate(records):
        length = PANEL_LENGTHS[index // RECORDS_PER_LENGTH]
        anchor = length == 32 and index % RECORDS_PER_LENGTH < ANCHOR_RECORDS_PER_STYLE
        result.append(candidate.metadata(pool="fresh_evaluation", length_stratum=length, anchor=anchor))
        if anchor:
            anchors.append(candidate.record_id)
    return result, anchors


def build_selection(
    *,
    datasets: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_path: Path,
    source_paths: Mapping[str, Sequence[Path]],
    fit_records_path: Path,
    exclusion_paths: Sequence[Path],
    selection_seed: int,
    argv: Sequence[str],
    root: Path,
) -> dict[str, Any]:
    if tuple(datasets) != STYLES:
        raise PreparationError("datasets must contain the three fixed P04 styles in order")
    exclusions = collect_exclusions(exclusion_paths)
    fit = _fit_replay_descriptor(fit_records_path)
    # Fit IDs are checked directly even when the exclusion scanner encounters a
    # legacy metadata shape that omits a style hint.
    fit_value = json.loads(fit_records_path.expanduser().resolve().read_text(encoding="utf-8"))
    fit_ids = {str(row["record_id"]) for row in fit_value["records"]}
    exclusions.ids.update(fit_ids)
    seen_hashes: set[str] = set()
    seen_sequences: set[str] = set()
    style_counts = {
        "correction": balanced_counts(CORRECTION_RECORD_COUNT, len(STYLES)),
        "validation": balanced_counts(VALIDATION_RECORD_COUNT, len(STYLES)),
        "panel": balanced_counts(PANEL_RECORD_COUNT, len(STYLES)),
    }
    pools: dict[str, list[dict[str, Any]]] = {"correction": [], "validation": [], "panel": []}
    skipped_by_style: dict[str, dict[str, int]] = {}
    anchors: list[str] = []
    for style_index, style in enumerate(STYLES):
        selected, skipped = _select_style(
            style,
            datasets[style],
            tokenizer,
            seed=selection_seed,
            exclusions=exclusions,
            seen_hashes=seen_hashes,
            seen_sequences=seen_sequences,
            correction_count=style_counts["correction"][style_index],
            validation_count=style_counts["validation"][style_index],
            panel_count=style_counts["panel"][style_index],
        )
        skipped_by_style[style] = skipped
        pools["correction"].extend(
            candidate.metadata(pool="public_correction") for candidate in selected["correction"]
        )
        pools["validation"].extend(
            candidate.metadata(pool="public_validation") for candidate in selected["validation"]
        )
        panel_rows, style_anchors = _panel_rows(selected["panel"])
        pools["panel"].extend(panel_rows)
        anchors.extend(style_anchors)
    if len(pools["correction"]) != CORRECTION_RECORD_COUNT:
        raise PreparationError("correction quota mismatch")
    if len(pools["validation"]) != VALIDATION_RECORD_COUNT:
        raise PreparationError("validation quota mismatch")
    if len(pools["panel"]) != PANEL_RECORD_COUNT:
        raise PreparationError("panel quota mismatch")
    all_ids = [row["record_id"] for pool in pools.values() for row in pool]
    if len(set(all_ids)) != len(all_ids):
        raise PreparationError("public correction, validation, and panel pools overlap")
    panel_by_style_length = {
        style: {
            str(length): [
                row["record_id"]
                for row in pools["panel"]
                if row["style"] == style and row["length_stratum"] == length
            ]
            for length in PANEL_LENGTHS
        }
        for style in STYLES
    }
    panel_ids = [row["record_id"] for row in pools["panel"]]
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_SELECTION_READY_NO_MODEL_NO_EVALUATION_TRUTH",
        "created_utc": _utc_now(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "cut_depth": 4,
            "hidden_size": 2048,
        },
        "tokenizer": _tokenizer_descriptor(tokenizer_path),
        "selection": {
            "seed": selection_seed,
            "order": "sha256(TRR-P04|style|row:index|seed), then row index",
            "minimum_post_bos_tokens": MAX_PANEL_POST_BOS,
            "cross_pool_deduplicate": ["public_record_sha256", "truncated_sequence_sha256"],
            "source_text_or_token_ids_written": False,
            "selection_adapts_to_scores": False,
            "style_order": list(STYLES),
            "panel_lengths_post_bos": list(PANEL_LENGTHS),
            "records_per_style_length": RECORDS_PER_LENGTH,
            "anchor_rule": "first four records in each style's 32-token panel cell",
        },
        "sources": {
            style: _source_descriptor(style, source_paths[style]) for style in STYLES
        },
        "pools": {
            "fit_replay": fit,
            "correction": {
                "role": "separate public correction-training pool",
                "record_count": len(pools["correction"]),
                "records": pools["correction"],
                "labels_public": True,
                "target_condition": "public_base",
            },
            "validation": {
                "role": "public-only development pool",
                "record_count": len(pools["validation"]),
                "records": pools["validation"],
                "labels_public": True,
                "target_condition": "public_base",
            },
            "fresh_evaluation": {
                "role": "fresh evaluator-side panel index",
                "record_count": len(pools["panel"]),
                "independent_source_records": len(pools["panel"]),
                "records": pools["panel"],
                "target_conditions": ["public_base", "p04_evaluator_target_update_v1"],
                "teacher_access": False,
                "student_training_access": False,
            },
        },
        "panel": {
            "record_count": len(pools["panel"]),
            "independent_source_records": len(pools["panel"]),
            "paired_target_conditions": True,
            "records_by_style_length": panel_by_style_length,
            "anchor_record_ids": anchors,
            "anchor_record_count": len(anchors),
            "anchor_scored_positions_per_target": len(anchors) * 32,
            "anchor_denominator_separate": True,
            "bootstrap_cluster": "source_record_id",
        },
        "teacher_evidence_plan": {
            "candidate_budget": 32,
            "proposal_budget": 32,
            "difficult_positions": 256,
            "random_audit_positions": 128,
            "total_positions": 384,
            "source_pool": "correction",
            "target_condition": "public_base",
            "generated": False,
        },
        "training_seeds": [1737, 2711],
        "exclusions": {
            "sources": exclusions.sources,
            "identity_counts": {
                "record_ids": len(exclusions.ids),
                "content_hashes": len(exclusions.hashes),
                "source_indices": sum(len(values) for values in exclusions.indices.values()),
            },
            "fit_replay_ids_added": len(fit_ids),
            "rule": "skip whole candidate source record on any exact public identity/hash/index collision",
        },
        "execution": {
            "argv": list(argv),
            "python": sys.executable,
            "started_utc": None,
            "ended_utc": None,
            "model_loaded": False,
            "evaluation_truth_accessed": False,
            "network_used": False,
            "safe_environment": _safe_environment(),
            "max_rss_bytes": None,
            "large_source_hashing": False,
            "source_hashes_are_parent_metadata_or_caller_supplied": True,
            "root": str(root),
        },
        "skipped_candidates": skipped_by_style,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path(TOKENIZER_SNAPSHOT))
    parser.add_argument("--pile-arrow", type=Path, action="append")
    parser.add_argument("--finance-arrow", type=Path, action="append")
    parser.add_argument("--alpaca-arrow", type=Path, action="append")
    parser.add_argument("--fit-records", type=Path)
    parser.add_argument("--exclusion", type=Path, action="append")
    parser.add_argument("--selection-seed", type=int, default=20260906)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    started = time.monotonic()
    started_utc = _utc_now()
    source_paths: dict[str, Sequence[Path]] = {
        "pile_plain": tuple(args.pile_arrow or (Path(value) for value in DEFAULT_ARROWS["pile_plain"])),
        "finance_chat": tuple(args.finance_arrow or (Path(value) for value in DEFAULT_ARROWS["finance_chat"])),
        "alpaca_instruction": tuple(args.alpaca_arrow or (Path(value) for value in DEFAULT_ARROWS["alpaca_instruction"])),
    }
    fit_records_path = args.fit_records or (root / "experiments/TRR-0004/fit/affine_fit_records.json")
    exclusion_paths = args.exclusion or [root / value for value in DEFAULT_EXCLUSIONS]
    try:
        from transformers import AutoTokenizer

        tokenizer_path = args.tokenizer.expanduser().resolve()
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, use_fast=True
        )
        if getattr(tokenizer, "bos_token_id", None) != BOS_TOKEN_ID:
            raise PreparationError("tokenizer BOS ID differs from the pinned public ID")
        datasets = {
            style: _load_arrow(source_paths[style]) for style in STYLES
        }
        result = build_selection(
            datasets=datasets,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            source_paths=source_paths,
            fit_records_path=fit_records_path,
            exclusion_paths=exclusion_paths,
            selection_seed=args.selection_seed,
            argv=list(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]),
            root=root,
        )
        result["execution"].update(
            {
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "max_rss_bytes": _max_rss_bytes(),
            }
        )
        output = args.output.expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise PreparationError(f"refusing to overwrite create-only output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except PreparationError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "fit_replay_records": result["pools"]["fit_replay"]["record_count"],
                "correction_records": result["pools"]["correction"]["record_count"],
                "validation_records": result["pools"]["validation"]["record_count"],
                "panel_records": result["panel"]["record_count"],
                "anchor_records": result["panel"]["anchor_record_count"],
                "model_loaded": result["execution"]["model_loaded"],
                "evaluation_truth_accessed": result["execution"]["evaluation_truth_accessed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

