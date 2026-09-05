#!/usr/bin/env python3
"""Produce the public inputs for the TRR-0004 fresh confirmation.

This command is intentionally a small producer.  It promotes the already
registered public selection rule, captures the four paired public-prefix
observation cells, prepares the separate public-label sidecar, and creates
the five-method registration.  Prediction, freezing, scoring, and plotting
belong to the later adapters and are deliberately absent here.

The ``select`` and ``capture`` subcommands never open evaluator truth.  The
``truth`` subcommand is a separate preparation-role command: it reads only the
selected public rows and writes the token-label sidecar outside the checkout.
The sidecar is not loaded back by this producer.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors.torch import save_file

import trr0004_fresh_confirmation as fc
from token_reconstruction.public_activation import (
    capture_public_prefix,
    pad_public_token_sequences,
    tensor_sha256,
)
from token_reconstruction.public_prefix import ContiguousPublicPrefix
from token_reconstruction.target_update import (
    TargetLoRAConfig,
    install_target_lora,
    load_target_lora,
)


TASK_ID = "TRR-0004"
PRODUCER_SCHEMA = "token-reconstruction.trr0004-confirmation-producer.v1"
SELECTION_SCHEMA = fc.PLAN_SCHEMA
CAPTURE_EVIDENCE_SCHEMA = "token-reconstruction.trr0004-confirmation-capture.v1"
TRUTH_PREPARATION_SCHEMA = "token-reconstruction.trr0004-confirmation-truth-preparation.v1"
METHOD_FREEZE_STATUSES = {
    "FROZEN_METHOD_REGISTRATION",
    "FROZEN_METHOD_STATES",
    "FROZEN",
    "ACTIVE_METHODS_FROZEN",
}
CAPTURE_BATCH_SIZE = 8
CAPTURE_SEQUENCE_TOKENS = 192
MIN_FREE_GPU_BYTES = 8 * 1024**3
MAX_RESERVED_GPU_BYTES = 8 * 1024**3
MAX_HOST_RSS_BYTES = 16 * 1024**3
DATE_STRING = "06 Aug 2026"
FINANCE_DATASET_FINGERPRINT = "4abbac8acaab4205"
PILE_DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
FINANCE_DATASET_ID = "Josephgflowers/Finance-Instruct-500k"
PILE_DATASET_ID = "NeelNanda/pile-10k"
ALPACA_DATASET_ID = "tatsu-lab/alpaca"


class ProducerError(fc.ConfirmationError):
    """Raised when public confirmation inputs cannot be produced safely."""


@dataclass(frozen=True)
class RenderedRecord:
    """A transient public record; token IDs never enter a written panel."""

    style: str
    raw_index: int
    record_id: str
    public_record_sha256: str
    token_ids: tuple[int, ...]
    full_token_count: int
    post_bos_token_count: int
    truncated_sequence_sha256: str

    def selection_metadata(self, *, sequence_tokens: int) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "public_record_sha256": self.public_record_sha256,
            "raw_index": self.raw_index,
            "source_index": self.raw_index,
            "valid_tokens": min(self.full_token_count, sequence_tokens),
            "full_token_count": self.full_token_count,
            "post_bos_token_count": self.post_bos_token_count,
            "truncated_sequence_sha256": self.truncated_sequence_sha256,
        }

    def panel_metadata(self, *, sequence_tokens: int) -> dict[str, Any]:
        # Keep this set in lockstep with fresh_confirmation._record_metadata.
        return {
            "record_id": self.record_id,
            "public_record_sha256": self.public_record_sha256,
            "raw_index": self.raw_index,
            "source_index": self.raw_index,
            "valid_tokens": min(self.full_token_count, sequence_tokens),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ProducerError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProducerError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProducerError(f"{description} must contain an object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser()
    if path.exists() or path.is_symlink():
        raise ProducerError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _require_repo_path(path: Path, root: Path, *, description: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ProducerError(f"{description} is unavailable: {path}")
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ProducerError(f"{description} must be inside the repository: {path}") from exc
    return path


def _require_external_file(path: Path, *, description: str) -> Path:
    original = path.expanduser()
    if original.is_symlink() or not original.is_file():
        raise ProducerError(f"{description} must be a regular file: {original}")
    return original.resolve()


def _require_public_asset_file(path: Path, *, description: str) -> Path:
    """Resolve a public cache link to its regular immutable blob."""

    original = path.expanduser()
    resolved = original.resolve()
    if not resolved.is_file():
        raise ProducerError(f"{description} is unavailable: {original}")
    return resolved


def _require_external_destination(path: Path, *, description: str) -> Path:
    """Return a new external file path without creating or following it."""

    original = path.expanduser()
    if original.exists() or original.is_symlink():
        raise ProducerError(f"{description} is create-only and already exists: {original}")
    original.parent.mkdir(parents=True, exist_ok=True)
    return original.resolve()


def _create_output_root(path: Path, root: Path, *, description: str) -> Path:
    """Create a versioned output root without following an existing link."""

    path = path.expanduser()
    if path.exists() or path.is_symlink():
        raise ProducerError(f"{description} is create-only and already exists: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ProducerError(f"{description} must be inside the repository: {path}") from exc
    path.mkdir(parents=True)
    return path.resolve()


def _device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise ProducerError("CUDA was requested but is unavailable")
    return torch.device(raw)


def _public_padding_id(tokenizer: Any) -> int:
    """Use the declared EOT fallback without mutating the tokenizer."""

    configured = getattr(tokenizer, "pad_token_id", None)
    if configured is not None:
        if int(configured) != fc.PAD_TOKEN_ID:
            raise ProducerError("tokenizer padding ID differs from declared public ID")
        return fc.PAD_TOKEN_ID
    try:
        resolved = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        token = tokenizer.convert_ids_to_tokens(fc.PAD_TOKEN_ID)
    except Exception as exc:
        raise ProducerError("tokenizer has no declared public padding fallback") from exc
    if resolved != fc.PAD_TOKEN_ID or token != "<|end_of_text|>":
        raise ProducerError("tokenizer has no declared public padding fallback")
    return fc.PAD_TOKEN_ID


def _tokenizer_descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ProducerError(f"public tokenizer snapshot is unavailable: {path}")
    files: dict[str, Any] = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
        candidate = path / name
        resolved = candidate.resolve()
        if not resolved.is_file():
            continue
        files[name] = {
            "path": str(candidate),
            "resolved_path": str(resolved),
            "bytes": int(resolved.stat().st_size),
            "sha256": _sha256_file(resolved),
            "symlink": candidate.is_symlink(),
        }
    if not files:
        raise ProducerError("public tokenizer snapshot has no readable metadata files")
    return {"path": str(path), "files": files}


def _load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_dir():
        raise ProducerError(f"public tokenizer snapshot is unavailable: {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if getattr(tokenizer, "bos_token_id", None) != fc.BOS_TOKEN_ID:
        raise ProducerError("public tokenizer BOS ID changed")
    _public_padding_id(tokenizer)
    return tokenizer


def _tokenizer_ids(output: Any) -> list[int]:
    if hasattr(output, "keys") and "input_ids" in output:
        output = output["input_ids"]
    elif hasattr(output, "input_ids"):
        output = output.input_ids
    if hasattr(output, "tolist"):
        output = output.tolist()
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise ProducerError("tokenizer did not return one-dimensional input IDs")
    result: list[int] = []
    for value in output:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProducerError("tokenizer returned a non-integer token ID")
        if value < 0 or value >= fc.VOCAB_SIZE:
            raise ProducerError("tokenizer returned an out-of-vocabulary token ID")
        result.append(int(value))
    return result


def _sequence_sha256(token_ids: Sequence[int]) -> str:
    values = torch.tensor(list(token_ids), dtype=torch.int32).numpy().tobytes(order="C")
    return _sha256_bytes(values)


def _text_value(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProducerError(f"public row field {key!r} is not text")
    return value


def _render_pile(row: Mapping[str, Any], index: int, tokenizer: Any) -> RenderedRecord:
    text = _text_value(row, "text")
    raw_hash = _sha256_bytes(text.encode("utf-8"))
    ids = [fc.BOS_TOKEN_ID, *_tokenizer_ids(tokenizer(text, add_special_tokens=False))]
    if len(ids) < 2:
        raise ProducerError(f"Pile row {index} has no current token")
    record_id = f"pile10k-{index:05d}-{raw_hash[:16]}"
    truncated = ids[: fc.SEQUENCE_TOKENS["pile"]]
    return RenderedRecord(
        style="pile",
        raw_index=index,
        record_id=record_id,
        public_record_sha256=raw_hash,
        token_ids=tuple(ids),
        full_token_count=len(ids),
        post_bos_token_count=len(ids) - 1,
        truncated_sequence_sha256=_sequence_sha256(truncated),
    )


def _finance_fields(row: Mapping[str, Any]) -> tuple[str | None, str, str]:
    # Finance-Instruct's reviewed public renderer uses system/user/assistant.
    # The fallback names support equivalent local Arrow exports without
    # changing the rendering rule.
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


def _render_finance(row: Mapping[str, Any], index: int, tokenizer: Any) -> RenderedRecord | None:
    system, user, assistant = _finance_fields(row)
    if not user or not assistant:
        return None
    content_hash = _sha256_bytes(
        json.dumps([system, user, assistant], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
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
        ids = _tokenizer_ids(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                date_string=DATE_STRING,
            )
        )
    except Exception as exc:
        raise ProducerError(f"Finance row {index} chat-template rendering failed") from exc
    if not ids or ids[0] != fc.BOS_TOKEN_ID:
        raise ProducerError(f"Finance row {index} lost the declared BOS token")
    record_id = f"finance-public-{index:06d}-{content_hash[:16]}"
    truncated = ids[: fc.SEQUENCE_TOKENS["finance"]]
    return RenderedRecord(
        style="finance",
        raw_index=index,
        record_id=record_id,
        public_record_sha256=content_hash,
        token_ids=tuple(ids),
        full_token_count=len(ids),
        post_bos_token_count=len(ids) - 1,
        truncated_sequence_sha256=_sequence_sha256(truncated),
    )


def _load_arrow_dataset(paths: Sequence[Path]) -> Any:
    from datasets import Dataset, concatenate_datasets

    if not paths:
        raise ProducerError("at least one public Arrow path is required")
    datasets = []
    for path in paths:
        path = _require_external_file(path, description="public Arrow cache")
        try:
            datasets.append(Dataset.from_file(str(path)))
        except Exception as exc:
            raise ProducerError(f"unable to load public Arrow cache: {path}") from exc
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def _dataset_descriptor(paths: Sequence[Path], *, dataset_id: str, revision: str | None = None, fingerprint: str | None = None) -> dict[str, Any]:
    files = []
    for path in paths:
        path = _require_external_file(path, description=f"{dataset_id} Arrow cache")
        files.append({
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        })
    result: dict[str, Any] = {"id": dataset_id, "split": "train", "arrow_files": files}
    if revision is not None:
        result["revision"] = revision
    if fingerprint is not None:
        result["fingerprint"] = fingerprint
    return result


def _validate_frozen_source_binding(
    plan: Mapping[str, Any],
    *,
    pile_descriptor: Mapping[str, Any],
    finance_descriptor: Mapping[str, Any],
    tokenizer_descriptor: Mapping[str, Any],
) -> None:
    frozen = plan.get("public_sources_frozen")
    if not isinstance(frozen, Mapping):
        raise ProducerError("selection plan has no frozen public source descriptors")
    for name, actual in (("pile", pile_descriptor), ("finance", finance_descriptor), ("tokenizer", tokenizer_descriptor)):
        declared = frozen.get(name)
        if not isinstance(declared, Mapping) or dict(declared) != dict(actual):
            raise ProducerError(f"{name} public source differs from the frozen selection input")


def _infer_style(value: Any, *, hint: str | None = None) -> str | None:
    text = (str(hint or "") + " " + str(value or "")).casefold()
    if "finance" in text:
        return "finance"
    if "pile" in text:
        return "pile"
    return None


@dataclass
class ExclusionSets:
    ids: dict[str, set[str]]
    hashes: dict[str, set[str]]
    indices: dict[str, set[int]]
    sources: list[dict[str, Any]]


_SENSITIVE_METADATA_KEYS = {"token_ids", "input_ids", "labels", "source_text", "truth", "oracle"}
_HASH_KEYS = {"public_record_sha256", "text_sha256", "content_sha256", "rendered_sha256"}
_INDEX_KEYS = {"raw_index", "dataset_index", "row_index", "source_index", "index"}


def _scan_exclusion_metadata(value: Any, *, hint: str | None, result: ExclusionSets) -> None:
    """Read only public identity fields from an explicit metadata JSON."""

    if isinstance(value, Mapping):
        local_style = _infer_style(value.get("dataset"), hint=hint)
        if local_style is None:
            local_style = _infer_style(value.get("record_id"), hint=hint)
        record_id = value.get("record_id")
        if isinstance(record_id, str) and record_id:
            style = _infer_style(record_id, hint=local_style)
            if style is not None:
                result.ids[style].add(record_id)
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            lowered = key.casefold().replace("-", "_")
            if lowered in _SENSITIVE_METADATA_KEYS or any(fragment in lowered for fragment in _SENSITIVE_METADATA_KEYS):
                # Do not descend into source/token/truth payloads, even if an
                # old record happens to contain them.
                continue
            if lowered in _HASH_KEYS and isinstance(child, str) and len(child) == 64:
                style = _infer_style(value.get("record_id"), hint=local_style)
                if style is not None:
                    result.hashes[style].add(child)
            if lowered in _INDEX_KEYS and isinstance(child, int) and not isinstance(child, bool):
                style = local_style
                if style is not None and child >= 0:
                    result.indices[style].add(int(child))
            _scan_exclusion_metadata(child, hint=local_style or hint, result=result)
    elif isinstance(value, list):
        for child in value:
            _scan_exclusion_metadata(child, hint=hint, result=result)


def _default_exclusion_paths(root: Path) -> list[Path]:
    """Known public split declarations; absent sources are recorded honestly."""

    source_repo = root.parent.parent if root.parent.name == ".worktrees" else root
    return [
        root / "experiments/TRR-0003/evidence/control/panel.json",
        root / "experiments/TRR-0003/evidence/control/plan.json",
        source_repo / "outputs/TRR-0003/track_b/public_fit_v2/fit_records.json",
        source_repo / "outputs/TRR-0003/track_b/public_validation_slice_v2/public_validation_records.json",
        root / "experiments/TRR-0004/fit/adapter_v2/affine_fit_records.json",
        root / "experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json",
        root / "experiments/TRR-0004/fit/adapter_v2/public_fit_manifest.json",
        root / "experiments/TRR-0004/alpaca_split_plan.json",
        root / "experiments/TRR-0002/configuration-search/public-pile/records.json",
        root / "experiments/TRR-0002/configuration-search/public-finance/records.json",
    ]


def _collect_exclusions(paths: Sequence[Path]) -> ExclusionSets:
    result = ExclusionSets(
        ids={style: set() for style in fc.STYLE_ORDER},
        hashes={style: set() for style in fc.STYLE_ORDER},
        indices={style: set() for style in fc.STYLE_ORDER},
        sources=[],
    )
    for original in paths:
        path = original.expanduser().resolve()
        source: dict[str, Any] = {"path": str(path), "available": False, "known_exact_ids": False}
        if path.is_symlink() or not path.is_file():
            result.sources.append(source)
            continue
        source.update({"available": True, "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)})
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProducerError(f"public exclusion metadata is invalid: {path}") from exc
        before = sum(len(values) for values in result.ids.values()) + sum(len(values) for values in result.hashes.values())
        _scan_exclusion_metadata(value, hint=path.name + " " + str(path.parent), result=result)
        after = sum(len(values) for values in result.ids.values()) + sum(len(values) for values in result.hashes.values())
        source["known_exact_ids"] = after > before
        source["extracted_identity_counts"] = {
            "pile_ids": len(result.ids["pile"]),
            "finance_ids": len(result.ids["finance"]),
            "pile_hashes": len(result.hashes["pile"]),
            "finance_hashes": len(result.hashes["finance"]),
            "pile_indices": len(result.indices["pile"]),
            "finance_indices": len(result.indices["finance"]),
        }
        result.sources.append(source)
    return result


def _excluded(record: RenderedRecord, exclusions: ExclusionSets) -> tuple[bool, str | None]:
    style = record.style
    if record.record_id in exclusions.ids[style]:
        return True, "public_record_id"
    if record.public_record_sha256 in exclusions.hashes[style]:
        return True, "public_record_hash"
    if record.raw_index in exclusions.indices[style]:
        return True, "public_source_index"
    return False, None


def _select_style_records(
    dataset: Any,
    *,
    style: str,
    tokenizer: Any,
    exclusions: ExclusionSets,
    records: int = fc.RECORDS_PER_STYLE,
    seen_truncated_sequences: set[str] | None = None,
) -> tuple[list[RenderedRecord], dict[str, int]]:
    sequence_tokens = fc.SEQUENCE_TOKENS[style]
    minimum_post_bos = sequence_tokens - 1 if style == "pile" else 32
    selected: list[RenderedRecord] = []
    seen_truncated = seen_truncated_sequences if seen_truncated_sequences is not None else set()
    skipped: dict[str, int] = {"too_short": 0, "excluded": 0, "duplicate_truncated_sequence": 0, "invalid": 0}
    for index in range(len(dataset)):
        row = dataset[index]
        if not isinstance(row, Mapping):
            raise ProducerError(f"{style} public row {index} is malformed")
        try:
            candidate = _render_pile(row, index, tokenizer) if style == "pile" else _render_finance(row, index, tokenizer)
        except ProducerError:
            skipped["invalid"] += 1
            continue
        if candidate is None:
            skipped["invalid"] += 1
            continue
        if candidate.post_bos_token_count < minimum_post_bos:
            skipped["too_short"] += 1
            continue
        blocked, reason = _excluded(candidate, exclusions)
        if blocked:
            skipped["excluded"] += 1
            continue
        if candidate.truncated_sequence_sha256 in seen_truncated:
            skipped["duplicate_truncated_sequence"] += 1
            continue
        seen_truncated.add(candidate.truncated_sequence_sha256)
        selected.append(candidate)
        if len(selected) == records:
            break
    if len(selected) != records:
        raise ProducerError(f"{style} public dataset has only {len(selected)} eligible records; need {records}")
    if len({row.record_id for row in selected}) != records:
        raise ProducerError(f"{style} selected public record IDs are not distinct")
    if len({row.truncated_sequence_sha256 for row in selected}) != records:
        raise ProducerError(f"{style} selected truncated token sequences are not distinct")
    return selected, skipped


def _validate_method_freeze(path: Path) -> dict[str, Any]:
    value = _load_json(path, description="method freeze marker")
    method_ids = value.get("method_ids")
    if method_ids is None and isinstance(value.get("methods"), list):
        method_ids = [row.get("id") for row in value["methods"] if isinstance(row, Mapping)]
    if tuple(method_ids or ()) != fc.METHOD_IDS:
        raise ProducerError("method freeze marker does not contain the exact five registered methods")
    status = value.get("status")
    if status not in METHOD_FREEZE_STATUSES:
        raise ProducerError(f"method freeze marker is not frozen: {status!r}")
    return value


def _selection_plan(
    prospective: Mapping[str, Any],
    *,
    root: Path,
    method_freeze_path: Path,
    method_freeze: Mapping[str, Any],
    pile: Sequence[RenderedRecord],
    finance: Sequence[RenderedRecord],
    exclusions: ExclusionSets,
    skipped: Mapping[str, Mapping[str, int]],
    pile_descriptor: Mapping[str, Any],
    finance_descriptor: Mapping[str, Any],
    tokenizer_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    value = json.loads(json.dumps(prospective))
    if value.get("schema") != SELECTION_SCHEMA or value.get("task_id") != TASK_ID:
        raise ProducerError("prospective selection plan identity changed")
    selected = {"pile": list(pile), "finance": list(finance)}
    selected_ids = {style: [row.record_id for row in rows] for style, rows in selected.items()}
    selected_hashes = {
        style: {row.record_id: row.public_record_sha256 for row in rows}
        for style, rows in selected.items()
    }
    selected_metadata = {
        style: [row.selection_metadata(sequence_tokens=fc.SEQUENCE_TOKENS[style]) for row in rows]
        for style, rows in selected.items()
    }
    sequence_hashes = {
        style: [row.truncated_sequence_sha256 for row in rows] for style, rows in selected.items()
    }
    code_commit = _git_commit(root)
    if code_commit is None:
        raise ProducerError("unable to resolve full execution commit")
    value.update(
        {
            "status": "FROZEN_PUBLIC_SELECTION_NO_TRUTH",
            "producer_schema": PRODUCER_SCHEMA,
            "selected_at_utc": _utc_now(),
            "execution": {
                **(value.get("execution") if isinstance(value.get("execution"), Mapping) else {}),
                "git_commit": code_commit,
                "producer": str(Path(__file__).resolve()),
                "python": sys.executable,
                "model_loaded": False,
                "observations_generated": False,
                "truth_opened": False,
                "network_used": False,
            },
            "method_freeze": {
                "record": fc.file_record(method_freeze_path, repository_root=root),
                "status": method_freeze.get("status"),
                "method_ids": list(fc.METHOD_IDS),
                "marker_sha256": _sha256_file(method_freeze_path),
            },
            "public_sources_frozen": {
                "pile": dict(pile_descriptor),
                "finance": dict(finance_descriptor),
                "tokenizer": dict(tokenizer_descriptor),
            },
        }
    )
    selection = value.get("selection_rule")
    if not isinstance(selection, Mapping):
        raise ProducerError("prospective plan selection rule is absent")
    selection = dict(selection)
    selection.update(
        {
            "record_ids_selected": selected_ids,
            "record_hashes_selected": selected_hashes,
            "selected_records": selected_metadata,
            "truncated_sequence_sha256_selected": sequence_hashes,
            "algorithm": (
                "Render each public source in stored row order with the pinned renderer; require the declared "
                "minimum post-BOS length; exclude identities from every explicit known public metadata source; "
                "reject duplicate hashes of the truncated sequence; take the first 16 eligible rows per dataset."
            ),
            "deduplicate_truncated_sequences": True,
            "source_text_or_token_ids_written": False,
            "selection_exclusions": {
                "sources": list(exclusions.sources),
                "identity_counts": {
                    "pile_ids": len(exclusions.ids["pile"]),
                    "finance_ids": len(exclusions.ids["finance"]),
                    "pile_hashes": len(exclusions.hashes["pile"]),
                    "finance_hashes": len(exclusions.hashes["finance"]),
                    "pile_indices": len(exclusions.indices["pile"]),
                    "finance_indices": len(exclusions.indices["finance"]),
                },
                "unknown_historical_a1_ids_do_not_block": True,
            },
            "eligibility_diagnostics": {style: dict(values) for style, values in skipped.items()},
        }
    )
    value["selection_rule"] = selection
    value["selection_summary"] = {
        "records_per_style": fc.RECORDS_PER_STYLE,
        "styles": list(fc.STYLE_ORDER),
        "distinct_records_total": 2 * fc.RECORDS_PER_STYLE,
        "paired_conditions": list(fc.CONDITION_ORDER),
        "selected_record_ids_sha256": _json_sha256(selected_ids),
        "selected_record_hashes_sha256": _json_sha256(selected_hashes),
        "selected_sequence_hashes_sha256": _json_sha256(sequence_hashes),
    }
    return value


def select_public(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    prospective_path = _require_repo_path(args.prospective_plan, root, description="prospective confirmation plan")
    method_freeze_path = _require_repo_path(args.method_freeze, root, description="method freeze marker")
    prospective = _load_json(prospective_path, description="prospective confirmation plan")
    method_freeze = _validate_method_freeze(method_freeze_path)
    tokenizer = _load_tokenizer(args.tokenizer)
    pile_paths = tuple(args.pile_arrow or ())
    finance_paths = tuple(args.finance_arrow or ())
    pile_dataset = _load_arrow_dataset(pile_paths)
    finance_dataset = _load_arrow_dataset(finance_paths)
    exclusion_paths_list = list(_default_exclusion_paths(root))
    for candidate in args.exclude_source or ():
        if candidate.expanduser().resolve() not in {value.expanduser().resolve() for value in exclusion_paths_list}:
            exclusion_paths_list.append(candidate)
    exclusion_paths = tuple(exclusion_paths_list)
    exclusions = _collect_exclusions(exclusion_paths)
    seen_truncated_sequences: set[str] = set()
    pile, pile_skipped = _select_style_records(
        pile_dataset,
        style="pile",
        tokenizer=tokenizer,
        exclusions=exclusions,
        seen_truncated_sequences=seen_truncated_sequences,
    )
    finance, finance_skipped = _select_style_records(
        finance_dataset,
        style="finance",
        tokenizer=tokenizer,
        exclusions=exclusions,
        seen_truncated_sequences=seen_truncated_sequences,
    )
    pile_descriptor = _dataset_descriptor(pile_paths, dataset_id=PILE_DATASET_ID, revision=PILE_DATASET_REVISION)
    finance_descriptor = _dataset_descriptor(
        finance_paths, dataset_id=FINANCE_DATASET_ID, fingerprint=FINANCE_DATASET_FINGERPRINT
    )
    tokenizer_descriptor = _tokenizer_descriptor(args.tokenizer)
    value = _selection_plan(
        prospective,
        root=root,
        method_freeze_path=method_freeze_path,
        method_freeze=method_freeze,
        pile=pile,
        finance=finance,
        exclusions=exclusions,
        skipped={"pile": pile_skipped, "finance": finance_skipped},
        pile_descriptor=pile_descriptor,
        finance_descriptor=finance_descriptor,
        tokenizer_descriptor=tokenizer_descriptor,
    )
    output = args.output.expanduser().resolve()
    _write_create_only(output, value)
    return {
        "task_id": TASK_ID,
        "status": value["status"],
        "selection_plan": str(output),
        "selection_plan_sha256": _sha256_file(output),
        "record_counts": {style: len(rows) for style, rows in (("pile", pile), ("finance", finance))},
        "truth_opened": False,
    }


def _selected_metadata(plan: Mapping[str, Any], style: str) -> list[Mapping[str, Any]]:
    selection = plan.get("selection_rule")
    if not isinstance(selection, Mapping):
        raise ProducerError("selection plan rule is absent")
    rows = selection.get("selected_records")
    if not isinstance(rows, Mapping) or not isinstance(rows.get(style), list):
        raise ProducerError(f"selection plan has no frozen {style} record metadata")
    selected = [row for row in rows[style] if isinstance(row, Mapping)]
    if len(selected) != fc.RECORDS_PER_STYLE:
        raise ProducerError(f"selection plan has wrong {style} record count")
    return selected


def _materialize_selected(
    plan: Mapping[str, Any],
    *,
    style: str,
    dataset: Any,
    tokenizer: Any,
) -> list[RenderedRecord]:
    expected = _selected_metadata(plan, style)
    result: list[RenderedRecord] = []
    by_index = {int(row["raw_index"]): row for row in expected if isinstance(row.get("raw_index"), int)}
    if len(by_index) != fc.RECORDS_PER_STYLE:
        raise ProducerError(f"selection plan {style} source indices are incomplete")
    for index in sorted(by_index):
        row = dataset[index]
        if not isinstance(row, Mapping):
            raise ProducerError(f"{style} selected public row {index} is malformed")
        candidate = _render_pile(row, index, tokenizer) if style == "pile" else _render_finance(row, index, tokenizer)
        if candidate is None:
            raise ProducerError(f"selected Finance row {index} is not renderable")
        declared = by_index[index]
        checks = {
            "record_id": candidate.record_id,
            "public_record_sha256": candidate.public_record_sha256,
            "full_token_count": candidate.full_token_count,
            "post_bos_token_count": candidate.post_bos_token_count,
            "truncated_sequence_sha256": candidate.truncated_sequence_sha256,
        }
        for key, actual in checks.items():
            if declared.get(key) != actual:
                raise ProducerError(f"selected {style} row {index} changed: {key}")
        if candidate.post_bos_token_count < (fc.SEQUENCE_TOKENS[style] - 1 if style == "pile" else 32):
            raise ProducerError(f"selected {style} row {index} is shorter than its declared minimum")
        result.append(candidate)
    if [row.record_id for row in result] != [str(row["record_id"]) for row in expected]:
        raise ProducerError(f"selected {style} row order changed")
    if len({row.truncated_sequence_sha256 for row in result}) != len(result):
        raise ProducerError(f"selected {style} truncated sequences are duplicated")
    return result


def _mask_positions(batch: Any, *, sequence_tokens: int) -> tuple[list[list[int]], list[list[int]]]:
    mask = batch.attention_mask[:, :sequence_tokens].to(torch.long).contiguous()
    positions = batch.position_ids[:, :sequence_tokens].to(torch.long).contiguous()
    return mask.tolist(), positions.tolist()


def _runtime_snapshot(model_snapshot: Path) -> dict[str, Any]:
    model_snapshot = model_snapshot.expanduser().resolve()
    if model_snapshot.is_symlink() or not model_snapshot.is_dir():
        raise ProducerError(f"public model snapshot is unavailable: {model_snapshot}")
    files: dict[str, Any] = {}
    for name in ("config.json", "generation_config.json", "model.safetensors"):
        path = model_snapshot / name
        resolved = path.resolve()
        if not resolved.is_file():
            raise ProducerError(f"public model snapshot file is unavailable: {path}")
        files[name] = {
            "path": str(path),
            "resolved_path": str(resolved),
            "bytes": int(resolved.stat().st_size),
            "sha256": _sha256_file(resolved),
            "symlink": path.is_symlink(),
        }
    if files["model.safetensors"]["bytes"] != 2_471_645_608 or files["model.safetensors"]["sha256"] != "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f":
        raise ProducerError("public model weight hash or size differs from the pinned snapshot")
    return {
        "model": {"id": fc.MODEL_ID, "revision": fc.MODEL_REVISION, "snapshot_path": str(model_snapshot)},
        "files": files,
    }


def _load_lora_config(path: Path) -> tuple[TargetLoRAConfig, dict[str, Any]]:
    value = _load_json(path, description="public LoRA configuration")
    config: Mapping[str, Any] | None = None
    if isinstance(value.get("config"), Mapping):
        config = value["config"]
    if config is None:
        conditions = value.get("conditions")
        if isinstance(conditions, list):
            for condition in conditions:
                if isinstance(condition, Mapping) and condition.get("id") == "public_lora_2601":
                    training = condition.get("training")
                    if isinstance(training, Mapping) and isinstance(training.get("config"), Mapping):
                        config = training["config"]
                        break
    if config is None and value.get("id") == "public_lora_2601":
        config = value
    if config is None:
        raise ProducerError("public_lora_2601 configuration is absent")
    required = ("layers", "modules", "rank", "alpha", "seed")
    if any(key not in config for key in required):
        raise ProducerError("public_lora_2601 configuration is incomplete")
    normalized = {
        "layers": tuple(int(item) for item in config["layers"]),
        "modules": tuple(str(item) for item in config["modules"]),
        "rank": int(config["rank"]),
        "alpha": float(config["alpha"]),
        "seed": int(config["seed"]),
    }
    if normalized["layers"] != (0, 1, 2, 3) or normalized["modules"] != ("q_proj", "v_proj") or normalized["rank"] != 4:
        raise ProducerError("public_lora_2601 geometry changed")
    return TargetLoRAConfig(**normalized), {key: (list(val) if isinstance(val, tuple) else val) for key, val in normalized.items()}


def _load_base_prefix(model_snapshot: Path, *, device: torch.device) -> tuple[ContiguousPublicPrefix, dict[str, Any]]:
    try:
        import trr0004_prepare_public_activations as prep

        prefix, snapshot, model_config = prep._load_public_prefix(
            model_snapshot, device=device, cut_depth=fc.CUT_DEPTH
        )
    except Exception as exc:
        raise ProducerError("public base prefix loading failed") from exc
    return prefix, {"snapshot": snapshot, "config": model_config, "condition": "public_base"}


def _load_shifted_prefix(
    model_snapshot: Path,
    *,
    device: torch.device,
    lora_config: TargetLoRAConfig,
    lora_update: Path,
) -> tuple[ContiguousPublicPrefix, dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    # Hashing is performed before the target update is loaded.  The update is
    # the public synthetic diagnostic and is never read by the reconstructor.
    snapshot = _runtime_snapshot(model_snapshot)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_snapshot), local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    if model.config.hidden_size != fc.HIDDEN_SIZE or model.config.vocab_size != fc.VOCAB_SIZE:
        raise ProducerError("public model geometry changed")
    model.requires_grad_(False)
    try:
        installed = install_target_lora(model, lora_config)
        load_target_lora(installed, lora_update)
    except Exception as exc:
        del model
        gc.collect()
        raise ProducerError("public_lora_2601 update installation failed") from exc
    prefix = ContiguousPublicPrefix(model, cut_depth=fc.CUT_DEPTH).to(device).eval()
    return prefix, {
        "snapshot": snapshot,
        "config": {
            "hidden_size": fc.HIDDEN_SIZE,
            "vocab_size": fc.VOCAB_SIZE,
            "cut_depth": fc.CUT_DEPTH,
            "torch_dtype": str(next(model.parameters()).dtype),
        },
        "condition": "public_lora_2601",
        "lora_config": {
            "layers": list(lora_config.layers),
            "modules": list(lora_config.modules),
            "rank": lora_config.rank,
            "alpha": lora_config.alpha,
            "seed": lora_config.seed,
        },
        "lora_update": {"path": str(lora_update), "bytes": int(lora_update.stat().st_size), "sha256": _sha256_file(lora_update)},
    }


def _guard_helpers() -> tuple[Any, Any]:
    try:
        import trr0004_prepare_public_activations as prep
        return prep._resource_preflight, prep._enforce_resource_ceiling
    except Exception as exc:
        raise ProducerError("public activation resource guard is unavailable") from exc


def _capture_condition(
    *,
    condition: str,
    model_snapshot: Path,
    token_batches: Mapping[str, Any],
    output_root: Path,
    root: Path,
    device: torch.device,
    lora_config: TargetLoRAConfig | None,
    lora_update: Path | None,
    selection_sha256: str,
    guard_preflight: Any,
    guard_ceiling: Any,
) -> tuple[dict[str, Path], dict[str, Any]]:
    # Guard before loading the multi-billion-byte public model into device
    # memory.  The same guard is checked after qualification and each batch.
    guard_preflight(
        device,
        min_free_gpu_bytes=MIN_FREE_GPU_BYTES,
        max_reserved_gpu_bytes=MAX_RESERVED_GPU_BYTES,
        max_host_rss_bytes=MAX_HOST_RSS_BYTES,
    )
    if condition == "public_base":
        prefix, load_evidence = _load_base_prefix(model_snapshot, device=device)
    else:
        assert lora_config is not None and lora_update is not None
        prefix, load_evidence = _load_shifted_prefix(
            model_snapshot,
            device=device,
            lora_config=lora_config,
            lora_update=lora_update,
        )
    guard_ceiling_call = lambda: guard_ceiling(
        device,
        max_reserved_gpu_bytes=MAX_RESERVED_GPU_BYTES,
        max_host_rss_bytes=MAX_HOST_RSS_BYTES,
    )
    # The 128-token Finance cell is the largest representative.  Qualify the
    # public prefix at the fixed 8x192 generation geometry before any cells.
    try:
        import trr0004_prepare_public_activations as prep

        qualification = prep._qualify_public_prefix_padding(
            prefix,
            token_batches["finance"],
            device=device,
            batch_size=CAPTURE_BATCH_SIZE,
        )
    except Exception as exc:
        raise ProducerError(f"{condition} largest-geometry qualification failed") from exc
    guard_ceiling_call()
    paths: dict[str, Path] = {}
    capture_metrics: dict[str, Any] = {}
    # Capture Finance before Pile after the largest qualification.  Both are
    # generated by the same fixed public-prefix path and batch geometry.
    for style in ("finance", "pile"):
        batch = token_batches[style]
        started = time.perf_counter()
        try:
            activations = capture_public_prefix(
                prefix,
                batch,
                device=device,
                batch_size=CAPTURE_BATCH_SIZE,
                resource_check=guard_ceiling_call,
            )
        except Exception as exc:
            raise ProducerError(f"{condition} {style} public-prefix capture failed") from exc
        sequence_tokens = fc.SEQUENCE_TOKENS[style]
        sliced = activations[:, :sequence_tokens].contiguous()
        path = output_root / "observations" / style / f"{condition}.safetensors"
        if path.exists() or path.is_symlink():
            raise ProducerError(f"observation artifact is create-only: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": "token-reconstruction.trr0004-fresh-confirmation-observation.v1",
            "task_id": TASK_ID,
            "style": style,
            "condition": condition,
            "cut_depth": str(fc.CUT_DEPTH),
            "hidden_size": str(fc.HIDDEN_SIZE),
            "source_generation_path": "public_prefix.forward_full",
            "generation_batch_size": str(CAPTURE_BATCH_SIZE),
            "generation_sequence_tokens": str(CAPTURE_SEQUENCE_TOKENS),
            "selection_plan_sha256": selection_sha256,
            "record_count": str(sliced.shape[0]),
            "sequence_tokens": str(sequence_tokens),
            "activations_sha256": tensor_sha256(sliced),
            "attention_mask_sha256": tensor_sha256(batch.attention_mask[:, :sequence_tokens]),
            "position_ids_sha256": tensor_sha256(batch.position_ids[:, :sequence_tokens]),
            "target_weights_available_to_reconstructor": "true" if condition == "public_base" else "false",
        }
        # The selection hash is supplied by the caller through a transient
        # module-level context to keep this helper's public API compact.
        save_file({"activations": sliced}, str(path), metadata=metadata)
        paths[style] = path
        capture_metrics[style] = {
            "elapsed_seconds": time.perf_counter() - started,
            "shape": list(sliced.shape),
            "activations_sha256": tensor_sha256(sliced),
            "source_generation": "fixed batch-8 x 192 public_prefix.forward_full",
            "primary_geometry": [CAPTURE_BATCH_SIZE, CAPTURE_SEQUENCE_TOKENS],
        }
        guard_ceiling_call()
    del prefix
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return paths, {"load": load_evidence, "qualification": qualification, "captures": capture_metrics}


def _observation_descriptor(path: Path, root: Path) -> dict[str, Any]:
    descriptor = fc.file_record(path, repository_root=root)
    descriptor.update({"tensor_key": "activations", "row_indices": list(range(fc.RECORDS_PER_STYLE))})
    return descriptor


def capture_public(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    selection_path = _require_repo_path(args.selection_plan, root, description="frozen selection plan")
    plan = _load_json(selection_path, description="frozen selection plan")
    if plan.get("schema") != SELECTION_SCHEMA or plan.get("task_id") != TASK_ID or plan.get("status") != "FROZEN_PUBLIC_SELECTION_NO_TRUTH":
        raise ProducerError("selection plan is not a frozen no-truth public selection")
    method_rows = plan.get("methods_prospective")
    if not isinstance(method_rows, list) or tuple(
        row.get("id") for row in method_rows if isinstance(row, Mapping)
    ) != fc.METHOD_IDS:
        raise ProducerError("selection plan method set changed")
    frozen_sources = plan.get("public_sources_frozen")
    declared_tokenizer = frozen_sources.get("tokenizer") if isinstance(frozen_sources, Mapping) else None
    if (
        not isinstance(declared_tokenizer, Mapping)
        or Path(str(declared_tokenizer.get("path"))).resolve()
        != args.tokenizer.expanduser().resolve()
    ):
        raise ProducerError("capture tokenizer differs from the frozen selection tokenizer")
    selection = plan.get("selection_rule")
    if not isinstance(selection, Mapping) or not selection.get("record_ids_selected"):
        raise ProducerError("selection plan has no frozen selected identities")
    tokenizer = _load_tokenizer(args.tokenizer)
    pile_paths = tuple(args.pile_arrow or ())
    finance_paths = tuple(args.finance_arrow or ())
    pile_descriptor = _dataset_descriptor(pile_paths, dataset_id=PILE_DATASET_ID, revision=PILE_DATASET_REVISION)
    finance_descriptor = _dataset_descriptor(finance_paths, dataset_id=FINANCE_DATASET_ID, fingerprint=FINANCE_DATASET_FINGERPRINT)
    _validate_frozen_source_binding(
        plan,
        pile_descriptor=pile_descriptor,
        finance_descriptor=finance_descriptor,
        tokenizer_descriptor=_tokenizer_descriptor(args.tokenizer),
    )
    pile_dataset = _load_arrow_dataset(pile_paths)
    finance_dataset = _load_arrow_dataset(finance_paths)
    pile_rows = _materialize_selected(plan, style="pile", dataset=pile_dataset, tokenizer=tokenizer)
    finance_rows = _materialize_selected(plan, style="finance", dataset=finance_dataset, tokenizer=tokenizer)
    token_batches = {
        style: pad_public_token_sequences(
            [list(row.token_ids[: fc.SEQUENCE_TOKENS[style]]) for row in rows],
            maximum_tokens=CAPTURE_SEQUENCE_TOKENS,
            pad_token_id=_public_padding_id(tokenizer),
            bos_token_id=fc.BOS_TOKEN_ID,
            vocab_size=fc.VOCAB_SIZE,
        )
        for style, rows in (("pile", pile_rows), ("finance", finance_rows))
    }
    output_root = _create_output_root(args.output_root, root, description="capture output root")
    model_snapshot = args.model_snapshot.expanduser().resolve()
    runtime = _runtime_snapshot(model_snapshot)
    lora_config, lora_config_metadata = _load_lora_config(args.lora_config)
    lora_update = _require_external_file(args.lora_update, description="public_lora_2601 update")
    device = _device(args.device)
    guard_preflight, guard_ceiling = _guard_helpers()
    selection_sha256 = _sha256_file(selection_path)
    started = _utc_now()
    conditions: dict[str, Any] = {}
    observations: dict[str, dict[str, Path]] = {}
    try:
        for condition in fc.CONDITION_ORDER:
            paths, evidence = _capture_condition(
                condition=condition,
                model_snapshot=model_snapshot,
                token_batches=token_batches,
                output_root=output_root,
                root=root,
                device=device,
                lora_config=lora_config if condition == "public_lora_2601" else None,
                lora_update=lora_update if condition == "public_lora_2601" else None,
                selection_sha256=selection_sha256,
                guard_preflight=guard_preflight,
                guard_ceiling=guard_ceiling,
            )
            observations[condition] = paths
            conditions[condition] = evidence
    except Exception as exc:
        failure = {
            "schema": f"{CAPTURE_EVIDENCE_SCHEMA}-failure",
            "task_id": TASK_ID,
            "status": "FAILED_PRESERVED",
            "started_utc": started,
            "ended_utc": _utc_now(),
            "error": repr(exc),
            "selection_plan": {"path": str(selection_path), "sha256": _sha256_file(selection_path)},
            "truth_opened": False,
        }
        _write_create_only(output_root / "capture_failure.json", failure)
        raise
    source_hash = _sha256_file(selection_path)
    styles = [
        {
            "id": "pile",
            "records": fc.RECORDS_PER_STYLE,
            "sequence_tokens": fc.SEQUENCE_TOKENS["pile"],
            "hidden_size": fc.HIDDEN_SIZE,
            "input_style": "plain Pile text",
            "source": pile_descriptor,
        },
        {
            "id": "finance",
            "records": fc.RECORDS_PER_STYLE,
            "sequence_tokens": fc.SEQUENCE_TOKENS["finance"],
            "hidden_size": fc.HIDDEN_SIZE,
            "input_style": "Finance chat-template rendering",
            "source": finance_descriptor,
        },
    ]
    records_by_style = {"pile": pile_rows, "finance": finance_rows}
    cells: list[dict[str, Any]] = []
    for style in fc.STYLE_ORDER:
        sequence_tokens = fc.SEQUENCE_TOKENS[style]
        mask, positions = _mask_positions(token_batches[style], sequence_tokens=sequence_tokens)
        records = [row.panel_metadata(sequence_tokens=sequence_tokens) for row in records_by_style[style]]
        for condition in fc.CONDITION_ORDER:
            cells.append(
                {
                    "id": f"{style}__{condition}",
                    "style": style,
                    "condition": condition,
                    "shift_role": "matched_public_control" if condition == "public_base" else "single_public_shift_diagnostic",
                    "records": records,
                    "attention_mask": mask,
                    "position_ids": positions,
                    "observation": _observation_descriptor(observations[condition][style], root),
                    "geometry": {
                        "records": fc.RECORDS_PER_STYLE,
                        "sequence_tokens": sequence_tokens,
                        "hidden_size": fc.HIDDEN_SIZE,
                        "cut_depth": fc.CUT_DEPTH,
                    },
                }
            )
    panel = {
        "schema": fc.PANEL_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_FRESH_CONFIRMATION_PANEL",
        "panel_id": "trr0004-fresh-confirmation-v1",
        "source_material_included": False,
        "model": {"id": fc.MODEL_ID, "revision": fc.MODEL_REVISION},
        "cut_depth": fc.CUT_DEPTH,
        "hidden_size": fc.HIDDEN_SIZE,
        "selection_plan_sha256": source_hash,
        "observation_generation": {
            "path": "public_prefix.forward_full",
            "same_public_prefix_path": True,
            "batch_size": CAPTURE_BATCH_SIZE,
            "sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "padding_semantics": "right-padding; active outputs qualified bit-exact under future-pad perturbation",
            "primary_capture_geometry": "fixed batch-8 x 192",
            "alternate_batch_policy": "unpadded batch-1 diagnostic only and not used",
        },
        "styles": styles,
        "conditions": [
            {"id": "public_base", "role": "matched public control", "weights_available_to_reconstructor": True, "online_prefix_calls_allowed": True},
            {"id": "public_lora_2601", "role": "one synthetic target-shift diagnostic", "weights_available_to_reconstructor": False, "online_prefix_calls_allowed": True},
        ],
        "cells": cells,
        "method_output_contract": {
            "method_ids": list(fc.METHOD_IDS),
            "artifact_template": "<output>/<style>/<condition>/<method_id>.safetensors",
            "required_tensors": ["predictions"],
            "optional_diagnostics": ["candidates", "candidate_scores", "selection_scores"],
            "all_cells_required_before_evaluation": True,
        },
        "canonical_status": {
            "new_track_a_methods": "NOT_RUN",
            "new_track_b_methods": "NOT_RUN",
            "dual_benchmark_comparison": "INCOMPLETE",
        },
    }
    panel_path = output_root / "panel.json"
    _write_create_only(panel_path, panel)
    try:
        fc.load_fresh_panel(panel_path, repository_root=root)
    except Exception as exc:
        raise ProducerError("captured public panel failed its frozen contract") from exc
    evidence = {
        "schema": CAPTURE_EVIDENCE_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_PUBLIC_PANEL_NO_TRUTH",
        "started_utc": started,
        "ended_utc": _utc_now(),
        "execution": {
            "git_commit": _git_commit(root),
            "producer": fc.file_record(Path(__file__).resolve(), repository_root=root),
            "python": sys.executable,
            "platform": platform.platform(),
            "device": str(device),
            "truth_opened": False,
            "network_used": False,
        },
        "selection_plan": {"path": str(selection_path), "sha256": source_hash},
        "panel": {"path": str(panel_path), "sha256": _sha256_file(panel_path)},
        "tokenizer": _tokenizer_descriptor(args.tokenizer),
        "runtime": runtime,
        "lora_config": {"path": str(args.lora_config.resolve()), "sha256": _sha256_file(args.lora_config), "config": lora_config_metadata},
        "lora_update": {"path": str(lora_update), "bytes": int(lora_update.stat().st_size), "sha256": _sha256_file(lora_update)},
        "conditions": conditions,
        "fixed_geometry": {"batch_records": CAPTURE_BATCH_SIZE, "sequence_tokens": CAPTURE_SEQUENCE_TOKENS, "cut_depth": fc.CUT_DEPTH, "hidden_size": fc.HIDDEN_SIZE},
        "resource_limits": {"minimum_free_gpu_bytes": MIN_FREE_GPU_BYTES, "maximum_reserved_gpu_bytes": MAX_RESERVED_GPU_BYTES, "maximum_host_rss_bytes": MAX_HOST_RSS_BYTES},
        "observation_paths": {
            condition: {style: {"path": str(path), "sha256": _sha256_file(path), "bytes": int(path.stat().st_size)} for style, path in paths.items()}
            for condition, paths in observations.items()
        },
    }
    _write_create_only(output_root / "capture_evidence.json", evidence)
    return {"task_id": TASK_ID, "status": evidence["status"], "output_root": str(output_root), "panel": str(panel_path), "panel_sha256": evidence["panel"]["sha256"], "truth_opened": False}


def _truth_preparation(
    *,
    root: Path,
    selection_path: Path,
    panel_path: Path,
    output_root: Path,
    pile_rows: Sequence[RenderedRecord],
    finance_rows: Sequence[RenderedRecord],
    pile_descriptor: Mapping[str, Any],
    finance_descriptor: Mapping[str, Any],
    tokenizer_path: Path,
    truth_sidecar: Path,
) -> dict[str, Any]:
    value = {
        "schema": TRUTH_PREPARATION_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_LABEL_PREPARATION_SEPARATE_ROLE",
        "created_at_utc": _utc_now(),
        "producer": fc.file_record(Path(__file__).resolve(), repository_root=root),
        "execution": {
            "git_commit": _git_commit(root),
            "python": sys.executable,
            "truth_opened_by_reconstructor": False,
            "source_text_written": False,
            "token_ids_written": False,
        },
        "selection_plan": {"path": str(selection_path), "sha256": _sha256_file(selection_path)},
        "panel": {"path": str(panel_path), "sha256": _sha256_file(panel_path)},
        "public_sources": {"pile": dict(pile_descriptor), "finance": dict(finance_descriptor)},
        "tokenizer": {"path": str(tokenizer_path.resolve()), "snapshot": fc.MODEL_REVISION},
        "paired_conditions": True,
        "record_counts": {"pile": len(pile_rows), "finance": len(finance_rows)},
        "sidecar_path": str(truth_sidecar),
        "sidecar_role": "private evaluator-label sidecar prepared separately; not loaded by this producer",
    }
    _write_create_only(output_root / "truth_preparation.json", value)
    return value


def prepare_truth(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    selection_path = _require_repo_path(args.selection_plan, root, description="frozen selection plan")
    panel_path = _require_repo_path(args.panel, root, description="frozen public panel")
    plan = _load_json(selection_path, description="frozen selection plan")
    if plan.get("status") != "FROZEN_PUBLIC_SELECTION_NO_TRUTH":
        raise ProducerError("selection plan is not frozen")
    panel = fc.load_fresh_panel(panel_path, repository_root=root)
    tokenizer = _load_tokenizer(args.tokenizer)
    pile_paths = tuple(args.pile_arrow or ())
    finance_paths = tuple(args.finance_arrow or ())
    pile_descriptor = _dataset_descriptor(pile_paths, dataset_id=PILE_DATASET_ID, revision=PILE_DATASET_REVISION)
    finance_descriptor = _dataset_descriptor(finance_paths, dataset_id=FINANCE_DATASET_ID, fingerprint=FINANCE_DATASET_FINGERPRINT)
    _validate_frozen_source_binding(
        plan,
        pile_descriptor=pile_descriptor,
        finance_descriptor=finance_descriptor,
        tokenizer_descriptor=_tokenizer_descriptor(args.tokenizer),
    )
    pile_dataset = _load_arrow_dataset(pile_paths)
    finance_dataset = _load_arrow_dataset(finance_paths)
    pile_rows = _materialize_selected(plan, style="pile", dataset=pile_dataset, tokenizer=tokenizer)
    finance_rows = _materialize_selected(plan, style="finance", dataset=finance_dataset, tokenizer=tokenizer)
    panel_records = {style: [row["record_id"] for row in _selected_metadata(plan, style)] for style in fc.STYLE_ORDER}
    if panel_records != {style: list(next(cell for cell in panel["cells"] if cell["style"] == style)["records"][index]["record_id"] for index in range(fc.RECORDS_PER_STYLE)) for style in fc.STYLE_ORDER}:
        raise ProducerError("public panel and selected source records do not agree")
    truth_batches = {
        style: pad_public_token_sequences(
            [list(row.token_ids[: fc.SEQUENCE_TOKENS[style]]) for row in rows],
            maximum_tokens=CAPTURE_SEQUENCE_TOKENS,
            pad_token_id=_public_padding_id(tokenizer),
            bos_token_id=fc.BOS_TOKEN_ID,
            vocab_size=fc.VOCAB_SIZE,
        )
        for style, rows in (("pile", pile_rows), ("finance", finance_rows))
    }
    cells = fc.load_fresh_cells(panel, repository_root=root)
    truth: dict[str, torch.Tensor] = {}
    for cell in cells:
        batch = truth_batches[cell.style]
        truth[cell.cell_id] = batch.token_ids[:, : cell.sequence_tokens].to(torch.int64).clone().contiguous()
        if not torch.equal(batch.attention_mask[:, : cell.sequence_tokens].to(torch.long), cell.attention_mask.to(torch.long)):
            raise ProducerError(f"public truth preparation mask differs from panel: {cell.cell_id}")
        if not torch.equal(batch.position_ids[:, : cell.sequence_tokens].to(torch.long), cell.position_ids.to(torch.long)):
            raise ProducerError(f"public truth preparation positions differ from panel: {cell.cell_id}")
    truth_sidecar = _require_external_destination(args.truth_sidecar, description="private truth sidecar destination")
    try:
        truth_sidecar.resolve().relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise ProducerError("private truth sidecar must be outside the repository")
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.relative_to(root.resolve())
    except ValueError as exc:
        raise ProducerError("truth preparation output root must be inside the repository") from exc
    if output_root.exists() or output_root.is_symlink():
        if output_root.is_symlink() or not output_root.is_dir():
            raise ProducerError("truth preparation output root is not a regular directory")
    else:
        output_root.mkdir(parents=True)
    preparation = _truth_preparation(
        root=root,
        selection_path=selection_path,
        panel_path=panel_path,
        output_root=output_root,
        pile_rows=pile_rows,
        finance_rows=finance_rows,
        pile_descriptor=pile_descriptor,
        finance_descriptor=finance_descriptor,
        tokenizer_path=args.tokenizer,
        truth_sidecar=truth_sidecar,
    )
    preparation_record = fc.file_record(output_root / "truth_preparation.json", repository_root=root)
    cells_for_binding = cells
    placeholder = {"path": str(truth_sidecar.resolve()), "bytes": 0, "sha256": "0" * 64}
    binding = fc.build_confirmation_truth_binding(
        panel_sha256=_sha256_file(panel_path),
        selection_plan_sha256=_sha256_file(selection_path),
        cells=cells_for_binding,
        truth=truth,
        preparation=preparation_record,
        sidecar=placeholder,
    )
    # write_confirmation_truth_sidecar clones every tensor at the save
    # boundary.  The values are already separate clones above so paired
    # conditions cannot alias a safetensors storage block.
    fc.write_confirmation_truth_sidecar(truth_sidecar, cells=cells_for_binding, truth=truth, binding=binding)
    sidecar_record = fc.external_file_record(truth_sidecar)
    binding = fc.build_confirmation_truth_binding(
        panel_sha256=_sha256_file(panel_path),
        selection_plan_sha256=_sha256_file(selection_path),
        cells=cells_for_binding,
        truth=truth,
        preparation=preparation_record,
        sidecar=sidecar_record,
    )
    binding_path = output_root / "truth_binding.json"
    _write_create_only(binding_path, binding)
    # No sidecar validation call is made here: the sidecar remains unopened by
    # this producer after its separate preparation write.
    return {"task_id": TASK_ID, "status": "PUBLIC_TRUTH_PREPARED_SEPARATE_ROLE", "truth_binding": str(binding_path), "truth_sidecar": str(truth_sidecar), "truth_opened": False}


def _resolve_spec_paths(values: Any, *, root: Path, description: str) -> tuple[Path, ...]:
    if not isinstance(values, list) or not values:
        raise ProducerError(f"{description} is missing")
    return tuple(_require_repo_path(Path(str(value)), root, description=description) for value in values)


def register_methods(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    panel_path = _require_repo_path(args.panel, root, description="frozen public panel")
    selection_path = _require_repo_path(args.selection_plan, root, description="frozen selection plan")
    spec_path = _require_repo_path(args.binding_spec, root, description="method binding specification")
    spec = _load_json(spec_path, description="method binding specification")
    raw_methods = spec.get("methods") if isinstance(spec.get("methods"), Mapping) else spec.get("bindings")
    if not isinstance(raw_methods, Mapping):
        raw_methods = spec
    bindings: dict[str, Mapping[str, Any]] = {}
    for method in fc.METHOD_SPECS:
        method_id = method["id"]
        raw = raw_methods.get(method_id)
        if not isinstance(raw, Mapping):
            raise ProducerError(f"binding specification omits {method_id}")
        if all(key in raw for key in ("panel", "method_state", "method_config", "code", "runtime_assets")):
            binding = dict(raw)
            if binding.get("method_id") != method_id or binding.get("method_rule") != method["rule"]:
                raise ProducerError(f"serialized binding rule changed: {method_id}")
            bindings[method_id] = binding
            continue
        state_paths = _resolve_spec_paths(raw.get("method_state_paths"), root=root, description=f"{method_id} state paths")
        config_paths = _resolve_spec_paths(raw.get("method_config_paths"), root=root, description=f"{method_id} config paths")
        code_paths = _resolve_spec_paths(raw.get("code_paths"), root=root, description=f"{method_id} code paths")
        runtime_raw = raw.get("runtime_asset_paths")
        if not isinstance(runtime_raw, Mapping) or set(runtime_raw) != set(fc.RUNTIME_ASSET_ROLES):
            raise ProducerError(f"{method_id} runtime asset paths are incomplete")
        runtime_paths = {
            role: _require_public_asset_file(
                Path(str(value)), description=f"{method_id} runtime asset {role}"
            )
            for role, value in runtime_raw.items()
        }
        code_commit = raw.get("code_commit") or _git_commit(root)
        if not isinstance(code_commit, str):
            raise ProducerError(f"{method_id} code commit is absent")
        bindings[method_id] = fc.make_confirmation_binding(
            panel_path=panel_path,
            repository_root=root,
            method_id=method_id,
            method_rule=method["rule"],
            method_state_paths=state_paths,
            method_config_paths=config_paths,
            code_paths=code_paths,
            code_commit=code_commit,
            runtime_asset_paths=runtime_paths,
        )
    output = args.output.expanduser().resolve()
    payload = fc.build_confirmation_registration(
        panel_path=panel_path,
        selection_plan_path=selection_path,
        repository_root=root,
        bindings=bindings,
        output_path=output,
    )
    return {"task_id": TASK_ID, "status": payload["status"], "registration": str(output), "registration_sha256": _sha256_file(output), "method_ids": list(fc.METHOD_IDS)}


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 and all(char in "0123456789abcdef" for char in value) else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="promote the frozen public selection rule")
    select.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    select.add_argument("--prospective-plan", type=Path, required=True)
    select.add_argument("--method-freeze", type=Path, required=True)
    select.add_argument("--tokenizer", type=Path, required=True)
    select.add_argument("--pile-arrow", type=Path, action="append", required=True)
    select.add_argument("--finance-arrow", type=Path, action="append", required=True)
    select.add_argument("--exclude-source", type=Path, action="append")
    select.add_argument("--output", type=Path, required=True)

    capture = sub.add_parser("capture", help="capture four paired public-prefix observation cells")
    capture.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    capture.add_argument("--selection-plan", type=Path, required=True)
    capture.add_argument("--tokenizer", type=Path, required=True)
    capture.add_argument("--pile-arrow", type=Path, action="append", required=True)
    capture.add_argument("--finance-arrow", type=Path, action="append", required=True)
    capture.add_argument("--model-snapshot", type=Path, required=True)
    capture.add_argument("--lora-config", type=Path, required=True)
    capture.add_argument("--lora-update", type=Path, required=True)
    capture.add_argument("--output-root", type=Path, required=True)
    capture.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    truth = sub.add_parser("truth", help="prepare the separate public-label sidecar")
    truth.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    truth.add_argument("--selection-plan", type=Path, required=True)
    truth.add_argument("--panel", type=Path, required=True)
    truth.add_argument("--tokenizer", type=Path, required=True)
    truth.add_argument("--pile-arrow", type=Path, action="append", required=True)
    truth.add_argument("--finance-arrow", type=Path, action="append", required=True)
    truth.add_argument("--output-root", type=Path, required=True)
    truth.add_argument("--truth-sidecar", type=Path, required=True)

    register = sub.add_parser("register", help="write the exact five-method registration")
    register.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    register.add_argument("--selection-plan", type=Path, required=True)
    register.add_argument("--panel", type=Path, required=True)
    register.add_argument("--binding-spec", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        result = select_public(args)
    elif args.command == "capture":
        result = capture_public(args)
    elif args.command == "truth":
        result = prepare_truth(args)
    elif args.command == "register":
        result = register_methods(args)
    else:  # pragma: no cover - argparse enforces command choices
        raise ProducerError(f"unknown producer command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProducerError as exc:
        print(f"trr0004_produce_confirmation: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
