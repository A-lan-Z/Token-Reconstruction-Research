#!/usr/bin/env python3
"""Produce the TRR-0005 fresh public confirmation panel.

This adapter owns only public source reservation, paired source metadata, and
public full-prefix observations. The first freeze is source-free: the eight
method IDs, state/code bindings, public validation choice, and decision-plan
digest must be frozen before select or capture can read a reserved row. The
panel stores IDs, public rendered digests, and compact geometry; it never
stores source text, token IDs, targets, or private evaluator truth.

capture uses the accepted TRR-0004 public-prefix implementation at its fixed
8-by-192 padded forward geometry, then emits compact 128-token observations
for the fresh panel. truth is a separate preparation-role command and writes
a public-label sidecar only outside this checkout.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from token_reconstruction.public_activation import (
    capture_public_prefix,
    pad_public_token_sequences,
    save_public_artifact,
)
from token_reconstruction.trr0005_contract import (
    BOS_TOKEN_ID,
    CONDITION_ORDER,
    EXPECTED_CELL_IDS,
    INVALID_TOKEN_ID,
    METHOD_IDS,
    PANEL_SCHEMA,
    RECORDS_PER_DOMAIN,
    RESERVED_SOURCE_POOLS,
    SEQUENCE_TOKENS,
    STYLE_ORDER,
    TASK_ID,
    ContractError,
    valid_sha256,
    validate_method_ids,
    validate_panel_descriptor,
)
from token_reconstruction.trr0005_public_corpus import (
    SOURCE_DATASETS,
    SOURCE_PARTITIONS,
    deterministic_row_order,
    source_record_id,
    validate_partition_index,
)


PRODUCER_SCHEMA = "token-reconstruction.trr0005-confirmation-producer.v1"
SELECTION_SCHEMA = "token-reconstruction.trr0005-fresh-source-selection.v1"
CAPTURE_SCHEMA = "token-reconstruction.trr0005-fresh-confirmation-capture.v1"
TRUTH_SCHEMA = "token-reconstruction.trr0005-fresh-confirmation-truth-preparation.v1"
OBSERVATION_SCHEMA = "token-reconstruction.trr0005-fresh-confirmation-observation.v1"
PUBLIC_SELECTION_SCHEMA = "token-reconstruction.trr0005-public-validation-selection.v1"

SELECTION_SEED = 5005
CAPTURE_BATCH_SIZE = 8
CAPTURE_SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
CUT_DEPTH = 4
VOCAB_SIZE = 128256
PADDING_TOKEN_ID = 128001
PAD_TOKEN_ID = PADDING_TOKEN_ID
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
DATE_STRING = "06 Aug 2026"
MAX_HOST_RSS_BYTES = 16 * 1024**3
MAX_RESERVED_GPU_BYTES = 8 * 1024**3
MIN_FREE_GPU_BYTES = 8 * 1024**3

# These names are deliberately narrow. A metadata source is explicit and
# identity-only; the scanner never descends into source/token/truth payloads.
_ID_KEYS = {"record_id", "source_record_id", "public_record_id"}
_HASH_KEYS = {
    "public_record_sha256",
    "normalized_content_sha256",
    "tokenized_record_sha256",
    "token_ids_sha256",
    "rendered_sha256",
    "content_sha256",
    "text_sha256",
    "truncated_sequence_sha256",
    "final_sequence_sha256",
}
_INDEX_KEYS = {"row_index", "raw_index", "source_index", "dataset_index", "index"}
_PRIVATE_KEY_FRAGMENTS = (
    "token_ids",
    "input_ids",
    "labels",
    "source_text",
    "plaintext",
    "oracle",
    "private_truth",
    "target_tokens",
)


class ProducerError(ContractError):
    """Raised when the public producer cannot satisfy the frozen contract."""


@dataclass(frozen=True)
class FreshRecord:
    """Transient rendered public source row; token IDs are never serialized."""

    style: str
    dataset_key: str
    dataset_id: str
    split: str
    revision: str
    row_index: int
    record_id: str
    public_record_sha256: str
    token_ids: tuple[int, ...]
    final_sequence_sha256: str

    @property
    def full_token_count(self) -> int:
        return len(self.token_ids)

    @property
    def post_bos_token_count(self) -> int:
        return len(self.token_ids) - 1

    def selection_metadata(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "public_record_sha256": self.public_record_sha256,
            "dataset_key": self.dataset_key,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "revision": self.revision,
            "row_index": self.row_index,
            "source_index": self.row_index,
            "full_token_count": self.full_token_count,
            "post_bos_token_count": self.post_bos_token_count,
            "valid_tokens": SEQUENCE_TOKENS,
            "final_sequence_sha256": self.final_sequence_sha256,
        }

    def panel_metadata(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "public_record_sha256": self.public_record_sha256,
            "raw_index": self.row_index,
            "source_index": self.row_index,
            "valid_tokens": SEQUENCE_TOKENS,
        }


@dataclass
class ExclusionSets:
    ids: dict[str, set[str]]
    hashes: dict[str, set[str]]
    indices: dict[str, set[int]]
    sources: list[dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ProducerError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProducerError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProducerError(f"{description} must be a JSON object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser()
    if path.exists() or path.is_symlink():
        raise ProducerError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _git_commit(root: Path) -> str | None:
    try:
        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _file_record(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ProducerError(f"required file is unavailable: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
    }
    if hash_file:
        result["sha256"] = _sha256_file(path)
    return result


def _new_file_destination(path: Path, *, description: str) -> Path:
    path = path.expanduser()
    if path.exists() or path.is_symlink():
        raise ProducerError(f"{description} is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _new_repo_root(path: Path, root: Path, *, description: str) -> Path:
    path = path.expanduser()
    if path.exists() or path.is_symlink():
        raise ProducerError(f"{description} is create-only and already exists: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ProducerError(f"{description} must be inside the repository") from exc
    path.mkdir(parents=True)
    return path.resolve()


def _outside_repo_destination(path: Path, root: Path, *, description: str) -> Path:
    path = path.expanduser()
    if path.exists() or path.is_symlink():
        raise ProducerError(f"{description} is create-only and already exists: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    raise ProducerError(f"{description} must be outside the repository")


def _require_sha(value: Any, *, description: str) -> str:
    try:
        return valid_sha256(value, description=description)
    except ContractError as exc:
        raise ProducerError(str(exc)) from exc


def _nested_value(value: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _has_forbidden_preselection_payload(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold().replace("-", "_")
            if lowered in {
                "source_record_id",
                "public_record_sha256",
                "row_index",
                "raw_index",
                "token_ids",
                "input_ids",
                "labels",
                "panel_sha256",
                "fresh_panel",
                "holdout_ids",
            }:
                return True
            if _has_forbidden_preselection_payload(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_forbidden_preselection_payload(child) for child in value)
    return False


def _validate_public_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        try:
            from trr0005_score_confirmation import validate_public_validation_selection
        except ModuleNotFoundError:
            from scripts.trr0005_score_confirmation import validate_public_validation_selection

        validate_public_validation_selection(value)
    except Exception as exc:
        raise ProducerError("frozen public-validation selection is invalid") from exc
    return dict(value)


def _validate_preselection(
    path: Path,
    *,
    decision_plan_path: Path | None = None,
    public_validation_selection_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the source-free method freeze before any reserved row access."""

    marker = _load_json(path, description="method freeze marker")
    method_ids = marker.get("method_ids")
    if method_ids is None and isinstance(marker.get("methods"), list):
        method_ids = [
            row.get("id") for row in marker["methods"] if isinstance(row, Mapping)
        ]
    try:
        validate_method_ids(method_ids or ())
    except ContractError as exc:
        raise ProducerError("method freeze does not contain the exact eight methods") from exc

    allowed_statuses = {
        "FROZEN_METHOD_PRESELECTION",
        "FROZEN_METHOD_REGISTRATION",
        "FROZEN_METHOD_STATES",
        "ACTIVE_METHODS_FROZEN",
        "FROZEN",
    }
    if marker.get("status") not in allowed_statuses:
        raise ProducerError("method freeze is not in a frozen preselection state")
    if marker.get("truth_opened") is True or marker.get("fresh_evaluation_started") is True:
        raise ProducerError("method freeze was written after truth or fresh evaluation")
    if _has_forbidden_preselection_payload(marker):
        raise ProducerError("method freeze contains fresh source or panel payload")

    code_commit = _nested_value(
        marker,
        "code_commit",
        "executed_code_commit",
        "executable_code_commit",
    )
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        executed = marker.get("executed_code")
        if isinstance(executed, Mapping):
            code_commit = _nested_value(executed, "commit", "code_commit", "sha")
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise ProducerError("method freeze lacks a full executed code commit")
    code_commit = code_commit.lower()
    if any(ch not in "0123456789abcdef" for ch in code_commit):
        raise ProducerError("method freeze code commit is not hexadecimal")

    decision_digest = marker.get("decision_plan_sha256")
    if decision_digest is None and isinstance(marker.get("decision_plan"), Mapping):
        decision_digest = _nested_value(marker["decision_plan"], "sha256", "digest")
    if decision_plan_path is not None:
        decision_path = decision_plan_path.expanduser().resolve()
        actual = _sha256_file(decision_path)
        if decision_digest != actual:
            raise ProducerError("decision-plan digest differs from the frozen marker")
    if not isinstance(decision_digest, str) or len(decision_digest) != 64:
        raise ProducerError("method freeze lacks a decision-plan SHA-256 binding")
    _require_sha(decision_digest, description="decision plan")

    bindings = marker.get("state_bindings")
    if bindings is None:
        bindings = marker.get("method_states")
    if bindings is None and isinstance(marker.get("methods"), list):
        bindings = {}
        for row in marker["methods"]:
            if isinstance(row, Mapping) and isinstance(row.get("id"), str):
                state = row.get("method_state", row.get("state"))
                bindings[row["id"]] = {
                    "status": row.get("status", "FROZEN"),
                    "method_state": state,
                    "state_sha256": row.get("state_sha256"),
                }
    if isinstance(bindings, list):
        bindings = {
            row.get("method_id"): row
            for row in bindings
            if isinstance(row, Mapping) and isinstance(row.get("method_id"), str)
        }
    if not isinstance(bindings, Mapping) or tuple(bindings) != METHOD_IDS:
        # A mapping written with sorted keys is accepted by identity, but the
        # method_ids array above remains the canonical execution order.
        if not isinstance(bindings, Mapping) or set(bindings) != set(METHOD_IDS):
            raise ProducerError("method freeze lacks all eight ordered state bindings")
    state_digests: dict[str, str] = {}
    for method_id in METHOD_IDS:
        binding = bindings.get(method_id)
        if not isinstance(binding, Mapping):
            raise ProducerError(f"method freeze binding is malformed: {method_id}")
        if binding.get("status") in {"PENDING_STATE", "UNFIT", "UNSELECTED"}:
            raise ProducerError(f"method freeze state is not complete: {method_id}")
        nested = binding.get("method_state")
        digest = _nested_value(binding, "state_sha256", "state_hash", "sha256")
        if digest is None and isinstance(nested, Mapping):
            digest = _nested_value(nested, "sha256", "state_sha256", "hash")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ProducerError(f"method freeze lacks a state hash: {method_id}")
        state_digests[method_id] = _require_sha(digest, description=f"{method_id} state")

    selection = marker.get("public_validation_selection")
    if selection is None:
        selection = marker.get("frozen_public_validation_selection")
    if selection is None:
        selection = marker.get("selection")
    if selection is None and public_validation_selection_path is not None:
        selection = _load_json(
            public_validation_selection_path,
            description="public-validation selection",
        )
    if not isinstance(selection, Mapping):
        raise ProducerError("method freeze lacks public-validation selection")
    selection_value = _validate_public_selection(selection)

    return {
        "path": str(path.expanduser().resolve()),
        "sha256": _sha256_file(path.expanduser().resolve()),
        "status": marker["status"],
        "code_commit": code_commit,
        "decision_plan_sha256": decision_digest,
        "method_ids": list(METHOD_IDS),
        "state_sha256": state_digests,
        "public_validation_selection": selection_value,
    }


def _validate_method_freeze(path: Path) -> dict[str, Any]:
    """Compatibility name for callers that used the TRR4 producer helper."""

    return _validate_preselection(path)


def _require_external_destination(path: Path, *, description: str) -> Path:
    """Return a create-only destination for synthetic tests and truth output."""

    path = path.expanduser()
    if path.exists() or path.is_symlink():
        raise ProducerError(f"{description} is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _infer_style(value: Any, *, hint: str = "") -> str | None:
    text = f"{hint} {value or ''}".casefold()
    if "finance" in text:
        return "finance"
    if "pile" in text:
        return "pile"
    return None


def _scan_identity_metadata(
    value: Any,
    *,
    hint: str,
    result: ExclusionSets,
) -> None:
    if isinstance(value, Mapping):
        local_style = _infer_style(value.get("dataset"), hint=hint)
        local_style = local_style or _infer_style(value.get("domain"), hint=hint)
        local_style = local_style or _infer_style(value.get("record_id"), hint=hint)
        for key, child in value.items():
            lowered = str(key).casefold().replace("-", "_")
            if lowered not in _HASH_KEYS and any(
                fragment in lowered for fragment in _PRIVATE_KEY_FRAGMENTS
            ):
                continue
            if lowered in _ID_KEYS and isinstance(child, str) and child:
                style = _infer_style(child, hint=local_style or hint)
                if style is not None:
                    result.ids[style].add(child)
            if lowered in _HASH_KEYS and isinstance(child, str) and len(child) == 64:
                style = local_style or _infer_style(child, hint=hint)
                if style is not None:
                    result.hashes[style].add(child)
            if lowered in _INDEX_KEYS and isinstance(child, int) and not isinstance(child, bool):
                if local_style is not None and child >= 0:
                    result.indices[local_style].add(int(child))
            _scan_identity_metadata(
                child,
                hint=f"{local_style or hint} {lowered}",
                result=result,
            )
    elif isinstance(value, list):
        for child in value:
            _scan_identity_metadata(child, hint=hint, result=result)


def _default_exclusion_paths(root: Path) -> list[Path]:
    """Return explicit prior public metadata files used for identity exclusion."""

    source_repo = root.parent.parent if root.parent.name == ".worktrees" else root
    return [
        root / "experiments/TRR-0003/evidence/control/panel.json",
        root / "experiments/TRR-0003/evidence/control/plan.json",
        root / "experiments/TRR-0003/footing/panel.json",
        root / "experiments/TRR-0003/evidence/control/track_b_fit_records.json",
        root / "experiments/TRR-0004/fit/adapter_v2/affine_fit_records.json",
        root / "experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json",
        root / "experiments/TRR-0004/fit/adapter_v2/public_fit_manifest.json",
        root / "experiments/TRR-0004/fit/affine_fit_records.json",
        root / "experiments/TRR-0004/fit/affine_validation_records.json",
        root / "experiments/TRR-0004/fresh_confirmation_v1/selection_plan.json",
        root / "experiments/TRR-0004/fresh_confirmation_v1/panel_capture/panel.json",
        root / "experiments/TRR-0004/alpaca_split_plan.json",
        root / "experiments/TRR-0005/corpus/corpus_plan.json",
        source_repo / "outputs/TRR-0003/track_b/public_fit_v2/fit_records.json",
        source_repo / "outputs/TRR-0003/track_b/public_validation_slice_v2/public_validation_records.json",
    ]


def _collect_exclusions(paths: Sequence[Path]) -> ExclusionSets:
    result = ExclusionSets(
        ids={style: set() for style in STYLE_ORDER},
        hashes={style: set() for style in STYLE_ORDER},
        indices={style: set() for style in STYLE_ORDER},
        sources=[],
    )
    for original in paths:
        path = original.expanduser().resolve()
        source: dict[str, Any] = {
            "path": str(path),
            "available": False,
            "identity_only": True,
        }
        if path.is_symlink() or not path.is_file():
            result.sources.append(source)
            continue
        source.update(
            {
                "available": True,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProducerError(f"public exclusion metadata is invalid: {path}") from exc
        before = sum(len(v) for v in result.ids.values()) + sum(
            len(v) for v in result.hashes.values()
        )
        _scan_identity_metadata(
            value,
            hint=f"{path.name} {path.parent}",
            result=result,
        )
        after = sum(len(v) for v in result.ids.values()) + sum(
            len(v) for v in result.hashes.values()
        )
        source["new_identity_count"] = after - before
        result.sources.append(source)
    return result


def _blocked(record: FreshRecord, exclusions: ExclusionSets) -> str | None:
    if record.record_id in exclusions.ids[record.style]:
        return "public_source_id"
    if record.public_record_sha256 in exclusions.hashes[record.style]:
        return "public_rendered_hash"
    if record.final_sequence_sha256 in exclusions.hashes[record.style]:
        return "public_final_sequence_hash"
    if record.row_index in exclusions.indices[record.style]:
        return "public_source_index"
    return None


def _load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_dir():
        raise ProducerError(f"public tokenizer snapshot is unavailable: {path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            use_fast=True,
        )
    except Exception as exc:
        raise ProducerError("public tokenizer could not be loaded offline") from exc
    if int(getattr(tokenizer, "bos_token_id", -1)) != BOS_TOKEN_ID:
        raise ProducerError("public tokenizer BOS ID differs from the contract")
    padding = getattr(tokenizer, "pad_token_id", PADDING_TOKEN_ID)
    if padding is None:
        padding = PADDING_TOKEN_ID
    if int(padding) != PADDING_TOKEN_ID:
        raise ProducerError("public tokenizer padding ID differs from the contract")
    return tokenizer


def _tokenizer_descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_dir():
        raise ProducerError(f"public tokenizer snapshot is unavailable: {path}")
    files: dict[str, Any] = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        candidate = path / name
        # Hugging Face snapshot directories normally expose immutable blob
        # files through symlinks. Resolve the file for its byte binding while
        # retaining the snapshot path in the descriptor.
        if not candidate.resolve().is_file():
            continue
        files[name] = {
            **_file_record(candidate),
            "snapshot_path": str(candidate),
            "symlink": candidate.is_symlink(),
        }
    if not files:
        raise ProducerError("public tokenizer has no readable metadata files")
    return {"path": str(path), "files": files}


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
    values: list[int] = []
    for token in output:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ProducerError("tokenizer returned a non-integer ID")
        if token < 0 or token >= VOCAB_SIZE:
            raise ProducerError("tokenizer returned an out-of-vocabulary ID")
        values.append(int(token))
    return values


def _sequence_digest(token_ids: Sequence[int]) -> str:
    return _sha256_bytes(
        torch.tensor(list(token_ids), dtype=torch.int32).numpy().tobytes(order="C")
    )


def _text_value(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProducerError(f"public row field {key!r} is not text")
    return value


def _source_spec(style: str) -> Mapping[str, Any]:
    if style not in SOURCE_PARTITIONS:
        raise ProducerError(f"fresh source style is not reserved: {style}")
    return SOURCE_PARTITIONS[style]


def _render_pile(row: Mapping[str, Any], index: int, tokenizer: Any) -> FreshRecord:
    text_value = _text_value(row, "text")
    digest = _sha256_bytes(text_value.encode("utf-8"))
    try:
        ids = _tokenizer_ids(tokenizer(text_value, add_special_tokens=False))
    except Exception as exc:
        raise ProducerError(f"Pile row {index} tokenization failed") from exc
    if ids and ids[0] == BOS_TOKEN_ID:
        ids = ids[1:]
    ids = [BOS_TOKEN_ID, *ids]
    if len(ids) < SEQUENCE_TOKENS:
        raise ProducerError(f"Pile row {index} is shorter than 128 declared tokens")
    spec = _source_spec("pile")
    record_id = source_record_id(
        str(spec["dataset_id"]),
        str(spec["split"]),
        str(spec["revision"]),
        index,
    )
    return FreshRecord(
        "pile",
        "pile",
        str(spec["dataset_id"]),
        str(spec["split"]),
        str(spec["revision"]),
        index,
        record_id,
        digest,
        tuple(ids),
        _sequence_digest(ids[:SEQUENCE_TOKENS]),
    )


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


def _render_finance(row: Mapping[str, Any], index: int, tokenizer: Any) -> FreshRecord:
    system, user, assistant = _finance_fields(row)
    if not user or not assistant:
        raise ProducerError(f"Finance row {index} has no user/assistant content")
    content = [system, user, assistant]
    digest = _sha256_bytes(_canonical_json(content).encode("utf-8"))
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
    if not ids or ids[0] != BOS_TOKEN_ID:
        raise ProducerError(f"Finance row {index} lost the declared BOS token")
    if len(ids) < SEQUENCE_TOKENS:
        raise ProducerError(f"Finance row {index} is shorter than 128 declared tokens")
    spec = _source_spec("finance")
    record_id = source_record_id(
        str(spec["dataset_id"]),
        str(spec["split"]),
        str(spec["revision"]),
        index,
    )
    return FreshRecord(
        "finance",
        "finance",
        str(spec["dataset_id"]),
        str(spec["split"]),
        str(spec["revision"]),
        index,
        record_id,
        digest,
        tuple(ids),
        _sequence_digest(ids[:SEQUENCE_TOKENS]),
    )


def _render_row(style: str, row: Mapping[str, Any], index: int, tokenizer: Any) -> FreshRecord:
    if not isinstance(row, Mapping):
        raise ProducerError(f"{style} public row {index} is malformed")
    return _render_pile(row, index, tokenizer) if style == "pile" else _render_finance(row, index, tokenizer)


def _load_arrow_dataset(paths: Sequence[Path]) -> Any:
    from datasets import Dataset, concatenate_datasets

    if not paths:
        raise ProducerError("one or more public Arrow caches are required")
    loaded = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise ProducerError(f"public Arrow cache is unavailable: {path}")
        try:
            loaded.append(Dataset.from_file(str(path)))
        except Exception as exc:
            raise ProducerError(f"public Arrow cache could not be loaded: {path}") from exc
    return loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)


def _dataset_descriptor(
    paths: Sequence[Path],
    *,
    style: str,
) -> dict[str, Any]:
    spec = _source_spec(style)
    return {
        "dataset_key": style,
        "dataset_id": str(spec["dataset_id"]),
        "split": str(spec["split"]),
        "revision": str(spec["revision"]),
        "arrow_files": [_file_record(path) for path in paths],
        "reserved_holdout": dict(spec),
    }


def _read_reserved_row(dataset: Any, *, style: str, row_index: int) -> Mapping[str, Any]:
    # This ordering is a security invariant: the dataset subscript is reached
    # only after the partition validator accepts the exact holdout index.
    try:
        validate_partition_index(style, row_index, role="holdout")
    except Exception as exc:
        raise ProducerError(
            f"{style} reserved row partition guard rejected {row_index}: {exc}"
        ) from exc
    try:
        row = dataset[row_index]
    except Exception as exc:
        raise ProducerError(f"{style} reserved row {row_index} could not be read") from exc
    if not isinstance(row, Mapping):
        raise ProducerError(f"{style} reserved row {row_index} is malformed")
    return row


def _select_domain(
    dataset: Any,
    *,
    style: str,
    tokenizer: Any,
    exclusions: ExclusionSets,
    seen_final_sequences: set[str],
) -> tuple[list[FreshRecord], dict[str, int]]:
    spec = _source_spec(style)
    stop = int(spec["holdout_reserve_stop"])
    if len(dataset) < stop:
        raise ProducerError(f"{style} cache has {len(dataset)} rows; need reserved stop {stop}")
    selected: list[FreshRecord] = []
    skipped = {
        "excluded_id": 0,
        "excluded_hash": 0,
        "excluded_index": 0,
        "invalid": 0,
        "duplicate_final_sequence": 0,
    }
    order = deterministic_row_order(
        range(int(spec["holdout_reserve_start"]), stop),
        dataset_key=f"{style}-future-holdout",
        seed=SELECTION_SEED,
    )
    for index in order:
        expected_id = source_record_id(
            str(spec["dataset_id"]),
            str(spec["split"]),
            str(spec["revision"]),
            index,
        )
        if expected_id in exclusions.ids[style]:
            skipped["excluded_id"] += 1
            continue
        if index in exclusions.indices[style]:
            skipped["excluded_index"] += 1
            continue
        row = _read_reserved_row(dataset, style=style, row_index=index)
        try:
            candidate = _render_row(style, row, index, tokenizer)
        except ProducerError:
            skipped["invalid"] += 1
            continue
        blocked = _blocked(candidate, exclusions)
        if blocked == "public_source_id":
            skipped["excluded_id"] += 1
            continue
        if blocked in {"public_rendered_hash", "public_final_sequence_hash"}:
            skipped["excluded_hash"] += 1
            continue
        if blocked == "public_source_index":
            skipped["excluded_index"] += 1
            continue
        if candidate.final_sequence_sha256 in seen_final_sequences:
            skipped["duplicate_final_sequence"] += 1
            continue
        seen_final_sequences.add(candidate.final_sequence_sha256)
        selected.append(candidate)
        if len(selected) == RECORDS_PER_DOMAIN:
            break
    if len(selected) != RECORDS_PER_DOMAIN:
        raise ProducerError(
            f"{style} reserved pool yielded {len(selected)} records; need {RECORDS_PER_DOMAIN}"
        )
    if len({row.record_id for row in selected}) != RECORDS_PER_DOMAIN:
        raise ProducerError(f"{style} selected source IDs are not unique")
    return selected, skipped


def _selection_plan(
    *,
    root: Path,
    freeze: Mapping[str, Any],
    decision_plan: Mapping[str, Any] | None,
    pile_descriptor: Mapping[str, Any],
    finance_descriptor: Mapping[str, Any],
    tokenizer_descriptor: Mapping[str, Any],
    selected: Mapping[str, Sequence[FreshRecord]],
    exclusions: ExclusionSets,
    skipped: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    commit = _git_commit(root)
    if commit is None:
        raise ProducerError("cannot bind the selection plan to a full code commit")
    selected_metadata = {
        style: [record.selection_metadata() for record in selected[style]]
        for style in STYLE_ORDER
    }
    selected_ids = {
        style: [record.record_id for record in selected[style]]
        for style in STYLE_ORDER
    }
    selected_hashes = {
        style: [record.final_sequence_sha256 for record in selected[style]]
        for style in STYLE_ORDER
    }
    return {
        "schema": SELECTION_SCHEMA,
        "producer_schema": PRODUCER_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_FRESH_SOURCE_SELECTION_NO_TRUTH",
        "selection_seed": SELECTION_SEED,
        "sequence_tokens": SEQUENCE_TOKENS,
        "records_per_domain": RECORDS_PER_DOMAIN,
        "source_ranges_half_open": {
            style: [
                int(SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
                int(SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
            ]
            for style in STYLE_ORDER
        },
        "target_conditions": list(CONDITION_ORDER),
        "paired_conditions": True,
        "public_sources_frozen": {
            "pile": dict(pile_descriptor),
            "finance": dict(finance_descriptor),
            "tokenizer": dict(tokenizer_descriptor),
        },
        "method_freeze": {
            "path": str(freeze["path"]),
            "bytes": int(Path(freeze["path"]).stat().st_size),
            "sha256": str(freeze["sha256"]),
            "status": freeze["status"],
            "code_commit": freeze["code_commit"],
            "decision_plan_sha256": freeze["decision_plan_sha256"],
            "method_ids": list(METHOD_IDS),
            "state_sha256": dict(freeze["state_sha256"]),
        },
        "method_freeze_sha256": str(freeze["sha256"]),
        "decision_plan": (
            {
                "path": str(Path(decision_plan["path"]).resolve()),
                "sha256": str(decision_plan["sha256"]),
            }
            if isinstance(decision_plan, Mapping)
            else {"sha256": freeze["decision_plan_sha256"]}
        ),
        "public_validation_selection": dict(freeze["public_validation_selection"]),
        "selection_rule": {
            "algorithm": (
                "Use deterministic row order over each reserved half-open pool; "
                "read only after the method freeze; reject known public identities, "
                "rendered hashes, and duplicate final 128-token sequences; retain "
                "the first 128 eligible rows per domain."
            ),
            "identity_exclusions": True,
            "duplicate_final_sequence_exclusion": True,
            "source_text_or_token_ids_written": False,
            "record_ids": selected_ids,
            "final_sequence_sha256": selected_hashes,
            "records": selected_metadata,
        },
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
        },
        "selection_diagnostics": {
            style: {
                **dict(skipped[style]),
                "selected": len(selected[style]),
                "pool_size": int(SOURCE_PARTITIONS[style]["holdout_reserve_stop"])
                - int(SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
            }
            for style in STYLE_ORDER
        },
        "execution": {
            "started_utc": _utc_now(),
            "code_commit": commit,
            "producer": str(Path(__file__).resolve()),
            "python": sys.executable,
            "network_used": False,
            "model_loaded": False,
            "truth_opened": False,
            "fresh_source_pool_contents_opened": True,
        },
    }


def _plan_records(plan: Mapping[str, Any], style: str) -> list[Mapping[str, Any]]:
    selection = plan.get("selection_rule")
    if not isinstance(selection, Mapping):
        raise ProducerError("selection plan has no selection rule")
    records = selection.get("records")
    if not isinstance(records, Mapping) or not isinstance(records.get(style), list):
        raise ProducerError(f"selection plan has no {style} records")
    values = records[style]
    if len(values) != RECORDS_PER_DOMAIN or any(not isinstance(row, Mapping) for row in values):
        raise ProducerError(f"selection plan has wrong {style} record count")
    return list(values)


def _validate_selection_plan(
    path: Path,
    *,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _load_json(path, description="fresh source selection plan")
    if plan.get("schema") != SELECTION_SCHEMA or plan.get("task_id") != TASK_ID:
        raise ProducerError("fresh selection plan schema or task ID changed")
    if plan.get("status") != "FROZEN_FRESH_SOURCE_SELECTION_NO_TRUTH":
        raise ProducerError("fresh selection plan is not frozen")
    if plan.get("selection_seed") != SELECTION_SEED:
        raise ProducerError("fresh selection seed changed")
    if plan.get("sequence_tokens") != SEQUENCE_TOKENS:
        raise ProducerError("fresh declared sequence geometry changed")
    if plan.get("records_per_domain") != RECORDS_PER_DOMAIN:
        raise ProducerError("fresh source count changed")
    expected_ranges = {
        style: [
            int(SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
            int(SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
        ]
        for style in STYLE_ORDER
    }
    if plan.get("source_ranges_half_open") != expected_ranges:
        raise ProducerError("fresh reserved source ranges changed")
    if plan.get("method_freeze_sha256") != freeze["sha256"]:
        raise ProducerError("selection plan is bound to a different method freeze")
    public_sources = plan.get("public_sources_frozen")
    if not isinstance(public_sources, Mapping):
        raise ProducerError("selection plan lacks frozen public source descriptors")
    for style in STYLE_ORDER:
        if not isinstance(public_sources.get(style), Mapping):
            raise ProducerError(f"selection plan lacks {style} source descriptor")
        rows = _plan_records(plan, style)
        ids = [row.get("record_id") for row in rows]
        if any(not isinstance(value, str) or not value for value in ids):
            raise ProducerError(f"{style} selection has an empty source ID")
        if len(set(ids)) != RECORDS_PER_DOMAIN:
            raise ProducerError(f"{style} selection has duplicate source IDs")
        hashes = []
        for row in rows:
            if set(row) - {
                "record_id",
                "public_record_sha256",
                "dataset_key",
                "dataset_id",
                "split",
                "revision",
                "row_index",
                "source_index",
                "full_token_count",
                "post_bos_token_count",
                "valid_tokens",
                "final_sequence_sha256",
            }:
                raise ProducerError(f"{style} selection contains private/unapproved metadata")
            if row.get("dataset_key") != style:
                raise ProducerError(f"{style} selection dataset key changed")
            index = row.get("row_index")
            if not isinstance(index, int):
                raise ProducerError(f"{style} selection row index is malformed")
            try:
                validate_partition_index(style, index, role="holdout")
            except Exception as exc:
                raise ProducerError(f"{style} selection row escaped the holdout range") from exc
            spec = _source_spec(style)
            expected_id = source_record_id(
                str(spec["dataset_id"]), str(spec["split"]), str(spec["revision"]), index
            )
            if row.get("record_id") != expected_id:
                raise ProducerError(f"{style} selection source ID changed")
            if row.get("dataset_id") != spec["dataset_id"] or row.get("split") != spec["split"] or row.get("revision") != spec["revision"]:
                raise ProducerError(f"{style} selection dataset revision changed")
            if row.get("source_index") != index:
                raise ProducerError(f"{style} selection source index changed")
            _require_sha(row.get("public_record_sha256"), description=f"{style} public row")
            _require_sha(row.get("final_sequence_sha256"), description=f"{style} final sequence")
            full_count = row.get("full_token_count")
            post_count = row.get("post_bos_token_count")
            if (not isinstance(full_count, int) or not isinstance(post_count, int)
                    or full_count < SEQUENCE_TOKENS or post_count != full_count - 1):
                raise ProducerError(f"{style} selection row length metadata changed")
            if row.get("valid_tokens") != SEQUENCE_TOKENS:
                raise ProducerError(f"{style} selection valid token count changed")
            hashes.append(row["final_sequence_sha256"])
        if len(set(hashes)) != len(hashes):
            raise ProducerError(f"{style} selection has duplicate final sequences")
    all_hashes = [
        row["final_sequence_sha256"]
        for style in STYLE_ORDER
        for row in _plan_records(plan, style)
    ]
    if len(set(all_hashes)) != len(all_hashes):
        raise ProducerError("fresh selection has duplicate final sequences across domains")
    return plan


def _materialize_selected(
    plan: Mapping[str, Any],
    *,
    style: str,
    dataset: Any,
    tokenizer: Any,
) -> list[FreshRecord]:
    expected = _plan_records(plan, style)
    result: list[FreshRecord] = []
    for declared in expected:
        index = int(declared["row_index"])
        row = _read_reserved_row(dataset, style=style, row_index=index)
        candidate = _render_row(style, row, index, tokenizer)
        checks = candidate.selection_metadata()
        for key in (
            "record_id",
            "public_record_sha256",
            "dataset_key",
            "dataset_id",
            "split",
            "revision",
            "row_index",
            "full_token_count",
            "post_bos_token_count",
            "final_sequence_sha256",
        ):
            if str(checks[key]) != str(declared.get(key)):
                raise ProducerError(f"{style} selected row {index} changed: {key}")
        result.append(candidate)
    if [row.record_id for row in result] != [
        str(row["record_id"]) for row in expected
    ]:
        raise ProducerError(f"{style} selected source order changed")
    return result


def _device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise ProducerError("CUDA requested but unavailable")
    if raw not in {"cpu", "cuda"}:
        raise ProducerError(f"unsupported capture device: {raw}")
    return torch.device(raw)


def _guard_helpers() -> tuple[Any, Any]:
    try:
        import trr0004_prepare_public_activations as prep

        return prep._resource_preflight, prep._enforce_resource_ceiling
    except Exception as exc:
        raise ProducerError("TRR4 public resource guard is unavailable") from exc


def _live_resource_guard(device: torch.device) -> dict[str, Any]:
    preflight, _ceiling = _guard_helpers()
    try:
        result = preflight(
            device,
            min_free_gpu_bytes=MIN_FREE_GPU_BYTES,
            max_reserved_gpu_bytes=MAX_RESERVED_GPU_BYTES,
            max_host_rss_bytes=MAX_HOST_RSS_BYTES,
        )
    except Exception as exc:
        raise ProducerError("public capture resource preflight failed") from exc
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except Exception as exc:
        raise ProducerError("live host memory guard is unavailable") from exc
    if available < 10 * 1024**3:
        raise ProducerError(f"only {available} host bytes are available; need at least 10 GiB")
    result["host_available_bytes"] = available
    result["minimum_host_available_bytes"] = 10 * 1024**3
    return result


def _capture_prefix(
    *,
    condition: str,
    model_snapshot: Path,
    lora_config_path: Path | None,
    lora_update: Path | None,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    import trr0004_produce_confirmation as trr4

    _live_resource_guard(device)
    if condition == "public_base":
        prefix, evidence = trr4._load_base_prefix(model_snapshot, device=device)
    else:
        if lora_config_path is None or lora_update is None:
            raise ProducerError("public_lora_2601 requires both config and update")
        config, normalized = trr4._load_lora_config(lora_config_path)
        if lora_update.is_symlink() or not lora_update.is_file():
            raise ProducerError("public_lora_2601 update is unavailable")
        prefix, evidence = trr4._load_shifted_prefix(
            model_snapshot,
            device=device,
            lora_config=config,
            lora_update=lora_update.expanduser().resolve(),
        )
        evidence["lora_config_path"] = str(lora_config_path.expanduser().resolve())
        evidence["lora_config"] = normalized
    return prefix, evidence


def _save_compact_observation(
    path: Path,
    *,
    activations: torch.Tensor,
    batch: Any,
    cell_id: str,
    capture_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise ProducerError(f"observation artifact is create-only: {path}")
    compact = activations[:, :SEQUENCE_TOKENS].contiguous()
    mask = batch.attention_mask[:, :SEQUENCE_TOKENS].to(torch.uint8).contiguous()
    positions = batch.position_ids[:, :SEQUENCE_TOKENS].to(torch.int64).contiguous()
    if tuple(compact.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise ProducerError("fresh observation geometry changed")
    if not mask.eq(1).all().item():
        raise ProducerError("fresh records are not full 128-token rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": OBSERVATION_SCHEMA,
        "task_id": TASK_ID,
        "cell_id": cell_id,
        "shape": str(list(compact.shape)),
        "cut_depth": str(CUT_DEPTH),
        "hidden_size": str(HIDDEN_SIZE),
        "sequence_tokens": str(SEQUENCE_TOKENS),
        "capture_sequence_tokens": str(CAPTURE_SEQUENCE_TOKENS),
        "batch_records": str(CAPTURE_BATCH_SIZE),
        "public_full_forward": "true",
        "target_truth_accessed": "false",
        "producer_only_lora": str(cell_id.endswith("__public_lora_2601")).lower(),
        "capture_metadata": _canonical_json(capture_metadata),
    }
    save_file(
        {
            "activations": compact.detach().cpu().contiguous(),
            "attention_mask": mask,
            "position_ids": positions,
        },
        str(path),
        metadata=metadata,
    )
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "tensor_key": "activations",
        "shape": [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, HIDDEN_SIZE],
        "attention_mask_key": "attention_mask",
        "position_ids_key": "position_ids",
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "public_full_forward": True,
        "producer_only_lora": cell_id.endswith("__public_lora_2601"),
    }


def _capture_condition(
    *,
    condition: str,
    records: Mapping[str, Sequence[FreshRecord]],
    model_snapshot: Path,
    lora_config_path: Path | None,
    lora_update: Path | None,
    raw_root: Path,
    manifest_root: Path,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    prefix, load_evidence = _capture_prefix(
        condition=condition,
        model_snapshot=model_snapshot,
        lora_config_path=lora_config_path,
        lora_update=lora_update,
        device=device,
    )
    batches = {
        style: pad_public_token_sequences(
            [list(record.token_ids[:SEQUENCE_TOKENS]) for record in records[style]],
            maximum_tokens=CAPTURE_SEQUENCE_TOKENS,
            pad_token_id=PADDING_TOKEN_ID,
            bos_token_id=BOS_TOKEN_ID,
            vocab_size=VOCAB_SIZE,
        )
        for style in STYLE_ORDER
    }
    # The largest representative cell is qualified at exactly the proven
    # padded 8-by-192 shape before any observation is emitted.
    import trr0004_prepare_public_activations as prep

    qualification = prep._qualify_public_prefix_padding(
        prefix,
        batches["finance"],
        device=device,
        batch_size=CAPTURE_BATCH_SIZE,
    )
    _live_resource_guard(device)
    observations: dict[str, dict[str, Any]] = {}
    capture_rows: dict[str, Any] = {
        "condition": condition,
        "qualification": qualification,
        "load": load_evidence,
        "styles": {},
    }
    _preflight, ceiling = _guard_helpers()
    for style in STYLE_ORDER:
        cell_id = f"{style}__{condition}"
        started = time.perf_counter()
        batch = batches[style]
        activations = capture_public_prefix(
            prefix,
            batch,
            device=device,
            batch_size=CAPTURE_BATCH_SIZE,
            resource_check=lambda: ceiling(
                device,
                max_reserved_gpu_bytes=MAX_RESERVED_GPU_BYTES,
                max_host_rss_bytes=MAX_HOST_RSS_BYTES,
            ),
        )
        if tuple(activations.shape) != (
            RECORDS_PER_DOMAIN,
            CAPTURE_SEQUENCE_TOKENS,
            HIDDEN_SIZE,
        ):
            raise ProducerError(f"{cell_id} capture geometry changed")
        raw_path = raw_root / f"{cell_id}.padded192.safetensors"
        save_public_artifact(
            raw_path,
            activations=activations,
            token_batch=batch,
            metadata={
                "schema": CAPTURE_SCHEMA,
                "task_id": TASK_ID,
                "cell_id": cell_id,
                "public_full_forward": "true",
                "target_truth_accessed": "false",
                "producer_only_lora": str(condition == "public_lora_2601").lower(),
                "capture_batch_records": str(CAPTURE_BATCH_SIZE),
                "capture_sequence_tokens": str(CAPTURE_SEQUENCE_TOKENS),
            },
        )
        observation_path = manifest_root / "observations" / f"{cell_id}.safetensors"
        observation = _save_compact_observation(
            observation_path,
            activations=activations,
            batch=batch,
            cell_id=cell_id,
            capture_metadata={"raw_path": str(raw_path.resolve())},
        )
        observation["raw_public_artifact"] = {
            "path": str(raw_path.resolve()),
            "bytes": int(raw_path.stat().st_size),
            "sha256": _sha256_file(raw_path),
            "shape": [RECORDS_PER_DOMAIN, CAPTURE_SEQUENCE_TOKENS, HIDDEN_SIZE],
        }
        observations[cell_id] = observation
        capture_rows["styles"][style] = {
            "seconds": time.perf_counter() - started,
            "raw_artifact": observation["raw_public_artifact"],
            "observation": observation,
        }
        _live_resource_guard(device)
    del prefix
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return observations, capture_rows


def _build_panel(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    records: Mapping[str, Sequence[FreshRecord]],
    observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    full_mask = [[1] * SEQUENCE_TOKENS for _ in range(RECORDS_PER_DOMAIN)]
    positions = [list(range(SEQUENCE_TOKENS)) for _ in range(RECORDS_PER_DOMAIN)]
    for style in STYLE_ORDER:
        for condition in CONDITION_ORDER:
            cell_id = f"{style}__{condition}"
            observation = observations.get(cell_id)
            if not isinstance(observation, Mapping):
                raise ProducerError(f"missing observation descriptor: {cell_id}")
            cells[cell_id] = {
                "cell_id": cell_id,
                "style": style,
                "condition": condition,
                "records": [record.panel_metadata() for record in records[style]],
                "attention_mask": full_mask,
                "position_ids": positions,
                "observation": dict(observation),
            }
    panel = {
        "schema": PANEL_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_FRESH_CONFIRMATION_PANEL",
        "panel_id": "trr0005-fresh-public-v1",
        "sequence_tokens": SEQUENCE_TOKENS,
        "records_per_domain": RECORDS_PER_DOMAIN,
        "hidden_size": HIDDEN_SIZE,
        "cut_depth": CUT_DEPTH,
        "styles": list(STYLE_ORDER),
        "conditions": list(CONDITION_ORDER),
        "cells": cells,
        "selection_plan": {
            "path": str(plan_path.expanduser().resolve()),
            "bytes": int(plan_path.stat().st_size),
            "sha256": _sha256_file(plan_path),
        },
        "method_freeze_sha256": plan.get("method_freeze_sha256"),
        "public_validation_selection": dict(plan["public_validation_selection"]),
        "public_material_only": True,
        "truth_opened": False,
        "observation_contract": {
            "shape": [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, HIDDEN_SIZE],
            "cut_depth": CUT_DEPTH,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "capture_batch_records": CAPTURE_BATCH_SIZE,
            "public_full_forward": True,
            "lora_condition": "producer_only",
        },
    }
    try:
        validate_panel_descriptor(panel)
    except ContractError as exc:
        raise ProducerError(f"constructed panel failed contract validation: {exc}") from exc
    return panel


def _frozen_source_paths(plan: Mapping[str, Any], style: str) -> tuple[Path, ...]:
    sources = plan.get("public_sources_frozen")
    descriptor = sources.get(style) if isinstance(sources, Mapping) else None
    files = descriptor.get("arrow_files") if isinstance(descriptor, Mapping) else None
    if not isinstance(files, list) or not files:
        raise ProducerError(f"selection plan has no {style} Arrow files")
    paths: list[Path] = []
    for row in files:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ProducerError(f"selection plan {style} Arrow descriptor is malformed")
        paths.append(Path(row["path"]))
    return tuple(paths)


def _validate_frozen_source_descriptors(
    plan: Mapping[str, Any],
    *,
    pile_paths: Sequence[Path],
    finance_paths: Sequence[Path],
    tokenizer_path: Path,
) -> None:
    actual = {
        "pile": _dataset_descriptor(pile_paths, style="pile"),
        "finance": _dataset_descriptor(finance_paths, style="finance"),
        "tokenizer": _tokenizer_descriptor(tokenizer_path),
    }
    frozen = plan.get("public_sources_frozen")
    if not isinstance(frozen, Mapping):
        raise ProducerError("selection plan has no frozen source descriptors")
    for key, value in actual.items():
        if dict(frozen.get(key, {})) != dict(value):
            raise ProducerError(f"{key} source descriptor differs from the frozen selection input")


def select_public(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()

    # This call is intentionally before tokenizer loading, Arrow loading, or
    # any dataset subscript. It is the first operation that can authorize a
    # reserved source read.
    freeze = _validate_preselection(
        args.method_freeze,
        decision_plan_path=args.decision_plan,
        public_validation_selection_path=args.public_validation_selection,
    )
    output_path = args.output.expanduser().resolve()
    try:
        output_path.relative_to(root / "experiments" / "TRR-0005")
    except ValueError as exc:
        raise ProducerError("selection plan output must be under experiments/TRR-0005") from exc
    if output_path.exists() or output_path.is_symlink():
        raise ProducerError(f"selection plan is create-only and already exists: {output_path}")
    decision_plan: dict[str, Any] | None = None
    if args.decision_plan is not None:
        decision_path = args.decision_plan.expanduser().resolve()
        decision_plan = {
            "path": str(decision_path),
            "sha256": _sha256_file(decision_path),
        }

    tokenizer_path = args.tokenizer.expanduser().resolve()
    tokenizer = _load_tokenizer(tokenizer_path)
    pile_paths = tuple(path.expanduser().resolve() for path in args.pile_arrow)
    finance_paths = tuple(path.expanduser().resolve() for path in args.finance_arrow)
    pile_descriptor = _dataset_descriptor(pile_paths, style="pile")
    finance_descriptor = _dataset_descriptor(finance_paths, style="finance")
    tokenizer_descriptor = _tokenizer_descriptor(tokenizer_path)
    pile_dataset = _load_arrow_dataset(pile_paths)
    finance_dataset = _load_arrow_dataset(finance_paths)

    exclusion_paths = _default_exclusion_paths(root)
    exclusion_paths.extend(path.expanduser().resolve() for path in args.exclude_source)
    exclusions = _collect_exclusions(exclusion_paths)
    seen_final: set[str] = set()
    pile, pile_skipped = _select_domain(
        pile_dataset,
        style="pile",
        tokenizer=tokenizer,
        exclusions=exclusions,
        seen_final_sequences=seen_final,
    )
    finance, finance_skipped = _select_domain(
        finance_dataset,
        style="finance",
        tokenizer=tokenizer,
        exclusions=exclusions,
        seen_final_sequences=seen_final,
    )
    plan = _selection_plan(
        root=root,
        freeze=freeze,
        decision_plan=decision_plan,
        pile_descriptor=pile_descriptor,
        finance_descriptor=finance_descriptor,
        tokenizer_descriptor=tokenizer_descriptor,
        selected={"pile": pile, "finance": finance},
        exclusions=exclusions,
        skipped={"pile": pile_skipped, "finance": finance_skipped},
    )
    _write_create_only(output_path, plan)
    return {
        "task_id": TASK_ID,
        "status": plan["status"],
        "selection_plan": str(output_path),
        "selection_plan_sha256": _sha256_file(output_path),
        "records_per_domain": RECORDS_PER_DOMAIN,
        "truth_opened": False,
    }


def capture_public(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    freeze = _validate_preselection(
        args.method_freeze,
        decision_plan_path=args.decision_plan,
        public_validation_selection_path=args.public_validation_selection,
    )
    plan_path = args.selection_plan.expanduser().resolve()
    plan = _validate_selection_plan(plan_path, freeze=freeze)

    tokenizer_path = args.tokenizer.expanduser().resolve()
    pile_paths = tuple(
        path.expanduser().resolve() for path in (args.pile_arrow or _frozen_source_paths(plan, "pile"))
    )
    finance_paths = tuple(
        path.expanduser().resolve() for path in (args.finance_arrow or _frozen_source_paths(plan, "finance"))
    )
    _validate_frozen_source_descriptors(
        plan,
        pile_paths=pile_paths,
        finance_paths=finance_paths,
        tokenizer_path=tokenizer_path,
    )
    tokenizer = _load_tokenizer(tokenizer_path)
    pile_dataset = _load_arrow_dataset(pile_paths)
    finance_dataset = _load_arrow_dataset(finance_paths)
    records = {
        "pile": _materialize_selected(
            plan, style="pile", dataset=pile_dataset, tokenizer=tokenizer
        ),
        "finance": _materialize_selected(
            plan, style="finance", dataset=finance_dataset, tokenizer=tokenizer
        ),
    }

    raw_root = args.raw_root.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    try:
        raw_root.relative_to(root / "outputs" / "TRR-0005")
        manifest_root.relative_to(root / "experiments" / "TRR-0005")
    except ValueError as exc:
        raise ProducerError("capture roots are outside their task-local locations") from exc
    raw_root = _new_repo_root(raw_root, root, description="raw capture root")
    manifest_root = _new_repo_root(manifest_root, root, description="capture manifest root")
    started = time.perf_counter()
    captures: dict[str, Any] = {}
    observations: dict[str, dict[str, Any]] = {}
    try:
        for condition in CONDITION_ORDER:
            condition_observations, condition_capture = _capture_condition(
                condition=condition,
                records=records,
                model_snapshot=args.model_snapshot.expanduser().resolve(),
                lora_config_path=args.lora_config,
                lora_update=args.lora_update,
                raw_root=raw_root,
                manifest_root=manifest_root,
                device=_device(args.device),
            )
            observations.update(condition_observations)
            captures[condition] = condition_capture
        panel = _build_panel(
            plan=plan,
            plan_path=plan_path,
            records=records,
            observations=observations,
        )
        panel_path = manifest_root / "panel.json"
        observation_manifest_path = manifest_root / "observations.json"
        capture_path = manifest_root / "capture.json"
        _write_create_only(panel_path, panel)
        _write_create_only(
            observation_manifest_path,
            {
                "schema": OBSERVATION_SCHEMA,
                "task_id": TASK_ID,
                "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
                "observations": observations,
            },
        )
        capture = {
            "schema": CAPTURE_SCHEMA,
            "task_id": TASK_ID,
            "status": "PUBLIC_FRESH_CAPTURE_COMPLETE_NO_TRUTH",
            "selection_plan": {
                "path": str(plan_path),
                "bytes": int(plan_path.stat().st_size),
                "sha256": _sha256_file(plan_path),
            },
            "method_freeze_sha256": freeze["sha256"],
            "panel": {
                "path": str(panel_path),
                "bytes": int(panel_path.stat().st_size),
                "sha256": _sha256_file(panel_path),
            },
            "observations": {
                "path": str(observation_manifest_path),
                "bytes": int(observation_manifest_path.stat().st_size),
                "sha256": _sha256_file(observation_manifest_path),
            },
            "conditions": captures,
            "elapsed_seconds": time.perf_counter() - started,
            "target_truth_accessed": False,
            "truth_opened": False,
            "code_commit": _git_commit(root),
        }
        _write_create_only(capture_path, capture)
    except Exception as exc:
        failure_path = manifest_root / "capture_failure.json"
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_create_only(
                failure_path,
                {
                    "schema": CAPTURE_SCHEMA,
                    "task_id": TASK_ID,
                    "status": "PUBLIC_FRESH_CAPTURE_FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "selection_plan_sha256": _sha256_file(plan_path),
                    "method_freeze_sha256": freeze["sha256"],
                    "truth_opened": False,
                },
            )
        raise
    return {
        "task_id": TASK_ID,
        "status": capture["status"],
        "panel": str(panel_path),
        "observations": str(observation_manifest_path),
        "capture": str(capture_path),
        "truth_opened": False,
    }


def _record_id_digest(records: Sequence[FreshRecord]) -> str:
    return _json_sha256([record.record_id for record in records])


def prepare_truth(args: argparse.Namespace) -> dict[str, Any]:
    """Write the evaluator-side public labels with explicit file bindings."""

    root = args.repository_root.expanduser().resolve()
    freeze = _validate_preselection(
        args.method_freeze,
        decision_plan_path=args.decision_plan,
        public_validation_selection_path=args.public_validation_selection,
    )
    plan_path = args.selection_plan.expanduser().resolve()
    plan = _validate_selection_plan(plan_path, freeze=freeze)
    panel_path = args.panel.expanduser().resolve()
    panel = _load_json(panel_path, description="fresh panel")
    try:
        validate_panel_descriptor(panel)
    except ContractError as exc:
        raise ProducerError(f"fresh panel is invalid: {exc}") from exc
    if panel.get("selection_plan", {}).get("sha256") != _sha256_file(plan_path):
        raise ProducerError("panel is bound to a different selection plan")

    tokenizer_path = args.tokenizer.expanduser().resolve()
    pile_paths = tuple(
        path.expanduser().resolve() for path in (args.pile_arrow or _frozen_source_paths(plan, "pile"))
    )
    finance_paths = tuple(
        path.expanduser().resolve() for path in (args.finance_arrow or _frozen_source_paths(plan, "finance"))
    )
    _validate_frozen_source_descriptors(
        plan,
        pile_paths=pile_paths,
        finance_paths=finance_paths,
        tokenizer_path=tokenizer_path,
    )
    tokenizer = _load_tokenizer(tokenizer_path)
    records = {
        "pile": _materialize_selected(
            plan,
            style="pile",
            dataset=_load_arrow_dataset(pile_paths),
            tokenizer=tokenizer,
        ),
        "finance": _materialize_selected(
            plan,
            style="finance",
            dataset=_load_arrow_dataset(finance_paths),
            tokenizer=tokenizer,
        ),
    }
    for style in STYLE_ORDER:
        expected = tuple(
            row["record_id"] for row in panel["cells"][f"{style}__public_base"]["records"]
        )
        if expected != tuple(record.record_id for record in records[style]):
            raise ProducerError(f"truth row order does not match panel for {style}")

    tensors: dict[str, torch.Tensor] = {}
    record_digests: dict[str, str] = {}
    for style in STYLE_ORDER:
        labels = torch.tensor(
            [list(record.token_ids[:SEQUENCE_TOKENS]) for record in records[style]],
            dtype=torch.int64,
        )
        if tuple(labels.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
            raise ProducerError(f"truth label geometry changed: {style}")
        mask = torch.ones_like(labels, dtype=torch.uint8)
        positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.int64).repeat(
            RECORDS_PER_DOMAIN, 1
        )
        record_digests[style] = _record_id_digest(records[style])
        for condition in CONDITION_ORDER:
            cell_id = f"{style}__{condition}"
            tensors[f"{cell_id}__token_ids"] = labels
            tensors[f"{cell_id}__attention_mask"] = mask
            tensors[f"{cell_id}__position_ids"] = positions

    truth_path = _outside_repo_destination(
        args.truth_output,
        root,
        description="truth sidecar",
    )
    metadata = {
        "schema": TRUTH_SCHEMA,
        "task_id": TASK_ID,
        "panel_sha256": _sha256_file(panel_path),
        "selection_plan_sha256": _sha256_file(plan_path),
        "method_freeze_sha256": freeze["sha256"],
        "record_ids_sha256_pile": record_digests["pile"],
        "record_ids_sha256_finance": record_digests["finance"],
        "record_ids_pile": _canonical_json([record.record_id for record in records["pile"]]),
        "record_ids_finance": _canonical_json([record.record_id for record in records["finance"]]),
        "truth_source": "public selected rows; evaluator-side labels only",
        "truth_opened": "false",
    }
    save_file(tensors, str(truth_path), metadata=metadata)
    truth_record = {
        "path": str(truth_path),
        "bytes": int(truth_path.stat().st_size),
        "sha256": _sha256_file(truth_path),
    }
    observation_hashes = {
        cell_id: panel["cells"][cell_id]["observation"]["sha256"]
        for cell_id in EXPECTED_CELL_IDS
    }
    truth_manifest = {
        "schema": TRUTH_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_TRUTH_PREPARED_OUTSIDE_RECONSTRUCTION_ROOT",
        "truth_file": truth_record,
        "panel": {
            "path": str(panel_path),
            "bytes": int(panel_path.stat().st_size),
            "sha256": _sha256_file(panel_path),
        },
        "selection_plan": {
            "path": str(plan_path),
            "bytes": int(plan_path.stat().st_size),
            "sha256": _sha256_file(plan_path),
        },
        "method_freeze_sha256": freeze["sha256"],
        "observation_sha256": observation_hashes,
        "record_ids_sha256": record_digests,
        "record_ids": {
            style: [record.record_id for record in records[style]]
            for style in STYLE_ORDER
        },
        "cell_order": list(EXPECTED_CELL_IDS),
        "truth_tensor_keys": sorted(tensors),
        "truth_opened": False,
        "reconstruction_root_contains_truth": False,
    }
    manifest_path = (
        args.truth_manifest
        if args.truth_manifest is not None
        else truth_path.with_name(truth_path.stem + ".manifest.json")
    )
    manifest_path = _outside_repo_destination(
        manifest_path,
        root,
        description="truth binding manifest",
    )
    _write_create_only(manifest_path, truth_manifest)
    return {
        "task_id": TASK_ID,
        "status": truth_manifest["status"],
        "truth_file": truth_record,
        "truth_manifest": {
            "path": str(manifest_path),
            "bytes": int(manifest_path.stat().st_size),
            "sha256": _sha256_file(manifest_path),
        },
        "truth_opened": False,
    }


def validate_truth_binding(
    manifest_path: Path,
    *,
    panel_path: Path,
    selection_plan_path: Path,
    truth_path: Path,
) -> dict[str, Any]:
    """Validate the producer truth descriptor after the public matrix gate.

    Footing should call this only after its complete public prediction/timing
    gate has passed. The check binds the actual sidecar bytes, panel and plan
    bytes, every observation digest, cell order, tensor keys, and the ordered
    panel record IDs stored in the safetensors metadata. A same-shaped sidecar
    with swapped rows or different panel/plan bindings therefore fails closed.
    """

    manifest = _load_json(manifest_path, description="truth binding manifest")
    if manifest.get("schema") != TRUTH_SCHEMA or manifest.get("task_id") != TASK_ID:
        raise ProducerError("truth binding manifest identity changed")
    if manifest.get("status") != "PUBLIC_TRUTH_PREPARED_OUTSIDE_RECONSTRUCTION_ROOT":
        raise ProducerError("truth binding manifest is not prepared")
    truth_path = truth_path.expanduser().resolve()
    truth_record = manifest.get("truth_file")
    if not isinstance(truth_record, Mapping) or truth_record.get("path") != str(truth_path):
        raise ProducerError("truth sidecar path differs from its binding manifest")
    actual_truth = _file_record(truth_path)
    if dict(truth_record) != actual_truth:
        raise ProducerError("truth sidecar bytes or hash differ from its binding manifest")

    panel_path = panel_path.expanduser().resolve()
    plan_path = selection_plan_path.expanduser().resolve()
    panel = _load_json(panel_path, description="fresh panel")
    validate_panel_descriptor(panel)
    panel_record = _file_record(panel_path)
    plan_record = _file_record(plan_path)
    declared_panel = manifest.get("panel")
    declared_plan = manifest.get("selection_plan")
    if not isinstance(declared_panel, Mapping) or dict(declared_panel) != panel_record:
        raise ProducerError("truth sidecar is bound to a different panel")
    if not isinstance(declared_plan, Mapping) or dict(declared_plan) != plan_record:
        raise ProducerError("truth sidecar is bound to a different selection plan")
    if manifest.get("method_freeze_sha256") != panel.get("method_freeze_sha256"):
        raise ProducerError("truth sidecar is bound to a different method freeze")
    if manifest.get("cell_order") != list(EXPECTED_CELL_IDS):
        raise ProducerError("truth sidecar cell order changed")

    expected_ids: dict[str, list[str]] = {}
    expected_observation_hashes: dict[str, str] = {}
    for cell_id in EXPECTED_CELL_IDS:
        cell = panel["cells"].get(cell_id)
        if not isinstance(cell, Mapping):
            raise ProducerError(f"truth sidecar panel cell is absent: {cell_id}")
        expected_ids[cell_id] = [row["record_id"] for row in cell["records"]]
        observation = cell.get("observation")
        if not isinstance(observation, Mapping) or not isinstance(observation.get("sha256"), str):
            raise ProducerError(f"panel observation binding is absent: {cell_id}")
        expected_observation_hashes[cell_id] = str(observation["sha256"])
    if manifest.get("observation_sha256") != expected_observation_hashes:
        raise ProducerError("truth sidecar observation bindings changed")
    expected_style_ids = {
        style: expected_ids[f"{style}__public_base"] for style in STYLE_ORDER
    }
    if manifest.get("record_ids") != expected_style_ids:
        raise ProducerError("truth sidecar ordered record IDs differ from the panel")
    expected_record_digests = {
        style: _json_sha256(expected_style_ids[style]) for style in STYLE_ORDER
    }
    if manifest.get("record_ids_sha256") != expected_record_digests:
        raise ProducerError("truth sidecar record-ID digests differ from the panel")

    try:
        with safe_open(truth_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            expected_keys = {
                f"{cell_id}__{suffix}"
                for cell_id in EXPECTED_CELL_IDS
                for suffix in ("token_ids", "attention_mask", "position_ids")
            }
            if keys != expected_keys:
                raise ProducerError("truth sidecar tensor keys or cell order changed")
            metadata = dict(handle.metadata() or {})
            if metadata.get("panel_sha256") != panel_record["sha256"]:
                raise ProducerError("truth tensor panel binding changed")
            if metadata.get("selection_plan_sha256") != plan_record["sha256"]:
                raise ProducerError("truth tensor selection-plan binding changed")
            if metadata.get("method_freeze_sha256") != manifest.get("method_freeze_sha256"):
                raise ProducerError("truth tensor method-freeze binding changed")
            for style in STYLE_ORDER:
                raw = metadata.get(f"record_ids_{style}")
                if not isinstance(raw, str):
                    raise ProducerError(f"truth tensor record-ID metadata is absent: {style}")
                try:
                    observed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ProducerError(f"truth tensor record-ID metadata is invalid: {style}") from exc
                if observed != expected_style_ids[style]:
                    raise ProducerError(f"truth tensor row order differs from panel: {style}")
                if metadata.get(f"record_ids_sha256_{style}") != _json_sha256(expected_style_ids[style]):
                    raise ProducerError(f"truth tensor record-ID digest differs from panel: {style}")
            for cell_id in EXPECTED_CELL_IDS:
                labels = handle.get_tensor(f"{cell_id}__token_ids")
                mask = handle.get_tensor(f"{cell_id}__attention_mask")
                positions = handle.get_tensor(f"{cell_id}__position_ids")
                if tuple(labels.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
                    raise ProducerError(f"truth tensor geometry changed: {cell_id}")
                if tuple(mask.shape) != tuple(labels.shape) or tuple(positions.shape) != tuple(labels.shape):
                    raise ProducerError(f"truth tensor auxiliary geometry changed: {cell_id}")
    except ProducerError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ProducerError("truth sidecar could not be validated") from exc
    return {
        "task_id": TASK_ID,
        "status": "TRUTH_BINDING_VALIDATED_AFTER_PUBLIC_GATE",
        "truth_file": actual_truth,
        "panel_sha256": panel_record["sha256"],
        "selection_plan_sha256": plan_record["sha256"],
        "cell_order": list(EXPECTED_CELL_IDS),
        "row_order_validated": True,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    freeze = _validate_preselection(
        args.method_freeze,
        decision_plan_path=args.decision_plan,
        public_validation_selection_path=args.public_validation_selection,
    )
    return {
        "task_id": TASK_ID,
        "status": "PREFLIGHT_METHOD_FREEZE_ACCEPTED_NO_SOURCE_READ",
        "method_freeze_sha256": freeze["sha256"],
        "method_ids": list(METHOD_IDS),
        "selection_seed": SELECTION_SEED,
        "source_ranges_half_open": {
            style: [
                int(SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
                int(SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
            ]
            for style in STYLE_ORDER
        },
        "records_per_domain": RECORDS_PER_DOMAIN,
        "sequence_tokens": SEQUENCE_TOKENS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "source_pool_contents_scanned": False,
        "truth_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def freeze_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--method-freeze", type=Path, required=True)
        command.add_argument("--decision-plan", type=Path)
        command.add_argument("--public-validation-selection", type=Path)

    check = sub.add_parser("preflight")
    check.add_argument("--repository-root", type=Path, default=Path("."))
    freeze_args(check)

    select = sub.add_parser("select")
    select.add_argument("--repository-root", type=Path, default=Path("."))
    freeze_args(select)
    select.add_argument("--tokenizer", type=Path, required=True)
    select.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    select.add_argument("--exclude-source", type=Path, nargs="*", default=[])
    select.add_argument("--output", type=Path, required=True)

    capture = sub.add_parser("capture")
    capture.add_argument("--repository-root", type=Path, default=Path("."))
    freeze_args(capture)
    capture.add_argument("--selection-plan", type=Path, required=True)
    capture.add_argument("--tokenizer", type=Path, required=True)
    capture.add_argument("--pile-arrow", type=Path, nargs="*")
    capture.add_argument("--finance-arrow", type=Path, nargs="*")
    capture.add_argument("--model-snapshot", type=Path, required=True)
    capture.add_argument("--lora-config", type=Path)
    capture.add_argument("--lora-update", type=Path)
    capture.add_argument("--raw-root", type=Path, required=True)
    capture.add_argument("--manifest-root", type=Path, required=True)
    capture.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    truth = sub.add_parser("truth")
    truth.add_argument("--repository-root", type=Path, default=Path("."))
    freeze_args(truth)
    truth.add_argument("--selection-plan", type=Path, required=True)
    truth.add_argument("--panel", type=Path, required=True)
    truth.add_argument("--tokenizer", type=Path, required=True)
    truth.add_argument("--pile-arrow", type=Path, nargs="*")
    truth.add_argument("--finance-arrow", type=Path, nargs="*")
    truth.add_argument("--truth-output", type=Path, required=True)
    truth.add_argument("--truth-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args)
    elif args.command == "select":
        result = select_public(args)
    elif args.command == "capture":
        result = capture_public(args)
    elif args.command == "truth":
        result = prepare_truth(args)
    else:  # pragma: no cover
        raise ProducerError(f"unknown producer command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProducerError, ContractError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0005 producer error: {exc}")

