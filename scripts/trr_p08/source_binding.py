"""Metadata-only P08 source and capture bindings.

This module verifies frozen descriptor files and approved opaque reservation
exports before any future source enumeration. It never loads a model, dataset
row, observation tensor, truth payload, or token array.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


class SourceBindingError(ValueError):
    """Raised when a P08 metadata binding violates its frozen contract."""


FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "answer",
        "answers",
        "correctness",
        "evaluator_truth",
        "input_ids",
        "labels",
        "plaintext",
        "private_truth",
        "source_text",
        "target_labels",
        "token_ids",
        "truth",
        "truth_ids",
        "truth_labels",
        "truth_tokens",
    }
)


def sha256_file(path: str | Path) -> str:
    """Hash one regular descriptor file, rejecting symlinks and directories."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise SourceBindingError(f"descriptor is not a regular file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_descriptor(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str = "descriptor",
) -> dict[str, Any]:
    """Verify a descriptor hash without interpreting its payload."""

    candidate = Path(path).expanduser()
    actual = sha256_file(candidate)
    if actual != expected_sha256:
        raise SourceBindingError(
            f"{label} SHA256 mismatch for {candidate}: expected {expected_sha256}, got {actual}"
        )
    return {
        "label": label,
        "path": str(candidate.resolve()),
        "bytes": candidate.stat().st_size,
        "sha256": actual,
    }


def reject_payload_keys(value: Any, *, path: str = "$") -> None:
    """Fail closed if metadata contains a source/truth payload branch."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in FORBIDDEN_PAYLOAD_KEYS:
                raise SourceBindingError(f"payload key {path}.{key} is forbidden")
            reject_payload_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_payload_keys(child, path=f"{path}[{index}]")


def load_json_descriptor(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str = "descriptor",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and parse a JSON metadata descriptor with payload-key checks."""

    reference = verify_descriptor(path, expected_sha256, label=label)
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceBindingError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SourceBindingError(f"{label} must be a JSON object")
    reject_payload_keys(value)
    return value, reference


def bind_opaque_reservation(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_schema: str,
    expected_counts: Mapping[str, int],
    label: str = "opaque reservation",
) -> dict[str, Any]:
    """Verify an approved hash-only reservation and return counts only."""

    value, reference = load_json_descriptor(path, expected_sha256, label=label)
    if value.get("schema") != expected_schema:
        raise SourceBindingError(
            f"{label} schema mismatch: {value.get('schema')!r}"
        )
    privacy = value.get("privacy_boundary")
    if not isinstance(privacy, Mapping) or privacy.get("hash_only") is not True:
        raise SourceBindingError(f"{label} is not explicitly hash-only")
    forbidden_flags = (
        "contains_model_weights",
        "contains_record_ids",
        "contains_source_indices",
        "contains_source_text",
        "contains_target_labels",
        "contains_token_ids",
        "contains_truth",
    )
    if any(privacy.get(flag) is True for flag in forbidden_flags):
        raise SourceBindingError(f"{label} privacy boundary permits forbidden data")
    hashes = value.get("hashes")
    if not isinstance(hashes, Mapping):
        raise SourceBindingError(f"{label} has no hash sets")
    counts = {}
    for key, expected in expected_counts.items():
        entries = hashes.get(key)
        # Published reservation exports use a metadata object for each hash
        # field. The actual reservation is the nested ``values`` array;
        # top-level mapping keys such as ``distinct_count`` are summaries and
        # must never be counted as hash values. Keep accepting the compact
        # list form for the synthetic/legacy contract, but bind both forms by
        # the actual values length.
        if isinstance(entries, Mapping):
            values = entries.get("values")
        else:
            values = entries
        if not isinstance(values, list) or len(values) != expected:
            raise SourceBindingError(
                f"{label} count mismatch for {key}: expected {expected}"
            )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in value)
            for value in values
        ):
            raise SourceBindingError(f"{label} has malformed hash values for {key}")
        counts[key] = len(values)
    return {
        "label": label,
        "path": reference["path"],
        "bytes": reference["bytes"],
        "sha256": reference["sha256"],
        "schema": expected_schema,
        "hash_only": True,
        "counts": counts,
    }


def bind_plan_descriptor(
    path: str | Path,
    expected_sha256: str,
    *,
    task_id: str = "TRR-P08",
) -> dict[str, Any]:
    """Bind the current P08 plan without accepting source-selection fields."""

    value, reference = load_json_descriptor(
        path, expected_sha256, label="P08 plan"
    )
    if value.get("task_id") != task_id:
        raise SourceBindingError("P08 plan task ID changed")
    if value.get("status") == "FROZEN_SOURCE_UNIVERSE":
        raise SourceBindingError(
            "source-universe freeze must be represented by a separate create-only receipt"
        )
    return {
        "label": "P08 plan",
        "path": reference["path"],
        "bytes": reference["bytes"],
        "sha256": reference["sha256"],
        "status": value.get("status"),
        "fresh_selection_started": False,
    }
