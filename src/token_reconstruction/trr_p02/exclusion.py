"""Capture exact public diagnostic cases for later confirmation exclusion.

This module is deliberately model-free.  It accepts only the explicitly known
public token sequences and identifiers used by a diagnostic plan, binds them to
the plan's content hash, and emits an immutable exclusion record.  The record
matches exact case/context/endpoint tuples; it never creates a global token or
token-type exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


DEFAULT_BOS_TOKEN_ID = 128000
DEFAULT_VOCAB_SIZE = 128256
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ExclusionCaptureError(ValueError):
    """Raised when a public diagnostic exclusion record is incomplete."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a regular, non-symlink source plan file."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ExclusionCaptureError(f"source plan must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExclusionCaptureError(f"{field} must be a non-empty string")
    return value


def _token_ids(
    values: Sequence[int],
    *,
    field: str,
    vocab_size: int,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ExclusionCaptureError(f"{field} must be a sequence of token IDs")
    if not values and not allow_empty:
        raise ExclusionCaptureError(f"{field} must be non-empty")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ExclusionCaptureError(f"{field} contains a non-integer token ID")
        if value < 0 or value >= int(vocab_size):
            raise ExclusionCaptureError(f"{field} contains an out-of-vocabulary token ID")
        result.append(int(value))
    return tuple(result)


@dataclass(frozen=True)
class PublicDiagnosticCase:
    """One fully specified public teacher-prefix diagnostic case.

    ``sequence_token_ids`` is required to equal the exact concatenation of the
    known ``context_token_ids`` and ``endpoint_token_ids``.  The endpoint IDs
    are identifiers for the public endpoint/query, while endpoint token IDs
    preserve the exact tokens supplied at that endpoint.
    """

    case_id: str
    context_id: str
    endpoint_id: str
    context_token_ids: tuple[int, ...]
    endpoint_token_ids: tuple[int, ...]
    sequence_token_ids: tuple[int, ...]
    position_ids: tuple[int, ...] = ()
    attention_mask: tuple[int, ...] = ()
    artifact_path: str | None = None
    artifact_sha256: str | None = None

    def validate(
        self,
        *,
        bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ) -> None:
        _identifier(self.case_id, field="case_id")
        _identifier(self.context_id, field="context_id")
        _identifier(self.endpoint_id, field="endpoint_id")
        context = _token_ids(
            self.context_token_ids,
            field="context_token_ids",
            vocab_size=vocab_size,
        )
        endpoint = _token_ids(
            self.endpoint_token_ids,
            field="endpoint_token_ids",
            vocab_size=vocab_size,
        )
        sequence = _token_ids(
            self.sequence_token_ids,
            field="sequence_token_ids",
            vocab_size=vocab_size,
        )
        if context[0] != int(bos_token_id):
            raise ExclusionCaptureError("context_token_ids must begin with the declared BOS")
        if sequence != context + endpoint:
            raise ExclusionCaptureError(
                "sequence_token_ids must equal context_token_ids + endpoint_token_ids"
            )
        if self.position_ids:
            positions = _token_ids(
                self.position_ids,
                field="position_ids",
                vocab_size=2**63 - 1,
            )
            if positions != tuple(range(len(sequence))):
                raise ExclusionCaptureError("position_ids must cover the exact sequence from zero")
        if self.attention_mask:
            mask = _token_ids(
                self.attention_mask,
                field="attention_mask",
                vocab_size=2,
            )
            if len(mask) != len(sequence) or any(value != 1 for value in mask):
                raise ExclusionCaptureError("attention_mask must be all ones for this unpadded case")
        if self.artifact_sha256 is not None and not _SHA256_RE.fullmatch(self.artifact_sha256):
            raise ExclusionCaptureError("artifact_sha256 must be a lowercase SHA-256 digest")

    def as_dict(
        self,
        *,
        bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ) -> dict[str, Any]:
        self.validate(bos_token_id=bos_token_id, vocab_size=vocab_size)
        return {
            "case_id": self.case_id,
            "context_id": self.context_id,
            "endpoint_id": self.endpoint_id,
            "context_token_ids": list(self.context_token_ids),
            "endpoint_token_ids": list(self.endpoint_token_ids),
            "sequence_token_ids": list(self.sequence_token_ids),
            "sequence_length": len(self.sequence_token_ids),
            "position_ids": list(self.position_ids),
            "attention_mask": list(self.attention_mask),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }


def capture_exclusion_record(
    source_plan_path: Path,
    cases: Sequence[PublicDiagnosticCase],
    *,
    generated_utc: str | None = None,
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    model_revision: str = "9213176726f574b556790deb65791e0c5aa438b6",
    cut_depth: int = 4,
    bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> dict[str, Any]:
    """Bind exact public cases to the immutable source-plan hash.

    This function performs no model access and accepts no source truth.  The
    caller must provide the exact public token IDs captured by the diagnostic
    runner before any later confirmation truth is opened.
    """

    source_plan_path = Path(source_plan_path)
    source_hash = sha256_file(source_plan_path)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, PublicDiagnosticCase):
            raise ExclusionCaptureError("cases must contain PublicDiagnosticCase values")
        case.validate(bos_token_id=bos_token_id, vocab_size=vocab_size)
        if case.case_id in seen:
            raise ExclusionCaptureError(f"duplicate case_id: {case.case_id}")
        seen.add(case.case_id)
        rows.append(case.as_dict(bos_token_id=bos_token_id, vocab_size=vocab_size))
    timestamp = generated_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(timestamp, str) or not timestamp:
        raise ExclusionCaptureError("generated_utc must be a non-empty string")
    return {
        "schema": "token-reconstruction.trr-p02.public-diagnostic-exclusion.v1",
        "task_id": "TRR-P02",
        "status": "CAPTURED_PUBLIC_DIAGNOSTIC_CASES",
        "generated_utc": timestamp,
        "model": {
            "id": _identifier(model_id, field="model_id"),
            "revision": _identifier(model_revision, field="model_revision"),
            "cut_depth": int(cut_depth),
            "bos_token_id": int(bos_token_id),
            "vocab_size": int(vocab_size),
        },
        "source_plan": {
            "path": str(source_plan_path),
            "sha256": source_hash,
        },
        "cases": rows,
        "exclusion_rule": {
            "match_fields": [
                "case_id",
                "context_id",
                "endpoint_id",
                "sequence_token_ids",
            ],
            "future_confirmation": "exclude only the exact listed case/context/endpoint tuples and sequences",
            "global_token_type_exclusion": False,
            "token_ids_are_global_exclusions": False,
        },
        "metrics": {
            "status": "PENDING_SEPARATE_DIAGNOSTIC_ANALYSIS",
            "values": [],
        },
        "truth_access": "none; public diagnostic metadata only",
    }


def write_exclusion_record(destination: Path, record: Mapping[str, Any]) -> None:
    """Create a JSON record once, refusing to overwrite an existing capture."""

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ExclusionCaptureError(f"refusing to overwrite exclusion record: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
