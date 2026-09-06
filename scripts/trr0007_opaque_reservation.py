#!/usr/bin/env python3
"""Export a hash-only source/sequence reservation for a future public selector.

The exporter consumes only a completed identity-only TRR-0007 source-selection
ledger. It never loads Arrow rows, a tokenizer, token IDs, source text, labels,
model weights, or truth. The output is deliberately smaller and less revealing
than the input: it contains only SHA-256 values for rendered public sources and
final selected sequences, with no record IDs, source indices, domain labels, or
text.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


TASK_ID = "TRR-0007"
SELECTION_SCHEMA = "token-reconstruction.trr0007-source-selection.v1"
SELECTION_STATUS = "FROZEN_TRR0007_SOURCE_SELECTION_NO_TRUTH"
RESERVATION_SCHEMA = "token-reconstruction.trr0007-opaque-source-sequence-reservation.v1"
RESERVATION_STATUS = "READY_FOR_FUTURE_PUBLIC_SELECTION_HASH_ONLY"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ReservationError(ValueError):
    """Raised when an identity ledger cannot be exported safely."""


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


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ReservationError(f"selection ledger must be a regular file: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _newline_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(values) + "\n").encode("ascii"))


def _json_digest(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return _sha256_bytes(payload.encode("utf-8"))


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return _sha256_bytes(payload.encode("utf-8"))


def _hash_summary(values: Sequence[str]) -> dict[str, Any]:
    ordered = sorted(set(values))
    return {
        "values": ordered,
        "ordered_count": len(ordered),
        "distinct_count": len(ordered),
        "ordered_newline_sha256": _newline_digest(ordered),
        "unique_set_newline_sha256": _newline_digest(ordered),
        "unique_set_canonical_json_sha256": _json_digest(ordered),
    }


def _load_selection(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReservationError(f"cannot parse selection ledger: {path}") from exc
    if not isinstance(payload, dict):
        raise ReservationError("selection ledger must contain an object")
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ReservationError("selection ledger schema is not the frozen TRR-0007 source-selection schema")
    if payload.get("task_id") != TASK_ID or payload.get("status") != SELECTION_STATUS:
        raise ReservationError("selection ledger is not a completed no-truth TRR-0007 selection")
    forbidden_true = (
        "truth_opened",
        "truth_created",
        "source_text_or_target_labels",
        "private_or_truth_payload_read",
        "source_text_written",
        "token_ids_written",
    )
    if any(payload.get(key) is True for key in forbidden_true):
        raise ReservationError("selection ledger records forbidden truth, source, or token access")
    rule = payload.get("selection_rule")
    if not isinstance(rule, Mapping) or rule.get("source_text_or_token_ids_written") is not False:
        raise ReservationError("selection ledger does not certify identity-only output")
    records = rule.get("records")
    if not isinstance(records, Mapping) or not records:
        raise ReservationError("selection ledger has no selected record metadata")
    return payload, record


def _extract_hashes(payload: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    records = payload["selection_rule"]["records"]
    if not isinstance(records, Mapping):  # guarded by _load_selection
        raise ReservationError("selection records are not a mapping")
    source_values: list[str] = []
    sequence_values: list[str] = []
    for _, rows in records.items():
        if not isinstance(rows, list) or not rows:
            raise ReservationError("each selection group must contain selected records")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ReservationError("selected record metadata must be objects")
            source = row.get("public_record_sha256")
            sequence = row.get("final_sequence_sha256")
            if not isinstance(source, str) or HEX64.fullmatch(source) is None:
                raise ReservationError("selected metadata has an invalid public source SHA-256")
            if not isinstance(sequence, str) or HEX64.fullmatch(sequence) is None:
                raise ReservationError("selected metadata has an invalid final-sequence SHA-256")
            source_values.append(source)
            sequence_values.append(sequence)
    if len(source_values) != len(set(source_values)):
        raise ReservationError("selection ledger contains duplicate public source hashes")
    if len(sequence_values) != len(set(sequence_values)):
        raise ReservationError("selection ledger contains duplicate final-sequence hashes")
    return sorted(source_values), sorted(sequence_values)


def export_reservation(selection_path: Path, output_path: Path) -> dict[str, Any]:
    payload, selection_record = _load_selection(selection_path.expanduser().resolve())
    source_values, sequence_values = _extract_hashes(payload)
    hashes = {
        "public_record_sha256": _hash_summary(source_values),
        "final_sequence_sha256": _hash_summary(sequence_values),
    }
    core = {
        "schema": RESERVATION_SCHEMA,
        "hashes": {
            "public_record_sha256": source_values,
            "final_sequence_sha256": sequence_values,
        },
    }
    reservation_digest = _canonical_digest(core)
    output = output_path.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ReservationError(f"reservation output is create-only and already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": RESERVATION_SCHEMA,
        "task_id": TASK_ID,
        "status": RESERVATION_STATUS,
        "created_utc": _utc_now(),
        "purpose": "hash-only source and final-sequence exclusions for a future public selection",
        "input_selection_ledger": selection_record,
        "hash_conventions": {
            "public_record_sha256": "SHA-256 rendered public-source fingerprint from selection_rule.records metadata",
            "final_sequence_sha256": "SHA-256 final selected token-sequence fingerprint from selection_rule.records metadata",
            "canonical_order": "lexicographic order within each hash set; no domain or selection-order metadata is exported",
        },
        "counts": {
            "public_record_sha256": len(source_values),
            "final_sequence_sha256": len(sequence_values),
        },
        "hashes": hashes,
        "reservation_digest_sha256": reservation_digest,
        "privacy_boundary": {
            "hash_only": True,
            "contains_source_text": False,
            "contains_record_ids": False,
            "contains_source_indices": False,
            "contains_domain_or_style_labels": False,
            "contains_target_labels": False,
            "contains_token_ids": False,
            "contains_model_weights": False,
            "contains_truth": False,
        },
    }
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    result["output"] = {"path": str(output), "bytes": int(output.stat().st_size), "sha256": _sha256_file(output)}
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="completed identity-only TRR-0007 source-selection ledger")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/TRR-0007/coordination/p06_opaque_source_sequence_reservation.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = export_reservation(args.selection, args.output)
    except (OSError, ReservationError, ValueError) as exc:
        print(f"TRR-0007 opaque reservation export failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
