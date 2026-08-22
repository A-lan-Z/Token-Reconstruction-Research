"""Fail-closed opaque-record and commitment protocol for TRR-0001-R1."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any, Iterable, Mapping, Sequence


PUBLIC_COMMITMENT_SCHEMA = "token-reconstruction.trr0001-r1-selection-commitment.v1"
PRIVATE_SELECTION_SCHEMA = "token-reconstruction.trr0001-r1-private-selection.v1"
REVEAL_SCHEMA = "token-reconstruction.trr0001-r1-selection-reveal.v1"
OBSERVATION_INDEX_SCHEMA = "token-reconstruction.trr0001-r1-observation-index.v1"
SANITIZED_CONFIG_SCHEMA = "token-reconstruction.trr0001-r1-sanitized-config.v1"
OPAQUE_ID_PATTERN = re.compile(r"^blind-r1-[0-9]{6}$")

_FORBIDDEN_KEYS = {
    "dataset_index",
    "dataset_indices",
    "index",
    "indices",
    "text",
    "source_text",
    "source_tokens",
    "source_token_ids",
    "token_ids",
    "text_sha256",
    "source_sha256",
    "source_hash",
    "selection_key",
    "selection_key_hex",
    "selection_seed",
    "record_selection_seed",
    "private_mapping",
    "truth",
}


class BlindProtocolError(RuntimeError):
    """Raised when a public interface could reveal source identity or truth."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def opaque_record_ids(count: int = 64) -> list[str]:
    if count <= 0:
        raise BlindProtocolError("opaque record count must be positive")
    return [f"blind-r1-{ordinal:06d}" for ordinal in range(1, count + 1)]


def require_opaque_record_order(values: Sequence[Any], *, count: int = 64) -> list[str]:
    expected = opaque_record_ids(count)
    observed = [str(value) for value in values]
    if observed != expected:
        raise BlindProtocolError("opaque record IDs or their order changed")
    if any(OPAQUE_ID_PATTERN.fullmatch(value) is None for value in observed):
        raise BlindProtocolError("record ID is not opaque")
    return observed


def _walk_forbidden(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            forbidden_fragment = any(
                fragment in lowered
                for fragment in ("source_text", "source_token", "selection_key", "private_mapping")
            )
            if lowered == "selection_key_disclosed" and child is False:
                forbidden_fragment = False
            if lowered in _FORBIDDEN_KEYS or forbidden_fragment:
                raise BlindProtocolError(f"forbidden public field at {path}.{key}")
            _walk_forbidden(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, path=f"{path}[{index}]")


def reject_source_metadata(value: Any) -> None:
    """Reject source-resolving metadata anywhere in a pre-freeze public payload."""
    _walk_forbidden(value)


def require_exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, label: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        missing = sorted(expected_set - set(value))
        extra = sorted(set(value) - expected_set)
        raise BlindProtocolError(f"{label} fields changed; missing={missing}, extra={extra}")


def selection_sort_key(
    key: bytes, *, dataset_revision: str, dataset_index: int, text_sha256: str
) -> bytes:
    if len(key) != 32:
        raise BlindProtocolError("selection key must contain exactly 256 bits")
    if dataset_index < 0 or not re.fullmatch(r"[0-9a-f]{64}", text_sha256):
        raise BlindProtocolError("invalid private source identity")
    message = f"{dataset_revision}\0{dataset_index}\0{text_sha256}".encode("ascii")
    return hmac.new(key, message, hashlib.sha256).digest()


def select_private_records(
    *,
    key: bytes,
    dataset_revision: str,
    rows: Iterable[tuple[int, str, Sequence[int]]],
    excluded_indices: set[int],
    count: int = 64,
) -> list[dict[str, Any]]:
    """Select eligible rows by secret-keyed ordering and assign opaque IDs."""

    eligible: list[tuple[bytes, int, str, list[int]]] = []
    observed: set[int] = set()
    for dataset_index, text, source_tokens in rows:
        if dataset_index in observed:
            raise BlindProtocolError("dataset iterator duplicated an index")
        observed.add(dataset_index)
        if dataset_index in excluded_indices or len(source_tokens) < 39:
            continue
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        token_ids = [128000, *[int(token) for token in source_tokens[:39]]]
        eligible.append(
            (
                selection_sort_key(
                    key,
                    dataset_revision=dataset_revision,
                    dataset_index=dataset_index,
                    text_sha256=text_sha256,
                ),
                dataset_index,
                text_sha256,
                token_ids,
            )
        )
    if len(eligible) < count:
        raise BlindProtocolError("fewer eligible fresh rows than requested")
    eligible.sort(key=lambda item: (item[0], item[1]))
    records: list[dict[str, Any]] = []
    for record_id, (_, dataset_index, text_sha256, token_ids) in zip(
        opaque_record_ids(count), eligible[:count]
    ):
        records.append(
            {
                "record_id": record_id,
                "dataset_index": dataset_index,
                "text_sha256": text_sha256,
                "token_ids": token_ids,
            }
        )
    return records


def commitment_payload(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require_opaque_record_order([row.get("record_id") for row in records], count=len(records))
    normalized: list[dict[str, Any]] = []
    for row in records:
        require_exact_keys(
            row,
            {"record_id", "dataset_index", "text_sha256", "token_ids"},
            label="private selection record",
        )
        token_ids = [int(value) for value in row["token_ids"]]
        if len(token_ids) != 40 or token_ids[0] != 128000:
            raise BlindProtocolError("private selection token geometry changed")
        normalized.append(
            {
                "record_id": str(row["record_id"]),
                "dataset_index": int(row["dataset_index"]),
                "text_sha256": str(row["text_sha256"]),
                "token_ids": token_ids,
            }
        )
    return {
        "schema": "token-reconstruction.trr0001-r1-selection-commitment-payload.v1",
        "records": normalized,
    }


def commitment_digest(key: bytes, records: Sequence[Mapping[str, Any]]) -> str:
    if len(key) != 32:
        raise BlindProtocolError("selection key must contain exactly 256 bits")
    return hmac.new(key, canonical_bytes(commitment_payload(records)), hashlib.sha256).hexdigest()


def public_commitment(
    *,
    key: bytes,
    records: Sequence[Mapping[str, Any]],
    dataset_id: str,
    dataset_revision: str,
    created_utc: str,
) -> dict[str, Any]:
    order = require_opaque_record_order([row.get("record_id") for row in records])
    return {
        "schema": PUBLIC_COMMITMENT_SCHEMA,
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R1",
        "created_utc": created_utc,
        "scheme": "HMAC-SHA256 over canonical private mapping with an evaluator-private 256-bit key",
        "commitment": commitment_digest(key, records),
        "record_count": 64,
        "opaque_record_order": order,
        "dataset": {"id": dataset_id, "revision": dataset_revision, "split": "train"},
        "selection_algorithm": "eligible non-excluded rows ordered by HMAC-SHA256 of revision, row number, and text digest; first 64; row metadata and key withheld until post-freeze reveal",
        "eligibility": "at least 39 source tokens; prepend exactly one declared BOS",
        "disjointness": "all original target-update, inverse-train, development, blind, excluded-attempt, and reproducibility records are excluded",
        "reveal_gate": "only after reconstruction outputs, sanitized configuration, access evidence, and route are frozen and verified",
        "source_identity_disclosed": False,
        "selection_key_disclosed": False,
    }


def private_selection_document(
    *, key: bytes, records: Sequence[Mapping[str, Any]], created_utc: str
) -> dict[str, Any]:
    if len(key) != 32:
        raise BlindProtocolError("selection key must contain exactly 256 bits")
    payload = commitment_payload(records)
    return {
        "schema": PRIVATE_SELECTION_SCHEMA,
        "created_utc": created_utc,
        "selection_key_hex": key.hex(),
        "records": payload["records"],
    }


def reveal_document(private: Mapping[str, Any], *, revealed_utc: str) -> dict[str, Any]:
    require_exact_keys(
        private,
        {"schema", "created_utc", "selection_key_hex", "records"},
        label="private selection",
    )
    if private["schema"] != PRIVATE_SELECTION_SCHEMA:
        raise BlindProtocolError("private selection schema changed")
    key_hex = str(private["selection_key_hex"])
    try:
        key = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise BlindProtocolError("selection key is not hexadecimal") from exc
    full_records = commitment_payload(private["records"])["records"]
    mapping = [
        {
            "record_id": row["record_id"],
            "dataset_index": row["dataset_index"],
            "text_sha256": row["text_sha256"],
        }
        for row in full_records
    ]
    return {
        "schema": REVEAL_SCHEMA,
        "revealed_utc": revealed_utc,
        "selection_key_hex": key.hex(),
        "records": mapping,
    }


def validate_public_commitment(value: Mapping[str, Any]) -> None:
    require_exact_keys(
        value,
        {
            "schema", "task_id", "revision_id", "created_utc", "scheme", "commitment",
            "record_count", "opaque_record_order", "dataset", "selection_algorithm",
            "eligibility", "disjointness", "reveal_gate", "source_identity_disclosed",
            "selection_key_disclosed",
        },
        label="public commitment",
    )
    _walk_forbidden(value)
    if value["schema"] != PUBLIC_COMMITMENT_SCHEMA or value["record_count"] != 64:
        raise BlindProtocolError("public commitment identity or record count changed")
    require_opaque_record_order(value["opaque_record_order"])
    if not re.fullmatch(r"[0-9a-f]{64}", str(value["commitment"])):
        raise BlindProtocolError("public commitment digest is invalid")
    if value["source_identity_disclosed"] is not False or value["selection_key_disclosed"] is not False:
        raise BlindProtocolError("public commitment claims a disclosure")


def verify_reveal(
    *,
    public: Mapping[str, Any],
    reveal: Mapping[str, Any],
    dataset_revision: str,
    excluded_indices: set[int],
    dataset_rows: Sequence[str],
    tokenizer: Any,
) -> dict[str, Any]:
    validate_public_commitment(public)
    require_exact_keys(
        reveal,
        {"schema", "revealed_utc", "selection_key_hex", "records"},
        label="selection reveal",
    )
    if reveal["schema"] != REVEAL_SCHEMA:
        raise BlindProtocolError("selection reveal schema changed")
    try:
        key = bytes.fromhex(str(reveal["selection_key_hex"]))
    except ValueError as exc:
        raise BlindProtocolError("revealed selection key is not hexadecimal") from exc
    if len(key) != 32:
        raise BlindProtocolError("revealed selection key length changed")
    records = reveal["records"]
    if not isinstance(records, list) or len(records) != 64:
        raise BlindProtocolError("revealed mapping record count changed")
    for row in records:
        require_exact_keys(
            row,
            {"record_id", "dataset_index", "text_sha256"},
            label="revealed mapping record",
        )
    require_opaque_record_order([row["record_id"] for row in records])
    if [row["record_id"] for row in records] != public["opaque_record_order"]:
        raise BlindProtocolError("revealed opaque order differs from committed order")
    expected_rows: list[tuple[int, str, list[int]]] = []
    for index, text in enumerate(dataset_rows):
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        expected_rows.append((index, text, tokens))
    expected = select_private_records(
        key=key,
        dataset_revision=dataset_revision,
        rows=expected_rows,
        excluded_indices=excluded_indices,
    )
    expected_mapping = [
        {
            "record_id": row["record_id"],
            "dataset_index": row["dataset_index"],
            "text_sha256": row["text_sha256"],
        }
        for row in expected
    ]
    if expected_mapping != records:
        raise BlindProtocolError("revealed mapping is not the committed selection algorithm output")
    if not hmac.compare_digest(
        commitment_digest(key, expected),
        str(public["commitment"]),
    ):
        raise BlindProtocolError("revealed selection does not match commitment")
    return {
        "verified": True,
        "commitment": public["commitment"],
        "records": len(records),
        "disjoint_from_original_records": all(
            int(row["dataset_index"]) not in excluded_indices for row in records
        ),
        "opaque_order_verified": True,
        "eligibility_verified": True,
    }


def validate_observation_index(value: Mapping[str, Any]) -> None:
    require_exact_keys(
        value,
        {"schema", "records", "entries", "source_material_included"},
        label="observation index",
    )
    _walk_forbidden(value)
    if value["schema"] != OBSERVATION_INDEX_SCHEMA or value["source_material_included"] is not False:
        raise BlindProtocolError("observation index schema or disclosure flag changed")
    records = value["records"]
    if not isinstance(records, list) or any(set(record) != {"record_id"} for record in records):
        raise BlindProtocolError("observation records must contain only opaque IDs")
    require_opaque_record_order([record["record_id"] for record in records])
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) != 6:
        raise BlindProtocolError("observation arm coverage changed")
    expected = {(condition, cut) for condition in ("matched_public", "unavailable_target_lora") for cut in (0, 4, 8)}
    observed: set[tuple[str, int]] = set()
    for entry in entries:
        require_exact_keys(entry, {"condition", "cut_depth", "path", "bytes", "sha256"}, label="observation entry")
        observed.add((str(entry["condition"]), int(entry["cut_depth"])))
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"])) or int(entry["bytes"]) <= 0:
            raise BlindProtocolError("observation artifact record is invalid")
        if str(entry["path"]).startswith("/") or ".." in str(entry["path"]).split("/"):
            raise BlindProtocolError("observation path must remain relative")
    if observed != expected:
        raise BlindProtocolError("observation arm set changed")


def validate_sanitized_config(value: Mapping[str, Any]) -> None:
    require_exact_keys(
        value,
        {
            "schema", "task_id", "revision_id", "model", "observation_index",
            "inverse_states", "record_order", "condition_order", "cut_order", "geometry",
            "methods", "execution", "access_contract", "truth_or_source_inputs",
        },
        label="sanitized configuration",
    )
    _walk_forbidden(value)
    if value["schema"] != SANITIZED_CONFIG_SCHEMA:
        raise BlindProtocolError("sanitized configuration schema changed")
    require_exact_keys(
        value["model"],
        {"id", "revision", "dtype", "attention_implementation"},
        label="model",
    )
    require_exact_keys(
        value["observation_index"],
        {"path", "bytes", "sha256"},
        label="observation index artifact",
    )
    require_opaque_record_order(value["record_order"])
    if (
        value["condition_order"] != ["matched_public", "unavailable_target_lora"]
        or value["cut_order"] != [0, 4, 8]
    ):
        raise BlindProtocolError("condition or cut order changed")
    require_exact_keys(
        value["geometry"],
        {"records", "sequence_tokens", "scored_tokens_per_record", "hidden_size", "candidate_budget"},
        label="geometry",
    )
    expected_geometry = {
        "records": 64,
        "sequence_tokens": 40,
        "scored_tokens_per_record": 39,
        "hidden_size": 2048,
        "candidate_budget": 16,
    }
    if value["geometry"] != expected_geometry:
        raise BlindProtocolError("sanitized reconstruction geometry changed")
    if value["methods"] != ["direct_inverse", "causal_public_surrogate_search"]:
        raise BlindProtocolError("method set changed")
