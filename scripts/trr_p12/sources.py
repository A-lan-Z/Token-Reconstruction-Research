"""Create the TRR-P12 additive opaque identity exclusion union.

The curator consumes only the explicitly named P11 r14/r17/root-release
receipts and the already-opened confirmation512 and transfer64 selection
ledgers.  It extracts identity fields from the declared domain row arrays
only; it never recursively walks arbitrary metadata.  Payload keys are
ignored and no source text, token IDs, activations, weights, labels, or truth
are emitted.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


TASK_ID = "TRR-P12"
SCHEMA = "token-reconstruction.trr-p12-identity-union.v1"
STATUS = "IDENTITY_UNION_COMPLETE_NO_PAYLOAD"
DOMAINS = ("finance", "pile")
P11_UNION_FIELDS = (
    "record_id",
    "source_index",
    "rendered_sha256",
    "tokenized_record_sha256",
    "h40_sequence_sha256",
    "h128_sequence_sha256",
    "h129_sequence_sha256",
    "trr0002_active_token_ids_sha256",
    "trr0002_h40_token_ids_sha256",
)
P11_UNION_FIELD_SET = frozenset(P11_UNION_FIELDS)
HASH_FIELDS = frozenset(set(P11_UNION_FIELDS) - {"record_id", "source_index"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are producer-bound canonical row fields.  In particular, the two
# sequence aliases are the names emitted by the validated P11 selector; no
# unverified token_ids or arbitrary hash fields are admitted.
ROW_FIELD_ALIASES = {
    "record_id": "record_id",
    "source_index": "source_index",
    "public_record_sha256": "rendered_sha256",
    "rendered_sha256": "rendered_sha256",
    "tokenized_record_sha256": "tokenized_record_sha256",
    "h40_sequence_sha256": "h40_sequence_sha256",
    "h128_sequence_sha256": "h128_sequence_sha256",
    "final_sequence_sha256": "h128_sequence_sha256",
    "h129_sequence_sha256": "h129_sequence_sha256",
    "trr0002_active_token_ids_sha256": "trr0002_active_token_ids_sha256",
    "trr0002_h40_token_ids_sha256": "trr0002_h40_token_ids_sha256",
}
PAYLOAD_KEYS = frozenset(
    {
        "attention_mask",
        "input_ids",
        "labels",
        "position_ids",
        "plaintext",
        "source_text",
        "target_tokens",
        "token_ids",
        "activations",
        "hidden_states",
        "logits",
        "weights",
    }
)


class CuratorError(RuntimeError):
    """Raised when a source or release binding fails closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_file(value: str | Path, *, root: Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise CuratorError(f"{label} is unavailable or symlinked: {path}")
    return path


def file_binding(
    value: str | Path,
    *,
    root: Path,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = resolve_file(value, root=root, label=label)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise CuratorError(f"{label} SHA-256 changed: {digest} != {expected_sha256}")
    return path, {"path": str(path), "bytes": path.stat().st_size, "sha256": digest, "readonly": True}


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CuratorError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise CuratorError(f"{label} must be a JSON object")
    return dict(value)


def namespace_for_row(domain: str, row: Mapping[str, Any]) -> str:
    declared = row.get("dataset_key", row.get("domain", domain))
    if not isinstance(declared, str) or declared.casefold() != domain:
        raise CuratorError(f"row domain mismatch: expected {domain}, got {declared!r}")
    dataset_id = row.get("dataset_id", row.get("source_dataset_id", "*"))
    split = row.get("split", row.get("dataset_split", "*"))
    revision = row.get("revision", row.get("dataset_revision", "*"))
    for value, label in ((dataset_id, "dataset_id"), (split, "split"), (revision, "revision")):
        if not isinstance(value, str):
            raise CuratorError(f"row {label} is not a string")
    return "|".join((domain, dataset_id or "*", split or "*", revision or "*"))


def add_identity(
    fields: dict[str, dict[str, set[str | int]]],
    field: str,
    value: Any,
    namespace: str,
    *,
    location: str,
) -> None:
    if field == "record_id":
        if not isinstance(value, str) or not value:
            raise CuratorError(f"malformed record_id at {location}")
    elif field == "source_index":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CuratorError(f"malformed source_index at {location}")
    else:
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value.casefold()):
            raise CuratorError(f"malformed {field} at {location}")
        value = value.casefold()
    fields.setdefault(field, {}).setdefault(namespace, set()).add(value)


def explicit_rows(payload: Mapping[str, Any], *, label: str) -> tuple[dict[str, list[Mapping[str, Any]]], str]:
    """Return only a declared domain-row container, never arbitrary metadata."""
    candidates: list[tuple[str, Any]] = []
    selection_rule = payload.get("selection_rule")
    if isinstance(selection_rule, Mapping):
        candidates.append(("selection_rule.records", selection_rule.get("records")))
    candidates.append(("records_by_domain", payload.get("records_by_domain")))
    candidates.append(("records", payload.get("records")))
    for container_name, candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if all(isinstance(candidate.get(domain), list) for domain in DOMAINS):
            rows = {domain: candidate[domain] for domain in DOMAINS}
            if all(all(isinstance(row, Mapping) for row in rows[domain]) for domain in DOMAINS):
                return rows, container_name
    raise CuratorError(f"{label} has no explicit finance/pile row container")


def extract_rows(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_rows_per_domain: int,
) -> tuple[dict[str, dict[str, set[str | int]]], dict[str, Any]]:
    rows_by_domain, container_name = explicit_rows(payload, label=label)
    fields: dict[str, dict[str, set[str | int]]] = {}
    domain_counts: dict[str, dict[str, int]] = {}
    for domain in DOMAINS:
        rows = rows_by_domain[domain]
        if len(rows) != expected_rows_per_domain:
            raise CuratorError(f"{label} {domain} row count {len(rows)} != {expected_rows_per_domain}")
        domain_required: dict[str, set[str | int]] = {field: set() for field in ("record_id", "source_index", "rendered_sha256", "h128_sequence_sha256")}
        for row_index, row in enumerate(rows):
            seen_row: dict[str, set[str | int]] = {field: set() for field in domain_required}
            namespace = namespace_for_row(domain, row)
            for raw_key, value in row.items():
                key = str(raw_key).casefold().replace("-", "_")
                if key in PAYLOAD_KEYS:
                    continue
                field = ROW_FIELD_ALIASES.get(key)
                if field is None:
                    continue
                add_identity(fields, field, value, namespace, location=f"{label}/{domain}/{row_index}/{raw_key}")
                if field in seen_row:
                    seen_row[field].add(value.casefold() if isinstance(value, str) else value)
            if any(not seen_row[field] for field in seen_row):
                missing = [field for field, values in seen_row.items() if not values]
                raise CuratorError(f"{label} row lacks required identity fields {missing}: {domain}/{row_index}")
            for field, values in seen_row.items():
                domain_required[field].update(values)
        domain_counts[domain] = {
            field: len(values)
            for field, values in domain_required.items()
        }
        if any(value != expected_rows_per_domain for value in domain_counts[domain].values()):
            raise CuratorError(f"{label} {domain} identity counts are not exact: {domain_counts[domain]}")
    return fields, {"row_container": container_name, "rows_per_domain": expected_rows_per_domain, "domain_identity_counts": domain_counts}


def merge_fields(destination: dict[str, dict[str, set[str | int]]], source: Mapping[str, Mapping[str, set[str | int]]]) -> None:
    for field, by_namespace in source.items():
        for namespace, values in by_namespace.items():
            destination.setdefault(field, {}).setdefault(namespace, set()).update(values)


def serializable_fields(fields: Mapping[str, Mapping[str, set[str | int]]]) -> dict[str, dict[str, list[str | int]]]:
    return {
        field: {
            namespace: sorted(values, key=lambda item: (isinstance(item, str), str(item)))
            for namespace, values in sorted(by_namespace.items())
        }
        for field, by_namespace in sorted(fields.items())
    }


def counts(fields: Mapping[str, Mapping[str, set[str | int]]]) -> dict[str, int]:
    return {field: sum(len(values) for values in by_namespace.values()) for field, by_namespace in sorted(fields.items())}


def namespace_counts(fields: Mapping[str, Mapping[str, set[str | int]]]) -> dict[str, dict[str, int]]:
    return {field: {namespace: len(values) for namespace, values in sorted(by_namespace.items())} for field, by_namespace in sorted(fields.items())}


def read_base_union(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, set[str | int]]]]:
    resolved, binding = file_binding(path, root=root, label="P11 r14 identity union", expected_sha256="125275eab38c66117d45f5e0df4085058dae8a5c939fff1554cab7e608eb1fe1")
    payload = load_json(resolved, label="P11 r14 identity union")
    if payload.get("schema") != "token-reconstruction.trr-p11-identity-union.v1" or payload.get("status") != "IDENTITY_UNION_COMPLETE_NO_PAYLOAD":
        raise CuratorError("P11 r14 union schema/status changed")
    boundary = payload.get("access_boundary")
    if not isinstance(boundary, Mapping):
        raise CuratorError("P11 r14 union boundary is absent")
    forbidden = ("source_text_read", "source_text_serialized", "source_tokens_serialized", "token_values_emitted", "truth_or_scores_read", "p03_holdout_accessed", "new_selection_started")
    if any(boundary.get(key) is True for key in forbidden):
        raise CuratorError("P11 r14 union boundary is open")
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise CuratorError("P11 r14 union fields are absent")
    fields: dict[str, dict[str, set[str | int]]] = {}
    for raw_field, by_namespace in raw_fields.items():
        field = str(raw_field)
        if field not in P11_UNION_FIELD_SET or not isinstance(by_namespace, Mapping):
            raise CuratorError(f"P11 r14 contains an unapproved field: {field}")
        for raw_namespace, values in by_namespace.items():
            if not isinstance(raw_namespace, str) or len(raw_namespace.split("|")) != 4 or not isinstance(values, list):
                raise CuratorError(f"P11 r14 namespace/value shape changed: {field}")
            for value in values:
                add_identity(fields, field, value, raw_namespace, location=f"P11 r14/{field}")
    if payload.get("identity_counts", payload.get("counts")) != counts(fields):
        raise CuratorError("P11 r14 internal identity counts changed")
    return payload, binding, fields


def receipt_binding(path: Path, *, root: Path, label: str, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, binding = file_binding(path, root=root, label=label, expected_sha256=expected_sha256)
    return load_json(resolved, label=label), binding


def source_binding(
    path: Path,
    *,
    root: Path,
    label: str,
    role: str,
    producer: str,
    expected_sha256: str,
    expected_rows_per_domain: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, set[str | int]]]]:
    resolved, binding = file_binding(path, root=root, label=label, expected_sha256=expected_sha256)
    payload = load_json(resolved, label=label)
    fields, row_metadata = extract_rows(payload, label=label, expected_rows_per_domain=expected_rows_per_domain)
    return payload, {
        **binding,
        "label": label,
        "role": role,
        "known_producer": producer,
        "identity_counts": counts(fields),
        "namespace_counts": namespace_counts(fields),
        **row_metadata,
        "payload_emitted": False,
    }, fields


def receipt_closed(payload: Mapping[str, Any], *, label: str) -> None:
    boundary = payload.get("access_boundary")
    if isinstance(boundary, Mapping):
        if any(boundary.get(key) is True for key in ("p03_holdout_accessed", "truth_opened", "source_text_written", "source_tokens_serialized", "token_values_emitted")):
            raise CuratorError(f"{label} boundary is open")
    if payload.get("p03_holdout_accessed") is True or payload.get("truth_opened") is True:
        raise CuratorError(f"{label} records forbidden access")


def build_union(
    *,
    root: Path,
    base_union: Path,
    r17_audit: Path,
    root_release: Path,
    confirmation: Path,
    transfer: Path,
    output: Path,
) -> dict[str, Any]:
    base_payload, base_binding, fields = read_base_union(base_union, root=root)
    r17_payload, r17_binding = receipt_binding(r17_audit, root=root, label="P11 r17 recovery identity audit", expected_sha256="43416d0d1945821812278df8dfc3ee7639f872e9c8407abdf245fb281bda9d65")
    release_payload, release_binding = receipt_binding(root_release, root=root, label="P11 root selection release", expected_sha256="0e1816711965b0601a2e00ce4c187bded9f0c0fca4493e71ef9478a772659f95")
    receipt_closed(r17_payload, label="P11 r17 audit")
    receipt_closed(release_payload, label="P11 root release")
    if r17_payload.get("coverage_complete") is not True or r17_payload.get("selection_release") is not True:
        raise CuratorError("P11 r17 audit is not complete and released")
    if release_payload.get("selection_release") is False:
        raise CuratorError("P11 root release does not release selection")
    confirmation_payload, confirmation_binding, confirmation_fields = source_binding(
        confirmation,
        root=root,
        label="P11 confirmation512 source selection",
        role="opened_confirmation_source_selection",
        producer="scripts/trr_p11/source_selector.py::_selection_row via select_sources",
        expected_sha256="fe7129e9d7230100d1900facea98b15b8000f382a0bd7035b2039b4d9429bf68",
        expected_rows_per_domain=256,
    )
    transfer_payload, transfer_binding, transfer_fields = source_binding(
        transfer,
        root=root,
        label="P11 transfer64 source selection",
        role="opened_transfer_source_selection",
        producer="TRR-0011 natural 128-token selector; release prepared by scripts/trr_p11/prepare_transfer_reservation.py",
        expected_sha256="8674ea5c824c5f3bc4aa10c44439422654572af546684567046f4a5ffb717c0f",
        expected_rows_per_domain=32,
    )
    merge_fields(fields, confirmation_fields)
    merge_fields(fields, transfer_fields)
    source_records = [
        {"label": "P11 r14 identity union", "role": "immutable_base_identity_union", **base_binding, "known_producer": "P11 published identity_union_export_r14", "identity_counts": base_payload.get("identity_counts", base_payload.get("counts", {})), "payload_emitted": False},
        {"label": "P11 r17 recovery identity audit", "role": "release_receipt", **r17_binding, "known_producer": "P11 reviewed r17 audit", "identity_counts": {}, "payload_emitted": False},
        {"label": "P11 root selection release", "role": "release_receipt", **release_binding, "known_producer": "P11 root selection release", "identity_counts": {}, "payload_emitted": False},
        confirmation_binding,
        transfer_binding,
    ]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": STATUS,
        "revision": "r2",
        "created_utc": utc_now(),
        "purpose": "Additive opaque P11 exclusion union for the P12 public trajectory study.",
        "access_boundary": {
            "source_text_read": False,
            "source_tokens_serialized": False,
            "token_values_emitted": False,
            "activation_payload_read": False,
            "target_weights_read": False,
            "truth_or_scores_read": False,
            "new_selection_started": False,
            "target_training_started": False,
            "p03_holdout_accessed": False,
            "payload_emitted": False,
        },
        "base_binding": base_binding,
        "receipt_bindings": {"r17_audit": r17_binding, "root_selection_release": release_binding},
        "source_bindings": source_records,
        "known_producers": {
            "confirmation512": confirmation_binding["known_producer"],
            "transfer64": transfer_binding["known_producer"],
        },
        "fields": serializable_fields(fields),
        "identity_counts": counts(fields),
        "namespace_counts": namespace_counts(fields),
        "pending_additive_layers": [
            {"owner": "Agent1/TRR-0013", "status": "PENDING_ROOT_RESERVATIONS", "append_only": True},
            {"owner": "P12 target-training fitting sources", "status": "PENDING_ROOT_PREREGISTRATION", "append_only": True},
            {"owner": "P12 reconstruction panel", "status": "PENDING_ROOT_PREREGISTRATION", "append_only": True},
        ],
        "selection_started": False,
        "target_training_started": False,
        "p03_holdout_accessed": False,
    }
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise CuratorError(f"output is create-only: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output), "identity_counts": payload["identity_counts"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--base-union", type=Path, required=True)
    parser.add_argument("--r17-audit", type=Path, required=True)
    parser.add_argument("--root-release", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_union(
        root=args.root.resolve(),
        base_union=args.base_union,
        r17_audit=args.r17_audit,
        root_release=args.root_release,
        confirmation=args.confirmation,
        transfer=args.transfer,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
