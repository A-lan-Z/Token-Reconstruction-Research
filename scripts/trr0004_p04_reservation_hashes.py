#!/usr/bin/env python3
"""Create the P04 opaque reservation-hash exchange.

This producer intentionally consumes only P04 metadata ledgers.  It never opens
an observation tensor, source text/token payload, evaluator truth file, model
weight file, or the P04 target-update tensor.  Row identifiers are held only
long enough to derive order digests and overlap counts.  The exchange writes
per-record public fingerprints (which are already opaque SHA-256 values), but
never writes record IDs, source text, token IDs, or any other row payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TASK_ID = "TRR-P04"
SCHEMA = "token-reconstruction.trr-p04-reservation-hashes.v1"

INPUT_FILES = {
    "selection": Path("experiments/TRR-P04/setup/public_selection-r2.json"),
    "pool_manifest": Path("experiments/TRR-P04/setup/public-pools-r2/pool_manifest.json"),
    "fit_replay_ledger": Path("experiments/TRR-P04/setup/public-pools-r2/pool_manifest.json"),
    "correction": Path("experiments/TRR-P04/setup/public-pools-r2/correction_records.json"),
    "validation": Path("experiments/TRR-P04/setup/public-pools-r2/validation_records.json"),
    "fresh_evaluation": Path("experiments/TRR-P04/setup/public-pools-r2/fresh_panel_index.json"),
    "target_selection_audit": Path(
        "experiments/TRR-P04/setup/target-selection-audit-r4/target_selection_audit.json"
    ),
    "target_private_selection_audit": Path(
        "experiments/TRR-P04/private/evaluator_target_update/target_selection_audit.json"
    ),
    "target_preparation_receipt": Path(
        "experiments/TRR-P04/private/evaluator_target_update/target_preparation_receipt.json"
    ),
    "target_plan": Path("experiments/TRR-P04/setup/evaluator_target_plan.json"),
}

EXPECTED_COUNTS = {"correction": 256, "validation": 192, "fresh_evaluation": 72}
HASH_CONVENTION_SOURCES = (
    Path("scripts/trr_p04/prepare_panel.py"),
    Path("scripts/trr_p04/prepare_evaluator_target.py"),
    Path("src/token_reconstruction/public_activation.py"),
    Path("src/token_reconstruction/p04_training.py"),
    Path("src/token_reconstruction/alpaca_split.py"),
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    candidate = here.parents[1]
    if (candidate / "experiments/TRR-P04").exists():
        return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "experiments/TRR-P04").exists():
        return cwd
    raise RuntimeError("cannot locate P04 repository root")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("metadata ledger must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest_lines(values: Iterable[str]) -> str:
    """P04 ordered digest: UTF-8 value followed by LF, including final LF."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _digest_block(values: Sequence[str]) -> dict[str, Any]:
    values = [str(value) for value in values]
    distinct = sorted(set(values))
    return {
        "ordered_count": len(values),
        "distinct_count": len(distinct),
        "ordered_newline_sha256": _digest_lines(values),
        "ordered_canonical_json_sha256": _canonical_sha256(values),
        "unique_set_canonical_json_sha256": _canonical_sha256(distinct),
    }


def _normalise_path(root: Path, path: Any) -> Any:
    if not isinstance(path, (str, Path)):
        return path
    try:
        candidate = Path(path)
        if candidate.is_absolute():
            return str(candidate.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        pass
    return str(path) if isinstance(path, Path) else path


def _ledger_descriptor(root: Path, path: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": _normalise_path(root, resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
        "schema": metadata.get("schema"),
        "status": metadata.get("status"),
        "task_id": metadata.get("task_id"),
    }


def _code_descriptor(root: Path, path: Path) -> dict[str, Any]:
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    return {
        "path": _normalise_path(root, resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _supplied_descriptor(root: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allowed = (
        "path",
        "bytes",
        "sha256",
        "hash_source",
        "available",
        "required",
        "record_count",
    )
    return {
        key: _normalise_path(root, value[key]) if key == "path" else value[key]
        for key in allowed
        if key in value
    }


def _row_count_summary(rows: Sequence[Mapping[str, Any]], *, anchor_key: str | None = None) -> dict[str, Any]:
    lengths = Counter(int(row["post_bos_token_count"]) for row in rows)
    result: dict[str, Any] = {
        "record_count": len(rows),
        "distinct_post_bos_lengths": len(lengths),
        "count_by_post_bos_length": {str(key): lengths[key] for key in sorted(lengths)},
    }
    if anchor_key is not None:
        result["anchor_record_count"] = sum(bool(row.get(anchor_key, False)) for row in rows)
    return result


def _opaque_hash_reservation(values: Sequence[str], *, source_field: str) -> dict[str, Any]:
    """Return ordered and set views of opaque per-record SHA-256 strings."""
    values = [str(value) for value in values]
    aggregate = _digest_block(values)
    return {
        "available": True,
        "source_field": source_field,
        "ordered_values": values,
        "unique_values": sorted(set(values)),
        **aggregate,
    }


def _unavailable_hash_reservation(*, source_field: str, reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "source_field": source_field,
        "ordered_values": None,
        "unique_values": None,
        "reason": reason,
    }


def _individual_public_hashes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    public_values = [str(row["public_record_sha256"]) for row in rows]
    sequence_values = [str(row["truncated_sequence_sha256"]) for row in rows]
    pair_values = [
        _canonical_sha256(
            {
                "public_record_sha256": public_value,
                "truncated_sequence_sha256": sequence_value,
            }
        )
        for public_value, sequence_value in zip(public_values, sequence_values)
    ]
    return {
        "public_record_sha256": _opaque_hash_reservation(
            public_values, source_field="public_record_sha256"
        ),
        "truncated_sequence_sha256": _opaque_hash_reservation(
            sequence_values, source_field="truncated_sequence_sha256"
        ),
        "ordered_public_sequence_pair_sha256": _opaque_hash_reservation(
            pair_values,
            source_field=(
                "canonical JSON pair of public_record_sha256 and "
                "truncated_sequence_sha256"
            ),
        ),
    }


def _reservation_hashes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "opaque_record_ids": [str(row["record_id"]) for row in rows],
        "opaque_public_record_fingerprints": [str(row["public_record_sha256"]) for row in rows],
        "opaque_truncated_sequence_fingerprints": [
            str(row["truncated_sequence_sha256"]) for row in rows
        ],
    }
    aggregate = {name: _digest_block(values) for name, values in fields.items()}
    individual = _individual_public_hashes(rows)
    individual_digest = _canonical_sha256(individual)
    return {
        **aggregate,
        "individual_opaque_hashes": individual,
        "individual_opaque_hashes_digest_sha256": individual_digest,
        "reservation_digest_sha256": _canonical_sha256(
            {"aggregate": aggregate, "individual_opaque_hashes_digest_sha256": individual_digest}
        ),
    }

def _check_public_row_schema(rows: Sequence[Mapping[str, Any]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError("unexpected P04 public ledger count")
    required = {
        "record_id",
        "public_record_sha256",
        "truncated_sequence_sha256",
        "post_bos_token_count",
    }
    for row in rows:
        if not required.issubset(row):
            raise ValueError("public ledger row is missing required opaque fields")
        for key in ("record_id", "public_record_sha256", "truncated_sequence_sha256"):
            value = row[key]
            if not isinstance(value, str) or not value:
                raise ValueError("public ledger opaque field is not a nonempty string")


def _overlap_counts(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    return {
        "opaque_record_id_overlap_count": len(
            {str(row["record_id"]) for row in left}
            & {str(row["record_id"]) for row in right}
        ),
        "opaque_public_record_fingerprint_overlap_count": len(
            {str(row["public_record_sha256"]) for row in left}
            & {str(row["public_record_sha256"]) for row in right}
        ),
        "opaque_truncated_sequence_fingerprint_overlap_count": len(
            {str(row["truncated_sequence_sha256"]) for row in left}
            & {str(row["truncated_sequence_sha256"]) for row in right}
        ),
    }


def _safe_overlap_ledger_descriptors(root: Path, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in metadata.get("overlap_audit", {}).get("ledgers", []):
        if not isinstance(item, Mapping):
            raise ValueError("invalid target overlap ledger descriptor")
        allowed = (
            "path",
            "bytes",
            "sha256",
            "available",
            "required",
            "text_fingerprint_count",
            "sequence_fingerprint_count",
            "text_hash_collisions",
            "sequence_hash_collisions",
        )
        result.append(
            {
                key: _normalise_path(root, item[key]) if key == "path" else item[key]
                for key in allowed
                if key in item
            }
        )
    return result


def _target_audit_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    audit = metadata.get("overlap_audit", {})
    selected = metadata.get("selected_rows", {})
    return {
        "target_record_count": int(audit.get("target_records", selected.get("count", 0))),
        "target_text_fingerprint_count": int(audit.get("target_text_fingerprint_count", 0)),
        "target_sequence_fingerprint_count": int(audit.get("target_sequence_fingerprint_count", 0)),
        "text_collision_count": int(audit.get("text_collisions", 0)),
        "sequence_collision_count": int(audit.get("sequence_collisions", 0)),
        "overlap_status": audit.get("status"),
        "selected_row_order_sha256": selected.get("row_order_sha256"),
    }


def _public_reservation(
    root: Path,
    name: str,
    path: Path,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    anchor_key: str | None = None,
) -> dict[str, Any]:
    _check_public_row_schema(rows, EXPECTED_COUNTS[name])
    return {
        "counts": _row_count_summary(rows, anchor_key=anchor_key),
        "hashes": _reservation_hashes(rows),
        "ledger": _ledger_descriptor(root, path, metadata),
        "selection_descriptor": _supplied_descriptor(root, metadata.get("selection")),
        "source_code_descriptor": _supplied_descriptor(root, metadata.get("source_code")),
        "metadata_contract": {
            "evaluation_truth_included": bool(metadata.get("evaluation_truth_included", False)),
            "source_text_included": bool(metadata.get("source_text_included", False)),
            "token_ids_included": bool(metadata.get("token_ids_included", False)),
            "selection_adapts_to_scores": bool(
                metadata.get("selection", {}).get("selection_adapts_to_scores", False)
            ),
        },
    }


def build_exchange(root: Path) -> dict[str, Any]:
    loaded = {name: _read_json(root / path) for name, path in INPUT_FILES.items()}
    selection = loaded["selection"]
    pool_manifest = loaded["pool_manifest"]
    correction = loaded["correction"]
    validation = loaded["validation"]
    fresh = loaded["fresh_evaluation"]
    target_audit = loaded["target_selection_audit"]
    private_target_audit = loaded["target_private_selection_audit"]
    target_receipt = loaded["target_preparation_receipt"]
    target_plan = loaded["target_plan"]

    correction_rows = correction.get("records", [])
    validation_rows = validation.get("records", [])
    fresh_rows = fresh.get("records", [])
    if not all(isinstance(rows, list) for rows in (correction_rows, validation_rows, fresh_rows)):
        raise ValueError("P04 public pool records must be arrays")

    public = {
        "correction": _public_reservation(
            root,
            "correction",
            INPUT_FILES["correction"],
            correction,
            correction_rows,
        ),
        "validation": _public_reservation(
            root,
            "validation",
            INPUT_FILES["validation"],
            validation,
            validation_rows,
        ),
        "fresh_evaluation": _public_reservation(
            root,
            "fresh_evaluation",
            INPUT_FILES["fresh_evaluation"],
            fresh,
            fresh_rows,
            anchor_key="anchor",
        ),
    }

    fit_selection = selection.get("pools", {}).get("fit_replay", {})
    fit_manifest = pool_manifest.get("replay", {}).get("record_manifest", {})
    fit_manifest_path_value = fit_manifest.get("path")
    fit_manifest_descriptor: dict[str, Any] = _supplied_descriptor(root, fit_manifest)
    fit_individual: dict[str, Any]
    replay_metadata: dict[str, Any] | None = None
    replay_rows: list[Mapping[str, Any]] | None = None
    if isinstance(fit_manifest_path_value, str) and fit_manifest_path_value:
        replay_path = Path(fit_manifest_path_value).expanduser().resolve()
        replay_metadata = _read_json(replay_path)
        candidate_rows = replay_metadata.get("records")
        if not isinstance(candidate_rows, list) or len(candidate_rows) != int(fit_selection.get("record_count", 1200)):
            raise ValueError("fit replay metadata count is unavailable or inconsistent")
        allowed_replay_fields = {
            "active_token_count",
            "full_token_count",
            "padded_length",
            "post_bos_token_count",
            "record_id",
            "rendered_char_count",
            "rendered_sha256",
            "row_index",
        }
        if any(
            not isinstance(row, Mapping) or set(row) - allowed_replay_fields
            for row in candidate_rows
        ):
            raise ValueError("fit replay metadata contains a non-metadata field")
        if any(
            not isinstance(row.get("record_id"), str)
            or not isinstance(row.get("rendered_sha256"), str)
            or not row.get("rendered_sha256")
            for row in candidate_rows
        ):
            raise ValueError("fit replay metadata lacks opaque rendered fingerprints")
        replay_rows = candidate_rows
        replay_id_digest = _digest_lines(str(row["record_id"]) for row in replay_rows)
        supplied_id_digest = fit_selection.get("record_ids_sha256")
        if supplied_id_digest and replay_id_digest != supplied_id_digest:
            raise ValueError("fit replay record-order digest disagrees with selection metadata")
        fit_individual = {
            "rendered_sha256": _opaque_hash_reservation(
                [str(row["rendered_sha256"]) for row in replay_rows],
                source_field="rendered_sha256",
            ),
            "public_record_sha256": _unavailable_hash_reservation(
                source_field="public_record_sha256",
                reason="replay metadata exposes rendered_sha256 rather than this P04 field name",
            ),
            "truncated_sequence_sha256": _unavailable_hash_reservation(
                source_field="truncated_sequence_sha256",
                reason="not present in the immutable replay metadata ledger",
            ),
        }
        fit_manifest_descriptor = _ledger_descriptor(root, replay_path, replay_metadata)
    else:
        fit_individual = {
            "rendered_sha256": _unavailable_hash_reservation(
                source_field="rendered_sha256", reason="replay metadata path is unavailable"
            ),
            "public_record_sha256": _unavailable_hash_reservation(
                source_field="public_record_sha256", reason="replay metadata path is unavailable"
            ),
            "truncated_sequence_sha256": _unavailable_hash_reservation(
                source_field="truncated_sequence_sha256", reason="replay metadata path is unavailable"
            ),
        }
    fit_replay = {
        "counts": {
            "record_count": int(fit_selection.get("record_count", pool_manifest.get("replay", {}).get("rows", 0))),
            "observation_position_count": int(pool_manifest.get("replay", {}).get("positions", 0)),
        },
        "hashes": {
            "opaque_record_id_order_sha256": fit_selection.get("record_ids_sha256"),
            "individual_hashes_digest_sha256": _canonical_sha256(fit_individual),
        },
        "individual_opaque_hashes": fit_individual,
        "ledger": _ledger_descriptor(root, INPUT_FILES["fit_replay_ledger"], pool_manifest),
        "record_manifest_descriptor": fit_manifest_descriptor,
        "selection_pool_descriptor": _supplied_descriptor(root, fit_selection),
        "hash_provenance": "record_id_order_sha256 is copied from and checked against the linked replay metadata ledger; only its rendered_sha256 field is opened for individual opaque fingerprints",
    }

    public_pool_pairs: dict[str, Any] = {}
    for left_name, right_name in (
        ("correction", "validation"),
        ("correction", "fresh_evaluation"),
        ("validation", "fresh_evaluation"),
    ):
        left_rows = {"correction": correction_rows, "validation": validation_rows, "fresh_evaluation": fresh_rows}[left_name]
        right_rows = {"correction": correction_rows, "validation": validation_rows, "fresh_evaluation": fresh_rows}[right_name]
        public_pool_pairs[f"{left_name}_vs_{right_name}"] = _overlap_counts(left_rows, right_rows)

    target_artifact = target_receipt.get("target_artifact", {})
    target_selection = target_receipt.get("selection", {})
    target_plan_meta = target_receipt.get("plan", {})
    targetfit = {
        "counts": {
            "selected_update_record_count": int(target_selection.get("records", 0)),
            "target_text_fingerprint_count": int(
                target_receipt.get("overlap_audit", {}).get("target_text_fingerprint_count", 0)
            ),
            "target_sequence_fingerprint_count": int(
                target_receipt.get("overlap_audit", {}).get("target_sequence_fingerprint_count", 0)
            ),
            "overlap_ledger_count": len(target_receipt.get("overlap_audit", {}).get("ledgers", [])),
        },
        "hashes": {
            "selected_update_row_order_sha256": target_selection.get("row_order_sha256"),
            "selection_descriptor_sha256": target_selection.get("sha256"),
            "plan_descriptor_sha256": target_plan_meta.get("sha256"),
            "target_update_artifact_sha256_from_receipt": target_artifact.get("sha256"),
        },
        "target_artifact_descriptor_from_receipt": {
            "path": _normalise_path(root, target_artifact.get("path")),
            "bytes": target_artifact.get("bytes"),
            "sha256": target_artifact.get("sha256"),
            "serialized_source_text": bool(target_artifact.get("serialized_source_text", False)),
            "serialized_source_tokens": bool(target_artifact.get("serialized_source_tokens", False)),
            "read_by_reservation_hash_producer": False,
        },
        "individual_opaque_hashes": {
            "public_record_sha256": _unavailable_hash_reservation(
                source_field="public_record_sha256",
                reason="target audit and preparation receipt serialize counts only, with no per-record public fingerprint array",
            ),
            "truncated_sequence_sha256": _unavailable_hash_reservation(
                source_field="truncated_sequence_sha256",
                reason="target audit and preparation receipt serialize counts only, with no per-record sequence fingerprint array",
            ),
        },
        "selection_audit": _ledger_descriptor(
            root, INPUT_FILES["target_selection_audit"], target_audit
        ),
        "private_selection_audit": _ledger_descriptor(
            root, INPUT_FILES["target_private_selection_audit"], private_target_audit
        ),
        "preparation_receipt": _ledger_descriptor(
            root, INPUT_FILES["target_preparation_receipt"], target_receipt
        ),
        "target_plan": _ledger_descriptor(root, INPUT_FILES["target_plan"], target_plan),
        "overlap_audit": _target_audit_summary(target_audit),
        "overlap_ledger_descriptors": _safe_overlap_ledger_descriptors(root, target_audit),
        "metadata_only_boundary": {
            "target_update_tensor_opened": False,
            "target_weights_opened": False,
            "evaluator_truth_opened": False,
            "source_text_or_tokens_opened": False,
        },
    }

    # The public index and its sealed fresh ledger should identify the same ordered
    # panel.  Compare hashes only; do not emit the row IDs.
    selection_panel_rows = selection.get("pools", {}).get("fresh_evaluation", {}).get("records", [])
    if not isinstance(selection_panel_rows, list):
        raise ValueError("selection fresh panel metadata is not an array")
    if len(selection_panel_rows) != len(fresh_rows):
        raise ValueError("selection and fresh panel counts disagree")
    selection_panel_id_hash = _digest_lines(str(row["record_id"]) for row in selection_panel_rows)
    fresh_panel_id_hash = public["fresh_evaluation"]["hashes"]["opaque_record_ids"]["ordered_newline_sha256"]
    selection_panel_fp_hash = _digest_lines(str(row["public_record_sha256"]) for row in selection_panel_rows)
    fresh_panel_fp_hash = public["fresh_evaluation"]["hashes"]["opaque_public_record_fingerprints"]["ordered_newline_sha256"]
    selection_fresh_equivalence = {
        "record_count": len(fresh_rows),
        "ordered_opaque_record_id_hash_equal": selection_panel_id_hash == fresh_panel_id_hash,
        "ordered_opaque_public_fingerprint_hash_equal": selection_panel_fp_hash == fresh_panel_fp_hash,
    }

    exchange: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "READY_FOR_TRR0006_RESERVATION_EXCHANGE",
        "producer": {
            "script": _ledger_descriptor(
                root, Path(__file__).resolve(), {"schema": "producer-script", "status": "frozen-for-replay", "task_id": TASK_ID}
            ),
            "input_scope": "P04 metadata ledgers and the linked immutable replay record-metadata ledger listed below; no observation, source, truth, model-weight, or target-update payload was opened",
        },
        "privacy_boundary": {
            "raw_record_ids_emitted": False,
            "raw_sequence_tokens_emitted": False,
            "source_text_emitted": False,
            "source_or_token_payload_opened": False,
            "target_weight_tensor_opened": False,
            "evaluator_truth_opened": False,
            "P03_holdout_opened": False,
        },
        "hash_conventions": {
            "canonical_json_sha256": "SHA256(UTF-8 json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)); this is the prepare_panel canonical_json/json_sha256 convention used for exchange aggregates",
            "metadata_json_file_serialization": "P04 metadata ledgers are written with json.dumps(value, indent=2, sort_keys=True) and a final LF; their file SHA256 descriptors hash those exact UTF-8 bytes",
            "ordered_opaque_value_sha256": "SHA256 of each declared opaque UTF-8 string followed by LF, in ledger order, including the final LF; this is prepare_panel.digest_lines and public_activation.record_ids_sha256",
            "unique_set_canonical_json_sha256": "SHA256 of canonical JSON for the sorted unique opaque-string set",
            "opaque_fields": {
                "record_id": "metadata record_id; treated as an opaque string and never emitted per record",
                "public_record_fingerprint": "metadata public_record_sha256; an opaque SHA256 string, emitted only as ordered/set values",
                "truncated_sequence_fingerprint": "metadata truncated_sequence_sha256; an opaque SHA256 string, emitted only as ordered/set values",
            },
            "public_text_fingerprint_generation": {
                "pile_plain": "SHA256(raw text UTF-8 bytes); _text_value converts null to empty text and performs no trim or Unicode normalization",
                "finance_chat": "system/user/assistant values convert null to empty text and are stripped; missing user falls back to stripped instruction plus optional '\n\n' plus stripped input, and missing assistant falls back to stripped output; hash compact JSON [system-or-null,user,assistant] with ensure_ascii=False and separators=(',', ':') as UTF-8",
                "alpaca_instruction": "instruction/input/output values convert null to empty text; user is instruction plus optional '\n\n' plus input, capped at 1200 characters, output is capped at 1200 characters; render tokenizer.apply_chat_template([{'role':'user','content':user}], tokenize=False, add_generation_prompt=True) plus output, then SHA256 rendered UTF-8 bytes",
            },
            "sequence_fingerprint_generation": {
                "public_panel": "token IDs are produced with add_special_tokens=False, include the declared BOS at index zero, and hash token_ids[:1+128] (BOS plus at most 128 post-BOS IDs)",
                "target_update_audit": "the target preparation helper uses the same little-endian signed-int32 packing convention; no target per-record sequence values are serialized into this exchange",
                "binary_encoding": "SHA256(struct.pack('<' + 'i'*N, *(int(value) for value in values))); no JSON wrapper and no dtype conversion beyond signed int32 representability",
            },
            "tensor_sha256": "For tensor digests used by P04 runtime assets: detach().cpu().contiguous(), prefix UTF-8 compact sorted JSON {'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}, then append exact contiguous CPU bytes from view(torch.uint8).numpy().tobytes(order='C')",
            "ordered_pair_sha256": "Each emitted ordered pair digest is SHA256 of canonical JSON {'public_record_sha256': value, 'truncated_sequence_sha256': value} using the exchange canonical JSON convention; the ordered arrays retain the original row pairing",
            "fit_replay_supplied_hash": "fit replay opaque_record_id_order_sha256 is copied from the existing P04 selection ledger and checked against the linked replay metadata record order; its individual rendered_sha256 values are emitted when present, while missing P04 sequence fields are explicit unavailable markers",
            "targetfit_supplied_hashes": "target row-order, selection, plan, and target-update artifact SHA256 values are copied from existing P04 audit/receipt metadata; per-record target public/sequence arrays are explicitly unavailable and the target-update tensor is not opened here",
            "implementation_sources": [
                {
                    **_code_descriptor(root, path),
                    "symbols": symbols,
                }
                for path, symbols in (
                    (HASH_CONVENTION_SOURCES[0], ["canonical_json", "json_sha256", "digest_lines", "sequence_sha256"]),
                    (HASH_CONVENTION_SOURCES[1], ["_digest_lines", "_sequence_hash"]),
                    (HASH_CONVENTION_SOURCES[2], ["tensor_sha256", "record_ids_sha256"]),
                    (HASH_CONVENTION_SOURCES[3], ["tensor_sha256", "canonical_hash"]),
                    (HASH_CONVENTION_SOURCES[4], ["historical_user_text", "historical_rendered_text"]),
                )
            ],
        },
        "ledger_descriptors": {
            "selection": _ledger_descriptor(root, INPUT_FILES["selection"], selection),
            "pool_manifest": _ledger_descriptor(root, INPUT_FILES["pool_manifest"], pool_manifest),
            "fit_replay_record_manifest": fit_replay["record_manifest_descriptor"],
            "correction": public["correction"]["ledger"],
            "validation": public["validation"]["ledger"],
            "fresh_evaluation": public["fresh_evaluation"]["ledger"],
            "target_selection_audit": targetfit["selection_audit"],
            "target_private_selection_audit": targetfit["private_selection_audit"],
            "target_preparation_receipt": targetfit["preparation_receipt"],
            "target_plan": targetfit["target_plan"],
        },
        "reservations": {
            "fit_replay": fit_replay,
            "correction": public["correction"],
            "validation": public["validation"],
            "fresh_evaluation": public["fresh_evaluation"],
            "targetfit": targetfit,
        },
        "counts": {
            "selection_panel_record_count": int(selection.get("panel", {}).get("record_count", 0)),
            "selection_panel_anchor_record_count": int(selection.get("panel", {}).get("anchor_record_count", 0)),
            "selection_panel_independent_source_record_count": int(
                selection.get("panel", {}).get("independent_source_records", 0)
            ),
            "selection_identity_content_hash_count": int(
                selection.get("exclusions", {}).get("identity_counts", {}).get("content_hashes", 0)
            ),
            "selection_identity_record_id_count": int(
                selection.get("exclusions", {}).get("identity_counts", {}).get("record_ids", 0)
            ),
            "selection_identity_source_index_count": int(
                selection.get("exclusions", {}).get("identity_counts", {}).get("source_indices", 0)
            ),
            "fit_replay_excluded_record_count": int(selection.get("exclusions", {}).get("fit_replay_ids_added", 0)),
        },
        "overlap_counts": {
            "public_pool_pairs": public_pool_pairs,
            "target_audit": _target_audit_summary(target_audit),
            "target_audit_private_reconciliation": {
                "metadata_count_match": _target_audit_summary(target_audit)["target_record_count"]
                == _target_audit_summary(private_target_audit)["target_record_count"],
                "metadata_text_fingerprint_count_match": _target_audit_summary(target_audit)[
                    "target_text_fingerprint_count"
                ]
                == _target_audit_summary(private_target_audit)["target_text_fingerprint_count"],
                "metadata_sequence_fingerprint_count_match": _target_audit_summary(target_audit)[
                    "target_sequence_fingerprint_count"
                ]
                == _target_audit_summary(private_target_audit)["target_sequence_fingerprint_count"],
                "metadata_collision_counts_match": (
                    _target_audit_summary(target_audit)["text_collision_count"]
                    == _target_audit_summary(private_target_audit)["text_collision_count"]
                    and _target_audit_summary(target_audit)["sequence_collision_count"]
                    == _target_audit_summary(private_target_audit)["sequence_collision_count"]
                ),
            },
        },
        "selection_fresh_panel_reconciliation": selection_fresh_equivalence,
    }

    exchange["exchange_digest_sha256"] = _canonical_sha256(
        {
            "reservations": exchange["reservations"],
            "overlap_counts": exchange["overlap_counts"],
            "selection_fresh_panel_reconciliation": exchange["selection_fresh_panel_reconciliation"],
        }
    )
    return exchange


def _assert_no_raw_row_values(output: Mapping[str, Any], loaded_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    serialized = json.dumps(output, ensure_ascii=False, sort_keys=True)
    for rows in loaded_rows.values():
        for row in rows:
            for key in ("record_id",):
                value = row.get(key)
                if isinstance(value, str) and value and value in serialized:
                    raise AssertionError("reservation exchange contains a raw row value")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/TRR-P04/coordination/reservation_hashes.json"),
    )
    args = parser.parse_args()
    root = _repo_root()
    output = build_exchange(root)
    loaded = {name: _read_json(root / path) for name, path in INPUT_FILES.items()}
    _assert_no_raw_row_values(
        output,
        {
            name: value.get("records", [])
            for name, value in loaded.items()
            if name in {"correction", "validation", "fresh_evaluation"}
        },
    )
    destination = args.output if args.output.is_absolute() else root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(destination)
    print(output["exchange_digest_sha256"])


if __name__ == "__main__":
    main()
