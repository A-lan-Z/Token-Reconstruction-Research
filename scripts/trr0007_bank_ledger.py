"""Verify the final public TRR-0007 broader-bank identity ledgers.

The verifier reads only JSON identity/provenance ledgers.  It does not load
source rows, token tensors, model weights, or truth.  Both the count-only
inventory and fresh selector call it before loading a tokenizer or Arrow
source so the final v5 parent/source/sequence exclusion contract cannot be
silently replaced by a stale generic exclusion file.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any


TASK_ID = "TRR-0007"
FINAL_BANK_SCHEMA = "token-reconstruction.trr0007-final-bank-ledger.v1"
EXCLUSION_SCHEMA = "token-reconstruction.trr0007-public-bank-exclusions.v1"
BANK_RECEIPT_SCHEMA = "token-reconstruction.trr0007-public-bank-receipt-binding.v1"
CORPUS_SCHEMA = "token-reconstruction.trr0005-public-corpus-plan.v1"
BANK_STATUS = "METADATA_BANK_READY_FOR_REAL_P0_CAPTURE"
EXCLUSION_STATUS = "PUBLIC_PARENT_SELECTION_EXCLUSIONS_BOUND"
CORPUS_STATUS = "PREPARED_PUBLIC_DATA_NO_MODEL_FORWARD"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# These are the reviewed final-v5 ledger hashes.  They make a stale v1-v4
# directory fail before any public source/tokenizer is opened.
FINAL_V5_SHA256 = {
    "exclusion_manifest": "bd1359f1184091570023e22a7682d1f97c08f8f05e47f69f6b3e6be089cd0181",
    "selected_parent_rows": "7de2ce7fca1d9489f652e0229b7cbfeacd625ca6ebd37c56d4405962da79f547",
    "corpus_plan": "35d74424df60a1d62f7962850eaa3132fa57dd7001024f915cc4fc470a6d0e76",
}
EXPECTED_EXCLUSION_COUNTS = {
    "record_ids": 2449,
    "source_row_keys": 1848,
    "opaque_sequence_or_reservation_digests": 4073,
}
EXPECTED_PARENT_ROWS = 120
EXPECTED_PARENT_ROWS_BY_DOMAIN = {
    "controlled_pile_context": 60,
    "controlled_finance_context": 60,
}

PREFIX_LEDGER_SCHEMA = "token-reconstruction.trr0007-public-fitting-prefix-exclusions.v1"
PREFIX_LEDGER_STATUS = "PUBLIC_FIT_PREFIX_HASHES_ONLY"
PREFIX_LEDGER_V3_SHA256 = "c4993d24d838ce2635b28b6736b85ffa849045f37e3a1e38b3904f3cbdb709e1"
PREFIX_LEDGER_V3_RELATIVE_PATH = "experiments/TRR-0007/support/public_fit_prefix_exclusions_v3.json"
EXPECTED_PREFIX_COUNTS = {
    "fit_artifacts": 2,
    "rows_per_artifact": [1200, 1200],
    "eligible_rows_per_artifact": [350, 350],
    "eligible_rows_total_across_artifacts": 700,
    "union_hashes_by_original_style": {"finance": 179, "other": 78, "pile": 213},
    "collector_hashes_by_fresh_style": {"pile": 470, "finance": 470},
    "union_hashes_total": 470,
}


class BankLedgerError(ValueError):
    """Raised when the reviewed final public bank ledgers are unavailable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise BankLedgerError(f"{label} is unavailable: {resolved}")
    record = {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        raise BankLedgerError(
            f"{label} hash is not the reviewed final v5 artifact: "
            f"expected {expected_sha256}, got {record['sha256']}"
        )
    return record


def _json(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(str(record["path"])).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BankLedgerError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise BankLedgerError(f"{label} must be a JSON object")
    return value


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BankLedgerError(f"{label} is not a lowercase SHA-256")
    return value


def _verify_exclusion_manifest(value: Mapping[str, Any]) -> dict[str, int]:
    if (
        value.get("schema") != EXCLUSION_SCHEMA
        or value.get("task_id") != TASK_ID
        or value.get("status") != EXCLUSION_STATUS
    ):
        raise BankLedgerError("final parent exclusion manifest identity/status changed")
    for key in ("private_truth_accessed", "public_development_truth_opened", "source_text_retained"):
        if value.get(key) is not False:
            raise BankLedgerError(f"final parent exclusion manifest is open: {key}")
    counts = value.get("selector_exclusion_set_counts")
    if counts != EXPECTED_EXCLUSION_COUNTS:
        raise BankLedgerError("final parent exclusion set counts changed")
    sets = value.get("selector_exclusion_sets")
    if not isinstance(sets, Mapping):
        raise BankLedgerError("final parent exclusion sets are absent")
    for key, expected in EXPECTED_EXCLUSION_COUNTS.items():
        entries = sets.get(key)
        if not isinstance(entries, list) or len(entries) != expected or len(set(map(str, entries))) != expected:
            raise BankLedgerError(f"final parent exclusion set is malformed: {key}")
    binding = value.get("exclusion_binding")
    if not isinstance(binding, Mapping) or binding.get("opaque_p04_exchange_applied") is not True:
        raise BankLedgerError("final parent exclusion provenance is incomplete")
    return dict(EXPECTED_EXCLUSION_COUNTS)


def _verify_parent_rows(value: Mapping[str, Any]) -> dict[str, Any]:
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_PARENT_ROWS:
        raise BankLedgerError("final selected-parent ledger row count changed")
    required = {
        "dataset_key", "domain", "row_index", "source_record_id",
        "rendered_sha256", "constructed_sequence_sha256",
    }
    source_ids: set[str] = set()
    sequence_hashes: set[str] = set()
    by_domain: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise BankLedgerError(f"final selected-parent row {index} is incomplete")
        domain = row.get("domain")
        if domain not in EXPECTED_PARENT_ROWS_BY_DOMAIN:
            raise BankLedgerError(f"final selected-parent domain changed: {domain}")
        source_id = row.get("source_record_id")
        if not isinstance(source_id, str) or not source_id or source_id in source_ids:
            raise BankLedgerError("final selected-parent source IDs are duplicated or invalid")
        for key in ("rendered_sha256", "constructed_sequence_sha256"):
            _digest(row.get(key), label=f"final selected-parent {key}")
        if isinstance(row.get("row_index"), bool) or not isinstance(row.get("row_index"), int) or row["row_index"] < 0:
            raise BankLedgerError("final selected-parent source row index is invalid")
        source_ids.add(source_id)
        sequence_hash = str(row["constructed_sequence_sha256"])
        if sequence_hash in sequence_hashes:
            raise BankLedgerError("final selected-parent constructed sequences are duplicated")
        sequence_hashes.add(sequence_hash)
        by_domain[str(domain)] += 1
    if dict(by_domain) != EXPECTED_PARENT_ROWS_BY_DOMAIN:
        raise BankLedgerError("final selected-parent domain counts changed")
    return {
        "rows": EXPECTED_PARENT_ROWS,
        "rows_by_domain": dict(EXPECTED_PARENT_ROWS_BY_DOMAIN),
    }


def _verify_corpus_plan(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema") != CORPUS_SCHEMA
        or value.get("task_id") != "TRR-0005"
        or value.get("status") != CORPUS_STATUS
    ):
        raise BankLedgerError("final broader-bank corpus plan identity/status changed")
    support = value.get("trr0007_support")
    if not isinstance(support, Mapping):
        raise BankLedgerError("final broader-bank support provenance is absent")
    access = support.get("access_contract")
    if not isinstance(access, Mapping):
        raise BankLedgerError("final broader-bank access contract is absent")
    for key in ("private_truth_accessed", "public_development_rows_used_for_selection", "reserved_holdout_rows_scanned", "source_text_retained", "target_weights_accessed"):
        if access.get(key) is not False:
            raise BankLedgerError(f"final broader-bank access contract is open: {key}")
    if support.get("selected_token_id_count") != 3600 or support.get("natural_slot_count") != 1080:
        raise BankLedgerError("final broader-bank geometry/counts changed")
    if support.get("replacement_policy", {}).get("total_occurrences") != 3600:
        raise BankLedgerError("final broader-bank replacement count changed")


def load_prefix_exclusion_ledger(
    *,
    repository_root: Path,
    path: Path,
) -> dict[str, Any]:
    """Verify the reviewed all-bank/all-style public fitting prefix ledger.

    The ledger is identity-only.  Its collector buckets repeat the union of
    all 128-token fitting prefixes under each future Pile and Finance style,
    which is required because a prefix collision is a cross-style exclusion.
    """

    root = Path(repository_root).expanduser().resolve()
    record = _record(path, label="public fitting prefix exclusion ledger", expected_sha256=PREFIX_LEDGER_V3_SHA256)
    task_root = (root / "experiments" / TASK_ID).resolve()
    try:
        Path(record["path"]).relative_to(task_root)
    except ValueError as exc:
        raise BankLedgerError("public fitting prefix ledger is outside the task root") from exc
    payload = _json(record, label="public fitting prefix exclusion ledger")
    if (
        payload.get("schema") != PREFIX_LEDGER_SCHEMA
        or payload.get("task_id") != TASK_ID
        or payload.get("status") != PREFIX_LEDGER_STATUS
    ):
        raise BankLedgerError("public fitting prefix ledger identity/status changed")
    sequence = payload.get("sequence_convention")
    if not isinstance(sequence, Mapping) or sequence.get("hash_key") != "final_sequence_sha256" or sequence.get("hash_algorithm") != "SHA-256" or sequence.get("prefix_tokens_including_bos") != 128 or sequence.get("active_rows_only") is not True:
        raise BankLedgerError("public fitting prefix sequence convention changed")
    privacy = payload.get("privacy_boundary")
    if not isinstance(privacy, Mapping) or privacy.get("public_fit_artifacts_only") is not True:
        raise BankLedgerError("public fitting prefix ledger privacy boundary changed")
    for key in ("contains_source_text", "contains_record_ids", "contains_source_indices", "contains_token_ids", "contains_target_labels", "contains_truth", "contains_model_weights", "contains_activations"):
        if privacy.get(key) is not False:
            raise BankLedgerError(f"public fitting prefix ledger contains forbidden payload: {key}")
    execution = payload.get("execution")
    if not isinstance(execution, Mapping) or execution.get("fresh_source_rows_opened") is not False or execution.get("truth_opened") is not False or execution.get("model_loaded") is not False or execution.get("network_used") is not False:
        raise BankLedgerError("public fitting prefix ledger execution boundary changed")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise BankLedgerError("public fitting prefix ledger counts are absent")
    for key, expected in EXPECTED_PREFIX_COUNTS.items():
        if counts.get(key) != expected:
            raise BankLedgerError(f"public fitting prefix ledger count changed: {key}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise BankLedgerError("public fitting prefix artifact list changed")
    if {row.get("artifact_id") for row in artifacts if isinstance(row, Mapping)} != {"current_enriched_public_fit", "improved_public_fit"}:
        raise BankLedgerError("public fitting prefix artifact identities changed")
    collectors = payload.get("collector_exclusions_by_fresh_style")
    if not isinstance(collectors, Mapping):
        raise BankLedgerError("public fitting prefix collector buckets are absent")
    for style, expected in EXPECTED_PREFIX_COUNTS["collector_hashes_by_fresh_style"].items():
        entries = collectors.get(style)
        if not isinstance(entries, list) or len(entries) != expected:
            raise BankLedgerError(f"public fitting prefix collector bucket changed: {style}")
        values = []
        for index, item in enumerate(entries):
            if not isinstance(item, Mapping) or set(item) != {"final_sequence_sha256"}:
                raise BankLedgerError(f"public fitting prefix collector entry is malformed: {style}/{index}")
            values.append(_digest(item.get("final_sequence_sha256"), label=f"public fitting prefix {style}/{index}"))
        if len(set(values)) != expected:
            raise BankLedgerError(f"public fitting prefix collector bucket is duplicated: {style}")
    return {
        "schema": PREFIX_LEDGER_SCHEMA,
        "task_id": TASK_ID,
        "status": PREFIX_LEDGER_STATUS,
        "file": record,
        "sequence_convention": {
            "hash_key": "final_sequence_sha256",
            "hash_algorithm": "SHA-256",
            "prefix_tokens_including_bos": 128,
            "active_rows_only": True,
        },
        "counts": dict(EXPECTED_PREFIX_COUNTS),
        "collector_styles": ["pile", "finance"],
        "all_bank_all_style_union": True,
    }


def load_final_bank_ledgers(
    *,
    repository_root: Path,
    exclusion_manifest: Path,
    selected_parent_rows: Path,
    corpus_plan: Path,
) -> dict[str, Any]:
    """Verify final v5 JSON ledgers and return identity-only descriptors."""

    root = Path(repository_root).expanduser().resolve()
    paths = {
        "exclusion_manifest": Path(exclusion_manifest),
        "selected_parent_rows": Path(selected_parent_rows),
        "corpus_plan": Path(corpus_plan),
    }
    records = {
        key: _record(path, label=f"final bank {key}", expected_sha256=FINAL_V5_SHA256[key])
        for key, path in paths.items()
    }
    # Keep a root argument in the API and reject paths that escape only through
    # symlink resolution.  The reviewed ledgers live in the task tree.
    task_root = (root / "experiments" / TASK_ID).resolve()
    for key, record in records.items():
        try:
            Path(record["path"]).relative_to(task_root)
        except ValueError as exc:
            raise BankLedgerError(f"final bank {key} is outside the task root") from exc
    exclusion = _json(records["exclusion_manifest"], label="final parent exclusion manifest")
    parents = _json(records["selected_parent_rows"], label="final selected-parent ledger")
    corpus = _json(records["corpus_plan"], label="final broader-bank corpus plan")
    exclusion_counts = _verify_exclusion_manifest(exclusion)
    parent_summary = _verify_parent_rows(parents)
    _verify_corpus_plan(corpus)
    return {
        "schema": FINAL_BANK_SCHEMA,
        "task_id": TASK_ID,
        "status": BANK_STATUS,
        "files": records,
        "exclusion_set_counts": exclusion_counts,
        "selected_parent_rows": parent_summary,
        "source_and_sequence_ledgers_verified": True,
    }
