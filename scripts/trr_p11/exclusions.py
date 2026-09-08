#!/usr/bin/env python3
"""TRR-P11 metadata-only exclusion audit.

This module is a narrow continuation of the P10 identity union.  It reuses the
P10 generic metadata walker, but replaces its broad P04 walker with a strict
producer-aware loader, binds every previously opened aggregate panel to its
selection ledger, and provides one optional public-token H128 curator pass.

The curator pass may read only a public fit payload's ``token_ids`` and
``attention_mask`` tensors.  It never reads activations, model weights, source
text, labels, or evaluation truth, and it never serializes token values.
Coverage remains explicitly partial until every missing producer/row mapping is
resolved.  This module never selects new sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from scripts.trr_p10 import build_exclusion_audit as p10


TASK_ID = "TRR-P11"
SCHEMA = "token-reconstruction.trr-p11-identity-exclusion-audit.v1"
P04_PATH = "experiments/TRR-0006/coordination/p04_reservation_hashes.json"
P04_SHA256 = "98f8dfcab0977b4bcafa47d97a86a410ab37359b897b9b553746afa7df5c7904"
P04_PRODUCER_SHA256 = "a622db694e0efeac0eb8cfe4ab2004e0490fe21e0c03b5ac1e11de603f1b53eb"
P04_PRODUCER_COMMIT = "561fc5ce9913af63824b9e4ee9c22063b147df20"
P04_HELPER_OBJECTS: tuple[tuple[str, str, str], ...] = (
    ("scripts/trr_p04/prepare_panel.py", "f423ef596a718a7c8a8480e6211295b97bdfd806", "26c003fc37a80c549ca04ebbf0dd629ae09026fad5f4afc21af0adcca72db97f"),
    ("scripts/trr_p04/prepare_evaluator_target.py", "63d4016f2e555e8460ea566cf967237dc94fb0c3", "bbf4fe9ac443f49f11c12f4a2fd3a5e5eef2cd67bf46573384eb13ffb2cbbd04"),
    ("src/token_reconstruction/p04_training.py", "15d847c40469adbcbd1688014703b204848f880f", "f214dc02d4b3ba5854cd87174c61dd24f02a75a486ce7a297dd3d1d9cabe7486"),
)
P04_LEDGER_OBJECTS: dict[str, tuple[str, str, int]] = {
    "correction": ("experiments/TRR-P04/setup/public-pools-r2/correction_records.json", "2dad95db8f1f796a476e812e6d7eb0a232a325852e6a17ddc3187088c72c2ea2", 256),
    "validation": ("experiments/TRR-P04/setup/public-pools-r2/validation_records.json", "e9aca619f9973f5131c60086f525ba899ec4788561fce68480427e9cdc26f3fc", 192),
    "fresh_evaluation": ("experiments/TRR-P04/setup/public-pools-r2/fresh_panel_index.json", "e3eb40cd0d18529c98d274eb102880f6480ddf055d3da52422a304a0ff6935f1", 72),
}
BOS_TOKEN_ID = 128000

# Six aggregate panels whose individual rows are represented by an immutable
# selection ledger.  PR20 has one extra binding layer and is handled below.
AGGREGATE_BINDINGS: tuple[dict[str, str], ...] = (
    {
        "label": "trr0006_panel",
        "panel": "experiments/TRR-0006/panel_capture_v1/panel.json",
        "selection": "experiments/TRR-0006/source_selection.json",
    },
    {
        "label": "trr0006_public_observation_panel",
        "panel": "experiments/TRR-0006/public_observations_v1/panel.json",
        "selection": "experiments/TRR-0006/source_selection.json",
    },
    {
        "label": "trr0007_eval_panel",
        "panel": "experiments/TRR-0007/evaluation/public_observations/panel.json",
        "selection": "experiments/TRR-0007/selection/source_selection.json",
    },
    {
        "label": "trr0008_eval_panel",
        "panel": "experiments/TRR-0008/evaluation/public_observations_v1/panel.json",
        "selection": "experiments/TRR-0008/selection/source_selection.json",
    },
    {
        "label": "trr0009_eval_panel",
        "panel": "experiments/TRR-0009/evaluation/public_observations_v2/panel.json",
        "selection": "experiments/TRR-0009/selection_v2/source_selection.json",
    },
    {
        "label": "pr20_source_panel",
        "panel": "@pr20/experiments/TRR-0010/evaluation/public_capture_watchdog_r5/observations_v1/panel.json",
        "selection": "@pr20/experiments/TRR-0010/evaluation/source_selection.json",
    },
)

_HEX = set("0123456789abcdefABCDEF")


class ExclusionAuditError(p10.AuditError):
    """Raised when a P11 exclusion invariant cannot be established."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExclusionAuditError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExclusionAuditError(f"{label} root is not an object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _digest_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve(path: str, *, root: Path, pr20_root: Path | None) -> Path:
    if path.startswith("@pr20/"):
        if pr20_root is None:
            raise ExclusionAuditError("@pr20 path requested without --pr20-root")
        return (pr20_root / path.removeprefix("@pr20/")).resolve()
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _descriptor_path(
    descriptor: Mapping[str, Any], *, root: Path, pr20_root: Path | None, label: str
) -> Path:
    if not isinstance(descriptor, Mapping):
        raise ExclusionAuditError(f"{label} descriptor is malformed")
    raw = descriptor.get("path")
    if not isinstance(raw, str) or not raw:
        raise ExclusionAuditError(f"{label} descriptor has no path")
    path = _resolve(raw, root=root, pr20_root=pr20_root)
    if path.is_symlink() or not path.is_file():
        raise ExclusionAuditError(f"{label} descriptor target is unavailable: {path}")
    actual_bytes = path.stat().st_size
    actual_sha = p10.sha256_file(path)
    try:
        declared_bytes = int(descriptor["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExclusionAuditError(f"{label} descriptor bytes are malformed") from exc
    if declared_bytes != actual_bytes or descriptor.get("sha256") != actual_sha:
        raise ExclusionAuditError(f"{label} descriptor changed")
    return path


def _new_bundle(spec: p10.SourceSpec, path: Path, value: Mapping[str, Any]) -> p10.IdentityBundle:
    return p10.IdentityBundle(
        label=spec.label,
        role=spec.role,
        path=path,
        sha256=p10.sha256_file(path),
        bytes=path.stat().st_size,
        schema=value.get("schema") if isinstance(value.get("schema"), str) else None,
        status=value.get("status") if isinstance(value.get("status"), str) else None,
    )


def _verify_p04_opaque_summary(info: Mapping[str, Any], *, label: str) -> list[str]:
    """Check the producer's ordered/set commitments and return its values.

    Values remain in memory only long enough to insert them into the identity
    set.  Receipts expose counts/commitments, never the opaque arrays.
    """
    if info.get("available") is not True:
        return []
    values = info.get("ordered_values")
    unique = info.get("unique_values")
    if not isinstance(values, list) or not isinstance(unique, list):
        raise ExclusionAuditError(f"P04 {label} lacks ordered/unique opaque values")
    if any(not _is_sha256(value) for value in values + unique):
        raise ExclusionAuditError(f"P04 {label} contains a malformed opaque SHA")
    expected_unique = sorted(set(value.casefold() for value in values))
    actual_unique = [value.casefold() for value in unique]
    if actual_unique != expected_unique:
        raise ExclusionAuditError(f"P04 {label} unique values are not sorted/deduplicated")
    if info.get("ordered_count") != len(values) or info.get("distinct_count") != len(expected_unique):
        raise ExclusionAuditError(f"P04 {label} count commitment mismatch")
    if info.get("ordered_canonical_json_sha256") != _json_digest(values):
        raise ExclusionAuditError(f"P04 {label} ordered canonical commitment mismatch")
    if info.get("ordered_newline_sha256") != _digest_lines(values):
        raise ExclusionAuditError(f"P04 {label} ordered newline commitment mismatch")
    if info.get("unique_set_canonical_json_sha256") != _json_digest(expected_unique):
        raise ExclusionAuditError(f"P04 {label} unique canonical commitment mismatch")
    return [value.casefold() for value in values]


@dataclass(frozen=True)
class P04Load:
    bundle: p10.IdentityBundle
    proof: dict[str, Any]
    gaps: list[dict[str, Any]]


def _git_object_bytes(root: Path, commit: str, relative_path: str) -> bytes | None:
    """Read a retained Git object without materializing it in the worktree."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative_path}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _p04_producer_git_proof(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    producer_bytes = _git_object_bytes(root, P04_PRODUCER_COMMIT, "scripts/trr0004_p04_reservation_hashes.py")
    records.append({
        "path": "scripts/trr0004_p04_reservation_hashes.py",
        "commit": P04_PRODUCER_COMMIT,
        "expected_sha256": P04_PRODUCER_SHA256,
        "available": producer_bytes is not None,
        "sha256_match": producer_bytes is not None and hashlib.sha256(producer_bytes).hexdigest() == P04_PRODUCER_SHA256,
    })
    for relative_path, commit, expected in P04_HELPER_OBJECTS:
        data = _git_object_bytes(root, commit, relative_path)
        records.append({
            "path": relative_path,
            "commit": commit,
            "expected_sha256": expected,
            "available": data is not None,
            "sha256_match": data is not None and hashlib.sha256(data).hexdigest() == expected,
        })
    # These two helpers are part of the current P11 checkout and are also
    # listed by the P04 producer as convention sources.
    for relative_path, expected in (
        ("src/token_reconstruction/public_activation.py", "3be43dd5b6a5918a48f289971ab2a833977a7f9d87094e6001c5c7cdc60ab76f"),
        ("src/token_reconstruction/alpaca_split.py", "fa9a15fd4cf92ffa06be3bd77888324536180e8a2fd2c43ae14f20e470a3626a"),
    ):
        path = root / relative_path
        available = path.is_file() and not path.is_symlink()
        records.append({
            "path": relative_path,
            "source": "current_worktree",
            "expected_sha256": expected,
            "available": available,
            "sha256_match": available and p10.sha256_file(path) == expected,
        })
    return {
        "records": records,
        "all_expected_source_bytes_verified": all(item["available"] and item["sha256_match"] for item in records),
    }


def _recover_p04_ledger(root: Path, pool: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    relative_path, expected_sha, expected_count = P04_LEDGER_OBJECTS[pool]
    data = _git_object_bytes(root, "81542a6ac87b22ca3a1b9a48dacf1e0e0afb1bf3", relative_path)
    if data is None or hashlib.sha256(data).hexdigest() != expected_sha:
        return None, {"pool": pool, "available": False, "expected_sha256": expected_sha, "record_count": expected_count}
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExclusionAuditError(f"recovered P04 {pool} ledger is invalid JSON") from exc
    rows = value.get("records")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ExclusionAuditError(f"recovered P04 {pool} ledger record count changed")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("record_id"), str)
            or not row["record_id"]
            or not _is_sha256(row.get("public_record_sha256"))
            or not _is_sha256(row.get("truncated_sequence_sha256"))
            or not isinstance(row.get("row_index"), int)
            or not isinstance(row.get("full_token_count"), int)
            or not isinstance(row.get("post_bos_token_count"), int)
        ):
            raise ExclusionAuditError(f"recovered P04 {pool} ledger row {index} is malformed")
    return value, {
        "pool": pool,
        "available": True,
        "commit": "81542a6ac87b22ca3a1b9a48dacf1e0e0afb1bf3",
        "path": relative_path,
        "sha256": expected_sha,
        "record_count": len(rows),
        "per_record_identity_fields": ["record_id", "public_record_sha256", "row_index", "dataset_id", "dataset_revision", "truncated_sequence_sha256"],
    }


def _consumer_proof(root: Path) -> dict[str, Any]:
    """Record surviving downstream proof for the H129 convention."""
    files = (
        "scripts/trr0005_produce_confirmation.py",
        "scripts/trr0008_select_public.py",
        "scripts/trr0009_select_public.py",
    )
    records: list[dict[str, Any]] = []
    for relative in files:
        path = root / relative
        if not path.is_file():
            records.append({"path": relative, "available": False})
            continue
        text = path.read_text(encoding="utf-8")
        if relative.endswith("produce_confirmation.py"):
            convention = "_sequence_digest" in text and "torch.int32" in text
        else:
            convention = "SEQUENCE_TOKENS + 1" in text and "_p06_sequence_digest" in text
        records.append(
            {
                "path": relative,
                "available": True,
                "sha256": p10.sha256_file(path),
                "h129_consumer_pattern_verified": convention,
            }
        )
    return {"files": records, "all_h129_consumer_patterns_verified": all(item.get("h129_consumer_pattern_verified") is True for item in records)}


def load_p04(root: Path) -> P04Load:
    path = (root / P04_PATH).resolve()
    if not path.is_file() or path.is_symlink():
        raise ExclusionAuditError(f"required P04 exchange is unavailable: {path}")
    actual_sha = p10.sha256_file(path)
    if actual_sha != P04_SHA256:
        raise ExclusionAuditError("P04 exchange SHA-256 changed")
    value = _read_json(path, label="P04 exchange")
    if value.get("schema") != "token-reconstruction.trr-p04-reservation-hashes.v1":
        raise ExclusionAuditError("P04 exchange schema changed")
    if value.get("task_id") != "TRR-P04":
        raise ExclusionAuditError("P04 exchange task identity changed")
    if not _is_sha256(value.get("exchange_digest_sha256")):
        raise ExclusionAuditError("P04 exchange digest is malformed")
    conventions = value.get("hash_conventions")
    if not isinstance(conventions, Mapping):
        raise ExclusionAuditError("P04 hash conventions are absent")
    required_fragments = {
        "canonical_json_sha256": "json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)",
        "ordered_opaque_value_sha256": "SHA256 of each declared opaque UTF-8 string followed by LF",
        "ordered_pair_sha256": "canonical JSON {'public_record_sha256': value, 'truncated_sequence_sha256': value}",
        "unique_set_canonical_json_sha256": "SHA256 of canonical JSON for the sorted unique opaque-string set",
    }
    convention_checks = {
        key: isinstance(conventions.get(key), str) and fragment in conventions[key]
        for key, fragment in required_fragments.items()
    }
    seq = conventions.get("sequence_fingerprint_generation")
    binary = conventions.get("sequence_fingerprint_generation", {}).get("binary_encoding", "") if isinstance(seq, Mapping) else ""
    panel = conventions.get("sequence_fingerprint_generation", {}).get("public_panel", "") if isinstance(seq, Mapping) else ""
    convention_checks.update(
        {
            "signed_int32_binary": "struct.pack('<' + 'i'*N" in binary,
            "bos_plus_128_h129": "token_ids[:1+128]" in panel and "BOS" in panel,
        }
    )
    if not all(convention_checks.values()):
        raise ExclusionAuditError("P04 declared hash convention changed")

    spec = p10.SourceSpec("trr0006_p04_opaque", "opaque_reservation", P04_PATH)
    bundle = _new_bundle(spec, path, value)
    gaps: list[dict[str, Any]] = []
    pool_proof: dict[str, Any] = {}
    ledger_proofs: dict[str, Any] = {}
    reservations = value.get("reservations")
    if not isinstance(reservations, Mapping):
        raise ExclusionAuditError("P04 reservations are absent")
    for pool in ("correction", "validation", "fresh_evaluation"):
        reservation = reservations.get(pool)
        if not isinstance(reservation, Mapping):
            raise ExclusionAuditError(f"P04 {pool} reservation is absent")
        hashes = reservation.get("hashes", {}).get("individual_opaque_hashes") if isinstance(reservation.get("hashes"), Mapping) else None
        if not isinstance(hashes, Mapping):
            raise ExclusionAuditError(f"P04 {pool} individual opaque hashes are absent")
        public = _verify_p04_opaque_summary(hashes.get("public_record_sha256", {}), label=f"{pool}.public_record_sha256")
        truncated = _verify_p04_opaque_summary(hashes.get("truncated_sequence_sha256", {}), label=f"{pool}.truncated_sequence_sha256")
        pair = _verify_p04_opaque_summary(hashes.get("ordered_public_sequence_pair_sha256", {}), label=f"{pool}.ordered_pair")
        if len(public) != len(truncated) or len(public) != len(pair):
            raise ExclusionAuditError(f"P04 {pool} paired opaque counts disagree")
        reservation_hashes = reservation.get("hashes")
        if not isinstance(reservation_hashes, Mapping):
            raise ExclusionAuditError(f"P04 {pool} hash block is malformed")
        individual = reservation_hashes.get("individual_opaque_hashes")
        if not isinstance(individual, Mapping):
            raise ExclusionAuditError(f"P04 {pool} individual hash block is malformed")
        individual_digest = reservation_hashes.get("individual_opaque_hashes_digest_sha256")
        if individual_digest != _json_digest(individual):
            raise ExclusionAuditError(f"P04 {pool} individual hash digest mismatch")
        aggregate = {
            key: reservation_hashes.get(key)
            for key in ("opaque_record_ids", "opaque_public_record_fingerprints", "opaque_truncated_sequence_fingerprints")
        }
        reservation_digest = reservation_hashes.get("reservation_digest_sha256")
        if reservation_digest != _json_digest({"aggregate": aggregate, "individual_opaque_hashes_digest_sha256": individual_digest}):
            raise ExclusionAuditError(f"P04 {pool} reservation digest mismatch")
        namespace = p10.Namespace("*", "*", "*", "*")
        for digest in public:
            bundle.add("rendered_sha256", digest, namespace)
        for digest in truncated:
            bundle.add("h129_sequence_sha256", digest, namespace)
        ledger, ledger_proof = _recover_p04_ledger(root, pool)
        ledger_proofs[pool] = ledger_proof
        if ledger is None:
            gaps.append({"field": f"{pool}.individual_record_identity", "reason": "historical P04 ledger Git object is unavailable"})
        else:
            rows = ledger["records"]
            row_public = [str(row["public_record_sha256"]).casefold() for row in rows]
            row_truncated = [str(row["truncated_sequence_sha256"]).casefold() for row in rows]
            if row_public != public or row_truncated != truncated:
                raise ExclusionAuditError(f"P04 {pool} opaque arrays do not bind recovered ledger order")
            row_pairs = [_json_digest({"public_record_sha256": a, "truncated_sequence_sha256": b}) for a, b in zip(row_public, row_truncated)]
            if row_pairs != pair:
                raise ExclusionAuditError(f"P04 {pool} ordered pair array does not bind recovered ledger order")
            eligible_by_length = 0
            for row in rows:
                if int(row["full_token_count"]) != int(row["post_bos_token_count"]) + 1:
                    raise ExclusionAuditError(f"P04 {pool} full/post-BOS length identity changed")
                style = p10._style(row.get("style")) or "*"
                row_namespace = p10.Namespace(style, str(row.get("dataset_id", "*")), "train", str(row.get("dataset_revision", "*")))
                bundle.add("record_id", str(row["record_id"]), row_namespace)
                # rendered/H129 are global fields; the wildcard entries above
                # already carry them.  Do not duplicate global values per row
                # namespace in the receipt counts.
                bundle.add("source_index", int(row["row_index"]), row_namespace)
                if int(row["full_token_count"]) >= 128:
                    eligible_by_length += 1
            ledger_proof["eligible_rows_by_length_for_h128"] = eligible_by_length
            ledger_proof["h128_values_present"] = False
            gaps.append({"field": f"{pool}.h128_sequence_sha256", "reason": "recovered public ledger proves every row is H128-eligible by length but publishes H129 only; exact H128 requires public token inputs and producer preprocessing"})
        pool_proof[pool] = {
            "public_record_sha256": {"available": True, "count": len(public)},
            "truncated_sequence_sha256": {"available": True, "count": len(truncated), "mapping": "H129 = BOS plus 128 post-BOS IDs"},
            "ordered_pair": {"available": True, "count": len(pair)},
            "recovered_individual_ledger": ledger_proof,
        }
    fit_replay = reservations.get("fit_replay")
    fit_hashes = fit_replay.get("individual_opaque_hashes") if isinstance(fit_replay, Mapping) else None
    if not isinstance(fit_hashes, Mapping):
        raise ExclusionAuditError("P04 fit_replay individual hashes are absent")
    fit_rendered = fit_hashes.get("rendered_sha256")
    if not isinstance(fit_rendered, Mapping):
        raise ExclusionAuditError("P04 fit_replay rendered hash descriptor is absent")
    fit_values = _verify_p04_opaque_summary(fit_rendered, label="fit_replay.rendered_sha256")
    for digest in fit_values:
        bundle.add("rendered_sha256", digest, p10.Namespace())
    gaps.extend(
        [
            {"field": "fit_replay.public_record_sha256", "reason": "replay metadata exposes rendered_sha256 under a different field; no per-record public field mapping is emitted"},
            {"field": "fit_replay.truncated_sequence_sha256", "reason": "immutable replay metadata contains no per-record sequence array"},
            {"field": "targetfit.public_record_sha256", "reason": "target preparation/audit serialize counts only"},
            {"field": "targetfit.truncated_sequence_sha256", "reason": "target preparation/audit serialize counts only"},
        ]
    )
    bundle.metadata_notes.extend(
        [
            "P04 public_record_sha256 opaque values admitted as rendered_sha256 only for correction/validation/fresh_evaluation, after ordered/set commitment checks",
            "P04 truncated_sequence_sha256 opaque values admitted as H129 only after the declared BOS+128 and signed-int32 convention checks",
            "fit_replay rendered_sha256 admitted separately; unavailable public/sequence aliases remain explicit gaps",
            "P04 public pool Git-object ledgers were recovered and bound by exact file SHA, preserving individual record IDs/rendered hashes/row indices without source payload",
        ]
    )
    producer_git_proof = _p04_producer_git_proof(root)
    recomputed_exchange_digest = _json_digest({
        "reservations": value["reservations"],
        "overlap_counts": value["overlap_counts"],
        "selection_fresh_panel_reconciliation": value["selection_fresh_panel_reconciliation"],
    })
    exchange_digest_match = recomputed_exchange_digest == value["exchange_digest_sha256"]
    if not exchange_digest_match:
        raise ExclusionAuditError("P04 top-level exchange digest mismatch")
    proof = {
        "status": "PASS_PRODUCER_CONVENTION_VERIFIED_PARTIAL_H128",
        "exchange_path": str(path),
        "exchange_sha256": actual_sha,
        "producer_descriptor_sha256": P04_PRODUCER_SHA256,
        "producer_source_bytes_available": producer_git_proof["all_expected_source_bytes_verified"],
        "producer_git_object_proof": producer_git_proof,
        "declared_convention_checks": convention_checks,
        "pool_counts": pool_proof,
        "recovered_ledger_proof": ledger_proofs,
        "fit_replay_rendered_count": len(fit_values),
        "consumer_proof": _consumer_proof(root),
        "individual_record_ids_available": all(item.get("available") is True for item in ledger_proofs.values()),
        "targetfit_individual_hashes_available": False,
        "top_level_exchange_digest_recomputed": True,
        "top_level_exchange_digest_match": exchange_digest_match,
        "top_level_exchange_digest_equation": "SHA256(canonical JSON of reservations, overlap_counts, selection_fresh_panel_reconciliation)",
    }
    return P04Load(bundle=bundle, proof=proof, gaps=gaps)


def _augment_trr0002_finance_h128(bundle: p10.IdentityBundle, path: Path) -> p10.IdentityBundle:
    """Derive canonical H128 only for exact historical Finance inputs.

    TRR-0002's producer stored an active-token digest and a padded input row.
    For rows whose declared active width is at least 128, the surviving
    metadata itself is an exact public token input, so the canonical first-128
    digest can be derived transiently without opening source text or truth.
    Short rows remain explicitly H128-inapplicable.
    """
    value = _read_json(path, label="TRR2 Finance records")
    rows = value.get("records")
    dataset_id = value.get("dataset")
    if not isinstance(rows, list) or not isinstance(dataset_id, str) or not dataset_id:
        raise ExclusionAuditError("TRR2 Finance metadata cannot support H128 derivation")
    namespace = p10.Namespace("finance", dataset_id, "train", "*")
    eligible = 0
    short = 0
    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExclusionAuditError(f"TRR2 Finance row {row_number} is malformed")
        ids = row.get("input_ids")
        valid = row.get("valid_tokens")
        if not isinstance(ids, list) or not isinstance(valid, int) or valid <= 0 or valid > len(ids):
            raise ExclusionAuditError(f"TRR2 Finance row {row_number} lacks exact active input width")
        if valid < 128:
            short += 1
            continue
        if ids[0] != BOS_TOKEN_ID:
            raise ExclusionAuditError(f"TRR2 Finance row {row_number} does not begin with BOS")
        if any(not isinstance(token, int) or isinstance(token, bool) for token in ids[:valid]):
            raise ExclusionAuditError(f"TRR2 Finance row {row_number} contains a non-integer active token")
        digest = p10._raw_int32_digest(ids[:128])
        bundle.add("h128_sequence_sha256", digest, namespace)
        eligible += 1
    bundle.metadata_notes.append(
        f"TRR2 Finance canonical H128 derived transiently from exact active input_ids for {eligible} rows; {short} rows are verified shorter than 128"
    )
    return bundle


def load_bundle(spec: p10.SourceSpec, *, root: Path, pr20_root: Path | None = None) -> p10.IdentityBundle | None:
    """Load a P11 source with strict P04 and bounded historical H128 paths."""
    if spec.label == "trr0006_p04_opaque":
        return load_p04(root).bundle
    bundle = p10.load_bundle(spec, root=root, pr20_root=pr20_root)
    if bundle is not None and spec.label == "trr0002_public_finance_records":
        bundle = _augment_trr0002_finance_h128(bundle, bundle.path)
    return bundle


def _load_specs(root: Path, pr20_root: Path | None) -> tuple[list[p10.IdentityBundle], P04Load]:
    specs = list(p10.source_specs())
    p04 = load_p04(root)
    bundles: list[p10.IdentityBundle] = []
    for spec in specs:
        if spec.label == "trr0006_p04_opaque":
            bundles.append(p04.bundle)
            continue
        bundle = load_bundle(spec, root=root, pr20_root=pr20_root)
        if bundle is not None:
            bundles.append(bundle)
    if len(bundles) != len(specs):
        present = {bundle.label for bundle in bundles}
        raise ExclusionAuditError(f"required P11 identity sources missing: {sorted({spec.label for spec in specs} - present)}")
    return bundles, p04


def _selection_rows(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping) or not isinstance(rule.get("records"), Mapping):
        raise ExclusionAuditError("selection ledger has no selection_rule.records mapping")
    return rule["records"]


def validate_panel_selection(panel: Mapping[str, Any], selection: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Strictly bind an aggregate panel's declared row digest/counts to rows."""
    if panel.get("public_material_only") is not True or panel.get("truth_opened") is not False or panel.get("same_sources_across_targets") is not True:
        raise ExclusionAuditError(f"{label} access flags do not satisfy the public pre-truth contract")
    declared_panel_digest = panel.get("record_ids_sha256")
    declared_selection_digest = selection.get("selection_rule", {}).get("record_ids_sha256")
    if not isinstance(declared_panel_digest, Mapping) or not isinstance(declared_selection_digest, Mapping):
        raise ExclusionAuditError(f"{label} record digest map is absent")
    rows = _selection_rows(selection)
    declared_counts = panel.get("records_by_domain")
    if declared_counts is None:
        scalar = panel.get("records_per_domain")
        declared_counts = {domain: scalar for domain in rows} if isinstance(scalar, int) else None
    if not isinstance(declared_counts, Mapping):
        raise ExclusionAuditError(f"{label} domain counts are absent")
    domain_results: dict[str, Any] = {}
    for domain, domain_rows in rows.items():
        if not isinstance(domain, str) or not isinstance(domain_rows, list):
            raise ExclusionAuditError(f"{label} domain rows are malformed")
        expected_count = declared_counts.get(domain)
        if not isinstance(expected_count, int) or len(domain_rows) != expected_count:
            raise ExclusionAuditError(f"{label} {domain} row count mismatch")
        record_ids: list[str] = []
        required_hashes = ("public_record_sha256", "final_sequence_sha256")
        for row in domain_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str) or not row["record_id"]:
                raise ExclusionAuditError(f"{label} {domain} contains a malformed record_id")
            if any(not _is_sha256(row.get(key)) for key in required_hashes):
                raise ExclusionAuditError(f"{label} {domain} contains a malformed identity hash")
            record_ids.append(row["record_id"])
        if len(record_ids) != len(set(record_ids)):
            raise ExclusionAuditError(f"{label} {domain} contains duplicate record IDs")
        digest = _json_digest(record_ids)
        if digest != declared_selection_digest.get(domain) or digest != declared_panel_digest.get(domain):
            raise ExclusionAuditError(f"{label} {domain} record digest mismatch")
        domain_results[domain] = {
            "records": len(record_ids),
            "computed_record_ids_sha256": digest,
            "panel_digest_match": True,
            "selection_digest_match": True,
        }
    extra_panel_domains = set(declared_panel_digest) - set(rows)
    if extra_panel_domains:
        raise ExclusionAuditError(f"{label} declares unbound panel domains")
    return {
        "label": label,
        "status": "PASS",
        "domains": domain_results,
        "public_material_only": True,
        "same_sources_across_targets": True,
        "truth_opened": False,
    }


def _verify_panel_selection_descriptor(
    panel_path: Path,
    selection_path: Path,
    *,
    root: Path,
    pr20_root: Path | None,
    label: str,
) -> dict[str, Any]:
    panel = _read_json(panel_path, label=f"{label} panel")
    selection = _read_json(selection_path, label=f"{label} selection")
    descriptor = panel.get("selection_plan")
    if not isinstance(descriptor, Mapping):
        raise ExclusionAuditError(f"{label} panel selection_plan descriptor is absent")
    pointed = _descriptor_path(descriptor, root=root, pr20_root=pr20_root, label=f"{label} panel selection_plan")
    if pointed != selection_path.resolve():
        # Historical panel descriptors may carry a sibling worktree absolute
        # path; byte identity is the binding, while the expected path is the
        # explicit inventory path.
        if p10.sha256_file(pointed) != p10.sha256_file(selection_path):
            raise ExclusionAuditError(f"{label} panel selection pointer does not bind the expected ledger")
    return validate_panel_selection(panel, selection, label=label)


def _verify_pr20_binding(root: Path, pr20_root: Path) -> dict[str, Any]:
    panel_path = _resolve(AGGREGATE_BINDINGS[-1]["panel"], root=root, pr20_root=pr20_root)
    panel = _read_json(panel_path, label="PR20 panel")
    panel_descriptor = panel.get("selection_plan")
    binding_path = _descriptor_path(panel_descriptor, root=root, pr20_root=pr20_root, label="PR20 panel→binding")
    binding = _read_json(binding_path, label="PR20 source-selection binding")
    ledger_descriptor = binding.get("selection_ledger")
    ledger_path = _descriptor_path(ledger_descriptor, root=root, pr20_root=pr20_root, label="PR20 binding→selection")
    selection = _read_json(ledger_path, label="PR20 selection ledger")
    binding_counts = binding.get("records_by_domain")
    selection_counts = selection.get("records_by_domain")
    if binding_counts != selection_counts:
        raise ExclusionAuditError("PR20 binding and selection domain counts differ")
    binding_digest = binding.get("record_ids_sha256")
    selection_digest = selection.get("selection_rule", {}).get("record_ids_sha256")
    if binding_digest != selection_digest:
        raise ExclusionAuditError("PR20 binding and selection record digests differ")
    result = validate_panel_selection(panel, selection, label="pr20_source_panel")
    result["panel_to_binding"] = {"status": "PASS", "binding_sha256": p10.sha256_file(binding_path)}
    result["binding_to_selection"] = {"status": "PASS", "selection_sha256": p10.sha256_file(ledger_path)}
    return result


def validate_aggregate_bindings(*, root: Path, pr20_root: Path | None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in AGGREGATE_BINDINGS[:-1]:
        panel_path = _resolve(item["panel"], root=root, pr20_root=pr20_root)
        selection_path = _resolve(item["selection"], root=root, pr20_root=pr20_root)
        if not panel_path.is_file() or not selection_path.is_file():
            raise ExclusionAuditError(f"aggregate binding source unavailable: {item['label']}")
        results.append(_verify_panel_selection_descriptor(panel_path, selection_path, root=root, pr20_root=pr20_root, label=item["label"]))
    if pr20_root is None:
        raise ExclusionAuditError("PR20 root is required for aggregate binding audit")
    results.append(_verify_pr20_binding(root, pr20_root))
    return {
        "status": "PASS_ALL_SIX",
        "count": len(results),
        "bindings": results,
        "pr20_final_sources_are_development_overlap": True,
    }


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ExclusionAuditError("public token payload tensor row is not list-like")
    return value


def _row_namespace(row: Mapping[str, Any]) -> p10.Namespace:
    style = row.get("dataset_key") or row.get("dataset") or "*"
    style_text = p10._style(style) or "*"
    dataset_id = row.get("dataset_id") or "*"
    split = row.get("split") or "train"
    revision = row.get("revision") or "*"
    return p10.Namespace(str(style_text), str(dataset_id), str(split), str(revision))


def rehash_public_token_rows(
    rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Any],
    mask_rows: Sequence[Any],
    *,
    payload_path: Path,
    payload_sha256: str | None = None,
) -> tuple[p10.IdentityBundle, dict[str, Any]]:
    """Derive canonical H128 from a public token payload without retaining IDs."""
    if len(rows) != len(token_rows) or len(rows) != len(mask_rows):
        raise ExclusionAuditError("public payload and metadata row counts differ")
    bundle = p10.IdentityBundle(
        "agent1_b0_public_token_payload",
        "inherited_fitting",
        payload_path,
        payload_sha256 or p10.sha256_file(payload_path),
        payload_path.stat().st_size,
        status="public_token_ids_hashed_transiently",
    )
    short = 0
    eligible = 0
    seen_ids: set[str] = set()
    h128_count = 0
    for index, (row, raw_ids, raw_mask) in enumerate(zip(rows, token_rows, mask_rows)):
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str) or not row["record_id"]:
            raise ExclusionAuditError(f"public payload metadata row {index} lacks record_id")
        record_id = row["record_id"]
        if record_id in seen_ids:
            raise ExclusionAuditError("public payload metadata contains duplicate record_id")
        seen_ids.add(record_id)
        ids = [int(value) for value in _as_list(raw_ids)]
        mask = [int(value) for value in _as_list(raw_mask)]
        if len(ids) != len(mask) or not ids or any(value not in (0, 1) for value in mask):
            raise ExclusionAuditError(f"public payload row {index} mask geometry is malformed")
        active_count = sum(mask)
        if active_count <= 0 or any(mask[pos] == 0 and any(mask[pos + 1 :]) for pos in range(len(mask))):
            raise ExclusionAuditError(f"public payload row {index} mask is not a prefix")
        if ids[0] != BOS_TOKEN_ID:
            raise ExclusionAuditError(f"public payload row {index} BOS identity changed")
        bundle.add("record_id", record_id, _row_namespace(row))
        rendered = row.get("rendered_sha256")
        if _is_sha256(rendered):
            bundle.add("rendered_sha256", rendered.casefold(), _row_namespace(row))
        if active_count < 128:
            short += 1
            continue
        eligible += 1
        digest = p10._raw_int32_digest(ids[:128])
        bundle.add("h128_sequence_sha256", digest, _row_namespace(row))
        h128_count += 1
    return bundle, {
        "status": "PASS_PUBLIC_TOKEN_H128_REHASH",
        "rows_seen": len(rows),
        "h128_rows": h128_count,
        "short_rows_h128_inapplicable": short,
        "eligible_rows": eligible,
        "token_values_emitted": False,
        "source_text_read": False,
        "activations_read": False,
        "model_loaded": False,
        "payload_path": str(payload_path),
        "payload_sha256": bundle.sha256,
        "payload_bytes": bundle.bytes,
        "tensors_read": ["token_ids", "attention_mask"],
        "tensors_not_read": ["activations", "position_ids", "post_bos_selector_large", "post_bos_selector_small"],
        "sequence_convention": "first 128 active BOS-inclusive IDs, SHA-256 little-endian signed-int32 bytes",
    }


def rehash_public_token_payload(*, payload_path: Path, metadata_path: Path) -> tuple[p10.IdentityBundle, dict[str, Any]]:
    """Read only ``token_ids`` and ``attention_mask`` from a safetensors payload."""
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ExclusionAuditError("safetensors is required for the bounded public-token pass") from exc
    metadata = _read_json(metadata_path, label="public token metadata")
    rows = metadata.get("records")
    if not isinstance(rows, list):
        raise ExclusionAuditError("public token metadata lacks records list")
    payload_path = payload_path.resolve()
    if not payload_path.is_file() or payload_path.is_symlink():
        raise ExclusionAuditError(f"public token payload is unavailable: {payload_path}")
    payload_sha = p10.sha256_file(payload_path)
    with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        required = {"token_ids", "attention_mask"}
        if not required.issubset(keys):
            raise ExclusionAuditError("public token payload lacks token_ids/attention_mask")
        token_tensor = handle.get_tensor("token_ids")
        mask_tensor = handle.get_tensor("attention_mask")
        if tuple(token_tensor.shape) != tuple(mask_tensor.shape) or len(token_tensor.shape) != 2:
            raise ExclusionAuditError("public token payload tensor geometry differs")
        token_rows = token_tensor.tolist()
        mask_rows = mask_tensor.tolist()
    bundle, receipt = rehash_public_token_rows(rows, token_rows, mask_rows, payload_path=payload_path, payload_sha256=payload_sha)
    receipt.update(
        {
            "tensor_keys_present": sorted(keys),
            "token_tensor_shape": [int(value) for value in token_tensor.shape],
            "token_tensor_dtype": str(token_tensor.dtype),
            "mask_tensor_dtype": str(mask_tensor.dtype),
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": p10.sha256_file(metadata_path),
        }
    )
    return bundle, receipt


def _record_rows(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield record mappings from a metadata JSON tree without reading payloads."""
    if isinstance(value, Mapping):
        if isinstance(value.get("record_id"), str):
            yield value
        for child in value.values():
            yield from _record_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _record_rows(child)


def _sequence_gap_report(
    bundles: Sequence[p10.IdentityBundle], *, root: Path, p04: P04Load
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for bundle in bundles:
        direct = bundle.counts().get("h128_sequence_sha256", 0)
        raw: Any = None
        try:
            raw = _read_json(bundle.path, label=bundle.label)
        except ExclusionAuditError:
            raw = None
        rows = list(_record_rows(raw)) if raw is not None else []
        full_lengths: list[int] = []
        for row in rows:
            if isinstance(row.get("full_token_count"), int):
                full_lengths.append(int(row["full_token_count"]))
            elif isinstance(row.get("active_token_count"), int):
                full_lengths.append(int(row["active_token_count"]))
            elif isinstance(row.get("post_bos_token_count"), int):
                full_lengths.append(int(row["post_bos_token_count"]) + 1)
        # TRR-0002 Pile was opened under the historical tensor-header H40
        # convention. It is a known 40-token observation, so requiring a
        # later H128 identity would misstate a producer limitation as a
        # recoverable missing hash. Keep H40 in the union and make this
        # convention override explicit in the receipt.
        if bundle.label == "trr0002_public_pile_records":
            short = None
            eligible = 0
            missing = 0
            status = "H40_ONLY_H128_INAPPLICABLE_TO_OPENED_40_TOKEN_OBSERVATION"
        # Finance retains the TRR2 producer's active-token digest. Where its
        # exact active input is 128 tokens, the same producer bytes also prove
        # canonical H128; shorter rows remain H128-inapplicable.
        elif bundle.label == "trr0002_public_finance_records":
            short = sum(isinstance(row.get("valid_tokens"), int) and row["valid_tokens"] < 128 for row in rows)
            eligible = sum(isinstance(row.get("valid_tokens"), int) and row["valid_tokens"] >= 128 for row in rows)
            missing = max(0, eligible - direct)
            status = "PARTIAL_H128_DERIVED_FROM_TRR2_ACTIVE_INPUT_PLUS_VERIFIED_SHORT_ROWS" if missing == 0 else "MISSING_TRR2_FINANCE_H128_FOR_ELIGIBLE_ROWS"
        elif bundle.label == "trr_p09_nested_b1":
            short = sum(row.get("final_sequence_eligibility") == "ineligible_shorter_than_128_active_tokens_no_padding_hash" for row in rows)
            eligible = sum(row.get("final_sequence_eligibility") == "eligible_first_128_active_int32_ids_including_bos" for row in rows)
            missing = max(0, eligible - direct)
            status = "PARTIAL_DIRECT_H128_PLUS_VERIFIED_SHORT_ROWS" if missing == 0 else "PARTIAL_DIRECT_H128_WITH_UNMATCHED_ELIGIBLE_ROWS"
        elif full_lengths:
            eligible = sum(length >= 128 for length in full_lengths)
            short = sum(length < 128 for length in full_lengths)
            missing = max(0, eligible - direct)
            status = "PARTIAL_H128_DIRECT_PLUS_SHORT_CLASSIFICATION" if missing == 0 else "MISSING_RECOVERABLE_H128_FOR_ELIGIBLE_ROWS"
        elif direct:
            short = None
            eligible = None
            missing = None
            status = "DIRECT_H128_WITHOUT_COMPLETE_ROW_LENGTH_CLASSIFICATION"
        elif bundle.label == "trr0006_p04_opaque":
            short = None
            eligible = None
            missing = None
            status = "H129_ONLY_H128_NOT_DERIVABLE_FROM_OPAQUE_VALUES"
        elif rows:
            short = None
            eligible = None
            missing = len(rows)
            status = "MISSING_RECOVERABLE_H128_EXACT_RENDERER_OR_TOKEN_INPUT"
        else:
            short = None
            eligible = None
            missing = None
            status = "NO_INDIVIDUAL_ROWS_OR_H128"
        report.append(
            {
                "label": bundle.label,
                "role": bundle.role,
                "record_rows_seen": len(rows),
                "direct_h128_count": direct,
                "verified_short_rows_h128_inapplicable": short,
                "eligible_rows_by_declared_length": eligible,
                "eligible_rows_without_h128": missing,
                "status": status,
            }
        )
    # P04's explicit unavailable targetfit fields are kept alongside the
    # per-source sequence diagnosis to prevent an aggregate count from being
    # misread as complete individual coverage.
    report.append(
        {
            "label": "trr0006_p04_targetfit",
            "role": "opaque_reservation",
            "status": "INDIVIDUAL_IDENTITIES_UNAVAILABLE",
            "public_record_sha256": "unavailable_counts_only",
            "h129_sequence_sha256": "unavailable_counts_only",
        }
    )
    return report


def _synthetic_probe() -> dict[str, Any]:
    values = list(range(192))
    fingerprints = p10.candidate_sequence_fingerprints(values)
    ns = p10.Namespace("pile", "fixture", "train", "fixture")
    bundle = p10.IdentityBundle("fixture", "regression", Path("."), "", 0)
    bundle.add("record_id", "reused-validation-record", ns)
    bundle.add("rendered_sha256", "a" * 64, ns)
    bundle.add("h128_sequence_sha256", fingerprints["h128_sequence_sha256"], ns)
    bundle.add("h129_sequence_sha256", fingerprints["h129_sequence_sha256"], ns)
    reasons = p10.check_candidate({**fingerprints, "record_id": "reused-validation-record", "style": "pile", "dataset_id": "fixture", "split": "train", "revision": "fixture"}, bundle)
    return {
        "long_candidate_prefixes_distinct": fingerprints["h128_sequence_sha256"] != fingerprints["h129_sequence_sha256"],
        "helper_fields_consumed_by_checker": {reason["field"] for reason in reasons} == {"record_id", "h128_sequence_sha256", "h129_sequence_sha256"},
        "h128_label_does_not_match_h129": p10.check_candidate({"h128_sequence_sha256": fingerprints["h129_sequence_sha256"], "style": "pile", "dataset_id": "fixture"}, bundle) == [],
    }


def build_audit(
    *,
    root: Path,
    pr20_root: Path | None = None,
    public_payload: Path | None = None,
    public_payload_metadata: Path | None = None,
) -> dict[str, Any]:
    bundles, p04 = _load_specs(root, pr20_root)
    rehash_bundle: p10.IdentityBundle | None = None
    rehash_receipt: dict[str, Any]
    if public_payload is not None or public_payload_metadata is not None:
        if public_payload is None or public_payload_metadata is None:
            raise ExclusionAuditError("public payload and metadata must be supplied together")
        rehash_bundle, rehash_receipt = rehash_public_token_payload(payload_path=public_payload, metadata_path=public_payload_metadata)
        bundles_for_union = [*bundles, rehash_bundle]
    else:
        bundles_for_union = bundles
        rehash_receipt = {
            "status": "PENDING_ROOT_LEASE",
            "rows_seen": 0,
            "h128_rows": 0,
            "short_rows_h128_inapplicable": 0,
            "payload_not_opened": True,
            "planned_payload": "TRR-0007/support/broader_capture_v2/enriched_fit_cut4.safetensors",
            "planned_metadata": "TRR-0007/support/broader_capture_v2/enriched_fit_records.json",
            "planned_payload_sha256": "a55814759dfa9d2567587935063fc49e44d8bff949c50014793deb982ebdf35d",
            "planned_payload_bytes": 947176760,
            "resource_plan": {
                "rows": 1200,
                "tensors_read": ["token_ids", "attention_mask"],
                "tensor_bytes_estimate": 1200 * 192 * 4 + 1200 * 192,
                "tensors_not_read": ["activations", "position_ids", "post_bos_selector_large", "post_bos_selector_small"],
                "model_loaded": False,
                "expected_peak_tensor_bytes": 1200 * 192 * 4 + 1200 * 192,
                "safety_margin": "payload is 947176760 bytes on disk; safetensors selective reads avoid activation/model allocation",
            },
        }
    union = p10.merge_bundles(bundles_for_union)
    by_label = {bundle.label: bundle for bundle in bundles}
    aggregate = validate_aggregate_bindings(root=root, pr20_root=pr20_root)
    sequence_report = _sequence_gap_report(bundles, root=root, p04=p04)
    unknown = [
        {
            "label": bundle.label,
            "role": bundle.role,
            "keys": dict(sorted(bundle.unknown_identity_keys.items())),
            "reason": "unrecognized identity/hash fields are reported but excluded without producer proof",
        }
        for bundle in bundles
        if bundle.unknown_identity_keys
    ]
    probe = _synthetic_probe()
    identity_gaps = [
        {
            "label": bundle.label,
            "role": bundle.role,
            "reason": "loaded metadata contains no individual comparable identity fields",
        }
        for bundle in bundles
        if not bundle.values
    ]
    descriptor_only_labels = {
        "trr0002_calibration",
        "trr0005_public_validation_selection",
        "trr0007_selection_exclusions",
        "trr0008_selection_exclusions",
        "trr0009_selection_v2_exclusions",
        "trr_p09_public_validation_audit",
        "pr20_source_selection_binding",
    }
    source_inventory = []
    for bundle in bundles_for_union:
        source_inventory.append(
            {
                "label": bundle.label,
                "role": bundle.role,
                "path": str(bundle.path),
                "bytes": bundle.bytes,
                "sha256": bundle.sha256,
                "schema": bundle.schema,
                "status": bundle.status,
                "identity_counts": bundle.counts(),
                "namespace_counts": bundle.namespace_counts(),
                "unrecognized_identity_key_counts": dict(sorted(bundle.unknown_identity_keys.items())),
                "metadata_notes": bundle.metadata_notes,
            }
        )
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PARTIAL_CANONICAL_SEQUENCE_EXCLUSION_AUDIT",
        "coverage_complete": False,
        "selection_release": False,
        "source_count": len(bundles),
        "source_inventory": source_inventory,
        "required_source_classes": ["gradient_fitting", "inherited_fitting", "checkpoint_selection", "calibration", "previously_opened_development", "previously_opened_evaluation"],
        "union_identity_counts": union.counts(),
        "union_namespace_counts": union.namespace_counts(),
        "required_sources": [spec.label for spec in p10.source_specs()],
        "source_class_coverage": {
            "gradient_fitting": {
                "status": "INCLUDED_METADATA_LEDGER_UNION",
                "sources": ["trr0003_fit_records", "trr0004_affine_fit", "trr0004_adapter_v2_fit", "trr0005_original_fit", "trr0005_enriched_fit", "trr0007_original_fit", "trr0007_enriched_fit", "trr_p09_nested_b1"],
            },
            "inherited_fitting": {
                "status": "INCLUDED_WHEN_EXPLICIT_SOURCE_OR_PUBLIC_PAYLOAD_IS_BOUND",
                "sources": ["trr0001_manifest", "trr0001_plan", "trr0001_r1_selection_reveal", "trr0002_public_finance_records", "trr0002_public_pile_records", "trr_p09_nested_b1"],
            },
            "checkpoint_selection": {
                "status": "INCLUDED_METADATA_LEDGER_UNION",
                "sources": ["trr0002_configuration_winner", "trr0004_selection_plan", "trr0009_selection_v2", "pr20_source_selection_binding"],
            },
            "calibration": {
                "status": "INCLUDED_METADATA_LEDGER_UNION",
                "sources": ["trr0002_calibration", "trr0004_affine_validation", "trr0004_adapter_v2_validation", "trr0005_public_validation_selection", "trr_p09_public_validation_audit"],
            },
            "previously_opened_development": {
                "status": "INCLUDED_METADATA_LEDGER_UNION",
                "sources": ["trr0001_manifest", "trr0001_plan", "trr0001_r1_selection_reveal", "trr0002_fresh_observation_index", "trr0002_public_finance_records", "trr0002_public_pile_records", "trr0007_selection_exclusions", "trr0008_selection_exclusions", "trr0009_selection_v2_exclusions"],
            },
            "previously_opened_evaluation": {
                "status": "INCLUDED_AGGREGATE_BINDINGS_AND_SELECTION_LEDGER",
                "sources": ["trr0004_selection_plan", "trr0006_selection", "trr0006_panel", "trr0006_public_observation_panel", "trr0007_selection", "trr0007_eval_panel", "trr0008_selection", "trr0008_eval_panel", "trr0009_original_selection", "trr0009_selection_v2", "trr0009_eval_panel", "pr20_source_selection_binding", "pr20_source_selection", "pr20_source_panel", "trr0003_opened_panel_metadata", "trr0004_opened_panel_metadata", "trr0005_opened_panel_metadata"],
            },
        },
        "agent1_replication_assets": {
            "status": "PENDING_AGENT1_HANDOFF",
            "included": False,
            "note": "New B0/B1 identities and selection metadata are not invented or used before handoff.",
        },
        "trr0009_selection_manifest_required_and_loaded": "trr0009_selection_v2" in by_label,
        "aggregate_panel_binding": aggregate,
        "p04_convention_proof": p04.proof,
        "canonical_sequence_audit": {
            "status": "PARTIAL_EXACT_PREFIX_AUDIT",
            "hash_convention": "SHA-256 of first 128 active BOS-inclusive IDs encoded as little-endian signed int32 bytes",
            "h128_and_h129_are_distinct_namespaces": True,
            "per_source": sequence_report,
            "public_payload_rehash": rehash_receipt,
        },
        "coverage": {
            "coverage_complete": False,
            "accessible_explicit_sources_loaded": True,
            "accessible_explicit_source_count": len(bundles),
            "explicit_source_spec_count": len(p10.source_specs()),
            "descriptor_only_sources": sorted(descriptor_only_labels),
            "identity_gaps": identity_gaps,
            "unrecognized_identity_key_gaps": unknown,
            "aggregate_panel_binding_status": aggregate["status"],
            "p04_targetfit_individual_hashes_available": False,
            "public_payload_h128_rehash_status": rehash_receipt["status"],
        },
        "coverage_gaps": [
            "P03 sealed holdout is intentionally unopened and absent from this inventory.",
            "P04 targetfit public_record_sha256 and truncated_sequence_sha256 arrays are unavailable; aggregate counts are not treated as zero overlap.",
            "P04 producer/helper bytes and the three public pool ledgers are verified from retained Git objects; the exchange convention and top-level digest are recomputed, while P04 H128 remains absent from the exchange and must be derived from exact public token inputs. P04 targetfit remains counts-only.",
            "Inherited fitting rows with no direct H128 are classified per-source below; verified short rows are H128-inapplicable, while eligible rows without exact token inputs remain unresolved.",
            "TRR-0002 Pile remains H40-only because every opened row is exactly 40 tokens; TRR-0002 Finance H128 is derived only for exact active rows of length at least 128, with shorter rows marked inapplicable. Historical active/H40 fields remain candidate rejection keys.",
            "No new candidate rows were scanned and no selection/capture/prediction/truth operation was performed.",
            "PR20 final sources are explicitly development/selection-overlapping and cannot be reused as an independent final panel.",
        ] + p04.gaps,
        "access_boundary": {
            "source_text_read": False,
            "source_or_token_payload_emitted": False,
            "historical_panel_json_parsed": True,
            "historical_panel_identity_fields_inspected": True,
            "historical_panel_payload_fields_skipped": True,
            "activation_payload_read": False,
            "model_loaded": False,
            "truth_or_scores_read": False,
            "new_panel_selected": False,
            "p03_holdout_accessed": False,
            "public_token_ids_hashed_only_when_root_lease_supplied": True,
        },
        "hash_conventions": {
            "record_id": "global exact string identity",
            "rendered_sha256": "global only when producer field is explicitly bound to rendered/public record bytes",
            "h128_sequence_sha256": "global SHA-256 of exactly first 128 active BOS-inclusive IDs as little-endian signed-int32 bytes",
            "h129_sequence_sha256": "separate SHA-256 of first 129 active BOS-inclusive IDs; never compared with H128",
            "source_index": "dataset/style/split/revision namespace scoped",
            "trr0002_active_token_ids_sha256": "TRR2 Finance active IDs, little-endian signed int32 bytes; exact active rows of length at least 128 additionally prove canonical H128",
            "trr0002_h40_token_ids_sha256": "TRR2 Pile H40 tensor-header plus little-endian int64 bytes; H128 is inapplicable to its exact 40-token opened rows",
        },
        "regression": probe,
        "notes": [
            "This is a prior-identity union and canonical-prefix audit, not source selection or capacity certification.",
            "Descriptor-only pointer/count receipts are not interpreted as zero overlap.",
            "The P10 receipts and code remain unchanged; this fresh P11 receipt supersedes no prior artifact.",
        ],
        "superseded_artifacts": [],
    }


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--pr20-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-token-payload", type=Path)
    parser.add_argument("--public-token-metadata", type=Path)
    args = parser.parse_args(argv)
    result = build_audit(
        root=args.root.resolve(),
        pr20_root=args.pr20_root.resolve(),
        public_payload=args.public_token_payload.resolve() if args.public_token_payload else None,
        public_payload_metadata=args.public_token_metadata.resolve() if args.public_token_metadata else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("xb") as handle:
            handle.write(canonical_json(result))
    except FileExistsError as exc:
        raise ExclusionAuditError(f"refusing to overwrite existing P11 audit receipt: {args.output}") from exc
    print(json.dumps({"output": str(args.output), "status": result["status"], "source_count": result["source_count"], "aggregate": result["aggregate_panel_binding"]["status"], "rehash": result["canonical_sequence_audit"]["public_payload_rehash"]["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
