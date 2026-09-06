#!/usr/bin/env python3
"""Build a count-only eligibility inventory for the TRR-0006 public panel.

The inventory reads only the public Pile and Finance Arrow caches and the
identity-only metadata files listed in the output.  It reuses the TRR-0005
renderer, partition guard, and identity scanner, but it never writes source
rows, token IDs, truth, a selection plan, or observations.  A later selection
step must consume this aggregate evidence only after the sample size and
complete plan have been frozen.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

from scripts import trr0005_produce_confirmation as trusted
from token_reconstruction.trr0005_contract import STYLE_ORDER
from token_reconstruction.trr0005_public_corpus import (
    SOURCE_PARTITIONS,
    deterministic_row_order,
    source_record_id,
)


TASK_ID = "TRR-0006"
INVENTORY_SCHEMA = "token-reconstruction.trr0006-eligibility-inventory.v1"
SELECTION_SEED = 5005
SEQUENCE_TOKENS = 128
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
P04_EXCHANGE_PATH = Path(
    "/tmp/trr-p04/experiments/TRR-P04/coordination/reservation_hashes.json"
)
P04_EXCHANGE_SHA256 = "98f8dfcab0977b4bcafa47d97a86a410ab37359b897b9b553746afa7df5c7904"

# This is the aggregate exchange received from P04.  The individual values
# are intentionally not consumed by this producer.  In particular, target-fit
# row identities/ranges and replay sequence hashes are unavailable to TRR-0006
# and cannot be treated as an empty reservation.
P04_EXCHANGE_COUNTS = {
    "fit_replay_excluded_record_count": 1200,
    "selection_identity_content_hash_count": 1312,
    "selection_identity_record_id_count": 1280,
    "selection_identity_source_index_count": 56,
    "selection_panel_anchor_record_count": 12,
    "selection_panel_independent_source_record_count": 72,
    "selection_panel_record_count": 72,
    "targetfit_selected_update_record_count": 256,
    "targetfit_sequence_fingerprint_count": 256,
    "targetfit_text_fingerprint_count": 256,
}


class EligibilityError(RuntimeError):
    """Raised when the count-only inventory cannot be built safely."""


@dataclass
class DomainCounts:
    style: str
    range_start: int
    range_stop: int
    pool_size: int
    scanned_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    excluded_id: int = 0
    excluded_index: int = 0
    excluded_hash: int = 0
    duplicate_rendered_source: int = 0
    duplicate_final_sequence: int = 0
    excluded_opaque_source_hash: int = 0
    excluded_opaque_sequence_hash: int = 0
    p04_sequence_hash_unavailable: int = 0
    eligible_unique: int = 0
    valid_identity_commitments: list[dict[str, str]] | None = None
    eligible_identity_commitments: list[dict[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.valid_identity_commitments is None:
            self.valid_identity_commitments = []
        if self.eligible_identity_commitments is None:
            self.eligible_identity_commitments = []

    def as_dict(self, *, requested_per_domain: int | None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "style": self.style,
            "source_range_half_open": [self.range_start, self.range_stop],
            "pool_size": self.pool_size,
            "scanned_rows": self.scanned_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "excluded_id": self.excluded_id,
            "excluded_index": self.excluded_index,
            "excluded_hash": self.excluded_hash,
            "duplicate_rendered_source": self.duplicate_rendered_source,
            "duplicate_final_sequence": self.duplicate_final_sequence,
            "excluded_opaque_source_hash": self.excluded_opaque_source_hash,
            "excluded_opaque_sequence_hash": self.excluded_opaque_sequence_hash,
            "p04_sequence_hash_unavailable": self.p04_sequence_hash_unavailable,
            "eligible_unique": self.eligible_unique,
            "valid_identity_commitment_sha256": _commitment_digest(
                self.valid_identity_commitments or []
            ),
            "eligible_identity_commitment_sha256": _commitment_digest(
                self.eligible_identity_commitments or []
            ),
        }
        if requested_per_domain is not None:
            result["capacity_for_requested_per_domain"] = {
                "requested": requested_per_domain,
                "sufficient": self.eligible_unique >= requested_per_domain,
                "surplus_or_shortfall": self.eligible_unique - requested_per_domain,
            }
        return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _commitment_digest(commitments: Sequence[Mapping[str, str]]) -> str:
    canonical = "\n".join(
        _canonical_json(value)
        for value in sorted(
            (dict(value) for value in commitments),
            key=lambda value: _canonical_json(value),
        )
    )
    return _sha256_bytes((canonical + "\n").encode("utf-8"))


def _identity_commitment(record: Any) -> dict[str, str]:
    """Return a digest-only public identity commitment for a transient row."""

    return {
        "record_id_sha256": _sha256_bytes(str(record.record_id).encode("utf-8")),
        "public_record_sha256": str(record.public_record_sha256),
        "final_sequence_sha256": str(record.final_sequence_sha256),
    }


def _known_exclusion_paths(root: Path) -> list[Path]:
    """Return accessible identity metadata, adding TRR-0005 opened records."""

    paths = list(trusted._default_exclusion_paths(root))
    paths.extend(
        [
            root / "experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json",
            root / "experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json",
            root / "experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations.json",
            root / "experiments/TRR-0005/public_activation_v1/capture_manifest_receipt.json",
        ]
    )
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _describe_p04_exchange(path: Path) -> dict[str, Any]:
    """Verify the approved exchange without parsing its opaque row values."""

    path = path.expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "expected_sha256": P04_EXCHANGE_SHA256,
        "metadata_consumed_for_row_exclusions": False,
        "targetfit_per_record_ids_available": False,
        "targetfit_source_ranges_available": False,
        "targetfit_replay_sequence_hashes_available": False,
        "aggregate_counts": dict(P04_EXCHANGE_COUNTS),
    }
    if path.is_symlink() or not path.is_file():
        result.update({"available": False, "verified": False})
        return result
    actual = _sha256_file(path)
    result.update(
        {
            "available": True,
            "bytes": int(path.stat().st_size),
            "sha256": actual,
            "verified": actual == P04_EXCHANGE_SHA256,
        }
    )
    if actual != P04_EXCHANGE_SHA256:
        raise EligibilityError(
            f"approved P04 exchange hash changed: expected {P04_EXCHANGE_SHA256}, got {actual}"
        )
    return result


@dataclass(frozen=True)
class OpaqueExclusions:
    """Opaque P04 identities usable for conservative public-source exclusion."""

    source_hashes: frozenset[str]
    sequence_hashes_129: frozenset[str]
    fields: dict[str, dict[str, Any]]
    exchange: dict[str, Any]


def _opaque_field_values(
    info: Any,
    *,
    label: str,
) -> tuple[set[str], dict[str, Any]]:
    """Validate an exchange field while returning values only to the scanner."""

    if not isinstance(info, Mapping):
        return set(), {"available": False, "reason": "field_missing"}
    available = info.get("available") is True
    summary: dict[str, Any] = {
        "available": available,
        "source_field": info.get("source_field"),
        "ordered_count": info.get("ordered_count"),
        "distinct_count": info.get("distinct_count"),
    }
    for key in (
        "ordered_canonical_json_sha256",
        "ordered_newline_sha256",
        "unique_set_canonical_json_sha256",
    ):
        if key in info:
            summary[key] = info[key]
    if not available:
        summary["reason"] = info.get("reason", "unavailable")
        return set(), summary
    values = info.get("unique_values")
    ordered = info.get("ordered_values")
    if not isinstance(values, list) or not isinstance(ordered, list):
        raise EligibilityError(f"approved P04 {label} values are unavailable")
    if not all(isinstance(value, str) and len(value) == 64 for value in values + ordered):
        raise EligibilityError(f"approved P04 {label} contains a malformed digest")
    if len(set(values)) != int(info.get("distinct_count", -1)):
        raise EligibilityError(f"approved P04 {label} distinct count changed")
    if len(ordered) != int(info.get("ordered_count", -1)):
        raise EligibilityError(f"approved P04 {label} ordered count changed")
    checks = {
        "ordered_canonical_json_sha256": _sha256_bytes(_canonical_json(ordered).encode("utf-8")),
        "ordered_newline_sha256": _sha256_bytes(("\n".join(ordered) + "\n").encode("utf-8")),
        "unique_set_canonical_json_sha256": _sha256_bytes(
            _canonical_json(sorted(set(values))).encode("utf-8")
        ),
    }
    for key, actual in checks.items():
        declared = info.get(key)
        if declared is not None and declared != actual:
            raise EligibilityError(f"approved P04 {label} {key} changed")
    summary["applied_distinct_count"] = len(set(values))
    return set(values), summary


def _load_p04_opaque_exclusions(path: Path) -> OpaqueExclusions:
    """Load only approved opaque hash arrays; never open an underlying ledger."""

    path = path.expanduser().resolve()
    exchange = _describe_p04_exchange(path)
    if not exchange.get("available") or not exchange.get("verified"):
        return OpaqueExclusions(frozenset(), frozenset(), {}, exchange)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EligibilityError("approved P04 exchange is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise EligibilityError("approved P04 exchange must be a JSON object")
    reservations = payload.get("reservations")
    if not isinstance(reservations, Mapping):
        raise EligibilityError("approved P04 exchange has no reservations")
    source_hashes: set[str] = set()
    sequence_hashes: set[str] = set()
    fields: dict[str, dict[str, Any]] = {}
    for reservation in ("correction", "validation", "fresh_evaluation"):
        section = reservations.get(reservation)
        if not isinstance(section, Mapping):
            raise EligibilityError(f"approved P04 exchange lacks {reservation} reservation")
        individual = section.get("hashes", {}).get("individual_opaque_hashes")
        if not isinstance(individual, Mapping):
            raise EligibilityError(f"approved P04 exchange lacks {reservation} opaque hashes")
        for field, destination in (
            ("public_record_sha256", source_hashes),
            ("truncated_sequence_sha256", sequence_hashes),
        ):
            values, summary = _opaque_field_values(
                individual.get(field), label=f"{reservation}.{field}"
            )
            fields[f"{reservation}.{field}"] = summary
            destination.update(values)
    replay = reservations.get("fit_replay")
    if not isinstance(replay, Mapping):
        raise EligibilityError("approved P04 exchange lacks fit_replay reservation")
    replay_individual = replay.get("individual_opaque_hashes")
    if not isinstance(replay_individual, Mapping):
        raise EligibilityError("approved P04 exchange lacks fit_replay opaque hashes")
    values, summary = _opaque_field_values(
        replay_individual.get("rendered_sha256"), label="fit_replay.rendered_sha256"
    )
    fields["fit_replay.rendered_sha256"] = summary
    source_hashes.update(values)
    for field in ("public_record_sha256", "truncated_sequence_sha256"):
        _values, summary = _opaque_field_values(
            replay_individual.get(field), label=f"fit_replay.{field}"
        )
        fields[f"fit_replay.{field}"] = summary
    targetfit = reservations.get("targetfit")
    if isinstance(targetfit, Mapping):
        target_individual = targetfit.get("individual_opaque_hashes")
        if isinstance(target_individual, Mapping):
            for field in ("public_record_sha256", "truncated_sequence_sha256"):
                _values, summary = _opaque_field_values(
                    target_individual.get(field), label=f"targetfit.{field}"
                )
                fields[f"targetfit.{field}"] = summary
    exchange["metadata_consumed_for_row_exclusions"] = True
    exchange["applied_source_hash_count"] = len(source_hashes)
    exchange["applied_sequence_hash_129_count"] = len(sequence_hashes)
    exchange["targetfit_per_record_ids_available"] = False
    exchange["targetfit_source_ranges_available"] = False
    exchange["targetfit_replay_sequence_hashes_available"] = False
    return OpaqueExclusions(frozenset(source_hashes), frozenset(sequence_hashes), fields, exchange)


def _classify_valid_candidate(
    candidate: Any,
    *,
    style: str,
    exclusions: Any,
    opaque: OpaqueExclusions,
    seen_public_hashes: set[str],
    seen_final_sequences: set[str],
) -> str:
    """Classify one rendered candidate and update duplicate state if eligible."""

    blocked = trusted._blocked(candidate, exclusions)
    if blocked == "public_source_id":
        return "excluded_id"
    if blocked == "public_source_index":
        return "excluded_index"
    if blocked in {"public_rendered_hash", "public_final_sequence_hash"}:
        return "excluded_hash"
    if candidate.public_record_sha256 in opaque.source_hashes:
        return "excluded_opaque_source_hash"
    if len(candidate.token_ids) >= SEQUENCE_TOKENS + 1:
        p04_sequence_hash = trusted._sequence_digest(candidate.token_ids[: SEQUENCE_TOKENS + 1])
        if p04_sequence_hash in opaque.sequence_hashes_129:
            return "excluded_opaque_sequence_hash"
    if candidate.public_record_sha256 in seen_public_hashes:
        return "duplicate_rendered_source"
    if candidate.final_sequence_sha256 in seen_final_sequences:
        return "duplicate_final_sequence"
    seen_public_hashes.add(candidate.public_record_sha256)
    seen_final_sequences.add(candidate.final_sequence_sha256)
    return "eligible"


def _scan_domain(
    dataset: Any,
    *,
    style: str,
    tokenizer: Any,
    exclusions: Any,
    opaque: OpaqueExclusions,
    seen_public_hashes: set[str],
    seen_final_sequences: set[str],
) -> DomainCounts:
    spec = SOURCE_PARTITIONS[style]
    start = int(spec["holdout_reserve_start"])
    stop = int(spec["holdout_reserve_stop"])
    if len(dataset) < stop:
        raise EligibilityError(f"{style} cache has {len(dataset)} rows; need {stop}")
    stats = DomainCounts(
        style=style,
        range_start=start,
        range_stop=stop,
        pool_size=stop - start,
    )
    order = deterministic_row_order(
        range(start, stop),
        dataset_key=f"{style}-future-holdout",
        seed=SELECTION_SEED,
    )
    for index in order:
        stats.scanned_rows += 1
        expected_id = source_record_id(
            str(spec["dataset_id"]),
            str(spec["split"]),
            str(spec["revision"]),
            index,
        )
        if expected_id in exclusions.ids[style]:
            stats.excluded_id += 1
            continue
        if index in exclusions.indices[style]:
            stats.excluded_index += 1
            continue
        row = trusted._read_reserved_row(dataset, style=style, row_index=index)
        try:
            candidate = trusted._render_row(style, row, index, tokenizer)
        except trusted.ProducerError:
            stats.invalid_rows += 1
            continue
        stats.valid_rows += 1
        stats.valid_identity_commitments.append(_identity_commitment(candidate))
        if len(candidate.token_ids) < SEQUENCE_TOKENS + 1:
            stats.p04_sequence_hash_unavailable += 1
        reason = _classify_valid_candidate(
            candidate,
            style=style,
            exclusions=exclusions,
            opaque=opaque,
            seen_public_hashes=seen_public_hashes,
            seen_final_sequences=seen_final_sequences,
        )
        if reason == "excluded_id":
            stats.excluded_id += 1
        elif reason == "excluded_index":
            stats.excluded_index += 1
        elif reason == "excluded_hash":
            stats.excluded_hash += 1
        elif reason == "excluded_opaque_source_hash":
            stats.excluded_opaque_source_hash += 1
        elif reason == "excluded_opaque_sequence_hash":
            stats.excluded_opaque_sequence_hash += 1
        elif reason == "duplicate_rendered_source":
            stats.duplicate_rendered_source += 1
        elif reason == "duplicate_final_sequence":
            stats.duplicate_final_sequence += 1
        elif reason == "eligible":
            stats.eligible_unique += 1
            stats.eligible_identity_commitments.append(_identity_commitment(candidate))
        else:  # pragma: no cover - defensive fail-closed branch
            raise EligibilityError(f"unknown candidate classification: {reason}")
    return stats


def _public_file_descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise EligibilityError(f"public input file is unavailable: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    started_utc = _utc_now()
    if output.exists() or output.is_symlink():
        raise EligibilityError(f"inventory output is create-only and already exists: {output}")
    if args.requested_per_domain is not None and args.requested_per_domain <= 0:
        raise EligibilityError("requested per-domain capacity must be positive")

    tokenizer_path = args.tokenizer.expanduser().resolve()
    tokenizer = trusted._load_tokenizer(tokenizer_path)
    pile_paths = tuple(path.expanduser().resolve() for path in args.pile_arrow)
    finance_paths = tuple(path.expanduser().resolve() for path in args.finance_arrow)
    datasets = {
        "pile": trusted._load_arrow_dataset(pile_paths),
        "finance": trusted._load_arrow_dataset(finance_paths),
    }
    exclusions_paths = _known_exclusion_paths(root)
    exclusions_paths.extend(path.expanduser().resolve() for path in args.exclude_source)
    exclusions = trusted._collect_exclusions(exclusions_paths)
    opaque = _load_p04_opaque_exclusions(args.p04_exchange)
    seen_public_hashes: set[str] = set()
    seen_final_sequences: set[str] = set()
    domains: dict[str, DomainCounts] = {}
    for style in STYLE_ORDER:
        domains[style] = _scan_domain(
            datasets[style],
            style=style,
            tokenizer=tokenizer,
            exclusions=exclusions,
            opaque=opaque,
            seen_public_hashes=seen_public_hashes,
            seen_final_sequences=seen_final_sequences,
        )

    p04_exchange = _describe_p04_exchange(args.p04_exchange)
    exclusion_sources = []
    for source in exclusions.sources:
        exclusion_sources.append(
            {
                key: source[key]
                for key in ("path", "available", "bytes", "sha256", "new_identity_count")
                if key in source
            }
        )
    ended_utc = _utc_now()
    result: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "task_id": TASK_ID,
        "status": "ELIGIBILITY_INVENTORY_COMPLETE_NO_SELECTION_NO_TRUTH",
        "selection_status": "NOT_STARTED",
        "sample_size_status": "NOT_FROZEN_PROJECTION_ONLY",
        "requested_per_domain": args.requested_per_domain,
        "source_contract": {
            "selection_seed": SELECTION_SEED,
            "natural_distribution_preserved": True,
            "source_ranges_half_open": {
                style: [
                    int(SOURCE_PARTITIONS[style]["holdout_reserve_start"]),
                    int(SOURCE_PARTITIONS[style]["holdout_reserve_stop"]),
                ]
                for style in STYLE_ORDER
            },
            "sequence_tokens_including_bos": SEQUENCE_TOKENS,
            "scoring_post_bos_tokens": SEQUENCE_TOKENS - 1,
            "capture_batch_records": CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "pairing": "Future selected rows will be paired across public_base and public_lora_2601; this inventory emits no rows.",
        },
        "public_inputs": {
            "pile": [_public_file_descriptor(path) for path in pile_paths],
            "finance": [_public_file_descriptor(path) for path in finance_paths],
            "tokenizer": trusted._tokenizer_descriptor(tokenizer_path),
        },
        "exclusion_policy": {
            "identity_only": True,
            "sources": exclusion_sources,
            "identity_counts": {
                style: {
                    "ids": len(exclusions.ids[style]),
                    "hashes": len(exclusions.hashes[style]),
                    "indices": len(exclusions.indices[style]),
                }
                for style in STYLE_ORDER
            },
            "duplicate_source_and_sequence_deduplication": True,
            "private_or_truth_payload_read": False,
        },
        "opaque_reservations": {
            "p04_exchange": opaque.exchange,
            "applied_field_summaries": opaque.fields,
            "applied_source_hash_count": len(opaque.source_hashes),
            "applied_sequence_hash_129_count": len(opaque.sequence_hashes_129),
            "known_p04_rows_excluded_by_identity": {
                style: domains[style].excluded_opaque_source_hash
                + domains[style].excluded_opaque_sequence_hash
                for style in STYLE_ORDER
            },
            "limitation": "P04 target-fit per-record IDs/ranges and replay sequence hashes were unavailable; no disjointness is inferred for those rows. Available correction, validation, fresh-evaluation, and replay opaque hashes were applied without emitting their values.",
        },
        "domains": {
            style: domains[style].as_dict(requested_per_domain=args.requested_per_domain)
            for style in STYLE_ORDER
        },
        "totals": {
            "unique_source_records_scanned": sum(
                domains[style].scanned_rows for style in STYLE_ORDER
            ),
            "eligible_unique_across_domains": sum(
                domains[style].eligible_unique for style in STYLE_ORDER
            ),
            "cross_domain_duplicate_rendered_or_sequence_state": {
                "rendered_hashes_seen": len(seen_public_hashes),
                "final_sequences_seen": len(seen_final_sequences),
            },
        },
        "execution": {
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "command": list(sys.argv),
            "code_commit": _git_commit(root),
            "producer_source": _public_file_descriptor(Path(__file__)),
            "trusted_trr0005_producer_source": _public_file_descriptor(Path(trusted.__file__)),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": {
                package: _package_version(package)
                for package in ("torch", "datasets", "transformers", "safetensors")
                if _package_version(package) is not None
            },
            "model_loaded": False,
            "target_loaded": False,
            "truth_created_or_opened": False,
            "selection_performed": False,
            "network_used": False,
            "opaque_hashes_applied": True,
        },
        "limitations": [
            "P04 target-fit disjointness cannot be certified from the approved aggregate exchange because per-record IDs, source ranges, and replay sequence hashes are unavailable.",
            "Counts are a public-data inventory only; no source row is selected, frozen, or truth-bound.",
            "The requested sample size remains a planning input and must be frozen once by the owner before selection.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {
        "task_id": TASK_ID,
        "status": result["status"],
        "inventory": str(output),
        "inventory_sha256": _sha256_file(output),
        "selection_performed": False,
        "truth_created_or_opened": False,
        "domains": {
            style: result["domains"][style]["eligible_unique"] for style in STYLE_ORDER
        },
    }


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        import subprocess

        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inventory = parser.add_subparsers(dest="command", required=True).add_parser(
        "inventory", help="scan public source ranges and emit aggregate eligibility counts"
    )
    inventory.add_argument("--repository-root", type=Path, default=Path("."))
    inventory.add_argument("--tokenizer", type=Path, required=True)
    inventory.add_argument("--pile-arrow", type=Path, nargs="+", required=True)
    inventory.add_argument("--finance-arrow", type=Path, nargs="+", required=True)
    inventory.add_argument("--exclude-source", type=Path, nargs="*", default=[])
    inventory.add_argument("--p04-exchange", type=Path, default=P04_EXCHANGE_PATH)
    inventory.add_argument("--requested-per-domain", type=int)
    inventory.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inventory":
        result = build_inventory(args)
    else:  # pragma: no cover
        raise EligibilityError(f"unknown command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EligibilityError, trusted.ProducerError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0006 eligibility error: {exc}")
