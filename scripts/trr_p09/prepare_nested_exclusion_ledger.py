#!/usr/bin/env python3
"""Prepare the identity-only exclusion ledger for the frozen P09 B1 bank.

The input records and token tensor are already-published public fitting
artifacts.  This helper reads their metadata and public token IDs on CPU, then
writes only record identities, source indices, rendered/public hashes, and
first-128 hashes for rows whose actual fitting input has at least 128 active
tokens.  It exercises the real TRR-0005 exclusion scanner before succeeding.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

from scripts.trr0005_produce_confirmation import _collect_exclusions, _sequence_digest


SCHEMA = "token-reconstruction.trr-p09-nested-b1-exclusion-ledger.v1"
VALIDATION_SCHEMA = "token-reconstruction.trr-p09-nested-b1-exclusion-ledger-validation.v1"
MANIFEST_SCHEMA = "token-reconstruction.trr-p09-nested-b1-exclusion-ledger-manifest.v1"
B0_ROWS = 1200
SEQUENCE_WIDTH = 192
BOS_TOKEN_ID = 128000
STYLES = ("pile", "finance")
IDENTITY_HASH_KEYS = ("rendered_sha256", "public_record_sha256", "final_sequence_sha256")
FORBIDDEN_OUTPUT_KEYS = {
    "token_ids",
    "input_ids",
    "labels",
    "source_text",
    "plaintext",
    "target_tokens",
    "replacement_token_ids",
}


class LedgerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_create_only(path: Path, value: Any) -> str:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise LedgerError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(value)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def require_hex(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LedgerError(f"{field} is not a SHA-256 hex digest")
    lowered = value.lower()
    if any(char not in "0123456789abcdef" for char in lowered):
        raise LedgerError(f"{field} is not a SHA-256 hex digest")
    return lowered


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_identity_only(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold().replace("-", "_")
            if lowered in FORBIDDEN_OUTPUT_KEYS:
                raise LedgerError(f"forbidden payload field leaked into ledger: {key}")
            _assert_identity_only(child)
    elif isinstance(value, list):
        for child in value:
            _assert_identity_only(child)


def load_inputs(records_path: Path, inputs_path: Path) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f"records JSON is unreadable: {records_path}") from exc
    if not isinstance(records, list) or len(records) != 12000:
        raise LedgerError("records must contain exactly 12,000 rows")
    if records_path.is_symlink() or not records_path.is_file():
        raise LedgerError("records path is not a regular file")
    with safe_open(str(inputs_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"token_ids", "attention_mask", "position_ids"}:
            raise LedgerError("public input tensor keys changed")
        tokens = handle.get_tensor("token_ids").contiguous()
        mask = handle.get_tensor("attention_mask").contiguous()
    if tuple(tokens.shape) != (12000, SEQUENCE_WIDTH) or tuple(mask.shape) != (12000, SEQUENCE_WIDTH):
        raise LedgerError("public input tensor geometry changed")
    if tokens.dtype != torch.int32 or mask.dtype != torch.uint8:
        raise LedgerError("public input tensor dtypes changed")
    return records, tokens, mask


def active_count(mask: torch.Tensor, row: int) -> int:
    values = mask[row]
    count = int(values.sum().item())
    if not bool(values[:count].eq(1).all().item()):
        raise LedgerError(f"row {row} attention mask is not a one-prefix mask")
    if not bool(values[count:].eq(0).all().item()):
        raise LedgerError(f"row {row} attention mask has nonzero padding")
    return count


def expected_sets(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, set[Any]]]:
    expected = {
        "ids": {style: set() for style in STYLES},
        "hashes": {style: set() for style in STYLES},
        "indices": {style: set() for style in STYLES},
    }
    for row in rows:
        style = row.get("dataset")
        if style not in STYLES:
            continue
        for field in ("record_id", "source_record_id"):
            value = row.get(field)
            if isinstance(value, str) and value:
                expected["ids"][style].add(value)
        for field in IDENTITY_HASH_KEYS:
            value = row.get(field)
            if isinstance(value, str):
                expected["hashes"][style].add(value)
        expected["indices"][style].add(int(row["source_index"]))
    return expected


def build_ledger(
    *,
    root: Path,
    records_path: Path,
    inputs_path: Path,
    producer_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records, tokens, mask = load_inputs(records_path, inputs_path)
    mismatch: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    active_counts: list[int] = []
    sidecar_h128_count = 0
    short_with_sidecar = 0
    eligible_by_stratum: Counter[str] = Counter()
    short_by_stratum: Counter[str] = Counter()
    source_index_by_dataset: dict[str, set[int]] = {key: set() for key in ("alpaca", *STYLES)}
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise LedgerError(f"record {index} is not an object")
        if int(raw.get("global_row", -1)) != index:
            raise LedgerError(f"record {index} global_row changed")
        dataset = raw.get("dataset_key")
        if dataset not in {"alpaca", *STYLES}:
            raise LedgerError(f"record {index} has an unknown dataset: {dataset!r}")
        record_id = raw.get("record_id")
        source_record_id = raw.get("source_record_id")
        if not isinstance(record_id, str) or not record_id or not isinstance(source_record_id, str) or not source_record_id:
            raise LedgerError(f"record {index} lacks identity IDs")
        source_index = raw.get("source_row_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            raise LedgerError(f"record {index} has an invalid source row index")
        rendered = require_hex(raw.get("rendered_sha256"), field=f"record {index} rendered_sha256")
        public_record = raw.get("public_record_sha256")
        if public_record is not None:
            public_record = require_hex(public_record, field=f"record {index} public_record_sha256")
        stratum = raw.get("stratum")
        if not isinstance(stratum, str) or not stratum:
            raise LedgerError(f"record {index} lacks stratum")
        active = active_count(mask, index)
        if active != int(raw.get("target_full_token_count", -1)):
            raise LedgerError(f"record {index} active length differs from metadata")
        if active <= 0 or int(tokens[index, 0].item()) != BOS_TOKEN_ID:
            raise LedgerError(f"record {index} violates BOS/active input contract")
        active_counts.append(active)
        sidecar = raw.get("sequence_h128_sha256")
        if sidecar is not None:
            sidecar = require_hex(sidecar, field=f"record {index} sequence_h128_sha256")
            sidecar_h128_count += 1
        entry: dict[str, Any] = {
            "bank_row": index,
            "bank_membership": "B0_prefix" if index < B0_ROWS else "B1_addition",
            "dataset": dataset,
            "stratum": stratum,
            "synthetic": bool(raw.get("synthetic", False)),
            "record_id": record_id,
            "source_record_id": source_record_id,
            "source_index": int(source_index),
            "rendered_sha256": rendered,
            "active_token_count": active,
        }
        if public_record is not None:
            entry["public_record_sha256"] = public_record
        source_index_by_dataset[dataset].add(int(source_index))
        if active >= 128:
            final_hash = _sequence_digest(tokens[index, :128].tolist())
            if sidecar is not None and sidecar != final_hash:
                mismatch.append({"bank_row": index, "stored": sidecar, "computed": final_hash})
            entry["final_sequence_sha256"] = final_hash
            entry["final_sequence_eligibility"] = "eligible_first_128_active_int32_ids_including_bos"
            eligible_by_stratum[stratum] += 1
        else:
            if sidecar is not None:
                short_with_sidecar += 1
            entry["final_sequence_eligibility"] = "ineligible_shorter_than_128_active_tokens_no_padding_hash"
            short_by_stratum[stratum] += 1
        rows.append(entry)
    if mismatch:
        raise LedgerError(f"{len(mismatch)} stored H128 hashes differ from public input tokens")
    counts = Counter(row["dataset"] for row in rows)
    ledger = {
        "schema": SCHEMA,
        "status": "IDENTITY_ONLY_NESTED_B1_COMPLETE_NO_EVALUATION_TRUTH",
        "task_id": "TRR-P09",
        "bank": "B1_nested_current_prefix_plus_10800_additions",
        "record_count": len(rows),
        "bank_counts": {"B0_prefix": B0_ROWS, "B1_addition": len(rows) - B0_ROWS},
        "dataset_counts": dict(sorted(counts.items())),
        "identity_fields": [
            "record_id",
            "source_record_id",
            "source_index",
            "rendered_sha256",
            "public_record_sha256_when_published",
            "final_sequence_sha256_when_actual_active_count_at_least_128",
        ],
        "sequence_hash_rule": "producer _sequence_digest over first 128 actual active int32 token IDs including BOS; short rows have no padded hash",
        "style_compatibility": {
            "trr0005_styles": list(STYLES),
            "alpaca_preserved_as_dataset_metadata": True,
            "alpaca_is_not_classified_as_pile_or_finance": True,
            "consumer": "scripts/trr0005_produce_confirmation.py::_collect_exclusions",
        },
        "records": rows,
    }
    _assert_identity_only(ledger)
    expected = expected_sets(rows)
    return ledger, {
        "schema": VALIDATION_SCHEMA,
        "status": "PASS_REAL_COLLECTOR_COMPLETE",
        "record_count": len(rows),
        "active_count": {"eligible_at_least_128": len([value for value in active_counts if value >= 128]), "ineligible_short": len([value for value in active_counts if value < 128])},
        "eligible_by_stratum": dict(sorted(eligible_by_stratum.items())),
        "short_by_stratum": dict(sorted(short_by_stratum.items())),
        "source_h128_metadata_count": sidecar_h128_count,
        "short_rows_with_source_h128_metadata_not_exposed": short_with_sidecar,
        "source_index_counts": {key: len(value) for key, value in sorted(source_index_by_dataset.items())},
        "expected_collector_sets": {kind: {style: sorted(values) for style, values in styles.items()} for kind, styles in expected.items()},
        "input_contract": {
            "records": {"path": str(records_path.resolve()), "bytes": records_path.stat().st_size, "sha256": sha256_file(records_path)},
            "inputs": {"path": str(inputs_path.resolve()), "bytes": inputs_path.stat().st_size, "sha256": sha256_file(inputs_path)},
            "token_tensor": "public fitting token_ids read on CPU; no activations, source text, or evaluation truth",
        },
        "canonical_hash_source": {"path": str(producer_path.resolve()), "bytes": producer_path.stat().st_size, "sha256": sha256_file(producer_path), "function": "_sequence_digest"},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repository_root.expanduser().resolve()
    records_path = args.records.expanduser().resolve()
    inputs_path = args.inputs.expanduser().resolve()
    producer_path = root / "scripts/trr0005_produce_confirmation.py"
    ledger_path = args.ledger.expanduser().resolve()
    validation_path = args.validation.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not records_path.is_file() or not inputs_path.is_file() or not producer_path.is_file():
        raise LedgerError("one or more immutable input files are unavailable")
    ledger, validation = build_ledger(root=root, records_path=records_path, inputs_path=inputs_path, producer_path=producer_path)
    ledger_sha = write_create_only(ledger_path, ledger)
    collected = _collect_exclusions([ledger_path])
    expected = validation.pop("expected_collector_sets")
    actual = {
        "ids": {style: sorted(collected.ids[style]) for style in STYLES},
        "hashes": {style: sorted(collected.hashes[style]) for style in STYLES},
        "indices": {style: sorted(collected.indices[style]) for style in STYLES},
    }
    if actual != expected:
        raise LedgerError("TRR-0005 collector did not consume the ledger identity sets exactly")
    alpaca_ids = {row["record_id"] for row in ledger["records"] if row["dataset"] == "alpaca"}
    if any(value in collected.ids[style] for style in STYLES for value in alpaca_ids):
        raise LedgerError("an Alpaca identity was misclassified as Pile or Finance")
    validation["collector"] = {
        "source_count": len(collected.sources),
        "new_identity_count": collected.sources[0].get("new_identity_count"),
        "actual_sets_sha256": hashlib.sha256(canonical_json(actual)).hexdigest(),
        "status": "PASS_IDS_HASHES_AND_INDICES_EXACT",
    }
    validation["ledger"] = {"path": str(ledger_path), "bytes": ledger_path.stat().st_size, "sha256": ledger_sha}
    validation_sha = write_create_only(validation_path, validation)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "COMPLETE_IDENTITY_ONLY_LEDGER_AND_REAL_CONSUMER_CHECK",
        "task_id": "TRR-P09",
        "generator": {"path": str(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size, "sha256": sha256_file(Path(__file__))},
        "repository_root": str(root),
        "repository_head_at_run": git_head(root),
        "command": " ".join([sys.executable, *sys.argv]),
        "inputs": validation["input_contract"],
        "outputs": {
            "ledger": {"path": str(ledger_path), "bytes": ledger_path.stat().st_size, "sha256": ledger_sha},
            "validation": {"path": str(validation_path), "bytes": validation_path.stat().st_size, "sha256": validation_sha},
        },
        "counts": {
            "records": ledger["record_count"],
            "b0_prefix": B0_ROWS,
            "b1_additions": ledger["record_count"] - B0_ROWS,
            "eligible_first128": validation["active_count"]["eligible_at_least_128"],
            "short_ineligible": validation["active_count"]["ineligible_short"],
        },
        "access_boundary": "Only existing public fitting records and public token_ids/attention_mask were read on CPU; no model, H activation, new source, evaluation panel, evaluation truth, or P03 holdout was accessed.",
    }
    write_create_only(manifest_path, manifest)
    print(json.dumps({"ledger": str(ledger_path), "ledger_sha256": ledger_sha, "validation": str(validation_path), "validation_sha256": validation_sha, "manifest": str(manifest_path), "records": ledger["record_count"], "eligible_first128": validation["active_count"]["eligible_at_least_128"], "short_ineligible": validation["active_count"]["ineligible_short"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LedgerError, OSError, RuntimeError, ValueError) as exc:
        print(f"nested P09 exclusion ledger failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
