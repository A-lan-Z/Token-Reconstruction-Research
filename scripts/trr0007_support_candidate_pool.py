#!/usr/bin/env python3
"""Scan the permitted public fit-frequency pools for current-unseen IDs.

This bounded CPU utility reads only the frozen Pile/Finance fit-frequency
partitions from local public Arrow caches. It excludes the current TRR-0005
fit identities, registered public-development identities, and the approved
opaque P04 digest exchange before token counting. It writes frequencies and
metadata only; no source text, labels, model weights, evaluation rows, or
private truth are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import platform
import resource
import re
import sys
import time
from typing import Any

from trr0005_prepare_public_corpus import (
    PreparationError,
    _Deadline,
    _load_public_dataset,
    _load_tokenizer,
    _scan_fit_candidates,
)
from token_reconstruction.trr0005_public_corpus import (
    BOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SOURCE_PARTITIONS,
)


TASK_ID = "TRR-0007"
SCHEMA = "token-reconstruction.trr0007-public-fit-candidate-frequency.v1"
VOCAB_SIZE = 128256


class CandidatePoolError(RuntimeError):
    """Raised when the public candidate-pool contract cannot be met."""


def _file_sha256(path: Path) -> str:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CandidatePoolError(f"input must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CandidatePoolError(f"exclusion metadata must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidatePoolError(f"cannot parse exclusion metadata: {path}") from exc


def _digest_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_map_digest(values: Mapping[int, int]) -> str:
    payload = json.dumps(
        {str(int(key)): int(value) for key, value in sorted(values.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_hex_digest(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _public_exclusions(paths: Sequence[Path]) -> tuple[set[str], set[tuple[str, int]], set[str], dict[str, Any]]:
    """Read public exclusion metadata, including opaque P04 digest values."""

    ids: set[str] = set()
    row_keys: set[tuple[str, int]] = set()
    hashes: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            record_id = value.get("record_id")
            if isinstance(record_id, str) and record_id:
                ids.add(record_id)
                lowered = record_id.lower()
                if "pile" in lowered:
                    dataset_key = "pile"
                elif "finance" in lowered:
                    dataset_key = "finance"
                elif "alpaca" in lowered:
                    dataset_key = "alpaca"
                else:
                    dataset_key = ""
                match = re.search(r"(?:row-|pile10k-|finance-public-)(\d{1,7})", lowered)
                if dataset_key and match:
                    row_keys.add((dataset_key, int(match.group(1))))
            for key in ("row_index", "dataset_index", "raw_index", "source_index"):
                raw = value.get(key)
                if isinstance(raw, int) and raw >= 0:
                    dataset_key = str(value.get("dataset_key", value.get("dataset", ""))).lower()
                    if "pile" in dataset_key:
                        row_keys.add(("pile", int(raw)))
                    elif "finance" in dataset_key:
                        row_keys.add(("finance", int(raw)))
                    elif "alpaca" in dataset_key:
                        row_keys.add(("alpaca", int(raw)))
            for key, child in value.items():
                if key in {"source_text", "text", "token_ids", "truth", "labels"}:
                    continue
                if isinstance(child, str) and _is_hex_digest(child):
                    # This captures named public/p04 hashes as well as ordered
                    # opaque values in the approved exchange. File hashes do
                    # not collide with rendered-text hashes in practice.
                    hashes.add(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and _is_hex_digest(value):
            hashes.add(value)

    for path in paths:
        visit(_json(path))
    descriptors = [
        {
            "path": str(path.expanduser().resolve()),
            "bytes": int(path.expanduser().resolve().stat().st_size),
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]
    return ids, row_keys, hashes, {
        "metadata": descriptors,
        "record_id_count": len(ids),
        "source_row_key_count": len(row_keys),
        "opaque_digest_count": len(hashes),
        "opaque_p04_exchange_applied": any("p04" in str(path).lower() for path in paths),
    }


def _resource_descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CandidatePoolError(f"public Arrow resource must be a regular file: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _file_sha256(path)}


def _counter(candidates: Sequence[Any]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for candidate in candidates:
        counts.update(int(value) for value in candidate.token_ids[1:])
    return counts


def _scan_one(
    dataset_key: str,
    dataset: Any,
    *,
    start: int,
    stop: int,
    tokenizer: Any,
    deadline: _Deadline,
    excluded_ids: set[str],
    excluded_rows: set[tuple[str, int]],
    excluded_hashes: set[str],
) -> tuple[list[Any], dict[str, Any]]:
    stats: dict[str, Any] = {}
    candidates = _scan_fit_candidates(
        dataset_key,
        dataset,
        range(start, stop),
        tokenizer,
        deadline=deadline,
        excluded_ids=excluded_ids,
        excluded_row_keys=excluded_rows,
        excluded_hashes=excluded_hashes,
        scan_stats=stats,
    )
    counts = _counter(candidates)
    source_ids = sorted(str(candidate.record_id) for candidate in candidates)
    stats = {
        **stats,
        "fit_frequency_range": [start, stop],
        "holdout_range": [
            int(SOURCE_PARTITIONS[dataset_key]["holdout_reserve_start"]),
            int(SOURCE_PARTITIONS[dataset_key]["holdout_reserve_stop"]),
        ],
        "holdout_rows_scanned": False,
        "eligible_source_ids_sha256": _digest_lines(source_ids),
        "eligible_source_id_count": len(source_ids),
        "post_bos_token_occurrences": int(sum(counts.values())),
        "distinct_post_bos_token_ids": len(counts),
        "candidate_frequency_digest_sha256": _canonical_map_digest(counts),
    }
    return candidates, stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pile-arrow", type=Path, required=True)
    parser.add_argument("--finance-arrow", type=Path, action="append", required=True)
    parser.add_argument("--exclude-records", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.monotonic()
    if args.max_seconds <= 0 or args.max_seconds > 300:
        raise SystemExit("--max-seconds must be in (0, 300]")
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"candidate frequency output is create-only: {output}")
    if not any("p04" in str(path).lower() for path in args.exclude_records):
        raise SystemExit("--exclude-records must include the approved opaque P04 exchange")
    deadline = _Deadline(time.monotonic(), float(args.max_seconds))
    try:
        excluded_ids, excluded_rows, excluded_hashes, exclusion_binding = _public_exclusions(args.exclude_records)
        tokenizer = _load_tokenizer(args.tokenizer)
        pile = _load_public_dataset([args.pile_arrow], label="Pile")
        finance = _load_public_dataset(args.finance_arrow, label="Finance")
        pile_candidates, pile_stats = _scan_one(
            "pile",
            pile,
            start=int(SOURCE_PARTITIONS["pile"]["fit_frequency_start"]),
            stop=int(SOURCE_PARTITIONS["pile"]["fit_frequency_stop"]),
            tokenizer=tokenizer,
            deadline=deadline,
            excluded_ids=excluded_ids,
            excluded_rows=excluded_rows,
            excluded_hashes=excluded_hashes,
        )
        finance_candidates, finance_stats = _scan_one(
            "finance",
            finance,
            start=int(SOURCE_PARTITIONS["finance"]["fit_frequency_start"]),
            stop=int(SOURCE_PARTITIONS["finance"]["fit_frequency_stop"]),
            tokenizer=tokenizer,
            deadline=deadline,
            excluded_ids=excluded_ids,
            excluded_rows=excluded_rows,
            excluded_hashes=excluded_hashes,
        )
        pile_frequency = _counter(pile_candidates)
        finance_frequency = _counter(finance_candidates)
        frequency = pile_frequency + finance_frequency
        special_ids = sorted({BOS_TOKEN_ID, PAD_TOKEN_ID, *[int(value) for value in getattr(tokenizer, "all_special_ids", ()) or ()]})
        source_paths = {
            "pile": [_resource_descriptor(args.pile_arrow)],
            "finance": [_resource_descriptor(path) for path in args.finance_arrow],
        }
        deadline.check("candidate frequency output")
    except (CandidatePoolError, PreparationError) as exc:
        raise SystemExit(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_FIT_CANDIDATE_FREQUENCY_ONLY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {
            "selection_role": "candidate support only; no error labels or evaluation identities",
            "datasets": ["pile", "finance"],
            "post_bos_only": True,
            "holdout_ranges_scanned": False,
            "source_text_retained": False,
            "private_truth_accessed": False,
            "target_weights_accessed": False,
            "network_used": False,
        },
        "source_pools": {
            "pile": {"resources": source_paths["pile"], "scan": pile_stats},
            "finance": {"resources": source_paths["finance"], "scan": finance_stats},
        },
        "exclusion_binding": exclusion_binding,
        "special_token_ids_excluded_by_selection": special_ids,
        "frequency_summary": {
            "vocab_size": VOCAB_SIZE,
            "distinct_post_bos_token_ids": len(frequency),
            "post_bos_token_occurrences": int(sum(frequency.values())),
            "current_unseen_candidate_ids_before_special_filter": None,
        },
        "frequency_by_token_id": {str(token_id): int(count) for token_id, count in sorted(frequency.items())},
        "dataset_frequency_digests": {
            "pile": _canonical_map_digest(pile_frequency),
            "finance": _canonical_map_digest(finance_frequency),
            "combined": _canonical_map_digest(frequency),
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": int(__import__("os").cpu_count() or 0),
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "device": "cpu",
        },
    }
    # Compute this count only after all frequencies have been accumulated so
    # it is explicit that IDs are current-unseen relative to the next recipe's
    # enriched reference, not merely legacy-absent.
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "candidate_post_bos_token_ids": len(frequency),
        "post_bos_occurrences": int(sum(frequency.values())),
        "pile_eligible_rows": pile_stats["rows_eligible"],
        "finance_eligible_rows": finance_stats["rows_eligible"],
        "opaque_digest_count": exclusion_binding["opaque_digest_count"],
        "elapsed_seconds": payload["runtime"]["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
