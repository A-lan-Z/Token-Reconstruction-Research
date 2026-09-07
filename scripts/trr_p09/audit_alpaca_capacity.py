#!/usr/bin/env python3
"""Run the bounded public Alpaca capacity audit for TRR-P09.

The audit renders and tokenizes the named public Arrow rows only to count
length eligibility and exact identity exclusions.  It never loads a model,
public activations, target weights, evaluation truth, or a P09 selection.  A
row's text and token IDs are kept transiently for the hash/length calculation
and are never written to the receipt.

The opaque exclusion ledger deliberately has an explicit namespace.  Older
P07 opaque values are not silently compared with a P09 rendered-text digest:
the resulting zero-hit count is diagnostic only and cannot certify
disjointness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping


DATASET_ID = "tatsu-lab/alpaca"
DATASET_REVISION = "dce01c9b08f87459cf36a430d809084718273017"
TOKENIZER_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
ARROW_SHA256 = "f45103036ed651f4c06d0a3c3e0fb7d53acb3074ed5c8e804a69c1efc1cea794"
BOS_TOKEN_ID = 128000
MAX_TOKENS = 192
MIN_FULL_TOKENS = 32
LENGTH_THRESHOLDS = (32, 64, 96, 128, 160, 191)
ROW_RE = re.compile(r":row-(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_index(value: str) -> int | None:
    match = ROW_RE.search(value)
    return int(match.group(1)) if match else None


def _walk_public_identity(value: Any) -> Iterable[tuple[str, Any]]:
    """Yield identity fields from one already-selected metadata record."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"record_id", "source_record_id", "rendered_sha256"}:
                yield key, item
            yield from _walk_public_identity(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_public_identity(item)


def _metadata_records(path: Path) -> list[Mapping[str, Any]]:
    """Read only published record lists, not unrelated planning fields."""

    metadata = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(metadata.get("records"), list):
        return [item for item in metadata["records"] if isinstance(item, Mapping)]
    registration = metadata.get("registration")
    if isinstance(registration, Mapping):
        records: list[Mapping[str, Any]] = []
        for name in ("fit", "validation"):
            section = registration.get(name)
            if isinstance(section, Mapping) and isinstance(section.get("records"), list):
                records.extend(item for item in section["records"] if isinstance(item, Mapping))
        return records
    return []


def _add_identity_metadata(path: Path, row_indices: set[int], rendered_hashes: set[str]) -> None:
    for record in _metadata_records(path):
        for key, value in _walk_public_identity(record):
            if key in {"record_id", "source_record_id"} and isinstance(value, str):
                if value.startswith(f"{DATASET_ID}/"):
                    index = _row_index(value)
                    if index is not None:
                        row_indices.add(index)
            elif key == "rendered_sha256" and isinstance(value, str):
                rendered_hashes.add(value)


def _load_exclusions(path: Path, *metadata_paths: Path) -> dict[str, Any]:
    exclusion = json.loads(path.read_text(encoding="utf-8"))
    selector = exclusion.get("selector_exclusion_sets", {})
    row_indices: set[int] = set()
    for value in selector.get("record_ids", []):
        if isinstance(value, str) and value.startswith(f"{DATASET_ID}/"):
            index = _row_index(value)
            if index is not None:
                row_indices.add(index)
    for value in selector.get("source_row_keys", []):
        if not isinstance(value, Mapping):
            continue
        if str(value.get("dataset_key")) != DATASET_ID:
            continue
        try:
            row_indices.add(int(value["row_index"]))
        except (KeyError, TypeError, ValueError):
            continue
    rendered_hashes: set[str] = set()
    for path in metadata_paths:
        if path.exists():
            _add_identity_metadata(path, row_indices, rendered_hashes)

    opaque = {
        str(value)
        for value in selector.get("opaque_sequence_or_reservation_digests", [])
        if isinstance(value, str)
    }
    return {
        "row_indices": row_indices,
        "rendered_hashes": rendered_hashes,
        "opaque_digests": opaque,
        "selector_record_id_count": len(selector.get("record_ids", [])),
        "selector_row_key_count": len(selector.get("source_row_keys", [])),
        "selector_opaque_count": len(opaque),
        "opaque_or_rendered_union_count": len(opaque | rendered_hashes),
        "opaque_rendered_overlap_count": len(opaque & rendered_hashes),
    }


def _render_rows(rows: list[Mapping[str, Any]], tokenizer: Any) -> list[str]:
    rendered: list[str] = []
    for row in rows:
        user = str(row.get("instruction") or "")[:1200]
        inp = str(row.get("input") or "")
        if inp:
            user += "\n\n" + inp
        output = str(row.get("output") or "")[:1200]
        rendered.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            + output
        )
    return rendered


def run_audit(
    *,
    arrow_path: Path,
    tokenizer_path: Path,
    exclusion_path: Path,
    metadata_paths: tuple[Path, ...],
    batch_size: int = 256,
) -> dict[str, Any]:
    # Heavy public dataset/tokenizer imports happen only when this explicitly
    # authorized count-only command is invoked.
    from datasets import Dataset
    from transformers import AutoTokenizer

    arrow_path = arrow_path.expanduser().resolve()
    tokenizer_path = tokenizer_path.expanduser().resolve()
    exclusion_path = exclusion_path.expanduser().resolve()
    if sha256_file(arrow_path) != ARROW_SHA256:
        raise RuntimeError("the pinned public Alpaca Arrow changed")
    exclusions = _load_exclusions(exclusion_path, *metadata_paths)
    dataset = Dataset.from_file(str(arrow_path))
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, use_fast=True
    )
    if len(dataset) != 52002 or tokenizer.bos_token_id != BOS_TOKEN_ID:
        raise RuntimeError("pinned public dataset/tokenizer identity changed")

    raw = Counter()
    post_bins = Counter()
    reasons = Counter()
    usable_ge = {str(n): 0 for n in LENGTH_THRESHOLDS}
    row_hits = hash_hits = both_hits = 0
    usable = length_eligible = 0
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        rendered = _render_rows(rows, tokenizer)
        encoded = tokenizer(rendered, add_special_tokens=False, padding=False, truncation=False)
        for offset, (text, input_ids) in enumerate(zip(rendered, encoded["input_ids"])):
            index = start + offset
            if not input_ids or int(input_ids[0]) != BOS_TOKEN_ID:
                raise RuntimeError(f"unexpected BOS/tokenizer output at row {index}")
            full = min(len(input_ids), MAX_TOKENS)
            post_bos = full - 1
            rendered_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            row_hit = index in exclusions["row_indices"]
            # This comparison is retained only as a namespace diagnostic.  It
            # is not applied to eligibility because opaque P07 values are not
            # guaranteed to be rendered-text SHA-256 values.
            rendered_hit = rendered_hash in exclusions["rendered_hashes"]
            opaque_hit = rendered_hash in exclusions["opaque_digests"]
            raw["total"] += 1
            raw["full_ge_32"] += int(full >= MIN_FULL_TOKENS)
            if full >= MIN_FULL_TOKENS:
                length_eligible += 1
            row_hits += int(row_hit)
            hash_hits += int(rendered_hit or opaque_hit)
            both_hits += int(row_hit and (rendered_hit or opaque_hit))
            if row_hit and (rendered_hit or opaque_hit):
                reasons["row+rendered_hash"] += 1
            elif row_hit:
                reasons["row"] += 1
            elif rendered_hit or opaque_hit:
                reasons["rendered_hash"] += 1
            if row_hit or rendered_hit or full < MIN_FULL_TOKENS:
                continue
            usable += 1
            for threshold in LENGTH_THRESHOLDS:
                usable_ge[str(threshold)] += int(post_bos >= threshold)
            if post_bos < 64:
                post_bins["eligible_32_63"] += 1
            elif post_bos < 96:
                post_bins["eligible_64_95"] += 1
            elif post_bos < 128:
                post_bins["eligible_96_127"] += 1
            elif post_bos < 160:
                post_bins["eligible_128_159"] += 1
            else:
                post_bins["eligible_160_191"] += 1

    return {
        "schema": "token-reconstruction.trr-p09-public-alpaca-capacity-count.v2",
        "task_id": "TRR-P09",
        "created_utc": utc_now(),
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "train",
            "rows": len(dataset),
            "arrow_path": str(arrow_path),
            "arrow_bytes": arrow_path.stat().st_size,
            "arrow_sha256": ARROW_SHA256,
        },
        "tokenizer": {
            "snapshot_revision": TOKENIZER_REVISION,
            "path": str(tokenizer_path),
            "bos_token_id": BOS_TOKEN_ID,
        },
        "rendering": {
            "max_user_chars": 1200,
            "max_output_chars": 1200,
            "max_tokens": MAX_TOKENS,
            "minimum_full_tokens": MIN_FULL_TOKENS,
        },
        "exclusions": {
            "metadata_path": str(exclusion_path),
            "metadata_sha256": sha256_file(exclusion_path),
            "row_indices_after_public_metadata_union": len(exclusions["row_indices"]),
            "rendered_text_sha256_values": len(exclusions["rendered_hashes"]),
            "opaque_sequence_or_reservation_digests": len(exclusions["opaque_digests"]),
            "opaque_or_rendered_union_count": exclusions["opaque_or_rendered_union_count"],
            "opaque_rendered_overlap_count": exclusions["opaque_rendered_overlap_count"],
            "row_hits": row_hits,
            "rendered_text_sha256_hits": hash_hits,
            "row_and_rendered_hash_hits": both_hits,
            "reason_counts": dict(reasons),
            "opaque_digest_namespace": "P07 sequence/reservation digests; not assumed to be rendered_text_sha256",
            "opaque_zero_hit_interpretation": "zero rendered-text matches are a canonicalization diagnostic, not proof of disjointness",
            "eligibility_applied": "keyed row identities and rendered-text hashes only; opaque values require namespace-specific final audit",
        },
        "counts": {
            "raw": dict(raw),
            "length_eligible": length_eligible,
            "usable_after_keyed_exclusions": usable,
            "usable_post_bos_ge": usable_ge,
            "post_bos_bins": dict(post_bins),
        },
        "access_boundary": {
            "public_source_rows_read": True,
            "public_tokenizer_read": True,
            "public_model_loaded": False,
            "public_forward_count": 0,
            "activations_loaded": False,
            "target_weights_loaded": False,
            "private_or_final_truth_opened": False,
            "p09_selection_executed": False,
            "source_text_or_token_ids_written": False,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": _git_head(Path(__file__).resolve().parents[2]),
            "command_scope": "metadata/tokenization count-only; no model/selection/fit/evaluation",
        },
    }


def _git_head(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    result = run_audit(
        arrow_path=args.arrow,
        tokenizer_path=args.tokenizer,
        exclusion_path=args.exclusion_manifest,
        metadata_paths=tuple(args.metadata),
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
