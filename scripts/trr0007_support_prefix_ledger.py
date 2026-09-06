#!/usr/bin/env python3
"""Bind public fitting-prefix sequence exclusions for the TRR-0007 evaluator.

This helper reads only the already materialized public current and improved
fitting token tensors plus their public fit metadata.  For every row with at
least 128 active tokens it computes the first 128-token digest using the
trusted TRR-0005 ``_sequence_digest`` implementation and emits only
identity hashes and file provenance.  It never reads Arrow rows, source text,
labels, model weights, activations, or truth.

The emitted ``final_sequence_sha256`` key is intentionally the key consumed by
the trusted ``_collect_exclusions`` scanner.  Descriptive hashes are nested
below explicit ``pile``/``finance``/``other`` source buckets, while the full
union is repeated under both fresh ``pile`` and ``finance`` collector buckets:
any fitting prefix can collide with either future fresh style and must be
excluded regardless of its original style.  The ``other`` bucket records the
Alpaca rows for complete fit-prefix coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from safetensors import safe_open

from scripts import trr0005_produce_confirmation as trusted


TASK_ID = "TRR-0007"
SCHEMA = "token-reconstruction.trr0007-public-fitting-prefix-exclusions.v1"
STATUS = "PUBLIC_FIT_PREFIX_HASHES_ONLY"
PREFIX_TOKENS = 128
STYLE_ORDER = ("pile", "finance", "other")


class PrefixLedgerError(ValueError):
    """Raised when the public fitting-prefix inputs are not self-consistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PrefixLedgerError(f"{label} is unavailable or is a symlink: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _load_metadata(path: Path, *, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = _file_record(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrefixLedgerError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("records"), list):
        raise PrefixLedgerError(f"{label} does not contain a records list")
    rows = payload["records"]
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise PrefixLedgerError(f"{label} records are malformed")
    return record, [dict(row) for row in rows]


def _style_for_row(row: Mapping[str, Any]) -> str:
    dataset = row.get("dataset_key")
    if dataset == "pile":
        return "pile"
    if dataset == "finance":
        return "finance"
    if dataset == "alpaca":
        return "other"
    raise PrefixLedgerError(f"fit metadata has an unsupported dataset_key: {dataset!r}")


def _validate_metadata_row(row: Mapping[str, Any], index: int) -> None:
    if row.get("slot") != index:
        raise PrefixLedgerError(f"fit metadata slot changed at row {index}")
    active = row.get("active_token_count")
    if isinstance(active, bool) or not isinstance(active, int) or active < 0:
        raise PrefixLedgerError(f"fit metadata active_token_count is invalid at row {index}")


def _digest_prefix(values: Any) -> str:
    # This is the exact trusted public-selector convention: torch int32,
    # native C-order bytes, SHA-256.  Calling the trusted helper avoids a
    # second local serialization convention.
    return trusted._sequence_digest([int(value) for value in values])


def _hash_records(values: list[str]) -> list[dict[str, str]]:
    """Represent hashes as keyed objects for the trusted recursive scanner."""

    # The scanner recognizes a hash only when ``final_sequence_sha256`` is a
    # scalar mapping value. A bare list of strings would silently contribute
    # zero exclusions, so keep one keyed object per digest.
    return [{"final_sequence_sha256": value} for value in sorted(values)]


def _read_artifact(
    *,
    artifact_id: str,
    token_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    token_record = _file_record(token_path, label=f"{artifact_id} token tensor")
    metadata_record, rows = _load_metadata(
        metadata_path, label=f"{artifact_id} fit metadata"
    )
    try:
        with safe_open(str(token_path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"token_ids", "attention_mask"}
            if not required.issubset(keys):
                raise PrefixLedgerError(
                    f"{artifact_id} token tensor lacks token_ids/attention_mask"
                )
            token_ids = handle.get_tensor("token_ids")
            attention_mask = handle.get_tensor("attention_mask")
    except PrefixLedgerError:
        raise
    except Exception as exc:
        raise PrefixLedgerError(f"cannot read {artifact_id} public token tensor") from exc

    if tuple(token_ids.shape) != tuple(attention_mask.shape):
        raise PrefixLedgerError(f"{artifact_id} token and mask shapes differ")
    if len(token_ids.shape) != 2 or tuple(token_ids.shape)[1] < PREFIX_TOKENS:
        raise PrefixLedgerError(f"{artifact_id} token tensor is shorter than 128 columns")
    if int(token_ids.shape[0]) != len(rows):
        raise PrefixLedgerError(
            f"{artifact_id} metadata/token row counts differ: {len(rows)} vs {token_ids.shape[0]}"
        )

    hashes: dict[str, list[str]] = {style: [] for style in STYLE_ORDER}
    eligible_by_domain: Counter[str] = Counter()
    eligible_rows = 0
    for index, row in enumerate(rows):
        _validate_metadata_row(row, index)
        mask = attention_mask[index].tolist()
        active = int(sum(int(value) != 0 for value in mask))
        if active != int(row["active_token_count"]):
            raise PrefixLedgerError(
                f"{artifact_id} active count mismatch at row {index}: "
                f"metadata {row['active_token_count']} vs mask {active}"
            )
        if any(int(value) not in (0, 1) for value in mask):
            raise PrefixLedgerError(f"{artifact_id} attention mask is not binary at row {index}")
        if mask[:active] != [1] * active or mask[active:] != [0] * (len(mask) - active):
            raise PrefixLedgerError(f"{artifact_id} attention mask is not a contiguous prefix at row {index}")
        if active < PREFIX_TOKENS:
            continue
        eligible_rows += 1
        style = _style_for_row(row)
        digest = _digest_prefix(token_ids[index, :PREFIX_TOKENS].tolist())
        hashes[style].append(digest)
        eligible_by_domain[str(row.get("domain", "unknown"))] += 1

    for style in STYLE_ORDER:
        values = hashes[style]
        if len(values) != len(set(values)):
            raise PrefixLedgerError(f"{artifact_id} has duplicate 128-token prefixes in {style}")

    return {
        "artifact_id": artifact_id,
        "token_tensor": token_record,
        "fit_metadata": metadata_record,
        "tensor_shape": [int(value) for value in token_ids.shape],
        "eligible_row_count": eligible_rows,
        "eligible_rows_by_domain": dict(sorted(eligible_by_domain.items())),
        "hash_counts_by_style": {
            style: len(hashes[style]) for style in STYLE_ORDER
        },
        "hashes_by_style": {
            style: _hash_records(hashes[style])
            for style in STYLE_ORDER
        },
    }


def build_ledger(
    *,
    repository_root: Path,
    current_tokens: Path,
    current_metadata: Path,
    improved_tokens: Path,
    improved_metadata: Path,
) -> dict[str, Any]:
    root = repository_root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise PrefixLedgerError(f"repository root is unavailable: {root}")
    artifacts = [
        _read_artifact(
            artifact_id="current_enriched_public_fit",
            token_path=current_tokens,
            metadata_path=current_metadata,
        ),
        _read_artifact(
            artifact_id="improved_public_fit",
            token_path=improved_tokens,
            metadata_path=improved_metadata,
        ),
    ]
    union: dict[str, set[str]] = {style: set() for style in STYLE_ORDER}
    for artifact in artifacts:
        for style in STYLE_ORDER:
            union[style].update(
                record["final_sequence_sha256"]
                for record in artifact["hashes_by_style"][style]
            )
    total_eligible = sum(int(artifact["eligible_row_count"]) for artifact in artifacts)
    all_union = set().union(*union.values())
    result = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": STATUS,
        "created_utc": _utc_now(),
        "purpose": (
            "Identity-only exclusion of first 128-token prefixes from every "
            "current/improved public fitting row with at least 128 active tokens."
        ),
        "sequence_convention": {
            "hash_key": "final_sequence_sha256",
            "hash_algorithm": "SHA-256",
            "serialization": "torch.int32 native C-order bytes",
            "trusted_implementation": "scripts/trr0005_produce_confirmation.py:_sequence_digest",
            "prefix_tokens_including_bos": PREFIX_TOKENS,
            "active_rows_only": True,
            "mask_rule": "binary contiguous active prefix; first 128 active token IDs",
        },
        "artifacts": artifacts,
        "union_hashes_by_original_style": {
            style: _hash_records(sorted(union[style]))
            for style in STYLE_ORDER
        },
        "collector_exclusions_by_fresh_style": {
            style: _hash_records(sorted(all_union))
            for style in ("pile", "finance")
        },
        "counts": {
            "fit_artifacts": len(artifacts),
            "rows_per_artifact": [
                int(artifact["tensor_shape"][0]) for artifact in artifacts
            ],
            "eligible_rows_per_artifact": [
                int(artifact["eligible_row_count"]) for artifact in artifacts
            ],
            "eligible_rows_total_across_artifacts": total_eligible,
            "union_hashes_by_original_style": {
                style: len(union[style]) for style in STYLE_ORDER
            },
            "collector_hashes_by_fresh_style": {
                style: len(all_union) for style in ("pile", "finance")
            },
            "union_hashes_total": len(all_union),
        },
        "privacy_boundary": {
            "public_fit_artifacts_only": True,
            "contains_source_text": False,
            "contains_record_ids": False,
            "contains_source_indices": False,
            "contains_token_ids": False,
            "contains_target_labels": False,
            "contains_model_weights": False,
            "contains_activations": False,
            "contains_truth": False,
        },
        "execution": {
            "code_commit": _git_commit(root),
            "script": str(Path(__file__).resolve()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "network_used": False,
            "fresh_source_rows_opened": False,
            "truth_opened": False,
            "model_loaded": False,
        },
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--current-tokens",
        type=Path,
        default=Path("../TRR-0005/experiments/TRR-0005/corpus/coverage_mix_v1/constructed_public_tokens.safetensors"),
    )
    parser.add_argument(
        "--current-metadata",
        type=Path,
        default=Path("../TRR-0005/experiments/TRR-0005/public_activation_v1/enriched_fit_records.json"),
    )
    parser.add_argument(
        "--improved-tokens",
        type=Path,
        default=Path("experiments/TRR-0007/support/broader_capture_v2/enriched_fit_cut4.safetensors"),
    )
    parser.add_argument(
        "--improved-metadata",
        type=Path,
        default=Path("experiments/TRR-0007/support/broader_capture_v2/enriched_fit_records.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/TRR-0007/support/public_fit_prefix_exclusions_v1.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_ledger(
            repository_root=args.repository_root,
            current_tokens=args.current_tokens,
            current_metadata=args.current_metadata,
            improved_tokens=args.improved_tokens,
            improved_metadata=args.improved_metadata,
        )
        output = args.output.expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise PrefixLedgerError(f"output is create-only and already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        result["output"] = _file_record(output, label="prefix ledger output")
    except (OSError, PrefixLedgerError, RuntimeError, ValueError) as exc:
        print(f"TRR-0007 public fitting-prefix ledger failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
