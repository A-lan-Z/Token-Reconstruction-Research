#!/usr/bin/env python3
"""Register the public nested Alpaca fit/validation stream for TRR-0004.

The command is preparation only.  It reads a local public Arrow cache and a
local tokenizer, writes metadata without source text or token IDs, and never
loads a model or opens evaluator-private truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from token_reconstruction.alpaca_split import (
    ALPACA_CACHE_REVISION,
    ALPACA_DATASET_ID,
    ALPACA_SPLIT,
    DEFAULT_BOS_TOKEN_ID,
    HISTORICAL_DATASET_FINGERPRINT,
    HISTORICAL_FIT_CANDIDATE_ROWS,
    HISTORICAL_MAX_TOKENS,
    HISTORICAL_MIN_FULL_TOKENS,
    HISTORICAL_SELECTION_SEED,
    build_split_registration,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular public metadata file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _tokenizer_files(path: Path) -> list[dict[str, Any]]:
    # Explicit allow-list prevents accidentally hashing or recording model
    # weights from a checkpoint snapshot.
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    )
    result = []
    for name in names:
        candidate = path / name
        if candidate.is_file():
            result.append(_file_record(candidate))
    if not result:
        raise ValueError(f"no tokenizer files found under {path}")
    return result


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-arrow", type=Path, required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=ALPACA_CACHE_REVISION)
    parser.add_argument("--dataset-fingerprint", default=HISTORICAL_DATASET_FINGERPRINT)
    parser.add_argument("--revision-ref", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument(
        "--tokenizer-revision",
        default="9213176726f574b556790deb65791e0c5aa438b6",
    )
    parser.add_argument("--pile-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-records", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    arrow = args.dataset_arrow.expanduser().resolve()
    info = args.dataset_info.expanduser().resolve()
    revision_ref = args.revision_ref.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    pile_receipt = args.pile_receipt.expanduser().resolve()
    for path in (arrow, info, revision_ref, pile_receipt):
        if not path.is_file():
            raise SystemExit(f"missing required public metadata file: {path}")
    if not tokenizer_path.is_dir():
        raise SystemExit(f"missing tokenizer directory: {tokenizer_path}")

    try:
        from datasets import Dataset
        from transformers import AutoTokenizer

        dataset = Dataset.from_file(str(arrow))
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, use_fast=True
        )
    except Exception as exc:
        raise SystemExit(f"public cache/tokenizer load failed: {exc}") from exc

    info_json = json.loads(info.read_text(encoding="utf-8"))
    split_info = info_json.get("splits", {}).get(ALPACA_SPLIT, {})
    if split_info.get("num_examples") != len(dataset):
        raise SystemExit("Arrow row count disagrees with dataset_info.json")
    if tuple(dataset.column_names) != ("instruction", "input", "output", "text"):
        raise SystemExit(f"unexpected public Alpaca columns: {dataset.column_names}")
    if tokenizer.bos_token_id != DEFAULT_BOS_TOKEN_ID:
        raise SystemExit(
            f"unexpected tokenizer BOS ID {tokenizer.bos_token_id}; expected {DEFAULT_BOS_TOKEN_ID}"
        )
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("tokenizer has no chat template")

    registration = build_split_registration(
        dataset,
        tokenizer,
        dataset_revision=args.dataset_revision,
        dataset_fingerprint=args.dataset_fingerprint,
        selection_seed=HISTORICAL_SELECTION_SEED,
        fit_candidate_rows=HISTORICAL_FIT_CANDIDATE_ROWS,
        expected_fit_records=HISTORICAL_FIT_CANDIDATE_ROWS,
        validation_records=args.validation_records,
        minimum_full_tokens=HISTORICAL_MIN_FULL_TOKENS,
        maximum_tokens=HISTORICAL_MAX_TOKENS,
        expected_bos_token_id=DEFAULT_BOS_TOKEN_ID,
    )

    script_path = Path(__file__).resolve()
    module_path = root / "src/token_reconstruction/alpaca_split.py"
    plan = {
        "schema": "token-reconstruction.trr0004-public-alpaca-split.v1",
        "task_id": "TRR-0004",
        "status": "REGISTERED_PUBLIC_FIT_SPLIT_NO_CONFIRMATION_GENERATED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "argv": [str(arg) for arg in (sys.argv if argv is None else [str(script_path), *argv])],
            "python": sys.executable,
            "git_commit": _git_commit(root),
            "script": _file_record(script_path),
            "module": _file_record(module_path),
            "model_loaded": False,
            "truth_accessed": False,
            "network_used": False,
        },
        "source": {
            "dataset": {
                "id": ALPACA_DATASET_ID,
                "split": ALPACA_SPLIT,
                "revision": args.dataset_revision,
                "revision_ref": _file_record(revision_ref),
                "arrow": _file_record(arrow),
                "dataset_info": _file_record(info),
                "dataset_info_num_examples": split_info.get("num_examples"),
                "dataset_info_num_bytes": split_info.get("num_bytes"),
                "loaded_fingerprint": getattr(dataset, "_fingerprint", None),
            },
            "tokenizer": {
                "id": args.tokenizer_id,
                "revision": args.tokenizer_revision,
                "path": str(tokenizer_path),
                "bos_token_id": tokenizer.bos_token_id,
                "files": _tokenizer_files(tokenizer_path),
            },
            "public_pile_validation_reference": {
                "role": "existing public validation metadata; not read for row selection",
                "receipt": _file_record(pile_receipt),
                "expected_records": 24,
            },
        },
        "registration": registration,
        "historical_provenance_boundary": {
            "retained_a1_exact_fit_rows_available": False,
            "retained_a1_exact_fit_duration_available": False,
            "interpretation": "this is a controlled public recipe recreation; it does not retroactively establish the retained TRR-0002 lens provenance",
        },
        "future_confirmation_exclusions": {
            "fail_closed": True,
            "required_public_metadata_sources": [
                "historical_fitting",
                "historical_evaluation",
                "current_fitting",
                "current_evaluation",
            ],
            "private_evaluator_contents_allowed": False,
            "generated_in_this_step": False,
        },
    }
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite existing plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "fit_records": registration["fit"]["record_count"],
        "fit_small_post_bos": registration["fit"]["small_nested"]["post_bos_positions"],
        "fit_large_post_bos": registration["fit"]["large_nested"]["post_bos_positions"],
        "validation_records": registration["validation"]["record_count"],
        "truth_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

