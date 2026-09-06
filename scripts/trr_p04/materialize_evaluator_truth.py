#!/usr/bin/env python3
"""Materialize the private P04 evaluator truth after a strict joint freeze.

This command is evaluator-side only.  It first validates the root-produced
``FROZEN_BEFORE_TRUTH`` receipt and its panel/truth-path binding, then reads
the pinned public source rows and writes two separate private JSONL files.
The two conditions share source records, but separate files make the scoring
boundary explicit.  Truth files and this receipt must live outside the
repository and are never consumed by the activation-only reconstruction
process.

``--preflight-only`` checks the freeze and output contract without loading a
tokenizer or source rows.  A real invocation must be authorized by the
freeze receipt itself; there is no bypass flag.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Mapping, Sequence

from scripts.trr_p04 import prepare_evaluator_observations as evaluator


TASK_ID = "TRR-P04"
FREEZE_SCHEMA = "token-reconstruction.trr-p04-freeze.v1"
TRUTH_SCHEMA = "token-reconstruction.trr-p04-private-truth.v1"
PREP_SCHEMA = "token-reconstruction.trr-p04-truth-materialization.v1"
CONDITIONS = ("public_base", "p04_evaluator_target_update_v1")
DEFAULT_TOKENIZER = evaluator.DEFAULT_TOKENIZER
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


class TruthMaterializationError(RuntimeError):
    """Raised when private truth preparation fails closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _safe_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TruthMaterializationError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TruthMaterializationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TruthMaterializationError(f"{label} must be a JSON object")
    return value


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_declared_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TruthMaterializationError(f"{label} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _validate_freeze_authorization(
    freeze_path: Path,
    *,
    selection_path: Path,
    truth_dir: Path,
) -> dict[str, Any]:
    """Validate only public freeze metadata; never opens a truth file."""

    freeze_path = freeze_path.expanduser().resolve()
    selection_path = selection_path.expanduser().resolve()
    truth_dir = truth_dir.expanduser().resolve()
    freeze = _load_json(freeze_path, label="joint freeze receipt")
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("task_id") != TASK_ID:
        raise TruthMaterializationError("joint freeze identity changed")
    if freeze.get("status") != "FROZEN_BEFORE_TRUTH":
        raise TruthMaterializationError("truth materialization requires FROZEN_BEFORE_TRUTH")
    for field in ("panel_frozen", "predictions_frozen", "all_states_frozen", "truth_open_allowed"):
        if freeze.get(field) is not True:
            raise TruthMaterializationError(f"joint freeze is missing required flag {field}")
    if freeze.get("truth_accessed") is not False:
        raise TruthMaterializationError("joint freeze truth_accessed must be false")
    boundary = freeze.get("truth_boundary")
    if isinstance(boundary, Mapping):
        if boundary.get("prediction_and_panel_validation_completed_before_truth") is not True:
            raise TruthMaterializationError("joint freeze has not completed pre-truth validation")
        if boundary.get("truth_rows_not_loaded_by_freezer") is not True:
            raise TruthMaterializationError("joint freezer reports truth rows were loaded")

    panel = freeze.get("panel")
    if not isinstance(panel, Mapping):
        raise TruthMaterializationError("joint freeze panel binding is absent")
    actual_panel_sha = _sha256_file(selection_path)
    declared_panel = _resolve_declared_path(panel.get("path"), base=freeze_path.parent, label="joint freeze panel")
    if declared_panel != selection_path or panel.get("sha256") != actual_panel_sha:
        raise TruthMaterializationError("joint freeze panel binding does not match frozen selection")

    if truth_dir.exists():
        if truth_dir.is_symlink() or not truth_dir.is_dir() or any(truth_dir.iterdir()):
            raise TruthMaterializationError(f"truth output must be a new empty directory: {truth_dir}")
    if _is_within(truth_dir, REPOSITORY_ROOT):
        raise TruthMaterializationError("private truth output must be outside the repository")

    declared_truth = freeze.get("truth_files")
    if not isinstance(declared_truth, list) or len(declared_truth) != len(CONDITIONS):
        raise TruthMaterializationError("joint freeze truth_files must list both conditions")
    by_condition: dict[str, Mapping[str, Any]] = {}
    for row in declared_truth:
        if not isinstance(row, Mapping) or row.get("condition") not in CONDITIONS:
            raise TruthMaterializationError("joint freeze truth-file descriptor is malformed")
        condition = str(row["condition"])
        if condition in by_condition:
            raise TruthMaterializationError("joint freeze truth-file conditions are duplicated")
        declared = _resolve_declared_path(row.get("path"), base=truth_dir, label=f"truth file {condition}")
        expected = (truth_dir / f"{condition}.jsonl").resolve()
        if declared != expected:
            raise TruthMaterializationError(f"joint freeze truth path is not bound to truth-dir for {condition}")
        by_condition[condition] = row
    if set(by_condition) != set(CONDITIONS):
        raise TruthMaterializationError("joint freeze truth-files do not cover both conditions")
    return {
        "path": str(freeze_path),
        "bytes": int(freeze_path.stat().st_size),
        "sha256": _sha256_file(freeze_path),
        "panel": {"path": str(selection_path), "bytes": int(selection_path.stat().st_size), "sha256": actual_panel_sha},
        "truth_dir": str(truth_dir),
        "truth_files": by_condition,
    }


def _write_exclusive(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TruthMaterializationError(f"refusing to overwrite create-only truth output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _truth_rows(
    rows: Sequence[Mapping[str, Any]], sequences: Sequence[Sequence[int]]
) -> list[dict[str, Any]]:
    if len(rows) != 72 or len(sequences) != len(rows):
        raise TruthMaterializationError("fresh panel truth geometry changed")
    result: list[dict[str, Any]] = []
    for index, (row, sequence) in enumerate(zip(rows, sequences)):
        values = [int(value) for value in sequence]
        expected_length = int(row["length_stratum"])
        if len(values) != expected_length + 1:
            raise TruthMaterializationError(f"fresh row {index} source geometry changed")
        if values[0] != evaluator.BOS_TOKEN_ID:
            raise TruthMaterializationError(f"fresh row {index} lost BOS geometry")
        if any(value < 0 or value >= evaluator.VOCAB_SIZE for value in values):
            raise TruthMaterializationError(f"fresh row {index} contains an invalid token ID")
        # The scorer consumes post-BOS labels.  Keep the BOS in capture
        # geometry, but do not duplicate it in either private score file.
        result.append({"record_id": str(row["record_id"]), "token_ids": values[1:]})
    return result


def _write_truth_files(
    truth_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    sequences: Sequence[Sequence[int]],
) -> dict[str, dict[str, Any]]:
    truth_rows = _truth_rows(rows, sequences)
    truth_dir = truth_dir.expanduser().resolve()
    if truth_dir.exists():
        if truth_dir.is_symlink() or not truth_dir.is_dir() or any(truth_dir.iterdir()):
            raise TruthMaterializationError(f"truth output must be a new empty directory: {truth_dir}")
    if _is_within(truth_dir, REPOSITORY_ROOT):
        raise TruthMaterializationError("private truth output must be outside the repository")
    truth_dir.mkdir(parents=True, exist_ok=False)
    text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in truth_rows)
    result: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        path = truth_dir / f"{condition}.jsonl"
        _write_exclusive(path, text)
        result[condition] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
            "records": len(truth_rows),
            "post_bos_tokens": sum(len(row["token_ids"]) for row in truth_rows),
        }
    return result


def materialize(
    *,
    selection_path: Path,
    freeze_path: Path,
    truth_dir: Path,
    tokenizer_path: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    selection_path = selection_path.expanduser().resolve()
    freeze_binding = _validate_freeze_authorization(
        freeze_path,
        selection_path=selection_path,
        truth_dir=truth_dir,
    )
    selection = evaluator._load_object(selection_path, label="P04 frozen selection")
    rows, panel = evaluator._validate_selection(selection, selection_path=selection_path)
    tokenizer_path = tokenizer_path.expanduser().resolve()
    if tokenizer_path.is_symlink() or not tokenizer_path.exists():
        raise TruthMaterializationError(f"pinned tokenizer is unavailable: {tokenizer_path}")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    except Exception as exc:
        raise TruthMaterializationError("pinned tokenizer loading failed") from exc
    datasets = evaluator._load_fresh_sources(selection)
    sequences, safe_rows = evaluator._materialize_fresh(selection, rows, datasets=datasets, tokenizer=tokenizer)
    if [row["record_id"] for row in safe_rows] != [row["record_id"] for row in rows]:
        raise TruthMaterializationError("truth materialization changed panel order")
    truth_files = _write_truth_files(truth_dir, rows=rows, sequences=sequences)
    receipt = {
        "schema": f"{PREP_SCHEMA}-receipt.v1",
        "task_id": TASK_ID,
        "status": "PASS_EVALUATOR_TRUTH_MATERIALIZED_AFTER_FREEZE",
        "created_utc": _utc_now(),
        "freeze": freeze_binding,
        "selection": {**panel, "path": str(selection_path), "sha256": _sha256_file(selection_path)},
        "truth_files": truth_files,
        "geometry": {
            "records_per_condition": len(rows),
            "conditions": list(CONDITIONS),
            "post_bos_tokens_per_condition": sum(int(row["length_stratum"]) for row in rows),
            "length_strata": [16, 32, 64, 128],
        },
        "access": {
            "freeze_validated_before_source_read": True,
            "source_rows_read": True,
            "source_text_materialized_transiently": True,
            "source_tokens_materialized_transiently": True,
            "truth_written_separately_per_condition": True,
            "evaluation_truth_read_before_materialization": False,
            "evaluation_truth_read_for_scoring": False,
            "target_update_weights_loaded": False,
            "student_states_loaded": False,
        },
        "execution": {
            "argv": list(argv),
            "safe_environment": _safe_environment(),
            "python": sys.version,
            "platform": platform.platform(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    _write_exclusive(Path(truth_dir).expanduser().resolve() / "truth_materialization_receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def preflight(*, selection_path: Path, freeze_path: Path, truth_dir: Path, argv: Sequence[str]) -> dict[str, Any]:
    started = time.perf_counter()
    selection_path = selection_path.expanduser().resolve()
    freeze_binding = _validate_freeze_authorization(
        freeze_path,
        selection_path=selection_path,
        truth_dir=truth_dir,
    )
    value = {
        "schema": f"{PREP_SCHEMA}-preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS_FREEZE_AUTHORIZATION_NO_SOURCE_NO_TRUTH_READ",
        "created_utc": _utc_now(),
        "freeze": freeze_binding,
        "selection": {"path": str(selection_path), "bytes": int(selection_path.stat().st_size), "sha256": _sha256_file(selection_path)},
        "truth_dir": str(truth_dir.expanduser().resolve()),
        "expected_truth_files": [f"{condition}.jsonl" for condition in CONDITIONS],
        "access": {
            "freeze_metadata_read": True,
            "source_rows_read": False,
            "source_tokens_read": False,
            "evaluation_truth_read": False,
            "target_update_weights_loaded": False,
        },
        "execution": {
            "argv": list(argv),
            "safe_environment": _safe_environment(),
            "python": sys.version,
            "platform": platform.platform(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    out = truth_dir.expanduser().resolve()
    if out.exists():
        if out.is_symlink() or not out.is_dir() or any(out.iterdir()):
            raise TruthMaterializationError(f"truth preflight output must be a new empty directory: {out}")
    if _is_within(out, REPOSITORY_ROOT):
        raise TruthMaterializationError("private truth output must be outside the repository")
    out.mkdir(parents=True, exist_ok=False)
    _write_exclusive(out / "truth_materialization_preflight.json", json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    effective_argv = list(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv])
    try:
        if args.preflight_only:
            value = preflight(
                selection_path=args.selection,
                freeze_path=args.freeze,
                truth_dir=args.truth_dir,
                argv=effective_argv,
            )
        else:
            value = materialize(
                selection_path=args.selection,
                freeze_path=args.freeze,
                truth_dir=args.truth_dir,
                tokenizer_path=args.tokenizer,
                argv=effective_argv,
            )
    except TruthMaterializationError as exc:
        print(f"P04 truth materialization failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": value["status"], "truth_dir": str(args.truth_dir.expanduser().resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
