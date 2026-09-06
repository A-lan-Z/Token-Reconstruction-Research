#!/usr/bin/env python3
"""Materialize the private TRR-P06 evaluator truth after the joint freeze.

This helper is deliberately separate from the public capture and scorer.  Its
first operation is the no-truth joint-freeze validator from ``score_frozen``;
only a validated ``FROZEN_P06_MATRIX_NO_TRUTH`` receipt permits reading the
selected public rows.  The resulting token arrays and binding manifest are
written to a create-only directory outside the reconstruction repository.

The helper is preparation code only until an authorized invocation supplies
``--execute``.  It never loads a model, target update, labels, or prediction
payloads, and it never copies source text into an artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import trr0005_produce_confirmation as trusted  # noqa: E402
from scripts.trr_p06 import capture_public as capture  # noqa: E402
from scripts.trr_p06 import score_frozen as scorer  # noqa: E402


TASK_ID = "TRR-P06"
TRUTH_SCHEMA = scorer.TRUTH_SCHEMA
TRUTH_STATUS = "TRUTH_READY_AFTER_JOINT_FREEZE"
SEQUENCE_TOKENS = scorer.SEQUENCE_TOKENS
SCORED_POST_BOS = scorer.SCORED_POST_BOS
VOCABULARY_SIZE = scorer.VOCABULARY_SIZE
BOS_TOKEN_ID = scorer.BOS_TOKEN_ID
DOMAINS = tuple(scorer.DOMAINS)
TARGETS = tuple(scorer.TARGETS)
TRUTH_FILE_SCHEMA = "token-reconstruction.trr-p06-private-truth-array.v1"


class TruthMaterializationError(RuntimeError):
    """Raised when post-freeze evaluator truth preparation fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TruthMaterializationError(f"file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TruthMaterializationError(f"file is unavailable: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TruthMaterializationError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TruthMaterializationError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise TruthMaterializationError(f"{description} must be a JSON object")
    return dict(value)


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
    return value or None


def _create_only_directory(path: Path, *, repository_root: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.exists() or raw.is_symlink():
        raise TruthMaterializationError(f"truth output directory is create-only: {raw}")
    resolved = raw.resolve(strict=False)
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError:
        resolved.mkdir(parents=True, mode=0o700)
        os.chmod(resolved, 0o700)
        return resolved
    raise TruthMaterializationError(f"truth output directory must be outside the repository: {resolved}")


def _write_array_create_only(path: Path, array: np.ndarray) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TruthMaterializationError(f"truth array is create-only: {path}")
    if array.dtype != np.int32 or tuple(array.shape) != (scorer.RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
        raise TruthMaterializationError(f"truth array geometry changed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    return _file_record(path)


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TruthMaterializationError(f"truth manifest is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    return _file_record(path)


def _source_paths(selection: Mapping[str, Any], domain: str) -> tuple[Path, ...]:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get(domain) if isinstance(sources, Mapping) else None
    files = descriptor.get("arrow_files") if isinstance(descriptor, Mapping) else None
    if not isinstance(files, list) or not files:
        raise TruthMaterializationError(f"selection has no frozen {domain} Arrow descriptor")
    result: list[Path] = []
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise TruthMaterializationError(f"selection {domain} Arrow descriptor is malformed")
        result.append(Path(str(item["path"])).expanduser().resolve())
    return tuple(result)


def _tokenizer_path(selection: Mapping[str, Any]) -> Path:
    sources = selection.get("public_sources_frozen")
    descriptor = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise TruthMaterializationError("selection has no frozen tokenizer descriptor")
    return Path(str(descriptor["path"])).expanduser().resolve()


def _validate_source_binding(
    selection: Mapping[str, Any],
    *,
    pile_paths: Sequence[Path],
    finance_paths: Sequence[Path],
    tokenizer_path: Path,
) -> None:
    actual = {
        "pile": trusted._dataset_descriptor(tuple(pile_paths), style="pile"),
        "finance": trusted._dataset_descriptor(tuple(finance_paths), style="finance"),
        "tokenizer": trusted._tokenizer_descriptor(tokenizer_path),
    }
    frozen = selection.get("public_sources_frozen")
    if not isinstance(frozen, Mapping):
        raise TruthMaterializationError("selection lacks frozen public source descriptors")
    for key in ("pile", "finance", "tokenizer"):
        if dict(actual[key]) != dict(frozen.get(key, {})):
            raise TruthMaterializationError(f"frozen public source binding changed: {key}")


def _materialize_records(
    *,
    selection_path: Path,
    universe_path: Path,
    pile_paths: Sequence[Path],
    finance_paths: Sequence[Path],
    tokenizer_path: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, list[Any]]]:
    selection, selected_rows = capture._validate_selection(selection_path)
    if selection.get("source_universe", {}).get("sha256") != _sha256_file(universe_path):
        raise TruthMaterializationError("source universe hash differs from selection binding")
    capture._validate_universe_binding(universe_path, selection)
    _validate_source_binding(
        selection,
        pile_paths=pile_paths,
        finance_paths=finance_paths,
        tokenizer_path=tokenizer_path,
    )
    tokenizer = trusted._load_tokenizer(tokenizer_path)
    datasets = {
        "pile": trusted._load_arrow_dataset(tuple(pile_paths)),
        "finance": trusted._load_arrow_dataset(tuple(finance_paths)),
    }
    records = capture._materialize_selected(selected_rows, datasets=datasets, tokenizer=tokenizer)
    for domain in DOMAINS:
        if len(records[domain]) != scorer.RECORDS_PER_DOMAIN:
            raise TruthMaterializationError(f"materialized source count changed: {domain}")
        declared_ids = [str(row["record_id"]) for row in selected_rows[domain]]
        actual_ids = [str(record.record_id) for record in records[domain]]
        if actual_ids != declared_ids:
            raise TruthMaterializationError(f"materialized source order changed: {domain}")
    return selection, selected_rows, records


def _arrays_and_hashes(
    records: Mapping[str, Sequence[Any]],
    *,
    expected: Mapping[str, Sequence[str]],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    arrays: dict[str, np.ndarray] = {}
    observed: dict[str, list[str]] = {}
    for domain in DOMAINS:
        values = [list(record.token_ids[:SEQUENCE_TOKENS]) for record in records[domain]]
        array = np.asarray(values, dtype=np.int32)
        if array.shape != (scorer.RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
            raise TruthMaterializationError(f"truth geometry changed: {domain}")
        if np.any(array[:, 0] != BOS_TOKEN_ID) or np.any(array < 0) or np.any(array >= VOCABULARY_SIZE):
            raise TruthMaterializationError(f"truth token range or BOS changed: {domain}")
        hashes = [hashlib.sha256(np.ascontiguousarray(row, dtype=np.int32).tobytes(order="C")).hexdigest() for row in array]
        if list(hashes) != list(expected[domain]):
            raise TruthMaterializationError(f"truth sequence fingerprints differ from frozen selection: {domain}")
        arrays[domain] = array
        observed[domain] = hashes
    return arrays, observed


def prepare_truth(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise TruthMaterializationError("truth materialization requires explicit --execute after joint freeze")
    root = Path(args.repository_root).expanduser().resolve()
    output_dir_arg = Path(args.output_dir).expanduser()
    freeze_path = Path(args.joint_freeze).expanduser().resolve()
    prediction_path = Path(args.prediction_manifest).expanduser().resolve()
    selection_path = Path(args.source_selection).expanduser().resolve()
    universe_path = Path(args.universe).expanduser().resolve()

    # No source rows or truth files are touched before this complete metadata
    # and prediction/state validation succeeds.
    validated = scorer.validate_joint_freeze(
        repository_root=root,
        freeze_receipt_path=freeze_path,
        prediction_manifest_path=prediction_path,
    )
    freeze_record = validated["freeze_receipt"]
    selection_record = validated["bindings"]["source_selection"]
    if selection_record["sha256"] != _sha256_file(selection_path):
        raise TruthMaterializationError("source selection path differs from joint-freeze binding")
    bound_selection = Path(str(validated["bindings"]["source_selection"]["path"])).expanduser()
    if not bound_selection.is_absolute():
        bound_selection = root / bound_selection
    if bound_selection.resolve() != selection_path:
        raise TruthMaterializationError("source selection path is not the joint-freeze source selection")

    # Only after the complete public gate and selection binding succeed do we
    # create the private destination. A failed pre-truth gate leaves no output
    # directory behind to be mistaken for a prepared sidecar.
    output_dir = _create_only_directory(output_dir_arg, repository_root=root)
    selection_meta = _load_json(selection_path, description="P06 source selection")
    source_paths = {
        "pile": tuple(Path(value).expanduser().resolve() for value in (_source_paths(selection_meta, "pile"))),
        "finance": tuple(Path(value).expanduser().resolve() for value in (_source_paths(selection_meta, "finance"))),
    }
    tokenizer_path = Path(args.tokenizer).expanduser().resolve() if args.tokenizer else _tokenizer_path(selection_meta)
    pile_paths = tuple(args.pile_arrow) if args.pile_arrow else source_paths["pile"]
    finance_paths = tuple(args.finance_arrow) if args.finance_arrow else source_paths["finance"]
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    selection, selected_rows, records = _materialize_records(
        selection_path=selection_path,
        universe_path=universe_path,
        pile_paths=pile_paths,
        finance_paths=finance_paths,
        tokenizer_path=tokenizer_path,
    )
    expected_sequences = validated["student_metadata"]["final_sequence_sha256"]
    arrays, observed_sequences = _arrays_and_hashes(records, expected=expected_sequences)

    files: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        files[domain] = _write_array_create_only(output_dir / f"{domain}__token_ids.npy", arrays[domain])

    manifest = {
        "schema": TRUTH_SCHEMA,
        "task_id": TASK_ID,
        "status": TRUTH_STATUS,
        "joint_freeze_sha256": freeze_record["sha256"],
        "source_selection_sha256": selection_record["sha256"],
        "observation_manifest_sha256": validated["bindings"]["observation_manifest"]["sha256"],
        "record_ids_sha256": dict(validated["student_metadata"]["record_ids_sha256"]),
        "final_sequence_sha256": observed_sequences,
        "domains": files,
        "records_per_domain": scorer.RECORDS_PER_DOMAIN,
        "sequence_tokens_including_bos": SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS,
        "vocabulary_size": VOCABULARY_SIZE,
        "target_conditions": list(TARGETS),
        "labels_shared_across_target_conditions": True,
        "truth_output_outside_repository": True,
        "source_text_written": False,
        "source_text_persisted": False,
        "source_text_loaded_transiently": True,
        "target_labels_loaded": False,
        "model_loaded": False,
        "truth_opened": True,
        "truth_materialized": True,
        "evaluator_truth_accessed": True,
        "scorer_truth_opened": False,
        "materialization_truth_read_after_joint_freeze": True,
        "code_commit": _git_commit(root),
        "joint_freeze_code_commit": _load_json(freeze_path, description="P06 joint freeze").get("code_commit"),
        "selection": {
            "path": str(selection_path),
            "bytes": int(selection_path.stat().st_size),
            "sha256": selection_record["sha256"],
            "records_per_domain": int(selection.get("records_per_domain", -1)),
        },
        "execution": {
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started_clock, 6),
            "command": list(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "truth_files_created": True,
        },
    }
    manifest_record = _write_json_create_only(output_dir / "truth.manifest.json", manifest)
    return {
        "task_id": TASK_ID,
        "status": TRUTH_STATUS,
        "truth_manifest": manifest_record,
        "truth_files": files,
        "joint_freeze_sha256": freeze_record["sha256"],
        "source_selection_sha256": selection_record["sha256"],
        "truth_opened": True,
        "truth_materialized": True,
        "evaluator_truth_accessed": True,
        "scorer_truth_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required acknowledgment after joint freeze")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--joint-freeze", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--pile-arrow", type=Path, nargs="*")
    parser.add_argument("--finance-arrow", type=Path, nargs="*")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare_truth(args)
    except (TruthMaterializationError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-P06 truth materialization failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
