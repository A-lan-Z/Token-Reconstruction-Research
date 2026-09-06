#!/usr/bin/env python3
"""Freeze all P04 predictions before the evaluator truth gate.

This command is intentionally metadata-only.  It validates the fresh public
panel and every required affine/S/H/D prediction group for both paired target
conditions, plus the separate native A1+A2 anchor groups.  It never opens an
evaluation-truth file.  The resulting receipt is the only input that permits
``score_predictions.py`` to read private truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

from scripts.trr_p04 import score_predictions as scorer


FREEZE_SCHEMA = "token-reconstruction.trr-p04-freeze.v1"
PREDICTION_SCHEMA = scorer.PREDICTION_SCHEMA
METHODS = scorer.DEFAULT_METHODS
SEEDS = scorer.DEFAULT_SEEDS
CONDITIONS = scorer.DEFAULT_CONDITIONS


class FreezeError(ValueError):
    """Raised when a public prediction set is incomplete or inconsistent."""


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


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FreezeError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FreezeError(f"{description} must be an object")
    return value


def _prediction_descriptor(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FreezeError(f"prediction file is unavailable: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _expected_groups() -> list[tuple[str, int | None, str, bool]]:
    return [
        (method, seed, condition, False)
        for condition in CONDITIONS
        for seed in SEEDS
        for method in METHODS
    ] + [("native_a1_a2", None, condition, True) for condition in CONDITIONS]


def _read_prediction_groups(
    paths: Sequence[Path],
    *,
    panel: Mapping[str, Mapping[str, Any]],
    expected: Sequence[tuple[str, int | None, str, bool]],
) -> dict[tuple[str, int | None, str, bool], set[str]]:
    expected_set = set(expected)
    groups: dict[tuple[str, int | None, str, bool], set[str]] = {}
    for path in paths:
        rows = scorer._read_jsonl(path, description=f"prediction file {path}")
        for line_number, row in enumerate(rows, start=1):
            group, record_id, _ = scorer._prediction_row(
                row,
                panel=panel,
                description=f"prediction file {path} line {line_number}",
            )
            if group not in expected_set:
                raise FreezeError(f"prediction group is not part of the frozen P04 set: {group}")
            if group not in groups:
                groups[group] = set()
            if record_id in groups[group]:
                raise FreezeError(f"prediction record is duplicated in group {group}: {record_id}")
            groups[group].add(record_id)
    all_ids = set(panel)
    anchor_ids = {record_id for record_id, row in panel.items() if row["anchor"]}
    for group in expected:
        if group not in groups:
            raise FreezeError(f"prediction group is missing: {group}")
        required = anchor_ids if group[3] else all_ids
        if groups[group] != required:
            raise FreezeError(f"prediction group has incomplete record coverage: {group}")
    return groups


def _state_descriptors(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the eight implementation-owned state bindings without loading tensors."""

    if path.is_symlink() or not path.is_file():
        raise FreezeError(f"student-state manifest is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"student-state manifest is invalid JSON: {path}") from exc
    rows = value.get("states") if isinstance(value, Mapping) else None
    required = {(method, seed) for method in METHODS for seed in SEEDS}
    if not isinstance(rows, list) or len(rows) != len(required):
        raise FreezeError("student-state manifest must bind all eight method/seed states")
    descriptors: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FreezeError(f"student-state descriptor {index} is malformed")
        method = row.get("method_id")
        seed = row.get("seed")
        if method not in METHODS or seed not in SEEDS:
            raise FreezeError(f"student-state descriptor {index} has unexpected identity")
        identity = (str(method), int(seed))
        if identity in seen:
            raise FreezeError("student-state manifest has duplicate identities")
        seen.add(identity)
        state_path_value = row.get("path")
        if not isinstance(state_path_value, str) or not state_path_value:
            raise FreezeError(f"student-state descriptor {index} has no path")
        state_path = Path(state_path_value).expanduser()
        if not state_path.is_absolute():
            state_path = path.parent / state_path
        state_path = state_path.resolve()
        descriptor = {
            "method_id": str(method),
            "seed": int(seed),
            "path": str(state_path),
            "bytes": int(row.get("bytes", -1)),
            "sha256": row.get("sha256"),
        }
        actual = _prediction_descriptor(state_path)
        if descriptor["bytes"] != actual["bytes"] or descriptor["sha256"] != actual["sha256"]:
            raise FreezeError(f"student-state manifest hash or size changed: {state_path}")
        descriptors.append(descriptor)
    if seen != required:
        raise FreezeError("student-state manifest is incomplete")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}, {"states": descriptors}


def build_freeze(
    *,
    panel_path: Path,
    prediction_paths: Sequence[Path],
    anchor_prediction_paths: Sequence[Path],
    state_manifest_path: Path,
    truth_dir: Path,
    output_path: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    started = time.monotonic()
    panel = scorer._load_json(panel_path, description="P04 public panel")
    panel_records = scorer._validate_panel(panel)
    expected = _expected_groups()
    all_prediction_paths = [*prediction_paths, *anchor_prediction_paths]
    if not prediction_paths or not anchor_prediction_paths:
        raise FreezeError("both student/reference and native A1+A2 anchor prediction files are required")
    descriptors = [_prediction_descriptor(path) for path in all_prediction_paths]
    _read_prediction_groups(all_prediction_paths, panel=panel_records, expected=expected)
    state_manifest_descriptor, state_payload = _state_descriptors(state_manifest_path.expanduser().resolve())
    panel_sha = _sha256_file(panel_path)
    freeze = {
        "schema": FREEZE_SCHEMA,
        "task_id": scorer.TASK_ID,
        "status": "FROZEN_BEFORE_TRUTH",
        "created_utc": _utc_now(),
        "panel_frozen": True,
        "predictions_frozen": True,
        "all_states_frozen": True,
        "truth_open_allowed": True,
        "truth_accessed": False,
        "panel": {
            "path": str(panel_path),
            "bytes": panel_path.stat().st_size,
            "sha256": panel_sha,
        },
        "prediction_files": descriptors,
        "state_manifest": state_manifest_descriptor,
        "state_files": state_payload["states"],
        "prediction_groups": [
            {
                "method_id": method,
                "seed": seed,
                "condition": condition,
                "anchor": anchor,
            }
            for method, seed, condition, anchor in expected
        ],
        "truth_files": [
            {
                "condition": condition,
                "path": str((truth_dir / f"{condition}.jsonl").resolve()),
                "content_hash_recorded_after_gate": True,
            }
            for condition in CONDITIONS
        ],
        "truth_boundary": {
            "prediction_and_panel_validation_completed_before_truth": True,
            "truth_rows_not_loaded_by_freezer": True,
            "student_inference_is_activation_only": True,
        },
        "execution": {
            "argv": list(argv),
            "python": sys.executable,
            "started_utc": None,
            "ended_utc": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "max_rss_bytes": _max_rss_bytes(),
            "truth_accessed": False,
            "model_loaded": False,
        },
    }
    freeze["execution"]["started_utc"] = freeze["created_utc"]
    if output_path.exists() or output_path.is_symlink():
        raise FreezeError(f"refusing to overwrite freeze receipt: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return freeze


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path, action="append", required=True)
    parser.add_argument("--anchor-prediction-file", type=Path, action="append", required=True)
    parser.add_argument("--state-manifest", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        build_freeze(
            panel_path=args.panel.expanduser().resolve(),
            prediction_paths=[path.expanduser().resolve() for path in args.prediction_file],
            anchor_prediction_paths=[path.expanduser().resolve() for path in args.anchor_prediction_file],
            state_manifest_path=args.state_manifest.expanduser().resolve(),
            truth_dir=args.truth_dir,
            output_path=args.output.expanduser().resolve(),
            argv=list(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]),
        )
    except FreezeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"output": str(args.output.resolve()), "status": "FROZEN_BEFORE_TRUTH"}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

