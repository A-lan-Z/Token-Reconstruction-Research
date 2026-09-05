#!/usr/bin/env python3
"""Materialize the preregistered public validation row slice for Track B.

This preparation reads only public calibration metadata until all disjointness
checks pass.  Safetensors row slices are then used so the fitting runner never
receives the original 32-row truth file, whose first eight rows are the shared
panel.  The script contains no evaluator truth or model calls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch


TASK_ID = "TRR-0003"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
BOS_TOKEN_ID = 128000
VOCAB_SIZE = 128256
START = 8
STOP = 32


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"regular JSON file required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def record_ids_from_panel(panel: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for cell in panel.get("cells", []):
        if not isinstance(cell, dict) or cell.get("condition") != "public_base":
            continue
        for row in cell.get("records", []):
            if not isinstance(row, dict) or not isinstance(row.get("record_id"), str):
                raise RuntimeError("panel record IDs are malformed")
            result.add(row["record_id"])
    if len(result) != 16:
        raise RuntimeError(f"expected 16 distinct shared panel records, found {len(result)}")
    return result


def declared_split_ids(plan: dict[str, Any], split: str) -> set[str]:
    rows = plan["data"]["selection"]["splits"][split]["records"]
    result = {str(row["record_id"]) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate IDs in TRR-0001 {split}")
    return result


def slice_tensor(path: Path, key: str, start: int, stop: int) -> tuple[torch.Tensor, dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"regular safetensors file required: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {key}:
            raise RuntimeError(f"{path} must contain only {key!r}")
        source_shape = tuple(handle.get_slice(key).get_shape())
        if len(source_shape) not in (2, 3) or source_shape[0] != 32:
            raise RuntimeError(f"unexpected source geometry for {path}: {source_shape}")
        result = handle.get_slice(key)[start:stop].contiguous()
        metadata = handle.metadata() or {}
    if result.shape[0] != stop - start:
        raise RuntimeError(f"row slice geometry changed for {path}")
    return result, {str(k): str(v) for k, v in metadata.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--panel", type=Path, default=Path("experiments/TRR-0003/footing/panel.json"))
    parser.add_argument("--source-records", type=Path, default=Path("outputs/TRR-0002/public-calibration/records.json"))
    parser.add_argument("--source-observations", type=Path, default=Path("outputs/TRR-0002/public-calibration/observations/public_base_cut4.safetensors"))
    parser.add_argument("--source-truth", type=Path, default=Path("outputs/TRR-0002/public-calibration/truth.safetensors"))
    parser.add_argument("--trr1-plan", type=Path, default=Path("experiments/TRR-0001/plan.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"output must be create-only: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()

    # Read metadata and declared IDs before opening either validation tensor.
    source_records = load_json((root / args.source_records).resolve())
    development = source_records.get("development")
    if not isinstance(development, list) or len(development) != 32:
        raise RuntimeError("public calibration development ledger must contain exactly 32 rows")
    selected_rows = development[START:STOP]
    selected_ids = [str(row["record_id"]) for row in selected_rows]
    if len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("validation row slice contains duplicate record IDs")
    panel = load_json((root / args.panel).resolve())
    panel_ids = record_ids_from_panel(panel)
    trr1_plan = load_json((root / args.trr1_plan).resolve())
    split_ids = {
        split: declared_split_ids(trr1_plan, split)
        for split in ("inverse_train", "target_update_train", "blind_evaluation")
    }
    selected_set = set(selected_ids)
    overlap_counts = {"panel": len(selected_set & panel_ids)}
    overlap_counts.update({split: len(selected_set & ids) for split, ids in split_ids.items()})
    if any(overlap_counts.values()):
        raise RuntimeError(f"validation slice overlap detected before label access: {overlap_counts}")

    source_records_path = (root / args.source_records).resolve()
    source_observations_path = (root / args.source_observations).resolve()
    source_truth_path = (root / args.source_truth).resolve()
    observations, observation_metadata = slice_tensor(source_observations_path, "activations", START, STOP)
    truth, truth_metadata = slice_tensor(source_truth_path, "token_ids", START, STOP)
    if tuple(observations.shape) != (STOP - START, 40, 2048):
        raise RuntimeError(f"validation observations geometry changed: {tuple(observations.shape)}")
    if tuple(truth.shape) != (STOP - START, 40):
        raise RuntimeError(f"validation truth geometry changed: {tuple(truth.shape)}")
    if truth.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise RuntimeError("validation labels must be integer token IDs")
    if truth[:, 0].ne(BOS_TOKEN_ID).any().item() or truth[:, 1:].lt(0).any().item() or truth[:, 1:].ge(VOCAB_SIZE).any().item():
        raise RuntimeError("validation labels have invalid token IDs")
    if not observations.dtype.is_floating_point or not torch.isfinite(observations).all().item():
        raise RuntimeError("validation observations are not finite floating point")

    observation_path = output / "public_validation_observations.safetensors"
    truth_path = output / "public_validation_truth.safetensors"
    records_path = output / "public_validation_records.json"
    save_file(
        {"activations": observations},
        observation_path,
        metadata={
            "schema": "token-reconstruction.trr0003-track-b-public-validation-slice.v1",
            "task_id": TASK_ID,
            "source_asset_sha256": sha256_file(source_observations_path),
            "source_row_slice": "[8:32)",
            "record_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
            "truth_source_not_included": "false",
        },
    )
    save_file(
        {"token_ids": truth},
        truth_path,
        metadata={
            "schema": "token-reconstruction.trr0003-track-b-public-validation-label-slice.v1",
            "task_id": TASK_ID,
            "source_asset_sha256": sha256_file(source_truth_path),
            "source_row_slice": "[8:32)",
            "record_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
            "truth_role": "public auxiliary validation only",
        },
    )
    write_json_exclusive(
        records_path,
        {
            "schema": "token-reconstruction.trr0003-track-b-public-validation-records.v1",
            "task_id": TASK_ID,
            "source_records": file_record(source_records_path, root),
            "source_split": "development",
            "row_slice": {"start": START, "stop": STOP, "stop_is_exclusive": True},
            "records": [
                {"record_id": row["record_id"], "source_row": START + i, "public_record_sha256": row.get("text_sha256")}
                for i, row in enumerate(selected_rows)
            ],
            "disjointness_checked_before_label_access": True,
            "overlap_counts": overlap_counts,
        },
    )
    evidence = {
        "schema": "token-reconstruction.trr0003-track-b-public-validation-preparation.v1",
        "task_id": TASK_ID,
        "track": "track_b",
        "started_utc": utc_now(),
        "ended_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "command": {"argv": [str(value) for value in sys.argv], "cwd": str(Path.cwd())},
        "git_commit_at_start_and_end": git_commit(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "panel": file_record((root / args.panel).resolve(), root),
        "trr1_plan": file_record((root / args.trr1_plan).resolve(), root),
        "source_records": file_record(source_records_path, root),
        "source_observations": file_record(source_observations_path, root),
        "source_truth": file_record(source_truth_path, root),
        "source_tensor_metadata": {"observations": observation_metadata, "truth": truth_metadata},
        "selection": {
            "split": "development",
            "row_slice": {"start": START, "stop": STOP, "stop_is_exclusive": True},
            "records": len(selected_ids),
            "record_ids": selected_ids,
        },
        "disjointness": {
            "panel": sorted(panel_ids),
            "fit_split": "inverse_train",
            "canonical_split": "blind_evaluation",
            "overlap_counts": overlap_counts,
            "checked_before_validation_label_access": True,
        },
        "outputs": {
            "observations": file_record(observation_path, root),
            "truth": file_record(truth_path, root),
            "records": file_record(records_path, root),
        },
        "geometry": {"observations": list(observations.shape), "truth": list(truth.shape)},
        "dependencies": {"python": sys.version, "torch": torch.__version__},
        "truth_role": "public auxiliary validation only; no evaluator-private labels",
    }
    write_json_exclusive(output / "validation_slice_evidence.json", evidence)
    print(json.dumps({"status": "validation_slice_ready", "output_root": str(output), "records": len(selected_ids), "overlap_counts": overlap_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
