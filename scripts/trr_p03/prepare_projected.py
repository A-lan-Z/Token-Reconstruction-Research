#!/usr/bin/env python3
"""Prepare one frozen full-vocabulary projected-prototype table.

Projection is construction state derived from the public prototype table and
the already published Alpaca lens. The resulting artifact is reused by every
matched/shifted Stage-1 arm, so reconstruction never reprojects the vocabulary
once per query chunk.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import resource
import sys
import time
from typing import Any

from safetensors.torch import save_file
import torch

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
for _path in (_SOURCE_ROOT, _SOURCE_ROOT / "src", _SOURCE_ROOT / "scripts" / "trr_p01"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from token_reconstruction.trr_p01 import PrototypeTable
from token_reconstruction.trr_p01.historical_comparators import (
    HISTORICAL_LENS_ARTIFACT_SHA256,
    load_published_frozen_lens,
)
from token_reconstruction.trr_p03.io import (
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    P03IOError,
    create_only_directory,
    file_record,
    sha256_file,
    write_json_exclusive,
)
from token_reconstruction.trr_p03.readouts import (
    PROJECTED_SCHEMA,
    project_prototypes,
)


TASK_ID = "TRR-P03"
VOCAB_SIZE = 128256
DEFAULT_REQUIRED_BYTES = 10 * 1024**3
DEFAULT_EXPECTED_PEAK_BYTES = 8 * 1024**3
DEFAULT_SEED = 20260906


def _configure_runtime(seed: int) -> None:
    if seed < 0:
        raise ProjectionPreparationError("seed must be non-negative")
    try:
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise ProjectionPreparationError(
            "Torch thread configuration must happen before projected preparation"
        ) from exc
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class ProjectionPreparationError(RuntimeError):
    """Raised when projected table preparation cannot be safely completed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _guard(required: int, expected: int) -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemAvailable", "MemTotal"}:
                values[key] = int(rest.strip().split()[0]) * 1024
    except (OSError, UnicodeError, ValueError, IndexError):
        pass
    available = int(values.get("MemAvailable", 0))
    total = int(values.get("MemTotal", 0))
    if available <= 0 or total <= 0 or available < int(required):
        raise ProjectionPreparationError(
            f"CPU resource guard failed closed: available={available} required={required} total={total}"
        )
    if required <= expected:
        raise ProjectionPreparationError("resource reservation must exceed expected peak")
    return {
        "status": "PASS",
        "required_bytes": int(required),
        "expected_peak_bytes": int(expected),
        "safety_margin_bytes": int(required - expected),
        "available_bytes_before": available,
        "total_bytes": total,
        "cuda_allocation": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--historical-lens", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prototype-chunk-size", type=int, default=8192)
    parser.add_argument("--required-bytes", type=int, default=DEFAULT_REQUIRED_BYTES)
    parser.add_argument("--expected-peak-bytes", type=int, default=DEFAULT_EXPECTED_PEAK_BYTES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--implementation-commit", default="UNBOUND_PRECOMMIT")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_runtime(args.seed)
    if args.prototype_chunk_size <= 0:
        raise ProjectionPreparationError("prototype chunk size must be positive")
    prototype_path = args.prototype.resolve()
    lens_path = args.historical_lens.resolve()
    if sha256_file(lens_path) != HISTORICAL_LENS_ARTIFACT_SHA256:
        raise ProjectionPreparationError("historical lens hash changed")
    guard = _guard(int(args.required_bytes), int(args.expected_peak_bytes))
    root = create_only_directory(args.output_root.resolve())
    progress_path = root / "phase_progress.jsonl"
    progress_path.touch()
    started_utc = _utc_now()
    started = time.perf_counter()
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": "resource_guard", "timestamp_utc": started_utc, **guard},
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
    table = PrototypeTable.load(
        prototype_path,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_cut_depth=CUT_DEPTH,
        expected_vocab_size=VOCAB_SIZE,
        expected_hidden_size=HIDDEN_SIZE,
    )
    lens = load_published_frozen_lens(lens_path, device=torch.device("cpu"))
    projection_started = time.perf_counter()
    projected = project_prototypes(
        table.prototypes,
        lens,
        prototype_chunk_size=args.prototype_chunk_size,
    )
    projection_seconds = time.perf_counter() - projection_started
    projected_path = root / "projected_prototypes.safetensors"
    if projected_path.exists() or projected_path.is_symlink():
        raise ProjectionPreparationError("projected artifact already exists")
    save_file(
        {"prototypes": projected},
        projected_path,
        metadata={
            "schema": PROJECTED_SCHEMA,
            "task_id": TASK_ID,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "cut_depth": str(CUT_DEPTH),
            "vocab_size": str(VOCAB_SIZE),
            "hidden_size": str(HIDDEN_SIZE),
            "dtype": "float32",
            "source_prototype_sha256": sha256_file(prototype_path),
            "lens_sha256": HISTORICAL_LENS_ARTIFACT_SHA256,
            "truth_opened": "false",
        },
    )
    projected_path.chmod(0o444)
    evidence_path = root / "preparation_evidence.json"
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr-p03-projected-preparation.v1",
            "task_id": TASK_ID,
            "status": "PROJECTED_TABLE_PREPARED_FOR_REUSE",
            "truth_opened": False,
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "implementation_commit": args.implementation_commit,
            "command": {"argv": [str(value) for value in sys.argv], "cwd": os.getcwd()},
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "seed": int(args.seed),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            },
            "prototype": file_record(prototype_path),
            "historical_lens": file_record(lens_path),
            "projected": file_record(projected_path, root=root),
            "geometry": {"vocab_size": VOCAB_SIZE, "hidden_size": HIDDEN_SIZE},
            "projection": {
                "metric": "cosine",
                "chunk_size": int(args.prototype_chunk_size),
                "output_dtype": "float32",
                "full_table_constructed_once": True,
                "reuse_scope": "all Stage-1 matched/shifted reconstruction arms",
            },
            "resource_guard": guard,
            "phases": {
                "table_and_lens_load_seconds": projection_started - started,
                "projection_seconds": projection_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            "peak_memory": {
                "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            },
        },
    )
    finish_path = root / "preparation_finish.json"
    write_json_exclusive(
        finish_path,
        {
            "schema": "token-reconstruction.trr-p03-projected-preparation-finish.v1",
            "task_id": TASK_ID,
            "status": "PROJECTED_TABLE_READY_BEFORE_RECONSTRUCTION",
            "truth_opened": False,
            "projected": file_record(projected_path, root=root),
            "evidence": file_record(evidence_path, root=root),
        },
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "projected": str(projected_path),
                "evidence": str(evidence_path),
                "elapsed_seconds": projection_seconds,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProjectionPreparationError, P03IOError) as exc:
        print(f"TRR-P03 projected preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
