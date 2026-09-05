#!/usr/bin/env python3
"""Freeze and score the TRR-P01 first-post-BOS identity diagnostic.

This diagnostic consumes only the public prototype table and the saved,
truth-free qualification outputs.  It writes the prediction artifact and a
receipt before reading the pilot plan's probe-token labels.  Thus the identity
comparison cannot influence the predictions or table lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from token_reconstruction.trr_p01 import PrototypeTable  # noqa: E402
from common import (  # noqa: E402
    BOS_TOKEN_ID,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    TASK_ID,
    VOCAB_SIZE,
    digest_tensor,
    file_record,
    load_json,
    require_create_only_directory,
    require_create_only_file,
    sha256_file,
    utc_now,
    validate_public_plan,
    write_json_exclusive,
)


POST_BOS_PREDICTION_SCHEMA = "token-reconstruction.trr-p01-post-bos-predictions.v1"
POST_BOS_FREEZE_SCHEMA = "token-reconstruction.trr-p01-post-bos-freeze.v1"
QUALIFICATION_SCHEMA = "token-reconstruction.trr-p01-qualification-output.v1"
QUALIFICATION_REPORT_SCHEMA = "token-reconstruction.trr-p01-qualification.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, default=None)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--prototype-chunk-size", type=int, default=8192)
    parser.add_argument("--implementation-commit", default=None)
    return parser


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"artifact must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"artifact must be a regular file: {path}")
    return file_record(path)


def _host_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(rest.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    values["process_max_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return values


def _runtime_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "kernel": platform.uname()._asdict(),
        "pid": os.getpid(),
        "selected_device": "cpu",
        "cuda_allocated": False,
    }


def _load_public_query(build_root: Path) -> tuple[torch.Tensor, Path, Path]:
    report_path = build_root / "qualification.json"
    chosen_path = build_root / "qualification_chosen.safetensors"
    if report_path.is_symlink() or not report_path.is_file():
        raise RuntimeError("qualification report is unavailable")
    if chosen_path.is_symlink() or not chosen_path.is_file():
        raise RuntimeError("qualification output is unavailable")
    # The report is hashed as a public input but deliberately not parsed yet:
    # its probe-token labels are part of the post-freeze comparison input.
    with safe_open(chosen_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"prototype_outputs"}:
            raise RuntimeError("qualification output fields changed")
        metadata = handle.metadata() or {}
        if metadata.get("schema") != QUALIFICATION_SCHEMA:
            raise RuntimeError("qualification output schema changed")
        if metadata.get("task_id") != TASK_ID or metadata.get("truth_opened") != "false":
            raise RuntimeError("qualification output truth state changed")
        queries = handle.get_tensor("prototype_outputs")
    if queries.dtype != torch.bfloat16 or tuple(queries.shape) != (256, HIDDEN_SIZE):
        raise RuntimeError("qualification output geometry or dtype changed")
    if not torch.isfinite(queries).all().item():
        raise RuntimeError("qualification output contains non-finite values")
    return queries, report_path, chosen_path


def _live_guard(table_path: Path, query_rows: int) -> dict[str, Any]:
    memory = _host_memory()
    table_bytes = int(table_path.stat().st_size)
    query_bytes = int(query_rows * HIDDEN_SIZE * 2)
    # The CPU table is already a committed 501 MiB artifact.  Include query,
    # score, and allocator headroom in the preflight estimate.
    raw_required = table_bytes + query_bytes * 8
    required = int(raw_required / 0.70)
    available = memory.get("MemAvailable")
    if available is None or available <= 0:
        raise RuntimeError(
            "CPU memory guard failed closed: /proc/meminfo MemAvailable is "
            "missing or non-positive"
        )
    if available < required:
        raise RuntimeError(
            f"CPU memory guard failed closed: available={available} required={required}"
        )
    return {
        "status": "PASS",
        "required_bytes": required,
        "raw_required_bytes": raw_required,
        "table_bytes": table_bytes,
        "query_bytes": query_bytes,
        "memory": memory,
        "safety_fraction": 0.70,
    }


def _freeze_predictions(
    output_root: Path,
    *,
    table_path: Path,
    qualification_path: Path,
    qualification_report_path: Path,
    plan_path: Path,
    queries: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    query_chunk_size: int,
    prototype_chunk_size: int,
    implementation_commit: str,
    started_utc: str,
    timing: Mapping[str, float],
    preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    prediction_path = output_root / "post_bos_predictions.safetensors"
    require_create_only_file(prediction_path)
    table_hash = _sha256(table_path)
    qualification_hash = _sha256(qualification_path)
    save_file(
        {key: value.to(torch.int32).contiguous() for key, value in predictions.items()},
        prediction_path,
        metadata={
            "schema": POST_BOS_PREDICTION_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": "false",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "cut_depth": "4",
            "bos_token_id": str(BOS_TOKEN_ID),
            "vocab_size": str(VOCAB_SIZE),
            "hidden_size": str(HIDDEN_SIZE),
            "table_sha256": table_hash,
            "qualification_output_sha256": qualification_hash,
        },
    )
    prediction_record = _record(prediction_path)
    freeze_path = output_root / "post_bos_freeze.json"
    require_create_only_file(freeze_path)
    receipt = {
        "schema": POST_BOS_FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": "POST_BOS_PREDICTIONS_FROZEN_BEFORE_LABELS",
        "created_utc": utc_now(),
        "truth_opened": False,
        "probe_labels_loaded": False,
        "implementation_commit": implementation_commit,
        "command": {
            "argv": [str(value) for value in sys.argv],
            "cwd": os.getcwd(),
            "selected_device": "cpu",
        },
        "runtime": _runtime_record(),
        "preflight": dict(preflight),
        "timing": dict(timing),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "cut_depth": 4,
            "bos_token_id": BOS_TOKEN_ID,
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
        },
        "plan": _record(plan_path),
        "qualification": {
            "report": _record(qualification_report_path),
            "output": _record(qualification_path),
            "query_shape": list(queries.shape),
            "query_dtype": str(queries.dtype).replace("torch.", ""),
            "query_digest": digest_tensor(queries),
        },
        "prototype": _record(table_path),
        "predictions": {
            "artifact": prediction_record,
            "keys": sorted(predictions),
            "shape": [int(queries.shape[0])],
            "dtype": "int32",
            "query_chunk_size": query_chunk_size,
            "prototype_chunk_size": prototype_chunk_size,
            "truth_opened": False,
        },
    }
    write_json_exclusive(freeze_path, receipt)
    return receipt, prediction_path, freeze_path


def _compare_after_freeze(
    plan_path: Path,
    qualification_report_path: Path,
    *,
    frozen_plan_sha256: str,
    frozen_qualification_report_sha256: str,
    predictions: Mapping[str, torch.Tensor],
    margins: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # The receipt binds both public label sources by bytes before either is
    # parsed. Recheck the bindings before opening the expected IDs.
    current_plan_sha256 = _sha256(plan_path)
    if current_plan_sha256 != frozen_plan_sha256:
        raise RuntimeError("pilot plan changed after predictions were frozen")
    current_report_sha256 = _sha256(qualification_report_path)
    if current_report_sha256 != frozen_qualification_report_sha256:
        raise RuntimeError("qualification report changed after predictions were frozen")

    # This is the first read of the expected probe-token labels.
    plan = load_json(plan_path)
    validate_public_plan(plan)
    qualification = plan.get("qualification")
    if not isinstance(qualification, Mapping):
        raise RuntimeError("qualification plan section is missing")
    expected_values = qualification.get("probe_token_ids")
    if not isinstance(expected_values, list) or len(expected_values) != 256:
        raise RuntimeError("qualification probe labels are missing")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in expected_values):
        raise RuntimeError("qualification probe labels must be integer IDs")
    expected = list(expected_values)
    if len(set(expected)) != len(expected) or sorted(expected) != expected:
        raise RuntimeError("qualification probe labels are not fixed ascending IDs")

    # The qualification report is a second public source of the frozen query
    # order. It must agree exactly with the plan before any identity score is
    # reported.
    report = load_json(qualification_report_path)
    if not isinstance(report, Mapping):
        raise RuntimeError("qualification report is not an object")
    if report.get("schema") != QUALIFICATION_REPORT_SCHEMA:
        raise RuntimeError("qualification report schema changed")
    if report.get("task_id") != TASK_ID or report.get("truth_opened") is not False:
        raise RuntimeError("qualification report truth state changed")
    reported_values = report.get("probe_token_ids")
    if not isinstance(reported_values, list) or len(reported_values) != 256:
        raise RuntimeError("qualification report probe labels are missing")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in reported_values):
        raise RuntimeError("qualification report probe labels must be integer IDs")
    reported = list(reported_values)
    if reported != expected:
        raise RuntimeError(
            "qualification report probe IDs do not match the frozen pilot plan"
        )
    arms: list[dict[str, Any]] = []
    for metric in ("cosine", "l2"):
        predicted = [int(value) for value in predictions[metric].tolist()]
        margin_values = [float(value) for value in margins[metric].tolist()]
        if any(not torch.isfinite(torch.tensor(value)).item() for value in margin_values):
            raise RuntimeError(f"non-finite {metric} margin")
        mismatches = []
        for index, (expected_id, predicted_id, margin) in enumerate(
            zip(expected, predicted, margin_values, strict=True), 1
        ):
            if expected_id != predicted_id:
                mismatches.append(
                    {
                        "probe_index": index,
                        "expected_token_id": expected_id,
                        "predicted_token_id": predicted_id,
                        "offset": predicted_id - expected_id,
                        "margin": margin,
                        "tie": margin == 0.0,
                    }
                )
        tie_count = sum(margin == 0.0 for margin in margin_values)
        collision_count = len(predicted) - len(set(predicted))
        arms.append(
            {
                "metric": metric,
                "probe_count": len(expected),
                "identity_count": sum(left == right for left, right in zip(expected, predicted, strict=True)),
                "identity_rate": sum(left == right for left, right in zip(expected, predicted, strict=True)) / len(expected),
                "predictions": predicted,
                "expected": expected,
                "mismatches": mismatches,
                "tie_count": tie_count,
                "nonidentity_tie_count": sum(item["tie"] for item in mismatches),
                "prediction_collision_count": collision_count,
                "status": "PASS" if not mismatches else "DIAGNOSTIC_NONIDENTITY",
            }
        )
    result = {
        "schema": "token-reconstruction.trr-p01-bos-identity.v2",
        "task_id": TASK_ID,
        "status": "PASS" if all(arm["status"] == "PASS" for arm in arms) else "DIAGNOSTIC_NONIDENTITY",
        "truth_opened": False,
        "probe_labels_loaded_after_freeze": True,
        "plan": _record(plan_path),
        "arms": arms,
    }
    return result, plan


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.query_chunk_size <= 0 or args.prototype_chunk_size <= 0:
        raise RuntimeError("lookup chunk sizes must be positive")
    build_root = args.build_root.resolve()
    plan_path = args.plan
    if plan_path.is_symlink() or not plan_path.is_file():
        raise RuntimeError("pilot plan is unavailable")
    output_root = require_create_only_directory(args.output_root.resolve())
    prototype_path = (args.prototype or (build_root / "boundary_prototypes.safetensors")).resolve()
    if prototype_path.is_symlink() or not prototype_path.is_file():
        raise RuntimeError("prototype table is unavailable")
    started_utc = utc_now()
    started = time.perf_counter()
    preflight = _live_guard(prototype_path, 256)
    write_json_exclusive(
        output_root / "preflight.json",
        {
            "schema": "token-reconstruction.trr-p01-post-bos-preflight.v1",
            "task_id": TASK_ID,
            "truth_opened": False,
            "selected_device": "cpu",
            "started_utc": started_utc,
            "guard": preflight,
            "prototype": _record(prototype_path),
            "qualification_path": str((build_root / "qualification_chosen.safetensors").resolve()),
            "plan_path": str(args.plan.resolve()),
        },
    )
    table_started = time.perf_counter()
    table = PrototypeTable.load(
        prototype_path,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_cut_depth=4,
        expected_vocab_size=VOCAB_SIZE,
        expected_hidden_size=HIDDEN_SIZE,
    )
    queries, qualification_report_path, qualification_path = _load_public_query(build_root)
    load_seconds = time.perf_counter() - table_started
    if args.query_chunk_size != 256 or args.prototype_chunk_size != 8192:
        raise RuntimeError("only the frozen diagnostic chunk sizes are accepted")
    lookup_started = time.perf_counter()
    predictions: dict[str, torch.Tensor] = {}
    margins: dict[str, torch.Tensor] = {}
    for metric in ("cosine", "l2"):
        nearest = table.nearest(
            queries,
            metric=metric,
            query_chunk_size=args.query_chunk_size,
            prototype_chunk_size=args.prototype_chunk_size,
        )
        predictions[metric] = nearest.predictions.cpu()
        margins[metric] = nearest.margins.cpu()
        if not torch.isfinite(nearest.scores).all().item() or not torch.isfinite(nearest.margins).all().item():
            raise RuntimeError(f"non-finite nearest scores for {metric}")
    lookup_seconds = time.perf_counter() - lookup_started
    timing = {
        "table_and_query_load_seconds": load_seconds,
        "lookup_seconds": lookup_seconds,
        "pre_freeze_seconds": time.perf_counter() - started,
    }
    implementation_commit = args.implementation_commit or os.environ.get(
        "TRR_P01_IMPLEMENTATION_COMMIT", "UNBOUND_PRECOMMIT"
    )
    receipt, prediction_path, freeze_path = _freeze_predictions(
        output_root,
        table_path=prototype_path,
        qualification_path=qualification_path,
        qualification_report_path=qualification_report_path,
        plan_path=plan_path,
        queries=queries,
        predictions=predictions,
        query_chunk_size=args.query_chunk_size,
        prototype_chunk_size=args.prototype_chunk_size,
        implementation_commit=implementation_commit,
        started_utc=started_utc,
        timing=timing,
        preflight=preflight,
    )
    # The receipt is written before this call.  No expected IDs or label list
    # has been read from the pilot plan before this point.
    result, plan = _compare_after_freeze(
        plan_path,
        qualification_report_path,
        frozen_plan_sha256=receipt["plan"]["sha256"],
        frozen_qualification_report_sha256=receipt["qualification"]["report"]["sha256"],
        predictions=predictions,
        margins=margins,
    )
    result.update(
        {
            "created_utc": started_utc,
            "scored_utc": utc_now(),
            "implementation_commit": implementation_commit,
            "model": receipt["model"],
            "prototype": receipt["prototype"],
            "plan": receipt["plan"],
            "qualification": receipt["qualification"],
            "prediction_freeze": {
                "receipt": _record(freeze_path),
                "artifact": _record(prediction_path),
            },
            "timing": {
                **timing,
                "post_freeze_label_comparison_seconds": time.perf_counter() - started - timing["pre_freeze_seconds"],
                "total_seconds": time.perf_counter() - started,
            },
            "runtime": _runtime_record(),
            "memory": _host_memory(),
            "preflight": preflight,
            "candidate_simulations": 0,
            "public_prefix_token_evaluations": 0,
            "code": _record(Path(__file__).resolve()),
        }
    )
    result_path = output_root / "post_bos_identity.json"
    write_json_exclusive(result_path, result)
    print(json.dumps({"status": result["status"], "output": str(result_path)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"check_post_bos: {exc}", file=sys.stderr)
        raise SystemExit(2)
