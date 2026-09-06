#!/usr/bin/env python3
"""Qualify the TRR-0008 evaluator path against the frozen TRR-0007 IDs.

This is a source-free, truth-free preflight.  It opens only the already
registered TRR-0007 public observations, frozen decoder states, public
embedding, and archived prediction IDs.  Every scientific TRR-0008 method and
public cell is passed through :func:`trr0008_eval_runner.predict_current_h`
and compared element-for-element with the corresponding archived IDs.  No
prediction tensor is written; the receipt stores only tensor digests and
resource/code bindings.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from scripts import trr0008_eval_contract as contract
from scripts import trr0008_eval_runner as runner
from scripts import trr0008_timing as timing


QUALIFIER_SCHEMA = "token-reconstruction.trr0008-runner-qualifier.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0008-runner-qualifier-failure.v1"
QUALIFIER_STATUS = "RUNNER_PATH_EXACT_EQUIVALENCE_PASS"
FAILURE_STATUS = "RUNNER_PATH_QUALIFIER_FAILED_CLOSED"
RECORDS_PER_CELL = 128
MAXIMUM_SECONDS = 600
GUARD_INTERVAL = 8
TRR7_RUN_SCHEMA = "token-reconstruction.trr0007-prediction-run.v1"


class QualifierError(contract.ContractError):
    """Raised when the source-free evaluator qualifier cannot certify a pass."""


def _root(value: Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_symlink() or not path.is_dir():
        raise QualifierError(f"{label} is unavailable: {path}")
    return path


def _task_path(value: Path, *, root: Path, description: str) -> Path:
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    task_root = (root / "experiments" / contract.TASK_ID).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise QualifierError(f"{description} must be under {task_root}: {path}") from exc
    if path.is_symlink():
        raise QualifierError(f"{description} is a symlink: {path}")
    return path


def _file_record(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise QualifierError(f"{description} is unavailable: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": contract.sha256_file(path),
    }


def _write_create_only(path: Path, value: Mapping[str, Any], *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise QualifierError(f"{description} is create-only and already exists: {path}")
    contract.write_create_only(path, value)
    return _file_record(path, description=description)


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QualifierError("cannot resolve qualifier executable commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise QualifierError("qualifier executable commit is not a full hash")
    return value


def _current_code_bindings() -> dict[str, dict[str, Any]]:
    return {
        "qualifier": _file_record(Path(__file__), description="qualifier source"),
        "runner": _file_record(Path(runner.__file__), description="TRR8 evaluator runner"),
        "contract": _file_record(Path(contract.__file__), description="TRR8 evaluator contract"),
        "timing_loader": _file_record(Path(timing.__file__), description="TRR8 public loader"),
    }


def _validate_run_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = timing._load_json(path, label="TRR-0007 archived prediction run")
    if (
        run.get("schema") != TRR7_RUN_SCHEMA
        or run.get("task_id") != "TRR-0007"
        or run.get("status") != "COMPLETE_PUBLIC_PREDICTIONS_NO_TRUTH"
        or run.get("truth_opened") is not False
        or run.get("candidate_arrays_persisted") is not False
        or run.get("predictions_complete") is not True
    ):
        raise QualifierError("TRR-0007 archived run is not a complete public no-truth run")
    return run, _file_record(path, description="TRR-0007 archived prediction run")


def _first_mismatch(actual: torch.Tensor, expected: torch.Tensor) -> list[int]:
    mismatch = torch.nonzero(actual.ne(expected), as_tuple=False)
    return mismatch[0].tolist() if int(mismatch.shape[0]) else []


def _cell_tensors(cell: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cell, Mapping):
        activations = cell.get("activations")
        valid_mask = cell.get("valid_mask", cell.get("attention_mask"))
    else:
        activations = getattr(cell, "activations", None)
        valid_mask = getattr(cell, "valid_mask", None)
    if not isinstance(activations, torch.Tensor) or not isinstance(valid_mask, torch.Tensor):
        raise QualifierError("qualifier cell lacks activations or valid mask")
    return activations, valid_mask


def compare_runner_matrix(
    models: Mapping[str, torch.nn.Module],
    embedding: torch.Tensor,
    cells: Mapping[str, Any],
    archived: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    records: int = RECORDS_PER_CELL,
    guard: Any | None = None,
) -> dict[str, Any]:
    """Run the actual evaluator adapter and compare all archived rows exactly.

    ``guard`` is an optional callback used by the real run.  Keeping it
    injectable makes the CPU fixture deterministic without weakening the
    production fail-closed guard.
    """

    if records != RECORDS_PER_CELL:
        raise QualifierError(f"qualifier requires exactly {RECORDS_PER_CELL} rows per cell")
    if tuple(models) != tuple(contract.METHOD_ORDER):
        raise QualifierError("qualifier model set differs from the four scientific methods")
    if set(cells) != set(contract.CELL_ORDER):
        raise QualifierError("qualifier cell set differs from the four public cells")
    expected_keys = {
        f"{method_id}::{cell_id}"
        for method_id in contract.METHOD_ORDER
        for cell_id in contract.CELL_ORDER
    }
    if set(archived) != expected_keys:
        raise QualifierError("qualifier archived matrix is incomplete")

    results: dict[str, Any] = {}
    for method_id in contract.METHOD_ORDER:
        model = models[method_id]
        for cell_id in contract.CELL_ORDER:
            key = f"{method_id}::{cell_id}"
            activations, valid_mask = _cell_tensors(cells[cell_id])
            target = torch.as_tensor(archived[key]).detach().cpu().contiguous()
            expected_shape = (records, contract.STORED_SEQUENCE_TOKENS)
            if tuple(activations.shape) != (records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE):
                raise QualifierError(f"qualifier activation geometry changed: {cell_id}")
            if tuple(valid_mask.shape) != (records, contract.STORED_SEQUENCE_TOKENS):
                raise QualifierError(f"qualifier mask geometry changed: {cell_id}")
            if tuple(target.shape) != expected_shape or target.dtype != torch.long:
                raise QualifierError(f"qualifier archived prediction geometry changed: {key}")
            actual = torch.empty_like(target)
            for row_index in range(records):
                if guard is not None and (row_index % GUARD_INTERVAL == 0):
                    guard(f"before_{method_id}_{cell_id}_{row_index}")
                runner._synchronize(device)
                prediction = runner.predict_current_h(
                    model,
                    embedding,
                    activations[row_index],
                    valid_mask[row_index],
                    device=device,
                ).detach().cpu().contiguous()
                runner._synchronize(device)
                if tuple(prediction.shape) != (contract.STORED_SEQUENCE_TOKENS,):
                    raise QualifierError(f"qualifier prediction geometry changed: {key}/{row_index}")
                if not torch.equal(prediction, target[row_index]):
                    first = _first_mismatch(prediction, target[row_index])
                    raise QualifierError(f"runner/archive mismatch: {key}/{row_index}/{first}")
                actual[row_index] = prediction
            if guard is not None:
                guard(f"after_{method_id}_{cell_id}")
            results[key] = {
                "records": records,
                "prediction_sha256": contract.tensor_digest(actual),
                "archived_prediction_sha256": contract.tensor_digest(target),
                "exact_match": True,
            }
            del actual
    return results


def qualify(
    *,
    repository_root: Path,
    trr7_root: Path,
    output_path: Path,
    failure_path: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Execute the bounded source-free qualifier and write one receipt."""

    root = _root(repository_root, label="repository root")
    parent_root = _root(trr7_root, label="TRR-0007 root")
    output = _task_path(output_path, root=root, description="qualifier receipt")
    failure = _task_path(failure_path, root=root, description="qualifier failure receipt")
    started_utc = timing._utc_now()
    started = time.perf_counter()
    device = torch.device(device_name)
    guard_checks: list[dict[str, Any]] = []
    models: dict[str, torch.nn.Module] = {}
    embedding: torch.Tensor | None = None
    initial_code_commit: str | None = None
    initial_code_bindings: dict[str, Any] = {}

    def guard(stage: str) -> None:
        guard_checks.append(
            timing._guard(
                device,
                started=started,
                maximum_seconds=MAXIMUM_SECONDS,
                stage=stage,
            )
        )

    try:
        initial_code_commit = _git_head(root)
        initial_code_bindings = _current_code_bindings()
        if device.type == "cuda" and not torch.cuda.is_available():
            raise QualifierError("CUDA is unavailable")
        guard("before_registration")
        config = timing.TimingConfig(
            repository_root=root,
            trr7_root=parent_root,
            output_path=output,
            device=device_name,
            records_per_cell=RECORDS_PER_CELL,
            blocks=timing.DEFAULT_BLOCKS,
            warmup_runs=timing.DEFAULT_WARMUP_RUNS,
            maximum_seconds=MAXIMUM_SECONDS,
        )
        registration = timing._validate_registration(config)
        run_path = parent_root / timing.TRR7_RUN_MANIFEST
        run_manifest, run_record = _validate_run_manifest(run_path)
        numerical = timing._configure_numerics()
        guard("after_numerics")
        cells = timing._load_observations(
            Path(registration["observation_path"]),
            records=RECORDS_PER_CELL,
            repository_root=parent_root,
        )
        guard("after_observations")
        embedding, embedding_evidence = timing._load_embedding(
            registration,
            repository_root=root,
            device=device,
        )
        guard("after_embedding")
        loaded_models, model_evidence = timing._load_models(
            registration,
            trr7_root=parent_root,
            device=device,
        )
        models = {method_id: loaded_models[method_id] for method_id in contract.METHOD_ORDER}
        guard("after_models")
        archived: dict[str, torch.Tensor] = {}
        archived_evidence: dict[str, Any] = {}
        for method_id in contract.METHOD_ORDER:
            for cell_id in contract.CELL_ORDER:
                key = f"{method_id}::{cell_id}"
                value, evidence = timing._load_archived_prediction(
                    run_manifest,
                    trr7_root=parent_root,
                    method_id=method_id,
                    cell_id=cell_id,
                )
                archived[key] = value
                archived_evidence[key] = evidence
        guard("after_archived_predictions")
        matrix = compare_runner_matrix(
            models,
            embedding,
            cells,
            archived,
            device=device,
            records=RECORDS_PER_CELL,
            guard=guard,
        )
        guard("after_matrix")
        final_code_commit = _git_head(root)
        final_code_bindings = _current_code_bindings()
        if final_code_commit != initial_code_commit or final_code_bindings != initial_code_bindings:
            raise QualifierError("qualifier executable or source bindings changed during the run")
        receipt = {
            "schema": QUALIFIER_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": QUALIFIER_STATUS,
            "purpose": "exact source-free equivalence of the actual TRR-0008 evaluator path to archived TRR-0007 predictions",
            "methods": list(contract.METHOD_ORDER),
            "cells": list(contract.CELL_ORDER),
            "records_per_cell": RECORDS_PER_CELL,
            "matrix": matrix,
            "truth_opened": False,
            "source_text_or_target_labels": False,
            "candidate_arrays_persisted": False,
            "inputs": {
                "trr7_registration": registration["file"],
                "trr7_method_freeze": registration["method_freeze_file"],
                "trr7_observation_manifest": registration["observation_file"],
                "trr7_run_manifest": run_record,
                "trr7_archived_predictions": archived_evidence,
                "public_embedding": embedding_evidence,
            },
            "code": {
                "commit": initial_code_commit,
                "bindings": initial_code_bindings,
                "final_commit": final_code_commit,
                "final_bindings": final_code_bindings,
                "trr7_registered_code": registration["code"],
            },
            "numerical_settings": numerical,
            "resource_guard": {
                "maximum_seconds": MAXIMUM_SECONDS,
                "minimum_host_available_bytes": timing.DEFAULT_MIN_HOST_AVAILABLE_BYTES,
                "maximum_rss_bytes": timing.DEFAULT_MAX_RSS_BYTES,
                "minimum_free_gpu_bytes": timing.DEFAULT_MIN_FREE_GPU_BYTES,
                "maximum_reserved_gpu_bytes": timing.DEFAULT_MAX_RESERVED_GPU_BYTES,
                "checks": guard_checks,
            },
            "execution": {
                "device": str(device),
                "started_utc": started_utc,
                "ended_utc": timing._utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "command": list(sys.argv),
            },
        }
        _write_create_only(output, receipt, description="runner qualifier receipt")
        return receipt
    except Exception as exc:
        failure_receipt = {
            "schema": FAILURE_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": FAILURE_STATUS,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "truth_opened": False,
            "source_text_or_target_labels": False,
            "candidate_arrays_persisted": False,
            "execution": {
                "device": str(device),
                "started_utc": started_utc,
                "ended_utc": timing._utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "command": list(sys.argv),
            },
            "resource_guard_checks": guard_checks,
            "initial_code_commit": initial_code_commit,
            "initial_code_bindings": initial_code_bindings,
        }
        try:
            _write_create_only(failure, failure_receipt, description="runner qualifier failure receipt")
        except Exception:
            pass
        if isinstance(exc, QualifierError):
            raise
        raise QualifierError("runner qualifier failed closed") from exc
    finally:
        for model in models.values():
            model.cpu()
        models.clear()
        del embedding
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the source-free qualifier")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--trr7-root",
        type=Path,
        required=True,
        help="already-opened TRR-0007 worktree root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/TRR-0008/evaluation/qualifier/runner_qualifier.json"),
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=Path("experiments/TRR-0008/evaluation/qualifier/runner_qualifier.failure.json"),
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        print("TRR-0008 runner qualifier requires explicit --execute", file=sys.stderr)
        return 2
    try:
        value = qualify(
            repository_root=args.repository_root,
            trr7_root=args.trr7_root,
            output_path=args.output,
            failure_path=args.failure_output,
            device_name=args.device,
        )
    except (QualifierError, contract.ContractError, OSError, ValueError, RuntimeError) as exc:
        print(f"TRR-0008 runner qualifier failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(Path(args.output).expanduser().resolve()),
                "matrix_count": len(value["matrix"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
