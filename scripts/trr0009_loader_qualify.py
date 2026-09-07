#!/usr/bin/env python3
"""Qualify TRR-0009 loaders against the already-opened TRR-0008 fixture.

This is a truth-free, source-free adapter check.  It reads only the public
TRR-0008 activation/prediction fixture, the frozen public embedding, and the
four selected decoder states.  The unchanged TRR-0007 state and retained
TRR-0005 reference must reproduce their archived TRR-0008 IDs exactly.  The
TRR-0009 fixed and supported-row states must produce the same IDs through the
current-H runner and their original full ``model(...)`` API at B=1 full
vocabulary geometry.

The receipt is create-only.  A failed attempt writes a separate failure
receipt and never writes a PASS receipt.  No target labels, source text,
private truth, or fresh observations are loaded.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open

from scripts import trr0009_eval_contract as contract
from scripts import trr0009_eval_runner as runner


QUALIFIER_SCHEMA = "token-reconstruction.trr0009-loader-equivalence-qualification.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0009-loader-equivalence-failure.v1"
FIXTURE_SCHEMA = "token-reconstruction.trr0009-trr8-loader-equivalence-fixture.v1"
DEFAULT_FIXTURE = Path("experiments/TRR-0009/evaluation/trr8_loader_equivalence_fixture.json")
DEFAULT_METHOD_FREEZE = Path("experiments/TRR-0009/training/method_freeze.json")
DEFAULT_OUTPUT = Path("experiments/TRR-0009/evaluation/loader_qualification_v1/qualification.json")
DEFAULT_FAILURE = Path("experiments/TRR-0009/evaluation/loader_qualification_v1/failure.json")


class QualificationError(contract.ContractError):
    """Raised when the bounded loader/output qualification fails closed."""


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise QualificationError(f"repository root unavailable: {root}")
    return root


def _task_path(value: Path, *, root: Path, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    task_root = (root / "experiments" / contract.TASK_ID).resolve()
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise QualificationError(f"{description} must be below {task_root}: {path}") from exc
    if path.is_symlink():
        raise QualificationError(f"{description} is a symlink: {path}")
    return path


def _file_record(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{description} unavailable: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": contract.sha256_file(path),
    }


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QualificationError("cannot resolve qualifier commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise QualificationError("qualifier commit is not a full hash")
    return value


def _working_tree_status(root: Path) -> list[str]:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QualificationError("cannot record working-tree status") from exc
    return [line for line in value.splitlines() if line]


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        return contract.load_json(path, description=description)
    except Exception as exc:
        raise QualificationError(str(exc)) from exc


def _require_false(value: Mapping[str, Any], keys: tuple[str, ...], *, description: str) -> None:
    if any(value.get(key) is True for key in keys):
        raise QualificationError(f"{description} records forbidden truth/source access")


def _check_record_against_descriptor(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    description: str,
) -> dict[str, Any]:
    try:
        return contract.validate_file_record(descriptor, repository_root=root, description=description, verify=True)
    except Exception as exc:
        raise QualificationError(str(exc)) from exc


def _fixture_and_bindings(
    fixture_path: Path,
    method_freeze_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixture = _load_json(fixture_path, description="TRR-0008 loader fixture")
    if fixture.get("schema") != FIXTURE_SCHEMA or fixture.get("task_id") != contract.TASK_ID:
        raise QualificationError("fixture schema or task identity changed")
    _require_false(
        fixture,
        ("truth_opened", "target_labels_loaded", "source_text_loaded", "source_text_written", "token_ids_written"),
        description="fixture",
    )
    if fixture.get("cell_order") != list(contract.CELL_ORDER):
        raise QualificationError("fixture cell order differs from the frozen evaluator contract")
    records_to_check = int(fixture.get("records_to_check_per_cell", 0))
    if records_to_check <= 0 or records_to_check > 8:
        raise QualificationError("fixture must check between one and eight rows per cell")
    _check_record_against_descriptor(fixture["embedding"], root=root, description="fixture embedding")
    for name in ("observation_manifest", "registration", "run_manifest"):
        _check_record_against_descriptor(fixture[name], root=root, description=f"fixture {name}")
    methods = fixture.get("methods")
    if not isinstance(methods, Mapping):
        raise QualificationError("fixture method bindings are absent")
    for method_id in (contract.UNCHANGED_METHOD_ID, contract.PUBLISHED_REFERENCE_METHOD_ID):
        row = methods.get(method_id)
        if not isinstance(row, Mapping):
            raise QualificationError(f"fixture archived method binding is absent: {method_id}")
        _check_record_against_descriptor(row["state"], root=root, description=f"fixture {method_id} state")
        loader = row.get("loader")
        if not isinstance(loader, Mapping) or loader.get("interface") != contract.LOADER_INTERFACE:
            raise QualificationError(f"fixture loader interface changed: {method_id}")
        predictions = row.get("predictions")
        if not isinstance(predictions, Mapping) or set(predictions) != set(contract.CELL_ORDER):
            raise QualificationError(f"fixture archived prediction coverage changed: {method_id}")
        for cell_id in contract.CELL_ORDER:
            pred = predictions[cell_id]
            if not isinstance(pred, Mapping):
                raise QualificationError(f"fixture prediction binding malformed: {method_id}/{cell_id}")
            _check_record_against_descriptor(pred, root=root, description=f"fixture {method_id}/{cell_id} prediction")
            _check_record_against_descriptor(pred["observation"], root=root, description=f"fixture {cell_id} observation")
            if int(pred.get("records", 0)) < records_to_check:
                raise QualificationError(f"fixture prediction has too few rows: {method_id}/{cell_id}")
    method_freeze = _load_json(method_freeze_path, description="TRR-0009 method freeze")
    if method_freeze.get("task_id") != contract.TASK_ID or method_freeze.get("state_selection_frozen") is not True:
        raise QualificationError("TRR-0009 method freeze is not frozen")
    if method_freeze.get("source_selection_started") is not False or method_freeze.get("truth_opened") is not False:
        raise QualificationError("TRR-0009 method freeze is outside the pre-source truth-free boundary")
    state_bindings = method_freeze.get("state_bindings")
    if not isinstance(state_bindings, Mapping):
        raise QualificationError("TRR-0009 method freeze state bindings are absent")
    for method_id in (contract.FIXED_METHOD_ID, contract.ADAPTABLE_METHOD_ID):
        row = state_bindings.get(method_id)
        if not isinstance(row, Mapping):
            raise QualificationError(f"TRR-0009 selected state binding is absent: {method_id}")
        _check_record_against_descriptor(row["state"], root=root, description=f"TRR-0009 {method_id} state")
        loader = row.get("loader")
        if not isinstance(loader, Mapping) or loader.get("interface") != contract.LOADER_INTERFACE:
            raise QualificationError(f"TRR-0009 loader interface changed: {method_id}")
        for argument_group in ("path_args", "tensor_args"):
            for arg_name, binding in loader.get(argument_group, {}).items():
                _check_record_against_descriptor(binding, root=root, description=f"TRR-0009 {method_id} {argument_group}.{arg_name}")
    return fixture, method_freeze, {
        "fixture": _file_record(fixture_path, description="fixture descriptor"),
        "method_freeze": _file_record(method_freeze_path, description="TRR-0009 method freeze"),
    }


def _read_fixture_row(
    observation: Mapping[str, Any],
    *,
    records_to_check: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(str(observation["path"])).expanduser().resolve()
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise QualificationError(f"fixture observation keys changed: {path}")
            activations = handle.get_slice("activations")[:records_to_check]
            mask = handle.get_slice("attention_mask")[:records_to_check]
            positions = handle.get_slice("position_ids")[:records_to_check]
    except QualificationError:
        raise
    except Exception as exc:
        raise QualificationError(f"fixture observation could not be opened: {path}") from exc
    if tuple(activations.shape) != (records_to_check, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE):
        raise QualificationError(f"fixture activation shape changed: {path}")
    if tuple(mask.shape) != (records_to_check, contract.STORED_SEQUENCE_TOKENS) or tuple(positions.shape) != tuple(mask.shape):
        raise QualificationError(f"fixture sidecar shape changed: {path}")
    if activations.dtype != torch.bfloat16 or mask.dtype not in (torch.bool, torch.uint8) or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise QualificationError(f"fixture observation dtype changed: {path}")
    mask_bool = mask.to(dtype=torch.bool).contiguous()
    positions_long = positions.to(dtype=torch.long).contiguous()
    expected_positions = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.long).unsqueeze(0).expand_as(positions_long)
    if not torch.isfinite(activations.float()).all().item() or not mask_bool.all().item() or not torch.equal(positions_long, expected_positions):
        raise QualificationError(f"fixture observation values changed: {path}")
    return activations.contiguous(), mask_bool, positions_long


def _read_archived_rows(
    prediction: Mapping[str, Any],
    *,
    root: Path,
    records_to_check: int,
) -> tuple[torch.Tensor, str]:
    path = Path(_check_record_against_descriptor(prediction, root=root, description="archived prediction")["path"])
    try:
        value, _metadata = contract.load_prediction_file(path, records=int(prediction["records"]))
    except Exception as exc:
        raise QualificationError(f"archived prediction could not be loaded: {path}") from exc
    rows = value[:records_to_check].contiguous()
    expected_digest = prediction.get("prediction_sha256")
    if not isinstance(expected_digest, str):
        raise QualificationError(f"archived prediction digest is absent: {path}")
    # The descriptor digest covers the complete archived tensor.  This avoids
    # silently validating a different prefix of a changed file.
    if contract.tensor_digest(value) != expected_digest:
        raise QualificationError(f"archived prediction tensor digest changed: {path}")
    return rows, contract.tensor_digest(rows)


def _prediction_from_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, contract.STORED_SEQUENCE_TOKENS):
        raise QualificationError(f"full forward geometry changed: {tuple(logits.shape)}")
    if tuple(logits.shape[2:]) != (contract.VOCABULARY_SIZE,) or not torch.isfinite(logits).all().item():
        raise QualificationError("full forward vocabulary geometry or finiteness changed")
    raw = torch.full((contract.STORED_SEQUENCE_TOKENS,), contract.INVALID_TOKEN_ID, dtype=torch.long, device=logits.device)
    raw[0] = contract.BOS_TOKEN_ID
    raw[1:] = logits[0, 1:].argmax(dim=-1).to(dtype=torch.long)
    return contract.normalize_prediction(raw.cpu(), valid_mask)


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    value: dict[str, Any] = {
        "host_available_bytes": runner._host_available_bytes(),
        "process_peak_rss_bytes": runner._rss_bytes(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        value.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_free_bytes": int(torch.cuda.mem_get_info(device)[0]),
            }
        )
    else:
        value.update(
            {
                "cuda_peak_allocated_bytes": None,
                "cuda_peak_reserved_bytes": None,
                "cuda_reserved_bytes": None,
                "cuda_free_bytes": None,
            }
        )
    return value


def _runtime_record(root: Path) -> dict[str, Any]:
    return {
        "argv": list(sys.argv),
        "python": sys.executable,
        "cwd": str(Path.cwd().resolve()),
        "repository_root": str(root),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
            if key in os.environ
        },
    }


def _state_row(method_id: str, method_freeze: Mapping[str, Any]) -> dict[str, Any]:
    if method_id in (contract.FIXED_METHOD_ID, contract.ADAPTABLE_METHOD_ID):
        source = method_freeze["state_bindings"][method_id]
        return {"id": method_id, "state": source["state"], "loader": source["loader"]}
    raise QualificationError(f"cannot construct TRR-0009 state row: {method_id}")


def _archived_row(method_id: str, fixture: Mapping[str, Any], cell_id: str) -> Mapping[str, Any] | None:
    if method_id not in (contract.UNCHANGED_METHOD_ID, contract.PUBLISHED_REFERENCE_METHOD_ID):
        return None
    return fixture["methods"][method_id]["predictions"][cell_id]


def qualify(
    *,
    repository_root: Path,
    fixture_path: Path = DEFAULT_FIXTURE,
    method_freeze_path: Path = DEFAULT_METHOD_FREEZE,
    output_path: Path = DEFAULT_OUTPUT,
    failure_path: Path = DEFAULT_FAILURE,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Run the bounded B=1 loader/output qualification."""

    root = _root(repository_root)
    fixture_path = Path(fixture_path).expanduser()
    if not fixture_path.is_absolute():
        fixture_path = root / fixture_path
    fixture_path = fixture_path.resolve()
    method_freeze_path = Path(method_freeze_path).expanduser()
    if not method_freeze_path.is_absolute():
        method_freeze_path = root / method_freeze_path
    method_freeze_path = method_freeze_path.resolve()
    output = _task_path(output_path, root=root, description="qualification receipt")
    failure = _task_path(failure_path, root=root, description="qualification failure receipt")
    started_utc = runner._utc_now()
    started = time.perf_counter()
    device = torch.device(device_name)
    initial_head: str | None = None
    stage = "startup"
    resource_before: dict[str, Any] | None = None
    code_bindings: dict[str, dict[str, Any]] = {}
    try:
        initial_head = _git_head(root)
        if output.exists() or output.is_symlink():
            raise QualificationError(f"qualification receipt is create-only: {output}")
        if failure.exists() or failure.is_symlink():
            raise QualificationError(f"qualification failure receipt is create-only: {failure}")
        fixture, method_freeze, binding_records = _fixture_and_bindings(fixture_path, method_freeze_path, root=root)
        records_to_check = int(fixture["records_to_check_per_cell"])
        code_bindings = {
            "qualifier": _file_record(Path(__file__), description="qualifier source"),
            "runner": _file_record(Path(runner.__file__), description="TRR-0009 evaluator runner"),
            "model": _file_record(root / "scripts/trr0009_model.py", description="TRR-0009 model"),
            "contract": _file_record(root / "scripts/trr0009_eval_contract.py", description="TRR-0009 evaluator contract"),
        }
        numerics = runner._configure_numerics(contract.NUMERICAL_SETTINGS)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise QualificationError("CUDA is unavailable")
        resource_before = _resource_snapshot(device)
        runner._guard(device=device, guard=contract.RESOURCE_GUARD, started=started, stage="before_embedding")
        embedding, embedding_evidence = runner._load_embedding({"runtime_embedding": fixture["embedding"]}, root=root, device=device)
        runner._guard(device=device, guard=contract.RESOURCE_GUARD, started=started, stage="after_embedding")
        method_rows: dict[str, Mapping[str, Any]] = {}
        method_rows[contract.UNCHANGED_METHOD_ID] = dict(fixture["methods"][contract.UNCHANGED_METHOD_ID]) | {"id": contract.UNCHANGED_METHOD_ID}
        method_rows[contract.PUBLISHED_REFERENCE_METHOD_ID] = dict(fixture["methods"][contract.PUBLISHED_REFERENCE_METHOD_ID]) | {"id": contract.PUBLISHED_REFERENCE_METHOD_ID}
        method_rows[contract.FIXED_METHOD_ID] = _state_row(contract.FIXED_METHOD_ID, method_freeze)
        method_rows[contract.ADAPTABLE_METHOD_ID] = _state_row(contract.ADAPTABLE_METHOD_ID, method_freeze)
        method_results: dict[str, Any] = {}
        resource_by_method: dict[str, Any] = {}
        for method_id in contract.METHOD_ORDER:
            stage = f"load_{method_id}"
            runner._guard(device=device, guard=contract.RESOURCE_GUARD, started=started, stage=stage)
            load_started = time.perf_counter()
            model, state_evidence = runner._load_decoder(method_rows[method_id], root=root, device=device)
            load_seconds = float(time.perf_counter() - load_started)
            method_results[method_id] = {
                "loader": state_evidence,
                "model_preparation_seconds": load_seconds,
                "cells": {},
            }
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for cell_id in contract.CELL_ORDER:
                stage = f"{method_id}/{cell_id}"
                runner._guard(device=device, guard=contract.RESOURCE_GUARD, started=started, stage=f"before_{stage}")
                archived_binding = _archived_row(method_id, fixture, cell_id)
                observation = fixture["methods"][contract.UNCHANGED_METHOD_ID]["predictions"][cell_id]["observation"]
                # All archived method rows bind the same observation bytes; use
                # the control binding and reject any silent divergence.
                if archived_binding is not None and observation["sha256"] != archived_binding["observation"]["sha256"]:
                    raise QualificationError(f"method observation binding differs: {cell_id}")
                activations, valid_mask, positions = _read_fixture_row(observation, records_to_check=records_to_check)
                archived_rows: torch.Tensor | None = None
                archived_digest: str | None = None
                if archived_binding is not None:
                    archived_rows, archived_digest = _read_archived_rows(
                        archived_binding,
                        root=root,
                        records_to_check=records_to_check,
                    )
                row_results: list[dict[str, Any]] = []
                for row_index in range(records_to_check):
                    runner._guard(device=device, guard=contract.RESOURCE_GUARD, started=started, stage=f"{stage}/{row_index}")
                    activation = activations[row_index]
                    mask = valid_mask[row_index]
                    runner_prediction = runner.predict_current_h(model, embedding, activation, mask, device=device)
                    staged = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
                    staged_mask = mask.to(device=device, dtype=torch.bool).unsqueeze(0)
                    with torch.inference_mode():
                        full_logits = model(staged, staged_mask, embedding)
                    direct_prediction = _prediction_from_logits(full_logits, mask)
                    if not torch.equal(runner_prediction, direct_prediction):
                        raise QualificationError(f"current-H/full-forward ID mismatch: {method_id}/{cell_id}/{row_index}")
                    archived_match: bool | None = None
                    if archived_binding is not None:
                        if archived_rows is None:
                            raise QualificationError(f"archived rows missing: {method_id}/{cell_id}")
                        expected = archived_rows[row_index]
                        archived_match = bool(torch.equal(runner_prediction, expected))
                        if not archived_match:
                            mismatch = torch.nonzero(runner_prediction.ne(expected), as_tuple=False)
                            first = mismatch[0].tolist() if int(mismatch.shape[0]) else []
                            raise QualificationError(f"archived ID mismatch: {method_id}/{cell_id}/{row_index}/{first}")
                    row_results.append(
                        {
                            "row_index": row_index,
                            "runner_prediction_sha256": contract.tensor_digest(runner_prediction),
                            "full_forward_prediction_sha256": contract.tensor_digest(direct_prediction),
                            "archived_prediction_sha256": None if archived_binding is None else contract.tensor_digest(archived_rows[row_index]),
                            "archived_prefix_sha256": archived_digest,
                            "runner_equals_full_forward": True,
                            "archived_exact_match": archived_match,
                            "full_forward_shape": list(full_logits.shape),
                            "full_forward_dtype": str(full_logits.dtype),
                            "full_forward_finite": True,
                        }
                    )
                    del full_logits, staged, staged_mask, runner_prediction, direct_prediction
                    gc.collect()
                method_results[method_id]["cells"][cell_id] = {
                    "records_checked": records_to_check,
                    "rows": row_results,
                    "archived_exact_match": None if archived_binding is None else True,
                }
                del activations, valid_mask, positions
                gc.collect()
            resource_by_method[method_id] = _resource_snapshot(device)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        final_head = _git_head(root)
        if final_head != initial_head:
            raise QualificationError(f"git HEAD changed during qualification: {initial_head} -> {final_head}")
        final_code_bindings = {
            "qualifier": _file_record(Path(__file__), description="qualifier source"),
            "runner": _file_record(Path(runner.__file__), description="TRR-0009 evaluator runner"),
            "model": _file_record(root / "scripts/trr0009_model.py", description="TRR-0009 model"),
            "contract": _file_record(root / "scripts/trr0009_eval_contract.py", description="TRR-0009 evaluator contract"),
        }
        if final_code_bindings != code_bindings:
            raise QualificationError("bound qualifier source changed during execution")
        receipt = {
            "schema": QUALIFIER_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "TRR8_PUBLIC_FIXTURE_LOADER_EQUIVALENCE_PASS",
            "started_utc": started_utc,
            "ended_utc": runner._utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "device": str(device),
            "runtime": _runtime_record(root),
            "code_commit": initial_head,
            "working_tree_status_at_start": _working_tree_status(root),
            "working_tree_status_at_end": _working_tree_status(root),
            "numerical_settings": numerics,
            "resource_guard": dict(contract.RESOURCE_GUARD),
            "code_bindings": code_bindings,
            "code_bindings_final": final_code_bindings,
            "resource_before": resource_before,
            "resource_after": _resource_snapshot(device),
            "resource_by_method": resource_by_method,
            "fixture": binding_records["fixture"],
            "method_freeze": binding_records["method_freeze"],
            "embedding": embedding_evidence,
            "records_per_cell": int(fixture["records_to_check_per_cell"]),
            "cell_order": list(contract.CELL_ORDER),
            "method_order": list(contract.METHOD_ORDER),
            "methods": method_results,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "truth_opened": False,
            "fresh_source_selection": False,
            "candidate_arrays_persisted": False,
        }
        if output.exists() or output.is_symlink():
            raise QualificationError(f"qualification receipt is create-only: {output}")
        contract.write_create_only(output, receipt)
        return receipt
    except BaseException as exc:
        failure_payload = {
            "schema": FAILURE_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "TRR8_PUBLIC_FIXTURE_LOADER_EQUIVALENCE_FAILED_CLOSED",
            "started_utc": started_utc,
            "ended_utc": runner._utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "stage": stage,
            "device": str(device),
            "runtime": _runtime_record(root),
            "code_commit": initial_head,
            "code_bindings": code_bindings,
            "resource_snapshot": _resource_snapshot(device),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "truth_opened": False,
            "target_labels_loaded": False,
            "source_text_loaded": False,
            "candidate_arrays_persisted": False,
        }
        try:
            if not failure.exists() and not failure.is_symlink():
                contract.write_create_only(failure, failure_payload)
        except Exception:
            pass
        if isinstance(exc, QualificationError):
            raise
        raise QualificationError("loader equivalence qualification failed closed") from exc
    finally:
        gc.collect()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--method-freeze", type=Path, default=DEFAULT_METHOD_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failure", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        qualify(
            repository_root=args.repository_root,
            fixture_path=args.fixture,
            method_freeze_path=args.method_freeze,
            output_path=args.output,
            failure_path=args.failure,
            device_name=args.device,
        )
    except QualificationError as exc:
        print(f"loader qualification failed closed: {exc}", flush=True)
        return 2
    print("loader qualification PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
