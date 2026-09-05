#!/usr/bin/env python3
"""Task-local TRR-0005 prediction and timing contract.

The actual model runner owns device setup and state loading.  This module
contains the small shared wrapper it must use for every method: one warmup
call, one measured call per source record, and an exact predicted-ID equality
check between those calls.  It also builds the source-free prediction receipt
consumed by the freeze adapter.  It deliberately does not select a holdout,
capture public activations, open truth, or run a model at import time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from safetensors.torch import save_file
import torch

from token_reconstruction.trr0005_contract import (
    BOS_TOKEN_ID,
    CANDIDATE_POLICIES,
    ContractError,
    EXPECTED_CELL_IDS,
    INVALID_TOKEN_ID,
    METHOD_IDS,
    METHOD_SPEC_BY_ID,
    PREDICTION_SCHEMA,
    RECORDS_PER_DOMAIN,
    SEQUENCE_TOKENS,
    TASK_ID,
    TIMING_CONTRACT,
    distribution_state_id,
    validate_prediction_descriptor,
)


SCHEMA = "token-reconstruction.trr0005-prediction-receipt.v1"


class PredictionError(ContractError):
    """Raised when prediction timing or output metadata violates TRR5."""


def canonical_state_path(fit_root: Path, *, distribution: str, method_id: str) -> Path:
    """Return the selected-state path emitted by the parallel fit runner."""

    distribution_aliases = {
        "original": "original",
        "enriched": "enriched",
        "original_like_alpaca_v1": "original",
        "coverage_mix_v1": "enriched",
    }
    if distribution not in distribution_aliases:
        raise PredictionError(f"unknown TRR-0005 fitting distribution: {distribution}")
    distribution_directory = distribution_aliases[distribution]
    if method_id not in {
        "joint_full_affine",
        "affine_causal_h_attention128",
        "affine_trained_diagonal_attention128",
    }:
        raise PredictionError(f"state path requires one of the six fitted method IDs: {method_id}")
    return Path(fit_root) / distribution_directory / method_id / "selected.safetensors"


def _ids(value: Any, *, description: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.long).contiguous().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PredictionError(f"{description} is not integer-like") from exc
    if tensor.ndim != 1 or tensor.shape[0] != SEQUENCE_TOKENS:
        raise PredictionError(f"{description} must have shape [{SEQUENCE_TOKENS}]")
    if tensor[0].item() != BOS_TOKEN_ID:
        raise PredictionError(f"{description} does not retain BOS")
    active = tensor.ne(INVALID_TOKEN_ID)
    invalid_positions = (~active).nonzero(as_tuple=False).flatten()
    if invalid_positions.numel() and active[invalid_positions[0] + 1 :].any().item():
        raise PredictionError(f"{description} has a non-contiguous padding suffix")
    # -1 is allowed only in the right-padded suffix; active post-BOS IDs must
    # be valid vocabulary IDs.
    active_post_bos = active.clone()
    active_post_bos[0] = False
    if tensor[active_post_bos].lt(0).any().item() or tensor[active_post_bos].ge(128256).any().item():
        raise PredictionError(f"{description} has an invalid active token ID")
    return tensor


def run_warmed_prediction(
    *,
    method_id: str,
    records: Sequence[Any],
    predict_one: Callable[[Any], Any],
    warmup_runs_per_record: int = 1,
    measured_runs_per_record: int = 1,
    synchronize: Callable[[], None] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the fixed 1+1 contract and require warmup/measured ID identity."""

    if method_id not in METHOD_IDS:
        raise PredictionError(f"unknown TRR-0005 method: {method_id}")
    if len(records) != RECORDS_PER_DOMAIN:
        raise PredictionError(f"prediction cell needs {RECORDS_PER_DOMAIN} records")
    if warmup_runs_per_record != 1 or measured_runs_per_record != 1:
        raise PredictionError("TRR-0005 requires exactly one warmup and one measured call")
    synchronize = synchronize or (lambda: None)
    outputs: list[torch.Tensor] = []
    warmup_seconds = 0.0
    measured_seconds = 0.0
    per_record_measured: list[float] = []
    for index, record in enumerate(records):
        started = time.perf_counter()
        warmup = _ids(predict_one(record), description=f"warmup prediction {index}")
        synchronize()
        warmup_seconds += time.perf_counter() - started
        started = time.perf_counter()
        measured = _ids(predict_one(record), description=f"measured prediction {index}")
        synchronize()
        elapsed = time.perf_counter() - started
        measured_seconds += elapsed
        per_record_measured.append(elapsed)
        if not torch.equal(warmup, measured):
            raise PredictionError(f"warmup/measured predicted IDs differ for record {index}")
        outputs.append(measured)
    predictions = torch.stack(outputs, dim=0)
    return predictions, {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "method_id": method_id,
        "records": len(records),
        "warmup_runs_per_record": 1,
        "measured_runs_per_record": 1,
        "warmup_seconds_sum": warmup_seconds,
        "measured_seconds_sum": measured_seconds,
        "per_record_measured_seconds": per_record_measured,
        "timed_interval_total_seconds": warmup_seconds + measured_seconds,
        "warmup_output_exact_match_measured": True,
        "measured_output_selected": True,
    }


def prediction_descriptor(
    *,
    cell_id: str,
    method_id: str,
    predictions: torch.Tensor,
    timing: Mapping[str, Any],
    panel_sha256: str | None = None,
    selection_plan_sha256: str | None = None,
    observation_sha256: str | None = None,
    candidate_budget: int | None = None,
    public_prefix_calls: int = 0,
    candidate_simulations: int = 0,
) -> dict[str, Any]:
    """Build and validate the source-free descriptor used by the gate."""

    if cell_id not in EXPECTED_CELL_IDS or method_id not in METHOD_IDS:
        raise PredictionError(f"unknown prediction binding: {cell_id}/{method_id}")
    tensor = torch.as_tensor(predictions, dtype=torch.long).contiguous().cpu()
    if tuple(tensor.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
        raise PredictionError("prediction tensor geometry changed")
    policy = CANDIDATE_POLICIES[method_id]
    descriptor: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "task_id": TASK_ID,
        "cell_id": cell_id,
        "method_id": method_id,
        "canonical_method_id": method_id,
        "shape": list(tensor.shape),
        "candidate_policy": policy,
        "candidate_arrays_present": False,
        "warmup_runs_per_record": timing.get("warmup_runs_per_record"),
        "measured_runs_per_record": timing.get("measured_runs_per_record"),
        "warmup_output_exact_match_measured": timing.get("warmup_output_exact_match_measured"),
        "measured_output_selected": timing.get("measured_output_selected"),
        "candidate_budget": candidate_budget,
        "candidate_output": "omitted_after_decision" if policy == "output_only" else None,
        "public_prefix_calls": int(public_prefix_calls),
        "candidate_simulations": int(candidate_simulations),
        "panel_sha256": panel_sha256,
        "selection_plan_sha256": selection_plan_sha256,
        "observation_sha256": observation_sha256,
        # The tensor is kept in-memory for the scorer; JSON receipts omit it.
        "predictions": tensor,
    }
    validate_prediction_descriptor(
        descriptor,
        cell_id=cell_id,
        method_id=method_id,
    )
    return descriptor



def write_prediction_artifact(
    path: Path,
    *,
    cell_id: str,
    method_id: str,
    predictions: torch.Tensor,
    binding: Mapping[str, Any],
    panel_sha256: str,
    selection_plan_sha256: str,
    observation_sha256: str,
    repository_root: Path | None = None,
    hidden_size: int = 2048,
    cut_depth: int = 4,
) -> dict[str, Any]:
    """Serialize one TRR5 prediction using the TRR4 artifact contract.

    Every method uses the same compact ``predictions`` tensor.  In particular,
    the A2 output-only port does not write candidate arrays.  The returned
    record is ready to attach to the JSON timing receipt and is rehashed again
    by the scorer's pretruth gate.
    """

    if cell_id not in EXPECTED_CELL_IDS or method_id not in METHOD_IDS:
        raise PredictionError(f"unknown prediction binding: {cell_id}/{method_id}")
    if not isinstance(binding, Mapping):
        raise PredictionError("prediction binding is absent")
    for name, value in (
        ("panel_sha256", panel_sha256),
        ("selection_plan_sha256", selection_plan_sha256),
        ("observation_sha256", observation_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise PredictionError(f"{name} must be a lowercase SHA-256 digest")
    tensor = torch.as_tensor(predictions).contiguous().cpu()
    if tuple(tensor.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
        raise PredictionError("prediction artifact geometry changed")
    if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise PredictionError("prediction artifact IDs must be integer")
    if path.exists() or path.is_symlink():
        raise PredictionError(f"prediction artifact is create-only: {path}")
    metadata = {
        "schema": PREDICTION_SCHEMA,
        "task_id": TASK_ID,
        "panel_sha256": panel_sha256,
        "selection_plan_sha256": selection_plan_sha256,
        "observation_sha256": observation_sha256,
        "cell_id": cell_id,
        "style": cell_id.split("__", 1)[0],
        "condition": cell_id.split("__", 1)[1],
        "method_id": method_id,
        "geometry_json": json.dumps(
            {
                "records": RECORDS_PER_DOMAIN,
                "sequence_tokens": SEQUENCE_TOKENS,
                "hidden_size": int(hidden_size),
                "cut_depth": int(cut_depth),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "binding_json": json.dumps(dict(binding), sort_keys=True, separators=(",", ":")),
        "candidate_policy": CANDIDATE_POLICIES[method_id],
        "candidate_output": "omitted_after_decision"
        if CANDIDATE_POLICIES[method_id] == "output_only"
        else "forbidden",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"predictions": tensor.to(dtype=torch.int64)}, str(path), metadata=metadata)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if repository_root is not None:
        root = repository_root.expanduser().resolve()
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise PredictionError("prediction artifact escaped repository root") from exc
        record_path = relative
    else:
        record_path = str(path.resolve())
    return {
        "path": record_path,
        "bytes": int(path.stat().st_size),
        "sha256": digest,
    }


def write_prediction_receipt(path: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Write a compact create-only receipt without serializing predictions."""

    if path.exists() or path.is_symlink():
        raise PredictionError(f"prediction receipt is create-only: {path}")
    value = {
        key: item
        for key, item in descriptor.items()
        if key != "predictions"
    }
    value["prediction_sha256"] = _tensor_sha256(torch.as_tensor(descriptor["predictions"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}, sort_keys=True).encode("utf-8"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "PredictionError",
    "SCHEMA",
    "canonical_state_path",
    "prediction_descriptor",
    "run_warmed_prediction",
    "write_prediction_artifact",
    "write_prediction_receipt",
]
