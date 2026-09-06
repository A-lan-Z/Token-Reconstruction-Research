#!/usr/bin/env python3
"""Run the frozen TRR-0006 enriched causal/diagonal prediction pair.

The runner is intentionally task-local because TRR-0005's public runner binds
128 records and four methods.  It accepts one already-frozen TRR-0006
registration, validates the producer's source-free observation manifest, and
streams one fixed chunk of observations at a time.  Each record is predicted
with one warmup and one measured call; IDs must match exactly before an output
is selected.  Truth, source text, target labels, candidate arrays, and model
training are outside this program.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Iterator

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts import trr0006_prediction_contract as contract


class RunnerError(contract.ContractError):
    """Raised when a frozen prediction run cannot proceed safely."""


class _Chunk:
    def __init__(self, start: int, stop: int, activations: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> None:
        self.start = start
        self.stop = stop
        self.activations = activations
        self.mask = mask
        self.positions = positions



def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repository_root(value: Path) -> Path:
    root = value.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RunnerError(f"repository root is unavailable: {root}")
    return root


def _resolve(value: str, root: Path, *, description: str) -> Path:
    try:
        return contract.resolve_path(value, repository_root=root, description=description)
    except contract.ContractError as exc:
        raise RunnerError(str(exc)) from exc


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError("cannot resolve executable source commit") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise RunnerError("executable source commit is not a full lowercase hash")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": contract.sha256_file(path)}


def _configure_numerics(settings: dict[str, Any]) -> dict[str, Any]:
    """Pin the fixture-matched numerical recipe before any CUDA/model work."""

    try:
        expected = contract.validate_numerical_settings(settings)
        torch.set_num_threads(int(expected["cpu_intraop_threads"]))
        torch.set_num_interop_threads(int(expected["cpu_interop_threads"]))
        torch.backends.cuda.matmul.allow_tf32 = bool(expected["cuda_matmul_allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(expected["cuda_cudnn_allow_tf32"])
        torch.set_float32_matmul_precision(str(expected["float32_matmul_precision"]))
    except (contract.ContractError, AttributeError, RuntimeError, ValueError) as exc:
        raise RunnerError("cannot configure the registered numerical recipe") from exc
    observed = {
        "activation_input_dtype": expected["activation_input_dtype"],
        "staged_activation_dtype": expected["staged_activation_dtype"],
        "staged_mask_dtype": expected["staged_mask_dtype"],
        "decoder_compute_dtype": expected["decoder_compute_dtype"],
        "embedding_dtype": expected["embedding_dtype"],
        "autocast": bool(expected["autocast"]),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_intraop_threads": int(torch.get_num_threads()),
        "cpu_interop_threads": int(torch.get_num_interop_threads()),
    }
    if observed != expected:
        raise RunnerError(f"configured numerical recipe differs from registration: {observed}")
    return observed


def _verify_code_bindings(registration: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    head = _git_head(root)
    if head != registration["code_commit"]:
        raise RunnerError(
            f"registration code commit differs from executable HEAD: {registration['code_commit']} != {head}"
        )
    result: list[dict[str, Any]] = []
    for index, binding in enumerate(registration["code_bindings"]):
        try:
            record = contract.validate_file_record(
                binding,
                repository_root=root,
                description=f"code binding {index}",
                verify=True,
            )
        except contract.ContractError as exc:
            raise RunnerError(str(exc)) from exc
        record["role"] = binding["role"]
        result.append(record)
    return result


def _guard(
    *,
    device: torch.device,
    guard: dict[str, Any],
    started: float,
    stage: str,
) -> dict[str, Any]:
    max_seconds = float(guard.get("maximum_seconds", 1800.0))
    elapsed = time.perf_counter() - started
    if elapsed > max_seconds:
        raise RunnerError(f"wall-time guard expired at {stage}: {elapsed:.3f}s")
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if rss > int(guard["maximum_rss_bytes"]):
        raise RunnerError(f"RSS guard failed at {stage}: {rss} > {guard['maximum_rss_bytes']}")
    available = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                available = int(fields[1]) * 1024
                break
    except (OSError, UnicodeError):
        pass
    if available is None:
        raise RunnerError(f"host available-memory guard unavailable at {stage}")
    if available < int(guard["minimum_host_available_bytes"]):
        raise RunnerError(
            f"host available-memory guard failed at {stage}: {available} < {guard['minimum_host_available_bytes']}"
        )
    result: dict[str, Any] = {
        "stage": stage,
        "elapsed_seconds": float(elapsed),
        "process_max_rss_bytes": rss,
        "host_available_bytes": available,
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        reserved = int(torch.cuda.memory_reserved(device))
        allocated = int(torch.cuda.memory_allocated(device))
        if int(free) < int(guard["minimum_free_gpu_bytes"]):
            raise RunnerError(f"GPU free-memory guard failed at {stage}: {free}")
        if reserved > int(guard["maximum_reserved_gpu_bytes"]):
            raise RunnerError(f"GPU reserved-memory guard failed at {stage}: {reserved}")
        result["gpu"] = {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "reserved_bytes": reserved,
            "allocated_bytes": allocated,
        }
    return result


def _load_embedding(registration: dict[str, Any], root: Path, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    binding = registration["runtime_assets"]["normalized_public_E"]
    try:
        record = contract.validate_file_record(
            binding,
            repository_root=root,
            description="normalized public E",
            verify=True,
        )
    except contract.ContractError as exc:
        raise RunnerError(str(exc)) from exc
    started = time.perf_counter()
    try:
        cpu_table = load_file(record["path"], device="cpu")
        if set(cpu_table) != {"embeddings"}:
            raise RunnerError("normalized public E must contain only embeddings")
        table_cpu = cpu_table["embeddings"].contiguous()
        if table_cpu.dtype != torch.float32 or tuple(table_cpu.shape) != (contract.VOCAB_SIZE, contract.HIDDEN_SIZE):
            raise RunnerError("normalized public E geometry or dtype changed")
        table = table_cpu.to(device=device).contiguous()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError("normalized public E load failed") from exc
    finally:
        if "cpu_table" in locals():
            del cpu_table
        if "table_cpu" in locals():
            del table_cpu
        gc.collect()
    return table, {
        "path": record["path"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "dtype": "torch.float32",
        "shape": [contract.VOCAB_SIZE, contract.HIDDEN_SIZE],
        "load_seconds": float(time.perf_counter() - started),
    }


def _iter_observation_chunks(
    cell: dict[str, Any],
    *,
    records: int,
    chunk_records: int,
) -> Iterator[_Chunk]:
    observation = cell["observation"]
    path = Path(observation["path"])
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"activations", "attention_mask", "position_ids"}
            if keys != required:
                raise RunnerError(f"observation tensor keys changed for {cell['cell_id']}: {sorted(keys)}")
            activations_slice = handle.get_slice("activations")
            mask_slice = handle.get_slice("attention_mask")
            positions_slice = handle.get_slice("position_ids")
            expected_h = (records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE)
            expected_side = (records, contract.STORED_SEQUENCE_TOKENS)
            if tuple(activations_slice.get_shape()) != expected_h:
                raise RunnerError(f"observation activation geometry changed for {cell['cell_id']}")
            if tuple(mask_slice.get_shape()) != expected_side or tuple(positions_slice.get_shape()) != expected_side:
                raise RunnerError(f"observation sidecar geometry changed for {cell['cell_id']}")
            for start in range(0, records, chunk_records):
                stop = min(start + chunk_records, records)
                activations = activations_slice[start:stop]
                mask = mask_slice[start:stop]
                positions = positions_slice[start:stop]
                if activations.dtype != torch.bfloat16 or mask.dtype not in (torch.bool, torch.uint8) or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                    raise RunnerError(f"observation tensor dtypes changed for {cell['cell_id']}")
                if not torch.isfinite(activations.float()).all().item():
                    raise RunnerError(f"observation contains non-finite activation for {cell['cell_id']} rows {start}:{stop}")
                if mask.dtype == torch.uint8 and ((mask != 0) & (mask != 1)).any().item():
                    raise RunnerError(f"observation mask is not binary for {cell['cell_id']} rows {start}:{stop}")
                valid = mask.to(torch.bool)
                # The registered estimand is a full 128-position clip: BOS
                # plus exactly 127 post-BOS positions.  A shorter right-padded
                # row would silently change the denominator and is rejected.
                if not valid.all().item():
                    raise RunnerError(f"observation clip is not fully valid for {cell['cell_id']} rows {start}:{stop}")
                expected_positions = torch.arange(
                    contract.STORED_SEQUENCE_TOKENS, dtype=torch.long
                ).unsqueeze(0).expand_as(positions)
                if not torch.equal(positions.to(torch.long), expected_positions):
                    raise RunnerError(f"observation positions are not 0..127 for {cell['cell_id']} rows {start}:{stop}")
                yield _Chunk(start, stop, activations.contiguous(), valid.contiguous(), positions.contiguous())
    except RunnerError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RunnerError(f"observation chunk read failed for {cell['cell_id']}") from exc


@torch.inference_mode()
def _predict_one(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    # Preserve the retained _JointAdapter numerical boundary exactly.
    staged_h = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
    staged_mask = valid_mask.to(device=device, dtype=torch.bool).unsqueeze(0)
    logits = model(staged_h, staged_mask, embeddings)
    if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, int(activation.shape[0])):
        raise RunnerError(f"decoder returned unexpected geometry: {tuple(logits.shape)}")
    raw = logits.argmax(dim=-1)[0].to(device="cpu", dtype=torch.long).contiguous()
    return contract.normalize_prediction(raw, valid_mask)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _warm_measured(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    started = time.perf_counter()
    warm = _predict_one(model, embeddings, activation, valid_mask, device=device)
    _sync(device)
    warm_seconds = time.perf_counter() - started
    started = time.perf_counter()
    measured = _predict_one(model, embeddings, activation, valid_mask, device=device)
    _sync(device)
    measured_seconds = time.perf_counter() - started
    if not torch.equal(warm, measured):
        raise RunnerError("warmup and measured predicted IDs differ")
    return measured, {
        "warmup_seconds": float(warm_seconds),
        "measured_seconds": float(measured_seconds),
        "warmup_output_exact_match_measured": True,
        "measured_output_selected": True,
    }


def _load_model(
    state: dict[str, Any],
    *,
    method_id: str,
    base_method_id: str,
    root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        expected_binding = contract.validate_published_state_binding(state, method_id)
        state_path = contract.resolve_path(state["path"], repository_root=root, description="decoder state")
        actual = contract.validate_file_record(
            state,
            repository_root=root,
            description="decoder state",
            verify=True,
        )
        with safe_open(str(state_path), framework="pt", device="cpu") as handle:
            state_metadata = dict(handle.metadata() or {})
        metadata_evidence = contract.validate_published_state_metadata(state_metadata, method_id)
    except contract.ContractError as exc:
        raise RunnerError(str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunnerError(f"decoder state metadata read failed: {state_path}") from exc
    if actual["sha256"] != expected_binding["sha256"] or actual["bytes"] != expected_binding["bytes"]:
        raise RunnerError(f"decoder state is not the exact published selected file: {state_path}")
    started = time.perf_counter()
    from token_reconstruction.trr0005_joint_decoder import load_decoder_state

    try:
        model = load_decoder_state(
            state_path,
            method_id=base_method_id,
            hidden_size=contract.HIDDEN_SIZE,
            vocabulary_size=contract.VOCAB_SIZE,
            context_width=128,
        ).to(device=device).eval()
        model.requires_grad_(False)
    except Exception as exc:
        raise RunnerError(f"decoder state load failed: {state_path}") from exc
    _sync(device)
    expected_mode = expected_binding["attention_mode"]
    expected_score = expected_binding["attention_score_mode"]
    if getattr(model, "attention_mode", None) != expected_mode or getattr(model, "attention_score_mode", None) != expected_score:
        raise RunnerError(f"loaded decoder attention semantics changed for {method_id}")
    return model, {
        "path": actual["path"],
        "bytes": actual["bytes"],
        "sha256": actual["sha256"],
        "method_id": method_id,
        "base_method_id": base_method_id,
        "source_commit": expected_binding["source_commit"],
        "published_parent_commit": contract.PUBLISHED_PARENT_COMMIT,
        "post_score_maintenance_commit": contract.POST_SCORE_MAINTENANCE_COMMIT,
        "loader": "token_reconstruction.trr0005_joint_decoder.load_decoder_state",
        "metadata": metadata_evidence,
        "load_seconds": float(time.perf_counter() - started),
        "attention_mode": getattr(model, "attention_mode", None),
        "attention_score_mode": getattr(model, "attention_score_mode", None),
    }

def _prediction_path(output_root: Path, cell_id: str, method_id: str) -> Path:
    style, condition = cell_id.split("__", 1)
    return output_root / style / condition / f"{method_id}.safetensors"


def _run_method(
    *,
    method_id: str,
    method_binding: dict[str, Any],
    cells: dict[str, Any],
    registration: dict[str, Any],
    output_root: Path,
    root: Path,
    device: torch.device,
    embeddings: torch.Tensor,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, state_evidence = _load_model(
        method_binding["state"],
        method_id=method_id,
        base_method_id=method_binding["base_method_id"],
        root=root,
        device=device,
    )
    if device.type == "cuda":
        state_evidence["cuda_peak_allocated_bytes_after_load"] = int(torch.cuda.max_memory_allocated(device))
        state_evidence["cuda_peak_reserved_bytes_after_load"] = int(torch.cuda.max_memory_reserved(device))
    guard = registration["resource_guard"]
    _guard(device=device, guard=guard, started=started, stage=f"after_{method_id}_load")
    prediction_entries: dict[str, Any] = {}
    timing_entries: dict[str, Any] = {}
    records = int(registration["records_per_domain"])
    chunk_records = int(registration["geometry"]["chunk_records"])
    try:
        for cell_id in registration["cell_order"]:
            cell = cells[cell_id]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            predictions = torch.empty((records, contract.STORED_SEQUENCE_TOKENS), dtype=torch.long)
            warmup_seconds = 0.0
            measured_seconds = 0.0
            per_record: list[float] = []
            rows_seen = 0
            chunks_seen = 0
            for chunk in _iter_observation_chunks(cell, records=records, chunk_records=chunk_records):
                if chunk.start != rows_seen:
                    raise RunnerError(f"observation chunks are not contiguous for {cell_id}")
                chunks_seen += 1
                for offset in range(chunk.stop - chunk.start):
                    row = chunk.start + offset
                    result, timing = _warm_measured(
                        model,
                        embeddings,
                        chunk.activations[offset],
                        chunk.mask[offset],
                        device=device,
                    )
                    predictions[row] = result
                    warmup_seconds += float(timing["warmup_seconds"])
                    measured_seconds += float(timing["measured_seconds"])
                    per_record.append(float(timing["measured_seconds"]))
                    rows_seen += 1
                _guard(device=device, guard=guard, started=started, stage=f"after_{method_id}_{cell_id}_rows_{rows_seen}")
            if rows_seen != records or chunks_seen != records // chunk_records:
                raise RunnerError(f"incomplete observation chunk coverage for {cell_id}: rows={rows_seen}, chunks={chunks_seen}")
            predictions = contract.validate_prediction_tensor(predictions, records=records)
            artifact = _prediction_path(output_root, cell_id, method_id)
            if artifact.exists() or artifact.is_symlink():
                raise RunnerError(f"prediction artifact is not create-only: {artifact}")
            metadata = {
                "schema": contract.PREDICTION_SCHEMA,
                "task_id": contract.TASK_ID,
                "registration_sha256": contract.sha256_file(Path(registration["_path"])),
                "observation_manifest_sha256": registration["observation_manifest"]["sha256"],
                "observation_sha256": cell["observation"]["sha256"],
                "cell_id": cell_id,
                "method_id": method_id,
                "records": str(records),
                "sequence_tokens": str(contract.STORED_SEQUENCE_TOKENS),
                "capture_sequence_tokens": str(contract.CAPTURE_SEQUENCE_TOKENS),
                "hidden_size": str(contract.HIDDEN_SIZE),
                "candidate_arrays_persisted": "false",
                "truth_opened": "false",
            }
            artifact.parent.mkdir(parents=True, exist_ok=True)
            save_file({"predictions": predictions}, str(artifact), metadata=metadata)
            artifact_record = _file_record(artifact)
            prediction_entry = {
                "schema": contract.PREDICTION_SCHEMA,
                "task_id": contract.TASK_ID,
                "cell_id": cell_id,
                "method_id": method_id,
                "records": records,
                "shape": [records, contract.STORED_SEQUENCE_TOKENS],
                "prediction_artifact": artifact_record,
                "prediction_sha256": contract.tensor_digest(predictions),
                "observation": dict(cell["observation"]),
                "state": dict(state_evidence),
                "registration_sha256": contract.sha256_file(Path(registration["_path"])),
                "truth_opened": False,
                "candidate_arrays_persisted": False,
            }
            timing_entry = {
                "schema": contract.TIMING_SCHEMA,
                "task_id": contract.TASK_ID,
                "cell_id": cell_id,
                "method_id": method_id,
                "records": records,
                "warmup_runs_per_record": 1,
                "measured_runs_per_record": 1,
                "warmup_seconds_sum": warmup_seconds,
                "measured_seconds_sum": measured_seconds,
                "timed_interval_total_seconds": warmup_seconds + measured_seconds,
                "per_record_measured_seconds": per_record,
                "warmup_output_exact_match_measured": True,
                "measured_output_selected": True,
                "steady_interval": "CPU activation H -> CUDA FP32 preprocessing -> decoder -> predicted IDs CPU",
                "chunk_records": chunk_records,
                "chunks": chunks_seen,
                "load_seconds_separate": state_evidence["load_seconds"],
                "peak_memory": {
                    "scope": "cell-local CUDA counters reset before observation iteration; RSS is process-cumulative",
                    "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
                },
                "prediction_artifact": artifact_record,
                "prediction_sha256": prediction_entry["prediction_sha256"],
                "truth_opened": False,
            }
            receipt = dict(timing_entry)
            receipt["prediction_artifact"] = artifact_record
            receipt_path = artifact.with_suffix(".run.json")
            contract.write_create_only(receipt_path, receipt)
            prediction_entries[cell_id] = prediction_entry
            timing_entries[cell_id] = timing_entry
            del predictions
            gc.collect()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return prediction_entries, timing_entries


def _output_root(registration: dict[str, Any], root: Path) -> Path:
    raw = Path(str(registration["output_root"])).expanduser()
    if not raw.is_absolute():
        raw = root / raw
    output = raw.resolve()
    task_root = (root / "experiments/TRR-0006").resolve()
    try:
        output.relative_to(task_root)
    except ValueError as exc:
        raise RunnerError(f"output root must be task-owned under {task_root}: {output}") from exc
    if output.exists() and output.is_symlink():
        raise RunnerError(f"output root is a symlink: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def execute(*, registration_path: Path, repository_root: Path, device_name: str = "cuda") -> dict[str, Any]:
    root = _repository_root(repository_root)
    registration_path = registration_path.expanduser().resolve()
    registration = contract.load_registration(registration_path)
    registration["_path"] = str(registration_path)
    numerical_evidence = _configure_numerics(registration["numerical_settings"])
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RunnerError("TRR-0006 prediction execution requires CUDA")
    device = torch.device("cuda")
    output_root = _output_root(registration, root)
    started = time.perf_counter()
    started_utc = _utc_now()
    failure_path = output_root / "failure.json"
    try:
        code_bindings = _verify_code_bindings(registration, root)
        _, parsed_observations, observation_record = contract.load_observation_manifest(
            registration,
            repository_root=root,
            verify_assets=True,
        )
        guard_before = _guard(
            device=device,
            guard=registration["resource_guard"],
            started=started,
            stage="before_process_start_load",
        )
        embeddings, embedding_evidence = _load_embedding(registration, root, device)
        guard_after_embedding = _guard(
            device=device,
            guard=registration["resource_guard"],
            started=started,
            stage="after_embedding_load",
        )
        predictions: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        for method_id in registration["method_ids"]:
            method_predictions, method_timings = _run_method(
                method_id=method_id,
                method_binding=registration["methods"][method_id],
                cells=parsed_observations["cells"],
                registration=registration,
                output_root=output_root,
                root=root,
                device=device,
                embeddings=embeddings,
                started=started,
            )
            predictions.update({f"{cell}::{method_id}": value for cell, value in method_predictions.items()})
            timings.update({f"{cell}::{method_id}": value for cell, value in method_timings.items()})
        registration_record = _file_record(registration_path)
        run_manifest = {
            "schema": contract.RUN_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH",
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "registration": registration_record,
            "observation_manifest": observation_record,
            "code_commit": registration["code_commit"],
            "code_bindings": code_bindings,
            "source_lineage": {
                "scientific_source_commit": contract.SCIENTIFIC_SOURCE_COMMIT,
                "published_parent_commit": contract.PUBLISHED_PARENT_COMMIT,
                "post_score_maintenance_commit": contract.POST_SCORE_MAINTENANCE_COMMIT,
                "maintenance_inference_equivalence_proven": False,
            },
            "runtime_assets": {"normalized_public_E": embedding_evidence},
            "numerical_settings": numerical_evidence,
            "geometry": dict(registration["geometry"]),
            "records_per_domain": registration["records_per_domain"],
            "cell_order": list(registration["cell_order"]),
            "method_ids": list(registration["method_ids"]),
            "resource_guard": {
                "before_process_start_load": guard_before,
                "after_embedding_load": guard_after_embedding,
                "final": _guard(device=device, guard=registration["resource_guard"], started=started, stage="final"),
            },
            "predictions_count": len(predictions),
            "timings_count": len(timings),
            "predictions_complete": len(predictions) == len(registration["cell_order"]) * len(registration["method_ids"]),
            "timing_decisions_complete": len(timings) == len(predictions),
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "candidate_arrays_persisted": False,
        }
        if not run_manifest["predictions_complete"] or not run_manifest["timing_decisions_complete"]:
            raise RunnerError("complete prediction/timing matrix was not produced")
        contract.write_create_only(output_root / "predictions.json", {
            "schema": "token-reconstruction.trr0006-prediction-descriptor-manifest.v1",
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH",
            "registration_sha256": registration_record["sha256"],
            "records_per_domain": registration["records_per_domain"],
            "cell_order": list(registration["cell_order"]),
            "method_ids": list(registration["method_ids"]),
            "predictions": predictions,
            "truth_opened": False,
        })
        contract.write_create_only(output_root / "timings.json", {
            "schema": "token-reconstruction.trr0006-timing-descriptor-manifest.v1",
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_TIMINGS_COMPLETE_NO_TRUTH",
            "registration_sha256": registration_record["sha256"],
            "records_per_domain": registration["records_per_domain"],
            "cell_order": list(registration["cell_order"]),
            "method_ids": list(registration["method_ids"]),
            "timings": timings,
            "truth_opened": False,
        })
        contract.write_create_only(output_root / "run_manifest.json", run_manifest)
        return run_manifest
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            contract.write_create_only(failure_path, {
                "schema": contract.FAILURE_SCHEMA,
                "task_id": contract.TASK_ID,
                "status": "FAILED_PRESERVED_NO_TRUTH",
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "registration": str(registration_path),
                "truth_opened": False,
                "source_text_loaded": False,
                "target_labels_loaded": False,
            })
        if isinstance(exc, RunnerError):
            raise
        if isinstance(exc, contract.ContractError):
            raise RunnerError(str(exc)) from exc
        raise RunnerError("prediction run failed") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = execute(
        registration_path=args.registration,
        repository_root=args.repository_root,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
