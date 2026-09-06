"""Run the four TRR-0008 current-H-only decoders on public observations.

The runner has no source or truth input.  It loads the frozen registration,
stages BF16 observations as FP32, projects each current hidden state with the
registered decoder, performs a full public-vocabulary argmax, and writes
create-only ID artifacts.  Timing preparation, warmup, measured work, and
serialization are recorded separately.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts import trr0008_eval_contract as contract


class RunnerError(contract.ContractError):
    pass


@dataclass(frozen=True)
class Chunk:
    start: int
    stop: int
    activations: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RunnerError(f"repository root is unavailable: {root}")
    return root


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError("cannot resolve runner commit") from exc


def _rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    return value * 1024 if sys.platform != "darwin" else value


def _host_available_bytes() -> int | None:
    try:
        value = Path("/proc/meminfo").read_text(encoding="ascii")
        for line in value.splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def _configure_numerics(settings: Mapping[str, Any]) -> dict[str, Any]:
    expected = contract.NUMERICAL_SETTINGS
    for key, value in expected.items():
        if settings.get(key) != value:
            raise RunnerError(f"numerical setting changed: {key}")
    torch.set_num_threads(int(settings["cpu_intraop_threads"]))
    try:
        torch.set_num_interop_threads(int(settings["cpu_interop_threads"]))
    except RuntimeError as exc:
        # Continuing with a different inter-op pool changes the qualified
        # execution environment, so this is a failed-closed preparation error.
        raise RunnerError("unable to set frozen CPU inter-op threads") from exc
    if torch.get_num_threads() != int(settings["cpu_intraop_threads"]):
        raise RunnerError("effective CPU intra-op threads differ from registration")
    if torch.get_num_interop_threads() != int(settings["cpu_interop_threads"]):
        raise RunnerError("effective CPU inter-op threads differ from registration")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(settings["cuda_matmul_allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(settings["cuda_cudnn_allow_tf32"])
    torch.set_float32_matmul_precision(str(settings["float32_matmul_precision"]))
    return {
        "settings": dict(settings),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _gpu_compute_apps() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnerError("GPU process telemetry is unavailable") from exc
    if result.returncode != 0:
        raise RunnerError(f"GPU process telemetry failed: {result.stderr.strip()}")
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] and fields[0] != str(os.getpid()):
            rows.append({"pid": fields[0], "process_name": fields[1] if len(fields) > 1 else "", "used_memory": fields[2] if len(fields) > 2 else ""})
    return rows


def _guard(*, device: torch.device, guard: Mapping[str, Any], started: float, stage: str) -> None:
    if time.perf_counter() - started > float(guard["maximum_seconds"]):
        raise RunnerError(f"resource time guard exceeded at {stage}")
    rss = _rss_bytes()
    if rss is None:
        raise RunnerError(f"process RSS telemetry unavailable at {stage}")
    if rss > int(guard["maximum_rss_bytes"]):
        raise RunnerError(f"process RSS guard exceeded at {stage}")
    host = _host_available_bytes()
    if host is None:
        raise RunnerError(f"host-memory telemetry unavailable at {stage}")
    if host < int(guard["minimum_host_available_bytes"]):
        raise RunnerError(f"host-memory guard exceeded at {stage}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RunnerError("CUDA is unavailable")
        foreign = _gpu_compute_apps()
        if foreign:
            raise RunnerError(f"GPU is not exclusive at {stage}: {foreign!r}")
        free, reserved = torch.cuda.mem_get_info(device)
        if int(free) < int(guard["minimum_free_gpu_bytes"]):
            raise RunnerError(f"free-GPU guard exceeded at {stage}")
        if int(torch.cuda.memory_reserved(device)) > int(guard["maximum_reserved_gpu_bytes"]):
            raise RunnerError(f"reserved-GPU guard exceeded at {stage}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_embedding(registration: Mapping[str, Any], *, root: Path, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    binding = registration["runtime_assets"]["normalized_public_E"]
    record = contract.validate_file_record(binding, repository_root=root, description="normalized public E", verify=True)
    started = time.perf_counter()
    values = load_file(str(record["path"]), device="cpu")
    if set(values) != {"embeddings"}:
        raise RunnerError("normalized public E must contain only embeddings")
    table_cpu = values["embeddings"].detach().contiguous()
    if table_cpu.dtype != torch.float32 or tuple(table_cpu.shape) != (contract.VOCABULARY_SIZE, contract.HIDDEN_SIZE):
        raise RunnerError("normalized public E geometry or dtype changed")
    table = table_cpu.to(device=device).contiguous()
    _synchronize(device)
    del values, table_cpu
    gc.collect()
    return table, {
        "path": record["path"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "shape": [contract.VOCABULARY_SIZE, contract.HIDDEN_SIZE],
        "dtype": "torch.float32",
        "load_seconds": float(time.perf_counter() - started),
    }


def _iter_observation_chunks(cell: Mapping[str, Any], *, records: int, chunk_records: int) -> Iterator[Chunk]:
    observation = cell.get("observation", cell)
    path = Path(str(observation["path"])).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"observation is unavailable: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise RunnerError("observation tensor keys changed")
            activation_slice = handle.get_slice("activations")
            mask_slice = handle.get_slice("attention_mask")
            position_slice = handle.get_slice("position_ids")
            expected_h = (records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE)
            expected_side = (records, contract.STORED_SEQUENCE_TOKENS)
            if tuple(activation_slice.get_shape()) != expected_h:
                raise RunnerError("observation activation geometry changed")
            if tuple(mask_slice.get_shape()) != expected_side or tuple(position_slice.get_shape()) != expected_side:
                raise RunnerError("observation sidecar geometry changed")
            if records % chunk_records:
                raise RunnerError("record count must be divisible by chunk size")
            for start in range(0, records, chunk_records):
                stop = start + chunk_records
                activations = activation_slice[start:stop]
                mask_raw = mask_slice[start:stop]
                positions = position_slice[start:stop]
                if activations.dtype != torch.bfloat16:
                    raise RunnerError("observation activations must be BF16")
                if mask_raw.dtype not in (torch.bool, torch.uint8):
                    raise RunnerError("observation mask dtype changed")
                if positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                    raise RunnerError("observation position dtype changed")
                if not torch.isfinite(activations.float()).all().item():
                    raise RunnerError("observation contains non-finite values")
                mask = mask_raw.to(torch.bool).contiguous()
                if not mask.all().item():
                    raise RunnerError("TRR-0008 clips must be fully valid")
                expected_positions = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.long).unsqueeze(0).expand_as(positions)
                if not torch.equal(positions.to(torch.long), expected_positions):
                    raise RunnerError("observation positions are not 0..127")
                yield Chunk(start, stop, activations.contiguous(), mask, positions.to(torch.long).contiguous())
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"observation read failed: {path}") from exc


def _iter_rows(cell: Mapping[str, Any], *, records: int) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    seen = 0
    for chunk in _iter_observation_chunks(cell, records=records, chunk_records=contract.CHUNK_RECORDS):
        for offset in range(chunk.stop - chunk.start):
            row = chunk.start + offset
            yield row, chunk.activations[offset], chunk.mask[offset], chunk.positions[offset]
            seen += 1
    if seen != records:
        raise RunnerError(f"observation rows cover {seen}, expected {records}")


@torch.inference_mode()
def predict_current_h(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    activation: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Predict one clip from only its current-H matrix and valid mask."""

    staged = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
    mask = valid_mask.to(device=device, dtype=torch.bool).unsqueeze(0)
    try:
        projected = model.projected_hidden(staged, mask)
        positions = torch.arange(1, contract.STORED_SEQUENCE_TOKENS, device=device, dtype=torch.long)
        rows = torch.zeros_like(positions)
        logits = model.logits_from_rows(projected, rows, positions, embedding)
    except AttributeError:
        logits_full = model(staged, mask, embedding)
        if logits_full.ndim != 3 or tuple(logits_full.shape[:2]) != (1, contract.STORED_SEQUENCE_TOKENS):
            raise RunnerError(f"decoder returned unexpected geometry: {tuple(logits_full.shape)}")
        logits = logits_full[0, 1:]
    if tuple(logits.shape) != (contract.SCORED_POST_BOS_TOKENS, contract.VOCABULARY_SIZE):
        raise RunnerError(f"decoder logits geometry changed: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all().item():
        raise RunnerError("decoder emitted non-finite logits")
    raw = torch.full((contract.STORED_SEQUENCE_TOKENS,), contract.INVALID_TOKEN_ID, dtype=torch.long)
    raw[0] = contract.BOS_TOKEN_ID
    raw[1:] = logits.argmax(dim=-1).to(device="cpu", dtype=torch.long)
    return contract.normalize_prediction(raw, valid_mask)


def _load_decoder(row: Mapping[str, Any], *, root: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    method_id = str(row["id"])
    state = contract.validate_file_record(row["state"], repository_root=root, description=f"{method_id} state", verify=True)
    loader_desc = row["loader"]
    try:
        module = importlib.import_module(str(loader_desc["module"]))
        function = getattr(module, str(loader_desc["function"]))
        model = function(Path(state["path"]), **dict(loader_desc.get("kwargs", {})))
        if not isinstance(model, torch.nn.Module):
            raise TypeError("loader did not return a module")
        model.requires_grad_(False)
        model = model.to(device=device).eval()
        _synchronize(device)
    except Exception as exc:
        raise RunnerError(f"decoder {method_id} could not be loaded") from exc
    if int(getattr(model, "hidden_size", -1)) != contract.HIDDEN_SIZE or int(getattr(model, "vocabulary_size", -1)) != contract.VOCABULARY_SIZE:
        raise RunnerError(f"decoder {method_id} geometry changed")
    return model, {
        "method_id": method_id,
        "state": state,
        "loader": f"{loader_desc['module']}.{loader_desc['function']}",
        "parameter_count": int(sum(int(p.numel()) for p in model.parameters())),
    }


def _prediction_path(output_root: Path, cell_id: str, method_id: str) -> Path:
    style, condition = cell_id.split("__", 1)
    return output_root / style / condition / f"{method_id}.safetensors"


def _write_prediction(
    path: Path,
    *,
    prediction: torch.Tensor,
    registration: Mapping[str, Any],
    cell_id: str,
    records: int,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise RunnerError(f"prediction artifact is not create-only: {path}")
    checked = contract.validate_prediction_tensor(prediction, records=records)
    metadata = {
        "schema": contract.PREDICTION_SCHEMA,
        "task_id": contract.TASK_ID,
        "registration_sha256": registration["registration_sha256"],
        "cell_id": cell_id,
        "method_id": path.stem,
        "records": str(records),
        "geometry_json": json.dumps({"records": records, **contract.STATIC_GEOMETRY}, sort_keys=True),
        "truth_opened": "false",
        "candidate_arrays_persisted": "false",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"predictions": checked}, str(path), metadata=metadata)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": contract.sha256_file(path),
        "prediction_sha256": contract.tensor_digest(checked),
        "records": records,
    }


def _output_root(registration: Mapping[str, Any], *, root: Path) -> Path:
    output = Path(str(registration["output_root"])).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    task_root = (root / "experiments" / "TRR-0008").resolve()
    try:
        output.relative_to(task_root)
    except ValueError as exc:
        raise RunnerError(f"output root must be below {task_root}") from exc
    if output.is_symlink():
        raise RunnerError("output root is a symlink")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _run_method(
    *,
    row: Mapping[str, Any],
    registration: Mapping[str, Any],
    observations: Mapping[str, Any],
    embedding: torch.Tensor,
    output_root: Path,
    root: Path,
    device: torch.device,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    load_started = time.perf_counter()
    model, state_evidence = _load_decoder(row, root=root, device=device)
    state_evidence["model_preparation_seconds"] = float(time.perf_counter() - load_started)
    predictions: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    try:
        for cell_id in contract.CELL_ORDER:
            records = contract.records_for_cell(observations, cell_id)
            cell = contract._as_cells(observations)[cell_id]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            values = torch.empty((records, contract.STORED_SEQUENCE_TOKENS), dtype=torch.long)
            warmup_sum = 0.0
            measured_sum = 0.0
            per_record: list[float] = []
            for row_index, activation, mask, _positions in _iter_rows(cell, records=records):
                _guard(device=device, guard=registration["resource_guard"], started=started, stage=f"before_{row['id']}_{cell_id}_{row_index}")
                _synchronize(device)
                t0 = time.perf_counter()
                warm = predict_current_h(model, embedding, activation, mask, device=device)
                _synchronize(device)
                warm_seconds = time.perf_counter() - t0
                _synchronize(device)
                t1 = time.perf_counter()
                measured = predict_current_h(model, embedding, activation, mask, device=device)
                _synchronize(device)
                measured_seconds = time.perf_counter() - t1
                if not torch.equal(warm, measured):
                    raise RunnerError(f"warmup/measured IDs differ: {row['id']}/{cell_id}/{row_index}")
                values[row_index] = measured
                warmup_sum += warm_seconds
                measured_sum += measured_seconds
                per_record.append(float(measured_seconds))
            artifact_path = _prediction_path(output_root, cell_id, str(row["id"]))
            artifact = _write_prediction(
                artifact_path,
                prediction=values,
                registration=registration,
                cell_id=cell_id,
                records=records,
            )
            timing = {
                "schema": contract.TIMING_SCHEMA,
                "task_id": contract.TASK_ID,
                "method_id": str(row["id"]),
                "cell_id": cell_id,
                "records": records,
                "warmup_runs_per_record": 1,
                "measured_runs_per_record": 1,
                "warmup_seconds_sum": float(warmup_sum),
                "measured_seconds_sum": float(measured_sum),
                "per_record_measured_seconds": per_record,
                "warmup_output_exact_match_measured": True,
                "measured_output_selected": True,
                "timed_interval": "synchronized BF16 H staging -> FP32 current-H decoder -> full-vocabulary argmax -> CPU IDs",
                "model_preparation_seconds": state_evidence["model_preparation_seconds"],
                "peak_memory": {
                    "process_max_rss_bytes": _rss_bytes(),
                    "host_available_bytes": _host_available_bytes(),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
                },
                "prediction_artifact": artifact,
                "truth_opened": False,
                "candidate_arrays_persisted": False,
            }
            contract.write_create_only(artifact_path.with_suffix(".run.json"), timing)
            predictions[cell_id] = artifact | {"method_id": row["id"], "cell_id": cell_id, "state": state_evidence}
            timings[f"{row['id']}::{cell_id}"] = timing
            del values
            gc.collect()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return predictions, timings


def _execute_impl(*, registration_path: Path, repository_root: Path, device_name: str = "cuda") -> dict[str, Any]:
    root = _root(repository_root)
    registration_path = Path(registration_path).expanduser().resolve()
    registration = contract.load_registration(registration_path, repository_root=root, verify_assets=False)
    registration["registration_sha256"] = contract.sha256_file(registration_path)
    if registration.get("code_commit") != _git_head(root):
        raise RunnerError("registration code_commit does not match executable HEAD")
    numerical_evidence = _configure_numerics(registration["numerical_settings"])
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RunnerError("CUDA is unavailable")
    started = time.perf_counter()
    _guard(device=device, guard=registration["resource_guard"], started=started, stage="before_public_inputs")
    observation_path = Path(registration["observation_manifest"]["path"])
    observations = contract.validate_observation_manifest(
        contract.load_json(observation_path, description="TRR-0008 observations"),
        repository_root=root,
        verify_assets=True,
    )
    embedding, embedding_evidence = _load_embedding(registration, root=root, device=device)
    _guard(device=device, guard=registration["resource_guard"], started=started, stage="after_public_embedding")
    output_root = _output_root(registration, root=root)
    if (output_root / "registration.json").exists() or (output_root / "registration.json").is_symlink():
        raise RunnerError("output root already contains registration.json")
    (output_root / "registration.json").write_bytes(registration_path.read_bytes())
    all_predictions: dict[str, Any] = {}
    all_timings: dict[str, Any] = {}
    for row in registration["methods"]:
        prediction_rows, timing_rows = _run_method(
            row=row,
            registration=registration,
            observations=observations,
            embedding=embedding,
            output_root=output_root,
            root=root,
            device=device,
            started=started,
        )
        all_predictions.update({f"{row['id']}::{cell}": value for cell, value in prediction_rows.items()})
        all_timings.update(timing_rows)
    run_manifest = {
        "schema": contract.RUN_SCHEMA,
        "task_id": contract.TASK_ID,
        "registration": {"path": str(registration_path), "sha256": registration["registration_sha256"]},
        "code_commit": registration["code_commit"],
        "numerical_settings": numerical_evidence,
        "device": str(device),
        "observation_manifest": registration["observation_manifest"],
        "embedding": embedding_evidence,
        "predictions": all_predictions,
        "timings": all_timings,
        "elapsed_seconds": float(time.perf_counter() - started),
        "truth_opened": False,
        "candidate_arrays_persisted": False,
        "completed_utc": _utc_now(),
    }
    contract.write_create_only(output_root / "run_manifest.json", run_manifest)
    del embedding
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_manifest



def _failure_output_root(registration_path: Path, *, root: Path) -> Path:
    """Best-effort output-root resolution for a create-only failure receipt."""

    try:
        raw = contract.load_json(registration_path, description="TRR-0008 registration")
        value = raw.get("output_root")
        if isinstance(value, str) and value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            task_root = (root / "experiments" / contract.TASK_ID).resolve()
            try:
                path.relative_to(task_root)
            except ValueError:
                return Path(registration_path).expanduser().resolve().parent
            return path
    except Exception:
        pass
    return Path(registration_path).expanduser().resolve().parent


def _write_failure_receipt(
    *,
    registration_path: Path,
    root: Path,
    started_utc: str,
    started_clock: float,
    exc: BaseException,
) -> None:
    output_root = _failure_output_root(registration_path, root=root)
    output_root.mkdir(parents=True, exist_ok=True)
    failure_path = output_root / "failure.json"
    partial_paths: list[str] = []
    if output_root.is_dir():
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path != failure_path:
                partial_paths.append(str(path))
    registration_record: dict[str, Any] | None = None
    try:
        registration_record = {
            "path": str(Path(registration_path).resolve()),
            "bytes": int(Path(registration_path).stat().st_size),
            "sha256": contract.sha256_file(Path(registration_path)),
        }
    except Exception:
        pass
    receipt = {
        "schema": "token-reconstruction.trr0008-runner-failure.v1",
        "task_id": contract.TASK_ID,
        "status": "FAILED_NO_TRUTH",
        "registration": registration_record,
        "stage": "runner_execute",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "elapsed_seconds": float(time.perf_counter() - started_clock),
        "partial_artifacts": partial_paths,
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }
    try:
        contract.write_create_only(failure_path, receipt)
    except Exception:
        # The original runner error remains authoritative; a pre-existing or
        # unwritable failure path must not obscure it.
        pass


def execute(*, registration_path: Path, repository_root: Path, device_name: str = "cuda") -> dict[str, Any]:
    root = _root(repository_root)
    registration_path = Path(registration_path).expanduser().resolve()
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    try:
        return _execute_impl(
            registration_path=registration_path,
            repository_root=root,
            device_name=device_name,
        )
    except Exception as exc:
        _write_failure_receipt(
            registration_path=registration_path,
            root=root,
            started_utc=started_utc,
            started_clock=started_clock,
            exc=exc,
        )
        if isinstance(exc, RunnerError):
            raise
        if isinstance(exc, contract.ContractError):
            raise RunnerError(str(exc)) from exc
        raise RunnerError("prediction run failed") from exc

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = execute(registration_path=args.registration, repository_root=args.repository_root, device_name=args.device)
    except (RunnerError, contract.ContractError) as exc:
        print(f"TRR-0008 runner failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": value["schema"], "elapsed_seconds": value["elapsed_seconds"], "prediction_count": len(value["predictions"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
