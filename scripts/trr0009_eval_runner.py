"""Run the frozen TRR-0009 current-H method matrix before truth is opened."""
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

from scripts import trr0009_eval_contract as contract


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
        raise RunnerError(f"repository root unavailable: {root}")
    return root


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError("cannot resolve runner code commit") from exc
    if len(value) != 40:
        raise RunnerError("runner code commit is not a full hash")
    return value


def _rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * 1024 if sys.platform != "darwin" else value
    except (AttributeError, OSError, ValueError):
        return None


def _host_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def _gpu_compute_apps() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnerError("GPU process telemetry unavailable") from exc
    if result.returncode != 0:
        raise RunnerError(f"GPU process telemetry failed: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] and fields[0] != str(os.getpid()):
            rows.append({"pid": fields[0], "process_name": fields[1] if len(fields) > 1 else "", "used_memory": fields[2] if len(fields) > 2 else ""})
    return rows


def _configure_numerics(settings: Mapping[str, Any]) -> dict[str, Any]:
    if dict(settings) != contract.NUMERICAL_SETTINGS:
        raise RunnerError("numerical settings changed")
    try:
        torch.set_num_threads(int(settings["cpu_intraop_threads"]))
        torch.set_num_interop_threads(int(settings["cpu_interop_threads"]))
    except RuntimeError as exc:
        raise RunnerError("unable to set CPU numerical settings") from exc
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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _guard(*, device: torch.device, guard: Mapping[str, Any], started: float, stage: str) -> None:
    if time.perf_counter() - started > float(guard["maximum_seconds"]):
        raise RunnerError(f"resource time guard exceeded at {stage}")
    rss = _rss_bytes()
    host = _host_available_bytes()
    if rss is None or host is None:
        raise RunnerError(f"memory telemetry unavailable at {stage}")
    if rss > int(guard["maximum_rss_bytes"]):
        raise RunnerError(f"RSS guard exceeded at {stage}")
    if host < int(guard["minimum_host_available_bytes"]):
        raise RunnerError(f"host-memory guard exceeded at {stage}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RunnerError("CUDA is unavailable")
        foreign = _gpu_compute_apps()
        if foreign:
            raise RunnerError(f"foreign GPU compute process at {stage}: {foreign!r}")
        free, _total = torch.cuda.mem_get_info(device)
        if int(free) < int(guard["minimum_free_gpu_bytes"]):
            raise RunnerError(f"free-GPU guard exceeded at {stage}")
        if int(torch.cuda.memory_reserved(device)) > int(guard["maximum_reserved_gpu_bytes"]):
            raise RunnerError(f"reserved-GPU guard exceeded at {stage}")


def _load_embedding(registration: Mapping[str, Any], *, root: Path, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    binding = contract.validate_file_record(registration["runtime_embedding"], repository_root=root, description="normalized public embedding", verify=True)
    started = time.perf_counter()
    try:
        values = load_file(binding["path"], device="cpu")
        if set(values) != {"embeddings"}:
            raise RunnerError("embedding must contain only embeddings")
        table_cpu = values["embeddings"].detach().contiguous()
        if table_cpu.dtype != torch.float32 or tuple(table_cpu.shape) != (contract.VOCABULARY_SIZE, contract.HIDDEN_SIZE):
            raise RunnerError("embedding geometry or dtype changed")
        table = table_cpu.to(device=device).contiguous()
        _synchronize(device)
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError("embedding could not be loaded") from exc
    finally:
        if "values" in locals():
            del values
        if "table_cpu" in locals():
            del table_cpu
        gc.collect()
    return table, {"binding": binding, "shape": [contract.VOCABULARY_SIZE, contract.HIDDEN_SIZE], "dtype": "torch.float32", "load_seconds": float(time.perf_counter() - started)}


def _iter_observation_chunks(cell: Mapping[str, Any], *, records: int) -> Iterator[Chunk]:
    observation = cell.get("observation", cell)
    path = Path(str(observation["path"])).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"observation unavailable: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise RunnerError("observation tensor keys changed")
            activation_slice = handle.get_slice("activations")
            mask_slice = handle.get_slice("attention_mask")
            position_slice = handle.get_slice("position_ids")
            if tuple(activation_slice.get_shape()) != (records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE):
                raise RunnerError("observation activation geometry changed")
            if tuple(mask_slice.get_shape()) != (records, contract.STORED_SEQUENCE_TOKENS) or tuple(position_slice.get_shape()) != (records, contract.STORED_SEQUENCE_TOKENS):
                raise RunnerError("observation sidecar geometry changed")
            if records % contract.CHUNK_RECORDS:
                raise RunnerError("record count is not divisible by chunk size")
            expected_positions = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.long).unsqueeze(0)
            for start in range(0, records, contract.CHUNK_RECORDS):
                stop = start + contract.CHUNK_RECORDS
                activations = activation_slice[start:stop]
                mask_raw = mask_slice[start:stop]
                positions = position_slice[start:stop]
                if activations.dtype != torch.bfloat16 or mask_raw.dtype not in (torch.bool, torch.uint8):
                    raise RunnerError("observation dtype changed")
                if not torch.isfinite(activations.float()).all().item():
                    raise RunnerError("observation contains non-finite values")
                mask = mask_raw.to(torch.bool).contiguous()
                if not mask.all().item() or not torch.equal(positions.to(torch.long), expected_positions.expand_as(positions)):
                    raise RunnerError("observation mask or positions changed")
                yield Chunk(start, stop, activations.contiguous(), mask, positions.to(torch.long).contiguous())
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"observation read failed: {path}") from exc


def _iter_rows(cell: Mapping[str, Any], *, records: int) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    seen = 0
    for chunk in _iter_observation_chunks(cell, records=records):
        for offset in range(chunk.stop - chunk.start):
            row = chunk.start + offset
            yield row, chunk.activations[offset], chunk.mask[offset], chunk.positions[offset]
            seen += 1
    if seen != records:
        raise RunnerError(f"observation rows cover {seen}, expected {records}")


@torch.inference_mode()
def predict_current_h(model: torch.nn.Module, embedding: torch.Tensor, activation: torch.Tensor, valid_mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
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
    if tuple(logits.shape) != (contract.SCORED_POST_BOS_TOKENS, contract.VOCABULARY_SIZE) or not torch.isfinite(logits).all().item():
        raise RunnerError("decoder logits geometry or finiteness changed")
    raw = torch.full((contract.STORED_SEQUENCE_TOKENS,), contract.INVALID_TOKEN_ID, dtype=torch.long)
    raw[0] = contract.BOS_TOKEN_ID
    raw[1:] = logits.argmax(dim=-1).to(device="cpu", dtype=torch.long)
    return contract.normalize_prediction(raw, valid_mask)


def _load_tensor_arg(binding: Mapping[str, Any], *, root: Path) -> torch.Tensor:
    record = contract.validate_file_record(binding, repository_root=root, description="loader tensor argument", verify=True)
    path = Path(record["path"])
    if path.suffix == ".json":
        value = contract.load_json(path, description="loader tensor argument")
        if "values" not in value:
            raise RunnerError("JSON tensor argument lacks values")
        return torch.as_tensor(value["values"])
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if len(keys) != 1:
                raise RunnerError("tensor argument must contain one tensor")
            return handle.get_tensor(keys[0])
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"tensor argument unavailable: {path}") from exc


def _load_decoder(row: Mapping[str, Any], *, root: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    method_id = str(row["id"])
    state = contract.validate_file_record(row["state"], repository_root=root, description=f"{method_id} state", verify=True)
    loader_desc = row["loader"]
    kwargs = dict(loader_desc.get("kwargs", {}))
    for name, binding in loader_desc.get("path_args", {}).items():
        record = contract.validate_file_record(binding, repository_root=root, description=f"{method_id} path argument {name}", verify=True)
        kwargs[str(name)] = Path(record["path"])
    for name, binding in loader_desc.get("tensor_args", {}).items():
        kwargs[str(name)] = _load_tensor_arg(binding, root=root)
    try:
        module = importlib.import_module(str(loader_desc["module"]))
        function = getattr(module, str(loader_desc["function"]))
        model = function(Path(state["path"]), **kwargs)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("loader did not return torch.nn.Module")
        model.requires_grad_(False)
        model = model.to(device=device).eval()
        _synchronize(device)
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"decoder {method_id} could not be loaded") from exc
    if int(getattr(model, "hidden_size", -1)) != contract.HIDDEN_SIZE or int(getattr(model, "vocabulary_size", -1)) != contract.VOCABULARY_SIZE:
        raise RunnerError(f"decoder {method_id} geometry changed")
    return model, {"method_id": method_id, "state": state, "loader": f"{loader_desc['module']}.{loader_desc['function']}", "parameter_count": int(sum(int(p.numel()) for p in model.parameters()))}


def _output_root(registration: Mapping[str, Any], *, root: Path) -> Path:
    output = Path(str(registration["output_root"])).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    task_root = (root / "experiments" / contract.TASK_ID).resolve()
    try:
        output.relative_to(task_root)
    except ValueError as exc:
        raise RunnerError("output root must be below TRR-0009 task root") from exc
    if output.is_symlink():
        raise RunnerError("output root is a symlink")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_prediction(path: Path, *, prediction: torch.Tensor, registration: Mapping[str, Any], cell_id: str, method_id: str, records: int) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise RunnerError(f"prediction artifact is not create-only: {path}")
    checked = contract.validate_prediction_tensor(prediction, records=records)
    metadata = {
        "schema": contract.PREDICTION_SCHEMA,
        "task_id": contract.TASK_ID,
        "registration_sha256": str(registration["registration_sha256"]),
        "cell_id": cell_id,
        "method_id": method_id,
        "records": str(records),
        "geometry_json": json.dumps({"records": records, **contract.STATIC_GEOMETRY}, sort_keys=True),
        "truth_opened": "false",
        "candidate_arrays_persisted": "false",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"predictions": checked}, str(path), metadata=metadata)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": contract.sha256_file(path), "prediction_sha256": contract.tensor_digest(checked), "records": records}


def _code_and_input_recheck(registration: Mapping[str, Any], *, root: Path, registration_path: Path, initial_registration_sha256: str) -> None:
    try:
        contract.validate_registration(registration, repository_root=root, verify_assets=True)
    except contract.ContractError as exc:
        raise RunnerError(f"registration binding changed: {exc}") from exc
    if _git_head(root) != str(registration["code_commit"]):
        raise RunnerError("runner code commit changed during execution")
    current_registration_sha256 = contract.sha256_file(Path(registration_path))
    if current_registration_sha256 != str(initial_registration_sha256):
        raise RunnerError("registration file changed during execution")


def execute(*, registration_path: Path, repository_root: Path, device_name: str = "cuda") -> dict[str, Any]:
    root = _root(repository_root)
    registration_path = Path(registration_path).expanduser().resolve()
    started = time.perf_counter()
    started_utc = _utc_now()
    try:
        registration = contract.load_json(registration_path, description="TRR-0009 registration")
        contract.validate_registration(registration, repository_root=root, verify_assets=True)
        initialization = registration.get("initialization_equivalence")
        if not isinstance(initialization, Mapping) or initialization.get("status") != "PASS":
            raise RunnerError("initialization/harness output equivalence is missing or failed")
        initial_registration_sha256 = contract.sha256_file(registration_path)
        registration["registration_sha256"] = initial_registration_sha256
        if _git_head(root) != str(registration["code_commit"]):
            raise RunnerError("registration code commit is not current")
        numerics = _configure_numerics(registration["numerical_settings"])
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RunnerError("CUDA is unavailable")
        _guard(device=device, guard=registration["resource_guard"], started=started, stage="initial")
        output_root = _output_root(registration, root=root)
        observations = contract.load_json(Path(registration["observation_manifest"]["path"]), description="observation manifest")
        embedding, embedding_evidence = _load_embedding(registration, root=root, device=device)
        methods = {str(row["id"]): row for row in registration["methods"]}
        prediction_rows: dict[str, Any] = {}
        timing_rows: dict[str, Any] = {}
        model_startup: dict[str, Any] = {}
        for method_id in contract.METHOD_ORDER:
            row = methods[method_id]
            load_started = time.perf_counter()
            model, state_evidence = _load_decoder(row, root=root, device=device)
            state_evidence["model_preparation_seconds"] = float(time.perf_counter() - load_started)
            model_startup[method_id] = state_evidence
            for cell_id in contract.CELL_ORDER:
                records = contract.records_for_cell(observations, cell_id)
                cell = contract._as_cells(observations)[cell_id]
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                values = torch.empty((records, contract.STORED_SEQUENCE_TOKENS), dtype=torch.long)
                warm_sum = 0.0
                measured_sum = 0.0
                per_record: list[float] = []
                for row_index, activation, mask, _positions in _iter_rows(cell, records=records):
                    _guard(device=device, guard=registration["resource_guard"], started=started, stage=f"before_{method_id}_{cell_id}_{row_index}")
                    _synchronize(device)
                    t0 = time.perf_counter(); warm = predict_current_h(model, embedding, activation, mask, device=device); _synchronize(device)
                    warm_seconds = time.perf_counter() - t0
                    _synchronize(device)
                    t1 = time.perf_counter(); measured = predict_current_h(model, embedding, activation, mask, device=device); _synchronize(device)
                    measured_seconds = time.perf_counter() - t1
                    if not torch.equal(warm, measured):
                        raise RunnerError(f"warmup/measured IDs differ: {method_id}/{cell_id}/{row_index}")
                    values[row_index] = measured
                    warm_sum += warm_seconds; measured_sum += measured_seconds; per_record.append(float(measured_seconds))
                prediction_path = contract.expected_prediction_path(output_root, cell_id=cell_id, method_id=method_id)
                artifact = _write_prediction(prediction_path, prediction=values, registration=registration, cell_id=cell_id, method_id=method_id, records=records)
                timing_path = contract.expected_timing_path(output_root, cell_id=cell_id, method_id=method_id)
                timing_payload = {
                    "schema": contract.TIMING_SCHEMA,
                    "task_id": contract.TASK_ID,
                    "method_id": method_id,
                    "cell_id": cell_id,
                    "records": records,
                    "warmup_runs_per_record": 1,
                    "measured_runs_per_record": 1,
                    "warmup_seconds_sum": float(warm_sum),
                    "measured_seconds_sum": float(measured_sum),
                    "per_record_measured_seconds": per_record,
                    "warmup_output_exact_match_measured": True,
                    "measured_output_selected": True,
                    "timed_interval": "synchronized BF16 current-H staging -> FP32 decoder -> full-vocabulary argmax -> CPU IDs",
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
                timing_record = contract.write_create_only(timing_path, timing_payload)
                key = f"{method_id}::{cell_id}"
                prediction_rows[key] = artifact | {"method_id": method_id, "cell_id": cell_id}
                timing_rows[key] = timing_record | {"method_id": method_id, "cell_id": cell_id, "records": records, "payload": timing_payload}
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _code_and_input_recheck(registration, root=root, registration_path=registration_path, initial_registration_sha256=initial_registration_sha256)
        registration_record = {"path": str(registration_path), "bytes": registration_path.stat().st_size, "sha256": contract.sha256_file(registration_path)}
        result = {
            "schema": contract.RUN_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_PREDICTIONS_COMPLETE_BEFORE_TRUTH",
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "code_commit": registration["code_commit"],
            "registration": registration_record,
            "observation_manifest": registration["observation_manifest"],
            "predictions": prediction_rows,
            "timings": timing_rows,
            "model_startup": model_startup,
            "runtime_embedding": embedding_evidence,
            "numerical_settings": numerics,
            "truth_opened": False,
            "candidate_arrays_persisted": False,
        }
        run_path = output_root / "run_manifest.json"
        result["run_manifest"] = contract.write_create_only(run_path, result)
        return result
    except BaseException as exc:
        failure_path = registration_path.parent / "run_manifest.failure.json"
        try:
            contract.write_create_only(failure_path, {"schema": "token-reconstruction.trr0009-run-failure.v1", "task_id": contract.TASK_ID, "status": "PUBLIC_EVALUATION_FAILED_CLOSED", "truth_opened": False, "candidate_arrays_persisted": False, "started_utc": started_utc, "ended_utc": _utc_now(), "error_type": type(exc).__name__, "error": str(exc)})
        except Exception:
            pass
        if isinstance(exc, RunnerError):
            raise
        raise RunnerError("TRR-0009 evaluation failed closed") from exc
    finally:
        gc.collect()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute(registration_path=args.registration, repository_root=args.repository_root, device_name=args.device)
    except Exception as exc:
        print(f"TRR-0009 evaluation failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"], "run_manifest": result.get("run_manifest")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
