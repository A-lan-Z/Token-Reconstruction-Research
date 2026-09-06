"""TRR-0008 balanced warmed-runtime diagnostic.

The diagnostic reuses the exact TRR-0007 row-prediction boundary and only
opens public observations, frozen decoder states, and archived prediction
IDs.  It performs a full source-free exact-equivalence check before timing.
The five paths are then measured in fixed balanced blocks so a persistent
order or thermal effect is visible rather than absorbed into one sequential
method run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from scripts import trr0007_eval_contract as contract
from scripts import trr0007_eval_runner as frozen_runner


TASK_ID = "TRR-0008"
SCHEMA = "token-reconstruction.trr0008-balanced-timing.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0008-timing-failure.v1"
PARENT_COMMIT = "91925feb86dc03b912324b342a10282df27d79cd"
REFERENCE_METHOD_ID = contract.REFERENCE_METHOD_ID
METHOD_IDS = (
    contract.REFERENCE_METHOD_ID,
    *contract.STUDENT_METHOD_IDS,
)
CANDIDATE_METHOD_ID = "improved_public_bank__residual_mlp512"
TRR7_OBSERVATION_MANIFEST = "experiments/TRR-0007/evaluation/public_observations/observations.json"
TRR7_REGISTRATION = "experiments/TRR-0007/evaluation/registration.json"
TRR7_RUN_MANIFEST = "experiments/TRR-0007/evaluation/predictions/run_manifest.json"
TRR7_METHOD_FREEZE = "experiments/TRR-0007/method_freeze.json"

# These are deliberately fixed for the registered pilot.  Smaller values are
# useful only in isolated unit tests; a real receipt records any deviation.
DEFAULT_RECORDS_PER_CELL = 32
DEFAULT_BLOCKS = 10
DEFAULT_WARMUP_RUNS = 1
DEFAULT_SEED = 8008
DEFAULT_MAX_SECONDS = 600
DEFAULT_MAX_RSS_BYTES = 16 * 2**30
DEFAULT_MIN_HOST_AVAILABLE_BYTES = 10 * 2**30
DEFAULT_MIN_FREE_GPU_BYTES = 8 * 2**30
DEFAULT_MAX_RESERVED_GPU_BYTES = 6 * 2**30
QUALIFICATION_THRESHOLD = 1.25
CI_LEVEL = 0.95

_T_CRITICAL_95 = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.57058183564,
    6: 2.446911846,
    7: 2.364624251,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138851,
    11: 2.20098516,
    12: 2.17881283,
    13: 2.160368656,
    14: 2.144786688,
    15: 2.131449546,
    16: 2.119905299,
    17: 2.109815578,
    18: 2.10092204,
    19: 2.093024054,
    20: 2.085963447,
    21: 2.079613845,
    22: 2.073873068,
    23: 2.06865761,
    24: 2.063898562,
    25: 2.059538553,
    26: 2.055529439,
    27: 2.051830516,
    28: 2.048407142,
    29: 2.045229642,
}


class TimingError(RuntimeError):
    """Raised when the source-free timing contract cannot be certified."""


@dataclass(frozen=True)
class CellData:
    cell_id: str
    activations: torch.Tensor
    valid_mask: torch.Tensor
    position_ids: torch.Tensor
    descriptor: dict[str, Any]


@dataclass(frozen=True)
class TimingConfig:
    repository_root: Path
    trr7_root: Path
    output_path: Path
    device: str = "cuda"
    records_per_cell: int = DEFAULT_RECORDS_PER_CELL
    blocks: int = DEFAULT_BLOCKS
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    seed: int = DEFAULT_SEED
    maximum_seconds: int = DEFAULT_MAX_SECONDS


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _regular_file(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TimingError(f"{label} is not a regular file: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path, label="hashed asset").open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, label: str, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    record = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }
    if expected is not None:
        if int(expected.get("bytes", -1)) != record["bytes"] or str(expected.get("sha256")) != record["sha256"]:
            raise TimingError(f"{label} binding does not match {path}")
    return record


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TimingError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TimingError(f"{label} must be a JSON object")
    return value


def _resolve(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TimingError(f"{label} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return _regular_file(path, label=label)


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TimingError("cannot resolve executable source commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise TimingError("executable source commit is not a full hash")
    return value


def _configure_numerics() -> dict[str, Any]:
    settings = dict(contract.NUMERICAL_SETTINGS)
    try:
        torch.set_num_threads(int(settings["cpu_intraop_threads"]))
        torch.set_num_interop_threads(int(settings["cpu_interop_threads"]))
    except RuntimeError as exc:
        raise TimingError("unable to set the frozen CPU numerical settings") from exc
    torch.backends.cuda.matmul.allow_tf32 = bool(settings["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(settings["cuda_cudnn_allow_tf32"])
    torch.set_float32_matmul_precision(str(settings["float32_matmul_precision"]))
    return settings


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _host_available_bytes() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except OSError as exc:
        raise TimingError("host available-memory telemetry is unavailable") from exc
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    raise TimingError("host available-memory telemetry is unavailable")


def _load_average() -> list[float] | None:
    try:
        fields = Path("/proc/loadavg").read_text(encoding="ascii").split()
        return [float(value) for value in fields[:3]]
    except (OSError, ValueError):
        try:
            return [float(value) for value in os.getloadavg()]
        except (AttributeError, OSError):
            return None


def _nvidia_query(arguments: Sequence[str]) -> list[dict[str, str]]:
    command = ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TimingError("nvidia-smi telemetry is unavailable") from exc
    if result.returncode != 0:
        raise TimingError(f"nvidia-smi telemetry failed: {result.stderr.strip()}")
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if fields:
            rows.append({str(index): value for index, value in enumerate(fields)})
    return rows


def _gpu_telemetry(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise TimingError("CUDA was requested but is unavailable")
    rows = _nvidia_query(
        [
            "--query-gpu=index,uuid,temperature.gpu,utilization.gpu,memory.used,memory.free,power.draw",
        ]
    )
    if not rows:
        raise TimingError("nvidia-smi returned no GPU telemetry")
    index = str(device.index if device.index is not None else torch.cuda.current_device())
    selected = next((row for row in rows if row.get("0") == index), rows[0])
    free, total = torch.cuda.mem_get_info(device)
    return {
        "nvidia_smi": selected,
        "cuda_free_bytes": int(free),
        "cuda_total_bytes": int(total),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
    }


def _compute_apps() -> list[dict[str, str]]:
    rows = _nvidia_query(["--query-compute-apps=pid,process_name,used_memory"])
    result: list[dict[str, str]] = []
    for row in rows:
        if row.get("0") and row.get("0") != str(os.getpid()):
            result.append(
                {
                    "pid": row.get("0", ""),
                    "process_name": row.get("1", ""),
                    "used_memory": row.get("2", ""),
                }
            )
    return result


def _telemetry(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "utc": _utc_now(),
        "monotonic_seconds": time.perf_counter(),
        "process_max_rss_bytes": _rss_bytes(),
        "host_available_bytes": _host_available_bytes(),
        "load_average": _load_average(),
    }
    gpu = _gpu_telemetry(device)
    if gpu is not None:
        result["gpu"] = gpu
    return result


def _light_guard(*, started: float, maximum_seconds: int, stage: str) -> None:
    if time.perf_counter() - started > maximum_seconds:
        raise TimingError(f"wall-time guard expired at {stage}")
    if _host_available_bytes() < DEFAULT_MIN_HOST_AVAILABLE_BYTES:
        raise TimingError(f"host available-memory guard failed at {stage}")
    if _rss_bytes() > DEFAULT_MAX_RSS_BYTES:
        raise TimingError(f"host RSS guard failed at {stage}")


def _guard(device: torch.device, *, started: float, maximum_seconds: int, stage: str) -> dict[str, Any]:
    if time.perf_counter() - started > maximum_seconds:
        raise TimingError(f"wall-time guard expired at {stage}")
    available = _host_available_bytes()
    rss = _rss_bytes()
    if available < DEFAULT_MIN_HOST_AVAILABLE_BYTES:
        raise TimingError(f"host available-memory guard failed at {stage}: {available}")
    if rss > DEFAULT_MAX_RSS_BYTES:
        raise TimingError(f"host RSS guard failed at {stage}: {rss}")
    result: dict[str, Any] = {
        "stage": stage,
        "status": "PASS",
        "elapsed_seconds": float(time.perf_counter() - started),
        "host_available_bytes": available,
        "process_max_rss_bytes": rss,
    }
    if device.type == "cuda":
        gpu = _gpu_telemetry(device)
        assert gpu is not None
        apps = _compute_apps()
        if apps:
            raise TimingError(f"GPU is not exclusive at {stage}: {apps!r}")
        if int(gpu["cuda_free_bytes"]) < DEFAULT_MIN_FREE_GPU_BYTES:
            raise TimingError(f"GPU free-memory guard failed at {stage}")
        if int(gpu["cuda_reserved_bytes"]) > DEFAULT_MAX_RESERVED_GPU_BYTES:
            raise TimingError(f"GPU reservation guard failed at {stage}")
        result["gpu"] = gpu
        result["compute_apps"] = apps
    return result


def _validate_registration(config: TimingConfig) -> dict[str, Any]:
    registration_path = config.trr7_root / TRR7_REGISTRATION
    registration = _load_json(registration_path, label="TRR-0007 registration")
    if registration.get("task_id") != "TRR-0007":
        raise TimingError("registration task ID changed")
    if registration.get("truth_opened") is not False or registration.get("source_text_or_target_labels") is not False:
        raise TimingError("registration indicates truth or source access")
    if tuple(registration.get("cell_order", ())) != contract.CELL_ORDER:
        raise TimingError("registration cell order changed")
    registered_ids = tuple(registration.get("method_ids", ()))
    if not set(METHOD_IDS).issubset(registered_ids):
        raise TimingError("registration does not contain the five frozen timing methods")
    code_bindings = registration.get("code_bindings")
    if not isinstance(code_bindings, list):
        raise TimingError("registration code bindings are missing")
    binding_map = {row.get("role"): row for row in code_bindings if isinstance(row, Mapping)}
    required_bindings = {
        "evaluation_contract": "scripts/trr0007_eval_contract.py",
        "evaluation_runner": "scripts/trr0007_eval_runner.py",
        "retained_decoder_numerics": "src/token_reconstruction/trr0005_joint_decoder.py",
        "positionwise_numerics": "src/token_reconstruction/trr0007_positionwise.py",
    }
    verified_code: dict[str, Any] = {}
    for role, relative in required_bindings.items():
        row = binding_map.get(role)
        if not isinstance(row, Mapping) or row.get("path") != relative:
            raise TimingError(f"missing registered code binding: {role}")
        verified_code[role] = _file_record(
            config.repository_root / relative,
            label=f"registered code {role}",
            expected=row,
        )
    method_rows = {str(row.get("id")): row for row in registration.get("methods", ()) if isinstance(row, Mapping)}
    selected_rows: dict[str, dict[str, Any]] = {}
    for method_id in METHOD_IDS:
        row = method_rows.get(method_id)
        if row is None or row.get("kind") != "decoder":
            raise TimingError(f"frozen method row is missing: {method_id}")
        selected_rows[method_id] = dict(row)
        _file_record(
            _resolve(row.get("state", {}).get("path"), base=config.trr7_root, label=f"state {method_id}"),
            label=f"state {method_id}",
            expected=row["state"],
        )
    method_freeze_binding = registration.get("method_freeze")
    if not isinstance(method_freeze_binding, Mapping):
        raise TimingError("method-freeze binding is missing")
    method_freeze_path = _resolve(
        method_freeze_binding.get("path"), base=config.trr7_root, label="method freeze"
    )
    method_freeze_file = _file_record(
        method_freeze_path, label="method freeze", expected=method_freeze_binding
    )
    method_freeze = _load_json(method_freeze_path, label="method freeze")
    if (
        method_freeze.get("status") != "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION"
        or method_freeze.get("source_accessed") is not False
        or method_freeze.get("target_loaded") is not False
        or method_freeze.get("fresh_evaluation_started") is not False
    ):
        raise TimingError("method freeze is not a source-free pre-evaluation freeze")
    reference_freeze = method_freeze.get("retained_reference")
    if not isinstance(reference_freeze, Mapping) or reference_freeze.get("sha256") != selected_rows[REFERENCE_METHOD_ID]["state"]["sha256"]:
        raise TimingError("retained reference state differs from method freeze")
    frozen_state_hashes = registration.get("method_freeze_state_sha256")
    state_bindings = method_freeze.get("state_bindings")
    if not isinstance(frozen_state_hashes, Mapping) or not isinstance(state_bindings, Mapping):
        raise TimingError("method-freeze selected-state hashes are missing")
    for method_id in contract.STUDENT_METHOD_IDS:
        freeze_row = state_bindings.get(method_id)
        if not isinstance(freeze_row, Mapping) or freeze_row.get("state_sha256") != selected_rows[method_id]["state"]["sha256"]:
            raise TimingError(f"student state differs from method freeze: {method_id}")
        if frozen_state_hashes.get(method_id) != selected_rows[method_id]["state"]["sha256"]:
            raise TimingError(f"registration state digest differs from method freeze: {method_id}")

    observation_binding = registration.get("observation_manifest")
    if not isinstance(observation_binding, Mapping):
        raise TimingError("observation manifest binding is missing")
    observation_path = _resolve(
        observation_binding.get("path"), base=config.trr7_root, label="observation manifest"
    )
    verified_observation = _file_record(
        observation_path,
        label="observation manifest",
        expected=observation_binding,
    )
    return {
        "path": str(registration_path.resolve()),
        "file": _file_record(registration_path, label="TRR-0007 registration"),
        "registration": registration,
        "methods": selected_rows,
        "method_freeze_path": str(method_freeze_path),
        "method_freeze_file": method_freeze_file,
        "observation_path": str(observation_path),
        "observation_file": verified_observation,
        "code": verified_code,
    }


def _load_observations(
    observation_path: Path,
    *,
    records: int,
    cell_ids: Sequence[str] = contract.CELL_ORDER,
    repository_root: Path | None = None,
) -> dict[str, CellData]:
    manifest = _load_json(observation_path, label="public observation manifest")
    if manifest.get("truth_opened") is not False or manifest.get("source_text_loaded") is not False:
        raise TimingError("observation manifest indicates truth/source access")
    if manifest.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise TimingError("observation manifest status changed")
    cell_rows = {str(row.get("cell_id")): row for row in manifest.get("cells", ()) if isinstance(row, Mapping)}
    result: dict[str, CellData] = {}
    for cell_id in cell_ids:
        row = cell_rows.get(cell_id)
        if row is None or not isinstance(row.get("observation"), Mapping):
            raise TimingError(f"observation cell missing: {cell_id}")
        descriptor = dict(row["observation"])
        base_root = repository_root if repository_root is not None else observation_path.parents[4]
        path = _resolve(descriptor.get("path"), base=base_root, label=f"observation {cell_id}")
        _file_record(path, label=f"observation {cell_id}", expected=descriptor)
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                required = {"activations", "attention_mask", "position_ids"}
                if not required.issubset(keys):
                    raise TimingError(f"observation {cell_id} lacks required tensors")
                forbidden = {"tokens", "truth", "source", "input_ids", "labels", "target"}
                if keys.intersection(forbidden):
                    raise TimingError(f"observation {cell_id} contains a forbidden payload")
                activations = handle.get_tensor("activations")[:records].contiguous()
                valid_mask = handle.get_tensor("attention_mask")[:records].to(dtype=torch.bool).contiguous()
                position_ids = handle.get_tensor("position_ids")[:records].to(dtype=torch.long).contiguous()
        except TimingError:
            raise
        except Exception as exc:
            raise TimingError(f"observation {cell_id} could not be read") from exc
        expected_shape = (records, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE)
        if tuple(activations.shape) != expected_shape or activations.dtype != torch.bfloat16:
            raise TimingError(f"observation {cell_id} activation geometry or dtype changed")
        if tuple(valid_mask.shape) != expected_shape[:2] or tuple(position_ids.shape) != expected_shape[:2]:
            raise TimingError(f"observation {cell_id} mask/position geometry changed")
        expected_positions = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.long).expand(records, -1)
        if not torch.equal(position_ids, expected_positions):
            raise TimingError(f"observation {cell_id} position IDs changed")
        if not valid_mask[:, 0].all().item():
            raise TimingError(f"observation {cell_id} has an invalid BOS mask")
        result[cell_id] = CellData(cell_id, activations, valid_mask, position_ids, descriptor)
    return result


def _load_embedding(registration: Mapping[str, Any], *, repository_root: Path, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    assets = registration["registration"].get("runtime_assets")
    binding = assets.get("normalized_public_E") if isinstance(assets, Mapping) else None
    if not isinstance(binding, Mapping):
        raise TimingError("normalized public embedding binding is missing")
    path = _resolve(binding.get("path"), base=repository_root, label="normalized public embedding")
    record = _file_record(path, label="normalized public embedding", expected=binding)
    started = time.perf_counter()
    try:
        values = load_file(str(path), device="cpu")
        if set(values) != {"embeddings"}:
            raise TimingError("normalized public embedding contains unexpected tensors")
        table_cpu = values["embeddings"].detach().contiguous()
        if table_cpu.dtype != torch.float32 or tuple(table_cpu.shape) != (contract.VOCAB_SIZE, contract.HIDDEN_SIZE):
            raise TimingError("normalized public embedding geometry or dtype changed")
        table = table_cpu.to(device=device).contiguous()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except TimingError:
        raise
    except Exception as exc:
        raise TimingError("normalized public embedding could not be loaded") from exc
    return table, {**record, "shape": [contract.VOCAB_SIZE, contract.HIDDEN_SIZE], "dtype": "torch.float32", "load_seconds": time.perf_counter() - started}


def _load_models(
    registration: Mapping[str, Any],
    *,
    trr7_root: Path,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    models: dict[str, torch.nn.Module] = {}
    evidence: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        row = registration["methods"][method_id]
        state_binding = row["state"]
        state_path = _resolve(state_binding.get("path"), base=trr7_root, label=f"state {method_id}")
        loader = row.get("loader")
        if not isinstance(loader, Mapping):
            raise TimingError(f"loader binding is missing: {method_id}")
        started = time.perf_counter()
        try:
            module = importlib.import_module(str(loader["module"]))
            function = getattr(module, str(loader["function"]))
            model = function(state_path, **dict(loader.get("kwargs", {})))
            if not isinstance(model, torch.nn.Module):
                raise TypeError("loader did not return a torch module")
            model.requires_grad_(False)
            model = model.to(device=device).eval()
            frozen_runner._synchronize(device)
        except Exception as exc:
            raise TimingError(f"frozen model could not be loaded: {method_id}") from exc
        models[method_id] = model
        evidence[method_id] = {
            "state": _file_record(state_path, label=f"state {method_id}", expected=state_binding),
            "loader": f"{loader['module']}.{loader['function']}",
            "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
            "parameter_count": int(sum(int(value.numel()) for value in model.parameters())),
            "load_seconds": float(time.perf_counter() - started),
        }
    return models, evidence


def _verify_alias_execution_identity(
    models: Mapping[str, torch.nn.Module], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the registered alias shares weights and class with the reference."""

    alias_id = "current_enriched__trained_diagonal"
    reference = models[REFERENCE_METHOD_ID]
    alias = models[alias_id]
    reference_state = {name: value.detach().cpu().contiguous() for name, value in reference.state_dict().items()}
    alias_state = {name: value.detach().cpu().contiguous() for name, value in alias.state_dict().items()}
    state_equal = set(reference_state) == set(alias_state) and all(
        torch.equal(reference_state[name], alias_state[name]) for name in reference_state
    )
    class_equal = (
        reference.__class__.__module__ == alias.__class__.__module__
        and reference.__class__.__qualname__ == alias.__class__.__qualname__
    )
    if not state_equal or not class_equal:
        raise TimingError("registered current alias is not an identical-weight, identical-class execution path")
    return {
        "reference_class": f"{reference.__class__.__module__}.{reference.__class__.__qualname__}",
        "alias_class": f"{alias.__class__.__module__}.{alias.__class__.__qualname__}",
        "class_exact": class_equal,
        "state_keys_exact": set(reference_state) == set(alias_state),
        "state_tensors_exact": state_equal,
        "state_hash_reference": evidence[REFERENCE_METHOD_ID]["state"]["sha256"],
        "state_hash_alias": evidence[alias_id]["state"]["sha256"],
    }


def _load_archived_prediction(
    run_manifest: Mapping[str, Any], *, trr7_root: Path, method_id: str, cell_id: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    entries = run_manifest.get("predictions")
    key = f"{method_id}::{cell_id}"
    entry = entries.get(key) if isinstance(entries, Mapping) else None
    if not isinstance(entry, Mapping):
        raise TimingError(f"archived prediction entry is missing: {key}")
    artifact = entry.get("prediction_artifact")
    if not isinstance(artifact, Mapping):
        raise TimingError(f"archived prediction artifact is missing: {key}")
    path = _resolve(artifact.get("path"), base=trr7_root, label=f"archived prediction {key}")
    record = _file_record(path, label=f"archived prediction {key}", expected=artifact)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"predictions"}:
                raise TimingError(f"archived prediction {key} has unexpected tensors")
            prediction = handle.get_tensor("predictions").to(dtype=torch.long).contiguous()
    except TimingError:
        raise
    except Exception as exc:
        raise TimingError(f"archived prediction {key} could not be read") from exc
    if tuple(prediction.shape) != (contract.RECORDS_PER_DOMAIN, contract.STORED_SEQUENCE_TOKENS):
        raise TimingError(f"archived prediction {key} geometry changed")
    if contract.tensor_digest(prediction) != entry.get("prediction_sha256"):
        raise TimingError(f"archived prediction {key} tensor digest changed")
    return prediction, {**record, "tensor_sha256": contract.tensor_digest(prediction)}


def _equivalence_check(
    models: Mapping[str, torch.nn.Module],
    embedding: torch.Tensor,
    cells: Mapping[str, CellData],
    archived: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    outputs: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        for cell_id in contract.CELL_ORDER:
            cell = cells[cell_id]
            target = archived[f"{method_id}::{cell_id}"]
            actual = torch.empty_like(target)
            for row_index in range(contract.RECORDS_PER_DOMAIN):
                frozen_runner._synchronize(device)
                prediction = frozen_runner._predict_decoder_row(
                    models[method_id], embedding, cell.activations[row_index], cell.valid_mask[row_index], device=device
                )
                if not torch.equal(prediction, target[row_index]):
                    mismatch = torch.nonzero(prediction.ne(target[row_index]), as_tuple=False)
                    first = mismatch[0].tolist() if int(mismatch.shape[0]) else []
                    raise TimingError(f"archived prediction mismatch before timing: {method_id}/{cell_id}/{first}")
                actual[row_index] = prediction
            outputs[f"{method_id}::{cell_id}"] = {
                "records": contract.RECORDS_PER_DOMAIN,
                "prediction_sha256": contract.tensor_digest(actual),
                "exact_match_archived": True,
            }
    reference = archived
    alias_matches_reference: dict[str, bool] = {}
    alias = "current_enriched__trained_diagonal"
    for cell_id in contract.CELL_ORDER:
        alias_matches_reference[cell_id] = torch.equal(
            reference[f"{REFERENCE_METHOD_ID}::{cell_id}"],
            reference[f"{alias}::{cell_id}"],
        )
    if not all(alias_matches_reference.values()):
        raise TimingError("registered current alias is not exactly equal to archived reference predictions")
    outputs["alias_matches_reference"] = alias_matches_reference
    return {"status": "PASS", "elapsed_seconds": time.perf_counter() - started, "entries": outputs}


def _balanced_orders(*, method_ids: Sequence[str], cell_ids: Sequence[str], blocks: int, seed: int) -> list[dict[str, Any]]:
    """Build a deterministic cyclic schedule with no outcome-dependent order."""

    if not method_ids or blocks <= 0 or not cell_ids:
        raise ValueError("method IDs, cells, and blocks must be non-empty")
    method_ids = tuple(method_ids)
    orders: list[dict[str, Any]] = []
    # The seed affects a fixed permutation only.  It is recorded before timing
    # and never changed in response to an observed duration.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(len(method_ids), generator=generator).tolist()
    permuted = tuple(method_ids[index] for index in permutation)
    for block_index in range(blocks):
        for cell_index, cell_id in enumerate(cell_ids):
            offset = (block_index + cell_index) % len(permuted)
            rotated = permuted[offset:] + permuted[:offset]
            # Ten registered blocks are five rotations followed by their
            # reversals.  Every method therefore occupies every order
            # position exactly twice per cell.
            order = rotated if block_index < len(permuted) else tuple(reversed(rotated))
            orders.append(
                {
                    "block_index": block_index,
                    "cell_index": cell_index,
                    "cell_id": cell_id,
                    "order": list(order),
                }
            )
    encoded = json.dumps(orders, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return [{"schedule": orders, "sha256": hashlib.sha256(encoded).hexdigest()}]


def _schedule_rows(*, method_ids: Sequence[str], cell_ids: Sequence[str], blocks: int, seed: int) -> tuple[list[dict[str, Any]], str]:
    wrapped = _balanced_orders(method_ids=method_ids, cell_ids=cell_ids, blocks=blocks, seed=seed)[0]
    return list(wrapped["schedule"]), str(wrapped["sha256"])


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio_summary(values: Sequence[float], *, threshold: float = QUALIFICATION_THRESHOLD) -> dict[str, Any]:
    values = [float(value) for value in values]
    if len(values) < 2:
        raise ValueError("at least two blocks are required for a confidence interval")
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)
    df = len(values) - 1
    critical = _T_CRITICAL_95.get(df, 1.96)
    margin = critical * stdev / math.sqrt(len(values))
    lower, upper = mean - margin, mean + margin
    if upper <= threshold:
        decision = "PASS"
    elif lower > threshold:
        decision = "FAIL"
    else:
        decision = "INCONCLUSIVE"
    return {
        "blocks": len(values),
        "block_ratios": values,
        "mean_ratio": mean,
        "stdev_ratio": stdev,
        "ci_level": CI_LEVEL,
        "ci_lower": lower,
        "ci_upper": upper,
        "threshold": threshold,
        "decision": decision,
    }


def _method_variability(seconds: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in seconds]
    if not values:
        raise ValueError("method has no measured rows")
    return {
        "records": len(values),
        "mean_seconds": statistics.fmean(values),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "p05_seconds": _quantile(values, 0.05),
        "p50_seconds": _quantile(values, 0.50),
        "p95_seconds": _quantile(values, 0.95),
        "max_seconds": max(values),
    }


def _run_timed_cell(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    cell: CellData,
    *,
    device: torch.device,
    warmup_runs: int,
    method_id: str,
    block_index: int,
    started: float,
    maximum_seconds: int,
) -> dict[str, Any]:
    if warmup_runs != 1:
        raise TimingError("the registered timing contract requires exactly one warmup run")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    warmup_seconds: list[float] = []
    measured_seconds: list[float] = []
    outputs: list[torch.Tensor] = []
    for row_index in range(int(cell.activations.shape[0])):
        # Full GPU/process polling is performed at method-cell boundaries by
        # _timed_blocks.  This lightweight elapsed-time check between rows
        # keeps the fail-closed wall bound without inserting nvidia-smi calls
        # into the measured workload.
        if row_index % 8 == 0:
            _light_guard(started=started, maximum_seconds=maximum_seconds, stage=f"row_{method_id}_{cell.cell_id}_{block_index}_{row_index}")
        frozen_runner._synchronize(device)
        t0 = time.perf_counter()
        warm = frozen_runner._predict_decoder_row(
            model, embedding, cell.activations[row_index], cell.valid_mask[row_index], device=device
        )
        frozen_runner._synchronize(device)
        warmup_seconds.append(float(time.perf_counter() - t0))
        frozen_runner._synchronize(device)
        t1 = time.perf_counter()
        measured = frozen_runner._predict_decoder_row(
            model, embedding, cell.activations[row_index], cell.valid_mask[row_index], device=device
        )
        frozen_runner._synchronize(device)
        measured_seconds.append(float(time.perf_counter() - t1))
        if not torch.equal(warm, measured):
            raise TimingError(f"warmup/measured prediction mismatch: {method_id}/{cell.cell_id}/{row_index}")
        outputs.append(measured)
    if device.type == "cuda":
        peak = {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    else:
        peak = {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
    measured_tensor = torch.stack(outputs, dim=0)
    return {
        "method_id": method_id,
        "cell_id": cell.cell_id,
        "block_index": block_index,
        "records": int(cell.activations.shape[0]),
        "warmup_runs_per_record": warmup_runs,
        "measured_runs_per_record": 1,
        "warmup_seconds_sum": float(sum(warmup_seconds)),
        "measured_seconds_sum": float(sum(measured_seconds)),
        "warmup_variability": _method_variability(warmup_seconds),
        "measured_variability": _method_variability(measured_seconds),
        "per_record_measured_seconds": measured_seconds,
        "warmup_output_exact_match_measured": True,
        "prediction_sha256": contract.tensor_digest(measured_tensor),
        "peak_memory": {**peak, "process_max_rss_bytes": _rss_bytes()},
        "steady_interval": "TRR-0007 _predict_decoder_row with explicit device synchronization",
    }


def _timed_blocks(
    models: Mapping[str, torch.nn.Module],
    embedding: torch.Tensor,
    cells: Mapping[str, CellData],
    *,
    orders: Sequence[Mapping[str, Any]],
    device: torch.device,
    config: TimingConfig,
    started: float,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block_index in range(config.blocks):
        _guard(device, started=started, maximum_seconds=config.maximum_seconds, stage=f"before_block_{block_index}")
        block_started = time.perf_counter()
        before = _telemetry(device)
        cell_orders = [row for row in orders if int(row["block_index"]) == block_index]
        if len(cell_orders) != len(contract.CELL_ORDER):
            raise TimingError(f"order schedule is incomplete for block {block_index}")
        entries: list[dict[str, Any]] = []
        for order_row in cell_orders:
            cell_id = str(order_row["cell_id"])
            cell = cells[cell_id]
            for order_index, method_id in enumerate(order_row["order"]):
                method_id = str(method_id)
                guard_started = time.perf_counter()
                guard_before = _guard(
                    device,
                    started=started,
                    maximum_seconds=config.maximum_seconds,
                    stage=f"before_method_cell_{method_id}_{cell_id}_{block_index}",
                )
                guard_before_seconds = time.perf_counter() - guard_started
                entry = _run_timed_cell(
                    models[method_id],
                    embedding,
                    cell,
                    device=device,
                    warmup_runs=config.warmup_runs,
                    method_id=method_id,
                    block_index=block_index,
                    started=started,
                    maximum_seconds=config.maximum_seconds,
                )
                guard_started = time.perf_counter()
                guard_after = _guard(
                    device,
                    started=started,
                    maximum_seconds=config.maximum_seconds,
                    stage=f"after_method_cell_{method_id}_{cell_id}_{block_index}",
                )
                guard_after_seconds = time.perf_counter() - guard_started
                entry["order_index"] = order_index
                entry["resource_guard_before"] = guard_before
                entry["resource_guard_after"] = guard_after
                entry["resource_guard_overhead_seconds"] = guard_before_seconds + guard_after_seconds
                entries.append(entry)
        frozen_runner._synchronize(device)
        after = _telemetry(device)
        blocks.append(
            {
                "block_index": block_index,
                "order_by_cell": {str(row["cell_id"]): list(row["order"]) for row in cell_orders},
                "started_utc": before["utc"],
                "ended_utc": after["utc"],
                "wall_seconds": float(time.perf_counter() - block_started),
                "telemetry_before": before,
                "telemetry_after": after,
                "entries": entries,
            }
        )
        _guard(device, started=started, maximum_seconds=config.maximum_seconds, stage=f"after_block_{block_index}")
    return blocks


def _summarize_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Keep totals indexed by block/cell/method so the candidate decision never
    # pools cells with different activation/target conditions.
    totals: dict[int, dict[str, dict[str, float]]] = {}
    for block in blocks:
        block_index = int(block["block_index"])
        per_cell: dict[str, dict[str, float]] = {}
        for entry in block["entries"]:
            cell_id = str(entry["cell_id"])
            method_id = str(entry["method_id"])
            per_cell.setdefault(cell_id, {})[method_id] = float(entry["measured_seconds_sum"])
        totals[block_index] = per_cell

    comparison: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        if method_id == REFERENCE_METHOD_ID:
            continue
        by_cell: dict[str, dict[str, Any]] = {}
        pooled_ratios: list[float] = []
        for cell_id in contract.CELL_ORDER:
            ratios: list[float] = []
            for block_index in sorted(totals):
                values = totals[block_index].get(cell_id, {})
                denominator = values.get(REFERENCE_METHOD_ID, 0.0)
                numerator = values.get(method_id, 0.0)
                if denominator <= 0.0 or numerator <= 0.0:
                    raise TimingError(f"missing timing total for {method_id}/{cell_id}/{block_index}")
                ratios.append(numerator / denominator)
                pooled_values = totals[block_index]
                pooled_denominator = sum(
                    pooled_values.get(other_cell, {}).get(REFERENCE_METHOD_ID, 0.0)
                    for other_cell in contract.CELL_ORDER
                )
                pooled_numerator = sum(
                    pooled_values.get(other_cell, {}).get(method_id, 0.0)
                    for other_cell in contract.CELL_ORDER
                )
                if pooled_denominator <= 0.0:
                    raise TimingError(f"reference timing total is zero in block {block_index}")
                if cell_id == contract.CELL_ORDER[0]:
                    pooled_ratios.append(pooled_numerator / pooled_denominator)
            by_cell[cell_id] = _ratio_summary(ratios)
        comparison[method_id] = {
            "by_cell": by_cell,
            "pooled_descriptive": _ratio_summary(pooled_ratios),
            "pooled_used_for_qualification": False,
        }

    candidate_cells = comparison[CANDIDATE_METHOD_ID]["by_cell"]
    failed_cells = [cell_id for cell_id, value in candidate_cells.items() if value["ci_lower"] > QUALIFICATION_THRESHOLD]
    passed_cells = [cell_id for cell_id, value in candidate_cells.items() if value["ci_upper"] <= QUALIFICATION_THRESHOLD]
    if failed_cells:
        candidate_decision = "FAIL"
    elif len(passed_cells) == len(contract.CELL_ORDER):
        candidate_decision = "PASS"
    else:
        candidate_decision = "INCONCLUSIVE"

    alias_id = "current_enriched__trained_diagonal"
    alias_cells = comparison[alias_id]["by_cell"]
    alias_invalid_cells = [
        cell_id
        for cell_id, value in alias_cells.items()
        if value["ci_lower"] > 1.05 or value["ci_upper"] < 0.95
    ]
    alias_pass_cells = [
        cell_id
        for cell_id, value in alias_cells.items()
        if value["ci_lower"] >= 0.95 and value["ci_upper"] <= 1.05
    ]
    if alias_invalid_cells:
        alias_decision = "FAIL"
    elif len(alias_pass_cells) == len(contract.CELL_ORDER):
        alias_decision = "PASS"
    else:
        alias_decision = "INCONCLUSIVE"
    alias_control = {
        "method_id": alias_id,
        "exact_prediction_equivalence_required": True,
        "runtime_ratio_band": [0.95, 1.05],
        "invalid_persistent_deviation_cells": alias_invalid_cells,
        "fully_contained_cells": alias_pass_cells,
        "runtime_order_control_valid": alias_decision == "PASS",
        "decision": alias_decision,
        "qualification_role": "required_order_control_for_candidate_cost_claim",
    }
    raw_candidate_decision = candidate_decision
    if alias_decision == "FAIL":
        # The timing path is invalid as a comparative measurement.  Do not
        # call this a candidate budget failure when the identical-weight order
        # control itself demonstrates a persistent deviation.
        candidate_decision = "INVALID_ALIAS_CONTROL"
    elif alias_decision != "PASS":
        # A wide or otherwise inconclusive alias interval cannot support either
        # candidate outcome, even when the raw candidate interval is decisive.
        candidate_decision = "INCONCLUSIVE"

    all_entries = [entry for block in blocks for entry in block["entries"]]
    method_record_times: dict[str, list[float]] = {method_id: [] for method_id in METHOD_IDS}
    for entry in all_entries:
        method_record_times[str(entry["method_id"])].extend(entry["per_record_measured_seconds"])
    return {
        "reference_method_id": REFERENCE_METHOD_ID,
        "candidate_method_id": CANDIDATE_METHOD_ID,
        "qualification": {
            "decision": candidate_decision,
            "raw_per_cell_decision": raw_candidate_decision,
            "alias_control_required": True,
            "measurement_valid": alias_decision == "PASS",
            "cost_failure_demonstrated": alias_decision == "PASS" and raw_candidate_decision == "FAIL",
            "threshold": QUALIFICATION_THRESHOLD,
            "per_cell": candidate_cells,
            "failed_cells": failed_cells,
            "passed_cells": passed_cells,
            "pooled_descriptive": comparison[CANDIDATE_METHOD_ID]["pooled_descriptive"],
        },
        "comparisons_vs_reference": comparison,
        "alias_control": alias_control,
        "record_variability": {
            method_id: _method_variability(values) for method_id, values in method_record_times.items()
        },
        "block_total_seconds": totals,
    }


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TimingError(f"output is create-only and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _failure_receipt(config: TimingConfig, *, started_utc: str, exc: BaseException) -> dict[str, Any]:
    return {
        "schema": FAILURE_SCHEMA,
        "task_id": TASK_ID,
        "status": "FAILED_CLOSED",
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "repository_root": str(config.repository_root),
        "trr7_root": str(config.trr7_root),
        "output_path": str(config.output_path),
        "device": config.device,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "truth_opened": False,
        "source_text_or_target_labels": False,
    }


def run(config: TimingConfig) -> dict[str, Any]:
    started_utc = _utc_now()
    started = time.perf_counter()
    if config.records_per_cell <= 0 or config.records_per_cell > contract.RECORDS_PER_DOMAIN:
        raise TimingError("timing record count must be in 1..128")
    if config.blocks <= 1:
        raise TimingError("at least two timing blocks are required")
    if config.warmup_runs != 1:
        raise TimingError("the frozen timing contract requires one warmup per record")
    try:
        numerics = _configure_numerics()
        device = torch.device(config.device)
        registration = _validate_registration(config)
        _guard(device, started=started, maximum_seconds=config.maximum_seconds, stage="initial")
        embedding, embedding_evidence = _load_embedding(registration, repository_root=config.repository_root, device=device)
        cells = _load_observations(
            Path(registration["observation_path"]),
            records=contract.RECORDS_PER_DOMAIN,
            repository_root=config.repository_root,
        )
        models, model_evidence = _load_models(registration, trr7_root=config.trr7_root, device=device)
        alias_identity = _verify_alias_execution_identity(models, model_evidence)
        run_manifest_path = config.trr7_root / TRR7_RUN_MANIFEST
        run_manifest = _load_json(run_manifest_path, label="TRR-0007 prediction run manifest")
        if (
            run_manifest.get("status") != "COMPLETE_PUBLIC_PREDICTIONS_NO_TRUTH"
            or run_manifest.get("truth_opened") is not False
            or run_manifest.get("candidate_arrays_persisted") is not False
            or run_manifest.get("predictions_complete") is not True
        ):
            raise TimingError("archived run manifest is not a complete source-free prediction receipt")
        archived: dict[str, torch.Tensor] = {}
        archived_evidence: dict[str, Any] = {}
        for method_id in METHOD_IDS:
            for cell_id in contract.CELL_ORDER:
                tensor, evidence = _load_archived_prediction(
                    run_manifest, trr7_root=config.trr7_root, method_id=method_id, cell_id=cell_id
                )
                archived[f"{method_id}::{cell_id}"] = tensor
                archived_evidence[f"{method_id}::{cell_id}"] = evidence
        equivalence = _equivalence_check(models, embedding, cells, archived, device=device)
        # Timing rows are a fixed prefix of the already validated full cells.
        timed_cells = {
            cell_id: CellData(
                cell.cell_id,
                cell.activations[: config.records_per_cell],
                cell.valid_mask[: config.records_per_cell],
                cell.position_ids[: config.records_per_cell],
                cell.descriptor,
            )
            for cell_id, cell in cells.items()
        }
        orders, schedule_sha256 = _schedule_rows(
            method_ids=METHOD_IDS, cell_ids=contract.CELL_ORDER, blocks=config.blocks, seed=config.seed
        )
        timing_blocks = _timed_blocks(
            models,
            embedding,
            timed_cells,
            orders=orders,
            device=device,
            config=config,
            started=started,
        )
        summary = _summarize_blocks(timing_blocks)
        result = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "status": "TIMING_COMPLETE",
            "truth_opened": False,
            "source_text_or_target_labels": False,
            "candidate_arrays_persisted": False,
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "code_commit": _git_head(config.repository_root),
            "parent_commit": PARENT_COMMIT,
            "repository_root": str(config.repository_root),
            "trr7_root": str(config.trr7_root),
            "registration": registration,
            "numerical_settings": numerics,
            "configuration": {
                "device": str(device),
                "records_per_cell": config.records_per_cell,
                "equivalence_records_per_cell": contract.RECORDS_PER_DOMAIN,
                "blocks": config.blocks,
                "warmup_runs_per_record": config.warmup_runs,
                "seed": config.seed,
                "maximum_seconds": config.maximum_seconds,
                "threshold": QUALIFICATION_THRESHOLD,
                "ci_level": CI_LEVEL,
            },
            "runtime_embedding": embedding_evidence,
            "model_startup": model_evidence,
            "alias_execution_identity": alias_identity,
            "archived_predictions": archived_evidence,
            "equivalence": equivalence,
            "order_schedule": {"rows": orders, "sha256": schedule_sha256},
            "blocks": timing_blocks,
            "summary": summary,
            "resource_guard": {
                "minimum_free_gpu_bytes": DEFAULT_MIN_FREE_GPU_BYTES,
                "maximum_reserved_gpu_bytes": DEFAULT_MAX_RESERVED_GPU_BYTES,
                "maximum_rss_bytes": DEFAULT_MAX_RSS_BYTES,
                "minimum_host_available_bytes": DEFAULT_MIN_HOST_AVAILABLE_BYTES,
                "maximum_seconds": config.maximum_seconds,
            },
        }
        _write_create_only(config.output_path, result)
        return result
    except BaseException as exc:
        failure_path = config.output_path.with_name(config.output_path.stem + ".failure.json")
        try:
            _write_create_only(failure_path, _failure_receipt(config, started_utc=started_utc, exc=exc))
        except Exception:
            pass
        if isinstance(exc, TimingError):
            raise
        raise TimingError("TRR-0008 timing failed closed") from exc
    finally:
        gc.collect()
        if "device" in locals() and isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = Path(__file__).resolve().parents[1]
    default_trr7_root = repository_root.parent / "TRR-0007"
    if not default_trr7_root.is_dir():
        default_trr7_root = repository_root / ".worktrees" / "TRR-0007"
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--trr7-root", type=Path, default=default_trr7_root)
    parser.add_argument("--output", type=Path, default=repository_root / "experiments" / "TRR-0008" / "timing" / "result.json")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--records-per-cell", type=int, default=DEFAULT_RECORDS_PER_CELL)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maximum-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = TimingConfig(
        repository_root=args.repository_root.expanduser().resolve(),
        trr7_root=args.trr7_root.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        device=args.device,
        records_per_cell=args.records_per_cell,
        blocks=args.blocks,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        maximum_seconds=args.maximum_seconds,
    )
    try:
        result = run(config)
    except Exception as exc:
        print(f"TRR-0008 timing failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(config.output_path), "summary": result["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
