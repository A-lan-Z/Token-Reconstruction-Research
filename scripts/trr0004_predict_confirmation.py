#!/usr/bin/env python3
"""Run the frozen TRR-0004 public five-method confirmation matrix.

This driver is intentionally an execution adapter, rather than another
selection or scoring framework.  The public panel, selection plan, method
registration, and every method state are loaded through
``trr0004_fresh_confirmation``.  The driver then runs the five registered
methods on the same four cells.  Each record has one warmup call and three
measured calls; the first measured prediction is retained only after the
other two measured predictions compare exactly.

The steady interval is the existing helper's ``CPU activation H -> device
preprocessing -> method -> CPU token IDs`` interval.  State/model loading,
hashing, CUDA initialization, and diagnostic candidate materialization are
reported separately.  No target weights, evaluator-private truth, teacher
prefix, or A2 fallback is loaded by this script.

The historical A1 comparator is adapted to direct top-1 output for a fair
standalone runtime measurement.  It does not materialize top-k ranks or
candidate tensors.  Historical A1+A2 keeps the established top-512 proposal
and fixed K=256 public-prefix selector; its candidate tensors are retained in
the artifact as required by that method's registration.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource as sys_resource
import subprocess
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch

import trr0004_fresh_confirmation as fc
from token_reconstruction.causal_decoder_extension import (
    CausalResidualDecoder,
    FrozenAffineBase,
    build_causal_extension,
    validate_runtime_embeddings,
)
from token_reconstruction.footing import file_record, sha256_file, tensor_sha256
from token_reconstruction.historical_affine_ce import (
    direct_prediction_tensor,
    load_historical_affine_ce,
)
from token_reconstruction.historical_inputlens_bridge import (
    load_historical_lens_checkpoint,
    validate_normalized_embeddings,
)


TASK_ID = "TRR-0004"
SCRIPT_SCHEMA = "token-reconstruction.trr0004-fresh-confirmation-run.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0004-fresh-confirmation-run-failure.v1"
DEFAULT_MINIMUM_FREE_GIB = 8.0
DEFAULT_MAXIMUM_RESERVED_GIB = 6.0
DEFAULT_MAXIMUM_RSS_GIB = 16.0
DEFAULT_MAX_SECONDS = 1800.0
DEFAULT_A1_CHUNK = 256
DEFAULT_A2_K = 256
DEFAULT_A2_PROPOSAL_K = 512
DEFAULT_RECORD_BATCH_SIZE = 1

M_A1 = "historical_alpaca_a1"
M_A2 = "frozen_a1_a2_k256"
M_AFFINE = "historical_affine_ce_no_vocab_bias"
M_ATTENTION = "causal_h_attention128"
M_MLP = "positionwise_mlp256"

EXPECTED_METHOD_IDS = (M_A1, M_A2, M_AFFINE, M_ATTENTION, M_MLP)
EXPECTED_TRACKS = {
    M_A1: "comparator",
    M_A2: "comparator",
    M_AFFINE: "track_b",
    M_ATTENTION: "track_b",
    M_MLP: "track_b",
}
# The standalone A1 decision is top-1 only.  A2 remains the only method in
# this matrix for which candidates are a deployed artifact requirement.
EXPECTED_CANDIDATE_POLICIES = {
    M_A1: "forbidden",
    M_A2: "required",
    M_AFFINE: "forbidden",
    M_ATTENTION: "forbidden",
    M_MLP: "forbidden",
}
METHOD_STATE_COUNT = {method_id: 1 for method_id in EXPECTED_METHOD_IDS}


class PredictionRunnerError(RuntimeError):
    """Raised when a public prediction run cannot be bound or completed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PredictionRunnerError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PredictionRunnerError("unable to resolve executable git commit") from exc
    commit = result.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise PredictionRunnerError("executable git commit is not a full lowercase hash")
    return commit


def _git_status(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PredictionRunnerError("unable to capture git status") from exc
    return result.stdout


def _source_path(path: Path, *, root: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PredictionRunnerError(f"source file is unavailable: {path}")
    return path


def _source_records(root: Path, *, reference_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Hash every executable repository module used by this driver.

    The public reference is loaded from its historical external path, so it
    is recorded with an absolute path and hash in the run receipt.  Method
    registrations should also include that descriptor when their binding
    producer supports external public code resources.
    """

    repo_paths = (
        Path(__file__),
        Path(fc.__file__),
        root / "src/token_reconstruction/footing.py",
        root / "src/token_reconstruction/component_crossover.py",
        root / "src/token_reconstruction/a1a2_configuration_search.py",
        root / "src/token_reconstruction/dual_benchmark.py",
        root / "src/token_reconstruction/historical_affine_ce.py",
        root / "src/token_reconstruction/causal_decoder_extension.py",
        root / "src/token_reconstruction/historical_inputlens_bridge.py",
        root / "scripts/trr0003_footing_compare.py",
    )
    records: dict[str, dict[str, Any]] = {}
    for candidate in repo_paths:
        path = _source_path(candidate, root=root)
        key = path.relative_to(root.resolve()).as_posix()
        records[key] = file_record(path, repository_root=root.resolve())
    if reference_path is not None:
        path = _source_path(reference_path, root=root)
        records["external_public_prefix_reference"] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "binding_role": "public prefix implementation imported by the A1+A2 comparator",
        }
    return records


def _rusage_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  Keep the platform branch explicit so a
    # receipt never silently labels a byte count from another convention.
    value = int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _nvidia_query(query: str) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PredictionRunnerError(f"nvidia-smi query failed: {query}") from exc
    fields = [field.strip() for field in query.split(",")]
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            raise PredictionRunnerError(f"nvidia-smi returned malformed row for {query}")
        rows.append(dict(zip(fields, values)))
    if not rows:
        raise PredictionRunnerError(f"nvidia-smi returned no GPUs for {query}")
    return rows


def _compute_app_probe() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PredictionRunnerError("nvidia-smi compute-app query failed") from exc
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip() or "no running processes" in line.casefold():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) >= 3:
            rows.append({"pid": values[0], "process_name": values[1], "used_memory": values[2]})
    return rows


def _resource_limits(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        "minimum_free_gpu_bytes": int(float(args.minimum_free_gib) * 2**30),
        "maximum_gpu_reserved_bytes": int(float(args.maximum_reserved_gib) * 2**30),
        "maximum_host_rss_bytes": int(float(args.maximum_rss_gib) * 2**30),
        "maximum_wall_seconds": float(args.max_seconds),
    }


def _resource_preflight(
    args: argparse.Namespace,
    device: torch.device,
    *,
    stage: str,
    started: float,
) -> dict[str, Any]:
    """Fail closed on low free memory, another GPU process, or thermal risk."""

    limits = _resource_limits(args)
    if time.perf_counter() - started > limits["maximum_wall_seconds"]:
        raise PredictionRunnerError(f"wall-time guard expired at {stage}")
    rss = _rusage_rss_bytes()
    if rss > limits["maximum_host_rss_bytes"]:
        raise PredictionRunnerError(f"host RSS guard failed at {stage}: {rss} bytes")
    result: dict[str, Any] = {
        "stage": stage,
        "host_rss_bytes": rss,
        "limits": dict(limits),
    }
    if device.type != "cuda":
        result["status"] = "host_only_no_gpu_check"
        return result
    if not torch.cuda.is_available():
        raise PredictionRunnerError("CUDA was requested but is unavailable")
    free, total = torch.cuda.mem_get_info(device)
    reserved = int(torch.cuda.memory_reserved(device))
    gpu_rows = _nvidia_query("index,temperature.gpu,utilization.gpu,memory.free,memory.used")
    all_apps = _compute_app_probe()
    # nvidia-smi reports this runner once it has initialized CUDA.  Exclude
    # only that PID; every other compute process remains a fail-closed error.
    apps = [row for row in all_apps if row.get("pid") != str(os.getpid())]
    temperature_values: list[float] = []
    for row in gpu_rows:
        try:
            temperature_values.append(float(row["temperature.gpu"]))
        except (KeyError, ValueError) as exc:
            raise PredictionRunnerError("nvidia-smi temperature is malformed") from exc
    if apps:
        raise PredictionRunnerError(f"GPU is not exclusive at {stage}: {apps!r}")
    if any(value >= 85.0 for value in temperature_values):
        raise PredictionRunnerError(f"GPU temperature guard failed at {stage}: {temperature_values!r}")
    if int(free) < limits["minimum_free_gpu_bytes"]:
        raise PredictionRunnerError(f"GPU free-memory guard failed at {stage}: {int(free)} bytes")
    if reserved > limits["maximum_gpu_reserved_bytes"]:
        raise PredictionRunnerError(f"GPU reservation guard failed at {stage}: {reserved} bytes")
    result.update(
        {
            "status": "PASS",
            "cuda_free_bytes": int(free),
            "cuda_total_bytes": int(total),
            "cuda_reserved_bytes": reserved,
            "nvidia_smi_gpu": gpu_rows,
            "nvidia_smi_compute_apps": apps,
            "nvidia_smi_compute_apps_all": all_apps,
            "temperatures_c": temperature_values,
        }
    )
    return result


def _peak_memory(device: torch.device) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "process_max_rss_bytes": _rusage_rss_bytes(),
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return result


def _peak_memory_envelope(peaks: Sequence[Mapping[str, int | None]]) -> dict[str, int | None]:
    """Return the component-wise maximum over cold and per-cell peaks."""

    keys = (
        "process_max_rss_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    )
    result: dict[str, int | None] = {}
    for key in keys:
        values = [int(peak[key]) for peak in peaks if peak.get(key) is not None]
        result[key] = max(values) if values else None
    return result


def _geometry_estimate(cells: Sequence[fc.FreshCell]) -> dict[str, Any]:
    largest = max(cells, key=lambda cell: cell.sequence_tokens)
    embedding_bytes = fc.VOCAB_SIZE * fc.HIDDEN_SIZE * 4
    activation_bytes = largest.records * largest.sequence_tokens * fc.HIDDEN_SIZE * 2
    logits_bytes = largest.sequence_tokens * fc.VOCAB_SIZE * 4
    return {
        "largest_cell": largest.cell_id,
        "largest_shape": list(largest.shape),
        "a2_candidate_budget": DEFAULT_A2_K,
        "a2_proposal_budget": DEFAULT_A2_PROPOSAL_K,
        "record_batch_size": DEFAULT_RECORD_BATCH_SIZE,
        "embedding_table_float32_bytes": embedding_bytes,
        "largest_bf16_activation_bytes": activation_bytes,
        "one_record_full_float32_logits_bytes": logits_bytes,
        "working_set_guard_ceiling_bytes": int(DEFAULT_MAXIMUM_RESERVED_GIB * 2**30),
        "calculation": "embedding V*H*4 + largest-cell BF16 H + one-record V*S*4 logits, with public-prefix/cache and allocator workspace covered by the conservative 6 GiB CUDA reservation guard",
        "qualification_requirement": "largest 128-position A2 K256 cell must pass before the four-cell matrix",
    }


def _asset_paths(binding: Mapping[str, Any], *, key: str, root: Path, method_id: str) -> list[Path]:
    values = binding.get(key)
    if not isinstance(values, list) or not values:
        raise PredictionRunnerError(f"{method_id} binding has no {key} assets")
    paths: list[Path] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise PredictionRunnerError(f"{method_id} {key}[{index}] binding is malformed")
        try:
            path = fc._asset_path(value, repository_root=root, description=f"{method_id} {key}[{index}]")
        except Exception as exc:
            raise PredictionRunnerError(f"{method_id} {key}[{index}] cannot be validated") from exc
        paths.append(path)
    return paths


def _validate_method_registration(
    *,
    registration_path: Path,
    registration: Mapping[str, Any],
    root: Path,
    current_commit: str,
) -> dict[str, Any]:
    method_ids = tuple(registration["method_ids"])
    if method_ids != EXPECTED_METHOD_IDS:
        raise PredictionRunnerError(f"fresh confirmation method order changed: {method_ids!r}")
    if set(registration["tracks"]) != set(EXPECTED_METHOD_IDS):
        raise PredictionRunnerError("fresh confirmation track registration is incomplete")
    if dict(registration["tracks"]) != EXPECTED_TRACKS:
        raise PredictionRunnerError(f"fresh confirmation method tracks changed: {registration['tracks']!r}")
    if dict(registration["candidate_policies"]) != EXPECTED_CANDIDATE_POLICIES:
        raise PredictionRunnerError(
            "fresh confirmation candidate policies must make standalone A1 top-1 only and retain A2 candidates"
        )
    bindings = registration["bindings"]
    if set(bindings) != set(EXPECTED_METHOD_IDS):
        raise PredictionRunnerError("fresh confirmation method bindings are incomplete")
    registration_source = fc._load_json(registration_path, description="fresh confirmation registration")
    rows = registration_source.get("methods")
    if not isinstance(rows, list) or [row.get("id") for row in rows if isinstance(row, Mapping)] != list(EXPECTED_METHOD_IDS):
        raise PredictionRunnerError("fresh confirmation method rows do not match the fixed five")
    bound_code_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise PredictionRunnerError("fresh confirmation method row is malformed")
        method_id = str(row.get("id"))
        if row.get("track") != EXPECTED_TRACKS[method_id] or row.get("candidate_policy") != EXPECTED_CANDIDATE_POLICIES[method_id]:
            raise PredictionRunnerError(f"fresh confirmation method row policy changed: {method_id}")
        binding = bindings[method_id]
        if binding.get("code_commit") != current_commit:
            raise PredictionRunnerError(
                f"{method_id} code binding commit {binding.get('code_commit')!r} does not match executable HEAD {current_commit}"
            )
        for code in binding.get("code", []):
            if isinstance(code, Mapping) and isinstance(code.get("path"), str):
                bound_code_paths.add(str(code["path"]))
        state_paths = _asset_paths(binding, key="method_state", root=root, method_id=method_id)
        if len(state_paths) != METHOD_STATE_COUNT[method_id]:
            raise PredictionRunnerError(f"{method_id} must bind exactly one frozen state artifact")
        _asset_paths(binding, key="method_config", root=root, method_id=method_id)
    runner_relative = Path(__file__).resolve().relative_to(root.resolve()).as_posix()
    if runner_relative not in bound_code_paths:
        raise PredictionRunnerError("method registration does not bind the executed five-method driver")
    return {
        "method_ids": method_ids,
        "bindings": {method: dict(bindings[method]) for method in method_ids},
        "candidate_policies": dict(registration["candidate_policies"]),
        "tracks": dict(registration["tracks"]),
        "registration_source": registration_source,
    }


def _single_state(binding: Mapping[str, Any], *, method_id: str, root: Path) -> Path:
    paths = _asset_paths(binding, key="method_state", root=root, method_id=method_id)
    if len(paths) != 1:
        raise PredictionRunnerError(f"{method_id} requires one state file, got {len(paths)}")
    return paths[0]


def _resolve_source_repo(root: Path) -> Path:
    if root.parent.name == ".worktrees":
        return root.parent.parent.resolve()
    return root.resolve()


def _default_path(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_file() or candidate.is_dir():
        return candidate.resolve()
    source = _resolve_source_repo(root) / relative
    return source.resolve()


def _runtime_asset_path(binding: Mapping[str, Any], *, role: str) -> Path:
    """Resolve and re-hash one external public runtime asset from a binding."""

    assets = binding.get("runtime_assets")
    if not isinstance(assets, Mapping):
        raise PredictionRunnerError("selected method has no runtime asset binding")
    descriptor = assets.get(role)
    if not isinstance(descriptor, Mapping):
        raise PredictionRunnerError(f"selected method runtime asset is absent: {role}")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise PredictionRunnerError(f"runtime asset path is not absolute: {role}")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        raise PredictionRunnerError(f"runtime asset is unavailable: {role}: {path}")
    path = path.resolve()
    actual = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    if dict(descriptor) != actual:
        raise PredictionRunnerError(f"runtime asset binding changed: {role}")
    return path


def _load_normalized_embeddings(
    *,
    path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load the pinned normalized E table without instantiating the public model."""

    started = time.perf_counter()
    try:
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"embeddings"}:
            raise PredictionRunnerError("public normalized embedding resource must contain only embeddings")
        embeddings_cpu = tensors["embeddings"].contiguous()
        validate_normalized_embeddings(
            embeddings_cpu,
            vocabulary_size=fc.VOCAB_SIZE,
            check_unit_norm=True,
        )
        digest = tensor_sha256(embeddings_cpu)
        embeddings = embeddings_cpu.to(device=device).contiguous()
        validate_runtime_embeddings(
            embeddings,
            hidden_size=fc.HIDDEN_SIZE,
            vocab_size=fc.VOCAB_SIZE,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del tensors, embeddings_cpu
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        evidence = {
            "loader_scope": "normalized public embedding table only; no public model or prefix instantiated",
            "load_seconds": elapsed,
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "tensor_sha256": digest,
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
        }
        return embeddings, evidence
    except PredictionRunnerError:
        raise
    except Exception as exc:
        raise PredictionRunnerError(f"public normalized embedding load failed: {path}") from exc


def _cpu_normalized_public_embedding(reference: Any, weight: torch.Tensor) -> torch.Tensor:
    """Recreate the registered E with the historical TRR-0003 CPU path.

    TRR-0003 copied the BF16 public embedding weight to CPU before the
    reference's FP32 normalization. Keeping that device boundary explicit
    avoids treating CPU/CUDA reduction differences as a resource
    mismatch while retaining an exact byte-level integrity check.
    """

    if not isinstance(weight, torch.Tensor):
        raise PredictionRunnerError("public model embedding weight is not a tensor")
    try:
        normalized = reference.normalize_public_embeddings(weight.detach().cpu())
    except Exception as exc:
        raise PredictionRunnerError("public model embedding normalization failed") from exc
    if not isinstance(normalized, torch.Tensor):
        raise PredictionRunnerError("public model embedding normalization returned a non-tensor")
    return normalized.detach().cpu().contiguous()


def _require_exact_public_embedding_binding(
    expected: torch.Tensor,
    registered: torch.Tensor,
) -> None:
    """Require exact equality after matching the registered resource device."""

    registered_cpu = registered.detach().cpu().contiguous()
    try:
        matches = (
            expected.dtype == registered_cpu.dtype
            and tuple(expected.shape) == tuple(registered_cpu.shape)
            and torch.equal(expected, registered_cpu)
        )
    finally:
        del registered_cpu
    if not matches:
        raise PredictionRunnerError(
            "registered normalized embedding table differs from CPU public model embedding"
        )


def _load_public_prefix(
    *,
    snapshot: Path,
    reference_path: Path,
    lens_path: Path,
    embedding_path: Path,
    device: torch.device,
) -> tuple[Any, Any, torch.Tensor, dict[str, Any]]:
    """Load the public P0 prefix only for the A1+A2 comparator."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise PredictionRunnerError("the confirmation driver requires CUDA for the public prefix")
    try:
        legacy = importlib.import_module("trr0003_footing_compare")
        reference = legacy._import_reference(reference_path)
        from transformers import AutoModelForCausalLM

        started = time.perf_counter()
        full = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        full.requires_grad_(False)
        if int(full.config.hidden_size) != fc.HIDDEN_SIZE or int(full.config.vocab_size) != fc.VOCAB_SIZE:
            raise PredictionRunnerError("public model geometry changed")
        precut = reference.PublicP0Precut(full, (0, 1, 2, 3)).to(device).eval()
        model_embeddings_cpu = _cpu_normalized_public_embedding(
            reference, precut.embed_tokens.weight
        )
        lens = reference.load_frozen_lens(lens_path, device=device)
        embeddings, embedding_evidence = _load_normalized_embeddings(path=embedding_path, device=device)
        _require_exact_public_embedding_binding(model_embeddings_cpu, embeddings)
        model_embedding_state_sha256 = reference.state_sha256(
            {"normalized_embedding": model_embeddings_cpu}, domain=b"ersoy-public-p0-normalized-embedding-v1"
        )
        del model_embeddings_cpu
        del full
        gc.collect()
        torch.cuda.empty_cache()
        runtime = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }
        evidence = {
            "loader_scope": "public model/P0 prefix, retained A1 lens, and registered normalized embedding table",
            "model": {
                "id": fc.MODEL_ID,
                "revision": fc.MODEL_REVISION,
                "snapshot": str(snapshot),
                "local_files_only": True,
                "dtype": "bfloat16",
                "attention_implementation": "sdpa",
            },
            "prefix_layers_state_sha256": reference.state_sha256(
                dict(precut.layers.state_dict()), domain=b"ersoy-public-p0-layers-v1"
            ),
            "embedding_weight_state_sha256": reference.state_sha256(
                {"embed_tokens.weight": precut.embed_tokens.weight}, domain=b"ersoy-public-p0-embedding-v1"
            ),
            "normalized_embedding_state_sha256": model_embedding_state_sha256,
            "normalized_embedding_construction": {
                "source": "BF16 public P0 embedding weight detached to CPU, then reference FP32 row normalization",
                "registered_resource": "TRR-0003 public_normalized_embeddings.safetensors",
                "runtime_table_used_by_a2": "registered CPU-normalized FP32 table copied to CUDA",
                "numeric_port": "CPU normalization boundary is explicit; native CUDA-normalized A2 equivalence is not claimed",
            },
            "lens_state_sha256": reference.state_sha256(
                dict(lens.state_dict()), domain=b"ersoy-a1-lens-state-v1"
            ),
            "model_load_seconds": time.perf_counter() - started,
            "public_embedding_load_seconds": embedding_evidence["load_seconds"],
            "runtime": runtime,
            "public_resources": {
                "snapshot": str(snapshot),
                "reference_implementation": {
                    "path": str(reference_path),
                    "bytes": int(reference_path.stat().st_size),
                    "sha256": sha256_file(reference_path),
                },
                "retained_lens": {
                    "path": str(lens_path),
                    "bytes": int(lens_path.stat().st_size),
                    "sha256": sha256_file(lens_path),
                },
                "normalized_embedding": embedding_evidence,
            },
        }
        return precut, lens, embeddings, evidence
    except PredictionRunnerError:
        raise
    except Exception as exc:
        raise PredictionRunnerError("public model/prefix load failed") from exc


def _load_standalone_resources(
    *,
    method_id: str,
    embedding_path: Path,
    lens_path: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module | None, torch.Tensor, dict[str, Any]]:
    """Load E and one standalone decoder's retained public resources only."""

    embeddings, embedding_evidence = _load_normalized_embeddings(path=embedding_path, device=device)
    lens: torch.nn.Module | None = None
    lens_evidence: dict[str, Any] | None = None
    if method_id == M_A1:
        if lens_path is None:
            raise PredictionRunnerError("standalone A1 requires the retained lens state")
        lens = load_historical_lens_checkpoint(lens_path, device=device)
        lens_evidence = {
            "path": str(lens_path),
            "bytes": int(lens_path.stat().st_size),
            "sha256": sha256_file(lens_path),
            "state_sha256": lens.lens_state_sha256,
            "loader": "token_reconstruction.historical_inputlens_bridge.HistoricalInputLensBridge",
        }
    return lens, embeddings, {
        "loader_scope": f"selected method {method_id} plus normalized public embedding table; no public model or P0 prefix instantiated",
        "model": {
            "id": fc.MODEL_ID,
            "revision": fc.MODEL_REVISION,
            "loaded": False,
        },
        "model_load_seconds": 0.0,
        "public_embedding_load_seconds": embedding_evidence["load_seconds"],
        "public_prefix_loaded": False,
        "public_resources": {
            "normalized_embedding": embedding_evidence,
            "retained_lens": lens_evidence,
        },
    }


def _normalize_prediction_for_timing(
    predictions: torch.Tensor,
    mask: torch.Tensor,
    *,
    sequence_tokens: int,
) -> torch.Tensor:
    if predictions.ndim != 1 or int(predictions.shape[0]) != sequence_tokens:
        raise PredictionRunnerError("method prediction has wrong per-record geometry")
    mask = mask.to(device=predictions.device, dtype=torch.bool)
    if not bool(mask[0].item()):
        raise PredictionRunnerError("prediction input does not begin with an active BOS")
    output = torch.full((sequence_tokens,), fc.INVALID_TOKEN_ID, dtype=torch.long, device=predictions.device)
    output[mask] = predictions.to(torch.long)[mask]
    output[0] = fc.BOS_TOKEN_ID
    active = output[mask]
    if active.lt(0).any().item() or active.ge(fc.VOCAB_SIZE).any().item():
        raise PredictionRunnerError("method emitted an invalid active token")
    return output


def _validate_normalized_batch_prediction(predictions: torch.Tensor, cell: fc.FreshCell) -> None:
    """Validate IDs already normalized inside the timed predictor callback.

    This post-timing check must not rewrite the prediction: BOS and right-pad
    handling is part of the method callback, so the measured interval includes
    the exact IDs that are later serialized.
    """

    if tuple(predictions.shape) != tuple(cell.attention_mask.shape):
        raise PredictionRunnerError(f"prediction batch geometry changed for {cell.cell_id}")
    predictions = predictions.to(device="cpu", dtype=torch.long)
    mask = cell.attention_mask.to(device="cpu", dtype=torch.bool)
    if not mask[:, 0].all().item():
        raise PredictionRunnerError(f"fresh panel cell lacks an active BOS: {cell.cell_id}")
    if not predictions[:, 0].eq(fc.BOS_TOKEN_ID).all().item():
        raise PredictionRunnerError(f"method did not normalize BOS inside timed callback in {cell.cell_id}")
    if not predictions[~mask].eq(fc.INVALID_TOKEN_ID).all().item():
        raise PredictionRunnerError(f"method did not normalize right-padding inside timed callback in {cell.cell_id}")
    if predictions[mask].lt(0).any().item() or predictions[mask].ge(fc.VOCAB_SIZE).any().item():
        raise PredictionRunnerError(f"method emitted an invalid active token in {cell.cell_id}")


class _A1Adapter:
    def __init__(self, *, lens: torch.nn.Module, embeddings: torch.Tensor) -> None:
        self.lens = lens
        self.embeddings = embeddings
        self.calls = 0
        self.scored_positions = 0
        self._cell_baseline = {"calls": 0, "scored_positions": 0}

    def begin_cell(self) -> None:
        self._cell_baseline = {
            "calls": self.calls,
            "scored_positions": self.scored_positions,
        }

    @torch.inference_mode()
    def __call__(self, row_h: torch.Tensor, row_mask: torch.Tensor, row_positions: torch.Tensor) -> torch.Tensor:
        del row_positions
        self.calls += 1
        scored = row_mask.to(torch.bool).clone()
        scored[0] = False
        output = torch.full((int(row_h.shape[0]),), fc.INVALID_TOKEN_ID, dtype=torch.long, device=row_h.device)
        output[0] = fc.BOS_TOKEN_ID
        if scored.any().item():
            logits = self.lens(row_h[scored], self.embeddings).float()
            if not torch.isfinite(logits).all().item():
                raise PredictionRunnerError("historical A1 logits are non-finite")
            output[scored] = logits.argmax(dim=-1).to(torch.long)
            self.scored_positions += int(scored.sum().item())
        return output

    def evidence(self) -> dict[str, Any]:
        return {
            "calls": self.calls - self._cell_baseline["calls"],
            "scored_positions": self.scored_positions - self._cell_baseline["scored_positions"],
            "candidate_simulations": 0,
            "public_prefix_calls": 0,
            "a2_fallback": False,
            "candidate_output": "forbidden; standalone direct top-1 only",
        }


class _A2Adapter:
    def __init__(self, *, precut: Any, lens: torch.nn.Module, embeddings: torch.Tensor, device: torch.device, policy: Any) -> None:
        self.precut = precut
        self.lens = lens
        self.embeddings = embeddings
        self.device = device
        self.policy = policy
        self.calls = 0
        self.proposal_seconds = 0.0
        self.candidate_simulations = 0
        self.executed_candidate_simulations = 0
        self.prefix_commit_tokens = 0
        self.prefix_calls = 0
        self._record_proposals: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._cell_baseline = {
            "calls": 0,
            "proposal_seconds": 0.0,
            "candidate_simulations": 0,
            "executed_candidate_simulations": 0,
            "prefix_commit_tokens": 0,
            "prefix_calls": 0,
        }

    def begin_cell(self) -> None:
        self._record_proposals = []
        self._cell_baseline = {
            "calls": self.calls,
            "proposal_seconds": self.proposal_seconds,
            "candidate_simulations": self.candidate_simulations,
            "executed_candidate_simulations": self.executed_candidate_simulations,
            "prefix_commit_tokens": self.prefix_commit_tokens,
            "prefix_calls": self.prefix_calls,
        }

    @torch.inference_mode()
    def __call__(self, row_h: torch.Tensor, row_mask: torch.Tensor, row_positions: torch.Tensor) -> torch.Tensor:
        # The frozen TRR3 helpers intentionally use CPU observations/masks as
        # their public API and move only the scored rows to CUDA.  This adapter
        # stages the row back to that API inside the measured interval; the
        # staging is included in timing and disclosed in the method evidence.
        observations = row_h.detach().to(device="cpu").view(1, row_h.shape[0], row_h.shape[1])
        attention_mask = row_mask.detach().to(device="cpu", dtype=torch.long).view(1, row_h.shape[0])
        position_ids = row_positions.detach().to(device="cpu", dtype=torch.long).view(1, row_h.shape[0])
        legacy = importlib.import_module("trr0003_footing_compare")
        before_prefix = int(getattr(self.precut, "checked_cache_transitions", 0))
        proposal = legacy.propose_public_a1(
            observations=observations,
            attention_mask=attention_mask,
            lens=self.lens,
            normalized_embeddings=self.embeddings,
            max_k=DEFAULT_A2_PROPOSAL_K,
            chunk=DEFAULT_A1_CHUNK,
        )
        decoded = legacy.decode_policy(
            observations=observations,
            attention_mask=attention_mask,
            position_ids=position_ids,
            candidates=proposal.candidates[:, :, :DEFAULT_A2_K].contiguous(),
            a1_confidence=proposal.top1_confidence,
            precut=self.precut,
            device=self.device,
            policy=self.policy,
            record_batch_size=DEFAULT_RECORD_BATCH_SIZE,
        )
        after_prefix = int(getattr(self.precut, "checked_cache_transitions", 0))
        if after_prefix < before_prefix:
            raise PredictionRunnerError("public-prefix transition counter moved backwards")
        self.calls += 1
        self.proposal_seconds += float(proposal.elapsed_seconds)
        self.candidate_simulations += int(decoded.candidate_simulations)
        self.executed_candidate_simulations += int(decoded.executed_candidate_simulations)
        self.prefix_commit_tokens += int(decoded.prefix_commit_tokens)
        self.prefix_calls += after_prefix - before_prefix
        # run_warmed_prediction uses one warmup plus three measured calls.  The
        # first measured proposal is the diagnostic candidate tensor retained
        # for this record; candidate materialization is never a truth decision.
        if self.calls % 4 == 2:
            self._record_proposals.append(
                (proposal.candidates[0].detach().cpu().contiguous(), proposal.scores[0].detach().cpu().contiguous())
            )
        return decoded.predictions[0].to(device=row_h.device, dtype=torch.long)

    def candidate_tensors(self, *, records: int, sequence_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self._record_proposals) != records:
            raise PredictionRunnerError(
                f"A2 candidate diagnostics cover {len(self._record_proposals)} records, expected {records}"
            )
        candidates = torch.stack([value[0] for value in self._record_proposals]).to(torch.long).contiguous()
        scores = torch.stack([value[1] for value in self._record_proposals]).to(torch.float32).contiguous()
        if tuple(candidates.shape) != (records, sequence_tokens, DEFAULT_A2_PROPOSAL_K):
            raise PredictionRunnerError("A2 candidate diagnostic geometry changed")
        return candidates, scores

    def evidence(self) -> dict[str, Any]:
        return {
            "calls": self.calls - self._cell_baseline["calls"],
            "proposal_seconds_sum": self.proposal_seconds - self._cell_baseline["proposal_seconds"],
            "candidate_budget": DEFAULT_A2_K,
            "proposal_budget": DEFAULT_A2_PROPOSAL_K,
            "candidate_simulations": self.candidate_simulations - self._cell_baseline["candidate_simulations"],
            "executed_candidate_simulations": self.executed_candidate_simulations - self._cell_baseline["executed_candidate_simulations"],
            "prefix_commit_tokens": self.prefix_commit_tokens - self._cell_baseline["prefix_commit_tokens"],
            "public_prefix_calls": self.prefix_calls - self._cell_baseline["prefix_calls"],
            "record_batch_size": DEFAULT_RECORD_BATCH_SIZE,
            "a2_fallback": False,
            "candidate_output": "required; first measured proposal per record retained",
            "input_adapter": "legacy TRR3 proposal/decode helpers receive a CPU row and move scored activations to CUDA; this staging is inside the timed interval",
        }


class _AffineAdapter:
    def __init__(self, *, model: torch.nn.Module, embeddings: torch.Tensor) -> None:
        self.model = model
        self.embeddings = embeddings
        self.calls = 0
        self._cell_baseline = 0

    def begin_cell(self) -> None:
        self._cell_baseline = self.calls

    @torch.inference_mode()
    def __call__(self, row_h: torch.Tensor, row_mask: torch.Tensor, row_positions: torch.Tensor) -> torch.Tensor:
        del row_positions
        self.calls += 1
        prediction = direct_prediction_tensor(
            self.model,
            row_h,
            self.embeddings,
            device=row_h.device,
            batch_size=512,
        ).to(device=row_h.device, dtype=torch.long)
        return prediction

    def evidence(self) -> dict[str, Any]:
        return {"calls": self.calls - self._cell_baseline, "public_prefix_calls": 0, "candidate_simulations": 0, "a2_fallback": False}


class _ContextAdapter:
    def __init__(self, *, model: CausalResidualDecoder, embeddings: torch.Tensor) -> None:
        self.model = model
        self.embeddings = embeddings
        self.calls = 0
        self._cell_baseline = 0

    def begin_cell(self) -> None:
        self._cell_baseline = self.calls

    @torch.inference_mode()
    def __call__(self, row_h: torch.Tensor, row_mask: torch.Tensor, row_positions: torch.Tensor) -> torch.Tensor:
        del row_positions
        self.calls += 1
        logits = self.model(
            row_h.unsqueeze(0),
            row_mask.to(device=row_h.device, dtype=torch.bool).unsqueeze(0),
            self.embeddings,
        )
        return logits.argmax(dim=-1)[0].to(torch.long)

    def evidence(self) -> dict[str, Any]:
        return {"calls": self.calls - self._cell_baseline, "public_prefix_calls": 0, "candidate_simulations": 0, "a2_fallback": False}


def _load_context_state(path: Path, *, method_id: str, device: torch.device) -> CausalResidualDecoder:
    try:
        state = load_file(str(path), device="cpu")
        expected_base = {"W": state["base.W"], "b": state["base.b"], "s": state["base.s"]}
        model = build_causal_extension(FrozenAffineBase.from_state_dict(expected_base), method_id)
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
        return model.to(device=device).eval()
    except Exception as exc:
        raise PredictionRunnerError(f"cannot load contextual decoder state for {method_id}: {path}") from exc


def _load_method_adapters(
    *,
    method_id: str,
    registration: Mapping[str, Any],
    root: Path,
    precut: Any | None,
    lens: torch.nn.Module | None,
    embeddings: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Load exactly one registered adapter for this isolated process."""

    if method_id not in EXPECTED_METHOD_IDS:
        raise PredictionRunnerError(f"unknown confirmation method: {method_id}")
    adapters: dict[str, Any] = {}
    if method_id == M_A1:
        if lens is None:
            raise PredictionRunnerError("retained public A1 lens did not load")
        adapters[method_id] = _A1Adapter(lens=lens, embeddings=embeddings)
        return adapters
    if method_id == M_A2:
        if precut is None or lens is None:
            raise PredictionRunnerError("A1+A2 requires the public P0 prefix and retained A1 lens")
        legacy = importlib.import_module("trr0003_footing_compare")
        policy = legacy._fixed_k256_policy()
        a1_state = _single_state(registration["bindings"][M_A1], method_id=M_A1, root=root)
        a2_state = _single_state(registration["bindings"][M_A2], method_id=M_A2, root=root)
        if sha256_file(a1_state) != sha256_file(a2_state):
            raise PredictionRunnerError("A1 and A1+A2 do not bind the same retained lens state")
        adapters[method_id] = _A2Adapter(
            precut=precut,
            lens=lens,
            embeddings=embeddings,
            device=device,
            policy=policy,
        )
        return adapters
    if method_id == M_AFFINE:
        affine_state = _single_state(registration["bindings"][M_AFFINE], method_id=M_AFFINE, root=root)
        adapters[method_id] = _AffineAdapter(
            model=load_historical_affine_ce(
                affine_state,
                hidden_size=fc.HIDDEN_SIZE,
                vocab_size=fc.VOCAB_SIZE,
                bias_mode="none",
                device=device,
            ),
            embeddings=embeddings,
        )
        return adapters
    if method_id == M_ATTENTION:
        attention_state = _single_state(registration["bindings"][M_ATTENTION], method_id=M_ATTENTION, root=root)
        adapters[method_id] = _ContextAdapter(
            model=_load_context_state(attention_state, method_id=M_ATTENTION, device=device),
            embeddings=embeddings,
        )
        return adapters
    if method_id == M_MLP:
        mlp_state = _single_state(registration["bindings"][M_MLP], method_id=M_MLP, root=root)
        adapters[method_id] = _ContextAdapter(
            model=_load_context_state(mlp_state, method_id=M_MLP, device=device),
            embeddings=embeddings,
        )
        return adapters
    raise PredictionRunnerError(f"unsupported confirmation method: {method_id}")


def _normalized_predictor(adapter: Any, cell: fc.FreshCell):
    """Apply the final BOS/right-padding decision inside each timed call."""

    def predict(row_h: torch.Tensor, row_mask: torch.Tensor, row_positions: torch.Tensor) -> torch.Tensor:
        raw = adapter(row_h, row_mask, row_positions)
        return _normalize_prediction_for_timing(raw, row_mask, sequence_tokens=cell.sequence_tokens)

    return predict


def _write_prediction(
    *,
    path: Path,
    cell: fc.FreshCell,
    method_id: str,
    predictions: torch.Tensor,
    binding: Mapping[str, Any],
    panel_sha256: str,
    selection_plan_sha256: str,
    candidates: torch.Tensor | None = None,
    candidate_scores: torch.Tensor | None = None,
) -> None:
    if path.exists() or path.is_symlink():
        raise PredictionRunnerError(f"prediction artifact already exists: {path}")
    if tuple(predictions.shape) != tuple(cell.attention_mask.shape):
        raise PredictionRunnerError(f"prediction geometry changed for {cell.cell_id}/{method_id}")
    if (candidates is None) != (candidate_scores is None):
        raise PredictionRunnerError("candidate IDs and scores must be supplied together")
    if candidates is not None and (tuple(candidates.shape[:2]) != tuple(predictions.shape) or tuple(candidate_scores.shape) != tuple(candidates.shape)):
        raise PredictionRunnerError("candidate diagnostic geometry changed")
    metadata = {
        "schema": fc.PREDICTION_SCHEMA,
        "task_id": TASK_ID,
        "panel_sha256": panel_sha256,
        "selection_plan_sha256": selection_plan_sha256,
        "observation_sha256": cell.observation_sha256,
        "cell_id": cell.cell_id,
        "style": cell.style,
        "condition": cell.condition,
        "method_id": method_id,
        "geometry_json": json.dumps(
            {
                "records": fc.RECORDS_PER_STYLE,
                "sequence_tokens": cell.sequence_tokens,
                "hidden_size": fc.HIDDEN_SIZE,
                "cut_depth": fc.CUT_DEPTH,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "binding_json": json.dumps(dict(binding), sort_keys=True, separators=(",", ":")),
    }
    tensors: dict[str, torch.Tensor] = {"predictions": predictions.to(device="cpu", dtype=torch.int64).contiguous()}
    if candidates is not None and candidate_scores is not None:
        serialized_candidates = candidates.to(device="cpu", dtype=torch.int64).contiguous()
        serialized_scores = candidate_scores.to(device="cpu", dtype=torch.float32).contiguous()
        active = cell.attention_mask.to(torch.bool)
        serialized_candidates[:, 0, :] = fc.BOS_TOKEN_ID
        serialized_candidates[~active] = fc.INVALID_TOKEN_ID
        serialized_scores[:, 0, :] = 0.0
        serialized_scores[~active] = float("-inf")
        tensors["candidates"] = serialized_candidates
        tensors["candidate_scores"] = serialized_scores
        metadata["candidate_serialization_json"] = json.dumps(
            {
                "bos_candidate_placeholder": "repeated_known_bos",
                "bos_candidate_score_placeholder": "finite_zero",
                "padded_candidates": "invalid_token_id",
                "padded_scores": "negative_infinity",
                "bos_row_excluded_from_scoring": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def _timing_summary(timing: Mapping[str, Any], *, adapter: Any, cell: fc.FreshCell, path: Path, root: Path, peak: Mapping[str, Any]) -> dict[str, Any]:
    records = timing.get("records")
    if not isinstance(records, list) or len(records) != cell.records:
        raise PredictionRunnerError("warmed timing receipt is incomplete")
    warmup_runs = timing.get("warmup_runs")
    measured_runs = timing.get("measured_runs")
    if warmup_runs != 1 or measured_runs != 3:
        raise PredictionRunnerError("warmed timing run counts changed")
    mismatches = [row.get("record_index") for row in records if not bool(row.get("repeated_prediction_exact"))]
    if mismatches:
        raise PredictionRunnerError(f"repeated prediction mismatch at records {mismatches!r}")
    per_record_warmup_seconds = [
        sum(float(value) for value in row["warmup_seconds"]) for row in records
    ]
    per_record_measured_seconds = [
        sum(float(value) for value in row["measured_seconds"]) for row in records
    ]
    warmup_seconds = sum(per_record_warmup_seconds)
    measured_seconds = sum(per_record_measured_seconds)
    return {
        "cell_id": cell.cell_id,
        "method_id": getattr(adapter, "method_id", None),
        "records": cell.records,
        "active_tokens": int(cell.attention_mask.to(torch.bool).sum().item()),
        "scored_tokens": int(cell.attention_mask.to(torch.bool).sum().item()) - cell.records,
        "warmup_runs_per_record": int(warmup_runs),
        "measured_runs_per_record": int(measured_runs),
        "warmup_seconds_sum": warmup_seconds,
        "measured_seconds_sum": measured_seconds,
        "per_record_warmup_seconds": per_record_warmup_seconds,
        "per_record_measured_seconds": per_record_measured_seconds,
        "per_record_measured_mean_seconds": [value / int(measured_runs) for value in per_record_measured_seconds],
        "per_record_timing_records": [dict(row) for row in records],
        "timed_interval_total_seconds": float(timing["total_elapsed_seconds"]),
        "per_record_total_seconds": float(timing["total_elapsed_seconds"]) / cell.records,
        "steady_interval": "CPU activation H -> device preprocessing -> method prediction -> predicted IDs CPU",
        "cold_costs_separate": True,
        "peak_memory": dict(peak),
        "method_specific": adapter.evidence(),
        "artifact": str(path.relative_to(root).as_posix()),
    }


def _method_adapter_name(adapter: Any, method_id: str) -> None:
    # The generic timing helper does not know method IDs.  Keep the ID in the
    # adapter so evidence remains unambiguous without changing that helper.
    adapter.method_id = method_id


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    method_id = str(args.method)
    if method_id not in EXPECTED_METHOD_IDS:
        raise PredictionRunnerError(f"unknown confirmation method: {method_id}")
    started_perf = time.perf_counter()
    started_utc = utc_now()
    commit_start = _git_commit(root)
    status_start = _git_status(root)
    panel_path = args.panel.expanduser().resolve()
    plan_path = args.selection_plan.expanduser().resolve()
    registration_path = args.registration.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise PredictionRunnerError(f"prediction output must be a new path: {output_root}")
    output_root.mkdir(parents=True)

    panel = fc.load_fresh_panel(panel_path, repository_root=root)
    cells = tuple(
        sorted(
            fc.load_fresh_cells(panel, repository_root=root),
            key=lambda cell: (-cell.sequence_tokens, cell.cell_id),
        )
    )
    if len(cells) != 4:
        raise PredictionRunnerError(f"fresh panel has {len(cells)} cells, expected four")
    if max(cell.sequence_tokens for cell in cells) < 128:
        raise PredictionRunnerError("largest qualification geometry is not the required 128-position cell")
    panel_sha256 = sha256_file(panel_path)
    plan_sha256 = sha256_file(plan_path)
    registration = fc.load_confirmation_registration(
        registration_path,
        repository_root=root,
        panel_path=panel_path,
        selection_plan_path=plan_path,
    )
    registration_data = _validate_method_registration(
        registration_path=registration_path,
        registration=registration,
        root=root,
        current_commit=commit_start,
    )
    # The helper returns the plan hash from the panel; recheck the supplied
    # path before model loading so a swapped selection descriptor cannot be
    # hidden by a matching filename.
    if str(panel.get("selection_plan_sha256")) != plan_sha256:
        raise PredictionRunnerError("panel selection-plan hash differs from supplied plan")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise PredictionRunnerError("confirmation prediction execution is CUDA-only")
    preflight = _resource_preflight(args, device, stage=f"before_{method_id}_resource_load", started=started_perf)
    geometry = _geometry_estimate(cells)
    snapshot = args.model_snapshot.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    selected_binding = registration_data["bindings"][method_id]
    embedding_path = _runtime_asset_path(selected_binding, role="public_embedding_table")

    lens: torch.nn.Module | None = None
    lens_path: Path | None = None
    if method_id in (M_A1, M_A2):
        lens_binding = _single_state(registration_data["bindings"][M_A1], method_id=M_A1, root=root)
        lens_path = args.lens.expanduser().resolve() if args.lens is not None else lens_binding
        if sha256_file(lens_path) != sha256_file(lens_binding):
            raise PredictionRunnerError("explicit retained lens path differs from registered A1 state")

    source_start = _source_records(
        root,
        reference_path=reference_path if method_id == M_A2 else None,
    )
    if method_id == M_A2:
        precut, lens, embeddings, model_evidence = _load_public_prefix(
            snapshot=snapshot,
            reference_path=reference_path,
            lens_path=lens_path,
            embedding_path=embedding_path,
            device=device,
        )
    else:
        precut = None
        lens, embeddings, model_evidence = _load_standalone_resources(
            method_id=method_id,
            embedding_path=embedding_path,
            lens_path=lens_path,
            device=device,
        )
    if tuple(embeddings.shape) != (fc.VOCAB_SIZE, fc.HIDDEN_SIZE):
        raise PredictionRunnerError("loaded public embedding geometry changed")
    validate_runtime_embeddings(embeddings, hidden_size=fc.HIDDEN_SIZE, vocab_size=fc.VOCAB_SIZE)
    _resource_preflight(args, device, stage=f"after_{method_id}_public_resource_load", started=started_perf)

    selected_state_path = _single_state(selected_binding, method_id=method_id, root=root)
    method_load_started = time.perf_counter()
    adapters = _load_method_adapters(
        method_id=method_id,
        registration=registration_data,
        root=root,
        precut=precut,
        lens=lens,
        embeddings=embeddings,
        device=device,
    )
    method_load_seconds = time.perf_counter() - method_load_started
    _resource_preflight(args, device, stage=f"after_{method_id}_state_load", started=started_perf)
    cold_peak_memory = _peak_memory(device)
    model_evidence["selected_method"] = method_id
    model_evidence["selected_method_state"] = {
        "path": str(selected_state_path),
        "bytes": int(selected_state_path.stat().st_size),
        "sha256": sha256_file(selected_state_path),
    }
    model_evidence["method_state_load_seconds"] = method_load_seconds
    model_evidence["cold_peak_memory"] = cold_peak_memory

    method_timings: dict[str, list[dict[str, Any]]] = {method_id: []}
    per_cell_peak_memory: dict[str, dict[str, int | None]] = {}
    prediction_count = 0
    startup_boundary_perf: float | None = None
    startup_boundary_utc: str | None = None
    adapter = adapters[method_id]
    _method_adapter_name(adapter, method_id)
    for cell_index, cell in enumerate(cells):
        _resource_preflight(args, device, stage=f"before_{cell.cell_id}_{method_id}", started=started_perf)
        begin_cell = getattr(adapter, "begin_cell", None)
        if callable(begin_cell):
            begin_cell()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timed_predictor = _normalized_predictor(adapter, cell)
        if cell_index == 0:
            startup_boundary_utc = utc_now()
            startup_boundary_perf = time.perf_counter()
        prediction, timing = fc.run_warmed_prediction(
            observations=cell.activations,
            attention_mask=cell.attention_mask,
            position_ids=cell.position_ids,
            predictor=timed_predictor,
            device=device,
            warmup_runs=1,
            measured_runs=3,
        )
        if tuple(prediction.shape) != tuple(cell.attention_mask.shape):
            raise PredictionRunnerError(f"prediction geometry changed for {cell.cell_id}/{method_id}")
        # The predictor callback already applies BOS/right-padding normalization
        # inside the timed interval.  Re-validate the serialized batch shape
        # here before writing the create-only artifact.
        _validate_normalized_batch_prediction(prediction, cell)
        prediction = prediction.to(device="cpu", dtype=torch.long).contiguous()
        path = fc.expected_prediction_path(output_root, cell=cell, method_id=method_id)
        candidates = candidate_scores = None
        if method_id == M_A2:
            candidates, candidate_scores = adapter.candidate_tensors(
                records=cell.records,
                sequence_tokens=cell.sequence_tokens,
            )
        _write_prediction(
            path=path,
            cell=cell,
            method_id=method_id,
            predictions=prediction,
            candidates=candidates,
            candidate_scores=candidate_scores,
            binding=registration_data["bindings"][method_id],
            panel_sha256=panel_sha256,
            selection_plan_sha256=plan_sha256,
        )
        # Validate the artifact immediately, still before any possible
        # truth path is touched by another process.
        fc.validate_confirmation_prediction(
            path,
            cell=cell,
            panel_sha256=panel_sha256,
            selection_plan_sha256=plan_sha256,
            expected_method_id=method_id,
            expected_binding=registration_data["bindings"][method_id],
            candidate_policy=registration_data["candidate_policies"][method_id],
            repository_root=root,
        )
        peak = _peak_memory(device)
        per_cell_peak_memory[cell.cell_id] = dict(peak)
        timing_record = _timing_summary(
            timing,
            adapter=adapter,
            cell=cell,
            path=path,
            root=root,
            peak=peak,
        )
        timing_record["prediction_sha256"] = tensor_sha256(prediction)
        method_timings[method_id].append(timing_record)
        _json_dump(
            output_root / cell.style / cell.condition / f"{method_id}.run.json",
            {
                "schema": f"{SCRIPT_SCHEMA}-cell-method-v1",
                "task_id": TASK_ID,
                "status": "public_prediction_complete_no_truth",
                "cell": {"id": cell.cell_id, "style": cell.style, "condition": cell.condition, "shape": list(cell.shape)},
                "method": timing_record,
                "model": model_evidence,
            },
        )
        prediction_count += 1
        _resource_preflight(args, device, stage=f"after_{cell.cell_id}_{method_id}", started=started_perf)

    # This selected-method check proves completeness/integrity before a caller
    # hands this method's output root to the five-method truth gate.  It never
    # opens evaluator truth; the footing orchestrator performs the full 20-file
    # check after all five isolated processes have completed.
    validated = fc.validate_complete_confirmation_predictions(
        output_root,
        panel_path=panel_path,
        repository_root=root,
        method_ids=(method_id,),
        expected_bindings={method_id: registration_data["bindings"][method_id]},
        candidate_policies={method_id: registration_data["candidate_policies"][method_id]},
    )
    source_end = _source_records(
        root,
        reference_path=reference_path if method_id == M_A2 else None,
    )
    commit_end = _git_commit(root)
    status_end = _git_status(root)
    if commit_start != commit_end:
        raise PredictionRunnerError(f"git HEAD changed during prediction run: {commit_start} -> {commit_end}")
    if source_start != source_end:
        raise PredictionRunnerError("an executable source/resource binding changed during prediction run")
    if startup_boundary_perf is None or startup_boundary_utc is None:
        raise PredictionRunnerError("startup timing boundary was not captured")
    ended_perf = time.perf_counter()
    overall_peak_memory = _peak_memory_envelope([cold_peak_memory, *per_cell_peak_memory.values()])
    evidence = {
        "schema": SCRIPT_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH_OPENED",
        "claim_scope": "one selected method across four fresh public cells; no truth scoring or replacement claim",
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "wall_seconds": ended_perf - started_perf,
        "startup": {
            "started_utc": started_utc,
            "boundary_utc": startup_boundary_utc,
            "seconds": startup_boundary_perf - started_perf,
            "boundary": "entry to _run through immediately before the first run_warmed_prediction call",
            "includes": [
                "git/panel/selection/registration validation and hashing",
                "public resource loading",
                "selected method state loading",
                "CUDA initialization and resource probes",
                "largest-cell preflight",
            ],
            "excludes": [
                "Python interpreter and module-import time",
                "CLI parsing and argument/default resolution",
                "the first timed warmup and measured prediction calls",
            ],
        },
        "command": {
            "argv": [str(value) for value in sys.argv],
            "cwd": str(Path.cwd()),
            "python": sys.executable,
            "environment": {
                key: os.environ.get(key)
                for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM", "TRANSFORMERS_OFFLINE")
                if os.environ.get(key) is not None
            },
        },
        "git": {
            "commit_start": commit_start,
            "commit_end": commit_end,
            "status_start": status_start,
            "status_end": status_end,
            "source_files_unchanged": True,
        },
        "panel": {"path": str(panel_path), "sha256": panel_sha256, "cells": [cell.cell_id for cell in cells]},
        "selection_plan": {"path": str(plan_path), "sha256": plan_sha256},
        "registration": {
            "path": str(registration_path),
            "sha256": sha256_file(registration_path),
            "registered_method_ids": list(EXPECTED_METHOD_IDS),
            "executed_method_id": method_id,
        },
        "geometry_and_preflight": {
            "geometry": geometry,
            "before_resource_load": preflight,
            "limits": _resource_limits(args),
            "execution_order": [cell.cell_id for cell in cells],
            "largest_cell_first": cells[0].sequence_tokens == max(cell.sequence_tokens for cell in cells),
            "largest_cell_qualification": "reused exact registered 128-position geometry; largest cell executes first under the live guard",
        },
        "model": model_evidence,
        "peak_memory": overall_peak_memory,
        "cold_peak_memory": cold_peak_memory,
        "per_cell_peak_memory": per_cell_peak_memory,
        "source_records": source_end,
        "method_timings": method_timings,
        "prediction_artifacts": {
            "count": prediction_count,
            "validated_count": len(validated),
            "expected_cells": len(cells),
            "truth_opened": False,
            "full_five_method_gate_deferred": True,
        },
        "runtime_components": {
            method_id: {
                "public_prefix_calls": sum(int(row["method_specific"].get("public_prefix_calls", 0)) for row in rows),
                "candidate_simulations": sum(int(row["method_specific"].get("executed_candidate_simulations", row["method_specific"].get("candidate_simulations", 0))) for row in rows),
                "candidate_policy": registration_data["candidate_policies"][method_id],
                "a2_fallback": False,
                "public_prefix_loaded": method_id == M_A2,
            }
            for method_id, rows in method_timings.items()
        },
        "cold_costs": {
            "public_model_or_embedding_load": {
                "model_seconds": model_evidence.get("model_load_seconds", 0.0),
                "embedding_seconds": model_evidence.get("public_embedding_load_seconds", 0.0),
                "public_prefix_loaded": method_id == M_A2,
            },
            "method_state_load_seconds": method_load_seconds,
            "startup_seconds": startup_boundary_perf - started_perf,
            "startup_boundary": "entry to _run through immediately before the first run_warmed_prediction call",
            "cold_peak_memory": cold_peak_memory,
            "overall_peak_memory": overall_peak_memory,
            "steady_state": "only the per-record warmed timing rows; no cold cost is folded into token accuracy",
            "candidate_diagnostics": "A1 top-k is omitted; A2 first-measured candidate arrays are serialized as required and their proposal time is included in A2 steady timing",
        },
        "status_before_truth": "four cells for the selected method validated; full five-method gate deferred to the footing orchestrator",
    }
    _json_dump(output_root / "run_evidence.json", evidence)
    print(json.dumps({"status": evidence["status"], "output_root": str(output_root), "method": method_id, "predictions": prediction_count}, sort_keys=True))
    return evidence


def _failure_receipt(
    *,
    args: argparse.Namespace,
    exc: BaseException,
    started_utc: str,
    started_perf: float,
    root: Path,
    output_root: Path,
    commit_start: str | None,
    status_start: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": FAILURE_SCHEMA,
        "task_id": TASK_ID,
        "status": "FAILED_CLOSED_NO_TRUTH_OPENED",
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started_perf,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "command": {"argv": [str(value) for value in sys.argv], "cwd": str(Path.cwd()), "python": sys.executable},
        "git": {"commit_start": commit_start, "status_start": status_start, "commit_end": _git_commit(root) if root.exists() else None},
        "resource_limits": _resource_limits(args),
        "truth_opened": False,
    }
    if output_root.exists() and output_root.is_dir() and not output_root.is_symlink():
        try:
            _json_dump(output_root / "failure_receipt.json", result)
        except PredictionRunnerError:
            pass
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--method", choices=EXPECTED_METHOD_IDS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--lens", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_GIB)
    parser.add_argument("--maximum-reserved-gib", type=float, default=DEFAULT_MAXIMUM_RESERVED_GIB)
    parser.add_argument("--maximum-rss-gib", type=float, default=DEFAULT_MAXIMUM_RSS_GIB)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.expanduser().resolve()
    if args.model_snapshot is None:
        args.model_snapshot = Path(
            "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
            + fc.MODEL_REVISION
        )
    if args.reference is None:
        args.reference = _default_path(
            root,
            "outputs/TRR-0002/configuration-search/fresh-blind-code/reference/strict_bos/round001_teacher.py",
        )
    if args.lens is None:
        args.lens = _default_path(root, "outputs/TRR-0002/blind/reconstructor_input/public_a1_lens.pt")
    args.model_snapshot = args.model_snapshot.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    args.lens = args.lens.expanduser().resolve()
    started_perf = time.perf_counter()
    started_utc = utc_now()
    commit_start: str | None = None
    status_start: str | None = None
    try:
        commit_start = _git_commit(root)
        status_start = _git_status(root)
        _run(args)
        return 0
    except Exception as exc:
        failure = _failure_receipt(
            args=args,
            exc=exc,
            started_utc=started_utc,
            started_perf=started_perf,
            root=root,
            output_root=args.output_root.expanduser().resolve(),
            commit_start=commit_start,
            status_start=status_start,
        )
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
