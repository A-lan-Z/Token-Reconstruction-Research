#!/usr/bin/env python3
"""Run the registered TRR-0003 Track A checkpoint-only pilot.

The runner consumes only a sanitized boundary-activation panel and the pinned
public checkpoint.  It has no truth loader, no tokenizer, no fitted inverse,
and no candidate-by-candidate prefix simulation.  The iteration-zero cell is
the required identity projection from the observed boundary activation; the
remaining cells use the fixed reverse pre-norm residual iteration implemented
in checkpoint_inverse.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import save as save_tensors
import torch
from transformers import AutoModelForCausalLM

from token_reconstruction.checkpoint_inverse import (
    CheckpointInverseError,
    CheckpointInverseResult,
    clamp_known_bos,
    forward_public_embeddings,
    invert_public_prefix,
    nearest_public_embeddings,
)
from token_reconstruction.dual_benchmark import validate_observations
from token_reconstruction.experiment_runtime import synchronize, utc_now
from token_reconstruction.footing import (
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PREDICTION_SCHEMA,
    TASK_ID,
    FootingError,
    PanelCell,
    file_record,
    load_cell,
    load_panel,
    sha256_file,
)
from token_reconstruction.public_prefix import ContiguousPublicPrefix


METHOD_ID = "checkpoint_reverse_fixed_point_euclidean_k16"
METHOD_SCHEMA = "token-reconstruction.trr0003-track-a-registration.v1"
EVIDENCE_SCHEMA = "token-reconstruction.trr0003-track-a-evidence.v1"
RESOURCE_SCHEMA = "token-reconstruction.public-model-resource-manifest.v1"
BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
DEFAULT_ITERATIONS = (0, 1, 2, 4, 8, 16, 32)
DEFAULT_DAMPING = 0.5
DEFAULT_TOP_K = 16
DEFAULT_VOCAB_CHUNK_SIZE = 8192
DEFAULT_MIN_FREE_BYTES = 10 * 1024**3
DEFAULT_PROBE_BYTES = 600 * 1024**2
DEFAULT_MAX_SECONDS = 3600.0


class TrackAError(RuntimeError):
    """Raised when a Track A input, resource, or output contract fails."""


@dataclass(frozen=True)
class ObservationBundle:
    activations: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    activation_key: str
    metadata: dict[str, str]
    source_path: Path
    record_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResourceBundle:
    manifest_path: Path
    manifest_sha256: str
    snapshot_path: Path
    model: dict[str, str]
    files: tuple[dict[str, Any], ...]
    total_bytes: int


@dataclass(frozen=True)
class LoadedPublicState:
    prefix: ContiguousPublicPrefix
    embedding_weight: torch.Tensor
    prefix_digest: str
    embedding_digest: str
    parameter_bytes: int
    preparation_seconds: float
    preparation_peak: dict[str, int]


@dataclass
class ResourceGuard:
    started: float
    max_seconds: float

    def check(self, phase: str) -> None:
        elapsed = time.perf_counter() - self.started
        if elapsed > self.max_seconds:
            raise TrackAError(
                f"resource wall limit exceeded during {phase}: "
                f"{elapsed:.3f}s > {self.max_seconds:.3f}s"
            )


def _safe_relative(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrackAError(f"{description} path is absent")
    if "\\" in value:
        raise TrackAError(f"{description} path must use POSIX separators")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise TrackAError(f"{description} path is unsafe: {value}")
    return value


def _external_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrackAError(f"resource is unavailable: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _repo_record(path: Path, repository_root: Path) -> dict[str, Any]:
    try:
        return file_record(path, repository_root=repository_root)
    except FootingError as exc:
        raise TrackAError(str(exc)) from exc


def _json_load(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise TrackAError(f"JSON input must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackAError(f"invalid JSON: {path}") from exc


def _json_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise TrackAError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise TrackAError(f"refusing to overwrite artifact: {path}") from exc
    finally:
        if "descriptor" in locals() and descriptor >= 0:
            os.close(descriptor)


def _safetensors_create(
    path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]
) -> None:
    if path.exists() or path.is_symlink():
        raise TrackAError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = save_tensors(
            {
                key: value.detach().to(device="cpu").contiguous()
                for key, value in tensors.items()
            },
            metadata=dict(metadata),
        )
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise TrackAError(f"refusing to overwrite artifact: {path}") from exc
    finally:
        if "descriptor" in locals() and descriptor >= 0:
            os.close(descriptor)


def _metadata_private(value: str) -> bool:
    lowered = value.casefold().strip()
    return lowered not in {"", "0", "false", "no", "none", "null"}


def _validate_observation_metadata(metadata: Mapping[str, str]) -> None:
    private_fragments = (
        "truth",
        "oracle",
        "token_ids",
        "input_ids",
        "labels",
        "source_text",
        "source_material",
    )
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TrackAError("observation metadata must be string-valued")
        lowered = key.casefold().replace("-", "_")
        if any(fragment in lowered for fragment in private_fragments) and _metadata_private(value):
            raise TrackAError(f"observation metadata exposes private state: {key}")
    explicit_markers = (
        "truth_included",
        "source_truth_included",
        "source_material_included",
        "source_text_included",
    )
    present = [key for key in explicit_markers if key in metadata]
    if not present:
        raise TrackAError(
            "observation metadata lacks an explicit no-source/no-truth assertion"
        )
    if any(_metadata_private(metadata[key]) for key in present):
        raise TrackAError("observation metadata asserts private material is present")


def _activation_keys(keys: set[str]) -> list[str]:
    result = []
    for key in sorted(keys):
        if key in {"activation", "activations"} or key.endswith(".activation") or key.endswith(
            ".activations"
        ):
            result.append(key)
    return result


def _paired_key(activation_key: str, suffix: str) -> str:
    if activation_key in {"activation", "activations"}:
        return suffix
    base = activation_key.rsplit(".", 1)[0]
    return f"{base}.{suffix}"


def load_observation(
    path: Path, *, activation_key: str | None = None
) -> ObservationBundle:
    """Load a direct sanitized observation file without opening any truth."""

    if path.is_symlink() or not path.is_file():
        raise TrackAError(f"observation must be a regular file: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if not isinstance(metadata, dict):
                raise TrackAError("observation metadata is malformed")
            _validate_observation_metadata(metadata)
            keys = set(handle.keys())
            candidates = _activation_keys(keys)
            selected = activation_key
            if selected is None:
                if len(candidates) != 1:
                    raise TrackAError(
                        "activation key is ambiguous; provide --activation-key"
                    )
                selected = candidates[0]
            if selected not in keys or selected not in candidates:
                raise TrackAError(f"activation key is not an allowed public field: {selected}")
            allowed = {selected}
            mask_key = _paired_key(selected, "attention_mask")
            position_key = _paired_key(selected, "position_ids")
            for key in (mask_key, position_key):
                if key in keys:
                    allowed.add(key)
            unexpected = keys - allowed
            if unexpected:
                raise TrackAError(
                    f"observation contains unapproved tensor fields: {sorted(unexpected)}"
                )
            activations = handle.get_tensor(selected).contiguous()
            if mask_key in keys:
                attention_mask = handle.get_tensor(mask_key).to(torch.long).contiguous()
            else:
                attention_mask = torch.ones(activations.shape[:2], dtype=torch.long)
            if position_key in keys:
                position_ids = handle.get_tensor(position_key).to(torch.long).contiguous()
            else:
                position_ids = torch.arange(
                    activations.shape[1], dtype=torch.long
                ).view(1, -1).expand(activations.shape[0], -1)
    except TrackAError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrackAError(f"observation is unreadable: {path}") from exc
    if not activations.dtype.is_floating_point:
        raise TrackAError("observation activations must be floating point")
    try:
        validate_observations(activations, attention_mask, position_ids)
    except Exception as exc:
        raise TrackAError(f"observation geometry failed validation: {exc}") from exc
    return ObservationBundle(
        activations=activations,
        attention_mask=attention_mask,
        position_ids=position_ids,
        activation_key=str(selected),
        metadata=dict(metadata),
        source_path=path,
        record_ids=tuple(str(index) for index in range(activations.shape[0])),
    )


def _load_resource_manifest(
    path: Path, *, model_path: Path, model_id: str, revision: str
) -> ResourceBundle:
    raw = _json_load(path)
    if not isinstance(raw, dict) or raw.get("schema") != RESOURCE_SCHEMA:
        raise TrackAError("public resource manifest schema changed")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise TrackAError("public resource manifest model identity is absent")
    if model.get("id") != model_id or model.get("revision") != revision:
        raise TrackAError("public resource manifest model identity changed")
    snapshot_value = raw.get("snapshot_path")
    if not isinstance(snapshot_value, str):
        raise TrackAError("public resource manifest snapshot path is absent")
    snapshot = Path(snapshot_value).expanduser().resolve()
    if snapshot != model_path.expanduser().resolve():
        raise TrackAError("resource manifest does not bind the requested model path")
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise TrackAError("public model snapshot is not a regular directory")
    rows = raw.get("files")
    if not isinstance(rows, list) or not rows:
        raise TrackAError("public resource manifest has no files")
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TrackAError("public resource manifest file row is malformed")
        relative = _safe_relative(row.get("path"), description="public resource")
        if relative in seen:
            raise TrackAError("public resource manifest repeats a file")
        seen.add(relative)
        actual = snapshot / relative
        if actual.is_symlink():
            resolved = actual.resolve()
            if not resolved.is_file():
                raise TrackAError(f"public resource symlink is unavailable: {relative}")
        elif not actual.is_file():
            raise TrackAError(f"public resource is unavailable: {relative}")
        expected_bytes = row.get("bytes")
        expected_sha = row.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise TrackAError(f"public resource byte count is invalid: {relative}")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise TrackAError(f"public resource hash is invalid: {relative}")
        if int(actual.stat().st_size) != expected_bytes or sha256_file(actual) != expected_sha:
            raise TrackAError(f"public resource hash changed: {relative}")
        verified.append(
            {
                "path": relative,
                "bytes": expected_bytes,
                "sha256": expected_sha,
            }
        )
    suffixes = tuple(row["path"].casefold() for row in verified)
    if not any(value.endswith(".safetensors") for value in suffixes):
        raise TrackAError("public resource manifest does not bind model weights")
    if not any(value.endswith("config.json") for value in suffixes):
        raise TrackAError("public resource manifest does not bind model configuration")
    return ResourceBundle(
        manifest_path=path.resolve(),
        manifest_sha256=sha256_file(path),
        snapshot_path=snapshot,
        model={"id": model_id, "revision": revision},
        files=tuple(verified),
        total_bytes=sum(int(row["bytes"]) for row in verified),
    )


def _hash_tensor_bytes(digest: "hashlib._Hash", tensor: torch.Tensor) -> int:
    contiguous = tensor.detach().contiguous()
    raw = contiguous.view(torch.uint8).reshape(-1)
    byte_count = int(raw.numel())
    chunk = 8 * 1024 * 1024
    for start in range(0, byte_count, chunk):
        digest.update(raw[start : start + chunk].to(device="cpu").numpy().tobytes())
    return byte_count


def _hash_module_state(module: torch.nn.Module) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    tensor_count = 0
    total_bytes = 0
    named: list[tuple[str, torch.Tensor]] = []
    named.extend((name, value) for name, value in module.named_parameters())
    named.extend((name, value) for name, value in module.named_buffers())
    seen: set[str] = set()
    for name, value in sorted(named, key=lambda item: item[0]):
        if name in seen:
            continue
        seen.add(name)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(int(dim) for dim in value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        total_bytes += _hash_tensor_bytes(digest, value)
        tensor_count += 1
    if tensor_count == 0:
        raise TrackAError("public prefix has no hashable state")
    return digest.hexdigest(), total_bytes, tensor_count


def _hash_tensor(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(int(dim) for dim in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    _hash_tensor_bytes(digest, tensor)
    return digest.hexdigest()


def _configure_deterministic_execution() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _resource_preflight(
    *, min_free_bytes: int, probe_bytes: int
) -> dict[str, int | bool]:
    if not torch.cuda.is_available():
        raise TrackAError("Track A execution requires CUDA")
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError as exc:
        raise TrackAError("unable to query live CUDA memory") from exc
    if int(free_bytes) < min_free_bytes + probe_bytes:
        raise TrackAError(
            "live CUDA free memory is below the requested safety margin: "
            f"free={free_bytes} required={min_free_bytes + probe_bytes}"
        )
    probe_elements = max(1, math.ceil(probe_bytes / torch.tensor([], dtype=torch.bfloat16).element_size()))
    probe = None
    try:
        probe = torch.empty((probe_elements,), dtype=torch.bfloat16, device="cuda")
        probe.fill_(0)
        synchronize()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        raise TrackAError("CUDA allocation probe failed") from exc
    finally:
        del probe
    synchronize()
    return {
        "free_bytes_before_probe": int(free_bytes),
        "total_bytes": int(total_bytes),
        "probe_bytes_requested": int(probe_bytes),
        "probe_passed": True,
        "minimum_free_bytes_after_probe": int(min_free_bytes),
    }


def _load_public_state(
    *,
    model_path: Path,
    model_revision: str,
    resource: ResourceBundle,
    min_free_bytes: int,
) -> LoadedPublicState:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise TrackAError("Track A execution requires CUDA")
    device = torch.device("cuda")
    try:
        full = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                revision=model_revision,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
    except Exception as exc:
        raise TrackAError(f"pinned public model could not be loaded: {exc}") from exc
    full.requires_grad_(False)
    config = getattr(full, "config", None)
    if config is None or config.hidden_size != HIDDEN_SIZE or config.vocab_size != 128256:
        raise TrackAError("loaded public model geometry changed")
    try:
        prefix = ContiguousPublicPrefix(full, CUT_DEPTH).to(device).eval()
    except Exception as exc:
        raise TrackAError(f"public prefix construction failed: {exc}") from exc
    prefix.requires_grad_(False)
    embedding_weight = prefix.embed_tokens.weight
    if tuple(embedding_weight.shape) != (128256, HIDDEN_SIZE):
        raise TrackAError("loaded public embedding geometry changed")
    if embedding_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TrackAError("loaded public embedding dtype changed")
    del full
    synchronize()
    preparation_peak = {
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    torch.cuda.reset_peak_memory_stats()
    prefix_digest, parameter_bytes, _ = _hash_module_state(prefix)
    embedding_digest = _hash_tensor(embedding_weight)
    synchronize()
    if int(torch.cuda.mem_get_info()[0]) < min_free_bytes:
        raise TrackAError("public model load left less than the reserved CUDA margin")
    return LoadedPublicState(
        prefix=prefix,
        embedding_weight=embedding_weight,
        prefix_digest=prefix_digest,
        embedding_digest=embedding_digest,
        parameter_bytes=parameter_bytes,
        preparation_seconds=time.perf_counter() - started,
        preparation_peak=preparation_peak,
    )


def _relative_l2(error: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(error.float().reshape(-1))
    denominator = torch.linalg.vector_norm(reference.float().reshape(-1)).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _per_position_relative_l2(
    actual: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    error = (actual.float() - reference.float()).square().sum(dim=-1).sqrt()
    denominator = reference.float().square().sum(dim=-1).sqrt().clamp_min(1e-12)
    return (error / denominator).to(device="cpu", dtype=torch.float32)


def _valid_record(
    *,
    cell: PanelCell,
    record_index: int,
) -> tuple[int, torch.Tensor]:
    mask = cell.attention_mask[record_index].to(torch.bool)
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    valid_tokens = int(positions.numel())
    if valid_tokens < 2 or not torch.equal(
        positions, torch.arange(valid_tokens, dtype=torch.long)
    ):
        raise TrackAError(
            f"record {record_index} is not contiguous right-padded input"
        )
    expected = torch.arange(valid_tokens, dtype=torch.long)
    supplied = cell.position_ids[record_index, :valid_tokens].to(torch.long)
    if not torch.equal(supplied, expected):
        raise TrackAError(
            f"record {record_index} position IDs are not arange(valid_length)"
        )
    return valid_tokens, mask


def _prediction_validity(
    predictions: torch.Tensor,
    candidates: torch.Tensor,
    scores: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    selected: Sequence[int],
) -> None:
    selected_set = set(int(index) for index in selected)
    for row_index in range(predictions.shape[0]):
        mask = attention_mask[row_index].to(torch.bool)
        if row_index not in selected_set:
            if predictions[row_index].ne(INVALID_TOKEN_ID).any().item():
                raise TrackAError("unselected prediction row was populated")
            continue
        if int(predictions[row_index, 0]) != BOS_TOKEN_ID:
            raise TrackAError("prediction row does not begin with BOS")
        if predictions[row_index][mask].lt(0).any().item():
            raise TrackAError("active prediction contains an invalid token")
        if predictions[row_index][~mask].ne(INVALID_TOKEN_ID).any().item():
            raise TrackAError("padded prediction is not invalid")
        if candidates[row_index][mask].lt(0).any().item():
            raise TrackAError("active candidate contains an invalid token")
        if candidates[row_index][~mask].ne(INVALID_TOKEN_ID).any().item():
            raise TrackAError("padded candidate row was not invalid")
        active_scores = scores[row_index][mask]
        if not torch.isfinite(active_scores).all().item():
            raise TrackAError("active candidate score is non-finite")
        if scores[row_index][~mask].ne(float("-inf")).any().item():
            raise TrackAError("padded candidate score was not invalid")


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _resource_binding(resource: ResourceBundle) -> dict[str, Any]:
    return {
        "manifest": _external_record(resource.manifest_path),
        "snapshot_path": str(resource.snapshot_path),
        "model": resource.model,
        "files": [dict(row) for row in resource.files],
        "total_bytes": resource.total_bytes,
    }


def _method_binding(
    *,
    repository_root: Path,
    panel_path: Path,
    method_config_path: Path,
    resource: ResourceBundle,
    state: LoadedPublicState,
    algorithm: Mapping[str, Any],
) -> dict[str, Any]:
    code_paths = (
        Path(__file__).resolve(),
        repository_root / "src/token_reconstruction/checkpoint_inverse.py",
        repository_root / "src/token_reconstruction/public_prefix.py",
        repository_root / "src/token_reconstruction/dual_benchmark.py",
        repository_root / "src/token_reconstruction/footing.py",
    )
    code = [_repo_record(path, repository_root) for path in code_paths]
    method_state = [
        _repo_record(method_config_path, repository_root),
        _external_record(resource.manifest_path),
    ]
    code_commit = _git_head()
    if code_commit is None:
        raise TrackAError("Track A requires a full Git code commit before execution")
    return {
        "panel": _repo_record(panel_path, repository_root),
        "method_state": method_state,
        "code": code,
        "code_commit": code_commit,
        "public_resource": _resource_binding(resource),
        "loaded_public_prefix": {
            "state_sha256": state.prefix_digest,
            "embedding_sha256": state.embedding_digest,
            "parameter_bytes": state.parameter_bytes,
            "model_id": resource.model["id"],
            "model_revision": resource.model["revision"],
        },
        "algorithm": dict(algorithm),
    }


def _algorithm_config(
    *, iterations: Sequence[int], damping: float, top_k: int, vocab_chunk_size: int
) -> dict[str, Any]:
    return {
        "method_id": METHOD_ID,
        "method_schema": METHOD_SCHEMA,
        "architecture": "reverse_pre_norm_residual_fixed_point",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "cut_depth": CUT_DEPTH,
        "initial_estimate": "observed_boundary_activation",
        "reverse_order": "layer_{cut_depth_minus_one}_to_layer_0; mlp_then_attention",
        "fixed_point_update": "estimate <- (1-damping)*estimate + damping*(target-public_branch(estimate))",
        "damping": float(damping),
        "iterations": [int(value) for value in iterations],
        "zero_fit_identity_baseline": {
            "iterations": 0,
            "estimate": "observed_boundary_activation",
            "inversion_steps": 0,
            "projection": "same_exact_euclidean_public_embedding_scan",
        },
        "projection": "squared_euclidean",
        "top_k": top_k,
        "vocab_chunk_size": vocab_chunk_size,
        "projection_tie_break": "distance_then_token_id",
        "record_batch_size": 1,
        "candidate_prefix_simulations": 0,
        "truth_opened": False,
        "fitted_parameters": False,
        "auxiliary_training_steps": 0,
        "teacher_prefix_diagnostic": False,
        "branch_forward_calls_per_step": 2,
        "deterministic_execution": "torch_deterministic_algorithms_true; sdpa_math_only",
        "sequence_policy": "strip_only_right_padding; position_ids_must_equal_arange",
    }


def _validate_method_config(
    raw: Any,
    *,
    expected_algorithm: Mapping[str, Any],
) -> None:
    if not isinstance(raw, dict):
        raise TrackAError("Track A method config is not an object")
    if raw.get("schema") != METHOD_SCHEMA or raw.get("task_id") != TASK_ID:
        raise TrackAError("Track A method config identity changed")
    if raw.get("method_id") != METHOD_ID:
        raise TrackAError("Track A method config method ID changed")
    if raw.get("truth_opened") is not False or raw.get("fitted_parameters") is not False:
        raise TrackAError("Track A method config permits private or fitted state")
    declared = raw.get("algorithm")
    if declared != dict(expected_algorithm):
        raise TrackAError("Track A algorithm settings differ from registered settings")


def _diagnostic_metadata(
    *,
    panel_sha256: str,
    cell: PanelCell,
    binding: Mapping[str, Any],
    algorithm: Mapping[str, Any],
    iterations: int,
) -> dict[str, str]:
    return {
        "schema": EVIDENCE_SCHEMA,
        "task_id": TASK_ID,
        "method_id": METHOD_ID,
        "truth_opened": "false",
        "panel_sha256": panel_sha256,
        "cell_id": cell.cell_id,
        "style": cell.style,
        "condition": cell.condition,
        "iteration": str(iterations),
        "geometry_json": json.dumps(
            {
                "records": cell.records,
                "sequence_tokens": cell.sequence_tokens,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "binding_json": json.dumps(dict(binding), sort_keys=True, separators=(",", ":")),
    }


def _prediction_metadata(
    *,
    panel_sha256: str,
    cell: PanelCell,
    binding: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "schema": PREDICTION_SCHEMA,
        "task_id": TASK_ID,
        "panel_sha256": panel_sha256,
        "cell_id": cell.cell_id,
        "style": cell.style,
        "condition": cell.condition,
        "method_id": METHOD_ID,
        "truth_opened": "false",
        "geometry_json": json.dumps(
            {
                "records": cell.records,
                "sequence_tokens": cell.sequence_tokens,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "binding_json": json.dumps(dict(binding), sort_keys=True, separators=(",", ":")),
    }


def _process_iteration(
    *,
    cell: PanelCell,
    selected: Sequence[int],
    iterations: int,
    damping: float,
    top_k: int,
    vocab_chunk_size: int,
    prefix: ContiguousPublicPrefix,
    embedding_weight: torch.Tensor,
    guard: ResourceGuard,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    records, sequence_tokens = cell.records, cell.sequence_tokens
    predictions = torch.full(
        (records, sequence_tokens), INVALID_TOKEN_ID, dtype=torch.int32
    )
    candidates = torch.full(
        (records, sequence_tokens, top_k), INVALID_TOKEN_ID, dtype=torch.int32
    )
    candidate_scores = torch.full(
        (records, sequence_tokens, top_k), float("-inf"), dtype=torch.float32
    )
    continuous_residual = torch.full(
        (records, sequence_tokens), float("nan"), dtype=torch.float32
    )
    discrete_residual = torch.full(
        (records, sequence_tokens), float("nan"), dtype=torch.float32
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record_index in selected:
        guard.check(f"iteration {iterations}, record {record_index}")
        valid_tokens, _ = _valid_record(cell=cell, record_index=record_index)
        target = cell.activations[record_index, :valid_tokens].to(
            device="cuda", dtype=torch.float32
        ).unsqueeze(0)
        if not torch.isfinite(target).all().item():
            raise TrackAError("target activation became non-finite after device transfer")
        synchronize()
        record_started = time.perf_counter()
        result: CheckpointInverseResult | None = None
        if iterations == 0:
            estimate = clamp_known_bos(target, embedding_weight, bos_token_id=BOS_TOKEN_ID)
            branch_stats: list[dict[str, Any]] = []
            all_finite = bool(torch.isfinite(estimate).all().item())
        else:
            try:
                result = invert_public_prefix(
                    prefix,
                    target,
                    iterations=iterations,
                    damping=damping,
                )
            except CheckpointInverseError as exc:
                raise TrackAError(
                    f"checkpoint inversion failed at iteration {iterations}, "
                    f"record {record_index}: {exc}"
                ) from exc
            branch_stats = [item.as_dict() for item in result.branch_stats_reverse_order]
            all_finite = result.all_finite
            if not all_finite:
                raise TrackAError(
                    f"non-finite checkpoint inversion at iteration {iterations}, "
                    f"record {record_index}; cell is fail-closed"
                )
            estimate = clamp_known_bos(
                result.embedding_estimate, embedding_weight, bos_token_id=BOS_TOKEN_ID
            )
        if not torch.isfinite(estimate).all().item():
            raise TrackAError("non-finite embedding estimate; cell is fail-closed")
        top_ids, distances = nearest_public_embeddings(
            estimate[0],
            embedding_weight,
            top_k=top_k,
            vocab_chunk_size=vocab_chunk_size,
            normalize=False,
        )
        if not torch.isfinite(distances).all().item():
            raise TrackAError("non-finite public embedding projection")
        top_ids = top_ids.to(device="cuda", dtype=torch.long)
        predictions[record_index, :valid_tokens] = top_ids[:, 0].to(device="cpu", dtype=torch.int32)
        predictions[record_index, 0] = BOS_TOKEN_ID
        candidates[record_index, :valid_tokens] = top_ids.to(device="cpu", dtype=torch.int32)
        candidate_scores[record_index, :valid_tokens] = (-distances).to(
            device="cpu", dtype=torch.float32
        )
        continuous = forward_public_embeddings(prefix, estimate)
        discrete_input = embedding_weight[
            predictions[record_index, :valid_tokens].to(device="cuda", dtype=torch.long)
        ].unsqueeze(0)
        discrete = forward_public_embeddings(prefix, discrete_input)
        if not torch.isfinite(continuous).all().item() or not torch.isfinite(
            discrete
        ).all().item():
            raise TrackAError("non-finite public cycle diagnostic")
        continuous_pos = _per_position_relative_l2(continuous[0], target[0])
        discrete_pos = _per_position_relative_l2(discrete[0], target[0])
        continuous_residual[record_index, :valid_tokens] = continuous_pos
        discrete_residual[record_index, :valid_tokens] = discrete_pos
        branch_forward_calls = sum(
            2 * len(item.get("steps", [])) for item in branch_stats
        )
        cycle_layer_calls = 2 * CUT_DEPTH
        rows.append(
            {
                "record_index": int(record_index),
                "record_id": cell.record_ids[record_index],
                "valid_tokens": valid_tokens,
                "iterations": int(iterations),
                "zero_fit_identity": iterations == 0,
                "all_finite": bool(all_finite),
                "inference_seconds": time.perf_counter() - record_started,
                "branch_forward_calls": branch_forward_calls,
                "cycle_forward_passes": 2,
                "public_prefix_layer_evaluations": branch_forward_calls + cycle_layer_calls,
                "candidate_prefix_simulations": 0,
                "continuous_cycle_relative_l2_scored": _relative_l2(
                    continuous[0, 1:], target[0, 1:]
                ),
                "discrete_cycle_relative_l2_scored": _relative_l2(
                    discrete[0, 1:], target[0, 1:]
                ),
                "continuous_residual_by_position": continuous_pos.tolist(),
                "discrete_residual_by_position": discrete_pos.tolist(),
                "branch_stats_reverse_order": branch_stats,
            }
        )
        del target, estimate, top_ids, distances, continuous, discrete, discrete_input
        synchronize()
    active_records = len(rows)
    scored_positions = sum(max(0, int(row["valid_tokens"]) - 1) for row in rows)
    if rows:
        mean_continuous = sum(
            float(row["continuous_cycle_relative_l2_scored"]) for row in rows
        ) / active_records
        mean_discrete = sum(
            float(row["discrete_cycle_relative_l2_scored"]) for row in rows
        ) / active_records
    else:
        mean_continuous = math.nan
        mean_discrete = math.nan
    _prediction_validity(
        predictions,
        candidates,
        candidate_scores,
        attention_mask=cell.attention_mask,
        selected=selected,
    )
    aggregate = {
        "records": active_records,
        "scored_positions": scored_positions,
        "mean_continuous_cycle_relative_l2_scored": mean_continuous,
        "mean_discrete_cycle_relative_l2_scored": mean_discrete,
        "inference_seconds": time.perf_counter() - started,
        "branch_forward_calls": sum(int(row["branch_forward_calls"]) for row in rows),
        "cycle_forward_passes": 2 * active_records,
        "public_prefix_layer_evaluations": sum(
            int(row["public_prefix_layer_evaluations"]) for row in rows
        ),
        "candidate_prefix_simulations": 0,
        "truth_opened": False,
    }
    return (
        {
            "predictions": predictions,
            "candidates": candidates,
            "candidate_scores": candidate_scores,
            "continuous_residual": continuous_residual,
            "discrete_residual": discrete_residual,
        },
        rows,
        aggregate,
    )


def _parse_iterations(value: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if result != DEFAULT_ITERATIONS:
        raise TrackAError(
            f"registered iteration ladder is {DEFAULT_ITERATIONS}, received {result}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("experiments/TRR-0003/footing/panel.json"))
    parser.add_argument("--style", choices=("pile", "finance"), required=True)
    parser.add_argument(
        "--condition",
        choices=("public_base", "public_lora_2601"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=None,
        help="separate root for iteration diagnostics; defaults beside output-root",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument(
        "--method-config",
        type=Path,
        default=Path("experiments/TRR-0003/track_a/preregistration.json"),
    )
    parser.add_argument("--iterations", nargs="+", type=int, default=list(DEFAULT_ITERATIONS))
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--vocab-chunk-size", type=int, default=DEFAULT_VOCAB_CHUNK_SIZE
    )
    parser.add_argument("--record-indices", nargs="+", type=int, default=None)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--probe-bytes", type=int, default=DEFAULT_PROBE_BYTES)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    panel_path = args.panel.resolve()
    panel = load_panel(panel_path, repository_root=repository_root)
    cell = load_cell(
        panel,
        style=args.style,
        condition=args.condition,
        repository_root=repository_root,
    )
    iterations = _parse_iterations(args.iterations)
    if not math.isfinite(args.damping) or not 0.0 < args.damping <= 1.0:
        raise TrackAError("damping must lie in (0,1]")
    if args.top_k != DEFAULT_TOP_K:
        raise TrackAError("registered top_k is fixed at 16")
    if args.vocab_chunk_size <= 0:
        raise TrackAError("vocab chunk size must be positive")
    selected = (
        tuple(range(cell.records))
        if args.record_indices is None
        else tuple(int(index) for index in args.record_indices)
    )
    if not selected or len(set(selected)) != len(selected):
        raise TrackAError("record indices must be nonempty and unique")
    if any(index < 0 or index >= cell.records for index in selected):
        raise TrackAError("record index is outside the panel cell")
    output_root = args.output_root.resolve()
    diagnostic_root = (
        args.diagnostic_root.resolve()
        if args.diagnostic_root is not None
        else output_root.parent / f"{output_root.name}.track_a_diagnostics.{args.style}.{args.condition}"
    )
    if diagnostic_root.exists() or diagnostic_root.is_symlink():
        raise TrackAError(f"diagnostic root already exists: {diagnostic_root}")
    output_path = output_root / cell.style / cell.condition / f"{METHOD_ID}.safetensors"
    if set(selected) == set(range(cell.records)) and (
        output_path.exists() or output_path.is_symlink()
    ):
        raise TrackAError(f"prediction output already exists: {output_path}")
    raw_config = _json_load(args.method_config.resolve())
    algorithm = _algorithm_config(
        iterations=iterations,
        damping=args.damping,
        top_k=args.top_k,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    _validate_method_config(raw_config, expected_algorithm=algorithm)
    _configure_deterministic_execution()
    resource_bundle = _load_resource_manifest(
        args.resource_manifest.resolve(),
        model_path=args.model_path,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
    )
    panel_sha256 = sha256_file(panel_path)
    preflight = _resource_preflight(
        min_free_bytes=int(args.min_free_bytes),
        probe_bytes=int(args.probe_bytes),
    )
    started_utc = utc_now()
    guard = ResourceGuard(time.perf_counter(), float(args.max_seconds))
    try:
        state = _load_public_state(
            model_path=args.model_path,
            model_revision=MODEL_REVISION,
            resource=resource_bundle,
            min_free_bytes=int(args.min_free_bytes),
        )
        binding = _method_binding(
            repository_root=repository_root,
            panel_path=panel_path,
            method_config_path=args.method_config.resolve(),
            resource=resource_bundle,
            state=state,
            algorithm=algorithm,
        )
        diagnostic_root.mkdir(parents=True, exist_ok=False)
        iteration_reports: list[dict[str, Any]] = []
        tensor_by_iteration: dict[int, dict[str, torch.Tensor]] = {}
        for iteration in iterations:
            guard.check(f"before iteration {iteration}")
            tensors, rows, aggregate = _process_iteration(
                cell=cell,
                selected=selected,
                iterations=iteration,
                damping=args.damping,
                top_k=args.top_k,
                vocab_chunk_size=args.vocab_chunk_size,
                prefix=state.prefix,
                embedding_weight=state.embedding_weight,
                guard=guard,
            )
            diagnostic_path = diagnostic_root / cell.style / cell.condition / METHOD_ID / (
                f"iteration_{iteration:03d}.safetensors"
            )
            diagnostic_metadata = _diagnostic_metadata(
                panel_sha256=panel_sha256,
                cell=cell,
                binding=binding,
                algorithm=algorithm,
                iterations=iteration,
            )
            _safetensors_create(
                diagnostic_path,
                tensors,
                metadata=diagnostic_metadata,
            )
            tensor_by_iteration[iteration] = tensors
            iteration_reports.append(
                {
                    "iterations": iteration,
                    "zero_fit_identity": iteration == 0,
                    "records": rows,
                    "aggregate": aggregate,
                    "artifact": _external_record(diagnostic_path),
                }
            )
        final_iteration = iterations[-1]
        final_tensors = tensor_by_iteration[final_iteration]
        full_panel = set(selected) == set(range(cell.records))
        prediction_record: dict[str, Any] | None = None
        if full_panel:
            final_metadata = _prediction_metadata(
                panel_sha256=panel_sha256,
                cell=cell,
                binding=binding,
            )
            _safetensors_create(
                output_path,
                {
                    "predictions": final_tensors["predictions"],
                    "candidates": final_tensors["candidates"],
                    "candidate_scores": final_tensors["candidate_scores"],
                },
                metadata=final_metadata,
            )
            prediction_record = _external_record(output_path)
        guard.check("before evidence")
        method_peak = {
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        evidence_path = diagnostic_root / cell.style / cell.condition / METHOD_ID / "evidence.json"
        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "task_id": TASK_ID,
            "status": "PREDICTIONS_FROZEN" if full_panel else "DIAGNOSTIC_SUBSET_FROZEN",
            "method_id": METHOD_ID,
            "method_kind": "checkpoint_only_no_fit",
            "truth_opened": False,
            "source_material_included": False,
            "canonical_comparison_complete": False,
            "exploratory_status": "pilot; matched control is diagnostic only",
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "command": " ".join(str(value) for value in [sys.executable, *sys.argv]),
            "git_head_at_execution": _git_head(),
            "panel": {
                "path": str(panel_path),
                "sha256": panel_sha256,
                "cell_id": cell.cell_id,
                "style": cell.style,
                "condition": cell.condition,
                "record_ids": list(cell.record_ids),
                "selected_record_indices": list(selected),
                "records": cell.records,
                "sequence_tokens": cell.sequence_tokens,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
                "observation_dtype": str(cell.activations.dtype),
                "target_weight_available_to_method": False,
            },
            "algorithm": algorithm,
            "method_state_binding": binding,
            "public_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "resource_manifest": _resource_binding(resource_bundle),
                "loaded_prefix_sha256": state.prefix_digest,
                "loaded_embedding_sha256": state.embedding_digest,
                "loaded_parameter_bytes": state.parameter_bytes,
                "public_module_dtype": str(next(state.prefix.parameters()).dtype),
                "inverse_accumulation_dtype": "torch.float32",
                "public_forward_cycle_dtype": str(state.embedding_weight.dtype),
            },
            "preparation": {
                "fresh_training_steps": 0,
                "fresh_adaptation_steps": 0,
                "resource_validation_seconds": None,
                "model_load_and_state_digest_seconds": state.preparation_seconds,
                "retained_model_resource_bytes": resource_bundle.total_bytes,
                "retained_loaded_parameter_bytes": state.parameter_bytes,
            },
            "resource_preflight": preflight,
            "memory": {
                "preparation_peak": state.preparation_peak,
                "method_peak_after_reset": method_peak,
            },
            "cost_contract": {
                "record_batch_size": 1,
                "sequence_execution": "one unpadded valid record at a time",
                "branch_forward_calls_per_fixed_point_step": 2,
                "candidate_prefix_simulations": 0,
                "public_prefix_layer_evaluations_include_cycle_forwards": True,
            },
            "iterations": iteration_reports,
            "final_iteration": final_iteration,
            "prediction": prediction_record,
            "diagnostic_root": str(diagnostic_root),
            "accuracy": {
                "token_accuracy": None,
                "completely_reconstructed_records": None,
                "top_k_recall": None,
                "truth_opened": False,
            },
        }
        _json_create(evidence_path, evidence)
        return {
            "status": evidence["status"],
            "method_id": METHOD_ID,
            "cell_id": cell.cell_id,
            "final_iteration": final_iteration,
            "prediction": str(output_path) if prediction_record else None,
            "diagnostics": str(diagnostic_root),
            "truth_opened": False,
        }
    except Exception as exc:
        failure_path = diagnostic_root / "failure.json"
        if not failure_path.exists() and not failure_path.is_symlink():
            try:
                _json_create(
                    failure_path,
                    {
                        "schema": "token-reconstruction.trr0003-track-a-failure.v1",
                        "task_id": TASK_ID,
                        "method_id": METHOD_ID,
                        "status": "FAILED_CLOSED",
                        "truth_opened": False,
                        "cell_id": cell.cell_id,
                        "selected_record_indices": list(selected),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "git_head_at_execution": _git_head(),
                        "panel_sha256": panel_sha256,
                    },
                )
            except Exception:
                pass
        if isinstance(exc, TrackAError):
            raise
        raise TrackAError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (TrackAError, FootingError) as exc:
        print(
            json.dumps(
                {"status": "FAILED_CLOSED", "error": str(exc), "truth_opened": False},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
