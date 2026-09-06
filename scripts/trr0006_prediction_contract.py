"""Task-local TRR-0006 prediction contract and integrity helpers.

The TRR-0005 contract is intentionally fixed to 128 records and four cells, so
TRR-0006 keeps this parameterized contract separate.  It binds exactly the two
published enriched methods, the producer's source-free observation manifest,
and the runtime resources used by the prediction runner.  It contains no truth
reader and never loads source-token arrays.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch

TASK_ID = "TRR-0006"
REGISTRATION_SCHEMA = "token-reconstruction.trr0006-frozen-pair-prediction-registration.v1"
OBSERVATION_SCHEMA = "token-reconstruction.trr0006-public-observation-manifest.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0006-prediction.v1"
TIMING_SCHEMA = "token-reconstruction.trr0006-prediction-timing.v1"
RUN_SCHEMA = "token-reconstruction.trr0006-prediction-run.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0006-prediction-run-failure.v1"

# Keep the published TRR-0005 contract order so producer, runner, and scorer
# can join artifacts without sorting or inferring a new order.
CELL_ORDER = (
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
)
METHOD_IDS = (
    "enriched__affine_causal_h_attention128",
    "enriched__affine_trained_diagonal_attention128",
)
BASE_METHOD_IDS = {
    METHOD_IDS[0]: "affine_causal_h_attention128",
    METHOD_IDS[1]: "affine_trained_diagonal_attention128",
}
METHOD_RULES = {
    METHOD_IDS[0]: "joint affine path plus zero-initialized causal H_0..H_i attention correction",
    METHOD_IDS[1]: "joint affine path plus trained current-position-only diagonal attention correction",
}
# These are the exact public TRR-0005 files, rather than an arbitrary state
# that happens to deserialize with the same architecture.  The state source
# was the scientific implementation at ``da82``; ``3a7`` is the reviewed
# publication tree carrying those unchanged selected files.
SCIENTIFIC_SOURCE_COMMIT = "da82f6cac45e09ae83452198344c547553cb4433"
PUBLISHED_PARENT_COMMIT = "3a7e8f579e713c3e41d02639237042ca26fd019b"
POST_SCORE_MAINTENANCE_COMMIT = "1dba67a8dc75844727866cb4273da28a311df216"
PUBLISHED_STATE_BINDINGS = {
    METHOD_IDS[0]: {
        "path": "experiments/TRR-0005/joint_fit_qknorm_v1/enriched/affine_causal_h_attention128/selected.safetensors",
        "bytes": 20990668,
        "sha256": "ee910b14ad6f282bb933ea44ad24453cb5cce1470c65dbc09d8bcc16f3e8abfd",
        "source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "attention_mode": "causal",
        "attention_score_mode": "cosine_scale4",
        "selected_step": "1900",
    },
    METHOD_IDS[1]: {
        "path": "experiments/TRR-0005/joint_fit_v1/enriched/affine_trained_diagonal_attention128/selected.safetensors",
        "bytes": 20990652,
        "sha256": "696eb9fc951e85356a06575faf18a2011616692a086bdac3b2fa368e69d599a2",
        "source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "attention_mode": "diagonal",
        "attention_score_mode": "dot_product",
        "selected_step": "1600",
    },
}
# Every file here is imported or executed by the prediction process.  The
# third entry is the numerical implementation behind load_decoder_state; the
# retained TRR-0005 runner is deliberately not claimed as an executed input.
CODE_BINDING_SPECS = (
    ("prediction_runner", "scripts/trr0006_run_predictions.py"),
    ("prediction_contract", "scripts/trr0006_prediction_contract.py"),
    ("frozen_decoder_numerics", "src/token_reconstruction/trr0005_joint_decoder.py"),
)
STYLE_ORDER = ("pile", "finance")
CONDITION_ORDER = ("public_base", "public_lora_2601")
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
STORED_SEQUENCE_TOKENS = 128
SCORED_SEQUENCE_TOKENS = STORED_SEQUENCE_TOKENS
SCORED_POST_BOS_TOKENS = STORED_SEQUENCE_TOKENS - 1
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
NORMALIZED_PUBLIC_E_BYTES = 1050673488
NORMALIZED_PUBLIC_E_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
# These values match the already-qualified fixture command/environment.  The
# runner sets and records them before CUDA or model work begins.
NUMERICAL_SETTINGS = {
    "activation_input_dtype": "torch.bfloat16",
    "staged_activation_dtype": "torch.float32",
    "staged_mask_dtype": "torch.bool",
    "decoder_compute_dtype": "torch.float32",
    "embedding_dtype": "torch.float32",
    "autocast": False,
    "cuda_matmul_allow_tf32": False,
    "cuda_cudnn_allow_tf32": True,
    "float32_matmul_precision": "highest",
    "cpu_intraop_threads": 8,
    "cpu_interop_threads": 32,
}
MIN_FREE_GPU_BYTES = 8 * 2**30
MAX_RESERVED_GPU_BYTES = 6 * 2**30
MAX_RSS_BYTES = 16 * 2**30
MIN_HOST_AVAILABLE_BYTES = 10 * 2**30

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    """Raised when a TRR-0006 binding is incomplete or changed."""


def sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"asset is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{description} must be an object")
    return value


def resolve_path(value: Any, *, repository_root: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{description} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{description} is unavailable: {path}")
    return path


def validate_file_record(
    value: Mapping[str, Any],
    *,
    repository_root: Path,
    description: str,
    verify: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{description} binding is not an object")
    path = resolve_path(value.get("path"), repository_root=repository_root, description=description)
    raw_bytes = value.get("bytes")
    digest = value.get("sha256")
    if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes <= 0:
        raise ContractError(f"{description} bytes are invalid")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ContractError(f"{description} SHA-256 is invalid")
    record = {"path": str(path), "bytes": int(raw_bytes), "sha256": digest}
    if verify:
        actual_bytes = int(path.stat().st_size)
        actual_digest = sha256_file(path)
        if actual_bytes != record["bytes"] or actual_digest != record["sha256"]:
            raise ContractError(f"{description} binding does not match the file: {path}")
    return record


def _require_int(value: Any, *, description: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{description} must be an integer >= {minimum}")
    return int(value)


def _require_hash(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContractError(f"{description} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ContractError(f"{description} must be a full lowercase commit hash")
    return value


def validate_published_state_binding(state: Mapping[str, Any], method_id: str) -> dict[str, Any]:
    """Validate the exact retained TRR-0005 selected state for one method."""

    if method_id not in PUBLISHED_STATE_BINDINGS:
        raise ContractError(f"unknown frozen method: {method_id}")
    expected = PUBLISHED_STATE_BINDINGS[method_id]
    if not isinstance(state, Mapping):
        raise ContractError(f"state binding is malformed: {method_id}")
    for key in ("path", "bytes", "sha256", "source_commit"):
        if state.get(key) != expected[key]:
            raise ContractError(f"published state binding changed for {method_id}: {key}")
    return dict(expected)


def validate_published_state_metadata(metadata: Mapping[str, Any], method_id: str) -> dict[str, Any]:
    """Validate state metadata before load_decoder_state constructs a model."""

    if not isinstance(metadata, Mapping) or method_id not in PUBLISHED_STATE_BINDINGS:
        raise ContractError(f"state metadata is malformed: {method_id}")
    expected = PUBLISHED_STATE_BINDINGS[method_id]
    base_method_id = BASE_METHOD_IDS[method_id]
    exact = {
        "schema": "token-reconstruction.trr0005-joint-decoder.v1",
        "task_id": "TRR-0005",
        "canonical_method_id": method_id,
        "method_id": base_method_id,
        "distribution": "enriched",
        "attention_mode": expected["attention_mode"],
        "context_width": "128",
        "qkv_init_seed": "4005",
        "selected_step": expected["selected_step"],
    }
    for key, value in exact.items():
        if str(metadata.get(key)) != value:
            raise ContractError(f"published state metadata changed for {method_id}: {key}")
    # The selected diagonal state predates explicit score-mode metadata and
    # load_decoder_state's frozen default is dot_product.  Causal must carry
    # the repaired cosine/QK-normalized marker explicitly.
    score_mode = metadata.get("attention_score_mode", "dot_product")
    if score_mode != expected["attention_score_mode"]:
        raise ContractError(f"published state attention score changed for {method_id}")
    result = dict(exact)
    result["attention_score_mode"] = score_mode
    return result


def validate_numerical_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise ContractError("numerical settings are absent")
    for key, expected in NUMERICAL_SETTINGS.items():
        if settings.get(key) != expected:
            raise ContractError(f"numerical setting changed: {key}")
    if set(settings) != set(NUMERICAL_SETTINGS):
        raise ContractError("numerical settings contain an unregistered field")
    return dict(NUMERICAL_SETTINGS)


def validate_registration(registration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate shape and frozen decisions without reading assets."""
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise ContractError("TRR-0006 registration schema changed")
    if registration.get("task_id") != TASK_ID:
        raise ContractError("TRR-0006 registration task ID changed")
    if registration.get("status") != "FROZEN_PREDICTION_REGISTRATION":
        raise ContractError("prediction registration is not frozen")
    count = _require_int(registration.get("records_per_domain"), description="records_per_domain")
    if count % CAPTURE_BATCH_RECORDS != 0:
        raise ContractError("records_per_domain must be divisible by capture batch size 8")
    if registration.get("cell_order") != list(CELL_ORDER):
        raise ContractError("cell order changed")
    if registration.get("method_ids") != list(METHOD_IDS):
        raise ContractError("method order changed")
    geometry = registration.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ContractError("registration geometry is absent")
    expected_geometry = {
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        # This is the stored coordinate width including BOS.  The explicit
        # post-BOS field below is the metric denominator used by the scorer.
        "scored_sequence_tokens": SCORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "chunk_records": CAPTURE_BATCH_RECORDS,
    }
    for key, expected in expected_geometry.items():
        if geometry.get(key) != expected:
            raise ContractError(f"registration geometry changed: {key}")
    if registration.get("truth_opened") is not False:
        raise ContractError("registration truth flag is not closed")
    if registration.get("candidate_arrays_persisted") is not False:
        raise ContractError("candidate arrays are not permitted")
    runtime = registration.get("runtime_assets")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("normalized_public_E"), Mapping):
        raise ContractError("normalized public E binding is absent")
    embedding = runtime["normalized_public_E"]
    if embedding.get("bytes") != NORMALIZED_PUBLIC_E_BYTES or embedding.get("sha256") != NORMALIZED_PUBLIC_E_SHA256:
        raise ContractError("normalized public E binding changed")
    if embedding.get("shape") != [VOCAB_SIZE, HIDDEN_SIZE] or embedding.get("dtype") != "torch.float32":
        raise ContractError("normalized public E geometry or dtype changed")
    methods = registration.get("methods")
    if not isinstance(methods, Mapping) or list(methods) != list(METHOD_IDS):
        raise ContractError("method bindings are incomplete or reordered")
    for method_id in METHOD_IDS:
        row = methods[method_id]
        if not isinstance(row, Mapping):
            raise ContractError(f"method binding is malformed: {method_id}")
        if row.get("base_method_id") != BASE_METHOD_IDS[method_id]:
            raise ContractError(f"base method binding changed: {method_id}")
        if row.get("decision_rule") != METHOD_RULES[method_id]:
            raise ContractError(f"decision rule changed: {method_id}")
        state = row.get("state")
        if not isinstance(state, Mapping):
            raise ContractError(f"state binding is absent: {method_id}")
        validate_published_state_binding(state, method_id)
    observation = registration.get("observation_manifest")
    if not isinstance(observation, Mapping):
        raise ContractError("observation manifest binding is absent")
    _require_hash(observation.get("sha256"), description="observation manifest")
    output_root = registration.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ContractError("output_root is absent")
    timing = registration.get("timing_contract")
    if not isinstance(timing, Mapping):
        raise ContractError("timing contract is absent")
    if timing.get("warmup_runs_per_record") != 1 or timing.get("measured_runs_per_record") != 1:
        raise ContractError("TRR-0006 requires exactly one warmup and one measured call")
    if timing.get("repeat_integrity") != "Require warmup and measured predicted IDs to match exactly":
        raise ContractError("timing repeat-integrity rule changed")
    guard = registration.get("resource_guard")
    if not isinstance(guard, Mapping):
        raise ContractError("resource guard is absent")
    for key in ("minimum_free_gpu_bytes", "maximum_reserved_gpu_bytes", "maximum_rss_bytes", "minimum_host_available_bytes", "maximum_seconds"):
        _require_int(guard.get(key), description=f"resource guard {key}")
    validate_numerical_settings(registration.get("numerical_settings"))
    code = registration.get("code_bindings")
    if not isinstance(code, list) or len(code) != len(CODE_BINDING_SPECS):
        raise ContractError("code bindings must cover the runner, contract, and decoder numerics")
    for index, (role, path) in enumerate(CODE_BINDING_SPECS):
        row = code[index]
        if not isinstance(row, Mapping):
            raise ContractError(f"code binding {index} is malformed")
        if row.get("role") != role or row.get("path") != path:
            raise ContractError(f"required code binding {role} is missing or reordered")
        _require_hash(row.get("sha256"), description=f"code binding {index}")
        _require_int(row.get("bytes"), description=f"code binding {index} bytes")
    _require_commit(registration.get("code_commit"), description="registration code_commit")
    return dict(registration)


def load_registration(path: Path) -> dict[str, Any]:
    value = load_json(path, description="TRR-0006 prediction registration")
    return validate_registration(value)


def validate_observation_manifest(
    manifest: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    repository_root: Path,
    verify_assets: bool = True,
) -> dict[str, Any]:
    if manifest.get("schema") != OBSERVATION_SCHEMA:
        raise ContractError("observation manifest schema changed")
    if manifest.get("task_id") != TASK_ID:
        raise ContractError("observation manifest task ID changed")
    if manifest.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise ContractError("observation manifest is not frozen")
    count = registration["records_per_domain"]
    if manifest.get("records_per_domain") != count:
        raise ContractError("observation record count differs from registration")
    if manifest.get("cell_order") != list(CELL_ORDER):
        raise ContractError("observation cell order changed")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or [row.get("cell_id") for row in cells if isinstance(row, Mapping)] != list(CELL_ORDER):
        raise ContractError("observation cells are incomplete or reordered")
    parsed: dict[str, Any] = {}
    pair_digests: dict[str, str] = {}
    for row in cells:
        if not isinstance(row, Mapping):
            raise ContractError("observation cell row is malformed")
        cell_id = row.get("cell_id")
        if cell_id not in CELL_ORDER:
            raise ContractError(f"unknown observation cell: {cell_id}")
        style, condition = str(cell_id).split("__", 1)
        if row.get("style") != style or row.get("condition") != condition:
            raise ContractError(f"observation cell identity changed: {cell_id}")
        record_digest = _require_hash(row.get("record_ids_sha256"), description=f"record IDs {cell_id}")
        pair_digests.setdefault(style, record_digest)
        if pair_digests[style] != record_digest:
            raise ContractError(f"public-base/LoRA source pairing changed: {style}")
        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise ContractError(f"observation binding is absent: {cell_id}")
        shape = observation.get("shape")
        if shape != [count, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE]:
            raise ContractError(f"stored observation shape changed: {cell_id}")
        if observation.get("stored_sequence_tokens") != STORED_SEQUENCE_TOKENS:
            raise ContractError(f"stored sequence length changed: {cell_id}")
        if observation.get("scored_post_bos_tokens") != SCORED_POST_BOS_TOKENS:
            raise ContractError(f"scored post-BOS length changed: {cell_id}")
        if observation.get("capture_batch_records") != CAPTURE_BATCH_RECORDS:
            raise ContractError(f"capture batch provenance changed: {cell_id}")
        if observation.get("capture_sequence_tokens") != CAPTURE_SEQUENCE_TOKENS:
            raise ContractError(f"capture sequence provenance changed: {cell_id}")
        if observation.get("activations_key") != "activations" or observation.get("attention_mask_key") != "attention_mask" or observation.get("position_ids_key") != "position_ids":
            raise ContractError(f"observation tensor keys changed: {cell_id}")
        if observation.get("public_full_forward") is not True:
            raise ContractError(f"public full-forward provenance missing: {cell_id}")
        expected_lora = condition == "public_lora_2601"
        if observation.get("producer_only_lora") is not expected_lora:
            raise ContractError(f"producer-only LoRA provenance changed: {cell_id}")
        binding = validate_file_record(
            observation,
            repository_root=repository_root,
            description=f"observation {cell_id}",
            verify=verify_assets,
        )
        parsed[cell_id] = {
            "cell_id": cell_id,
            "style": style,
            "condition": condition,
            "record_ids_sha256": record_digest,
            "observation": binding,
            "shape": list(shape),
            "capture_batch_records": CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
            "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
            "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        }
    return {"records_per_domain": count, "cell_order": list(CELL_ORDER), "cells": parsed, "record_ids_sha256": pair_digests}


def load_observation_manifest(
    registration: Mapping[str, Any],
    *,
    repository_root: Path,
    verify_assets: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding = registration["observation_manifest"]
    path = resolve_path(binding.get("path"), repository_root=repository_root, description="observation manifest")
    actual = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    if actual["bytes"] != binding.get("bytes") or actual["sha256"] != binding.get("sha256"):
        raise ContractError("observation manifest binding does not match the file")
    manifest = load_json(path, description="observation manifest")
    parsed = validate_observation_manifest(
        manifest,
        registration=registration,
        repository_root=repository_root,
        verify_assets=verify_assets,
    )
    return manifest, parsed, actual


def normalize_prediction(raw: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(raw, dtype=torch.long).detach().cpu().contiguous()
    mask = torch.as_tensor(valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    if values.ndim != 1 or mask.ndim != 1 or tuple(values.shape) != tuple(mask.shape):
        raise ContractError("prediction and mask vectors differ")
    if not bool(mask[0].item()):
        raise ContractError("prediction mask has no BOS")
    output = torch.full_like(values, INVALID_TOKEN_ID)
    output[mask] = values[mask]
    output[0] = BOS_TOKEN_ID
    active = output[mask]
    if active.lt(0).any().item() or active.ge(VOCAB_SIZE).any().item():
        raise ContractError("prediction contains an invalid active ID")
    return output


def validate_prediction_tensor(
    predictions: torch.Tensor,
    *,
    records: int,
    sequence_tokens: int = STORED_SEQUENCE_TOKENS,
) -> torch.Tensor:
    value = torch.as_tensor(predictions, dtype=torch.long).detach().cpu().contiguous()
    if tuple(value.shape) != (records, sequence_tokens):
        raise ContractError(f"prediction shape is not [{records}, {sequence_tokens}]")
    if value.lt(INVALID_TOKEN_ID).any().item():
        raise ContractError("prediction contains an invalid padding ID")
    if not value[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise ContractError("prediction BOS column changed")
    active = value.ge(0)
    if active[:, 0].logical_not().any().item():
        raise ContractError("prediction has an invalid BOS")
    if value[active].ge(VOCAB_SIZE).any().item():
        raise ContractError("prediction active IDs exceed the vocabulary")
    for row in range(records):
        invalid = (~active[row]).nonzero(as_tuple=False).flatten()
        if invalid.numel() and active[row, int(invalid[0].item()) + 1 :].any().item():
            raise ContractError("prediction padding is not a suffix")
    return value


def write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
