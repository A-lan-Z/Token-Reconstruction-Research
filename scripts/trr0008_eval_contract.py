"""Small immutable contract for the TRR-0008 frozen-model evaluation.

The TRR-0007 runner already established the tensor convention used here.  This
module carries only the TRR-0008 bindings: four scientific methods, an
optional same-weight timing alias, four paired public cells, and a source-free
observation/prediction artifact format.  It deliberately has no truth loader
and no source-selection code.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch
from safetensors import safe_open


TASK_ID = "TRR-0008"
REGISTRATION_SCHEMA = "token-reconstruction.trr0008-frozen-evaluation-registration.v1"
OBSERVATION_SCHEMA = "token-reconstruction.trr0008-public-observation-manifest.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0008-prediction.v1"
TIMING_SCHEMA = "token-reconstruction.trr0008-prediction-timing.v1"
RUN_SCHEMA = "token-reconstruction.trr0008-prediction-run.v1"
SCORE_SCHEMA = "token-reconstruction.trr0008-score.v1"

CELL_ORDER = (
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
)
DOMAIN_ORDER = ("pile", "finance")
TARGET_ORDER = ("public_base", "public_lora_2601")

REFERENCE_METHOD_ID = "trr6__enriched_trained_diagonal_attention128"
CURRENT_RESIDUAL_METHOD_ID = "current_enriched__residual_mlp512"
IMPROVED_RESIDUAL_METHOD_ID = "improved_public_bank__residual_mlp512"
IMPROVED_DIAGONAL_METHOD_ID = "improved_public_bank__trained_diagonal"
# This is a loader/order control only.  It is intentionally excluded from the
# scientific method order and from all scoring decisions.
TIMING_CONTROL_METHOD_ID = "current_enriched__trained_diagonal"
METHOD_ORDER = (
    REFERENCE_METHOD_ID,
    CURRENT_RESIDUAL_METHOD_ID,
    IMPROVED_RESIDUAL_METHOD_ID,
    IMPROVED_DIAGONAL_METHOD_ID,
)
TIMING_METHOD_ORDER = (*METHOD_ORDER, TIMING_CONTROL_METHOD_ID)
PRIMARY_METHOD_ID = IMPROVED_RESIDUAL_METHOD_ID
NO_A2_METHOD_ID = "bounded_a1_a2_k256_p0"

METHOD_MODEL_IDS = {
    CURRENT_RESIDUAL_METHOD_ID: "trr0007_residual_mlp512",
    IMPROVED_RESIDUAL_METHOD_ID: "trr0007_residual_mlp512",
    IMPROVED_DIAGONAL_METHOD_ID: "trr0007_current_positionwise",
    TIMING_CONTROL_METHOD_ID: "trr0007_current_positionwise",
}
METHOD_SUPPORT = {
    CURRENT_RESIDUAL_METHOD_ID: "current_enriched",
    IMPROVED_RESIDUAL_METHOD_ID: "improved_public_bank",
    IMPROVED_DIAGONAL_METHOD_ID: "improved_public_bank",
    TIMING_CONTROL_METHOD_ID: "current_enriched",
}
METHOD_CAPACITY = {
    CURRENT_RESIDUAL_METHOD_ID: "residual_mlp512",
    IMPROVED_RESIDUAL_METHOD_ID: "residual_mlp512",
    IMPROVED_DIAGONAL_METHOD_ID: "trained_diagonal",
    TIMING_CONTROL_METHOD_ID: "trained_diagonal",
}

REFERENCE_LOADER = {
    "module": "token_reconstruction.trr0005_joint_decoder",
    "function": "load_decoder_state",
    "kwargs": {
        "context_width": 128,
        "hidden_size": 2048,
        "method_id": "affine_trained_diagonal_attention128",
        "vocabulary_size": 128256,
    },
}
POSITIONWISE_LOADER = {
    "module": "token_reconstruction.trr0007_positionwise",
    "function": "load_positionwise_model_state",
    "kwargs": {
        "context_width": 128,
        "hidden_size": 2048,
        "vocabulary_size": 128256,
    },
}

HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
STORED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
# Existing TRR-0007 public observations use 128 rows per domain. This is a
# timing-fixture constant only; fresh TRR-0008 rows are bound by the plan.
TIMING_RECORDS_PER_DOMAIN = 128
RECORDS_PER_DOMAIN = TIMING_RECORDS_PER_DOMAIN
CHUNK_RECORDS = 8

STATIC_GEOMETRY = {
    "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
    "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
    "hidden_size": HIDDEN_SIZE,
    "vocabulary_size": VOCABULARY_SIZE,
    "chunk_records": CHUNK_RECORDS,
}
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
RESOURCE_GUARD = {
    "minimum_free_gpu_bytes": 8 * 2**30,
    "maximum_reserved_gpu_bytes": 6 * 2**30,
    "maximum_rss_bytes": 16 * 2**30,
    "minimum_host_available_bytes": 10 * 2**30,
    "maximum_seconds": 600,
}
REFERENCE_STATE_SHA256 = "696eb9fc951e85356a06575faf18a2011616692a086bdac3b2fa368e69d599a2"
PUBLIC_E_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    """Raised for an incomplete or changed TRR-0008 binding."""


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
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
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


def canonical_json_digest(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("value cannot be canonically encoded") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def resolve_path(value: Any, *, repository_root: Path, description: str, require_file: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{description} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(repository_root).expanduser().resolve() / path
    path = path.resolve()
    if path.is_symlink():
        raise ContractError(f"{description} is a symlink: {path}")
    if require_file and not path.is_file():
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
        raise ContractError(f"{description} binding is malformed")
    path = resolve_path(value.get("path"), repository_root=repository_root, description=description)
    try:
        size = int(value.get("bytes"))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{description} byte count is malformed") from exc
    digest = value.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ContractError(f"{description} hash is malformed")
    record = {"path": str(path), "bytes": size, "sha256": digest}
    if verify:
        actual_size = int(path.stat().st_size)
        actual_digest = sha256_file(path)
        if actual_size != size or actual_digest != digest:
            raise ContractError(f"{description} hash or size changed")
    return record


def _as_cells(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("cells")
    if isinstance(raw, Mapping):
        result = {str(k): v for k, v in raw.items() if isinstance(v, Mapping)}
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        result = {
            str(row.get("cell_id")): row
            for row in raw
            if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
        }
    else:
        raise ContractError("observation cells are absent")
    if set(result) != set(CELL_ORDER):
        raise ContractError("observation cell order or membership changed")
    return result


def records_for_cell(manifest_or_registration: Mapping[str, Any], cell_id: str) -> int:
    """Return a bound record count without assuming 128 fresh rows."""

    if cell_id not in CELL_ORDER:
        raise ContractError(f"unknown cell: {cell_id}")
    cells = _as_cells(manifest_or_registration)
    cell = cells[cell_id]
    raw = cell.get("records")
    if raw is None and isinstance(cell.get("observation"), Mapping):
        raw = cell["observation"].get("records")
    if raw is None:
        counts = manifest_or_registration.get("records_by_domain")
        if isinstance(counts, Mapping):
            raw = counts.get(cell_id.split("__", 1)[0])
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"record count is absent for {cell_id}") from exc
    if count <= 0:
        raise ContractError(f"record count is non-positive for {cell_id}")
    return count


def validate_observation_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
    verify_assets: bool = True,
) -> dict[str, Any]:
    if manifest.get("task_id") not in (TASK_ID, "TRR-0007"):
        raise ContractError("observation task identity changed")
    schema = manifest.get("schema")
    if schema not in (OBSERVATION_SCHEMA, "token-reconstruction.trr0007-public-observation-manifest.v1"):
        raise ContractError("observation schema is not a public observation manifest")
    if manifest.get("truth_opened") is True or manifest.get("target_labels_loaded") is True:
        raise ContractError("observation manifest records truth or labels")
    if manifest.get("source_text_loaded") is True or manifest.get("source_text_written") is True:
        raise ContractError("observation manifest records source text")
    cells = _as_cells(manifest)
    counts: dict[str, int] = {}
    checked_cells: list[dict[str, Any]] = []
    for cell_id in CELL_ORDER:
        cell = cells[cell_id]
        if cell.get("cell_id") != cell_id:
            raise ContractError(f"observation cell ID changed: {cell_id}")
        observation = cell.get("observation")
        if not isinstance(observation, Mapping):
            observation = cell
        record = validate_file_record(
            observation,
            repository_root=repository_root,
            description=f"observation {cell_id}",
            verify=verify_assets,
        )
        count = records_for_cell(manifest, cell_id)
        counts[cell_id.split("__", 1)[0]] = count
        shape = observation.get("shape")
        if list(shape or []) != [count, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE]:
            raise ContractError(f"observation geometry changed: {cell_id}")
        for key in ("activations_key", "attention_mask_key", "position_ids_key"):
            if observation.get(key) != {"activations_key": "activations", "attention_mask_key": "attention_mask", "position_ids_key": "position_ids"}[key]:
                raise ContractError(f"observation tensor key changed: {cell_id}")
        checked_cells.append(dict(cell) | {"records": count, "observation": record})
    bound_counts = manifest.get("records_by_domain")
    if bound_counts is not None and dict(bound_counts) != counts:
        raise ContractError("observation records_by_domain changed")
    return dict(manifest) | {"cells": checked_cells, "records_by_domain": counts}


def normalize_prediction(raw: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(raw, dtype=torch.long).detach().cpu().contiguous()
    mask = torch.as_tensor(valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    if values.ndim != 1 or mask.ndim != 1 or values.shape != mask.shape:
        raise ContractError("prediction and mask geometry differ")
    if values.numel() != STORED_SEQUENCE_TOKENS or not bool(mask[0].item()):
        raise ContractError("prediction must contain 128 positions and BOS")
    output = torch.full_like(values, INVALID_TOKEN_ID)
    output[mask] = values[mask]
    output[0] = BOS_TOKEN_ID
    active = output[mask]
    if active.lt(0).any().item() or active.ge(VOCABULARY_SIZE).any().item():
        raise ContractError("prediction contains an invalid vocabulary ID")
    return output


def validate_prediction_tensor(predictions: torch.Tensor, *, records: int) -> torch.Tensor:
    value = torch.as_tensor(predictions, dtype=torch.long).detach().cpu().contiguous()
    if tuple(value.shape) != (records, STORED_SEQUENCE_TOKENS):
        raise ContractError(f"prediction shape changed: expected {(records, STORED_SEQUENCE_TOKENS)}")
    if value[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise ContractError("prediction BOS column changed")
    active = value.ge(0)
    if active[:, 0].logical_not().any().item():
        raise ContractError("prediction BOS became invalid")
    if value[active].ge(VOCABULARY_SIZE).any().item():
        raise ContractError("prediction exceeds vocabulary")
    if value[~active].ne(INVALID_TOKEN_ID).any().item():
        raise ContractError("invalid prediction rows are not -1")
    for row in range(records):
        invalid = (~active[row]).nonzero(as_tuple=False).flatten()
        if invalid.numel() and active[row, int(invalid[0]) + 1 :].any().item():
            raise ContractError("prediction padding is not a suffix")
    return value


def expected_prediction_path(output_root: Path, *, cell_id: str, method_id: str) -> Path:
    if cell_id not in CELL_ORDER or method_id not in METHOD_ORDER:
        raise ContractError("unknown scientific cell or method")
    style, condition = cell_id.split("__", 1)
    return Path(output_root) / style / condition / f"{method_id}.safetensors"


def expected_timing_path(output_root: Path, *, cell_id: str, method_id: str) -> Path:
    return expected_prediction_path(output_root, cell_id=cell_id, method_id=method_id).with_suffix(".run.json")


def write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise ContractError(f"refusing to overwrite artifact: {path}") from exc


def validate_registration(registration: Mapping[str, Any], *, verify_assets: bool = False) -> dict[str, Any]:
    if registration.get("schema") != REGISTRATION_SCHEMA or registration.get("task_id") != TASK_ID:
        raise ContractError("registration schema or task identity changed")
    if registration.get("truth_opened") is True or registration.get("source_text_or_target_labels") is True:
        raise ContractError("registration records forbidden truth/source access")
    methods = registration.get("methods")
    if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)):
        raise ContractError("registration methods are absent")
    rows = {str(row.get("id")): row for row in methods if isinstance(row, Mapping)}
    if set(rows) != set(METHOD_ORDER):
        raise ContractError("registration scientific method set changed")
    if tuple(registration.get("method_order", ())) != METHOD_ORDER:
        raise ContractError("registration method order changed")
    if NO_A2_METHOD_ID in rows or registration.get("a2") is not None:
        raise ContractError("A2 is not part of the TRR-0008 registration")
    if registration.get("cell_order") != list(CELL_ORDER):
        raise ContractError("registration cell order changed")
    geometry = registration.get("geometry")
    if not isinstance(geometry, Mapping) or any(geometry.get(k) != v for k, v in STATIC_GEOMETRY.items()):
        raise ContractError("registration geometry changed")
    counts = registration.get("records_by_domain")
    if not isinstance(counts, Mapping) or set(counts) != set(DOMAIN_ORDER):
        raise ContractError("registration records_by_domain is absent")
    for domain in DOMAIN_ORDER:
        try:
            if int(counts[domain]) <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ContractError("registration records_by_domain is malformed") from exc
    expected_records = {cell: int(counts[cell.split("__", 1)[0]]) for cell in CELL_ORDER}
    for method_id, row in rows.items():
        if row.get("cells") != list(CELL_ORDER):
            raise ContractError(f"registration coverage changed: {method_id}")
        if row.get("records_per_cell") != expected_records:
            raise ContractError(f"registration record counts changed: {method_id}")
        if not isinstance(row.get("state"), Mapping) or not isinstance(row.get("loader"), Mapping):
            raise ContractError(f"registration loader/state missing: {method_id}")
        if method_id == REFERENCE_METHOD_ID:
            expected_loader = REFERENCE_LOADER
        else:
            expected_loader = dict(POSITIONWISE_LOADER) | {
                "kwargs": dict(POSITIONWISE_LOADER["kwargs"]) | {"method_id": METHOD_MODEL_IDS[method_id]}
            }
        if row["loader"] != expected_loader:
            raise ContractError(f"registration loader changed: {method_id}")
        state = row["state"]
        if not isinstance(state.get("path"), str) or not _SHA256.fullmatch(str(state.get("sha256", ""))):
            raise ContractError(f"registration state binding malformed: {method_id}")
        if verify_assets:
            validate_file_record(state, repository_root=Path(registration["repository_root"]), description=f"{method_id} state")
    for asset_name, expected_schema in (("timing_plan", "token-reconstruction.trr0008-timing-plan.v1"), ("timing_receipt", "token-reconstruction.trr0008-timing-receipt.v1")):
        asset = registration.get(asset_name)
        if asset is not None:
            if not isinstance(asset, Mapping) or not isinstance(asset.get("path"), str) or not _SHA256.fullmatch(str(asset.get("sha256", ""))):
                raise ContractError(f"{asset_name} binding is malformed")
    observation = registration.get("observation_manifest")
    embedding = registration.get("runtime_assets", {}).get("normalized_public_E") if isinstance(registration.get("runtime_assets"), Mapping) else None
    for label, value in (("observation manifest", observation), ("normalized public E", embedding)):
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str) or not _SHA256.fullmatch(str(value.get("sha256", ""))):
            raise ContractError(f"{label} binding is malformed")
    if registration.get("timing_control", {}).get("id") == TIMING_CONTROL_METHOD_ID:
        alias = registration["timing_control"]
        if alias.get("loader") != dict(POSITIONWISE_LOADER) | {"kwargs": dict(POSITIONWISE_LOADER["kwargs"]) | {"method_id": METHOD_MODEL_IDS[TIMING_CONTROL_METHOD_ID]}}:
            raise ContractError("timing control loader changed")
    return dict(registration)


def load_registration(path: Path, *, repository_root: Path, verify_assets: bool = False) -> dict[str, Any]:
    registration = load_json(path, description="TRR-0008 registration")
    registration.setdefault("repository_root", str(Path(repository_root).expanduser().resolve()))
    return validate_registration(registration, verify_assets=verify_assets)


def validate_prediction_artifact(
    path: Path,
    *,
    registration: Mapping[str, Any],
    cell_id: str,
    method_id: str,
    records: int | None = None,
    verify_hash: bool = True,
) -> dict[str, Any]:
    if method_id not in METHOD_ORDER or cell_id not in CELL_ORDER:
        raise ContractError("unknown prediction cell or method")
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"prediction artifact is unavailable: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            metadata = dict(handle.metadata() or {})
            if keys != {"predictions"}:
                raise ContractError("prediction contains unexpected tensors")
            predictions = handle.get_tensor("predictions")
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"prediction artifact is unreadable: {path}") from exc
    if metadata.get("schema") != PREDICTION_SCHEMA or metadata.get("task_id") != TASK_ID:
        raise ContractError("prediction artifact identity changed")
    if metadata.get("cell_id") != cell_id or metadata.get("method_id") != method_id:
        raise ContractError("prediction artifact cell or method changed")
    if metadata.get("truth_opened") != "false" or metadata.get("candidate_arrays_persisted") != "false":
        raise ContractError("prediction truth/candidate flags are open")
    if metadata.get("registration_sha256") != registration.get("registration_sha256"):
        raise ContractError("prediction registration binding changed")
    if records is None:
        records = records_for_cell(registration, cell_id)
    geometry_raw = metadata.get("geometry_json")
    try:
        geometry = json.loads(str(geometry_raw))
    except json.JSONDecodeError as exc:
        raise ContractError("prediction geometry metadata is invalid") from exc
    expected_geometry = {"records": records, **STATIC_GEOMETRY}
    if geometry != expected_geometry:
        raise ContractError("prediction geometry changed")
    checked = validate_prediction_tensor(predictions, records=records)
    artifact = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    return {
        "artifact": artifact,
        "prediction_sha256": tensor_digest(checked),
        "records": records,
        "cell_id": cell_id,
        "method_id": method_id,
    }
