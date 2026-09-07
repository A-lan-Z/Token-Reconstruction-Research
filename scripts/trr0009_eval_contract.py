"""Task-local, parameterized public evaluation contract for TRR-0009.

The contract intentionally keeps panel counts and state paths in the frozen
registration.  It contains no source/truth loader and does not choose methods.
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

TASK_ID = "TRR-0009"
REGISTRATION_SCHEMA = "token-reconstruction.trr0009-evaluation-registration.v1"
OBSERVATION_SCHEMA = "token-reconstruction.trr0009-public-observation-manifest.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0009-prediction.v1"
TIMING_SCHEMA = "token-reconstruction.trr0009-prediction-timing.v1"
RUN_SCHEMA = "token-reconstruction.trr0009-prediction-run.v1"
TIMING_PLAN_SCHEMA = "token-reconstruction.trr0009-timing-plan.v1"
FREEZE_SCHEMA = "token-reconstruction.trr0009-public-freeze.v1"

CELL_ORDER = (
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
)
DOMAIN_ORDER = ("pile", "finance")
TARGET_ORDER = ("public_base", "public_lora_2601")

UNCHANGED_METHOD_ID = "unchanged_anchor"
FIXED_METHOD_ID = "continued_fixed_readout"
ADAPTABLE_METHOD_ID = "continued_adaptable_readout"
PUBLISHED_REFERENCE_METHOD_ID = "published_reference"
METHOD_ORDER = (
    UNCHANGED_METHOD_ID,
    FIXED_METHOD_ID,
    ADAPTABLE_METHOD_ID,
    PUBLISHED_REFERENCE_METHOD_ID,
)
METHOD_ROLES = {
    UNCHANGED_METHOD_ID: "unchanged_anchor",
    FIXED_METHOD_ID: "continued_fixed_readout",
    ADAPTABLE_METHOD_ID: "continued_adaptable_readout",
    PUBLISHED_REFERENCE_METHOD_ID: "published_reference",
}
LOADER_INTERFACE = "trr0009.current_h.full_vocabulary.v1"
PRIMARY_METHOD_ID = ADAPTABLE_METHOD_ID
PRIMARY_CONTROL_METHOD_ID = FIXED_METHOD_ID

HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
STORED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    """Raised when a frozen TRR-0009 binding is incomplete or changed."""


def public_frequency_metadata_is_truth_free(metadata: Mapping[str, Any]) -> bool:
    """Accept public fitting buffers while rejecting private/fresh evaluation data.

    A fitting-frequency export may have loaded public fitting tensors or support
    buffers.  Those flags do not imply that evaluation truth was opened.  A
    generic ``tensor_data_loaded`` flag is accepted only when it is explicitly
    paired with a public-fitting flag; otherwise it remains fail-closed.
    """
    if not isinstance(metadata, Mapping):
        return False

    def true_flag(value: Any) -> bool:
        return value is True or (isinstance(value, str) and value.lower() == "true")

    forbidden = (
        "truth_accessed",
        "fresh_data_accessed",
        "source_text_loaded",
        "target_labels_loaded",
        "private_truth_accessed",
        "private_or_truth_payload_read",
    )
    if any(true_flag(metadata.get(key)) for key in forbidden):
        return False
    public_fitting = any(
        true_flag(metadata.get(key))
        for key in ("public_fitting_tensors_loaded", "support_state_buffers_loaded")
    )
    if true_flag(metadata.get("tensor_data_loaded")) and not public_fitting:
        return False
    return True


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
    digest.update(json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}, sort_keys=True, separators=(",", ":")).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("value cannot be canonically encoded") from exc
    return hashlib.sha256(raw.encode()).hexdigest()


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
    if size < 0:
        raise ContractError(f"{description} byte count is negative")
    digest = value.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ContractError(f"{description} hash is malformed")
    result = {"path": str(path), "bytes": size, "sha256": digest}
    if verify and (path.stat().st_size != size or sha256_file(path) != digest):
        raise ContractError(f"{description} hash or size changed")
    return result


def write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
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
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _as_cells(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("cells")
    if isinstance(raw, Mapping):
        rows = {str(k): v for k, v in raw.items() if isinstance(v, Mapping)}
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        rows = {str(row.get("cell_id")): row for row in raw if isinstance(row, Mapping) and row.get("cell_id")}
    else:
        raise ContractError("cells are absent")
    if set(rows) != set(CELL_ORDER):
        raise ContractError("cell membership differs from the frozen four-cell interface")
    return rows


def records_for_cell(manifest: Mapping[str, Any], cell_id: str) -> int:
    if cell_id not in CELL_ORDER:
        raise ContractError(f"unknown cell: {cell_id}")
    cell = _as_cells(manifest)[cell_id]
    raw = cell.get("records")
    if raw is None and isinstance(cell.get("observation"), Mapping):
        raw = cell["observation"].get("records")
    if raw is None and isinstance(manifest.get("records_by_domain"), Mapping):
        raw = manifest["records_by_domain"].get(cell_id.split("__", 1)[0])
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"record count is absent for {cell_id}") from exc
    if count <= 0:
        raise ContractError(f"record count is non-positive for {cell_id}")
    return count


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
    if tuple(value.shape) != (int(records), STORED_SEQUENCE_TOKENS):
        raise ContractError(f"prediction shape differs: expected {(records, STORED_SEQUENCE_TOKENS)}")
    if value[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise ContractError("prediction BOS column changed")
    active = value.ge(0)
    if active[:, 0].logical_not().any().item() or value[active].ge(VOCABULARY_SIZE).any().item():
        raise ContractError("prediction contains invalid active IDs")
    if value[~active].ne(INVALID_TOKEN_ID).any().item():
        raise ContractError("invalid prediction positions are not -1")
    for row in range(int(records)):
        invalid = (~active[row]).nonzero(as_tuple=False).flatten()
        if invalid.numel() and active[row, int(invalid[0]) + 1 :].any().item():
            raise ContractError("prediction padding is not a suffix")
    return value


def expected_prediction_path(output_root: Path, *, cell_id: str, method_id: str) -> Path:
    if cell_id not in CELL_ORDER or method_id not in METHOD_ORDER:
        raise ContractError("unknown cell or method")
    style, condition = cell_id.split("__", 1)
    return Path(output_root) / style / condition / f"{method_id}.safetensors"


def expected_timing_path(output_root: Path, *, cell_id: str, method_id: str) -> Path:
    return expected_prediction_path(output_root, cell_id=cell_id, method_id=method_id).with_suffix(".run.json")


def validate_observation_manifest(manifest: Mapping[str, Any], *, repository_root: Path, verify_assets: bool = True) -> dict[str, Any]:
    if manifest.get("task_id") != TASK_ID or manifest.get("schema") != OBSERVATION_SCHEMA:
        raise ContractError("observation identity/schema changed")
    forbidden = ("truth_opened", "target_labels_loaded", "source_text_loaded", "source_text_written", "token_ids_written")
    if any(manifest.get(key) is True for key in forbidden):
        raise ContractError("observation manifest records forbidden truth/source access")
    cells = _as_cells(manifest)
    checked: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for cell_id in CELL_ORDER:
        cell = cells[cell_id]
        if cell.get("cell_id") != cell_id:
            raise ContractError(f"observation cell ID changed: {cell_id}")
        observation = cell.get("observation") if isinstance(cell.get("observation"), Mapping) else cell
        record = validate_file_record(observation, repository_root=repository_root, description=f"observation {cell_id}", verify=verify_assets)
        records = records_for_cell(manifest, cell_id)
        shape = observation.get("shape")
        if list(shape or []) != [records, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE]:
            raise ContractError(f"observation geometry changed: {cell_id}")
        if observation.get("activations_key") != "activations" or observation.get("attention_mask_key") != "attention_mask" or observation.get("position_ids_key") != "position_ids":
            raise ContractError(f"observation tensor keys changed: {cell_id}")
        checked.append(dict(cell) | {"records": records, "observation": record})
        counts[cell_id.split("__", 1)[0]] = records
    if isinstance(manifest.get("records_by_domain"), Mapping) and dict(manifest["records_by_domain"]) != counts:
        raise ContractError("records_by_domain changed")
    return dict(manifest) | {"cells": checked, "records_by_domain": counts}


def _validate_loader(loader: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(loader, Mapping) or not isinstance(loader.get("module"), str) or not isinstance(loader.get("function"), str):
        raise ContractError(f"{description} loader is malformed")
    if loader.get("interface") != LOADER_INTERFACE or loader.get("current_h_only") is not True or loader.get("full_vocabulary") is not True or loader.get("history_enabled") is not False or loader.get("a2_enabled") is not False:
        raise ContractError(f"{description} loader semantics are not the frozen current-H/full-vocabulary interface")
    kwargs = loader.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ContractError(f"{description} loader kwargs are malformed")
    for name in ("path_args", "tensor_args"):
        values = loader.get(name, {})
        if not isinstance(values, Mapping):
            raise ContractError(f"{description} {name} are malformed")
        for arg, binding in values.items():
            if not isinstance(arg, str) or not isinstance(binding, Mapping):
                raise ContractError(f"{description} {name} binding is malformed")
    return dict(loader)


def validate_registration(registration: Mapping[str, Any], *, repository_root: Path, verify_assets: bool = True) -> dict[str, Any]:
    if registration.get("schema") != REGISTRATION_SCHEMA or registration.get("task_id") != TASK_ID:
        raise ContractError("registration schema or task identity changed")
    if any(registration.get(key) is True for key in ("truth_opened", "source_text_or_target_labels", "candidate_arrays_persisted", "a2_enabled", "history_enabled")):
        raise ContractError("registration records forbidden truth, A2, history, or candidate state")
    methods = registration.get("methods")
    if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes, bytearray)):
        raise ContractError("registration methods are absent")
    rows = {str(row.get("id")): row for row in methods if isinstance(row, Mapping)}
    if tuple(registration.get("method_order", ())) != METHOD_ORDER or set(rows) != set(METHOD_ORDER):
        raise ContractError("registration method matrix changed")
    if registration.get("cell_order") != list(CELL_ORDER):
        raise ContractError("registration cell order changed")
    geometry = registration.get("geometry")
    if not isinstance(geometry, Mapping) or any(geometry.get(k) != v for k, v in STATIC_GEOMETRY.items()):
        raise ContractError("registration geometry changed")
    counts = registration.get("records_by_domain")
    if not isinstance(counts, Mapping) or set(counts) != set(DOMAIN_ORDER):
        raise ContractError("registration records_by_domain is absent")
    expected_records = {cell: int(counts[cell.split("__", 1)[0]]) for cell in CELL_ORDER}
    if any(value <= 0 for value in expected_records.values()):
        raise ContractError("registration record counts must be positive")
    for method_id in METHOD_ORDER:
        row = rows[method_id]
        if row.get("role") != METHOD_ROLES[method_id] or row.get("cells") != list(CELL_ORDER) or row.get("records_per_cell") != expected_records:
            raise ContractError(f"registration role or coverage changed: {method_id}")
        state = row.get("state")
        if not isinstance(state, Mapping):
            raise ContractError(f"state binding missing: {method_id}")
        validate_file_record(state, repository_root=repository_root, description=f"{method_id} state", verify=verify_assets)
        _validate_loader(row.get("loader"), description=method_id)
        for name, values in (("path_args", row["loader"].get("path_args", {})), ("tensor_args", row["loader"].get("tensor_args", {}))):
            for arg, binding in values.items():
                validate_file_record(binding, repository_root=repository_root, description=f"{method_id} loader {name}.{arg}", verify=verify_assets)
    for name in ("observation_manifest", "source_selection", "capture_receipt", "frequency_reference", "runtime_embedding", "timing_plan"):
        binding = registration.get(name)
        if not isinstance(binding, Mapping):
            raise ContractError(f"registration {name} binding is absent")
        validate_file_record(binding, repository_root=repository_root, description=name, verify=verify_assets)
    observation = load_json(Path(registration["observation_manifest"]["path"]), description="observation manifest")
    validate_observation_manifest(observation, repository_root=repository_root, verify_assets=verify_assets)
    code_bindings = registration.get("code_bindings")
    if not isinstance(code_bindings, Sequence) or isinstance(code_bindings, (str, bytes, bytearray)) or not code_bindings:
        raise ContractError("registration code bindings are absent")
    for index, binding in enumerate(code_bindings):
        validate_file_record(binding, repository_root=repository_root, description=f"code binding {index}", verify=verify_assets)
    commit = registration.get("code_commit")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ContractError("registration code commit is malformed")
    guard = registration.get("resource_guard")
    if not isinstance(guard, Mapping) or any(key not in guard for key in RESOURCE_GUARD):
        raise ContractError("resource guard is incomplete")
    return dict(registration) | {"methods": [rows[method_id] for method_id in METHOD_ORDER]}


def load_prediction_file(path: Path, *, records: int, expected_metadata: Mapping[str, str] | None = None) -> tuple[torch.Tensor, dict[str, str]]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"prediction file unavailable: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            metadata = dict(handle.metadata() or {})
            if keys != {"predictions"}:
                raise ContractError("prediction tensor keys changed")
            value = handle.get_tensor("predictions")
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"prediction file could not be opened: {path}") from exc
    if expected_metadata:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != str(expected):
                raise ContractError(f"prediction metadata changed: {key}")
    return validate_prediction_tensor(value, records=records), metadata
