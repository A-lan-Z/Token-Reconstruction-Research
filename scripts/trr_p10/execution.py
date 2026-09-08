"""Fail-closed TRR-P10 frozen-package and paired-inventory adapters.

The adapter has two deliberately separate phases:

* :func:`validate_frozen_package` reads only metadata and frozen prediction
  tensors.  It rejects truth/source flags, unbound candidate materialization,
  changed hashes, incomplete method/cell matrices, and malformed prediction
  geometry.
* :func:`load_truth_after_freeze` may be called only with a previously validated
  package and a truth descriptor whose freeze/registration bindings match that
  package.  It then loads the truth sidecar for scoring.

This module does not select records, read source text, run a model, produce
candidate arrays, or infer A1 proposal failures from a final prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from safetensors import safe_open


TASK_ID = "TRR-P10"
METHOD_ORDER = ("expanded_fixed", "current_fixed", "frozen_a1_a2_k256")
CELL_ORDER = (
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
)
DOMAIN_ORDER = ("pile", "finance")
TARGET_ORDER = ("public_base", "public_lora_2601")
BOS_TOKEN_ID = 128000
VOCABULARY_SIZE = 128256
STORED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127

PREDICTION_SCHEMAS = {
    "token-reconstruction.trr0010-prediction.v1",
    "token-reconstruction.trr0009-prediction.v1",
    "token-reconstruction.trr-p10-prediction.v1",
}
PRETRUTH_STATUSES = {
    "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH",
    "PUBLIC_FREEZE_VALIDATED_BEFORE_TRUTH",
    "FROZEN_EVALUATION_REGISTRATION_BEFORE_TRUTH",
    "P10_PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH",
    "P10_IMMUTABLE_INFERENCE_PACKAGE_READY",
}
TRUTH_PREPARED_STATUSES = {
    "TRR0010_TRUTH_PREPARED_AFTER_PUBLIC_FREEZE",
    "TRR0009_TRUTH_PREPARED_AFTER_PUBLIC_GATE",
    "TRR-P10_TRUTH_PREPARED_AFTER_PUBLIC_FREEZE",
    "P10_TRUTH_BOUND_AFTER_PUBLIC_FREEZE",
}
FORBIDDEN_TRUE_FLAGS = (
    "truth_opened",
    "target_labels_loaded",
    "source_text_loaded",
    "source_text_written",
    "token_ids_written",
)
TRACE_BINDING_KEYS = ("candidate_trace", "proposal_trace", "rank_trace")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExecutionError(ValueError):
    """Raised when the P10 execution or evaluation boundary is invalid."""


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _is_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.lower() == "false")


def _assert_truth_free(
    payload: Mapping[str, Any],
    *,
    description: str,
    allow_trace: bool = False,
) -> None:
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(payload.get(key)):
            raise ExecutionError(f"{description} records forbidden access: {key}")
    if _is_true(payload.get("candidate_arrays_persisted")) and not allow_trace:
        raise ExecutionError(f"{description} records candidate materialization without a bound truth-free trace")


def sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ExecutionError(f"asset is not a regular file: {path}")
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


def indices_digest(indices: list[int] | tuple[int, ...]) -> str:
    """Hash an ordered record-index list using the repository JSON convention."""
    normalized = [int(index) for index in indices]
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_path(value: Any, *, base: Path, description: str) -> Path:
    if not isinstance(value, (str, Path)) or not value:
        raise ExecutionError(f"{description} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ExecutionError(f"{description} is unavailable or symlinked: {path}")
    return path


def file_record(
    path: Path,
    *,
    base: Path,
    description: str,
    declared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_path(path if isinstance(path, Path) else path, base=base, description=description)
    actual = {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }
    if declared is not None:
        for key in ("path", "bytes", "sha256"):
            if key not in declared:
                raise ExecutionError(f"{description} binding lacks {key}")
        declared_path = _resolve_path(declared["path"], base=base, description=description)
        if declared_path != resolved:
            raise ExecutionError(f"{description} path binding changed")
        try:
            declared_bytes = int(declared["bytes"])
        except (TypeError, ValueError) as exc:
            raise ExecutionError(f"{description} byte binding is malformed") from exc
        if declared_bytes != actual["bytes"]:
            raise ExecutionError(f"{description} byte binding changed")
        if declared.get("sha256") != actual["sha256"] or _SHA256.fullmatch(str(declared.get("sha256"))) is None:
            raise ExecutionError(f"{description} hash binding changed or is malformed")
    return actual


def _load_json(path: Path, *, base: Path, description: str, declared: Mapping[str, Any] | None = None) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    record = file_record(path, base=base, description=description, declared=declared)
    resolved = Path(record["path"])
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"{description} is invalid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise ExecutionError(f"{description} must be a JSON object")
    return resolved, value, record


def _ref_record(value: Any, *, base: Path, description: str) -> tuple[Path, dict[str, Any]]:
    if isinstance(value, Mapping):
        if "path" not in value:
            raise ExecutionError(f"{description} file binding lacks path")
        path = _resolve_path(value["path"], base=base, description=description)
        return path, file_record(path, base=base, description=description, declared=value)
    path = _resolve_path(value, base=base, description=description)
    return path, file_record(path, base=base, description=description)


def _nested_json_ref(value: Any, *, base: Path, description: str) -> tuple[Path | None, dict[str, Any], dict[str, Any] | None]:
    """Resolve a JSON reference or accept an embedded payload.

    P10's package may embed a small receipt while the inherited TRR-0010
    freeze stores a path binding.  Both forms are accepted, but path bindings
    always require a matching byte/hash record.
    """
    if isinstance(value, Mapping) and "path" in value:
        path, payload, record = _load_json(Path(str(value["path"])), base=base, description=description, declared=value)
        return path, payload, record
    if isinstance(value, Mapping):
        return None, dict(value), None
    if isinstance(value, (str, Path)):
        path, payload, record = _load_json(Path(value), base=base, description=description)
        return path, payload, record
    raise ExecutionError(f"{description} is absent or malformed")


def _same_record(actual: Mapping[str, Any], expected: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise ExecutionError(f"{description} {key} binding changed")


def _validate_prediction_trace(
    binding: Mapping[str, Any],
    *,
    base: Path,
    description: str,
) -> dict[str, dict[str, Any]]:
    """Validate optional opaque truth-free traces on one prediction binding."""
    result: dict[str, dict[str, Any]] = {}
    for label in TRACE_BINDING_KEYS:
        value = binding.get(label)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ExecutionError(f"{description} {label} trace binding is malformed")
        trace_path, trace_record = _ref_record(value, base=base, description=f"{description} {label} trace")
        if _is_true(value.get("truth_opened")) or _is_true(value.get("target_labels_loaded")):
            raise ExecutionError(f"{description} {label} trace crosses the truth boundary")
        try:
            if trace_path.suffix.lower() == ".json":
                trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
                if not isinstance(trace_payload, Mapping):
                    raise ExecutionError(f"{description} {label} trace JSON is not an object")
                _assert_truth_free(trace_payload, description=f"{description} {label} trace", allow_trace=True)
            elif trace_path.suffix.lower() == ".safetensors":
                with safe_open(str(trace_path), framework="pt", device="cpu") as handle:
                    _assert_truth_free(dict(handle.metadata() or {}), description=f"{description} {label} trace", allow_trace=True)
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"{description} {label} trace is unreadable") from exc
        result[label] = trace_record | {"label": label}
    return result


def _mapping_from_prediction_bindings(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = payload.get("predictions", payload.get("prediction_bindings"))
    if not isinstance(raw, Mapping):
        raise ExecutionError("frozen prediction bindings are absent")
    result: dict[str, Mapping[str, Any]] = {}
    for key, value in raw.items():
        key_text = str(key)
        if "::" in key_text and isinstance(value, Mapping):
            result[key_text] = value
            continue
        if isinstance(value, Mapping):
            # Also accept {method: {cell: binding}} package form.
            for cell, binding in value.items():
                if isinstance(binding, Mapping):
                    result[f"{key_text}::{cell}"] = binding
    return result


def _records_by_cell(payload: Mapping[str, Any]) -> dict[str, int]:
    raw = payload.get("records_by_cell")
    if isinstance(raw, Mapping):
        result = {str(k): int(v) for k, v in raw.items()}
    else:
        raw_domain = payload.get("records_by_domain")
        if not isinstance(raw_domain, Mapping):
            raise ExecutionError("record counts are absent from frozen package")
        result = {
            cell: int(raw_domain[cell.split("__", 1)[0]])
            for cell in CELL_ORDER
            if cell.split("__", 1)[0] in raw_domain
        }
    if set(result) != set(CELL_ORDER) or any(value <= 0 for value in result.values()):
        raise ExecutionError("frozen record counts do not cover the four required cells")
    return result


def _binding_count(value: Any, *, description: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionError(f"{description} record count is malformed") from exc
    if count <= 0:
        raise ExecutionError(f"{description} record count is not positive")
    return count


def _lookup_matrix_value(payload: Mapping[str, Any], names: tuple[str, ...], *, method: str, cell: str) -> Any:
    key = f"{method}::{cell}"
    for name in names:
        matrix = payload.get(name)
        if not isinstance(matrix, Mapping):
            continue
        if key in matrix:
            return matrix[key]
        method_value = matrix.get(method)
        if isinstance(method_value, Mapping) and cell in method_value:
            return method_value[cell]
    return None


def _method_record_count(
    payload: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    method: str,
    cell: str,
    fallback: int,
) -> int:
    """Resolve a method/cell row count without assuming shared denominators."""
    for source in (binding, payload):
        if not isinstance(source, Mapping):
            continue
        for field in ("records", "record_count"):
            if field in source and not isinstance(source[field], Mapping):
                return _binding_count(source[field], description=f"{method}/{cell}")
    mapped = _lookup_matrix_value(
        payload,
        ("records_by_method_cell", "record_counts_by_method_cell", "method_record_counts"),
        method=method,
        cell=cell,
    )
    if mapped is not None:
        return _binding_count(mapped, description=f"{method}/{cell}")
    return int(fallback)


def _subset_candidate(binding: Mapping[str, Any], payload: Mapping[str, Any], *, method: str, cell: str) -> Any:
    names = ("source_subset", "subset", "subset_binding", "record_subset")
    for name in names:
        value = binding.get(name)
        if value is not None:
            return value
    return _lookup_matrix_value(
        payload,
        ("subset_bindings", "source_subsets", "record_subsets", "subsets"),
        method=method,
        cell=cell,
    )


def _parse_subset_indices(
    binding: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    base: Path,
    method: str,
    cell: str,
    records: int,
    full_records: int,
) -> tuple[int, ...] | None:
    """Validate an optional ordered subset and bind it to its parent panel.

    A smaller A1+A2 prediction may only be paired to a full-panel prediction
    through explicit row indices.  The adapter never aligns two unequal arrays
    by position alone.
    """
    candidate = _subset_candidate(binding, payload, method=method, cell=cell)
    if candidate is None:
        if method == "frozen_a1_a2_k256" and records < full_records:
            raise ExecutionError(f"{method}/{cell} is smaller than the primary panel without a source subset")
        if records != full_records and method != "frozen_a1_a2_k256":
            raise ExecutionError(f"{method}/{cell} has an unsupported reduced denominator")
        return None
    if not isinstance(candidate, Mapping):
        raise ExecutionError(f"{method}/{cell} source subset binding is malformed")
    parent_cell = candidate.get("parent_cell", candidate.get("cell_id", cell))
    if str(parent_cell) != cell:
        raise ExecutionError(f"{method}/{cell} source subset parent cell changed")
    raw_indices = candidate.get("indices")
    if raw_indices is None:
        raw_indices = candidate.get("record_indices", candidate.get("global_indices"))
    if not isinstance(raw_indices, (list, tuple)):
        # An opaque hash cannot align the post-freeze truth sidecar.  Require
        # the ordered indices in the immutable package rather than guessing.
        raise ExecutionError(f"{method}/{cell} source subset indices are absent")
    indices: list[int] = []
    for raw in raw_indices:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ExecutionError(f"{method}/{cell} source subset index is malformed")
        index = int(raw)
        if index < 0 or index >= full_records:
            raise ExecutionError(f"{method}/{cell} source subset index is out of range")
        indices.append(index)
    if len(indices) != records or len(set(indices)) != len(indices):
        raise ExecutionError(f"{method}/{cell} source subset count or uniqueness changed")
    declared_count = candidate.get("records", candidate.get("record_count", len(indices)))
    if _binding_count(declared_count, description=f"{method}/{cell} source subset") != len(indices):
        raise ExecutionError(f"{method}/{cell} source subset record count changed")
    declared_digest = candidate.get("indices_sha256")
    if declared_digest is None:
        declared_digest = candidate.get("record_indices_sha256", candidate.get("global_indices_sha256"))
    if not isinstance(declared_digest, str) or _SHA256.fullmatch(declared_digest) is None:
        raise ExecutionError(f"{method}/{cell} source subset index digest is absent or malformed")
    actual_digest = indices_digest(indices)
    if declared_digest != actual_digest:
        raise ExecutionError(f"{method}/{cell} source subset index digest changed")
    return tuple(indices)


def _metadata_geometry(
    metadata: Mapping[str, Any],
    *,
    method_id: str,
    cell_id: str,
    records: int,
    allow_candidate_traces: bool = False,
) -> None:
    schema = metadata.get("schema")
    if schema not in PREDICTION_SCHEMAS:
        raise ExecutionError(f"prediction schema is not recognized: {method_id}/{cell_id}")
    if str(metadata.get("method_id")) != method_id or str(metadata.get("cell_id")) != cell_id:
        raise ExecutionError(f"prediction identity changed: {method_id}/{cell_id}")
    if _is_true(metadata.get("truth_opened")):
        raise ExecutionError(f"prediction records truth access: {method_id}/{cell_id}")
    if _is_true(metadata.get("candidate_arrays_persisted")) and not allow_candidate_traces:
        raise ExecutionError(f"prediction records candidate materialization without a bound truth-free trace: {method_id}/{cell_id}")
    try:
        declared_records = int(metadata.get("records"))
    except (TypeError, ValueError) as exc:
        raise ExecutionError(f"prediction record count is malformed: {method_id}/{cell_id}") from exc
    if declared_records != records:
        raise ExecutionError(f"prediction record count changed: {method_id}/{cell_id}")
    geometry = metadata.get("geometry_json")
    if geometry is not None:
        try:
            geometry_payload = json.loads(str(geometry))
        except json.JSONDecodeError as exc:
            raise ExecutionError(f"prediction geometry metadata is malformed: {method_id}/{cell_id}") from exc
        expected = {
            "records": records,
            "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
            "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
            "vocabulary_size": VOCABULARY_SIZE,
            "bos_token_id": BOS_TOKEN_ID,
        }
        for key, value in expected.items():
            if key in geometry_payload and int(geometry_payload[key]) != value:
                raise ExecutionError(f"prediction geometry changed: {method_id}/{cell_id}")


def _load_prediction(
    binding: Mapping[str, Any],
    *,
    base: Path,
    method_id: str,
    cell_id: str,
    records: int,
    allow_candidate_traces: bool = False,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, str]]:
    path, record = _ref_record(binding, base=base, description=f"prediction {method_id}/{cell_id}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"predictions"}:
                raise ExecutionError(f"prediction tensor keys changed: {method_id}/{cell_id}")
            values = handle.get_tensor("predictions").detach().cpu().contiguous()
            metadata = dict(handle.metadata() or {})
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError(f"prediction artifact is unreadable: {method_id}/{cell_id}") from exc
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ExecutionError(f"prediction dtype is not integer: {method_id}/{cell_id}")
    if tuple(values.shape) != (records, STORED_SEQUENCE_TOKENS):
        raise ExecutionError(f"prediction geometry changed: {method_id}/{cell_id}")
    if values[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise ExecutionError(f"prediction BOS column changed: {method_id}/{cell_id}")
    if values.lt(0).any().item() or values.ge(VOCABULARY_SIZE).any().item():
        raise ExecutionError(f"prediction contains invalid token IDs: {method_id}/{cell_id}")
    _metadata_geometry(
        metadata,
        method_id=method_id,
        cell_id=cell_id,
        records=records,
        allow_candidate_traces=allow_candidate_traces,
    )
    digest = tensor_digest(values)
    declared_digest = binding.get("prediction_sha256")
    if declared_digest is None:
        raise ExecutionError(f"prediction tensor digest is absent: {method_id}/{cell_id}")
    if declared_digest != digest:
        raise ExecutionError(f"prediction tensor digest changed: {method_id}/{cell_id}")
    return record, values.to(dtype=torch.long), {str(k): str(v) for k, v in metadata.items()}


@dataclass(frozen=True)
class FrozenPackage:
    """Validated frozen predictions and their immutable provenance bindings."""

    freeze_path: Path
    freeze_record: dict[str, Any]
    freeze_payload: dict[str, Any]
    registration_path: Path
    registration_record: dict[str, Any]
    registration_payload: dict[str, Any]
    run_manifest_record: dict[str, Any] | None
    trace_records: dict[str, dict[str, Any]]
    cells: tuple[str, ...]
    methods: tuple[str, ...]
    # Full-panel truth rows, used as the parent denominator for each cell.
    records_by_cell: dict[str, int]
    # Actual row count of every method/cell prediction; denominators may differ.
    records_by_method_cell: dict[str, int]
    # Ordered parent-row indices for reduced method/cell predictions.
    subset_indices_by_method_cell: dict[str, tuple[int, ...] | None]
    predictions: dict[str, torch.Tensor]
    prediction_records: dict[str, dict[str, Any]]
    prediction_metadata: dict[str, dict[str, str]]


def validate_frozen_package(
    freeze_path: Path,
    *,
    repository_root: Path | None = None,
    registration_path: Path | None = None,
) -> FrozenPackage:
    """Validate a complete truth-free freeze/registration/prediction package.

    ``freeze_path`` may point to an inherited public-freeze receipt or a P10
    wrapper containing ``public_freeze``/``freeze``.  The normalized result
    always binds the registration and all three requested method matrices.
    """
    freeze_input = Path(freeze_path).expanduser().resolve()
    base = Path(repository_root or freeze_input.parent).expanduser().resolve()
    _, outer, outer_record = _load_json(freeze_input, base=base, description="frozen package")
    _assert_truth_free(outer, description="frozen package")

    embedded = outer
    embedded_path = freeze_input
    embedded_record = outer_record
    for key in ("public_freeze", "freeze", "freeze_receipt"):
        value = outer.get(key)
        if value is not None:
            embedded_path, embedded, embedded_record = _nested_json_ref(value, base=base, description=f"{key} receipt")
            break
    _assert_truth_free(embedded, description="public freeze")
    status = embedded.get("status")
    if status not in PRETRUTH_STATUSES:
        raise ExecutionError(f"public freeze is not a recognized pretruth receipt: {status!r}")
    if not _is_false(embedded.get("truth_opened")):
        raise ExecutionError("public freeze truth boundary is not explicitly closed")

    records_by_cell = _records_by_cell(embedded)
    raw_methods = embedded.get("method_order") or embedded.get("methods")
    if isinstance(raw_methods, Mapping):
        raw_methods = list(raw_methods)
    if not isinstance(raw_methods, (list, tuple)):
        raise ExecutionError("public freeze method order is absent")
    methods = tuple(str(item.get("id")) if isinstance(item, Mapping) else str(item) for item in raw_methods)
    if any(method not in methods for method in METHOD_ORDER):
        raise ExecutionError("public freeze does not include expanded-fixed/current-fixed/A1+A2")

    cells_raw = embedded.get("cell_order") or embedded.get("cells")
    if isinstance(cells_raw, Mapping):
        cells_raw = list(cells_raw)
    if not isinstance(cells_raw, (list, tuple)):
        raise ExecutionError("public freeze cell order is absent")
    cells = tuple(str(item.get("cell_id")) if isinstance(item, Mapping) else str(item) for item in cells_raw)
    if any(cell not in cells for cell in CELL_ORDER):
        raise ExecutionError("public freeze does not include the required paired cells")

    actual_registration_value = registration_path
    if actual_registration_value is None:
        actual_registration_value = embedded.get("registration") or outer.get("registration")
    if actual_registration_value is None:
        raise ExecutionError("frozen package registration binding is absent")
    registration_file, registration, registration_record = _nested_json_ref(actual_registration_value, base=base, description="frozen registration")
    if registration_file is None:
        raise ExecutionError("frozen registration must be a hash-bound file, not an embedded object")
    declared_registration = embedded.get("registration") or outer.get("registration")
    if registration_path is not None and declared_registration is not None:
        declared_file, _declared_payload, declared_record = _nested_json_ref(
            declared_registration,
            base=base,
            description="declared frozen registration",
        )
        if declared_file is None or declared_record is None:
            raise ExecutionError("frozen package registration must be a hash-bound file")
        _same_record(
            registration_record,
            declared_record,
            description="frozen package/registration",
        )
    _assert_truth_free(registration, description="frozen registration")
    if not _is_false(registration.get("truth_opened")):
        raise ExecutionError("frozen registration truth boundary is not explicitly closed")
    registration_methods = registration.get("method_order") or registration.get("methods")
    if isinstance(registration_methods, Mapping):
        registration_methods = list(registration_methods)
    if not isinstance(registration_methods, (list, tuple)):
        raise ExecutionError("frozen registration method order is absent")
    registration_method_ids = tuple(str(item.get("id")) if isinstance(item, Mapping) else str(item) for item in registration_methods)
    if any(method not in registration_method_ids for method in METHOD_ORDER):
        raise ExecutionError("frozen registration omits a required method")
    registration_cells = registration.get("cell_order") or registration.get("cells")
    if isinstance(registration_cells, Mapping):
        registration_cells = list(registration_cells)
    if not isinstance(registration_cells, (list, tuple)) or any(cell not in [str(item.get("cell_id")) if isinstance(item, Mapping) else str(item) for item in registration_cells] for cell in CELL_ORDER):
        raise ExecutionError("frozen registration omits a required cell")
    if registration.get("records_by_cell") is not None or registration.get("records_by_domain") is not None:
        if _records_by_cell(registration) != records_by_cell:
            raise ExecutionError("frozen registration record counts differ from public freeze")

    run_manifest_record: dict[str, Any] | None = None
    run_value = embedded.get("run_manifest") or outer.get("run_manifest")
    if run_value is not None:
        run_path, run_payload, run_manifest_record = _nested_json_ref(run_value, base=base, description="prediction run manifest")
        _ = run_path
        _assert_truth_free(run_payload, description="prediction run manifest")
        if not _is_false(run_payload.get("truth_opened")):
            raise ExecutionError("prediction run manifest truth boundary is not explicitly closed")

    prediction_bindings = _mapping_from_prediction_bindings(embedded)
    if not prediction_bindings:
        prediction_bindings = _mapping_from_prediction_bindings(outer)
    trace_records: dict[str, dict[str, Any]] = {}
    prediction_trace_records: dict[str, dict[str, dict[str, Any]]] = {}
    for prediction_key, prediction_binding in prediction_bindings.items():
        local = _validate_prediction_trace(
            prediction_binding,
            base=base,
            description=f"prediction {prediction_key}",
        )
        if local:
            prediction_trace_records[prediction_key] = local
            for trace_key, trace_record in local.items():
                trace_records[f"{prediction_key}::{trace_key}"] = trace_record
    required_keys = {f"{method}::{cell}" for method in METHOD_ORDER for cell in CELL_ORDER}
    if not required_keys.issubset(prediction_bindings):
        missing = sorted(required_keys - set(prediction_bindings))
        raise ExecutionError(f"frozen prediction matrix is incomplete: {missing}")

    predictions: dict[str, torch.Tensor] = {}
    prediction_records: dict[str, dict[str, Any]] = {}
    prediction_metadata: dict[str, dict[str, str]] = {}
    records_by_method_cell: dict[str, int] = {}
    subset_indices_by_method_cell: dict[str, tuple[int, ...] | None] = {}
    for method in METHOD_ORDER:
        for cell in CELL_ORDER:
            key = f"{method}::{cell}"
            binding = prediction_bindings[key]
            records = _method_record_count(
                embedded,
                binding,
                method=method,
                cell=cell,
                fallback=records_by_cell[cell],
            )
            subset_indices = _parse_subset_indices(
                binding,
                embedded,
                base=base,
                method=method,
                cell=cell,
                records=records,
                full_records=records_by_cell[cell],
            )
            record, values, metadata = _load_prediction(
                binding,
                base=base,
                method_id=method,
                cell_id=cell,
                records=records,
                allow_candidate_traces=bool(prediction_trace_records.get(key)),
            )
            predictions[key] = values
            records_by_method_cell[key] = records
            subset_indices_by_method_cell[key] = subset_indices
            prediction_records[key] = record | {"prediction_sha256": tensor_digest(values)}
            prediction_metadata[key] = metadata
    if embedded_path is None:
        embedded_path = freeze_input
    return FrozenPackage(
        freeze_path=embedded_path,
        freeze_record=embedded_record,
        freeze_payload=dict(embedded),
        registration_path=registration_file,
        registration_record=registration_record,
        registration_payload=dict(registration),
        run_manifest_record=run_manifest_record,
        trace_records=trace_records,
        cells=tuple(CELL_ORDER),
        methods=tuple(METHOD_ORDER),
        records_by_cell=records_by_cell,
        records_by_method_cell=records_by_method_cell,
        subset_indices_by_method_cell=subset_indices_by_method_cell,
        predictions=predictions,
        prediction_records=prediction_records,
        prediction_metadata=prediction_metadata,
    )


def _truth_keys(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = payload.get("truth_tensor_keys")
    if isinstance(raw, (list, tuple)):
        expected = [f"{cell}__token_ids" for cell in CELL_ORDER]
        if list(raw) != expected:
            raise ExecutionError("truth tensor key order changed")
        return {cell: f"{cell}__token_ids" for cell in CELL_ORDER}
    return {cell: cell for cell in CELL_ORDER}


def _descriptor_binding(payload: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if payload.get(name) is not None:
            return payload[name]
    return None


def load_truth_after_freeze(
    frozen: FrozenPackage,
    truth_descriptor_path: Path,
    *,
    truth_path: Path | None = None,
    repository_root: Path | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load truth only after the frozen package and its descriptor agree.

    The descriptor can itself remain marked ``truth_opened=false`` because it
    records a sidecar prepared after the pretruth freeze.  The sidecar is still
    bound by bytes/hash and cell geometry before any truth tensor is accepted.
    """
    descriptor_input = Path(truth_descriptor_path).expanduser().resolve()
    base = Path(repository_root or descriptor_input.parent).expanduser().resolve()
    _, descriptor, descriptor_record = _load_json(descriptor_input, base=base, description="truth descriptor")
    if descriptor.get("status") not in TRUTH_PREPARED_STATUSES and descriptor.get("prepared_after_public_freeze") is not True and descriptor.get("prepared_after_public_gate") is not True:
        raise ExecutionError("truth descriptor does not state that it was prepared after public freeze")
    if descriptor.get("prepared_after_public_freeze") is False or descriptor.get("prepared_after_public_gate") is False:
        raise ExecutionError("truth descriptor records preparation before public freeze")

    freeze_value = _descriptor_binding(descriptor, ("public_freeze", "freeze", "receipt", "freeze_receipt"))
    if freeze_value is not None:
        if not isinstance(freeze_value, Mapping):
            raise ExecutionError("truth descriptor freeze binding is malformed")
        actual = file_record(frozen.freeze_path, base=base, description="frozen public receipt")
        _same_record(actual, freeze_value, description="truth descriptor/freeze")
    registration_value = descriptor.get("registration")
    if registration_value is not None:
        if not isinstance(registration_value, Mapping):
            raise ExecutionError("truth descriptor registration binding is malformed")
        actual = file_record(frozen.registration_path, base=base, description="frozen registration")
        _same_record(actual, registration_value, description="truth descriptor/registration")

    declared_truth = _descriptor_binding(descriptor, ("truth_payload", "sidecar", "truth_sidecar"))
    if declared_truth is None and truth_path is None:
        raise ExecutionError("truth sidecar binding is absent")
    if declared_truth is not None:
        truth_file, truth_record = _ref_record(declared_truth, base=base, description="truth sidecar")
    else:
        truth_file = _resolve_path(truth_path, base=base, description="truth sidecar")
        truth_record = file_record(truth_file, base=base, description="truth sidecar")
    if truth_path is not None:
        supplied = file_record(Path(truth_path), base=base, description="supplied truth sidecar")
        _same_record(supplied, truth_record, description="truth sidecar")

    expected_counts = _records_by_cell(descriptor)
    if expected_counts != frozen.records_by_cell:
        raise ExecutionError("truth descriptor record counts differ from frozen panel")
    try:
        with safe_open(str(truth_file), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            keys = set(handle.keys())
            key_map = _truth_keys(descriptor)
            values: dict[str, torch.Tensor] = {}
            for cell in CELL_ORDER:
                key = key_map[cell]
                if key not in keys:
                    raise ExecutionError(f"truth tensor is missing: {cell}")
                tensor = handle.get_tensor(key).detach().cpu().contiguous()
                if tuple(tensor.shape) != (frozen.records_by_cell[cell], STORED_SEQUENCE_TOKENS):
                    raise ExecutionError(f"truth geometry changed: {cell}")
                if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    raise ExecutionError(f"truth dtype is not integer: {cell}")
                if tensor[:, 0].ne(BOS_TOKEN_ID).any().item():
                    raise ExecutionError(f"truth BOS column changed: {cell}")
                if tensor[:, 1:].lt(0).any().item() or tensor[:, 1:].ge(VOCABULARY_SIZE).any().item():
                    raise ExecutionError(f"truth contains invalid token IDs: {cell}")
                values[cell] = tensor.to(dtype=torch.long)
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError("truth sidecar is unreadable") from exc

    return values, {
        "schema": "token-reconstruction.trr-p10-truth-load.v1",
        "truth_descriptor": descriptor_record,
        "truth_sidecar": truth_record,
        "truth_sidecar_metadata": {str(k): str(v) for k, v in metadata.items()},
        "status": "TRUTH_LOADED_AFTER_PUBLIC_FREEZE",
    }


def _correctness(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    if tuple(prediction.shape) != tuple(truth.shape):
        raise ExecutionError("prediction/truth geometry differs")
    return prediction[:, 1:].eq(truth[:, 1:])


def _pair_counts(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    both = left & right
    left_only = left & ~right
    right_only = ~left & right
    neither = ~left & ~right
    return {
        "both_correct": int(both.sum().item()),
        "left_only_correct": int(left_only.sum().item()),
        "right_only_correct": int(right_only.sum().item()),
        "both_wrong": int(neither.sum().item()),
        "denominator": int(left.numel()),
        "by_position": [
            {
                "position": int(index + 1),
                "both_correct": int(both[:, index].sum().item()),
                "left_only_correct": int(left_only[:, index].sum().item()),
                "right_only_correct": int(right_only[:, index].sum().item()),
                "both_wrong": int(neither[:, index].sum().item()),
            }
            for index in range(left.shape[1])
        ],
    }


def _record_pair_counts(left: torch.Tensor, right: torch.Tensor) -> dict[str, int]:
    return {
        "both_correct": int((left & right).sum().item()),
        "left_only_correct": int((left & ~right).sum().item()),
        "right_only_correct": int((~left & right).sum().item()),
        "both_wrong": int((~left & ~right).sum().item()),
        "denominator": int(left.numel()),
    }


def _rank_availability(frozen: FrozenPackage) -> dict[str, Any]:
    """Report rank/proposal evidence without reconstructing unavailable top-K."""
    result: dict[str, Any] = {}
    a1_keys = ("proposal_true_rank", "true_token_proposal_rank", "a1_true_rank", "true_in_proposal")
    decoder_keys = ("decoder_true_rank", "a2_true_rank", "winner_true_rank", "selected_rank", "true_token_rank")
    for label, keys in (("a1_proposal", a1_keys), ("a2_decoder_ranking", decoder_keys)):
        found = []
        for key, metadata in frozen.prediction_metadata.items():
            present = [field for field in keys if field in metadata]
            if present:
                found.append({"prediction": key, "fields": present})
        if found:
            result[label] = {"status": "AVAILABLE_METADATA_ONLY", "evidence": found}
            continue
        trace_evidence = []
        for trace_key, trace_record in frozen.trace_records.items():
            label_text = str(trace_record.get("label", "")).lower()
            if label == "a1_proposal" and not any(token in label_text for token in ("proposal", "candidate")):
                continue
            if label == "a2_decoder_ranking" and not any(token in label_text for token in ("rank", "decoder")):
                continue
            trace_evidence.append({"trace": trace_key, "path": trace_record.get("path")})
        if trace_evidence:
            result[label] = {"status": "AVAILABLE_BOUND_TRACE_OPAQUE", "evidence": trace_evidence}
        else:
            result[label] = {
                "status": "UNAVAILABLE",
                "reason": "frozen_prediction_artifacts_contain_only_final_predictions;_no_topk_or_true_rank_trace",
                "method": "frozen_a1_a2_k256" if label == "a1_proposal" else "frozen_a1_a2_k256",
            }
    return result


def _alignment_summary(frozen: FrozenPackage, key: str) -> dict[str, Any]:
    method, cell = key.split("::", 1)
    indices = frozen.subset_indices_by_method_cell[key]
    if indices is None:
        return {
            "mode": "full_panel",
            "records": int(frozen.records_by_method_cell[key]),
            "parent_records": int(frozen.records_by_cell[cell]),
        }
    return {
        "mode": "parent_subset",
        "records": int(len(indices)),
        "parent_records": int(frozen.records_by_cell[cell]),
        "indices_sha256": indices_digest(list(indices)),
        "method": method,
        "cell": cell,
    }


def _aligned_pair(
    frozen: FrozenPackage,
    cell: str,
    left_method: str,
    right_method: str,
    per_method: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    left_key = f"{left_method}::{cell}"
    right_key = f"{right_method}::{cell}"
    left_indices = frozen.subset_indices_by_method_cell[left_key]
    right_indices = frozen.subset_indices_by_method_cell[right_key]
    left = per_method[left_method]
    right = per_method[right_method]
    if left_indices is None and right_indices is None:
        if tuple(left.shape) != tuple(right.shape):
            raise ExecutionError(f"cannot pair unequal full-panel arrays: {left_key} vs {right_key}")
        return left, right, {
            "mode": "full_panel",
            "records": int(left.shape[0]),
            "parent_records": int(frozen.records_by_cell[cell]),
        }
    if left_indices is not None and right_indices is not None:
        if left_indices != right_indices:
            raise ExecutionError(f"paired subset indices differ: {left_key} vs {right_key}")
        if tuple(left.shape) != tuple(right.shape):
            raise ExecutionError(f"paired subset arrays have different geometry: {left_key} vs {right_key}")
        return left, right, {
            "mode": "parent_subset",
            "records": int(len(left_indices)),
            "parent_records": int(frozen.records_by_cell[cell]),
            "indices_sha256": indices_digest(list(left_indices)),
        }
    if left_indices is None:
        # The full-panel method is projected onto the right method's explicit
        # parent rows before any pair category is computed.
        assert right_indices is not None
        selector = torch.tensor(right_indices, dtype=torch.long)
        left = left.index_select(0, selector)
        mode = "right_method_parent_subset"
        indices = right_indices
    else:
        assert right_indices is None
        selector = torch.tensor(left_indices, dtype=torch.long)
        right = right.index_select(0, selector)
        mode = "left_method_parent_subset"
        indices = left_indices
    if tuple(left.shape) != tuple(right.shape):
        raise ExecutionError(f"subset projection did not align pair: {left_key} vs {right_key}")
    return left, right, {
        "mode": mode,
        "records": int(len(indices)),
        "parent_records": int(frozen.records_by_cell[cell]),
        "indices_sha256": indices_digest(list(indices)),
    }


def build_error_inventory(
    frozen: FrozenPackage,
    truth: Mapping[str, torch.Tensor],
    *,
    truth_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create paired expanded/current/A1+A2 error inventories after truth.

    Inventories are kept per cell.  No domains or target conditions are pooled,
    and rank/proposal decomposition is reported as unavailable unless explicit
    rank evidence was serialized in the frozen package.
    """
    if set(truth) != set(CELL_ORDER):
        raise ExecutionError("truth cells do not match frozen cells")
    cells: dict[str, Any] = {}
    for cell in CELL_ORDER:
        per_method: dict[str, torch.Tensor] = {}
        method_summary: dict[str, Any] = {}
        target = torch.as_tensor(truth[cell]).detach().cpu().contiguous()
        if tuple(target.shape) != (frozen.records_by_cell[cell], STORED_SEQUENCE_TOKENS):
            raise ExecutionError(f"truth geometry differs from frozen full panel: {cell}")
        for method in METHOD_ORDER:
            key = f"{method}::{cell}"
            indices = frozen.subset_indices_by_method_cell[key]
            method_target = target
            if indices is not None:
                method_target = target.index_select(0, torch.tensor(indices, dtype=torch.long))
            correct = _correctness(frozen.predictions[key], method_target)
            per_method[method] = correct
            method_summary[method] = {
                "records": int(correct.shape[0]),
                "scored_post_bos_tokens": int(correct.numel()),
                "correct_tokens": int(correct.sum().item()),
                "token_errors": int((~correct).sum().item()),
                "exact_correct_records": int(correct.all(dim=1).sum().item()),
                "exact_error_records": int((~correct.all(dim=1)).sum().item()),
                "alignment": _alignment_summary(frozen, key),
            }
        pair_specs = {
            "expanded_fixed_vs_current_fixed": ("expanded_fixed", "current_fixed"),
            "expanded_fixed_vs_a1_a2": ("expanded_fixed", "frozen_a1_a2_k256"),
            "current_fixed_vs_a1_a2": ("current_fixed", "frozen_a1_a2_k256"),
        }
        pairs: dict[str, Any] = {}
        for name, (left_method, right_method) in pair_specs.items():
            left, right, alignment = _aligned_pair(frozen, cell, left_method, right_method, per_method)
            pairs[name] = {
                "left_method": left_method,
                "right_method": right_method,
                "alignment": alignment,
                "tokens": _pair_counts(left, right),
                "records_exact": _record_pair_counts(left.all(dim=1), right.all(dim=1)),
            }
        cells[cell] = {
            "records": int(target.shape[0]),
            "method_summary": method_summary,
            "pairwise": pairs,
            "rank_and_proposal_diagnostics": _rank_availability(frozen),
        }
    result = {
        "schema": "token-reconstruction.trr-p10-paired-error-inventory.v1",
        "task_id": TASK_ID,
        "status": "SCORED_AFTER_PUBLIC_FREEZE",
        "freeze": frozen.freeze_record,
        "registration": frozen.registration_record,
        "truth_binding": dict(truth_binding or {}),
        "method_order": list(METHOD_ORDER),
        "cell_order": list(CELL_ORDER),
        "records_by_cell": dict(frozen.records_by_cell),
        "records_by_method_cell": dict(frozen.records_by_method_cell),
        "subset_bindings": {
            key: _alignment_summary(frozen, key)
            for key, indices in frozen.subset_indices_by_method_cell.items()
            if indices is not None
        },
        "cells": cells,
        "decomposition_policy": {
            "a1_proposal_failures": "UNAVAILABLE unless explicit true-in-proposal/rank evidence is bound",
            "decoder_ranking_failures": "UNAVAILABLE unless explicit decoder rank evidence is bound",
            "final_prediction_mismatch_is_not_a_candidate_rank": True,
        },
    }
    return result


def write_inventory(path: Path, inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Write a create-only JSON inventory and return its file binding."""
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ExecutionError(f"refusing to overwrite inventory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(inventory, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise ExecutionError(f"refusing to overwrite inventory: {path}") from exc
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


__all__ = [
    "CELL_ORDER",
    "METHOD_ORDER",
    "ExecutionError",
    "FrozenPackage",
    "build_error_inventory",
    "file_record",
    "indices_digest",
    "load_truth_after_freeze",
    "sha256_file",
    "tensor_digest",
    "validate_frozen_package",
    "write_inventory",
]
