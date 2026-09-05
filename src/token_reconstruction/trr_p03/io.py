"""Opaque observation and immutable prediction-bundle I/O for TRR-P03.

The public observation format supports both one-record files and the grouped
six-record files declared by the setup interface.  In either form the loader
returns one validated :class:`BoundaryObservation` per opaque record.  Truth
sidecars are intentionally not read here; only the post-freeze scorer opens
them.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from token_reconstruction.access import AccessContractError, BoundaryObservation


OBSERVATION_INDEX_SCHEMA = "token-reconstruction.trr-p03-observation-index.v1"
OBSERVATION_INDEX_TEMPLATE_SCHEMA = "token-reconstruction.trr-p03-observation-index-template.v1"
OBSERVATION_BUNDLE_SCHEMA = "token-reconstruction.trr-p03-observation-bundle.v1"
PANEL_SCHEMA = "token-reconstruction.trr-p03-panel.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr-p03-predictions.v1"
FREEZE_SCHEMA = "token-reconstruction.trr-p03-freeze-receipt.v1"
TRUTH_SCHEMA = "token-reconstruction.trr-p03-truth.v1"
HIDDEN_SIZE = 2048
BOS_TOKEN_ID = 128000
VOCAB_SIZE = 128256
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
CUT_DEPTH = 4


class P03IOError(RuntimeError):
    """Raised when a task-local artifact is malformed or mutable."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise P03IOError(f"artifact must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise P03IOError(f"artifact must be a regular file: {path}")
    label = str(path)
    if root is not None:
        try:
            label = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            label = str(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise P03IOError(f"JSON input must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P03IOError(f"invalid JSON: {path}") from exc


def write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise P03IOError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise P03IOError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise P03IOError(f"JSONL input must be a regular file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise P03IOError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise P03IOError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def create_only_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise P03IOError(f"output directory must be create-only: {path}")
    path.mkdir(parents=True)
    return path.resolve()


def create_only_file(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise P03IOError(f"output file must be create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _digest_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    descriptor = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(descriptor + b"\0" + raw).hexdigest()


def _relative_path(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise P03IOError(f"{name} must be a relative path")
    parts = Path(value).parts
    if ".." in parts or "" in parts:
        raise P03IOError(f"{name} must not escape the observation root")
    return value.replace("\\", "/")


def _validate_public_record_fields(record: Mapping[str, Any]) -> None:
    # Keep this exact set aligned with the setup interface.  In particular,
    # style/category/stage and all source-side IDs are evaluator-only.
    allowed = {
        "record_id",
        "sequence_length",
        "path",
        "bytes",
        "sha256",
        "shape",
        "dtype",
        "mask_digest",
        "position_digest",
    }
    if set(record) - allowed:
        raise P03IOError("observation index exposes unexpected per-record fields")
    record_id = record.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise P03IOError("observation record_id must be non-empty")
    try:
        length = int(record.get("sequence_length", 0))
    except (TypeError, ValueError) as exc:
        raise P03IOError("observation sequence length is invalid") from exc
    if length < 2:
        raise P03IOError("every observation must contain BOS and one scored token")
    _relative_path(record.get("path"), name="observation path")
    if list(record.get("shape", ())) != [1, length, HIDDEN_SIZE]:
        raise P03IOError("observation shape disagrees with sequence length")
    if not isinstance(record.get("bytes"), int) or int(record["bytes"]) <= 0:
        raise P03IOError("observation byte size is invalid")
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise P03IOError("observation hash is invalid")


def _common_index_checks(index: Mapping[str, Any]) -> None:
    if index.get("schema") != OBSERVATION_INDEX_SCHEMA:
        raise P03IOError("unsupported TRR-P03 observation index schema")
    if index.get("truth_opened") is not False or index.get("source_truth_included") is not False:
        raise P03IOError("observation index is not truth-free")
    model = index.get("model")
    if not isinstance(model, Mapping) or model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise P03IOError("public model identity changed")
    try:
        cut_depth = int(index.get("cut_depth", -1))
        bos = int(index.get("bos_token_id", -1))
    except (TypeError, ValueError) as exc:
        raise P03IOError("observation cut or BOS identity is invalid") from exc
    if cut_depth != CUT_DEPTH or bos != BOS_TOKEN_ID:
        raise P03IOError("observation cut or BOS identity changed")


def _validate_group_descriptor(group: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "bundle_id",
        "stage",
        "scored_tokens",
        "sequence_length",
        "record_ids",
        "relative_path",
        "keys",
        "expected_shapes",
        "bytes",
        "sha256",
    }
    if not required.issubset(group):
        raise P03IOError("observation bundle descriptor is incomplete")
    if set(group) - required:
        raise P03IOError("observation bundle descriptor exposes unexpected fields")
    bundle_id = group.get("bundle_id")
    stage = group.get("stage")
    if not isinstance(bundle_id, str) or not bundle_id or not isinstance(stage, str) or not stage:
        raise P03IOError("observation bundle identity is invalid")
    try:
        scored = int(group["scored_tokens"])
        sequence = int(group["sequence_length"])
    except (TypeError, ValueError) as exc:
        raise P03IOError("observation bundle geometry is invalid") from exc
    if scored <= 0 or sequence != scored + 1:
        raise P03IOError("observation bundle sequence geometry changed")
    ids = group.get("record_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids):
        raise P03IOError("observation bundle record IDs are invalid")
    if len(set(ids)) != len(ids):
        raise P03IOError("observation bundle record IDs are duplicated")
    _relative_path(group.get("relative_path"), name="observation bundle path")
    keys = group.get("keys")
    expected_keys = {"activations": "activations", "attention_mask": "attention_mask", "position_ids": "position_ids"}
    if keys != expected_keys:
        raise P03IOError("observation bundle tensor keys changed")
    shapes = group.get("expected_shapes")
    expected_shapes = {
        "activations": [len(ids), sequence, HIDDEN_SIZE],
        "attention_mask": [len(ids), sequence],
        "position_ids": [len(ids), sequence],
    }
    if shapes != expected_shapes:
        raise P03IOError("observation bundle tensor shapes changed")
    if not isinstance(group.get("bytes"), int) or group["bytes"] <= 0:
        raise P03IOError("observation bundle byte size is invalid")
    digest = group.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise P03IOError("observation bundle hash is invalid")
    return dict(group)


def validate_observation_index(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate a truth-free index and return ordered record descriptors.

    The public JSON may contain either ``records`` (one-record artifacts) or
    ``bundles`` (the setup interface's grouped artifacts).  Group descriptors
    are expanded into internal row descriptors; the private ``_bundle_row``
    marker is never serialized and is consumed only by the loader.
    """

    if not isinstance(index, Mapping):
        raise P03IOError("observation index root must be an object")
    _common_index_checks(index)
    records = index.get("records")
    if isinstance(records, list) and records:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise P03IOError("observation record is not an object")
            _validate_public_record_fields(record)
            record_id = str(record["record_id"])
            if record_id in seen:
                raise P03IOError("observation record IDs are duplicated")
            seen.add(record_id)
            result.append(dict(record))
        return result

    groups = index.get("bundles")
    if not isinstance(groups, list) or not groups:
        raise P03IOError("observation index has no records or bundles")
    result = []
    seen: set[str] = set()
    seen_paths: set[str] = set()
    for raw_group in groups:
        if not isinstance(raw_group, Mapping):
            raise P03IOError("observation bundle is not an object")
        group = _validate_group_descriptor(raw_group)
        path = str(group["relative_path"])
        if path in seen_paths:
            raise P03IOError("observation bundle paths are duplicated")
        seen_paths.add(path)
        length = int(group["sequence_length"])
        for row_index, record_id in enumerate(group["record_ids"]):
            if record_id in seen:
                raise P03IOError("observation record IDs are duplicated")
            seen.add(record_id)
            result.append(
                {
                    "record_id": record_id,
                    "sequence_length": length,
                    "path": path,
                    "bytes": int(group["bytes"]),
                    "sha256": str(group["sha256"]),
                    "shape": [1, length, HIDDEN_SIZE],
                    "dtype": "bfloat16",
                    "mask_digest": "",
                    "position_digest": "",
                    "_bundle_row": row_index,
                }
            )
    return result


def save_boundary_observation(observation: BoundaryObservation, path: Path) -> str:
    """Write one observation with create-only safetensors semantics."""

    create_only_file(path)
    try:
        observation.validate()
    except AccessContractError as exc:
        raise P03IOError("boundary observation violates access contract") from exc
    metadata = {
        "schema": observation.schema,
        "task_schema": OBSERVATION_INDEX_SCHEMA,
        "cut_depth": str(observation.cut_depth),
        "source_id": observation.source_id,
        "metadata_json": json.dumps(
            dict(observation.metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    }
    save_file(
        {
            "activation": observation.activation.detach().cpu().contiguous(),
            "attention_mask": observation.attention_mask.detach().cpu().to(torch.int64).contiguous(),
            "position_ids": observation.position_ids.detach().cpu().to(torch.int64).contiguous(),
        },
        path,
        metadata=metadata,
    )
    return sha256_file(path)


def save_observation_bundle(
    *,
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    path: Path,
    bundle_id: str,
    stage: str,
    record_ids: list[str],
) -> str:
    """Write one grouped opaque bundle and return its file hash."""

    if path.exists() or path.is_symlink():
        raise P03IOError(f"observation bundle already exists: {path}")
    if not isinstance(bundle_id, str) or not bundle_id or not isinstance(stage, str) or not stage:
        raise P03IOError("observation bundle identity is invalid")
    if not isinstance(record_ids, list) or not record_ids or len(set(record_ids)) != len(record_ids):
        raise P03IOError("observation bundle record IDs are invalid")
    if activations.ndim != 3 or attention_mask.ndim != 2 or position_ids.ndim != 2:
        raise P03IOError("observation bundle tensors have invalid rank")
    count, sequence, hidden = map(int, activations.shape)
    if count != len(record_ids) or sequence < 2 or hidden != HIDDEN_SIZE:
        raise P03IOError("observation bundle activation geometry changed")
    if tuple(attention_mask.shape) != (count, sequence) or tuple(position_ids.shape) != (count, sequence):
        raise P03IOError("observation bundle mask/position geometry changed")
    if activations.dtype != torch.bfloat16:
        raise P03IOError("observation bundle activations must be bfloat16")
    if not torch.isfinite(activations).all().item():
        raise P03IOError("observation bundle activations are invalid")
    if attention_mask.dtype.is_floating_point or position_ids.dtype.is_floating_point:
        raise P03IOError("observation bundle masks/positions must be integral")
    if not torch.logical_or(attention_mask.eq(0), attention_mask.eq(1)).all().item():
        raise P03IOError("observation bundle mask is not binary")
    expected = torch.arange(sequence, dtype=torch.long).view(1, -1)
    if not torch.equal(position_ids.to(torch.long), expected.expand(count, -1)):
        raise P03IOError("observation bundle positions are not contiguous")
    if not attention_mask.to(torch.bool).all().item():
        raise P03IOError("observation bundle rows must be fully active")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "activations": activations.detach().cpu().contiguous(),
            "attention_mask": attention_mask.detach().cpu().to(torch.int64).contiguous(),
            "position_ids": position_ids.detach().cpu().to(torch.int64).contiguous(),
        },
        path,
        metadata={
            "schema": OBSERVATION_BUNDLE_SCHEMA,
            "task_id": "TRR-P03",
            "bundle_id": bundle_id,
            "stage": stage,
            "cut_depth": str(CUT_DEPTH),
            "truth_opened": "false",
            "source_truth_included": "false",
            "record_ids_json": json.dumps(record_ids, separators=(",", ":")),
        },
    )
    return sha256_file(path)


def load_boundary_observation(path: Path) -> BoundaryObservation:
    if path.is_symlink() or not path.is_file():
        raise P03IOError(f"observation must be a regular file: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activation", "attention_mask", "position_ids"}:
                raise P03IOError("observation tensor fields changed")
            metadata = handle.metadata() or {}
            required = {"schema", "task_schema", "cut_depth", "source_id", "metadata_json"}
            if set(metadata) != required or metadata["task_schema"] != OBSERVATION_INDEX_SCHEMA:
                raise P03IOError("observation metadata fields changed")
            if metadata["schema"] != "token-reconstruction.boundary-observation.v1":
                raise P03IOError("observation access schema changed")
            observation = BoundaryObservation(
                activation=handle.get_tensor("activation"),
                attention_mask=handle.get_tensor("attention_mask"),
                position_ids=handle.get_tensor("position_ids"),
                cut_depth=int(metadata["cut_depth"]),
                source_id=metadata["source_id"],
                metadata=json.loads(metadata["metadata_json"]),
            )
    except P03IOError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise P03IOError(f"invalid observation artifact: {path}") from exc
    try:
        observation.validate()
    except AccessContractError as exc:
        raise P03IOError("observation violates access contract") from exc
    if tuple(observation.activation.shape) != (1, int(observation.activation.shape[1]), HIDDEN_SIZE):
        raise P03IOError("observation geometry changed")
    return observation


def _resolve_under(base: Path, relative: str) -> Path:
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise P03IOError("observation path escaped index directory") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise P03IOError(f"observation artifact missing: {candidate}")
    return candidate


def _load_grouped_file(
    path: Path,
    *,
    descriptor: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise P03IOError("observation bundle tensor fields changed")
            metadata = handle.metadata() or {}
            if metadata:
                if metadata.get("schema") != OBSERVATION_BUNDLE_SCHEMA or metadata.get("truth_opened") != "false" or metadata.get("source_truth_included") != "false":
                    raise P03IOError("observation bundle metadata is not truth-free")
            activation = handle.get_tensor("activations")
            mask = handle.get_tensor("attention_mask")
            positions = handle.get_tensor("position_ids")
    except P03IOError:
        raise
    except Exception as exc:
        raise P03IOError(f"invalid observation bundle: {path}") from exc
    expected_shapes = descriptor["expected_shapes"]
    if list(activation.shape) != list(expected_shapes["activations"]):
        raise P03IOError("observation bundle activation shape changed")
    if list(mask.shape) != list(expected_shapes["attention_mask"]) or list(positions.shape) != list(expected_shapes["position_ids"]):
        raise P03IOError("observation bundle mask/position shape changed")
    if activation.dtype != torch.bfloat16 or not torch.isfinite(activation).all().item():
        raise P03IOError("observation bundle activation dtype or values changed")
    if mask.dtype.is_floating_point or positions.dtype.is_floating_point:
        raise P03IOError("observation bundle mask/position dtype changed")
    expected_positions = torch.arange(int(descriptor["sequence_length"]), dtype=torch.long).view(1, -1)
    if not torch.equal(positions.to(torch.long), expected_positions.expand(positions.shape[0], -1)):
        raise P03IOError("observation bundle positions changed")
    if not mask.to(torch.bool).all().item():
        raise P03IOError("observation bundle contains inactive rows")
    return activation, mask.to(torch.int64), positions.to(torch.int64)


def load_index_and_observations(
    index_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[BoundaryObservation]]:
    """Load and validate a public index, expanding grouped bundles."""

    index = read_json(index_path)
    if not isinstance(index, Mapping):
        raise P03IOError("observation index root must be an object")
    records = validate_observation_index(index)
    base = index_path.parent.resolve()
    grouped_by_path: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    descriptors_by_path: dict[str, Mapping[str, Any]] = {
        str(group["relative_path"]): group
        for group in index.get("bundles", [])
        if isinstance(group, Mapping)
    }
    observations: list[BoundaryObservation] = []
    for record in records:
        relative = str(record["path"])
        candidate = _resolve_under(base, relative)
        if int(record.get("bytes", -1)) != candidate.stat().st_size or str(record.get("sha256")) != sha256_file(candidate):
            raise P03IOError(f"observation hash changed: {candidate}")
        row_index = record.get("_bundle_row")
        if row_index is None:
            observation = load_boundary_observation(candidate)
            if int(observation.activation.shape[1]) != int(record["sequence_length"]):
                raise P03IOError("observation sequence length changed")
            observations.append(observation)
            continue
        descriptor = descriptors_by_path.get(relative)
        if descriptor is None:
            raise P03IOError("grouped observation descriptor is missing")
        if relative not in grouped_by_path:
            grouped_by_path[relative] = _load_grouped_file(candidate, descriptor=descriptor)
        activation, mask, positions = grouped_by_path[relative]
        row_index = int(row_index)
        if row_index < 0 or row_index >= activation.shape[0]:
            raise P03IOError("grouped observation row index is invalid")
        rid = str(record["record_id"])
        observation = BoundaryObservation(
            activation=activation[row_index : row_index + 1].contiguous(),
            attention_mask=mask[row_index : row_index + 1].contiguous(),
            position_ids=positions[row_index : row_index + 1].contiguous(),
            cut_depth=CUT_DEPTH,
            source_id=rid,
            metadata={"sequence_length": int(record["sequence_length"])},
        )
        try:
            observation.validate()
        except AccessContractError as exc:
            raise P03IOError("grouped observation violates access contract") from exc
        observations.append(observation)
    return dict(index), records, observations


def freeze_prediction_bundle(
    *,
    root: Path,
    plan_hash: str,
    implementation_commit: str,
    artifacts: Iterable[Path],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash prediction artifacts before truth is read and make them read-only."""

    if not isinstance(plan_hash, str) or len(plan_hash) != 64:
        raise P03IOError("prediction freeze requires a plan hash")
    root = root.resolve()
    entries: list[dict[str, Any]] = []
    for artifact in sorted({path.resolve() for path in artifacts}):
        try:
            relative = artifact.relative_to(root).as_posix()
        except ValueError as exc:
            raise P03IOError("frozen artifact escaped prediction root") from exc
        if artifact.is_symlink() or not artifact.is_file():
            raise P03IOError(f"frozen artifact must be a regular file: {artifact}")
        entries.append(
            {
                "path": relative,
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    if not entries:
        raise P03IOError("prediction freeze has no artifacts")
    payload = {
        "schema": FREEZE_SCHEMA,
        "task_id": "TRR-P03",
        "status": "PREDICTIONS_FROZEN_BEFORE_TRUTH",
        "truth_opened": False,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan_sha256": plan_hash,
        "implementation_commit": implementation_commit,
        "frozen_root": str(root),
        "entries": entries,
        "metadata": dict(metadata or {}),
    }
    return payload


def write_freeze_receipt(root: Path, payload: Mapping[str, Any]) -> Path:
    receipt = root / "freeze_receipt.json"
    write_json_exclusive(receipt, payload)
    for entry in payload["entries"]:
        (root / str(entry["path"])).chmod(0o444)
    receipt.chmod(0o444)
    return receipt


def verify_freeze_receipt(root: Path) -> dict[str, Any]:
    receipt = root / "freeze_receipt.json"
    payload = read_json(receipt)
    if not isinstance(payload, Mapping) or payload.get("schema") != FREEZE_SCHEMA:
        raise P03IOError("prediction freeze receipt schema changed")
    if payload.get("truth_opened") is not False or payload.get("status") != "PREDICTIONS_FROZEN_BEFORE_TRUTH":
        raise P03IOError("prediction outputs were not frozen before truth")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise P03IOError("prediction freeze receipt has no entries")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise P03IOError("prediction freeze entry is malformed")
        relative = str(entry["path"])
        if relative in seen or relative.startswith("/") or ".." in Path(relative).parts:
            raise P03IOError("prediction freeze path is invalid")
        seen.add(relative)
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root.resolve())
        except ValueError as exc:
            raise P03IOError("prediction freeze path escaped root") from exc
        if artifact.is_symlink() or not artifact.is_file():
            raise P03IOError(f"frozen artifact missing: {artifact}")
        if artifact.stat().st_size != int(entry.get("bytes", -1)) or sha256_file(artifact) != entry.get("sha256"):
            raise P03IOError(f"frozen artifact hash changed: {artifact}")
    return dict(payload)


__all__ = [
    "BOS_TOKEN_ID",
    "CUT_DEPTH",
    "FREEZE_SCHEMA",
    "HIDDEN_SIZE",
    "MODEL_ID",
    "MODEL_REVISION",
    "OBSERVATION_BUNDLE_SCHEMA",
    "OBSERVATION_INDEX_SCHEMA",
    "OBSERVATION_INDEX_TEMPLATE_SCHEMA",
    "P03IOError",
    "PANEL_SCHEMA",
    "PREDICTION_SCHEMA",
    "TRUTH_SCHEMA",
    "VOCAB_SIZE",
    "create_only_directory",
    "create_only_file",
    "file_record",
    "freeze_prediction_bundle",
    "load_boundary_observation",
    "load_index_and_observations",
    "read_json",
    "read_jsonl",
    "save_boundary_observation",
    "save_observation_bundle",
    "sha256_file",
    "validate_observation_index",
    "verify_freeze_receipt",
    "write_freeze_receipt",
    "write_json_exclusive",
    "write_jsonl_exclusive",
]
