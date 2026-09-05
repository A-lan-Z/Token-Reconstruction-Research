#!/usr/bin/env python3
"""Fit the bounded TRR-0004 activation-context extensions.

This runner consumes a freshly fitted public historical-style affine state and
public cut-4 activation/token tensors.  The affine base stays frozen while a
zero-initialized causal H-only attention path and a parameter-matched
positionwise MLP path are fitted with the same public position schedule.

The full-vocabulary projection is applied only to the selected loss rows (at
most 512 per batch).  Validation selection uses public labels only.  Runtime
reconstruction is represented by the saved decoder state and activation-only
forward path; this runner contains no public-prefix calls, candidate search, or
A2 fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from token_reconstruction.causal_decoder_extension import (
    CAUSAL_ATTENTION_METHOD,
    EXTENSION_METHODS,
    CausalDecoderExtensionError,
    CausalResidualDecoder,
    FrozenAffineBase,
    build_causal_extension,
    validate_runtime_embeddings,
)
from token_reconstruction.historical_affine_ce import (
    file_sha256,
    validate_normalized_embedding_table,
)


TASK_ID = "TRR-0004"
SCHEMA = "token-reconstruction.trr0004-contextual-fit.v1"
FIT_DATA_SCHEMA = "token-reconstruction.trr0004-public-fit-data.v1"
BOS_TOKEN_ID = 128000
DEFAULT_RECORD_BATCH_SIZE = 8
DEFAULT_POSITION_BUDGET = 512
DEFAULT_STEPS = 3000
DEFAULT_SUBSET_STEPS = 600
DEFAULT_VALIDATION_EVERY = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_SEED = 1737
DEFAULT_MINIMUM_FREE_GIB = 8.0
DEFAULT_MAXIMUM_GPU_RESERVED_GIB = 6.0
DEFAULT_MAXIMUM_HOST_RSS_GIB = 16.0
DEFAULT_MAX_SECONDS = 1200.0


class ContextualFitError(RuntimeError):
    """Raised when the public contextual-fit contract cannot be established."""


@dataclass(frozen=True)
class PublicContextData:
    fit_observations: torch.Tensor
    fit_truth: torch.Tensor
    fit_valid_mask: torch.Tensor
    validation_observations: torch.Tensor
    validation_truth: torch.Tensor
    validation_valid_mask: torch.Tensor
    embedding_table: torch.Tensor
    fit_record_ids: tuple[str, ...]
    validation_record_ids: tuple[str, ...]
    validation_groups: tuple[str, ...]
    validation_native_geometries: tuple[tuple[int, int, int], ...]
    validation_padding: Mapping[str, Any]
    resource_records: Mapping[str, Mapping[str, Any]]
    registration_sha256: str
    fit_manifest_sha256: str | None

    @property
    def hidden_size(self) -> int:
        return int(self.fit_observations.shape[-1])

    @property
    def vocabulary_size(self) -> int:
        return int(self.embedding_table.shape[0])


@dataclass(frozen=True)
class PositionSchedule:
    record_indices: torch.Tensor
    selected_mask: torch.Tensor
    seed: int
    position_budget: int
    record_batch_size: int

    @property
    def steps(self) -> int:
        return int(self.record_indices.shape[0])


@dataclass(frozen=True)
class _InputPaths:
    fit_observations: Path
    fit_truth: Path
    fit_records: Path
    validation_observations: tuple[Path, ...]
    validation_truth: tuple[Path, ...]
    validation_records: tuple[Path, ...]
    embedding_table: Path
    fit_valid_mask: Path | None = None
    validation_valid_mask: Path | None = None
    fit_combined_artifact: Path | None = None
    validation_combined_artifact: tuple[Path, ...] = ()
    resource_records: Mapping[str, Mapping[str, Any]] | None = None
    fit_manifest_sha256: str | None = None


def _regular_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ContextualFitError(f"{label} must be a regular file: {path}")
    return path


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _tensor_digest(value: torch.Tensor) -> str:
    """Hash tensor shape, dtype, and exact bytes, including BF16/scalars."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}, sort_keys=True).encode("utf-8")
    )
    # Flatten before the byte view: torch disallows a direct uint8 view of a
    # zero-dimensional float tensor even though the underlying bytes are valid.
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[1]
    return (
        Path(__file__).resolve(),
        root / "src/token_reconstruction/causal_decoder_extension.py",
        root / "src/token_reconstruction/historical_affine_ce.py",
    )


def _source_records() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in _source_paths():
        record = _file_record(path, label="executed source")
        result[str(path)] = record
    return result


def _source_bundle_hash(records: Mapping[str, Mapping[str, Any]]) -> str:
    return _canonical_hash(records)


def _json_load(path: Path, *, label: str) -> Any:
    path = _regular_file(path, label=label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextualFitError(f"cannot parse {label}: {path}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ContextualFitError(f"artifact is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_tensor(path: Path, *, key: str, label: str) -> torch.Tensor:
    path = _regular_file(path, label=label)
    try:
        tensors = load_file(str(path), device="cpu")
    except Exception as exc:  # pragma: no cover - backend-specific
        raise ContextualFitError(f"cannot load {label}: {path}") from exc
    if set(tensors) != {key}:
        raise ContextualFitError(f"{label} must contain exactly the {key!r} tensor")
    return tensors[key].contiguous()


def _manifest_resource(
    root: Path, resources: Mapping[str, Any], name: str, *, verify_hash: bool = True
) -> tuple[Path, dict[str, Any]]:
    entry = resources.get(name)
    if not isinstance(entry, Mapping):
        raise ContextualFitError(f"fit manifest resource {name!r} is missing")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ContextualFitError(f"fit manifest resource {name!r} has no path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    expected_sha = entry.get("sha256")
    expected_bytes = entry.get("bytes")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ContextualFitError(f"fit manifest resource {name!r} hash is missing")
    if expected_bytes is not None and int(expected_bytes) < 0:
        raise ContextualFitError(f"fit manifest resource {name!r} byte count is invalid")
    # Hashing a public truth tensor is deferred until record identity and
    # overlap checks have passed.  Opening a regular file to verify its path is
    # safe; reading its bytes before the split gate would violate the loader
    # ordering contract even though these labels are public.
    if verify_hash:
        record = _file_record(path, label=name)
        if record["sha256"] != expected_sha:
            raise ContextualFitError(f"fit manifest resource {name!r} hash changed")
        if expected_bytes is not None and int(expected_bytes) != record["bytes"]:
            raise ContextualFitError(f"fit manifest resource {name!r} byte count changed")
        merged = dict(entry)
        merged.update(record)
    else:
        merged = dict(entry)
        merged["path"] = str(path.resolve())
        merged["_deferred_expected_sha256"] = expected_sha
    return path.resolve(), merged


def _resolve_input_paths(args: argparse.Namespace) -> _InputPaths:
    if args.fit_manifest is not None:
        manifest_path = _regular_file(args.fit_manifest, label="fit manifest")
        manifest = _json_load(manifest_path, label="fit manifest")
        if not isinstance(manifest, Mapping) or manifest.get("schema") != FIT_DATA_SCHEMA:
            raise ContextualFitError("fit manifest schema changed")
        resources = manifest.get("resources")
        if not isinstance(resources, Mapping):
            raise ContextualFitError("fit manifest resources are missing")
        root = manifest_path.parent
        names = (
            "fit_observations",
            "fit_truth",
            "fit_records",
            "validation_observations",
            "validation_truth",
            "validation_records",
            "embedding_table",
        )
        resolved: dict[str, Path] = {}
        records: dict[str, Mapping[str, Any]] = {}
        for name in names:
            resolved[name], records[name] = _manifest_resource(
                root, resources, name, verify_hash=name not in {"fit_truth", "validation_truth"}
            )
        optional: dict[str, Path | None] = {}
        for name in ("fit_valid_mask", "validation_valid_mask"):
            if name in resources:
                optional[name], records[name] = _manifest_resource(root, resources, name)
            else:
                optional[name] = None
        return _InputPaths(
            fit_observations=resolved["fit_observations"],
            fit_truth=resolved["fit_truth"],
            fit_records=resolved["fit_records"],
            validation_observations=(resolved["validation_observations"],),
            validation_truth=(resolved["validation_truth"],),
            validation_records=(resolved["validation_records"],),
            embedding_table=resolved["embedding_table"],
            fit_valid_mask=optional["fit_valid_mask"],
            validation_valid_mask=optional["validation_valid_mask"],
            resource_records=records,
            fit_manifest_sha256=file_sha256(manifest_path),
        )

    if args.fit_artifact is not None or args.validation_artifact:
        validation_artifacts = tuple(args.validation_artifact or ())
        if args.fit_artifact is None or not validation_artifacts:
            raise ContextualFitError("combined artifact mode requires a fit artifact and at least one validation artifact")
        validation_records = tuple(args.validation_records or ())
        required_artifact = ("fit_records", "embedding_table")
        missing_artifact = [name for name in required_artifact if getattr(args, name) is None]
        if missing_artifact or len(validation_records) != len(validation_artifacts):
            detail = ", ".join(f"--{name.replace('_', '-') }" for name in missing_artifact)
            if len(validation_records) != len(validation_artifacts):
                detail = (detail + ", " if detail else "") + "one --validation-records per validation artifact"
            raise ContextualFitError("combined artifact mode requires: " + detail)
        fit_artifact = _regular_file(args.fit_artifact, label="fit public activation artifact")
        validation_paths = tuple(
            _regular_file(path, label="validation public activation artifact")
            for path in validation_artifacts
        )
        return _InputPaths(
            fit_observations=fit_artifact,
            fit_truth=fit_artifact,
            fit_records=args.fit_records,
            validation_observations=validation_paths,
            validation_truth=validation_paths,
            validation_records=validation_records,
            embedding_table=args.embedding_table,
            fit_valid_mask=fit_artifact,
            validation_valid_mask=None,
            fit_combined_artifact=fit_artifact,
            validation_combined_artifact=validation_paths,
            resource_records=None,
            fit_manifest_sha256=None,
        )

    required = (
        "fit_observations",
        "fit_truth",
        "fit_records",
        "validation_observations",
        "validation_truth",
        "validation_records",
        "embedding_table",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ContextualFitError(
            "explicit tensor mode requires: " + ", ".join(f"--{name.replace('_', '-') }" for name in missing)
        )
    optional_fit_mask = args.fit_valid_mask
    optional_validation_mask = args.validation_valid_mask
    validation_observations = tuple(args.validation_observations) if isinstance(args.validation_observations, list) else (args.validation_observations,)
    validation_truth = tuple(args.validation_truth) if isinstance(args.validation_truth, list) else (args.validation_truth,)
    validation_records = tuple(args.validation_records) if isinstance(args.validation_records, list) else (args.validation_records,)
    if not (len(validation_observations) == len(validation_truth) == len(validation_records)):
        raise ContextualFitError("validation observation/truth/record resource counts must agree")
    return _InputPaths(
        fit_observations=args.fit_observations,
        fit_truth=args.fit_truth,
        fit_records=args.fit_records,
        validation_observations=validation_observations,
        validation_truth=validation_truth,
        validation_records=validation_records,
        embedding_table=args.embedding_table,
        fit_valid_mask=optional_fit_mask,
        validation_valid_mask=optional_validation_mask,
        resource_records=None,
        fit_manifest_sha256=None,
    )


def _record_list(path: Path, *, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _json_load(path, label=label)
    values = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not values:
        raise ContextualFitError(f"{label} has no records")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or not isinstance(value.get("record_id"), str):
            raise ContextualFitError(f"{label} record {index} has no record_id")
        record = dict(value)
        record_id = str(record["record_id"])
        if record_id in seen:
            raise ContextualFitError(f"{label} contains duplicate record {record_id}")
        seen.add(record_id)
        records.append(record)
    return records, _file_record(path, label=label)


def _registration_ids(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    payload = _json_load(path, label="split registration")
    if not isinstance(payload, Mapping):
        raise ContextualFitError("split registration must be an object")
    if payload.get("contains_token_ids") is True or payload.get("contains_source_text") is True:
        raise ContextualFitError("split registration must contain metadata only")
    result: list[tuple[str, ...]] = []
    for name in ("fit", "validation"):
        section = payload.get(name)
        if not isinstance(section, Mapping) or not isinstance(section.get("records"), list):
            raise ContextualFitError(f"split registration {name} records are missing")
        ids: list[str] = []
        for index, record in enumerate(section["records"]):
            if not isinstance(record, Mapping) or not isinstance(record.get("record_id"), str):
                raise ContextualFitError(f"split registration {name} record {index} has no ID")
            ids.append(str(record["record_id"]))
        if len(ids) != len(set(ids)):
            raise ContextualFitError(f"split registration {name} IDs are duplicated")
        result.append(tuple(ids))
    if set(result[0]).intersection(result[1]):
        raise ContextualFitError("split registration fit and validation overlap")
    return result[0], result[1], _file_record(path, label="split registration")


def _record_group(record: Mapping[str, Any]) -> str:
    for key in ("style", "group", "source", "dataset", "domain"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return "public"


def _groups_from_path(path: Path | None, records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if path is None:
        return tuple(_record_group(record) for record in records)
    payload = _json_load(path, label="validation groups")
    values = payload.get("groups") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list):
        raise ContextualFitError("validation groups must be a list or an object with groups")
    groups: list[str] = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            groups.append(value)
        elif isinstance(value, Mapping) and isinstance(value.get("group"), str):
            groups.append(str(value["group"]))
        else:
            raise ContextualFitError(f"validation group {index} is malformed")
    if len(groups) != len(records) or any(not value for value in groups):
        raise ContextualFitError("validation group count does not match validation records")
    return tuple(groups)


def _derive_valid_mask(records: Sequence[Mapping[str, Any]], *, rows: int, positions: int) -> torch.Tensor:
    if len(records) != rows:
        raise ContextualFitError("record count does not match tensor rows")
    mask = torch.zeros((rows, positions), dtype=torch.bool)
    for index, record in enumerate(records):
        raw = record.get("full_token_count")
        if raw is None and record.get("sequence_length") is not None:
            raw = record.get("sequence_length")
        if raw is None and record.get("post_bos_token_count") is not None:
            raw = int(record["post_bos_token_count"]) + 1
        count = positions if raw is None else int(raw)
        if count < 2 or count > positions:
            raise ContextualFitError(
                f"record {record.get('record_id')} has invalid valid length {count} for {positions} positions"
            )
        mask[index, :count] = True
    return mask


def _validate_mask(value: torch.Tensor, *, rows: int, positions: int, label: str) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (rows, positions):
        raise ContextualFitError(f"{label} must have shape [{rows}, {positions}]")
    if value.dtype not in (torch.bool, torch.uint8):
        raise ContextualFitError(f"{label} must be boolean")
    if value.dtype == torch.uint8 and not torch.logical_or(value.eq(0), value.eq(1)).all().item():
        raise ContextualFitError(f"{label} must contain only binary values")
    result = value.to(dtype=torch.bool).contiguous()
    if not result[:, 0].all().item() or not result[:, 1:].any(dim=1).all().item():
        raise ContextualFitError(f"{label} must include BOS and at least one post-BOS position")
    return result


def _validate_observations(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 1 or value.shape[2] <= 0:
        raise ContextualFitError(f"{label} must be [records, positions>1, hidden]")
    if not value.dtype.is_floating_point:
        raise ContextualFitError(f"{label} must be floating point")
    if not torch.isfinite(value).all().item():
        raise ContextualFitError(f"{label} contains non-finite values")
    return value.contiguous()


def _validate_truth(
    value: torch.Tensor,
    *,
    rows: int,
    positions: int,
    mask: torch.Tensor,
    vocabulary_size: int,
    label: str,
) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (rows, positions):
        raise ContextualFitError(f"{label} must have shape [{rows}, {positions}]")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ContextualFitError(f"{label} must be integer")
    result = value.to(dtype=torch.long).contiguous()
    if result[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise ContextualFitError(f"{label} rows must begin with BOS token {BOS_TOKEN_ID}")
    scored_mask = mask.clone()
    scored_mask[:, 0] = False
    scored = result[scored_mask]
    if scored.lt(0).any().item() or scored.ge(vocabulary_size).any().item():
        raise ContextualFitError(f"{label} contains an out-of-range token on a valid position")
    return result


def _load_combined_component(path: Path, *, key: str, label: str) -> torch.Tensor:
    """Read one named tensor from a footing combined activation artifact.

    ``safe_open`` is intentionally used instead of ``load_file`` here.  The
    combined artifact also contains public token IDs and selectors; loading
    only the requested activation or mask component keeps the split/geometry
    gate ahead of the truth component read.
    """

    path = _regular_file(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if key not in keys:
                raise ContextualFitError(f"{label} is missing combined tensor {key!r}")
            return handle.get_tensor(key).contiguous()
    except ContextualFitError:
        raise
    except Exception as exc:  # pragma: no cover - backend-specific
        raise ContextualFitError(f"cannot load {label}: {path}") from exc


def _pad_validation_parts(
    observations: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, int, int], ...], dict[str, Any]]:
    """Right-pad validation styles for one masked causal pass.

    Footing keeps the native geometry of each public style (currently 192 for
    Alpaca and 40 for Pile).  The decoder's causal/padding contract makes a
    right-padding port exact on active positions; the focused integration test
    exercises that equivalence before any fit is allowed to start.
    """

    if not observations or len(observations) != len(masks):
        raise ContextualFitError("validation observation and mask parts are empty or misaligned")
    hidden_sizes = {int(value.shape[2]) for value in observations}
    if len(hidden_sizes) != 1:
        raise ContextualFitError("validation activation parts have different hidden sizes")
    max_positions = max(int(value.shape[1]) for value in observations)
    total_rows = sum(int(value.shape[0]) for value in observations)
    dtype = observations[0].dtype
    padded_observations = torch.zeros(
        (total_rows, max_positions, int(next(iter(hidden_sizes)))), dtype=dtype
    )
    padded_masks = torch.zeros((total_rows, max_positions), dtype=torch.bool)
    native_geometries: list[tuple[int, int, int]] = []
    cursor = 0
    for index, (value, mask) in enumerate(zip(observations, masks)):
        rows, positions, hidden = map(int, value.shape)
        if tuple(mask.shape) != (rows, positions):
            raise ContextualFitError(f"validation mask part {index} does not match observations")
        native_geometries.append((rows, positions, hidden))
        padded_observations[cursor : cursor + rows, :positions] = value
        padded_masks[cursor : cursor + rows, :positions] = mask
        cursor += rows
    did_pad = any(positions != max_positions for _, positions, _ in native_geometries)
    return (
        padded_observations.contiguous(),
        padded_masks.contiguous(),
        tuple(native_geometries),
        {
            "mode": "right_pad_to_max_sequence_for_masked_causal_pass" if did_pad else "none",
            "native_geometries": [list(geometry) for geometry in native_geometries],
            "max_positions": max_positions,
            "masked_padding": bool(did_pad),
            "output_equivalence_check": "tests/test_trr0004_contextual_fit.py::test_combined_validation_adapter_preserves_native_active_outputs",
        },
    )


def _pad_truth_parts(
    truth_parts: Sequence[torch.Tensor],
    *,
    max_positions: int,
) -> torch.Tensor:
    total_rows = sum(int(value.shape[0]) for value in truth_parts)
    padded = torch.zeros((total_rows, max_positions), dtype=torch.long)
    cursor = 0
    for value in truth_parts:
        rows, positions = map(int, value.shape)
        padded[cursor : cursor + rows, :positions] = value
        cursor += rows
    return padded.contiguous()


def _load_context_data(args: argparse.Namespace) -> PublicContextData:
    input_paths = _resolve_input_paths(args)
    registration_ids = _registration_ids(args.registration)
    fit_records, fit_record_file = _record_list(input_paths.fit_records, label="fit record manifest")
    validation_records: list[dict[str, Any]] = []
    validation_record_parts: list[list[dict[str, Any]]] = []
    validation_record_files: list[dict[str, Any]] = []
    for index, path in enumerate(input_paths.validation_records):
        records, record_file = _record_list(path, label=f"validation record manifest {index}")
        validation_record_parts.append(records)
        validation_records.extend(records)
        validation_record_files.append(record_file)
    fit_ids = tuple(str(record["record_id"]) for record in fit_records)
    validation_ids = tuple(str(record["record_id"]) for record in validation_records)
    if len(validation_ids) != len(set(validation_ids)):
        raise ContextualFitError("validation record manifests contain duplicate IDs across styles")
    if fit_ids != registration_ids[0] or validation_ids != registration_ids[1]:
        raise ContextualFitError("tensor record manifests do not exactly match the split registration")
    if set(fit_ids).intersection(validation_ids):
        raise ContextualFitError("fit and validation records overlap before public labels are opened")

    # Observations and masks are public activation resources.  Labels are
    # deliberately loaded only after identity/overlap and all observation
    # geometry checks have passed.
    if input_paths.fit_combined_artifact is not None:
        fit_x = _validate_observations(
            _load_combined_component(
                input_paths.fit_combined_artifact,
                key="activations",
                label="fit combined activations",
            ),
            label="fit observations",
        )
        fit_mask = _validate_mask(
            _load_combined_component(
                input_paths.fit_combined_artifact,
                key="attention_mask",
                label="fit combined attention mask",
            ),
            rows=int(fit_x.shape[0]),
            positions=int(fit_x.shape[1]),
            label="fit attention mask",
        )
    else:
        fit_x = _validate_observations(
            _load_tensor(input_paths.fit_observations, key="activations", label="fit observations"),
            label="fit observations",
        )
        if input_paths.fit_valid_mask is None:
            fit_mask = _derive_valid_mask(
                fit_records, rows=int(fit_x.shape[0]), positions=int(fit_x.shape[1])
            )
        else:
            fit_mask = _validate_mask(
                _load_tensor(input_paths.fit_valid_mask, key="valid_mask", label="fit valid mask"),
                rows=int(fit_x.shape[0]),
                positions=int(fit_x.shape[1]),
                label="fit valid mask",
            )
    if fit_x.shape[0] != len(fit_ids):
        raise ContextualFitError("fit record manifest does not match observation row count")

    validation_observation_parts: list[torch.Tensor] = []
    validation_mask_parts: list[torch.Tensor] = []
    record_cursor = 0
    if input_paths.validation_combined_artifact:
        if len(input_paths.validation_combined_artifact) != len(input_paths.validation_records):
            raise ContextualFitError("combined validation artifact and record resource counts must agree")
        for index, path in enumerate(input_paths.validation_combined_artifact):
            part_records = validation_record_parts[index]
            observation = _validate_observations(
                _load_combined_component(
                    path, key="activations", label=f"validation combined activations {index}"
                ),
                label=f"validation observations {index}",
            )
            mask = _validate_mask(
                _load_combined_component(
                    path, key="attention_mask", label=f"validation combined attention mask {index}"
                ),
                rows=int(observation.shape[0]),
                positions=int(observation.shape[1]),
                label=f"validation attention mask {index}",
            )
            part_count = len(part_records)
            if int(observation.shape[0]) != part_count:
                raise ContextualFitError(f"validation record manifest {index} does not match observation rows")
            validation_observation_parts.append(observation)
            validation_mask_parts.append(mask)
            record_cursor += part_count
    else:
        if len(input_paths.validation_observations) != len(input_paths.validation_records):
            raise ContextualFitError("validation observation and record resource counts must agree")
        for index, path in enumerate(input_paths.validation_observations):
            observation = _validate_observations(
                _load_tensor(path, key="activations", label=f"validation observations {index}"),
                label=f"validation observations {index}",
            )
            part_records = validation_records[record_cursor : record_cursor + int(observation.shape[0])]
            if len(part_records) != int(observation.shape[0]):
                raise ContextualFitError(f"validation record manifest {index} does not match observation rows")
            if input_paths.validation_valid_mask is not None:
                if len(input_paths.validation_observations) != 1:
                    raise ContextualFitError("one validation valid mask cannot serve multiple observation parts")
                mask = _validate_mask(
                    _load_tensor(
                        input_paths.validation_valid_mask,
                        key="valid_mask",
                        label="validation valid mask",
                    ),
                    rows=int(observation.shape[0]),
                    positions=int(observation.shape[1]),
                    label="validation valid mask",
                )
            else:
                mask = _derive_valid_mask(
                    part_records,
                    rows=int(observation.shape[0]),
                    positions=int(observation.shape[1]),
                )
            validation_observation_parts.append(observation)
            validation_mask_parts.append(mask)
            record_cursor += int(observation.shape[0])
    if record_cursor != len(validation_records):
        raise ContextualFitError("validation record manifests do not match all observation parts")
    validation_x, validation_mask, validation_native_geometries, validation_padding = _pad_validation_parts(
        validation_observation_parts, validation_mask_parts
    )
    if int(fit_x.shape[2]) != int(validation_x.shape[2]):
        raise ContextualFitError("fit and validation hidden sizes differ")

    embedding_table = _load_tensor(
        input_paths.embedding_table, key="embeddings", label="public embedding table"
    )
    if embedding_table.ndim != 2 or embedding_table.shape[1] != fit_x.shape[2]:
        raise ContextualFitError("public embedding table geometry does not match hidden size")
    if not embedding_table.dtype.is_floating_point:
        raise ContextualFitError("public embedding table must be floating point")
    validate_runtime_embeddings(
        embedding_table,
        hidden_size=int(fit_x.shape[2]),
        vocab_size=int(embedding_table.shape[0]),
    )
    try:
        validate_normalized_embedding_table(
            embedding_table,
            vocabulary_size=int(embedding_table.shape[0]),
            hidden_size=int(fit_x.shape[2]),
            require_unit_norm=True,
        )
    except Exception as exc:
        raise ContextualFitError("public embedding table must be normalized") from exc

    # Only public auxiliary labels are opened after all split/resource and
    # observation geometry checks.  Combined artifacts are read by component
    # only at this point, after their activation/mask components passed.
    if input_paths.fit_combined_artifact is not None:
        fit_y = _load_combined_component(
            input_paths.fit_combined_artifact, key="token_ids", label="fit combined public labels"
        )
    else:
        fit_y = _load_tensor(input_paths.fit_truth, key="token_ids", label="fit public labels")
    validation_truth_parts: list[torch.Tensor] = []
    if input_paths.validation_combined_artifact:
        for index, path in enumerate(input_paths.validation_combined_artifact):
            validation_truth_parts.append(
                _load_combined_component(
                    path, key="token_ids", label=f"validation combined public labels {index}"
                )
            )
    else:
        validation_truth_parts = [
            _load_tensor(path, key="token_ids", label=f"validation public labels {index}")
            for index, path in enumerate(input_paths.validation_truth)
        ]
    if len(validation_truth_parts) != len(validation_observation_parts):
        raise ContextualFitError("validation truth and observation part counts must agree")
    fit_y = _validate_truth(
        fit_y,
        rows=int(fit_x.shape[0]),
        positions=int(fit_x.shape[1]),
        mask=fit_mask,
        vocabulary_size=int(embedding_table.shape[0]),
        label="fit public labels",
    )
    validation_truth_parts = [
        _validate_truth(
            truth,
            rows=int(observation.shape[0]),
            positions=int(observation.shape[1]),
            mask=mask,
            vocabulary_size=int(embedding_table.shape[0]),
            label=f"validation public labels {index}",
        )
        for index, (truth, observation, mask) in enumerate(
            zip(validation_truth_parts, validation_observation_parts, validation_mask_parts)
        )
    ]
    validation_y = _pad_truth_parts(
        validation_truth_parts, max_positions=int(validation_x.shape[1])
    )

    registration_payload = _json_load(args.registration, label="split registration")
    expected_fit_positions = None
    if isinstance(registration_payload, Mapping):
        fit_section = registration_payload.get("fit")
        if isinstance(fit_section, Mapping):
            nested = fit_section.get("large_nested")
            if isinstance(nested, Mapping) and nested.get("post_bos_positions") is not None:
                expected_fit_positions = int(nested["post_bos_positions"])
    actual_fit_positions = int(fit_mask[:, 1:].sum().item())
    if expected_fit_positions is not None and expected_fit_positions != actual_fit_positions:
        raise ContextualFitError(
            f"fit valid-position count {actual_fit_positions} does not match registration {expected_fit_positions}"
        )

    resource_records: dict[str, Mapping[str, Any]] = {
        "fit_records": fit_record_file,
    }
    if len(validation_record_files) == 1:
        resource_records["validation_records"] = validation_record_files[0]
    else:
        for index, record in enumerate(validation_record_files):
            resource_records[f"validation_records_{index}"] = record
    if input_paths.resource_records is not None:
        resource_records.update(input_paths.resource_records)
        deferred_truths = [("fit_truth", input_paths.fit_truth)]
        if input_paths.validation_truth:
            deferred_truths.append(("validation_truth", input_paths.validation_truth[0]))
        for name, path in deferred_truths:
            entry = resource_records.get(name)
            if entry is not None:
                expected_sha = entry.get("_deferred_expected_sha256")
                actual = _file_record(path, label=name)
                if expected_sha is not None and actual["sha256"] != expected_sha:
                    raise ContextualFitError(f"fit manifest resource {name!r} hash changed")
                expected_bytes = entry.get("bytes")
                if expected_bytes is not None and int(expected_bytes) != actual["bytes"]:
                    raise ContextualFitError(f"fit manifest resource {name!r} byte count changed")
                merged = dict(entry)
                merged.pop("_deferred_expected_sha256", None)
                merged.update(actual)
                resource_records[name] = merged
    elif input_paths.fit_combined_artifact is not None:
        resource_records["fit_combined_artifact"] = _file_record(
            input_paths.fit_combined_artifact, label="fit combined artifact"
        )
        for index, path in enumerate(input_paths.validation_combined_artifact):
            resource_records[f"validation_combined_artifact_{index}"] = _file_record(
                path, label=f"validation combined artifact {index}"
            )
        resource_records["embedding_table"] = _file_record(
            input_paths.embedding_table, label="embedding table"
        )
    else:
        resource_records.update(
            {
                "fit_observations": _file_record(
                    input_paths.fit_observations, label="fit observations"
                ),
                "fit_truth": _file_record(input_paths.fit_truth, label="fit truth"),
                "embedding_table": _file_record(
                    input_paths.embedding_table, label="embedding table"
                ),
            }
        )
        for index, path in enumerate(input_paths.validation_observations):
            resource_records[f"validation_observations_{index}"] = _file_record(
                path, label=f"validation observations {index}"
            )
        for index, path in enumerate(input_paths.validation_truth):
            resource_records[f"validation_truth_{index}"] = _file_record(
                path, label=f"validation truth {index}"
            )
        if input_paths.fit_valid_mask is not None:
            resource_records["fit_valid_mask"] = _file_record(
                input_paths.fit_valid_mask, label="fit valid mask"
            )
        if input_paths.validation_valid_mask is not None:
            resource_records["validation_valid_mask"] = _file_record(
                input_paths.validation_valid_mask, label="validation valid mask"
            )
    return PublicContextData(
        fit_observations=fit_x,
        fit_truth=fit_y,
        fit_valid_mask=fit_mask,
        validation_observations=validation_x,
        validation_truth=validation_y,
        validation_valid_mask=validation_mask,
        embedding_table=embedding_table,
        fit_record_ids=fit_ids,
        validation_record_ids=validation_ids,
        validation_groups=_groups_from_path(args.validation_groups, validation_records),
        validation_native_geometries=validation_native_geometries,
        validation_padding=validation_padding,
        resource_records=resource_records,
        registration_sha256=_file_record(args.registration, label="split registration")["sha256"],
        fit_manifest_sha256=input_paths.fit_manifest_sha256,
    )

def checkpoint_steps(steps: int, *, validation_every: int = DEFAULT_VALIDATION_EVERY) -> tuple[int, ...]:
    """Return the common early and then regular public-validation checkpoints."""

    if steps <= 0 or validation_every <= 0:
        raise ContextualFitError("checkpoint schedule settings must be positive")
    points = {0, 25, 50, 75, 100, 150, 200}
    points.update(range(300, steps + 1, validation_every))
    points.add(int(steps))
    return tuple(sorted(point for point in points if point <= steps))


def build_position_schedule(
    valid_mask: torch.Tensor,
    *,
    steps: int,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    position_budget: int = DEFAULT_POSITION_BUDGET,
    seed: int = DEFAULT_SEED,
) -> PositionSchedule:
    """Precompute one reproducible batch/position schedule for both methods."""

    if valid_mask.ndim != 2 or valid_mask.shape[0] < record_batch_size or valid_mask.shape[1] <= 1:
        raise ContextualFitError("valid mask is too small for the declared record batch")
    if steps <= 0 or record_batch_size <= 0 or position_budget <= 0:
        raise ContextualFitError("schedule settings must be positive")
    if position_budget > DEFAULT_POSITION_BUDGET:
        raise ContextualFitError("position budget cannot exceed the registered 512-row cap")
    mask = valid_mask.detach().cpu().to(dtype=torch.bool)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    record_count, sequence = map(int, mask.shape)
    permutations: list[torch.Tensor] = []
    cursor = 0
    batches: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []
    for _ in range(steps):
        if not permutations or cursor + record_batch_size > record_count:
            permutations.append(torch.randperm(record_count, generator=generator))
            cursor = 0
        record_indices = permutations[-1][cursor : cursor + record_batch_size].clone()
        cursor += record_batch_size
        batch_mask = torch.zeros((record_batch_size, sequence), dtype=torch.bool)
        candidate_lists: list[torch.Tensor] = []
        for batch_row, record_index in enumerate(record_indices.tolist()):
            del batch_row
            candidates = torch.nonzero(mask[record_index], as_tuple=False).flatten()
            candidates = candidates[candidates.ne(0)]
            if candidates.numel() == 0:
                raise ContextualFitError("a sampled record has no post-BOS valid positions")
            order = torch.randperm(int(candidates.numel()), generator=generator)
            candidate_lists.append(candidates[order])
        # The budget is total selected positions for the complete record batch,
        # rather than a per-record multiplier.  Give every record one position
        # when K permits, then fill the remaining rows from one shared random
        # pool.  This keeps both methods on exactly the same <=512 projection
        # rows and avoids accidentally turning 8 x 512 into a 4096-row fit.
        all_pairs = torch.cat(
            [
                torch.stack(
                    [
                        torch.full_like(candidates, batch_row),
                        candidates,
                    ],
                    dim=1,
                )
                for batch_row, candidates in enumerate(candidate_lists)
            ],
            dim=0,
        )
        mandatory_count = min(record_batch_size, position_budget)
        mandatory = (
            torch.tensor(
                [[batch_row, int(candidate_lists[batch_row][0])] for batch_row in range(mandatory_count)],
                dtype=torch.long,
            )
            if mandatory_count
            else torch.empty((0, 2), dtype=torch.long)
        )
        if mandatory.numel():
            all_keys = all_pairs[:, 0] * sequence + all_pairs[:, 1]
            mandatory_keys = mandatory[:, 0] * sequence + mandatory[:, 1]
            remaining = all_pairs[~torch.isin(all_keys, mandatory_keys)]
        else:
            remaining = all_pairs
        remaining_order = torch.randperm(int(remaining.shape[0]), generator=generator)
        extra_count = min(position_budget - mandatory_count, int(remaining.shape[0]))
        chosen_pairs = torch.cat([mandatory, remaining[remaining_order[:extra_count]]], dim=0)
        batch_mask[chosen_pairs[:, 0], chosen_pairs[:, 1]] = True
        batches.append(record_indices)
        selected.append(batch_mask)
    return PositionSchedule(
        record_indices=torch.stack(batches).contiguous(),
        selected_mask=torch.stack(selected).contiguous(),
        seed=int(seed),
        position_budget=int(position_budget),
        record_batch_size=int(record_batch_size),
    )


def schedule_digest(schedule: PositionSchedule) -> str:
    return _canonical_hash(
        {
            "record_indices": _tensor_digest(schedule.record_indices),
            "selected_mask": _tensor_digest(schedule.selected_mask),
            "seed": schedule.seed,
            "position_budget": schedule.position_budget,
            "record_batch_size": schedule.record_batch_size,
        }
    )


def _save_schedule(
    path: Path,
    *,
    main: PositionSchedule,
    subset: PositionSchedule,
    fit_record_ids: Sequence[str],
) -> dict[str, Any]:
    tensors = {
        "main.record_indices": main.record_indices.to(torch.int64),
        "main.selected_mask": main.selected_mask,
        "subset.record_indices": subset.record_indices.to(torch.int64),
        "subset.selected_mask": subset.selected_mask,
    }
    metadata = {
        "schema": "token-reconstruction.trr0004-position-schedule.v1",
        "main_schedule_sha256": schedule_digest(main),
        "subset_schedule_sha256": schedule_digest(subset),
        "fit_record_order_sha256": _canonical_hash(list(fit_record_ids)),
        "main_seed": str(main.seed),
        "subset_seed": str(subset.seed),
        "record_batch_size": str(main.record_batch_size),
        "position_budget": str(main.position_budget),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ContextualFitError(f"schedule artifact is create-only: {path}")
    save_file(tensors, str(path), metadata=metadata)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "main": {
            "steps": main.steps,
            "seed": main.seed,
            "schedule_sha256": schedule_digest(main),
            "record_indices_shape": list(main.record_indices.shape),
            "selected_mask_shape": list(main.selected_mask.shape),
        },
        "subset": {
            "steps": subset.steps,
            "seed": subset.seed,
            "schedule_sha256": schedule_digest(subset),
            "record_indices_shape": list(subset.record_indices.shape),
            "selected_mask_shape": list(subset.selected_mask.shape),
        },
    }


def _load_base_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = _regular_file(path, label="fresh affine base state")
    if path.suffix == ".safetensors":
        try:
            raw = load_file(str(path), device="cpu")
        except Exception as exc:  # pragma: no cover
            raise ContextualFitError(f"cannot load affine state: {path}") from exc
    else:
        try:
            raw_value = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:  # pragma: no cover
            raise ContextualFitError(f"cannot load affine state: {path}") from exc
        if isinstance(raw_value, Mapping) and isinstance(raw_value.get("state_dict"), Mapping):
            raw = raw_value["state_dict"]
        elif isinstance(raw_value, Mapping) and isinstance(raw_value.get("sd"), Mapping):
            raw = raw_value["sd"]
        elif isinstance(raw_value, Mapping):
            raw = raw_value
        else:
            raise ContextualFitError("affine base state must be a tensor mapping")
    if set(raw) != {"W", "b", "s"}:
        raise ContextualFitError("fresh affine base must contain exactly W, b, and s; vocabulary bias is excluded")
    state: dict[str, torch.Tensor] = {}
    for key in ("W", "b", "s"):
        value = raw[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ContextualFitError(f"affine base {key} must be float32")
        if not torch.isfinite(value).all().item():
            raise ContextualFitError(f"affine base {key} is non-finite")
        state[key] = value.detach().cpu().contiguous().clone()
    try:
        base = FrozenAffineBase.from_state_dict(state)
    except CausalDecoderExtensionError as exc:
        raise ContextualFitError("fresh affine base geometry is invalid") from exc
    state_record = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "state_tensor_sha256": {key: _tensor_digest(value) for key, value in state.items()},
        "hidden_size": base.hidden_size,
        "state_bytes": sum(int(value.numel()) * value.element_size() for value in state.values()),
        "method_id": "historical_affine_ce_no_vocab_bias",
        "vocab_bias": False,
    }
    return state, state_record


def _choose_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise ContextualFitError("CUDA was requested but is unavailable")
    if raw not in ("cpu", "cuda"):
        raise ContextualFitError(f"unsupported device {raw}")
    return torch.device(raw)


def _resource_preflight(
    data: PublicContextData,
    base_state: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    record_batch_size: int,
    position_budget: int,
    minimum_free_gib: float,
    maximum_gpu_reserved_gib: float,
    maximum_host_rss_gib: float,
) -> dict[str, Any]:
    hidden = int(data.hidden_size)
    sequence = int(data.fit_observations.shape[1])
    vocab = int(data.vocabulary_size)
    embedding_bytes = int(data.embedding_table.numel()) * data.embedding_table.element_size()
    state_bytes = sum(int(value.numel()) * value.element_size() for value in base_state.values())
    batch_hidden_bytes = record_batch_size * sequence * hidden * 4
    selected_logits_bytes = position_budget * vocab * 4
    # This is a conservative planning estimate: table plus forward hidden,
    # attention/MLP workspace, two logits copies for CE autograd, and state.
    estimated_peak = embedding_bytes + 4 * selected_logits_bytes + 8 * batch_hidden_bytes + state_bytes + 2 * 4_300_000
    result: dict[str, Any] = {
        "device": str(device),
        "hidden_size": hidden,
        "sequence_length": sequence,
        "vocabulary_size": vocab,
        "record_batch_size": record_batch_size,
        "position_budget": position_budget,
        "embedding_bytes": embedding_bytes,
        "affine_base_state_bytes": state_bytes,
        "batch_hidden_bytes_float32": batch_hidden_bytes,
        "selected_logits_bytes_float32": selected_logits_bytes,
        "estimated_peak_bytes_conservative": int(estimated_peak),
        "estimated_peak_gib_conservative": float(estimated_peak / 2**30),
        "maximum_gpu_reserved_bytes": int(maximum_gpu_reserved_gib * 2**30),
        "maximum_host_rss_bytes": int(maximum_host_rss_gib * 2**30),
        "qualification_required": True,
        "estimate_is_measured": False,
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(
            {
                "cuda_free_bytes_before_load": int(free),
                "cuda_total_bytes": int(total),
                "minimum_free_bytes": int(minimum_free_gib * 2**30),
            }
        )
        if free < int(minimum_free_gib * 2**30):
            raise ContextualFitError(
                f"CUDA preflight requires {minimum_free_gib:.1f} GiB free; observed {free / 2**30:.2f} GiB"
            )
        if estimated_peak >= free:
            raise ContextualFitError(
                "conservative contextual-fit peak estimate exceeds currently free CUDA memory"
            )
    return result


def _resource_limits(args: argparse.Namespace) -> dict[str, int]:
    gib = 2**30
    return {
        "minimum_free_gpu_bytes": int(args.minimum_free_gib * gib),
        "maximum_gpu_reserved_bytes": int(args.maximum_gpu_reserved_gib * gib),
        "maximum_host_rss_bytes": int(args.maximum_host_rss_gib * gib),
    }


def _resource_guard(
    args: argparse.Namespace,
    device: torch.device,
    *,
    deadline: float | None = None,
    stage: str,
) -> dict[str, int | None | str]:
    """Fail closed on wall time, host RSS, and live device memory limits.

    This uses PyTorch's live allocator/device queries rather than launching a
    per-step nvidia-smi process. The caller may invoke it at step boundaries
    and between major phases without retaining every observation.
    """

    if deadline is not None and time.perf_counter() >= deadline:
        raise ContextualFitError(f"contextual fit exceeded its wall-time guard at {stage}")
    limits = _resource_limits(args)
    # Linux reports ru_maxrss in KiB; this runner is qualified on Linux.
    rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if rss_bytes > limits["maximum_host_rss_bytes"]:
        raise ContextualFitError(
            f"host RSS guard exceeded at {stage}: {rss_bytes} > {limits['maximum_host_rss_bytes']}"
        )
    result: dict[str, int | None | str] = {
        "stage": stage,
        "host_max_rss_bytes": rss_bytes,
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
        "cuda_reserved_bytes": None,
    }
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        reserved_bytes = int(torch.cuda.memory_reserved(device))
        result.update(
            {
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                "cuda_reserved_bytes": reserved_bytes,
            }
        )
        if free_bytes < limits["minimum_free_gpu_bytes"]:
            raise ContextualFitError(
                f"GPU free-memory guard exceeded at {stage}: {free_bytes} < {limits['minimum_free_gpu_bytes']}"
            )
        if reserved_bytes > limits["maximum_gpu_reserved_bytes"]:
            raise ContextualFitError(
                f"GPU reserved-memory guard exceeded at {stage}: {reserved_bytes} > {limits['maximum_gpu_reserved_bytes']}"
            )
    return result


def _runtime_memory(device: torch.device) -> dict[str, int]:
    result = {
        "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return result


def _state_cpu(model: CausalResidualDecoder) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous().clone()
        for key, value in model.state_dict().items()
    }


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    return _canonical_hash({key: _tensor_digest(value) for key, value in sorted(state.items())})


def _valid_scored_mask(mask: torch.Tensor) -> torch.Tensor:
    result = mask.to(dtype=torch.bool).clone()
    result[:, 0] = False
    return result


def _evaluate_dataset(
    model: CausalResidualDecoder,
    observations: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    groups: Sequence[str],
    *,
    device: torch.device,
    record_batch_size: int,
    projection_budget: int,
) -> dict[str, Any]:
    if len(groups) != observations.shape[0]:
        raise ContextualFitError("evaluation groups do not match record count")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_rows = 0
    exact_records = 0
    projection_calls = 0
    projected_rows = 0
    max_projection_rows = 0
    group_total: dict[str, int] = {}
    group_correct: dict[str, int] = {}
    record_totals: list[int] = []
    record_correct: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, int(observations.shape[0]), record_batch_size):
            stop = min(start + record_batch_size, int(observations.shape[0]))
            activation = observations[start:stop].to(device=device, dtype=torch.float32)
            valid = valid_mask[start:stop].to(device=device, dtype=torch.bool)
            labels = truth[start:stop].to(device=device, dtype=torch.long)
            selected = _valid_scored_mask(valid)
            if not selected.any().item():
                raise ContextualFitError("evaluation batch has no post-BOS positions")
            projected_hidden = model.projected_hidden(activation, valid)
            coords = selected.nonzero(as_tuple=False)
            local_totals = torch.bincount(coords[:, 0], minlength=stop - start).to(device="cpu")
            local_correct = torch.zeros(stop - start, dtype=torch.long)
            for chunk_start in range(0, int(coords.shape[0]), projection_budget):
                chunk = coords[chunk_start : chunk_start + projection_budget]
                chunk_mask = torch.zeros_like(selected)
                chunk_mask[chunk[:, 0], chunk[:, 1]] = True
                logits = model.logits_from_projected_hidden(
                    projected_hidden, chunk_mask, embedding_table
                )
                target = labels[chunk[:, 0], chunk[:, 1]]
                total_loss += float(F.cross_entropy(logits, target, reduction="sum").cpu())
                predictions = logits.argmax(dim=-1)
                correct = predictions.eq(target)
                total_correct += int(correct.sum().cpu())
                total_rows += int(target.numel())
                local_correct += torch.bincount(
                    chunk[:, 0].to(device="cpu"),
                    weights=correct.to(device="cpu", dtype=torch.float32),
                    minlength=stop - start,
                ).to(dtype=torch.long)
                projection_calls += 1
                projected_rows += int(target.numel())
                max_projection_rows = max(max_projection_rows, int(target.numel()))
            for local_index in range(stop - start):
                rows = int(local_totals[local_index])
                correct = int(local_correct[local_index])
                record_totals.append(rows)
                record_correct.append(correct)
                if correct == rows:
                    exact_records += 1
                group = str(groups[start + local_index])
                group_total[group] = group_total.get(group, 0) + rows
                group_correct[group] = group_correct.get(group, 0) + correct
    if total_rows <= 0:
        raise ContextualFitError("evaluation produced no scored rows")
    group_accuracy = {
        group: group_correct[group] / group_total[group]
        for group in sorted(group_total)
    }
    style_balanced = sum(group_accuracy.values()) / len(group_accuracy)
    return {
        "loss": total_loss / total_rows,
        "token_accuracy": total_correct / total_rows,
        "correct_tokens": total_correct,
        "token_rows": total_rows,
        "exact_records": exact_records,
        "record_count": len(record_totals),
        "group_token_accuracy": group_accuracy,
        "style_balanced_token_accuracy": style_balanced,
        "projection_calls": projection_calls,
        "projection_rows": projected_rows,
        "max_projection_rows": max_projection_rows,
        "evaluation_seconds": time.perf_counter() - started,
    }


def _train_batch(
    model: CausalResidualDecoder,
    observations: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    record_indices: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float,
) -> dict[str, float | int]:
    model.train()
    activation = observations.index_select(0, record_indices).to(device=device, dtype=torch.float32)
    valid = valid_mask.index_select(0, record_indices).to(device=device, dtype=torch.bool)
    labels = truth.index_select(0, record_indices).to(device=device, dtype=torch.long)
    selected = selected_mask.to(device=device, dtype=torch.bool)
    if (selected & ~valid).any().item() or (selected[:, 0]).any().item():
        raise ContextualFitError("sampler selected an invalid or BOS position")
    projected_hidden = model.projected_hidden(activation, valid)
    logits = model.logits_from_projected_hidden(projected_hidden, selected, embedding_table)
    target = labels[selected]
    if target.numel() == 0 or target.numel() > DEFAULT_POSITION_BUDGET:
        raise ContextualFitError("training projection row budget changed")
    loss = F.cross_entropy(logits, target)
    if not torch.isfinite(loss).item():
        raise ContextualFitError("contextual fit loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = list(model.added_path.parameters())
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise ContextualFitError("contextual fit gradient is non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, gradient_clip_norm, error_if_nonfinite=True
    )
    optimizer.step()
    for parameter in parameters:
        if not torch.isfinite(parameter).all().item():
            raise ContextualFitError("contextual fit parameter became non-finite")
    correct = int(logits.detach().argmax(dim=-1).eq(target).sum().cpu())
    return {
        "loss": float(loss.detach().cpu()),
        "token_accuracy": correct / int(target.numel()),
        "correct_tokens": correct,
        "token_rows": int(target.numel()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }


def _save_state(path: Path, state: Mapping[str, torch.Tensor], *, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ContextualFitError(f"decoder state is create-only: {path}")
    tensors = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    save_file(tensors, str(path), metadata={str(k): str(v) for k, v in metadata.items()})
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "state_sha256": _state_digest(tensors),
        "state_bytes": sum(int(value.numel()) * value.element_size() for value in tensors.values()),
        "tensor_sha256": {key: _tensor_digest(value) for key, value in tensors.items()},
    }


def _train_one(
    method_id: str,
    *,
    base_state: Mapping[str, torch.Tensor],
    observations: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]] | None,
    schedule: PositionSchedule,
    device: torch.device,
    seed: int,
    steps: int,
    validation_every: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    output_dir: Path,
    run_deadline: float | None,
    resource_guard: Callable[[str], Mapping[str, Any]] | None,
    mode: str,
    evaluation_label: str = "validation",
) -> dict[str, Any]:
    if steps != schedule.steps:
        raise ContextualFitError("training steps and schedule length differ")
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = build_causal_extension(FrozenAffineBase.from_state_dict(base_state), method_id).to(device)
    model.train()
    trainable = list(model.added_path.parameters())
    if not trainable or any(not parameter.requires_grad for parameter in trainable):
        raise ContextualFitError("added path has no trainable parameters")
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise ContextualFitError("affine base was not frozen")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    curve: list[dict[str, Any]] = []
    checkpoints = set(checkpoint_steps(steps, validation_every=validation_every))
    best_metric = -float("inf")
    best_step = 0
    best_state = _state_cpu(model)
    total_validation_seconds = 0.0
    total_projection_calls = 0
    total_projection_rows = 0
    max_projection_rows = 0
    final_fit_metrics: dict[str, Any] | None = None
    final_fit_evaluation_seconds = 0.0
    started = time.perf_counter()

    def record_curve(step: int, train_point: Mapping[str, Any] | None) -> None:
        nonlocal best_metric, best_step, best_state
        point: dict[str, Any] = {
            "step": int(step),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_batch": dict(train_point) if train_point is not None else None,
        }
        if validation is not None:
            val_x, val_y, val_mask, val_groups = validation
            val_started = time.perf_counter()
            metrics = _evaluate_dataset(
                model,
                val_x,
                val_y,
                val_mask,
                embedding_table,
                val_groups,
                device=device,
                record_batch_size=schedule.record_batch_size,
                projection_budget=schedule.position_budget,
            )
            nonlocal_total = metrics["evaluation_seconds"]
            point[evaluation_label] = metrics
            nonlocal total_validation_seconds, total_projection_calls, total_projection_rows, max_projection_rows
            total_validation_seconds += float(nonlocal_total)
            total_projection_calls += int(metrics["projection_calls"])
            total_projection_rows += int(metrics["projection_rows"])
            max_projection_rows = max(max_projection_rows, int(metrics["max_projection_rows"]))
            metric = float(metrics["style_balanced_token_accuracy"])
            # The unchanged affine base is a separate control.  A contextual
            # extension must be selected from a nonzero training checkpoint,
            # even when every learned checkpoint is below the step-0 control.
            if mode == "main" and step > 0 and metric > best_metric:
                best_metric = metric
                best_step = int(step)
                best_state = _state_cpu(model)
            point["validation_wall_seconds"] = time.perf_counter() - val_started
        curve.append(point)

    record_curve(0, None)
    if resource_guard is not None:
        resource_guard(f"{method_id}:{mode}:after_step_0_validation")
    for step_index in range(steps):
        if run_deadline is not None and time.perf_counter() >= run_deadline:
            raise ContextualFitError("contextual fit exceeded its wall-time guard")
        if resource_guard is not None:
            resource_guard(f"{method_id}:{mode}:before_step_{step_index + 1}")
        train_point = _train_batch(
            model,
            observations,
            truth,
            valid_mask,
            embedding_table,
            schedule.record_indices[step_index],
            schedule.selected_mask[step_index],
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
        )
        scheduler.step()
        step = step_index + 1
        if step in checkpoints:
            record_curve(step, train_point)
        if resource_guard is not None:
            resource_guard(f"{method_id}:{mode}:after_step_{step}")
    if mode == "main":
        # This is a selection-independent capacity diagnostic on every public
        # fit position after the last optimizer update. It is deliberately
        # kept separate from the changing minibatch curve and validation cost.
        if resource_guard is not None:
            resource_guard(f"{method_id}:{mode}:before_final_fit_evaluation")
        final_fit_started = time.perf_counter()
        final_fit_metrics = _evaluate_dataset(
            model,
            observations,
            truth,
            valid_mask,
            embedding_table,
            tuple("fit_public" for _ in range(int(observations.shape[0]))),
            device=device,
            record_batch_size=schedule.record_batch_size,
            projection_budget=schedule.position_budget,
        )
        final_fit_evaluation_seconds = time.perf_counter() - final_fit_started
        if resource_guard is not None:
            resource_guard(f"{method_id}:{mode}:after_final_fit_evaluation")
    fit_and_validation_seconds = time.perf_counter() - started
    if validation is None:
        best_step = steps
        best_state = _state_cpu(model)
        best_metric = float("nan")
    curve_path = output_dir / f"{mode}_curve.json"
    _write_json(curve_path, {"schema": "token-reconstruction.trr0004-learning-curve.v1", "method_id": method_id, "mode": mode, "curve": curve})
    state_path = output_dir / ("selected.safetensors" if mode == "main" else "subset_final.safetensors")
    state_record = _save_state(
        state_path,
        best_state if mode == "main" else _state_cpu(model),
        metadata={
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "method_id": method_id,
            "mode": mode,
            "selected_step": best_step if mode == "main" else steps,
            "selection_metric": "public_validation_style_balanced_token_accuracy" if mode == "main" else "final_public_fit_subset_state",
        },
    )
    return {
        "method_id": method_id,
        "mode": mode,
        "seed": int(seed),
        "steps": int(steps),
        "checkpoint_steps": sorted(checkpoints),
        "record_batch_size": schedule.record_batch_size,
        "position_budget": schedule.position_budget,
        "schedule_sha256": schedule_digest(schedule),
        "curve": {
            "path": str(curve_path),
            "bytes": int(curve_path.stat().st_size),
            "sha256": file_sha256(curve_path),
            "points": len(curve),
        },
        "selected_step": int(best_step if mode == "main" else steps),
        "best_validation_style_balanced_token_accuracy": (
            None if mode != "main" or math.isnan(best_metric) else best_metric
        ),
        "final_fit_subset_token_accuracy": (
            curve[-1].get("fit_subset", {}).get("token_accuracy")
            if mode == "subset" and curve and isinstance(curve[-1].get("fit_subset"), Mapping)
            else None
        ),
        "state": state_record,
        "added_parameters": model.trainable_parameters(),
        "added_state_bytes": sum(
            int(value.numel()) * value.element_size() for value in model.added_path.state_dict().values()
        ),
        "training_seconds_including_validation": fit_and_validation_seconds,
        "final_fit_evaluation": final_fit_metrics,
        "final_fit_evaluation_seconds": final_fit_evaluation_seconds,
        "final_fit_evaluation_selection_independent": mode == "main",
        "final_fit_projection_calls": (
            None if final_fit_metrics is None else int(final_fit_metrics["projection_calls"])
        ),
        "final_fit_projection_rows": (
            None if final_fit_metrics is None else int(final_fit_metrics["projection_rows"])
        ),
        "final_fit_max_projection_rows": (
            None if final_fit_metrics is None else int(final_fit_metrics["max_projection_rows"])
        ),
        "total_run_seconds": time.perf_counter() - started,
        "validation_seconds": total_validation_seconds,
        "validation_projection_calls": total_projection_calls,
        "validation_projection_rows": total_projection_rows,
        "validation_max_projection_rows": max_projection_rows,
        "runtime_components": {
            "public_prefix_calls": 0,
            "candidate_simulations": 0,
            "a2_fallback": False,
            "teacher_prefix": False,
            "full_vocab_projection_during_training": "selected_rows_only",
        },
        "peak_memory": _runtime_memory(device),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path)
    parser.add_argument("--fit-artifact", type=Path)
    parser.add_argument("--validation-artifact", type=Path, action="append")
    parser.add_argument("--fit-observations", type=Path)
    parser.add_argument("--fit-truth", type=Path)
    parser.add_argument("--fit-records", type=Path)
    parser.add_argument("--fit-valid-mask", type=Path)
    parser.add_argument("--validation-observations", type=Path, action="append")
    parser.add_argument("--validation-truth", type=Path, action="append")
    parser.add_argument("--validation-records", type=Path, action="append")
    parser.add_argument("--validation-valid-mask", type=Path)
    parser.add_argument("--validation-groups", type=Path)
    parser.add_argument("--embedding-table", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--subset-steps", type=int, default=DEFAULT_SUBSET_STEPS)
    parser.add_argument("--subset-records", type=int, default=8)
    parser.add_argument("--record-batch-size", type=int, default=DEFAULT_RECORD_BATCH_SIZE)
    parser.add_argument("--position-budget", type=int, default=DEFAULT_POSITION_BUDGET)
    parser.add_argument("--validation-every", type=int, default=DEFAULT_VALIDATION_EVERY)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_GIB)
    parser.add_argument("--maximum-gpu-reserved-gib", type=float, default=DEFAULT_MAXIMUM_GPU_RESERVED_GIB)
    parser.add_argument("--maximum-host-rss-gib", type=float, default=DEFAULT_MAXIMUM_HOST_RSS_GIB)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.subset_steps <= 0:
        raise ContextualFitError("steps and subset-steps must be positive")
    if args.record_batch_size != DEFAULT_RECORD_BATCH_SIZE:
        raise ContextualFitError("the registered comparison requires record-batch-size 8")
    if args.position_budget <= 0 or args.position_budget > DEFAULT_POSITION_BUDGET:
        raise ContextualFitError("position-budget must be between 1 and 512")
    if args.subset_records <= 0:
        raise ContextualFitError("subset-records must be positive")
    if args.validation_every <= 0:
        raise ContextualFitError("validation-every must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ContextualFitError("learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ContextualFitError("weight-decay must be finite and non-negative")
    if not math.isfinite(args.gradient_clip_norm) or args.gradient_clip_norm <= 0:
        raise ContextualFitError("gradient-clip-norm must be finite and positive")
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib <= 0:
        raise ContextualFitError("minimum-free-gib must be finite and positive")
    if not math.isfinite(args.maximum_gpu_reserved_gib) or args.maximum_gpu_reserved_gib <= 0:
        raise ContextualFitError("maximum-gpu-reserved-gib must be finite and positive")
    if not math.isfinite(args.maximum_host_rss_gib) or args.maximum_host_rss_gib <= 0:
        raise ContextualFitError("maximum-host-rss-gib must be finite and positive")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ContextualFitError("max-seconds must be finite and positive")


def run_fit(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    started_utc = _utc_now()
    source_start = _source_records()
    git_start = _git_commit()
    data = _load_context_data(args)
    base_state, base_record = _load_base_state(args.base_state)
    if data.hidden_size != int(base_state["W"].shape[0]):
        raise ContextualFitError("activation hidden size and affine base hidden size differ")
    device = _choose_device(args.device)
    preflight = _resource_preflight(
        data,
        base_state,
        device=device,
        record_batch_size=args.record_batch_size,
        position_budget=args.position_budget,
        minimum_free_gib=args.minimum_free_gib,
        maximum_gpu_reserved_gib=args.maximum_gpu_reserved_gib,
        maximum_host_rss_gib=args.maximum_host_rss_gib,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ContextualFitError(f"output root is create-only: {output_root}")
    output_root.mkdir(parents=True)
    deadline = time.perf_counter() + args.max_seconds
    guard_summary: dict[str, Any] = {
        "checks": 0,
        "first": None,
        "last": None,
        "minimum_cuda_free_bytes": None,
        "maximum_cuda_reserved_bytes": None,
        "maximum_host_rss_bytes": 0,
        "limits": _resource_limits(args),
    }

    def check_resources(stage: str) -> Mapping[str, Any]:
        observation = _resource_guard(args, device, deadline=deadline, stage=stage)
        guard_summary["checks"] += 1
        if guard_summary["first"] is None:
            guard_summary["first"] = dict(observation)
        guard_summary["last"] = dict(observation)
        host_rss = int(observation["host_max_rss_bytes"])
        guard_summary["maximum_host_rss_bytes"] = max(
            int(guard_summary["maximum_host_rss_bytes"]), host_rss
        )
        free = observation.get("cuda_free_bytes")
        if free is not None:
            previous_free = guard_summary["minimum_cuda_free_bytes"]
            guard_summary["minimum_cuda_free_bytes"] = (
                int(free) if previous_free is None else min(int(previous_free), int(free))
            )
        reserved = observation.get("cuda_reserved_bytes")
        if reserved is not None:
            previous_reserved = guard_summary["maximum_cuda_reserved_bytes"]
            guard_summary["maximum_cuda_reserved_bytes"] = (
                int(reserved)
                if previous_reserved is None
                else max(int(previous_reserved), int(reserved))
            )
        return observation

    try:
        check_resources("after_preflight_before_embedding")
        embedding_transfer_started = time.perf_counter()
        embedding_table = data.embedding_table.to(device=device)
        embedding_transfer_seconds = time.perf_counter() - embedding_transfer_started
        check_resources("after_embedding_transfer")
        main_schedule = build_position_schedule(
            data.fit_valid_mask,
            steps=args.steps,
            record_batch_size=args.record_batch_size,
            position_budget=args.position_budget,
            seed=args.seed,
        )
        if args.subset_records > data.fit_observations.shape[0]:
            raise ContextualFitError("subset-records exceeds public fitting records")
        subset_schedule = build_position_schedule(
            data.fit_valid_mask[: args.subset_records],
            steps=args.subset_steps,
            record_batch_size=args.record_batch_size,
            position_budget=args.position_budget,
            seed=args.seed + 1,
        )
        check_resources("before_first_method")
        schedule_record = _save_schedule(
            output_root / "position_schedule.safetensors",
            main=main_schedule,
            subset=subset_schedule,
            fit_record_ids=data.fit_record_ids,
        )
        methods: dict[str, Any] = {}
        validation = (
            data.validation_observations,
            data.validation_truth,
            data.validation_valid_mask,
            data.validation_groups,
        )
        for method_id in EXTENSION_METHODS:
            method_dir = output_root / method_id
            main_result = _train_one(
                method_id,
                base_state=base_state,
                observations=data.fit_observations,
                truth=data.fit_truth,
                valid_mask=data.fit_valid_mask,
                embedding_table=embedding_table,
                validation=validation,
                schedule=main_schedule,
                device=device,
                seed=args.seed,
                steps=args.steps,
                validation_every=args.validation_every,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                output_dir=method_dir,
                run_deadline=deadline,
                resource_guard=check_resources,
                mode="main",
            )
            subset_result = _train_one(
                method_id,
                base_state=base_state,
                observations=data.fit_observations[: args.subset_records],
                truth=data.fit_truth[: args.subset_records],
                valid_mask=data.fit_valid_mask[: args.subset_records],
                embedding_table=embedding_table,
                validation=(
                    data.fit_observations[: args.subset_records],
                    data.fit_truth[: args.subset_records],
                    data.fit_valid_mask[: args.subset_records],
                    tuple("fit_subset" for _ in range(args.subset_records)),
                ),
                schedule=subset_schedule,
                device=device,
                seed=args.seed,
                steps=args.subset_steps,
                validation_every=args.validation_every,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                output_dir=method_dir,
                run_deadline=deadline,
                resource_guard=check_resources,
                mode="subset",
                evaluation_label="fit_subset",
            )
            methods[method_id] = {"main": main_result, "subset": subset_result}
            check_resources(f"after_{method_id}_subset")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        check_resources("after_all_methods")
        source_end = _source_records()
        git_end = _git_commit()
        source_unchanged = source_start == source_end and git_start == git_end
        if not source_unchanged:
            raise ContextualFitError("executed source or git HEAD changed during contextual fit")
        evidence = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "status": "complete",
            "method_ids": list(EXTENSION_METHODS),
            "interpretation": "exploratory public-data contextual extension fit; not independent confirmation",
            "public_labels_only": True,
            "target_weights_accessed": False,
            "current_evaluator_truth_accessed": False,
            "public_prefix_calls": 0,
            "candidate_simulations": 0,
            "a2_fallback": False,
            "base_state": base_record,
            "data": {
                "registration": {
                    "path": str(args.registration.expanduser().resolve()),
                    "sha256": data.registration_sha256,
                },
                "fit_manifest_sha256": data.fit_manifest_sha256,
                "resources": dict(data.resource_records),
                "fit_record_count": len(data.fit_record_ids),
                "validation_record_count": len(data.validation_record_ids),
                "fit_record_order_sha256": _canonical_hash(list(data.fit_record_ids)),
                "validation_record_order_sha256": _canonical_hash(list(data.validation_record_ids)),
                "fit_valid_post_bos_positions": int(data.fit_valid_mask[:, 1:].sum().item()),
                "validation_valid_post_bos_positions": int(data.validation_valid_mask[:, 1:].sum().item()),
                "fit_observation_tensor_sha256": _tensor_digest(data.fit_observations),
                "fit_truth_tensor_sha256": _tensor_digest(data.fit_truth),
                "validation_observation_tensor_sha256": _tensor_digest(data.validation_observations),
                "validation_truth_tensor_sha256": _tensor_digest(data.validation_truth),
                "embedding_tensor_sha256": _tensor_digest(data.embedding_table),
                "hidden_size": data.hidden_size,
                "vocabulary_size": data.vocabulary_size,
                "sequence_length": int(data.fit_observations.shape[1]),
                "validation_groups": sorted(set(data.validation_groups)),
                "validation_group_record_counts": {
                    group: int(sum(value == group for value in data.validation_groups))
                    for group in sorted(set(data.validation_groups))
                },
                "validation_native_geometries": [list(geometry) for geometry in data.validation_native_geometries],
                "validation_padding": dict(data.validation_padding),
                "truth_contract": "public current-token labels H_i -> token_ids[i], scoring post-BOS valid positions",
            },
            "fixed_settings": {
                "fit_steps": args.steps,
                "subset_steps": args.subset_steps,
                "validation_every": args.validation_every,
                "record_batch_size": args.record_batch_size,
                "position_budget": args.position_budget,
                "position_budget_scope": "total selected positions per record batch, not per record",
                "subset_records": args.subset_records,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "gradient_clip_norm": args.gradient_clip_norm,
                "scheduler": "CosineAnnealingLR",
                "seed": args.seed,
                "base_frozen": True,
                "base_state_keys": ["W", "b", "s"],
                "vocabulary_bias": False,
                "input_normalization": "F.layer_norm(x, (hidden_size,), weight=None, bias=None, eps=1e-5)",
                "context_input": "activation H_0..H_i with causal mask; no token or teacher input",
                "selection_rule": "step 0 is retained as the exact-base diagnostic; select the earliest nonzero checkpoint attaining maximum style-balanced public validation token accuracy",
            },
            "sampler": {
                "schedule": schedule_record,
                "same_main_schedule_for_methods": True,
                "same_subset_schedule_for_methods": True,
                "post_bos_only": True,
            },
            "preparation": {
                "embedding_transfer_seconds": embedding_transfer_seconds,
                "embedding_transfer_once": True,
                "public_fit_preparation_external_to_runner": True,
            },
            "resource_preflight": preflight,
            "resource_guard": guard_summary,
            "methods": methods,
            "execution": {
                "argv": list(sys.argv),
                "cwd": str(Path.cwd()),
                "python": sys.executable,
                "python_version": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "git_commit_at_start": git_start,
                "git_commit_at_end": git_end,
                "source_hashes_at_start": source_start,
                "source_hashes_at_end": source_end,
                "source_unchanged": source_unchanged,
                "code_bundle_sha256": _source_bundle_hash(source_start),
                "peak_memory": _runtime_memory(device),
            },
            "runtime_components": {
                "deployed_inputs": "cut-4 activation tensor, fixed normalized public embedding table, selected decoder state",
                "public_prefix_calls": 0,
                "candidate_simulations": 0,
                "a2_fallback": False,
                "teacher_prefix": False,
                "full_vocab_projection": "only selected loss rows during fit/validation; full rows available to activation-only inference adapter",
            },
        }
        _write_json(output_root / "run_evidence.json", evidence)
        return evidence
    except Exception as exc:
        failure = {
            "schema": "token-reconstruction.trr0004-contextual-fit-failure.v1",
            "task_id": TASK_ID,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_utc": started_utc,
            "failed_at_utc": _utc_now(),
            "git_commit_at_start": git_start,
            "source_hashes_at_start": source_start,
            "resource_preflight": preflight,
            "resource_guard": guard_summary,
        }
        failure_path = output_root / "failure.json"
        if not failure_path.exists():
            _write_json(failure_path, failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_fit(args)
    except (ContextualFitError, CausalDecoderExtensionError, OSError, ValueError) as exc:
        print(f"TRR-0004 contextual fit failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
