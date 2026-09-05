#!/usr/bin/env python3
"""Freeze and score the TRR-P01 blind prediction interface.

The pilot has two deliberately separate interfaces.  A reconstructor sees one
opaque public arm and writes predictions; the scorer verifies that arm and the
prediction files before it opens the private truth.  This module contains the
small integrity gate between those processes.  It does not load a model and
never needs CUDA.

The public arm contract is the one produced by the task-local preparation
script::

    <arm>/sanitized_config.json
    <arm>/observation_index.json
    <arm>/observations.safetensors

The prediction contract contains both ``predictions.safetensors`` (one int32
matrix per ``method.metric`` key) and ``predictions.jsonl`` (one row per
record and key).  Both files are checked against one another and then bound
in a create-only receipt.  ``truth`` is intentionally absent from the freeze
functions; it is opened only by :func:`score_frozen`, after validation has
completed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))

import torch
from safetensors import safe_open

# ``common.py`` is task-local and is also used by the preparation and
# reconstruction scripts.  Keep this module importable both as
# ``scripts.trr_p01.freeze_score`` and as a direct script.
try:  # pragma: no cover - import branch depends on invocation style
    from .common import (
        BOS_TOKEN_ID,
        CONFIG_SCHEMA,
        FREEZE_SCHEMA,
        HIDDEN_SIZE,
        MODEL_ID,
        MODEL_REVISION,
        OBSERVATION_INDEX_SCHEMA,
        OBSERVATION_SCHEMA,
        PilotError,
    )
except SyntaxError:  # pragma: no cover - defensive for old Python parsers
    raise
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    from common import (  # type: ignore[no-redef]
        BOS_TOKEN_ID,
        CONFIG_SCHEMA,
        FREEZE_SCHEMA,
        HIDDEN_SIZE,
        MODEL_ID,
        MODEL_REVISION,
        OBSERVATION_INDEX_SCHEMA,
        OBSERVATION_SCHEMA,
        PilotError,
    )

try:
    from .common import (
        SEQUENCE_TOKENS,
        SCORED_TOKENS,
        TASK_ID,
        file_record as _common_file_record,
        load_json as _common_load_json,
        load_public_interface,
        observation_row_digest,
        read_jsonl as _common_read_jsonl,
        sha256_file as _common_sha256_file,
        utc_now as _common_utc_now,
        write_json_exclusive as _common_write_json_exclusive,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    from common import (  # type: ignore[no-redef]
        SEQUENCE_TOKENS,
        SCORED_TOKENS,
        TASK_ID,
        file_record as _common_file_record,
        load_json as _common_load_json,
        load_public_interface,
        observation_row_digest,
        read_jsonl as _common_read_jsonl,
        sha256_file as _common_sha256_file,
        utc_now as _common_utc_now,
        write_json_exclusive as _common_write_json_exclusive,
    )


class FreezeScoreError(PilotError):
    """Raised when a frozen public/prediction contract is invalid."""


# Safetensors metadata is string-valued.  The JSONL contract uses JSON native
# values.  These are the required fields, while unknown additional fields are
# retained in the file hash and therefore remain bound by the receipt.
_ROW_FIELDS = {
    "record_id",
    "method",
    "metric",
    "sequence_length",
    "prediction_tokens",
    "mask_digest",
    "position_digest",
    "observation_digest",
    "model_id",
    "model_revision",
    "cut_depth",
    "vocab_size",
    "hidden_size",
    "config_sha256",
    "evidence_sha256",
    "table_sha256",
    "truth_opened",
}
_FORBIDDEN_ROW_FIELDS = {
    "condition",
    "source",
    "source_row",
    "source_row_index",
    "text",
    "text_hash",
    "target",
    "target_path",
    "truth",
    "correct",
    "style",
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PRIVATE_TRUTH_SCHEMA = "token-reconstruction.trr-p01-private-truth.v1"


_KNOWN_FILE_NAMES = {
    "config": "sanitized_config.json",
    "index": "observation_index.json",
    "observation": "observations.safetensors",
}


def _error(message: str) -> FreezeScoreError:
    return FreezeScoreError(message)


def _path(value: os.PathLike[str] | str) -> Path:
    return Path(value).expanduser()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise _error(f"{label} must be a regular file: {path}")
    return path


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise _error(f"{label} must be a regular directory: {path}")
    return path


def _sha256(path: Path) -> str:
    _regular_file(path, "artifact")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise _error(f"cannot read artifact: {path}") from exc
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path | None = None, label: str | None = None) -> dict[str, Any]:
    path = _regular_file(path, "artifact")
    resolved = path.resolve()
    if label is None:
        if root is not None:
            try:
                label = resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                label = str(resolved)
        else:
            label = str(resolved)
    return {"path": label, "bytes": int(path.stat().st_size), "sha256": _sha256(path)}


def _load_json(path: Path, label: str = "JSON input") -> Any:
    _regular_file(path, label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(f"invalid {label}: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _regular_file(path, "prediction JSONL")
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise _error(f"cannot open prediction JSONL: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise _error(f"blank prediction JSONL line at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _error(f"invalid prediction JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise _error(f"prediction JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise _error(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise _error(f"cannot write output: {path}") from exc


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise _error("value cannot be represented canonically") from exc


def _digest_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"{label} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(f"{label} must be an integer")
    return int(value)


def _json_metadata(metadata: Mapping[str, str], key: str, *, required: bool = False) -> Any:
    if key not in metadata:
        if required:
            raise _error(f"prediction metadata field is missing: {key}")
        return None
    value: Any = metadata[key]
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    try:
        # NumPy does not expose torch.bfloat16 in the supported runtime.  A
        # byte view preserves the exact storage bytes for every ordinary CPU
        # dtype, including BF16 activation rows.
        raw = value.view(torch.uint8).numpy().tobytes()
    except (TypeError, RuntimeError) as exc:
        raise _error("cannot digest tensor") from exc
    descriptor = _canonical({"dtype": str(value.dtype), "shape": list(value.shape)}).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + raw).hexdigest()


def _resolve_public(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise _error(f"{label} path is not a safe relative path")
    base = root.resolve()
    candidate_unresolved = base / relative
    if candidate_unresolved.is_symlink():
        raise _error(f"{label} is a symlink: {relative}")
    candidate = candidate_unresolved.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise _error(f"{label} escaped public root: {relative}") from exc
    return _regular_file(candidate, label)


def _record_from_declared(value: Mapping[str, Any], root: Path, label: str) -> tuple[dict[str, Any], Path | None]:
    """Verify a ``{path, bytes, sha256}`` identity against ``root``."""

    if not isinstance(value, Mapping):
        raise _error(f"{label} identity is not an object")
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise _error(f"{label} identity has no path")
    path = _resolve_public(root, path_value, label)
    actual = _file_record(path, root=root)
    if "bytes" in value and _integer(value["bytes"], f"{label} bytes") != actual["bytes"]:
        raise _error(f"{label} byte count changed")
    if "sha256" in value and _sha(value["sha256"], f"{label} sha256") != actual["sha256"]:
        raise _error(f"{label} hash changed")
    return actual, path


def _artifact_record(path_value: Any, root: Path, label: str, *, declared: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], Path]:
    if not isinstance(path_value, str):
        raise _error(f"{label} path is missing")
    path = _resolve_public(root, path_value, label)
    actual = _file_record(path, root=root)
    if declared is not None:
        if "bytes" in declared and _integer(declared["bytes"], f"{label} bytes") != actual["bytes"]:
            raise _error(f"{label} byte count changed")
        if "sha256" in declared and _sha(declared["sha256"], f"{label} sha256") != actual["sha256"]:
            raise _error(f"{label} hash changed")
    return actual, path


def _load_safetensor(path: Path, label: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    _regular_file(path, label)
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = dict(handle.metadata() or {})
            tensors = {key: handle.get_tensor(key) for key in keys}
    except Exception as exc:  # safetensors exposes several exception classes
        if isinstance(exc, FreezeScoreError):
            raise
        raise _error(f"invalid {label}: {path}") from exc
    return tensors, metadata


def _observation_public_snapshot(public_dir: Path) -> dict[str, Any]:
    """Load and verify the exact condition-free public arm.

    This is intentionally local instead of delegating to ``common.py``.  The
    shared helper predates torch's BF16 NumPy limitation and its digest helper
    cannot serialize a BF16 tensor on the pilot runtime.  Keeping the gate
    here makes the verifier independent and gives it one explicit, tested
    byte-level digest implementation.
    """

    root = _regular_directory(public_dir.resolve(), "public arm root")
    config_path = root / _KNOWN_FILE_NAMES["config"]
    index_path = root / _KNOWN_FILE_NAMES["index"]
    observation_path = root / _KNOWN_FILE_NAMES["observation"]
    config = _load_json(config_path, "sanitized config")
    index = _load_json(index_path, "observation index")
    if not isinstance(config, Mapping) or not isinstance(index, Mapping):
        raise _error("public config or observation index is not an object")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("task_id") != TASK_ID
        or config.get("truth_opened") is not False
        or config.get("source_truth_included") is not False
        or "condition" in config
    ):
        raise _error("sanitized public config schema, condition, or truth state changed")
    expected_ids = [f"p01-r{position:04d}" for position in range(1, 17)]
    if config.get("record_order") != expected_ids:
        raise _error("sanitized record order changed")
    expected_geometry = {
        "records": 16,
        "sequence_tokens": SEQUENCE_TOKENS,
        "scored_tokens": SCORED_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "cut_depth": 4,
    }
    if config.get("geometry") != expected_geometry:
        raise _error("sanitized geometry changed")
    expected_index_geometry = {
        "records": 16,
        "sequence_tokens": SEQUENCE_TOKENS,
        "scored_tokens": SCORED_TOKENS,
        "hidden_size": HIDDEN_SIZE,
    }
    if (
        index.get("schema") != OBSERVATION_INDEX_SCHEMA
        or index.get("task_id") != TASK_ID
        or index.get("truth_opened") is not False
        or index.get("source_material_included") is not False
        or index.get("geometry") != expected_index_geometry
    ):
        raise _error("observation index schema, geometry, or truth state changed")
    rows = index.get("records")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise _error("observation index record count changed")
    record_rows: list[dict[str, Any]] = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or set(row) != {
            "record_id",
            "sequence_length",
            "mask_digest",
            "position_digest",
            "observation_digest",
        }:
            raise _error(f"observation index row {number} exposes non-stage metadata")
        if not isinstance(row.get("record_id"), str):
            raise _error(f"observation index row {number} has an invalid opaque ID")
        if row.get("sequence_length") != SEQUENCE_TOKENS:
            raise _error("observation sequence length changed")
        record_rows.append(dict(row))
    if [row["record_id"] for row in record_rows] != expected_ids:
        raise _error("opaque record order changed")
    if len({row["record_id"] for row in record_rows}) != len(record_rows):
        raise _error("observation index contains duplicate opaque IDs")
    all_mask_digest = _tensor_digest(torch.ones(SEQUENCE_TOKENS, dtype=torch.int64))
    all_position_digest = _tensor_digest(torch.arange(SEQUENCE_TOKENS, dtype=torch.int64))
    for row in record_rows:
        if row["mask_digest"] != all_mask_digest or row["position_digest"] != all_position_digest:
            raise _error(f"mask or position digest changed for {row['record_id']}")
        _sha(row["observation_digest"], "observation digest")
    observation_identity = index.get("observation")
    if not isinstance(observation_identity, Mapping) or set(observation_identity) != {"path", "bytes", "sha256"}:
        raise _error("observation index artifact identity changed")
    observation_path = _resolve_public(root, str(observation_identity["path"]), "observation artifact")
    observation_record = _file_record(observation_path, root=root)
    if _integer(observation_identity["bytes"], "observation bytes") != observation_record["bytes"]:
        raise _error("observation byte count changed")
    if _sha(observation_identity["sha256"], "observation sha256") != observation_record["sha256"]:
        raise _error("observation hash changed")

    model = config.get("model")
    if not isinstance(model, Mapping):
        raise _error("sanitized config model identity is missing")
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise _error("public model identity changed")
    for field in ("hidden_size", "vocab_size", "cut_depth", "bos_token_id"):
        if not isinstance(model.get(field), int) or isinstance(model.get(field), bool):
            raise _error(f"public model field is invalid: {field}")
    if (
        model.get("hidden_size") != HIDDEN_SIZE
        or model.get("vocab_size") != 128256
        or model.get("cut_depth") != 4
        or model.get("bos_token_id") != BOS_TOKEN_ID
    ):
        raise _error("public model geometry or BOS identity changed")

    observation_tensors, observation_metadata = _load_safetensor(observation_path, "observation artifact")
    if set(observation_tensors) != {"activations"}:
        raise _error("observation tensor fields changed")
    if observation_metadata != {
        "schema": OBSERVATION_SCHEMA,
        "opaque_records": "true",
        "source_truth_included": "false",
    }:
        raise _error("observation metadata or truth state changed")
    observations = observation_tensors["activations"]
    if tuple(observations.shape) != (16, SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise _error("observation tensor geometry changed")
    if observations.dtype != torch.bfloat16:
        raise _error("observation tensor dtype changed")
    if not torch.isfinite(observations).all().item():
        raise _error("observation tensor contains non-finite values")
    for row, observation in zip(record_rows, observations):
        if row["observation_digest"] != _tensor_digest(observation):
            raise _error(f"observation digest changed for {row['record_id']}")
    return {
        "root": str(root),
        "config": dict(config),
        "index": dict(index),
        "geometry": dict(config["geometry"]),
        "model": dict(model),
        "config_record": _file_record(config_path, root=root),
        "index_record": _file_record(index_path, root=root),
        "observation_record": observation_record,
        "observation_metadata": observation_metadata,
        "observation_records": record_rows,
        "record_ids": expected_ids,
        "observations": observations,
    }


def _declared_prediction_keys(config: Mapping[str, Any]) -> set[str] | None:
    """Return the exact declared method.metric arm set when available.

    The base pilot declares ``methods`` and ``metric_order``, which means the
    Cartesian product is frozen.  Comparator configs can instead provide an
    explicit ``prediction_arms`` (or ``arms``) list when their metrics do not
    form that product.  Returning ``None`` is reserved for a config that truly
    does not declare its arm matrix.
    """

    explicit = config.get("prediction_arms")
    if explicit is None:
        explicit = config.get("method_metric_order")
    if explicit is None:
        explicit = config.get("arms")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise _error("sanitized prediction arm declaration is invalid")
        keys: set[str] = set()
        for item in explicit:
            if isinstance(item, str):
                method, metric = _prediction_key_parts(item)
                key = f"{method}.{metric}"
            elif isinstance(item, Mapping) and isinstance(item.get("method"), str) and isinstance(item.get("metric"), str):
                method = str(item["method"])
                metric = str(item["metric"])
                if not method or not metric:
                    raise _error("sanitized prediction arm declaration is invalid")
                key = f"{method}.{metric}"
            else:
                raise _error("sanitized prediction arm declaration is invalid")
            if key in keys:
                raise _error("sanitized prediction arm declaration has duplicates")
            keys.add(key)
        return keys
    methods, metrics = _declared_methods(config)
    if methods is None or metrics is None:
        return None
    return {f"{method}.{metric}" for method in methods for metric in metrics}


def _declared_methods(config: Mapping[str, Any]) -> tuple[set[str] | None, set[str] | None]:
    methods_value = config.get("methods")
    if methods_value is None:
        methods_value = config.get("method_order")
    metrics_value = config.get("metric_order")
    methods: set[str] | None = None
    metrics: set[str] | None = None
    if methods_value is not None:
        if not isinstance(methods_value, list) or not all(isinstance(item, str) and item for item in methods_value):
            raise _error("sanitized method declaration is invalid")
        methods = set(methods_value)
    if metrics_value is not None:
        if not isinstance(metrics_value, list) or not all(isinstance(item, str) and item for item in metrics_value):
            raise _error("sanitized metric declaration is invalid")
        metrics = set(metrics_value)
    return methods, metrics


def _prediction_paths(predictions_path: os.PathLike[str] | str) -> tuple[Path, Path]:
    path = _path(predictions_path)
    if path.is_dir():
        return path / "predictions.safetensors", path / "predictions.jsonl"
    if path.suffix == ".safetensors":
        return path, path.with_name("predictions.jsonl")
    if path.suffix == ".jsonl":
        return path.with_name("predictions.safetensors"), path
    raise _error("predictions path must be a directory, .safetensors, or .jsonl")


def _prediction_key_parts(key: str) -> tuple[str, str]:
    if not isinstance(key, str) or "." not in key:
        raise _error(f"prediction tensor key is not method.metric: {key!r}")
    method, metric = key.rsplit(".", 1)
    if not method or not metric:
        raise _error(f"prediction tensor key is not method.metric: {key!r}")
    return method, metric


def _prediction_identity(config: Mapping[str, Any], public_dir: Path) -> dict[str, Any]:
    """Find the declared prototype/table identity without opening private data."""

    candidates: list[dict[str, Any]] = []

    def visit(value: Any, keys: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if lowered in {
                    "table",
                    "table_identity",
                    "prototype_table",
                    "prototype_table_identity",
                    "table_artifact",
                    "prototype_artifact",
                    "prototype",
                } and isinstance(child, Mapping):
                    candidates.append(dict(child))
                visit(child, keys + (lowered,))
        elif isinstance(value, list):
            for child in value:
                visit(child, keys)

    visit(config)
    selected: dict[str, Any] | None = None
    for candidate in candidates:
        if any(key in candidate for key in ("sha256", "table_sha256", "prototype_table_sha256", "path")):
            selected = candidate
            break

    if selected is None:
        # A config may expose only a scalar table_sha256 at the top level.
        scalar = config.get("table_sha256") or config.get("prototype_table_sha256")
        if scalar is None:
            return {"declared": False, "sha256": None, "artifact": None}
        return {"declared": True, "sha256": _sha(scalar, "table sha256"), "artifact": None}

    scalar = selected.get("sha256") or selected.get("table_sha256") or selected.get("prototype_table_sha256")
    path_value = selected.get("path")
    artifact: dict[str, Any] | None = None
    if path_value is not None:
        # Table files are public assets but may be placed next to the arm or
        # at a task-local path named by the config.  Relative paths must stay
        # inside the public arm; absolute paths are accepted only as regular
        # files and are recorded explicitly, never followed through symlinks.
        if not isinstance(path_value, str):
            raise _error("table artifact path is invalid")
        candidate_path = Path(path_value).expanduser()
        if candidate_path.is_absolute():
            table_path = _regular_file(candidate_path, "table artifact")
            artifact = _file_record(table_path)
            if "bytes" in selected and _integer(selected["bytes"], "table bytes") != artifact["bytes"]:
                raise _error("table artifact byte count changed")
            if "sha256" in selected and _sha(selected["sha256"], "table sha256") != artifact["sha256"]:
                raise _error("table artifact hash changed")
        else:
            artifact, _ = _artifact_record(path_value, public_dir, "table artifact", declared=selected)
    if scalar is None and artifact is not None:
        scalar = artifact["sha256"]
    if scalar is None:
        raise _error("table identity has no sha256")
    scalar = _sha(scalar, "table sha256")
    if artifact is not None and artifact["sha256"] != scalar:
        raise _error("table artifact hash disagrees with config")
    return {"declared": True, "sha256": scalar, "artifact": artifact}


def _extract_code_identity(value: Any, base: Path) -> dict[str, Any]:
    """Extract an evidence-bound executable/source identity if one is declared."""

    hashes: list[str] = []
    commits: list[str] = []
    paths: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                lowered = str(key).lower()
                if lowered in {"implementation_commit", "code_commit", "source_commit"} and child is not None:
                    if not isinstance(child, str) or not child:
                        raise _error(f"{key} must be a non-empty string")
                    commits.append(child)
                if lowered in {
                    "code_sha256",
                    "source_sha256",
                    "executable_sha256",
                    "script_sha256",
                } and child is not None:
                    hashes.append(_sha(child, f"{key}"))
                if lowered in {"code_files", "source_files", "executable_files", "code_artifacts"}:
                    if not isinstance(child, list):
                        raise _error(f"{key} must be a list")
                    for item in child:
                        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                            candidate = Path(str(item["path"])).expanduser()
                            if not candidate.is_absolute():
                                candidate = base / candidate
                            artifact = _file_record(_regular_file(candidate, "code artifact"))
                            if "bytes" in item and _integer(item["bytes"], "code bytes") != artifact["bytes"]:
                                raise _error("code artifact byte count changed")
                            if "sha256" in item and _sha(item["sha256"], "code sha256") != artifact["sha256"]:
                                raise _error("code artifact hash changed")
                            paths.append(artifact)
                        elif isinstance(item, str):
                            candidate = Path(item).expanduser()
                            if not candidate.is_absolute():
                                candidate = base / candidate
                            paths.append(_file_record(_regular_file(candidate, "code artifact")))
                        else:
                            raise _error("code artifact entry is invalid")
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    hashes = sorted(set(hashes))
    commits = sorted(set(commits))
    paths = sorted(paths, key=lambda item: str(item["path"]))
    if not hashes and not commits and not paths:
        return {"status": "evidence_hash_only", "sha256": None, "commits": [], "files": []}
    return {"status": "bound", "sha256": hashes, "commits": commits, "files": paths}


@dataclass
class _PredictionBundle:
    tensor_path: Path
    jsonl_path: Path
    tensor_record: dict[str, Any]
    jsonl_record: dict[str, Any]
    metadata: dict[str, str]
    tensors: dict[str, torch.Tensor]
    rows: list[dict[str, Any]]
    arms: list[dict[str, Any]]
    row_digest: str
    config_sha256: str
    evidence_sha256: str
    table_sha256: str


def _load_prediction_bundle(public: Mapping[str, Any], predictions_path: os.PathLike[str] | str, evidence_path: os.PathLike[str] | str | None) -> _PredictionBundle:
    root = Path(str(public["root"])).resolve()
    tensor_path, jsonl_path = _prediction_paths(predictions_path)
    tensor_path = _regular_file(tensor_path, "prediction tensor artifact")
    jsonl_path = _regular_file(jsonl_path, "prediction JSONL artifact")
    tensor_record = _file_record(tensor_path)
    jsonl_record = _file_record(jsonl_path)
    tensors, metadata = _load_safetensor(tensor_path, "prediction tensor artifact")
    if metadata.get("schema") != "token-reconstruction.trr-p01-predictions.v1":
        raise _error("prediction tensor schema changed")
    if metadata.get("task_id") != TASK_ID or metadata.get("truth_opened") != "false":
        raise _error("prediction tensor metadata or truth state changed")
    if not tensors:
        raise _error("prediction tensor artifact is empty")
    methods, metrics = _declared_methods(public["config"])
    declared_keys = _declared_prediction_keys(public["config"])
    tensor_parts: dict[str, tuple[str, str]] = {}
    expected_shape = (len(public["record_ids"]), int(public["geometry"].get("sequence_tokens", SEQUENCE_TOKENS)))
    for key, tensor in tensors.items():
        method, metric = _prediction_key_parts(key)
        if methods is not None and method not in methods:
            raise _error(f"prediction method is not declared in config: {method}")
        if metrics is not None and metric not in metrics:
            raise _error(f"prediction metric is not declared in config: {metric}")
        if tuple(tensor.shape) != expected_shape:
            raise _error(f"prediction tensor geometry changed for {key}")
        if tensor.dtype != torch.int32:
            raise _error(f"prediction tensor dtype changed for {key}")
        tensor_parts[key] = (method, metric)

    rows = _read_jsonl(jsonl_path)
    if not rows:
        raise _error("prediction JSONL is empty")
    record_ids = [str(item) for item in public["record_ids"]]
    geometry = public["geometry"]
    model = public["model"]
    sequence_tokens = int(geometry.get("sequence_tokens", SEQUENCE_TOKENS))
    vocab_size = _integer(model.get("vocab_size"), "model vocab_size")
    bos_token_id = _integer(model.get("bos_token_id"), "model bos_token_id")
    cut_depth = _integer(model.get("cut_depth"), "model cut_depth")
    hidden_size = _integer(model.get("hidden_size"), "model hidden_size")
    config_sha256 = _sha(public["config_record"]["sha256"], "config sha256")
    table_identity = _prediction_identity(public["config"], root)
    evidence_values: list[str] = []
    table_values: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    row_core: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        missing = sorted(_ROW_FIELDS.difference(row))
        if missing:
            raise _error(f"prediction JSONL row {row_number} is missing fields: {', '.join(missing)}")
        forbidden = sorted(_FORBIDDEN_ROW_FIELDS.intersection(row))
        if forbidden:
            raise _error(f"prediction row {row_number} exposes forbidden fields: {', '.join(forbidden)}")
        if not isinstance(row["record_id"], str) or not row["record_id"]:
            raise _error(f"prediction row {row_number} has an invalid opaque ID")
        method = row["method"]
        metric = row["metric"]
        if not isinstance(method, str) or not method or not isinstance(metric, str) or not metric:
            raise _error(f"prediction row {row_number} method/metric is invalid")
        if methods is not None and method not in methods:
            raise _error(f"prediction row method is not declared in config: {method}")
        if metrics is not None and metric not in metrics:
            raise _error(f"prediction row metric is not declared in config: {metric}")
        key = f"{method}.{metric}"
        if key not in tensor_parts:
            raise _error(f"prediction JSONL row has no tensor arm: {key}")
        if _integer(row["sequence_length"], f"prediction row {row_number} sequence_length") != sequence_tokens:
            raise _error(f"prediction row {row_number} sequence length changed")
        tokens = row["prediction_tokens"]
        if not isinstance(tokens, list) or len(tokens) != sequence_tokens:
            raise _error(f"prediction row {row_number} token shape changed")
        parsed_tokens: list[int] = []
        for position, token in enumerate(tokens):
            value = _integer(token, f"prediction row {row_number} token {position}")
            if value < 0 or value >= vocab_size:
                raise _error(f"prediction row {row_number} token is outside vocabulary")
            parsed_tokens.append(value)
        if parsed_tokens[0] != bos_token_id:
            raise _error(f"prediction row {row_number} BOS token changed")
        if row["truth_opened"] is not False:
            raise _error(f"prediction row {row_number} truth state changed")
        if _integer(row["cut_depth"], f"prediction row {row_number} cut_depth") != cut_depth:
            raise _error(f"prediction row {row_number} cut depth changed")
        if _integer(row["vocab_size"], f"prediction row {row_number} vocab_size") != vocab_size:
            raise _error(f"prediction row {row_number} vocabulary changed")
        if _integer(row["hidden_size"], f"prediction row {row_number} hidden_size") != hidden_size:
            raise _error(f"prediction row {row_number} hidden size changed")
        if row["model_id"] != model.get("id") or row["model_revision"] != model.get("revision"):
            raise _error(f"prediction row {row_number} model identity changed")
        if _sha(row["config_sha256"], f"prediction row {row_number} config_sha256") != config_sha256:
            raise _error(f"prediction row {row_number} config hash changed")
        evidence_sha256 = _sha(row["evidence_sha256"], f"prediction row {row_number} evidence_sha256")
        table_sha256 = _sha(row["table_sha256"], f"prediction row {row_number} table_sha256")
        evidence_values.append(evidence_sha256)
        table_values.append(table_sha256)
        if table_identity["sha256"] is not None and table_sha256 != table_identity["sha256"]:
            raise _error(f"prediction row {row_number} table hash disagrees with config")
        expected_row = next((item for item in public["observation_records"] if item["record_id"] == row["record_id"]), None)
        if expected_row is None:
            raise _error(f"prediction row {row_number} has a foreign opaque ID")
        for digest_name in ("mask_digest", "position_digest", "observation_digest"):
            if row[digest_name] != expected_row[digest_name]:
                raise _error(f"prediction row {row_number} {digest_name} changed")
        grouped.setdefault((method, metric), []).append({**row, "prediction_tokens": parsed_tokens})
        row_core.append(
            {
                "record_id": row["record_id"],
                "method": method,
                "metric": metric,
                "prediction_tokens": parsed_tokens,
                "mask_digest": row["mask_digest"],
                "position_digest": row["position_digest"],
                "observation_digest": row["observation_digest"],
            }
        )

    if len(set(evidence_values)) != 1:
        raise _error("prediction rows disagree on evidence hash")
    if len(set(table_values)) != 1:
        raise _error("prediction rows disagree on table hash")
    evidence_sha256 = evidence_values[0]
    table_sha256 = table_values[0]
    if evidence_path is not None:
        evidence_file = _regular_file(_path(evidence_path), "reconstructor evidence")
        if _sha256(evidence_file) != evidence_sha256:
            raise _error("reconstructor evidence hash changed")

    if set(grouped) != set(tensor_parts.values()):
        raise _error("prediction tensor/JSONL arm coverage changed")
    if declared_keys is not None and set(tensor_parts) != declared_keys:
        missing = sorted(declared_keys.difference(tensor_parts))
        extra = sorted(set(tensor_parts).difference(declared_keys))
        raise _error(f"prediction arm matrix differs from config (missing={missing}, extra={extra})")
    arms: list[dict[str, Any]] = []
    # Preserve the tensor's deterministic key order for the receipt, while
    # requiring each JSONL arm to use the exact opaque order from the index.
    for key in tensors:
        method, metric = tensor_parts[key]
        arm_rows = grouped.get((method, metric), [])
        if len(arm_rows) != len(record_ids):
            raise _error(f"prediction arm {key} does not cover every record exactly once")
        arm_ids = [row["record_id"] for row in arm_rows]
        if arm_ids != record_ids:
            if len(set(arm_ids)) != len(arm_ids):
                raise _error(f"prediction arm {key} contains duplicate opaque IDs")
            raise _error(f"prediction arm {key} opaque ID order or coverage changed")
        tensor_rows = tensors[key].to(torch.long).tolist()
        for index, row in enumerate(arm_rows):
            if tensor_rows[index] != row["prediction_tokens"]:
                raise _error(f"prediction tensor and JSONL disagree for {key} row {index + 1}")
        arms.append(
            {
                "tensor_key": key,
                "method": method,
                "metric": metric,
                "record_ids": arm_ids,
                "prediction_tokens": [list(row["prediction_tokens"]) for row in arm_rows],
                "row_digests": [_digest_value(row) for row in arm_rows],
            }
        )

    return _PredictionBundle(
        tensor_path=tensor_path,
        jsonl_path=jsonl_path,
        tensor_record=tensor_record,
        jsonl_record=jsonl_record,
        metadata=metadata,
        tensors=tensors,
        rows=rows,
        arms=arms,
        row_digest=_digest_value(row_core),
        config_sha256=config_sha256,
        evidence_sha256=evidence_sha256,
        table_sha256=table_sha256,
    )


def _snapshot(public_dir: os.PathLike[str] | str, predictions_path: os.PathLike[str] | str, evidence_path: os.PathLike[str] | str | None) -> dict[str, Any]:
    public = _observation_public_snapshot(_path(public_dir))
    table = _prediction_identity(public["config"], Path(public["root"]))
    bundle = _load_prediction_bundle(public, predictions_path, evidence_path)
    evidence_record: dict[str, Any] | None = None
    evidence_json: Any = None
    if evidence_path is not None:
        evidence_file = _regular_file(_path(evidence_path), "reconstructor evidence")
        evidence_record = _file_record(evidence_file)
        evidence_json = _load_json(evidence_file, "reconstructor evidence")
    code_identity = _extract_code_identity(evidence_json, Path(public["root"])) if evidence_json is not None else {
        "status": "evidence_hash_only",
        "sha256": None,
        "commits": [],
        "files": [],
    }
    if table["sha256"] is None:
        table = {**table, "sha256": bundle.table_sha256}
    if table["sha256"] != bundle.table_sha256:
        raise _error("prediction table hash is not bound to public configuration")
    return {
        "public": {
            "root": public["root"],
            "config": public["config_record"],
            "observation_index": public["index_record"],
            "observation": public["observation_record"],
            "observation_metadata": public["observation_metadata"],
            "geometry": public["geometry"],
            "model": public["model"],
            "record_ids": public["record_ids"],
            "observation_records": public["observation_records"],
        },
        "table": table,
        "prediction": {
            "tensor": bundle.tensor_record,
            "jsonl": bundle.jsonl_record,
            "metadata": bundle.metadata,
            "row_digest": bundle.row_digest,
            "arms": bundle.arms,
            "config_sha256": bundle.config_sha256,
            "evidence_sha256": bundle.evidence_sha256,
            "table_sha256": bundle.table_sha256,
        },
        "evidence": evidence_record,
        "code_identity": code_identity,
    }


def freeze_predictions(
    public_dir: os.PathLike[str] | str,
    predictions_path: os.PathLike[str] | str,
    receipt_path: os.PathLike[str] | str,
    evidence_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Validate and create a prediction receipt without opening truth.

    ``public_dir`` is one opaque arm root.  ``predictions_path`` can be either
    its prediction directory, either member of the prediction pair, or the
    tensor artifact itself.  The output is create-only.
    """

    receipt_file = _path(receipt_path)
    if receipt_file.exists() or receipt_file.is_symlink():
        raise _error(f"freeze receipt is create-only: {receipt_file}")
    snapshot = _snapshot(public_dir, predictions_path, evidence_path)
    receipt = {
        "schema": FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_AND_VERIFIED_BEFORE_TRUTH_OPEN",
        "created_utc": _common_utc_now(),
        "truth_opened": False,
        "public": snapshot["public"],
        "table": snapshot["table"],
        "prediction": snapshot["prediction"],
        "evidence": snapshot["evidence"],
        "code_identity": snapshot["code_identity"],
        "validation": {
            "observation_hashes_verified": True,
            "prediction_tensor_jsonl_consistent": True,
            "opaque_ids_complete_and_ordered": True,
            "truth_opened": False,
        },
    }
    _write_json_exclusive(receipt_file, receipt)
    return receipt


def _same_value(expected: Any, actual: Any, label: str) -> None:
    if expected != actual:
        raise _error(f"{label} changed after freeze")


def validate_frozen(
    public_dir: os.PathLike[str] | str,
    predictions_path: os.PathLike[str] | str,
    receipt_path: os.PathLike[str] | str,
    evidence_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Revalidate a receipt and all public/prediction bytes before truth."""

    receipt_file = _regular_file(_path(receipt_path), "freeze receipt")
    receipt = _load_json(receipt_file, "freeze receipt")
    if not isinstance(receipt, Mapping):
        raise _error("freeze receipt is not an object")
    if (
        receipt.get("schema") != FREEZE_SCHEMA
        or receipt.get("task_id") != TASK_ID
        or receipt.get("status") != "FROZEN_AND_VERIFIED_BEFORE_TRUTH_OPEN"
        or receipt.get("truth_opened") is not False
    ):
        raise _error("freeze receipt schema or truth state changed")
    snapshot = _snapshot(public_dir, predictions_path, evidence_path)
    for section in ("public", "table", "prediction", "evidence", "code_identity"):
        _same_value(receipt.get(section), snapshot[section], f"frozen {section}")
    return {
        "status": "FREEZE_VERIFIED_BEFORE_TRUTH_OPEN",
        "truth_opened": False,
        "receipt": dict(receipt),
        "snapshot": snapshot,
    }


def _truth_paths(truth_path: os.PathLike[str] | str, manifest_path: os.PathLike[str] | str | None) -> tuple[Path, Path | None]:
    supplied = _path(truth_path)
    if supplied.is_dir():
        truth_file = supplied / "private_truth.safetensors"
        if manifest_path is None:
            for candidate in (supplied / "private_manifest.json", supplied / "manifest.json"):
                if candidate.exists():
                    manifest_path = candidate
                    break
    else:
        truth_file = supplied
    return truth_file, (_path(manifest_path) if manifest_path is not None else None)


def _private_manifest(path: Path | None, record_ids: Sequence[str]) -> dict[str, Any]:
    if path is None:
        return {"record_ids": list(record_ids), "styles": [None] * len(record_ids), "source_present": False}
    value = _load_json(path, "private manifest")
    if not isinstance(value, Mapping):
        raise _error("private manifest is not an object")
    rows = value.get("records")
    if isinstance(rows, list):
        ids: list[str] = []
        styles: list[str | None] = []
        for number, row in enumerate(rows, 1):
            if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str):
                raise _error(f"private manifest row {number} is invalid")
            ids.append(str(row["record_id"]))
            style = row.get("style")
            styles.append(style if isinstance(style, str) else None)
    else:
        order = value.get("record_order")
        if not isinstance(order, list) or not all(isinstance(item, str) for item in order):
            raise _error("private manifest record order is missing")
        ids = [str(item) for item in order]
        styles_value = value.get("styles")
        if styles_value is None:
            styles = [None] * len(ids)
        elif isinstance(styles_value, list) and len(styles_value) == len(ids) and all(isinstance(item, str) for item in styles_value):
            styles = [str(item) for item in styles_value]
        else:
            raise _error("private manifest style geometry changed")
    if ids != list(record_ids) or len(set(ids)) != len(ids):
        raise _error("private manifest opaque ID order or coverage changed")
    if len(styles) != len(ids):
        raise _error("private manifest style geometry changed")
    return {"record_ids": ids, "styles": styles, "source_present": any(row is not None for row in styles)}


def _load_truth(truth_path: Path, record_ids: Sequence[str], geometry: Mapping[str, Any], model: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    tensors, metadata = _load_safetensor(truth_path, "private truth")
    if set(tensors) != {"input_ids"}:
        raise _error("private truth tensor fields changed")
    if metadata:
        if metadata.get("schema") not in (None, PRIVATE_TRUTH_SCHEMA):
            raise _error("private truth schema changed")
        if metadata.get("task_id") not in (None, TASK_ID):
            raise _error("private truth task identity changed")
        if metadata.get("truth_opened") not in (None, "false"):
            raise _error("private truth truth-state metadata changed")
        if metadata.get("source_truth_included") not in (None, "true"):
            raise _error("private truth source-state metadata changed")
    truth = tensors["input_ids"]
    expected_shape = (len(record_ids), int(geometry.get("sequence_tokens", SEQUENCE_TOKENS)))
    if tuple(truth.shape) != expected_shape or truth.dtype != torch.int64:
        raise _error("private truth shape or dtype changed")
    if not torch.isfinite(truth.to(torch.float32)).all().item():
        raise _error("private truth contains non-finite values")
    vocab_size = _integer(model.get("vocab_size"), "model vocab_size")
    bos_token_id = _integer(model.get("bos_token_id"), "model bos_token_id")
    if bool((truth < 0).any().item()) or bool((truth >= vocab_size).any().item()):
        raise _error("private truth token is outside vocabulary")
    if bool((truth[:, 0] != bos_token_id).any().item()):
        raise _error("private truth BOS token changed")
    return truth, metadata


def _aggregate_metrics(correct: torch.Tensor, record_ids: Sequence[str], styles: Sequence[str | None]) -> dict[str, Any]:
    if correct.ndim != 2:
        raise _error("score correctness shape is invalid")
    records, scored_tokens = correct.shape
    correct_tokens = int(correct.sum().item())
    exact_records_mask = correct.all(dim=1)
    first_errors: list[int | None] = []
    for row in correct:
        failures = torch.nonzero(~row, as_tuple=False)
        first_errors.append(None if failures.numel() == 0 else int(failures[0].item()) + 1)
    per_position = [
        {
            "position": int(position + 1),
            "correct_tokens": int(correct[:, position].sum().item()),
            "scored_tokens": int(records),
            "token_accuracy": float(correct[:, position].float().mean().item()),
        }
        for position in range(scored_tokens)
    ]
    per_style: dict[str, dict[str, Any]] = {}
    for style in sorted({item for item in styles if item is not None}):
        selected = torch.tensor([item == style for item in styles], dtype=torch.bool)
        subset = correct[selected]
        if subset.numel() == 0:
            continue
        per_style[str(style)] = {
            "records": int(subset.shape[0]),
            "correct_tokens": int(subset.sum().item()),
            "scored_tokens": int(subset.numel()),
            "token_accuracy": float(subset.float().mean().item()),
            "records_allcorrect": int(subset.all(dim=1).sum().item()),
            "exact_record_rate": float(subset.all(dim=1).float().mean().item()),
        }
    return {
        "records": int(records),
        "record_ids": list(record_ids),
        "correct_tokens": correct_tokens,
        "scored_tokens": int(correct.numel()),
        "token_accuracy": float(correct_tokens / correct.numel()),
        "records_allcorrect": int(exact_records_mask.sum().item()),
        "exact_record_rate": float(exact_records_mask.float().mean().item()),
        "first_error_position": first_errors,
        "per_position": per_position,
        "per_style": per_style,
    }


def score_frozen(
    public_dir: os.PathLike[str] | str,
    predictions_path: os.PathLike[str] | str,
    receipt_path: os.PathLike[str] | str,
    truth_path: os.PathLike[str] | str,
    *,
    evidence_path: os.PathLike[str] | str | None = None,
    private_manifest_path: os.PathLike[str] | str | None = None,
    condition: str | None = None,
    output_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Verify a frozen arm, then open private truth and compute exact metrics."""

    # This call must remain before even resolving/opening the private truth or
    # its manifest.  Hidden tamper tests rely on this ordering.
    verified = validate_frozen(public_dir, predictions_path, receipt_path, evidence_path)
    snapshot = verified["snapshot"]
    truth_file, manifest_file = _truth_paths(truth_path, private_manifest_path)
    truth_file = _regular_file(truth_file, "private truth")
    manifest = _private_manifest(manifest_file, snapshot["public"]["record_ids"])
    truth, truth_metadata = _load_truth(
        truth_file,
        snapshot["public"]["record_ids"],
        snapshot["public"]["geometry"],
        snapshot["public"]["model"],
    )
    truth_record = _file_record(truth_file)
    manifest_record = _file_record(manifest_file) if manifest_file is not None else None
    styles = manifest["styles"]
    scored_arms: list[dict[str, Any]] = []
    for arm in snapshot["prediction"]["arms"]:
        predictions = torch.tensor(arm["prediction_tokens"], dtype=torch.long)
        if tuple(predictions.shape) != tuple(truth.shape):
            raise _error(f"truth and prediction geometry disagree for {arm['tensor_key']}")
        if bool((predictions[:, 0] != int(snapshot["public"]["model"]["bos_token_id"])).any().item()):
            raise _error(f"prediction BOS changed for {arm['tensor_key']}")
        correct = predictions[:, 1:].eq(truth[:, 1:])
        metrics = _aggregate_metrics(correct, snapshot["public"]["record_ids"], styles)
        scored_arms.append(
            {
                "method": arm["method"],
                "metric": arm["metric"],
                "tensor_key": arm["tensor_key"],
                "metrics": metrics,
            }
        )

    result = {
        "schema": "token-reconstruction.trr-p01-score.v1",
        "task_id": TASK_ID,
        "status": "SCORED_AFTER_VERIFIED_FREEZE",
        "scored_utc": _common_utc_now(),
        "condition": condition,
        "truth_opened_after_freeze_verification": True,
        "freeze_verification": {
            "status": verified["status"],
            "truth_opened_before_validation": False,
            "receipt": _file_record(_path(receipt_path)),
        },
        "truth": {
            "artifact": truth_record,
            "private_manifest": manifest_record,
            "metadata": truth_metadata,
        },
        "geometry": snapshot["public"]["geometry"],
        "arms": scored_arms,
        "method_order": [arm["tensor_key"] for arm in scored_arms],
    }
    if output_path is not None:
        _write_json_exclusive(_path(output_path), result)
    return result


def score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Short alias retained for callers that use the CLI verb as an API."""

    return score_frozen(*args, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--public-dir", type=Path, required=True)
        subparser.add_argument("--predictions", type=Path, required=True)
        subparser.add_argument("--receipt", type=Path, required=True)
        subparser.add_argument("--evidence", type=Path)

    freeze = subparsers.add_parser("freeze", help="verify and create a receipt before truth")
    add_shared(freeze)

    validate = subparsers.add_parser("validate", help="verify a receipt without opening truth")
    add_shared(validate)

    scored = subparsers.add_parser("score", help="verify the receipt, then score private truth")
    add_shared(scored)
    scored.add_argument("--truth", type=Path, required=True)
    scored.add_argument("--private-manifest", type=Path)
    scored.add_argument("--condition")
    scored.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_predictions(args.public_dir, args.predictions, args.receipt, args.evidence)
        elif args.command == "validate":
            result = validate_frozen(args.public_dir, args.predictions, args.receipt, args.evidence)
        else:
            result = score_frozen(
                args.public_dir,
                args.predictions,
                args.receipt,
                args.truth,
                evidence_path=args.evidence,
                private_manifest_path=args.private_manifest,
                condition=args.condition,
                output_path=args.output,
            )
    except (FreezeScoreError, PilotError) as exc:
        print(f"freeze_score: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result.get("status"), "arms": len(result.get("prediction", result).get("arms", [])) if isinstance(result.get("prediction", result), Mapping) else None}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
