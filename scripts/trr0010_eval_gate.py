"""Truth-blind public-output gate for TRR-0010.

This is a small task-local port of the TRR-0009 pre-truth boundary.  It
deliberately validates only public registration, input/state/code bindings,
prediction files, and timing receipts.  There is no truth loader, tokenizer,
source reader, or scorer in this module.

The six-method matrix is frozen by the task design: the unchanged shared
starting state, four crossed arms, and one historical A1+A2 comparator.  The
adapter keeps that matrix explicit so a missing, duplicate, or foreign
prediction cannot be mistaken for a complete run.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import torch
from safetensors import safe_open


TASK_ID = "TRR-0010"
REGISTRATION_SCHEMA = "token-reconstruction.trr0010-evaluation-registration.v1"
RUN_SCHEMA = "token-reconstruction.trr0010-prediction-run.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0010-prediction.v1"
TIMING_SCHEMA = "token-reconstruction.trr0010-prediction-timing.v1"
OBSERVATION_SCHEMA = "token-reconstruction.trr0010-public-observation.v1"
FREEZE_SCHEMA = "token-reconstruction.trr0010-public-freeze.v1"

REGISTRATION_STATUS = "FROZEN_EVALUATION_REGISTRATION_BEFORE_TRUTH"
RUN_STATUS = "PUBLIC_PREDICTIONS_COMPLETE_BEFORE_TRUTH"
FREEZE_STATUS = "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH"

# The baseline name follows the v4 proposal's declared contrast vocabulary.
# A1+A2 is one retained historical comparator, never two methods.
UNCHANGED_METHOD_ID = "unchanged_shared_start"
CURRENT_FIXED_METHOD_ID = "current_fixed"
CURRENT_DIRECTIONAL_METHOD_ID = "current_directional"
EXPANDED_FIXED_METHOD_ID = "expanded_fixed"
EXPANDED_DIRECTIONAL_METHOD_ID = "expanded_directional"
A1_A2_METHOD_ID = "frozen_a1_a2_k256"
METHOD_ORDER = (
    UNCHANGED_METHOD_ID,
    CURRENT_FIXED_METHOD_ID,
    CURRENT_DIRECTIONAL_METHOD_ID,
    EXPANDED_FIXED_METHOD_ID,
    EXPANDED_DIRECTIONAL_METHOD_ID,
    A1_A2_METHOD_ID,
)
METHOD_ROLES = {
    UNCHANGED_METHOD_ID: "unchanged_shared_starting_reference",
    CURRENT_FIXED_METHOD_ID: "current_fixed",
    CURRENT_DIRECTIONAL_METHOD_ID: "current_directional",
    EXPANDED_FIXED_METHOD_ID: "expanded_fixed",
    EXPANDED_DIRECTIONAL_METHOD_ID: "expanded_directional",
    A1_A2_METHOD_ID: "historical_a1_a2_comparator",
}

CELL_ORDER = (
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
)
DOMAIN_ORDER = ("pile", "finance")
TARGET_ORDER = ("public_base", "public_lora_2601")
RECORDS_BY_DOMAIN = {"finance": 128, "pile": 128}
RECORDS_PER_CELL = 128

STORED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
STATIC_GEOMETRY = {
    "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
    "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
    "vocabulary_size": VOCABULARY_SIZE,
    "bos_token_id": BOS_TOKEN_ID,
}
LOADER_INTERFACE = "trr0010.current_h.full_vocabulary.v1"
A1_A2_LOADER_INTERFACE = "trr0010.a1_a2.reconstructed_prefix_k256.v1"
A1_A2_REQUIRED_RESOURCES = ("public_embedding_table", "retained_a1_lens", "public_p0_prefix")
OBSERVATION_KEYS = {"activations", "attention_mask", "position_ids"}
# Cut-4 public H is [records, 128, 2048].  Synthetic tests explicitly
# monkeypatch this constant to a small width so they exercise the same header
# checks without allocating production-sized activation payloads.
OBSERVATION_HIDDEN_SIZE = 2048

# Every deployed direct method binds the immutable public E and its decoder
# state.  Directional methods deploy the separately exported effective W in
# place of E, while still binding the decoder state.  These records are kept
# in the method resource map rather than inferred from the row's generic
# state field, so a loader cannot silently substitute an unregistered asset.
DIRECT_METHOD_IDS = (
    UNCHANGED_METHOD_ID,
    CURRENT_FIXED_METHOD_ID,
    EXPANDED_FIXED_METHOD_ID,
)
DIRECTIONAL_METHOD_IDS = (
    CURRENT_DIRECTIONAL_METHOD_ID,
    EXPANDED_DIRECTIONAL_METHOD_ID,
)
METHOD_RESOURCE_REQUIREMENTS = {
    **{method_id: ("public_embedding_table", "base_decoder_state") for method_id in DIRECT_METHOD_IDS},
    **{method_id: ("effective_readout_w", "base_decoder_state") for method_id in DIRECTIONAL_METHOD_IDS},
    A1_A2_METHOD_ID: A1_A2_REQUIRED_RESOURCES,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FALSE_FLAGS = (
    "truth_opened",
    "source_text_loaded",
    "target_labels_loaded",
    "source_text_written",
    "token_ids_written",
    "candidate_arrays_persisted",
)


class GateError(ValueError):
    """Raised when a public TRR-0010 binding is incomplete or changed."""


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise GateError(f"repository root is unavailable: {root}")
    return root


def _within(path: Path, root: Path, *, description: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GateError(f"{description} is outside repository root") from exc
    return path


def _resolve_file(value: Any, *, root: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GateError(f"{description} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.is_symlink():
        raise GateError(f"{description} is a symlink: {path}")
    path = path.resolve()
    _within(path, root, description=description)
    if not path.is_file():
        raise GateError(f"{description} is unavailable: {path}")
    return path


def _resolve_dir(value: Any, *, root: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GateError(f"{description} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.is_symlink():
        raise GateError(f"{description} is a symlink: {path}")
    path = path.resolve()
    _within(path, root, description=description)
    if not path.is_dir():
        raise GateError(f"{description} is unavailable: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def _record(value: Any, *, root: Path, description: str) -> dict[str, Any]:
    """Validate a hashed file, including explicitly marked shared read-only assets.

    Public E/lens/prefix files and HF cache paths can live outside this
    worktree.  They remain fail-closed: an external or symlinked path must
    carry ``readonly=true`` and its bytes and SHA-256 are checked every time.
    Task-owned outputs continue to use ``_resolve_file`` and must remain under
    the task root.
    """
    if not isinstance(value, Mapping):
        raise GateError(f"{description} binding is malformed")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise GateError(f"{description} path is absent")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    readonly = value.get("readonly") is True
    if path.is_symlink() and not readonly:
        raise GateError(f"{description} symlink must be explicitly readonly")
    path = path.resolve()
    try:
        path.relative_to(root)
        inside_root = True
    except ValueError:
        inside_root = False
    if not inside_root and not readonly:
        raise GateError(f"{description} external path must be explicitly readonly")
    if not path.is_file():
        raise GateError(f"{description} is unavailable: {path}")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise GateError(f"{description} byte count is malformed")
    digest = value.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise GateError(f"{description} hash is malformed")
    checked = {"path": str(path), "bytes": int(size), "sha256": digest}
    if readonly:
        checked["readonly"] = True
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise GateError(f"{description} hash or size changed")
    return checked


def file_record(path: Path, *, root: Path, readonly: bool = False) -> dict[str, Any]:
    """Create a canonical file record, optionally for a shared read-only asset."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file() and not path.is_symlink():
        raise GateError(f"file is unavailable: {path}")
    resolved = path.resolve()
    binding: dict[str, Any] = {
        "path": str(path),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if readonly:
        binding["readonly"] = True
    return _record(binding, root=root, description="file")


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{description} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise GateError(f"{description} must be an object")
    return payload


def _recorded_json(value: Any, *, root: Path, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    checked = _record(value, root=root, description=description)
    payload = _load_json(Path(checked["path"]), description=description)
    _require_truth_free(payload, description=description)
    return checked, payload


def _validate_observation_bindings(
    bindings: Any,
    *,
    selection_payload: Mapping[str, Any],
    root: Path,
    description: str,
) -> dict[str, Any]:
    """Bind each actual activation file and its source-record order digest.

    A selection JSON digest by itself is insufficient: every cell must also
    bind the hashed activation payload and carry the same ordered source-ID
    digest.  Safetensor metadata is checked without materializing activations.
    """
    if not isinstance(bindings, Mapping) or set(bindings) != set(CELL_ORDER):
        raise GateError(f"{description} cell bindings are incomplete or foreign")
    selection_ids = selection_payload.get("record_ids_sha256")
    if not isinstance(selection_ids, Mapping) or set(selection_ids) != set(DOMAIN_ORDER):
        raise GateError("source selection record-ID order digests are absent")
    for domain in DOMAIN_ORDER:
        if not isinstance(selection_ids.get(domain), str) or _SHA256.fullmatch(selection_ids[domain]) is None:
            raise GateError(f"source selection record-ID digest is malformed: {domain}")
    checked: dict[str, Any] = {}
    for cell_id in CELL_ORDER:
        row = bindings[cell_id]
        if not isinstance(row, Mapping) or row.get("cell_id") != cell_id or row.get("records") != RECORDS_PER_CELL:
            raise GateError(f"{description} row identity changed: {cell_id}")
        domain = cell_id.split("__", 1)[0]
        record_digest = row.get("record_ids_sha256")
        if not isinstance(record_digest, str) or _SHA256.fullmatch(record_digest) is None:
            raise GateError(f"{description} record-ID digest is malformed: {cell_id}")
        if record_digest != selection_ids[domain]:
            raise GateError(f"{description} record-ID order differs from source selection: {cell_id}")
        shape = row.get("shape")
        expected_shape_values = [RECORDS_PER_CELL, STORED_SEQUENCE_TOKENS, OBSERVATION_HIDDEN_SIZE]
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
            or shape != expected_shape_values
        ):
            raise GateError(f"{description} activation geometry changed: {cell_id}")
        observation = row.get("observation")
        checked_observation = _record(observation, root=root, description=f"{description} observation {cell_id}")
        try:
            with safe_open(checked_observation["path"], framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
                keys = set(handle.keys())
                tensor_headers = {
                    key: {
                        "shape": list(handle.get_slice(key).get_shape()),
                        "dtype": str(handle.get_slice(key).get_dtype()),
                    }
                    for key in OBSERVATION_KEYS
                }
        except Exception as exc:
            raise GateError(f"{description} observation payload is unreadable: {cell_id}") from exc
        if keys != OBSERVATION_KEYS:
            raise GateError(f"{description} observation tensor keys changed: {cell_id}")
        expected_headers = {
            "activations": {
                "shape": expected_shape_values,
                "dtype": "BF16",
            },
            "attention_mask": {
                "shape": [RECORDS_PER_CELL, STORED_SEQUENCE_TOKENS],
                "dtype": "U8",
            },
            "position_ids": {
                "shape": [RECORDS_PER_CELL, STORED_SEQUENCE_TOKENS],
                "dtype": "I64",
            },
        }
        for key, expected_header in expected_headers.items():
            actual_header = tensor_headers.get(key)
            if actual_header is None or actual_header["shape"] != expected_header["shape"]:
                raise GateError(f"{description} observation tensor header geometry changed: {cell_id}/{key}")
            if actual_header["dtype"] != expected_header["dtype"]:
                raise GateError(f"{description} observation tensor header dtype changed: {cell_id}/{key}")
        expected_shape = json.dumps(expected_shape_values)
        expected_metadata = {
            "schema": OBSERVATION_SCHEMA,
            "task_id": TASK_ID,
            "cell_id": cell_id,
            "records": str(RECORDS_PER_CELL),
            "shape": expected_shape,
            "record_ids_sha256": record_digest,
            "truth_opened": "false",
            "source_text_written": "false",
            "token_ids_written": "false",
            "target_labels_loaded": "false",
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise GateError(f"{description} observation metadata changed: {cell_id}/{key}")
        checked[cell_id] = {
            "cell_id": cell_id,
            "records": RECORDS_PER_CELL,
            "shape": list(shape),
            "record_ids_sha256": record_digest,
            "observation": checked_observation,
        }
    return checked


def _same_record(left: Mapping[str, Any], right: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(left.get(key)) != str(right.get(key)):
            raise GateError(f"{description} {key} binding changed")


def _same_record_map(
    left: Mapping[str, Any], right: Mapping[str, Any], *, root: Path, description: str
) -> dict[str, Any]:
    if set(left) != set(right):
        raise GateError(f"{description} keys changed")
    checked: dict[str, Any] = {}
    for key in sorted(right):
        actual = _record(left[key], root=root, description=f"{description}.{key}")
        expected = _record(right[key], root=root, description=f"{description}.{key} expected")
        _same_record(actual, expected, description=f"{description}.{key}")
        checked[str(key)] = expected
    return checked


def _require_truth_free(value: Mapping[str, Any], *, description: str) -> None:
    for key in _FALSE_FLAGS:
        raw = value.get(key)
        if raw is True or (isinstance(raw, str) and raw.lower() == "true"):
            raise GateError(f"{description} records forbidden truth/source access")


def _require_false_flags(value: Mapping[str, Any], *, description: str) -> None:
    _require_truth_free(value, description=description)
    for key in _FALSE_FLAGS:
        if value.get(key) is not False:
            raise GateError(f"{description} must explicitly bind {key}=false")


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GateError("cannot resolve current code commit") from exc
    if _COMMIT.fullmatch(value) is None:
        raise GateError("current code commit is not a full hash")
    return value


def _expected_key(method_id: str, cell_id: str) -> str:
    return f"{method_id}::{cell_id}"


def _expected_prediction_path(output_root: Path, *, method_id: str, cell_id: str) -> Path:
    style, condition = cell_id.split("__", 1)
    return output_root / "predictions" / style / condition / f"{method_id}.safetensors"


def _expected_timing_path(output_root: Path, *, method_id: str, cell_id: str) -> Path:
    return output_root / "timings" / cell_id.split("__", 1)[0] / cell_id.split("__", 1)[1] / f"{method_id}.run.json"


def _validate_registration(registration: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    if registration.get("schema") != REGISTRATION_SCHEMA or registration.get("task_id") != TASK_ID:
        raise GateError("registration schema or task identity changed")
    if registration.get("status") != REGISTRATION_STATUS:
        raise GateError("registration is not frozen before truth")
    _require_false_flags(registration, description="registration")
    if list(registration.get("method_order", ())) != list(METHOD_ORDER):
        raise GateError("registration method matrix changed")
    if list(registration.get("cell_order", ())) != list(CELL_ORDER):
        raise GateError("registration cell order changed")
    counts = registration.get("records_by_domain")
    if not isinstance(counts, Mapping) or dict(counts) != RECORDS_BY_DOMAIN:
        raise GateError("registration domain counts changed")
    geometry = registration.get("geometry")
    if not isinstance(geometry, Mapping) or dict(geometry) != STATIC_GEOMETRY:
        raise GateError("registration geometry changed")
    code_commit = registration.get("code_commit")
    if not isinstance(code_commit, str) or _COMMIT.fullmatch(code_commit) is None:
        raise GateError("registration code commit is malformed")
    if registration.get("initialization_equivalence") != {"status": "PASS"}:
        raise GateError("initialization/harness output equivalence is missing or failed")

    task_root = (root / "experiments" / TASK_ID).resolve()
    output_root = _resolve_dir(registration.get("output_root"), root=root, description="prediction root")
    try:
        output_root.relative_to(task_root)
    except ValueError as exc:
        raise GateError("prediction root is outside the TRR-0010 task root") from exc

    contract_binding = _record(registration.get("contract_binding"), root=root, description="contract")
    input_bindings = registration.get("input_bindings")
    if not isinstance(input_bindings, Mapping) or not input_bindings:
        raise GateError("registration input bindings are absent")
    checked_inputs: dict[str, Any] = {}
    input_payloads: dict[str, dict[str, Any]] = {}
    for name, binding in input_bindings.items():
        checked, payload = _recorded_json(binding, root=root, description=f"input {name}")
        checked_inputs[str(name)] = checked
        input_payloads[str(name)] = payload
    selection_payload = input_payloads.get("source_selection")
    if not isinstance(selection_payload, Mapping):
        raise GateError("source_selection input binding is absent")
    observation_bindings = _validate_observation_bindings(
        registration.get("observation_bindings"),
        selection_payload=selection_payload,
        root=root,
        description="registration",
    )

    timing_plan = _record(registration.get("timing_plan"), root=root, description="timing plan")
    code_bindings = registration.get("code_bindings")
    if not isinstance(code_bindings, Mapping) or not code_bindings:
        raise GateError("registration code bindings are absent")
    checked_code: dict[str, Any] = {}
    for name, binding in code_bindings.items():
        checked_code[str(name)] = _record(binding, root=root, description=f"code {name}")

    rows = registration.get("methods")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise GateError("registration methods are absent")
    if len(rows) != len(METHOD_ORDER):
        raise GateError("registration method rows are duplicated or incomplete")
    by_id: dict[str, Mapping[str, Any]] = {}
    checked_states: dict[str, Any] = {}
    checked_resources: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise GateError("registration method row is malformed")
        method_id = str(row["id"])
        if method_id in by_id or method_id not in METHOD_ORDER:
            raise GateError("registration method rows contain duplicate or foreign method")
        by_id[method_id] = row
        if row.get("role") != METHOD_ROLES[method_id] or list(row.get("cells", ())) != list(CELL_ORDER):
            raise GateError(f"registration method role/coverage changed: {method_id}")
        counts = row.get("records_per_cell")
        if not isinstance(counts, Mapping) or dict(counts) != {cell: RECORDS_PER_CELL for cell in CELL_ORDER}:
            raise GateError(f"registration method counts changed: {method_id}")
        loader = row.get("loader")
        if not isinstance(loader, Mapping):
            raise GateError(f"registration loader is absent: {method_id}")
        checked_state = _record(row.get("state"), root=root, description=f"state {method_id}")
        if method_id == A1_A2_METHOD_ID:
            if loader.get("interface") != A1_A2_LOADER_INTERFACE:
                raise GateError(f"registration A1+A2 loader interface changed: {method_id}")
            expected_semantics = {
                "current_h_only": False,
                "full_vocabulary": True,
                "history_enabled": False,
                "a2_enabled": True,
                "reconstructed_prefix": True,
                "candidate_k": 256,
            }
            for name, expected in expected_semantics.items():
                actual = loader.get(name)
                if isinstance(expected, bool):
                    matches = actual is expected
                else:
                    matches = actual == expected
                if not matches:
                    raise GateError(f"registration A1+A2 loader semantics changed: {method_id}/{name}")
        else:
            if loader.get("interface") != LOADER_INTERFACE:
                raise GateError(f"registration loader interface changed: {method_id}")
            for flag in ("current_h_only", "full_vocabulary", "history_enabled", "a2_enabled"):
                expected = flag in ("current_h_only", "full_vocabulary")
                if loader.get(flag) is not expected:
                    raise GateError(f"registration loader semantics changed: {method_id}/{flag}")

        required_resources = METHOD_RESOURCE_REQUIREMENTS[method_id]
        resources = row.get("resources")
        if not isinstance(resources, Mapping) or not set(required_resources).issubset(resources):
            if method_id == A1_A2_METHOD_ID:
                raise GateError("registration A1+A2 E/lens/public-prefix resources are incomplete")
            resource_label = "public E and base/decoder state" if method_id in DIRECT_METHOD_IDS else "effective W and base/decoder state"
            raise GateError(f"registration {method_id} {resource_label} resources are incomplete")
        checked_resources[method_id] = {
            name: _record(resources[name], root=root, description=f"{method_id} resource {name}")
            for name in required_resources
        }
        if "base_decoder_state" in required_resources:
            _same_record(
                checked_resources[method_id]["base_decoder_state"],
                checked_state,
                description=f"{method_id} base/decoder state",
            )
        checked_states[method_id] = checked_state
    if set(by_id) != set(METHOD_ORDER):
        raise GateError("registration method matrix is incomplete")

    return {
        "registration": dict(registration),
        "output_root": str(output_root),
        "contract_binding": contract_binding,
        "input_bindings": checked_inputs,
        "timing_plan": timing_plan,
        "code_bindings": checked_code,
        "state_bindings": checked_states,
        "method_resources": checked_resources,
        "observation_bindings": observation_bindings,
    }


def _load_prediction(binding: Mapping[str, Any], *, root: Path, registration_sha: str, method_id: str, cell_id: str) -> dict[str, Any]:
    checked = _record(binding, root=root, description=f"prediction {method_id}/{cell_id}")
    expected_metadata = {
        "schema": PREDICTION_SCHEMA,
        "task_id": TASK_ID,
        "registration_sha256": registration_sha,
        "method_id": method_id,
        "cell_id": cell_id,
        "records": str(RECORDS_PER_CELL),
        "truth_opened": "false",
        "candidate_arrays_persisted": "false",
        "geometry_json": json.dumps({"records": RECORDS_PER_CELL, **STATIC_GEOMETRY}, sort_keys=True),
    }
    try:
        with safe_open(checked["path"], framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            if set(handle.keys()) != {"predictions"}:
                raise GateError(f"prediction tensor keys changed: {method_id}/{cell_id}")
            raw_values = handle.get_tensor("predictions")
            if raw_values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise GateError(f"prediction dtype is not a signed integer: {method_id}/{cell_id}")
            values = raw_values.detach().cpu().contiguous()
    except GateError:
        raise
    except Exception as exc:
        raise GateError(f"prediction artifact is unreadable: {method_id}/{cell_id}") from exc
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise GateError(f"prediction metadata changed: {method_id}/{cell_id}/{key}")
    if tuple(values.shape) != (RECORDS_PER_CELL, STORED_SEQUENCE_TOKENS):
        raise GateError(f"prediction geometry changed: {method_id}/{cell_id}")
    if values[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise GateError(f"prediction BOS column changed: {method_id}/{cell_id}")
    if values.lt(0).any().item() or values.ge(VOCABULARY_SIZE).any().item():
        raise GateError(f"prediction contains incomplete or invalid token IDs: {method_id}/{cell_id}")
    return checked | {"prediction_sha256": tensor_digest(values), "metadata": metadata}


def _load_timing(binding: Mapping[str, Any], *, root: Path, method_id: str, cell_id: str, prediction: Mapping[str, Any], registration_sha: str) -> dict[str, Any]:
    checked = _record(binding, root=root, description=f"timing {method_id}/{cell_id}")
    payload = _load_json(Path(checked["path"]), description=f"timing {method_id}/{cell_id}")
    _require_false_flags(payload, description=f"timing {method_id}/{cell_id}")
    if payload.get("schema") != TIMING_SCHEMA or payload.get("task_id") != TASK_ID:
        raise GateError(f"timing identity changed: {method_id}/{cell_id}")
    if payload.get("method_id") != method_id or payload.get("cell_id") != cell_id or payload.get("records") != RECORDS_PER_CELL:
        raise GateError(f"timing method/cell/record binding changed: {method_id}/{cell_id}")
    if payload.get("registration_sha256") != registration_sha:
        raise GateError(f"timing registration binding changed: {method_id}/{cell_id}")
    if payload.get("warmup_runs_per_record") != 1 or payload.get("measured_runs_per_record") != 1:
        raise GateError(f"timing warmup/measured count changed: {method_id}/{cell_id}")
    if payload.get("warmup_output_exact_match_measured") is not True or payload.get("measured_output_selected") is not True:
        raise GateError(f"timing output equivalence missing: {method_id}/{cell_id}")
    durations = payload.get("per_record_measured_seconds")
    if not isinstance(durations, list) or len(durations) != RECORDS_PER_CELL:
        raise GateError(f"timing per-record receipt is malformed: {method_id}/{cell_id}")
    try:
        for index, value in enumerate(durations):
            seconds = float(value)
            if not math.isfinite(seconds) or seconds < 0.0:
                raise GateError(f"timing duration is not finite and nonnegative: {method_id}/{cell_id}/{index}")
        for name in ("warmup_seconds_sum", "measured_seconds_sum", "model_preparation_seconds"):
            seconds = float(payload[name])
            if not math.isfinite(seconds) or seconds < 0.0:
                raise GateError(f"timing cost is not finite and nonnegative: {method_id}/{cell_id}/{name}")
    except GateError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError(f"timing cost is malformed: {method_id}/{cell_id}") from exc
    peak = payload.get("peak_memory")
    if not isinstance(peak, Mapping):
        raise GateError(f"timing peak-memory receipt is incomplete: {method_id}/{cell_id}")
    try:
        process_rss = int(peak["process_max_rss_bytes"])
        host_available = int(peak["host_available_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError(f"timing peak-memory receipt is malformed: {method_id}/{cell_id}") from exc
    if process_rss < 0 or host_available < 0:
        raise GateError(f"timing peak-memory receipt is negative: {method_id}/{cell_id}")
    artifact = payload.get("prediction_artifact")
    if not isinstance(artifact, Mapping):
        raise GateError(f"timing prediction artifact is absent: {method_id}/{cell_id}")
    checked_artifact = _record(artifact, root=root, description=f"timing prediction {method_id}/{cell_id}")
    _same_record(checked_artifact, prediction, description=f"timing/prediction {method_id}/{cell_id}")
    if artifact.get("prediction_sha256") != prediction.get("prediction_sha256"):
        raise GateError(f"timing prediction tensor binding changed: {method_id}/{cell_id}")
    return checked | {"payload": payload}


def _revalidate_runtime_records(value: Any, *, root: Path, description: str) -> None:
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            _record(value, root=root, description=description)
            return
        for key, nested in value.items():
            _revalidate_runtime_records(nested, root=root, description=f"{description}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _revalidate_runtime_records(nested, root=root, description=f"{description}[{index}]")

def validate_public_outputs(
    *,
    registration_path: Path,
    run_manifest_path: Path,
    repository_root: Path,
    require_current_head: bool = False,
) -> dict[str, Any]:
    """Validate the complete six-method public matrix without opening truth."""
    root = _root(repository_root)
    registration_path = _resolve_file(str(registration_path), root=root, description="registration")
    run_manifest_path = _resolve_file(str(run_manifest_path), root=root, description="run manifest")
    registration = _load_json(registration_path, description="registration")
    checked = _validate_registration(registration, root=root)
    if require_current_head and _git_head(root) != registration["code_commit"]:
        raise GateError("current code commit differs from frozen registration")
    registration_record = {"path": str(registration_path), "bytes": registration_path.stat().st_size, "sha256": sha256_file(registration_path)}
    run = _load_json(run_manifest_path, description="run manifest")
    if run.get("schema") != RUN_SCHEMA or run.get("task_id") != TASK_ID or run.get("status") != RUN_STATUS:
        raise GateError("run manifest identity or status changed")
    _require_false_flags(run, description="run manifest")
    runtime_recheck = run.get("runtime_recheck")
    if isinstance(runtime_recheck, Mapping):
        _revalidate_runtime_records(runtime_recheck, root=root, description="runtime recheck")
    if run.get("code_commit") != registration.get("code_commit"):
        raise GateError("run manifest code binding changed")
    nested_registration = run.get("registration")
    if not isinstance(nested_registration, Mapping):
        raise GateError("run manifest registration binding is absent")
    _same_record(nested_registration, registration_record, description="run manifest registration")
    if not isinstance(run.get("input_bindings"), Mapping):
        raise GateError("run manifest input bindings are absent")
    run_inputs = _same_record_map(run["input_bindings"], registration["input_bindings"], root=root, description="run manifest inputs")
    run_selection_payload = _load_json(Path(run_inputs["source_selection"]["path"]), description="run source selection")
    run_observation_bindings = _validate_observation_bindings(
        run.get("observation_bindings"),
        selection_payload=run_selection_payload,
        root=root,
        description="run manifest",
    )
    if run_observation_bindings != checked["observation_bindings"]:
        raise GateError("run manifest observation payload/order bindings changed")
    if not isinstance(run.get("code_bindings"), Mapping):
        raise GateError("run manifest code bindings are absent")
    run_code = _same_record_map(run["code_bindings"], registration["code_bindings"], root=root, description="run manifest code")
    run_timing_plan = _record(run.get("timing_plan"), root=root, description="run timing plan")
    _same_record(run_timing_plan, checked["timing_plan"], description="run timing plan")
    output_root = Path(checked["output_root"]).resolve()
    if run_manifest_path != (output_root / "run_manifest.json").resolve():
        raise GateError("run manifest is from a different prediction root")
    predictions = run.get("predictions")
    timings = run.get("timings")
    if not isinstance(predictions, Mapping) or not isinstance(timings, Mapping):
        raise GateError("run manifest prediction/timing maps are absent")
    expected_keys = {_expected_key(method, cell) for method in METHOD_ORDER for cell in CELL_ORDER}
    if set(predictions) != expected_keys or set(timings) != expected_keys:
        raise GateError("run manifest method-by-cell matrix is incomplete, duplicated, or foreign")
    checked_predictions: dict[str, Any] = {}
    checked_timings: dict[str, Any] = {}
    for method_id in METHOD_ORDER:
        for cell_id in CELL_ORDER:
            key = _expected_key(method_id, cell_id)
            prediction_binding = predictions[key]
            timing_binding = timings[key]
            if not isinstance(prediction_binding, Mapping) or not isinstance(timing_binding, Mapping):
                raise GateError(f"run manifest artifact binding is malformed: {key}")
            expected_prediction = _expected_prediction_path(output_root, method_id=method_id, cell_id=cell_id).resolve()
            expected_timing = _expected_timing_path(output_root, method_id=method_id, cell_id=cell_id).resolve()
            if _resolve_file(prediction_binding.get("path"), root=root, description=f"prediction {key}") != expected_prediction:
                raise GateError(f"prediction path changed: {key}")
            if _resolve_file(timing_binding.get("path"), root=root, description=f"timing {key}") != expected_timing:
                raise GateError(f"timing path changed: {key}")
            prediction = _load_prediction(prediction_binding, root=root, registration_sha=registration_record["sha256"], method_id=method_id, cell_id=cell_id)
            timing = _load_timing(timing_binding, root=root, method_id=method_id, cell_id=cell_id, prediction=prediction, registration_sha=registration_record["sha256"])
            checked_predictions[key] = prediction
            checked_timings[key] = timing

    return {
        "schema": FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": FREEZE_STATUS,
        "registration": registration_record,
        "run_manifest": {"path": str(run_manifest_path), "bytes": run_manifest_path.stat().st_size, "sha256": sha256_file(run_manifest_path)},
        "contract_binding": checked["contract_binding"],
        "input_bindings": run_inputs,
        "timing_plan": checked["timing_plan"],
        "code_bindings": run_code,
        "state_bindings": checked["state_bindings"],
        "method_resources": checked["method_resources"],
        "observation_bindings": checked["observation_bindings"],
        "method_order": list(METHOD_ORDER),
        "cell_order": list(CELL_ORDER),
        "records_by_domain": dict(RECORDS_BY_DOMAIN),
        "code_commit": registration["code_commit"],
        "output_root": str(output_root),
        "initialization_equivalence": registration["initialization_equivalence"],
        "predictions": checked_predictions,
        "timings": checked_timings,
        "truth_opened": False,
        "source_text_written": False,
        "source_text_loaded": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }


def write_freeze(*, registration_path: Path, run_manifest_path: Path, output_path: Path, repository_root: Path, require_current_head: bool = False) -> dict[str, Any]:
    root = _root(repository_root)
    receipt = validate_public_outputs(
        registration_path=registration_path,
        run_manifest_path=run_manifest_path,
        repository_root=root,
        require_current_head=require_current_head,
    )
    output_path = Path(output_path).expanduser()
    if not output_path.is_absolute():
        output_path = root / output_path
    if output_path.is_symlink():
        raise GateError(f"refusing to overwrite freeze receipt: {output_path}")
    output_path = output_path.resolve()
    _within(output_path, root, description="freeze output")
    task_root = (root / "experiments" / TASK_ID).resolve()
    try:
        output_path.relative_to(task_root)
    except ValueError as exc:
        raise GateError("freeze output is outside the TRR-0010 task root") from exc
    if output_path.exists():
        raise GateError(f"refusing to overwrite freeze receipt: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise GateError(f"refusing to overwrite freeze receipt: {output_path}") from exc
    return receipt | {"freeze_receipt": {"path": str(output_path), "bytes": output_path.stat().st_size, "sha256": sha256_file(output_path)}}


def validate_before_truth(*, freeze_path: Path, repository_root: Path, require_current_head: bool = False) -> dict[str, Any]:
    """Revalidate every bound public artifact immediately before truth access."""
    root = _root(repository_root)
    freeze_path = _resolve_file(str(freeze_path), root=root, description="public freeze receipt")
    freeze = _load_json(freeze_path, description="public freeze receipt")
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("task_id") != TASK_ID or freeze.get("status") != FREEZE_STATUS:
        raise GateError("public freeze receipt identity/status changed")
    _require_false_flags(freeze, description="public freeze receipt")
    registration_record = _record(freeze.get("registration"), root=root, description="frozen registration")
    run_record = _record(freeze.get("run_manifest"), root=root, description="frozen run manifest")
    refreshed = validate_public_outputs(
        registration_path=Path(registration_record["path"]),
        run_manifest_path=Path(run_record["path"]),
        repository_root=root,
        require_current_head=require_current_head,
    )
    for key in (
        "registration",
        "run_manifest",
        "contract_binding",
        "input_bindings",
        "timing_plan",
        "code_bindings",
        "state_bindings",
        "method_resources",
        "observation_bindings",
        "method_order",
        "cell_order",
        "records_by_domain",
        "code_commit",
        "output_root",
        "initialization_equivalence",
        "predictions",
        "timings",
    ):
        if freeze.get(key) != refreshed.get(key):
            raise GateError(f"public freeze binding changed: {key}")
    return refreshed


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--registration", type=Path, required=True)
    validate.add_argument("--run-manifest", type=Path, required=True)
    validate.add_argument("--require-current-head", action="store_true")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--registration", type=Path, required=True)
    freeze.add_argument("--run-manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--require-current-head", action="store_true")
    before = sub.add_parser("validate-before-truth")
    before.add_argument("--freeze", type=Path, required=True)
    before.add_argument("--require-current-head", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = validate_public_outputs(
                registration_path=args.registration,
                run_manifest_path=args.run_manifest,
                repository_root=args.repository_root,
                require_current_head=args.require_current_head,
            )
        elif args.command == "freeze":
            result = write_freeze(
                registration_path=args.registration,
                run_manifest_path=args.run_manifest,
                output_path=args.output,
                repository_root=args.repository_root,
                require_current_head=args.require_current_head,
            )
        else:
            result = validate_before_truth(
                freeze_path=args.freeze,
                repository_root=args.repository_root,
                require_current_head=args.require_current_head,
            )
    except GateError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "PASS", "task_id": TASK_ID, "command": args.command}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
