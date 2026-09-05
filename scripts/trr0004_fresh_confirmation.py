#!/usr/bin/env python3
"""Prospective TRR-0004 confirmation panel and fail-closed evaluation adapter.

The adapter is deliberately small.  It validates a future public panel,
prediction artifacts, and the method/state/config/code bindings needed for a
confirmation run.  It does not select records, load a model, generate
observations, or open evaluator truth while the selection rule is prospective.

When a frozen panel exists, callers must validate all four cells and every
registered method before their truth loader is called.  The timing helper is
also kept here so the steady interval has one unambiguous definition: one
record's CPU activation is moved to the selected device, the method runs, and
the predicted IDs return to CPU.  Model/state/prefix loads and hashing belong
to separately recorded cold phases.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import sys
import time
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from token_reconstruction.dual_benchmark import validate_observations
from token_reconstruction.footing import (
    FootingError,
    external_file_record,
    file_record,
    make_binding,
    sha256_file,
)
from token_reconstruction.freeze import FreezeError, require_truth_open_allowed


TASK_ID = "TRR-0004"
PANEL_SCHEMA = "token-reconstruction.trr0004-fresh-confirmation-panel.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0004-fresh-confirmation-prediction.v1"
REGISTRATION_SCHEMA = "token-reconstruction.trr0004-fresh-confirmation-registration.v1"
TRUTH_BINDING_SCHEMA = "token-reconstruction.trr0004-fresh-confirmation-truth-binding.v1"
TRUTH_SIDECAR_SCHEMA = "token-reconstruction.trr0004-fresh-confirmation-truth-sidecar.v1"
PLAN_SCHEMA = "token-reconstruction.trr0004-fresh-confirmation-plan.v1"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
CUT_DEPTH = 4
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
PAD_TOKEN_ID = 128001
STYLE_ORDER = ("pile", "finance")
CONDITION_ORDER = ("public_base", "public_lora_2601")
RECORDS_PER_STYLE = 16
SEQUENCE_TOKENS = {"pile": 40, "finance": 128}

METHOD_SPECS: tuple[dict[str, str], ...] = (
    {
        "id": "historical_alpaca_a1",
        "track": "comparator",
        "candidate_policy": "forbidden",
        "rule": "retained standalone historical A1 direct argmax; top-k is diagnostic only",
    },
    {
        "id": "frozen_a1_a2_k256",
        "track": "comparator",
        "candidate_policy": "required",
        "rule": "retained A1 proposals scored by fixed public-prefix A2 K256",
    },
    {
        "id": "historical_affine_ce_no_vocab_bias",
        "track": "track_b",
        "candidate_policy": "forbidden",
        "rule": "fresh public-fit affine decoder without vocabulary bias; selected public-fit checkpoint step 1900",
    },
    {
        "id": "causal_h_attention128",
        "track": "track_b",
        "candidate_policy": "forbidden",
        "rule": "compact causal contextual decoder over activation history",
    },
    {
        "id": "positionwise_mlp256",
        "track": "track_b",
        "candidate_policy": "forbidden",
        "rule": "parameter-matched compact nonlinear decoder",
    },
)
METHOD_IDS = tuple(row["id"] for row in METHOD_SPECS)
CANDIDATE_POLICIES = {row["id"]: row["candidate_policy"] for row in METHOD_SPECS}
_METHOD_TRACKS = {row["id"]: row["track"] for row in METHOD_SPECS}
RUNTIME_ASSET_ROLES = (
    "public_embedding_table",
    "public_prefix_checkpoint",
    "public_prefix_config",
)


class ConfirmationError(FootingError):
    """Raised when a prospective or frozen TRR-0004 input is unsafe."""


@dataclass(frozen=True)
class FreshCell:
    """One public style/condition cell from a frozen confirmation panel."""

    cell_id: str
    style: str
    condition: str
    record_ids: tuple[str, ...]
    activations: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    observation_path: Path
    observation_sha256: str

    @property
    def records(self) -> int:
        return int(self.activations.shape[0])

    @property
    def sequence_tokens(self) -> int:
        return int(self.activations.shape[1])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.activations.shape)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_relative(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfirmationError(f"{description} path is absent")
    if "\\" in value:
        raise ConfirmationError(f"{description} path must use POSIX separators")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise ConfirmationError(f"{description} path is unsafe: {value}")
    return value


def _valid_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ConfirmationError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _safe_method_id(value: Any, *, description: str = "method ID") -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
    ):
        raise ConfirmationError(f"{description} is unsafe")
    return value


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConfirmationError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmationError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ConfirmationError(f"{description} root must be an object")
    return value


def _reject_private_panel_keys(value: Any, *, path: str = "panel") -> None:
    """Reject source-token/truth fields from the public panel descriptor."""

    forbidden = ("oracle", "token_ids", "input_ids", "labels", "source_text")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfirmationError(f"{path} keys must be strings")
            lowered = key.casefold().replace("-", "_")
            if any(fragment in lowered for fragment in forbidden):
                raise ConfirmationError(f"{path}.{key} is private source/evaluator state")
            _reject_private_panel_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private_panel_keys(child, path=f"{path}[{index}]")


def _cell_id(style: str, condition: str) -> str:
    return f"{style}__{condition}"


def expected_cell_ids() -> tuple[str, ...]:
    return tuple(_cell_id(style, condition) for style in STYLE_ORDER for condition in CONDITION_ORDER)


def _asset_path(
    descriptor: Mapping[str, Any],
    *,
    repository_root: Path,
    description: str,
) -> Path:
    relative = _safe_relative(descriptor.get("path"), description=description)
    path = repository_root.resolve() / relative
    if path.is_symlink() or not path.is_file():
        raise ConfirmationError(f"{description} is unavailable: {relative}")
    current = file_record(path, repository_root=repository_root.resolve())
    for key in ("path", "bytes", "sha256"):
        if descriptor.get(key) != current[key]:
            raise ConfirmationError(f"{description} hash or size changed: {relative}")
    return path


def _validate_mask_positions(
    row: Mapping[str, Any], *, records: int, sequence_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        mask = torch.tensor(row["attention_mask"], dtype=torch.long)
        positions = torch.tensor(row["position_ids"], dtype=torch.long)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfirmationError("panel mask or position fields are malformed") from exc
    expected_shape = (records, sequence_tokens)
    if tuple(mask.shape) != expected_shape or tuple(positions.shape) != expected_shape:
        raise ConfirmationError("panel mask or position geometry changed")
    if mask.lt(0).any().item() or mask.gt(1).any().item():
        raise ConfirmationError("panel attention mask is not binary")
    try:
        # The TRR3 validator is the shared causal/right-padding geometry rule.
        validate_observations(
            torch.zeros((records, sequence_tokens, HIDDEN_SIZE), dtype=torch.float32),
            mask,
            positions,
        )
    except Exception as exc:
        raise ConfirmationError(f"panel mask or position contract failed: {exc}") from exc
    return mask, positions


def _record_metadata(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    allowed = {"record_id", "public_record_sha256", "raw_index", "valid_tokens", "source_index"}
    if set(row) - allowed:
        raise ConfirmationError(f"panel record {index} contains unapproved fields")
    record_id = row.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ConfirmationError(f"panel record {index} has no public record ID")
    _valid_sha256(row.get("public_record_sha256"), description=f"panel record {index} hash")
    for key in ("raw_index", "valid_tokens", "source_index"):
        if key in row and (not isinstance(row[key], int) or row[key] < 0):
            raise ConfirmationError(f"panel record {index} has invalid {key}")
    # Return only stable public metadata used for pair equality checks.
    return {key: row[key] for key in sorted(row)}


def _validate_panel_identity(panel: Mapping[str, Any]) -> None:
    if panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != TASK_ID:
        raise ConfirmationError("confirmation panel identity changed")
    if panel.get("status") != "FROZEN_FRESH_CONFIRMATION_PANEL":
        raise ConfirmationError("confirmation panel is not frozen")
    if panel.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ConfirmationError("confirmation panel model identity changed")
    if panel.get("cut_depth") != CUT_DEPTH or panel.get("hidden_size") != HIDDEN_SIZE:
        raise ConfirmationError("confirmation panel cut or hidden-size geometry changed")
    if panel.get("source_material_included") is not False:
        raise ConfirmationError("confirmation panel includes source material")
    _valid_sha256(panel.get("selection_plan_sha256"), description="selection plan binding")
    capture = panel.get("observation_generation")
    if not isinstance(capture, Mapping):
        raise ConfirmationError("observation generation contract is absent")
    if (
        capture.get("batch_size") != 8
        or capture.get("sequence_tokens") != 192
        or capture.get("same_public_prefix_path") is not True
        or capture.get("path") != "public_prefix.forward_full"
    ):
        raise ConfirmationError("observation generation geometry or path changed")


def _validate_panel_specs(panel: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    styles = panel.get("styles")
    if not isinstance(styles, list) or [row.get("id") for row in styles if isinstance(row, Mapping)] != list(STYLE_ORDER):
        raise ConfirmationError("confirmation panel style order or completeness changed")
    style_specs: dict[str, Mapping[str, Any]] = {}
    for row in styles:
        if not isinstance(row, Mapping):
            raise ConfirmationError("confirmation panel style row is malformed")
        style = row.get("id")
        if style not in STYLE_ORDER or style in style_specs:
            raise ConfirmationError("confirmation panel style ID is invalid")
        if (
            row.get("records") != RECORDS_PER_STYLE
            or row.get("sequence_tokens") != SEQUENCE_TOKENS[str(style)]
            or row.get("hidden_size") != HIDDEN_SIZE
        ):
            raise ConfirmationError(f"confirmation panel geometry changed for {style}")
        style_specs[str(style)] = row
    conditions = panel.get("conditions")
    if not isinstance(conditions, list) or [row.get("id") for row in conditions if isinstance(row, Mapping)] != list(CONDITION_ORDER):
        raise ConfirmationError("confirmation panel condition order or completeness changed")
    seen: set[str] = set()
    for row in conditions:
        if not isinstance(row, Mapping):
            raise ConfirmationError("confirmation panel condition row is malformed")
        condition = row.get("id")
        if condition not in CONDITION_ORDER or condition in seen:
            raise ConfirmationError("confirmation panel condition ID is invalid")
        seen.add(str(condition))
        if condition == "public_base" and row.get("weights_available_to_reconstructor") is not True:
            raise ConfirmationError("public base availability declaration changed")
        if condition == "public_lora_2601" and row.get("weights_available_to_reconstructor") is not False:
            raise ConfirmationError("LoRA target availability declaration changed")
    return style_specs


def load_fresh_panel(path: Path, *, repository_root: Path) -> dict[str, Any]:
    """Validate a frozen public panel and its hashed observation descriptors."""

    panel = _load_json(path, description="confirmation panel")
    _reject_private_panel_keys(panel)
    _validate_panel_identity(panel)
    style_specs = _validate_panel_specs(panel)
    cells = panel.get("cells")
    if not isinstance(cells, list) or [row.get("id") for row in cells if isinstance(row, Mapping)] != list(expected_cell_ids()):
        raise ConfirmationError("confirmation panel cell order or completeness changed")

    records_by_style: dict[str, list[dict[str, Any]]] = {}
    masks_by_style: dict[str, torch.Tensor] = {}
    positions_by_style: dict[str, torch.Tensor] = {}
    for row in cells:
        if not isinstance(row, Mapping):
            raise ConfirmationError("confirmation panel cell is malformed")
        style = row.get("style")
        condition = row.get("condition")
        if style not in STYLE_ORDER or condition not in CONDITION_ORDER or row.get("id") != _cell_id(style, condition):
            raise ConfirmationError("confirmation panel cell identity changed")
        sequence_tokens = int(style_specs[str(style)]["sequence_tokens"])
        record_rows = row.get("records")
        if not isinstance(record_rows, list) or len(record_rows) != RECORDS_PER_STYLE:
            raise ConfirmationError(f"confirmation panel record count changed for {style}")
        records = [_record_metadata(value, index=index) for index, value in enumerate(record_rows) if isinstance(value, Mapping)]
        if len(records) != RECORDS_PER_STYLE:
            raise ConfirmationError(f"confirmation panel record rows are malformed for {style}")
        if len({value["record_id"] for value in records}) != RECORDS_PER_STYLE:
            raise ConfirmationError(f"confirmation panel record IDs are duplicated for {style}")
        mask, positions = _validate_mask_positions(row, records=RECORDS_PER_STYLE, sequence_tokens=sequence_tokens)
        if style not in records_by_style:
            records_by_style[str(style)] = records
            masks_by_style[str(style)] = mask
            positions_by_style[str(style)] = positions
        else:
            if records != records_by_style[str(style)]:
                raise ConfirmationError(f"paired {style} public records changed")
            if not torch.equal(mask, masks_by_style[str(style)]) or not torch.equal(positions, positions_by_style[str(style)]):
                raise ConfirmationError(f"paired {style} masks or positions changed")
        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise ConfirmationError(f"observation asset is absent for {row.get('id')}")
        if observation.get("tensor_key") != "activations" or observation.get("row_indices") != list(range(RECORDS_PER_STYLE)):
            raise ConfirmationError(f"observation row selection changed for {row.get('id')}")
        _asset_path(observation, repository_root=repository_root, description=f"observation {row.get('id')}")
        expected_geometry = {
            "records": RECORDS_PER_STYLE,
            "sequence_tokens": sequence_tokens,
            "hidden_size": HIDDEN_SIZE,
            "cut_depth": CUT_DEPTH,
        }
        if row.get("geometry") != expected_geometry:
            raise ConfirmationError(f"observation geometry declaration changed for {row.get('id')}")
        expected_role = "matched_public_control" if condition == "public_base" else "single_public_shift_diagnostic"
        if row.get("shift_role") != expected_role:
            raise ConfirmationError(f"condition role changed for {row.get('id')}")
    return panel


def load_fresh_cells(panel: Mapping[str, Any], *, repository_root: Path) -> tuple[FreshCell, ...]:
    """Load the already validated public observations for all four cells."""

    _reject_private_panel_keys(panel)
    _validate_panel_identity(panel)
    style_specs = _validate_panel_specs(panel)
    result: list[FreshCell] = []
    rows_by_id = {row.get("id"): row for row in panel.get("cells", []) if isinstance(row, Mapping)}
    for style in STYLE_ORDER:
        for condition in CONDITION_ORDER:
            cell_id = _cell_id(style, condition)
            row = rows_by_id.get(cell_id)
            if not isinstance(row, Mapping):
                raise ConfirmationError(f"confirmation cell is absent: {cell_id}")
            record_ids = tuple(str(value["record_id"]) for value in row["records"])
            sequence_tokens = int(style_specs[style]["sequence_tokens"])
            mask, positions = _validate_mask_positions(row, records=RECORDS_PER_STYLE, sequence_tokens=sequence_tokens)
            observation = row["observation"]
            observation_path = _asset_path(observation, repository_root=repository_root, description=f"observation {cell_id}")
            try:
                with safe_open(observation_path, framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != {"activations"}:
                        raise ConfirmationError(f"observation tensor fields changed for {cell_id}")
                    full = handle.get_tensor("activations").contiguous()
            except ConfirmationError:
                raise
            except (OSError, RuntimeError, ValueError) as exc:
                raise ConfirmationError(f"observation is unreadable for {cell_id}") from exc
            if full.ndim != 3 or tuple(full.shape[1:]) != (sequence_tokens, HIDDEN_SIZE):
                raise ConfirmationError(f"observation tensor geometry changed for {cell_id}")
            row_indices = torch.tensor(observation["row_indices"], dtype=torch.long)
            if full.shape[0] <= int(row_indices.max().item()):
                raise ConfirmationError(f"observation rows are unavailable for {cell_id}")
            activations = full.index_select(0, row_indices).contiguous()
            if tuple(activations.shape) != (RECORDS_PER_STYLE, sequence_tokens, HIDDEN_SIZE):
                raise ConfirmationError(f"observation selection geometry changed for {cell_id}")
            if not activations.dtype.is_floating_point or not torch.isfinite(activations).all().item():
                raise ConfirmationError(f"observation values are invalid for {cell_id}")
            result.append(
                FreshCell(
                    cell_id=cell_id,
                    style=style,
                    condition=condition,
                    record_ids=record_ids,
                    activations=activations,
                    attention_mask=mask,
                    position_ids=positions,
                    observation_path=observation_path,
                    observation_sha256=str(observation["sha256"]),
                )
            )
    return tuple(result)


def _validate_binding_asset_group(
    binding: Mapping[str, Any], *, key: str, repository_root: Path
) -> None:
    values = binding.get(key)
    if not isinstance(values, list) or not values:
        raise ConfirmationError(f"confirmation binding has no {key} assets")
    for value in values:
        if not isinstance(value, Mapping):
            raise ConfirmationError(f"confirmation {key} binding is malformed")
        path = _asset_path(value, repository_root=repository_root, description=f"bound {key} asset")
        current = file_record(path, repository_root=repository_root)
        if dict(value) != current:
            raise ConfirmationError(f"confirmation {key} asset descriptor changed: {path}")


def _external_runtime_record(path: Path, *, description: str) -> dict[str, Any]:
    """Record one public runtime file, including files outside the checkout."""

    raw = path.expanduser()
    if raw.is_symlink():
        raise ConfirmationError(f"{description} cannot be a symbolic link: {raw}")
    try:
        return external_file_record(raw)
    except FootingError as exc:
        raise ConfirmationError(f"{description} is unavailable: {path}") from exc


def _validate_runtime_assets(
    binding: Mapping[str, Any], *, repository_root: Path
) -> None:
    assets = binding.get("runtime_assets")
    if not isinstance(assets, Mapping) or set(assets) != set(RUNTIME_ASSET_ROLES):
        raise ConfirmationError(
            "confirmation runtime assets must include exactly "
            + ", ".join(RUNTIME_ASSET_ROLES)
        )
    for role in RUNTIME_ASSET_ROLES:
        descriptor = assets.get(role)
        if not isinstance(descriptor, Mapping):
            raise ConfirmationError(f"confirmation runtime asset is malformed: {role}")
        path_value = descriptor.get("path")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ConfirmationError(f"confirmation runtime asset path is not absolute: {role}")
        path = Path(path_value)
        if path.is_symlink() or not path.is_file():
            raise ConfirmationError(f"confirmation runtime asset is unavailable: {role}")
        try:
            current = external_file_record(path)
        except FootingError as exc:
            raise ConfirmationError(f"confirmation runtime asset is unavailable: {role}") from exc
        if dict(descriptor) != current:
            raise ConfirmationError(f"confirmation runtime asset changed: {role}")


def make_confirmation_binding(
    *,
    panel_path: Path,
    repository_root: Path,
    method_id: str,
    method_rule: str,
    method_state_paths: Sequence[Path],
    method_config_paths: Sequence[Path],
    code_paths: Sequence[Path],
    code_commit: str,
    runtime_asset_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build a per-method binding including state, code, and public runtime files."""

    _safe_method_id(method_id)
    if not isinstance(method_rule, str) or not method_rule:
        raise ConfirmationError("method rule is absent")
    if re.fullmatch(r"[0-9a-f]{40}", code_commit or "") is None:
        raise ConfirmationError("confirmation code commit must be a full lowercase commit")
    if not method_config_paths:
        raise ConfirmationError("method configuration binding is required")
    if not isinstance(runtime_asset_paths, Mapping) or set(runtime_asset_paths) != set(RUNTIME_ASSET_ROLES):
        raise ConfirmationError(
            "confirmation runtime assets must include exactly "
            + ", ".join(RUNTIME_ASSET_ROLES)
        )
    if not code_paths or any(path.suffix != ".py" for path in code_paths):
        raise ConfirmationError("confirmation code binding must include Python source files")
    actual_commit = _git_commit(repository_root.resolve())
    if actual_commit is not None and actual_commit != code_commit:
        raise ConfirmationError("confirmation code commit does not match the execution checkout")
    binding = make_binding(
        panel_path=panel_path,
        repository_root=repository_root,
        method_state_paths=method_state_paths,
        code_paths=code_paths,
        code_commit=code_commit,
    )
    binding["method_id"] = method_id
    binding["method_rule"] = method_rule
    binding["method_config"] = [
        file_record(path, repository_root=repository_root) for path in method_config_paths
    ]
    binding["runtime_assets"] = {
        role: _external_runtime_record(path, description=f"runtime asset {role}")
        for role, path in runtime_asset_paths.items()
    }
    return binding


def validate_confirmation_binding(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    repository_root: Path,
    expected_method_id: str | None = None,
) -> None:
    """Require exact binding equality and rehash every bound local asset."""

    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise ConfirmationError("confirmation input/state/config/code binding changed")
    for key in ("panel", "method_state", "method_config", "code"):
        if key == "panel":
            value = actual.get(key)
            if not isinstance(value, Mapping):
                raise ConfirmationError("confirmation panel binding is absent")
            path = _asset_path(value, repository_root=repository_root, description="bound panel")
            if dict(value) != file_record(path, repository_root=repository_root):
                raise ConfirmationError("confirmation panel binding changed")
        else:
            _validate_binding_asset_group(actual, key=key, repository_root=repository_root)
    method_id = _safe_method_id(actual.get("method_id"))
    if expected_method_id is not None and method_id != expected_method_id:
        raise ConfirmationError("confirmation method binding changed")
    if not isinstance(actual.get("method_rule"), str) or not actual["method_rule"]:
        raise ConfirmationError("confirmation method rule is absent")
    if re.fullmatch(r"[0-9a-f]{40}", str(actual.get("code_commit", ""))) is None:
        raise ConfirmationError("confirmation code commit binding is invalid")
    _validate_runtime_assets(actual, repository_root=repository_root)


def expected_prediction_path(output_root: Path, *, cell: FreshCell, method_id: str) -> Path:
    _safe_method_id(method_id)
    return output_root / cell.style / cell.condition / f"{method_id}.safetensors"


def _prediction_metadata(handle: Any) -> dict[str, str]:
    metadata = handle.metadata() or {}
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ConfirmationError("prediction metadata must contain string values")
    required = {
        "schema",
        "task_id",
        "panel_sha256",
        "selection_plan_sha256",
        "observation_sha256",
        "cell_id",
        "style",
        "condition",
        "method_id",
        "geometry_json",
        "binding_json",
    }
    if not required.issubset(metadata):
        raise ConfirmationError("prediction metadata is incomplete")
    return metadata


def _json_object(value: str, *, description: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfirmationError(f"{description} is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ConfirmationError(f"{description} must be an object")
    return decoded


def _validate_prediction_tensor(
    predictions: torch.Tensor, *, cell: FreshCell
) -> None:
    if tuple(predictions.shape) != (RECORDS_PER_STYLE, cell.sequence_tokens) or predictions.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ConfirmationError(f"prediction geometry or dtype changed for {cell.cell_id}")
    if predictions[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise ConfirmationError(f"prediction BOS token changed for {cell.cell_id}")
    active = cell.attention_mask.to(torch.bool)
    if predictions[active].lt(0).any().item() or predictions[active].ge(VOCAB_SIZE).any().item():
        raise ConfirmationError(f"active prediction is invalid or out of vocabulary for {cell.cell_id}")
    if predictions[~active].ne(INVALID_TOKEN_ID).any().item():
        raise ConfirmationError(f"padded prediction is not marked invalid for {cell.cell_id}")


def _validate_candidates(
    tensors: Mapping[str, torch.Tensor], *, cell: FreshCell
) -> bool:
    keys = set(tensors)
    if not keys.issubset({"predictions", "candidates", "candidate_scores", "selection_scores"}):
        raise ConfirmationError(f"prediction tensor fields are unexpected for {cell.cell_id}")
    has_candidates = "candidates" in keys
    if ("candidate_scores" in keys or "selection_scores" in keys) and not has_candidates:
        raise ConfirmationError(f"candidate scores are present without candidates for {cell.cell_id}")
    if not has_candidates:
        return False
    candidates = tensors["candidates"]
    if candidates.ndim != 3 or tuple(candidates.shape[:2]) != (RECORDS_PER_STYLE, cell.sequence_tokens) or candidates.shape[2] <= 0:
        raise ConfirmationError(f"candidate geometry changed for {cell.cell_id}")
    if candidates.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ConfirmationError(f"candidate IDs are not integer for {cell.cell_id}")
    active = cell.attention_mask.to(torch.bool)
    if candidates[active].lt(0).any().item() or candidates[active].ge(VOCAB_SIZE).any().item():
        raise ConfirmationError(f"active candidate is invalid or out of vocabulary for {cell.cell_id}")
    if candidates[~active].ne(INVALID_TOKEN_ID).any().item():
        raise ConfirmationError(f"padded candidate is not marked invalid for {cell.cell_id}")
    for score_name in ("candidate_scores", "selection_scores"):
        if score_name not in tensors:
            continue
        scores = tensors[score_name]
        if scores.shape != candidates.shape or scores.dtype not in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ):
            raise ConfirmationError(f"{score_name} geometry or dtype changed for {cell.cell_id}")
        if not torch.isfinite(scores[active]).all().item():
            raise ConfirmationError(f"{score_name} contains a non-finite active score for {cell.cell_id}")
        if scores[~active].ne(float("-inf")).any().item():
            raise ConfirmationError(f"{score_name} padded score is not -inf for {cell.cell_id}")
    return True


def validate_confirmation_prediction(
    path: Path,
    *,
    cell: FreshCell,
    panel_sha256: str,
    selection_plan_sha256: str,
    expected_method_id: str,
    expected_binding: Mapping[str, Any],
    candidate_policy: str,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate one prediction artifact without opening truth."""

    if path.is_symlink() or not path.is_file():
        raise ConfirmationError(f"prediction artifact is unavailable: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = _prediction_metadata(handle)
            _reject_private_panel_keys(metadata, path="prediction.metadata")
            if metadata["schema"] != PREDICTION_SCHEMA or metadata["task_id"] != TASK_ID:
                raise ConfirmationError("prediction artifact identity changed")
            if metadata["panel_sha256"] != panel_sha256 or metadata["selection_plan_sha256"] != selection_plan_sha256:
                raise ConfirmationError("prediction panel or selection-plan binding changed")
            if metadata["observation_sha256"] != cell.observation_sha256:
                raise ConfirmationError(f"prediction observation binding changed for {cell.cell_id}")
            if metadata["cell_id"] != cell.cell_id or metadata["style"] != cell.style or metadata["condition"] != cell.condition:
                raise ConfirmationError("prediction cell binding changed")
            method_id = _safe_method_id(metadata["method_id"])
            if method_id != expected_method_id:
                raise ConfirmationError("prediction method ID changed")
            geometry = _json_object(metadata["geometry_json"], description="prediction geometry")
            expected_geometry = {
                "records": RECORDS_PER_STYLE,
                "sequence_tokens": cell.sequence_tokens,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
            }
            if geometry != expected_geometry:
                raise ConfirmationError("prediction geometry binding changed")
            binding = _json_object(metadata["binding_json"], description="prediction binding")
            validate_confirmation_binding(
                binding,
                expected_binding,
                repository_root=repository_root,
                expected_method_id=expected_method_id,
            )
            keys = set(handle.keys())
            if "predictions" not in keys:
                raise ConfirmationError("prediction tensor is absent")
            tensors = {key: handle.get_tensor(key) for key in keys}
            _validate_prediction_tensor(tensors["predictions"], cell=cell)
            has_candidates = _validate_candidates(tensors, cell=cell)
            if candidate_policy not in ("required", "optional", "forbidden"):
                raise ConfirmationError("candidate policy is invalid")
            if candidate_policy == "required" and not has_candidates:
                raise ConfirmationError(f"candidate tensors are required for {expected_method_id}")
            if candidate_policy == "forbidden" and has_candidates:
                raise ConfirmationError(f"candidate tensors are forbidden for {expected_method_id}")
    except ConfirmationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfirmationError(f"prediction artifact is unreadable: {path}") from exc
    return {
        "path": path,
        "cell_id": cell.cell_id,
        "method_id": expected_method_id,
        "candidate_policy": candidate_policy,
        "has_candidates": has_candidates,
        "tensor_fields": tuple(sorted(keys)),
    }


def validate_complete_confirmation_predictions(
    output_root: Path,
    *,
    panel_path: Path,
    repository_root: Path,
    method_ids: Sequence[str],
    expected_bindings: Mapping[str, Mapping[str, Any]],
    candidate_policies: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Validate exactly 4 × N public prediction artifacts before truth."""

    if not method_ids or len(set(method_ids)) != len(method_ids):
        raise ConfirmationError("confirmation method IDs are empty or duplicated")
    if set(expected_bindings) != set(method_ids) or set(candidate_policies) != set(method_ids):
        raise ConfirmationError("confirmation method bindings or policies are incomplete")
    if output_root.is_symlink() or not output_root.is_dir():
        raise ConfirmationError("confirmation prediction output root is unavailable")
    panel = load_fresh_panel(panel_path, repository_root=repository_root)
    cells = load_fresh_cells(panel, repository_root=repository_root)
    panel_sha256 = sha256_file(panel_path)
    selection_plan_sha256 = str(panel["selection_plan_sha256"])
    expected_paths: set[Path] = set()
    validated: list[dict[str, Any]] = []
    for cell in cells:
        for method_id in method_ids:
            path = expected_prediction_path(output_root, cell=cell, method_id=method_id)
            expected_paths.add(path.resolve())
            validated.append(
                validate_confirmation_prediction(
                    path,
                    cell=cell,
                    panel_sha256=panel_sha256,
                    selection_plan_sha256=selection_plan_sha256,
                    expected_method_id=method_id,
                    expected_binding=expected_bindings[method_id],
                    candidate_policy=candidate_policies[method_id],
                    repository_root=repository_root,
                )
            )
    actual_paths = {
        path.resolve()
        for path in output_root.rglob("*.safetensors")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != expected_paths:
        missing = sorted(str(value) for value in expected_paths - actual_paths)
        extra = sorted(str(value) for value in actual_paths - expected_paths)
        raise ConfirmationError(f"confirmation prediction set is incomplete: missing={missing!r} extra={extra!r}")
    return validated


def _validate_public_file_descriptor(
    descriptor: Any, *, repository_root: Path, description: str
) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise ConfirmationError(f"{description} descriptor is absent")
    path = _asset_path(descriptor, repository_root=repository_root, description=description)
    current = file_record(path, repository_root=repository_root)
    if dict(descriptor) != current:
        raise ConfirmationError(f"{description} binding changed")
    return dict(current)


def _validate_registered_method_rows(
    method_ids: Sequence[str], methods: Sequence[Any]
) -> None:
    """Require the frozen confirmation registry to match this task's methods."""

    if tuple(method_ids) != METHOD_IDS:
        raise ConfirmationError("confirmation method set is not the registered five-method set")
    if len(methods) != len(METHOD_SPECS):
        raise ConfirmationError("confirmation method registration has the wrong method count")
    for index, (method_id, spec, row) in enumerate(zip(method_ids, METHOD_SPECS, methods)):
        if not isinstance(row, Mapping):
            raise ConfirmationError(f"confirmation method row is malformed: {index}")
        if row.get("id") != method_id or method_id != spec["id"]:
            raise ConfirmationError("confirmation method registration order changed")
        if row.get("track") != spec["track"]:
            raise ConfirmationError(f"confirmation method track changed: {method_id}")
        if row.get("candidate_policy") != spec["candidate_policy"]:
            raise ConfirmationError(f"confirmation candidate policy changed: {method_id}")
        if row.get("rule") != spec["rule"]:
            raise ConfirmationError(f"confirmation method rule changed: {method_id}")
        binding = row.get("binding")
        if not isinstance(binding, Mapping) or binding.get("method_id") != method_id:
            raise ConfirmationError(f"confirmation method binding is absent: {method_id}")
        if binding.get("method_rule") != spec["rule"]:
            raise ConfirmationError(f"confirmation bound method rule changed: {method_id}")


def load_confirmation_registration(
    path: Path,
    *,
    repository_root: Path,
    panel_path: Path,
    selection_plan_path: Path,
) -> dict[str, Any]:
    """Validate a frozen method registration and return its binding maps."""

    value = _load_json(path, description="confirmation method registration")
    _reject_private_panel_keys(value, path="registration")
    if value.get("schema") != REGISTRATION_SCHEMA or value.get("task_id") != TASK_ID:
        raise ConfirmationError("confirmation method registration identity changed")
    if value.get("status") != "FROZEN_METHOD_REGISTRATION":
        raise ConfirmationError("confirmation method registration is not frozen")
    panel_record = _validate_public_file_descriptor(value.get("panel"), repository_root=repository_root, description="registration panel")
    actual_panel = file_record(panel_path, repository_root=repository_root)
    if panel_record != actual_panel:
        raise ConfirmationError("registration panel binding changed")
    selection_record = _validate_public_file_descriptor(value.get("selection_plan"), repository_root=repository_root, description="registration selection plan")
    actual_selection = file_record(selection_plan_path, repository_root=repository_root)
    if selection_record != actual_selection:
        raise ConfirmationError("registration selection-plan binding changed")
    methods = value.get("methods")
    method_ids = value.get("method_ids")
    if not isinstance(methods, list) or not isinstance(method_ids, list) or method_ids != [row.get("id") for row in methods if isinstance(row, Mapping)]:
        raise ConfirmationError("confirmation method registration is incomplete")
    if not method_ids or len(set(method_ids)) != len(method_ids):
        raise ConfirmationError("confirmation method registration IDs are invalid")
    _validate_registered_method_rows(method_ids, methods)
    bindings: dict[str, Mapping[str, Any]] = {}
    policies: dict[str, str] = {}
    tracks: dict[str, str] = {}
    for row in methods:
        if not isinstance(row, Mapping):
            raise ConfirmationError("confirmation method row is malformed")
        method_id = _safe_method_id(row.get("id"))
        if row.get("track") not in ("comparator", "track_a", "track_b"):
            raise ConfirmationError(f"confirmation method track is invalid: {method_id}")
        policy = row.get("candidate_policy")
        if policy not in ("required", "optional", "forbidden"):
            raise ConfirmationError(f"confirmation candidate policy is invalid: {method_id}")
        binding = row.get("binding")
        if not isinstance(binding, Mapping):
            raise ConfirmationError(f"confirmation method binding is absent: {method_id}")
        validate_confirmation_binding(binding, binding, repository_root=repository_root, expected_method_id=method_id)
        bindings[method_id] = dict(binding)
        policies[method_id] = str(policy)
        tracks[method_id] = str(row["track"])
    if value.get("method_ids") != list(method_ids):
        raise ConfirmationError("confirmation method registration order changed")
    return {
        "schema": REGISTRATION_SCHEMA,
        "task_id": TASK_ID,
        "path": path.resolve(),
        "method_ids": tuple(str(value) for value in method_ids),
        "bindings": bindings,
        "candidate_policies": policies,
        "tracks": tracks,
        "panel": panel_record,
        "selection_plan": selection_record,
    }


def build_confirmation_registration(
    *,
    panel_path: Path,
    selection_plan_path: Path,
    repository_root: Path,
    bindings: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create the exact five-method registration after states are frozen."""

    if set(bindings) != set(METHOD_IDS) or tuple(bindings) != METHOD_IDS:
        raise ConfirmationError("confirmation registration bindings are not the registered five-method set")
    root = repository_root.resolve()
    panel = load_fresh_panel(panel_path, repository_root=root)
    plan = _load_json(selection_plan_path, description="confirmation selection plan")
    selection = plan.get("selection_rule")
    if not isinstance(selection, Mapping) or selection.get("record_ids_selected") is None or selection.get("record_hashes_selected") is None:
        raise ConfirmationError("confirmation selection plan is still prospective")
    panel_record = file_record(panel_path, repository_root=root)
    selection_record = file_record(selection_plan_path, repository_root=root)
    methods: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        method_id = spec["id"]
        binding = bindings.get(method_id)
        if not isinstance(binding, Mapping):
            raise ConfirmationError(f"confirmation method binding is absent: {method_id}")
        validate_confirmation_binding(binding, binding, repository_root=root, expected_method_id=method_id)
        if binding.get("method_rule") != spec["rule"]:
            raise ConfirmationError(f"confirmation method rule does not match registration: {method_id}")
        methods.append({
            "id": method_id,
            "track": spec["track"],
            "candidate_policy": spec["candidate_policy"],
            "rule": spec["rule"],
            "binding": dict(binding),
        })
    payload = {
        "schema": REGISTRATION_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_METHOD_REGISTRATION",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "panel": panel_record,
        "selection_plan": selection_record,
        "method_ids": list(METHOD_IDS),
        "methods": methods,
        "truth_opened": False,
    }
    _write_json_create_only(output_path, payload)
    return payload


def _truth_tensor_digest(value: torch.Tensor) -> str:
    """Hash canonical integer truth tensor bytes without exposing token values."""

    tensor = value.detach().cpu().to(torch.int64).contiguous()
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _truth_row_digests(value: torch.Tensor) -> list[str]:
    return [_truth_tensor_digest(row) for row in value.detach().cpu().to(torch.int64)]


def build_confirmation_truth_binding(
    *,
    panel_sha256: str,
    selection_plan_sha256: str,
    cells: Sequence[FreshCell],
    truth: Mapping[str, torch.Tensor],
    preparation: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit a prepared public-label sidecar without retaining its token values."""

    _valid_sha256(panel_sha256, description="confirmation truth panel hash")
    _valid_sha256(selection_plan_sha256, description="confirmation truth selection-plan hash")
    if not isinstance(preparation, Mapping):
        raise ConfirmationError("confirmation truth preparation is absent")
    preparation_record = dict(preparation)
    _valid_sha256(preparation_record.get("sha256"), description="confirmation truth preparation hash")
    if not isinstance(preparation_record.get("path"), str) or not preparation_record["path"]:
        raise ConfirmationError("confirmation truth preparation path is absent")
    if not isinstance(preparation_record.get("bytes"), int) or preparation_record["bytes"] < 0:
        raise ConfirmationError("confirmation truth preparation byte count is invalid")
    if not isinstance(sidecar, Mapping):
        raise ConfirmationError("confirmation truth sidecar record is absent")
    sidecar_record = dict(sidecar)
    sidecar_path = sidecar_record.get("path")
    if not isinstance(sidecar_path, str) or not Path(sidecar_path).is_absolute():
        raise ConfirmationError("confirmation truth sidecar path must be absolute")
    if not isinstance(sidecar_record.get("bytes"), int) or sidecar_record["bytes"] < 0:
        raise ConfirmationError("confirmation truth sidecar byte count is invalid")
    _valid_sha256(sidecar_record.get("sha256"), description="confirmation truth sidecar hash")
    expected_cells = [cell.cell_id for cell in cells]
    if list(truth) != expected_cells:
        raise ConfirmationError("confirmation truth cells are incomplete or out of order")
    token_shapes: dict[str, list[int]] = {}
    token_digests: dict[str, str] = {}
    row_digests: dict[str, list[str]] = {}
    record_ids = {cell.cell_id: list(cell.record_ids) for cell in cells}
    masks = {cell.cell_id: cell.attention_mask.to(torch.long).tolist() for cell in cells}
    positions = {cell.cell_id: cell.position_ids.to(torch.long).tolist() for cell in cells}
    normalized: dict[str, torch.Tensor] = {}
    for cell in cells:
        value = truth.get(cell.cell_id)
        if not isinstance(value, torch.Tensor):
            raise ConfirmationError(f"confirmation truth tensor is absent: {cell.cell_id}")
        if tuple(value.shape) != tuple(cell.attention_mask.shape) or value.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise ConfirmationError(f"confirmation truth geometry or dtype changed: {cell.cell_id}")
        value = value.detach().cpu().to(torch.int64).contiguous()
        active = cell.attention_mask.to(torch.bool)
        if value[active].lt(0).any().item() or value[active].ge(VOCAB_SIZE).any().item():
            raise ConfirmationError(f"confirmation truth token range changed: {cell.cell_id}")
        if value[~active].ne(PAD_TOKEN_ID).any().item():
            raise ConfirmationError(f"confirmation truth padding changed: {cell.cell_id}")
        if value[:, 0].ne(BOS_TOKEN_ID).any().item():
            raise ConfirmationError(f"confirmation truth BOS changed: {cell.cell_id}")
        normalized[cell.cell_id] = value
        token_shapes[cell.cell_id] = list(value.shape)
        token_digests[cell.cell_id] = _truth_tensor_digest(value)
        row_digests[cell.cell_id] = _truth_row_digests(value)
    for style in STYLE_ORDER:
        base = normalized[f"{style}__public_base"]
        shifted = normalized[f"{style}__public_lora_2601"]
        if not torch.equal(base, shifted):
            raise ConfirmationError(f"paired confirmation truth changed for {style}")
    return {
        "schema": TRUTH_BINDING_SCHEMA,
        "task_id": TASK_ID,
        "panel_sha256": panel_sha256,
        "selection_plan_sha256": selection_plan_sha256,
        "preparation": preparation_record,
        "preparation_sha256": preparation_record["sha256"],
        "sidecar": sidecar_record,
        "cell_order_json": _canonical_json(expected_cells),
        "cell_order_sha256": _json_sha256(expected_cells),
        "record_ids_sha256": _json_sha256(record_ids),
        "attention_mask_sha256": _json_sha256(masks),
        "position_ids_sha256": _json_sha256(positions),
        "token_shapes_json": _canonical_json(token_shapes),
        "token_tensor_sha256_json": _canonical_json(token_digests),
        "token_row_digests_json": _canonical_json(row_digests),
        "token_order_sha256": _json_sha256({"cell_order": expected_cells, "row_digests": row_digests}),
        "required_tensor_keys_json": _canonical_json([
            key
            for cell_id in expected_cells
            for key in (f"{cell_id}__token_ids", f"{cell_id}__attention_mask", f"{cell_id}__position_ids")
        ]),
        "paired_conditions": True,
    }


def confirmation_truth_sidecar_metadata(binding: Mapping[str, Any]) -> dict[str, str]:
    """Return the string metadata committed into the private truth sidecar."""

    if binding.get("schema") != TRUTH_BINDING_SCHEMA or binding.get("task_id") != TASK_ID:
        raise ConfirmationError("confirmation truth binding identity changed")
    fields = (
        "panel_sha256", "selection_plan_sha256", "preparation_sha256",
        "cell_order_json", "cell_order_sha256",
        "record_ids_sha256", "attention_mask_sha256", "position_ids_sha256",
        "token_shapes_json", "token_tensor_sha256_json", "token_row_digests_json",
        "token_order_sha256", "required_tensor_keys_json",
    )
    result = {"schema": TRUTH_SIDECAR_SCHEMA, "task_id": TASK_ID}
    for field in fields:
        value = binding.get(field)
        if not isinstance(value, str) or not value:
            raise ConfirmationError(f"confirmation truth binding field is absent: {field}")
        result[field] = value
    return result


def write_confirmation_truth_sidecar(
    path: Path,
    *,
    cells: Sequence[FreshCell],
    truth: Mapping[str, torch.Tensor],
    binding: Mapping[str, Any],
) -> None:
    """Write a create-only paired sidecar in the separate preparation role."""

    if path.exists() or path.is_symlink():
        raise ConfirmationError(f"confirmation truth sidecar is create-only: {path}")
    metadata = confirmation_truth_sidecar_metadata(binding)
    tensors: dict[str, torch.Tensor] = {}
    for cell in cells:
        value = truth[cell.cell_id].detach().cpu().to(torch.int32).contiguous()
        tensors[f"{cell.cell_id}__token_ids"] = value
        tensors[f"{cell.cell_id}__attention_mask"] = cell.attention_mask.to(torch.int32).contiguous()
        tensors[f"{cell.cell_id}__position_ids"] = cell.position_ids.to(torch.int32).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def _validate_truth_binding_descriptor(
    binding: Mapping[str, Any], *, panel_sha256: str, selection_plan_sha256: str, cells: Sequence[FreshCell]
) -> None:
    if not isinstance(binding, Mapping) or binding.get("schema") != TRUTH_BINDING_SCHEMA or binding.get("task_id") != TASK_ID:
        raise ConfirmationError("confirmation truth binding identity changed")
    if binding.get("panel_sha256") != panel_sha256 or binding.get("selection_plan_sha256") != selection_plan_sha256:
        raise ConfirmationError("confirmation truth binding panel or selection plan changed")
    sidecar = binding.get("sidecar")
    if not isinstance(sidecar, Mapping) or not isinstance(sidecar.get("path"), str) or not Path(str(sidecar["path"])).is_absolute():
        raise ConfirmationError("confirmation truth sidecar descriptor is absent")
    if not isinstance(sidecar.get("bytes"), int) or sidecar["bytes"] < 0:
        raise ConfirmationError("confirmation truth sidecar byte count is invalid")
    _valid_sha256(sidecar.get("sha256"), description="confirmation truth sidecar hash")
    preparation = binding.get("preparation")
    if not isinstance(preparation, Mapping) or preparation.get("sha256") != binding.get("preparation_sha256"):
        raise ConfirmationError("confirmation truth preparation binding changed")
    _valid_sha256(binding.get("preparation_sha256"), description="confirmation truth preparation hash")
    expected_cells = [cell.cell_id for cell in cells]
    if binding.get("cell_order_json") != _canonical_json(expected_cells) or binding.get("cell_order_sha256") != _json_sha256(expected_cells):
        raise ConfirmationError("confirmation truth cell order changed")
    expected_record_ids = {cell.cell_id: list(cell.record_ids) for cell in cells}
    expected_masks = {cell.cell_id: cell.attention_mask.to(torch.long).tolist() for cell in cells}
    expected_positions = {cell.cell_id: cell.position_ids.to(torch.long).tolist() for cell in cells}
    expected_public_digests = {
        "record_ids_sha256": _json_sha256(expected_record_ids),
        "attention_mask_sha256": _json_sha256(expected_masks),
        "position_ids_sha256": _json_sha256(expected_positions),
    }
    for key, expected in expected_public_digests.items():
        _valid_sha256(binding.get(key), description=f"confirmation truth {key}")
        if binding[key] != expected:
            raise ConfirmationError(f"confirmation truth {key} does not match the public panel")
    for field in ("token_shapes_json", "token_tensor_sha256_json", "token_row_digests_json"):
        value = binding.get(field)
        if not isinstance(value, str):
            raise ConfirmationError(f"confirmation truth field is absent: {field}")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfirmationError(f"confirmation truth field is invalid JSON: {field}") from exc
        if not isinstance(parsed, Mapping) or set(parsed) != set(expected_cells):
            raise ConfirmationError(f"confirmation truth cell coverage changed: {field}")
    required = binding.get("required_tensor_keys_json")
    expected_keys = [
        key
        for cell_id in expected_cells
        for key in (f"{cell_id}__token_ids", f"{cell_id}__attention_mask", f"{cell_id}__position_ids")
    ]
    if not isinstance(required, str) or json.loads(required) != expected_keys:
        raise ConfirmationError("confirmation truth tensor key binding changed")
    _valid_sha256(binding.get("token_order_sha256"), description="confirmation truth token order hash")


def validate_confirmation_truth_sidecar(
    path: Path, *, cells: Sequence[FreshCell], truth_binding: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Validate sidecar bytes and content after the complete prediction gate."""

    sidecar = truth_binding.get("sidecar")
    if not isinstance(sidecar, Mapping) or path.resolve() != Path(str(sidecar.get("path"))).resolve():
        raise ConfirmationError("confirmation truth sidecar path binding changed")
    if path.is_symlink() or not path.is_file():
        raise ConfirmationError(f"confirmation truth sidecar is unavailable: {path}")
    if int(path.stat().st_size) != int(sidecar["bytes"]) or sha256_file(path) != sidecar["sha256"]:
        raise ConfirmationError("confirmation truth sidecar hash or size changed")
    expected_metadata = confirmation_truth_sidecar_metadata(truth_binding)
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                raise ConfirmationError("confirmation truth sidecar metadata changed")
            expected_keys = set(json.loads(truth_binding["required_tensor_keys_json"]))
            if set(handle.keys()) != expected_keys:
                raise ConfirmationError("confirmation truth sidecar tensor set changed")
            result: dict[str, torch.Tensor] = {}
            for cell in cells:
                token_ids = handle.get_tensor(f"{cell.cell_id}__token_ids").to(torch.int64).contiguous()
                mask = handle.get_tensor(f"{cell.cell_id}__attention_mask").to(torch.long)
                positions = handle.get_tensor(f"{cell.cell_id}__position_ids").to(torch.long)
                if tuple(token_ids.shape) != tuple(cell.attention_mask.shape):
                    raise ConfirmationError(f"confirmation truth token geometry changed: {cell.cell_id}")
                active = cell.attention_mask.to(torch.bool)
                if token_ids[active].lt(0).any().item() or token_ids[active].ge(VOCAB_SIZE).any().item() or token_ids[:, 0].ne(BOS_TOKEN_ID).any().item():
                    raise ConfirmationError(f"confirmation truth token range or BOS changed: {cell.cell_id}")
                if token_ids[~active].ne(PAD_TOKEN_ID).any().item():
                    raise ConfirmationError(f"confirmation truth padding changed: {cell.cell_id}")
                if not torch.equal(mask, cell.attention_mask.to(torch.long)):
                    raise ConfirmationError(f"confirmation truth mask pairing changed: {cell.cell_id}")
                if not torch.equal(positions, cell.position_ids.to(torch.long)):
                    raise ConfirmationError(f"confirmation truth position pairing changed: {cell.cell_id}")
                result[cell.cell_id] = token_ids
    except ConfirmationError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ConfirmationError(f"confirmation truth sidecar is unreadable: {path}") from exc
    expected_shapes = json.loads(truth_binding["token_shapes_json"])
    expected_tensors = json.loads(truth_binding["token_tensor_sha256_json"])
    expected_rows = json.loads(truth_binding["token_row_digests_json"])
    actual_shapes = {cell_id: list(value.shape) for cell_id, value in result.items()}
    actual_tensors = {cell_id: _truth_tensor_digest(value) for cell_id, value in result.items()}
    actual_rows = {cell_id: _truth_row_digests(value) for cell_id, value in result.items()}
    if actual_shapes != expected_shapes or actual_tensors != expected_tensors or actual_rows != expected_rows:
        raise ConfirmationError("confirmation truth token digest or geometry changed")
    for style in STYLE_ORDER:
        if not torch.equal(result[f"{style}__public_base"], result[f"{style}__public_lora_2601"]):
            raise ConfirmationError(f"paired confirmation truth token order changed for {style}")
    return result


def validate_before_confirmation_truth(
    *,
    receipt_path: Path,
    repository_root: Path,
    truth_path: Path,
    output_root: Path,
    panel_path: Path,
    selection_plan_path: Path,
    registration_path: Path,
    method_ids: Sequence[str],
    expected_bindings: Mapping[str, Mapping[str, Any]],
    candidate_policies: Mapping[str, str],
    truth_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove complete public state before the caller may invoke its truth loader."""

    try:
        receipt = require_truth_open_allowed(
            receipt_path=receipt_path,
            repository_root=repository_root,
            truth_path=truth_path,
        )
    except FreezeError as exc:
        raise ConfirmationError(f"confirmation freeze receipt rejected: {exc}") from exc
    panel = load_fresh_panel(panel_path, repository_root=repository_root)
    cells = load_fresh_cells(panel, repository_root=repository_root)
    panel_sha256 = sha256_file(panel_path)
    selection_plan_record = _validate_public_file_descriptor(
        file_record(selection_plan_path, repository_root=repository_root),
        repository_root=repository_root,
        description="selection plan",
    )
    if panel["selection_plan_sha256"] != selection_plan_record["sha256"]:
        raise ConfirmationError("confirmation panel selection-plan binding changed")
    registration = load_confirmation_registration(
        registration_path,
        repository_root=repository_root,
        panel_path=panel_path,
        selection_plan_path=selection_plan_path,
    )
    if tuple(method_ids) != METHOD_IDS or tuple(method_ids) != registration["method_ids"]:
        raise ConfirmationError("confirmation method registration changed")
    if set(expected_bindings) != set(METHOD_IDS) or set(candidate_policies) != set(METHOD_IDS):
        raise ConfirmationError("confirmation method bindings or policies are not the registered five-method set")
    for spec in METHOD_SPECS:
        method_id = spec["id"]
        binding = expected_bindings[method_id]
        if candidate_policies[method_id] != spec["candidate_policy"]:
            raise ConfirmationError(f"confirmation candidate policy changed: {method_id}")
        if not isinstance(binding, Mapping) or binding.get("method_rule") != spec["rule"]:
            raise ConfirmationError(f"confirmation method rule binding changed: {method_id}")
    if dict(expected_bindings) != dict(registration["bindings"]):
        raise ConfirmationError("confirmation method state/config/code bindings changed")
    if dict(candidate_policies) != dict(registration["candidate_policies"]):
        raise ConfirmationError("confirmation candidate policy registration changed")
    _validate_truth_binding_descriptor(
        truth_binding,
        panel_sha256=panel_sha256,
        selection_plan_sha256=selection_plan_record["sha256"],
        cells=cells,
    )
    sidecar_path = Path(str(truth_binding["sidecar"]["path"])).resolve()
    if truth_path.resolve() != sidecar_path:
        raise ConfirmationError("confirmation truth sidecar path binding changed")
    try:
        output_relative = output_root.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ConfirmationError("confirmation prediction output root is outside repository") from exc
    metadata = receipt.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ConfirmationError("confirmation freeze metadata is absent")
    if (
        metadata.get("task_id") != TASK_ID
        or metadata.get("panel_sha256") != panel_sha256
        or metadata.get("selection_plan_sha256") != selection_plan_record["sha256"]
        or metadata.get("method_ids") != list(method_ids)
        or metadata.get("registration_sha256") != sha256_file(registration_path)
        or metadata.get("truth_binding") != dict(truth_binding)
        or receipt.get("frozen_root") != output_relative
    ):
        raise ConfirmationError("confirmation freeze receipt does not bind requested inputs")
    validate_complete_confirmation_predictions(
        output_root,
        panel_path=panel_path,
        repository_root=repository_root,
        method_ids=method_ids,
        expected_bindings=expected_bindings,
        candidate_policies=candidate_policies,
    )
    return receipt


def open_truth_after_confirmation_gate(
    *,
    truth_loader: Callable[[Path], Any],
    gate_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Invoke a caller's truth loader only after the complete public gate."""

    receipt = validate_before_confirmation_truth(**dict(gate_kwargs))
    panel = load_fresh_panel(Path(str(gate_kwargs["panel_path"])), repository_root=Path(str(gate_kwargs["repository_root"])))
    cells = load_fresh_cells(panel, repository_root=Path(str(gate_kwargs["repository_root"])))
    validate_confirmation_truth_sidecar(
        Path(str(gate_kwargs["truth_path"])),
        cells=cells,
        truth_binding=gate_kwargs["truth_binding"],
    )
    return receipt, truth_loader(Path(str(gate_kwargs["truth_path"])))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_warmed_prediction(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    predictor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device | str = "cpu",
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run one warmup plus three measured per-record predictions.

    The predictor is supplied by the method adapter.  It receives one row of
    activations, mask, and positions on ``device`` and must return one row of
    integer predicted IDs.  The first measured result is the designated output
    for accuracy.  Later measured results are compared with it exactly, so
    nondeterminism is reported rather than hidden by an average.
    """

    device = torch.device(device)
    if observations.ndim != 3 or attention_mask.shape != observations.shape[:2] or position_ids.shape != attention_mask.shape:
        raise ConfirmationError("warmed timing input geometry changed")
    if warmup_runs != 1 or measured_runs != 3:
        raise ConfirmationError("confirmation timing requires one warmup and three measured runs")
    if observations.shape[0] == 0:
        raise ConfirmationError("warmed timing input is empty")
    outputs: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    def invoke(row_index: int) -> tuple[torch.Tensor, float]:
        _synchronize(device)
        started = time.perf_counter()
        row_h = observations[row_index].to(device=device, non_blocking=False)
        row_mask = attention_mask[row_index].to(device=device, dtype=torch.long, non_blocking=False)
        row_positions = position_ids[row_index].to(device=device, dtype=torch.long, non_blocking=False)
        output = predictor(row_h, row_mask, row_positions)
        if not isinstance(output, torch.Tensor):
            raise ConfirmationError("predictor did not return a tensor")
        output = output.detach().to(device="cpu", dtype=torch.long).contiguous()
        if tuple(output.shape) not in {(int(observations.shape[1]),), (1, int(observations.shape[1]))}:
            raise ConfirmationError("predictor returned an unexpected per-record geometry")
        output = output.reshape(int(observations.shape[1]))
        _synchronize(device)
        return output, time.perf_counter() - started

    for row_index in range(int(observations.shape[0])):
        warmup_seconds: list[float] = []
        for _ in range(warmup_runs):
            _, elapsed = invoke(row_index)
            warmup_seconds.append(elapsed)
        measured: list[torch.Tensor] = []
        measured_seconds: list[float] = []
        for _ in range(measured_runs):
            output, elapsed = invoke(row_index)
            measured.append(output)
            measured_seconds.append(elapsed)
        mismatch_runs = [index + 1 for index, output in enumerate(measured[1:], start=1) if not torch.equal(output, measured[0])]
        outputs.append(measured[0])
        records.append(
            {
                "record_index": row_index,
                "warmup_runs": warmup_runs,
                "warmup_seconds": warmup_seconds,
                "measured_runs": measured_runs,
                "measured_seconds": measured_seconds,
                "designated_measured_run": 1,
                "repeated_prediction_exact": not mismatch_runs,
                "mismatch_runs": mismatch_runs,
                "steady_interval": "CPU activation H->device preprocessing + predictor + predicted IDs device->CPU",
                "shared_resources_resident": True,
            }
        )
    return torch.stack(outputs), {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "records": records,
        "total_elapsed_seconds": time.perf_counter() - total_started,
        "cold_costs_separate": True,
        "method_specific_prefix_calls_and_candidate_simulations": "record separately for each method; this generic timing helper does not assume zero",
    }


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None


def _small_external_record(path: Path, *, known_sha256: str | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        return {"path": str(path), "available": False, "bytes": None, "sha256": None}
    size = int(path.stat().st_size)
    digest = known_sha256 if known_sha256 else (sha256_file(path) if size <= 5 * 1024 * 1024 else None)
    return {
        "path": str(path),
        "available": True,
        "bytes": size,
        "sha256": digest,
        "hash_status": (
            "pinned" if known_sha256 else ("computed" if digest else "deferred_until_selection_freeze")
        ),
    }


def _repository_record(path: Path, *, repository_root: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        return {"path": str(path.relative_to(repository_root.resolve())), "available": False}
    return {"path": path.relative_to(repository_root.resolve()).as_posix(), **file_record(path, repository_root=repository_root)}


def build_prospective_plan(*, repository_root: Path, generated_at_utc: str | None = None) -> dict[str, Any]:
    """Build the no-records selection registration for later method freeze."""

    root = repository_root.resolve()
    trr3_panel = root / "experiments/TRR-0003/evidence/control/panel.json"
    trr3_plan = root / "experiments/TRR-0003/evidence/control/plan.json"
    trr4_alpaca_plan = root / "experiments/TRR-0004/alpaca_split_plan.json"
    trr4_fit_records = root / "experiments/TRR-0004/fit/adapter_v2/affine_fit_records.json"
    trr4_validation_records = root / "experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json"
    trr4_fit_manifest = root / "experiments/TRR-0004/fit/adapter_v2/public_fit_manifest.json"
    source_repo = root.parent.parent if root.parent.name == ".worktrees" else root
    pile_validation_records = source_repo / "outputs/TRR-0003/track_b/public_validation_slice_v2/public_validation_records.json"
    pile_fit_records = source_repo / "outputs/TRR-0003/track_b/public_fit_v2/fit_records.json"
    pile_receipt = source_repo / "outputs/TRR-0003/track_b/public_validation_slice_v2/validation_slice_evidence.json"
    pile_validation_records = pile_validation_records.resolve()
    pile_fit_records = pile_fit_records.resolve()
    pile_receipt = pile_receipt.resolve()
    sources = {
        "alpaca": {
            "id": "tatsu-lab/alpaca",
            "split": "train",
            "revision": "dce01c9b08f87459cf36a430d809084718273017",
            "arrow": _small_external_record(
                Path("/home/alanz/.cache/huggingface/datasets/tatsu-lab___alpaca/default/0.0.0/dce01c9b08f87459cf36a430d809084718273017/alpaca-train.arrow"),
                known_sha256="f45103036ed651f4c06d0a3c3e0fb7d53acb3074ed5c8e804a69c1efc1cea794",
            ),
            "role": "current public fitting recipe only; excluded from confirmation",
        },
        "pile": {
            "id": "NeelNanda/pile-10k",
            "split": "train",
            "revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
            "arrow": _small_external_record(
                Path("/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"),
                known_sha256="77ddf02e2a69373a944bc8bc8ac8f7b9926f5c62203d727341a24d709bf81113",
            ),
            "dataset_info": _small_external_record(Path("/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/dataset_info.json")),
            "role": "fresh public confirmation candidates",
        },
        "finance": {
            "id": "Josephgflowers/Finance-Instruct-500k",
            "split": "train",
            "fingerprint": "4abbac8acaab4205",
            "dataset_info": _small_external_record(Path("/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/dataset_info.json")),
            "role": "fresh public confirmation candidates",
        },
    }
    exclusion_sources = [
        {
            "id": "trr3_public_panel",
            "role": "historical public development panel IDs; exclude matching Pile/Finance records",
            "record": _repository_record(trr3_panel, repository_root=root),
            "known_exact_ids": True,
        },
        {
            "id": "trr3_public_plan",
            "role": "historical public fitting/evaluation split declarations",
            "record": _repository_record(trr3_plan, repository_root=root),
            "known_exact_ids": False,
        },
        {
            "id": "trr3_public_pile_fit",
            "role": "historical public Pile fitting IDs",
            "record": _small_external_record(pile_fit_records, known_sha256="7aee0f6cb452bb1df401c920ca2a628d32fb204d144d72e4419fc8bd34a3a08e"),
            "known_exact_ids": True,
        },
        {
            "id": "trr3_public_pile_validation",
            "role": "historical public validation Pile IDs",
            "record": _small_external_record(pile_validation_records, known_sha256="446e6259482b730b3e22cdc693d37b406031141d7ac034de68ecaa5cc49456fb"),
            "known_exact_ids": True,
        },
        {
            "id": "trr4_alpaca_fit",
            "role": "current public Alpaca fitting rows; dataset-disjoint from confirmation",
            "record": _repository_record(trr4_fit_records, repository_root=root),
            "known_exact_ids": True,
        },
        {
            "id": "trr4_alpaca_validation",
            "role": "current public Alpaca validation rows; dataset-disjoint from confirmation",
            "record": _repository_record(trr4_validation_records, repository_root=root),
            "known_exact_ids": True,
        },
        {
            "id": "trr4_alpaca_fit_manifest",
            "role": "current public fit resource declaration",
            "record": _repository_record(trr4_fit_manifest, repository_root=root),
            "known_exact_ids": True,
        },
        {
            "id": "trr4_alpaca_split_plan",
            "role": "current public Alpaca split declaration and future exclusion policy",
            "record": _repository_record(trr4_alpaca_plan, repository_root=root),
            "known_exact_ids": True,
        },
    ]
    return {
        "schema": PLAN_SCHEMA,
        "task_id": TASK_ID,
        "status": "PROSPECTIVE_SELECTION_RULE_NO_RECORDS_SELECTED",
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "execution": {
            "git_commit": _git_commit(root),
            "script": str(Path(__file__).resolve()),
            "python": sys.executable,
            "model_loaded": False,
            "observations_generated": False,
            "truth_opened": False,
            "network_used": False,
        },
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "cut_depth": CUT_DEPTH, "hidden_size": HIDDEN_SIZE},
        "observation_generation_contract": {
            "path": "public_prefix.forward_full",
            "same_public_prefix_path": True,
            "batch_size": 8,
            "sequence_tokens": 192,
            "padding_semantics": "right-padding; active outputs must be bit-exact under future-pad perturbation",
            "alternate_batch_policy": "unpadded batch-1 is diagnostic only and cannot replace the primary path",
        },
        "styles": [
            {
                "id": "pile",
                "records": RECORDS_PER_STYLE,
                "sequence_tokens": SEQUENCE_TOKENS["pile"],
                "input_style": "plain Pile text",
                "minimum_post_bos_tokens": 39,
            },
            {
                "id": "finance",
                "records": RECORDS_PER_STYLE,
                "sequence_tokens": SEQUENCE_TOKENS["finance"],
                "input_style": "Finance chat-template rendering",
                "minimum_post_bos_tokens": 32,
            },
        ],
        "conditions": [
            {"id": "public_base", "weights_available_to_reconstructor": True, "role": "matched public control"},
            {"id": "public_lora_2601", "weights_available_to_reconstructor": False, "role": "one synthetic target-shift diagnostic"},
        ],
        "public_sources": sources,
        "tokenizer": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot": "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/" + MODEL_REVISION,
            "bos_token_id": BOS_TOKEN_ID,
            "padding_token_id": PAD_TOKEN_ID,
            "local_files_only": True,
        },
        "selection_rule": {
            "record_ids_selected": None,
            "record_hashes_selected": None,
            "algorithm": "After method/config freeze, render each source in stored order with the pinned tokenizer/template; filter minimum post-BOS length; exclude public IDs/content hashes from every listed metadata source; stable-sort by (dataset, source row index, public record ID); take the first 16 eligible rows per dataset.",
            "pile": {"take_first_tokens": 40, "pair_rows_across_conditions": True},
            "finance": {"truncate_or_right_pad_to": 128, "pair_rows_across_conditions": True},
            "bos_token_id": BOS_TOKEN_ID,
            "padding_token_id": PAD_TOKEN_ID,
            "source_text_or_token_ids_written": False,
            "exact_ids_and_hashes_written_only_after": ["method state/config freeze", "public selection metadata freeze"],
        },
        "exclusion_sources": exclusion_sources,
        "historical_provenance_boundary": {
            "retained_a1_exact_fit_record_ids_available": False,
            "retained_a1_exact_fit_duration_available": False,
            "unknown_historical_ids_do_not_block": "dataset-disjoint Pile/Finance selection using the known public metadata above",
            "no_private_evaluator_contents_used": True,
        },
        "methods_prospective": [dict(row) for row in METHOD_SPECS],
        "registration_boundary": "Method state/config paths, exact selected records, activation paths, and truth binding are written only after methods are frozen. This plan does not activate a benchmark method.",
        "timing_contract": {
            "warmup_runs_per_record_method": 1,
            "measured_runs_per_record_method": 3,
            "accuracy_output": "first measured run after exact repeat check",
            "steady_interval": "CPU H input -> device preprocessing -> method prediction -> predicted IDs CPU",
            "shared_resources_resident": True,
            "cold_costs_separate": ["state/config/model load", "prefix/embedding load", "input I/O", "hashing", "GPU initialization"],
            "cuda_synchronize_boundaries": True,
            "method_specific_costs": "prefix calls and candidate simulations are measured from each method's implementation; this plan does not assume zero",
            "standalone_decoder_runtime": {"prefix_calls": 0, "candidate_simulations": 0, "a2_fallback": False},
        },
        "truth_gate_contract": {
            "load_order": ["freeze receipt", "panel and public observations", "method registration", "all prediction artifacts", "truth loader"],
            "required_matrix": "4 cells × every registered method",
            "truth_loader_called_by_adapter": False,
            "private_sidecar_contents_not_read_before_gate": True,
        },
        "uncertainty_and_coverage": {
            "selection_counts": "unknown until public metadata freeze",
            "position_coverage": "Pile uses first 39 post-BOS positions; Finance reports active post-BOS positions through max128 and padding counts per row",
            "target_shift_coverage": "one synthetic LoRA2601 condition; not evidence for broad target/SFT transfer",
            "style_coverage": "one plain-text and one Finance chat-template style",
        },
        "coverage_analysis": {
            "shared_reference": "TRR-0004 large public Alpaca fitting labels; this is a coverage reference, not the retained A1's observed/unobserved set",
            "token_frequency_bins": ["0", "1-4", "5-19", "20+"],
            "position_bins_post_bos": ["1-15", "16-39", "40-79", "80+"],
            "denominator": "active scored tokens, reported per style and target condition",
            "interval_unit": "complete-record paired resampling within each style/condition; no independent-token confidence intervals",
            "counts": "computed only after exact public labels and frozen confirmation records are available",
        },
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ConfirmationError(f"refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_frozen_public_matrix(
    *,
    repository_root: Path,
    plan_path: Path,
    panel_path: Path,
    registration_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run the narrow public-only five-method validation command.

    This is intentionally a validator/adapter rather than a model runner:
    method owners produce the frozen prediction tensors, then this command
    proves that the selected plan, panel, registration, and all 20 artifacts
    agree before a separate truth gate is invoked.
    """

    root = repository_root.resolve()
    plan = _load_json(plan_path, description="confirmation selection plan")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("task_id") != TASK_ID:
        raise ConfirmationError("confirmation selection plan identity changed")
    selection = plan.get("selection_rule")
    if not isinstance(selection, Mapping) or selection.get("record_ids_selected") is None or selection.get("record_hashes_selected") is None:
        raise ConfirmationError("confirmation selection plan is still prospective")
    if plan.get("methods_prospective") and tuple(row.get("id") for row in plan["methods_prospective"] if isinstance(row, Mapping)) != METHOD_IDS:
        raise ConfirmationError("confirmation selection plan method set changed")
    panel = load_fresh_panel(panel_path, repository_root=root)
    plan_record = file_record(plan_path, repository_root=root)
    if panel.get("selection_plan_sha256") != plan_record["sha256"]:
        raise ConfirmationError("confirmation panel is bound to a different selection plan")
    registration = load_confirmation_registration(
        registration_path,
        repository_root=root,
        panel_path=panel_path,
        selection_plan_path=plan_path,
    )
    if registration["method_ids"] != METHOD_IDS:
        raise ConfirmationError("confirmation registration does not contain the fixed five methods")
    validated = validate_complete_confirmation_predictions(
        output_root,
        panel_path=panel_path,
        repository_root=root,
        method_ids=registration["method_ids"],
        expected_bindings=registration["bindings"],
        candidate_policies=registration["candidate_policies"],
    )
    return {
        "schema": "token-reconstruction.trr0004-public-matrix-validation.v1",
        "task_id": TASK_ID,
        "status": "PUBLIC_MATRIX_COMPLETE_NO_TRUTH_OPENED",
        "panel": file_record(panel_path, repository_root=root),
        "selection_plan": plan_record,
        "registration": file_record(registration_path, repository_root=root),
        "method_ids": list(registration["method_ids"]),
        "cells": len(expected_cell_ids()),
        "prediction_artifacts": len(validated),
        "truth_opened": False,
        "truth_loader_called": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate an already frozen public matrix")
    parser.add_argument("--output", type=Path, default=Path("experiments/TRR-0004/fresh_confirmation_plan.json"), help="prospective plan output")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", type=Path, help="frozen selection plan for --validate")
    parser.add_argument("--panel", type=Path, help="frozen public panel for --validate")
    parser.add_argument("--registration", type=Path, help="frozen method registration for --validate")
    parser.add_argument("--predictions", type=Path, help="frozen prediction root for --validate")
    parser.add_argument("--result", type=Path, help="create-only JSON result for --validate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.expanduser().resolve()
    if args.validate:
        required = {
            "--plan": args.plan,
            "--panel": args.panel,
            "--registration": args.registration,
            "--predictions": args.predictions,
            "--result": args.result,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit("--validate requires " + ", ".join(missing))
        result = validate_frozen_public_matrix(
            repository_root=root,
            plan_path=args.plan.expanduser().resolve(),
            panel_path=args.panel.expanduser().resolve(),
            registration_path=args.registration.expanduser().resolve(),
            output_root=args.predictions.expanduser().resolve(),
        )
        _write_json_create_only(args.result.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 0
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite existing plan: {output}")
    plan = build_prospective_plan(repository_root=root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "status": plan["status"],
        "styles": {row["id"]: row["records"] for row in plan["styles"]},
        "methods": list(METHOD_IDS),
        "observations_generated": False,
        "truth_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
