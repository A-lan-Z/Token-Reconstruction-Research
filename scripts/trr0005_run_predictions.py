#!/usr/bin/env python3
"""Run the frozen TRR-0005 public prediction matrix.

The driver consumes only the frozen public panel, its public observations, the
panel-bound method registration, and the already selected TRR-0005 states.  It
loads one method at a time, calls the shared predictor with exactly one warmup
and one measured call per record, and writes one compact prediction artifact
and timing receipt for every cell/method pair.  It never loads source text,
target labels, or evaluator-private truth.

The six new methods use the joint decoder states.  The retained A1 and A1+A2
anchors reuse the reviewed TRR-0004 adapters; A2 keeps its fixed K=256
proposal/selection rule but omits candidate tensors from the TRR-0005 output.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource as sys_resource
import subprocess
import sys
import time
from typing import Any

from safetensors import safe_open
import torch

import trr0004_predict_confirmation as legacy
from token_reconstruction.footing import (
    FootingError,
    external_file_record,
    file_record,
    sha256_file,
)
from token_reconstruction.historical_inputlens_bridge import (
    load_historical_lens_checkpoint,
)
from token_reconstruction.trr0005_contract import (
    BOS_TOKEN_ID,
    CANDIDATE_POLICIES,
    CONDITION_ORDER,
    ContractError,
    EXPECTED_CELL_IDS,
    INVALID_TOKEN_ID,
    METHOD_IDS,
    PANEL_SCHEMA,
    RECORDS_PER_DOMAIN,
    REGISTRATION_SCHEMA,
    SEQUENCE_TOKENS,
    STYLE_ORDER,
    TASK_ID,
    validate_panel_descriptor,
    validate_registration,
)
from token_reconstruction.trr0005_joint_decoder import (
    DEFAULT_CONTEXT_WIDTH,
    JointDecoderError,
    load_decoder_state,
)
from trr0005_predict_confirmation import (
    PredictionError,
    prediction_descriptor,
    run_warmed_prediction,
    write_prediction_artifact,
    write_prediction_receipt,
)


# Keep task geometry local and explicit in the executable.
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
CUT_DEPTH = 4
A2_METHOD_ID = "frozen_a1_a2_k256"
JOINT_STATE_METHODS = (
    "joint_full_affine",
    "affine_causal_h_attention128",
    "affine_trained_diagonal_attention128",
)
RUNTIME_EMBEDDING_ROLE = "public_embedding_table"
RUNTIME_P0_ROLES = ("public_prefix_checkpoint", "public_prefix_config")
SCRIPT_SCHEMA = "token-reconstruction.trr0005-fresh-confirmation-run.v1"
FAILURE_SCHEMA = "token-reconstruction.trr0005-fresh-confirmation-run-failure.v1"
QUALIFICATION_SCHEMA = "token-reconstruction.trr0005-archived-record-qualification.v1"
QUALIFICATION_FAILURE_SCHEMA = "token-reconstruction.trr0005-archived-record-qualification-failure.v1"
DEFAULT_MINIMUM_FREE_GIB = 8.0
DEFAULT_MAXIMUM_RESERVED_GIB = 6.0
DEFAULT_MAXIMUM_RSS_GIB = 16.0
DEFAULT_MAX_SECONDS = 1800.0
DEFAULT_FIT_ROOT = Path("experiments/TRR-0005/joint_fit_v1")
DEFAULT_CAUSAL_FIT_ROOT = Path("experiments/TRR-0005/joint_fit_qknorm_v1")


class PredictionRunnerError(ContractError):
    """Raised when a frozen public prediction run cannot proceed."""


@dataclass(frozen=True)
class FreshRecord:
    """One source-free public activation row passed to a method adapter."""

    record_id: str
    activation: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


@dataclass(frozen=True)
class FreshCell:
    """One of the four 128-record public cells."""

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


@dataclass(frozen=True)
class RegisteredMethod:
    method_id: str
    binding: dict[str, Any]
    state_path: Path
    config_paths: tuple[Path, ...]
    code_paths: tuple[Path, ...]
    runtime_paths: dict[str, Path]


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PredictionRunnerError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionRunnerError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PredictionRunnerError(f"{description} must be a JSON object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PredictionRunnerError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PredictionRunnerError("unable to resolve executable git commit") from exc
    commit = result.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise PredictionRunnerError("executable git commit is not a full lowercase hash")
    return commit


def _rusage_rss_bytes() -> int:
    value = int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _canonical_path(value: Any, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PredictionRunnerError(f"{description} path is absent")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise PredictionRunnerError(f"{description} path must be absolute at runtime")
    return raw.resolve()


def _asset_path(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    description: str,
    external_allowed: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one descriptor and require its path/size/hash to be current."""

    raw = descriptor.get("path")
    if not isinstance(raw, str) or not raw:
        raise PredictionRunnerError(f"{description} path is absent")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        path = candidate.resolve()
        if not external_allowed:
            try:
                path.relative_to(root.resolve())
            except ValueError as exc:
                raise PredictionRunnerError(f"{description} escaped repository root") from exc
        actual = external_file_record(path)
    else:
        if "\\" in raw:
            raise PredictionRunnerError(f"{description} path must use POSIX separators")
        relative = PurePosixPath(raw)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
            or relative.as_posix() != raw
        ):
            raise PredictionRunnerError(f"{description} path is unsafe: {raw}")
        path = (root / raw).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise PredictionRunnerError(f"{description} escaped repository root") from exc
        actual = file_record(path, repository_root=root)
    if path.is_symlink() or not path.is_file():
        raise PredictionRunnerError(f"{description} is unavailable: {path}")
    expected = {key: descriptor.get(key) for key in ("path", "bytes", "sha256")}
    if expected != actual:
        raise PredictionRunnerError(
            f"{description} binding changed: expected {expected!r}, observed {actual!r}"
        )
    return path, actual


def _observation_descriptor(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    value = cell.get("observation")
    if isinstance(value, Mapping):
        if isinstance(value.get("file"), Mapping):
            return value["file"]
        if isinstance(value.get("observation"), Mapping):
            return value["observation"]
        return value
    raise PredictionRunnerError("panel cell has no observation descriptor")


def _panel_cells(panel: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cells = panel.get("cells")
    if isinstance(cells, Mapping):
        rows: list[Mapping[str, Any]] = []
        for cell_id in EXPECTED_CELL_IDS:
            row = cells.get(cell_id)
            if not isinstance(row, Mapping):
                raise PredictionRunnerError(f"panel cell is absent or malformed: {cell_id}")
            rows.append(row)
        if set(cells) != set(EXPECTED_CELL_IDS):
            raise PredictionRunnerError("panel has extra or missing cells")
        return rows
    if isinstance(cells, list):
        if [row.get("id", row.get("cell_id")) for row in cells if isinstance(row, Mapping)] != list(EXPECTED_CELL_IDS):
            raise PredictionRunnerError("panel cell order or completeness changed")
        return [row for row in cells if isinstance(row, Mapping)]
    raise PredictionRunnerError("panel cells are malformed")


def _mask_positions(cell: Mapping[str, Any], *, cell_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    geometry = cell.get("geometry")
    source = geometry if isinstance(geometry, Mapping) else cell
    mask_value = source.get("attention_mask", source.get("mask"))
    positions_value = source.get("position_ids", source.get("positions"))
    if mask_value is None or positions_value is None:
        raise PredictionRunnerError(f"panel geometry is incomplete: {cell_id}")
    try:
        mask = torch.as_tensor(mask_value, dtype=torch.long).contiguous().cpu()
        positions = torch.as_tensor(positions_value, dtype=torch.long).contiguous().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PredictionRunnerError(f"panel geometry is malformed: {cell_id}") from exc
    expected = (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS)
    if tuple(mask.shape) != expected or tuple(positions.shape) != expected:
        raise PredictionRunnerError(f"panel geometry changed: {cell_id}")
    if mask.lt(0).any().item() or mask.gt(1).any().item():
        raise PredictionRunnerError(f"panel mask is not binary: {cell_id}")
    bool_mask = mask.to(torch.bool)
    if not bool_mask[:, 0].all().item() or (bool_mask[:, 1:] > bool_mask[:, :-1]).any().item():
        raise PredictionRunnerError(f"panel mask is not BOS/right-padded: {cell_id}")
    expected_positions = bool_mask.to(torch.long).cumsum(1).sub(1).clamp_min(0)
    if not torch.equal(positions, expected_positions):
        raise PredictionRunnerError(f"panel positions disagree with mask: {cell_id}")
    return bool_mask, positions


def _record_ids(cell: Mapping[str, Any], *, cell_id: str) -> tuple[str, ...]:
    values = cell.get("records")
    if not isinstance(values, list) or len(values) != RECORDS_PER_DOMAIN:
        raise PredictionRunnerError(f"panel record count changed: {cell_id}")
    result: list[str] = []
    allowed = {"record_id", "public_record_sha256", "raw_index", "valid_tokens", "source_index"}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) - allowed:
            raise PredictionRunnerError(f"panel record metadata is malformed: {cell_id}/{index}")
        record_id = value.get("record_id")
        digest = value.get("public_record_sha256")
        if not isinstance(record_id, str) or not record_id:
            raise PredictionRunnerError(f"panel record ID is absent: {cell_id}/{index}")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise PredictionRunnerError(f"panel record digest is invalid: {cell_id}/{index}")
        result.append(record_id)
    if len(set(result)) != len(result):
        raise PredictionRunnerError(f"panel record IDs are duplicated: {cell_id}")
    return tuple(result)


def _load_observation(
    descriptor: Mapping[str, Any],
    *,
    cell_id: str,
    root: Path,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, Path, dict[str, Any]]:
    path, record = _asset_path(
        descriptor,
        root=root,
        description=f"observation {cell_id}",
        external_allowed=True,
    )
    if path.suffix.casefold() != ".safetensors":
        raise PredictionRunnerError(f"observation must be safetensors for {cell_id}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if "activations" not in keys:
                raise PredictionRunnerError(f"observation activations are absent: {cell_id}")
            unexpected = keys - {"activations", "attention_mask", "position_ids"}
            if unexpected:
                raise PredictionRunnerError(f"observation tensors are unexpected: {cell_id}: {unexpected!r}")
            activations = handle.get_tensor("activations").contiguous()
            observed_mask = handle.get_tensor("attention_mask").contiguous() if "attention_mask" in keys else None
            observed_positions = handle.get_tensor("position_ids").contiguous() if "position_ids" in keys else None
    except PredictionRunnerError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise PredictionRunnerError(f"observation is unreadable: {cell_id}") from exc
    expected_shape = (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, HIDDEN_SIZE)
    if tuple(activations.shape) != expected_shape or not activations.dtype.is_floating_point:
        raise PredictionRunnerError(f"observation geometry or dtype changed: {cell_id}")
    if not torch.isfinite(activations).all().item():
        raise PredictionRunnerError(f"observation contains non-finite values: {cell_id}")
    if observed_mask is not None and tuple(observed_mask.shape) != expected_shape[:2]:
        raise PredictionRunnerError(f"observation mask geometry changed: {cell_id}")
    if observed_positions is not None and tuple(observed_positions.shape) != expected_shape[:2]:
        raise PredictionRunnerError(f"observation position geometry changed: {cell_id}")
    return activations, observed_mask, observed_positions, path, record


def load_trr5_panel(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], tuple[FreshCell, ...], dict[str, Any]]:
    """Validate and load the four source-free public observation cells."""

    root = repository_root.expanduser().resolve()
    panel_path = path.expanduser().resolve()
    panel = _load_json(panel_path, description="TRR-0005 panel")
    if panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != TASK_ID:
        raise PredictionRunnerError("TRR-0005 panel identity changed")
    if panel.get("status") != "FROZEN_FRESH_CONFIRMATION_PANEL":
        raise PredictionRunnerError("TRR-0005 panel is not frozen")
    if panel.get("sequence_tokens") != SEQUENCE_TOKENS or panel.get("records_per_domain") != RECORDS_PER_DOMAIN:
        raise PredictionRunnerError("TRR-0005 panel geometry changed")
    if panel.get("hidden_size", HIDDEN_SIZE) != HIDDEN_SIZE:
        raise PredictionRunnerError("TRR-0005 panel hidden size changed")
    if panel.get("cut_depth", panel.get("observation_contract", {}).get("cut_depth", CUT_DEPTH)) != CUT_DEPTH:
        raise PredictionRunnerError("TRR-0005 panel cut depth changed")
    # The contract expects a mapping.  Keep the scorer's older list form
    # readable as a compatibility port while validating the same content.
    contract_panel = dict(panel)
    rows = _panel_cells(panel)
    if isinstance(panel.get("cells"), list):
        contract_panel["cells"] = {
            str(row.get("id", row.get("cell_id"))): row for row in rows
        }
    try:
        validate_panel_descriptor(contract_panel)
    except ContractError as exc:
        raise PredictionRunnerError(f"panel failed TRR-0005 contract validation: {exc}") from exc
    panel_record = file_record(panel_path, repository_root=root)

    cells: list[FreshCell] = []
    paired: dict[str, tuple[tuple[str, ...], torch.Tensor, torch.Tensor]] = {}
    for expected_cell_id, cell in zip(EXPECTED_CELL_IDS, rows):
        cell_id = cell.get("cell_id", cell.get("id"))
        style, condition = expected_cell_id.split("__", 1)
        if cell_id != expected_cell_id or cell.get("style") != style or cell.get("condition") != condition:
            raise PredictionRunnerError(f"panel cell identity changed: {expected_cell_id}")
        record_ids = _record_ids(cell, cell_id=expected_cell_id)
        mask, positions = _mask_positions(cell, cell_id=expected_cell_id)
        activations, observed_mask, observed_positions, observation_path, observation_record = _load_observation(
            _observation_descriptor(cell), cell_id=expected_cell_id, root=root
        )
        if observed_mask is not None and not torch.equal(observed_mask.to(torch.long), mask.to(torch.long)):
            raise PredictionRunnerError(f"observation mask binding changed: {expected_cell_id}")
        if observed_positions is not None and not torch.equal(observed_positions.to(torch.long), positions):
            raise PredictionRunnerError(f"observation position binding changed: {expected_cell_id}")
        descriptor = _observation_descriptor(cell)
        declared_shape = descriptor.get("shape")
        if declared_shape is not None and list(declared_shape) != list(activations.shape):
            raise PredictionRunnerError(f"observation declared geometry changed: {expected_cell_id}")
        if condition == CONDITION_ORDER[0]:
            paired[style] = (record_ids, mask.clone(), positions.clone())
        else:
            prior_ids, prior_mask, prior_positions = paired[style]
            if record_ids != prior_ids or not torch.equal(mask, prior_mask) or not torch.equal(positions, prior_positions):
                raise PredictionRunnerError(f"paired public geometry changed: {style}")
        cells.append(
            FreshCell(
                cell_id=expected_cell_id,
                style=style,
                condition=condition,
                record_ids=record_ids,
                activations=activations,
                attention_mask=mask,
                position_ids=positions,
                observation_path=observation_path,
                observation_sha256=str(observation_record["sha256"]),
            )
        )
    return panel, tuple(cells), panel_record


def _validate_plan_binding(
    plan_path: Path,
    *,
    panel: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_path = plan_path.expanduser().resolve()
    plan = _load_json(plan_path, description="TRR-0005 selection plan")
    plan_record = file_record(plan_path, repository_root=root)
    candidates: list[Mapping[str, Any]] = []
    if isinstance(panel.get("selection_plan"), Mapping):
        candidates.append(panel["selection_plan"])
    if isinstance(panel.get("selection_plan_file"), Mapping):
        candidates.append(panel["selection_plan_file"])
    panel_hash = panel.get("selection_plan_sha256")
    if isinstance(panel_hash, str) and panel_hash != plan_record["sha256"]:
        raise PredictionRunnerError("panel selection-plan hash differs from supplied plan")
    for descriptor in candidates:
        if descriptor.get("sha256") != plan_record["sha256"]:
            raise PredictionRunnerError("panel selection-plan descriptor changed")
        if isinstance(descriptor.get("path"), str):
            raw = Path(descriptor["path"]).expanduser()
            expected_path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
            if expected_path != plan_path:
                raise PredictionRunnerError("panel selection-plan path differs from supplied plan")
    selection = plan.get("public_validation_selection")
    if not isinstance(selection, Mapping):
        raise PredictionRunnerError("selection plan has no frozen public-validation selection")
    try:
        from trr0005_score_confirmation import validate_public_validation_selection

        normalized = validate_public_validation_selection(selection)
    except Exception as exc:
        raise PredictionRunnerError("frozen public-validation selection is invalid") from exc
    return plan, plan_record, normalized


def _group_descriptors(
    binding: Mapping[str, Any],
    *,
    key: str,
    method_id: str,
) -> list[Mapping[str, Any]]:
    value = binding.get(key)
    if isinstance(value, Mapping):
        value = [value]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, Mapping) for row in value)
    ):
        raise PredictionRunnerError(f"{method_id} binding has no valid {key} group")
    return list(value)


def _binding_entry(entry: Any, *, method_id: str) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise PredictionRunnerError(f"method binding is malformed: {method_id}")
    nested = entry.get("binding")
    if isinstance(nested, Mapping):
        return dict(nested)
    return dict(entry)


def _validate_registration_descriptor(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    description: str,
    external_allowed: bool = False,
) -> tuple[Path, dict[str, Any]]:
    return _asset_path(
        descriptor,
        root=root,
        description=description,
        external_allowed=external_allowed,
    )


def _validate_registration(
    path: Path,
    *,
    repository_root: Path,
    panel_path: Path,
    plan_path: Path,
    panel_record: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    current_commit: str,
) -> tuple[dict[str, Any], dict[str, RegisteredMethod], dict[str, Any]]:
    """Validate all state/code/runtime bytes before any model is loaded."""

    root = repository_root.expanduser().resolve()
    registration = _load_json(path, description="TRR-0005 method registration")
    try:
        validate_registration(registration, require_frozen=True)
    except ContractError as exc:
        raise PredictionRunnerError(
            f"registration failed TRR-0005 contract validation: {exc}"
        ) from exc
    if registration.get("code_commit") != current_commit:
        raise PredictionRunnerError(
            "registration code commit "
            f"{registration.get('code_commit')!r} differs from executable HEAD {current_commit}"
        )

    for key, expected in (("panel", panel_record), ("selection_plan", plan_record)):
        value = registration.get(key)
        if isinstance(value, Mapping):
            bound_path, actual = _validate_registration_descriptor(
                value,
                root=root,
                description=f"registration {key}",
            )
            expected_path = panel_path if key == "panel" else plan_path
            if bound_path != expected_path or dict(actual) != dict(expected):
                raise PredictionRunnerError(f"registration {key} binding changed")

    bindings = registration.get("state_bindings")
    if not isinstance(bindings, Mapping) or tuple(bindings) != METHOD_IDS:
        if not isinstance(bindings, Mapping) or set(bindings) != set(METHOD_IDS):
            raise PredictionRunnerError("registration state bindings are incomplete")
    methods: dict[str, RegisteredMethod] = {}
    shared_embedding_record: dict[str, Any] | None = None
    for method_id in METHOD_IDS:
        binding = _binding_entry(bindings.get(method_id), method_id=method_id)
        if binding.get("method_id") not in (None, method_id):
            raise PredictionRunnerError(f"method ID binding changed: {method_id}")
        bound_panel = binding.get("panel")
        if not isinstance(bound_panel, Mapping):
            raise PredictionRunnerError(f"panel binding is absent: {method_id}")
        panel_bound_path, panel_bound_record = _validate_registration_descriptor(
            bound_panel,
            root=root,
            description=f"bound panel {method_id}",
        )
        if (
            panel_bound_path != panel_path
            or dict(panel_bound_record) != dict(panel_record)
        ):
            raise PredictionRunnerError(f"panel binding changed: {method_id}")

        state_descriptors = _group_descriptors(
            binding, key="method_state", method_id=method_id
        )
        if len(state_descriptors) != 1:
            raise PredictionRunnerError(
                f"{method_id} must bind exactly one state artifact"
            )
        state_path, _ = _validate_registration_descriptor(
            state_descriptors[0],
            root=root,
            description=f"state {method_id}",
        )
        config_paths: list[Path] = []
        for index, descriptor in enumerate(
            _group_descriptors(binding, key="method_config", method_id=method_id)
        ):
            config_path, _ = _validate_registration_descriptor(
                descriptor,
                root=root,
                description=f"method config {method_id}[{index}]",
            )
            config_paths.append(config_path)
        code_paths: list[Path] = []
        for index, descriptor in enumerate(
            _group_descriptors(binding, key="code", method_id=method_id)
        ):
            code_path, _ = _validate_registration_descriptor(
                descriptor,
                root=root,
                description=f"code {method_id}[{index}]",
            )
            if code_path.suffix != ".py":
                raise PredictionRunnerError(
                    f"code binding is not Python source: {code_path}"
                )
            code_paths.append(code_path)

        assets = binding.get("runtime_assets")
        expected_roles = (
            (RUNTIME_EMBEDDING_ROLE, *RUNTIME_P0_ROLES)
            if method_id == A2_METHOD_ID
            else (RUNTIME_EMBEDDING_ROLE,)
        )
        if not isinstance(assets, Mapping) or set(assets) != set(expected_roles):
            raise PredictionRunnerError(
                f"runtime assets for {method_id} must contain exactly {expected_roles!r}"
            )
        runtime_paths: dict[str, Path] = {}
        runtime_records: dict[str, Any] = {}
        for role in expected_roles:
            descriptor = assets.get(role)
            if not isinstance(descriptor, Mapping):
                raise PredictionRunnerError(
                    f"runtime asset is malformed: {method_id}/{role}"
                )
            runtime_path, runtime_record = _validate_registration_descriptor(
                descriptor,
                root=root,
                description=f"runtime asset {method_id}/{role}",
                external_allowed=True,
            )
            runtime_paths[role] = runtime_path
            runtime_records[role] = runtime_record
        if shared_embedding_record is None:
            shared_embedding_record = dict(
                runtime_records[RUNTIME_EMBEDDING_ROLE]
            )
        elif runtime_records[RUNTIME_EMBEDDING_ROLE] != shared_embedding_record:
            raise PredictionRunnerError(
                "methods do not bind the same normalized public embedding table"
            )
        methods[method_id] = RegisteredMethod(
            method_id=method_id,
            binding=binding,
            state_path=state_path,
            config_paths=tuple(config_paths),
            code_paths=tuple(code_paths),
            runtime_paths=runtime_paths,
        )

    a1_path = methods["historical_alpaca_a1"].state_path
    a2_path = methods[A2_METHOD_ID].state_path
    if (
        a1_path.stat().st_size != a2_path.stat().st_size
        or sha256_file(a1_path) != sha256_file(a2_path)
    ):
        raise PredictionRunnerError(
            "A1 and A1+A2 do not bind the same retained lens state"
        )
    driver_path = Path(__file__).resolve()
    bound_code_paths = {
        code_path for method in methods.values() for code_path in method.code_paths
    }
    if driver_path not in bound_code_paths:
        raise PredictionRunnerError(
            "registration does not bind the executed TRR5 prediction driver"
        )
    registration_record = file_record(
        path.expanduser().resolve(), repository_root=root
    )
    return registration, methods, {
        "path": str(path.expanduser().resolve()),
        "file": registration_record,
        "code_commit": current_commit,
        "shared_embedding_record": shared_embedding_record,
    }


class _JointAdapter:
    """Frozen inference adapter for one jointly trained TRR5 decoder."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        embeddings: torch.Tensor,
        method_id: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.embeddings = embeddings
        self.method_id = method_id
        self.device = device
        self.calls = 0
        self._cell_calls = 0

    def begin_cell(self) -> None:
        self._cell_calls = self.calls

    @torch.inference_mode()
    def __call__(
        self,
        row_h: torch.Tensor,
        row_mask: torch.Tensor,
        row_positions: torch.Tensor,
    ) -> torch.Tensor:
        del row_positions
        self.calls += 1
        activation = row_h.to(device=self.device, dtype=torch.float32).unsqueeze(0)
        mask = row_mask.to(device=self.device, dtype=torch.bool).unsqueeze(0)
        logits = self.model(activation, mask, self.embeddings)
        if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, SEQUENCE_TOKENS):
            raise PredictionRunnerError(
                f"joint decoder returned unexpected logits geometry: {self.method_id}"
            )
        return logits.argmax(dim=-1)[0].to(dtype=torch.long)

    def evidence(self) -> dict[str, Any]:
        return {
            "calls": self.calls - self._cell_calls,
            "public_prefix_calls": 0,
            "candidate_simulations": 0,
            "a2_fallback": False,
            "candidate_output": "forbidden",
            "context_width": DEFAULT_CONTEXT_WIDTH,
            "architecture": self.method_id,
        }


def _normalize_prediction(
    raw: Any,
    *,
    mask: torch.Tensor,
    method_id: str,
) -> torch.Tensor:
    try:
        values = torch.as_tensor(raw, dtype=torch.long)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PredictionRunnerError(
            f"{method_id} prediction is not integer-like"
        ) from exc
    if values.ndim != 1 or int(values.shape[0]) != SEQUENCE_TOKENS:
        raise PredictionRunnerError(
            f"{method_id} prediction must have shape [{SEQUENCE_TOKENS}]"
        )
    values = values.contiguous()
    active = mask.to(device=values.device, dtype=torch.bool)
    if not bool(active[0].item()):
        raise PredictionRunnerError("fresh observation row has no active BOS")
    output = torch.full(
        (SEQUENCE_TOKENS,),
        INVALID_TOKEN_ID,
        dtype=torch.long,
        device=values.device,
    )
    output[active] = values[active]
    output[0] = BOS_TOKEN_ID
    scored = output[active]
    if scored.lt(0).any().item() or scored.ge(VOCAB_SIZE).any().item():
        raise PredictionRunnerError(f"{method_id} emitted an invalid active token")
    return output


def _timed_predictor(adapter: Any):
    def predict(record: FreshRecord) -> torch.Tensor:
        # The retained A1 adapter's public implementation expects H/mask/
        # positions on the lens device.  Stage those tensors inside the timed
        # callback so its transfer cost is part of the declared interval.
        if bool(getattr(adapter, "input_device_required", False)):
            target = torch.device(getattr(adapter, "device"))
            activation = record.activation.to(device=target)
            mask = record.attention_mask.to(device=target)
            positions = record.position_ids.to(device=target)
            raw = adapter(activation, mask, positions)
            normalize_mask = mask
        else:
            raw = adapter(record.activation, record.attention_mask, record.position_ids)
            normalize_mask = record.attention_mask
        return _normalize_prediction(
            raw,
            mask=normalize_mask,
            method_id=str(getattr(adapter, "method_id", "unknown")),
        )

    return predict


def _qualification_asset(
    path: Path,
    *,
    root: Path,
    description: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and hash one public qualification asset before model loading."""

    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise PredictionRunnerError(f"{description} is unavailable: {resolved}")
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        record = external_file_record(resolved)
    else:
        record = file_record(resolved, repository_root=root)
    return resolved, dict(record)


def _qualification_method(
    *,
    method_id: str,
    state_path: Path,
    embedding_path: Path,
    root: Path,
    embedding_asset: tuple[Path, dict[str, Any]] | None = None,
    p0_checkpoint: tuple[Path, dict[str, Any]] | None = None,
    p0_config: tuple[Path, dict[str, Any]] | None = None,
) -> RegisteredMethod:
    state_path, state_record = _qualification_asset(
        state_path,
        root=root,
        description=f"qualification state {method_id}",
    )
    if embedding_asset is None:
        embedding_path, embedding_record = _qualification_asset(
            embedding_path,
            root=root,
            description="qualification normalized public E",
        )
    else:
        embedding_path, embedding_record = embedding_asset
    assets: dict[str, Any] = {
        RUNTIME_EMBEDDING_ROLE: embedding_record,
    }
    if method_id == A2_METHOD_ID:
        if p0_checkpoint is None or p0_config is None:
            raise PredictionRunnerError("qualification A2 is missing registered public P0 assets")
        assets["public_prefix_checkpoint"] = dict(p0_checkpoint[1])
        assets["public_prefix_config"] = dict(p0_config[1])
    binding = {
        "method_id": method_id,
        "method_state": [state_record],
        "runtime_assets": assets,
        "qualification_binding": True,
    }
    return RegisteredMethod(
        method_id=method_id,
        binding=binding,
        state_path=state_path,
        config_paths=(),
        code_paths=(),
        runtime_paths={
            role: Path(value["path"])
            for role, value in assets.items()
        },
    )


def _qualification_methods(
    args: argparse.Namespace,
    *,
    root: Path,
) -> tuple[dict[str, RegisteredMethod], dict[str, Any]]:
    required = {
        "--embedding-table": args.embedding_table,
        "--lens": args.lens,
        "--p0-checkpoint": args.p0_checkpoint,
        "--p0-config": args.p0_config,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise PredictionRunnerError(
            "archived qualification requires " + ", ".join(missing)
        )
    embedding_path, embedding_record = _qualification_asset(
        args.embedding_table,
        root=root,
        description="qualification normalized public E",
    )
    lens_path, lens_record = _qualification_asset(
        args.lens,
        root=root,
        description="qualification retained A1 lens",
    )
    p0_checkpoint, p0_checkpoint_record = _qualification_asset(
        args.p0_checkpoint,
        root=root,
        description="qualification public P0 checkpoint",
    )
    p0_config, p0_config_record = _qualification_asset(
        args.p0_config,
        root=root,
        description="qualification public P0 config",
    )
    p0 = (p0_checkpoint, p0_checkpoint_record)
    config = (p0_config, p0_config_record)
    methods: dict[str, RegisteredMethod] = {}
    methods["historical_alpaca_a1"] = _qualification_method(
        method_id="historical_alpaca_a1",
        state_path=lens_path,
        embedding_path=embedding_path,
        root=root,
        embedding_asset=(embedding_path, embedding_record),
    )
    methods[A2_METHOD_ID] = _qualification_method(
        method_id=A2_METHOD_ID,
        state_path=lens_path,
        embedding_path=embedding_path,
        root=root,
        embedding_asset=(embedding_path, embedding_record),
        p0_checkpoint=p0,
        p0_config=config,
    )
    fit_root, causal_fit_root = _resolve_fit_roots(args)
    for distribution in ("original", "enriched"):
        for base_method in JOINT_STATE_METHODS:
            method_id = f"{distribution}__{base_method}"
            state_root = causal_fit_root if base_method == "affine_causal_h_attention128" else fit_root
            state_path = state_root / distribution / base_method / "selected.safetensors"
            methods[method_id] = _qualification_method(
                method_id=method_id,
                state_path=state_path,
                embedding_path=embedding_path,
                root=root,
                embedding_asset=(embedding_path, embedding_record),
            )
    expected = {
        "historical_alpaca_a1",
        A2_METHOD_ID,
        *(f"{distribution}__{base_method}" for distribution in ("original", "enriched") for base_method in JOINT_STATE_METHODS),
    }
    if set(methods) != expected:
        raise PredictionRunnerError("qualification method set is incomplete")
    return methods, {
        "normalized_embedding": embedding_record,
        "retained_lens": lens_record,
        "public_prefix_checkpoint": p0_checkpoint_record,
        "public_prefix_config": p0_config_record,
        "fit_root": str(fit_root),
        "causal_fit_root": str(causal_fit_root),
    }


def _resolve_fit_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the fitted-state root and the independently repairable causal root."""

    fit_root = args.fit_root.expanduser().resolve()
    supplied_causal = getattr(args, "causal_fit_root", None)
    causal_fit_root = (
        supplied_causal.expanduser().resolve()
        if supplied_causal is not None
        else DEFAULT_CAUSAL_FIT_ROOT.expanduser().resolve()
    )
    return fit_root, causal_fit_root


def _validate_joint_state_roots(
    methods: Mapping[str, RegisteredMethod],
    *,
    fit_root: Path,
    causal_fit_root: Path,
) -> None:
    """Require registration paths to expose the deployed base/repair split."""

    for distribution in ("original", "enriched"):
        for base_method in JOINT_STATE_METHODS:
            method_id = f"{distribution}__{base_method}"
            root = causal_fit_root if base_method == "affine_causal_h_attention128" else fit_root
            expected = (
                root / distribution / base_method / "selected.safetensors"
            ).expanduser().resolve()
            actual = methods[method_id].state_path.expanduser().resolve()
            if actual != expected:
                raise PredictionRunnerError(
                    f"{method_id} state binding does not match its canonical root: "
                    f"expected {expected}, got {actual}"
                )


def _load_archived_qualification_record(
    path: Path,
    *,
    record_index: int,
    record_id: str,
) -> tuple[FreshRecord, dict[str, Any]]:
    """Load one archived public H row, without opening token IDs or truth."""

    if record_index < 0:
        raise PredictionRunnerError("qualification record index must be nonnegative")
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PredictionRunnerError(f"archived qualification observation is unavailable: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != {"activations"}:
                raise PredictionRunnerError(
                    "archived qualification observation must contain only activations"
                )
            activations = handle.get_tensor("activations")
            if activations.ndim != 3 or tuple(activations.shape[1:]) != (SEQUENCE_TOKENS, HIDDEN_SIZE):
                raise PredictionRunnerError("archived qualification observation geometry changed")
            if record_index >= int(activations.shape[0]):
                raise PredictionRunnerError("qualification record index exceeds archived observation")
            activation = activations[record_index].contiguous()
            metadata = dict(handle.metadata() or {})
    except PredictionRunnerError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise PredictionRunnerError("archived qualification observation is unreadable") from exc
    if not activation.dtype.is_floating_point or not torch.isfinite(activation).all().item():
        raise PredictionRunnerError("archived qualification activation is invalid")
    if not isinstance(record_id, str) or not record_id:
        raise PredictionRunnerError("qualification record ID is empty")
    mask = torch.ones(SEQUENCE_TOKENS, dtype=torch.bool)
    positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.long)
    return FreshRecord(
        record_id=record_id,
        activation=activation,
        attention_mask=mask,
        position_ids=positions,
    ), {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "tensor_key": "activations",
        "shape": [int(activations.shape[0]), SEQUENCE_TOKENS, HIDDEN_SIZE],
        "record_index": int(record_index),
        "record_id": record_id,
        "metadata": metadata,
    }


def _load_archived_prediction_row(
    path: Path,
    *,
    record_index: int,
    method_id: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Read only one archived prediction row for an exact adapter check."""

    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PredictionRunnerError(f"archived prediction is unavailable: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if "predictions" not in keys:
                raise PredictionRunnerError("archived prediction tensor is absent")
            shape = tuple(handle.get_slice("predictions").get_shape())
            if len(shape) != 2 or shape[1] != SEQUENCE_TOKENS or record_index >= shape[0]:
                raise PredictionRunnerError("archived prediction geometry changed")
            predictions = handle.get_tensor("predictions")[record_index].contiguous()
            metadata = dict(handle.metadata() or {})
    except PredictionRunnerError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError, AttributeError) as exc:
        raise PredictionRunnerError("archived prediction is unreadable") from exc
    if metadata.get("task_id") != "TRR-0004" or metadata.get("method_id") != method_id:
        raise PredictionRunnerError(f"archived prediction binding changed: {path}")
    if metadata.get("cell_id") != "finance__public_base":
        raise PredictionRunnerError(f"archived prediction is not Finance public_base: {path}")
    row = predictions.to(dtype=torch.long, device="cpu").contiguous()
    if row[0].item() != BOS_TOKEN_ID:
        raise PredictionRunnerError(f"archived {method_id} prediction lacks BOS")
    if row.lt(0).any().item() or row.ge(VOCAB_SIZE).any().item():
        raise PredictionRunnerError(f"archived {method_id} prediction has invalid IDs")
    return row, {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "tensor_key": "predictions",
        "shape": list(shape),
        "record_index": int(record_index),
        "method_id": method_id,
        "metadata": metadata,
    }


def _one_record_warm_measured(
    adapter: Any,
    record: FreshRecord,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run exactly one warmup and one measured call for a qualifier row."""

    synchronize = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else (lambda: None)
    )
    predictor = _timed_predictor(adapter)
    started = time.perf_counter()
    warmup = predictor(record)
    synchronize()
    warmup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    measured = predictor(record)
    synchronize()
    measured_seconds = time.perf_counter() - started
    if not torch.equal(warmup.to("cpu"), measured.to("cpu")):
        raise PredictionRunnerError(
            f"qualification warmup/measured IDs differ for {record.record_id}"
        )
    return measured.to(device="cpu", dtype=torch.long).contiguous(), {
        "schema": "token-reconstruction.trr0005-qualification-timing.v1",
        "records": 1,
        "sequence_tokens": SEQUENCE_TOKENS,
        "warmup_runs_per_record": 1,
        "measured_runs_per_record": 1,
        "warmup_seconds": float(warmup_seconds),
        "measured_seconds": float(measured_seconds),
        "timed_interval_total_seconds": float(warmup_seconds + measured_seconds),
        "adapter_calls_total": 2,
        "warmup_adapter_calls": 1,
        "measured_adapter_calls": 1,
        "adapter_call_scope": "one warmup plus one measured call for this record",
        "warmup_output_exact_match_measured": True,
        "measured_output_selected": True,
    }


def _run_archived_qualification(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise A1/A2 and all six selected joint decoders on one public H row."""

    root = args.repository_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise PredictionRunnerError(
            f"qualification output root must be new: {output_root}"
        )
    try:
        output_root.relative_to(root)
    except ValueError as exc:
        raise PredictionRunnerError("qualification output root must be inside repository root") from exc
    if args.qualification_observation is None:
        raise PredictionRunnerError("archived qualification requires --qualification-observation")
    if args.qualification_a1_archive is None or args.qualification_a2_archive is None:
        raise PredictionRunnerError(
            "archived qualification requires --qualification-a1-archive and --qualification-a2-archive"
        )
    if args.model_snapshot is None or args.reference is None:
        raise PredictionRunnerError(
            "archived qualification requires --model-snapshot and --reference"
        )
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = _git_commit(root)
    device = _choose_device(args.device)
    if device.type != "cuda":
        raise PredictionRunnerError("archived A1/A2 qualification requires CUDA")
    output_root.mkdir(parents=True)
    try:
        record_id = args.qualification_record_id or f"archived-finance-public-{args.qualification_record_index:03d}"
        record, observation = _load_archived_qualification_record(
            args.qualification_observation,
            record_index=int(args.qualification_record_index),
            record_id=record_id,
        )
        methods, runtime = _qualification_methods(args, root=root)
        archived = {
            "a1": _load_archived_prediction_row(
                args.qualification_a1_archive,
                record_index=int(args.qualification_record_index),
                method_id="historical_alpaca_a1",
            ),
            "a2": _load_archived_prediction_row(
                args.qualification_a2_archive,
                record_index=int(args.qualification_record_index),
                method_id=A2_METHOD_ID,
            ),
        }
        method_results: dict[str, Any] = {}
        for method_id in (
            "historical_alpaca_a1",
            A2_METHOD_ID,
            *(f"{distribution}__{base_method}" for distribution in ("original", "enriched") for base_method in JOINT_STATE_METHODS),
        ):
            _resource_guard(
                args,
                device=device,
                stage=f"before_{method_id}_qualification_load",
                started=started,
            )
            method = methods[method_id]
            adapter: Any | None = None
            embeddings: torch.Tensor | None = None
            load_started = time.perf_counter()
            try:
                adapter, embeddings, load_evidence = _load_adapter(
                    method,
                    args=args,
                    device=device,
                )
                loaded_seconds = time.perf_counter() - load_started
                _resource_guard(
                    args,
                    device=device,
                    stage=f"after_{method_id}_qualification_load",
                    started=started,
                )
                begin_cell = getattr(adapter, "begin_cell", None)
                if callable(begin_cell):
                    begin_cell()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                prediction, timing = _one_record_warm_measured(
                    adapter,
                    record,
                    device=device,
                )
                expected = None
                exact_match = None
                if method_id == "historical_alpaca_a1":
                    expected = archived["a1"][0]
                elif method_id == A2_METHOD_ID:
                    expected = archived["a2"][0]
                if expected is not None:
                    exact_match = bool(torch.equal(prediction, expected))
                    if not exact_match:
                        raise PredictionRunnerError(
                            f"{method_id} differs from archived Finance prediction"
                        )
                method_results[method_id] = {
                    "method_id": method_id,
                    "state": {
                        "path": str(method.state_path),
                        "bytes": int(method.state_path.stat().st_size),
                        "sha256": sha256_file(method.state_path),
                    },
                    "runtime_load": dict(load_evidence),
                    "state_load_seconds": float(loaded_seconds),
                    "timing": timing,
                    "prediction_sha256": _tensor_digest(prediction),
                    "prediction_shape": list(prediction.shape),
                    "active_tokens": int(record.attention_mask.sum().item()),
                    "archived_exact_match": exact_match,
                    "method_specific": _adapter_evidence(adapter, method_id=method_id),
                    "peak_memory": _peak_memory(device),
                }
            finally:
                _clear_ephemeral_a2(adapter)
                del adapter
                del embeddings
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            _resource_guard(
                args,
                device=device,
                stage=f"after_{method_id}_qualification",
                started=started,
            )
        driver_record = file_record(Path(__file__).resolve(), repository_root=root)
        evidence = {
            "schema": QUALIFICATION_SCHEMA,
            "task_id": TASK_ID,
            "status": "ARCHIVED_FINANCE_QUALIFICATION_COMPLETE_NO_TRUTH",
            "started_utc": started_utc,
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.perf_counter() - started,
            "git_commit": commit,
            "driver": driver_record,
            "command": {
                "argv": list(sys.argv),
                "cwd": str(Path.cwd()),
                "python": sys.executable,
            },
            "device": str(device),
            "record": {
                "record_id": record.record_id,
                "record_index": int(args.qualification_record_index),
                "sequence_tokens": SEQUENCE_TOKENS,
                "hidden_size": HIDDEN_SIZE,
                "active_tokens": int(record.attention_mask.sum().item()),
                "observation": observation,
                "a1_archived_prediction": archived["a1"][1],
                "a2_archived_prediction": archived["a2"][1],
            },
            "runtime_assets": runtime,
            "methods": method_results,
            "method_count": len(method_results),
            "joint_method_count": sum(1 for method_id in method_results if "__" in method_id),
            "warmup_runs_per_record": 1,
            "measured_runs_per_record": 1,
            "warmup_measured_ids_exact": all(
                bool(value["timing"]["warmup_output_exact_match_measured"])
                for value in method_results.values()
            ),
            "archived_a1_a2_ids_exact": all(
                method_results[method_id]["archived_exact_match"] is True
                for method_id in ("historical_alpaca_a1", A2_METHOD_ID)
            ),
            "fresh_panel_loaded": False,
            "truth_opened": False,
            "target_labels_loaded": False,
            "future_activation_reads": False,
            "candidate_arrays_persisted": False,
        }
        _write_create_only(output_root / "qualification.json", evidence)
        return evidence
    except Exception as exc:
        failure_path = output_root / "failure.json"
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_create_only(
                failure_path,
                {
                    "schema": QUALIFICATION_FAILURE_SCHEMA,
                    "task_id": TASK_ID,
                    "status": "FAILED_PRESERVED_NO_TRUTH",
                    "started_utc": started_utc,
                    "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "argv": list(sys.argv),
                    "git_commit": commit,
                    "truth_opened": False,
                    "target_labels_loaded": False,
                },
            )
        raise


def _runtime_embedding(
    path: Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        embeddings, evidence = legacy._load_normalized_embeddings(
            path=path,
            device=device,
        )
    except Exception as exc:
        raise PredictionRunnerError(
            f"normalized public embedding load failed: {path}"
        ) from exc
    if tuple(embeddings.shape) != (VOCAB_SIZE, HIDDEN_SIZE):
        raise PredictionRunnerError("normalized public embedding geometry changed")
    return embeddings, evidence


def _load_joint_adapter(
    method: RegisteredMethod,
    *,
    device: torch.device,
) -> tuple[_JointAdapter, dict[str, Any]]:
    base_method = method.method_id.split("__", 1)[1]
    if base_method not in JOINT_STATE_METHODS:
        raise PredictionRunnerError(f"unsupported joint state ID: {base_method}")
    started = time.perf_counter()
    try:
        model = load_decoder_state(
            method.state_path,
            method_id=base_method,
            hidden_size=HIDDEN_SIZE,
            vocabulary_size=VOCAB_SIZE,
            context_width=DEFAULT_CONTEXT_WIDTH,
        ).to(device=device).eval()
        model.requires_grad_(False)
    except (JointDecoderError, RuntimeError, ValueError) as exc:
        raise PredictionRunnerError(
            f"joint decoder state could not be loaded: {method.state_path}"
        ) from exc
    return _JointAdapter(
        model=model,
        embeddings=torch.empty(0),
        method_id=method.method_id,
        device=device,
    ), {
        "loader": "token_reconstruction.trr0005_joint_decoder.load_decoder_state",
        "state_path": str(method.state_path),
        "state_bytes": int(method.state_path.stat().st_size),
        "state_sha256": sha256_file(method.state_path),
        "load_seconds": time.perf_counter() - started,
        "method_id": method.method_id,
        "base_method_id": base_method,
    }


def _load_adapter(
    method: RegisteredMethod,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    """Load exactly one method and its registered public runtime resources."""

    embedding_path = method.runtime_paths[RUNTIME_EMBEDDING_ROLE]
    method_id = method.method_id
    resource_started = time.perf_counter()
    if method_id == A2_METHOD_ID:
        if args.model_snapshot is None or args.reference is None:
            raise PredictionRunnerError(
                "A2 requires --model-snapshot and --reference for its registered public P0 path"
            )
        lens_path = method.state_path
        if args.lens is not None:
            explicit_lens = args.lens.expanduser().resolve()
            if explicit_lens != lens_path or sha256_file(explicit_lens) != sha256_file(lens_path):
                raise PredictionRunnerError(
                    "explicit --lens differs from the registered retained A1 state"
                )
            lens_path = explicit_lens
        try:
            precut, lens, embeddings, public_evidence = legacy._load_public_prefix(
                snapshot=args.model_snapshot.expanduser().resolve(),
                reference_path=args.reference.expanduser().resolve(),
                lens_path=lens_path,
                embedding_path=embedding_path,
                device=device,
            )
            legacy_module = __import__("trr0003_footing_compare")
            policy = legacy_module._fixed_k256_policy()
        except Exception as exc:
            raise PredictionRunnerError("A2 public P0/A1 resource load failed") from exc
        adapter = legacy._A2Adapter(
            precut=precut,
            lens=lens,
            embeddings=embeddings,
            device=device,
            policy=policy,
        )
        adapter.method_id = method_id
        evidence = {
            "loader_scope": "retained A1+A2 output-only adapter with registered public P0",
            "public_prefix_loaded": True,
            "public_resources": public_evidence,
            "runtime_load_seconds": time.perf_counter() - resource_started,
        }
        return adapter, embeddings, evidence

    if method_id == "historical_alpaca_a1":
        lens, embeddings_evidence = None, None
        embeddings, embedding_evidence = _runtime_embedding(
            embedding_path,
            device=device,
        )
        try:
            lens = load_historical_lens_checkpoint(
                method.state_path,
                device=device,
            )
        except Exception as exc:
            raise PredictionRunnerError(
                f"retained A1 lens load failed: {method.state_path}"
            ) from exc
        adapter = legacy._A1Adapter(lens=lens, embeddings=embeddings)
        adapter.method_id = method_id
        # _A1Adapter is inherited from TRR4 and has no explicit device field;
        # expose the registered runtime device for the timed staging boundary.
        adapter.device = device
        adapter.input_device_required = True
        evidence = {
            "loader_scope": "retained standalone A1 and registered normalized public E; no P0",
            "public_prefix_loaded": False,
            "public_resources": {
                "normalized_embedding": embedding_evidence,
                "retained_lens": {
                    "path": str(method.state_path),
                    "bytes": int(method.state_path.stat().st_size),
                    "sha256": sha256_file(method.state_path),
                    "loader": "token_reconstruction.historical_inputlens_bridge.load_historical_lens_checkpoint",
                },
            },
            "runtime_load_seconds": time.perf_counter() - resource_started,
        }
        return adapter, embeddings, evidence

    embeddings, embedding_evidence = _runtime_embedding(
        embedding_path,
        device=device,
    )
    adapter, state_evidence = _load_joint_adapter(method, device=device)
    adapter.embeddings = embeddings
    evidence = {
        "loader_scope": "joint TRR5 decoder and registered normalized public E; no P0",
        "public_prefix_loaded": False,
        "public_resources": {"normalized_embedding": embedding_evidence},
        "state": state_evidence,
        "runtime_load_seconds": time.perf_counter() - resource_started,
    }
    return adapter, embeddings, evidence


def _adapter_evidence(adapter: Any, *, method_id: str) -> dict[str, Any]:
    value = getattr(adapter, "evidence", None)
    result = dict(value()) if callable(value) else {}
    result["method_id"] = method_id
    result["calls_scope"] = "warmup plus measured calls for every record in the cell"
    if method_id == A2_METHOD_ID:
        result.update(
            {
                "candidate_output": "omitted_after_decision",
                "candidate_arrays_persisted": False,
                "candidate_policy": CANDIDATE_POLICIES[method_id],
                "a2_fallback": False,
            }
        )
    else:
        result.setdefault("candidate_output", "forbidden")
        result.setdefault("candidate_arrays_persisted", False)
    return result


def _clear_ephemeral_a2(adapter: Any) -> None:
    # The inherited A2 helper may retain a diagnostic proposal in memory.  It
    # is never written and is cleared before proceeding to the next cell.
    values = getattr(adapter, "_record_proposals", None)
    if isinstance(values, list):
        values.clear()


def _host_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                return int(fields[1]) * 1024
    except (OSError, UnicodeError):
        return None
    return None


def _resource_guard(
    args: argparse.Namespace,
    *,
    device: torch.device,
    stage: str,
    started: float,
) -> dict[str, Any]:
    if time.perf_counter() - started > float(args.max_seconds):
        raise PredictionRunnerError(f"wall-time guard expired at {stage}")
    rss = _rusage_rss_bytes()
    maximum_rss = int(float(args.maximum_rss_gib) * 2**30)
    if rss > maximum_rss:
        raise PredictionRunnerError(
            f"host RSS guard failed at {stage}: {rss} > {maximum_rss} bytes"
        )
    available = _host_available_bytes()
    minimum_available = int(float(args.minimum_host_available_gib) * 2**30)
    if available is not None and available < minimum_available:
        raise PredictionRunnerError(
            f"host available-memory guard failed at {stage}: "
            f"{available} < {minimum_available} bytes"
        )
    try:
        gpu = legacy._resource_preflight(
            args,
            device,
            stage=stage,
            started=started,
        )
    except Exception as exc:
        if isinstance(exc, PredictionRunnerError):
            raise
        raise PredictionRunnerError(f"GPU/resource guard failed at {stage}") from exc
    return {
        "stage": stage,
        "host_rss_bytes": rss,
        "host_available_bytes": available,
        "minimum_host_available_bytes": minimum_available,
        "gpu": gpu,
    }


def _peak_memory(device: torch.device) -> dict[str, int | None]:
    peak: dict[str, int | None] = {
        "process_max_rss_bytes": _rusage_rss_bytes(),
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        peak["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        peak["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return peak


def _tensor_digest(value: torch.Tensor) -> str:
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


def _validate_prediction_batch(
    predictions: torch.Tensor,
    *,
    cell: FreshCell,
    method_id: str,
) -> torch.Tensor:
    value = predictions.to(device="cpu", dtype=torch.long).contiguous()
    if tuple(value.shape) != (RECORDS_PER_DOMAIN, SEQUENCE_TOKENS):
        raise PredictionRunnerError(
            f"prediction geometry changed: {cell.cell_id}/{method_id}"
        )
    active = cell.attention_mask.to(torch.bool)
    if not value[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise PredictionRunnerError(
            f"prediction BOS changed: {cell.cell_id}/{method_id}"
        )
    if value[active].lt(0).any().item() or value[active].ge(VOCAB_SIZE).any().item():
        raise PredictionRunnerError(
            f"prediction ID range changed: {cell.cell_id}/{method_id}"
        )
    if value[~active].ne(INVALID_TOKEN_ID).any().item():
        raise PredictionRunnerError(
            f"prediction padding changed: {cell.cell_id}/{method_id}"
        )
    return value


def _cell_records(cell: FreshCell) -> tuple[FreshRecord, ...]:
    return tuple(
        FreshRecord(
            record_id=cell.record_ids[index],
            activation=cell.activations[index].contiguous(),
            attention_mask=cell.attention_mask[index].contiguous(),
            position_ids=cell.position_ids[index].contiguous(),
        )
        for index in range(cell.records)
    )


def _fit_evidence_summary(
    fit_root: Path,
    *,
    causal_fit_root: Path | None = None,
    repository_root: Path,
) -> dict[str, Any]:
    """Copy public fit curves/costs using the deployed base/causal roots."""

    fitted_root = fit_root.expanduser().resolve()
    causal_root = (
        causal_fit_root.expanduser().resolve()
        if causal_fit_root is not None
        else fitted_root
    )

    def load_source(root: Path, label: str) -> dict[str, Any]:
        evidence_path = root / "run_evidence.json"
        evidence = _load_json(evidence_path, description=f"TRR-0005 {label} fit evidence")
        if evidence.get("task_id") != TASK_ID or evidence.get("final_holdout_loaded") is True:
            raise PredictionRunnerError(
                f"{label} fit evidence is not an accepted public development run"
            )
        distributions = evidence.get("distributions")
        if not isinstance(distributions, Mapping):
            raise PredictionRunnerError(f"{label} fit evidence has no distribution summaries")
        return {
            "label": label,
            "root": root,
            "path": evidence_path,
            "evidence": evidence,
            "distributions": distributions,
            "record": file_record(evidence_path, repository_root=repository_root),
        }

    fitted = load_source(fitted_root, "fitted")
    causal = fitted if causal_root == fitted_root else load_source(causal_root, "causal")
    fixed = fitted["evidence"].get("fixed_settings")
    causal_fixed = causal["evidence"].get("fixed_settings")
    result: dict[str, Any] = {
        # Keep the historical top-level fields for readers, while making the
        # deployed source of every method explicit below.
        "path": str(fitted["path"]),
        "file": dict(fitted["record"]),
        "fit_roots": {
            "fitted": str(fitted_root),
            "causal": str(causal_root),
        },
        "sources": {
            label: {
                "root": str(source["root"]),
                "path": str(source["path"]),
                "file": dict(source["record"]),
                "git_commit": source["evidence"].get("git_commit"),
                "status": source["evidence"].get("status"),
            }
            for label, source in (("fitted", fitted), ("causal", causal))
        },
        "git_commit": fitted["evidence"].get("git_commit"),
        "causal_git_commit": causal["evidence"].get("git_commit"),
        "status": fitted["evidence"].get("status"),
        "elapsed_seconds": fitted["evidence"].get("elapsed_seconds"),
        "fixed_settings": {
            key: fixed.get(key)
            for key in (
                "methods",
                "optimizer",
                "learning_rate",
                "gradient_clip_norm",
                "position_budget",
                "qkv_init_seed",
                "output_correction_initialization",
                "identity_affine_initialization",
                "attention_score_mode",
                "causal_only",
            )
            if isinstance(fixed, Mapping) and key in fixed
        },
        "causal_fixed_settings": {
            key: causal_fixed.get(key)
            for key in (
                "methods",
                "optimizer",
                "learning_rate",
                "gradient_clip_norm",
                "position_budget",
                "qkv_init_seed",
                "output_correction_initialization",
                "identity_affine_initialization",
                "attention_score_mode",
                "causal_only",
            )
            if isinstance(causal_fixed, Mapping) and key in causal_fixed
        },
        "sampler_cross_distribution": fitted["evidence"].get("sampler_cross_distribution"),
        "causal_sampler_cross_distribution": causal["evidence"].get(
            "sampler_cross_distribution"
        ),
        "distributions": {},
    }
    for distribution in ("original", "enriched"):
        row = fitted["distributions"].get(distribution)
        if not isinstance(row, Mapping):
            raise PredictionRunnerError(
                f"fitted fit evidence is missing distribution summary: {distribution}"
            )
        causal_row = causal["distributions"].get(distribution)
        if not isinstance(causal_row, Mapping):
            raise PredictionRunnerError(
                f"causal fit evidence is missing distribution summary: {distribution}"
            )
        fitted_method_rows = row.get("methods")
        causal_method_rows = causal_row.get("methods")
        if not isinstance(fitted_method_rows, Mapping) or not isinstance(causal_method_rows, Mapping):
            raise PredictionRunnerError(
                f"fit evidence is missing method curves: {distribution}"
            )
        selected: dict[str, Any] = {}
        method_sources: dict[str, str] = {}
        for method_name in fitted_method_rows:
            source = (
                causal
                if str(method_name) == "affine_causal_h_attention128"
                else fitted
            )
            source_row = source["distributions"].get(distribution)
            source_method_rows = source_row.get("methods") if isinstance(source_row, Mapping) else None
            if not isinstance(source_method_rows, Mapping):
                raise PredictionRunnerError(
                    f"{source['label']} fit evidence is missing method curves: {distribution}"
                )
            method_row = source_method_rows.get(method_name)
            if not isinstance(method_row, Mapping):
                raise PredictionRunnerError(
                    f"{source['label']} fit evidence is missing method: {distribution}/{method_name}"
                )
            curve = method_row.get("curve")
            curve_summary = None
            if isinstance(curve, Mapping) and isinstance(curve.get("path"), str):
                curve_path = Path(curve["path"]).expanduser().resolve()
                try:
                    curve_file = file_record(curve_path, repository_root=repository_root)
                except (FootingError, OSError, ValueError):
                    # Curves may be in a sibling worktree only when the fit
                    # evidence explicitly bound them; retain the source record.
                    curve_file = {
                        "path": str(curve_path),
                        "bytes": curve.get("bytes"),
                        "sha256": curve.get("sha256"),
                    }
                curve_summary = {
                    "path": str(curve_path),
                    "bytes": curve.get("bytes"),
                    "sha256": curve.get("sha256"),
                    "points": curve.get("points"),
                    "current_file": curve_file,
                }
            method_sources[str(method_name)] = source["label"]
            selected[str(method_name)] = {
                "canonical_method_id": method_row.get("canonical_method_id"),
                "selected_step": method_row.get("selected_step"),
                "best_validation_style_balanced_token_accuracy": method_row.get(
                    "best_validation_style_balanced_token_accuracy"
                ),
                "checkpoint_steps": method_row.get("checkpoint_steps"),
                "curve": curve_summary,
                "optimization_update_seconds": method_row.get(
                    "optimization_update_seconds"
                ),
                "selection_validation_seconds": method_row.get(
                    "selection_validation_seconds"
                ),
                "final_fit_diagnostic_seconds": method_row.get(
                    "final_fit_diagnostic_seconds"
                ),
                "state_io_seconds": method_row.get("state_io_seconds"),
                "arm_wall_seconds": method_row.get("arm_wall_seconds"),
                "timing_accounting": method_row.get("timing_accounting"),
                "source_fit_root": str(source["root"]),
                "source_evidence_path": str(source["path"]),
                "source_attention_score_mode": source["evidence"].get(
                    "fixed_settings", {}
                ).get("attention_score_mode")
                if isinstance(source["evidence"].get("fixed_settings"), Mapping)
                else None,
            }
        result["distributions"][distribution] = {
            "contract_distribution_id": row.get("contract_distribution_id"),
            "fit_geometry": row.get("fit_geometry"),
            "fit_record_count": row.get("fit_record_count"),
            "fit_post_bos_positions": row.get("fit_post_bos_positions"),
            "validation_geometry": row.get("validation_geometry"),
            "preparation_timing": row.get("preparation_timing"),
            "method_sources": method_sources,
            "methods": selected,
        }
    return result


def _method_timing_receipt(
    *,
    timing: Mapping[str, Any],
    cell: FreshCell,
    method_id: str,
    adapter: Any,
    artifact: Mapping[str, Any],
    load_evidence: Mapping[str, Any],
    peak: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    method_specific = _adapter_evidence(adapter, method_id=method_id)
    value = dict(timing)
    # The warmed predictor carries a legacy timing schema; it must not
    # overwrite the prediction descriptor schema when merged.
    value.pop("schema", None)
    value.update(
        {
            # Keep the stable prediction descriptor's ``schema`` intact
            # when this timing mapping is merged into it.  The receipt's
            # timing identity lives in a separate field.
            "timing_schema": "token-reconstruction.trr0005-prediction-receipt.v1",
            "task_id": TASK_ID,
            "cell_id": cell.cell_id,
            "method_id": method_id,
            "records": cell.records,
            # ``shape`` belongs to the prediction descriptor contract and must
            # remain [records, sequence_tokens] after this receipt is merged
            # into it.  Keep the activation geometry under its own key.
            "shape": [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS],
            "observation_shape": list(cell.shape),
            "active_tokens": int(cell.attention_mask.to(torch.bool).sum().item()),
            "scored_tokens": int(cell.attention_mask.to(torch.bool).sum().item())
            - cell.records,
            "steady_interval": (
                "CPU activation H -> device preprocessing -> method execution "
                "-> predicted IDs CPU"
            ),
            "synchronization": (
                "torch.cuda.synchronize after each warmup and measured call"
                if getattr(adapter, "device", torch.device("cpu")).type == "cuda"
                else "host-only synchronization callback"
            ),
            "cold_costs_separate": True,
            "runtime_load_seconds": load_evidence.get("runtime_load_seconds"),
            "peak_memory": dict(peak),
            "method_specific": method_specific,
            "adapter_calls_total": int(method_specific.get("calls", 0)),
            "warmup_adapter_calls": cell.records,
            "measured_adapter_calls": cell.records,
            "adapter_call_scope": "warmup plus measured calls for every record in the cell",
            "prediction_artifact": dict(artifact),
            "prediction_sha256": _tensor_digest(
                torch.as_tensor(timing["predictions_for_digest"])
            )
            if "predictions_for_digest" in timing
            else None,
            "artifact_relative_to_root": str(
                Path(str(artifact["path"])).as_posix()
            ),
        }
    )
    value.pop("predictions_for_digest", None)
    try:
        value["artifact_relative_to_root"] = str(
            Path(str(artifact["path"])).resolve().relative_to(root.resolve()).as_posix()
        )
    except ValueError:
        value["artifact_relative_to_root"] = str(artifact["path"])
    return value


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise PredictionRunnerError(f"invalid device: {value}") from exc
    if device.type not in ("cpu", "cuda"):
        raise PredictionRunnerError("device must be cpu, cuda, or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PredictionRunnerError("CUDA was requested but is unavailable")
    return device


def _run_method(
    *,
    method: RegisteredMethod,
    cells: Sequence[FreshCell],
    args: argparse.Namespace,
    device: torch.device,
    root: Path,
    output_root: Path,
    panel_sha256: str,
    selection_plan_sha256: str,
    started: float,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, Any],
]:
    method_id = method.method_id
    guards: list[dict[str, Any]] = []
    guards.append(
        _resource_guard(
            args,
            device=device,
            stage=f"before_{method_id}_resource_load",
            started=started,
        )
    )
    load_started = time.perf_counter()
    adapter: Any | None = None
    embeddings: torch.Tensor | None = None
    try:
        adapter, embeddings, load_evidence = _load_adapter(
            method,
            args=args,
            device=device,
        )
        load_evidence = dict(load_evidence)
        load_evidence.setdefault(
            "runtime_load_seconds", time.perf_counter() - load_started
        )
        guards.append(
            _resource_guard(
                args,
                device=device,
                stage=f"after_{method_id}_resource_load",
                started=started,
            )
        )
        predictions: dict[tuple[str, str], dict[str, Any]] = {}
        timings: dict[tuple[str, str], dict[str, Any]] = {}
        method_cells: list[dict[str, Any]] = []
        for cell in cells:
            guards.append(
                _resource_guard(
                    args,
                    device=device,
                    stage=f"before_{cell.cell_id}_{method_id}",
                    started=started,
                )
            )
            begin_cell = getattr(adapter, "begin_cell", None)
            if callable(begin_cell):
                begin_cell()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            records = _cell_records(cell)
            synchronize = (
                (lambda: torch.cuda.synchronize(device))
                if device.type == "cuda"
                else (lambda: None)
            )
            timed_started = time.perf_counter()
            prediction, timing = run_warmed_prediction(
                method_id=method_id,
                records=records,
                predict_one=_timed_predictor(adapter),
                warmup_runs_per_record=1,
                measured_runs_per_record=1,
                synchronize=synchronize,
            )
            timed_elapsed = time.perf_counter() - timed_started
            prediction_cpu = _validate_prediction_batch(
                prediction,
                cell=cell,
                method_id=method_id,
            )
            peak = _peak_memory(device)
            method_specific = _adapter_evidence(adapter, method_id=method_id)
            descriptor = prediction_descriptor(
                cell_id=cell.cell_id,
                method_id=method_id,
                predictions=prediction_cpu,
                timing=timing,
                panel_sha256=panel_sha256,
                selection_plan_sha256=selection_plan_sha256,
                observation_sha256=cell.observation_sha256,
                candidate_budget=(
                    256 if method_id == A2_METHOD_ID else None
                ),
                public_prefix_calls=int(
                    method_specific.get("public_prefix_calls", 0)
                ),
                candidate_simulations=int(
                    method_specific.get("candidate_simulations", 0)
                ),
            )
            artifact_path = (
                output_root
                / cell.style
                / cell.condition
                / f"{method_id}.safetensors"
            )
            artifact = write_prediction_artifact(
                artifact_path,
                cell_id=cell.cell_id,
                method_id=method_id,
                predictions=prediction_cpu,
                binding=method.binding,
                panel_sha256=panel_sha256,
                selection_plan_sha256=selection_plan_sha256,
                observation_sha256=cell.observation_sha256,
                repository_root=root,
                hidden_size=HIDDEN_SIZE,
                cut_depth=CUT_DEPTH,
            )
            descriptor["prediction_artifact"] = dict(artifact)
            descriptor["method_specific"] = method_specific
            descriptor["peak_memory"] = dict(peak)
            descriptor["runtime_load_seconds"] = load_evidence.get(
                "runtime_load_seconds"
            )
            descriptor["measured_elapsed_seconds"] = timed_elapsed
            timing_value = _method_timing_receipt(
                timing={
                    **timing,
                    "predictions_for_digest": prediction_cpu,
                },
                cell=cell,
                method_id=method_id,
                adapter=adapter,
                artifact=artifact,
                load_evidence=load_evidence,
                peak=peak,
                root=root,
            )
            # Keep method-specific output-only evidence on both the descriptor
            # and its timing receipt.  The stable writer strips only the
            # in-memory tensor, leaving all public metadata serializable.
            descriptor.update(timing_value)
            receipt_path = (
                output_root
                / cell.style
                / cell.condition
                / f"{method_id}.run.json"
            )
            receipt = write_prediction_receipt(receipt_path, descriptor)
            descriptor_without_tensor = dict(receipt)
            predictions[(cell.cell_id, method_id)] = descriptor_without_tensor
            timings[(cell.cell_id, method_id)] = descriptor_without_tensor
            method_cells.append(
                {
                    "cell_id": cell.cell_id,
                    "artifact": dict(artifact),
                    "receipt": {
                        "path": str(receipt_path.relative_to(root).as_posix()),
                        "bytes": int(receipt_path.stat().st_size),
                        "sha256": sha256_file(receipt_path),
                    },
                    "timed_elapsed_seconds": timed_elapsed,
                    "timing": dict(timing_value),
                }
            )
            _clear_ephemeral_a2(adapter)
            guards.append(
                _resource_guard(
                    args,
                    device=device,
                    stage=f"after_{cell.cell_id}_{method_id}",
                    started=started,
                )
            )
        return predictions, timings, {
            "method_id": method_id,
            "state_path": str(method.state_path),
            "state_sha256": sha256_file(method.state_path),
            "runtime_load": load_evidence,
            "cells": method_cells,
            "guards": guards,
            "peak_memory": _peak_memory(device),
        }
    finally:
        _clear_ephemeral_a2(adapter)
        del adapter
        del embeddings
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    missing = [
        name
        for name, value in (
            ("--panel", args.panel),
            ("--selection-plan", args.selection_plan),
            ("--registration", args.registration),
        )
        if value is None
    ]
    if missing:
        raise PredictionRunnerError("full public prediction run requires " + ", ".join(missing))
    panel_path = args.panel.expanduser().resolve()
    plan_path = args.selection_plan.expanduser().resolve()
    registration_path = args.registration.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise PredictionRunnerError(
            f"prediction output root must be a new path: {output_root}"
        )
    try:
        output_root.relative_to(root)
    except ValueError as exc:
        raise PredictionRunnerError(
            "prediction output root must be inside repository root"
        ) from exc
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = _git_commit(root)
    device = _choose_device(args.device)
    output_root.mkdir(parents=True)
    try:
        panel, cells, panel_record = load_trr5_panel(
            panel_path,
            repository_root=root,
        )
        plan, plan_record, selection = _validate_plan_binding(
            plan_path,
            panel=panel,
            root=root,
        )
        registration, methods, registration_meta = _validate_registration(
            registration_path,
            repository_root=root,
            panel_path=panel_path,
            plan_path=plan_path,
            panel_record=panel_record,
            plan_record=plan_record,
            current_commit=commit,
        )
        fit_root, causal_fit_root = _resolve_fit_roots(args)
        _validate_joint_state_roots(
            methods,
            fit_root=fit_root,
            causal_fit_root=causal_fit_root,
        )
        fit_summary = _fit_evidence_summary(
            fit_root,
            causal_fit_root=causal_fit_root,
            repository_root=root,
        )
        _resource_guard(
            args,
            device=device,
            stage="before_public_prediction_matrix",
            started=started,
        )
        panel_sha256 = str(panel_record["sha256"])
        selection_plan_sha256 = str(plan_record["sha256"])
        prediction_descriptors: dict[tuple[str, str], dict[str, Any]] = {}
        timing_descriptors: dict[tuple[str, str], dict[str, Any]] = {}
        method_runs: dict[str, Any] = {}
        for method_id in METHOD_IDS:
            method_predictions, method_timings, method_evidence = _run_method(
                method=methods[method_id],
                cells=cells,
                args=args,
                device=device,
                root=root,
                output_root=output_root,
                panel_sha256=panel_sha256,
                selection_plan_sha256=selection_plan_sha256,
                started=started,
            )
            prediction_descriptors.update(method_predictions)
            timing_descriptors.update(method_timings)
            method_runs[method_id] = method_evidence
        expected = {
            (cell_id, method_id)
            for cell_id in EXPECTED_CELL_IDS
            for method_id in METHOD_IDS
        }
        if set(prediction_descriptors) != expected or set(timing_descriptors) != expected:
            raise PredictionRunnerError("public prediction matrix is incomplete")
        prediction_manifest = {
            "schema": "token-reconstruction.trr0005-prediction-descriptor-manifest.v1",
            "task_id": TASK_ID,
            "status": "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH",
            "panel": dict(panel_record),
            "selection_plan": dict(plan_record),
            "registration": dict(registration_meta["file"]),
            "method_ids": list(METHOD_IDS),
            "cells": list(EXPECTED_CELL_IDS),
            "predictions": {
                f"{cell_id}::{method_id}": value
                for (cell_id, method_id), value in sorted(
                    prediction_descriptors.items()
                )
            },
            "truth_opened": False,
        }
        timing_manifest = {
            "schema": "token-reconstruction.trr0005-timing-descriptor-manifest.v1",
            "task_id": TASK_ID,
            "status": "PUBLIC_TIMINGS_COMPLETE_NO_TRUTH",
            "panel_sha256": panel_sha256,
            "selection_plan_sha256": selection_plan_sha256,
            "method_ids": list(METHOD_IDS),
            "cells": list(EXPECTED_CELL_IDS),
            "timings": {
                f"{cell_id}::{method_id}": value
                for (cell_id, method_id), value in sorted(
                    timing_descriptors.items()
                )
            },
            "truth_opened": False,
        }
        prediction_manifest_path = output_root / "predictions.json"
        timing_manifest_path = output_root / "timings.json"
        _write_create_only(prediction_manifest_path, prediction_manifest)
        _write_create_only(timing_manifest_path, timing_manifest)
        finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        evidence = {
            "schema": SCRIPT_SCHEMA,
            "task_id": TASK_ID,
            "status": "PUBLIC_PREDICTION_MATRIX_COMPLETE_NO_TRUTH",
            "started_utc": started_utc,
            "ended_utc": finished_utc,
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(device),
            "git_commit": commit,
            "panel": dict(panel_record),
            "selection_plan": dict(plan_record),
            "registration": dict(registration_meta),
            "selection": dict(selection),
            "method_ids": list(METHOD_IDS),
            "cells": list(EXPECTED_CELL_IDS),
            "prediction_count": len(prediction_descriptors),
            "timing_count": len(timing_descriptors),
            "prediction_manifest": {
                "path": str(prediction_manifest_path.relative_to(root).as_posix()),
                "bytes": int(prediction_manifest_path.stat().st_size),
                "sha256": sha256_file(prediction_manifest_path),
            },
            "timing_manifest": {
                "path": str(timing_manifest_path.relative_to(root).as_posix()),
                "bytes": int(timing_manifest_path.stat().st_size),
                "sha256": sha256_file(timing_manifest_path),
            },
            "methods": method_runs,
            "fit_development_summary": fit_summary,
            "runtime_contract": {
                "warmup_runs_per_record": 1,
                "measured_runs_per_record": 1,
                "warmup_output_exact_match_measured": True,
                "records_per_domain": RECORDS_PER_DOMAIN,
                "sequence_tokens": SEQUENCE_TOKENS,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
                "a2_candidate_output": "omitted_after_decision",
            },
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "future_activation_reads": False,
        }
        evidence_path = output_root / "run_evidence.json"
        _write_create_only(evidence_path, evidence)
        evidence["run_evidence"] = {
            "path": str(evidence_path.relative_to(root).as_posix()),
            "bytes": int(evidence_path.stat().st_size),
            "sha256": sha256_file(evidence_path),
        }
        # The create-only evidence file intentionally contains its own
        # pre-write fields; the returned mapping includes the final record.
        return evidence
    except Exception as exc:
        failure_path = output_root / "failure.json"
        if not failure_path.exists() and not failure_path.is_symlink():
            failure = {
                "schema": FAILURE_SCHEMA,
                "task_id": TASK_ID,
                "status": "FAILED_PRESERVED",
                "started_utc": started_utc,
                "ended_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "argv": list(sys.argv),
                "git_commit": commit,
                "panel_path": str(panel_path),
                "selection_plan_path": str(plan_path),
                "registration_path": str(registration_path),
                "truth_opened": False,
            }
            _write_create_only(failure_path, failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--selection-plan", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=DEFAULT_FIT_ROOT,
        help="root for the retained affine and trained-diagonal states/evidence",
    )
    parser.add_argument(
        "--causal-fit-root",
        type=Path,
        default=DEFAULT_CAUSAL_FIT_ROOT,
        help="root for the deployed causal states/evidence; defaults to the bounded q/k repair",
    )
    parser.add_argument(
        "--qualification-only",
        action="store_true",
        help="run the one-record archived Finance128 qualification and stop",
    )
    parser.add_argument(
        "--qualification-observation",
        type=Path,
        help="archived TRR4 Finance public_base activation safetensors",
    )
    parser.add_argument(
        "--qualification-a1-archive",
        type=Path,
        help="archived TRR4 Finance A1 prediction artifact",
    )
    parser.add_argument(
        "--qualification-a2-archive",
        type=Path,
        help="archived TRR4 Finance A2 prediction artifact",
    )
    parser.add_argument("--qualification-record-index", type=int, default=0)
    parser.add_argument("--qualification-record-id", type=str)
    parser.add_argument(
        "--embedding-table",
        type=Path,
        help="registered normalized public E; required only for qualification",
    )
    parser.add_argument(
        "--p0-checkpoint",
        type=Path,
        help="registered public P0 checkpoint blob; required only for qualification",
    )
    parser.add_argument(
        "--p0-config",
        type=Path,
        help="registered public P0 config blob; required only for qualification",
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        help="local public model snapshot; required only for frozen_a1_a2_k256",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="public-prefix reference implementation; required only for A2",
    )
    parser.add_argument(
        "--lens",
        type=Path,
        help="optional explicit retained A1 state; must equal the registered A1 state",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=DEFAULT_MINIMUM_FREE_GIB,
    )
    parser.add_argument(
        "--maximum-reserved-gib",
        type=float,
        default=DEFAULT_MAXIMUM_RESERVED_GIB,
    )
    parser.add_argument(
        "--maximum-rss-gib",
        type=float,
        default=DEFAULT_MAXIMUM_RSS_GIB,
    )
    parser.add_argument(
        "--minimum-host-available-gib",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = (
            _run_archived_qualification(args)
            if args.qualification_only
            else _run(args)
        )
    except (PredictionRunnerError, PredictionError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"TRR-0005 prediction error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
