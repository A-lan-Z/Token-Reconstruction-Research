"""Small, fail-closed interfaces for the TRR-0003 shared pilot.

The footing panel is a public development control.  This module deliberately
contains no truth loader.  A reconstruction process can load a panel cell and
its public activation asset, validate its own prediction artifact, and prove
that a complete set of method cells is ready for freezing before any evaluator
opens a private sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch

from .dual_benchmark import validate_observations
from .freeze import FreezeError, require_truth_open_allowed


PANEL_SCHEMA = "token-reconstruction.trr0003-footing-panel.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0003-footing-prediction.v1"
TASK_ID = "TRR-0003"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
CUT_DEPTH = 4
HIDDEN_SIZE = 2048
BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
STYLE_ORDER = ("pile", "finance")
CONDITION_ORDER = ("public_base", "public_lora_2601")
PANEL_RECORDS_PER_STYLE = 8


class FootingError(RuntimeError):
    """Raised when the shared panel or a method artifact is not trustworthy."""


@dataclass(frozen=True)
class PanelCell:
    """One style/condition cell loaded from the sanitized panel."""

    cell_id: str
    style: str
    condition: str
    record_ids: tuple[str, ...]
    activations: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    observation_path: Path

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.activations.shape)

    @property
    def records(self) -> int:
        return int(self.activations.shape[0])

    @property
    def sequence_tokens(self) -> int:
        return int(self.activations.shape[1])


def _safe_relative(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise FootingError(f"{description} path is absent")
    if "\\" in value:
        raise FootingError(f"{description} path must use POSIX separators")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise FootingError(f"{description} path is unsafe: {value}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, repository_root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FootingError(f"asset must be a regular file: {path}")
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise FootingError(f"asset is outside repository root: {path}") from exc
    return {
        "path": _safe_relative(relative, description="asset"),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _asset_path(asset: Mapping[str, Any], *, repository_root: Path, description: str) -> Path:
    relative = _safe_relative(asset.get("path"), description=description)
    root = repository_root.resolve()
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise FootingError(f"{description} is unavailable: {relative}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FootingError(f"{description} escaped repository root: {relative}") from exc
    expected_bytes = asset.get("bytes")
    expected_sha = asset.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise FootingError(f"{description} byte count is invalid")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise FootingError(f"{description} hash is invalid")
    if int(path.stat().st_size) != expected_bytes or sha256_file(path) != expected_sha:
        raise FootingError(f"{description} hash or size changed: {relative}")
    return path


def _reject_private_keys(value: Any, *, path: str = "panel") -> None:
    """Reject fields that could carry source tokens or evaluator truth."""

    forbidden = ("truth", "oracle", "token_ids", "input_ids", "labels")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FootingError(f"{path} keys must be strings")
            lowered = key.casefold().replace("-", "_")
            if any(fragment in lowered for fragment in forbidden):
                raise FootingError(f"{path}.{key} is private evaluator/source state")
            _reject_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private_keys(child, path=f"{path}[{index}]")


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FootingError(f"panel must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FootingError(f"panel is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FootingError("panel root must be an object")
    _reject_private_keys(value)
    return value


def _mask_and_positions(
    cell: Mapping[str, Any], *, records: int, sequence_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        mask = torch.tensor(cell["attention_mask"], dtype=torch.long)
        positions = torch.tensor(cell["position_ids"], dtype=torch.long)
    except (KeyError, TypeError, ValueError) as exc:
        raise FootingError("panel cell masks or positions are malformed") from exc
    expected = (records, sequence_tokens)
    if tuple(mask.shape) != expected or tuple(positions.shape) != expected:
        raise FootingError("panel cell mask/position geometry changed")
    # A zero activation tensor is enough to exercise the common geometry
    # validator without opening any private source or truth asset.
    try:
        validate_observations(
            torch.zeros((records, sequence_tokens, HIDDEN_SIZE), dtype=torch.float32),
            mask,
            positions,
        )
    except Exception as exc:
        raise FootingError(f"panel cell geometry is invalid: {exc}") from exc
    return mask, positions


def _cell_id(style: str, condition: str) -> str:
    return f"{style}__{condition}"


def expected_cell_ids() -> tuple[str, ...]:
    return tuple(_cell_id(style, condition) for style in STYLE_ORDER for condition in CONDITION_ORDER)


def load_panel(path: Path, *, repository_root: Path | None = None) -> dict[str, Any]:
    """Load and validate the sanitized panel, without loading evaluator truth."""

    panel = _load_json(path)
    if panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != TASK_ID:
        raise FootingError("panel identity changed")
    if panel.get("status") != "RETROSPECTIVE_DEVELOPMENT_PANEL":
        raise FootingError("panel is not the declared retrospective development panel")
    if panel.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise FootingError("panel model identity changed")
    if panel.get("cut_depth") != CUT_DEPTH or panel.get("hidden_size") != HIDDEN_SIZE:
        raise FootingError("panel cut or hidden-size geometry changed")
    if panel.get("source_material_included") is not False:
        raise FootingError("panel includes private source or truth inputs")

    styles = panel.get("styles")
    if not isinstance(styles, list) or any(not isinstance(row, Mapping) for row in styles):
        raise FootingError("panel style rows are malformed")
    if [row.get("id") for row in styles] != list(STYLE_ORDER):
        raise FootingError("panel style order changed")
    style_specs: dict[str, Mapping[str, Any]] = {}
    for row in styles:
        if not isinstance(row, Mapping):
            raise FootingError("panel style row is malformed")
        style = row.get("id")
        if style not in STYLE_ORDER or style in style_specs:
            raise FootingError("panel style IDs are invalid")
        if row.get("records") != 8 or row.get("hidden_size") != HIDDEN_SIZE:
            raise FootingError("panel style record or hidden-size geometry changed")
        if row.get("sequence_tokens") not in (40, 128):
            raise FootingError("panel style sequence geometry changed")
        style_specs[str(style)] = row

    conditions = panel.get("conditions")
    if not isinstance(conditions, list) or any(not isinstance(row, Mapping) for row in conditions):
        raise FootingError("panel condition rows are malformed")
    if [row.get("id") for row in conditions] != list(CONDITION_ORDER):
        raise FootingError("panel condition order changed")
    condition_specs: dict[str, Mapping[str, Any]] = {}
    for row in conditions:
        if not isinstance(row, Mapping):
            raise FootingError("panel condition row is malformed")
        condition = row.get("id")
        if condition not in CONDITION_ORDER or condition in condition_specs:
            raise FootingError("panel condition IDs are invalid")
        condition_specs[str(condition)] = row

    cells = panel.get("cells")
    if not isinstance(cells, list) or any(not isinstance(row, Mapping) for row in cells):
        raise FootingError("panel cells are malformed")
    if [row.get("id") for row in cells] != list(expected_cell_ids()):
        raise FootingError("panel cell order or completeness changed")
    seen_by_style: dict[str, set[str]] = {style: set() for style in STYLE_ORDER}
    ordered_by_style: dict[str, tuple[str, ...]] = {}
    masks_by_style: dict[str, torch.Tensor] = {}
    positions_by_style: dict[str, torch.Tensor] = {}
    for row in cells:
        if not isinstance(row, Mapping):
            raise FootingError("panel cell is malformed")
        cell_id = row.get("id")
        style = row.get("style")
        condition = row.get("condition")
        if cell_id != _cell_id(style, condition) or style not in style_specs or condition not in condition_specs:
            raise FootingError("panel cell identity changed")
        record_rows = row.get("records")
        if not isinstance(record_rows, list) or len(record_rows) != 8:
            raise FootingError("panel cell record count changed")
        ids: list[str] = []
        for record in record_rows:
            if not isinstance(record, Mapping) or set(record) - {
                "record_id",
                "public_record_sha256",
                "tokenized_record_sha256",
                "raw_index",
                "valid_tokens",
            }:
                raise FootingError("panel cell contains unapproved record fields")
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or not record_id or record_id in ids:
                raise FootingError("panel record IDs are invalid or duplicated")
            ids.append(record_id)
        spec = style_specs[str(style)]
        sequence_tokens = int(spec["sequence_tokens"])
        mask, positions = _mask_and_positions(row, records=8, sequence_tokens=sequence_tokens)
        if str(style) not in ordered_by_style:
            ordered_by_style[str(style)] = tuple(ids)
            masks_by_style[str(style)] = mask
            positions_by_style[str(style)] = positions
            seen_by_style[str(style)].update(ids)
        else:
            if tuple(ids) != ordered_by_style[str(style)]:
                raise FootingError("paired condition record order changed")
            if not torch.equal(mask, masks_by_style[str(style)]) or not torch.equal(
                positions, positions_by_style[str(style)]
            ):
                raise FootingError("paired condition masks or positions changed")
        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise FootingError("panel observation asset is absent")
        row_indices = observation.get("row_indices")
        if row_indices != list(range(PANEL_RECORDS_PER_STYLE)):
            raise FootingError("panel observation row selection changed")
        if repository_root is not None:
            _asset_path(observation, repository_root=repository_root, description="panel observation")
        if condition == "public_lora_2601" and row.get("shift_role") != "single_public_shift_diagnostic":
            raise FootingError("shifted condition role changed")
    if sum(len(ids) for ids in seen_by_style.values()) != 16:
        raise FootingError("panel must contain 16 distinct records per condition")
    return panel


def load_cell(
    panel: Mapping[str, Any],
    *,
    style: str,
    condition: str,
    repository_root: Path,
) -> PanelCell:
    """Load one validated public observation cell from the shared panel."""

    if style not in STYLE_ORDER or condition not in CONDITION_ORDER:
        raise FootingError("unknown panel cell")
    cell_id = _cell_id(style, condition)
    cells = {row.get("id"): row for row in panel.get("cells", []) if isinstance(row, Mapping)}
    row = cells.get(cell_id)
    if not isinstance(row, Mapping):
        raise FootingError(f"panel cell is absent: {cell_id}")
    spec = next((value for value in panel["styles"] if value.get("id") == style), None)
    if not isinstance(spec, Mapping):
        raise FootingError("panel style specification is absent")
    records = row["records"]
    record_ids = tuple(str(record["record_id"]) for record in records)
    sequence_tokens = int(spec["sequence_tokens"])
    mask, positions = _mask_and_positions(row, records=8, sequence_tokens=sequence_tokens)
    observation_path = _asset_path(
        row["observation"], repository_root=repository_root, description="panel observation"
    )
    row_indices = row["observation"].get("row_indices")
    if row_indices != list(range(PANEL_RECORDS_PER_STYLE)):
        raise FootingError("panel observation row selection changed")
    try:
        with safe_open(observation_path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations"}:
                raise FootingError("panel observation tensor fields changed")
            full_activations = handle.get_tensor("activations").contiguous()
    except FootingError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise FootingError(f"panel observation is unreadable: {observation_path}") from exc
    if full_activations.ndim != 3 or tuple(full_activations.shape[1:]) != (sequence_tokens, HIDDEN_SIZE):
        raise FootingError(f"panel observation geometry changed for {cell_id}")
    if full_activations.shape[0] <= max(row_indices):
        raise FootingError(f"panel observation row selection is unavailable for {cell_id}")
    activations = full_activations.index_select(
        0, torch.tensor(row_indices, dtype=torch.long)
    ).contiguous()
    expected_shape = (8, sequence_tokens, HIDDEN_SIZE)
    if tuple(activations.shape) != expected_shape:
        raise FootingError(f"panel observation geometry changed for {cell_id}")
    if not activations.dtype.is_floating_point or not torch.isfinite(activations).all().item():
        raise FootingError(f"panel observation values are invalid for {cell_id}")
    return PanelCell(
        cell_id=cell_id,
        style=style,
        condition=condition,
        record_ids=record_ids,
        activations=activations,
        attention_mask=mask,
        position_ids=positions,
        observation_path=observation_path,
    )


def load_all_cells(panel: Mapping[str, Any], *, repository_root: Path) -> tuple[PanelCell, ...]:
    return tuple(
        load_cell(panel, style=style, condition=condition, repository_root=repository_root)
        for style in STYLE_ORDER
        for condition in CONDITION_ORDER
    )


def make_binding(
    *,
    panel_path: Path,
    repository_root: Path,
    method_state_paths: Sequence[Path],
    code_paths: Sequence[Path],
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Build the binding that every prediction artifact must carry."""

    if not method_state_paths or not code_paths:
        raise FootingError("method state and executable code bindings are required")
    binding: dict[str, Any] = {
        "panel": file_record(panel_path, repository_root=repository_root),
        "method_state": [
            file_record(path, repository_root=repository_root) for path in method_state_paths
        ],
        "code": [file_record(path, repository_root=repository_root) for path in code_paths],
    }
    if code_commit is not None:
        if not isinstance(code_commit, str) or len(code_commit) != 40:
            raise FootingError("code commit binding must be a full commit")
        binding["code_commit"] = code_commit
    return binding


def _validate_binding_files(binding: Mapping[str, Any], *, repository_root: Path) -> None:
    for group in ("panel", "method_state", "code"):
        if group not in binding:
            raise FootingError(f"prediction binding omits {group}")
        values = binding[group] if group != "panel" else [binding[group]]
        if not isinstance(values, list) or not values:
            raise FootingError(f"prediction binding has no {group} assets")
        for asset in values:
            if not isinstance(asset, Mapping):
                raise FootingError(f"prediction {group} binding is malformed")
            path = _asset_path(asset, repository_root=repository_root, description=f"bound {group} asset")
            current = file_record(path, repository_root=repository_root)
            if any(asset.get(key) != current[key] for key in ("path", "bytes", "sha256")):
                raise FootingError(f"prediction {group} asset changed: {path}")
    if "code_commit" in binding and (
        not isinstance(binding["code_commit"], str) or len(binding["code_commit"]) != 40
    ):
        raise FootingError("prediction code commit binding is invalid")


def validate_binding(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
) -> None:
    """Compare recorded binding with current bytes before truth is opened."""

    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise FootingError("prediction input/state/code binding changed")
    if repository_root is not None:
        _validate_binding_files(actual, repository_root=repository_root)
    else:
        for group in ("panel", "method_state", "code"):
            if group not in actual:
                raise FootingError(f"prediction binding omits {group}")


def _metadata(handle: Any) -> dict[str, str]:
    metadata = handle.metadata() or {}
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise FootingError("prediction metadata must be string-valued JSON fields")
    required = {
        "schema",
        "task_id",
        "panel_sha256",
        "cell_id",
        "style",
        "condition",
        "method_id",
        "geometry_json",
        "binding_json",
    }
    if not required.issubset(metadata):
        raise FootingError("prediction metadata is incomplete")
    return metadata


def _json_object(value: str, *, description: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise FootingError(f"{description} is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise FootingError(f"{description} must be an object")
    return decoded


def validate_prediction_artifact(
    path: Path,
    *,
    cell: PanelCell,
    panel_sha256: str,
    expected_method_id: str | None = None,
    expected_binding: Mapping[str, Any] | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Validate one prediction artifact without reading any truth sidecar."""

    if path.is_symlink() or not path.is_file():
        raise FootingError(f"prediction artifact is unavailable: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = _metadata(handle)
            if metadata["schema"] != PREDICTION_SCHEMA or metadata["task_id"] != TASK_ID:
                raise FootingError("prediction artifact identity changed")
            if metadata["panel_sha256"] != panel_sha256:
                raise FootingError("prediction panel binding changed")
            if metadata["cell_id"] != cell.cell_id or metadata["style"] != cell.style or metadata["condition"] != cell.condition:
                raise FootingError("prediction cell binding changed")
            method_id = metadata["method_id"]
            if not method_id or "/" in method_id or method_id in (".", ".."):
                raise FootingError("prediction method ID is invalid")
            if expected_method_id is not None and method_id != expected_method_id:
                raise FootingError("prediction method ID changed")
            geometry = _json_object(metadata["geometry_json"], description="prediction geometry")
            if geometry != {
                "records": cell.records,
                "sequence_tokens": cell.sequence_tokens,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
            }:
                raise FootingError("prediction geometry binding changed")
            binding = _json_object(metadata["binding_json"], description="prediction binding")
            if expected_binding is not None:
                validate_binding(binding, expected_binding, repository_root=repository_root)
            keys = set(handle.keys())
            allowed = {"predictions", "candidates", "candidate_scores", "selection_scores"}
            if not keys.issubset(allowed) or "predictions" not in keys:
                raise FootingError("prediction tensor fields are incomplete or unexpected")
            predictions = handle.get_tensor("predictions")
            if tuple(predictions.shape) != cell.attention_mask.shape or predictions.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise FootingError("prediction tensor geometry or dtype changed")
            if predictions[:, 0].ne(BOS_TOKEN_ID).any().item():
                raise FootingError("prediction BOS token changed")
            active = cell.attention_mask.to(torch.bool)
            if predictions[active].lt(0).any().item() or predictions[active].ge(128256).any().item():
                raise FootingError("active prediction contains an invalid or out-of-vocabulary token")
            if predictions[~active].ne(INVALID_TOKEN_ID).any().item():
                raise FootingError("padded prediction is not marked invalid")
            for name in ("candidates", "candidate_scores", "selection_scores"):
                if name not in keys:
                    continue
                if name != "candidates" and "candidates" not in keys:
                    raise FootingError(f"{name} is present without candidates")
                tensor = handle.get_tensor(name)
                if name == "candidates":
                    if tensor.ndim != 3 or tuple(tensor.shape[:2]) != cell.attention_mask.shape or tensor.shape[2] <= 0:
                        raise FootingError("candidate geometry changed")
                    if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                        raise FootingError("candidate IDs must be integer")
                    if tensor[active].lt(0).any().item() or tensor[active].ge(128256).any().item():
                        raise FootingError("active candidate contains an invalid or out-of-vocabulary token")
                else:
                    if tensor.ndim != 3 or tuple(tensor.shape) != tuple(handle.get_tensor("candidates").shape):
                        raise FootingError("candidate score geometry changed")
                    if tensor.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                        raise FootingError("candidate scores must be floating point")
                    if not torch.isfinite(tensor[active]).all().item():
                        raise FootingError("active candidate score is non-finite")
    except FootingError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise FootingError(f"prediction artifact is unreadable: {path}") from exc
    return {"method_id": method_id, "binding": binding, "metadata": metadata}


def expected_prediction_path(output_root: Path, *, cell: PanelCell, method_id: str) -> Path:
    if not method_id or "/" in method_id or method_id in (".", ".."):
        raise FootingError("method ID is not a safe path component")
    return output_root / cell.style / cell.condition / f"{method_id}.safetensors"


def validate_complete_prediction_set(
    output_root: Path,
    *,
    panel: Mapping[str, Any],
    panel_path: Path,
    repository_root: Path,
    method_ids: Sequence[str],
    expected_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Check completeness and integrity for every method/style/condition cell."""

    if len(set(method_ids)) != len(method_ids) or not method_ids:
        raise FootingError("method IDs are empty or duplicated")
    if expected_bindings is None or set(expected_bindings) != set(method_ids):
        raise FootingError("expected input/state/code bindings are incomplete")
    if output_root.is_symlink() or not output_root.is_dir():
        raise FootingError("prediction output root is not a regular directory")
    panel_sha = sha256_file(panel_path)
    cells = load_all_cells(panel, repository_root=repository_root)
    expected_paths: set[Path] = set()
    validated: list[dict[str, Any]] = []
    for cell in cells:
        for method_id in method_ids:
            path = expected_prediction_path(output_root, cell=cell, method_id=method_id)
            expected_paths.add(path.resolve())
            binding = expected_bindings[method_id]
            validated.append(
                validate_prediction_artifact(
                    path,
                    cell=cell,
                    panel_sha256=panel_sha,
                    expected_method_id=method_id,
                    expected_binding=binding,
                    repository_root=repository_root,
                )
            )
    actual_paths = {
        path.resolve()
        for path in output_root.rglob("*.safetensors")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise FootingError(f"prediction artifact set is incomplete: missing={missing!r} extra={extra!r}")
    return validated


def validate_before_truth(
    *,
    receipt_path: Path,
    repository_root: Path,
    truth_path: Path,
    output_root: Path,
    panel: Mapping[str, Any],
    panel_path: Path,
    method_ids: Sequence[str],
    expected_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run every public completeness/binding check before truth may be read."""

    try:
        payload = require_truth_open_allowed(
            receipt_path=receipt_path,
            repository_root=repository_root,
            truth_path=truth_path,
        )
    except FreezeError as exc:
        raise FootingError(f"freeze receipt rejected: {exc}") from exc
    if expected_bindings is None or set(expected_bindings) != set(method_ids):
        raise FootingError("truth gate requires one binding for every expected method")
    try:
        output_relative = output_root.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise FootingError("prediction output root is outside repository root") from exc
    if payload.get("frozen_root") != output_relative:
        raise FootingError("freeze receipt does not bind the requested output root")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise FootingError("freeze receipt metadata is absent")
    if metadata.get("panel_sha256") != sha256_file(panel_path):
        raise FootingError("freeze receipt does not bind the requested panel")
    if metadata.get("method_ids") != list(method_ids):
        raise FootingError("freeze receipt method registration changed")
    validate_complete_prediction_set(
        output_root,
        panel=panel,
        panel_path=panel_path,
        repository_root=repository_root,
        method_ids=method_ids,
        expected_bindings=expected_bindings,
    )
    return payload
