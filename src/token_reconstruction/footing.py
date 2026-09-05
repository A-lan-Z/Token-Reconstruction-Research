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
import re
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch

from .dual_benchmark import validate_observations
from .freeze import FreezeError, require_truth_open_allowed


PANEL_SCHEMA = "token-reconstruction.trr0003-footing-panel.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0003-footing-prediction.v1"
REGISTRATION_SCHEMA = "token-reconstruction.trr0003-method-registration.v1"
TRUTH_BINDING_SCHEMA = "token-reconstruction.trr0003-truth-binding.v1"
TRUTH_SIDECAR_SCHEMA = "token-reconstruction.trr0003-panel-truth-sidecar.v1"
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise FootingError(f"{description} must be a lowercase SHA-256 digest")
    return value


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor shape, dtype, and exact contiguous CPU bytes."""

    if not isinstance(value, torch.Tensor):
        raise FootingError("tensor digest input is not a tensor")
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"shape": list(contiguous.shape), "dtype": str(contiguous.dtype)}).encode(
            "utf-8"
        )
    )
    digest.update(contiguous.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def external_file_record(path: Path) -> dict[str, Any]:
    """Record a private sidecar without requiring it to live in the repository."""

    if path.is_symlink() or not path.is_file():
        raise FootingError(f"private sidecar must be a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


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


def _truth_tensor_valid(value: torch.Tensor, *, cell: PanelCell) -> None:
    if tuple(value.shape) != cell.attention_mask.shape or value.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise FootingError(f"truth tensor geometry or dtype changed for {cell.cell_id}")
    if value[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise FootingError(f"truth BOS token changed for {cell.cell_id}")
    if value.lt(0).any().item() or value.ge(128256).any().item():
        raise FootingError(f"truth token range changed for {cell.cell_id}")


def _truth_digest_payload(
    cells: Sequence[PanelCell], truth: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    expected_cells = tuple(cell.cell_id for cell in cells)
    if tuple(truth) != expected_cells:
        raise FootingError("truth cells are incomplete or out of order")
    token_tensor_digests: dict[str, str] = {}
    token_row_digests: dict[str, list[str]] = {}
    shapes: dict[str, list[int]] = {}
    record_ids: dict[str, list[str]] = {}
    masks: dict[str, list[list[int]]] = {}
    positions: dict[str, list[list[int]]] = {}
    for cell in cells:
        value = truth.get(cell.cell_id)
        if value is None:
            raise FootingError(f"truth cell is absent: {cell.cell_id}")
        _truth_tensor_valid(value, cell=cell)
        # Token IDs are integer semantics; canonicalize their dtype before
        # committing bytes so int32/int64 sidecar serialization is equivalent.
        canonical = value.to(torch.int64)
        token_tensor_digests[cell.cell_id] = tensor_sha256(canonical)
        token_row_digests[cell.cell_id] = [tensor_sha256(row) for row in canonical]
        shapes[cell.cell_id] = list(value.shape)
        record_ids[cell.cell_id] = list(cell.record_ids)
        masks[cell.cell_id] = cell.attention_mask.to(torch.long).tolist()
        positions[cell.cell_id] = cell.position_ids.to(torch.long).tolist()
    return {
        "cell_order": list(expected_cells),
        "record_ids": record_ids,
        "attention_mask": masks,
        "position_ids": positions,
        "token_shapes": shapes,
        "token_tensor_sha256": token_tensor_digests,
        "token_row_digests": token_row_digests,
    }


def build_truth_binding(
    *,
    panel_sha256: str,
    cells: Sequence[PanelCell],
    truth: Mapping[str, torch.Tensor],
    preparation: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the private-sidecar commitment in a separate preparation role.

    The returned object contains only hashes, geometry, and public panel record
    identifiers.  It can be copied into the frozen registration/receipt while
    the sidecar and its token values remain private.
    """

    _valid_sha256(panel_sha256, description="truth binding panel hash")
    if not isinstance(preparation, Mapping):
        raise FootingError("truth preparation record is absent")
    preparation_record = dict(preparation)
    preparation_sha256 = _valid_sha256(
        preparation_record.get("sha256"), description="truth preparation hash"
    )
    preparation_path = preparation_record.get("path")
    if not isinstance(preparation_path, str) or not preparation_path:
        raise FootingError("truth preparation path is absent")
    if not isinstance(preparation_record.get("bytes"), int) or preparation_record["bytes"] < 0:
        raise FootingError("truth preparation byte count is invalid")
    if not isinstance(sidecar, Mapping):
        raise FootingError("truth sidecar record is absent")
    sidecar_record = dict(sidecar)
    sidecar_path = sidecar_record.get("path")
    if not isinstance(sidecar_path, str) or not sidecar_path:
        raise FootingError("truth sidecar path is absent")
    if not isinstance(sidecar_record.get("bytes"), int) or sidecar_record["bytes"] < 0:
        raise FootingError("truth sidecar byte count is invalid")
    _valid_sha256(sidecar_record.get("sha256"), description="truth sidecar hash")

    payload = _truth_digest_payload(cells, truth)
    cell_order = payload["cell_order"]
    record_ids = payload["record_ids"]
    masks = payload["attention_mask"]
    positions = payload["position_ids"]
    token_shapes = payload["token_shapes"]
    token_tensor_digests = payload["token_tensor_sha256"]
    token_row_digests = payload["token_row_digests"]
    return {
        "schema": TRUTH_BINDING_SCHEMA,
        "task_id": TASK_ID,
        "panel_sha256": panel_sha256,
        "preparation": preparation_record,
        "preparation_sha256": preparation_sha256,
        "sidecar": sidecar_record,
        "cell_order_json": _canonical_json(cell_order),
        "cell_order_sha256": _json_sha256(cell_order),
        "record_ids_sha256": _json_sha256(record_ids),
        "attention_mask_sha256": _json_sha256(masks),
        "position_ids_sha256": _json_sha256(positions),
        "token_shapes_json": _canonical_json(token_shapes),
        "token_tensor_sha256_json": _canonical_json(token_tensor_digests),
        "token_row_digests_json": _canonical_json(token_row_digests),
        "token_order_sha256": _json_sha256(
            {"cell_order": cell_order, "row_digests": token_row_digests}
        ),
        "required_tensor_keys_json": _canonical_json(
            [
                key
                for cell_id in cell_order
                for key in (
                    f"{cell_id}__token_ids",
                    f"{cell_id}__attention_mask",
                    f"{cell_id}__position_ids",
                )
            ]
        ),
        "paired_conditions": True,
    }


def truth_sidecar_metadata(binding: Mapping[str, Any]) -> dict[str, str]:
    """Return safetensors string metadata for a prepared sidecar."""

    if binding.get("schema") != TRUTH_BINDING_SCHEMA or binding.get("task_id") != TASK_ID:
        raise FootingError("truth binding identity changed")
    fields = (
        "panel_sha256",
        "preparation_sha256",
        "cell_order_json",
        "cell_order_sha256",
        "record_ids_sha256",
        "attention_mask_sha256",
        "position_ids_sha256",
        "token_shapes_json",
        "token_tensor_sha256_json",
        "token_row_digests_json",
        "token_order_sha256",
        "required_tensor_keys_json",
    )
    result = {
        "schema": TRUTH_SIDECAR_SCHEMA,
        "task_id": TASK_ID,
    }
    for field in fields:
        value = binding.get(field)
        if not isinstance(value, str) or not value:
            raise FootingError(f"truth binding field is absent: {field}")
        result[field] = value
    return result


def _validate_truth_binding_contract(
    binding: Mapping[str, Any], *, panel_sha256: str, cells: Sequence[PanelCell]
) -> None:
    if not isinstance(binding, Mapping):
        raise FootingError("truth binding is absent")
    if binding.get("schema") != TRUTH_BINDING_SCHEMA or binding.get("task_id") != TASK_ID:
        raise FootingError("truth binding identity changed")
    if binding.get("panel_sha256") != panel_sha256:
        raise FootingError("truth binding panel changed")
    _valid_sha256(binding.get("preparation_sha256"), description="truth preparation hash")
    preparation = binding.get("preparation")
    if not isinstance(preparation, Mapping):
        raise FootingError("truth preparation record is absent")
    if preparation.get("sha256") != binding.get("preparation_sha256"):
        raise FootingError("truth preparation binding changed")
    sidecar = binding.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise FootingError("truth sidecar record is absent")
    if not isinstance(sidecar.get("path"), str) or not sidecar["path"]:
        raise FootingError("truth sidecar path is absent")
    if not isinstance(sidecar.get("bytes"), int) or sidecar["bytes"] < 0:
        raise FootingError("truth sidecar byte count is invalid")
    _valid_sha256(sidecar.get("sha256"), description="truth sidecar hash")
    expected_cells = tuple(cell.cell_id for cell in cells)
    if binding.get("cell_order_json") != _canonical_json(list(expected_cells)):
        raise FootingError("truth cell order binding changed")
    if binding.get("cell_order_sha256") != _json_sha256(list(expected_cells)):
        raise FootingError("truth cell order digest changed")
    for field in (
        "record_ids_sha256",
        "attention_mask_sha256",
        "position_ids_sha256",
        "token_order_sha256",
    ):
        _valid_sha256(binding.get(field), description=f"truth {field}")
    # JSON encoded digest maps are checked for exact cell coverage and digest
    # shape here; their values are checked against opened tensors later.
    for field in ("token_shapes_json", "token_tensor_sha256_json", "token_row_digests_json"):
        value = binding.get(field)
        if not isinstance(value, str):
            raise FootingError(f"truth binding field is absent: {field}")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FootingError(f"truth binding field is invalid JSON: {field}") from exc
        # The canonical JSON encoder sorts object keys.  Cell order is
        # committed separately above; maps only need exact cell coverage here.
        if not isinstance(parsed, Mapping) or set(parsed) != set(expected_cells):
            raise FootingError(f"truth binding cell coverage changed: {field}")
    required = binding.get("required_tensor_keys_json")
    if not isinstance(required, str):
        raise FootingError("truth required tensor key binding is absent")
    try:
        required_keys = json.loads(required)
    except json.JSONDecodeError as exc:
        raise FootingError("truth required tensor key binding is invalid JSON") from exc
    expected_keys = [
        key
        for cell_id in expected_cells
        for key in (
            f"{cell_id}__token_ids",
            f"{cell_id}__attention_mask",
            f"{cell_id}__position_ids",
        )
    ]
    if required_keys != expected_keys:
        raise FootingError("truth required tensor keys changed")


def validate_truth_sidecar(
    path: Path,
    *,
    cells: Sequence[PanelCell],
    truth_binding: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Validate a prepared sidecar after the public freeze gate has passed."""

    panel_sha256 = str(truth_binding.get("panel_sha256", ""))
    _validate_truth_binding_contract(truth_binding, panel_sha256=panel_sha256, cells=cells)
    sidecar = truth_binding["sidecar"]
    if path.is_symlink() or not path.is_file():
        raise FootingError(f"truth sidecar is unavailable: {path}")
    if str(path.resolve()) != str(Path(str(sidecar["path"])).resolve()):
        raise FootingError("truth sidecar path binding changed")
    if int(path.stat().st_size) != int(sidecar["bytes"]) or sha256_file(path) != sidecar["sha256"]:
        raise FootingError("truth sidecar hash or size changed")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if not isinstance(metadata, Mapping):
                raise FootingError("truth sidecar metadata is malformed")
            expected_metadata = truth_sidecar_metadata(truth_binding)
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                raise FootingError("truth sidecar preparation or row digest binding changed")
            expected_keys = set(json.loads(truth_binding["required_tensor_keys_json"]))
            if set(handle.keys()) != expected_keys:
                raise FootingError("truth sidecar tensor set changed")
            result: dict[str, torch.Tensor] = {}
            for cell in cells:
                token_ids = handle.get_tensor(f"{cell.cell_id}__token_ids").to(torch.long)
                mask = handle.get_tensor(f"{cell.cell_id}__attention_mask").to(torch.long)
                positions = handle.get_tensor(f"{cell.cell_id}__position_ids").to(torch.long)
                _truth_tensor_valid(token_ids, cell=cell)
                if not torch.equal(mask, cell.attention_mask.to(torch.long)):
                    raise FootingError(f"truth mask pairing changed for {cell.cell_id}")
                if not torch.equal(positions, cell.position_ids.to(torch.long)):
                    raise FootingError(f"truth position/token order pairing changed for {cell.cell_id}")
                result[cell.cell_id] = token_ids
    except FootingError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise FootingError(f"truth sidecar is unreadable: {path}") from exc

    digest_payload = _truth_digest_payload(cells, result)
    if digest_payload["token_tensor_sha256"] != json.loads(truth_binding["token_tensor_sha256_json"]):
        raise FootingError("truth token digest changed")
    if digest_payload["token_row_digests"] != json.loads(truth_binding["token_row_digests_json"]):
        raise FootingError("truth row/token digest changed")
    if digest_payload["token_shapes"] != json.loads(truth_binding["token_shapes_json"]):
        raise FootingError("truth token geometry digest changed")
    if _json_sha256(
        {"cell_order": digest_payload["cell_order"], "row_digests": digest_payload["token_row_digests"]}
    ) != truth_binding["token_order_sha256"]:
        raise FootingError("truth token order digest changed")
    if _json_sha256(digest_payload["record_ids"]) != truth_binding["record_ids_sha256"]:
        raise FootingError("truth record ID digest changed")
    if _json_sha256(digest_payload["attention_mask"]) != truth_binding["attention_mask_sha256"]:
        raise FootingError("truth mask digest changed")
    if _json_sha256(digest_payload["position_ids"]) != truth_binding["position_ids_sha256"]:
        raise FootingError("truth position digest changed")
    for style in STYLE_ORDER:
        base = result[f"{style}__public_base"]
        shifted = result[f"{style}__public_lora_2601"]
        if not torch.equal(base, shifted):
            raise FootingError(f"paired truth token order changed for {style}")
    return result


def _validate_binding_shape(binding: Mapping[str, Any], *, description: str) -> None:
    if not isinstance(binding, Mapping):
        raise FootingError(f"{description} is absent")
    for group in ("panel", "method_state", "code"):
        values = binding.get(group) if group != "panel" else [binding.get(group)]
        if not isinstance(values, list) or not values or any(
            not isinstance(value, Mapping) for value in values
        ):
            raise FootingError(f"{description} has no valid {group} binding")
    if "code_commit" in binding and (
        not isinstance(binding["code_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", binding["code_commit"]) is None
    ):
        raise FootingError(f"{description} code commit binding is invalid")


def load_method_registry(
    path: Path,
    *,
    repository_root: Path,
    panel: Mapping[str, Any],
    panel_path: Path,
    require_truth_binding: bool = True,
) -> dict[str, Any]:
    """Load the task-local full method registration before common scoring.

    A registry is a small integration manifest.  It declares every comparator,
    Track A, and Track B method that must be present in the merged output and
    gives the source bundle layout used by :func:`merge` in the comparator
    runner.  It is intentionally distinct from the permanent dual-benchmark
    registry because these methods remain exploratory for TRR-0003.
    """

    if path.is_symlink() or not path.is_file():
        raise FootingError(f"method registry is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FootingError(f"method registry is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise FootingError("method registry root must be an object")
    if value.get("schema") != REGISTRATION_SCHEMA or value.get("task_id") != TASK_ID:
        raise FootingError("method registry identity changed")
    if value.get("status") not in (
        "EXPLORATORY_METHOD_BUNDLE_REGISTRATION",
        "MERGED_EXPLORATORY_METHOD_BUNDLE_REGISTRATION",
    ):
        raise FootingError("method registry status is not declared")
    root = repository_root.resolve()
    panel_record = value.get("panel")
    if not isinstance(panel_record, Mapping):
        raise FootingError("method registry panel binding is absent")
    actual_panel_record = file_record(panel_path, repository_root=root)
    if dict(panel_record) != actual_panel_record:
        raise FootingError("method registry panel binding changed")
    methods = value.get("methods")
    if not isinstance(methods, list) or not methods:
        raise FootingError("method registry has no methods")
    method_ids: list[str] = []
    normalized_methods: list[dict[str, Any]] = []
    bindings: dict[str, Mapping[str, Any]] = {}
    candidate_policies: dict[str, str] = {}
    for row in methods:
        if not isinstance(row, Mapping):
            raise FootingError("method registry method row is malformed")
        method_id = row.get("id")
        if (
            not isinstance(method_id, str)
            or not method_id
            or "/" in method_id
            or "\\" in method_id
            or method_id in (".", "..")
        ):
            raise FootingError("method registry method ID is unsafe")
        if method_id in method_ids:
            raise FootingError("method registry method IDs are duplicated")
        track = row.get("track")
        if track not in ("comparator", "track_a", "track_b"):
            raise FootingError(f"method registry track is invalid: {method_id}")
        binding = row.get("binding")
        _validate_binding_shape(binding, description=f"method registry binding for {method_id}")
        candidate_policy = row.get("candidate_policy", "optional")
        if candidate_policy not in ("required", "optional", "forbidden"):
            raise FootingError(f"method registry candidate policy is invalid: {method_id}")
        bundle = row.get("bundle")
        if not isinstance(bundle, Mapping):
            raise FootingError(f"method registry bundle is absent: {method_id}")
        layout = bundle.get("layout")
        if layout == "canonical":
            relative = _safe_relative(bundle.get("root"), description=f"{method_id} bundle")
            bundle_root = root / relative
            if bundle_root.is_symlink() or not bundle_root.is_dir():
                raise FootingError(f"method registry bundle root is unavailable: {relative}")
            normalized_bundle = {"layout": layout, "root": relative}
        elif layout == "archive_cells":
            cells = bundle.get("cells")
            if not isinstance(cells, Mapping) or set(cells) != set(expected_cell_ids()):
                raise FootingError(f"method registry archive cell set is incomplete: {method_id}")
            normalized_cells: dict[str, dict[str, Any]] = {}
            for cell_id in expected_cell_ids():
                descriptor = cells.get(cell_id)
                if not isinstance(descriptor, Mapping):
                    raise FootingError(f"method registry archive cell is malformed: {cell_id}")
                relative = _safe_relative(
                    descriptor.get("path"), description=f"{method_id} archive cell"
                )
                archive_path = root / relative
                if archive_path.is_symlink() or not archive_path.is_file():
                    raise FootingError(f"method registry archive is unavailable: {relative}")
                tensor_key = descriptor.get("tensor_key")
                if not isinstance(tensor_key, str) or not tensor_key.startswith("prediction."):
                    raise FootingError(f"method registry archive tensor key is invalid: {cell_id}")
                normalized_cells[cell_id] = {
                    "path": relative,
                    "tensor_key": tensor_key,
                }
                for field in ("input_tensor_sha256", "archive_sha256"):
                    if field in descriptor:
                        _valid_sha256(
                            descriptor[field],
                            description=f"method registry {field} for {cell_id}",
                        )
                        normalized_cells[cell_id][field] = descriptor[field]
                if "archive_sha256" in descriptor and sha256_file(archive_path) != descriptor["archive_sha256"]:
                    raise FootingError(f"method registry archive hash changed: {relative}")
                mask_padded = descriptor.get("mask_padded", False)
                if not isinstance(mask_padded, bool):
                    raise FootingError(f"method registry mask_padded flag is invalid: {cell_id}")
                normalized_cells[cell_id]["mask_padded"] = mask_padded
            normalized_bundle = {"layout": layout, "cells": normalized_cells}
        else:
            raise FootingError(f"method registry bundle layout is invalid: {method_id}")
        method_ids.append(method_id)
        bindings[method_id] = dict(binding)
        candidate_policies[method_id] = candidate_policy
        normalized_methods.append(
            {
                "id": method_id,
                "track": track,
                "candidate_policy": candidate_policy,
                "binding": dict(binding),
                "bundle": normalized_bundle,
            }
        )
    declared_ids = value.get("method_ids", method_ids)
    if declared_ids != method_ids:
        raise FootingError("method registry method order or completeness changed")
    truth_binding = value.get("truth_binding")
    if require_truth_binding:
        _validate_truth_binding_contract(
            truth_binding, panel_sha256=actual_panel_record["sha256"], cells=load_all_cells(panel, repository_root=root)
        )
    elif truth_binding is not None:
        _validate_truth_binding_contract(
            truth_binding, panel_sha256=actual_panel_record["sha256"], cells=load_all_cells(panel, repository_root=root)
        )
    return {
        "schema": REGISTRATION_SCHEMA,
        "task_id": TASK_ID,
        "status": value.get("status"),
        "path": path.resolve(),
        "method_ids": tuple(method_ids),
        "methods": tuple(normalized_methods),
        "bindings": bindings,
        "candidate_policies": candidate_policies,
        "truth_binding": dict(truth_binding) if isinstance(truth_binding, Mapping) else None,
        "panel": dict(panel_record),
    }


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
            if not method_id or "/" in method_id or "\\" in method_id or method_id in (".", ".."):
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
    return {
        "method_id": method_id,
        "binding": binding,
        "metadata": metadata,
        "tensor_fields": tuple(sorted(keys)),
    }


def expected_prediction_path(output_root: Path, *, cell: PanelCell, method_id: str) -> Path:
    if not method_id or "/" in method_id or "\\" in method_id or method_id in (".", ".."):
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
    candidate_policies: Mapping[str, str] | None = None,
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
            artifact = validate_prediction_artifact(
                path,
                cell=cell,
                panel_sha256=panel_sha,
                expected_method_id=method_id,
                expected_binding=binding,
                repository_root=repository_root,
            )
            if candidate_policies is not None:
                if set(candidate_policies) != set(method_ids):
                    raise FootingError("candidate policy registration is incomplete")
                policy = candidate_policies[method_id]
                if policy not in ("required", "optional", "forbidden"):
                    raise FootingError(f"candidate policy is invalid: {method_id}")
                has_candidates = "candidates" in artifact["tensor_fields"]
                if policy == "required" and not has_candidates:
                    raise FootingError(f"candidate tensors are required: {method_id}")
                if policy == "forbidden" and has_candidates:
                    raise FootingError(f"candidate tensors are forbidden: {method_id}")
            validated.append(artifact)
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
    candidate_policies: Mapping[str, str] | None = None,
    registration_path: Path | None = None,
    truth_binding: Mapping[str, Any] | None = None,
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
    if candidate_policies is not None and set(candidate_policies) != set(method_ids):
        raise FootingError("truth gate requires one candidate policy for every expected method")
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
    if registration_path is not None:
        registration_record = file_record(
            registration_path, repository_root=repository_root
        )
        if metadata.get("registration_sha256") != registration_record["sha256"]:
            raise FootingError("freeze receipt method registry binding changed")
    if registration_path is not None and truth_binding is None:
        raise FootingError("common truth gate requires a truth binding")
    if truth_binding is not None:
        cells = load_all_cells(panel, repository_root=repository_root)
        _validate_truth_binding_contract(
            truth_binding,
            panel_sha256=sha256_file(panel_path),
            cells=cells,
        )
        if metadata.get("truth_binding") != dict(truth_binding):
            raise FootingError("freeze receipt truth binding changed")
        sidecar = truth_binding["sidecar"]
        if str(truth_path.resolve()) != str(Path(str(sidecar["path"])).resolve()):
            raise FootingError("truth sidecar path binding changed")
        if truth_path.is_symlink() or not truth_path.is_file():
            raise FootingError("truth sidecar is unavailable")
        if int(truth_path.stat().st_size) != int(sidecar["bytes"]):
            raise FootingError("truth sidecar size binding changed")
    validate_complete_prediction_set(
        output_root,
        panel=panel,
        panel_path=panel_path,
        repository_root=repository_root,
        method_ids=method_ids,
        expected_bindings=expected_bindings,
        candidate_policies=candidate_policies,
    )
    return payload
