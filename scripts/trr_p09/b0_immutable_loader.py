#!/usr/bin/env python3
"""Lazy adapter for the published TRR-P09 B0 prefix bank.

The published B0 bank is one monolithic safetensors file produced by the
TRR-0007 public capture.  The P09 capture contract publishes B1 as immutable
64-row shards, so :class:`CombinedStreamedBankLoader` cannot consume B0
directly: it deliberately rejects payloads outside a bank root and requires
one shard sidecar per payload.  This module keeps that distinction explicit.

``B0ImmutableLoader`` binds the already-published B0 payload and metadata,
checks only safetensors headers during construction, and reads requested
contiguous row windows lazily.  It never copies or rewrites the B0 payload.
``CombinedB0StreamedBankLoader`` exposes the same ``get_records`` and
``iter_batches`` interface as the production streamed loader while composing
B0 rows ``[0, 1200)`` with a captured P09 B1 manifest beginning at row 1200.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from scripts.trr_p09.prepare_streamed_bank import (
    BankContractError,
    BankGeometry,
    StreamBatch,
    StreamedBankLoader,
    _expected_active_prefix_positions,
    file_record,
    verify_file_record,
)


TASK_ID = "TRR-P09"
B0_BINDING_SCHEMA = "token-reconstruction.trr-p09-b0-immutable-loader-binding.v1"
_PAYLOAD_KEYS = ("activations", "attention_mask", "position_ids", "token_ids")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BankContractError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise BankContractError(f"{label} must be a JSON object: {path}")
    return value


def _stat_snapshot(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise BankContractError(f"{label} disappeared or became a symlink: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
    }


def _record_id_digest(record_ids: Sequence[str]) -> str:
    """Match the published TRR-0007 record-ID digest convention."""

    return hashlib.sha256(("\n".join(record_ids) + "\n").encode("utf-8")).hexdigest()


def _dtype_name(value: Any) -> str:
    return str(value).upper().replace("TORCH.", "")


def _validate_header(
    payload_path: Path,
    *,
    geometry: BankGeometry,
    row_count: int,
    allowed_extra_keys: set[str],
) -> dict[str, dict[str, Any]]:
    expected_shapes = {
        "activations": [row_count, geometry.sequence_tokens, geometry.hidden_size],
        "attention_mask": [row_count, geometry.sequence_tokens],
        "position_ids": [row_count, geometry.sequence_tokens],
        "token_ids": [row_count, geometry.sequence_tokens],
    }
    allowed_dtypes = {
        "activations": {"BF16"},
        "attention_mask": {"BOOL", "U8"},
        "position_ids": {"I32", "I64"},
        "token_ids": {"I32", "I64"},
    }
    try:
        with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            missing = set(_PAYLOAD_KEYS) - keys
            unexpected = keys - set(_PAYLOAD_KEYS) - set(allowed_extra_keys)
            if missing:
                raise BankContractError(f"B0 payload is missing tensor keys: {sorted(missing)}")
            if unexpected:
                raise BankContractError(f"B0 payload has unexpected tensor keys: {sorted(unexpected)}")
            headers: dict[str, dict[str, Any]] = {}
            for key in _PAYLOAD_KEYS:
                view = handle.get_slice(key)
                shape = [int(value) for value in view.get_shape()]
                dtype = _dtype_name(view.get_dtype())
                if shape != expected_shapes[key]:
                    raise BankContractError(
                        f"B0 payload {key} shape changed: {shape} vs {expected_shapes[key]}"
                    )
                if dtype not in allowed_dtypes[key]:
                    raise BankContractError(f"B0 payload {key} dtype changed: {dtype}")
                headers[key] = {"shape": shape, "dtype": dtype}
    except BankContractError:
        raise
    except Exception as exc:
        raise BankContractError(f"cannot inspect B0 payload header: {payload_path}") from exc
    return headers


def _validate_window(tensors: Mapping[str, torch.Tensor], *, geometry: BankGeometry) -> None:
    rows = int(torch.as_tensor(tensors["activations"]).shape[0])
    geometry.validate()
    expected = (rows, geometry.sequence_tokens)
    if tuple(tensors["activations"].shape) != (rows, geometry.sequence_tokens, geometry.hidden_size):
        raise BankContractError("B0 activation window shape changed")
    for key in ("attention_mask", "position_ids", "token_ids"):
        if tuple(tensors[key].shape) != expected:
            raise BankContractError(f"B0 {key} window shape changed")
    if tensors["activations"].dtype != torch.bfloat16:
        raise BankContractError("B0 activations are not BF16")
    if tensors["attention_mask"].dtype not in (torch.bool, torch.uint8):
        raise BankContractError("B0 attention_mask dtype changed")
    if tensors["position_ids"].dtype not in (torch.int32, torch.int64):
        raise BankContractError("B0 position_ids dtype changed")
    if tensors["token_ids"].dtype not in (torch.int32, torch.int64):
        raise BankContractError("B0 token_ids dtype changed")
    mask = tensors["attention_mask"].to(dtype=torch.bool)
    if not bool(mask[:, 0].all().item()):
        raise BankContractError("B0 row has no active BOS/first position")
    positions = tensors["position_ids"].to(dtype=torch.long)
    if not torch.equal(positions, _expected_active_prefix_positions(mask)):
        raise BankContractError("B0 position_ids do not use active-prefix/zero-padding convention")


class B0ImmutableLoader:
    """Read metadata-bound B0 rows without copying the published H payload."""

    interface = "trr-p09.b0-immutable-loader.v1"

    def __init__(self, binding_path: Path, *, device: str = "cpu") -> None:
        self.binding_path = Path(binding_path).expanduser().resolve()
        self.binding = _load_object(self.binding_path, label="B0 binding")
        if self.binding.get("schema") != B0_BINDING_SCHEMA:
            raise BankContractError("B0 binding schema differs")
        if self.binding.get("task_id") != TASK_ID:
            raise BankContractError("B0 binding task differs")
        self._artifacts: dict[str, Path] = {}
        self._statics: list[dict[str, Any]] = []
        artifacts = self.binding.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise BankContractError("B0 binding artifacts are absent")
        for role in ("payload", "records", "published_manifest", "position_contract", "corpus_plan"):
            descriptor = artifacts.get(role)
            if not isinstance(descriptor, Mapping):
                raise BankContractError(f"B0 binding artifact is absent: {role}")
            actual = verify_file_record(descriptor, label=f"B0 {role}")
            path = Path(actual["path"]).resolve()
            self._artifacts[role] = path
            self._statics.append(_stat_snapshot(path, label=f"B0 {role}"))

        bank = self.binding.get("bank")
        if not isinstance(bank, Mapping):
            raise BankContractError("B0 binding bank descriptor is absent")
        self.expanded_row_origin = int(bank.get("expanded_row_origin", -1))
        self.record_count = int(bank.get("record_count", -1))
        if self.expanded_row_origin != 0 or self.record_count <= 0:
            raise BankContractError("B0 must be a nonempty row-zero prefix")
        self.global_row_stop = self.record_count
        geometry_value = bank.get("geometry")
        if not isinstance(geometry_value, Mapping):
            raise BankContractError("B0 geometry is absent")
        self.geometry = BankGeometry(
            sequence_tokens=int(geometry_value.get("sequence_tokens", -1)),
            hidden_size=int(geometry_value.get("hidden_size", -1)),
            batch_records=int(geometry_value.get("loader_batch_records", -1)),
            shard_records=int(geometry_value.get("shard_records", -1)),
            hidden_dtype=str(geometry_value.get("hidden_dtype", "")),
        )
        self.geometry.validate()
        if self.record_count % self.geometry.batch_records:
            raise BankContractError("B0 record count is not loader-batch aligned")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BankContractError("requested B0 loader CUDA device is unavailable")
        self.device = torch.device(device)

        published = _load_object(self._artifacts["published_manifest"], label="published B0 manifest")
        records_doc = _load_object(self._artifacts["records"], label="published B0 records")
        position_contract = _load_object(self._artifacts["position_contract"], label="B0 position contract")
        self._validate_metadata(bank, published, records_doc, position_contract)
        self.records = [dict(row) for row in records_doc["records"]]
        allowed_extra = set(str(value) for value in bank.get("allowed_extra_tensor_keys", []))
        self.tensor_headers = _validate_header(
            self._artifacts["payload"],
            geometry=self.geometry,
            row_count=self.record_count,
            allowed_extra_keys=allowed_extra,
        )

    def _validate_metadata(
        self,
        bank: Mapping[str, Any],
        published: Mapping[str, Any],
        records_doc: Mapping[str, Any],
        position_contract: Mapping[str, Any],
    ) -> None:
        rows = records_doc.get("records")
        if not isinstance(rows, list) or len(rows) != self.record_count:
            raise BankContractError("published B0 record metadata count differs")
        ids: list[str] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise BankContractError(f"published B0 row {index} is malformed")
            if int(row.get("slot", -1)) != index:
                raise BankContractError(f"published B0 row {index} slot differs")
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id or record_id in seen:
                raise BankContractError(f"published B0 row {index} record_id is invalid or duplicated")
            seen.add(record_id)
            ids.append(record_id)
        expected_digest = str(self.binding.get("record_ids", {}).get("sha256", ""))
        actual_digest = _record_id_digest(ids)
        if actual_digest != expected_digest:
            raise BankContractError("published B0 record-ID digest differs")
        if published.get("record_ids_sha256") != expected_digest:
            raise BankContractError("B0 binding disagrees with published record-ID digest")
        if int(published.get("fit_record_count", -1)) != self.record_count:
            raise BankContractError("published B0 manifest record count differs")
        geometry = published.get("geometry")
        if not isinstance(geometry, Mapping) or list(geometry.get("fit", [])) != [
            self.record_count,
            self.geometry.sequence_tokens,
            self.geometry.hidden_size,
        ]:
            raise BankContractError("published B0 geometry differs")
        payload = position_contract.get("payload")
        observed = position_contract.get("observed")
        contract_geometry = position_contract.get("geometry")
        if not isinstance(payload, Mapping) or payload.get("sha256") != self.binding["artifacts"]["payload"]["sha256"]:
            raise BankContractError("B0 position contract payload binding differs")
        if not isinstance(observed, Mapping) or observed.get("active_arange") is not True or observed.get("inactive_position_values") != [0]:
            raise BankContractError("B0 position contract does not bind zero-padded positions")
        if not isinstance(contract_geometry, Mapping) or int(contract_geometry.get("records", -1)) != self.record_count:
            raise BankContractError("B0 position contract geometry differs")

    def _assert_static(self) -> None:
        for expected in self._statics:
            current = _stat_snapshot(Path(str(expected["path"])), label="immutable B0 artifact")
            for key in ("bytes", "mtime_ns", "inode"):
                if int(current[key]) != int(expected[key]):
                    raise BankContractError(f"immutable B0 artifact changed after integrity gate: {expected['path']}")

    def _read_window(self, start: int, stop: int) -> dict[str, torch.Tensor]:
        self._assert_static()
        try:
            with safe_open(str(self._artifacts["payload"]), framework="pt", device="cpu") as handle:
                tensors = {
                    key: handle.get_slice(key)[start:stop].contiguous()
                    for key in _PAYLOAD_KEYS
                }
        except BankContractError:
            raise
        except Exception as exc:
            raise BankContractError(f"cannot read scheduled B0 window {start}:{stop}") from exc
        _validate_window(tensors, geometry=self.geometry)
        return tensors

    def get_records(self, global_indices: Sequence[int]) -> StreamBatch:
        requested = [int(index) for index in global_indices]
        if not requested:
            raise BankContractError("B0 get_records requires at least one row")
        if any(index < 0 or index >= self.global_row_stop for index in requested):
            raise BankContractError("B0 schedule row is outside the prefix")

        # Coalesce sorted unique rows into small contiguous windows.  Repeated
        # and out-of-order requests are scattered back into caller order.
        unique = sorted(set(requested))
        runs: list[tuple[int, int]] = []
        run_start = run_previous = unique[0]
        for index in unique[1:]:
            if index != run_previous + 1:
                runs.append((run_start, run_previous + 1))
                run_start = index
            run_previous = index
        runs.append((run_start, run_previous + 1))
        windows = {start: self._read_window(start, stop) for start, stop in runs}

        def row_tensor(key: str, index: int) -> torch.Tensor:
            for start, stop in runs:
                if start <= index < stop:
                    return windows[start][key][index - start]
            raise BankContractError(f"B0 row {index} was not read")

        tensors = {
            key: torch.stack([row_tensor(key, index) for index in requested], dim=0)
            for key in _PAYLOAD_KEYS
        }
        mask = tensors["attention_mask"].to(dtype=torch.bool)
        positions = tensors["position_ids"].to(dtype=torch.long)
        rows = [self.records[index] for index in requested]
        sequence_ids = tuple(
            str(row.get("sequence_id") or row.get("sequence_h128_sha256") or row["record_id"])
            for row in rows
        )
        return StreamBatch(
            activations=tensors["activations"].to(self.device),
            token_ids=tensors["token_ids"].to(self.device),
            attention_mask=mask.to(self.device),
            position_ids=positions.to(self.device),
            global_rows=tuple(requested),
            record_ids=tuple(str(row["record_id"]) for row in rows),
            sequence_ids=sequence_ids,
        )

    def iter_batches(self) -> Iterator[StreamBatch]:
        for start in range(0, self.record_count, self.geometry.batch_records):
            yield self.get_records(range(start, start + self.geometry.batch_records))


class CombinedB0StreamedBankLoader:
    """Compose monolithic published B0 with a normal P09 B1 bank."""

    interface = "trr-p09.combined-b0-streamed-bank-loader.v1"

    def __init__(
        self,
        b0_binding_path: Path,
        b1_manifest_path: Path,
        *,
        device: str = "cpu",
    ) -> None:
        self.prefix = B0ImmutableLoader(b0_binding_path, device=device)
        self.addition = StreamedBankLoader(b1_manifest_path, device=device)
        if self.prefix.expanded_row_origin != 0:
            raise BankContractError("B0 prefix must begin at expanded row zero")
        if self.prefix.global_row_stop != self.addition.expanded_row_origin:
            raise BankContractError("B0/B1 banks have a gap or overlap")
        if self.prefix.geometry != self.addition.geometry:
            raise BankContractError("B0/B1 bank geometry differs")
        self.geometry = self.prefix.geometry
        self.device = self.prefix.device
        self.expanded_row_origin = 0
        self.global_row_stop = self.addition.global_row_stop

    def get_records(self, global_indices: Sequence[int]) -> StreamBatch:
        requested = [int(index) for index in global_indices]
        if not requested:
            raise BankContractError("combined get_records requires at least one row")
        if any(index < 0 or index >= self.global_row_stop for index in requested):
            raise BankContractError("combined schedule row is outside the bank")
        prefix_positions = [(out, index) for out, index in enumerate(requested) if index < self.prefix.global_row_stop]
        addition_positions = [(out, index) for out, index in enumerate(requested) if index >= self.addition.expanded_row_origin]
        tensors: dict[str, list[torch.Tensor | None]] = {key: [None] * len(requested) for key in _PAYLOAD_KEYS}
        record_ids: list[str | None] = [None] * len(requested)
        sequence_ids: list[str | None] = [None] * len(requested)

        def scatter(child: Any, positions: list[tuple[int, int]]) -> None:
            if not positions:
                return
            batch = child.get_records([index for _, index in positions])
            for child_index, (output_index, _) in enumerate(positions):
                for key, value in (
                    ("activations", batch.activations),
                    ("attention_mask", batch.attention_mask),
                    ("position_ids", batch.position_ids),
                    ("token_ids", batch.token_ids),
                ):
                    tensors[key][output_index] = value[child_index]
                record_ids[output_index] = batch.record_ids[child_index]
                sequence_ids[output_index] = batch.sequence_ids[child_index]

        scatter(self.prefix, prefix_positions)
        scatter(self.addition, addition_positions)
        if any(value is None for values in tensors.values() for value in values) or any(
            value is None for value in record_ids + sequence_ids
        ):
            raise BankContractError("combined row metadata is incomplete")
        return StreamBatch(
            activations=torch.stack([value for value in tensors["activations"] if value is not None], dim=0),
            token_ids=torch.stack([value for value in tensors["token_ids"] if value is not None], dim=0),
            attention_mask=torch.stack([value for value in tensors["attention_mask"] if value is not None], dim=0),
            position_ids=torch.stack([value for value in tensors["position_ids"] if value is not None], dim=0),
            global_rows=tuple(requested),
            record_ids=tuple(value for value in record_ids if value is not None),
            sequence_ids=tuple(value for value in sequence_ids if value is not None),
        )

    def iter_batches(self) -> Iterator[StreamBatch]:
        yield from self.prefix.iter_batches()
        yield from self.addition.iter_batches()

