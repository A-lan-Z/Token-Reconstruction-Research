"""Lazy adapter for the signed P09 public validation preparation manifest.

The preparation manifest binds public H observations separately from the public
128-token labels/masks/positions.  This module verifies those bindings once and
serves only requested validation rows; it never opens source text or final
truth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from scripts.trr_p09.fixed_control_caller import DEFAULT_VALIDATION_DOMAINS, join_public_validation_labels
from scripts.trr_p09.prepare_streamed_bank import StreamBatch


SCHEMA = "token-reconstruction.trr-p09-public-validation-preparation.v1"
TASK_ID = "TRR-P09"
SEQUENCE_TOKENS = 128
HIDDEN_SIZE = 2048


class PublicValidationLoaderError(RuntimeError):
    """Raised when the public validation preparation contract is not met."""


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicValidationLoaderError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PublicValidationLoaderError(f"{label} is not a JSON object")
    return value


def _file(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PublicValidationLoaderError(f"{label} is not a regular file: {path}")
    return path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, label: str) -> dict[str, Any]:
    path = _file(path, label)
    return {"label": label, "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha(path)}


class PublicValidationLoader:
    """Hash-bound random-access public validation loader.

    Rows are in the frozen Finance[0:256], Pile[256:384] namespace.  H files
    are already the trusted H128 view produced by the preparation job; the
    adapter requires that shape and checks their mask/position sidecars against
    the separately materialized public label payloads.
    """

    sequence_tokens = SEQUENCE_TOKENS
    hidden_size = HIDDEN_SIZE

    def __init__(self, manifest_path: Path, rows_path: Path) -> None:
        self.manifest_path = _file(manifest_path, "public validation manifest")
        self.rows_path = _file(rows_path, "public validation rows")
        manifest = _json(self.manifest_path, "public validation manifest")
        if manifest.get("schema") != SCHEMA or manifest.get("task_id") != TASK_ID:
            raise PublicValidationLoaderError("public validation manifest schema/task differs")
        if manifest.get("status") != "PUBLIC_VALIDATION_PREPARED_NO_TRUTH":
            raise PublicValidationLoaderError("public validation manifest is not no-truth prepared")
        if manifest.get("truth_opened") is not False or manifest.get("source_text_persisted") is not False:
            raise PublicValidationLoaderError("public validation manifest crosses truth/source boundary")
        if manifest.get("domains") != list(DEFAULT_VALIDATION_DOMAINS):
            raise PublicValidationLoaderError("public validation domain order differs")
        if manifest.get("global_row_order") != "Finance_then_Pile":
            raise PublicValidationLoaderError("public validation global order differs")
        if int(manifest.get("record_count", -1)) != 384 or int(manifest.get("sequence_tokens_including_bos", -1)) != SEQUENCE_TOKENS or int(manifest.get("hidden_size", -1)) != HIDDEN_SIZE:
            raise PublicValidationLoaderError("public validation geometry/count differs")
        bindings = manifest.get("input_bindings")
        if not isinstance(bindings, Mapping) or bindings.get("public_base_only") is not True:
            raise PublicValidationLoaderError("public validation observations are not public-base only")
        if bindings.get("truth_opened") is True or bindings.get("source_text_written") is True:
            raise PublicValidationLoaderError("public validation bindings cross truth/source boundary")

        row_value = _json(self.rows_path, "public validation rows")
        rows = row_value.get("rows", row_value.get("records"))
        if not isinstance(rows, list) or len(rows) != 384 or any(not isinstance(row, Mapping) for row in rows):
            raise PublicValidationLoaderError("public validation rows are malformed")
        rows = [dict(row) for row in rows]
        self.label_join = join_public_validation_labels(rows, required_domains=DEFAULT_VALIDATION_DOMAINS)
        declared_join = manifest.get("label_join")
        if not isinstance(declared_join, Mapping) or declared_join.get("semantic_sha256") != self.label_join.semantic_sha256:
            raise PublicValidationLoaderError("public validation label join digest differs")
        expected_partition = {domain: list(self.label_join.rows_by_domain[domain]) for domain in DEFAULT_VALIDATION_DOMAINS}
        if declared_join.get("rows_by_domain") != expected_partition:
            raise PublicValidationLoaderError("public validation label partition differs")
        if tuple(int(row.get("global_row", -1)) for row in rows) != tuple(range(384)):
            raise PublicValidationLoaderError("public validation rows are not contiguous")
        self.manifest = manifest
        self.rows = tuple(rows)
        self.record_ids = tuple(row.record_id for row in self.label_join.rows)
        self._offsets = {"Finance": 0, "Pile": 256}
        self._counts = {"Finance": 256, "Pile": 128}
        self._h_paths: dict[str, Path] = {}
        self._label_paths: dict[str, Path] = {}
        self._labels: dict[str, dict[str, torch.Tensor]] = {}
        self._validate_payloads(manifest)

    def _validate_payloads(self, manifest: Mapping[str, Any]) -> None:
        observations = manifest.get("observation_h_join")
        payloads = manifest.get("payloads")
        if not isinstance(observations, list) or not isinstance(payloads, Mapping):
            raise PublicValidationLoaderError("public validation payload bindings are incomplete")
        for domain in DEFAULT_VALIDATION_DOMAINS:
            offset = self._offsets[domain]
            count = self._counts[domain]
            observed = [dict(row) for row in observations if isinstance(row, Mapping) and row.get("domain") == domain]
            observed.sort(key=lambda row: int(row.get("global_row", -1)))
            if len(observed) != count:
                raise PublicValidationLoaderError(f"public validation H count differs for {domain}")
            for local, row in enumerate(observed):
                if int(row.get("global_row", -1)) != offset + local or int(row.get("observation_row", -1)) != local:
                    raise PublicValidationLoaderError(f"public validation H row order differs for {domain}")
                if str(row.get("record_id", "")) != self.record_ids[offset + local]:
                    raise PublicValidationLoaderError(f"public validation H record identity differs for {domain}")
            first = observed[0]
            h_path = _file(Path(str(first.get("h_path", ""))), f"public validation {domain} H payload")
            if int(first.get("h_bytes", -1)) != h_path.stat().st_size or str(first.get("h_sha256", "")) != _sha(h_path):
                raise PublicValidationLoaderError(f"public validation {domain} H payload hash differs")
            with safe_open(str(h_path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                    raise PublicValidationLoaderError(f"public validation {domain} H keys differ")
                shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
                if shapes["activations"] != (count, SEQUENCE_TOKENS, HIDDEN_SIZE) or shapes["attention_mask"] != (count, SEQUENCE_TOKENS) or shapes["position_ids"] != (count, SEQUENCE_TOKENS):
                    raise PublicValidationLoaderError(f"public validation {domain} H geometry differs")
                metadata = dict(handle.metadata() or {})
                if metadata.get("truth_opened") == "true" or metadata.get("source_text_written") == "true":
                    raise PublicValidationLoaderError(f"public validation {domain} H metadata crosses truth/source boundary")
                h_mask = handle.get_tensor("attention_mask").to(dtype=torch.bool)
                h_positions = handle.get_tensor("position_ids").to(dtype=torch.long)
            payload = payloads.get(domain)
            if not isinstance(payload, Mapping) or not isinstance(payload.get("file"), Mapping):
                raise PublicValidationLoaderError(f"public validation {domain} label payload is absent")
            descriptor = payload["file"]
            label_path = _file(Path(str(descriptor.get("path", ""))), f"public validation {domain} labels")
            if int(descriptor.get("bytes", -1)) != label_path.stat().st_size or str(descriptor.get("sha256", "")) != _sha(label_path):
                raise PublicValidationLoaderError(f"public validation {domain} label payload hash differs")
            with safe_open(str(label_path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {"token_ids", "attention_mask", "position_ids"}:
                    raise PublicValidationLoaderError(f"public validation {domain} label keys differ")
                labels = {key: handle.get_tensor(key).contiguous() for key in handle.keys()}
                if any(tuple(value.shape) != (count, SEQUENCE_TOKENS) for value in labels.values()):
                    raise PublicValidationLoaderError(f"public validation {domain} label geometry differs")
                metadata = dict(handle.metadata() or {})
                if metadata.get("truth_opened") == "true" or metadata.get("source_text_written") == "true":
                    raise PublicValidationLoaderError(f"public validation {domain} label metadata crosses truth/source boundary")
            if not torch.equal(labels["attention_mask"].to(dtype=torch.bool), h_mask) or not torch.equal(labels["position_ids"].to(dtype=torch.long), h_positions):
                raise PublicValidationLoaderError(f"public validation {domain} labels disagree with H sidecars")
            self._h_paths[domain] = h_path
            self._label_paths[domain] = label_path
            self._labels[domain] = labels

    def get_records(self, global_indices: Sequence[int]) -> StreamBatch:
        requested = tuple(int(row) for row in global_indices)
        if not requested or any(row < 0 or row >= 384 for row in requested):
            raise PublicValidationLoaderError("public validation row is outside [0,384)")
        activation_rows: list[torch.Tensor | None] = [None] * len(requested)
        token_rows: list[torch.Tensor | None] = [None] * len(requested)
        mask_rows: list[torch.Tensor | None] = [None] * len(requested)
        position_rows: list[torch.Tensor | None] = [None] * len(requested)
        grouped: dict[str, list[tuple[int, int]]] = {domain: [] for domain in DEFAULT_VALIDATION_DOMAINS}
        for output_index, global_row in enumerate(requested):
            domain = "Finance" if global_row < 256 else "Pile"
            grouped[domain].append((output_index, global_row - self._offsets[domain]))
        for domain, entries in grouped.items():
            if not entries:
                continue
            local_indices = [local for _, local in entries]
            with safe_open(str(self._h_paths[domain]), framework="pt", device="cpu") as handle:
                activations = handle.get_slice("activations")[local_indices].contiguous()
            index_tensor = torch.tensor(local_indices, dtype=torch.long)
            labels = self._labels[domain]
            tokens = labels["token_ids"].index_select(0, index_tensor)
            masks = labels["attention_mask"].index_select(0, index_tensor).to(dtype=torch.bool)
            positions = labels["position_ids"].index_select(0, index_tensor).to(dtype=torch.long)
            for inner, (output_index, _) in enumerate(entries):
                activation_rows[output_index] = activations[inner]
                token_rows[output_index] = tokens[inner]
                mask_rows[output_index] = masks[inner]
                position_rows[output_index] = positions[inner]
        if any(value is None for value in activation_rows + token_rows + mask_rows + position_rows):
            raise PublicValidationLoaderError("public validation row materialization failed")
        return StreamBatch(
            activations=torch.stack([value for value in activation_rows if value is not None]),
            token_ids=torch.stack([value for value in token_rows if value is not None]),
            attention_mask=torch.stack([value for value in mask_rows if value is not None]),
            position_ids=torch.stack([value for value in position_rows if value is not None]),
            global_rows=requested,
            record_ids=tuple(self.record_ids[row] for row in requested),
            sequence_ids=tuple(self.record_ids[row] for row in requested),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "manifest": _record(self.manifest_path, "public validation manifest"),
            "rows": _record(self.rows_path, "public validation rows"),
            "label_join_sha256": self.label_join.semantic_sha256,
            "record_count": 384,
            "domains": list(DEFAULT_VALIDATION_DOMAINS),
            "sequence_tokens": SEQUENCE_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "h_payloads": {domain: _record(path, f"public validation {domain} H payload") for domain, path in self._h_paths.items()},
            "label_payloads": {domain: _record(path, f"public validation {domain} labels") for domain, path in self._label_paths.items()},
        }
