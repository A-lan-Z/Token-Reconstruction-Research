"""Factored TRR-P11 public observation capture adapter.

This module owns the post-selection public capture boundary only.  It consumes
an already released identity-only selection, delegates source materialization
and prefix capture to the qualified TRR5/TRR6 producer helpers, and writes
sanitized BF16 observations plus hash-bound receipts.  Heavy dependencies are
imported only inside the explicit ``execute=True`` path, so metadata preflight
and tests cannot load a model or GPU accidentally.

The optional A1+A2 K256 comparator is represented by an explicit availability
preflight.  A missing or unavailable package is a technical blocker recorded by
its caller; it is never silently dropped or replaced using new outcomes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from scripts.trr_p11 import source_selector as selector

TASK_ID = selector.TASK_ID
OBSERVATION_SCHEMA = selector.OBSERVATION_SCHEMA
CAPTURE_SCHEMA = selector.CAPTURE_SCHEMA
CAPTURE_STATUS = selector.CAPTURE_STATUS
DOMAIN_ORDER = selector.DOMAIN_ORDER
TARGET_ORDER = selector.TARGET_ORDER
CELL_ORDER = selector.CELL_ORDER
OPTIONAL_COMPARATOR = selector.OPTIONAL_COMPARATOR
RECORDS_PER_DOMAIN = selector.RECORDS_PER_DOMAIN
A1_A2_RECORDS_PER_DOMAIN = selector.A1_A2_RECORDS_PER_DOMAIN
STORED_SEQUENCE_TOKENS = selector.STORED_SEQUENCE_TOKENS
SCORED_POST_BOS_TOKENS = selector.SCORED_POST_BOS_TOKENS
CAPTURE_BATCH_RECORDS = selector.CAPTURE_BATCH_RECORDS
CAPTURE_SEQUENCE_TOKENS = selector.CAPTURE_SEQUENCE_TOKENS
HIDDEN_SIZE = selector.HIDDEN_SIZE

# This is the single canonical evaluator-only binding.  At execution time the
# value is checked against the current P11 manifest before the LoRA is loaded.
LORA_UPDATE_SHA256 = "eea7bb49f801b61df2e26a8f59af7c3096f6f3a2604404e16e589443bcfba595"

_SHA256_HEX = frozenset("0123456789abcdef")


class CaptureAdapterError(selector.CaptureError):
    """Raised when a P11 capture or comparator preflight fails closed."""


class ComparatorPreflightError(CaptureAdapterError):
    """Raised for a malformed comparator descriptor or hash mismatch."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CaptureAdapterError(f"asset is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CaptureAdapterError("value cannot be represented as canonical JSON") from exc
    return _sha256_bytes(encoded)


def _record(path: Path, *, root: Path, description: str, allow_external: bool = False) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise CaptureAdapterError(f"{description} is unavailable: {resolved}")
    if not allow_external:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise CaptureAdapterError(f"{description} is outside the repository root: {resolved}") from exc
    return {"path": str(resolved), "bytes": int(resolved.stat().st_size), "sha256": _sha256_file(resolved)}


def _write_create_only(path: Path, payload: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise CaptureAdapterError(f"{description} is create-only: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CaptureAdapterError(f"{description} is outside the repository root: {resolved}") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:  # pragma: no cover
        raise CaptureAdapterError(f"{description} is create-only: {resolved}") from exc
    return _record(resolved, root=root, description=description)


def _validate_p11_source_descriptors(inputs: Mapping[str, Any], *, root: Path) -> None:
    """Compare only the fields frozen by P11, without legacy extra fields."""
    for domain in DOMAIN_ORDER:
        frozen = inputs.get(domain)
        if not isinstance(frozen, Mapping):
            raise CaptureAdapterError(f"P11 source descriptor is absent: {domain}")
        expected_meta = {"dataset_key": domain, **selector._DATASET_META[domain]}
        for key, expected in expected_meta.items():
            if frozen.get(key) != expected:
                raise CaptureAdapterError(f"P11 {domain} source {key} changed")
        files = frozen.get("arrow_files")
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes, bytearray)) or not files:
            raise CaptureAdapterError(f"P11 {domain} Arrow files are absent")
        actual_files = []
        for index, item in enumerate(files):
            if not isinstance(item, Mapping):
                raise CaptureAdapterError(f"P11 {domain} Arrow file binding is malformed: {index}")
            actual = _record(Path(str(item.get("path", ""))), root=root, description=f"P11 {domain} Arrow file {index}", allow_external=True)
            if any(item.get(key) != actual[key] for key in ("path", "bytes", "sha256")):
                raise CaptureAdapterError(f"P11 {domain} Arrow file binding changed: {index}")
            actual_files.append(actual)
        if list(files) != actual_files:
            raise CaptureAdapterError(f"P11 {domain} Arrow descriptor changed")
    tokenizer = inputs.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise CaptureAdapterError("P11 tokenizer descriptor is absent")
    tokenizer_path = Path(str(tokenizer.get("path", ""))).expanduser().resolve()
    if tokenizer_path.is_symlink() or not tokenizer_path.is_dir():
        raise CaptureAdapterError(f"P11 tokenizer snapshot is unavailable: {tokenizer_path}")
    files = tokenizer.get("files")
    if not isinstance(files, Mapping) or not files:
        raise CaptureAdapterError("P11 tokenizer file bindings are absent")
    for name, item in files.items():
        if not isinstance(item, Mapping):
            raise CaptureAdapterError(f"P11 tokenizer file binding is malformed: {name}")
        actual = _record(tokenizer_path / str(name), root=root, description=f"P11 tokenizer file {name}", allow_external=True)
        if any(item.get(key) != actual[key] for key in ("path", "bytes", "sha256")):
            raise CaptureAdapterError(f"P11 tokenizer file binding changed: {name}")


def _capture_output_root(path: Path, *, root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.resolve()
    allowed = (root / "experiments" / TASK_ID / "evaluation").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise CaptureAdapterError(f"capture output must be below {allowed}") from exc
    if resolved.exists() or resolved.is_symlink():
        raise CaptureAdapterError(f"capture output root is create-only: {resolved}")
    return resolved


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def validate_full_capture_geometry(cell_id: str, shape: Sequence[int]) -> tuple[int, int, int]:
    """Validate the transient producer output shape before compaction."""
    if cell_id not in CELL_ORDER:
        raise CaptureAdapterError(f"unknown P11 cell: {cell_id}")
    actual = tuple(int(value) for value in shape)
    expected = (RECORDS_PER_DOMAIN, CAPTURE_SEQUENCE_TOKENS, HIDDEN_SIZE)
    if actual != expected:
        raise CaptureAdapterError(f"{cell_id} full capture geometry changed: {actual} != {expected}")
    return expected


def validate_observation_geometry(
    cell_id: str,
    *,
    activation_shape: Sequence[int],
    activation_dtype: str,
    attention_mask_shape: Sequence[int],
    attention_mask_dtype: str,
    position_ids_shape: Sequence[int],
    position_ids_dtype: str,
    attention_mask_all_one: bool = True,
    position_ids_contiguous: bool = True,
    finite: bool = True,
    records: int = RECORDS_PER_DOMAIN,
) -> dict[str, Any]:
    """Validate sanitized tensor metadata without importing torch."""
    if cell_id not in CELL_ORDER:
        raise CaptureAdapterError(f"unknown P11 cell: {cell_id}")
    if isinstance(records, bool) or int(records) != RECORDS_PER_DOMAIN:
        raise CaptureAdapterError(f"{cell_id} record count changed")
    expected_activation = (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE)
    expected_sidecar = (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS)
    if tuple(int(value) for value in activation_shape) != expected_activation:
        raise CaptureAdapterError(f"{cell_id} compact activation geometry changed")
    if str(activation_dtype).lower().replace("torch.", "") not in {"bfloat16", "bf16"}:
        raise CaptureAdapterError(f"{cell_id} compact activation dtype changed")
    if tuple(int(value) for value in attention_mask_shape) != expected_sidecar or tuple(int(value) for value in position_ids_shape) != expected_sidecar:
        raise CaptureAdapterError(f"{cell_id} sidecar geometry changed")
    if str(attention_mask_dtype).lower().replace("torch.", "") not in {"uint8", "bool"}:
        raise CaptureAdapterError(f"{cell_id} attention-mask dtype changed")
    if str(position_ids_dtype).lower().replace("torch.", "") not in {"int64", "long"}:
        raise CaptureAdapterError(f"{cell_id} position-id dtype changed")
    if not attention_mask_all_one:
        raise CaptureAdapterError(f"{cell_id} attention mask is not all ones for the stored clip")
    if not position_ids_contiguous:
        raise CaptureAdapterError(f"{cell_id} position IDs are not contiguous")
    if not finite:
        raise CaptureAdapterError(f"{cell_id} activation contains non-finite values")
    return {
        "shape": list(expected_activation),
        "attention_mask_shape": list(expected_sidecar),
        "position_ids_shape": list(expected_sidecar),
        "activation_dtype": "bfloat16",
        "attention_mask_dtype": str(attention_mask_dtype).lower().replace("torch.", ""),
        "position_ids_dtype": "int64",
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
    }


def _validate_declared_comparator_subset(subset: Mapping[str, Any]) -> dict[str, Any]:
    if subset.get("records_per_domain") != A1_A2_RECORDS_PER_DOMAIN:
        raise ComparatorPreflightError("A1+A2 comparator must use the preregistered first 128 records per domain")
    if subset.get("first_records_per_domain") is not True or subset.get("performance_based_drop") is not False:
        raise ComparatorPreflightError("A1+A2 comparator subset rule changed")
    if tuple(subset.get("domains", DOMAIN_ORDER)) != DOMAIN_ORDER:
        raise ComparatorPreflightError("A1+A2 comparator domain order changed")
    expected_positions = A1_A2_RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS
    if subset.get("token_positions_per_cell", expected_positions) != expected_positions:
        raise ComparatorPreflightError("A1+A2 comparator token geometry changed")
    return {
        "records_per_domain": A1_A2_RECORDS_PER_DOMAIN,
        "first_records_per_domain": True,
        "performance_based_drop": False,
        "domains": list(DOMAIN_ORDER),
        "token_positions_per_cell": expected_positions,
    }


def inspect_comparator_availability(binding: Mapping[str, Any] | None, *, root: Path) -> dict[str, Any]:
    """Return explicit A1+A2 package availability without loading a model."""
    if binding is None:
        return {
            "method": OPTIONAL_COMPARATOR,
            "status": "BLOCKED_TECHNICAL",
            "preregistered": True,
            "blocker_id": "A1_A2_PACKAGE_UNAVAILABLE",
            "reason": "Agent1 comparator package descriptor is absent; the comparator remains registered and cannot be silently dropped.",
            "model_loaded": False,
            "truth_opened": False,
        }
    if not isinstance(binding, Mapping):
        raise ComparatorPreflightError("A1+A2 comparator descriptor is malformed")
    subset = binding.get("subset")
    if not isinstance(subset, Mapping):
        raise ComparatorPreflightError("A1+A2 comparator subset binding is absent")
    normalized_subset = _validate_declared_comparator_subset(subset)
    status = str(binding.get("status", ""))
    if status in {"PENDING_AGENT1_PACKAGE", "BLOCKED_TECHNICAL", "UNAVAILABLE"}:
        reason = binding.get("reason")
        if not isinstance(reason, str) or not reason:
            reason = "Agent1 comparator package is not yet available; no outcome-based replacement is permitted."
        return {
            "method": OPTIONAL_COMPARATOR,
            "status": "BLOCKED_TECHNICAL",
            "preregistered": binding.get("preregistered") is True,
            "blocker_id": str(binding.get("blocker_id", "A1_A2_PACKAGE_UNAVAILABLE")),
            "reason": reason,
            "subset": normalized_subset,
            "model_loaded": False,
            "truth_opened": False,
        }
    if status != "READY":
        raise ComparatorPreflightError(f"A1+A2 comparator status is not explicit: {status}")
    package = binding.get("package")
    if not isinstance(package, Mapping):
        raise ComparatorPreflightError("A1+A2 comparator package binding is absent")
    path_value = package.get("path")
    expected_sha = package.get("sha256")
    if not isinstance(path_value, str) or not path_value or not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(c not in _SHA256_HEX for c in expected_sha):
        raise ComparatorPreflightError("A1+A2 comparator package file binding is malformed")
    package_path = Path(path_value).expanduser()
    if not package_path.is_absolute():
        package_path = root / package_path
    if not package_path.is_file() or package_path.is_symlink():
        return {
            "method": OPTIONAL_COMPARATOR,
            "status": "BLOCKED_TECHNICAL",
            "preregistered": True,
            "blocker_id": "A1_A2_PACKAGE_UNAVAILABLE",
            "reason": f"Bound comparator package is unavailable: {package_path.resolve()}",
            "subset": normalized_subset,
            "model_loaded": False,
            "truth_opened": False,
        }
    actual = _record(package_path, root=root, description="A1+A2 comparator package")
    if actual["sha256"] != expected_sha:
        raise ComparatorPreflightError("A1+A2 comparator package hash changed")
    return {
        "method": OPTIONAL_COMPARATOR,
        "status": "AVAILABLE_METADATA_ONLY",
        "preregistered": True,
        "subset": normalized_subset,
        "package": actual,
        "model_loaded": False,
        "truth_opened": False,
    }


def _load_target_lora_binding(root: Path) -> str:
    manifest_path = root / "experiments" / TASK_ID / "manifest.json"
    try:
        manifest, _record_value = selector._load_json(manifest_path, root=root, description="P11 manifest")
        expected = manifest["evaluation_contract"]["target_resource_binding"]["public_lora_2601"]["sha256"]
    except (KeyError, TypeError, selector.P11PipelineError) as exc:
        raise CaptureAdapterError("P11 public_lora_2601 resource binding is unavailable") from exc
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in _SHA256_HEX for c in expected):
        raise CaptureAdapterError("P11 public_lora_2601 resource binding is malformed")
    if expected != LORA_UPDATE_SHA256:
        raise CaptureAdapterError("P11 public_lora_2601 canonical binding differs from the registered manifest")
    return expected


def _tensor_digest(value: Any) -> str:
    """Match the repository tensor digest without importing torch at module load."""
    import torch

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


def _observation_tensor_bindings(path: Path) -> dict[str, str]:
    """Read only sanitized tensors to bind their exact evaluator inputs."""
    from safetensors.torch import load_file

    try:
        tensors = load_file(str(path), device="cpu")
    except Exception as exc:
        raise CaptureAdapterError(f"P11 observation is not a readable safetensors file: {path}") from exc
    required = {"activations", "attention_mask", "position_ids"}
    if set(tensors) != required:
        raise CaptureAdapterError(f"P11 observation tensor keys changed: {path}")
    return {key: _tensor_digest(tensors[key]) for key in sorted(required)}


def _validate_hash_map(value: Any, *, description: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"activations", "attention_mask", "position_ids"}:
        raise CaptureAdapterError(f"{description} tensor bindings are incomplete")
    result: dict[str, str] = {}
    for key in ("activations", "attention_mask", "position_ids"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in _SHA256_HEX for char in digest):
            raise CaptureAdapterError(f"{description}/{key} tensor hash is malformed")
        result[key] = digest
    return result


def _save_observation(
    path: Path,
    *,
    activations: Any,
    attention_mask: Any,
    position_ids: Any,
    cell_id: str,
    selection_sha256: str,
    record_ids_sha256: str,
    root: Path,
) -> dict[str, Any]:
    """Save sanitized compact tensors; torch/safetensors load only here."""
    import torch
    from safetensors.torch import save_file

    activation = torch.as_tensor(activations).detach().cpu().contiguous()
    mask = torch.as_tensor(attention_mask).detach().cpu().contiguous()
    positions = torch.as_tensor(position_ids).detach().cpu().contiguous()
    validate_full_capture_geometry(cell_id, (int(activation.shape[0]), CAPTURE_SEQUENCE_TOKENS, HIDDEN_SIZE)) if activation.ndim == 3 and int(activation.shape[1]) == CAPTURE_SEQUENCE_TOKENS else None
    if activation.ndim != 3 or tuple(activation.shape[1:]) != (STORED_SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise CaptureAdapterError(f"{cell_id} compact activation geometry changed")
    validate_observation_geometry(
        cell_id,
        activation_shape=tuple(int(value) for value in activation.shape),
        activation_dtype=str(activation.dtype),
        attention_mask_shape=tuple(int(value) for value in mask.shape),
        attention_mask_dtype=str(mask.dtype),
        position_ids_shape=tuple(int(value) for value in positions.shape),
        position_ids_dtype=str(positions.dtype),
        attention_mask_all_one=bool(mask.to(torch.bool).all().item()),
        position_ids_contiguous=bool(torch.equal(positions.to(torch.long), torch.arange(STORED_SEQUENCE_TOKENS, dtype=torch.long).expand(RECORDS_PER_DOMAIN, -1))),
        finite=bool(torch.isfinite(activation.float()).all().item()),
    )
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise CaptureAdapterError(f"observation output is create-only: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "activations": activation,
            "attention_mask": mask.to(torch.uint8),
            "position_ids": positions.to(torch.int64),
        },
        str(destination),
        metadata={
            "schema": "token-reconstruction.trr-p11-public-observation.v1",
            "task_id": TASK_ID,
            "cell_id": cell_id,
            "shape": json.dumps([RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE]),
            "capture_batch_records": str(CAPTURE_BATCH_RECORDS),
            "capture_sequence_tokens": str(CAPTURE_SEQUENCE_TOKENS),
            "selection_plan_sha256": str(selection_sha256),
            "record_ids_sha256": str(record_ids_sha256),
            "source_text_written": "false",
            "token_ids_written": "false",
            "target_labels_loaded": "false",
            "truth_opened": "false",
        },
    )
    descriptor = _record(destination, root=root, description=f"P11 observation {cell_id}")
    descriptor.update({
        "cell_id": cell_id,
        "shape": [RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE],
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "activations_key": "activations",
        "attention_mask_key": "attention_mask",
        "position_ids_key": "position_ids",
        "public_full_forward": True,
        "producer_only_lora": cell_id.endswith("__public_lora_2601"),
    })
    return descriptor


def build_observation_manifest(
    *,
    selection_record: Mapping[str, Any],
    selection_payload: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    record_ids_sha256: Mapping[str, str],
    root: Path,
    record_order_by_domain: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Bind all four sanitized cells and rehash each file before truth."""
    cells: list[dict[str, Any]] = []
    for cell_id in CELL_ORDER:
        domain, target = cell_id.split("__", 1)
        observation = observations.get(cell_id)
        if not isinstance(observation, Mapping):
            raise CaptureAdapterError(f"missing P11 observation: {cell_id}")
        actual = _record(Path(str(observation.get("path", ""))), root=root, description=f"P11 observation {cell_id}")
        for key in ("path", "bytes", "sha256"):
            if observation.get(key) != actual[key]:
                raise CaptureAdapterError(f"P11 observation binding changed: {cell_id}/{key}")
        if observation.get("shape") != [RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE]:
            raise CaptureAdapterError(f"P11 observation geometry changed: {cell_id}")
        normalized = dict(observation)
        if record_order_by_domain is not None:
            declared_order = normalized.get("record_order")
            if declared_order is None:
                declared_order = record_order_by_domain.get(domain)
            if not isinstance(declared_order, Sequence) or isinstance(declared_order, (str, bytes, bytearray)):
                raise CaptureAdapterError(f"P11 observation record order is absent: {cell_id}")
            declared_order = [str(item) for item in declared_order]
            if len(declared_order) != RECORDS_PER_DOMAIN or any(not item for item in declared_order):
                raise CaptureAdapterError(f"P11 observation record order changed: {cell_id}")
            normalized["record_order"] = declared_order
            normalized["record_order_sha256"] = _canonical_digest(declared_order)
            tensor_sha = normalized.get("tensor_sha256")
            if tensor_sha is None:
                tensor_sha = _observation_tensor_bindings(Path(str(actual["path"])))
            normalized["tensor_sha256"] = _validate_hash_map(tensor_sha, description=f"P11 observation {cell_id}")
        cells.append({
            "cell_id": cell_id,
            "domain": domain,
            "target": target,
            "records": RECORDS_PER_DOMAIN,
            "record_ids_sha256": str(record_ids_sha256[domain]),
            "observation": normalized,
        })
    return {
        "schema": OBSERVATION_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "records_by_domain": {domain: RECORDS_PER_DOMAIN for domain in DOMAIN_ORDER},
        "cell_order": list(CELL_ORDER),
        "cells": cells,
        "selection_plan": dict(selection_record),
        "selection_plan_sha256": selection_record.get("sha256"),
        "source_ranges_half_open": {domain: list(selector.SOURCE_RANGES[domain]) for domain in DOMAIN_ORDER},
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "source_pairing": {"same_record_ids_across_targets": True, "record_ids_sha256": dict(record_ids_sha256)},
        "public_material_only": True,
        "source_text_loaded": True,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "truth_opened": False,
        "p03_holdout_accessed": False,
    }


def _materialize_selected(context: selector.SelectionContext, *, trusted: Any, datasets: Mapping[str, Any], tokenizer: Any) -> dict[str, list[Any]]:
    """Reuse the qualified renderer and verify every frozen identity field."""
    records: dict[str, list[Any]] = {}
    for domain in DOMAIN_ORDER:
        values: list[Any] = []
        for declared in context.rows[domain]:
            index = int(declared["row_index"])
            row = trusted._read_reserved_row(datasets[domain], style=domain, row_index=index)
            try:
                candidate = trusted._render_row(domain, row, index, tokenizer)
            except Exception as exc:
                raise CaptureAdapterError(f"frozen {domain} row {index} no longer renders") from exc
            actual = selector._selection_row(candidate)
            for key in ("record_id", "public_record_sha256", "dataset_key", "dataset_id", "split", "revision", "row_index", "source_index", "full_token_count", "post_bos_token_count", "valid_tokens", "final_sequence_sha256", "h128_sequence_sha256", "h129_sequence_sha256"):
                if str(actual.get(key)) != str(declared.get(key)):
                    raise CaptureAdapterError(f"frozen {domain} row {index} changed: {key}")
            if len(candidate.token_ids) < STORED_SEQUENCE_TOKENS:
                raise CaptureAdapterError(f"frozen {domain} row {index} is shorter than the stored clip")
            values.append(candidate)
        records[domain] = values
    return records


def capture_public(
    *,
    selection_path: Path,
    model_snapshot: Path,
    output_root: Path,
    repository_root: Path,
    lora_config: Path | None = None,
    lora_update: Path | None = None,
    device: str = "cuda",
    execute: bool = False,
) -> dict[str, Any]:
    """Capture four public target cells after explicit release."""
    if not execute:
        raise CaptureAdapterError("P11 public capture requires explicit execute=True")
    root = Path(repository_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CaptureAdapterError(f"repository root is unavailable: {root}")
    output = _capture_output_root(Path(output_root), root=root)
    selection = selector.load_selection(Path(selection_path), root=root)
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    failure_path = output / "failure.json"
    output.mkdir(parents=True)
    try:
        # Heavy modules and public payload loading are reachable only here.
        import gc
        import torch
        from scripts import trr0005_produce_confirmation as trusted
        from scripts import trr0006_capture_public as capture_helpers

        inputs = selector._normalize_source_inputs(selection.payload["public_sources_frozen"], root=root)
        pile_paths = tuple(Path(item["path"]) for item in inputs["pile"]["arrow_files"])
        finance_paths = tuple(Path(item["path"]) for item in inputs["finance"]["arrow_files"])
        tokenizer_path = Path(inputs["tokenizer"]["path"])
        _validate_p11_source_descriptors(inputs, root=root)
        tokenizer = trusted._load_tokenizer(tokenizer_path)
        datasets = {
            "pile": trusted._load_arrow_dataset(pile_paths),
            "finance": trusted._load_arrow_dataset(finance_paths),
        }
        records = _materialize_selected(selection, trusted=trusted, datasets=datasets, tokenizer=tokenizer)
        batches = capture_helpers._batches(records)
        record_ids_sha256 = {domain: _canonical_digest([row["record_id"] for row in selection.rows[domain]]) for domain in DOMAIN_ORDER}
        torch_device = trusted._device(device)
        model_path = Path(model_snapshot).expanduser().resolve()
        if model_path.is_symlink() or not model_path.is_dir():
            raise CaptureAdapterError(f"P11 model snapshot is unavailable: {model_path}")
        if (lora_config is None) != (lora_update is None):
            raise CaptureAdapterError("public_lora_2601 requires both LoRA config and update")
        lora_config_path = Path(lora_config).expanduser().resolve() if lora_config is not None else None
        lora_update_path = Path(lora_update).expanduser().resolve() if lora_update is not None else None
        if lora_config_path is not None and (lora_config_path.is_symlink() or not lora_config_path.is_file()):
            raise CaptureAdapterError(f"public_lora_2601 config is unavailable: {lora_config_path}")
        if lora_update_path is None or lora_update_path.is_symlink() or not lora_update_path.is_file():
            raise CaptureAdapterError("public_lora_2601 update is unavailable")
        expected_lora_sha = _load_target_lora_binding(root)
        lora_update_record = _record(lora_update_path, root=root, description="public_lora_2601 update", allow_external=True)
        if lora_update_record["sha256"] != expected_lora_sha:
            raise CaptureAdapterError("public_lora_2601 update hash differs from the registered evaluator-only binding")
        observations: dict[str, dict[str, Any]] = {}
        conditions: dict[str, Any] = {}
        for condition in TARGET_ORDER:
            current, receipt = capture_helpers._capture_condition(
                condition=condition,
                records=records,
                batches=batches,
                model_snapshot=model_path,
                lora_config_path=lora_config_path,
                lora_update=lora_update_path,
                output_root=output,
                records_per_domain=RECORDS_PER_DOMAIN,
                record_ids_sha256=record_ids_sha256,
                selection_sha256=selection.record["sha256"],
                device=torch_device,
            )
            observations.update(current)
            conditions[condition] = receipt
            del current
            gc.collect()
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
        observation_payload = build_observation_manifest(
            selection_record=selection.record,
            selection_payload=selection.payload,
            observations=observations,
            record_ids_sha256=record_ids_sha256,
            root=root,
            record_order_by_domain={
                domain: [str(row["record_id"]) for row in selection.rows[domain]]
                for domain in DOMAIN_ORDER
            },
        )
        observation_record = _write_create_only(output / "observations.json", observation_payload, root=root, description="P11 observation manifest")
        panel_payload = {
            "schema": "token-reconstruction.trr-p11-public-source-panel.v1",
            "task_id": TASK_ID,
            "status": "FROZEN_SOURCE_PANEL_NO_TRUTH",
            "records_by_domain": {domain: RECORDS_PER_DOMAIN for domain in DOMAIN_ORDER},
            "cell_order": list(CELL_ORDER),
            "record_ids_sha256": record_ids_sha256,
            "selection_plan": dict(selection.record),
            "observation_manifest": dict(observation_record),
            "same_sources_across_targets": True,
            "public_material_only": True,
            "truth_opened": False,
            "p03_holdout_accessed": False,
        }
        panel_record = _write_create_only(output / "panel.json", panel_payload, root=root, description="P11 source panel")
        capture_payload = {
            "schema": CAPTURE_SCHEMA,
            "task_id": TASK_ID,
            "status": CAPTURE_STATUS,
            "selection_plan": dict(selection.record),
            "observation_manifest": dict(observation_record),
            "panel": dict(panel_record),
            "conditions": conditions,
            "geometry": {
                "records_per_domain": RECORDS_PER_DOMAIN,
                "capture_batch_records": CAPTURE_BATCH_RECORDS,
                "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
                "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
                "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
                "hidden_size": HIDDEN_SIZE,
                "cells": len(CELL_ORDER),
            },
            "public_inputs": {
                "pile": inputs["pile"],
                "finance": inputs["finance"],
                "tokenizer": inputs["tokenizer"],
                "model_snapshot": {"path": str(model_path)},
                "lora_update": lora_update_record,
            },
            "execution": {
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started_clock,
                "command": list(sys.argv),
                "code_commit": _git_head(root),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "device": str(torch_device),
                "model_loaded_by_producer": True,
                "source_text_read": True,
                "source_text_written": False,
                "token_ids_written": False,
                "target_labels_loaded": False,
                "truth_opened": False,
                "p03_holdout_accessed": False,
            },
            "source_text_written": False,
            "token_ids_written": False,
            "target_labels_loaded": False,
            "truth_opened": False,
            "p03_holdout_accessed": False,
        }
        capture_record = _write_create_only(output / "capture.json", capture_payload, root=root, description="P11 capture receipt")
        return {
            "task_id": TASK_ID,
            "status": CAPTURE_STATUS,
            "observation_manifest": observation_record,
            "panel": panel_record,
            "capture": capture_record,
            "truth_opened": False,
        }
    except Exception as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            failure = {
                "schema": CAPTURE_SCHEMA,
                "task_id": TASK_ID,
                "status": "PUBLIC_OBSERVATIONS_CAPTURE_FAILED_NO_TRUTH",
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "selection_plan": str(Path(selection_path).expanduser().resolve()),
                "truth_opened": False,
                "source_text_written": False,
                "token_ids_written": False,
                "p03_holdout_accessed": False,
            }
            _write_create_only(failure_path, failure, root=root, description="P11 capture failure receipt")
        if isinstance(exc, CaptureAdapterError):
            raise
        raise CaptureAdapterError("P11 public observation capture failed") from exc


__all__ = [
    "CaptureAdapterError",
    "ComparatorPreflightError",
    "build_observation_manifest",
    "capture_public",
    "inspect_comparator_availability",
    "validate_full_capture_geometry",
    "validate_observation_geometry",
]
